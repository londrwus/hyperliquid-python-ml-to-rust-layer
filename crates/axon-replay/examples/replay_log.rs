//! Replay a captured session through the **production chain** and print what it
//! produced.
//!
//! This binary is the seam `axon.backtest` drives (`python/axon/backtest/`). Python
//! does not parse the log itself and does not reimplement any of the semantics: it
//! runs *this*, which runs [`CoreHandler`](axon_runtime::CoreHandler) — the same
//! handler the live runtime installs, fanning each event out to the book, the mark
//! cache and the order tracker in ADR-0013 §1's order — and
//! [`IntentSource`](axon_runtime::IntentSource), the same strategy adapter, behind
//! [`ReplaySource`], which is the same bus and the same `run_blocking_clocked` loop.
//! `docs/07-parity-and-testing.md` is emphatic that parity comes from running the same
//! code, and a second decoder in Python — or a second fan-out in this crate — would be
//! a second implementation to keep in agreement forever.
//!
//! Usage:
//!
//! ```text
//! replay_log <log.jsonl> [--trace <out.jsonl>] [--order event-time|as-captured]
//!                        [--signals <log.signals.jsonl> | --no-signals]
//! ```
//!
//! `--signals` defaults to the sibling file: `<log>` with its extension replaced by
//! `.signals.jsonl`, when one exists. A convention rather than a search, and the
//! summary names the signal log's own `source` so a run can never be silently missing
//! the half of the session that decides anything.
//!
//! Stdout is one JSON summary object. `--trace` additionally writes one JSON row per
//! event: the chain's state as of that event, which is what a golden comparison diffs.
//!
//! **What this does not do.** An order in the summary is what the strategy *asked
//! for*; it reached no venue, got no acknowledgement and no fill, and the replay
//! deliberately does not write it into the tracker. The positions here are the ones
//! the captured session's own fills produced. See ADR-0018 §7.

#![deny(unsafe_code)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use axon_replay::{ReplayOrder, ReplaySource, SignalLog};

/// The chain driver, shared verbatim with `tests/golden_chain.rs` — one driver, so the
/// binary Python runs and the tests that guard it cannot drift apart. Each consumer
/// uses a different part of its surface, hence the allow.
#[path = "chain/mod.rs"]
#[allow(dead_code)]
mod chain;

use chain::{replay_chain, resolve_grid, ChainOptions};

struct Args {
    log: String,
    trace: Option<String>,
    order: ReplayOrder,
    signals: Option<String>,
    no_signals: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut log = None;
    let mut trace = None;
    let mut signals = None;
    let mut no_signals = false;
    let mut order = ReplayOrder::EventTime;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--trace" => trace = Some(it.next().ok_or("--trace needs a path")?),
            "--signals" => signals = Some(it.next().ok_or("--signals needs a path")?),
            "--no-signals" => no_signals = true,
            "--order" => {
                order = match it.next().ok_or("--order needs a value")?.as_str() {
                    "event-time" => ReplayOrder::EventTime,
                    "as-captured" => ReplayOrder::AsCaptured,
                    other => return Err(format!("unknown --order {other:?}")),
                }
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other:?}")),
            other => log = Some(other.to_string()),
        }
    }
    Ok(Args {
        log: log.ok_or(
            "usage: replay_log <log.jsonl> [--trace <out.jsonl>] [--order …] [--signals <path>]",
        )?,
        trace,
        order,
        signals,
        no_signals,
    })
}

/// `foo.jsonl` → `foo.signals.jsonl`.
fn sibling_signals(log: &str) -> PathBuf {
    Path::new(log).with_extension("signals.jsonl")
}

fn load_signals(args: &Args) -> Result<SignalLog, String> {
    if args.no_signals {
        return Ok(SignalLog::empty());
    }
    // An explicit `--signals` that does not resolve is a *failure*, not a fallback to
    // an empty feed: a run that silently replayed no strategy would report a green
    // golden over the half of the chain the operator asked for.
    if let Some(p) = &args.signals {
        return SignalLog::open(p).map_err(|e| format!("{p}: {e}"));
    }
    let sibling = sibling_signals(&args.log);
    if sibling.exists() {
        return SignalLog::open(&sibling).map_err(|e| format!("{}: {e}", sibling.display()));
    }
    Ok(SignalLog::empty())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("replay_log: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let src = ReplaySource::open(&args.log)
        .map_err(|e| format!("{}: {e}", args.log))?
        .with_order(args.order);

    let opts = ChainOptions {
        order: args.order,
        signals: load_signals(&args)?,
        // Only when somebody asked for it. A `ChainRow` carries two `BTreeMap`s, so a
        // retained trace is far larger than the log it came from — and a soak tape is
        // exactly what ADR-0027 made replayable, so holding one would put the memory
        // bound straight back one layer up.
        keep_trace: args.trace.is_some(),
        // `None`: plan on the grid the log itself declares (ADR-0027). A log written
        // before `SCHEMA_VERSION` 2 is refused outright by `ReplaySource::open` above, so
        // the only way to reach this binary with no grid is a session that recorded
        // without one — and `resolve_grid` says so below rather than rounding on a guess.
        ..ChainOptions::default()
    };
    // Said out loud, on stderr, because the alternative is a silent difference in the one
    // place built to detect differences. A live session plans on the venue's grid; a
    // replay without one plans every price the grid would have moved — urgency-3
    // slippage, a band — unrounded, and a golden diff reports it as a strategy change.
    // Stderr rather than the summary object: `axon.backtest` parses stdout, and the
    // summary is a compared artifact that a schema bump of its own has to land with
    // (ADR-0025 §out-of-scope 2, still open).
    let grid = resolve_grid(&src, &opts);
    if grid.note.is_empty() {
        eprintln!(
            "replay_log: planning on the grid this log declared ({} instrument(s))",
            grid.table.len()
        );
    } else {
        eprintln!("replay_log: {}", grid.note);
    }
    // Said before the summary, because the summary's `late_arrivals` is the number this
    // explains. A candle's `ts_event` is the moment its bar *will* close and the venue
    // republishes the bar it is still filling, so the default event-time traversal
    // delivers every forming frame at its close — after the events it actually preceded —
    // and every one of those events counts as an inversion. The run is still
    // deterministic and still comparable against itself; what it is not is the
    // interleaving the live session saw.
    let shape = src.report();
    if shape.late_arrivals > 0 {
        eprintln!(
            "replay_log: {} of {} records arrived behind the event-time high-water mark, \
             {} of them behind a stamp the venue did not set (a forming bar, or a ticker \
             carrying no venue time). That is a stamping convention and a reconnect \
             replaying its snapshot, not a feed fault. The venue's own market data: \
             {}/{} out of order. Replay with --order as-captured for the interleaving the \
             session actually saw (ADR-0018 s4).",
            shape.late_arrivals,
            shape.events,
            shape.behind_derived_stamps,
            shape.reordered_by_the_feed(),
            shape.venue_stamped,
        );
    }
    let (summary, rows) = replay_chain(&src, &opts);

    if let Some(path) = &args.trace {
        let file = File::create(path).map_err(|e| format!("{path}: {e}"))?;
        let mut out = BufWriter::new(file);
        for row in &rows {
            // A trace write that fails must not leave a short file behind that a
            // golden comparison would happily diff against a reference.
            serde_json::to_writer(&mut out, row).map_err(|e| format!("{path}: {e}"))?;
            out.write_all(b"\n").map_err(|e| format!("{path}: {e}"))?;
        }
        out.flush().map_err(|e| format!("{path}: {e}"))?;
    }

    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|e| e.to_string())?
    );
    Ok(())
}
