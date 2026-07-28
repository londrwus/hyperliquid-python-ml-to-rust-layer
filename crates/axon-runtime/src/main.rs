//! The `axon` binary — argument parsing and a startup banner, nothing else.
//!
//! Everything the process actually does lives in the library so it can be tested
//! without spawning a binary. With no arguments this runs the **offline** session:
//! the real core over a canned event stream, no network and no key (see
//! `axon_runtime::selftest`), which is what `./run.sh runtime` exercises.

use std::path::PathBuf;
use std::process::ExitCode;

use axon_runtime::config::{RuntimeConfig, ENV_CONFIG_PATH};

const USAGE: &str = "\
axon - the Axon execution runtime

USAGE:
    axon [--config PATH] [--capture PATH | --no-capture] [--check] [--dump-config]
    axon --flatten [--config PATH]

OPTIONS:
    --config PATH   session config (TOML). Defaults to $AXON_CONFIG, else the
                    built-in offline config.
    --capture PATH  record this session to PATH, with the signals it read beside it
                    as PATH-with-.signals.jsonl. Replay it afterwards with
                    `cargo run -p axon-replay --example replay_log -- PATH`.
                    A log only gets that name once the recording closed cleanly;
                    an interrupted one is left as PATH.partial.
    --no-capture    do not record, whatever the config file says.
    --check         validate the config and exit without running a session.
    --dump-config   print the resolved config as TOML and exit.
    --flatten       PLACES REAL ORDERS. Adopt the venue's own position in every
                    configured symbol and drive it to zero, then read the venue
                    again and report what it says. Not a session: no market-data
                    socket, no dead-man's switch, no signal ring, no capture.

                    This is the operator cleanup pass, and it exists because the
                    documented one did not work. `--flatten-only` on the Python
                    producer emits a target of zero, which the planner subtracts
                    from the *tracked* position -- so against a tracker that has
                    not learned the position it is a no-op, and on 2026-07-27 an
                    operator working around that hand-wrote a target and turned a
                    -0.01 short into a +0.01 long with one order.

                    Every order this sends is a close: reduce-only, sized from a
                    fresh venue read taken immediately before it. So it cannot
                    overshoot, and a partial fill shrinks the next attempt instead
                    of being added to it. The urgency is a ladder (IOC, then a
                    crossing GTC, then post-only) because on 2026-07-27 Hyperliquid
                    accepted post-only orders exclusively after a network upgrade
                    and refused every IOC -- a single urgency is a single point of
                    failure on the exit path.

                    It leaves working orders alone: cancel-all on Hyperliquid is
                    account-wide and would sweep a running session's quotes.
                    Exit code 1 if any symbol is not flat, or if the venue could
                    not be read -- an unknown position is never reported as flat.
    -h, --help      this text.

The signing key is read from AXON_HL_SECRET_KEY and must never appear in a config
file. Mainnet additionally requires AXON_ALLOW_MAINNET=1.";

struct Args {
    config: Option<PathBuf>,
    flatten: bool,
    /// `Some(Some(path))` records there, `Some(None)` refuses to record, `None` leaves
    /// the config alone. Three states because an operator who passes `--no-capture` is
    /// overriding a file that said otherwise, and collapsing that into a `bool` would
    /// make the flag mean "off" on every run that omitted it.
    capture: Option<Option<String>>,
    check: bool,
    dump: bool,
    help: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        config: None,
        flatten: false,
        capture: None,
        check: false,
        dump: false,
        help: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--config" => {
                args.config = Some(PathBuf::from(
                    it.next().ok_or("--config needs a path".to_string())?,
                ))
            }
            "--capture" => {
                args.capture = Some(Some(it.next().ok_or("--capture needs a path".to_string())?))
            }
            "--no-capture" => args.capture = Some(None),
            "--flatten" => args.flatten = true,
            "--check" => args.check = true,
            "--dump-config" => args.dump = true,
            "-h" | "--help" => args.help = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.help {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let mut cfg = match RuntimeConfig::resolve(args.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Applied after `resolve`, and re-validated: `--capture` can name a path that
    // collides with one of the rings, and a capture that truncates the signal ring
    // presents as the strategy going quiet, nowhere near its cause.
    match args.capture {
        Some(Some(path)) => {
            cfg.capture.enabled = true;
            cfg.capture.path = path;
        }
        Some(None) => cfg.capture.enabled = false,
        None => {}
    }
    if let Err(e) = cfg.validate() {
        eprintln!("config: {e}");
        return ExitCode::FAILURE;
    }

    if args.dump {
        match toml::to_string_pretty(&cfg) {
            Ok(t) => println!("{t}"),
            Err(e) => {
                eprintln!("config: {e}");
                return ExitCode::FAILURE;
            }
        }
        return ExitCode::SUCCESS;
    }

    banner(&cfg, args.config.as_deref());
    if args.check {
        println!("config OK");
        return ExitCode::SUCCESS;
    }

    if args.flatten {
        // A separate arm rather than a mode of `run`, because it is not a session and
        // sharing the entry point would make the next reader look for the core thread it
        // deliberately does not have.
        return match axon_runtime::session::run_flatten(&cfg, Default::default()) {
            Ok(reports) => {
                let unflat: Vec<_> = reports.iter().filter(|r| !r.flat()).collect();
                if unflat.is_empty() {
                    println!("flatten: every configured symbol is flat on the venue's own word");
                    ExitCode::SUCCESS
                } else {
                    // Loud, and a failing exit code, because this is the line a script
                    // wraps: "the pass ran" and "the account is flat" are different
                    // claims, and only the second one is worth acting on.
                    for r in unflat {
                        eprintln!("flatten: NOT FLAT - {r}");
                    }
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("flatten: {e}");
                ExitCode::FAILURE
            }
        };
    }

    match axon_runtime::run(&cfg) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("axon: {e}");
            ExitCode::FAILURE
        }
    }
}

/// What is about to happen, before it happens. The daily action cost of the safety
/// loop is included because it is the least obvious running cost in the system: the
/// venue's address budget is cumulative, so on a fresh account the dead-man's switch
/// can be its largest consumer.
fn banner(cfg: &RuntimeConfig, path: Option<&std::path::Path>) {
    let source = path
        .map(|p| p.display().to_string())
        .or_else(|| std::env::var(ENV_CONFIG_PATH).ok())
        .unwrap_or_else(|| "built-in default (offline)".to_string());

    println!("── axon runtime ─────────────────────────────────────────────────");
    println!("config     : {source}");
    println!(
        "environment: {:?} on {:?}  venue {}",
        cfg.environment, cfg.venue.network, cfg.venue.name
    );
    println!(
        "strategy   : {} v{}  symbols={:?} -> coins={:?}",
        cfg.strategy.name,
        cfg.strategy.version,
        cfg.strategy.symbols,
        cfg.coins()
    );
    println!("risk       : {:?}", cfg.strategy.risk.to_limits());
    println!(
        "safety     : dead-man's switch {} lead={}ms re-arm={}ms ({} actions/day)",
        if cfg.safety.dead_mans_switch {
            "ON"
        } else {
            "OFF"
        },
        cfg.safety.lead_ms,
        cfg.safety.rearm_interval_ms,
        cfg.dms_actions_per_day(),
    );
    println!(
        "reconcile  : every {}ms  marks expire after {}ms",
        cfg.reconcile.interval_ms, cfg.session.mark_max_age_ms
    );
    println!(
        "contract   : SIGNAL_SIZE={}  SCHEMA_VERSION={}  RING_HEADER_SIZE={}",
        axon_contracts::SIGNAL_SIZE,
        axon_contracts::SCHEMA_VERSION,
        axon_contracts::RING_HEADER_SIZE,
    );
    println!(
        "signal ring: {} (cap {})",
        cfg.ipc.signal_ring_path, cfg.ipc.capacity
    );
    // The other direction, and the one place an operator learns *before* a session
    // starts whether it is publishing. Without it, "Python computed no features" and
    // "the strategy had no opinion" are the same silence, discovered an hour in. Both
    // paths are named because one config switch creates *three* files (ADR-0028 for the
    // bars, ADR-0034 for the beacon): a bar consumer pointed at the wrong sibling would
    // wait forever on a healthy session, and a parity monitor pointed at the wrong
    // beacon would resolve every silence into "the publisher is dead" on a session that
    // is beating perfectly well one path over.
    println!(
        "md ring    : {}",
        if cfg.md_ring.enabled {
            format!(
                "{} (cap {}, policy {:?}); bars {}; beacon {}",
                cfg.md_ring.path,
                cfg.md_ring.capacity,
                cfg.md_ring.policy,
                axon_runtime::mdring::bar_ring_path(&cfg.md_ring.path).display(),
                axon_ipc::beacon_path(&cfg.md_ring.path).display(),
            )
        } else {
            // Naming the beacon in the OFF case too, because its absence is the more
            // confusing state: a monitor with no beacon cannot tell a quiet publisher
            // from a dead one, which is the whole reason ADR-0034 exists.
            "OFF - Python computes no features from this session, and nothing beats".to_string()
        }
    );
    // Printed because "the strategy is not trading" and "the runtime is not listening"
    // are the same symptom, and this is the line that tells them apart before a session
    // starts rather than an hour into one.
    // Printed for the same reason the intent line is: a soak that produced no artifact
    // is discovered hours later, and this is where "am I recording?" is answerable
    // before the session starts rather than after it ends.
    println!(
        "capture    : {}",
        if cfg.capture.enabled {
            format!(
                "{} (+ signals {}, cap {} MiB, queue {})",
                cfg.capture.path,
                if cfg.capture.signals { "ON" } else { "OFF" },
                cfg.capture.max_bytes / (1024 * 1024),
                cfg.capture.queue_capacity,
            )
        } else {
            "OFF - this session will leave nothing to replay".to_string()
        }
    );
    println!(
        "intents    : {}",
        if cfg.intent.enabled {
            format!(
                "ON  drain every {}ms, <= {} per pass, signals expire after {}ms",
                cfg.intent.drain_interval_ms,
                cfg.intent.max_per_drain,
                cfg.intent.max_signal_age_ms
            )
        } else {
            "OFF - this session reads the venue and never places an order".to_string()
        }
    );
    // Who is allowed to speak, and what bounds the sum of what they say (ADR-0038).
    // Printed *before* the session starts for the same reason the market-data ring is:
    // "one strategy is silent" and "one strategy was never wired" are the same status
    // line an hour in, and only one of them is a fault in the strategy.
    if cfg.intent.enabled {
        let producers = cfg.producers();
        println!("strategies : {}", producers.len());
        for p in &producers {
            let scope = if p.symbols.is_empty() {
                "every configured instrument".to_string()
            } else {
                p.symbols.join(", ")
            };
            let silence = match (p.silence_ms, p.on_silence) {
                (0, _) => "never called silent".to_string(),
                (ms, axon_runtime::config::OnSilenceMode::Hold) => {
                    format!("held after {ms}ms of silence")
                }
                (ms, axon_runtime::config::OnSilenceMode::Flat) => {
                    format!("FLATTENED after {ms}ms of silence")
                }
            };
            let alloc = if p.max_gross_notional.is_zero() {
                "no allocation".to_string()
            } else {
                format!("<= {} gross", p.max_gross_notional)
            };
            println!(
                "  {:>10} : {} -> {scope}; {alloc}; {silence}",
                p.name, p.ring_path
            );
        }
        println!(
            "portfolio  : {}",
            if cfg.portfolio.is_declared() {
                format!(
                    "gross <= {}, net <= {}, <= {} instruments; overlap {:?}, scale-to-fit {}",
                    cfg.portfolio.max_gross_notional,
                    cfg.portfolio.max_net_notional,
                    cfg.portfolio.max_symbols,
                    cfg.portfolio.overlap,
                    cfg.portfolio.scale_to_fit,
                )
            } else {
                "no bound declared - only the per-instrument limits in [strategy.risk] apply"
                    .to_string()
            }
        );
    }
    println!("─────────────────────────────────────────────────────────────────");
}
