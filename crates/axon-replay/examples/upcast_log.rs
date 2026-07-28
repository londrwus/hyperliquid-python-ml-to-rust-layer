//! Carry a pre-ADR-0027 event log forward to `SCHEMA_VERSION` 2.
//!
//! ```text
//! upcast_log <v1.jsonl> <v2.jsonl>
//! ```
//!
//! A separate binary and not a flag on `replay_log`, because the two do different
//! things to a caller's confidence. `replay_log` reads a log and refuses one it cannot
//! interpret; this *rewrites* a file, and rewriting is the operation that turns "I
//! cannot replay this" into "I can" without changing what the session actually did. That
//! deserves its own command, its own name in a shell history, and its own output file.
//!
//! What comes out declares **no instrument grid** ([`InstrumentSet::Undeclared`]), which
//! is the only true statement about a v1 log: it really did not carry one. So a replay of
//! the result plans unconstrained and says so on every run — the pre-ADR-0027 behaviour,
//! now explicit and opted into for one named file rather than applied to every stored log
//! by a reader being lenient.
//!
//! **The signal log beside it cannot be upcast**, and this does not try. `Signal` gained
//! `max_order_age_ms`, whose zero means "defer to the operator's ceiling" rather than
//! "absent", so an upcast signal log would replay with orders aged against a lifetime the
//! live session never set. Replay an upcast event log with `--no-signals`: it re-observes
//! the session faithfully and does not re-decide it. To get both halves, re-capture.

#![deny(unsafe_code)]

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::process::ExitCode;

use axon_replay::upcast::upcast_v1;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        eprintln!("usage: upcast_log <v1.jsonl> <v2.jsonl>");
        return ExitCode::from(2);
    };
    if input == output {
        // In place would mean truncating the source before the first record is read.
        // A live tape is not something to lose to a shell shortcut.
        eprintln!("upcast_log: refusing to write over the log being read");
        return ExitCode::from(2);
    }

    let src = match File::open(input) {
        Ok(f) => BufReader::new(f),
        Err(e) => {
            eprintln!("upcast_log: {input}: {e}");
            return ExitCode::from(2);
        }
    };
    let out = match File::create(output) {
        Ok(f) => BufWriter::new(f),
        Err(e) => {
            eprintln!("upcast_log: {output}: {e}");
            return ExitCode::from(2);
        }
    };

    match upcast_v1(src, out) {
        Ok(report) => {
            eprintln!(
                "upcast_log: {} events from {input} -> {output} (created_ns {} preserved). \
                 The result declares NO instrument grid, because the log it came from \
                 carried none: a replay of it plans unconstrained and will say so. Its \
                 signal log cannot be upcast - replay with --no-signals, or re-capture.",
                report.events, report.created_ns
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("upcast_log: {input}: {e}");
            ExitCode::from(2)
        }
    }
}
