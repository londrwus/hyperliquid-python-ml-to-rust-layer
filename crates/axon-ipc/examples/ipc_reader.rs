//! Open a ring and drain `count` signals, printing one per line so a test driver
//! can compare them. Used by the cross-language round-trip test for the
//! **Python → Rust** direction: the Python client writes, this reads and prints,
//! the test compares against what Python wrote.
//!
//! Output: one line per signal, space-separated fields:
//!   seq ts_event symbol_id target_qty price_band urgency ttl_ms model_version flags schema_version kind
//!
//! Usage: `ipc_reader <path> <count>`

use axon_ipc::Consumer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: ipc_reader <path> <count>");
    let count: u64 = args
        .next()
        .expect("missing <count>")
        .parse()
        .expect("count must be an integer");

    let consumer = Consumer::open(&path).expect("open ring");
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    use std::io::Write;

    let mut got = 0u64;
    let mut idle = 0u64;
    while got < count {
        match consumer.try_pop() {
            Some(s) => {
                writeln!(
                    out,
                    "{} {} {} {} {} {} {} {} {} {} {}",
                    s.seq,
                    s.ts_event,
                    s.symbol_id,
                    s.target_qty,
                    s.price_band,
                    s.urgency,
                    s.ttl_ms,
                    s.model_version,
                    s.flags,
                    s.schema_version,
                    s.kind,
                )
                .expect("write stdout");
                got += 1;
                idle = 0;
            }
            None => {
                idle += 1;
                if idle > 100_000_000 {
                    eprintln!("ipc_reader: timed out after {got}/{count} signals");
                    std::process::exit(2);
                }
                std::thread::yield_now();
            }
        }
    }
    out.flush().expect("flush");
    eprintln!("ipc_reader: read {count} signals from {path}");
}
