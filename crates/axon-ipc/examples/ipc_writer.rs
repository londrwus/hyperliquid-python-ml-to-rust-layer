//! Create a ring and write `count` deterministic target-position signals into it.
//!
//! Used by the cross-language round-trip test (`python/tests/test_roundtrip.py`)
//! for the **Rust → Python** direction: this writes, the Python client reads and
//! checks. The `make_signal` formula below is mirrored byte-for-byte in
//! `python/axon/signals/_fixtures.py`.
//!
//! Usage: `ipc_writer <path> <count> [capacity]`
//! (capacity defaults to the next power of two ≥ count, so every push fits).

use axon_contracts::{Signal, FLAG_REDUCE_ONLY};
use axon_ipc::Producer;

/// Deterministic signal for index `i` of `n` — MUST match the Python fixture.
fn make_signal(i: u64, n: u64) -> Signal {
    let flags = if i % 2 == 0 { FLAG_REDUCE_ONLY } else { 0 };
    Signal::target_position(
        i,                                     // seq
        1_700_000_000_000_000_000 + i as i64,  // ts_event
        (i % 7) as u32,                        // symbol_id
        (i as i64 - n as i64 / 2) * 1_000_000, // target_qty (fixed-point)
        (i % 4) as u8,                         // urgency
        0,                                     // price_band
        500,                                   // ttl_ms
        1,                                     // model_version
        flags,                                 // flags
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: ipc_writer <path> <count> [capacity]");
    let count: u64 = args
        .next()
        .expect("missing <count>")
        .parse()
        .expect("count must be an integer");
    let capacity: u64 = match args.next() {
        Some(s) => s.parse().expect("capacity must be an integer"),
        None => count.next_power_of_two().max(2),
    };

    let producer = Producer::create(&path, capacity).expect("create ring");
    let mut i = 0u64;
    while i < count {
        if producer.try_push(&make_signal(i, count)) {
            i += 1;
        } else {
            std::thread::yield_now(); // ring full (only if capacity < count and a slow reader)
        }
    }
    producer.flush().expect("flush ring");
    eprintln!("ipc_writer: wrote {count} signals to {path} (capacity {capacity})");
}
