//! Create a bar ring and publish `count` deterministic `MdBar`s.
//!
//! The `md_writer` of ADR-0028's second record: it stands in for the Rust core's bar
//! publisher in the cross-language round-trip test (`python/tests/test_md_ring.py`),
//! so the *record layout* can be proven to cross the boundary without starting a
//! session — a test that had to start one would be testing the session.
//!
//! The `make_md_bar` formula below is mirrored in `python/axon/marketdata/_fixtures.py`.
//!
//! Usage: `md_bar_writer <path> <count> [capacity]`
//! (capacity defaults to the next power of two ≥ count, so every push fits).

use axon_contracts::{MdBar, MD_BAR_FLAG_FIRST_BAR, MD_BAR_FLAG_GAP_BEFORE};
use axon_ipc::MdBarProducer;

/// One minute, in the units the record uses.
const INTERVAL_MS: u32 = 60_000;
const INTERVAL_NS: i64 = 60_000_000_000;
const EPOCH: i64 = 1_700_000_000_000_000_000;

/// Deterministic bar for index `i` — MUST match the Python fixture.
///
/// Bar 0 carries `first_bar` and every seventh carries `gap_before`, so both
/// continuity flags are on the wire. A fixture that only ever emitted clean bars
/// would let a reader that dropped `flags` entirely pass the round-trip, and the
/// flags are the only thing on this record that can say a rolling feature window
/// spans a hole the venue left.
fn make_md_bar(i: u64) -> MdBar {
    let open_time = EPOCH + (i as i64) * INTERVAL_NS;
    let flags = match i {
        0 => MD_BAR_FLAG_FIRST_BAR,
        n if n % 7 == 0 => MD_BAR_FLAG_GAP_BEFORE,
        _ => 0,
    };
    let base = 5_000_000_000_000 + (i as i64) * 250_000; // 50,000.0 + i·0.0025, fixed-point
    MdBar::new(
        i,                       // seq
        open_time + INTERVAL_NS, // ts_event: the close, i.e. venue T + 1 ms
        open_time,
        (i % 7) as u32, // symbol_id
        INTERVAL_MS,
        flags,
    )
    .with_ohlcv(
        base,                        // open
        base + 1_000_000,            // high, above every other price here
        base - 750_000,              // low, below every other price here
        base + 250_000,              // close
        50_000_000 + (i as i64) * 3, // volume
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: md_bar_writer <path> <count> [capacity]");
    let count: u64 = args
        .next()
        .expect("missing <count>")
        .parse()
        .expect("count must be an integer");
    let capacity: u64 = match args.next() {
        Some(s) => s.parse().expect("capacity must be an integer"),
        None => count.next_power_of_two().max(2),
    };

    let producer = MdBarProducer::create(&path, capacity).expect("create bar ring");
    let mut i = 0u64;
    while i < count {
        if producer.try_push(&make_md_bar(i)) {
            i += 1;
        } else {
            std::thread::yield_now(); // ring full (only if capacity < count and a slow reader)
        }
    }
    producer.flush().expect("flush ring");
    eprintln!("md_bar_writer: wrote {count} bars to {path} (capacity {capacity})");
}
