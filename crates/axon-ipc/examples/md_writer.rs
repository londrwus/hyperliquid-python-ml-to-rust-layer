//! Create a market-data ring and publish `count` deterministic `MdSlice`s, beating a
//! beacon beside it the way the core's pass loop does.
//!
//! Stands in for the Rust core's market-data publisher (`docs/01` step 2) in the
//! cross-language round-trip test (`python/tests/test_md_ring.py`): this writes,
//! `axon.marketdata` reads a *batch* per call and checks every byte. The
//! `make_md_slice` formula below is mirrored in
//! `python/axon/marketdata/_fixtures.py`.
//!
//! Usage: `md_writer <path> <count> [capacity] [beats]`
//! (capacity defaults to the next power of two ≥ count, so every push fits. `beats` is
//! the **total** pass-loop beats the run records, including the shutdown beat; it
//! defaults to `count + 1` and is clamped up to that floor.)
//!
//! **`beats > count` is the whole point of the fourth argument.** The beacon
//! (ADR-0030) exists to distinguish a quiet publisher from a dead one, and the only
//! thing that can demonstrate that is one process whose ring stops while its beat count
//! does not. So this writes the beacon here rather than in an example of its own: a
//! beacon with no ring beside it is a configuration that never occurs and cannot show
//! the property it was built for. The trailing passes advance `coalesced` and the event
//! clock and leave `published` alone — ADR-0030's "quiet market" case, in which the feed
//! is busy and the top of book is not.

use axon_contracts::{MdSlice, MD_KIND_QUOTE, MD_KIND_TRADE};
use axon_ipc::{beacon_path, MdBeacon, MdBeat, MdProducer};

/// Deterministic slice for index `i` — MUST match the Python fixture.
///
/// Alternating quote/trade updates keep both `kind` paths, and the aggressor flag,
/// on the wire: a fixture that only ever emitted quotes would let a broken `flags`
/// or `last_trade_*` mapping pass the round-trip. Every third record carries **no
/// ticker**, for the same reason: a fixture where the mark tail was always populated
/// could not tell a reader that back-fills the sentinel from one that respects it.
fn make_md_slice(i: u64) -> MdSlice {
    let is_trade = i % 3 == 0;
    let kind = if is_trade {
        MD_KIND_TRADE
    } else {
        MD_KIND_QUOTE
    };
    let base = 5_000_000_000_000 + (i as i64) * 250_000; // 50,000.0 + i·0.0025, fixed-point
    let s = MdSlice::new(
        i,                                    // seq
        1_700_000_000_000_000_000 + i as i64, // ts_event
        (i % 7) as u32,                       // symbol_id
        kind,
    )
    .with_bbo(
        base,                         // bid_px
        100_000_000 + i as i64,       // bid_sz
        base + 500_000,               // ask_px
        250_000_000 - (i as i64) * 2, // ask_sz
    )
    .with_last_trade(
        base + 250_000,                            // px (between the quotes)
        50_000_000 + (i as i64) * 3,               // sz
        1_700_000_000_000_000_000 + i as i64 - 17, // the print's own, earlier, time
        i % 2 == 0,                                // aggressor was the seller
    );
    if i % 5 == 4 {
        return s; // no ticker seen yet: the whole tail stays at its zero sentinel
    }
    s.with_ticker(
        base + 125_000,                        // mark_px, between bid and ask
        base - 1_000_000,                      // index_px, below it: a positive basis
        (1_250 + i as i64, 3_600_000_000_000), // funding: rate, hourly interval in ns
        // Hyperliquid stamps no venue time on `activeAssetCtx`, so half the fixture
        // exercises the venue-timed case and half the receipt-only one — a reader
        // that collapsed the two clocks passes only one of them.
        if i % 2 == 0 {
            0
        } else {
            1_700_000_000_000_000_000 + i as i64 - 40
        },
        1_700_000_000_000_000_000 + i as i64 - 33, // ts_ingest, always present
    )
}

/// The beacon's wall clock, as a formula rather than a reading.
///
/// A real session passes the pass loop's own `SystemClock::now_ns()` — one read per
/// iteration, shared with the mark cache's liveness clock. A fixture must not, or the
/// bytes it writes stop being reproducible; this keeps the field **non-zero** (so the
/// reader's "there was a wall clock" path is exercised against real Rust bytes) and
/// deterministic at the same time. `0` is the sentinel for a session with no wall clock
/// at all, and a fixture that emitted it would leave that distinction untested.
fn fixture_wall_ns(beat: u64) -> u64 {
    1_700_000_000_000_000_000 + beat * 1_000_000
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: md_writer <path> <count> [capacity] [beats]");
    let count: u64 = args
        .next()
        .expect("missing <count>")
        .parse()
        .expect("count must be an integer");
    let capacity: u64 = match args.next() {
        Some(s) => s.parse().expect("capacity must be an integer"),
        None => count.next_power_of_two().max(2),
    };
    // Total beats the run will record, *including* the final shutdown beat — one pass
    // per push plus one for the shutdown is the floor, so a smaller request is clamped
    // up rather than silently producing a different number than the caller asked for.
    let beats: u64 = match args.next() {
        Some(s) => s.parse().expect("beats must be an integer"),
        None => count + 1,
    }
    .max(count + 1);

    let producer = MdProducer::create(&path, capacity).expect("create md ring");
    let bpath = beacon_path(&path);
    let beacon = MdBeacon::create(&bpath).expect("create md beacon");

    let mut beat = MdBeat::default();
    let mut i = 0u64;
    while i < count {
        if producer.try_push(&make_md_slice(i)) {
            i += 1;
            beat.published = i;
            beat.last_event_ns = 1_700_000_000_000_000_000 + i as i64;
        } else {
            std::thread::yield_now(); // ring full (only if capacity < count and a slow reader)
        }
        // On the pass, not inside the push: a beacon that only advanced when a record
        // was written would carry exactly what the ring already carries.
        beat.wall_ns = fixture_wall_ns(beacon.beats() + 1);
        beacon.beat(beat);
    }
    // Passes on which the ring gained nothing. This is the state the whole file exists
    // to make legible: `published` frozen, `beats` climbing.
    while beacon.beats() + 1 < beats {
        beat.coalesced += 1;
        beat.last_event_ns += 1;
        beat.wall_ns = fixture_wall_ns(beacon.beats() + 1);
        beacon.beat(beat);
    }

    producer.flush().expect("flush ring");
    // The publisher shutting down on purpose, which a reader must be able to tell from
    // a crash even though it has to treat the feed as gone either way.
    beat.wall_ns = fixture_wall_ns(beacon.beats() + 1);
    beacon.stop(beat);
    eprintln!(
        "md_writer: wrote {count} md slices to {path} (capacity {capacity}); \
         {} beacon beats to {}",
        beacon.beats(),
        bpath.display()
    );
}
