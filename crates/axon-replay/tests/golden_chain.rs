//! The golden test for the **whole** replayed chain: book → mark cache → order
//! tracker → strategy adapter.
//!
//! `crates/axon-replay/src/replay.rs` already proves that two replays of one log
//! dispatch the same events in the same order. That is necessary and it is not
//! sufficient: everything above the bottom rung of `docs/07-parity-and-testing.md`
//! compares *reconciliation* and *strategy output*, and neither was under a golden
//! until this file existed. ADR-0018 said as much in its own consequences.
//!
//! Every test here drives the production chain through `examples/chain/mod.rs` — the
//! same driver the `replay_log` binary and therefore `axon.backtest` use. A second
//! driver here would make these tests a statement about a harness nobody ships.

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, RwLock};

use axon_core::{
    BookSnapshot, Cloid, Decimal, Event, EventHandler, ExecEvent, Fill, Level, Liquidity,
    ManualClock, MarketEvent, Nanos, OrderId, Side, SymbolId,
};
use axon_execution::OrderTracker;
use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid};
use axon_replay::chain::{digest, Cell, ChainRow, ChainSummary, PlannedOrder};
use axon_replay::{
    Capture, EventLog, InstrumentSet, LogHeader, ReplayOrder, ReplaySource, SignalLog,
};
use rust_decimal_macros::dec;

/// The chain driver, shared verbatim with the `replay_log` binary. Each consumer uses
/// a different part of its surface, hence the allow.
#[path = "../examples/chain/mod.rs"]
#[allow(dead_code)]
mod chain;

use chain::{chain_summary, replay_chain, ChainOptions, ChainProbe};

const BTC: SymbolId = SymbolId::new(1);

fn fixture(name: &str) -> String {
    format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn committed() -> (ReplaySource, SignalLog) {
    (
        ReplaySource::open(fixture("session.jsonl")).expect("the committed log must still parse"),
        SignalLog::open(fixture("session.signals.jsonl"))
            .expect("the committed signal log must still parse"),
    )
}

fn options(signals: SignalLog) -> ChainOptions {
    ChainOptions {
        order: ReplayOrder::EventTime,
        signals,
        ..ChainOptions::default()
    }
}

/// What a caller with no table to hand passes: "this venue has no rules". Said out
/// loud in every test that does not care, so the ones that do are visible.
fn no_grid() -> Arc<InstrumentTable> {
    Arc::new(InstrumentTable::unconstrained())
}

/// The fixture's two instruments in the shape a venue publishes them: `szDecimals` 5
/// and 4, so `6 - szDecimals` price decimals capped at five significant figures, a
/// `10^-szDecimals` lot and a $10 minimum (ADR-0025 §1).
///
/// This is the table a *live* session would have built from one `meta` read, and
/// therefore the one a replay of that session has to be handed to reproduce it.
fn declared_grids() -> InstrumentTable {
    let mut t = InstrumentTable::new();
    for (id, sz) in [(1u32, 5u32), (2, 4)] {
        t.insert(InstrumentSpec {
            symbol_id: SymbolId::new(id),
            price: PriceGrid::decimals_with_sig_figs(6 - sz, 5).unwrap(),
            size: SizeGrid::decimals(sz).unwrap(),
            min_notional: Some(dec!(10)),
        });
    }
    t
}

/// Serialize `events` into a log the replay source can open, with a fixed header so
/// nothing about the run depends on when it ran, and no declared grid.
fn log_of(events: &[Event]) -> ReplaySource {
    log_of_with(InstrumentSet::Undeclared, events)
}

/// The same, for a log that carries the grid its session planned against.
fn log_of_with(instruments: InstrumentSet, events: &[Event]) -> ReplaySource {
    let mut cap =
        Capture::with_header(Vec::new(), LogHeader::new("unit-test", 0), instruments).unwrap();
    for ev in events {
        cap.record(ev).unwrap();
    }
    let bytes = cap.finish().unwrap();
    ReplaySource::new(EventLog::read(&bytes[..]).unwrap())
}

/// The committed session's events, re-emitted into a log that **declares** the grid of
/// [`declared_grids`].
///
/// The committed fixture itself declares no grid, on purpose: it is a synthetic session
/// with no venue, so "this venue imposes none" is the true sentence and any tick size in
/// it would be an invention the Python golden gate then diffs against. This is the same
/// events with a real venue's shape attached, which is what a *live* capture looks like.
fn committed_on_a_grid() -> ReplaySource {
    let log = EventLog::open(fixture("session.jsonl")).expect("the committed log must parse");
    let events: Vec<Event> = log
        .records()
        .iter()
        .map(|r| Event::from(r.event.clone()))
        .collect();
    log_of_with(InstrumentSet::of(&declared_grids()), &events)
}

fn money(row: &ChainRow, column: &str) -> Option<Decimal> {
    match row.values.get(column) {
        Some(Cell::Money(d)) => Some(*d),
        _ => None,
    }
}

fn count(row: &ChainRow, column: &str) -> Option<i64> {
    match row.values.get(column) {
        Some(Cell::Count(c)) => Some(*c),
        _ => None,
    }
}

fn decision<'a>(row: &'a ChainRow, name: &str) -> &'a str {
    row.decisions.get(name).map(String::as_str).unwrap_or("")
}

// ── the golden property, now at order level ──────────────────────────────────

#[test]
fn two_replays_produce_identical_orders_and_identical_cloids() {
    // The property every rung above this one rests on, extended past market data. A
    // `cloid` is derived from the signal rather than minted from a counter precisely so
    // an order-level diff is possible (ADR-0014 §5); if two runs over one log could
    // disagree about *which* order they sent, a shadow-trading diff would be measuring
    // its own noise and a retried submit would become a second position.
    let (src, signals) = committed();

    let (first, rows_a) = replay_chain(&src, &options(signals.clone()));
    let (second, rows_b) = replay_chain(&src, &options(signals));

    assert_eq!(digest(&rows_a), digest(&rows_b), "two replays diverged");
    assert_eq!(first.orders, second.orders, "the orders differed");
    assert_eq!(first.cancels, second.cancels);
    assert_eq!(first.symbols, second.symbols, "reconciled state differed");
    assert_eq!(first.signals, second.signals);

    // Not vacuous: the fixture really does plan orders, on both instruments, and the
    // ids really are the planner's (bit 127 is `CLOID_PLANNER_TAG`).
    assert_eq!(first.orders.len(), 4);
    assert_eq!(first.cancels.len(), 1);
    let ids: Vec<&str> = first.orders.iter().map(|o| o.cloid.as_str()).collect();
    assert!(
        ids.iter()
            .all(|c| c.starts_with("0x9") || c.starts_with("0x8")),
        "a planner cloid always has bit 127 set: {ids:?}"
    );
    assert_eq!(
        ids.len(),
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        "one signal, one id"
    );
}

#[test]
fn the_committed_signal_log_replays_to_the_orders_it_was_built_to_produce() {
    // A golden that only compares a run against itself passes forever while detecting
    // nothing. This states what the fixture is *for*, so a change to the planner, the
    // urgency table or the reader shows up here as a named difference rather than as a
    // regenerated reference nobody read.
    let (src, signals) = committed();
    let (summary, _) = replay_chain(&src, &options(signals));

    let described: Vec<String> = summary
        .orders
        .iter()
        .map(|o| {
            format!(
                "{} {:?} {} @{} {:?}{}",
                o.symbol_id,
                o.side,
                o.qty,
                o.price.unwrap(),
                o.tif,
                if o.reduce_only { " reduce-only" } else { "" }
            )
        })
        .collect();
    assert_eq!(
        described,
        vec![
            // urgency 0 rests at the near touch and may not pay a taker fee
            "1 Buy 0.03000000 @49999.0 PostOnly",
            // urgency 2 crosses the spread but stays a limit
            "2 Sell 0.50000000 @2999.4 Gtc",
            // the delta, not the target: 0.05 wanted, 0.02 already held
            "1 Buy 0.03000000 @50004.0 Gtc",
            // FLAG_CLOSE flattens exactly what the log's own fill left, reduce-only
            "1 Sell 0.02 @49757.9600 Ioc reduce-only",
        ]
    );
    assert_eq!(
        summary.cancels[0].target, "oid:56968034936",
        "an order whose cloid we did not mint is cancelled by venue id, or the venue \
         never sees the cancel and the stale quote stays resting"
    );
    assert_eq!(
        summary.signals.expired, 1,
        "the record that aged in the ring"
    );
    assert_eq!(summary.signals.accepted, 4);
}

#[test]
fn a_replay_handed_no_grid_plans_a_price_the_session_it_reproduces_could_not_have_sent() {
    // The divergence the golden gate cannot see by itself. A live session plans under
    // `Precision::Known` and rounds; this harness plans under `Precision::Unconstrained`
    // and does not. `PlannedOrder::price` is compared **exactly** by
    // `python/axon/backtest/golden.py`, so every urgency-2/3 order in a real capture
    // comes back as a price flip the strategy never made — inside the harness whose one
    // job is to tell a strategy change from a harness change. `docs/07` makes that diff
    // promotion gate #5, so the gate fails for an artefact.
    //
    // It is latent only while the committed fixture's own numbers happen to sit on the
    // grid nobody declared for it. Declare one and the last order moves.
    let (src, signals) = committed();
    let (loose, _) = replay_chain(&src, &options(signals.clone()));
    let (gridded, _) = replay_chain(
        &src,
        &ChainOptions {
            instruments: Some(Arc::new(declared_grids())),
            ..options(signals)
        },
    );

    let px = |s: &ChainSummary, i: usize| s.orders[i].price.expect("a limit order has a price");
    let grids = declared_grids();
    let btc = grids.get(BTC).expect("the fixture's own instrument");

    // The urgency-3 flatten: bid × (1 - 50 bps) is 49757.96, and a five-significant-
    // figure grid at that magnitude has a tick of one whole dollar.
    assert_eq!(px(&loose, 3), dec!(49757.9600));
    assert_eq!(
        px(&gridded, 3),
        dec!(49757),
        "a marketable sell floors onto the grid, which is what the session sent"
    );
    assert!(
        !btc.price.is_valid(px(&loose, 3)),
        "the unconstrained replay planned a price `order_wire` refuses outright - a \
         golden diff against a live capture would report it as a strategy change"
    );
    assert!(btc.price.is_valid(px(&gridded, 3)));

    // And nothing else moves: same orders, same ids, same sides, same sizes. The grid
    // changes prices and only prices, so a diff between these two runs is readable.
    assert_eq!(loose.orders.len(), gridded.orders.len());
    for (a, b) in loose.orders.iter().zip(&gridded.orders) {
        assert_eq!(
            (a.cloid.as_str(), a.side, a.qty, a.reduce_only),
            (b.cloid.as_str(), b.side, b.qty, b.reduce_only)
        );
    }
    assert_eq!(loose.cancels, gridded.cancels);
    assert_eq!(loose.signals, gridded.signals);
}

#[test]
fn a_log_that_declares_a_grid_replays_at_the_prices_that_grid_produces() {
    // ADR-0027's whole point, end to end and with no caller help. The log carries the
    // table; `replay_chain` is handed `instruments: None`; the planner rounds. Before the
    // version bump the only way to get this was to *tell* the harness the grid out of
    // band, which meant every real capture replayed at prices the session it reproduces
    // could not have sent — and the golden gate reported that as a strategy flip.
    let signals = SignalLog::open(fixture("session.signals.jsonl")).expect("the signal log");
    let (from_log, _) = replay_chain(&committed_on_a_grid(), &options(signals.clone()));
    let (told, _) = replay_chain(
        &committed_on_a_grid(),
        &ChainOptions {
            instruments: Some(Arc::new(declared_grids())),
            ..options(signals)
        },
    );

    assert_eq!(
        from_log.orders, told.orders,
        "a log that declares its grid must plan exactly what a caller handing that same \
         grid over plans, or the table survived the round trip in name only"
    );
    assert_eq!(
        from_log.orders[3].price,
        Some(dec!(49757)),
        "the urgency-3 flatten floors onto the one-dollar tick five significant figures \
         imply at that magnitude - the price the session sent, not bid x (1 - 50bps)"
    );
    assert!(
        declared_grids()
            .get(BTC)
            .expect("the fixture's instrument")
            .price
            .is_valid(from_log.orders[3].price.unwrap()),
        "a replayed price `order_wire` would refuse is a replay of a session that could \
         not have happened"
    );
}

#[test]
fn a_caller_supplied_grid_overrides_the_log_rather_than_the_other_way_round() {
    // The precedence, pinned. A log's own table is the default *because* it is the one
    // the session used; an override exists for the caller who knows better and has to
    // say so. Silently preferring the log would make the override useless the day a
    // pre-ADR-0027 capture's grid is recovered from somewhere else; silently preferring
    // the argument would make every default caller round on `ChainOptions`' idea of a
    // venue instead of the recording's.
    let signals = SignalLog::open(fixture("session.signals.jsonl")).expect("the signal log");
    let (overridden, _) = replay_chain(
        &committed_on_a_grid(),
        &ChainOptions {
            instruments: Some(no_grid()),
            ..options(signals)
        },
    );
    assert_eq!(
        overridden.orders[3].price,
        Some(dec!(49757.9600)),
        "the caller asked for no grid and got no grid, over a log that declares one"
    );
}

#[test]
fn the_as_captured_traversal_has_a_golden_of_its_own() {
    // Until this existed, `--order as-captured` was exercised and *pinned to nothing*:
    // the two traversals were only ever compared against each other, so a change that
    // moved both moved silently. It is the traversal that matters most — ADR-0018 §4
    // says it is the only order a live-captured reference may legitimately be compared
    // under, which makes it the order every real soak tape will be replayed in.
    let (src, signals) = committed();
    let as_captured = src.clone().with_order(ReplayOrder::AsCaptured);
    let (summary, rows) = replay_chain(
        &as_captured,
        &ChainOptions {
            order: ReplayOrder::AsCaptured,
            ..options(signals.clone())
        },
    );

    let described: Vec<String> = summary
        .orders
        .iter()
        .map(|o| {
            format!(
                "{} {:?} {} @{} {:?}{}",
                o.symbol_id,
                o.side,
                o.qty,
                o.price.unwrap(),
                o.tif,
                if o.reduce_only { " reduce-only" } else { "" }
            )
        })
        .collect();
    assert_eq!(
        described,
        vec![
            "1 Buy 0.03000000 @49999.0 PostOnly",
            "2 Sell 0.50000000 @2999.4 Gtc",
            "1 Buy 0.03000000 @50004.0 Gtc",
            "1 Sell 0.02 @49757.9600 Ioc reduce-only",
        ]
    );
    assert_eq!(summary.order, "as-captured");
    assert_eq!(
        summary.late_arrivals, 1,
        "the fixture's out-of-order arrival"
    );

    // The orders happen to match the event-time traversal's, and that is a fact about
    // this fixture rather than a property: its one late arrival is a trade print that
    // moves no decision. The *trace* does differ, which is what keeps the assertion above
    // from being a second copy of the event-time golden — and what would have caught a
    // traversal that silently sorted.
    let (_, event_time_rows) = replay_chain(&src, &options(signals));
    assert_ne!(
        digest(&rows),
        digest(&event_time_rows),
        "the two traversals produced the same trace, so one of them is not doing what it \
         says"
    );
}

// ── reconciliation ───────────────────────────────────────────────────────────

#[test]
fn a_replayed_fill_is_attributed_to_the_order_that_caused_it() {
    // A fill that lands on the wrong order — or on none — leaves the position right and
    // the order book of our own wrong, so the next plan cancels an order that is
    // already gone and leaves working the one it meant to pull. The trace records the
    // tracker's *own* attribution, read back out of its index rather than re-derived.
    let (src, signals) = committed();
    let (_, rows) = replay_chain(&src, &options(signals));

    let attributed: Vec<&ChainRow> = rows
        .iter()
        .filter(|r| !decision(r, "fill").is_empty())
        .collect();
    assert_eq!(attributed.len(), 1, "the fixture carries exactly one fill");
    let row = attributed[0];
    assert_eq!(
        decision(row, "fill"),
        "0x0000000000000000000000000000002a filled=0.02",
        "the fill belongs to the order the venue said it belonged to"
    );
    assert_eq!(money(row, "position_qty"), Some(dec!(0.02)));
    assert_eq!(
        count(row, "orphan_fills"),
        Some(0),
        "an unattributed fill is a diverged view, not a rounding difference"
    );
    // The order was adopted (we never submitted it), partially filled, then cancelled
    // with 0.03 unfilled. The resting remainder is exposure until that cancel lands.
    assert_eq!(money(row, "resting_qty"), Some(dec!(0.03)));
    assert_eq!(money(row, "risk_qty"), Some(dec!(0.05)));
}

#[test]
fn the_mark_cache_sees_the_book_update_before_the_fill_that_depends_on_it() {
    // ADR-0013 §1's ordering, made executable. Inside one event the fan-out runs book,
    // then marks (the updater reads the book it has just updated), then the tracker.
    // Reverse the first two and every mark is one frame stale, so the risk gate sizes
    // the fill below against the price that existed before the move that caused it.
    let events = [
        book(dec!(100), dec!(102), 1),
        book(dec!(200), dec!(202), 2),
        fill(dec!(1), dec!(201), 3),
    ];
    let (_, rows) = replay_chain(&log_of(&events), &ChainOptions::default());

    assert_eq!(money(&rows[0], "mark_px"), Some(dec!(101)));
    assert_eq!(
        money(&rows[1], "mark_px"),
        Some(dec!(201)),
        "the mark came from the book this very event updated, not the previous one"
    );
    // …and the tracker ran last, on top of both: the fill's own row already carries the
    // post-move mark, and the position it moved.
    assert_eq!(money(&rows[2], "mark_px"), Some(dec!(201)));
    assert_eq!(money(&rows[2], "position_qty"), Some(dec!(1)));
    assert_eq!(
        money(&rows[1], "position_qty"),
        Some(Decimal::ZERO),
        "the fill had not arrived yet"
    );
}

// ── a tracker that stopped answering ─────────────────────────────────────────

/// Poison the tracker's lock the way production does: a panic on some *other* path
/// while the write guard is held.
///
/// A log cannot arrange this from the inside — a panic in `OrderTracker::on_event`
/// unwinds the replay thread rather than being caught by anything — and the live shape
/// is a panic at the async submit edge, which leaves the core loop running against a
/// tracker that has stopped answering. The panic message this prints on stderr is the
/// test working, not the test failing.
fn poison(tracker: &Arc<RwLock<OrderTracker>>) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _held = tracker.write().expect("the lock starts healthy");
        panic!("a panic under the tracker's write guard");
    }));
    assert!(tracker.read().is_err(), "the lock is not actually poisoned");
}

#[test]
fn a_poisoned_tracker_reads_as_absent_and_never_as_a_flat_book() {
    // The failure this exists to prevent: a panic elsewhere poisons the tracker's lock,
    // `CoreHandler` keeps the session alive by dropping the execution events it can no
    // longer apply, and every later trace row reports position 0, risk 0, resting 0, no
    // open orders and no orphan fills — byte-identical to a genuinely flat session.
    // Both sides of a golden agree on that lie, so the comparison stays green while the
    // harness certifies exactly the outage it exists to detect.
    let clock = ManualClock::new(0);
    let mut probe = ChainProbe::new(&clock, Vec::new(), no_grid());
    for ev in [book(dec!(100), dec!(102), 1), fill(dec!(1), dec!(101), 2)] {
        probe.on_event(ev.ts_event(), &ev);
    }
    assert_eq!(money(&probe.rows()[1], "position_qty"), Some(dec!(1)));
    assert_eq!(count(&probe.rows()[1], "dropped_exec_events"), Some(0));
    assert!(
        matches!(
            probe.rows()[1].values.get("orphan_fills"),
            Some(Cell::Count(_))
        ),
        "a tracker that can be read always answers this one"
    );

    poison(probe.core().tracker());

    // A market-data tail after the poisoning: the rows that used to say nothing at all,
    // because only a fill's own row ever carried "unreadable".
    let tail = book(dec!(200), dec!(202), 3);
    probe.on_event(tail.ts_event(), &tail);
    let row = &probe.rows()[2];
    for column in [
        "position_qty",
        "risk_qty",
        "resting_qty",
        "open_orders",
        "orphan_fills",
    ] {
        assert_eq!(
            row.values.get(column),
            Some(&Cell::Absent),
            "{column} reported a reading the tracker could not give"
        );
    }
    // …while the market-data half of the same row still reads, so this is absence
    // exactly where the state is unknown and not a row that gave up.
    assert_eq!(money(row, "best_bid"), Some(dec!(200)));

    // And the event the fan-out had to drop is counted on the row that lost it, so a
    // poisoning is a *diff* in the golden rather than an absence two degraded runs
    // would share.
    let lost = fill(dec!(1), dec!(201), 4);
    probe.on_event(lost.ts_event(), &lost);
    let row = &probe.rows()[3];
    assert_eq!(count(row, "dropped_exec_events"), Some(1));
    assert_eq!(
        decision(row, "fill"),
        "unreadable",
        "a fill nobody could attribute is not a fill attributed to nothing"
    );
}

#[test]
fn a_summary_over_a_poisoned_tracker_reports_no_reading_not_no_exposure() {
    // The same lie one level up. `symbols` is sampled once at the end of a run, so a
    // poisoning anywhere in the session decides what the *final* state claims to be —
    // and "flat, nothing working" is the answer that makes a session which lost track
    // of its own orders look like one that went home clean.
    let src = log_of(&[book(dec!(100), dec!(102), 1), fill(dec!(1), dec!(101), 2)]);
    let opts = ChainOptions::default();
    let clock = ManualClock::new(0);
    let mut probe = ChainProbe::new(&clock, Vec::new(), no_grid());
    let report = src.run(&clock, &mut probe);

    let healthy = chain_summary(&probe, &src, &opts, &report);
    assert_eq!(healthy.symbols[&BTC.get()].position_qty, Some(dec!(1)));
    assert_eq!(healthy.dropped_exec_events, 0);

    poison(probe.core().tracker());
    let lost = fill(dec!(1), dec!(201), 3);
    probe.on_event(lost.ts_event(), &lost);
    let degraded = chain_summary(&probe, &src, &opts, &report);

    assert_eq!(degraded.symbols[&BTC.get()].position_qty, None);
    assert_eq!(degraded.symbols[&BTC.get()].risk_qty, None);
    assert_eq!(degraded.symbols[&BTC.get()].open_orders, None);
    assert_eq!(
        degraded.symbols[&BTC.get()].mid,
        Some(dec!(101)),
        "the book is still readable; it is the tracker that went dark"
    );
    assert_eq!(
        degraded.dropped_exec_events, 1,
        "a fill the tracker never saw has to reach the summary, or two degraded runs \
         compare equal and the golden certifies the outage"
    );
    assert_ne!(
        healthy, degraded,
        "a session that lost its order state must not summarize like one that did not"
    );
}

// ── the refusal ──────────────────────────────────────────────────────────────

#[test]
fn a_replay_that_would_place_an_order_never_manufactures_a_fill_for_it() {
    // The line ADR-0018 §7 draws, at order level. The market in a log is a recording,
    // not a counterparty: the four orders the strategy plans below reach no venue, so
    // none is acknowledged, none rests, none fills, and the position at the end is
    // exactly the one the *captured* session's own fill produced. A harness that wrote
    // a planned order into the tracker would be inventing an `OrderAck` and reporting
    // a P&L nobody agreed to.
    let (src, signals) = committed();
    let clock = ManualClock::new(0);
    let mut probe = ChainProbe::new(&clock, signals.records().to_vec(), no_grid());
    src.run(&clock, &mut probe);

    assert_eq!(probe.orders().len(), 4, "the strategy did want to trade");
    let tracker = probe.core().tracker().read().unwrap();
    for o in probe.orders() {
        let cloid = Cloid::new(u128::from_str_radix(&o.cloid[2..], 16).unwrap());
        assert!(
            tracker.order(cloid).is_none(),
            "a planned order entered the tracker: {}",
            o.cloid
        );
    }
    assert_eq!(
        tracker.position(BTC).qty,
        dec!(0.02),
        "only the log's own fill moved the position"
    );
    assert_eq!(tracker.orphan_fills(), 0);
    assert_eq!(
        tracker.open_count(),
        0,
        "nothing is working: the adopted order was cancelled and ours were never sent"
    );

    // And the trace says so on every row, not just at the end: the position column
    // takes exactly one step, at the one fill the log contains.
    let steps = probe
        .rows()
        .windows(2)
        .filter(|w| {
            w[0].symbol_id == Some(BTC.get())
                && w[1].symbol_id == Some(BTC.get())
                && money(&w[0], "position_qty") != money(&w[1], "position_qty")
        })
        .count();
    assert_eq!(steps, 1, "a position moved without a fill in the log");
}

// ── the harness's own invariants ─────────────────────────────────────────────

#[test]
fn a_pass_with_nothing_due_is_a_no_op_so_skipping_one_is_free() {
    // The replay polls the strategy adapter only where a record is readable, because
    // the runtime paces its passes on a wall clock and a replay drains a log faster
    // than real time — poll as fast as the log arrives and *which* events a pass landed
    // after becomes a property of the machine. The shortcut is only sound while a pass
    // that admits nothing touches nothing, so this runs the faithful schedule (a pass
    // after every event, parking each time) and demands the same answer.
    let (src, signals) = committed();
    let (fast, fast_rows) = replay_chain(&src, &options(signals.clone()));
    let (faithful, faithful_rows) = replay_chain(
        &src,
        &ChainOptions {
            poll_every_event: true,
            ..options(signals)
        },
    );

    assert_eq!(digest(&fast_rows), digest(&faithful_rows));
    assert_eq!(fast.orders, faithful.orders);
    assert_eq!(fast.cancels, faithful.cancels);
    assert_eq!(fast.signals, faithful.signals);
    assert_eq!(fast.symbols, faithful.symbols);
    assert!(
        faithful.intent_passes > fast.intent_passes,
        "the two schedules must actually differ, or this test proves nothing"
    );
    assert_eq!(fast.intent_passes, 5, "one pass per readable record");
}

#[test]
fn a_log_with_no_signals_still_replays_the_whole_chain() {
    // A market-data-only capture is the ordinary case and must not be a degraded one:
    // the fan-out, the marks and the tracker all still run, the adapter simply has
    // nothing to admit. Reporting this as an error — or silently substituting a
    // different chain — would make the harness untrustworthy exactly where most logs
    // land.
    let (src, _) = committed();
    let (summary, rows) = replay_chain(&src, &ChainOptions::default());

    assert!(summary.signal_source.is_none());
    assert_eq!(summary.orders, Vec::<PlannedOrder>::new());
    assert_eq!(summary.intent_passes, 0);
    assert_eq!(summary.events, 59);
    assert_eq!(rows.len(), 59);
    assert_eq!(
        summary.symbols[&BTC.get()].position_qty,
        Some(dec!(0.02)),
        "reconciliation is unaffected by whether a strategy was attached"
    );
}

#[test]
fn every_row_carries_every_column_even_where_the_column_does_not_apply() {
    // A golden comparison checks column *sets* before values. A row that dropped a
    // column when it had nothing to put there would be reported as a structural
    // mismatch on every later run, which buries the one reading that actually went
    // missing.
    let (src, signals) = committed();
    let (_, rows) = replay_chain(&src, &options(signals));

    let expected: Vec<&str> = rows[0].values.keys().copied().collect();
    assert_eq!(expected.len(), 16);
    for row in &rows {
        let names: Vec<&str> = row.values.keys().copied().collect();
        assert_eq!(names, expected, "row {} changed the column set", row.seq);
        assert_eq!(
            row.decisions.keys().copied().collect::<Vec<_>>(),
            vec!["cancel", "fill", "plan"]
        );
    }
    // The account snapshot is not per-symbol, so its per-symbol columns are absent —
    // absent, not zero, because a mark of zero and a missing mark are the difference
    // between a risk gate sizing against nothing and one failing closed.
    let account = rows
        .iter()
        .find(|r| r.symbol_id.is_none())
        .expect("the fixture carries one account snapshot");
    assert_eq!(account.values.get("mark_px"), Some(&Cell::Absent));
    assert_eq!(account.values.get("position_qty"), Some(&Cell::Absent));
    assert_eq!(
        account.values.get("orphan_fills"),
        Some(&Cell::Count(0)),
        "an account-level count still applies"
    );
    assert_eq!(
        account.values.get("dropped_exec_events"),
        Some(&Cell::Count(0)),
        "the fan-out's own health is a session-level reading, not a per-symbol one"
    );
}

#[test]
fn the_summary_names_the_signal_log_by_provenance_not_by_path() {
    // A path differs between checkouts, so a golden reference that carried one would
    // fail on every machine but the one that generated it — a harness manufacturing
    // its own divergence.
    let (src, signals) = committed();
    let (summary, _) = replay_chain(&src, &options(signals));
    let json = serde_json::to_string(&summary).unwrap();
    assert!(
        !json.contains("testdata/"),
        "a path leaked into the summary"
    );
    assert_eq!(
        summary.signal_source.as_deref(),
        Some("synthetic:two-symbol-session signals (make_fixture_log)")
    );
}

#[test]
fn a_replay_reports_the_same_schema_the_python_side_reads() {
    // The version is what turns "the trace was reshaped" from a meaningless comparison
    // into a failed one. Pinned here so a change to `ChainRow` breaks in the crate that
    // owns it rather than surfacing as an unexplained Python failure.
    let summary: ChainSummary = replay_chain(&committed().0, &ChainOptions::default()).0;
    assert_eq!(summary.schema, "axon.backtest");
    assert_eq!(summary.schema_version, 3);
}

// ── fixtures ─────────────────────────────────────────────────────────────────

fn book(bid: Decimal, ask: Decimal, ts_event: Nanos) -> Event {
    Event::Market(MarketEvent::Book(BookSnapshot {
        symbol_id: BTC,
        bids: vec![Level::new(bid, dec!(1))],
        asks: vec![Level::new(ask, dec!(1))],
        ts_event,
    }))
}

fn fill(qty: Decimal, price: Decimal, ts_event: Nanos) -> Event {
    Event::Exec(ExecEvent::Fill(Fill {
        symbol_id: BTC,
        order_id: OrderId::new(7),
        cloid: None,
        side: Side::Buy,
        qty,
        price,
        fee: Decimal::ZERO,
        closed_pnl: Decimal::ZERO,
        liquidity: Liquidity::Taker,
        trade_id: 1,
        ts_event,
    }))
}
