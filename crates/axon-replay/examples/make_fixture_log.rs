//! Regenerate the committed fixtures the Rust and Python golden tests both replay:
//! `crates/axon-replay/testdata/session.jsonl` (the event log) and, with `--signals`,
//! `session.signals.jsonl` (what the strategy sent during it).
//!
//! ```text
//! cargo run -p axon-replay --example make_fixture_log -- \
//!     crates/axon-replay/testdata/session.jsonl \
//!     --signals crates/axon-replay/testdata/session.signals.jsonl
//! ```
//!
//! The two are generated together on purpose. A signal is a decision *about* a market
//! moment, so its timestamps only mean anything against the events beside them: split
//! the generators and the next edit to the session silently leaves every signal
//! pointing at a book that has moved.
//!
//! The session is **synthetic**, and the log's `source` says so. A capture of real
//! Hyperliquid traffic would be the better fixture and is what this harness is for,
//! but a committed one would carry a real account's fills into the repository, and a
//! fixture nobody can regenerate offline is a fixture that rots. What matters for the
//! bottom rung of `docs/07` is that the log exercises every event variant, contains a
//! genuine out-of-order arrival, and is produced by the **real** [`Capture`] path
//! rather than hand-written JSON — a hand-written fixture would encode this author's
//! belief about the format instead of the format.
//!
//! Every timestamp and price is fixed, so regenerating byte-identically is the
//! expected outcome. `created_ns` is pinned for the same reason: a wall-clock stamp
//! would make the file differ on every run and the diff would be noise.

use std::process::ExitCode;

use axon_contracts::{Signal, FLAG_CLOSE};
use axon_core::{
    AccountSnapshot, Bbo, BookSnapshot, CancelReason, Candle, CandleInterval, Cloid, Decimal,
    Event, ExecEvent, Fill, Funding, Level, Liquidity, MarketEvent, Nanos, OrderId, OrderStatus,
    OrderUpdate, Side, SymbolId, Ticker, Trade,
};
use axon_providers::InstrumentTable;
use axon_replay::signals::{SIGNAL_SCHEMA, SIGNAL_SCHEMA_VERSION};
use axon_replay::{Capture, InstrumentSet, LogHeader, SignalLog, SignalRecord};
use rust_decimal_macros::dec;

const MS: Nanos = 1_000_000;
/// An arbitrary but fixed session start. Pinned so the fixture is reproducible.
const T0: Nanos = 1_774_000_000_000_000_000;

const BTC: SymbolId = SymbolId::new(1);
const ETH: SymbolId = SymbolId::new(2);

/// `whole.tenths` as an exact decimal — avoids float literals anywhere near a price.
fn px(tenths: i64) -> Decimal {
    Decimal::new(tenths, 1)
}

fn book(symbol_id: SymbolId, mid_tenths: i64, ts_event: Nanos) -> Event {
    let bids = (0..5)
        .map(|j| Level::new(px(mid_tenths - 10 - j * 10), Decimal::new(5 + j, 1)))
        .collect();
    let asks = (0..5)
        .map(|j| Level::new(px(mid_tenths + 10 + j * 10), Decimal::new(4 + j, 1)))
        .collect();
    Event::Market(MarketEvent::Book(BookSnapshot {
        symbol_id,
        bids,
        asks,
        ts_event,
    }))
}

fn ticker(symbol_id: SymbolId, mark_tenths: i64, ts_ingest: Nanos) -> Event {
    Event::Market(MarketEvent::Ticker(Ticker {
        symbol_id,
        mark_px: px(mark_tenths),
        index_px: Some(px(mark_tenths - 5)),
        mid_px: Some(px(mark_tenths + 1)),
        funding: Some(Funding {
            rate: dec!(0.0000125),
            interval: 3_600 * 1_000_000_000,
        }),
        open_interest: Some(dec!(688.11)),
        // Hyperliquid's activeAssetCtx carries no venue timestamp, so the fixture
        // keeps the receipt-time case in the log — it is the one event whose ordering
        // key is not the venue's, and the harness has to survive it.
        ts_venue: None,
        ts_ingest,
    }))
}

fn bbo(symbol_id: SymbolId, mid_tenths: i64, ts_event: Nanos) -> Event {
    Event::Market(MarketEvent::Bbo(Bbo {
        symbol_id,
        bid_px: px(mid_tenths - 10),
        bid_sz: dec!(0.7),
        ask_px: px(mid_tenths + 10),
        ask_sz: dec!(0.4),
        ts_event,
    }))
}

fn trade(symbol_id: SymbolId, tenths: i64, sz: Decimal, side: Side, ts_event: Nanos) -> Event {
    Event::Market(MarketEvent::Trade(Trade {
        symbol_id,
        px: px(tenths),
        sz,
        side,
        ts_event,
    }))
}

fn session() -> Vec<Event> {
    let mut ev = vec![
        book(BTC, 500_000, T0),
        ticker(BTC, 500_005, T0 + MS),
        book(ETH, 30_000, T0 + 2 * MS),
        ticker(ETH, 30_002, T0 + 3 * MS),
    ];

    for i in 0..10i64 {
        let t = T0 + (10 + 5 * i) * MS;
        let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
        ev.push(bbo(BTC, 500_000 + i * 10, t));
        ev.push(trade(
            BTC,
            500_000 + i * 10 + 5,
            Decimal::new(i + 1, 2),
            side,
            t + MS,
        ));
        ev.push(bbo(ETH, 30_000 + i * 2, t + 2 * MS));
        ev.push(trade(
            ETH,
            30_000 + i * 2 + 1,
            Decimal::new(i + 3, 1),
            side.opposite(),
            t + 3 * MS,
        ));

        // Hyperliquid resends `l2Book` as a full snapshot, so the book only moves when
        // one arrives. Without these the mid column would be constant for the whole
        // trace and a golden diff would have almost nothing to catch.
        if i % 3 == 0 {
            ev.push(book(BTC, 500_000 + i * 10, t + 4 * MS));
            ev.push(book(ETH, 30_000 + i * 2, t + 4 * MS));
        }
        if i == 5 {
            ev.push(ticker(BTC, 500_055, t + 4 * MS));
        }

        if i == 4 {
            // Our own order working through the middle of the session, so the log
            // carries both event families interleaved by event time — the property
            // ADR-0010 put them on one bus for.
            ev.push(Event::Exec(ExecEvent::Order(OrderUpdate {
                symbol_id: BTC,
                order_id: OrderId::new(56_968_034_936),
                cloid: Some(Cloid::new(0x0000_0000_0000_0000_0000_0000_0000_002a)),
                side: Side::Buy,
                status: OrderStatus::Resting,
                price: px(500_030).into(),
                orig_qty: dec!(0.05),
                remaining_qty: dec!(0.05),
                cancel_reason: None,
                ts_event: t + 4 * MS,
            })));
            ev.push(Event::Exec(ExecEvent::Fill(Fill {
                symbol_id: BTC,
                order_id: OrderId::new(56_968_034_936),
                cloid: Some(Cloid::new(0x0000_0000_0000_0000_0000_0000_0000_002a)),
                side: Side::Buy,
                qty: dec!(0.02),
                price: px(500_030),
                fee: dec!(0.0125),
                closed_pnl: Decimal::ZERO,
                liquidity: Liquidity::Maker,
                trade_id: 987_654_321,
                ts_event: t + 5 * MS,
            })));
        }
        if i == 7 {
            ev.push(Event::Exec(ExecEvent::Order(OrderUpdate {
                symbol_id: BTC,
                order_id: OrderId::new(56_968_034_936),
                cloid: Some(Cloid::new(0x0000_0000_0000_0000_0000_0000_0000_002a)),
                side: Side::Buy,
                status: OrderStatus::Cancelled,
                price: px(500_030).into(),
                orig_qty: dec!(0.05),
                // The unfilled remainder, not zero: a cancel of a partially filled
                // order is the case that turns into a phantom position when a format
                // or a tracker gets it wrong, so the fixture contains one.
                remaining_qty: dec!(0.03),
                cancel_reason: Some(CancelReason::Requested),
                ts_event: t + 4 * MS,
            })));
        }
    }

    let tail = T0 + 60 * MS;
    ev.push(Event::Market(MarketEvent::Candle(Candle {
        symbol_id: BTC,
        interval: CandleInterval::M1,
        open: px(500_000),
        high: px(500_095),
        low: px(499_990),
        close: px(500_090),
        volume: dec!(0.55),
        open_time: T0,
        ts_event: tail,
    })));
    ev.push(Event::Exec(ExecEvent::Account(AccountSnapshot {
        equity: dec!(10432.75),
        withdrawable: dec!(9932.75),
        margin_used: dec!(500),
        ts_event: tail + MS,
    })));

    // One genuinely late arrival, at the end of the file but stamped mid-session.
    // A real feed does this and a fixture without one would let an ordering bug pass:
    // `ReplayOrder::EventTime` and `AsCaptured` produce different runs from this log,
    // and the harness has to say so rather than paper over it.
    ev.push(trade(BTC, 500_042, dec!(0.01), Side::Sell, T0 + 32 * MS));

    ev
}

/// The grid this session "planned against": **none**, declared as such.
///
/// [`InstrumentTable::unconstrained`] rather than a fabricated Hyperliquid grid, and
/// rather than [`InstrumentSet::Undeclared`], because the three states mean three things
/// and only one of them is true here. This session had no venue: nothing published a
/// tick, so "this venue imposes no price grid" is the honest sentence, and it is the
/// sentence that makes a replay of the fixture plan exactly what the generator's own
/// numbers say. `Undeclared` would mean "a grid existed and we failed to record it",
/// which is a different fixture describing a different bug.
///
/// A *made-up* grid would be worse than either. The fixture is the artifact the Python
/// golden gate diffs, and pinning it to invented tick sizes would make every one of
/// those assertions a statement about this author's guess at a venue. The grid a replay
/// really has to round on is exercised where it is real — by
/// `a_log_that_declares_a_grid_replays_at_the_prices_that_grid_produces` in
/// `tests/golden_chain.rs`, and by a session recording its own table
/// (`SessionRecorder::start_with`).
fn instruments() -> InstrumentSet {
    InstrumentSet::of(&InstrumentTable::unconstrained())
}

/// A quantity on the wire: `contracts/schema.toml` fixes the scale at `10^-8`.
fn units(whole_hundredths: i64) -> i64 {
    whole_hundredths * 1_000_000
}

/// The signals the strategy sent during that session.
///
/// Five records, each chosen because it is the *only* one exercising something the
/// order-level golden would otherwise never see:
///
/// 1. An opening buy with nothing working — the plain path, and the one that proves a
///    `cloid` is derived from the record rather than minted from a counter.
/// 2. A sell on the second instrument, so a golden diff catches a plan computed
///    against the wrong symbol's book.
/// 3. A target raised while the venue's own order rests. This is the cancel-then-order
///    path, and the cancel has to be re-addressed to the venue's order id: the
///    `cloid` on an adopted order may be one the tracker synthesized, which the venue
///    has never seen (ADR-0020).
/// 4. A record that sat in the ring past its own TTL. Released late on purpose —
///    `release_ts` is the only way a replay can reproduce an expiry, and a stale
///    target acted on late is systematically late in the direction the market moved.
/// 5. A flatten. `FLAG_CLOSE` ignores the target field and implies reduce-only, so the
///    order that comes out must be a reduce-only IOC for exactly the position the
///    log's own fill left behind — never a fraction more.
///
/// Record 3 also carries a **stated `ts_cause`** and it is the only one that does. A
/// field that is zero in every committed fixture is a field no golden exercises: the
/// `cause` latency stage would read empty on every replay, which is indistinguishable
/// from the stage not existing. One record states a cause and four state none, so the
/// golden covers both readings of the field rather than only the default.
fn signals() -> Vec<SignalRecord> {
    let ms = |n: i64| T0 + n * MS;
    vec![
        // 1. Be long 0.03 BTC, passively (urgency 0 is post-only at the near touch).
        SignalRecord::now(&Signal::target_position(
            1,
            ms(12),
            BTC.get(),
            units(3),
            0,
            0,
            500,
            1,
            0,
        )),
        // 2. Be short 0.5 ETH, crossing the spread but staying a limit (urgency 2).
        SignalRecord::now(&Signal::target_position(
            2,
            ms(23),
            ETH.get(),
            -units(50),
            2,
            0,
            500,
            1,
            0,
        )),
        // 3. Raise the BTC target while order 56968034936 is still resting. This is also
        //    the one record with a stated cause: the book at +30 ms is what it answers, so
        //    the `cause` stage reads 7 ms on a replay of this log.
        SignalRecord::now(
            &Signal::target_position(3, ms(37), BTC.get(), units(5), 1, 0, 500, 1, 0)
                .with_ts_cause(ms(30)),
        ),
        // 4. Decided at +20 ms with a 5 ms TTL, readable only at +50 ms: expired
        //    before it was ever seen, and the reader must refuse it.
        SignalRecord::released_at(
            ms(50),
            &Signal::target_position(4, ms(20), BTC.get(), 0, 0, 0, 5, 1, 0),
        ),
        // 5. Get out, as fast as the venue allows (urgency 3 is IOC through the touch).
        SignalRecord::now(&Signal::target_position(
            5,
            ms(56),
            BTC.get(),
            0,
            3,
            0,
            500,
            1,
            FLAG_CLOSE,
        )),
    ]
}

fn write_signals(path: &str) -> ExitCode {
    let header = LogHeader::for_schema(
        SIGNAL_SCHEMA,
        SIGNAL_SCHEMA_VERSION,
        "synthetic:two-symbol-session signals (make_fixture_log)",
        T0,
    );
    let records = signals();
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = SignalLog::write(std::io::BufWriter::new(file), &header, &records) {
        eprintln!("{path}: {e}");
        return ExitCode::from(2);
    }
    eprintln!("wrote {} signals to {path}", records.len());
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut signals_path = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--signals" => match args.next() {
                Some(p) => signals_path = Some(p),
                None => {
                    eprintln!("--signals needs a path");
                    return ExitCode::from(2);
                }
            },
            other => path = Some(other.to_string()),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: make_fixture_log <out.jsonl> [--signals <out.signals.jsonl>]");
        return ExitCode::from(2);
    };
    let header = LogHeader::new("synthetic:two-symbol-session (make_fixture_log)", T0);
    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    let mut cap = match Capture::with_header(std::io::BufWriter::new(file), header, instruments()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    let events = session();
    for ev in &events {
        if let Err(e) = cap.record(ev) {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    }
    let stats = cap.stats();
    if let Err(e) = cap.finish() {
        eprintln!("{path}: {e}");
        return ExitCode::from(2);
    }
    eprintln!(
        "wrote {} events to {path} ({} late arrival(s))",
        stats.events, stats.late_arrivals
    );
    match signals_path {
        Some(p) => write_signals(&p),
        None => ExitCode::SUCCESS,
    }
}
