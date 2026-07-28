//! Fixtures shared by this crate's unit tests. Test-only; never compiled into a
//! release build.

use std::io::{self, Write};

use axon_core::{
    Candle, CandleInterval, Clock, Cloid, Decimal, Event, EventHandler, ExecEvent, Fill, Liquidity,
    ManualClock, MarketEvent, Nanos, OrderId, Side, SymbolId, Ticker, Trade,
};
use rust_decimal_macros::dec;

use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid};

use crate::capture::Capture;
use crate::instruments::InstrumentSet;
use crate::log::{LogHeader, LoggedEvent};

/// A public trade print on symbol 1.
pub fn a_trade(ts: Nanos) -> Event {
    a_trade_at(ts, dec!(100.5))
}

/// The same, at a caller-chosen price — so events sharing a timestamp stay
/// distinguishable in an ordering assertion.
pub fn a_trade_at(ts: Nanos, px: Decimal) -> Event {
    Event::Market(MarketEvent::Trade(Trade {
        symbol_id: SymbolId::new(1),
        px,
        sz: dec!(2),
        side: Side::Buy,
        ts_event: ts,
    }))
}

/// One of our own executions, on a different symbol so the two families are easy to
/// tell apart in a trace.
pub fn a_fill(ts: Nanos) -> Event {
    Event::Exec(ExecEvent::Fill(Fill {
        symbol_id: SymbolId::new(7),
        order_id: OrderId::new(11),
        cloid: Some(Cloid::new(22)),
        side: Side::Buy,
        qty: dec!(1),
        price: dec!(100),
        fee: dec!(0.01),
        closed_pnl: Decimal::ZERO,
        liquidity: Liquidity::Taker,
        trade_id: 33,
        ts_event: ts,
    }))
}

/// A bar for `symbol 1`, stamped the way a venue stamps one: `ts_event` is
/// `open_time + interval`, the moment the bar *will* close.
///
/// `open_ms`/`interval_ms` in milliseconds so a test reads as the timeline it describes.
pub fn a_candle(open_ms: Nanos, interval_ms: Nanos) -> Event {
    let ms = 1_000_000;
    Event::Market(MarketEvent::Candle(Candle {
        symbol_id: SymbolId::new(1),
        interval: CandleInterval::M1,
        open: dec!(100),
        high: dec!(101),
        low: dec!(99),
        close: dec!(100.5),
        volume: dec!(3),
        open_time: open_ms * ms,
        ts_event: (open_ms + interval_ms) * ms,
    }))
}

/// A ticker the way Hyperliquid's `activeAssetCtx` sends one: **no venue timestamp**,
/// so its ordering key is our own receipt clock (ADR-0011). 65% of a real soak tape.
pub fn a_receipt_stamped_ticker(ts_ingest: Nanos) -> Event {
    Event::Market(MarketEvent::Ticker(Ticker {
        symbol_id: SymbolId::new(1),
        mark_px: dec!(100),
        index_px: None,
        mid_px: None,
        funding: None,
        open_interest: None,
        ts_venue: None,
        ts_ingest,
    }))
}

/// The same frame from a venue that *does* stamp it — the case that must stay inside
/// the measured subset, so the exclusion is about the missing stamp and not about
/// tickers.
pub fn a_venue_stamped_ticker(ts_venue: Nanos) -> Event {
    Event::Market(MarketEvent::Ticker(Ticker {
        symbol_id: SymbolId::new(1),
        mark_px: dec!(100),
        index_px: None,
        mid_px: None,
        funding: None,
        open_interest: None,
        ts_venue: Some(ts_venue),
        ts_ingest: ts_venue + 1,
    }))
}

/// The two instruments the fixtures use, in the shape a venue publishes them:
/// `szDecimals` 5 and 4, so `6 - szDecimals` price decimals capped at five significant
/// figures, a `10^-szDecimals` lot and a $10 minimum (ADR-0025 §1).
///
/// This is the table a *live* session would have built from one `meta` read, and
/// therefore the one a replay of that session has to be handed to reproduce it.
pub fn declared_grids() -> InstrumentSet {
    let mut table = InstrumentTable::new();
    for (id, sz) in [(1u32, 5u32), (2, 4)] {
        table.insert(InstrumentSpec {
            symbol_id: SymbolId::new(id),
            price: PriceGrid::decimals_with_sig_figs(6 - sz, 5).unwrap(),
            size: SizeGrid::decimals(sz).unwrap(),
            min_notional: Some(dec!(10)),
        });
    }
    InstrumentSet::of(&table)
}

/// Serialize `events` into a complete log with a **fixed** header, so a test can
/// compare whole files byte for byte without the wall-clock `created_ns` making
/// every run different.
///
/// Declares no grid, which is what a writer that was never handed one records. Tests
/// about rounding use [`log_bytes_with`].
pub fn log_bytes(events: &[Event]) -> String {
    log_bytes_with(InstrumentSet::Undeclared, events)
}

/// The same, for a log that carries the grid its session planned against.
pub fn log_bytes_with(instruments: InstrumentSet, events: &[Event]) -> String {
    let header = LogHeader::new("unit-test", 0);
    let mut cap = Capture::with_header(Vec::new(), header, instruments).unwrap();
    for ev in events {
        cap.record(ev).unwrap();
    }
    String::from_utf8(cap.finish().unwrap()).unwrap()
}

/// A sink that accepts `limit` lines and then reports a full disk.
///
/// Counting *lines* rather than bytes keeps the failure landing in a predictable
/// place: the header is exactly one line, so `after_lines(1)` fails on the first
/// record regardless of how long the header happens to be.
#[derive(Debug)]
pub struct FailingWriter {
    lines: usize,
    limit: usize,
}

impl FailingWriter {
    pub fn after_lines(limit: usize) -> Self {
        Self { lines: 0, limit }
    }
}

impl Write for FailingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.lines >= self.limit {
            return Err(io::Error::other("no space left on device"));
        }
        self.lines += buf.iter().filter(|b| **b == b'\n').count();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A handler that reduces a run to a byte string.
///
/// Every line records the clock as the handler saw it, the event's own time, and the
/// event's canonical JSON. Including the clock is the point: it is what turns "this
/// handler read wall-clock time" from an invisible source of drift into a failed
/// byte comparison.
pub struct DigestHandler<'a> {
    clock: &'a ManualClock,
    out: Vec<u8>,
    clock_readings: Vec<Nanos>,
    event_times: Vec<Nanos>,
    trade_prices: Vec<Decimal>,
    market: u64,
    exec: u64,
}

impl<'a> DigestHandler<'a> {
    pub fn new(clock: &'a ManualClock) -> Self {
        Self {
            clock,
            out: Vec::new(),
            clock_readings: Vec::new(),
            event_times: Vec::new(),
            trade_prices: Vec::new(),
            market: 0,
            exec: 0,
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.out
    }

    pub fn clock_readings(&self) -> &[Nanos] {
        &self.clock_readings
    }

    pub fn event_times(&self) -> &[Nanos] {
        &self.event_times
    }

    pub fn trade_prices(&self) -> &[Decimal] {
        &self.trade_prices
    }

    pub fn market_events(&self) -> u64 {
        self.market
    }

    pub fn exec_events(&self) -> u64 {
        self.exec
    }
}

impl EventHandler for DigestHandler<'_> {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        let now = self.clock.now_ns();
        self.clock_readings.push(now);
        self.event_times.push(ts_event);
        match event {
            Event::Market(MarketEvent::Trade(t)) => {
                self.market += 1;
                self.trade_prices.push(t.px);
            }
            Event::Market(_) => self.market += 1,
            Event::Exec(_) => self.exec += 1,
        }
        let json = serde_json::to_string(&LoggedEvent::from(event)).unwrap();
        writeln!(self.out, "{now} {ts_event} {json}").unwrap();
    }
}
