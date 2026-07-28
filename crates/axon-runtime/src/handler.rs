//! [`CoreHandler`] — the one [`EventHandler`] the deterministic loop drives, fanning
//! each event out to the three pieces of state a session keeps.
//!
//! There is exactly one handler because there is exactly one ordering. Giving the
//! book, the order tracker and the mark cache their own consumers would let a fill be
//! applied against a book that had not yet seen the trade which caused it, and replay
//! would stop matching live — the property ADR-0008 and ADR-0010 exist to protect.
//!
//! The fan-out order inside a single event is load-bearing too:
//!
//! 0. **The capture tap first**, when a session is recording (ADR-0018). ADR-0018 puts
//!    capture in the handler *chain*; there is exactly one handler here, so it goes at
//!    the head of the one fan-out — a second consumer on the bus is precisely what
//!    ADR-0013 §1 forbids. First, rather than last beside the other output, because it
//!    is the only stop that records the fan-out's **input**: it reads no state anything
//!    above it produces, so nothing is gained by waiting, and the event a log must never
//!    be missing is the last one before a crash — which is exactly the one a tap placed
//!    after the state consumers would drop. The tap itself cannot stall: it is a
//!    non-blocking hand-off to a writer thread, and it is `None` unless an operator
//!    asked for a recording.
//! 1. **Market data first** of the state consumers. The mark updater reads the book for
//!    a mid, so the book must already have this event applied or the mid lags by one
//!    frame. It is also where each symbol's book acquires its event time
//!    (`MarketDataProcessor::book_ts`), which is what lets [`crate::quote`] tell a dead
//!    `l2Book` from a quiet one.
//! 2. **Marks second**, so the risk gate's price is current before any order derived
//!    from this event could be submitted.
//! 3. **The tracker third.** Nothing above it reads order state, and it is the only
//!    consumer behind a lock, so the lock is held for the shortest possible span.
//! 4. **The market-data ring last**, when one is configured (ADR-0012). It is an
//!    *output* derived from state, not state anything above reads: a slice published
//!    mid-fan-out would describe a market the core itself had not finished forming, and
//!    Python's features would then be computed on a frame the executing core never held.
//!    Publishing after the tracker also keeps the ring write outside its critical
//!    section, which the submit path's risk context contends for from the async edge.
//!
//! One thing this handler deliberately does **not** do with every event is take its
//! clock from it — see [`advances_the_clock`].

use std::sync::{Arc, RwLock};

use axon_core::{Decimal, Event, EventHandler, MarketEvent, Nanos, SymbolId};
use axon_execution::{MarkCache, OrderTracker};
use axon_marketdata::MarketDataProcessor;

use crate::capture::CaptureTap;
use crate::mdring::MdPublisher;

/// Whether this event's `ts_event` is a moment the session **observed**, and may
/// therefore be *considered* for the core's clock.
///
/// Answering yes admits an event to the high-water comparison in
/// [`CoreHandler::on_event`]; it does not by itself move the clock. The two questions
/// are genuinely separate and the split is load-bearing — see that comparison for the
/// half this predicate cannot answer.
///
/// Exactly one event answers no, and the reason is worth stating rather than listing.
/// A [`Candle`](axon_core::Candle)'s `ts_event` is `open_time + interval` — arithmetic
/// on the bar's identity, not a time the venue reported — so it is the *same number* on
/// the bar's first frame and its last, and it is the bar's **close** time either way. It
/// is a sorting key, not an observation, and a venue republishing the bar it is still
/// filling stamps every one of those frames with a moment that has not arrived. That is
/// measured, not assumed: 13 minutes on Hyperliquid's own socket drew **1 321 candle
/// frames describing 69 bars** — one BTC 5-minute bar 192 times — every frame of a bar
/// carrying the same `T`, and 1 317 of the 1 321 received *before* the close they were
/// stamped with. The venue marked none of them final
/// (`axon_provider_hyperliquid::ws::decode::decode_candle`).
///
/// Believing that stamp is what ADR-0020 §2's pass schedule then runs on, and it fails
/// twice over with nothing on the line to say so:
///
/// 1. every record on the signal ring is aged against a clock up to a whole interval
///    ahead of reality, so a target written a moment ago is refused as `expired`;
/// 2. worse, `last_pass_ns` is left an interval in the future, and the schedule's
///    subtraction is *signed* by design — a late arrival must not trigger a pass — so
///    **no further pass runs at all** until event time catches up. The strategy stops
///    trading, `data_lag_ms` saturates to zero, and the status line reads `OK` beside
///    what looks like a quiet market.
///
/// Note what this is not. It is not "skip forming candles": nothing here can tell a
/// forming bar from a closed one, and on Hyperliquid nothing anywhere can, because the
/// venue publishes no finality bit and frequently sends no frame at or after `T` at all.
/// The rule needs no such knowledge — a candle's close time is a computed label whether
/// or not the bar behind it is finished, so it is never evidence about *now*. Finality,
/// where anyone needs it, is **derived** from the stream instead: [`crate::mdring`]
/// closes a bar when a frame for a later `open_time` arrives, and
/// `axon.strategies.data.closed_rows` applies the same rule offline. Two components
/// already answer this question identically without a flag; this is the third, and it
/// does not need one either.
///
/// Exhaustive on purpose: a future [`MarketEvent`] whose timestamp is derived rather
/// than reported has to answer this question *here*, because "the core clock quietly ran
/// ahead of the market" is not a failure that announces itself.
fn advances_the_clock(event: &Event) -> bool {
    match event {
        Event::Market(MarketEvent::Candle(_)) => false,
        Event::Market(
            MarketEvent::Bbo(_)
            | MarketEvent::Book(_)
            | MarketEvent::Trade(_)
            // A `Ticker`'s key may be our own receipt clock rather than the venue's
            // (ADR-0011 §3), which is still a moment that *happened* — unlike a close
            // time, it can never be in the future.
            | MarketEvent::Ticker(_),
        ) => true,
        Event::Exec(_) => true,
    }
}

/// The composite core-side consumer.
pub struct CoreHandler {
    /// The session recording, when one was asked for (ADR-0018). `None` is the default
    /// and is why a bare `cargo run --bin axon` writes no log.
    capture: Option<CaptureTap>,
    market: MarketDataProcessor,
    /// Behind a lock because the submit path's risk context reads it from the async
    /// edge (ADR-0010). The critical section here is one `match`, and nothing awaits
    /// inside it.
    tracker: Arc<RwLock<OrderTracker>>,
    marks: Arc<MarkCache>,
    /// The Rust→Python market-data ring, when the session publishes one. `None` is the
    /// default and is why a bare `cargo run --bin axon` creates no file.
    md: Option<MdPublisher>,
    events: u64,
    /// Events that could not be applied to the tracker because a panic poisoned its
    /// lock. Non-zero means our order state has silently stopped tracking the venue.
    dropped_exec_events: u64,
    /// Candles applied. Counted, and not merely folded into `events`, because a candle
    /// is the one market event that does **not** move `last_ts`
    /// ([`advances_the_clock`]) — so it is the one feed that can arrive in volume and
    /// leave the session with no clock at all.
    candles: u64,
    last_ts: Nanos,
}

impl CoreHandler {
    pub fn new(tracker: Arc<RwLock<OrderTracker>>, marks: Arc<MarkCache>) -> Self {
        Self {
            capture: None,
            market: MarketDataProcessor::new(),
            tracker,
            marks,
            md: None,
            events: 0,
            dropped_exec_events: 0,
            candles: 0,
            last_ts: 0,
        }
    }

    /// Record every event this handler is given. Separate from [`CoreHandler::new`]
    /// because recording is optional and a session that does not do it should not have
    /// to say so.
    pub fn with_capture(mut self, tap: CaptureTap) -> Self {
        self.capture = Some(tap);
        self
    }

    /// Attach a market-data publisher. Separate from [`CoreHandler::new`] because
    /// publishing is optional and a session that does not do it should not have to say
    /// so.
    pub fn with_md_publisher(mut self, md: MdPublisher) -> Self {
        self.md = Some(md);
        self
    }

    /// The publisher, if this session has one. `None` is what the status line prints as
    /// absence rather than as zeros.
    pub fn md(&self) -> Option<&MdPublisher> {
        self.md.as_ref()
    }

    pub fn market(&self) -> &MarketDataProcessor {
        &self.market
    }

    pub fn tracker(&self) -> &Arc<RwLock<OrderTracker>> {
        &self.tracker
    }

    pub fn marks(&self) -> &Arc<MarkCache> {
        &self.marks
    }

    pub fn events(&self) -> u64 {
        self.events
    }

    /// The event-time **high-water mark** — the newest moment this session has
    /// observed, and so the age of our view of the market.
    ///
    /// A high-water mark and not the last event's stamp: a stream can hand us an
    /// observation older than one we already applied, and the newest thing we have seen
    /// does not stop being the newest thing we have seen when something older arrives
    /// behind it. A candle is not a candidate at all; see [`advances_the_clock`].
    pub fn last_ts(&self) -> Nanos {
        self.last_ts
    }

    /// Bars that arrived while the core still has no clock — i.e. a session whose only
    /// market data is candles.
    ///
    /// The one shape of feed [`advances_the_clock`] leaves worse off, so it gets a
    /// number instead of silence. With `last_ts` at zero, ADR-0020 §2 skips **every**
    /// intent pass, and a session that runs no passes is indistinguishable from a
    /// strategy with no opinion — the same ambiguity `SIGNAL RING DETACHED` exists to
    /// remove. `BARS BUT NO CLOCK n` on the status line is the difference.
    ///
    /// Derived rather than latched, so it falls back to zero the instant any observed
    /// event starts the clock: a session that merely *began* with a bar must not carry
    /// a warning for the rest of its life.
    pub fn bars_without_a_clock(&self) -> u64 {
        if self.last_ts == 0 {
            self.candles
        } else {
            0
        }
    }

    /// Execution events lost to a poisoned tracker lock. A session with a non-zero
    /// count here is trading on a stale order book of its own and must be halted.
    pub fn dropped_exec_events(&self) -> u64 {
        self.dropped_exec_events
    }

    /// Feed the mark cache from this event.
    ///
    /// The match is exhaustive on purpose: a new [`MarketEvent`] variant carrying a
    /// price should force a decision here rather than being silently ignored by a
    /// wildcard, because "the risk gate quietly stopped seeing prices" is not a
    /// failure that announces itself.
    fn update_marks(&self, event: &Event) {
        let Event::Market(m) = event else { return };
        match m {
            // The venue's own mark: what margin, liquidation and unrealized PnL are
            // computed against, so it is the only price that measures the same
            // quantity the venue will liquidate us on.
            // `ts_event()` rather than a field: Hyperliquid's ticker carries no venue
            // timestamp, so the accessor falls back to receipt time. That fallback is
            // right here — a mark's *age* is what the gate needs, and receipt time is
            // the only clock that can measure it when the venue supplies none.
            MarketEvent::Ticker(t) => self.marks.set_mark(t.symbol_id, t.mark_px, t.ts_event()),
            // Fallbacks, used only while no venue mark is live (the cache enforces
            // that precedence). A two-sided quote is a consensus price; that is what
            // makes it an acceptable stand-in.
            MarketEvent::Bbo(b) => {
                let mid = (b.bid_px + b.ask_px) / Decimal::TWO;
                self.marks.set_mid(b.symbol_id, mid, b.ts_event);
            }
            MarketEvent::Book(s) => {
                if let Some(mid) = self.market.mid(s.symbol_id) {
                    self.marks.set_mid(s.symbol_id, mid, s.ts_event);
                }
            }
            // A trade print is deliberately **not** a mark source. One print in a thin
            // book is an outlier, and a quiet instrument's last print can be minutes
            // old while the book has moved — a stale price wearing a fresh timestamp,
            // which is exactly what the expiry rule cannot catch.
            MarketEvent::Trade(_) | MarketEvent::Candle(_) => {}
        }
    }

    /// Positions the tracker currently holds, for the status line.
    pub fn positions(&self, symbols: &[SymbolId]) -> Vec<(SymbolId, Decimal)> {
        let Ok(t) = self.tracker.read() else {
            return Vec::new();
        };
        symbols
            .iter()
            .map(|s| (*s, t.position(*s).qty))
            .filter(|(_, q)| !q.is_zero())
            .collect()
    }
}

impl EventHandler for CoreHandler {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        self.events += 1;
        // The clock is a claim about how much of the market this session has *seen*, so
        // it moves under two conditions, and both are needed.
        //
        // First, only an event that reports a moment may be considered at all. A candle
        // reports a close time computed from the bar's identity, which for a bar still
        // forming has not happened — see [`advances_the_clock`] for what believing it
        // costs.
        //
        // Second, `max` and not assignment, because `last_ts` is a **high-water mark**
        // and a high-water mark that goes down is not one. Hyperliquid replays its whole
        // `userFills` snapshot on every reconnect, carrying the original execution times
        // — up to 2.94 hours stale on a 1 h 44 m soak. Those fills pass the predicate
        // honestly: a fill's `ts_event` genuinely is a moment somebody observed, just an
        // old one. Assigning it dragged the core's clock to a fixed historical instant
        // while `marks 3/3` sat beside it on the status line, so the session looked
        // perfectly healthy while every downstream consumer of this number was reasoning
        // about a market hours gone: ADR-0020 §2's pass schedule ages the signal ring
        // against it, and `data_lag_ms` reported hours of lag on a live feed. It
        // self-healed on the next market frame, which is why nothing caught it in
        // 13 939 records of replay. The exact mirror of the forming-candle failure on
        // this same line — candles pushed the clock forward, fills pull it back — and
        // the predicate was never the wrong half.
        //
        // What a maximum gives up is real and small: a genuinely out-of-order *live*
        // stream no longer rewinds the clock, so `data_lag_ms` reports the age of the
        // newest observation rather than of the latest arrival. That is the quantity
        // anyone reading it wants, and it is the same clamp two consumers already had to
        // apply downstream for want of it — `RecordingSource::observe_event_time` and
        // `CapturedSignals::observe_event_time`, both of which exist because this line
        // used to rewind. What it does *not* give up is any record: the fan-out below is
        // unconditional, so a late fill still reaches the tracker, the recording and the
        // ring in arrival order exactly as before. Ordering is the log's business
        // (`OrderingWatch` counts every inversion); this is only the clock.
        if advances_the_clock(event) {
            self.last_ts = self.last_ts.max(ts_event);
        } else {
            self.candles += 1;
        }

        // First: the recording is of what the core was *handed*, and it reads nothing
        // the consumers below produce. Placed after them, the one event a log must never
        // lack — the last one before a panic — is the one it would lack.
        if let Some(cap) = &self.capture {
            cap.on_event(event);
        }

        self.market.on_event(ts_event, event);
        self.update_marks(event);

        match self.tracker.write() {
            Ok(mut t) => t.on_event(ts_event, event),
            // A poisoned lock means a panic left the tracker's state unknown. Dropping
            // the event is the only option left here, and trading is already refused —
            // `TrackerRiskContext` reports no mark price on a poisoned tracker, which
            // fails the gate closed. What is missing without this counter is *visibility*:
            // an order book that silently stopped updating under-reports exposure, and
            // the status line is where that has to show up.
            Err(_) => {
                if matches!(event, Event::Exec(_)) {
                    self.dropped_exec_events += 1;
                }
            }
        }

        // Last: every consumer above has this event applied, so the slice describes a
        // market state the core actually held. Non-blocking and allocation-free by
        // construction — see [`crate::mdring`].
        if let Some(md) = self.md.as_mut() {
            md.on_event(ts_event, event, &self.market);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{
        Bbo, BookSnapshot, Cloid, ExecEvent, Fill, Level, Liquidity, OrderId, Side, Ticker, Trade,
    };
    use rust_decimal_macros::dec;

    const SYM: SymbolId = SymbolId::new(1);

    fn handler() -> CoreHandler {
        CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::new()),
        )
    }

    fn feed(h: &mut CoreHandler, ev: Event) {
        h.on_event(ev.ts_event(), &ev);
    }

    fn bbo(bid: Decimal, ask: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Bbo(Bbo {
            symbol_id: SYM,
            bid_px: bid,
            bid_sz: dec!(1),
            ask_px: ask,
            ask_sz: dec!(1),
            ts_event: ts,
        }))
    }

    fn ticker(mark: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Ticker(Ticker {
            symbol_id: SYM,
            mark_px: mark,
            index_px: None,
            mid_px: None,
            funding: None,
            open_interest: None,
            ts_venue: Some(ts),
            ts_ingest: ts,
        }))
    }

    #[test]
    fn one_event_reaches_every_consumer() {
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 10));
        assert_eq!(h.events(), 1);
        assert_eq!(h.market().bbo(SYM).unwrap().bid_px, dec!(99));
        assert_eq!(h.marks().get(SYM), Some(dec!(100)), "mid became the mark");
        assert_eq!(h.last_ts(), 10);
    }

    #[test]
    fn the_venue_mark_beats_the_book_mid() {
        // The gate must measure notional in the price the venue margins us on; the mid
        // is a different quantity and is only a stand-in.
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 10));
        feed(&mut h, ticker(dec!(100.4), 11));
        assert_eq!(h.marks().get(SYM), Some(dec!(100.4)));
        feed(&mut h, bbo(dec!(98), dec!(102), 12));
        assert_eq!(
            h.marks().get(SYM),
            Some(dec!(100.4)),
            "a later BBO must not displace a live venue mark"
        );
    }

    #[test]
    fn a_trade_print_never_becomes_the_mark() {
        // A single print is not a price the risk gate can size against.
        let mut h = handler();
        feed(
            &mut h,
            Event::Market(MarketEvent::Trade(Trade {
                symbol_id: SYM,
                px: dec!(12_345),
                sz: dec!(1),
                side: Side::Buy,
                ts_event: 5,
            })),
        );
        assert_eq!(h.marks().get(SYM), None);
        assert_eq!(h.market().last_trade(SYM).unwrap().px, dec!(12_345));
    }

    #[test]
    fn a_book_snapshot_marks_from_the_updated_book_not_the_previous_one() {
        // Ordering inside the fan-out: the processor has to apply the snapshot before
        // the mark is derived, or every mark is one frame stale.
        let mut h = handler();
        feed(
            &mut h,
            Event::Market(MarketEvent::Book(BookSnapshot {
                symbol_id: SYM,
                bids: vec![Level::new(dec!(100), dec!(1))],
                asks: vec![Level::new(dec!(102), dec!(1))],
                ts_event: 1,
            })),
        );
        assert_eq!(h.marks().get(SYM), Some(dec!(101)));
    }

    #[test]
    fn a_fill_moves_the_tracker_and_leaves_the_book_alone() {
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 1));
        feed(
            &mut h,
            Event::Exec(ExecEvent::Fill(Fill {
                symbol_id: SYM,
                order_id: OrderId::new(7),
                cloid: Some(Cloid::new(1)),
                side: Side::Buy,
                qty: dec!(2),
                price: dec!(100),
                fee: dec!(0),
                closed_pnl: dec!(0),
                liquidity: Liquidity::Taker,
                trade_id: 42,
                ts_event: 2,
            })),
        );
        assert_eq!(h.tracker().read().unwrap().position(SYM).qty, dec!(2));
        assert_eq!(h.market().bbo(SYM).unwrap().bid_px, dec!(99));
        assert_eq!(h.positions(&[SYM]), vec![(SYM, dec!(2))]);
        assert_eq!(h.dropped_exec_events(), 0);
    }

    fn fill_ev(ts: Nanos, trade_id: u64) -> Event {
        Event::Exec(ExecEvent::Fill(Fill {
            symbol_id: SYM,
            order_id: OrderId::new(7),
            cloid: Some(Cloid::new(1)),
            side: Side::Buy,
            qty: dec!(2),
            price: dec!(100),
            fee: dec!(0),
            closed_pnl: dec!(0),
            liquidity: Liquidity::Taker,
            trade_id,
            ts_event: ts,
        }))
    }

    #[test]
    fn a_replayed_fill_never_drags_the_clock_back_to_when_it_happened() {
        // Hyperliquid replays the *whole* `userFills` snapshot on every reconnect,
        // stamped with the original execution times — up to 2.94 hours stale on the tape
        // this came from, and 18 of them on each of 37 reconnects. Assigning `last_ts`
        // from one pinned the core's clock to a fixed historical instant with `marks 3/3`
        // sitting beside it on the same line: three status lines 820 s apart reported
        // lags 820 064 ms apart, which is a clock that has stopped rather than one that
        // is behind. Everything downstream then reasons about a market hours gone —
        // ADR-0020 §2 ages the signal ring against this number — and the only reason it
        // was survivable is that the next market frame silently repaired it.
        let hour = 3_600_000_000_000i64;
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 3 * hour));
        assert_eq!(h.last_ts(), 3 * hour);

        feed(&mut h, fill_ev(hour / 2, 42));
        assert_eq!(
            h.last_ts(),
            3 * hour,
            "a fill from 2.5 hours ago is an old observation, not a new clock"
        );
        // Refused as a clock, not as an event: the fan-out still ran, so the fill
        // reaches the tracker, the recording and the ring exactly as before. Dropping it
        // would be a different bug — a position we hold and cannot see.
        assert_eq!(h.events(), 2);
        assert_eq!(h.tracker().read().unwrap().position(SYM).qty, dec!(2));
        assert_eq!(h.dropped_exec_events(), 0);

        // A fill that really is the newest thing we have seen still moves it, which is
        // what keeps `Event::Exec` an honest answer to `advances_the_clock`.
        feed(&mut h, fill_ev(4 * hour, 43));
        assert_eq!(h.last_ts(), 4 * hour);
    }

    #[test]
    fn a_late_market_frame_leaves_the_high_water_mark_where_it_was() {
        // The same property for the feed the venue stamps itself. `last_ts` is the
        // newest moment observed; an inversion behind it is the log's business — the
        // capture's `OrderingWatch` counts every one — and never the clock's. Two
        // downstream consumers already clamp this exact rewind for want of the clamp
        // being here (`RecordingSource` and `CapturedSignals`), and a clock that rewinds
        // strands every signal record above the old mark for the rest of a replay.
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 100));
        feed(&mut h, bbo(dec!(98), dec!(102), 90));
        assert_eq!(h.last_ts(), 100);
        assert_eq!(h.events(), 2, "still applied, just not believed as a clock");
        assert_eq!(
            h.market().bbo(SYM).unwrap().bid_px,
            dec!(98),
            "the book takes the frame it was handed"
        );
        feed(&mut h, bbo(dec!(97), dec!(103), 110));
        assert_eq!(h.last_ts(), 110);
    }

    fn candle_ev(open_time: Nanos, len: Nanos) -> Event {
        use axon_core::{Candle, CandleInterval};
        Event::Market(MarketEvent::Candle(Candle {
            symbol_id: SYM,
            interval: CandleInterval::M1,
            open: dec!(100),
            high: dec!(101),
            low: dec!(99),
            close: dec!(100),
            volume: dec!(1),
            open_time,
            // `open_time + interval`, which is what both decoders compute and what the
            // venue republishes on every frame of a bar it is still filling.
            ts_event: open_time + len,
        }))
    }

    #[test]
    fn a_forming_bar_never_becomes_the_core_clock() {
        // The failure: a venue republishes the bar it is still filling, each frame
        // carrying the close time a whole interval away. Advancing `last_ts` to it puts
        // the core's event clock ahead of anything that has happened — every signal on
        // the ring is then aged against the future and refused as expired, and
        // ADR-0020 §2's schedule, anchored an interval ahead, runs no pass at all until
        // event time catches up. Nothing about it is visible: `data_lag_ms` saturates to
        // zero and the line reads `OK`.
        let minute = 60_000_000_000i64;
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 10));
        assert_eq!(h.last_ts(), 10);

        feed(&mut h, candle_ev(10, minute));
        assert_eq!(
            h.last_ts(),
            10,
            "a close time is arithmetic on the bar's identity, not a moment we saw"
        );
        // Refused as a clock, not as an event: the fan-out still ran, so the bar reaches
        // the processor, the publisher and any recording exactly as before. Dropping it
        // would be a different bug — a hole in a feature window nothing can see.
        assert_eq!(h.events(), 2);
        assert_eq!(h.market().last_candle(SYM).unwrap().close, dec!(100));

        // And the closed bar is refused for the same reason, because from one frame the
        // two are the same event. That is the cost, and it is paid in microseconds
        // wherever any other feed is live.
        feed(&mut h, bbo(dec!(98), dec!(102), 20));
        feed(&mut h, candle_ev(20, minute));
        assert_eq!(
            h.last_ts(),
            20,
            "the clock still tracks the market, not the bar"
        );
    }

    #[test]
    fn a_session_fed_only_bars_says_so_instead_of_going_quiet() {
        // The one feed the rule above leaves worse off. With `last_ts` at zero ADR-0020
        // §2 skips every intent pass, so a bars-only session trades nothing — and
        // `NO MARKET DATA` cannot catch it, because events *are* arriving. Every counter
        // reads healthy. This number is the only thing that says why.
        let minute = 60_000_000_000i64;
        let mut h = handler();
        assert_eq!(h.bars_without_a_clock(), 0, "nothing has arrived yet");

        feed(&mut h, candle_ev(0, minute));
        feed(&mut h, candle_ev(minute, minute));
        assert_eq!(h.last_ts(), 0, "no observed event has started the clock");
        assert_eq!(h.bars_without_a_clock(), 2);

        // Derived, not latched: one real quote and the session is no longer blind, so
        // the warning must clear rather than accuse a healthy feed forever.
        feed(&mut h, bbo(dec!(99), dec!(101), 3 * minute));
        assert_eq!(h.bars_without_a_clock(), 0);
        assert_eq!(h.last_ts(), 3 * minute);
    }

    #[test]
    fn a_session_without_a_publisher_reports_absence_not_zeros() {
        // The default. A session that publishes nothing must be distinguishable from one
        // whose feed has gone quiet, and `None` is how that difference reaches the
        // status line.
        let mut h = handler();
        feed(&mut h, bbo(dec!(99), dec!(101), 10));
        assert!(h.md().is_none());
    }

    #[test]
    fn a_published_slice_describes_the_state_this_event_left_behind() {
        // The fan-out order, asserted from the outside: publishing before the processor
        // applied the event would put the *previous* frame's book on the ring under this
        // frame's timestamp, and every Python feature would be one update stale.
        use crate::config::MdRingConfig;
        use crate::mdring::{MdPublisher, MdWritePolicy};

        let path = std::env::temp_dir()
            .join(format!("axon-handler-md-{}.ring", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let md = MdPublisher::open(
            &MdRingConfig {
                enabled: true,
                path: path.clone(),
                capacity: 8,
                policy: MdWritePolicy::OnChange,
            },
            10_000_000_000,
        )
        .unwrap()
        .unwrap();

        let mut h = handler().with_md_publisher(md);
        feed(&mut h, bbo(dec!(99), dec!(101), 10));
        feed(&mut h, bbo(dec!(98), dec!(102), 20));
        assert_eq!(h.md().unwrap().stats().published, 2);

        let c = axon_ipc::MdConsumer::open(&path).unwrap();
        let first = c.try_pop().unwrap();
        assert_eq!(first.ts_event, 10);
        assert_eq!(
            first.bid_px,
            axon_strategy::decimal_to_fixed(dec!(99)).unwrap()
        );
        let second = c.try_pop().unwrap();
        assert_eq!(second.ts_event, 20);
        assert_eq!(
            second.bid_px,
            axon_strategy::decimal_to_fixed(dec!(98)).unwrap()
        );
        drop(c);
        drop(h);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_candle_reaches_the_bar_ring_through_the_same_one_fan_out() {
        // The fan-out is exhaustive by construction, but a `Candle` moves no field the
        // *slice* record has — so before ADR-0028 it fell straight through, and a
        // bar-driven strategy could not be fed at all. This asserts the one handler
        // routes it, from the outside: unwire `md.on_event` for candles and the bar ring
        // stays empty while every other counter looks healthy.
        use crate::config::MdRingConfig;
        use crate::mdring::{bar_ring_path, MdPublisher, MdWritePolicy};
        use axon_core::{Candle, CandleInterval};

        let path = std::env::temp_dir()
            .join(format!("axon-handler-bars-{}.ring", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let md = MdPublisher::open(
            &MdRingConfig {
                enabled: true,
                path: path.clone(),
                capacity: 8,
                policy: MdWritePolicy::OnChange,
            },
            10_000_000_000,
        )
        .unwrap()
        .unwrap();
        let bars_path = bar_ring_path(&path);

        let minute = 60_000_000_000i64;
        let candle = |open_time: i64| {
            Event::Market(MarketEvent::Candle(Candle {
                symbol_id: SYM,
                interval: CandleInterval::M1,
                open: dec!(100),
                high: dec!(110),
                low: dec!(90),
                close: dec!(105),
                volume: dec!(7),
                open_time,
                ts_event: open_time + minute,
            }))
        };

        let mut h = handler().with_md_publisher(md);
        feed(&mut h, candle(0));
        feed(&mut h, candle(minute)); // the venue moved on: bar 0 is final
        assert_eq!(h.md().unwrap().stats().bars_published, 1);
        assert_eq!(
            h.md().unwrap().stats().published,
            0,
            "a candle is not a slice: it moves no field MdSlice carries"
        );
        h.md().unwrap().flush().unwrap();

        let c = axon_ipc::MdBarConsumer::open(&bars_path).unwrap();
        let bar = c.try_pop().unwrap();
        assert_eq!(bar.open_time, 0);
        assert_eq!(bar.ts_event, minute, "the close, never the open");
        assert_eq!(
            bar.close,
            axon_strategy::decimal_to_fixed(dec!(105)).unwrap()
        );
        drop(c);
        drop(h);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bars_path);
    }
}
