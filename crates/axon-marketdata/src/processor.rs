//! The market-data processor: the core-side consumer that turns normalized
//! [`MarketEvent`]s into maintained state.
//!
//! It is an [`EventHandler`], so the deterministic
//! [`event_loop`](axon_core::event_loop) drives it one event at a time. It owns
//! the per-symbol [`OrderBook`]s and a small cache of the latest BBO, trade, candle
//! and [`Ticker`], and exposes read-only queries the strategy/risk layers build on.
//! No I/O, no locks — plain single-threaded state.
//!
//! Every cached value keeps the event time it was true as of, because the risk layer's
//! question is never "what is the mark?" but "what is the mark, and is it recent
//! enough to act on?" — see [`MarketDataProcessor::mark_px`].

use std::collections::HashMap;

use axon_core::{
    Bbo, Candle, Decimal, Event, EventHandler, MarketEvent, Nanos, SymbolId, Ticker, Trade,
};

use crate::OrderBook;

/// Maintains order books + a latest-value cache from the normalized event stream.
#[derive(Debug, Default)]
pub struct MarketDataProcessor {
    books: HashMap<SymbolId, OrderBook>,
    /// Event time of the snapshot each symbol's book currently holds.
    ///
    /// Kept beside the levels rather than derived from `last_ts`, which is *global*:
    /// a frozen `l2Book` on one instrument is invisible from a global clock that every
    /// other instrument keeps advancing, and the quote is then handed out as current.
    /// Anyone judging a book's age needs this number, and a consumer that maintained
    /// its own copy would be a second structure describing one book — which is how the
    /// copies come to disagree about which one is stale.
    book_ts: HashMap<SymbolId, Nanos>,
    bbo: HashMap<SymbolId, Bbo>,
    last_trade: HashMap<SymbolId, Trade>,
    last_candle: HashMap<SymbolId, Candle>,
    last_ticker: HashMap<SymbolId, Ticker>,
    /// Event-time **high-water mark**: the newest event time applied, not the last one
    /// (0 before any). A maximum rather than an assignment because a `userFills` snapshot
    /// replayed on reconnect carries execution times hours old — 3.62 hours was the deepest
    /// backward step measured on a real soak tape — and a high-water mark that goes down is
    /// not one. `CoreHandler::last_ts` holds the same rule; two clocks called `last_ts` that
    /// answered differently is how they come to disagree.
    last_ts: Nanos,
}

impl MarketDataProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace a symbol's book from an L2 snapshot (Hyperliquid `l2Book` semantics).
    fn apply_book(&mut self, snap: &axon_core::BookSnapshot) {
        let mut book = OrderBook::new(snap.symbol_id);
        for lvl in &snap.bids {
            book.set_bid(lvl.px, lvl.sz);
        }
        for lvl in &snap.asks {
            book.set_ask(lvl.px, lvl.sz);
        }
        self.books.insert(snap.symbol_id, book);
        // Assigned, not kept as a high-water mark: the book *is* whichever snapshot
        // arrived last, so the age recorded has to be the age of the levels held. A
        // high-water mark would leave a ten-second-old book wearing the newest
        // arrival's timestamp, which is a stale quote that reads as fresh.
        self.book_ts.insert(snap.symbol_id, snap.ts_event);
    }

    /// The maintained order book for `symbol`, if one has been seen.
    pub fn book(&self, symbol: SymbolId) -> Option<&OrderBook> {
        self.books.get(&symbol)
    }

    /// Event time of the snapshot `symbol`'s book currently holds, or `None` if it has
    /// never been quoted.
    ///
    /// The only honest way to age a book. [`last_ts`](Self::last_ts) is global and says
    /// nothing about one instrument: a symbol whose `l2Book` stopped arriving keeps
    /// looking current for as long as any *other* symbol is trading.
    pub fn book_ts(&self, symbol: SymbolId) -> Option<Nanos> {
        self.book_ts.get(&symbol).copied()
    }

    /// The latest best bid/offer for `symbol` (from the `bbo` feed), if any.
    pub fn bbo(&self, symbol: SymbolId) -> Option<&Bbo> {
        self.bbo.get(&symbol)
    }

    /// The latest public trade print for `symbol`, if any.
    pub fn last_trade(&self, symbol: SymbolId) -> Option<&Trade> {
        self.last_trade.get(&symbol)
    }

    /// The latest candle for `symbol`, if any. (Keyed by symbol; subscribing to
    /// more than one interval for a symbol keeps only the most recent.)
    pub fn last_candle(&self, symbol: SymbolId) -> Option<&Candle> {
        self.last_candle.get(&symbol)
    }

    /// The latest [`Ticker`] for `symbol`, if any — mark, index, funding and open
    /// interest, together with the times they were true as of.
    pub fn ticker(&self, symbol: SymbolId) -> Option<&Ticker> {
        self.last_ticker.get(&symbol)
    }

    /// The latest mark price for `symbol` **paired with the event time it was true
    /// as of**.
    ///
    /// The pairing is the point. A *stale* mark is more dangerous than a missing one:
    /// a missing mark makes the risk gate fail closed, which costs one trade, while a
    /// mark from five minutes ago sails through the same gate and sizes a position
    /// against a price that no longer exists. Returning a bare `Decimal` would let
    /// every call site forget that, so the timestamp travels with the price and the
    /// caller decides what "too old" means — only it knows its own horizon.
    ///
    /// The time is [`Ticker::ts_event`], which for a venue that does not stamp its
    /// ticker (Hyperliquid does not) is our *receipt* time. That is the right clock
    /// for a staleness test anyway — it is the age of our knowledge, which is what a
    /// caller is actually asking about — but it means a gap here can mean either "the
    /// venue went quiet" or "our socket did", and only the latter is visible in
    /// [`Ticker::is_venue_timed`].
    pub fn mark_px(&self, symbol: SymbolId) -> Option<(Decimal, Nanos)> {
        self.last_ticker
            .get(&symbol)
            .map(|t| (t.mark_px, t.ts_event()))
    }

    /// Mid price for `symbol`: from the L2 book if it has both sides, else from
    /// the latest BBO. `None` if neither is populated.
    pub fn mid(&self, symbol: SymbolId) -> Option<Decimal> {
        if let Some(m) = self.books.get(&symbol).and_then(|b| b.mid()) {
            return Some(m);
        }
        self.bbo
            .get(&symbol)
            .map(|b| (b.bid_px + b.ask_px) / Decimal::from(2))
    }

    /// Event time of the last event applied.
    pub fn last_ts(&self) -> Nanos {
        self.last_ts
    }

    /// How many distinct symbols have a book.
    pub fn tracked_symbols(&self) -> usize {
        self.books.len()
    }
}

impl EventHandler for MarketDataProcessor {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        self.last_ts = self.last_ts.max(ts_event);
        // Execution events share the bus so they order against market data by event
        // time, but they are the order tracker's business, not the book's.
        let Event::Market(m) = event else { return };
        match m {
            MarketEvent::Book(snap) => self.apply_book(snap),
            MarketEvent::Bbo(b) => {
                self.bbo.insert(b.symbol_id, b.clone());
            }
            MarketEvent::Trade(t) => {
                self.last_trade.insert(t.symbol_id, t.clone());
            }
            MarketEvent::Candle(c) => {
                self.last_candle.insert(c.symbol_id, c.clone());
            }
            MarketEvent::Ticker(t) => {
                self.last_ticker.insert(t.symbol_id, t.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{BookSnapshot, CandleInterval, Funding, Level, Side};
    use rust_decimal_macros::dec;

    fn sym() -> SymbolId {
        SymbolId::new(1)
    }

    fn feed(p: &mut MarketDataProcessor, ev: MarketEvent) {
        let e = Event::Market(ev);
        p.on_event(e.ts_event(), &e);
    }

    #[test]
    fn exec_events_are_ignored_by_the_book() {
        // The two kinds share one bus so they order by event time, but a fill is the
        // order tracker's business. The book must skip it without panicking and
        // without disturbing its own state.
        use axon_core::{AccountSnapshot, ExecEvent};
        let mut p = MarketDataProcessor::new();
        feed(
            &mut p,
            MarketEvent::Bbo(axon_core::Bbo {
                symbol_id: sym(),
                bid_px: dec!(99),
                bid_sz: dec!(1),
                ask_px: dec!(101),
                ask_sz: dec!(1),
                ts_event: 10,
            }),
        );
        let before = p.tracked_symbols();

        let e = Event::Exec(ExecEvent::Account(AccountSnapshot {
            equity: dec!(1000),
            withdrawable: dec!(1000),
            margin_used: dec!(0),
            ts_event: 20,
        }));
        p.on_event(e.ts_event(), &e);

        assert_eq!(p.tracked_symbols(), before);
        assert_eq!(p.bbo(sym()).unwrap().bid_px, dec!(99), "book untouched");
    }

    fn book_at(ts: Nanos, bid: Decimal, ask: Decimal) -> MarketEvent {
        MarketEvent::Book(BookSnapshot {
            symbol_id: sym(),
            bids: vec![Level::new(bid, dec!(1))],
            asks: vec![Level::new(ask, dec!(1))],
            ts_event: ts,
        })
    }

    #[test]
    fn a_book_that_stopped_arriving_is_not_aged_by_another_symbols_traffic() {
        // `last_ts` is global: it advances on every event from every instrument. Aging a
        // book with it means an `l2Book` subscription that a reconnect failed to restore
        // stays "current" for as long as anything else is trading — and the planner
        // prices a limit into a market that has since moved.
        let mut p = MarketDataProcessor::new();
        feed(&mut p, book_at(10, dec!(100), dec!(101)));
        let other = SymbolId::new(2);
        feed(
            &mut p,
            MarketEvent::Trade(Trade {
                symbol_id: other,
                px: dec!(7),
                sz: dec!(1),
                side: axon_core::Side::Buy,
                ts_event: 5_000,
            }),
        );
        assert_eq!(p.last_ts(), 5_000, "the global clock did move");
        assert_eq!(p.book_ts(sym()), Some(10), "this book did not");
        assert_eq!(p.book_ts(other), None, "and that one has no book at all");
    }

    #[test]
    fn a_late_snapshot_dates_the_book_by_the_levels_it_holds_not_by_its_arrival() {
        // `apply_book` replaces the book with whatever arrived last, so a high-water
        // mark here would leave ten-second-old levels wearing the newest event's time —
        // a stale quote that every downstream age check reads as fresh.
        let mut p = MarketDataProcessor::new();
        feed(&mut p, book_at(20_000, dec!(100), dec!(101)));
        feed(&mut p, book_at(1_000, dec!(90), dec!(91))); // a late frame
        assert_eq!(p.book_ts(sym()), Some(1_000));
        assert_eq!(p.book(sym()).unwrap().best_bid(), Some((dec!(90), dec!(1))));
    }

    #[test]
    fn applies_book_snapshot_and_reports_mid() {
        let mut p = MarketDataProcessor::new();
        feed(
            &mut p,
            MarketEvent::Book(BookSnapshot {
                symbol_id: sym(),
                bids: vec![
                    Level::new(dec!(100), dec!(2)),
                    Level::new(dec!(99), dec!(5)),
                ],
                asks: vec![
                    Level::new(dec!(101), dec!(3)),
                    Level::new(dec!(102), dec!(1)),
                ],
                ts_event: 10,
            }),
        );
        let book = p.book(sym()).unwrap();
        assert_eq!(book.best_bid(), Some((dec!(100), dec!(2))));
        assert_eq!(book.best_ask(), Some((dec!(101), dec!(3))));
        assert_eq!(p.mid(sym()), Some(dec!(100.5)));
        assert_eq!(p.last_ts(), 10);
        assert_eq!(p.tracked_symbols(), 1);
    }

    #[test]
    fn snapshot_replaces_previous_book() {
        let mut p = MarketDataProcessor::new();
        feed(
            &mut p,
            MarketEvent::Book(BookSnapshot {
                symbol_id: sym(),
                bids: vec![Level::new(dec!(100), dec!(2))],
                asks: vec![Level::new(dec!(101), dec!(3))],
                ts_event: 1,
            }),
        );
        // A later snapshot with different levels fully replaces the old book.
        feed(
            &mut p,
            MarketEvent::Book(BookSnapshot {
                symbol_id: sym(),
                bids: vec![Level::new(dec!(90), dec!(4))],
                asks: vec![Level::new(dec!(95), dec!(1))],
                ts_event: 2,
            }),
        );
        let book = p.book(sym()).unwrap();
        assert_eq!(book.best_bid(), Some((dec!(90), dec!(4))));
        assert_eq!(book.best_ask(), Some((dec!(95), dec!(1))));
        assert_eq!(book.bid_levels(), 1); // old 100 level is gone
    }

    #[test]
    fn caches_bbo_and_last_trade_and_falls_back_to_bbo_mid() {
        let mut p = MarketDataProcessor::new();
        feed(
            &mut p,
            MarketEvent::Bbo(Bbo {
                symbol_id: sym(),
                bid_px: dec!(100),
                bid_sz: dec!(1),
                ask_px: dec!(102),
                ask_sz: dec!(1),
                ts_event: 5,
            }),
        );
        feed(
            &mut p,
            MarketEvent::Trade(Trade {
                symbol_id: sym(),
                px: dec!(101),
                sz: dec!(3),
                side: Side::Sell,
                ts_event: 6,
            }),
        );
        // No book yet → mid falls back to the BBO midpoint.
        assert_eq!(p.mid(sym()), Some(dec!(101)));
        assert_eq!(p.bbo(sym()).unwrap().ask_px, dec!(102));
        assert_eq!(p.last_trade(sym()).unwrap().side, Side::Sell);
    }

    #[test]
    fn unknown_symbol_has_no_state() {
        let p = MarketDataProcessor::new();
        assert!(p.book(sym()).is_none());
        assert!(p.mid(sym()).is_none());
        assert!(p.last_trade(sym()).is_none());
        assert!(p.last_candle(sym()).is_none());
        assert!(p.ticker(sym()).is_none());
        assert!(
            p.mark_px(sym()).is_none(),
            "no mark must read as absent, never as zero"
        );
    }

    #[test]
    fn caches_last_candle() {
        let mut p = MarketDataProcessor::new();
        feed(
            &mut p,
            MarketEvent::Candle(Candle {
                symbol_id: sym(),
                interval: CandleInterval::M1,
                open: dec!(100),
                high: dec!(105),
                low: dec!(99),
                close: dec!(104),
                volume: dec!(12.5),
                open_time: 60_000,
                ts_event: 119_999,
            }),
        );
        let c = p.last_candle(sym()).unwrap();
        assert_eq!(c.close, dec!(104));
        assert_eq!(c.interval, CandleInterval::M1);
        assert_eq!(p.last_ts(), 119_999);
    }

    fn ticker(mark: Decimal, ts: Nanos) -> Ticker {
        Ticker {
            symbol_id: sym(),
            mark_px: mark,
            index_px: Some(dec!(50000)),
            mid_px: Some(dec!(50000.5)),
            funding: Some(Funding {
                rate: dec!(0.0000125),
                interval: 3_600_000_000_000,
            }),
            open_interest: Some(dec!(688.11)),
            // Hyperliquid's activeAssetCtx carries no venue time, so the receipt
            // clock is the ordering key — the case the cache must handle.
            ts_venue: None,
            ts_ingest: ts,
        }
    }

    #[test]
    fn caches_the_latest_ticker_and_replaces_the_previous_one() {
        let mut p = MarketDataProcessor::new();
        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50001.5), 1_000)));
        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50010), 2_000)));
        let t = p.ticker(sym()).unwrap();
        assert_eq!(t.mark_px, dec!(50010));
        assert_eq!(t.open_interest, Some(dec!(688.11)));
        assert_eq!(t.funding.unwrap().rate, dec!(0.0000125));
        assert_eq!(p.last_ts(), 2_000);
    }

    #[test]
    fn a_late_arrival_never_drags_the_global_clock_back_to_when_it_happened() {
        // Hyperliquid replays its whole `userFills` snapshot on every reconnect, carrying
        // execution times hours stale — 3.62 hours was the deepest backward step measured
        // on a real soak tape. An assignment here would pin the clock to that historical
        // instant, so anything ageing against it reads a session that is hours behind while
        // its feeds are perfectly current. `last_ts` is a high-water mark; a high-water mark
        // that goes down is not one.
        let mut p = MarketDataProcessor::new();
        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50010), 100_000)));
        assert_eq!(p.last_ts(), 100_000);

        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50001.5), 1_000)));
        assert_eq!(
            p.last_ts(),
            100_000,
            "a late arrival is still applied, but it is not evidence of a newer 'now'"
        );
        // The cache still took the late event: ordering is the log's business, not ours.
        assert_eq!(p.ticker(sym()).unwrap().mark_px, dec!(50001.5));
    }

    #[test]
    fn a_mark_price_is_never_handed_out_without_its_event_time() {
        // A stale mark passes a risk check that a missing one would fail closed on, so
        // the accessor must make the age impossible to drop on the floor: the price
        // and the time it was true as of come out together or not at all.
        let mut p = MarketDataProcessor::new();
        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50001.5), 7_000)));
        assert_eq!(p.mark_px(sym()), Some((dec!(50001.5), 7_000)));

        // A venue that does stamp its ticker must have that time reported, not our
        // receipt time — otherwise a staleness test on a venue-timed feed would
        // measure the wrong clock and never see the venue itself lagging.
        let stamped = Ticker {
            ts_venue: Some(6_500),
            ..ticker(dec!(50002), 7_400)
        };
        feed(&mut p, MarketEvent::Ticker(stamped));
        assert_eq!(p.mark_px(sym()), Some((dec!(50002), 6_500)));
    }

    #[test]
    fn a_ticker_does_not_disturb_the_book_or_the_mid() {
        // The ticker carries the venue's own mid, but `mid()` is a book-derived
        // quantity. Silently answering it from the ticker would mix two feeds with
        // different cadences behind one accessor.
        let mut p = MarketDataProcessor::new();
        feed(&mut p, MarketEvent::Ticker(ticker(dec!(50001.5), 1_000)));
        assert!(p.book(sym()).is_none());
        assert_eq!(p.mid(sym()), None);
        assert_eq!(p.tracked_symbols(), 0);
    }
}
