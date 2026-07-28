//! One answer to "what is the top of book", and the per-symbol clock that makes it
//! judgeable.
//!
//! Two consumers ask that question on every event and they must never get two answers:
//! [`crate::intent`] prices an order against it, and [`crate::mdring`] hands it to
//! Python as the input to a feature. Two implementations would drift — silently, each
//! passing its own tests — and the strategy would then be computing features on one
//! book while its orders were priced against another. That is the divergence ADR-0012
//! §1 refuses a second venue connection to avoid, reintroduced inside one process.
//!
//! Two decisions here, and each names a failure a live session has.
//!
//! **1. The `bbo` feed wins while it is fresh; the L2 book takes over when it is not.**
//! Precedence alone is what the planner used to do, and it is wrong in one specific and
//! reachable way: a reconnect that fails to re-subscribe `bbo` leaves that cache frozen
//! at its pre-disconnect value while `l2Book` keeps flowing, and every quote priced or
//! published from then on is the old one wearing the current event's timestamp. It is
//! also self-perpetuating — the same frozen quote plans the same order, which
//! `WorkingOrder::is` then matches, so the stale limit is deliberately left resting.
//! Ageing the `bbo` and falling back is what makes that outage self-correcting.
//! *Freshest-wins* was the obvious alternative and it is worse: the two feeds disagree
//! transiently about size, so a session subscribed to both would alternate between two
//! slightly different states and [`crate::mdring`]'s change test would stop coalescing
//! anything — `on_change` silently degraded into `every_update` for the configuration
//! almost everyone runs.
//!
//! **2. A quote is aged in event time, and stale collapses into missing.** A stale book
//! is well-formed: both sides present, not crossed, every check short of its age
//! passing. [`MarketDataProcessor::last_ts`] is *global* and cannot age one instrument:
//! a frozen `l2Book` on one symbol keeps looking current for as long as any other symbol
//! is trading, while a live `activeAssetCtx` on it keeps `MarkCache` fresh and vouches
//! for it. The per-symbol timestamp is [`MarketDataProcessor::book_ts`], kept beside the
//! levels it describes. This module briefly carried its own copy; one book with two
//! recorded ages is a disagreement waiting for the day the two are updated from
//! different places.

use axon_core::{Decimal, Nanos, SymbolId};
use axon_marketdata::MarketDataProcessor;

/// A two-sided top of book, with the event time it was true as of.
///
/// The timestamp is the quote's own, which is **not** the time of the event that asked
/// for it: a trade print arriving now says nothing about when the book behind it last
/// moved, and conflating the two is how a dead quote acquires a current timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopOfBook {
    pub bid_px: Decimal,
    pub bid_sz: Decimal,
    pub ask_px: Decimal,
    pub ask_sz: Decimal,
    pub ts_event: Nanos,
}

/// What the feeds have to say about an instrument right now.
///
/// Three cases and no fourth, because the fourth is the bug: "we have never seen this
/// instrument quoted" and "the quote we have stopped moving" call for different
/// responses — one is a configuration, the other is a fault — and collapsing them into
/// a bare `None` is how a dead feed comes to look like a quiet one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopState {
    /// A quote inside the window, from whichever feed still has one.
    Fresh(TopOfBook),
    /// A quote exists and every feed carrying one has gone quiet past the window.
    Stale,
    /// No feed has ever quoted this instrument two-sided.
    Unseen,
}

/// The top of book for `symbol` at event time `now`, or why there is not one.
///
/// `max_age_ns` is deliberately the *mark cache's* window: an instrument the risk gate
/// has already declared too stale to size a position in is not one the planner should
/// be quoting into or Python should be computing features on, and two windows would
/// mean two answers to one question.
pub fn top_of_book(
    market: &MarketDataProcessor,
    symbol: SymbolId,
    now: Nanos,
    max_age_ns: Nanos,
) -> TopState {
    let bbo = market.bbo(symbol).map(|b| TopOfBook {
        bid_px: b.bid_px,
        bid_sz: b.bid_sz,
        ask_px: b.ask_px,
        ask_sz: b.ask_sz,
        ts_event: b.ts_event,
    });
    // Both sides or neither. A one-sided book cannot be priced against and cannot be
    // published — the record's own sentinel convention reads a zero bid as "nothing
    // seen yet" — so half a quote is no quote here rather than a quote with a hole.
    let book = market
        .book_ts(symbol)
        .zip(market.book(symbol))
        .and_then(|(ts, b)| {
            let (bid_px, bid_sz) = b.best_bid()?;
            let (ask_px, ask_sz) = b.best_ask()?;
            Some(TopOfBook {
                bid_px,
                bid_sz,
                ask_px,
                ask_sz,
                ts_event: ts,
            })
        });

    // Precedence, then age: the `bbo` feed exists to answer exactly this question, so
    // it answers it for as long as it is still speaking.
    for candidate in [bbo, book].into_iter().flatten() {
        if now.saturating_sub(candidate.ts_event) <= max_age_ns {
            return TopState::Fresh(candidate);
        }
    }
    match bbo.is_some() || book.is_some() {
        true => TopState::Stale,
        false => TopState::Unseen,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Bbo, BookSnapshot, Event, EventHandler, Level, MarketEvent};
    use rust_decimal_macros::dec;

    const BTC: SymbolId = SymbolId::new(0);
    const SEC: Nanos = 1_000_000_000;
    const WINDOW: Nanos = 10 * SEC;

    #[derive(Default)]
    struct Feeds {
        market: MarketDataProcessor,
    }

    impl Feeds {
        /// Exactly the order `CoreHandler` applies an event in.
        fn feed(&mut self, ev: Event) {
            self.market.on_event(ev.ts_event(), &ev);
        }

        fn top(&self, now: Nanos) -> TopState {
            top_of_book(&self.market, BTC, now, WINDOW)
        }
    }

    fn bbo(bid: Decimal, ask: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Bbo(Bbo {
            symbol_id: BTC,
            bid_px: bid,
            bid_sz: dec!(2),
            ask_px: ask,
            ask_sz: dec!(3),
            ts_event: ts,
        }))
    }

    fn book(bid: Decimal, ask: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Book(BookSnapshot {
            symbol_id: BTC,
            bids: vec![Level::new(bid, dec!(10))],
            asks: vec![Level::new(ask, dec!(10))],
            ts_event: ts,
        }))
    }

    #[test]
    fn an_instrument_nobody_has_quoted_is_absent_rather_than_stale() {
        // The two need different responses: one is how the session was configured, the
        // other is a feed that died. Reported as one, a dead feed looks like a quiet one.
        let f = Feeds::default();
        assert_eq!(f.top(SEC), TopState::Unseen);
    }

    #[test]
    fn a_live_bbo_answers_even_while_a_book_is_also_flowing() {
        // The two feeds disagree about size transiently, so switching between them per
        // event would make every frame differ and stop the publisher coalescing anything.
        let mut f = Feeds::default();
        f.feed(bbo(dec!(100), dec!(101), SEC));
        f.feed(book(dec!(100), dec!(101), 2 * SEC));
        let TopState::Fresh(t) = f.top(2 * SEC) else {
            panic!("{:?}", f.top(2 * SEC))
        };
        assert_eq!(t.bid_sz, dec!(2), "the bbo's own size, not the book's");
        assert_eq!(t.ts_event, SEC);
    }

    #[test]
    fn a_frozen_bbo_hands_over_to_the_book_that_is_still_moving() {
        // The reachable production form: a reconnect re-subscribes `l2Book` and not
        // `bbo`. Without the handover the frozen quote is reported forever, under
        // whatever timestamp the newest event happens to carry.
        let mut f = Feeds::default();
        f.feed(bbo(dec!(100), dec!(101), SEC));
        f.feed(book(dec!(200), dec!(201), 20 * SEC));
        let TopState::Fresh(t) = f.top(20 * SEC) else {
            panic!("the live book must answer once the bbo has gone quiet")
        };
        assert_eq!(t.bid_px, dec!(200));
        assert_eq!(t.ts_event, 20 * SEC);
    }

    #[test]
    fn a_quote_every_feed_has_stopped_moving_reads_as_stale_not_as_a_price() {
        // A stale book is well formed: two sides, not crossed, every check but its age
        // passing. Handed over anyway it prices a limit into a market that has gone.
        let mut f = Feeds::default();
        f.feed(bbo(dec!(100), dec!(101), SEC));
        f.feed(book(dec!(100), dec!(101), SEC));
        assert!(
            matches!(f.top(SEC + WINDOW), TopState::Fresh(_)),
            "the edge"
        );
        assert_eq!(f.top(SEC + WINDOW + 1), TopState::Stale);
    }

    #[test]
    fn a_late_snapshot_dates_the_book_by_the_levels_it_actually_holds() {
        // `apply_book` replaces the book with whatever arrived last, so a high-water
        // mark here would leave ten-second-old levels wearing the newest event's time.
        let mut f = Feeds::default();
        f.feed(book(dec!(100), dec!(101), 20 * SEC));
        f.feed(book(dec!(90), dec!(91), SEC)); // a late frame
        assert_eq!(f.market.book_ts(BTC), Some(SEC));
        assert_eq!(f.top(20 * SEC), TopState::Stale);
    }

    #[test]
    fn a_one_sided_book_is_no_quote_rather_than_half_of_one() {
        // A zero bid is the record's sentinel for "nothing seen yet", so publishing one
        // side would be publishing a price nobody quoted.
        let mut f = Feeds::default();
        f.feed(Event::Market(MarketEvent::Book(BookSnapshot {
            symbol_id: BTC,
            bids: vec![Level::new(dec!(100), dec!(1))],
            asks: vec![],
            ts_event: SEC,
        })));
        assert_eq!(f.top(SEC), TopState::Unseen);
    }
}
