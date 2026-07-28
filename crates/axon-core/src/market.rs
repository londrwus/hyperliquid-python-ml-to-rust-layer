//! Normalized market-data vocabulary — the venue-agnostic value types the bus
//! carries. Adapters (e.g. `axon-provider-hyperliquid`) decode venue JSON into
//! these; `axon-marketdata` consumes them to maintain books and a cache.
//!
//! This lives in `axon-core` (not a provider crate) because it is *domain*
//! vocabulary, not a venue concern: `axon-providers` re-exports it so the
//! `MarketData` port still speaks these types. Prices and sizes are fixed-point
//! ([`Decimal`]), never `f64`, and every event is stamped with its own event time —
//! with one honest exception, [`Ticker`], for the venues that publish a mark price
//! without a timestamp on it.

use crate::clock::Nanos;
use crate::ids::SymbolId;
use crate::{Decimal, Side};
use serde::{Deserialize, Serialize};

/// Candle/bar interval (normalized across venues). Adapters map these to venue
/// wire strings (Hyperliquid: `"1m"`, `"1h"`, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CandleInterval {
    M1,
    M5,
    M15,
    H1,
    H4,
    D1,
}

/// One price level in an order book: a price and the aggregate size resting there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Level {
    pub px: Decimal,
    pub sz: Decimal,
}

impl Level {
    pub fn new(px: Decimal, sz: Decimal) -> Self {
        Self { px, sz }
    }
}

/// A full L2 book snapshot for one instrument (Hyperliquid's `l2Book` sends the
/// top-N levels as a snapshot each update, so "replace the book" is the correct
/// model). `bids` are highest-first, `asks` lowest-first, as the venue sends them;
/// consumers must not rely on that and should re-sort if they need a guarantee.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookSnapshot {
    pub symbol_id: SymbolId,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub ts_event: Nanos,
}

/// Best bid/offer snapshot. Fixed-point via [`Decimal`]; event-time stamped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bbo {
    pub symbol_id: SymbolId,
    pub bid_px: Decimal,
    pub bid_sz: Decimal,
    pub ask_px: Decimal,
    pub ask_sz: Decimal,
    pub ts_event: Nanos,
}

/// A public trade print.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trade {
    pub symbol_id: SymbolId,
    pub px: Decimal,
    pub sz: Decimal,
    /// Aggressor side.
    pub side: Side,
    pub ts_event: Nanos,
}

/// An OHLCV candle for one instrument at one [`CandleInterval`].
///
/// # `ts_event` is where this bar sorts, not a moment anyone observed
///
/// It is the bar's **close** time — `open_time + interval`, arithmetic on the bar's
/// identity rather than a time a venue reported. Two consequences, and they are the
/// whole contract of this type:
///
/// - **It is the same number on a bar's first frame and its last.** Venues republish
///   the bar they are still filling, so a `Candle` describing a minute that is three
///   seconds old carries the stamp of a close that is fifty-seven seconds away. 1 321
///   frames measured off Hyperliquid's socket described 69 bars; every frame for a bar
///   carried the same close time, and 1 317 of the 1 321 arrived before it
///   (`axon_provider_hyperliquid::ws::decode::decode_candle`).
/// - **So a `Candle`'s timestamp may be in the future, and nothing on this struct says
///   whether it is.** That is deliberate rather than an omission: the only venue this
///   workspace trades publishes no finality bit and sends no frame at or after the
///   close, so a `closed` field here could only ever be filled in by guessing, and a
///   guess that is wrong in the optimistic direction is a bar of look-ahead.
///
/// Finality is therefore **derived from the stream** by whoever needs it — a frame for a
/// later `open_time` is the venue moving on, which is the only evidence of closure that
/// does not require the venue to volunteer any. `axon_runtime::mdring` closes bars that
/// way and `axon.strategies.data.closed_rows` applies the same rule offline, so the live
/// and research paths agree on *which* bars exist. A consumer that cannot derive it must
/// not believe the stamp instead: `axon_runtime::handler::advances_the_clock` refuses a
/// candle as the core's clock for exactly this reason.
///
/// `open_time` is the bar's identity and is what feature windows and de-duplication key
/// on; `ts_event` exists so a closed bar sorts *after* the trades inside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candle {
    pub symbol_id: SymbolId,
    pub interval: CandleInterval,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    /// The bar's identity: when it opened. Stable across every republication of it,
    /// which is what makes it the key finality is derived from.
    pub open_time: Nanos,
    /// The bar's close time, `open_time + interval` — see the type's own docs. Not a
    /// receipt time and not a claim that the bar is finished.
    pub ts_event: Nanos,
}

/// A venue's periodic funding charge on a perpetual: the rate **and** the interval
/// it is charged over, which are kept together because either one alone is a wrong
/// number.
///
/// Venues disagree on the interval — Hyperliquid funds hourly, most CEX perps every
/// eight hours — and each publishes the rate for *its own* interval with no unit
/// attached. A bare `funding_rate` field therefore makes an expected-carry
/// calculation silently eight times wrong the day a second venue is added, and that
/// is the failure mode this pairing exists to prevent. The rate is stored exactly as
/// the venue published it, never rescaled to a common period, so accrued funding
/// still reconciles against the venue's own charges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Funding {
    /// Fraction of position notional charged once per [`interval`](Self::interval).
    /// Positive means longs pay shorts.
    pub rate: Decimal,
    /// How often `rate` is charged, in nanoseconds.
    pub interval: Nanos,
}

/// Per-instrument reference prices and perp statistics — the "ticker" a venue
/// publishes alongside, but independently of, the book (Hyperliquid:
/// `activeAssetCtx`). This is where the core's notion of a **mark price** comes from.
///
/// The field set follows the rule stated in [`exec`](crate::exec): a field earns its
/// place only if the core, the risk gate or a strategy genuinely needs it *and* any
/// venue could plausibly supply it. Two families of Hyperliquid field are excluded on
/// that basis, and the exclusions are the interesting part:
///
/// - **Rolling 24h summaries** (`prevDayPx`, `dayNtlVlm`). Each venue defines the
///   window differently — UTC midnight on one, trailing-24h on the next — so a
///   normalized field would invite comparisons between numbers that do not mean the
///   same thing. Anything a strategy wants from them is computable from [`Candle`]s
///   over an interval we name explicitly.
/// - **Derived quantities** (`premium`, `impactPxs`). Hyperliquid computes both from
///   venue-specific inputs (its impact-notional depth constant), so two venues'
///   `premium` are not the same statistic. Basis is `mark_px - index_px`, which a
///   consumer can compute and which means one thing everywhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ticker {
    pub symbol_id: SymbolId,
    /// The venue's mark price — what margin, liquidation and unrealized PnL are
    /// computed against, and therefore the price a pre-trade notional check has to
    /// use rather than the last trade or the mid.
    ///
    /// Deliberately not optional. A venue that cannot supply a mark must simply not
    /// emit a `Ticker`, because the one thing a consumer of this event may rely on is
    /// having a mark; an `Option` here would push that check to every call site and
    /// one of them would forget.
    pub mark_px: Decimal,
    /// The external spot reference the venue tracks — Hyperliquid names it the
    /// *oracle* price, CEXes name it the index; they are the same concept and the
    /// venue-neutral name is used here. `None` on instruments with no external
    /// reference.
    ///
    /// Kept because `mark_px - index_px` is the basis, which is what distinguishes a
    /// mark being dragged by a local squeeze from one tracking the underlying.
    pub index_px: Option<Decimal>,
    /// The venue's own mid price. **`None` when the book is one-sided** — genuinely
    /// absent, never zero and never back-filled from `mark_px`. Substituting the mark
    /// would report a tradeable price that nobody is quoting: a mark is an averaged
    /// reference, and a strategy that reads it as a mid will size a cross against a
    /// price the book cannot honour.
    pub mid_px: Option<Decimal>,
    /// Current funding, on instruments that fund at all. `None` for spot.
    pub funding: Option<Funding>,
    /// Open interest in **base** units — the same units as [`Trade::sz`], so it is
    /// directly comparable with sizes on this instrument. An adapter for a venue that
    /// publishes open interest in contracts or in quote notional must convert.
    ///
    /// It earns its place by being the only quantity here that no other feed can
    /// yield: books, trades and candles all describe *flow*, while open interest is
    /// the size still outstanding.
    pub open_interest: Option<Decimal>,
    /// The venue's own event time, when the venue stamps one on its ticker.
    ///
    /// `None` on Hyperliquid: `activeAssetCtx` is the one feed there that carries no
    /// timestamp at all — `bbo`, `trades` and `candle` all do. The absence is modeled
    /// explicitly instead of being back-filled from
    /// [`ts_ingest`](Self::ts_ingest) because the two are not interchangeable for the
    /// purpose that `ts_event` exists to serve. An event ordered on *our* receipt
    /// clock does not reproduce on replay, and a `Ticker` that looked venue-stamped
    /// like every other event would let the Phase-5 parity harness compare two runs
    /// whose interleaving was never comparable — a divergence that shows up as a
    /// mysterious PnL gap, not as a failure. Making the field `Option` means a
    /// consumer that cares has to look, and one that does not cannot be misled.
    pub ts_venue: Option<Nanos>,
    /// When this process received the update — always present, because the
    /// deterministic core needs an ordering key even when the venue supplies none.
    ///
    /// Where both times exist, `ts_ingest - ts_venue` is the feed latency
    /// (`docs/05-latency-model.md`).
    pub ts_ingest: Nanos,
}

impl Ticker {
    /// The ordering key: the venue's time when there is one, our receipt time when
    /// there is not.
    ///
    /// The fallback is decided here, once, rather than at each consumer — otherwise
    /// two handlers would order the same event differently and the bus would stop
    /// being a single total order.
    pub fn ts_event(&self) -> Nanos {
        self.ts_venue.unwrap_or(self.ts_ingest)
    }

    /// Whether this event was ordered on the venue's clock and therefore reproduces
    /// on replay. `false` means the ordering key came from a wall clock, so a capture
    /// must record [`ts_ingest`](Self::ts_ingest) alongside the frame or the replay
    /// will not match.
    pub fn is_venue_timed(&self) -> bool {
        self.ts_venue.is_some()
    }
}

/// A normalized market-data event. Wrapped by [`crate::event::Event`] to flow on
/// the bus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketEvent {
    Bbo(Bbo),
    Trade(Trade),
    Book(BookSnapshot),
    Candle(Candle),
    Ticker(Ticker),
}

impl MarketEvent {
    /// The event's own timestamp — the ordering key the deterministic core uses.
    pub fn ts_event(&self) -> Nanos {
        match self {
            MarketEvent::Bbo(b) => b.ts_event,
            MarketEvent::Trade(t) => t.ts_event,
            MarketEvent::Book(s) => s.ts_event,
            MarketEvent::Candle(c) => c.ts_event,
            // The one variant whose key may be a receipt time rather than the
            // venue's — see [`Ticker::ts_venue`].
            MarketEvent::Ticker(t) => t.ts_event(),
        }
    }

    /// The instrument this event concerns.
    pub fn symbol_id(&self) -> SymbolId {
        match self {
            MarketEvent::Bbo(b) => b.symbol_id,
            MarketEvent::Trade(t) => t.symbol_id,
            MarketEvent::Book(s) => s.symbol_id,
            MarketEvent::Candle(c) => c.symbol_id,
            MarketEvent::Ticker(t) => t.symbol_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn market_event_reports_ts_and_symbol() {
        let ev = MarketEvent::Trade(Trade {
            symbol_id: SymbolId::new(7),
            px: dec!(100.5),
            sz: dec!(2),
            side: Side::Buy,
            ts_event: 42,
        });
        assert_eq!(ev.ts_event(), 42);
        assert_eq!(ev.symbol_id(), SymbolId::new(7));
    }

    fn ticker(mid_px: Option<Decimal>) -> Ticker {
        Ticker {
            symbol_id: SymbolId::new(3),
            mark_px: dec!(50001.5),
            index_px: Some(dec!(50000)),
            mid_px,
            funding: Some(Funding {
                rate: dec!(0.0000125),
                interval: 3_600_000_000_000,
            }),
            open_interest: Some(dec!(688.11)),
            ts_venue: None,
            ts_ingest: 9_000,
        }
    }

    #[test]
    fn ticker_orders_and_routes_like_every_other_market_event() {
        // A new variant that forgets its `ts_event`/`symbol_id` arm would order at the
        // wrong point in the deterministic queue, which is a silent replay divergence
        // rather than a crash.
        let ev = MarketEvent::Ticker(ticker(Some(dec!(50000.5))));
        assert_eq!(ev.ts_event(), 9_000);
        assert_eq!(ev.symbol_id(), SymbolId::new(3));
    }

    #[test]
    fn an_ingest_stamped_ticker_cannot_pass_itself_off_as_venue_timed() {
        // The failure this guards: an adapter for a feed with no venue timestamp
        // writes receipt time into the ordering key, the event becomes
        // indistinguishable from a reproducible one, and a replay silently reorders
        // against a live capture. `ts_venue` stays absent so the difference survives
        // into every consumer, including serde.
        let t = ticker(Some(dec!(50000.5)));
        assert!(!t.is_venue_timed());
        assert_eq!(
            t.ts_event(),
            t.ts_ingest,
            "the receipt clock is the fallback"
        );
        let back: Ticker = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(back.ts_venue, None, "provenance survives a round trip");

        // A venue that does stamp its ticker wins over the receipt clock, and the gap
        // between the two is the feed latency.
        let stamped = Ticker {
            ts_venue: Some(8_800),
            ..t
        };
        assert!(stamped.is_venue_timed());
        assert_eq!(stamped.ts_event(), 8_800);
        assert_eq!(stamped.ts_ingest - 8_800, 200);
    }

    #[test]
    fn an_absent_mid_stays_absent_and_is_never_the_mark() {
        // A one-sided book has no mid. Round-tripping must not quietly turn that into
        // zero or into the mark price — a consumer reading the mark as a mid would
        // quote against a price the book cannot honour.
        let t = ticker(None);
        let j = serde_json::to_string(&t).unwrap();
        assert!(j.contains("\"mid_px\":null"), "absent, not defaulted: {j}");
        let back: Ticker = serde_json::from_str(&j).unwrap();
        assert_eq!(back.mid_px, None);
        assert_eq!(back.mark_px, dec!(50001.5), "the mark is untouched");
    }

    #[test]
    fn funding_cannot_be_stored_without_the_interval_it_accrues_over() {
        // The rate is kept exactly as the venue published it; the interval is what
        // makes it a number you can multiply by a holding period. Hyperliquid's hourly
        // rate read as an eight-hourly one is an 8x error in expected carry.
        let t = ticker(None);
        let f = t.funding.unwrap();
        assert_eq!(f.rate, dec!(0.0000125), "not rescaled to a common period");
        assert_eq!(f.interval, 3_600 * 1_000_000_000, "one hour in nanoseconds");
    }

    fn bar(close: Decimal, volume: Decimal) -> Candle {
        const MINUTE: Nanos = 60_000_000_000;
        Candle {
            symbol_id: SymbolId::new(0),
            interval: CandleInterval::M1,
            open: dec!(64340),
            high: dec!(64340),
            low: dec!(64340),
            close,
            volume,
            open_time: 1_785_053_340_000_000_000,
            // `open_time + interval`, which is what every adapter computes and what a
            // venue repeats on every republication of this bar.
            ts_event: 1_785_053_340_000_000_000 + MINUTE,
        }
    }

    #[test]
    fn two_states_of_one_bar_are_told_apart_by_open_time_and_never_by_the_stamp() {
        // The property everything downstream leans on. A venue republishes the bar it is
        // still filling — measured at 1 321 frames for 69 bars on Hyperliquid — and every
        // frame carries the same close time, because that stamp is `open_time +
        // interval` rather than anything the venue observed. So `ts_event` cannot
        // separate a partial from a close, and a consumer that tried would either
        // collapse a bar's whole history into one event or treat a mid-bar price as
        // final. `open_time` is the identity; finality is derived from a *later* one
        // arriving (`axon_runtime::mdring`, `axon.strategies.data.closed_rows`).
        let forming = bar(dec!(64340.0), dec!(0.00083));
        let full = bar(dec!(64352.0), dec!(1.07768));

        assert_eq!(forming.open_time, full.open_time, "one bar");
        assert_eq!(
            forming.ts_event, full.ts_event,
            "the stamp is arithmetic on the identity, so it moves with neither"
        );
        assert_ne!(forming, full, "…while the bar underneath them did move");
        assert_eq!(
            full.ts_event - full.open_time,
            60_000_000_000,
            "exactly one interval: nothing here is a receipt time"
        );

        // And it survives serde, because this type rides the capture log and the replay
        // path: a reader that recovered a different stamp would re-order the tape.
        let back: Candle = serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
        assert_eq!(back, full);
    }

    #[test]
    fn book_snapshot_serde_roundtrip() {
        let snap = BookSnapshot {
            symbol_id: SymbolId::new(1),
            bids: vec![
                Level::new(dec!(100), dec!(3)),
                Level::new(dec!(99), dec!(5)),
            ],
            asks: vec![Level::new(dec!(101), dec!(2))],
            ts_event: 1_000,
        };
        let j = serde_json::to_string(&snap).unwrap();
        let back: BookSnapshot = serde_json::from_str(&j).unwrap();
        assert_eq!(snap, back);
    }
}
