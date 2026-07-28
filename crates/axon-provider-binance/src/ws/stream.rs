//! Stream naming: normalized [`Feed`] → Binance stream name, and back.
//!
//! Binance has no subscription *object*. Where Hyperliquid sends
//! `{"method":"subscribe","subscription":{"type":"l2Book","coin":"BTC"}}`, Binance
//! sends `{"method":"SUBSCRIBE","params":["btcusdt@depth20@100ms"],"id":1}` — the
//! feed, the instrument and every parameter are concatenated into one lower-case
//! string. So this module is the whole of `sub.rs`, and it has a second job that
//! Hyperliquid's does not:
//!
//! **The stream name is the only thing that says what a frame means.**
//! `btcusdt@depth20@100ms` (a 20-level snapshot) and `btcusdt@depth@100ms` (an
//! incremental diff) both arrive as `"e":"depthUpdate"` with an identical field set —
//! `E`, `T`, `s`, `U`, `u`, `pu`, `b`, `a`. Both captured frames are in
//! [`super::decode`]'s fixtures and nothing inside either one distinguishes them. A
//! diff applied as a snapshot replaces the book with the two levels that happened to
//! change; a snapshot applied as a diff is merely slow. The first is silent and
//! total.
//!
//! That is why this adapter talks to the **combined** endpoint
//! (`/stream?streams=…`), whose envelope carries the stream name, rather than the raw
//! one (`/ws/<stream>`), whose frames are the bare payload. It costs one JSON level
//! of nesting and it makes the ambiguity unrepresentable.

use axon_providers::{CandleInterval, Feed};

/// How many levels a partial-depth stream carries.
///
/// The venue offers exactly these three and nothing between them, which is why it is
/// an enum: `depth15` is not a slower `depth20`, it is a stream that does not exist
/// and a subscription that silently yields nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookDepth {
    Levels5,
    Levels10,
    Levels20,
}

impl BookDepth {
    pub const fn levels(self) -> u32 {
        match self {
            BookDepth::Levels5 => 5,
            BookDepth::Levels10 => 10,
            BookDepth::Levels20 => 20,
        }
    }
}

/// How often a depth stream publishes.
///
/// `Ms250` is the venue's default when no speed is named, and naming it anyway is
/// deliberate: a stream name with no speed suffix is a *different string*, and the
/// day the default changes the subscription silently changes cadence with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateSpeed {
    Ms100,
    Ms250,
    Ms500,
}

impl UpdateSpeed {
    pub const fn wire(self) -> &'static str {
        match self {
            UpdateSpeed::Ms100 => "100ms",
            UpdateSpeed::Ms250 => "250ms",
            UpdateSpeed::Ms500 => "500ms",
        }
    }
}

/// What a stream name says a frame is.
///
/// Deliberately *not* derived from the payload's own `"e"` field. Two of these share
/// one event type, and the one that is dangerous to confuse is the one this enum
/// exists to separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    /// `<sym>@depth<N>[@speed]` — the top N levels, replaced wholesale. The only
    /// depth stream this adapter accepts.
    PartialDepth,
    /// `<sym>@depth[@speed]` — incremental diffs against a REST snapshot, keyed by
    /// `U`/`u`/`pu` sequence numbers. Refused: maintaining it needs a resync
    /// state machine, and reading one as a snapshot is a silently wrong book.
    DiffDepth,
    /// `<sym>@aggTrade` — trades aggregated by (price, side, taker order).
    AggTrade,
    /// `<sym>@bookTicker` — best bid/offer on every change.
    BookTicker,
    /// `<sym>@markPrice[@1s]` — mark, index, funding rate, next funding time.
    MarkPrice,
    /// `<sym>@kline_<interval>` — OHLCV, updated continuously and closed once.
    Kline,
}

/// Binance's interval token for a normalized [`CandleInterval`].
pub const fn kline_interval(i: CandleInterval) -> &'static str {
    match i {
        CandleInterval::M1 => "1m",
        CandleInterval::M5 => "5m",
        CandleInterval::M15 => "15m",
        CandleInterval::H1 => "1h",
        CandleInterval::H4 => "4h",
        CandleInterval::D1 => "1d",
    }
}

/// The normalized interval for a Binance token. `None` for the nine intervals the
/// venue offers and [`CandleInterval`] has no word for (`3m`, `30m`, `2h`, `6h`,
/// `8h`, `12h`, `3d`, `1w`, `1M`).
pub fn candle_interval(token: &str) -> Option<CandleInterval> {
    Some(match token {
        "1m" => CandleInterval::M1,
        "5m" => CandleInterval::M5,
        "15m" => CandleInterval::M15,
        "1h" => CandleInterval::H1,
        "4h" => CandleInterval::H4,
        "1d" => CandleInterval::D1,
        _ => return None,
    })
}

/// The stream name for `feed` on `symbol` (lower case, as the venue requires).
///
/// `depth` and `speed` only apply to [`Feed::L2Book`]; every other feed ignores them.
/// They are arguments rather than adapter state so that this stays a pure function —
/// the same reason ADR-0011 made `ts_ingest` a decoder parameter instead of a clock
/// read.
pub fn stream_name(feed: Feed, symbol: &str, depth: BookDepth, speed: UpdateSpeed) -> String {
    match feed {
        Feed::L2Book => format!("{symbol}@depth{}@{}", depth.levels(), speed.wire()),
        Feed::Trades => format!("{symbol}@aggTrade"),
        Feed::Bbo => format!("{symbol}@bookTicker"),
        Feed::Candles(i) => format!("{symbol}@kline_{}", kline_interval(i)),
        // The 1-second variant, not the bare `<sym>@markPrice`, which publishes every
        // three seconds. A mark that can be three seconds old is a risk gate sizing
        // against a price that has moved, and ADR-0011 already warns that any
        // staleness threshold has to sit above the feed's own floor.
        Feed::Ticker => format!("{symbol}@markPrice@1s"),
    }
}

/// What kind of frame a stream name produces, and the candle interval if it is one.
///
/// Parses the part **after the first `@`**, because the symbol is everything before
/// it and symbols contain no `@`. Returns `None` for a name this adapter never
/// subscribes to, which the decoder treats as a frame to ignore rather than as an
/// error — the venue can add a stream to a connection we never asked for it on
/// (it does not, today) and an unknown name must not tear down a healthy socket.
pub fn stream_kind(name: &str) -> Option<StreamKind> {
    let (_symbol, rest) = name.split_once('@')?;
    // `markPrice@1s` and `depth20@100ms` both carry a second `@`; the kind is decided
    // by the first token and the speed suffix is not part of it.
    let token = rest.split('@').next().unwrap_or(rest);
    if let Some(levels) = token.strip_prefix("depth") {
        return Some(if levels.is_empty() {
            // `depth@100ms` — the diff stream. The distinction the payload does not
            // make, made here.
            StreamKind::DiffDepth
        } else if levels.bytes().all(|b| b.is_ascii_digit()) {
            StreamKind::PartialDepth
        } else {
            return None;
        });
    }
    Some(match token {
        "aggTrade" => StreamKind::AggTrade,
        "bookTicker" => StreamKind::BookTicker,
        "markPrice" => StreamKind::MarkPrice,
        t if t.starts_with("kline_") => StreamKind::Kline,
        _ => return None,
    })
}

/// The candle interval a `kline_*` stream name carries.
pub fn stream_kline_interval(name: &str) -> Option<CandleInterval> {
    let (_symbol, rest) = name.split_once('@')?;
    candle_interval(rest.strip_prefix("kline_")?)
}

/// A `SUBSCRIBE` request frame for `streams`.
///
/// `id` is the venue's request correlation id, echoed on the `{"result":null,"id":n}`
/// acknowledgement. It is required — a request without one is refused — and it is not
/// a nonce: it correlates, it does not authorize.
pub fn subscribe_msg(streams: &[String], id: u64) -> String {
    serde_json::json!({ "method": "SUBSCRIBE", "params": streams, "id": id }).to_string()
}

/// An `UNSUBSCRIBE` request frame.
pub fn unsubscribe_msg(streams: &[String], id: u64) -> String {
    serde_json::json!({ "method": "UNSUBSCRIBE", "params": streams, "id": id }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_depth_stream_and_a_diff_stream_are_told_apart_by_name_because_nothing_else_can() {
        // The frames are indistinguishable: both are `"e":"depthUpdate"` with the same
        // eight fields. Applying a diff as a snapshot replaces the whole book with the
        // one or two levels that changed — a book that is wrong, complete-looking, and
        // updated a hundred times a second.
        assert_eq!(
            stream_kind("btcusdt@depth20@100ms"),
            Some(StreamKind::PartialDepth)
        );
        assert_eq!(
            stream_kind("btcusdt@depth@100ms"),
            Some(StreamKind::DiffDepth)
        );
        // Both spellings of each, with and without the speed suffix.
        assert_eq!(
            stream_kind("btcusdt@depth5"),
            Some(StreamKind::PartialDepth)
        );
        assert_eq!(stream_kind("btcusdt@depth"), Some(StreamKind::DiffDepth));
    }

    #[test]
    fn the_kind_of_a_stream_is_the_first_token_and_not_the_speed_after_it() {
        assert_eq!(
            stream_kind("btcusdt@markPrice@1s"),
            Some(StreamKind::MarkPrice)
        );
        assert_eq!(
            stream_kind("btcusdt@markPrice"),
            Some(StreamKind::MarkPrice)
        );
        assert_eq!(stream_kind("btcusdt@aggTrade"), Some(StreamKind::AggTrade));
        assert_eq!(
            stream_kind("ethusdt@bookTicker"),
            Some(StreamKind::BookTicker)
        );
        assert_eq!(stream_kind("btcusdt@kline_1m"), Some(StreamKind::Kline));
        assert_eq!(stream_kind("btcusdt@forceOrder"), None);
        assert_eq!(stream_kind("btcusdt"), None, "no feed at all");
        assert_eq!(stream_kind("btcusdt@depthX"), None, "not a level count");
    }

    #[test]
    fn every_feed_the_port_defines_names_a_stream_that_exists() {
        // The port's `Feed` is closed, so this is a total mapping and a new variant
        // upstream would not compile here — which is the point of `match`ing it
        // exhaustively rather than defaulting.
        let cases = [
            (Feed::L2Book, "btcusdt@depth20@100ms"),
            (Feed::Trades, "btcusdt@aggTrade"),
            (Feed::Bbo, "btcusdt@bookTicker"),
            (Feed::Candles(CandleInterval::M1), "btcusdt@kline_1m"),
            (Feed::Ticker, "btcusdt@markPrice@1s"),
        ];
        for (feed, expected) in cases {
            let name = stream_name(feed, "btcusdt", BookDepth::Levels20, UpdateSpeed::Ms100);
            assert_eq!(name, expected);
            assert!(
                stream_kind(&name).is_some(),
                "{name} is a name this adapter cannot decode"
            );
        }
    }

    #[test]
    fn a_book_subscription_never_names_the_diff_stream_however_it_is_configured() {
        // `stream_name` is the only builder, so this is the property that keeps the
        // decoder's refusal from ever firing in production: a `Feed::L2Book` always
        // carries a level count and therefore is always a snapshot.
        for depth in [BookDepth::Levels5, BookDepth::Levels10, BookDepth::Levels20] {
            for speed in [UpdateSpeed::Ms100, UpdateSpeed::Ms250, UpdateSpeed::Ms500] {
                let name = stream_name(Feed::L2Book, "ethusdt", depth, speed);
                assert_eq!(
                    stream_kind(&name),
                    Some(StreamKind::PartialDepth),
                    "{name} is not a snapshot stream"
                );
            }
        }
    }

    #[test]
    fn a_candle_interval_round_trips_and_the_venues_extra_ones_are_refused() {
        for i in [
            CandleInterval::M1,
            CandleInterval::M5,
            CandleInterval::M15,
            CandleInterval::H1,
            CandleInterval::H4,
            CandleInterval::D1,
        ] {
            assert_eq!(candle_interval(kline_interval(i)), Some(i));
            let name = stream_name(
                Feed::Candles(i),
                "btcusdt",
                BookDepth::Levels20,
                UpdateSpeed::Ms100,
            );
            assert_eq!(stream_kline_interval(&name), Some(i));
        }
        // The venue offers nine more. Mapping `30m` onto the nearest word we have
        // would hand a strategy bars of a length it did not ask for.
        for extra in ["3m", "30m", "2h", "6h", "8h", "12h", "3d", "1w", "1M"] {
            assert_eq!(candle_interval(extra), None, "{extra} must not be mapped");
        }
    }

    #[test]
    fn a_subscribe_frame_carries_every_stream_in_one_request_with_a_correlation_id() {
        let streams = vec!["btcusdt@aggTrade".to_string(), "btcusdt@bookTicker".into()];
        let v: serde_json::Value = serde_json::from_str(&subscribe_msg(&streams, 7)).unwrap();
        assert_eq!(v["method"], "SUBSCRIBE");
        assert_eq!(v["params"][0], "btcusdt@aggTrade");
        assert_eq!(v["params"][1], "btcusdt@bookTicker");
        assert_eq!(v["id"], 7);
        let u: serde_json::Value = serde_json::from_str(&unsubscribe_msg(&streams, 8)).unwrap();
        assert_eq!(u["method"], "UNSUBSCRIBE");
        assert_eq!(u["id"], 8);
    }
}
