//! Pure decoders: Binance USD-M combined-stream JSON → normalized [`Event`]s.
//!
//! The offline-testable heart of the adapter, and the direct analogue of
//! `axon_provider_hyperliquid::ws::decode`. Every venue quirk is absorbed here:
//! prices and sizes arrive as **strings** and become fixed-point [`Decimal`] (never
//! `f64`), milliseconds become event-time nanoseconds, and symbol strings become
//! [`SymbolId`] via the
//! [`SymbolTable`]. These functions do no I/O and are pinned to frames captured off
//! the venue's own socket.
//!
//! Four things here are decisions rather than translation, and each one is a place
//! the second venue behaved differently from the first:
//!
//! 1. **Which of the two timestamps is the event time.** Every futures frame carries
//!    both `E` (when the venue pushed it) and `T` (when the thing happened). They are
//!    not the same — the captured `aggTrade` has them 154 ms apart — and `T` is the
//!    one the core must order on. `E` is a push time, which is the venue's half of
//!    `ts_ingest`, not an event time.
//! 2. **The aggressor side is stated backwards.** `"m": true` means *the buyer was
//!    the maker*, so the aggressor was the seller. Reading `m` as "was a buy" inverts
//!    order flow on every print, and every downstream statistic built on it, without
//!    a single malformed frame.
//! 3. **A mark-price frame is venue-timed.** ADR-0011 made
//!    [`Ticker::ts_venue`](axon_core::Ticker) an `Option` because Hyperliquid's
//!    `activeAssetCtx` carries no timestamp. Binance's `markPriceUpdate` carries `E`,
//!    so this adapter fills it in — the first time that `Option` has been `Some`, and
//!    the reason it was worth being an `Option` rather than a doc comment.
//! 4. **An unclosed candle is not emitted at all.** See `decode_kline` below; the
//!    reason is that [`Candle`] has no way to say "not final".

use std::str::FromStr;

use axon_core::{
    Bbo, BookSnapshot, Candle, Decimal, Event, Funding, Level, MarketEvent, Nanos, Side, SymbolId,
    Ticker, Trade,
};
use serde::Deserialize;

use super::stream::{stream_kind, stream_kline_interval, StreamKind};
use crate::symbols::SymbolTable;
use crate::MS_TO_NS;

/// Why a frame failed to decode.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown symbol: {0}")]
    UnknownSymbol(String),
    #[error("bad number for {field}: {value:?}")]
    BadNumber { field: &'static str, value: String },
    #[error("malformed {0} message")]
    Malformed(&'static str),
    /// A stream this adapter recognizes and refuses, as distinct from one it has
    /// never heard of (those are ignored). It exists for the incremental depth
    /// stream: `<sym>@depth@100ms` is byte-compatible with `<sym>@depth20@100ms` and
    /// means something completely different, so silence here would be a book that is
    /// wrong, complete-looking, and refreshed ten times a second.
    #[error("unsupported stream: {0}")]
    UnsupportedStream(&'static str),
    /// A payload with no `stream` wrapper around it.
    ///
    /// The raw endpoint (`/ws/<name>`) delivers the bare payload, and a bare
    /// `depthUpdate` cannot be told from a diff. Treating it as a control frame would
    /// mean a connection to the wrong endpoint yields no events and no errors — a
    /// session that looks healthy and is deaf.
    #[error("frame carries no stream name; connect to /stream?streams=, not /ws/")]
    Unenveloped,
    /// A `markPriceUpdate` with an empty funding rate — a **delivery** contract, not a
    /// perpetual.
    ///
    /// The same second line of defence ADR-0011 §5 builds for spot: the universe
    /// decode already refuses dated contracts by `contractType`, and this refuses one
    /// again at the frame, so a subscription assembled by hand cannot produce a
    /// `Ticker` whose funding field describes a charge the instrument does not levy.
    #[error("{symbol} sends no funding rate; it is a dated contract, not a perpetual")]
    NotAPerpetual { symbol: String },
}

type Result<T> = std::result::Result<T, DecodeError>;

// ── on-wire shapes (undocumented extras like `st`, `ps`, `nq` are ignored) ────

/// The combined-stream envelope. `stream` is what makes a `depthUpdate` legible.
#[derive(Deserialize)]
struct Envelope {
    stream: String,
    data: serde_json::Value,
}

/// A `[price, size]` pair. A fixed-length array rather than a `Vec`, so a level with
/// a third element (which spot's REST book carries and futures' does not) fails the
/// decode instead of being silently truncated to the two fields we happen to read.
type RawLevel = [String; 2];

#[derive(Deserialize)]
struct DepthData {
    #[serde(rename = "s")]
    symbol: String,
    /// Transaction time: when the matching engine held this book state.
    #[serde(rename = "T")]
    transact_time: i64,
    #[serde(rename = "b")]
    bids: Vec<RawLevel>,
    #[serde(rename = "a")]
    asks: Vec<RawLevel>,
}

#[derive(Deserialize)]
struct AggTradeData {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p")]
    px: String,
    #[serde(rename = "q")]
    qty: String,
    /// Trade time. Not `E`, which is when the venue pushed the frame.
    #[serde(rename = "T")]
    trade_time: i64,
    /// **Was the buyer the maker.** `true` ⇒ the aggressor sold.
    #[serde(rename = "m")]
    buyer_is_maker: bool,
}

#[derive(Deserialize)]
struct BookTickerData {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "b")]
    bid_px: String,
    #[serde(rename = "B")]
    bid_sz: String,
    #[serde(rename = "a")]
    ask_px: String,
    #[serde(rename = "A")]
    ask_sz: String,
    #[serde(rename = "T")]
    transact_time: i64,
}

#[derive(Deserialize)]
struct MarkPriceData {
    #[serde(rename = "s")]
    symbol: String,
    /// The venue's own event time — present here and absent on Hyperliquid's
    /// equivalent, which is the whole reason `Ticker::ts_venue` is an `Option`.
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "p")]
    mark_px: String,
    /// Index price. Empty on contracts with no external reference, which is why it is
    /// an `Option` after the empty-string check rather than a required field.
    #[serde(rename = "i", default)]
    index_px: Option<String>,
    /// Funding rate for the period. Empty string on a delivery contract.
    #[serde(rename = "r")]
    funding_rate: String,
}

#[derive(Deserialize)]
struct KlineBody {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "v")]
    volume: String,
    /// Whether this bar is closed. The field the normalized vocabulary has no word
    /// for — see [`decode_kline`].
    #[serde(rename = "x")]
    is_closed: bool,
}

#[derive(Deserialize)]
struct KlineData {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kline: KlineBody,
}

/// The REST book snapshot (`GET /fapi/v1/depth`). Note the absence: unlike
/// Hyperliquid's `l2Book`, it does **not** echo the symbol, so the caller has to
/// supply it — see [`decode_rest_depth`].
#[derive(Deserialize)]
struct RestDepth {
    #[serde(rename = "T")]
    transact_time: i64,
    bids: Vec<RawLevel>,
    asks: Vec<RawLevel>,
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Venue decimal *string* → fixed-point [`Decimal`].
///
/// Binance sends every price and size as a string for exactly the reason we parse
/// them this way. The captured BTC mark is `"64346.70000000"`; through an `f64` it
/// is not that number, and the difference lands on the tick.
fn parse_dec(field: &'static str, s: &str) -> Result<Decimal> {
    Decimal::from_str(s).map_err(|_| DecodeError::BadNumber {
        field,
        value: s.to_string(),
    })
}

fn to_level(raw: &RawLevel) -> Result<Level> {
    Ok(Level::new(
        parse_dec("px", &raw[0])?,
        parse_dec("sz", &raw[1])?,
    ))
}

fn resolve(symbols: &SymbolTable, symbol: &str) -> Result<SymbolId> {
    symbols
        .id(symbol)
        .ok_or_else(|| DecodeError::UnknownSymbol(symbol.to_string()))
}

// ── per-feed decoders ────────────────────────────────────────────────────────

/// A partial-depth frame → [`MarketEvent::Book`].
///
/// `ts_event` is `T`, the matching engine's transaction time, and **not** `E`. The
/// two differ by the venue's own push latency; on the captured frame that is 43 ms,
/// and 43 ms of a hundred-updates-per-second book is four books' worth of ordering.
/// The core's rule is the event's own time, and `T` is when this book existed.
fn decode_depth(data: DepthData, symbols: &SymbolTable) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.symbol)?;
    Ok(MarketEvent::Book(BookSnapshot {
        symbol_id,
        bids: data.bids.iter().map(to_level).collect::<Result<_>>()?,
        asks: data.asks.iter().map(to_level).collect::<Result<_>>()?,
        ts_event: data.transact_time * MS_TO_NS,
    }))
}

/// An `aggTrade` frame → [`MarketEvent::Trade`].
///
/// The aggressor side is the inverse of `m`, and getting it backwards is the most
/// expensive one-character mistake available on this venue: nothing malforms, no
/// frame is dropped, and every order-flow statistic downstream — signed volume, trade
/// imbalance, any strategy that reads aggression — has its sign flipped for the life
/// of the session.
///
/// The size is the *aggregate* over one taker order at one price, not one execution.
/// That is the trade a book consumer wants and it is not the trade a fill reconciler
/// wants, which is why this feed is public market data and never a fill source.
fn decode_agg_trade(data: AggTradeData, symbols: &SymbolTable) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.symbol)?;
    Ok(MarketEvent::Trade(Trade {
        symbol_id,
        px: parse_dec("p", &data.px)?,
        sz: parse_dec("q", &data.qty)?,
        side: if data.buyer_is_maker {
            Side::Sell
        } else {
            Side::Buy
        },
        ts_event: data.trade_time * MS_TO_NS,
    }))
}

/// A `bookTicker` frame → [`MarketEvent::Bbo`].
///
/// A side quoted at zero yields **no event**, the same way Hyperliquid's one-sided
/// `bbo` does. A `Bbo` with a zero bid is not a wide market, it is a market with no
/// bid, and a consumer computing a mid from it gets half the ask.
fn decode_book_ticker(data: BookTickerData, symbols: &SymbolTable) -> Result<Option<MarketEvent>> {
    let symbol_id = resolve(symbols, &data.symbol)?;
    let bid_px = parse_dec("b", &data.bid_px)?;
    let ask_px = parse_dec("a", &data.ask_px)?;
    if bid_px.is_zero() || ask_px.is_zero() {
        return Ok(None);
    }
    Ok(Some(MarketEvent::Bbo(Bbo {
        symbol_id,
        bid_px,
        bid_sz: parse_dec("B", &data.bid_sz)?,
        ask_px,
        ask_sz: parse_dec("A", &data.ask_sz)?,
        ts_event: data.transact_time * MS_TO_NS,
    })))
}

/// A `markPriceUpdate` frame → [`MarketEvent::Ticker`].
///
/// Three fields of the normalized [`Ticker`] have no source on this stream and are
/// reported absent rather than approximated:
///
/// - **`mid_px`** — the mark-price stream carries no book. `bookTicker` does, on a
///   different stream with a different cadence, and folding one into the other would
///   hand a consumer a mid that is not contemporaneous with the mark beside it.
/// - **`open_interest`** — Binance publishes it on `GET /fapi/v1/openInterest` only,
///   at one weight per symbol per call. There is no stream. `None` is the honest
///   answer; a polled value stamped with a streamed frame's time would be a number
///   that looks live and is up to a poll interval old.
/// - **The funding *interval*** — which is why `symbols` is a parameter here. The
///   frame carries the rate and the *next* funding time and never the period between
///   them, so the period comes from the table
///   ([`SymbolTable::funding_interval`](crate::symbols::SymbolTable::funding_interval)),
///   which is populated from `GET /fapi/v1/fundingInfo` and otherwise defaults to
///   eight hours. ADR-0011 pairs the rate with its interval precisely so this cannot
///   be dropped on the floor; on Hyperliquid the constant is the only source, here it
///   is the fallback.
fn decode_mark_price(
    data: MarkPriceData,
    symbols: &SymbolTable,
    ts_ingest: Nanos,
) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.symbol)?;
    if data.funding_rate.is_empty() {
        return Err(DecodeError::NotAPerpetual {
            symbol: data.symbol,
        });
    }
    // An index of `""` means "this contract tracks no external reference", which is a
    // different statement from a bad number and must not become one.
    let index_px = match data.index_px.as_deref() {
        None | Some("") => None,
        Some(s) => Some(parse_dec("i", s)?),
    };
    Ok(MarketEvent::Ticker(Ticker {
        symbol_id,
        mark_px: parse_dec("p", &data.mark_px)?,
        index_px,
        mid_px: None,
        funding: Some(Funding {
            rate: parse_dec("r", &data.funding_rate)?,
            interval: symbols.funding_interval(symbol_id),
        }),
        open_interest: None,
        // The payoff of ADR-0011's `Option`. This venue stamps its ticker, so the
        // event is venue-timed, reproduces on replay, and `is_venue_timed()` says so —
        // where the same field is `None` on Hyperliquid and the same consumer knows
        // not to compare orderings.
        ts_venue: Some(data.event_time * MS_TO_NS),
        ts_ingest,
    }))
}

/// A `kline` frame → [`MarketEvent::Candle`], **and only when the bar has closed**.
///
/// This is the one place the normalized vocabulary came up a field short, and the
/// shortfall is real rather than cosmetic. [`Candle`] documents `ts_event` as "the
/// candle's close time — the point at which it is final for ordering", and it has no
/// way to say *not final*. Binance republishes the in-progress bar on every trade:
/// the 85-second capture behind these fixtures holds 112 `kline_1m` frames for one
/// symbol and **one** of them is closed. All 112 carry the same `T`, so emitting them
/// all would put 111 events on the bus that claim to be the same final bar and are
/// each a different bar — and an event-time sort cannot separate them, because their
/// ordering keys are identical.
///
/// Dropping them is the [`decode_book_ticker`] precedent: skip rather than fabricate.
/// What it costs is stated plainly — **a strategy cannot see an in-progress bar
/// through this adapter**, and closing that needs an `is_final` flag on
/// [`Candle`] in `axon-core`, which is a port change and not an adapter one.
///
/// `ts_event` is `T + 1 ms` for the reason the Hyperliquid decoder documents at
/// length: `T` is the bar's *last* millisecond, so a bar stamped `T` sorts equal to
/// the trades printed inside it, and the research side stamps the same bar the same
/// way (`axon.strategies.data.CLOSE_STAMP_OFFSET_MS`). Both venues agree because both
/// publish a last-millisecond close, which is a small piece of evidence that the
/// convention is about bars rather than about Hyperliquid.
fn decode_kline(
    data: KlineData,
    interval: axon_core::CandleInterval,
    symbols: &SymbolTable,
) -> Result<Option<MarketEvent>> {
    let symbol_id = resolve(symbols, &data.symbol)?;
    let k = data.kline;
    if !k.is_closed {
        return Ok(None);
    }
    Ok(Some(MarketEvent::Candle(Candle {
        symbol_id,
        interval,
        open: parse_dec("o", &k.open)?,
        high: parse_dec("h", &k.high)?,
        low: parse_dec("l", &k.low)?,
        close: parse_dec("c", &k.close)?,
        volume: parse_dec("v", &k.volume)?,
        open_time: k.open_time * MS_TO_NS,
        ts_event: (k.close_time + 1) * MS_TO_NS,
    })))
}

// ── entry points ─────────────────────────────────────────────────────────────

/// Decode one combined-stream text frame into zero or more normalized [`Event`]s.
///
/// Control frames — the `{"result":null,"id":n}` subscription acknowledgement and the
/// `{"error":{…}}` refusal — decode to an empty vec; they are not errors, and the
/// error frame's message is available separately through [`venue_error`] so a caller
/// can log it without tearing down a healthy socket.
///
/// `ts_ingest` is the moment the frame arrived, in event-time nanoseconds. Only the
/// mark-price feed reads it, and even there it is the *ingest* half of a `Ticker`
/// whose `ts_venue` is genuinely present. It is a parameter rather than a clock read
/// inside the decoder for ADR-0011 §4's reason: a hidden wall-clock read would make
/// every ticker assertion non-deterministic and put tokio-adjacent state into the
/// offline half of the adapter.
///
/// Unlike Hyperliquid's dispatcher this one routes on the **stream name**, not on the
/// payload's own event type. Two streams share `"e":"depthUpdate"` and mean opposite
/// things; the name is the only discriminator the venue provides.
pub fn decode_ws_message(raw: &str, symbols: &SymbolTable, ts_ingest: Nanos) -> Result<Vec<Event>> {
    let env: Envelope = match serde_json::from_str::<Envelope>(raw) {
        Ok(env) => env,
        Err(e) => {
            // No `stream` field. Either a control frame (fine) or a bare payload from
            // the raw endpoint (not fine, and silent if we let it pass).
            let v: serde_json::Value = serde_json::from_str(raw).map_err(|_| e)?;
            if v.get("e").is_some() {
                return Err(DecodeError::Unenveloped);
            }
            return Ok(Vec::new());
        }
    };

    let events = match stream_kind(&env.stream) {
        Some(StreamKind::PartialDepth) => {
            vec![decode_depth(serde_json::from_value(env.data)?, symbols)?.into()]
        }
        Some(StreamKind::AggTrade) => {
            vec![decode_agg_trade(serde_json::from_value(env.data)?, symbols)?.into()]
        }
        Some(StreamKind::BookTicker) => {
            decode_book_ticker(serde_json::from_value(env.data)?, symbols)?
                .into_iter()
                .map(Event::from)
                .collect()
        }
        Some(StreamKind::MarkPrice) => {
            vec![decode_mark_price(serde_json::from_value(env.data)?, symbols, ts_ingest)?.into()]
        }
        Some(StreamKind::Kline) => {
            let interval =
                stream_kline_interval(&env.stream).ok_or(DecodeError::Malformed("kline"))?;
            decode_kline(serde_json::from_value(env.data)?, interval, symbols)?
                .into_iter()
                .map(Event::from)
                .collect()
        }
        // Refused loudly, and this is the single most important refusal in the crate.
        // The frame is byte-compatible with a partial-depth snapshot; reading it as one
        // replaces the entire book with the two levels that changed. Falling through to
        // the ignore path below would make a mis-typed stream name a subscription that
        // looks healthy and produces nothing.
        Some(StreamKind::DiffDepth) => {
            return Err(DecodeError::UnsupportedStream("depth (incremental diff)"))
        }
        // A stream we never subscribe to. Ignored rather than refused: an unknown name
        // must not close a socket that is carrying four healthy feeds.
        None => Vec::new(),
    };
    Ok(events)
}

/// Extract the message from a venue error frame: `{"error":{"code":..,"msg":".."}}`.
///
/// A rejected subscription arrives as an ordinary frame on a healthy socket, exactly
/// as it does on Hyperliquid. Callers must log it and keep the connection: tearing
/// down on a malformed subscription produces an infinite reconnect loop that never
/// fixes the malformed subscription and drops every good feed on each pass.
///
/// Returns the venue's own text, allocated. This is a diagnostics path — an error
/// frame is rare — so a proper parse beats hand-scanning for `"msg":"`, which would
/// start silently returning `None` the day the venue adds a space after a colon.
pub fn venue_error(raw: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let err = v.get("error")?;
    let msg = err.get("msg")?.as_str()?;
    match err.get("code").and_then(serde_json::Value::as_i64) {
        Some(code) => Some(format!("{msg} (code {code})")),
        None => Some(msg.to_owned()),
    }
}

/// Decode a REST `GET /fapi/v1/depth` snapshot into a [`MarketEvent::Book`].
///
/// `symbol_id` is an **argument**, because the response does not carry the symbol.
/// Hyperliquid's `l2Book` echoes `coin` and its REST decoder resolves it like any
/// frame; Binance answers a per-symbol query with a per-symbol body and leaves the
/// correlation to the caller. Passing the id in rather than threading a symbol string
/// through means the one place that can get it wrong is the call site that already
/// chose the symbol.
pub fn decode_rest_depth(raw: &str, symbol_id: SymbolId) -> Result<MarketEvent> {
    let d: RestDepth = serde_json::from_str(raw)?;
    Ok(MarketEvent::Book(BookSnapshot {
        symbol_id,
        bids: d.bids.iter().map(to_level).collect::<Result<_>>()?,
        asks: d.asks.iter().map(to_level).collect::<Result<_>>()?,
        ts_event: d.transact_time * MS_TO_NS,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::CandleInterval;
    use rust_decimal_macros::dec;

    // ── captured frames ──────────────────────────────────────────────────────
    //
    // Every constant below marked "captured" is one line, byte for byte, out of an
    // 85-second capture off `wss://fstream.binancefuture.com/stream?streams=…`
    // (Binance USD-M futures **testnet**, BTCUSDT + ETHUSDT, 2026-07-26; 2 353 frames).
    // Production is geo-blocked from this host — `fapi.binance.com` answers HTTP 451 —
    // so the testnet is the only Binance these fixtures could come from, and that is a
    // caveat on them, not a claim about them. The wire format is the same; the
    // liquidity is not, and neither is the presence of the undocumented `st` field.
    //
    // They are reproduced whole rather than tidied, because the things this feed
    // actually gets wrong are only visible in the real payload: the two timestamps
    // that differ, the undocumented `ps`/`nq`/`st` fields, the spaces Binance's kline
    // serializer puts after commas and no other frame has, and — the important one —
    // that the diff and snapshot depth frames are the same shape.

    /// Captured. The 20-level snapshot. Note `"e":"depthUpdate"`.
    const DEPTH20_FRAME: &str = r#"{"stream":"btcusdt@depth20@100ms","data":{"e":"depthUpdate","E":1785048483093,"T":1785048483050,"s":"BTCUSDT","ps":"BTCUSDT","U":359220626938,"u":359220628357,"pu":359220619477,"b":[["64336.00","73.3346"],["64335.90","419.4944"],["64335.80","189.7059"],["64335.70","55.1235"],["64335.40","6.8312"],["64335.30","7.8341"],["64335.20","7.2485"],["64335.10","7.2231"],["64334.80","7.1981"],["64334.70","7.3117"],["64334.60","942.7709"],["64334.50","2510.5178"],["64334.40","1256.3930"],["64334.30","0.0112"],["64317.80","0.0010"],["64317.70","0.0010"],["64317.30","78.9597"],["64315.30","0.0100"],["64309.30","0.0010"],["64304.60","0.0028"]],"a":[["64346.70","589.8041"],["64347.00","77.6888"],["64347.50","544.6208"],["64347.60","855.7032"],["64348.80","622.4083"],["64349.00","544.6122"],["64349.10","0.0032"],["64349.20","77.6888"],["64349.60","49.5082"],["64350.00","77.6903"],["64350.70","66.0919"],["64350.80","66.0919"],["64351.00","687.6992"],["64351.10","51.8759"],["64351.90","51.8759"],["64352.00","621.6074"],["64352.60","77.6888"],["64352.90","49.5082"],["64353.20","47.3973"],["64353.30","49.5082"]],"st":1}}"#;

    /// Captured. The **incremental diff**, from the same socket in the same second.
    /// Same event type, same field names, two levels instead of forty.
    const DEPTH_DIFF_FRAME: &str = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1785048482829,"T":1785048482718,"s":"BTCUSDT","ps":"BTCUSDT","U":359220619477,"u":359220619477,"pu":359220613550,"b":[],"a":[["64346.70","589.8041"]],"st":1}}"#;

    /// Captured. `"m":false` — the buyer was the taker, so the aggressor bought.
    /// `E` and `T` are 154 ms apart on this frame.
    const AGG_TRADE_FRAME: &str = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1785048482872,"a":300238285,"s":"BTCUSDT","p":"64346.70","q":"0.0200","nq":"0.0200","f":521512568,"l":521512568,"T":1785048482718,"m":false,"st":1}}"#;

    /// Captured. `"m":true` — the buyer was the **maker**, so the aggressor sold.
    const AGG_TRADE_MAKER_FRAME: &str = r#"{"stream":"ethusdt@aggTrade","data":{"e":"aggTrade","E":1785048482856,"a":181556408,"s":"ETHUSDT","p":"1882.58","q":"12.363","nq":"12.363","f":307384337,"l":307384337,"T":1785048482852,"m":true,"st":1}}"#;

    /// Captured.
    const BOOK_TICKER_FRAME: &str = r#"{"stream":"btcusdt@bookTicker","data":{"e":"bookTicker","u":359220650272,"s":"BTCUSDT","ps":"BTCUSDT","b":"64336.00","B":"73.3346","a":"64346.70","A":"589.8026","T":1785048483771,"E":1785048483771,"st":1}}"#;

    /// Captured. The frame Hyperliquid's equivalent has no `E` on.
    const MARK_PRICE_FRAME: &str = r#"{"stream":"btcusdt@markPrice@1s","data":{"e":"markPriceUpdate","E":1785048483000,"s":"BTCUSDT","p":"64346.70000000","ap":"64346.70000000","P":"64335.31235580","i":"64399.61608696","r":"0.00005301","T":1785052800000,"st":1}}"#;

    /// Captured. `"x":false` — the bar is still running. 111 of the capture's 112
    /// `kline_1m` frames look like this.
    const KLINE_OPEN_FRAME: &str = r#"{"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1785048483771,"s":"BTCUSDT","k":{"t":1785048480000, "T":1785048539999, "s":"BTCUSDT", "i":"1m", "f":521512559, "L":521512569, "o":"64346.70", "c":"64346.70", "h":"64346.70", "l":"64346.70", "v":"0.0502", "n":11, "x":false, "q":"3230.204340", "V":"0.0502", "Q":"3230.204340", "B":"0"}}}"#;

    /// Captured. The one closed bar in the capture — same `t` and `T` as the frame
    /// above, different OHLCV, `"x":true`.
    const KLINE_CLOSED_FRAME: &str = r#"{"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1785048540333,"s":"BTCUSDT","k":{"t":1785048480000, "T":1785048539999, "s":"BTCUSDT", "i":"1m", "f":521512559, "L":521512793, "o":"64346.70", "c":"64346.70", "h":"64346.70", "l":"64336.00", "v":"24.8783", "n":235, "x":true, "q":"1600832.457590", "V":"24.4997", "Q":"1576474.845990", "B":"0"}}}"#;

    /// Captured (`GET /fapi/v1/depth?symbol=BTCUSDT&limit=5`). No `symbol` field.
    const REST_DEPTH_BODY: &str = r#"{"lastUpdateId":359226328428,"E":1785048698972,"T":1785048698968,"bids":[["64336.00","69.0738"],["64335.90","420.0066"],["64335.80","189.7059"],["64335.70","55.1235"],["64335.40","6.8312"]],"asks":[["64346.70","525.0349"],["64347.00","77.6888"],["64347.50","544.6208"],["64347.60","855.7032"],["64348.80","622.4083"]]}"#;

    /// A fixed stand-in for "when the frame arrived". Only the mark-price feed reads
    /// it; pinning it keeps those assertions deterministic, which is the reason the
    /// decoder takes it as an argument instead of reading a clock.
    const TS_INGEST: Nanos = 1_785_048_483_500 * MS_TO_NS;

    fn symbols() -> SymbolTable {
        SymbolTable::from_ordered(["BTCUSDT", "ETHUSDT"])
    }

    fn market(ev: &Event) -> &MarketEvent {
        match ev {
            Event::Market(m) => m,
            Event::Exec(e) => panic!("expected a market event, got {e:?}"),
        }
    }

    fn one(frame: &str) -> Event {
        let mut evs = decode_ws_message(frame, &symbols(), TS_INGEST).expect("decodes");
        assert_eq!(evs.len(), 1, "expected exactly one event from {frame}");
        evs.pop().unwrap()
    }

    #[test]
    fn decodes_a_captured_twenty_level_book_snapshot() {
        let ev = one(DEPTH20_FRAME);
        let MarketEvent::Book(snap) = market(&ev) else {
            panic!("expected a Book event");
        };
        assert_eq!(snap.symbol_id, SymbolId::new(0));
        assert_eq!(snap.bids.len(), 20);
        assert_eq!(snap.asks.len(), 20);
        assert_eq!(snap.bids[0], Level::new(dec!(64336.00), dec!(73.3346)));
        assert_eq!(snap.asks[0], Level::new(dec!(64346.70), dec!(589.8041)));
        // Highest-first bids, lowest-first asks, as the venue sends them.
        assert!(snap.bids[0].px > snap.bids[1].px);
        assert!(snap.asks[0].px < snap.asks[1].px);
    }

    #[test]
    fn a_book_is_stamped_with_the_engines_transaction_time_and_not_the_push_time() {
        // Both are on the frame and they are 43 ms apart. `E` is when the venue pushed
        // the bytes — our own `ts_ingest` measures the same thing one hop later — while
        // `T` is when this book existed. On a stream publishing ten times a second, 43
        // ms of mis-stamping reorders four books against every trade beside them.
        let ev = one(DEPTH20_FRAME);
        let MarketEvent::Book(snap) = market(&ev) else {
            panic!("expected Book");
        };
        assert_eq!(snap.ts_event, 1_785_048_483_050 * MS_TO_NS, "T");
        assert_ne!(snap.ts_event, 1_785_048_483_093 * MS_TO_NS, "not E");
        assert_eq!(1_785_048_483_093i64 - 1_785_048_483_050, 43);
    }

    #[test]
    fn the_incremental_depth_stream_is_refused_instead_of_being_read_as_a_snapshot() {
        // **The refusal this crate exists to make.** These two frames came off one
        // socket in the same second. Their event type, their field names and their
        // types are identical; only the stream name differs. Decoded as a snapshot, the
        // diff below replaces a 40-level book with one ask — a book that is wrong,
        // looks complete, and is refreshed ten times a second with no error anywhere.
        let diff: serde_json::Value = serde_json::from_str(DEPTH_DIFF_FRAME).unwrap();
        let snap: serde_json::Value = serde_json::from_str(DEPTH20_FRAME).unwrap();
        assert_eq!(diff["data"]["e"], snap["data"]["e"], "same event type");
        let keys = |v: &serde_json::Value| {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(
            keys(&diff["data"]),
            keys(&snap["data"]),
            "and the same field set: nothing in the payload tells them apart"
        );

        let err = decode_ws_message(DEPTH_DIFF_FRAME, &symbols(), TS_INGEST).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::UnsupportedStream("depth (incremental diff)")
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_maker_buy_is_an_aggressor_sell_because_m_says_who_was_passive_not_who_bought() {
        // The one-character mistake with no symptom. `m: true` means the *buyer* was
        // the maker, so the trade was initiated by a seller. Read as "was a buy" it
        // inverts signed volume, trade imbalance and every aggression feature for the
        // whole session, and nothing malforms to say so.
        let buy = one(AGG_TRADE_FRAME);
        let MarketEvent::Trade(t) = market(&buy) else {
            panic!("expected Trade");
        };
        assert_eq!(t.symbol_id, SymbolId::new(0));
        assert_eq!(t.px, dec!(64346.70));
        assert_eq!(t.sz, dec!(0.0200));
        assert_eq!(t.side, Side::Buy, "m == false: the taker bought");

        let sell = one(AGG_TRADE_MAKER_FRAME);
        let MarketEvent::Trade(t) = market(&sell) else {
            panic!("expected Trade");
        };
        assert_eq!(t.symbol_id, SymbolId::new(1));
        assert_eq!(t.px, dec!(1882.58));
        assert_eq!(t.side, Side::Sell, "m == true: the taker sold");
    }

    #[test]
    fn a_trade_is_stamped_with_the_trade_time_and_not_the_frame_push_time() {
        // 154 ms apart on this captured frame, which on a busy tape is hundreds of
        // prints out of order. `T` is when the trade happened; `E` is when Binance got
        // round to telling us.
        let ev = one(AGG_TRADE_FRAME);
        let MarketEvent::Trade(t) = market(&ev) else {
            panic!("expected Trade");
        };
        assert_eq!(t.ts_event, 1_785_048_482_718 * MS_TO_NS, "T");
        assert_ne!(t.ts_event, 1_785_048_482_872 * MS_TO_NS, "not E");
    }

    #[test]
    fn decodes_a_captured_book_ticker() {
        let ev = one(BOOK_TICKER_FRAME);
        let MarketEvent::Bbo(q) = market(&ev) else {
            panic!("expected Bbo");
        };
        assert_eq!(q.symbol_id, SymbolId::new(0));
        assert_eq!(q.bid_px, dec!(64336.00));
        assert_eq!(q.bid_sz, dec!(73.3346));
        assert_eq!(q.ask_px, dec!(64346.70));
        assert_eq!(q.ask_sz, dec!(589.8026));
        assert_eq!(q.ts_event, 1_785_048_483_771 * MS_TO_NS);
        assert!(q.ask_px > q.bid_px);
    }

    #[test]
    fn a_side_quoted_at_zero_yields_no_bbo_rather_than_a_book_with_no_bid() {
        // Hand-written: an empty side is not reachable on a liquid testnet symbol, so
        // there is no captured frame for it. The shape is the venue's own, with one
        // price zeroed. A `Bbo` carrying a zero bid is not a wide market — a consumer
        // computing a mid from it gets half the ask, and sizes a cross against it.
        let empty_bid = BOOK_TICKER_FRAME.replace(r#""b":"64336.00""#, r#""b":"0.00""#);
        assert!(decode_ws_message(&empty_bid, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
        let empty_ask = BOOK_TICKER_FRAME.replace(r#""a":"64346.70""#, r#""a":"0""#);
        assert!(decode_ws_message(&empty_ask, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
    }

    fn ticker_from(frame: &str, symbols: &SymbolTable) -> Ticker {
        let mut evs = decode_ws_message(frame, symbols, TS_INGEST).expect("decodes");
        assert_eq!(evs.len(), 1);
        match evs.pop().unwrap() {
            Event::Market(MarketEvent::Ticker(t)) => t,
            other => panic!("expected Ticker, got {other:?}"),
        }
    }

    #[test]
    fn a_mark_price_frame_is_venue_timed_which_is_the_first_time_that_option_has_been_some() {
        // ADR-0011 made `ts_venue` an `Option` on the strength of one venue that sends
        // no timestamp, and the rejected alternative was a single `ts_event` field with
        // a doc comment. This is the assertion that decides between them: the same type
        // carries a real venue time here and `None` on Hyperliquid, and
        // `is_venue_timed()` — which the parity harness reads to decide whether a feed
        // can be compared across runs at all — answers differently for the two without
        // either adapter knowing about the other.
        let t = ticker_from(MARK_PRICE_FRAME, &symbols());
        assert_eq!(t.ts_venue, Some(1_785_048_483_000 * MS_TO_NS));
        assert_eq!(t.ts_ingest, TS_INGEST);
        assert!(t.is_venue_timed(), "this feed replays deterministically");
        assert_eq!(t.ts_event(), t.ts_venue.unwrap(), "the venue's clock wins");
        // And the gap between the two is the feed latency, which is only a meaningful
        // number because both are present (docs/05-latency-model.md).
        assert_eq!(t.ts_ingest - t.ts_event(), 500 * MS_TO_NS);
    }

    #[test]
    fn decodes_a_captured_mark_price_frame_and_leaves_absent_what_it_does_not_carry() {
        // The exclusions are the decision. This stream has no book and no open
        // interest; back-filling `mid_px` from `mark_px` would report a tradeable price
        // nobody is quoting, and stamping a polled open interest onto a streamed frame
        // would make a number that is up to a poll old look live.
        let t = ticker_from(MARK_PRICE_FRAME, &symbols());
        assert_eq!(t.symbol_id, SymbolId::new(0));
        assert_eq!(t.mark_px, dec!(64346.70000000));
        assert_eq!(t.index_px, Some(dec!(64399.61608696)));
        assert_eq!(t.mid_px, None, "no book on this stream");
        assert_eq!(t.open_interest, None, "REST only, at one weight per call");
        // Mark and index genuinely differ on this frame, so a decoder that read one
        // field into both would still look correct without this assertion.
        assert_ne!(t.mark_px, t.index_px.unwrap());
    }

    #[test]
    fn the_funding_rate_is_stamped_with_the_period_this_venue_charges_it_over() {
        // ADR-0011's 8x error, and the day it was written to catch. Hyperliquid funds
        // hourly and publishes the hourly rate; this venue funds every eight hours. The
        // frame carries the rate and the *next* funding time and never the period, so
        // the period comes from the symbol table — published where `fundingInfo` says
        // so, assumed at eight hours where it does not.
        assert!(
            !MARK_PRICE_FRAME.contains("nterval"),
            "the venue would have to start sending a period for this to be decoded"
        );
        let mut s = symbols();
        let f = ticker_from(MARK_PRICE_FRAME, &s)
            .funding
            .expect("perps fund");
        assert_eq!(f.rate, dec!(0.00005301), "stored exactly as published");
        assert_eq!(f.interval, crate::DEFAULT_FUNDING_INTERVAL_NS);
        assert_ne!(
            f.interval,
            3_600 * 1_000_000_000,
            "an hourly interval here is the 8x carry error"
        );

        // …and a symbol the venue publishes a period for gets that one instead.
        s.set_funding_hours("BTCUSDT", 4);
        let f = ticker_from(MARK_PRICE_FRAME, &s).funding.unwrap();
        assert_eq!(f.interval, 4 * 3_600 * 1_000_000_000);
    }

    #[test]
    fn a_dated_contract_context_is_refused_rather_than_given_a_funding_rate_of_zero() {
        // Hand-written from the venue's documented delivery shape: a quarterly's
        // `markPriceUpdate` carries `"r":""` and `"T":0`. Parsing `""` as a rate would
        // fail as a bad number, which is true but uninformative; refusing by name says
        // the thing an operator needs to hear. This is ADR-0011 §5's two-layer refusal:
        // `decode_exchange_info` already declines dated contracts by `contractType`,
        // and this declines one again at the frame.
        let frame = MARK_PRICE_FRAME.replace(r#""r":"0.00005301""#, r#""r":"""#);
        let err = decode_ws_message(&frame, &symbols(), TS_INGEST).unwrap_err();
        assert!(
            matches!(err, DecodeError::NotAPerpetual { ref symbol } if symbol == "BTCUSDT"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_contract_with_no_external_reference_reports_no_index_rather_than_a_bad_number() {
        // Hand-written. `"i":""` means this contract tracks nothing external — a
        // different statement from a malformed number, and `index_px` is already an
        // `Option` for exactly that case.
        let frame = MARK_PRICE_FRAME.replace(r#""i":"64399.61608696""#, r#""i":"""#);
        let t = ticker_from(&frame, &symbols());
        assert_eq!(t.index_px, None);
        assert_eq!(t.mark_px, dec!(64346.70000000), "the mark is untouched");
    }

    #[test]
    fn an_unclosed_bar_is_dropped_because_the_vocabulary_cannot_say_not_final() {
        // 111 of the capture's 112 kline frames are this one, all carrying the same
        // `T`. Emitting them would put 111 events on the bus that claim to be the same
        // final bar and are each a different bar — with identical ordering keys, so an
        // event-time sort cannot even separate them. Dropping is the `bookTicker`
        // precedent: skip rather than fabricate. The cost is real and is stated on
        // `decode_kline`: no in-progress bar reaches a strategy through this adapter.
        let open: serde_json::Value = serde_json::from_str(KLINE_OPEN_FRAME).unwrap();
        let closed: serde_json::Value = serde_json::from_str(KLINE_CLOSED_FRAME).unwrap();
        assert_eq!(
            open["data"]["k"]["T"], closed["data"]["k"]["T"],
            "same close time, different bar contents"
        );
        assert_ne!(open["data"]["k"]["v"], closed["data"]["k"]["v"]);

        assert!(decode_ws_message(KLINE_OPEN_FRAME, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
        assert_eq!(
            decode_ws_message(KLINE_CLOSED_FRAME, &symbols(), TS_INGEST)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn a_closed_bar_is_stamped_one_ms_past_its_last_millisecond_on_this_venue_too() {
        // The same convention the Hyperliquid decoder documents, and the same reason:
        // `T` is the bar's last millisecond, so a bar stamped `T` sorts equal to every
        // trade printed inside it. Both venues publish a last-millisecond close, which
        // is what makes `CLOSE_STAMP_OFFSET_MS` a fact about bars rather than about
        // Hyperliquid — and the research side stamps the same bar the same way, without
        // which `align_by_event_time` intersects to nothing.
        let ev = one(KLINE_CLOSED_FRAME);
        let MarketEvent::Candle(c) = market(&ev) else {
            panic!("expected Candle");
        };
        assert_eq!(c.symbol_id, SymbolId::new(0));
        assert_eq!(c.interval, CandleInterval::M1);
        assert_eq!(c.open, dec!(64346.70));
        assert_eq!(c.high, dec!(64346.70));
        assert_eq!(c.low, dec!(64336.00));
        assert_eq!(c.close, dec!(64346.70));
        assert_eq!(c.volume, dec!(24.8783));
        assert_eq!(c.open_time, 1_785_048_480_000 * MS_TO_NS);
        assert_eq!(c.ts_event, (1_785_048_539_999 + 1) * MS_TO_NS);
        assert_eq!(c.ts_event, c.open_time + 60 * 1_000 * MS_TO_NS);
        assert!(c.ts_event > 1_785_048_539_999 * MS_TO_NS);
    }

    #[test]
    fn the_candle_interval_comes_from_the_stream_name_not_from_the_payload() {
        // Both carry it (`k.i` is `"1m"`), and the stream name is the one that decides
        // — the same rule the depth streams force. One discriminator, used everywhere,
        // rather than two that can disagree.
        let ev = one(KLINE_CLOSED_FRAME);
        let MarketEvent::Candle(c) = market(&ev) else {
            panic!("expected Candle");
        };
        assert_eq!(c.interval, CandleInterval::M1);

        // An interval the normalized vocabulary has no word for is an error, not a
        // nearest match: a strategy handed 30-minute bars it asked 15 for computes
        // every feature over the wrong window.
        let renamed = KLINE_CLOSED_FRAME.replace("btcusdt@kline_1m", "btcusdt@kline_30m");
        assert!(matches!(
            decode_ws_message(&renamed, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::Malformed("kline")
        ));
    }

    #[test]
    fn a_bare_payload_from_the_raw_endpoint_is_an_error_and_not_a_control_frame() {
        // The `/ws/<stream>` endpoint sends the payload with no envelope. Since a bare
        // `depthUpdate` cannot be told from a diff, and since a frame with no `stream`
        // key otherwise looks exactly like a subscription acknowledgement, letting it
        // through the ignore path would make a connection to the wrong endpoint produce
        // no events, no errors, and a session that reads as healthy and is deaf.
        let env: serde_json::Value = serde_json::from_str(DEPTH20_FRAME).unwrap();
        let bare = env["data"].to_string();
        assert!(matches!(
            decode_ws_message(&bare, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::Unenveloped
        ));
    }

    #[test]
    fn subscription_acknowledgements_are_ignored_and_carry_no_events() {
        assert!(
            decode_ws_message(r#"{"result":null,"id":1}"#, &symbols(), TS_INGEST)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_stream_we_never_subscribed_to_is_ignored_rather_than_closing_the_socket() {
        // Liquidations arrive on `<sym>@forceOrder`. Nothing here asks for them, and
        // an unknown name must not be able to tear down a connection carrying four
        // healthy feeds.
        let frame = r#"{"stream":"btcusdt@forceOrder","data":{"e":"forceOrder","E":1,"o":{}}}"#;
        assert!(decode_ws_message(frame, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn venue_error_extracts_the_message_and_emits_no_event() {
        // A refused subscription is an ordinary frame on a healthy socket. Reconnecting
        // over it would resend the same bad subscription forever while dropping every
        // good one.
        let frame = r#"{"error":{"code":2,"msg":"Invalid request: invalid stream"},"id":1}"#;
        let msg = venue_error(frame).expect("error frames carry a message");
        assert!(msg.contains("Invalid request"), "got {msg:?}");
        assert!(msg.contains("code 2"), "and the venue's code: {msg:?}");
        assert!(decode_ws_message(frame, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn venue_error_ignores_every_other_frame() {
        assert_eq!(venue_error(DEPTH20_FRAME), None);
        assert_eq!(venue_error(r#"{"result":null,"id":1}"#), None);
        assert_eq!(venue_error("not json at all"), None);
    }

    #[test]
    fn an_unknown_symbol_is_an_error_rather_than_a_silently_dropped_frame() {
        let frame = AGG_TRADE_FRAME.replace(r#""s":"BTCUSDT""#, r#""s":"SOLUSDT""#);
        let err = decode_ws_message(&frame, &symbols(), TS_INGEST).unwrap_err();
        assert!(matches!(err, DecodeError::UnknownSymbol(ref s) if s == "SOLUSDT"));
    }

    #[test]
    fn a_price_that_is_not_a_number_is_an_error() {
        let frame = AGG_TRADE_FRAME.replace(r#""p":"64346.70""#, r#""p":"NaN""#);
        assert!(matches!(
            decode_ws_message(&frame, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::BadNumber { field: "p", .. }
        ));
    }

    #[test]
    fn a_book_level_with_a_third_element_fails_rather_than_being_truncated() {
        // Spot's REST book carries a third, always-empty element in each level and
        // futures' does not. A `Vec<Vec<String>>` would read the first two and carry on;
        // a fixed-length array refuses, which is what makes pointing this decoder at
        // the wrong product a decode failure rather than a plausible book.
        let frame =
            DEPTH20_FRAME.replace(r#"["64336.00","73.3346"]"#, r#"["64336.00","73.3346",[]]"#);
        assert!(matches!(
            decode_ws_message(&frame, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::Json(_)
        ));
    }

    #[test]
    fn decodes_the_rest_snapshot_against_a_symbol_the_response_does_not_carry() {
        // The difference from Hyperliquid in one assertion: `POST /info l2Book` echoes
        // `coin` and this body does not, so the correlation is the caller's job. Passing
        // the id in means the one place that can get it wrong already chose the symbol.
        assert!(
            !REST_DEPTH_BODY.contains("BTCUSDT"),
            "the fixture has to keep the venue's actual shape"
        );
        let MarketEvent::Book(snap) = decode_rest_depth(REST_DEPTH_BODY, SymbolId::new(0)).unwrap()
        else {
            panic!("expected Book");
        };
        assert_eq!(snap.symbol_id, SymbolId::new(0));
        assert_eq!(snap.bids.len(), 5);
        assert_eq!(snap.bids[0], Level::new(dec!(64336.00), dec!(69.0738)));
        assert_eq!(snap.asks[0], Level::new(dec!(64346.70), dec!(525.0349)));
        assert_eq!(snap.ts_event, 1_785_048_698_968 * MS_TO_NS);
    }

    #[test]
    fn every_decoded_event_carries_the_symbol_and_the_time_the_bus_will_order_it_on() {
        // The cheap total check: a new arm that forgot to fill either field would sort
        // to the front of the deterministic queue or route to instrument zero, and
        // both are silent.
        for frame in [
            DEPTH20_FRAME,
            AGG_TRADE_FRAME,
            BOOK_TICKER_FRAME,
            MARK_PRICE_FRAME,
            KLINE_CLOSED_FRAME,
        ] {
            let ev = one(frame);
            assert!(ev.ts_event() > 0, "no event time on {frame}");
            let m = market(&ev);
            assert_eq!(m.symbol_id(), SymbolId::new(0), "unrouted event: {frame}");
            assert_eq!(
                m.ts_event(),
                ev.ts_event(),
                "the bus and the market event must agree on the key"
            );
        }
    }
}
