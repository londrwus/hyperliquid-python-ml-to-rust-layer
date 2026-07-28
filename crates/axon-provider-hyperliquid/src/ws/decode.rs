//! Pure decoders: Hyperliquid WS/REST JSON → normalized [`Event`]s.
//!
//! This is the offline-testable heart of the adapter. Every venue quirk is
//! absorbed here: prices/sizes arrive as **strings** and become fixed-point
//! [`Decimal`](axon_core::Decimal) (never `f64`); timestamps arrive in **ms** and
//! become event-time **ns**; coin names become [`SymbolId`](axon_core::SymbolId)
//! via the [`SymbolMap`]. These functions do no I/O, so they are unit-tested
//! against captured message shapes with zero network.
//!
//! Public market feeds are decoded here; the account-scoped *user* channels live in
//! [`super::user`] but arrive on the same socket, so [`decode_ws_message`] is the
//! single dispatcher for both and returns the unified [`Event`].

use std::str::FromStr;

use axon_core::{
    Bbo, BookSnapshot, Candle, CandleInterval, Decimal, Event, Funding, Level, MarketEvent, Nanos,
    Side, SymbolId, Ticker, Trade,
};
use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid, SpecError};
use serde::Deserialize;

use super::funding::ASSUMED_FUNDING_INTERVAL_NS;
use super::user::decode_user_channel;
use crate::symbol_map::SymbolMap;
use crate::{MIN_ORDER_NOTIONAL_USD, PERP_MAX_DECIMALS, PRICE_SIG_FIGS};

/// Hyperliquid sends timestamps in milliseconds; the core keys on nanoseconds.
/// Shared with [`super::user`] so there is exactly one ms→ns conversion in the crate.
pub(crate) const MS_TO_NS: Nanos = 1_000_000;

/// Why a message failed to decode.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown coin: {0}")]
    UnknownCoin(String),
    #[error("bad number for {field}: {value:?}")]
    BadNumber { field: &'static str, value: String },
    #[error("malformed {0} message")]
    Malformed(&'static str),
    /// A channel we recognize and deliberately refuse to decode, rather than one we
    /// have never heard of (those are ignored). It exists for `activeSpotAssetCtx`:
    /// a spot context is a *different shape* wearing a nearly identical name, and
    /// staying silent about it would leave a subscription that looks healthy and
    /// never produces a price.
    #[error("unsupported channel: {0}")]
    UnsupportedChannel(&'static str),
    /// An asset whose declared `szDecimals` will not build a price grid or a lot.
    ///
    /// A refusal rather than a skipped asset: an asset missing from the instrument
    /// table fails closed later, at the first order, with nothing attached saying why.
    #[error("asset {coin} declares a szDecimals that is not a grid: {reason}")]
    BadPrecision { coin: String, reason: SpecError },
}

type Result<T> = std::result::Result<T, DecodeError>;

// ── on-wire shapes (extra fields like `n`, `hash`, `tid` are ignored) ──────────

#[derive(Deserialize)]
struct RawLevel {
    px: String,
    sz: String,
}

/// Every frame the venue sends: a channel name, and on all but one a payload.
///
/// `data` is **defaulted, not required**, because the heartbeat reply is
/// `{"channel":"pong"}` — eighteen bytes with no `data` field at all (captured in
/// `testdata/non-data-frames.jsonl`). Requiring it turned every heartbeat into
/// `WS decode error (frame dropped): json: missing field 'data' at line 1 column 18`:
/// 46 of them in a 1 h 44 m soak, on the exact line where a *real* decode failure
/// appears. That is the damage — an operator who sees that line 46 times learns to
/// skim past the one message that would tell them a decoder had drifted from the
/// venue.
///
/// It loosens the envelope and nothing else. An absent payload deserializes to
/// `Value::Null`, which every channel arm below still rejects, so a `l2Book` frame
/// that lost its payload is as loud as it ever was — see
/// `a_data_channel_with_no_payload_is_still_an_error`.
#[derive(Deserialize)]
struct Envelope {
    channel: String,
    #[serde(default)]
    data: serde_json::Value,
}

#[derive(Deserialize)]
struct L2Data {
    coin: String,
    time: i64,
    /// `[bids, asks]`; bids highest-first, asks lowest-first.
    levels: Vec<Vec<RawLevel>>,
}

#[derive(Deserialize)]
struct RawTrade {
    coin: String,
    /// `"B"` = aggressor bought, `"A"` = aggressor sold.
    side: String,
    px: String,
    sz: String,
    time: i64,
}

#[derive(Deserialize)]
struct BboData {
    coin: String,
    time: i64,
    /// `[bid, ask]`; either side may be `null` when that book side is empty.
    bbo: Vec<Option<RawLevel>>,
}

#[derive(Deserialize)]
struct RawCandle {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "s")]
    coin: String,
    #[serde(rename = "i")]
    interval: String,
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
}

/// The **perp** asset context carried by `activeAssetCtx`.
///
/// `oraclePx`, `funding` and `openInterest` are required, not `Option`, on purpose:
/// they are precisely the three fields a *spot* context lacks. Making them mandatory
/// means a spot payload that somehow reached this decoder fails to deserialize
/// instead of decoding into a perfectly plausible perp ticker with three values
/// quietly missing — which is the shape of bug that survives review and shows up as a
/// funding-carry strategy trading an instrument that has no funding.
///
/// The 24h summary (`prevDayPx`, `dayNtlVlm`) and the derived `premium`/`impactPxs`
/// are present on the wire and deliberately dropped; see [`Ticker`] for why they are
/// not part of the normalized vocabulary.
#[derive(Deserialize)]
struct PerpCtx {
    #[serde(rename = "markPx")]
    mark_px: String,
    #[serde(rename = "oraclePx")]
    oracle_px: String,
    /// `null` on a one-sided book. Absent and null both mean "no mid", and neither
    /// means "use the mark".
    #[serde(rename = "midPx", default)]
    mid_px: Option<String>,
    funding: String,
    #[serde(rename = "openInterest")]
    open_interest: String,
}

#[derive(Deserialize)]
struct ActiveAssetCtx {
    coin: String,
    ctx: PerpCtx,
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Venue decimal *string* → fixed-point [`Decimal`]. Shared with [`super::user`].
pub(crate) fn parse_dec(field: &'static str, s: &str) -> Result<Decimal> {
    Decimal::from_str(s).map_err(|_| DecodeError::BadNumber {
        field,
        value: s.to_string(),
    })
}

fn to_level(raw: &RawLevel) -> Result<Level> {
    Ok(Level::new(
        parse_dec("px", &raw.px)?,
        parse_dec("sz", &raw.sz)?,
    ))
}

fn resolve(symbols: &SymbolMap, coin: &str) -> Result<SymbolId> {
    symbols
        .id(coin)
        .ok_or_else(|| DecodeError::UnknownCoin(coin.to_string()))
}

/// Map Hyperliquid's interval string to a normalized [`CandleInterval`].
fn hl_interval(s: &str) -> Option<CandleInterval> {
    Some(match s {
        "1m" => CandleInterval::M1,
        "5m" => CandleInterval::M5,
        "15m" => CandleInterval::M15,
        "1h" => CandleInterval::H1,
        "4h" => CandleInterval::H4,
        "1d" => CandleInterval::D1,
        _ => return None,
    })
}

// ── per-feed decoders ────────────────────────────────────────────────────────

fn decode_l2(data: L2Data, symbols: &SymbolMap) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.coin)?;
    if data.levels.len() != 2 {
        return Err(DecodeError::Malformed("l2Book"));
    }
    let bids = data.levels[0].iter().map(to_level).collect::<Result<_>>()?;
    let asks = data.levels[1].iter().map(to_level).collect::<Result<_>>()?;
    Ok(MarketEvent::Book(BookSnapshot {
        symbol_id,
        bids,
        asks,
        ts_event: data.time * MS_TO_NS,
    }))
}

fn decode_bbo(data: BboData, symbols: &SymbolMap) -> Result<Option<MarketEvent>> {
    let symbol_id = resolve(symbols, &data.coin)?;
    if data.bbo.len() != 2 {
        return Err(DecodeError::Malformed("bbo"));
    }
    // A one-sided book yields no complete BBO — skip rather than fabricate.
    let (Some(bid), Some(ask)) = (&data.bbo[0], &data.bbo[1]) else {
        return Ok(None);
    };
    Ok(Some(MarketEvent::Bbo(Bbo {
        symbol_id,
        bid_px: parse_dec("bid_px", &bid.px)?,
        bid_sz: parse_dec("bid_sz", &bid.sz)?,
        ask_px: parse_dec("ask_px", &ask.px)?,
        ask_sz: parse_dec("ask_sz", &ask.sz)?,
        ts_event: data.time * MS_TO_NS,
    })))
}

/// `candle` → [`MarketEvent::Candle`].
///
/// `ts_event` is `T + 1 ms`, not `T`. The venue's `T` is the bar's **last**
/// millisecond, so a bar stamped `T` sorts *equal* to every trade printed inside that
/// millisecond — and an event-time sort is then free to hand a strategy the closed bar
/// before the tick that closed it, which is a feature computed on a bar that had not
/// happened yet. `T + 1` is the instant the bar is final for ordering, which is exactly
/// what [`axon_core::Candle::ts_event`](axon_core::Candle) already claims it is.
///
/// It is also the stamp the research side puts on the same bar
/// (`axon.strategies.data.CLOSE_STAMP_OFFSET_MS`). That agreement is load-bearing rather
/// than tidy: the two halves being one millisecond apart makes an `align_by_event_time`
/// between an online bar feature and its offline recompute intersect to *nothing* — the
/// stamps differ by 1e6 ns on a grid 3.6e12 ns wide, so no pair ever collides — and the
/// feature-parity gate then fails as "an empty matrix proves nothing", a long way from
/// the cause.
///
/// # The venue republishes the bar it is still filling, and never says it is done
///
/// Measured, because it decides what every consumer of this event may believe. Over
/// 12.9 minutes on `wss://api.hyperliquid.xyz/ws` and its testnet twin (BTC/ETH/SOL 1m,
/// BTC 5m, testnet BTC/ETH 1m), **1 321 candle frames described 69 bars** — 63 of them
/// republished, one BTC 5-minute bar 192 times and one BTC minute 62 times. Every frame
/// for a bar carries the *same* `t` and `T`, verified across all 69, so two frames for
/// one bar are indistinguishable by timestamp: only `c`/`h`/`l`/`v`/`n` move. And **1 317
/// of the 1 321 arrived before `T`** — only 4 of 69 bars ever received a frame at or
/// after their own close, which is nowhere near often enough to read as a close marker.
///
/// Two things follow, and both are why this decoder does *not* try to label finality:
///
/// - **There is no `x`.** Binance's kline carries one and its adapter drops every
///   non-final frame on the strength of it. Hyperliquid publishes no such field, and no
///   receipt-time rule can stand in for it: `ts_ingest >= ts_event` would have been
///   `false` for 1 317 of the 1 321 frames, so a `Candle::closed` filled in that way
///   would be very nearly a constant on this venue — a field that all but always says
///   "forming" is a place for a future reader to believe something, not a fix.
/// - **Dropping the in-progress frames would drop the bars.** Because the venue sends
///   nothing after `T`, the last frame it sends *is* the bar — and against the venue's
///   own `candleSnapshot` it was byte-identical for 6 of 7 consecutive BTC minutes. The
///   seventh differed by 0.001 in `v`: its last frame landed 35 ms before the close and
///   a trade arrived in the tail. So a decoder filtering for "final" frames would emit
///   nothing at all for most bars, which is the one failure worse than a forming bar
///   reaching the bus — a hole in a feature window that nothing downstream can see.
///
/// So a forming bar **does** reach the bus, and finality is derived from the stream by
/// whoever needs it: `axon_runtime::mdring` closes a bar when a frame for a later
/// `open_time` arrives, and `axon.strategies.data.closed_rows` applies the same rule
/// offline. The one consumer that cannot derive it is the core's own clock, which is why
/// `axon_runtime::handler::advances_the_clock` refuses a computed close time outright
/// instead of trying to tell these frames apart.
fn decode_candle(data: RawCandle, symbols: &SymbolMap) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.coin)?;
    let interval = hl_interval(&data.interval).ok_or(DecodeError::Malformed("candle"))?;
    Ok(MarketEvent::Candle(Candle {
        symbol_id,
        interval,
        open: parse_dec("open", &data.open)?,
        high: parse_dec("high", &data.high)?,
        low: parse_dec("low", &data.low)?,
        close: parse_dec("close", &data.close)?,
        volume: parse_dec("volume", &data.volume)?,
        open_time: data.open_time * MS_TO_NS,
        ts_event: (data.close_time + 1) * MS_TO_NS,
    }))
}

/// `activeAssetCtx` → [`MarketEvent::Ticker`].
///
/// `ts_ingest` is the frame's arrival time. Unlike every other feed here, this one
/// carries **no** venue timestamp, so there is nothing to convert from ms and nothing
/// to order on but our own clock. That is recorded honestly — `ts_venue` stays `None`
/// — rather than presented as a venue time, because an event whose ordering key came
/// from a wall clock does not reproduce on replay and the Phase-5 parity harness has
/// to be able to tell the difference.
///
/// The frame is missing a *second* thing the core needs: the funding **interval**.
/// [`ASSUMED_FUNDING_INTERVAL_NS`] fills it in, and the name is the warning — nothing
/// in this payload would contradict it if the venue changed cadence, so the check
/// lives in [`super::funding`] and runs against `fundingHistory` instead.
fn decode_ticker(
    data: ActiveAssetCtx,
    symbols: &SymbolMap,
    ts_ingest: Nanos,
) -> Result<MarketEvent> {
    let symbol_id = resolve(symbols, &data.coin)?;
    let ctx = data.ctx;
    Ok(MarketEvent::Ticker(Ticker {
        symbol_id,
        mark_px: parse_dec("markPx", &ctx.mark_px)?,
        // Hyperliquid's "oracle" price is what every other venue calls the index: the
        // external spot reference. The normalized field takes the venue-neutral name.
        index_px: Some(parse_dec("oraclePx", &ctx.oracle_px)?),
        mid_px: ctx
            .mid_px
            .as_deref()
            .map(|s| parse_dec("midPx", s))
            .transpose()?,
        funding: Some(Funding {
            rate: parse_dec("funding", &ctx.funding)?,
            interval: ASSUMED_FUNDING_INTERVAL_NS,
        }),
        open_interest: Some(parse_dec("openInterest", &ctx.open_interest)?),
        ts_venue: None,
        ts_ingest,
    }))
}

fn decode_trades(trades: Vec<RawTrade>, symbols: &SymbolMap) -> Result<Vec<MarketEvent>> {
    trades
        .into_iter()
        .map(|t| {
            let symbol_id = resolve(symbols, &t.coin)?;
            let side = match t.side.as_str() {
                "B" => Side::Buy,
                "A" => Side::Sell,
                _ => return Err(DecodeError::Malformed("trades")),
            };
            Ok(MarketEvent::Trade(Trade {
                symbol_id,
                px: parse_dec("px", &t.px)?,
                sz: parse_dec("sz", &t.sz)?,
                side,
                ts_event: t.time * MS_TO_NS,
            }))
        })
        .collect()
}

// ── entry points ─────────────────────────────────────────────────────────────

/// Decode one raw WS text frame into zero or more normalized [`Event`]s.
///
/// Market and account-scoped frames share one socket, and they must share one
/// event stream: a fill has to be orderable against the book update that caused it
/// (see [`axon_core::Event`]), so both are decoded here and published on the same
/// bus rather than through two parallel paths.
///
/// Non-data frames (`subscriptionResponse`, `pong`, `error`, unknown channels)
/// decode to an empty vec — they are not errors. A one-sided `bbo` also yields no
/// event, and unmappable *entries* within a user frame are skipped rather than
/// failing the frame (see [`decode_user_channel`]).
///
/// `ts_ingest` is the moment this frame arrived, in event-time nanoseconds. Every
/// feed that carries a venue timestamp uses that and ignores this argument; the sole
/// exception is `activeAssetCtx`. It is a *parameter* rather than a `SystemClock`
/// read inside the decoder so these functions stay pure and offline-testable — a
/// hidden wall-clock read would make every ticker assertion non-deterministic, and
/// the decoders' testability is the whole reason they are separated from the client.
pub fn decode_ws_message(raw: &str, symbols: &SymbolMap, ts_ingest: Nanos) -> Result<Vec<Event>> {
    let env: Envelope = serde_json::from_str(raw)?;
    let events = match env.channel.as_str() {
        "l2Book" => vec![decode_l2(serde_json::from_value(env.data)?, symbols)?.into()],
        "trades" => decode_trades(serde_json::from_value(env.data)?, symbols)?
            .into_iter()
            .map(Event::from)
            .collect(),
        "bbo" => decode_bbo(serde_json::from_value(env.data)?, symbols)?
            .into_iter()
            .map(Event::from)
            .collect(),
        "candle" => vec![decode_candle(serde_json::from_value(env.data)?, symbols)?.into()],
        "activeAssetCtx" => {
            vec![decode_ticker(serde_json::from_value(env.data)?, symbols, ts_ingest)?.into()]
        }
        // Spot contexts arrive on their own channel with a different `ctx` shape —
        // `circulatingSupply`/`totalSupply` where the perp has funding, oracle price
        // and open interest. Refusing loudly is the point: falling through to the
        // "unknown channel" path would leave a subscription that looks healthy and
        // silently never yields a mark price, and reusing the perp arm would invent a
        // funding rate and an open interest for an instrument that has neither.
        "activeSpotAssetCtx" => return Err(DecodeError::UnsupportedChannel("activeSpotAssetCtx")),
        // Everything else goes to the user decoder, which owns the mapping from
        // reply-channel name to user channel and returns nothing for the frames it
        // does not recognize (subscriptionResponse / pong / error / …). Keeping the
        // list in one place is what stops the `userEvents` → `"user"` rename from
        // being re-derived, and mis-derived, in two files.
        channel => decode_user_channel(channel, &env.data, symbols)?
            .into_iter()
            .map(Event::from)
            .collect(),
    };
    Ok(events)
}

/// Extract the message from a venue error frame: `{"channel":"error","data":"…"}`.
///
/// A rejected subscription is **not** a transport failure — it arrives as an
/// ordinary data frame on a healthy socket. Callers must log it and keep the
/// connection: tearing down and reconnecting on a malformed subscription produces an
/// infinite reconnect loop that never fixes the malformed subscription.
///
/// Returns the **unescaped** message. It allocates, deliberately: an error frame is
/// rare and this is a diagnostics path, so robustness beats saving one allocation.
/// A borrowed `&str` would force hand-scanning for `"data":"` (serde's zero-copy
/// `&str` refuses any string needing unescaping, which is exactly this shape) — and a
/// hand-scan silently returns `None` the day the venue adds a space after the colon,
/// i.e. it fails by *quietly stopping logging*, which is the worst way for a
/// diagnostic to break. `None` for any other channel or unparseable input.
pub fn venue_error(raw: &str) -> Option<String> {
    let env: serde_json::Value = serde_json::from_str(raw).ok()?;
    if env.get("channel")?.as_str()? != "error" {
        return None;
    }
    // Non-string `data` is possible in principle; there is nothing to log, and it is
    // not a panic.
    Some(env.get("data")?.as_str()?.to_owned())
}

/// Decode a REST `POST /info` `l2Book` snapshot (used to seed a book before WS
/// updates apply). The REST body is the bare l2 object, with no channel envelope.
pub fn decode_rest_l2(raw: &str, symbols: &SymbolMap) -> Result<MarketEvent> {
    decode_l2(serde_json::from_str(raw)?, symbols)
}

/// One `meta` response, both halves.
///
/// Deliberately one decode and not two: the asset index and the lot come out of the
/// *same* `universe` array, and two reads of `meta` taken a moment apart can disagree
/// after a listing — at which point an order is signed with one asset's index and the
/// other asset's size. The live fill test already says exactly this about its own two
/// facts; this is where the sentence belongs.
#[derive(Debug, Clone)]
pub struct Universe {
    pub symbols: SymbolMap,
    pub instruments: InstrumentTable,
}

/// Decode a `POST /info {"type":"meta"}` response into both the [`SymbolMap`] and the
/// [`InstrumentTable`].
///
/// The coin at `universe` index `i` becomes `SymbolId(i)`, matching Hyperliquid's
/// asset indexing (`docs/research/hyperliquid-execution.md`), and its `szDecimals`
/// becomes that instrument's price grid and lot (ADR-0025).
///
/// A `szDecimals` that will not build a spec is a [`DecodeError`], **not** a skipped
/// asset. An asset silently absent from the table is one that fails closed at startup
/// with no reason attached to it, which is a session that looks healthy and refuses
/// every order for one instrument.
pub fn decode_universe(raw: &str) -> Result<Universe> {
    #[derive(Deserialize)]
    struct AssetInfo {
        name: String,
        /// Required, not `Option`: an absent lot is not a lot, and defaulting it would
        /// pick whatever number happens to work for BTC and then quietly send an
        /// order on a coarser coin that the venue rejects. The same reasoning as
        /// `PerpCtx`'s three mandatory fields above.
        #[serde(rename = "szDecimals")]
        sz_decimals: u32,
    }
    #[derive(Deserialize)]
    struct Meta {
        universe: Vec<AssetInfo>,
    }
    let meta: Meta = serde_json::from_str(raw)?;
    let mut symbols = SymbolMap::new();
    let mut instruments = InstrumentTable::new();
    for (i, asset) in meta.universe.into_iter().enumerate() {
        let id = SymbolId::new(i as u32);
        instruments.insert(perp_spec(id, &asset.name, asset.sz_decimals)?);
        symbols.insert(asset.name, id);
    }
    Ok(Universe {
        symbols,
        instruments,
    })
}

/// One perp's number format, from the one number the venue publishes about it.
fn perp_spec(id: SymbolId, coin: &str, sz_decimals: u32) -> Result<InstrumentSpec> {
    let bad = |reason| DecodeError::BadPrecision {
        coin: coin.to_string(),
        reason,
    };
    Ok(InstrumentSpec {
        symbol_id: id,
        price: PriceGrid::decimals_with_sig_figs(
            PERP_MAX_DECIMALS.saturating_sub(sz_decimals),
            PRICE_SIG_FIGS,
        )
        .map_err(bad)?,
        size: SizeGrid::decimals(sz_decimals).map_err(bad)?,
        min_notional: Some(MIN_ORDER_NOTIONAL_USD),
    })
}

/// The symbol half alone, for the four callers that only address instruments.
///
/// Kept so `fetch_meta`, the examples and the `#[ignore]`d live tests keep compiling:
/// a wider mechanical diff across the signing-adjacent crate buys nothing here.
pub fn decode_meta(raw: &str) -> Result<SymbolMap> {
    Ok(decode_universe(raw)?.symbols)
}

/// The venue's two non-data frames, captured off the wire rather than imagined.
///
/// One connection to `wss://api.hyperliquid-testnet.xyz/ws` on 2026-07-26: a `bbo`
/// subscribe followed by `{"method":"ping"}`, both replies written down byte for
/// byte. Kept as a file because the *hand-written* version of this fixture is what
/// let the heartbeat bug live for a phase — the test asserted
/// `{"channel":"pong","data":null}`, a frame Hyperliquid does not send, and so it
/// passed happily while every real pong was being logged as a decode error. A
/// decoder is held to what the venue sends, and the only way to know that is to
/// have caught one. [`super::client`]'s `the_venues_non_data_frames_still_match_the_fixture`
/// re-checks this file against the live venue.
#[cfg(test)]
pub(crate) const NON_DATA_FRAMES: &str = include_str!("../../testdata/non-data-frames.jsonl");

/// One captured frame from [`NON_DATA_FRAMES`], by channel.
#[cfg(test)]
pub(crate) fn captured_frame(channel: &str) -> &'static str {
    let key = format!(r#""channel":"{channel}""#);
    NON_DATA_FRAMES
        .lines()
        .find(|line| line.contains(&key))
        .unwrap_or_else(|| panic!("no captured {channel} frame in testdata/non-data-frames.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn symbols() -> SymbolMap {
        SymbolMap::from_perps(["BTC", "ETH"])
    }

    /// A fixed stand-in for "when the frame arrived". Every feed but `activeAssetCtx`
    /// ignores it; pinning it here keeps the ticker tests deterministic, which is the
    /// reason the decoder takes it as an argument instead of reading a clock.
    const TS_INGEST: Nanos = 1_700_000_000_500 * MS_TO_NS;

    /// The dispatcher returns the unified [`Event`] because user channels ride the
    /// same socket; these market-feed tests unwrap back to the market half.
    fn market(ev: &Event) -> &MarketEvent {
        match ev {
            Event::Market(m) => m,
            Event::Exec(e) => panic!("expected a market event, got {e:?}"),
        }
    }

    // Representative Hyperliquid WS frames (see docs/research/hyperliquid-execution.md).
    const L2_FRAME: &str = r#"{"channel":"l2Book","data":{"coin":"BTC","time":1700000000000,
        "levels":[[{"px":"50000.0","sz":"1.5","n":3},{"px":"49999.0","sz":"2.0","n":5}],
                  [{"px":"50001.0","sz":"0.8","n":2},{"px":"50002.0","sz":"1.1","n":4}]]}}"#;

    const TRADES_FRAME: &str = r#"{"channel":"trades","data":[
        {"coin":"BTC","side":"B","px":"50000.5","sz":"0.10","time":1700000000001,"hash":"0xabc","tid":1},
        {"coin":"ETH","side":"A","px":"3000.25","sz":"2.00","time":1700000000002,"hash":"0xdef","tid":2}]}"#;

    const BBO_FRAME: &str = r#"{"channel":"bbo","data":{"coin":"ETH","time":1700000000003,
        "bbo":[{"px":"2999.5","sz":"5.0","n":1},{"px":"3000.5","sz":"4.0","n":2}]}}"#;

    const CANDLE_FRAME: &str = r#"{"channel":"candle","data":{"t":1700000000000,"T":1700000059999,
        "s":"BTC","i":"1m","o":"50000.0","c":"50010.0","h":"50020.0","l":"49990.0","v":"12.5","n":100}}"#;

    /// Three frames for **one** BTC 1-minute bar, captured verbatim off
    /// `wss://api.hyperliquid.xyz/ws` on 2026-07-26 and reproduced byte for byte. The
    /// bar opened at 11:09:00Z; these are the first frame, one from the middle, and the
    /// last the venue ever sent for it. Their receipt times are
    /// [`FORMING_BAR_RECV_MS`], recorded alongside.
    ///
    /// They are here because the thing they prove cannot be shown with a single frame:
    /// `t` and `T` are identical across all three while `c`, `h`, `v` and `n` all move,
    /// so a partial and a close are one event as far as any timestamp is concerned. The
    /// last of them is also the *whole* bar — `candleSnapshot` returned `c=64352.0`,
    /// `v=1.07768` for this minute — and it still arrived two seconds before the close.
    const FORMING_BAR_FRAMES: [&str; 3] = [
        r#"{"channel":"candle","data":{"t":1785053340000,"T":1785053399999,"s":"BTC","i":"1m","o":"64340.0","c":"64340.0","h":"64340.0","l":"64340.0","v":"0.00083","n":1}}"#,
        r#"{"channel":"candle","data":{"t":1785053340000,"T":1785053399999,"s":"BTC","i":"1m","o":"64340.0","c":"64346.0","h":"64347.0","l":"64340.0","v":"0.15124","n":21}}"#,
        r#"{"channel":"candle","data":{"t":1785053340000,"T":1785053399999,"s":"BTC","i":"1m","o":"64340.0","c":"64352.0","h":"64354.0","l":"64340.0","v":"1.07768","n":79}}"#,
    ];

    /// When each of [`FORMING_BAR_FRAMES`] was read off the socket, in venue
    /// milliseconds. The bar's `T` is 1785053399999, so the last frame preceded its own
    /// close time by 1 994 ms.
    const FORMING_BAR_RECV_MS: [i64; 3] = [1785053340448, 1785053357355, 1785053398005];

    /// Captured verbatim off the Hyperliquid **testnet** socket (BTC, 2026-07-25).
    /// Reproduced byte-for-byte rather than tidied, because the two things this feed
    /// gets wrong in practice — the missing `time` field and the extra `dayBaseVlm`
    /// the API reference does not document — are only visible in the real payload.
    const TICKER_FRAME: &str = r#"{"channel":"activeAssetCtx","data":{"coin":"BTC",
        "ctx":{"funding":"0.0000125","openInterest":"63.91844","prevDayPx":"64031.0",
        "dayNtlVlm":"1274069.4620399997","premium":"0.0","oraclePx":"64232.0","markPx":"64254.0",
        "midPx":"64257.5","impactPxs":["63990.0","64674.9"],"dayBaseVlm":"19.8655"}}}"#;

    /// The spot sibling: a near-identical channel name over a different `ctx` — no
    /// funding, no oracle price, no open interest, plus supply figures a perp has no
    /// concept of.
    const SPOT_TICKER_FRAME: &str = r#"{"channel":"activeSpotAssetCtx","data":{"coin":"@1","ctx":{
        "dayNtlVlm":"8906.0","prevDayPx":"0.22916","markPx":"0.23482","midPx":"0.234775",
        "circulatingSupply":"598274725.27","totalSupply":"1000000000.0","coin":"@1"}}}"#;

    #[test]
    fn decodes_l2_snapshot_to_book_event() {
        let evs = decode_ws_message(L2_FRAME, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 1);
        let MarketEvent::Book(snap) = market(&evs[0]) else {
            panic!("expected a Book event");
        };
        assert_eq!(snap.symbol_id, SymbolId::new(0)); // BTC
        assert_eq!(snap.ts_event, 1_700_000_000_000 * 1_000_000);
        assert_eq!(snap.bids[0], Level::new(dec!(50000.0), dec!(1.5)));
        assert_eq!(snap.asks[0], Level::new(dec!(50001.0), dec!(0.8)));
        assert_eq!(snap.bids.len(), 2);
        assert_eq!(snap.asks.len(), 2);
    }

    #[test]
    fn decodes_trade_batch_with_aggressor_sides() {
        let evs = decode_ws_message(TRADES_FRAME, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 2);
        let MarketEvent::Trade(a) = market(&evs[0]) else {
            panic!("expected Trade");
        };
        assert_eq!(a.symbol_id, SymbolId::new(0));
        assert_eq!(a.px, dec!(50000.5));
        assert_eq!(a.side, Side::Buy); // "B"
        let MarketEvent::Trade(b) = market(&evs[1]) else {
            panic!("expected Trade");
        };
        assert_eq!(b.symbol_id, SymbolId::new(1)); // ETH
        assert_eq!(b.side, Side::Sell); // "A"
    }

    #[test]
    fn decodes_bbo_frame() {
        let evs = decode_ws_message(BBO_FRAME, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 1);
        let MarketEvent::Bbo(q) = market(&evs[0]) else {
            panic!("expected Bbo");
        };
        assert_eq!(q.symbol_id, SymbolId::new(1));
        assert_eq!(q.bid_px, dec!(2999.5));
        assert_eq!(q.ask_px, dec!(3000.5));
    }

    #[test]
    fn decodes_candle_frame() {
        let evs = decode_ws_message(CANDLE_FRAME, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 1);
        let MarketEvent::Candle(c) = market(&evs[0]) else {
            panic!("expected Candle");
        };
        assert_eq!(c.symbol_id, SymbolId::new(0)); // BTC
        assert_eq!(c.interval, axon_core::CandleInterval::M1);
        assert_eq!(c.open, dec!(50000.0));
        assert_eq!(c.high, dec!(50020.0));
        assert_eq!(c.low, dec!(49990.0));
        assert_eq!(c.close, dec!(50010.0));
        assert_eq!(c.volume, dec!(12.5));
        assert_eq!(c.open_time, 1_700_000_000_000 * 1_000_000);
        assert_eq!(c.ts_event, 1_700_000_060_000 * 1_000_000); // T + 1 ms: the instant the bar is final
    }

    #[test]
    fn a_candle_is_stamped_one_ms_past_its_last_millisecond_so_it_never_ties_with_its_own_trades() {
        // The venue's `T` is the bar's last millisecond, not the instant after it. A bar
        // stamped `T` sorts equal to a trade printed in that same millisecond, and an
        // event-time sort that breaks the tie the wrong way hands the strategy a closed
        // bar before the tick that closed it — a feature computed on the future.
        let evs = decode_ws_message(CANDLE_FRAME, &symbols(), TS_INGEST).unwrap();
        let MarketEvent::Candle(c) = market(&evs[0]) else {
            panic!("expected Candle");
        };
        let close_time_ms: i64 = 1_700_000_059_999;
        assert_eq!(c.ts_event, (close_time_ms + 1) * MS_TO_NS);
        // Strictly after the bar's own last millisecond, and exactly one interval past
        // the open — a stamp inside the bar would order it among its own inputs.
        assert!(c.ts_event > close_time_ms * MS_TO_NS);
        assert_eq!(c.interval, axon_core::CandleInterval::M1);
        assert_eq!(c.ts_event, c.open_time + 60 * 1_000 * MS_TO_NS);
    }

    #[test]
    fn a_republished_forming_bar_is_stamped_with_a_close_that_has_not_happened() {
        // The hazard, from real frames: the venue pushes the bar it is still filling and
        // stamps every push with the moment that bar *will* close. Anything treating
        // that stamp as "what time it is" — the core's clock did, until
        // `axon_runtime::handler::advances_the_clock` — runs up to a whole interval
        // ahead of the market. There, it ages every signal on the ring against the
        // future and refuses it as expired, and leaves the pass schedule anchored past
        // any event that will arrive, so the strategy silently stops trading.
        let mut decoded = Vec::new();
        for (frame, recv_ms) in FORMING_BAR_FRAMES.iter().zip(FORMING_BAR_RECV_MS) {
            let evs = decode_ws_message(frame, &symbols(), recv_ms * MS_TO_NS).unwrap();
            let MarketEvent::Candle(c) = market(&evs[0]) else {
                panic!("expected Candle");
            };
            assert!(
                c.ts_event > recv_ms * MS_TO_NS,
                "the ordering key is in this frame's own future: \
                 ts_event {} vs receipt {}",
                c.ts_event,
                recv_ms * MS_TO_NS
            );
            decoded.push(c.clone());
        }

        // One bar, three observations of it — and the two fields any consumer could
        // order or de-duplicate on are byte-identical across all three. That is the
        // whole finding: a partial and a close are indistinguishable here, so no
        // consumer of a *single* frame can know which it holds.
        for c in &decoded[1..] {
            assert_eq!(c.open_time, decoded[0].open_time);
            assert_eq!(c.ts_event, decoded[0].ts_event, "same T on every frame");
            assert_eq!(c.symbol_id, decoded[0].symbol_id);
            assert_eq!(c.interval, CandleInterval::M1);
        }
        // …while the bar itself moved under them, so these are genuinely three different
        // states and not a repeated frame.
        assert_eq!(decoded[0].close, dec!(64340.0));
        assert_eq!(decoded[2].close, dec!(64352.0));
        assert_eq!(decoded[0].volume, dec!(0.00083));
        assert_eq!(decoded[2].volume, dec!(1.07768));
        assert!(decoded[2].high > decoded[0].high);

        // The last frame is the bar the venue's own `candleSnapshot` reports for this
        // minute — and it arrived 1 994 ms *before* the close it is stamped with. So
        // "wait for a frame at or after `T`" is not a rule that yields bars on this
        // venue; it yields nothing, for this bar and for 65 of the 69 measured.
        let close_ns = decoded[2].ts_event;
        assert_eq!(
            close_ns - FORMING_BAR_RECV_MS[2] * MS_TO_NS,
            1_995 * MS_TO_NS,
            "T + 1 ms, less the receipt time of the last frame the venue sent"
        );
        assert_eq!(decoded[2].open_time + 60 * 1_000 * MS_TO_NS, close_ns);
    }

    fn ticker_from(frame: &str) -> axon_core::Ticker {
        let evs = decode_ws_message(frame, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 1);
        let MarketEvent::Ticker(t) = market(&evs[0]) else {
            panic!("expected Ticker");
        };
        t.clone()
    }

    #[test]
    fn decodes_active_asset_ctx_frame() {
        let t = ticker_from(TICKER_FRAME);
        assert_eq!(t.symbol_id, SymbolId::new(0)); // BTC
        assert_eq!(t.mark_px, dec!(64254.0));
        assert_eq!(t.mid_px, Some(dec!(64257.5)));
        // Hyperliquid's "oracle" price is the venue-neutral index price.
        assert_eq!(t.index_px, Some(dec!(64232.0)));
        assert_eq!(t.open_interest, Some(dec!(63.91844)));
        let f = t.funding.expect("perps fund");
        assert_eq!(f.rate, dec!(0.0000125));
        assert_eq!(
            f.interval, ASSUMED_FUNDING_INTERVAL_NS,
            "Hyperliquid's rate is per hour, and the interval must travel with it"
        );
        // Three genuinely different prices in the same frame. Any pair of them
        // collapsed together still yields a Ticker that reads as correct, so they are
        // asserted apart rather than only asserted present.
        assert_ne!(t.mark_px, t.mid_px.unwrap());
        assert_ne!(t.mark_px, t.index_px.unwrap());
    }

    #[test]
    fn a_ticker_is_ingest_stamped_because_the_venue_sends_no_time() {
        // The frame genuinely has no `time` field. Recording that honestly is what
        // keeps a replay from silently reordering against a live capture; inventing a
        // venue timestamp here would hide it from every consumer.
        assert!(
            !TICKER_FRAME.contains("\"time\""),
            "the fixture must keep the venue's actual shape"
        );
        let t = ticker_from(TICKER_FRAME);
        assert_eq!(t.ts_venue, None);
        assert_eq!(t.ts_ingest, TS_INGEST);
        assert_eq!(t.ts_event(), TS_INGEST);
        assert!(!t.is_venue_timed());

        // The bus orders on that fallback, not on a default 0 — which would sort every
        // ticker to the very front of the deterministic queue.
        let evs = decode_ws_message(TICKER_FRAME, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs[0].ts_event(), TS_INGEST);
    }

    #[test]
    fn the_stamped_funding_interval_is_the_one_the_cadence_check_verifies() {
        use crate::ws::funding::decode_funding_cadence;

        // The frame carries a rate and no period, so the period is stamped from a
        // constant — and that is only safe while it is the *same* constant the venue
        // is checked against. If the stamp and the check drift apart, the live
        // cross-check keeps passing while every Ticker carries a different number and
        // carry is wrong by the ratio, which is precisely the silent failure the
        // named constant exists to prevent.
        assert!(
            !TICKER_FRAME.contains("nterval"),
            "the venue would have to start sending a period for this to be decoded"
        );
        let stamped = ticker_from(TICKER_FRAME)
            .funding
            .expect("perps fund")
            .interval;
        let gap_ms = stamped / MS_TO_NS;
        let two_periods = |gap: i64| format!(r#"[{{"time":0}},{{"time":{gap}}}]"#);

        assert!(
            !decode_funding_cadence(&two_periods(gap_ms))
                .unwrap()
                .differs(),
            "a venue funding at exactly the stamped period must read as agreeing"
        );
        assert!(
            decode_funding_cadence(&two_periods(gap_ms / 2))
                .unwrap()
                .differs(),
            "a venue funding twice as often must be flagged, not absorbed"
        );
    }

    #[test]
    fn a_null_mid_is_none_and_is_never_filled_in_from_the_mark() {
        // Hyperliquid nulls `midPx` on a one-sided book. Substituting `markPx` would
        // hand a strategy a "mid" that no resting order supports.
        let frame = r#"{"channel":"activeAssetCtx","data":{"coin":"BTC","ctx":{
            "dayNtlVlm":"1.0","prevDayPx":"64031.0","markPx":"64254.0","midPx":null,
            "funding":"0.0000125","openInterest":"63.91844","oraclePx":"64232.0",
            "premium":"0.0","impactPxs":null}}}"#;
        let t = ticker_from(frame);
        assert_eq!(t.mid_px, None);
        assert_eq!(t.mark_px, dec!(64254.0), "the mark is still reported");

        // An omitted `midPx` must mean the same thing as an explicit null: the venue's
        // shared context type declares the field optional, so both shapes are on the
        // wire and only one of them is exercised by the frame above.
        let omitted = r#"{"channel":"activeAssetCtx","data":{"coin":"BTC","ctx":{
            "markPx":"64254.0","funding":"0.0000125","openInterest":"63.91844",
            "oraclePx":"64232.0"}}}"#;
        assert_eq!(ticker_from(omitted).mid_px, None);
    }

    #[test]
    fn spot_asset_ctx_is_refused_instead_of_decoded_as_a_perp() {
        // Two channels one word apart, two different `ctx` shapes. Silently ignoring
        // the spot frame would leave a subscription that looks healthy and never
        // produces a price; decoding it as a perp would invent funding and open
        // interest for an instrument that has neither.
        let err = decode_ws_message(SPOT_TICKER_FRAME, &symbols(), TS_INGEST).unwrap_err();
        assert!(
            matches!(err, DecodeError::UnsupportedChannel("activeSpotAssetCtx")),
            "got {err:?}"
        );
    }

    #[test]
    fn a_spot_context_on_the_perp_channel_fails_rather_than_losing_three_fields() {
        // Second line of defence, independent of the channel name: `funding`,
        // `oraclePx` and `openInterest` are required, so a spot-shaped ctx cannot
        // decode into a plausible perp ticker with those three quietly absent.
        let frame = r#"{"channel":"activeAssetCtx","data":{"coin":"BTC","ctx":{
            "dayNtlVlm":"8906.0","prevDayPx":"0.22916","markPx":"0.23482","midPx":"0.234775",
            "circulatingSupply":"598274725.27","totalSupply":"1000000000.0"}}}"#;
        assert!(matches!(
            decode_ws_message(frame, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::Json(_)
        ));
    }

    #[test]
    fn unknown_candle_interval_is_an_error() {
        let frame = r#"{"channel":"candle","data":{"t":1,"T":2,"s":"BTC","i":"3m",
            "o":"1","c":"1","h":"1","l":"1","v":"1","n":1}}"#;
        assert!(matches!(
            decode_ws_message(frame, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::Malformed("candle")
        ));
    }

    #[test]
    fn one_sided_bbo_yields_no_event() {
        let frame = r#"{"channel":"bbo","data":{"coin":"BTC","time":1,"bbo":[{"px":"100","sz":"1","n":1},null]}}"#;
        assert!(decode_ws_message(frame, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn subscription_response_and_pong_are_ignored() {
        let sub = captured_frame("subscriptionResponse");
        let pong = captured_frame("pong");
        // The finding, pinned so it cannot be re-imagined: the heartbeat reply has no
        // `data` field at all. The previous hand-written `{"channel":"pong","data":null}`
        // is not a frame this venue sends, and asserting on it is what let 46 spurious
        // `WS decode error (frame dropped)` lines through a 1 h 44 m soak.
        assert_eq!(pong, r#"{"channel":"pong"}"#);
        assert_eq!(
            pong.len(),
            18,
            "eighteen bytes, and none of them are `data`"
        );
        for frame in [sub, pong] {
            assert!(
                decode_ws_message(frame, &symbols(), TS_INGEST)
                    .expect("a non-data frame is not a decode error")
                    .is_empty(),
                "{frame} should decode to nothing at all"
            );
        }
    }

    #[test]
    fn a_data_channel_with_no_payload_is_still_an_error() {
        // The guard on defaulting `Envelope::data`. Tolerating an absent payload is
        // for the heartbeat and nothing else: a book frame that arrived without its
        // levels must fail loudly rather than decode into an empty book, which
        // downstream would read as a market with no bids and no asks.
        for frame in [
            r#"{"channel":"l2Book"}"#,
            r#"{"channel":"bbo"}"#,
            r#"{"channel":"activeAssetCtx"}"#,
        ] {
            assert!(
                matches!(
                    decode_ws_message(frame, &symbols(), TS_INGEST).unwrap_err(),
                    DecodeError::Json(_)
                ),
                "{frame} must not decode"
            );
        }
    }

    #[test]
    fn unknown_coin_is_an_error() {
        let frame = r#"{"channel":"l2Book","data":{"coin":"DOGE","time":1,"levels":[[],[]]}}"#;
        let err = decode_ws_message(frame, &symbols(), TS_INGEST).unwrap_err();
        assert!(matches!(err, DecodeError::UnknownCoin(c) if c == "DOGE"));
    }

    #[test]
    fn bad_price_string_is_an_error() {
        let frame = r#"{"channel":"l2Book","data":{"coin":"BTC","time":1,"levels":[[{"px":"NaN","sz":"1"}],[]]}}"#;
        assert!(matches!(
            decode_ws_message(frame, &symbols(), TS_INGEST).unwrap_err(),
            DecodeError::BadNumber { field: "px", .. }
        ));
    }

    const META_BODY: &str = r#"{"universe":[
            {"name":"BTC","szDecimals":5,"maxLeverage":50},
            {"name":"ETH","szDecimals":4,"maxLeverage":50},
            {"name":"SOL","szDecimals":2,"maxLeverage":20}]}"#;

    #[test]
    fn decodes_meta_into_indexed_symbol_map() {
        let map = decode_meta(META_BODY).unwrap();
        assert_eq!(map.id("BTC"), Some(SymbolId::new(0)));
        assert_eq!(map.id("ETH"), Some(SymbolId::new(1)));
        assert_eq!(map.id("SOL"), Some(SymbolId::new(2)));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn the_universe_yields_the_lot_and_the_asset_index_from_one_response() {
        // `szDecimals` is on the wire and was being thrown away, so nothing above the
        // adapter could answer "how many decimals may a BTC size have?" — and every
        // computed price and size went out unrounded. Both facts come out of the same
        // array read, because two reads of `meta` can disagree after a listing.
        let u = decode_universe(META_BODY).unwrap();
        assert_eq!(u.symbols.id("BTC"), Some(SymbolId::new(0)));
        assert_eq!(u.instruments.len(), 3);

        let btc = u.instruments.get(SymbolId::new(0)).expect("BTC has a grid");
        assert_eq!(btc.size.increment(), dec!(0.00001), "szDecimals 5");
        // 6 - 5 = 1 decimal place, and five significant figures on top of that: at a
        // real BTC price the significant-figure rule is what binds, and it clamps to
        // the integer exemption.
        assert_eq!(btc.price.tick_at(dec!(1234.5)), dec!(0.1));
        assert_eq!(btc.price.tick_at(dec!(108234)), dec!(1));
        assert!(btc.price.is_valid(dec!(108234)));
        assert!(!btc.price.is_valid(dec!(108234.567)));
        assert_eq!(btc.min_notional, Some(crate::MIN_ORDER_NOTIONAL_USD));

        let sol = u.instruments.get(SymbolId::new(2)).expect("SOL has a grid");
        assert_eq!(sol.size.increment(), dec!(0.01), "szDecimals 2");
        assert_ne!(
            sol.size.increment(),
            btc.size.increment(),
            "two coins on one venue do not share a lot - which is the whole reason \
             this is not a Capabilities field"
        );
    }

    #[test]
    fn an_asset_with_no_sz_decimals_is_refused_rather_than_given_btcs_precision() {
        // Defaulting here is the dangerous version of the bug: the missing field would
        // silently take whatever number happens to suit BTC, and every order on a
        // coarser coin would come back `tickRejected` with nothing in the log saying
        // the lot was invented. The field is required, so the response fails to decode.
        let body = r#"{"universe":[{"name":"BTC","maxLeverage":50}]}"#;
        let err = decode_universe(body).unwrap_err();
        assert!(
            matches!(err, DecodeError::Json(_)),
            "a missing lot must fail the decode: {err:?}"
        );
        assert!(
            err.to_string().contains("szDecimals"),
            "and say which field: {err}"
        );
        // The wrapper cannot be the way around it either.
        assert!(decode_meta(body).is_err());
    }

    #[test]
    fn an_asset_whose_declared_precision_is_not_a_grid_names_the_coin() {
        // These numbers come out of a venue's JSON. A 200-decimal lot must be a decode
        // error naming the asset, not a panic in a live session and not an asset
        // quietly absent from the table.
        let body = r#"{"universe":[{"name":"BTC","szDecimals":5},
                                   {"name":"WAT","szDecimals":200}]}"#;
        let err = decode_universe(body).unwrap_err();
        assert!(
            matches!(err, DecodeError::BadPrecision { ref coin, .. } if coin == "WAT"),
            "got {err:?}"
        );
    }

    #[test]
    fn user_channel_frames_route_to_exec_events_on_the_same_bus() {
        // The point of the unified `Event`: one dispatcher, one stream.
        let frame = r#"{"channel":"orderUpdates","data":[{"order":{"coin":"BTC","side":"B",
            "limitPx":"64020.0","sz":"1.0","oid":502464968365,"timestamp":1784976929583,
            "origSz":"1.0"},"status":"open","statusTimestamp":1784976929583}]}"#;
        let evs = decode_ws_message(frame, &symbols(), TS_INGEST).unwrap();
        assert_eq!(evs.len(), 1);
        assert!(matches!(
            &evs[0],
            Event::Exec(axon_core::ExecEvent::Order(_))
        ));
        assert_eq!(evs[0].ts_event(), 1_784_976_929_583 * 1_000_000);
    }

    #[test]
    fn venue_error_extracts_the_message_and_emits_no_event() {
        // A bad subscription is an ordinary frame on a healthy socket.
        let frame = r#"{"channel":"error","data":"Error parsing JSON into valid websocket request: {\"method\":\"subscribe\",...}"}"#;
        let msg = venue_error(frame).expect("error frames carry a message");
        assert!(
            msg.starts_with("Error parsing JSON into valid websocket request:"),
            "got {msg:?}"
        );
        assert!(
            msg.contains("subscribe"),
            "the venue echoes our request back: {msg:?}"
        );
        // Unescaped, so the echoed request is readable in a log rather than a wall of
        // backslashes.
        assert!(
            msg.contains(r#""method":"subscribe""#),
            "escapes should be resolved: {msg:?}"
        );
        assert!(!msg.contains('\\'), "no stray backslashes left: {msg:?}");
        // The connection stays up precisely because this is not a decode failure.
        assert!(decode_ws_message(frame, &symbols(), TS_INGEST)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn venue_error_ignores_every_other_frame() {
        assert_eq!(venue_error(L2_FRAME), None);
        assert_eq!(
            venue_error(r#"{"channel":"subscriptionResponse","data":{"method":"subscribe"}}"#),
            None
        );
        assert_eq!(venue_error("not json at all"), None);
        // Well-formed error channel, non-string data: nothing to log, not a panic.
        assert_eq!(
            venue_error(r#"{"channel":"error","data":{"code":1}}"#),
            None
        );
    }

    #[test]
    fn venue_error_survives_whitespace_in_the_frame() {
        // Regression: a hand-scan for the literal `"data":"` returns None the day the
        // venue pretty-prints, which would silently stop error logging. Parse properly.
        let spaced = "{ \"channel\" : \"error\" , \"data\" : \"bad subscription\" }";
        assert_eq!(venue_error(spaced).as_deref(), Some("bad subscription"));
    }

    #[test]
    fn decodes_rest_snapshot() {
        let body = r#"{"coin":"ETH","time":1700000000000,"levels":[[{"px":"3000","sz":"1"}],[{"px":"3001","sz":"2"}]]}"#;
        let MarketEvent::Book(snap) = decode_rest_l2(body, &symbols()).unwrap() else {
            panic!("expected Book");
        };
        assert_eq!(snap.symbol_id, SymbolId::new(1));
        assert_eq!(snap.bids[0], Level::new(dec!(3000), dec!(1)));
    }
}
