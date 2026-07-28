//! Pure decoders for Hyperliquid's **user** (account-scoped) WS channels →
//! normalized [`ExecEvent`]s.
//!
//! This is the execution mirror of [`super::decode`] and obeys the same rules: no
//! I/O, [`Decimal`] parsed from the venue's decimal *strings* (never `f64`),
//! timestamps in **ms** converted to event-time **ns**, coin names resolved through
//! the [`SymbolMap`]. Everything here is unit-tested against captured frames.
//!
//! Four venue facts shape this file, each of which has silently broken an
//! integration before:
//!
//! - **There is no auth on these channels.** A subscription carries only a `user`
//!   address — no signature, no token. That address must be the **master** (or
//!   sub-account) address; passing the agent/API-wallet address that *signs* orders
//!   returns empty results forever, with no error.
//! - **`userEvents` frames arrive on channel `"user"`**, not `"userEvents"`. The
//!   subscription type and the reply channel name differ for this one feed, so the
//!   two names live together in [`UserChannel`] rather than being written out here.
//! - **Only `userFills` ever snapshots.** `userEvents` is purely incremental (an
//!   idle account produces zero frames) and `orderUpdates` has no `isSnapshot`
//!   field at all. Resting-order state after a reconnect must therefore be
//!   recovered from `POST /info` (`openOrders` / `frontendOpenOrders`), never by
//!   waiting on this channel — waiting is an infinite wait.
//! - **Fill batches are heterogeneous.** Because user channels are account-scoped
//!   rather than coin-scoped, one frame can name coins we never subscribed to
//!   (HIP-3 dex-prefixed perps like `"xyz:SP500"`, spot pairs like `"@107"`). Such
//!   entries are skipped individually and logged; a single unmapped coin must not
//!   drop a batch of good fills.

use axon_core::{
    CancelReason, Cloid, Decimal, ExecEvent, Fill, Liquidity, OrderId, OrderStatus, OrderUpdate,
    Side,
};
use serde::Deserialize;
use serde_json::Value;

use super::decode::{parse_dec, DecodeError, MS_TO_NS};
use super::sub::UserChannel;
use crate::symbol_map::SymbolMap;

type Result<T> = std::result::Result<T, DecodeError>;

// ── on-wire shapes ───────────────────────────────────────────────────────────
//
// Fields the normalized types have no home for (`dir`, `hash`, `startPosition`,
// `feeToken`, `builderFee`, `twapId`, `liquidation`, `timestamp`, …) are simply not
// modeled: serde ignores unknown keys, and an extra field the venue adds tomorrow
// must never fail a frame. Two of them are worth naming for the trap they set —
// `dir` ("Open Long", "Close Short") is a *display* string that nothing may branch
// on, and `hash` is all-zeros for TWAP and some internal fills, so it is never an
// identity key. `tid` is the identity key.

#[derive(Deserialize)]
struct RawFill {
    coin: String,
    px: String,
    sz: String,
    /// `"B"` = bid ⇒ we bought; `"A"` = ask ⇒ we sold.
    side: String,
    /// Execution time, epoch ms.
    time: i64,
    oid: u64,
    /// Undocumented on this payload but real: a hex string, `null`, or absent
    /// entirely, hence `Option` plus `default`.
    #[serde(default)]
    cloid: Option<String>,
    /// The venue's taker flag: `true` ⇒ we crossed the spread.
    crossed: bool,
    /// Signed and inclusive of any builder fee; negative is a maker rebate.
    #[serde(default)]
    fee: Option<String>,
    #[serde(rename = "closedPnl", default)]
    closed_pnl: Option<String>,
    /// Unique execution id — the tracker's dedup key across reconnects and
    /// snapshot replays.
    tid: u64,
}

/// `userEvents` data is an externally-tagged union carrying exactly **one** key per
/// message: `fills`, `funding`, `liquidation` (snake_case inside, unlike the rest of
/// the API), or `nonUserCancel`. Only `fills` is modeled; the others leave `fills`
/// as `None` and produce no event.
///
/// `nonUserCancel` produces nothing **on purpose**. It carries only `{coin, oid}` —
/// no sizes — and the very same cancellation arrives fully-formed on `orderUpdates`
/// moments later. Synthesizing an [`OrderUpdate`] here would mean inventing
/// `orig_qty: 0`, which corrupts the tracker's monotonic fill accounting; arriving a
/// few milliseconds earlier is not worth a wrong quantity.
#[derive(Deserialize)]
struct UserEventsData {
    #[serde(default)]
    fills: Option<Vec<RawFill>>,
}

/// `userFills` data. `isSnapshot` is deliberately absent from this struct and read
/// separately by [`fills_is_snapshot`]: it changes nothing about decoding, because
/// snapshot fills are ordinary fills that the tracker dedups on `tid`.
#[derive(Deserialize)]
struct UserFillsData {
    fills: Vec<RawFill>,
}

#[derive(Deserialize)]
struct RawOrder {
    coin: String,
    side: String,
    /// Absent for order kinds the venue reports without one.
    #[serde(rename = "limitPx", default)]
    limit_px: Option<String>,
    /// **Remaining** size, not the submitted size.
    sz: String,
    oid: u64,
    /// May be absent or `null`.
    #[serde(default)]
    cloid: Option<String>,
    /// The size the order was submitted with.
    #[serde(rename = "origSz")]
    orig_sz: String,
}

#[derive(Deserialize)]
struct RawOrderUpdate {
    order: RawOrder,
    /// One of ~29 venue strings; see [`hl_status`].
    status: String,
    /// When the *status* changed, epoch ms. The nested `order.timestamp` is
    /// order-*creation* time, so using it would order status transitions by when
    /// the order was born instead of when it changed — every update for one order
    /// would share a timestamp.
    #[serde(rename = "statusTimestamp")]
    status_timestamp: i64,
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// `"B"` = bid, `"A"` = ask. The venue uses this pair on fills and orders alike.
fn hl_side(s: &str) -> Option<Side> {
    match s {
        "B" => Some(Side::Buy),
        "A" => Some(Side::Sell),
        _ => None,
    }
}

/// Optional decimal string → [`Decimal`], reading absent/`null` as zero.
///
/// Dropping an entire fill because the venue omitted `fee` or `closedPnl` would
/// corrupt position accounting; reporting the missing number as zero is the smaller
/// error, and the periodic account snapshot is what catches any resulting drift.
fn parse_opt_dec(field: &'static str, s: Option<&String>) -> Result<Decimal> {
    match s {
        Some(v) => parse_dec(field, v),
        None => Ok(Decimal::ZERO),
    }
}

/// An entry we cannot map is skipped, not fatal — but never silently. A frame
/// naming an unmapped coin is normal (account-scoped feeds report every asset we
/// touch), yet it is also exactly how a genuinely missing [`SymbolMap`] entry would
/// present, so it has to be visible.
fn skipped(what: &str, detail: &str) {
    eprintln!("hyperliquid: skipping {what} ({detail})");
}

// ── public classifiers ───────────────────────────────────────────────────────

/// Map a Hyperliquid order-status string to our `(status, cancel_reason)` pair.
///
/// The venue's set is **open** — it has grown repeatedly, and is 29 values as of
/// 2026-07-25 — so the exact table below is followed by a `*Rejected` / `*Canceled`
/// suffix fallback, and anything still unrecognized returns `None`.
///
/// `None` means *the caller emits no event*, and that is the only safe default: a
/// status coerced to the wrong state either makes a dead order look live (so we
/// wait forever on an order that will never fill, and never re-post) or makes a
/// live order look dead (so we double up on exposure we already have). A missing
/// update is recoverable — the `POST /info` reconcile corrects it — while a wrong
/// one silently poisons the tracker.
pub fn hl_status(s: &str) -> Option<(OrderStatus, Option<CancelReason>)> {
    Some(match s {
        // Working states. `triggered` means a stop/take-profit *activated* and is
        // now a live resting order — not that it filled.
        "open" | "triggered" => (OrderStatus::Resting, None),
        "filled" => (OrderStatus::Filled, None),
        "rejected" => (OrderStatus::Rejected, None),
        // Cancels differ only in *why*, and the why is operationally load-bearing:
        // our own cancel is routine, a dead-man's switch firing means we lost the
        // heartbeat, and a liquidation is an incident.
        "canceled" => (OrderStatus::Cancelled, Some(CancelReason::Requested)),
        "scheduledCancel" => (OrderStatus::Cancelled, Some(CancelReason::DeadMansSwitch)),
        "liquidatedCanceled" => (OrderStatus::Cancelled, Some(CancelReason::Liquidation)),
        "marginCanceled"
        | "vaultWithdrawalCanceled"
        | "openInterestCapCanceled"
        | "selfTradeCanceled"
        | "reduceOnlyCanceled"
        | "siblingFilledCanceled"
        | "delistedCanceled" => (OrderStatus::Cancelled, Some(CancelReason::Venue)),
        // The ~20 `*Rejected` variants (tickRejected, badAloPxRejected,
        // oracleRejected, positionFlipAtOpenInterestCapRejected, …) all mean the
        // same thing to us, so the suffix arm covers them rather than an
        // enumeration that would go stale on the venue's next release. Note
        // lowercase `"rejected"` is matched exactly above — it has no suffix.
        _ if s.ends_with("Rejected") => (OrderStatus::Rejected, None),
        // `Cancel` as well as `Canceled`: `scheduledCancel` shows the venue is not
        // consistent about the participle.
        _ if s.ends_with("Canceled") || s.ends_with("Cancel") => {
            (OrderStatus::Cancelled, Some(CancelReason::Unspecified))
        }
        _ => return None,
    })
}

/// Parse the venue's client order id — a 16-byte hex value, with or without the
/// `0x` prefix — into a [`Cloid`].
///
/// A wrong-length or non-hex value yields `None`, i.e. "this fill has no client
/// id", rather than an error. An order we cannot correlate back to our own request
/// is still an order whose fill we must account for, so failing the frame would be
/// strictly worse than losing the correlation.
pub fn parse_cloid(s: &str) -> Option<Cloid> {
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if hex.len() != 32 {
        return None;
    }
    u128::from_str_radix(hex, 16).ok().map(Cloid::new)
}

/// Whether a `userFills` frame is the initial replay.
///
/// The first frame after subscribing carries `isSnapshot: true`; later frames carry
/// `false` **or omit the field entirely**, so absence must read as `false`. This is
/// informational only — snapshot fills are decoded like any other and deduped
/// downstream on [`Fill::trade_id`], which is precisely why the tracker dedups.
pub fn fills_is_snapshot(data: &Value) -> bool {
    data.get("isSnapshot")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

// ── per-channel decoders ─────────────────────────────────────────────────────

/// One wire fill → one [`ExecEvent::Fill`]. `Ok(None)` when the coin is unmapped.
fn decode_fill(raw: &RawFill, symbols: &SymbolMap) -> Result<Option<ExecEvent>> {
    let Some(symbol_id) = symbols.id(&raw.coin) else {
        skipped("fill", &format!("unmapped coin {:?}", raw.coin));
        return Ok(None);
    };
    // An unreadable side is different in kind from an unmapped coin: we cannot tell
    // whether we bought or sold, so there is no safe partial answer.
    let side = hl_side(&raw.side).ok_or(DecodeError::Malformed("fill"))?;
    Ok(Some(ExecEvent::Fill(Fill {
        symbol_id,
        order_id: OrderId::new(raw.oid),
        cloid: raw.cloid.as_deref().and_then(parse_cloid),
        side,
        qty: parse_dec("sz", &raw.sz)?,
        price: parse_dec("px", &raw.px)?,
        fee: parse_opt_dec("fee", raw.fee.as_ref())?,
        closed_pnl: parse_opt_dec("closedPnl", raw.closed_pnl.as_ref())?,
        liquidity: if raw.crossed {
            Liquidity::Taker
        } else {
            Liquidity::Maker
        },
        trade_id: raw.tid,
        ts_event: raw.time * MS_TO_NS,
    })))
}

fn decode_fills(fills: &[RawFill], symbols: &SymbolMap) -> Result<Vec<ExecEvent>> {
    let mut out = Vec::with_capacity(fills.len());
    for raw in fills {
        out.extend(decode_fill(raw, symbols)?);
    }
    Ok(out)
}

/// One wire order update → one [`ExecEvent::Order`]. `Ok(None)` when the coin is
/// unmapped or the status is unclassifiable (see [`hl_status`]).
fn decode_order_update(raw: &RawOrderUpdate, symbols: &SymbolMap) -> Result<Option<ExecEvent>> {
    let Some(symbol_id) = symbols.id(&raw.order.coin) else {
        skipped(
            "order update",
            &format!("unmapped coin {:?}", raw.order.coin),
        );
        return Ok(None);
    };
    let Some((status, cancel_reason)) = hl_status(&raw.status) else {
        skipped("order update", &format!("unknown status {:?}", raw.status));
        return Ok(None);
    };
    let side = hl_side(&raw.order.side).ok_or(DecodeError::Malformed("order update"))?;
    Ok(Some(ExecEvent::Order(OrderUpdate {
        symbol_id,
        order_id: OrderId::new(raw.order.oid),
        cloid: raw.order.cloid.as_deref().and_then(parse_cloid),
        side,
        // Note there is no `PartiallyFilled` mapping: the venue reports `open` for a
        // partially-filled resting order too. The distinction is derivable — and is
        // derived, by `OrderUpdate::filled_qty()` — from the two quantities below,
        // so inferring it here would only add a second source of truth.
        status,
        price: raw
            .order
            .limit_px
            .as_deref()
            .map(|p| parse_dec("limitPx", p))
            .transpose()?,
        orig_qty: parse_dec("origSz", &raw.order.orig_sz)?,
        remaining_qty: parse_dec("sz", &raw.order.sz)?,
        cancel_reason,
        ts_event: raw.status_timestamp * MS_TO_NS,
    })))
}

/// Decode the `data` of one user-channel frame into zero or more [`ExecEvent`]s.
///
/// `channel` is the frame's own `channel` value — `"user"` (what a `userEvents`
/// subscription actually replies on), `"userFills"`, or `"orderUpdates"`. Any other
/// channel yields an empty vec so the top-level dispatcher can hand everything it
/// does not recognize straight here, without maintaining a second name table.
pub fn decode_user_channel(
    channel: &str,
    data: &Value,
    symbols: &SymbolMap,
) -> Result<Vec<ExecEvent>> {
    match UserChannel::from_reply_channel(channel) {
        Some(UserChannel::UserEvents) => match UserEventsData::deserialize(data)?.fills {
            Some(fills) => decode_fills(&fills, symbols),
            // funding / liquidation / nonUserCancel — tolerated, nothing to emit.
            None => Ok(vec![]),
        },
        Some(UserChannel::UserFills) => {
            decode_fills(&UserFillsData::deserialize(data)?.fills, symbols)
        }
        Some(UserChannel::OrderUpdates) => {
            // Unlike every other channel, this `data` is a **bare array** — no
            // wrapper object and no `isSnapshot` field anywhere.
            let updates = Vec::<RawOrderUpdate>::deserialize(data)?;
            let mut out = Vec::with_capacity(updates.len());
            for raw in &updates {
                out.extend(decode_order_update(raw, symbols)?);
            }
            Ok(out)
        }
        None => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::SymbolId;
    use rust_decimal_macros::dec;

    /// Includes a HIP-3 dex-prefixed perp: those coin names are mapped like any
    /// other, they just do not come from `meta.universe`.
    fn symbols() -> SymbolMap {
        let mut m = SymbolMap::from_perps(["BTC", "ETH"]);
        m.insert("xyz:SP500", SymbolId::new(42));
        m
    }

    /// Split a captured frame the way the top-level dispatcher does.
    fn decode_frame(raw: &str, symbols: &SymbolMap) -> Result<Vec<ExecEvent>> {
        let v: Value = serde_json::from_str(raw).unwrap();
        let channel = v["channel"].as_str().unwrap().to_string();
        decode_user_channel(&channel, &v["data"], symbols)
    }

    fn data_of(raw: &str) -> Value {
        serde_json::from_str::<Value>(raw).unwrap()["data"].clone()
    }

    // ── captured frames (live-probed 2026-07-25) ─────────────────────────────

    /// `userEvents` replies on channel `"user"`, and its data is a one-key union.
    const USER_EVENTS_FILL: &str = r#"{"channel":"user","data":{"fills":[{"coin":"BTC","px":"64007.0","sz":"0.00018","side":"A","time":1784976932073,"startPosition":"-50.81882","dir":"Open Short","closedPnl":"0.0","hash":"0x0000000000000000000000000000000000000000000000000000000000000000","oid":502464503020,"crossed":false,"fee":"0.000829","tid":351400627290190,"feeToken":"USDC","twapId":null}]}}"#;

    const USER_FILLS_SNAPSHOT: &str = r#"{"channel":"userFills","data":{"isSnapshot":true,"user":"0x9be9...522f","fills":[{"coin":"xyz:SP500","px":"7429.1","sz":"1.149","side":"B","time":1784916125478,"startPosition":"0.0","dir":"Open Long","closedPnl":"0.0","hash":"0xa7bc...7e68","oid":502150498105,"crossed":false,"fee":"4.489271","builderFee":"4.268017","tid":955441961359817,"feeToken":"USDC","twapId":null}]}}"#;

    const ORDER_UPDATES: &str = r#"{"channel":"orderUpdates","data":[{"order":{"coin":"BTC","side":"B","limitPx":"64020.0","sz":"1.17163","oid":502464968365,"timestamp":1784976929583,"origSz":"1.17163","cloid":"0xffabb2d185b75ea5ca0a5cd346b21507"},"status":"badAloPxRejected","statusTimestamp":1784976929583}]}"#;

    // ── fills ────────────────────────────────────────────────────────────────

    #[test]
    fn user_events_frame_decodes_one_fill() {
        let evs = decode_frame(USER_EVENTS_FILL, &symbols()).unwrap();
        assert_eq!(evs.len(), 1);
        let ExecEvent::Fill(f) = &evs[0] else {
            panic!("expected a Fill");
        };
        assert_eq!(f.symbol_id, SymbolId::new(0)); // BTC
        assert_eq!(f.order_id, OrderId::new(502464503020));
        assert_eq!(f.cloid, None, "this payload carries no cloid at all");
        assert_eq!(f.side, Side::Sell, "\"A\" = ask = we sold");
        assert_eq!(f.qty, dec!(0.00018));
        assert_eq!(f.price, dec!(64007.0));
        assert_eq!(f.fee, dec!(0.000829));
        assert_eq!(f.closed_pnl, Decimal::ZERO);
        assert_eq!(f.liquidity, Liquidity::Maker, "crossed:false = we rested");
        assert_eq!(f.trade_id, 351400627290190);
        assert_eq!(f.ts_event, 1_784_976_932_073 * 1_000_000, "ms → ns");
    }

    #[test]
    fn crossed_means_we_were_the_taker() {
        let frame = r#"{"channel":"user","data":{"fills":[{"coin":"BTC","px":"1","sz":"1",
            "side":"B","time":1,"closedPnl":"0.0","oid":1,"crossed":true,"fee":"-0.5","tid":9}]}}"#;
        let evs = decode_frame(frame, &symbols()).unwrap();
        let ExecEvent::Fill(f) = &evs[0] else {
            panic!("expected a Fill");
        };
        assert_eq!(f.liquidity, Liquidity::Taker);
        assert_eq!(f.side, Side::Buy);
        assert_eq!(f.fee, dec!(-0.5), "a negative fee is a maker rebate");
    }

    #[test]
    fn user_fills_snapshot_decodes_like_any_other_batch() {
        let evs = decode_frame(USER_FILLS_SNAPSHOT, &symbols()).unwrap();
        assert_eq!(evs.len(), 1);
        let ExecEvent::Fill(f) = &evs[0] else {
            panic!("expected a Fill");
        };
        assert_eq!(f.symbol_id, SymbolId::new(42)); // xyz:SP500
        assert_eq!(f.side, Side::Buy);
        assert_eq!(f.qty, dec!(1.149));
        assert_eq!(f.price, dec!(7429.1));
        // `fee` is already inclusive of `builderFee`; adding them would double-count.
        assert_eq!(f.fee, dec!(4.489271));
        assert_eq!(f.trade_id, 955441961359817);
        assert!(fills_is_snapshot(&data_of(USER_FILLS_SNAPSHOT)));
    }

    #[test]
    fn absent_is_snapshot_reads_as_false() {
        let live = r#"{"channel":"userFills","data":{"user":"0x1","fills":[]}}"#;
        assert!(
            !fills_is_snapshot(&data_of(live)),
            "later frames omit the field entirely"
        );
        let explicit =
            r#"{"channel":"userFills","data":{"isSnapshot":false,"user":"0x1","fills":[]}}"#;
        assert!(!fills_is_snapshot(&data_of(explicit)));
        assert!(decode_frame(live, &symbols()).unwrap().is_empty());
    }

    // ── order updates ────────────────────────────────────────────────────────

    #[test]
    fn order_update_uses_status_timestamp_and_both_quantities() {
        let evs = decode_frame(ORDER_UPDATES, &symbols()).unwrap();
        assert_eq!(evs.len(), 1);
        let ExecEvent::Order(o) = &evs[0] else {
            panic!("expected an OrderUpdate");
        };
        assert_eq!(o.symbol_id, SymbolId::new(0));
        assert_eq!(o.order_id, OrderId::new(502464968365));
        assert_eq!(
            o.cloid,
            Some(Cloid::new(0xffabb2d185b75ea5ca0a5cd346b21507))
        );
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.status, OrderStatus::Rejected); // badAloPxRejected
        assert_eq!(o.cancel_reason, None);
        assert_eq!(o.price, Some(dec!(64020.0)));
        assert_eq!(o.orig_qty, dec!(1.17163), "origSz");
        assert_eq!(o.remaining_qty, dec!(1.17163), "sz is the *remaining* size");
        assert_eq!(o.ts_event, 1_784_976_929_583 * 1_000_000);
    }

    #[test]
    fn partially_filled_resting_order_keeps_both_quantities() {
        // The venue says `open` whether or not part of the order has traded; the
        // filled amount is derived from the quantities, not from the status.
        let frame = r#"{"channel":"orderUpdates","data":[{"order":{"coin":"ETH","side":"A",
            "limitPx":"3000.0","sz":"0.4","oid":7,"timestamp":10,"origSz":"1.0","cloid":null},
            "status":"open","statusTimestamp":20}]}"#;
        let evs = decode_frame(frame, &symbols()).unwrap();
        let ExecEvent::Order(o) = &evs[0] else {
            panic!("expected an OrderUpdate");
        };
        assert_eq!(o.status, OrderStatus::Resting);
        assert_eq!(o.cloid, None, "null cloid");
        assert_eq!(o.filled_qty(), dec!(0.6));
        assert_eq!(o.ts_event, 20 * 1_000_000);
    }

    #[test]
    fn order_update_without_limit_price_has_no_price() {
        let frame = r#"{"channel":"orderUpdates","data":[{"order":{"coin":"BTC","side":"B",
            "sz":"0.0","oid":8,"timestamp":1,"origSz":"1.0"},"status":"canceled",
            "statusTimestamp":2}]}"#;
        let evs = decode_frame(frame, &symbols()).unwrap();
        let ExecEvent::Order(o) = &evs[0] else {
            panic!("expected an OrderUpdate");
        };
        assert_eq!(o.price, None);
        assert_eq!(o.status, OrderStatus::Cancelled);
        assert_eq!(o.cancel_reason, Some(CancelReason::Requested));
    }

    // ── status classification ────────────────────────────────────────────────

    #[test]
    fn hl_status_covers_the_full_venue_enumeration() {
        use CancelReason as R;
        use OrderStatus as S;
        // All 29 values the venue is known to send, verified 2026-07-25.
        let table: [(&str, S, Option<R>); 29] = [
            ("open", S::Resting, None),
            ("triggered", S::Resting, None),
            ("filled", S::Filled, None),
            ("rejected", S::Rejected, None),
            ("canceled", S::Cancelled, Some(R::Requested)),
            ("scheduledCancel", S::Cancelled, Some(R::DeadMansSwitch)),
            ("liquidatedCanceled", S::Cancelled, Some(R::Liquidation)),
            ("marginCanceled", S::Cancelled, Some(R::Venue)),
            ("vaultWithdrawalCanceled", S::Cancelled, Some(R::Venue)),
            ("openInterestCapCanceled", S::Cancelled, Some(R::Venue)),
            ("selfTradeCanceled", S::Cancelled, Some(R::Venue)),
            ("reduceOnlyCanceled", S::Cancelled, Some(R::Venue)),
            ("siblingFilledCanceled", S::Cancelled, Some(R::Venue)),
            ("delistedCanceled", S::Cancelled, Some(R::Venue)),
            ("tickRejected", S::Rejected, None),
            ("minTradeNtlRejected", S::Rejected, None),
            ("perpMarginRejected", S::Rejected, None),
            ("reduceOnlyRejected", S::Rejected, None),
            ("badAloPxRejected", S::Rejected, None),
            ("iocCancelRejected", S::Rejected, None),
            ("badTriggerPxRejected", S::Rejected, None),
            ("marketOrderNoLiquidityRejected", S::Rejected, None),
            (
                "positionIncreaseAtOpenInterestCapRejected",
                S::Rejected,
                None,
            ),
            ("positionFlipAtOpenInterestCapRejected", S::Rejected, None),
            ("tooAggressiveAtOpenInterestCapRejected", S::Rejected, None),
            ("openInterestIncreaseRejected", S::Rejected, None),
            ("insufficientSpotBalanceRejected", S::Rejected, None),
            ("oracleRejected", S::Rejected, None),
            ("perpMaxPositionRejected", S::Rejected, None),
        ];
        for (wire, status, reason) in table {
            assert_eq!(hl_status(wire), Some((status, reason)), "status {wire:?}");
        }
        // `iocCancelRejected` is the trap: it contains "Cancel" but is a rejection.
        assert_eq!(
            hl_status("iocCancelRejected"),
            Some((OrderStatus::Rejected, None))
        );
    }

    #[test]
    fn unknown_statuses_use_suffixes_then_refuse_to_guess() {
        assert_eq!(
            hl_status("somethingNewRejected"),
            Some((OrderStatus::Rejected, None)),
            "suffix fallback for a value the venue added after we shipped"
        );
        assert_eq!(
            hl_status("somethingNewCanceled"),
            Some((OrderStatus::Cancelled, Some(CancelReason::Unspecified)))
        );
        assert_eq!(
            hl_status("somethingNewCancel"),
            Some((OrderStatus::Cancelled, Some(CancelReason::Unspecified)))
        );
        assert_eq!(
            hl_status("teleported"),
            None,
            "unclassifiable: emit nothing rather than guess live-vs-dead"
        );
    }

    #[test]
    fn unclassifiable_status_drops_only_that_entry() {
        let frame = r#"{"channel":"orderUpdates","data":[
            {"order":{"coin":"BTC","side":"B","limitPx":"1","sz":"1","oid":1,"timestamp":1,"origSz":"1"},
             "status":"teleported","statusTimestamp":2},
            {"order":{"coin":"ETH","side":"A","limitPx":"2","sz":"0","oid":2,"timestamp":1,"origSz":"2"},
             "status":"filled","statusTimestamp":3}]}"#;
        let evs = decode_frame(frame, &symbols()).unwrap();
        assert_eq!(evs.len(), 1, "the good update survives");
        let ExecEvent::Order(o) = &evs[0] else {
            panic!("expected an OrderUpdate");
        };
        assert_eq!(o.order_id, OrderId::new(2));
        assert_eq!(o.status, OrderStatus::Filled);
    }

    // ── tolerated non-fill unions ────────────────────────────────────────────

    #[test]
    fn funding_liquidation_and_non_user_cancel_produce_no_events() {
        let funding = r#"{"channel":"user","data":{"funding":{"time":1784976932073,"coin":"BTC",
            "usdc":"-0.5","szi":"0.1","fundingRate":"0.0000125"}}}"#;
        let liquidation = r#"{"channel":"user","data":{"liquidation":{"lid":42,
            "liquidator":"0xabc","liquidated_user":"0xdef","liquidated_ntl_pos":"1000.0",
            "liquidated_account_value":"12.5"}}}"#;
        // Deliberately produces nothing: it carries no sizes, and the same cancel
        // arrives fully-formed on `orderUpdates`.
        let non_user_cancel =
            r#"{"channel":"user","data":{"nonUserCancel":[{"coin":"BTC","oid":123}]}}"#;
        for frame in [funding, liquidation, non_user_cancel] {
            let evs = decode_frame(frame, &symbols()).expect("must not error");
            assert!(evs.is_empty(), "expected no events from {frame}");
        }
    }

    // ── partial tolerance ────────────────────────────────────────────────────

    #[test]
    fn unmapped_coin_is_skipped_without_dropping_the_batch() {
        let frame = r#"{"channel":"userFills","data":{"isSnapshot":false,"user":"0x1","fills":[
            {"coin":"@107","px":"1.5","sz":"10","side":"B","time":5,"closedPnl":"0.0","oid":1,"crossed":true,"fee":"0.1","tid":11},
            {"coin":"BTC","px":"64000.0","sz":"0.5","side":"B","time":6,"closedPnl":"0.0","oid":2,"crossed":true,"fee":"0.2","tid":12},
            {"coin":"PURR/USDC","px":"0.3","sz":"100","side":"A","time":7,"closedPnl":"0.0","oid":3,"crossed":false,"fee":"0.3","tid":13}]}}"#;
        let evs = decode_frame(frame, &symbols()).unwrap();
        assert_eq!(evs.len(), 1, "only the mapped coin survives");
        let ExecEvent::Fill(f) = &evs[0] else {
            panic!("expected a Fill");
        };
        assert_eq!(f.trade_id, 12);
        assert_eq!(f.symbol_id, SymbolId::new(0));
    }

    #[test]
    fn unreadable_side_is_malformed() {
        let frame = r#"{"channel":"user","data":{"fills":[{"coin":"BTC","px":"1","sz":"1",
            "side":"X","time":1,"closedPnl":"0.0","oid":1,"crossed":true,"fee":"0","tid":1}]}}"#;
        assert!(matches!(
            decode_frame(frame, &symbols()).unwrap_err(),
            DecodeError::Malformed("fill")
        ));
    }

    #[test]
    fn bad_price_string_is_an_error() {
        let frame = r#"{"channel":"user","data":{"fills":[{"coin":"BTC","px":"NaN","sz":"1",
            "side":"B","time":1,"closedPnl":"0.0","oid":1,"crossed":true,"fee":"0","tid":1}]}}"#;
        assert!(matches!(
            decode_frame(frame, &symbols()).unwrap_err(),
            DecodeError::BadNumber { field: "px", .. }
        ));
    }

    #[test]
    fn non_user_channels_yield_nothing() {
        // The dispatcher funnels everything it does not recognize through here.
        let data = serde_json::json!({"method": "subscribe"});
        for channel in ["subscriptionResponse", "pong", "error", "l2Book"] {
            assert!(decode_user_channel(channel, &data, &symbols())
                .unwrap()
                .is_empty());
        }
        // The subscription *type* is not the reply channel: frames for a
        // `userEvents` subscription arrive as `"user"`, so matching on the type
        // string would decode nothing forever.
        assert!(decode_user_channel("userEvents", &data, &symbols())
            .unwrap()
            .is_empty());
    }

    // ── cloid ────────────────────────────────────────────────────────────────

    #[test]
    fn parse_cloid_round_trips_and_rejects_wrong_lengths() {
        let want = Cloid::new(0xffabb2d185b75ea5ca0a5cd346b21507);
        assert_eq!(
            parse_cloid("0xffabb2d185b75ea5ca0a5cd346b21507"),
            Some(want)
        );
        assert_eq!(
            parse_cloid("ffabb2d185b75ea5ca0a5cd346b21507"),
            Some(want),
            "the 0x prefix is optional"
        );
        // Round-trip through the canonical 0x-prefixed, zero-padded 32-hex form.
        assert_eq!(parse_cloid(&format!("{:#034x}", want.get())), Some(want));
        assert_eq!(
            parse_cloid("0x00000000000000000000000000000001"),
            Some(Cloid::new(1)),
            "leading zeros are significant to the length check, not the value"
        );
        assert_eq!(parse_cloid("0x01"), None, "too short");
        assert_eq!(parse_cloid(""), None);
        assert_eq!(
            parse_cloid("0xffabb2d185b75ea5ca0a5cd346b2150700"),
            None,
            "too long"
        );
        assert_eq!(
            parse_cloid("0xzzabb2d185b75ea5ca0a5cd346b21507"),
            None,
            "right length, not hex"
        );
    }
}
