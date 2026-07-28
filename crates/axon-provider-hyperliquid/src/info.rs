//! `POST /info` **account & order reads** — the reconciliation and cancel-all
//! sweep data source.
//!
//! Why this module has to exist: Hyperliquid's `orderUpdates` WS channel **never
//! sends a snapshot**. After a restart, or after a dropped socket, the only way to
//! learn what is still resting — and which terminal statuses we missed while the
//! socket was down — is to read it back over REST. The same reads are the input to
//! `cancel_all`: the venue has no native cancel-all action, so a sweep enumerates
//! open orders and cancels them one by one.
//!
//! Shape of the module, mirroring [`ws::decode`](crate::ws::decode): every
//! `decode_*` is a **pure** function of a `&str` plus a [`SymbolMap`], unit-tested
//! offline against payloads captured from the live venue (probed 2026-07-25); the
//! `fetch_*` wrappers are the thin async edge and reuse
//! [`MAINNET_INFO`](crate::ws::MAINNET_INFO)/[`TESTNET_INFO`](crate::ws::TESTNET_INFO)
//! and [`HlError`]. These endpoints need no authentication — an address is a public
//! query key — which is why there is no signer here.
//!
//! Two venue behaviours drive the error handling:
//!
//! - **Coins can be unknown to us.** HIP-3 builder perps (`"xyz:SP500"`) and spot
//!   pairs (`"@107"`) appear in the same arrays as the perps we trade. One unknown
//!   coin must not blank out the rest of the response, so those rows are dropped and
//!   reported in [`Decoded::skipped_coins`] instead of failing the whole read.
//! - **`status` is an open string set.** The venue adds values without notice. A
//!   status we cannot classify is left as [`None`] rather than guessed: coercing an
//!   unknown string would make a dead order look live (leaking risk) or a live order
//!   look dead (leaking an un-cancelled order). The classifier is shared with the
//!   `orderUpdates` WS decoder ([`hl_status`]) so the two paths can never disagree
//!   about what a status means — which is the whole point of reconciling one against
//!   the other.

use axon_core::{
    AccountSnapshot, CancelReason, Cloid, Decimal, Nanos, OrderId, OrderStatus, OrderUpdate,
    Position, Side, SymbolId,
};
use axon_providers::CancelId;
use rust_decimal::prelude::ToPrimitive;
use serde::Deserialize;

use crate::encode::cloid_hex;
use crate::symbol_map::SymbolMap;
use crate::ws::decode::{parse_dec, MS_TO_NS};
use crate::ws::{hl_status, parse_cloid, DecodeError, HlError};

/// Every account starts with this many actions before volume-based growth. The
/// venue's own cap is `BASE_REQUEST_CAP + floor(cumVlm)`, live-confirmed against
/// `userRateLimit` — exposed so a rate governor can predict its budget from traded
/// volume instead of polling for it.
pub const BASE_REQUEST_CAP: u64 = 10_000;

type Result<T> = std::result::Result<T, DecodeError>;
type HlResult<T> = std::result::Result<T, HlError>;

// ── on-wire shapes (unlisted fields such as `children` are ignored) ───────────

/// The order body. Identical across `openOrders`, `frontendOpenOrders`,
/// `historicalOrders` and `orderStatus`; the richer endpoints simply populate more
/// of it, so the extras are all optional and `openOrders` just leaves them unset.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawOrder {
    coin: String,
    /// `"A"` = Sell (ask), `"B"` = Buy (bid).
    side: String,
    limit_px: String,
    /// Size still resting (for a live order); `origSz` is what was submitted.
    sz: String,
    oid: u64,
    timestamp: i64,
    orig_sz: String,
    #[serde(default)]
    reduce_only: bool,
    #[serde(default)]
    order_type: Option<String>,
    /// `null` for trigger orders even though the docs type it as a string.
    #[serde(default)]
    tif: Option<String>,
    /// A `0x`-prefixed 32-hex-char string, `null` for an order placed without a
    /// client id, or absent entirely — hence `Option` plus `default`.
    ///
    /// **Plain `openOrders` does echo it**, measured on testnet 2026-07-26
    /// (`live_cancel_testnet`): `oid=57015962788` came back carrying
    /// `0x98c5df9445c737e50000000100000003`, the very cloid the planner minted. This
    /// file used to say the field was absent there and that `None` therefore meant
    /// "not reported"; it does not. Nothing should be built on either reading — the
    /// venue documents the field only for the frontend endpoint, so treat a present
    /// cloid as a bonus and an absent one as "no client id **or** not reported",
    /// which is exactly what `Option` already says.
    #[serde(default)]
    cloid: Option<String>,
    #[serde(default)]
    is_trigger: bool,
    #[serde(default)]
    trigger_px: Option<String>,
    #[serde(default)]
    trigger_condition: Option<String>,
    #[serde(default)]
    is_position_tpsl: bool,
}

/// One `historicalOrders` row — the order body plus the status it reached.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawOrderRow {
    order: RawOrder,
    status: String,
    status_timestamp: i64,
}

/// `orderStatus` envelope. Outer `status` is `"order"` when found and
/// `"unknownOid"` when not; the order itself is **doubly nested**
/// (`order.order`) because the inner object is the same row shape as
/// `historicalOrders`.
#[derive(Deserialize)]
struct RawOrderStatusReply {
    status: String,
    #[serde(default)]
    order: Option<RawOrderRow>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMarginSummary {
    account_value: String,
    total_margin_used: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawClearinghouseState {
    /// Cross + isolated combined. `crossMarginSummary` is the cross-only subset,
    /// which we do not need: equity and margin used are account-level numbers.
    margin_summary: RawMarginSummary,
    withdrawable: String,
    #[serde(default)]
    cross_maintenance_margin_used: Option<String>,
    asset_positions: Vec<RawAssetPosition>,
    time: i64,
}

#[derive(Deserialize)]
struct RawAssetPosition {
    /// The sibling `type` field (`"oneWay"`) is the position mode; we only trade
    /// one-way, and the mode does not change the numbers below.
    position: RawPosition,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawPosition {
    coin: String,
    /// **Signed** position size: negative is short. The one field that makes this
    /// endpoint directly comparable with our own fill-derived position.
    szi: String,
    #[serde(default)]
    entry_px: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawExtraAgent {
    name: String,
    address: String,
    valid_until: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRateLimit {
    cum_vlm: String,
    n_requests_used: u64,
    n_requests_cap: u64,
    /// Defaulted rather than required: it is present on every payload seen so far, but
    /// losing the whole budget read because one optional field vanished would blind the
    /// governor entirely.
    #[serde(default)]
    n_requests_surplus: u64,
}

// ── normalized results ───────────────────────────────────────────────────────

/// A decoded array plus the rows we had to drop.
///
/// The venue mixes assets we do not track into the same arrays (HIP-3 builder
/// perps, spot pairs), so a strict decode would throw away a perfectly good BTC
/// order because an unrelated `"@107"` row sat next to it. Dropping is the right
/// behaviour, silence is not: `skipped_coins` exists so a reconciler can say
/// "this view is incomplete" out loud instead of concluding we have no open orders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded<T> {
    pub items: Vec<T>,
    /// Coins present in the response but absent from the [`SymbolMap`], in
    /// response order, with duplicates preserved (one entry per dropped row).
    pub skipped_coins: Vec<String>,
}

impl<T> Decoded<T> {
    /// Whether every row in the response was understood — i.e. whether `items`
    /// can be treated as the venue's complete answer.
    pub fn is_complete(&self) -> bool {
        self.skipped_coins.is_empty()
    }
}

/// One order as the venue describes it.
///
/// Named for its main use (the open-order sweep), but `historicalOrders` and
/// `orderStatus` return the same body for orders that are long dead — see
/// [`OrderWithStatus`], which pairs this with the status the order reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenOrder {
    pub symbol_id: SymbolId,
    /// The venue's coin string, kept alongside `symbol_id` so operational logs can
    /// name the asset without a reverse `SymbolMap` lookup.
    pub coin: String,
    pub order_id: OrderId,
    /// The client id, when the endpoint reports one.
    ///
    /// `None` means "no client id **or** not reported", and the two cannot be told
    /// apart from here. This used to say plain `openOrders` never carries the field,
    /// so `None` there was unambiguous — the venue disagrees: on testnet 2026-07-26
    /// `openOrders` returned our planner-minted cloid on a resting BTC order
    /// (`live_cancel_testnet`). It is documented only for the frontend endpoint, so
    /// treat a present value as a bonus and never build on its absence.
    pub cloid: Option<Cloid>,
    pub side: Side,
    pub limit_px: Decimal,
    /// Size still resting.
    pub sz: Decimal,
    /// Size the order was submitted with.
    pub orig_sz: Decimal,
    pub reduce_only: bool,
    /// `"Limit"`, `"Market"`, `"Stop Market"`, `"Stop Limit"`, … Left as the venue's
    /// string: nothing above this layer branches on it, and the set is open.
    pub order_type: Option<String>,
    /// `"Gtc"`/`"Ioc"`/`"Alo"`, or `None` — trigger orders report a null `tif`.
    pub tif: Option<String>,
    pub is_trigger: bool,
    pub trigger_px: Option<Decimal>,
    /// Human-readable trigger, e.g. `"Price below 63515"`.
    pub trigger_condition: Option<String>,
    pub is_position_tpsl: bool,
    /// Order placement time (the venue's `timestamp`), in ns.
    pub ts_event: Nanos,
}

impl OpenOrder {
    /// The cancel this order would need in a sweep. Hyperliquid keys cancels on
    /// `(asset, oid)`, so the symbol has to ride along — and the oid is preferred
    /// over the cloid because the oid is the identity the venue is guaranteed to
    /// report and guaranteed to recognize. A cloid may be absent from the reply
    /// (`openOrders` documents no such field), and an id nobody minted is refused
    /// outright: a cancel addressed to a synthesized cloid comes back
    /// `"Order was never placed, already canceled, or filled. asset=3"`, observed on
    /// testnet 2026-07-26.
    pub fn cancel_id(&self) -> CancelId {
        CancelId::OrderId {
            symbol: self.symbol_id,
            order_id: self.order_id,
        }
    }

    /// As a normalized [`OrderUpdate`].
    ///
    /// `status` is [`OrderStatus::Resting`] because these endpoints return *only*
    /// live orders — it is a fact of the endpoint, not a guess. A partially filled
    /// order still shows as `Resting` (it is on the book); the partial is visible
    /// through [`OrderUpdate::filled_qty`], so nothing is lost.
    pub fn to_order_update(&self) -> OrderUpdate {
        OrderUpdate {
            symbol_id: self.symbol_id,
            order_id: self.order_id,
            cloid: self.cloid,
            side: self.side,
            status: OrderStatus::Resting,
            price: Some(self.limit_px),
            orig_qty: self.orig_sz,
            remaining_qty: self.sz,
            cancel_reason: None,
            ts_event: self.ts_event,
        }
    }
}

/// An order plus the lifecycle state the venue reports for it
/// (`historicalOrders`, `orderStatus`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderWithStatus {
    pub order: OpenOrder,
    /// The venue's raw status string, retained verbatim: it is the only evidence
    /// available when `status` below is `None`, and the exact wording is what an
    /// operator needs in order to add a new value to [`hl_status`].
    pub venue_status: String,
    /// `None` when the venue's status string is outside the classifier — never
    /// guessed. Callers must emit no event for those rows.
    pub status: Option<OrderStatus>,
    pub cancel_reason: Option<CancelReason>,
    /// When the order *reached* this status (`statusTimestamp`), in ns. This, not
    /// the placement `timestamp`, is the event time of the state change.
    pub status_ts: Nanos,
}

impl OrderWithStatus {
    /// As a normalized [`OrderUpdate`], or `None` for an unclassifiable status.
    ///
    /// `remaining_qty` is forced to zero for a fill: nothing rests once an order
    /// is fully executed, whatever `sz` still says. Cancels and rejects keep the
    /// venue's `sz` — for those it is the *unfilled* remainder, which is what makes
    /// [`OrderUpdate::filled_qty`] come out right. Position math still comes from
    /// the fills feed; these quantities exist to reconcile resting state.
    pub fn to_order_update(&self) -> Option<OrderUpdate> {
        let status = self.status?;
        let remaining = match status {
            OrderStatus::Filled => Decimal::ZERO,
            _ => self.order.sz,
        };
        Some(OrderUpdate {
            symbol_id: self.order.symbol_id,
            order_id: self.order.order_id,
            cloid: self.order.cloid,
            side: self.order.side,
            status,
            price: Some(self.order.limit_px),
            orig_qty: self.order.orig_sz,
            remaining_qty: remaining,
            cancel_reason: self.cancel_reason,
            ts_event: self.status_ts,
        })
    }
}

/// Answer to an `orderStatus` query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderStatusReply {
    Found(Box<OrderWithStatus>),
    /// The venue answered `{"status":"unknownOid"}`. This is a **normal** answer,
    /// not an error: an oid the venue never knew, or one aged out of its history.
    /// The distinction matters for reconciliation — "unknown" means stop tracking,
    /// whereas a transport error means try again.
    UnknownOid,
}

/// `clearinghouseState`: the account snapshot plus the venue's own positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserState {
    pub account: AccountSnapshot,
    /// The venue's positions, signed (`szi`). Compared against our fill-derived
    /// positions to detect drift; `realized_pnl` is left zero because the venue
    /// reports lifetime PnL differently and we keep our own accounting.
    pub positions: Vec<Position>,
    /// Maintenance margin on the cross account — the input to liquidation-distance
    /// monitoring, and free here rather than one more round-trip later.
    pub cross_maintenance_margin_used: Decimal,
    /// Positions whose coin is unknown to the [`SymbolMap`] (see [`Decoded`]).
    pub skipped_coins: Vec<String>,
}

/// An approved API ("agent") wallet and its hard expiry.
///
/// `extraAgents` is **undocumented**. It is live and stable as of 2026-07-25 and is
/// the only canonical way to check when our API wallet stops being able to sign, so
/// it is worth depending on — but the shape is unspecified and may change without
/// notice, which is why the decode is pinned to a captured payload in the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraAgent {
    pub name: String,
    /// The agent wallet address, lower-case `0x` hex as the venue returns it.
    pub address: String,
    /// Expiry in ns (the venue's `validUntil` is epoch ms).
    pub valid_until: Nanos,
}

impl ExtraAgent {
    /// Whether this agent can still sign at `now` (ns). An expired agent produces
    /// signature failures on every action, so this is worth checking well before
    /// the deadline rather than discovering it mid-session.
    pub fn is_valid_at(&self, now: Nanos) -> bool {
        now < self.valid_until
    }
}

/// `userRateLimit`: our own action budget, as typed data for the rate governor.
///
/// Hyperliquid's limit is volume-gated, so the cap is not a constant — it grows
/// with traded volume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimitStatus {
    /// Cumulative traded volume in USDC.
    pub cum_vlm: Decimal,
    pub n_requests_used: u64,
    pub n_requests_cap: u64,
    /// Actions bought ahead via `reserveRequestWeight`, i.e. `max(0, reserved − used)`.
    ///
    /// Surfaced because [`crate::governor::RateGovernor::observe_rate_limit_status`]
    /// takes it: without it the governor has to be fed a hard-coded `0` and would
    /// under-count a budget that was paid for.
    pub n_requests_surplus: u64,
}

impl RateLimitStatus {
    /// The venue's effective action limit: the volume-derived cap **plus** any weight
    /// bought ahead with `reserveRequestWeight`. Ignoring the surplus would silently
    /// discard budget that was paid for in USDC.
    pub fn effective_cap(&self) -> u64 {
        self.n_requests_cap.saturating_add(self.n_requests_surplus)
    }

    /// Actions still available. Saturating: the venue can report `used > cap`
    /// transiently, and an underflow panic in a rate governor is worse than a zero.
    pub fn remaining(&self) -> u64 {
        self.effective_cap().saturating_sub(self.n_requests_used)
    }

    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }

    /// The cap implied by `cum_vlm` via the venue's live-confirmed formula
    /// (`BASE_REQUEST_CAP + floor(cumVlm)`). Lets the governor project its budget
    /// forward from volume it has already traded, without another read.
    pub fn expected_cap(&self) -> u64 {
        BASE_REQUEST_CAP + self.cum_vlm.floor().to_u64().unwrap_or(0)
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn parse_dec_opt(field: &'static str, s: Option<&String>) -> Result<Option<Decimal>> {
    match s {
        Some(v) => Ok(Some(parse_dec(field, v)?)),
        None => Ok(None),
    }
}

fn ms_to_ns(ms: i64) -> Nanos {
    ms * MS_TO_NS
}

/// `"A"` is the ask side (a sell), `"B"` the bid side (a buy).
fn hl_side(s: &str) -> Result<Side> {
    match s {
        "B" => Ok(Side::Buy),
        "A" => Ok(Side::Sell),
        _ => Err(DecodeError::Malformed("order side")),
    }
}

// ── pure decoders ────────────────────────────────────────────────────────────

fn to_open_order(raw: RawOrder, symbol_id: SymbolId) -> Result<OpenOrder> {
    Ok(OpenOrder {
        symbol_id,
        order_id: OrderId::new(raw.oid),
        // A cloid the venue sent but we cannot parse fails the decode instead of
        // becoming `None`. On the WS fill path dropping the correlation is the
        // lesser evil (the fill still has to be accounted for), but this is the
        // *reconciliation* read: `None` here reads as "order has no client id",
        // which would invite a duplicate order for one that is already live.
        cloid: match raw.cloid.as_deref() {
            Some(c) => Some(parse_cloid(c).ok_or_else(|| DecodeError::BadNumber {
                field: "cloid",
                value: c.to_string(),
            })?),
            None => None,
        },
        side: hl_side(&raw.side)?,
        limit_px: parse_dec("limitPx", &raw.limit_px)?,
        sz: parse_dec("sz", &raw.sz)?,
        orig_sz: parse_dec("origSz", &raw.orig_sz)?,
        reduce_only: raw.reduce_only,
        order_type: raw.order_type,
        tif: raw.tif,
        is_trigger: raw.is_trigger,
        trigger_px: parse_dec_opt("triggerPx", raw.trigger_px.as_ref())?,
        trigger_condition: raw.trigger_condition,
        is_position_tpsl: raw.is_position_tpsl,
        ts_event: ms_to_ns(raw.timestamp),
        coin: raw.coin,
    })
}

fn to_order_with_status(row: RawOrderRow, symbol_id: SymbolId) -> Result<OrderWithStatus> {
    let mapped = hl_status(&row.status);
    Ok(OrderWithStatus {
        order: to_open_order(row.order, symbol_id)?,
        status: mapped.map(|(s, _)| s),
        cancel_reason: mapped.and_then(|(_, r)| r),
        status_ts: ms_to_ns(row.status_timestamp),
        venue_status: row.status,
    })
}

/// Decode an `openOrders` **or** `frontendOpenOrders` response.
///
/// One decoder for both: `frontendOpenOrders` is a strict superset of the fields
/// (it adds `cloid`, `orderType`, `tif` and the trigger block) at the same
/// rate-limit weight, so it is the one to reach for when reconciling by cloid —
/// plain `openOrders` cannot do that at all. `children` (attached TP/SL legs) is
/// ignored: those are not independently resting orders until they activate, at
/// which point they appear as rows of their own.
pub fn decode_open_orders(raw: &str, symbols: &SymbolMap) -> Result<Decoded<OpenOrder>> {
    let rows: Vec<RawOrder> = serde_json::from_str(raw)?;
    let mut items = Vec::with_capacity(rows.len());
    let mut skipped_coins = Vec::new();
    for row in rows {
        match symbols.id(&row.coin) {
            Some(id) => items.push(to_open_order(row, id)?),
            None => skipped_coins.push(row.coin),
        }
    }
    Ok(Decoded {
        items,
        skipped_coins,
    })
}

/// Decode a `historicalOrders` response — at most the 2000 most recent orders.
///
/// This is how terminal statuses missed while the WS was down are recovered: it is
/// the only endpoint that reports, in bulk, what *became* of orders that are no
/// longer open.
pub fn decode_historical_orders(
    raw: &str,
    symbols: &SymbolMap,
) -> Result<Decoded<OrderWithStatus>> {
    let rows: Vec<RawOrderRow> = serde_json::from_str(raw)?;
    let mut items = Vec::with_capacity(rows.len());
    let mut skipped_coins = Vec::new();
    for row in rows {
        match symbols.id(&row.order.coin) {
            Some(id) => items.push(to_order_with_status(row, id)?),
            None => skipped_coins.push(row.order.coin),
        }
    }
    Ok(Decoded {
        items,
        skipped_coins,
    })
}

/// Decode an `orderStatus` response (single order, found or not).
///
/// Unlike the array endpoints, an unknown coin here is an **error**: there are no
/// neighbouring rows to protect, and answering `UnknownOid` for an order the venue
/// did know would make a live order look gone. An unexpected outer `status` string
/// is an error for the same reason.
pub fn decode_order_status(raw: &str, symbols: &SymbolMap) -> Result<OrderStatusReply> {
    let reply: RawOrderStatusReply = serde_json::from_str(raw)?;
    match reply.status.as_str() {
        "order" => {
            let row = reply.order.ok_or(DecodeError::Malformed("orderStatus"))?;
            let symbol_id = symbols
                .id(&row.order.coin)
                .ok_or_else(|| DecodeError::UnknownCoin(row.order.coin.clone()))?;
            Ok(OrderStatusReply::Found(Box::new(to_order_with_status(
                row, symbol_id,
            )?)))
        }
        "unknownOid" => Ok(OrderStatusReply::UnknownOid),
        _ => Err(DecodeError::Malformed("orderStatus")),
    }
}

/// Decode a `clearinghouseState` response into an [`AccountSnapshot`] plus the
/// venue's signed positions.
///
/// `equity`/`margin_used` come from `marginSummary` (cross **and** isolated), not
/// `crossMarginSummary`: risk is an account-level question. An unfunded account
/// answers with `"0.0"` everywhere and an empty `assetPositions`, which decodes
/// cleanly to a flat snapshot — that is a valid state, not a failure.
pub fn decode_user_state(raw: &str, symbols: &SymbolMap) -> Result<UserState> {
    let state: RawClearinghouseState = serde_json::from_str(raw)?;
    let mut positions = Vec::with_capacity(state.asset_positions.len());
    let mut skipped_coins = Vec::new();
    for ap in state.asset_positions {
        let p = ap.position;
        let Some(symbol_id) = symbols.id(&p.coin) else {
            skipped_coins.push(p.coin);
            continue;
        };
        positions.push(Position {
            symbol_id,
            qty: parse_dec("szi", &p.szi)?,
            avg_px: parse_dec_opt("entryPx", p.entry_px.as_ref())?.unwrap_or(Decimal::ZERO),
            realized_pnl: Decimal::ZERO,
        });
    }
    Ok(UserState {
        account: AccountSnapshot {
            equity: parse_dec("accountValue", &state.margin_summary.account_value)?,
            withdrawable: parse_dec("withdrawable", &state.withdrawable)?,
            margin_used: parse_dec("totalMarginUsed", &state.margin_summary.total_margin_used)?,
            ts_event: ms_to_ns(state.time),
        },
        positions,
        cross_maintenance_margin_used: parse_dec_opt(
            "crossMaintenanceMarginUsed",
            state.cross_maintenance_margin_used.as_ref(),
        )?
        .unwrap_or(Decimal::ZERO),
        skipped_coins,
    })
}

/// Decode an `extraAgents` response. See [`ExtraAgent`] on why we rely on an
/// undocumented endpoint.
pub fn decode_extra_agents(raw: &str) -> Result<Vec<ExtraAgent>> {
    let rows: Vec<RawExtraAgent> = serde_json::from_str(raw)?;
    Ok(rows
        .into_iter()
        .map(|a| ExtraAgent {
            name: a.name,
            address: a.address,
            valid_until: ms_to_ns(a.valid_until),
        })
        .collect())
}

/// Decode a `userRateLimit` response.
pub fn decode_rate_limit(raw: &str) -> Result<RateLimitStatus> {
    let r: RawRateLimit = serde_json::from_str(raw)?;
    Ok(RateLimitStatus {
        cum_vlm: parse_dec("cumVlm", &r.cum_vlm)?,
        n_requests_used: r.n_requests_used,
        n_requests_cap: r.n_requests_cap,
        n_requests_surplus: r.n_requests_surplus,
    })
}

// ── async edge (the only I/O in this module) ──────────────────────────────────

/// POST one `/info` request body and return the raw response text.
///
/// A fresh client per call, matching [`ws::rest`](crate::ws::rest); these reads are
/// low-frequency (startup, reconnect, periodic reconciliation), so connection reuse
/// is not worth an owned client here.
///
/// Bounded by [`INFO_TIMEOUT`](crate::ws::rest::INFO_TIMEOUT), and the same one number
/// as `ws::rest` so the two halves of `/info` cannot come to disagree about how long a
/// venue is allowed to say nothing. Unbounded, the failure is not a refused connection
/// but an endpoint that completes the handshake and then answers nothing: the periodic
/// reconciliation loop parks inside `send().await` and never runs again, and the session
/// keeps trading against a position it has stopped re-reading. That is silent — nothing
/// errors, nothing retries, and the drift only surfaces as a fill nobody expected.
async fn post_info(info_url: &str, body: &serde_json::Value) -> HlResult<String> {
    Ok(reqwest::Client::builder()
        .timeout(crate::ws::rest::INFO_TIMEOUT)
        .connect_timeout(crate::ws::rest::INFO_TIMEOUT)
        .build()?
        .post(info_url)
        .json(body)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?)
}

/// Fetch open orders. `dex` selects a HIP-3 builder-deployed perp dex; `None`
/// sends `""`, which the venue reads as the first (native) perp dex.
///
/// Prefer [`fetch_frontend_open_orders`] — same rate-limit weight, more fields.
pub async fn fetch_open_orders(
    info_url: &str,
    user: &str,
    dex: Option<&str>,
    symbols: &SymbolMap,
) -> HlResult<Decoded<OpenOrder>> {
    let body = serde_json::json!({
        "type": "openOrders", "user": user, "dex": dex.unwrap_or("")
    });
    Ok(decode_open_orders(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch open orders *with* cloids and trigger info — the sweep/reconcile read.
pub async fn fetch_frontend_open_orders(
    info_url: &str,
    user: &str,
    dex: Option<&str>,
    symbols: &SymbolMap,
) -> HlResult<Decoded<OpenOrder>> {
    let body = serde_json::json!({
        "type": "frontendOpenOrders", "user": user, "dex": dex.unwrap_or("")
    });
    Ok(decode_open_orders(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch the recent order history (up to the 2000 most recent).
pub async fn fetch_historical_orders(
    info_url: &str,
    user: &str,
    symbols: &SymbolMap,
) -> HlResult<Decoded<OrderWithStatus>> {
    let body = serde_json::json!({ "type": "historicalOrders", "user": user });
    Ok(decode_historical_orders(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch the status of one order by venue `oid`. Weight 2 — the cheapest per-order
/// read, so this is what a targeted "did that order actually die?" check should use.
pub async fn fetch_order_status_by_oid(
    info_url: &str,
    user: &str,
    oid: OrderId,
    symbols: &SymbolMap,
) -> HlResult<OrderStatusReply> {
    let body = serde_json::json!({ "type": "orderStatus", "user": user, "oid": oid.get() });
    Ok(decode_order_status(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch the status of one order by our `cloid`.
///
/// Note the venue's surprise: the request key is **`oid` either way** — a number
/// addresses the venue's id, a hex string addresses ours. There is no `cloid` key.
pub async fn fetch_order_status_by_cloid(
    info_url: &str,
    user: &str,
    cloid: Cloid,
    symbols: &SymbolMap,
) -> HlResult<OrderStatusReply> {
    let body = serde_json::json!({ "type": "orderStatus", "user": user, "oid": cloid_hex(cloid) });
    Ok(decode_order_status(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch the account snapshot + the venue's positions (`clearinghouseState`).
pub async fn fetch_user_state(
    info_url: &str,
    user: &str,
    symbols: &SymbolMap,
) -> HlResult<UserState> {
    let body = serde_json::json!({ "type": "clearinghouseState", "user": user });
    Ok(decode_user_state(
        &post_info(info_url, &body).await?,
        symbols,
    )?)
}

/// Fetch approved API wallets and their expiries (`extraAgents`).
pub async fn fetch_extra_agents(info_url: &str, user: &str) -> HlResult<Vec<ExtraAgent>> {
    let body = serde_json::json!({ "type": "extraAgents", "user": user });
    Ok(decode_extra_agents(&post_info(info_url, &body).await?)?)
}

/// Fetch our own action budget (`userRateLimit`).
pub async fn fetch_user_rate_limit(info_url: &str, user: &str) -> HlResult<RateLimitStatus> {
    let body = serde_json::json!({ "type": "userRateLimit", "user": user });
    Ok(decode_rate_limit(&post_info(info_url, &body).await?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn symbols() -> SymbolMap {
        SymbolMap::from_perps(["BTC", "ETH"])
    }

    // ── payloads captured from the live venue on 2026-07-25 ───────────────────

    const OPEN_ORDERS: &str = r#"[{"coin":"BTC","side":"A","limitPx":"60339.0","sz":"0.19304",
        "oid":502462803257,"timestamp":1784976343254,"origSz":"0.19304","reduceOnly":true}]"#;

    const FRONTEND_OPEN_ORDERS: &str = r#"[{"coin":"BTC","side":"A","limitPx":"60339.0",
        "sz":"0.19304","oid":502462803257,"timestamp":1784976343254,
        "triggerCondition":"Price below 63515","isTrigger":true,"triggerPx":"63515.0",
        "children":[],"isPositionTpsl":false,"reduceOnly":true,"orderType":"Stop Market",
        "origSz":"0.19304","tif":null,"cloid":null}]"#;

    const HISTORICAL_ORDERS: &str = r#"[{"order":{"coin":"BTC","side":"A","limitPx":"60339.0",
        "sz":"0.19304","oid":502462803257,"timestamp":1784976343254,
        "triggerCondition":"Price below 63515","isTrigger":true,"triggerPx":"63515.0",
        "children":[],"isPositionTpsl":false,"reduceOnly":true,"orderType":"Stop Market",
        "origSz":"0.19304","tif":null,"cloid":null},"status":"open",
        "statusTimestamp":1784976343254}]"#;

    const ORDER_STATUS_FOUND: &str = r#"{"status":"order","order":{"order":{"coin":"BTC",
        "side":"A","limitPx":"60339.0","sz":"0.19304","oid":502462803257,
        "timestamp":1784976343254,"origSz":"0.19304","tif":null,"cloid":null},
        "status":"open","statusTimestamp":1784976343254}}"#;

    const ORDER_STATUS_UNKNOWN: &str = r#"{"status":"unknownOid"}"#;

    const EMPTY_STATE: &str = r#"{"marginSummary":{"accountValue":"0.0","totalNtlPos":"0.0",
        "totalRawUsd":"0.0","totalMarginUsed":"0.0"},"crossMarginSummary":{"accountValue":"0.0",
        "totalNtlPos":"0.0","totalRawUsd":"0.0","totalMarginUsed":"0.0"},
        "crossMaintenanceMarginUsed":"0.0","withdrawable":"0.0","assetPositions":[],
        "time":1784977518165}"#;

    const FUNDED_STATE: &str = r#"{"marginSummary":{"accountValue":"5432.1","totalNtlPos":"5163.5",
        "totalRawUsd":"268.6","totalMarginUsed":"258.1"},"crossMarginSummary":{
        "accountValue":"5432.1","totalNtlPos":"5163.5","totalRawUsd":"268.6",
        "totalMarginUsed":"258.1"},"crossMaintenanceMarginUsed":"129.05",
        "withdrawable":"5174.0","assetPositions":[{"type":"oneWay","position":{"coin":"BTC",
        "szi":"-0.08068","entryPx":"64000.0","positionValue":"5163.5","unrealizedPnl":"12.3",
        "returnOnEquity":"0.01","leverage":{"type":"cross","value":20},
        "liquidationPx":"70000.0","marginUsed":"258.1","maxLeverage":40,
        "cumFunding":{"allTime":"1.0","sinceOpen":"0.5","sinceChange":"0.1"}}}],
        "time":1784977518165}"#;

    const EXTRA_AGENTS: &str = r#"[{"name":"KM-36z2p1",
        "address":"0xa1b2c3d4e5f607182930a1b2c3d4e5f607182930","validUntil":1793708044866}]"#;

    const RATE_LIMIT: &str = r#"{"cumVlm":"5760209.9800000004","nRequestsUsed":3053,
        "nRequestsCap":5770209,"nRequestsSurplus":0}"#;

    /// The full status enumeration verified on 2026-07-25 (29 values).
    const ALL_STATUSES: [&str; 29] = [
        "open",
        "filled",
        "canceled",
        "triggered",
        "rejected",
        "marginCanceled",
        "vaultWithdrawalCanceled",
        "openInterestCapCanceled",
        "selfTradeCanceled",
        "reduceOnlyCanceled",
        "siblingFilledCanceled",
        "delistedCanceled",
        "liquidatedCanceled",
        "scheduledCancel",
        "tickRejected",
        "minTradeNtlRejected",
        "perpMarginRejected",
        "reduceOnlyRejected",
        "badAloPxRejected",
        "iocCancelRejected",
        "badTriggerPxRejected",
        "marketOrderNoLiquidityRejected",
        "positionIncreaseAtOpenInterestCapRejected",
        "positionFlipAtOpenInterestCapRejected",
        "tooAggressiveAtOpenInterestCapRejected",
        "openInterestIncreaseRejected",
        "insufficientSpotBalanceRejected",
        "oracleRejected",
        "perpMaxPositionRejected",
    ];

    // ── openOrders / frontendOpenOrders ───────────────────────────────────────

    #[test]
    fn decodes_open_orders() {
        let d = decode_open_orders(OPEN_ORDERS, &symbols()).unwrap();
        assert!(d.is_complete());
        assert_eq!(d.items.len(), 1);
        let o = &d.items[0];
        assert_eq!(o.symbol_id, SymbolId::new(0)); // BTC
        assert_eq!(o.coin, "BTC");
        assert_eq!(o.order_id, OrderId::new(502_462_803_257));
        assert_eq!(o.side, Side::Sell); // "A"
        assert_eq!(o.limit_px, dec!(60339.0));
        assert_eq!(o.sz, dec!(0.19304));
        assert_eq!(o.orig_sz, dec!(0.19304));
        assert!(o.reduce_only);
        assert_eq!(o.ts_event, 1_784_976_343_254 * 1_000_000);
        // This endpoint does not expose a cloid at all.
        assert_eq!(o.cloid, None);
        assert_eq!(o.order_type, None);
        assert!(!o.is_trigger);
    }

    #[test]
    fn decodes_frontend_open_orders_with_trigger_block_and_null_tif() {
        let d = decode_open_orders(FRONTEND_OPEN_ORDERS, &symbols()).unwrap();
        let o = &d.items[0];
        assert_eq!(o.order_type.as_deref(), Some("Stop Market"));
        assert_eq!(o.tif, None, "tif is null for trigger orders");
        assert!(o.is_trigger);
        assert_eq!(o.trigger_px, Some(dec!(63515.0)));
        assert_eq!(o.trigger_condition.as_deref(), Some("Price below 63515"));
        assert!(!o.is_position_tpsl);
        assert_eq!(o.cloid, None); // null on the wire
    }

    #[test]
    fn open_order_maps_to_a_resting_update_and_a_cancel_id() {
        let d = decode_open_orders(OPEN_ORDERS, &symbols()).unwrap();
        let u = d.items[0].to_order_update();
        assert_eq!(u.status, OrderStatus::Resting);
        assert!(!u.status.is_terminal());
        assert_eq!(u.price, Some(dec!(60339.0)));
        assert_eq!(u.orig_qty, dec!(0.19304));
        assert_eq!(u.remaining_qty, dec!(0.19304));
        assert_eq!(u.filled_qty(), Decimal::ZERO);
        assert_eq!(u.cancel_reason, None);
        assert_eq!(
            d.items[0].cancel_id(),
            CancelId::OrderId {
                symbol: SymbolId::new(0),
                order_id: OrderId::new(502_462_803_257),
            }
        );
    }

    #[test]
    fn cloid_is_parsed_from_the_venues_hex_string() {
        let body = r#"[{"coin":"ETH","side":"B","limitPx":"3000.0","sz":"1.0","oid":7,
            "timestamp":1,"origSz":"2.0","cloid":"0x000000000000000000000000000000ff"}]"#;
        let d = decode_open_orders(body, &symbols()).unwrap();
        assert_eq!(d.items[0].cloid, Some(Cloid::new(255)));
        assert_eq!(d.items[0].side, Side::Buy); // "B"
                                                // Partial fill is visible through the derived quantity, not the status.
        assert_eq!(d.items[0].to_order_update().filled_qty(), dec!(1.0));
    }

    #[test]
    fn a_malformed_cloid_fails_the_decode_rather_than_losing_our_key() {
        let body = r#"[{"coin":"BTC","side":"B","limitPx":"1","sz":"1","oid":1,"timestamp":1,
            "origSz":"1","cloid":"0xnothex"}]"#;
        assert!(matches!(
            decode_open_orders(body, &symbols()).unwrap_err(),
            DecodeError::BadNumber { field: "cloid", .. }
        ));
    }

    #[test]
    fn unknown_coins_are_skipped_without_failing_their_neighbours() {
        // HIP-3 builder perps and spot pairs share these arrays with our perps.
        let body = r#"[
            {"coin":"BTC","side":"A","limitPx":"1","sz":"1","oid":1,"timestamp":1,"origSz":"1"},
            {"coin":"xyz:SP500","side":"A","limitPx":"1","sz":"1","oid":2,"timestamp":1,"origSz":"1"},
            {"coin":"@107","side":"B","limitPx":"1","sz":"1","oid":3,"timestamp":1,"origSz":"1"},
            {"coin":"ETH","side":"B","limitPx":"1","sz":"1","oid":4,"timestamp":1,"origSz":"1"}]"#;
        let d = decode_open_orders(body, &symbols()).unwrap();
        assert_eq!(d.items.len(), 2);
        assert_eq!(d.items[0].order_id, OrderId::new(1));
        assert_eq!(d.items[1].order_id, OrderId::new(4));
        assert_eq!(d.skipped_coins, vec!["xyz:SP500", "@107"]);
        assert!(!d.is_complete(), "the caller must know the view is partial");
    }

    #[test]
    fn a_bad_side_string_is_an_error() {
        let body = r#"[{"coin":"BTC","side":"X","limitPx":"1","sz":"1","oid":1,"timestamp":1,
            "origSz":"1"}]"#;
        assert!(matches!(
            decode_open_orders(body, &symbols()).unwrap_err(),
            DecodeError::Malformed("order side")
        ));
    }

    // ── historicalOrders ──────────────────────────────────────────────────────

    #[test]
    fn decodes_historical_orders() {
        let d = decode_historical_orders(HISTORICAL_ORDERS, &symbols()).unwrap();
        assert!(d.is_complete());
        assert_eq!(d.items.len(), 1);
        let r = &d.items[0];
        assert_eq!(r.venue_status, "open");
        assert_eq!(r.status, Some(OrderStatus::Resting));
        assert_eq!(r.cancel_reason, None);
        assert_eq!(r.status_ts, 1_784_976_343_254 * 1_000_000);
        assert_eq!(r.order.symbol_id, SymbolId::new(0));
        assert_eq!(r.order.order_id, OrderId::new(502_462_803_257));
        let u = r.to_order_update().expect("classified status");
        assert_eq!(u.status, OrderStatus::Resting);
        assert_eq!(u.ts_event, r.status_ts, "event time is the status change");
    }

    #[test]
    fn recovers_terminal_statuses_missed_while_the_ws_was_down() {
        let body = r#"[
            {"order":{"coin":"BTC","side":"B","limitPx":"100","sz":"0","oid":1,"timestamp":1,
             "origSz":"2"},"status":"filled","statusTimestamp":5},
            {"order":{"coin":"ETH","side":"A","limitPx":"200","sz":"3","oid":2,"timestamp":2,
             "origSz":"4"},"status":"scheduledCancel","statusTimestamp":6},
            {"order":{"coin":"ETH","side":"A","limitPx":"200","sz":"1","oid":3,"timestamp":3,
             "origSz":"1"},"status":"minTradeNtlRejected","statusTimestamp":7}]"#;
        let d = decode_historical_orders(body, &symbols()).unwrap();
        let ups: Vec<OrderUpdate> = d
            .items
            .iter()
            .filter_map(OrderWithStatus::to_order_update)
            .collect();
        assert_eq!(ups.len(), 3);
        assert!(ups.iter().all(|u| u.status.is_terminal()));

        assert_eq!(ups[0].status, OrderStatus::Filled);
        assert_eq!(ups[0].remaining_qty, Decimal::ZERO);
        assert_eq!(ups[0].filled_qty(), dec!(2)); // nothing rests after a fill

        assert_eq!(ups[1].status, OrderStatus::Cancelled);
        assert_eq!(ups[1].cancel_reason, Some(CancelReason::DeadMansSwitch));
        // A cancel keeps the venue's unfilled remainder so `filled_qty` is right.
        assert_eq!(ups[1].remaining_qty, dec!(3));
        assert_eq!(ups[1].filled_qty(), dec!(1));

        assert_eq!(ups[2].status, OrderStatus::Rejected);
        assert_eq!(ups[2].cancel_reason, None);
    }

    #[test]
    fn an_unclassifiable_status_emits_no_event_but_is_still_reported() {
        let body = r#"[{"order":{"coin":"BTC","side":"B","limitPx":"1","sz":"1","oid":9,
            "timestamp":1,"origSz":"1"},"status":"quantumTunneled","statusTimestamp":2}]"#;
        let d = decode_historical_orders(body, &symbols()).unwrap();
        let r = &d.items[0];
        assert_eq!(r.status, None, "never guessed");
        assert_eq!(r.venue_status, "quantumTunneled", "evidence is kept");
        assert!(r.to_order_update().is_none(), "no event, not a wrong event");
    }

    // ── orderStatus ───────────────────────────────────────────────────────────

    #[test]
    fn decodes_the_doubly_nested_order_status_shape() {
        let reply = decode_order_status(ORDER_STATUS_FOUND, &symbols()).unwrap();
        let OrderStatusReply::Found(r) = reply else {
            panic!("expected Found");
        };
        assert_eq!(r.order.order_id, OrderId::new(502_462_803_257));
        assert_eq!(r.order.symbol_id, SymbolId::new(0));
        assert_eq!(r.order.side, Side::Sell);
        assert_eq!(r.order.limit_px, dec!(60339.0));
        assert_eq!(r.status, Some(OrderStatus::Resting));
        assert_eq!(r.status_ts, 1_784_976_343_254 * 1_000_000);
    }

    #[test]
    fn unknown_oid_is_a_normal_not_found_answer() {
        // Emphatically not an error: it means "stop tracking", where a transport
        // error would mean "retry".
        assert_eq!(
            decode_order_status(ORDER_STATUS_UNKNOWN, &symbols()).unwrap(),
            OrderStatusReply::UnknownOid
        );
    }

    #[test]
    fn an_unmapped_coin_or_odd_envelope_fails_a_single_order_query() {
        let other_dex = r#"{"status":"order","order":{"order":{"coin":"xyz:SP500","side":"A",
            "limitPx":"1","sz":"1","oid":1,"timestamp":1,"origSz":"1"},"status":"open",
            "statusTimestamp":1}}"#;
        assert!(matches!(
            decode_order_status(other_dex, &symbols()).unwrap_err(),
            DecodeError::UnknownCoin(c) if c == "xyz:SP500"
        ));
        // An unexpected outer status must not collapse into "not found".
        assert!(matches!(
            decode_order_status(r#"{"status":"somethingNew"}"#, &symbols()).unwrap_err(),
            DecodeError::Malformed("orderStatus")
        ));
        assert!(matches!(
            decode_order_status(r#"{"status":"order"}"#, &symbols()).unwrap_err(),
            DecodeError::Malformed("orderStatus")
        ));
    }

    // ── status mapping (shared with the `orderUpdates` WS decoder) ────────────
    //
    // These live here as well as next to `hl_status` because *this* module is the
    // reconciliation path: `historicalOrders` is the only place the rarer terminal
    // statuses (liquidatedCanceled, scheduledCancel, the `*Rejected` family) are
    // ever observed in bulk, so the table's coverage is this module's contract too.

    #[test]
    fn every_verified_venue_status_is_classified() {
        for s in ALL_STATUSES {
            assert!(
                hl_status(s).is_some(),
                "status {s:?} fell through the table"
            );
        }
        assert_eq!(ALL_STATUSES.len(), 29);
    }

    #[test]
    fn status_table_maps_reasons_exactly() {
        use CancelReason::*;
        use OrderStatus::*;
        let cases = [
            ("open", (Resting, None)),
            ("triggered", (Resting, None)),
            ("filled", (Filled, None)),
            ("canceled", (Cancelled, Some(Requested))),
            ("scheduledCancel", (Cancelled, Some(DeadMansSwitch))),
            ("liquidatedCanceled", (Cancelled, Some(Liquidation))),
            ("marginCanceled", (Cancelled, Some(Venue))),
            ("vaultWithdrawalCanceled", (Cancelled, Some(Venue))),
            ("openInterestCapCanceled", (Cancelled, Some(Venue))),
            ("selfTradeCanceled", (Cancelled, Some(Venue))),
            ("reduceOnlyCanceled", (Cancelled, Some(Venue))),
            ("siblingFilledCanceled", (Cancelled, Some(Venue))),
            ("delistedCanceled", (Cancelled, Some(Venue))),
            ("rejected", (Rejected, None)),
            ("tickRejected", (Rejected, None)),
            ("oracleRejected", (Rejected, None)),
        ];
        for (s, want) in cases {
            assert_eq!(hl_status(s), Some(want), "status {s:?}");
        }
    }

    #[test]
    fn unknown_statuses_fall_back_on_suffix_then_give_up() {
        // Values the venue may add later.
        assert_eq!(
            hl_status("someNewThingRejected"),
            Some((OrderStatus::Rejected, None))
        );
        assert_eq!(
            hl_status("someNewThingCanceled"),
            Some((OrderStatus::Cancelled, Some(CancelReason::Unspecified)))
        );
        assert_eq!(
            hl_status("someNewCancel"),
            Some((OrderStatus::Cancelled, Some(CancelReason::Unspecified)))
        );
        // Anything else: no event beats a wrong event.
        assert_eq!(hl_status("marketMadeUp"), None);
        assert_eq!(hl_status(""), None);
    }

    // ── clearinghouseState ────────────────────────────────────────────────────

    #[test]
    fn decodes_an_unfunded_clearinghouse_state_cleanly() {
        let s = decode_user_state(EMPTY_STATE, &symbols()).unwrap();
        assert_eq!(s.account.equity, Decimal::ZERO);
        assert_eq!(s.account.withdrawable, Decimal::ZERO);
        assert_eq!(s.account.margin_used, Decimal::ZERO);
        assert_eq!(s.account.ts_event, 1_784_977_518_165 * 1_000_000);
        assert!(s.positions.is_empty());
        assert!(s.skipped_coins.is_empty());
        assert_eq!(s.cross_maintenance_margin_used, Decimal::ZERO);
    }

    #[test]
    fn decodes_a_funded_state_into_a_signed_short_position() {
        let s = decode_user_state(FUNDED_STATE, &symbols()).unwrap();
        assert_eq!(s.account.equity, dec!(5432.1));
        assert_eq!(s.account.margin_used, dec!(258.1));
        assert_eq!(s.account.withdrawable, dec!(5174.0));
        assert_eq!(s.cross_maintenance_margin_used, dec!(129.05));
        assert_eq!(s.positions.len(), 1);
        let p = &s.positions[0];
        assert_eq!(p.symbol_id, SymbolId::new(0));
        assert_eq!(p.qty, dec!(-0.08068), "szi is signed: negative is short");
        assert!(p.qty.is_sign_negative());
        assert_eq!(p.avg_px, dec!(64000.0));
        assert_eq!(p.realized_pnl, Decimal::ZERO); // we keep our own accounting
        assert!(!p.is_flat());
    }

    #[test]
    fn a_position_in_an_unknown_coin_is_skipped_not_fatal() {
        let body = r#"{"marginSummary":{"accountValue":"1.0","totalMarginUsed":"0.5"},
            "withdrawable":"0.5","crossMaintenanceMarginUsed":"0.25","assetPositions":[
            {"type":"oneWay","position":{"coin":"@107","szi":"5.0","entryPx":"1.0"}},
            {"type":"oneWay","position":{"coin":"ETH","szi":"1.5","entryPx":"3000.0"}}],
            "time":1}"#;
        let s = decode_user_state(body, &symbols()).unwrap();
        assert_eq!(s.positions.len(), 1);
        assert_eq!(s.positions[0].symbol_id, SymbolId::new(1)); // ETH
        assert_eq!(s.positions[0].qty, dec!(1.5));
        assert_eq!(s.skipped_coins, vec!["@107"]);
    }

    // ── extraAgents / userRateLimit ────────────────────────────────────────────

    #[test]
    fn decodes_extra_agents() {
        // Pinned to the captured payload: the endpoint is undocumented, so this
        // test is the tripwire if the venue changes the shape.
        let agents = decode_extra_agents(EXTRA_AGENTS).unwrap();
        assert_eq!(agents.len(), 1);
        let a = &agents[0];
        assert_eq!(a.name, "KM-36z2p1");
        assert_eq!(a.address, "0xa1b2c3d4e5f607182930a1b2c3d4e5f607182930");
        assert_eq!(a.valid_until, 1_793_708_044_866 * 1_000_000);
        assert!(a.is_valid_at(1_784_977_518_165 * 1_000_000));
        assert!(!a.is_valid_at(1_793_708_044_867 * 1_000_000));
        assert!(decode_extra_agents("[]").unwrap().is_empty());
    }

    #[test]
    fn decodes_user_rate_limit_and_the_cap_formula_holds() {
        let r = decode_rate_limit(RATE_LIMIT).unwrap();
        assert_eq!(r.cum_vlm, dec!(5760209.9800000004));
        assert_eq!(r.n_requests_used, 3053);
        assert_eq!(r.n_requests_cap, 5_770_209);
        assert_eq!(r.remaining(), 5_770_209 - 3053);
        assert!(!r.is_exhausted());
        // Live-confirmed: cap == 10_000 + floor(cumVlm).
        assert_eq!(r.expected_cap(), r.n_requests_cap);
        assert_eq!(r.expected_cap(), BASE_REQUEST_CAP + 5_760_209);
        assert_eq!(r.n_requests_surplus, 0);
    }

    #[test]
    fn purchased_surplus_is_surfaced_for_the_governor() {
        // `reserveRequestWeight` buys extra actions; the governor takes the surplus as
        // an input, so dropping the field would silently under-count a paid-for budget.
        let with_surplus = r#"{"cumVlm":"0.0","nRequestsUsed":10,"nRequestsCap":10000,
            "nRequestsSurplus":2500}"#;
        let r = decode_rate_limit(with_surplus).unwrap();
        assert_eq!(r.n_requests_surplus, 2_500);
        // And it must actually raise the budget — surfacing the field without using it
        // would discard weight that was paid for in USDC.
        assert_eq!(r.effective_cap(), 12_500);
        assert_eq!(r.remaining(), 12_500 - 10);
        // Absent field must not fail the whole read — a blind governor is worse.
        let missing = r#"{"cumVlm":"0.0","nRequestsUsed":10,"nRequestsCap":10000}"#;
        assert_eq!(decode_rate_limit(missing).unwrap().n_requests_surplus, 0);
    }

    #[test]
    fn remaining_budget_saturates_instead_of_panicking() {
        let r = RateLimitStatus {
            cum_vlm: Decimal::ZERO,
            n_requests_used: 10_050,
            n_requests_cap: 10_000,
            n_requests_surplus: 0,
        };
        assert_eq!(r.remaining(), 0);
        assert!(r.is_exhausted());
    }

    /// Live read against mainnet `/info` for a public address. Ignored by default
    /// so `cargo test` stays offline; run with `./run.sh live`, or with:
    /// `AXON_HL_USER=0x... cargo test -p axon-provider-hyperliquid -- --ignored live_info_reads`.
    ///
    /// `AXON_HL_USER` is an *override*, and falls back to the account `.env` already
    /// names. It has to: `AXON_HL_USER` is defined nowhere in this repo — not in
    /// `.env.example`, not in `with-env.sh` — so demanding it made `./run.sh live`
    /// impossible to pass as shipped, and because cargo abandons a run at the first
    /// failing test *binary*, this one panic also stopped `live_fill_testnet` (a
    /// separate binary) from ever executing. A live suite that cannot reach its own
    /// last test is worse than one that fails: it fails somewhere else.
    ///
    /// The override still earns its keep. This reads **mainnet**, and the address in
    /// `.env` is a testnet master, so the fallback exercises every decoder against
    /// real mainnet payloads for an *empty* account — the `open.items.first()`
    /// round-trip below simply does not fire. Point `AXON_HL_USER` at a busy mainnet
    /// address to exercise that branch.
    #[tokio::test]
    #[ignore = "hits the live Hyperliquid /info endpoint; needs AXON_HL_USER or AXON_HL_ACCOUNT_ADDRESS"]
    async fn live_info_reads() {
        use crate::ws::{fetch_meta, MAINNET_INFO};

        let user = std::env::var("AXON_HL_USER")
            .or_else(|_| std::env::var("AXON_HL_ACCOUNT_ADDRESS"))
            .expect("AXON_HL_USER, or AXON_HL_ACCOUNT_ADDRESS from .env");
        let symbols = fetch_meta(MAINNET_INFO).await.expect("meta");

        let state = fetch_user_state(MAINNET_INFO, &user, &symbols)
            .await
            .expect("clearinghouseState");
        eprintln!(
            "equity={} margin_used={} positions={} skipped={:?}",
            state.account.equity,
            state.account.margin_used,
            state.positions.len(),
            state.skipped_coins
        );

        let open = fetch_frontend_open_orders(MAINNET_INFO, &user, None, &symbols)
            .await
            .expect("frontendOpenOrders");
        eprintln!(
            "open orders={} complete={}",
            open.items.len(),
            open.is_complete()
        );

        let rl = fetch_user_rate_limit(MAINNET_INFO, &user)
            .await
            .expect("userRateLimit");
        assert_eq!(rl.expected_cap(), rl.n_requests_cap, "cap formula drifted");

        let agents = fetch_extra_agents(MAINNET_INFO, &user)
            .await
            .expect("extraAgents");
        eprintln!("agents={agents:?}");

        // Every open order must round-trip through the cheap per-order read.
        if let Some(o) = open.items.first() {
            let reply = fetch_order_status_by_oid(MAINNET_INFO, &user, o.order_id, &symbols)
                .await
                .expect("orderStatus");
            assert!(matches!(reply, OrderStatusReply::Found(_)));
        }
    }
}
