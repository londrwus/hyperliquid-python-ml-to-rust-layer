//! Order/cancel → Hyperliquid **wire action** encoding (the input to
//! [`sign`](crate::sign)). These structs serialize (via `serde`/msgpack) to the
//! exact shapes the venue expects; **field order is significant** because the
//! same bytes are hashed for the signature (ADR-0009).
//!
//! Compact keys (`docs/research/hyperliquid-execution.md`): `a` asset index,
//! `b` isBuy, `p` price, `s` size, `r` reduceOnly, `t` type, `c` cloid. Prices and
//! sizes are strings. This increment covers **limit** orders + cancels; market-order
//! synthesis (IOC-at-slippage) and trigger orders land with the execution path.

use axon_core::{Cloid, Decimal, OrderId, Side, SymbolId, Tif};
use axon_providers::{CancelId, InstrumentTable, OrderRequest, Precision, PrecisionError};
use serde::Serialize;

/// Why an order could not be encoded for Hyperliquid.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("Hyperliquid has no FOK time-in-force")]
    UnsupportedTif,
    #[error("limit order requires a price (market-order synthesis lands with execution)")]
    MissingPrice,
    #[error("scheduled cancel lead {lead_ms}ms is below the venue minimum {min_ms}ms")]
    ScheduleCancelTooSoon { lead_ms: u64, min_ms: u64 },
    /// The order breaks the instrument's own number format. **Not** rounded here — see
    /// [`order_wire`].
    #[error("this order breaks the instrument's grid: {0}")]
    Precision(#[from] PrecisionError),
    /// We hold no grid for this asset, and the order would add exposure.
    #[error("no precision is known for asset {asset}; refusing anything that adds exposure")]
    PrecisionUnknown { asset: u32 },
}

// ── wire shapes (field order matters — it is hashed) ─────────────────────────

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct LimitWire {
    tif: &'static str,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct OrderTypeWire {
    limit: LimitWire,
}

/// A single order in a Hyperliquid `order` action.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct OrderWire {
    pub a: u32,
    pub b: bool,
    pub p: String,
    pub s: String,
    pub r: bool,
    pub t: OrderTypeWire,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub c: Option<String>,
}

/// A Hyperliquid `order` action (a batch of orders + grouping).
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct OrderAction {
    #[serde(rename = "type")]
    kind: &'static str,
    orders: Vec<OrderWire>,
    grouping: &'static str,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct CancelWire {
    a: u32,
    o: u64,
}

/// A Hyperliquid `cancel` action (by venue order id).
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct CancelAction {
    #[serde(rename = "type")]
    kind: &'static str,
    cancels: Vec<CancelWire>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct CancelByCloidWire {
    asset: u32,
    cloid: String,
}

/// A Hyperliquid `cancelByCloid` action (by our client order id).
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct CancelByCloidAction {
    #[serde(rename = "type")]
    kind: &'static str,
    cancels: Vec<CancelByCloidWire>,
}

// ── mapping helpers ──────────────────────────────────────────────────────────

/// Hyperliquid's TIF string. FOK is not offered (surfaced as an error early).
fn tif_str(tif: Tif) -> Result<&'static str, EncodeError> {
    Ok(match tif {
        Tif::Gtc => "Gtc",
        Tif::Ioc => "Ioc",
        Tif::PostOnly => "Alo", // add-liquidity-only (post-only)
        Tif::Fok => return Err(EncodeError::UnsupportedTif),
    })
}

/// Format a fixed-point value for the wire: no trailing zeros, no exponent.
///
/// This *only* formats. [`order_wire`] refuses anything not already on the
/// instrument's grid, so a change to this formatter can never quietly become a change
/// to the rounding — which is the one way a price could be altered after the `Plan`
/// that recorded it was written.
fn wire_decimal(d: Decimal) -> String {
    d.normalize().to_string()
}

/// A 128-bit client order id as Hyperliquid's `0x`-prefixed 32-hex-char string.
///
/// `pub(crate)` because `info` addresses orders by cloid too (`orderStatus` accepts a
/// cloid under the `oid` key) — one formatter, so the two paths cannot disagree about
/// zero-padding.
pub(crate) fn cloid_hex(c: Cloid) -> String {
    format!("0x{:032x}", c.get())
}

// ── public builders ──────────────────────────────────────────────────────────

/// Encode one limit [`OrderRequest`] into an [`OrderWire`].
///
/// **This never rounds.** A price it would have to change is a *planner* bug, not a
/// market event, and refusing is how it says so at the one place a wrong number cannot
/// get past: `submit_orders → order_action → order_wire` and `modify → modify_action →
/// order_wire` are the only two routes to the bytes. Rounding here would make the order
/// sent differ from the `Plan` that was recorded — and that plan is what the chain
/// summary logs, what `WorkingOrder::is()` compares against on the next pass, and what
/// the parity harness diffs, so a silently mutated price makes all three describe an
/// order that was never sent.
///
/// `precision` is a **required argument and a three-way enum, not an `Option`**: an
/// adapter author cannot encode an order without naming what is known about the
/// instrument's grid, and `Unconstrained` is a word that has to be typed. That is the
/// same structural move `GuardedClient` makes with risk (ADR-0010), one level down and
/// cheaper — `order_wire` already has the unbypassable property a wrapper would have to
/// manufacture.
pub fn order_wire(req: &OrderRequest, precision: Precision<'_>) -> Result<OrderWire, EncodeError> {
    match precision {
        Precision::Known(spec) => spec.check(req)?,
        Precision::Unconstrained => {}
        Precision::Unknown => {
            // The same asymmetry as the halt switch and the risk gate, for the same
            // reason: whatever went wrong, removing exposure must keep working while
            // adding it must not. An unvalidated close is very likely valid — its price
            // is the venue's own touch and its size is a sum of venue-reported fills —
            // while an unvalidated open is a `tickRejected` that reads like a signing
            // bug from in here.
            if !req.reduce_only {
                return Err(EncodeError::PrecisionUnknown {
                    asset: req.symbol_id.get(),
                });
            }
        }
    }
    let price = req.price.ok_or(EncodeError::MissingPrice)?;
    Ok(OrderWire {
        a: req.symbol_id.get(),
        b: matches!(req.side, Side::Buy),
        p: wire_decimal(price),
        s: wire_decimal(req.qty),
        r: req.reduce_only,
        t: OrderTypeWire {
            limit: LimitWire {
                tif: tif_str(req.tif)?,
            },
        },
        c: Some(cloid_hex(req.cloid)),
    })
}

/// Encode a batch of limit orders into an `order` action (`grouping: "na"`).
///
/// One bad leg refuses the **whole batch**. A half-submitted batch leaves the caller's
/// plan half-executed and its position somewhere neither the plan nor the venue
/// describes — and the caller has no way to find out which half went.
pub fn order_action(
    reqs: &[OrderRequest],
    instruments: &InstrumentTable,
) -> Result<OrderAction, EncodeError> {
    let orders = reqs
        .iter()
        .map(|r| order_wire(r, instruments.precision(r.symbol_id)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OrderAction {
        kind: "order",
        orders,
        grouping: "na",
    })
}

/// Encode a `cancel` action from `(symbol, venue order id)` pairs.
pub fn cancel_action(items: &[(SymbolId, OrderId)]) -> CancelAction {
    CancelAction {
        kind: "cancel",
        cancels: items
            .iter()
            .map(|(sym, oid)| CancelWire {
                a: sym.get(),
                o: oid.get(),
            })
            .collect(),
    }
}

/// Encode a `cancelByCloid` action from `(symbol, cloid)` pairs.
pub fn cancel_by_cloid_action(items: &[(SymbolId, Cloid)]) -> CancelByCloidAction {
    CancelByCloidAction {
        kind: "cancelByCloid",
        cancels: items
            .iter()
            .map(|(sym, cloid)| CancelByCloidWire {
                asset: sym.get(),
                cloid: cloid_hex(*cloid),
            })
            .collect(),
    }
}

/// A Hyperliquid `scheduleCancel` action — the **dead-man's switch**.
///
/// Arming it tells the venue "cancel everything I have open at this absolute UTC
/// millisecond unless I tell you otherwise." A process that crashes, wedges, or
/// loses its network stops re-arming, the deadline passes, and the venue flattens
/// our resting orders for us. It is the only protection that survives *us* dying,
/// which is exactly the case a client-side cancel cannot cover.
///
/// `time` is **omitted entirely** to disarm, never sent as `null`: the action is
/// msgpack'd and hashed for the signature (ADR-0009), so field *presence* changes
/// the hash. A `"time": null` would sign a different action than the venue parses
/// and be rejected as a bad signature — hence `skip_serializing_if`.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ScheduleCancelAction {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    time: Option<u64>,
}

/// Minimum lead time the venue enforces on a scheduled cancel: the deadline must be
/// at least 5 s in the future.
pub const SCHEDULE_CANCEL_MIN_LEAD_MS: u64 = 5_000;

/// Maximum times the switch may fire per UTC day before the venue stops honouring it.
/// Firing is meant to be an emergency, not a routine; a system that trips this has a
/// liveness bug, and burning the budget leaves the account unprotected for the rest
/// of the day.
pub const SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY: u32 = 10;

/// Encode a `scheduleCancel` action.
///
/// `deadline_ms` is an **absolute** UTC epoch-millisecond timestamp, not a duration.
/// `None` disarms. Callers should validate the 5 s minimum lead against their own
/// clock before signing — see [`schedule_cancel_deadline`].
pub fn schedule_cancel_action(deadline_ms: Option<u64>) -> ScheduleCancelAction {
    ScheduleCancelAction {
        kind: "scheduleCancel",
        time: deadline_ms,
    }
}

/// Compute a valid deadline `lead_ms` into the future from `now_ms`, refusing a lead
/// the venue would reject.
///
/// Returned as a `Result` rather than silently clamping: a caller asking for a 1 s
/// dead-man's switch has a mistaken mental model of the protection they are getting,
/// and quietly giving them 5 s would hide that.
pub fn schedule_cancel_deadline(now_ms: u64, lead_ms: u64) -> Result<u64, EncodeError> {
    if lead_ms < SCHEDULE_CANCEL_MIN_LEAD_MS {
        return Err(EncodeError::ScheduleCancelTooSoon {
            lead_ms,
            min_ms: SCHEDULE_CANCEL_MIN_LEAD_MS,
        });
    }
    Ok(now_ms + lead_ms)
}

/// The order to modify, addressed by venue `oid` (number) or our `cloid` (hex).
#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(untagged)]
enum OidRef {
    Num(u64),
    Cloid(String),
}

/// A Hyperliquid `modify` action: replace a live order with a new one.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ModifyAction {
    #[serde(rename = "type")]
    kind: &'static str,
    oid: OidRef,
    order: OrderWire,
}

/// Encode a `modify` action targeting an existing order (by oid or cloid) with the
/// replacement described by `req`.
pub fn modify_action(
    id: CancelId,
    req: &OrderRequest,
    instruments: &InstrumentTable,
) -> Result<ModifyAction, EncodeError> {
    let oid = match id {
        CancelId::OrderId { order_id, .. } => OidRef::Num(order_id.get()),
        CancelId::Cloid { cloid, .. } => OidRef::Cloid(cloid_hex(cloid)),
    };
    Ok(ModifyAction {
        kind: "modify",
        oid,
        // The second of the two routes to the bytes, and the one the planner never
        // takes: a `modify` request is built by a caller, not by `Planner::plan`, so
        // this is the only thing standing between a hand-built replacement and a
        // `tickRejected`.
        order: order_wire(req, instruments.precision(req.symbol_id))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::Tif;
    use axon_providers::{InstrumentSpec, PriceGrid, SizeGrid};
    use rust_decimal_macros::dec;
    use serde_json::json;

    fn req(side: Side, tif: Tif) -> OrderRequest {
        OrderRequest::limit(
            SymbolId::new(3),
            side,
            dec!(0.50),
            dec!(50000.0),
            tif,
            Cloid::new(255),
        )
    }

    /// A venue with no declared grids. These tests are about the **wire shape** — the
    /// keys, the field order, the msgpack bytes that get hashed — and always were;
    /// naming the precision is what makes that explicit rather than accidental.
    fn unconstrained() -> InstrumentTable {
        InstrumentTable::unconstrained()
    }

    /// Testnet BTC's real grid: `szDecimals: 5`, so a 0.1 increment, five significant
    /// figures, and the venue's $10 minimum.
    fn btc_table() -> InstrumentTable {
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: SymbolId::new(3),
            price: PriceGrid::decimals_with_sig_figs(1, 5).unwrap(),
            size: SizeGrid::decimals(5).unwrap(),
            min_notional: Some(crate::MIN_ORDER_NOTIONAL_USD),
        });
        t
    }

    #[test]
    fn encodes_a_gtc_buy_limit() {
        let w = order_wire(&req(Side::Buy, Tif::Gtc), Precision::Unconstrained).unwrap();
        assert_eq!(w.a, 3);
        assert!(w.b);
        assert_eq!(w.p, "50000"); // trailing zero trimmed
        assert_eq!(w.s, "0.5"); //   "
        assert!(!w.r);
        assert_eq!(w.t.limit.tif, "Gtc");
        assert_eq!(w.c.as_deref(), Some("0x000000000000000000000000000000ff"));
    }

    #[test]
    fn order_action_serializes_to_expected_json() {
        let action = order_action(&[req(Side::Sell, Tif::PostOnly)], &unconstrained()).unwrap();
        let v = serde_json::to_value(&action).unwrap();
        assert_eq!(
            v,
            json!({
                "type": "order",
                "orders": [{
                    "a": 3, "b": false, "p": "50000", "s": "0.5", "r": false,
                    "t": { "limit": { "tif": "Alo" } },
                    "c": "0x000000000000000000000000000000ff"
                }],
                "grouping": "na"
            })
        );
    }

    #[test]
    fn fok_is_rejected() {
        assert!(matches!(
            order_wire(&req(Side::Buy, Tif::Fok), Precision::Unconstrained),
            Err(EncodeError::UnsupportedTif)
        ));
    }

    #[test]
    fn market_order_without_price_is_rejected() {
        let mut r = req(Side::Buy, Tif::Ioc);
        r.price = None;
        assert!(matches!(
            order_wire(&r, Precision::Unconstrained),
            Err(EncodeError::MissingPrice)
        ));
    }

    #[test]
    fn an_off_grid_price_is_refused_at_the_wire_rather_than_silently_rounded() {
        // The refusal is the point. Rounding here would make the bytes sent differ from
        // the `Plan` that was recorded — the plan the chain summary logs, the plan
        // `WorkingOrder::is()` compares against next pass, the plan the parity harness
        // diffs — so all three would describe an order that was never sent.
        let t = btc_table();
        let mut r = req(Side::Buy, Tif::Gtc);
        r.price = Some(dec!(108234.567)); // eight significant figures
        let err = order_wire(&r, t.precision(SymbolId::new(3))).unwrap_err();
        assert!(
            matches!(
                err,
                EncodeError::Precision(axon_providers::PrecisionError::Tick { .. })
            ),
            "got {err:?}"
        );
        // …and the same price one grid step away goes straight through.
        r.price = Some(dec!(108234));
        assert_eq!(
            order_wire(&r, t.precision(SymbolId::new(3))).unwrap().p,
            "108234"
        );
    }

    #[test]
    fn an_off_lot_size_is_refused_at_the_wire() {
        // Sizes reach the planner as `Decimal::new(v, 8)` — eight decimal places — and
        // BTC's lot is five. Without this the residue goes out and comes back rejected.
        let t = btc_table();
        let mut r = req(Side::Buy, Tif::Gtc);
        r.price = Some(dec!(108234));
        r.qty = dec!(0.00123456);
        assert!(matches!(
            order_wire(&r, t.precision(SymbolId::new(3))),
            Err(EncodeError::Precision(
                axon_providers::PrecisionError::Lot { .. }
            ))
        ));
        r.qty = dec!(0.00123);
        assert!(order_wire(&r, t.precision(SymbolId::new(3))).is_ok());
    }

    #[test]
    fn an_order_for_an_instrument_with_no_spec_is_refused_unless_it_reduces() {
        // The same asymmetry as the halt switch: whatever went wrong, removing exposure
        // must keep working while adding it must not. An empty table is what a client
        // nobody handed a universe to holds, and it must not be the thing that stops a
        // position being closed.
        let empty = InstrumentTable::new();
        let mut r = req(Side::Buy, Tif::Gtc);
        r.price = Some(dec!(108234));
        assert!(matches!(
            order_wire(&r, empty.precision(SymbolId::new(3))),
            Err(EncodeError::PrecisionUnknown { asset: 3 })
        ));

        r.reduce_only = true;
        let w = order_wire(&r, empty.precision(SymbolId::new(3)))
            .expect("a close goes out unvalidated rather than not at all");
        assert!(w.r);
    }

    #[test]
    fn a_batch_is_refused_whole_when_one_leg_is_off_grid() {
        // A half-submitted batch leaves the caller's plan half-executed and its position
        // somewhere neither the plan nor the venue describes, with no way to find out
        // which half went.
        let t = btc_table();
        let good = OrderRequest::limit(
            SymbolId::new(3),
            Side::Buy,
            dec!(0.001),
            dec!(108234),
            Tif::Gtc,
            Cloid::new(1),
        );
        let bad = OrderRequest::limit(
            SymbolId::new(3),
            Side::Buy,
            dec!(0.001),
            dec!(108234.567),
            Tif::Gtc,
            Cloid::new(2),
        );
        assert!(order_action(std::slice::from_ref(&good), &t).is_ok());
        assert!(order_action(&[good.clone(), bad.clone()], &t).is_err());
        assert!(
            order_action(&[bad, good], &t).is_err(),
            "order within the batch must not decide it"
        );
    }

    #[test]
    fn a_modify_is_checked_at_the_same_wire_the_planner_never_reaches() {
        // `modify` is built by a caller, not by `Planner::plan`, so this is the only
        // thing between a hand-built replacement and a `tickRejected`.
        let t = btc_table();
        let mut r = req(Side::Buy, Tif::Gtc);
        r.price = Some(dec!(108234.567));
        let id = CancelId::OrderId {
            symbol: SymbolId::new(3),
            order_id: OrderId::new(42),
        };
        assert!(matches!(
            modify_action(id, &r, &t),
            Err(EncodeError::Precision(_))
        ));
    }

    #[test]
    fn cancel_actions_serialize() {
        let by_oid = cancel_action(&[(SymbolId::new(3), OrderId::new(9))]);
        assert_eq!(
            serde_json::to_value(&by_oid).unwrap(),
            json!({ "type": "cancel", "cancels": [{ "a": 3, "o": 9 }] })
        );
        let by_cloid = cancel_by_cloid_action(&[(SymbolId::new(3), Cloid::new(255))]);
        assert_eq!(
            serde_json::to_value(&by_cloid).unwrap(),
            json!({ "type": "cancelByCloid", "cancels": [{ "asset": 3, "cloid": "0x000000000000000000000000000000ff" }] })
        );
    }

    #[test]
    fn modify_action_addresses_by_oid_or_cloid() {
        let base = req(Side::Buy, Tif::Gtc);

        let by_oid = modify_action(
            CancelId::OrderId {
                symbol: SymbolId::new(3),
                order_id: OrderId::new(42),
            },
            &base,
            &unconstrained(),
        )
        .unwrap();
        let v = serde_json::to_value(&by_oid).unwrap();
        assert_eq!(v["type"], "modify");
        assert_eq!(v["oid"], 42); // numeric oid
        assert_eq!(v["order"]["a"], 3);
        assert_eq!(v["order"]["t"]["limit"]["tif"], "Gtc");

        let by_cloid = modify_action(
            CancelId::Cloid {
                symbol: SymbolId::new(3),
                cloid: Cloid::new(255),
            },
            &base,
            &unconstrained(),
        )
        .unwrap();
        let v2 = serde_json::to_value(&by_cloid).unwrap();
        assert_eq!(v2["oid"], "0x000000000000000000000000000000ff"); // cloid hex
    }

    #[test]
    fn schedule_cancel_omits_time_when_disarming() {
        // Field ABSENCE is load-bearing: the action is msgpack'd and hashed for the
        // signature, so `"time": null` would sign different bytes than the venue
        // parses and come back as an invalid signature.
        let disarm = serde_json::to_value(schedule_cancel_action(None)).unwrap();
        assert_eq!(disarm, json!({ "type": "scheduleCancel" }));
        assert!(
            disarm.get("time").is_none(),
            "disarm must omit `time` entirely, not send null"
        );

        let arm = serde_json::to_value(schedule_cancel_action(Some(1_784_976_990_000))).unwrap();
        assert_eq!(
            arm,
            json!({ "type": "scheduleCancel", "time": 1_784_976_990_000u64 })
        );
    }

    #[test]
    fn schedule_cancel_msgpack_differs_between_armed_and_disarmed() {
        // The same point, proven at the layer that actually matters: the hashed bytes.
        let disarm = rmp_serde::to_vec_named(&schedule_cancel_action(None)).unwrap();
        let armed = rmp_serde::to_vec_named(&schedule_cancel_action(Some(1))).unwrap();
        assert_ne!(disarm, armed);
        // fixmap(1) for disarm vs fixmap(2) for armed — the map header itself changes.
        assert_eq!(disarm[0], 0x81);
        assert_eq!(armed[0], 0x82);
    }

    #[test]
    fn schedule_cancel_deadline_enforces_the_five_second_minimum() {
        let now = 1_784_976_990_000;
        assert_eq!(schedule_cancel_deadline(now, 5_000).unwrap(), now + 5_000);
        assert_eq!(schedule_cancel_deadline(now, 30_000).unwrap(), now + 30_000);
        // A caller asking for 1 s has the wrong mental model of their protection;
        // clamping silently would hide that, so it is an error.
        assert!(matches!(
            schedule_cancel_deadline(now, 4_999),
            Err(EncodeError::ScheduleCancelTooSoon {
                lead_ms: 4_999,
                min_ms: 5_000
            })
        ));
    }

    #[test]
    fn order_action_signs_end_to_end() {
        // Ties encode + sign together: a real order action signs, the signature is
        // deterministic (RFC-6979) for a fixed (action, nonce), and nonce-sensitive.
        use crate::sign::HlSigner;
        let signer = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            false,
        )
        .unwrap();
        let action = order_action(&[req(Side::Buy, Tif::Gtc)], &unconstrained()).unwrap();

        let a = signer
            .sign_l1_action(&action, 1_700_000_000_000, None, None)
            .unwrap();
        let a_again = signer
            .sign_l1_action(&action, 1_700_000_000_000, None, None)
            .unwrap();
        let b = signer
            .sign_l1_action(&action, 1_700_000_000_001, None, None)
            .unwrap();

        assert!(a.v == 27 || a.v == 28);
        assert_eq!(a, a_again, "same action+nonce → identical signature");
        assert_ne!(a, b, "different nonce → different signature");
    }
}
