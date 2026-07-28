//! Order/cancel → the Binance **query string** (the input to [`crate::sign`]).
//!
//! Hyperliquid's `encode` builds a msgpack action whose field order is significant
//! because the same bytes are hashed for the signature. This does the same job with a
//! different alphabet: Binance signs the HMAC over the query string *exactly as
//! sent*, so parameter order is equally load-bearing here — it just looks like a URL
//! instead of a map. That is the whole of the difference at this layer, and it is
//! smaller than it looks.
//!
//! Three things about the venue do surface, and none of them is cosmetic:
//!
//! - **A market order must not carry a `timeInForce`.** The venue answers
//!   `-1106 Parameter 'timeInForce' sent when not required`, so
//!   [`OrderRequest::tif`](axon_providers::OrderRequest) is *dropped* for
//!   `OrderType::Market`. On Hyperliquid a market order is a synthesized IOC limit
//!   and the TIF is the mechanism; here it is a field that must not exist.
//! - **A market order must not carry a price either**, which is the first time
//!   `OrderRequest::price: Option` has been `None` on a live path. The Hyperliquid
//!   encoder refuses that outright (`EncodeError::MissingPrice`), which was right for
//!   a venue with no native market order and would be wrong here.
//! - **A cancel needs the symbol *string*.** `CancelId` carries a
//!   [`SymbolId`], and Hyperliquid's encoder puts that number
//!   straight on the wire because it *is* the venue's asset index. Binance needs the
//!   name, so [`cancel_params`] takes the symbol table that Hyperliquid's equivalent
//!   does not need.
//!
//! **This module never rounds**, for the reason ADR-0025 gives: a price silently
//! changed at the wire is a price the recorded `Plan` does not contain, and that plan
//! is what the chain summary logs, what `WorkingOrder::is()` compares on the next
//! pass, and what the parity harness diffs. It refuses instead.

use axon_core::{Cloid, Decimal, OrderType, Side, SymbolId, Tif};
use axon_providers::{CancelId, OrderRequest, Precision, PrecisionError};

use crate::symbols::SymbolTable;
use crate::{MAX_CLIENT_ORDER_ID_LEN, RECV_WINDOW_MS};

/// Why an order could not be encoded for Binance.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("a limit order requires a price")]
    MissingPrice,
    /// A market order arrived with a price. Refused rather than dropped: the caller
    /// computed a number and believes it will be honoured, and the venue would ignore
    /// it silently.
    #[error("a market order must not carry a price ({0}); the venue would ignore it")]
    PriceOnMarketOrder(Decimal),
    #[error("trigger orders are not encoded yet (the venue has STOP/TAKE_PROFIT types)")]
    UnsupportedTrigger,
    /// The order breaks the instrument's own number format. **Not** rounded here.
    #[error("this order breaks the instrument's grid: {0}")]
    Precision(#[from] PrecisionError),
    /// We hold no grid for this instrument, and the order would add exposure.
    #[error("no precision is known for {symbol}; refusing anything that adds exposure")]
    PrecisionUnknown { symbol: SymbolId },
    /// A `SymbolId` with no entry in the table.
    #[error("no venue symbol for {0}")]
    UnknownSymbol(SymbolId),
    /// A parameter value that cannot go into a query string unescaped.
    ///
    /// Unreachable for values this module generates — every one is a decimal, a hex
    /// cloid, a venue keyword or a boolean — and it is a refusal rather than an
    /// escape because the value would then have to be escaped *identically* in the
    /// string we sign and the string we send. A mismatch there is an invalid
    /// signature, which is the single most expensive thing to debug on any venue.
    #[error(
        "parameter {key}={value:?} needs escaping; refusing rather than signing two \
             different strings"
    )]
    UnsafeValue { key: &'static str, value: String },
    /// A client order id longer than the venue accepts.
    #[error("client order id {id:?} is {len} bytes, over the venue's {max}")]
    ClientOrderIdTooLong { id: String, len: usize, max: usize },
}

/// An ordered parameter list.
///
/// Ordered, and a `Vec` rather than a map, because the signature is taken over the
/// serialized string: two orderings are two different signatures over the same
/// request. A `HashMap` here would make the wire bytes depend on hash iteration order
/// and the failures would be intermittent — the worst possible shape for a signing
/// bug.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Params(Vec<(&'static str, String)>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, key: &'static str, value: impl Into<String>) {
        self.0.push((key, value.into()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> + '_ {
        self.0.iter().map(|(k, v)| (*k, v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `k=v&k=v&…`, in insertion order, with every value checked to be safe raw.
    ///
    /// This string is both what gets signed and what gets sent. There is exactly one
    /// producer of it so the two cannot diverge — the failure that would cause is an
    /// `-1022 Signature for this request is not valid`, which from inside the process
    /// is indistinguishable from a wrong secret.
    pub fn query_string(&self) -> Result<String, EncodeError> {
        let mut out = String::new();
        for (k, v) in &self.0 {
            if !is_query_safe(v) {
                return Err(EncodeError::UnsafeValue {
                    key: k,
                    value: v.clone(),
                });
            }
            if !out.is_empty() {
                out.push('&');
            }
            out.push_str(k);
            out.push('=');
            out.push_str(v);
        }
        Ok(out)
    }
}

/// Whether a value can appear in a query string with no escaping.
///
/// Deliberately narrow: alphanumerics and the four characters our own values use.
/// Widening it is a decision about the signature, not about URLs.
fn is_query_safe(v: &str) -> bool {
    !v.is_empty()
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'x'))
}

/// Format a fixed-point value for the wire: no trailing zeros, no exponent.
///
/// This *only* formats. [`order_params`] refuses anything not already on the
/// instrument's grid, so a change to this formatter can never quietly become a change
/// to the rounding.
fn wire_decimal(d: Decimal) -> String {
    d.normalize().to_string()
}

/// A 128-bit [`Cloid`] as the venue's `newClientOrderId`.
///
/// Hex, not decimal, and the choice is forced. Binance accepts at most
/// [`MAX_CLIENT_ORDER_ID_LEN`] characters from `^[\.A-Z\:/a-z0-9_-]{1,36}$`; the hex
/// form is a constant 34 characters and a decimal `u128` runs to 39. It also matches
/// the Hyperliquid adapter's `cloid_hex` byte for byte, which is not required by
/// anything and is worth keeping: a `cloid` that reads the same in two venues' logs
/// is one a human can correlate.
pub fn client_order_id(c: Cloid) -> String {
    format!("0x{:032x}", c.get())
}

const fn side_str(side: Side) -> &'static str {
    match side {
        Side::Buy => "BUY",
        Side::Sell => "SELL",
    }
}

/// Binance's `timeInForce` token.
///
/// All four normalized TIFs exist here, which is one more than Hyperliquid offers —
/// `Tif::Fok` is refused there and is a first-class `FOK` here. `GTX` is the venue's
/// name for post-only, and it behaves the way ADR-0025 §3 assumes: an order that
/// would cross is **rejected**, not demoted, which is what makes rounding a passive
/// price away from the market the correct direction on this venue too.
const fn tif_str(tif: Tif) -> &'static str {
    match tif {
        Tif::Gtc => "GTC",
        Tif::Ioc => "IOC",
        Tif::Fok => "FOK",
        Tif::PostOnly => "GTX",
    }
}

/// Encode one [`OrderRequest`] into the parameters for `POST /fapi/v1/order`.
///
/// `precision` is a **required argument and a three-way enum, not an `Option`** — the
/// same structural move `order_wire` makes on Hyperliquid, and it transferred without
/// a change: an adapter author cannot encode an order without naming what is known
/// about the instrument's grid, and `Unconstrained` is a word that has to be typed.
///
/// `timestamp_ms` is an argument rather than a clock read, so this function is pure
/// and its output is byte-comparable in a test. Binance has no nonce; the timestamp
/// *is* the replay window, and a hidden clock read here would make every signature
/// assertion non-deterministic.
pub fn order_params(
    req: &OrderRequest,
    symbols: &SymbolTable,
    precision: Precision<'_>,
    timestamp_ms: u64,
) -> Result<Params, EncodeError> {
    match precision {
        Precision::Known(spec) => spec.check(req)?,
        Precision::Unconstrained => {}
        Precision::Unknown => {
            // ADR-0010's asymmetry, applied again and unchanged by the venue: whatever
            // went wrong, removing exposure must keep working while adding it must not.
            if !req.reduce_only {
                return Err(EncodeError::PrecisionUnknown {
                    symbol: req.symbol_id,
                });
            }
        }
    }
    if req.trigger.is_some() {
        return Err(EncodeError::UnsupportedTrigger);
    }
    let symbol = symbols
        .symbol(req.symbol_id)
        .ok_or(EncodeError::UnknownSymbol(req.symbol_id))?;

    let mut p = Params::new();
    p.push("symbol", symbol);
    p.push("side", side_str(req.side));
    match req.order_type {
        OrderType::Limit => {
            let price = req.price.ok_or(EncodeError::MissingPrice)?;
            p.push("type", "LIMIT");
            p.push("timeInForce", tif_str(req.tif));
            p.push("quantity", wire_decimal(req.qty));
            p.push("price", wire_decimal(price));
        }
        OrderType::Market => {
            // Both omissions are refusals in disguise. A `timeInForce` on a market
            // order is `-1106`; a price on one is accepted and ignored, which is worse,
            // because the caller computed a number and believes it binds.
            if let Some(px) = req.price {
                return Err(EncodeError::PriceOnMarketOrder(px));
            }
            p.push("type", "MARKET");
            p.push("quantity", wire_decimal(req.qty));
        }
    }
    // Sent explicitly in both directions rather than only when true. The venue's
    // default depends on the account's position mode, which this process cannot see;
    // an omitted parameter means "whatever that account is configured for" and a
    // reduce-only intent silently becoming an opening one is the failure that costs
    // the most.
    p.push("reduceOnly", if req.reduce_only { "true" } else { "false" });
    let cloid = client_order_id(req.cloid);
    if cloid.len() > MAX_CLIENT_ORDER_ID_LEN {
        return Err(EncodeError::ClientOrderIdTooLong {
            len: cloid.len(),
            id: cloid,
            max: MAX_CLIENT_ORDER_ID_LEN,
        });
    }
    p.push("newClientOrderId", cloid);
    p.push("recvWindow", RECV_WINDOW_MS.to_string());
    p.push("timestamp", timestamp_ms.to_string());
    Ok(p)
}

/// Encode a cancel into the parameters for `DELETE /fapi/v1/order`.
///
/// Takes `symbols` because [`CancelId`] carries a [`SymbolId`] and this venue needs
/// the name. That one argument is the whole shape of the difference: Hyperliquid's
/// `cancel_action` puts the id straight on the wire, because there the id *is* the
/// venue's asset index.
pub fn cancel_params(
    id: CancelId,
    symbols: &SymbolTable,
    timestamp_ms: u64,
) -> Result<Params, EncodeError> {
    let symbol = symbols
        .symbol(id.symbol())
        .ok_or(EncodeError::UnknownSymbol(id.symbol()))?;
    let mut p = Params::new();
    p.push("symbol", symbol);
    match id {
        CancelId::OrderId { order_id, .. } => p.push("orderId", order_id.get().to_string()),
        // `origClientOrderId`, not `newClientOrderId`: the venue uses different
        // parameter names for the id you are *assigning* and the id you are
        // *addressing*, and sending the wrong one cancels nothing and reports success.
        CancelId::Cloid { cloid, .. } => p.push("origClientOrderId", client_order_id(cloid)),
    }
    p.push("recvWindow", RECV_WINDOW_MS.to_string());
    p.push("timestamp", timestamp_ms.to_string());
    Ok(p)
}

/// Encode a per-symbol cancel-all into the parameters for
/// `DELETE /fapi/v1/allOpenOrders`.
///
/// **Per symbol.** The port's [`ExecutionClient::cancel_all`](axon_providers::ExecutionClient)
/// takes no argument, so a venue-wide flatten is N of these — one per instrument the
/// session holds orders on — and there is no single request that means "cancel
/// everything". Hyperliquid has the same hole and closes it by reading open orders
/// first; the port models neither, which is a real gap and is recorded in ADR-0023
/// rather than papered over here.
pub fn cancel_all_params(
    symbol: SymbolId,
    symbols: &SymbolTable,
    timestamp_ms: u64,
) -> Result<Params, EncodeError> {
    let name = symbols
        .symbol(symbol)
        .ok_or(EncodeError::UnknownSymbol(symbol))?;
    let mut p = Params::new();
    p.push("symbol", name);
    p.push("recvWindow", RECV_WINDOW_MS.to_string());
    p.push("timestamp", timestamp_ms.to_string());
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::OrderId;
    use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid};
    use rust_decimal_macros::dec;

    const TS: u64 = 1_591_702_613_943;

    fn symbols() -> SymbolTable {
        SymbolTable::from_ordered(["BTCUSDT", "ETHUSDT"])
    }

    /// BTCUSDT's real filters from the captured `exchangeInfo`: tick 0.10, lot 0.0001,
    /// minimum notional 50 USDT.
    fn btc_table() -> InstrumentTable {
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: SymbolId::new(0),
            price: PriceGrid::increment(dec!(0.10)).unwrap(),
            size: SizeGrid::step(dec!(0.0001)).unwrap(),
            min_notional: Some(dec!(50)),
        });
        t
    }

    fn limit(side: Side, tif: Tif) -> OrderRequest {
        OrderRequest::limit(
            SymbolId::new(0),
            side,
            dec!(0.0100),
            dec!(64346.70),
            tif,
            Cloid::new(255),
        )
    }

    fn market(qty: Decimal) -> OrderRequest {
        OrderRequest {
            symbol_id: SymbolId::new(0),
            side: Side::Buy,
            qty,
            price: None,
            order_type: OrderType::Market,
            tif: Tif::Ioc,
            reduce_only: false,
            trigger: None,
            cloid: Cloid::new(255),
        }
    }

    #[test]
    fn encodes_a_gtc_buy_limit_into_the_exact_string_that_gets_signed() {
        let t = btc_table();
        let p = order_params(
            &limit(Side::Buy, Tif::Gtc),
            &symbols(),
            t.precision(SymbolId::new(0)),
            TS,
        )
        .unwrap();
        assert_eq!(
            p.query_string().unwrap(),
            "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=0.01&price=64346.7\
             &reduceOnly=false&newClientOrderId=0x000000000000000000000000000000ff\
             &recvWindow=5000&timestamp=1591702613943"
        );
    }

    #[test]
    fn a_market_order_carries_neither_a_price_nor_a_time_in_force() {
        // Both would be errors at the venue, and only one of them is loud. `-1106` for
        // the TIF; a price is accepted and *ignored*, which means the caller's computed
        // number silently does not bind. This is also the first path in the workspace
        // where `OrderRequest::price` is legitimately `None` — Hyperliquid's encoder
        // refuses that outright, correctly, for a venue with no native market order.
        let p = order_params(
            &market(dec!(0.01)),
            &symbols(),
            Precision::Unconstrained,
            TS,
        )
        .unwrap();
        assert_eq!(p.get("type"), Some("MARKET"));
        assert_eq!(p.get("price"), None);
        assert_eq!(p.get("timeInForce"), None, "-1106 if this is sent");
        assert_eq!(p.get("quantity"), Some("0.01"));

        // …and a price on one is refused rather than dropped.
        let mut m = market(dec!(0.01));
        m.price = Some(dec!(64346.70));
        assert!(matches!(
            order_params(&m, &symbols(), Precision::Unconstrained, TS),
            Err(EncodeError::PriceOnMarketOrder(_))
        ));
    }

    #[test]
    fn every_normalized_tif_maps_including_the_one_the_first_venue_refuses() {
        for (tif, wire) in [
            (Tif::Gtc, "GTC"),
            (Tif::Ioc, "IOC"),
            (Tif::Fok, "FOK"),
            (Tif::PostOnly, "GTX"),
        ] {
            let p = order_params(
                &limit(Side::Sell, tif),
                &symbols(),
                Precision::Unconstrained,
                TS,
            )
            .unwrap();
            assert_eq!(p.get("timeInForce"), Some(wire));
            assert_eq!(p.get("side"), Some("SELL"));
        }
    }

    #[test]
    fn an_off_tick_price_is_refused_at_the_wire_rather_than_silently_rounded() {
        // ADR-0025's rule, unchanged by the venue: the planner rounds, the encoder
        // refuses. Rounding here would make the bytes sent differ from the `Plan` that
        // was recorded — the plan the chain summary logs, the plan `WorkingOrder::is()`
        // compares next pass, the plan the parity harness diffs.
        let t = btc_table();
        let mut r = limit(Side::Buy, Tif::Gtc);
        r.price = Some(dec!(64346.75)); // half a tick
        let err = order_params(&r, &symbols(), t.precision(SymbolId::new(0)), TS).unwrap_err();
        assert!(
            matches!(err, EncodeError::Precision(PrecisionError::Tick { .. })),
            "got {err:?}"
        );
        r.price = Some(dec!(64346.70));
        assert!(order_params(&r, &symbols(), t.precision(SymbolId::new(0)), TS).is_ok());
    }

    #[test]
    fn an_off_step_size_is_refused_at_the_wire() {
        let t = btc_table();
        let mut r = limit(Side::Buy, Tif::Gtc);
        r.qty = dec!(0.00123456); // eight decimals against a 0.0001 step
        assert!(matches!(
            order_params(&r, &symbols(), t.precision(SymbolId::new(0)), TS),
            Err(EncodeError::Precision(PrecisionError::Lot { .. }))
        ));
    }

    #[test]
    fn an_order_below_this_symbols_own_minimum_notional_is_refused_unless_it_reduces() {
        // BTCUSDT's minimum is 50 USDT — not 10, not 5, not a constant in our source.
        // The reduce-only exemption is ADR-0025 §4's, and it transferred verbatim: we
        // never refuse to de-risk on our own opinion.
        let t = btc_table();
        let mut r = limit(Side::Buy, Tif::Gtc);
        r.qty = dec!(0.0001); // 0.0001 * 64346.70 = 6.43 USDT
        assert!(matches!(
            order_params(&r, &symbols(), t.precision(SymbolId::new(0)), TS),
            Err(EncodeError::Precision(
                PrecisionError::BelowMinNotional { .. }
            ))
        ));
        r.reduce_only = true;
        let p = order_params(&r, &symbols(), t.precision(SymbolId::new(0)), TS).unwrap();
        assert_eq!(p.get("reduceOnly"), Some("true"));
    }

    #[test]
    fn an_instrument_with_no_grid_is_refused_unless_the_order_reduces_exposure() {
        // The same asymmetry as the halt switch and the risk gate. An empty table is
        // what a client nobody handed a universe to holds, and it must not be the thing
        // that stops a position being closed.
        let empty = InstrumentTable::new();
        let mut r = limit(Side::Buy, Tif::Gtc);
        assert!(matches!(
            order_params(&r, &symbols(), empty.precision(SymbolId::new(0)), TS),
            Err(EncodeError::PrecisionUnknown { .. })
        ));
        r.reduce_only = true;
        assert!(order_params(&r, &symbols(), empty.precision(SymbolId::new(0)), TS).is_ok());
    }

    #[test]
    fn reduce_only_is_stated_in_both_directions_because_the_default_is_an_account_setting() {
        // The venue's default depends on the account's position mode, which this
        // process cannot observe. An omitted parameter means "whatever that account is
        // configured for", and a reduce-only intent that silently opens is the most
        // expensive way for a default to be wrong.
        let open = order_params(
            &limit(Side::Buy, Tif::Gtc),
            &symbols(),
            Precision::Unconstrained,
            TS,
        )
        .unwrap();
        assert_eq!(open.get("reduceOnly"), Some("false"));
        let mut r = limit(Side::Sell, Tif::Gtc);
        r.reduce_only = true;
        let close = order_params(&r, &symbols(), Precision::Unconstrained, TS).unwrap();
        assert_eq!(close.get("reduceOnly"), Some("true"));
    }

    #[test]
    fn a_cancel_addresses_by_the_id_it_was_given_and_never_by_the_assignment_parameter() {
        // `newClientOrderId` assigns; `origClientOrderId` addresses. Sending the former
        // on a cancel matches nothing, and the venue answers that it cancelled nothing —
        // which reads like an order that had already gone.
        let by_oid = cancel_params(
            CancelId::OrderId {
                symbol: SymbolId::new(0),
                order_id: OrderId::new(42),
            },
            &symbols(),
            TS,
        )
        .unwrap();
        assert_eq!(
            by_oid.query_string().unwrap(),
            "symbol=BTCUSDT&orderId=42&recvWindow=5000&timestamp=1591702613943"
        );

        let by_cloid = cancel_params(
            CancelId::Cloid {
                symbol: SymbolId::new(1),
                cloid: Cloid::new(255),
            },
            &symbols(),
            TS,
        )
        .unwrap();
        assert_eq!(
            by_cloid.get("origClientOrderId"),
            Some("0x000000000000000000000000000000ff")
        );
        assert_eq!(by_cloid.get("newClientOrderId"), None);
        assert_eq!(by_cloid.get("symbol"), Some("ETHUSDT"));
    }

    #[test]
    fn a_cancel_all_names_one_symbol_because_the_venue_has_no_request_that_means_everything() {
        // `ExecutionClient::cancel_all()` takes no argument, so a venue-wide flatten is
        // N requests. Recorded as a test rather than a comment because it is a port gap
        // both venues share and neither expresses.
        let p = cancel_all_params(SymbolId::new(0), &symbols(), TS).unwrap();
        assert_eq!(p.get("symbol"), Some("BTCUSDT"));
        assert!(matches!(
            cancel_all_params(SymbolId::new(9), &symbols(), TS),
            Err(EncodeError::UnknownSymbol(_))
        ));
    }

    #[test]
    fn a_symbol_id_with_no_venue_name_is_refused_rather_than_sent_as_a_number() {
        // The difference from Hyperliquid in one assertion. There the id *is* the asset
        // index and goes on the wire; here it is ours, means nothing to the venue, and
        // an order built from an unmapped id would name a symbol that does not exist.
        let r = OrderRequest::limit(
            SymbolId::new(7),
            Side::Buy,
            dec!(1),
            dec!(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        assert!(matches!(
            order_params(&r, &symbols(), Precision::Unconstrained, TS),
            Err(EncodeError::UnknownSymbol(SymbolId(7)))
        ));
    }

    #[test]
    fn a_trigger_order_is_refused_by_name_rather_than_encoded_as_a_plain_limit() {
        // The venue has STOP and TAKE_PROFIT types; this increment does not encode
        // them. Dropping the trigger would send a resting limit where the caller asked
        // for a stop — an order that is valid, accepted, and not the one requested.
        let mut r = limit(Side::Buy, Tif::Gtc);
        r.trigger = Some(axon_providers::Trigger {
            price: dec!(60000),
            kind: axon_providers::TriggerKind::StopLoss,
            market: true,
        });
        assert!(matches!(
            order_params(&r, &symbols(), Precision::Unconstrained, TS),
            Err(EncodeError::UnsupportedTrigger)
        ));
    }

    #[test]
    fn parameter_order_is_preserved_because_two_orderings_are_two_signatures() {
        // A `HashMap` here would make the signed bytes depend on hash iteration order,
        // and the resulting `-1022` would be intermittent — the worst shape a signing
        // bug can take.
        let p = order_params(
            &limit(Side::Buy, Tif::Gtc),
            &symbols(),
            Precision::Unconstrained,
            TS,
        )
        .unwrap();
        let keys: Vec<&str> = p.iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec![
                "symbol",
                "side",
                "type",
                "timeInForce",
                "quantity",
                "price",
                "reduceOnly",
                "newClientOrderId",
                "recvWindow",
                "timestamp",
            ]
        );
        // `timestamp` last of the signed parameters, so `signature` appends cleanly
        // after it — the venue requires the signature to be the final parameter.
        assert_eq!(keys.last(), Some(&"timestamp"));
    }

    #[test]
    fn a_value_that_would_need_escaping_is_refused_rather_than_escaped() {
        // Unreachable through `order_params` — every value it produces is a decimal, a
        // hex cloid, a venue keyword or a boolean — and the refusal exists because an
        // escaped value would have to be escaped *identically* in the string we sign
        // and the string we send. A mismatch is an invalid signature, and an invalid
        // signature is indistinguishable from a wrong secret.
        let mut p = Params::new();
        p.push("symbol", "BTC USDT");
        assert!(matches!(
            p.query_string(),
            Err(EncodeError::UnsafeValue { key: "symbol", .. })
        ));
        let mut ok = Params::new();
        ok.push("newClientOrderId", client_order_id(Cloid::new(u128::MAX)));
        assert!(ok.query_string().is_ok(), "the hex cloid is safe raw");
    }

    #[test]
    fn a_cloid_is_hex_because_the_decimal_form_would_not_fit_the_venues_field() {
        // 34 characters against a 36-character cap, constant width regardless of value.
        // It is also byte-identical to the Hyperliquid adapter's `cloid_hex`, which
        // nothing requires and which makes one order correlatable across two venues'
        // logs by eye.
        assert_eq!(
            client_order_id(Cloid::new(255)),
            "0x000000000000000000000000000000ff"
        );
        assert_eq!(client_order_id(Cloid::new(u128::MAX)).len(), 34);
        assert!(client_order_id(Cloid::new(u128::MAX)).len() <= MAX_CLIENT_ORDER_ID_LEN);
    }

    #[test]
    fn a_price_on_the_grid_keeps_its_value_through_the_formatter() {
        // The formatter only trims; it must never move a number. `64346.70` on a 0.10
        // tick becomes `64346.7`, which is the same value at a different scale and is
        // what the venue's own book prints back.
        assert_eq!(wire_decimal(dec!(64346.70)), "64346.7");
        assert_eq!(wire_decimal(dec!(0.0100)), "0.01");
        assert_eq!(wire_decimal(dec!(3)), "3");
        assert_eq!(
            Decimal::from_str_exact(&wire_decimal(dec!(64346.70))).unwrap(),
            dec!(64346.70)
        );
    }
}
