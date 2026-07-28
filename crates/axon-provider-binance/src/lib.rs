//! # axon-provider-binance
//!
//! Binance **USD-M futures** adapter — the second venue behind the [`axon_providers`]
//! traits, and therefore the first real test of ADR-0004's claim that adding a venue
//! is one adapter and nothing else changes (ADR-0023).
//!
//! **This adapter has never been run against Binance.** Every decoder here is pinned
//! to frames captured off the venue's own socket, and every path that could reach the
//! network is behind `#[ignore]`. Nothing in it has placed an order, and no code in
//! this crate can: there is no HTTP client for `/fapi/v1/order` and no HMAC, by
//! choice (see [`sign`]). Read "offline-verified", never "proven".
//!
//! ## What is Hyperliquid-shaped and what is not
//!
//! The parts that transferred without argument: the normalized vocabulary
//! ([`MarketEvent`](axon_core::MarketEvent), [`Ticker`](axon_core::Ticker),
//! [`Funding`](axon_core::Funding)), the [`Capabilities`] descriptor, the
//! [`InstrumentTable`](axon_providers::InstrumentTable) and its
//! [`PriceGrid`](axon_providers::PriceGrid)/[`SizeGrid`](axon_providers::SizeGrid),
//! the [`MarketData`](axon_providers::MarketData) port, and the bus. The venue
//! differences that had to be absorbed *here* rather than upstream:
//!
//! - **The venue publishes no asset index.** Hyperliquid's `SymbolId` *is* the number
//!   that goes on the wire; Binance addresses everything by symbol string, so ids are
//!   ours to invent. See [`symbols::SymbolTable`] for the ordering that makes them
//!   deterministic and for the case where they still move.
//! - **A price grid is a fixed `tickSize`, not a significant-figure rule.**
//!   `PriceGrid` composes both, so this is one constructor and no new enum arm —
//!   which is the single strongest piece of evidence that ADR-0025 got the shape
//!   right.
//! - **The mark-price stream carries the venue's own event time**, so
//!   [`Ticker::ts_venue`](axon_core::Ticker::ts_venue) is `Some` here and `None` on
//!   Hyperliquid. ADR-0011 modelled that absence as an `Option` on the strength of one
//!   venue; a second venue is what turns that from a hunch into a fact.
//! - **Two streams share one event type.** `<sym>@depth20@100ms` (a snapshot) and
//!   `<sym>@depth@100ms` (a diff) both arrive as `"e":"depthUpdate"` with the same
//!   fields. Nothing in the payload distinguishes them, so this adapter reads the
//!   *stream name* and refuses the diff outright ([`ws::decode`]).
//!
//! ## Layout
//!
//! - [`symbols`] — symbol string ↔ [`SymbolId`](axon_core::SymbolId), and the funding
//!   interval the mark-price stream never carries.
//! - [`exchange_info`] — `GET /fapi/v1/exchangeInfo` → the [`InstrumentTable`](axon_providers::InstrumentTable).
//! - [`ws`] — stream naming, the pure decoders, the REST reads, and the live client.
//! - [`encode`] — order/cancel → the exact query string Binance signs.
//! - [`sign`] — the signing seam, and what is deliberately missing from it.

#![deny(unsafe_code)]

pub mod encode;
pub mod exchange_info;
pub mod sign;
pub mod symbols;
pub mod ws;

pub use encode::{
    cancel_all_params, cancel_params, client_order_id, order_params, EncodeError, Params,
};
pub use exchange_info::{
    decode_exchange_info, decode_funding_info, ExchangeInfoError, SkipReason, Skipped, Universe,
};
pub use sign::{
    hex_lower, sign_request, signing_payload, BinanceCredentials, SignedRequest, API_KEY_HEADER,
    SIGNATURE_PARAM,
};
pub use symbols::{SymbolTable, SymbolTableError};
pub use ws::{
    decode_rest_depth, decode_ws_message, fetch_universe, venue_error, BinanceMarketData, BnError,
    BookDepth, DecodeError, StreamKind, UpdateSpeed,
};

use axon_core::{Nanos, OrderType, Tif};
use axon_providers::{Capabilities, RateLimitModel, SignatureScheme};

/// Canonical venue name, used in capability errors and anywhere a venue is named.
///
/// **Not** `"binance"`. Binance runs three separate matching engines behind three
/// separate wire formats — spot, USD-M futures, COIN-M futures — with different
/// endpoints, different symbol universes and, on spot, a partial-depth frame that
/// carries no event time at all. A `SymbolId` resolved against one of them and used
/// against another addresses a different instrument, successfully and silently. The
/// suffix is what stops a future spot adapter from sharing this one's name.
pub const VENUE: &str = "binance-usdm";

/// Max orders per `POST /fapi/v1/batchOrders` call.
///
/// Five, against Hyperliquid's twenty — which is exactly why `max_batch` is a
/// [`Capabilities`] field and not a constant anybody upstream may assume.
pub const MAX_BATCH: u32 = 5;

/// How long a signed request stays valid, in milliseconds.
///
/// Binance has no nonce. A request is admitted when
/// `timestamp <= serverTime + 1000` and `serverTime - timestamp <= recvWindow`, so
/// replay protection is a *time window* rather than Hyperliquid's monotonic
/// per-address counter. Nothing upstream cares — the port never mentions a nonce —
/// but it means this adapter needs no `NonceManager` and gets no idempotency from
/// one either: the only thing that makes a resend safe is `newClientOrderId`, i.e.
/// our own [`Cloid`](axon_core::Cloid).
pub const RECV_WINDOW_MS: u64 = 5_000;

/// The venue's default funding period, in hours, for a symbol
/// `GET /fapi/v1/fundingInfo` does not mention.
///
/// The mark-price stream carries a funding *rate* and the *next* funding time, and
/// never the period between them. `fundingInfo` publishes `fundingIntervalHours` per
/// symbol — 616 entries against a 730-row `exchangeInfo` when this crate was written,
/// every one of them 8 — and says nothing about the rest.
/// Eight hours is the documented default, and it is an *assumption* exactly like
/// Hyperliquid's `ASSUMED_FUNDING_INTERVAL_NS`: nothing on the wire would contradict
/// it, every carry number would just quietly scale, and the only symptom is a funding
/// P&L that never reconciles.
///
/// The difference from Hyperliquid is that here the venue *does* publish the answer
/// for most symbols, so [`SymbolTable`] carries a per-symbol interval and this
/// constant is only the fallback.
pub const DEFAULT_FUNDING_INTERVAL_HOURS: i64 = 8;

/// [`DEFAULT_FUNDING_INTERVAL_HOURS`] in nanoseconds, ready for
/// [`Funding::interval`](axon_core::Funding::interval).
pub const DEFAULT_FUNDING_INTERVAL_NS: Nanos =
    DEFAULT_FUNDING_INTERVAL_HOURS * 3_600 * 1_000_000_000;

/// Binance timestamps are milliseconds; the core keys on nanoseconds. One conversion
/// factor for the whole crate, so no decoder can invent a second one.
pub const MS_TO_NS: Nanos = 1_000_000;

/// Max length of `newClientOrderId`, in bytes.
///
/// A real constraint on a port type: [`Cloid`](axon_core::Cloid) is a `u128` and
/// Binance wants a string of at most 36 characters from
/// `^[\.A-Z\:/a-z0-9_-]{1,36}$`. The hex form `0x{:032x}` is 34 characters and every
/// character is in the class, so it fits — but only just, and a decimal `u128` at up
/// to 39 digits would not. [`encode::client_order_id`] is where that is decided.
pub const MAX_CLIENT_ORDER_ID_LEN: usize = 36;

/// The signing scheme this adapter uses: HMAC-SHA256 over a query string.
///
/// One value where Hyperliquid needs two. Its `SIGNING_SCHEMES` array exists because
/// mixing L1-action and user-signed EIP-712 is the venue's #1 invalid-signature
/// cause; Binance has no such split, and the port already had
/// [`SignatureScheme::HmacApiKey`] waiting for it.
pub const SIGNING_SCHEME: SignatureScheme = SignatureScheme::HmacApiKey;

/// Binance USD-M futures' declared capabilities.
///
/// Two differences from Hyperliquid worth reading as evidence about the descriptor:
/// `native_market_orders` is **true** (there is a real `MARKET` type, so no
/// IOC-at-slippage synthesis and no price to compute), and [`Tif::Fok`] is
/// **supported**, which is the one TIF Hyperliquid refuses. Both are already
/// expressible; neither needed a new field.
pub fn capabilities() -> Capabilities {
    Capabilities {
        venue: VENUE,
        order_types: &[OrderType::Limit, OrderType::Market],
        // GTC / IOC / FOK / GTX (post-only). The venue also offers GTD, which the
        // normalized `Tif` has no word for — see ADR-0023 on why that is a missing
        // *order lifetime*, not a missing TIF.
        tifs: &[Tif::Gtc, Tif::Ioc, Tif::PostOnly, Tif::Fok],
        max_batch: MAX_BATCH,
        native_market_orders: true,
        reduce_only: true,
        rate_limit_model: RateLimitModel::WeightPerIp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Decimal, Side, SymbolId};
    use axon_providers::OrderRequest;

    #[test]
    fn the_descriptor_says_the_two_things_that_differ_from_the_first_venue() {
        // Both of these are places the router would otherwise carry a venue's name.
        // Hyperliquid synthesizes a market order as an IOC limit at deep slippage and
        // has no FOK at all; if either fact had been assumed anywhere above the
        // adapter, this is the test that would not compile.
        let c = capabilities();
        assert_eq!(c.venue, "binance-usdm");
        assert!(
            c.native_market_orders,
            "a real MARKET type means no synthesized price"
        );
        assert!(c.supports_tif(Tif::Fok));
        assert!(c.supports_tif(Tif::PostOnly), "GTX");
        assert_eq!(c.max_batch, 5, "not Hyperliquid's 20");
        assert!(matches!(c.rate_limit_model, RateLimitModel::WeightPerIp));
    }

    #[test]
    fn a_fok_order_the_first_venue_refuses_is_accepted_here_by_the_same_check() {
        // The capability check is venue-agnostic code running against two different
        // descriptors and reaching two different answers. That is the whole claim of
        // ADR-0004 in one assertion.
        let mut req = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::from(100),
            Tif::Fok,
            Cloid::new(1),
        );
        assert!(capabilities().check(&req).is_ok());
        req.reduce_only = true;
        assert!(capabilities().check(&req).is_ok());
    }

    #[test]
    fn the_default_funding_interval_is_eight_hours_and_not_hyperliquids_one() {
        // The 8x error ADR-0011 named. A rate published per eight hours, read as an
        // hourly one, is an expected-carry number wrong by the ratio with nothing on
        // the wire to reveal it — and the day it happens is the day a second venue is
        // added, which is today.
        assert_eq!(DEFAULT_FUNDING_INTERVAL_NS, 8 * 3_600 * 1_000_000_000);
        assert_ne!(DEFAULT_FUNDING_INTERVAL_NS, 3_600 * 1_000_000_000);
    }

    #[test]
    fn a_hex_cloid_fits_the_venues_client_order_id_budget_and_a_decimal_one_would_not() {
        // 34 characters against a 36-character cap. The decimal form of a u128 runs to
        // 39, so the choice of representation in `encode` is load-bearing rather than
        // cosmetic: it is the difference between an idempotent resend and a
        // `-1100 Illegal characters found in parameter 'newClientOrderId'`.
        assert_eq!(format!("0x{:032x}", u128::MAX).len(), 34);
        assert!(format!("0x{:032x}", u128::MAX).len() <= MAX_CLIENT_ORDER_ID_LEN);
        assert!(u128::MAX.to_string().len() > MAX_CLIENT_ORDER_ID_LEN);
    }
}
