//! # axon-provider-hyperliquid
//!
//! Hyperliquid adapter — the first concrete venue behind the [`axon_providers`]
//! traits (`docs/04-provider-abstraction.md`, `docs/research/hyperliquid-execution.md`).
//! All Hyperliquid quirks live here and never leak upward:
//!
//! - **Signing:** two EIP-712 schemes — L1 actions (phantom-agent, msgpack hash,
//!   chainId 1337) and user-signed admin actions (direct EIP-712). Mixing them is
//!   the #1 "invalid signature" cause, so they are isolated behind the `Signer`.
//! - **Nonces:** ms-timestamp windows; the venue tracks the 100 highest nonces
//!   per address (one agent wallet per process ⇒ one nonce tracker).
//! - **Market orders:** synthetic — an IOC limit at deep slippage (there is no
//!   native market order), hence `native_market_orders = false`.
//! - **Rate limits:** volume-gated (≈1 request / 1 USDC traded); cancels get a
//!   higher cap, so keep a cancel budget in reserve.
//! - **In-block priority:** cancel → post-only → GTC → IOC.
//!
//! Phase 2 implements read-only WS market data ([`ws`]); Phase 3 adds signing
//! ([`sign`]), execution ([`exchange`]), the authenticated user channels and
//! `POST /info` reads that make reconciliation possible ([`ws::user`], [`info`]),
//! and the [`governor`] that keeps us inside the venue's rate limits while always
//! holding cancel capacity in reserve. This module also ships the [`capabilities`]
//! descriptor the router relies on and the
//! [`SymbolMap`](symbol_map::SymbolMap) coin↔id translation.

#![deny(unsafe_code)]

pub mod encode;
pub mod exchange;
pub mod governor;
pub mod info;
pub mod nonce;
pub mod sign;
pub mod symbol_map;
pub mod ws;

pub use encode::{
    cancel_action, cancel_by_cloid_action, modify_action, order_action, schedule_cancel_action,
    schedule_cancel_deadline, EncodeError, SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY,
    SCHEDULE_CANCEL_MIN_LEAD_MS,
};
pub use exchange::ExchangeClient;
pub use governor::{
    action_weight, info_weight, ActionKind, Budget, Decision, GovernorConfig, RateGovernor, Refusal,
};
pub use info::{
    fetch_extra_agents, fetch_frontend_open_orders, fetch_historical_orders, fetch_open_orders,
    fetch_order_status_by_cloid, fetch_order_status_by_oid, fetch_user_rate_limit,
    fetch_user_state, Decoded, ExtraAgent, OpenOrder, OrderStatusReply, OrderWithStatus,
    RateLimitStatus, UserState,
};
pub use nonce::NonceManager;
pub use sign::{
    hyperliquid_chain, AgentName, ApproveAgent, HlSigner, RpcSignature, SignError,
    SignatureChainId, SignedUserAction, UserSignedAction, MAX_AGENT_VALIDITY_MS,
};
pub use symbol_map::SymbolMap;
pub use ws::{
    decode_universe, decode_user_channel, fetch_universe, hl_status, parse_cloid, venue_error,
    HlError, HyperliquidMarketData, Universe, UserChannel,
};

use axon_core::{Decimal, OrderType, Tif};
use axon_providers::{Capabilities, RateLimitModel, SignatureScheme};

/// Canonical venue name used in capability errors and the `SymbolMap`.
pub const VENUE: &str = "hyperliquid";

/// Max orders per batch call.
pub const MAX_BATCH: u32 = 20;

/// A **perp** price carries at most `PERP_MAX_DECIMALS - szDecimals` decimal places.
///
/// The venue's tick-and-lot rule, half of it. Breaking it is a `tickRejected`, which
/// from inside this process is indistinguishable from a signing bug (ADR-0025).
pub const PERP_MAX_DECIMALS: u32 = 6;

/// Spot's budget for the same rule.
///
/// Named here rather than left to be looked up the day `decode_spot_meta` lands, so
/// there is one place to reach for and no temptation to reuse the perp number. Spot's
/// `szDecimals` also lives somewhere else entirely — `spotMeta.tokens[]`, keyed by the
/// pair's base token in a different index space — which is why spot is refused rather
/// than approximated for now (ADR-0025 §Spot).
pub const SPOT_MAX_DECIMALS: u32 = 8;

/// …and at most this many significant figures, **integers exempt**. The other half.
pub const PRICE_SIG_FIGS: u32 = 5;

/// Hyperliquid's minimum order value, in USDC.
///
/// A number *we* assert and the venue owns: it is not in `meta`, so the only thing
/// that will ever tell us it changed is a `minTradeNtlRejected`. `Decimal::TEN` rather
/// than a macro because `rust_decimal_macros` is a dev-dependency here and this is a
/// real `const`.
pub const MIN_ORDER_NOTIONAL_USD: Decimal = Decimal::TEN;

/// The two L1/user signing schemes this adapter must implement (Phase 3).
pub const SIGNING_SCHEMES: [SignatureScheme; 2] = [
    SignatureScheme::Eip712L1Action,
    SignatureScheme::Eip712UserSigned,
];

/// Hyperliquid's declared capabilities.
///
/// Order types are `Limit` plus a `Market` that the order mapper synthesizes as
/// an IOC-at-slippage limit (`native_market_orders = false`). TIFs are GTC, IOC,
/// and PostOnly (ALO).
pub fn capabilities() -> Capabilities {
    Capabilities {
        venue: VENUE,
        order_types: &[OrderType::Limit, OrderType::Market],
        tifs: &[Tif::Gtc, Tif::Ioc, Tif::PostOnly],
        max_batch: MAX_BATCH,
        native_market_orders: false,
        reduce_only: true,
        rate_limit_model: RateLimitModel::VolumeGated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Decimal, Side, SymbolId};
    use axon_providers::OrderRequest;

    #[test]
    fn descriptor_matches_venue_reality() {
        let c = capabilities();
        assert_eq!(c.venue, "hyperliquid");
        assert_eq!(c.max_batch, 20);
        assert!(!c.native_market_orders); // synthetic market orders
        assert!(matches!(c.rate_limit_model, RateLimitModel::VolumeGated));
    }

    #[test]
    fn accepts_gtc_limit_rejects_fok() {
        let c = capabilities();
        let mut req = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::from(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        assert!(c.check(&req).is_ok());
        req.tif = Tif::Fok; // Hyperliquid does not expose FOK
        assert!(c.check(&req).is_err());
    }
}
