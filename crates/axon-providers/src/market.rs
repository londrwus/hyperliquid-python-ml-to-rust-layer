//! Market-data feeds and the auth/signing seam.
//!
//! The normalized *data* vocabulary ([`Bbo`], [`Trade`], [`BookSnapshot`],
//! [`MarketEvent`], …) is domain vocabulary and lives in `axon-core`; it is
//! **re-exported here** so the [`MarketData`](crate::traits::MarketData) port and
//! adapters speak one set of types. What stays local to this crate is the *port*
//! concerns: which [`Feed`]s can be subscribed, and how requests are signed.
//!
//! The concrete transport by which events reach the core bus was the Phase-2
//! "OPEN" decision — now settled as `axon-core`'s in-process
//! [`bus`](axon_core::bus) (a bounded crossbeam channel; ADR-0008).

use serde::{Deserialize, Serialize};

/// Normalized market-data value types — defined in `axon-core`, re-exported so
/// this port and its adapters share one vocabulary.
pub use axon_core::{
    Bbo, BookSnapshot, Candle, CandleInterval, Funding, Level, MarketEvent, Ticker, Trade,
};

/// A subscribable data feed (normalized across venues).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feed {
    /// Full L2 order book.
    L2Book,
    /// Public trade prints.
    Trades,
    /// Best bid/offer.
    Bbo,
    /// OHLCV candles at an interval.
    Candles(CandleInterval),
    /// Reference prices and perp statistics — mark, index, funding, open interest
    /// ([`Ticker`]). Carries no order-book state; a venue publishes it on its own
    /// cadence, which is why it is a feed of its own rather than a field on
    /// [`Bbo`].
    Ticker,
}

/// The two families of signing scheme. Mixing Hyperliquid's two EIP-712 variants
/// is the #1 "invalid signature" cause — hence one explicit seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureScheme {
    /// CEX HMAC over the request.
    HmacApiKey,
    /// Hyperliquid L1 action (phantom-agent, msgpack hash, chainId 1337).
    Eip712L1Action,
    /// Hyperliquid user-signed admin action (direct EIP-712).
    Eip712UserSigned,
}

/// Credentials — a CEX key pair or a DEX wallet.
#[derive(Debug, Clone)]
pub enum Credentials {
    ApiKey { key: String, secret: String },
    Wallet { address: String },
}

/// The signing seam. Implementations live behind the adapter (agent wallet, KMS,
/// remote signer) so the hot key never leaks upward.
pub trait Signer: Send + Sync {
    fn scheme(&self) -> SignatureScheme;
    /// Sign `payload`, returning the signature bytes.
    fn sign(&self, payload: &[u8]) -> Result<Vec<u8>, crate::ProviderError>;
}
