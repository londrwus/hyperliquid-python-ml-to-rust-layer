//! # axon-providers
//!
//! The **outward port** of the hexagon (`docs/04-provider-abstraction.md`,
//! ADR-0004). Venues sit behind these traits; every venue quirk — Hyperliquid's
//! wallet signing, ms-timestamp nonces, synthetic market orders, volume-gated
//! rate limits — is absorbed in an *adapter*, never here.
//!
//! Above the line these types define, everything generalizes: order intent, book
//! state, positions/risk, lifecycle events. Adding a venue is implementing
//! [`ExecutionClient`] + [`MarketData`] + [`AccountState`] + a [`Capabilities`]
//! descriptor and an [`InstrumentTable`] — and touching nothing in the core or any
//! other adapter.
//!
//! Two descriptors, because they answer different questions and change at different
//! times: [`Capabilities`] is what the *venue* can express and is compiled in;
//! [`InstrumentTable`] is each *instrument's* number format and arrives over the
//! network at startup (ADR-0025).

#![deny(unsafe_code)]

pub mod capabilities;
pub mod error;
pub mod instrument;
pub mod market;
pub mod order;
pub mod traits;

pub use capabilities::{Capabilities, RateLimitModel};
pub use error::ProviderError;
pub use instrument::{
    InstrumentSpec, InstrumentTable, Precision, PrecisionError, PriceGrid, PriceIntent, SizeGrid,
    SpecError,
};
pub use market::{
    Bbo, BookSnapshot, Candle, CandleInterval, Credentials, Feed, Funding, Level, MarketEvent,
    SignatureScheme, Signer, Ticker, Trade,
};
pub use order::{
    AccountSnapshot, CancelAck, CancelId, CancelReason, ExecEvent, Fill, Liquidity, OrderAck,
    OrderRequest, OrderStatus, OrderUpdate, Trigger, TriggerKind,
};
pub use traits::{AccountState, ExecutionClient, MarketData};
