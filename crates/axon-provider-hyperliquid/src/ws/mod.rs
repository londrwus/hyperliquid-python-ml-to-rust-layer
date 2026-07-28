//! Hyperliquid streaming ingest: WS market data **and** account-scoped execution
//! reports, plus REST snapshot seeding.
//!
//! - [`decode`] — pure JSON → normalized [`Event`](axon_core::Event) dispatcher for
//!   the public market feeds (offline-tested, the heart of the adapter).
//! - [`user`] — pure decoders for the *user* channels (`userEvents`, `userFills`,
//!   `orderUpdates`) → normalized [`ExecEvent`](axon_core::ExecEvent).
//! - [`sub`] — subscription request builders, for coin feeds and user channels.
//! - [`funding`] — the funding interval the ticker frame never carries, and the
//!   cross-check that stops the assumption from going quietly stale.
//! - [`client`] — the live `tokio-tungstenite` WS client (the async edge).
//! - [`rest`] — `POST /info` snapshot seeding.
//!
//! Both kinds of frame arrive on one socket and are published onto one bus, because
//! a fill must be orderable against the book update that caused it.

pub mod client;
pub mod decode;
pub mod funding;
pub mod rest;
pub mod sub;
pub mod user;

pub use client::{HlError, HyperliquidMarketData};
pub use decode::{
    decode_meta, decode_rest_l2, decode_universe, decode_ws_message, venue_error, DecodeError,
    Universe,
};
pub use funding::{
    decode_funding_cadence, FundingCadence, ASSUMED_FUNDING_INTERVAL_NS, CADENCE_WINDOW_MS,
};
pub use rest::{
    fetch_funding_cadence, fetch_l2_snapshot, fetch_meta, fetch_universe, MAINNET_INFO,
    TESTNET_INFO,
};
pub use sub::{
    ping_msg, subscribe_msg, subscribe_user_msg, unsubscribe_msg, unsubscribe_user_msg, UserChannel,
};
pub use user::{decode_user_channel, fills_is_snapshot, hl_status, parse_cloid};
