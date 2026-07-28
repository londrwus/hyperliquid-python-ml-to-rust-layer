//! Binance USD-M streaming ingest and the public REST reads.
//!
//! - [`stream`] — normalized [`Feed`](axon_providers::Feed) ↔ Binance stream name.
//!   Small, and load-bearing: the stream name is the only thing that says whether a
//!   `depthUpdate` is a snapshot or a diff.
//! - [`decode`] — pure JSON → normalized [`Event`](axon_core::Event) dispatcher,
//!   pinned to captured frames. The offline-testable heart of the adapter.
//! - [`rest`] — `exchangeInfo`, `fundingInfo`, `depth`, `openInterest`. All
//!   unauthenticated `GET`s, all bounded by an explicit timeout.
//! - [`client`] — the live `tokio-tungstenite` client (the async edge).
//!
//! There is no user-channel module. Binance's account stream needs a `listenKey`
//! obtained with an API key and kept alive every 30 minutes, and there is no key in
//! this repo — so the execution *reports* half of ADR-0010 has no adapter here at
//! all. That absence is deliberate and is recorded in ADR-0023 rather than left for a
//! reader to discover from a missing file.

pub mod client;
pub mod decode;
pub mod rest;
pub mod stream;

pub use client::{BinanceMarketData, BnError};
pub use decode::{decode_rest_depth, decode_ws_message, venue_error, DecodeError};
pub use rest::{
    fetch_depth_snapshot, fetch_exchange_info, fetch_open_interest, fetch_universe, MAINNET_REST,
    REST_TIMEOUT, TESTNET_REST,
};
pub use stream::{
    candle_interval, kline_interval, stream_kind, stream_kline_interval, stream_name,
    subscribe_msg, unsubscribe_msg, BookDepth, StreamKind, UpdateSpeed,
};
