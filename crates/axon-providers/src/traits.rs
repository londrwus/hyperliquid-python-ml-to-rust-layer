//! The provider trait triad. `async` sits only at these edges (venue I/O); the
//! deterministic core never awaits. Object-safe via `async-trait` so adapters can
//! be held as `Arc<dyn ExecutionClient>` behind a registry.

use async_trait::async_trait;
use axon_core::Position;

use crate::capabilities::Capabilities;
use crate::error::ProviderError;
use crate::market::Feed;
use crate::order::{CancelAck, CancelId, OrderAck, OrderRequest};

/// Submit and manage orders at a venue.
#[async_trait]
pub trait ExecutionClient: Send + Sync {
    /// What this adapter supports (so the router can reject impossible requests).
    fn capabilities(&self) -> &Capabilities;

    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError>;

    /// Batch submit (Hyperliquid caps at `capabilities().max_batch`).
    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError>;

    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError>;

    async fn cancel_all(&self) -> Result<(), ProviderError>;

    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError>;
}

/// Subscribe to normalized market data. Delivery of the resulting event stream
/// onto the core bus is wired at construction; the stream type is a Phase-2
/// decision (`OPEN` in docs/04) so it is intentionally not fixed here yet.
#[async_trait]
pub trait MarketData: Send + Sync {
    fn capabilities(&self) -> &Capabilities;
    async fn subscribe(&self, feed: Feed) -> Result<(), ProviderError>;
    async fn unsubscribe(&self, feed: Feed) -> Result<(), ProviderError>;
}

/// Query and stream account state (positions, orders, fills, funding).
#[async_trait]
pub trait AccountState: Send + Sync {
    async fn positions(&self) -> Result<Vec<Position>, ProviderError>;
    async fn open_orders(&self) -> Result<Vec<OrderAck>, ProviderError>;
    // balances() + stream_account_events() land with the execution path (Phase 3).
}
