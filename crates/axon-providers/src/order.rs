//! The normalized order model (see `docs/04-provider-abstraction.md`). Each
//! adapter maps this to its venue; unsupported combinations become a typed
//! [`ProviderError::Unsupported`](crate::ProviderError) early.

use axon_core::{Cloid, Decimal, OrderId, OrderType, Side, SymbolId, Tif};
use serde::{Deserialize, Serialize};

// The execution vocabulary (order status, fills, lifecycle updates) is defined in
// `axon-core` and re-exported here, the same way `market` re-exports the market
// types — the normalized language is the core's, and adapters translate into it.
pub use axon_core::{
    AccountSnapshot, CancelReason, ExecEvent, Fill, Liquidity, OrderStatus, OrderUpdate,
};

/// A conditional trigger attached to an order (take-profit / stop-loss).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trigger {
    pub price: Decimal,
    pub kind: TriggerKind,
    /// Execute as a market order when triggered (vs. a resting limit).
    pub market: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerKind {
    TakeProfit,
    StopLoss,
}

/// A venue-agnostic order request. The `cloid` makes it idempotent + reconcilable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderRequest {
    pub symbol_id: SymbolId,
    pub side: Side,
    pub qty: Decimal,
    /// Limit price; `None` for market orders.
    pub price: Option<Decimal>,
    pub order_type: OrderType,
    pub tif: Tif,
    pub reduce_only: bool,
    pub trigger: Option<Trigger>,
    pub cloid: Cloid,
}

impl OrderRequest {
    /// A plain limit order (GTC, not reduce-only, no trigger).
    pub fn limit(
        symbol_id: SymbolId,
        side: Side,
        qty: Decimal,
        price: Decimal,
        tif: Tif,
        cloid: Cloid,
    ) -> Self {
        Self {
            symbol_id,
            side,
            qty,
            price: Some(price),
            order_type: OrderType::Limit,
            tif,
            reduce_only: false,
            trigger: None,
            cloid,
        }
    }
}

/// A venue-agnostic acknowledgement of a placed/modified order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderAck {
    pub cloid: Cloid,
    pub order_id: Option<OrderId>,
    pub status: OrderStatus,
}

/// Acknowledgement of a cancel. A cancel identifies its order by `cloid` *or*
/// venue `oid`, so each is optional — the ack echoes back whichever was known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelAck {
    pub cloid: Option<Cloid>,
    pub order_id: Option<OrderId>,
}

/// How to identify an order to cancel/modify: by our `cloid` or the venue `oid`,
/// each paired with the instrument — venues key cancels on `(asset, id)`, so the
/// symbol is required, not incidental.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelId {
    Cloid { symbol: SymbolId, cloid: Cloid },
    OrderId { symbol: SymbolId, order_id: OrderId },
}

impl CancelId {
    /// The instrument this cancel targets.
    pub fn symbol(&self) -> SymbolId {
        match self {
            CancelId::Cloid { symbol, .. } | CancelId::OrderId { symbol, .. } => *symbol,
        }
    }
}
