//! # axon-execution
//!
//! The order manager (lifecycle + reconciliation) and, in Phase 3, the execution
//! engine that routes intents to the provider, exploits Hyperliquid's in-block
//! priority (cancel → post-only → GTC → IOC), and reconciles fills/updates back
//! onto the bus.
//!
//! This phase ships the [`OrderManager`] state machine keyed on `cloid`, which the
//! routing/reconciliation logic will drive, [`OrderTracker`] (venue truth, rebuilt
//! from the execution events on the bus), and the **submit pipeline** — three
//! [`ExecutionClient`](axon_providers::ExecutionClient) wrappers, each enforcing one
//! rule structurally rather than by convention, because a rule a caller is merely
//! expected to consult is one forgotten call site away from an unlimited position:
//!
//! ```text
//! HaltableClient   (halt)    — a dying or unprotected session adds no exposure
//!   └─ GuardedClient (guard) — pre-trade risk, checked before any budget is spent,
//!       │                      plus the de-risk-only mode a tripped loss limit imposes
//!       └─ GovernedClient (limiter) — the venue rate budget, cancels never refused
//!           └─ the venue client
//! ```
//!
//! The order is not arbitrary. Risk sits above rate because a risk check is a free
//! local computation while rate budget is a finite resource, and an order that risk
//! will refuse must not spend any of it. Halt sits above both because when the
//! dead-man's switch is failing there is nothing left worth checking.
//!
//! [`loss`] is the money side of that gate, and it is deliberately **not** a fourth
//! wrapper: a loss limit that refused every order would strand the exposure that caused
//! the loss, so it has to distinguish an order that adds from one that removes — which
//! means it needs a position, which means it belongs where the position already is.
//!
//! [`marks`] holds the price side of that gate: the [`MarkCache`] the risk context
//! reads, including the rule that expires a price whose feed has gone quiet.
//!
//! [`inflight`] is the other thing shared across the sync/async seam: the set of
//! symbols with an unfinished intent at the venue, which is what stops one target
//! becoming two orders while the tracker has not yet heard about the first
//! (ADR-0020 §3). It lives here beside [`HaltSwitch`] for the same reason that does —
//! both are lock-free state the deterministic core reads and the tokio edge writes.

#![deny(unsafe_code)]

pub mod guard;
pub mod halt;
pub mod inflight;
pub mod limiter;
pub mod loss;
pub mod marks;
pub mod tracker;

pub use guard::{GuardedClient, RiskContext, StaticRiskContext, TrackerRiskContext};
// Re-exported beside the gate that enforces them, because a caller wiring a portfolio
// bound into `GuardedClient::with_portfolio` should not have to know which crate the
// arithmetic lives in — the same reason `axon-strategy` re-exports `Precision`.
pub use axon_risk::{PortfolioEngine, PortfolioExposure, PortfolioLimits, PortfolioReject};
pub use halt::{HaltState, HaltSwitch, HaltableClient};
pub use inflight::InFlight;
pub use limiter::{GovernedClient, RateLimiter};
pub use loss::{LossBreach, LossLimiter, LossLimits, LossScope};
pub use marks::{MarkCache, MarkQuote, MarkSource};
pub use tracker::{Drift, OrderTracker, TrackedOrder};

use axon_core::Cloid;
use axon_providers::OrderStatus;
use std::collections::HashMap;

/// Tracks the venue-reported status of every live order by its `cloid`.
#[derive(Debug, Default)]
pub struct OrderManager {
    open: HashMap<Cloid, OrderStatus>,
}

impl OrderManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin tracking a freshly submitted order.
    pub fn track(&mut self, cloid: Cloid) {
        self.open.insert(cloid, OrderStatus::Accepted);
    }

    /// Apply a lifecycle update. Terminal states drop the order from the book.
    pub fn update(&mut self, cloid: Cloid, status: OrderStatus) {
        match status {
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected => {
                self.open.remove(&cloid);
            }
            live => {
                self.open.insert(cloid, live);
            }
        }
    }

    pub fn status(&self, cloid: Cloid) -> Option<OrderStatus> {
        self.open.get(&cloid).copied()
    }

    /// Count of orders still live (not in a terminal state).
    pub fn open_count(&self) -> usize {
        self.open.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_lifecycle_to_terminal() {
        let mut m = OrderManager::new();
        let c = Cloid::new(1);
        m.track(c);
        assert_eq!(m.status(c), Some(OrderStatus::Accepted));
        m.update(c, OrderStatus::Resting);
        assert_eq!(m.status(c), Some(OrderStatus::Resting));
        assert_eq!(m.open_count(), 1);
        m.update(c, OrderStatus::Filled);
        assert_eq!(m.status(c), None); // terminal → removed
        assert_eq!(m.open_count(), 0);
    }
}
