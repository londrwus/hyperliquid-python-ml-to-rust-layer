//! The rate budget on the submit path, expressed the same way the risk gate is: as
//! an [`ExecutionClient`] the venue client sits *inside*.
//!
//! Every venue publishes a request budget, and every venue's budget has the same
//! catastrophic corner — being throttled while holding orders you cannot cancel.
//! Hyperliquid deliberately makes that corner avoidable by giving cancels a strictly
//! larger allowance than placements (the arithmetic lives in
//! `axon-provider-hyperliquid`'s governor), and that guarantee only holds if the
//! client never spends cancel headroom on places. [`GovernedClient`] is where that
//! promise is kept on the call path:
//!
//! - **A placement may be refused.** It is charged first and sent only if admitted, so
//!   the local budget can never drift above the venue's.
//! - **A cancel is charged but never refused.** Refusing our own unwind is the exact
//!   state the separate allowance exists to prevent, so a cancel that does not fit the
//!   local estimate is still sent — the venue's larger ceiling is the real authority,
//!   and being wrong about a cancel costs a `429` while being wrong in the other
//!   direction strands live exposure.
//!
//! The budget itself is venue-specific, so it enters through the [`RateLimiter`] trait
//! and the concrete governor is wired in by the runtime. Note what the trait does
//! *not* take: a timestamp. A rate limit is a wall-clock property of the network edge,
//! not an event-time property of the core, so the implementation reads its own clock —
//! passing the core's event time in would silently mis-account a replay.

use async_trait::async_trait;
use axon_providers::{
    CancelAck, CancelId, Capabilities, ExecutionClient, OrderAck, OrderRequest, ProviderError,
};

/// A venue request budget, as the submit path sees it.
pub trait RateLimiter: Send + Sync {
    /// Charge for a placement of `orders` orders, or refuse it. The charge happens
    /// only on success, so a refusal costs nothing.
    fn admit_place(&self, orders: u32) -> Result<(), ProviderError>;

    /// Charge for a cancel of `orders` orders. Returns `false` when it did not fit the
    /// local budget — informational only: the caller sends it regardless.
    fn charge_cancel(&self, orders: u32) -> bool;
}

/// Wraps an [`ExecutionClient`] so every order-adding action is paced.
///
/// Stack it **inside** the risk gate ([`GuardedClient`](crate::GuardedClient)):
/// risk is a free local check and rate budget is a real, finite resource, so an order
/// that risk will refuse must not spend any of it first.
pub struct GovernedClient<C, L> {
    inner: C,
    limiter: L,
}

impl<C, L> GovernedClient<C, L> {
    pub fn new(inner: C, limiter: L) -> Self {
        Self { inner, limiter }
    }

    pub fn limiter(&self) -> &L {
        &self.limiter
    }
}

#[async_trait]
impl<C, L> ExecutionClient for GovernedClient<C, L>
where
    C: ExecutionClient,
    L: RateLimiter + 'static,
{
    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.limiter.admit_place(1)?;
        self.inner.place_order(req).await
    }

    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
        // Charged as `len` orders, not as one request: the address-side budget is
        // per order even though the per-IP weight is per request, and pricing a batch
        // of twenty as one action is how a client walks into a limit it thought it was
        // nowhere near.
        self.limiter.admit_place(reqs.len() as u32)?;
        self.inner.place_batch(reqs).await
    }

    /// Charged, never refused — see the module docs.
    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
        self.limiter.charge_cancel(1);
        self.inner.cancel(id).await
    }

    /// Charged as a single request even though the sweep sends many.
    ///
    /// `cancel_all` reads the open orders and batch-cancels them behind this call, so
    /// the true count is not visible here. Under-charging is acceptable precisely
    /// because it can only ever cause *cancels* to be over-admitted, and the periodic
    /// `userRateLimit` read the runtime performs overwrites the local estimate with
    /// the venue's own accounting, so the drift is bounded by one poll interval.
    async fn cancel_all(&self) -> Result<(), ProviderError> {
        self.limiter.charge_cancel(1);
        self.inner.cancel_all().await
    }

    /// A modify can raise size, so it is an order-adding action and is paced like one.
    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.limiter.admit_place(1)?;
        self.inner.modify(id, req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Decimal, OrderId, Side, SymbolId, Tif};
    use axon_providers::{OrderStatus, RateLimitModel};
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    const SYM: SymbolId = SymbolId::new(1);

    /// Counts what reached the venue, so a test can prove a refusal never hit the wire.
    #[derive(Default)]
    struct SpyClient {
        caps: Option<Capabilities>,
        placed: AtomicUsize,
        cancelled: AtomicUsize,
    }

    impl SpyClient {
        fn new() -> Self {
            Self {
                caps: Some(Capabilities {
                    venue: "spy",
                    order_types: &[axon_core::OrderType::Limit],
                    tifs: &[Tif::Gtc],
                    max_batch: 20,
                    native_market_orders: true,
                    reduce_only: true,
                    rate_limit_model: RateLimitModel::None,
                }),
                placed: AtomicUsize::new(0),
                cancelled: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl ExecutionClient for SpyClient {
        fn capabilities(&self) -> &Capabilities {
            self.caps.as_ref().unwrap()
        }
        async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
            self.placed.fetch_add(1, Ordering::Relaxed);
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: None,
                status: OrderStatus::Resting,
            })
        }
        async fn place_batch(
            &self,
            reqs: Vec<OrderRequest>,
        ) -> Result<Vec<OrderAck>, ProviderError> {
            self.placed.fetch_add(reqs.len(), Ordering::Relaxed);
            Ok(reqs
                .into_iter()
                .map(|r| OrderAck {
                    cloid: r.cloid,
                    order_id: None,
                    status: OrderStatus::Resting,
                })
                .collect())
        }
        async fn cancel(&self, _id: CancelId) -> Result<CancelAck, ProviderError> {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
            Ok(CancelAck {
                cloid: None,
                order_id: None,
            })
        }
        async fn cancel_all(&self) -> Result<(), ProviderError> {
            self.cancelled.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        async fn modify(
            &self,
            _id: CancelId,
            req: OrderRequest,
        ) -> Result<OrderAck, ProviderError> {
            self.placed.fetch_add(1, Ordering::Relaxed);
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: None,
                status: OrderStatus::Resting,
            })
        }
    }

    /// A budget with `place_credits` placements in it and an unlimited cancel side —
    /// the shape of every real venue's asymmetry, in miniature.
    #[derive(Default)]
    struct Budget {
        place_credits: AtomicU32,
        cancels_charged: AtomicU32,
        cancels_over_budget: AtomicU32,
    }

    impl RateLimiter for Budget {
        fn admit_place(&self, orders: u32) -> Result<(), ProviderError> {
            let left = self.place_credits.load(Ordering::Relaxed);
            if orders > left {
                return Err(ProviderError::RateLimited(format!(
                    "place budget exhausted: {orders} > {left}"
                )));
            }
            self.place_credits.fetch_sub(orders, Ordering::Relaxed);
            Ok(())
        }

        fn charge_cancel(&self, orders: u32) -> bool {
            self.cancels_charged.fetch_add(orders, Ordering::Relaxed);
            // Pretend the cancel ceiling is also spent, to prove the client sends anyway.
            self.cancels_over_budget.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    fn buy(qty: Decimal) -> OrderRequest {
        OrderRequest::limit(SYM, Side::Buy, qty, dec!(100), Tif::Gtc, Cloid::new(1))
    }

    fn governed(credits: u32) -> GovernedClient<SpyClient, Budget> {
        let budget = Budget::default();
        budget.place_credits.store(credits, Ordering::Relaxed);
        GovernedClient::new(SpyClient::new(), budget)
    }

    #[tokio::test]
    async fn a_rate_refused_order_never_reaches_the_venue() {
        let g = governed(0);
        let err = g.place_order(buy(dec!(1))).await.unwrap_err();
        assert!(matches!(err, ProviderError::RateLimited(_)));
        assert_eq!(g.inner.placed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_cancel_is_sent_even_with_no_budget_left() {
        // The one invariant that matters: being rate-limited must never mean being
        // unable to remove exposure.
        let g = governed(0);
        assert!(g
            .cancel(CancelId::OrderId {
                symbol: SYM,
                order_id: OrderId::new(7),
            })
            .await
            .is_ok());
        assert!(g.cancel_all().await.is_ok());
        assert_eq!(g.inner.cancelled.load(Ordering::Relaxed), 2);
        assert_eq!(
            g.limiter().cancels_over_budget.load(Ordering::Relaxed),
            2,
            "both were over budget and both went out anyway"
        );
    }

    #[tokio::test]
    async fn a_batch_is_charged_per_order_not_per_request() {
        // Twenty orders priced as one request is how a client believes it has budget
        // it does not have; the venue counts each order.
        let g = governed(5);
        let reqs: Vec<_> = (0..6).map(|_| buy(dec!(1))).collect();
        assert!(g.place_batch(reqs).await.is_err());
        assert_eq!(g.inner.placed.load(Ordering::Relaxed), 0);

        let reqs: Vec<_> = (0..5).map(|_| buy(dec!(1))).collect();
        assert!(g.place_batch(reqs).await.is_ok());
        assert_eq!(g.inner.placed.load(Ordering::Relaxed), 5);
        assert_eq!(g.limiter().place_credits.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn placements_stop_before_cancels_do() {
        // Structural restatement of the venue's own guarantee: exhausting the place
        // side leaves the unwind path fully open.
        let g = governed(1);
        assert!(g.place_order(buy(dec!(1))).await.is_ok());
        assert!(g.place_order(buy(dec!(1))).await.is_err());
        assert!(g.cancel_all().await.is_ok(), "unwind still possible");
    }

    #[tokio::test]
    async fn a_modify_is_paced_like_a_placement() {
        let g = governed(0);
        let id = CancelId::OrderId {
            symbol: SYM,
            order_id: OrderId::new(7),
        };
        assert!(g.modify(id, buy(dec!(1))).await.is_err());
        assert_eq!(g.inner.placed.load(Ordering::Relaxed), 0);
    }
}
