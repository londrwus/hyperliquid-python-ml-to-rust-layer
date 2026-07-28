//! The session's "stop trading" switch, and the [`ExecutionClient`] that enforces it.
//!
//! Two situations demand that new exposure stop immediately while *removing* exposure
//! keeps working: the dead-man's switch failing to re-arm (the venue-side protection
//! is about to lapse), and shutdown (the sweep that follows is a read-then-write, so
//! any order admitted after the read survives it). Both need the same primitive, and
//! both need it to be structural — a flag every call site is expected to consult is
//! one forgotten call site away from placing an order into a session that is dying.
//!
//! Hence [`HaltableClient`]: it *is* an `ExecutionClient` and owns the one below it.
//! Cancels pass through in every state, for the same reason the risk gate never
//! blocks them — refusing a cancel cannot reduce risk, it can only strand it.
//!
//! [`HaltState::Stopped`] is deliberately one-way. A halt raised by the safety loop
//! can be cleared when protection comes back; a shutdown must not be, or a task that
//! resumes trading after the sweep has run would leave orders resting behind a process
//! that is already gone.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use axon_providers::{
    CancelAck, CancelId, Capabilities, ExecutionClient, OrderAck, OrderRequest, ProviderError,
};

/// Whether the session is accepting new exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaltState {
    /// Normal operation.
    Running,
    /// Recoverable: new orders refused, cancels allowed, may resume.
    Halted,
    /// Terminal: the session is shutting down and will not trade again.
    Stopped,
}

const RUNNING: u8 = 0;
const HALTED: u8 = 1;
const STOPPED: u8 = 2;

/// A shared, cheap-to-read trading switch. Clone the `Arc`, not the switch.
#[derive(Debug)]
pub struct HaltSwitch {
    state: AtomicU8,
}

impl Default for HaltSwitch {
    fn default() -> Self {
        Self::new()
    }
}

impl HaltSwitch {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(RUNNING),
        }
    }

    pub fn state(&self) -> HaltState {
        match self.state.load(Ordering::Acquire) {
            RUNNING => HaltState::Running,
            HALTED => HaltState::Halted,
            _ => HaltState::Stopped,
        }
    }

    /// Whether a new order would be admitted.
    pub fn is_accepting(&self) -> bool {
        self.state() == HaltState::Running
    }

    /// Refuse new orders until [`resume`](Self::resume). A no-op once stopped —
    /// escalation only ever moves one way.
    pub fn halt(&self) {
        let _ = self
            .state
            .compare_exchange(RUNNING, HALTED, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// Clear a recoverable halt. Cannot revive a stopped session.
    pub fn resume(&self) {
        let _ = self
            .state
            .compare_exchange(HALTED, RUNNING, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// Terminal stop. Nothing clears this.
    pub fn stop(&self) {
        self.state.store(STOPPED, Ordering::Release);
    }
}

/// Wraps an [`ExecutionClient`] so a halted session cannot place, and can always
/// cancel.
pub struct HaltableClient<C> {
    inner: C,
    switch: Arc<HaltSwitch>,
}

impl<C> HaltableClient<C> {
    pub fn new(inner: C, switch: Arc<HaltSwitch>) -> Self {
        Self { inner, switch }
    }

    pub fn switch(&self) -> &Arc<HaltSwitch> {
        &self.switch
    }

    /// A refusal is a `ProviderError::Rejected` so it travels the same path as a
    /// venue rejection and a risk refusal: to the caller, the order did not happen,
    /// and the reason string says which gate stopped it.
    fn admit(&self) -> Result<(), ProviderError> {
        match self.switch.state() {
            HaltState::Running => Ok(()),
            HaltState::Halted => Err(ProviderError::Rejected(
                "halted: the session is not accepting new orders (cancels still allowed)".into(),
            )),
            HaltState::Stopped => Err(ProviderError::Rejected(
                "stopped: the session is shutting down".into(),
            )),
        }
    }
}

#[async_trait]
impl<C: ExecutionClient> ExecutionClient for HaltableClient<C> {
    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.admit()?;
        self.inner.place_order(req).await
    }

    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
        self.admit()?;
        self.inner.place_batch(reqs).await
    }

    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
        self.inner.cancel(id).await
    }

    async fn cancel_all(&self) -> Result<(), ProviderError> {
        self.inner.cancel_all().await
    }

    /// Gated: a modify is a new order in all but name, and one placed into a session
    /// that is winding down is an order nobody will be left to manage.
    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.admit()?;
        self.inner.modify(id, req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Decimal, OrderId, Side, SymbolId, Tif};
    use axon_providers::{OrderStatus, RateLimitModel};
    use rust_decimal_macros::dec;
    use std::sync::atomic::AtomicUsize;

    const SYM: SymbolId = SymbolId::new(1);

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
            Ok(Vec::new())
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

    fn buy() -> OrderRequest {
        OrderRequest::limit(
            SYM,
            Side::Buy,
            Decimal::ONE,
            dec!(100),
            Tif::Gtc,
            Cloid::new(1),
        )
    }

    fn client() -> (HaltableClient<SpyClient>, Arc<HaltSwitch>) {
        let switch = Arc::new(HaltSwitch::new());
        (
            HaltableClient::new(SpyClient::new(), switch.clone()),
            switch,
        )
    }

    #[tokio::test]
    async fn a_halted_session_refuses_new_orders_but_still_cancels() {
        // The asymmetry is the whole point: whatever went wrong, we must still be able
        // to take exposure off.
        let (c, switch) = client();
        switch.halt();
        let err = c.place_order(buy()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("halted")));
        assert_eq!(c.inner.placed.load(Ordering::Relaxed), 0);

        assert!(c
            .cancel(CancelId::OrderId {
                symbol: SYM,
                order_id: OrderId::new(1),
            })
            .await
            .is_ok());
        assert!(c.cancel_all().await.is_ok());
        assert_eq!(c.inner.cancelled.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn a_recovered_session_trades_again() {
        let (c, switch) = client();
        switch.halt();
        assert!(c.place_order(buy()).await.is_err());
        switch.resume();
        assert!(c.place_order(buy()).await.is_ok());
        assert_eq!(c.inner.placed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_stopped_session_can_never_be_resumed() {
        // Shutdown runs a read-then-write sweep. A task that resumed trading behind it
        // would leave an order resting with no process left to manage it.
        let (c, switch) = client();
        switch.stop();
        switch.resume();
        assert_eq!(switch.state(), HaltState::Stopped);
        switch.halt();
        assert_eq!(switch.state(), HaltState::Stopped, "no downgrade either");
        let err = c.place_order(buy()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("stopped")));
        assert!(c.cancel_all().await.is_ok(), "the sweep must still work");
    }

    #[tokio::test]
    async fn a_running_session_is_transparent() {
        let (c, _switch) = client();
        assert!(c.place_order(buy()).await.is_ok());
        assert!(c.place_batch(vec![buy(), buy()]).await.is_ok());
        assert_eq!(c.inner.placed.load(Ordering::Relaxed), 3);
    }
}
