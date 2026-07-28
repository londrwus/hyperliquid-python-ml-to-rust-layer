//! The pre-trade risk gate — an [`ExecutionClient`] wrapper that no order can go
//! around.
//!
//! [`axon_risk::RiskEngine`] is a pure function of `(request, position, mark_px)`.
//! Making it *unbypassable* is a separate, structural job: if callers hold a bare
//! provider client and are merely expected to consult risk first, then one forgotten
//! call site is an unlimited position. So the gate is expressed as a type —
//! [`GuardedClient`] *is* an `ExecutionClient`, and the only handle the strategy side
//! ever receives. The venue client is moved inside it and is no longer reachable.
//!
//! Three deliberate asymmetries:
//!
//! - **Cancels are never gated.** Refusing a cancel cannot reduce risk, and a gate
//!   that can block a cancel converts a stale price feed into an un-exitable
//!   position. `cancel`/`cancel_all` pass straight through.
//! - **A batch is checked cumulatively, not per order.** Twenty orders that each sit
//!   under `max_position` can collectively blow through it. The gate projects each
//!   order onto a running position as if it fills in full, which is the only safe
//!   assumption for a resting order.
//! - **A missing mark price fails closed for anything that could add exposure.** An
//!   unpriced notional check is not a check. Reduce-only orders are exempt, because
//!   they never consult the notional cap and blocking them would prevent flattening.
//!   A price whose feed has gone quiet counts as missing here — [`crate::marks`]
//!   expires it rather than handing the gate a number the market has moved away from.
//! - **A portfolio bound is measured across every instrument at once, and it can refuse
//!   an order that every per-symbol limit permits.** Ten instruments each at 90 % of
//!   their own `max_notional` are inside every limit the session declares. See
//!   [`axon_risk::portfolio`]; the piece that lives here is
//!   [`RiskContext::portfolio`], and its `None` is the fail-closed case — a context that
//!   cannot enumerate the book is not a context reporting an empty one.
//!
//! The gate projects only the orders in the call it is given; everything already
//! resting at the venue comes from whatever [`RiskContext`] reports. That makes the
//! choice of context the difference between a gate that works and one that only looks
//! like it does:
//!
//! - [`StaticRiskContext`] reports filled position only, so a *sequence* of separate
//!   `place_order` calls can exceed `max_position` in aggregate while each call
//!   passes. It is for offline tests and paths with no live order state.
//! - [`TrackerRiskContext`] reports [`OrderTracker::risk_position`] — filled position
//!   plus the worst case that every live order fills — which closes that gap. Use it
//!   anywhere real orders are being sent.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use axon_core::{Decimal, Position, SymbolId};
use axon_providers::{
    CancelAck, CancelId, Capabilities, ExecutionClient, OrderAck, OrderRequest, ProviderError,
};
use axon_risk::{PortfolioEngine, RiskEngine, RiskReject};

use crate::loss::LossLimiter;
use crate::marks::MarkCache;
use crate::tracker::OrderTracker;

/// The world-state a pre-trade check needs, supplied by the caller.
///
/// Kept as a trait so the gate stays testable offline and so the order tracker can
/// later provide a richer view (filled position *plus* resting-order exposure)
/// without changing the gate.
pub trait RiskContext: Send + Sync {
    /// Current position for `symbol`. Flat if unknown — an unknown symbol has no
    /// exposure yet, so a flat position is the honest answer, not a guess.
    ///
    /// The production implementation reports the **projected** position: filled plus
    /// the worst case that every resting order fills. That is what a size cap has to be
    /// measured against, and it is not what every question wants — see
    /// [`Self::held_position`].
    fn position(&self, symbol: SymbolId) -> Position;

    /// What we actually **hold** in `symbol`, with nothing projected onto it.
    ///
    /// Defaults to [`Self::position`], which is correct for every context whose two
    /// answers coincide, and is overridden by [`TrackerRiskContext`] where they do not.
    ///
    /// The distinction exists because a size cap and a de-risk check want opposite
    /// conservatism. `max_position` must count a resting order as if it filled, or
    /// twenty of them slip past one at a time. The de-risk mode a tripped
    /// [`LossLimiter`] imposes must **not**: flat with a resting buy projects long, so a
    /// sell would read as a reduction while in fact opening a short — the switch would
    /// admit new exposure on the session it was tripped to stop.
    fn held_position(&self, symbol: SymbolId) -> Position {
        self.position(symbol)
    }

    /// Latest mark price for `symbol`, or `None` when there is no usable price.
    fn mark_px(&self, symbol: SymbolId) -> Option<Decimal>;

    /// The **whole** book — every instrument carrying exposure, projected and priced —
    /// or `None` when this context cannot enumerate one.
    ///
    /// `None` is not an empty book, and conflating the two is the failure this signature
    /// exists to prevent. An empty [`PortfolioExposure`] says "nothing is held", against
    /// which every portfolio bound passes; `None` says "I cannot see what is held", and
    /// [`GuardedClient`] turns that into a refusal for anything that could add exposure.
    /// A default of `Some(empty)` would make every context that never implemented this
    /// silently satisfy a declared portfolio limit.
    ///
    /// Defaulted to `None` rather than required so that adding a portfolio bound did not
    /// force every existing context — including the harness ones outside this crate — to
    /// answer a question they have no way to answer.
    fn portfolio(&self) -> Option<axon_risk::PortfolioExposure> {
        None
    }
}

/// A trivial [`RiskContext`] over in-memory maps — for tests and for the offline
/// paths that have no live market data yet.
#[derive(Debug, Default, Clone)]
pub struct StaticRiskContext {
    positions: HashMap<SymbolId, Position>,
    marks: HashMap<SymbolId, Decimal>,
}

impl StaticRiskContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_position(mut self, position: Position) -> Self {
        self.positions.insert(position.symbol_id, position);
        self
    }

    pub fn with_mark(mut self, symbol: SymbolId, px: Decimal) -> Self {
        self.marks.insert(symbol, px);
        self
    }

    pub fn set_position(&mut self, position: Position) {
        self.positions.insert(position.symbol_id, position);
    }

    pub fn set_mark(&mut self, symbol: SymbolId, px: Decimal) {
        self.marks.insert(symbol, px);
    }
}

impl RiskContext for StaticRiskContext {
    fn position(&self, symbol: SymbolId) -> Position {
        self.positions
            .get(&symbol)
            .cloned()
            .unwrap_or_else(|| Position::flat(symbol))
    }

    fn mark_px(&self, symbol: SymbolId) -> Option<Decimal> {
        self.marks.get(&symbol).copied()
    }

    /// Every position this context was told about, in symbol order.
    ///
    /// Sorted rather than in `HashMap` order because [`axon_risk::PortfolioReject::Unpriced`]
    /// names the *first* leg with no mark, and an error message that changes between two
    /// runs of the same test is an error message nobody trusts.
    fn portfolio(&self) -> Option<axon_risk::PortfolioExposure> {
        let mut symbols: Vec<SymbolId> = self.positions.keys().copied().collect();
        symbols.sort();
        let mut book = axon_risk::PortfolioExposure::with_capacity(symbols.len());
        for s in symbols {
            book.set(s, self.position(s).qty, self.mark_px(s));
        }
        Some(book)
    }
}

/// The production [`RiskContext`]: exposure from the [`OrderTracker`], prices from a
/// [`MarkCache`].
///
/// Reports [`OrderTracker::risk_position`], not the filled position — so twenty
/// resting orders are counted before the twenty-first is allowed. This is the context
/// that makes the gate whole.
///
/// One consequence worth knowing rather than rediscovering: because the reported
/// position is inflated by resting orders, a **reduce-only** order is measured against
/// that inflated figure, so the gate will admit a reduce-only order larger than the
/// position actually held. That is deliberate and safe — a reduce-only order cannot add
/// exposure at the venue, which caps it at the real position — and the alternative
/// (measuring reduce-only against filled position while measuring everything else
/// against worst-case) would need two position views and could block a legitimate
/// unwind. Erring toward letting de-risking orders through is the correct direction.
///
/// A poisoned lock **fails closed**: the context reports a flat position and no mark,
/// which the gate turns into a refusal for anything that adds exposure. Trading on an
/// unknown position is strictly worse than not trading.
#[derive(Clone)]
pub struct TrackerRiskContext {
    tracker: Arc<RwLock<OrderTracker>>,
    marks: Arc<MarkCache>,
}

impl TrackerRiskContext {
    pub fn new(tracker: Arc<RwLock<OrderTracker>>, marks: Arc<MarkCache>) -> Self {
        Self { tracker, marks }
    }

    pub fn tracker(&self) -> &Arc<RwLock<OrderTracker>> {
        &self.tracker
    }

    pub fn marks(&self) -> &Arc<MarkCache> {
        &self.marks
    }
}

impl RiskContext for TrackerRiskContext {
    fn position(&self, symbol: SymbolId) -> Position {
        match self.tracker.read() {
            Ok(t) => t.risk_position(symbol),
            // Fail closed: an unknown position must not read as "flat and safe" for a
            // risk-increasing order. Reporting flat here is safe only because the
            // missing mark below forces a refusal anyway.
            Err(_) => Position::flat(symbol),
        }
    }

    /// Filled position only — see [`RiskContext::held_position`] for why the two differ
    /// and why a de-risk check must have this one.
    ///
    /// A poisoned lock reports flat, and flat is the **strictest** answer this can give:
    /// nothing reduces a flat position, so a tripped loss limit refuses everything
    /// rather than admitting an order it cannot show is de-risking. That is the opposite
    /// direction from the fail-closed above and it is the same principle — when our own
    /// state is unknown, do not act on a guess.
    fn held_position(&self, symbol: SymbolId) -> Position {
        match self.tracker.read() {
            Ok(t) => t.position(symbol),
            Err(_) => Position::flat(symbol),
        }
    }

    fn mark_px(&self, symbol: SymbolId) -> Option<Decimal> {
        if self.tracker.read().is_err() {
            return None; // poisoned tracker → refuse rather than guess
        }
        // `MarkCache::get` withholds an expired price, so a dead feed reaches the gate
        // as a *missing* mark and fails closed. That collapse is intentional: a price
        // the feed stopped confirming is worse than no price, because it still passes
        // a notional check (see [`crate::marks`]).
        self.marks.get(symbol)
    }

    /// The projected book: every instrument the tracker knows exposure in, valued at the
    /// mark cache's current price.
    ///
    /// **Projected**, matching [`Self::position`], because a gross notional cap is a size
    /// cap and a size cap has to count a resting order as if it fills — twenty of them
    /// slip past one at a time otherwise. The de-risk exemption asks a different question
    /// and gets [`Self::held_position`] instead; `axon_risk::PortfolioEngine::check` takes
    /// both for exactly that reason.
    ///
    /// A poisoned lock is `None` and not an empty book — trading against a portfolio we
    /// cannot read is the thing this whole return type exists to refuse.
    fn portfolio(&self) -> Option<axon_risk::PortfolioExposure> {
        let t = self.tracker.read().ok()?;
        let symbols = t.exposed_symbols();
        let mut book = axon_risk::PortfolioExposure::with_capacity(symbols.len());
        for s in symbols {
            book.set(s, t.risk_position(s).qty, self.marks.get(s));
        }
        Some(book)
    }
}

/// Wraps an [`ExecutionClient`] so every risk-increasing action clears
/// [`RiskEngine`] first.
pub struct GuardedClient<C, X> {
    inner: C,
    risk: RiskEngine,
    ctx: X,
    /// The money bound. Always present — an unconfigured session gets
    /// [`LossLimiter::undeclared`], so the wiring a live session runs is the wiring
    /// every test runs and there is no `if let` between an order and the switch.
    loss: Arc<LossLimiter>,
    /// The across-symbol bound, present on the same terms and for the same reason: the
    /// default declares nothing and refuses nothing.
    ///
    /// It sits **here** rather than only in the intent source, and the duplication is
    /// deliberate. The intent source *scales* targets so a session with a portfolio bound
    /// converges to a position it is allowed to hold; this refuses an order that would
    /// breach it. Only one of those is a guarantee — a scaler is arithmetic a call site
    /// can get wrong or bypass, and every previous gate in this pipeline was made
    /// structural for the same reason.
    portfolio: PortfolioEngine,
}

impl<C, X> GuardedClient<C, X> {
    pub fn new(inner: C, risk: RiskEngine, ctx: X) -> Self {
        Self {
            inner,
            risk,
            ctx,
            loss: Arc::new(LossLimiter::undeclared()),
            portfolio: PortfolioEngine::default(),
        }
    }

    /// Put an across-symbol bound in force.
    ///
    /// A builder for the same reason [`Self::with_loss_limiter`] is: every existing
    /// caller wants the undeclared default, and a fifth positional argument is how the
    /// wrong value lands in the wrong slot.
    pub fn with_portfolio(mut self, portfolio: PortfolioEngine) -> Self {
        self.portfolio = portfolio;
        self
    }

    pub fn portfolio(&self) -> &PortfolioEngine {
        &self.portfolio
    }

    /// Put a loss limit in force.
    ///
    /// A builder rather than a fourth constructor argument because every existing
    /// caller — including every offline path — wants the undeclared default, and a
    /// fourth positional argument is how the wrong `Arc` ends up in the slot.
    pub fn with_loss_limiter(mut self, loss: Arc<LossLimiter>) -> Self {
        self.loss = loss;
        self
    }

    pub fn loss_limiter(&self) -> &Arc<LossLimiter> {
        &self.loss
    }

    /// The risk engine in force. Read-only on purpose: limits are configuration,
    /// not something a strategy may widen at runtime.
    pub fn risk(&self) -> &RiskEngine {
        &self.risk
    }

    /// The risk context, for callers that keep it up to date out-of-band.
    pub fn context(&self) -> &X {
        &self.ctx
    }

    pub fn context_mut(&mut self) -> &mut X {
        &mut self.ctx
    }
}

/// A rejection is a `ProviderError::Rejected` so it travels the same path as a
/// venue rejection — callers already handle that, and a locally-refused order is
/// indistinguishable in consequence from one the venue refused.
fn reject(err: RiskReject) -> ProviderError {
    ProviderError::Rejected(format!("risk: {err}"))
}

impl<C, X> GuardedClient<C, X>
where
    C: ExecutionClient,
    X: RiskContext,
{
    /// Resolve the mark price to check `req` against, failing closed when a
    /// risk-increasing order has no price to be measured in.
    fn mark_for(&self, req: &OrderRequest) -> Result<Decimal, ProviderError> {
        match self.ctx.mark_px(req.symbol_id) {
            Some(px) => Ok(px),
            // Reduce-only returns before the notional cap is consulted, so the value
            // is unused; zero keeps the de-risking path open.
            None if req.reduce_only => Ok(Decimal::ZERO),
            None => Err(ProviderError::Rejected(format!(
                "risk: no mark price for {} - refusing to check exposure blind",
                req.symbol_id
            ))),
        }
    }

    /// The money gate: while a declared loss bound is tripped, only orders that take
    /// exposure **off** get out.
    ///
    /// Checked before the size caps, and the ordering says something: a session in
    /// de-risk-only mode has already failed the question the size caps are asking, so
    /// the reason on the refusal should be the one the operator has to act on. It reads
    /// against [`RiskContext::held_position`] rather than the projected one — see that
    /// method for why a projected position would let this admit new exposure.
    ///
    /// A `reduce_only` order is *not* waved through on the flag alone. The flag is a
    /// request to the venue; [`axon_risk::reduces_exposure`] is arithmetic about the
    /// position we hold, and the two disagree exactly when it matters — a reduce-only
    /// order against a flat book reduces nothing and the venue would simply drop it,
    /// while a session that let it through would be a session whose kill switch could
    /// be talked past by setting a bit.
    fn check_loss(&self, req: &OrderRequest, held: Decimal) -> Result<(), ProviderError> {
        if !self.loss.is_tripped() {
            return Ok(());
        }
        if axon_risk::reduces_exposure(req.side, req.qty, held) {
            return Ok(());
        }
        Err(ProviderError::Rejected(format!(
            "loss limit: {} - de-risk only, and this order does not reduce {} \
             (held {held})",
            self.loss
                .breach()
                .map(|b| b.to_string())
                .unwrap_or_else(|| "tripped".into()),
            req.symbol_id
        )))
    }

    /// The across-symbol gate: what the *book* is allowed to be, rather than what one
    /// instrument is.
    ///
    /// `book` is `None` when the caller could not read one. That is refused rather than
    /// treated as an empty portfolio — but only for an order that could add exposure,
    /// which is why the de-risk question is asked here too and not left to the engine: a
    /// context that cannot enumerate the book can still say what one symbol holds, and an
    /// exit must not be blocked by the bookkeeping.
    fn check_portfolio(
        &self,
        req: &OrderRequest,
        book: Option<&axon_risk::PortfolioExposure>,
        held: Decimal,
    ) -> Result<(), ProviderError> {
        if !self.portfolio.is_declared() {
            return Ok(());
        }
        let Some(book) = book else {
            if axon_risk::reduces_exposure(req.side, req.qty, held) {
                return Ok(());
            }
            return Err(ProviderError::Rejected(format!(
                "portfolio: {}",
                axon_risk::PortfolioReject::Unreadable
            )));
        };
        self.portfolio
            .check(req, book, held, self.ctx.mark_px(req.symbol_id))
            .map_err(|e| ProviderError::Rejected(format!("portfolio: {e}")))
    }

    /// Check one order against the position the caller believes it holds.
    ///
    /// Three views of the world, because the three gates ask different questions:
    /// `position` is the projected single-symbol figure a size cap needs, `held` is the
    /// filled one a de-risk check needs (see [`RiskContext::held_position`]), and `book`
    /// is every instrument at once, which is the only thing a portfolio bound can be
    /// measured against.
    ///
    /// The order of the three is the order an operator wants the reason in: money first
    /// (a tripped session has already failed the question the others ask), then the book,
    /// then this one instrument.
    fn check_one(
        &self,
        req: &OrderRequest,
        position: &Position,
        held: Decimal,
        book: Option<&axon_risk::PortfolioExposure>,
    ) -> Result<(), ProviderError> {
        self.check_loss(req, held)?;
        self.check_portfolio(req, book, held)?;
        let mark = self.mark_for(req)?;
        self.risk.check(req, position, mark).map_err(reject)
    }

    /// Check a whole batch, carrying each order's assumed full fill forward so the
    /// batch is measured as a unit.
    fn check_batch(&self, reqs: &[OrderRequest]) -> Result<(), ProviderError> {
        let max_batch = self.inner.capabilities().max_batch as usize;
        if reqs.len() > max_batch {
            return Err(ProviderError::Rejected(format!(
                "risk: batch of {} exceeds venue max_batch {max_batch}",
                reqs.len()
            )));
        }

        let mut projected: HashMap<SymbolId, Position> = HashMap::new();
        // The held position is carried forward too, and for the same reason the
        // projected one is: a batch of two sells against a long of one would otherwise
        // see each of them reduce the *same* long, and the pair would flip the position
        // to a short while a tripped loss limit reported that both were de-risking.
        let mut held: HashMap<SymbolId, Decimal> = HashMap::new();
        // And so is the book, which is the case the portfolio bound was written for: a
        // batch that opens five instruments at 30 % of the gross cap each is five orders
        // that individually fit and together do not. Read once, before the loop, because
        // a re-read per order would take the tracker's lock five times and could see a
        // fill land between two orders of one batch.
        let mut book = self.ctx.portfolio();
        for req in reqs {
            let position = projected
                .entry(req.symbol_id)
                .or_insert_with(|| self.ctx.position(req.symbol_id));
            let held_qty = held
                .entry(req.symbol_id)
                .or_insert_with(|| self.ctx.held_position(req.symbol_id).qty);
            self.check_one(req, position, *held_qty, book.as_ref())?;
            if let Some(b) = book.as_mut() {
                b.apply(
                    req.symbol_id,
                    req.side,
                    req.qty,
                    self.ctx.mark_px(req.symbol_id),
                );
            }
            // Assume it fills in full: a resting order is exposure the moment it is
            // accepted, so the next order in the batch must be measured against it.
            let fill_px = req
                .price
                .unwrap_or_else(|| self.ctx.mark_px(req.symbol_id).unwrap_or(position.avg_px));
            position.apply_fill(req.side, req.qty, fill_px);
            *held_qty += match req.side {
                axon_core::Side::Buy => req.qty,
                axon_core::Side::Sell => -req.qty,
            };
        }
        Ok(())
    }
}

#[async_trait]
impl<C, X> ExecutionClient for GuardedClient<C, X>
where
    C: ExecutionClient,
    X: RiskContext + 'static,
{
    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.check_one(
            &req,
            &self.ctx.position(req.symbol_id),
            self.ctx.held_position(req.symbol_id).qty,
            self.ctx.portfolio().as_ref(),
        )?;
        self.inner.place_order(req).await
    }

    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
        self.check_batch(&reqs)?;
        self.inner.place_batch(reqs).await
    }

    /// Ungated: see the module docs. A refused cancel can only ever raise risk.
    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
        self.inner.cancel(id).await
    }

    /// Ungated for the same reason as [`cancel`](Self::cancel).
    async fn cancel_all(&self) -> Result<(), ProviderError> {
        self.inner.cancel_all().await
    }

    /// Gated: a modify can raise qty, so it is a risk-increasing action and is
    /// checked exactly like a fresh order.
    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.check_one(
            &req,
            &self.ctx.position(req.symbol_id),
            self.ctx.held_position(req.symbol_id).qty,
            self.ctx.portfolio().as_ref(),
        )?;
        self.inner.modify(id, req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Side, Tif};
    use axon_providers::{OrderStatus, RateLimitModel};
    use axon_risk::RiskLimits;
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const SYM: SymbolId = SymbolId::new(1);

    /// Counts what actually reached the venue, so a test can prove a refusal never
    /// hit the wire (the whole point of the gate).
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
                    order_types: &[axon_core::OrderType::Limit, axon_core::OrderType::Market],
                    tifs: &[Tif::Gtc, Tif::Ioc, Tif::PostOnly],
                    max_batch: 3,
                    native_market_orders: true,
                    reduce_only: true,
                    rate_limit_model: RateLimitModel::None,
                }),
                placed: AtomicUsize::new(0),
                cancelled: AtomicUsize::new(0),
            }
        }
        fn placed(&self) -> usize {
            self.placed.load(Ordering::Relaxed)
        }
        fn cancelled(&self) -> usize {
            self.cancelled.load(Ordering::Relaxed)
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

    fn limits() -> RiskLimits {
        RiskLimits {
            max_position: dec!(10),
            max_notional: dec!(1_000_000),
            max_order_qty: dec!(5),
        }
    }

    fn buy(qty: Decimal, cloid: u128) -> OrderRequest {
        OrderRequest::limit(SYM, Side::Buy, qty, dec!(100), Tif::Gtc, Cloid::new(cloid))
    }

    fn guarded(ctx: StaticRiskContext) -> GuardedClient<SpyClient, StaticRiskContext> {
        GuardedClient::new(SpyClient::new(), RiskEngine::new(limits()), ctx)
    }

    // ── the across-symbol gate ───────────────────────────────────────────────

    const ETH: SymbolId = SymbolId::new(2);
    const SOL: SymbolId = SymbolId::new(3);

    fn order(symbol: SymbolId, side: Side, qty: Decimal, cloid: u128) -> OrderRequest {
        OrderRequest::limit(symbol, side, qty, dec!(100), Tif::Gtc, Cloid::new(cloid))
    }

    fn with_portfolio(
        ctx: StaticRiskContext,
        p: axon_risk::PortfolioLimits,
    ) -> GuardedClient<SpyClient, StaticRiskContext> {
        guarded(ctx).with_portfolio(PortfolioEngine::new(p))
    }

    fn positioned(symbol: SymbolId, qty: Decimal, mark: Decimal) -> Position {
        let mut p = Position::flat(symbol);
        p.qty = qty;
        p.avg_px = mark;
        p
    }

    #[tokio::test]
    async fn a_book_inside_every_per_symbol_limit_can_still_be_refused_by_the_portfolio() {
        // The whole reason this gate exists. Each instrument sits well inside
        // `max_position` (10) and `max_notional` (1 000 000); together they are 900 of
        // notional against a 1 000 cap, and the tenth order takes them past it. No
        // per-symbol limit can see that, which is why it is a separate object rather
        // than a tighter number in the existing one.
        let ctx = StaticRiskContext::new()
            .with_position(positioned(SYM, dec!(5), dec!(100)))
            .with_mark(SYM, dec!(100))
            .with_position(positioned(ETH, dec!(4), dec!(100)))
            .with_mark(ETH, dec!(100))
            .with_mark(SOL, dec!(100));
        let g = with_portfolio(
            ctx,
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(1000),
                ..Default::default()
            },
        );
        let err = g
            .place_order(order(SOL, Side::Buy, dec!(2), 1))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ProviderError::Rejected(ref m) if m.contains("portfolio: projected gross")),
            "{err:?}"
        );
        assert_eq!(g.inner.placed(), 0);

        // …and the order that fits goes out, so this is a bound rather than a wall.
        assert!(g
            .place_order(order(SOL, Side::Buy, dec!(1), 2))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn a_portfolio_bound_never_refuses_the_order_that_gets_us_out() {
        // The rule the module is shaped around, checked where it is enforced rather than
        // only where it is computed: the book is already three times its cap, and the
        // exit still reaches the venue. A gate that refused here would pin the account
        // into exactly the position somebody is trying to leave.
        let ctx = StaticRiskContext::new()
            .with_position(positioned(SYM, dec!(10), dec!(100)))
            .with_mark(SYM, dec!(100))
            .with_position(positioned(ETH, dec!(10), dec!(100)))
            .with_mark(ETH, dec!(100));
        let g = with_portfolio(
            ctx,
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(600),
                ..Default::default()
            },
        );
        assert!(g
            .place_order(order(SYM, Side::Sell, dec!(4), 1))
            .await
            .is_ok());
        assert_eq!(g.inner.placed(), 1);

        let err = g
            .place_order(order(SYM, Side::Buy, dec!(1), 2))
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("portfolio")));
    }

    #[tokio::test]
    async fn a_batch_is_measured_against_the_book_it_is_building() {
        // Three orders that each fit against the *starting* book and do not fit against
        // each other — the multi-instrument version of `batch_is_checked_cumulatively`,
        // and the case a portfolio bound read once and never advanced would miss.
        let ctx = StaticRiskContext::new()
            .with_mark(SYM, dec!(100))
            .with_mark(ETH, dec!(100))
            .with_mark(SOL, dec!(100));
        let g = with_portfolio(
            ctx.clone(),
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(500),
                ..Default::default()
            },
        );
        let err = g
            .place_batch(vec![
                order(SYM, Side::Buy, dec!(2), 1),
                order(ETH, Side::Buy, dec!(2), 2),
                order(SOL, Side::Buy, dec!(2), 3),
            ])
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("portfolio")));
        assert_eq!(g.inner.placed(), 0, "the whole batch must be refused");

        // Two of the three fit, so the refusal above is arithmetic and not a blanket ban.
        let g = with_portfolio(
            ctx,
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(500),
                ..Default::default()
            },
        );
        assert!(g
            .place_batch(vec![
                order(SYM, Side::Buy, dec!(2), 1),
                order(ETH, Side::Buy, dec!(2), 2),
            ])
            .await
            .is_ok());
        assert_eq!(g.inner.placed(), 2);
    }

    #[tokio::test]
    async fn a_context_that_cannot_read_the_book_refuses_new_exposure_and_still_lets_an_exit_out() {
        // `None` is not an empty book. A context that cannot enumerate the portfolio
        // would otherwise satisfy every declared bound by saying nothing, which is the
        // most comfortable possible reading of a session that has lost sight of its own
        // positions.
        struct Blind(StaticRiskContext);
        impl RiskContext for Blind {
            fn position(&self, s: SymbolId) -> Position {
                self.0.position(s)
            }
            fn mark_px(&self, s: SymbolId) -> Option<Decimal> {
                self.0.mark_px(s)
            }
            // …and deliberately no `portfolio` override: the trait default.
        }

        let inner = StaticRiskContext::new()
            .with_position(positioned(SYM, dec!(5), dec!(100)))
            .with_mark(SYM, dec!(100));
        let g = GuardedClient::new(SpyClient::new(), RiskEngine::new(limits()), Blind(inner))
            .with_portfolio(PortfolioEngine::new(axon_risk::PortfolioLimits {
                max_gross_notional: dec!(100_000),
                ..Default::default()
            }));
        let err = g
            .place_order(order(SYM, Side::Buy, dec!(1), 1))
            .await
            .unwrap_err();
        assert!(
            matches!(err, ProviderError::Rejected(ref m) if m.contains("no portfolio exposure")),
            "{err:?}"
        );
        assert_eq!(g.inner.placed(), 0);

        // The exit is still asked of `held_position`, which a blind context can answer,
        // so a bookkeeping failure never blocks a reduction.
        assert!(g
            .place_order(order(SYM, Side::Sell, dec!(2), 2))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn an_undeclared_portfolio_changes_nothing_about_an_existing_session() {
        // Every session written before this gate existed gets `PortfolioEngine::default`,
        // and it must be indistinguishable from not having one — upgrading a binary is
        // not allowed to stop a session.
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        assert!(!g.portfolio().is_declared());
        assert!(g.place_order(buy(dec!(3), 1)).await.is_ok());
        assert_eq!(g.inner.placed(), 1);
    }

    #[tokio::test]
    async fn passes_a_compliant_order_through() {
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        assert!(g.place_order(buy(dec!(3), 1)).await.is_ok());
        assert_eq!(g.inner.placed(), 1);
    }

    #[tokio::test]
    async fn refusal_never_reaches_the_venue() {
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        let err = g.place_order(buy(dec!(6), 1)).await.unwrap_err(); // > max_order_qty
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("risk:")));
        assert_eq!(g.inner.placed(), 0, "a refused order must not be sent");
    }

    #[tokio::test]
    async fn batch_is_checked_cumulatively_not_per_order() {
        // Three orders of 4 are each under max_order_qty (5) and each under
        // max_position (10) in isolation, but together they reach 12.
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        let err = g
            .place_batch(vec![buy(dec!(4), 1), buy(dec!(4), 2), buy(dec!(4), 3)])
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("projected position")));
        assert_eq!(g.inner.placed(), 0, "the whole batch must be refused");

        // Two of them (8) still fit.
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        assert!(g
            .place_batch(vec![buy(dec!(4), 1), buy(dec!(4), 2)])
            .await
            .is_ok());
        assert_eq!(g.inner.placed(), 2);
    }

    #[tokio::test]
    async fn batch_larger_than_venue_max_is_refused() {
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        let reqs: Vec<_> = (0..4).map(|i| buy(dec!(1), i)).collect(); // max_batch = 3
        let err = g.place_batch(reqs).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("max_batch")));
        assert_eq!(g.inner.placed(), 0);
    }

    #[tokio::test]
    async fn missing_mark_price_fails_closed_but_lets_reduce_only_out() {
        let g = guarded(StaticRiskContext::new()); // no mark at all
        let err = g.place_order(buy(dec!(1), 1)).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("no mark price")));
        assert_eq!(g.inner.placed(), 0);

        // Flattening a long must still be possible with no price feed.
        let mut long = Position::flat(SYM);
        long.qty = dec!(3);
        let g = guarded(StaticRiskContext::new().with_position(long));
        let mut flatten = buy(dec!(3), 2);
        flatten.side = Side::Sell;
        flatten.reduce_only = true;
        assert!(g.place_order(flatten).await.is_ok());
        assert_eq!(g.inner.placed(), 1);
    }

    #[tokio::test]
    async fn cancels_are_never_gated() {
        // An empty context (no marks, no positions) refuses every placement, yet
        // cancels must still get through.
        let g = guarded(StaticRiskContext::new());
        assert!(g
            .cancel(CancelId::OrderId {
                symbol: SYM,
                order_id: axon_core::OrderId::new(7),
            })
            .await
            .is_ok());
        assert!(g.cancel_all().await.is_ok());
        assert_eq!(g.inner.cancelled(), 2);
    }

    #[tokio::test]
    async fn modify_is_gated_like_a_placement() {
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        let id = CancelId::OrderId {
            symbol: SYM,
            order_id: axon_core::OrderId::new(7),
        };
        let err = g.modify(id, buy(dec!(6), 1)).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(_)));
        assert_eq!(g.inner.placed(), 0);
        assert!(g.modify(id, buy(dec!(2), 2)).await.is_ok());
    }

    #[tokio::test]
    async fn existing_position_is_counted() {
        let mut long = Position::flat(SYM);
        long.qty = dec!(8);
        let g = guarded(
            StaticRiskContext::new()
                .with_position(long)
                .with_mark(SYM, dec!(100)),
        );
        // 8 + 5 = 13 > max_position 10
        assert!(g.place_order(buy(dec!(5), 1)).await.is_err());
        // 8 + 2 = 10 is exactly at the cap
        assert!(g.place_order(buy(dec!(2), 2)).await.is_ok());
    }

    #[tokio::test]
    async fn tracker_context_closes_the_sequential_orders_gap() {
        // The gap: three separate place_order calls of 4 each pass individually
        // against filled position (still flat), and together rest 12 against a
        // max_position of 10. The tracker-backed context counts resting exposure, so
        // the third is refused.
        let tracker = Arc::new(RwLock::new(OrderTracker::new()));
        let marks = Arc::new(MarkCache::new());
        marks.set_mark(SYM, dec!(100), 1);
        let ctx = TrackerRiskContext::new(tracker.clone(), marks);
        let g = GuardedClient::new(SpyClient::new(), RiskEngine::new(limits()), ctx);

        for i in 1..=2u128 {
            let req = buy(dec!(4), i);
            let ack = g.place_order(req.clone()).await.expect("first two fit");
            // The runtime does this wiring; here it is explicit so the test shows it.
            tracker.write().unwrap().on_ack(
                &req,
                &OrderAck {
                    order_id: Some(axon_core::OrderId::new(i as u64)),
                    ..ack
                },
                i as i64,
            );
        }
        assert_eq!(g.inner.placed(), 2);
        assert_eq!(tracker.read().unwrap().resting_exposure(SYM), dec!(8));

        let err = g.place_order(buy(dec!(4), 3)).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("projected position")));
        assert_eq!(g.inner.placed(), 2, "the third never reached the venue");

        // With StaticRiskContext, which sees only filled position, it would go through
        // — which is exactly why the production path must not use it.
        let loose = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        assert!(loose.place_order(buy(dec!(4), 3)).await.is_ok());
    }

    // ── the loss limit ───────────────────────────────────────────────────────

    use crate::loss::{LossBreach, LossLimiter, LossScope};

    fn tripped() -> Arc<LossLimiter> {
        let l = Arc::new(LossLimiter::new(crate::loss::LossLimits {
            session: dec!(1),
            day: Decimal::ZERO,
        }));
        l.trip(LossBreach {
            scope: LossScope::Session,
            loss: dec!(2),
            limit: dec!(1),
            marked: false,
        });
        l
    }

    #[tokio::test]
    async fn a_tripped_loss_limit_refuses_new_exposure_and_still_lets_the_position_out() {
        // The whole shape of this gate, and the reason it is not `HaltSwitch::halt`. A
        // session that has lost more than its operator declared does not want to stop
        // acting; it wants to get out, and getting out is an *order*. Halting it strands
        // the exposure that caused the loss in the market that is causing it.
        let mut long = Position::flat(SYM);
        long.qty = dec!(3);
        let g = guarded(
            StaticRiskContext::new()
                .with_position(long)
                .with_mark(SYM, dec!(100)),
        )
        .with_loss_limiter(tripped());

        // Adding: refused, and never reaches the venue.
        let err = g.place_order(buy(dec!(1), 1)).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::Rejected(ref m) if m.contains("loss limit") && m.contains("de-risk only")),
            "{err:?}"
        );
        assert_eq!(g.inner.placed(), 0);

        // Reducing: through, with no reduce-only flag needed and no mark consulted.
        let mut out = buy(dec!(2), 2);
        out.side = Side::Sell;
        assert!(g.place_order(out).await.is_ok());

        // Flattening exactly: through. Reaching zero is a reduction.
        let mut flat = buy(dec!(3), 3);
        flat.side = Side::Sell;
        assert!(g.place_order(flat).await.is_ok());
        assert_eq!(g.inner.placed(), 2);

        // Cancels were never gated and this must not change that: refusing a cancel
        // while trying to get smaller can only strand the position.
        assert!(g.cancel_all().await.is_ok());
    }

    #[tokio::test]
    async fn a_tripped_loss_limit_refuses_an_order_that_flips_through_zero() {
        // The case a magnitude-only check waves through: long 3, sell 5 lands at short
        // 2, whose magnitude is smaller while the exposure is on the other side of the
        // market. A kill switch that admitted this would let a losing session reverse
        // into a new position and call it de-risking.
        let mut long = Position::flat(SYM);
        long.qty = dec!(3);
        let g = guarded(
            StaticRiskContext::new()
                .with_position(long)
                .with_mark(SYM, dec!(100)),
        )
        .with_loss_limiter(tripped());
        let mut flip = buy(dec!(5), 1);
        flip.side = Side::Sell;
        let err = g.place_order(flip).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("loss limit")));
        assert_eq!(g.inner.placed(), 0);
    }

    #[tokio::test]
    async fn a_reduce_only_flag_does_not_get_past_a_tripped_loss_limit_on_its_own() {
        // The flag is a request to the venue; the switch is arithmetic about the
        // position we hold. They disagree exactly where it matters — a reduce-only order
        // against a flat book reduces nothing, the venue would drop it, and a session
        // that let it through would be one whose kill switch can be talked past by
        // setting a bit.
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)))
            .with_loss_limiter(tripped());
        let mut ro = buy(dec!(1), 1);
        ro.reduce_only = true;
        let err = g.place_order(ro).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("loss limit")));
        assert_eq!(g.inner.placed(), 0);
    }

    #[tokio::test]
    async fn a_tripped_loss_limit_measures_what_is_held_and_not_what_is_projected() {
        // `risk_position` inflates the position by every resting order, which is right
        // for a size cap and wrong here: flat with a resting buy of 4 projects long 4,
        // so a sell of 4 would read as flattening while in fact opening a short 4 — the
        // switch admitting new exposure on the session it was tripped to stop.
        let tracker = Arc::new(RwLock::new(OrderTracker::new()));
        let marks = Arc::new(MarkCache::new());
        marks.set_mark(SYM, dec!(100), 1);
        let ctx = TrackerRiskContext::new(tracker.clone(), marks);
        let g = GuardedClient::new(SpyClient::new(), RiskEngine::new(limits()), ctx)
            .with_loss_limiter(tripped());

        // One resting buy of 4, no fills. Held is flat; projected is long 4.
        let req = buy(dec!(4), 1);
        tracker.write().unwrap().on_ack(
            &req,
            &OrderAck {
                cloid: req.cloid,
                order_id: Some(axon_core::OrderId::new(1)),
                status: OrderStatus::Resting,
            },
            1,
        );
        assert_eq!(g.context().position(SYM).qty, dec!(4), "projected");
        assert_eq!(g.context().held_position(SYM).qty, Decimal::ZERO, "held");

        let mut sell = buy(dec!(4), 2);
        sell.side = Side::Sell;
        let err = g.place_order(sell).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::Rejected(ref m) if m.contains("loss limit")),
            "a sell from flat opens a short, whatever the projection says: {err:?}"
        );
        assert_eq!(g.inner.placed(), 0);
    }

    #[tokio::test]
    async fn a_batch_cannot_flip_a_position_past_a_tripped_loss_limit_one_reducing_order_at_a_time()
    {
        // Two sells of 1 against a long of 1 each "reduce" the same long if the held
        // position is not carried forward, and together they leave a short. The batch is
        // measured as a unit for the de-risk question exactly as it already is for the
        // size caps.
        let mut long = Position::flat(SYM);
        long.qty = dec!(1);
        let g = guarded(
            StaticRiskContext::new()
                .with_position(long)
                .with_mark(SYM, dec!(100)),
        )
        .with_loss_limiter(tripped());
        let mut a = buy(dec!(1), 1);
        a.side = Side::Sell;
        let mut b = buy(dec!(1), 2);
        b.side = Side::Sell;
        let err = g.place_batch(vec![a.clone(), b]).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("loss limit")));
        assert_eq!(g.inner.placed(), 0, "the whole batch must be refused");

        // The one that really does flatten still goes.
        assert!(g.place_batch(vec![a]).await.is_ok());
    }

    #[tokio::test]
    async fn an_untripped_loss_limit_is_completely_transparent() {
        // The default every offline path and every config written before this existed
        // gets. A gate that arrives switched on is one that takes a session down on an
        // upgrade.
        let g = guarded(StaticRiskContext::new().with_mark(SYM, dec!(100)));
        assert!(g.place_order(buy(dec!(3), 1)).await.is_ok());
        assert!(!g.loss_limiter().is_tripped());

        let declared = Arc::new(LossLimiter::new(crate::loss::LossLimits {
            session: dec!(1),
            day: dec!(1),
        }));
        let g =
            guarded(StaticRiskContext::new().with_mark(SYM, dec!(100))).with_loss_limiter(declared);
        assert!(
            g.place_order(buy(dec!(3), 2)).await.is_ok(),
            "declared but clear"
        );
    }

    #[tokio::test]
    async fn notional_cap_is_enforced_against_the_mark_not_the_limit_price() {
        let g = GuardedClient::new(
            SpyClient::new(),
            RiskEngine::new(RiskLimits {
                max_position: dec!(10),
                max_notional: dec!(1000),
                max_order_qty: dec!(5),
            }),
            // Order prices at 100 but the mark is 500: 3 * 500 = 1500 > 1000.
            StaticRiskContext::new().with_mark(SYM, dec!(500)),
        );
        let err = g.place_order(buy(dec!(3), 1)).await.unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("notional")));
    }
}
