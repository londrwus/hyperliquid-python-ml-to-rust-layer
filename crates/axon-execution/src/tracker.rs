//! [`OrderTracker`] — our model of what the venue believes, rebuilt from the
//! execution events on the bus.
//!
//! An [`OrderAck`] tells us a submit succeeded. It does not tell us the order is
//! still there, how much of it filled, or whether the venue cancelled it out from
//! under us. Only the venue's own lifecycle stream does, so the tracker consumes
//! [`ExecEvent`]s as an [`EventHandler`] and treats the venue as the authority.
//!
//! Four failure modes shape the design; each has a test named after it:
//!
//! 1. **Duplicate fills.** A reconnect replays a snapshot, so the same execution
//!    arrives twice. Applying it twice silently doubles the position — the worst kind
//!    of bug, because nothing errors. Every fill is deduped on
//!    [`Fill::trade_id`](axon_core::Fill::trade_id).
//! 2. **Orders we never submitted.** After a restart, or with a second process on the
//!    same account, the venue reports orders this tracker has no record of. Ignoring
//!    them means a cancel-all sweep misses them and risk under-counts exposure, so
//!    they are *adopted* rather than dropped.
//! 3. **Out-of-order updates.** Streams reorder, and a stale `Resting` arriving after
//!    a `Filled` must not resurrect a finished order. Terminal is terminal.
//! 4. **Resting exposure.** A resting order is exposure the moment it is accepted, not
//!    when it fills. [`OrderTracker::risk_position`] therefore reports the filled
//!    position *plus* the worst case that every live order fills — which is the view
//!    the pre-trade gate must check against, and what closes the aggregation gap
//!    documented in [`crate::guard`].
//! 5. **Fields the venue never reports.** A venue's order list carries no
//!    time-in-force and no reduce-only flag, so an adopted order has both only as far
//!    as somebody is willing to guess. [`TrackedOrder::tif`] and
//!    [`TrackedOrder::reduce_only`] are therefore `Option`s: known for an order we
//!    acked, `None` for one we adopted, and never invented. A guessed `Gtc` that
//!    happened to match the order a caller was about to place would leave the wrong
//!    quote resting and report success (ADR-0020 §7).

use std::collections::{HashMap, HashSet, VecDeque};

use axon_core::{
    Cloid, Decimal, Event, EventHandler, ExecEvent, Fill, Liquidity, Nanos, OrderId, OrderStatus,
    OrderUpdate, Position, Side, SymbolId, Tif,
};
use axon_providers::{OrderAck, OrderRequest};

/// One order as we currently understand it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedOrder {
    /// Our client id. Synthesized for an adopted order the venue reported without one.
    pub cloid: Cloid,
    /// The venue id, once known. `None` between a local submit and the venue's reply.
    pub order_id: Option<OrderId>,
    pub symbol_id: SymbolId,
    pub side: Side,
    pub price: Option<Decimal>,
    pub orig_qty: Decimal,
    pub filled_qty: Decimal,
    pub status: OrderStatus,
    /// Event time of the newest update applied. Used to reject stale updates.
    pub last_update: Nanos,
    /// The order's time-in-force — `None` when nobody can vouch for it.
    ///
    /// A venue's order list carries no time-in-force, so the only orders whose TIF is
    /// knowable are the ones we submitted ourselves: [`OrderTracker::on_ack`] takes it
    /// off the [`OrderRequest`] that produced the ack. For an *adopted* order this is
    /// `None` and must stay `None` — a plausible-looking `Gtc` is worse than no answer,
    /// because a caller comparing it against an order it is about to place would leave
    /// the wrong quote resting and never find out (ADR-0020 §7). The type says "I do
    /// not know" so that the comparison can refuse instead of guessing.
    pub tif: Option<Tif>,
    /// Whether the order may only reduce the position — `None` when unknown, for the
    /// same reason as [`Self::tif`], and it is the more dangerous of the two: a
    /// reduce-only quote mistaken for an opening one leaves the strategy sitting flat
    /// behind an order that cannot ever move it.
    pub reduce_only: Option<bool>,
    /// Event time at which this order started working, as far as we can tell.
    ///
    /// The ack's event time for an order we placed. For an adopted one it is the first
    /// frame we saw, which is a **lower bound** on its real age — the venue does not
    /// tell us when it was placed, so an order inherited across a restart looks younger
    /// than it is. That direction is the safe one: it can only make an age-based rule
    /// hold an order longer than intended, never cancel one earlier.
    pub placed_ts: Nanos,
    /// True when the venue told us about an order we never submitted.
    pub adopted: bool,
}

impl TrackedOrder {
    /// Size still working at the venue.
    pub fn remaining_qty(&self) -> Decimal {
        (self.orig_qty - self.filled_qty).max(Decimal::ZERO)
    }

    /// Signed remaining size: positive for a resting buy, negative for a resting sell.
    fn signed_remaining(&self) -> Decimal {
        match self.side {
            Side::Buy => self.remaining_qty(),
            Side::Sell => -self.remaining_qty(),
        }
    }
}

/// How our position accounting compares with the venue's account snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Drift {
    /// Equity the venue last reported.
    pub venue_equity: Decimal,
    /// Event time of that snapshot.
    pub ts_event: Nanos,
}

/// Our own accounting, split by whether the trade happened during this session.
///
/// **A process does not start with an empty fill history, and the reason is the
/// venue.** Hyperliquid's `userFills` replays a snapshot of recent fills on every
/// subscribe, and the tracker applies them: that is what makes a restarted process
/// know its own position. It also means these totals are non-zero seconds after
/// startup, describing trades that happened before this session existed — measured
/// live on 2026-07-27, where a session that had placed no order at all opened with
/// `r +0.0161 fee 0.1031` over 22 replayed fills.
///
/// Reporting that as the session's P&L would be wrong twice: the bottom line would be
/// somebody else's, and the drift against `accountValue` — which *already* includes
/// those trades, because they are in the balance — would carry a permanent offset that
/// makes the cross-check useless for detecting anything new.
///
/// **The split is on the fill's own `ts_event`, not on when it arrived.** Arrival order
/// is not a fact anyone controls: the first measurement of this took the baseline from
/// the first `clearinghouseState` reply, and the `userFills` snapshot landed *after* it,
/// so every replayed fill counted as the session's. A trade's execution time is the
/// venue's own statement about when it happened, and a trade that happened before this
/// process started cannot be this process's work whatever order the frames arrive in.
///
/// One case this does **not** get right, and it is inherent rather than an oversight:
/// a session that inherits an open position and closes it books the whole realized
/// P&L, including the part that accrued before it started, while `accountValue` had
/// already moved with the mark. The account is left flat between sessions precisely so
/// this does not arise — and when it does, the drift figure is what reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Money {
    /// Realized P&L over closed quantity, summed across symbols.
    pub realized: Decimal,
    /// Fee, signed as [`Fill::fee`] is: positive is a cost, negative is a rebate.
    pub fees: Decimal,
    /// The venue's own realized-P&L attribution over the same fills.
    pub venue_closed_pnl: Decimal,
    pub maker_fills: u64,
    pub taker_fills: u64,
}

impl Money {
    fn apply(&mut self, f: &Fill, realized_delta: Decimal) {
        self.realized += realized_delta;
        self.fees += f.fee;
        self.venue_closed_pnl += f.closed_pnl;
        match f.liquidity {
            Liquidity::Maker => self.maker_fills += 1,
            Liquidity::Taker => self.taker_fills += 1,
        }
    }

    pub fn fills(&self) -> u64 {
        self.maker_fills + self.taker_fills
    }
}

impl std::ops::Add for Money {
    type Output = Money;

    fn add(self, o: Money) -> Money {
        Money {
            realized: self.realized + o.realized,
            fees: self.fees + o.fees,
            venue_closed_pnl: self.venue_closed_pnl + o.venue_closed_pnl,
            maker_fills: self.maker_fills + o.maker_fills,
            taker_fills: self.taker_fills + o.taker_fills,
        }
    }
}

/// Reconciles acks and venue lifecycle events into a coherent order/position book.
#[derive(Debug, Default)]
pub struct OrderTracker {
    orders: HashMap<Cloid, TrackedOrder>,
    /// Venue id → our id, so an event that only carries an `oid` still finds its order.
    by_order_id: HashMap<OrderId, Cloid>,
    positions: HashMap<SymbolId, Position>,
    /// Execution ids already applied — the dedup set. Bounded by
    /// [`Self::MAX_SEEN_TRADES`] so a long-lived process cannot grow it without limit.
    seen_trades: HashSet<u64>,
    /// The same ids in arrival order, so the oldest can be evicted when the window is
    /// full. A `VecDeque`, not a `Vec`: `Vec::remove(0)` would memmove the whole
    /// 100k-element buffer on every fill once the window filled up, which is exactly
    /// the kind of hidden O(n) the hot path must not have.
    seen_order: VecDeque<u64>,
    last_snapshot: Option<Drift>,
    /// The **first** account snapshot this tracker ever saw, kept for the whole
    /// session. Without it there is no baseline to measure a session's own P&L
    /// against: `venue_equity` alone answers "what is the account worth", and the
    /// question a live session has is "what has *this session* done to it".
    first_snapshot: Option<Drift>,
    /// Venue execution time at or after which a fill is **this session's**. `0` counts
    /// everything, which is what a backtest and a replay want: there is no earlier
    /// session for a canned log to inherit from. See [`Money`].
    session_start_ns: Nanos,
    /// Fills stamped before [`Self::session_start_ns`] — the venue's replay.
    inherited: Money,
    /// Fills stamped at or after it: what this session actually did.
    session: Money,
    /// Fills that arrived for an order we have no record of at all, counted so a
    /// caller can tell "nothing happened" from "something happened we can't explain".
    ///
    /// The money above is accumulated **here** rather than in a P&L component of its
    /// own for one reason: this is where the dedup lives. A fee total that
    /// double-counts a fill replayed after a reconnect is worse than no fee total,
    /// because it is wrong in the direction that makes a losing session look
    /// profitable, and nothing else in the process knows which trade ids have already
    /// been applied.
    orphan_fills: u64,
    /// Times [`Self::adopt_position`] changed a position. See that method for why this
    /// is an operator action and not something the reconcile poll may do.
    adopted_positions: u64,
}

impl OrderTracker {
    /// Cap on the dedup set. A venue trade id is only replayed near a reconnect, so
    /// remembering the most recent window is sufficient; unbounded growth is not.
    pub const MAX_SEEN_TRADES: usize = 100_000;

    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful submit. `req` supplies what the ack omits — side, size and
    /// price — which is exactly what the tracker needs to reason about exposure.
    pub fn on_ack(&mut self, req: &OrderRequest, ack: &OrderAck, ts_event: Nanos) {
        if let Some(oid) = ack.order_id {
            self.by_order_id.insert(oid, ack.cloid);
        }
        let order = TrackedOrder {
            cloid: ack.cloid,
            order_id: ack.order_id,
            symbol_id: req.symbol_id,
            side: req.side,
            price: req.price,
            orig_qty: req.qty,
            filled_qty: Decimal::ZERO,
            status: ack.status,
            last_update: ts_event,
            // The one moment either of these is knowable: the request is in hand.
            // Nothing the venue sends afterwards carries them, so an order that only
            // ever arrives through `apply_update` can never learn them.
            tif: Some(req.tif),
            reduce_only: Some(req.reduce_only),
            placed_ts: ts_event,
            adopted: false,
        };
        // A submit that came back already terminal (an IOC that filled outright, or a
        // rejection) is recorded only through its effects — there is nothing to track.
        if ack.status.is_terminal() {
            if ack.status == OrderStatus::Filled {
                // The fill event carries the price and quantity; do not guess here.
                self.orders.insert(ack.cloid, order);
            }
            return;
        }
        self.orders.insert(ack.cloid, order);
    }

    /// Overwrite our position in `symbol` with the venue's own, and count it.
    ///
    /// **This is not on the streaming path and must not be put there.** The reconcile
    /// poll reports [`crate::Drift`]-shaped position divergence and deliberately never
    /// writes it, for the reason `axon_runtime::reconcile` gives about orders: a view
    /// this process corrected itself into agreement is a view that can no longer
    /// disagree, and the disagreement is the finding. A fresh session's tracker converges
    /// on the venue by the `userFills` replay instead, which is the right mechanism
    /// because a fill is evidence and a snapshot is a summary.
    ///
    /// What this exists for is the one case that mechanism does not cover: an **operator
    /// flattening a position the tracker has not learned yet**. Measured on 2026-07-27 —
    /// the documented flatten emits a target of zero, the planner subtracts it from the
    /// tracked position, a tracker that believes it is flat produces a delta of zero and
    /// therefore no order, and the operator who worked around that by hand-writing a
    /// target turned a −0.01 short into a +0.01 long with one order. "Go flat" must not
    /// depend on a race with a fill replay.
    ///
    /// Three things it refuses to do, and each is a way this could corrupt the accounting
    /// it is standing next to:
    ///
    /// - **It does not touch `realized_pnl`, the fees, or the fill counts.** A position
    ///   snapshot is not a trade. Synthesizing the difference as a `Fill` would be the
    ///   obvious shortcut and it would book a P&L nobody earned, on the side that makes a
    ///   losing session look better.
    /// - **It does not invent an average price.** `avg_px` comes from the venue's own
    ///   `entryPx`; without one the existing average is kept, because a fabricated entry
    ///   price is an unrealized P&L that is wrong by an unknown amount.
    /// - **It does not clear anything else.** Resting orders, the dedup set and the
    ///   equity baselines are untouched: this answers "how much do I hold" and nothing
    ///   else.
    ///
    /// Returns the previous quantity, so the caller can report what it changed rather
    /// than announcing a correction it might not have made.
    pub fn adopt_position(
        &mut self,
        symbol: SymbolId,
        qty: Decimal,
        avg_px: Option<Decimal>,
    ) -> Decimal {
        let p = self
            .positions
            .entry(symbol)
            .or_insert_with(|| Position::flat(symbol));
        let was = p.qty;
        p.qty = qty;
        if let Some(px) = avg_px {
            p.avg_px = px;
        }
        if was != qty {
            self.adopted_positions += 1;
        }
        was
    }

    /// How many times a position has been overwritten from the venue's own snapshot.
    ///
    /// Reported for the reason this crate applies to every other counter: a correction
    /// that happened and was not reported is indistinguishable from one that did not, and
    /// this one silently changes what every subsequent plan is arithmetic about.
    pub fn adopted_positions(&self) -> u64 {
        self.adopted_positions
    }

    /// Filled position for `symbol` — what we actually hold.
    pub fn position(&self, symbol: SymbolId) -> Position {
        self.positions
            .get(&symbol)
            .cloned()
            .unwrap_or_else(|| Position::flat(symbol))
    }

    /// The position risk should be checked against: what we hold **plus** the worst
    /// case that every live order fills in full.
    ///
    /// A gate that only sees filled position will happily add a 21st order to twenty
    /// resting ones. This is the number that prevents that.
    pub fn risk_position(&self, symbol: SymbolId) -> Position {
        let mut p = self.position(symbol);
        let resting: Decimal = self
            .orders
            .values()
            .filter(|o| o.symbol_id == symbol && !o.status.is_terminal())
            .map(|o| o.signed_remaining())
            .sum();
        p.qty += resting;
        p
    }

    /// Every instrument this tracker knows exposure in — held or resting — sorted.
    ///
    /// The union of the two, not just the positions: an account that is flat everywhere
    /// and has ten resting orders has ten instruments' worth of projected exposure, and
    /// a portfolio bound measured over positions alone would call it empty.
    ///
    /// **Sorted**, because the two maps behind it are `HashMap`s and a portfolio check
    /// derived from them must reach the same verdict on a replay as it did live —
    /// `axon_risk::PortfolioReject::Unpriced` names the first unpriced leg, and "first"
    /// has to mean something.
    pub fn exposed_symbols(&self) -> Vec<SymbolId> {
        let mut out: Vec<SymbolId> = self.positions.keys().copied().collect();
        for o in self.orders.values() {
            if !o.status.is_terminal() && !out.contains(&o.symbol_id) {
                out.push(o.symbol_id);
            }
        }
        out.sort();
        out
    }

    /// Signed sum of remaining size on live orders for `symbol`.
    pub fn resting_exposure(&self, symbol: SymbolId) -> Decimal {
        self.orders
            .values()
            .filter(|o| o.symbol_id == symbol && !o.status.is_terminal())
            .map(|o| o.signed_remaining())
            .sum()
    }

    pub fn order(&self, cloid: Cloid) -> Option<&TrackedOrder> {
        self.orders.get(&cloid)
    }

    pub fn order_by_id(&self, order_id: OrderId) -> Option<&TrackedOrder> {
        self.by_order_id
            .get(&order_id)
            .and_then(|c| self.orders.get(c))
    }

    /// Every order still working at the venue — the input to a cancel-all sweep.
    pub fn open_orders(&self) -> impl Iterator<Item = &TrackedOrder> {
        self.orders.values().filter(|o| !o.status.is_terminal())
    }

    pub fn open_count(&self) -> usize {
        self.open_orders().count()
    }

    /// The venue's last account snapshot, for drift checks against our own math.
    pub fn last_snapshot(&self) -> Option<Drift> {
        self.last_snapshot
    }

    /// The venue's **first** account snapshot — the equity this session started from.
    ///
    /// The pair is what makes a session-scoped P&L possible at all. A single reading
    /// says what the account is worth; only the difference says what this session did,
    /// and an operator watching a live run is asking the second question.
    pub fn first_snapshot(&self) -> Option<Drift> {
        self.first_snapshot
    }

    /// Venue execution time from which a fill counts as this session's.
    ///
    /// Set once, by the composition root, from the wall clock at startup — a **named**
    /// wall-clock use in the same class as the dead-man's-switch deadline, and for a
    /// related reason: "did this trade happen before I existed" is a question about the
    /// world, and the session's own birth has no event time. `0` (the default) counts
    /// every fill, which is what a backtest and a replay want.
    pub fn set_session_start(&mut self, ns: Nanos) {
        self.session_start_ns = ns;
    }

    /// What **this session** did: fills stamped at or after [`Self::set_session_start`].
    pub fn session_money(&self) -> Money {
        self.session
    }

    /// What the venue replayed from before it. See [`Money`].
    pub fn inherited_money(&self) -> Money {
        self.inherited
    }

    /// Both together — every fill this process has applied.
    pub fn total_money(&self) -> Money {
        self.session + self.inherited
    }

    /// Fills that referenced an order we know nothing about. Non-zero means our view
    /// and the venue's have diverged and a resync is warranted.
    pub fn orphan_fills(&self) -> u64 {
        self.orphan_fills
    }

    /// Drop terminal orders. Called by the owner at a quiet moment — the tracker does
    /// not evict on its own, so a caller can still inspect an order's final state.
    pub fn retire_terminal(&mut self) {
        let dead: Vec<Cloid> = self
            .orders
            .iter()
            .filter(|(_, o)| o.status.is_terminal())
            .map(|(c, _)| *c)
            .collect();
        for cloid in dead {
            if let Some(o) = self.orders.remove(&cloid) {
                if let Some(oid) = o.order_id {
                    self.by_order_id.remove(&oid);
                }
            }
        }
    }

    /// Remember a trade id, evicting the oldest when the window is full.
    fn remember_trade(&mut self, trade_id: u64) {
        if self.seen_order.len() >= Self::MAX_SEEN_TRADES {
            if let Some(oldest) = self.seen_order.pop_front() {
                self.seen_trades.remove(&oldest);
            }
        }
        self.seen_order.push_back(trade_id);
        self.seen_trades.insert(trade_id);
    }

    fn apply_fill(&mut self, f: &Fill) {
        // Dedup first: a replayed snapshot must be a no-op, not a doubled position.
        if self.seen_trades.contains(&f.trade_id) {
            return;
        }
        self.remember_trade(f.trade_id);

        let position = self
            .positions
            .entry(f.symbol_id)
            .or_insert_with(|| Position::flat(f.symbol_id));
        // Measured across the call rather than read off the position afterwards: what
        // this fill realized is a *delta*, and a running total cannot be attributed to
        // one side of the session boundary.
        let realized_before = position.realized_pnl;
        position.apply_fill(f.side, f.qty, f.price);
        let realized_delta = position.realized_pnl - realized_before;

        // Above the attribution below, deliberately, so an **orphan** fill — one for an
        // order this process has no record of — is counted in the money exactly as it
        // is already counted in the position. The alternative reads worse than it
        // sounds: a session on a shared account would report a position it did not ask
        // for and a P&L that excludes what that position cost, so the two halves of the
        // same fill would be accounted on opposite sides of a filter. The venue-equity
        // cross-check in the runtime's P&L view is where a fill that was never ours
        // becomes visible, and it can only work if both halves are on the same side.
        if f.ts_event < self.session_start_ns {
            self.inherited.apply(f, realized_delta);
        } else {
            self.session.apply(f, realized_delta);
        }

        // Attribute the fill to its order, by cloid if the venue echoed one, else by
        // venue id. A fill for an order we never saw still moved the position — it is
        // counted above and flagged here rather than silently dropped.
        let cloid = f
            .cloid
            .filter(|c| self.orders.contains_key(c))
            .or_else(|| self.by_order_id.get(&f.order_id).copied());
        let Some(cloid) = cloid else {
            self.orphan_fills += 1;
            return;
        };
        let Some(order) = self.orders.get_mut(&cloid) else {
            self.orphan_fills += 1;
            return;
        };
        order.filled_qty += f.qty;
        if order.order_id.is_none() {
            order.order_id = Some(f.order_id);
            self.by_order_id.insert(f.order_id, cloid);
        }
        order.last_update = order.last_update.max(f.ts_event);
        // A fill does not by itself prove the order is done — the venue's own update
        // does — but quantities can decide it unambiguously.
        if order.filled_qty >= order.orig_qty {
            order.status = OrderStatus::Filled;
        } else if !order.status.is_terminal() {
            order.status = OrderStatus::PartiallyFilled;
        }
    }

    fn apply_update(&mut self, u: &OrderUpdate) {
        let cloid = u
            .cloid
            .filter(|c| self.orders.contains_key(c))
            .or_else(|| self.by_order_id.get(&u.order_id).copied());

        match cloid.and_then(|c| self.orders.get_mut(&c)) {
            Some(order) => {
                // Terminal is terminal: nothing can resurrect a finished order.
                if order.status.is_terminal() {
                    return;
                }

                // Filled quantity is *monotonic* — it can only ever grow — so the
                // venue's absolute figure and our running sum of fills combine with
                // `max`. That is what makes the tracker immune to the two channels'
                // clocks disagreeing: whichever frame arrives second, the answer is
                // the same. Trusting the newer timestamp instead would discard a
                // fill whenever the lifecycle stream lagged the fill stream.
                let venue_filled = (u.orig_qty - u.remaining_qty).max(Decimal::ZERO);
                order.filled_qty = order.filled_qty.max(venue_filled);

                // A terminal state must never be missed, whatever its timestamp — an
                // order we believe is live when it is not is the dangerous direction.
                // Non-terminal transitions defer to the newest frame so a reordered
                // one cannot walk the order backwards.
                let fresh = u.ts_event >= order.last_update;
                if u.status.is_terminal() || fresh {
                    order.status = u.status;
                }
                if fresh {
                    order.orig_qty = u.orig_qty;
                    if let Some(px) = u.price {
                        order.price = Some(px);
                    }
                }
                // Derived state must stay consistent with the monotonic quantity: a
                // fully-filled order is filled regardless of what the frame claimed.
                if order.filled_qty >= order.orig_qty && !order.status.is_terminal() {
                    order.status = OrderStatus::Filled;
                }
                order.last_update = order.last_update.max(u.ts_event);
                if order.order_id.is_none() {
                    order.order_id = Some(u.order_id);
                }
                self.by_order_id.insert(u.order_id, order.cloid);
            }
            None => self.adopt(u),
        }
    }

    /// Take ownership of an order the venue reports that we have no record of.
    ///
    /// Synthesizes a `cloid` from the venue id so the order is addressable in the same
    /// map as everything else. `adopted` marks it so a caller can tell it apart —
    /// notably, it may belong to another process on the same account.
    fn adopt(&mut self, u: &OrderUpdate) {
        if u.status.is_terminal() {
            // Nothing to adopt: a terminal order we never knew about needs no tracking.
            return;
        }
        let cloid = u
            .cloid
            .unwrap_or_else(|| Cloid::new(u.order_id.get() as u128));
        self.by_order_id.insert(u.order_id, cloid);
        self.orders.insert(
            cloid,
            TrackedOrder {
                cloid,
                order_id: Some(u.order_id),
                symbol_id: u.symbol_id,
                side: u.side,
                price: u.price,
                orig_qty: u.orig_qty,
                filled_qty: (u.orig_qty - u.remaining_qty).max(Decimal::ZERO),
                status: u.status,
                last_update: u.ts_event,
                // Unknown, and left unknown on purpose. `OrderUpdate` carries neither
                // field because the venue's order list does not, and synthesizing the
                // common case here would put a guess where a caller expects a fact.
                tif: None,
                reduce_only: None,
                // The first frame we saw, not the placement: see [`TrackedOrder::placed_ts`].
                placed_ts: u.ts_event,
                adopted: true,
            },
        );
    }
}

impl EventHandler for OrderTracker {
    fn on_event(&mut self, _ts_event: Nanos, event: &Event) {
        // Market data shares the bus so that fills order against the trades that
        // caused them; the book itself is the market-data processor's business.
        let Event::Exec(e) = event else { return };
        match e {
            ExecEvent::Fill(f) => self.apply_fill(f),
            ExecEvent::Order(u) => self.apply_update(u),
            ExecEvent::Account(a) => {
                let drift = Drift {
                    venue_equity: a.equity,
                    ts_event: a.ts_event,
                };
                // `get_or_insert` and not an assignment: the baseline is the *first*
                // reading and it must survive every later one, because it is the only
                // thing that turns an account balance into this session's own result.
                self.first_snapshot.get_or_insert(drift);
                self.last_snapshot = Some(drift);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{AccountSnapshot, Liquidity, Tif};
    use rust_decimal_macros::dec;

    const SYM: SymbolId = SymbolId::new(1);

    fn req(side: Side, qty: Decimal, px: Decimal, cloid: u128) -> OrderRequest {
        OrderRequest::limit(SYM, side, qty, px, Tif::Gtc, Cloid::new(cloid))
    }

    fn ack(cloid: u128, oid: u64, status: OrderStatus) -> OrderAck {
        OrderAck {
            cloid: Cloid::new(cloid),
            order_id: Some(OrderId::new(oid)),
            status,
        }
    }

    #[test]
    fn exposed_symbols_is_the_union_of_positions_and_live_orders_and_is_sorted() {
        // A portfolio bound measured over *positions* alone calls an account with ten
        // resting orders and no fills empty — which is exactly the account that is about
        // to have ten positions. And the order has to be stable: the two maps behind this
        // are `HashMap`s, and the portfolio check derived from it names the first
        // unpriced leg, so a replay must reach the same verdict as the session it
        // reproduces.
        let mut t = OrderTracker::new();
        let eth = SymbolId::new(2);
        let sol = SymbolId::new(3);

        // A position with nothing resting…
        t.adopt_position(sol, dec!(1), Some(dec!(100)));
        assert_eq!(t.exposed_symbols(), vec![sol]);

        // …and a resting order with no position. Both count.
        let mut r = req(Side::Buy, dec!(1), dec!(100), 7);
        r.symbol_id = eth;
        t.on_ack(&r, &ack(7, 70, OrderStatus::Resting), 1);
        assert_eq!(
            t.exposed_symbols(),
            vec![eth, sol],
            "sorted, not hash order"
        );

        // A terminal order is not exposure.
        let mut cancelled = update(
            70,
            Some(7),
            Side::Buy,
            OrderStatus::Cancelled,
            dec!(1),
            dec!(1),
            2,
        );
        cancelled.symbol_id = eth;
        feed(&mut t, ExecEvent::Order(cancelled));
        assert_eq!(t.exposed_symbols(), vec![sol]);
    }

    fn fill(
        oid: u64,
        cloid: Option<u128>,
        side: Side,
        qty: Decimal,
        px: Decimal,
        tid: u64,
    ) -> Fill {
        Fill {
            symbol_id: SYM,
            order_id: OrderId::new(oid),
            cloid: cloid.map(Cloid::new),
            side,
            qty,
            price: px,
            fee: dec!(0),
            closed_pnl: dec!(0),
            liquidity: Liquidity::Maker,
            trade_id: tid,
            ts_event: 100,
        }
    }

    fn update(
        oid: u64,
        cloid: Option<u128>,
        side: Side,
        status: OrderStatus,
        orig: Decimal,
        remaining: Decimal,
        ts: Nanos,
    ) -> OrderUpdate {
        OrderUpdate {
            symbol_id: SYM,
            order_id: OrderId::new(oid),
            cloid: cloid.map(Cloid::new),
            side,
            status,
            price: Some(dec!(100)),
            orig_qty: orig,
            remaining_qty: remaining,
            cancel_reason: None,
            ts_event: ts,
        }
    }

    fn feed(t: &mut OrderTracker, e: ExecEvent) {
        let ev = Event::Exec(e);
        t.on_event(ev.ts_event(), &ev);
    }

    #[test]
    fn tracks_an_ack_then_a_fill_to_completion() {
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(2), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        assert_eq!(t.open_count(), 1);
        assert_eq!(
            t.order_by_id(OrderId::new(10)).unwrap().cloid,
            Cloid::new(1)
        );

        feed(
            &mut t,
            ExecEvent::Fill(fill(10, Some(1), Side::Buy, dec!(2), dec!(100), 555)),
        );
        assert_eq!(t.position(SYM).qty, dec!(2));
        assert_eq!(t.order(Cloid::new(1)).unwrap().status, OrderStatus::Filled);
        assert_eq!(t.open_count(), 0);
    }

    #[test]
    fn duplicate_fills_are_applied_once() {
        // A reconnect replays the snapshot. Twice-applied means a doubled position and
        // nothing to alert on, so this is the single most important test here.
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(5), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        let f = fill(10, Some(1), Side::Buy, dec!(2), dec!(100), 777);
        feed(&mut t, ExecEvent::Fill(f.clone()));
        feed(&mut t, ExecEvent::Fill(f.clone()));
        feed(&mut t, ExecEvent::Fill(f));
        assert_eq!(t.position(SYM).qty, dec!(2), "fill applied exactly once");
        assert_eq!(t.order(Cloid::new(1)).unwrap().filled_qty, dec!(2));
    }

    #[test]
    fn orders_we_never_submitted_are_adopted() {
        // After a restart the venue still holds our orders. Ignoring them would hide
        // real exposure and make a cancel-all sweep incomplete.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Order(update(
                99,
                None,
                Side::Sell,
                OrderStatus::Resting,
                dec!(3),
                dec!(3),
                5,
            )),
        );
        assert_eq!(t.open_count(), 1);
        let o = t.order_by_id(OrderId::new(99)).unwrap();
        assert!(o.adopted);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(t.resting_exposure(SYM), dec!(-3));

        // A terminal update for an unknown order needs no tracking.
        let mut t2 = OrderTracker::new();
        feed(
            &mut t2,
            ExecEvent::Order(update(
                98,
                None,
                Side::Sell,
                OrderStatus::Cancelled,
                dec!(3),
                dec!(0),
                5,
            )),
        );
        assert_eq!(t2.open_count(), 0);
        assert!(t2.order_by_id(OrderId::new(98)).is_none());
    }

    #[test]
    fn an_adopted_order_reports_an_unknown_tif_rather_than_a_plausible_one() {
        // The failure this prevents is silent and total: `Gtc`/not-reduce-only is the
        // overwhelmingly common shape, so a guess would usually be right — and on the
        // occasion it was wrong (a reduce-only order left resting by a previous
        // incarnation) the caller comparing it against the opening order it was about
        // to place would leave that quote alone, and the strategy would sit flat behind
        // an order that cannot ever move it. `None` is the only answer that lets the
        // comparison refuse.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Order(update(
                99,
                None,
                Side::Sell,
                OrderStatus::Resting,
                dec!(3),
                dec!(3),
                5,
            )),
        );
        let o = t.order_by_id(OrderId::new(99)).unwrap();
        assert!(o.adopted);
        assert_eq!(o.tif, None, "the venue's order list does not carry one");
        assert_eq!(o.reduce_only, None);
        assert_eq!(o.placed_ts, 5, "the first frame we saw, a lower bound");

        // …and nothing the venue sends afterwards teaches it, because no venue frame
        // carries either field. An order that arrived this way stays unknown for life.
        feed(
            &mut t,
            ExecEvent::Order(update(
                99,
                None,
                Side::Sell,
                OrderStatus::PartiallyFilled,
                dec!(3),
                dec!(1),
                9,
            )),
        );
        let o = t.order_by_id(OrderId::new(99)).unwrap();
        assert_eq!(o.tif, None);
        assert_eq!(o.reduce_only, None);
    }

    #[test]
    fn an_acked_order_carries_the_tif_and_reduce_only_the_venue_never_echoes() {
        // The ack is the one moment both are knowable — the request is in hand. Drop
        // them here and they are unrecoverable, which is what forced the runtime to
        // keep a second map beside the tracker and what made a restarted session
        // cancel and replace every order its predecessor left.
        let mut t = OrderTracker::new();
        let mut req = req(Side::Buy, dec!(2), dec!(100), 1);
        req.tif = Tif::PostOnly;
        req.reduce_only = true;
        t.on_ack(&req, &ack(1, 10, OrderStatus::Resting), 4_000);

        let o = t.order(Cloid::new(1)).unwrap();
        assert!(!o.adopted);
        assert_eq!(o.tif, Some(Tif::PostOnly));
        assert_eq!(o.reduce_only, Some(true));
        assert_eq!(o.placed_ts, 4_000, "event time, not wall clock");

        // A later venue frame moves the quantities and leaves the fields it does not
        // carry alone — it must never downgrade a known TIF to a guess.
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::PartiallyFilled,
                dec!(2),
                dec!(1),
                5_000,
            )),
        );
        let o = t.order(Cloid::new(1)).unwrap();
        assert_eq!(o.tif, Some(Tif::PostOnly));
        assert_eq!(o.reduce_only, Some(true));
        assert_eq!(o.placed_ts, 4_000, "placement, not the last update");
        assert_eq!(o.last_update, 5_000);
    }

    #[test]
    fn stale_updates_never_resurrect_a_terminal_order() {
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(1), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::Cancelled,
                dec!(1),
                dec!(0),
                20,
            )),
        );
        assert_eq!(
            t.order(Cloid::new(1)).unwrap().status,
            OrderStatus::Cancelled
        );

        // A reordered "still resting" from before the cancel must be ignored.
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::Resting,
                dec!(1),
                dec!(1),
                15,
            )),
        );
        assert_eq!(
            t.order(Cloid::new(1)).unwrap().status,
            OrderStatus::Cancelled,
            "terminal is terminal"
        );
        assert_eq!(t.open_count(), 0);
    }

    #[test]
    fn risk_position_counts_resting_orders_as_exposure() {
        // The gap this closes: two resting buys of 4 plus a held 2 is 10 of exposure,
        // not 2. A gate that only sees filled position would allow an 11th unit.
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(2), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(10, Some(1), Side::Buy, dec!(2), dec!(100), 1)),
        );
        t.on_ack(
            &req(Side::Buy, dec!(4), dec!(99), 2),
            &ack(2, 11, OrderStatus::Resting),
            2,
        );
        t.on_ack(
            &req(Side::Buy, dec!(4), dec!(98), 3),
            &ack(3, 12, OrderStatus::Resting),
            3,
        );

        assert_eq!(t.position(SYM).qty, dec!(2), "filled only");
        assert_eq!(
            t.risk_position(SYM).qty,
            dec!(10),
            "filled + worst-case rest"
        );
        assert_eq!(t.resting_exposure(SYM), dec!(8));
    }

    #[test]
    fn filled_qty_is_monotonic_across_out_of_order_channels() {
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(10), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(10, Some(1), Side::Buy, dec!(3), dec!(100), 1)),
        );
        let o = t.order(Cloid::new(1)).unwrap();
        assert_eq!(o.status, OrderStatus::PartiallyFilled);
        assert_eq!(o.remaining_qty(), dec!(7));

        // The venue says only 4 remain, i.e. 6 filled — a fill never reached us. Its
        // frame is stamped t=50, *older* than the fill at t=100, because the two
        // channels are stamped independently. Filled quantity never decreases, so the
        // truth is at least 6 whichever frame is newer, and dropping the "stale" one
        // would leave us under-counting a real position.
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::PartiallyFilled,
                dec!(10),
                dec!(4),
                50,
            )),
        );
        assert_eq!(t.order(Cloid::new(1)).unwrap().filled_qty, dec!(6));
        assert_eq!(t.resting_exposure(SYM), dec!(4));

        // And a later frame reporting *less* fill cannot walk it back.
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::PartiallyFilled,
                dec!(10),
                dec!(9),
                200,
            )),
        );
        assert_eq!(t.order(Cloid::new(1)).unwrap().filled_qty, dec!(6));
    }

    #[test]
    fn a_terminal_update_lands_even_when_stamped_older() {
        // Believing an order is live when the venue has cancelled it is the dangerous
        // direction, so a terminal transition is never dropped for being "stale".
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(1), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            500,
        );
        feed(
            &mut t,
            ExecEvent::Order(update(
                10,
                Some(1),
                Side::Buy,
                OrderStatus::Cancelled,
                dec!(1),
                dec!(1),
                5, // long before the ack's own timestamp
            )),
        );
        assert_eq!(
            t.order(Cloid::new(1)).unwrap().status,
            OrderStatus::Cancelled
        );
        assert_eq!(t.open_count(), 0);
        assert_eq!(t.resting_exposure(SYM), dec!(0));
    }

    #[test]
    fn a_fill_for_an_unknown_order_still_moves_the_position_and_is_flagged() {
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Fill(fill(42, None, Side::Sell, dec!(1), dec!(100), 9)),
        );
        assert_eq!(
            t.position(SYM).qty,
            dec!(-1),
            "the position moved whether we knew the order or not"
        );
        assert_eq!(t.orphan_fills(), 1, "and the divergence is visible");
    }

    #[test]
    fn adopting_the_venues_position_moves_the_position_and_nothing_else() {
        // The operator flatten's one write, and the three things it must not do. The
        // tempting shortcut is to synthesize the difference as a `Fill`: that would book a
        // realized P&L nobody earned, add a fee nobody paid, and count a trade that never
        // happened — all in the direction that makes a losing session look better.
        let mut t = OrderTracker::new();
        let mut f = fill(1, None, Side::Buy, dec!(2), dec!(100), 1);
        f.fee = dec!(0.05);
        f.closed_pnl = dec!(0.5);
        feed(&mut t, ExecEvent::Fill(f));
        let money_before = t.total_money();
        assert_eq!(t.position(SYM).qty, dec!(2));

        // The venue says we are short 1 at 90 — a fill we never saw.
        let was = t.adopt_position(SYM, dec!(-1), Some(dec!(90)));
        assert_eq!(was, dec!(2), "the previous belief, returned to be reported");
        assert_eq!(t.position(SYM).qty, dec!(-1));
        assert_eq!(t.position(SYM).avg_px, dec!(90));
        assert_eq!(t.adopted_positions(), 1);
        assert_eq!(
            t.total_money(),
            money_before,
            "a position snapshot is not a trade"
        );

        // No entry price means the existing average is kept, never a zero: a fabricated
        // entry price is an unrealized P&L wrong by an unknown amount, and a zero one
        // prices it at the whole notional.
        t.adopt_position(SYM, dec!(-3), None);
        assert_eq!(t.position(SYM).avg_px, dec!(90));
        assert_eq!(t.adopted_positions(), 2);

        // Adopting what we already believe is not a correction and is not counted — the
        // count exists to say a correction *happened*, and a reconcile-shaped caller
        // agreeing with us every fifteen seconds would drown that.
        t.adopt_position(SYM, dec!(-3), None);
        assert_eq!(t.adopted_positions(), 2);
    }

    #[test]
    fn adopting_a_position_leaves_resting_orders_and_the_equity_baseline_alone() {
        // It answers "how much do I hold" and nothing else. Clearing the open orders would
        // hide exposure from `risk_position`, which is the direction that places the order
        // that breaches the limit; resetting the equity baseline would silently rebase the
        // session's own P&L and the day's loss bound with it.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Account(axon_core::AccountSnapshot {
                equity: dec!(1000),
                withdrawable: dec!(1000),
                margin_used: Decimal::ZERO,
                ts_event: 1,
            }),
        );
        let req = OrderRequest::limit(SYM, Side::Buy, dec!(4), dec!(100), Tif::Gtc, Cloid::new(7));
        t.on_ack(
            &req,
            &OrderAck {
                cloid: Cloid::new(7),
                order_id: Some(OrderId::new(7)),
                status: OrderStatus::Resting,
            },
            1,
        );
        assert_eq!(t.open_count(), 1);

        t.adopt_position(SYM, dec!(-1), None);
        assert_eq!(t.open_count(), 1, "the resting order is still exposure");
        assert_eq!(
            t.risk_position(SYM).qty,
            dec!(3),
            "short 1 plus a resting buy of 4"
        );
        assert_eq!(t.first_snapshot().map(|d| d.venue_equity), Some(dec!(1000)));
    }

    #[test]
    fn an_orphan_fills_fee_is_counted_on_the_same_side_as_the_position_it_moved() {
        // The two halves of one fill must not be accounted on opposite sides of a
        // filter. An orphan fill moves the position whether we knew the order or not
        // (the test above), so its cost has to move the money too — otherwise a session
        // on a shared account reports a position it did not ask for and a P&L that
        // excludes what that position cost, which reads as free exposure.
        let mut t = OrderTracker::new();
        let mut f = fill(42, None, Side::Sell, dec!(1), dec!(100), 9);
        f.fee = dec!(0.045);
        f.closed_pnl = dec!(-0.5);
        feed(&mut t, ExecEvent::Fill(f));
        assert_eq!(t.orphan_fills(), 1);
        assert_eq!(t.total_money().fees, dec!(0.045));
        assert_eq!(t.total_money().venue_closed_pnl, dec!(-0.5));
        assert_eq!(t.total_money().maker_fills, 1);
    }

    #[test]
    fn a_maker_rebate_is_a_negative_fee_and_is_not_clamped_away() {
        // `Fill::fee` is signed and a rebate is the negative case. Summing magnitudes
        // instead would make a rebate *cost* money, and on a post-only strategy that is
        // every fill.
        let mut t = OrderTracker::new();
        let mut f = fill(1, None, Side::Buy, dec!(1), dec!(100), 1);
        f.fee = dec!(-0.01);
        f.liquidity = Liquidity::Maker;
        feed(&mut t, ExecEvent::Fill(f));
        let mut g = fill(2, None, Side::Sell, dec!(1), dec!(100), 2);
        g.fee = dec!(0.045);
        g.liquidity = Liquidity::Taker;
        feed(&mut t, ExecEvent::Fill(g));
        assert_eq!(t.total_money().fees, dec!(0.035));
        let m = t.total_money();
        assert_eq!((m.maker_fills, m.taker_fills), (1, 1), "counted apart");
    }

    #[test]
    fn a_fill_stamped_before_the_session_started_is_the_venues_replay_and_not_ours() {
        // The split that keeps 22 replayed fills out of a session's bottom line. On the
        // fill's own execution time and not on arrival order, because arrival order is
        // not a fact anyone controls: the first version of this baselined at the first
        // `clearinghouseState` reply, and `userFills` landed after it.
        let mut t = OrderTracker::new();
        t.set_session_start(1_000);
        let mut old_fill = fill(1, None, Side::Buy, dec!(1), dec!(100), 1);
        old_fill.ts_event = 999;
        old_fill.fee = dec!(0.05);
        feed(&mut t, ExecEvent::Fill(old_fill));
        let mut ours = fill(2, None, Side::Sell, dec!(1), dec!(102), 2);
        ours.ts_event = 1_000;
        ours.fee = dec!(0.02);
        feed(&mut t, ExecEvent::Fill(ours));

        assert_eq!(t.inherited_money().fees, dec!(0.05));
        assert_eq!(t.inherited_money().fills(), 1);
        assert_eq!(
            t.session_money().fees,
            dec!(0.02),
            "at the boundary is ours"
        );
        assert_eq!(t.session_money().fills(), 1);
        assert_eq!(
            t.total_money().fees,
            dec!(0.07),
            "and the total is still whole"
        );
        // The realized P&L of a close is attributed to the fill that closed it — this
        // session's — which is the inherited-position case the docs name as inherent.
        assert_eq!(t.session_money().realized, dec!(2));
        assert_eq!(t.inherited_money().realized, Decimal::ZERO);
    }

    #[test]
    fn a_tracker_with_no_session_start_counts_every_fill_as_its_own() {
        // The backtest and replay case: a canned log has no earlier session to inherit
        // from, and a default that hid its fills would make an offline P&L read zero.
        let mut t = OrderTracker::new();
        let mut f = fill(1, None, Side::Buy, dec!(1), dec!(100), 1);
        f.ts_event = 1;
        f.fee = dec!(0.05);
        feed(&mut t, ExecEvent::Fill(f));
        assert_eq!(t.session_money().fees, dec!(0.05));
        assert_eq!(t.inherited_money().fills(), 0);
    }

    #[test]
    fn the_first_account_snapshot_survives_every_later_one() {
        // The baseline is what turns an account balance into *this session's* result,
        // and an assignment where the insert belongs would make the two readings equal
        // on every line — reporting every session as flat.
        let mut t = OrderTracker::new();
        for (i, equity) in [dec!(1000), dec!(1001), dec!(998)].into_iter().enumerate() {
            feed(
                &mut t,
                ExecEvent::Account(AccountSnapshot {
                    equity,
                    withdrawable: equity,
                    margin_used: dec!(0),
                    ts_event: 100 + i as Nanos,
                }),
            );
        }
        assert_eq!(t.first_snapshot().unwrap().venue_equity, dec!(1000));
        assert_eq!(t.last_snapshot().unwrap().venue_equity, dec!(998));
    }

    #[test]
    fn account_snapshots_are_retained_for_drift_checks() {
        let mut t = OrderTracker::new();
        assert!(t.last_snapshot().is_none());
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1234.5),
                withdrawable: dec!(1000),
                margin_used: dec!(234.5),
                ts_event: 900,
            }),
        );
        let d = t.last_snapshot().unwrap();
        assert_eq!(d.venue_equity, dec!(1234.5));
        assert_eq!(d.ts_event, 900);
    }

    #[test]
    fn retiring_terminal_orders_keeps_the_position() {
        let mut t = OrderTracker::new();
        t.on_ack(
            &req(Side::Buy, dec!(2), dec!(100), 1),
            &ack(1, 10, OrderStatus::Resting),
            1,
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(10, Some(1), Side::Buy, dec!(2), dec!(100), 1)),
        );
        assert!(t.order(Cloid::new(1)).is_some());
        t.retire_terminal();
        assert!(t.order(Cloid::new(1)).is_none());
        assert!(t.order_by_id(OrderId::new(10)).is_none());
        assert_eq!(t.position(SYM).qty, dec!(2), "position survives retirement");
    }

    #[test]
    fn the_dedup_window_is_bounded_and_evicts_oldest_first() {
        // Guards two things: the set cannot grow without limit in a long-lived process,
        // and eviction is FIFO so the ids most likely to be replayed (the newest) are
        // the ones still remembered.
        let mut t = OrderTracker::new();
        let n = OrderTracker::MAX_SEEN_TRADES;
        for tid in 0..(n as u64 + 5) {
            feed(
                &mut t,
                ExecEvent::Fill(fill(10, None, Side::Buy, dec!(1), dec!(100), tid)),
            );
        }
        assert_eq!(t.seen_order.len(), n, "window stays bounded");
        assert!(!t.seen_trades.contains(&0), "oldest ids were evicted");
        assert!(t.seen_trades.contains(&(n as u64 + 4)), "newest is kept");

        // An evicted id would be applied again — the documented edge. Assert it so the
        // limitation is visible rather than a surprise.
        let before = t.position(SYM).qty;
        feed(
            &mut t,
            ExecEvent::Fill(fill(10, None, Side::Buy, dec!(1), dec!(100), 0)),
        );
        assert_eq!(
            t.position(SYM).qty,
            before + dec!(1),
            "a replay older than the window is not deduped - known limitation"
        );
    }

    #[test]
    fn market_events_on_the_shared_bus_are_ignored() {
        use axon_core::{Bbo, MarketEvent};
        let mut t = OrderTracker::new();
        let ev = Event::Market(MarketEvent::Bbo(Bbo {
            symbol_id: SYM,
            bid_px: dec!(99),
            bid_sz: dec!(1),
            ask_px: dec!(101),
            ask_sz: dec!(1),
            ts_event: 1,
        }));
        t.on_event(ev.ts_event(), &ev);
        assert_eq!(t.open_count(), 0);
        assert_eq!(t.position(SYM).qty, dec!(0));
    }
}
