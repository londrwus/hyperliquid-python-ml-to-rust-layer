//! Adopt the venue's position and go flat — the operator path that did not exist.
//!
//! ## What went wrong, precisely
//!
//! On 2026-07-27 a live session was cut off mid-position by a venue outage, and getting
//! the account flat again took three attempts and made it worse once. Every step of that
//! was a documented path behaving as documented:
//!
//! 1. **`reconcile` reports position drift and never writes it** — correctly, because a
//!    view this process corrected itself into agreement is a view that can no longer
//!    disagree ([`crate::reconcile`]). A fresh tracker learns its position from the
//!    `userFills` replay instead.
//! 2. **The documented flatten emits a target of zero** onto the signal ring
//!    (`--flatten-only`), and the planner turns a target into an order by subtracting the
//!    **tracked** position. A tracker that has not yet heard the fill replay is flat, so
//!    the delta is zero and the flatten is a no-op — against the one position it was run
//!    for.
//! 3. **So the operator hand-wrote a target.** They read "flat", asked for `+0.01`
//!    against a tracker that in fact knew about a `−0.01` short, the planner correctly
//!    sent one order for `0.02`, and the account went from short `0.01` to **long
//!    `0.01`**.
//! 4. **And then the exit itself was refused.** Post-upgrade Hyperliquid was accepting
//!    post-only orders *exclusively*, so `--flatten-urgency take` — an IOC — was rejected
//!    on every attempt: `Only post-only orders allowed immediately after network upgrade`.
//!
//! Four independent things, one outcome: the exit path failed during the only real
//! incident of the day.
//!
//! ## The three properties this module has instead
//!
//! **Every order is sized from a fresh venue read, and never from an operator's number.**
//! [`FlattenVenue::position`] is consulted before each attempt, so a partial fill shrinks
//! the next order rather than being added to it. There is no arithmetic here an operator
//! can get wrong, because there is no number here an operator supplies.
//!
//! **Every order is a close, and a close cannot flip.** The plan is built from a
//! `FLAG_CLOSE` signal, so the planner sets `reduce_only` and takes the size from the
//! position rather than from the record ([`axon_strategy::Planner`]). The 2026-07-27
//! overshoot is not something this makes unlikely; it makes it unrepresentable.
//!
//! **The urgency is a ladder, not a setting.** The venue that refuses an IOC is exactly
//! the venue an operator cannot wait for, so a refusal escalates *down* the aggression
//! table rather than giving up: an IOC through the far touch first because it leaves no
//! residue resting, and a post-only quote last because it is the only thing a
//! post-only-only venue will take. Each rung is retried, each attempt is recorded with
//! the venue's own words, and the whole thing is bounded.
//!
//! ## What it deliberately does not do
//!
//! **It does not conclude the account is flat.** [`FlattenReport::flat`] is a **venue
//! read taken after the last attempt**, not a claim about what was sent — the same
//! distinction the Phase-6 runbook draws by reading `clearinghouseState` twice. A
//! post-only rung can leave a quote resting that fills a minute later, and this says so
//! rather than waiting for it.
//!
//! **It does not run beside a trading session.** `cancel_all` on Hyperliquid is
//! account-wide, and two sessions on one account sweep each other's orders. This is a
//! cleanup pass for an account nothing else is driving.
//!
//! **It is not the strategy's flatten.** `--flatten-on-exit` on the producer is still the
//! right thing for a session winding down normally: it goes through the same ring, the
//! same reader and the same planner as every other decision, which is what makes a
//! capture of that session replayable. This is what an operator reaches for when that has
//! already failed.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axon_contracts::{Signal, FLAG_CLOSE};
use axon_core::{Decimal, Nanos, SymbolId};
use axon_providers::{ExecutionClient, InstrumentTable, ProviderError};
use axon_strategy::{urgency_rule, NoOrder, PlanContext, Planner, PlannerConfig, Quote};

/// The most aggressive urgency first, then down the table.
///
/// The order is the argument. An IOC through the far touch (`3`) leaves no residue
/// resting, which is what an unmanaged position needs — a half-filled exit that rests is
/// still an unmanaged position, only smaller. A GTC at the far touch (`2`) crosses
/// without the slippage band, which is what a venue that dislikes IOCs may still accept.
/// Post-only (`0`) rests at the near touch and may not fill at all, and it is on the
/// ladder because on 2026-07-27 it was the **only** thing the venue would take.
///
/// `1` (GTC at the near touch) is deliberately absent: it is strictly less likely to fill
/// than `2` and strictly less likely to be accepted than `0`, so it would spend a rung's
/// worth of attempts and a rung's worth of metered requests to be dominated on both axes.
pub const URGENCY_LADDER: [u8; 3] = [3, 2, 0];

/// How hard to try, and how long to let the venue answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlattenPolicy {
    /// Attempts per rung, before moving to a less aggressive one.
    pub attempts_per_rung: u32,
    /// How long to wait after a placement before re-reading the position.
    ///
    /// Load-bearing rather than polite: the next order's size comes from that read, so a
    /// wait shorter than the venue's own settlement makes the second attempt size itself
    /// against a position the first attempt has already reduced. Hyperliquid commits in
    /// roughly 0.2 s and its published p99 is 0.5–0.9 s, so the default is comfortably
    /// past both.
    pub settle: Duration,
    /// How long to leave a **resting** rung's quote before giving up on it. Only the
    /// post-only rung reaches this; an IOC is done the moment it is acked.
    pub rest_for: Duration,
}

impl Default for FlattenPolicy {
    fn default() -> Self {
        Self {
            // Two, not one: a transport error and a venue refusal are indistinguishable
            // from the caller's side, and one of them is worth repeating. Not more,
            // because the rung below is a better answer than the same rung again.
            attempts_per_rung: 2,
            settle: Duration::from_millis(2_000),
            // Long enough for a maker quote at the touch to be taken on a liquid perp,
            // short enough that an operator watching this is not left guessing.
            rest_for: Duration::from_secs(20),
        }
    }
}

/// What the venue says about one instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VenuePosition {
    /// Signed size. Positive is long.
    pub qty: Decimal,
    /// The venue's own entry price, when it reports one. Never invented — see
    /// [`axon_execution::OrderTracker::adopt_position`].
    pub avg_px: Option<Decimal>,
}

/// The venue reads a flatten needs. A trait so every branch below — a refusal on every
/// rung, a partial fill, a position that changes under us — is assertable with no
/// network.
#[async_trait]
pub trait FlattenVenue: Send + Sync {
    /// What we hold in `symbol`, from the venue's own snapshot.
    async fn position(&self, symbol: SymbolId) -> Result<VenuePosition, String>;
    /// `(bid, ask)` to price the exit against.
    async fn top_of_book(&self, symbol: SymbolId) -> Result<(Decimal, Decimal), String>;
}

/// One placement and what came back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenAttempt {
    pub urgency: u8,
    pub qty: Decimal,
    pub price: Decimal,
    /// The venue's own words on a refusal. Kept verbatim: `Only post-only orders allowed
    /// immediately after network upgrade` is a sentence no local classification of
    /// errors would have predicted, and it is the sentence that explains the run.
    pub error: Option<String>,
}

impl FlattenAttempt {
    pub fn accepted(&self) -> bool {
        self.error.is_none()
    }
}

/// What the whole pass did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenReport {
    pub symbol: SymbolId,
    /// The venue's position when the pass started.
    pub started_at: Decimal,
    /// What the tracker believed before it was told. Reported because the **gap** is the
    /// finding: it is the number that made the documented flatten a no-op.
    pub tracked_before: Option<Decimal>,
    pub attempts: Vec<FlattenAttempt>,
    /// The venue's position on a read taken **after** the last attempt. `None` when that
    /// read itself failed, which is not the same as flat and must never be reported as
    /// flat.
    pub ended_at: Option<Decimal>,
    /// Why the pass stopped.
    pub outcome: FlattenOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlattenOutcome {
    /// A venue read after the last attempt says the position is gone.
    Flat,
    /// The venue reported flat before anything was placed.
    AlreadyFlat,
    /// Every rung was tried and something is still held.
    Exhausted,
    /// The planner produced no order and no more aggressive price would change that —
    /// the residue is finer than the instrument's own grid, so no order can express it.
    /// Only a venue-side close can clear this, and it is named rather than retried.
    Unquotable { reason: String },
    /// A venue read failed and the pass will not guess.
    ReadFailed(String),
}

impl FlattenReport {
    /// Whether the account is flat in this symbol **on the venue's own word**.
    pub fn flat(&self) -> bool {
        matches!(
            self.outcome,
            FlattenOutcome::Flat | FlattenOutcome::AlreadyFlat
        )
    }
}

impl std::fmt::Display for FlattenReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "flatten {}: venue held {}", self.symbol, self.started_at)?;
        if let Some(t) = self.tracked_before {
            if t != self.started_at {
                // The whole reason this module exists, on the one line an operator reads.
                write!(f, " (the tracker believed {t})")?;
            }
        }
        write!(f, ", {} attempt(s)", self.attempts.len())?;
        match &self.ended_at {
            Some(q) => write!(f, ", venue now {q}")?,
            None => write!(f, ", venue position UNKNOWN")?,
        }
        write!(f, " - {:?}", self.outcome)
    }
}

/// Adopt the venue's position into `tracker` and drive it to zero.
///
/// `tracker` is written **once**, before anything is placed, so that a session left
/// running behind this pass plans against the position the venue reports rather than the
/// one it inferred. Every order this function sends is sized from its own fresh read and
/// not from the tracker, so the adoption is for everybody else's benefit, not this
/// function's — which is deliberate: a flatten that depended on the write it just made
/// would be a flatten with the same single point of failure as the one it replaces.
#[allow(clippy::too_many_arguments)] // an operator entry point wires many things; a struct only moves the list
pub async fn adopt_and_flatten(
    symbol: SymbolId,
    venue: &dyn FlattenVenue,
    client: &dyn ExecutionClient,
    tracker: Option<&std::sync::RwLock<axon_execution::OrderTracker>>,
    instruments: &Arc<InstrumentTable>,
    planner_cfg: PlannerConfig,
    policy: FlattenPolicy,
    now: Nanos,
) -> FlattenReport {
    let mut report = FlattenReport {
        symbol,
        started_at: Decimal::ZERO,
        tracked_before: None,
        attempts: Vec::new(),
        ended_at: None,
        outcome: FlattenOutcome::AlreadyFlat,
    };

    let start = match venue.position(symbol).await {
        Ok(p) => p,
        Err(e) => {
            report.outcome = FlattenOutcome::ReadFailed(e);
            return report;
        }
    };
    report.started_at = start.qty;
    report.ended_at = Some(start.qty);

    // Told once, loudly. A tracker that disagrees with the venue about a position is the
    // condition that made the documented flatten a no-op, and an operator has to see the
    // correction rather than infer it from a later status line.
    if let Some(t) = tracker {
        if let Ok(mut t) = t.write() {
            let was = t.adopt_position(symbol, start.qty, start.avg_px);
            report.tracked_before = Some(was);
            if was != start.qty {
                eprintln!(
                    "flatten: adopted the venue's position on {symbol}: {was} -> {} \
                     (the tracker had not learned it; this is what makes a target of \
                     zero a no-op)",
                    start.qty
                );
            }
        }
    }

    if start.qty.is_zero() {
        report.outcome = FlattenOutcome::AlreadyFlat;
        return report;
    }

    // One planner for the whole pass, and it is the *production* one: the exit is priced,
    // rounded and flagged by the same code a live session's exit would be, so nothing
    // here can be legal in a way a session's own order is not.
    let planner = Planner::new(planner_cfg);
    report.outcome = FlattenOutcome::Exhausted;
    // Every placement this pass makes must mint a distinct `cloid`, and the only field of
    // the id this pass varies is `seq`. See `close_signal`.
    let mut attempt_no: u32 = 0;

    for urgency in URGENCY_LADDER {
        for _ in 0..policy.attempts_per_rung.max(1) {
            // The authoritative read, every single time. This is the property the
            // 2026-07-27 overshoot came from not having.
            let held = match venue.position(symbol).await {
                Ok(p) => p.qty,
                Err(e) => {
                    report.ended_at = None;
                    report.outcome = FlattenOutcome::ReadFailed(e);
                    return report;
                }
            };
            report.ended_at = Some(held);
            if held.is_zero() {
                report.outcome = FlattenOutcome::Flat;
                return report;
            }

            let (bid, ask) = match venue.top_of_book(symbol).await {
                Ok(q) => q,
                Err(e) => {
                    // A missing book is not a reason to stop trying: the next rung may be
                    // reachable once the feed answers, and the *position* read above
                    // succeeded, so the pass still knows what it is holding.
                    report.attempts.push(FlattenAttempt {
                        urgency,
                        qty: held.abs(),
                        price: Decimal::ZERO,
                        error: Some(format!("no top of book: {e}")),
                    });
                    tokio::time::sleep(policy.settle).await;
                    continue;
                }
            };

            attempt_no += 1;
            let sig = close_signal(symbol, urgency, now, attempt_no);
            let ctx = PlanContext {
                now,
                position: held,
                quote: Some(Quote::new(bid, ask)),
                working: &[],
                precision: instruments.precision(symbol),
            };
            let plan = planner.plan(&sig, &ctx);
            let Some(req) = plan.orders.first() else {
                // The planner has refused, and *why* decides whether trying harder helps.
                // A grid refusal is a property of the instrument and no price fixes it; a
                // price refusal may be fixed by the next rung.
                match plan.no_order {
                    Some(
                        NoOrder::BelowLotSize { .. }
                        | NoOrder::UnknownPrecision { .. }
                        | NoOrder::BelowMinNotional { .. }
                        | NoOrder::BelowMinQty { .. },
                    ) => {
                        report.outcome = FlattenOutcome::Unquotable {
                            reason: format!("{:?}", plan.no_order),
                        };
                        return report;
                    }
                    other => {
                        report.attempts.push(FlattenAttempt {
                            urgency,
                            qty: held.abs(),
                            price: Decimal::ZERO,
                            error: Some(format!("planner produced no order: {other:?}")),
                        });
                        break; // the next rung prices differently
                    }
                }
            };

            let mut attempt = FlattenAttempt {
                urgency,
                qty: req.qty,
                price: req.price.unwrap_or(Decimal::ZERO),
                error: None,
            };
            match client.place_order(req.clone()).await {
                Ok(_) => {
                    eprintln!(
                        "flatten: {:?} {} @ {} urgency {} ({:?}) placed",
                        req.side, req.qty, attempt.price, urgency, req.tif
                    );
                }
                Err(e) => {
                    // Verbatim. `Only post-only orders allowed immediately after network
                    // upgrade` is a sentence no local error taxonomy would have guessed,
                    // and it is the sentence that explains the whole incident.
                    attempt.error = Some(match e {
                        ProviderError::Rejected(m) => m,
                        other => other.to_string(),
                    });
                    eprintln!(
                        "flatten: urgency {urgency} refused: {}",
                        attempt.error.as_deref().unwrap_or("")
                    );
                }
            }
            let refused = !attempt.accepted();
            report.attempts.push(attempt);
            if refused {
                // A refusal is the rung's answer, not the attempt's: the same order at
                // the same urgency will be refused the same way, and each retry costs a
                // metered request the *next* rung needs.
                break;
            }
            // A resting rung has to be given time to fill; a marketable one is decided
            // the moment it is acked.
            let wait = if urgency_rule(urgency).is_marketable() {
                policy.settle
            } else {
                policy.rest_for
            };
            tokio::time::sleep(wait).await;
        }
    }

    // The last word is the venue's, taken after everything: "the strategy asked to be
    // flat" is a request and a flat account is an observation.
    match venue.position(symbol).await {
        Ok(p) => {
            report.ended_at = Some(p.qty);
            if p.qty.is_zero() {
                report.outcome = FlattenOutcome::Flat;
            }
        }
        Err(e) => {
            report.ended_at = None;
            report.outcome = FlattenOutcome::ReadFailed(e);
        }
    }
    report
}

/// A `FLAG_CLOSE` record for `symbol`, on this pass's `attempt`.
///
/// `FLAG_CLOSE` rather than a target of zero, and the difference is the whole safety
/// property: the planner ignores the record's quantity field entirely for a close, takes
/// the size from the position it is given, and sets `reduce_only` — so the order cannot
/// overshoot into the opposite side however wrong anything upstream is about the size.
///
/// **`attempt` goes in the `seq` field and it is not decoration.** `cloid_for` derives the
/// client id from `(ts_event, seq, symbol)` and from nothing else — not the size, not the
/// urgency, not the TIF. This pass holds one `ts_event` for all of its attempts on
/// purpose (it is a single operator action, and a clock read per attempt would make a
/// replay of the pass mint different ids), so without a distinct `seq` every rung and
/// every retry would carry the **same** id: the venue would de-duplicate the repair into
/// nothing while every counter here reported an order placed. That is exactly the failure
/// ADR-0036 records for a naive re-quote, arriving by a different route.
///
/// A `seq` of zero is otherwise harmless here because this never touches the signal ring:
/// nothing reads these records and no `SignalReader` ages them against a baseline.
fn close_signal(symbol: SymbolId, urgency: u8, now: Nanos, attempt: u32) -> Signal {
    Signal::target_position(
        u64::from(attempt),
        now,
        symbol.get(),
        0,
        urgency,
        0,
        0,
        u32::MAX,
        FLAG_CLOSE,
    )
}

/// The Hyperliquid reads, over `/info`.
///
/// Both endpoints are **free** — only signed `/exchange` actions are metered — so a pass
/// that re-reads the position before every attempt costs nothing but latency. That is
/// what makes "size every order from a fresh read" affordable rather than a trade-off.
pub struct HlFlattenVenue {
    pub info_url: String,
    pub account: String,
    pub symbols: axon_provider_hyperliquid::SymbolMap,
}

#[async_trait]
impl FlattenVenue for HlFlattenVenue {
    async fn position(&self, symbol: SymbolId) -> Result<VenuePosition, String> {
        let state = axon_provider_hyperliquid::fetch_user_state(
            &self.info_url,
            &self.account,
            &self.symbols,
        )
        .await
        .map_err(|e| format!("clearinghouseState: {e}"))?;
        // A symbol the venue does not list is flat, and that is the venue's own
        // convention rather than an assumption: `assetPositions` omits flat positions.
        // But a coin it *could not decode* is not flat, it is unknown — so a skipped coin
        // that is the one being flattened has to be an error rather than a zero.
        if state
            .skipped_coins
            .iter()
            .any(|c| self.symbols.id(c) == Some(symbol))
        {
            return Err(format!(
                "the venue reported {symbol} under a coin this map cannot decode"
            ));
        }
        Ok(state
            .positions
            .iter()
            .find(|p| p.symbol_id == symbol)
            .map(|p| VenuePosition {
                qty: p.qty,
                // Kept only when the venue gave one. `Position::avg_px` is zero for a
                // row with no entry price, and adopting a zero average would price an
                // unrealized P&L at the whole notional.
                avg_px: (!p.avg_px.is_zero()).then_some(p.avg_px),
            })
            .unwrap_or(VenuePosition {
                qty: Decimal::ZERO,
                avg_px: None,
            }))
    }

    async fn top_of_book(&self, symbol: SymbolId) -> Result<(Decimal, Decimal), String> {
        let coin = self
            .symbols
            .coin(symbol)
            .ok_or_else(|| format!("no coin for {symbol}"))?;
        let ev =
            axon_provider_hyperliquid::ws::fetch_l2_snapshot(&self.info_url, coin, &self.symbols)
                .await
                .map_err(|e| format!("l2Book: {e}"))?;
        // The REST snapshot decodes to a `Book`; anything else means the decoder changed
        // under us, which is worth an error rather than a guessed price.
        let axon_core::MarketEvent::Book(b) = ev else {
            return Err("l2Book did not decode to a book".into());
        };
        match (b.bids.first(), b.asks.first()) {
            // An empty side is a book we will not price against. The planner would refuse
            // it anyway (`Quote::is_usable`), but saying so here names the cause.
            (Some(bid), Some(ask)) => Ok((bid.px, ask.px)),
            _ => Err("one side of the book is empty".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{OrderStatus, Side, Tif};
    use axon_providers::{
        CancelAck, CancelId, Capabilities, InstrumentSpec, OrderAck, OrderRequest, PriceGrid,
        RateLimitModel, SizeGrid,
    };
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const BTC: SymbolId = SymbolId::new(3);
    const SEC: Nanos = 1_000_000_000;

    /// Millisecond waits rather than paused time: paused time would need tokio's
    /// `test-util` feature in this crate, and a 1 ms sleep exercises the same ordering for
    /// the same reason the real policy waits at all — the next order is sized from a read
    /// taken after it.
    fn fast() -> FlattenPolicy {
        FlattenPolicy {
            attempts_per_rung: 2,
            settle: Duration::from_millis(1),
            rest_for: Duration::from_millis(1),
        }
    }

    fn table() -> Arc<InstrumentTable> {
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: BTC,
            price: PriceGrid::decimals_with_sig_figs(1, 5).unwrap(),
            size: SizeGrid::decimals(5).unwrap(),
            min_notional: Some(dec!(10)),
        });
        Arc::new(t)
    }

    /// One account, shared between the fake venue and the fake client.
    ///
    /// The position moves because an order **filled**, not because a read counter
    /// advanced. That matters for what these tests are able to prove: the property under
    /// test is "each order is sized from a read taken after the previous one settled", and
    /// a script indexed by read count would pass whether or not the code re-read at all.
    #[derive(Default)]
    struct Account {
        qty: Mutex<Decimal>,
        /// What fraction of each **accepted** order fills, in order. A `0.5` is a partial
        /// fill — the case that left $8.50 stranded on 2026-07-27. Runs out into `1`.
        fills: Mutex<std::collections::VecDeque<Decimal>>,
    }

    impl Account {
        fn holding(qty: Decimal) -> Arc<Self> {
            Arc::new(Self {
                qty: Mutex::new(qty),
                fills: Mutex::new(std::collections::VecDeque::new()),
            })
        }
        fn with_fills(qty: Decimal, fills: &[Decimal]) -> Arc<Self> {
            Arc::new(Self {
                qty: Mutex::new(qty),
                fills: Mutex::new(fills.iter().copied().collect()),
            })
        }
        fn qty(&self) -> Decimal {
            *self.qty.lock().unwrap()
        }
        /// Apply an accepted close of `qty` on `side`.
        fn fill(&self, side: Side, qty: Decimal) {
            let fraction = self
                .fills
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Decimal::ONE);
            let filled = qty * fraction;
            let signed = match side {
                Side::Buy => filled,
                Side::Sell => -filled,
            };
            let mut held = self.qty.lock().unwrap();
            *held += signed;
        }
    }

    /// The venue's read side.
    struct ScriptedVenue {
        account: Arc<Account>,
        book: Option<(Decimal, Decimal)>,
        reads: AtomicUsize,
        read_error: Option<String>,
    }

    impl ScriptedVenue {
        fn on(account: Arc<Account>) -> Self {
            Self {
                account,
                book: Some((dec!(65000), dec!(65001))),
                reads: AtomicUsize::new(0),
                read_error: None,
            }
        }
        fn reads(&self) -> usize {
            self.reads.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl FlattenVenue for ScriptedVenue {
        async fn position(&self, _symbol: SymbolId) -> Result<VenuePosition, String> {
            if let Some(e) = &self.read_error {
                return Err(e.clone());
            }
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(VenuePosition {
                qty: self.account.qty(),
                avg_px: Some(dec!(64000)),
            })
        }
        async fn top_of_book(&self, _symbol: SymbolId) -> Result<(Decimal, Decimal), String> {
            self.book.ok_or_else(|| "book unavailable".to_string())
        }
    }

    /// A client that refuses whatever TIFs it is told to, with the venue's own wording,
    /// and moves the shared account when it does not.
    struct PickyClient {
        caps: Capabilities,
        refuse: Vec<Tif>,
        placed: Mutex<Vec<OrderRequest>>,
        account: Arc<Account>,
    }

    impl PickyClient {
        fn new(account: Arc<Account>, refuse: Vec<Tif>) -> Self {
            Self {
                caps: Capabilities {
                    venue: "picky",
                    order_types: &[axon_core::OrderType::Limit],
                    tifs: &[Tif::Gtc, Tif::Ioc, Tif::PostOnly],
                    max_batch: 20,
                    native_market_orders: false,
                    reduce_only: true,
                    rate_limit_model: RateLimitModel::None,
                },
                refuse,
                placed: Mutex::new(Vec::new()),
                account,
            }
        }
        fn placed(&self) -> Vec<OrderRequest> {
            self.placed.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ExecutionClient for PickyClient {
        fn capabilities(&self) -> &Capabilities {
            &self.caps
        }
        async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
            if self.refuse.contains(&req.tif) {
                return Err(ProviderError::Rejected(
                    "order rejected: Only post-only orders allowed immediately after \
                     network upgrade"
                        .into(),
                ));
            }
            self.placed.lock().unwrap().push(req.clone());
            // The account moves because an order filled. A partial fill is scripted on the
            // account, not here, so the fake client has no opinion about it.
            self.account.fill(req.side, req.qty);
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: None,
                status: OrderStatus::Resting,
            })
        }
        async fn place_batch(
            &self,
            _reqs: Vec<OrderRequest>,
        ) -> Result<Vec<OrderAck>, ProviderError> {
            unreachable!("the flatten places one order at a time")
        }
        async fn cancel(&self, _id: CancelId) -> Result<CancelAck, ProviderError> {
            Ok(CancelAck {
                cloid: None,
                order_id: None,
            })
        }
        async fn cancel_all(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn modify(
            &self,
            _id: CancelId,
            _req: OrderRequest,
        ) -> Result<OrderAck, ProviderError> {
            unreachable!()
        }
    }

    async fn run(
        venue: &ScriptedVenue,
        client: &PickyClient,
        tracker: Option<&std::sync::RwLock<axon_execution::OrderTracker>>,
    ) -> FlattenReport {
        adopt_and_flatten(
            BTC,
            venue,
            client,
            tracker,
            &table(),
            PlannerConfig::default(),
            fast(),
            SEC,
        )
        .await
    }

    #[tokio::test]
    async fn a_position_the_tracker_never_learned_is_adopted_and_closed() {
        // The failure this module exists for. `reconcile` reports drift and never writes
        // it, so a fresh tracker is flat; the documented flatten emits a target of zero,
        // the planner subtracts it from a flat position, and the delta is zero. The
        // flatten is a no-op against the one position it was run for.
        let tracker = std::sync::RwLock::new(axon_execution::OrderTracker::new());
        assert!(tracker.read().unwrap().position(BTC).is_flat());

        // Short 0.01 at the venue, closed by the first order.
        let acct = Account::holding(dec!(-0.01));
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![]);
        let report = run(&venue, &client, Some(&tracker)).await;

        assert_eq!(report.started_at, dec!(-0.01));
        assert_eq!(report.tracked_before, Some(Decimal::ZERO), "it was flat");
        assert_eq!(
            tracker.read().unwrap().position(BTC).qty,
            dec!(-0.01),
            "and now it is not: everything else planning behind this pass sees the venue's \
             number"
        );
        assert_eq!(tracker.read().unwrap().adopted_positions(), 1);

        let orders = client.placed();
        assert_eq!(orders.len(), 1, "{orders:?}");
        assert_eq!(orders[0].side, Side::Buy, "buying back a short");
        assert_eq!(orders[0].qty, dec!(0.01));
        assert!(orders[0].reduce_only, "a close, and it cannot overshoot");
        assert!(report.flat(), "{report}");
        assert_eq!(report.outcome, FlattenOutcome::Flat);
    }

    #[tokio::test]
    async fn the_size_comes_from_the_venue_on_every_attempt_so_a_partial_fill_shrinks_it() {
        // The operator error of 2026-07-27, made unrepresentable. A target named once
        // against a stale position sends a delta of the wrong size; here every order is
        // sized from a read taken immediately before it, and the order is reduce-only, so
        // an overshoot cannot be expressed even if the read were wrong.
        //
        // Long 0.01. The first accepted order half-fills, leaving 0.005; the second
        // fills outright. Post-only only, so the ladder falls through to the resting rung
        // and gets its two attempts there — which is exactly the shape a partial fill
        // produces.
        let acct = Account::with_fills(dec!(0.01), &[dec!(0.5), Decimal::ONE]);
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![Tif::Ioc, Tif::Gtc]);
        let report = run(&venue, &client, None).await;

        let sizes: Vec<Decimal> = client.placed().iter().map(|o| o.qty).collect();
        assert_eq!(sizes, vec![dec!(0.01), dec!(0.005)], "{report}");
        assert!(
            client.placed().iter().all(|o| o.reduce_only),
            "every one a close"
        );
        assert!(report.flat(), "{report}");
    }

    #[tokio::test]
    async fn a_venue_that_accepts_only_post_only_orders_is_still_flattened() {
        // Measured on 2026-07-27: post-upgrade Hyperliquid accepted post-only orders
        // exclusively, so `--flatten-urgency take` was refused on every attempt and the
        // documented way to flatten did not work. A single urgency is a single point of
        // failure on the exit path; the ladder is what removes it.
        let acct = Account::holding(dec!(0.02));
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![Tif::Ioc, Tif::Gtc]);
        let report = run(&venue, &client, None).await;

        // The aggressive rungs were tried and refused, once each: a refusal is the
        // rung's answer, and a retry would spend a metered request the next rung needs.
        let refused: Vec<u8> = report
            .attempts
            .iter()
            .filter(|a| !a.accepted())
            .map(|a| a.urgency)
            .collect();
        assert_eq!(refused, vec![3, 2], "{report}");
        assert!(
            report.attempts[0]
                .error
                .as_deref()
                .unwrap()
                .contains("Only post-only orders allowed"),
            "the venue's own words, kept: {report:?}"
        );

        let placed = client.placed();
        assert_eq!(placed.len(), 1, "{placed:?}");
        assert_eq!(placed[0].tif, Tif::PostOnly, "the rung that worked");
        assert!(report.flat(), "{report}");
    }

    #[tokio::test]
    async fn a_venue_that_refuses_everything_reports_the_position_it_could_not_close() {
        // The honest terminal state. Reporting flat here — or reporting nothing — is how
        // an operator walks away from an open position.
        let acct = Account::holding(dec!(0.02));
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![Tif::Ioc, Tif::Gtc, Tif::PostOnly]);
        let report = run(&venue, &client, None).await;

        assert!(!report.flat(), "{report}");
        assert_eq!(report.outcome, FlattenOutcome::Exhausted);
        assert_eq!(report.ended_at, Some(dec!(0.02)));
        assert_eq!(
            report.attempts.len(),
            URGENCY_LADDER.len(),
            "one refusal per rung and no more: {report:?}"
        );
        assert!(client.placed().is_empty());
    }

    #[tokio::test]
    async fn an_already_flat_account_places_nothing() {
        let acct = Account::holding(Decimal::ZERO);
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![]);
        let report = run(&venue, &client, None).await;
        assert_eq!(report.outcome, FlattenOutcome::AlreadyFlat);
        assert!(report.flat());
        assert!(client.placed().is_empty());
        assert_eq!(venue.reads(), 1, "and it does not keep asking");
    }

    #[tokio::test]
    async fn a_failed_venue_read_is_never_reported_as_a_flat_account() {
        // "The strategy asked to be flat" is a request; a flat account is an observation.
        // A pass that cannot make the observation must say so — this is the branch that
        // decides whether an operator goes back and looks.
        let acct = Account::holding(dec!(0.01));
        let mut venue = ScriptedVenue::on(acct.clone());
        venue.read_error = Some("502 Bad Gateway".into());
        let client = PickyClient::new(acct, vec![]);
        let report = run(&venue, &client, None).await;
        assert!(!report.flat(), "{report}");
        assert_eq!(
            report.outcome,
            FlattenOutcome::ReadFailed("502 Bad Gateway".into())
        );
        assert_eq!(report.ended_at, None, "unknown, not zero");
        assert!(report.to_string().contains("UNKNOWN"), "{report}");
        assert!(client.placed().is_empty(), "nothing sized from a guess");
    }

    #[tokio::test]
    async fn a_residue_finer_than_the_lot_is_named_rather_than_laddered_forever() {
        // No price helps with a size the instrument cannot express, so escalating the
        // urgency would spend every rung's requests to be refused identically. Named, and
        // named as the thing it is: only a venue-side close can clear it.
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: BTC,
            price: PriceGrid::decimals_with_sig_figs(1, 5).unwrap(),
            // Lot = one whole coin against a fraction held.
            size: SizeGrid::decimals(0).unwrap(),
            min_notional: Some(dec!(10)),
        });
        let acct = Account::holding(dec!(0.5));
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![]);
        let report = adopt_and_flatten(
            BTC,
            &venue,
            &client,
            None,
            &Arc::new(t),
            PlannerConfig::default(),
            fast(),
            SEC,
        )
        .await;
        assert!(
            matches!(report.outcome, FlattenOutcome::Unquotable { .. }),
            "{report}"
        );
        assert!(client.placed().is_empty());
        assert_eq!(report.attempts.len(), 0, "not one wasted request");
    }

    #[tokio::test]
    async fn a_residue_under_the_venue_minimum_is_still_sent_rather_than_declared_unquotable() {
        // The other side of the line the planner draws, and the case that stranded a
        // position on 2026-07-27: $8.50 of BTC under a $10 venue minimum. It is a close,
        // it is on the grid, and the venue's published rule does not say whether it
        // exempts closes — so it goes, and if the venue refuses we find out loudly
        // instead of holding the position on our own opinion.
        let acct = Account::holding(dec!(0.00013));
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![]);
        let report = run(&venue, &client, None).await;
        let placed = client.placed();
        assert_eq!(placed.len(), 1, "{report}");
        assert_eq!(placed[0].qty, dec!(0.00013));
        assert!(
            placed[0].qty * placed[0].price.unwrap() < dec!(10),
            "under the venue's minimum, and sent anyway"
        );
        assert!(report.flat(), "{report}");
    }

    #[test]
    fn the_ladder_runs_from_the_least_patient_rung_to_the_only_one_a_broken_venue_takes() {
        // The order is the argument, so it is pinned. An IOC first because a half-filled
        // exit that rests is still an unmanaged position; post-only last because on
        // 2026-07-27 it was the only thing the venue would accept.
        assert_eq!(URGENCY_LADDER, [3, 2, 0]);
        assert!(urgency_rule(3).is_marketable());
        assert_eq!(urgency_rule(3).tif, Tif::Ioc);
        assert!(urgency_rule(2).is_marketable());
        assert_eq!(urgency_rule(0).tif, Tif::PostOnly);
        assert!(!urgency_rule(0).is_marketable());
        assert!(
            !URGENCY_LADDER.contains(&1),
            "rung 1 is dominated on both axes: less likely to fill than 2, less likely \
             to be accepted than 0"
        );
    }

    #[test]
    fn a_close_signal_never_carries_a_size_for_anything_to_get_wrong() {
        // `FLAG_CLOSE` rather than a target of zero: the planner ignores the record's
        // quantity entirely, takes the size from the position, and sets `reduce_only`. The
        // overshoot that turned a -0.01 short into a +0.01 long is not made unlikely by
        // this, it is made unrepresentable.
        let s = close_signal(BTC, 3, SEC, 1);
        assert!(s.is_close());
        assert_eq!(s.target_qty, 0);
        assert_eq!(s.symbol_id, BTC.get());
        assert_eq!(s.ts_event, SEC);
    }

    #[test]
    fn every_attempt_in_one_pass_mints_a_distinct_client_id() {
        // `cloid_for` reads `(ts_event, seq, symbol)` and nothing else — not the size, not
        // the urgency, not the TIF. This pass deliberately holds one `ts_event` across all
        // of its attempts, so without a distinct `seq` the second rung would carry the
        // first rung's id and the venue would de-duplicate the repair into nothing while
        // every counter here reported an order placed. ADR-0036 records that exact failure
        // for a naive re-quote; this is the same failure by a different route.
        let a = close_signal(BTC, 3, SEC, 1);
        let b = close_signal(BTC, 2, SEC, 2);
        let same_rung_retry = close_signal(BTC, 3, SEC, 2);
        assert_ne!(axon_strategy::cloid_for(&a), axon_strategy::cloid_for(&b));
        assert_ne!(
            axon_strategy::cloid_for(&a),
            axon_strategy::cloid_for(&same_rung_retry)
        );
        // …and the urgency alone would *not* have been enough, which is why `seq` carries
        // it rather than the urgency field being trusted to.
        assert_eq!(
            axon_strategy::cloid_for(&close_signal(BTC, 3, SEC, 1)),
            axon_strategy::cloid_for(&close_signal(BTC, 0, SEC, 1)),
        );
    }

    #[tokio::test]
    async fn two_placements_in_one_pass_reach_the_venue_under_different_ids() {
        // The end-to-end version of the test above, because the counter has to actually be
        // threaded through: a pass that placed twice under one id would report two orders
        // and leave one position.
        let acct = Account::with_fills(dec!(0.01), &[dec!(0.5), Decimal::ONE]);
        let venue = ScriptedVenue::on(acct.clone());
        let client = PickyClient::new(acct, vec![Tif::Ioc, Tif::Gtc]);
        run(&venue, &client, None).await;
        let ids: Vec<_> = client.placed().iter().map(|o| o.cloid).collect();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1], "the venue would de-duplicate these");
    }
}
