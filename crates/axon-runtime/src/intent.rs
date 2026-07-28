//! The intent source: the join that turns a Python target position into an order at
//! the venue.
//!
//! Until this module existed the two halves of Phase 4 were both finished and neither
//! was connected. `axon-strategy` could validate a `Signal` and plan orders from it;
//! `axon-runtime` could halt, risk-check, rate-limit and sign a submit. Nothing called
//! the second with the output of the first. This is that call:
//!
//! ```text
//!   Python strategy ─▶ signal ring ─▶ SignalReader ─▶ Planner ─▶ Haltable→Guarded→Governed→Exchange
//!   ────────── another process ──────┤────── axon-core thread ──────┤──── tokio edge ────
//! ```
//!
//! **The seam is where the sync/async split already was.** [`SignalReader::drain`] and
//! [`Planner::plan`] are pure and synchronous, so they belong on the deterministic core
//! thread — that is the whole reason a replayed session re-plans to the same orders with
//! the same `cloid`s (ADR-0014). The submit path is async and belongs on the edge. A
//! bounded [`crossbeam_channel`] carries `(cancels, orders)` between them, exactly like
//! the event bus carries market data the other way, and for the same reason: nothing on
//! the core thread ever has a runtime handle in scope to accidentally `await` into.
//!
//! Six things here are decisions rather than plumbing, each with a test named after
//! what it prevents:
//!
//! 0. **The pass schedule is event time, like everything else.** How often a pass runs
//!    decides which signals share one pass — and a signal that shares a pass with a
//!    newer one for the same symbol is `superseded` rather than planned. Paced on a
//!    wall clock, that grouping becomes a property of how fast the machine drained the
//!    bus, and a replay fast enough to be worth running plans *fewer* orders than the
//!    session it is reproducing. The determinism the whole seam was arranged for is
//!    only real if the schedule is a function of the event stream.
//! 1. **One state per pass.** The position, the top of book and the working orders are
//!    all read under one tracker lock at one event time — the event time the reader
//!    also ages the signal against. Two reads at two instants would let a plan be
//!    computed against a position that had already moved.
//! 2. **One intent in flight per symbol.** The core will not plan for a symbol again
//!    until the edge has finished submitting what it was last handed for it. The order
//!    a pass places is not *working* until its ack lands, so a second plan on top of an
//!    unsubmitted one recomputes the same delta and ends up holding twice the target.
//!    Per **symbol**, because nothing in that rule ever needed BTC's round trip to hold
//!    up ETH — a target for a symbol still at the venue is held back rather than
//!    dropped, and re-aged under the reader's own rule when it is finally planned.
//! 3. **Newest target per symbol wins inside a pass**, for the same reason — and it is
//!    free, because a target-position signal is self-contained (ADR-0006) and an older
//!    one carries no information the newer one lacks.
//! 4. **Cancels are submitted before orders.** Hyperliquid processes
//!    `cancel > post-only > GTC > IOC` inside one block, so the pair cannot both be
//!    live; submitting the order first opens a window in which we hold double the
//!    intended exposure (ADR-0014 §6).
//! 5. **A missing ring is a degraded state, not a failure and not silence.** Python may
//!    legitimately start after Rust. The absence is logged once, counted, retried, and
//!    surfaced on the status line — because a session that believes it is trading and
//!    is not is the worst outcome this component has.
//! 6. **The pass sweeps orders the planner can no longer reach** (ADR-0031).
//!    `Planner::outlived` bounds how long a resting order may keep its place, and it
//!    runs *on a signal* — so a strategy that has stopped emitting leaves its last
//!    quote at the venue with the one mechanism that would pull it never running
//!    again. Everything else in this file is driven by a record arriving; the sweeper
//!    is driven by the pass itself, so it advances when nothing arrives. Its cancel is
//!    never risk-gated, for the reason spelled out on
//!    [`sweep_overage_orders`](fn@sweep_overage_orders).

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axon_contracts::Signal;
use axon_core::{Decimal, Nanos, SymbolId};
use axon_execution::{HaltState, HaltSwitch, InFlight, OrderTracker};
use axon_ipc::Consumer;
use axon_providers::{CancelId, ExecutionClient, InstrumentTable};
use axon_strategy::{
    NoOrder, Plan, PlanContext, Planner, Quote, ReplaySource, SignalReader, SignalSource,
    WorkingOrder,
};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::capture::RecordingSource;
use crate::config::IntentConfig;
use crate::handler::CoreHandler;
use crate::health::SessionHealth;
use crate::quote::{top_of_book, TopState};

/// What one accepted signal decided, ready for the venue.
///
/// `plan.cancels` come before `plan.orders` and the submitter must keep that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    pub symbol_id: SymbolId,
    /// The signal's own sequence number, so a venue action can be traced back to the
    /// record that caused it without a side channel.
    pub seq: u64,
    /// The signal's event time. Also the timestamp the tracker records an ack under —
    /// order state stays in event time like everything else.
    pub ts_event: Nanos,
    pub plan: Plan,
}

/// Everything the intent source has counted. The denominator for the operator's one
/// real question: *are we trading on the signals Python thinks it sent?*
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntentStats {
    /// Records that passed every validation check.
    pub accepted: u64,
    /// Records refused, whatever the reason.
    pub rejected: u64,
    /// …of which were past their validity window.
    pub expired: u64,
    /// …and of which walked `seq` **backwards**, which is what a restarted producer
    /// and a replayed ring both look like (ADR-0014 §1).
    ///
    /// Surfaced rather than merely counted because it is the only evidence of a
    /// failure that otherwise presents as a strategy with nothing to say: every
    /// record a restarted Python process writes is refused until its sequence passes
    /// the old baseline, so `accepted` stays flat while the producer is emitting
    /// normally and its own counters climb. "Persist `first_seq`, or restart both
    /// sides together" is an operational requirement nothing enforces — this is the
    /// number that says it was not met.
    pub stale_seq: u64,
    /// Distinct producer loss events, and the records they imply.
    pub gaps: u64,
    pub missing: u64,
    /// Records stamped further ahead of our clock than the reader tolerates.
    pub ahead_of_clock: u64,
    /// Accepted records displaced inside one pass by a newer target for the same
    /// symbol. Not a loss — but a rising count means the strategy is deciding faster
    /// than the venue can be told.
    pub superseded: u64,
    /// Plans that produced at least one order.
    pub planned: u64,
    /// Plans that produced none. Normal: "already at target" is the common case.
    pub no_order: u64,
    /// …of which had no usable top of book. This one *is* a degradation: the
    /// strategy had an opinion and we could not price it.
    pub no_quote: u64,
    /// …and of which were refused by the instrument's own grid — a size finer than the
    /// lot, a notional under the venue minimum, a price the tick cannot represent, a
    /// band the grid is coarser than.
    ///
    /// A climbing count on an otherwise healthy session is a strategy emitting targets
    /// finer than the instrument, not a bug — which is exactly why it must not hide
    /// inside `no_order`, where it would be indistinguishable from "already at target".
    pub precision_refusals: u64,
    /// …and of which were refused because we do not know the instrument's grid at all.
    ///
    /// Not a tuning problem: the session has stopped trading that instrument entirely,
    /// and the counters that would otherwise tell you are the counters that stop rising.
    pub unknown_precision: u64,
    /// Plans lost because the core→edge queue was full.
    pub dropped: u64,
    /// Times a target was held back because an intent for that **symbol** was still at
    /// the venue. Per symbol, not per pass: a slow submit on BTC no longer delays ETH.
    ///
    /// Held, not dropped — the record is carried to the next pass, where a newer target
    /// for the same symbol displaces it exactly as one does inside a pass. Normal in
    /// ones and twos around a venue round trip; a count that climbs while `orders`
    /// stays flat is the submitter having stopped answering.
    pub busy: u64,
    /// Whether that has now lasted longer, in event time, than a signal stays worth
    /// acting on. Past that point nothing on the ring can still become an order, so the
    /// counters above stop rising and the session reads as quiet rather than as stuck.
    pub stalled: bool,
    /// Passes skipped because no market data has arrived yet, so there is neither a
    /// clock to age a signal against nor a book to price it with.
    pub blind: u64,
    /// Passes skipped because the tracker's lock was poisoned — our own order state
    /// is unknown, and planning against it would be planning against a guess.
    pub poisoned: u64,
    /// Resting orders the **sweeper** asked the venue to cancel because they had
    /// outlived `intent.max_order_age_ms` with no signal to supersede them.
    ///
    /// Zero on a healthy session, because a strategy that keeps speaking has every
    /// one of its orders re-derived by the planner long before this age. Non-zero is
    /// therefore always a statement about the *producer*: it stopped, or it never
    /// started, or the ring is detached, or these are orders a previous incarnation
    /// of this process left behind. See [`IntentSource::poll`].
    pub swept: u64,
    /// …of which had already been asked to go at least once, a full lifetime earlier.
    ///
    /// The only evidence available here that the venue is not answering our cancels.
    /// A failed cancel is counted at the edge (`SessionHealth::intent_cancel_failure`),
    /// but a cancel that *succeeds* and never removes the order — a lost reply, a
    /// venue that acked and did nothing — is counted nowhere else, and it is the same
    /// outcome from this side: the exposure is still there and nobody asked it to go.
    pub resweeps: u64,
    /// Orders the pass re-quoted for a target the strategy still holds but has no
    /// working order for — the sweeper's other half (ADR-0031, amended; ADR-0036).
    ///
    /// Non-zero means the composition Phase 6 observed live actually happened: a
    /// resting order was swept, the target that produced it was never re-emitted
    /// (a target position is idempotent, so a strategy correctly says nothing), and
    /// something had to speak for it. Each one is a **fresh price for an old target**,
    /// never a new target and never a larger one.
    pub requotes: u64,
    /// Symbols holding a position, with a target that is not met, no working order,
    /// and no re-quote budget left. **This is the state the live run sat in for
    /// twelve minutes with nothing on the line to say so**, and it is now named.
    ///
    /// Absolute, not cumulative: it is a property of the current pass, and a running
    /// total would go on reporting a hole that had since been filled.
    pub unquoted: u64,
    /// Symbols holding a position the strategy **cannot** close, because the delta to
    /// its target is finer than the instrument's own grid or below the venue's minimum
    /// notional.
    ///
    /// Distinct from [`Self::unquoted`], and the distinction is the whole point: an
    /// unquoted target is one nothing is *working toward* and a re-quote fixes it; a
    /// stranded position is one no order can express, so re-quoting it forever would
    /// place nothing and count nothing. Only an operator can clear it.
    ///
    /// Measured live on 2026-07-27, on the first run of the re-quote: a closing buy for
    /// `0.0003` BTC partially filled `0.00017`, the sweeper pulled the remainder a
    /// minute later, and the `0.00013` left over is **$8.5** — under both the
    /// `min_order_qty` floor and the venue's own $10 minimum. The re-quote was correct
    /// to place nothing and the status line had no word for what was left.
    ///
    /// **That case is no longer stranded, and this counter is narrower than the run
    /// that produced it.** `Planner` now exempts an order that takes a position to
    /// exactly flat from both the `min_order_qty` floor and the venue minimum — a close
    /// is not churn, and refusing one on our own opinion means nobody ever finds out. So
    /// what is left here is the residue below **one lot**, which no limit order can
    /// express at any price and which on a venue whose positions are sums of
    /// lot-sized fills should never arise at all. A non-zero count is therefore either a
    /// mid-session re-precisioning or a venue whose fills are finer than its own grid —
    /// both of which an operator needs to know about, and neither of which a re-quote
    /// can fix.
    ///
    /// Absolute, like `unquoted`, and for the same reason.
    pub stranded: u64,
    /// Failed attempts to open the ring file, summed over every producer.
    pub attach_failures: u64,
    /// Whether **every** declared producer's ring is open right now.
    ///
    /// All of them, not any: with one producer this is the field it always was, and with
    /// several "some ring is attached" is the reading that lets a dead strategy hide
    /// behind a live one.
    pub attached: bool,
    /// How many producers' rings are *not* open. Named on the status line, because with
    /// several producers "detached" without a count is a fault nobody can act on.
    pub detached: u32,
    /// How many producers this session declared. `1` for every config written before
    /// ADR-0038.
    pub producers: u32,
    /// Records refused because another strategy already claims the instrument and
    /// `portfolio.overlap` is `exclusive`.
    ///
    /// A *config* error surfacing at runtime — `RuntimeConfig::validate` refuses a
    /// declared overlap — so a non-zero count means a producer emitted for an instrument
    /// it did not declare, which the counter below also records.
    pub overlap_refused: u64,
    /// Records refused because their producer never declared that instrument.
    ///
    /// Not pedantry: the declared universe is what makes an overlap catchable at
    /// startup, so a producer that speaks outside it has escaped the check that keeps two
    /// strategies off one position.
    pub out_of_scope: u64,
    /// Symbols planned from a sum of two or more strategies' claims.
    pub netted: u64,
    /// Contributions kept even though their author had gone silent past its window.
    /// Exposure nobody is currently speaking for, which must never be invisible.
    pub silent_held: u64,
    /// Contributions driven to zero because their author went silent and the operator
    /// asked for that (`on_silence = "flat"`).
    pub silent_flat: u64,
    /// Netted targets that dropped a `price_band` because the contributors disagreed
    /// about it. See [`axon_strategy::TargetBook::net`].
    pub band_dropped: u64,
    /// Passes on which a producer's own claims were scaled down to its declared
    /// allocation.
    pub strategy_scaled: u64,
    /// Passes on which every target was scaled down to fit the portfolio bound.
    pub portfolio_scaled: u64,
    /// The factor the last such pass applied, in basis points (`10000` = unscaled).
    ///
    /// A number rather than a flag because "the portfolio is binding" and "the portfolio
    /// is binding at 3 % of what the strategies asked for" are different sessions, and
    /// only the second one is a misconfiguration.
    pub portfolio_scale_bps: u32,
    /// Instruments a breadth cap kept the session out of this pass.
    ///
    /// Absolute, like `unquoted`: it describes the current pass, and a running total
    /// would go on reporting a cap that has since stopped binding.
    pub breadth_denied: u32,
    /// Passes on which an allocation could not be computed because some claimed
    /// instrument had no usable mark.
    ///
    /// The scale is then **not applied** — an allocation derived from a partial book is
    /// a smaller, entirely plausible number, and scaling on it would quietly shrink
    /// every position because one feed went quiet. The `GuardedClient` bound still holds
    /// and fails closed, so the session refuses new exposure rather than mis-sizing it.
    pub alloc_unpriced: u64,
}

// ── where records come from ──────────────────────────────────────────────────

/// A [`SignalSource`] that may not be there yet.
///
/// Separate from `SignalSource` because attaching is the *owner's* problem, not the
/// validator's: the reader's job is to refuse records, and a file that does not exist
/// has no records to refuse.
pub trait Attachable {
    /// Make the source readable if it is not, pacing retries against `now_ms` (wall
    /// clock; this decides how often we `open(2)`, never what an order is worth).
    /// Returns whether a record can be read now.
    fn ensure(&mut self, now_ms: u64) -> bool;

    fn is_attached(&self) -> bool {
        true
    }

    fn attach_failures(&self) -> u64 {
        0
    }

    /// The core's event clock, handed over before each drain.
    ///
    /// A no-op for a source that only hands records out, and the whole mechanism for the
    /// two that care *when* a record crossed the boundary: [`RecordingSource`] stamps it
    /// as the record's `release_ts`, and [`CapturedSignals`] withholds a record until
    /// the clock has reached it. The ring carries only the moment the strategy decided,
    /// so without this the gap that makes a signal stale would exist nowhere and a
    /// replay could never reproduce an expiry.
    ///
    /// Defaulted rather than required so that adding it did not force every existing
    /// source — including the harness feeds outside this crate — to write an empty body.
    ///
    /// [`RecordingSource`]: crate::capture::RecordingSource
    /// [`CapturedSignals`]: crate::capture::CapturedSignals
    fn observe_event_time(&mut self, _now: Nanos) {}
}

impl Attachable for ReplaySource {
    /// A canned source is always there — it *is* the records.
    fn ensure(&mut self, _now_ms: u64) -> bool {
        true
    }
}

/// The production source: the shared-memory ring, opened lazily and re-opened after
/// it goes away.
///
/// A missing ring file is the ordinary startup race (Python has not started yet), not
/// an error worth refusing to run over. It is also the one degradation that must never
/// be silent, so the first failure is logged, every failure is counted, and the status
/// line says `SIGNAL RING DETACHED` for as long as it lasts.
#[derive(Debug)]
pub struct LazyRing {
    path: PathBuf,
    consumer: Option<Consumer>,
    retry_ms: u64,
    next_try_ms: u64,
    failures: u64,
    /// Whether the current outage has already been reported. Cleared on a successful
    /// attach, so a ring that flaps logs each time it goes — but a ring that is simply
    /// not there yet logs once, not once per second forever.
    reported: bool,
}

impl LazyRing {
    pub fn new(path: impl AsRef<Path>, retry_ms: u64) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            consumer: None,
            retry_ms: retry_ms.max(1),
            next_try_ms: 0,
            failures: 0,
            reported: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl SignalSource for LazyRing {
    #[inline]
    fn next_signal(&mut self) -> Option<Signal> {
        self.consumer.as_mut()?.next_signal()
    }
}

impl Attachable for LazyRing {
    fn ensure(&mut self, now_ms: u64) -> bool {
        if self.consumer.is_some() {
            return true;
        }
        if now_ms < self.next_try_ms {
            return false;
        }
        self.next_try_ms = now_ms.saturating_add(self.retry_ms);
        match Consumer::open(&self.path) {
            Ok(c) => {
                self.consumer = Some(c);
                self.reported = false;
                println!("intent: attached to signal ring {}", self.path.display());
                true
            }
            Err(e) => {
                self.failures += 1;
                if !self.reported {
                    self.reported = true;
                    eprintln!(
                        "intent: signal ring {} unavailable ({e}) - the session keeps \
                         running with no intent source and retries every {} ms",
                        self.path.display(),
                        self.retry_ms
                    );
                }
                false
            }
        }
    }

    fn is_attached(&self) -> bool {
        self.consumer.is_some()
    }

    fn attach_failures(&self) -> u64 {
        self.failures
    }
}

// ── the core→edge handoff ────────────────────────────────────────────────────

/// The bounded queue between the deterministic core and the async submitter.
#[derive(Debug)]
pub struct IntentQueue {
    tx: Sender<Intent>,
    rx: Receiver<Intent>,
    /// Which **symbols** have an intent handed over but not yet finished at the venue.
    /// The core reads this to decide whether it may plan for a symbol again; nothing
    /// else depends on it.
    inflight: Arc<InFlight>,
}

impl IntentQueue {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = bounded(capacity.max(1));
        Self {
            tx,
            rx,
            inflight: Arc::new(InFlight::new()),
        }
    }

    /// The core's end.
    pub fn sink(&self) -> IntentSink {
        IntentSink::Queue {
            tx: self.tx.clone(),
            inflight: self.inflight.clone(),
        }
    }

    /// The edge's end.
    pub fn receiver(&self) -> Receiver<Intent> {
        self.rx.clone()
    }

    pub fn inflight(&self) -> Arc<InFlight> {
        self.inflight.clone()
    }
}

/// Where a finished plan goes.
#[derive(Debug)]
pub enum IntentSink {
    /// Live: over the queue to the submit pipeline.
    Queue {
        tx: Sender<Intent>,
        inflight: Arc<InFlight>,
    },
    /// Offline: kept, so `cargo run --bin axon` can assert what the join produced
    /// without a venue, a socket, or a tokio runtime in the process.
    Record(Vec<Intent>),
}

impl IntentSink {
    /// Hand off one plan. `false` means it was dropped, which is only ever the
    /// queue being full.
    fn send(&mut self, intent: Intent) -> bool {
        match self {
            IntentSink::Queue { tx, inflight } => {
                // Claimed *before* the send, or the edge could finish and release
                // before the core had claimed, and the gate would let a second intent
                // through on top of the first.
                let symbol = intent.symbol_id;
                inflight.claim(symbol);
                match tx.try_send(intent) {
                    Ok(()) => true,
                    Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                        // Released, or this symbol is gated for the rest of the session
                        // over an intent nobody will ever submit — which reads on the
                        // status line exactly like a strategy with no opinion about it.
                        inflight.release(symbol);
                        false
                    }
                }
            }
            // Recording *is* completion, so nothing is ever in flight offline.
            IntentSink::Record(out) => {
                out.push(intent);
                true
            }
        }
    }

    /// Whether an intent for `symbol` is still at the venue.
    fn inflight(&self, symbol: SymbolId) -> bool {
        match self {
            IntentSink::Queue { inflight, .. } => inflight.contains(symbol),
            IntentSink::Record(_) => false,
        }
    }

    /// `(anything outstanding, releases so far)` — the two numbers the stall watcher
    /// needs, and it needs both. Occupancy alone describes a healthy multi-symbol
    /// session just as well as a wedged one; what a wedge alone produces is occupancy
    /// with a completion count that has stopped moving.
    fn progress(&self) -> (bool, u64) {
        match self {
            IntentSink::Queue { inflight, .. } => (!inflight.is_empty(), inflight.completed()),
            IntentSink::Record(_) => (false, 0),
        }
    }
}

// ── the source itself ────────────────────────────────────────────────────────

/// What the core loop needs from an intent source, object-safe so the loop does not
/// have to be generic over where signals come from.
pub trait IntentPoll {
    /// Run one pass at event time `now`. `now_ms` is wall clock and is used only to
    /// pace ring re-open attempts.
    fn poll(&mut self, now: Nanos, now_ms: u64, handler: &CoreHandler);

    fn stats(&self) -> IntentStats;

    /// One line per declared producer, for the status line.
    ///
    /// Separate from [`Self::stats`] because `IntentStats` is `Copy` and this is not:
    /// a name is a `String`, and putting one in the summary struct would make every
    /// status assembly allocate whether or not anybody had two strategies. Defaulted to
    /// empty so a source with no notion of producers — the replay harness — says
    /// nothing rather than inventing one.
    fn strategies(&self) -> Vec<StrategyLine> {
        Vec::new()
    }
}

/// What one producer looks like on the status line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyLine {
    pub name: String,
    /// Whether its ring is open right now.
    pub attached: bool,
    pub accepted: u64,
    pub rejected: u64,
    pub expired: u64,
    pub stale_seq: u64,
    /// Instruments it currently claims.
    pub claims: usize,
    /// Milliseconds of **event time** since it last stated anything, or `None` if it
    /// never has. Event time, like the silence rule it reports on: a producer that has
    /// gone quiet on a session whose market data has also stopped is a different fault,
    /// and it has its own detectors.
    pub silent_ms: Option<u64>,
    /// Whether that silence is past this producer's declared window, so its claims are
    /// either being held without an author or driven to zero.
    pub silent: bool,
    /// Records it emitted for an instrument it never declared.
    pub out_of_scope: u64,
}

/// One signal producer: a validating reader over one ring, and the policy attached to it.
struct Producer<S> {
    id: axon_strategy::StrategyId,
    name: String,
    reader: SignalReader<S>,
    /// The instruments this producer may speak for. **Empty means every one of them** —
    /// which is what a single-producer session declares, and what makes that session the
    /// same code path as a declared one rather than a special case.
    symbols: Vec<u32>,
    /// The most gross notional this producer's own claims may add up to. `0` is
    /// unbounded. An allocation, not a gate: a claim past it is scaled, never refused.
    max_gross_notional: Decimal,
    /// Whether its ring was open at the last pass.
    attached: bool,
    out_of_scope: u64,
}

impl<S> Producer<S> {
    /// Whether this producer is allowed to speak for `symbol`.
    fn owns(&self, symbol: u32) -> bool {
        self.symbols.is_empty() || self.symbols.contains(&symbol)
    }
}

/// One record in a pass, and which producer put it there.
#[derive(Debug, Clone, Copy)]
struct PassRecord {
    strategy: axon_strategy::StrategyId,
    signal: Signal,
}

/// One producer as the caller supplies it: a source, and everything the operator
/// declared about it.
///
/// A struct rather than five positional arguments per producer, for the reason
/// `CoreControl` gives: a `Decimal` and a `Vec<u32>` in the wrong slots still compile.
pub struct NamedSource<S> {
    pub name: String,
    pub source: S,
    /// The instruments this producer may speak for; empty is the session universe.
    pub symbols: Vec<u32>,
    pub policy: axon_strategy::StrategyPolicy,
    pub max_gross_notional: Decimal,
}

/// How much of what the strategies asked for the session is allowed to work toward.
///
/// **Two stages, and they answer different questions.** A per-strategy factor bounds one
/// producer's own book — its allocation — so a strategy that has decided to be enormous
/// crowds out the others' share rather than the account's. A portfolio factor then bounds
/// the sum of what is left. Applying only the second would let one runaway producer eat
/// the whole budget proportionally; applying only the first would let three well-behaved
/// producers add up to more than the account may hold.
///
/// **Scaling, not refusing.** The `GuardedClient` bound is the guarantee and it refuses;
/// this makes the target the session works toward *reachable*, so a portfolio that binds
/// converges to the largest book it is allowed to hold instead of emitting a target that
/// is refused on every pass. Without it a binding bound presents as orders that keep
/// failing rather than as a limit that is working.
#[derive(Debug, Clone, Default)]
struct Allocation {
    /// Indexed by [`axon_strategy::StrategyId::get`]. `1` means unscaled.
    per_strategy: Vec<Decimal>,
    /// `None` means unscaled — which is not the same as `Some(1)`, because
    /// [`axon_strategy::scale_fixed`] returns the producer's own bytes untouched only
    /// when no multiply happens at all.
    portfolio: Option<Decimal>,
    /// Instruments a breadth cap will not let the session open this pass. Their target
    /// is driven to zero, which from flat is simply no order — the claim stays in the
    /// book and opens when room appears.
    denied: Vec<u32>,
}

impl Allocation {
    /// The adjustment one claim gets. Identity when nothing binds, and identity is
    /// load-bearing: it is what keeps an unallocated session putting exactly the bytes
    /// its producer wrote on the wire.
    fn apply(&self, symbol: u32, strategy: axon_strategy::StrategyId, qty: i64) -> i64 {
        if self.denied.contains(&symbol) {
            return 0;
        }
        let s = self
            .per_strategy
            .get(strategy.get() as usize)
            .copied()
            .unwrap_or(Decimal::ONE);
        let k = match self.portfolio {
            Some(p) => s * p,
            None => s,
        };
        axon_strategy::scale_fixed(qty, k)
    }
}

/// **Superseded by [`axon_strategy::TargetBook`]**, which holds the same thing per
/// *(strategy, symbol)* rather than per symbol and therefore also answers the netting
/// question. The doc comment is kept here in full because it is the argument for why a
/// held target exists at all, and that argument did not change when the type did.
///
/// A target the strategy stated and has not withdrawn, kept so the pass can speak for
/// it after the sweeper has pulled its quote.
///
/// **Why the runtime holds this at all.** A target position is idempotent, so a
/// strategy that has not changed its mind correctly says nothing (ADR-0006). The
/// sweeper cancels a resting order that no signal spoke for (ADR-0031). Compose the
/// two and a held target's quote is cancelled and never re-quoted — measured live on
/// 2026-07-26, where a short sat open with **no working order for about twelve
/// minutes**. Each rule is right; the composition is not, and the missing piece is
/// that nothing in the system was holding the target the position was working toward.
///
/// **Why a *bounded* re-quote and not an unbounded one.** The sweeper's subject is a
/// producer that has stopped, and a planner that re-quotes forever would hand a dead
/// strategy an immortal quote — undoing exactly the protection the sweeper is. The
/// budget makes the failure terminate: after `max_requotes` the session stops placing
/// and starts *saying* that a position is unquoted, which is the one thing the live
/// run could not do.
///
/// **Why not the two alternatives.** Teaching the sweeper to spare an order whose
/// symbol is not yet at target would gut it — a quote for an unreached target is
/// precisely the quote that goes stale. Requiring producers to re-assert on a timer
/// makes every strategy responsible for a runtime invariant, and the first one written
/// by somebody who had not read this comment would sit unquoted again.
/// Drains every producer's ring, nets what they asked for, and hands the result to the
/// submit pipeline.
pub struct IntentSource<S> {
    /// One per declared producer. A session with no `[[strategy.producer]]` tables has
    /// exactly one entry here, built from `[ipc]` — the same type, so there is no branch
    /// anywhere below between one strategy and several. The single-producer path is the
    /// one that has run at a venue; a branch would mean the tested path is not the
    /// shipped one.
    producers: Vec<Producer<S>>,
    /// What every producer has asked for, and what those claims add up to per instrument
    /// (ADR-0038). Replaces the per-symbol `HeldTarget` list, which could only ever
    /// describe one author.
    book: axon_strategy::TargetBook,
    /// Bounds across every instrument at once. Inert until an operator declares one.
    portfolio: axon_risk::PortfolioLimits,
    /// Whether the pass scales its targets to fit those bounds, or leaves the guard to
    /// refuse the orders. See [`Allocation`].
    scale_to_fit: bool,
    planner: Planner,
    sink: IntentSink,
    halt: Arc<HaltSwitch>,
    /// Each instrument's tick and lot (ADR-0025).
    ///
    /// The **same** `Arc` the venue client holds. Two tables that can drift apart is a
    /// planner rounding to one grid while the encoder refuses against another, and
    /// every refusal then reads in the log exactly like a venue rejection.
    instruments: Arc<InstrumentTable>,
    max_per_drain: usize,
    /// How much *event time* must pass between two drains.
    interval_ns: Nanos,
    /// How long an unfinished batch may block passes before the path is called stalled.
    ///
    /// The reader's own ceiling, reused rather than reinvented: once a batch has been
    /// in flight longer than a signal is allowed to be old, every record that arrived
    /// while it was stuck would be refused as expired anyway. Two numbers here would be
    /// two answers to "how long is too long".
    stall_after_ns: Nanos,
    /// The window past which a top of book stops being something we will price an
    /// order against. Deliberately the *same* number the mark cache expires prices at:
    /// an instrument the risk gate has already declared too stale to size a position
    /// in is not one the planner should be quoting into either, and two windows would
    /// mean two answers to one question.
    quote_max_age_ns: Nanos,
    /// The operator's ceiling on a resting order's age, or `None` when they set no
    /// bound — in which case there is nothing for the sweeper to enforce and it does
    /// not run. The same number, and the same reading of zero, as
    /// [`axon_strategy::PlannerConfig::max_order_age_ms`]: the planner applies the
    /// shorter of it and the signal's own request, and the sweeper can only apply this
    /// one, because its whole subject is a producer that has stopped making requests.
    order_lifetime_ns: Option<Nanos>,
    /// How much *event time* must pass between two sweeps.
    sweep_interval_ns: Nanos,
    /// Event time of the last pass. See [`IntentSource::poll`] for why this is not an
    /// `Instant`.
    last_pass_ns: Option<Nanos>,
    /// Event time of the last sweep, on the same clock and for the same reason as
    /// [`Self::last_pass_ns`].
    last_sweep_ns: Option<Nanos>,
    /// Event time at which the batch currently in flight first blocked a pass, cleared
    /// the moment one runs. The stall bound is measured from here.
    blocked_since_ns: Option<Nanos>,
    stats: IntentStats,
    /// Newest accepted signal per **(producer, symbol)** in this pass. Reused across
    /// passes so a pass allocates nothing beyond what the planner itself does.
    ///
    /// Per producer as well as per symbol, because two strategies' targets for one
    /// instrument are two facts that add — collapsing them here would be the silent
    /// overwrite `axon_strategy::book` exists to prevent.
    pass: Vec<PassRecord>,
    /// Targets held back because their symbol still had an intent at the venue, at most
    /// one per (producer, symbol). They are put back into `pass` at the start of the next
    /// one, *before* the drain, so a newer target for the same pair displaces them
    /// exactly as it would inside a pass — and re-aged through
    /// [`SignalReader::still_fresh`], on their **own producer's** reader, before they can
    /// become an order, so a record cannot expire in here unseen.
    carried: Vec<PassRecord>,
    /// Instruments this pass has a fresh claim for and will plan once each. Distinct
    /// from `pass`, which has one entry per contributing producer.
    touched: Vec<u32>,
    working: Vec<WorkingOrder>,
    /// Every order the sweeper has asked the venue to cancel, and the event time it
    /// last asked at. Pruned to what is still open on every sweep, so it is bounded by
    /// the open-order count rather than by the session's length.
    ///
    /// This is what makes the sweeper *level*-triggered without making it a shout: the
    /// trigger is a property of tracker state, so an order the venue did not remove is
    /// still over-age on the next sweep and would be re-cancelled every second
    /// forever. Signed `/exchange` actions are metered (~10 203 a day), so a cancel a
    /// second against a venue that is not answering would spend the day's budget on
    /// one stuck order and leave nothing for the orders that could still be pulled.
    swept_at: Vec<(axon_core::Cloid, Nanos)>,
    /// Scratch for one sweep, reused so that the scan — which runs whether or not
    /// anything is over age — allocates nothing.
    sweeping: Vec<(SymbolId, axon_core::Cloid)>,
    /// Releases the edge had made when the last pass looked. See [`IntentSink::progress`].
    last_completed: u64,
    /// How many times one target may be re-quoted before the session gives up and
    /// says so. `0` disables the re-quote entirely, which is the behaviour every
    /// session before ADR-0036 had.
    requote_limit: u32,
    /// Where this pass records how old the decisions it acted on were (ADR-0036).
    ///
    /// Not an `Option`: [`LatencyBook::undeclared`] measures and declares nothing, so a
    /// caller that never wires a budget still gets the distribution, and the code path
    /// a live session takes is the one every test takes.
    latency: Arc<crate::latency::LatencyBook>,
}

/// The production instantiation: validation over the shared-memory ring, with the
/// session recording teed off it.
///
/// The recording wrapper is in the type whether or not a session is capturing, because
/// [`RecordingSource`] with no tap is a transparent forward — one wiring rather than two
/// means the shape a live session runs is the shape the tests run.
pub type RingIntentSource = IntentSource<RecordingSource<LazyRing>>;

impl<S: SignalSource + Attachable> IntentSource<S> {
    /// The single-producer source: one ring, no netting question to answer.
    ///
    /// Kept as the primary constructor because it is what every session that has ever
    /// run uses, and because it builds a one-element producer list rather than a special
    /// case — so the multi-producer machinery below is exercised by every existing test
    /// at width one.
    pub fn new(
        source: S,
        cfg: &IntentConfig,
        instruments: Arc<InstrumentTable>,
        quote_max_age_ns: Nanos,
        halt: Arc<HaltSwitch>,
        sink: IntentSink,
    ) -> Self {
        Self::multi(
            vec![NamedSource {
                name: "strategy".to_string(),
                source,
                symbols: Vec::new(),
                policy: axon_strategy::StrategyPolicy::new(axon_strategy::StrategyId::new(0)),
                max_gross_notional: Decimal::ZERO,
            }],
            cfg,
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Exclusive,
            true,
            instruments,
            quote_max_age_ns,
            halt,
            sink,
        )
    }

    /// Several producers, one account.
    ///
    /// `sources` is the resolved producer list — `RuntimeConfig::producers()` is what
    /// builds it, including the single-producer case, so the runtime has no branch for
    /// session width.
    #[allow(clippy::too_many_arguments)]
    pub fn multi(
        sources: Vec<NamedSource<S>>,
        cfg: &IntentConfig,
        portfolio: axon_risk::PortfolioLimits,
        overlap: axon_strategy::Overlap,
        scale_to_fit: bool,
        instruments: Arc<InstrumentTable>,
        quote_max_age_ns: Nanos,
        halt: Arc<HaltSwitch>,
        sink: IntentSink,
    ) -> Self {
        let policies: Vec<axon_strategy::StrategyPolicy> =
            sources.iter().map(|s| s.policy).collect();
        let producers: Vec<Producer<S>> = sources
            .into_iter()
            .map(|s| Producer {
                id: s.policy.id,
                name: s.name,
                reader: SignalReader::with_config(s.source, cfg.reader_config()),
                symbols: s.symbols,
                max_gross_notional: s.max_gross_notional,
                attached: false,
                out_of_scope: 0,
            })
            .collect();
        Self {
            producers,
            book: axon_strategy::TargetBook::new(policies, overlap),
            portfolio,
            scale_to_fit,
            planner: Planner::new(cfg.planner_config()),
            sink,
            halt,
            instruments,
            max_per_drain: cfg.max_per_drain.max(1),
            interval_ns: ms_to_ns(cfg.drain_interval_ms.max(1)),
            stall_after_ns: ms_to_ns(cfg.max_signal_age_ms as u64),
            quote_max_age_ns,
            // Zero is "I set no bound", never "already expired" — the same reading the
            // planner turns on, and filtering it out here rather than comparing against
            // it makes the inverted one unrepresentable rather than merely avoided.
            order_lifetime_ns: (cfg.max_order_age_ms != 0)
                .then(|| ms_to_ns(cfg.max_order_age_ms as u64)),
            sweep_interval_ns: ms_to_ns(cfg.sweep_interval_ms.max(1)),
            last_pass_ns: None,
            last_sweep_ns: None,
            blocked_since_ns: None,
            stats: IntentStats::default(),
            pass: Vec::with_capacity(cfg.max_per_drain.min(64)),
            carried: Vec::new(),
            touched: Vec::new(),
            working: Vec::new(),
            swept_at: Vec::new(),
            sweeping: Vec::new(),
            last_completed: 0,
            requote_limit: cfg.max_requotes,
            latency: Arc::new(crate::latency::LatencyBook::undeclared()),
        }
    }

    /// Record signal ages into the session's own book instead of a private one.
    ///
    /// A builder rather than a constructor argument because every existing caller —
    /// including every test in this file — wants the default, and a tenth positional
    /// argument on `new` is how a `bool` ends up in the wrong slot.
    pub fn with_latency(mut self, book: Arc<crate::latency::LatencyBook>) -> Self {
        self.latency = book;
        self
    }

    /// Everything an offline run planned, taken out of the recording sink.
    pub fn take_recorded(&mut self) -> Vec<Intent> {
        match &mut self.sink {
            IntentSink::Record(out) => std::mem::take(out),
            IntentSink::Queue { .. } => Vec::new(),
        }
    }
}

impl<S: SignalSource + Attachable> IntentPoll for IntentSource<S> {
    fn poll(&mut self, now: Nanos, now_ms: u64, handler: &CoreHandler) {
        // Split the borrows up front: one pass touches the producers, the book, the
        // planner, the sink and four buffers, and they are all fields of the same struct.
        let Self {
            producers,
            book,
            portfolio,
            scale_to_fit,
            planner,
            sink,
            halt,
            instruments,
            max_per_drain,
            interval_ns,
            stall_after_ns,
            quote_max_age_ns,
            order_lifetime_ns,
            sweep_interval_ns,
            last_pass_ns,
            last_sweep_ns,
            blocked_since_ns,
            stats,
            pass,
            carried,
            touched,
            working,
            swept_at,
            sweeping,
            last_completed,
            requote_limit,
            latency,
        } = self;

        // The pass schedule is a function of the event stream and of nothing else.
        // Pacing on `Instant::elapsed` instead makes *which* signals land in one pass a
        // property of how fast the machine ran: replay a capture through any driver that
        // feeds events faster than `drain_interval_ms` of wall time per event and two
        // live passes collapse into one, the older target is counted `superseded`
        // instead of planned, and the parity harness reports a divergence that is an
        // artefact of replay speed. It is also why the only replay that reproduced a
        // live session had to `sleep` a millisecond per event to do it.
        //
        // A late arrival walks `now` backwards, and the subtraction is signed, so it
        // cannot trigger a pass — the same high-water reasoning `RecordingSource` uses
        // for `release_ts`, and for the same reason: a record's schedule has to be
        // reproducible from the log.
        if let Some(prev) = *last_pass_ns {
            if now.saturating_sub(prev) < *interval_ns {
                return;
            }
        }
        *last_pass_ns = Some(now);

        // A stopped session will not trade again, so consuming records would only
        // discard them; a *halted* one keeps draining, because its cancels still have
        // to reach the venue and pulling a stale quote is exactly what a halt is for.
        if halt.state() == HaltState::Stopped {
            return;
        }

        // Every producer is attached independently, and the fleet is reported rather
        // than "a ring". `attached` is **all** of them, not any: with several producers
        // "something is attached" is the reading that lets a dead strategy hide behind a
        // live one, which is the exact failure this counter was added to prevent one
        // producer at a time.
        let mut attach_failures = 0u64;
        let mut detached = 0u32;
        for p in producers.iter_mut() {
            p.attached = p.reader.source_mut().ensure(now_ms);
            if !p.attached {
                detached += 1;
            }
            attach_failures += p.reader.source().attach_failures();
        }
        stats.producers = producers.len() as u32;
        stats.detached = detached;
        stats.attached = detached == 0;
        stats.attach_failures = attach_failures;

        // No market data yet means no clock to age a signal against, no book to price
        // it with, and no clock to age a resting order on either. Draining now would
        // advance the reader's `seq` baseline and silently throw away the strategy's
        // first targets; leaving them on the ring costs nothing, because the pass that
        // can use them is the first one after an event arrives — and until one does,
        // there is nothing here that could trade.
        if now == 0 {
            stats.blind += 1;
            return;
        }

        // A detached ring no longer ends the pass. It used to, and that was the exact
        // hole the sweeper exists to close: "Python is not there" is the strongest
        // possible statement that no signal is coming, and it was the one condition
        // under which nothing downstream ran — including the only thing that could
        // pull the quotes the last live producer left resting.

        // A wedged submitter is the failure here that hides behind counters which
        // *stop*: the symbols it is stuck on produce no orders, and on a
        // single-instrument session that is the whole status line. It is judged on
        // progress rather than on occupancy, because a session trading several
        // instruments legitimately has something outstanding almost all the time —
        // "in flight for two seconds" describes that session as well as a wedged one,
        // and "nothing has completed for two seconds" describes only the wedge.
        // Measured in event time, so a replay of the capture reaches the same verdict.
        let (outstanding, completed) = sink.progress();
        if completed != *last_completed {
            *last_completed = completed;
            *blocked_since_ns = None;
        }
        if outstanding {
            let since = *blocked_since_ns.get_or_insert(now);
            stats.stalled = now.saturating_sub(since) > *stall_after_ns;
        } else {
            *blocked_since_ns = None;
            stats.stalled = false;
        }

        pass.clear();
        // Whatever the last pass could not act on goes in first, so a record read below
        // for the same (producer, symbol) still displaces it: newest target wins, whether
        // the older one arrived in this pass or was held over from the last.
        //
        // Re-aged on the way in, through **its own producer's** reader and into that
        // reader's `expired` counter. A held record is a decision about a market that
        // kept moving while we could not act on it, and one that goes stale in here has
        // to be as visible as one that went stale on the ring — otherwise this really
        // would be the second queue that ages signals where nobody can watch them. Its
        // own reader, because each producer has its own sequence and its own counters,
        // and a record aged against somebody else's would be counted against a strategy
        // that never wrote it.
        for held in carried.drain(..) {
            if let Some(p) = producers.iter_mut().find(|p| p.id == held.strategy) {
                if p.reader.still_fresh(&held.signal, now) {
                    pass.push(held);
                }
            }
        }
        let mut superseded = 0u64;
        for p in producers.iter_mut() {
            if !p.attached {
                continue;
            }
            // The event time every record read in this pass becomes visible at. Handed
            // over before the drain, because a record's `release_ts` is a property of
            // the moment it was read and the reader is about to read some.
            p.reader.source_mut().observe_event_time(now);
            let id = p.id;
            // `max_per_drain` is spent **per producer**, not shared across them. The
            // bound exists so one producer's burst cannot starve market-data processing
            // on the same thread, and that is a property of each producer's own ring; a
            // shared budget would instead make a chatty strategy able to starve a quiet
            // one, silently, in a way that reads on the status line as the quiet one
            // having nothing to say.
            p.reader.drain(now, *max_per_drain, |sig| {
                // An older target from the *same producer* for the same symbol carries
                // nothing the newer one lacks — that self-containedness is why ADR-0006
                // chose this shape — and acting on both would place two orders for one
                // intent. An older target from a *different* producer is a different
                // claim and is not displaced: two strategies' opinions about one
                // instrument are two facts (ADR-0038).
                match pass
                    .iter_mut()
                    .find(|r| r.strategy == id && r.signal.symbol_id == sig.symbol_id)
                {
                    Some(slot) => {
                        slot.signal = sig;
                        superseded += 1;
                    }
                    None => pass.push(PassRecord {
                        strategy: id,
                        signal: sig,
                    }),
                }
            });
        }
        // Summed across producers, because these are the session's numbers. The
        // per-producer split is on the status line through `strategies()`, which is
        // where an operator asks *which* strategy stopped.
        let mut totals = axon_strategy::SignalStats::default();
        for p in producers.iter() {
            let r = p.reader.stats();
            totals.accepted += r.accepted;
            totals.expired += r.expired;
            totals.stale_seq += r.stale_seq;
            totals.gaps += r.gaps;
            totals.missing += r.missing;
            totals.ahead_of_clock += r.ahead_of_clock;
            totals.schema_version += r.schema_version;
            totals.unknown_kind += r.unknown_kind;
            totals.reserved_not_zero += r.reserved_not_zero;
        }
        stats.accepted = totals.accepted;
        stats.rejected = totals.rejected();
        stats.expired = totals.expired;
        stats.stale_seq = totals.stale_seq;
        stats.gaps = totals.gaps;
        stats.missing = totals.missing;
        stats.ahead_of_clock = totals.ahead_of_clock;
        stats.superseded += superseded;

        // The sweep runs on its own, slower cadence, on the same event clock and with
        // the same signed subtraction as the pass schedule above. Slower because it is
        // the only thing here that reads the tracker when nothing arrived: at the pass
        // cadence a session on a busy feed would take the tracker's read lock and walk
        // its open orders once per millisecond of event time, on the deterministic core
        // thread, to answer a question about a sixty-second bound.
        let sweep_due = order_lifetime_ns.is_some()
            && last_sweep_ns.is_none_or(|prev| now.saturating_sub(prev) >= *sweep_interval_ns);
        if sweep_due {
            // Stamped here rather than after the sweep, exactly as `last_pass_ns` is
            // stamped before the returns below it: this is a *schedule*, and a pass that
            // could not read the tracker has still had its turn. Stamping it later
            // instead would leave every subsequent pass sweep-due on a poisoned lock, so
            // `poisoned` would climb once a millisecond on a session nobody is
            // signalling — a counter that describes the pass rate rather than the fault.
            *last_sweep_ns = Some(now);
        }
        if pass.is_empty() && !sweep_due {
            return;
        }

        // One read of the tracker for the whole pass. Every plan in it is then
        // computed against one position and one set of working orders, taken at one
        // instant — two reads would let a fill land between two symbols' plans and
        // make one of them arithmetic about a position that no longer existed.
        let Ok(tracker) = handler.tracker().read() else {
            // A poisoned lock means a panic left our order state unknown. Planning a
            // delta against a position we cannot read is guessing at exposure, and
            // guessing low is the direction that places the order which breaches the
            // limit.
            stats.poisoned += 1;
            return;
        };

        // ── stage 1: record what each producer is now asking for ─────────────
        //
        // Split from the planning below because with several producers a symbol can
        // appear in one pass more than once — one entry per contributing strategy — and
        // it must be planned exactly **once**, from the sum. Planning inside this loop
        // would send one order per contributor for one netted intent, which is the
        // doubled position ADR-0020 §3's in-flight rule exists to prevent, arriving by a
        // route that rule cannot see.
        touched.clear();
        for rec in pass.iter() {
            let symbol = SymbolId::new(rec.signal.symbol_id);
            // Per symbol, not per pass. One intent at a time for one instrument is the
            // rule that matters — the tracker learns about an order when its ack lands,
            // so a second plan for the same symbol computes the same delta against the
            // same position and one target becomes two orders. Nothing in that rule
            // needed BTC's round trip to hold up ETH, which is what the global counter
            // this replaced did.
            if sink.inflight(symbol) {
                stats.busy += 1;
                carried.push(*rec);
                continue;
            }
            // A producer speaking for an instrument it never declared has escaped the
            // check that keeps two strategies off one position: the declared universe is
            // what lets `RuntimeConfig::validate` catch an overlap at startup, and a
            // record outside it is an overlap nobody could have caught.
            if let Some(p) = producers
                .iter_mut()
                .find(|p| p.id == rec.strategy && !p.owns(rec.signal.symbol_id))
            {
                p.out_of_scope += 1;
                continue;
            }
            // How old this decision was when we finally acted on it — recorded here
            // and not at the drain, because a record held back by the in-flight rule
            // above spends that time too, and a distribution that stopped the clock at
            // the read would report the queue as free. Both stamps are the reader's:
            // `now` is the pass's event time and `ts_event` is the producer's own
            // clock, which is exactly the pair `SignalReader::still_fresh` compares, so
            // this is a distribution *of the admission arithmetic* rather than a second
            // opinion about it.
            latency.record_span_ns(crate::latency::Stage::SignalAge, rec.signal.ts_event, now);
            // …and how old the *observation* was when the strategy decided. A separate
            // stage rather than a widening of the one above, because they are measured on
            // different clocks and answer to different people: `sig` is ring→pass and is
            // this runtime's to fix, while this one is bar→decision and is the producer's.
            // Folding them would produce one number nobody could act on.
            //
            // Recorded here rather than at the drain for the same reason `sig` is: a
            // record held back by the in-flight rule spends that time too. It is skipped
            // entirely when the producer stated no cause, or stated one after the
            // decision — `cause_age_ns` refuses both rather than reporting a zero, and a
            // zero on the stage whose whole point is that it reads in *seconds* is the one
            // value an operator would never question.
            if let Some(age) = rec.signal.cause_age_ns() {
                latency.record(
                    crate::latency::Stage::CauseToDecision,
                    (age / 1_000_000) as u64,
                );
            }
            // The strategy has spoken for this symbol, so the target it stated becomes
            // the one the pass will speak for later — and the re-quote budget resets,
            // because a producer that is talking is the evidence the budget waits for.
            // Recorded here rather than at the drain so that a record held back by the
            // in-flight rule above does not displace the target actually being worked
            // toward; it is carried, and it lands here on the pass that acts on it.
            if let Err(e) = book.state(rec.strategy, &rec.signal, now) {
                match e {
                    axon_strategy::BookReject::Overlap { .. } => stats.overlap_refused += 1,
                    // Unreachable from a configured session — `producers` and the book's
                    // policy list are built from the same resolved list — and counted
                    // rather than ignored, because the alternative is a claim that
                    // silently never lands. Charged to the producer that emitted it, so
                    // the per-strategy line names it rather than leaving an operator to
                    // work out which of four it was.
                    axon_strategy::BookReject::UnknownStrategy { .. } => {
                        if let Some(p) = producers.iter_mut().find(|p| p.id == rec.strategy) {
                            p.out_of_scope += 1;
                        }
                    }
                }
                continue;
            }
            if !touched.contains(&rec.signal.symbol_id) {
                touched.push(rec.signal.symbol_id);
            }
        }
        // Summed **after** the loop that fills it, not before. Incrementing the session
        // total beside each producer's would double-count within a pass and then correct
        // itself on the next one — a counter that is right whenever nobody is reading it.
        stats.out_of_scope = producers.iter().map(|p| p.out_of_scope).sum();

        // ── stage 2: how much of it the account is allowed to work toward ────
        let budgets: Vec<(axon_strategy::StrategyId, Decimal)> = producers
            .iter()
            .map(|p| (p.id, p.max_gross_notional))
            .collect();
        let alloc = allocate(AllocArgs {
            book,
            budgets: &budgets,
            marks: handler.marks(),
            tracker: &tracker,
            portfolio,
            scale_to_fit: *scale_to_fit,
            now,
            stats,
        });

        // ── stage 3: one plan per instrument, from the sum ───────────────────
        for symbol_id in touched.iter().copied() {
            let symbol = SymbolId::new(symbol_id);
            let Some(net) = book.net(symbol_id, now, |sid, qty| alloc.apply(symbol_id, sid, qty))
            else {
                continue;
            };
            if net.synthesized && net.contributors > 1 {
                stats.netted += 1;
            }
            let sig = net.signal;
            project_working(&tracker, symbol, working);
            // Bound to a local first, so this immutable borrow of the table does not
            // collide with the reborrow of `working` in the same literal.
            let precision = instruments.precision(symbol);
            let ctx = PlanContext {
                // The pass's event time, so the planner can age a working order
                // (`WorkingOrder::placed_ts`) against the same clock the reader ages
                // signals against. An input, never a clock read: that is what keeps a
                // replayed session planning the same orders.
                now,
                position: tracker.position(symbol).qty,
                quote: quote_for(handler, symbol, now, *quote_max_age_ns),
                working,
                precision,
            };
            let mut plan = planner.plan(&sig, &ctx);
            retarget_cancels(&tracker, &mut plan);

            if plan.orders.is_empty() {
                stats.no_order += 1;
                match plan.no_order {
                    Some(NoOrder::NoQuote) => stats.no_quote += 1,
                    // Broken out of `no_order` because "already at target" is the
                    // common, healthy case and these are not: one says the strategy is
                    // asking for sizes the instrument cannot express, the other says
                    // the session has stopped trading this instrument altogether.
                    Some(
                        NoOrder::BelowLotSize { .. }
                        | NoOrder::BelowMinNotional { .. }
                        | NoOrder::PriceNotRepresentable { .. }
                        | NoOrder::RoundedUnfillable { .. },
                    ) => stats.precision_refusals += 1,
                    Some(NoOrder::UnknownPrecision { .. }) => stats.unknown_precision += 1,
                    _ => {}
                }
            } else {
                stats.planned += 1;
            }
            if plan.is_noop() {
                continue;
            }
            if !sink.send(Intent {
                symbol_id: symbol,
                seq: sig.seq,
                ts_event: sig.ts_event,
                plan,
            }) {
                stats.dropped += 1;
            }
        }
        let book_stats = book.stats();
        stats.silent_held = book_stats.silent_held;
        stats.silent_flat = book_stats.silent_flat;
        stats.band_dropped = book_stats.band_dropped;

        // Last, and after the planning loop rather than before it, because a symbol a
        // signal just spoke for has had every one of its working orders re-decided —
        // the planner emits its cancels on every no-order path, and its own age bound
        // is `min(signal, operator)`, so an order it chose to leave resting is by
        // construction inside the ceiling the sweeper enforces. Sweeping first would
        // race a decision that was about to be made anyway.
        if sweep_due {
            if let Some(lifetime) = *order_lifetime_ns {
                sweep_overage_orders(SweepArgs {
                    tracker: &tracker,
                    now,
                    lifetime,
                    planned: touched,
                    sink,
                    swept_at,
                    scratch: sweeping,
                    stats,
                });
            }
            // …and then speak for the targets nothing else will. On the sweep's cadence
            // rather than the pass's, because this reads the tracker for a question
            // about a bound measured in tens of seconds, and because the state it is
            // looking for is *created* by the sweep above — an order cancelled here is
            // still working until the venue confirms it, so the re-quote lands on a
            // later tick by construction rather than by a delay anyone tuned.
            // Not while halted. The submit pipeline refuses a placement anyway and
            // structurally, so this is not the protection — it is the difference
            // between spending a target's whole re-quote budget on a halt that is about
            // to clear and still having one when it does. A halt also *causes* the
            // condition this looks for: the sweep above cancels, nothing replaces it,
            // and every symbol looks unquoted at once.
            if halt.state() == HaltState::Running {
                requote_held_targets(RequoteArgs {
                    tracker: &tracker,
                    now,
                    planned: touched,
                    book,
                    alloc: &alloc,
                    limit: *requote_limit,
                    planner,
                    instruments,
                    quote_max_age_ns: *quote_max_age_ns,
                    handler,
                    sink,
                    working,
                    stats,
                });
            }
        }
    }

    fn stats(&self) -> IntentStats {
        self.stats
    }

    fn strategies(&self) -> Vec<StrategyLine> {
        self.producers
            .iter()
            .map(|p| {
                let r = p.reader.stats();
                let policy = self
                    .book
                    .policies()
                    .iter()
                    .find(|q| q.id == p.id)
                    .copied()
                    .unwrap_or_else(|| axon_strategy::StrategyPolicy::new(p.id));
                let last = self.book.last_stated(p.id);
                let since = last.map(|t| self.last_pass_ns.unwrap_or(t).saturating_sub(t).max(0));
                StrategyLine {
                    name: p.name.clone(),
                    attached: p.attached,
                    accepted: r.accepted,
                    rejected: r.rejected(),
                    expired: r.expired,
                    stale_seq: r.stale_seq,
                    claims: self
                        .book
                        .claimed_symbols()
                        .filter(|s| self.book.claim(p.id, *s).is_some())
                        .count(),
                    silent_ms: since.map(|ns| (ns / 1_000_000) as u64),
                    silent: policy.silence_ns > 0 && since.is_some_and(|ns| ns > policy.silence_ns),
                    out_of_scope: p.out_of_scope,
                }
            })
            .collect()
    }
}

/// Everything the allocator reads. A struct for the reason [`SweepArgs`] is one: the
/// caller has already destructured `IntentSource` and every field here is separately
/// borrowed from it.
struct AllocArgs<'a> {
    book: &'a axon_strategy::TargetBook,
    /// `(producer, its own gross allocation)`. Zero is unbounded.
    budgets: &'a [(axon_strategy::StrategyId, Decimal)],
    marks: &'a axon_execution::MarkCache,
    tracker: &'a OrderTracker,
    portfolio: &'a axon_risk::PortfolioLimits,
    scale_to_fit: bool,
    now: Nanos,
    stats: &'a mut IntentStats,
}

/// Work out how much of what the strategies asked for this account may work toward.
///
/// Three stages, and the order between them is the design:
///
/// 1. **Each producer against its own allocation.** A strategy that has decided to be
///    enormous crowds out *its own* future claims rather than the other strategies'
///    share. Skipping this and scaling only at the portfolio level is proportional, and
///    proportional is the wrong answer here: one runaway producer would shrink three
///    well-behaved ones by the same factor it shrank itself.
/// 2. **The netted book against the portfolio bound**, applied to what stage 1 left.
///    Both the gross and the net ceiling scale linearly — `|Σ k·q·m| = k·|Σ q·m|` — so a
///    single factor can satisfy either, and the binding one is the smaller.
/// 3. **Breadth**, which no factor can express: a cap on how many instruments may carry
///    exposure is satisfied by not opening one, never by opening it smaller.
///
/// **An unpriced book is not scaled at all**, and that is the opposite of the fail-closed
/// rule the *gate* follows — deliberately. A gross computed over the legs that happen to
/// be priced is a smaller, entirely plausible number, so scaling on it would quietly
/// shrink every position in the account because one feed went quiet. The gate still
/// refuses new exposure on the same book (`PortfolioReject::Unpriced`), so the session
/// stops adding rather than mis-sizing; here, doing nothing is what leaves the existing
/// positions alone.
fn allocate(args: AllocArgs<'_>) -> Allocation {
    let AllocArgs {
        book,
        budgets,
        marks,
        tracker,
        portfolio,
        scale_to_fit,
        now,
        stats,
    } = args;

    let mut alloc = Allocation {
        per_strategy: vec![Decimal::ONE; budgets.len()],
        portfolio: None,
        denied: Vec::new(),
    };
    stats.breadth_denied = 0;
    let symbols: Vec<u32> = book.claimed_symbols().collect();
    if symbols.is_empty() {
        return alloc;
    }

    // Stage 1 — each producer's own book against its allocation.
    for (i, (id, budget)) in budgets.iter().enumerate() {
        if *budget <= Decimal::ZERO {
            continue;
        }
        let mut gross = Decimal::ZERO;
        let mut unpriced = false;
        for sym in &symbols {
            let Some(sig) = book.claim(*id, *sym) else {
                continue;
            };
            // A closing claim asks for nothing, so it weighs nothing — the same reading
            // `TargetBook::net` gives `FLAG_CLOSE`, and the two must agree or a producer
            // would be allotted against a target it is not asking for.
            let want =
                axon_strategy::fixed_to_decimal(if sig.is_close() { 0 } else { sig.target_qty })
                    .abs();
            if want.is_zero() {
                continue;
            }
            match marks.get(SymbolId::new(*sym)) {
                Some(m) => gross += want * m,
                None => {
                    unpriced = true;
                    break;
                }
            }
        }
        if unpriced {
            stats.alloc_unpriced += 1;
            continue;
        }
        if let Some(k) = axon_risk::gross_scale(gross, *budget) {
            alloc.per_strategy[i] = k;
            stats.strategy_scaled += 1;
        }
    }

    // Stage 2 — the netted book against the portfolio bound.
    if scale_to_fit
        && (portfolio.max_gross_notional > Decimal::ZERO
            || portfolio.max_net_notional > Decimal::ZERO)
    {
        let staged = |sid: axon_strategy::StrategyId, qty: i64| {
            let k = alloc
                .per_strategy
                .get(sid.get() as usize)
                .copied()
                .unwrap_or(Decimal::ONE);
            axon_strategy::scale_fixed(qty, k)
        };
        let mut gross = Decimal::ZERO;
        let mut net = Decimal::ZERO;
        let mut unpriced = false;
        for sym in &symbols {
            let want =
                axon_strategy::fixed_to_decimal(book.net_qty(*sym, now, staged).unwrap_or(0));
            if want.is_zero() {
                continue;
            }
            match marks.get(SymbolId::new(*sym)) {
                Some(m) => {
                    gross += want.abs() * m;
                    net += want * m;
                }
                None => {
                    unpriced = true;
                    break;
                }
            }
        }
        if unpriced {
            stats.alloc_unpriced += 1;
        } else {
            let by_gross = axon_risk::gross_scale(gross, portfolio.max_gross_notional);
            let by_net = axon_risk::gross_scale(net.abs(), portfolio.max_net_notional);
            alloc.portfolio = match (by_gross, by_net) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if let Some(k) = alloc.portfolio {
                stats.portfolio_scaled += 1;
                // Reported as a number rather than a flag: "the portfolio is binding"
                // and "the portfolio is binding at 3 % of what was asked for" are
                // different sessions, and only the second one is a misconfiguration.
                let bps = (k * Decimal::from(10_000)).round();
                stats.portfolio_scale_bps =
                    rust_decimal::prelude::ToPrimitive::to_u32(&bps).unwrap_or(0);
            } else {
                stats.portfolio_scale_bps = 10_000;
            }
        }
    }

    // Stage 3 — breadth. Not a factor: a cap on how many instruments may carry exposure
    // is satisfied by not opening one, never by opening it smaller.
    if portfolio.max_symbols > 0 {
        let scaled = |sid: axon_strategy::StrategyId, qty: i64| {
            let s = alloc
                .per_strategy
                .get(sid.get() as usize)
                .copied()
                .unwrap_or(Decimal::ONE);
            let k = match alloc.portfolio {
                Some(p) => s * p,
                None => s,
            };
            axon_strategy::scale_fixed(qty, k)
        };
        // Everything already carrying a position occupies a slot, whether or not any
        // strategy still claims it — an instrument nobody speaks for is exactly the one
        // an operator has to close, which is what a breadth cap is counting.
        let open: Vec<SymbolId> = tracker
            .exposed_symbols()
            .into_iter()
            .filter(|s| !tracker.position(*s).qty.is_zero())
            .collect();
        let mut candidates: Vec<(u32, Decimal)> = Vec::new();
        for sym in &symbols {
            let id = SymbolId::new(*sym);
            if open.contains(&id) {
                continue;
            }
            let want =
                axon_strategy::fixed_to_decimal(book.net_qty(*sym, now, scaled).unwrap_or(0));
            if want.is_zero() {
                continue;
            }
            // An unpriced candidate sorts last rather than being denied outright: it is
            // a real claim, and the gate will refuse it on the unpriced book anyway.
            let notional = marks
                .get(id)
                .map(|m| want.abs() * m)
                .unwrap_or(Decimal::ZERO);
            candidates.push((*sym, notional));
        }
        // Largest first, symbol id as the tie-break. Deterministic is the requirement —
        // a replay has to admit the same instruments — and *largest* is the defensible
        // way to be deterministic: a breadth cap should keep the positions the
        // strategies care most about, not the ones whose symbol id happens to be low.
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut room = (portfolio.max_symbols as usize).saturating_sub(open.len());
        for (sym, _) in candidates {
            if room > 0 {
                room -= 1;
                continue;
            }
            alloc.denied.push(sym);
        }
        stats.breadth_denied = alloc.denied.len() as u32;
    }

    alloc
}

/// Everything one sweep touches. A struct rather than eight arguments because the
/// caller has already destructured `IntentSource` and every one of these is a
/// separately borrowed field of it.
struct SweepArgs<'a> {
    tracker: &'a OrderTracker,
    /// The pass's event time. See [`sweep_overage_orders`] for why it is that and not
    /// a wall clock.
    now: Nanos,
    /// The operator's ceiling, in nanoseconds of event time.
    lifetime: Nanos,
    /// The instruments this pass planned for. Already decided, so the sweeper must not
    /// touch them.
    planned: &'a [u32],
    sink: &'a mut IntentSink,
    swept_at: &'a mut Vec<(axon_core::Cloid, Nanos)>,
    scratch: &'a mut Vec<(SymbolId, axon_core::Cloid)>,
    stats: &'a mut IntentStats,
}

/// Cancel every order that has outlived the operator's ceiling with no signal to
/// supersede it (ADR-0031).
///
/// **This is the half of an order lifetime the planner cannot own.** `Planner::outlived`
/// bounds a resting order's age and runs *on a signal*, so it is only ever reached by a
/// strategy that is still speaking. A strategy that has stopped — crashed, stalled,
/// still warming up, restarted with a rewound `seq`, or simply not started yet — leaves
/// its last quote resting at the venue, and the mechanism designed to bound that
/// order's age never runs again. A continuously re-targeting strategy needs this; a
/// *silent* one needs it most, because for it nothing else will ever ask.
///
/// **The clock is event time, and that is not the reflex answer here.** The house rule
/// is event time everywhere, and the named exceptions are the ones where the *venue* or
/// a *timer* owns the deadline — a dead-man's-switch re-arm, a reconnect backoff. The
/// tempting exception is this one, on the grounds that a silent strategy is precisely
/// the case where a clock may have stopped. It does not survive the arithmetic:
/// `TrackedOrder::placed_ts` is an event-time stamp, so a wall-clock `now` measured
/// against it is a subtraction between two different clocks, and a replay of yesterday's
/// capture would find every order a day old and sweep the lot. It does not survive the
/// facts either. The clock that a silent *strategy* stops is the signal ring's, and the
/// core's event clock is fed by market data from an entirely different producer, so it
/// goes on advancing through exactly the outage this exists for. What stops event time
/// is a dead *market-data* feed, and that is a different failure with its own detectors
/// (`MarkCache`'s wall-clock liveness, `STALE MARKS`, and the switch) — and one no clock
/// choice here could rescue anyway, because [`IntentSource::poll`]'s own schedule is on
/// event time (ADR-0020 §2), so a frozen event clock runs no pass for a sweeper to sit
/// on. Reaching for a wall clock in the measurement while the schedule stayed on the
/// event clock would buy nothing and forfeit replay determinism for every cancel.
///
/// **The sweeper's cancel is never risk-gated, and it depends on that rather than
/// merely enjoying it.** `GuardedClient::cancel` is ungated by construction (ADR-0010
/// §3) and `HaltableClient` passes cancels through while refusing placements, so this
/// reaches the venue on precisely the sessions that can no longer place an order. The
/// failure mode a gate here would create is exact: a cancel *reduces* exposure, so a
/// gate that can refuse one turns a stale mark, a breached limit or an instrument with
/// no price into an account pinned into the position it is trying to leave. Every input
/// that would make a gate say no — no mark, over the limit, nothing priced — is a
/// reason to want the quote *gone*, and the sweeper's whole subject is a session where
/// nobody else is going to ask.
fn sweep_overage_orders(args: SweepArgs<'_>) {
    let SweepArgs {
        tracker,
        now,
        lifetime,
        planned,
        sink,
        swept_at,
        scratch,
        stats,
    } = args;

    // Bounded by the open-order count, not by the session's length: an order the venue
    // has finished with can never come back, so its entry is dead weight.
    swept_at.retain(|(c, _)| tracker.order(*c).is_some_and(|o| !o.status.is_terminal()));

    scratch.clear();
    for o in tracker.open_orders() {
        if now.saturating_sub(o.placed_ts) <= lifetime {
            continue;
        }
        let symbol = o.symbol_id;
        // A symbol this pass planned for has already had its orders decided, and a
        // symbol with an intent still at the venue must not be handed a second one: the
        // in-flight claim is a per-symbol bit, so two claims against one release would
        // leave that symbol gated for the rest of the session over an intent nobody is
        // waiting on. Neither is a loss — the order is still over-age on the next sweep.
        if sink.inflight(symbol) || planned.contains(&symbol.get()) {
            continue;
        }
        // Re-asked on the same bound it enforces, not on the sweep cadence. The trigger
        // is a property of tracker state, so an order the venue did not remove is still
        // over-age a second later and would be cancelled once a second forever — and
        // signed `/exchange` actions are metered, so one stuck order would spend the
        // day's budget and leave none for the orders that could still be pulled.
        if swept_at
            .iter()
            .any(|(c, at)| *c == o.cloid && now.saturating_sub(*at) <= lifetime)
        {
            continue;
        }
        scratch.push((symbol, o.cloid));
    }
    if scratch.is_empty() {
        return;
    }

    // Sorted, for the reason `project_working` is sorted: the tracker holds orders in a
    // `HashMap` whose iteration order varies between runs, and an unsorted sweep would
    // issue a different sequence of venue actions on each replay of one session.
    scratch.sort_unstable_by_key(|(s, c)| (s.get(), c.get()));

    // One intent per symbol, for the in-flight bit's sake — see above.
    let mut i = 0;
    while i < scratch.len() {
        let symbol = scratch[i].0;
        let first = i;
        let mut cancels = Vec::new();
        while i < scratch.len() && scratch[i].0 == symbol {
            cancels.push(CancelId::Cloid {
                symbol,
                cloid: scratch[i].1,
            });
            i += 1;
        }
        let mut plan = Plan {
            cancels,
            orders: Vec::new(),
            no_order: None,
        };
        // The same re-addressing a planned cancel gets, and this is where it matters
        // most: the orders that outlive a producer are disproportionately the ones a
        // *previous* incarnation of this process left resting, and those are adopted,
        // so their `cloid` is one the tracker synthesized and the venue has never seen.
        retarget_cancels(tracker, &mut plan);
        let n = plan.cancels.len();
        if !sink.send(Intent {
            symbol_id: symbol,
            // No signal produced this, and `0` is not a sequence any record can carry
            // (`SignalReader` counts from 1), so a venue action traced back to `seq 0`
            // says "the sweeper, not the strategy" without a second field to carry it.
            seq: 0,
            // Only ever recorded against an ack, and a sweep has no orders to ack. It
            // is the pass's event time so that a recorded sweep still sorts where it
            // happened.
            ts_event: now,
            plan,
        }) {
            stats.dropped += 1;
            // Deliberately not written into `swept_at`: a plan the queue refused never
            // reached the venue, and marking it asked would suppress the retry for a
            // full lifetime over a failure that lasted one pass.
            continue;
        }
        for (_, cloid) in &scratch[first..i] {
            match swept_at.iter_mut().find(|(c, _)| c == cloid) {
                Some((_, at)) => {
                    *at = now;
                    stats.resweeps += 1;
                }
                None => swept_at.push((*cloid, now)),
            }
            stats.swept += 1;
        }
        // Logged rather than put on the status line, which this crate's own rule says
        // is the same as uncounted — and it is the honest arrangement for now: the line
        // is assembled in `core.rs`, and a sweep is rare enough (at most one ask per
        // order per lifetime) that a line per event is a better record than a counter.
        eprintln!(
            "intent: sweeper cancelled {n} order(s) on {symbol} - resting longer than \
             intent.max_order_age_ms with no signal to supersede them"
        );
    }
}

/// Everything one re-quote pass touches.
struct RequoteArgs<'a, 'h> {
    tracker: &'a OrderTracker,
    /// The pass's event time, and the stamp the re-quoted record carries. See
    /// [`requote_held_targets`] for why the record is re-stamped at all.
    now: Nanos,
    /// The instruments this pass already acted on. A symbol in here has just had every
    /// one of its working orders re-decided by the planner and must not be touched again.
    planned: &'a [u32],
    /// Every producer's standing claim. What used to be a private list of held targets:
    /// the book holds the same thing per *(strategy, symbol)*, so a re-quote of a netted
    /// target is the sum its contributors are still asking for rather than whichever one
    /// spoke last.
    book: &'a mut axon_strategy::TargetBook,
    /// The pass's own allocation, applied to the re-quote exactly as it was to the plan.
    /// Without it a re-quote would put back the *unallocated* target and quietly undo
    /// the portfolio bound one symbol at a time, on the pass nobody is watching.
    alloc: &'a Allocation,
    limit: u32,
    planner: &'a Planner,
    instruments: &'a Arc<InstrumentTable>,
    quote_max_age_ns: Nanos,
    handler: &'h CoreHandler,
    sink: &'a mut IntentSink,
    /// The pass's own working-order scratch, reused. The planning loop above has
    /// finished with it by the time this runs, and this scan happens on every sweep
    /// tick whether or not anything is unquoted — the same reason the sweeper's scratch
    /// is a field rather than a local.
    working: &'a mut Vec<WorkingOrder>,
    stats: &'a mut IntentStats,
}

/// Put a fresh quote behind a target the strategy still holds and nothing is working
/// toward.
///
/// This is the second half of ADR-0031. The sweeper answers "pull a quote nobody
/// speaks for"; this answers "and then what", which the first live sweep proved was an
/// open question — both swept orders were exits, the strategy's target was unchanged
/// so nothing re-emitted it, and a short sat open with no working order for about
/// twelve minutes. See [`HeldTarget`] for why the answer is a bounded re-quote and not
/// one of the two alternatives.
///
/// Five conditions, and each one is load-bearing:
///
/// 1. **A signal must not have spoken for the symbol in this pass.** If it did, the
///    planner has already decided everything about that symbol, including deliberately
///    leaving an order resting.
/// 2. **Nothing may be in flight for the symbol.** The same rule the planning loop
///    obeys, and for the same reason: an order is not *working* until its ack lands,
///    so re-quoting on top of an unsubmitted one holds twice the target.
/// 3. **There must be no working order for the symbol.** A re-quote while the swept
///    order is still live at the venue would double the exposure — and immediately
///    after a sweep the order *is* still live, because a cancel is not a removal until
///    the venue says so. That is what makes this land a tick or more later without a
///    delay anybody had to tune.
/// 4. **The target must not already be met.** A position that matches its target needs
///    no quote, and re-planning it every second would be a plan that never places an
///    order and a counter that never stops climbing.
/// 5. **The budget must not be spent.** After it is, the symbol is *reported* as
///    unquoted rather than quoted again — see [`IntentStats::unquoted`].
///
/// **The record is re-stamped with the pass's event time, and its `seq` is not.**
/// `cloid_for` derives the client id from `(ts_event, seq, symbol)`, so re-planning
/// the original record byte for byte would mint the id of the order that was just
/// cancelled — the venue would de-duplicate it into nothing and the position would
/// stay unquoted while the counters claimed otherwise. Re-stamping is also what the
/// re-quote *is*: a new moment, on a target that has not changed. Keeping `seq` is
/// what lets a venue action be traced back to the record that decided the target, and
/// it is what makes the new id provably distinct from any producer's — a record
/// carrying that seq again would be refused as `stale_seq` before it could be planned.
fn requote_held_targets(args: RequoteArgs<'_, '_>) {
    let RequoteArgs {
        tracker,
        now,
        planned,
        book,
        alloc,
        limit,
        planner,
        instruments,
        quote_max_age_ns,
        handler,
        sink,
        working,
        stats,
    } = args;

    // Absolute, not cumulative: these answer "is anything unquoted or stranded *now*",
    // and a running total would go on reporting a hole that has since been filled.
    stats.unquoted = 0;
    stats.stranded = 0;

    // Collected first because the loop nets each symbol, which needs `&mut book`.
    let symbols: Vec<u32> = book.claimed_symbols().collect();
    for symbol_id in symbols {
        let symbol = SymbolId::new(symbol_id);
        if planned.contains(&symbol_id) || sink.inflight(symbol) {
            continue;
        }
        project_working(tracker, symbol, working);
        if !working.is_empty() {
            continue;
        }
        let ctx = PlanContext {
            now,
            position: tracker.position(symbol).qty,
            quote: quote_for(handler, symbol, now, quote_max_age_ns),
            working,
            precision: instruments.precision(symbol),
        };
        // A re-stamped copy: the same netted target, this moment. Never a larger target
        // and never a different one — everything a re-quote may change is the price.
        //
        // The target is re-netted rather than remembered, and that is the multi-strategy
        // shape of "never a larger target": between the sweep and now a contributor may
        // have gone silent past its window, and re-quoting a remembered sum would put
        // back exposure the silence policy had already withdrawn.
        let Some(net) = book.net(symbol_id, now, |sid, qty| alloc.apply(symbol_id, sid, qty))
        else {
            continue;
        };
        let mut sig = net.signal;
        sig.ts_event = now;
        let plan = planner.plan(&sig, &ctx);
        if plan.orders.is_empty() {
            // Condition 4, answered by the planner rather than by arithmetic here: it
            // owns what "already at target" means, including the grid and the no-op
            // band, and a second opinion about that is a second planner.
            //
            // But "already at target" and "the delta is too small to express" are two
            // very different answers wearing the same empty plan, and only the planner
            // can tell them apart. The second one is a position **no order can close**,
            // so nothing here or upstream will ever act on it again.
            if matches!(
                plan.no_order,
                Some(
                    NoOrder::BelowLotSize { .. }
                        | NoOrder::BelowMinNotional { .. }
                        | NoOrder::PriceNotRepresentable { .. }
                        | NoOrder::RoundedUnfillable { .. }
                )
            ) {
                stats.stranded += 1;
            }
            continue;
        }
        if book.requotes(symbol_id) >= limit {
            // The terminal state, and the one the live run had no words for. Counted
            // every pass it persists, so the status line can say it is *still* true.
            stats.unquoted += 1;
            continue;
        }
        book.note_requote(symbol_id);
        stats.requotes += 1;
        eprintln!(
            "intent: re-quoted a held target on {symbol} - no working order and no new \
             signal ({}/{} of the budget spent)",
            book.requotes(symbol_id),
            limit
        );
        if !sink.send(Intent {
            symbol_id: symbol,
            seq: sig.seq,
            ts_event: now,
            plan,
        }) {
            stats.dropped += 1;
            // Given back, because the plan never reached the venue: charging a queue
            // failure to the budget would spend a session's re-quotes on an outage
            // that lasted one pass.
            book.forgive_requote(symbol_id);
            stats.requotes -= 1;
        }
    }
}

/// Project the tracker's open orders for `symbol` into the planner's local view.
///
/// Sorted by `cloid`, which is not cosmetic: the tracker holds orders in a `HashMap`
/// and its iteration order varies between runs. An unsorted projection would make the
/// planner emit its cancels in a different sequence each time, and a replayed session
/// would issue a different sequence of venue actions than the live one it is supposed
/// to be identical to.
fn project_working(tracker: &OrderTracker, symbol: SymbolId, out: &mut Vec<WorkingOrder>) {
    out.clear();
    for o in tracker.open_orders().filter(|o| o.symbol_id == symbol) {
        out.push(WorkingOrder {
            cloid: o.cloid,
            symbol_id: o.symbol_id,
            side: o.side,
            price: o.price,
            remaining_qty: o.remaining_qty(),
            // Straight through, `None` and all. The tracker knows these for an order it
            // acked and nothing for one it adopted, and the planner refuses to compare
            // against a `None` — which is the whole mechanism. This used to be a second
            // map kept beside the tracker, because `TrackedOrder` carried neither field;
            // a map that a restart emptied while the orders were still resting is what
            // made every inherited order un-restable.
            tif: o.tif,
            reduce_only: o.reduce_only,
            placed_ts: o.placed_ts,
        });
    }
    out.sort_unstable_by_key(|w| w.cloid.get());
}

/// Re-address the cancels for orders whose `cloid` we did not mint.
///
/// The planner cancels by `cloid` because that is the identity it knows. But an
/// adopted order's `cloid` may be one the *tracker* synthesized from a venue order id
/// to key it into its own map — the venue has never seen it. A cancel sent under an id
/// the venue does not recognize fails, and the stale quote it was supposed to remove
/// stays resting exactly where somebody else's taker is looking for it. The venue's own
/// order id is never ambiguous, so anything we did not place is cancelled by that.
fn retarget_cancels(tracker: &OrderTracker, plan: &mut Plan) {
    for c in plan.cancels.iter_mut() {
        let CancelId::Cloid { symbol, cloid } = *c else {
            continue;
        };
        // `adopted` is the tracker's own record of "this order arrived from the venue
        // rather than from a submit of ours", which is exactly the question. It replaces
        // a side map of the cloids this process minted — one a restart emptied, so every
        // inherited order was re-addressed correctly for the accidental reason that the
        // map was empty rather than because anything knew where the id came from.
        match tracker.order(cloid) {
            Some(o) if o.adopted => {
                if let Some(order_id) = o.order_id {
                    *c = CancelId::OrderId { symbol, order_id };
                }
            }
            _ => {}
        }
    }
}

/// Milliseconds of config as nanoseconds of event time.
fn ms_to_ns(ms: u64) -> Nanos {
    (ms as Nanos).saturating_mul(1_000_000)
}

/// The top of book the planner prices against, or `None` if we will not vouch for one.
///
/// Two independent freshness tests, because they catch different failures. The mark
/// cache answers "has this instrument gone quiet?" against the risk gate's own window
/// and its own liveness clock — which keeps advancing on a dead feed, where event time
/// would freeze and call every stale price fresh (ADR-0013 §2). Then
/// [`top_of_book`](crate::quote::top_of_book) answers "is *this* quote current?", which
/// the mark cannot: a `Ticker` takes strict precedence in [`MarkCache`], so a live
/// venue mark keeps vouching for an instrument whose book stopped moving minutes ago.
/// That was the hole the old fallback had — it aged the `bbo` and then handed over a
/// frozen L2 book with no test at all, and priced a resting limit against a market that
/// had gone. Worse, the next pass recomputed the same order from the same frozen book,
/// matched it against the one already working, and left it there on purpose.
///
/// Stale collapses into missing, and the planner already fails closed on missing —
/// which cancels the working orders rather than leaving a quote out in a market we can
/// no longer see.
///
/// [`MarkCache`]: axon_execution::MarkCache
fn quote_for(
    handler: &CoreHandler,
    symbol: SymbolId,
    now: Nanos,
    max_age_ns: Nanos,
) -> Option<Quote> {
    handler.marks().get(symbol)?;
    match top_of_book(handler.market(), symbol, now, max_age_ns) {
        TopState::Fresh(t) => Some(Quote::new(t.bid_px, t.ask_px)),
        TopState::Stale | TopState::Unseen => None,
    }
}

// ── the async edge ───────────────────────────────────────────────────────────

/// Submit one intent: every cancel first, then every order.
///
/// The ordering is the point and it is not a preference. Hyperliquid processes
/// `cancel > post-only > GTC > IOC` within a block, so a cancel and its replacement
/// submitted together cannot both be live. Submitting the order first inverts that and
/// leaves a window holding double the intended exposure (ADR-0014 §6).
///
/// A failed cancel does **not** suppress the order. The overwhelmingly common cause is
/// that the order is already gone — filled or cancelled — and refusing to place would
/// stall the strategy on the venue's own success. The residual case, a venue erroring
/// while an order still rests, is bounded by [`OrderTracker::risk_position`], which
/// counts that resting size as exposure the risk gate has to admit the new order on top
/// of. The planner does not model it; the gate does.
pub async fn submit_intent(
    client: &dyn ExecutionClient,
    intent: &Intent,
    halt: &HaltSwitch,
    tracker: &RwLock<OrderTracker>,
    health: &SessionHealth,
    latency: &crate::latency::LatencyBook,
) {
    for id in &intent.plan.cancels {
        match client.cancel(*id).await {
            Ok(_) => health.note_intent_cancel(),
            Err(e) => {
                health.note_intent_cancel_failure();
                eprintln!("intent: cancel {id:?} failed: {e}");
            }
        }
    }
    if intent.plan.orders.is_empty() {
        return;
    }
    // The pipeline refuses a placement while halted anyway, and structurally, which is
    // why that wrapper exists. Checking here as well is not belt-and-braces for its own
    // sake: a `ProviderError::Rejected` reads in the log exactly like a venue rejection,
    // and an operator needs to be able to tell "the venue said no" from "we stopped
    // ourselves" without reading the message.
    if !halt.is_accepting() {
        health.note_intent_halted(intent.plan.orders.len() as u64);
        return;
    }
    for req in &intent.plan.orders {
        // Wall clock, and a **named exception** in the same class as the dead-man's
        // switch deadline: neither of these two spans orders anything. The first is a
        // network round trip, and no event time exists for "how long the venue took to
        // answer" — the venue's own stamp is on the other side of the thing being
        // measured.
        let sent_ms = crate::dms::now_ms();
        match client.place_order(req.clone()).await {
            Ok(ack) => {
                let acked_ms = crate::dms::now_ms();
                latency.record(
                    crate::latency::Stage::SubmitAck,
                    acked_ms.saturating_sub(sent_ms),
                );
                // Decision to order-at-the-venue, end to end. `ts_event` is the
                // producer's stamp; on a live session that is *also* a wall clock (the
                // named exception in `axon.strategies.live_runner`), so this is a
                // same-clock difference despite spanning two processes. On a replay it
                // is event time and this number is meaningless — which is why nothing
                // offline reaches this function at all.
                latency.record_span_ns(
                    crate::latency::Stage::DecisionToAck,
                    intent.ts_event,
                    (acked_ms as Nanos).saturating_mul(1_000_000),
                );
                health.note_intent_order();
                // The tracker has to learn about this order now, not when the venue's
                // own frame arrives. Until it does, `risk_position` under-counts the
                // exposure we just added, and the next plan sees nothing to cancel —
                // which is how one target becomes two orders.
                match tracker.write() {
                    Ok(mut t) => t.on_ack(req, &ack, intent.ts_event),
                    Err(_) => health.note_intent_untracked(),
                }
            }
            Err(e) => {
                health.note_intent_failure();
                eprintln!("intent: place {:?} failed: {e}", req.cloid);
            }
        }
    }
}

/// Everything the submit task needs, bundled so the supervisor's signature stays
/// readable.
pub struct PumpConfig {
    pub rx: Receiver<Intent>,
    pub inflight: Arc<InFlight>,
    pub halt: Arc<HaltSwitch>,
    pub tracker: Arc<RwLock<OrderTracker>>,
    pub health: Arc<SessionHealth>,
    pub latency: Arc<crate::latency::LatencyBook>,
    pub poll: Duration,
}

/// Drain the intent queue into the submit pipeline until asked to stop.
///
/// Polls rather than blocking on a receive, for the same reason the core loop does: a
/// task that is blocked on a channel cannot also hear a shutdown, and shutdown is
/// exactly when intents must stop first (see [`crate::shutdown`]).
pub async fn pump(
    client: Arc<dyn ExecutionClient>,
    cfg: PumpConfig,
    stop: &tokio::sync::Notify,
) -> u64 {
    let mut handled = 0u64;
    loop {
        while let Ok(intent) = cfg.rx.try_recv() {
            submit_intent(
                client.as_ref(),
                &intent,
                &cfg.halt,
                &cfg.tracker,
                &cfg.health,
                &cfg.latency,
            )
            .await;
            // Only now is this symbol finished, so only now may the core plan for it
            // again. Every other symbol was never waiting on it.
            cfg.inflight.release(intent.symbol_id);
            handled += 1;
        }
        tokio::select! {
            _ = stop.notified() => return handled,
            _ = tokio::time::sleep(cfg.poll) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use async_trait::async_trait;
    use axon_core::{
        Bbo, Cloid, Decimal, Event, EventHandler, ExecEvent, MarketEvent, OrderId, OrderStatus,
        OrderUpdate, Side, Tif,
    };
    use axon_execution::MarkCache;
    use axon_providers::{
        CancelAck, Capabilities, OrderAck, OrderRequest, ProviderError, RateLimitModel,
    };
    use axon_strategy::cloid_for;
    use rust_decimal_macros::dec;
    use std::sync::Mutex;

    const BTC: SymbolId = SymbolId::new(0);
    const SEC: Nanos = 1_000_000_000;
    const MS: Nanos = 1_000_000;

    fn handler() -> CoreHandler {
        CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::never_expires()),
        )
    }

    fn feed(h: &mut CoreHandler, ev: Event) {
        h.on_event(ev.ts_event(), &ev);
    }

    fn quoted(h: &mut CoreHandler, ts: Nanos) {
        feed(
            h,
            Event::Market(MarketEvent::Bbo(Bbo {
                symbol_id: BTC,
                bid_px: dec!(100),
                bid_sz: dec!(5),
                ask_px: dec!(101),
                ask_sz: dec!(5),
                ts_event: ts,
            })),
        );
    }

    fn resting(h: &mut CoreHandler, oid: u64, cloid: Option<u128>, qty: Decimal, ts: Nanos) {
        feed(
            h,
            Event::Exec(ExecEvent::Order(OrderUpdate {
                symbol_id: BTC,
                order_id: OrderId::new(oid),
                cloid: cloid.map(Cloid::new),
                side: Side::Buy,
                status: OrderStatus::Resting,
                price: Some(dec!(99)),
                orig_qty: qty,
                remaining_qty: qty,
                cancel_reason: None,
                ts_event: ts,
            })),
        );
    }

    fn sig(seq: u64, ts: Nanos, target: i64, urgency: u8, ttl_ms: u32) -> Signal {
        Signal::target_position(seq, ts, BTC.get(), target, urgency, 0, ttl_ms, 1, 0)
    }

    /// A venue that declares no instrument grids.
    ///
    /// These tests are about the pass schedule, the in-flight gate, the projection and
    /// the submit ordering — never about rounding, which has its own tests in
    /// `axon-strategy`. Saying "no grids" out loud is what keeps that true: an empty
    /// table would mean `Unknown`, and every one of them would start refusing.
    fn no_grids() -> Arc<InstrumentTable> {
        Arc::new(InstrumentTable::unconstrained())
    }

    /// The grid the two precision tests price against: a whole-unit tick, a 0.01 lot,
    /// and a $10 minimum — the venue's own shape, coarsened so one digit shows it.
    fn coarse_grid() -> Arc<InstrumentTable> {
        use axon_providers::{InstrumentSpec, PriceGrid, SizeGrid};
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: BTC,
            price: PriceGrid::increment(dec!(1)).unwrap(),
            size: SizeGrid::decimals(2).unwrap(),
            min_notional: Some(dec!(10)),
        });
        Arc::new(t)
    }

    /// Testnet's most common shape, not a corner: `szDecimals: 0`, so the lot is one
    /// whole coin and any sub-coin target quantizes away to nothing.
    fn whole_lot_grid() -> Arc<InstrumentTable> {
        use axon_providers::{InstrumentSpec, PriceGrid, SizeGrid};
        let mut t = InstrumentTable::new();
        t.insert(InstrumentSpec {
            symbol_id: BTC,
            price: PriceGrid::decimals_with_sig_figs(6, 5).unwrap(),
            size: SizeGrid::decimals(0).unwrap(),
            min_notional: Some(dec!(10)),
        });
        Arc::new(t)
    }

    /// A source whose records become readable at the event times they were recorded as
    /// released at — the only way to put two records on the ring at two different
    /// moments, and the same mechanism a replayed session uses.
    fn released(records: Vec<(Nanos, Signal)>) -> IntentSource<crate::capture::CapturedSignals> {
        let cfg = RuntimeConfig::default();
        IntentSource::new(
            crate::capture::CapturedSignals::new(
                records
                    .iter()
                    .map(|(ts, s)| axon_replay::SignalRecord::released_at(*ts, s))
                    .collect(),
            ),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        )
    }

    /// A source over canned records, recording what it planned.
    fn source(records: Vec<Signal>) -> (IntentSource<ReplaySource>, Arc<HaltSwitch>) {
        let halt = Arc::new(HaltSwitch::new());
        let cfg = RuntimeConfig::default();
        (
            IntentSource::new(
                ReplaySource::new(records),
                &cfg.intent,
                no_grids(),
                cfg.mark_max_age_ns(),
                halt.clone(),
                IntentSink::Record(Vec::new()),
            ),
            halt,
        )
    }

    // ── many strategies, one account (ADR-0038) ──────────────────────────────

    const ETH: SymbolId = SymbolId::new(1);
    const SOL: SymbolId = SymbolId::new(2);

    fn sig_on(symbol: SymbolId, seq: u64, ts: Nanos, target: i64) -> Signal {
        Signal::target_position(seq, ts, symbol.get(), target, 0, 0, 60_000, 1, 0)
    }

    fn quoted_on(h: &mut CoreHandler, symbol: SymbolId, px: Decimal, ts: Nanos) {
        feed(
            h,
            Event::Market(MarketEvent::Bbo(Bbo {
                symbol_id: symbol,
                bid_px: px,
                bid_sz: dec!(500),
                ask_px: px + dec!(1),
                ask_sz: dec!(500),
                ts_event: ts,
            })),
        );
    }

    /// A multi-producer source over canned records, recording what it planned.
    ///
    /// `producers` is `(name, symbols it owns, its own gross allocation, records)`.
    fn fleet(
        producers: Vec<(&str, Vec<SymbolId>, Decimal, Vec<Signal>)>,
        portfolio: axon_risk::PortfolioLimits,
        overlap: axon_strategy::Overlap,
    ) -> IntentSource<ReplaySource> {
        let cfg = RuntimeConfig::default();
        let sources = producers
            .into_iter()
            .enumerate()
            .map(|(i, (name, symbols, budget, records))| NamedSource {
                name: name.to_string(),
                source: ReplaySource::new(records),
                symbols: symbols.into_iter().map(|s| s.get()).collect(),
                policy: axon_strategy::StrategyPolicy::new(axon_strategy::StrategyId::new(
                    i as u16,
                )),
                max_gross_notional: budget,
            })
            .collect();
        IntentSource::multi(
            sources,
            &cfg.intent,
            portfolio,
            overlap,
            true,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        )
    }

    #[test]
    fn two_strategies_on_one_instrument_produce_one_order_for_the_sum() {
        // The whole feature in one assertion, and the failure it replaces is silent: a
        // pass that planned per record would send *two* orders for one netted intent —
        // the doubled position ADR-0020 §3's in-flight rule exists to prevent, arriving
        // by a route that rule cannot see, because both plans are computed in the same
        // pass against the same tracker read.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 300_000_000)],
                ),
                (
                    "beta",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, -100_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Net,
        );
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "one instrument, one order: {out:?}");
        assert_eq!(out[0].plan.orders[0].qty, dec!(2), "3 long plus 1 short");
        assert_eq!(out[0].plan.orders[0].side, Side::Buy);
        assert_eq!(src.stats().netted, 1);
        assert_eq!(src.stats().accepted, 2, "both records were admitted");
        assert_eq!(src.stats().planned, 1, "and one plan came out");
        // The netted id is in its own space, so it can never collide with a cloid minted
        // from a producer's own record.
        assert!(out[0].seq & axon_strategy::NET_SEQ_TAG != 0);
    }

    #[test]
    fn each_producer_keeps_its_own_sequence_so_one_restart_does_not_refuse_the_others() {
        // The reason there is one ring per producer rather than one ring with a
        // strategy id on the record: `seq` is what proves nothing was lost, and it is
        // per writer. Interleaved on one ring, beta's records would walk alpha's
        // baseline backwards and every one of them would be refused as `stale_seq` —
        // the failure that reads on the status line exactly like a strategy with
        // nothing to say.
        let mut h = handler();
        quoted(&mut h, SEC);
        quoted_on(&mut h, ETH, dec!(50), SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 900, SEC, 100_000_000)],
                ),
                // A much lower sequence, which on a shared ring would be a rewind.
                (
                    "beta",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 100_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        assert_eq!(src.stats().stale_seq, 0);
        assert_eq!(src.take_recorded().len(), 2, "both instruments planned");
    }

    #[test]
    fn a_producer_speaking_outside_its_declared_universe_is_refused_and_named() {
        // The declared universe is what lets `RuntimeConfig::validate` catch two
        // strategies on one position at startup. A record outside it is an overlap that
        // escaped the one check designed to find it, so it is refused rather than netted
        // — and counted per producer, because "a signal was out of scope" on a
        // four-strategy session sends somebody to read four transcripts.
        let mut h = handler();
        quoted(&mut h, SEC);
        quoted_on(&mut h, ETH, dec!(50), SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 100_000_000)],
                ),
                (
                    "beta",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 200_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        assert_eq!(src.stats().out_of_scope, 1);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "only beta's own claim reached the planner");
        assert_eq!(
            out[0].plan.orders[0].qty,
            dec!(2),
            "beta's 2, not alpha's 1 as well"
        );
        let lines = src.strategies();
        assert_eq!(lines[0].name, "alpha");
        assert_eq!(lines[0].out_of_scope, 1);
        assert_eq!(lines[1].out_of_scope, 0);
    }

    #[test]
    fn exclusive_overlap_refuses_the_second_claim_at_runtime_too() {
        // `validate` refuses the *config*, so this is the runtime backstop for a
        // producer that emits outside its declared scope on a session whose scopes are
        // open. Both are needed: the config check cannot see a record, and this one
        // cannot see a config.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 300_000_000)],
                ),
                (
                    "beta",
                    vec![],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 300_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        assert_eq!(src.stats().overlap_refused, 1);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].plan.orders[0].qty, dec!(3), "alpha's claim alone");
    }

    #[test]
    fn a_portfolio_gross_bound_scales_every_target_rather_than_refusing_them() {
        // Refusing is what the guard does and it is the guarantee; scaling is what makes
        // the target *reachable*. Without it a binding bound presents as orders that keep
        // failing on every pass rather than as a limit that is working, and the position
        // sits permanently short of a target it can never reach.
        //
        // Two instruments whose **marks** are the book mids — 100.5 and 50.5, since
        // `MarkCache` takes a mid when no venue mark has arrived — each asked for 3
        // units: gross 453 against a bound of 226.5, so everything halves. The bound is
        // written against the mid rather than the touch on purpose: the allocator prices
        // a book, and the number it prices it at is the one the risk gate uses.
        let mut h = handler();
        quoted(&mut h, SEC);
        quoted_on(&mut h, ETH, dec!(50), SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 300_000_000)],
                ),
                (
                    "beta",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 300_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(226.5),
                ..Default::default()
            },
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 2);
        for o in &out {
            assert_eq!(o.plan.orders[0].qty, dec!(1.5), "halved: {o:?}");
        }
        assert_eq!(src.stats().portfolio_scaled, 1);
        assert_eq!(src.stats().portfolio_scale_bps, 5_000);
        // A scaled claim is no longer the record its author wrote, so it is synthesized
        // and carries a netted id — the strategy did not choose this size, the allocator
        // did, and the `cloid` must not say otherwise.
        assert!(out.iter().all(|o| o.seq & axon_strategy::NET_SEQ_TAG != 0));
    }

    #[test]
    fn a_producers_own_allocation_binds_before_the_portfolio_does() {
        // Proportional scaling alone is the wrong answer to a runaway producer: it would
        // shrink three well-behaved strategies by the same factor it shrank the one that
        // asked for too much. Alpha asks for 10 units of an instrument marked at 100.5
        // against a 201 allocation, so alpha alone is cut to a fifth; beta is untouched.
        let mut h = handler();
        quoted(&mut h, SEC);
        quoted_on(&mut h, ETH, dec!(50), SEC);
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    dec!(201),
                    vec![sig_on(BTC, 1, SEC, 1_000_000_000)],
                ),
                (
                    "beta",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 200_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits::default(),
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        let mut out = src.take_recorded();
        out.sort_by_key(|o| o.symbol_id.get());
        assert_eq!(
            out[0].plan.orders[0].qty,
            dec!(2),
            "alpha: 10 cut to 201/100.5"
        );
        assert_eq!(out[1].plan.orders[0].qty, dec!(2), "beta: untouched");
        assert_eq!(src.stats().strategy_scaled, 1);
        assert_eq!(src.stats().portfolio_scaled, 0);
    }

    #[test]
    fn a_breadth_cap_keeps_the_largest_claims_and_never_forces_an_exit() {
        // Breadth is the one portfolio bound no factor can express: a cap on how many
        // instruments may carry exposure is satisfied by not opening one, never by
        // opening it smaller. Three claims, a cap of two, and the smallest is the one
        // held back — deterministically, because a replay has to admit the same
        // instruments as the session it reproduces.
        let mut h = handler();
        quoted(&mut h, SEC);
        quoted_on(&mut h, ETH, dec!(50), SEC);
        quoted_on(&mut h, SOL, dec!(10), SEC);
        let mut src = fleet(
            vec![
                (
                    "a",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 100_000_000)],
                ),
                (
                    "b",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 100_000_000)],
                ),
                (
                    "c",
                    vec![SOL],
                    Decimal::ZERO,
                    vec![sig_on(SOL, 1, SEC, 100_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits {
                max_symbols: 2,
                ..Default::default()
            },
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        let opened: Vec<u32> = out
            .iter()
            .filter(|o| !o.plan.orders.is_empty())
            .map(|o| o.symbol_id.get())
            .collect();
        assert_eq!(opened.len(), 2, "the cap held: {out:?}");
        assert!(
            !opened.contains(&SOL.get()),
            "the $10 claim is the one denied"
        );
        assert_eq!(src.stats().breadth_denied, 1);
    }

    #[test]
    fn nothing_is_scaled_when_a_claimed_instrument_has_no_mark() {
        // The opposite of the *gate's* fail-closed rule, and deliberately so. A gross
        // computed over the legs that happen to be priced is a smaller, entirely
        // plausible number, so scaling on it would quietly shrink every position in the
        // account because one feed went quiet. The guard still refuses new exposure on
        // the same book; doing nothing here is what leaves the existing positions alone.
        let mut h = handler();
        quoted(&mut h, SEC);
        // ETH is claimed and never quoted, so it has no mark.
        let mut src = fleet(
            vec![
                (
                    "alpha",
                    vec![BTC],
                    Decimal::ZERO,
                    vec![sig_on(BTC, 1, SEC, 300_000_000)],
                ),
                (
                    "beta",
                    vec![ETH],
                    Decimal::ZERO,
                    vec![sig_on(ETH, 1, SEC, 300_000_000)],
                ),
            ],
            axon_risk::PortfolioLimits {
                max_gross_notional: dec!(1),
                ..Default::default()
            },
            axon_strategy::Overlap::Exclusive,
        );
        src.poll(SEC, 0, &h);
        assert!(src.stats().alloc_unpriced > 0);
        assert_eq!(src.stats().portfolio_scaled, 0, "no factor was applied");
        let out = src.take_recorded();
        let btc = out
            .iter()
            .find(|o| o.symbol_id == BTC)
            .expect("BTC planned");
        assert_eq!(
            btc.plan.orders[0].qty,
            dec!(3),
            "unscaled, at the full target"
        );
    }

    #[test]
    fn one_strategy_is_still_the_same_session_it_always_was() {
        // The property that lets this land at all: a session with one producer must plan
        // exactly what it planned before ADR-0038 — the same record, the same seq, and
        // therefore the same `cloid`, which is what makes a replay of an old capture
        // still reproduce it.
        let mut h = handler();
        quoted(&mut h, SEC);
        let record = sig(1, SEC, 300_000_000, 0, 500);
        let (mut src, _) = source(vec![record]);
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].seq, 1,
            "the producer's own sequence, not a netted one"
        );
        assert_eq!(out[0].plan.orders[0].cloid, cloid_for(&record));
        assert_eq!(src.stats().producers, 1);
        assert_eq!(src.stats().netted, 0);
        assert!(
            src.strategies().len() == 1 && src.strategies()[0].name == "strategy",
            "one line, and the status line will not print it"
        );
    }

    #[test]
    fn the_order_that_reaches_the_venue_is_the_delta_not_the_target() {
        // The failure this whole join exists to avoid: "be long 3" while already long 2
        // submitted as an order for 3, ending at 5, compounding on every signal.
        let mut h = handler();
        quoted(&mut h, SEC);
        h.tracker()
            .write()
            .unwrap()
            .on_event(0, &fill_event(dec!(2), 1));

        let (mut src, _) = source(vec![sig(1, SEC, 300_000_000, 0, 500)]);
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1);
        let o = &out[0].plan.orders[0];
        assert_eq!(o.qty, dec!(1), "3 target minus 2 held");
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.price, Some(dec!(100)), "urgency 0 joins the bid");
        assert_eq!(src.stats().planned, 1);
    }

    fn fill_event(qty: Decimal, trade_id: u64) -> Event {
        Event::Exec(ExecEvent::Fill(axon_core::Fill {
            symbol_id: BTC,
            order_id: OrderId::new(99),
            cloid: None,
            side: Side::Buy,
            qty,
            price: dec!(100),
            fee: Decimal::ZERO,
            closed_pnl: Decimal::ZERO,
            liquidity: axon_core::Liquidity::Taker,
            trade_id,
            ts_event: SEC,
        }))
    }

    #[test]
    fn a_working_order_is_cancelled_before_its_replacement_is_placed() {
        // Backwards, this is a window in which the old order and the new one are both
        // live and we hold double the intended exposure.
        let mut h = handler();
        quoted(&mut h, SEC);
        resting(&mut h, 7, Some(0xBEEF), dec!(1), SEC);

        let (mut src, _) = source(vec![sig(1, SEC, 400_000_000, 0, 500)]);
        src.poll(SEC, 0, &h);
        let plan = &src.take_recorded()[0].plan;
        assert_eq!(plan.cancels.len(), 1, "the superseded order is pulled");
        assert_eq!(plan.orders.len(), 1);
        // The type itself carries the ordering; assert the caller cannot mistake it.
        assert_eq!(
            plan.cancels[0],
            CancelId::OrderId {
                symbol: BTC,
                order_id: OrderId::new(7)
            },
            "an order we did not mint the cloid for is cancelled by venue id"
        );
    }

    #[test]
    fn a_stale_signal_never_becomes_an_order() {
        // A late target-position signal is a firm opinion about a market that has
        // already gone. Acting on it is systematically late in the direction the market
        // already moved.
        let mut h = handler();
        quoted(&mut h, 10 * SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 300_000_000, 0, 500)]);
        src.poll(10 * SEC, 0, &h);
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().expired, 1);
        assert_eq!(src.stats().accepted, 0);
    }

    #[test]
    fn a_signal_that_states_no_ttl_is_governed_by_the_operators_ceiling() {
        // The reconciled reading of `ttl_ms == 0` (ADR-0020 §4), asserted on the joined
        // path rather than only in the reader: a producer with no opinion about
        // staleness trades under `intent.max_signal_age_ms` and nothing longer. If this
        // ever read as "never expires", the strategy that never thought about the
        // question would be the one with no protection at all.
        let mut h = handler();
        quoted(&mut h, 3 * SEC);

        let (mut src, _) = source(vec![sig(1, 2 * SEC, 100_000_000, 0, 0)]);
        src.poll(3 * SEC, 0, &h); // 1 s old against a 2 s ceiling
        assert_eq!(src.take_recorded().len(), 1);

        let (mut src, _) = source(vec![sig(1, 0, 100_000_000, 0, 0)]);
        src.poll(3 * SEC, 0, &h); // 3 s old against the same ceiling
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().expired, 1);
    }

    #[test]
    fn a_replayed_signal_plans_the_same_cloid_twice() {
        // A submit that times out has to be retried, and a retry that mints a fresh id
        // is a second order. Two independent sources over the same record must agree.
        let mut h = handler();
        quoted(&mut h, SEC);
        let record = sig(1, SEC, 100_000_000, 0, 500);

        let mut ids = Vec::new();
        for _ in 0..2 {
            let (mut src, _) = source(vec![record]);
            src.poll(SEC, 0, &h);
            ids.push(src.take_recorded()[0].plan.orders[0].cloid);
        }
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0], cloid_for(&record));
    }

    #[test]
    fn only_the_newest_target_for_a_symbol_is_acted_on_in_one_pass() {
        // Two targets in one pass, planned in sequence, would each compute their delta
        // against a position neither of them had moved yet — two orders for one intent.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![
            sig(1, SEC, 100_000_000, 0, 500),
            sig(2, SEC, 500_000_000, 0, 500),
        ]);
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "one order, not two");
        assert_eq!(out[0].plan.orders[0].qty, dec!(5), "the newest target");
        assert_eq!(
            src.stats().accepted,
            2,
            "both records were read and counted"
        );
        assert_eq!(src.stats().superseded, 1);
    }

    #[test]
    fn a_session_with_no_market_data_leaves_the_signals_on_the_ring() {
        // Consuming them would advance the reader's seq baseline and throw the
        // strategy's first targets away, with nothing to show for it: there is no clock
        // to age them against and no book to price them with.
        let h = handler();
        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 500)]);
        src.poll(0, 0, &h);
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().blind, 1);
        assert_eq!(src.stats().accepted, 0, "nothing was consumed");
    }

    #[test]
    fn an_instrument_with_no_usable_quote_still_pulls_its_working_orders() {
        // A quote we cannot price against does not make the resting order safe — it
        // makes it a stale quote against a market we can no longer see.
        let mut h = handler();
        resting(&mut h, 7, Some(0xBEEF), dec!(1), SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 300_000_000, 0, 500)]);
        src.poll(SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1);
        assert!(out[0].plan.orders.is_empty());
        assert_eq!(out[0].plan.cancels.len(), 1);
        assert_eq!(src.stats().no_quote, 1);
    }

    #[test]
    fn a_stale_quote_is_refused_the_same_way_a_missing_one_is() {
        // Pricing a limit off a book that stopped updating puts an order at a price
        // that no longer exists, and unlike a missing book it does not announce itself.
        let mut h = CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::never_expires()),
        );
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![sig(1, 60 * SEC, 300_000_000, 0, 0)]);
        // 59 s later, against a 10 s window.
        src.poll(60 * SEC, 0, &h);
        assert_eq!(src.stats().no_quote, 1);
    }

    /// A source over canned records with a declared instrument table.
    fn source_on(
        records: Vec<Signal>,
        instruments: Arc<InstrumentTable>,
    ) -> IntentSource<ReplaySource> {
        let cfg = RuntimeConfig::default();
        IntentSource::new(
            ReplaySource::new(records),
            &cfg.intent,
            instruments,
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        )
    }

    #[test]
    fn a_grid_refusal_is_counted_apart_from_the_ordinary_no_order() {
        // Folded into `no_order` this is invisible: "already at target" is the common,
        // healthy case and shares the counter. A strategy emitting $5 targets against a
        // $10 venue minimum would then look exactly like a strategy that had nothing to
        // do — and the fix (the strategy's sizing) is nowhere in that picture.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut src = source_on(vec![sig(1, SEC, 5_000_000, 0, 500)], coarse_grid());
        src.poll(SEC, 0, &h);

        assert!(src.take_recorded().is_empty());
        let s = src.stats();
        assert_eq!(s.no_order, 1);
        assert_eq!(s.precision_refusals, 1, "$5 against a $10 minimum");
        assert_eq!(s.unknown_precision, 0);
        assert_eq!(s.no_quote, 0, "the book was fine; the size was not");
    }

    #[test]
    fn a_target_the_lot_erases_is_counted_instead_of_passing_as_already_at_target() {
        // The same counter, reached by the case that is *common* rather than exotic: a
        // whole-coin lot and a target worth a fraction of one. The delta is zero and the
        // position is zero, so the planner used to answer `AlreadyAtTarget` — the healthy
        // variant, which falls through the `_ => {}` arm above and is counted nowhere.
        // The session then prints `sig N/0 sent 0+0c | OK` on every pass while the
        // strategy is asking to trade, which is exactly the reading this file's own docs
        // say must never be indistinguishable from a strategy that chose to be flat.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut src = source_on(vec![sig(1, SEC, 36_000_000, 1, 500)], whole_lot_grid());
        src.poll(SEC, 0, &h);

        assert!(src.take_recorded().is_empty());
        let s = src.stats();
        assert_eq!(s.no_order, 1);
        assert_eq!(s.precision_refusals, 1, "0.36 against a whole-coin lot");
        assert_eq!(s.planned, 0);
        assert_eq!(s.unknown_precision, 0, "the grid is known; the size is not");
    }

    #[test]
    fn an_instrument_with_no_declared_grid_is_counted_where_it_cannot_be_read_as_quiet() {
        // The failure that hides behind counters which *stop*: with no grid the session
        // refuses every order that would add exposure, so `planned` freezes and nothing
        // else moves. This is the only number that says which of the two happened.
        let mut h = handler();
        quoted(&mut h, SEC);
        // A table that HAS grids and does not have this one — which is `Unknown`, not
        // `Unconstrained`. The two must not collapse.
        let mut src = source_on(
            vec![sig(1, SEC, 100_000_000, 0, 500)],
            Arc::new(InstrumentTable::new()),
        );
        src.poll(SEC, 0, &h);

        assert!(src.take_recorded().is_empty());
        let s = src.stats();
        assert_eq!(s.unknown_precision, 1);
        assert_eq!(
            s.precision_refusals, 0,
            "a different diagnosis, and it says so"
        );
        assert_eq!(s.planned, 0);
    }

    #[test]
    fn a_missing_ring_degrades_instead_of_panicking() {
        // Python may legitimately start after Rust. The absence has to be survivable,
        // counted, and visible — a silent no-op here is a session that believes it is
        // trading and is not.
        let mut h = handler();
        quoted(&mut h, SEC);
        let path = std::env::temp_dir().join(format!("axon-absent-{}.ring", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            LazyRing::new(&path, 1),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        src.poll(SEC, 1_000, &h);
        assert!(!src.stats().attached);
        assert_eq!(src.stats().attach_failures, 1);
        assert!(src.take_recorded().is_empty());

        // …and it attaches once the producer shows up, without a restart. The event
        // clock has to move for a second pass to run at all — that is the pass
        // schedule — while `now_ms` is what paces the re-open attempt itself.
        let producer = axon_ipc::Producer::create(&path, 8).unwrap();
        assert!(producer.try_push(&sig(1, SEC, 100_000_000, 0, 500)));
        src.poll(SEC + 2 * MS, 9_000, &h);
        assert!(src.stats().attached);
        assert_eq!(
            src.take_recorded().len(),
            1,
            "and trades on the first record"
        );

        drop(producer);
        drop(src);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_second_intent_for_a_symbol_is_not_planned_until_the_first_has_been_submitted() {
        // Otherwise both plans compute the same delta against the same position,
        // because the first order is not working anywhere the tracker can see.
        let mut h = handler();
        quoted(&mut h, SEC);
        let queue = IntentQueue::new(8);
        let mut cfg = RuntimeConfig::default();
        // One record per pass, so the two targets land in two passes rather than
        // collapsing into one — the in-flight gate is about what happens *between*
        // passes.
        cfg.intent.max_per_drain = 1;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![
                sig(1, SEC, 100_000_000, 0, 500),
                sig(2, SEC, 500_000_000, 0, 500),
            ]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            queue.sink(),
        );
        src.poll(SEC, 0, &h);
        assert_eq!(queue.receiver().len(), 1);

        // The edge has not finished with BTC, so the next pass plans nothing for BTC.
        // Event time is what moves it on: a pass is due once the clock has advanced a
        // drain interval, and nothing here waits on a wall clock to make that true.
        src.poll(SEC + 2 * MS, 0, &h);
        assert_eq!(src.stats().busy, 1);
        assert!(!src.stats().stalled, "one round trip is not a stall");
        assert_eq!(queue.receiver().len(), 1, "nothing added on top");

        // Once it has, the source moves on — and the target it could not act on is
        // still there to act on. Dropping it instead would lose a decision the
        // strategy made and never emitted again, for a gate that was only ever meant
        // to delay it.
        let _ = queue.receiver().recv().unwrap();
        queue.inflight().release(BTC);
        src.poll(SEC + 4 * MS, 0, &h);
        assert_eq!(queue.receiver().len(), 1);
    }

    #[test]
    fn a_slow_submit_on_one_symbol_no_longer_holds_up_another() {
        // ADR-0020's minus column: the gate was a single global counter, so BTC's round
        // trip delayed the next pass for ETH as well — correct, and coarse enough that
        // one slow instrument stopped the whole strategy trading.
        const ETH: SymbolId = SymbolId::new(1);
        let mut h = handler();
        quoted(&mut h, SEC);
        feed(
            &mut h,
            Event::Market(MarketEvent::Bbo(Bbo {
                symbol_id: ETH,
                bid_px: dec!(200),
                bid_sz: dec!(5),
                ask_px: dec!(201),
                ask_sz: dec!(5),
                ts_event: SEC,
            })),
        );

        let queue = IntentQueue::new(8);
        let mut cfg = RuntimeConfig::default();
        cfg.intent.max_per_drain = 1;
        let eth = Signal::target_position(2, SEC + MS, ETH.get(), 100_000_000, 0, 0, 500, 1, 0);
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 100_000_000, 0, 500), eth]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            queue.sink(),
        );

        // BTC goes to the venue and nothing ever answers for it.
        src.poll(SEC, 0, &h);
        assert_eq!(queue.receiver().len(), 1);
        assert!(queue.inflight().contains(BTC));

        // ETH is planned on the very next pass, with BTC still outstanding.
        src.poll(SEC + 2 * MS, 0, &h);
        assert_eq!(queue.receiver().len(), 2, "{:?}", src.stats());
        assert_eq!(src.stats().busy, 0, "ETH was never waiting on BTC");
        assert!(queue.inflight().contains(ETH));
        assert!(queue.inflight().contains(BTC), "and BTC is still gated");
    }

    #[test]
    fn a_target_held_back_by_the_gate_expires_where_it_can_be_counted() {
        // The objection to holding anything back at all (ADR-0020 §3): a second queue
        // behind the ring ages signals invisibly. It does not age invisibly if it ages
        // under the reader's own rule and lands in the reader's own counter — which is
        // the difference between a queue and a held record.
        let mut h = handler();
        quoted(&mut h, SEC);
        let queue = IntentQueue::new(8);
        let mut cfg = RuntimeConfig::default();
        cfg.intent.max_per_drain = 1;
        // The re-quote is off for this test and only for this test. It is a different
        // mechanism (ADR-0036) and it fires here for a good reason — the first target
        // was planned, nothing acked it, so nothing is working toward it — which would
        // put a record on the queue and make the assertion below ambiguous about
        // *which* record it found. The claim under test is about the expired one.
        cfg.intent.max_requotes = 0;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![
                sig(1, SEC, 100_000_000, 0, 500),
                sig(2, SEC, 500_000_000, 0, 500),
            ]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            queue.sink(),
        );
        src.poll(SEC, 0, &h);
        src.poll(SEC + 2 * MS, 0, &h);
        assert_eq!(src.stats().busy, 1, "the second target is held");
        assert_eq!(src.stats().expired, 0);

        // The venue answers a second later — long past the record's own 500 ms window.
        quoted(&mut h, 2 * SEC);
        let _ = queue.receiver().recv().unwrap();
        queue.inflight().release(BTC);
        src.poll(2 * SEC, 0, &h);
        assert_eq!(queue.receiver().len(), 0, "a stale target is not sent");
        assert_eq!(src.stats().expired, 1, "and it is counted, not lost");
    }

    #[test]
    fn two_targets_a_pass_apart_stay_two_orders_however_fast_the_events_arrive() {
        // The pass schedule decides which signals share a pass, and a signal that shares
        // one with a newer target for the same symbol is `superseded` rather than
        // planned. Paced on a wall clock, a replay that drains a log faster than real
        // time collapses the two into one and plans a single order where the live
        // session placed two — a parity divergence caused by replay speed, not logic.
        //
        // Driven here with no sleep at all: the whole point is that the event stream
        // alone decides.
        let mut h = handler();
        quoted(&mut h, SEC);
        // Two passes a drain interval apart in *event* time, the second record not yet
        // readable during the first — exactly what the live session saw.
        let mut src = released(vec![
            (SEC, sig(1, SEC, 100_000_000, 0, 500)),
            (SEC + MS, sig(2, SEC + MS, 500_000_000, 0, 500)),
        ]);
        src.poll(SEC, 0, &h);
        src.poll(SEC + MS, 0, &h);

        let out = src.take_recorded();
        assert_eq!(out.len(), 2, "one order per pass: {:?}", src.stats());
        assert_eq!(out[0].plan.orders[0].qty, dec!(1));
        // Still the whole target: a recording sink never acks, so the tracker's position
        // has not moved between the two passes. Live, the in-flight gate is what stops a
        // second pass running against an unacknowledged first — the schedule is what
        // this test is about, and the schedule produced two decisions rather than one.
        assert_eq!(out[1].plan.orders[0].qty, dec!(5));
        assert_eq!(src.stats().superseded, 0, "neither displaced the other");
    }

    #[test]
    fn a_second_pass_inside_one_drain_interval_of_event_time_is_not_run() {
        // The other half of the same rule. Two records that *did* arrive inside one
        // interval belong to one pass, whatever the machine was doing, so the newest
        // target wins and the older is counted rather than sent.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut src = released(vec![
            (SEC, sig(1, SEC, 100_000_000, 0, 500)),
            (SEC + MS / 2, sig(2, SEC, 500_000_000, 0, 500)),
        ]);
        src.poll(SEC, 0, &h);
        src.poll(SEC + MS / 2, 0, &h);

        assert_eq!(src.take_recorded().len(), 1, "the second pass was not due");
        assert_eq!(
            src.stats().accepted,
            1,
            "and the record is still on the ring"
        );
    }

    #[test]
    fn a_frozen_book_behind_a_live_venue_mark_is_refused_the_same_way_a_missing_one_is() {
        // The mark cache gives a venue `Ticker` strict precedence over book mids, so a
        // live `activeAssetCtx` keeps vouching for an instrument whose `l2Book`
        // subscription died on a reconnect. Priced off that frozen book, an urgency-2
        // buy rests at an ask the market left minutes ago — and the next pass recomputes
        // the identical order, matches it against the one already working, and leaves it
        // there on purpose.
        let mut h = handler();
        feed(
            &mut h,
            Event::Market(MarketEvent::Book(axon_core::BookSnapshot {
                symbol_id: BTC,
                bids: vec![axon_core::Level::new(dec!(49_999), dec!(10))],
                asks: vec![axon_core::Level::new(dec!(50_001), dec!(10))],
                ts_event: SEC,
            })),
        );
        // A venue mark 60 s later: `marks.get()` is fresh and says nothing about the book.
        feed(
            &mut h,
            Event::Market(MarketEvent::Ticker(axon_core::Ticker {
                symbol_id: BTC,
                mark_px: dec!(48_000),
                index_px: None,
                mid_px: None,
                funding: None,
                open_interest: None,
                ts_venue: Some(60 * SEC),
                ts_ingest: 60 * SEC,
            })),
        );
        assert!(h.marks().get(BTC).is_some(), "the mark is live");

        let (mut src, _) = source(vec![sig(1, 60 * SEC, 300_000_000, 2, 0)]);
        src.poll(60 * SEC, 0, &h);
        let out = src.take_recorded();
        assert_eq!(src.stats().no_quote, 1, "59 s old against a 10 s window");
        assert!(
            out.is_empty() || out[0].plan.orders.is_empty(),
            "a stale book must not price an order: {out:?}"
        );
    }

    #[test]
    fn a_busy_multi_symbol_session_is_never_mistaken_for_a_stalled_one() {
        // The false positive a per-symbol gate makes reachable and a global one did
        // not: with one batch in flight at a time the set emptied between every pair
        // of batches, so "something outstanding" was a usable stall signal. Per symbol
        // it is not — a session trading enough instruments always has something
        // outstanding. Progress is the signal; occupancy is not.
        const ETH: SymbolId = SymbolId::new(1);
        let mut h = handler();
        quoted(&mut h, SEC);
        let queue = IntentQueue::new(8);
        let cfg = RuntimeConfig::default();
        let ceiling = (cfg.intent.max_signal_age_ms as Nanos) * MS;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 100_000_000, 0, 0)]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            queue.sink(),
        );
        src.poll(SEC, 0, &h);

        // ETH never answers, so the set is occupied without interruption for far longer
        // than the ceiling — while BTC turns over the whole time.
        queue.inflight().claim(ETH);
        for i in 1..=8i64 {
            let t = SEC + (i * ceiling) / 2;
            queue.inflight().release(BTC);
            queue.inflight().claim(BTC);
            quoted(&mut h, t);
            src.poll(t, 0, &h);
            assert!(!queue.inflight().is_empty(), "never idle for a moment");
            assert!(!src.stats().stalled, "pass {i}: {:?}", src.stats());
        }

        // Stop making progress and the same occupancy is now a stall.
        let t = SEC + 8 * ceiling;
        quoted(&mut h, t);
        src.poll(t, 0, &h);
        src.poll(t + 2 * ceiling, 0, &h);
        assert!(src.stats().stalled, "{:?}", src.stats());
    }

    #[test]
    fn a_submitter_that_never_answers_is_reported_as_stalled_rather_than_quiet() {
        // The wedge hides behind counters that *stop*: the core will not plan again
        // until the edge finishes, so it stops draining the ring, `accepted` freezes and
        // the status line reads like a strategy with nothing to say. Measured in event
        // time, so a replay of the capture reaches the same verdict.
        let mut h = handler();
        quoted(&mut h, SEC);
        let queue = IntentQueue::new(8);
        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 100_000_000, 0, 500)]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            queue.sink(),
        );
        src.poll(SEC, 0, &h);
        assert_eq!(queue.receiver().len(), 1, "the intent is with the edge");

        // Nothing ever drains it. Inside the ceiling this is an ordinary round trip…
        src.poll(SEC + 2 * MS, 0, &h);
        assert!(!src.stats().stalled);
        // …past it, no signal that arrived in the meantime could still have been acted
        // on, which is the definition of the path having stopped working.
        let ceiling = (cfg.intent.max_signal_age_ms as Nanos) * MS;
        src.poll(SEC + ceiling + 10 * MS, 0, &h);
        assert!(src.stats().stalled, "{:?}", src.stats());
        assert!(
            queue.inflight().contains(BTC),
            "and it is still outstanding"
        );

        // `busy` is deliberately NOT the evidence, and this is why: the producer went
        // quiet, so no target was ever held back, and the counter that used to carry
        // this diagnosis reads zero on the exact session it is supposed to describe.
        // The wedge is visible because nothing *completed*, not because anything piled
        // up behind it.
        assert_eq!(src.stats().busy, 0);
    }

    #[test]
    fn a_poisoned_tracker_abandons_the_pass_and_is_counted_where_it_can_be_seen() {
        // Our own order state is unreadable, so a delta computed against it is a guess —
        // and guessing low is the direction that places the order which breaches the
        // limit. On a quiet account no exec event arrives to raise EXEC EVENTS DROPPED
        // either, so this counter is the only evidence the session is no longer trading.
        let mut h = handler();
        quoted(&mut h, SEC);
        let tracker = h.tracker().clone();
        let _ = std::thread::spawn(move || {
            let _guard = tracker.write().unwrap();
            panic!("a panic under the tracker lock");
        })
        .join();
        assert!(h.tracker().read().is_err(), "the lock must be poisoned");

        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 500)]);
        src.poll(SEC, 0, &h);
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().poisoned, 1);
    }

    #[test]
    fn a_forming_candle_pushes_the_core_clock_into_the_future_and_stops_the_pass() {
        // The live hazard this file could characterize and not fix. It is asserted here
        // rather than only beside the fix because the blast radius is entirely inside
        // this file and is invisible from the place that causes it.
        //
        // A candle's `ts_event` is `open_time + interval` — the moment the bar *closes*
        // — and a venue republishes the **forming** bar many times before that moment
        // arrives, every frame carrying the same stamp. Nothing on `axon_core::Candle`
        // tells a forming bar from a closed one, and on Hyperliquid nothing can: the
        // venue publishes no finality bit and often sends no frame at or after the close
        // at all. So `CoreHandler` does not try to tell them apart — it refuses a
        // *computed* close time as a clock either way (`handler::advances_the_clock`).
        //
        // Without that refusal two things break here, and neither says anything:
        //
        // 1. every record on the ring is aged against that clock, so a signal written a
        //    moment ago looks a minute old and is refused as expired;
        // 2. worse, `last_pass_ns` is left a minute ahead, and the schedule's
        //    subtraction is signed *by design* (a late arrival must not trigger a pass).
        //    So no pass runs at all until event time catches up — one pass per bar,
        //    against a clock that is always a bar ahead.
        //
        // The session reported `sig N/M` with `expired` climbing and `OK` beside it.
        let mut h = handler();
        quoted(&mut h, SEC);

        // One record per pass, so "did a pass run?" is answerable by counting records
        // rather than by inspecting the schedule that is on trial here.
        let mut cfg = RuntimeConfig::default();
        cfg.intent.max_per_drain = 1;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![
                sig(1, SEC, 100_000_000, 0, 500),
                sig(2, SEC + MS, 200_000_000, 0, 500),
            ]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );

        // A forming one-minute bar that opened at `SEC`: it closes a minute from now,
        // and that is the stamp the venue puts on every republication of it.
        let close_ns = SEC + 60 * SEC;
        feed(
            &mut h,
            Event::Market(MarketEvent::Candle(axon_core::Candle {
                symbol_id: BTC,
                interval: axon_core::CandleInterval::M1,
                open: dec!(100),
                high: dec!(101),
                low: dec!(99),
                close: dec!(100),
                volume: dec!(1),
                open_time: SEC,
                ts_event: close_ns,
            })),
        );
        assert_eq!(
            h.last_ts(),
            SEC,
            "the clock still stands where the market left it, not where a bar will close"
        );
        assert!(
            h.last_ts() < close_ns,
            "nothing may put the core ahead of anything that has happened"
        );

        // So the signal is admitted and planned, rather than judged a minute old.
        src.poll(h.last_ts(), 0, &h);
        assert_eq!(
            src.stats().expired,
            0,
            "a signal written a moment ago is not a minute old"
        );
        assert_eq!(src.stats().accepted, 1);
        assert_eq!(
            src.take_recorded().len(),
            1,
            "the target became an intent instead of being dropped"
        );

        // And the schedule is anchored in the present, so the next quote still runs a
        // pass. This is the half that failed silently: the signed subtraction is correct
        // — it is what stops a late arrival triggering a pass — and it is exactly what
        // makes a schedule anchored a minute ahead stop scheduling anything at all.
        quoted(&mut h, SEC + 2 * MS);
        assert_eq!(h.last_ts(), SEC + 2 * MS, "event time moved forwards");
        src.poll(h.last_ts(), 0, &h);
        assert_eq!(
            src.stats().accepted,
            2,
            "a second pass ran: the schedule waits on the market, not on a bar close"
        );
        assert_eq!(src.stats().expired, 0, "{:?}", src.stats());
    }

    // ── the sweeper (ADR-0031) ───────────────────────────────────────────────

    /// An order **we** placed, so its `cloid` is one the venue has seen and its
    /// `placed_ts` is the ack's event time.
    fn placed(h: &CoreHandler, cloid: u128, qty: Decimal, ts: Nanos) {
        let req = OrderRequest::limit(
            BTC,
            Side::Buy,
            qty,
            dec!(99),
            Tif::PostOnly,
            Cloid::new(cloid),
        );
        let ack = OrderAck {
            cloid: Cloid::new(cloid),
            order_id: Some(OrderId::new(cloid as u64)),
            status: OrderStatus::Resting,
        };
        h.tracker().write().unwrap().on_ack(&req, &ack, ts);
    }

    /// One second past the default minute, which is the interesting side of the bound.
    const OVER_AGE: Nanos = SEC + 61 * SEC;

    #[test]
    fn a_quote_left_by_a_strategy_that_went_silent_is_pulled_without_a_signal() {
        // The half of an order lifetime the planner cannot own. `Planner::outlived`
        // bounds a resting order's age and runs *on a signal*, so a strategy that
        // crashed, stalled or never warmed up leaves its last quote at the venue with
        // the one mechanism that would pull it never running again — and a target
        // position nobody is renewing is a stale quote, which is what somebody else's
        // taker is looking for.
        let mut h = handler();
        quoted(&mut h, SEC);
        placed(&h, 0xA1, dec!(1), SEC);

        let (mut src, _) = source(vec![]); // the producer never speaks at all
        src.poll(SEC, 0, &h);
        assert!(
            src.take_recorded().is_empty(),
            "a minute has not passed yet"
        );

        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "{:?}", src.stats());
        assert!(out[0].plan.orders.is_empty(), "a sweep places nothing");
        assert_eq!(
            out[0].plan.cancels,
            vec![CancelId::Cloid {
                symbol: BTC,
                cloid: Cloid::new(0xA1)
            }]
        );
        assert_eq!(out[0].seq, 0, "no record produced it, and it says so");
        assert_eq!(src.stats().swept, 1);
        assert_eq!(src.stats().accepted, 0, "no signal was involved");
    }

    #[test]
    fn a_detached_ring_no_longer_stops_the_only_thing_that_can_pull_a_quote() {
        // "Python is not there" is the strongest possible statement that no signal is
        // coming — and it was the one condition under which the pass returned before
        // reaching anything that could act. So the case that most needs a sweeper was
        // exactly the case that ran nothing.
        let mut h = handler();
        quoted(&mut h, SEC);
        placed(&h, 0xA1, dec!(1), SEC);

        let path = std::env::temp_dir().join(format!("axon-sweep-{}.ring", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            LazyRing::new(&path, 1),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        src.poll(SEC, 1_000, &h);
        assert!(!src.stats().attached);

        src.poll(OVER_AGE, 9_000, &h);
        assert!(!src.stats().attached, "still no producer");
        assert_eq!(
            src.stats().swept,
            1,
            "and the quote still went: {:?}",
            src.stats()
        );
        assert_eq!(src.take_recorded().len(), 1);
    }

    #[test]
    fn an_adopted_order_is_swept_by_the_venues_own_id_rather_than_a_cloid_it_never_saw() {
        // An adopted order's `cloid` is one the *tracker* synthesized to key it into its
        // own map. A cancel sent under it fails, and the stale quote it was supposed to
        // remove stays resting. Adopted orders are also disproportionately what a sweep
        // finds: an order that outlives its producer is usually one a previous
        // incarnation of this process left behind, and this session never minted its id.
        let mut h = handler();
        quoted(&mut h, SEC);
        resting(&mut h, 7, Some(0xBEEF), dec!(1), SEC);

        let (mut src, _) = source(vec![]);
        src.poll(SEC, 0, &h);
        src.poll(OVER_AGE, 0, &h);

        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "{:?}", src.stats());
        assert_eq!(
            out[0].plan.cancels,
            vec![CancelId::OrderId {
                symbol: BTC,
                order_id: OrderId::new(7)
            }]
        );
    }

    #[test]
    fn a_sweep_the_venue_ignored_is_re_asked_once_a_lifetime_rather_than_once_a_second() {
        // The trigger is a property of tracker state, so an order the venue did not
        // remove is still over-age on the very next sweep. Unbounded, that is one signed
        // `/exchange` action per sweep interval against a venue that is not answering —
        // and the action budget is metered (~10 203 a day), so one stuck order would
        // spend the day and leave nothing for the orders that could still be pulled.
        let mut h = handler();
        quoted(&mut h, SEC);
        placed(&h, 0xA1, dec!(1), SEC);

        let (mut src, _) = source(vec![]);
        src.poll(SEC, 0, &h);
        src.poll(OVER_AGE, 0, &h);
        assert_eq!(src.take_recorded().len(), 1);

        // The venue never answers, so the order is still open and still over age. Twelve
        // more sweeps inside one lifetime, and not one of them re-asks.
        for i in 1..=12i64 {
            src.poll(OVER_AGE + i * 5 * SEC, 0, &h);
        }
        assert!(src.take_recorded().is_empty(), "{:?}", src.stats());
        assert_eq!(src.stats().swept, 1);
        assert_eq!(src.stats().resweeps, 0);

        // A full lifetime later it is a *new* failure — the first ask demonstrably did
        // not work — so it is asked again, and counted where that can be read.
        src.poll(OVER_AGE + 61 * SEC, 0, &h);
        assert_eq!(src.take_recorded().len(), 1);
        assert_eq!(src.stats().swept, 2);
        assert_eq!(
            src.stats().resweeps,
            1,
            "escalation is a count, not a kill switch"
        );
    }

    #[test]
    fn a_halted_session_still_pulls_the_quotes_it_left_resting() {
        // The asymmetry the halt switch exists for, applied to the sweeper: whatever
        // went wrong, removing exposure must keep working while adding it must not. A
        // sweeper that stopped at the halt would leave a stale quote out for exactly as
        // long as the session was in the state that says it cannot manage one.
        let mut h = handler();
        quoted(&mut h, SEC);
        placed(&h, 0xA1, dec!(1), SEC);

        let (mut src, halt) = source(vec![]);
        src.poll(SEC, 0, &h);
        halt.halt();
        src.poll(OVER_AGE, 0, &h);
        assert_eq!(src.stats().swept, 1, "{:?}", src.stats());
        assert_eq!(src.take_recorded()[0].plan.cancels.len(), 1);
    }

    #[test]
    fn an_operator_who_set_no_order_lifetime_gets_no_sweeper() {
        // `0` is "I set no bound", not "already expired" — the same reading the planner
        // gives it (ADR-0020 §4 transferred). Inverted, the sweeper would cancel every
        // order in the book on the first pass after it was placed, which is the one
        // reading that turns a safety mechanism into an outage.
        let mut h = handler();
        quoted(&mut h, SEC);
        placed(&h, 0xA1, dec!(1), SEC);

        let mut cfg = RuntimeConfig::default();
        cfg.intent.max_order_age_ms = 0;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        src.poll(SEC, 0, &h);
        src.poll(SEC + 3_600 * SEC, 0, &h);
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().swept, 0);
    }

    #[test]
    fn a_sweep_that_cannot_read_the_tracker_counts_the_fault_and_not_the_pass_rate() {
        // The sweep is the only thing in a pass that reads the tracker when *nothing
        // arrived*, so it is also the only thing that can turn `POISONED TRACKER n` into
        // a measure of how fast the core loop spins. The cadence is a schedule, like the
        // pass schedule above it: a sweep that could not read the tracker has still had
        // its turn, and the next one is due a sweep interval later — not next pass.
        let mut h = handler();
        quoted(&mut h, SEC);
        let tracker = h.tracker().clone();
        let _ = std::thread::spawn(move || {
            let _guard = tracker.write().unwrap();
            panic!("a panic under the tracker lock");
        })
        .join();
        assert!(h.tracker().read().is_err(), "the lock must be poisoned");

        let (mut src, _) = source(vec![]); // nobody is signalling
        for i in 0..50i64 {
            src.poll(SEC + i * MS, 0, &h);
        }
        assert_eq!(
            src.stats().poisoned,
            1,
            "fifty passes inside one sweep interval: {:?}",
            src.stats()
        );
    }

    #[test]
    fn a_symbol_a_signal_just_spoke_for_is_not_swept_on_top_of_its_own_cancel() {
        // Two intents for one symbol out of one pass, and the second claims an in-flight
        // bit that a single release then clears — which leaves that symbol gated for the
        // rest of the session over an intent nobody is waiting on. It is also redundant:
        // every no-order path already emits the planner's cancels, and the planner's own
        // age bound is `min(signal, operator)`, so an order it chose to leave resting is
        // by construction inside the ceiling the sweeper enforces.
        let mut h = handler();
        quoted(&mut h, OVER_AGE);
        placed(&h, 0xA1, dec!(1), SEC);

        let (mut src, _) = source(vec![sig(1, OVER_AGE, 300_000_000, 0, 500)]);
        src.poll(OVER_AGE, 0, &h);

        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "one intent, and it is the planner's");
        assert_eq!(out[0].seq, 1, "the signal's sequence, not the sweeper's 0");
        assert_eq!(out[0].plan.cancels.len(), 1, "the planner pulled it");
        assert_eq!(out[0].plan.orders.len(), 1, "and replaced it");
        assert_eq!(src.stats().swept, 0);
    }

    // ── the re-quote: the sweeper's other half (ADR-0031 amended, ADR-0036) ──

    /// Take the order the tracker holds terminal, the way a confirmed cancel does.
    ///
    /// A cancel *asked for* is not a removal, and the difference is what makes the
    /// re-quote land a tick later rather than on top of a live order — so a test that
    /// skipped this step would be testing a state the venue never produces.
    fn cancel_confirmed(h: &mut CoreHandler, oid: u64, cloid: u128, qty: Decimal, ts: Nanos) {
        feed(
            h,
            Event::Exec(ExecEvent::Order(OrderUpdate {
                symbol_id: BTC,
                order_id: OrderId::new(oid),
                cloid: Some(Cloid::new(cloid)),
                side: Side::Buy,
                status: OrderStatus::Cancelled,
                price: Some(dec!(99)),
                orig_qty: qty,
                remaining_qty: qty,
                cancel_reason: Some(axon_core::CancelReason::Requested),
                ts_event: ts,
            })),
        );
    }

    #[test]
    fn a_target_whose_quote_was_swept_is_re_quoted_at_a_fresh_price() {
        // The composition Phase 6 measured at a venue, end to end. The strategy states a
        // target; the order rests; the sweeper pulls it because no signal spoke for it;
        // and — before this — nothing re-quoted, because a target position is idempotent
        // and the strategy correctly says nothing when nothing has changed. A short sat
        // open with no working order for about twelve minutes.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 60_000)]);
        src.poll(SEC, 0, &h);
        let first = src.take_recorded();
        assert_eq!(
            first.len(),
            1,
            "the strategy's own order: {:?}",
            src.stats()
        );
        let placed_cloid = first[0].plan.orders[0].cloid;
        // The order the planner just decided on is now resting at the venue.
        resting(&mut h, 0xA1, Some(placed_cloid.get()), dec!(1), SEC);

        // A minute later the sweeper pulls it, and nothing has spoken since.
        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        let sweep = src.take_recorded();
        assert_eq!(sweep.len(), 1, "the sweep: {:?}", src.stats());
        assert_eq!(src.stats().swept, 1);
        assert!(sweep[0].plan.orders.is_empty(), "a sweep places nothing");
        assert_eq!(
            src.stats().requotes,
            0,
            "not while the order is still working"
        );

        // The venue confirms, and the next sweep tick finds a target with nothing
        // working toward it.
        cancel_confirmed(&mut h, 0xA1, placed_cloid.get(), dec!(1), OVER_AGE + SEC);
        quoted(&mut h, OVER_AGE + 2 * SEC);
        src.poll(OVER_AGE + 2 * SEC, 0, &h);

        let out = src.take_recorded();
        assert_eq!(out.len(), 1, "the re-quote: {:?}", src.stats());
        assert_eq!(src.stats().requotes, 1);
        assert_eq!(out[0].plan.orders.len(), 1);
        assert_eq!(out[0].seq, 1, "traceable to the record that set the target");
        assert_ne!(
            out[0].plan.orders[0].cloid, placed_cloid,
            "a re-quote that minted the swept order's own id would be de-duplicated \
             into nothing by the venue while every counter here claimed success"
        );
        assert_eq!(
            out[0].plan.orders[0].qty,
            dec!(1),
            "the delta to the SAME target, never a larger one"
        );
    }

    #[test]
    fn a_dead_producers_target_is_re_quoted_a_bounded_number_of_times_and_then_named() {
        // The bound is what keeps the repair from undoing the sweeper. The sweeper's
        // whole subject is a producer that has stopped, so an unbounded re-quote would
        // hand a dead strategy an immortal quote. After the budget the session stops
        // placing and starts *saying* the position is unquoted — the one thing the live
        // run could not do.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 60_000)]);
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        // Three re-quotes, one per sweep tick, and each one never reaches the tracker
        // because nothing acks it — which is exactly the shape of a producer whose
        // orders keep being pulled.
        let mut at = OVER_AGE;
        for expected in 1..=3 {
            quoted(&mut h, at);
            src.poll(at, 0, &h);
            assert_eq!(src.stats().requotes, expected, "tick at {at}");
            assert_eq!(src.take_recorded().len(), 1);
            at += 2 * SEC;
        }

        quoted(&mut h, at);
        src.poll(at, 0, &h);
        assert_eq!(src.stats().requotes, 3, "the budget is spent");
        assert!(src.take_recorded().is_empty(), "and nothing more is placed");
        assert_eq!(src.stats().unquoted, 1, "it is reported instead of hidden");
    }

    /// A source that hands each record over only once the core's clock has reached its
    /// `ts_event`.
    ///
    /// [`ReplaySource`] hands over everything on the first drain, which is right for the
    /// tests it was written for and wrong for any test about a producer that speaks
    /// *later*: a record stamped a minute ahead is admitted immediately (the reader
    /// tolerates a signal from the future and counts it `ahead_of_clock`), so a
    /// two-record `ReplaySource` is a one-pass source with a superseded record in it.
    #[derive(Debug, Default)]
    struct GatedSource {
        pending: Vec<Signal>,
        now: Nanos,
    }

    impl GatedSource {
        fn new(records: Vec<Signal>) -> Self {
            Self {
                pending: records,
                now: 0,
            }
        }
    }

    impl SignalSource for GatedSource {
        fn next_signal(&mut self) -> Option<Signal> {
            let i = self.pending.iter().position(|s| s.ts_event <= self.now)?;
            Some(self.pending.remove(i))
        }
    }

    impl Attachable for GatedSource {
        fn ensure(&mut self, _now_ms: u64) -> bool {
            true
        }
        fn observe_event_time(&mut self, now: Nanos) {
            self.now = now;
        }
    }

    #[test]
    fn a_new_signal_gives_the_symbol_its_whole_re_quote_budget_back() {
        // Reset, not decremented: a producer that is talking is the evidence the budget
        // was waiting for, and a session that spent its three re-quotes during one
        // outage must not be permanently unable to re-quote afterwards.
        let mut h = handler();
        quoted(&mut h, SEC);
        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            GatedSource::new(vec![
                sig(1, SEC, 100_000_000, 0, 60_000),
                sig(2, OVER_AGE + 9 * SEC, 200_000_000, 0, 60_000),
            ]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        let mut at = OVER_AGE;
        for _ in 0..3 {
            quoted(&mut h, at);
            src.poll(at, 0, &h);
            let _ = src.take_recorded();
            at += 2 * SEC;
        }
        assert_eq!(src.stats().requotes, 3, "spent");

        // The strategy speaks again, at a new target.
        quoted(&mut h, OVER_AGE + 9 * SEC);
        src.poll(OVER_AGE + 9 * SEC, 0, &h);
        assert_eq!(src.stats().unquoted, 0, "a signal spoke for it this pass");
        let _ = src.take_recorded();

        quoted(&mut h, OVER_AGE + 11 * SEC);
        src.poll(OVER_AGE + 11 * SEC, 0, &h);
        assert_eq!(
            src.stats().requotes,
            4,
            "the budget came back: {:?}",
            src.stats()
        );
    }

    #[test]
    fn a_symbol_with_a_working_order_is_never_re_quoted_on_top_of_it() {
        // The doubling guard. Immediately after a sweep the swept order is *still
        // working* — a cancel is not a removal until the venue says so — and a re-quote
        // there would leave two live orders for one target.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 60_000)]);
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();
        placed(&h, 0xA1, dec!(1), SEC);

        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        let out = src.take_recorded();
        assert_eq!(src.stats().swept, 1, "swept: {:?}", src.stats());
        assert_eq!(src.stats().requotes, 0);
        assert!(
            out.iter().all(|i| i.plan.orders.is_empty()),
            "nothing was placed while the order was still live: {out:?}"
        );
    }

    #[test]
    fn a_position_already_at_its_target_is_not_re_quoted_every_second_forever() {
        // Condition 4, answered by the planner rather than by arithmetic here. Without
        // it a flat session with a flat target would place nothing and count a re-quote
        // on every sweep tick — a counter that describes the sweep cadence.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, _) = source(vec![sig(1, SEC, 100_000_000, 0, 60_000)]);
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();
        // The order filled: the position is now the target.
        feed(
            &mut h,
            Event::Exec(ExecEvent::Fill(axon_core::Fill {
                symbol_id: BTC,
                order_id: OrderId::new(0xA1),
                cloid: None,
                side: Side::Buy,
                qty: dec!(1),
                price: dec!(100),
                fee: dec!(0),
                closed_pnl: dec!(0),
                liquidity: axon_core::Liquidity::Maker,
                trade_id: 1,
                ts_event: SEC,
            })),
        );

        for at in [OVER_AGE, OVER_AGE + 2 * SEC, OVER_AGE + 4 * SEC] {
            quoted(&mut h, at);
            src.poll(at, 0, &h);
        }
        assert_eq!(src.stats().requotes, 0, "{:?}", src.stats());
        assert_eq!(
            src.stats().unquoted,
            0,
            "and it is not an unquoted target either"
        );
        assert!(src.take_recorded().is_empty());
    }

    #[test]
    fn a_halted_session_does_not_spend_a_targets_budget_on_orders_it_cannot_place() {
        // A halt refuses placements structurally, so this is not the protection. It is
        // the difference between spending a target's whole budget on a halt that is
        // about to clear and still having one when it does — and a halt is itself a
        // *cause* of the unquoted state, because the shutdown sweep cancels everything
        // at once.
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, halt) = source(vec![sig(1, SEC, 100_000_000, 0, 60_000)]);
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        halt.halt();
        for at in [OVER_AGE, OVER_AGE + 2 * SEC] {
            quoted(&mut h, at);
            src.poll(at, 0, &h);
        }
        assert_eq!(src.stats().requotes, 0, "{:?}", src.stats());

        halt.resume();
        quoted(&mut h, OVER_AGE + 4 * SEC);
        src.poll(OVER_AGE + 4 * SEC, 0, &h);
        assert_eq!(src.stats().requotes, 1, "the budget survived the halt");
    }

    /// Leave `qty` of BTC on the tracker's books at `price`, as a venue fill would.
    fn hold(h: &CoreHandler, qty: Decimal, price: Decimal) {
        h.tracker().write().unwrap().on_event(
            SEC,
            &Event::Exec(ExecEvent::Fill(axon_core::Fill {
                symbol_id: BTC,
                order_id: OrderId::new(1),
                cloid: None,
                side: Side::Buy,
                qty,
                price,
                fee: dec!(0),
                closed_pnl: dec!(0),
                liquidity: axon_core::Liquidity::Maker,
                trade_id: 1,
                ts_event: SEC,
            })),
        );
    }

    #[test]
    fn a_residue_under_the_venue_minimum_is_closed_rather_than_left_for_an_operator() {
        // **Measured live on 2026-07-27, on the re-quote's first run, and this test
        // asserted the wrong outcome for it.** A closing buy for 0.0003 BTC partially
        // filled 0.00017, the sweeper pulled the remainder a minute later, and 0.00013
        // BTC — about $8.50, under the venue's $10 minimum and under the session's own
        // `min_order_qty` — was left with no working order. The session named the state
        // (`STRANDED POSITION`) and nothing could act on it.
        //
        // Naming it was the right *first* move and it is not the fix. The floors that
        // refused every close are ours, not the venue's: `min_order_qty` is a churn
        // bound and a close is not churn, and the venue's published minimum does not say
        // whether it exempts a close. Both now exempt an order that takes the position
        // to exactly flat — so the residue is quoted, and if the venue really does
        // refuse it we find out loudly instead of holding the position on our own
        // opinion. See `axon_strategy::planner`.
        let mut h = handler();
        quoted(&mut h, SEC);
        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 0, 0, 60_000)]),
            &cfg.intent,
            // A $10 minimum against a 100-unit price: the residue below is worth $1.
            coarse_grid(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        // Flat target, and one lot of position left over.
        hold(&h, dec!(0.01), dec!(100));
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        assert_eq!(src.stats().stranded, 0, "{:?}", src.stats());
        assert_eq!(src.stats().requotes, 1, "the residue got a quote");
        let out = src.take_recorded();
        let order = &out[0].plan.orders[0];
        assert_eq!(order.side, Side::Sell);
        assert_eq!(order.qty, dec!(0.01), "the whole residue");
        assert!(
            order.qty * order.price.unwrap() < dec!(10),
            "and it really is under the venue's minimum"
        );
    }

    #[test]
    fn a_residue_finer_than_the_lot_is_named_rather_than_re_quoted_forever() {
        // What is left of `stranded` once the two floors we chose stop refusing closes:
        // a residue no *price* can help with, because it is smaller than the smallest
        // size the instrument can express. `unquoted` would be the wrong word — nothing
        // is unquoted, it is unquotable, and no number of re-quotes fixes it.
        //
        // On a venue whose positions are sums of lot-sized fills this cannot arise,
        // which is the point: a non-zero count here means the grid changed under a live
        // position, and that is a thing an operator has to be told rather than a thing
        // the session can plan its way out of.
        let mut h = handler();
        quoted(&mut h, SEC);
        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 0, 0, 60_000)]),
            &cfg.intent,
            // Lot = one whole coin, and half a coin held against it.
            whole_lot_grid(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        hold(&h, dec!(0.5), dec!(100));
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        assert_eq!(src.stats().stranded, 1, "{:?}", src.stats());
        assert_eq!(
            src.stats().unquoted,
            0,
            "not the same state and not the same fix"
        );
        assert_eq!(src.stats().requotes, 0, "and nothing was placed");
        assert!(src.take_recorded().is_empty());
    }

    #[test]
    fn an_operator_who_turned_the_re_quote_off_gets_the_old_behaviour_exactly() {
        // `0` is the escape hatch, and it has to be a real one: every session before
        // ADR-0036 left the position unquoted, and an operator who wants that back must
        // be able to have it without also losing the *report* that it happened.
        let mut h = handler();
        quoted(&mut h, SEC);
        let mut cfg = RuntimeConfig::default();
        cfg.intent.max_requotes = 0;
        let mut src = IntentSource::new(
            ReplaySource::new(vec![sig(1, SEC, 100_000_000, 0, 60_000)]),
            &cfg.intent,
            no_grids(),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        src.poll(SEC, 0, &h);
        let _ = src.take_recorded();

        quoted(&mut h, OVER_AGE);
        src.poll(OVER_AGE, 0, &h);
        assert_eq!(src.stats().requotes, 0);
        assert!(src.take_recorded().is_empty());
        assert_eq!(
            src.stats().unquoted,
            1,
            "off means placing nothing, not seeing nothing"
        );
    }

    #[test]
    fn a_stopped_session_stops_reading_signals_altogether() {
        let mut h = handler();
        quoted(&mut h, SEC);
        let (mut src, halt) = source(vec![sig(1, SEC, 100_000_000, 0, 500)]);
        halt.stop();
        src.poll(SEC, 0, &h);
        assert!(src.take_recorded().is_empty());
        assert_eq!(src.stats().accepted, 0, "the record is still on the ring");
    }

    // ── the async edge ───────────────────────────────────────────────────────

    #[derive(Default)]
    struct SpyClient {
        /// Every call in the order it was made — the assertion surface for
        /// "cancels before orders".
        calls: Mutex<Vec<String>>,
        fail_places: bool,
    }

    const CAPS: Capabilities = Capabilities {
        venue: "spy",
        order_types: &[axon_core::OrderType::Limit],
        tifs: &[Tif::Gtc, Tif::PostOnly, Tif::Ioc],
        max_batch: 20,
        native_market_orders: false,
        reduce_only: true,
        rate_limit_model: RateLimitModel::None,
    };

    #[async_trait]
    impl ExecutionClient for SpyClient {
        fn capabilities(&self) -> &Capabilities {
            &CAPS
        }
        async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("place {}", req.qty));
            if self.fail_places {
                return Err(ProviderError::Rejected("spy refuses".into()));
            }
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: Some(OrderId::new(1234)),
                status: OrderStatus::Resting,
            })
        }
        async fn place_batch(&self, _r: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
            unimplemented!("the intent path submits one order per signal")
        }
        async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("cancel {:?}", id.symbol()));
            Ok(CancelAck {
                cloid: None,
                order_id: None,
            })
        }
        async fn cancel_all(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn modify(&self, _i: CancelId, _r: OrderRequest) -> Result<OrderAck, ProviderError> {
            unimplemented!("the planner cancels and replaces; it never modifies")
        }
    }

    fn intent_with_cancel() -> Intent {
        Intent {
            symbol_id: BTC,
            seq: 1,
            ts_event: SEC,
            plan: Plan {
                cancels: vec![CancelId::Cloid {
                    symbol: BTC,
                    cloid: Cloid::new(9),
                }],
                orders: vec![OrderRequest::limit(
                    BTC,
                    Side::Buy,
                    dec!(2),
                    dec!(100),
                    Tif::Gtc,
                    Cloid::new(1),
                )],
                no_order: None,
            },
        }
    }

    #[tokio::test]
    async fn the_submitter_sends_every_cancel_before_any_order() {
        let spy = SpyClient::default();
        let halt = HaltSwitch::new();
        let tracker = RwLock::new(OrderTracker::new());
        let health = SessionHealth::new(0);
        let book = crate::latency::LatencyBook::undeclared();
        submit_intent(&spy, &intent_with_cancel(), &halt, &tracker, &health, &book).await;

        let calls = spy.calls.lock().unwrap().clone();
        assert_eq!(calls, vec!["cancel SymbolId(0)", "place 2"]);
        assert_eq!(health.intent_orders(), 1);
        assert_eq!(health.intent_cancels(), 1);
    }

    #[tokio::test]
    async fn a_submitted_order_enters_the_tracker_before_the_venue_confirms_it() {
        // Until it does, the risk gate under-counts the exposure we just added and the
        // next plan sees nothing to cancel — one target, two orders.
        let spy = SpyClient::default();
        let halt = HaltSwitch::new();
        let tracker = RwLock::new(OrderTracker::new());
        let health = SessionHealth::new(0);
        let book = crate::latency::LatencyBook::undeclared();
        submit_intent(&spy, &intent_with_cancel(), &halt, &tracker, &health, &book).await;

        let t = tracker.read().unwrap();
        assert_eq!(t.open_count(), 1);
        assert_eq!(
            t.risk_position(BTC).qty,
            dec!(2),
            "resting counts as exposure"
        );
        assert_eq!(
            t.order(Cloid::new(1)).unwrap().last_update,
            SEC,
            "event time"
        );
    }

    #[tokio::test]
    async fn a_halted_session_pulls_its_quotes_and_places_nothing() {
        // The asymmetry the halt switch exists for: whatever went wrong, removing
        // exposure must keep working while adding it must not.
        let spy = SpyClient::default();
        let halt = HaltSwitch::new();
        halt.halt();
        let tracker = RwLock::new(OrderTracker::new());
        let health = SessionHealth::new(0);
        let book = crate::latency::LatencyBook::undeclared();
        submit_intent(&spy, &intent_with_cancel(), &halt, &tracker, &health, &book).await;

        assert_eq!(
            spy.calls.lock().unwrap().clone(),
            vec!["cancel SymbolId(0)"]
        );
        assert_eq!(health.intent_halted(), 1);
        assert_eq!(health.intent_orders(), 0);
        assert_eq!(tracker.read().unwrap().open_count(), 0);
    }

    #[tokio::test]
    async fn a_refused_order_is_counted_and_never_enters_the_tracker() {
        // A phantom order in the tracker over-counts exposure forever and the gate
        // starts refusing trades nobody placed.
        let spy = SpyClient {
            fail_places: true,
            ..SpyClient::default()
        };
        let halt = HaltSwitch::new();
        let tracker = RwLock::new(OrderTracker::new());
        let health = SessionHealth::new(0);
        let book = crate::latency::LatencyBook::undeclared();
        submit_intent(&spy, &intent_with_cancel(), &halt, &tracker, &health, &book).await;
        assert_eq!(health.intent_failures(), 1);
        assert_eq!(tracker.read().unwrap().open_count(), 0);
    }

    #[tokio::test]
    async fn a_cancel_reaches_the_venue_through_a_risk_gate_that_refuses_everything_else() {
        // The property the sweeper depends on rather than merely enjoys (ADR-0010 §3,
        // ADR-0031). A cancel *reduces* exposure, so a gate that could refuse one would
        // pin the account into the position it is trying to leave — and every input that
        // makes the gate say no (no mark, over the limit, nothing priced) is a reason to
        // want the quote gone. The sweeper's whole subject is a session where nobody
        // else is ever going to ask for it.
        //
        // The second assertion is what stops this being vacuous: an empty risk context
        // has no mark for the symbol, so the placement beside the cancel is refused, and
        // the gate is demonstrably live rather than merely present.
        use axon_execution::{GuardedClient, StaticRiskContext};
        use axon_risk::{RiskEngine, RiskLimits};

        let guarded = GuardedClient::new(
            SpyClient::default(),
            RiskEngine::new(RiskLimits {
                max_position: dec!(1),
                max_notional: dec!(1),
                max_order_qty: dec!(1),
            }),
            StaticRiskContext::new(),
        );
        let halt = HaltSwitch::new();
        let tracker = RwLock::new(OrderTracker::new());
        let health = SessionHealth::new(0);
        let book = crate::latency::LatencyBook::undeclared();
        submit_intent(
            &guarded,
            &intent_with_cancel(),
            &halt,
            &tracker,
            &health,
            &book,
        )
        .await;

        assert_eq!(health.intent_cancels(), 1, "the cancel was never gated");
        assert_eq!(
            health.intent_failures(),
            1,
            "and the gate was live: the placement beside it was refused"
        );
        assert_eq!(tracker.read().unwrap().open_count(), 0);
    }

    #[tokio::test]
    async fn the_pump_releases_the_core_only_once_the_venue_has_answered() {
        let queue = IntentQueue::new(4);
        let mut sink = queue.sink();
        assert!(sink.send(intent_with_cancel()));
        assert!(queue.inflight().contains(BTC));

        let stop = Arc::new(tokio::sync::Notify::new());
        let client: Arc<dyn ExecutionClient> = Arc::new(SpyClient::default());
        let task = {
            let stop = stop.clone();
            let cfg = PumpConfig {
                rx: queue.receiver(),
                inflight: queue.inflight(),
                halt: Arc::new(HaltSwitch::new()),
                tracker: Arc::new(RwLock::new(OrderTracker::new())),
                health: Arc::new(SessionHealth::new(0)),
                latency: Arc::new(crate::latency::LatencyBook::undeclared()),
                poll: Duration::from_millis(1),
            };
            tokio::spawn(async move { pump(client, cfg, &stop).await })
        };
        // The pump has to have run before the flag can clear, so this loop terminates
        // only if the handoff actually works.
        while !queue.inflight().is_empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        stop.notify_one();
        assert_eq!(task.await.unwrap(), 1);
    }
}
