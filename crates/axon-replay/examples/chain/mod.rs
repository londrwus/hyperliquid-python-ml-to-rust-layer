//! The replayed chain — **the production chain**, not a model of it.
//!
//! Each event goes through [`CoreHandler`], which is the exact handler a live session
//! installs: market data first, then the mark cache (whose updater reads the book it
//! has just updated), then the order tracker last because it is the only consumer
//! behind a lock (ADR-0013 §1). Then one pass of [`IntentSource`], which is the exact
//! strategy adapter a live session runs: [`SignalReader`](axon_strategy::SignalReader)
//! validates the record and [`Planner`](axon_strategy::Planner) turns it into orders.
//!
//! **Nothing here fans an event out, prices a quote, projects a working order or mints
//! a `cloid`.** That is the whole point. A second fan-out in the harness would drift
//! from the live one — silently, since both would keep passing their own tests — and
//! parity would become a claim about this file instead of a claim about the code
//! (`docs/07-parity-and-testing.md`). What this file *does* own is the schedule, the
//! signal feed and the readout, and each of those is argued for below.
//!
//! It lives under `examples/` rather than in `src/` because the runtime is a
//! **dev**-dependency: the production edge has to run the other way, so a live session
//! can record itself by adding `Capture` to its own handler chain. `tests/` pulls this
//! module in with `#[path]`, so the binary Python drives and the golden tests are
//! driving one driver rather than two.
//!
//! ## Two schedules, and why they are what they are
//!
//! **The pass schedule.** Live, the core loop runs one intent pass per iteration,
//! paced by [`IntentConfig::drain_interval_ms`] against the **event** clock — so which
//! signals land in one pass is a property of the log and of nothing else, and a replay
//! at full speed gets the live session's passes. Nothing here parks, and nothing here
//! reimplements the pacing. The harness only skips calling `poll` where the signal feed
//! has no record due, which is observationally identical because a pass that admits no
//! record touches no state at all (`a_pass_with_nothing_due_is_a_no_op_so_skipping_one_is_free`
//! in `tests/` is what keeps that true), and [`ChainOptions::poll_every_event`] runs the
//! faithful schedule so a test can pin the two together.
//!
//! **The release schedule.** A `Signal` carries the time the strategy *decided*, not
//! the time the record became readable; the ring does not carry the second and the
//! difference is what makes a signal stale. [`SignalRecord::release_ts`] carries it, and
//! [`SignalFeed`] hands a record to the reader only once the core's event clock has
//! reached it. Feeding everything on the first pass instead would collapse a session's
//! worth of decisions into one and would make an expiry impossible to replay.
//!
//! ## The grid, and where it now comes from
//!
//! A live session plans on the venue's **grid** and rounds (ADR-0025). A log used to
//! carry no instrument table, so a replay planned the same signals *unrounded* and its
//! `PlannedOrder` prices differed from the ones the session sent — on exactly the orders
//! rounding touches, and `price` is compared exactly by the golden harness. That is a
//! harness difference wearing a strategy change's clothes.
//!
//! `SCHEMA_VERSION` 2 puts the table in the log (ADR-0027), so the default is now to
//! **take the grid from the log being replayed** and a caller that wants a different one
//! has to say so ([`ChainOptions::instruments`]). What has not changed is that
//! [`ChainProbe::new`] takes it as a **required argument**: the driver resolves where it
//! comes from, and the probe still cannot be built without naming it. A log that declares
//! nothing — one written by a session that was never handed a table — still falls back to
//! `Unconstrained`, loudly, and the divergence that causes is pinned by a test rather
//! than discovered in a diff.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use axon_contracts::Signal;
use axon_core::{Clock, Decimal, Event, EventHandler, ExecEvent, ManualClock, Nanos, SymbolId};
use axon_execution::{HaltSwitch, MarkCache, OrderTracker};
use axon_providers::{CancelId, InstrumentTable, OrderRequest};
use axon_replay::chain::{
    cloid_hex, Cell, ChainRow, ChainSummary, PlannedCancel, PlannedOrder, SignalCounters,
    SymbolState, RESULT_SCHEMA, RESULT_SCHEMA_VERSION,
};
use axon_replay::{ReplayOrder, ReplayReport, ReplaySource, SignalLog, SignalRecord};
use axon_runtime::capture::CapturedSignals;
use axon_runtime::config::RuntimeConfig;
use axon_runtime::handler::CoreHandler;
use axon_runtime::intent::{Attachable, Intent, IntentPoll, IntentSink, IntentSource};
use axon_strategy::SignalSource;

/// The signal records a replay feeds the reader, released at their recorded times.
///
/// **The release rule is not written here.** It is
/// [`axon_runtime::CapturedSignals`](axon_runtime::capture::CapturedSignals) — the
/// recorder's own counterpart, the thing a live session's log reads back through — and
/// this type is only the sharing around it. A harness that reimplemented "released
/// when the core's clock reaches `release_ts`" would be a second copy of a rule that
/// already has two ends, and the copies drift in the direction nobody looks: a golden
/// run comparing a replay that admits a signal against a live session that did not.
///
/// Shared by `Rc<RefCell<_>>` because [`IntentSource`] owns the source and never lends
/// it back, while the probe has to know two things about it: what event time the core
/// has reached (to release records) and whether anything is due (to decide whether a
/// pass is worth running). Interior mutability is the cheap way to have one feed with
/// two readers on one thread; a channel would be a second ordering to get wrong.
#[derive(Clone)]
pub struct SignalFeed {
    inner: Rc<RefCell<CapturedSignals>>,
}

impl SignalFeed {
    pub fn new(records: Vec<SignalRecord>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(CapturedSignals::new(records))),
        }
    }

    pub fn observe(&self, now: Nanos) {
        self.inner.borrow_mut().observe_event_time(now);
    }

    /// Whether a record is readable right now — the probe's cue to run a pass.
    pub fn due(&self) -> bool {
        self.inner.borrow().due()
    }

    /// How many records the feed carries. Named for the count rather than `len`
    /// because "empty" is the interesting question about a collection and not about a
    /// producer that simply had nothing to say.
    pub fn record_count(&self) -> usize {
        self.inner.borrow().record_count()
    }
}

impl SignalSource for SignalFeed {
    fn next_signal(&mut self) -> Option<Signal> {
        self.inner.borrow_mut().next_signal()
    }
}

impl Attachable for SignalFeed {
    /// A canned feed is always there — it *is* the records. The production source is a
    /// memory-mapped ring that may not exist yet, and re-opening it is the owner's
    /// business; a replay has no such state to recover from.
    fn ensure(&mut self, _now_ms: u64) -> bool {
        true
    }

    fn observe_event_time(&mut self, now: Nanos) {
        self.observe(now);
    }
}

/// Knobs the replay binary exposes. Everything else comes from
/// [`RuntimeConfig::default`], so the replayed session is configured the way a live
/// one is rather than the way this harness finds convenient.
pub struct ChainOptions {
    pub order: ReplayOrder,
    /// Signals to feed the strategy adapter. Empty is legitimate: a market-data-only
    /// capture still replays the whole chain, the adapter simply admits nothing.
    pub signals: SignalLog,
    /// Run a pass after *every* event, the way the live loop does, instead of only
    /// where a record is due.
    ///
    /// Off by default because a pass with nothing due does no work worth the call. It
    /// exists so a test can pin the two schedules together: if the cheap schedule ever
    /// stopped producing what the faithful one produces, the harness would be quietly
    /// replaying a different program from the live one.
    pub poll_every_event: bool,
    /// Keep every [`ChainRow`] in memory, so the caller can diff or write them.
    ///
    /// On by default, because every golden comparison there is wants the rows. Off is
    /// for the case M3 exists for: a multi-hour tape. The log itself is streamed and the
    /// event-time index costs 16 bytes a record, but a `ChainRow` carries two `BTreeMap`s
    /// and is two orders of magnitude larger — so a probe that retained them would put
    /// the memory bound back exactly where ADR-0027 took it out of, one layer up, and a
    /// soak replay would die holding a trace nobody asked for.
    pub keep_trace: bool,
    /// A grid to override the log's own with (ADR-0025, ADR-0027).
    ///
    /// `None` — the default — means **use the table the log declared**, which is the
    /// only setting that reproduces the session's own prices. `Some` is for a caller
    /// that knows better than the log: a test comparing two grids, or a replay of a
    /// pre-ADR-0027 recording whose grid has been recovered from somewhere else.
    ///
    /// An override, not a fallback, because the two mistakes are not symmetric. A
    /// replay that rounds differently from the session it reproduces plans different
    /// **prices**, and `price` is a field the golden comparison diffs exactly — so the
    /// divergence lands inside the harness built to detect divergence, and it must take
    /// a deliberate act to cause.
    pub instruments: Option<Arc<InstrumentTable>>,
}

impl Default for ChainOptions {
    fn default() -> Self {
        Self {
            order: ReplayOrder::EventTime,
            signals: SignalLog::empty(),
            poll_every_event: false,
            keep_trace: true,
            instruments: None,
        }
    }
}

/// Where a replay's grid came from, so a run can say it out loud.
///
/// Stderr and not the summary: `axon.backtest` parses stdout, and the summary is a
/// compared artifact that a `RESULT_SCHEMA_VERSION` bump of its own would have to land
/// with (ADR-0025 §out-of-scope 2, still open).
pub struct ResolvedGrid {
    pub table: Arc<InstrumentTable>,
    /// One line for an operator. Empty when the log's own declared table was used
    /// unmodified, which is the case nobody needs to be told about.
    pub note: String,
}

/// Decide which grid a replay plans on, and say what that means where it is not the
/// log's own.
///
/// The fall-back for a log that declared nothing is `Unconstrained` and **not** an empty
/// table: an empty `InstrumentTable` means every symbol is `Precision::Unknown`, which
/// refuses every order that would add exposure, so a harness that used one would report
/// its own ignorance as the strategy's output — a session being refused all day reads
/// exactly like a session that chose to be flat.
pub fn resolve_grid(src: &ReplaySource, opts: &ChainOptions) -> ResolvedGrid {
    if let Some(table) = &opts.instruments {
        return ResolvedGrid {
            table: table.clone(),
            note: format!(
                "replay: planning on a caller-supplied grid ({} instrument(s)), not the \
                 one the log declared",
                table.len()
            ),
        };
    }
    match src.instruments().to_table() {
        Ok(Some(table)) => {
            // A venue with no grids at all is a *declared* answer and a legitimate one —
            // a simulator, or the synthetic fixture. It is still worth a line, because
            // the prices that come out of it are unrounded and somebody comparing them
            // against a live capture has to know which of the two they are looking at.
            let note = if table.is_unconstrained() && table.is_empty() {
                "replay: this log declares a venue with no instrument grid at all - \
                 prices are not rounded, which is correct for a simulated session and \
                 wrong for a comparison against a live one"
                    .to_string()
            } else {
                String::new()
            };
            ResolvedGrid {
                table: Arc::new(table),
                note,
            }
        }
        Ok(None) => ResolvedGrid {
            table: Arc::new(InstrumentTable::unconstrained()),
            note: "replay: this log declares no instrument grid - planning unconstrained, \
                   so prices are NOT rounded the way the captured session's were"
                .to_string(),
        },
        // A grid this build cannot rebuild is a *format* problem, and pretending it is a
        // venue with no rules would quietly hand back the permissive program again.
        Err(e) => ResolvedGrid {
            table: Arc::new(InstrumentTable::unconstrained()),
            note: format!(
                "replay: the log's instrument grid could not be rebuilt ({e}) - planning \
                 unconstrained, so prices are NOT rounded the way the captured session's \
                 were"
            ),
        },
    }
}

/// The columns every row carries, whether or not they apply to it.
///
/// Emitted even when absent, because a golden comparison compares column *sets* before
/// values: a row that quietly dropped a column would be reported as a structural
/// mismatch on every later run rather than as the one missing reading it is. Listed in
/// the order a `BTreeMap` yields them, so the `debug_assert` below is a plain equality
/// and not a set comparison that could pass while a name was misspelled twice.
const COLUMNS: [&str; 16] = [
    "best_ask",
    "best_bid",
    "dropped_exec_events",
    "last_trade_px",
    "mark_px",
    "mark_ts",
    "mid",
    "open_orders",
    "orphan_fills",
    "planned_cancels",
    "planned_orders",
    "position_qty",
    "resting_qty",
    "risk_qty",
    "signals_accepted",
    "signals_rejected",
];

/// What the probe reads out of production state for one symbol.
///
/// Every field is *read*, never derived. The moment the probe computes a mid of its
/// own it is a second implementation of the thing under test.
///
/// The tracker's four columns are `Option` for a different reason from the market
/// ones: a book that has never traded is genuinely absent, while a tracker behind a
/// poisoned lock is *unreadable*. Both are `None` here because the trace has one
/// honest answer for the two — `Cell::Absent` — and the alternative for the second is
/// a zero that reads as a flat position.
struct Sample {
    mid: Option<Decimal>,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    last_trade_px: Option<Decimal>,
    mark_px: Option<Decimal>,
    mark_ts: Option<Nanos>,
    position_qty: Option<Decimal>,
    risk_qty: Option<Decimal>,
    resting_qty: Option<Decimal>,
    open_orders: Option<usize>,
}

/// Drives the production chain over one replayed event and records what it did.
pub struct ChainProbe<'a> {
    /// The production fan-out. Owned, not reimplemented.
    core: CoreHandler,
    /// The production strategy adapter, recording rather than submitting: an offline
    /// run has no venue, and an `IntentSink::Queue` with nothing draining it would
    /// stall the core on the in-flight gate after one pass.
    intent: IntentSource<SignalFeed>,
    feed: SignalFeed,
    clock: &'a ManualClock,
    /// See [`ChainOptions::poll_every_event`].
    poll_every_event: bool,
    /// See [`ChainOptions::keep_trace`].
    keep_trace: bool,
    rows: Vec<ChainRow>,
    orders: Vec<PlannedOrder>,
    cancels: Vec<PlannedCancel>,
    per_symbol_events: BTreeMap<u32, u64>,
    seq: u64,
    passes: u64,
}

impl<'a> ChainProbe<'a> {
    /// `instruments` is a **required argument**, for the reason `order_wire`'s
    /// `precision` is (ADR-0025 §2): a caller cannot re-plan a captured session without
    /// naming what it claims to know about the grid.
    ///
    /// Get it wrong in the permissive direction and the harness is quietly replaying a
    /// different program from the one it is reproducing. A live session plans under
    /// `Precision::Known` and *rounds*: an urgency-3 buy against a 64 452 ask prices at
    /// 64 774.26 and the order sent is 64 775. Replay that capture against
    /// [`InstrumentTable::unconstrained`] and `quantize` is the identity, so the
    /// replayed [`PlannedOrder`] carries 64 774.26 — a price `order_wire` would refuse
    /// outright, compared exactly by `python/axon/backtest/golden.py`, and reported as a
    /// strategy change on every rounded order in the capture. That is the bottom rung of
    /// the parity ladder failing for a reason the strategy did not cause.
    ///
    /// A captured log carries its own table since `SCHEMA_VERSION` 2 (ADR-0027), so the
    /// driver above resolves this from the log by default and only a caller that
    /// deliberately overrides it, or a log written before the bump, ever plans loosely.
    /// The argument stays required even so: `replay_chain` is one caller of many, and a
    /// probe that could be built without naming a grid is a probe that will be.
    pub fn new(
        clock: &'a ManualClock,
        records: Vec<SignalRecord>,
        instruments: Arc<InstrumentTable>,
    ) -> Self {
        let cfg = RuntimeConfig::default();
        let feed = SignalFeed::new(records);
        Self {
            core: CoreHandler::new(
                Arc::new(RwLock::new(OrderTracker::new())),
                // The live window, not `never_expires`: a replay that could not expire
                // a mark would be a replay of a different program from the one that
                // has to refuse an order when a feed goes quiet.
                Arc::new(MarkCache::with_max_age(cfg.mark_max_age_ns())),
            ),
            intent: IntentSource::new(
                feed.clone(),
                &cfg.intent,
                instruments,
                cfg.mark_max_age_ns(),
                Arc::new(HaltSwitch::new()),
                IntentSink::Record(Vec::new()),
            ),
            feed,
            clock,
            poll_every_event: false,
            keep_trace: true,
            rows: Vec::new(),
            orders: Vec::new(),
            cancels: Vec::new(),
            per_symbol_events: BTreeMap::new(),
            seq: 0,
            passes: 0,
        }
    }

    /// See [`ChainOptions::poll_every_event`].
    pub fn poll_on_every_event(mut self) -> Self {
        self.poll_every_event = true;
        self
    }

    /// See [`ChainOptions::keep_trace`]. The summary is unaffected: `trace_rows` counts
    /// the probe's own sequence, not the rows it happens to still be holding.
    pub fn without_trace(mut self) -> Self {
        self.keep_trace = false;
        self
    }

    pub fn rows(&self) -> &[ChainRow] {
        &self.rows
    }

    pub fn orders(&self) -> &[PlannedOrder] {
        &self.orders
    }

    pub fn cancels(&self) -> &[PlannedCancel] {
        &self.cancels
    }

    pub fn core(&self) -> &CoreHandler {
        &self.core
    }

    pub fn passes(&self) -> u64 {
        self.passes
    }

    fn sample(&self, sym: SymbolId) -> Sample {
        let (best_bid, best_ask) = match self.core.market().book(sym) {
            Some(b) => (b.best_bid().map(|(p, _)| p), b.best_ask().map(|(p, _)| p)),
            // No book yet: fall back to the BBO feed, which is the same fallback
            // `mid()` makes internally, so the three columns describe one view rather
            // than two feeds at different cadences.
            None => match self.core.market().bbo(sym) {
                Some(b) => (Some(b.bid_px), Some(b.ask_px)),
                None => (None, None),
            },
        };
        let mark = self.core.marks().quote(sym);
        // A poisoned tracker means our order state is unknown, and every column below
        // goes absent together. Reporting zeros would put "we hold nothing" in the
        // golden file, which is the reading that makes a breach look like a flat book —
        // and worse, two degraded runs would agree on it and pass.
        let tracker = self.core.tracker().read().ok();
        Sample {
            mid: self.core.market().mid(sym),
            best_bid,
            best_ask,
            last_trade_px: self.core.market().last_trade(sym).map(|t| t.px),
            mark_px: mark.map(|q| q.px),
            mark_ts: mark.map(|q| q.ts_event),
            position_qty: tracker.as_ref().map(|t| t.position(sym).qty),
            risk_qty: tracker.as_ref().map(|t| t.risk_position(sym).qty),
            resting_qty: tracker.as_ref().map(|t| t.resting_exposure(sym)),
            // Per symbol, filtered off the tracker's own iterator rather than
            // `open_count()`: every other column on this row describes one instrument,
            // and a global count sitting among them reads as this instrument's and
            // would make a second symbol's stale quote look like this one's.
            open_orders: tracker
                .as_ref()
                .map(|t| t.open_orders().filter(|o| o.symbol_id == sym).count()),
        }
    }

    /// Fills the tracker could not attribute — or `None` when it could not be asked.
    ///
    /// `unwrap_or(0)` here would report "every fill was attributed" for a session whose
    /// tracker had stopped answering, in the very column an operator reads to find out
    /// whether our view and the venue's still agree.
    fn orphan_fills(&self) -> Option<u64> {
        self.core.tracker().read().ok().map(|t| t.orphan_fills())
    }

    /// How the tracker attributed this event, if it was a fill.
    ///
    /// Read back out of the tracker's own index rather than re-deriving the lookup
    /// rule: a probe that reimplemented "cloid first, then venue id" would agree with
    /// `OrderTracker` on the day it was written and would stop reporting the real
    /// attribution the day the rule changed — which is the day this column matters.
    fn attribution(&self, event: &Event, orphans_before: Option<u64>) -> String {
        let Event::Exec(ExecEvent::Fill(f)) = event else {
            return String::new();
        };
        let Ok(t) = self.core.tracker().read() else {
            return "unreadable".to_string();
        };
        // A count nobody took cannot be compared with one we did. Substituting zero for
        // the missing side would make the first fill after a poisoning look like an
        // orphan, which is a different diagnosis with a different response.
        let Some(before) = orphans_before else {
            return "unreadable".to_string();
        };
        if t.orphan_fills() > before {
            // A fill for an order we have no record of still moved the position. It is
            // named here rather than left blank because "nothing happened" and
            // "something happened we cannot explain" are different sessions.
            return "orphan".to_string();
        }
        match t.order_by_id(f.order_id) {
            Some(o) => format!("{} filled={}", cloid_hex(o.cloid.get()), o.filled_qty),
            None => "unattributed".to_string(),
        }
    }

    /// One intent pass, exactly where the core loop runs one: after the event has been
    /// applied, so the position, the book and the working orders a plan is computed
    /// against are the state that event left behind (ADR-0020).
    fn pass(&mut self) -> (usize, usize) {
        if !self.poll_every_event && !self.feed.due() {
            return (0, 0);
        }
        // Nothing parks here. `IntentSource` paces its passes on the *event* clock, so
        // which signals land in one pass is a property of the log and not of how fast
        // this loop ran — the reason the harness can replay a capture at full speed and
        // still get the live session's orders. This used to `sleep` the drain interval
        // per pass, back when the runtime paced on `Instant::elapsed`; that sleep was
        // the wall clock the harness claims not to have.
        self.passes += 1;
        // `now_ms` is wall clock and paces nothing but re-opening a ring file. The
        // feed is in memory and always attached, so zero is the honest value — there is
        // no wall clock in this chain, which is the property the whole harness rests on.
        self.intent.poll(self.core.last_ts(), 0, &self.core);

        let mut orders = 0;
        let mut cancels = 0;
        for intent in self.intent.take_recorded() {
            for c in &intent.plan.cancels {
                cancels += 1;
                self.cancels.push(PlannedCancel {
                    signal_seq: intent.seq,
                    ts_event: self.core.last_ts(),
                    symbol_id: intent.symbol_id.get(),
                    target: cancel_target(c),
                });
            }
            for o in &intent.plan.orders {
                orders += 1;
                self.orders.push(planned(&intent, o, self.core.last_ts()));
            }
        }
        (orders, cancels)
    }
}

fn cancel_target(c: &CancelId) -> String {
    match c {
        CancelId::Cloid { cloid, .. } => format!("cloid:{}", cloid_hex(cloid.get())),
        CancelId::OrderId { order_id, .. } => format!("oid:{}", order_id.get()),
    }
}

fn planned(intent: &Intent, o: &OrderRequest, ts_event: Nanos) -> PlannedOrder {
    PlannedOrder {
        signal_seq: intent.seq,
        ts_event,
        symbol_id: o.symbol_id.get(),
        cloid: cloid_hex(o.cloid.get()),
        side: o.side,
        qty: o.qty,
        price: o.price,
        tif: o.tif,
        reduce_only: o.reduce_only,
    }
}

/// The one-line rendering of an order in a decision column.
///
/// Compared exactly, so it deliberately carries everything that would make it a
/// *different* order at a venue: side, size, limit, time-in-force, reduce-only and the
/// id. A flipped decision here is a different order sent for real, and no tolerance
/// makes that acceptable.
fn describe(o: &PlannedOrder) -> String {
    let px = o
        .price
        .map_or_else(|| "market".to_string(), |p| p.to_string());
    format!(
        "{:?} {} @{} {:?}{} {}",
        o.side,
        o.qty,
        px,
        o.tif,
        if o.reduce_only { " reduce-only" } else { "" },
        o.cloid
    )
}

impl EventHandler for ChainProbe<'_> {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        let orphans_before = self.orphan_fills();

        // The production fan-out, verbatim: book → marks → tracker, in one handler,
        // in one order. Everything below merely reads what it left behind.
        self.core.on_event(ts_event, event);

        // Records written up to this market moment are now readable. Advancing the
        // release clock *after* the fan-out is what makes a plan see the event that
        // justified it rather than the one before.
        self.feed.observe(ts_event);
        let (planned_orders, planned_cancels) = self.pass();
        let new_orders = &self.orders[self.orders.len() - planned_orders..];
        let new_cancels = &self.cancels[self.cancels.len() - planned_cancels..];

        let symbol = match event {
            Event::Market(m) => Some(m.symbol_id()),
            Event::Exec(x) => x.symbol_id(),
        };
        if let Some(s) = symbol {
            *self.per_symbol_events.entry(s.get()).or_default() += 1;
        }

        let s = symbol.map(|s| self.sample(s));
        let stats = self.intent.stats();
        let mut values = BTreeMap::new();
        values.insert("mid", Cell::money(s.as_ref().and_then(|s| s.mid)));
        values.insert("best_bid", Cell::money(s.as_ref().and_then(|s| s.best_bid)));
        values.insert("best_ask", Cell::money(s.as_ref().and_then(|s| s.best_ask)));
        values.insert(
            "last_trade_px",
            Cell::money(s.as_ref().and_then(|s| s.last_trade_px)),
        );
        values.insert("mark_px", Cell::money(s.as_ref().and_then(|s| s.mark_px)));
        values.insert("mark_ts", Cell::count(s.as_ref().and_then(|s| s.mark_ts)));
        values.insert(
            "position_qty",
            Cell::money(s.as_ref().and_then(|s| s.position_qty)),
        );
        values.insert("risk_qty", Cell::money(s.as_ref().and_then(|s| s.risk_qty)));
        values.insert(
            "resting_qty",
            Cell::money(s.as_ref().and_then(|s| s.resting_qty)),
        );
        values.insert(
            "open_orders",
            Cell::count(s.as_ref().and_then(|s| s.open_orders).map(|n| n as i64)),
        );
        values.insert(
            "orphan_fills",
            Cell::count(self.orphan_fills().map(|n| n as i64)),
        );
        // Not per-symbol, and never absent: this counter lives on the handler rather
        // than behind the lock that fails. The absences above are not enough on their
        // own — two degraded runs share them and compare equal — so this is what turns
        // a poisoning into a diff, on the row that lost the event.
        values.insert(
            "dropped_exec_events",
            Cell::from(self.core.dropped_exec_events()),
        );
        values.insert("signals_accepted", Cell::from(stats.accepted));
        values.insert("signals_rejected", Cell::from(stats.rejected));
        values.insert("planned_orders", Cell::from(planned_orders));
        values.insert("planned_cancels", Cell::from(planned_cancels));
        // A column added on one branch and not to `COLUMNS` would be a column some rows
        // carry and others do not, which a golden comparison reports as a structural
        // mismatch on every later run — burying the one reading that went missing.
        debug_assert!(
            values.keys().copied().eq(COLUMNS.iter().copied()),
            "the row's columns drifted from COLUMNS"
        );

        let decisions = BTreeMap::from([
            ("fill", self.attribution(event, orphans_before)),
            (
                "plan",
                new_orders
                    .iter()
                    .map(describe)
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
            (
                "cancel",
                new_cancels
                    .iter()
                    .map(|c| c.target.clone())
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
        ]);

        let row = ChainRow {
            seq: self.seq,
            ts_event,
            clock_ns: self.clock.now_ns(),
            symbol_id: symbol.map(|s| s.get()),
            kind: match event {
                Event::Market(_) => "market",
                Event::Exec(_) => "exec",
            },
            values,
            decisions,
        };
        if self.keep_trace {
            self.rows.push(row);
        }
        self.seq += 1;
    }
}

/// Replay `src` through the production chain and collect everything a golden
/// comparison looks at.
///
/// One entry point so the binary Python drives and this crate's tests exercise the
/// same driver — two entry points would be two chains, which is the failure this whole
/// module is arranged to avoid.
pub fn replay_chain(src: &ReplaySource, opts: &ChainOptions) -> (ChainSummary, Vec<ChainRow>) {
    let clock = ManualClock::new(0);
    let mut probe = ChainProbe::new(
        &clock,
        opts.signals.records().to_vec(),
        resolve_grid(src, opts).table,
    );
    if opts.poll_every_event {
        probe = probe.poll_on_every_event();
    }
    if !opts.keep_trace {
        probe = probe.without_trace();
    }
    let report = src.run(&clock, &mut probe);
    let summary = chain_summary(&probe, src, opts, &report);
    (summary, probe.rows)
}

/// Fold what the probe accumulated into the one object two runs are compared on.
///
/// Split out of [`replay_chain`] rather than inlined so a test can drive the probe
/// itself and still summarize it the way the binary does. A poisoned tracker is the
/// case that needs it: nothing inside a log can arrange one — a panic under the write
/// guard unwinds the replay thread rather than being caught — so the only way to
/// replay a session that lost its order state is to poison the lock from outside and
/// then summarize. A second summarizer written for that test would be a second answer
/// to "what did this run do".
pub fn chain_summary(
    probe: &ChainProbe<'_>,
    src: &ReplaySource,
    opts: &ChainOptions,
    report: &ReplayReport,
) -> ChainSummary {
    let mut symbols = BTreeMap::new();
    for (id, events) in probe.per_symbol_events.clone() {
        let s = probe.sample(SymbolId::new(id));
        symbols.insert(
            id,
            SymbolState {
                events,
                mid: s.mid,
                best_bid: s.best_bid,
                best_ask: s.best_ask,
                last_trade_px: s.last_trade_px,
                mark_px: s.mark_px,
                mark_ts: s.mark_ts,
                position_qty: s.position_qty,
                risk_qty: s.risk_qty,
                open_orders: s.open_orders,
            },
        );
    }

    let stats = probe.intent.stats();
    ChainSummary {
        schema: RESULT_SCHEMA,
        schema_version: RESULT_SCHEMA_VERSION,
        source: src.header().source.clone(),
        // The log's provenance, never its path: a path differs between checkouts and
        // would turn a golden comparison into a comparison of working directories.
        signal_source: (!opts.signals.is_empty()).then(|| opts.signals.header().source.clone()),
        order: match opts.order {
            ReplayOrder::EventTime => "event-time",
            ReplayOrder::AsCaptured => "as-captured",
        },
        events: report.events,
        first_ts: report.first_ts,
        last_ts: report.last_ts,
        late_arrivals: report.late_arrivals,
        // Read off the handler, not counted here: the fan-out is the only place that
        // knows an event never reached the tracker, and a harness that recounted it
        // would be reporting its own arithmetic instead of the core's.
        dropped_exec_events: probe.core.dropped_exec_events(),
        trace_rows: probe.seq,
        intent_passes: probe.passes,
        signals: SignalCounters {
            records: probe.feed.record_count() as u64,
            accepted: stats.accepted,
            rejected: stats.rejected,
            expired: stats.expired,
            superseded: stats.superseded,
            planned: stats.planned,
            no_quote: stats.no_quote,
        },
        orders: probe.orders.clone(),
        cancels: probe.cancels.clone(),
        symbols,
    }
}
