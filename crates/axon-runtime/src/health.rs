//! The one line an operator watches a session by.
//!
//! A session that is quietly broken looks exactly like a session that is quietly
//! idle: no orders, no errors, no output. Every field here exists to tell those two
//! apart — how old our view of the market is, whether the risk gate still has prices,
//! whether the venue-side protection is still armed, and whether anything we believe
//! disagrees with what the venue reports.
//!
//! [`SessionHealth`] is the shared counter block the async edges write to;
//! [`StatusSnapshot`] is the pure, printable value the core thread assembles from it.
//! The split keeps rendering testable without a session: the format is the interface
//! an operator reads under pressure, so it is worth asserting on.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use axon_core::Decimal;

/// Counters written by the edges, read by the status line.
#[derive(Debug)]
pub struct SessionHealth {
    started_ms: u64,
    reconcile_ok_ms: AtomicU64,
    reconcile_failures: AtomicU64,
    adopted_orders: AtomicU64,
    /// Orders we believe are live that the venue's snapshot did not list.
    venue_missing: AtomicU64,
    /// Symbols whose position disagrees with the venue's own accounting.
    position_drift: AtomicU64,
    /// Reconciliation events dropped because the bus was full.
    bus_full_drops: AtomicU64,
    /// Times the safety loop has halted trading.
    dms_halts: AtomicU64,
    // ── the intent path (ADR-0020) ──
    // Written by the submit task, read by the status line. Together with the core's
    // own `IntentStats` these answer the operator's question in one place: Python sent
    // N signals, we accepted M of them, and P orders actually reached the venue.
    /// Orders the venue accepted from a plan.
    intent_orders: AtomicU64,
    /// Cancels sent from a plan.
    intent_cancels: AtomicU64,
    /// Placements the venue (or a gate below us) refused.
    intent_failures: AtomicU64,
    /// Cancels that did not go through. Usually the order was already gone.
    intent_cancel_failures: AtomicU64,
    /// Orders suppressed because the session was halted. Counted separately from a
    /// refusal so an operator can tell "the venue said no" from "we stopped ourselves".
    intent_halted: AtomicU64,
    /// Acks that could not be recorded because the tracker's lock was poisoned. Every
    /// one of these is an order at the venue that our own risk view does not know about.
    intent_untracked: AtomicU64,
    /// Shutdowns that had to abort the submit task instead of waiting it out. Each one
    /// is a placement that may still reach the venue *after* the sweep read the book.
    submitter_abandoned: AtomicU64,
}

impl SessionHealth {
    pub fn new(started_ms: u64) -> Self {
        Self {
            started_ms,
            reconcile_ok_ms: AtomicU64::new(0),
            reconcile_failures: AtomicU64::new(0),
            adopted_orders: AtomicU64::new(0),
            venue_missing: AtomicU64::new(0),
            position_drift: AtomicU64::new(0),
            bus_full_drops: AtomicU64::new(0),
            dms_halts: AtomicU64::new(0),
            intent_orders: AtomicU64::new(0),
            intent_cancels: AtomicU64::new(0),
            intent_failures: AtomicU64::new(0),
            intent_cancel_failures: AtomicU64::new(0),
            intent_halted: AtomicU64::new(0),
            intent_untracked: AtomicU64::new(0),
            submitter_abandoned: AtomicU64::new(0),
        }
    }

    pub fn started_ms(&self) -> u64 {
        self.started_ms
    }

    pub fn note_reconcile_ok(&self, at_ms: u64) {
        self.reconcile_ok_ms.store(at_ms, Ordering::Relaxed);
    }

    pub fn note_reconcile_failure(&self) {
        self.reconcile_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_adopted(&self, n: u64) {
        self.adopted_orders.fetch_add(n, Ordering::Relaxed);
    }

    /// Absolute, not additive: these are properties of the latest snapshot, and a
    /// running total would keep reporting a divergence that has since been resolved.
    pub fn set_venue_missing(&self, n: u64) {
        self.venue_missing.store(n, Ordering::Relaxed);
    }

    pub fn set_position_drift(&self, n: u64) {
        self.position_drift.store(n, Ordering::Relaxed);
    }

    pub fn note_bus_full_drop(&self) {
        self.bus_full_drops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_dms_halt(&self) {
        self.dms_halts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reconcile_ok_ms(&self) -> Option<u64> {
        match self.reconcile_ok_ms.load(Ordering::Relaxed) {
            0 => None,
            v => Some(v),
        }
    }

    pub fn reconcile_failures(&self) -> u64 {
        self.reconcile_failures.load(Ordering::Relaxed)
    }

    pub fn adopted_orders(&self) -> u64 {
        self.adopted_orders.load(Ordering::Relaxed)
    }

    pub fn venue_missing(&self) -> u64 {
        self.venue_missing.load(Ordering::Relaxed)
    }

    pub fn position_drift(&self) -> u64 {
        self.position_drift.load(Ordering::Relaxed)
    }

    pub fn bus_full_drops(&self) -> u64 {
        self.bus_full_drops.load(Ordering::Relaxed)
    }

    pub fn dms_halts(&self) -> u64 {
        self.dms_halts.load(Ordering::Relaxed)
    }

    pub fn note_intent_order(&self) {
        self.intent_orders.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_intent_cancel(&self) {
        self.intent_cancels.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_intent_failure(&self) {
        self.intent_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_intent_cancel_failure(&self) {
        self.intent_cancel_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_intent_halted(&self, n: u64) {
        self.intent_halted.fetch_add(n, Ordering::Relaxed);
    }

    pub fn note_intent_untracked(&self) {
        self.intent_untracked.fetch_add(1, Ordering::Relaxed);
    }

    pub fn intent_orders(&self) -> u64 {
        self.intent_orders.load(Ordering::Relaxed)
    }

    pub fn intent_cancels(&self) -> u64 {
        self.intent_cancels.load(Ordering::Relaxed)
    }

    pub fn intent_failures(&self) -> u64 {
        self.intent_failures.load(Ordering::Relaxed)
    }

    pub fn intent_cancel_failures(&self) -> u64 {
        self.intent_cancel_failures.load(Ordering::Relaxed)
    }

    pub fn intent_halted(&self) -> u64 {
        self.intent_halted.load(Ordering::Relaxed)
    }

    pub fn intent_untracked(&self) -> u64 {
        self.intent_untracked.load(Ordering::Relaxed)
    }

    pub fn note_submitter_abandoned(&self) {
        self.submitter_abandoned.fetch_add(1, Ordering::Relaxed);
    }

    pub fn submitter_abandoned(&self) -> u64 {
        self.submitter_abandoned.load(Ordering::Relaxed)
    }
}

/// The rate budget, flattened for printing so the status line does not depend on the
/// venue crate's snapshot type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLine {
    pub ip_used: u32,
    pub ip_limit: u32,
    pub address_used: u64,
    pub address_limit: u64,
    pub throttled: bool,
}

/// The Py→Rust join, flattened for printing.
///
/// It exists to answer one question in one glance: **are we trading on the signals
/// Python thinks it sent?** A session with a dead producer, a schema-drifted producer
/// and a healthy-but-silent producer all look identical without these numbers — no
/// orders, no errors, no output.
/// Not `Copy`, since ADR-0038 put a producer's *name* on it. That is a deliberate
/// cost: the alternative is a fixed-size id an operator has to map back to a config
/// entry at 03:00, and the status line is assembled once per `status_interval_ms`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentLine {
    /// Records that passed validation, and records refused for any reason.
    pub accepted: u64,
    pub rejected: u64,
    /// …of which were past their validity window. Broken out because a rising expiry
    /// count means the producer and the consumer are drifting apart in time, which is
    /// not the same problem as a producer writing the wrong bytes.
    pub expired: u64,
    /// …and of which walked `seq` backwards, which is a restarted producer or a
    /// replayed ring (ADR-0014 §1).
    ///
    /// Broken out for the same reason `expired` is, and it is the more dangerous of
    /// the two: every record the restarted producer writes is refused until its
    /// sequence passes the old baseline, so `accepted` stays *flat* while Python's own
    /// counters climb — the two sides disagree about whether anything is happening and
    /// neither says so. "Persist `first_seq`, or restart both sides together" was an
    /// operational requirement with no evidence attached to it.
    pub stale_seq: u64,
    /// Producer loss events the `seq` stream implies.
    pub gaps: u64,
    /// Plans that produced at least one order.
    pub planned: u64,
    /// Plans that could not be priced — the strategy had an opinion and we did not
    /// have a book to express it against.
    pub no_quote: u64,
    /// Plans the instrument's own grid refused: a size finer than the lot, a notional
    /// under the venue minimum, a price the tick cannot represent. A rising count on an
    /// otherwise healthy session is a strategy emitting targets finer than the
    /// instrument — a tuning problem, and one that must not hide inside `no_order`
    /// beside the "already at target" that is the common case.
    pub precision_refusals: u64,
    /// Plans refused because we do not know the instrument's grid at all. Not a tuning
    /// problem: the session has stopped trading that instrument, and the counters that
    /// would otherwise say so are the counters that stop rising.
    pub unknown_precision: u64,
    /// Plans lost between the core and the venue because the queue was full.
    pub dropped: u64,
    /// Passes skipped because the edge had not finished the last batch, and whether
    /// that has lasted longer than a signal stays worth acting on.
    ///
    /// Both are here because a wedged submitter is the one degradation that hides
    /// behind numbers that *stop moving*: the core stops draining the ring, so
    /// `accepted` freezes and the line reads exactly like a strategy with nothing to
    /// say. One stuck HTTP call and the session is no longer trading at all.
    pub busy: u64,
    pub stalled: bool,
    /// Passes abandoned because the tracker's lock was poisoned. Our own order state is
    /// unreadable, so the session prints `open 0 flat` while holding a position and
    /// plans nothing at all.
    pub poisoned: u64,
    /// Plans that produced no order at all — overwhelmingly "already at target", which
    /// is the healthy case and is exactly why it belongs on the line: without it,
    /// `accepted` climbing while `orders` stays flat has two readings and only one of
    /// them is fine.
    pub no_order: u64,
    /// Records the reader admitted from *ahead* of the core's event clock.
    ///
    /// The ordinary case on this venue rather than a fault — the core's clock is a
    /// high-water mark over observed market data and runs 1.4-1.9 s behind wall time,
    /// while a producer stamps its decision with its own wall clock. It is on the line
    /// because the alternative reading is clock skew between two hosts, and the two are
    /// indistinguishable without knowing the number.
    pub ahead_of_clock: u64,
    /// Resting orders the sweeper asked the venue to cancel (ADR-0031), and those it
    /// had to ask about more than once.
    ///
    /// Counted since the sweeper existed and reported nowhere until now, which this
    /// module's own rule says is the same as not counted. `swept` is a number that
    /// moves in production; `resweeps` is the only evidence available on this side
    /// that the venue is accepting cancels and not acting on them.
    pub swept: u64,
    pub resweeps: u64,
    /// Fresh quotes placed for a target the strategy still holds (ADR-0036).
    pub requotes: u64,
    /// Symbols holding a position that nothing is working toward and nothing will —
    /// the state that used to be silent. Absolute, not cumulative.
    pub unquoted: u64,
    /// Symbols holding a position **no order can close**, because the delta to target
    /// is below the instrument's lot or the venue's minimum notional. Only an operator
    /// can clear it, so it is a different warning from `unquoted`.
    pub stranded: u64,
    /// Orders and cancels the venue accepted.
    pub orders: u64,
    pub cancels: u64,
    /// Submits that failed, and orders the halt refused.
    pub failures: u64,
    pub halted: u64,
    /// Acks we could not record. Each one is an order the venue has and we do not.
    pub untracked: u64,
    /// Whether **every** declared producer's ring is open right now.
    pub attached: bool,
    /// How many are not, and how many there are. `1` producer is every session written
    /// before ADR-0038.
    pub detached: u32,
    pub producers: u32,
    /// Instruments planned from a sum of two or more strategies' claims.
    ///
    /// Zero on every single-producer session by construction, which is what makes a
    /// non-zero value here mean the netting path is live rather than merely configured.
    pub netted: u64,
    /// Claims kept even though their author had gone silent past its window — exposure
    /// nobody is currently speaking for — and claims driven to zero by that silence.
    pub silent_held: u64,
    pub silent_flat: u64,
    /// Records refused because another strategy already claims the instrument, or
    /// because their producer never declared it.
    pub overlap_refused: u64,
    pub out_of_scope: u64,
    /// Netted targets that dropped a `price_band` the contributors disagreed about.
    pub band_dropped: u64,
    /// Passes on which the allocation bound: a producer scaled to its own share, or
    /// everything scaled to fit the portfolio, and the factor the last one applied.
    pub strategy_scaled: u64,
    pub portfolio_scaled: u64,
    pub portfolio_scale_bps: u32,
    /// Instruments a breadth cap kept the session out of, this pass. Absolute.
    pub breadth_denied: u32,
    /// Passes on which an allocation could not be computed because a claimed instrument
    /// had no usable mark, so **nothing was scaled**. The gate still refuses new
    /// exposure; this says the sizing did not silently shrink instead.
    pub alloc_unpriced: u64,
    /// One line per declared producer. Empty on a session with one, because the
    /// aggregate above already says everything there is to say about it.
    pub strategies: Vec<StrategyStatus>,
}

/// One producer's own numbers, for a session running several.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyStatus {
    pub name: String,
    pub attached: bool,
    pub accepted: u64,
    pub rejected: u64,
    pub expired: u64,
    pub stale_seq: u64,
    pub claims: usize,
    /// Milliseconds of event time since it last stated anything, or `None` if it never
    /// has.
    pub silent_ms: Option<u64>,
    /// Whether that is past its declared window.
    pub silent: bool,
    pub out_of_scope: u64,
}

/// The Rust→Python market-data ring, flattened for printing.
///
/// One question again: **is Python seeing the market we are trading on?** A publisher
/// whose ring is full and a publisher with nothing to say produce the same silence at
/// the reader, and only one of them means the feature history has holes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MdLine {
    /// Slices the ring accepted.
    pub published: u64,
    /// Slices refused because the ring was full. Each is a `seq` gap the Python reader
    /// also sees — this number is here for the case where nobody is reading at all.
    pub dropped: u64,
    /// Updates that moved nothing the record carries. Not a loss; a rising count beside
    /// a flat `published` just means the feed is busier than the top of book.
    pub coalesced: u64,
    /// Updates skipped because a value could not be held exactly at the wire scale.
    pub unrepresentable: u64,
    /// Slices sent with no quote because every feed carrying one had gone quiet. The
    /// record has no field for the quote's own age, so this is the only place a dead
    /// `bbo` subscription behind a live trade feed can be seen at all.
    pub stale_quote: u64,
    /// Ring depth and size: the early warning `dropped` is the late one for.
    pub queued: u64,
    pub capacity: u64,
    /// Closed bars written to the *second* ring, and bars it refused.
    ///
    /// Kept apart from `published`/`dropped` because a drop means something different
    /// on each ring, and averaging them would lose the distinction. The slice ring
    /// fills within seconds of a reader stalling, so a drop there is ambiguous between
    /// "slow" and "dead". The bar ring takes one record a *minute*: filling it means
    /// nothing has read it for hours, and there is no reading of that which is a slow
    /// consumer. Counted and not surfaced would leave that as a one-shot `eprintln!`
    /// scrolled off the top of a soak's log.
    pub bars_published: u64,
    pub bars_dropped: u64,
}

/// The session recording, flattened for printing.
///
/// One question: **will there be an artifact at the end of this?** A recording that
/// stopped at 02:00 and a recording that is running produce the same silence, and only
/// one of them leaves a log worth replaying. The queue depth is here for the same reason
/// the market-data ring's is: it is the number that moves *before* the stop, which is
/// the only point at which an operator can still do something about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureLine {
    pub events: u64,
    pub signals: u64,
    pub bytes: u64,
    /// Records the recording is short by, counted from the moment it stopped. A number
    /// rather than an inference from the file size, because the two disagree and only
    /// one of them is the truth.
    pub missed: u64,
    pub queued: u64,
    pub capacity: u64,
    /// `None` while the recording is still running.
    pub stopped: Option<crate::capture::CaptureStop>,
}

/// Bytes as an operator reads them. KiB below a megabyte, because a test capture
/// rendering as `0MiB` reads like a recording that is not happening.
fn size(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if bytes >= MIB {
        format!("{}MiB", bytes / MIB)
    } else {
        format!("{}KiB", bytes / 1024)
    }
}

/// Everything the status line prints, as plain values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusSnapshot {
    /// e.g. `"sandbox/testnet"` or `"backtest/offline"`.
    pub mode: String,
    pub uptime_s: u64,
    pub events: u64,
    pub bus_len: usize,
    /// Age of the newest event, in ms. `None` offline, where wall-clock age is
    /// meaningless.
    pub data_lag_ms: Option<u64>,
    pub marks_fresh: usize,
    pub marks_total: usize,
    /// Named symbols with an expired price — named, because "1 stale" does not tell
    /// an operator which instrument stopped being tradeable.
    pub stale_marks: Vec<String>,
    pub open_orders: usize,
    pub positions: Vec<(String, Decimal)>,
    pub orphan_fills: u64,
    pub adopted_orders: u64,
    pub venue_missing: u64,
    pub position_drift: u64,
    pub dropped_exec_events: u64,
    /// Candles received while the core still has no event clock. Non-zero means this
    /// session's only market data is bars, and a bar's timestamp is a close time rather
    /// than a moment anyone observed — so nothing has started the clock and no intent
    /// pass will run. See [`crate::handler::CoreHandler::bars_without_a_clock`].
    pub bars_without_a_clock: u64,
    pub bus_full_drops: u64,
    /// Shutdowns that had to abort the submit task rather than wait for it. Only ever
    /// non-zero on the closing line, and it is the one thing that line has to say: an
    /// order the aborted task had already sent can arrive at the venue after the sweep
    /// read the book, so the session may be exiting with exposure it does not know about.
    pub submitter_abandoned: u64,
    /// Whether this session is supposed to have a dead-man's switch at all. An
    /// offline run has no venue to arm one at, and reporting that as a fault would
    /// train an operator to ignore the one warning that matters most.
    pub dms_expected: bool,
    /// Protection remaining, in seconds. `None` when the switch is not armed.
    pub dms_remaining_s: Option<u64>,
    pub dms_failures: u64,
    /// Times protection drained into the halt band or past it with **nothing having
    /// failed** — the safety loop stopped running (frozen, descheduled, wedged) and the
    /// venue-side deadline kept counting down without it.
    ///
    /// Its own number because it is the only one that survives. The re-arm that follows
    /// a stall succeeds, and in succeeding it pushes `dms_remaining_s` back to full,
    /// clears the halt within microseconds and leaves `dms_failures` at the zero it
    /// never left. A soak watched two consecutive status lines read `dms 0s` — the venue
    /// switch had actually fired — and then `dms 55s`, with no warning on either.
    pub dms_lapses: u64,
    pub rate: Option<RateLine>,
    /// Age of the last successful reconciliation poll, in seconds.
    pub reconcile_age_s: Option<u64>,
    pub reconcile_failures: u64,
    pub halted: bool,
    /// The intent path. `None` when the session has no intent source at all, which is
    /// a deliberate read-only configuration and not a fault — printing zeros there
    /// would read as "the strategy is quiet" rather than "there is no strategy".
    pub intent: Option<IntentLine>,
    /// The market-data ring. `None` when the session publishes no ring, for the same
    /// reason as `intent`: not publishing is a configuration, not a degradation.
    pub md: Option<MdLine>,
    /// The session recording. `None` when nobody asked for one — again absence rather
    /// than zeros, because "not recording" and "recording nothing" need different
    /// responses and only one of them is an emergency.
    pub capture: Option<CaptureLine>,
    /// The money view (ADR-0036). `None` for a session with no venue to have a P&L at
    /// — an offline run's positions are arithmetic, and printing a bottom line for
    /// them would put a number where a reader expects an account.
    pub pnl: Option<crate::pnl::PnlSnapshot>,
    /// The operator's two P&L alarms, carried alongside the snapshot rather than
    /// baked into it: [`crate::pnl::snapshot`] is a pure view of what happened, and
    /// what counts as too much is a policy the config states.
    pub pnl_limits: crate::config::PnlConfig,
    /// Latency against declared budgets. Always present in a session that measures —
    /// the *block* is suppressed when nothing has been sampled, which is a different
    /// statement from "this session does not measure latency".
    pub latency: crate::latency::LatencySnapshot,
    /// The loss-based kill switch, if it has tripped.
    ///
    /// Distinct from every other money field on this line, and the distinction is the
    /// whole reason it is a separate field: `pnl_limits` are alarms that warn, and this
    /// is a **gate that has already refused an order**. An operator reading the line has
    /// to be able to tell "we are losing money" from "we have stopped adding to it".
    pub loss: Option<axon_execution::LossBreach>,
    /// The bounds in force, so a session that declared none says so by omission rather
    /// than by looking identical to one that declared some and is inside them.
    pub loss_limits: axon_execution::LossLimits,
    /// Why the day's baseline is not on disk, when a daily bound was declared anyway.
    /// A daily bound reduced to a session bound is the quiet kind of wrong.
    pub daybook_fault: Option<crate::daybook::DayBookFault>,
}

impl StatusSnapshot {
    /// The problems worth waking someone for, most serious first.
    ///
    /// Assembled separately from the body so the line ends with what is wrong rather
    /// than requiring an operator to diff two dozen numbers against their memory.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.dms_expected && self.dms_remaining_s.is_none() {
            w.push("DMS NOT ARMED".to_string());
        }
        // Second only to having no switch at all, and ranked above `HALTED` because a
        // halt is a state an operator can see for themselves while this is a state that
        // has already ended: the successful re-arm that follows a stall repairs every
        // other field on the line. Cumulative, so it stays up after the recovery — a
        // session that lost protection and got it back still lost protection, and one
        // occurrence means the venue-side switch may have fired and cancelled everything
        // that was resting.
        if self.dms_lapses > 0 {
            w.push(format!("DMS PROTECTION LAPSED {}", self.dms_lapses));
        }
        // Above `HALTED`, and the ranking is an argument rather than a preference. Both
        // say orders are being refused; only this one says *why*, and only this one is
        // permanent. A halt is something the dead-man's switch or a shutdown raised and
        // something a recovery clears, so an operator seeing it knows to wait. This
        // clears on nothing but a restart, and the position it is holding open is the
        // one that caused it — which makes it the line that has to be acted on, not
        // waited out.
        if let Some(b) = &self.loss {
            w.push(format!("LOSS LIMIT TRIPPED - {b} - DE-RISK ONLY"));
        }
        if self.halted {
            w.push("HALTED".to_string());
        }
        // A daily bound whose baseline is not on disk is a session bound wearing a day's
        // name: every number on this line is right and the guarantee is not the one the
        // operator configured. `RuntimeConfig::validate` refuses the case where no path
        // was given at all, so reaching here means a path was given and the disk said no
        // — which is a running session that must keep running and must not be believed
        // about its day.
        if self.loss_limits.day > Decimal::ZERO {
            if let Some(f) = &self.daybook_fault {
                w.push(format!("DAILY LOSS BOUND NOT PERSISTED ({f})"));
            }
        }
        // A session up for a minute that has seen *nothing* — no events, no marks — is
        // a dead feed, not a quiet market. Every other warning here is computed from
        // data that arrived, so a socket that never came up trips none of them:
        // `data_lag_ms` is suppressed while `last_ts` is 0, and `stale_symbols()` over
        // an empty `MarkCache` is empty. Without this the line prints
        // `ev 0 | marks 0/0 | ... | OK` for hours.
        //
        // A minute, so a session that is merely still starting does not alarm: the WS
        // handshake, the universe fetch and the first snapshot all land well inside it.
        if self.uptime_s >= 60 && self.events == 0 && self.marks_total == 0 {
            w.push(format!("NO MARKET DATA {}s", self.uptime_s));
        }
        // Bars are arriving and nothing else is. A candle's `ts_event` is its close
        // time — computed from the bar's identity, not a moment anyone observed — so it
        // cannot start the event clock, and with the clock at zero ADR-0020 §2 runs no
        // intent pass at all. The warning above cannot catch this: events *are*
        // arriving, `marks_total` may be non-zero, `data_lag_ms` is suppressed rather
        // than wrong, and the whole line otherwise reads as a strategy with no opinion.
        // The fix is a quote feed — `bbo` or `l2Book` — not a longer wait.
        if self.bars_without_a_clock > 0 {
            w.push(format!("BARS BUT NO CLOCK {}", self.bars_without_a_clock));
        }
        if self.submitter_abandoned > 0 {
            w.push("SUBMITTER ABANDONED".to_string());
        }
        if self.dropped_exec_events > 0 {
            w.push(format!("EXEC EVENTS DROPPED {}", self.dropped_exec_events));
        }
        if !self.stale_marks.is_empty() {
            w.push(format!("STALE MARKS {}", self.stale_marks.join(",")));
        }
        if self.position_drift > 0 {
            w.push(format!("POSITION DRIFT {}", self.position_drift));
        }
        if self.venue_missing > 0 {
            w.push(format!("ORDERS UNCONFIRMED {}", self.venue_missing));
        }
        if self.orphan_fills > 0 {
            w.push(format!("ORPHAN FILLS {}", self.orphan_fills));
        }
        if self.bus_full_drops > 0 {
            w.push(format!("BUS FULL x{}", self.bus_full_drops));
        }
        if self.rate.is_some_and(|r| r.throttled) {
            w.push("RATE THROTTLED".to_string());
        }
        if let Some(i) = self.intent.as_ref() {
            // Our own order state being unreadable outranks the rest: the session goes
            // on printing `open 0 flat` while holding a position, and plans nothing
            // against it, so every other number here is describing a session that
            // stopped some time ago.
            if i.poisoned > 0 {
                w.push(format!("POISONED TRACKER {}", i.poisoned));
            }
            // Nothing has come back from the edge for longer than a signal stays worth
            // acting on. The core stops draining, so the counters *below* freeze rather
            // than rise: without this the line reads as a quiet strategy.
            if i.stalled {
                w.push("INTENT STALLED".to_string());
            }
            // The ring being gone is the difference between a strategy with no opinion
            // and a strategy nobody is listening to. With several producers the *count*
            // is load-bearing and so are the names: "detached" without them is a fault
            // nobody can act on, and one dead strategy among four looks on every other
            // number exactly like four healthy ones trading a little less.
            if !i.attached {
                if i.producers > 1 {
                    let names: Vec<&str> = i
                        .strategies
                        .iter()
                        .filter(|s| !s.attached)
                        .map(|s| s.name.as_str())
                        .collect();
                    w.push(format!(
                        "SIGNAL RING DETACHED {}/{} ({})",
                        i.detached,
                        i.producers,
                        names.join(", ")
                    ));
                } else {
                    w.push("SIGNAL RING DETACHED".to_string());
                }
            }
            // A producer that is attached and has stopped speaking. Its claims are
            // either being held with no author — exposure nobody is currently deciding
            // about — or being driven to zero, and both are things an operator has to be
            // told rather than left to infer from a target that stopped moving.
            //
            // Named, because "a strategy is silent" on a four-strategy session sends
            // somebody to read four transcripts.
            let quiet: Vec<String> = i
                .strategies
                .iter()
                .filter(|s| s.silent)
                .map(|s| match s.silent_ms {
                    Some(ms) => format!("{} {}s", s.name, ms / 1_000),
                    None => s.name.clone(),
                })
                .collect();
            if !quiet.is_empty() {
                w.push(format!("STRATEGY SILENT {}", quiet.join(", ")));
            }
            // A record for an instrument its producer never declared. The declared
            // universe is what lets `RuntimeConfig::validate` catch two strategies on one
            // position at startup, so a record outside it is an overlap that escaped the
            // one check designed to find it.
            if i.out_of_scope > 0 {
                w.push(format!("SIGNAL OUT OF SCOPE {}", i.out_of_scope));
            }
            // The second claim on an instrument, refused. Uppercase because it is the
            // shape of two strategies fighting over one position, which is what
            // `overlap = "exclusive"` exists to make impossible rather than merely rare.
            if i.overlap_refused > 0 {
                w.push(format!("OVERLAP REFUSED {}", i.overlap_refused));
            }
            // The portfolio bound is binding and the session is working toward less than
            // its strategies asked for. Not a fault — it is the bound doing its job — and
            // it must be visible, because the alternative reading of every position being
            // smaller than the strategy intended is a strategy that has changed its mind.
            if i.portfolio_scale_bps > 0 && i.portfolio_scale_bps < 10_000 {
                w.push(format!("PORTFOLIO SCALED {}%", i.portfolio_scale_bps / 100));
            }
            if i.breadth_denied > 0 {
                w.push(format!("PORTFOLIO BREADTH -{}", i.breadth_denied));
            }
            // Nothing was scaled, because a claimed instrument had no usable mark. The
            // gate still refuses new exposure on the same book; this says the *sizing*
            // did not silently shrink every position because one feed went quiet.
            if i.alloc_unpriced > 0 {
                w.push(format!("ALLOCATION UNPRICED {}", i.alloc_unpriced));
            }
            // A netted target that dropped a price band its contributors disagreed
            // about. A risk bound that stopped applying must never be silent.
            if i.band_dropped > 0 {
                w.push(format!("PRICE BAND DROPPED {}", i.band_dropped));
            }
            // A ring that is attached and whose every record is being refused is the
            // *same* symptom with a different cause, and it is the one nobody looks for:
            // the producer restarted, its `seq` rewound, and the reader will go on
            // refusing until the sequence passes the old baseline. Uppercase because
            // this is never a condition a healthy session passes through — a single
            // occurrence means the two sides were restarted independently.
            if i.stale_seq > 0 {
                w.push(format!("SIGNAL SEQ REWOUND {}", i.stale_seq));
            }
            // A position nothing is working toward and nothing will: the state the
            // first live sweep left behind for twelve minutes with the status line
            // reporting it *accurately* and nobody able to see it. Ranked above the
            // submit failures below because those are events and this is a condition
            // that persists until an operator acts.
            if i.unquoted > 0 {
                w.push(format!("UNQUOTED TARGET {}", i.unquoted));
            }
            // Ranked with `unquoted` because it is the same operator question — is
            // anything open that nothing will act on — with a different answer to
            // "who fixes it". Nothing in this process can.
            if i.stranded > 0 {
                w.push(format!("STRANDED POSITION {}", i.stranded));
            }
            if i.untracked > 0 {
                w.push(format!("ORDERS UNTRACKED {}", i.untracked));
            }
            if i.dropped > 0 {
                w.push(format!("INTENTS DROPPED {}", i.dropped));
            }
            if i.failures > 0 {
                w.push(format!("SUBMIT FAILURES {}", i.failures));
            }
        }
        if let Some(m) = self.md {
            // Python infers the same loss from the `seq` gaps, but only if something is
            // reading. This is the number that shows the feed is being dropped when
            // nothing is.
            if m.dropped > 0 {
                w.push(format!("MD SLICES DROPPED {}", m.dropped));
            }
            // Rare enough to be a bug rather than a condition: a venue price the wire's
            // 10^-8 scale cannot hold exactly means the contract and the venue disagree.
            if m.unrepresentable > 0 {
                w.push(format!("MD SLICES UNREPRESENTABLE {}", m.unrepresentable));
            }
            // Python is being fed slices with no top of book at all, which it cannot
            // distinguish from an instrument nobody has quoted yet. Almost always a
            // subscription a reconnect failed to restore.
            if m.stale_quote > 0 {
                w.push(format!("MD QUOTE STALE {}", m.stale_quote));
            }
        }
        if let Some(c) = self.capture {
            // The failure a recording has to shout about: from here on the log is a
            // prefix, and a prefix that nobody was told about replays green over a
            // session that ended early — which is the worst artifact this produces.
            if let Some(reason) = c.stopped {
                w.push(format!("CAPTURE STOPPED ({reason}) MISSING {}", c.missed));
            }
        }
        if let Some(p) = &self.pnl {
            // Above everything else the money block can say, because it is the reason
            // the rest of it says nothing.
            if !p.readable {
                w.push("PNL UNREADABLE".to_string());
            }
            // Ranked above the loss alarm below it, and that ordering is the whole
            // reason this warning exists: a position nobody can price is a position
            // whose loss nobody can see, so the quieter alarm is quiet *because* of
            // this one. Naming the symbol, for the same reason `STALE MARKS` does.
            if !p.unpriced.is_empty() {
                w.push(format!("POSITION UNPRICED {}", p.unpriced.join(",")));
            }
            // A magnitude in the config, compared against a signed net figure: the
            // session warns when what it has done is *worse* than minus the bound.
            if self.pnl_limits.max_session_loss > Decimal::ZERO {
                if let Some(net) = p.net() {
                    if net < -self.pnl_limits.max_session_loss {
                        w.push(format!(
                            "SESSION LOSS {net:.4} PAST {:.4}",
                            self.pnl_limits.max_session_loss
                        ));
                    }
                }
            }
            // On its absolute value, because either side may be the larger and a
            // one-sided check would miss the more alarming direction: our accounting
            // claiming a profit the venue has not paid.
            if self.pnl_limits.equity_drift_alarm > Decimal::ZERO {
                if let Some(d) = p.drift {
                    if d.abs() > self.pnl_limits.equity_drift_alarm {
                        w.push(format!("PNL DRIFT {d:+.4}"));
                    }
                }
            }
        }
        for over in self.latency.over_budget() {
            w.push(format!("LATENCY {over}"));
        }
        w
    }
}

fn hms(seconds: u64) -> String {
    format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60
    )
}

impl fmt::Display for StatusSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "axon {} {} | ev {} bus {}",
            hms(self.uptime_s),
            self.mode,
            self.events,
            self.bus_len
        )?;
        if let Some(lag) = self.data_lag_ms {
            write!(f, " lag {lag}ms")?;
        }
        write!(
            f,
            " | marks {}/{} | open {}",
            self.marks_fresh, self.marks_total, self.open_orders
        )?;
        if self.positions.is_empty() {
            write!(f, " flat")?;
        } else {
            for (sym, qty) in &self.positions {
                write!(f, " {sym}{qty:+}")?;
            }
        }
        if self.dms_expected {
            match self.dms_remaining_s {
                Some(s) => write!(f, " | dms {s}s")?,
                None => write!(f, " | dms -")?,
            }
            // One parenthesised group so an operator reads the switch's whole history in
            // one glance, and so the shape of a session with failures but no stalls is
            // unchanged. `late` is the count of stalls, not a duration: the duration is
            // gone by the time this prints.
            if self.dms_failures > 0 || self.dms_lapses > 0 {
                write!(f, " (")?;
                if self.dms_failures > 0 {
                    write!(f, "fail {}", self.dms_failures)?;
                    if self.dms_lapses > 0 {
                        write!(f, " ")?;
                    }
                }
                if self.dms_lapses > 0 {
                    write!(f, "late {}", self.dms_lapses)?;
                }
                write!(f, ")")?;
            }
        }
        if let Some(r) = self.rate {
            write!(
                f,
                " | rate ip {}/{} addr {}/{}",
                r.ip_used, r.ip_limit, r.address_used, r.address_limit
            )?;
        }
        match self.reconcile_age_s {
            Some(s) => write!(f, " | rec {s}s")?,
            None if self.reconcile_failures > 0 || self.rate.is_some() => {
                write!(f, " | rec never")?
            }
            None => {}
        }
        if self.reconcile_failures > 0 {
            write!(f, " (fail {})", self.reconcile_failures)?;
        }
        // Accepted/refused first, then what actually reached the venue: the gap between
        // those two numbers is the whole diagnosis when a strategy "is not trading".
        if let Some(i) = self.intent.as_ref() {
            write!(f, " | sig {}/{}", i.accepted, i.rejected)?;
            if i.gaps > 0 {
                write!(f, " gap {}", i.gaps)?;
            }
            write!(f, " sent {}+{}c", i.orders, i.cancels)?;
            if i.no_quote > 0 {
                write!(f, " noquote {}", i.no_quote)?;
            }
            if i.precision_refusals > 0 {
                write!(f, " prec {}", i.precision_refusals)?;
            }
            // Uppercase, like the other loud states: a session that has quietly stopped
            // being able to trade one instrument otherwise reads exactly like a quiet
            // strategy, because the only evidence is counters that no longer move.
            if i.unknown_precision > 0 {
                write!(f, " NOSPEC {}", i.unknown_precision)?;
            }
            if i.halted > 0 {
                write!(f, " held {}", i.halted)?;
            }
            // Printed rather than only warned on, because it is the number that moves
            // *before* the stall: passes piling up behind an edge that has stopped
            // answering is the shape of a submitter about to wedge.
            if i.busy > 0 {
                write!(f, " busy {}", i.busy)?;
            }
            // Only once non-zero. A healthy strategy sweeps nothing and re-quotes
            // nothing, and a permanent `swept 0 requote 0` on every line is three
            // fields an operator learns to skip past — which is how the field that
            // finally moves gets missed.
            if i.swept > 0 {
                write!(f, " swept {}", i.swept)?;
                if i.resweeps > 0 {
                    write!(f, " (re {})", i.resweeps)?;
                }
            }
            if i.requotes > 0 {
                write!(f, " requote {}", i.requotes)?;
            }
            // Everything below is silent on a single-producer session, which is every
            // session written before ADR-0038: a permanent `net 0 scaled 0` on every
            // line is fields an operator learns to skip past, and that is how the one
            // that finally moves gets missed.
            if i.producers > 1 {
                write!(f, " | strat {}", i.producers)?;
                if i.detached > 0 {
                    write!(f, " DETACHED {}", i.detached)?;
                }
                if i.netted > 0 {
                    write!(f, " net {}", i.netted)?;
                }
                if i.silent_held > 0 || i.silent_flat > 0 {
                    write!(f, " silent {}h/{}f", i.silent_held, i.silent_flat)?;
                }
                // The factor rather than a flag: "the portfolio is binding" and "the
                // portfolio is binding at 3 % of what the strategies asked for" are
                // different sessions and only the second is a misconfiguration.
                if i.portfolio_scaled > 0 {
                    write!(f, " alloc {}bp", i.portfolio_scale_bps)?;
                }
                if i.strategy_scaled > 0 {
                    write!(f, " clamp {}", i.strategy_scaled)?;
                }
                if i.breadth_denied > 0 {
                    write!(f, " breadth -{}", i.breadth_denied)?;
                }
            }
        }
        // The ring depth rather than a rate: it is the number that moves *before* the
        // drops start, which is the only point at which an operator can still act.
        if let Some(m) = self.md {
            write!(f, " | md {} q {}/{}", m.published, m.queued, m.capacity)?;
            if m.coalesced > 0 {
                write!(f, " coal {}", m.coalesced)?;
            }
            // Only once there are bars at all: a session with no candle subscription
            // would otherwise print a permanent `bars 0`, which reads as a feed that has
            // stopped rather than one nobody asked for. `BARDROP` is uppercase for the
            // reason above — on a once-a-minute ring it cannot mean a slow reader.
            if m.bars_published > 0 || m.bars_dropped > 0 {
                write!(f, " bars {}", m.bars_published)?;
                if m.bars_dropped > 0 {
                    write!(f, " BARDROP {}", m.bars_dropped)?;
                }
            }
        }
        // Events and signals separately: a recording with a healthy event count and no
        // signals is one whose replay will re-observe the session and re-decide nothing,
        // and that is not visible from a single total.
        if let Some(c) = self.capture {
            write!(
                f,
                " | cap {}+{}s {} q {}/{}",
                c.events,
                c.signals,
                size(c.bytes),
                c.queued,
                c.capacity
            )?;
        }
        // Adoptions are worth showing but are not a fault: a restart legitimately finds
        // its predecessor's orders resting, and seeing the count is how an operator
        // confirms the recovery happened at all.
        if self.adopted_orders > 0 {
            write!(f, " adopted {}", self.adopted_orders)?;
        }
        // The money, and then the clock. Last of the blocks and immediately before the
        // warnings, because they are the two an operator scrolls to the end of an
        // hour's log to read — and because everything above them is the machinery that
        // produced them.
        if let Some(p) = &self.pnl {
            write!(f, " | {p}")?;
        }
        if self.latency.any_samples() {
            write!(f, " | {}", self.latency)?;
        }
        let warnings = self.warnings();
        if warnings.is_empty() {
            write!(f, " | OK")
        } else {
            write!(f, " | {}", warnings.join(" · "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn healthy() -> StatusSnapshot {
        StatusSnapshot {
            mode: "sandbox/testnet".into(),
            uptime_s: 3_912,
            events: 12_043,
            bus_len: 0,
            data_lag_ms: Some(12),
            marks_fresh: 2,
            marks_total: 2,
            stale_marks: vec![],
            open_orders: 3,
            positions: vec![("BTC".into(), dec!(0.021))],
            orphan_fills: 0,
            adopted_orders: 0,
            venue_missing: 0,
            position_drift: 0,
            dropped_exec_events: 0,
            bars_without_a_clock: 0,
            bus_full_drops: 0,
            submitter_abandoned: 0,
            dms_expected: true,
            dms_remaining_s: Some(47),
            dms_failures: 0,
            dms_lapses: 0,
            rate: Some(RateLine {
                ip_used: 24,
                ip_limit: 1200,
                address_used: 812,
                address_limit: 10_000,
                throttled: false,
            }),
            reconcile_age_s: Some(3),
            reconcile_failures: 0,
            halted: false,
            // A live session that is trading: a small profit, priced, with the venue
            // agreeing to within a funding payment.
            pnl: Some(crate::pnl::PnlSnapshot {
                realized: dec!(0.0288),
                // The venue replayed 22 fills at subscribe; this session's own four are
                // what the line reports, and the gap is named rather than hidden.
                realized_all: dec!(0.0449),
                fees: dec!(0.0117),
                fees_all: dec!(0.1148),
                fills_all: 26,
                venue_closed_pnl: dec!(0.0288),
                unrealized: Some(dec!(-0.0021)),
                unpriced: vec![],
                gross_exposure: dec!(19.44),
                fills: 4,
                maker_fills: 4,
                taker_fills: 0,
                equity: Some(dec!(998.9132)),
                equity_at_start: Some(dec!(998.8965)),
                equity_delta: Some(dec!(0.0167)),
                drift: Some(dec!(-0.0017)),
                readable: true,
            }),
            pnl_limits: crate::config::PnlConfig::default(),
            latency: crate::latency::LatencyBook::undeclared().snapshot(),
            loss: None,
            loss_limits: axon_execution::LossLimits::default(),
            daybook_fault: None,
            intent: Some(IntentLine {
                accepted: 412,
                rejected: 3,
                expired: 3,
                stale_seq: 0,
                gaps: 0,
                planned: 118,
                no_order: 294,
                ahead_of_clock: 412,
                swept: 0,
                resweeps: 0,
                requotes: 0,
                unquoted: 0,
                stranded: 0,
                no_quote: 0,
                precision_refusals: 0,
                unknown_precision: 0,
                dropped: 0,
                busy: 0,
                stalled: false,
                poisoned: 0,
                orders: 118,
                cancels: 96,
                failures: 0,
                halted: 0,
                untracked: 0,
                attached: true,
                detached: 0,
                producers: 1,
                netted: 0,
                silent_held: 0,
                silent_flat: 0,
                overlap_refused: 0,
                out_of_scope: 0,
                band_dropped: 0,
                strategy_scaled: 0,
                portfolio_scaled: 0,
                portfolio_scale_bps: 0,
                breadth_denied: 0,
                alloc_unpriced: 0,
                strategies: Vec::new(),
            }),
            md: Some(MdLine {
                published: 9_004,
                dropped: 0,
                coalesced: 3_039,
                unrepresentable: 0,
                stale_quote: 0,
                queued: 31,
                capacity: 4_096,
                bars_published: 0,
                bars_dropped: 0,
            }),
            capture: Some(CaptureLine {
                events: 12_043,
                signals: 412,
                bytes: 43 * 1024 * 1024,
                missed: 0,
                queued: 4,
                capacity: 16_384,
                stopped: None,
            }),
        }
    }

    #[test]
    fn a_healthy_session_renders_one_line_ending_in_ok() {
        let line = healthy().to_string();
        assert!(!line.contains('\n'), "the status line must stay one line");
        assert!(line.starts_with("axon 1:05:12 sandbox/testnet"), "{line}");
        assert!(line.contains("marks 2/2"), "{line}");
        assert!(line.contains("BTC+0.021"), "{line}");
        assert!(line.contains("dms 47s"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_stale_mark_is_named_not_counted() {
        // "1 stale" does not tell an operator which instrument just became untradeable.
        let mut s = healthy();
        s.marks_fresh = 1;
        s.stale_marks = vec!["ETH".into()];
        let line = s.to_string();
        assert!(line.contains("marks 1/2"), "{line}");
        assert!(line.contains("STALE MARKS ETH"), "{line}");
        assert!(!line.ends_with("OK"));
    }

    #[test]
    fn a_session_that_never_received_an_event_is_not_printed_as_ok() {
        // The one failure every other warning here is blind to. Each of them is derived
        // from data that *arrived*, so a socket that never came up trips none: there is
        // no lag to measure and no mark to go stale. Half an hour of `ev 0 ... | OK` is
        // the worst line this file can print, because it is the one an operator reads
        // and then goes back to sleep.
        let mut s = healthy();
        s.uptime_s = 1_800;
        s.events = 0;
        s.marks_fresh = 0;
        s.marks_total = 0;
        s.data_lag_ms = None;
        let line = s.to_string();
        assert!(line.contains("NO MARKET DATA 1800s"), "{line}");
        assert!(!line.ends_with("OK"), "{line}");
    }

    #[test]
    fn a_session_that_has_only_just_started_is_still_ok() {
        // The companion to the above: startup is not an alarm. The WS handshake, the
        // universe fetch and the first snapshot all land inside the first minute, and a
        // warning that fires every time the process boots is one nobody reads.
        let mut s = healthy();
        s.uptime_s = 5;
        s.events = 0;
        s.marks_fresh = 0;
        s.marks_total = 0;
        s.data_lag_ms = None;
        let line = s.to_string();
        assert!(!line.contains("NO MARKET DATA"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_session_fed_nothing_but_bars_does_not_read_as_a_quiet_strategy() {
        // The gap `NO MARKET DATA` cannot cover, and the one the clock rule opens. A
        // candle's `ts_event` is its close time, so it never starts the event clock —
        // and ADR-0020 §2 runs no intent pass while that clock is zero. Events are
        // arriving, so `ev` climbs; `data_lag_ms` is honestly absent rather than wrong;
        // and every remaining number describes a healthy session that will never trade.
        let mut s = healthy();
        s.events = 940;
        s.bars_without_a_clock = 940;
        s.data_lag_ms = None;
        let line = s.to_string();
        assert!(line.contains("BARS BUT NO CLOCK 940"), "{line}");
        assert!(
            !line.contains("NO MARKET DATA"),
            "events did arrive: {line}"
        );
        assert!(!line.ends_with("OK"), "{line}");

        // And it is a state, not a tally: one quote starts the clock and the session is
        // healthy again, so the handler reports zero and the line must go quiet with it.
        s.bars_without_a_clock = 0;
        s.data_lag_ms = Some(4);
        assert!(!s.to_string().contains("BARS BUT NO CLOCK"));
    }

    #[test]
    fn an_unarmed_dead_mans_switch_is_the_first_thing_reported() {
        // The most serious state a running session can be in: everything else is a
        // degraded view of the world, this one is no protection at all.
        let mut s = healthy();
        s.dms_remaining_s = None;
        s.dms_failures = 2;
        s.halted = true;
        let w = s.warnings();
        assert_eq!(w.first().map(String::as_str), Some("DMS NOT ARMED"));
        let line = s.to_string();
        assert!(line.contains("dms - (fail 2)"), "{line}");
        assert!(line.contains("HALTED"), "{line}");
    }

    #[test]
    fn protection_that_lapsed_and_came_back_still_says_so() {
        // The failure this number exists for: a frozen process resumes, its next re-arm
        // succeeds, and *every other field on the line is repaired by that success*.
        // `dms_remaining_s` is full again, `dms_failures` never left zero because nothing
        // failed, and the halt is cleared microseconds after it is raised — so it is
        // gone before any status line samples it. A soak watched two consecutive lines
        // read `dms 0s` (the venue switch had fired) and then `dms 55s`, both `OK`.
        // Counted and not surfaced is the same as uncounted.
        let mut s = healthy();
        s.dms_remaining_s = Some(55);
        s.dms_failures = 0;
        s.halted = false;
        s.dms_lapses = 1;

        let w = s.warnings();
        assert!(
            w.contains(&"DMS PROTECTION LAPSED 1".to_string()),
            "a healthy-looking line must still carry the stall: {w:?}"
        );
        let line = s.to_string();
        assert!(line.contains("dms 55s (late 1)"), "{line}");
        assert!(!line.ends_with("OK"), "{line}");

        // Both halves of the switch's history read in one glance when both happened.
        s.dms_failures = 2;
        assert!(s.to_string().contains("dms 55s (fail 2 late 1)"), "{s}");
    }

    #[test]
    fn divergence_from_the_venue_is_visible_without_reading_logs() {
        let mut s = healthy();
        s.orphan_fills = 2;
        s.position_drift = 1;
        s.venue_missing = 3;
        s.dropped_exec_events = 5;
        let line = s.to_string();
        assert!(line.contains("EXEC EVENTS DROPPED 5"), "{line}");
        assert!(line.contains("POSITION DRIFT 1"), "{line}");
        assert!(line.contains("ORDERS UNCONFIRMED 3"), "{line}");
        assert!(line.contains("ORPHAN FILLS 2"), "{line}");
    }

    #[test]
    fn an_offline_session_omits_the_fields_that_would_be_lies() {
        // No wall-clock lag and no rate budget offline: printing zeros there would
        // read as "healthy and connected".
        let mut s = healthy();
        s.mode = "backtest/offline".into();
        s.data_lag_ms = None;
        s.rate = None;
        s.reconcile_age_s = None;
        s.dms_expected = false;
        s.dms_remaining_s = None;
        let line = s.to_string();
        assert!(!line.contains("lag"), "{line}");
        assert!(!line.contains("rate"), "{line}");
        assert!(!line.contains("rec"), "{line}");
        assert!(
            !line.contains("dms"),
            "an offline run has no venue to arm: {line}"
        );
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_producer_nobody_is_listening_to_is_named_not_inferred() {
        // Three different sessions look identical without this: a dead Python process,
        // a Python process writing a schema this build cannot read, and a strategy that
        // simply has no opinion today. Only the first is an emergency.
        let mut s = healthy();
        let i = s.intent.as_mut().unwrap();
        i.attached = false;
        i.accepted = 0;
        i.rejected = 0;
        let w = s.warnings();
        assert!(w.contains(&"SIGNAL RING DETACHED".to_string()), "{w:?}");
        assert!(s.to_string().contains("sig 0/0"), "{s}");
    }

    #[test]
    fn a_producer_whose_sequence_rewound_is_named_rather_than_read_as_a_quiet_strategy() {
        // The *fourth* way a session with a healthy feed places no orders, and the only
        // one with no name until now: Python restarted, its `seq` went back to 0, and the
        // reader refuses every record until the sequence passes the old baseline
        // (ADR-0014 §1). The ring is attached, the feed is fine, `rejected` climbs — and
        // an operator reading `sig 412/3` has no way to know that the 3 means "this
        // producer will never be listened to again" rather than "three late records".
        // Meanwhile Python's own counters say it is emitting normally, so the two sides
        // disagree about whether anything is happening and neither of them says so.
        let mut s = healthy();
        let i = s.intent.as_mut().unwrap();
        i.stale_seq = 3;
        assert!(i.attached, "the ring is fine; the sequence is not");
        let w = s.warnings();
        assert!(w.contains(&"SIGNAL SEQ REWOUND 3".to_string()), "{w:?}");
        assert!(!s.to_string().ends_with("OK"));
    }

    #[test]
    fn signals_accepted_and_orders_sent_are_both_on_the_line() {
        // The gap between them is the whole diagnosis when someone asks why a strategy
        // that is definitely emitting has not traded.
        let line = healthy().to_string();
        assert!(line.contains("sig 412/3"), "{line}");
        assert!(line.contains("sent 118+96c"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn an_intent_dropped_between_the_core_and_the_venue_is_never_silent() {
        let mut s = healthy();
        let i = s.intent.as_mut().unwrap();
        i.dropped = 2;
        i.failures = 1;
        i.untracked = 1;
        let line = s.to_string();
        assert!(line.contains("INTENTS DROPPED 2"), "{line}");
        assert!(line.contains("SUBMIT FAILURES 1"), "{line}");
        assert!(line.contains("ORDERS UNTRACKED 1"), "{line}");
    }

    #[test]
    fn a_wedged_submitter_is_named_rather_than_read_off_a_frozen_counter() {
        // One `place_order` on a stalled connection never returns, so the core stops
        // draining the ring: `accepted` freezes, `sent` freezes, and every number on the
        // line stays exactly where it was. Without this warning the session reports OK
        // while it has stopped trading altogether — the counters an operator would check
        // are the ones that cannot move.
        let mut s = healthy();
        let i = s.intent.as_mut().unwrap();
        i.busy = 4_012;
        i.stalled = true;
        let line = s.to_string();
        assert!(line.contains("busy 4012"), "{line}");
        assert!(line.contains("INTENT STALLED"), "{line}");
        assert!(!line.ends_with("OK"));
    }

    #[test]
    fn a_poisoned_tracker_is_never_reported_as_a_flat_and_healthy_session() {
        // A panic under the tracker lock leaves our order state unreadable: `open` and
        // the position list both read empty, the intent pass abandons every pass, and on
        // a quiet account no exec event ever arrives to raise EXEC EVENTS DROPPED. The
        // session would print `open 0 flat | OK` while holding a position.
        let mut s = healthy();
        s.open_orders = 0;
        s.positions = vec![];
        s.intent.as_mut().unwrap().poisoned = 900;
        let w = s.warnings();
        assert!(w.contains(&"POISONED TRACKER 900".to_string()), "{w:?}");
        assert!(!s.to_string().ends_with("OK"), "{s}");
    }

    #[test]
    fn a_shutdown_that_could_not_stop_the_submitter_says_so_on_the_closing_line() {
        // The last line the process prints is the only place this can be said: an
        // aborted submit task may still have put an order at the venue after the sweep
        // read the book, so the session is exiting with exposure nobody swept.
        let mut s = healthy();
        s.submitter_abandoned = 1;
        assert!(
            s.warnings().contains(&"SUBMITTER ABANDONED".to_string()),
            "{s}"
        );
    }

    #[test]
    fn a_market_data_feed_that_stopped_quoting_is_named_not_left_to_python() {
        // `MdSlice` has no field for the quote's own age, so a reader cannot tell a
        // minutes-old bid from a live one. When the publisher withholds the quote
        // entirely, this is the only signal that says why the fields went to zero.
        let mut s = healthy();
        s.md.as_mut().unwrap().stale_quote = 42;
        let w = s.warnings();
        assert!(w.contains(&"MD QUOTE STALE 42".to_string()), "{w:?}");
    }

    #[test]
    fn a_read_only_session_prints_no_signal_counters_at_all() {
        // Zeros here would read as "the strategy is quiet" rather than "there is no
        // strategy", and those need different responses.
        let mut s = healthy();
        s.intent = None;
        let line = s.to_string();
        assert!(!line.contains("sig"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_dropping_market_data_ring_is_never_silent() {
        // From Python a full ring and a quiet market look the same; from here they must
        // not. The depth is printed alongside so the warning is not the first sign.
        let mut s = healthy();
        let m = s.md.as_mut().unwrap();
        m.dropped = 17;
        m.queued = 4_096;
        let line = s.to_string();
        assert!(line.contains("md 9004 q 4096/4096"), "{line}");
        assert!(line.contains("MD SLICES DROPPED 17"), "{line}");
    }

    #[test]
    fn a_session_that_publishes_no_ring_prints_no_md_counters() {
        // Zeros would read as "the feed died", which needs a different response from
        // "nobody asked for a feed".
        let mut s = healthy();
        s.md = None;
        let line = s.to_string();
        assert!(!line.contains("md "), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn coalescing_is_reported_but_is_not_a_fault() {
        // A busy book whose top does not move is the normal case, not a degradation —
        // it must be visible without being alarming.
        let line = healthy().to_string();
        assert!(line.contains("coal 3039"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_recording_that_stopped_is_the_loudest_thing_on_the_line() {
        // A capture that quietly stopped leaves a log that looks complete and is not,
        // and the replay of it is green over a session that ended early. Nothing else
        // downstream can detect that, so it has to be said here.
        let mut s = healthy();
        let c = s.capture.as_mut().unwrap();
        c.stopped = Some(crate::capture::CaptureStop::QueueFull);
        c.missed = 8_210;
        let line = s.to_string();
        assert!(
            line.contains("CAPTURE STOPPED (queue full) MISSING 8210"),
            "{line}"
        );
        assert!(!line.ends_with("OK"));
    }

    #[test]
    fn a_recording_shows_events_and_signals_separately() {
        // A healthy event count with no signals is a recording whose replay will
        // re-observe the session and re-decide nothing — invisible in a single total.
        let line = healthy().to_string();
        assert!(line.contains("cap 12043+412s 43MiB q 4/16384"), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_session_that_records_nothing_prints_no_capture_counters() {
        // Zeros would read as "the recording has stalled", which needs a different
        // response from "nobody asked for a recording".
        let mut s = healthy();
        s.capture = None;
        let line = s.to_string();
        assert!(!line.contains("cap "), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn a_small_recording_is_not_rendered_as_zero_megabytes() {
        // `0MiB` next to a rising event count reads as a recording that is not
        // happening, which is the one thing this line exists to disprove.
        assert_eq!(size(0), "0KiB");
        assert_eq!(size(64 * 1024), "64KiB");
        assert_eq!(size(3 * 1024 * 1024), "3MiB");
    }

    #[test]
    fn counters_are_shared_and_absolute_where_it_matters() {
        let h = SessionHealth::new(1_000);
        h.note_adopted(2);
        h.note_adopted(1);
        assert_eq!(h.adopted_orders(), 3, "adoptions accumulate");

        h.set_position_drift(2);
        h.set_position_drift(0);
        assert_eq!(
            h.position_drift(),
            0,
            "drift is a property of the latest poll, not a running total"
        );

        assert_eq!(h.reconcile_ok_ms(), None);
        h.note_reconcile_ok(5_000);
        assert_eq!(h.reconcile_ok_ms(), Some(5_000));
    }

    // ── the money and the clock (ADR-0036) ──

    #[test]
    fn the_money_block_prints_the_bottom_line_and_the_venues_own_answer_beside_it() {
        // Both, always, and never one derived from the other. An operator reading a
        // single figure cannot tell a strategy that made money from a session whose
        // accounting has drifted away from the venue's.
        let line = healthy().to_string();
        assert!(
            line.contains("pnl +0.0150"),
            "net = 0.0288 - 0.0117 - 0.0021: {line}"
        );
        assert!(line.contains("(r +0.0288 fee 0.0117 u -0.0021)"), "{line}");
        assert!(line.contains("4m/0t"), "the maker/taker split: {line}");
        assert!(line.contains("eq 998.9132 +0.0167 drift -0.0017"), "{line}");
    }

    #[test]
    fn a_session_with_no_venue_prints_no_money_block_at_all() {
        // Offline. A `pnl +0.0000` on a backtest is a number where a reader expects an
        // account, and it is the reading that would let someone quote a canned event
        // stream's arithmetic as a result.
        let mut s = healthy();
        s.pnl = None;
        let line = s.to_string();
        assert!(!line.contains("pnl "), "{line}");
        assert!(line.ends_with("| OK"), "{line}");
    }

    #[test]
    fn an_unpriced_position_warns_and_outranks_the_loss_alarm_it_silences() {
        // The composition that matters: `net()` is `None` while anything is unpriced,
        // so the loss alarm *cannot* fire — and if the only warning were the quiet one,
        // a session holding a position whose feed had died would read as healthy.
        let mut s = healthy();
        let p = s.pnl.as_mut().unwrap();
        p.unrealized = None;
        p.unpriced = vec!["BTC".into()];
        s.pnl_limits = crate::config::PnlConfig {
            max_session_loss: dec!(1),
            equity_drift_alarm: Decimal::ZERO,
        };
        let warnings = s.warnings();
        assert!(
            warnings.iter().any(|w| w == "POSITION UNPRICED BTC"),
            "{warnings:?}"
        );
        assert!(
            !warnings.iter().any(|w| w.starts_with("SESSION LOSS")),
            "the loss alarm has nothing to judge: {warnings:?}"
        );
        assert!(s.to_string().contains("pnl - (r"), "{}", s.to_string());
    }

    #[test]
    fn the_loss_alarm_fires_on_a_net_worse_than_minus_the_declared_magnitude() {
        // The sign convention, pinned. `max_session_loss` is a magnitude and the net
        // it is compared against is signed; getting that backwards makes the alarm
        // either permanently on or permanently off, and both read as "no alarm".
        let mut s = healthy();
        s.pnl_limits = crate::config::PnlConfig {
            max_session_loss: dec!(0.5),
            equity_drift_alarm: Decimal::ZERO,
        };
        s.pnl.as_mut().unwrap().realized = dec!(-0.4);
        assert!(
            !s.warnings().iter().any(|w| w.starts_with("SESSION LOSS")),
            "-0.4138 is inside a 0.5 bound"
        );
        s.pnl.as_mut().unwrap().realized = dec!(-0.6);
        let w = s.warnings();
        assert!(
            w.iter()
                .any(|w| w.starts_with("SESSION LOSS -0.6138 PAST 0.5000")),
            "{w:?}"
        );
    }

    #[test]
    fn the_drift_alarm_fires_in_both_directions() {
        // Our accounting claiming a profit the venue has not paid is the more alarming
        // of the two, and a one-sided comparison would be blind to exactly that one.
        let mut s = healthy();
        s.pnl_limits = crate::config::PnlConfig {
            max_session_loss: Decimal::ZERO,
            equity_drift_alarm: dec!(0.01),
        };
        s.pnl.as_mut().unwrap().drift = Some(dec!(0.05));
        assert!(s.warnings().iter().any(|w| w == "PNL DRIFT +0.0500"));
        s.pnl.as_mut().unwrap().drift = Some(dec!(-0.05));
        assert!(s.warnings().iter().any(|w| w == "PNL DRIFT -0.0500"));
    }

    #[test]
    fn a_tripped_loss_limit_outranks_the_halt_it_looks_like() {
        // Both say orders are being refused; only this one says why, and only this one
        // is permanent. A halt is something a recovery clears, so an operator seeing it
        // knows to wait — and waiting is exactly the wrong response here, because what
        // this is holding open is the position that caused it.
        let mut s = healthy();
        s.halted = true;
        s.loss = Some(axon_execution::LossBreach {
            scope: axon_execution::LossScope::Session,
            loss: dec!(0.62),
            limit: dec!(0.5),
            marked: true,
        });
        let w = s.warnings();
        let loss = w.iter().position(|x| x.starts_with("LOSS LIMIT")).unwrap();
        let halt = w.iter().position(|x| x == "HALTED").unwrap();
        assert!(loss < halt, "{w:?}");
        assert!(
            w[loss].contains("session loss 0.6200 past the declared 0.5000"),
            "{w:?}"
        );
        assert!(w[loss].contains("marked, not realized"), "{w:?}");
        assert!(w[loss].contains("DE-RISK ONLY"), "{w:?}");
    }

    #[test]
    fn a_daily_bound_that_cannot_persist_its_baseline_says_so_rather_than_downgrading_quietly() {
        // The quiet kind of wrong: every number on this line is right and the guarantee
        // is not the one the operator configured. `validate` refuses a daily bound with
        // no path at all, so reaching here means the disk said no to a session that must
        // keep running and must not be believed about its day.
        let mut s = healthy();
        s.loss_limits = axon_execution::LossLimits {
            session: Decimal::ZERO,
            day: dec!(5),
        };
        s.daybook_fault = Some(crate::daybook::DayBookFault::NotPersisted);
        assert!(
            s.warnings()
                .iter()
                .any(|w| w.starts_with("DAILY LOSS BOUND NOT PERSISTED")),
            "{:?}",
            s.warnings()
        );

        // …and a session that declared no daily bound is not nagged about a book it
        // never asked for.
        s.loss_limits = axon_execution::LossLimits::default();
        assert!(
            !s.warnings()
                .iter()
                .any(|w| w.starts_with("DAILY LOSS BOUND")),
            "{:?}",
            s.warnings()
        );
    }

    #[test]
    fn an_undeclared_alarm_never_fires_however_bad_the_session_is() {
        // Zero is "no bound declared". A default that fires is a default nobody chose,
        // and the first thing it trains an operator to do is ignore the line.
        let mut s = healthy();
        s.pnl_limits = crate::config::PnlConfig::default();
        s.pnl.as_mut().unwrap().realized = dec!(-500);
        s.pnl.as_mut().unwrap().drift = Some(dec!(-400));
        let w = s.warnings();
        assert!(w.is_empty(), "{w:?}");
    }

    #[test]
    fn the_sweeper_and_the_re_quote_reach_the_line_they_used_to_be_invisible_on() {
        // `swept` and `resweeps` were counted from the day the sweeper existed and
        // reported nowhere, so the first live firing had to be read off the venue
        // rather than off this line. `requote` and `unquoted` are new and would have
        // repeated the mistake. Only printed once non-zero: a healthy strategy sweeps
        // nothing, and three permanent zeros are three fields an operator learns to
        // skip past — which is how the one that finally moves gets missed.
        let mut s = healthy();
        assert!(
            !s.to_string().contains("swept"),
            "silent when nothing was swept"
        );

        let i = s.intent.as_mut().unwrap();
        i.swept = 2;
        i.resweeps = 1;
        i.requotes = 3;
        i.unquoted = 1;
        i.stranded = 1;
        let line = s.to_string();
        assert!(line.contains("swept 2 (re 1)"), "{line}");
        assert!(line.contains("requote 3"), "{line}");
        assert!(line.contains("UNQUOTED TARGET 1"), "{line}");
        assert!(
            line.contains("STRANDED POSITION 1"),
            "an unquoted target and a position no order can express are two different \
             operator problems and only one of them has a fix inside this process: \
             {line}"
        );
    }

    #[test]
    fn an_unreadable_money_view_is_the_loudest_thing_on_the_block() {
        // It outranks every other money warning because it is the reason they are all
        // silent: `net()` is `None`, so no loss alarm can fire, and every figure the
        // block would otherwise print is a default.
        let mut s = healthy();
        s.pnl = Some(crate::pnl::PnlSnapshot::unreadable());
        s.pnl_limits = crate::config::PnlConfig {
            max_session_loss: dec!(0.01),
            equity_drift_alarm: dec!(0.01),
        };
        let w = s.warnings();
        assert_eq!(
            w.first().map(String::as_str),
            Some("PNL UNREADABLE"),
            "{w:?}"
        );
        assert!(!w.iter().any(|x| x.starts_with("SESSION LOSS")), "{w:?}");
        assert!(
            s.to_string().contains("| pnl UNREADABLE |"),
            "{}",
            s.to_string()
        );
    }

    #[test]
    fn a_latency_stage_over_its_budget_reaches_the_warning_list_by_name() {
        // The point of declaring a budget at all: the breach has to be legible without
        // an operator holding last week's percentiles in their head.
        let mut s = healthy();
        let book = crate::latency::LatencyBook::new([0, 2_000, 0, 0], 25);
        for _ in 0..3 {
            book.record(crate::latency::Stage::SignalAge, 9_406);
        }
        book.record(crate::latency::Stage::SignalAge, 100);
        s.latency = book.snapshot();
        let line = s.to_string();
        assert!(line.contains("lat sig 10000/10000·9406 3/4"), "{line}");
        assert!(line.contains("LATENCY sig 3/4 OVER 2000ms"), "{line}");
    }
}
