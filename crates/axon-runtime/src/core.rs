//! The deterministic core thread: drain the bus, dispatch, print, repeat.
//!
//! This loop is the one place in a live session that is **not** async, and keeping it
//! that way is the point (ADR-0008). It runs on its own OS thread with no runtime
//! handle in scope, so nothing here can accidentally `await`, block a runtime worker,
//! or make event ordering depend on how tokio happened to schedule two tasks.
//!
//! It does four things per iteration, in this order: drain every buffered event
//! through the [`CoreHandler`], advance the mark cache's liveness clock, run one pass
//! of the intent source, and — on a slow cadence — print the status line.
//!
//! The order is the design. Draining first means the status line describes the state
//! *after* everything received so far, never a half-applied view. And the intent pass
//! comes after the drain for a stronger reason: the position, the book and the working
//! orders a plan is computed against are then the state that event
//! [`CoreHandler::last_ts`] left behind — which is the same clock the reader ages the
//! signal against. One clock, one state, no way for a plan to be arithmetic about a
//! market it had only half seen (ADR-0020).
//!
//! The pass is offered on **every** iteration and paces itself on the event clock, so
//! how often the loop spins changes nothing about which signals share a pass. That is
//! deliberate: it is the difference between a replay reproducing a session and a replay
//! reproducing the machine it ran on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axon_core::{drain_available, Clock, EventReceiver, SymbolId, SystemClock};
use axon_execution::HaltSwitch;
use axon_provider_hyperliquid::RateGovernor;

use crate::capture::CaptureTap;
use crate::dms::{now_ms, DmsState};
use crate::handler::CoreHandler;
use crate::health::{CaptureLine, IntentLine, MdLine, RateLine, SessionHealth, StatusSnapshot};
use crate::intent::{IntentPoll, IntentStats, StrategyLine};

/// What the status line reads off the intent source in one go.
///
/// Two calls rather than one because [`IntentStats`] is `Copy` and the per-producer
/// lines are not — a `String` per producer on a `Copy` summary would make every status
/// assembly allocate whether or not anybody had two strategies. Bundling them here
/// keeps the two reads at one instant, which is what stops the aggregate and the
/// per-producer detail describing different passes.
pub struct IntentView {
    pub stats: IntentStats,
    pub strategies: Vec<StrategyLine>,
}

impl IntentView {
    pub fn of(src: &dyn IntentPoll) -> Self {
        Self {
            stats: src.stats(),
            strategies: src.strategies(),
        }
    }
}

/// Everything the core loop needs that is not the handler itself.
///
/// Plain fields, built with a struct literal by the composition root: a constructor
/// taking a dozen positional arguments is exactly how a `bool` ends up in the wrong
/// slot.
pub struct CoreControl {
    pub stop: Arc<AtomicBool>,
    /// How long to park when the bus is empty.
    pub poll: Duration,
    pub status_every: Duration,
    /// Whether to age mark prices against the wall clock.
    ///
    /// Live only. Offline, event time is the only time that exists and a wall-clock
    /// reading would make an otherwise deterministic run depend on how fast the
    /// machine drained the bus.
    pub wall_time: bool,
    pub mode: String,
    /// `(id, venue name)` for the configured universe, so the status line can say
    /// `BTC` instead of `sym#0`.
    pub symbols: Vec<(SymbolId, String)>,
    pub health: Arc<SessionHealth>,
    pub halt: Arc<HaltSwitch>,
    pub dms: Arc<DmsState>,
    /// Present only in a live session; its absence is what tells the status line not
    /// to report a missing dead-man's switch as a fault.
    pub governor: Option<Arc<RateGovernor>>,
    pub dms_expected: bool,
    /// A read-only view of the session recording, when there is one.
    ///
    /// The loop holds a *tap* rather than the recorder: the recorder is what ends the
    /// recording, and giving that to the loop would give it a way to join the writer
    /// thread in the middle of a drain.
    pub capture: Option<CaptureTap>,
    /// Whether this session has an account whose money is worth reporting.
    ///
    /// True exactly when there is a venue. An offline run's positions are arithmetic
    /// over a canned event stream and its "P&L" is a property of the fixture, so
    /// printing one would put a number where a reader expects an account — the same
    /// reasoning `dms_expected` uses for the safety switch.
    pub pnl_expected: bool,
    /// The operator's P&L alarms. Copied rather than borrowed because it is two
    /// `Decimal`s and the snapshot is assembled on a different thread from the config.
    pub pnl_limits: crate::config::PnlConfig,
    /// Declared latency budgets and what has been measured against them. Shared with
    /// the submit task, which writes two of the three stages.
    pub latency: Arc<crate::latency::LatencyBook>,
    /// The loss-based kill switch, shared with the risk gate that enforces it.
    ///
    /// **This loop is the only thing that trips it**, and it does so on the status-line
    /// cadence, because the P&L view is what it judges and that is when the P&L view is
    /// assembled. The lag is real, bounded by `session.status_interval_ms`, and worth
    /// stating rather than hiding: it is the same lag every number on that line already
    /// has, and the alternative — recomputing a mark-to-market on every event — would
    /// put a walk of the position book on the hot path to bound a quantity that moves
    /// on the timescale of a fill.
    pub loss: Arc<axon_execution::LossLimiter>,
    /// The UTC day's equity baseline, which is what makes `loss`'s day bound survive a
    /// restart. Absent when no daily bound was declared.
    pub daybook: Option<Arc<crate::daybook::DayBook>>,
}

impl CoreControl {
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn name(&self, id: SymbolId) -> String {
        self.symbols
            .iter()
            .find(|(s, _)| *s == id)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| id.to_string())
    }

    /// Feed the money view to the kill switch, and say so on the pass that trips it.
    ///
    /// Three refusals are the substance here, and each of them is a way this could have
    /// been silently wrong:
    ///
    /// - **An unreadable tracker judges nothing.** A poisoned lock produces a snapshot
    ///   whose every figure is absent rather than zero, and feeding those absences to a
    ///   bound would either trip on nothing or, worse, read `realized 0` as a session
    ///   that has lost nothing. `POISONED TRACKER` is already on the line; this stays
    ///   quiet and leaves the switch where it was.
    /// - **A session with no venue judges nothing.** `pnl` is `None` on an offline run,
    ///   whose "P&L" is a property of a canned fixture.
    /// - **The day bound is judged on the venue's own equity and only when there is
    ///   one.** Before the first `clearinghouseState` reply there is no reading and no
    ///   baseline, and treating that absence as a flat day is right — treating it as a
    ///   breach would stop every session at startup.
    ///
    /// `now_ms` is wall clock, and the exception is the day book's, argued there: no
    /// event time answers "is it tomorrow yet".
    fn judge_loss(&self, pnl: Option<&crate::pnl::PnlSnapshot>, now_ms: u64) {
        let Some(p) = pnl.filter(|p| p.readable) else {
            return;
        };
        // `net()` when a mark vouches for every held symbol, and what has actually been
        // closed and paid when one does not. See `LossLimiter::observe`: silence on an
        // unpriced position is defensible for a warning and is not defensible for a kill
        // switch, because it would make a dead feed a way to switch the bound off.
        let day_loss = self
            .daybook
            .as_ref()
            .zip(p.equity)
            .and_then(|(book, eq)| book.observe(eq, now_ms));
        if let Some(breach) = self.loss.observe(p.net(), p.realized_net(), day_loss) {
            // Once, on the pass that trips. The status line then carries it for as long
            // as it lasts, which is the rest of the session.
            eprintln!(
                "LOSS LIMIT TRIPPED - {breach}. New exposure is refused from here; \
                 orders that reduce a position and every cancel still go out. \
                 Nothing but a restart clears this."
            );
        }
    }

    /// Assemble the printable view of the session.
    ///
    /// `intent` is `None` for a read-only session — one with no intent source at all —
    /// and that absence is printed as absence rather than as zeros.
    pub fn snapshot(
        &self,
        handler: &CoreHandler,
        bus_len: usize,
        intent: Option<IntentView>,
    ) -> StatusSnapshot {
        let now_ms = now_ms();
        let (marks_fresh, marks_total) = handler.marks().coverage();
        let (open_orders, orphan_fills) = match handler.tracker().read() {
            Ok(t) => (t.open_count(), t.orphan_fills()),
            Err(_) => (0, 0),
        };
        let ids: Vec<SymbolId> = self.symbols.iter().map(|(s, _)| *s).collect();

        // The money. Assembled under the *same* tracker read as everything else on this
        // line would be the ideal; it takes its own, and that is a deliberate and
        // bounded inconsistency: the read above answers "how many orders are open" and
        // this one answers "what are they worth", the status line is not a decision, and
        // holding the lock across both would put the P&L assembly inside the critical
        // section the submit path contends for.
        let pnl = self.pnl_expected.then(|| match handler.tracker().read() {
            Ok(t) => crate::pnl::snapshot(&t, handler.marks(), &self.symbols),
            // A poisoned tracker already shouts through `POISONED TRACKER`; a fabricated
            // zero P&L beside it would be the quieter and more believable of the two
            // lies.
            //
            // ADR-0036 records this exact defect as *found and fixed*, and the fix did
            // not reach here: the comment above said what the code below must not do,
            // and the code below did it — `snapshot()` over a fresh, empty
            // `OrderTracker` returns `readable: true` and prints `pnl +0.0000`, so a
            // session holding a position it could not see reported that it had done
            // nothing. `PnlSnapshot::unreadable()` existed the whole time and had no
            // production caller.
            Err(_) => crate::pnl::PnlSnapshot::unreadable(),
        });
        self.judge_loss(pnl.as_ref(), now_ms);

        StatusSnapshot {
            mode: self.mode.clone(),
            uptime_s: now_ms.saturating_sub(self.health.started_ms()) / 1_000,
            events: handler.events(),
            bus_len,
            // Only meaningful against a wall clock: offline, `last_ts` is a point in
            // the captured log and its distance from "now" means nothing.
            data_lag_ms: (self.wall_time && handler.last_ts() > 0).then(|| {
                SystemClock
                    .now_ns()
                    .saturating_sub(handler.last_ts())
                    .max(0) as u64
                    / 1_000_000
            }),
            marks_fresh,
            marks_total,
            stale_marks: handler
                .marks()
                .stale_symbols()
                .into_iter()
                .map(|s| self.name(s))
                .collect(),
            open_orders,
            positions: handler
                .positions(&ids)
                .into_iter()
                .map(|(s, q)| (self.name(s), q))
                .collect(),
            orphan_fills,
            adopted_orders: self.health.adopted_orders(),
            venue_missing: self.health.venue_missing(),
            position_drift: self.health.position_drift(),
            dropped_exec_events: handler.dropped_exec_events(),
            bars_without_a_clock: handler.bars_without_a_clock(),
            bus_full_drops: self.health.bus_full_drops(),
            submitter_abandoned: self.health.submitter_abandoned(),
            dms_expected: self.dms_expected,
            dms_remaining_s: self
                .dms
                .deadline_ms()
                .map(|d| d.saturating_sub(now_ms) / 1_000),
            dms_failures: self.dms.failures(),
            // The one part of the switch's history that outlives its own recovery: a
            // stalled loop re-arms successfully the moment it resumes, and that success
            // repairs `dms_remaining_s` and clears the halt before this line is next
            // assembled.
            dms_lapses: self.dms.lapses(),
            rate: self.governor.as_ref().map(|g| {
                let s = g.snapshot(now_ms);
                RateLine {
                    ip_used: s.ip_weight_used,
                    ip_limit: s.ip_weight_limit,
                    address_used: s.address_used,
                    address_limit: s.address_limit,
                    throttled: s.throttled,
                }
            }),
            reconcile_age_s: self
                .health
                .reconcile_ok_ms()
                .map(|t| now_ms.saturating_sub(t) / 1_000),
            reconcile_failures: self.health.reconcile_failures(),
            halted: !self.halt.is_accepting(),
            // Two halves of one answer: what the core made of the records (its own
            // counters) and what the edge managed to do with them (the shared health
            // block). Reported together because either alone is misleading.
            intent: intent.map(|v| {
                let s = v.stats;
                IntentLine {
                    accepted: s.accepted,
                    rejected: s.rejected,
                    expired: s.expired,
                    // Counted since ADR-0014 and reported nowhere until now, which is the
                    // same as uncounted: a restarted producer is refused silently and the
                    // line reads as a strategy with nothing to say.
                    stale_seq: s.stale_seq,
                    gaps: s.gaps,
                    planned: s.planned,
                    no_quote: s.no_quote,
                    precision_refusals: s.precision_refusals,
                    unknown_precision: s.unknown_precision,
                    dropped: s.dropped,
                    // Four more that were counted and reported nowhere, which this crate's
                    // own rule calls the same as uncounted. `swept` in particular is a
                    // number that moves in production — the Phase-6 live run had to read
                    // its two sweeps off the venue rather than off its own status line.
                    no_order: s.no_order,
                    ahead_of_clock: s.ahead_of_clock,
                    swept: s.swept,
                    resweeps: s.resweeps,
                    requotes: s.requotes,
                    unquoted: s.unquoted,
                    stranded: s.stranded,
                    // The two that say the path has stopped working entirely. Counted and
                    // not reported is the same as not counted: what an operator sees is a
                    // frozen `accepted` and a session that says OK.
                    busy: s.busy,
                    stalled: s.stalled,
                    poisoned: s.poisoned,
                    orders: self.health.intent_orders(),
                    cancels: self.health.intent_cancels(),
                    failures: self.health.intent_failures(),
                    halted: self.health.intent_halted(),
                    untracked: self.health.intent_untracked(),
                    attached: s.attached,
                    // The fleet (ADR-0038). Every one of these is zero or absent on a
                    // single-producer session, which is what keeps the line an operator has
                    // been reading for months unchanged.
                    detached: s.detached,
                    producers: s.producers,
                    netted: s.netted,
                    silent_held: s.silent_held,
                    silent_flat: s.silent_flat,
                    overlap_refused: s.overlap_refused,
                    out_of_scope: s.out_of_scope,
                    band_dropped: s.band_dropped,
                    strategy_scaled: s.strategy_scaled,
                    portfolio_scaled: s.portfolio_scaled,
                    portfolio_scale_bps: s.portfolio_scale_bps,
                    breadth_denied: s.breadth_denied,
                    alloc_unpriced: s.alloc_unpriced,
                    // Only for a session running several: with one producer the aggregate
                    // above already says everything there is to say, and a second copy of it
                    // under a name is a line nobody reads.
                    strategies: if s.producers > 1 {
                        v.strategies
                            .into_iter()
                            .map(|l| crate::health::StrategyStatus {
                                name: l.name,
                                attached: l.attached,
                                accepted: l.accepted,
                                rejected: l.rejected,
                                expired: l.expired,
                                stale_seq: l.stale_seq,
                                claims: l.claims,
                                silent_ms: l.silent_ms,
                                silent: l.silent,
                                out_of_scope: l.out_of_scope,
                            })
                            .collect()
                    } else {
                        Vec::new()
                    },
                }
            }),
            // The other direction of the boundary. `None` when this session publishes no
            // market-data ring, so "nobody asked for one" never reads as "the feed died".
            md: handler.md().map(|p| {
                let s = p.stats();
                MdLine {
                    published: s.published,
                    dropped: s.dropped,
                    coalesced: s.coalesced,
                    unrepresentable: s.unrepresentable,
                    stale_quote: s.stale_quote,
                    queued: s.queued,
                    capacity: s.capacity,
                    bars_published: s.bars_published,
                    bars_dropped: s.bars_dropped,
                }
            }),
            pnl,
            pnl_limits: self.pnl_limits,
            loss: self.loss.breach(),
            loss_limits: *self.loss.limits(),
            daybook_fault: self.daybook.as_ref().and_then(|d| d.fault()),
            latency: self.latency.snapshot(),
            // Whether this session will leave an artifact behind. `None` when nobody
            // asked for one — a recording that stopped and a recording that was never
            // started need different responses.
            capture: self.capture.as_ref().map(|c| {
                let s = c.progress();
                CaptureLine {
                    events: s.events,
                    signals: s.signals,
                    bytes: s.bytes,
                    missed: s.missed,
                    queued: s.queued,
                    capacity: s.capacity,
                    stopped: s.stopped,
                }
            }),
        }
    }
}

/// Run until stopped, then drain what is left and return the final snapshot.
///
/// The loop exits only once the stop flag is set **and** the bus is empty: the last
/// events of a session are the cancel acknowledgements from the shutdown sweep, and
/// exiting before applying them would leave the closing status line claiming orders
/// are still resting when they are not.
pub fn run(
    rx: &EventReceiver,
    handler: &mut CoreHandler,
    ctl: &CoreControl,
    mut intent: Option<&mut dyn IntentPoll>,
) -> StatusSnapshot {
    let mut last_status = Instant::now();
    loop {
        let drained = drain_available(rx, handler);

        // One wall-clock read per iteration, shared by the two things that need one.
        // Neither is a trading decision: the first ages prices, the second paces how
        // often we try to open a file.
        let wall_ns = if ctl.wall_time {
            // The liveness clock for mark staleness. It has to advance even when no
            // event arrives — a dead feed is exactly the case where nothing does, and
            // event time alone would freeze and call every stale price fresh.
            let ns = SystemClock.now_ns();
            handler.marks().observe_now(ns);
            ns
        } else {
            0
        };

        // The market-data beacon (ADR-0030/0034), driven from the pass rather than from
        // an event: the state it exists to make legible is the one in which nothing
        // arrived, and a beacon advanced by `on_event` freezes exactly there. `wall_ns`
        // is the loop's single reading, shared — it becomes `last_beat_ns`, the third
        // named wall-clock exception in this codebase, because the absence of an event
        // has no event time. `0` offline, which the reader reads as "no wall clock".
        //
        // `None` when this session publishes no ring, so nothing is created by a session
        // nobody asked to publish.
        if let Some(md) = handler.md() {
            md.beat(handler.last_ts(), wall_ns as u64);
        }

        if let Some(src) = intent.as_deref_mut() {
            src.poll(handler.last_ts(), (wall_ns / 1_000_000) as u64, handler);
        }

        if last_status.elapsed() >= ctl.status_every {
            last_status = Instant::now();
            println!(
                "{}",
                ctl.snapshot(handler, rx.len(), intent.as_deref().map(IntentView::of))
            );
        }

        if ctl.stopping() && rx.is_empty() {
            // A final beat marked as a deliberate shutdown, after the drain that emptied
            // the bus so the counters it carries describe the whole session. A reader
            // must still treat the feed as gone, but "the session ended" and "the session
            // died" want different words from whoever is woken up at 03:00, and this is
            // the only side that can tell them apart.
            if let Some(md) = handler.md() {
                md.beat_stopped(handler.last_ts(), wall_ns as u64);
            }
            return ctl.snapshot(handler, 0, intent.as_deref().map(IntentView::of));
        }
        if drained == 0 {
            std::thread::sleep(ctl.poll);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selftest;
    use axon_core::bus;
    use axon_execution::{MarkCache, OrderTracker};
    use std::sync::RwLock;

    // ── the loss kill switch, judged where the money view is assembled ───────

    use rust_decimal_macros::dec;

    /// A money view with `realized` set and everything else quiet. `unrealized: None`
    /// makes `net()` `None`, which is the unpriced case.
    fn money(realized: rust_decimal::Decimal, priced: bool) -> crate::pnl::PnlSnapshot {
        crate::pnl::PnlSnapshot {
            realized,
            realized_all: realized,
            fees: rust_decimal::Decimal::ZERO,
            fees_all: rust_decimal::Decimal::ZERO,
            venue_closed_pnl: rust_decimal::Decimal::ZERO,
            unrealized: priced.then_some(rust_decimal::Decimal::ZERO),
            unpriced: if priced { vec![] } else { vec!["BTC".into()] },
            gross_exposure: rust_decimal::Decimal::ZERO,
            fills: 1,
            maker_fills: 1,
            taker_fills: 0,
            fills_all: 1,
            equity: None,
            equity_at_start: None,
            equity_delta: None,
            drift: None,
            readable: true,
        }
    }

    fn with_loss(session: rust_decimal::Decimal) -> CoreControl {
        let mut c = control(Arc::new(AtomicBool::new(true)), false);
        c.pnl_expected = true;
        c.loss = Arc::new(axon_execution::LossLimiter::new(
            axon_execution::LossLimits {
                session,
                day: rust_decimal::Decimal::ZERO,
            },
        ));
        c
    }

    #[test]
    fn the_status_pass_is_what_trips_the_loss_limit() {
        // The core loop is the only writer, because the money view is what the bound
        // judges and this is where it is assembled. The lag is `status_interval_ms` and
        // it is the same lag every other number on that line already has.
        let c = with_loss(dec!(0.5));
        c.judge_loss(Some(&money(dec!(-0.4), true)), 0);
        assert!(!c.loss.is_tripped(), "inside the bound");
        c.judge_loss(Some(&money(dec!(-0.6), true)), 0);
        assert!(c.loss.is_tripped());
        assert_eq!(c.loss.breach().unwrap().loss, dec!(0.6));
    }

    #[test]
    fn an_unreadable_money_view_judges_nothing_rather_than_reading_absence_as_zero() {
        // A poisoned tracker produces a snapshot whose every figure is absent rather
        // than measured. Feeding those to a bound would read `realized 0` as a session
        // that has lost nothing — the same fabricated zero this module already refuses
        // to print, arriving through a different door and this time deciding something.
        let c = with_loss(dec!(0.5));
        c.judge_loss(Some(&crate::pnl::PnlSnapshot::unreadable()), 0);
        assert!(!c.loss.is_tripped());
        // …and a session with no venue at all judges nothing either.
        c.judge_loss(None, 0);
        assert!(!c.loss.is_tripped());
    }

    #[test]
    fn a_session_that_cannot_price_its_position_is_still_judged_on_what_it_has_paid() {
        // ADR-0036 §2 makes `net()` None whenever a held symbol has no fresh mark, and
        // the P&L *warning* is silent for as long as that lasts. Silence is defensible
        // for a warning beside a louder `POSITION UNPRICED`; for a kill switch it would
        // make a dead feed a way to switch the bound off, on exactly the session that
        // has a position and cannot see it.
        let c = with_loss(dec!(0.5));
        let unpriced = money(dec!(-0.9), false);
        assert_eq!(unpriced.net(), None, "the precondition this test is about");
        c.judge_loss(Some(&unpriced), 0);
        assert!(c.loss.is_tripped());
        assert!(
            !c.loss.breach().unwrap().marked,
            "and the line says the loss is spent, not marked"
        );
    }

    #[test]
    fn the_day_bound_reads_the_venues_equity_through_the_book_that_outlives_the_process() {
        // Two accountings, never reconciled. Here ours is healthy and the venue's is
        // not, which is the case a session bound alone cannot see.
        let mut c = control(Arc::new(AtomicBool::new(true)), false);
        c.pnl_expected = true;
        c.loss = Arc::new(axon_execution::LossLimiter::new(
            axon_execution::LossLimits {
                session: dec!(100),
                day: dec!(1),
            },
        ));
        c.daybook = Some(Arc::new(crate::daybook::DayBook::in_memory()));

        let mut p = money(dec!(-0.01), true);
        p.equity = Some(dec!(1000));
        c.judge_loss(Some(&p), 0);
        assert!(!c.loss.is_tripped(), "the first reading only baselines");

        p.equity = Some(dec!(998.5));
        c.judge_loss(Some(&p), 0);
        assert!(c.loss.is_tripped());
        let b = c.loss.breach().unwrap();
        assert_eq!(b.scope, axon_execution::LossScope::Day);
        assert_eq!(b.loss, dec!(1.5));
    }

    fn control(stop: Arc<AtomicBool>, wall_time: bool) -> CoreControl {
        CoreControl {
            stop,
            poll: Duration::from_micros(100),
            status_every: Duration::from_secs(3_600), // never fires inside a test
            wall_time,
            mode: "test/offline".into(),
            symbols: vec![
                (SymbolId::new(0), "BTC".into()),
                (SymbolId::new(1), "ETH".into()),
            ],
            health: Arc::new(SessionHealth::new(now_ms())),
            halt: Arc::new(HaltSwitch::new()),
            dms: Arc::new(DmsState::default()),
            governor: None,
            dms_expected: false,
            capture: None,
            pnl_expected: false,
            pnl_limits: crate::config::PnlConfig::default(),
            latency: Arc::new(crate::latency::LatencyBook::undeclared()),
            loss: Arc::new(axon_execution::LossLimiter::undeclared()),
            daybook: None,
        }
    }

    #[test]
    fn the_loop_drains_everything_queued_before_it_exits() {
        // The closing snapshot has to describe the end of the session, not the moment
        // the stop flag happened to be set — the sweep's cancel acks arrive last.
        let (tx, rx) = bus(64);
        let events = selftest::events(SymbolId::new(0), SymbolId::new(1));
        let expected = events.len() as u64;
        for ev in events {
            tx.send(ev).unwrap();
        }
        drop(tx);

        let stop = Arc::new(AtomicBool::new(true)); // already stopping
        let ctl = control(stop, false);
        let mut handler = CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::never_expires()),
        );
        let snap = run(&rx, &mut handler, &ctl, None);

        assert_eq!(snap.events, expected);
        assert_eq!(snap.bus_len, 0);
        assert!(snap.data_lag_ms.is_none(), "offline has no wall-clock lag");
        assert!(snap.rate.is_none(), "no governor offline");
        assert!(
            snap.intent.is_none(),
            "a loop with no intent source reports absence, not zeros"
        );
    }

    #[test]
    fn the_intent_source_sees_the_state_the_events_left_behind() {
        // The ordering inside the loop: if the intent pass ran before the drain, the
        // first plan of a session would price against an empty book and compute its
        // delta against a position the fills in the same batch had already moved.
        use crate::config::RuntimeConfig;
        use crate::intent::{IntentSink, IntentSource};
        use axon_strategy::ReplaySource;

        let (tx, rx) = bus(64);
        for ev in selftest::events(SymbolId::new(0), SymbolId::new(1)) {
            tx.send(ev).unwrap();
        }
        drop(tx);

        let cfg = RuntimeConfig::default();
        let mut src = IntentSource::new(
            ReplaySource::new(selftest::signals(SymbolId::new(0), SymbolId::new(1))),
            &cfg.intent,
            // The same declared grids the offline session uses, so this drives the
            // whole pass rather than a version of it with the rounding switched off.
            Arc::new(selftest::instruments([SymbolId::new(0), SymbolId::new(1)])),
            cfg.mark_max_age_ns(),
            Arc::new(HaltSwitch::new()),
            IntentSink::Record(Vec::new()),
        );
        let ctl = control(Arc::new(AtomicBool::new(true)), false);
        let mut handler = CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::never_expires()),
        );
        let snap = run(&rx, &mut handler, &ctl, Some(&mut src));

        let line = snap.intent.expect("the intent source is reported");
        assert!(line.attached);
        assert_eq!(line.planned, 2, "one order per instrument");
        assert_eq!(line.orders, 0, "offline nothing reaches a venue");
        assert!(!src.take_recorded().is_empty());
    }

    #[test]
    fn a_stop_signal_ends_the_loop_even_with_a_silent_bus() {
        // A blocking receive would hang here forever; the poll is what lets a session
        // with a dead feed still shut down.
        let (_tx, rx) = bus(4);
        let stop = Arc::new(AtomicBool::new(false));
        let ctl = control(stop.clone(), false);
        let mut handler = CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::never_expires()),
        );
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            stop.store(true, Ordering::Release);
        });
        let snap = run(&rx, &mut handler, &ctl, None);
        assert_eq!(snap.events, 0);
    }
}
