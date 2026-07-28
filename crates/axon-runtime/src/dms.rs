//! The re-arming dead-man's switch — the loop that must die last.
//!
//! `scheduleCancel` is the only protection that survives *this process* dying: a
//! client-side cancel cannot run if we have crashed, wedged or lost the network,
//! while a deadline held by the venue fires regardless. The venue enforces a 5 s
//! minimum lead and honours at most
//! [`SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY`](axon_provider_hyperliquid::SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY)
//! firings per UTC day, so the protection is real but not free and not infinite.
//!
//! **A failed re-arm is not a warning.** It is the protection expiring on a clock we
//! do not control, so the escalation is graded by how much protection is left rather
//! than by how many attempts have failed:
//!
//! | remaining protection | response |
//! |---|---|
//! | more than one interval | retry — one missed beat is what the lead is sized for |
//! | one interval or less | **halt placements**: we are one failure from unprotected, and adding exposure now is adding exposure we may not be able to remove |
//! | none | **unprotected**: shut the session down |
//!
//! Halting before the deadline rather than at it is the whole design. Waiting until
//! protection has actually lapsed means the orders placed in the final interval are
//! the ones left stranded.
//!
//! **Protection runs out on its own, and that is the case the switch exists for.** The
//! table above is a function of *remaining protection*, not of whether anything
//! errored, so the loop consults it twice per pass: once on the elapsed clock before it
//! tries anything ([`DmsPolicy::on_elapsed`]), and once on the outcome
//! ([`DmsPolicy::on_armed`] / [`DmsPolicy::on_failure`]). Grading only the outcome is
//! what a 1 h 44 m soak found: freezing the process with `SIGSTOP` for 80 s against a
//! 60 s lead drove the venue-side switch to fire, and the session — whose every `arm()`
//! had succeeded and whose next one succeeded too — printed no `HALTED`, no
//! `UNPROTECTED` and nothing on stderr, then reported healthy on both sides of the gap.
//! A stalled process arms nothing and *fails* nothing, so an error-triggered ladder is
//! a ladder with no rungs on the path that matters.
//!
//! What the elapsed check delivers is bounded and worth stating plainly: **a resumed
//! process escalates instead of continuing as if nothing happened.** It cannot protect
//! a stalled one. A process that is not running executes no checks; the venue-side
//! deadline is the only thing covering that window, and it is precisely because we
//! cannot narrow the gap that we must not paper over it.
//!
//! [`DmsPolicy`] is a pure state machine over an externally supplied `now_ms` — no
//! clock, no I/O, no tokio — for the same reason the nonce manager and rate governor
//! are: every branch of an escalation path has to be testable offline, because the
//! branches only run on the worst day.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axon_execution::HaltSwitch;
use axon_provider_hyperliquid::ExchangeClient;
use axon_providers::ProviderError;

use crate::health::SessionHealth;

/// Consecutive failures tolerated before a switch that was *never* armed is treated
/// as absent protection.
///
/// A first-arm failure has no deadline to measure against, so it cannot be graded by
/// remaining time. Three attempts distinguishes a transient 502 from a key, clock or
/// permission problem that will not fix itself.
pub const UNARMED_FAILURES_BEFORE_UNPROTECTED: u32 = 3;

/// Why the ladder was consulted. The rung is graded by protection remaining; this is
/// the sentence that says what spent it.
///
/// Carried rather than inferred because the two need different responses from whoever
/// reads the line: a re-arm failure is the venue or the network, and waiting is a
/// reasonable thing to do about it; a loop that did not run is *this process*, and
/// waiting is not. Reporting the second as "re-arm failed (attempt 0)" would send an
/// operator to the wrong machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cause {
    /// The venue refused the re-arm or could not be reached, `attempt` times running.
    ArmFailed { attempt: u32 },
    /// Nothing failed. The loop did not run for `silent_ms` — frozen, descheduled,
    /// starved of an executor thread, or wedged inside a call that never returned — and
    /// protection drained on a clock nobody was watching.
    LoopLate { silent_ms: u64 },
}

impl Cause {
    /// One clause naming what happened, so the console line says *why* the ladder was
    /// consulted and not only which rung it landed on.
    fn clause(&self) -> String {
        match self {
            Cause::ArmFailed { attempt } => format!("re-arm failed (attempt {attempt})"),
            Cause::LoopLate { silent_ms } => {
                format!("the re-arm loop did not run for {silent_ms} ms")
            }
        }
    }
}

/// What the supervisor should do about the current state of the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Escalation {
    /// Armed and healthy.
    Healthy { deadline_ms: u64, remaining_ms: u64 },
    /// Protection to spare.
    Retry { remaining_ms: u64, cause: Cause },
    /// One interval or less of protection left: stop adding exposure.
    Halt { remaining_ms: u64, cause: Cause },
    /// No protection remains: the deadline we last armed has passed, or none was ever
    /// accepted.
    ///
    /// **This is a statement about our own clock, never about the venue's.** Whether
    /// the venue-side switch actually fired is something no `/exchange` reply told us,
    /// and on the one occasion it mattered the answer was no — during the 2026-07-27
    /// testnet outage the deadline lapsed, this rung fired, and when the venue came
    /// back the order placed before it was **still resting**. See [`Cause::clause`].
    Unprotected { cause: Cause },
}

/// Tracks the armed deadline and grades both a failed re-arm and a drained clock
/// against it.
#[derive(Debug, Clone, Copy)]
pub struct DmsPolicy {
    lead_ms: u64,
    rearm_interval_ms: u64,
    deadline_ms: Option<u64>,
    /// When the last successful arm landed. Kept beside the deadline rather than
    /// derived from `deadline - lead_ms`: the deadline is whatever the venue accepted,
    /// and a clamped or rounded one would turn "how long were we silent" into a guess
    /// exactly when the answer matters.
    armed_at_ms: Option<u64>,
    failures: u32,
    /// Times protection drained into the halt band or past it with **nothing having
    /// failed** — i.e. this process stopped re-arming. Cumulative and deliberately
    /// **never cleared**, unlike `failures`: the successful arm that follows a stall is
    /// what erases every other trace of it, and a session that lost protection and
    /// recovered still lost protection. It is what the status line reports as `late n`.
    lapses: u32,
    /// Whether the elapsed check has already halted for the protection it has now.
    /// Cleared by a successful arm. Without it a venue outage prints the same halt line
    /// every few seconds and buries the one that mattered.
    stall_halted: bool,
}

impl DmsPolicy {
    pub fn new(lead_ms: u64, rearm_interval_ms: u64) -> Self {
        Self {
            lead_ms,
            rearm_interval_ms,
            deadline_ms: None,
            armed_at_ms: None,
            failures: 0,
            lapses: 0,
            stall_halted: false,
        }
    }

    pub fn lead_ms(&self) -> u64 {
        self.lead_ms
    }

    pub fn rearm_interval_ms(&self) -> u64 {
        self.rearm_interval_ms
    }

    pub fn deadline_ms(&self) -> Option<u64> {
        self.deadline_ms
    }

    pub fn failures(&self) -> u32 {
        self.failures
    }

    /// Times protection drained with nothing having failed — see [`DmsPolicy::lapses`]'
    /// field docs. Never cleared by recovery.
    pub fn lapses(&self) -> u32 {
        self.lapses
    }

    /// Protection left at `now_ms`, in milliseconds. Zero means none.
    ///
    /// `now_ms` is **wall-clock** epoch milliseconds, and this is one of the few places
    /// in the runtime where that is the right answer rather than a bug. The deadline is
    /// a wall-clock instant held by the venue: `scheduleCancel` fires when *its* clock
    /// passes that number, whether or not our feed is delivering, whether or not our
    /// event clock has moved, and whether or not this process is running. Ageing it in
    /// event time would make a dead feed — the case this whole module exists for — read
    /// as infinite protection, which is the same shape of mistake ADR-0013 §2 fixed for
    /// mark staleness with a two-source clock. Wall time is also why a suspended
    /// *machine* is caught: `Instant` stops across a suspend and the venue's deadline
    /// does not.
    pub fn remaining_ms(&self, now_ms: u64) -> u64 {
        self.deadline_ms
            .map(|d| d.saturating_sub(now_ms))
            .unwrap_or(0)
    }

    /// Record a successful arm. Clears the failure count, so a session that recovers
    /// is allowed to trade again.
    pub fn on_armed(&mut self, deadline_ms: u64, now_ms: u64) -> Escalation {
        self.deadline_ms = Some(deadline_ms);
        self.armed_at_ms = Some(now_ms);
        self.failures = 0;
        // Protection is full again, so the next stall gets its own line rather than
        // being swallowed as a repeat of this one. `lapses` is deliberately not
        // cleared here: recovery is not evidence that nothing happened.
        self.stall_halted = false;
        Escalation::Healthy {
            deadline_ms,
            remaining_ms: deadline_ms.saturating_sub(now_ms),
        }
    }

    /// Grade the protection that is left **before** attempting anything, on the elapsed
    /// wall clock rather than on an outcome.
    ///
    /// [`on_failure`](Self::on_failure) answers "the venue said no". This answers the
    /// question nothing else asks: *is the protection still there?* A process that
    /// froze, was descheduled or wedged inside a call arms nothing and fails nothing,
    /// so it walks ADR-0013 §3's entire ladder without touching a rung — and the arm
    /// that follows the gap succeeds, which resets the deadline and leaves the session
    /// reporting healthy on both sides of it.
    ///
    /// `None` while more than one interval remains, which is the whole of a healthy
    /// session: the venue's lead is at least **3×** the re-arm interval
    /// (`config::MIN_LEAD_TO_REARM_RATIO`), so a re-arm that lands on time leaves at
    /// least two intervals of protection standing. Reaching the halt band therefore
    /// takes a full missed beat, not scheduler jitter — the same one missed beat the
    /// 3:1 ratio is sized to survive.
    ///
    /// That arithmetic is also what makes the cause unambiguous. With no failures
    /// recorded, the last event was a successful arm at `armed_at_ms`, so
    /// `remaining <= interval` implies `now - armed_at >= lead - interval >= 2 ×
    /// interval`: the loop was due to run a whole interval ago and did not. With
    /// failures recorded the drain is already explained, and is reported as such rather
    /// than blamed on a loop that ran perfectly well and was refused.
    ///
    /// Nothing here can *prevent* the gap — a process that is not running runs no
    /// checks. It makes a resumed process escalate rather than continue.
    pub fn on_elapsed(&mut self, now_ms: u64) -> Option<Escalation> {
        // Never armed: there is no deadline to measure against, and grading a session
        // that has not started yet as "unprotected" would shut it down before its first
        // arm. That case is graded on attempts, by `on_failure`.
        let deadline = self.deadline_ms?;
        let remaining = deadline.saturating_sub(now_ms);
        if remaining > self.rearm_interval_ms {
            return None;
        }
        let cause = if self.failures == 0 {
            Cause::LoopLate {
                silent_ms: now_ms.saturating_sub(self.armed_at_ms.unwrap_or(now_ms)),
            }
        } else {
            Cause::ArmFailed {
                attempt: self.failures,
            }
        };
        if remaining == 0 {
            // The deadline has passed: the venue has cancelled everything we had
            // resting, and one of the day's ten triggers is spent. Reported every time,
            // never suppressed — this is the rung that ends the session.
            if matches!(cause, Cause::LoopLate { .. }) {
                self.lapses = self.lapses.saturating_add(1);
            }
            return Some(Escalation::Unprotected { cause });
        }
        if self.stall_halted {
            return None;
        }
        self.stall_halted = true;
        if matches!(cause, Cause::LoopLate { .. }) {
            self.lapses = self.lapses.saturating_add(1);
        }
        Some(Escalation::Halt {
            remaining_ms: remaining,
            cause,
        })
    }

    /// Record a failed arm and grade it.
    pub fn on_failure(&mut self, now_ms: u64) -> Escalation {
        self.failures = self.failures.saturating_add(1);
        let cause = Cause::ArmFailed {
            attempt: self.failures,
        };
        match self.deadline_ms {
            // Never armed: there is no deadline to measure against, so grade on
            // attempts. Refusing to call this "unprotected" immediately avoids
            // shutting a session down over one transient error at startup.
            None if self.failures < UNARMED_FAILURES_BEFORE_UNPROTECTED => Escalation::Halt {
                remaining_ms: 0,
                cause,
            },
            None => Escalation::Unprotected { cause },
            Some(deadline) => {
                let remaining = deadline.saturating_sub(now_ms);
                if remaining == 0 {
                    Escalation::Unprotected { cause }
                } else if remaining <= self.rearm_interval_ms {
                    // One interval or less: the next attempt is the last one that can
                    // land before the deadline. Stop adding exposure now, while it is
                    // still removable.
                    Escalation::Halt {
                        remaining_ms: remaining,
                        cause,
                    }
                } else {
                    Escalation::Retry {
                        remaining_ms: remaining,
                        cause,
                    }
                }
            }
        }
    }

    /// How long to wait before the next attempt.
    ///
    /// A failure retries at a quarter of the interval rather than waiting a full one:
    /// once a re-arm has failed, the deadline is fixed and every millisecond spent
    /// waiting is protection spent. It is floored so a persistent failure cannot turn
    /// into a request storm against a venue that is already unhappy.
    pub fn next_attempt_in(&self) -> Duration {
        if self.failures == 0 {
            Duration::from_millis(self.rearm_interval_ms)
        } else {
            Duration::from_millis((self.rearm_interval_ms / 4).max(500))
        }
    }
}

/// The venue side of the switch, as a trait so the loop is testable without a network
/// or a key.
#[async_trait]
pub trait DeadMansSwitch: Send + Sync {
    /// Push the cancel-everything deadline `lead_ms` into the future, returning the
    /// absolute deadline the venue now holds.
    async fn arm(&self, lead_ms: u64) -> Result<u64, ProviderError>;
    /// Remove the deadline.
    async fn disarm(&self) -> Result<(), ProviderError>;
}

#[async_trait]
impl DeadMansSwitch for ExchangeClient {
    async fn arm(&self, lead_ms: u64) -> Result<u64, ProviderError> {
        self.arm_dead_mans_switch(lead_ms).await
    }

    async fn disarm(&self) -> Result<(), ProviderError> {
        self.cancel_scheduled_cancel().await
    }
}

/// Shared state the re-arm loop publishes for the status line and for shutdown.
#[derive(Debug, Default)]
pub struct DmsState {
    /// Absolute deadline the venue holds, epoch ms. 0 = not armed.
    deadline_ms: AtomicU64,
    failures: AtomicU64,
    lapses: AtomicU64,
}

impl DmsState {
    pub fn deadline_ms(&self) -> Option<u64> {
        match self.deadline_ms.load(Ordering::Relaxed) {
            0 => None,
            d => Some(d),
        }
    }

    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    /// Times protection drained into the halt band or past it with nothing having
    /// failed.
    ///
    /// Published as its own number because the successful re-arm that follows a stall
    /// destroys every other trace: `deadline_ms` is pushed forward, `failures` is zero
    /// throughout, the halt is cleared within microseconds and two consecutive status
    /// lines read `dms 0s` then `dms 55s`. A counter that only rises is the one thing
    /// an operator can still see afterwards.
    pub fn lapses(&self) -> u64 {
        self.lapses.load(Ordering::Relaxed)
    }

    fn publish(&self, policy: &DmsPolicy) {
        self.deadline_ms
            .store(policy.deadline_ms.unwrap_or(0), Ordering::Relaxed);
        self.failures
            .store(u64::from(policy.failures), Ordering::Relaxed);
        self.lapses
            .store(u64::from(policy.lapses), Ordering::Relaxed);
    }
}

/// Wall-clock epoch milliseconds. The switch is a venue-side deadline, so it is the
/// only part of the runtime that is *supposed* to reason in wall time rather than
/// event time — the venue's clock is the one that fires it.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The console line one escalation produces, or `None` when it says nothing.
///
/// Split out of [`apply`] so the wording can be **asserted** rather than reviewed. It
/// is a pure function of the escalation for the same reason [`DmsPolicy`] is a pure
/// state machine: the branch that matters only runs on the worst day, and the one that
/// matters most is the one an operator reads at 3am.
pub fn console_line(escalation: Escalation) -> Option<String> {
    match escalation {
        Escalation::Healthy { .. } => None,
        Escalation::Retry {
            remaining_ms,
            cause,
        } => Some(format!(
            "dead-man's switch: {}, {remaining_ms} ms of protection left - retrying",
            cause.clause()
        )),
        Escalation::Halt {
            remaining_ms,
            cause,
        } => Some(format!(
            "dead-man's switch: {}, only {remaining_ms} ms of protection left - \
             HALTING new orders (cancels still allowed)",
            cause.clause()
        )),
        // **"The deadline has passed", not "the switch has fired."** This line used to
        // claim the second, and the claim was wrong in the only way that matters: it is
        // what an operator acts on at 3am. Nothing on this side observes the venue's
        // switch. A `scheduleCancel` deadline elapsing means *our* protection is gone,
        // which is reason enough to shut down; it does **not** mean the venue executed
        // the cancel, and during the 2026-07-27 testnet outage it had not — the order
        // placed before the deadline lapsed was still resting when the venue came back.
        // An operator told the switch fired stops looking for resting orders, which is
        // the one thing they must not stop doing here. The honest line sends them to
        // `openOrders`.
        Escalation::Unprotected { cause } => Some(format!(
            "dead-man's switch: UNPROTECTED - {} and no protection remains; \
             the deadline we armed has passed - shutting the session down. \
             Whether the venue acted on that deadline is NOT known from here: \
             check openOrders before concluding the account is clean",
            cause.clause()
        )),
    }
}

/// Apply an escalation to the session's switches, returning `true` when the session
/// should shut down.
pub fn apply(escalation: Escalation, halt: &HaltSwitch, health: &SessionHealth) -> bool {
    if let Some(line) = console_line(escalation) {
        eprintln!("{line}");
    }
    match escalation {
        Escalation::Healthy { .. } => {
            // Only clears a halt this loop raised; a shutdown-stopped switch is
            // terminal and `resume` cannot revive it.
            halt.resume();
            false
        }
        Escalation::Retry { .. } => false,
        Escalation::Halt { .. } => {
            halt.halt();
            health.note_dms_halt();
            false
        }
        Escalation::Unprotected { .. } => {
            halt.halt();
            health.note_dms_halt();
            true
        }
    }
}

/// The re-arm loop.
///
/// It runs until `stop` is signalled and is deliberately the **last** task the
/// supervisor cancels: every other task can die and the venue-side deadline still
/// protects us, but if this loop stops while orders are resting, protection lapses on
/// a clock nobody is watching. Returning `Err` means the session must shut down.
pub async fn run<D: DeadMansSwitch + ?Sized>(
    dms: &D,
    policy: &mut DmsPolicy,
    state: &DmsState,
    halt: &Arc<HaltSwitch>,
    health: &SessionHealth,
    stop: &tokio::sync::Notify,
) {
    loop {
        // Before anything else, and on the wall clock: how much protection is actually
        // left *now*. This runs on the success path too, which is the point — the case
        // the switch exists for is a process that stopped arming, not one whose arms
        // are being refused, and such a process produces no error to grade. If we were
        // frozen through the deadline this is the only line of code that will ever
        // notice; the arm below would push a fresh deadline and report healthy over the
        // gap. It notices and reacts — it cannot close a window during which no code of
        // ours ran at all.
        if let Some(lapse) = policy.on_elapsed(now_ms()) {
            state.publish(policy);
            if apply(lapse, halt, health) {
                // Protection is already gone, so there is nothing left for another arm
                // to preserve — and pushing a fresh deadline here would hand the
                // shutdown sequence (ADR-0013 §6) a deadline it then has to reason
                // about, for a session that is ending either way.
                return;
            }
        }

        let escalation = match dms.arm(policy.lead_ms()).await {
            Ok(deadline) => policy.on_armed(deadline, now_ms()),
            Err(e) => {
                eprintln!("dead-man's switch: arm failed: {e}");
                policy.on_failure(now_ms())
            }
        };
        state.publish(policy);
        if apply(escalation, halt, health) {
            return;
        }

        tokio::select! {
            // Biased so a shutdown signal is honoured even when the timer is also
            // ready: the shutdown sequence decides whether the switch stays armed,
            // and one more re-arm would push a deadline it has to reason about.
            biased;
            _ = stop.notified() => return,
            _ = tokio::time::sleep(policy.next_attempt_in()) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MIN_LEAD_TO_REARM_RATIO;

    const LEAD: u64 = 60_000;
    const INTERVAL: u64 = 20_000;

    fn policy() -> DmsPolicy {
        DmsPolicy::new(LEAD, INTERVAL)
    }

    #[test]
    fn the_unprotected_line_reports_our_own_deadline_and_never_claims_the_venue_acted() {
        // Measured on 2026-07-27: Hyperliquid testnet went down, this rung fired, the
        // line said "the venue-side switch has fired" — and when the venue came back the
        // order placed before the outage was **still resting**. Nothing on this side can
        // observe the venue's switch; it can only observe a deadline elapsing. The
        // failure the wording causes is specific and expensive: an operator who is told
        // the switch fired stops looking for resting orders, which is the one thing they
        // must not stop doing on exactly this line.
        let line = console_line(Escalation::Unprotected {
            cause: Cause::ArmFailed { attempt: 3 },
        })
        .expect("the terminal rung always says something");
        assert!(
            !line.contains("has fired"),
            "an inference stated as an observation: {line}"
        );
        assert!(
            line.contains("the deadline we armed has passed"),
            "the honest claim is about our clock: {line}"
        );
        assert!(
            line.contains("openOrders"),
            "and it must send the operator somewhere: {line}"
        );
        assert!(line.contains("re-arm failed (attempt 3)"), "{line}");

        // The other rungs are unchanged, and the healthy one still says nothing — a
        // switch that narrated every successful arm would bury this line.
        assert!(console_line(Escalation::Healthy {
            deadline_ms: 1,
            remaining_ms: 1
        })
        .is_none());
        assert!(console_line(Escalation::Halt {
            remaining_ms: 5_000,
            cause: Cause::LoopLate { silent_ms: 80_000 },
        })
        .expect("halting is worth a line")
        .contains("HALTING new orders (cancels still allowed)"));
    }

    #[test]
    fn a_healthy_rearm_clears_earlier_failures() {
        let mut p = policy();
        p.on_armed(100_000, 40_000);
        assert!(matches!(p.on_failure(45_000), Escalation::Retry { .. }));
        assert_eq!(p.failures(), 1);
        assert!(matches!(
            p.on_armed(110_000, 50_000),
            Escalation::Healthy {
                remaining_ms: 60_000,
                ..
            }
        ));
        assert_eq!(p.failures(), 0, "recovery must let the session trade again");
    }

    #[test]
    fn a_failed_rearm_halts_before_protection_expires_not_after() {
        // The point of the whole module: the orders placed in the last interval are
        // the ones that get stranded, so stop placing while protection still remains.
        let mut p = policy();
        p.on_armed(100_000, 40_000); // 60 s of protection

        // 40 s left: one missed beat is what the 3:1 lead ratio is sized for.
        assert!(matches!(
            p.on_failure(60_000),
            Escalation::Retry {
                remaining_ms: 40_000,
                cause: Cause::ArmFailed { attempt: 1 }
            }
        ));
        // 15 s left, under one 20 s interval: the next attempt is the last that can
        // land, so stop adding exposure now.
        assert!(matches!(
            p.on_failure(85_000),
            Escalation::Halt {
                remaining_ms: 15_000,
                cause: Cause::ArmFailed { attempt: 2 }
            }
        ));
        // Past the deadline: the venue has cancelled everything and we are naked.
        assert!(matches!(
            p.on_failure(100_001),
            Escalation::Unprotected {
                cause: Cause::ArmFailed { attempt: 3 }
            }
        ));
    }

    #[test]
    fn a_frozen_process_that_resumes_escalates_instead_of_arming_over_the_gap() {
        // The soak's reproducer, in policy terms: `kill -STOP` a session with a 60 s
        // lead, wait 80 s, `kill -CONT`. Every arm before the freeze succeeded and the
        // arm after it succeeds too, so nothing ever calls `on_failure` and the
        // error-graded ladder is never consulted. What the venue did in between is fire
        // the switch. Observed before this check existed: `dms 0s` on one status line,
        // `dms 55s` on the next, no `HALTED`, no `UNPROTECTED`, nothing on stderr.
        let mut p = policy();
        p.on_armed(60_000, 0);
        assert_eq!(p.on_elapsed(20_000), None, "the beat it was scheduled for");

        // 80 s later. The deadline passed 20 s ago and the venue has cancelled
        // everything we had resting.
        let e = p.on_elapsed(80_000);
        assert!(
            matches!(
                e,
                Some(Escalation::Unprotected {
                    cause: Cause::LoopLate { silent_ms: 80_000 }
                })
            ),
            "{e:?}"
        );
        assert_eq!(p.lapses(), 1);
        assert_eq!(
            p.failures(),
            0,
            "nothing failed - blaming the venue would send an operator to the wrong box"
        );
    }

    #[test]
    fn a_stalled_loop_halts_while_the_orders_it_placed_can_still_be_cancelled() {
        // The soak's 48 s freeze, which reached `dms 9s` — deep inside the band
        // ADR-0013 §3 halts in — and said nothing. Halting *before* the deadline is the
        // whole point: the orders placed in the final interval are the ones stranded
        // when the venue-side switch fires, and 9 s before the deadline they can still
        // be cancelled by us.
        let mut p = policy();
        p.on_armed(60_000, 0);
        let e = p.on_elapsed(51_000);
        assert!(
            matches!(
                e,
                Some(Escalation::Halt {
                    remaining_ms: 9_000,
                    cause: Cause::LoopLate { silent_ms: 51_000 }
                })
            ),
            "{e:?}"
        );
        assert_eq!(p.lapses(), 1);
        assert!(
            p.remaining_ms(51_000) > 0,
            "the halt has to land while protection still remains, not after"
        );
    }

    #[test]
    fn an_ordinary_rearm_cycle_never_halts_a_healthy_session() {
        // The bands exist so ordinary jitter is tolerated, and a safety check that
        // stops a healthy session is not a safety check. Run the tightest configuration
        // the config validator permits — `lead == 3 × interval` — because that is where
        // the margin is thinnest.
        let interval = 20_000u64;
        let lead = interval * MIN_LEAD_TO_REARM_RATIO;
        let mut p = DmsPolicy::new(lead, interval);
        let mut now = 0u64;
        for _ in 0..50 {
            p.on_armed(now + lead, now);
            now += interval;
            assert_eq!(p.on_elapsed(now), None, "a re-arm that landed on time");
        }
        assert_eq!(p.lapses(), 0);

        // And with jitter: anything short of a whole missed beat still leaves more than
        // one interval standing, which is exactly what the 3:1 ratio is sized for.
        p.on_armed(now + lead, now);
        assert_eq!(
            p.on_elapsed(now + 2 * interval - 1),
            None,
            "late, not lapsed"
        );
        // A full interval late is a missed beat, and that is the rung.
        assert!(matches!(
            p.on_elapsed(now + 2 * interval),
            Some(Escalation::Halt { .. })
        ));
    }

    #[test]
    fn a_recovered_stall_leaves_the_session_trading_and_the_evidence_standing() {
        // Two halves of one property. A halt raised by the elapsed check must clear on
        // the next successful arm — protection is full again, and a session left halted
        // forever by a scheduling hiccup is its own outage. But the *count* must not
        // clear, because that same successful arm is what erases every other trace: the
        // deadline is pushed forward, `failures` was zero throughout, and the status
        // line goes straight back to reading healthy.
        let halt = HaltSwitch::new();
        let health = SessionHealth::new(0);
        let mut p = policy();
        p.on_armed(60_000, 0);

        let lapse = p
            .on_elapsed(51_000)
            .expect("a stall this deep must escalate");
        assert!(!apply(lapse, &halt, &health));
        assert!(
            !halt.is_accepting(),
            "no new exposure while protection is thin"
        );
        assert_eq!(health.dms_halts(), 1);

        assert!(!apply(p.on_armed(111_000, 51_050), &halt, &health));
        assert!(halt.is_accepting(), "a recovered session must trade again");
        assert_eq!(
            p.lapses(),
            1,
            "recovery is not evidence that nothing happened"
        );
        assert_eq!(p.failures(), 0);

        // …and the halt is not re-raised on the next healthy beat.
        assert_eq!(p.on_elapsed(71_050), None);
        assert!(halt.is_accepting());
    }

    #[test]
    fn a_venue_outage_is_never_reported_as_a_stalled_loop() {
        // The elapsed check runs on every pass, including the ones where a re-arm has
        // already been refused. It must not then blame the loop — the loop ran, and it
        // must not double-count a drain `on_failure` is already grading and printing.
        let mut p = policy();
        p.on_armed(100_000, 40_000); // 60 s of protection
        assert!(matches!(p.on_failure(60_000), Escalation::Retry { .. }));

        // Retrying at a quarter interval, now inside the halt band.
        let first = p.on_elapsed(85_000);
        assert!(
            matches!(
                first,
                Some(Escalation::Halt {
                    remaining_ms: 15_000,
                    cause: Cause::ArmFailed { attempt: 1 }
                })
            ),
            "{first:?}"
        );
        assert_eq!(p.lapses(), 0, "the loop is running; the venue is refusing");
        // Said once. Repeating it every five seconds buries the line that mattered.
        assert_eq!(p.on_elapsed(90_000), None);
        assert_eq!(p.on_elapsed(95_000), None);
        // Except the rung that ends the session, which is never suppressed.
        assert!(matches!(
            p.on_elapsed(100_001),
            Some(Escalation::Unprotected {
                cause: Cause::ArmFailed { attempt: 1 }
            })
        ));
    }

    #[test]
    fn a_switch_that_has_never_armed_is_not_shut_down_by_the_elapsed_check() {
        // A session starts halted with no deadline, and `remaining_ms` reports zero for
        // "never armed" exactly as it does for "expired". Grading the first on the
        // ladder would shut every session down before its first arm ever left. The
        // unarmed case is graded on attempts, by `on_failure`, and only that.
        let mut p = policy();
        assert_eq!(p.remaining_ms(0), 0);
        assert_eq!(p.on_elapsed(0), None);
        assert_eq!(p.on_elapsed(10_000_000), None);
        assert_eq!(p.lapses(), 0);
    }

    #[test]
    fn a_switch_that_never_armed_halts_first_and_gives_up_after_three_tries() {
        // With no deadline there is nothing to measure, so a startup failure must not
        // instantly kill the session — but it must not trade either.
        let mut p = policy();
        assert!(matches!(p.on_failure(1_000), Escalation::Halt { .. }));
        assert!(matches!(p.on_failure(2_000), Escalation::Halt { .. }));
        assert!(matches!(
            p.on_failure(3_000),
            Escalation::Unprotected {
                cause: Cause::ArmFailed { attempt: 3 }
            }
        ));
    }

    #[test]
    fn a_failure_retries_faster_than_the_normal_cadence() {
        // After a failure the deadline is fixed, so waiting a full interval spends
        // protection that cannot be recovered.
        let mut p = policy();
        assert_eq!(p.next_attempt_in(), Duration::from_millis(INTERVAL));
        p.on_armed(100_000, 40_000);
        assert_eq!(p.next_attempt_in(), Duration::from_millis(INTERVAL));
        p.on_failure(60_000);
        assert_eq!(p.next_attempt_in(), Duration::from_millis(INTERVAL / 4));

        // …but never fast enough to storm a venue that is already failing.
        let mut tight = DmsPolicy::new(5_000, 1_000);
        tight.on_failure(0);
        assert_eq!(tight.next_attempt_in(), Duration::from_millis(500));
    }

    #[test]
    fn an_escalation_moves_the_session_switch_and_only_shuts_down_when_unprotected() {
        let halt = HaltSwitch::new();
        let health = SessionHealth::new(0);

        assert!(!apply(
            Escalation::Retry {
                remaining_ms: 40_000,
                cause: Cause::ArmFailed { attempt: 1 }
            },
            &halt,
            &health
        ));
        assert!(halt.is_accepting(), "a retry does not stop trading");

        assert!(!apply(
            Escalation::Halt {
                remaining_ms: 5_000,
                cause: Cause::ArmFailed { attempt: 2 }
            },
            &halt,
            &health
        ));
        assert!(!halt.is_accepting());
        assert_eq!(health.dms_halts(), 1);

        assert!(!apply(
            Escalation::Healthy {
                deadline_ms: 1,
                remaining_ms: 60_000
            },
            &halt,
            &health
        ));
        assert!(halt.is_accepting(), "recovery resumes trading");

        assert!(
            apply(
                Escalation::Unprotected {
                    cause: Cause::ArmFailed { attempt: 3 }
                },
                &halt,
                &health
            ),
            "no protection left must end the session"
        );
        assert!(!halt.is_accepting());
    }

    #[test]
    fn published_state_tracks_the_policy() {
        let mut p = policy();
        let s = DmsState::default();
        assert_eq!(s.deadline_ms(), None);
        p.on_armed(123_456, 100_000);
        s.publish(&p);
        assert_eq!(s.deadline_ms(), Some(123_456));
        assert_eq!(s.failures(), 0);
        assert_eq!(s.lapses(), 0);
        p.on_failure(110_000);
        s.publish(&p);
        assert_eq!(s.failures(), 1);

        // Counted and not published is the same as uncounted: a stall that recovers
        // leaves nothing else behind for the status line to show.
        let mut q = policy();
        q.on_armed(60_000, 0);
        q.on_elapsed(51_000);
        q.on_armed(120_000, 51_000);
        s.publish(&q);
        assert_eq!(s.lapses(), 1);
        assert_eq!(
            s.failures(),
            0,
            "nothing failed, and the line must not say so"
        );
    }

    /// A spy switch that fails `fail_first` times, then succeeds.
    struct SpySwitch {
        fail_first: std::sync::atomic::AtomicU32,
        arms: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl DeadMansSwitch for SpySwitch {
        async fn arm(&self, lead_ms: u64) -> Result<u64, ProviderError> {
            self.arms.fetch_add(1, Ordering::Relaxed);
            if self
                .fail_first
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(ProviderError::Network("spy".into()));
            }
            Ok(now_ms() + lead_ms)
        }
        async fn disarm(&self) -> Result<(), ProviderError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn the_loop_stops_on_notify_and_leaves_the_switch_armed() {
        // Shutdown decides whether the deadline stands; the loop must not push it
        // forward once asked to stop.
        let spy = SpySwitch {
            fail_first: std::sync::atomic::AtomicU32::new(0),
            arms: std::sync::atomic::AtomicU32::new(0),
        };
        let halt = Arc::new(HaltSwitch::new());
        let health = SessionHealth::new(0);
        let state = DmsState::default();
        let stop = tokio::sync::Notify::new();
        let mut p = DmsPolicy::new(5_000, 5_000);

        // One arm, then the stop wins the select because it is biased first.
        stop.notify_one();
        run(&spy, &mut p, &state, &halt, &health, &stop).await;
        assert_eq!(spy.arms.load(Ordering::Relaxed), 1);
        assert!(state.deadline_ms().is_some(), "the venue still holds it");
    }

    #[tokio::test]
    async fn the_loop_returns_once_protection_is_gone() {
        // Three failed first arms with no deadline to fall back on ends the session
        // rather than looping forever with no protection.
        let spy = SpySwitch {
            fail_first: std::sync::atomic::AtomicU32::new(3),
            arms: std::sync::atomic::AtomicU32::new(0),
        };
        let halt = Arc::new(HaltSwitch::new());
        let health = SessionHealth::new(0);
        let state = DmsState::default();
        let stop = tokio::sync::Notify::new();
        // A 1 ms floor is not reachable (500 ms is), so keep the interval tiny and let
        // the retry floor cap it: three attempts at 500 ms is 1 s of test time.
        let mut p = DmsPolicy::new(5_000, 5_000);
        tokio::time::timeout(
            Duration::from_secs(5),
            run(&spy, &mut p, &state, &halt, &health, &stop),
        )
        .await
        .expect("the loop must return, not hang");
        assert_eq!(spy.arms.load(Ordering::Relaxed), 3);
        assert!(!halt.is_accepting());
    }

    /// A switch whose deadline is already gone by the time the loop next looks — what a
    /// process that froze through its own lead observes on resume, without needing to
    /// freeze a test.
    struct LapsedSwitch {
        arms: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl DeadMansSwitch for LapsedSwitch {
        async fn arm(&self, _lead_ms: u64) -> Result<u64, ProviderError> {
            self.arms.fetch_add(1, Ordering::Relaxed);
            Ok(now_ms())
        }
        async fn disarm(&self) -> Result<(), ProviderError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn the_loop_ends_the_session_when_it_wakes_past_its_own_deadline() {
        // The wiring, from the outside: every `arm` here succeeds, so nothing reaches
        // `on_failure` and the error-graded ladder stays untouched — which is exactly
        // the shape of the session the soak froze. Before the elapsed check the loop
        // ran forever, re-arming happily over a deadline that had already fired.
        let spy = LapsedSwitch {
            arms: std::sync::atomic::AtomicU32::new(0),
        };
        let halt = Arc::new(HaltSwitch::new());
        let health = SessionHealth::new(0);
        let state = DmsState::default();
        let stop = tokio::sync::Notify::new();
        let mut p = DmsPolicy::new(60, 20);

        tokio::time::timeout(
            Duration::from_secs(5),
            run(&spy, &mut p, &state, &halt, &health, &stop),
        )
        .await
        .expect("a session with no protection left must end, not loop");

        assert!(
            !halt.is_accepting(),
            "and it must stop trading on the way out"
        );
        assert_eq!(health.dms_halts(), 1);
        assert_eq!(
            state.lapses(),
            1,
            "the one number that outlives the recovery"
        );
        assert_eq!(
            spy.arms.load(Ordering::Relaxed),
            1,
            "no fresh deadline for the shutdown sequence to reason about (ADR-0013 §6)"
        );
    }
}
