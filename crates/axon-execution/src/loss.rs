//! The loss-based kill switch: the one gate in this pipeline that is about **money**
//! rather than about size.
//!
//! Every other pre-trade limit in this crate bounds a *quantity* —
//! [`RiskLimits::max_position`], `max_notional`, `max_order_qty`. All three answer "how
//! big may this get", and none of them answers "how much may this cost". A session
//! whose strategy is quietly wrong stays inside every one of them for as long as anyone
//! lets it run, which is what [ADR-0036] left open on purpose:
//!
//! > A loss limit that pulls the trading switch is a risk control, belongs with the
//! > other gates in `axon-execution`, and needs its own argument — not least because
//! > the first thing it would do is refuse the orders that *reduce* a position.
//!
//! This module is that argument, and the parenthesis is its whole shape.
//!
//! ## Why this is not [`HaltSwitch`]
//!
//! [`HaltSwitch::halt`](crate::HaltSwitch::halt) refuses **every** placement. That is
//! right for the two situations it was built for — a lapsing dead-man's switch and a
//! shutdown sweep — because in both of them the correct next action is a *cancel*, and
//! cancels pass through. It is wrong here. A session that has lost more than its
//! operator declared does not want to stop acting; it wants to **get out**, and getting
//! out is an order. Halting it strands the exposure that caused the loss in the market
//! that is causing it, which is the same trap [ADR-0031] identified for risk-gating a
//! cancel and is strictly worse: a cancel merely fails to reduce risk, and this would
//! actively hold it.
//!
//! So a tripped limit puts the session into **de-risk-only**: an order is admitted if
//! and only if it strictly takes exposure off the position we actually hold
//! ([`axon_risk::reduces_exposure`]). Cancels were never gated and still are not.
//!
//! ## Three decisions worth stating rather than discovering
//!
//! **It is one-way.** Nothing un-trips it but a restart, and the reason is the mark: a
//! bound measured partly on unrealized P&L is measured on a number that moves both
//! ways, so a re-arming limit would let a session resume trading on a bounce and stop
//! again on the next tick — flickering in and out of the market at exactly the moment
//! nobody is watching, and doing it fastest when volatility is highest. The same
//! reasoning makes [`HaltState::Stopped`](crate::HaltState::Stopped) terminal.
//!
//! **It measures the position we hold, not the one risk projects.**
//! [`OrderTracker::risk_position`](crate::OrderTracker::risk_position) inflates the
//! position by every resting order, which is exactly right for a size cap and exactly
//! wrong here: flat with a resting buy projects long, so a *sell* would read as a
//! reduction and would in fact open a short. The de-risk question is about filled
//! exposure and nothing else.
//!
//! **It judges two independent numbers and trips on either.** The session bound is our
//! own average-cost accounting; the day bound is the venue's `accountValue` against a
//! baseline that **survives a restart**. They are the same pair [ADR-0036] reports side
//! by side and refuses to reconcile, used here for the reason the pair exists: a
//! crash-restart loop resets our accounting and cannot reset the venue's, and a daily
//! limit that a restart clears is not a daily limit.
//!
//! [ADR-0031]: https://docs.rs/axon
//! [ADR-0036]: https://docs.rs/axon
//! [`RiskLimits::max_position`]: axon_risk::RiskLimits::max_position

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use axon_core::Decimal;

/// Declared loss bounds, as **magnitudes** in quote units. `0` is *no bound declared*.
///
/// Magnitudes rather than percentages, and zero-means-unset rather than
/// zero-means-stop, for the same two reasons [`crate::marks`] and the P&L alarms give:
/// the account this runs against is shared so its balance is not a scale anyone chose,
/// and a default that fires is a default nobody chose.
///
/// A negative value is the typo that fails in the dangerous direction — the figure
/// these are compared against is itself usually negative, so `-5` written where `5` was
/// meant is a limit that is tripped from the first tick. [`LossLimits::validate`]
/// refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LossLimits {
    /// Bound on this **session's** own bottom line, from our average-cost accounting.
    pub session: Decimal,
    /// Bound on the **UTC day's** equity change, from the venue's `accountValue`
    /// against a baseline that survives a restart.
    pub day: Decimal,
}

impl LossLimits {
    /// Whether anything here can ever fire.
    pub fn is_declared(&self) -> bool {
        self.session > Decimal::ZERO || self.day > Decimal::ZERO
    }

    /// Reject the sign error that would arm the switch permanently at startup.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [("session", self.session), ("day", self.day)] {
            if v < Decimal::ZERO {
                return Err(format!(
                    "max_{name}_loss {v} is negative; it is a magnitude, and the figure \
                     it is compared against is itself usually negative - a negative \
                     bound is a kill switch that is tripped before the first order"
                ));
            }
        }
        Ok(())
    }
}

/// Which bound tripped, on what reading.
///
/// Carries the numbers rather than a flag, because the line an operator reads at 3am
/// has to say *how far past* the bound the session got: "the day limit tripped" and
/// "the day limit tripped by two cents" send them to different places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossBreach {
    pub scope: LossScope,
    /// The loss as a **magnitude** — positive means money lost.
    pub loss: Decimal,
    /// The bound it passed.
    pub limit: Decimal,
    /// Whether the reading behind it came from a mark, and is therefore an opinion
    /// about an open position rather than a closed trade.
    ///
    /// Reported because the two deserve different urgency: a realized loss is spent and
    /// a marked one can come back. It changes nothing about the trip — see the
    /// one-way argument in the module docs.
    pub marked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossScope {
    /// Our own accounting, since this process started.
    Session,
    /// The venue's `accountValue`, since the UTC day's baseline.
    Day,
}

impl fmt::Display for LossScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LossScope::Session => f.write_str("session"),
            LossScope::Day => f.write_str("day"),
        }
    }
}

impl fmt::Display for LossBreach {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} loss {:.4} past the declared {:.4}",
            self.scope, self.loss, self.limit
        )?;
        if self.marked {
            // Named, so the line does not read as a closed loss when it is a mark.
            f.write_str(" (marked, not realized)")?;
        }
        Ok(())
    }
}

/// The switch. Clone the `Arc`, not this.
///
/// [`Self::is_tripped`] is one relaxed atomic load, because it sits on the submit path
/// and this crate's rule is that risk must never be the bottleneck. The breach *detail*
/// lives behind a lock that only the status line reads.
#[derive(Debug)]
pub struct LossLimiter {
    limits: LossLimits,
    tripped: AtomicBool,
    detail: RwLock<Option<LossBreach>>,
}

impl LossLimiter {
    pub fn new(limits: LossLimits) -> Self {
        Self {
            limits,
            tripped: AtomicBool::new(false),
            detail: RwLock::new(None),
        }
    }

    /// A limiter that can never fire — what a session with no declared bound gets, and
    /// what every offline path gets.
    ///
    /// Present so the wiring is unconditional: the code path a live session takes is the
    /// one the tests take, the same reason [`crate::marks::MarkCache::never_expires`]
    /// and `LatencyBook::undeclared` exist.
    pub fn undeclared() -> Self {
        Self::new(LossLimits::default())
    }

    pub fn limits(&self) -> &LossLimits {
        &self.limits
    }

    /// Whether the session is in de-risk-only mode. The hot-path read.
    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::Acquire)
    }

    /// What tripped it, for the status line. `None` while clear, and also when the
    /// detail lock is poisoned — a missing explanation must never be able to make
    /// [`Self::is_tripped`] lie, so the two are read independently.
    pub fn breach(&self) -> Option<LossBreach> {
        self.detail.read().ok().and_then(|d| *d)
    }

    /// Judge a fresh reading and trip if either declared bound has been passed.
    ///
    /// `session_net` is our own bottom line (signed: negative is a loss) or `None` when
    /// it cannot be computed — a poisoned tracker, or a held position with no fresh
    /// mark. `session_realized` is the part that never needed a mark and is therefore
    /// **always** available, and it is what the session bound falls back to.
    ///
    /// **The fallback is the point.** [ADR-0036] §2 makes `net()` `None` the moment any
    /// held symbol goes unpriced, and the loss *warning* is silent for as long as that
    /// lasts. Silence is defensible for a warning beside a louder `POSITION UNPRICED`;
    /// it is not defensible for a kill switch, because a dead feed would be a way to
    /// switch the limit off. So an unpriced session is still judged on what it has
    /// actually closed and paid, which cannot be wrong in either direction.
    ///
    /// `day_loss` is the venue's own answer — the day baseline minus `accountValue` now,
    /// as a **magnitude** — or `None` before the venue has answered or before a baseline
    /// exists. It is never derived from the session figures; that independence is the
    /// whole reason there are two.
    ///
    /// Returns the breach **only on the pass that trips**, so a caller can log it once
    /// rather than once per status line.
    pub fn observe(
        &self,
        session_net: Option<Decimal>,
        session_realized: Decimal,
        day_loss: Option<Decimal>,
    ) -> Option<LossBreach> {
        if self.is_tripped() {
            return None;
        }
        let breach = self.judge(session_net, session_realized, day_loss)?;
        self.trip(breach);
        Some(breach)
    }

    /// The pure half of [`Self::observe`] — no state, so every branch is assertable.
    pub fn judge(
        &self,
        session_net: Option<Decimal>,
        session_realized: Decimal,
        day_loss: Option<Decimal>,
    ) -> Option<LossBreach> {
        if self.limits.session > Decimal::ZERO {
            // `net()` when it exists, and what has actually been closed and paid when it
            // does not. Both are signed with a loss negative, so the bound is passed
            // when the figure is below `-limit`.
            let (figure, marked) = match session_net {
                Some(n) => (n, true),
                None => (session_realized, false),
            };
            if figure < -self.limits.session {
                return Some(LossBreach {
                    scope: LossScope::Session,
                    loss: -figure,
                    limit: self.limits.session,
                    marked,
                });
            }
        }
        if self.limits.day > Decimal::ZERO {
            if let Some(loss) = day_loss {
                if loss > self.limits.day {
                    return Some(LossBreach {
                        scope: LossScope::Day,
                        // `accountValue` is marked to the venue's own price, so any
                        // open position makes this an opinion too. Said plainly rather
                        // than assumed either way.
                        loss,
                        limit: self.limits.day,
                        marked: true,
                    });
                }
            }
        }
        None
    }

    /// Trip the switch directly. One-way: a second call cannot change the reason, so
    /// the breach recorded is the **first** one, which is the one that explains the
    /// state.
    pub fn trip(&self, breach: LossBreach) {
        if let Ok(mut d) = self.detail.write() {
            if d.is_none() {
                *d = Some(breach);
            }
        }
        self.tripped.store(true, Ordering::Release);
    }
}

impl Default for LossLimiter {
    fn default() -> Self {
        Self::undeclared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn limiter(session: Decimal, day: Decimal) -> LossLimiter {
        LossLimiter::new(LossLimits { session, day })
    }

    #[test]
    fn an_undeclared_bound_can_never_fire() {
        // The default, and what every offline path and every config written before this
        // existed gets. A kill switch that arrives switched on is a kill switch that
        // takes a session down on an upgrade.
        let l = LossLimiter::undeclared();
        assert!(l
            .observe(Some(dec!(-1_000_000)), dec!(-1_000_000), Some(dec!(1e9)))
            .is_none());
        assert!(!l.is_tripped());
        assert!(!l.limits().is_declared());
    }

    #[test]
    fn a_session_loss_past_the_bound_trips_once_and_reports_the_first_reason() {
        let l = limiter(dec!(0.5), Decimal::ZERO);
        assert!(l.observe(Some(dec!(-0.4)), dec!(-0.4), None).is_none());
        assert!(!l.is_tripped(), "at 0.4 of a 0.5 bound");

        let b = l
            .observe(Some(dec!(-0.6)), dec!(-0.5), None)
            .expect("past the bound");
        assert_eq!(b.scope, LossScope::Session);
        assert_eq!(b.loss, dec!(0.6));
        assert!(l.is_tripped());

        // Only the pass that trips reports, so a caller logs it once rather than once
        // per status line — and the recorded reason stays the first one.
        assert!(l.observe(Some(dec!(-9)), dec!(-9), None).is_none());
        assert_eq!(l.breach().unwrap().loss, dec!(0.6));
    }

    #[test]
    fn an_unpriced_session_is_judged_on_what_it_has_actually_closed_and_paid() {
        // The hole a mark-only bound would leave, and it is the dangerous kind: ADR-0036
        // makes `net()` None whenever a held symbol has no fresh price, so a dead feed
        // would be a way to switch the kill switch off — on precisely the session that
        // has a position and cannot see it. Realized-less-fees never needs a mark and
        // cannot be wrong in either direction, so that is what it falls back to.
        let l = limiter(dec!(0.5), Decimal::ZERO);
        let b = l
            .observe(None, dec!(-0.75), None)
            .expect("no mark is not a reason to stop judging");
        assert_eq!(b.loss, dec!(0.75));
        assert!(
            !b.marked,
            "and the line must say this one is spent, not marked"
        );
        assert!(b.to_string().contains("session loss 0.7500"), "{b}");
    }

    #[test]
    fn an_unpriced_session_inside_its_realized_bound_keeps_trading() {
        // The mirror. The fallback must not become "assume the worst": a session whose
        // closed trades are fine and whose open position cannot be priced has not
        // breached anything, and stopping it would make a feed hiccup a trading halt.
        let l = limiter(dec!(0.5), Decimal::ZERO);
        assert!(l.observe(None, dec!(-0.1), None).is_none());
        assert!(!l.is_tripped());
    }

    #[test]
    fn the_day_bound_is_judged_on_the_venues_own_number_and_trips_independently() {
        // Two accountings, never reconciled (ADR-0036). The session figure here is
        // healthy and the venue's is not — which is exactly the case a single bound
        // cannot see: a restart resets our accounting and cannot reset the venue's.
        let l = limiter(dec!(100), dec!(1));
        let b = l
            .observe(Some(dec!(-0.1)), dec!(-0.1), Some(dec!(1.5)))
            .expect("the day bound stands on its own");
        assert_eq!(b.scope, LossScope::Day);
        assert_eq!(b.loss, dec!(1.5));
        assert!(b.to_string().starts_with("day loss 1.5000"), "{b}");
    }

    #[test]
    fn a_day_bound_with_no_venue_reading_yet_judges_nothing() {
        // Before the first `clearinghouseState` reply there is no baseline and no
        // reading. Treating absence as zero loss is right; treating it as a breach would
        // stop every session at startup.
        let l = limiter(Decimal::ZERO, dec!(1));
        assert!(l.observe(Some(dec!(-50)), dec!(-50), None).is_none());
        assert!(!l.is_tripped());
    }

    #[test]
    fn the_switch_is_one_way_because_a_marked_loss_moves_both_ways() {
        // A re-arming limit would let a session resume on a bounce and stop again on the
        // next tick, flickering in and out of the market fastest when volatility is
        // highest and nobody is watching.
        let l = limiter(dec!(0.5), Decimal::ZERO);
        l.observe(Some(dec!(-0.6)), dec!(0), None).unwrap();
        assert!(l.is_tripped());
        // The position recovers completely. The switch does not.
        assert!(l.observe(Some(dec!(5)), dec!(5), None).is_none());
        assert!(l.is_tripped(), "nothing but a restart clears this");
    }

    #[test]
    fn a_negative_bound_is_refused_because_it_would_arm_the_switch_at_startup() {
        // The figure a bound is compared against is itself usually negative, so `-5`
        // written where `5` was meant is a session that never places an order — and the
        // symptom is a strategy that emits and never trades, which reads exactly like a
        // warmup that has not finished.
        assert!(LossLimits {
            session: dec!(-5),
            day: Decimal::ZERO
        }
        .validate()
        .is_err());
        assert!(LossLimits {
            session: Decimal::ZERO,
            day: dec!(-1)
        }
        .validate()
        .is_err());
        assert!(LossLimits {
            session: dec!(5),
            day: Decimal::ZERO
        }
        .validate()
        .is_ok());
        assert!(LossLimits::default().validate().is_ok(), "unset is legal");
    }
}
