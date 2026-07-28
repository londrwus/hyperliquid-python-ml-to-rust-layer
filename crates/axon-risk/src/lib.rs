//! # axon-risk
//!
//! Pre-trade risk checks that every order intent crosses on the hot path
//! (`docs/01-architecture.md`). Kept branch-light and allocation-free: risk must
//! never be the bottleneck. Limits come from the strategy's config and are
//! enforced here, so a strategy cannot bypass them.
//!
//! Two scopes, deliberately separate. [`RiskLimits`] bounds **one instrument** and is
//! what every session has had since Phase 1; [`portfolio`] bounds the **set** of them
//! and only means anything once more than one strategy shares an account. They are not
//! nested: a book can be inside every per-symbol limit and outside every portfolio one.

#![deny(unsafe_code)]

pub mod portfolio;

pub use portfolio::{
    gross_scale, PortfolioEngine, PortfolioExposure, PortfolioLeg, PortfolioLimits, PortfolioReject,
};

use axon_core::{Decimal, Position, Side};
use axon_providers::OrderRequest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Hard limits applied to every intent for one instrument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Max absolute net position (in base units).
    pub max_position: Decimal,
    /// Max absolute notional exposure (position magnitude × mark price).
    pub max_notional: Decimal,
    /// Max size of any single order.
    pub max_order_qty: Decimal,
}

/// Why an intent was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RiskReject {
    #[error("order qty {qty} exceeds max {max}")]
    OrderTooLarge { qty: Decimal, max: Decimal },
    #[error("projected position {projected} exceeds max {max}")]
    PositionLimit { projected: Decimal, max: Decimal },
    #[error("projected notional {notional} exceeds max {max}")]
    NotionalLimit { notional: Decimal, max: Decimal },
    #[error("reduce-only order would grow or flip the position")]
    ReduceOnlyViolation,
}

/// Whether an order of `qty` on `side` strictly takes exposure **off** `position`.
///
/// True when the projected magnitude is smaller than the current one *and* the sign
/// does not flip through zero. Both halves are load-bearing and the second is the one
/// that is easy to miss: long 2 with a sell of 3 lands at short 1, whose magnitude is
/// smaller (2 → 1) while the exposure it opens is on the other side of the market.
/// Reaching exactly flat is a reduction and is the point of the whole thing.
///
/// One definition, two callers — the reduce-only rule in [`RiskEngine::check`] and the
/// de-risk-only mode a tripped loss limit puts the session into
/// (`axon_execution::LossLimiter`). Two copies of this arithmetic would be two answers
/// to "may this order go out while we are trying to get smaller", and the copy that
/// drifted would be the one enforcing the kill switch.
pub fn reduces_exposure(side: Side, qty: Decimal, position: Decimal) -> bool {
    // Nothing reduces a flat position; every order from flat opens exposure.
    if position.is_zero() {
        return false;
    }
    let signed = match side {
        Side::Buy => qty,
        Side::Sell => -qty,
    };
    let projected = position + signed;
    if projected.is_zero() {
        return true;
    }
    projected.is_sign_positive() == position.is_sign_positive() && projected.abs() < position.abs()
}

/// The pre-trade gate.
#[derive(Debug, Clone)]
pub struct RiskEngine {
    limits: RiskLimits,
}

impl RiskEngine {
    pub fn new(limits: RiskLimits) -> Self {
        Self { limits }
    }

    pub fn limits(&self) -> &RiskLimits {
        &self.limits
    }

    /// Check `req` against `position` (current) at `mark_px`. `Ok(())` clears it.
    pub fn check(
        &self,
        req: &OrderRequest,
        position: &Position,
        mark_px: Decimal,
    ) -> Result<(), RiskReject> {
        if req.qty > self.limits.max_order_qty {
            return Err(RiskReject::OrderTooLarge {
                qty: req.qty,
                max: self.limits.max_order_qty,
            });
        }

        let signed = match req.side {
            Side::Buy => req.qty,
            Side::Sell => -req.qty,
        };
        let projected_signed = position.qty + signed;
        let projected = projected_signed.abs();

        // Reduce-only may only shrink toward zero; it de-risks, so it bypasses the
        // size caps but must not grow the position AND must not flip it through
        // zero to the opposite side. A magnitude-only check misses the flip case
        // (e.g. long 2, reduce-only Sell 3 → short 1: magnitude drops 2→1 but the
        // sign flips, opening unintended opposite-side exposure).
        //
        // Note this admits a reduce-only order that changes *nothing* (qty 0 against a
        // held position, or any order against a flat book), which
        // [`reduces_exposure`] deliberately does not: a no-op cannot add exposure, so
        // refusing it here would be a refusal with no risk behind it, whereas the loss
        // switch is asking the stricter question "is this order taking exposure off".
        if req.reduce_only {
            let flips = !position.qty.is_zero()
                && !projected_signed.is_zero()
                && (projected_signed.is_sign_positive() != position.qty.is_sign_positive());
            if projected > position.qty.abs() || flips {
                return Err(RiskReject::ReduceOnlyViolation);
            }
            return Ok(());
        }

        if projected > self.limits.max_position {
            return Err(RiskReject::PositionLimit {
                projected,
                max: self.limits.max_position,
            });
        }
        let notional = projected * mark_px;
        if notional > self.limits.max_notional {
            return Err(RiskReject::NotionalLimit {
                notional,
                max: self.limits.max_notional,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, SymbolId, Tif};
    use axon_providers::OrderRequest;
    use rust_decimal_macros::dec;

    fn engine() -> RiskEngine {
        RiskEngine::new(RiskLimits {
            max_position: dec!(10),
            max_notional: dec!(10000),
            max_order_qty: dec!(5),
        })
    }

    fn buy(qty: Decimal, reduce_only: bool) -> OrderRequest {
        let mut r = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            qty,
            dec!(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        r.reduce_only = reduce_only;
        r
    }

    #[test]
    fn reducing_exposure_is_a_strictly_smaller_magnitude_on_the_same_side_or_flat() {
        // The shared definition behind two gates — the reduce-only rule here and the
        // de-risk-only mode a tripped loss limit imposes. Two copies of this arithmetic
        // would be two answers to "may this order go out while we are trying to get
        // smaller", and the copy that drifted would be the one enforcing the switch.
        let long = dec!(3);
        assert!(reduces_exposure(Side::Sell, dec!(1), long));
        assert!(reduces_exposure(Side::Sell, dec!(3), long), "exactly flat");
        assert!(!reduces_exposure(Side::Buy, dec!(1), long), "grows it");
        assert!(
            !reduces_exposure(Side::Sell, dec!(5), long),
            "long 3 sell 5 lands at short 2: the magnitude falls and the exposure moves \
             to the other side of the market"
        );

        // Symmetric on a short.
        let short = dec!(-3);
        assert!(reduces_exposure(Side::Buy, dec!(1), short));
        assert!(reduces_exposure(Side::Buy, dec!(3), short));
        assert!(!reduces_exposure(Side::Buy, dec!(5), short));
        assert!(!reduces_exposure(Side::Sell, dec!(1), short));

        // Nothing reduces a flat position, and that is the answer that matters most: it
        // is what makes a de-risk gate refuse every order on a session whose own state
        // it cannot read.
        assert!(!reduces_exposure(Side::Buy, dec!(1), Decimal::ZERO));
        assert!(!reduces_exposure(Side::Sell, dec!(1), Decimal::ZERO));
        assert!(
            !reduces_exposure(Side::Sell, Decimal::ZERO, long),
            "a zero-size order takes nothing off"
        );
    }

    #[test]
    fn passes_within_limits() {
        let p = Position::flat(SymbolId::new(1));
        assert!(engine().check(&buy(dec!(3), false), &p, dec!(100)).is_ok());
    }

    #[test]
    fn rejects_oversized_order() {
        let p = Position::flat(SymbolId::new(1));
        let err = engine()
            .check(&buy(dec!(6), false), &p, dec!(100))
            .unwrap_err();
        assert!(matches!(err, RiskReject::OrderTooLarge { .. }));
    }

    #[test]
    fn rejects_position_over_limit() {
        let mut p = Position::flat(SymbolId::new(1));
        p.qty = dec!(8);
        // 8 + 5 = 13 > 10
        let err = engine()
            .check(&buy(dec!(5), false), &p, dec!(100))
            .unwrap_err();
        assert!(matches!(err, RiskReject::PositionLimit { .. }));
    }

    #[test]
    fn rejects_notional_over_limit() {
        let p = Position::flat(SymbolId::new(1));
        // qty 5 within order cap and position cap (5 <= 10), but 5 * 3000 = 15000 > 10000
        let err = engine()
            .check(&buy(dec!(5), false), &p, dec!(3000))
            .unwrap_err();
        assert!(matches!(err, RiskReject::NotionalLimit { .. }));
    }

    #[test]
    fn reduce_only_shrink_is_allowed_over_caps() {
        let mut p = Position::flat(SymbolId::new(1));
        p.qty = dec!(9);
        // selling reduce-only shrinks 9 → 5, fine even though notional caps ignored
        let mut r = buy(dec!(4), true);
        r.side = Side::Sell;
        assert!(engine().check(&r, &p, dec!(100)).is_ok());
    }

    #[test]
    fn reduce_only_that_grows_is_rejected() {
        let mut p = Position::flat(SymbolId::new(1));
        p.qty = dec!(2);
        // buying reduce-only while long grows the position → reject
        let err = engine()
            .check(&buy(dec!(1), true), &p, dec!(100))
            .unwrap_err();
        assert!(matches!(err, RiskReject::ReduceOnlyViolation));
    }

    #[test]
    fn reduce_only_that_flips_through_zero_is_rejected() {
        // Regression: magnitude drops (2 → 1) but the sign flips (long → short),
        // opening opposite-side exposure a reduce-only order must never create.
        let mut p = Position::flat(SymbolId::new(1));
        p.qty = dec!(2);
        let mut r = buy(dec!(3), true); // reduce-only Sell 3
        r.side = Side::Sell;
        let err = engine().check(&r, &p, dec!(100)).unwrap_err();
        assert!(matches!(err, RiskReject::ReduceOnlyViolation));

        // Exact-to-flat (Sell 2) is still allowed — it reaches zero without flipping.
        let mut ok = buy(dec!(2), true);
        ok.side = Side::Sell;
        assert!(engine().check(&ok, &p, dec!(100)).is_ok());
    }
}
