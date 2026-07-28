//! Exact conversion between the boundary's fixed-point wire integers and the
//! core's [`Decimal`].
//!
//! `contracts/schema.toml` puts prices and quantities on the wire as signed
//! integers in units of `10^-FIXED_POINT_DECIMALS`. [`axon_contracts::from_fixed`]
//! exists for tooling and goes through `f64`, which is fine for printing and wrong
//! for trading: `0.1` has no exact binary representation, so an `f64` round trip
//! turns a target of `0.1` into `0.100000000000000005…` and every downstream
//! comparison against the position (`target == current`) starts failing for
//! reasons nobody can see. These conversions never touch a float.

use axon_contracts::{FIXED_POINT_DECIMALS, FIXED_POINT_SCALE};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;

// `Decimal::new` panics above 28 decimal places. The scale comes from the schema,
// so pin the assumption here rather than discovering it as a runtime panic on the
// order path.
const _: () = assert!(
    FIXED_POINT_DECIMALS <= 28,
    "contracts/schema.toml fixed_point.decimals exceeds Decimal's scale range"
);

/// Wire integer → [`Decimal`], exactly.
///
/// `Decimal` stores a 96-bit mantissa plus a base-10 scale, so this is a
/// reinterpretation rather than a division: every `i64` maps to a distinct,
/// exactly-representable value.
#[inline]
pub fn fixed_to_decimal(wire: i64) -> Decimal {
    Decimal::new(wire, FIXED_POINT_DECIMALS)
}

/// [`Decimal`] → wire integer, or `None` when the value cannot be represented
/// exactly at the contract's scale (more than 8 decimal places, or out of `i64`).
///
/// Returning `None` rather than rounding is deliberate: a silent round here would
/// mean Python and Rust disagree about the target position by a hair, which
/// resolves into a permanent dust order that never reaches the target.
#[inline]
pub fn decimal_to_fixed(value: Decimal) -> Option<i64> {
    let scaled = value.checked_mul(Decimal::from(FIXED_POINT_SCALE))?;
    if !scaled.fract().is_zero() {
        return None;
    }
    scaled.to_i64()
}

/// Multiply a wire quantity by a factor at or below one, landing back on the contract's
/// scale by truncating **toward zero**.
///
/// This is the one place in the project where a quantity is deliberately rounded rather
/// than refused, and both halves of that need saying.
///
/// **Why rounding at all.** [`decimal_to_fixed`] refuses a value finer than the wire's
/// eight decimals, and it is right to: a target that quietly loses a hair leaves a
/// permanent dust order chasing a position it never reaches. But a portfolio allocation
/// is a *ratio* — `max_gross / gross` is 0.6382978… as often as not — so multiplying an
/// exact target by one produces a value the wire cannot carry every time it binds.
/// Refusing there would mean a declared portfolio bound silently stopped a session from
/// trading, which is the failure mode this project keeps naming.
///
/// **Why toward zero, specifically.** Away from zero would let a scaled target come out
/// *larger* than its allocation by one wire unit — the bound being applied would then be
/// the one thing the arithmetic could exceed. Toward zero can only ever under-fill, and
/// an under-filled allocation is a smaller position than the operator permitted, which is
/// the direction with no failure in it. Half-up would be right on average and wrong in
/// the direction that matters, which is not a trade this file makes.
///
/// A factor at or above one returns the input **unchanged and untouched**, so a session
/// with nothing to scale gets exactly the bytes its producer wrote — the property the
/// single-producer path depends on.
pub fn scale_fixed(wire: i64, factor: Decimal) -> i64 {
    if factor >= Decimal::ONE {
        return wire;
    }
    if factor <= Decimal::ZERO {
        return 0;
    }
    let scaled = fixed_to_decimal(wire) * factor;
    let truncated =
        scaled.round_dp_with_strategy(FIXED_POINT_DECIMALS, rust_decimal::RoundingStrategy::ToZero);
    // The truncation put it on the contract's scale by construction, so the `None` arm
    // is only reachable for a magnitude past `i64` — which a factor below one cannot
    // create from a value that was already representable. Zero is the safe answer to a
    // number we cannot express: it places no order.
    decimal_to_fixed(truncated).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn a_factor_of_one_or_more_returns_the_producers_own_bytes() {
        // The property the single-producer path depends on: an unscaled session must put
        // exactly what its strategy wrote on the wire, not a value that happens to be
        // numerically equal after a multiply and a round.
        for wire in [0i64, 1, -1, 30_000, -70_000_001, i64::MAX] {
            assert_eq!(scale_fixed(wire, Decimal::ONE), wire);
            assert_eq!(scale_fixed(wire, dec!(2)), wire, "never scaled up");
        }
    }

    #[test]
    fn scaling_truncates_toward_zero_so_an_allocation_is_never_exceeded() {
        // Away from zero would let a scaled target come out larger than the allocation
        // being applied — the bound would be the one thing the arithmetic could exceed.
        // 3 * 0.3333... at eight decimals is the case that shows the direction.
        assert_eq!(scale_fixed(30_000_001, dec!(0.5)), 15_000_000);
        assert_eq!(scale_fixed(-30_000_001, dec!(0.5)), -15_000_000);
        // A ratio with no exact decimal expansion, both signs, each landing short.
        let third = Decimal::ONE / Decimal::from(3);
        assert_eq!(scale_fixed(100, third), 33);
        assert_eq!(scale_fixed(-100, third), -33);
    }

    #[test]
    fn a_scaled_target_that_rounds_to_nothing_is_zero_rather_than_a_dust_order() {
        // The floor case: one wire unit scaled by a half is half a unit, which the
        // contract cannot express. Zero places no order; anything else would be a
        // quantity the encoder refuses.
        assert_eq!(scale_fixed(1, dec!(0.5)), 0);
        assert_eq!(scale_fixed(-1, dec!(0.5)), 0);
        assert_eq!(scale_fixed(12_345, Decimal::ZERO), 0);
        assert_eq!(
            scale_fixed(12_345, dec!(-1)),
            0,
            "a negative factor is not a flip"
        );
    }

    #[test]
    fn a_fixed_point_value_round_trips_exactly() {
        for wire in [
            0i64,
            1,
            -1,
            100_000_000,
            -125_000_000,
            999_999_999_999_999,
            -999_999_999_999_999,
        ] {
            let d = fixed_to_decimal(wire);
            assert_eq!(
                decimal_to_fixed(d),
                Some(wire),
                "wire {wire} must survive the round trip unchanged"
            );
        }
    }

    #[test]
    fn a_value_f64_cannot_hold_converts_exactly() {
        // 0.1 is the canonical float trap: `0.1_f64 * 1e8` is 10000000.000000002.
        assert_eq!(fixed_to_decimal(10_000_000), dec!(0.1));
        assert_eq!(fixed_to_decimal(1), dec!(0.00000001));
        assert_eq!(fixed_to_decimal(-70_000_001), dec!(-0.70000001));
        // And the exactness holds under arithmetic, which is the property that
        // makes `target == current` a trustworthy test downstream.
        let three_tenths = fixed_to_decimal(10_000_000) * Decimal::from(3);
        assert_eq!(three_tenths, dec!(0.3));
    }

    #[test]
    fn a_value_finer_than_the_contract_scale_is_refused_not_rounded() {
        // 9 decimal places; the wire only carries 8.
        assert_eq!(decimal_to_fixed(dec!(0.000000005)), None);
        assert_eq!(decimal_to_fixed(dec!(1.000000001)), None);
        // Trailing zeros beyond the scale are still exact.
        assert_eq!(decimal_to_fixed(dec!(1.5000000000)), Some(150_000_000));
    }

    #[test]
    fn a_value_beyond_i64_is_refused_rather_than_wrapping() {
        assert_eq!(decimal_to_fixed(dec!(1000000000000)), None);
    }
}
