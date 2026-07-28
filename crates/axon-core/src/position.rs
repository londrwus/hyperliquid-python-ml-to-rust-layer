//! Signed-position accounting. A single [`Position`] tracks net quantity (long
//! positive, short negative), the average entry price, and realized PnL, updating
//! correctly across opens, partial reductions, and flips through zero.

use crate::enums::Side;
use crate::ids::SymbolId;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Net position in one instrument.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub symbol_id: SymbolId,
    /// Signed quantity: `> 0` long, `< 0` short, `0` flat.
    pub qty: Decimal,
    /// Volume-weighted average entry price of the open quantity (`0` when flat).
    pub avg_px: Decimal,
    /// Cumulative realized PnL from closed quantity.
    pub realized_pnl: Decimal,
}

impl Position {
    /// A flat position for `symbol_id`.
    pub fn flat(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            qty: Decimal::ZERO,
            avg_px: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
        }
    }

    #[inline]
    pub fn is_flat(&self) -> bool {
        self.qty.is_zero()
    }

    /// Unrealized PnL marked at `mark_px`.
    pub fn unrealized_pnl(&self, mark_px: Decimal) -> Decimal {
        (mark_px - self.avg_px) * self.qty
    }

    /// Apply a fill of `qty` (a positive magnitude) at `px` on `side`, updating
    /// net quantity, average price, and realized PnL.
    ///
    /// Handles three cases: (1) opening or increasing in the same direction —
    /// re-average; (2) reducing without crossing zero — book realized PnL, keep
    /// the average; (3) flipping through zero — realize the closed part, then
    /// re-open the remainder at `px`.
    pub fn apply_fill(&mut self, side: Side, qty: Decimal, px: Decimal) {
        debug_assert!(qty > Decimal::ZERO, "fill qty must be a positive magnitude");
        let signed = match side {
            Side::Buy => qty,
            Side::Sell => -qty,
        };
        let before = self.qty;

        let same_direction =
            before.is_zero() || before.is_sign_positive() == signed.is_sign_positive();

        if same_direction {
            // Opening or adding: re-average over the combined magnitude.
            let new_qty = before + signed;
            let notional = self.avg_px * before.abs() + px * qty;
            self.avg_px = if new_qty.is_zero() {
                Decimal::ZERO
            } else {
                notional / new_qty.abs()
            };
            self.qty = new_qty;
            return;
        }

        // Opposite direction → we are closing some or all of the position.
        let closing = qty.min(before.abs());
        let dir = if before.is_sign_positive() {
            Decimal::ONE
        } else {
            -Decimal::ONE
        };
        // Long closed by a sell earns (px - avg); short closed by a buy earns (avg - px).
        self.realized_pnl += dir * (px - self.avg_px) * closing;

        let new_qty = before + signed;
        if new_qty.is_zero() {
            self.qty = Decimal::ZERO;
            self.avg_px = Decimal::ZERO;
        } else if new_qty.is_sign_positive() == before.is_sign_positive() {
            // Still same side, just smaller — average unchanged.
            self.qty = new_qty;
        } else {
            // Flipped through zero — the remainder opens fresh at this fill price.
            self.qty = new_qty;
            self.avg_px = px;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn pos() -> Position {
        Position::flat(SymbolId::new(1))
    }

    #[test]
    fn open_long_sets_average() {
        let mut p = pos();
        p.apply_fill(Side::Buy, dec!(2), dec!(100));
        assert_eq!(p.qty, dec!(2));
        assert_eq!(p.avg_px, dec!(100));
        assert_eq!(p.realized_pnl, dec!(0));
    }

    #[test]
    fn add_to_long_reaverages() {
        let mut p = pos();
        p.apply_fill(Side::Buy, dec!(2), dec!(100));
        p.apply_fill(Side::Buy, dec!(2), dec!(120));
        assert_eq!(p.qty, dec!(4));
        assert_eq!(p.avg_px, dec!(110)); // (100*2 + 120*2)/4
    }

    #[test]
    fn partial_close_books_realized_pnl() {
        let mut p = pos();
        p.apply_fill(Side::Buy, dec!(4), dec!(100));
        p.apply_fill(Side::Sell, dec!(1), dec!(130));
        assert_eq!(p.qty, dec!(3));
        assert_eq!(p.avg_px, dec!(100)); // average unchanged on reduction
        assert_eq!(p.realized_pnl, dec!(30)); // (130-100)*1
    }

    #[test]
    fn short_then_cover_pnl() {
        let mut p = pos();
        p.apply_fill(Side::Sell, dec!(2), dec!(100));
        assert_eq!(p.qty, dec!(-2));
        p.apply_fill(Side::Buy, dec!(2), dec!(90));
        assert!(p.is_flat());
        assert_eq!(p.realized_pnl, dec!(20)); // (100-90)*2
        assert_eq!(p.avg_px, dec!(0));
    }

    #[test]
    fn flip_through_zero_reopens_at_fill_price() {
        let mut p = pos();
        p.apply_fill(Side::Buy, dec!(2), dec!(100)); // long 2 @ 100
        p.apply_fill(Side::Sell, dec!(5), dec!(110)); // sell 5 → short 3
        assert_eq!(p.qty, dec!(-3));
        assert_eq!(p.avg_px, dec!(110)); // remainder opens at 110
        assert_eq!(p.realized_pnl, dec!(20)); // closed 2 @ (110-100)
    }

    #[test]
    fn unrealized_tracks_mark() {
        let mut p = pos();
        p.apply_fill(Side::Buy, dec!(3), dec!(100));
        assert_eq!(p.unrealized_pnl(dec!(105)), dec!(15)); // (105-100)*3
    }
}
