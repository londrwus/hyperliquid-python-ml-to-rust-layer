//! Normalized **execution** vocabulary — what came back from a venue about *our*
//! orders, as opposed to the public market data in [`market`](crate::market).
//!
//! This is the mirror image of the market vocabulary and follows the same rule
//! (ADR-0008): the normalized types live here in `axon-core`, and each provider
//! translates its own wire format into them. Nothing in this module may be shaped by
//! one venue's JSON — a field earns its place only if it is something the core, the
//! risk gate, or a strategy genuinely needs, and any venue could supply.
//!
//! Two facts drive the design:
//!
//! - **A fill and an order-state change are different events.** A venue can report a
//!   partial fill without the order leaving the book, and can cancel an order that
//!   never filled. Collapsing both into one "order status" type loses the fill
//!   quantities that position math needs, which is why [`Fill`] and [`OrderUpdate`]
//!   are separate.
//! - **Reconciliation needs identity and dedup.** A reconnect replays a snapshot, so
//!   the same fill can arrive twice. [`Fill::trade_id`] is the venue's own execution
//!   identifier, carried through purely so the tracker can drop duplicates. Without
//!   it, one dropped WebSocket connection double-counts a position.

use serde::{Deserialize, Serialize};

use crate::clock::Nanos;
use crate::enums::Side;
use crate::ids::{Cloid, OrderId, SymbolId};
use rust_decimal::Decimal;

/// Which side of the book an execution took liquidity from.
///
/// Kept because it is the difference between paying a fee and earning a rebate, and
/// because maker/taker ratio is the headline diagnostic for whether a post-only
/// strategy is actually posting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Liquidity {
    /// We were resting; someone else crossed to us.
    Maker,
    /// We crossed the spread.
    Taker,
}

/// The lifecycle state of one of our orders, as the venue reports it.
///
/// Lives here rather than in `axon-providers` so `axon-core` can own the whole
/// execution vocabulary; `axon-providers` re-exports it, exactly as it re-exports
/// the market types.
///
/// `Accepted` is *our* state, not the venue's: it marks the window between a
/// successful submit and the venue's first word about the order. Distinguishing it
/// from `Resting` is what lets the tracker notice an order that was acknowledged but
/// never confirmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    /// Submitted and acknowledged locally; no venue state yet.
    Accepted,
    /// Live on the book.
    Resting,
    /// Partially executed and still live.
    PartiallyFilled,
    /// Fully executed. Terminal.
    Filled,
    /// Removed without completing — by us, by the venue, or by a dead-man's switch.
    /// Terminal.
    Cancelled,
    /// Refused. Terminal.
    Rejected,
}

impl OrderStatus {
    /// Whether no further updates can arrive for this order.
    ///
    /// The tracker keys its "forget this order" decision on exactly this, so the
    /// definition lives with the type rather than being re-derived at each call site.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected
        )
    }
}

/// Why an order left the book. Venues distinguish these and the difference matters
/// operationally: a self-cancel is routine, a liquidation is an incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    /// We asked for it.
    Requested,
    /// A dead-man's switch / scheduled cancel fired.
    DeadMansSwitch,
    /// The venue pulled it — insufficient margin, self-trade prevention, expiry.
    Venue,
    /// The venue pulled it as part of reducing or liquidating the account.
    Liquidation,
    /// The venue did not say.
    Unspecified,
}

/// One execution against one of our orders.
///
/// `qty` is a positive magnitude and `side` carries the direction, matching
/// [`Position::apply_fill`](crate::position::Position::apply_fill) so a fill can be
/// applied without sign juggling at the call site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fill {
    pub symbol_id: SymbolId,
    /// The venue's order id. Always present — a fill implies the venue knew the order.
    pub order_id: OrderId,
    /// Our client id, when the venue echoes it back. Absent for orders we did not
    /// place through this process (a manual order, or one from before a restart).
    pub cloid: Option<Cloid>,
    pub side: Side,
    /// Executed quantity, positive magnitude.
    pub qty: Decimal,
    pub price: Decimal,
    /// Fee paid, signed: positive is a cost, negative is a maker rebate.
    pub fee: Decimal,
    /// Realized PnL the venue attributes to this execution, when it reports one.
    /// We keep our own accounting in [`Position`](crate::position::Position); this is
    /// the venue's number, retained for reconciliation against ours.
    pub closed_pnl: Decimal,
    pub liquidity: Liquidity,
    /// The venue's execution identifier — the dedup key across reconnects.
    pub trade_id: u64,
    pub ts_event: Nanos,
}

/// A change in an order's resting state at the venue.
///
/// Quantities are absolute, not deltas: a venue snapshot after a reconnect gives
/// absolute numbers, and mixing the two representations is how position drift starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderUpdate {
    pub symbol_id: SymbolId,
    pub order_id: OrderId,
    pub cloid: Option<Cloid>,
    pub side: Side,
    pub status: OrderStatus,
    /// Limit price. `None` for an order the venue reports without one.
    pub price: Option<Decimal>,
    /// Size the order was submitted with.
    pub orig_qty: Decimal,
    /// Size not (yet) executed.
    ///
    /// For a live order this is what is still working on the book. For a **terminal**
    /// one it is the part that never filled — *not* zero. That distinction matters:
    /// zeroing it on a cancel would make [`filled_qty`](Self::filled_qty) claim the
    /// whole order executed, turning a partially-filled cancel into a phantom position.
    /// Only a fully-`Filled` order has zero remaining.
    pub remaining_qty: Decimal,
    /// Set when `status` is [`OrderStatus::Cancelled`], otherwise `None`.
    pub cancel_reason: Option<CancelReason>,
    pub ts_event: Nanos,
}

impl OrderUpdate {
    /// Quantity executed so far, derived rather than stored so it cannot disagree
    /// with `orig_qty`/`remaining_qty`.
    pub fn filled_qty(&self) -> Decimal {
        self.orig_qty - self.remaining_qty
    }
}

/// The venue's view of our account, as of `ts_event`.
///
/// A periodic absolute snapshot is the only way to detect drift between what we think
/// we hold and what the venue thinks we hold — incremental fills alone can never
/// reveal a fill we never received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    /// Total account value in quote currency.
    pub equity: Decimal,
    /// Value that could be withdrawn — equity minus margin in use.
    pub withdrawable: Decimal,
    /// Margin currently committed to open positions.
    pub margin_used: Decimal,
    pub ts_event: Nanos,
}

/// Normalized execution events — the counterpart of
/// [`MarketEvent`](crate::market::MarketEvent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecEvent {
    Fill(Fill),
    Order(OrderUpdate),
    Account(AccountSnapshot),
}

impl ExecEvent {
    pub fn ts_event(&self) -> Nanos {
        match self {
            ExecEvent::Fill(f) => f.ts_event,
            ExecEvent::Order(o) => o.ts_event,
            ExecEvent::Account(a) => a.ts_event,
        }
    }

    /// The instrument this event concerns. `None` for account-level events, which are
    /// not scoped to one symbol.
    pub fn symbol_id(&self) -> Option<SymbolId> {
        match self {
            ExecEvent::Fill(f) => Some(f.symbol_id),
            ExecEvent::Order(o) => Some(o.symbol_id),
            ExecEvent::Account(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use rust_decimal_macros::dec;

    const SYM: SymbolId = SymbolId::new(7);

    fn fill(side: Side, qty: Decimal, price: Decimal, trade_id: u64) -> Fill {
        Fill {
            symbol_id: SYM,
            order_id: OrderId::new(1),
            cloid: Some(Cloid::new(1)),
            side,
            qty,
            price,
            fee: dec!(0.01),
            closed_pnl: Decimal::ZERO,
            liquidity: Liquidity::Taker,
            trade_id,
            ts_event: 1_000,
        }
    }

    #[test]
    fn terminal_states_are_exactly_the_three_that_end_an_order() {
        for s in [
            OrderStatus::Filled,
            OrderStatus::Cancelled,
            OrderStatus::Rejected,
        ] {
            assert!(s.is_terminal(), "{s:?} should be terminal");
        }
        for s in [
            OrderStatus::Accepted,
            OrderStatus::Resting,
            OrderStatus::PartiallyFilled,
        ] {
            assert!(!s.is_terminal(), "{s:?} should not be terminal");
        }
    }

    #[test]
    fn a_fill_applies_to_a_position_without_sign_juggling() {
        // The point of `qty` being a positive magnitude with `side` alongside: the
        // fill can go straight into position math.
        let mut p = Position::flat(SYM);
        let f = fill(Side::Buy, dec!(2), dec!(100), 1);
        p.apply_fill(f.side, f.qty, f.price);
        assert_eq!(p.qty, dec!(2));
        assert_eq!(p.avg_px, dec!(100));

        let f = fill(Side::Sell, dec!(3), dec!(110), 2);
        p.apply_fill(f.side, f.qty, f.price);
        assert_eq!(p.qty, dec!(-1)); // flipped through zero
        assert_eq!(p.realized_pnl, dec!(20)); // closed 2 @ +10
    }

    #[test]
    fn a_partially_filled_cancel_keeps_its_unfilled_remainder() {
        // The tempting invariant "terminal implies remaining_qty == 0" is wrong: it
        // would make a cancel of a half-filled order look like a complete fill, which
        // is a phantom position.
        let u = OrderUpdate {
            symbol_id: SYM,
            order_id: OrderId::new(1),
            cloid: None,
            side: Side::Buy,
            status: OrderStatus::Cancelled,
            price: Some(dec!(100)),
            orig_qty: dec!(10),
            remaining_qty: dec!(6),
            cancel_reason: Some(CancelReason::Requested),
            ts_event: 1,
        };
        assert!(u.status.is_terminal());
        assert_eq!(
            u.filled_qty(),
            dec!(4),
            "4 of 10 executed before the cancel"
        );
    }

    #[test]
    fn filled_qty_is_derived_so_it_cannot_disagree() {
        let u = OrderUpdate {
            symbol_id: SYM,
            order_id: OrderId::new(1),
            cloid: None,
            side: Side::Buy,
            status: OrderStatus::PartiallyFilled,
            price: Some(dec!(100)),
            orig_qty: dec!(5),
            remaining_qty: dec!(2),
            cancel_reason: None,
            ts_event: 1,
        };
        assert_eq!(u.filled_qty(), dec!(3));
    }

    #[test]
    fn exec_event_exposes_time_and_optional_symbol() {
        let f = ExecEvent::Fill(fill(Side::Buy, dec!(1), dec!(100), 1));
        assert_eq!(f.ts_event(), 1_000);
        assert_eq!(f.symbol_id(), Some(SYM));

        let a = ExecEvent::Account(AccountSnapshot {
            equity: dec!(1000),
            withdrawable: dec!(900),
            margin_used: dec!(100),
            ts_event: 2_000,
        });
        assert_eq!(a.ts_event(), 2_000);
        assert_eq!(a.symbol_id(), None, "account events are not per-symbol");
    }

    #[test]
    fn status_serde_uses_snake_case_on_the_wire() {
        // The Python side and any persisted golden file depend on these strings.
        let j = serde_json::to_string(&OrderStatus::PartiallyFilled).unwrap();
        assert_eq!(j, "\"partially_filled\"");
        let r: OrderStatus = serde_json::from_str("\"resting\"").unwrap();
        assert_eq!(r, OrderStatus::Resting);
    }
}
