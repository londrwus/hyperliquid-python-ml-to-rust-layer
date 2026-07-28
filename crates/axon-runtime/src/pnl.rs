//! What a live session has done to the account, as two independent answers.
//!
//! Every other number a session prints is about *machinery* — records accepted,
//! orders placed, quotes swept. This module is the only one about **money**, and the
//! roadmap's Phase 6 asked for it in the same breath as parity and latency because a
//! session that trades and is not watched is not evidence of anything.
//!
//! The design has one idea in it, and everything else follows from it:
//!
//! > **Our accounting and the venue's are reported side by side and never reconciled
//! > into one figure.**
//!
//! We compute realized P&L from our own average-cost book ([`Position::realized_pnl`]),
//! fees from the fills we applied, and unrealized from the mark cache. The venue
//! reports an `accountValue`. Subtracting the session's first `accountValue` from the
//! latest gives a completely independent number for the same period. Merging them
//! would destroy the only cross-check a live session has — and they are *expected* to
//! differ, for reasons that are features of the venue rather than bugs here:
//!
//! - **Funding.** A perp pays or receives funding every hour. It moves `accountValue`
//!   and it is not a fill, so no fill-derived accounting can see it.
//! - **Anything else on the account.** The testnet key this project uses is shared —
//!   22 orphan fills were on the status line for the whole of Phase 6 — and every
//!   position another actor opens moves the same `accountValue`.
//! - **The venue's own mark.** Its unrealized uses its oracle/mark price; ours uses
//!   [`MarkCache`], which prefers a venue `Ticker` but falls back to a book mid.
//!
//! So [`PnlSnapshot::drift`] is a **reported quantity, never a correction**. It is the
//! same shape as [`crate::reconcile::PositionDrift`] and it is there for the same
//! reason: two views of one fact, and a number that says how far apart they are.
//!
//! ## Both sides are measured from one instant, and the venue is why
//!
//! A process does **not** start with an empty fill history. Hyperliquid's `userFills`
//! replays a snapshot of recent fills on every subscribe, and the tracker applies them
//! — that is what lets a restarted process know its own position. The first live run of
//! this module opened, with no order placed, at `r +0.0161 fee 0.1031` over 22 replayed
//! fills, and a drift of `-0.0869` against an equity that had not moved.
//!
//! Neither number was wrong; the *subtraction* was. `accountValue` already contains
//! those trades, so comparing a lifetime-ish P&L against a session-scoped equity delta
//! carries a permanent offset — and a cross-check with a constant offset in it cannot
//! detect anything new, which is the only thing it is for. The split is made in the
//! tracker, on each fill's **own execution time** rather than on when it arrived
//! (`axon_execution::Money`): the first attempt baselined at the first
//! `clearinghouseState` reply and the `userFills` snapshot landed *after* it, so every
//! replayed fill still counted as the session's.
//!
//! ## What this module refuses to do
//!
//! **It will not price an unrealized P&L off a stale mark.** [`MarkCache::get`]
//! returns `None` once a price is past the session's `mark_max_age_ms`, and a symbol
//! with an open position and no fresh price makes [`PnlSnapshot::unrealized`] `None`
//! for the whole session rather than contributing zero. Contributing zero is the
//! tempting shape and it is the dangerous one: an instrument whose feed died stops
//! moving the P&L, so a position going wrong reads as a position going nowhere. The
//! unpriced symbols are *named* on the status line, exactly as stale marks are.
//!
//! **It will not halt.** Monitoring and risk are separate decisions with separate
//! evidence, and a loss limit that pulls the trading switch is a risk control that
//! belongs with the others in `axon-execution`, argued for on its own terms. What a
//! configured [`PnlConfig::max_session_loss`] buys here is a **warning**, ranked with
//! the rest on the status line.

use std::fmt;

use axon_core::{Decimal, SymbolId};
use axon_execution::{MarkCache, OrderTracker};

/// The money view of a session, assembled once per status line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PnlSnapshot {
    /// Realized P&L from **our** average-cost accounting, summed over the configured
    /// universe and **since the session's baseline** (see the module docs). Fees are
    /// *not* in it — [`Position::apply_fill`] takes price and quantity and never sees
    /// the fee — which is why the next field exists.
    ///
    /// [`Position::apply_fill`]: axon_core::Position::apply_fill
    pub realized: Decimal,
    /// The same figure with **nothing subtracted**: every fill the tracker has applied,
    /// including whatever the venue replayed at subscribe time.
    ///
    /// Kept beside the session figure rather than instead of it, for the same reason
    /// the venue's `closed_pnl` is kept beside ours: two numbers, one of which answers
    /// "what did this session do" and the other "what does this process believe about
    /// the account". Collapsing them is how a session reports somebody else's trades as
    /// its own result.
    pub realized_all: Decimal,
    /// Fee since the baseline, signed as the venue signs it: positive is a cost,
    /// negative is a maker rebate.
    pub fees: Decimal,
    /// Fee over every fill the tracker applied, baseline or not.
    pub fees_all: Decimal,
    /// The venue's own realized-P&L attribution over the same fills. Reported beside
    /// `realized` rather than instead of it: they are computed from different
    /// accounting and the difference is the finding.
    pub venue_closed_pnl: Decimal,
    /// Mark-to-market on open positions, or `None` when any symbol we hold has no
    /// price fresh enough to vouch for. See the module docs for why this is not zero.
    pub unrealized: Option<Decimal>,
    /// Symbols with an open position and no fresh mark — named, because "1 unpriced"
    /// does not tell an operator which position is unaccounted for.
    pub unpriced: Vec<String>,
    /// Absolute notional at risk, over the symbols that *could* be priced. Present
    /// even when `unrealized` is `None`, because "how much is on the table" is
    /// answerable from a partial view and "what is it worth" is not.
    pub gross_exposure: Decimal,
    /// Fills **since the baseline**, split by which side of the book they took.
    pub fills: u64,
    pub maker_fills: u64,
    pub taker_fills: u64,
    /// Fills the tracker has applied in total. A large gap between this and `fills` is
    /// the venue's replay, and it is the number that explains an `ORPHAN FILLS` count.
    pub fills_all: u64,
    /// The venue's latest `accountValue`, and this session's first.
    pub equity: Option<Decimal>,
    pub equity_at_start: Option<Decimal>,
    /// `equity - equity_at_start`: the venue's own answer for this session, arrived at
    /// without any of our accounting.
    pub equity_delta: Option<Decimal>,
    /// `net()` minus `equity_delta` — how far the two answers are apart. `None` until
    /// both exist. **Never** used to adjust either side.
    pub drift: Option<Decimal>,
    /// Whether the tracker could be read at all when this was assembled.
    ///
    /// `false` is a *poisoned lock*: a panic left our own order state unreadable, so
    /// every figure above is a default rather than a measurement. It gets its own flag
    /// because the alternative — assembling from an empty tracker — produces
    /// `pnl +0.0000`, which is both the most believable and the most wrong thing this
    /// module can print. A session holding a position and unable to see it must not
    /// report that it has done nothing.
    pub readable: bool,
}

impl PnlSnapshot {
    /// Our own bottom line: realized, less fees, plus mark-to-market.
    ///
    /// `None` whenever `unrealized` is, which is the whole point — a bottom line that
    /// silently omits an unpriced position is a bottom line that is wrong by exactly
    /// the amount nobody can see.
    pub fn net(&self) -> Option<Decimal> {
        if !self.readable {
            return None;
        }
        self.unrealized.map(|u| self.realized - self.fees + u)
    }

    /// The view a session assembles when its own order state cannot be read.
    ///
    /// Every number absent rather than zero. See [`Self::readable`].
    pub fn unreadable() -> Self {
        Self {
            realized: Decimal::ZERO,
            realized_all: Decimal::ZERO,
            fees: Decimal::ZERO,
            fees_all: Decimal::ZERO,
            venue_closed_pnl: Decimal::ZERO,
            unrealized: None,
            unpriced: Vec::new(),
            gross_exposure: Decimal::ZERO,
            fills: 0,
            maker_fills: 0,
            taker_fills: 0,
            fills_all: 0,
            equity: None,
            equity_at_start: None,
            equity_delta: None,
            drift: None,
            readable: false,
        }
    }

    /// The part of the answer that does not depend on a mark at all.
    ///
    /// Worth having separately because it is the number that cannot be wrong: closed
    /// trades and the fees paid on them. A session whose `net()` is unknown still
    /// knows this.
    pub fn realized_net(&self) -> Decimal {
        self.realized - self.fees
    }
}

/// Assemble the money view.
///
/// Pure: a tracker snapshot, a mark cache and the configured universe in, a value
/// out. The status line's rendering and the warnings are then testable with no
/// session, no venue and no clock — which is what lets the awkward cases (an unpriced
/// position, a venue that has never answered) be asserted rather than hoped for.
///
/// `symbols` is the **configured** universe rather than the positions the tracker
/// happens to hold, for the same reason [`crate::reconcile::diff`] uses it: a symbol
/// we are flat in is a symbol whose P&L is knowably zero, and one the venue reports
/// that we never configured is not this session's to account for.
pub fn snapshot(
    tracker: &OrderTracker,
    marks: &MarkCache,
    symbols: &[(SymbolId, String)],
) -> PnlSnapshot {
    let mut unrealized = Some(Decimal::ZERO);
    let mut gross_exposure = Decimal::ZERO;
    let mut unpriced = Vec::new();

    // `None` until the venue has answered once, and zero is the right baseline then:
    // before the first `accountValue` there is no equity delta to compare against
    // either, so `drift` is `None` and the two halves stay consistent.
    // Over **every** symbol the tracker knows, not just the configured universe, and
    // the reason is the one that also puts an orphan fill's fee in the total: the two
    // halves of one fill must not be accounted on opposite sides of a filter, or a
    // session reports a position it did not ask for and excludes what that position
    // cost. The mark-to-market below *is* per-universe, because a price is something a
    // session either subscribed to or does not have.
    let session = tracker.session_money();
    let total = tracker.total_money();

    for (id, name) in symbols {
        let p = tracker.position(*id);
        if p.is_flat() {
            continue;
        }
        // `get`, not `last_known`: the second one is what the reconnect path uses to
        // avoid throwing away a price it may still want, and using it here would let a
        // dead feed's last quote go on valuing a live position indefinitely.
        match marks.get(*id) {
            Some(px) => {
                gross_exposure += (p.qty * px).abs();
                if let Some(u) = unrealized.as_mut() {
                    *u += p.qty * (px - p.avg_px);
                }
            }
            None => {
                unpriced.push(name.clone());
                unrealized = None;
            }
        }
    }

    let equity = tracker.last_snapshot().map(|d| d.venue_equity);
    let equity_at_start = tracker.first_snapshot().map(|d| d.venue_equity);
    let equity_delta = equity.zip(equity_at_start).map(|(now, then)| now - then);

    let mut snap = PnlSnapshot {
        realized: session.realized,
        realized_all: total.realized,
        fees: session.fees,
        fees_all: total.fees,
        venue_closed_pnl: session.venue_closed_pnl,
        unrealized,
        unpriced,
        gross_exposure,
        fills: session.fills(),
        maker_fills: session.maker_fills,
        taker_fills: session.taker_fills,
        fills_all: total.fills(),
        equity,
        equity_at_start,
        equity_delta,
        drift: None,
        readable: true,
    };
    snap.drift = snap
        .net()
        .zip(equity_delta)
        .map(|(ours, theirs)| ours - theirs);
    snap
}

/// How the money view prints on the status line.
///
/// Compact on purpose: an operator reads this beside a dozen other blocks under
/// pressure, and every field here has to earn its characters. `net` first because it
/// is the question; the decomposition after it because a net figure with no story is
/// one nobody can act on.
///
/// A `net` of `-` is not an omission — it is the module's refusal to price an
/// unpriced position, and the `UNPRICED` warning names the symbol.
impl fmt::Display for PnlSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.readable {
            // Uppercase and alone: there is nothing else on this block that would not
            // be a fabrication.
            return write!(f, "pnl UNREADABLE");
        }
        write!(f, "pnl ")?;
        match self.net() {
            Some(n) => write!(f, "{n:+.4}")?,
            None => write!(f, "-")?,
        }
        write!(f, " (r {:+.4} fee {:.4}", self.realized, self.fees)?;
        match self.unrealized {
            Some(u) => write!(f, " u {u:+.4})")?,
            None => write!(f, " u -)")?,
        }
        // The fill split, only once there is one. A permanent `0m/0t` on every
        // read-only session would train an operator to skip the whole block.
        if self.fills > 0 {
            write!(f, " {}m/{}t", self.maker_fills, self.taker_fills)?;
        }
        // The venue's replay, named where it cannot be mistaken for this session's
        // work. Without it a status line showing `pnl +0.0000` beside `ORPHAN FILLS 22`
        // gives an operator no way to tell "we have done nothing" from "we are not
        // counting anything".
        if self.fills_all > self.fills {
            write!(f, " (+{} pre)", self.fills_all - self.fills)?;
        }
        if let Some(eq) = self.equity {
            write!(f, " | eq {eq:.4}")?;
            if let Some(d) = self.equity_delta {
                write!(f, " {d:+.4}")?;
            }
            // Only when it can be computed, and always when it can: a drift that is
            // printed only past a threshold is a drift nobody watches converge.
            if let Some(d) = self.drift {
                write!(f, " drift {d:+.4}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{
        AccountSnapshot, Event, EventHandler, ExecEvent, Fill, Liquidity, Nanos, OrderId, Side,
    };
    use rust_decimal_macros::dec;

    const BTC: SymbolId = SymbolId::new(0);
    const ETH: SymbolId = SymbolId::new(1);
    const SEC: Nanos = 1_000_000_000;

    fn universe() -> Vec<(SymbolId, String)> {
        vec![(BTC, "BTC".into()), (ETH, "ETH".into())]
    }

    fn fill(sym: SymbolId, side: Side, qty: Decimal, px: Decimal, fee: Decimal, id: u64) -> Fill {
        Fill {
            symbol_id: sym,
            order_id: OrderId::new(id),
            cloid: None,
            side,
            qty,
            price: px,
            fee,
            closed_pnl: Decimal::ZERO,
            liquidity: Liquidity::Maker,
            trade_id: id,
            ts_event: SEC,
        }
    }

    /// Stamp a fill as having executed before the session started.
    fn old(mut f: Fill) -> Fill {
        f.ts_event = SEC - 1;
        f
    }

    fn feed(t: &mut OrderTracker, e: ExecEvent) {
        let ev = Event::Exec(e);
        t.on_event(ev.ts_event(), &ev);
    }

    #[test]
    fn an_unpriced_open_position_makes_the_whole_answer_unknown_rather_than_smaller() {
        // The failure this prevents: a symbol whose feed died contributes 0 to the
        // mark-to-market, so a position going badly wrong reads as a position going
        // nowhere, and the session's bottom line is wrong by exactly the amount that
        // is invisible. `None` and a named symbol is the only honest report.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.1), 1)),
        );

        let marks = MarkCache::never_expires();
        // ETH is flat and unpriced, which must not matter; BTC is held and unpriced,
        // which must.
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(snap.unrealized, None);
        assert_eq!(snap.unpriced, vec!["BTC".to_string()]);
        assert_eq!(snap.net(), None);
        // …and the half that never needed a price is still answered.
        assert_eq!(snap.realized_net(), dec!(-0.1));
    }

    #[test]
    fn a_flat_symbol_without_a_mark_is_not_reported_as_unpriced() {
        // The mirror of the test above, and the reason `unpriced` is computed from
        // held positions rather than from mark coverage: a session configured for six
        // instruments and holding one would otherwise report five phantom faults and
        // refuse to price a P&L that is perfectly knowable.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.1), 1)),
        );
        let marks = MarkCache::never_expires();
        marks.set_mark(BTC, dec!(110), SEC);

        let snap = snapshot(&t, &marks, &universe());
        assert!(snap.unpriced.is_empty(), "ETH is flat: {:?}", snap.unpriced);
        assert_eq!(snap.unrealized, Some(dec!(10)));
        assert_eq!(snap.net(), Some(dec!(9.9)), "10 unrealized less 0.1 of fee");
        assert_eq!(snap.gross_exposure, dec!(110));
    }

    #[test]
    fn fees_are_subtracted_from_the_bottom_line_and_kept_visible_beside_it() {
        // A net figure that folded the fee in would make the two most common live
        // outcomes — a strategy with no edge, and a strategy whose edge is smaller
        // than its costs — indistinguishable on the status line.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Buy, dec!(2), dec!(100), dec!(0.09), 1)),
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Sell, dec!(2), dec!(101), dec!(0.09), 2)),
        );
        let marks = MarkCache::never_expires();

        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(snap.realized, dec!(2), "2 units, 1 of price improvement");
        assert_eq!(snap.fees, dec!(0.18));
        assert_eq!(snap.net(), Some(dec!(1.82)));
        // Flat, so the mark-to-market is a known zero rather than an unknown.
        assert_eq!(snap.unrealized, Some(Decimal::ZERO));
    }

    #[test]
    fn a_replayed_fill_moves_neither_the_position_nor_the_fee_total() {
        // The reason the fee total lives in the tracker: this dedup. A reconnect
        // replays a snapshot, and a fee accumulator anywhere else would double-count
        // it — in the direction that makes a losing session look profitable.
        let mut t = OrderTracker::new();
        let f = fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.05), 7);
        feed(&mut t, ExecEvent::Fill(f.clone()));
        feed(&mut t, ExecEvent::Fill(f.clone()));
        feed(&mut t, ExecEvent::Fill(f));

        let marks = MarkCache::never_expires();
        marks.set_mark(BTC, dec!(100), SEC);
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(snap.fees, dec!(0.05));
        assert_eq!(snap.fills, 1);
        assert_eq!(snap.gross_exposure, dec!(100), "one unit, not three");
    }

    #[test]
    fn the_venues_answer_and_ours_are_both_reported_and_neither_is_corrected() {
        // Funding, and anything else on a shared account, moves `accountValue` without
        // producing a fill. The drift figure exists to make that visible; a component
        // that "reconciled" the two would report a clean session by construction.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1000),
                withdrawable: dec!(1000),
                margin_used: Decimal::ZERO,
                ts_event: SEC,
            }),
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.1), 1)),
        );
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Sell, dec!(1), dec!(102), dec!(0.1), 2)),
        );
        // The venue says the account gained 1.5; our fills say 2 less 0.2 of fee = 1.8.
        // The 0.3 is funding as far as anything here can tell, and it stays visible.
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1001.5),
                withdrawable: dec!(1001.5),
                margin_used: Decimal::ZERO,
                ts_event: 2 * SEC,
            }),
        );

        let marks = MarkCache::never_expires();
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(
            snap.equity_at_start,
            Some(dec!(1000)),
            "the FIRST reading, kept"
        );
        assert_eq!(snap.equity, Some(dec!(1001.5)));
        assert_eq!(snap.equity_delta, Some(dec!(1.5)));
        assert_eq!(snap.net(), Some(dec!(1.8)));
        assert_eq!(snap.drift, Some(dec!(0.3)));
    }

    #[test]
    fn the_venues_replayed_fill_history_is_not_reported_as_this_sessions_result() {
        // Measured live on 2026-07-27: a session that had placed no order opened at
        // `r +0.0161 fee 0.1031` over 22 fills `userFills` replayed at subscribe, and
        // drifted -0.0869 against an equity that had not moved. Neither number was
        // wrong; the subtraction was, because `accountValue` already contains those
        // trades. Both sides are marked at one instant instead.
        let mut t = OrderTracker::new();
        t.set_session_start(SEC);
        // The venue replays two old fills the instant we subscribe. Their `ts_event` is
        // before the session started, which is the whole basis of the split — and they
        // arrive *before* the first `accountValue`, which is the ordering the first
        // version of this got wrong.
        feed(
            &mut t,
            ExecEvent::Fill(old(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.05), 1))),
        );
        feed(
            &mut t,
            ExecEvent::Fill(old(fill(
                BTC,
                Side::Sell,
                dec!(1),
                dec!(103),
                dec!(0.05),
                2,
            ))),
        );
        // …and then answers `clearinghouseState`, which already reflects them.
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1000),
                withdrawable: dec!(1000),
                margin_used: Decimal::ZERO,
                ts_event: SEC,
            }),
        );

        let marks = MarkCache::never_expires();
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(
            snap.realized,
            Decimal::ZERO,
            "this session has done nothing"
        );
        assert_eq!(snap.fees, Decimal::ZERO);
        assert_eq!(snap.fills, 0);
        assert_eq!(snap.net(), Some(Decimal::ZERO));
        assert_eq!(
            snap.drift,
            Some(Decimal::ZERO),
            "and it agrees with the venue"
        );
        // …and the process's own totals are still there, unsubtracted.
        assert_eq!(snap.realized_all, dec!(3));
        assert_eq!(snap.fees_all, dec!(0.10));
        assert_eq!(snap.fills_all, 2);
        assert!(snap.to_string().contains("(+2 pre)"), "{snap}");
    }

    #[test]
    fn a_fill_after_the_baseline_is_this_sessions_and_is_measured_against_its_equity() {
        // The other side of the same mechanism: everything after the mark counts, and
        // the drift is then a statement about funding and about fills nobody here saw,
        // rather than about the replay.
        let mut t = OrderTracker::new();
        t.set_session_start(SEC);
        feed(
            &mut t,
            ExecEvent::Fill(old(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.05), 1))),
        );
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1000),
                withdrawable: dec!(1000),
                margin_used: Decimal::ZERO,
                ts_event: SEC,
            }),
        );
        // Now this session trades: it closes the inherited long at 102.
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Sell, dec!(1), dec!(102), dec!(0.02), 2)),
        );
        feed(
            &mut t,
            ExecEvent::Account(AccountSnapshot {
                equity: dec!(1001.98),
                withdrawable: dec!(1001.98),
                margin_used: Decimal::ZERO,
                ts_event: 2 * SEC,
            }),
        );

        let marks = MarkCache::never_expires();
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(snap.realized, dec!(2), "the 2 points this session closed");
        assert_eq!(snap.fees, dec!(0.02), "and only the fee it paid");
        assert_eq!(snap.fills, 1);
        assert_eq!(snap.net(), Some(dec!(1.98)));
        assert_eq!(snap.equity_delta, Some(dec!(1.98)));
        assert_eq!(snap.drift, Some(Decimal::ZERO));
    }

    #[test]
    fn an_unreadable_tracker_reports_nothing_rather_than_a_believable_zero() {
        // The poisoned-lock path. Assembling from an empty tracker instead would print
        // `pnl +0.0000`, which is simultaneously the most believable thing this module
        // can say and the most wrong: a session holding a position it cannot see would
        // report that it had done nothing at all.
        let snap = PnlSnapshot::unreadable();
        assert!(!snap.readable);
        assert_eq!(snap.net(), None);
        assert_eq!(snap.to_string(), "pnl UNREADABLE");
    }

    #[test]
    fn a_session_the_venue_has_never_answered_reports_absence_rather_than_zero() {
        // An offline run, or a live one before its first reconcile poll. Zero equity
        // and zero drift would read as "the account is worth nothing and we agree".
        let t = OrderTracker::new();
        let marks = MarkCache::never_expires();
        let snap = snapshot(&t, &marks, &universe());
        assert_eq!(snap.equity, None);
        assert_eq!(snap.equity_delta, None);
        assert_eq!(snap.drift, None);
        assert_eq!(
            snap.net(),
            Some(Decimal::ZERO),
            "flat and priced: a known zero"
        );
    }

    #[test]
    fn the_rendered_line_says_unknown_rather_than_printing_a_number_it_does_not_have() {
        // The format is the interface an operator reads under pressure. A `-` that
        // silently became `+0.0000` would be the one rendering bug that turns a
        // refusal to guess into a guess.
        let mut t = OrderTracker::new();
        feed(
            &mut t,
            ExecEvent::Fill(fill(BTC, Side::Buy, dec!(1), dec!(100), dec!(0.1), 1)),
        );
        let marks = MarkCache::never_expires();
        let line = snapshot(&t, &marks, &universe()).to_string();
        assert!(line.starts_with("pnl - "), "{line}");
        assert!(line.contains("u -)"), "{line}");
        assert!(
            !line.contains("eq "),
            "no venue reading, no equity block: {line}"
        );
    }
}
