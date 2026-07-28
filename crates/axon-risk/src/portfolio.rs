//! Bounds on the **set** of positions, rather than on any one of them.
//!
//! [`RiskLimits`](crate::RiskLimits) is per instrument: `max_position`,
//! `max_notional` and `max_order_qty` each answer a question about one symbol, and a
//! session that trades one symbol through one strategy needs nothing else. The moment
//! several strategies run against one account that stops being true, and it stops being
//! true in a way no per-symbol limit can see: ten instruments each sitting at 90 % of
//! their own `max_notional` are inside every limit the session declares and hold nine
//! times the exposure the operator had in mind.
//!
//! So this module bounds three quantities that only exist across symbols:
//!
//! - **Gross notional** — `Σ |qty_i| · mark_i`. What the account has at risk if every
//!   position is wrong at once. The bound most operators mean when they say "how big
//!   is this allowed to get".
//! - **Net notional** — `|Σ qty_i · mark_i|`. Directional exposure to the market as a
//!   whole. A book that is long BTC and short ETH is smaller by this measure than by
//!   gross, which is the entire point of running strategies that disagree.
//! - **Breadth** — how many instruments may carry exposure at once. Not expressible as
//!   a notional at all: twenty $5 positions and one $100 position are the same gross
//!   and are not the same operational problem, because each instrument is a feed that
//!   can go stale, a grid that can be wrong and a position somebody has to close.
//!
//! ## The rule that shapes every branch below
//!
//! **A portfolio bound never refuses an order that reduces its own leg.** This is the
//! same lesson [`crate::reduces_exposure`]'s two existing callers learned — the
//! reduce-only rule and `axon_execution::LossLimiter` — and it is sharper here, because
//! a portfolio bound is the limit most likely to be *already breached* at the moment
//! somebody needs to get out. A gate that refuses a de-risking order because the book
//! is too big pins the account into exactly the position it is trying to leave.
//!
//! One consequence is stated rather than discovered: `|net|` can grow past
//! [`PortfolioLimits::max_net_notional`] through de-risking alone. Long 100 and short
//! 40 nets to 60; reduce the short to 10 and the net is 90, on an order that took
//! exposure off. The bound therefore binds on *new* exposure and never on an exit, and
//! that is the honest description of what it guarantees.
//!
//! ## Unpriced is refused, not assumed
//!
//! A gross figure computed over a book where one leg has no mark is not a gross figure.
//! [`PortfolioExposure::gross`] returns `None` in that case rather than summing what it
//! can, and the engine turns that into a refusal for anything that could add exposure —
//! the same fail-closed shape `axon_execution::GuardedClient` already applies to a
//! missing mark on a single order, for the same reason: an unpriced check is not a
//! check.

use axon_core::{Decimal, Side, SymbolId};
use axon_providers::OrderRequest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::reduces_exposure;

/// Bounds across every instrument at once. `0` is **no bound declared**, on the same
/// argument [`axon_execution::LossLimits`] gives: a default that fires is a default
/// nobody chose, and this whole table is inert until an operator writes a number in it.
///
/// The two notionals are magnitudes in quote units rather than fractions of equity,
/// matching every other money bound in this project — the account is shared and its
/// balance is not a scale anyone chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PortfolioLimits {
    /// Ceiling on `Σ |qty_i| · mark_i`.
    #[serde(default)]
    pub max_gross_notional: Decimal,
    /// Ceiling on `|Σ qty_i · mark_i|`.
    #[serde(default)]
    pub max_net_notional: Decimal,
    /// How many instruments may carry a non-zero position at once. `0` is unbounded.
    ///
    /// Enforced only against **opening a new one**: an order in an instrument the book
    /// is already exposed to is never refused on breadth, because refusing it would
    /// include refusing the order that closes it.
    #[serde(default)]
    pub max_symbols: u32,
}

impl PortfolioLimits {
    /// Whether anything here can ever refuse an order.
    pub fn is_declared(&self) -> bool {
        self.max_gross_notional > Decimal::ZERO
            || self.max_net_notional > Decimal::ZERO
            || self.max_symbols > 0
    }

    /// Reject the sign error that would refuse every order from startup.
    ///
    /// A negative notional cap is not a loose bound, it is a bound nothing can satisfy:
    /// a gross is a sum of magnitudes and is never below zero, so `-1` refuses the first
    /// order and every one after it. The same shape [`axon_execution::LossLimits`]
    /// refuses, arriving from the other direction.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("max_gross_notional", self.max_gross_notional),
            ("max_net_notional", self.max_net_notional),
        ] {
            if v < Decimal::ZERO {
                return Err(format!(
                    "portfolio.{name} {v} is negative; it is a magnitude bounding a sum \
                     of magnitudes, so a negative value refuses every order rather than \
                     permitting them"
                ));
            }
        }
        // A net ceiling above the gross one can never bind, because `|Σ x| <= Σ |x|` for
        // every book. Said rather than silently tolerated: an operator who wrote one
        // believes they have a directional bound and has a decoration.
        if self.max_gross_notional > Decimal::ZERO
            && self.max_net_notional > self.max_gross_notional
        {
            return Err(format!(
                "portfolio.max_net_notional {} is above max_gross_notional {}, and \
                 |sum| <= sum of |.| for every book - so the net bound can never bind \
                 and is not the directional limit it looks like",
                self.max_net_notional, self.max_gross_notional
            ));
        }
        Ok(())
    }
}

/// One instrument's contribution to the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortfolioLeg {
    pub symbol: SymbolId,
    /// Signed: `> 0` long, `< 0` short.
    pub qty: Decimal,
    /// The mark this leg is valued at, or `None` when nothing usable is known.
    ///
    /// An `Option` rather than a zero because the two are different facts and only one
    /// of them is safe: a leg priced at zero contributes nothing to a gross and reads
    /// as a book that is smaller than it is.
    pub mark: Option<Decimal>,
}

/// The book, as much of it as the caller can see.
///
/// Deliberately a flat `Vec` rather than a map: a session holds a handful of
/// instruments, the scans below are all linear in that handful, and a `HashMap` would
/// buy nothing but a non-deterministic iteration order — which this type must not have,
/// because [`PortfolioEngine`]'s breadth decision depends on the order it considers
/// legs in and a replay has to reach the same verdict.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PortfolioExposure {
    legs: Vec<PortfolioLeg>,
}

impl PortfolioExposure {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            legs: Vec::with_capacity(n),
        }
    }

    /// Record what one instrument holds. Replaces any earlier entry for it, so a caller
    /// building this from two sources cannot double-count a symbol.
    pub fn set(&mut self, symbol: SymbolId, qty: Decimal, mark: Option<Decimal>) {
        match self.legs.iter_mut().find(|l| l.symbol == symbol) {
            Some(leg) => {
                leg.qty = qty;
                leg.mark = mark;
            }
            None => self.legs.push(PortfolioLeg { symbol, qty, mark }),
        }
    }

    /// Builder form, for tests and for call sites assembling a literal.
    pub fn with(mut self, symbol: SymbolId, qty: Decimal, mark: Option<Decimal>) -> Self {
        self.set(symbol, qty, mark);
        self
    }

    pub fn legs(&self) -> &[PortfolioLeg] {
        &self.legs
    }

    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    /// What this book holds in `symbol`. Flat when the symbol is not in it — an
    /// instrument nobody has traded has no exposure, which is a fact rather than a
    /// guess.
    pub fn qty(&self, symbol: SymbolId) -> Decimal {
        self.legs
            .iter()
            .find(|l| l.symbol == symbol)
            .map(|l| l.qty)
            .unwrap_or(Decimal::ZERO)
    }

    pub fn mark(&self, symbol: SymbolId) -> Option<Decimal> {
        self.legs
            .iter()
            .find(|l| l.symbol == symbol)
            .and_then(|l| l.mark)
    }

    /// Project an order onto this book as if it fills in full.
    ///
    /// Full-fill is the only safe assumption for a resting order, and it is the same one
    /// `axon_execution::GuardedClient::check_batch` already makes when it carries a
    /// batch forward order by order.
    pub fn apply(&mut self, symbol: SymbolId, side: Side, qty: Decimal, mark: Option<Decimal>) {
        let signed = match side {
            Side::Buy => qty,
            Side::Sell => -qty,
        };
        match self.legs.iter_mut().find(|l| l.symbol == symbol) {
            Some(leg) => {
                leg.qty += signed;
                // A mark arriving with the order fills a gap and never overwrites a
                // reading the caller already had: the book's own price is the one the
                // rest of the sum is measured in.
                if leg.mark.is_none() {
                    leg.mark = mark;
                }
            }
            None => self.legs.push(PortfolioLeg {
                symbol,
                qty: signed,
                mark,
            }),
        }
    }

    /// `Σ |qty_i| · mark_i`, or `None` if any non-flat leg has no mark.
    ///
    /// The `None` is the whole value of this function. Summing the legs that happen to
    /// be priced would produce a smaller, entirely plausible number on exactly the book
    /// whose price feed has died — and a bound compared against it would read a session
    /// that cannot see half its positions as a session that does not have them.
    pub fn gross(&self) -> Option<Decimal> {
        let mut total = Decimal::ZERO;
        for leg in &self.legs {
            if leg.qty.is_zero() {
                continue;
            }
            total += leg.qty.abs() * leg.mark?;
        }
        Some(total)
    }

    /// `|Σ qty_i · mark_i|`, or `None` under the same rule as [`Self::gross`].
    pub fn net(&self) -> Option<Decimal> {
        let mut total = Decimal::ZERO;
        for leg in &self.legs {
            if leg.qty.is_zero() {
                continue;
            }
            total += leg.qty * leg.mark?;
        }
        Some(total.abs())
    }

    /// How many instruments carry a non-zero position.
    pub fn open_symbols(&self) -> u32 {
        self.legs.iter().filter(|l| !l.qty.is_zero()).count() as u32
    }

    /// Every symbol with a non-zero position, in the order the legs were recorded.
    pub fn open(&self) -> impl Iterator<Item = &PortfolioLeg> {
        self.legs.iter().filter(|l| !l.qty.is_zero())
    }
}

/// Why the portfolio refused an order.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PortfolioReject {
    #[error("projected gross notional {gross} exceeds portfolio max {max}")]
    GrossNotional { gross: Decimal, max: Decimal },
    #[error("projected net notional {net} exceeds portfolio max {max}")]
    NetNotional { net: Decimal, max: Decimal },
    #[error(
        "the book holds {symbol} unpriced, so the portfolio total cannot be computed - \
         refusing to add exposure blind"
    )]
    Unpriced { symbol: SymbolId },
    #[error(
        "opening {symbol} would make {open} instruments carry exposure, past the \
         portfolio max of {max}"
    )]
    SymbolCount {
        symbol: SymbolId,
        open: u32,
        max: u32,
    },
    #[error(
        "no portfolio exposure could be read, and a declared portfolio bound cannot be \
         checked against an unknown book"
    )]
    Unreadable,
}

/// The across-symbol gate.
#[derive(Debug, Clone, Default)]
pub struct PortfolioEngine {
    limits: PortfolioLimits,
}

impl PortfolioEngine {
    pub fn new(limits: PortfolioLimits) -> Self {
        Self { limits }
    }

    pub fn limits(&self) -> &PortfolioLimits {
        &self.limits
    }

    pub fn is_declared(&self) -> bool {
        self.limits.is_declared()
    }

    /// Check `req` against the book as it stands *before* the order.
    ///
    /// Three inputs rather than two, and the split matters. `before` is the **projected**
    /// book — every resting order counted as if it fills — because that is what a size
    /// cap has to be measured against, and a gross notional cap is a size cap. `held` is
    /// what the account actually holds in this order's own instrument, because the
    /// de-risk exemption below is asking a different question: flat with a resting buy
    /// projects long, so measured against the projection a *sell* would read as a
    /// reduction while in fact opening a short. `axon_execution::LossLimiter` splits the
    /// same pair for the same reason.
    ///
    /// `mark` is the price this order's own instrument is valued at, supplied separately
    /// because the caller may know a price for a symbol the book has no leg for yet —
    /// the first order in an instrument is exactly that case.
    pub fn check(
        &self,
        req: &OrderRequest,
        before: &PortfolioExposure,
        held: Decimal,
        mark: Option<Decimal>,
    ) -> Result<(), PortfolioReject> {
        if !self.limits.is_declared() {
            return Ok(());
        }

        // The exemption that shapes the module. An order taking exposure off its own leg
        // passes whatever the totals say, because the alternative is a bound that gets
        // tighter exactly as it is breached and refuses the exit while doing it.
        if reduces_exposure(req.side, req.qty, held) {
            return Ok(());
        }
        let projected = before.qty(req.symbol_id);

        // Breadth first, and against the book *before* the order: the question is whether
        // this order opens an instrument that was not open, so it has to be asked before
        // the projection makes it open.
        if self.limits.max_symbols > 0 && projected.is_zero() && !req.qty.is_zero() {
            let open = before.open_symbols();
            if open >= self.limits.max_symbols {
                return Err(PortfolioReject::SymbolCount {
                    symbol: req.symbol_id,
                    open: open + 1,
                    max: self.limits.max_symbols,
                });
            }
        }

        if self.limits.max_gross_notional.is_zero() && self.limits.max_net_notional.is_zero() {
            return Ok(());
        }

        let mut after = before.clone();
        after.apply(req.symbol_id, req.side, req.qty, mark);

        // An unpriced leg is named, so the log says which feed to go and look at rather
        // than reporting an arithmetic failure.
        let unpriced = || {
            after
                .open()
                .find(|l| l.mark.is_none())
                .map(|l| PortfolioReject::Unpriced { symbol: l.symbol })
                .unwrap_or(PortfolioReject::Unreadable)
        };

        if self.limits.max_gross_notional > Decimal::ZERO {
            let gross = after.gross().ok_or_else(unpriced)?;
            if gross > self.limits.max_gross_notional {
                return Err(PortfolioReject::GrossNotional {
                    gross,
                    max: self.limits.max_gross_notional,
                });
            }
        }
        if self.limits.max_net_notional > Decimal::ZERO {
            let net = after.net().ok_or_else(unpriced)?;
            if net > self.limits.max_net_notional {
                return Err(PortfolioReject::NetNotional {
                    net,
                    max: self.limits.max_net_notional,
                });
            }
        }
        Ok(())
    }
}

/// The factor a set of desired targets must be multiplied by to fit `max`, or `None`
/// when it already fits.
///
/// Separate from [`PortfolioEngine::check`] and used before it rather than instead of
/// it, because the two do different jobs and both are needed. The gate refuses an order
/// that would breach the bound, which is the guarantee; scaling makes the targets the
/// session works toward *feasible*, so a strategy asking for more than the portfolio
/// allows converges to the largest position it is permitted instead of emitting a
/// target that is refused forever. Without the scale, a bound that binds presents
/// exactly like a strategy whose orders keep failing.
///
/// `None` when `gross` is zero, when no bound is declared, or when the book already
/// fits — the caller then applies nothing, which is not the same as applying a factor
/// of one: a factor of one still multiplies every target and would turn exact wire
/// quantities into values the fixed-point contract may not represent.
pub fn gross_scale(gross: Decimal, max: Decimal) -> Option<Decimal> {
    if max <= Decimal::ZERO || gross <= max || gross.is_zero() {
        return None;
    }
    max.checked_div(gross)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Tif};
    use rust_decimal_macros::dec;

    const BTC: SymbolId = SymbolId::new(1);
    const ETH: SymbolId = SymbolId::new(2);
    const SOL: SymbolId = SymbolId::new(3);

    fn order(symbol: SymbolId, side: Side, qty: Decimal) -> OrderRequest {
        OrderRequest::limit(symbol, side, qty, dec!(100), Tif::Gtc, Cloid::new(1))
    }

    fn engine(gross: Decimal, net: Decimal, symbols: u32) -> PortfolioEngine {
        PortfolioEngine::new(PortfolioLimits {
            max_gross_notional: gross,
            max_net_notional: net,
            max_symbols: symbols,
        })
    }

    #[test]
    fn an_undeclared_portfolio_refuses_nothing() {
        // The whole table is inert until somebody writes a number in it, so upgrading a
        // binary can never stop a session that was running yesterday.
        let e = PortfolioEngine::default();
        assert!(!e.is_declared());
        let huge = order(BTC, Side::Buy, dec!(1000));
        assert!(e
            .check(
                &huge,
                &PortfolioExposure::new(),
                Decimal::ZERO,
                Some(dec!(60000))
            )
            .is_ok());
    }

    #[test]
    fn gross_counts_both_sides_and_net_cancels_them() {
        // The distinction the two bounds exist for: a book long one coin and short
        // another is the same gross as one long both, and a very different net. Running
        // strategies that disagree is the reason to have both numbers.
        let book = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(-1), Some(dec!(40)));
        assert_eq!(book.gross(), Some(dec!(140)));
        assert_eq!(book.net(), Some(dec!(60)));

        let same_side = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(1), Some(dec!(40)));
        assert_eq!(
            same_side.gross(),
            Some(dec!(140)),
            "gross cannot tell them apart"
        );
        assert_eq!(same_side.net(), Some(dec!(140)), "net can");
    }

    #[test]
    fn a_gross_over_an_unpriced_book_is_absent_rather_than_partial() {
        // The failure this prevents: summing the legs that happen to be priced produces
        // a smaller, entirely plausible number on exactly the book whose feed died, and
        // a bound compared against it reads a session that cannot see half its positions
        // as a session that does not have them.
        let book = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(-1), None);
        assert_eq!(book.gross(), None);
        assert_eq!(book.net(), None);

        // A *flat* leg with no mark is not a hole: it contributes nothing whatever it is
        // worth, so the totals are still knowable.
        let flat = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, Decimal::ZERO, None);
        assert_eq!(flat.gross(), Some(dec!(100)));
    }

    #[test]
    fn an_order_that_reduces_its_own_leg_is_never_refused_however_big_the_book_is() {
        // The rule the whole module is shaped around. A portfolio bound is the limit
        // most likely to be already breached when somebody needs to get out, and a gate
        // that refuses the exit pins the account into the position it is leaving — the
        // same trap ADR-0031 named for cancels and ADR-0037 for the loss switch.
        let e = engine(dec!(50), Decimal::ZERO, 0);
        let book = PortfolioExposure::new()
            .with(BTC, dec!(10), Some(dec!(100)))
            .with(ETH, dec!(10), Some(dec!(100)));
        assert_eq!(book.gross(), Some(dec!(2000)), "far past the bound already");

        let exit = order(BTC, Side::Sell, dec!(10));
        assert!(e.check(&exit, &book, dec!(10), Some(dec!(100))).is_ok());

        // …and one that adds to it is refused, on the same book.
        let add = order(BTC, Side::Buy, dec!(1));
        assert!(matches!(
            e.check(&add, &book, dec!(10), Some(dec!(100))),
            Err(PortfolioReject::GrossNotional { .. })
        ));
    }

    #[test]
    fn de_risking_may_grow_the_net_past_its_bound_and_that_is_the_documented_trade() {
        // Long 100 and short 40 nets to 60; reducing the short to 10 nets 90, on an
        // order that took exposure off. Asserted rather than left as prose because a
        // future reader will otherwise read this as a hole: the net bound binds on new
        // exposure and never on an exit, and that is what it guarantees.
        let e = engine(Decimal::ZERO, dec!(70), 0);
        let book = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(-1), Some(dec!(40)));
        assert_eq!(book.net(), Some(dec!(60)), "inside the bound to start with");

        // Buying back three quarters of the short reduces ETH and lands the net at 90.
        let cover = order(ETH, Side::Buy, dec!(0.75));
        assert!(e.check(&cover, &book, dec!(-1), Some(dec!(40))).is_ok());
        let mut after = book.clone();
        after.apply(ETH, Side::Buy, dec!(0.75), Some(dec!(40)));
        assert_eq!(after.net(), Some(dec!(90)), "past the bound, by de-risking");
    }

    #[test]
    fn adding_exposure_to_an_unpriced_book_is_refused_and_names_the_symbol() {
        // Fail closed, and say which feed to go and look at: an operator reading
        // "portfolio arithmetic failed" has to guess, and the guess is what costs the
        // next twenty minutes.
        let e = engine(dec!(1000), Decimal::ZERO, 0);
        let book = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(-1), None);
        let add = order(SOL, Side::Buy, dec!(1));
        assert_eq!(
            e.check(&add, &book, Decimal::ZERO, Some(dec!(10))),
            Err(PortfolioReject::Unpriced { symbol: ETH })
        );
    }

    #[test]
    fn breadth_refuses_a_new_instrument_and_never_one_already_open() {
        // Twenty $5 positions and one $100 position are the same gross and are not the
        // same operational problem — each instrument is a feed that can go stale and a
        // position somebody has to close. And the cap must never be able to refuse the
        // order that *closes* one, which is why it is asked against the book before the
        // order and only when the leg is flat.
        let e = engine(Decimal::ZERO, Decimal::ZERO, 2);
        let book = PortfolioExposure::new()
            .with(BTC, dec!(1), Some(dec!(100)))
            .with(ETH, dec!(1), Some(dec!(100)));

        let third = order(SOL, Side::Buy, dec!(1));
        assert!(matches!(
            e.check(&third, &book, Decimal::ZERO, Some(dec!(10))),
            Err(PortfolioReject::SymbolCount { max: 2, .. })
        ));

        // Adding to one of the two already open is fine — breadth has not changed.
        assert!(e
            .check(
                &order(BTC, Side::Buy, dec!(1)),
                &book,
                dec!(1),
                Some(dec!(100))
            )
            .is_ok());
        // As is closing one.
        assert!(e
            .check(
                &order(ETH, Side::Sell, dec!(1)),
                &book,
                dec!(1),
                Some(dec!(100))
            )
            .is_ok());

        // And a book with a flat leg in it has room: the leg is recorded, not open.
        let with_flat = book.clone().with(SOL, Decimal::ZERO, Some(dec!(10)));
        assert_eq!(with_flat.open_symbols(), 2);
    }

    #[test]
    fn a_negative_bound_is_refused_because_it_would_permit_nothing() {
        // A gross is a sum of magnitudes, so a negative ceiling is not a loose bound —
        // it refuses the first order and every one after it. The mirror of
        // `LossLimits::validate`, which refuses the same typo from the other direction.
        let bad = PortfolioLimits {
            max_gross_notional: dec!(-1),
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        assert!(PortfolioLimits::default().validate().is_ok());
    }

    #[test]
    fn a_net_ceiling_above_the_gross_one_is_refused_as_a_decoration() {
        // |sum| <= sum of |.| for every book, so this configuration can never bind. An
        // operator who wrote it believes they have a directional bound and has nothing.
        let never_binds = PortfolioLimits {
            max_gross_notional: dec!(100),
            max_net_notional: dec!(200),
            max_symbols: 0,
        };
        assert!(never_binds.validate().is_err());
        // Equal is legal and is the ordinary "no netting benefit claimed" setting.
        let equal = PortfolioLimits {
            max_net_notional: dec!(100),
            ..never_binds
        };
        assert!(equal.validate().is_ok());
    }

    #[test]
    fn the_scale_factor_is_absent_when_nothing_needs_scaling() {
        // Absent rather than one: a factor of one still multiplies every target, and the
        // product of an exact wire quantity and a Decimal one is a value the fixed-point
        // contract may not be able to represent — so "no scaling" has to mean "do not
        // multiply", not "multiply by the identity".
        assert_eq!(gross_scale(dec!(50), dec!(100)), None);
        assert_eq!(
            gross_scale(dec!(100), dec!(100)),
            None,
            "exactly at the bound"
        );
        assert_eq!(gross_scale(dec!(0), dec!(100)), None);
        assert_eq!(gross_scale(dec!(500), Decimal::ZERO), None, "undeclared");

        // And when it does bind it is the ratio, exactly.
        assert_eq!(gross_scale(dec!(200), dec!(100)), Some(dec!(0.5)));
        assert_eq!(gross_scale(dec!(400), dec!(100)), Some(dec!(0.25)));
    }

    #[test]
    fn setting_a_leg_twice_replaces_it_rather_than_double_counting() {
        // The failure this prevents: a caller assembling the book from the tracker and
        // then correcting one symbol from a fresher read would otherwise hold two legs
        // for one instrument and report twice the gross.
        let mut book = PortfolioExposure::new();
        book.set(BTC, dec!(1), Some(dec!(100)));
        book.set(BTC, dec!(2), Some(dec!(100)));
        assert_eq!(book.legs().len(), 1);
        assert_eq!(book.gross(), Some(dec!(200)));
    }

    #[test]
    fn projecting_an_order_keeps_the_marks_the_book_was_measured_in() {
        // A mark arriving with an order fills a gap and never overwrites one the book
        // already had — otherwise one leg of a sum would be valued at a different price
        // from the rest of it depending on which order happened to be checked.
        let mut book = PortfolioExposure::new().with(BTC, dec!(1), Some(dec!(100)));
        book.apply(BTC, Side::Buy, dec!(1), Some(dec!(999)));
        assert_eq!(book.gross(), Some(dec!(200)), "still valued at 100");

        // …and it does fill a gap on a leg that had none.
        let mut fresh = PortfolioExposure::new().with(ETH, dec!(1), None);
        assert_eq!(fresh.gross(), None);
        fresh.apply(ETH, Side::Buy, dec!(1), Some(dec!(50)));
        assert_eq!(fresh.gross(), Some(dec!(100)));
    }
}
