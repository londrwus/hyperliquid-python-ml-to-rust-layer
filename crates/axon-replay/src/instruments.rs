//! The **instrument grids a recording session planned against**, in the log's own
//! wire form.
//!
//! Until this existed a captured log carried no table, so a replay of it planned under
//! [`Precision::Unconstrained`] while every live session plans under
//! [`Precision::Known`] and *rounds* (ADR-0025). The two disagree on exactly the orders
//! the grid moved — urgency-3 slippage, every price band — and
//! [`PlannedOrder::price`](crate::PlannedOrder::price) is compared **exactly** by
//! `python/axon/backtest/golden.py`. So a golden diff of a real capture reported a
//! strategy flip on every rounded order, inside the harness whose one job is to tell a
//! strategy change from a harness change. ADR-0025 recorded that as a hole its own
//! increment made *bigger*; ADR-0027 closes it by putting the table in the log.
//!
//! Three decisions, each the opposite of something that reads as obviously fine:
//!
//! - **The set owns its wire form**, exactly as [`LoggedEvent`](crate::LoggedEvent) and
//!   [`LoggedSignal`](crate::LoggedSignal) do. Deriving `Serialize` on
//!   [`InstrumentSpec`] would bind a persisted format to a type on the signing path and
//!   make every refactor of it a silent format change — and it would put serde
//!   attributes on the port, where ADR-0004 keeps venue-shaped concerns out.
//! - **The price grid is written as the two numbers it *is*, not as an enum.**
//!   ADR-0025 §1 argued a two-variant `{ Digits, Grid }` shape is a venue leak wearing
//!   a port's clothes, because venue three adds arm three; a log that reintroduced the
//!   enum would reintroduce the leak one layer down and would need a new arm — and a
//!   `SCHEMA_VERSION` bump — for a venue the port itself handles by setting a field.
//! - **[`InstrumentSet::Undeclared`] is a third state, not an empty table.** An empty
//!   [`InstrumentTable`] means every symbol is [`Precision::Unknown`], which refuses
//!   every order that would add exposure; a replay of a session's decisions rendered as
//!   refusals is the harness reporting its own ignorance as the strategy's output. And
//!   an *unconstrained* table is a different sentence again. Three states on the wire
//!   because there are three states in the type this reconstructs.

use axon_core::{Decimal, SymbolId};
use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid, SpecError};
use serde::{Deserialize, Serialize};

/// One instrument's number format, as a log carries it.
///
/// Flat, and every field is a number a venue publishes or a `Decimal` derived from one.
/// The conversion back is an exhaustive struct literal for the same reason
/// [`LoggedSignal`](crate::LoggedSignal)'s is: a field added to [`InstrumentSpec`] is a
/// compile error here until the log has been taught to carry it, where a
/// `..Default::default()` tail would let a whole rule replay as absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggedInstrument {
    pub symbol_id: u32,
    /// The floor of the price grid. `0` means the venue imposes none.
    pub price_increment: Decimal,
    /// The significant-figure cap that coarsens the grid as the price grows, when the
    /// venue has one. `null` is a venue whose tick is constant, which is a *different*
    /// grid from one whose cap happens not to bind at today's price.
    pub price_sig_figs: Option<u32>,
    /// The lot. `0` means the venue imposes none.
    pub size_increment: Decimal,
    /// Smallest order value the venue accepts, in quote currency.
    pub min_notional: Option<Decimal>,
}

impl LoggedInstrument {
    /// Write one spec down.
    ///
    /// Reads the grid through the port's own accessors rather than recomputing it: a
    /// writer that derived `max_decimals` from the increment's scale would be a second
    /// definition of [`PriceGrid`]'s shape, living in the crate that persists it, and
    /// it would keep compiling on the day that shape gains a third number.
    pub fn of(spec: &InstrumentSpec) -> Self {
        Self {
            symbol_id: spec.symbol_id.get(),
            price_increment: spec.price.min_increment(),
            price_sig_figs: spec.price.sig_figs(),
            size_increment: spec.size.increment(),
            min_notional: spec.min_notional,
        }
    }

    /// Read one spec back.
    pub fn to_spec(self) -> Result<InstrumentSpec, SpecError> {
        Ok(InstrumentSpec {
            symbol_id: SymbolId::new(self.symbol_id),
            price: PriceGrid::from_parts(self.price_increment, self.price_sig_figs)?,
            // Zero is "no lot", which is a state `step` refuses by design — it is the
            // one number where "the venue has no rule" and "the venue's rule is absurd"
            // share a value, and only the constructor pair can tell them apart.
            size: if self.size_increment.is_zero() {
                SizeGrid::unconstrained()
            } else {
                SizeGrid::step(self.size_increment)?
            },
            min_notional: self.min_notional,
        })
    }
}

/// What a recording session knew about instrument precision.
///
/// [`Undeclared`](Self::Undeclared) is not a degraded [`Declared`](Self::Declared) with
/// no entries. It says *this recording carries no grid*, which is the honest answer for
/// a writer that was never handed one, and it is what lets a reader fall back loudly
/// instead of silently refusing every order the session actually sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentSet {
    /// The session that wrote this log declared no instrument table.
    ///
    /// A replay of it plans **unconstrained**, which is more permissive than any live
    /// session, so its prices are not the prices that session sent. The reader says so;
    /// it does not pretend.
    Undeclared,
    /// The grids the session planned against.
    Declared {
        /// What a symbol absent from `instruments` meant to the session — the
        /// [`InstrumentTable`]'s own two "not here" answers, which are different
        /// sentences and must not collapse (ADR-0025 §1).
        unconstrained: bool,
        /// Ordered by `symbol_id`, so two captures of one session are byte-identical.
        instruments: Vec<LoggedInstrument>,
    },
}

impl InstrumentSet {
    /// Write a table down.
    pub fn of(table: &InstrumentTable) -> Self {
        Self::Declared {
            unconstrained: table.is_unconstrained(),
            instruments: table.specs().iter().map(LoggedInstrument::of).collect(),
        }
    }

    /// Read a table back, or `None` when the log declared none.
    ///
    /// `None` rather than an empty table, because the caller has to *decide* what to do
    /// about it and an empty table would let it not notice: an empty
    /// [`InstrumentTable`] refuses every opening order, so a replay handed one would
    /// report the session's decisions as precision refusals and the counter an operator
    /// reads would blame the strategy.
    pub fn to_table(&self) -> Result<Option<InstrumentTable>, SpecError> {
        let InstrumentSet::Declared {
            unconstrained,
            instruments,
        } = self
        else {
            return Ok(None);
        };
        let mut table = if *unconstrained {
            InstrumentTable::unconstrained()
        } else {
            InstrumentTable::new()
        };
        for logged in instruments {
            table.insert(logged.to_spec()?);
        }
        Ok(Some(table))
    }

    /// Whether this log carries a grid at all.
    pub fn is_declared(&self) -> bool {
        matches!(self, InstrumentSet::Declared { .. })
    }

    /// How many instruments it names. Zero for [`Undeclared`](Self::Undeclared), and
    /// also for a declared-but-empty table — the two are told apart by
    /// [`is_declared`](Self::is_declared), never by this.
    pub fn len(&self) -> usize {
        match self {
            InstrumentSet::Undeclared => 0,
            InstrumentSet::Declared { instruments, .. } => instruments.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_providers::Precision;
    use rust_decimal_macros::dec;

    /// The Hyperliquid perp shape: `6 - szDecimals` decimal places capped at five
    /// significant figures, a `10^-szDecimals` lot, and the venue's $10 minimum.
    fn perp(id: u32, sz: u32) -> InstrumentSpec {
        InstrumentSpec {
            symbol_id: SymbolId::new(id),
            price: PriceGrid::decimals_with_sig_figs(6 - sz, 5).unwrap(),
            size: SizeGrid::decimals(sz).unwrap(),
            min_notional: Some(dec!(10)),
        }
    }

    #[test]
    fn a_grid_that_went_through_the_log_quantizes_identically_to_the_one_that_wrote_it() {
        // The whole claim, and the only one that matters: a table written down and read
        // back must *round the same way*. Comparing the structs would pass over a
        // rebuild that lost `sig_figs`, because the increment alone still looks like a
        // grid — and the price it then plans is one the venue refuses outright.
        let original = perp(1, 5);
        let back = LoggedInstrument::of(&original).to_spec().unwrap();
        assert_eq!(back, original);

        for px in [
            dec!(49757.96),
            dec!(64774.26),
            dec!(0.0012345),
            dec!(100000),
        ] {
            for side in [axon_core::Side::Buy, axon_core::Side::Sell] {
                for intent in [
                    axon_providers::PriceIntent::Passive,
                    axon_providers::PriceIntent::Marketable,
                ] {
                    assert_eq!(
                        back.price.quantize(px, side, intent),
                        original.price.quantize(px, side, intent),
                        "{px} {side:?} {intent:?} rounded differently after a round trip"
                    );
                }
            }
        }
    }

    #[test]
    fn an_unconstrained_table_does_not_read_back_as_a_venue_with_unknown_grids() {
        // The two "not here" answers, which ADR-0025 §1 kept apart on purpose. A table
        // that lost `unconstrained` on the way through a log comes back refusing every
        // order that would add exposure — a replay reporting its own ignorance as the
        // strategy's output, and reporting it as `precision_refusals`, which sends an
        // operator to look at the venue.
        let loose = InstrumentSet::of(&InstrumentTable::unconstrained())
            .to_table()
            .unwrap()
            .expect("declared");
        assert!(matches!(
            loose.precision(SymbolId::new(9)),
            Precision::Unconstrained
        ));

        let strict = InstrumentSet::of(&InstrumentTable::new())
            .to_table()
            .unwrap()
            .expect("declared");
        assert!(matches!(
            strict.precision(SymbolId::new(9)),
            Precision::Unknown
        ));
    }

    #[test]
    fn a_log_that_declared_nothing_is_not_a_log_that_declared_an_empty_table() {
        // The distinction the fall-back rests on. Both carry zero instruments, and
        // handing the second to a planner refuses every opening order; a reader that
        // told them apart by `len()` would silently turn "we recorded no grid" into
        // "the venue has grids and we know none of them".
        assert!(InstrumentSet::Undeclared.to_table().unwrap().is_none());
        assert!(InstrumentSet::of(&InstrumentTable::new())
            .to_table()
            .unwrap()
            .is_some());
        assert_eq!(InstrumentSet::Undeclared.len(), 0);
        assert!(!InstrumentSet::Undeclared.is_declared());
    }

    #[test]
    fn instruments_are_written_in_symbol_order_so_two_captures_of_one_session_match() {
        // `InstrumentTable` is a `HashMap`, and `HashMap` iteration order is randomized
        // per process. Written out raw, one session recorded twice would produce two
        // different files — nondeterminism manufactured by the artifact whose entire
        // purpose is to be reproducible.
        let mut table = InstrumentTable::new();
        for id in [7u32, 2, 91, 1, 40] {
            table.insert(perp(id, 4));
        }
        let InstrumentSet::Declared { instruments, .. } = InstrumentSet::of(&table) else {
            panic!("declared");
        };
        let ids: Vec<u32> = instruments.iter().map(|i| i.symbol_id).collect();
        assert_eq!(ids, vec![1, 2, 7, 40, 91]);
    }

    #[test]
    fn an_instrument_record_pins_its_wire_shape_so_a_grid_change_cannot_pass_unnoticed() {
        // `SCHEMA_VERSION` is a human promise and this is what forces the human to keep
        // it: a field added to `InstrumentSpec`, a renamed one, a changed unit all break
        // here. Without it the table could drift while old logs kept deserializing into
        // the new shape, and a golden run would re-plan against a grid that no longer
        // means what it says.
        assert_eq!(
            serde_json::to_string(&InstrumentSet::of(&{
                let mut t = InstrumentTable::new();
                t.insert(perp(1, 5));
                t
            }))
            .unwrap(),
            r#"{"Declared":{"unconstrained":false,"instruments":[{"symbol_id":1,"price_increment":"0.1","price_sig_figs":5,"size_increment":"0.00001","min_notional":"10"}]}}"#,
            "money stays a decimal string, never a float"
        );
        assert_eq!(
            serde_json::to_string(&InstrumentSet::Undeclared).unwrap(),
            r#""Undeclared""#
        );
    }
}
