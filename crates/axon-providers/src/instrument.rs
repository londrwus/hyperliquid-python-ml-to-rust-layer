//! Per-**instrument** number format: the price grid, the lot, and the venue's
//! minimum order value (ADR-0025).
//!
//! [`Capabilities`](crate::Capabilities) answers "what can this *venue* express?" —
//! order types, TIFs, batch size, the rate model — and it is `&'static` and
//! const-constructible because those facts are compiled in. Tick and lot are not
//! venue facts at all: BTC and a $0.003 alt on the same venue have different grids,
//! and the numbers arrive over the network at startup. That is the whole reason this
//! is a separate type with its own table rather than four more fields on
//! `Capabilities`.
//!
//! The failure it exists to prevent is silent and total. A price the venue's grid
//! does not admit comes back `tickRejected`; a size finer than the lot comes back the
//! same way. From inside the process both read exactly like a signing bug — the
//! action was well-formed, the signature was valid, and the venue said no — so an
//! engine without this table looks like an engine with a broken signer, which is the
//! single most expensive place to be wrong.
//!
//! Nothing here is venue-specific. Hyperliquid populates it in
//! `axon_provider_hyperliquid::ws::decode_universe`; a CEX with a fixed
//! `PRICE_FILTER.tickSize` sets [`PriceGrid::increment`] and nothing else. A venue
//! gets a *field*, never an enum arm — the third venue must not force a `match` in
//! every consumer to grow.

use std::collections::HashMap;

use axon_core::{Decimal, Side, SymbolId};

use crate::order::OrderRequest;

/// Which way a price moves when it does not land on the grid.
///
/// Not "up/down" — the direction that preserves the order's *intent*, which is a
/// function of the side as well. Naming it after the intent is what stops a caller
/// choosing the cheap direction and converting a taker into a resting quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceIntent {
    /// The order is meant to rest. Rounds AWAY from the market.
    ///
    /// Not politeness: a post-only order is **rejected**, not demoted, if it would
    /// cross, so a passive buy rounded up into the spread is not a slightly worse
    /// quote — it is an `alo` rejection and a strategy sitting flat with nothing
    /// working.
    Passive,
    /// The order is meant to cross now. Rounds TOWARD the market.
    ///
    /// Rounding a taker away from the market can leave a limit that no longer
    /// crosses, which silently converts "take liquidity now" into a resting quote and
    /// leaves an unmanaged position — the exact outcome an urgent exit exists to
    /// prevent.
    Marketable,
}

/// How a venue quantizes an instrument's prices.
///
/// `increment` is the floor of the grid; `sig_figs` is a second cap that *coarsens*
/// the grid as the price grows, never past integer granularity. One shape, two
/// venues: a CEX sets `increment` alone and its tick is constant; Hyperliquid sets
/// both and BTC's tick widens from 0.1 to 1 as the price crosses 10 000.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceGrid {
    increment: Decimal,
    sig_figs: Option<u32>,
}

/// How a venue quantizes an instrument's sizes. `szDecimals: n` IS `increment = 10^-n`.
///
/// A one-field struct rather than a bare [`Decimal`] on [`InstrumentSpec`], because it
/// is the seam a CEX's `LOT_SIZE` needs (`minQty`/`maxQty` are fields here later, not a
/// second concept elsewhere) and because it keeps `quantize`/`is_valid` beside their
/// price counterparts, where a reviewer compares them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeGrid {
    increment: Decimal,
}

/// Everything the engine must know about one instrument's number format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstrumentSpec {
    pub symbol_id: SymbolId,
    pub price: PriceGrid,
    pub size: SizeGrid,
    /// Smallest order value the venue accepts, in quote currency. `None` = none.
    pub min_notional: Option<Decimal>,
}

/// What this session knows about one instrument's grid.
///
/// Three variants, not two, because "this venue has no rules" and "we do not know
/// this venue's rules" are different sentences. Collapsing them is how a backtest
/// becomes silently more permissive than the live session it claims to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision<'a> {
    Known(&'a InstrumentSpec),
    /// A venue with no instrument grids at all — a simulator, or a test that is not
    /// about rounding.
    Unconstrained,
    /// The venue has grids and this session does not know this one's.
    Unknown,
}

/// Per-instrument precision for one venue, keyed by the id everything above the
/// adapter speaks.
#[derive(Debug, Clone, Default)]
pub struct InstrumentTable {
    by_symbol: HashMap<SymbolId, InstrumentSpec>,
    /// What a symbol that is not in the map means. `false` (the default) = `Unknown`.
    unconstrained: bool,
}

/// A spec that could not be built.
///
/// Construction-time, and a `Result` rather than a `debug_assert` because these are
/// built from a venue's JSON: a `szDecimals` of 200 must be refused, not panic a live
/// session at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpecError {
    #[error("price increment {inc} is not positive")]
    BadIncrement { inc: Decimal },
    #[error("{what} {value} is out of range (max {max})")]
    OutOfRange {
        what: &'static str,
        value: u32,
        max: u32,
    },
}

/// An order the venue's own rules would reject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PrecisionError {
    #[error("price {px} is not a multiple of this instrument's tick {tick} at that price")]
    Tick { px: Decimal, tick: Decimal },
    #[error("size {qty} is not a multiple of the lot {lot}")]
    Lot { qty: Decimal, lot: Decimal },
    #[error("notional {notional} is below the venue minimum {min}")]
    BelowMinNotional { notional: Decimal, min: Decimal },
    #[error("no precision is known for {symbol:?}")]
    UnknownInstrument { symbol: SymbolId },
}

/// The widest scale a [`Decimal`] can hold. Past it a "finer grid" is not a grid.
const MAX_SCALE: u32 = 28;

// ── the arithmetic, with no `f64` at any step ────────────────────────────────

/// Decimal digits in a non-negative integer. `digits10(0) == 1`.
#[inline]
fn digits10(m: u128) -> u32 {
    if m == 0 {
        1
    } else {
        m.ilog10() + 1
    }
}

/// The decimal exponent `d`, defined by `10^(d-1) <= |px| < 10^d`.
///
/// Integer arithmetic on the mantissa and the scale, never `log10`: `(100000f64).log10()`
/// is `4.999…`, which drops a digit at exactly the magnitude where a five-significant-
/// figure rule bites, and the money on that path is a five-figure BTC price.
///
/// Normalizing first is what makes trailing zeros fall out: `100.00` has mantissa
/// `10000` and scale `2`, so `digits10` is 5 and `d = 5 - 2 = 3` — and `10^2 <= 100 <
/// 10^3`, which is the definition.
fn decimal_exponent(px: Decimal) -> i32 {
    let n = px.abs().normalize();
    if n.is_zero() {
        return 0;
    }
    digits10(n.mantissa().unsigned_abs()) as i32 - n.scale() as i32
}

/// The coarsest grid a `sig`-significant-figure cap permits at this price:
/// `10^(d - sig)`, **clamped at one**.
///
/// That clamp *is* the integer exemption, expressed as arithmetic rather than as a
/// special case a reader has to remember: past `sig` digits the quantum would exceed
/// one, and every venue that caps significant figures still accepts whole numbers.
fn sig_quantum(sig: u32, px: Decimal) -> Decimal {
    let k = decimal_exponent(px) - sig as i32;
    if k >= 0 {
        // Integers are always legal, however many significant figures they carry.
        return Decimal::ONE;
    }
    if -k > MAX_SCALE as i32 {
        // Finer than a Decimal can hold, so this rule cannot bind; the increment is
        // then the only thing that does.
        return Decimal::ZERO;
    }
    Decimal::new(1, (-k) as u32)
}

impl PriceGrid {
    /// A fixed grid: prices are exact multiples of `inc`. The CEX shape
    /// (`PRICE_FILTER.tickSize`).
    pub fn increment(inc: Decimal) -> Result<Self, SpecError> {
        if inc <= Decimal::ZERO {
            return Err(SpecError::BadIncrement { inc });
        }
        Ok(Self {
            increment: inc,
            sig_figs: None,
        })
    }

    /// At most `sig_figs` significant figures **or** an integer, and never more than
    /// `max_decimals` decimal places. The Hyperliquid shape.
    pub fn decimals_with_sig_figs(max_decimals: u32, sig_figs: u32) -> Result<Self, SpecError> {
        if max_decimals > MAX_SCALE {
            return Err(SpecError::OutOfRange {
                what: "max_decimals",
                value: max_decimals,
                max: MAX_SCALE,
            });
        }
        if sig_figs == 0 || sig_figs > MAX_SCALE {
            return Err(SpecError::OutOfRange {
                what: "sig_figs",
                value: sig_figs,
                max: MAX_SCALE,
            });
        }
        Ok(Self {
            increment: Decimal::new(1, max_decimals),
            sig_figs: Some(sig_figs),
        })
    }

    /// A venue that imposes no price grid at all (a simulator).
    ///
    /// `increment` is **zero**, which [`tick_at`](Self::tick_at) reads as "no grid".
    /// Chosen over a `1e-28` increment because a zero tick is not a tick: expressing
    /// "no constraint" as a number means one `checked_rem` away from a `None` that
    /// reads as "invalid price", and the honest answer to "quantize this" with no grid
    /// is the identity.
    pub fn unconstrained() -> Self {
        Self {
            increment: Decimal::ZERO,
            sig_figs: None,
        }
    }

    /// Rebuild a grid from the two numbers [`min_increment`](Self::min_increment) and
    /// [`sig_figs`](Self::sig_figs) report.
    ///
    /// The inverse of those two readings and nothing more, so a grid that has been
    /// written down and read back is the *same* grid rather than one reconstructed by
    /// somebody's arithmetic. A caller persisting a table (the event log, ADR-0027)
    /// would otherwise have to infer `max_decimals` from the increment's scale, which
    /// is a second definition of this type's shape living in a crate that does not own
    /// it — and the day the shape gains a third number, the copy keeps compiling.
    pub fn from_parts(increment: Decimal, sig_figs: Option<u32>) -> Result<Self, SpecError> {
        if increment < Decimal::ZERO {
            return Err(SpecError::BadIncrement { inc: increment });
        }
        if let Some(s) = sig_figs {
            if s == 0 || s > MAX_SCALE {
                return Err(SpecError::OutOfRange {
                    what: "sig_figs",
                    value: s,
                    max: MAX_SCALE,
                });
            }
            // A significant-figure cap with no floor under it is not a state this type
            // can hold: `tick_at` reads a zero increment as "no grid at all" and never
            // consults `sig_figs`, so accepting the pair would silently drop the cap.
            if increment.is_zero() {
                return Err(SpecError::BadIncrement { inc: increment });
            }
        }
        Ok(Self {
            increment,
            sig_figs,
        })
    }

    /// The floor of the grid — zero when there is none.
    ///
    /// Named `min_increment` rather than `increment` because [`increment`](Self::increment)
    /// is the constructor; one name for the two would be a compile error, and picking
    /// the shorter one for the getter would make the constructor the odd one out.
    pub fn min_increment(&self) -> Decimal {
        self.increment
    }

    /// The significant-figure cap, when this venue has one.
    pub fn sig_figs(&self) -> Option<u32> {
        self.sig_figs
    }

    /// The grid spacing *at this price*. Zero when there is no grid.
    pub fn tick_at(&self, px: Decimal) -> Decimal {
        if self.increment.is_zero() {
            return Decimal::ZERO;
        }
        match self.sig_figs {
            None => self.increment,
            Some(s) => self.increment.max(sig_quantum(s, px)),
        }
    }

    /// Move `px` onto the grid in the direction that preserves `intent`.
    ///
    /// A price already on the grid is returned **unchanged, scale included**. That is
    /// not cosmetic: the venue's own touch is the common input, and it must come back
    /// byte-identical or `WorkingOrder::is()` stops matching the venue's echo — at
    /// which point every pass cancel/replaces an order that was already correct and
    /// forfeits its queue priority for nothing.
    pub fn quantize(&self, px: Decimal, side: Side, intent: PriceIntent) -> Decimal {
        let tick = self.tick_at(px);
        // A non-positive price has no floor and no ceiling worth naming; `is_valid`
        // refuses it and the caller has to as well, so quantizing it would only invent
        // a number for an order that must not be sent.
        if tick.is_zero() || !px.is_sign_positive() || px.is_zero() {
            return px;
        }
        let Some(r) = px.checked_rem(tick) else {
            // Unreachable for a grid built through the constructors above (they refuse
            // a zero increment). Returning `px` untouched leaves `check` as the single
            // place an impossible price is refused, rather than inventing a second one.
            return px;
        };
        if r.is_zero() {
            return px;
        }
        let down = px - r; // px > 0 so r >= 0: this is the floor
        let up = down + tick;
        let out = match (side, intent) {
            (Side::Buy, PriceIntent::Passive) => down,
            (Side::Buy, PriceIntent::Marketable) => up,
            (Side::Sell, PriceIntent::Passive) => up,
            (Side::Sell, PriceIntent::Marketable) => down,
        };
        // Only on the path that actually moved: `1234.50` becomes `1234.5`, while an
        // untouched `49999.0` above keeps the scale the venue printed.
        out.normalize()
    }

    /// Whether the venue would accept `px` as-is.
    pub fn is_valid(&self, px: Decimal) -> bool {
        if px <= Decimal::ZERO {
            return false;
        }
        let tick = self.tick_at(px);
        if tick.is_zero() {
            return true;
        }
        matches!(px.checked_rem(tick), Some(r) if r.is_zero())
    }
}

impl SizeGrid {
    /// A fixed lot. `inc` must be positive.
    pub fn step(inc: Decimal) -> Result<Self, SpecError> {
        if inc <= Decimal::ZERO {
            return Err(SpecError::BadIncrement { inc });
        }
        Ok(Self { increment: inc })
    }

    /// `szDecimals: n` → a lot of `10^-n`.
    pub fn decimals(n: u32) -> Result<Self, SpecError> {
        if n > MAX_SCALE {
            return Err(SpecError::OutOfRange {
                what: "sz_decimals",
                value: n,
                max: MAX_SCALE,
            });
        }
        Ok(Self {
            increment: Decimal::new(1, n),
        })
    }

    /// A venue with no lot at all. Zero, for the same reason [`PriceGrid::unconstrained`]
    /// is zero.
    pub fn unconstrained() -> Self {
        Self {
            increment: Decimal::ZERO,
        }
    }

    pub fn increment(&self) -> Decimal {
        self.increment
    }

    /// Truncate toward zero. Signed in, signed out — a negative target quantizes to a
    /// smaller magnitude, never a larger one, so a close can never overshoot the
    /// position it is closing.
    pub fn quantize(&self, qty: Decimal) -> Decimal {
        if self.increment.is_zero() {
            return qty;
        }
        let Some(r) = qty.checked_rem(self.increment) else {
            return qty;
        };
        if r.is_zero() {
            return qty;
        }
        // `Decimal`'s remainder carries the *dividend's* sign, so this truncates toward
        // zero for both: `-7.5 % 1 = -0.5`, and `-7.5 - (-0.5) = -7`.
        (qty - r).normalize()
    }

    pub fn is_valid(&self, qty: Decimal) -> bool {
        self.increment.is_zero()
            || matches!(qty.checked_rem(self.increment), Some(r) if r.is_zero())
    }
}

impl InstrumentSpec {
    /// An instrument with no declared grid — a simulated venue, or a test that is not
    /// about rounding.
    pub fn unconstrained(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            price: PriceGrid::unconstrained(),
            size: SizeGrid::unconstrained(),
            min_notional: None,
        }
    }

    /// The refusal: exactly the rules [`PriceGrid::quantize`] and
    /// [`SizeGrid::quantize`] obey, so the thing that decides and the thing that
    /// refuses cannot disagree.
    ///
    /// `min_notional` is **not** applied to a reduce-only order, matching the planner
    /// exactly: we never refuse to de-risk. Any drift between these two is a session
    /// that plans orders its own encoder rejects, which in a log is indistinguishable
    /// from a venue outage.
    pub fn check(&self, req: &OrderRequest) -> Result<(), PrecisionError> {
        if let Some(px) = req.price {
            if !self.price.is_valid(px) {
                return Err(PrecisionError::Tick {
                    px,
                    tick: self.price.tick_at(px),
                });
            }
        }
        if !self.size.is_valid(req.qty) {
            return Err(PrecisionError::Lot {
                qty: req.qty,
                lot: self.size.increment(),
            });
        }
        if let (Some(min), Some(px)) = (self.min_notional, req.price) {
            let notional = req.qty * px;
            if !req.reduce_only && notional < min {
                return Err(PrecisionError::BelowMinNotional { notional, min });
            }
        }
        Ok(())
    }
}

impl InstrumentTable {
    /// A venue that *has* grids: a symbol not in the map is [`Precision::Unknown`].
    pub fn new() -> Self {
        Self::default()
    }

    /// A venue with no grids at all: every lookup is [`Precision::Unconstrained`].
    pub fn unconstrained() -> Self {
        Self {
            by_symbol: HashMap::new(),
            unconstrained: true,
        }
    }

    pub fn insert(&mut self, spec: InstrumentSpec) {
        self.by_symbol.insert(spec.symbol_id, spec);
    }

    pub fn get(&self, s: SymbolId) -> Option<&InstrumentSpec> {
        self.by_symbol.get(&s)
    }

    pub fn contains(&self, s: SymbolId) -> bool {
        self.by_symbol.contains_key(&s)
    }

    /// [`Precision::Known`] if present; otherwise whichever of the two "not here"
    /// answers this table was built to give.
    pub fn precision(&self, s: SymbolId) -> Precision<'_> {
        match self.by_symbol.get(&s) {
            Some(spec) => Precision::Known(spec),
            None if self.unconstrained => Precision::Unconstrained,
            None => Precision::Unknown,
        }
    }

    /// Every spec in the table, **ordered by symbol id**.
    ///
    /// Sorted, and allocating, because the map is a `HashMap`: its iteration order is
    /// randomized per process, and the caller this exists for writes the table into a
    /// log that two captures of one session are compared byte for byte (ADR-0027).
    /// Handing out the raw iteration order would make a recording differ from itself
    /// between runs, in the artifact whose whole job is to be reproducible.
    pub fn specs(&self) -> Vec<InstrumentSpec> {
        let mut out: Vec<InstrumentSpec> = self.by_symbol.values().copied().collect();
        out.sort_by_key(|s| s.symbol_id.get());
        out
    }

    /// Which of the two "not here" answers this table gives — see [`Precision`].
    ///
    /// Persisting a table means persisting this too. A table written down without it
    /// reads back as [`Precision::Unknown`] for every absent symbol, which refuses
    /// every order that adds exposure; written down *only* as its entries, a
    /// simulator's table would come back as a live venue's.
    pub fn is_unconstrained(&self) -> bool {
        self.unconstrained
    }

    pub fn len(&self) -> usize {
        self.by_symbol.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_symbol.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Tif};
    use std::str::FromStr;

    /// A literal, exactly. `rust_decimal_macros` is not a dependency of this crate and
    /// this module exists partly to keep it that way.
    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).expect("test literal")
    }

    /// The Hyperliquid perp shape: `6 - szDecimals` decimal places, 5 significant
    /// figures, integers exempt.
    fn perp(sz_decimals: u32) -> PriceGrid {
        PriceGrid::decimals_with_sig_figs(6 - sz_decimals, 5).expect("a real venue's numbers")
    }

    fn spec(sz_decimals: u32, min_notional: Option<Decimal>) -> InstrumentSpec {
        InstrumentSpec {
            symbol_id: SymbolId::new(3),
            price: perp(sz_decimals),
            size: SizeGrid::decimals(sz_decimals).unwrap(),
            min_notional,
        }
    }

    fn order(qty: Decimal, px: Decimal) -> OrderRequest {
        OrderRequest::limit(
            SymbolId::new(3),
            Side::Buy,
            qty,
            px,
            Tif::Gtc,
            Cloid::new(1),
        )
    }

    #[test]
    fn the_venues_own_published_price_examples_all_validate() {
        // Straight off the venue's tick-and-lot page. Getting any of these wrong is a
        // `tickRejected`, which from inside the process reads like a signing bug.
        assert!(perp(0).is_valid(d("1234.5")), "5 significant figures");
        assert!(!perp(0).is_valid(d("1234.56")), "6 significant figures");
        assert!(perp(0).is_valid(d("0.001234")), "6 decimal places");
        assert!(!perp(0).is_valid(d("0.0012345")), "7 decimal places");
        // szDecimals = 1 buys one fewer decimal place, and that is what binds here —
        // both of these are inside the significant-figure cap.
        assert!(perp(1).is_valid(d("0.01234")));
        assert!(!perp(1).is_valid(d("0.012345")));

        // Sizes: exactly szDecimals decimals.
        let lot = SizeGrid::decimals(3).unwrap();
        assert!(lot.is_valid(d("1.001")));
        assert!(!lot.is_valid(d("1.0001")));
    }

    #[test]
    fn an_integer_price_is_legal_past_the_significant_figure_cap() {
        // The documented exception, and the one that keeps BTC tradable at all: 108234
        // is six significant figures and the venue accepts it because it is a whole
        // number. Without the clamp in `sig_quantum` every BTC order would be refused.
        assert!(perp(5).is_valid(d("108234")));
        assert_eq!(perp(5).tick_at(d("108234")), Decimal::ONE);
        assert!(perp(0).is_valid(d("1234567")));
        // And the rule still binds one digit below the decimal point.
        assert!(!perp(5).is_valid(d("108775.17")));
    }

    #[test]
    fn the_tick_widens_with_the_price_instead_of_being_one_fixed_number() {
        // The reason `PriceGrid` is not a single `tick`: on one instrument the spacing
        // is 0.1 at four figures and 1 at six. A fixed tick is wrong at one end or the
        // other, and wrong quietly.
        let g = perp(5);
        assert_eq!(g.tick_at(d("1234.5")), d("0.1"));
        assert_eq!(g.tick_at(d("108234")), Decimal::ONE);
        assert_eq!(perp(0).tick_at(d("0.001234")), d("0.000001"));
    }

    #[test]
    fn a_price_below_one_keeps_the_digits_the_decimal_cap_allows_because_leading_zeros_are_not_significant(
    ) {
        // The bug the live test's private helper had: it counted "integer digits" as
        // zero for every sub-$1 price and capped at five decimals where the venue
        // allows six — throwing a digit away on every cheap asset, which on a passive
        // quote is the whole edge.
        let g = perp(0);
        assert!(g.is_valid(d("0.003429")), "six decimals, four significant");
        assert_eq!(
            g.quantize(d("0.0034285714"), Side::Buy, PriceIntent::Marketable),
            d("0.003429")
        );
        assert_eq!(
            g.quantize(d("0.0034285714"), Side::Buy, PriceIntent::Passive),
            d("0.003428")
        );
    }

    #[test]
    fn significant_figures_are_counted_without_a_float() {
        // `(100000f64).log10()` is 4.999…, so a float count says five digits where
        // there are six — and it says it at exactly the magnitude a five-figure rule
        // bites. Every one of these is a magnitude boundary.
        assert_eq!(decimal_exponent(d("100000")), 6);
        assert_eq!(decimal_exponent(d("99999.9")), 5);
        assert_eq!(decimal_exponent(d("9.99999")), 1);
        assert_eq!(
            decimal_exponent(d("100.00")),
            3,
            "trailing zeros are not digits"
        );
        assert_eq!(decimal_exponent(d("0.001")), -2);
        assert_eq!(decimal_exponent(Decimal::ZERO), 0);

        // And the decade crossing lands on a value every coarser tick also admits.
        let g = perp(0);
        let up = g.quantize(d("99999.9"), Side::Buy, PriceIntent::Marketable);
        assert_eq!(up, d("100000"));
        assert!(g.is_valid(up), "rounding up must not leave the grid");
        let up = g.quantize(d("9.99999"), Side::Buy, PriceIntent::Marketable);
        assert_eq!(up, Decimal::TEN, "9.9999|9 ceils to 10, normalized");
        assert!(g.is_valid(up));
    }

    #[test]
    fn a_quantized_price_is_always_a_price_the_validator_accepts() {
        // The property that makes the encoder's refusal a test of the planner's logic
        // instead of a test of two implementations: if these two ever disagree, the
        // session plans orders its own encoder rejects and the log reads like a venue
        // outage.
        let grids = [
            perp(0),
            perp(1),
            perp(4),
            perp(5),
            perp(6),
            PriceGrid::increment(d("0.25")).unwrap(),
            PriceGrid::increment(d("0.01")).unwrap(),
        ];
        let prices = [
            "0.0000001234",
            "0.0034285714",
            "0.012345",
            "1.234567",
            "9.99999",
            "99.999",
            "1234.56",
            "2999.47",
            "49757.96",
            "99999.9",
            "108775.17",
            "108234",
        ];
        for g in grids {
            for p in prices {
                let px = d(p);
                for side in [Side::Buy, Side::Sell] {
                    for intent in [PriceIntent::Passive, PriceIntent::Marketable] {
                        let q = g.quantize(px, side, intent);
                        // Zero is the one legal output that is not a legal price: a
                        // price below one tick has no representation on this grid at
                        // all, and flooring it is how the planner finds that out
                        // (`NoOrder::PriceNotRepresentable`). Anything else must be a
                        // price the validator — and therefore the venue — accepts.
                        assert!(
                            g.is_valid(q) || (q.is_zero() && px < g.tick_at(px)),
                            "quantize({px}, {side:?}, {intent:?}) = {q} is off its own grid \
                             (tick {})",
                            g.tick_at(q)
                        );
                        // And it never moves further than one tick, which is what bounds
                        // the price a marketable order pays for being rounded.
                        let moved = (q - px).abs();
                        assert!(
                            moved < g.tick_at(px).max(g.tick_at(q)) || moved.is_zero(),
                            "{px} -> {q} moved {moved}, more than one tick"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_price_already_on_the_grid_is_returned_unchanged_including_its_scale() {
        // The venue's own touch is the common input. If quantizing re-scaled it, the
        // planner's price would stop equalling the price the venue echoes back on
        // `orderUpdates`, `WorkingOrder::is()` would never match, and every pass would
        // cancel/replace an order that was already correct — forfeiting queue priority
        // on a change that never happened.
        let g = perp(5);
        let px = d("49999.0");
        assert_eq!(px.scale(), 1, "the fixture has to carry the trailing zero");
        for side in [Side::Buy, Side::Sell] {
            for intent in [PriceIntent::Passive, PriceIntent::Marketable] {
                let q = g.quantize(px, side, intent);
                assert_eq!(q, px);
                assert_eq!(q.scale(), 1, "the scale travels with the value");
            }
        }
    }

    #[test]
    fn a_fixed_increment_grid_needs_no_significant_figure_rule() {
        // The CEX shape. One field set, one tick everywhere, no `match` anywhere for a
        // second venue's arm.
        let g = PriceGrid::increment(d("0.25")).unwrap();
        assert_eq!(g.tick_at(d("1")), d("0.25"));
        assert_eq!(g.tick_at(d("1234567.89")), d("0.25"));
        assert!(
            g.is_valid(d("1234567.75")),
            "six significant figures is fine"
        );
        assert!(!g.is_valid(d("1234567.80")));
        assert_eq!(
            g.quantize(d("100.10"), Side::Buy, PriceIntent::Passive),
            d("100")
        );
        assert_eq!(
            g.quantize(d("100.10"), Side::Buy, PriceIntent::Marketable),
            d("100.25")
        );
    }

    #[test]
    fn a_size_is_truncated_toward_zero_so_a_close_can_never_overshoot_the_position() {
        // Rounding a size *up* overshoots the target — the same class of error as
        // sending the target instead of the delta. On a reduce it exceeds the position
        // and a flatten overshoots straight into an opposite-side one.
        let lot = SizeGrid::decimals(3).unwrap();
        assert_eq!(lot.quantize(d("1.0009")), d("1"));
        assert_eq!(lot.quantize(d("-1.0009")), d("-1"));
        assert_eq!(lot.quantize(d("-7.5")), d("-7.5"), "already on the lot");
        assert_eq!(SizeGrid::decimals(0).unwrap().quantize(d("-7.5")), d("-7"));
        // A residue finer than the lot goes to zero rather than to one lot.
        assert_eq!(lot.quantize(d("0.0004")), Decimal::ZERO);
        assert_eq!(lot.quantize(d("-0.0004")), Decimal::ZERO);
    }

    #[test]
    fn an_unconstrained_grid_is_the_identity_instead_of_a_division_by_a_zero_tick() {
        let g = PriceGrid::unconstrained();
        assert_eq!(g.tick_at(d("1234.5678")), Decimal::ZERO);
        assert_eq!(
            g.quantize(d("1234.5678"), Side::Buy, PriceIntent::Marketable),
            d("1234.5678")
        );
        assert!(g.is_valid(d("1234.5678")));
        assert!(
            !g.is_valid(Decimal::ZERO),
            "zero is not a price on any grid"
        );

        let l = SizeGrid::unconstrained();
        assert_eq!(l.quantize(d("0.000000001")), d("0.000000001"));
        assert!(l.is_valid(d("0.000000001")));
    }

    #[test]
    fn a_price_finer_than_the_whole_tick_floors_to_zero_rather_than_to_a_tick_it_is_not() {
        // The one output of `quantize` that is not a price. It is deliberate: an
        // instrument whose grid is coarser than the price being asked for cannot
        // express that order at all, and rounding *up* to one tick would send an order
        // at some multiple of the intended price. The planner reads the zero as
        // `PriceNotRepresentable` and sends nothing.
        let g = perp(5); // tick 0.1 down here
        assert_eq!(
            g.quantize(d("0.05"), Side::Buy, PriceIntent::Passive),
            Decimal::ZERO
        );
        assert!(!g.is_valid(Decimal::ZERO));
        // The marketable direction still reaches the first real tick.
        assert_eq!(
            g.quantize(d("0.05"), Side::Buy, PriceIntent::Marketable),
            d("0.1")
        );
    }

    #[test]
    fn a_nonsense_sz_decimals_is_refused_at_construction_rather_than_producing_a_grid_of_zero() {
        // These come out of a venue's JSON. A 200-decimal lot must be an error the
        // decoder reports, not a panic in a live session and not a silent grid of zero
        // that admits every price.
        assert!(matches!(
            SizeGrid::decimals(200),
            Err(SpecError::OutOfRange {
                what: "sz_decimals",
                ..
            })
        ));
        assert!(matches!(
            PriceGrid::decimals_with_sig_figs(200, 5),
            Err(SpecError::OutOfRange {
                what: "max_decimals",
                ..
            })
        ));
        assert!(matches!(
            PriceGrid::decimals_with_sig_figs(6, 0),
            Err(SpecError::OutOfRange {
                what: "sig_figs",
                ..
            })
        ));
        assert!(matches!(
            PriceGrid::increment(Decimal::ZERO),
            Err(SpecError::BadIncrement { .. })
        ));
        assert!(matches!(
            SizeGrid::step(d("-1")),
            Err(SpecError::BadIncrement { .. })
        ));
    }

    #[test]
    fn check_refuses_exactly_what_quantize_would_have_moved() {
        let s = spec(5, None);
        assert!(s.check(&order(d("0.00019"), d("108234"))).is_ok());
        assert!(matches!(
            s.check(&order(d("0.00019"), d("108234.567"))),
            Err(PrecisionError::Tick { .. })
        ));
        assert!(matches!(
            s.check(&order(d("0.000191"), d("108234"))),
            Err(PrecisionError::Lot { .. })
        ));
    }

    #[test]
    fn a_reduce_only_order_under_the_minimum_notional_passes_check_because_the_planner_lets_it_through(
    ) {
        // The anti-drift test. The planner sends a sub-minimum *close* on purpose —
        // refusing to de-risk on our own opinion strands a position — so an encoder
        // that refused it would turn a deliberate decision into a refusal nobody chose,
        // and the log would show a venue rejection that never happened.
        let s = spec(5, Some(Decimal::TEN));
        let mut open = order(d("0.00001"), d("100000"));
        assert_eq!(open.qty * open.price.unwrap(), Decimal::ONE);
        assert!(matches!(
            s.check(&open),
            Err(PrecisionError::BelowMinNotional { .. })
        ));
        open.reduce_only = true;
        assert!(s.check(&open).is_ok(), "a close is never refused for size");
    }

    #[test]
    fn a_table_says_unknown_or_unconstrained_and_never_confuses_the_two() {
        // A backtest that read an empty table as "no rules" would be silently more
        // permissive than the live session it claims to reproduce; a live session that
        // read a missing symbol as "no rules" would send it unrounded.
        let mut live = InstrumentTable::new();
        assert!(live.is_empty());
        assert_eq!(live.precision(SymbolId::new(3)), Precision::Unknown);
        let s = spec(5, Some(Decimal::TEN));
        live.insert(s);
        assert_eq!(live.len(), 1);
        assert!(live.contains(SymbolId::new(3)));
        assert_eq!(live.precision(SymbolId::new(3)), Precision::Known(&s));
        assert_eq!(live.precision(SymbolId::new(4)), Precision::Unknown);

        let sim = InstrumentTable::unconstrained();
        assert_eq!(sim.precision(SymbolId::new(3)), Precision::Unconstrained);
    }
}
