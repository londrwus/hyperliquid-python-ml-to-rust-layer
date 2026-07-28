//! [`Planner`] — the translation `docs/06-strategy-contract.md` promises: "the
//! strategy declares what position it WANTS; the Rust execution engine decides HOW
//! to get there".
//!
//! An accepted [`Signal`] plus the position we actually hold becomes a list of
//! [`OrderRequest`]s and the [`CancelId`]s they supersede. Everything the planner
//! does is a pure function of its inputs — no clock, no I/O, no async — so the
//! same call replays identically in a backtest ([07](../../../docs/07-parity-and-testing.md)).
//!
//! Five decisions carry the weight, each with a test named after what it prevents:
//!
//! 1. **The order is the delta, never the target.** A target-position signal says
//!    "be long 3". Sending an order for 3 while already long 2 ends up long 5. The
//!    planner subtracts, always.
//! 2. **Urgency is an explicit table** ([`URGENCY_TABLE`]), not a heuristic, so the
//!    same urgency always produces the same TIF and the same price anchor.
//! 3. **`price_band` is a hard wall, and a wall that makes the order pointless
//!    suppresses it.** A marketable order clamped to a price that cannot cross is
//!    a guaranteed no-op that still burns a nonce and a rate-limit credit, and
//!    still logs as if we tried.
//! 4. **`cloid` is derived from the signal's identity**, so re-submitting after a
//!    timeout is idempotent at the venue instead of doubling the position.
//! 5. **A superseded working order is cancelled, not left resting** — except when
//!    it is already the order we would place, in which case replacing it would
//!    forfeit queue priority for nothing. "Already the order" means every field the
//!    caller can *vouch* for matches (an unknown [`WorkingOrder::tif`] never does),
//!    with a size difference inside [`PlannerConfig::noop_band_bps`] forgiven, and
//!    only while the order is younger than [`PlannerConfig::max_order_age_ms`] — the
//!    exception preserves queue position, and an exception with no bound on it
//!    preserves orders nobody currently intends.
//! 6. **Prices and sizes are put on the instrument's own grid before they leave**
//!    (ADR-0025). The planner is the only place that knows the *intent*, so it is the
//!    only place that can round in the direction which preserves it — and the only
//!    place that can decide a size rounding to zero means "no order" rather than a
//!    zero-size order. The encoder then refuses anything off-grid, so the two cannot
//!    disagree without one of them saying so.

use axon_contracts::Signal;
use axon_core::{Bbo, Cloid, Decimal, Nanos, OrderType, Side, SymbolId, Tif};
use axon_providers::{CancelId, OrderRequest, Precision, PriceIntent};
use thiserror::Error;

use crate::fixed::fixed_to_decimal;

/// Basis points denominator. Exact in `Decimal`; there is no float on this path.
const BPS: i64 = 10_000;

/// Event-time nanoseconds per millisecond. Config is in ms; every clock here is ns.
const NS_PER_MS: Nanos = 1_000_000;

/// Which side of the spread a level prices against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// The passive touch: buy at the bid, sell at the ask. Adds liquidity.
    NearTouch,
    /// The aggressive touch: buy at the ask, sell at the bid. Crosses the spread.
    FarTouch,
}

/// One row of [`URGENCY_TABLE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UrgencyRule {
    pub tif: Tif,
    pub anchor: Anchor,
    /// Whether [`PlannerConfig::taker_slippage_bps`] is added beyond the anchor.
    pub slipped: bool,
}

impl UrgencyRule {
    /// True when the level's whole point is to trade *now*. These are the levels
    /// where a price band that cannot cross means "do not send".
    #[inline]
    pub fn is_marketable(&self) -> bool {
        matches!(self.anchor, Anchor::FarTouch)
    }
}

/// `urgency` → time-in-force and price anchor. Four levels; anything above the
/// last one saturates into it.
///
/// The ordering is not "more aggressive price" at every step — level 0 and level 1
/// price identically. What increases monotonically is **what we are willing to give
/// up to get the position on**:
///
/// | urgency | TIF        | anchor     | gives up             |
/// |---------|------------|------------|----------------------|
/// | 0       | `PostOnly` | near touch | fill certainty       |
/// | 1       | `Gtc`      | near touch | maker-only guarantee |
/// | 2       | `Gtc`      | far touch  | the spread           |
/// | 3+      | `Ioc`      | far touch  | the spread + slippage, and any unfilled remainder |
///
/// Why these four, in Hyperliquid's terms (`docs/04-provider-abstraction.md`):
///
/// - **0 is post-only because in-block priority is `cancel > post-only > GTC > IOC`.**
///   A passive quote submitted in the same block as somebody else's taker is
///   processed first, so level 0 is not merely "cheap", it is structurally ahead in
///   the queue. Post-only also *cannot* pay a taker fee, which is the guarantee a
///   market-making strategy is actually asking for.
/// - **1 is the same price as GTC**, and that difference matters: post-only is
///   **rejected** if the book moved under us between our snapshot and the venue's
///   block, so a strategy that must have an order working cannot use level 0. Level
///   1 buys certainty of *placement*, not of fill.
/// - **2 crosses the spread but stays a limit.** A partial fill leaves the
///   remainder resting at a price we chose, so slippage is bounded by one spread
///   instead of by the depth of the book.
/// - **3 is IOC, because Hyperliquid has no native market order** — a market order
///   there *is* an IOC limit priced through the book. Leaving no residue is the
///   point: an urgent exit that half-fills and rests is an unmanaged position.
pub const URGENCY_TABLE: [UrgencyRule; 4] = [
    UrgencyRule {
        tif: Tif::PostOnly,
        anchor: Anchor::NearTouch,
        slipped: false,
    },
    UrgencyRule {
        tif: Tif::Gtc,
        anchor: Anchor::NearTouch,
        slipped: false,
    },
    UrgencyRule {
        tif: Tif::Gtc,
        anchor: Anchor::FarTouch,
        slipped: false,
    },
    UrgencyRule {
        tif: Tif::Ioc,
        anchor: Anchor::FarTouch,
        slipped: true,
    },
];

/// The rule for `urgency`, saturating at the most aggressive level.
///
/// Saturating rather than rejecting: `urgency` is a `u8` and a strategy writing
/// `255` means "as fast as possible". Refusing the record would drop precisely the
/// signal you least want dropped, and clamping to *passive* would answer an urgent
/// exit with a resting quote.
#[inline]
pub fn urgency_rule(urgency: u8) -> UrgencyRule {
    let idx = (urgency as usize).min(URGENCY_TABLE.len() - 1);
    URGENCY_TABLE[idx]
}

/// The top of book the planner prices against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quote {
    pub bid_px: Decimal,
    pub ask_px: Decimal,
}

impl Quote {
    pub fn new(bid_px: Decimal, ask_px: Decimal) -> Self {
        Self { bid_px, ask_px }
    }

    pub fn from_bbo(bbo: &Bbo) -> Self {
        Self {
            bid_px: bbo.bid_px,
            ask_px: bbo.ask_px,
        }
    }

    /// A book we are willing to price against: both sides present and not crossed.
    ///
    /// A locked or crossed top of book means the feed is stale, mid-update, or
    /// one-sided. Pricing off it produces an order that is aggressive or passive by
    /// accident, so the planner treats it as "no quote" rather than guessing.
    pub fn is_usable(&self) -> bool {
        self.bid_px > Decimal::ZERO && self.ask_px > self.bid_px
    }

    pub fn mid(&self) -> Decimal {
        (self.bid_px + self.ask_px) / Decimal::TWO
    }
}

/// One of our orders still working at the venue, as far as the caller knows.
///
/// A deliberately small view rather than `axon_execution::TrackedOrder`: the
/// strategy adapter sits *upstream* of the execution engine (`docs/01`), and taking
/// a dependency the other way would invert that edge. The runtime projects its
/// tracker state into this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkingOrder {
    pub cloid: Cloid,
    pub symbol_id: SymbolId,
    pub side: Side,
    pub price: Option<Decimal>,
    /// Size still unfilled.
    pub remaining_qty: Decimal,
    /// The order's time-in-force, or `None` when nobody can vouch for it.
    ///
    /// A venue's order list carries no time-in-force and no reduce-only flag, so an
    /// order a restarted process *adopted* has both only as far as somebody is
    /// willing to guess — and a guess that happened to match the order we were about
    /// to place would leave a **reduce-only** quote resting in place of the one meant
    /// to open the position. The strategy then sits flat with an order working that
    /// cannot ever move it. `None` is what makes [`Self::is`] able to refuse instead
    /// of comparing against an invention (ADR-0020 §7).
    ///
    /// This is not merely `placed_by_us` renamed. That flag could only ever say *no*
    /// — every adopted order was cancelled and replaced, so a restarted session lost
    /// the queue position of every order its predecessor left. A venue that does
    /// report the field, or a tracker that survives the restart, can now say *yes*.
    pub tif: Option<Tif>,
    /// Whether the order may only shrink the position, or `None` when unknown. Same
    /// provenance rule as [`Self::tif`], and the more dangerous of the two to guess.
    pub reduce_only: Option<bool>,
    /// Event time at which this order started working.
    ///
    /// Event time, like everything else: it is subtracted from
    /// [`PlanContext::now`], never from a wall clock, so a replayed session ages an
    /// order exactly as the live one did. A caller that cannot date an order (an
    /// adopted one — the venue does not say when it was placed) passes the first
    /// moment it saw it, which is a lower bound and therefore errs toward keeping
    /// the order rather than pulling one early.
    pub placed_ts: Nanos,
}

impl WorkingOrder {
    /// True when this order is already, exactly, the order `req` asks for.
    ///
    /// An unknown `tif` or `reduce_only` never matches. That is the whole point of
    /// their being `Option`s: the comparison is what decides whether to leave a quote
    /// resting, so a field nobody can vouch for has to fail it.
    fn is(&self, req: &OrderRequest) -> bool {
        self.side == req.side
            && self.tif == Some(req.tif)
            && self.price == req.price
            && self.remaining_qty == req.qty
            && self.reduce_only == Some(req.reduce_only)
    }

    /// [`Self::is`], with a size difference of up to `band_bps` of the **resting**
    /// size forgiven. See [`PlannerConfig::noop_band_bps`] for why the band is
    /// measured against that and not against the target.
    ///
    /// Everything except the size still has to match exactly. A price that moved is
    /// never forgiven: the order would then be resting at a price we would no longer
    /// choose, which is a stale quote, and a stale quote is what somebody else's
    /// taker is looking for — the band is about not paying queue position for a
    /// change too small to matter, not about tolerating a wrong price.
    fn is_within_band(&self, req: &OrderRequest, band_bps: u32) -> bool {
        if self.is(req) {
            return true;
        }
        // A reduce-only order is never banded. "Get me out" that lands 20 bps short
        // of flat has not got us out, and the residue is a position nobody decided to
        // hold; the band trades exposure error for queue position, and there is no
        // queue position worth a position we are trying not to have.
        if band_bps == 0 || req.reduce_only {
            return false;
        }
        self.side == req.side
            && self.tif == Some(req.tif)
            && self.price == req.price
            && self.reduce_only == Some(req.reduce_only)
            && qty_within_band(self.remaining_qty, req.qty, band_bps)
    }
}

/// Whether `want` is within `band_bps` of the size already `resting`.
///
/// Multiplied out rather than divided: `Decimal` division by zero panics, and a
/// working order can legitimately report a zero remaining size for the moment
/// between its last fill and its terminal frame. Zero resting means there is no
/// queue position to preserve, so it is never a match.
fn qty_within_band(resting: Decimal, want: Decimal, band_bps: u32) -> bool {
    if resting <= Decimal::ZERO {
        return false;
    }
    (resting - want).abs() * Decimal::from(BPS) <= Decimal::from(i64::from(band_bps)) * resting
}

/// Why the planner produced no order. Present whenever [`Plan::orders`] is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NoOrder {
    #[error("already at the target position")]
    AlreadyAtTarget,

    #[error("the working order already is this order; replacing it would forfeit queue priority")]
    AlreadyWorking,

    /// The working order differs from the one we would place by less than
    /// [`PlannerConfig::noop_band_bps`], so it was left alone.
    ///
    /// Its own variant rather than [`AlreadyWorking`](Self::AlreadyWorking) because it
    /// is the only no-order outcome that means *we are deliberately carrying a small
    /// position error* — bounded by the band, and the operator's to widen or close.
    /// Folded into "already working" it would be indistinguishable from a strategy
    /// that had simply not changed its mind, and the knob's effect would be invisible.
    #[error(
        "the working order is within the no-op band: {resting} resting against {wanted} wanted"
    )]
    WithinNoOpBand { resting: Decimal, wanted: Decimal },

    #[error("delta {delta} is below the minimum order size {min}")]
    BelowMinQty { delta: Decimal, min: Decimal },

    #[error("no usable top of book to price against")]
    NoQuote,

    #[error("price_band {band} is not a positive price")]
    BadPriceBand { band: Decimal },

    #[error("price band clamps the limit to {limit}, which cannot cross the {touch} touch")]
    PriceBandUnfillable { limit: Decimal, touch: Decimal },

    #[error("reduce-only signal would grow the position")]
    ReduceOnlyWouldGrow,

    /// The delta is finer than the venue will accept, so there is no order to place.
    ///
    /// A zero-size order is a guaranteed rejection that still spends a nonce, a
    /// rate-limit credit and a log line saying we tried.
    ///
    /// Reached from two directions. The rare one is a *held* position left off the grid
    /// by a mid-session re-precision. The common one is a target smaller than a single
    /// lot asked for **from flat** — routine wherever `szDecimals` is 0 — which the lot
    /// erases entirely, leaving a zero delta that must not be reported as
    /// [`AlreadyAtTarget`](Self::AlreadyAtTarget): that is the healthy variant, it is
    /// counted nowhere, and a strategy being refused on every signal would read exactly
    /// like one that chose to be flat.
    #[error("size {delta} is finer than the instrument's lot {lot}")]
    BelowLotSize { delta: Decimal, lot: Decimal },

    /// The order is worth less than the venue's minimum. Only ever refused when it
    /// would *add* exposure — see [`Planner::plan`].
    #[error("notional {notional} is below the venue minimum {min}")]
    BelowMinNotional { notional: Decimal, min: Decimal },

    /// The instrument's grid is coarser than the price, so there is no legal price to
    /// send: rounding away leaves zero, and rounding toward the market would send an
    /// order at some multiple of the one asked for.
    #[error("price {price} does not survive this instrument's tick {tick} as a positive price")]
    PriceNotRepresentable { price: Decimal, tick: Decimal },

    /// A marketable order the *grid* — not the band — moved out of the market.
    ///
    /// Separate from [`PriceBandUnfillable`](Self::PriceBandUnfillable) because the
    /// operator's fix differs: one says "your band is too tight", this one says "this
    /// instrument's grid is coarser than your band". A shared variant would send
    /// somebody to change the wrong number.
    #[error(
        "the instrument's grid moves the limit to {limit}, which cannot cross the {touch} touch"
    )]
    RoundedUnfillable { limit: Decimal, touch: Decimal },

    /// We do not know this instrument's grid, and the order is not the one exemption
    /// (a full flatten at a price the venue itself printed).
    #[error("no precision is known for {symbol:?}")]
    UnknownPrecision { symbol: SymbolId },
}

/// What the planner wants done with this instrument, right now.
///
/// `cancels` are ordered **before** `orders` and the caller must submit them in
/// that order. On Hyperliquid the venue enforces the same ordering within a block
/// (`cancel > post-only > GTC > IOC`, `docs/04`), so even a same-block race cannot
/// leave the old and the new order both live. On a venue without that guarantee the
/// caller must await the cancel acks first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    pub cancels: Vec<CancelId>,
    pub orders: Vec<OrderRequest>,
    /// Why `orders` is empty. `None` exactly when an order was produced.
    pub no_order: Option<NoOrder>,
}

impl Plan {
    /// Nothing to send at all.
    pub fn is_noop(&self) -> bool {
        self.cancels.is_empty() && self.orders.is_empty()
    }
}

/// Tunables the strategy does not get to set — they belong to the process that
/// answers for the fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerConfig {
    /// How far *through* the far touch an urgency-3 IOC is priced, in basis points.
    ///
    /// The IOC needs to reach past the top level to fill the whole size; this is
    /// not the risk bound. `price_band` is the risk bound, and it clamps this.
    pub taker_slippage_bps: u32,
    /// Deltas smaller than this produce no order.
    ///
    /// Without a floor, a target that differs from the position by a dust amount
    /// re-sends an order on every signal — churn that costs rate-limit credits and
    /// queue position and never converges, because the venue's own lot size will
    /// not accept the residue anyway.
    pub min_order_qty: Decimal,

    /// How far the order we would place may differ in **size** from the one already
    /// resting before we bother replacing it, in basis points of the **resting**
    /// size. `0` disables the band, which is the default.
    ///
    /// [`Self::min_order_qty`] answers a different question and only blunts half the
    /// problem: it refuses an order when the *delta* is dust. It says nothing about
    /// the case where the delta is real but the target barely moved — a strategy
    /// nudging its target by a fraction of a percent every tick produces a perfectly
    /// respectable delta and a cancel/replace every tick, and each replacement puts a
    /// post-only order at the back of its price level for a change too small to have
    /// been worth the queue position.
    ///
    /// **Against the resting order, not against the target.** The thing at stake is
    /// the order we would cancel, so it is the only honest denominator: a band on the
    /// target would let a large position's small re-weight through while refusing a
    /// small position's identical one. And it is a *fraction* rather than an absolute
    /// size because one absolute number cannot be right for BTC and for a $138 coin
    /// at the same time — which is exactly why `min_order_qty` needs re-tuning per
    /// instrument and this does not.
    ///
    /// The cost is stated rather than hidden: leaving the order alone means carrying
    /// a position error of up to `noop_band_bps` of the resting size until the next
    /// decision that clears the band. That is the trade, it is bounded, and it is why
    /// the default is off and why a reduce-only order is never banded
    /// ([`WorkingOrder::is_within_band`]).
    pub noop_band_bps: u32,

    /// The operator's **ceiling** on how long a resting order may keep its place
    /// before it must be re-derived from a current decision, in milliseconds of
    /// **event time**. `0` means "I set no bound" — not "no bound may be set", which
    /// is a distinction [`Planner::order_lifetime_ns`] turns on.
    ///
    /// A signal's own `max_order_age_ms` is clamped against this, and the shorter of
    /// the two binds. The strategy may therefore ask for a shorter-lived quote and
    /// never a longer one, exactly as `ttl_ms` works and for the same reason: the
    /// operator answers for the fills.
    ///
    /// **This is not `ttl_ms`, and confusing the two is the standing mistake here.**
    /// `ttl_ms` is a signal *admission* window: `SignalReader` consumes it before the
    /// planner ever sees the record and refuses anything older. Nothing in it has ever
    /// applied to an order already at the venue, so `perp_bar`'s `ttl_ms = 60_000`
    /// buys an order exactly nothing.
    ///
    /// What this bounds is the leave-it-resting exception (ADR-0014 §6), and it is
    /// needed *because* that exception has been widened: with a real
    /// [`WorkingOrder::tif`] an adopted order can now take it, so an order inherited
    /// across a restart — one nobody in this process ever decided to place — could
    /// otherwise rest for as long as the strategy kept asking for the same target.
    /// Past this age the order is cancelled and replaced, so every resting order is
    /// eventually one a live decision produced.
    ///
    /// It is only half of an order lifetime, and the half that is the planner's. The
    /// planner runs on a signal, so this cannot pull a quote for a strategy that has
    /// gone silent; a sweeper on the pass has to do that, and it does not exist yet.
    pub max_order_age_ms: u32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            // 0.5%: enough to sweep several levels on a liquid perp, small enough
            // that an unbanded IOC on a thin book cannot walk the market.
            taker_slippage_bps: 50,
            min_order_qty: Decimal::ZERO,
            // Off. A band is an accepted position error, and one that appeared in a
            // deployment because a default changed would be an error nobody chose.
            noop_band_bps: 0,
            // On, at a minute. Three orders of magnitude above a Hyperliquid block
            // (~0.2 s), so a maker quote keeps its queue position through any
            // realistic re-quote cycle — and bounded, so nothing inherited from a
            // previous incarnation of this process outlives a minute of trading. The
            // alternative default, `0`, would mean the exception this bound exists to
            // limit was widened with nothing limiting it.
            max_order_age_ms: 60_000,
        }
    }
}

/// Everything about the world the planner is allowed to see.
///
/// Deliberately not the whole engine: a planner that could read the book, the
/// clock, or the venue would stop being replayable.
#[derive(Debug, Clone, Copy)]
pub struct PlanContext<'a> {
    /// The pass's event time — an **input**, not a clock.
    ///
    /// The distinction is the whole determinism argument, so it is worth being exact
    /// about: `plan` still reads no clock, and re-planning the same `(signal,
    /// context)` still produces the same orders and the same `cloid`s. What this adds
    /// is the ability to age a *working order* ([`WorkingOrder::placed_ts`]), which
    /// cannot be done from the signal alone. A caller that passed `SystemTime::now()`
    /// here would forfeit the parity harness, exactly as one that paced the pass
    /// schedule on a wall clock would; the runtime passes `CoreHandler::last_ts()`,
    /// the same event clock the reader ages signals against.
    pub now: Nanos,
    /// Signed position we actually hold: `> 0` long, `< 0` short. Filled quantity
    /// only — resting exposure is represented by `working`, not folded in here.
    pub position: Decimal,
    pub quote: Option<Quote>,
    /// Our orders still working at the venue. The planner filters to the signal's
    /// symbol itself, so passing the whole book is safe.
    pub working: &'a [WorkingOrder],
    /// What is known about this instrument's tick and lot (ADR-0025).
    ///
    /// A required *field* so a production call site — `axon-runtime`'s `IntentSource`,
    /// which builds this with a struct literal — cannot leave it out. The
    /// [`new`](Self::new)/[`flat`](Self::flat) constructors default it to
    /// [`Precision::Unconstrained`], which is what keeps the planner's existing tests —
    /// about the delta, the urgency table, the band and the cloid — passing unchanged
    /// rather than being rewritten to accommodate a rule none of them is testing.
    pub precision: Precision<'a>,
}

impl<'a> PlanContext<'a> {
    pub fn new(position: Decimal, quote: Quote) -> Self {
        Self {
            now: 0,
            position,
            quote: Some(quote),
            working: &[],
            precision: Precision::Unconstrained,
        }
    }

    pub fn flat(quote: Quote) -> Self {
        Self::new(Decimal::ZERO, quote)
    }

    pub fn with_working(mut self, working: &'a [WorkingOrder]) -> Self {
        self.working = working;
        self
    }

    pub fn with_now(mut self, now: Nanos) -> Self {
        self.now = now;
        self
    }

    pub fn with_precision(mut self, precision: Precision<'a>) -> Self {
        self.precision = precision;
        self
    }
}

/// Turns an accepted [`Signal`] into order intents.
#[derive(Debug, Clone, Copy, Default)]
pub struct Planner {
    cfg: PlannerConfig,
}

impl Planner {
    pub fn new(cfg: PlannerConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &PlannerConfig {
        &self.cfg
    }

    /// Plan the transition from `ctx.position` to the position `sig` asks for.
    ///
    /// `sig` must already have passed [`crate::SignalReader::admit`]; the planner
    /// re-checks nothing about the record's validity and would happily act on a
    /// stale one.
    pub fn plan(&self, sig: &Signal, ctx: &PlanContext<'_>) -> Plan {
        let symbol = SymbolId::new(sig.symbol_id);

        // Every path but "leave it resting" pulls the superseded orders, including
        // the paths that place nothing: an order resting against a target we have
        // already reached — or against a market we can no longer price — is a stale
        // quote, and a stale quote is exactly what somebody else's taker is looking
        // for. Cancelling costs nothing at the venue and is never risk-rejected.
        let cancels: Vec<CancelId> = ctx
            .working
            .iter()
            .filter(|w| w.symbol_id == symbol)
            .map(|w| CancelId::Cloid {
                symbol,
                cloid: w.cloid,
            })
            .collect();
        let refuse = |reason: NoOrder| Plan {
            cancels: cancels.clone(),
            orders: Vec::new(),
            no_order: Some(reason),
        };

        // FLAG_CLOSE means flatten: the target field is not consulted at all, so a
        // strategy can emit "get me out" without also having to be right about what
        // it currently holds.
        let mut target = if sig.is_close() {
            Decimal::ZERO
        } else {
            fixed_to_decimal(sig.target_qty)
        };

        // What the strategy actually asked for, before the lot took a bite out of it.
        // Kept because the quantized target alone cannot tell "we are already there"
        // apart from "the lot erased the whole request" — both leave a zero delta, and
        // they are not the same session (see the refusal below).
        let requested = target;

        // The **target** is put on the lot, never the delta. Sizes arrive over the ring
        // at eight decimal places and BTC's lot is five: quantize the delta instead and
        // a target of 0.00123456 against a held 0.00123 leaves a residue of 0.00000456
        // that truncates to zero on *every* signal, forever. `AlreadyAtTarget` becomes
        // unreachable, and a strategy that is correctly at its target reads as one that
        // is permanently refused. Quantize the target and the delta between two on-grid
        // numbers is on-grid by construction — FLAG_CLOSE's zero trivially so, and the
        // reduce-only projection below clamps to `[-position, 0]`, where the position
        // is a sum of venue-reported fills and therefore already a whole number of lots.
        if let Precision::Known(spec) = ctx.precision {
            target = spec.size.quantize(target);
        }

        // Closing implies reduce-only even when the strategy did not say so. If a
        // fill lands between our position read and the venue's block, an unqualified
        // flatten overshoots straight into an opposite-side position — the one
        // outcome a flatten must never produce.
        let reduce_only = sig.is_reduce_only() || sig.is_close();

        let raw_delta = target - ctx.position;
        let delta = if reduce_only {
            reduce_only_delta(raw_delta, ctx.position)
        } else {
            raw_delta
        };
        if delta.is_zero() {
            // A target the lot erased is not a target we are already at, and the
            // difference is judged **from flat**. With a position on the books the
            // residue really is smaller than one lot and `AlreadyAtTarget` is the
            // honest answer — that is the case above, and reporting it as a refusal is
            // the bug §4 of ADR-0025 exists to avoid. From flat there is no residue:
            // the strategy asked for exposure and the lot left it with none of it, on
            // every signal, forever. On the 125 of testnet's 210 perps whose lot is a
            // whole coin that is the *common* target, not a corner — and reported as
            // the healthy case it is counted nowhere, so `precision_refusals` stays
            // zero and a session asking to trade all day is indistinguishable on the
            // status line from one that chose to be flat.
            if let Precision::Known(spec) = ctx.precision {
                if ctx.position.is_zero() && target.is_zero() && !requested.is_zero() {
                    return refuse(NoOrder::BelowLotSize {
                        delta: requested.abs(),
                        lot: spec.size.increment(),
                    });
                }
            }
            return refuse(if raw_delta.is_zero() {
                NoOrder::AlreadyAtTarget
            } else {
                // The strategy asked to grow or flip under reduce-only. Clamping to
                // zero rather than sending it keeps the planner from handing the risk
                // gate an order the gate is certain to reject — and a risk rejection
                // is indistinguishable in the logs from a venue outage.
                NoOrder::ReduceOnlyWouldGrow
            });
        }

        let mut qty = delta.abs();

        // Idempotent on the normal path, because the target was already quantized —
        // so this is not a second decision. It is insurance against the one case the
        // step above cannot cover: if the venue re-precisions an asset mid-session, the
        // position we hold is off the *new* grid and so is the delta. A zero is "no
        // order", never a zero-size order — that is a guaranteed rejection which still
        // spends a nonce, a rate-limit credit and a log line saying we tried.
        if let Precision::Known(spec) = ctx.precision {
            let on_lot = spec.size.quantize(qty);
            if on_lot.is_zero() {
                return refuse(NoOrder::BelowLotSize {
                    delta: qty,
                    lot: spec.size.increment(),
                });
            }
            qty = on_lot;
        }

        // **Does this order take the position to exactly flat?** Computed from the
        // arithmetic rather than from `FLAG_CLOSE`, because the two floors below are
        // about what an order *does*, and the documented flatten path emits an ordinary
        // target of zero (`StrategyContext.emit_target`) rather than a close.
        let closes_out = !ctx.position.is_zero() && target.is_zero() && qty == ctx.position.abs();

        // The dust floor is a **churn** bound, and a close is not churn.
        //
        // `min_order_qty` exists so that a target differing from the position by a
        // rounding residue does not re-send an order on every signal — an oscillation
        // that never converges, because the venue's lot will not accept the residue
        // anyway. An order that takes the position to zero cannot be that. It converges
        // by construction; there is at most one of it; if it rests, the leave-it-alone
        // rule below leaves it; and if it is swept, the re-quote budget bounds how often
        // it comes back.
        //
        // Applying the floor to it instead produces a position **nothing can ever
        // close**. Measured live on 2026-07-27, twice in 59 minutes: a closing buy for
        // 0.0003 BTC partially filled 0.00017, the sweeper pulled the remainder a minute
        // later, and the 0.00013 BTC left over sat under a `min_order_qty` of 0.00016
        // with no order that could express it. The session named it (`STRANDED
        // POSITION`) and nothing could act on it — the planner refused every close, so
        // the re-quote correctly placed nothing, and only an operator could clear it.
        //
        // This is the same asymmetry [`InstrumentSpec::min_notional`] already has below,
        // and it rests on the same sentence: refusing a close on our *own* opinion means
        // nobody ever finds out, while sending one the venue refuses is loud, observable
        // and costs one metered request.
        if qty < self.cfg.min_order_qty && !closes_out {
            return refuse(NoOrder::BelowMinQty {
                delta: qty,
                min: self.cfg.min_order_qty,
            });
        }

        let side = if delta.is_sign_positive() {
            Side::Buy
        } else {
            Side::Sell
        };

        let Some(quote) = ctx.quote.filter(Quote::is_usable) else {
            return refuse(NoOrder::NoQuote);
        };
        let rule = urgency_rule(sig.urgency);
        let anchor_px = match (side, rule.anchor) {
            (Side::Buy, Anchor::NearTouch) => quote.bid_px,
            (Side::Buy, Anchor::FarTouch) => quote.ask_px,
            (Side::Sell, Anchor::NearTouch) => quote.ask_px,
            (Side::Sell, Anchor::FarTouch) => quote.bid_px,
        };
        let mut price = if rule.slipped {
            slipped(anchor_px, side, self.cfg.taker_slippage_bps)
        } else {
            anchor_px
        };

        // A negative band is not a price, whatever the instrument's grid is, so it is
        // refused before anything else looks at it.
        let band = if sig.price_band != 0 {
            let b = fixed_to_decimal(sig.price_band);
            if b <= Decimal::ZERO {
                return refuse(NoOrder::BadPriceBand { band: b });
            }
            Some(b)
        } else {
            None
        };

        if matches!(ctx.precision, Precision::Unknown) {
            // One governing sentence: **with no grid, the only order we will send is a
            // reduce-only order for the whole position at a price the venue itself
            // printed.** Both of those numbers came from the venue — the size is
            // `-position`, a sum of venue-reported fills and therefore a whole number
            // of lots, and the price is a touch the venue published and is therefore on
            // the venue's own grid. Everything else is a number *we* computed and
            // cannot be made legal without knowing the grid, so it is refused rather
            // than sent to be rejected.
            if !reduce_only || qty != ctx.position.abs() {
                return refuse(NoOrder::UnknownPrecision { symbol });
            }
            // No slippage: an urgency-3 exit is downgraded to the far touch, and the
            // downgrade is documented and counted (`IntentStats::unknown_precision`)
            // rather than silent.
            price = anchor_px;
            if let Some(b) = band {
                let clamped = match side {
                    Side::Buy => price.min(b),
                    Side::Sell => price.max(b),
                };
                if clamped != price {
                    return refuse(NoOrder::UnknownPrecision { symbol });
                }
            }
        } else {
            if let Some(b) = band {
                // The band is the *worst* price acceptable, so it is a ceiling for a buy
                // and a floor for a sell. It can only ever move the limit away from the
                // market, never through it.
                price = match side {
                    Side::Buy => price.min(b),
                    Side::Sell => price.max(b),
                };
            }
            // The banded, unquantized price. Kept because it is what distinguishes "the
            // band cannot cross" from "the grid cannot cross" below, and those two send
            // an operator to change different numbers.
            let pre = price;

            if let Precision::Known(spec) = ctx.precision {
                let intent = if rule.is_marketable() {
                    PriceIntent::Marketable
                } else {
                    PriceIntent::Passive
                };
                price = spec.price.quantize(price, side, intent);
                if let Some(b) = band {
                    // Quantized AWAY from the market — a buy's ceiling floors, a sell's
                    // floor ceils — and re-applied AFTER the price is. Without this
                    // second clamp a marketable buy rounded toward the market steps
                    // through its own band by up to one tick, and a risk bound violated
                    // by arithmetic is worse than one violated on purpose. The invariant
                    // that falls out is unconditional: the price sent is never worse
                    // than the band the strategy set.
                    let band_q = spec.price.quantize(b, side, PriceIntent::Passive);
                    if band_q <= Decimal::ZERO {
                        return refuse(NoOrder::PriceNotRepresentable {
                            price: b,
                            tick: spec.price.tick_at(b),
                        });
                    }
                    price = match side {
                        Side::Buy => price.min(band_q),
                        Side::Sell => price.max(band_q),
                    };
                }
                if price <= Decimal::ZERO {
                    return refuse(NoOrder::PriceNotRepresentable {
                        price: pre,
                        tick: spec.price.tick_at(pre),
                    });
                }
            }

            if rule.is_marketable() {
                // The band — or the grid — and the urgency now contradict each other:
                // the strategy asked to trade immediately at a price that cannot trade.
                // Quietly demoting the order to a resting quote would invent an intent
                // nobody expressed, and sending it anyway is a guaranteed no-op that
                // still consumes a nonce, a rate-limit credit, and a line in the log
                // that says we tried.
                let touch = match side {
                    Side::Buy => quote.ask_px,
                    Side::Sell => quote.bid_px,
                };
                let crosses = |p: Decimal| match side {
                    Side::Buy => p >= touch,
                    Side::Sell => p <= touch,
                };
                if !crosses(price) {
                    // A buy's final price never exceeds `pre` unless the grid moved it,
                    // so these two partition cleanly and the existing band tests are
                    // untouched.
                    return refuse(if crosses(pre) {
                        NoOrder::RoundedUnfillable {
                            limit: price,
                            touch,
                        }
                    } else {
                        NoOrder::PriceBandUnfillable {
                            limit: price,
                            touch,
                        }
                    });
                }
            }

            if let Precision::Known(spec) = ctx.precision {
                if let Some(min) = spec.min_notional {
                    let notional = qty * price;
                    // Refusing an order that ADDS exposure and cannot meet the minimum
                    // saves a guaranteed rejection. Refusing a CLOSE would be failing to
                    // flatten on our own opinion: the venue's published rule does not say
                    // whether it exempts closes, and guessing in the closed direction
                    // strands a position. If we are wrong the venue tells us, loudly and
                    // observably; if we refuse, nobody ever finds out.
                    //
                    // `closes_out` is here beside `reduce_only` because the flag is not
                    // the question. The documented flatten path emits a plain target of
                    // zero, which carries neither `FLAG_CLOSE` nor `FLAG_REDUCE_ONLY`, so
                    // keying only on the flag stranded exactly the residue this exemption
                    // was written for — the order that takes a position to flat is a
                    // close whatever bit it happens to carry.
                    if notional < min && !(reduce_only || closes_out) {
                        return refuse(NoOrder::BelowMinNotional { notional, min });
                    }
                }
            }
        }

        let req = OrderRequest {
            symbol_id: symbol,
            side,
            qty,
            price: Some(price),
            order_type: OrderType::Limit,
            tif: rule.tif,
            reduce_only,
            trigger: None,
            cloid: cloid_for(sig),
        };

        // Cancel/replace has a cost the venue charges in queue position: a replaced
        // post-only order goes to the back of its price level. When the working order
        // *is* the order we would send, replacing it is a strict loss, so leave it.
        // Only when it is the sole working order, though — otherwise the others still
        // have to go, and the delta was computed against a book with none of them in it.
        //
        // Three things have to hold, and each of them is a separate failure if it does
        // not. Every field must be **known**, or we would be comparing against an
        // invention (`tif`/`reduce_only` are `Option`s for exactly that). The order
        // must not have outlived its age bound, or an order nobody in this process
        // decided to place could rest for as long as the target held. And the size may
        // differ only inside the no-op band, which is the deliberate, bounded position
        // error the operator opted into.
        if ctx.working.len() == 1 && ctx.working[0].symbol_id == symbol {
            let resting = &ctx.working[0];
            if !self.outlived(resting, sig, ctx.now) {
                if resting.is(&req) {
                    return Plan {
                        cancels: Vec::new(),
                        orders: Vec::new(),
                        no_order: Some(NoOrder::AlreadyWorking),
                    };
                }
                if resting.is_within_band(&req, self.cfg.noop_band_bps) {
                    return Plan {
                        cancels: Vec::new(),
                        orders: Vec::new(),
                        no_order: Some(NoOrder::WithinNoOpBand {
                            resting: resting.remaining_qty,
                            wanted: req.qty,
                        }),
                    };
                }
            }
        }

        Plan {
            cancels,
            orders: vec![req],
            no_order: None,
        }
    }

    /// True when `w` has been working longer than an order is allowed to keep its
    /// place, measured in event time from [`WorkingOrder::placed_ts`].
    ///
    /// Only ever used to *deny* the leave-it-resting exception, never to emit a cancel
    /// on its own — every no-order path already cancels what is working, so an
    /// over-age order goes on the next signal whatever that signal decides. That is
    /// the half of an order lifetime the planner can honestly own: it runs on a
    /// signal, so it cannot pull a quote from a strategy that has stopped speaking.
    fn outlived(&self, w: &WorkingOrder, sig: &Signal, now: Nanos) -> bool {
        match self.order_lifetime_ns(sig.max_order_age_ms) {
            Some(limit) => now.saturating_sub(w.placed_ts) > limit,
            None => false,
        }
    }

    /// The lifetime an order placed for `sig` is actually allowed, in nanoseconds, or
    /// `None` when nobody has asked for a bound at all.
    ///
    /// **Zero is not an opinion, on either side, and it is the case that inverts if
    /// this is written carelessly.** The tempting one-liner is
    /// `requested.min(ceiling)`, and it is wrong in exactly the direction that hurts:
    /// `min(0, 60_000)` is `0`, so a strategy that never set the field — every
    /// strategy that exists today, and the Python default — would have every order it
    /// placed cancelled on the very next pass. That is ADR-0020 §4's argument
    /// transferred without a change: zero is the value of a field nobody wrote, so the
    /// safe reading is the operator's ceiling, never "already expired". Filtering the
    /// zeros out *before* the comparison makes the mistake unrepresentable rather than
    /// merely avoided.
    ///
    /// What is left is one sentence: **the binding lifetime is the shortest one anyone
    /// actually expressed.** A strategy can therefore only ever ask for a *shorter*
    /// life than the operator allows and never a longer one, which is the same
    /// asymmetry `ttl_ms` has and for the same reason — the operator answers for the
    /// fills. And an operator who sets no ceiling has not thereby overridden a strategy
    /// that asked for a short-lived quote: `0` there means "I set no bound", not "no
    /// bound may be set".
    fn order_lifetime_ns(&self, requested_ms: u32) -> Option<Nanos> {
        let binding = [requested_ms, self.cfg.max_order_age_ms]
            .into_iter()
            .filter(|ms| *ms != 0)
            .min()?;
        Some(Nanos::from(binding).saturating_mul(NS_PER_MS))
    }
}

/// Project `delta` onto what reduce-only permits: the closed interval between
/// flat and the position we hold, and nothing beyond it.
///
/// Growing is clamped to zero (no order). A delta that would flip through zero is
/// clamped to exactly flat, which is the strategy's own target projected onto the
/// reachable set — the closest we can legitimately get to what it asked for.
fn reduce_only_delta(delta: Decimal, position: Decimal) -> Decimal {
    if position.is_zero() {
        return Decimal::ZERO;
    }
    if position.is_sign_positive() {
        delta.clamp(-position, Decimal::ZERO)
    } else {
        delta.clamp(Decimal::ZERO, -position)
    }
}

/// Push `px` `bps` basis points *through* the touch, in the direction that fills.
fn slipped(px: Decimal, side: Side, bps: u32) -> Decimal {
    let frac = Decimal::from(i64::from(bps)) / Decimal::from(BPS);
    match side {
        Side::Buy => px * (Decimal::ONE + frac),
        Side::Sell => px * (Decimal::ONE - frac),
    }
}

/// Bit 127 of every planner-minted `cloid`.
///
/// It guarantees the id is non-zero, and it keeps the planner's id space disjoint
/// from `OrderTracker`'s adopted-order ids, which are a bare venue `oid` widened to
/// 128 bits and therefore always have this bit clear. Two orders sharing a cloid is
/// the one collision that makes reconciliation attribute a fill to the wrong order.
pub const CLOID_PLANNER_TAG: u128 = 1 << 127;

/// Derive the client order id from the signal's identity.
///
/// Deterministic, and that is the whole point: a submit that times out has to be
/// retried, and a retry that mints a fresh id is a second order. With this, the
/// venue's own cloid de-duplication makes the retry a no-op instead of a doubled
/// position. It also means a replayed session produces byte-identical order ids,
/// which is what lets the parity harness diff a backtest against a live run.
///
/// The layout, 128 bits:
///
/// | bits    | field                                    |
/// |---------|------------------------------------------|
/// | 127     | [`CLOID_PLANNER_TAG`]                    |
/// | 126..64 | `ts_event` nanoseconds (low 63 bits)      |
/// | 63..32  | `seq` (low 32 bits)                      |
/// | 31..0   | `symbol_id`                              |
///
/// Event-time nanoseconds are near-unique on their own; `seq` separates two
/// decisions inside one nanosecond, and `symbol_id` separates a multi-symbol
/// strategy's simultaneous decisions. Nothing here is a hash, so an operator
/// reading a cloid out of a venue log can recover which signal produced it.
///
/// One order per signal today. Slicing a target across several child orders would
/// need a leg index, and there is no room left for one — that change has to
/// re-cut this layout rather than extend it.
pub fn cloid_for(sig: &Signal) -> Cloid {
    let ts = (sig.ts_event as u64 as u128) & ((1u128 << 63) - 1);
    let seq = (sig.seq as u32) as u128;
    let sym = sig.symbol_id as u128;
    Cloid::new(CLOID_PLANNER_TAG | (ts << 64) | (seq << 32) | sym)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::{FLAG_CLOSE, FLAG_REDUCE_ONLY};
    use rust_decimal_macros::dec;

    const SYM: u32 = 7;

    fn quote() -> Quote {
        Quote::new(dec!(100), dec!(101))
    }

    /// A target-position signal: `target` in whole units, converted to the wire scale.
    fn sig(target: Decimal, urgency: u8) -> Signal {
        Signal::target_position(
            1,
            1_000_000_000,
            SYM,
            crate::fixed::decimal_to_fixed(target).unwrap(),
            urgency,
            0,
            500,
            42,
            0,
        )
    }

    fn banded(target: Decimal, urgency: u8, band: Decimal) -> Signal {
        let mut s = sig(target, urgency);
        s.price_band = crate::fixed::decimal_to_fixed(band).unwrap();
        s
    }

    fn planner() -> Planner {
        Planner::default()
    }

    fn only(plan: &Plan) -> &OrderRequest {
        assert!(plan.no_order.is_none(), "expected an order: {plan:?}");
        assert_eq!(plan.orders.len(), 1);
        &plan.orders[0]
    }

    // ── instrument grids the rounding tests price against ────────────────────

    use axon_providers::{InstrumentSpec, PriceGrid, SizeGrid};

    fn spec_of(price: PriceGrid, size: SizeGrid, min_notional: Option<Decimal>) -> InstrumentSpec {
        InstrumentSpec {
            symbol_id: SymbolId::new(SYM),
            price,
            size,
            min_notional,
        }
    }

    /// Testnet BTC's real shape: `szDecimals: 5`, so one decimal place capped at five
    /// significant figures — the tick is 0.1 at three digits and widens to 1 above
    /// 10 000 — a `1e-5` lot, and the venue's $10 minimum.
    fn btc() -> InstrumentSpec {
        spec_of(
            PriceGrid::decimals_with_sig_figs(1, 5).unwrap(),
            SizeGrid::decimals(5).unwrap(),
            Some(dec!(10)),
        )
    }

    /// A deliberately coarse instrument: a whole-unit tick, so a two-figure book is
    /// off-grid on both sides and the rounding direction is visible in one digit.
    fn coarse() -> InstrumentSpec {
        spec_of(
            PriceGrid::increment(dec!(1)).unwrap(),
            SizeGrid::decimals(2).unwrap(),
            None,
        )
    }

    /// Testnet ZEC's real shape: `szDecimals: 0`, so the lot is one whole coin — the
    /// grid 125 of testnet's 210 perps declare, and the one a sub-coin target vanishes
    /// on. Six price decimals capped at five figures, and the venue's $10 minimum.
    fn whole_lot() -> InstrumentSpec {
        spec_of(
            PriceGrid::decimals_with_sig_figs(6, 5).unwrap(),
            SizeGrid::decimals(0).unwrap(),
            Some(dec!(10)),
        )
    }

    /// A tenth-of-a-unit tick with no minimum: the grid a band can step through.
    fn tenths() -> InstrumentSpec {
        spec_of(
            PriceGrid::increment(dec!(0.1)).unwrap(),
            SizeGrid::decimals(5).unwrap(),
            None,
        )
    }

    #[test]
    fn the_target_is_quantized_before_the_delta_so_an_unreachable_target_reads_as_at_target() {
        // Quantize the *delta* instead and this is the failure: sizes arrive at eight
        // decimal places, BTC's lot is five, and the 0.00000456 residue truncates to
        // zero on every signal forever. `AlreadyAtTarget` becomes unreachable and a
        // strategy that is correctly at its target reads as one that is permanently
        // refused — a counter climbing on a perfectly healthy session.
        let s = btc();
        let ctx = PlanContext::new(dec!(0.00123), quote()).with_precision(Precision::Known(&s));
        let plan = planner().plan(&sig(dec!(0.00123456), 0), &ctx);
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyAtTarget));

        // And a reachable target still produces a delta that is itself on the lot,
        // because both ends of the subtraction are. Unquantized this would be
        // 0.10000456 — six decimal places past a five-decimal lot.
        let plan = planner().plan(&sig(dec!(0.10123456), 0), &ctx);
        assert_eq!(only(&plan).qty, dec!(0.1));
        assert!(s.size.is_valid(only(&plan).qty));
    }

    #[test]
    fn a_target_the_lot_erases_from_flat_is_a_counted_refusal_and_never_reads_as_at_target() {
        // A whole-coin lot is the majority shape on testnet, not a corner: a $50 target
        // on a $138 coin is 0.36 of one, the lot truncates it to nothing, and the delta
        // is zero with nothing held. Reported as `AlreadyAtTarget` that is the *healthy*
        // variant — it falls through `IntentSource`'s counter arms, so `precision_refusals`
        // stays zero, neither `prec` nor `NOSPEC` reaches the status line, and a session
        // asking to trade on every signal prints `sig N/0 sent 0+0c | OK` forever. A
        // strategy being silently refused must never be indistinguishable from one that
        // chose to be flat.
        let s = whole_lot();
        let ctx = PlanContext::flat(quote()).with_precision(Precision::Known(&s));
        let plan = planner().plan(&sig(dec!(0.36), 1), &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BelowLotSize {
                delta: dec!(0.36),
                lot: dec!(1)
            }),
            "the size the strategy asked for, and the lot that erased it"
        );
        // …and it is a different plan from the one for a strategy that is genuinely
        // where it asked to be, which is the confusion that hid this.
        assert_eq!(
            planner()
                .plan(
                    &sig(dec!(3), 1),
                    &PlanContext::new(dec!(3), quote()).with_precision(Precision::Known(&s))
                )
                .no_order,
            Some(NoOrder::AlreadyAtTarget)
        );
        // The residue case above stays the healthy one: a position already within one
        // lot of its target is at its target, however fine the request was.
        assert_eq!(
            planner()
                .plan(
                    &sig(dec!(3.36), 1),
                    &PlanContext::new(dec!(3), quote()).with_precision(Precision::Known(&s))
                )
                .no_order,
            Some(NoOrder::AlreadyAtTarget)
        );
    }

    #[test]
    fn a_marketable_price_is_never_rounded_out_of_the_market() {
        // Rounding a taker away from the market silently converts "take liquidity now"
        // into a resting quote and leaves an unmanaged position — the exact outcome the
        // urgency table says level 2 and 3 exist to prevent.
        let s = coarse();
        let q = Quote::new(dec!(100.5), dec!(101.5)); // both sides off a whole-unit grid
        let p = planner();

        let buy = p.plan(
            &sig(dec!(1), 2),
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        assert_eq!(only(&buy).price, Some(dec!(102)), "up, through the ask");
        assert!(only(&buy).price.unwrap() >= q.ask_px, "it still crosses");

        let sell = p.plan(
            &sig(dec!(-1), 2),
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        assert_eq!(only(&sell).price, Some(dec!(100)), "down, through the bid");
        assert!(only(&sell).price.unwrap() <= q.bid_px);
    }

    #[test]
    fn a_post_only_price_is_never_rounded_into_the_spread() {
        // Post-only is **rejected**, not demoted, if it would cross. A level-0 buy
        // rounded up into the spread is therefore not a slightly worse quote — it is an
        // `alo` rejection and a strategy sitting flat with nothing working.
        let s = coarse();
        let q = Quote::new(dec!(100.5), dec!(101.5));
        let p = planner();

        let buy = p.plan(
            &sig(dec!(1), 0),
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        let o = only(&buy);
        assert_eq!(o.tif, Tif::PostOnly);
        assert_eq!(o.price, Some(dec!(100)), "down, away from the ask");
        assert!(o.price.unwrap() <= q.bid_px, "it cannot cross");

        let sell = p.plan(
            &sig(dec!(-1), 0),
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        assert_eq!(only(&sell).price, Some(dec!(102)), "up, away from the bid");
        assert!(only(&sell).price.unwrap() >= q.ask_px);
    }

    #[test]
    fn rounding_never_moves_the_limit_past_the_price_band() {
        // Quantize a marketable buy toward the market and it steps through its own band
        // by up to one tick. A risk bound violated by arithmetic is worse than one
        // violated on purpose, so the band is re-applied — itself quantized away from
        // the market — after the price is.
        let s = tenths();
        let ctx = PlanContext::flat(quote()).with_precision(Precision::Known(&s));
        // Urgency 3 prices at 101 + 50 bps = 101.505; the ceil to the 0.1 grid is
        // 101.6, which is past the 101.55 the strategy set.
        let plan = planner().plan(&banded(dec!(1), 3, dec!(101.55)), &ctx);
        let o = only(&plan);
        assert_eq!(o.price, Some(dec!(101.5)));
        assert!(
            o.price.unwrap() <= dec!(101.55),
            "the price sent is never worse than the band"
        );
        assert!(o.price.unwrap() >= quote().ask_px, "and it still crosses");
    }

    #[test]
    fn a_grid_coarser_than_the_band_produces_no_order_rather_than_a_resting_quote() {
        // ADR-0014 §4's precedent, one layer down: the strategy asked to trade now at a
        // price that cannot trade. Its own variant, because the fix differs — "your band
        // is too tight" sends an operator to a different number than "this instrument's
        // grid is coarser than your band".
        let s = coarse();
        let q = Quote::new(dec!(100.5), dec!(101.5));
        let ctx = PlanContext::flat(q).with_precision(Precision::Known(&s));

        let plan = planner().plan(&banded(dec!(1), 2, dec!(101.9)), &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::RoundedUnfillable {
                limit: dec!(101),
                touch: dec!(101.5)
            }),
            "the band alone would have crossed; the grid is what stopped it"
        );

        // A band that could not cross even unquantized is still the band's fault, and
        // still says so.
        let plan = planner().plan(&banded(dec!(1), 2, dec!(100.9)), &ctx);
        assert!(matches!(
            plan.no_order,
            Some(NoOrder::PriceBandUnfillable { .. })
        ));
    }

    #[test]
    fn a_delta_finer_than_the_lot_produces_no_order_rather_than_a_zero_size_one() {
        // The case quantizing the target cannot cover: the venue re-precisions an asset
        // mid-session, so the position we already hold is off the *new* grid and so is
        // the delta. A zero-size order is a guaranteed rejection that still spends a
        // nonce, a rate-limit credit and a log line saying we tried.
        let s = btc();
        let mut close = sig(dec!(0), 0);
        close.flags = FLAG_CLOSE;
        let ctx = PlanContext::new(dec!(0.000004), quote()).with_precision(Precision::Known(&s));
        let plan = planner().plan(&close, &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BelowLotSize {
                delta: dec!(0.000004),
                lot: dec!(0.00001)
            })
        );
    }

    #[test]
    fn a_notional_under_the_venue_minimum_produces_no_order_rather_than_a_guaranteed_rejection() {
        // Below the venue's floor the order comes back `minTradeNtlRejected` — a
        // certainty, paid for with a nonce and a rate credit, that reads in the log
        // exactly like a risk refusal.
        let s = btc();
        let ctx = PlanContext::flat(quote()).with_precision(Precision::Known(&s));
        let plan = planner().plan(&sig(dec!(0.05), 0), &ctx);
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BelowMinNotional {
                notional: dec!(5.00),
                min: dec!(10)
            })
        );
        // At the floor it goes through.
        let plan = planner().plan(&sig(dec!(0.1), 0), &ctx);
        assert_eq!(only(&plan).qty, dec!(0.1));
    }

    #[test]
    fn a_reduce_only_order_under_the_minimum_notional_is_still_sent_because_we_never_refuse_to_de_risk(
    ) {
        // The venue's published rule does not say whether it exempts closes. Guessing in
        // the closed direction strands a position on our own opinion; if we are wrong
        // the venue tells us, loudly and observably, and if we refuse nobody ever finds
        // out. Same asymmetry as the risk gate's missing mark (ADR-0010).
        let s = btc();
        let mut close = sig(dec!(0), 0);
        close.flags = FLAG_CLOSE;
        let ctx = PlanContext::new(dec!(0.05), quote()).with_precision(Precision::Known(&s));
        let plan = planner().plan(&close, &ctx);
        let o = only(&plan);
        assert_eq!(o.qty, dec!(0.05));
        assert!(o.reduce_only);
        assert!(
            o.qty * o.price.unwrap() < dec!(10),
            "the notional really is under the minimum"
        );
    }

    #[test]
    fn an_unknown_grid_refuses_anything_that_adds_exposure() {
        // Fail closed for exposure, the way ADR-0010 fails closed on a missing mark. An
        // order priced or sized by us against a grid we cannot see is a `tickRejected`
        // that reads like a signing bug from inside the process.
        let ctx = PlanContext::flat(quote()).with_precision(Precision::Unknown);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::UnknownPrecision {
                symbol: SymbolId::new(SYM)
            })
        );
    }

    #[test]
    fn an_unknown_grid_still_flattens_at_a_price_the_venue_itself_printed() {
        // The one exemption, and it is narrow for a reason: both numbers came from the
        // venue. The size is `-position`, a sum of venue-reported fills and therefore a
        // whole number of lots; the price is a touch the venue published and is
        // therefore on the venue's own grid.
        let mut close = sig(dec!(0), 3);
        close.flags = FLAG_CLOSE;
        let ctx = PlanContext::new(dec!(4), quote()).with_precision(Precision::Unknown);
        let plan = planner().plan(&close, &ctx);
        let o = only(&plan);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(o.qty, dec!(4));
        assert!(o.reduce_only);
        assert_eq!(o.tif, Tif::Ioc, "the urgency table's own TIF still applies");
        assert_eq!(
            o.price,
            Some(dec!(100)),
            "the far touch, with the slippage dropped: a slipped price is a number WE \
             computed and cannot be made legal without a grid"
        );

        // And a band that would move that price takes the exemption away, because the
        // result would again be a number we computed.
        let mut banded_close = close;
        banded_close.price_band = crate::fixed::decimal_to_fixed(dec!(100.5)).unwrap();
        let plan = planner().plan(&banded_close, &ctx);
        assert_eq!(
            plan.no_order,
            Some(NoOrder::UnknownPrecision {
                symbol: SymbolId::new(SYM)
            })
        );
    }

    #[test]
    fn an_unknown_grid_refuses_a_partial_reduce_because_its_size_is_a_number_we_computed() {
        // The distinction between an exemption and a hole. A full flatten's size is the
        // venue's own; a partial reduce's is ours, and ours can be off-lot.
        let mut shrink = sig(dec!(1), 0);
        shrink.flags = FLAG_REDUCE_ONLY;
        let ctx = PlanContext::new(dec!(4), quote()).with_precision(Precision::Unknown);
        let plan = planner().plan(&shrink, &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::UnknownPrecision {
                symbol: SymbolId::new(SYM)
            })
        );
    }

    #[test]
    fn an_unknown_grid_still_cancels_the_orders_it_refuses_to_replace() {
        // Refusing to place is not a reason to leave a stale quote out in a market we
        // are no longer willing to price into — that is exactly what somebody else's
        // taker is looking for. Cancels are never gated (ADR-0010).
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_precision(Precision::Unknown);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(plan.cancels.len(), 1);
    }

    #[test]
    fn a_quantized_price_matches_the_venues_own_echo_so_the_order_is_not_replaced_every_pass() {
        // The latent churn bug this incidentally fixes. `WorkingOrder::price` comes from
        // the venue's `orderUpdates` echo and is therefore on-grid; an unquantized
        // planner price could never equal it, so the leave-it-resting exception never
        // fired on a coarse instrument and every pass cancel/replaced an order that was
        // already correct — paying queue position for a change that never happened.
        let s = tenths();
        let banded_sig = banded(dec!(1), 1, dec!(99.97));
        let unrounded = planner().plan(&banded_sig, &PlanContext::flat(quote()));
        assert_eq!(
            only(&unrounded).price,
            Some(dec!(99.97)),
            "off the 0.1 grid, so the venue could never echo it back"
        );

        let working = [WorkingOrder {
            cloid: cloid_for(&banded_sig),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(99.9)), // what the venue reports back
            remaining_qty: dec!(1),
            tif: Some(Tif::Gtc),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_precision(Precision::Known(&s));
        let plan = planner().plan(&banded_sig, &ctx);
        assert!(plan.is_noop());
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyWorking));
    }

    #[test]
    fn an_urgency_three_price_lands_on_the_grid_the_venue_will_accept() {
        // The exact case that was broken. At an ask of 108234, level 3 prices at
        // `108234 x 1.005 = 108775.17` — nine significant figures against a five-figure
        // cap — which the venue answers with `tickRejected`. Nothing in the process
        // could tell that apart from a signing bug.
        let s = btc();
        let q = Quote::new(dec!(108233), dec!(108234));
        let ctx = PlanContext::flat(q).with_precision(Precision::Known(&s));
        let plan = planner().plan(&sig(dec!(0.001), 3), &ctx);
        let o = only(&plan);

        assert!(
            !s.price.is_valid(dec!(108775.17)),
            "the unrounded slipped price is exactly what the venue refuses"
        );
        assert_eq!(o.price, Some(dec!(108776)), "ceil onto the whole-unit tick");
        assert!(s.price.is_valid(o.price.unwrap()));
        assert!(s.size.is_valid(o.qty));
        assert!(o.price.unwrap() >= q.ask_px, "and it still crosses");
        // The encoder's own refusal, run against the planner's own output: if these two
        // ever drifted, the session would plan orders its own wire rejects.
        assert!(s.check(o).is_ok());

        // The sell mirror: floor, so it still crosses the bid.
        let plan = planner().plan(&sig(dec!(-0.001), 3), &ctx);
        let o = only(&plan);
        assert_eq!(o.price, Some(dec!(107691)), "108233 - 50 bps, floored");
        assert!(s.price.is_valid(o.price.unwrap()));
        assert!(o.price.unwrap() <= q.bid_px);
    }

    #[test]
    fn the_same_signal_and_the_same_grid_yield_the_same_order_and_the_same_cloid() {
        // Rounding is a pure function of `(price, qty, spec)` and `cloid_for` never saw
        // it, so a replayed session still re-plans byte-identical intents — the property
        // the parity harness diffs a backtest against a live run on.
        let s = btc();
        let q = Quote::new(dec!(108233), dec!(108234));
        let record = sig(dec!(0.001), 3);
        let p = planner();

        let first = p.plan(
            &record,
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        let again = p.plan(
            &record,
            &PlanContext::flat(q).with_precision(Precision::Known(&s)),
        );
        assert_eq!(first, again, "same signal, same grid, same plan");

        // The grid genuinely moved the price, and the id did not move with it.
        let ungridded = p.plan(&record, &PlanContext::flat(q));
        assert_ne!(only(&first).price, only(&ungridded).price);
        assert_eq!(only(&first).cloid, only(&ungridded).cloid);
        assert_eq!(only(&first).cloid, cloid_for(&record));
    }

    #[test]
    fn the_order_is_the_delta_not_the_target() {
        // The failure this prevents: "be long 3" while already long 2 becomes an
        // order for 3, and the position ends at 5. Silent, and it compounds on every
        // signal until a risk limit finally catches it.
        let p = planner();
        let plan = p.plan(&sig(dec!(3), 0), &PlanContext::new(dec!(2), quote()));
        let o = only(&plan);
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.qty, dec!(1));

        // And the same going the other way, including across zero.
        let plan = p.plan(&sig(dec!(-1), 0), &PlanContext::new(dec!(2), quote()));
        let o = only(&plan);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(o.qty, dec!(3), "2 to close plus 1 to open short");
    }

    #[test]
    fn a_target_equal_to_the_current_position_produces_no_order() {
        let plan = planner().plan(&sig(dec!(2.5), 0), &PlanContext::new(dec!(2.5), quote()));
        assert!(plan.orders.is_empty());
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyAtTarget));
        assert!(plan.is_noop());
    }

    #[test]
    fn a_target_reached_while_orders_are_working_still_pulls_them() {
        // The orders are chasing a delta that no longer exists. Leaving them resting
        // is leaving a stale quote for somebody else's taker.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::new(dec!(2), quote()).with_working(&working);
        let plan = planner().plan(&sig(dec!(2), 0), &ctx);
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyAtTarget));
        assert_eq!(
            plan.cancels,
            vec![CancelId::Cloid {
                symbol: SymbolId::new(SYM),
                cloid: Cloid::new(5)
            }]
        );
    }

    #[test]
    fn flag_close_flattens_a_short_as_well_as_a_long() {
        let p = planner();
        let mut s = sig(dec!(999), 3); // target_qty must be ignored entirely
        s.flags = FLAG_CLOSE;

        let long = p.plan(&s, &PlanContext::new(dec!(4), quote()));
        let o = only(&long);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(o.qty, dec!(4));
        assert!(o.reduce_only, "a flatten must never open the other side");

        let short = p.plan(&s, &PlanContext::new(dec!(-4), quote()));
        let o = only(&short);
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.qty, dec!(4));
        assert!(o.reduce_only);

        // Already flat: nothing to close, and the bogus target is still ignored.
        let flat = p.plan(&s, &PlanContext::flat(quote()));
        assert!(flat.orders.is_empty());
        assert_eq!(flat.no_order, Some(NoOrder::AlreadyAtTarget));
    }

    #[test]
    fn a_reduce_only_signal_never_grows_or_flips_the_position() {
        let p = planner();
        let mut grow = sig(dec!(5), 0);
        grow.flags = FLAG_REDUCE_ONLY;
        let plan = p.plan(&grow, &PlanContext::new(dec!(2), quote()));
        assert_eq!(plan.no_order, Some(NoOrder::ReduceOnlyWouldGrow));

        // Asking to flip long 2 → short 3 is clamped to flat, not sent as a sell 5.
        let mut flip = sig(dec!(-3), 0);
        flip.flags = FLAG_REDUCE_ONLY;
        let plan = p.plan(&flip, &PlanContext::new(dec!(2), quote()));
        let o = only(&plan);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(o.qty, dec!(2), "clamped to flat, not 5");
        assert!(o.reduce_only);

        // A genuine reduction passes through untouched.
        let mut shrink = sig(dec!(1), 0);
        shrink.flags = FLAG_REDUCE_ONLY;
        let plan = p.plan(&shrink, &PlanContext::new(dec!(3), quote()));
        assert_eq!(only(&plan).qty, dec!(2));

        // Reduce-only from flat can only ever be a no-op.
        let plan = p.plan(&grow, &PlanContext::flat(quote()));
        assert_eq!(plan.no_order, Some(NoOrder::ReduceOnlyWouldGrow));
    }

    #[test]
    fn a_replayed_signal_yields_the_same_cloid() {
        // A submit that times out has to be retried. A retry with a fresh id is a
        // second order and a doubled position; with the same id the venue de-dupes.
        let p = planner();
        let s = sig(dec!(3), 0);
        let first = p.plan(&s, &PlanContext::flat(quote()));
        let retry = p.plan(&s, &PlanContext::flat(quote()));
        assert_eq!(only(&first).cloid, only(&retry).cloid);

        // The identity is the signal's, not the plan's: a different position (so a
        // different order size) for the same signal keeps the id.
        let resized = p.plan(&s, &PlanContext::new(dec!(1), quote()));
        assert_eq!(only(&first).cloid, only(&resized).cloid);

        // And different signals do not collide on any of the three identity fields.
        let mut other_seq = s;
        other_seq.seq += 1;
        let mut other_ts = s;
        other_ts.ts_event += 1;
        let mut other_sym = s;
        other_sym.symbol_id += 1;
        let ids = [
            cloid_for(&s),
            cloid_for(&other_seq),
            cloid_for(&other_ts),
            cloid_for(&other_sym),
        ];
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn a_planner_cloid_cannot_collide_with_an_adopted_order_id() {
        // `OrderTracker` synthesizes a cloid for an order it never submitted by
        // widening the venue oid. If a planner id ever landed on one, a fill would be
        // attributed to the wrong order and both positions would be wrong.
        let c = cloid_for(&sig(dec!(1), 0)).get();
        assert_ne!(c, 0);
        assert_ne!(c & CLOID_PLANNER_TAG, 0);
        let adopted = u128::from(u64::MAX); // the widest possible synthesized id
        assert_eq!(adopted & CLOID_PLANNER_TAG, 0);

        // An all-zero-identity signal still produces a usable, non-zero id.
        let zero = Signal::target_position(0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(cloid_for(&zero).get(), CLOID_PLANNER_TAG);
    }

    #[test]
    fn urgency_maps_to_the_documented_tif_table() {
        let p = planner();
        let ctx = PlanContext::flat(quote());

        let l0 = p.plan(&sig(dec!(1), 0), &ctx);
        assert_eq!(only(&l0).tif, Tif::PostOnly);
        assert_eq!(only(&l0).price, Some(dec!(100)), "buy joins the bid");

        let l1 = p.plan(&sig(dec!(1), 1), &ctx);
        assert_eq!(only(&l1).tif, Tif::Gtc);
        assert_eq!(
            only(&l1).price,
            Some(dec!(100)),
            "same price, different TIF"
        );

        let l2 = p.plan(&sig(dec!(1), 2), &ctx);
        assert_eq!(only(&l2).tif, Tif::Gtc);
        assert_eq!(only(&l2).price, Some(dec!(101)), "crosses to the ask");

        let l3 = p.plan(&sig(dec!(1), 3), &ctx);
        assert_eq!(only(&l3).tif, Tif::Ioc);
        assert_eq!(only(&l3).price, Some(dec!(101.505)), "ask + 50 bps");

        // Selling mirrors it exactly.
        let s0 = p.plan(&sig(dec!(-1), 0), &ctx);
        assert_eq!(only(&s0).price, Some(dec!(101)), "sell joins the ask");
        let s3 = p.plan(&sig(dec!(-1), 3), &ctx);
        assert_eq!(only(&s3).price, Some(dec!(99.5)), "bid - 50 bps");
    }

    #[test]
    fn an_urgency_above_the_table_saturates_instead_of_being_dropped() {
        let p = planner();
        let ctx = PlanContext::flat(quote());
        for urgency in [3u8, 4, 100, 255] {
            let plan = p.plan(&sig(dec!(1), urgency), &ctx);
            assert_eq!(only(&plan).tif, Tif::Ioc, "urgency {urgency}");
        }
        assert_eq!(urgency_rule(255), URGENCY_TABLE[URGENCY_TABLE.len() - 1]);
    }

    #[test]
    fn a_price_band_clamps_the_limit_without_moving_it_through_the_market() {
        let p = planner();
        let ctx = PlanContext::flat(quote());

        // A buy's band is a ceiling: it can only lower the limit.
        let plan = p.plan(&banded(dec!(1), 0, dec!(99.5)), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(99.5)));

        // A band above the anchor is not a licence to pay more.
        let plan = p.plan(&banded(dec!(1), 0, dec!(500)), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(100)));

        // A sell's band is a floor.
        let plan = p.plan(&banded(dec!(-1), 0, dec!(102)), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(102)));

        // And it bounds the taker slippage, which is what it is for.
        let plan = p.plan(&banded(dec!(1), 3, dec!(101.1)), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(101.1)));
    }

    #[test]
    fn a_price_band_that_inverts_the_order_suppresses_it() {
        // An IOC buy limited below the ask cannot trade. Sending it burns a nonce and
        // a rate-limit credit and writes a log line claiming we tried; demoting it to
        // a resting order would invent an intent the strategy never expressed.
        let p = planner();
        let ctx = PlanContext::flat(quote());

        let plan = p.plan(&banded(dec!(1), 3, dec!(100.5)), &ctx);
        assert!(plan.orders.is_empty());
        assert_eq!(
            plan.no_order,
            Some(NoOrder::PriceBandUnfillable {
                limit: dec!(100.5),
                touch: dec!(101)
            })
        );

        // Same for the marketable GTC level, and for a sell.
        let plan = p.plan(&banded(dec!(1), 2, dec!(100.5)), &ctx);
        assert!(matches!(
            plan.no_order,
            Some(NoOrder::PriceBandUnfillable { .. })
        ));
        let plan = p.plan(&banded(dec!(-1), 3, dec!(100.5)), &ctx);
        assert!(matches!(
            plan.no_order,
            Some(NoOrder::PriceBandUnfillable { .. })
        ));

        // Exactly at the touch still crosses.
        let plan = p.plan(&banded(dec!(1), 2, dec!(101)), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(101)));

        // A band is only a band when non-zero; 0 means "no band" (schema.toml).
        let plan = p.plan(&sig(dec!(1), 3), &ctx);
        assert_eq!(only(&plan).price, Some(dec!(101.505)));

        // A negative band is not a price.
        let plan = p.plan(&banded(dec!(1), 0, dec!(-5)), &ctx);
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BadPriceBand { band: dec!(-5) })
        );
    }

    #[test]
    fn a_passive_band_below_the_market_still_rests_rather_than_being_suppressed() {
        // The suppression rule is about contradicting an *aggressive* intent. A
        // passive buy that the band pushes deeper into the book is exactly what the
        // strategy asked for, and refusing it would be us overriding a valid intent.
        let plan = planner().plan(&banded(dec!(1), 0, dec!(50)), &PlanContext::flat(quote()));
        let o = only(&plan);
        assert_eq!(o.price, Some(dec!(50)));
        assert_eq!(o.tif, Tif::PostOnly);
    }

    #[test]
    fn a_missing_or_crossed_quote_produces_no_order() {
        // Pricing off a locked, crossed or one-sided book makes an order aggressive
        // or passive by accident. Fail closed instead.
        let p = planner();
        let s = sig(dec!(1), 0);
        let no_quote = PlanContext {
            now: 0,
            position: Decimal::ZERO,
            quote: None,
            working: &[],
            precision: Precision::Unconstrained,
        };
        assert_eq!(p.plan(&s, &no_quote).no_order, Some(NoOrder::NoQuote));

        for bad in [
            Quote::new(dec!(101), dec!(100)), // crossed
            Quote::new(dec!(100), dec!(100)), // locked
            Quote::new(dec!(0), dec!(100)),   // no bid
        ] {
            assert_eq!(
                p.plan(&s, &PlanContext::flat(bad)).no_order,
                Some(NoOrder::NoQuote),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn a_matching_working_order_is_left_resting_instead_of_replaced() {
        // Cancel/replace costs queue position: a replaced post-only order goes to the
        // back of its price level. Re-issuing an identical order is a strict loss.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote()).with_working(&working);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert!(plan.is_noop());
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyWorking));
    }

    #[test]
    fn an_order_whose_tif_nobody_can_vouch_for_is_replaced_rather_than_assumed_to_be_ours() {
        // The venue's order list carries no TIF and no reduce-only flag, so a caller
        // projecting an *adopted* order has nothing to fill them in from. If the
        // exception accepted an invention, a reduce-only order left over from a
        // previous incarnation would be mistaken for the opening order we are about to
        // place, and the strategy would sit flat forever behind a quote that cannot
        // move it. `None` is what makes the comparison refuse.
        let unknown = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: None,
            reduce_only: None,
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote()).with_working(&unknown);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert_eq!(plan.cancels.len(), 1, "cancelled, not left resting");
        assert_eq!(only(&plan).qty, dec!(1));

        // Either field alone being unknown is enough to refuse. The reduce-only flag
        // is the one that strands a strategy flat, so it must not be the one a partial
        // answer skips over.
        let half_known = [WorkingOrder {
            tif: Some(Tif::PostOnly),
            reduce_only: None,
            ..unknown[0]
        }];
        let ctx = PlanContext::flat(quote()).with_working(&half_known);
        assert_eq!(planner().plan(&sig(dec!(1), 0), &ctx).cancels.len(), 1);
    }

    #[test]
    fn an_adopted_order_whose_tif_is_known_keeps_its_queue_position_across_a_restart() {
        // The debt this pays. `placed_by_us` could only ever say *no*, so every order a
        // restarted session inherited was cancelled and replaced — a strict loss of
        // queue position on every order, paid on every restart, for a field the
        // process no longer had rather than a field that was wrong. Carrying the real
        // TIF means an inherited order that *is* the order we would place stays where
        // it is.
        let inherited = [WorkingOrder {
            cloid: Cloid::new(0xFACE), // synthesized from a venue oid: not ours
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote()).with_working(&inherited);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert!(plan.is_noop(), "left alone: {plan:?}");
        assert_eq!(plan.no_order, Some(NoOrder::AlreadyWorking));

        // …and a reduce-only order still cannot be mistaken for the opening one. This
        // is the case the old flag protected by refusing everything, and it is still
        // refused now that the field is real instead of guessed.
        let reduce_only = [WorkingOrder {
            reduce_only: Some(true),
            ..inherited[0]
        }];
        let ctx = PlanContext::flat(quote()).with_working(&reduce_only);
        assert_eq!(planner().plan(&sig(dec!(1), 0), &ctx).cancels.len(), 1);
    }

    #[test]
    fn a_working_order_past_its_age_bound_is_replaced_however_well_it_matches() {
        // The leave-it-resting exception has no natural end: as long as the strategy
        // keeps asking for the same target, the same order rests. That was tolerable
        // while only orders *this* process placed could take the exception. It is not
        // now that an inherited one can, because an order nobody in this process ever
        // decided to place could otherwise outlive every decision that followed it.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 1_000_000_000,
        }];
        let p = Planner::new(PlannerConfig {
            max_order_age_ms: 60_000,
            ..PlannerConfig::default()
        });

        // A minute later to the nanosecond is still inside the bound: the order that
        // has held its place for exactly as long as it was allowed keeps it.
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(61_000_000_000);
        assert_eq!(
            p.plan(&sig(dec!(1), 0), &ctx).no_order,
            Some(NoOrder::AlreadyWorking)
        );

        // Past it, the identical order is cancelled and replaced.
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(61_000_000_001);
        let plan = p.plan(&sig(dec!(1), 0), &ctx);
        assert_eq!(plan.cancels.len(), 1, "pulled: {plan:?}");
        assert_eq!(only(&plan).qty, dec!(1));

        // And `0` is an operator saying "no bound", not "expire immediately" — the
        // same asymmetry `ttl_ms == 0` settled, for the same reason: zero is what an
        // unset field contains, and a field nobody wrote must not stop a strategy.
        let unbounded = Planner::new(PlannerConfig {
            max_order_age_ms: 0,
            ..PlannerConfig::default()
        });
        assert_eq!(
            unbounded.plan(&sig(dec!(1), 0), &ctx).no_order,
            Some(NoOrder::AlreadyWorking)
        );
    }

    #[test]
    fn a_signal_that_states_no_order_lifetime_is_governed_by_the_operators_ceiling() {
        // The case that inverts if the clamp is written as `requested.min(ceiling)`:
        // `min(0, 60_000)` is 0, so a strategy that never set the field — which is
        // every strategy that exists, and the Python default — would have every order
        // it placed cancelled on the very next pass. Zero is the value of a field
        // nobody wrote, so it has to resolve to the ceiling BEFORE the comparison, and
        // never by it (ADR-0020 §4, the same argument `ttl_ms` settled).
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let p = Planner::new(PlannerConfig {
            max_order_age_ms: 60_000,
            ..PlannerConfig::default()
        });
        let silent = sig(dec!(1), 0);
        assert_eq!(silent.max_order_age_ms, 0, "the field nobody wrote");

        // One second old, against a ceiling of sixty. An order a second into its life
        // is not one anybody meant to pull.
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(1_000_000_000);
        assert_eq!(
            p.plan(&silent, &ctx).no_order,
            Some(NoOrder::AlreadyWorking),
            "a silent signal must not read as a zero-length lifetime"
        );

        // …and the ceiling really does still bind it, so "defer to the operator" is not
        // quietly "no bound at all".
        let ctx = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(61_000_000_000);
        assert_eq!(p.plan(&silent, &ctx).cancels.len(), 1);
    }

    #[test]
    fn a_strategy_can_ask_for_a_shorter_order_lifetime_but_never_a_longer_one() {
        // The asymmetry the operator's ceiling exists for, now that the wire carries a
        // per-signal field: the strategy knows how long its own quote is worth having
        // out, and the operator answers for the fills. The shortest lifetime anyone
        // actually expressed is the one that binds.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let p = Planner::new(PlannerConfig {
            max_order_age_ms: 60_000,
            ..PlannerConfig::default()
        });
        let at = |now: Nanos| {
            PlanContext::flat(quote())
                .with_working(&working)
                .with_now(now)
        };
        let asking = |ms: u32| {
            let mut s = sig(dec!(1), 0);
            s.max_order_age_ms = ms;
            s
        };

        // Five seconds requested under a sixty-second ceiling: the request binds, and
        // an order six seconds old goes even though the operator would have kept it.
        assert_eq!(
            p.plan(&asking(5_000), &at(4_000_000_000)).no_order,
            Some(NoOrder::AlreadyWorking)
        );
        assert_eq!(p.plan(&asking(5_000), &at(6_000_000_000)).cancels.len(), 1);
        assert_eq!(
            p.plan(&sig(dec!(1), 0), &at(6_000_000_000)).no_order,
            Some(NoOrder::AlreadyWorking),
            "the same instant, under the operator's ceiling alone"
        );

        // An hour requested under the same ceiling: the ceiling binds, and asking for
        // longer buys nothing. A strategy must not be able to raise its own limit above
        // what the process answering for the fills configured.
        assert_eq!(
            p.plan(&asking(3_600_000), &at(61_000_000_000))
                .cancels
                .len(),
            1
        );

        // An operator who set no ceiling has not thereby overridden a strategy that
        // asked for a short-lived quote — `0` there means "I set no bound", not "no
        // bound may be set". Read the other way, an explicit request would be silently
        // discarded by an operator who simply had no opinion.
        let no_ceiling = Planner::new(PlannerConfig {
            max_order_age_ms: 0,
            ..PlannerConfig::default()
        });
        assert_eq!(
            no_ceiling
                .plan(&asking(5_000), &at(6_000_000_000))
                .cancels
                .len(),
            1,
            "the strategy's own request still binds"
        );
        // And with neither side expressing one, there is no bound to apply.
        assert_eq!(
            no_ceiling
                .plan(&sig(dec!(1), 0), &at(86_400_000_000_000))
                .no_order,
            Some(NoOrder::AlreadyWorking)
        );
    }

    #[test]
    fn the_age_bound_reads_the_context_clock_and_never_a_wall_clock() {
        // The property that makes the whole thing replayable: `plan` takes the pass's
        // event time as an input, so the same signal against the same context re-plans
        // to the same order — cancels, cloid and all — however long ago the recording
        // was made. A `SystemTime::now()` in here would make a replayed session pull
        // orders the live one kept.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let p = planner();
        let s = sig(dec!(1), 0);
        let old = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(600 * 1_000_000_000);
        assert_eq!(p.plan(&s, &old), p.plan(&s, &old), "same inputs, same plan");
        assert_eq!(
            p.plan(&s, &old).cancels.len(),
            1,
            "ten minutes against a one-minute default"
        );

        // The per-signal field is on the same footing: it is read off the record, so it
        // moves with the record and not with when the record is re-planned. Ten runs of
        // one signal against one context are one answer.
        let mut brief = s;
        brief.max_order_age_ms = 5_000;
        let young = PlanContext::flat(quote())
            .with_working(&working)
            .with_now(4_000_000_000);
        let first = p.plan(&brief, &young);
        for _ in 0..9 {
            assert_eq!(p.plan(&brief, &young), first);
        }
        assert_eq!(first.no_order, Some(NoOrder::AlreadyWorking));

        // And the identity of the order never moved with any of it. A cloid that
        // depended on the age would make a replayed session address a different order
        // at the venue than the live one did — the parity harness diffs these.
        assert_eq!(only(&p.plan(&s, &old)).cloid, cloid_for(&s));
        assert_eq!(
            only(&p.plan(&brief, &old)).cloid,
            cloid_for(&brief),
            "and the age field is not part of the identity either"
        );
        assert_eq!(
            cloid_for(&s),
            cloid_for(&brief),
            "the id is the signal's identity, and a lifetime is content, not identity"
        );
    }

    #[test]
    fn a_target_that_barely_moved_leaves_the_order_resting_instead_of_churning() {
        // ADR-0014's own minus column, reachable since ADR-0020 joined the path: a
        // strategy nudging its target every tick cancel/replaces every tick, and each
        // replacement puts the post-only order at the back of its price level for a
        // change too small to have been worth it. `min_order_qty` does not reach this —
        // the *delta* here is a whole unit, entirely respectable; it is the difference
        // from what is already resting that is dust.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        // 50 bps of the resting size: half a percent of one unit.
        let p = Planner::new(PlannerConfig {
            noop_band_bps: 50,
            ..PlannerConfig::default()
        });
        let ctx = PlanContext::flat(quote()).with_working(&working);

        let plan = p.plan(&sig(dec!(1.004), 0), &ctx);
        assert!(plan.is_noop(), "left alone: {plan:?}");
        assert_eq!(
            plan.no_order,
            Some(NoOrder::WithinNoOpBand {
                resting: dec!(1),
                wanted: dec!(1.004)
            }),
            "its own reason, because it is a knob's effect and not 'nothing changed'"
        );
        // The band is symmetric: a target slightly *below* what is resting is the same
        // trade, and leaving it means carrying at most `noop_band_bps` of the resting
        // size as unintended exposure until the next decision that clears the band.
        assert!(p.plan(&sig(dec!(0.996), 0), &ctx).is_noop());

        // Past the band the order is replaced, and the delta is still the delta.
        let plan = p.plan(&sig(dec!(1.006), 0), &ctx);
        assert_eq!(plan.cancels.len(), 1);
        assert_eq!(only(&plan).qty, dec!(1.006), "not 0.006: the cancel lands");

        // And off by default, so a deployment cannot acquire a position error because
        // somebody changed a default.
        assert_eq!(PlannerConfig::default().noop_band_bps, 0);
        assert_eq!(
            planner().plan(&sig(dec!(1.004), 0), &ctx).cancels.len(),
            1,
            "unbanded, this is the churn the knob exists to stop"
        );
    }

    #[test]
    fn the_no_op_band_forgives_a_size_and_never_a_price_or_a_reduce_only() {
        // Two things the band must not reach. A price that moved means the resting
        // order is at a price we would no longer choose — that is a stale quote, which
        // is exactly what somebody else's taker is looking for, and no amount of queue
        // position is worth one. And "get me out" that stops 20 bps short of flat has
        // not got us out: the residue is a position nobody decided to hold, and the
        // band trades exposure error for queue position it has no reason to want here.
        let resting = WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        };
        let p = Planner::new(PlannerConfig {
            noop_band_bps: 500, // a deliberately wide 5%
            ..PlannerConfig::default()
        });

        // The size is inside the band; the price is not the price we would send now.
        let moved = [resting];
        let ctx = PlanContext::flat(Quote::new(dec!(99), dec!(101))).with_working(&moved);
        let plan = p.plan(&sig(dec!(1.01), 0), &ctx);
        assert_eq!(plan.cancels.len(), 1, "a stale price is never banded");
        assert_eq!(only(&plan).price, Some(dec!(99)));

        // A reduce-only close, with a resting reduce-only order the same distance away.
        let closing = [WorkingOrder {
            side: Side::Sell,
            price: Some(dec!(101)),
            remaining_qty: dec!(1.01),
            reduce_only: Some(true),
            ..resting
        }];
        let mut close = sig(dec!(0), 0);
        close.flags = FLAG_CLOSE;
        let ctx = PlanContext::new(dec!(1), quote()).with_working(&closing);
        let plan = p.plan(&close, &ctx);
        assert_eq!(
            plan.cancels.len(),
            1,
            "a flatten is never approximately done"
        );
        let o = only(&plan);
        assert_eq!(o.qty, dec!(1));
        assert!(o.reduce_only);
    }

    #[test]
    fn a_changed_target_cancels_the_working_orders_before_replacing_them() {
        let working = [
            WorkingOrder {
                cloid: Cloid::new(5),
                symbol_id: SymbolId::new(SYM),
                side: Side::Buy,
                price: Some(dec!(100)),
                remaining_qty: dec!(1),
                tif: Some(Tif::PostOnly),
                reduce_only: Some(false),
                placed_ts: 0,
            },
            WorkingOrder {
                cloid: Cloid::new(6),
                symbol_id: SymbolId::new(SYM),
                side: Side::Buy,
                price: Some(dec!(99)),
                remaining_qty: dec!(1),
                tif: Some(Tif::PostOnly),
                reduce_only: Some(false),
                placed_ts: 0,
            },
        ];
        let ctx = PlanContext::flat(quote()).with_working(&working);
        let plan = planner().plan(&sig(dec!(4), 0), &ctx);
        assert_eq!(plan.cancels.len(), 2, "both are superseded");
        let o = only(&plan);
        assert_eq!(
            o.qty,
            dec!(4),
            "the delta is against the filled position, which is what the cancels leave"
        );

        // Two working orders, one of which matches, is still a full replace — the
        // other has to go, and the delta assumed none of them were live.
        let ctx = PlanContext::flat(quote()).with_working(&working);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert_eq!(plan.cancels.len(), 2);
        assert_eq!(only(&plan).qty, dec!(1));
    }

    #[test]
    fn another_symbols_working_orders_are_never_cancelled() {
        // The planner is called per signal; the caller may hand it the whole book.
        let working = [WorkingOrder {
            cloid: Cloid::new(5),
            symbol_id: SymbolId::new(SYM + 1),
            side: Side::Buy,
            price: Some(dec!(100)),
            remaining_qty: dec!(1),
            tif: Some(Tif::PostOnly),
            reduce_only: Some(false),
            placed_ts: 0,
        }];
        let ctx = PlanContext::flat(quote()).with_working(&working);
        let plan = planner().plan(&sig(dec!(1), 0), &ctx);
        assert!(plan.cancels.is_empty());
        assert_eq!(only(&plan).symbol_id, SymbolId::new(SYM));
    }

    #[test]
    fn the_dust_floor_never_strands_a_position_it_could_have_closed() {
        // Measured live on 2026-07-27, twice in 59 minutes. A closing buy for 0.0003 BTC
        // partially filled 0.00017 at a maker price; the sweeper pulled the remainder a
        // minute later, exactly as designed; and 0.00013 BTC — about $8.50 — was left
        // with **no order that could ever express it**, because the session's
        // `min_order_qty` was 0.00016. The planner refused every close, so the re-quote
        // correctly placed nothing and the status line could only name the state
        // (`STRANDED POSITION`). Only an operator could clear it.
        //
        // `min_order_qty` is a churn bound and a close is not churn: it converges by
        // construction, there is at most one of it, and the sweeper and the re-quote
        // budget bound how often it can come back. The floor still binds on everything
        // that is not a close — the test below this one.
        let p = Planner::new(PlannerConfig {
            min_order_qty: dec!(0.00016),
            ..PlannerConfig::default()
        });
        let s = btc();
        let ctx = PlanContext::new(dec!(0.00013), quote()).with_precision(Precision::Known(&s));

        // The exact shape the live run produced: an ordinary target of zero, which is
        // what `--flatten-only` emits. It carries neither FLAG_CLOSE nor
        // FLAG_REDUCE_ONLY, so a flag-keyed exemption would miss it — which is why the
        // condition is arithmetic.
        let plan = p.plan(&sig(dec!(0), 0), &ctx);
        let o = only(&plan);
        assert_eq!(o.side, Side::Sell);
        assert_eq!(o.qty, dec!(0.00013), "the whole residue, not a rounded one");
        assert!(
            o.qty * o.price.unwrap() < dec!(10),
            "and it really is under the venue's minimum too: {:?}",
            o.qty * o.price.unwrap()
        );

        // A close the other way round — a short residue bought back — is the same case.
        let short = PlanContext::new(dec!(-0.00013), quote()).with_precision(Precision::Known(&s));
        assert_eq!(only(&p.plan(&sig(dec!(0), 0), &short)).side, Side::Buy);
    }

    #[test]
    fn the_dust_floor_still_refuses_a_partial_reduction_that_leaves_a_position_behind() {
        // The other half, and the reason the exemption is narrow. Shrinking 0.001 to
        // 0.0009 leaves 0.0009 on the books, so the next signal can ask for the same
        // dust again and again — which is the oscillation `min_order_qty` exists for.
        // Only an order that reaches *zero* is exempt, because only that one cannot
        // recur.
        let p = Planner::new(PlannerConfig {
            min_order_qty: dec!(0.00016),
            ..PlannerConfig::default()
        });
        let s = btc();
        let ctx = PlanContext::new(dec!(0.001), quote()).with_precision(Precision::Known(&s));
        let plan = p.plan(&sig(dec!(0.0009), 0), &ctx);
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BelowMinQty {
                delta: dec!(0.0001),
                min: dec!(0.00016)
            })
        );

        // And a *flat* account asking for a dust position is still refused: nothing is
        // being closed, so nothing is exempt.
        let flat = PlanContext::flat(quote()).with_precision(Precision::Known(&s));
        assert!(matches!(
            p.plan(&sig(dec!(0.0001), 0), &flat).no_order,
            Some(NoOrder::BelowMinQty { .. })
        ));
    }

    #[test]
    fn a_dust_delta_produces_no_order_instead_of_churning() {
        let p = Planner::new(PlannerConfig {
            min_order_qty: dec!(0.001),
            ..PlannerConfig::default()
        });
        let plan = p.plan(
            &sig(dec!(1.0000001), 0),
            &PlanContext::new(dec!(1), quote()),
        );
        assert_eq!(
            plan.no_order,
            Some(NoOrder::BelowMinQty {
                delta: dec!(0.0000001),
                min: dec!(0.001)
            })
        );
        // At the floor it goes through.
        let plan = p.plan(&sig(dec!(1.001), 0), &PlanContext::new(dec!(1), quote()));
        assert_eq!(only(&plan).qty, dec!(0.001));
    }

    #[test]
    fn the_wire_target_reaches_the_order_without_passing_through_a_float() {
        // 0.1 has no exact binary form; an f64 round trip would leave a residue that
        // makes "at target" never quite true and re-sends dust on every signal.
        let p = planner();
        let s = sig(dec!(0.1), 0);
        assert_eq!(
            only(&p.plan(&s, &PlanContext::flat(quote()))).qty,
            dec!(0.1)
        );
        assert_eq!(
            p.plan(&s, &PlanContext::new(dec!(0.1), quote())).no_order,
            Some(NoOrder::AlreadyAtTarget),
            "three tenths of a hair is still exactly zero"
        );
    }
}
