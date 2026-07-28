//! `GET /fapi/v1/exchangeInfo` → the [`InstrumentTable`], and the symbols it refuses.
//!
//! This is where ADR-0025's central claim gets tested against a venue it was not
//! written for. Hyperliquid publishes one number per asset, `szDecimals`, and its
//! price rule is a *significant-figure* rule: at most five figures and at most
//! `6 - szDecimals` decimal places, integers exempt. Binance publishes a
//! `PRICE_FILTER.tickSize` and its rule is a *tick* rule: a price is legal iff it is
//! an exact multiple. Those are different models, and ADR-0025 chose one struct with
//! two fields over a two-variant enum on the argument that a third venue must not
//! force a `match` arm anywhere.
//!
//! It holds. `tick_at(px) = max(increment, sig_quantum(px))` with `sig_figs: None`
//! degenerates to `increment`, so Binance sets one field, sets no arm, and every
//! consumer above the adapter is untouched. The whole mapping is:
//!
//! ```text
//! PRICE_FILTER.tickSize  ->  PriceGrid::increment
//! LOT_SIZE.stepSize      ->  SizeGrid::step
//! MIN_NOTIONAL.notional  ->  InstrumentSpec::min_notional
//! ```
//!
//! and the third line is the one ADR-0025 predicted in its own minus column, where it
//! called Hyperliquid's hard-coded `MIN_ORDER_NOTIONAL_USD` "a population weakness"
//! and noted that "a CEX publishes its own in `exchangeInfo`". It does, and it is
//! **per symbol**: 50 USDT on BTCUSDT, 20 on ETHUSDT, 5 on most, 0.001 on the
//! BTC-quoted `ETHBTC`. A constant would be wrong four ways in one response.
//!
//! What does **not** fit is documented at [`decode_exchange_info`], and pinned by the
//! test named `an_order_the_port_calls_valid_can_still_break_a_filter_it_cannot_hold`.

use axon_core::Decimal;
use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid, SpecError};
use serde::Deserialize;
use std::str::FromStr;

use crate::symbols::{SymbolTable, SymbolTableError};

/// Why a symbol in the response did not get an id or a grid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SkipReason {
    /// Not a perpetual: a quarterly, a weekly, or a delivering contract.
    ///
    /// Refused by name rather than approximated, the way ADR-0011 refuses spot. A
    /// dated future has an expiry and a basis that decays into it, and the normalized
    /// vocabulary has no word for either — so treating one as a perp yields a
    /// perfectly plausible `Ticker` whose funding field is meaningless and whose
    /// carry is a different quantity entirely.
    #[error("contract type {contract_type} is not a perpetual")]
    NotPerpetual { contract_type: String },
    /// Listed but not matching: `PENDING_TRADING`, `SETTLING`, `DELIVERING`,
    /// `PRE_SETTLE`.
    #[error("status is {status}, not TRADING")]
    NotTrading { status: String },
    /// The response carried no `PRICE_FILTER`, `LOT_SIZE` or `MIN_NOTIONAL`.
    #[error("no {0} filter")]
    MissingFilter(&'static str),
    /// A filter's numbers do not describe a grid — `tickSize: "0"` is the real case,
    /// and every symbol carrying it on the captured universe was `PENDING_TRADING`.
    #[error("{filter} is not a grid: {reason}")]
    NotAGrid {
        filter: &'static str,
        reason: SpecError,
    },
    /// A filter value that is not a number at all.
    #[error("{filter}.{field} is not a number: {value:?}")]
    BadNumber {
        filter: &'static str,
        field: &'static str,
        value: String,
    },
}

/// One symbol the universe decode did not keep, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub symbol: String,
    pub reason: SkipReason,
}

/// One `exchangeInfo` read, both halves — the same shape Hyperliquid's
/// `decode_universe` returns, for the same reason: the grid and the identity of an
/// instrument must come out of a single response, or an order can be addressed by one
/// read's identity at a size computed from another's.
///
/// The third field is new, and it is the venue difference. Hyperliquid's `meta`
/// universe is ~210 curated perps, so an asset that will not build a spec is an
/// anomaly and `decode_universe` fails the **whole** decode over it. Binance's
/// `exchangeInfo` is 730 rows in one response — perpetuals beside quarterlies,
/// `TRADING` beside `PENDING_TRADING` and `SETTLING` — and several of them carry, in
/// perfectly good faith, values that are not a grid. Failing whole would mean one
/// pending listing nobody asked to trade takes the adapter down at startup.
///
/// So this skips, and the skip is **recorded per symbol with its reason** rather than
/// counted. ADR-0025 is explicit that the danger of skipping is silence — "an asset
/// silently absent from the table fails closed later with nothing attached saying
/// why" — and [`Universe::why_skipped`] is what removes the silence: a startup that
/// cannot find a configured symbol can say which of five different things happened
/// to it.
#[derive(Debug, Clone)]
pub struct Universe {
    pub symbols: SymbolTable,
    pub instruments: InstrumentTable,
    pub skipped: Vec<Skipped>,
}

impl Universe {
    /// Why a symbol the caller asked for is not in this universe. `None` if it is
    /// present, or if the venue never mentioned it at all.
    pub fn why_skipped(&self, symbol: &str) -> Option<&SkipReason> {
        self.skipped
            .iter()
            .find(|s| s.symbol == symbol)
            .map(|s| &s.reason)
    }
}

/// Why an `exchangeInfo` body could not be read at all.
///
/// Distinct from [`SkipReason`], and the distinction is the point: a malformed body
/// is a failure of the *response*, and continuing past it would build a universe from
/// whatever happened to parse. A skipped symbol is a successful read of a row we
/// decline to trade.
#[derive(Debug, thiserror::Error)]
pub enum ExchangeInfoError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("symbol ids: {0}")]
    Symbols(#[from] SymbolTableError),
    /// Every row was skipped. Almost certainly the wrong endpoint (spot's
    /// `exchangeInfo` parses but has no `contractType`), and a session that started on
    /// an empty universe would refuse every order for a reason no message explains.
    #[error("no tradable perpetual in a response of {rows} rows - wrong endpoint?")]
    Empty { rows: usize },
}

// ── on-wire shapes ───────────────────────────────────────────────────────────

/// A `filters[]` entry. Internally tagged on `filterType`, with everything we do not
/// read collapsing into [`Filter::Other`] — the captured response carries seven
/// distinct filter types and three of them matter.
#[derive(Deserialize)]
#[serde(tag = "filterType")]
enum Filter {
    #[serde(rename = "PRICE_FILTER")]
    Price {
        #[serde(rename = "tickSize")]
        tick_size: String,
    },
    #[serde(rename = "LOT_SIZE")]
    Lot {
        #[serde(rename = "stepSize")]
        step_size: String,
    },
    #[serde(rename = "MIN_NOTIONAL")]
    MinNotional { notional: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct SymbolInfo {
    symbol: String,
    #[serde(rename = "contractType")]
    contract_type: String,
    status: String,
    /// The listing timestamp `SymbolTable::assign` orders on. Required, not
    /// defaulted: a missing one would sort to the front and take id 0, which is the
    /// one id a mistake is most likely to be silently correct against.
    #[serde(rename = "onboardDate")]
    onboard_date: i64,
    filters: Vec<Filter>,
}

#[derive(Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

/// The `contractType` this adapter trades. Everything else is a
/// [`SkipReason::NotPerpetual`].
///
/// `TRADIFI_PERPETUAL` (tokenised-equity perps, 37 of them on the captured universe)
/// is excluded too. It is shaped like a perpetual and funds like one, and it also
/// halts outside equity hours — a market-data gap that reads exactly like a dead
/// socket. Admitting it is a decision about market hours, not about parsing, and it
/// is not one this increment is making.
const PERPETUAL: &str = "PERPETUAL";

/// The only `status` in which an instrument has a real grid.
const TRADING: &str = "TRADING";

// ── decode ───────────────────────────────────────────────────────────────────

/// Decode `exchangeInfo` into ids, grids, and the list of symbols it declined.
///
/// **Four Binance filters have no home on the port, and all four are reachable.**
/// They are listed here rather than in the ADR alone because this is the function a
/// reader reaches for when an order comes back `-1013 Filter failure`:
///
/// 1. **`LOT_SIZE.minQty`.** [`SizeGrid`] holds one increment and no floor. Six
///    symbols on the captured universe have `minQty != stepSize` — `TUTUSDT` steps by
///    0.1 with a minimum of 1 — so a size of 0.5 is on the lot, may clear the $5
///    minimum notional, and is still rejected.
/// 2. **`LOT_SIZE.maxQty` and `PRICE_FILTER.minPrice`/`maxPrice`.** Named as out of
///    scope in ADR-0025 (#6) with no venue behind them. There is one now.
/// 3. **`MARKET_LOT_SIZE`.** A *second* lot that applies only to market orders, and
///    it genuinely differs: `ARKUSDT` steps by 3 as a limit and by 1 as a market,
///    with maxima of 3 333 and 1 000 000. [`InstrumentSpec`] has one [`SizeGrid`],
///    and this venue has native market orders, so the case is live rather than
///    theoretical.
/// 4. **`PERCENT_PRICE`.** A price band relative to the mark (±5% on most symbols).
///    Arguably risk rather than precision — but the planner's own `price_band` is the
///    thing that would collide with it, and neither knows about the other.
///
/// The first is the one that costs money quietly: `InstrumentSpec::check` accepts the
/// order, the encoder accepts it, and the venue refuses it — which is precisely the
/// "well-formed action, valid signature, refusal" shape ADR-0025 exists to make
/// impossible.
pub fn decode_exchange_info(raw: &str) -> Result<Universe, ExchangeInfoError> {
    let info: ExchangeInfo = serde_json::from_str(raw)?;
    let rows = info.symbols.len();

    // Two passes, because ids come from an ordering over the *kept* rows and a grid is
    // keyed by id. Splitting them is also what keeps a skipped row from consuming an
    // id: an id issued to something with no grid is one more way for a lookup to
    // return `Unknown` for a symbol that was never tradable.
    let mut kept: Vec<(SymbolInfo, PriceGrid, SizeGrid, Option<Decimal>)> = Vec::new();
    let mut skipped: Vec<Skipped> = Vec::new();
    for s in info.symbols {
        match grids_for(&s) {
            Ok((price, size, min_notional)) => kept.push((s, price, size, min_notional)),
            Err(reason) => skipped.push(Skipped {
                symbol: s.symbol,
                reason,
            }),
        }
    }
    if kept.is_empty() {
        return Err(ExchangeInfoError::Empty { rows });
    }

    let symbols = SymbolTable::assign(
        kept.iter()
            .map(|(s, ..)| (s.symbol.clone(), s.onboard_date)),
    )?;
    let mut instruments = InstrumentTable::new();
    for (s, price, size, min_notional) in &kept {
        let symbol_id = symbols
            .id(&s.symbol)
            .expect("every kept symbol was just assigned an id");
        instruments.insert(InstrumentSpec {
            symbol_id,
            price: *price,
            size: *size,
            min_notional: *min_notional,
        });
    }

    Ok(Universe {
        symbols,
        instruments,
        skipped,
    })
}

/// One symbol's grids, or the reason it has none.
fn grids_for(s: &SymbolInfo) -> Result<(PriceGrid, SizeGrid, Option<Decimal>), SkipReason> {
    if s.contract_type != PERPETUAL {
        return Err(SkipReason::NotPerpetual {
            contract_type: s.contract_type.clone(),
        });
    }
    if s.status != TRADING {
        return Err(SkipReason::NotTrading {
            status: s.status.clone(),
        });
    }

    let mut tick = None;
    let mut step = None;
    let mut notional = None;
    for f in &s.filters {
        match f {
            Filter::Price { tick_size } => tick = Some(tick_size),
            Filter::Lot { step_size } => step = Some(step_size),
            Filter::MinNotional { notional: n } => notional = Some(n),
            Filter::Other => {}
        }
    }

    let tick = tick.ok_or(SkipReason::MissingFilter("PRICE_FILTER"))?;
    let step = step.ok_or(SkipReason::MissingFilter("LOT_SIZE"))?;
    // Required, not defaulted to `None`. A symbol whose minimum notional we silently
    // treated as absent would let through orders the venue refuses, one per signal,
    // and ADR-0025's `min_notional: Option` means "this venue has none" — which is a
    // different sentence from "this response did not say".
    let notional = notional.ok_or(SkipReason::MissingFilter("MIN_NOTIONAL"))?;

    let price = PriceGrid::increment(dec("PRICE_FILTER", "tickSize", tick)?).map_err(|reason| {
        // `tickSize: "0"` is not hypothetical: `ELSAUSDT` carries it on the captured
        // universe, alongside `minPrice: "0"` and `pricePrecision: 0`, because the
        // symbol is `PENDING_TRADING` and has no grid yet. `PriceGrid::unconstrained`
        // would *accept* it and is exactly the wrong answer — it means "this venue has
        // no rules", and reading a venue's "not yet" as "no rules" is fail-open on the
        // one instrument we know least about.
        SkipReason::NotAGrid {
            filter: "PRICE_FILTER",
            reason,
        }
    })?;
    let size = SizeGrid::step(dec("LOT_SIZE", "stepSize", step)?).map_err(|reason| {
        SkipReason::NotAGrid {
            filter: "LOT_SIZE",
            reason,
        }
    })?;
    Ok((
        price,
        size,
        Some(dec("MIN_NOTIONAL", "notional", notional)?),
    ))
}

/// A filter string → [`Decimal`]. Never through a float: `0.000001` is a real tick on
/// this venue and a `f64` round trip of it is not `0.000001`.
fn dec(filter: &'static str, field: &'static str, s: &str) -> Result<Decimal, SkipReason> {
    Decimal::from_str(s).map_err(|_| SkipReason::BadNumber {
        filter,
        field,
        value: s.to_string(),
    })
}

/// Decode `GET /fapi/v1/fundingInfo` and stamp each symbol's period onto `symbols`.
///
/// Separate from [`decode_exchange_info`] because it is a separate request, and
/// separate requests are separate failures: a `fundingInfo` that times out must leave
/// a working universe behind, with every symbol on the documented eight-hour default,
/// rather than taking the instrument table down with it. What it costs is that the
/// two can disagree after a listing — the same seam ADR-0025 closed for `meta` by
/// insisting on one read. Here the venue forces two, and the mitigation is that the
/// second one only ever *narrows* an assumption we can already live with.
///
/// Symbols the table has never heard of are ignored: `fundingInfo` lists COIN-M pairs
/// (`BCHUSD_PERP`) beside the USD-M ones.
pub fn decode_funding_info(
    raw: &str,
    symbols: &mut SymbolTable,
) -> Result<usize, ExchangeInfoError> {
    #[derive(Deserialize)]
    struct Row {
        symbol: String,
        #[serde(rename = "fundingIntervalHours")]
        funding_interval_hours: i64,
    }
    let rows: Vec<Row> = serde_json::from_str(raw)?;
    let mut applied = 0;
    for r in rows {
        if symbols.id(&r.symbol).is_some() {
            symbols.set_funding_hours(&r.symbol, r.funding_interval_hours);
            applied += 1;
        }
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Side, SymbolId, Tif};
    use axon_providers::{OrderRequest, Precision};
    use rust_decimal_macros::dec as d;

    /// Seven symbol objects lifted **byte for byte** out of a
    /// `GET https://testnet.binancefuture.com/fapi/v1/exchangeInfo` response
    /// (2026-07-26), wrapped in that response's own envelope. Only the array is
    /// subsetted — every object, every field and every filter is the venue's own,
    /// including the ones that break.
    ///
    /// They were chosen because each one is a different way the port meets reality:
    /// BTCUSDT (tick 0.10, lot 0.0001, min notional 50), ETHUSDT (min notional 20),
    /// DOGEUSDT (whole-unit lot), ARKUSDT (**lot of 3** — not a power of ten, which
    /// Hyperliquid's `szDecimals` model cannot express at all), SPELLUSDT
    /// (`minQty` 100 against a `stepSize` of 1), ELSAUSDT (**`tickSize: "0"`**,
    /// `PENDING_TRADING`) and ETHBTC (quoted in BTC, minimum notional 0.001).
    const EXCHANGE_INFO: &str = include_str!("../testdata/exchange_info.subset.json");

    fn universe() -> Universe {
        decode_exchange_info(EXCHANGE_INFO).expect("the venue's own response")
    }

    fn spec<'a>(u: &'a Universe, symbol: &str) -> &'a InstrumentSpec {
        let id = u.symbols.id(symbol).unwrap_or_else(|| {
            panic!(
                "{symbol} is missing: {:?}",
                u.why_skipped(symbol).map(ToString::to_string)
            )
        });
        u.instruments.get(id).expect("a kept symbol has a grid")
    }

    #[test]
    fn a_fixed_tick_venue_needs_no_new_arm_anywhere_which_is_the_whole_layer_claim() {
        // ADR-0025 chose `PriceGrid { increment, sig_figs }` over a two-variant enum on
        // the argument that venue three would otherwise grow a `match` in every
        // consumer. This is venue two collecting on that bet: one constructor, no arm,
        // and `tick_at` is constant across price magnitudes where Hyperliquid's widens.
        let u = universe();
        let btc = spec(&u, "BTCUSDT");
        assert_eq!(btc.price.tick_at(d!(64346.70)), d!(0.10));
        assert_eq!(
            btc.price.tick_at(d!(1234.5)),
            d!(0.10),
            "a tick rule does not widen with the price the way a sig-fig rule does"
        );
        assert!(btc.price.is_valid(d!(64346.70)));
        assert!(
            !btc.price.is_valid(d!(64346.75)),
            "half a tick is off the grid even though it is only 7 significant figures"
        );
        assert!(
            btc.price.is_valid(d!(64346.7)),
            "and the same value at a different scale is the same price"
        );
        // Nine significant figures, which Hyperliquid refuses outright and this venue
        // is perfectly happy with as long as it lands on the tick.
        assert!(spec(&u, "DOGEUSDT").price.is_valid(d!(0.123450)));
    }

    #[test]
    fn the_minimum_notional_is_per_symbol_here_and_a_constant_would_be_wrong_four_ways() {
        // ADR-0025's minus column: `MIN_ORDER_NOTIONAL_USD` is a constant in our source
        // that Hyperliquid owns, and it called that "a population weakness" on the
        // grounds that a CEX publishes its own. It does, and one response disagrees
        // with itself four times over.
        let u = universe();
        assert_eq!(spec(&u, "BTCUSDT").min_notional, Some(d!(50)));
        assert_eq!(spec(&u, "ETHUSDT").min_notional, Some(d!(20)));
        assert_eq!(spec(&u, "DOGEUSDT").min_notional, Some(d!(5)));
        // And the units are the *quote* asset, not dollars. `ETHBTC` is quoted in BTC,
        // so its minimum is 0.001 BTC — a field typed as "quote currency" survives
        // that; one typed as USD would be out by five orders of magnitude.
        assert_eq!(spec(&u, "ETHBTC").min_notional, Some(d!(0.001)));
    }

    #[test]
    fn a_lot_of_three_is_expressible_here_and_would_not_be_under_a_decimals_model() {
        // `ARKUSDT` steps by 3 whole units. `szDecimals` — one integer, meaning
        // `10^-n` — cannot say that at all, and ADR-0025's decision to give `SizeGrid`
        // a general `step` constructor beside `decimals` is what makes it a one-liner
        // rather than a type change.
        let u = universe();
        let ark = spec(&u, "ARKUSDT");
        assert_eq!(ark.size.increment(), d!(3));
        assert!(ark.size.is_valid(d!(9)));
        assert!(!ark.size.is_valid(d!(10)));
        assert_eq!(ark.size.quantize(d!(10)), d!(9), "truncates toward zero");
        assert_eq!(ark.size.quantize(d!(-10)), d!(-9), "and so does a close");
    }

    #[test]
    fn a_tick_size_of_zero_is_refused_rather_than_read_as_a_venue_with_no_rules() {
        // The trap. `PriceGrid::unconstrained()` is also increment-zero, so mapping the
        // venue's `"0"` onto it type-checks, reads as tidy, and means "this venue has
        // no price rules" — fail-open on the one instrument we know least about. What
        // the venue means is "not yet": `ELSAUSDT` is `PENDING_TRADING` with
        // `minPrice: "0"` and `pricePrecision: 0`.
        let u = universe();
        assert_eq!(u.symbols.id("ELSAUSDT"), None);
        // …and it is refused as a *status*, before the zero tick is ever reached, which
        // is the more informative of the two messages.
        assert!(matches!(
            u.why_skipped("ELSAUSDT"),
            Some(SkipReason::NotTrading { status }) if status == "PENDING_TRADING"
        ));
    }

    #[test]
    fn one_ungriddable_row_skips_itself_instead_of_taking_the_whole_universe_down() {
        // The structural difference from Hyperliquid, and it is a difference in the
        // venue rather than in taste. `decode_universe` fails the entire `meta` decode
        // on one bad asset, which is right for ~210 curated perps. Binance answers with
        // 730 rows in one body — perpetuals beside quarterlies, TRADING beside
        // PENDING_TRADING — so failing whole would let one pending listing nobody asked
        // to trade stop the adapter from starting.
        let u = universe();
        assert_eq!(u.symbols.len(), 6, "six kept out of seven");
        assert_eq!(u.skipped.len(), 1);
        assert!(u.instruments.contains(u.symbols.id("BTCUSDT").unwrap()));
    }

    #[test]
    fn every_skipped_symbol_carries_the_reason_it_was_skipped() {
        // ADR-0025 on why skipping is dangerous: "an asset silently absent from the
        // table fails closed later with nothing attached saying why." A count would
        // reproduce exactly that. `why_skipped` is what lets a startup answer "you
        // configured ELSAUSDT and it is PENDING_TRADING" instead of "unknown symbol".
        let u = universe();
        let reason = u.why_skipped("ELSAUSDT").expect("recorded, not counted");
        assert!(
            reason.to_string().contains("PENDING_TRADING"),
            "the message has to name the venue's own word: {reason}"
        );
        assert_eq!(
            u.why_skipped("BTCUSDT"),
            None,
            "a kept symbol is not skipped"
        );
        assert_eq!(u.why_skipped("NOSUCHUSDT"), None, "nor is one never seen");
    }

    #[test]
    fn a_dated_contract_is_refused_by_name_rather_than_traded_as_a_perpetual() {
        // The same move ADR-0011 makes for spot. A quarterly has an expiry and a basis
        // that decays into it; decoded as a perp it yields a plausible `Ticker` whose
        // funding field describes a charge that instrument does not levy.
        let body = r#"{"symbols":[{"symbol":"BTCUSDT_260327","pair":"BTCUSDT",
            "contractType":"CURRENT_QUARTER","status":"TRADING","onboardDate":1700000000000,
            "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.10"},
                       {"filterType":"LOT_SIZE","stepSize":"0.001"},
                       {"filterType":"MIN_NOTIONAL","notional":"5"}]},
            {"symbol":"BTCUSDT","pair":"BTCUSDT","contractType":"PERPETUAL","status":"TRADING",
             "onboardDate":1569398400000,
             "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.10"},
                        {"filterType":"LOT_SIZE","stepSize":"0.001"},
                        {"filterType":"MIN_NOTIONAL","notional":"5"}]}]}"#;
        let u = decode_exchange_info(body).unwrap();
        assert_eq!(u.symbols.len(), 1);
        assert!(matches!(
            u.why_skipped("BTCUSDT_260327"),
            Some(SkipReason::NotPerpetual { contract_type }) if contract_type == "CURRENT_QUARTER"
        ));
    }

    #[test]
    fn a_symbol_with_no_minimum_notional_filter_is_skipped_rather_than_given_none() {
        // `min_notional: None` is a real sentence on the port — it means "this venue
        // has no minimum" — and it is not the same sentence as "this response did not
        // mention one". Defaulting would let sub-minimum orders through, one per
        // signal, each costing a rate credit and returning `-4164`.
        let body = r#"{"symbols":[{"symbol":"BTCUSDT","contractType":"PERPETUAL",
            "status":"TRADING","onboardDate":1569398400000,
            "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.10"},
                       {"filterType":"LOT_SIZE","stepSize":"0.001"}]}]}"#;
        let err = decode_exchange_info(body).unwrap_err();
        assert!(
            matches!(err, ExchangeInfoError::Empty { rows: 1 }),
            "got {err:?}"
        );
    }

    #[test]
    fn a_response_with_nothing_tradable_in_it_is_an_error_and_not_an_empty_universe() {
        // Spot's `exchangeInfo` parses against this shape and has no `contractType`, so
        // pointing the adapter at the wrong host yields zero perpetuals rather than a
        // parse failure. An empty universe would start, reconcile, print OK and refuse
        // every order for a reason nothing explains.
        let body = r#"{"symbols":[{"symbol":"BTCUSDT","contractType":"CURRENT_QUARTER",
            "status":"TRADING","onboardDate":1,"filters":[]}]}"#;
        assert!(matches!(
            decode_exchange_info(body),
            Err(ExchangeInfoError::Empty { rows: 1 })
        ));
    }

    #[test]
    fn an_order_the_port_calls_valid_can_still_break_a_filter_it_cannot_hold() {
        // **The finding.** `LOT_SIZE.minQty` has no home on `InstrumentSpec`, so
        // `SPELLUSDT` — `stepSize: 1`, `minQty: 100` — accepts a size of 1 through
        // every check this codebase has and the venue answers `-1013 Filter failure:
        // LOT_SIZE`. That is the exact shape ADR-0025 was written to eliminate: a
        // well-formed action, a valid signature, and a refusal that reads from inside
        // the process like a signing bug.
        //
        // This test asserts the gap rather than the fix, because the fix is two fields
        // on `SizeGrid` in a crate this workstream does not own. If it ever starts
        // failing, the gap was closed and this test should become its regression.
        let u = universe();
        let spell = spec(&u, "SPELLUSDT");
        let id = spell.symbol_id;
        assert_eq!(spell.size.increment(), d!(1));
        assert!(
            spell.size.is_valid(d!(1)),
            "on the lot, and 99 units below the venue's minimum quantity"
        );
        // Priced so the notional clears MIN_NOTIONAL as well: nothing on the port
        // objects.
        let req = OrderRequest::limit(id, Side::Buy, d!(1), d!(10), Tif::Gtc, Cloid::new(1));
        assert!(
            spell.check(&req).is_ok(),
            "the port has no `minQty`, so it cannot object"
        );
        assert!(matches!(
            u.instruments.precision(id),
            Precision::Known(s) if std::ptr::eq(s, spell)
        ));
    }

    #[test]
    fn funding_intervals_are_applied_by_symbol_and_foreign_ones_are_ignored() {
        // `fundingInfo` is a second request against a universe built by the first, and
        // it names COIN-M symbols this table has never heard of. Ignoring them is
        // correct; issuing them an id is not.
        let mut u = universe();
        let body = r#"[{"symbol":"BTCUSDT","adjustedFundingRateCap":"0.0075",
            "adjustedFundingRateFloor":"-0.0075","fundingIntervalHours":4,"disclaimer":true,
            "updateTime":null},
            {"symbol":"BCHUSD_PERP","adjustedFundingRateCap":"0.01725",
             "adjustedFundingRateFloor":"-0.01725","fundingIntervalHours":8,"disclaimer":true,
             "updateTime":null}]"#;
        let applied = decode_funding_info(body, &mut u.symbols).unwrap();
        assert_eq!(applied, 1, "only the symbol this universe holds");
        let btc = u.symbols.id("BTCUSDT").unwrap();
        assert_eq!(u.symbols.funding_interval(btc), 4 * 3_600 * 1_000_000_000);
        assert!(u.symbols.funding_is_published(btc));
        let eth = u.symbols.id("ETHUSDT").unwrap();
        assert!(
            !u.symbols.funding_is_published(eth),
            "and a symbol the response skipped keeps the assumed default"
        );
        assert_eq!(u.symbols.len(), 6, "no id was issued to BCHUSD_PERP");
    }

    #[test]
    fn ids_come_out_of_the_captured_universe_in_listing_order() {
        // Pinned against the venue's own `onboardDate`s so a change to the ordering
        // rule cannot land quietly: it would silently re-map every recorded capture.
        let u = universe();
        // BTCUSDT, DOGEUSDT and ETHUSDT were all onboarded in the same millisecond
        // (1569398400000) and are separated by name; the rest follow their listing
        // dates. The tie is not incidental — Binance onboarded whole batches at once,
        // so without the name tiebreak these three would be ordered by however the
        // response happened to serialize.
        assert_eq!(u.symbols.id("BTCUSDT"), Some(SymbolId::new(0)));
        assert_eq!(u.symbols.id("DOGEUSDT"), Some(SymbolId::new(1)));
        assert_eq!(u.symbols.id("ETHUSDT"), Some(SymbolId::new(2)));
        assert_eq!(u.symbols.id("SPELLUSDT"), Some(SymbolId::new(3)));
        assert_eq!(u.symbols.id("ETHBTC"), Some(SymbolId::new(4)));
        assert_eq!(u.symbols.id("ARKUSDT"), Some(SymbolId::new(5)));
        assert_eq!(u.symbols.symbol(SymbolId::new(0)), Some("BTCUSDT"));
        assert_eq!(u.symbols.exceeds_inflight_bound(), 0);
    }
}
