//! Symbol string ↔ [`SymbolId`], and the funding interval the wire never carries.
//!
//! Hyperliquid's `SymbolMap` translates a coin name to the venue's **own** asset
//! index, and that index is the number that goes on the wire — `SymbolId(3)` *is*
//! testnet BTC, to the venue as much as to us. Binance publishes no such number.
//! Every request and every frame addresses an instrument by its symbol string, so
//! there is nothing to mirror and the ids are ours to invent.
//!
//! That turns out to be the sharpest thing this adapter learned about the port, and
//! it is not the extra work — it is that **`SymbolId` is load-bearing in two places
//! that assume a venue supplies it**:
//!
//! 1. `axon_execution::InFlight` gives each symbol below `CAPACITY` (1024) its own
//!    bit and collapses everything at or past it into one shared overflow slot, which
//!    silently reinstates the global per-session gate that type exists to remove. Its
//!    own comment says the bound is safe because "`SymbolId` is a dense index into
//!    the venue's own `meta` universe rather than a hash". There is no such index
//!    here, so **the ids must be dense by construction** — a hash of the symbol name
//!    would be stable across restarts and would put all 730 instruments past the
//!    bound.
//! 2. A capture records `SymbolId`s and no symbol table, so a replay resolves them
//!    against whatever universe the replaying process holds.
//!
//! Dense and stable cannot both be had from a venue that supplies neither, so this
//! picks dense and then buys back as much stability as ordering can. See
//! [`SymbolTable::assign`].

use std::collections::HashMap;

use axon_core::{Nanos, SymbolId};

use crate::{DEFAULT_FUNDING_INTERVAL_NS, MS_TO_NS};

/// The id past which `axon_execution::InFlight` stops tracking symbols individually.
///
/// Named here rather than imported, because `axon-providers` is a leaf over
/// `axon-core` and this crate must not gain a dependency on `axon-execution` to state
/// a number — an adapter reaching up into the execution crate is exactly the edge
/// ADR-0004 exists to prevent. The cost of naming it twice is that the two can drift;
/// [`SymbolTable::exceeds_inflight_bound`] is how a session finds out, and the
/// alternative was silence.
pub const INFLIGHT_TRACKING_BOUND: u32 = 1024;

/// A universe that cannot be turned into an id assignment.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SymbolTableError {
    /// The same symbol string appeared twice in one `exchangeInfo` response.
    ///
    /// A refusal rather than a last-write-wins insert: the two rows would carry
    /// different filters, one of them would win the instrument table, and orders for
    /// the loser would be quantized to the winner's grid.
    #[error("symbol {symbol} appears more than once in one universe")]
    Duplicate { symbol: String },
}

/// Binance symbol ↔ [`SymbolId`], with the per-symbol funding period alongside.
///
/// The funding period rides here rather than on `InstrumentSpec` because it is not a
/// number format: it is a *market* fact the ticker decoder needs on every frame, and
/// `InstrumentSpec` is the type the planner and the encoder share. Putting it there
/// would have made a `Copy` two-grid struct carry a field neither of them reads.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    by_symbol: HashMap<String, SymbolId>,
    by_id: HashMap<SymbolId, String>,
    funding: HashMap<SymbolId, Nanos>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign dense ids to `universe`, ordered by `(onboard_date, symbol)`.
    ///
    /// The ordering is the whole decision, and neither obvious candidate works:
    ///
    /// - **Response order** is not stable at all. Binance's `symbols` array is not
    ///   sorted and its order is not documented, so two reads a day apart can permute
    ///   it and every id moves for no visible reason.
    /// - **Symbol name** is stable to read and maximally unstable to *listings*: a new
    ///   `AAVEUSDT` sorts ahead of `BTCUSDT` and shifts every id after it by one. A
    ///   capture replayed the next day then resolves every instrument to its
    ///   neighbour — a replay that runs clean and describes a different session.
    ///
    /// Listing time first means a new instrument almost always **appends**, so ids
    /// already issued keep their meaning across the event that happens weekly. The
    /// symbol name breaks ties, because Binance onboarded whole batches at the same
    /// millisecond (BTCUSDT and ETHUSDT share `1569398400000`) and a tie left to the
    /// response order would put the instability straight back.
    ///
    /// What this still does **not** survive is a *delisting*: removing a row compacts
    /// every id above it. That is the same hole as ADR-0025's out-of-scope #2 — a plan
    /// is a function of a network-fetched input the capture log does not carry — and
    /// it has the same fix, which is to persist the mapping rather than to derive it.
    /// Deriving it is what this does; ADR-0023 says so out loud rather than leaving a
    /// reader to assume ids are venue-issued the way Hyperliquid's are.
    pub fn assign<I>(universe: I) -> Result<Self, SymbolTableError>
    where
        I: IntoIterator<Item = (String, i64)>,
    {
        let mut rows: Vec<(String, i64)> = universe.into_iter().collect();
        rows.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let mut table = Self::new();
        for (symbol, _) in rows {
            if table.by_symbol.contains_key(&symbol) {
                return Err(SymbolTableError::Duplicate { symbol });
            }
            let id = SymbolId::new(table.by_symbol.len() as u32);
            table.by_id.insert(id, symbol.clone());
            table.by_symbol.insert(symbol, id);
        }
        Ok(table)
    }

    /// Build a table straight from an ordered list, for tests and examples.
    ///
    /// The list *is* the order — no sort — so a test can state the ids it wants
    /// instead of re-deriving [`assign`](Self::assign)'s tie-breaking in its head.
    pub fn from_ordered<I, S>(symbols: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut table = Self::new();
        for (i, s) in symbols.into_iter().enumerate() {
            let symbol = s.into();
            let id = SymbolId::new(i as u32);
            table.by_id.insert(id, symbol.clone());
            table.by_symbol.insert(symbol, id);
        }
        table
    }

    /// Resolve a venue symbol string (`"BTCUSDT"`, upper case) to its id.
    ///
    /// Case-sensitive on purpose. Every *payload* field Binance sends is upper case
    /// (`"s":"BTCUSDT"`); only the *stream name* is lower case (`btcusdt@aggTrade`),
    /// and no decoder resolves a symbol from a stream name. Normalizing here would
    /// mean an allocation and an uppercase pass per frame on the ingest path to
    /// tolerate an input that never arrives.
    pub fn id(&self, symbol: &str) -> Option<SymbolId> {
        self.by_symbol.get(symbol).copied()
    }

    /// Resolve an id back to the venue symbol string.
    pub fn symbol(&self, id: SymbolId) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// Record how often `symbol` funds, from `GET /fapi/v1/fundingInfo`.
    ///
    /// Keyed on the symbol string rather than the id, because `fundingInfo` is a
    /// second response and may name symbols this table has never heard of (it lists
    /// COIN-M pairs such as `BCHUSD_PERP` alongside the USD-M ones). Those are
    /// ignored, quietly and correctly: a funding period for an instrument we do not
    /// trade is not an error.
    pub fn set_funding_hours(&mut self, symbol: &str, hours: i64) {
        if let Some(id) = self.id(symbol) {
            self.funding.insert(id, hours * 3_600 * 1_000 * MS_TO_NS);
        }
    }

    /// How often `id` funds. [`DEFAULT_FUNDING_INTERVAL_NS`] when the venue did not
    /// say.
    ///
    /// The fallback is resolved *here*, once, for the same reason
    /// [`Ticker::ts_event`](axon_core::Ticker::ts_event) resolves its own: two call
    /// sites that each pick a default will eventually pick different ones, and the
    /// symptom is a carry number that disagrees with itself between two components.
    pub fn funding_interval(&self, id: SymbolId) -> Nanos {
        self.funding
            .get(&id)
            .copied()
            .unwrap_or(DEFAULT_FUNDING_INTERVAL_NS)
    }

    /// Whether the venue published this symbol's funding period, as opposed to us
    /// assuming it.
    ///
    /// The same distinction [`Ticker::is_venue_timed`](axon_core::Ticker::is_venue_timed)
    /// draws about time, for the same reason: a consumer that cares whether a number
    /// came from the venue has to be able to ask.
    pub fn funding_is_published(&self, id: SymbolId) -> bool {
        self.funding.contains_key(&id)
    }

    /// The lower-case symbol a stream name is built from (`BTCUSDT` → `btcusdt`).
    ///
    /// Allocates. It is called once per feed per subscription, off the ingest path;
    /// the alternative is storing both cases for 730 symbols to save an allocation
    /// nobody makes twice.
    pub fn stream_symbol(&self, id: SymbolId) -> Option<String> {
        self.symbol(id).map(str::to_lowercase)
    }

    pub fn len(&self) -> usize {
        self.by_symbol.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_symbol.is_empty()
    }

    /// Ids this table issued that `axon_execution::InFlight` will not track
    /// individually.
    ///
    /// Zero on every Binance universe seen so far (730 rows against a bound of 1024),
    /// and the reason to compute it anyway is that the failure is invisible: past the
    /// bound the per-symbol gate silently becomes the global one, so a session trading
    /// the tail of a large universe serializes its order flow with no error, no
    /// counter and no log line. A caller that prints this at startup finds out before
    /// it costs a fill.
    pub fn exceeds_inflight_bound(&self) -> usize {
        self.by_id
            .keys()
            .filter(|id| id.get() >= INFLIGHT_TRACKING_BOUND)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn universe() -> Vec<(String, i64)> {
        // Real onboard dates from the captured testnet `exchangeInfo`: BTCUSDT and
        // ETHUSDT share one to the millisecond, 1000SHIBUSDT came later.
        vec![
            ("ETHUSDT".to_string(), 1_569_398_400_000),
            ("BTCUSDT".to_string(), 1_569_398_400_000),
            ("1000SHIBUSDT".to_string(), 1_620_630_000_000),
        ]
    }

    #[test]
    fn ids_are_dense_from_zero_because_the_inflight_gate_silently_degrades_past_its_bound() {
        // `InFlight` gives ids below 1024 a bit each and shares one overflow slot
        // between everything else, so a sparse or hashed id space would put every
        // instrument past the bound and turn the per-symbol gate back into the global
        // one — with no error and no counter to notice it by.
        let t = SymbolTable::assign(universe()).unwrap();
        let mut ids: Vec<u32> = (0..t.len()).map(|i| i as u32).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
        for i in ids {
            assert!(t.symbol(SymbolId::new(i)).is_some(), "id {i} is unassigned");
        }
        assert_eq!(t.exceeds_inflight_bound(), 0);
    }

    #[test]
    fn a_new_listing_appends_instead_of_shifting_every_id_underneath_it() {
        // The failure this ordering prevents: sorted by name, a newly listed
        // `AAVEUSDT` takes id 0 and every existing instrument shifts by one. A capture
        // replayed the next day then resolves every `SymbolId` to its neighbour — a
        // replay that runs clean and describes a different session entirely.
        let before = SymbolTable::assign(universe()).unwrap();
        let btc = before.id("BTCUSDT").unwrap();
        let eth = before.id("ETHUSDT").unwrap();
        assert_eq!(btc, SymbolId::new(0), "tie broken by name: BTC before ETH");
        assert_eq!(eth, SymbolId::new(1));

        let mut later = universe();
        later.push(("AAVEUSDT".to_string(), 1_700_000_000_000));
        let after = SymbolTable::assign(later).unwrap();
        assert_eq!(after.id("BTCUSDT"), Some(btc), "BTC did not move");
        assert_eq!(after.id("ETHUSDT"), Some(eth));
        assert_eq!(after.id("AAVEUSDT"), Some(SymbolId::new(3)), "appended");
    }

    #[test]
    fn a_delisting_still_moves_every_id_above_it_and_that_is_the_hole() {
        // Stated as a test rather than a comment, because it is the one case the
        // ordering does not cover and a reader is entitled to know it is known. The
        // fix is a persisted mapping, which is a capture-log change (ADR-0025's
        // out-of-scope #2) and not an adapter one.
        let full = SymbolTable::assign(universe()).unwrap();
        let shib_before = full.id("1000SHIBUSDT").unwrap();
        let delisted: Vec<(String, i64)> = universe()
            .into_iter()
            .filter(|(s, _)| s != "BTCUSDT")
            .collect();
        let after = SymbolTable::assign(delisted).unwrap();
        assert_ne!(
            after.id("1000SHIBUSDT"),
            Some(shib_before),
            "a delisting compacts the ids above it - this is the documented hole"
        );
    }

    #[test]
    fn one_symbol_listed_twice_is_refused_rather_than_silently_overwritten() {
        // Two rows for one symbol carry two sets of filters. Last-write-wins would put
        // one of them in the instrument table and quantize the other's orders to the
        // wrong grid — accepted by the venue, filled, and the wrong size.
        let dupes = vec![
            ("BTCUSDT".to_string(), 1_569_398_400_000),
            ("BTCUSDT".to_string(), 1_600_000_000_000),
        ];
        assert!(matches!(
            SymbolTable::assign(dupes),
            Err(SymbolTableError::Duplicate { ref symbol }) if symbol == "BTCUSDT"
        ));
    }

    #[test]
    fn an_unpublished_funding_period_falls_back_to_eight_hours_and_says_that_it_did() {
        // `fundingInfo` lists 616 of 730 symbols. The other 114 fund on the documented
        // default, and a consumer that needs to know whether a rate came with its own
        // period — a carry strategy, the parity harness — has to be able to ask rather
        // than infer.
        let mut t = SymbolTable::assign(universe()).unwrap();
        let btc = t.id("BTCUSDT").unwrap();
        let eth = t.id("ETHUSDT").unwrap();
        assert_eq!(t.funding_interval(btc), DEFAULT_FUNDING_INTERVAL_NS);
        assert!(!t.funding_is_published(btc));

        t.set_funding_hours("BTCUSDT", 4);
        assert_eq!(t.funding_interval(btc), 4 * 3_600 * 1_000_000_000);
        assert!(t.funding_is_published(btc));
        assert_eq!(
            t.funding_interval(eth),
            DEFAULT_FUNDING_INTERVAL_NS,
            "one symbol's published period does not become another's"
        );

        // `fundingInfo` also names COIN-M symbols this table has never heard of.
        // Ignoring them is correct; panicking or inserting a stray id is not.
        t.set_funding_hours("BCHUSD_PERP", 8);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn a_stream_name_is_lower_case_while_every_payload_field_is_upper_case() {
        // The one place the two cases meet. Resolving a symbol from a stream name
        // would need an allocation per frame; resolving it from `data.s` needs none,
        // which is why `id()` is case-sensitive and this conversion is subscribe-time
        // only.
        let t = SymbolTable::assign(universe()).unwrap();
        let btc = t.id("BTCUSDT").unwrap();
        assert_eq!(t.stream_symbol(btc).as_deref(), Some("btcusdt"));
        assert_eq!(t.id("btcusdt"), None, "payloads never send lower case");
    }
}
