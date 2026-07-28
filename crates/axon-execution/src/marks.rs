//! [`MarkCache`] — the price the pre-trade gate measures exposure in, and the rule
//! for when a price stops counting.
//!
//! The gate ([`crate::guard`]) already fails closed on a **missing** mark. That is
//! only half the problem, and the smaller half: a *stale* mark is worse than a
//! missing one. A missing mark refuses the order and costs one trade; a mark from
//! five minutes ago sails through the notional check and sizes a position against a
//! price that no longer exists. So an entry here **expires**, and an expired entry is
//! indistinguishable from an absent one to every caller that matters —
//! [`MarkCache::get`] returns `None` and the gate refuses. Stale collapses into
//! missing on purpose: it is the one failure mode the gate already handles correctly.
//!
//! Two decisions the rest of the runtime depends on:
//!
//! **1. Where the price comes from.** In precedence order: the venue's own
//! [`mark price`](axon_core::Ticker::mark_px), then the book/BBO mid. The venue mark
//! wins because it is the number the venue itself computes margin, liquidation and
//! unrealized PnL against — checking our notional against anything else measures a
//! different quantity than the one that can liquidate us. The mid is the fallback for
//! instruments (or sessions) with no ticker feed, and it takes over only once the
//! venue mark has gone stale, so a mid tick can never quietly displace a live mark.
//! The **last trade is deliberately never a mark source**: one print in a thin book
//! is an outlier, and in a fast market the last print can be minutes old while the
//! book has moved — precisely the staleness this module exists to catch, wearing a
//! fresh timestamp.
//!
//! **2. What "now" means.** Staleness is a *liveness* question, not an ordering one,
//! so it cannot be answered with event time alone: a dead feed emits no events, event
//! time freezes, and a frozen clock declares everything fresh forever — the exact
//! case we need to detect. The cache therefore keeps a high-water "now" advanced by
//! two sources: every price it is handed (its `ts_event`) and, in a live session,
//! [`observe_now`](MarkCache::observe_now) from the supervisor's wall clock. Replay
//! and backtests never call `observe_now`, so their staleness is measured purely in
//! event time and stays bit-reproducible.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::RwLock;

use axon_core::{Decimal, Nanos, SymbolId};

/// Default age at which a mark stops counting: 10 s.
///
/// Chosen against the feeds, not the strategy: Hyperliquid publishes `activeAssetCtx`
/// every few seconds and a BBO far more often, so ten seconds of silence on a liquid
/// perp means the feed is broken rather than the market being quiet. Sessions on
/// slower instruments should widen it via [`MarkCache::with_max_age`] rather than
/// discovering that every order is refused.
pub const DEFAULT_MAX_AGE_NS: Nanos = 10_000_000_000;

/// Which feed a cached price came from. Load-bearing: it is what stops a book mid
/// from overwriting a live venue mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkSource {
    /// The venue's published mark (`Ticker::mark_px`) — what margin is computed on.
    VenueMark,
    /// Midpoint of the book or BBO, used where no venue mark is available.
    BookMid,
}

/// A cached price with the two things that make it judgeable: when it was true, and
/// where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkQuote {
    pub px: Decimal,
    /// The **event's own** timestamp, never receipt time — a price is as old as the
    /// market moment it describes.
    pub ts_event: Nanos,
    pub source: MarkSource,
}

/// Mark prices, kept separately from the order tracker because they come from market
/// data, not from order flow.
///
/// A map behind a lock rather than a channel: the gate reads one price per check and
/// must never block on a feed, and the runtime already owns the market-data handler
/// that writes here. All methods take `&self` so it can be shared as an `Arc`.
#[derive(Debug)]
pub struct MarkCache {
    marks: RwLock<HashMap<SymbolId, MarkQuote>>,
    max_age_ns: Nanos,
    /// High-water mark of observed time (see the module docs). Separate from the
    /// map's lock because the core loop advances it on every tick, including ticks
    /// where no price changed.
    now_ns: AtomicI64,
}

impl Default for MarkCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkCache {
    pub fn new() -> Self {
        Self::with_max_age(DEFAULT_MAX_AGE_NS)
    }

    pub fn with_max_age(max_age_ns: Nanos) -> Self {
        Self {
            marks: RwLock::new(HashMap::new()),
            max_age_ns,
            now_ns: AtomicI64::new(Nanos::MIN),
        }
    }

    /// A cache whose entries never expire — for offline tests and for replay over a
    /// captured log, where "how long ago" is not a meaningful question. Never use it
    /// in a live session: it reinstates exactly the stale-mark hole this type exists
    /// to close.
    pub fn never_expires() -> Self {
        Self::with_max_age(Nanos::MAX)
    }

    pub fn max_age_ns(&self) -> Nanos {
        self.max_age_ns
    }

    /// Advance the staleness clock. Monotonic — a late frame cannot rewind it, which
    /// would resurrect entries that had already expired.
    pub fn observe_now(&self, now_ns: Nanos) {
        self.now_ns.fetch_max(now_ns, Ordering::Relaxed);
    }

    /// Latest time this cache knows about.
    pub fn now_ns(&self) -> Nanos {
        self.now_ns.load(Ordering::Relaxed)
    }

    /// Record the venue's own mark price.
    pub fn set_mark(&self, symbol: SymbolId, px: Decimal, ts_event: Nanos) {
        self.set(symbol, px, ts_event, MarkSource::VenueMark);
    }

    /// Record a book/BBO midpoint as a *fallback* price. It is applied only when
    /// there is no live venue mark for the symbol — see [`Self::supersedes`].
    pub fn set_mid(&self, symbol: SymbolId, px: Decimal, ts_event: Nanos) {
        self.set(symbol, px, ts_event, MarkSource::BookMid);
    }

    fn set(&self, symbol: SymbolId, px: Decimal, ts_event: Nanos, source: MarkSource) {
        // A price is itself evidence of the current time; observe it before deciding
        // whether the incumbent has expired, or the first tick after a long silence
        // would be judged against a clock that predates it.
        self.observe_now(ts_event);
        let now = self.now_ns();
        let fresh = MarkQuote {
            px,
            ts_event,
            source,
        };
        // A poisoned lock drops the update rather than panicking: the reader side
        // already treats a poisoned cache as "no price", which fails the gate closed.
        if let Ok(mut m) = self.marks.write() {
            match m.get(&symbol) {
                Some(incumbent) if !Self::supersedes(&fresh, incumbent, now, self.max_age_ns) => {}
                _ => {
                    m.insert(symbol, fresh);
                }
            }
        }
    }

    /// Whether `fresh` should replace `incumbent`, given the current time.
    ///
    /// Three rules, each closing a specific hole:
    /// 1. **Never go back in time.** Feeds reorder; a delayed frame carrying an older
    ///    `ts_event` describes a market moment we have already passed, and letting it
    ///    land would both move the price backwards and reset the age clock, making a
    ///    stale entry look fresh.
    /// 2. **The venue mark always beats a mid**, because it is the price the venue
    ///    itself will margin and liquidate us on.
    /// 3. **A mid may take over only once the mark has expired.** If it could
    ///    overwrite a live mark, a single BBO tick would silently swap the quantity
    ///    the risk gate is measuring, and nothing would report it.
    fn supersedes(
        fresh: &MarkQuote,
        incumbent: &MarkQuote,
        now_ns: Nanos,
        max_age_ns: Nanos,
    ) -> bool {
        if fresh.ts_event < incumbent.ts_event {
            return false;
        }
        match (fresh.source, incumbent.source) {
            (MarkSource::VenueMark, _) => true,
            (MarkSource::BookMid, MarkSource::BookMid) => true,
            (MarkSource::BookMid, MarkSource::VenueMark) => {
                is_stale(incumbent.ts_event, now_ns, max_age_ns)
            }
        }
    }

    /// The usable mark for `symbol`, or `None` if there is none **or it has expired**.
    /// This is what [`RiskContext::mark_px`](crate::guard::RiskContext::mark_px) calls,
    /// so an expired price refuses the order.
    pub fn get(&self, symbol: SymbolId) -> Option<Decimal> {
        self.quote(symbol).map(|q| q.px)
    }

    /// [`Self::get`] with provenance and age attached.
    pub fn quote(&self, symbol: SymbolId) -> Option<MarkQuote> {
        let now = self.now_ns();
        let q = *self.marks.read().ok()?.get(&symbol)?;
        (!is_stale(q.ts_event, now, self.max_age_ns)).then_some(q)
    }

    /// The last price seen for `symbol` **whether or not it is still valid**.
    ///
    /// For status lines and diagnostics only. Naming it apart from [`Self::get`] is
    /// the point: an operator needs to see "BTC mark is 40 s old", while the gate must
    /// never be handed that number.
    pub fn last_known(&self, symbol: SymbolId) -> Option<MarkQuote> {
        self.marks.read().ok()?.get(&symbol).copied()
    }

    /// Age of the last price for `symbol`, in nanoseconds.
    pub fn age_ns(&self, symbol: SymbolId) -> Option<Nanos> {
        let now = self.now_ns();
        self.last_known(symbol)
            .map(|q| now.saturating_sub(q.ts_event))
    }

    /// `(usable, total)` — how many symbols have a live price out of how many have
    /// ever had one. A drop in the first number with the second unchanged is a feed
    /// dying, which is what the status line watches for.
    pub fn coverage(&self) -> (usize, usize) {
        let now = self.now_ns();
        match self.marks.read() {
            Ok(m) => (
                m.values()
                    .filter(|q| !is_stale(q.ts_event, now, self.max_age_ns))
                    .count(),
                m.len(),
            ),
            Err(_) => (0, 0),
        }
    }

    /// Symbols whose price has expired, sorted for a stable status line.
    pub fn stale_symbols(&self) -> Vec<SymbolId> {
        let now = self.now_ns();
        let mut out: Vec<SymbolId> = match self.marks.read() {
            Ok(m) => m
                .iter()
                .filter(|(_, q)| is_stale(q.ts_event, now, self.max_age_ns))
                .map(|(s, _)| *s)
                .collect(),
            Err(_) => Vec::new(),
        };
        out.sort_unstable_by_key(|s| s.get());
        out
    }
}

/// Saturating so a `Nanos::MAX` max-age (the never-expires cache) cannot overflow
/// into wrapping and declare everything stale.
fn is_stale(ts_event: Nanos, now_ns: Nanos, max_age_ns: Nanos) -> bool {
    now_ns.saturating_sub(ts_event) > max_age_ns
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    const BTC: SymbolId = SymbolId::new(1);
    const ETH: SymbolId = SymbolId::new(2);

    const SEC: Nanos = 1_000_000_000;

    fn cache() -> MarkCache {
        MarkCache::with_max_age(10 * SEC)
    }

    #[test]
    fn a_stale_mark_reads_as_missing_so_the_gate_fails_closed() {
        // The whole reason this type has a clock. A price the gate still trusts after
        // the feed died is how a notional check passes against a market that has moved.
        let c = cache();
        c.set_mark(BTC, dec!(50_000), 0);
        assert_eq!(c.get(BTC), Some(dec!(50_000)));

        c.observe_now(9 * SEC);
        assert_eq!(c.get(BTC), Some(dec!(50_000)), "still inside the window");

        c.observe_now(11 * SEC);
        assert_eq!(c.get(BTC), None, "expired -> the gate sees no price at all");
        assert_eq!(
            c.last_known(BTC).map(|q| q.px),
            Some(dec!(50_000)),
            "but an operator can still see what it was"
        );
        assert_eq!(c.age_ns(BTC), Some(11 * SEC));
        assert_eq!(c.stale_symbols(), vec![BTC]);
        assert_eq!(c.coverage(), (0, 1));
    }

    #[test]
    fn a_silent_symbol_expires_while_another_keeps_ticking() {
        // The clock is shared across symbols on purpose: a session streaming BTC and
        // ETH that stops receiving ETH must not keep trading ETH on a frozen price
        // merely because BTC is still alive.
        let c = cache();
        c.set_mark(BTC, dec!(50_000), 0);
        c.set_mark(ETH, dec!(3_000), 0);
        for i in 1..=12 {
            c.set_mark(BTC, dec!(50_000) + Decimal::from(i), i * SEC);
        }
        assert!(c.get(BTC).is_some(), "the live feed stays usable");
        assert_eq!(c.get(ETH), None, "the silent one expired");
        assert_eq!(c.coverage(), (1, 2));
    }

    #[test]
    fn an_out_of_order_frame_cannot_walk_the_mark_backwards() {
        // Reordering is normal on a reconnect. Accepting the older frame would move
        // the price backwards *and* reset the age clock, making a stale entry look new.
        let c = cache();
        c.set_mark(BTC, dec!(50_000), 5 * SEC);
        c.set_mark(BTC, dec!(49_000), SEC);
        let q = c.quote(BTC).unwrap();
        assert_eq!(q.px, dec!(50_000));
        assert_eq!(q.ts_event, 5 * SEC);
    }

    #[test]
    fn a_book_mid_never_displaces_a_live_venue_mark() {
        // The mark is what the venue margins us on; the mid is a different quantity.
        // A BBO tick arriving right after a ticker frame must not silently swap them.
        let c = cache();
        c.set_mark(BTC, dec!(50_000), SEC);
        c.set_mid(BTC, dec!(50_500), 2 * SEC);
        let q = c.quote(BTC).unwrap();
        assert_eq!(q.px, dec!(50_000));
        assert_eq!(q.source, MarkSource::VenueMark);
    }

    #[test]
    fn a_book_mid_takes_over_once_the_venue_mark_has_expired() {
        // Degrading to the mid beats having no price: no price means the gate refuses
        // every risk-increasing order, and a live mid is a far better estimate of
        // exposure than nothing at all.
        let c = cache();
        c.set_mark(BTC, dec!(50_000), 0);
        c.set_mid(BTC, dec!(50_500), 11 * SEC);
        let q = c.quote(BTC).unwrap();
        assert_eq!(q.px, dec!(50_500));
        assert_eq!(q.source, MarkSource::BookMid);

        // And the venue mark reclaims the slot the moment the ticker feed returns,
        // even though the mid is newer than the mark that preceded it.
        c.set_mark(BTC, dec!(50_100), 12 * SEC);
        assert_eq!(c.quote(BTC).unwrap().source, MarkSource::VenueMark);
    }

    #[test]
    fn a_never_expiring_cache_keeps_replay_deterministic() {
        // Backtests have no wall clock to age against; "how long ago" is meaningless
        // over a captured log, so expiry is opt-out rather than silently wrong.
        let c = MarkCache::never_expires();
        c.set_mark(BTC, dec!(50_000), 0);
        c.observe_now(Nanos::MAX);
        assert_eq!(c.get(BTC), Some(dec!(50_000)));
    }

    #[test]
    fn an_unseen_symbol_has_no_price_rather_than_a_zero() {
        let c = cache();
        assert_eq!(c.get(BTC), None);
        assert_eq!(c.age_ns(BTC), None);
        assert_eq!(c.coverage(), (0, 0));
    }
}
