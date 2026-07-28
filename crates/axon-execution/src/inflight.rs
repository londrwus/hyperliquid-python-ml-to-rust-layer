//! [`InFlight`] — which symbols currently have an intent at the venue.
//!
//! The rule this exists to keep is ADR-0020 §3's, and it is not a nicety: **the core
//! must not plan again for a symbol while the edge still has an intent for that symbol
//! in flight.** The tracker learns about an order when its ack comes back, not when it
//! is planned, so a second pass 200 µs later computes its delta against a position the
//! first pass has already committed to move — and one target becomes two orders, which
//! is precisely the compounding bug the delta rule (ADR-0014 §2) was written to
//! prevent, reintroduced by the plumbing underneath it.
//!
//! ADR-0020 shipped that gate as a single global counter, and named the cost in its own
//! minus column: *"a slow submit on BTC delays the next pass for ETH as well. Correct
//! but coarse."* This is the per-symbol version of the same guarantee.
//!
//! Three properties are deliberate, and each of them is a failure this could otherwise
//! have:
//!
//! 1. **It cannot poison.** The set is read on the deterministic core thread and
//!    written from the tokio edge. A `Mutex<HashSet<SymbolId>>` would do the same job
//!    and would add a *second* lock whose poisoning abandons the pass — the session
//!    already has `POISONED TRACKER n` for the first, and a panic anywhere under this
//!    one would stop the intent path with no order state to blame it on. Atomics have
//!    no such state.
//! 2. **It cannot allocate or block.** A claim is one `fetch_or`; a release is one
//!    `fetch_and`. Neither can wait on the other side of the seam, which matters
//!    because the waiting side would be the core loop.
//! 3. **It is exact for the symbols it covers and conservatively coarse beyond them,
//!    never wrong.** See [`InFlight::CAPACITY`].

use std::sync::atomic::{AtomicU64, Ordering};

use axon_core::SymbolId;

/// 64-bit words backing the bitset. `CAPACITY / 64`.
const WORDS: usize = InFlight::CAPACITY / 64;

/// The set of symbols with an unfinished intent at the venue.
///
/// Claiming is single-writer in practice — only the core thread plans — but the
/// operations are atomic anyway, because the release comes from the submit task and a
/// read-modify-write split across the two threads would be exactly the race the whole
/// structure exists to close.
#[derive(Debug)]
pub struct InFlight {
    words: [AtomicU64; WORDS],
    /// Symbols at or past [`InFlight::CAPACITY`], as a count rather than a bit.
    ///
    /// A bit would have to be *shared* by every out-of-range symbol, and sharing is
    /// unsafe in the one direction that matters: releasing A would clear the bit while
    /// B was still in flight, and the core would plan a second B intent on top of the
    /// first. A count admits at most one such symbol at a time — which is the old
    /// global gate, applied only past the bound — and can never under-report.
    overflow: AtomicU64,
    /// Total releases ever made. Monotonic, and the only thing here that answers
    /// *"is the edge still making progress?"*.
    ///
    /// Membership alone cannot answer it. On a session trading several instruments the
    /// set is legitimately non-empty almost all the time, so "something has been in
    /// flight for two seconds" is true of a perfectly healthy session and useless as a
    /// stall signal; what is not true of a healthy session is that *nothing completed*
    /// in two seconds. The global counter this replaced did not need it, because with
    /// one batch in flight at a time the set emptied between every pair of batches.
    completed: AtomicU64,
    /// Claims that had to go through [`Self::overflow`] because the symbol was past
    /// [`Self::CAPACITY`].
    ///
    /// The whole problem with a bound is that exceeding it changes nothing an operator
    /// can see: the gate quietly goes back to being global for those symbols and the
    /// session keeps reporting exactly what it reported before. A non-zero count here
    /// is the one piece of evidence that `CAPACITY` is now too small for the venue this
    /// build is pointed at.
    coarse_claims: AtomicU64,
}

impl Default for InFlight {
    fn default() -> Self {
        Self::new()
    }
}

impl InFlight {
    /// Symbols `0..CAPACITY` get a bit of their own; everything at or past it shares
    /// the global behaviour through [`InFlight::overflow`].
    ///
    /// **This is a number we choose, and it has to be justified against the widest
    /// venue rather than the nearest one.** `SymbolId` is a dense index into the
    /// venue's own universe, so the requirement is the instrument count of whatever
    /// venue is wired up — and those differ by more than an order of magnitude:
    /// Hyperliquid testnet lists ~210 perps, Binance USD-M lists **730**. The first
    /// draft of this constant was 1024 and its comment claimed the bound was "on the
    /// number of instruments a venue lists and not on anything we choose", which read
    /// as though it could not bind; against Binance it would have left 29 % headroom,
    /// and the failure past it is *silent* — the gate goes back to being global for
    /// those symbols and nothing in the output changes.
    ///
    /// 4096 costs 512 bytes, once per session, and is five and a half times the widest
    /// universe currently adapted. Past it the gate degrades to *coarser*, never to
    /// *wrong*: a second out-of-range symbol simply waits, exactly as every symbol did
    /// before this type existed — and [`InFlight::coarse_claims`] counts it, so the
    /// degradation is answerable instead of invisible.
    pub const CAPACITY: usize = 4096;

    pub fn new() -> Self {
        Self {
            words: [const { AtomicU64::new(0) }; WORDS],
            overflow: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            coarse_claims: AtomicU64::new(0),
        }
    }

    /// Take the slot for `symbol`. `false` means an intent for it is already in
    /// flight and nothing may be planned for it yet.
    ///
    /// Atomic, so two callers cannot both believe they claimed it.
    pub fn claim(&self, symbol: SymbolId) -> bool {
        match Self::slot(symbol) {
            Some((word, bit)) => self.words[word].fetch_or(bit, Ordering::AcqRel) & bit == 0,
            None => {
                // Counted whether or not it succeeds: both outcomes are the bound
                // biting, and the refusal is the more interesting of the two.
                self.coarse_claims.fetch_add(1, Ordering::AcqRel);
                self.overflow
                    .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            }
        }
    }

    /// Give the slot back, once the venue has answered for it.
    ///
    /// Idempotent: releasing a symbol that was never claimed is a no-op rather than an
    /// error, because the caller that has to release on a *failed* handoff cannot
    /// always tell whether the claim got that far, and a slot never released is a
    /// symbol that never trades again.
    pub fn release(&self, symbol: SymbolId) {
        match Self::slot(symbol) {
            Some((word, bit)) => {
                self.words[word].fetch_and(!bit, Ordering::AcqRel);
            }
            None => {
                self.overflow.store(0, Ordering::Release);
            }
        }
        // Bumped unconditionally, including for the idempotent no-op above. A release
        // that did not correspond to a claim is the unwind of a handoff that failed,
        // and that is progress too — the thing a stall watcher must not mistake for a
        // wedge is the path still moving, however it moved.
        self.completed.fetch_add(1, Ordering::AcqRel);
    }

    /// How many releases have happened. Only ever compared with itself: a caller asks
    /// "has this changed since I last looked?", never "how many".
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Whether an intent for `symbol` is still at the venue.
    pub fn contains(&self, symbol: SymbolId) -> bool {
        match Self::slot(symbol) {
            Some((word, bit)) => self.words[word].load(Ordering::Acquire) & bit != 0,
            None => self.overflow.load(Ordering::Acquire) != 0,
        }
    }

    /// Claims that fell past [`Self::CAPACITY`] and had to take the coarse slot.
    ///
    /// Non-zero means the bound is too small for the venue this build is pointed at,
    /// and that the per-symbol gate has silently become a global one for some of its
    /// instruments. Nothing else in a session's output changes when that happens, which
    /// is the entire reason this is counted.
    pub fn coarse_claims(&self) -> u64 {
        self.coarse_claims.load(Ordering::Acquire)
    }

    /// How many symbols are in flight. For the status line, not for a decision.
    pub fn len(&self) -> usize {
        let bits: u32 = self
            .words
            .iter()
            .map(|w| w.load(Ordering::Acquire).count_ones())
            .sum();
        bits as usize + self.overflow.load(Ordering::Acquire).min(1) as usize
    }

    /// Whether anything is outstanding.
    ///
    /// Short-circuits rather than going through [`Self::len`], because this one is on
    /// the pass path — the stall watcher asks it every drain interval — and with 64
    /// words a full count would be 64 atomic loads to answer a question the first
    /// non-zero word settles. Only the genuinely idle case reads them all, and that is
    /// the case with nothing else to do.
    pub fn is_empty(&self) -> bool {
        self.overflow.load(Ordering::Acquire) == 0
            && self.words.iter().all(|w| w.load(Ordering::Acquire) == 0)
    }

    /// Clear every slot. For a shutdown or a resync — never for a running session,
    /// where a slot released early is a symbol planned twice.
    pub fn clear(&self) {
        for w in &self.words {
            w.store(0, Ordering::Release);
        }
        self.overflow.store(0, Ordering::Release);
    }

    /// `(word, bit)` for a symbol inside the bound, `None` for one past it.
    #[inline]
    fn slot(symbol: SymbolId) -> Option<(usize, u64)> {
        let id = symbol.get() as usize;
        if id >= Self::CAPACITY {
            return None;
        }
        Some((id / 64, 1u64 << (id % 64)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BTC: SymbolId = SymbolId::new(3);
    const ETH: SymbolId = SymbolId::new(4);

    #[test]
    fn a_symbol_already_in_flight_cannot_be_claimed_twice_so_one_target_never_becomes_two_orders() {
        // The whole reason the gate exists. The tracker does not know about an order
        // until its ack lands, so a second plan for the same symbol computes the same
        // delta against the same position and the venue ends up holding twice the
        // target — silently, and compounding.
        let f = InFlight::new();
        assert!(f.claim(BTC));
        assert!(!f.claim(BTC), "the second claim must fail");
        assert!(f.contains(BTC));
        f.release(BTC);
        assert!(!f.contains(BTC));
        assert!(
            f.claim(BTC),
            "and the symbol trades again once the venue answers"
        );
    }

    #[test]
    fn one_symbols_round_trip_never_gates_another() {
        // ADR-0020's minus column, in one assertion: a slow submit on BTC used to delay
        // the next pass for ETH as well, because the gate was a single global counter.
        let f = InFlight::new();
        assert!(f.claim(BTC));
        assert!(f.claim(ETH), "ETH is not waiting on BTC");
        assert_eq!(f.len(), 2);
        f.release(BTC);
        assert!(!f.contains(BTC));
        assert!(
            f.contains(ETH),
            "and releasing one does not release the other"
        );
    }

    #[test]
    fn symbols_past_the_bound_degrade_to_the_old_global_gate_rather_than_sharing_a_slot() {
        // Sharing a bit would be unsafe in exactly one direction: releasing the first
        // out-of-range symbol would clear the bit while the second was still in flight,
        // and the core would plan a second intent on top of an unacknowledged one. So
        // past the bound at most one symbol is admitted at a time — coarse, which is
        // what every symbol got before this type existed, and never wrong.
        let f = InFlight::new();
        let far = SymbolId::new(InFlight::CAPACITY as u32);
        let farther = SymbolId::new(InFlight::CAPACITY as u32 + 7);

        assert_eq!(f.coarse_claims(), 0);
        assert!(f.claim(far));
        assert!(!f.claim(farther), "coarse past the bound");
        assert!(f.contains(farther), "and it says so rather than lying");
        // A symbol inside the bound is unaffected by the overflow slot.
        assert!(f.claim(BTC));
        f.release(far);
        assert!(f.claim(farther));

        // …and the degradation is answerable. Exceeding the bound changes nothing else
        // a session prints: the gate quietly goes back to being global for those
        // symbols. This counter is the only evidence that `CAPACITY` is too small for
        // the venue this build is pointed at.
        assert_eq!(f.coarse_claims(), 3);
    }

    #[test]
    fn the_bound_clears_the_widest_venue_universe_currently_adapted() {
        // The finding that raised this constant: it is a number *we* choose, against a
        // requirement that differs by more than an order of magnitude between venues.
        // Hyperliquid testnet lists ~210 perps and 1024 looked generous; Binance USD-M
        // lists 730, which left 29 % headroom for a failure that is invisible when it
        // arrives. A universe that fits must never touch the coarse slot.
        const BINANCE_USD_M_PERPS: u32 = 730;
        assert!(
            (BINANCE_USD_M_PERPS as usize) < InFlight::CAPACITY,
            "the widest adapted universe has to fit with room, not just fit"
        );

        let f = InFlight::new();
        for id in 0..BINANCE_USD_M_PERPS {
            assert!(f.claim(SymbolId::new(id)), "sym {id}");
        }
        assert_eq!(f.len(), BINANCE_USD_M_PERPS as usize);
        assert_eq!(
            f.coarse_claims(),
            0,
            "every instrument got a slot of its own"
        );
    }

    #[test]
    fn progress_is_visible_even_while_the_set_is_never_empty() {
        // The false positive membership alone would produce. On a session trading
        // several instruments the set is legitimately occupied almost all the time, so
        // "something has been in flight for two seconds" describes a healthy session as
        // well as a wedged one. What separates them is whether anything *completed*.
        let f = InFlight::new();
        assert_eq!(f.completed(), 0);
        assert!(f.claim(BTC));
        assert!(f.claim(ETH));

        let before = f.completed();
        f.release(BTC);
        assert!(f.claim(BTC), "BTC turns over…");
        assert!(!f.is_empty(), "…while the set never empties");
        assert!(
            f.completed() > before,
            "and the only number that says so has moved"
        );
    }

    #[test]
    fn releasing_a_slot_that_was_never_claimed_is_a_no_op_rather_than_a_corruption() {
        // The handoff can fail after the claim (a full queue), and the caller that
        // unwinds it cannot always tell how far it got. A release that had to be
        // matched would leave the symbol gated forever, which reads on the status line
        // exactly like a strategy with no opinion about that instrument.
        let f = InFlight::new();
        f.release(BTC);
        assert!(f.is_empty());
        assert!(f.claim(BTC));
        f.release(BTC);
        f.release(BTC);
        assert!(f.is_empty());
    }

    #[test]
    fn every_symbol_in_the_bound_gets_a_slot_of_its_own() {
        // Guards the word/bit arithmetic: an off-by-one in `slot` would alias two
        // symbols onto one bit, and aliasing is the failure the overflow count exists
        // to avoid — silently, because both symbols would still look healthy.
        let f = InFlight::new();
        for id in 0..InFlight::CAPACITY as u32 {
            assert!(f.claim(SymbolId::new(id)), "sym {id}");
        }
        assert_eq!(f.len(), InFlight::CAPACITY);
        for id in 0..InFlight::CAPACITY as u32 {
            assert!(f.contains(SymbolId::new(id)));
            f.release(SymbolId::new(id));
        }
        assert!(f.is_empty());
    }

    #[test]
    fn a_claim_and_its_release_cross_the_thread_seam_intact() {
        // The claim is made on the deterministic core thread and the release comes from
        // the submit task. A non-atomic read-modify-write here would drop one of the
        // two, and a dropped release is a symbol that never trades again.
        let f = std::sync::Arc::new(InFlight::new());
        assert!(f.claim(BTC));
        let edge = f.clone();
        std::thread::spawn(move || edge.release(BTC))
            .join()
            .unwrap();
        assert!(!f.contains(BTC));
    }
}
