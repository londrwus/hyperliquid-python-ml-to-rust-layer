//! What several strategies together want an account to hold.
//!
//! Everything else in this crate is written for one producer. [`SignalReader`] validates
//! a stream whose `seq` is one monotonic sequence, and [`Planner`] turns *a* record into
//! orders against *the* position. Both stay exactly as they are; what this module adds is
//! the object in between, which only exists once more than one strategy shares an
//! account:
//!
//! ```text
//!   strategy A ─ ring ─▶ reader ─┐
//!   strategy B ─ ring ─▶ reader ─┼─▶ TargetBook ──net per symbol──▶ Planner ─▶ venue
//!   strategy C ─ ring ─▶ reader ─┘
//! ```
//!
//! ## The one fact everything here follows from
//!
//! **The venue holds one position per instrument, and a target-position signal is a
//! claim on part of it.** Two strategies that both trade BTC are not two positions; they
//! are two claims on one. So the claims *add*, and the account works toward the sum. Any
//! other rule has one strategy silently overwriting another's exposure — which is what a
//! naive "newest signal per symbol wins" does, and it does it invisibly: both producers'
//! counters climb, both believe they are positioned, and the account holds whichever one
//! spoke last.
//!
//! Addition is the right operation and not merely a convenient one. A target position is
//! self-contained by construction (ADR-0006) — it says what its author wants held, not
//! how to change what is held — so summing two of them is summing two independent claims
//! rather than composing two instructions.
//!
//! ## Why netting is opt-in even though it is correct
//!
//! Two producers pointed at the same instrument is far more often a **mistake** than a
//! decision: a copy-pasted session config, a symbol left in a producer's universe after
//! it was moved. Netting them silently composes two strategies' risk into a position
//! neither author sized. So [`Overlap::Exclusive`] is the default and refuses the second
//! claim, loudly and counted; [`Overlap::Net`] is one declared word in the config, and
//! `RuntimeConfig::validate` refuses overlapping producer universes without it. The
//! accident is caught at startup and the deliberate case costs a line.
//!
//! ## A silent strategy still holds its claim, unless the operator said otherwise
//!
//! A producer that stops — crashed, restarted with a rewound `seq`, still warming up —
//! leaves a claim behind. Dropping it would flatten that strategy's share of the account
//! on a transient stall, into whatever market caused the stall; keeping it means the
//! account holds exposure nobody is currently speaking for. Neither is safe in general,
//! so [`OnSilence`] makes it the operator's declaration, the default is
//! [`OnSilence::Hold`] (the behaviour every session before this had, since there was
//! nothing to expire), and a held-but-silent contributor is **counted and named on the
//! status line** rather than being either quietly kept or quietly dropped.
//!
//! ## What a synthesized record may and may not carry
//!
//! When exactly one strategy contributes, the netted record **is** that strategy's own
//! record, byte for byte — same `seq`, same `ts_event`, therefore the same `cloid`. That
//! is not an optimization: it means every session that exists today, every capture, and
//! every replay of one plans precisely what it planned before this module was written.
//!
//! Only a genuinely multi-contributor symbol synthesizes, and then the fields are
//! combined by rules with a failure mode named for each one — see [`TargetBook::net`].
//! The synthesized `seq` carries [`NET_SEQ_TAG`] in its top bit, which keeps the two id
//! spaces disjoint: a `cloid` minted for a netted target can never collide with one
//! minted for a producer's own record, and an operator reading a `cloid` off the venue
//! can tell which kind of decision produced it.
//!
//! [`SignalReader`]: crate::SignalReader
//! [`Planner`]: crate::Planner

use axon_contracts::{Signal, FLAG_CLOSE, FLAG_REDUCE_ONLY};
use axon_core::{Nanos, SymbolId};

/// Top bit of a synthesized netted `seq`.
///
/// A producer would have to write 9.2 × 10^18 records to reach it, so the two spaces are
/// disjoint in practice as well as by declaration. It matters because
/// [`crate::cloid_for`] derives the client id from `(ts_event, seq, symbol)` and nothing
/// else: without the tag, a netted target and a single producer's record could mint the
/// same id for two different orders, and the venue would de-duplicate the second into
/// nothing while every counter here reported success — the trap ADR-0036 and
/// `flatten.rs` have each hit by a different route.
pub const NET_SEQ_TAG: u64 = 1 << 63;

/// Which producer a claim came from.
///
/// A `u16` index into the session's declared producer list, not a hash of a name: the
/// list is config, the order is stable, and an index is comparable and printable without
/// an allocation on the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StrategyId(u16);

impl StrategyId {
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// What to do with a claim whose author has stopped speaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnSilence {
    /// Keep the claim. The account goes on holding what the strategy last asked for.
    ///
    /// The default, and the behaviour of every session written before this module: a
    /// target position is idempotent, so "said nothing" has always meant "unchanged".
    /// Its cost is real and is why the silence is surfaced: a crashed producer's
    /// exposure stays on the book until an operator acts.
    #[default]
    Hold,
    /// Drop the claim to zero once the strategy has been silent past its window.
    ///
    /// A trading decision, not a cleanup: it flattens that strategy's share into
    /// whatever market it went silent in. Defensible for a producer whose opinion has a
    /// known shelf life (a bar strategy that must speak once a bar), and dangerous for
    /// one whose silence is ordinary.
    Flat,
}

/// How to treat two strategies claiming one instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overlap {
    /// Refuse the second claim. The default — see the module docs.
    #[default]
    Exclusive,
    /// Add the claims and work toward the sum.
    Net,
}

/// One producer's standing, as the operator declared it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyPolicy {
    pub id: StrategyId,
    pub on_silence: OnSilence,
    /// How long a strategy may say nothing before it is called silent, in nanoseconds of
    /// **event time**. `0` is "never call it silent", which is the only reading that is
    /// safe for a field nobody wrote — the same rule `ttl_ms` and `max_order_age_ms`
    /// turn on.
    pub silence_ns: Nanos,
}

impl StrategyPolicy {
    pub fn new(id: StrategyId) -> Self {
        Self {
            id,
            on_silence: OnSilence::Hold,
            silence_ns: 0,
        }
    }
}

/// Why a claim was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BookReject {
    #[error("{symbol} is already claimed by {holder} and overlap is exclusive")]
    Overlap {
        symbol: SymbolId,
        holder: StrategyId,
    },
    #[error("{strategy} is not a producer this session declared")]
    UnknownStrategy { strategy: StrategyId },
}

/// Everything the book has counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BookStats {
    /// Symbols planned from a sum of two or more claims.
    pub netted: u64,
    /// Claims refused because another strategy already held the instrument.
    pub refused_overlap: u64,
    /// Claims zeroed because their author had gone silent past its window.
    pub silent_flat: u64,
    /// Contributions kept even though their author had gone silent
    /// ([`OnSilence::Hold`]). Not an error, and it must never be invisible: this is
    /// exposure nobody is currently speaking for.
    pub silent_held: u64,
    /// Netted targets that dropped a price band because the contributors disagreed
    /// about it. See [`TargetBook::net`].
    pub band_dropped: u64,
    /// Sums that could not be represented on the wire's fixed-point scale and were
    /// clamped. A `u64` of these is a session whose targets are meaningless, not one
    /// that is slightly off.
    pub saturated: u64,
}

/// One strategy's standing claim on one instrument.
#[derive(Debug, Clone, Copy)]
struct Claim {
    strategy: StrategyId,
    signal: Signal,
    /// Event time at which the strategy last stated this. Not the record's own
    /// `ts_event`: a claim re-stated by a pass that could not act on it earlier is
    /// still a claim its author is currently making, and silence is about the author
    /// rather than about the decision.
    stated_ns: Nanos,
}

/// Per-symbol state that is not any one strategy's.
#[derive(Debug, Clone, Copy)]
struct SymbolState {
    symbol_id: u32,
    /// Monotonic, bumped every time a multi-contributor target is synthesized.
    ///
    /// Advanced on every synthesis rather than only on a *change*, because a
    /// re-planned target that reused its predecessor's `seq` would reuse its `cloid`,
    /// and the venue de-duplicates a repeated `cloid` into nothing while every counter
    /// reports success. Churn is not the cost it looks like: the planner compares a
    /// resting order field by field and never by id, so an unchanged target still
    /// resolves to `AlreadyWorking`.
    net_seq: u64,
    /// Re-quotes issued since **any** contributor last spoke. Per symbol rather than
    /// per strategy because a re-quote is about the one order working toward the one
    /// netted target, and a strategy that is talking is the evidence the budget waits
    /// for however many others are quiet.
    requotes: u32,
}

/// What one symbol's claims add up to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetTarget {
    /// The record to plan. Either a contributor's own, unchanged, or a synthesis.
    pub signal: Signal,
    /// How many claims went into it, counting the ones a silence policy zeroed.
    pub contributors: u8,
    /// How many of those came from a strategy that has gone silent.
    pub silent: u8,
    /// Whether [`Self::signal`] was built here rather than passed through.
    pub synthesized: bool,
}

/// Every strategy's standing claim, and what they add up to per instrument.
#[derive(Debug, Clone, Default)]
pub struct TargetBook {
    policies: Vec<StrategyPolicy>,
    claims: Vec<Claim>,
    symbols: Vec<SymbolState>,
    overlap: Overlap,
    stats: BookStats,
}

impl TargetBook {
    /// A book over the producers the session declared.
    ///
    /// The policy list is the authority on which strategies exist: [`Self::state`]
    /// refuses an id that is not in it, so a producer added to a ring path and forgotten
    /// in the config cannot contribute exposure nothing declared.
    pub fn new(policies: Vec<StrategyPolicy>, overlap: Overlap) -> Self {
        Self {
            policies,
            claims: Vec::new(),
            symbols: Vec::new(),
            overlap,
            stats: BookStats::default(),
        }
    }

    /// The single-producer book: one strategy, no overlap question to answer.
    pub fn single() -> Self {
        Self::new(
            vec![StrategyPolicy::new(StrategyId::new(0))],
            Overlap::Exclusive,
        )
    }

    pub fn stats(&self) -> BookStats {
        self.stats
    }

    pub fn overlap(&self) -> Overlap {
        self.overlap
    }

    pub fn policies(&self) -> &[StrategyPolicy] {
        &self.policies
    }

    /// Record what `strategy` now wants in `sig.symbol_id`, replacing its previous claim
    /// on that instrument.
    ///
    /// **Replacing, not accumulating** — a target position is what its author wants held,
    /// so the newest one carries everything an older one did. This is the same rule the
    /// pass already applied within a drain, moved to where it can also hold across
    /// passes and across producers.
    pub fn state(
        &mut self,
        strategy: StrategyId,
        sig: &Signal,
        now: Nanos,
    ) -> Result<(), BookReject> {
        if !self.policies.iter().any(|p| p.id == strategy) {
            return Err(BookReject::UnknownStrategy { strategy });
        }
        let symbol = sig.symbol_id;
        if self.overlap == Overlap::Exclusive {
            if let Some(other) = self
                .claims
                .iter()
                .find(|c| c.signal.symbol_id == symbol && c.strategy != strategy)
            {
                self.stats.refused_overlap += 1;
                return Err(BookReject::Overlap {
                    symbol: SymbolId::new(symbol),
                    holder: other.strategy,
                });
            }
        }
        match self
            .claims
            .iter_mut()
            .find(|c| c.strategy == strategy && c.signal.symbol_id == symbol)
        {
            Some(existing) => {
                existing.signal = *sig;
                existing.stated_ns = now;
            }
            None => self.claims.push(Claim {
                strategy,
                signal: *sig,
                stated_ns: now,
            }),
        }
        // A producer that is talking is what the re-quote budget waits for, so it resets
        // here — for the symbol, not for the strategy, because the budget belongs to the
        // one order working toward the one netted target.
        self.symbol_mut(symbol).requotes = 0;
        Ok(())
    }

    /// Every instrument any strategy currently claims, in first-claimed order.
    ///
    /// Deterministic rather than sorted, and deliberately so: the order decides which
    /// symbols a breadth cap admits, and a replay has to reach the same verdict as the
    /// session it reproduces. First-claimed is a function of the record stream; a
    /// `HashSet`'s iteration order is a function of the allocator.
    pub fn claimed_symbols(&self) -> impl Iterator<Item = u32> + '_ {
        self.claims
            .iter()
            .map(|c| c.signal.symbol_id)
            .enumerate()
            .filter(|(i, s)| {
                !self.claims[..*i]
                    .iter()
                    .any(|earlier| earlier.signal.symbol_id == *s)
            })
            .map(|(_, s)| s)
    }

    /// Whether any strategy claims `symbol`.
    pub fn is_claimed(&self, symbol: u32) -> bool {
        self.claims.iter().any(|c| c.signal.symbol_id == symbol)
    }

    /// What `strategy` last asked for in `symbol`, if anything.
    pub fn claim(&self, strategy: StrategyId, symbol: u32) -> Option<&Signal> {
        self.claims
            .iter()
            .find(|c| c.strategy == strategy && c.signal.symbol_id == symbol)
            .map(|c| &c.signal)
    }

    /// Event time at which `strategy` last stated anything at all, or `None` if it never
    /// has. What the status line reports a silence from.
    pub fn last_stated(&self, strategy: StrategyId) -> Option<Nanos> {
        self.claims
            .iter()
            .filter(|c| c.strategy == strategy)
            .map(|c| c.stated_ns)
            .max()
    }

    /// Drop every claim on `symbol`. The operator's flatten, expressed in the book.
    pub fn forget_symbol(&mut self, symbol: u32) {
        self.claims.retain(|c| c.signal.symbol_id != symbol);
    }

    /// How many times the pass has re-quoted `symbol`'s target since anyone spoke for it.
    pub fn requotes(&self, symbol: u32) -> u32 {
        self.symbols
            .iter()
            .find(|s| s.symbol_id == symbol)
            .map(|s| s.requotes)
            .unwrap_or(0)
    }

    pub fn note_requote(&mut self, symbol: u32) {
        self.symbol_mut(symbol).requotes += 1;
    }

    /// Give a spent re-quote back, for a plan that never reached the venue.
    ///
    /// Saturating rather than wrapping, and it has to be: charging a queue failure to
    /// the budget would spend a session's re-quotes on an outage that lasted one pass,
    /// and an underflow here would hand it an unbounded supply of them instead — the
    /// opposite failure, from the same line.
    pub fn forgive_requote(&mut self, symbol: u32) {
        let s = self.symbol_mut(symbol);
        s.requotes = s.requotes.saturating_sub(1);
    }

    /// What `symbol`'s claims add up to, or `None` when nobody claims it.
    ///
    /// `adjust` is applied to each contribution before the sum, and is how a per-strategy
    /// allocation reaches a book that holds no prices: the caller — which does hold the
    /// marks — clamps a strategy's claim to the notional it was allotted, and the netting
    /// rules below are unchanged by it. Identity is the ordinary case
    /// ([`Self::net_raw`]).
    ///
    /// ## The combining rules, and the failure each one prevents
    ///
    /// - **`target_qty` sums**, saturating at the wire's `i64` and counting it. Two
    ///   claims add because they are two claims on one position.
    /// - **`ts_event` is the newest contributor's.** It is the moment the netted target
    ///   became what it is, which is what a latency stage measures and what
    ///   `SignalReader::still_fresh` ages. Taking the oldest would make the whole net as
    ///   stale as its quietest member and expire a target three strategies just restated.
    /// - **`urgency` is the highest.** The most impatient contributor decides how the
    ///   netted delta is worked, because urgency is most often a statement about risk and
    ///   under-executing an urgent exit is the failure with no bound on it. The cost is
    ///   real and is stated rather than hidden: one strategy can put the whole net across
    ///   the spread.
    /// - **`reduce_only` and `close` propagate only when *every* contributor sets them.**
    ///   One strategy closing its share is not the account flattening — its contribution
    ///   is zero and the others still stand. Marking the netted order reduce-only while
    ///   somebody is opening would have the venue refuse an order that was correct.
    /// - **A `price_band` is dropped unless every contributor states the same one**, and
    ///   the drop is counted. A band is a per-decision bound whose direction depends on
    ///   the side of the order, and the side of a *netted* order is not known until the
    ///   planner subtracts the position — so a band combined from disagreeing claims
    ///   would be enforced in whichever direction happened to apply. Dropping it is loud
    ///   (`band_dropped`, and a status-line warning) and leaves the order bounded by the
    ///   urgency table and the per-symbol risk gate; combining it silently would not be.
    /// - **`max_order_age_ms` and `ttl_ms` take the shortest anyone actually expressed**,
    ///   zeros filtered out first — the same arithmetic `Planner::order_lifetime_ns`
    ///   uses, and for the same reason: zero is the value of a field nobody wrote.
    pub fn net(
        &mut self,
        symbol: u32,
        now: Nanos,
        adjust: impl Fn(StrategyId, i64) -> i64,
    ) -> Option<NetTarget> {
        // Gathered first so the borrow of `self.claims` is over before `symbol_mut`.
        let fold = self.fold_claims(symbol, now, adjust);

        if fold.contributors == 0 {
            return None;
        }

        self.stats.silent_flat += fold.silent_flat;
        self.stats.silent_held += fold.silent_held;
        if fold.saturated {
            self.stats.saturated += 1;
        }

        // The single-contributor pass-through. Byte for byte the producer's own record,
        // so a session with one strategy — every session that has ever run — plans
        // exactly what it planned before this module existed, down to the `cloid`.
        //
        // Guarded on the *adjustment* as well: a clamped claim is no longer the record
        // its author wrote, and passing it through would put a size on the wire under an
        // id that says the strategy chose it.
        //
        // **A lone `FLAG_CLOSE` passes through unconditionally, and the arithmetic above
        // does not apply to it.** A close contributes zero *size* to a sum, which is
        // right when there are other claims to sum it with — its own share is nothing —
        // and is not a statement about the record. `Planner::plan` does not consult
        // `target_qty` on a close at all, so "the adjusted quantity differs from the
        // stated one" is not a fact about a close; and an allocation that scaled one
        // would be scaling a de-risking instruction *down*, which is the one direction
        // no bound in this project is allowed to move an exit.
        //
        // This was found by the golden replay rather than by argument: without the
        // exemption a single-producer session synthesized its own flatten, and the order
        // was identical in side, size, price and TIF under a **different `cloid`** — the
        // one field the venue keys idempotency on.
        if fold.contributors == 1 && fold.live == 1 {
            if let Some((sig, qty)) = fold.sole {
                if sig.is_close() || qty == sig.target_qty {
                    return Some(NetTarget {
                        signal: sig,
                        contributors: 1,
                        silent: fold.silent,
                        synthesized: false,
                    });
                }
            }
        }

        if fold.band_disagreed {
            self.stats.band_dropped += 1;
        }
        self.stats.netted += 1;
        let state = self.symbol_mut(symbol);
        state.net_seq += 1;
        let seq = NET_SEQ_TAG | state.net_seq;

        let mut flags = 0u16;
        if fold.all_reduce {
            flags |= FLAG_REDUCE_ONLY;
        }
        if fold.all_close {
            flags |= FLAG_CLOSE;
        }
        // **The stamp is the newest claim somebody is still speaking for, and the pass's
        // own clock when nobody is.** A target driven to zero because every contributor
        // went silent is a decision this runtime made *now*; stamping it with the dead
        // producer's last `ts_event` would hand the reader a record older than its own
        // admission window, so the one target that exists to unwind an abandoned
        // position would be refused as expired on the pass that created it.
        let ts_event = if fold.live > 0 {
            fold.newest_live_ts
        } else {
            now
        };
        let mut out = Signal::target_position(
            seq,
            ts_event,
            symbol,
            fold.sum,
            fold.urgency,
            if fold.band_disagreed {
                0
            } else {
                fold.band.unwrap_or(0)
            },
            fold.ttl.unwrap_or(0),
            fold.model_version,
            flags,
        )
        .with_max_order_age_ms(fold.age.unwrap_or(0));
        if fold.newest_cause > 0 {
            out = out.with_ts_cause(fold.newest_cause);
        }
        Some(NetTarget {
            signal: out,
            contributors: fold.contributors,
            silent: fold.silent,
            synthesized: true,
        })
    }

    /// What `symbol` would net to, **without minting anything**.
    ///
    /// The allocator needs this and must not use [`Self::net`]: that advances the
    /// symbol's `net_seq` on every call, and an allocator asking "what would the whole
    /// book weigh" would burn one sequence per symbol per pass before a single order was
    /// planned — which is not wrong so much as it is a counter that stops describing what
    /// it is named after. Nothing here is recorded and no counter moves, so calling it
    /// twice is free and calling it never changes nothing.
    pub fn net_qty(
        &self,
        symbol: u32,
        now: Nanos,
        adjust: impl Fn(StrategyId, i64) -> i64,
    ) -> Option<i64> {
        let fold = self.fold_claims(symbol, now, adjust);
        (fold.contributors > 0).then_some(fold.sum)
    }

    /// The whole of the combining logic, over `&self` and touching no counter.
    ///
    /// Split out of [`Self::net`] so [`Self::net_qty`] can ask the same question without
    /// the side effects — two implementations of "what do these claims add up to" is
    /// exactly the shape this module exists to prevent one level up.
    fn fold_claims(
        &self,
        symbol: u32,
        now: Nanos,
        adjust: impl Fn(StrategyId, i64) -> i64,
    ) -> Fold {
        let mut fold = Fold::default();

        for claim in self.claims.iter().filter(|c| c.signal.symbol_id == symbol) {
            fold.contributors = fold.contributors.saturating_add(1);
            let policy = self
                .policies
                .iter()
                .find(|p| p.id == claim.strategy)
                .copied()
                .unwrap_or_else(|| StrategyPolicy::new(claim.strategy));
            // Silence is measured on the pass's event clock against the moment the
            // strategy last spoke — the same subtraction, and the same signed
            // comparison, the pass schedule and the sweeper use. A late arrival walks
            // `now` backwards and can therefore only ever make a strategy look *less*
            // silent, which is the direction that keeps a claim rather than dropping one.
            let quiet =
                policy.silence_ns > 0 && now.saturating_sub(claim.stated_ns) > policy.silence_ns;
            // A zeroed claim is one whose author has stopped speaking and whose operator
            // asked for that to mean flat. It contributes no *size* and is deliberately
            // still a contributor: the symbol stays claimed, so the pass keeps planning
            // it — which is how the account gets *to* zero rather than merely stopping
            // being asked to.
            let zeroed = quiet && policy.on_silence == OnSilence::Flat;
            if quiet {
                fold.silent = fold.silent.saturating_add(1);
                if zeroed {
                    fold.silent_flat += 1;
                } else {
                    fold.silent_held += 1;
                }
            }

            // A closing claim contributes nothing rather than flattening the account.
            // `FLAG_CLOSE` means "the target field is not consulted" (see `Planner::plan`),
            // and for one claim among several that means its own share is zero.
            let qty = if zeroed || claim.signal.is_close() {
                0
            } else {
                adjust(claim.strategy, claim.signal.target_qty)
            };
            match fold.sum.checked_add(qty) {
                Some(v) => fold.sum = v,
                None => {
                    fold.saturated = true;
                    fold.sum = if qty > 0 { i64::MAX } else { i64::MIN };
                }
            }

            // The metadata fold takes **every** claim, zeroed ones included: a strategy
            // whose share is being driven to zero still decided how it wants that worked
            // (its urgency), how long the order may rest, and whether it may only
            // shrink. What a zeroed claim must not contribute is its *stamp* — see
            // `newest_live_ts` below.
            fold.newest_cause = fold.newest_cause.max(claim.signal.ts_cause);
            fold.urgency = fold.urgency.max(claim.signal.urgency);
            fold.all_reduce &= claim.signal.is_reduce_only();
            fold.all_close &= claim.signal.is_close();
            match fold.band {
                None => fold.band = Some(claim.signal.price_band),
                Some(b) if b != claim.signal.price_band => fold.band_disagreed = true,
                Some(_) => {}
            }
            fold.ttl = shortest(fold.ttl, claim.signal.ttl_ms);
            fold.age = shortest(fold.age, claim.signal.max_order_age_ms);
            if claim.signal.ts_event >= fold.newest_ts {
                fold.newest_ts = claim.signal.ts_event;
                fold.model_version = claim.signal.model_version;
            }
            if !zeroed {
                fold.live = fold.live.saturating_add(1);
                fold.sole = Some((claim.signal, qty));
                fold.newest_live_ts = fold.newest_live_ts.max(claim.signal.ts_event);
            }
        }
        fold
    }

    /// [`Self::net`] with no per-strategy adjustment.
    pub fn net_raw(&mut self, symbol: u32, now: Nanos) -> Option<NetTarget> {
        self.net(symbol, now, |_, qty| qty)
    }

    fn symbol_mut(&mut self, symbol: u32) -> &mut SymbolState {
        if let Some(i) = self.symbols.iter().position(|s| s.symbol_id == symbol) {
            return &mut self.symbols[i];
        }
        self.symbols.push(SymbolState {
            symbol_id: symbol,
            net_seq: 0,
            requotes: 0,
        });
        self.symbols.last_mut().expect("just pushed")
    }
}

/// One symbol's claims, part-way through being added up.
///
/// A struct rather than sixteen locals because the loop that fills it is the only
/// interesting thing in this module and a wall of `let mut` at the top of it is where a
/// reader stops reading.
struct Fold {
    /// Claims seen, including the ones a silence policy zeroed.
    contributors: u8,
    /// …of which came from a strategy past its silence window.
    silent: u8,
    /// …of which were zeroed by it, and merely kept.
    silent_flat: u64,
    silent_held: u64,
    /// Claims that contributed a size. Zero means every contributor is being unwound.
    live: u8,
    /// The last live claim seen and the size it contributed, for the pass-through test.
    sole: Option<(Signal, i64)>,
    sum: i64,
    saturated: bool,
    /// Newest stamp over every claim — what `model_version` is taken from, because the
    /// audit question is "which model most recently had an opinion here".
    newest_ts: i64,
    /// Newest stamp over the claims somebody is still speaking for. See the comment at
    /// the assignment for why the two are different numbers.
    newest_live_ts: i64,
    model_version: u32,
    newest_cause: i64,
    urgency: u8,
    all_reduce: bool,
    all_close: bool,
    band: Option<i64>,
    band_disagreed: bool,
    ttl: Option<u32>,
    age: Option<u32>,
}

impl Default for Fold {
    fn default() -> Self {
        Self {
            contributors: 0,
            silent: 0,
            silent_flat: 0,
            silent_held: 0,
            live: 0,
            sole: None,
            sum: 0,
            saturated: false,
            newest_ts: i64::MIN,
            newest_live_ts: i64::MIN,
            model_version: 0,
            newest_cause: 0,
            urgency: 0,
            // `and`-folds over an empty set, so both start true and a single claim that
            // lacks the flag is enough to clear them.
            all_reduce: true,
            all_close: true,
            band: None,
            band_disagreed: false,
            ttl: None,
            age: None,
        }
    }
}

/// The shortest duration anyone actually expressed, ignoring the zeros.
///
/// Zero means "I set no bound" on both `ttl_ms` and `max_order_age_ms`, so a naive
/// `min` would let one contributor that never wrote the field impose an immediate
/// expiry on every other. The same inversion `Planner::order_lifetime_ns` documents,
/// arriving here from a different direction.
fn shortest(current: Option<u32>, candidate: u32) -> Option<u32> {
    match (current, candidate) {
        (c, 0) => c,
        (None, v) => Some(v),
        (Some(c), v) => Some(c.min(v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::Signal;

    const A: StrategyId = StrategyId::new(0);
    const B: StrategyId = StrategyId::new(1);
    const C: StrategyId = StrategyId::new(2);
    const BTC: u32 = 3;
    const ETH: u32 = 4;

    fn sig(seq: u64, ts: i64, symbol: u32, qty: i64) -> Signal {
        Signal::target_position(seq, ts, symbol, qty, 0, 0, 60_000, 1, 0)
    }

    fn book() -> TargetBook {
        TargetBook::new(
            vec![
                StrategyPolicy::new(A),
                StrategyPolicy::new(B),
                StrategyPolicy::new(C),
            ],
            Overlap::Net,
        )
    }

    #[test]
    fn one_strategys_target_passes_through_byte_for_byte() {
        // The property that makes this module safe to land: every session that has ever
        // run has one producer, so every one of them must plan exactly what it planned
        // before — same seq, same ts_event, therefore the same cloid, therefore the same
        // order at the venue.
        let mut b = book();
        let s = sig(7, 1_000, BTC, 50_000);
        b.state(A, &s, 1_000).unwrap();
        let net = b.net_raw(BTC, 1_000).unwrap();
        assert!(!net.synthesized);
        assert_eq!(net.signal, s, "the producer's own record, unaltered");
        assert_eq!(net.contributors, 1);
        assert_eq!(b.stats().netted, 0, "nothing was netted");
    }

    #[test]
    fn a_lone_close_passes_through_so_a_flatten_keeps_its_own_cloid() {
        // Found by the golden replay, not by argument. Without the exemption a
        // single-producer session synthesized its own flatten: the order was identical in
        // side, size, price and TIF under a **different `cloid`** — the one field the
        // venue keys idempotency on, and the field a capture's whole trace is compared by.
        //
        // The rule underneath it: `Planner::plan` does not consult `target_qty` on a
        // close, so "the adjusted quantity differs from the stated one" is not a fact
        // about a close. And an allocation that scaled one would be scaling a de-risking
        // instruction *down*.
        let mut b = book();
        let mut closing = sig(4, 1_000, BTC, 30_000);
        closing.flags |= FLAG_CLOSE;
        b.state(A, &closing, 1_000).unwrap();

        let net = b.net_raw(BTC, 1_000).unwrap();
        assert!(!net.synthesized);
        assert_eq!(net.signal, closing, "the producer's own record, unaltered");

        // …including under an allocation that would otherwise clamp it, because the
        // clamp has nothing to bite on: a close asks for flat.
        let mut b2 = book();
        b2.state(A, &closing, 1_000).unwrap();
        let scaled = b2.net(BTC, 1_000, |_, qty| qty / 10).unwrap();
        assert!(!scaled.synthesized);
        assert_eq!(scaled.signal, closing);
    }

    #[test]
    fn two_claims_on_one_instrument_add_rather_than_overwrite() {
        // The failure this prevents is silent and is what a naive "newest signal per
        // symbol wins" does: both producers' counters climb, both believe they are
        // positioned, and the account holds whichever spoke last.
        let mut b = book();
        b.state(A, &sig(1, 1_000, BTC, 30_000), 1_000).unwrap();
        b.state(B, &sig(1, 1_100, BTC, -10_000), 1_100).unwrap();
        let net = b.net_raw(BTC, 1_100).unwrap();
        assert!(net.synthesized);
        assert_eq!(net.signal.target_qty, 20_000);
        assert_eq!(net.contributors, 2);
        assert_eq!(
            net.signal.ts_event, 1_100,
            "the newest claim's stamp: the moment the net became what it is"
        );
        // Disjoint id space, so a netted cloid can never collide with a producer's.
        assert!(net.signal.seq & NET_SEQ_TAG != 0);
    }

    #[test]
    fn a_synthesized_seq_advances_on_every_synthesis_so_no_cloid_is_ever_reused() {
        // `cloid_for` is (ts_event, seq, symbol) and nothing else. Two plans that shared
        // a seq at one ts_event would mint one id for two orders, and the venue would
        // de-duplicate the second into nothing while every counter reported success —
        // the same trap ADR-0036's re-quote and flatten.rs's ladder each hit.
        let mut b = book();
        b.state(A, &sig(1, 1_000, BTC, 30_000), 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, 10_000), 1_000).unwrap();
        let first = b.net_raw(BTC, 1_000).unwrap().signal.seq;
        let second = b.net_raw(BTC, 1_000).unwrap().signal.seq;
        assert!(second > first, "even with nothing changed");

        // …and the counter is per symbol, so two instruments do not share a sequence.
        b.state(A, &sig(2, 1_000, ETH, 5), 1_000).unwrap();
        b.state(B, &sig(2, 1_000, ETH, 5), 1_000).unwrap();
        let eth = b.net_raw(ETH, 1_000).unwrap().signal.seq;
        assert_eq!(eth, NET_SEQ_TAG | 1);
    }

    #[test]
    fn exclusive_overlap_refuses_the_second_claim_and_names_the_holder() {
        // Two producers on one instrument is far more often a copy-pasted config than a
        // decision, and netting them silently composes two strategies' risk into a
        // position neither author sized.
        let mut b = TargetBook::new(
            vec![StrategyPolicy::new(A), StrategyPolicy::new(B)],
            Overlap::Exclusive,
        );
        b.state(A, &sig(1, 1_000, BTC, 30_000), 1_000).unwrap();
        let err = b.state(B, &sig(1, 1_100, BTC, 10_000), 1_100).unwrap_err();
        assert_eq!(
            err,
            BookReject::Overlap {
                symbol: SymbolId::new(BTC),
                holder: A
            }
        );
        assert_eq!(b.stats().refused_overlap, 1);
        // A's own claim is untouched, and re-stating it is still fine.
        assert_eq!(b.net_raw(BTC, 1_100).unwrap().signal.target_qty, 30_000);
        assert!(b.state(A, &sig(2, 1_200, BTC, 40_000), 1_200).is_ok());
    }

    #[test]
    fn a_producer_the_session_never_declared_cannot_contribute_exposure() {
        // A ring path added and a producer entry forgotten is a live producer nothing
        // declared, and its claims would be exposure with no policy, no allocation and
        // no name on the status line.
        let mut b = TargetBook::new(vec![StrategyPolicy::new(A)], Overlap::Net);
        assert_eq!(
            b.state(C, &sig(1, 1_000, BTC, 1), 1_000).unwrap_err(),
            BookReject::UnknownStrategy { strategy: C }
        );
        assert!(b.net_raw(BTC, 1_000).is_none());
    }

    #[test]
    fn a_silent_strategy_holds_its_claim_by_default_and_the_silence_is_counted() {
        // Hold is the behaviour every session before this module had, because a target
        // position is idempotent and "said nothing" has always meant "unchanged". Its
        // cost — exposure nobody is speaking for — must be visible rather than implied.
        let mut b = TargetBook::new(
            vec![StrategyPolicy {
                id: A,
                on_silence: OnSilence::Hold,
                silence_ns: 1_000,
            }],
            Overlap::Net,
        );
        b.state(A, &sig(1, 0, BTC, 30_000), 0).unwrap();
        let net = b.net_raw(BTC, 5_000).unwrap();
        assert_eq!(net.signal.target_qty, 30_000, "still held");
        assert_eq!(net.silent, 1);
        assert_eq!(b.stats().silent_held, 1);
    }

    #[test]
    fn a_silent_strategy_under_flat_contributes_zero_and_the_symbol_is_still_planned() {
        // Zeroing the claim is only half of it: the symbol has to keep being planned, or
        // the account never gets *to* zero — it just stops being asked to.
        let mut b = TargetBook::new(
            vec![
                StrategyPolicy {
                    id: A,
                    on_silence: OnSilence::Flat,
                    silence_ns: 1_000,
                },
                StrategyPolicy::new(B),
            ],
            Overlap::Net,
        );
        b.state(A, &sig(1, 0, BTC, 30_000), 0).unwrap();
        let net = b.net_raw(BTC, 5_000).unwrap();
        assert_eq!(net.signal.target_qty, 0);
        assert_eq!(net.contributors, 1, "still a claim, still planned");
        assert_eq!(b.stats().silent_flat, 1);

        // And it comes straight back when the strategy speaks again.
        b.state(A, &sig(2, 5_000, BTC, 30_000), 5_000).unwrap();
        assert_eq!(b.net_raw(BTC, 5_100).unwrap().signal.target_qty, 30_000);
    }

    #[test]
    fn a_zero_silence_window_never_calls_anyone_silent() {
        // Zero is the value of a field nobody wrote, so it has to be the safe reading —
        // the same rule ttl_ms and max_order_age_ms turn on. "Silent immediately" would
        // flatten every strategy that had not spoken in this exact nanosecond.
        let mut b = book(); // every policy defaults to silence_ns = 0
        b.state(A, &sig(1, 0, BTC, 30_000), 0).unwrap();
        let net = b.net_raw(BTC, i64::MAX / 2).unwrap();
        assert_eq!(net.silent, 0);
        assert_eq!(net.signal.target_qty, 30_000);
    }

    #[test]
    fn one_strategys_close_zeroes_its_own_share_and_never_the_account() {
        // FLAG_CLOSE means "do not consult the target field". For one claim among
        // several that is a statement about its own share; propagating it would flatten
        // exposure two other strategies are actively asking for.
        let mut b = book();
        let mut closing = sig(1, 1_000, BTC, 30_000);
        closing.flags |= FLAG_CLOSE;
        b.state(A, &closing, 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, 20_000), 1_000).unwrap();
        let net = b.net_raw(BTC, 1_000).unwrap();
        assert_eq!(net.signal.target_qty, 20_000, "only B's share survives");
        assert!(!net.signal.is_close(), "the account is not flattening");

        // When *everybody* closes, the flag propagates — and then it is the right one,
        // because a close implies reduce-only and cannot overshoot into a flip.
        let mut b2 = book();
        b2.state(A, &closing, 1_000).unwrap();
        let mut closing_b = sig(1, 1_000, BTC, 20_000);
        closing_b.flags |= FLAG_CLOSE;
        b2.state(B, &closing_b, 1_000).unwrap();
        assert!(b2.net_raw(BTC, 1_000).unwrap().signal.is_close());
    }

    #[test]
    fn reduce_only_propagates_only_when_every_contributor_asked_for_it() {
        // Marking a netted order reduce-only while one strategy is opening would have
        // the venue refuse an order that was correct — and a venue rejection reads in
        // the log exactly like an outage.
        let mut b = book();
        let mut ro = sig(1, 1_000, BTC, -10_000);
        ro.flags |= FLAG_REDUCE_ONLY;
        b.state(A, &ro, 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, 30_000), 1_000).unwrap();
        assert!(!b.net_raw(BTC, 1_000).unwrap().signal.is_reduce_only());

        let mut b2 = book();
        let mut ro_b = sig(1, 1_000, BTC, -5_000);
        ro_b.flags |= FLAG_REDUCE_ONLY;
        b2.state(A, &ro, 1_000).unwrap();
        b2.state(B, &ro_b, 1_000).unwrap();
        assert!(b2.net_raw(BTC, 1_000).unwrap().signal.is_reduce_only());
    }

    #[test]
    fn the_most_impatient_contributor_decides_how_the_net_is_worked() {
        // One order comes out, so one urgency goes in. The highest, because urgency is
        // most often a statement about risk and under-executing an urgent exit is the
        // failure with no bound on it — at the stated cost that one strategy can put the
        // whole net across the spread.
        let mut b = book();
        let mut urgent = sig(1, 1_000, BTC, -30_000);
        urgent.urgency = 3;
        b.state(A, &urgent, 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, 10_000), 1_000).unwrap();
        assert_eq!(b.net_raw(BTC, 1_000).unwrap().signal.urgency, 3);
    }

    #[test]
    fn a_price_band_the_contributors_disagree_about_is_dropped_and_counted() {
        // A band is a ceiling for a buy and a floor for a sell, and the side of a netted
        // order is not known until the planner subtracts the position — so a band
        // combined from disagreeing claims would be enforced in whichever direction
        // happened to apply. Dropping it is loud; combining it would not be.
        let mut b = book();
        let mut a = sig(1, 1_000, BTC, 30_000);
        a.price_band = 6_000_000_000_000;
        let mut c = sig(1, 1_000, BTC, 10_000);
        c.price_band = 5_000_000_000_000;
        b.state(A, &a, 1_000).unwrap();
        b.state(B, &c, 1_000).unwrap();
        let net = b.net_raw(BTC, 1_000).unwrap();
        assert_eq!(net.signal.price_band, 0);
        assert_eq!(b.stats().band_dropped, 1);

        // Agreement is carried through, so two strategies that share a bound keep it.
        let mut b2 = book();
        b2.state(A, &a, 1_000).unwrap();
        let mut same = sig(1, 1_000, BTC, 10_000);
        same.price_band = a.price_band;
        b2.state(B, &same, 1_000).unwrap();
        assert_eq!(
            b2.net_raw(BTC, 1_000).unwrap().signal.price_band,
            a.price_band
        );
        assert_eq!(b2.stats().band_dropped, 0);
    }

    #[test]
    fn the_binding_lifetime_is_the_shortest_anyone_expressed_and_zero_is_not_one() {
        // The inversion this prevents: a naive `min` would let one contributor that never
        // wrote the field impose an immediate expiry on every other. Exactly the mistake
        // `Planner::order_lifetime_ns` documents, arriving from a different direction.
        let mut b = book();
        let mut a = sig(1, 1_000, BTC, 30_000).with_max_order_age_ms(0);
        a.ttl_ms = 0;
        let c = sig(1, 1_000, BTC, 10_000).with_max_order_age_ms(15_000);
        b.state(A, &a, 1_000).unwrap();
        b.state(B, &c, 1_000).unwrap();
        let net = b.net_raw(BTC, 1_000).unwrap();
        assert_eq!(net.signal.max_order_age_ms, 15_000);
        assert_eq!(net.signal.ttl_ms, 60_000, "A's zero is not an opinion");

        assert_eq!(shortest(None, 0), None);
        assert_eq!(shortest(Some(10), 0), Some(10));
        assert_eq!(shortest(Some(10), 5), Some(5));
    }

    #[test]
    fn a_per_strategy_clamp_stops_a_pass_through_because_the_record_is_no_longer_its_authors() {
        // A clamped claim put on the wire under the author's own seq would say the
        // strategy chose that size. It did not — the allocator did.
        let mut b = book();
        b.state(A, &sig(9, 1_000, BTC, 30_000), 1_000).unwrap();
        let net = b.net(BTC, 1_000, |_, qty| qty / 3).unwrap();
        assert!(net.synthesized);
        assert_eq!(net.signal.target_qty, 10_000);
        assert!(net.signal.seq & NET_SEQ_TAG != 0);

        // …and an adjustment that changes nothing still passes through.
        let mut b2 = book();
        b2.state(A, &sig(9, 1_000, BTC, 30_000), 1_000).unwrap();
        assert!(!b2.net(BTC, 1_000, |_, qty| qty).unwrap().synthesized);
    }

    #[test]
    fn a_sum_that_overflows_the_wire_is_clamped_and_counted_rather_than_wrapping() {
        // Wrapping would turn two enormous longs into a short. There is no sane target
        // here either way; the point is that the counter says so.
        let mut b = book();
        b.state(A, &sig(1, 1_000, BTC, i64::MAX), 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, i64::MAX), 1_000).unwrap();
        let net = b.net_raw(BTC, 1_000).unwrap();
        assert_eq!(net.signal.target_qty, i64::MAX);
        assert_eq!(b.stats().saturated, 1);
    }

    #[test]
    fn claimed_symbols_is_first_claimed_order_and_lists_each_instrument_once() {
        // The order decides which symbols a breadth cap admits, so it has to be a
        // function of the record stream rather than of the allocator — a replay must
        // reach the same verdict as the session it reproduces.
        let mut b = book();
        b.state(A, &sig(1, 1_000, ETH, 1), 1_000).unwrap();
        b.state(B, &sig(1, 1_000, BTC, 1), 1_000).unwrap();
        b.state(C, &sig(1, 1_000, ETH, 1), 1_000).unwrap();
        assert_eq!(b.claimed_symbols().collect::<Vec<_>>(), vec![ETH, BTC]);
    }

    #[test]
    fn the_requote_budget_is_per_symbol_and_resets_when_anyone_speaks_for_it() {
        // Per symbol because a re-quote is about the one order working toward the one
        // netted target; reset by any contributor because a strategy that is talking is
        // the evidence the budget waits for, however many others are quiet.
        let mut b = book();
        b.state(A, &sig(1, 1_000, BTC, 1), 1_000).unwrap();
        b.note_requote(BTC);
        b.note_requote(BTC);
        assert_eq!(b.requotes(BTC), 2);
        assert_eq!(b.requotes(ETH), 0, "another instrument is untouched");
        b.state(B, &sig(1, 1_100, BTC, 1), 1_100).unwrap();
        assert_eq!(b.requotes(BTC), 0);
    }
}
