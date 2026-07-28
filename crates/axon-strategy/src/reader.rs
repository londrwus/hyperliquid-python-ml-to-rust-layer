//! [`SignalReader`] — the gate every Python signal crosses before it is allowed to
//! mean anything.
//!
//! The ring ([ADR-0006]) delivers 64 bytes. Those bytes are whatever the producer
//! wrote: a build of `axon.signals` from a different commit, a record from a
//! process that restarted mid-stream, or a decision made about a market that no
//! longer exists. Handing them straight to the planner would make the boundary's
//! versioning and TTL fields decorative. So the reader validates first and counts
//! everything it refuses.
//!
//! Five failure modes shape it; each has a test named after it:
//!
//! 1. **A layout that is not the one we compiled against.** `schema_version` is
//!    checked before any other field is read, because if the layout drifted then
//!    `target_qty` is not necessarily where we think it is, and a misread target is
//!    a position taken for no reason.
//! 2. **A record kind we do not implement.** `contracts/schema.toml` reserves
//!    `kind = 1` for an explicit order intent. Reading one as a target position
//!    would interpret an order-size field as a *position* field — same bytes,
//!    completely different meaning.
//! 3. **Unwritten bytes carrying data.** The record's tail is the extension slot,
//!    and since `max_order_age_ms` was carved out of it that slot is *two* runs —
//!    the 8-byte `reserved` and the 3-byte `pad0` in front of it. Non-zero in
//!    either means the producer is writing a field this build does not know about,
//!    and it means it *without* having bumped `schema_version` — the exact mistake
//!    the version field cannot catch on its own. Alignment padding a producer can
//!    write into hides an unversioned extension exactly as well as `reserved`
//!    does, and is easier to forget precisely because of its name.
//! 4. **A gap or a rewind in `seq`.** The ring is single-producer, so `seq` is one
//!    monotonic stream. A gap means the producer dropped records (a full ring) and
//!    is worth an alert; a rewind means a restart or a replay, and acting on it
//!    would move the target *backwards* in time.
//! 5. **A signal that outlived its TTL.** This is the one that looks like a
//!    judgement call and is not. A late target-position signal is not a weaker
//!    opinion about the current market — it is a firm opinion about a market that
//!    has already gone. Acting on it late is how a strategy buys the top of a move
//!    it correctly predicted and then correctly abandoned.
//!
//! [ADR-0006]: ../../../docs/adr/0006-signal-schema-and-spsc-ring.md

use axon_contracts::{Signal, KIND_TARGET_POSITION, SCHEMA_VERSION};
use axon_core::Nanos;
use axon_ipc::Consumer;
use thiserror::Error;

/// Nanoseconds per millisecond — `ttl_ms` is on the wire in ms, event time is ns.
const NS_PER_MS: i64 = 1_000_000;

/// How far ahead of our own clock a signal may be stamped before we count it as
/// clock skew. Not a rejection: the two processes stamp from different clocks and
/// a sub-millisecond disagreement is normal. Beyond it, TTL enforcement is
/// meaningless and an operator needs to know.
const CLOCK_SKEW_TOLERANCE_NS: i64 = NS_PER_MS;

/// Why a record was refused. Every variant is counted in [`SignalStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SignalReject {
    #[error(
        "schema_version {found} is not {expected}: not the layout this build compiled against"
    )]
    SchemaVersion { found: u8, expected: u8 },

    #[error(
        "record kind {kind} is not implemented (this build reads target-position records only)"
    )]
    UnknownKind { kind: u8 },

    #[error(
        "reserved bytes are not zero: the producer is writing a field this build does not know"
    )]
    ReservedNotZero,

    #[error("seq {seq} does not advance past {last}: a replayed or restarted producer")]
    StaleSeq { seq: u64, last: u64 },

    #[error("signal is {age_ms} ms old, past its {ttl_ms} ms validity window")]
    Expired { age_ms: i64, ttl_ms: i64 },
}

/// Counts of everything the reader has seen. The denominator for "are we actually
/// trading on the signals Python thinks it sent?".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SignalStats {
    /// Records that passed every check.
    pub accepted: u64,
    pub schema_version: u64,
    pub unknown_kind: u64,
    pub reserved_not_zero: u64,
    pub stale_seq: u64,
    pub expired: u64,
    /// Times `seq` jumped by more than one — i.e. distinct loss events.
    pub gaps: u64,
    /// Total records implied missing by those jumps.
    pub missing: u64,
    /// Records stamped further into the future than [`CLOCK_SKEW_TOLERANCE_NS`].
    /// Accepted, because refusing them would be refusing everything the moment the
    /// producer's clock drifts — but with the TTL check no longer meaning anything.
    pub ahead_of_clock: u64,
}

impl SignalStats {
    /// Total records refused, whatever the reason.
    pub fn rejected(&self) -> u64 {
        self.schema_version
            + self.unknown_kind
            + self.reserved_not_zero
            + self.stale_seq
            + self.expired
    }
}

/// Tunables for [`SignalReader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaderConfig {
    /// Hard ceiling on a signal's age, in milliseconds, regardless of its `ttl_ms`.
    ///
    /// Two things need this. A signal with `ttl_ms == 0` (the Python default, and
    /// what a strategy that never thought about staleness emits) would otherwise
    /// never expire. And a strategy that sets an absurd TTL should not be able to
    /// raise its own staleness limit above what the operator configured — the
    /// ceiling belongs to the process that has to answer for the fills.
    pub max_age_ms: u32,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        // Hyperliquid blocks are ~0.2 s (docs/04). Two seconds is several blocks of
        // slack for a mid-frequency strategy and still far short of "the market has
        // moved on"; a strategy needing more says so by raising this, not by
        // emitting a longer TTL.
        Self { max_age_ms: 2_000 }
    }
}

/// Where records come from. The production implementation is the ring
/// [`Consumer`]; [`ReplaySource`] backs offline tests and the replay harness.
///
/// A trait rather than a hard dependency on `Consumer` so validation can be
/// exercised — and a recorded session re-planned — without a memory-mapped file.
pub trait SignalSource {
    /// The next raw record, or `None` when the source is empty *for now*.
    fn next_signal(&mut self) -> Option<Signal>;
}

impl SignalSource for Consumer {
    #[inline]
    fn next_signal(&mut self) -> Option<Signal> {
        self.try_pop()
    }
}

/// An in-memory source: recorded or hand-built records, delivered in order.
#[derive(Debug, Clone, Default)]
pub struct ReplaySource {
    records: Vec<Signal>,
    next: usize,
}

impl ReplaySource {
    pub fn new(records: Vec<Signal>) -> Self {
        Self { records, next: 0 }
    }

    /// Append a record behind whatever has not been read yet.
    pub fn push(&mut self, sig: Signal) {
        self.records.push(sig);
    }

    /// Records not yet handed out.
    pub fn remaining(&self) -> usize {
        self.records.len() - self.next
    }
}

impl SignalSource for ReplaySource {
    fn next_signal(&mut self) -> Option<Signal> {
        let s = self.records.get(self.next).copied();
        if s.is_some() {
            self.next += 1;
        }
        s
    }
}

/// What one [`SignalReader::drain`] call did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DrainReport {
    /// Records taken off the source, accepted or not. `0` means the source was idle.
    pub read: usize,
    pub accepted: usize,
    pub rejected: usize,
}

/// Drains a [`SignalSource`] and admits only records this build can act on.
///
/// Synchronous and allocation-free by construction: it is called from the
/// deterministic core loop, which never touches tokio.
#[derive(Debug)]
pub struct SignalReader<S> {
    source: S,
    cfg: ReaderConfig,
    /// `None` until the first record. The first `seq` seen establishes the
    /// baseline rather than counting as a gap — a Rust process that starts against
    /// an already-running Python producer legitimately joins mid-stream.
    last_seq: Option<u64>,
    stats: SignalStats,
    last_reject: Option<SignalReject>,
}

/// The production reader: validation over the shared-memory ring.
pub type RingSignalReader = SignalReader<Consumer>;

impl<S: SignalSource> SignalReader<S> {
    pub fn new(source: S) -> Self {
        Self::with_config(source, ReaderConfig::default())
    }

    pub fn with_config(source: S, cfg: ReaderConfig) -> Self {
        Self {
            source,
            cfg,
            last_seq: None,
            stats: SignalStats::default(),
            last_reject: None,
        }
    }

    pub fn config(&self) -> &ReaderConfig {
        &self.cfg
    }

    /// The source itself, for a caller that has to service it out of band — the
    /// production source is a memory-mapped file that may not exist yet, and
    /// re-opening it is the owner's business, not the validator's.
    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    pub fn stats(&self) -> &SignalStats {
        &self.stats
    }

    /// The most recent refusal, so a caller can log *why* without a side channel.
    pub fn last_reject(&self) -> Option<SignalReject> {
        self.last_reject
    }

    /// Highest `seq` admitted so far.
    pub fn last_seq(&self) -> Option<u64> {
        self.last_seq
    }

    /// Take up to `max` records off the source at event time `now`, invoking `f`
    /// for each one that passes validation.
    ///
    /// `now` is the core's event-time clock, not `SystemTime::now()`: replaying a
    /// captured session must make the *same* staleness decisions it made live, and
    /// a wall clock would make every replayed signal infinitely stale.
    ///
    /// `max` bounds the work one loop iteration does, so a producer burst cannot
    /// starve market-data processing in the same thread.
    pub fn drain<F: FnMut(Signal)>(&mut self, now: Nanos, max: usize, mut f: F) -> DrainReport {
        let mut report = DrainReport::default();
        while report.read < max {
            let Some(sig) = self.source.next_signal() else {
                break;
            };
            report.read += 1;
            match self.admit(&sig, now) {
                Ok(()) => {
                    report.accepted += 1;
                    f(sig);
                }
                Err(_) => report.rejected += 1,
            }
        }
        report
    }

    /// Validate one record at event time `now`, updating the counters either way.
    ///
    /// Public so a caller (or a test) can drive validation over a record it already
    /// has. The check order is deliberate: layout first, then meaning, then
    /// ordering, then freshness — each check is only trustworthy once the ones
    /// before it have passed.
    pub fn admit(&mut self, sig: &Signal, now: Nanos) -> Result<(), SignalReject> {
        if sig.schema_version != SCHEMA_VERSION {
            self.stats.schema_version += 1;
            return Err(self.reject(SignalReject::SchemaVersion {
                found: sig.schema_version,
                expected: SCHEMA_VERSION,
            }));
        }
        if sig.kind != KIND_TARGET_POSITION {
            self.stats.unknown_kind += 1;
            return Err(self.reject(SignalReject::UnknownKind { kind: sig.kind }));
        }
        // The record's only remaining unwritten bytes. `max_order_age_ms` was carved out
        // of the old 15-byte `reserved` at offset 52 and `ts_cause` spent the last eight
        // at 56, so `pad0` is now the whole of it — and padding a producer may write into
        // is padding that hides an unversioned extension exactly as well as a named
        // `reserved` block would. The check is kept rather than deleted with the field it
        // used to guard: three bytes at offset 49 are still three bytes nothing has ever
        // been told to fill, and the day something starts filling them is the day this
        // refusal is the only evidence.
        if sig.pad0 != [0u8; 3] {
            self.stats.reserved_not_zero += 1;
            return Err(self.reject(SignalReject::ReservedNotZero));
        }

        if let Some(last) = self.last_seq {
            if sig.seq <= last {
                self.stats.stale_seq += 1;
                return Err(self.reject(SignalReject::StaleSeq { seq: sig.seq, last }));
            }
            let skipped = sig.seq - last - 1;
            if skipped > 0 {
                // Counted, *not* rejected. Missing an older target does not make the
                // newest one wrong — a target-position signal is self-contained, and
                // the whole reason for that shape (ADR-0006) is that it survives loss.
                // Refusing the newest record because an older one never arrived would
                // convert one dropped signal into an indefinite stall.
                self.stats.gaps += 1;
                self.stats.missing += skipped;
            }
        }

        let age_ns = now.saturating_sub(sig.ts_event);
        if age_ns < -CLOCK_SKEW_TOLERANCE_NS {
            self.stats.ahead_of_clock += 1;
        }
        let ttl_ns = self.effective_ttl_ns(sig.ttl_ms);
        if age_ns > ttl_ns {
            self.stats.expired += 1;
            return Err(self.reject(SignalReject::Expired {
                age_ms: age_ns / NS_PER_MS,
                ttl_ms: ttl_ns / NS_PER_MS,
            }));
        }

        // Only a record that will actually be acted on advances the sequence
        // baseline; a rejected record must not become the `last` a later valid one
        // is measured against.
        self.last_seq = Some(sig.seq);
        self.stats.accepted += 1;
        Ok(())
    }

    /// Re-check a record this reader already admitted, for a caller that had to hold
    /// it back rather than act on it.
    ///
    /// Freshness is the only one of the five checks whose answer can change after
    /// admission — layout, kind, reserved bytes and `seq` are properties of the record,
    /// and time is not. A caller that cannot act on an accepted record immediately (the
    /// runtime's per-symbol in-flight gate is the one that exists) therefore has
    /// exactly one thing to re-ask, and it must ask it *here*: a second staleness rule
    /// somewhere else would be a second answer to "how old is too old", and a record
    /// held outside the reader with no rule at all is the "second queue that ages
    /// signals invisibly" ADR-0020 §3 refused to build.
    ///
    /// A refusal is counted in [`SignalStats::expired`] — the same counter the first
    /// check would have used — so a signal that expired while it waited is one number
    /// with one meaning, and not a hole between two.
    pub fn still_fresh(&mut self, sig: &Signal, now: Nanos) -> bool {
        let age_ns = now.saturating_sub(sig.ts_event);
        if age_ns > self.effective_ttl_ns(sig.ttl_ms) {
            self.stats.expired += 1;
            self.reject(SignalReject::Expired {
                age_ms: age_ns / NS_PER_MS,
                ttl_ms: self.effective_ttl_ns(sig.ttl_ms) / NS_PER_MS,
            });
            return false;
        }
        true
    }

    /// The window this record is actually allowed, in nanoseconds: its own TTL,
    /// floored by nothing and capped by the operator's ceiling, with `0` meaning
    /// "the ceiling".
    ///
    /// Zero is the one value with a real choice behind it, and Python now spells it
    /// `TTL_OPERATOR_CEILING` rather than refusing to emit it (ADR-0020 §4). The
    /// deciding argument is that zero is also what a field nobody wrote contains, so
    /// whichever reading is chosen has to be the safe answer for a producer that
    /// never thought about staleness. "Never expires" gives that producer no
    /// protection at all; "already expired" turns an unset field into a strategy that
    /// silently stops trading, which looks exactly like a quiet market. Deferring to
    /// the operator's ceiling is the only reading that is neither.
    fn effective_ttl_ns(&self, ttl_ms: u32) -> i64 {
        let ceiling = i64::from(self.cfg.max_age_ms) * NS_PER_MS;
        if ttl_ms == 0 {
            return ceiling;
        }
        (i64::from(ttl_ms) * NS_PER_MS).min(ceiling)
    }

    fn reject(&mut self, r: SignalReject) -> SignalReject {
        self.last_reject = Some(r);
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::{FLAG_CLOSE, SIGNAL_SIZE};

    const MS: i64 = NS_PER_MS;

    fn sig(seq: u64, ts_ms: i64, ttl_ms: u32) -> Signal {
        Signal::target_position(seq, ts_ms * MS, 7, 100_000_000, 0, 0, ttl_ms, 1, 0)
    }

    fn reader() -> SignalReader<ReplaySource> {
        SignalReader::new(ReplaySource::default())
    }

    #[test]
    fn a_stale_signal_is_dropped_counted_and_never_acted_on() {
        // The failure mode: a signal that sat in the ring through a GC pause is a
        // decision about a market that no longer exists. Trading it late is worse
        // than not trading, because it is systematically late in the direction the
        // market already moved.
        let mut r = reader();
        let s = sig(1, 1_000, 500);
        let err = r.admit(&s, 1_600 * MS).unwrap_err();
        assert!(matches!(err, SignalReject::Expired { .. }), "got {err:?}");
        assert_eq!(r.stats().expired, 1);
        assert_eq!(r.stats().accepted, 0);
        assert_eq!(r.last_reject(), Some(err), "and the reason is surfaced");

        // 1 ms inside the window is still good — the boundary is not off by one.
        let mut r = reader();
        assert!(r.admit(&sig(1, 1_000, 500), 1_499 * MS).is_ok());
    }

    #[test]
    fn a_record_held_back_by_its_caller_ages_under_this_readers_rule_and_no_other() {
        // A caller that cannot act on an admitted record immediately — the per-symbol
        // in-flight gate is the one that exists — is holding a decision about a market
        // that keeps moving. If it re-checks staleness with its own arithmetic there
        // are two answers to "how old is too old"; if it re-checks nothing, the record
        // ages where no counter can see it, which is exactly the second queue ADR-0020
        // §3 refused to build.
        let mut r =
            SignalReader::with_config(ReplaySource::default(), ReaderConfig { max_age_ms: 2_000 });
        let s = sig(1, 1_000, 500);
        assert!(r.admit(&s, 1_000 * MS).is_ok());

        assert!(r.still_fresh(&s, 1_400 * MS), "inside its own window");
        assert_eq!(r.stats().expired, 0);

        assert!(!r.still_fresh(&s, 1_600 * MS));
        assert_eq!(
            r.stats().expired,
            1,
            "counted where the first check would have counted it"
        );
        assert!(matches!(
            r.last_reject(),
            Some(SignalReject::Expired { .. })
        ));

        // The operator's ceiling binds a held record exactly as it binds an arriving
        // one — a caller cannot buy a longer window by waiting.
        let mut r =
            SignalReader::with_config(ReplaySource::default(), ReaderConfig { max_age_ms: 1_000 });
        let patient = sig(1, 0, 60_000);
        assert!(r.admit(&patient, 0).is_ok());
        assert!(!r.still_fresh(&patient, 1_500 * MS));
    }

    #[test]
    fn a_rejected_signal_does_not_advance_the_sequence_baseline() {
        // Otherwise one expired record silently swallows the next valid one, which
        // shares its seq neighbourhood, and the drop is invisible.
        let mut r = reader();
        assert!(r.admit(&sig(5, 0, 100), 10_000 * MS).is_err());
        assert_eq!(r.last_seq(), None);
        assert!(r.admit(&sig(5, 10_000, 100), 10_000 * MS).is_ok());
        assert_eq!(r.last_seq(), Some(5));
    }

    #[test]
    fn a_signal_with_no_ttl_still_expires_at_the_operators_ceiling() {
        // `ttl_ms == 0` is the Python default. Treating "unset" as "never expires"
        // would mean the strategy that never thought about staleness is the one with
        // no protection at all.
        let mut r =
            SignalReader::with_config(ReplaySource::default(), ReaderConfig { max_age_ms: 2_000 });
        assert!(r.admit(&sig(1, 0, 0), 1_999 * MS).is_ok());
        let mut r =
            SignalReader::with_config(ReplaySource::default(), ReaderConfig { max_age_ms: 2_000 });
        assert!(matches!(
            r.admit(&sig(1, 0, 0), 2_001 * MS),
            Err(SignalReject::Expired { .. })
        ));
    }

    #[test]
    fn a_strategy_cannot_raise_its_own_staleness_ceiling() {
        let mut r =
            SignalReader::with_config(ReplaySource::default(), ReaderConfig { max_age_ms: 1_000 });
        // The record asks for a minute of validity; the operator allows one second.
        let err = r.admit(&sig(1, 0, 60_000), 1_500 * MS).unwrap_err();
        assert_eq!(
            err,
            SignalReject::Expired {
                age_ms: 1_500,
                ttl_ms: 1_000
            }
        );
    }

    #[test]
    fn a_sequence_gap_is_detected_without_dropping_the_newest_target() {
        // A gap means the producer lost records. The newest target is still the
        // truth; refusing it would turn one dropped signal into a stalled strategy.
        let mut r = reader();
        assert!(r.admit(&sig(10, 0, 0), 0).is_ok());
        assert!(
            r.admit(&sig(14, 0, 0), 0).is_ok(),
            "accepted despite the gap"
        );
        assert_eq!(r.stats().gaps, 1);
        assert_eq!(r.stats().missing, 3, "seq 11, 12 and 13 never arrived");
        assert_eq!(r.stats().accepted, 2);

        // Contiguous records do not register as a gap.
        assert!(r.admit(&sig(15, 0, 0), 0).is_ok());
        assert_eq!(r.stats().gaps, 1);
    }

    #[test]
    fn joining_a_running_producer_mid_stream_is_not_a_gap() {
        let mut r = reader();
        assert!(r.admit(&sig(9_000, 0, 0), 0).is_ok());
        assert_eq!(r.stats().gaps, 0, "the first record sets the baseline");
        assert_eq!(r.stats().missing, 0);
    }

    #[test]
    fn a_rewound_sequence_is_rejected_as_a_replay() {
        // A restarted producer begins at a low seq again. Acting on it would walk
        // the target backwards through decisions we already superseded.
        let mut r = reader();
        assert!(r.admit(&sig(10, 0, 0), 0).is_ok());
        assert_eq!(
            r.admit(&sig(4, 0, 0), 0).unwrap_err(),
            SignalReject::StaleSeq { seq: 4, last: 10 }
        );
        // An exact duplicate is the same refusal.
        assert_eq!(
            r.admit(&sig(10, 0, 0), 0).unwrap_err(),
            SignalReject::StaleSeq { seq: 10, last: 10 }
        );
        assert_eq!(r.stats().stale_seq, 2);
    }

    #[test]
    fn an_unknown_schema_version_is_rejected_before_any_field_is_trusted() {
        let mut r = reader();
        let mut s = sig(1, 0, 0);
        s.schema_version = SCHEMA_VERSION + 1;
        // Also corrupt the seq: if the version check did not come first, this would
        // have poisoned the baseline on the way past.
        s.seq = u64::MAX;
        assert!(matches!(
            r.admit(&s, 0),
            Err(SignalReject::SchemaVersion { .. })
        ));
        assert_eq!(r.last_seq(), None);
        assert_eq!(r.stats().schema_version, 1);
    }

    #[test]
    fn an_unknown_kind_is_rejected_rather_than_read_as_a_target_position() {
        // kind = 1 is reserved for the explicit order-intent variant. Its fields
        // occupy the same bytes with different meanings, so misreading one would
        // treat an order size as a whole target position.
        let mut r = reader();
        let mut s = sig(1, 0, 0);
        s.kind = 1;
        assert_eq!(
            r.admit(&s, 0).unwrap_err(),
            SignalReject::UnknownKind { kind: 1 }
        );
        assert_eq!(r.stats().unknown_kind, 1);
    }

    #[test]
    fn the_bytes_that_used_to_be_reserved_now_carry_a_cause_and_are_admitted() {
        // This test used to poke the last byte of an 8-byte `reserved` tail and require a
        // refusal. `ts_cause` spent that tail at schema version 3, so those bytes are now
        // a field with a meaning and a record that fills them is a record from a *newer*
        // producer, not a corrupt one.
        //
        // What protects an old reader from it is the version bump and nothing else — the
        // refusal this test used to assert was never the mechanism, it was a side effect of
        // the bytes being unnamed. The two assertions below are the mechanism, in both
        // directions.
        let mut r = reader();
        let mut s = sig(1, 0, 0);
        s.ts_cause = -60_000_000_000; // a bar that closed a minute before ts_event
        assert!(r.admit(&s, 0).is_ok(), "a stated cause is not an extension");
        assert_eq!(r.stats().reserved_not_zero, 0);

        // …and a producer built against the *old* layout is refused wholesale, which is
        // what makes the bump non-optional: its zeroed tail would otherwise decode as
        // "no cause stated" and read as healthy forever.
        let mut old = sig(2, 0, 0);
        old.schema_version = SCHEMA_VERSION - 1;
        assert_eq!(
            r.admit(&old, 0).unwrap_err(),
            SignalReject::SchemaVersion {
                found: SCHEMA_VERSION - 1,
                expected: SCHEMA_VERSION
            }
        );
    }

    #[test]
    fn a_nonzero_alignment_pad_is_rejected_the_same_way_the_reserved_tail_is() {
        // `pad0` exists only so `max_order_age_ms` lands naturally aligned, which makes it
        // the easiest place for a producer to write a field nobody versioned — and the
        // easiest one for a reader to forget, since it does not have "reserved" in the name.
        let mut r = reader();
        let mut s = sig(1, 0, 0);
        s.pad0[0] = 1;
        assert_eq!(r.admit(&s, 0).unwrap_err(), SignalReject::ReservedNotZero);
        assert_eq!(r.stats().reserved_not_zero, 1);
    }

    #[test]
    fn a_producer_clock_running_ahead_is_counted_not_silently_trusted() {
        // TTL enforcement is only as good as the two clocks agreeing. A signal
        // stamped in the future can never expire, so the skew itself is the alert.
        let mut r = reader();
        assert!(r.admit(&sig(1, 5_000, 100), 1_000 * MS).is_ok());
        assert_eq!(r.stats().ahead_of_clock, 1);
        // Sub-millisecond disagreement is normal and not counted.
        let mut r = reader();
        assert!(r.admit(&sig(1, 1_000, 100), 1_000 * MS - 1).is_ok());
        assert_eq!(r.stats().ahead_of_clock, 0);
    }

    #[test]
    fn draining_stops_at_the_batch_bound_and_keeps_the_rest() {
        let mut src = ReplaySource::default();
        for seq in 1..=5 {
            src.push(sig(seq, 0, 0));
        }
        let mut r = SignalReader::new(src);
        let mut got = Vec::new();
        let report = r.drain(0, 2, |s| got.push(s.seq));
        assert_eq!(report.read, 2);
        assert_eq!(report.accepted, 2);
        assert_eq!(got, vec![1, 2]);
        let report = r.drain(0, 10, |s| got.push(s.seq));
        assert_eq!(report.read, 3, "the source is drained, not the bound");
        assert_eq!(got, vec![1, 2, 3, 4, 5]);
        assert_eq!(r.drain(0, 10, |_| unreachable!()).read, 0, "idle source");
    }

    #[test]
    fn a_drain_reports_rejections_without_dropping_the_valid_records_around_them() {
        let mut src = ReplaySource::default();
        src.push(sig(1, 1_000, 100));
        src.push(sig(2, 0, 100)); // expired
        src.push(sig(3, 1_000, 100));
        let mut r = SignalReader::new(src);
        let mut got = Vec::new();
        let report = r.drain(1_000 * MS, 10, |s| got.push(s.seq));
        assert_eq!(
            report,
            DrainReport {
                read: 3,
                accepted: 2,
                rejected: 1
            }
        );
        assert_eq!(got, vec![1, 3]);
    }

    #[test]
    fn the_same_validation_applies_to_records_off_a_real_ring() {
        // The `Consumer` impl is the production path; exercise it end to end over a
        // real mmap so the trait plumbing cannot rot. Offline: a temp file, no network.
        use axon_ipc::Producer;

        let mut path = std::env::temp_dir();
        path.push(format!("axon-strategy-reader-{}.ring", std::process::id()));
        let producer = Producer::create(&path, 8).unwrap();
        let consumer = Consumer::open(&path).unwrap();

        let mut good = sig(1, 1_000, 500);
        good.flags = FLAG_CLOSE;
        assert!(producer.try_push(&good));
        assert!(producer.try_push(&sig(2, 0, 500))); // expired at `now`
        let mut bad_kind = sig(3, 1_000, 500);
        bad_kind.kind = 9;
        assert!(producer.try_push(&bad_kind));

        let mut r: RingSignalReader = SignalReader::new(consumer);
        let mut got = Vec::new();
        let report = r.drain(1_000 * MS, 16, |s| got.push(s));
        assert_eq!(report.read, 3);
        assert_eq!(report.accepted, 1);
        assert_eq!(report.rejected, 2);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].seq, 1);
        assert!(got[0].is_close());
        assert_eq!(r.stats().expired, 1);
        assert_eq!(r.stats().unknown_kind, 1);
        assert_eq!(core::mem::size_of::<Signal>(), SIGNAL_SIZE);

        drop(producer);
        drop(r);
        let _ = std::fs::remove_file(&path);
    }
}
