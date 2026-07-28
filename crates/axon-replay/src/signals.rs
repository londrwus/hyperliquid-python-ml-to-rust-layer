//! The **signal log**: what the strategy sent, kept beside the event log so a replay
//! can re-*decide* and not merely re-*observe*.
//!
//! An event log alone replays market data and the venue's answers to orders the
//! captured session sent. Drive the strategy adapter with it and the adapter has
//! nothing to read: [`SignalReader`](axon_strategy::SignalReader) drains a ring the
//! Python producer writes, and a ring is not an event. So the golden run needs a
//! second recording — the records that crossed that ring — or the reconciliation half
//! of the chain would be replayed against a strategy that never spoke.
//!
//! Three decisions, each preventing a way the pair could silently disagree:
//!
//! - **The log owns its wire form.** [`LoggedSignal`] mirrors
//!   [`Signal`](axon_contracts::Signal) rather than deriving `Serialize` on it —
//!   `Signal` is a `Pod` whose layout is pinned to `contracts/schema.toml`, and
//!   binding a persisted format to it would make every layout change a silent format
//!   change. The conversion back is an exhaustive struct literal, so a field added to
//!   `Signal` is a compile error here until it has been given a wire form.
//! - **Each record carries `release_ts`, which the `Signal` does not.** A record's
//!   `ts_event` is when the strategy *decided*; `release_ts` is when the record became
//!   visible to the reader. The ring carries only the first, and the difference is
//!   exactly what makes a signal stale — a decision written into the ring during a GC
//!   pause is a firm opinion about a market that has already gone. Without it a replay
//!   could never reproduce an expiry, and expiry is the reader's most important
//!   refusal.
//! - **Release times may not go backwards.** An SPSC ring hands records out in the
//!   order they were written, so a log claiming otherwise describes a session that
//!   never happened, and replaying it would certify an interleaving no producer could
//!   have produced.
//!
//! The log deliberately stops there. It is not a recording of the *ring* — no
//! capacity, no drop counts, no wrap. Those belong to `axon-ipc` and are its own
//! tests' business; what the chain under replay consumes is a sequence of records, and
//! that is what this file is.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use axon_contracts::Signal;
use axon_core::Nanos;
use serde::{Deserialize, Serialize};

use crate::error::ReplayError;
use crate::log::LogHeader;

/// The `schema` string every Axon signal log carries.
pub const SIGNAL_SCHEMA: &str = "axon.signallog";

/// The signal-log format version. **Bump this whenever
/// [`Signal`](axon_contracts::Signal) changes shape**, for the same reason the event
/// log has one: a build that reads a log written under a different layout would
/// re-plan against fields that no longer mean what they meant, and the golden
/// comparison would pass against a reference it no longer understands.
///
/// - `1` — the original 64-byte record, 15 reserved bytes.
/// - `2` — `max_order_age_ms` and its `pad0` carved out of `reserved`, which drops to
///   8 bytes. A v1 log's records would deserialize into the v2 shape with the new
///   field defaulted to zero, and zero on that field means "defer to the operator's
///   ceiling" rather than "absent" — a replayed order given a lifetime nobody set.
/// - `3` — `ts_cause` spends the last 8 reserved bytes, so the record is now fully
///   named. A v2 log has no `ts_cause` at all, and because [`LoggedSignal`] does **not**
///   default it, such a log fails to deserialize loudly rather than replaying every
///   record as "no cause stated" — which would report the `cause` latency stage as empty
///   on a replay of the very session where it was the largest number on the line.
pub const SIGNAL_SCHEMA_VERSION: u32 = 3;

/// The serializable mirror of [`Signal`].
///
/// Every field is written out, `reserved` included. It looks like noise in a diff and
/// it is not: the reader *refuses* a record whose reserved bytes are non-zero
/// (a producer built against a newer contract), and a format that dropped them could
/// never carry that case into a replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggedSignal {
    pub seq: u64,
    pub ts_event: Nanos,
    pub symbol_id: u32,
    pub target_qty: i64,
    pub price_band: i64,
    pub ttl_ms: u32,
    /// How long an order this signal places may keep its place, in ms.
    ///
    /// Written out beside `ttl_ms` and never folded into it: `ttl_ms` is a signal
    /// *admission* window the reader consumes before the planner sees the record, and a
    /// replay that lost this field would age every resting order against the wrong
    /// number — or, since zero means "defer to the operator's ceiling", against a
    /// ceiling nobody set.
    pub max_order_age_ms: u32,
    pub model_version: u32,
    pub flags: u16,
    pub urgency: u8,
    pub kind: u8,
    pub schema_version: u8,
    /// Explicit padding. Carried because the reader *refuses* a record whose padding is
    /// non-zero, and a format that dropped it could never carry that refusal into a
    /// replay — a corrupt record would replay as a clean one.
    pub pad0: [u8; 3],
    /// The event time of the observation the decision answered — a bar's own close.
    ///
    /// Carried for the same reason `max_order_age_ms` is: zero is a *meaningful* value on
    /// this field ("no cause stated"), so a log that dropped it would replay every record
    /// as having no cause and the `cause` latency stage would read empty on a replay of a
    /// session where it was the largest number on the line.
    pub ts_cause: i64,
}

impl From<&Signal> for LoggedSignal {
    fn from(s: &Signal) -> Self {
        Self {
            seq: s.seq,
            ts_event: s.ts_event,
            symbol_id: s.symbol_id,
            target_qty: s.target_qty,
            price_band: s.price_band,
            ttl_ms: s.ttl_ms,
            max_order_age_ms: s.max_order_age_ms,
            model_version: s.model_version,
            flags: s.flags,
            urgency: s.urgency,
            kind: s.kind,
            schema_version: s.schema_version,
            pad0: s.pad0,
            ts_cause: s.ts_cause,
        }
    }
}

impl From<LoggedSignal> for Signal {
    /// An exhaustive struct literal on purpose: a field added to [`Signal`] fails to
    /// compile here until the log has been taught to carry it. A `..Default::default()`
    /// tail would instead let a whole field quietly replay as zero, and zero is a
    /// meaningful value for every numeric field on this record.
    fn from(s: LoggedSignal) -> Self {
        Signal {
            seq: s.seq,
            ts_event: s.ts_event,
            target_qty: s.target_qty,
            price_band: s.price_band,
            symbol_id: s.symbol_id,
            ttl_ms: s.ttl_ms,
            max_order_age_ms: s.max_order_age_ms,
            model_version: s.model_version,
            flags: s.flags,
            schema_version: s.schema_version,
            urgency: s.urgency,
            kind: s.kind,
            pad0: s.pad0,
            ts_cause: s.ts_cause,
        }
    }
}

/// One recorded signal, plus the moment the reader could first have seen it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalRecord {
    /// Event time at which this record became visible to the reader.
    ///
    /// Normally equal to the signal's own `ts_event`. Where it is *later*, the record
    /// sat in the ring — a producer pause, a slow consumer — and the reader will judge
    /// its age from here. That gap is the only way a replay can reproduce an expiry,
    /// and an expired signal that gets acted on anyway is systematically late in the
    /// direction the market already moved.
    pub release_ts: Nanos,
    pub signal: LoggedSignal,
}

impl SignalRecord {
    /// A record visible the instant it was decided — the common case.
    pub fn now(signal: &Signal) -> Self {
        Self {
            release_ts: signal.ts_event,
            signal: LoggedSignal::from(signal),
        }
    }

    /// A record that reached the reader late.
    pub fn released_at(release_ts: Nanos, signal: &Signal) -> Self {
        Self {
            release_ts,
            signal: LoggedSignal::from(signal),
        }
    }
}

/// A parsed signal log, held in memory.
#[derive(Debug, Clone)]
pub struct SignalLog {
    header: LogHeader,
    records: Vec<SignalRecord>,
}

impl SignalLog {
    /// Parse a signal log from any reader.
    pub fn read(src: impl BufRead) -> Result<Self, ReplayError> {
        let mut lines = src.lines().enumerate();

        let first = lines.next().ok_or(ReplayError::MissingHeader)?.1?;
        let header: LogHeader =
            serde_json::from_str(&first).map_err(|source| ReplayError::Json { line: 1, source })?;
        header.check_as(SIGNAL_SCHEMA, SIGNAL_SCHEMA_VERSION)?;

        let mut records: Vec<SignalRecord> = Vec::new();
        for (idx, line) in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let lineno = idx + 1;
            let rec: SignalRecord =
                serde_json::from_str(&line).map_err(|source| ReplayError::Json {
                    line: lineno,
                    source,
                })?;
            if let Some(prev) = records.last() {
                if rec.release_ts < prev.release_ts {
                    return Err(ReplayError::SignalsOutOfOrder {
                        line: lineno,
                        released: rec.release_ts,
                        previous: prev.release_ts,
                    });
                }
            }
            records.push(rec);
        }
        Ok(Self { header, records })
    }

    /// Open and parse a signal log file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        Self::read(BufReader::new(File::open(path)?))
    }

    /// Write a signal log — the fixture generator's other half.
    ///
    /// Records go out in the order given and are **not** sorted here: the order a
    /// producer wrote them in is the thing being recorded, and sorting it would erase
    /// the one property [`read`](Self::read) then refuses to accept.
    pub fn write(
        mut out: impl Write,
        header: &LogHeader,
        records: &[SignalRecord],
    ) -> Result<(), ReplayError> {
        writeln!(out, "{}", serde_json::to_string(header).map_err(json)?)?;
        for rec in records {
            writeln!(out, "{}", serde_json::to_string(rec).map_err(json)?)?;
        }
        out.flush()?;
        Ok(())
    }

    /// An empty log, for a replay with no strategy attached. Not a degraded mode: a
    /// market-data-only capture legitimately has no signals, and the chain still runs
    /// — the adapter simply has nothing to admit.
    pub fn empty() -> Self {
        Self {
            header: LogHeader::for_schema(SIGNAL_SCHEMA, SIGNAL_SCHEMA_VERSION, "none", 0),
            records: Vec::new(),
        }
    }

    pub fn header(&self) -> &LogHeader {
        &self.header
    }

    pub fn records(&self) -> &[SignalRecord] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// A `serde_json` failure while *writing* has no line number to report, so it is
/// carried as I/O — which is what it always is in practice (a full disk).
fn json(e: serde_json::Error) -> ReplayError {
    ReplayError::Io(std::io::Error::other(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::FLAG_CLOSE;

    fn sig(seq: u64, ts: Nanos) -> Signal {
        Signal::target_position(seq, ts, 1, 100_000_000, 2, 0, 500, 7, FLAG_CLOSE)
    }

    fn bytes(records: &[SignalRecord]) -> String {
        let header = LogHeader::for_schema(SIGNAL_SCHEMA, SIGNAL_SCHEMA_VERSION, "unit-test", 0);
        let mut out = Vec::new();
        SignalLog::write(&mut out, &header, records).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn a_record_pins_its_wire_shape_so_a_contract_change_cannot_pass_unnoticed() {
        // `SIGNAL_SCHEMA_VERSION` is a human promise, and this is what forces the
        // human to keep it. A field added to `Signal`, a renamed one, a changed scale
        // all break here — without this the record could drift while old logs kept
        // deserializing into the new shape, and a golden run would re-plan a record
        // that no longer meant what it said.
        let rec = SignalRecord::now(&sig(1, 1_000));
        assert_eq!(
            serde_json::to_string(&rec).unwrap(),
            r#"{"release_ts":1000,"signal":{"seq":1,"ts_event":1000,"symbol_id":1,"target_qty":100000000,"price_band":0,"ttl_ms":500,"max_order_age_ms":0,"model_version":7,"flags":2,"urgency":2,"kind":0,"schema_version":3,"pad0":[0,0,0],"ts_cause":0}}"#
        );
    }

    #[test]
    fn a_signal_round_trips_through_the_log_unchanged() {
        // Every field, not a spot check: a dropped `flags` bit turns a flatten into an
        // opening order, and a dropped `urgency` turns an IOC exit into a resting quote.
        let original = sig(9, 4_242);
        let log = SignalLog::read(bytes(&[SignalRecord::now(&original)]).as_bytes()).unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(Signal::from(log.records()[0].signal), original);
    }

    #[test]
    fn a_log_written_by_an_incompatible_build_is_refused_not_reinterpreted() {
        // Written against the constant, never against a literal. A hard-coded "an
        // impossible future version" stops being impossible the day somebody bumps the
        // constant to it — and then this test substitutes nothing, reads a log this build
        // accepts, and passes while guarding nothing at all. That is the exact shape of
        // failure the version exists to prevent, so the guard must not have it.
        let text = bytes(&[SignalRecord::now(&sig(1, 10))]);
        let next = SIGNAL_SCHEMA_VERSION + 1;
        let bumped = text.replace(
            &format!("\"schema_version\":{SIGNAL_SCHEMA_VERSION},\"writer\""),
            &format!("\"schema_version\":{next},\"writer\""),
        );
        assert_ne!(
            bumped, text,
            "the header no longer looks the way this test edits it, so nothing was bumped"
        );
        assert!(
            matches!(
                SignalLog::read(bumped.as_bytes()).unwrap_err(),
                ReplayError::IncompatibleVersion { found, .. } if found == next
            ),
            "a version bump must stop the read"
        );

        // …and an event log handed to the signal reader is the wrong *file*, which has
        // to say so rather than fail as malformed JSON three lines in.
        let foreign = text.replace(SIGNAL_SCHEMA, "axon.eventlog");
        assert!(matches!(
            SignalLog::read(foreign.as_bytes()).unwrap_err(),
            ReplayError::ForeignLog { .. }
        ));
    }

    #[test]
    fn a_log_whose_records_were_released_out_of_order_is_refused() {
        // A ring is FIFO. A log claiming record 2 became visible before record 1
        // describes a producer that cannot exist, and replaying it would let the
        // harness certify an interleaving no live session could have seen.
        let text = bytes(&[
            SignalRecord::released_at(500, &sig(1, 100)),
            SignalRecord::released_at(400, &sig(2, 200)),
        ]);
        assert!(
            matches!(
                SignalLog::read(text.as_bytes()).unwrap_err(),
                ReplayError::SignalsOutOfOrder {
                    released: 400,
                    previous: 500,
                    ..
                }
            ),
            "the release order is what the log is *for*"
        );
    }

    #[test]
    fn a_record_may_be_released_later_than_it_was_decided() {
        // The gap is the point: it is the only way a replay reproduces a signal that
        // went stale in the ring, and staleness is the reader's most important refusal.
        let text = bytes(&[SignalRecord::released_at(9_000, &sig(1, 1_000))]);
        let log = SignalLog::read(text.as_bytes()).unwrap();
        assert_eq!(log.records()[0].release_ts, 9_000);
        assert_eq!(log.records()[0].signal.ts_event, 1_000);
    }

    #[test]
    fn a_headerless_signal_log_fails_loudly() {
        assert!(matches!(
            SignalLog::read(&b""[..]).unwrap_err(),
            ReplayError::MissingHeader
        ));
    }

    #[test]
    fn a_replay_with_no_strategy_attached_is_an_empty_log_not_an_error() {
        let log = SignalLog::empty();
        assert!(log.is_empty());
        assert_eq!(log.header().schema, SIGNAL_SCHEMA);
    }
}
