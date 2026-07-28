//! Failures a capture or a replay can hit.
//!
//! Every variant here names a way a log could be *misread* rather than merely be
//! unreadable, because a harness that silently degrades — a short log, a
//! reinterpreted schema, a dropped write — manufactures confidence, which is worse
//! than having no harness at all.

use axon_core::Nanos;

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("event log I/O failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("event log line {line}: malformed JSON: {source}")]
    Json {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("event log is empty: not even a header line")]
    MissingHeader,

    #[error("not an Axon {expected:?} log: schema is {found:?}")]
    ForeignLog {
        found: String,
        expected: &'static str,
    },

    #[error(
        "{schema:?} schema version {found} cannot be replayed by this build; \
         replay it with the build that wrote it"
    )]
    IncompatibleVersion { schema: &'static str, found: u32 },

    #[error(
        "event log schema version {found} predates the instrument table. A log written \
         before it carries no grid, so replaying it would plan every rounded price \
         unconstrained - prices the session that recorded it could not have sent, \
         reported as strategy flips by the harness built to detect them. Re-capture, \
         or replay it with the build that wrote it (ADR-0027)"
    )]
    LogPredatesInstruments { found: u32 },

    #[error(
        "event log carries no instrument line: the first body line of a log is the grid \
         the recording session planned against, and a log without one claims to carry a \
         grid and does not"
    )]
    MissingInstruments,

    #[error(
        "event log line {line}: a second instrument line - two logs concatenated into \
         one file replay as a session that never happened, with a `seq` that restarts \
         mid-stream and a second half recorded against a different grid"
    )]
    RepeatedInstruments { line: usize },

    #[error(
        "event log: instrument {symbol} declares a grid this build cannot rebuild: \
         {source}"
    )]
    BadInstrument {
        symbol: u32,
        #[source]
        source: axon_providers::SpecError,
    },

    #[error(
        "event log: the record at byte {offset} could not be read on the replay pass, \
         though it parsed when the log was opened - the file changed underneath the \
         replay, so what was dispatched is a prefix of what was reported"
    )]
    LogChanged { offset: u64 },

    #[error(
        "signal log line {line}: released at {released} but the previous record was \
         released at {previous} — a ring hands records out in the order they were \
         written, so a log claiming otherwise describes a session that never happened"
    )]
    SignalsOutOfOrder {
        line: usize,
        released: Nanos,
        previous: Nanos,
    },

    #[error(
        "event log line {line}: recorded ts_event {recorded} disagrees with the payload's \
         own {derived} — the ordering key has changed since capture, so this log cannot be \
         interleaved the way it was captured"
    )]
    OrderingKeyChanged {
        line: usize,
        recorded: Nanos,
        derived: Nanos,
    },

    #[error("the event bus closed after {published} events; the core stopped consuming")]
    BusClosed { published: u64 },
}
