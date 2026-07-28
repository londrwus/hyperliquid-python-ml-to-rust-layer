//! The on-disk event log: **JSONL**, one header line, then one record per line.
//!
//! JSONL rather than a packed binary frame because a golden log's whole value is
//! in being readable when a replay diverges. When two runs disagree the first
//! question is always "which event differs", and `diff` answering that in one
//! command is worth more than the bytes a binary format would save. The cost is
//! real and named in ADR-0018: serialization happens off the core thread now
//! (ADR-0018 §9) but it is still a few microseconds per event.
//!
//! ## The shape of a log
//!
//! ```text
//! {"schema":"axon.eventlog","schema_version":2,…}   ← LogHeader
//! {"Instruments":{…}}                               ← LogLine::Instruments, always first
//! {"Event":{"seq":0,…}}                             ← LogLine::Event, one per event
//! ```
//!
//! Three properties of that shape exist purely to stop a log from being *silently*
//! misread, which is the failure mode that would poison every parity gate built on
//! top of it:
//!
//! - The [`LogHeader`] pins `schema` and `schema_version`, and the reader **refuses**
//!   anything else. Every event type here is `serde`-derived and therefore
//!   *structurally* decoded: a field added with a default, or a variant renamed,
//!   would let an old log deserialize into a new shape that means something
//!   different. The version is the only thing standing between that and a green
//!   replay of a log the build can no longer interpret.
//! - Each [`LogRecord`] carries `ts_event` **next to** the payload that already
//!   contains it, and the reader rejects the pair when they disagree. That is not
//!   redundancy for its own sake: [`Ticker::ts_event`](axon_core::Ticker::ts_event)
//!   is *derived* (venue time when there is one, receipt time otherwise), so a
//!   change to that rule would silently re-key events the log had already ordered,
//!   and the replay would reinterleave against a reference captured under the old
//!   rule.
//! - The **first body line is the instrument table** ([`crate::instruments`]), and a
//!   log without one is refused rather than replayed unrounded. That is the whole of
//!   `SCHEMA_VERSION` 2: see [`SCHEMA_VERSION`] for what a v1 log now does.
//!
//! ## A log is read one line at a time
//!
//! [`LogReader`] streams. ADR-0018 accepted "a log is loaded whole" as a consequence
//! and set the capture's size cap where a log stops being *loadable* rather than where
//! a disk stops being writable — which is why a multi-hour soak stopped its own
//! recording. Reading a line at a time removes the reason for that cap; what it does
//! **not** remove is the reason ADR-0018 gave, and [`crate::replay`] is where that is
//! paid: event-time ordering still cannot begin until the last record has been seen,
//! so it costs an *index* (a couple of dozen bytes a record) rather than the records
//! themselves.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use axon_core::{Event, ExecEvent, MarketEvent, Nanos};
use serde::{Deserialize, Serialize};

use crate::error::ReplayError;
use crate::instruments::InstrumentSet;

/// The `schema` string every Axon event log carries.
pub const SCHEMA: &str = "axon.eventlog";

/// The event-log format version. **Bump this whenever the core `market` or `exec`
/// vocabulary changes shape** — adding a field, renaming a variant, changing a
/// serde attribute. A build that reads a log written under a different version
/// refuses it rather than guessing, because the alternative is a parity gate that
/// passes against a reference it no longer understands.
///
/// - `1` — a header and one flat record per event.
/// - `2` — every body line is a tagged [`LogLine`], and the first one is the
///   [`InstrumentSet`] the recording session planned against (ADR-0027).
pub const SCHEMA_VERSION: u32 = 2;

/// The first version that carries an instrument table. Everything below it is refused
/// by name — see [`ReplayError::LogPredatesInstruments`].
pub(crate) const FIRST_VERSION_WITH_INSTRUMENTS: u32 = 2;

/// The first line of a log: what wrote it, when, and against which schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogHeader {
    /// Always [`SCHEMA`]. Distinguishes an Axon event log from any other JSONL a
    /// path might point at, before a single record is parsed.
    pub schema: String,
    pub schema_version: u32,
    /// The crate + version that produced the log, e.g. `"axon-replay 0.0.1"`.
    /// Informational: it is what a post-mortem needs to find the build, but it is
    /// deliberately *not* checked, because pinning it would make every version bump
    /// invalidate every stored log while `schema_version` already covers the case
    /// that actually matters.
    pub writer: String,
    /// Free-form provenance — venue, network, symbols, session id. Whatever a human
    /// reading the log a month later needs to know where it came from.
    pub source: String,
    /// Wall-clock capture time, in nanoseconds since the epoch.
    ///
    /// The one wall-clock stamp in the whole format, and it is metadata only: it is
    /// never an ordering key and no handler ever sees it. Ordering comes from each
    /// event's own `ts_event`.
    pub created_ns: Nanos,
}

impl LogHeader {
    /// A header for an event log being written now by this build.
    pub fn new(source: impl Into<String>, created_ns: Nanos) -> Self {
        Self::for_schema(SCHEMA, SCHEMA_VERSION, source, created_ns)
    }

    /// The same, for a sibling format that carries its own schema name and version —
    /// today the signal log ([`crate::signals`]).
    ///
    /// One header type rather than one per format, because the two properties that
    /// make a header worth having (a name that says what the file is, a version the
    /// reader refuses to guess past) are identical for both. What must *not* be shared
    /// is the constant: a signal log answering to `axon.eventlog` would be accepted by
    /// the event reader and then fail as malformed JSON, which reads like corruption
    /// rather than like the wrong file.
    pub fn for_schema(
        schema: &'static str,
        schema_version: u32,
        source: impl Into<String>,
        created_ns: Nanos,
    ) -> Self {
        Self {
            schema: schema.to_string(),
            schema_version,
            writer: concat!("axon-replay ", env!("CARGO_PKG_VERSION")).to_string(),
            source: source.into(),
            created_ns,
        }
    }

    /// Whether this build can read the log without reinterpreting it.
    pub(crate) fn check_as(
        &self,
        schema: &'static str,
        schema_version: u32,
    ) -> Result<(), ReplayError> {
        if self.schema != schema {
            return Err(ReplayError::ForeignLog {
                found: self.schema.clone(),
                expected: schema,
            });
        }
        if self.schema_version != schema_version {
            return Err(ReplayError::IncompatibleVersion {
                schema,
                found: self.schema_version,
            });
        }
        Ok(())
    }

    fn check(&self) -> Result<(), ReplayError> {
        // The one version whose refusal names its own cause. Every log captured before
        // ADR-0027 carries no grid, and a reader that shrugged and planned unconstrained
        // would keep producing exactly the divergence the version bump exists to end —
        // silently, over an artifact that looks current. See `SCHEMA_VERSION`.
        if self.schema == SCHEMA && self.schema_version < FIRST_VERSION_WITH_INSTRUMENTS {
            return Err(ReplayError::LogPredatesInstruments {
                found: self.schema_version,
            });
        }
        self.check_as(SCHEMA, SCHEMA_VERSION)
    }
}

/// The serializable mirror of [`Event`].
///
/// The log owns its own wire form rather than deriving `Serialize` on
/// [`Event`] itself, for two reasons that pull the same way. Persistence is versioned
/// on its own schedule — binding the format to an in-memory enum's derive would make
/// every refactor of the core a silent format change. And the `From<&Event>` match
/// below is *exhaustive*, so a new `Event` variant fails to compile here until it has
/// been given a wire form; an untagged blob or a catch-all arm would instead drop that
/// whole class of event out of every capture, and nothing would report it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoggedEvent {
    Market(MarketEvent),
    Exec(ExecEvent),
}

impl LoggedEvent {
    /// The payload's own ordering key, computed exactly as the core computes it.
    pub fn ts_event(&self) -> Nanos {
        match self {
            LoggedEvent::Market(m) => m.ts_event(),
            LoggedEvent::Exec(e) => e.ts_event(),
        }
    }

    /// What this record's `ts_event` is evidence of. See [`StampSource`].
    pub(crate) fn stamp_source(&self) -> StampSource {
        StampSource::of_logged(self)
    }
}

/// What a record's `ts_event` is actually *evidence of*.
///
/// Every record has an ordering key; they are not all worth the same as testimony about
/// when something happened, and a metric that treats them as equal measures the format
/// rather than the venue. A 1 h 44 m testnet soak made the difference impossible to
/// ignore: **15.94% of its records were "late", and 0.08% of the venue-stamped market
/// data was.**
///
/// Three kinds, and each names a real venue behaviour rather than a bookkeeping detail:
///
/// - [`Venue`](Self::Venue) — the venue timestamped the moment this record describes.
///   The only testimony worth ordering against itself, and the whole subject of
///   [`ReplayReport::reordered_by_the_feed`](crate::ReplayReport::reordered_by_the_feed).
/// - [`Derived`](Self::Derived) — the key came from somewhere other than "the venue says
///   this happened now". Two cases on this venue and both are the ordinary case, not an
///   edge: a [`Candle`](axon_core::Candle) is stamped `open_time + interval`, the moment
///   the bar *will* close, and the venue republishes the bar it is still filling; and
///   `activeAssetCtx` carries **no venue timestamp at all**, so a [`Ticker`] is stamped at
///   receipt (ADR-0011) and was **65% of the soak tape**. Both walk the event-time
///   high-water mark ahead of the market — a bar by up to a whole interval, a ticker by
///   one network hop — and everything captured behind them looks late having done nothing.
/// - [`Execution`](Self::Execution) — our own execution stream. Not a feed at all: the
///   venue **replays the whole `userFills` snapshot on every reconnect**, and the soak saw
///   execution times up to **2.94 hours** stale arrive that way. A snapshot being re-sent
///   is not a feed delivering out of order, and counting it as one would make every
///   reconnect look like a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StampSource {
    Venue,
    Derived {
        /// Where the described window begins, when the record describes one.
        ///
        /// This is what *bounds* the excuse. A bar can explain a record inside its own
        /// window and nothing else; a record older than the whole window is behind the
        /// mark for a reason the bar cannot account for. `None` — a receipt stamp — has no
        /// window and therefore no bound, which is stated as a cost rather than hidden.
        window_start: Option<Nanos>,
    },
    Execution,
}

impl StampSource {
    fn of_market(m: &MarketEvent) -> Self {
        // Exhaustive, unlike most diagnostics in this crate, and deliberately so: this
        // decides what a published measurement about a venue is measured *over*. A new
        // market event silently defaulting into or out of that subset would change what
        // the headline number means with nothing anywhere to report it.
        match m {
            MarketEvent::Bbo(_) | MarketEvent::Trade(_) | MarketEvent::Book(_) => {
                StampSource::Venue
            }
            MarketEvent::Candle(c) => StampSource::Derived {
                window_start: Some(c.open_time),
            },
            // The one variant that decides at runtime. `is_venue_timed` is the core's own
            // predicate, not a copy of it: a venue that starts sending a timestamp on this
            // frame moves into the measured subset without anything here changing.
            MarketEvent::Ticker(t) if t.is_venue_timed() => StampSource::Venue,
            MarketEvent::Ticker(_) => StampSource::Derived { window_start: None },
        }
    }

    pub fn of(ev: &Event) -> Self {
        match ev {
            Event::Market(m) => Self::of_market(m),
            Event::Exec(_) => StampSource::Execution,
        }
    }

    pub fn of_logged(ev: &LoggedEvent) -> Self {
        match ev {
            LoggedEvent::Market(m) => Self::of_market(m),
            LoggedEvent::Exec(_) => StampSource::Execution,
        }
    }
}

/// Watches capture order for the two different ways it disagrees with event-time order.
///
/// One inversion, two possible culprits, and which one is blamed decides whether an
/// operator reads *"the feed is reordering"* or *"a bar is stamped for a window that had
/// not closed yet"*. Those call for opposite responses, and a single number that mixes
/// them reports a stamping convention as a soak finding.
///
/// It lives here, once, because [`Capture`](crate::Capture) counts this while writing and
/// [`ReplaySource`](crate::ReplaySource) counts it while reading, and two copies of a
/// rule this subtle would agree on the day they were written and drift after.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OrderingWatch {
    pub events: u64,
    /// The **first** record's time, in capture order.
    ///
    /// Not the same question as [`oldest_ts`](Self::oldest_ts), and both have a caller:
    /// a writer reports where its recording started, a reader reports the window the log
    /// covers. On a log with a late arrival they differ, which is exactly when somebody
    /// would otherwise assume they could not.
    pub first_ts: Option<Nanos>,
    /// The **earliest** time in the log — the start of the window it covers.
    pub oldest_ts: Option<Nanos>,
    /// The event-time high-water mark — **not** the previous record's time. A single
    /// late event must not make every event after it look late too.
    pub high: Option<Nanos>,
    pub late_arrivals: u64,
    pub behind_derived_stamps: u64,
    /// Records the **venue itself timestamped** — the denominator of the only ordering
    /// claim about a venue this format can support.
    pub venue_stamped: u64,
    /// Of those, how many arrived behind another venue-stamped record.
    ///
    /// Measured against *each other* and against nothing else, which is what makes it
    /// exact rather than a residue: a receipt-stamped ticker or a replayed snapshot
    /// cannot enter this count in either role, so there is nothing here to subtract and
    /// nothing to misattribute.
    pub venue_stamped_out_of_order: u64,
    /// The high-water mark of the venue-stamped subset. Separate from
    /// [`high`](Self::high) on purpose — one mark for the log, one for the measurement.
    venue_high: Option<Nanos>,
    /// Whether the record holding [`high`](Self::high) carried a derived stamp, and the
    /// window it described if it described one.
    ///
    /// Held rather than recomputed because the mark outlives the record that set it: one
    /// forming bar shadows every record captured for the rest of its window, and a
    /// receipt-stamped ticker shadows whatever follows it.
    mark_derived: Option<Option<Nanos>>,
}

impl OrderingWatch {
    /// Fold in one record, in **capture** order.
    pub fn observe(&mut self, ts_event: Nanos, stamp: StampSource) {
        self.events += 1;
        if self.first_ts.is_none() {
            self.first_ts = Some(ts_event);
        }
        self.oldest_ts = Some(self.oldest_ts.map_or(ts_event, |o| o.min(ts_event)));

        // ── the whole log: does capture order disagree with event-time order? ──
        match self.high {
            Some(h) if ts_event < h => {
                // `late_arrivals` counts every inversion, always. It is the number the
                // golden summary carries and the number that answers "do the two
                // traversals differ", and that answer does not depend on whose fault it
                // is — so the diagnostic below is a **subset** of it and never a
                // subtraction from it.
                self.late_arrivals += 1;
                // Attribution, bounded where a bound exists: a bar explains a record
                // inside its own window and nothing older, while a receipt stamp has no
                // window and so excuses whatever follows it. Best-effort by construction,
                // and the looser half is 65% of a real tape — which is exactly why this
                // explains `late_arrivals` and is *not* what the feed metric is built from.
                if matches!(self.mark_derived, Some(window)
                    if window.is_none_or(|open| ts_event >= open))
                {
                    self.behind_derived_stamps += 1;
                }
            }
            _ => {
                self.high = Some(ts_event);
                self.mark_derived = match stamp {
                    StampSource::Derived { window_start } => Some(window_start),
                    StampSource::Venue | StampSource::Execution => None,
                };
            }
        }

        // ── the venue's own market data, ordered against itself ──
        //
        // A second pass over a subset rather than a subtraction from the count above, and
        // that is the whole design. A subtraction inherits every misattribution its terms
        // make; this cannot misattribute, because a record carrying nothing but a receipt
        // stamp is not in the subset to be blamed *or* excused. It is the only ordering
        // claim about a venue this format can honestly support.
        if stamp == StampSource::Venue {
            self.venue_stamped += 1;
            match self.venue_high {
                Some(h) if ts_event < h => self.venue_stamped_out_of_order += 1,
                _ => self.venue_high = Some(ts_event),
            }
        }
    }
}

impl From<&Event> for LoggedEvent {
    fn from(ev: &Event) -> Self {
        match ev {
            Event::Market(m) => LoggedEvent::Market(m.clone()),
            Event::Exec(e) => LoggedEvent::Exec(e.clone()),
        }
    }
}

impl From<LoggedEvent> for Event {
    fn from(ev: LoggedEvent) -> Self {
        match ev {
            LoggedEvent::Market(m) => Event::Market(m),
            LoggedEvent::Exec(e) => Event::Exec(e),
        }
    }
}

/// One captured event, with the two keys replay orders on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogRecord {
    /// Position in the order the **bus** delivered the event, counted from zero.
    ///
    /// This is what preserves the interleaving a live session actually saw. Event
    /// time alone cannot: two events stamped the same nanosecond are ordered by
    /// arrival, and without `seq` a replay would be free to swap them.
    pub seq: u64,
    /// The event's own time — the ordering key, lifted out of the payload so the
    /// reader can cross-check it (see the module docs).
    pub ts_event: Nanos,
    pub event: LoggedEvent,
}

impl LogRecord {
    /// Refuse a record whose recorded key no longer matches its payload.
    fn check(&self, line: usize) -> Result<(), ReplayError> {
        let derived = self.event.ts_event();
        if derived != self.ts_event {
            return Err(ReplayError::OrderingKeyChanged {
                line,
                recorded: self.ts_event,
                derived,
            });
        }
        Ok(())
    }
}

/// One line of a log body.
///
/// A tagged enum rather than two positional line kinds, and the tag costs ~10 bytes on
/// every event line. It buys the property ADR-0018 §1 bought for [`LoggedEvent`]: the
/// match below is exhaustive, so a body line kind added later cannot be quietly dropped
/// by a reader that was written before it — it is a compile error until it is handled.
/// A positional second line would instead make the format's shape a rule living only in
/// prose, which is the same class of promise `SCHEMA_VERSION` already asks a human to
/// keep once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogLine {
    /// The instrument grids the recording session planned against. **First body line
    /// of every log**, before any event — see [`crate::instruments`].
    Instruments(InstrumentSet),
    Event(LogRecord),
}

/// Parse one body line. `line` is the file line number, for the error only.
pub(crate) fn parse_body(text: &str, line: usize) -> Result<LogLine, ReplayError> {
    let parsed: LogLine =
        serde_json::from_str(text).map_err(|source| ReplayError::Json { line, source })?;
    if let LogLine::Event(rec) = &parsed {
        rec.check(line)?;
    }
    Ok(parsed)
}

/// A log being read one record at a time.
///
/// The header and the instrument line are consumed by [`open`](Self::open), so a reader
/// that exists has already been version-checked and already knows what grid the session
/// planned on. Everything after that is an iterator of [`LogRecord`].
///
/// Streaming rather than loading whole because a multi-hour soak is the artifact this
/// crate is *for*, and a reader bounded by memory made the capture's size cap a
/// loadability cap rather than a disk cap (ADR-0018 §9). What streaming does not fix is
/// stated on [`crate::replay`]: event-time ordering still needs to see the last record
/// before it can emit the first, so that traversal pays an index.
pub struct LogReader<R: BufRead> {
    src: R,
    header: LogHeader,
    instruments: InstrumentSet,
    /// Number of the last line read, 1-based — what an error names.
    line: usize,
    /// Byte offset of the next line, so a caller can index a log it will re-read.
    offset: u64,
    /// Latched: once a line has failed, reading on would report a *second* failure at a
    /// position that no longer means anything.
    failed: bool,
    /// Reused across records. A `String` per line would be one allocation per event on
    /// the read path, which is the path a multi-hour tape spends all its time on.
    buf: String,
}

impl LogReader<BufReader<File>> {
    /// Open a log file for streaming.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        Self::open(BufReader::new(File::open(path)?))
    }
}

impl<R: BufRead> LogReader<R> {
    /// Read and check the header and the instrument line.
    pub fn open(mut src: R) -> Result<Self, ReplayError> {
        let mut offset = 0u64;
        let mut line = 0usize;

        let mut first = String::new();
        let n = src.read_line(&mut first)?;
        if n == 0 {
            return Err(ReplayError::MissingHeader);
        }
        offset += n as u64;
        line += 1;
        let header: LogHeader = serde_json::from_str(first.trim_end())
            .map_err(|source| ReplayError::Json { line: 1, source })?;
        header.check()?;

        // The instrument line is required, not optional. A v2 log without one would be a
        // log that says it carries a grid and does not, which is the single artifact this
        // version exists to make impossible.
        let instruments = loop {
            let mut text = String::new();
            let n = src.read_line(&mut text)?;
            if n == 0 {
                return Err(ReplayError::MissingInstruments);
            }
            offset += n as u64;
            line += 1;
            // A trailing newline is normal, and so is a blank line left by an
            // interrupted writer; neither is a corrupt log.
            if text.trim().is_empty() {
                continue;
            }
            match parse_body(text.trim_end(), line)? {
                LogLine::Instruments(set) => break set,
                LogLine::Event(_) => return Err(ReplayError::MissingInstruments),
            }
        };

        Ok(Self {
            src,
            header,
            instruments,
            line,
            offset,
            failed: false,
            buf: String::new(),
        })
    }

    pub fn header(&self) -> &LogHeader {
        &self.header
    }

    /// The grids the recording session planned against.
    pub fn instruments(&self) -> &InstrumentSet {
        &self.instruments
    }

    /// Byte offset of the line [`next`](Iterator::next) will read.
    ///
    /// Taken *before* a record so an index can point back at it. The event-time
    /// traversal is built out of these, which is what lets it order a log it never
    /// holds.
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl<R: BufRead> Iterator for LogReader<R> {
    type Item = Result<LogRecord, ReplayError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            self.buf.clear();
            match self.src.read_line(&mut self.buf) {
                Ok(0) => return None,
                Ok(n) => {
                    self.offset += n as u64;
                    self.line += 1;
                }
                Err(e) => {
                    self.failed = true;
                    return Some(Err(ReplayError::Io(e)));
                }
            }
            if self.buf.trim().is_empty() {
                continue;
            }
            return Some(match parse_body(self.buf.trim_end(), self.line) {
                Ok(LogLine::Event(rec)) => Ok(rec),
                // A second instrument line means two sessions were concatenated into one
                // file, which replays as a session that never happened.
                Ok(LogLine::Instruments(_)) => {
                    self.failed = true;
                    Err(ReplayError::RepeatedInstruments { line: self.line })
                }
                Err(e) => {
                    self.failed = true;
                    Err(e)
                }
            });
        }
    }
}

/// A parsed event log, held in memory.
///
/// The whole-file form of [`LogReader`], and it is *built* out of one — a second parser
/// would be a second opinion about what a log says. Kept because a fixture-sized log is
/// easier to hold than to stream, and because [`ReplaySource::new`](crate::ReplaySource::new)
/// takes one. For a session-sized log, open the path instead: it never comes into
/// memory at all.
#[derive(Debug, Clone)]
pub struct EventLog {
    header: LogHeader,
    instruments: InstrumentSet,
    records: Vec<LogRecord>,
}

impl EventLog {
    /// Parse a log from any reader.
    pub fn read(src: impl BufRead) -> Result<Self, ReplayError> {
        let mut reader = LogReader::open(src)?;
        let mut records = Vec::new();
        for rec in &mut reader {
            records.push(rec?);
        }
        Ok(Self {
            header: reader.header,
            instruments: reader.instruments,
            records,
        })
    }

    /// Open and parse a log file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        Self::read(BufReader::new(File::open(path)?))
    }

    pub fn header(&self) -> &LogHeader {
        &self.header
    }

    /// The grids the recording session planned against.
    pub fn instruments(&self) -> &InstrumentSet {
        &self.instruments
    }

    pub fn records(&self) -> &[LogRecord] {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{a_fill, a_trade, declared_grids, log_bytes, log_bytes_with};

    #[test]
    fn a_record_pins_its_wire_shape_so_a_vocabulary_change_cannot_pass_unnoticed() {
        // `schema_version` is a human promise, and this is the test that forces the
        // human to keep it: a field added to `Trade`, a renamed variant, a changed
        // serde attribute all break here. Without it the vocabulary could drift while
        // old logs kept deserializing into the new shape, and a golden replay would
        // compare two runs that meant different things by the same bytes.
        let line = LogLine::Event(LogRecord {
            seq: 0,
            ts_event: 1_000,
            event: LoggedEvent::from(&a_trade(1_000)),
        });
        assert_eq!(
            serde_json::to_string(&line).unwrap(),
            r#"{"Event":{"seq":0,"ts_event":1000,"event":{"Market":{"Trade":{"symbol_id":1,"px":"100.5","sz":"2","side":"buy","ts_event":1000}}}}}"#,
            "money stays a decimal string, never a float"
        );
    }

    #[test]
    fn a_log_written_by_an_incompatible_build_is_refused_not_reinterpreted() {
        // The whole point of the version: serde would happily decode most of a
        // next-version log into today's types, and the replay would be green and wrong.
        //
        // Written against the constant, never against a literal. A hard-coded
        // "impossible future version" stops being impossible the day somebody bumps the
        // constant to it, and then this test substitutes nothing, reads a log this build
        // accepts, and passes while guarding nothing — the exact shape of failure the
        // version exists to prevent.
        let bytes = log_bytes(&[a_trade(1)]);
        let next = SCHEMA_VERSION + 1;
        let bumped = bytes.replace(
            &format!("\"schema_version\":{SCHEMA_VERSION}"),
            &format!("\"schema_version\":{next}"),
        );
        assert_ne!(
            bumped, bytes,
            "nothing was bumped, so nothing is under test"
        );
        let err = EventLog::read(bumped.as_bytes()).unwrap_err();
        assert!(
            matches!(err, ReplayError::IncompatibleVersion { found, .. } if found == next),
            "{err}"
        );

        let foreign = bytes.replace("axon.eventlog", "someone.elses.jsonl");
        let err = EventLog::read(foreign.as_bytes()).unwrap_err();
        assert!(matches!(err, ReplayError::ForeignLog { .. }), "{err}");
    }

    #[test]
    fn a_log_from_before_the_instrument_table_is_refused_by_name_not_replayed_unrounded() {
        // The decision ADR-0027 had to make, made executable. A v1 log carries no grid,
        // so a reader that accepted it would plan under `Precision::Unconstrained` while
        // the session it reproduces planned under `Precision::Known` — the exact
        // divergence the version bump exists to end, still happening, now over a file
        // that looks current. The refusal names the cause, because the generic version
        // message sends an operator to look for a build rather than to re-capture.
        // A literal `1` here and not a constant: it names a specific historical format —
        // the one that existed before the instrument line — rather than a moving target.
        let v1 = log_bytes(&[a_trade(1)]).replace(
            &format!("\"schema_version\":{SCHEMA_VERSION}"),
            "\"schema_version\":1",
        );
        let err = EventLog::read(v1.as_bytes()).unwrap_err();
        assert!(
            matches!(err, ReplayError::LogPredatesInstruments { found: 1 }),
            "{err}"
        );
        assert!(
            err.to_string().to_lowercase().contains("re-capture"),
            "the message has to say what to do about it: {err}"
        );
    }

    #[test]
    fn a_log_whose_first_body_line_is_not_the_grid_is_refused() {
        // A v2 log without an instrument line claims to carry a grid and does not, which
        // is the one artifact this version exists to make impossible. Skipping the line
        // and reading on would put the format's most important promise behind a `null`
        // nobody looks at.
        let bytes = log_bytes(&[a_trade(1)]);
        let stripped: String = bytes
            .lines()
            .filter(|l| !l.starts_with(r#"{"Instruments""#))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(matches!(
            EventLog::read(stripped.as_bytes()).unwrap_err(),
            ReplayError::MissingInstruments
        ));
    }

    #[test]
    fn two_logs_concatenated_are_refused_rather_than_replayed_as_one_session() {
        // `cat a.jsonl b.jsonl > both.jsonl` is a thing an operator does at 03:00. The
        // result parses line by line and replays green over a session whose `seq` restarts
        // in the middle and whose second half was recorded against a different grid.
        // Whole file onto whole file: the second header lands where a record belongs and
        // fails to parse as one. Loud, and at the right line.
        let a = log_bytes(&[a_trade(1)]);
        let b = log_bytes(&[a_trade(2)]);
        assert!(matches!(
            EventLog::read(format!("{a}{b}").as_bytes()).unwrap_err(),
            ReplayError::Json { line: 4, .. }
        ));
        // Bodies spliced without the second header — the shape a `tail -n +2` produces,
        // and the one that would otherwise parse all the way through.
        let spliced: String = a
            .lines()
            .chain(b.lines().skip(1))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(matches!(
            EventLog::read(spliced.as_bytes()).unwrap_err(),
            ReplayError::RepeatedInstruments { .. }
        ));
    }

    #[test]
    fn a_record_whose_ordering_key_no_longer_matches_its_payload_is_refused() {
        // `Ticker::ts_event` is derived, so a change to the venue-time/receipt-time
        // fallback would re-key events a log had already ordered. Replay would then
        // interleave differently from the reference it is being compared against —
        // a divergence that looks like a strategy bug, not a format bug.
        let bytes = log_bytes(&[a_trade(1_000)])
            .replace("\"ts_event\":1000,\"event\"", "\"ts_event\":999,\"event\"");
        let err = EventLog::read(bytes.as_bytes()).unwrap_err();
        assert!(
            matches!(
                err,
                ReplayError::OrderingKeyChanged {
                    recorded: 999,
                    derived: 1_000,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_headerless_or_truncated_log_fails_loudly() {
        assert!(matches!(
            EventLog::read(&b""[..]).unwrap_err(),
            ReplayError::MissingHeader
        ));
        // A writer killed mid-line leaves a partial record. Skipping it would quietly
        // shorten the golden log, and a shorter replay still "passes" every check that
        // only compares the rows both runs produced.
        let truncated = format!("{}{{\"Event\":{{\"seq\":0,\"ts_ev", log_bytes(&[]));
        assert!(matches!(
            EventLog::read(truncated.as_bytes()).unwrap_err(),
            ReplayError::Json { line: 3, .. }
        ));
    }

    #[test]
    fn both_event_families_round_trip_through_the_log() {
        let events = vec![a_trade(10), a_fill(20)];
        let log = EventLog::read(log_bytes(&events).as_bytes()).unwrap();
        assert_eq!(log.header().schema, SCHEMA);
        assert_eq!(log.len(), 2);
        let back: Vec<Event> = log
            .records()
            .iter()
            .map(|r| Event::from(r.event.clone()))
            .collect();
        assert_eq!(back, events);
        assert_eq!(log.records()[1].seq, 1);
    }

    #[test]
    fn the_grid_a_session_planned_on_survives_the_round_trip() {
        // The point of the version bump: a replay of this log can round exactly the way
        // the session that wrote it did, without being told anything by its caller.
        let log =
            EventLog::read(log_bytes_with(declared_grids(), &[a_trade(10)]).as_bytes()).unwrap();
        let table = log
            .instruments()
            .to_table()
            .unwrap()
            .expect("the log declared one");
        let spec = table.get(axon_core::SymbolId::new(1)).expect("symbol 1");
        assert_eq!(
            spec.price.tick_at(rust_decimal_macros::dec!(49757.96)),
            rust_decimal_macros::dec!(1),
            "five significant figures at that magnitude is a one-dollar tick"
        );
    }

    #[test]
    fn a_blank_line_is_not_a_corrupt_log() {
        let mut bytes = log_bytes(&[a_trade(1)]);
        bytes.push('\n');
        assert_eq!(EventLog::read(bytes.as_bytes()).unwrap().len(), 1);
    }

    #[test]
    fn a_streamed_log_yields_exactly_what_loading_it_whole_yields() {
        // Two readers would be two opinions about what a log says, and the drift would
        // land inside the harness built to detect drift. There is one parser; this is
        // the test that keeps it that way.
        let events: Vec<Event> = (1..=50).map(|i| a_trade(i * 10)).collect();
        let bytes = log_bytes(&events);
        let whole = EventLog::read(bytes.as_bytes()).unwrap();
        let streamed: Vec<LogRecord> = LogReader::open(bytes.as_bytes())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(streamed, whole.records());
    }

    #[test]
    fn a_streamed_offset_points_at_the_record_that_follows_it() {
        // The index the event-time traversal is built from: `(ts_event, offset)` and
        // nothing else, so a log can be ordered without being held. An offset that
        // pointed one line out would replay a session in an order it never ran.
        let events: Vec<Event> = (1..=5).map(|i| a_trade(i * 10)).collect();
        let bytes = log_bytes(&events);
        let mut reader = LogReader::open(bytes.as_bytes()).unwrap();
        let mut offsets = Vec::new();
        loop {
            let at = reader.offset();
            match reader.next() {
                Some(rec) => offsets.push((at, rec.unwrap())),
                None => break,
            }
        }
        assert_eq!(offsets.len(), 5);
        for (at, rec) in offsets {
            let line = bytes[at as usize..].lines().next().unwrap();
            let LogLine::Event(there) = parse_body(line, 0).unwrap() else {
                panic!("an offset pointed at the preamble");
            };
            assert_eq!(there, rec);
        }
    }
}
