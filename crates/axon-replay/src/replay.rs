//! [`ReplaySource`] — pushing a captured log back through the production core.
//!
//! Replay wires itself exactly the way live does: a producer publishes onto the
//! bounded [`bus`](axon_core::bus), and the core drains it with
//! [`run_blocking_clocked`], one event at a time, on one thread. Nothing about the
//! consumer side knows it is being replayed. That is the entire point — parity comes
//! from the backtest *being* the live program with a different producer attached, not
//! from a second implementation that agrees with it (`docs/07-parity-and-testing.md`).
//!
//! The [`ManualClock`] is the other half. `run_blocking_clocked` sets it to each
//! event's own `ts_event` before dispatch, so a handler asking "what time is it?"
//! during replay gets the event's time, not this afternoon's. A handler that reads a
//! wall clock instead — a staleness check, a TTL, a timer — would behave differently
//! on every run, and the golden test below is what makes that visible.
//!
//! # A log is streamed, not held
//!
//! ADR-0018 accepted "a log is loaded whole" and drew the consequence honestly: log
//! size is bounded by memory, so the capture's size cap was set where an artifact stops
//! being *loadable* rather than where a disk stops being writable — and a multi-hour
//! soak therefore stopped its own recording. [`ReplaySource::open`] streams instead, and
//! what that costs depends on the traversal, which is the part ADR-0018's reasoning got
//! right and this does not overturn:
//!
//! - [`ReplayOrder::AsCaptured`] is **O(1)**. The file already is the order, so the
//!   producer reads a line, sends it, and forgets it. This is also the only order a
//!   live-captured reference may legitimately be compared under (§4 below), so the
//!   traversal a real soak tape needs is the one that costs nothing.
//! - [`ReplayOrder::EventTime`] cannot emit its first event until it has seen the last
//!   record — a feed that can be late can be late by any amount, and a bounded lookahead
//!   would be a guess. So it holds an **index**: `(ts_event, byte offset)`, ~24 bytes a
//!   record measured, and re-reads each payload from disk in order. The bound did not go
//!   away; it went from the size of the events to the size of a key, which is the
//!   difference between a session slice and a night of tape. Measured: a 163 MiB log of
//!   410 400 records peaks at 4.8 MB `as-captured` and 14.7 MB in event time.
//!
//! One reader, either way: [`LogReader`] is the only thing in the crate that turns a
//! line into a record, so the streaming path and [`EventLog`] cannot come to disagree
//! about what a log says.
//!
//! # What replay does *not* reproduce
//!
//! **The venue's response to orders this run would place.** A log contains the fills
//! and order updates the *captured* session received, for the orders the *captured*
//! session sent. Replay re-delivers those recordings verbatim. If a strategy under
//! replay decides to buy where the original did not, no fill will ever arrive for it,
//! and no book level will move because of it: the market in a log is a recording, not
//! a counterparty. Nor is queue position, partial-fill sequencing, latency, or
//! rejection behaviour reproduced.
//!
//! So replay answers "does this code, on this input, still produce what it produced
//! before?" — determinism, refactor safety, feature/model parity. It does **not**
//! answer "would this strategy have made money?" That question needs a simulated
//! venue that models fills against the book, which is a separate adapter behind the
//! provider port and is deliberately not this crate. The distinction is stated here,
//! in the type docs, and in ADR-0018 because a replay harness that blurs it
//! manufactures confidence, and manufactured confidence is worse than no harness.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use axon_core::{
    bus, run_blocking_clocked, Event, EventHandler, EventSender, ManualClock, Nanos, TimedQueue,
};

use crate::error::ReplayError;
use crate::instruments::InstrumentSet;
use crate::log::{parse_body, EventLog, LogHeader, LogLine, LogReader, OrderingWatch};

/// How a log's records are ordered before being republished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayOrder {
    /// Event time, ties broken by capture sequence. **The default.**
    ///
    /// This is the order the architecture claims the core runs in
    /// (`docs/01-architecture.md`: "keyed on event time, not processing time"), and
    /// for a log with no late arrivals it is identical to [`AsCaptured`](Self::AsCaptured)
    /// — the two only diverge on the events that arrived out of order, which is
    /// exactly where a divergence is worth knowing about.
    #[default]
    EventTime,
    /// Exactly the order the bus delivered them live.
    ///
    /// The right choice when the question is "reproduce what the live session
    /// experienced", including its out-of-order arrivals. Event-time ordering would
    /// answer a counterfactual instead: a run over an interleaving the live session
    /// never saw, whose output cannot legitimately be compared against a live
    /// reference.
    AsCaptured,
}

/// What a replay pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayReport {
    pub events: u64,
    /// Oldest and newest event time in the log — the window it covers.
    ///
    /// A property of the log, not of the traversal, so the two [`ReplayOrder`]s report
    /// the same window. Under [`ReplayOrder::AsCaptured`] the last event *dispatched*
    /// may be older than `last_ts`; that is what a late arrival means.
    pub first_ts: Option<Nanos>,
    pub last_ts: Option<Nanos>,
    /// Records captured behind the event-time high-water mark.
    ///
    /// Non-zero means [`ReplayOrder::EventTime`] and [`ReplayOrder::AsCaptured`]
    /// produce different runs from this log, so a comparison against a live-captured
    /// reference is only meaningful under `AsCaptured`. Reported rather than fixed,
    /// because silently sorting a late feed into shape is how a harness ends up
    /// certifying a session that never happened.
    ///
    /// It answers *"do the two traversals differ"* and nothing narrower. For *why* they
    /// differ, see [`behind_derived_stamps`](Self::behind_derived_stamps); for whether the
    /// **venue** reordered anything, see
    /// [`reordered_by_the_feed`](Self::reordered_by_the_feed), which is a different
    /// measurement and not a share of this one.
    pub late_arrivals: u64,
    /// How many of those [`late_arrivals`](Self::late_arrivals) sat behind a mark set by
    /// a record the **venue did not timestamp**.
    ///
    /// A subset, so the field above keeps both its meaning and its value — it is in the
    /// golden summary, and a metric that changed value with a diagnostic would make every
    /// stored reference wrong for a reason nobody could see. This *explains* that number's
    /// size; the number to alarm on is
    /// [`reordered_by_the_feed`](Self::reordered_by_the_feed), which is measured rather
    /// than subtracted from this.
    ///
    /// Two records do this and both are ordinary on a real venue. A candle's `ts_event`
    /// is the moment its bar *will* close and the venue republishes the bar it is still
    /// filling (1 321 frames for 69 bars, 1 317 arriving before their own stamp, one bar
    /// republished 192 times). And `activeAssetCtx` carries no venue timestamp at all, so
    /// a [`Ticker`](axon_core::Ticker) is stamped at receipt (ADR-0011) — **65% of a real
    /// soak tape**. Both walk the high-water mark ahead of the market, and everything
    /// captured behind them looks late having done nothing wrong.
    pub behind_derived_stamps: u64,
    /// Records the **venue itself timestamped** — the denominator of
    /// [`reordered_by_the_feed`](Self::reordered_by_the_feed).
    pub venue_stamped: u64,
    /// Of those, how many arrived behind another venue-stamped record.
    pub venue_stamped_out_of_order: u64,
    /// Records actually handed to the handler.
    ///
    /// Equal to [`events`](Self::events) on every healthy run. Smaller means the log
    /// changed underneath the replay — the file is scanned when it is opened and
    /// re-read as it is dispatched — and what the handler saw is therefore a *prefix*
    /// of what this report describes. A count that could only be inferred by comparing
    /// two other numbers is a count nobody compares; ADR-0018 §9's `events + missed`
    /// identity exists at the writing end for the same reason.
    pub dispatched: u64,
}

impl ReplayReport {
    /// Times the venue's own market-data feed delivered its own timestamps out of order,
    /// out of [`venue_stamped`](Self::venue_stamped) records that carried one.
    ///
    /// **The number to alarm on, and the only ordering claim about a venue this format
    /// can honestly make.** `late_arrivals` answers a different question — "do the
    /// traversals differ" — and on any live tape it is dominated by records the venue
    /// never timestamped: a 1 h 44 m soak reported 15.94% there and **0.08%** here.
    ///
    /// It is a *measurement*, not a residue. An earlier version subtracted the forming
    /// bars from `late_arrivals`; that inherited every misattribution its terms could make,
    /// and on a tape with a ticker feed or a reconnect in it — which is every live tape —
    /// it reported a network problem that was not there.
    ///
    /// What it does **not** cover, each excluded for a stated reason rather than an
    /// oversight: bars, whose stamp is a convention about a window rather than a moment;
    /// tickers with no venue time, which are ordered on our own receipt clock; and the
    /// execution stream, which the venue **replays wholesale on reconnect** (the soak saw
    /// fills arrive 2.94 hours stale that way) and which is therefore not a feed
    /// delivering late. A reordering inside any of those three is invisible here, and
    /// that is the price of the number meaning exactly one thing.
    pub fn reordered_by_the_feed(&self) -> u64 {
        self.venue_stamped_out_of_order
    }
}

/// Where a [`ReplaySource`]'s records come from.
///
/// Two variants and not one, because a fixture-sized log is easier to hold than to
/// re-read and a session-sized one must never be held at all. They are not two readers:
/// [`Tape::File`] streams through [`LogReader`], and [`EventLog`] is built by draining
/// exactly that.
#[derive(Debug, Clone)]
enum Tape {
    /// Already in memory — what [`ReplaySource::new`] wraps.
    Loaded(EventLog),
    /// On disk, read once at open to be checked and indexed, then re-read to be
    /// dispatched.
    File {
        path: PathBuf,
        header: LogHeader,
        instruments: InstrumentSet,
        /// The log's own shape, computed by the scan at open so that a caller learns
        /// about a corrupt log *before* a single event reaches a handler — which is what
        /// `EventLog::open` did for free when it loaded everything.
        report: ReplayReport,
    },
}

/// Republishes a captured event log onto a core bus.
#[derive(Debug, Clone)]
pub struct ReplaySource {
    tape: Tape,
    order: ReplayOrder,
    capacity: usize,
}

/// The report a watched pass yields. One conversion, so the streamed and the loaded
/// tape cannot describe the same file differently.
fn report_of(watch: &OrderingWatch) -> ReplayReport {
    ReplayReport {
        events: watch.events,
        // The *window*, not the first record: `ReplayReport` documents these two as
        // "oldest and newest event time in the log".
        first_ts: watch.oldest_ts,
        last_ts: watch.high,
        late_arrivals: watch.late_arrivals,
        behind_derived_stamps: watch.behind_derived_stamps,
        venue_stamped: watch.venue_stamped,
        venue_stamped_out_of_order: watch.venue_stamped_out_of_order,
        dispatched: 0,
    }
}

/// Bus depth used by [`ReplaySource::run`].
///
/// Only a buffer between the producer thread and the core loop: it changes how far
/// ahead the producer may get, never what the core observes, because a single
/// producer and a FIFO channel deliver in send order at any depth.
const DEFAULT_BUS_CAPACITY: usize = 1024;

impl ReplaySource {
    pub fn new(log: EventLog) -> Self {
        Self {
            tape: Tape::Loaded(log),
            order: ReplayOrder::default(),
            capacity: DEFAULT_BUS_CAPACITY,
        }
    }

    /// Open a log on disk and prepare to stream it.
    ///
    /// Scans the whole file once, which is what makes a malformed record an error *here*
    /// rather than a run that dispatched half a session and then gave up. Nothing but
    /// the header, the instrument table and the report survives the scan.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let path = path.as_ref().to_path_buf();
        let mut reader = LogReader::open_path(&path)?;
        let mut watch = OrderingWatch::default();
        for rec in &mut reader {
            let rec = rec?;
            watch.observe(rec.ts_event, rec.event.stamp_source());
        }
        let report = report_of(&watch);
        Ok(Self {
            tape: Tape::File {
                header: reader.header().clone(),
                instruments: reader.instruments().clone(),
                path,
                report,
            },
            order: ReplayOrder::default(),
            capacity: DEFAULT_BUS_CAPACITY,
        })
    }

    pub fn with_order(mut self, order: ReplayOrder) -> Self {
        self.order = order;
        self
    }

    pub fn with_bus_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn header(&self) -> &LogHeader {
        match &self.tape {
            Tape::Loaded(log) => log.header(),
            Tape::File { header, .. } => header,
        }
    }

    /// The grids the recording session planned against.
    ///
    /// The whole of `SCHEMA_VERSION` 2. A caller that ignores this plans *unconstrained*
    /// and gets prices the session it is reproducing could not have sent (ADR-0027).
    pub fn instruments(&self) -> &InstrumentSet {
        match &self.tape {
            Tape::Loaded(log) => log.instruments(),
            Tape::File { instruments, .. } => instruments,
        }
    }

    /// The log's own shape — how many records, the window they cover, and how their
    /// capture order disagrees with event time — independent of who consumes it.
    ///
    /// Public because a caller has to be able to ask *before* replaying: a log whose
    /// inversions are all forming bars needs a different sentence from one whose feed is
    /// reordering, and `dispatched` is the only field a run fills in.
    pub fn report(&self) -> ReplayReport {
        match &self.tape {
            Tape::File { report, .. } => *report,
            Tape::Loaded(log) => {
                let mut watch = OrderingWatch::default();
                for rec in log.records() {
                    watch.observe(rec.ts_event, rec.event.stamp_source());
                }
                report_of(&watch)
            }
        }
    }

    /// Drive `emit` over every event in traversal order, stopping early when it says so.
    ///
    /// The single place a traversal is defined, so the callers below cannot end up with
    /// several orders. Returns how many events were handed over **and** whatever went
    /// wrong — both, because a failure halfway through has already dispatched a prefix,
    /// and a count discarded with the error is the number a caller needs to know the
    /// run was short.
    fn traverse(&self, emit: impl FnMut(Event) -> bool) -> (u64, Option<ReplayError>) {
        let mut sent = 0u64;
        let outcome = self.traverse_inner(&mut sent, emit);
        (sent, outcome.err())
    }

    fn traverse_inner(
        &self,
        sent: &mut u64,
        mut emit: impl FnMut(Event) -> bool,
    ) -> Result<(), ReplayError> {
        match (&self.tape, self.order) {
            // In memory: the order is a permutation of a slice.
            (Tape::Loaded(log), order) => {
                let events: Vec<Event> = match order {
                    ReplayOrder::AsCaptured => log
                        .records()
                        .iter()
                        .map(|r| Event::from(r.event.clone()))
                        .collect(),
                    ReplayOrder::EventTime => {
                        let mut q = TimedQueue::new();
                        for rec in log.records() {
                            q.push(rec.ts_event, Event::from(rec.event.clone()));
                        }
                        let mut out = Vec::with_capacity(log.len());
                        while let Some((_, ev)) = q.pop() {
                            out.push(ev);
                        }
                        out
                    }
                };
                for ev in events {
                    // Counted only once it has been taken: `sent` is what the consumer
                    // saw, so a refused hand-off is not one of them.
                    if !emit(ev) {
                        break;
                    }
                    *sent += 1;
                }
            }
            // On disk, in the order it was written: one pass, nothing held.
            (Tape::File { path, .. }, ReplayOrder::AsCaptured) => {
                for rec in LogReader::open_path(path)? {
                    if !emit(Event::from(rec?.event)) {
                        break;
                    }
                    *sent += 1;
                }
            }
            // On disk, in event time: an index pass, then a seeking pass. Two passes and
            // not one because ordering cannot begin until the last record has been seen,
            // and the thing worth not holding is the payloads.
            (Tape::File { path, .. }, ReplayOrder::EventTime) => {
                // The core's own `TimedQueue`, keyed by `ts_event` with the byte offset
                // as its value — not a local `sort_by_key`. Reusing the core primitive is
                // what guarantees replay and a future event-time-scheduling core cannot
                // disagree about ties, and it is what makes `seq` unnecessary here: the
                // queue breaks ties on push order, which *is* capture order.
                let mut q: TimedQueue<u64> = TimedQueue::new();
                let mut reader = LogReader::open_path(path)?;
                loop {
                    let at = reader.offset();
                    match reader.next() {
                        Some(rec) => q.push(rec?.ts_event, at),
                        None => break,
                    }
                }

                let mut file = BufReader::new(File::open(path)?);
                let mut pos = 0u64;
                let mut line = String::new();
                while let Some((_, at)) = q.pop() {
                    // `seek_relative` keeps the buffer when the target is inside it, and
                    // a log is *nearly* in order, so the common step is the next line and
                    // costs no syscall at all. A plain `seek` would drop the read-ahead on
                    // every record and turn one pass into a few million.
                    if at != pos {
                        file.seek_relative(at as i64 - pos as i64)?;
                        pos = at;
                    }
                    line.clear();
                    let n = file.read_line(&mut line)?;
                    pos += n as u64;
                    // The scan at `open` parsed this line; if it will not parse now the
                    // file moved under us, and saying so beats dispatching a hole.
                    let LogLine::Event(rec) = parse_body(line.trim_end(), 0)
                        .map_err(|_| ReplayError::LogChanged { offset: at })?
                    else {
                        return Err(ReplayError::LogChanged { offset: at });
                    };
                    if !emit(Event::from(rec.event)) {
                        break;
                    }
                    *sent += 1;
                }
            }
        }
        Ok(())
    }

    /// Publish every event onto an existing bus, blocking on backpressure.
    ///
    /// For wiring replay into a runtime that owns its own bus and handler chain. The
    /// caller is responsible for draining `tx`'s receiver — with a bounded bus and no
    /// consumer this will block, exactly as a live producer would.
    pub fn publish(&self, tx: &EventSender) -> Result<ReplayReport, ReplayError> {
        let mut closed_after = None;
        let (sent, failed) = self.traverse(|ev| match tx.send(ev) {
            Ok(()) => true,
            Err(_) => {
                closed_after = Some(());
                false
            }
        });
        if let Some(e) = failed {
            return Err(e);
        }
        if closed_after.is_some() {
            return Err(ReplayError::BusClosed { published: sent });
        }
        let mut report = self.report();
        report.dispatched = sent;
        Ok(report)
    }

    /// Run the whole loop: bus, producer, and the core's own
    /// [`run_blocking_clocked`] driving `handler` against `clock`.
    ///
    /// The producer runs on a scoped thread so the bus behaves as it does live —
    /// bounded, with real backpressure — rather than being drained in lockstep by a
    /// single-threaded pump that could never fill it. The thread is a scheduling
    /// detail and not a source of nondeterminism: one producer plus a FIFO channel
    /// means the consumer observes exactly the send order, whatever the scheduler
    /// does in between.
    ///
    /// Infallible, and that is a claim [`open`](Self::open) earns rather than one made
    /// here: the file was read end to end and checked before this was called, so the only
    /// failure left is the file changing underneath the run, which lands in
    /// [`ReplayReport::dispatched`] rather than in a `Result` a caller could `let _ =`.
    pub fn run(&self, clock: &ManualClock, handler: &mut impl EventHandler) -> ReplayReport {
        let (tx, rx) = bus(self.capacity);
        let dispatched = std::thread::scope(|s| {
            let producer = s.spawn(move || {
                // A closed bus means the core stopped; there is nothing useful to do but
                // stop producing. A read error here is the file having moved (see
                // `traverse`), and it stops the producer for the same reason.
                let (sent, failed) = self.traverse(|ev| tx.send(ev).is_ok());
                drop(tx);
                if let Some(e) = failed {
                    // Allocation and a syscall on a path that runs once, if ever. The
                    // alternative is `dispatched < events` with nothing anywhere saying
                    // why, and a short run whose cause has to be guessed is the shape of
                    // failure this crate exists to refuse.
                    eprintln!("replay: the log stopped yielding records after {sent}: {e}");
                }
                sent
            });
            run_blocking_clocked(&rx, clock, handler);
            producer.join().unwrap_or(0)
        });
        let mut report = self.report();
        report.dispatched = dispatched;
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::Capture;
    use crate::test_support::{
        a_candle, a_fill, a_receipt_stamped_ticker, a_trade, a_trade_at, declared_grids, log_bytes,
        log_bytes_with, DigestHandler,
    };
    use axon_core::Clock;
    use rust_decimal::Decimal;

    /// One millisecond, so a test's numbers read as the timeline they describe.
    const MS: Nanos = 1_000_000;

    fn source(events: &[Event]) -> ReplaySource {
        ReplaySource::new(EventLog::read(log_bytes(events).as_bytes()).unwrap())
    }

    /// The same events, written to a real file so the streaming path is exercised.
    ///
    /// Returned with the path so the caller can delete it; a fixture that leaks files
    /// into `/tmp` fails on the machine that runs the gate twice.
    fn file_source(tag: &str, events: &[Event]) -> (ReplaySource, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "axon-replay-stream-{tag}-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, log_bytes(events)).unwrap();
        (ReplaySource::open(&path).unwrap(), path)
    }

    #[test]
    fn replaying_the_same_log_twice_is_byte_identical() {
        // The golden property, and the reason any of the rest is worth building: if
        // two runs over one log can differ, every comparison above this rung — model
        // parity, feature parity, shadow-trading diffs — is measuring noise. The
        // digest deliberately includes the clock reading each handler saw, so a
        // handler reading wall-clock time breaks this test rather than quietly
        // poisoning the gates built on it.
        let src = source(&[a_trade(30), a_fill(10), a_trade(20), a_trade(20)]);

        let clock = ManualClock::new(0);
        let mut first = DigestHandler::new(&clock);
        let r1 = src.run(&clock, &mut first);
        let first = first.into_bytes();

        let clock = ManualClock::new(0);
        let mut second = DigestHandler::new(&clock);
        let r2 = src.run(&clock, &mut second);

        assert_eq!(
            first,
            second.into_bytes(),
            "two replays of one log diverged"
        );
        assert_eq!(r1, r2);
        assert_eq!(r1.events, 4);
        assert_eq!(r1.dispatched, 4);
    }

    #[test]
    fn a_streamed_log_replays_exactly_what_a_loaded_one_does() {
        // The claim streaming has to earn. A soak tape is read a line at a time and a
        // fixture is held whole; if those two traversals could differ, every golden
        // taken one way and compared the other would be comparing two programs.
        let events: Vec<Event> = (1..=200)
            .map(|i| {
                if i % 7 == 0 {
                    a_fill(i * 10)
                } else {
                    a_trade(i * 10)
                }
            })
            .collect();
        let (streamed, path) = file_source("parity", &events);
        let loaded = source(&events);

        for order in [ReplayOrder::EventTime, ReplayOrder::AsCaptured] {
            let clock = ManualClock::new(0);
            let mut a = DigestHandler::new(&clock);
            let ra = streamed.clone().with_order(order).run(&clock, &mut a);
            let clock = ManualClock::new(0);
            let mut b = DigestHandler::new(&clock);
            let rb = loaded.clone().with_order(order).run(&clock, &mut b);
            assert_eq!(a.into_bytes(), b.into_bytes(), "{order:?} diverged");
            assert_eq!(ra, rb);
            assert_eq!(ra.dispatched, 200);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_streamed_event_time_replay_orders_late_arrivals_without_holding_the_log() {
        // The traversal that cannot be O(1), done the way it can be: an index of
        // `(ts_event, offset)` and a second read. The out-of-order records are the whole
        // reason event-time order exists, so the test that streaming preserves the order
        // has to contain some.
        let mut events: Vec<Event> = (1..=100).map(|i| a_trade(i * 10)).collect();
        events.push(a_trade(35));
        events.push(a_trade(5));
        let (src, path) = file_source("eventtime", &events);

        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        let report = src.run(&clock, &mut h);
        let mut expected: Vec<Nanos> = events.iter().map(|e| e.ts_event()).collect();
        expected.sort();
        assert_eq!(h.event_times(), expected);
        assert_eq!(report.late_arrivals, 2);
        assert_eq!(report.first_ts, Some(5));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_corrupt_record_is_refused_when_the_log_is_opened_not_halfway_through_the_run() {
        // Streaming's real risk: a reader that discovered a bad line mid-run would have
        // already handed a prefix to the handlers, and a harness reporting success over
        // a prefix is the exact artifact ADR-0018 refuses. The scan at `open` is what
        // buys the old behaviour back.
        let path = std::env::temp_dir().join(format!(
            "axon-replay-corrupt-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let good = log_bytes(&[a_trade(10), a_trade(20)]);
        std::fs::write(&path, format!("{good}{{\"Event\":{{\"seq\":2,\"ts_ev")).unwrap();
        let err = ReplaySource::open(&path).unwrap_err();
        assert!(matches!(err, ReplayError::Json { .. }), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_forming_bar_shows_up_as_its_own_diagnosis_rather_than_as_a_bad_feed() {
        // The reader's half of the same rule the writer applies, and the reason it is one
        // rule in one place: a soak's own status line and a replay of its tape must not
        // disagree about whether the feed reordered.
        //
        // The traversals really do differ here — event time delivers the bar at its close,
        // after the trades it actually preceded — so `late_arrivals` stays non-zero and
        // keeps its documented meaning. What is new is that the number beside it says the
        // venue's stamping did it, and `reordered_by_the_feed` comes out at zero.
        let events = [
            a_trade(10_000 * MS),
            a_candle(10_000, 60),
            a_trade(10_010 * MS),
            a_trade(10_020 * MS),
        ];
        let report = source(&events).report();
        assert_eq!(report.late_arrivals, 2, "the traversals differ");
        assert_eq!(report.behind_derived_stamps, 2);
        assert_eq!(
            report.reordered_by_the_feed(),
            0,
            "nothing here is the network's doing"
        );
        assert_eq!(
            report.venue_stamped, 3,
            "the bar's own stamp is a convention about a window, not a moment the venue \
             timestamped, so it is not in the subset that measures the venue"
        );

        // Event time still puts the bar last, which is what its own ordering key asks
        // for. The count explains that; it does not paper over it.
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        source(&events).run(&clock, &mut h);
        assert_eq!(
            h.event_times(),
            vec![10_000 * MS, 10_010 * MS, 10_020 * MS, 10_060 * MS],
            "the bar sorts to its close, because that is what its ts_event says"
        );
    }

    #[test]
    fn a_streamed_log_and_a_loaded_one_agree_about_whose_fault_an_inversion_is() {
        // Two readers would be two opinions, and this one is subtle enough to drift
        // quietly: the attribution depends on state carried *across* records (which bar
        // holds the high-water mark), which is exactly the kind of thing a second
        // implementation gets almost right.
        let events = [
            a_trade(10_000 * MS),
            a_candle(10_000, 60),
            a_trade(10_030 * MS),
            a_trade(9_000 * MS),
        ];
        let (streamed, path) = file_source("forming", &events);
        assert_eq!(streamed.report(), source(&events).report());
        assert_eq!(streamed.report().behind_derived_stamps, 1);
        assert_eq!(
            streamed.report().reordered_by_the_feed(),
            1,
            "the trade a second older than the bar's window really is out of order \
             against the venue-stamped trades around it"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_feed_metric_is_a_measurement_and_not_a_residue() {
        // The lesson a 1 h 44 m soak taught, made executable. The first version of this
        // subtracted the explained inversions from `late_arrivals`, and on a real tape
        // that residue was 409 "reorderings" of which **zero** were real — because a
        // receipt-stamped ticker (65% of the tape) advances the mark past venue-stamped
        // data by one network hop and nothing was subtracting for it.
        //
        // Here the subtraction would say 2 and the measurement says 0, which is the truth:
        // the three venue-stamped trades were delivered in perfect order.
        let events = [
            a_trade(10_000 * MS),
            a_receipt_stamped_ticker(10_500 * MS),
            a_trade(10_100 * MS),
            a_receipt_stamped_ticker(10_600 * MS),
            a_trade(10_200 * MS),
        ];
        let report = source(&events).report();

        assert_eq!(report.late_arrivals, 2);
        assert_eq!(
            report.late_arrivals - report.behind_derived_stamps,
            0,
            "the old subtraction happens to be right here; the tape is where it was not"
        );
        assert_eq!(report.venue_stamped, 3);
        assert_eq!(
            report.reordered_by_the_feed(),
            0,
            "measured over the venue's own stamps, ordered against each other"
        );
    }

    #[test]
    fn a_reconnect_replaying_its_whole_snapshot_is_not_a_feed_that_reorders() {
        // 36 deliberate disconnects in the soak, and `userFills` replays its entire
        // snapshot on every one — execution times up to 2.94 hours stale. Every one of
        // those is an inversion, and none of them is the market-data feed doing anything.
        // Counting them would make a reconnection test fail itself.
        let mut events = vec![a_trade(10_000 * MS)];
        events.extend((1..=5).map(|i| a_fill(i * 100 * MS)));
        events.push(a_trade(10_100 * MS));
        let report = source(&events).report();

        assert_eq!(report.late_arrivals, 5, "the traversals do differ");
        assert_eq!(report.venue_stamped, 2);
        assert_eq!(
            report.reordered_by_the_feed(),
            0,
            "a snapshot re-sent is not a feed delivering late"
        );
    }

    #[test]
    fn handlers_observe_event_time_not_wall_clock() {
        // `run_blocking_clocked` sets the ManualClock to each event's own timestamp
        // before dispatch. Without this a TTL or staleness check would measure the
        // age of the replay run instead of the age of the data, and would come out
        // differently every time the test ran.
        let src = source(&[a_trade(1_000), a_trade(2_500)]);
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        src.run(&clock, &mut h);
        assert_eq!(h.clock_readings(), vec![1_000, 2_500]);
        assert_eq!(clock.now_ns(), 2_500, "the clock ends at the last event");
    }

    #[test]
    fn late_arrivals_are_reordered_by_default_and_counted_either_way() {
        // Two honest answers, and the count is what stops a caller picking the wrong
        // one silently: event-time order is the architecture's claim, capture order is
        // what the live session actually experienced, and they differ on exactly the
        // late events.
        let events = [a_trade(30), a_trade(10), a_trade(20)];

        let clock = ManualClock::new(0);
        let mut sorted = DigestHandler::new(&clock);
        let report = source(&events).run(&clock, &mut sorted);
        assert_eq!(sorted.event_times(), vec![10, 20, 30]);
        assert_eq!(report.late_arrivals, 2);
        assert_eq!(report.first_ts, Some(10));

        let clock = ManualClock::new(0);
        let mut as_captured = DigestHandler::new(&clock);
        let report = source(&events)
            .with_order(ReplayOrder::AsCaptured)
            .run(&clock, &mut as_captured);
        assert_eq!(as_captured.event_times(), vec![30, 10, 20]);
        assert_eq!(
            report.late_arrivals, 2,
            "the log's shape does not change with how it is read"
        );
    }

    #[test]
    fn events_stamped_the_same_nanosecond_keep_their_captured_order() {
        // Event time alone does not order simultaneous events. If a reorder were
        // allowed here, a book snapshot and the trade that produced it could swap on
        // one run in a thousand — the kind of flake that gets a golden test disabled.
        let events: Vec<Event> = (1..=8).map(|i| a_trade_at(500, Decimal::from(i))).collect();
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        source(&events).run(&clock, &mut h);
        assert_eq!(
            h.trade_prices(),
            (1..=8).map(Decimal::from).collect::<Vec<_>>()
        );

        // …and the same through the streaming index, which keys on `ts_event` alone and
        // gets the tie-break from `TimedQueue`'s push order rather than from `seq`.
        let (src, path) = file_source("ties", &events);
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        src.run(&clock, &mut h);
        assert_eq!(
            h.trade_prices(),
            (1..=8).map(Decimal::from).collect::<Vec<_>>(),
            "the index lost the capture order of simultaneous events"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_captured_session_round_trips_through_replay_unchanged() {
        // Capture → replay → capture. The second log's records must equal the first's,
        // which is the end-to-end statement that neither serialization nor republishing
        // perturbs the stream.
        let events = [a_trade(10), a_fill(20), a_trade(30)];
        let original = log_bytes_with(declared_grids(), &events);

        let src = ReplaySource::new(EventLog::read(original.as_bytes()).unwrap());
        let header = src.header().clone();
        // The grid comes back out of the log and goes straight back in: a re-emitted log
        // that dropped it would replay unrounded, which is the whole hole ADR-0027 closed.
        let mut recapture =
            Capture::with_header(Vec::new(), header, src.instruments().clone()).unwrap();
        src.run(&ManualClock::new(0), &mut recapture);
        let bytes = recapture.finish().unwrap();

        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            original,
            "a round trip through the bus changed the log"
        );
    }

    #[test]
    fn replay_never_manufactures_the_venues_reply_to_an_order() {
        // The limitation, made executable. A handler that "would have traded" during
        // replay receives nothing back: the exec stream is whatever was captured, and
        // a market-data-only log yields market data only. A harness that synthesized a
        // fill here would be reporting a P&L the venue never agreed to.
        let src = source(&[a_trade(10), a_trade(20)]);
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        src.run(&clock, &mut h);
        assert_eq!(h.exec_events(), 0);
        assert_eq!(h.market_events(), 2);
    }

    #[test]
    fn publishing_onto_a_dead_bus_reports_where_it_stopped() {
        let src = source(&[a_trade(10), a_trade(20)]);
        let (tx, rx) = bus(1);
        drop(rx);
        let err = src.publish(&tx).unwrap_err();
        assert!(
            matches!(err, ReplayError::BusClosed { published: 0 }),
            "{err}"
        );
    }

    #[test]
    fn a_bus_narrower_than_the_log_still_delivers_every_event_in_order() {
        // Replay publishes through the same bounded bus a live adapter does, so the
        // producer blocks when the core falls behind. A lockstep pump would never
        // reach that path, and the deadlock it hides would only appear on a log longer
        // than the bus — i.e. on a real capture, never on a unit test.
        let events: Vec<Event> = (1..=200).map(|i| a_trade(i * 10)).collect();
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        let report = source(&events).with_bus_capacity(1).run(&clock, &mut h);
        assert_eq!(report.events, 200);
        assert_eq!(h.event_times().len(), 200);
        assert_eq!(h.event_times().first(), Some(&10));
        assert_eq!(h.event_times().last(), Some(&2_000));
    }

    #[test]
    fn the_committed_session_log_replays_identically_twice() {
        // The same fixture `python/tests/test_backtest.py` drives, guarded here too so
        // a change to the vocabulary or to the format fails in the crate that owns
        // them rather than surfacing as an unexplained Python failure. Regenerate with
        // `cargo run -p axon-replay --example make_fixture_log -- <path>`.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/session.jsonl");
        let src = ReplaySource::open(path).expect("committed fixture must still parse");

        let clock = ManualClock::new(0);
        let mut a = DigestHandler::new(&clock);
        let report = src.run(&clock, &mut a);
        let clock = ManualClock::new(0);
        let mut b = DigestHandler::new(&clock);
        src.run(&clock, &mut b);

        assert_eq!(a.into_bytes(), b.into_bytes());
        assert_eq!(report.events, 59);
        assert_eq!(report.dispatched, 59);
        assert_eq!(
            report.late_arrivals, 1,
            "the fixture carries one deliberate out-of-order arrival"
        );
    }

    #[test]
    fn an_empty_log_replays_to_nothing_without_reporting_a_window() {
        let src = source(&[]);
        let clock = ManualClock::new(0);
        let mut h = DigestHandler::new(&clock);
        let r = src.run(&clock, &mut h);
        assert_eq!(r.events, 0);
        assert_eq!(r.first_ts, None);
        assert_eq!(r.last_ts, None, "an empty window is absent, not zero");
    }
}
