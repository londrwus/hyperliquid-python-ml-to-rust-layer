//! [`Capture`] — the [`EventHandler`] that writes the bus to a log file.
//!
//! It is a handler like any other, so capture is enabled by adding it to the core
//! loop's handler chain, not by threading a "record" flag through the engine. What it
//! writes is what the core *saw*, in the order the core saw it, which is the only
//! sequence a replay could ever reproduce.
//!
//! One deliberate awkwardness: [`EventHandler::on_event`] returns nothing, so a failed
//! write has nowhere to go. It is **latched** and surfaced by [`Capture::finish`]
//! rather than logged and forgotten. A disk that fills mid-session would otherwise
//! leave a short log that replays perfectly and proves nothing — the harness would
//! report success over the events it managed to keep, and the missing tail would look
//! like a session that simply ended.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use axon_core::{Clock, Event, EventHandler, Nanos, SystemClock};

use crate::error::ReplayError;
use crate::instruments::InstrumentSet;
use crate::log::{LogHeader, LogLine, LogRecord, LoggedEvent, OrderingWatch};

/// What a capture session recorded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CaptureStats {
    pub events: u64,
    pub first_ts: Option<Nanos>,
    pub last_ts: Option<Nanos>,
    /// Events that arrived stamped **older** than one already captured.
    ///
    /// Counted at capture time because this is the only place the truth is
    /// observable: after the fact a log cannot say whether it was written out of
    /// order or written in order and later sorted. A non-zero count is the signal
    /// that a replay in event-time order will *not* reproduce this session's
    /// interleaving — see [`ReplayOrder`](crate::ReplayOrder).
    pub late_arrivals: u64,
    /// How many of those [`late_arrivals`](Self::late_arrivals) sat behind a mark set by
    /// a record the **venue did not timestamp** — a forming bar, or a receipt-stamped
    /// ticker — rather than behind a genuinely earlier event.
    ///
    /// A **subset**, not a second category: `late_arrivals` keeps its meaning (*the two
    /// traversals produce different runs from this log*), which is true either way and is
    /// what the golden compares. This explains its size. It does **not** define the feed
    /// metric — see [`venue_stamped_out_of_order`](Self::venue_stamped_out_of_order),
    /// which is measured directly rather than by subtracting this.
    ///
    /// Best-effort, and honest about which way it errs: a bar is blamed only for records
    /// inside its own window, but a receipt stamp has no window and so excuses whatever
    /// follows it. On a real tape that looser half is 65% of the records, which is
    /// precisely why the number to alarm on is measured and not derived from this.
    pub behind_derived_stamps: u64,
    /// Records the **venue itself timestamped**: trades, books, BBOs, and tickers that
    /// carried a venue time. The denominator of the one ordering claim about a venue this
    /// format can support.
    pub venue_stamped: u64,
    /// Of those, how many arrived behind another venue-stamped record — the feed
    /// genuinely delivering out of order.
    ///
    /// Exact for its subject, because the subset is ordered only against itself: a
    /// receipt-stamped ticker or a replayed `userFills` snapshot cannot enter this count
    /// in either role. Measured over a 1 h 44 m testnet soak across 36 deliberate
    /// disconnects, this was **3 in 3 876** while `late_arrivals` was 15.94% of the tape.
    pub venue_stamped_out_of_order: u64,
}

/// Serializes every [`Event`] the core sees into a JSONL log.
pub struct Capture<W: Write> {
    out: W,
    seq: u64,
    /// The ordering rule, shared verbatim with the reader (see [`OrderingWatch`]).
    watch: OrderingWatch,
    /// First write error, kept so [`finish`](Self::finish) can fail loudly.
    error: Option<ReplayError>,
}

impl Capture<BufWriter<File>> {
    /// Create (or truncate) a log file and write its preamble.
    ///
    /// Buffered: one `write` syscall per event would put the core thread into the
    /// kernel on every tick.
    pub fn create(
        path: impl AsRef<Path>,
        source: impl Into<String>,
        instruments: InstrumentSet,
    ) -> Result<Self, ReplayError> {
        Self::new(BufWriter::new(File::create(path)?), source, instruments)
    }
}

impl<W: Write> Capture<W> {
    /// Start a capture, writing the header and the instrument line immediately.
    ///
    /// `instruments` is a **required argument**, for the reason `order_wire`'s
    /// `precision` is (ADR-0025 §2): a writer cannot record a session without naming
    /// what that session knew about the venue's grid. [`InstrumentSet::Undeclared`] is
    /// a word that has to be typed, and a log carrying it says so to every reader —
    /// where a defaulted-empty table would have said "this venue has no rules" about a
    /// venue that has plenty.
    ///
    /// `created_ns` comes from the wall clock, which is correct here and only here:
    /// it is provenance metadata, never an ordering key.
    pub fn new(
        out: W,
        source: impl Into<String>,
        instruments: InstrumentSet,
    ) -> Result<Self, ReplayError> {
        Self::with_header(
            out,
            LogHeader::new(source, SystemClock.now_ns()),
            instruments,
        )
    }

    /// Start a capture with a caller-supplied header — for tests that need a
    /// byte-stable file, and for a replayer re-emitting a log with new provenance.
    ///
    /// The preamble goes out before the first event so that a log truncated by a crash
    /// is still identifiable, still version-checked, and still says which grid its
    /// prefix was planned against; written at close it would be missing from exactly
    /// the logs a post-mortem needs.
    pub fn with_header(
        mut out: W,
        header: LogHeader,
        instruments: InstrumentSet,
    ) -> Result<Self, ReplayError> {
        serde_json::to_writer(&mut out, &header)
            .map_err(|source| ReplayError::Json { line: 1, source })?;
        out.write_all(b"\n")?;
        serde_json::to_writer(&mut out, &LogLine::Instruments(instruments))
            .map_err(|source| ReplayError::Json { line: 2, source })?;
        out.write_all(b"\n")?;
        Ok(Self {
            out,
            seq: 0,
            watch: OrderingWatch::default(),
            error: None,
        })
    }

    /// Append one event. Callers driving the capture directly (rather than through
    /// the event loop) get the error immediately.
    pub fn record(&mut self, event: &Event) -> Result<(), ReplayError> {
        let ts_event = event.ts_event();
        let line = LogLine::Event(LogRecord {
            seq: self.seq,
            ts_event,
            event: LoggedEvent::from(event),
        });
        serde_json::to_writer(&mut self.out, &line).map_err(|source| ReplayError::Json {
            // Two preamble lines precede the first record, so record `n` is line `n + 3`.
            line: self.seq as usize + 3,
            source,
        })?;
        self.out.write_all(b"\n")?;

        self.seq += 1;
        // Counted through the shared watch and not inline, so what the writer says about
        // this log and what a reader later says about the same file cannot disagree.
        self.watch
            .observe(ts_event, crate::log::StampSource::of(event));
        Ok(())
    }

    pub fn stats(&self) -> CaptureStats {
        CaptureStats {
            events: self.watch.events,
            first_ts: self.watch.first_ts,
            last_ts: self.watch.high,
            late_arrivals: self.watch.late_arrivals,
            behind_derived_stamps: self.watch.behind_derived_stamps,
            venue_stamped: self.watch.venue_stamped,
            venue_stamped_out_of_order: self.watch.venue_stamped_out_of_order,
        }
    }

    /// Flush and close, surfacing the first error the handler path swallowed.
    pub fn finish(mut self) -> Result<W, ReplayError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

impl<W: Write> EventHandler for Capture<W> {
    fn on_event(&mut self, _ts_event: Nanos, event: &Event) {
        // Once the sink is broken, stop writing. Continuing would interleave records
        // after a half-written line and turn an honest short log into a corrupt one.
        if self.error.is_some() {
            return;
        }
        if let Err(e) = self.record(event) {
            self.error = Some(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        a_candle, a_fill, a_receipt_stamped_ticker, a_trade, a_venue_stamped_ticker,
        declared_grids, FailingWriter,
    };
    use crate::EventLog;
    use axon_core::{bus, run_blocking};

    /// One millisecond, so a test's numbers read as the timeline they describe.
    const MS: Nanos = 1_000_000;

    #[test]
    fn capture_writes_what_the_bus_delivered_in_the_order_it_delivered_it() {
        // Capture is an EventHandler, so it sees exactly the sequence the core saw —
        // which is the only sequence a replay could reproduce.
        let (tx, rx) = bus(16);
        tx.send(a_trade(30)).unwrap();
        tx.send(a_fill(10)).unwrap();
        tx.send(a_trade(20)).unwrap();
        drop(tx);

        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        run_blocking(&rx, &mut cap);
        let stats = cap.stats();
        let bytes = cap.finish().unwrap();

        assert_eq!(stats.events, 3);
        assert_eq!(stats.first_ts, Some(30));
        assert_eq!(
            stats.late_arrivals, 2,
            "10 and 20 both arrived behind the 30 high-water mark"
        );

        let log = EventLog::read(&bytes[..]).unwrap();
        let ts: Vec<Nanos> = log.records().iter().map(|r| r.ts_event).collect();
        assert_eq!(ts, vec![30, 10, 20], "capture order, not sorted order");
        let seqs: Vec<u64> = log.records().iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2]);
    }

    #[test]
    fn a_write_failure_is_latched_rather_than_producing_a_short_golden_log() {
        // The failure this exists to prevent: the disk fills, the tail of the session
        // is lost, and the replay of the surviving prefix is green. A golden test that
        // silently narrows its own scope is worse than one that fails.
        // The sink accepts the whole preamble and then dies, so the log has a valid
        // header, a valid instrument line and zero records — the shape that would
        // otherwise replay cleanly.
        let mut cap = Capture::new(
            FailingWriter::after_lines(2),
            "unit-test",
            InstrumentSet::Undeclared,
        )
        .unwrap();
        let ev = a_trade(1);
        cap.on_event(ev.ts_event(), &ev);
        cap.on_event(ev.ts_event(), &ev);
        let err = cap.finish().unwrap_err();
        assert!(
            matches!(err, ReplayError::Io(_) | ReplayError::Json { .. }),
            "{err}"
        );
    }

    #[test]
    fn a_late_event_does_not_make_every_later_event_look_late() {
        // `late_arrivals` counts events behind the high-water mark. Comparing against
        // the *previous* event instead would report one inversion per event after a
        // single late arrival, which would read as a broken feed.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        for ts in [10, 20, 5, 30, 40] {
            cap.record(&a_trade(ts)).unwrap();
        }
        assert_eq!(cap.stats().late_arrivals, 1);
        assert_eq!(cap.stats().last_ts, Some(40));
    }

    #[test]
    fn a_forming_bar_is_not_counted_as_a_feed_that_reorders() {
        // The failure this exists to prevent, and it is a *reporting* failure with real
        // consequences: a soak subscribes candles, the venue republishes each forming bar
        // dozens of times with `ts_event` set to the moment it *will* close, that stamp
        // walks the high-water mark minutes into the future, and every ordinary trade for
        // the rest of the window lands behind it. The operator reads thousands of "late
        // arrivals" and goes looking for a network problem that does not exist.
        //
        // A bar opening at 10 000 ms and closing at 10 060 ms, republished while it fills,
        // with real trades inside its window arriving after it — exactly what the wire
        // does.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_trade(10_000 * MS)).unwrap();
        cap.record(&a_candle(10_000, 60)).unwrap(); // stamped 10 060 ms: the close
        for ts in [10_010, 10_020, 10_030] {
            cap.record(&a_trade(ts * MS)).unwrap(); // inside the bar, captured after it
        }
        cap.record(&a_candle(10_000, 60)).unwrap(); // the same bar again, still forming

        let stats = cap.stats();
        assert_eq!(
            stats.late_arrivals, 3,
            "the traversals do differ, and by three"
        );
        assert_eq!(
            stats.behind_derived_stamps, 3,
            "…and all three are the bar's stamp, not the network"
        );
        // The measurement, which is what an operator alarms on: the venue-stamped trades
        // were in perfect order among themselves, and the bar never entered the subset.
        assert_eq!(stats.venue_stamped, 4);
        assert_eq!(stats.venue_stamped_out_of_order, 0);
    }

    #[test]
    fn a_ticker_the_venue_never_stamped_is_not_evidence_about_the_venues_ordering() {
        // The artefact a 1 h 44 m soak found, and the one that made the previous
        // subtraction useless: `activeAssetCtx` carries no venue timestamp, so a `Ticker`
        // is ordered on our own receipt clock (ADR-0011) and **always** advances the
        // high-water mark past venue-stamped data by one network hop. It was 65% of that
        // tape, so every venue-stamped frame behind one counted as "reordered" and the
        // residual reported a network problem that did not exist.
        //
        // Here: a receipt-stamped ticker lands between trades whose venue times are in
        // perfect order.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_trade(10_000 * MS)).unwrap();
        cap.record(&a_receipt_stamped_ticker(10_500 * MS)).unwrap();
        cap.record(&a_trade(10_100 * MS)).unwrap();
        cap.record(&a_trade(10_200 * MS)).unwrap();

        let stats = cap.stats();
        assert_eq!(stats.late_arrivals, 2, "the traversals differ, and they do");
        assert_eq!(
            stats.behind_derived_stamps, 2,
            "…because of the receipt clock"
        );
        assert_eq!(
            stats.venue_stamped, 3,
            "the ticker is not testimony about the venue's ordering"
        );
        assert_eq!(
            stats.venue_stamped_out_of_order, 0,
            "the venue delivered its own stamps in order, which is the finding"
        );
    }

    #[test]
    fn a_ticker_the_venue_did_stamp_stays_inside_the_measurement() {
        // The exclusion is about the missing stamp, not about tickers. A venue that sends
        // a timestamp on this frame moves it into the measured subset with nothing here
        // changing — and if such a frame really does arrive out of order, it counts.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_trade(10_000 * MS)).unwrap();
        cap.record(&a_venue_stamped_ticker(10_500 * MS)).unwrap();
        cap.record(&a_venue_stamped_ticker(10_200 * MS)).unwrap();

        let stats = cap.stats();
        assert_eq!(stats.venue_stamped, 3);
        assert_eq!(
            stats.venue_stamped_out_of_order, 1,
            "a venue-stamped frame behind another one is the real thing"
        );
    }

    #[test]
    fn a_replayed_execution_snapshot_is_not_the_feed_delivering_late() {
        // The other artefact the soak found: `userFills` replays its **whole snapshot** on
        // every reconnect, and the tape carried execution times up to 2.94 hours stale.
        // Those are genuinely old records genuinely arriving now — and counting them as
        // the feed reordering would make every reconnect look like a fault, on a session
        // whose entire purpose was to reconnect 36 times.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_trade(10_000 * MS)).unwrap();
        for stale in [1_000, 1_100, 1_200] {
            cap.record(&a_fill(stale * MS)).unwrap(); // the snapshot, replayed
        }
        cap.record(&a_trade(10_100 * MS)).unwrap();

        let stats = cap.stats();
        assert_eq!(stats.late_arrivals, 3, "the traversals differ, and they do");
        assert_eq!(
            stats.venue_stamped, 2,
            "the execution stream is not market data"
        );
        assert_eq!(
            stats.venue_stamped_out_of_order, 0,
            "a snapshot being re-sent is not a feed delivering out of order"
        );
    }

    #[test]
    fn a_bar_is_blamed_only_for_the_records_inside_its_own_window() {
        // The bound on the excuse. Once a forming bar holds the high-water mark it would
        // be easy to write off *everything* behind it, and that would hide a genuine
        // reordering behind whichever candle happened to be holding the mark. A record
        // older than the whole window is behind the mark for a reason the bar cannot
        // explain, so it stays a late arrival.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_trade(10_000 * MS)).unwrap();
        cap.record(&a_candle(10_000, 60)).unwrap();
        cap.record(&a_trade(10_030 * MS)).unwrap(); // inside the window: the bar's doing
        cap.record(&a_trade(9_000 * MS)).unwrap(); // a second older than the bar opened

        let stats = cap.stats();
        assert_eq!(stats.late_arrivals, 2);
        assert_eq!(
            stats.behind_derived_stamps, 1,
            "a record predating the whole window is not the bar's doing"
        );
    }

    #[test]
    fn a_closed_bar_shadows_nothing() {
        // A bar published at its close is stamped honestly: everything after it is after
        // it. Nothing here should be attributed to a bar at all, or the diagnostic would
        // excuse inversions on any feed that merely carries candles.
        let mut cap = Capture::new(Vec::new(), "unit-test", InstrumentSet::Undeclared).unwrap();
        cap.record(&a_candle(10_000, 60)).unwrap();
        for ts in [10_070, 10_080] {
            cap.record(&a_trade(ts * MS)).unwrap();
        }
        let stats = cap.stats();
        assert_eq!(stats.late_arrivals, 0);
        assert_eq!(stats.behind_derived_stamps, 0);
    }

    #[test]
    fn a_capture_to_a_file_is_flushed_before_anyone_can_read_it() {
        // `finish` flushes the BufWriter. Without that the tail of a session sits in
        // userspace and the log a replay opens is short by however much the buffer
        // held — a truncation with no error anywhere.
        let path = std::env::temp_dir().join(format!(
            "axon-replay-capture-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut cap = Capture::create(&path, "unit-test", declared_grids()).unwrap();
        for ts in [10, 20, 30] {
            cap.record(&a_trade(ts)).unwrap();
        }
        cap.finish().unwrap();

        let log = EventLog::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(log.len(), 3);
        assert_eq!(log.header().source, "unit-test");
        assert!(
            log.instruments().is_declared(),
            "the grid the writer was handed has to be in the file it wrote"
        );
    }

    #[test]
    fn a_capture_records_the_grid_it_was_handed_before_the_first_event() {
        // The ordering that makes the table usable: a reader has to know the grid before
        // it interprets a single record, because the grid is what the *planner* it drives
        // rounds against. Written after the events — or at close — it would be missing
        // from every log a crash produced, which is exactly the set a post-mortem reads.
        let mut cap = Capture::new(Vec::new(), "unit-test", declared_grids()).unwrap();
        cap.record(&a_trade(10)).unwrap();
        let text = String::from_utf8(cap.finish().unwrap()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with(r#"{"schema":"axon.eventlog""#));
        assert!(
            lines[1].starts_with(r#"{"Instruments":{"Declared""#),
            "{}",
            lines[1]
        );
        assert!(lines[2].starts_with(r#"{"Event":"#));
    }
}
