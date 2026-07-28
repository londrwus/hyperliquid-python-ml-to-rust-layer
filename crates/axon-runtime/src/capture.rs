//! Recording a live session, so that a replay of it is a replay of *this* session.
//!
//! [`axon_replay::Capture`] has been an [`EventHandler`](axon_core::EventHandler)
//! since ADR-0018 and until now nothing but a test ever installed one. That left the
//! bottom rung of `docs/07-parity-and-testing.md` resting on a synthetic fixture: the
//! golden harness had replayed a log written by a generator, never a log written by a
//! session. This module is the missing installation.
//!
//! A session records **two** files, because a replay needs both halves of what the
//! session saw:
//!
//! - `<path>` — the event log: every [`Event`] the core was handed, in the order it
//!   was handed them.
//! - `<path-with-.signals.jsonl>` — the signal log: every record the strategy adapter
//!   took off the ring, stamped with the event time at which it became readable.
//!   Without it a replay can re-observe a session but cannot re-decide it, and the
//!   `expired` and `gaps` counters a golden compares would all replay as zero.
//!
//! Four decisions here, and each has an answer that reads as obviously correct until
//! it costs the thing the recording exists for.
//!
//! **1. The core thread hands the record over; it never writes it.** `on_event` does a
//! `try_send` onto a bounded [`crossbeam_channel`] and a dedicated writer thread does
//! the serialization and the `write(2)`. The alternative — ADR-0018's own shape, a
//! `BufWriter` on the core thread — is amortized cheap and is *not* bounded: a
//! `write(2)` behind a full page cache, a foreign `fsync`, or a filesystem that has
//! gone away blocks for as long as it blocks, and it blocks the one thread that must
//! keep draining market data. A stall in the deterministic loop is a stale book, and a
//! stale book is worse than a missing recording every single time. Ordering survives
//! the move because both taps live on the core thread and the channel is FIFO, so the
//! writer observes exactly the sequence the core observed — the only sequence a replay
//! could reproduce.
//!
//! **2. A recording that cannot keep up stops; it does not drop.** This is the
//! opposite of the market-data ring's rule ([`crate::mdring`]) and the difference is
//! what each artifact is *for*. A dropped `MdSlice` costs one frame of a stream whose
//! every record is full state, so the newest supersedes the old. A dropped `Event`
//! costs the log its meaning: the replay of a log with a hole in it runs a session that
//! never happened, and it runs it *successfully*, which is the worst outcome available
//! here. A prefix is a truthful recording of a shorter session; a gap is a lie. So the
//! first refused hand-off latches [`CaptureStop::QueueFull`] and the recording ends
//! there. Everything lost from that moment — the refused hand-off, whatever was still
//! in the queue behind it, and every event after — lands in
//! [`CaptureOutcome::missed`], so `events + missed` is the whole session and the size of
//! the hole is a number rather than something inferred from a file size.
//!
//! **3. A log is only given its real name once it has been closed cleanly.** The writer
//! creates `<path>.partial` and renames it into place when — and only when —
//! [`SessionRecorder::finish`] says the session ran to its end. A capture stopped by a
//! full disk at 02:00 therefore does not leave a file at the path a harness would pick
//! up: a truncated log that parses is the one artifact that fails silently, because the
//! replay of it is green and narrower than it claims. A process killed outright leaves
//! the `.partial` too.
//!
//! "No stop was latched" is emphatically **not** the test. [`Drop`] runs on every
//! abnormal exit that unwinds or `?`-returns through the frame owning the recorder — a
//! core thread that panicked, a handler that failed to build, a thread that failed to
//! spawn — and in all of those the recording was perfectly healthy right up to the
//! moment the session was cut in half. So the rename is earned by an explicit
//! [`Rec::Finish`] and by nothing else; `Drop` sends [`Rec::Close`], which flushes and
//! stops the writer thread but leaves both files under their temporary names.
//!
//! The same reasoning applies one run earlier: [`SessionRecorder::start`] **removes**
//! whatever is already sitting at `<path>`. Once a recording has committed to owning
//! that name, a file left there by a previous session is exactly the artifact the
//! rename exists to prevent — Monday's complete log, at the path Tuesday's harness
//! reaches for, carrying a `source` string derived only from the config and therefore
//! identical between the two runs.
//!
//! **4. There is a size cap, it is a *disk* guard now, and there is still no rotation.**
//! ADR-0018 set the cap where the artifact stopped being **loadable**, because
//! `EventLog` held a log whole — which meant a multi-hour soak stopped its own recording
//! at 512 MiB. A log is streamed now ([`axon_replay::LogReader`], ADR-0027), so the cap
//! answers a different question: how much of this volume may a recording take from the
//! trading session that shares it. `max_bytes == 0` turns it off, for an operator who
//! has decided that answer is "all of it".
//!
//! Rotation is still out of scope, and for the reason ADR-0018 gave rather than the one
//! it happened to be attached to: event-time ordering cannot begin until the last record
//! has been seen, so a set of rotated segments is a set of files each of which replays a
//! *different* session from the one that was recorded, and stitching them is a merge
//! nothing in the harness performs. Reaching the cap still stops the capture loudly —
//! [`CaptureStop::SizeCap`] on the status line — instead of filling the volume.
//!
//! **5. A recording declares the instrument grid its session planned against.** A log
//! carried none until ADR-0027, so a replay of it planned under
//! `Precision::Unconstrained` while the session planned under `Precision::Known` and
//! *rounded* — and `PlannedOrder::price` is compared exactly by the golden harness, so
//! every rounded order in a real capture came back as a strategy flip the strategy never
//! made. [`SessionRecorder::start_with`] takes the session's own table;
//! [`SessionRecorder::start`] records [`InstrumentSet::Undeclared`] and **says so at
//! startup**, because a recording that cannot be replayed at the right prices is worth
//! knowing about while an operator is watching rather than at the first golden diff.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use axon_contracts::Signal;
use axon_core::{Clock, Event, Nanos, SystemClock};
use axon_providers::InstrumentTable;
use axon_replay::{
    Capture, InstrumentSet, LogHeader, ReplayError, SignalLog, SignalRecord, SIGNAL_SCHEMA,
    SIGNAL_SCHEMA_VERSION,
};
use axon_strategy::SignalSource;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

use crate::config::CaptureConfig;
use crate::intent::Attachable;

/// Why a recording stopped before the session did.
///
/// Every variant is a reason the log on disk is a **prefix** of the session, which is
/// the one thing an operator has to know about a capture and the one thing a truncated
/// file cannot say for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStop {
    /// The writer thread fell far enough behind that the hand-off queue filled.
    ///
    /// The core kept running — that is what the queue is for — but the records it could
    /// not hand over are gone, and a log with a hole is not replayable at all.
    QueueFull,
    /// A write failed: the volume filled, the mount went away, permissions changed.
    WriteFailed,
    /// The configured [`CaptureConfig::max_bytes`] was reached.
    SizeCap,
}

impl CaptureStop {
    /// The operator-facing name, short enough for the status line.
    pub fn as_str(self) -> &'static str {
        match self {
            CaptureStop::QueueFull => "queue full",
            CaptureStop::WriteFailed => "write failed",
            CaptureStop::SizeCap => "size cap",
        }
    }

    fn code(self) -> u8 {
        match self {
            CaptureStop::QueueFull => 1,
            CaptureStop::WriteFailed => 2,
            CaptureStop::SizeCap => 3,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(CaptureStop::QueueFull),
            2 => Some(CaptureStop::WriteFailed),
            3 => Some(CaptureStop::SizeCap),
            _ => None,
        }
    }
}

impl std::fmt::Display for CaptureStop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a finished recording left behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureOutcome {
    /// Events actually written.
    pub events: u64,
    /// Signal records actually written.
    pub signals: u64,
    /// Bytes across both files.
    pub bytes: u64,
    /// Events captured behind the event-time high-water mark. Non-zero means the log
    /// must be replayed `as-captured` to reproduce this session's interleaving
    /// (ADR-0018 §4).
    pub late_arrivals: u64,
    /// How many of those [`late_arrivals`](Self::late_arrivals) sat behind a record the
    /// **venue did not timestamp** — a forming bar, or a receipt-stamped ticker — rather
    /// than behind a genuinely earlier event.
    ///
    /// A subset, so the number above keeps its meaning. It explains that number's size;
    /// it is not what an operator should alarm on. See
    /// [`venue_stamped_out_of_order`](Self::venue_stamped_out_of_order) (ADR-0027 §8).
    pub behind_derived_stamps: u64,
    /// Records the **venue itself timestamped** during this session.
    pub venue_stamped: u64,
    /// Of those, how many arrived behind another venue-stamped record.
    ///
    /// **This is the soak's real ordering finding**, and it is on the outcome rather than
    /// only in a replay because an operator reads it without replaying anything. A 1 h
    /// 44 m testnet soak across 36 deliberate disconnects: `late_arrivals` 15.94% of the
    /// tape, this **3 in 3 876**. Reporting the first as a feed fault is the mistake this
    /// pair exists to prevent.
    pub venue_stamped_out_of_order: u64,
    /// Records the core could not hand over, counted from the moment the recording
    /// stopped. The size of the hole, so nobody has to estimate it from a file size.
    pub missed: u64,
    /// `None` when the recording ran to the end of the session.
    pub stopped: Option<CaptureStop>,
    /// Where the log ended up. Still `<path>.partial` when the recording stopped,
    /// because a truncated log must not be sitting at the path a harness reaches for.
    pub log_path: PathBuf,
    /// The signal log, when one was recorded.
    pub signal_path: Option<PathBuf>,
}

impl CaptureOutcome {
    /// Whether the log at [`log_path`](Self::log_path) is the whole session.
    pub fn complete(&self) -> bool {
        self.stopped.is_none()
    }
}

/// Counters the core thread writes and the status line reads.
///
/// Atomics rather than a lock because the reader is the status line: a recording must
/// not be able to make the core thread wait on anything, and that includes waiting on
/// its own bookkeeping.
#[derive(Debug, Default)]
struct RecorderState {
    events: AtomicU64,
    signals: AtomicU64,
    /// Shared with the [`Counted`] writers rather than mirrored from them: one counter
    /// updated where the bytes are actually produced, so the cap the writer enforces and
    /// the size the status line prints can never be two different numbers.
    bytes: Arc<AtomicU64>,
    missed: AtomicU64,
    /// `0` while running, otherwise a [`CaptureStop`] code. Latched once, by whichever
    /// side notices first: the core thread on a refused hand-off, the writer thread on
    /// a failed write.
    stopped: AtomicU8,
    /// The hand-off queue's depth, so a progress reading carries the denominator its
    /// `queued` number is only meaningful against.
    capacity: u64,
}

impl RecorderState {
    fn progress(&self, queued: u64) -> CaptureProgress {
        CaptureProgress {
            events: self.events.load(Ordering::Relaxed),
            signals: self.signals.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            missed: self.missed.load(Ordering::Relaxed),
            queued,
            capacity: self.capacity,
            stopped: self.stopped(),
        }
    }

    /// Returns whether *this* call latched the stop, so the reason is printed once
    /// rather than once per event for the rest of the session.
    fn stop(&self, reason: CaptureStop) -> bool {
        self.stopped
            .compare_exchange(0, reason.code(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn stopped(&self) -> Option<CaptureStop> {
        CaptureStop::from_code(self.stopped.load(Ordering::Acquire))
    }
}

/// One item on the hand-off queue.
///
/// Both event and signal records ride the *same* channel and are written by the same
/// thread. One queue rather than two because there is one disk, one place a write can
/// stall and one recording to declare stopped — two queues would let the two halves of
/// a session end at different points and still both look complete.
enum Rec {
    /// Held inline rather than boxed. A `Box` would shrink the queue's slots and cost a
    /// *second* allocation per event on the core thread; the clone is already the one
    /// unavoidable allocation there, and doubling it to save memory in a buffer sized in
    /// thousands is the wrong trade on this thread.
    Event(Event),
    Signal(SignalRecord),
    /// Stop the writer and flush, but leave both files under their temporary names.
    ///
    /// Sent by [`Drop`], which is every abnormal end: a panic unwinding through the
    /// session, a `?` on some other error. An explicit message rather than relying on
    /// every tap being dropped first — the taps are owned by the handler and the intent
    /// source, and making the shutdown depend on their drop order is how a join hangs.
    Close,
    /// Stop the writer, flush, **and** rename both logs into place.
    ///
    /// Sent by [`SessionRecorder::finish`] and by nothing else, because it is the only
    /// caller that knows the session reached its end. The rename is a claim about the
    /// whole file and there is no way to infer it from inside the writer: a recording
    /// killed at minute 30 of a 60-minute session has latched no stop and is healthy in
    /// every respect except the one that matters.
    Finish,
}

/// The core thread's end of the recording.
///
/// Cheap to clone and safe to hold on the deterministic path: every hand-off is a
/// non-blocking `try_send` plus a relaxed counter, and there is no branch on it that
/// can wait.
#[derive(Debug, Clone)]
pub struct CaptureTap {
    tx: Sender<Rec>,
    state: Arc<RecorderState>,
}

impl CaptureTap {
    /// The recording as it stands.
    ///
    /// Read through the tap rather than the recorder because the status line is
    /// assembled on the core thread, which holds a tap and not the recorder — the
    /// recorder is the thing that gets to end the session, and handing that to the loop
    /// would be handing it a way to join a thread mid-drain.
    pub fn progress(&self) -> CaptureProgress {
        self.state.progress(self.tx.len() as u64)
    }

    /// Hand one event to the writer. Never blocks and never fails the caller.
    pub fn on_event(&self, event: &Event) {
        // One clone, and it is the price of writing off-thread: the bus hands the core an
        // `&Event` and another thread needs to own it. It is also the *cheaper* half of
        // what ADR-0018 already accepted — serializing this event to JSON on the core
        // thread allocates too, and costs microseconds of CPU on top. A session that is
        // not recording pays nothing at all, because `CoreHandler` holds no tap.
        self.send(Rec::Event(event.clone()));
    }

    /// Hand one signal record over, stamped with the event time at which the reader
    /// could first have seen it.
    pub fn on_signal(&self, release_ts: Nanos, signal: &Signal) {
        self.send(Rec::Signal(SignalRecord::released_at(release_ts, signal)));
    }

    fn send(&self, rec: Rec) {
        // Already stopped: count what is being lost and get out. Continuing to send
        // would be writing records after a gap, and a log whose records are not
        // contiguous replays a session that never happened.
        if self.state.stopped().is_some() {
            self.state.missed.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let reason = match self.tx.try_send(rec) {
            Ok(()) => return,
            // The writer is behind. See the module docs, point 2: this ends the
            // recording rather than punching a hole in it.
            Err(TrySendError::Full(_)) => CaptureStop::QueueFull,
            // The writer thread has gone — it latched its own stop, or it panicked.
            Err(TrySendError::Disconnected(_)) => CaptureStop::WriteFailed,
        };
        self.state.missed.fetch_add(1, Ordering::Relaxed);
        if self.state.stop(reason) {
            // Allocates, once per session rather than once per event.
            eprintln!(
                "capture: recording stopped ({reason}) - the log on disk is a prefix of \
                 this session and will not be renamed into place"
            );
        }
    }
}

/// A [`SignalSource`] that copies every record it hands over into the recording.
///
/// The tee has to be here, at the source, and not in [`IntentSource`]'s drain
/// callback: the callback only sees the records the reader *accepted*, and a log
/// missing the expired and out-of-order ones would replay with every refusal counter at
/// zero — which is precisely the half of the strategy's behaviour a golden run exists
/// to pin. A record is recorded when it is read, whatever the reader then decides
/// about it.
///
/// [`IntentSource`]: crate::intent::IntentSource
#[derive(Debug)]
pub struct RecordingSource<S> {
    inner: S,
    /// `None` for a session that is not recording, which makes this wrapper free and
    /// lets the live wiring have exactly one shape.
    tap: Option<CaptureTap>,
    /// Event-time high-water mark, used as the record's `release_ts`.
    ///
    /// A high-water mark and not simply the last observed time: `CoreHandler::last_ts`
    /// follows the *event*, and a late arrival moves it backwards. Recording that
    /// backwards step would produce a log `SignalLog::read` refuses outright, because a
    /// ring hands records out in write order and a log claiming otherwise describes a
    /// session that could not have happened.
    release_ts: Nanos,
}

impl<S> RecordingSource<S> {
    pub fn new(inner: S, tap: Option<CaptureTap>) -> Self {
        Self {
            inner,
            tap,
            release_ts: Nanos::MIN,
        }
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S: SignalSource> SignalSource for RecordingSource<S> {
    #[inline]
    fn next_signal(&mut self) -> Option<Signal> {
        let sig = self.inner.next_signal()?;
        if let Some(tap) = &self.tap {
            tap.on_signal(self.release_ts, &sig);
        }
        Some(sig)
    }
}

impl<S: Attachable> Attachable for RecordingSource<S> {
    fn ensure(&mut self, now_ms: u64) -> bool {
        self.inner.ensure(now_ms)
    }

    fn is_attached(&self) -> bool {
        self.inner.is_attached()
    }

    fn attach_failures(&self) -> u64 {
        self.inner.attach_failures()
    }

    fn observe_event_time(&mut self, now: Nanos) {
        self.release_ts = self.release_ts.max(now);
        self.inner.observe_event_time(now);
    }
}

/// The read-back end of a recorded signal log: a [`SignalSource`] that releases each
/// record only once the core's event clock has reached its `release_ts`.
///
/// It lives next to the recorder on purpose. The pair is one claim — *what this writes,
/// that reads back identically* — and splitting it across crates is how the two ends
/// come to disagree about what `release_ts` means. Handing every record over on the
/// first pass instead would collapse a session's worth of decisions into one moment and
/// make an expiry impossible to reproduce, which is the reader's most important refusal.
#[derive(Debug, Clone)]
pub struct CapturedSignals {
    records: Vec<SignalRecord>,
    next: usize,
    now: Nanos,
}

impl CapturedSignals {
    pub fn new(records: Vec<SignalRecord>) -> Self {
        Self {
            records,
            next: 0,
            // Nothing is readable until the core has seen an event. Starting at zero
            // would release every record stamped at or before the epoch on the first
            // pass, before the market moment that produced it.
            now: Nanos::MIN,
        }
    }

    pub fn from_log(log: &SignalLog) -> Self {
        Self::new(log.records().to_vec())
    }

    /// Records not yet handed over.
    pub fn remaining(&self) -> usize {
        self.records.len() - self.next
    }

    /// How many records the log carried, handed over or not.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Whether the next record is readable at the clock this source has been told about.
    ///
    /// Exposed because a harness pacing its passes needs to answer "is a pass worth
    /// running?" without taking the record — and it must answer it with *this* rule.
    /// A caller that compared `release_ts` to a clock of its own would be a second
    /// copy of the release schedule, which is exactly how the two ends of a replay
    /// come to disagree about which signals a session saw.
    pub fn due(&self) -> bool {
        self.records
            .get(self.next)
            .is_some_and(|r| r.release_ts <= self.now)
    }
}

impl SignalSource for CapturedSignals {
    fn next_signal(&mut self) -> Option<Signal> {
        let rec = self.records.get(self.next)?;
        if rec.release_ts > self.now {
            return None;
        }
        self.next += 1;
        Some(Signal::from(rec.signal))
    }
}

impl Attachable for CapturedSignals {
    /// A recorded log is always there — it *is* the records. Re-attaching is the
    /// memory-mapped ring's problem and a replay has no such state to recover from.
    fn ensure(&mut self, _now_ms: u64) -> bool {
        true
    }

    fn observe_event_time(&mut self, now: Nanos) {
        // The same high-water clamp as the writer ([`RecordingSource::observe_event_time`]),
        // and for the same reason: `CoreHandler::last_ts` follows the *event*, so a late
        // arrival moves it backwards. A plain assignment rewinds this clock below a
        // `release_ts` the live session had already passed, and because `next` never
        // rewinds, that record and every record behind it stay blocked for good. Under
        // `ReplayOrder::AsCaptured` — the one order ADR-0018 §4 allows a live-captured
        // reference to be compared under — the replay then plans nothing where the live
        // session placed an order, or releases it k events late and rejects it as
        // `Expired`. One release rule, written once at each end, meaning the same thing.
        self.now = self.now.max(now);
    }
}

/// A writer that keeps a running byte count where the recorder can read it.
///
/// Wrapped *outside* the `BufWriter`, so the count is the size of the JSONL produced
/// rather than the size flushed so far — the cap has to be decided before a record is
/// written, not after the buffer happens to drain.
struct Counted<W> {
    inner: W,
    bytes: Arc<AtomicU64>,
}

impl<W: Write> Write for Counted<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// A file being written under a temporary name, plus the name it earns by being closed
/// cleanly. `None` for the in-memory sinks the tests drive.
struct Naming {
    tmp: PathBuf,
    final_path: PathBuf,
}

/// The writer thread's half: the two logs, their names, and the shared byte count.
struct Sink {
    events: Capture<Counted<Box<dyn Write + Send>>>,
    signals: Option<Counted<Box<dyn Write + Send>>>,
    event_naming: Option<Naming>,
    signal_naming: Option<Naming>,
    bytes: Arc<AtomicU64>,
}

impl Sink {
    fn record_signal(&mut self, rec: &SignalRecord) -> Result<(), ReplayError> {
        let Some(out) = self.signals.as_mut() else {
            return Ok(());
        };
        // Serialized here rather than through `SignalLog::write`, which takes the whole
        // slice: a session's records have to reach the disk as they happen, because a
        // log assembled at close is exactly the log a crash loses.
        let line =
            serde_json::to_string(rec).map_err(|e| ReplayError::Io(std::io::Error::other(e)))?;
        writeln!(out, "{line}")?;
        Ok(())
    }

    /// Flush both files and, only if the recording ran to the end, rename them into
    /// place. See the module docs, point 3.
    fn close(self, complete: bool) -> std::io::Result<()> {
        let Sink {
            events,
            signals,
            event_naming,
            signal_naming,
            bytes: _,
        } = self;
        // `finish` flushes; without it the tail of the session sits in userspace and the
        // log a replay opens is short by however much the buffer held.
        let mut first_err = events.finish().err().map(io_of);
        if let Some(mut out) = signals {
            if let Err(e) = out.flush() {
                first_err.get_or_insert(e);
            }
        }
        // The rename is a claim — *this file is all of it* — so a close that went wrong
        // has to withhold it just as a stop does. A final flush that failed is a missing
        // tail, and a log with a missing tail sitting at the path a harness reaches for
        // is the exact artifact this whole scheme exists to make impossible.
        if !complete || first_err.is_some() {
            return match first_err {
                Some(e) => Err(e),
                None => Ok(()),
            };
        }
        for naming in [event_naming, signal_naming].into_iter().flatten() {
            if let Err(e) = std::fs::rename(&naming.tmp, &naming.final_path) {
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn io_of(e: ReplayError) -> std::io::Error {
    match e {
        ReplayError::Io(e) => e,
        other => std::io::Error::other(other.to_string()),
    }
}

/// What the writer thread reports back when it is joined.
struct WriterEnd {
    late_arrivals: u64,
    behind_derived_stamps: u64,
    venue_stamped: u64,
    venue_stamped_out_of_order: u64,
    /// A rename or a final flush that failed. It arrives after the last event, so it
    /// cannot stop the recording — but it can mean the file is not where it says it is.
    close_error: Option<String>,
}

/// A running recording: the writer thread, the queue into it, and the counters both
/// ends share.
pub struct SessionRecorder {
    tx: Sender<Rec>,
    state: Arc<RecorderState>,
    writer: Option<JoinHandle<WriterEnd>>,
    log_path: PathBuf,
    signal_path: Option<PathBuf>,
}

impl SessionRecorder {
    /// Start recording, or `None` if the config does not ask for one.
    ///
    /// A failure here is **fatal to the session**, deliberately, and it is the opposite
    /// of what a mid-session failure does. Nothing has been lost yet and the operator is
    /// still watching: a bad path or a missing directory is a two-second fix now and an
    /// hour of soak with no artifact at the end of it otherwise. Mid-session the
    /// calculation inverts, because by then there is a position on the book and killing
    /// the process over a log file trades a recording for an unmanaged position.
    pub fn start(
        cfg: &CaptureConfig,
        source: impl Into<String>,
    ) -> Result<Option<Self>, ReplayError> {
        Self::start_inner(cfg, source, InstrumentSet::Undeclared)
    }

    /// Start recording, declaring the grid this session plans against.
    ///
    /// The one that closes ADR-0027's half: a replay of the resulting log rounds the way
    /// this session rounds, so a golden diff of it reports strategy changes and nothing
    /// else. Hand it the *same* `Arc<InstrumentTable>` the `IntentSource` and the
    /// `ExchangeClient` were built from — two tables that can drift apart would have the
    /// recording claim a grid the session never planned on, which is worse than claiming
    /// none.
    pub fn start_with(
        cfg: &CaptureConfig,
        source: impl Into<String>,
        instruments: &InstrumentTable,
    ) -> Result<Option<Self>, ReplayError> {
        Self::start_inner(cfg, source, InstrumentSet::of(instruments))
    }

    fn start_inner(
        cfg: &CaptureConfig,
        source: impl Into<String>,
        instruments: InstrumentSet,
    ) -> Result<Option<Self>, ReplayError> {
        if !cfg.enabled {
            return Ok(None);
        }
        let log_final = PathBuf::from(&cfg.path);
        if let Some(dir) = log_final.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let log_tmp = partial(&log_final);
        let signal_final = cfg.signals.then(|| signal_log_path(&log_final));
        let source = source.into();
        let created_ns = SystemClock.now_ns();
        let bytes = Arc::new(AtomicU64::new(0));

        let declared = instruments.is_declared();
        let grid_count = instruments.len();
        let events = Capture::with_header(
            Counted {
                inner: file(&log_tmp)?,
                bytes: bytes.clone(),
            },
            LogHeader::new(source.clone(), created_ns),
            instruments,
        )?;

        let mut signal_naming = None;
        let signals = match &signal_final {
            Some(final_path) => {
                let tmp = partial(final_path);
                let mut out = Counted {
                    inner: file(&tmp)?,
                    bytes: bytes.clone(),
                };
                // The header goes out before the first record for the same reason the
                // event log's does: a file truncated by a crash is still identifiable
                // and still version-checked.
                SignalLog::write(
                    &mut out,
                    &LogHeader::for_schema(
                        SIGNAL_SCHEMA,
                        SIGNAL_SCHEMA_VERSION,
                        source.clone(),
                        created_ns,
                    ),
                    &[],
                )?;
                signal_naming = Some(Naming {
                    tmp,
                    final_path: final_path.clone(),
                });
                Some(out)
            }
            None => None,
        };

        // Both partials exist, so the recording has committed to owning these names:
        // anything already at them belongs to a previous session and must go now. Left
        // in place it is the artifact the rename scheme exists to prevent, one run
        // later — last night's *complete* log sitting where a harness expects tonight's,
        // parsing cleanly, with a `source` string derived only from the config and
        // therefore identical between the two. Removed here rather than at close,
        // because the run that needs it removed is the one that never reaches a close.
        for stale in [Some(&log_final), signal_final.as_ref()]
            .into_iter()
            .flatten()
        {
            remove_stale(stale)?;
        }

        let sink = Sink {
            events,
            signals,
            event_naming: Some(Naming {
                tmp: log_tmp,
                final_path: log_final.clone(),
            }),
            signal_naming,
            bytes,
        };
        println!(
            "capture: recording to {} (signals {}, cap {}, grid {})",
            log_final.display(),
            signal_final
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "off".to_string()),
            if cfg.max_bytes == 0 {
                "off".to_string()
            } else {
                format!("{} MiB", cfg.max_bytes / (1024 * 1024))
            },
            // Named at startup, not at the first golden diff. A recording with no grid
            // replays every rounded price unconstrained, so it reproduces a *more
            // permissive* program than the session it recorded (ADR-0027) — and the one
            // moment that is cheap to fix is the moment an operator is still watching.
            if declared {
                format!("{grid_count} instrument(s)")
            } else {
                "UNDECLARED - a replay of this log will not round the way this session \
                 does"
                    .to_string()
            },
        );
        Ok(Some(Self::spawn(
            sink,
            cfg.queue_capacity,
            cfg.max_bytes,
            log_final,
            signal_final,
        )))
    }

    fn spawn(
        sink: Sink,
        queue_capacity: usize,
        max_bytes: u64,
        log_path: PathBuf,
        signal_path: Option<PathBuf>,
    ) -> Self {
        let queue_capacity = queue_capacity.max(1);
        let (tx, rx) = bounded(queue_capacity);
        let state = Arc::new(RecorderState {
            capacity: queue_capacity as u64,
            bytes: sink.bytes.clone(),
            ..RecorderState::default()
        });
        let writer = {
            let state = state.clone();
            std::thread::Builder::new()
                .name("axon-capture".into())
                .spawn(move || write_loop(rx, sink, state, max_bytes))
                .expect("the capture writer thread must start")
        };
        Self {
            tx,
            state,
            writer: Some(writer),
            log_path,
            signal_path,
        }
    }

    /// A handle for the core thread. Cloneable: the fan-out and the intent source each
    /// hold one, and both run on the same thread.
    pub fn tap(&self) -> CaptureTap {
        CaptureTap {
            tx: self.tx.clone(),
            state: self.state.clone(),
        }
    }

    /// The recording as it stands.
    pub fn stats(&self) -> CaptureProgress {
        self.state.progress(self.tx.len() as u64)
    }

    /// Stop the writer, wait for it to drain, and report what is on disk.
    ///
    /// Blocking, and only ever called once the core loop has returned — the queue may
    /// hold a session's last few events, and exiting before they are written would
    /// truncate the log at exactly the moment a post-mortem cares about.
    pub fn finish(mut self) -> CaptureOutcome {
        // A blocking send, not a `try_send`: this is the one message that must not be
        // dropped, and by now nothing is producing behind it. `Finish` rather than
        // `Close`, and only from here — it is what earns the rename, and this is the
        // only place that knows the session ran to its end.
        let _ = self.tx.send(Rec::Finish);
        let end = self
            .writer
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or(WriterEnd {
                late_arrivals: 0,
                behind_derived_stamps: 0,
                venue_stamped: 0,
                venue_stamped_out_of_order: 0,
                close_error: Some("the capture writer thread panicked".to_string()),
            });
        if let Some(e) = &end.close_error {
            self.state.stop(CaptureStop::WriteFailed);
            eprintln!("capture: {e}");
        }
        let stopped = self.state.stopped();
        let s = self.stats();
        CaptureOutcome {
            events: s.events,
            signals: s.signals,
            bytes: s.bytes,
            late_arrivals: end.late_arrivals,
            behind_derived_stamps: end.behind_derived_stamps,
            venue_stamped: end.venue_stamped,
            venue_stamped_out_of_order: end.venue_stamped_out_of_order,
            missed: s.missed,
            stopped,
            // The log only carries its real name when the recording was clean; anything
            // else is still `.partial`, and saying so is the whole point of the rename.
            log_path: match stopped {
                None => self.log_path.clone(),
                Some(_) => partial(&self.log_path),
            },
            signal_path: self.signal_path.clone().map(|p| match stopped {
                None => p,
                Some(_) => partial(&p),
            }),
        }
    }
}

/// The live counters, flattened the way the status line wants them.
///
/// Named for progress rather than stats because [`axon_replay::CaptureStats`] already
/// exists and describes the *file* — one name for two things is how a status line ends
/// up printing the wrong number confidently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CaptureProgress {
    pub events: u64,
    pub signals: u64,
    pub bytes: u64,
    pub missed: u64,
    pub queued: u64,
    pub capacity: u64,
    pub stopped: Option<CaptureStop>,
}

impl Drop for SessionRecorder {
    /// A recorder dropped without [`finish`](Self::finish) — a panic unwinding through
    /// the session, a `?` on some other error — still has to stop its thread and flush.
    /// Leaking it would leave the last events of the session in a userspace buffer and
    /// a thread parked on a queue nobody will ever close.
    ///
    /// It sends [`Rec::Close`], never [`Rec::Finish`]: what is on disk is a prefix of a
    /// session that ended for a reason nobody chose, and it must not be handed the name
    /// that means "this is all of it".
    fn drop(&mut self) {
        if self.writer.is_none() {
            return;
        }
        let _ = self.tx.send(Rec::Close);
        if let Some(h) = self.writer.take() {
            let _ = h.join();
        }
    }
}

fn file(path: &Path) -> std::io::Result<Box<dyn Write + Send>> {
    Ok(Box::new(BufWriter::new(File::create(path)?)))
}

/// Remove a previous session's log from a path this one now owns.
///
/// "It was not there" is the ordinary case and is success. Anything else is *not*
/// ignored: a stale log that cannot be removed is one a harness would still open, and
/// the rename at the end would fail against it too — better to say so at startup, while
/// nothing has been lost and an operator is still watching.
fn remove_stale(path: &Path) -> Result<(), ReplayError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(ReplayError::Io(e)),
    }
}

/// `foo.jsonl` → `foo.jsonl.partial`. An extra extension rather than a dotfile or a
/// sibling directory, so `ls` sorts the two next to each other and the relationship is
/// obvious to whoever finds one at 03:00.
fn partial(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".partial");
    PathBuf::from(s)
}

/// `foo.jsonl` → `foo.signals.jsonl`.
///
/// The same convention `axon-replay`'s `replay_log` binary looks for beside a log, so a
/// captured session replays by naming one path and nothing else. A convention rather
/// than a search: a harness that went hunting for a signal log could find the wrong
/// session's.
pub fn signal_log_path(log: &Path) -> PathBuf {
    log.with_extension("signals.jsonl")
}

/// The writer thread. Serialization and every `write(2)` happen here and nowhere else.
fn write_loop(
    rx: Receiver<Rec>,
    mut sink: Sink,
    state: Arc<RecorderState>,
    max_bytes: u64,
) -> WriterEnd {
    let mut late_arrivals = 0u64;
    let mut behind_derived_stamps = 0u64;
    let mut venue_stamped = 0u64;
    let mut venue_stamped_out_of_order = 0u64;
    let mut finished = false;
    for rec in rx.iter() {
        let payload = match rec {
            Rec::Close => break,
            Rec::Finish => {
                finished = true;
                break;
            }
            other => other,
        };
        // Nothing more is written once a stop is latched, whichever side latched it.
        // Writing past a hand-off the core could not make would produce a log whose
        // records are not contiguous, which is the artifact this refuses to create.
        //
        // A record still in the queue when the stop landed is counted as missed here and
        // nowhere else. Without that the totals would be short by up to a queue's worth,
        // and `events + missed` is only useful as an accounting identity — an operator
        // reading "8 210 missed" has to be able to trust it is not "8 210 plus whatever
        // was in flight".
        if state.stopped().is_some() {
            state.missed.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // Checked before the record, never during: a cap enforced mid-write leaves a
        // half-written line, and a half-written line is a corrupt log rather than a
        // short one.
        if max_bytes != 0 && sink.bytes.load(Ordering::Relaxed) >= max_bytes {
            if state.stop(CaptureStop::SizeCap) {
                eprintln!(
                    "capture: reached the {max_bytes}-byte cap - recording stopped. The cap \
                     is a disk guard, not a limit on what can be replayed: raise \
                     capture.max_bytes, or set it to 0 to let the recording run as long as \
                     the session does"
                );
            }
            state.missed.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let written = match &payload {
            Rec::Event(ev) => {
                let r = sink.events.record(ev);
                if r.is_ok() {
                    state.events.fetch_add(1, Ordering::Relaxed);
                    // Read back off the capture rather than re-derived here: the
                    // high-water rule that decides what "late" means belongs to
                    // `axon-replay`, and a second copy of it would drift.
                    let stats = sink.events.stats();
                    late_arrivals = stats.late_arrivals;
                    behind_derived_stamps = stats.behind_derived_stamps;
                    venue_stamped = stats.venue_stamped;
                    venue_stamped_out_of_order = stats.venue_stamped_out_of_order;
                }
                r
            }
            Rec::Signal(rec) => {
                let r = sink.record_signal(rec);
                if r.is_ok() {
                    state.signals.fetch_add(1, Ordering::Relaxed);
                }
                r
            }
            Rec::Close | Rec::Finish => unreachable!("both break out of the loop above"),
        };
        if let Err(e) = written {
            // The record that failed is lost too — possibly half-written, which is why
            // the file it went into never earns its final name.
            state.missed.fetch_add(1, Ordering::Relaxed);
            if state.stop(CaptureStop::WriteFailed) {
                eprintln!("capture: write failed ({e}) - recording stopped");
            }
        }
    }

    // Both halves are required. `finished` says the session reached its end;
    // `stopped()` says the log is contiguous. A recording that was cut off mid-session
    // has the second and not the first, and renaming on the second alone puts a
    // half-session log at exactly the path a harness reaches for — where it parses,
    // replays green, and says nothing about the half that is missing.
    let complete = finished && state.stopped().is_none();
    let close_error = sink.close(complete).err().map(|e| e.to_string());
    WriterEnd {
        late_arrivals,
        behind_derived_stamps,
        venue_stamped,
        venue_stamped_out_of_order,
        close_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{MarketEvent, Side, SymbolId, Trade};
    use axon_replay::EventLog;
    use rust_decimal_macros::dec;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    const SYM: SymbolId = SymbolId::new(1);
    const SEC: Nanos = 1_000_000_000;

    fn trade(ts: Nanos) -> Event {
        Event::Market(MarketEvent::Trade(Trade {
            symbol_id: SYM,
            px: dec!(100),
            sz: dec!(1),
            side: Side::Buy,
            ts_event: ts,
        }))
    }

    fn sig(seq: u64, ts: Nanos) -> Signal {
        Signal::target_position(seq, ts, SYM.get(), 100_000_000, 0, 0, 500, 1, 0)
    }

    fn temp(tag: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "axon-capture-{tag}-{}-{}.jsonl",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// A volume with `budget` bytes left on it, and ENOSPC after that.
    ///
    /// Byte-budgeted rather than call-budgeted because `serde_json` writes a record in
    /// many small pieces: a limit counted in calls would fire somewhere inside the
    /// header and test the startup path instead of the mid-session one.
    struct FailAfter {
        budget: usize,
    }

    impl Write for FailAfter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.budget == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    "no space left on device",
                ));
            }
            let n = buf.len().min(self.budget);
            self.budget -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A sink that takes `delay` to accept every write — a disk under load, or an
    /// `fsync` from a neighbour.
    struct SlowWriter {
        delay: Duration,
    }

    impl Write for SlowWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            std::thread::sleep(self.delay);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A recorder over caller-supplied sinks, with no files and therefore no rename.
    fn recorder_over(
        out: Box<dyn Write + Send>,
        queue_capacity: usize,
        max_bytes: u64,
    ) -> SessionRecorder {
        let bytes = Arc::new(AtomicU64::new(0));
        let events = Capture::with_header(
            Counted {
                inner: out,
                bytes: bytes.clone(),
            },
            LogHeader::new("unit-test", 0),
            InstrumentSet::Undeclared,
        )
        .expect("the header must be writable");
        let sink = Sink {
            events,
            signals: None,
            event_naming: None,
            signal_naming: None,
            bytes,
        };
        SessionRecorder::spawn(
            sink,
            queue_capacity,
            max_bytes,
            PathBuf::from("/dev/null"),
            None,
        )
    }

    fn cfg(path: &Path) -> CaptureConfig {
        CaptureConfig {
            enabled: true,
            path: path.to_string_lossy().into_owned(),
            signals: true,
            queue_capacity: 64,
            max_bytes: 1 << 20,
        }
    }

    /// A grid in the shape a venue publishes it, for the tests that care what a replay
    /// of the recording would round to.
    fn grids() -> InstrumentTable {
        let mut t = InstrumentTable::new();
        t.insert(axon_providers::InstrumentSpec {
            symbol_id: SYM,
            price: axon_providers::PriceGrid::decimals_with_sig_figs(1, 5).unwrap(),
            size: axon_providers::SizeGrid::decimals(5).unwrap(),
            min_notional: Some(dec!(10)),
        });
        t
    }

    #[test]
    fn a_capture_writes_what_it_was_handed_in_the_order_it_was_handed_it() {
        // The property the whole recording rests on: moving the write off the core
        // thread must not reorder anything, or a replay would interleave differently
        // from the session it claims to reproduce.
        let path = temp("order");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        for ts in [30, 10, 20] {
            tap.on_event(&trade(ts * SEC));
        }
        drop(tap);
        let out = rec.finish();

        assert!(out.complete(), "{out:?}");
        assert_eq!(out.events, 3);
        assert_eq!(out.late_arrivals, 2, "10 and 20 arrived behind 30");
        let log = EventLog::open(&out.log_path).expect("the finished log must parse");
        let ts: Vec<Nanos> = log.records().iter().map(|r| r.ts_event).collect();
        assert_eq!(ts, vec![30 * SEC, 10 * SEC, 20 * SEC], "capture order");
        cleanup(&out);
    }

    #[test]
    fn a_stalled_capture_writer_never_stalls_the_core() {
        // The failure this whole module is arranged around. A `BufWriter` on the core
        // thread would sit inside `write(2)` for as long as the disk takes; here the
        // core hands off with a `try_send`, the queue fills, and the *recording* is what
        // gives way. A stalled core is a stale book, and a stale book prices orders
        // against a market that has moved.
        let rec = recorder_over(
            Box::new(SlowWriter {
                delay: Duration::from_millis(5),
            }),
            2,
            u64::MAX,
        );
        let tap = rec.tap();
        let started = std::time::Instant::now();
        for i in 0..500 {
            tap.on_event(&trade(i * SEC));
        }
        let elapsed = started.elapsed();
        drop(tap);

        assert!(
            elapsed < Duration::from_millis(500),
            "500 hand-offs took {elapsed:?} against a writer that needs 5 ms each - the \
             core was waiting on the disk"
        );
        let out = rec.finish();
        assert_eq!(out.stopped, Some(CaptureStop::QueueFull), "{out:?}");
        assert!(out.missed > 0, "the size of the hole has to be a number");
        assert_eq!(
            out.events + out.missed,
            500,
            "every event is either recorded or counted as lost: {out:?}"
        );
    }

    #[test]
    fn a_capture_write_failure_stops_the_recording_and_not_the_session() {
        // A full disk mid-session must not take the process with it: there is a position
        // on the book by then, and killing the session over a log file trades a
        // recording for an unmanaged position.
        let rec = recorder_over(Box::new(FailAfter { budget: 800 }), 64, u64::MAX);
        let tap = rec.tap();
        for i in 0..50 {
            tap.on_event(&trade(i * SEC));
        }
        drop(tap);
        let out = rec.finish();
        assert_eq!(out.stopped, Some(CaptureStop::WriteFailed), "{out:?}");
        assert!(
            out.events > 0,
            "the prefix before the volume filled: {out:?}"
        );
        assert_eq!(
            out.events + out.missed,
            50,
            "every event is either on disk or counted as lost: {out:?}"
        );
    }

    #[test]
    fn a_stopped_capture_never_renames_its_log_into_place() {
        // The one artifact that fails silently is a truncated log sitting at the path a
        // harness reaches for: it parses, it replays, and it is green over a session
        // that ended early. The rename is the only thing that says "this is all of it".
        let path = temp("partial");
        let mut c = cfg(&path);
        c.max_bytes = 1; // the header alone is over the cap
        let rec = SessionRecorder::start(&c, "unit-test").unwrap().unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        drop(tap);
        let out = rec.finish();

        assert_eq!(out.stopped, Some(CaptureStop::SizeCap), "{out:?}");
        assert!(
            !path.exists(),
            "a truncated log must not be sitting at {}",
            path.display()
        );
        assert!(
            out.log_path.exists(),
            "…but it is still on disk for a post-mortem"
        );
        assert!(out.log_path.to_string_lossy().ends_with(".partial"));
        cleanup(&out);
    }

    #[test]
    fn the_size_reported_is_the_size_on_disk() {
        // The cap and the status line have to be reading one counter. Two counters agree
        // on the day they are written; the day they stop agreeing, the operator watches a
        // number that is not the one deciding when the recording dies.
        let path = temp("bytes");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        for i in 1..=20 {
            tap.on_event(&trade(i * SEC));
        }
        tap.on_signal(20 * SEC, &sig(1, 20 * SEC));
        drop(tap);
        let out = rec.finish();

        let on_disk = std::fs::metadata(&out.log_path).unwrap().len()
            + std::fs::metadata(out.signal_path.as_ref().unwrap())
                .unwrap()
                .len();
        assert_eq!(out.bytes, on_disk, "{out:?}");
        assert!(out.bytes > 0);
        cleanup(&out);
    }

    #[test]
    fn a_recording_that_ran_to_the_end_earns_its_name() {
        let path = temp("rename");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        drop(tap);
        let out = rec.finish();
        assert!(out.complete());
        assert_eq!(out.log_path, path);
        assert!(path.exists());
        assert!(!partial(&path).exists(), "the temporary name is gone");
        cleanup(&out);
    }

    #[test]
    fn a_recorder_dropped_without_finishing_never_earns_its_name() {
        // The path every abnormal end takes: the core thread panics at minute 30 of a
        // 60-minute session, `run_live` returns `Err(CorePanicked)`, and the recorder —
        // a local — is dropped on the way out. Nothing has stopped the recording, so a
        // rename conditioned on "no stop was latched" would put a half-session log at
        // exactly the path a harness reaches for, where it parses and replays green over
        // the half of the session that happened.
        let path = temp("dropped");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        tap.on_signal(SEC, &sig(1, SEC));
        drop(tap);
        drop(rec); // no `finish()`: the session did not reach its end

        assert!(
            !path.exists(),
            "a session that was cut short must not leave a log at {}",
            path.display()
        );
        assert!(
            !signal_log_path(&path).exists(),
            "…nor its signal sibling, which a harness finds by convention"
        );
        assert!(
            partial(&path).exists(),
            "…but the prefix is still on disk for a post-mortem"
        );
        let _ = std::fs::remove_file(partial(&path));
        let _ = std::fs::remove_file(partial(&signal_log_path(&path)));
    }

    #[test]
    fn a_new_recording_takes_the_path_away_from_the_session_that_used_it_last() {
        // A nightly soak writes to a fixed path. Monday finishes cleanly; Tuesday fills
        // the volume at 02:00 and withholds both renames. Without this, Tuesday's status
        // line says STOPPED while `session.jsonl` still holds *Monday's* complete log —
        // and a golden run over it is green against the wrong day, since `source` comes
        // from the config and is identical between the two.
        let path = temp("stalefinal");
        std::fs::write(&path, b"monday's complete log\n").unwrap();
        std::fs::write(signal_log_path(&path), b"monday's signals\n").unwrap();

        let mut c = cfg(&path);
        c.max_bytes = 1; // stops on the first record, so nothing is ever renamed
        let rec = SessionRecorder::start(&c, "unit-test").unwrap().unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        drop(tap);
        let out = rec.finish();

        assert_eq!(out.stopped, Some(CaptureStop::SizeCap), "{out:?}");
        assert!(
            !path.exists(),
            "the previous session's log is still at {}",
            path.display()
        );
        assert!(!signal_log_path(&path).exists());
        cleanup(&out);
    }

    #[test]
    fn a_recording_carries_the_grid_its_session_planned_against() {
        // The half of ADR-0027 that lives at the writing end. Without it a replay of this
        // log plans under `Precision::Unconstrained` while the session planned under
        // `Precision::Known` and rounded, so every price the grid moved comes back
        // different — reported by the golden harness as a strategy change nobody made.
        let path = temp("grid");
        let rec = SessionRecorder::start_with(&cfg(&path), "unit-test", &grids())
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        drop(tap);
        let out = rec.finish();

        let log = EventLog::open(&out.log_path).expect("the finished log must parse");
        let table = log
            .instruments()
            .to_table()
            .expect("a grid this build can rebuild")
            .expect("the recording declared one");
        assert_eq!(
            table.get(SYM).map(|s| s.price.tick_at(dec!(50000))),
            Some(dec!(1)),
            "the tick the session would have rounded to, read back out of its own log: \
             five significant figures at that magnitude is a whole dollar"
        );
        cleanup(&out);
    }

    #[test]
    fn a_recording_handed_no_grid_says_so_rather_than_claiming_the_venue_has_none() {
        // `Undeclared` is a third state and not an empty table. An empty one means every
        // symbol is `Precision::Unknown`, which refuses every order that adds exposure —
        // so a replay of it would report the session's own decisions as precision
        // refusals and send an operator to look at the venue.
        let path = temp("nogrid");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        tap.on_event(&trade(SEC));
        drop(tap);
        let out = rec.finish();

        let log = EventLog::open(&out.log_path).unwrap();
        assert_eq!(log.instruments(), &axon_replay::InstrumentSet::Undeclared);
        assert!(log.instruments().to_table().unwrap().is_none());
        cleanup(&out);
    }

    #[test]
    fn a_soak_that_subscribes_bars_does_not_report_the_venues_stamping_as_a_bad_feed() {
        // The end-to-end version of ADR-0027 §8, at the writing end, because this is the
        // number a soak operator reads without replaying anything. A venue republishes
        // the bar it is still filling and stamps every frame with the moment the bar
        // *will* close, so one frame walks the high-water mark a whole interval into the
        // future and every trade for the rest of that window lands behind it. Reported as
        // `late_arrivals` alone, a healthy feed looks like a reordering one.
        let ms = 1_000_000;
        let bar = |open_ms: Nanos| {
            Event::Market(MarketEvent::Candle(axon_core::Candle {
                symbol_id: SYM,
                interval: axon_core::CandleInterval::M1,
                open: dec!(100),
                high: dec!(101),
                low: dec!(99),
                close: dec!(100),
                volume: dec!(1),
                open_time: open_ms * ms,
                ts_event: (open_ms + 60_000) * ms,
            }))
        };
        let path = temp("formingbars");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        // A minute of trades with the same forming bar republished through it — the wire's
        // own shape, in miniature.
        for i in 0..30 {
            tap.on_event(&bar(0));
            tap.on_event(&trade(i * 2_000 * ms));
        }
        drop(tap);
        let out = rec.finish();

        assert_eq!(out.events, 60);
        assert!(out.late_arrivals > 0, "the traversals do differ: {out:?}");
        assert_eq!(
            out.behind_derived_stamps, out.late_arrivals,
            "every one of them is the bar's stamp, and none is the network: {out:?}"
        );
        // And the number an operator alarms on, measured rather than subtracted: the
        // venue-stamped trades were in perfect order among themselves.
        assert_eq!(out.venue_stamped, 30);
        assert_eq!(
            out.venue_stamped_out_of_order, 0,
            "a soak whose whole point is reconnecting must not report itself as a fault"
        );
        cleanup(&out);
    }

    #[test]
    fn a_cap_of_zero_lets_a_soak_run_as_long_as_the_session_does() {
        // The change ADR-0027 earns at this end. The cap used to be set where a log
        // stopped being *loadable*, because a replay held one whole; a multi-hour soak
        // therefore stopped its own recording at 512 MiB, loudly and uselessly. A log is
        // streamed now, so the cap is a disk guard — and an operator who has decided the
        // volume is theirs can turn it off without the recording quietly ignoring them.
        // A queue far wider than the run, so nothing here is a statement about the
        // writer keeping up — the only thing under test is that a zero cap does not stop
        // the recording.
        let rec = recorder_over(Box::new(std::io::sink()), 4096, 0);
        let tap = rec.tap();
        for i in 0..200 {
            tap.on_event(&trade(i * SEC));
        }
        drop(tap);
        let out = rec.finish();
        assert_eq!(out.stopped, None, "{out:?}");
        assert_eq!(out.events, 200);
        assert_eq!(out.missed, 0);
    }

    #[test]
    fn a_capture_that_could_not_be_created_refuses_to_start() {
        // Nothing has been lost yet and the operator is still watching. A session that
        // silently ran without the recording it was asked for is one whose soak produces
        // no artifact, discovered hours later.
        //
        // The unwritable path is a *file* used as a directory, so it fails with ENOTDIR
        // whoever the test runs as — a path under `/` would succeed for root and turn
        // this into a test that only holds on a developer's laptop.
        let blocker = temp("blocker");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let c = cfg(&blocker.join("nested").join("session.jsonl"));
        assert!(SessionRecorder::start(&c, "unit-test").is_err());
        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn capture_off_starts_no_thread_and_creates_no_file() {
        let path = temp("disabled");
        let mut c = cfg(&path);
        c.enabled = false;
        assert!(SessionRecorder::start(&c, "unit-test").unwrap().is_none());
        assert!(!path.exists());
        assert!(!partial(&path).exists());
    }

    #[test]
    fn a_recorded_signal_carries_the_release_time_the_ring_does_not() {
        // A `Signal` says when the strategy decided; the ring does not say when the
        // record became readable, and the gap between the two is the only thing that can
        // reproduce an expiry. Without it every replayed signal looks fresh.
        let path = temp("signals");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let mut src = RecordingSource::new(
            axon_strategy::ReplaySource::new(vec![sig(1, SEC), sig(2, 2 * SEC)]),
            Some(rec.tap()),
        );
        src.observe_event_time(5 * SEC);
        assert!(src.next_signal().is_some());
        src.observe_event_time(9 * SEC);
        assert!(src.next_signal().is_some());
        drop(src);
        let out = rec.finish();

        let log = SignalLog::open(out.signal_path.as_ref().unwrap()).expect("signal log");
        assert_eq!(log.len(), 2);
        assert_eq!(log.records()[0].release_ts, 5 * SEC);
        assert_eq!(log.records()[0].signal.ts_event, SEC, "decided earlier");
        assert_eq!(log.records()[1].release_ts, 9 * SEC);
        cleanup(&out);
    }

    #[test]
    fn a_late_event_never_makes_a_signal_log_the_reader_would_refuse() {
        // `CoreHandler::last_ts` follows the event, so one out-of-order arrival walks it
        // backwards. Recorded literally, the next record's `release_ts` would be lower
        // than its predecessor's — and `SignalLog::read` rejects that outright, because
        // an SPSC ring cannot hand records out in that order.
        let path = temp("latesig");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let mut src = RecordingSource::new(
            axon_strategy::ReplaySource::new(vec![sig(1, SEC), sig(2, SEC)]),
            Some(rec.tap()),
        );
        src.observe_event_time(9 * SEC);
        src.next_signal();
        src.observe_event_time(4 * SEC); // a late arrival
        src.next_signal();
        drop(src);
        let out = rec.finish();

        let log = SignalLog::open(out.signal_path.as_ref().unwrap())
            .expect("a recorded signal log must always be readable");
        assert_eq!(log.records()[1].release_ts, 9 * SEC, "the high-water mark");
        cleanup(&out);
    }

    #[test]
    fn a_recorded_signal_log_reads_back_through_the_same_release_schedule() {
        // The round trip this pair exists for: what `RecordingSource` writes,
        // `CapturedSignals` hands back at the same event times — not all at once, which
        // would collapse a session's decisions into a single moment.
        let path = temp("readback");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let mut src = RecordingSource::new(
            axon_strategy::ReplaySource::new(vec![sig(1, SEC), sig(2, 2 * SEC)]),
            Some(rec.tap()),
        );
        src.observe_event_time(SEC);
        src.next_signal();
        src.observe_event_time(8 * SEC);
        src.next_signal();
        drop(src);
        let out = rec.finish();

        let log = SignalLog::open(out.signal_path.as_ref().unwrap()).unwrap();
        let mut back = CapturedSignals::from_log(&log);
        assert!(
            back.next_signal().is_none(),
            "nothing before the first event"
        );
        back.observe_event_time(SEC);
        assert_eq!(back.next_signal().map(|s| s.seq), Some(1));
        assert!(back.next_signal().is_none(), "the second is not due yet");
        back.observe_event_time(8 * SEC);
        assert_eq!(back.next_signal().map(|s| s.seq), Some(2));
        assert_eq!(back.remaining(), 0);
        cleanup(&out);
    }

    #[test]
    fn a_late_arrival_does_not_withhold_a_signal_the_live_session_released() {
        // The reader's clock is the writer's clock, so it needs the writer's clamp. A
        // late-arriving event moves `CoreHandler::last_ts` backwards; if that rewound
        // the reader, every record at or above the old mark would stay blocked — and
        // because the cursor never rewinds, blocked for the rest of the replay. The
        // failure is silent and asymmetric: the replay plans nothing where the live
        // session placed an order, or plans it late enough to be rejected as `Expired`,
        // and a golden diff reports a strategy change that never happened.
        let mut back = CapturedSignals::new(vec![
            SignalRecord::now(&sig(1, 100 * SEC)),
            SignalRecord::now(&sig(2, 100 * SEC)),
        ]);
        back.observe_event_time(100 * SEC);
        assert_eq!(back.next_signal().map(|s| s.seq), Some(1));
        back.observe_event_time(90 * SEC); // the late arrival the writer survives
        assert_eq!(
            back.next_signal().map(|s| s.seq),
            Some(2),
            "a clock that rewinds strands every record behind it"
        );
        assert_eq!(back.remaining(), 0);
    }

    #[test]
    fn a_source_with_no_tap_records_nothing_and_still_hands_records_over() {
        // The wrapper is always in the wiring so a live session has one shape; a session
        // that is not recording must pay nothing for it.
        let mut src =
            RecordingSource::new(axon_strategy::ReplaySource::new(vec![sig(1, SEC)]), None);
        src.observe_event_time(SEC);
        assert_eq!(src.next_signal().map(|s| s.seq), Some(1));
    }

    #[test]
    fn a_log_truncated_at_a_record_boundary_is_refused_on_read() {
        // The record-level half of the same guarantee the rename gives at the file
        // level: a writer killed mid-line leaves a partial record, and skipping it would
        // quietly shorten a golden log while every check that only compares the rows
        // both runs produced stayed green.
        let path = temp("truncated");
        let rec = SessionRecorder::start(&cfg(&path), "unit-test")
            .unwrap()
            .unwrap();
        let tap = rec.tap();
        for i in 1..=5 {
            tap.on_event(&trade(i * SEC));
        }
        drop(tap);
        let out = rec.finish();

        let text = std::fs::read_to_string(&out.log_path).unwrap();
        let cut = &text[..text.len() - 20];
        std::fs::write(&out.log_path, cut).unwrap();
        let err = EventLog::open(&out.log_path).unwrap_err();
        assert!(
            matches!(err, ReplayError::Json { .. }),
            "a truncated log must fail loudly, got {err}"
        );
        cleanup(&out);
    }

    fn cleanup(out: &CaptureOutcome) {
        let _ = std::fs::remove_file(&out.log_path);
        if let Some(p) = &out.signal_path {
            let _ = std::fs::remove_file(p);
        }
    }
}
