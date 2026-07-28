//! The market-data **beacon**: the 64-byte sidecar that says a publisher is alive
//! when the ring cannot (ADR-0030, "the one seam this ADR specifies and does not
//! build").
//!
//! Under [`MdWritePolicy::OnChange`](../../axon_runtime/mdring/enum.MdWritePolicy.html)
//! a slice is written only when the state it carries actually moved. That is correct
//! and it makes an empty ring **ambiguous by design**: a quiet top of book and a dead
//! publisher produce the same nothing, and the Python parity monitor can only resolve
//! it with a timer — a guess with a deadline on it rather than a fact. This is the
//! same ambiguity [`axon.live.liveness`](../../../python/axon/live/liveness.py) removes
//! in the Python→Rust direction; nothing removed it in the Rust→Python one.
//!
//! So the publisher writes a counter that advances on the **pass loop** rather than on
//! the event handler. That distinction is the whole design: a beacon hung off
//! `on_event` carries exactly the information the ring already carries and buys
//! nothing. This one advances when *nothing arrives*, which is the case it exists for.
//!
//! ## Layout — 64 bytes, one cache line, little-endian, every field naturally aligned
//!
//! ```text
//!  0  u64  magic          = "AXONMDBN"
//!  8  u32  version        = 1
//! 12  u32  pid            publisher's process id (for the operator, not for logic)
//! 16  u64  beats          monotonic; the field a reader actually watches
//! 24  i64  last_event_ns  EVENT time high-water mark of the core (0 = nothing yet)
//! 32  u64  last_beat_ns   WALL clock at this beat (0 = this session has none)
//! 40  u32  published      slices the slice ring accepted        ┐
//! 44  u32  coalesced      updates OnChange suppressed           │ truncated to u32;
//! 48  u32  dropped        slices the ring refused (reader slow) │ read as a delta,
//! 52  u32  bars_published closed bars the bar ring accepted     │ never as a total
//! 56  u32  stale_quote    slices published with no quote at all ┘
//! 60  u32  flags          bit0 running, bit1 stopped cleanly; others unassigned
//! ```
//!
//! **The first 40 bytes and the last 4 are laid out exactly like
//! `axon.live.liveness`'s beacon, field for field**, because the two are mirror images
//! of one another across the same boundary and an operator who has learned one should
//! not have to learn the other. Only `[40, 60)` — the direction-specific payload —
//! differs. `magic` differs too, and it is the only thing standing between a reader and
//! decoding the *other* beacon's `signals` count as this one's `published`: the same
//! argument ADR-0012 §3 makes for `record_kind` over `record_size`, and it has a test
//! (`a_liveness_beacon_is_not_read_as_a_market_data_beacon`).
//!
//! ## Which clock, and why there are two
//!
//! The house rule is event time everywhere, with the exceptions named where they are
//! used. Both clocks are here, and each answers a question the other cannot.
//!
//! `last_event_ns` is **event time** — the core's own `ts_event` high-water mark, the
//! clock every ordering decision in the system is made on. It is what says whether the
//! *market* is moving.
//!
//! `last_beat_ns` is **wall clock (`CLOCK_REALTIME`), and it is a third named
//! exception** beside the dead-man's-switch deadline (wall clock because the *venue*
//! holds it) and the reconnect backoff (wall clock because it is not ordering). The
//! reason is the same shape as both: **the condition being detected is the absence of
//! events, and the absence of an event has no event time.** An event-time-only beacon
//! freezes at precisely the moment it is needed, because the clock that would advance
//! it *is* the thing that stopped — ADR-0030 §4 already makes this argument one process
//! later, for the monitor's own silence deadline; the beacon moves the measurement to
//! the side that actually knows whether it is alive. Nothing orders, ages, or admits
//! anything on `last_beat_ns`: it is read only as a difference, so that a **single**
//! read can say how long ago the publisher last ran instead of requiring two reads
//! spaced by a poll interval.
//!
//! `last_beat_ns == 0` means *this session has no wall clock* — an offline replay ages
//! nothing against one, and a reader must report that as "unknown" rather than as a
//! beat in 1970.
//!
//! ## Partial writes and torn reads
//!
//! The file is `ftruncate`d to 64 bytes before a single field is written, so a reader
//! never sees a short file with a plausible header; it sees 64 zero bytes, whose magic
//! is `0`, and is refused by name. There is no partial write in the `write(2)` sense at
//! all — the payload reaches the page through stores, not through a syscall.
//!
//! Within the 64 bytes: they are one cache line at offset 0 of a page, so nothing here
//! straddles a page or a line, and every field is naturally aligned. On x86_64 — the
//! platform ADR-0006 already assumes for this boundary — a naturally aligned store up
//! to 8 bytes is atomic, so **no field can be torn**. What remains possible is
//! *cross-field skew*: one read observing fields written by different beats.
//!
//! Skew is unavoidable — a reader's ten loads are not one instruction, and against a
//! pass loop spinning at kilohertz the payload really does run ahead of the count the
//! reader already latched. It is nonetheless safe, and by construction rather than by
//! timing:
//!
//! - Every payload field is **monotonically non-decreasing**, so any mixture of beats
//!   is a state the publisher genuinely passed through. A skewed read is never
//!   impossible, and no field ever goes backwards between two reads.
//! - `beats` is stored **last with `Release`** and loaded **first with `Acquire`**.
//!   Together those give the one guarantee that matters: **a payload field is always at
//!   least as new as the beat count reported beside it, never older.** So the reading
//!   that would turn a live publisher into a dead one — the count moving while the
//!   counters appear frozen — cannot be manufactured by skew at all. It can only be
//!   produced by a publisher that really did stop between its payload stores and its
//!   beat store and then stay stopped, which is a publisher that is dead. (The
//!   consequence to hold onto: `published - beats` is not a quantity, and nothing may
//!   compute it. Each is compared only against its own previous reading.)
//! - Every access on both sides is an atomic (`Relaxed` for the payload, `Release` /
//!   `Acquire` for `beats`), so there is no data race in the abstract machine either —
//!   which matters because a reader in the same process is exactly what this module's
//!   tests are. On x86_64 a relaxed atomic store compiles to the same `mov` a plain one
//!   would, so the hygiene is free.
//!
//! The Python reader narrows the window much further by copying all 64 bytes once and
//! parsing the copy; it can, because it is not obliged to be a data-race-free Rust
//! program. [`MdBeaconReader`] takes ten atomic loads instead and is meant for tools and
//! tests, not for a poll loop that has to be tight.
//!
//! A seqlock was rejected, and not for cost. It would make `beats` odd while a write is
//! in flight, so the field an operator watches would stop being a beat count; and it
//! would put a retry loop in the reader that can spin against a writer which died
//! mid-write — trading a staleness that is harmless *by direction* for an unbounded
//! stall in the one scenario the beacon exists to detect.
//!
//! ## Counters, and why five of them are `u32`
//!
//! 64 bytes is the budget and it is spent. Three 8-byte quantities are not negotiable —
//! the beat count, an event-time stamp and a wall-clock stamp — and with the 16-byte
//! header and `flags` that leaves 20 bytes for the payload. So the counters are `u32`
//! and they **wrap** rather than saturate. A reader must compute
//! `now.wrapping_sub(prev)`, which is exact for any two reads fewer than 2³² increments
//! apart — five days at ten thousand slices a second, against a monitor that polls per
//! window. Saturating would have been worse: a pinned counter yields a delta of zero
//! forever, so a live publisher's slice stream would read as quiet, which is exactly
//! the wrong answer to the only question this file is asked. The absolute value is not
//! meaningful and no reader should print it — `MdStats` on the status line is where the
//! totals live, and it is the one that has room for them.
//!
//! `stale_quote` earns its four bytes over `bars_dropped` on evidence: a reconnect that
//! failed to restore a `bbo` subscription put 45.8% of a 1 h 44 m soak under stale
//! marks with every other counter healthy, which is precisely the *alive but broken*
//! state a liveness object should be able to name.
//!
//! ## Where the file lives
//!
//! [`beacon_path`] appends `.beacon` to the slice ring's path, exactly as
//! `axon_runtime::mdring::bar_ring_path` inserts `bars` before its extension. Derived
//! rather than configured, for ADR-0028 §5's asymmetric-failure reason — one switch
//! cannot be half-turned — and with one extra benefit: ADR-0030 requires that
//! "validation must refuse a path equal to either ring's", and a path that is another
//! path plus a suffix **cannot** equal it. The rule discharges the requirement by
//! construction instead of by a check somebody has to remember to run.
//!
//! Put it on tmpfs with the rings. Every pass dirties the page, and on a real
//! filesystem the flusher would then write it back forever for no reason; `/dev/shm`
//! has no writeback at all. [`MdBeacon::flush`] is deliberately *not* called per beat —
//! that would be an `msync` syscall on the pass loop.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};

use memmap2::{MmapMut, MmapOptions};

/// Reads as ASCII `AXONMDBN` in a little-endian hexdump, matching the ring's idiom.
pub const MD_BEACON_MAGIC: u64 = u64::from_le_bytes(*b"AXONMDBN");
/// Bumped when the layout below changes. A reader refuses a version it does not know
/// rather than decoding fields that may have moved.
pub const MD_BEACON_VERSION: u32 = 1;
/// One cache line. Not a coincidence and not adjustable: see the module docs.
pub const MD_BEACON_SIZE: usize = 64;

pub const MD_BEACON_OFF_MAGIC: usize = 0;
pub const MD_BEACON_OFF_VERSION: usize = 8;
pub const MD_BEACON_OFF_PID: usize = 12;
pub const MD_BEACON_OFF_BEATS: usize = 16;
pub const MD_BEACON_OFF_LAST_EVENT_NS: usize = 24;
pub const MD_BEACON_OFF_LAST_BEAT_NS: usize = 32;
pub const MD_BEACON_OFF_PUBLISHED: usize = 40;
pub const MD_BEACON_OFF_COALESCED: usize = 44;
pub const MD_BEACON_OFF_DROPPED: usize = 48;
pub const MD_BEACON_OFF_BARS_PUBLISHED: usize = 52;
pub const MD_BEACON_OFF_STALE_QUOTE: usize = 56;
pub const MD_BEACON_OFF_FLAGS: usize = 60;

/// The publisher considers itself live.
pub const MD_BEACON_FLAG_RUNNING: u32 = 1;
/// The publisher shut down on purpose. A reader can tell this from a crash and say so —
/// though it must still treat the feed as gone, because a clean exit does not put the
/// market data back.
pub const MD_BEACON_FLAG_STOPPED: u32 = 2;

/// Errors from creating, opening or validating a beacon file.
#[derive(Debug, thiserror::Error)]
pub enum BeaconError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error(
        "not an Axon market-data beacon: bad magic {found:#018x} (expected {:#018x})",
        MD_BEACON_MAGIC
    )]
    BadMagic { found: u64 },

    #[error("unsupported beacon version {found} (this build expects {expected})")]
    Version { found: u32, expected: u32 },

    #[error("beacon file too small: {size} bytes, need {}", MD_BEACON_SIZE)]
    TooSmall { size: u64 },
}

/// Where the beacon lives, given the slice ring's path.
///
/// `/dev/shm/axon-md.ring` → `/dev/shm/axon-md.ring.beacon`. The suffix is *appended*
/// rather than substituted for the extension, and that is the load-bearing part: a
/// string cannot equal itself plus a suffix, so this can never name the slice ring's
/// own file however the ring's path was spelled — including the adversarial
/// `path = "…/axon-md.beacon"`. See the module docs.
pub fn beacon_path(md_ring_path: &str) -> PathBuf {
    PathBuf::from(format!("{md_ring_path}.beacon"))
}

/// One pass loop's worth of publisher state, as the caller already holds it.
///
/// Counters are taken as `u64` — the width `MdStats` keeps them at — so a call site
/// passes its fields straight through with no casts, and the truncation rule lives in
/// exactly one place ([`MdBeacon::beat`]) with the comment that explains it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MdBeat {
    /// The core's event-time high-water mark. `0` before any event has been seen.
    pub last_event_ns: i64,
    /// Wall clock at this pass. `0` when the session has none (offline), which the
    /// reader must report as "unknown" and not as a beat in 1970.
    pub wall_ns: u64,
    pub published: u64,
    pub coalesced: u64,
    pub dropped: u64,
    pub bars_published: u64,
    pub stale_quote: u64,
}

/// The publisher's end of the beacon. One per process; creates and truncates the file.
///
/// Every method takes `&self`, matching [`RingProducer`](crate::RingProducer): all
/// mutation goes through a mutable-provenance raw pointer into the mapping, so this can
/// be driven from a `&`-borrowed pass loop without a lock. Like the ring it is `Send`
/// but deliberately not `Sync` — two threads beating one beacon would interleave two
/// publishers' counters into one story.
#[derive(Debug)]
pub struct MdBeacon {
    mmap: MmapMut,
    base: *mut u8,
    _file: File,
}

// SAFETY: single-writer, exactly as for `RingProducer`. Moving the beacon to the thread
// that owns the pass loop is sound; sharing a `&MdBeacon` between two writers is not,
// and the missing `Sync` is what says so.
unsafe impl Send for MdBeacon {}

impl MdBeacon {
    /// Create (or reinitialise) the beacon at `path` and stamp its header.
    ///
    /// This end owns the file — it overwrites whatever is there — which is why it is
    /// created by the same switch that creates the rings and not from a path an operator
    /// types at it.
    ///
    /// **It reinitialises rather than `O_TRUNC`s, and that is not a detail.** Truncating
    /// takes the file to zero length before `set_len` puts it back, and a reader that has
    /// the page mapped takes a `SIGBUS` — not an error it can handle, a signal — for
    /// touching a page past EOF during that window. The failure would be a monitor killed
    /// by the publisher it restarted, at the moment it was most needed. Every field is
    /// written below, so nothing of the previous session survives anyway; the length
    /// simply never dips.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, BeaconError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            // `truncate(false)` stated rather than omitted: clippy asks for the intent to
            // be explicit here, and the intent is exactly the one in the doc comment
            // above — overwrite every byte, never shorten the file.
            .truncate(false)
            .open(path)?;
        file.set_len(MD_BEACON_SIZE as u64)?;
        // SAFETY: we own this freshly sized file; nothing else changes its length while
        // it is mapped.
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        // Mutable provenance, valid for the life of the mapping — `as_ptr()` would give
        // read-only provenance and writing through it is UB.
        let base = mmap.as_mut_ptr();
        let b = Self {
            mmap,
            base,
            _file: file,
        };
        // The payload before the header, so the window in which another process can see
        // a file whose magic is right and whose fields are not does not exist. Until the
        // magic lands the file is 64 zero bytes, which every reader here refuses by name.
        b.u64_at(MD_BEACON_OFF_BEATS).store(0, Ordering::Relaxed);
        b.i64_at(MD_BEACON_OFF_LAST_EVENT_NS)
            .store(0, Ordering::Relaxed);
        b.u64_at(MD_BEACON_OFF_LAST_BEAT_NS)
            .store(0, Ordering::Relaxed);
        for off in [
            MD_BEACON_OFF_PUBLISHED,
            MD_BEACON_OFF_COALESCED,
            MD_BEACON_OFF_DROPPED,
            MD_BEACON_OFF_BARS_PUBLISHED,
            MD_BEACON_OFF_STALE_QUOTE,
        ] {
            b.u32_at(off).store(0, Ordering::Relaxed);
        }
        b.u32_at(MD_BEACON_OFF_FLAGS)
            .store(MD_BEACON_FLAG_RUNNING, Ordering::Relaxed);
        b.u32_at(MD_BEACON_OFF_PID)
            .store(std::process::id(), Ordering::Relaxed);
        b.u32_at(MD_BEACON_OFF_VERSION)
            .store(MD_BEACON_VERSION, Ordering::Relaxed);
        b.u64_at(MD_BEACON_OFF_MAGIC)
            .store(MD_BEACON_MAGIC, Ordering::Release);
        b.mmap.flush()?;
        Ok(b)
    }

    #[inline]
    fn u64_at(&self, off: usize) -> &AtomicU64 {
        // SAFETY: `off + 8 <= MD_BEACON_SIZE`, the mapping is at least that long, the
        // offset is 8-aligned, and every access to it anywhere is atomic.
        unsafe { AtomicU64::from_ptr(self.base.add(off) as *mut u64) }
    }
    #[inline]
    fn i64_at(&self, off: usize) -> &AtomicI64 {
        unsafe { AtomicI64::from_ptr(self.base.add(off) as *mut i64) }
    }
    #[inline]
    fn u32_at(&self, off: usize) -> &AtomicU32 {
        unsafe { AtomicU32::from_ptr(self.base.add(off) as *mut u32) }
    }

    /// Publish one pass of the loop. Returns the new beat count.
    ///
    /// **Call this from the pass loop, not from the event handler.** A beacon written
    /// when a record is published carries the same information the ring already
    /// carries, and the case it has to distinguish — nothing arrived — is exactly the
    /// case in which `on_event` does not run.
    ///
    /// Allocation-free, syscall-free and lock-free: seven relaxed stores into one
    /// already-resident cache line and a release store. There is no `flush` here on
    /// purpose — that is an `msync`, and the pass loop is not the place for one.
    #[inline]
    pub fn beat(&self, b: MdBeat) -> u64 {
        self.publish(b, MD_BEACON_FLAG_RUNNING)
    }

    /// Publish a final beat marked as a deliberate shutdown, and flush it.
    ///
    /// A reader still has to treat the feed as gone — a clean exit does not put the
    /// market data back — but "the publisher stopped" and "the publisher died" want
    /// different words at 03:00, and only the publisher can tell them apart.
    pub fn stop(&self, b: MdBeat) -> u64 {
        let beats = self.publish(b, MD_BEACON_FLAG_STOPPED);
        let _ = self.mmap.flush();
        beats
    }

    #[inline]
    fn publish(&self, b: MdBeat, flags: u32) -> u64 {
        self.i64_at(MD_BEACON_OFF_LAST_EVENT_NS)
            .store(b.last_event_ns, Ordering::Relaxed);
        self.u64_at(MD_BEACON_OFF_LAST_BEAT_NS)
            .store(b.wall_ns, Ordering::Relaxed);
        // Truncated, not saturated, and this is the only place it happens. A wrapped
        // counter still yields an exact delta under `wrapping_sub` for any two reads
        // fewer than 2^32 increments apart; a saturated one yields zero forever, and a
        // live publisher would then read as a quiet one. See the module docs.
        self.u32_at(MD_BEACON_OFF_PUBLISHED)
            .store(b.published as u32, Ordering::Relaxed);
        self.u32_at(MD_BEACON_OFF_COALESCED)
            .store(b.coalesced as u32, Ordering::Relaxed);
        self.u32_at(MD_BEACON_OFF_DROPPED)
            .store(b.dropped as u32, Ordering::Relaxed);
        self.u32_at(MD_BEACON_OFF_BARS_PUBLISHED)
            .store(b.bars_published as u32, Ordering::Relaxed);
        self.u32_at(MD_BEACON_OFF_STALE_QUOTE)
            .store(b.stale_quote as u32, Ordering::Relaxed);
        self.u32_at(MD_BEACON_OFF_FLAGS)
            .store(flags, Ordering::Relaxed);
        let beats = self.u64_at(MD_BEACON_OFF_BEATS).load(Ordering::Relaxed) + 1;
        // Published last, with Release, after every field it describes — the same
        // discipline the ring's `head` uses, and the reason a reader that sees a new
        // count also sees the payload behind it.
        self.u64_at(MD_BEACON_OFF_BEATS)
            .store(beats, Ordering::Release);
        beats
    }

    /// The current beat count, for the writer's own status line.
    #[inline]
    pub fn beats(&self) -> u64 {
        self.u64_at(MD_BEACON_OFF_BEATS).load(Ordering::Relaxed)
    }

    /// Push the page to the backing file. Not needed for same-host visibility (the page
    /// cache is coherent), and deliberately not on the beat path.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// One read of a beacon file — everything the reader needs and nothing derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MdBeaconSnapshot {
    pub pid: u32,
    pub beats: u64,
    pub last_event_ns: i64,
    /// `0` means the publishing session had no wall clock, not 1970.
    pub last_beat_ns: u64,
    pub published: u32,
    pub coalesced: u32,
    pub dropped: u32,
    pub bars_published: u32,
    pub stale_quote: u32,
    pub flags: u32,
}

impl MdBeaconSnapshot {
    pub fn running(&self) -> bool {
        self.flags & MD_BEACON_FLAG_RUNNING != 0
    }
    pub fn stopped(&self) -> bool {
        self.flags & MD_BEACON_FLAG_STOPPED != 0
    }
}

/// The reader's end: maps the file once and re-reads it on demand.
///
/// Mapped rather than re-opened per poll because a monitor polls this for hours, and
/// because a `read(2)` of a file another process is storing into has the same skew
/// window with a syscall's cost added to it.
#[derive(Debug)]
pub struct MdBeaconReader {
    #[allow(dead_code)] // held to keep the mapping (and therefore `base`) alive
    mmap: MmapMut,
    base: *mut u8,
    _file: File,
}

// SAFETY: as for `RingConsumer` — one reader thread, moved not shared.
unsafe impl Send for MdBeaconReader {}

impl MdBeaconReader {
    /// Open and validate a beacon written by a publisher.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BeaconError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        if len < MD_BEACON_SIZE as u64 {
            return Err(BeaconError::TooSmall { size: len });
        }
        // SAFETY: the file is at least one beacon long; the header is validated below.
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let base = mmap.as_mut_ptr();
        let r = Self {
            mmap,
            base,
            _file: file,
        };
        // SAFETY: `base` is 8-aligned and the mapping is at least 64 bytes.
        let magic = unsafe { AtomicU64::from_ptr(base as *mut u64) }.load(Ordering::Acquire);
        if magic != MD_BEACON_MAGIC {
            return Err(BeaconError::BadMagic { found: magic });
        }
        let version = r.u32_at(MD_BEACON_OFF_VERSION).load(Ordering::Relaxed);
        if version != MD_BEACON_VERSION {
            return Err(BeaconError::Version {
                found: version,
                expected: MD_BEACON_VERSION,
            });
        }
        Ok(r)
    }

    #[inline]
    fn u64_at(&self, off: usize) -> &AtomicU64 {
        unsafe { AtomicU64::from_ptr(self.base.add(off) as *mut u64) }
    }
    #[inline]
    fn i64_at(&self, off: usize) -> &AtomicI64 {
        unsafe { AtomicI64::from_ptr(self.base.add(off) as *mut i64) }
    }
    #[inline]
    fn u32_at(&self, off: usize) -> &AtomicU32 {
        unsafe { AtomicU32::from_ptr(self.base.add(off) as *mut u32) }
    }

    /// One snapshot. `beats` is loaded **first, with `Acquire`**, so everything read
    /// after it is at least as new as the beat it names; see the module docs for why a
    /// skew of one beat in the other direction is harmless and why there is no retry.
    pub fn read(&self) -> MdBeaconSnapshot {
        let beats = self.u64_at(MD_BEACON_OFF_BEATS).load(Ordering::Acquire);
        MdBeaconSnapshot {
            pid: self.u32_at(MD_BEACON_OFF_PID).load(Ordering::Relaxed),
            beats,
            last_event_ns: self
                .i64_at(MD_BEACON_OFF_LAST_EVENT_NS)
                .load(Ordering::Relaxed),
            last_beat_ns: self
                .u64_at(MD_BEACON_OFF_LAST_BEAT_NS)
                .load(Ordering::Relaxed),
            published: self.u32_at(MD_BEACON_OFF_PUBLISHED).load(Ordering::Relaxed),
            coalesced: self.u32_at(MD_BEACON_OFF_COALESCED).load(Ordering::Relaxed),
            dropped: self.u32_at(MD_BEACON_OFF_DROPPED).load(Ordering::Relaxed),
            bars_published: self
                .u32_at(MD_BEACON_OFF_BARS_PUBLISHED)
                .load(Ordering::Relaxed),
            stale_quote: self
                .u32_at(MD_BEACON_OFF_STALE_QUOTE)
                .load(Ordering::Relaxed),
            flags: self.u32_at(MD_BEACON_OFF_FLAGS).load(Ordering::Relaxed),
        }
    }
}

/// Open, read once, and drop. For tools and one-shot checks; a poller should hold an
/// [`MdBeaconReader`].
pub fn read_md_beacon<P: AsRef<Path>>(path: P) -> Result<MdBeaconSnapshot, BeaconError> {
    Ok(MdBeaconReader::open(path)?.read())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64 as Counter;
    use std::thread;

    static COUNTER: Counter = Counter::new(0);

    fn temp_beacon(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        p.push(format!(
            "axon-beacon-{}-{}-{}.beacon",
            tag,
            std::process::id(),
            n
        ));
        p
    }

    #[test]
    fn a_beacon_advances_on_a_pass_that_carried_no_records() {
        // The entire reason this file exists. Under `OnChange` a quiet market and a dead
        // publisher both leave the ring empty; the only thing that can separate them is
        // a counter driven by the pass loop rather than by the event handler. If this
        // were written from `on_event` it would be flat here, and a reader would have no
        // way to tell this session from a corpse.
        let path = temp_beacon("quiet");
        let beacon = MdBeacon::create(&path).unwrap();
        let quiet = MdBeat {
            last_event_ns: 1_700_000_000_000_000_000,
            wall_ns: 42,
            coalesced: 9,
            ..MdBeat::default()
        };
        for _ in 0..5 {
            beacon.beat(quiet);
        }
        let snap = read_md_beacon(&path).unwrap();
        assert_eq!(snap.beats, 5, "the loop ran five times");
        assert_eq!(snap.published, 0, "and published nothing while it did");
        assert!(snap.running() && !snap.stopped());
        drop(beacon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_publisher_that_stopped_beating_is_what_a_dead_one_looks_like() {
        // The other half of the same sentence: the signal has to be *absence* of
        // advance, so two reads with no beat between them must be identical.
        let path = temp_beacon("dead");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat::default());
        let reader = MdBeaconReader::open(&path).unwrap();
        let first = reader.read();
        let second = reader.read();
        assert_eq!(first.beats, second.beats, "nothing beat in between");
        beacon.beat(MdBeat::default());
        assert_eq!(reader.read().beats, first.beats + 1);
        drop(beacon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn every_field_a_reader_watches_survives_the_round_trip() {
        let path = temp_beacon("fields");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat {
            last_event_ns: -7, // signed, and a reader that read it unsigned would say 1.8e19
            wall_ns: 1_700_000_000_123_456_789,
            published: 11,
            coalesced: 22,
            dropped: 33,
            bars_published: 44,
            stale_quote: 55,
        });
        let s = read_md_beacon(&path).unwrap();
        assert_eq!(s.pid, std::process::id());
        assert_eq!(s.beats, 1);
        assert_eq!(s.last_event_ns, -7);
        assert_eq!(s.last_beat_ns, 1_700_000_000_123_456_789);
        assert_eq!(
            (
                s.published,
                s.coalesced,
                s.dropped,
                s.bars_published,
                s.stale_quote
            ),
            (11, 22, 33, 44, 55),
            "five counters at five distinct offsets: a transposition passes every \
             single-field check and fails this one"
        );
        drop(beacon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_counter_past_the_u32_wrap_still_yields_an_exact_delta() {
        // The cost of spending four bytes on a counter, made explicit. The absolute is
        // meaningless after a wrap and no reader may print it; the *delta* — the only
        // thing anyone asks of it — stays exact.
        let path = temp_beacon("wrap");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat {
            published: u32::MAX as u64 - 1,
            ..MdBeat::default()
        });
        let before = read_md_beacon(&path).unwrap().published;
        beacon.beat(MdBeat {
            published: u32::MAX as u64 + 4,
            ..MdBeat::default()
        });
        let after = read_md_beacon(&path).unwrap().published;
        assert!(after < before, "it really did wrap");
        assert_eq!(
            after.wrapping_sub(before),
            5,
            "and the delta is still exact"
        );
        drop(beacon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_final_beat_says_the_publisher_stopped_on_purpose_rather_than_died() {
        let path = temp_beacon("stopped");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat::default());
        beacon.stop(MdBeat {
            published: 3,
            ..MdBeat::default()
        });
        let s = read_md_beacon(&path).unwrap();
        assert!(s.stopped(), "a clean exit is distinguishable from a crash");
        assert!(!s.running());
        assert_eq!(s.beats, 2, "the shutdown is itself a beat");
        drop(beacon);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_liveness_beacon_is_not_read_as_a_market_data_beacon() {
        // Both beacons are 64 bytes with an identical first 40, so nothing about the
        // size or the shape of the file can tell them apart — the magic is the only
        // thing that can, which is ADR-0012 §3's argument for `record_kind` over
        // `record_size` in a second place. Read the wrong one and `signals` becomes
        // `published`: a plausible number, silently wrong, forever.
        let path = temp_beacon("liveness");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat::default());
        drop(beacon);
        let liveness = u64::from_le_bytes(*b"AXONBEAT");
        assert_ne!(liveness, MD_BEACON_MAGIC, "the premise of this test");
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(MD_BEACON_OFF_MAGIC as u64)).unwrap();
            f.write_all(&liveness.to_le_bytes()).unwrap();
            f.sync_all().unwrap();
        }
        let err = read_md_beacon(&path).unwrap_err();
        assert!(matches!(err, BeaconError::BadMagic { .. }), "got {err:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_restarted_publisher_resets_the_beat_without_shortening_the_file() {
        // A new session must not inherit the old one's beat count — a reader would see
        // it jump and call it progress — and it must not shorten the file to do it.
        //
        // Be clear about which half this proves. The beat reset is asserted. The
        // no-shrink rule is **not**: the danger there is a reader taking a `SIGBUS` for
        // touching a page past EOF during the instant an `O_TRUNC` leaves the file at
        // zero length, and a sequential test cannot be inside that instant — swapping
        // `create` back to `.truncate(true)` leaves this test green. What is checked is
        // the observable consequence (the file is a full beacon afterwards, and a reader
        // that was already mapped still works); the reason for the code shape is in
        // `create`'s doc comment, and this test is the guard on it, not the evidence.
        let path = temp_beacon("restart");
        let first = MdBeacon::create(&path).unwrap();
        for _ in 0..9 {
            first.beat(MdBeat::default());
        }
        let watcher = MdBeaconReader::open(&path).unwrap();
        assert_eq!(watcher.read().beats, 9);
        drop(first);

        let second = MdBeacon::create(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            MD_BEACON_SIZE as u64
        );
        second.beat(MdBeat::default());
        assert_eq!(
            watcher.read().beats,
            1,
            "a restarted publisher starts its count again, and the watcher survives to see it"
        );
        drop(second);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_zeroed_file_is_refused_rather_than_read_as_a_beacon_at_beat_zero() {
        // The window between `ftruncate` and the header store, and also what a reader
        // finds if it is pointed at a path the publisher never created. "Beats 0, never
        // advancing" would read as a dead publisher, which is a *conclusion*; the
        // truthful answer is that this is not a beacon.
        let path = temp_beacon("zeroed");
        std::fs::write(&path, [0u8; MD_BEACON_SIZE]).unwrap();
        let err = read_md_beacon(&path).unwrap_err();
        assert!(
            matches!(err, BeaconError::BadMagic { found: 0 }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_short_file_is_refused_rather_than_read_past_its_end() {
        let path = temp_beacon("short");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let err = read_md_beacon(&path).unwrap_err();
        assert!(
            matches!(err, BeaconError::TooSmall { size: 16 }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_version_this_build_does_not_know_is_refused_rather_than_misread() {
        let path = temp_beacon("version");
        let beacon = MdBeacon::create(&path).unwrap();
        beacon.beat(MdBeat::default());
        drop(beacon);
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(MD_BEACON_OFF_VERSION as u64))
                .unwrap();
            f.write_all(&99u32.to_le_bytes()).unwrap();
            f.sync_all().unwrap();
        }
        let err = read_md_beacon(&path).unwrap_err();
        assert!(
            matches!(err, BeaconError::Version { found: 99, .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_beacon_can_never_name_the_same_file_as_either_ring() {
        // ADR-0030 asks for validation refusing a beacon path equal to either ring's.
        // Appending a suffix makes that unrepresentable instead, which is the stronger
        // form: there is no check here that anybody can forget to call.
        //
        // `bar_ring_path` is mirrored rather than imported because the dependency runs
        // the other way — `axon-runtime` depends on `axon-ipc`. If the two ever disagree
        // this test is the wrong place to notice, so it asserts the *structural*
        // property (a strict suffix cannot equal its own base) rather than a literal.
        fn bar_like(p: &str) -> String {
            match Path::new(p).extension().and_then(|e| e.to_str()) {
                Some(ext) => Path::new(p)
                    .with_extension(format!("bars.{ext}"))
                    .to_string_lossy()
                    .into_owned(),
                None => format!("{p}.bars"),
            }
        }
        for ring in [
            "/dev/shm/axon-md.ring",
            "/dev/shm/axon-md",
            "/dev/shm/axon-md.beacon", // the adversarial one
            "/dev/shm/axon-md.bars.ring",
            "/dev/shm/axon.md.ring.beacon",
        ] {
            let beacon = beacon_path(ring);
            assert_ne!(
                beacon,
                Path::new(ring),
                "{ring}: beacon collided with the slice ring"
            );
            assert_ne!(
                beacon,
                Path::new(&bar_like(ring)),
                "{ring}: beacon collided with the bar ring"
            );
        }
    }

    #[test]
    fn a_read_that_races_a_beat_never_shows_the_count_ahead_of_the_payload() {
        // The torn-read argument, run rather than asserted. The writer stores
        // `published = k` and then `beats = k`; the reader loads `beats` first. So the
        // payload may legitimately run *ahead* of the count — by as much as the reader
        // is slow — but the count can never run ahead of the payload, and that is the
        // only direction that could turn a live publisher into a dead-looking one.
        // Store `beats` before the payload instead and this reddens.
        let path = temp_beacon("race");
        let beacon = MdBeacon::create(&path).unwrap();
        let n = 200_000u64;
        let writer = thread::spawn(move || {
            for k in 1..=n {
                beacon.beat(MdBeat {
                    published: k,
                    coalesced: k,
                    last_event_ns: k as i64,
                    ..MdBeat::default()
                });
            }
            beacon
        });

        let reader = MdBeaconReader::open(&path).unwrap();
        let mut prev = reader.read();
        let mut reads = 0u64;
        while prev.beats < n {
            let s = reader.read();
            assert!(s.beats >= prev.beats, "beats went backwards");
            assert!(
                s.published.wrapping_sub(prev.published) < u32::MAX / 2,
                "published went backwards"
            );
            assert!(
                s.last_event_ns >= prev.last_event_ns,
                "the event clock went backwards"
            );
            // `published` is set to the same k the beat is numbered with, so this is the
            // publish discipline itself: the payload is never behind the count.
            assert!(
                s.published as u64 >= s.beats,
                "the beat count outran the payload it describes: beats {} vs published {}",
                s.beats,
                s.published
            );
            prev = s;
            reads += 1;
        }
        assert!(
            reads > 0,
            "the reader has to have actually raced the writer"
        );
        drop(writer.join().unwrap());
        let _ = std::fs::remove_file(&path);
    }
}
