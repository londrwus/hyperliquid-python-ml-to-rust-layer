//! # axon-ipc
//!
//! The shared-memory transport for the Python↔Rust boundary (ADR-0002, ADR-0006,
//! ADR-0012): a **single-producer, single-consumer (SPSC)** ring buffer laid over a
//! memory-mapped file. The byte layout is defined once in `contracts/schema.toml`
//! and surfaced through [`axon_contracts::layout`], so the Python client
//! (`python/axon/signals`, `python/axon/marketdata`) and this crate map the *same*
//! bytes.
//!
//! One ring implementation serves every direction, generic over
//! [`Record`]: [`Producer`]/[`Consumer`] carry `Signal`s Python→Rust,
//! [`MdProducer`]/[`MdConsumer`] carry `MdSlice`s Rust→Python, and
//! [`MdBarProducer`]/[`MdBarConsumer`] carry closed `MdBar`s the same way. Each is a
//! separate ring file — SPSC means one writer and one reader per ring, and a
//! bidirectional or multiplexed ring would be neither.
//!
//! There is deliberately **no** crate-level `RECORD_SIZE`. A ring's stride is
//! `R::SIZE` and nothing else; a single unqualified constant would name one record's
//! size as if it were *the* record size, which is the exact confusion ADR-0012's
//! explicit `record_kind` header exists to catch — a reader that trusts a stride and
//! maps the wrong struct reports `target_qty` as a bid price and keeps going. Reach
//! for `axon_contracts::SIGNAL_SIZE` or `MdSlice::SIZE` when a literal is needed, so
//! the name says which record it means. `MdSlice` and `MdBar` are **both 128 bytes**
//! (ADR-0028), so on those two the stride check cannot discriminate at all and
//! `record_kind` is the only thing that does.
//!
//! ## Why a memory-mapped *file* (not `shmem-ipc`/`iceoryx2`)?
//! It is the one primitive that is truly portable: on Linux the file lives in
//! `/dev/shm` (tmpfs — RAM-backed, zero disk I/O), on Windows in a temp dir, and
//! both `memmap2` (Rust) and the `mmap` module (Python) map it coherently across
//! processes. `shmem-ipc` is Linux-only and `iceoryx2` is pre-1.0; both remain
//! future optimizations behind this same [`Producer`]/[`Consumer`] API.
//!
//! ## Memory-ordering contract
//! Two monotonically increasing counters, `head` (producer-owned) and `tail`
//! (consumer-owned), sit on **separate 64-byte cache lines** to avoid false
//! sharing (a measured ~3× penalty — see `docs/research/python-rust-ipc.md`).
//!
//! - Producer: read `tail` (Acquire) to check for space → write the record bytes
//!   → publish by storing `head+1` (Release).
//! - Consumer: read `head` (Acquire) to check for data → read the record bytes →
//!   release the slot by storing `tail+1` (Release).
//!
//! The Release/Acquire pair makes the record write *happen-before* the reader's
//! observation of the new index. **Platform assumption: x86_64** (Windows dev,
//! Linux/WSL2 prod). The Python producer relies on x86-TSO store ordering (it
//! writes the payload before the index); an ARM target would additionally need an
//! explicit fence on the Python side.
//!
//! ## Safety
//! `head`/`tail` are only ever touched through `AtomicU64`; the data slots are
//! only ever touched by the owning side while it holds the slot. Sharing a
//! `&Producer` (or `&Consumer`) across threads would violate the SPSC contract —
//! keep one producer thread and one consumer thread.
//!
//! ## The beacon beside the ring
//! A ring says what arrived. It cannot say that *nothing* arrived, because under
//! `MdWritePolicy::OnChange` a quiet market and a dead publisher leave the same
//! empty ring (ADR-0030). [`beacon`] is the 64-byte mmap'd sidecar that closes
//! that: a counter driven by the core's **pass loop** rather than by its event
//! handler, so it advances when nothing arrives. It lives here for the same reason
//! the ring does — mapping a file is the `unsafe` this crate already holds, and
//! `#![deny(unsafe_code)]` stands everywhere else.

pub mod beacon;

pub use beacon::{
    beacon_path, read_md_beacon, BeaconError, MdBeacon, MdBeaconReader, MdBeaconSnapshot, MdBeat,
    MD_BEACON_FLAG_RUNNING, MD_BEACON_FLAG_STOPPED, MD_BEACON_MAGIC, MD_BEACON_SIZE,
    MD_BEACON_VERSION,
};

use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use axon_contracts::{layout, MdBar, MdSlice, Signal};
use memmap2::{MmapMut, MmapOptions};

pub use axon_contracts::Record;

/// The Python→Rust signal ring's writer end.
pub type Producer = RingProducer<Signal>;
/// The Python→Rust signal ring's reader end.
pub type Consumer = RingConsumer<Signal>;
/// The Rust→Python market-data ring's writer end (the Rust core publishes).
pub type MdProducer = RingProducer<MdSlice>;
/// The Rust→Python market-data ring's reader end — for Rust-side tests and tools;
/// the real consumer of this ring is `python/axon/marketdata`.
pub type MdConsumer = RingConsumer<MdSlice>;
/// The Rust→Python **bar** ring's writer end (ADR-0028): closed OHLCV bars.
pub type MdBarProducer = RingProducer<MdBar>;
/// The Rust→Python bar ring's reader end — for Rust-side tests and tools; the real
/// consumer is `python/axon/marketdata`'s `MdBarRingConsumer`.
pub type MdBarConsumer = RingConsumer<MdBar>;

/// Errors from creating, opening, or validating a ring.
#[derive(Debug, thiserror::Error)]
pub enum RingError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("capacity must be a power of two and > 0, got {0}")]
    BadCapacity(u64),

    #[error(
        "not an Axon ring: bad magic {found:#018x} (expected {:#018x})",
        layout::RING_MAGIC
    )]
    BadMagic { found: u64 },

    #[error("unsupported ring version {found} (this build expects {expected})")]
    RingVersion { found: u32, expected: u32 },

    #[error("record size mismatch: header says {found}, {record} is {expected}")]
    RecordSize {
        found: u32,
        expected: u32,
        record: &'static str,
    },

    #[error("ring carries record kind {found}, this endpoint reads kind {expected} ({record})")]
    RecordKind {
        found: u32,
        expected: u32,
        record: &'static str,
    },

    #[error("{record} schema version mismatch: header {found}, this build {expected}")]
    SchemaVersion {
        found: u32,
        expected: u32,
        record: &'static str,
    },

    #[error("mapped file too small: {size} bytes, need at least {need}")]
    TooSmall { size: u64, need: u64 },

    /// The beacon beside the ring failed to open. Folded in here rather than left as a
    /// second error type because the one caller creates all three files together
    /// (`MdPublisher::open`), and a separate `Result` there would be a `map_err` at the
    /// call site — which is where the beacon's wiring has to stay a single line if it is
    /// to stay wired.
    #[error("market-data beacon: {0}")]
    Beacon(#[from] beacon::BeaconError),
}

/// Total mapped length for a ring of `capacity` slots of `record_size` bytes.
#[inline]
fn total_len(capacity: u64, record_size: usize) -> u64 {
    layout::RING_HEADER_SIZE as u64 + capacity * record_size as u64
}

// ── Raw header helpers (control fields are written once, read once; not hot) ──
#[inline]
unsafe fn write_u64(base: *mut u8, off: usize, v: u64) {
    (base.add(off) as *mut u64).write(v);
}
#[inline]
unsafe fn write_u32(base: *mut u8, off: usize, v: u32) {
    (base.add(off) as *mut u32).write(v);
}
#[inline]
unsafe fn read_u64(base: *const u8, off: usize) -> u64 {
    (base.add(off) as *const u64).read()
}
#[inline]
unsafe fn read_u32(base: *const u8, off: usize) -> u32 {
    (base.add(off) as *const u32).read()
}

/// Validate the control header of an already-mapped region against record `R` and
/// return `capacity`.
fn validate_header<R: Record>(base: *const u8, mapped_len: u64) -> Result<u64, RingError> {
    // SAFETY: caller guarantees `mapped_len >= RING_HEADER_SIZE`, so every control
    // field offset is in bounds.
    let magic = unsafe { read_u64(base, layout::RING_OFF_MAGIC) };
    if magic != layout::RING_MAGIC {
        return Err(RingError::BadMagic { found: magic });
    }
    let ring_version = unsafe { read_u32(base, layout::RING_OFF_RING_VERSION) };
    if ring_version != layout::RING_VERSION {
        return Err(RingError::RingVersion {
            found: ring_version,
            expected: layout::RING_VERSION,
        });
    }
    let record_size = unsafe { read_u32(base, layout::RING_OFF_RECORD_SIZE) };
    if record_size as usize != R::SIZE {
        return Err(RingError::RecordSize {
            found: record_size,
            expected: R::SIZE as u32,
            record: R::NAME,
        });
    }
    // Checked *in addition to* the stride: equal strides do not imply equal records,
    // and a wrong-record reader that gets past this point decodes garbage that looks
    // like data (ADR-0012).
    let record_kind = unsafe { read_u32(base, layout::RING_OFF_RECORD_KIND) };
    if record_kind != R::KIND {
        return Err(RingError::RecordKind {
            found: record_kind,
            expected: R::KIND,
            record: R::NAME,
        });
    }
    let capacity = unsafe { read_u32(base, layout::RING_OFF_CAPACITY) } as u64;
    if capacity == 0 || !capacity.is_power_of_two() {
        return Err(RingError::BadCapacity(capacity));
    }
    let schema = unsafe { read_u32(base, layout::RING_OFF_RECORD_SCHEMA_VERSION) };
    if schema != R::SCHEMA_VERSION as u32 {
        return Err(RingError::SchemaVersion {
            found: schema,
            expected: R::SCHEMA_VERSION as u32,
            record: R::NAME,
        });
    }
    let need = total_len(capacity, R::SIZE);
    if mapped_len < need {
        return Err(RingError::TooSmall {
            size: mapped_len,
            need,
        });
    }
    Ok(capacity)
}

/// The writer end of a ring carrying record `R`. Exactly one per ring.
///
/// All reads/writes go through `base`, a pointer with **mutable provenance** from
/// [`MmapMut::as_mut_ptr`] (going through `as_ptr()` would give read-only
/// provenance — writing through that is UB). The raw-pointer field also makes
/// `RingProducer` neither `Send` nor `Sync` by default; we opt back into `Send` (an
/// endpoint may be *moved* to a thread) but deliberately NOT `Sync` — sharing a
/// `&RingProducer` across threads would break the single-producer contract and race.
#[derive(Debug)]
pub struct RingProducer<R: Record> {
    mmap: MmapMut,
    base: *mut u8,
    capacity: u64,
    mask: u64,
    data_off: usize,
    _file: File,
    _record: PhantomData<R>,
}

// SAFETY: SPSC — a RingProducer is used by one thread at a time; moving it between
// threads is sound. `base` aliases only producer-owned regions (head + free slots).
unsafe impl<R: Record> Send for RingProducer<R> {}

impl<R: Record> RingProducer<R> {
    /// Create (or truncate) a ring file of `capacity` slots and initialize its
    /// header. `capacity` must be a power of two that fits the `u32` header field.
    pub fn create<P: AsRef<Path>>(path: P, capacity: u64) -> Result<Self, RingError> {
        if capacity == 0 || !capacity.is_power_of_two() || capacity > u32::MAX as u64 {
            return Err(RingError::BadCapacity(capacity));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_len(capacity, R::SIZE))?;
        // SAFETY: we own this freshly sized file; no other process mutates its
        // length while mapped.
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        // Mutable-provenance base pointer, valid for the life of the mapping (the
        // OS mapping address is fixed, so it survives moving `mmap` into `p`).
        let base = mmap.as_mut_ptr();

        let p = Self {
            mmap,
            base,
            capacity,
            mask: capacity - 1,
            data_off: layout::RING_HEADER_SIZE,
            _file: file,
            _record: PhantomData,
        };
        // Write control fields, then zero the indices, then flush so a consumer
        // that opens the file immediately sees a valid header.
        // SAFETY: the map is at least `total_len(capacity, R::SIZE)` bytes; `base`
        // is valid + writable.
        unsafe {
            write_u64(base, layout::RING_OFF_MAGIC, layout::RING_MAGIC);
            write_u32(base, layout::RING_OFF_RING_VERSION, layout::RING_VERSION);
            write_u32(base, layout::RING_OFF_RECORD_SIZE, R::SIZE as u32);
            write_u32(base, layout::RING_OFF_CAPACITY, capacity as u32);
            write_u32(
                base,
                layout::RING_OFF_RECORD_SCHEMA_VERSION,
                R::SCHEMA_VERSION as u32,
            );
            write_u32(base, layout::RING_OFF_RECORD_KIND, R::KIND);
        }
        p.head().store(0, Ordering::Release);
        p.tail().store(0, Ordering::Release);
        p.mmap.flush()?;
        Ok(p)
    }

    #[inline]
    fn head(&self) -> &AtomicU64 {
        // SAFETY: RING_OFF_HEAD is an 8-aligned, in-bounds slot only accessed atomically.
        unsafe { AtomicU64::from_ptr(self.base.add(layout::RING_OFF_HEAD) as *mut u64) }
    }
    #[inline]
    fn tail(&self) -> &AtomicU64 {
        unsafe { AtomicU64::from_ptr(self.base.add(layout::RING_OFF_TAIL) as *mut u64) }
    }

    /// Try to push one record's raw bytes. Returns `false` if the ring is full.
    ///
    /// # Panics
    /// If `rec.len() != R::SIZE`. A short buffer would leave the tail of a
    /// published slot holding the previous occupant's bytes.
    #[inline]
    pub fn try_push_bytes(&self, rec: &[u8]) -> bool {
        assert_eq!(rec.len(), R::SIZE, "record must be exactly one stride");
        let head = self.head().load(Ordering::Relaxed);
        let tail = self.tail().load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.capacity {
            return false; // full
        }
        let slot = (head & self.mask) as usize;
        let off = self.data_off + slot * R::SIZE;
        // SAFETY: `off + R::SIZE <= total_len`, and this slot is free (checked
        // above), so the consumer is not reading it. `base` has write provenance.
        unsafe {
            let dst = self.base.add(off);
            std::ptr::copy_nonoverlapping(rec.as_ptr(), dst, R::SIZE);
        }
        // Publish: makes the record write visible before the new head.
        self.head().store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to push one record. Returns `false` if the ring is full.
    #[inline]
    pub fn try_push(&self, rec: &R) -> bool {
        self.try_push_bytes(bytemuck::bytes_of(rec))
    }

    /// Number of records currently queued.
    #[inline]
    pub fn len(&self) -> u64 {
        self.head()
            .load(Ordering::Relaxed)
            .wrapping_sub(self.tail().load(Ordering::Acquire))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Flush written pages to the backing file. Not required for same-host
    /// cross-process visibility (page-cache coherency handles that), but makes the
    /// handoff explicit — used by the cross-language round-trip example.
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// The reader end of a ring carrying record `R`. Exactly one per ring. See
/// [`RingProducer`] for the `base`-pointer / `Send`-not-`Sync` rationale.
#[derive(Debug)]
pub struct RingConsumer<R: Record> {
    /// Held to keep the mapping alive (dropping it unmaps `base`'s memory);
    /// all access goes through `base`.
    #[allow(dead_code)]
    mmap: MmapMut,
    base: *mut u8,
    capacity: u64,
    mask: u64,
    data_off: usize,
    _file: File,
    _record: PhantomData<R>,
}

// SAFETY: SPSC — a RingConsumer is used by one thread at a time; moving it between
// threads is sound. `base` aliases only consumer-owned regions (tail + full slots).
unsafe impl<R: Record> Send for RingConsumer<R> {}

impl<R: Record> RingConsumer<R> {
    /// Open an existing ring file (created by a [`RingProducer`] here or by the
    /// Python client) and validate its header against this build's contract —
    /// including that the ring actually carries record `R`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RingError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        if len < layout::RING_HEADER_SIZE as u64 {
            return Err(RingError::TooSmall {
                size: len,
                need: layout::RING_HEADER_SIZE as u64,
            });
        }
        // SAFETY: file is at least a header long; we validate the rest below.
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let base = mmap.as_mut_ptr();
        let capacity = validate_header::<R>(base, len)?;
        Ok(Self {
            mmap,
            base,
            capacity,
            mask: capacity - 1,
            data_off: layout::RING_HEADER_SIZE,
            _file: file,
            _record: PhantomData,
        })
    }

    #[inline]
    fn head(&self) -> &AtomicU64 {
        unsafe { AtomicU64::from_ptr(self.base.add(layout::RING_OFF_HEAD) as *mut u64) }
    }
    #[inline]
    fn tail(&self) -> &AtomicU64 {
        unsafe { AtomicU64::from_ptr(self.base.add(layout::RING_OFF_TAIL) as *mut u64) }
    }

    /// Try to pop one record's raw bytes into `out`. Returns `false` if empty.
    ///
    /// # Panics
    /// If `out.len() != R::SIZE`.
    #[inline]
    pub fn try_pop_bytes(&self, out: &mut [u8]) -> bool {
        assert_eq!(out.len(), R::SIZE, "output must be exactly one stride");
        let tail = self.tail().load(Ordering::Relaxed);
        let head = self.head().load(Ordering::Acquire);
        if tail == head {
            return false; // empty
        }
        let slot = (tail & self.mask) as usize;
        let off = self.data_off + slot * R::SIZE;
        // SAFETY: producer published this slot (head moved past it), so its bytes
        // are fully written and stable until we release it below.
        unsafe {
            let src = self.base.add(off);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), R::SIZE);
        }
        self.tail().store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Try to pop one record.
    #[inline]
    pub fn try_pop(&self) -> Option<R> {
        // Copy into an `R` rather than a byte array: `R::SIZE` is not usable as an
        // array length in a generic context, and an `R` is correctly aligned.
        let mut rec = R::zeroed();
        if self.try_pop_bytes(bytemuck::bytes_of_mut(&mut rec)) {
            Some(rec)
        } else {
            None
        }
    }

    /// Drain up to `max` records, calling `f` for each. Returns how many were read.
    pub fn drain<F: FnMut(R)>(&self, max: usize, mut f: F) -> usize {
        let mut n = 0;
        while n < max {
            match self.try_pop() {
                Some(s) => {
                    f(s);
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    #[inline]
    pub fn len(&self) -> u64 {
        self.head()
            .load(Ordering::Acquire)
            .wrapping_sub(self.tail().load(Ordering::Relaxed))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::{MD_KIND_QUOTE, MD_KIND_TRADE};
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_ring(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        p.push(format!(
            "axon-ipc-{}-{}-{}.ring",
            tag,
            std::process::id(),
            n
        ));
        p
    }

    fn sig(seq: u64) -> Signal {
        Signal::target_position(seq, seq as i64, (seq % 7) as u32, seq as i64, 0, 0, 0, 1, 0)
    }

    fn slice(seq: u64) -> MdSlice {
        MdSlice::new(seq, seq as i64, (seq % 7) as u32, MD_KIND_QUOTE)
            .with_bbo(100 + seq as i64, 1, 101 + seq as i64, 2)
            .with_ticker(
                100 + seq as i64,
                99,
                (125, 3_600_000_000_000),
                0,
                seq as i64,
            )
    }

    fn bar(seq: u64) -> MdBar {
        let open_time = (seq as i64) * 60_000_000_000;
        MdBar::new(
            seq,
            open_time + 60_000_000_000,
            open_time,
            (seq % 7) as u32,
            60_000,
            0,
        )
        .with_ohlcv(100, 110, 90, 105, 7)
    }

    #[test]
    fn single_push_pop() {
        let path = temp_ring("single");
        let producer = Producer::create(&path, 8).unwrap();
        let consumer = Consumer::open(&path).unwrap();
        assert!(consumer.try_pop().is_none());
        assert!(producer.try_push(&sig(42)));
        assert_eq!(consumer.len(), 1);
        let got = consumer.try_pop().unwrap();
        assert_eq!(got, sig(42));
        assert!(consumer.try_pop().is_none());
        drop(producer);
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn full_then_wraparound() {
        let path = temp_ring("full");
        let producer = Producer::create(&path, 2).unwrap();
        let consumer = Consumer::open(&path).unwrap();
        assert!(producer.try_push(&sig(1)));
        assert!(producer.try_push(&sig(2)));
        assert!(!producer.try_push(&sig(3))); // full
        assert_eq!(consumer.try_pop().unwrap().seq, 1);
        assert!(producer.try_push(&sig(3))); // slot freed → wraps around
        assert_eq!(consumer.try_pop().unwrap().seq, 2);
        assert_eq!(consumer.try_pop().unwrap().seq, 3);
        assert!(consumer.try_pop().is_none());
        drop(producer);
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn spsc_threaded_no_loss_in_order() {
        let path = temp_ring("threaded");
        let producer = Producer::create(&path, 64).unwrap(); // small → forces backpressure
        let consumer = Consumer::open(&path).unwrap();
        let n = 200_000u64;

        let producer_thread = thread::spawn(move || {
            let mut i = 0u64;
            while i < n {
                if producer.try_push(&sig(i)) {
                    i += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        let mut got = 0u64;
        while got < n {
            match consumer.try_pop() {
                Some(s) => {
                    assert_eq!(s.seq, got, "records must arrive in order with no loss");
                    got += 1;
                }
                None => thread::yield_now(),
            }
        }
        producer_thread.join().unwrap();
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn create_rejects_oversized_capacity() {
        // 2^32 is a valid u64 power of two but does not fit the u32 header field.
        let err = Producer::create(temp_ring("toobig"), 1u64 << 32).unwrap_err();
        assert!(matches!(err, RingError::BadCapacity(_)), "got {err:?}");
    }

    #[test]
    fn open_rejects_non_ring_file() {
        let path = temp_ring("garbage");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();
        let err = Consumer::open(&path).unwrap_err();
        assert!(matches!(err, RingError::BadMagic { .. }), "got {err:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn md_ring_carries_slices_in_order() {
        let path = temp_ring("md");
        let producer = MdProducer::create(&path, 8).unwrap();
        let consumer = MdConsumer::open(&path).unwrap();
        for i in 0..5 {
            assert!(producer.try_push(&slice(i)));
        }
        for i in 0..5 {
            assert_eq!(consumer.try_pop().unwrap(), slice(i));
        }
        assert!(consumer.try_pop().is_none());
        drop(producer);
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_signal_consumer_refuses_an_md_ring() {
        let path = temp_ring("wrongtype");
        let producer = MdProducer::create(&path, 4).unwrap();
        assert!(producer.try_push(&slice(1)));
        // Caught on stride here; `a_matching_stride_does_not_admit_the_wrong_record`
        // covers the case where the stride agrees.
        let err = Consumer::open(&path).unwrap_err();
        assert!(matches!(err, RingError::RecordSize { .. }), "got {err:?}");
        drop(producer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_matching_stride_does_not_admit_the_wrong_record() {
        // The two strides differ *today*. Forge a signal ring that claims the MdSlice
        // stride to prove the type check does not rest on that coincidence: an
        // MdSlice reader over Signal bytes would silently report a bid of whatever
        // `target_qty` happens to be.
        let path = temp_ring("stridematch");
        {
            let producer = Producer::create(&path, 4).unwrap();
            assert!(producer.try_push(&sig(1)));
            producer.flush().unwrap();
        }
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(layout::RING_OFF_RECORD_SIZE as u64))
            .unwrap();
        f.write_all(&(MdSlice::SIZE as u32).to_le_bytes()).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let err = MdConsumer::open(&path).unwrap_err();
        assert!(matches!(err, RingError::RecordKind { .. }), "got {err:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unstamped_kind_is_not_read_as_signal() {
        // Ring kind 0 is unassigned, so a producer that never wrote the field (a
        // pre-ADR-0012 writer, or a zeroed page) must fail loudly rather than
        // default into the signal ring.
        let path = temp_ring("nokind");
        {
            let producer = Producer::create(&path, 4).unwrap();
            producer.flush().unwrap();
        }
        let mut f = OpenOptions::new().write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(layout::RING_OFF_RECORD_KIND as u64))
            .unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let err = Consumer::open(&path).unwrap_err();
        assert!(
            matches!(err, RingError::RecordKind { found: 0, .. }),
            "got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_bar_ring_and_a_slice_ring_are_told_apart_with_no_stride_to_help() {
        // The case ADR-0012 §3 was written for, and it is no longer hypothetical:
        // `MdBar` and `MdSlice` are both 128 bytes, so `record_size` agrees on both
        // sides and cannot discriminate. Without the kind tag an `MdConsumer` over a
        // bar ring would report `open` as a bid price and `high` as a bid size —
        // plausible numbers, silently wrong, forever.
        let path = temp_ring("barvsslice");
        let producer = MdBarProducer::create(&path, 4).unwrap();
        assert_eq!(MdBar::SIZE, MdSlice::SIZE, "the premise of this test");
        assert!(producer.try_push(&bar(1)));

        let err = MdConsumer::open(&path).unwrap_err();
        assert!(matches!(err, RingError::RecordKind { .. }), "got {err:?}");
        // …and the reverse: a bar reader must not decode a slice ring either.
        drop(producer);
        let slices = MdProducer::create(&path, 4).unwrap();
        assert!(slices.try_push(&slice(1)));
        let err = MdBarConsumer::open(&path).unwrap_err();
        assert!(matches!(err, RingError::RecordKind { .. }), "got {err:?}");
        drop(slices);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bar_ring_carries_bars_in_order() {
        let path = temp_ring("bars");
        let producer = MdBarProducer::create(&path, 8).unwrap();
        let consumer = MdBarConsumer::open(&path).unwrap();
        for i in 0..5 {
            assert!(producer.try_push(&bar(i)));
        }
        for i in 0..5 {
            assert_eq!(consumer.try_pop().unwrap(), bar(i));
        }
        assert!(consumer.try_pop().is_none());
        drop(producer);
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn md_slices_survive_backpressure_intact() {
        // The md ring is the one direction where the producer (the Rust core) must
        // not block: this proves a full ring reports full instead of tearing the
        // record the consumer is mid-read on.
        let path = temp_ring("mdfull");
        let producer = MdProducer::create(&path, 2).unwrap();
        let consumer = MdConsumer::open(&path).unwrap();
        let hot = MdSlice::new(9, 9, 1, MD_KIND_TRADE).with_last_trade(7, 3, 8, true);
        assert!(producer.try_push(&hot));
        assert!(producer.try_push(&slice(2)));
        assert!(!producer.try_push(&slice(3)));
        assert_eq!(consumer.try_pop().unwrap(), hot);
        assert!(producer.try_push(&slice(3)));
        drop(producer);
        drop(consumer);
        let _ = std::fs::remove_file(&path);
    }
}
