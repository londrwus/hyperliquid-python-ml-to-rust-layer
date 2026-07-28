//! # axon-contracts
//!
//! The Rust half of Axon's language-neutral boundary contract. Everything here is
//! derived from and checked against [`contracts/schema.toml`] — the single source
//! of truth shared with Python (`python/axon/contracts`).
//!
//! - [`layout`] holds constants *generated* from the schema at build time.
//! - [`Signal`] (Python→Rust), [`MdSlice`] and [`MdBar`] (both Rust→Python) are the
//!   fixed-layout, `#[repr(C)]` records that cross the shared-memory rings. Their
//!   field offsets are asserted against [`layout`] at **compile time** (see the
//!   `const _` blocks below), so a struct that drifts from the schema simply fails
//!   to build.
//! - [`Record`] is what makes a type ring-carryable: its wire stride, its layout
//!   version, and the kind tag the ring stamps in its control block so a reader
//!   cannot map the wrong record type onto a matching stride (ADR-0012).
//!
//! Nothing in this crate does I/O; the ring transport lives in `axon-ipc`, which
//! consumes the [`layout`] ring constants exposed here.
//!
//! [`contracts/schema.toml`]: ../../../contracts/schema.toml

use bytemuck::{Pod, Zeroable};

/// Layout constants generated at build time from `contracts/schema.toml`.
///
/// This module is the machine-checked bridge between the schema and the structs:
/// the `Signal` struct asserts each field's byte offset against `OFF_*` here.
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/layout.rs"));
}

pub use layout::{
    FIXED_POINT_DECIMALS, FIXED_POINT_SCALE, FLAG_CLOSE, FLAG_REDUCE_ONLY, KIND_TARGET_POSITION,
    MD_BAR_ALIGN, MD_BAR_FLAG_FIRST_BAR, MD_BAR_FLAG_GAP_BEFORE, MD_BAR_KIND_CLOSED,
    MD_BAR_SCHEMA_VERSION, MD_BAR_SIZE, MD_FLAG_LAST_TRADE_SELL, MD_KIND_QUOTE, MD_KIND_SNAPSHOT,
    MD_KIND_TRADE, MD_SLICE_ALIGN, MD_SLICE_SCHEMA_VERSION, MD_SLICE_SIZE, RING_CACHE_LINE,
    RING_HEADER_SIZE, RING_KIND_MD_BAR, RING_KIND_MD_SLICE, RING_KIND_SIGNAL, RING_MAGIC,
    RING_VERSION, SCHEMA_VERSION, SIGNAL_ALIGN, SIGNAL_SIZE,
};

/// A fixed-layout record that an Axon ring can carry.
///
/// Implemented only by the `#[repr(C)]` structs in this crate, whose offsets are
/// compile-time-checked against `contracts/schema.toml`. The transport
/// (`axon-ipc`) is generic over this trait, which is what keeps one ring
/// implementation — one set of memory-ordering rules — serving both directions of
/// the boundary.
///
/// [`KIND`](Record::KIND) is the tag the producer writes into the ring's control
/// block and every consumer validates on open. It exists because [`SIZE`](Record::SIZE)
/// is *not* a type: two records with the same stride are byte-compatible and
/// mutually meaningless, and a mismatched decode of plausible-looking numbers is
/// worse than a failure to open.
pub trait Record: Pod {
    /// Human-readable record name, for error messages.
    const NAME: &'static str;
    /// Wire stride in bytes — must equal `size_of::<Self>()` (asserted per impl).
    const SIZE: usize;
    /// Layout version stamped in the record and cross-checked in the ring header.
    const SCHEMA_VERSION: u8;
    /// Which record type this is; see [`RING_KIND_SIGNAL`], [`RING_KIND_MD_SLICE`].
    const KIND: u32;
}

/// A strategy's per-decision output crossing the Python→Rust boundary.
///
/// Canonical shape is **target-position** (ADR-0006): the strategy declares the
/// position it wants; the Rust execution engine decides *how* to reach it. The
/// `kind` + `reserved` fields leave room for an explicit-order-intent variant
/// without changing the 64-byte stride.
///
/// Layout is fixed, little-endian, and padding-free (asserted below). Quantities
/// and prices are fixed-point integers in units of `10^-FIXED_POINT_DECIMALS`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Signal {
    /// Monotonic sequence number (gap detection + offline replay).
    pub seq: u64,
    /// Event-time nanoseconds — the event's own time, never wall-clock at receipt.
    pub ts_event: i64,
    /// Fixed-point signed target position (`10^-FIXED_POINT_DECIMALS` units).
    pub target_qty: i64,
    /// Worst-acceptable price (fixed-point); `0` means no band.
    pub price_band: i64,
    /// Canonical symbol id (resolved via the venue adapter's `SymbolMap`).
    pub symbol_id: u32,
    /// Signal validity window in milliseconds.
    pub ttl_ms: u32,
    /// Which model version produced this signal (audit / replay).
    pub model_version: u32,
    /// Bitfield: see [`FLAG_REDUCE_ONLY`], [`FLAG_CLOSE`].
    pub flags: u16,
    /// Layout version guard; readers reject unknown values loudly.
    pub schema_version: u8,
    /// Execution aggressiveness hint (`0` = most passive).
    pub urgency: u8,
    /// Record kind discriminant; see [`KIND_TARGET_POSITION`].
    pub kind: u8,
    /// Explicit padding to the next 4-byte boundary. MUST be zero.
    ///
    /// Spelled out rather than left implicit because `#[derive(Pod)]` rejects implicit
    /// padding outright, and because the schema's rule is that every byte of the record
    /// is named — an unnamed hole is a byte no reader can be asked to validate.
    pub pad0: [u8; 3],
    /// How long an order this signal places may keep its place at the venue, in
    /// milliseconds; `0` defers to the operator's ceiling.
    ///
    /// **Not `ttl_ms`.** That is a signal *admission* window, consumed by the reader
    /// before the planner ever sees the record and clamped against the operator's
    /// `max_signal_age_ms`; nothing in it has ever applied to an order already resting
    /// at the venue. This is the field the planner reads (ADR-0031).
    pub max_order_age_ms: u32,
    /// Event-time nanoseconds of the **observation this decision answers** — an m1
    /// bar's own close, not its arrival, and not the moment the strategy got round to
    /// it. `0` means the producer did not state one.
    ///
    /// **Why this is on the wire at all.** [`ts_event`](Self::ts_event) is the moment the
    /// strategy *decided*, which is the right stamp for admission (`ttl_ms` ages a
    /// decision, not a bar) and is useless for the largest latency in the system. A
    /// closed m1 bar reached the strategy 951 / 12 051 / **111 475** ms after its own
    /// close on 2026-07-27, and from the runtime's side a decision one second after a bar
    /// and one two minutes after it were the same record — so the gap could be observed
    /// only in the producer's private transcript and could never be given a ceiling.
    /// A budget nobody can measure against cannot regress.
    ///
    /// **It is a cross-clock span and that is named rather than hidden.** The cause is
    /// stamped by the venue and the decision by the producer, so
    /// `ts_event - ts_cause` includes whatever skew exists between those two clocks.
    /// Both are epoch nanoseconds, which is what makes the subtraction meaningful; the
    /// skew is bounded by the same NTP discipline every other cross-process figure here
    /// relies on, and it is small next to a twelve-second median.
    ///
    /// **A producer that does not know its cause writes `0`, and the stage is then not
    /// measured for that record** — the same reading `ttl_ms` and `max_order_age_ms` give
    /// zero. Not every strategy is on bars, and inventing a cause for a tick-driven one
    /// would report a latency that describes nothing.
    pub ts_cause: i64,
}

// ── Compile-time layout guard ────────────────────────────────────────────────
// If any field moves relative to `contracts/schema.toml`, the crate fails to
// build. `#[derive(Pod)]` additionally rejects any implicit padding.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<Signal>() == layout::SIGNAL_SIZE);
    assert!(align_of::<Signal>() == layout::SIGNAL_ALIGN);
    assert!(offset_of!(Signal, seq) == layout::OFF_SEQ);
    assert!(offset_of!(Signal, ts_event) == layout::OFF_TS_EVENT);
    assert!(offset_of!(Signal, target_qty) == layout::OFF_TARGET_QTY);
    assert!(offset_of!(Signal, price_band) == layout::OFF_PRICE_BAND);
    assert!(offset_of!(Signal, symbol_id) == layout::OFF_SYMBOL_ID);
    assert!(offset_of!(Signal, ttl_ms) == layout::OFF_TTL_MS);
    assert!(offset_of!(Signal, model_version) == layout::OFF_MODEL_VERSION);
    assert!(offset_of!(Signal, flags) == layout::OFF_FLAGS);
    assert!(offset_of!(Signal, schema_version) == layout::OFF_SCHEMA_VERSION);
    assert!(offset_of!(Signal, urgency) == layout::OFF_URGENCY);
    assert!(offset_of!(Signal, kind) == layout::OFF_KIND);
    assert!(offset_of!(Signal, pad0) == layout::OFF_PAD0);
    assert!(offset_of!(Signal, max_order_age_ms) == layout::OFF_MAX_ORDER_AGE_MS);
    assert!(offset_of!(Signal, ts_cause) == layout::OFF_TS_CAUSE);
    assert!(<Signal as Record>::SIZE == size_of::<Signal>());
};

impl Record for Signal {
    const NAME: &'static str = "Signal";
    const SIZE: usize = layout::SIGNAL_SIZE;
    const SCHEMA_VERSION: u8 = layout::SCHEMA_VERSION;
    const KIND: u32 = layout::RING_KIND_SIGNAL;
}

impl Signal {
    /// An all-zero signal (all fields default to `0`, kind = target-position).
    #[inline]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }

    /// Construct a target-position signal, stamping `schema_version` and `kind`.
    ///
    /// [`max_order_age_ms`](Self::max_order_age_ms) is deliberately **not** an
    /// argument: it defaults to `0`, which already means "defer to the operator's
    /// ceiling", so every existing call site keeps the behaviour it had. A producer
    /// that wants a specific order lifetime sets the field on the result — and a
    /// tenth positional `u32` beside `ttl_ms` and `model_version` is a transposition
    /// waiting to happen between two fields that are both durations and mean
    /// completely different things.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn target_position(
        seq: u64,
        ts_event: i64,
        symbol_id: u32,
        target_qty: i64,
        urgency: u8,
        price_band: i64,
        ttl_ms: u32,
        model_version: u32,
        flags: u16,
    ) -> Self {
        Self {
            seq,
            ts_event,
            target_qty,
            price_band,
            symbol_id,
            ttl_ms,
            model_version,
            flags,
            schema_version: SCHEMA_VERSION,
            urgency,
            kind: KIND_TARGET_POSITION,
            pad0: [0u8; 3],
            max_order_age_ms: 0,
            // `0` = "not stated", for the same reason `max_order_age_ms` defaults to it:
            // every existing call site keeps the behaviour it had, and a producer that
            // cannot name what caused a decision must not be made to invent one.
            ts_cause: 0,
        }
    }

    /// Set how long an order this signal places may keep its place at the venue.
    ///
    /// Chained rather than an argument to [`target_position`](Self::target_position);
    /// see that constructor for why.
    #[inline]
    pub fn with_max_order_age_ms(mut self, ms: u32) -> Self {
        self.max_order_age_ms = ms;
        self
    }

    /// Say what observation this decision answers — a bar's own close, in event-time ns.
    ///
    /// Chained for the same reason as [`with_max_order_age_ms`](Self::with_max_order_age_ms):
    /// a tenth positional argument beside two other `i64` timestamps is a transposition
    /// waiting to happen, and this one would transpose silently — a `ts_cause` in the
    /// `ts_event` slot is a plausible number that ages the record against the wrong
    /// clock.
    #[inline]
    pub fn with_ts_cause(mut self, ts_cause: i64) -> Self {
        self.ts_cause = ts_cause;
        self
    }

    /// How long this decision took to answer its cause, in nanoseconds — or `None` when
    /// the producer stated no cause, or stated one **after** the decision.
    ///
    /// A cause in the future is refused rather than clamped to zero. It is a producer
    /// that has confused the two stamps, or two clocks far enough apart to matter, and
    /// either way "0 ms" would be the most reassuring possible report of it.
    #[inline]
    pub fn cause_age_ns(&self) -> Option<i64> {
        if self.ts_cause == 0 || self.ts_cause > self.ts_event {
            return None;
        }
        Some(self.ts_event - self.ts_cause)
    }

    /// Zero-copy view of the record as its 64 wire bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Decode exactly `SIGNAL_SIZE` wire bytes into a `Signal`.
    ///
    /// Uses an **unaligned** read, so a byte buffer of any alignment (e.g. a stack
    /// `[u8; 64]`, which is only 1-aligned) is safe — `bytemuck::from_bytes` would
    /// panic on such a buffer because `Signal` is 8-aligned.
    ///
    /// # Panics
    /// If `bytes.len() != SIGNAL_SIZE`.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bytemuck::pod_read_unaligned(bytes)
    }

    /// True if the reduce-only flag is set.
    #[inline]
    pub fn is_reduce_only(&self) -> bool {
        self.flags & FLAG_REDUCE_ONLY != 0
    }

    /// True if the close/flatten flag is set (`target_qty` is then ignored).
    #[inline]
    pub fn is_close(&self) -> bool {
        self.flags & FLAG_CLOSE != 0
    }
}

/// One market-data update crossing the Rust→Python boundary.
///
/// The Rust core owns the venue connection and the book; this is the slice of that
/// state a Python feature computation needs, published per update so Python never
/// opens its own connection — two connections would mean features computed on a
/// book the executing core never saw, and the parity harness could not tell that
/// apart from a model change.
///
/// Every slice carries the **full** current state (BBO + last print + the venue's
/// mark and funding), not a delta: a consumer that falls behind and skips records
/// still sees a coherent snapshot rather than having to replay everything since the
/// last full update. `kind` says which part just moved. Layout is fixed,
/// little-endian, and padding-free (asserted below); prices and sizes are
/// fixed-point `10^-FIXED_POINT_DECIMALS` integers.
///
/// The mark/funding tail (schema version 2, ADR-0028) rides this record rather than
/// having one of its own because a ticker has no event time to stamp a record with:
/// Hyperliquid's `activeAssetCtx` carries no venue timestamp (ADR-0011), so a
/// ticker-triggered record could only be ordered on our receipt clock, and a record
/// ordered on the machine's clock does not reproduce on replay. Mark and funding are
/// state; they ride the next slice a venue-timed event triggers, carrying their own
/// two clocks so a consumer can tell how stale they are and which clock said so.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MdSlice {
    /// Monotonic sequence number (gap detection + offline replay).
    pub seq: u64,
    /// Event-time nanoseconds of this update — never wall-clock at receipt.
    pub ts_event: i64,
    /// Best bid price (fixed-point).
    pub bid_px: i64,
    /// Size resting at the best bid (fixed-point).
    pub bid_sz: i64,
    /// Best ask price (fixed-point).
    pub ask_px: i64,
    /// Size resting at the best ask (fixed-point).
    pub ask_sz: i64,
    /// Price of the last trade print (fixed-point); `0` if none yet.
    pub last_trade_px: i64,
    /// Size of the last trade print (fixed-point); `0` if none yet.
    pub last_trade_sz: i64,
    /// The last print's **own** event time; `0` if none yet. Distinct from
    /// `ts_event` so a feature can measure how stale the print is.
    pub last_trade_ts: i64,
    /// Canonical symbol id (resolved via the venue adapter's `SymbolMap`).
    pub symbol_id: u32,
    /// Bitfield: see [`MD_FLAG_LAST_TRADE_SELL`].
    pub flags: u16,
    /// Layout version guard; readers reject unknown values loudly.
    pub schema_version: u8,
    /// What caused this update: [`MD_KIND_QUOTE`], [`MD_KIND_TRADE`],
    /// [`MD_KIND_SNAPSHOT`].
    pub kind: u8,
    /// The venue's mark price (fixed-point); `0` if no ticker has been seen.
    ///
    /// What margin, liquidation and unrealized PnL are computed against — a
    /// different quantity from the mid and from the last print, and the only one
    /// the venue will actually liquidate against.
    pub mark_px: i64,
    /// The external spot reference the venue tracks (Hyperliquid's *oracle* price);
    /// `0` if the instrument has none. `mark_px - index_px` is the basis.
    pub index_px: i64,
    /// Fraction of position notional charged once per
    /// [`funding_interval_ns`](Self::funding_interval_ns), fixed-point. Positive
    /// means longs pay shorts. Exactly as the venue published it, never rescaled.
    pub funding_rate: i64,
    /// Nanoseconds between funding charges; `0` means the instrument does not fund
    /// (or the interval is unknown).
    ///
    /// Travels with the rate because either one alone is a wrong number: venues
    /// disagree on the period and publish the rate with no unit attached, so a
    /// bare rate makes an expected-carry calculation silently eight times wrong the
    /// day a second venue is added.
    pub funding_interval_ns: i64,
    /// The ticker's **own** venue event time; `0` when the venue stamped none.
    ///
    /// When non-zero it is on the same clock as [`ts_event`](Self::ts_event), so
    /// `ts_event - mark_ts_venue` is the mark's age and it replays. When zero there
    /// is no reproducible age to compute — see [`mark_ts_ingest`](Self::mark_ts_ingest).
    pub mark_ts_venue: i64,
    /// When this process received the ticker; `0` if none has been seen.
    ///
    /// Kept beside `mark_ts_venue` rather than folded into one field (ADR-0011):
    /// the pair is what distinguishes "no ticker at all" from "a ticker the venue
    /// did not timestamp", and where both exist their difference is the feed
    /// latency. One field would make those two cases look identical, and a consumer
    /// would have no way to know its mark age came from a wall clock.
    pub mark_ts_ingest: i64,
}

// ── Compile-time layout guard ────────────────────────────────────────────────
const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<MdSlice>() == layout::MD_SLICE_SIZE);
    assert!(align_of::<MdSlice>() == layout::MD_SLICE_ALIGN);
    assert!(offset_of!(MdSlice, seq) == layout::MD_OFF_SEQ);
    assert!(offset_of!(MdSlice, ts_event) == layout::MD_OFF_TS_EVENT);
    assert!(offset_of!(MdSlice, bid_px) == layout::MD_OFF_BID_PX);
    assert!(offset_of!(MdSlice, bid_sz) == layout::MD_OFF_BID_SZ);
    assert!(offset_of!(MdSlice, ask_px) == layout::MD_OFF_ASK_PX);
    assert!(offset_of!(MdSlice, ask_sz) == layout::MD_OFF_ASK_SZ);
    assert!(offset_of!(MdSlice, last_trade_px) == layout::MD_OFF_LAST_TRADE_PX);
    assert!(offset_of!(MdSlice, last_trade_sz) == layout::MD_OFF_LAST_TRADE_SZ);
    assert!(offset_of!(MdSlice, last_trade_ts) == layout::MD_OFF_LAST_TRADE_TS);
    assert!(offset_of!(MdSlice, symbol_id) == layout::MD_OFF_SYMBOL_ID);
    assert!(offset_of!(MdSlice, flags) == layout::MD_OFF_FLAGS);
    assert!(offset_of!(MdSlice, schema_version) == layout::MD_OFF_SCHEMA_VERSION);
    assert!(offset_of!(MdSlice, kind) == layout::MD_OFF_KIND);
    assert!(offset_of!(MdSlice, mark_px) == layout::MD_OFF_MARK_PX);
    assert!(offset_of!(MdSlice, index_px) == layout::MD_OFF_INDEX_PX);
    assert!(offset_of!(MdSlice, funding_rate) == layout::MD_OFF_FUNDING_RATE);
    assert!(offset_of!(MdSlice, funding_interval_ns) == layout::MD_OFF_FUNDING_INTERVAL_NS);
    assert!(offset_of!(MdSlice, mark_ts_venue) == layout::MD_OFF_MARK_TS_VENUE);
    assert!(offset_of!(MdSlice, mark_ts_ingest) == layout::MD_OFF_MARK_TS_INGEST);
    assert!(<MdSlice as Record>::SIZE == size_of::<MdSlice>());
    // The two records must not become stride-compatible by accident: if they ever
    // do, `record_size` stops telling the two rings apart and only `record_kind`
    // does. That is exactly why `record_kind` exists — this assert is a tripwire,
    // not a requirement (delete it and the kind check still holds). `MdBar` shares
    // this stride *on purpose*, which is the same argument arrived at from the
    // other side.
    assert!(layout::MD_SLICE_SIZE != layout::SIGNAL_SIZE);
};

impl Record for MdSlice {
    const NAME: &'static str = "MdSlice";
    const SIZE: usize = layout::MD_SLICE_SIZE;
    const SCHEMA_VERSION: u8 = layout::MD_SLICE_SCHEMA_VERSION;
    const KIND: u32 = layout::RING_KIND_MD_SLICE;
}

impl MdSlice {
    /// An all-zero slice (no quote, no print, kind = quote).
    #[inline]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }

    /// A slice for `symbol_id` at `ts_event`, stamping `schema_version` and `kind`.
    ///
    /// Quote, trade and ticker state are filled in by [`with_bbo`](Self::with_bbo),
    /// [`with_last_trade`](Self::with_last_trade) and
    /// [`with_ticker`](Self::with_ticker) — a single constructor taking fifteen
    /// same-typed `i64`s is a transposition waiting to happen.
    #[inline]
    pub fn new(seq: u64, ts_event: i64, symbol_id: u32, kind: u8) -> Self {
        Self {
            seq,
            ts_event,
            symbol_id,
            kind,
            schema_version: MD_SLICE_SCHEMA_VERSION,
            ..Self::zeroed()
        }
    }

    /// Set the best bid/offer.
    #[inline]
    pub fn with_bbo(mut self, bid_px: i64, bid_sz: i64, ask_px: i64, ask_sz: i64) -> Self {
        self.bid_px = bid_px;
        self.bid_sz = bid_sz;
        self.ask_px = ask_px;
        self.ask_sz = ask_sz;
        self
    }

    /// Set the last trade print. `ts` is the print's own event time.
    #[inline]
    pub fn with_last_trade(mut self, px: i64, sz: i64, ts: i64, is_sell: bool) -> Self {
        self.last_trade_px = px;
        self.last_trade_sz = sz;
        self.last_trade_ts = ts;
        if is_sell {
            self.flags |= MD_FLAG_LAST_TRADE_SELL;
        } else {
            self.flags &= !MD_FLAG_LAST_TRADE_SELL;
        }
        self
    }

    /// Set the ticker tail: mark, index, funding and the mark's two clocks.
    ///
    /// Taken as one call rather than six setters because the six fields are only
    /// meaningful together — a mark with no time cannot be aged, and a funding rate
    /// with no interval is a wrong number rather than a partial one. `funding` is
    /// `(rate, interval_ns)`; pass `(0, 0)` for an instrument that does not fund.
    #[inline]
    pub fn with_ticker(
        mut self,
        mark_px: i64,
        index_px: i64,
        funding: (i64, i64),
        ts_venue: i64,
        ts_ingest: i64,
    ) -> Self {
        self.mark_px = mark_px;
        self.index_px = index_px;
        self.funding_rate = funding.0;
        self.funding_interval_ns = funding.1;
        self.mark_ts_venue = ts_venue;
        self.mark_ts_ingest = ts_ingest;
        self
    }

    /// True if this slice carries a ticker at all.
    ///
    /// Keyed on `mark_ts_ingest`, never on `mark_px`: our receipt clock is the one
    /// field a ticker always supplies, whereas a zero mark is indistinguishable
    /// from an instrument the venue marks at zero.
    #[inline]
    pub fn has_ticker(&self) -> bool {
        self.mark_ts_ingest != 0
    }

    /// True if the carried mark is stamped with the **venue's** own time.
    ///
    /// When false the only time this mark has is our receipt clock, so any age
    /// derived from it measures the machine and does not replay (ADR-0011).
    #[inline]
    pub fn mark_is_venue_timed(&self) -> bool {
        self.mark_ts_venue != 0
    }

    /// True if the last print's aggressor was the seller.
    #[inline]
    pub fn last_trade_is_sell(&self) -> bool {
        self.flags & MD_FLAG_LAST_TRADE_SELL != 0
    }

    /// Zero-copy view of the record as its 128 wire bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Decode exactly `MD_SLICE_SIZE` wire bytes into an `MdSlice`.
    ///
    /// Unaligned read, for the same reason as [`Signal::from_bytes`].
    ///
    /// # Panics
    /// If `bytes.len() != MD_SLICE_SIZE`.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bytemuck::pod_read_unaligned(bytes)
    }
}

/// One **closed** OHLCV bar crossing the Rust→Python boundary (ADR-0028).
///
/// A second market-data record, on a second ring, because a bar is not a state
/// snapshot the way [`MdSlice`] is. A slice answers *what is true now*; a bar
/// answers *what happened over a closed interval*, and the difference has teeth:
/// two consecutive bars with identical OHLCV are two facts, not a repeat, so the
/// publisher's change test — which exists to keep a busy book from spending every
/// ring slot on records that carry no news — must never see one. A bar folded into
/// `MdSlice` as another `kind` would inherit that test, and a coalesced bar silently
/// shortens every rolling feature window downstream.
///
/// **`ts_event` is the bar's close, at `T + 1 ms`.** The venue's `T` is the bar's
/// *last* millisecond, so a bar stamped `T` sorts equal to every trade printed
/// inside it and an event-time sort may hand a strategy the closed bar before the
/// tick that closed it. `axon.strategies.data.CLOSE_STAMP_OFFSET_MS` and
/// `decode_candle` already agree on this; the agreement is load-bearing, because two
/// halves one millisecond apart make `align_by_event_time` intersect to *nothing*
/// and the feature-parity gate then fails as "an empty matrix proves nothing".
///
/// **Only closed bars are published, and there is no finality bit.** A forming bar's
/// `close` is a mid-bar price stamped with a close time in the future — the purest
/// lookahead available. The publisher enforces closure structurally (it emits a bar
/// only once the venue starts the next interval) rather than believing a frame that
/// claims to be final: 111 of 112 measured Binance `kline_1m` frames were in progress
/// and *all* carried the same close stamp, so the last partial and the close cannot be
/// told apart by timestamp, and `axon_core::Candle` has no `is_final` to ask. A bit
/// here would be a field a reader could rely on and a publisher could only guess at.
///
/// The stride is 128 bytes, the *same* as [`MdSlice`]: deliberately, and it is the
/// case ADR-0012 §3 built the ring's `record_kind` tag for. `record_size` cannot
/// tell these two apart, so the tag is the only thing that can.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MdBar {
    /// Monotonic sequence number, independent of the slice ring's.
    pub seq: u64,
    /// The bar's **close** time in nanoseconds — `T + 1 ms`, never the open time.
    pub ts_event: i64,
    /// The bar's own open time in nanoseconds.
    pub open_time: i64,
    /// First traded price in the interval (fixed-point).
    pub open: i64,
    /// Highest traded price in the interval (fixed-point).
    pub high: i64,
    /// Lowest traded price in the interval (fixed-point).
    pub low: i64,
    /// Last traded price in the interval (fixed-point).
    pub close: i64,
    /// Base-unit volume traded in the interval (fixed-point).
    pub volume: i64,
    /// Canonical symbol id (resolved via the venue adapter's `SymbolMap`).
    pub symbol_id: u32,
    /// Bar length in milliseconds (`60_000` for 1m). A number rather than an enum
    /// ordinal, so a consumer can compute the next expected `open_time` directly.
    pub interval_ms: u32,
    /// Bitfield: see [`MD_BAR_FLAG_GAP_BEFORE`], [`MD_BAR_FLAG_FIRST_BAR`].
    pub flags: u16,
    /// Layout version guard; readers reject unknown values loudly.
    pub schema_version: u8,
    /// Record kind discriminant; see [`MD_BAR_KIND_CLOSED`].
    pub kind: u8,
    /// Padding to 128 bytes. MUST be zero; reserved for future fields.
    pub reserved: [u8; 52],
}

// ── Compile-time layout guard ────────────────────────────────────────────────
const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<MdBar>() == layout::MD_BAR_SIZE);
    assert!(align_of::<MdBar>() == layout::MD_BAR_ALIGN);
    assert!(offset_of!(MdBar, seq) == layout::MD_BAR_OFF_SEQ);
    assert!(offset_of!(MdBar, ts_event) == layout::MD_BAR_OFF_TS_EVENT);
    assert!(offset_of!(MdBar, open_time) == layout::MD_BAR_OFF_OPEN_TIME);
    assert!(offset_of!(MdBar, open) == layout::MD_BAR_OFF_OPEN);
    assert!(offset_of!(MdBar, high) == layout::MD_BAR_OFF_HIGH);
    assert!(offset_of!(MdBar, low) == layout::MD_BAR_OFF_LOW);
    assert!(offset_of!(MdBar, close) == layout::MD_BAR_OFF_CLOSE);
    assert!(offset_of!(MdBar, volume) == layout::MD_BAR_OFF_VOLUME);
    assert!(offset_of!(MdBar, symbol_id) == layout::MD_BAR_OFF_SYMBOL_ID);
    assert!(offset_of!(MdBar, interval_ms) == layout::MD_BAR_OFF_INTERVAL_MS);
    assert!(offset_of!(MdBar, flags) == layout::MD_BAR_OFF_FLAGS);
    assert!(offset_of!(MdBar, schema_version) == layout::MD_BAR_OFF_SCHEMA_VERSION);
    assert!(offset_of!(MdBar, kind) == layout::MD_BAR_OFF_KIND);
    assert!(offset_of!(MdBar, reserved) == layout::MD_BAR_OFF_RESERVED);
    assert!(<MdBar as Record>::SIZE == size_of::<MdBar>());
};

impl Record for MdBar {
    const NAME: &'static str = "MdBar";
    const SIZE: usize = layout::MD_BAR_SIZE;
    const SCHEMA_VERSION: u8 = layout::MD_BAR_SCHEMA_VERSION;
    const KIND: u32 = layout::RING_KIND_MD_BAR;
}

impl MdBar {
    /// An all-zero bar (kind = closed, no flags).
    #[inline]
    pub fn zeroed() -> Self {
        Zeroable::zeroed()
    }

    /// A closed bar, stamping `schema_version` and `kind`.
    ///
    /// `ts_event` must be the bar's **close** (`T + 1 ms`), never its open; the two
    /// are one field apart and confusing them is a whole-bar lookahead leak that no
    /// downstream check can see. OHLCV is filled by [`with_ohlcv`](Self::with_ohlcv)
    /// so five same-typed `i64`s cannot be transposed at the call site.
    #[inline]
    pub fn new(
        seq: u64,
        ts_event: i64,
        open_time: i64,
        symbol_id: u32,
        interval_ms: u32,
        flags: u16,
    ) -> Self {
        Self {
            seq,
            ts_event,
            open_time,
            symbol_id,
            interval_ms,
            flags,
            schema_version: MD_BAR_SCHEMA_VERSION,
            kind: MD_BAR_KIND_CLOSED,
            ..Self::zeroed()
        }
    }

    /// Set the bar's open/high/low/close/volume.
    #[inline]
    pub fn with_ohlcv(mut self, open: i64, high: i64, low: i64, close: i64, volume: i64) -> Self {
        self.open = open;
        self.high = high;
        self.low = low;
        self.close = close;
        self.volume = volume;
        self
    }

    /// True if the venue should have printed a bar between this one and the previous
    /// one for the same instrument and interval, and did not.
    #[inline]
    pub fn has_gap_before(&self) -> bool {
        self.flags & MD_BAR_FLAG_GAP_BEFORE != 0
    }

    /// True if this is the first bar seen for its instrument and interval, so
    /// continuity with anything earlier is **unknown** rather than broken.
    #[inline]
    pub fn is_first_bar(&self) -> bool {
        self.flags & MD_BAR_FLAG_FIRST_BAR != 0
    }

    /// Zero-copy view of the record as its 128 wire bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Decode exactly `MD_BAR_SIZE` wire bytes into an `MdBar`.
    ///
    /// Unaligned read, for the same reason as [`Signal::from_bytes`].
    ///
    /// # Panics
    /// If `bytes.len() != MD_BAR_SIZE`.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bytemuck::pod_read_unaligned(bytes)
    }
}

/// Convert a real value to the fixed-point wire integer.
///
/// Convenience for tests/tooling. Production code on the Rust side should carry
/// exact fixed-point (`rust_decimal` / integer ticks), not `f64` — see
/// `crates/README.md`.
#[inline]
pub fn to_fixed(real: f64) -> i64 {
    (real * FIXED_POINT_SCALE as f64).round() as i64
}

/// Inverse of [`to_fixed`] (lossy; for tooling/inspection only).
#[inline]
pub fn from_fixed(fixed: i64) -> f64 {
    fixed as f64 / FIXED_POINT_SCALE as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_is_one_cache_line() {
        assert_eq!(core::mem::size_of::<Signal>(), 64);
        assert_eq!(SIGNAL_SIZE, 64);
        assert_eq!(SIGNAL_ALIGN, 8);
    }

    #[test]
    fn byte_roundtrip_is_lossless() {
        let s = Signal::target_position(
            7,
            1_700_000_000_000_000_000,
            3,
            -125_000_000,
            2,
            0,
            500,
            42,
            FLAG_REDUCE_ONLY,
        );
        let bytes = s.as_bytes();
        assert_eq!(bytes.len(), SIGNAL_SIZE);
        let back = Signal::from_bytes(bytes);
        assert_eq!(s, back);
        assert!(back.is_reduce_only());
        assert!(!back.is_close());
        assert_eq!(back.kind, KIND_TARGET_POSITION);
        assert_eq!(back.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn from_bytes_handles_unaligned_buffer() {
        // A `Signal` is 8-aligned; decoding from a 1-aligned slice must not panic.
        let s = Signal::target_position(9, 123, 4, -7, 1, 0, 250, 3, FLAG_CLOSE);
        let mut buf = [0u8; SIGNAL_SIZE + 1];
        buf[1..].copy_from_slice(s.as_bytes()); // start at offset 1 → not 8-aligned
        assert_eq!(Signal::from_bytes(&buf[1..]), s);
    }

    #[test]
    fn every_optional_field_is_unstated_rather_than_dirty() {
        // The named padding must be zero because a reader refuses a record whose padding
        // is not — that is a producer writing a field this build does not know about, and
        // a constructor that left it dirty would make every signal it built rejected at
        // the far end. The two optional fields must be zero because zero is what "the
        // producer said nothing" *means* for both of them, and a plausible non-zero
        // default would be a duration or a timestamp nobody chose.
        //
        // `reserved` is gone: `ts_cause` spent the last of it at schema version 3, so
        // there is no unnamed byte left in this record and the next field has to re-cut
        // the layout rather than extend it.
        let s = Signal::target_position(1, 1, 1, 1, 0, 0, 0, 0, 0);
        assert_eq!(s.pad0, [0u8; 3]);
        assert_eq!(s.max_order_age_ms, 0, "0 = defer to the operator's ceiling");
        assert_eq!(s.ts_cause, 0, "0 = no cause stated");
        assert_eq!(s.cause_age_ns(), None);
    }

    #[test]
    fn a_stated_cause_measures_the_gap_the_producers_transcript_used_to_own_alone() {
        // The largest latency in the system: 951 / 12 051 / 111 475 ms from an m1 bar's
        // close to the strategy acting on it, measured live on 2026-07-27 and visible
        // only in the producer's private transcript. On the wire it can have a ceiling.
        let bar_close = 1_000_000_000i64;
        let decided = bar_close + 12_051_000_000; // the measured median
        let s =
            Signal::target_position(1, decided, 3, 0, 0, 0, 60_000, 1, 0).with_ts_cause(bar_close);
        assert_eq!(s.cause_age_ns(), Some(12_051_000_000));
        assert_eq!(Signal::from_bytes(s.as_bytes()), s, "and it round-trips");
    }

    #[test]
    fn a_cause_in_the_future_is_refused_rather_than_reported_as_instant() {
        // A producer that has transposed the two stamps, or two clocks far enough apart
        // to matter. Clamping to zero would report the most reassuring possible number
        // for exactly the record whose stamps cannot be trusted — and `0 ms` on the
        // stage whose whole point is that it is measured in *seconds* is the one value an
        // operator would never question.
        let s = Signal::target_position(1, 1_000, 3, 0, 0, 0, 0, 0, 0).with_ts_cause(2_000);
        assert_eq!(s.cause_age_ns(), None);
        // Exactly simultaneous is legal and is zero: a decision on the tick that caused
        // it is a real thing, and it is not a clock error.
        let s = Signal::target_position(1, 1_000, 3, 0, 0, 0, 0, 0, 0).with_ts_cause(1_000);
        assert_eq!(s.cause_age_ns(), Some(0));
    }

    #[test]
    fn an_order_lifetime_is_a_different_field_from_the_admission_window() {
        // `ttl_ms` is consumed by the reader before the planner sees the record and is
        // clamped against the operator's `max_signal_age_ms`; it has never applied to an
        // order already resting at the venue. Carrying one number for both would mean a
        // strategy could not ask for a resting order at all without also asking to be
        // admitted for the same span.
        let s =
            Signal::target_position(1, 1, 1, 1, 0, 0, 60_000, 0, 0).with_max_order_age_ms(5_000);
        assert_eq!(s.ttl_ms, 60_000);
        assert_eq!(s.max_order_age_ms, 5_000);
        assert_eq!(Signal::from_bytes(s.as_bytes()), s);
    }

    #[test]
    fn zeroed_is_target_position_kind() {
        let s = Signal::zeroed();
        assert_eq!(s.kind, 0);
        assert_eq!(s.kind, KIND_TARGET_POSITION);
    }

    #[test]
    fn fixed_point_helpers() {
        assert_eq!(FIXED_POINT_SCALE, 100_000_000);
        assert_eq!(FIXED_POINT_DECIMALS, 8);
        assert_eq!(to_fixed(1.5), 150_000_000);
        assert_eq!(from_fixed(150_000_000), 1.5);
        assert_eq!(to_fixed(-0.25), -25_000_000);
    }

    fn md(i: u64) -> MdSlice {
        MdSlice::new(i, 1_700_000_000_000_000_000 + i as i64, 3, MD_KIND_TRADE)
            .with_bbo(
                to_fixed(50_000.5),
                to_fixed(1.25),
                to_fixed(50_001.0),
                to_fixed(0.5),
            )
            .with_last_trade(
                to_fixed(50_000.75),
                to_fixed(0.1),
                1_699_999_999_000_000_000,
                true,
            )
            .with_ticker(
                to_fixed(50_000.25),
                to_fixed(49_998.0),
                (to_fixed(0.0000125), 3_600_000_000_000),
                0, // Hyperliquid stamps no venue time on its ticker
                1_699_999_998_000_000_000,
            )
    }

    #[test]
    fn md_slice_is_two_cache_lines() {
        assert_eq!(core::mem::size_of::<MdSlice>(), 128);
        assert_eq!(MD_SLICE_SIZE, 128);
        assert_eq!(MD_SLICE_ALIGN, 8);
    }

    #[test]
    fn md_byte_roundtrip_is_lossless() {
        let s = md(11);
        let bytes = s.as_bytes();
        assert_eq!(bytes.len(), MD_SLICE_SIZE);
        let back = MdSlice::from_bytes(bytes);
        assert_eq!(s, back);
        assert!(back.last_trade_is_sell());
        assert_eq!(back.kind, MD_KIND_TRADE);
        assert_eq!(back.schema_version, MD_SLICE_SCHEMA_VERSION);
        assert_eq!(back.mark_px, to_fixed(50_000.25));
        assert_eq!(back.funding_interval_ns, 3_600_000_000_000);
    }

    #[test]
    fn a_slice_with_no_ticker_is_distinguishable_from_one_marked_at_zero() {
        // `mark_px == 0` cannot answer "has a ticker been seen?" — a venue is free to
        // mark an instrument at zero, and a consumer that keyed on the price would
        // read a real mark as an absent one. The receipt clock is the field a ticker
        // always supplies, so it is the one the sentinel is built on.
        let none = MdSlice::new(1, 10, 3, MD_KIND_QUOTE);
        assert!(!none.has_ticker());
        let marked_at_zero = none.with_ticker(0, 0, (0, 0), 0, 99);
        assert!(marked_at_zero.has_ticker());
        assert!(!marked_at_zero.mark_is_venue_timed());
    }

    #[test]
    fn a_venue_timed_mark_is_distinguishable_from_a_receipt_timed_one() {
        // The whole reason ADR-0011 keeps the two clocks apart: an age computed from a
        // receipt stamp measures this machine and does not replay, and a consumer that
        // could not tell the two apart would build a feature on it anyway.
        let hl = MdSlice::new(1, 10, 3, MD_KIND_QUOTE).with_ticker(1, 2, (3, 4), 0, 7);
        assert!(hl.has_ticker());
        assert!(!hl.mark_is_venue_timed());
        let stamped = hl.with_ticker(1, 2, (3, 4), 6, 7);
        assert!(stamped.mark_is_venue_timed());
    }

    fn bar(i: u64) -> MdBar {
        // 1m bars, closed at T+1ms. `open_time` is a whole minute; `ts_event` is the
        // instant after that minute's last millisecond.
        let open_time = 1_700_000_000_000_000_000 + (i as i64) * 60_000_000_000;
        MdBar::new(
            i,
            open_time + 60_000_000_000,
            open_time,
            3,
            60_000,
            if i == 0 { MD_BAR_FLAG_FIRST_BAR } else { 0 },
        )
        .with_ohlcv(
            to_fixed(50_000.0),
            to_fixed(50_010.0),
            to_fixed(49_990.0),
            to_fixed(50_005.0),
            to_fixed(12.5),
        )
    }

    #[test]
    fn md_bar_is_two_cache_lines() {
        assert_eq!(core::mem::size_of::<MdBar>(), 128);
        assert_eq!(MD_BAR_SIZE, 128);
        assert_eq!(MD_BAR_ALIGN, 8);
    }

    #[test]
    fn md_bar_byte_roundtrip_is_lossless() {
        let b = bar(3);
        let bytes = b.as_bytes();
        assert_eq!(bytes.len(), MD_BAR_SIZE);
        let back = MdBar::from_bytes(bytes);
        assert_eq!(b, back);
        assert_eq!(back.kind, MD_BAR_KIND_CLOSED);
        assert_eq!(back.schema_version, MD_BAR_SCHEMA_VERSION);
        assert_eq!(back.reserved, [0u8; 52]);
        assert_eq!(back.interval_ms, 60_000);
    }

    #[test]
    fn md_bar_from_bytes_handles_unaligned_buffer() {
        let b = bar(2);
        let mut buf = [0u8; MD_BAR_SIZE + 1];
        buf[1..].copy_from_slice(b.as_bytes()); // start at offset 1 → not 8-aligned
        assert_eq!(MdBar::from_bytes(&buf[1..]), b);
    }

    #[test]
    fn a_bar_closes_one_millisecond_after_its_last_millisecond() {
        // The stamp both languages agree on. A bar stamped at its venue `T` sorts
        // equal to the trades printed inside its own final millisecond, and the two
        // halves being 1 ms apart makes `align_by_event_time` intersect to nothing.
        let b = bar(0);
        assert_eq!(
            b.ts_event - b.open_time,
            (b.interval_ms as i64) * 1_000_000,
            "close is open + one whole interval, i.e. venue T + 1 ms"
        );
    }

    #[test]
    fn the_continuity_flags_say_different_things() {
        // "I have no history for this instrument" and "the venue skipped a bar" are
        // different facts. Collapsing them would either cry gap on every session's
        // first bar or let a feature window start mid-history in silence.
        assert!(bar(0).is_first_bar());
        assert!(!bar(0).has_gap_before());
        let holed = MdBar::new(9, 20, 10, 3, 60_000, MD_BAR_FLAG_GAP_BEFORE);
        assert!(holed.has_gap_before());
        assert!(!holed.is_first_bar());
        assert_ne!(MD_BAR_FLAG_GAP_BEFORE, MD_BAR_FLAG_FIRST_BAR);
    }

    #[test]
    fn a_bar_and_a_slice_share_a_stride_so_only_the_kind_tag_tells_them_apart() {
        // Not a defect — it is ADR-0012 §3's argument made concrete. The moment these
        // two strides matched, `record_size` stopped being able to identify a ring and
        // `record_kind` became the only check standing between a reader and decoding
        // an open price as a bid.
        assert_eq!(<MdBar as Record>::SIZE, <MdSlice as Record>::SIZE);
        assert_ne!(<MdBar as Record>::KIND, <MdSlice as Record>::KIND);
        assert_ne!(<MdBar as Record>::KIND, <Signal as Record>::KIND);
        assert_ne!(<MdBar as Record>::KIND, 0, "0 means `never stamped`");
    }

    #[test]
    fn md_from_bytes_handles_unaligned_buffer() {
        let s = md(2);
        let mut buf = [0u8; MD_SLICE_SIZE + 1];
        buf[1..].copy_from_slice(s.as_bytes()); // start at offset 1 → not 8-aligned
        assert_eq!(MdSlice::from_bytes(&buf[1..]), s);
    }

    #[test]
    fn a_buy_print_clears_the_sell_flag_a_previous_one_set() {
        // The builder is chained onto a reused slice in the publisher, so
        // `with_last_trade` must *assign* the side, not OR it in — otherwise a
        // symbol's aggressor flag latches to "sell" after its first sell print.
        let s = md(1).with_last_trade(1, 1, 1, false);
        assert!(!s.last_trade_is_sell());
    }

    #[test]
    fn the_two_records_are_distinguishable_by_kind() {
        assert_ne!(
            <Signal as Record>::KIND,
            <MdSlice as Record>::KIND,
            "a shared kind would let a consumer open the wrong ring"
        );
        assert_ne!(<Signal as Record>::KIND, 0, "0 means `never stamped`");
        assert_ne!(<MdSlice as Record>::KIND, 0, "0 means `never stamped`");
    }

    #[test]
    fn ring_constants_present() {
        assert_eq!(RING_HEADER_SIZE, 192);
        assert_eq!(RING_CACHE_LINE, 64);
        assert_eq!(RING_VERSION, 2);
        // "AXONRING" as little-endian bytes.
        assert_eq!(&RING_MAGIC.to_le_bytes(), b"AXONRING");
    }
}
