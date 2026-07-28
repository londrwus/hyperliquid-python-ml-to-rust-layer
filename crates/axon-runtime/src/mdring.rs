//! The market-data ring's **publisher** — the Rust→Python direction of the boundary
//! (ADR-0012), finally wired to the core's own event stream.
//!
//! ADR-0012 built the record, the transport and the Python reader, and its consequences
//! recorded what it had *not* built: the Rust-side publisher, leaving
//! `crates/axon-ipc/examples/md_writer.rs` as the only producer. That left the whole
//! Boundary-B story theoretical — Python could read a ring nothing production ever wrote
//! into, so features could only ever be computed on canned bytes. This is the missing
//! half, and ADR-0012's consequence has been amended to point here.
//!
//! Seven things here are decisions, and each has an answer that looks reasonable right
//! up until it costs something.
//!
//! **1. What triggers a write is a pure function of the event stream, never of the
//! clock.** [`MdWritePolicy`] is explicit and configurable, and both of its variants
//! depend only on the events that arrived and their contents. A wall-clock cadence —
//! "one slice every 5 ms" — is the obvious way to bound the rate, and it silently
//! makes the published stream depend on how fast the machine drained the bus. Replay
//! the same capture on a slower box and Python sees a different series, which is the
//! one thing the Phase-5 parity harness cannot survive: it compares a backtest against
//! a live session event for event, and a resampled feed reads as a model change rather
//! than as a sampling artefact. The same argument the core already makes for ageing
//! marks against `wall_time` only when live (ADR-0013 §2) applies here, and harder,
//! because this stream is an *input* to somebody else's model.
//!
//! **2. It runs on the core thread, inside the existing fan-out.** [`MdProducer`] is
//! lock-free and allocation-free on the publish path: `try_push` is an Acquire load, a
//! `copy_nonoverlapping` into a slot the consumer provably does not hold, and a Release
//! store. There is no `await`, no lock and no allocator call, so it cannot stall the
//! deterministic loop — which is what makes it safe to sit next to the order tracker.
//! The publish happens **last** in [`CoreHandler`](crate::handler::CoreHandler)'s
//! fan-out, after the book, the marks and the tracker: a slice is a statement about the
//! state the core holds at `ts_event`, and one emitted mid-fan-out would describe a
//! market the core itself had not finished forming. Publishing after the tracker also
//! keeps the ring write out of the tracker's critical section, which the submit path's
//! risk context contends for from the async edge.
//!
//! **3. A full ring drops the newest slice, and the drop is counted twice over.**
//! Dropping is correct — every slice is full state, so the newest supersedes the old
//! (ADR-0012 §2) — and stalling the execution core on a slow feature computation is the
//! one failure this direction must never have. But a silently-dropping feed and a
//! healthy one look identical from both sides, so: `seq` is spent on every *attempted*
//! push, which leaves exactly the gap `MdRingConsumer.dropped` infers a loss from, and
//! [`MdStats::dropped`] puts the same number on the operator's status line for the case
//! where nobody is reading at all. A coalesced update spends no `seq`, because it was
//! never a record and reporting it to Python as a hole would be a lie in the other
//! direction.
//!
//! **4. A quote too old to price against is not published as a quote.** `MdSlice`
//! carries one timestamp — the event that *triggered* the slice — and nothing that says
//! when the top of book it reports was last true. So a consumer cannot age the bid and
//! ask for itself, and "Python has the `ts_event` to decide" is false for exactly the
//! two fields most features are built from. A `bbo` subscription that a reconnect
//! failed to restore would otherwise be republished minutes later under a current
//! timestamp, on every trade, with `dropped` at zero and nothing on the status line.
//! The publisher therefore applies the planner's own window through
//! [`crate::quote::top_of_book`] and, when nothing still-moving answers, sends the slice
//! with **no quote at all** — zeros, which is the record's own sentinel for "nothing
//! seen yet" — and counts it in [`MdStats::stale_quote`]. Dropping the whole slice
//! instead would throw away the trade print riding on it, and a stale instrument would
//! become indistinguishable from an idle one.
//!
//! **5. Event time, from the event.** A slice is stamped with the `ts_event` the core
//! ordered on, and the last print keeps its own, older time — the distinction ADR-0012
//! §2 built `last_trade_ts` for. [`Ticker`](axon_core::Ticker) still never publishes a
//! slice of its own under *either* policy, because Hyperliquid's `activeAssetCtx`
//! carries no venue timestamp (ADR-0011): a ticker-triggered slice would be stamped
//! with *our receipt clock*, and a replay of the same capture would then publish a
//! different `ts_event` for the same event. That is the market-data ring quietly
//! ceasing to be reproducible. What a ticker now does is fill the record's mark and
//! funding tail (ADR-0028) on the next slice a venue-timed event triggers, carrying
//! *both* of its clocks so a consumer can tell how stale the mark is and which clock
//! measured it.
//!
//! **6. Bars go on their own ring, and only once they are provably closed.**
//! [`Candle`](axon_core::Candle) does not publish a slice either — it moves no field
//! `MdSlice` has — but it does drive [`MdBar`], on a second ring beside the first
//! (ADR-0028). Two things there are decisions rather than plumbing:
//!
//! A venue's `candle` subscription pushes the **forming** bar as it forms, several
//! times a minute, each frame carrying a `close` that is a mid-bar price and a
//! `ts_event` that is a close time in the *future*. Republishing those would be the
//! purest lookahead available and would run the core's event clock ahead of the market.
//! So the publisher closes bars itself, from the stream alone: a held bar is emitted
//! when a frame arrives for the **next** interval, because the venue moving on is the
//! only evidence that the previous bar is final. This is the same rule
//! `axon.strategies.data.closed_rows` applies offline, arrived at without a wall clock
//! — which is what makes the live and research paths agree on *which* bars exist. The
//! cost is stated where it lands: a session's last bar is never published, because
//! nothing ever proved it closed.
//!
//! Note what that rule does *not* rely on. [`Candle`](axon_core::Candle) carries no
//! `is_final`, and neither venue marks one. A measurement of 2 353 real Binance frames
//! found 111 of 112 `kline_1m` frames in progress and **all of them stamped with the
//! same `T`** — so the last partial and the close are indistinguishable by timestamp
//! alone. Hyperliquid republishes the bar it is filling (1 321 frames described 69
//! bars, one 5-minute bar 192 times) and for 65 of those 69 sent **nothing at all after
//! `T`**: it stops republishing and starts the next interval. Keying on `open_time`
//! makes both a non-event: every frame for a bar refines it and none publishes it,
//! whatever the venue does with its close stamp. Finality here is structural rather
//! than declared, which is why [`MdBar`] has no finality bit and `[md_bar.kinds]` has
//! no "forming" value to give one.
//!
//! The bounded cost of having no closing frame to wait for: what gets published is the
//! last frame **observed** before the venue moved on, so a trade printing in the final
//! milliseconds is missing from `volume`. Measured against `candleSnapshot` over seven
//! consecutive BTC minutes — one small sample — six bars matched exactly and one was
//! short by `0.001` in volume, its last frame having arrived 35 ms before the close.
//! The *mode* is the finding, not the rate. ADR-0028's consequences carry it, because
//! that is where somebody diffing a live ring against an offline recompute will look.
//! If `Candle` ever gains `is_final`, it buys exactly one thing: a bar could be
//! published on its own final frame instead of on the next one's arrival, which would
//! close the last-bar gap above. That is an improvement, not a requirement.
//!
//! **7. The beacon beats on the pass, not on the event, and it is the only thing that
//! can tell a quiet market from a dead publisher.** Under [`MdWritePolicy::OnChange`] a
//! flat top of book and a publisher that died write the same empty ring (ADR-0030), and
//! no amount of reading the ring resolves that — the evidence for "the core is still
//! running" is precisely the evidence a stopped core cannot produce. So a third file
//! sits beside the two rings ([`axon_ipc::MdBeacon`], ADR-0034), created by the same
//! switch and derived from the same path, and [`MdPublisher::beat`] is called from
//! [`crate::core::run`]'s loop rather than from [`MdPublisher::on_event`]. A beacon
//! advanced by `on_event` would carry exactly what the ring already carries and would
//! freeze in the one state it exists to describe.
//!
//! The counters it carries are `u32` and **wrap**; the totals stay here on
//! [`MdStats`] and on the status line, which is the side with room for them. A reader
//! takes deltas with `wrapping_sub` and prints none of them as a total — see
//! [`MdPublisher::beat_of`].
//!
//! And [`MdWritePolicy`] does not apply to bars at all. Two consecutive bars with
//! identical OHLCV are two facts, not a repeat; coalescing one away would silently
//! shorten every rolling feature window downstream, and a strategy would compute
//! confidently across a hole it had no way to see. Every closed bar is published
//! exactly once under either policy.

use std::path::{Path, PathBuf};

use axon_contracts::{
    MdBar, MdSlice, MD_BAR_FLAG_FIRST_BAR, MD_BAR_FLAG_GAP_BEFORE, MD_KIND_QUOTE, MD_KIND_SNAPSHOT,
    MD_KIND_TRADE,
};
use axon_core::{Candle, CandleInterval, Event, MarketEvent, Nanos, Side, SymbolId};
use axon_ipc::{beacon_path, MdBarProducer, MdBeacon, MdBeat, MdProducer, RingError};
use axon_marketdata::MarketDataProcessor;
use axon_strategy::decimal_to_fixed;
use serde::{Deserialize, Serialize};

use crate::config::MdRingConfig;
use crate::quote::{top_of_book, TopOfBook, TopState};

/// Bar length in milliseconds — the unit [`MdBar::interval_ms`] carries.
///
/// A `match` rather than arithmetic on the enum's discriminant, so a venue-neutral
/// interval added to [`CandleInterval`] fails to compile here instead of silently
/// publishing bars labelled with somebody else's length. Python's
/// `axon.strategies.data.INTERVAL_MS` is the same table.
fn interval_ms(interval: CandleInterval) -> u32 {
    match interval {
        CandleInterval::M1 => 60_000,
        CandleInterval::M5 => 300_000,
        CandleInterval::M15 => 900_000,
        CandleInterval::H1 => 3_600_000,
        CandleInterval::H4 => 14_400_000,
        CandleInterval::D1 => 86_400_000,
    }
}

/// Where the bar ring lives, given the slice ring's path.
///
/// Derived rather than configured, and the failure that decides it is asymmetric: two
/// independent settings mean an operator can enable the slice ring, forget the bar
/// ring, and get a bar-driven strategy that starts cleanly and then simply never has
/// an opinion — a silence indistinguishable from a quiet market. One switch cannot be
/// half-turned. The convention is the one [`crate::capture`] already uses for its
/// signal log: the sibling name, with `bars` inserted before the extension, and the
/// startup banner prints both paths so nothing about it is implicit at run time.
pub fn bar_ring_path(md_ring_path: &str) -> PathBuf {
    let p = Path::new(md_ring_path);
    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) => p.with_extension(format!("bars.{ext}")),
        None => PathBuf::from(format!("{md_ring_path}.bars")),
    }
}

/// When the publisher writes a slice.
///
/// Both variants are functions of the event stream alone; see the module docs for why
/// there is deliberately no time-based third.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MdWritePolicy {
    /// Write only when the state the record actually carries has moved.
    ///
    /// The default, and it is compression rather than sampling: a suppressed update is
    /// one whose slice would have been byte-identical to the last delivered one except
    /// for `seq` and `ts_event`. `MdSlice` is top-of-book plus the last print (ADR-0012
    /// names that limit), so on an `l2Book` feed most updates move nothing it can
    /// report — and each of those costs a ring slot, which is how the ring comes to be
    /// full at the exact moment something *does* happen.
    #[default]
    OnChange,
    /// Write on every quote, book and trade event for the instrument.
    ///
    /// Higher fidelity in one narrow sense — a consumer can count update arrivals, not
    /// just state changes — at the cost of spending the ring on records that carry no
    /// news. Worth it only when the consumer's features key on update *rate*.
    EveryUpdate,
}

/// What the publisher has counted, for the status line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MdStats {
    /// Slices the ring accepted.
    pub published: u64,
    /// Slices the ring refused because it was full — Python fell behind, or nobody is
    /// reading at all. Each one leaves a hole in `seq` that the consumer also sees.
    pub dropped: u64,
    /// Updates that moved nothing the record carries, under [`MdWritePolicy::OnChange`].
    /// Not a loss and deliberately not a `seq` gap; a rising count next to a flat
    /// `published` means the feed is busy and the top of book is not.
    pub coalesced: u64,
    /// Updates skipped because a price or size could not be held exactly at the wire's
    /// `10^-8` scale. Never zero-filled and never rounded: a rounded price is one the
    /// venue never quoted, and a feature computed from it is wrong in a way nothing
    /// downstream can detect.
    pub unrepresentable: u64,
    /// Slices published with no quote, because every feed carrying one had gone quiet
    /// past the mark window. The record has no field for the quote's own age, so a
    /// consumer cannot tell a stale bid from a live one — this is the number that can.
    pub stale_quote: u64,
    /// Slices published with the mark/funding tail zeroed because some ticker value
    /// could not be held exactly at the wire's `10^-8` scale.
    ///
    /// Zeroed rather than dropping the whole slice, which is the opposite of the rule
    /// for a quote price and deliberately so: the top of book and the last print on the
    /// same record are independently exact, and letting one venue's funding precision
    /// silence the book feed would be a far larger wrong than losing the mark. It is a
    /// lie of omission — Python reads the record's "no ticker yet" sentinel while there
    /// is one — which is exactly why it is counted rather than silent.
    pub unrepresentable_mark: u64,
    /// Slots currently queued, and the ring's total. Together they are the early
    /// warning `dropped` is the late one for.
    pub queued: u64,
    pub capacity: u64,

    // ── the bar ring (ADR-0028) ──
    /// Closed bars the bar ring accepted.
    pub bars_published: u64,
    /// Closed bars the bar ring refused because it was full. Each leaves a `seq` hole
    /// the consumer sees, and — because the gap flag is computed from the last bar
    /// actually *delivered* — the next bar that lands says `gap_before`.
    pub bars_dropped: u64,
    /// Candle frames that refined the bar still forming without closing one. The
    /// ordinary case on a live `candle` subscription, which pushes the forming bar
    /// several times a minute. Not a loss and not a `seq` gap.
    pub bars_forming: u64,
    /// Candle frames for an interval *older* than the one already held. A venue
    /// re-sending history, or a replay feeding out of order. Ignored rather than
    /// published, because emitting one would move a bar backwards in a consumer's
    /// event-time series, and counted because nothing else would say it happened.
    pub bars_out_of_order: u64,
    /// Closed bars skipped because a price or volume could not be held exactly at the
    /// wire's `10^-8` scale. The next bar published for that instrument reports
    /// `gap_before`, because from Python's side the history really does have a hole.
    pub bars_unrepresentable: u64,
    /// The bar ring's live depth and total, as [`queued`](Self::queued) is for slices.
    pub bars_queued: u64,
    pub bars_capacity: u64,
}

/// One instrument+interval's bar stream, as the publisher tracks it.
///
/// A `Vec` of these with a linear scan, for the same reason `last` is a `Vec`: a
/// universe is a handful of instruments at one or two intervals, and the scan wins
/// outright against a hash while allocating only when a new pair is first seen.
struct BarState {
    symbol: SymbolId,
    interval: CandleInterval,
    /// The most recent frame for the bar that is still forming. It is published only
    /// once a frame for a *later* interval proves it final — the venue moving on is the
    /// only evidence available that does not come from a wall clock.
    forming: Candle,
    /// `open_time` of the last bar actually **delivered** for this pair; `None` until
    /// one has been. Drives the continuity flags, and is deliberately not advanced by a
    /// bar that was dropped or skipped: from Python's side that history has a hole, and
    /// the next bar has to say so.
    last_delivered_open: Option<Nanos>,
}

/// Writes [`MdSlice`]s and closed [`MdBar`]s onto the two market-data rings from the
/// core's event stream.
///
/// Owns both ring **files**: [`MdProducer::create`] truncates whatever is at the path,
/// so exactly one process may publish to a given ring. That is why the config refuses
/// to point this at the signal ring's path — truncating the ring Python is producing
/// into shows up as "the strategy stopped signalling", nowhere near its cause.
///
/// Both rings come from one config switch and one [`bar_ring_path`] derivation. Two
/// independent enables would let an operator turn on slices, forget bars, and get a
/// bar-driven strategy that runs cleanly and never trades.
pub struct MdPublisher {
    producer: MdProducer,
    /// The bar ring. Created whenever publishing is on, even for a session subscribed
    /// to no candle feed: a consumer that can always open the file and find it empty is
    /// a far better failure than one that cannot tell "no bars yet" from "wrong path".
    bars: MdBarProducer,
    /// The liveness sidecar (ADR-0030/0034). Created with the rings and off with them:
    /// under [`MdWritePolicy::OnChange`] a quiet market and a dead publisher write the
    /// same empty ring, and this is the only thing that can tell Python which of the two
    /// it is looking at. Not separately configurable — one switch cannot be half-turned,
    /// and an operator who enabled slices and forgot the beacon would get a monitor that
    /// silently went back to guessing.
    beacon: MdBeacon,
    path: PathBuf,
    bar_path: PathBuf,
    /// Where the beacon lives. Held rather than re-derived so the banner, the accessor
    /// and the beat all name one string — the same reason `bar_path` is a field.
    beacon_path: PathBuf,
    policy: MdWritePolicy,
    /// The window past which the top of book stops being something this will report.
    /// The planner's window and the risk gate's, deliberately: features computed on a
    /// quote the planner would refuse to price against describe a market the executing
    /// core has already stopped believing in.
    quote_max_age_ns: Nanos,
    /// Next sequence number. Spent on every *attempted* push, so a drop leaves the gap
    /// `MdRingConsumer.dropped` is computed from.
    next_seq: u64,
    /// The last slice **delivered** per symbol, for the change test.
    ///
    /// A `Vec` with a linear scan rather than a `HashMap`: a universe is a handful of
    /// instruments, so the scan wins outright, and it allocates only the first time a
    /// symbol is seen — nothing on the steady-state hot path, which is the rule the
    /// surrounding code (`IntentSource`'s reused buffers) already holds to.
    last: Vec<(SymbolId, MdSlice)>,
    /// Next bar sequence number, independent of the slice ring's. Spent on every
    /// *attempted* push, for the same reason.
    next_bar_seq: u64,
    /// Per instrument+interval bar tracking. See [`BarState`].
    bar_state: Vec<BarState>,
    stats: MdStats,
    /// Whether the current run of drops has already been reported. Cleared by a
    /// successful push, so a ring that fills once logs once — and a ring that fills,
    /// drains and fills again logs each time, because that is a different fault.
    drop_reported: bool,
    /// The same, for the bar ring. Tracked separately because the two rings fill for
    /// entirely different reasons — the slice ring under load, the bar ring only if
    /// nobody has read it for hours — and one flag would hide the second behind the
    /// first.
    bar_drop_reported: bool,
}

impl MdPublisher {
    /// Create the ring described by `cfg`, or `None` if publishing is off.
    ///
    /// A failure here is fatal to the session rather than a degraded state, which is the
    /// opposite of the signal ring's rule and deliberately so. That ring is *opened*, so
    /// "not there yet" is the ordinary startup race with Python. This one is *created*,
    /// so a failure is a bad path or a permissions problem — nothing that retrying
    /// fixes, and a session that runs on with no feature feed is one whose strategy will
    /// silently never have an opinion.
    pub fn open(cfg: &MdRingConfig, quote_max_age_ns: Nanos) -> Result<Option<Self>, RingError> {
        if !cfg.enabled {
            return Ok(None);
        }
        let producer = MdProducer::create(&cfg.path, cfg.capacity)?;
        let capacity = producer.capacity();
        // The same capacity for both rings. A second knob would be a number nobody
        // could reason about — bars arrive once a minute against a book's thousands a
        // second — and at 128 bytes a record the whole bar ring is a rounding error on
        // tmpfs even at the slice ring's size.
        let bar_path = bar_ring_path(&cfg.path);
        let bars = MdBarProducer::create(&bar_path, cfg.capacity)?;
        // Third file, same switch, third derived path. `MdBeacon::create` reinitialises
        // rather than truncating, so a monitor that already has this page mapped when a
        // publisher restarts under it does not take a `SIGBUS` off a zero-length file.
        let beacon_path = beacon_path(&cfg.path);
        let beacon = MdBeacon::create(&beacon_path)?;
        // All three paths, because one config switch creates three files and a consumer
        // pointed at the wrong sibling waits forever on a healthy session. The beacon is
        // named here for the same reason the bar ring is: an operator cannot see a
        // derived path in their own config file.
        println!(
            "md ring: publishing to {} (cap {}, policy {:?}); bars to {}; beacon at {}",
            cfg.path,
            capacity,
            cfg.policy,
            bar_path.display(),
            beacon_path.display(),
        );
        Ok(Some(Self {
            producer,
            bars,
            beacon,
            path: PathBuf::from(&cfg.path),
            bar_path,
            beacon_path,
            policy: cfg.policy,
            quote_max_age_ns,
            next_seq: 0,
            last: Vec::new(),
            next_bar_seq: 0,
            bar_state: Vec::new(),
            stats: MdStats {
                capacity,
                bars_capacity: capacity,
                ..MdStats::default()
            },
            drop_reported: false,
            bar_drop_reported: false,
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Where closed bars go. Derived from [`path`](Self::path); see [`bar_ring_path`].
    pub fn bar_path(&self) -> &Path {
        &self.bar_path
    }

    /// Where the liveness sidecar lives. Derived from [`path`](Self::path); see
    /// [`axon_ipc::beacon_path`].
    pub fn beacon_path(&self) -> &Path {
        &self.beacon_path
    }

    /// Passes of the core loop this publisher has recorded. A `u64` and a genuine
    /// total — unlike the five counters on the beacon record, which wrap.
    pub fn beats(&self) -> u64 {
        self.beacon.beats()
    }

    pub fn policy(&self) -> MdWritePolicy {
        self.policy
    }

    /// The counters, with both rings' live depth folded in.
    pub fn stats(&self) -> MdStats {
        MdStats {
            queued: self.producer.len(),
            bars_queued: self.bars.len(),
            ..self.stats
        }
    }

    /// Beat the liveness sidecar. **Called once per pass of the core loop, never from
    /// [`on_event`](Self::on_event)** — a beacon that advanced only when a record was
    /// written would carry exactly what the ring already carries, and the state it
    /// exists to make legible is the one in which no event arrived at all.
    ///
    /// `wall_ns` becomes the record's `last_beat_ns`, and that field is the **third
    /// named wall-clock exception in this codebase**, beside the dead-man's-switch
    /// deadline (wall clock because the *venue* holds it) and the reconnect backoff
    /// (wall clock because it orders nothing). The reason is the same shape as both and
    /// it is not a convenience: **the condition being detected is the absence of events,
    /// and the absence of an event has no event time.** An event-time-only beacon
    /// freezes at exactly the moment it is needed, because the clock that would advance
    /// it *is* the thing that stopped. What keeps the exception bounded is that nothing
    /// orders, ages or admits anything on it — it is read only as a difference, so that
    /// one read can say how long ago the publisher last ran (ADR-0034 §1).
    ///
    /// `wall_ns` is passed in rather than read here: the core loop already takes exactly
    /// one wall-clock reading per iteration and shares it with the mark cache's liveness
    /// clock, and a second reading would be a second answer to the same question. It is
    /// `0` offline, which is the record's own sentinel for "this session had no wall
    /// clock" — a reader that treated the zero as a timestamp would report every offline
    /// session as 56 years dead.
    ///
    /// Allocation-free and syscall-free, because the core loop runs it every pass.
    #[inline]
    pub fn beat(&self, last_event_ns: Nanos, wall_ns: u64) {
        self.beacon.beat(self.beat_of(last_event_ns, wall_ns));
    }

    /// A final beat marked as a deliberate shutdown.
    ///
    /// A reader must still treat the feed as gone — a clean exit does not put the market
    /// data back — but "the session ended" and "the session died" want different words
    /// from whoever is woken up by it, and only this side knows which happened.
    pub fn beat_stopped(&self, last_event_ns: Nanos, wall_ns: u64) {
        self.beacon.stop(self.beat_of(last_event_ns, wall_ns));
    }

    /// The five counters the beacon carries, taken off [`MdStats`] as `u64`.
    ///
    /// They are stored as `u32` and **truncate**, which is the one thing a caller has to
    /// know about them: a wrapped counter still yields an exact delta under
    /// `wrapping_sub` for any two readings fewer than 2³² increments apart, and that is
    /// the only question a monitor asks. A *saturated* counter would yield a delta of
    /// zero forever once pinned, so a busy publisher would read as a quiet one — the
    /// wrong answer to the only question this file is asked. The price is that the
    /// absolute value stops meaning anything after a wrap, so **no reader prints one as
    /// a total**; the totals live on [`MdStats`] and the status line (ADR-0034 §2).
    ///
    /// Five, not nine: sixty-four bytes hold twenty of payload, and `stale_quote` earned
    /// the last four over `bars_dropped` on measurement — a reconnect that failed to
    /// restore a `bbo` subscription put 45.8% of a 1 h 44 m soak under stale marks with
    /// every other counter healthy, which is the *alive but broken* state a liveness
    /// object should be able to name.
    #[inline]
    fn beat_of(&self, last_event_ns: Nanos, wall_ns: u64) -> MdBeat {
        MdBeat {
            last_event_ns,
            wall_ns,
            published: self.stats.published,
            coalesced: self.stats.coalesced,
            dropped: self.stats.dropped,
            bars_published: self.stats.bars_published,
            stale_quote: self.stats.stale_quote,
        }
    }

    /// Publish whatever this event made true, if anything.
    ///
    /// `market` must already have the event applied — the slice is built
    /// from the processor's caches, so calling this first would publish the previous
    /// frame's book under this frame's timestamp.
    pub fn on_event(&mut self, ts_event: Nanos, event: &Event, market: &MarketDataProcessor) {
        let Event::Market(m) = event else { return };
        // Exhaustive on purpose. A new `MarketEvent` variant should force a decision
        // about whether it can move a published slice, rather than being swept up by a
        // wildcard into publishing a record stamped with a clock that does not replay.
        let (symbol, kind) = match m {
            MarketEvent::Bbo(b) => (b.symbol_id, MD_KIND_QUOTE),
            MarketEvent::Book(s) => (s.symbol_id, MD_KIND_SNAPSHOT),
            MarketEvent::Trade(t) => (t.symbol_id, MD_KIND_TRADE),
            // A candle publishes no *slice* — it moves no field `MdSlice` carries — but
            // it is the whole input to the bar ring. See the module docs, point 6.
            MarketEvent::Candle(c) => {
                self.on_candle(c);
                return;
            }
            // See the module docs, point 5: a Hyperliquid ticker is ordered on our
            // receipt clock, so it must not stamp a record of its own. Its mark and
            // funding reach Python on the next venue-timed slice instead, read out of
            // the processor's cache in `build`.
            MarketEvent::Ticker(_) => return,
        };

        // Asked once, here, so the answer and the accounting for it cannot drift apart.
        let top = match top_of_book(market, symbol, ts_event, self.quote_max_age_ns) {
            TopState::Fresh(t) => Some(t),
            // Published without a quote rather than with a dead one, and counted: the
            // slice still carries the print that triggered it, and the alternative — a
            // minutes-old bid under a current `ts_event` — is a number Python has no way
            // to age and every reason to trust.
            TopState::Stale => {
                self.stats.stale_quote += 1;
                None
            }
            TopState::Unseen => None,
        };

        let Some(slice) = self.build(symbol, ts_event, kind, top, market) else {
            self.stats.unrepresentable += 1;
            return;
        };

        if self.policy == MdWritePolicy::OnChange
            && self
                .last
                .iter()
                .any(|(s, prev)| *s == symbol && same_state(prev, &slice))
        {
            self.stats.coalesced += 1;
            return;
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        let record = MdSlice { seq, ..slice };
        if self.producer.try_push(&record) {
            self.stats.published += 1;
            self.drop_reported = false;
            // Remembered only on success. A dropped slice must not leave the publisher
            // believing Python holds that state: the next update would then coalesce
            // against a record nobody ever received, and the hole would never close.
            match self.last.iter_mut().find(|(s, _)| *s == symbol) {
                Some((_, prev)) => *prev = record,
                None => self.last.push((symbol, record)),
            }
        } else {
            self.stats.dropped += 1;
            if !self.drop_reported {
                self.drop_reported = true;
                // Allocates, but once per outage rather than once per update — the hot
                // path stays allocation-free.
                eprintln!(
                    "md ring: {} is full - dropping slices. The reader is behind, or \
                     nothing is reading. seq {seq} was not delivered",
                    self.path.display()
                );
            }
        }
    }

    /// Flush both mappings to their backing files.
    ///
    /// Not needed for same-host cross-process visibility — the page cache handles that —
    /// but it makes the handoff explicit at the end of a session, which is what the
    /// cross-language test reads.
    pub fn flush(&self) -> std::io::Result<()> {
        self.producer.flush()?;
        self.bars.flush()
    }

    /// One candle frame from the venue.
    ///
    /// A `candle` subscription pushes the bar that is still **forming**, repeatedly,
    /// each frame stamped with a close time that has not happened yet. So a frame is
    /// never itself a record: it replaces the bar being tracked, and what gets published
    /// is the *previous* bar, at the moment a frame for a later interval proves it
    /// final. The venue moving on is the only evidence of closure that does not come
    /// from a wall clock, and a wall clock here would make the published series depend
    /// on how fast this machine drained the bus.
    fn on_candle(&mut self, c: &Candle) {
        let idx = self
            .bar_state
            .iter()
            .position(|s| s.symbol == c.symbol_id && s.interval == c.interval);
        let Some(i) = idx else {
            // First frame for this instrument+interval. Nothing is closed by it: the
            // bar it describes may be forming, and publishing a forming bar is the
            // lookahead leak the whole design refuses.
            self.bar_state.push(BarState {
                symbol: c.symbol_id,
                interval: c.interval,
                forming: c.clone(),
                last_delivered_open: None,
            });
            self.stats.bars_forming += 1;
            return;
        };

        match c.open_time.cmp(&self.bar_state[i].forming.open_time) {
            // A later interval has started, so the one we were holding is final.
            std::cmp::Ordering::Greater => {
                let closed = self.bar_state[i].forming.clone();
                self.publish_bar(i, &closed);
                self.bar_state[i].forming = c.clone();
            }
            // The same bar, refined. Ordinary, and not news until it closes.
            std::cmp::Ordering::Equal => {
                self.bar_state[i].forming = c.clone();
                self.stats.bars_forming += 1;
            }
            // A bar older than the one already held: the venue re-sending history, or a
            // replay feeding out of order. Publishing it would move a bar *backwards* in
            // the consumer's event-time series, which is the one thing an event-time
            // feed may not do.
            std::cmp::Ordering::Less => self.stats.bars_out_of_order += 1,
        }
    }

    /// Publish one bar that has been proven closed.
    ///
    /// `state` is an index rather than a `&mut BarState` because the borrow has to end
    /// before `self.stats` and `self.bars` are touched.
    fn publish_bar(&mut self, state: usize, bar: &Candle) {
        let interval_ms = interval_ms(bar.interval);
        let interval_ns = interval_ms as i64 * 1_000_000;
        // Continuity is a property of what Python actually received, so it is computed
        // against the last *delivered* bar. A bar that was dropped or skipped leaves the
        // baseline where it was, and the next one that lands says `gap_before` — which
        // is the truth: that history has a hole.
        let flags = match self.bar_state[state].last_delivered_open {
            None => MD_BAR_FLAG_FIRST_BAR,
            Some(prev) if bar.open_time != prev + interval_ns => MD_BAR_FLAG_GAP_BEFORE,
            Some(_) => 0,
        };

        let Some(ohlcv) = fixed_ohlcv(bar) else {
            // Skipped rather than rounded, for the same reason a quote is: a rounded
            // close is a price the venue never printed, and every return feature derived
            // from it is wrong with nothing downstream able to notice.
            self.stats.bars_unrepresentable += 1;
            return;
        };

        let seq = self.next_bar_seq;
        self.next_bar_seq += 1;
        let record = MdBar::new(
            seq,
            // The bar's own close, `T + 1 ms`, exactly as `decode_candle` stamped it.
            // Never `open_time`: a bar stamped with its open is the textbook lookahead
            // leak, and it is one field away at every call site.
            bar.ts_event,
            bar.open_time,
            bar.symbol_id.get(),
            interval_ms,
            flags,
        )
        .with_ohlcv(ohlcv.0, ohlcv.1, ohlcv.2, ohlcv.3, ohlcv.4);

        if self.bars.try_push(&record) {
            self.stats.bars_published += 1;
            self.bar_drop_reported = false;
            self.bar_state[state].last_delivered_open = Some(bar.open_time);
        } else {
            self.stats.bars_dropped += 1;
            if !self.bar_drop_reported {
                self.bar_drop_reported = true;
                // Allocates, but once per outage. A bar ring that fills is a different
                // fault from a slice ring that fills — at one record a minute it means
                // nothing has read it for hours — so it gets its own line.
                eprintln!(
                    "md ring: {} is full - dropping closed bars. Nothing has read the \
                     bar ring. seq {seq} was not delivered",
                    self.bar_path.display()
                );
            }
        }
    }

    /// The slice this event leaves true, or `None` if some value cannot be carried
    /// exactly at the wire scale.
    ///
    /// `top` is [`crate::quote::top_of_book`]'s answer and nothing else — the same
    /// answer, from the same call, that the planner's `quote_for` prices against. Two
    /// different answers to "what is the top of book" would mean features were computed
    /// against one book while orders were priced against another, which is exactly the
    /// divergence ADR-0012 §1 refuses a second venue connection to avoid.
    ///
    /// `None` leaves the quote fields at zero, which the record's own sentinel
    /// convention reads as "nothing seen yet" — the honest answer both when nothing has
    /// ever quoted the instrument and when everything that did has stopped.
    fn build(
        &mut self,
        symbol: SymbolId,
        ts_event: Nanos,
        kind: u8,
        top: Option<TopOfBook>,
        market: &MarketDataProcessor,
    ) -> Option<MdSlice> {
        let mut s = MdSlice::new(0, ts_event, symbol.get(), kind);
        if let Some(t) = top {
            s = s.with_bbo(
                decimal_to_fixed(t.bid_px)?,
                decimal_to_fixed(t.bid_sz)?,
                decimal_to_fixed(t.ask_px)?,
                decimal_to_fixed(t.ask_sz)?,
            );
        }
        if let Some(t) = market.last_trade(symbol) {
            s = s.with_last_trade(
                decimal_to_fixed(t.px)?,
                decimal_to_fixed(t.sz)?,
                // The print's *own* event time, not this update's. Folding them together
                // would make a 200 ms-old trade look simultaneous with the quote that
                // triggered the slice — a leak that makes a backtest look better than
                // live (ADR-0012 §2).
                t.ts_event,
                t.side == Side::Sell,
            );
        }
        // The mark/funding tail (ADR-0028), read from the same processor cache the risk
        // gate marks against. A `Ticker` never triggers a slice of its own — it has no
        // venue timestamp to stamp one with — so this is the only route by which the
        // venue's mark reaches Python at all, and it rides an event that *does* replay.
        if let Some(t) = market.ticker(symbol) {
            match fixed_ticker(t) {
                Some(v) => s = s.with_ticker(v.0, v.1, (v.2, v.3), v.4, v.5),
                // The tail goes out as the record's own "nothing seen yet" sentinel
                // rather than taking the whole slice down with it. See
                // `MdStats::unrepresentable_mark` for why this one is not symmetric with
                // the quote-price rule above.
                None => self.stats.unrepresentable_mark += 1,
            }
        }
        Some(s)
    }
}

/// A candle's OHLCV at the wire scale, or `None` if any value cannot be held exactly.
fn fixed_ohlcv(c: &Candle) -> Option<(i64, i64, i64, i64, i64)> {
    Some((
        decimal_to_fixed(c.open)?,
        decimal_to_fixed(c.high)?,
        decimal_to_fixed(c.low)?,
        decimal_to_fixed(c.close)?,
        decimal_to_fixed(c.volume)?,
    ))
}

/// A ticker's mark/funding tail at the wire scale: `(mark, index, rate, interval, venue
/// ts, ingest ts)`, or `None` if any value cannot be held exactly.
///
/// All or nothing on purpose. A half-filled tail would hand Python a funding rate with
/// no interval — the one shape `axon_core::Funding` exists to prevent, because a rate
/// without its period is not an approximate number but a wrong one.
fn fixed_ticker(t: &axon_core::Ticker) -> Option<(i64, i64, i64, i64, i64, i64)> {
    let index = match t.index_px {
        Some(px) => decimal_to_fixed(px)?,
        None => 0,
    };
    let (rate, interval) = match t.funding {
        Some(f) => (decimal_to_fixed(f.rate)?, f.interval),
        None => (0, 0),
    };
    Some((
        decimal_to_fixed(t.mark_px)?,
        index,
        rate,
        interval,
        // `0` when the venue stamped none — Hyperliquid never does (ADR-0011). It is
        // kept distinct from `ts_ingest` so a consumer can tell "the venue said when"
        // from "we noticed when", which is the difference between an age that replays
        // and one that measures this machine.
        t.ts_venue.unwrap_or(0),
        t.ts_ingest,
    ))
}

impl Drop for MdPublisher {
    /// Flush on the way out so a session's last slices are on the file before the
    /// process that wrote them is gone. Best-effort: there is nothing useful to do with
    /// an error while unwinding.
    fn drop(&mut self) {
        let _ = self.producer.flush();
    }
}

/// Whether two slices say the same thing about the market.
///
/// `seq` and `ts_event` are excluded because they are what makes a repeat a repeat.
/// `kind` is excluded for a sharper reason: it is the *cause* of an update, and a
/// session subscribed to both `bbo` and `l2Book` — the shipped default — alternates
/// quote and snapshot on every frame. Comparing `kind` would make every frame differ,
/// and [`MdWritePolicy::OnChange`] would silently degrade into `EveryUpdate` for the
/// configuration almost everyone runs.
///
/// The mark's two clocks are excluded, and that is a decision rather than an omission.
/// `mark_ts_ingest` is a *receipt* stamp that moves on every `activeAssetCtx` frame,
/// which a live venue pushes constantly; including it would make every quote differ
/// from the last and turn [`MdWritePolicy::OnChange`] into `EveryUpdate` on any session
/// subscribed to a ticker — precisely the ring-slot exhaustion the policy exists to
/// prevent. A mark that has not moved is the same mark, whatever time it was restated
/// at. (`last_trade_ts` is included for the opposite reason: two prints at the same
/// price and size are two different events, and the timestamp is what says so.)
///
/// Destructured rather than field-compared so that a future field fails to compile here
/// instead of being quietly excluded from the change test.
fn same_state(a: &MdSlice, b: &MdSlice) -> bool {
    let MdSlice {
        seq: _,
        ts_event: _,
        kind: _,
        schema_version: _,
        symbol_id: _,
        mark_ts_venue: _,
        mark_ts_ingest: _,
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        last_trade_px,
        last_trade_sz,
        last_trade_ts,
        flags,
        mark_px,
        index_px,
        funding_rate,
        funding_interval_ns,
    } = *a;
    bid_px == b.bid_px
        && bid_sz == b.bid_sz
        && ask_px == b.ask_px
        && ask_sz == b.ask_sz
        && last_trade_px == b.last_trade_px
        && last_trade_sz == b.last_trade_sz
        && last_trade_ts == b.last_trade_ts
        && flags == b.flags
        && mark_px == b.mark_px
        && index_px == b.index_px
        && funding_rate == b.funding_rate
        && funding_interval_ns == b.funding_interval_ns
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_contracts::MD_FLAG_LAST_TRADE_SELL;
    use axon_core::{Bbo, BookSnapshot, Decimal, EventHandler, Funding, Level, Ticker, Trade};
    use axon_ipc::{read_md_beacon, MdBarConsumer, MdBeaconReader, MdConsumer};
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicU64, Ordering};

    const BTC: SymbolId = SymbolId::new(0);
    const ETH: SymbolId = SymbolId::new(1);
    const SEC: Nanos = 1_000_000_000;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(tag: &str) -> String {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!("axon-mdring-{tag}-{}-{n}.ring", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn cfg(path: &str, capacity: u64, policy: MdWritePolicy) -> MdRingConfig {
        MdRingConfig {
            enabled: true,
            path: path.to_string(),
            capacity,
            policy,
        }
    }

    /// The mark window every rig runs with — the shipped default, so the tests age a
    /// quote the way a session does.
    const WINDOW: Nanos = 10 * SEC;

    /// A publisher plus the state it reads from, driven together.
    struct Rig {
        market: MarketDataProcessor,
        md: MdPublisher,
        path: String,
    }

    impl Rig {
        fn new(tag: &str, capacity: u64, policy: MdWritePolicy) -> Self {
            let path = temp_path(tag);
            let md = MdPublisher::open(&cfg(&path, capacity, policy), WINDOW)
                .expect("create ring")
                .expect("enabled");
            Self {
                market: MarketDataProcessor::new(),
                md,
                path,
            }
        }

        /// Exactly the order `CoreHandler` uses: apply, then publish.
        fn feed(&mut self, ev: Event) {
            let ts = ev.ts_event();
            self.market.on_event(ts, &ev);
            self.md.on_event(ts, &ev, &self.market);
        }

        fn drain(&self) -> Vec<MdSlice> {
            self.md.flush().unwrap();
            let c = MdConsumer::open(&self.path).expect("open ring");
            let mut out = Vec::new();
            while let Some(s) = c.try_pop() {
                out.push(s);
            }
            out
        }

        fn drain_bars(&self) -> Vec<MdBar> {
            self.md.flush().unwrap();
            let c = MdBarConsumer::open(self.md.bar_path()).expect("open bar ring");
            let mut out = Vec::new();
            while let Some(b) = c.try_pop() {
                out.push(b);
            }
            out
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(bar_ring_path(&self.path));
        }
    }

    fn bbo(sym: SymbolId, bid: Decimal, ask: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Bbo(Bbo {
            symbol_id: sym,
            bid_px: bid,
            bid_sz: dec!(2),
            ask_px: ask,
            ask_sz: dec!(3),
            ts_event: ts,
        }))
    }

    fn trade(sym: SymbolId, px: Decimal, side: Side, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Trade(Trade {
            symbol_id: sym,
            px,
            sz: dec!(0.5),
            side,
            ts_event: ts,
        }))
    }

    fn book(sym: SymbolId, bid: Decimal, ask: Decimal, ts: Nanos) -> Event {
        Event::Market(MarketEvent::Book(BookSnapshot {
            symbol_id: sym,
            bids: vec![Level::new(bid, dec!(10))],
            asks: vec![Level::new(ask, dec!(10))],
            ts_event: ts,
        }))
    }

    const MIN: Nanos = 60 * SEC;

    /// One `candle` frame. `close` is a parameter because a *forming* bar's close moves
    /// on every frame, which is the whole reason a frame cannot itself be a record.
    fn candle(sym: SymbolId, interval: CandleInterval, open_time: Nanos, close: Decimal) -> Event {
        let len = interval_ms(interval) as i64 * 1_000_000;
        Event::Market(MarketEvent::Candle(Candle {
            symbol_id: sym,
            interval,
            open: dec!(100),
            high: dec!(110),
            low: dec!(90),
            close,
            volume: dec!(7),
            open_time,
            // `T + 1 ms`, i.e. the instant the bar is final for ordering — exactly what
            // `decode_candle` stamps and what `CLOSE_STAMP_OFFSET_MS` mirrors.
            ts_event: open_time + len,
        }))
    }

    /// A one-minute frame, the common case.
    fn m1(sym: SymbolId, open_time: Nanos, close: Decimal) -> Event {
        candle(sym, CandleInterval::M1, open_time, close)
    }

    fn ticker(sym: SymbolId, mark: Decimal, ts_ingest: Nanos) -> Event {
        Event::Market(MarketEvent::Ticker(Ticker {
            symbol_id: sym,
            mark_px: mark,
            index_px: Some(dec!(49_995)),
            mid_px: None,
            funding: Some(Funding {
                rate: dec!(0.0000125),
                interval: 3_600 * SEC,
            }),
            open_interest: Some(dec!(1234)),
            // Hyperliquid stamps none; the rig matches the venue it runs against.
            ts_venue: None,
            ts_ingest,
        }))
    }

    #[test]
    fn a_full_ring_drops_the_newest_slice_instead_of_blocking() {
        // Stalling the execution core until a Python feature computation catches up is
        // the one failure this direction of the boundary must never have. The ring is
        // sized to two so the third quote has nowhere to go.
        let mut r = Rig::new("full", 2, MdWritePolicy::EveryUpdate);
        for i in 0..5 {
            r.feed(bbo(BTC, dec!(100) + Decimal::from(i), dec!(101), SEC));
        }
        let s = r.md.stats();
        assert_eq!(s.published, 2);
        assert_eq!(s.dropped, 3, "the ring refused, the core kept going");

        // …and what did land is intact, not a torn record.
        let got = r.drain();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].bid_px, decimal_to_fixed(dec!(100)).unwrap());
        assert_eq!(got[1].bid_px, decimal_to_fixed(dec!(101)).unwrap());
    }

    #[test]
    fn a_dropped_slice_leaves_the_seq_gap_python_infers_the_loss_from() {
        // `MdRingConsumer.dropped` is `span - count`. If the publisher only spent a `seq`
        // on a successful push, a silently-dropping feed and a healthy one would be
        // indistinguishable from Python — the exact failure ADR-0012 §5 names.
        let mut r = Rig::new("gap", 2, MdWritePolicy::EveryUpdate);
        for i in 0..4 {
            r.feed(bbo(BTC, dec!(100) + Decimal::from(i), dec!(101), SEC));
        }
        // Two land (seq 0,1), two are refused (seq 2,3). Free a slot and publish again.
        let c = MdConsumer::open(&r.path).unwrap();
        assert_eq!(c.try_pop().unwrap().seq, 0);
        r.feed(bbo(BTC, dec!(200), dec!(201), SEC));
        drop(c);

        let seqs: Vec<u64> = r.drain().iter().map(|s| s.seq).collect();
        assert_eq!(
            seqs,
            vec![1, 4],
            "the two refused slices burned seq 2 and 3"
        );
    }

    #[test]
    fn a_coalesced_update_is_never_reported_to_python_as_a_drop() {
        // The other direction of the same lie: an update that carried no news must not
        // leave a hole, or a strategy would believe its feature history is broken and
        // refuse to trade on a perfectly complete stream.
        let mut r = Rig::new("coal", 8, MdWritePolicy::OnChange);
        for _ in 0..5 {
            r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        }
        r.feed(bbo(BTC, dec!(102), dec!(103), 2 * SEC));

        let s = r.md.stats();
        assert_eq!(s.published, 2);
        assert_eq!(s.coalesced, 4);
        assert_eq!(s.dropped, 0);
        let seqs: Vec<u64> = r.drain().iter().map(|s| s.seq).collect();
        assert_eq!(seqs, vec![0, 1], "gap-free: nothing was lost");
    }

    #[test]
    fn a_dropped_slice_is_not_remembered_as_delivered() {
        // If a refused slice updated the change baseline, the next update carrying the
        // same state would coalesce against a record Python never received and the hole
        // would never close.
        let mut r = Rig::new("nomem", 1, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC)); // lands
        r.feed(bbo(BTC, dec!(105), dec!(106), 2 * SEC)); // ring full, refused
        assert_eq!(r.md.stats().dropped, 1);

        let c = MdConsumer::open(&r.path).unwrap();
        assert_eq!(
            c.try_pop().unwrap().bid_px,
            decimal_to_fixed(dec!(100)).unwrap()
        );
        // The same state arrives again; it must be republished, not coalesced away.
        r.feed(bbo(BTC, dec!(105), dec!(106), 3 * SEC));
        drop(c);
        let got = r.drain();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].bid_px, decimal_to_fixed(dec!(105)).unwrap());
        assert_eq!(r.md.stats().coalesced, 0);
    }

    #[test]
    fn an_update_that_moves_nothing_the_record_carries_is_not_published() {
        // An L2 frame that only changed level five is a record that costs a ring slot
        // and tells Python nothing — and it is the slot the next real move needed.
        let mut r = Rig::new("nomove", 8, MdWritePolicy::OnChange);
        r.feed(book(BTC, dec!(100), dec!(101), SEC));
        r.feed(book(BTC, dec!(100), dec!(101), 2 * SEC));
        assert_eq!(r.md.stats().published, 1);
        assert_eq!(r.md.stats().coalesced, 1);
    }

    #[test]
    fn alternating_feeds_do_not_defeat_coalescing() {
        // The shipped default subscribes to `bbo` *and* `l2Book`. If the change test
        // compared `kind`, quote/snapshot would alternate on every frame, nothing would
        // ever match, and `on_change` would silently become `every_update` for almost
        // every real session.
        let mut r = Rig::new("alt", 16, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        for i in 1..6 {
            r.feed(book(BTC, dec!(100), dec!(101), i * SEC));
            r.feed(bbo(BTC, dec!(100), dec!(101), i * SEC));
        }
        assert_eq!(r.md.stats().published, 1);
        assert_eq!(r.md.stats().coalesced, 10);
    }

    #[test]
    fn a_frozen_bbo_cache_never_prices_a_slice_the_book_has_moved_past() {
        // The reachable production form: a reconnect re-subscribes `l2Book` and not
        // `bbo`, so `market.bbo()` is pinned at its pre-disconnect value for the rest of
        // the session. Preferring it unconditionally republishes that dead quote under
        // every later event's timestamp — and under `on_change` the book's real move is
        // coalesced away, so the ring reports nothing at all while `dropped` stays 0.
        let mut r = Rig::new("frozenbbo", 8, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        r.feed(book(BTC, dec!(200), dec!(201), SEC + WINDOW + 1));

        let got = r.drain();
        assert_eq!(got.len(), 2, "the book's move is a record, not a coalesce");
        assert_eq!(got[1].bid_px, decimal_to_fixed(dec!(200)).unwrap());
        assert_eq!(got[1].ask_px, decimal_to_fixed(dec!(201)).unwrap());
        assert_eq!(r.md.stats().stale_quote, 0, "a live feed still answered");
    }

    #[test]
    fn a_slice_carries_no_quote_at_all_rather_than_one_python_cannot_age() {
        // `MdSlice` has one timestamp and it belongs to the *triggering* event, so a
        // consumer cannot tell a bid from five minutes ago from a live one. When every
        // feed carrying a quote has gone quiet the fields go out as the record's own
        // "nothing seen yet" sentinel, and the count is what says why.
        let mut r = Rig::new("stalequote", 8, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        r.feed(trade(BTC, dec!(100.5), Side::Buy, SEC + WINDOW + 1));

        let got = r.drain();
        let print = got.last().unwrap();
        assert_eq!(print.kind, MD_KIND_TRADE, "the print still reaches Python");
        assert_eq!(print.bid_px, 0, "no quote, rather than a dead one");
        assert_eq!(print.ask_px, 0);
        assert_eq!(
            print.last_trade_px,
            decimal_to_fixed(dec!(100.5)).unwrap(),
            "the slice is not withheld: a stale instrument must stay distinguishable \
             from an idle one"
        );
        assert_eq!(r.md.stats().stale_quote, 1);
    }

    #[test]
    fn every_update_publishes_what_on_change_suppresses() {
        // The policy has to be a real choice, not a name: the same stream under the
        // other setting must produce a record per update.
        let mut r = Rig::new("every", 16, MdWritePolicy::EveryUpdate);
        for i in 0..5 {
            r.feed(bbo(BTC, dec!(100), dec!(101), i * SEC));
        }
        assert_eq!(r.md.stats().published, 5);
        assert_eq!(r.md.stats().coalesced, 0);
    }

    #[test]
    fn a_slice_carries_the_events_own_time_not_the_wall_clock() {
        // Everything downstream orders on this field. A receipt stamp here would make
        // every Python feature window measure the machine rather than the market.
        let mut r = Rig::new("ts", 8, MdWritePolicy::OnChange);
        let ancient = 1_600_000_000_000_000_000;
        r.feed(bbo(BTC, dec!(100), dec!(101), ancient));
        assert_eq!(r.drain()[0].ts_event, ancient);
    }

    #[test]
    fn the_last_print_keeps_its_own_older_time_when_a_quote_triggers_the_slice() {
        // Folding the two together would make a stale print look simultaneous with the
        // quote that carried it, and "how stale is the last trade" is a feature input.
        let mut r = Rig::new("print", 8, MdWritePolicy::OnChange);
        r.feed(trade(BTC, dec!(100.5), Side::Sell, SEC));
        r.feed(bbo(BTC, dec!(100), dec!(101), 5 * SEC));

        let got = r.drain();
        let quote = got.last().unwrap();
        assert_eq!(quote.kind, MD_KIND_QUOTE);
        assert_eq!(quote.ts_event, 5 * SEC);
        assert_eq!(quote.last_trade_ts, SEC, "the print's own, earlier time");
        assert_eq!(quote.last_trade_px, decimal_to_fixed(dec!(100.5)).unwrap());
        assert!(quote.last_trade_is_sell());
    }

    #[test]
    fn a_trade_slice_carries_the_full_state_not_just_the_print() {
        // Full state, not deltas (ADR-0012 §2): a consumer that fell behind and skipped
        // records must still see something coherent.
        let mut r = Rig::new("fullstate", 8, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        r.feed(trade(BTC, dec!(100.5), Side::Buy, 2 * SEC));

        let got = r.drain();
        let print = got.last().unwrap();
        assert_eq!(print.kind, MD_KIND_TRADE);
        assert_eq!(print.bid_px, decimal_to_fixed(dec!(100)).unwrap());
        assert_eq!(print.ask_px, decimal_to_fixed(dec!(101)).unwrap());
        assert_eq!(print.flags & MD_FLAG_LAST_TRADE_SELL, 0, "buy aggressor");
    }

    #[test]
    fn a_ticker_never_publishes_a_slice_stamped_with_our_receipt_clock() {
        // Hyperliquid's activeAssetCtx carries no venue time (ADR-0011), so a
        // ticker-triggered slice would be ordered on the clock of the machine that
        // received it, and a replay of the same capture would publish a different
        // `ts_event`. That is the ring quietly ceasing to be reproducible.
        let mut r = Rig::new("ticker", 8, MdWritePolicy::EveryUpdate);
        r.feed(Event::Market(MarketEvent::Ticker(Ticker {
            symbol_id: BTC,
            mark_px: dec!(50_000),
            index_px: None,
            mid_px: None,
            funding: None,
            open_interest: None,
            ts_venue: None,
            ts_ingest: 42,
        })));
        r.feed(Event::Market(MarketEvent::Candle(Candle {
            symbol_id: BTC,
            interval: CandleInterval::M1,
            open: dec!(1),
            high: dec!(2),
            low: dec!(1),
            close: dec!(2),
            volume: dec!(3),
            open_time: 0,
            ts_event: 60 * SEC,
        })));
        assert_eq!(r.md.stats().published, 0);
        assert!(r.drain().is_empty());
        // …and one candle frame closes nothing, so the bar ring is empty too: the bar
        // it describes may still be forming, and a forming bar has a close time in the
        // future.
        assert!(r.drain_bars().is_empty());
        assert_eq!(r.md.stats().bars_published, 0);
    }

    // ── the bar ring (ADR-0028) ──────────────────────────────────────────────

    #[test]
    fn a_forming_bar_is_never_published_because_its_close_has_not_happened_yet() {
        // A `candle` subscription pushes the bar as it forms, several times a minute,
        // each frame carrying a mid-bar `close` under a close time in the future.
        // Publishing one hands a strategy a bar that has not happened.
        let mut r = Rig::new("forming", 8, MdWritePolicy::EveryUpdate);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, 0, dec!(101)));
        r.feed(m1(BTC, 0, dec!(102)));
        assert!(r.drain_bars().is_empty());
        assert_eq!(r.md.stats().bars_forming, 3);
        assert_eq!(r.md.stats().bars_published, 0);
    }

    #[test]
    fn every_in_progress_frame_is_dropped_however_many_the_venue_sends_and_however_it_stamps_them()
    {
        // Measured, not assumed: agent M8 captured 2 353 real Binance frames and found
        // **111 of 112** `kline_1m` frames were in-progress — and every one of them
        // carried the *same* `T`, so `decode_candle`'s `T + 1 ms` gives them all an
        // identical `ts_event` and an event-time sort cannot separate the last partial
        // from the close. `axon_core::Candle` has no `is_final`, so a publisher that
        // trusted the frame would hand `perp_bar` a bar that then changed underneath
        // it — training/serving skew the feature-parity gate reports as a numeric
        // disagreement a long way from its cause.
        //
        // This publisher never asks. Finality is *structural*: frames are keyed on
        // `open_time`, so every frame for a bar — however many, however stamped — only
        // refines it, and nothing is published until the venue starts the next
        // interval. Delete the `Ordering::Equal` arm and this test fails; no venue
        // behaviour has to be trusted for it to hold.
        let mut r = Rig::new("inprogress", 16, MdWritePolicy::EveryUpdate);
        for i in 0..111 {
            r.feed(m1(BTC, 0, dec!(100) + Decimal::from(i)));
        }
        assert!(r.drain_bars().is_empty(), "111 partials, nothing published");
        assert_eq!(r.md.stats().bars_forming, 111);

        // The venue's own final frame for that bar is *also* not published on arrival:
        // it is indistinguishable from a partial by timestamp, so it is treated as one
        // more refinement and published when the next interval opens.
        r.feed(m1(BTC, 0, dec!(999)));
        assert!(r.drain_bars().is_empty());
        r.feed(m1(BTC, MIN, dec!(1_000)));
        let bars = r.drain_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(
            bars[0].close,
            decimal_to_fixed(dec!(999)).unwrap(),
            "the last frame's close is the bar's close"
        );
    }

    #[test]
    fn a_bar_is_published_only_once_the_venue_moves_on_to_the_next_one() {
        // The venue starting the next interval is the only evidence of closure that is
        // not a wall clock — and a wall-clock cadence would make the published series
        // depend on how fast this machine drained the bus.
        let mut r = Rig::new("closed", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, 0, dec!(105))); // still forming; the close moved
        r.feed(m1(BTC, MIN, dec!(106))); // the next bar starts -> the first is final

        let bars = r.drain_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close, decimal_to_fixed(dec!(105)).unwrap());
        assert_eq!(bars[0].open_time, 0);
    }

    #[test]
    fn the_last_bar_of_a_session_is_not_published_because_nothing_proved_it_closed() {
        // The stated cost of deriving closure from the stream. Publishing it on shutdown
        // would be publishing a bar that may still have been forming — which is the one
        // thing this whole path refuses.
        let mut r = Rig::new("lastbar", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, MIN, dec!(101)));
        assert_eq!(
            r.drain_bars().len(),
            1,
            "only the first bar is provably done"
        );
    }

    #[test]
    fn a_bar_carries_its_close_time_not_its_open_time() {
        // A bar stamped with its open time is the textbook lookahead leak: the strategy
        // appears to have acted on the close a whole bar early. The two fields are one
        // line apart in the constructor.
        let mut r = Rig::new("barts", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, MIN, dec!(101)));

        let b = r.drain_bars().remove(0);
        assert_eq!(b.open_time, 0);
        assert_eq!(
            b.ts_event, MIN,
            "close = open + one interval, i.e. venue T + 1ms"
        );
        assert_eq!(b.interval_ms, 60_000);
    }

    #[test]
    fn two_identical_consecutive_bars_are_both_published() {
        // `on_change` must not reach the bar ring. A flat market prints identical bars,
        // and coalescing one away silently shortens every rolling feature window
        // downstream — a strategy would then compute confidently across a hole.
        let mut r = Rig::new("flatbars", 8, MdWritePolicy::OnChange);
        for i in 0..4 {
            r.feed(m1(BTC, i * MIN, dec!(100)));
        }
        let bars = r.drain_bars();
        assert_eq!(bars.len(), 3, "three closed; the fourth is still forming");
        assert!(bars.iter().all(|b| b.close == bars[0].close));
        assert_eq!(
            bars.iter().map(|b| b.seq).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "gap-free: nothing was suppressed"
        );
    }

    #[test]
    fn the_first_bar_of_a_session_says_unknown_rather_than_gap() {
        // Continuity with anything earlier is genuinely unknown, not broken. Flagging it
        // as a gap would cry wolf on every session start; flagging nothing would let a
        // feature window begin mid-history in silence.
        let mut r = Rig::new("firstbar", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, MIN, dec!(101)));

        let b = r.drain_bars().remove(0);
        assert!(b.is_first_bar());
        assert!(!b.has_gap_before());
    }

    #[test]
    fn a_bar_the_venue_never_printed_is_flagged_so_a_window_does_not_span_the_hole() {
        // A halted feed, not a quiet market. Nothing else on the wire can say so: a
        // `seq` gap means the *ring* dropped, which is a different fault entirely.
        let mut r = Rig::new("bargap", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, MIN, dec!(101))); // closes bar 0
        r.feed(m1(BTC, 5 * MIN, dec!(102))); // closes bar 1 — and skips 2, 3, 4

        let bars = r.drain_bars();
        assert_eq!(bars.len(), 2);
        assert!(
            !bars[1].has_gap_before(),
            "bar 1 followed bar 0 immediately"
        );
        // The bar that *reports* the hole is the one after it, and it is not published
        // until the venue moves on again.
        r.feed(m1(BTC, 6 * MIN, dec!(103)));
        let later = r.drain_bars();
        assert_eq!(later.len(), 1);
        assert!(later[0].has_gap_before());
        assert!(!later[0].is_first_bar());
    }

    #[test]
    fn a_bar_older_than_the_one_held_never_moves_the_series_backwards() {
        // A venue re-sending history, or a replay feeding out of order. Publishing it
        // would put a bar behind one a consumer already holds, and an event-time series
        // that goes backwards is one nothing downstream can align on.
        let mut r = Rig::new("barback", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 5 * MIN, dec!(100)));
        r.feed(m1(BTC, 2 * MIN, dec!(99)));
        assert_eq!(r.md.stats().bars_out_of_order, 1);
        assert!(r.drain_bars().is_empty());
    }

    #[test]
    fn a_full_bar_ring_drops_the_bar_and_spends_the_seq_python_infers_the_loss_from() {
        // Same contract as the slice ring: the core never blocks, and a drop leaves
        // exactly the `seq` hole a consumer computes its loss from.
        let mut r = Rig::new("barfull", 2, MdWritePolicy::OnChange);
        for i in 0..6 {
            r.feed(m1(BTC, i * MIN, dec!(100) + Decimal::from(i)));
        }
        let s = r.md.stats();
        assert_eq!(s.bars_published, 2);
        assert_eq!(s.bars_dropped, 3, "the ring refused; the core kept going");
        let seqs: Vec<u64> = r.drain_bars().iter().map(|b| b.seq).collect();
        assert_eq!(seqs, vec![0, 1]);
    }

    #[test]
    fn a_dropped_bar_makes_the_next_one_report_the_hole_it_left() {
        // The continuity baseline advances only on a *delivered* bar. If a dropped bar
        // moved it, Python would receive a series with a missing row and no flag saying
        // so — the exact silence the flag exists to break.
        let mut r = Rig::new("bardropgap", 1, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(m1(BTC, MIN, dec!(101))); // bar 0 lands, filling the ring
        r.feed(m1(BTC, 2 * MIN, dec!(102))); // bar 1 is refused
        assert_eq!(r.md.stats().bars_dropped, 1);

        let c = MdBarConsumer::open(r.md.bar_path()).unwrap();
        assert_eq!(c.try_pop().unwrap().open_time, 0);
        r.feed(m1(BTC, 3 * MIN, dec!(103))); // bar 2 now fits
        drop(c);

        let bars = r.drain_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].open_time, 2 * MIN);
        assert!(bars[0].has_gap_before(), "bar 1 never reached Python");
    }

    #[test]
    fn each_instrument_and_interval_closes_on_its_own_schedule() {
        // One shared bar baseline would let BTC's 1m frames close ETH's, and a 5m series
        // would be closed by every 1m frame — publishing bars that had not happened.
        let mut r = Rig::new("barmulti", 16, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100)));
        r.feed(candle(BTC, CandleInterval::M5, 0, dec!(100)));
        r.feed(m1(ETH, 0, dec!(200)));
        r.feed(m1(BTC, MIN, dec!(101))); // closes BTC 1m only

        let bars = r.drain_bars();
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].symbol_id, BTC.get());
        assert_eq!(bars[0].interval_ms, 60_000);
    }

    #[test]
    fn a_bar_price_finer_than_the_wire_scale_is_skipped_rather_than_rounded() {
        // A rounded close is a price the venue never printed, and every return feature
        // derived from it is wrong with nothing downstream able to notice.
        let mut r = Rig::new("barinexact", 8, MdWritePolicy::OnChange);
        r.feed(m1(BTC, 0, dec!(100.000000005)));
        r.feed(m1(BTC, MIN, dec!(101)));
        assert_eq!(r.md.stats().bars_unrepresentable, 1);
        assert_eq!(r.md.stats().bars_published, 0);
        assert!(r.drain_bars().is_empty());
    }

    #[test]
    fn the_bar_ring_is_a_sibling_of_the_slice_ring_so_one_switch_cannot_be_half_turned() {
        // Two independent settings would let an operator enable slices, forget bars, and
        // run a bar-driven strategy that starts cleanly and simply never has an opinion.
        assert_eq!(
            bar_ring_path("/dev/shm/axon-md.ring"),
            Path::new("/dev/shm/axon-md.bars.ring")
        );
        assert_eq!(
            bar_ring_path("/dev/shm/axon-md"),
            Path::new("/dev/shm/axon-md.bars")
        );

        let r = Rig::new("sibling", 8, MdWritePolicy::OnChange);
        assert!(r.md.bar_path().exists(), "publishing on creates both rings");
        assert_ne!(r.md.bar_path(), r.md.path());
    }

    // ── the mark/funding tail (ADR-0028) ─────────────────────────────────────

    #[test]
    fn a_ticker_reaches_python_on_the_next_venue_timed_slice() {
        // The mark cannot stamp a record of its own — `activeAssetCtx` carries no venue
        // time (ADR-0011) — so this is the only route by which it crosses at all.
        let mut r = Rig::new("tick", 8, MdWritePolicy::OnChange);
        r.feed(ticker(BTC, dec!(50_000), 999));
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));

        let s = r.drain().remove(0);
        assert_eq!(
            s.ts_event, SEC,
            "ordered on the quote's venue time, not the ticker's"
        );
        assert_eq!(s.mark_px, decimal_to_fixed(dec!(50_000)).unwrap());
        assert_eq!(s.index_px, decimal_to_fixed(dec!(49_995)).unwrap());
        assert_eq!(s.funding_rate, decimal_to_fixed(dec!(0.0000125)).unwrap());
        assert_eq!(s.funding_interval_ns, 3_600 * SEC);
        assert!(s.has_ticker());
        assert!(
            !s.mark_is_venue_timed(),
            "Hyperliquid stamps none, and the record has to say so"
        );
        assert_eq!(s.mark_ts_ingest, 999);
    }

    #[test]
    fn a_slice_with_no_ticker_leaves_the_whole_tail_at_its_sentinel() {
        // Never back-filled from the mid or the last print: a fabricated mark is one the
        // venue would not margin against, and the risk gate and the feature computation
        // would then be reasoning about different prices.
        let mut r = Rig::new("notick", 8, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        let s = r.drain().remove(0);
        assert!(!s.has_ticker());
        assert_eq!(s.mark_px, 0);
        assert_eq!(s.funding_interval_ns, 0);
    }

    #[test]
    fn a_ticker_that_only_restates_the_same_mark_does_not_defeat_coalescing() {
        // A live venue pushes `activeAssetCtx` constantly. If the mark's *receipt* clock
        // were part of the change test, every quote would differ from the last and
        // `on_change` would silently become `every_update` on any session with a ticker
        // feed — which is how the ring comes to be full when something finally happens.
        let mut r = Rig::new("markcoal", 16, MdWritePolicy::OnChange);
        r.feed(ticker(BTC, dec!(50_000), 1));
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        for i in 2..8 {
            r.feed(ticker(BTC, dec!(50_000), i));
            r.feed(bbo(BTC, dec!(100), dec!(101), i * SEC));
        }
        assert_eq!(r.md.stats().published, 1);
        assert_eq!(r.md.stats().coalesced, 6);
    }

    #[test]
    fn a_mark_that_actually_moved_is_news_and_is_published() {
        // The other side of the same rule: the mark is a field the record carries, so a
        // change in it is a change in the record, quote or no quote.
        let mut r = Rig::new("markmove", 16, MdWritePolicy::OnChange);
        r.feed(ticker(BTC, dec!(50_000), 1));
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        r.feed(ticker(BTC, dec!(50_100), 2));
        r.feed(bbo(BTC, dec!(100), dec!(101), 2 * SEC));

        let got = r.drain();
        assert_eq!(got.len(), 2, "the book stood still; the mark did not");
        assert_eq!(got[1].mark_px, decimal_to_fixed(dec!(50_100)).unwrap());
    }

    #[test]
    fn an_inexact_funding_rate_zeroes_the_tail_rather_than_silencing_the_book() {
        // Deliberately asymmetric with the quote rule. The top of book and the last
        // print on the same record are independently exact; letting one venue's funding
        // precision take the book feed down with it would be the larger wrong. It is
        // still a lie of omission, which is why it is counted.
        let mut r = Rig::new("markinexact", 8, MdWritePolicy::OnChange);
        r.feed(Event::Market(MarketEvent::Ticker(Ticker {
            symbol_id: BTC,
            mark_px: dec!(50_000),
            index_px: None,
            mid_px: None,
            funding: Some(Funding {
                rate: dec!(0.000000000125),
                interval: 3_600 * SEC,
            }),
            open_interest: None,
            ts_venue: None,
            ts_ingest: 5,
        })));
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));

        let s = r.drain().remove(0);
        assert_eq!(r.md.stats().unrepresentable_mark, 1);
        assert_eq!(r.md.stats().published, 1, "the book still crossed");
        assert_eq!(s.bid_px, decimal_to_fixed(dec!(100)).unwrap());
        assert!(
            !s.has_ticker(),
            "the tail is the sentinel, not a rounded number"
        );
    }

    #[test]
    fn a_price_finer_than_the_wire_scale_is_skipped_rather_than_rounded() {
        // The wire carries 10^-8. A rounded price is one the venue never quoted, and a
        // feature computed from it is wrong with nothing downstream able to notice.
        let mut r = Rig::new("inexact", 8, MdWritePolicy::EveryUpdate);
        r.feed(bbo(BTC, dec!(100.000000005), dec!(101), SEC));
        assert_eq!(r.md.stats().unrepresentable, 1);
        assert_eq!(r.md.stats().published, 0);
        assert!(r.drain().is_empty());
    }

    #[test]
    fn each_instrument_has_its_own_change_baseline() {
        // One shared baseline would let BTC's quote suppress ETH's, and the missing
        // instrument would look exactly like one nobody is quoting.
        let mut r = Rig::new("multi", 16, MdWritePolicy::OnChange);
        r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
        r.feed(bbo(ETH, dec!(100), dec!(101), SEC));
        r.feed(bbo(BTC, dec!(100), dec!(101), 2 * SEC));

        let got = r.drain();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].symbol_id, BTC.get());
        assert_eq!(got[1].symbol_id, ETH.get());
        assert_eq!(r.md.stats().coalesced, 1);
    }

    #[test]
    fn two_runs_over_the_same_events_write_the_same_bytes() {
        // The property the Phase-5 parity harness rests on, asserted at the level of the
        // file rather than the record: same events in, byte-identical ring out — seq,
        // slot placement, head and tail included.
        fn run(tag: &str) -> Vec<u8> {
            let mut r = Rig::new(tag, 8, MdWritePolicy::OnChange);
            r.feed(bbo(BTC, dec!(100), dec!(101), SEC));
            r.feed(trade(BTC, dec!(100.5), Side::Sell, 2 * SEC));
            r.feed(book(ETH, dec!(2999), dec!(3001), 3 * SEC));
            r.feed(bbo(BTC, dec!(100), dec!(101), 4 * SEC)); // coalesced
            r.feed(bbo(BTC, dec!(100.5), dec!(101), 5 * SEC));
            r.md.flush().unwrap();
            std::fs::read(&r.path).unwrap()
        }
        assert_eq!(run("det-a"), run("det-b"));
    }

    #[test]
    fn publishing_off_creates_no_file_at_all() {
        // `cargo run --bin axon` must stay a process that touches nothing.
        let path = temp_path("disabled");
        let mut c = cfg(&path, 8, MdWritePolicy::OnChange);
        c.enabled = false;
        assert!(MdPublisher::open(&c, WINDOW).unwrap().is_none());
        assert!(!Path::new(&path).exists());
    }

    #[test]
    fn a_ring_that_cannot_be_created_stops_the_session_rather_than_degrading() {
        // Unlike the signal ring, this end *creates* the file: a failure is a bad path
        // or a permissions problem, and retrying cannot fix it. Running on would leave a
        // strategy with no feature feed and a session reporting OK.
        //
        // The unopenable path is one *inside a file*, not a merely-absent directory: an
        // absent directory is only unopenable where the filesystem root refuses writes,
        // which is a Unix property and not a Windows one. A file is not a directory
        // anywhere, and no test running beside this one can turn it into one.
        let blocker =
            std::env::temp_dir().join(format!("axon-not-a-dir-mdring-{}", std::process::id()));
        std::fs::write(&blocker, b"not a directory").expect("write the blocking file");
        let c = cfg(
            &blocker.join("md.ring").to_string_lossy(),
            8,
            MdWritePolicy::OnChange,
        );
        let opened = MdPublisher::open(&c, WINDOW);
        let _ = std::fs::remove_file(&blocker);
        assert!(opened.is_err());
    }

    #[test]
    fn a_counter_past_the_u32_ceiling_wraps_and_only_wrapping_sub_recovers_the_delta() {
        // The hazard ADR-0034 §2 accepted knowingly, asserted here so that the *first*
        // thing anybody who reads these five fields meets is the arithmetic they are
        // obliged to use. Sixty-four bytes leave twenty for the payload, so the counters
        // are `u32` and they truncate; a monitor that subtracted them naively would see
        // a busy publisher go negative — and, if the counters had been *saturated*
        // instead, would see it read as a quiet one forever, which is the wrong answer
        // to the only question this file is asked.
        //
        // Driven by writing `stats` directly rather than by publishing four billion
        // slices, which is the only way to be inside a wrap in under a second.
        let mut r = Rig::new("wrap", 8, MdWritePolicy::OnChange);
        let path = beacon_path(&r.path);

        r.md.stats.published = u32::MAX as u64 - 1;
        r.md.beat(SEC, 0);
        let before = read_md_beacon(&path).unwrap();
        assert_eq!(before.published, u32::MAX - 1);

        // Four more slices, straight across the ceiling.
        r.md.stats.published = u32::MAX as u64 + 3;
        r.md.beat(2 * SEC, 0);
        let after = read_md_beacon(&path).unwrap();

        // The wrapped absolute value, as the number rather than as "small": this is what
        // a reader that printed the field as a total would print.
        assert_eq!(after.published, 2);
        assert!(
            after.published < before.published,
            "the count went backwards, which is exactly why no reader may print it"
        );
        // And the delta, which is exact across the wrap and is the only thing the field
        // is for.
        assert_eq!(after.published.wrapping_sub(before.published), 4);

        // The real total is still here, on the side with room for it. `MdStats` is what
        // the status line prints; the beacon is not a second, smaller copy of it.
        assert_eq!(r.md.stats().published, u32::MAX as u64 + 3);
        assert_eq!(
            r.md.stats().published - (u32::MAX as u64 - 1),
            u64::from(after.published.wrapping_sub(before.published)),
            "the two sides agree about the delta and disagree about the total"
        );
    }

    #[test]
    fn restarting_the_publisher_reinitialises_the_beacon_and_a_reader_mapped_across_it_survives() {
        // Two failure modes from ADR-0034 §4 and its consequences, one test.
        //
        // The first is `O_TRUNC`: truncation takes the file to zero length before
        // `set_len` puts it back, and a monitor with the page mapped takes a `SIGBUS` —
        // a signal, not an error it can handle — for touching a page past EOF during
        // that window. The publisher would be killing its own monitor on restart, at the
        // moment the monitor was most needed. **This test does not cover that instant
        // and cannot**: it is sequential, so it is never inside the window, and swapping
        // `MdBeacon::create` back to `.truncate(true)` leaves it green. What it holds is
        // the observable consequence — the file is a full beacon before and after, and
        // the beat resets rather than the reader breaking.
        //
        // The second it *does* cover: a publisher that unlinked and recreated the file
        // would leave the already-mapped reader on a dead inode, reading a file nobody
        // writes — a monitor that reports a healthy session as dead forever, with no
        // error anywhere. The reader below is opened before the restart and asserted to
        // see the *new* publisher's beats, which is only possible if the same inode was
        // reinitialised in place.
        let path = temp_path("restart");
        let bpath = beacon_path(&path);
        let first = MdPublisher::open(&cfg(&path, 8, MdWritePolicy::OnChange), WINDOW)
            .unwrap()
            .unwrap();
        for _ in 0..5 {
            first.beat(SEC, 0);
        }
        assert_eq!(first.beats(), 5);

        // The monitor, mapped and holding the page across everything that follows.
        let mapped = MdBeaconReader::open(&bpath).expect("a beacon to watch");
        assert_eq!(mapped.read().beats, 5);
        assert_eq!(std::fs::metadata(&bpath).unwrap().len(), 64);

        drop(first);
        let second = MdPublisher::open(&cfg(&path, 8, MdWritePolicy::OnChange), WINDOW)
            .unwrap()
            .unwrap();

        // Full length throughout — never 0, which is the length that would have cost the
        // reader a signal.
        assert_eq!(std::fs::metadata(&bpath).unwrap().len(), 64);
        let restarted = mapped.read();
        assert_eq!(
            restarted.beats, 0,
            "a restart resets the count, not the file"
        );
        assert_eq!(restarted.pid, std::process::id());
        assert!(restarted.running());
        assert_eq!(
            restarted.published, 0,
            "nothing of the old session survives"
        );

        // The inode proof: writes from the publisher created *after* the mapping reach
        // the reader that was mapped *before* it.
        for _ in 0..3 {
            second.beat(2 * SEC, 0);
        }
        assert_eq!(
            mapped.read().beats,
            3,
            "the pre-restart mapping is on the same inode the new publisher writes"
        );
    }
}
