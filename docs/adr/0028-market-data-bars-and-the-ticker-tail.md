# ADR-0028 — A bar record, and what the 48 reserved bytes were for

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0012](0012-market-data-ring-and-multi-record-contract.md) built the market-data ring, and its
consequences named two limits it left in place. `MdSlice` is top-of-book plus the last print, and
48 of its 128 bytes are reserved "for mark price and funding (the Phase-2 ticker increment)". It
also recorded a publisher rule: **`Ticker` and `Candle` never produce a slice** — neither moves a
field the record has.

Both limits had the same consequence and nobody had said it out loud: **no Rust feed could reach a
Python `Bar`.** `axon.strategies.perp_bar` — the only real strategy in the tree — consumes closed
candles, and the ring could not carry one. The strategy could be trained offline from
`candleSnapshot` and could not be shadow-traded from a live session at all. The venue's mark and
funding were in the same position: the core holds them, the risk gate prices against them, and the
Python side had never seen one.

Five questions had to be answered, and each has an answer that looks reasonable until it costs
something.

1. **Is a bar a new record, or a `kind` inside `MdSlice`?** The stride has 48 spare bytes and
   `MdSlice` already has a `kind` discriminant, so the variant looks free.
2. **What fills the 48 reserved bytes** — and does mark/funding belong there or in a record of its
   own? Reserved bytes spent badly are unrecoverable without a stride change.
3. **How does the publisher know a bar has closed?** A `candle` subscription streams the bar that
   is still forming.
4. **Does the write policy apply to bars?** `MdWritePolicy::OnChange` is the shipped default and
   exists to stop a busy feed spending every ring slot on records that carry no news.
5. **Where does the bar ring's path come from?** The obvious answer is a second config field.

## Decision

### 1. A bar is a new record, `MdBar`, on its own ring — and it is 128 bytes, exactly like `MdSlice`

A bar is not a state snapshot. `MdSlice` answers *what is true now*; a bar answers *what happened
over a closed interval*. Three things follow that a variant inside `MdSlice` could not have given:

- **Two consecutive bars with identical OHLCV are two facts, not a repeat.** A flat market prints
  them, and `OnChange` would coalesce the second away. One deleted bar silently shortens every
  rolling feature window downstream, and the strategy computes confidently across a hole it cannot
  see. A bar riding `MdSlice` would inherit a change test written for state.
- **`MdSlice.kind` is the update's *cause*, never the record's type** — ADR-0012 §2 says so in
  those words, and §3 puts the record's type in the *ring header* for exactly this reason. Adding
  `bar` to `[md_slice.kinds]` would make one field mean two things, which is how the next reader
  gets it wrong.
- **The two have no fields in common past the header.** A bar has no top of book; a slice has no
  OHLCV. A union would be half-zeroed on every record whichever way it went.

The stride being identical is deliberate, and it is ADR-0012 §3's argument made concrete rather
than hypothetical. That ADR chose an explicit `record_kind` over `record_size` on the grounds that
equal strides are "a coincidence of the current field lists, not a contract" — and the coincidence
has now happened. `record_size` cannot discriminate `MdSlice` from `MdBar` at all; the kind tag is
the only check standing between a reader and reporting a bar's `open` as a bid price. `record_kind`
is therefore load-bearing rather than belt-and-braces, and it is asserted in both languages
(`a_bar_ring_and_a_slice_ring_are_told_apart_with_no_stride_to_help`,
`test_a_slice_reader_refuses_a_bar_ring_despite_the_matching_stride`).

`MdBar` carries `seq`, `ts_event`, `open_time`, OHLCV, `symbol_id`, `interval_ms`, `flags`,
`schema_version`, `kind` and 52 reserved bytes. Two field choices are worth naming:

- **`ts_event` is the bar's close, at `T + 1 ms`**, and it crosses the ring unchanged from
  `decode_candle`. `T` is the bar's *last* millisecond, so a bar stamped `T` sorts equal to every
  trade printed inside it and an event-time sort may hand a strategy the closed bar before the tick
  that closed it. `axon.strategies.data.CLOSE_STAMP_OFFSET_MS` already agrees, and the agreement is
  load-bearing: while the two halves were one millisecond apart, `align_by_event_time` intersected
  to **empty** and the feature-parity gate failed as "an empty matrix proves nothing", a long way
  from the cause. `bar_from_record` therefore copies `ts_event` rather than re-deriving it from
  `open_time + interval_ms`, which would be a second place for the two languages to drift.
- **`interval_ms` is a number, not an enum ordinal.** A `u32` of milliseconds is self-describing
  across the boundary, lets a consumer compute the next expected `open_time` without a lookup
  table, and tops out at 49 days — longer than any interval any venue quotes.

`flags` carries the two things nothing else on the record can say. `gap_before` means the venue
should have printed a bar between this one and the previous one and did not; `first_bar` means
there is no previous bar for this instrument and interval in this session, so continuity is
*unknown* rather than broken. They are separate flags because collapsing them would either cry gap
on every session's first bar or let a feature window begin mid-history in silence. Both are
distinct from a `seq` gap, which means the *ring* dropped — a different fault with a different
remedy. Nothing interpolates a missing bar: an invented close is a price nothing traded at, and
every return feature computed from it would be measuring our own arithmetic (the rule
`Candles.gaps` already holds to offline).

### 2. The 48 reserved bytes become mark, index, funding and the mark's two clocks

Six `i64`s tiling `[80,128)`: `mark_px`, `index_px`, `funding_rate`, `funding_interval_ns`,
`mark_ts_venue`, `mark_ts_ingest`. `MdSlice` goes to schema version 2; the stride does not move,
which is the entire point of having reserved them.

**Why the tail rides `MdSlice` rather than being an `MdTicker` record.** Because a ticker has no
event time to stamp a record with. Hyperliquid's `activeAssetCtx` carries no venue timestamp
([ADR-0011](0011-ticker-and-mark-price-feed.md)), so an `MdTicker`'s `ts_event` could only ever be
our receipt clock — and a record ordered on the machine's clock does not reproduce on replay. That
is precisely why ADR-0012's publisher refuses to emit a slice on a `Ticker` at all, and that
refusal stands. Mark and funding are *state*, like the top of book; they ride the next slice a
venue-timed event triggers. This resolves ADR-0011's tension rather than dodging it.

**Why both clocks, and not one field plus a flag.** `mark_ts_venue == 0, mark_ts_ingest != 0` says
"a ticker exists and the venue did not timestamp it"; `both == 0` says "no ticker yet". One field
cannot separate those. Where both exist, their difference is the feed latency
(`docs/05-latency-model.md`), and `ts_event - mark_ts_venue` is a mark age that *replays* — which a
receipt stamp never is. Agent M8's Binance measurement is the first evidence that this `Option` was
not just an argument: `ts_venue` is `Some` on Binance and `None` on Hyperliquid, so both cases are
now real and the ring has to carry both.

**What was dropped, and why it was that one.** `open_interest` did not fit; six `i64`s is exactly
48 bytes and narrowing a field to make a seventh fit would let arithmetic convenience decide the
wire layout. Of the five candidates it is the only one whose absence cannot make another number
*wrong*. A missing basis leaves `mark_px` ambiguous between "the perp is bid up" and "the
underlying moved"; a missing interval makes `funding_rate` not approximate but incorrect; missing
clocks make the mark unageable. Missing open interest is one fewer feature. It stays on
`axon_core::Ticker`, where ADR-0011 keeps it for being the only quantity no other feed can yield;
it simply does not cross this boundary yet.

`MdSlice` now has **zero** reserved bytes, and that is the stated cost. The next change to it is a
stride change — or, better, an `MdTicker` record, which becomes possible the day a venue that
stamps its ticker is the one being modelled.

Two rules the publisher applies to the tail, each of which the obvious alternative gets wrong:

- **The mark's clocks are excluded from the `OnChange` change test.** `mark_ts_ingest` is a receipt
  stamp that moves on every `activeAssetCtx` frame, and a live venue pushes those constantly.
  Including it would make every quote differ from the last and turn `OnChange` into `EveryUpdate`
  on any session with a ticker feed — precisely the ring-slot exhaustion the policy exists to
  prevent. A mark that has not moved is the same mark, whatever time it was restated at.
  (`last_trade_ts` *is* in the test, for the opposite reason: two prints at the same price and size
  are different events and the timestamp is what says so.)
- **An unrepresentable ticker value zeroes the tail; it does not drop the slice.** Deliberately
  asymmetric with the quote-price rule. The top of book and the last print on the same record are
  independently exact, and letting one venue's funding precision silence the book feed would be the
  larger wrong. It remains a lie of omission — Python reads the "no ticker yet" sentinel while
  there is one — which is why `MdStats::unrepresentable_mark` counts it.

### 3. The publisher closes bars itself, from the event stream, and the record has no finality bit

A `candle` subscription streams the **forming** bar, several times a minute, each frame carrying a
mid-bar `close` under a close time in the *future*. Publishing one is the purest lookahead
available. `axon_core::Candle` has no `is_final`, so the frame cannot be asked.

So the publisher holds the latest frame per instrument+interval and emits it only when a frame
arrives for a **later** `open_time`: the venue moving on is the only evidence of closure that is
not a wall clock. This is the same rule `axon.strategies.data.closed_rows` applies offline, reached
without a clock — which is what makes the live and research paths agree on *which* bars exist,
rather than agreeing by coincidence.

This is why there is no finality bit on `MdBar` and no `forming` value in `[md_bar.kinds]`.
Finality here is **structural rather than declared**, and the difference is not academic. Agent M8
captured 2 353 real Binance frames and found 111 of 112 `kline_1m` frames in progress — *all
stamped with the same `T`*, so under `decode_candle`'s `T + 1 ms` the last partial and the close
share an `ts_event` and an event-time sort cannot separate them. A publisher that trusted a frame
would hand `perp_bar` a bar that then changed underneath it: training/serving skew that the
feature-parity gate reports as a numeric disagreement a long way from its cause. Keying on
`open_time` makes all of that a non-event, and no venue behaviour has to be trusted for it to hold.

Hyperliquid behaves the same way for the same reason, measured separately: it republishes the bar
it is filling and marks no frame final, and for 65 of 69 observed bars it sends **nothing after
`T`** — it stops republishing and starts the next interval. So there is no closing frame to wait
for on either venue, and "the venue moved on" is not a heuristic standing in for one; it is the
only signal that exists. What that costs is quantified in the consequences below.

The other costs are real and are stated where they land. A bar arrives one venue frame after it
closed (under a second in practice; its `ts_event` is still its own close time, so nothing
downstream is skewed). And **a session's last bar is never published**, because nothing ever proved
it closed. A `Candle::is_final` in `axon-core` would buy exactly one thing — publishing on the
final frame rather than on the next one's arrival, which would close that gap. It is an
improvement, not a requirement, and it is not this ADR's to make.

A frame older than the one held is ignored and counted (`bars_out_of_order`): publishing it would
move a bar backwards in a consumer's event-time series, which is the one thing an event-time feed
may not do.

### 4. `MdWritePolicy` does not reach the bar ring

Both of its variants are pure functions of the event stream and there is deliberately no
time-based cadence — a wall-clock sampler makes the published stream depend on how fast the machine
drained the bus, and that is an input to somebody else's model. A closed bar is *naturally* an
event, not a sample, so it needs no policy: every closed bar is published exactly once under either
setting.

The two publisher rules ADR-0012 §5 established are preserved for bars as written. A full ring
drops the bar and **spends** a `seq`, leaving exactly the gap `MdBarRingConsumer.dropped` computes
loss from; a frame that merely refines a forming bar spends none, so suppression is never mistaken
for loss. The continuity baseline advances only on a *delivered* bar, so a dropped one makes the
next bar report `gap_before` — the ring's loss and the feed's hole are both visible, separately.

### 5. The bar ring's path is derived from the slice ring's, and the banner prints both

`bar_ring_path` inserts `bars` before the extension: `/dev/shm/axon-md.ring` →
`/dev/shm/axon-md.bars.ring`, the convention `CaptureConfig` already uses for its signal log. One
config switch, one derivation, two files.

Derived rather than configured because the failure is asymmetric. Two independent settings let an
operator enable slices, forget bars, and run a bar-driven strategy that starts cleanly, reports
healthy, and simply never has an opinion — a silence indistinguishable from a quiet market. One
switch cannot be half-turned. The implicitness is paid for by the startup banner, which now names
both paths (and says `OFF - Python computes no features from this session` when it is off): the one
place an operator learns whether a session publishes *before* it starts rather than an hour in.

The bar ring is created whenever publishing is on, even for a session subscribed to no candle feed.
A consumer that can always open the file and find it empty is a far better failure than one that
cannot tell "no bars yet" from "wrong path".

## Consequences

- **+** A Rust candle feed reaches a Python `Bar`. `MdBarRingConsumer.read_bars()` returns
  `axon.strategy.events.Bar` events that `StrategyRunner.handle` dispatches to `Strategy.on_bar`,
  which is what unblocks shadow-trading `perp_bar`. Asserted end to end, not argued.
- **+** The venue's mark, index and funding cross the boundary for the first time, on a record that
  replays, with both of the mark's clocks intact.
- **+** ADR-0012 §3's `record_kind` is now doing work no other check could do. The two md records
  share a stride, and both languages refuse the wrong one by name.
- **+** In-progress bars cannot reach a strategy, and that is a property of the publisher rather
  than of the venue — which matters, because both venues have now been measured and *neither* marks
  a frame final: Binance sends in-progress frames all stamped with the same `T`, Hyperliquid
  republishes the bar it is filling and then simply stops. Nothing on either wire could have been
  trusted to say.
- **−** `MdSlice` has **no** reserved bytes left. Open interest, the venue's own mid, and anything
  else from the ticker now need a stride change or a new record.
- **−** `MdSlice` goes to schema version 2, so Rust and Python must be deployed together for this
  ring as well. Consistent with ADR-0012's existing "no mixed-version window", and rings are
  recreated per session, so nothing persistent is invalidated.
- **−** A session's last bar per instrument is never published, and a bar arrives one venue frame
  after its close.
- **− A published bar is the venue's last *observed* frame for that interval, which is usually but
  not always the official bar.** Hyperliquid does not send a closing frame: agent M12 probed the
  live feed for 12.9 minutes and found it republishes the bar it is filling (1 321 frames described
  69 bars; 63 were republished, one 5-minute bar 192 times), marks none of them final, and for
  **65 of 69 bars sent nothing at all after `T`** — it stops republishing and starts the next one.
  So there is no post-`T` frame to wait for; the last frame before the close *is* the bar. When a
  trade prints between that frame and `T`, the published bar is short on `v` (and would be on the
  trade count, if the record carried one). Cross-checked against `POST /info
  {"type":"candleSnapshot"}` over **7 consecutive BTC minutes — one small sample — 6 matched
  exactly and 1 did not**: it was short by `0.001` in `v`, its final WS frame having landed 35 ms
  before the close. **Read the failure *mode*, not the rate:** seven bars say nothing useful about
  how often this happens, only that it happens and why.

  This is the one place to look when an offline `candleSnapshot` recompute does not exactly equal
  the live bar ring. Agent M5's shadow trading ([ADR-0029](0029-shadow-trading-and-the-continuous-diff.md)) diffs
  a serving path against an offline recompute and reports `max_abs_diff`; that number is exactly 0
  today because both halves are cached history. The first time it runs against a live ring, a
  non-zero **volume** column is expected venue behaviour and not a parity break — and a non-zero
  *price* column still is one, because OHLC only moves on a trade the final frame would have
  carried. Waiting for a post-`T` frame is not an available alternative on this venue; querying
  `candleSnapshot` to reconcile would be a second source for the same bar, which is the second
  view of the market ADR-0012 §1 refuses.
- **−** The bar ring's path is not independently configurable, so both rings must live on the same
  filesystem. A `md_ring.bar_path` override is a one-field addition if that ever matters.
- **−** The bar ring uses the slice ring's capacity, which at one bar a minute is enormously
  oversized. Deliberate: a second capacity knob would be a number nobody could reason about, and
  128 bytes a record on tmpfs is not a resource worth optimizing.
- **−** `MdSlice`'s ticker tail is only as fresh as the last venue-timed event. On an instrument
  with a ticker feed and no quotes, the mark does not cross at all. That is the honest consequence
  of refusing to stamp a record with a receipt clock, and the alternative was a ring that stops
  being reproducible.
- **Not closed here:** the recorded `release_ts` is still the core's event-time high-water mark at
  the pass that read the record, not the producer's write time, so a replay judges a signal very
  slightly fresher than the live session did. It looks like a `Signal` field and is not: closing it
  needs the producer's write time *on the clock the core orders by*, and Python has no access to
  that clock. A wall-clock stamp in the reserved bytes would put two clocks on one record and make
  the gap look closed while the number stayed incomparable. It needs a decision about how the two
  sides share a clock, which is an ADR of its own.

See [ADR-0012](0012-market-data-ring-and-multi-record-contract.md) (the ring and the record this
extends), [ADR-0011](0011-ticker-and-mark-price-feed.md) (why the ticker's two clocks stay
separate), [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the codegen and the reserved-byte bet),
[ADR-0022](0022-first-ml-strategy.md) (the strategy this unblocks),
[`contracts/`](../../contracts/README.md), [02](../02-python-rust-boundary.md).
