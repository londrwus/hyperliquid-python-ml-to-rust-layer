# ADR-0029 — Shadow trading `perp_bar`, and what a continuous diff can and cannot observe

**Status:** Accepted · **Date:** 2026-07-26

## Context

Rung 3 of the ladder in [07](../07-parity-and-testing.md) is *"shadow trade: live feed, would-be
orders, continuous diff vs offline"*. It has been the open Phase-5 line since Phase 5 began, and
until an hour before this was written it could not be attempted at all: `MdSlice` carried no bar
kind, so the Rust candle feed — which decodes, normalizes and caches `Candle` with a close-time
`ts_event` — had no path to the Python `Bar` this strategy consumes. [ADR-0028](0028-market-data-bars-and-the-ticker-tail.md)
closed that with a second ring.

So this ADR is the rung, and — more usefully — it is about which of those four words a run in
this repository can actually deliver today. Each has a comfortable wrong answer:

1. **"Live feed."** The obvious reading is *"we attached to a ring, therefore it was live"*. An
   `MdBar` record carries no source marker, so a bar written by the Rust core and one written by
   a harness are byte-identical to the reader. A process that infers *live* from the shape of its
   input prints the word on a run over a file.
2. **"Would-be orders."** `perp_bar` emits a *target position*. Calling the record on the signal
   ring an order is the overclaim [ADR-0018](0018-event-capture-and-golden-replay.md) already
   refuses in the other direction: a replayed order is not a filled order, and a shadow target is
   not even a planned one.
3. **"Continuous diff vs offline."** The obvious offline side is a fresh download of the same
   history. That compares two *feeds* as well as two code paths, and every venue gap reads as a
   parity failure.
4. **"A green rung."** The obvious reading is that the strategy works. Three green gates already
   say it does not: [ADR-0022](0022-first-ml-strategy.md) measured pooled out-of-sample AUC
   **0.5224** and a gross edge of **+1.16 bps** per decision, against **+2.99 bps** for a constant
   short over the model's own decision rows — a selection worth **−1.83 bps**. A fourth green
   number changes none of that, and quoting the AUC as a success is the exact misreading ADR-0022
   was written to prevent.

**And one constraint shaped the whole thing: this workstream may not touch the venue.** The
testnet account and the multi-hour soak belong elsewhere on this fan-out. So the question became
*what is the most of rung 3 that can honestly be climbed without a live session*, and *what
exactly remains unobserved* — because a rung honestly half-climbed and clearly labelled beats a
rung claimed.

## Decision

### 1. The offline reference is a recompute over exactly the bars this run was shown

`axon.strategies.shadow.ShadowTrader` keeps its own recording of every bar it handed the
strategy, and each window is diffed against a recompute over that recording — not against a
separately loaded history. Two reasons, and one consequence that has to be stated rather than
discovered.

The reasons: a second download compares two feeds as well as two code paths, and a venue gap
would then redden a gate about *arithmetic*. And a recompute that started at the window boundary
would be NaN through its own warmup and would disagree with a serving path whose buffer was
already warm — the recompute spans the whole recording for the same reason the serving buffer is
256 bars and not 25.

The consequence: **a bar the ring dropped reached neither side of the diff.** Both sides then
agree perfectly about a window that is silently one observation short, and every windowed feature
spanning the hole covers more calendar time than its window claims. Nothing in the comparison can
see this. `FeedHealth.ring_dropped` is therefore part of the verdict rather than a number in a
log line — it is the *only* thing in a shadow run that can see that fault.

That is also why `gap_before` and a `seq` hole are counted separately and never summed. They have
one symptom and opposite meanings: a venue gap is the market's, is reported and never repaired
(an interpolated bar is a close nothing traded at); a ring drop is ours, and it is the one that
invalidates the number printed beside it.

The same recording makes each window's offline side a **declared** owed set, and the monitor is
wired `scope="declared"` to say so. Under an inferring scope, rows a serving path missed at a
window's *opening* are excused as a late join **and taken out of `n_in_scope`** — the run's own
denominator shrinking to fit the damage while the ratio still reads perfect, which is the
invisible-denominator bug one level further out than the one ADR-0030 closed. There is nothing in
the data to catch it with: a path blind through a window's opening rows has exactly the same first
stamp as one that started on time. Declaring is the only way out.

An earlier revision of this module closed it differently — counting `n_offline_before` itself and
failing the run on it — and that was wrong for a reason worth recording, because it is a tempting
shape. Once the scope is declared that bucket is empty *by construction*, so the check could only
ever read zero: **a check that cannot fire is worse than no check, because it reads as
protection.** The blindness now arrives as `n_offline_within` and fails the window through
`Coverage.complete`, before any run-level total is consulted.

### 2. There is exactly one answer to "how many rows were owed"

[ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) made the alignment carry its
own denominator, and left `axon.strategies.training.feature_parity_gate` as the last caller with
a *second*, private one — `ServedFeatureParityReport`, which recomputed the same count. Two
implementations of one question agree on the day they are written and drift afterwards, and the
drift lands in the harness that exists to detect drift.

That class is gone. The gate now calls `aligned_feature_parity` and trims the reference it hands
it to `owed_rows(candles, replayed)` — a boolean mask, true exactly where a cold-started replay of
the shown span can produce a finite row. `Coverage` counts it, and `coverage.n_in_scope` *is* the
old `n_expected`, computed once. The values compared are still the **full-history** recompute's,
because a cold start matched against a recompute that saw everything is the whole point of the
`replay_bars_from` shape.

`owed_rows` does not need to intersect with the full recompute's finite rows, and the reason is
the same invariant the rung rests on: every span-finite row is also full-history finite, because a
window that fits inside the shown span is the same arithmetic on the same numbers either way. An
expanding transform would break that containment, and ADR-0022 §2 is why the spec has none.

**And the trim is load-bearing only because the gate says so.** `Coverage`'s default is to infer
the owed span from the online side's own first stamp — right for a live monitor whose window opens
mid-history, and blind here, because *a serving path that produced nothing for its first k owed
rows has exactly the same first stamp as one that started on time.* The two are indistinguishable
from the data, so excusing them is the honest default and a gate has to declare its way out. This
one passes `scope="declared"` (ADR-0030 §1a), which says the reference handed over *is* the owed
set; a late start then lands in `offline_within` and reddens the gate. That is a mode rather than
a span argument on purpose: a span would be a second description of the same fact, and a second
description can be narrower than the data and silently re-excuse the very rows this closes.

### 3. Drift is not measured, and the report says *not measured* rather than *stable*

The monitor is constructed with no reference sample. That is a decision with a measurement behind
it, not caution: ADR-0030 found that over `perp_bar`'s own history in 256-row windows, **with
feature parity green on every window**, PSI passes the conventional 0.25 band on 18 of 20 BTC
windows and peaks at 5.97. A shadow monitor that alarmed on drift would alarm forever, and an
operator who learns to ignore the drift line has learned to ignore the parity line — which has no
false positives at all. [ADR-0016](0016-feature-spec-and-parity-gates.md) §5 already records the
bands as the industry convention rather than something derived from these features; this is the
bill.

*Not measured* and *stable* are opposite statements and only one of them is true, so the report
prints the first one, with the reason.

### 4. The silence deadline is measured in bars, never in seconds

`DEFAULT_SILENCE_AFTER_NS` is 60 s, which is right for a quote feed and absurd here: a perfectly
healthy hourly `perp_bar` session says nothing for an hour at a time and would alarm as a dead
feed a minute in. The deadline is `SILENCE_AFTER_BARS` intervals, and the floor is **two**, not
one, because ADR-0028's publisher emits a bar only once the venue starts the next interval — a bar
legitimately arrives one frame *after* its own close, so a one-interval deadline fires on every
healthy bar in the session.

Silence is also the only thing here allowed to read a wall clock, for ADR-0030's reason: an
event-time deadline for *"nothing has arrived"* can never expire, because the clock that would
advance it is the thing that stopped.

### 5. A duplicate or backwards bar is refused, not repaired

Two bars for one instrument at one close time is a republished bar. Appending it twice makes every
window from that point one observation wide in the wrong place — **on both sides of the diff**, so
the comparison stays green over a corrupted recording. `ShadowError` says so at the first one, and
says the same about a bar that arrives strictly behind its predecessor, because a diff aligned on
event time cannot align a series that goes backwards.

Bars for other instruments are dispatched to the runner (so its event clock advances exactly as a
live session's would) and recorded for none of them: the recompute is over one series, and a mixed
one is nine columns of nonsense.

### 6. "Live" is an operator's attestation and is never inferred

`ShadowReport.venue_attested` defaults to `False` and nothing in the process may set it. The ring
carries bars, not provenance; `RingBarSource.describe()` says so in the string that reaches the
report, and the report prints **NOT ATTESTED AGAINST THE VENUE** unless the operator passed
`--attest-venue`. The claim is available and it has to be made out loud, which is the same shape
[ADR-0020](0020-runtime-intent-source.md) gives "read-only": a property nobody can arrive at by
accident.

### 7. A would-be order is a target, and it is read back off the ring

`ShadowSignal` is what the Rust planner *would have been handed*. Turning one into an order needs
a book to price against, an instrument's tick and lot grid to quantize onto
([ADR-0025](0025-instrument-precision-and-rounding.md)) and knowledge of what is already resting;
none of that exists in a shadow run, so nothing here says what order it would have become, what it
would have queued behind, or what it would have filled at.

It is decoded from a record popped off a real SPSC ring rather than out of the context, because
crossing the ring is the part that can fail. The strategy runs under the real
`axon.live.StrategyRunner`; nothing on the Python half is a stand-in.

## Result — the actual numbers

**No live Hyperliquid session was involved.** Nothing in this ADR was fed by a venue socket.
Three sources were driven, all of them real venue prints:

| source | what it exercises | what it cannot |
|---|---|---|
| cached mainnet `candleSnapshot` history (`data/candles/`) | the serving path, the runner, the signal ring, the continuous diff | the market-data ring |
| the committed 800-bar fixture | the same, offline in the default gate | the same |
| a real `MdBar` ring, written by `publish_bars` | additionally `MdBarRingConsumer`, the ring header's `record_kind`, the continuity flags, `RingBarSource` | the **Rust publisher**, which is where bar closure is *derived* |

**The continuous diff, with its denominator**, over the full cached mainnet history (208 days,
4,999 hourly bars per coin, no gaps), with the registered artifact `perp_bar_xgb@1` in the loop:

| | BTC | ETH |
|---|---|---|
| bars shown | 4,999 | 4,999 |
| decision rows served | 4,975 | 4,975 |
| **rows compared / rows owed** | **4,975 / 4,975** | **4,975 / 4,975** |
| windows | 79 | 79 |
| `max_abs_diff` | **0.000e+00** | **0.000e+00** |
| ring drops · venue gaps | 0 · 0 | 0 · 0 |

The offline gate reproduces unchanged through `aligned_feature_parity`: model parity `PASS` at
`TREE_EPS = 0` with 0 flips, feature parity `PASS` at `max_abs_diff = 0` over 4,975 rows × 9
columns per coin — now printed as `coverage=4975/4975` where it previously printed
`coverage=unchecked` — drift worst PSI 0.1166, and the walk-forward numbers identical to ADR-0022
to the last digit.

**And the number a shadow run produces that a walk-forward cannot: the turnover.**

| | BTC | ETH |
|---|---|---|
| would-be target changes | 1,446 | 1,566 |
| bars per change | 3.46 | 3.19 |
| position sides emitted | 2,413 | 2,677 |
| maker fees on that path | **3,619.5 bps = 36.2% of notional** | 4,015.5 bps = 40.2% |
| taker fees on the same path | 10,858.5 bps = **108.6% of notional** | 12,046.5 bps = 120.5% |

This is a **count of what was emitted, priced at the published schedule** — not a P&L. ADR-0022
refuses to net costs off the measured edge because that needs a turnover *model*, and a turnover
model beside the evaluator would be a second implementation of the strategy's own hysteresis. This
is the other thing: the turnover the strategy actually emitted, observed.

Two readings of it are safe, and one is not. Safe: **`urgency = 0` is not a preference, it is
load-bearing** — the same path at taker fees costs more than the whole position over 208 days.
And: **the strategy changes its mind every 3.5 hours against a four-hour label horizon**, so the
holding period and the question the model was asked barely overlap. Not safe: dividing the fee
total by the gross edge. The edge is a mean four-hour forward return per decision and the
turnover is per bar; the positions overlap, and no honest single number joins them without a
holding-period model this repo deliberately does not have.

**What none of this says.** It says the serving path computes what the offline recompute computes,
continuously, over 4,975 rows per coin, with the denominator visible. It says nothing whatsoever
about the market. ADR-0022's reading stands unmodified and this run strengthens rather than
softens it: a model with a −1.83 bps selection, which changes its mind every 3.5 hours, is not
tradeable, and a fourth green gate is a statement about fidelity.

## Consequences

- **+** Rung 3's machinery exists, is tested, and has been driven over 208 days of real venue
  prints with the real artifact in the loop — 4,975 rows per coin, `max_abs_diff` exactly 0, and
  the denominator printed beside it in every window rather than at the end.
- **+** The whole consumer half of ADR-0028 has been exercised end to end: real `MdBar` records,
  a real ring with its own `record_kind` header, `MdBarRingConsumer`, the continuity flags,
  `RingBarSource`, `StrategyRunner`, the signal ring, the diff.
- **+** "How many rows were owed" has one implementation again. `ServedFeatureParityReport` is
  gone and the green it protected is unchanged — the same 4,975 rows, now with
  `coverage=4975/4975` in the summary line instead of `coverage=unchecked`.
- **+** The turnover is a *measurement* rather than an assumption, and it is the first number in
  this repository that says something about `perp_bar`'s cost side from the strategy's own
  behaviour instead of from a fee table.
- **−** **This has not been run against the venue, and is not called proven.** No bar in any run
  above crossed a Hyperliquid socket into the Rust core. The Rust publisher — which *derives* bar
  closure from the venue starting the next interval, the one interesting decision it makes — has
  been stood in for, and a stand-in is not evidence about the thing it stands in for.
- **−** **An hourly strategy cannot climb this rung inside a session, and that is arithmetic.**
  The spec's longest window is 24 bars, so the serving path produces its first row on the **25th**
  closed bar; the publisher emits a bar only when the next one starts, and a session's last bar is
  never published. A 1h shadow session therefore has no opinion for **25 hours** — measured, not
  estimated — and its first *decision* came at exactly that bar in this run. At `m1` the same
  warmup is 25 minutes, and the model was fitted on hourly bars, so a minute feed would be
  serving a spec on data it was never trained on. Whoever runs the live half should plan a
  multi-day session or accept that they are watching warmup.
- **−** **A dropped bar is invisible to the diff by construction**, because the reference is a
  recompute over the bars the serving path was shown. `ring_dropped` is the whole defence and it
  is a counter, so a publisher that dropped *and* reset its sequence would defeat it. Closing that
  needs the market-data beacon ADR-0030 specified and nobody built.
- **+** **An online side that starts late reddens both the gate and the run**, which the private
  row count also did, so the collapse gave up nothing. Both close the same way — by *declaring*
  the scope rather than by adding a count. The gate: a serving path blind for its first 20 owed
  rows reads `feature parity FAIL: rows=756 … max_abs_diff=0.000e+00 … coverage=756/776`, naming
  *20 owed row(s) the online side never produced*. The run: the same blindness lands in one
  window's `n_offline_within` and takes it to `ALARM`, with the denominator still 776.
- **−** **The late start that was exercised is synthetic.** Both tests suppress rows from a
  subclass; no real serving path has ever started late here, and the failure mode this guards —
  a stricter live NaN guard, a defensive clamp, a Rust backend that returns NaN on inputs Python
  accepts — is still only reasoned about. What is now true is that it would be *seen*.
- **−** **This module was the third place one bug had to be closed in a single increment**, after
  the comparison and the gate ([ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) §1b).
  Each closure was found by the component built on top of the previous one, and this one was no
  exception: the shadow loop inherited a correct default that was wrong for the arrays it was
  passing. Three closures is not evidence the design was right first time, and the next consumer
  of a `Coverage` should expect to find a fourth rather than assume the class is now safe.
- **−** **The turnover was measured with the shipped artifact over the whole history**, most of
  which it was fitted on. The direction of that bias is at least known: a less confident model
  crosses a hysteresis band more often, not less, so an out-of-sample turnover is likely higher
  than the 1,446 changes above rather than lower.
- **−** **`publish_bars` is a second writer of bar records.** It exists so the consumer side can
  be exercised without a venue and it is marked as a harness in its own docstring, but it is a
  place where the flag semantics could drift from `axon_runtime::mdring`'s. It does not derive
  closure — it is handed bars a loader already proved closed — which is the part that matters.
- **−** **Nothing here alarms on drift, so nothing here would notice a regime change.** That is
  the correct default given a ~90% false-positive rate on these features, and it means a shadow
  run is silent about the one thing that would actually invalidate a fitted model.
- **−** **The recompute is over the whole recording, so a window costs O(n·w).** At an hourly
  cadence that is microseconds against an hour and it does not matter; a tick-cadence shadow run
  would need a trailing recompute bounded at the spec's longest window, which the finite-lookback
  rule makes exact rather than approximate.

## Amendment — 2026-07-26: the first live socket, and the fault the diff could not see

The Result section above opens **"No live Hyperliquid session was involved"**, and the
consequences say this **"has not been run against the venue, and is not called
proven."** Both sentences are now historical. `ShadowTrader` has been driven from the
Rust core's bar ring, fed by a live Hyperliquid **testnet** WebSocket, on **m1** bars,
for **63 minutes**, with the registered artifact `perp_bar_xgb@1` in the loop, over two
instruments read by one consumer. The session config is
`scripts/sessions/shadow-testnet-m1.toml`: `intent.enabled = false` and the dead-man's
switch **off**, so the session emitted no signed `/exchange` action of any kind — its
own banner reads `intents OFF` and `0 actions/day` — and it shut down on `open 0 flat`.

### 1. The transcript

`--attest-venue`, `--window 12`, one `RingBarSource` fanned out to two traders:

| | BTC (symbol 3) | ETH (symbol 4) |
|---|---|---|
| bars recorded | 58 | 62 |
| **bar coverage — delivered / what the cadence promised** | **58 / 63 (92.1%)** | **62 / 64 (96.9%)** |
| decision rows served | 30 | 36 |
| **rows compared / rows owed** | **30 / 30** | **36 / 36** |
| windows compared · skipped as warmup | 3 · 2 | 4 · 2 |
| `max_abs_diff`, each of nine columns | **0.000e+00** | **0.000e+00** |
| ring drops · `gap_before` flags | 0 · 7 | 0 · 7 |
| would-be targets | 1 | 1 |

`max_abs_diff` is exactly zero on every column, price and volume alike, and the serving
path produced every row it owed. The claim rung 3 was built to make holds against a
live socket, at the bit.

### 2. The headline number was at its most reassuring where the feed was worst

`PASS`, `max_abs_diff=0.000e+00`, `coverage=30/30` — and the feed delivered **58 of the
63 minutes in its own span**. Every field a reader looks at is identical to a flawless
run. This is the shape [ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md)
named one level up, now observed one level down rather than argued: **an intersection
cannot disagree with a bar that is not there, and neither can a recompute over the same
recording.** `BarCoverage` is the arithmetic that catches it, on the bars' own close
times — `(last − first) / interval + 1` — and never on a wall clock, which would
measure how long the reader happened to be attached.

### 3. The failure mode ADR-0028 predicted is not the one that bit

[ADR-0028](0028-market-data-bars-and-the-ticker-tail.md) expected a **non-zero volume
column** on the first live diff, because a published bar is the venue's last *observed*
frame and a trade printing after it is missing from `v`; agent M12 measured one short
bar in seven. Reconciled against `POST /info {"type":"candleSnapshot"}` over this run's
own span:

- **BTC: 58 of 58 delivered bars byte-identical to the venue's own — all five OHLCV
  fields, max abs diff 0 in fixed-point integers. ETH: 62 of 62, likewise.** Across
  both passes, **0 of 145 bars was short on volume**, and no price field disagreed on
  any bar, so there is no parity break to report. Stated as a count and not a rate:
  145 bars say the mode is rarer than one-in-seven and nothing more.
- **But 6 of 64 BTC minutes and 2 of 64 ETH minutes never reached the ring at all**,
  and the venue lists every one of them. All eight are `n = 0` trades, `v = 0`,
  `o = h = l = c`: **Hyperliquid's `candle` subscription sends no frame whatsoever for a
  minute in which nothing traded, while `candleSnapshot` synthesizes a flat bar for
  it.** ADR-0028 §3's rule is *"the venue moving on is the only evidence of closure"* —
  a minute the venue never described is one it never moves on from, so the publisher is
  right and the bar is simply absent. The next delivered bar carries `gap_before`,
  which is honest but counts **one** flag however wide the hole.

So the live-versus-offline disagreement was not in a column. It was in the **row set**,
and the volume column everyone was told to watch was clean.

### 4. `clv` is undefined on a tenth of live m1 bars, and never on the history it was fitted on

`close_location` is `(c − l) / (h − l)`, which is `0/0` when every trade in a bar
printed at one price. On this tape that is **6 of 58 BTC bars and 5 of 62 ETH** — and
**0 of 4 999** hourly bars in the cached history, and **0 of 900** in the committed
mainnet m1 fixture. It is a property of a thin tape, not of the interval. `finite_rows`
requires every column, so such a bar produces no row on *either* side; both agree, the
owed denominator shrinks to match, and the run correctly reads `30/30` rather than
`30/34`.

The consequence is about the strategy rather than the harness: **`perp_bar` had an
opinion on 30 of the 63 minutes in its span — 48%**, against 99.5% over hourly history
(4 975 of 4 999). Half of that loss is minutes the feed never delivered and half is
minutes too quiet for the spec to be defined on. Neither would have surfaced offline.

### 5. Four checks fired on a perfectly healthy feed, and each printed FAIL

Every one was unreachable from a file, and each would have been read as a broken run:

1. **The silence deadline was measured against the wrong clock.** `SILENCE_AFTER_BARS`
   is 2.5 **bar** intervals, but `ParityMonitor`'s progress clock advances when a
   **window** compares something, and a window is `window_bars` bars wide. Between two
   healthy flushes the monitor is legitimately silent for far longer than the deadline
   and answers `SILENT` to every question, which outranks `WARN` and takes
   `MonitorReport.passed` false. Measured: **296 SILENT verdicts in 160 seconds** of a
   feed delivering a bar a minute, before one window had closed.
   `ShadowTrader.heartbeat` now asks only once a **bar** is overdue.
2. **A window inside the spec's warmup was graded as a window that saw nothing.** With
   a 12-bar window and a 25-bar warmup the first two windows owe nothing and produce
   nothing, and *"a window that compared nothing is not a window that agreed"* is true
   of a dead feed and false of a warmup. Such a window is now counted and not offered —
   **only when the online side is empty too**, because rows the reference has nothing at
   are `online_unmatched` and must still reach the monitor. Invisible offline for an
   arithmetic reason: the shipped 64-bar window is wider than the warmup, so every
   history run has finite rows in its first window.
3. **A quiet instrument was graded as a dead feed.** Because the venue omits an empty
   minute entirely (§3), one instrument can go several intervals without a bar while
   the session is healthy — a live pass logged **15 consecutive SILENT verdicts on ETH
   while BTC's bars kept arriving**. The deadline is now armed by **any** bar the
   trader is dispatched, not only its own: another instrument's bar is proof the feed is
   alive, and an instrument's own silence is evidence of nothing. What that gives up is
   the ADR-0020 §7 shape — one subscription returning and another not — which stays
   *visible* as a collapsed `BarCoverage` rather than as an alarm. A single-symbol run
   has nothing to appeal to and degenerates to the old behaviour, which is an argument
   for shadowing two.
4. **A recording's age was printed as a feed's latency.** `ArrivalLag` subtracts a
   bar's own close from the wall clock at which it arrived; over a cached history that
   is the age of the file, and it printed *"876 would be refused as Expired"* on a
   flawless offline run. Liveness is not inferable here (§6), so neither is the meaning
   of the subtraction: the line is gated on `venue_attested` and gives the reason
   instead of the number.

The pattern deserves stating on its own, because it is four for four: **every defect
this session found was a check that fires on a healthy feed, not one that stays quiet on
a broken one.** The false alarm is the more dangerous failure — it is the one that gets
a real check deleted.

### 6. A would-be order on an m1 bar feed is born too old to be admitted

`StrategyRunner` stamps a signal with the **event's own** `ts_event` and never with a
wall clock, so a bar strategy's signal carries the bar's *close*, and the bar cannot
arrive before it. The live intent path judges that stamp against
`CoreHandler::last_ts()` — which agent P1 measured running **1 564 ms behind wall
time** on this venue — under `intent.max_signal_age_ms`, shipped at **2 000 ms**.
A shadow run reaches none of that, which is exactly the gap: **nothing here can refuse
a signal, so the harness will cheerfully report would-be orders a live session would
have dropped.** `ArrivalLag` therefore measures the age and prints the arithmetic.
Measured over a second live pass on the same session:

| | BTC | ETH |
|---|---|---|
| bars aged | 14 | 11 |
| arrival after own close, min / median / max | **990 / 6 076 / 30 675 ms** | **1 049 / 10 916 / 110 323 ms** |
| age at the pass (arrival − 1 564 ms) | −574 … +29 111 ms | −515 … +108 759 ms |
| **would be refused as `Expired` at the 2 000 ms ceiling** | **10 of 14** | **9 of 11** |

**19 of 25 would-be targets could not have reached the planner.** The cause is
ADR-0028 §3's closure rule meeting a thin tape: a bar is published when a frame for a
*later* `open_time` arrives, and on a feed averaging three frames per bar that frame can
land most of a minute into the next interval — or, if that interval is empty, not until
the one after, which is the 110-second case. The venue's own republication rate over
this session was **761 frames describing 175 bars, median 3 per bar, max 19, and 14
bars described by exactly one frame**.

Two readings are available and only one is safe. Safe: **a bar-driven strategy cannot
use the shipped 2 000 ms admission ceiling on this venue** — `intent.max_signal_age_ms`
has to be raised to at least the strategy's own `ttl_ms`, and the session config here
sets 60 000 ms with that reasoning written beside it. Not safe: concluding the signal is
*stale* in any market sense. Its `ts_event` is the bar's close, the bar is the last
closed bar, and no later one exists; what expires is the admission window, not the
opinion.

### 7. What this changes, and what it does not

- **+** Rung 3 has seen a live socket. The `−` consequence *"this has not been run
  against the venue, and is not called proven"* is discharged for the consumer half:
  the Rust publisher, `MdBarRingConsumer`, the continuity flags, `RingBarSource`,
  `StrategyRunner`, the signal ring and the continuous diff all ran on venue bars, and
  `max_abs_diff` is exactly 0 over every owed row.
- **+** The `−` consequence *"an hourly strategy cannot climb this rung inside a
  session"* is discharged by choosing m1: a 25-bar warmup is 25 minutes, and the first
  decision row landed at exactly that bar.
- **−** **Nothing here placed an order, and the whole planner half remains untouched.**
  A `ShadowSignal` is still a target the planner *would* have been handed, with no book,
  no grid and no knowledge of what is resting — and §6 now adds that most of them would
  not have been handed over at all.
- **−** **Ninety-four minutes on one venue is one tape.** The 0-of-145 volume result and
  the 8-of-128 missing-minute rate are counts from a testnet running 2–30 trades a
  minute. Mainnet BTC has no zero-trade minutes and would show neither this nor §4.
- **−** **The diff still cannot see a bar that is not there**, and this run is the
  evidence rather than the argument. `BarCoverage` makes it *visible* and deliberately
  does not make it *fail*, on `feed_gaps`' reasoning: a minute in which nothing traded is
  a minute the venue may legitimately not print, and a run that failed on that would
  teach an operator to ignore it.
- **−** **`ring_dropped` was 0 for the whole session**, so the one feed fault that *does*
  fail a run was never exercised live. It remains tested only against a harness writer.

See [ADR-0022](0022-first-ml-strategy.md) (the strategy, the numbers, and what a green ladder may
claim), [ADR-0028](0028-market-data-bars-and-the-ticker-tail.md) (the bar ring this consumes and
its closure rule), [ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) (the
monitor and the `Coverage` this run is measured with), [ADR-0016](0016-feature-spec-and-parity-gates.md)
(the three gates), [ADR-0018](0018-event-capture-and-golden-replay.md) (why a recording is not a
counterparty — the same refusal, one rung up), and [03](../03-ml-fidelity-and-features.md)
(training–serving skew, the thesis all of this implements).
