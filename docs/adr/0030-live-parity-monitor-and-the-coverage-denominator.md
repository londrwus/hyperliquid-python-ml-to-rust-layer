# ADR-0030 — The live parity monitor, and the denominator the intersection hid

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0016](0016-feature-spec-and-parity-gates.md) §4 decided that feature parity aligns on
event time rather than row position, and gave the reason: the online path samples while the
offline path recomputes a contiguous window, so one dropped sample shifts every subsequent row
and a positional comparison reports total divergence for a feed that is perfectly correct.

That decision is right, and it created a second failure that is strictly worse than the one it
fixed. **An intersection cannot disagree with a row that is not there.** A serving path that
emits a feature row on half the bars is compared on that half, agrees with it to the last bit,
and reports

```
feature parity PASS: rows=2488 cols=9 max_abs_diff=0.000e+00 ... mismatched=0
```

— every field a reader looks at identical to a healthy run, except `n_rows`, which nobody
reads. The gate's headline number is at its most reassuring exactly when the feed is at its most
broken. A stricter live NaN guard, a defensive clamp, and a Rust backend that returns NaN on
inputs the Python path accepts all fail in that shape: silently, and green.

The repair phase closed this for exactly one caller — `axon.strategies.training`'s
`ServedFeatureParityReport`, which recomputes what the serving path *owed* and holds it to that
number. That is the correct check and it lives in the wrong place: it is a property of the
alignment, and every other caller of `align_by_event_time` — including the live monitor this ADR
adds — inherits the blind spot.

Meanwhile [03](../03-ml-fidelity-and-features.md) lists *"a live parity monitor (the mandatory
backstop): sample live feature vectors, recompute via the offline path, alarm on divergence"* as
the fourth and last item of its feature-parity strategy, and [08](../08-roadmap.md) has carried
it as the open Phase-5 line: the gates exist, and nothing runs them continuously.

Four questions had to be answered, and each has a comfortable wrong answer.

1. **Where does coverage get asserted, given that some absence is legitimate?** A cold-started
   replay and a monitor window that opens mid-history both leave earlier offline rows unmatched.
   "Every offline row must match" is the obvious rule and it fires on every healthy run, which is
   how a real guard gets deleted rather than fixed.
2. **Should the aligner raise?** "Fail on them instead of dropping them" is the requirement, and
   the obvious reading is `raise`.
3. **What does an alarm *do*?** "Flatten the book" is the obvious answer for a safety monitor.
4. **What does the monitor report when it compared nothing?** The obvious answer is whatever the
   thresholds say, and every threshold is satisfied by an empty sample.

## Decision

**1. The alignment carries its own denominator, split by whether the absence was ever the online
side's fault.**

`align_by_event_time` returns an `Alignment` — a `tuple` subclass, so `left, right =
align_by_event_time(...)` and `idx[0]` still work and no existing call site changed — carrying a
`Coverage` of seven counts. The unmatched rows are classified against the online side's **own
event-time span**:

| bucket | meaning | verdict |
|---|---|---|
| `offline_before` | offline rows earlier than the online side's first event | **fine** — it was not running yet |
| `offline_within` | a gap *inside* the span it was running for | **failure** — the blind spot |
| `offline_after` | offline rows past its last event | **failure** — a feed that stopped |
| `online_unmatched` | an online row the reference has nothing at | **failure** — different events, not missing ones |

The leading/interior/trailing split is the load-bearing part. An "inside the span" rule alone
excuses a feed that produced perfect rows and then died — every remaining offline row is outside
the span, which is exactly the excuse a cold start legitimately uses. Both directions have a
test named after them.

**1a. Which rows were owed is the caller's to state, and `scope` is where it says so.**

The rule above infers the owed span from the online side's own first stamp. That inference is
right for a monitor and wrong for a gate, and the difference is not a matter of taste: a serving
path that was blind through its first *k* owed rows produces **exactly the same first stamp** as
one that started on time, so under inference alone the two are indistinguishable and excusing
them is the only honest answer. A gate knows better, and must be able to say so.

`scope` says **what the `offline_ts` argument means**:

- `"observed"` (default) — a *reference* recompute the online side may legitimately have joined
  late. Right for a monitor window that opens mid-history, and for a cold-started replay matched
  against a full-history recompute.
- `"declared"` — exactly the rows the online side owed; the caller has already trimmed it.
  Nothing is out of scope, and a late start is a fault.

Two shapes were rejected. A **named mode alone** (`"monitor"` / `"gate"`) is insufficient on the
facts: `feature_parity_gate` compares a 150-bar replay against a 4,999-bar history, and no mode
can recover "the last 150 bars" from two arrays that do not contain it. An **`online_span=(lo,
hi)` tuple** is sufficient but strictly more dangerous, and the reason generalizes: a span is a
*second description of the same fact*, and a second description can disagree with the first. A
span narrower than the truth silently re-excuses the leading rows — the exact bug being closed —
and nothing can detect it, because the rows in question are missing precisely when it matters.
Trimming the reference instead makes the two descriptions one, and a mis-trim is then caught in
**both** directions: too narrow and the online side's surplus rows land in `online_unmatched`;
too wide and the surplus lands in `offline_within`. Both are failures with names, and both have
a test. The claim under `"declared"` is about the array the caller has just built, so there is
nothing left for it to disagree with.

This is the second call of this kind made here on evidence rather than taste — the first was
backing `on_gap="raise"` out of the default (§2 below).

**1b. The same bug has three levels, and the scope is stated once for all of them.**

This bug class was found three times in one day, at three altitudes, and it is worth naming
because the third instance was inside the component built to prevent the first:

1. **One comparison.** `align_by_event_time` dropped rows the online path never produced, so a
   half-blind feed reported a perfect zero over the fraction it did produce. Closed by `Coverage`.
2. **A gate over a known span.** With the reference trimmed to the owed rows, the inferred-span
   rule still excused a *late start* — the leading rows left the denominator. Closed by §1a.
3. **A run of windows.** `ParityMonitor` aligned every window `"observed"`, so a driver whose
   window reference *is* the owed set — `axon.strategies.shadow` — had a serving path blind at a
   window's opening excused, and the run's own total denominator shrank to fit the damage.

Every instance is the same sentence: *an intersection cannot disagree with a row that is not
there, and whoever knows which rows should have been there is not the one doing the intersecting.*
So `MonitorConfig.scope` carries the answer down to the aligner rather than each driver
reinterpreting the counts it gets back. That is not tidiness: two places holding the same
assumption can disagree, and here they demonstrably did — the monitor called such a window `OK`
while a gate over the identical data returned `FAIL`. A test now pins that they cannot.

One substantive consequence follows, and it is not just a knob. `SILENT` means precisely *"I
cannot tell whether there was anything to compare"* — that is why it is a level of its own rather
than a synonym for `OK`. A declared scope has already answered that question, so a window that
owed rows and compared none is `ALARM` immediately, with no silence deadline left to resolve:
waiting sixty seconds would be waiting for information already in hand. A declared window that
owed *nothing* stays `SILENT` — no bar closed, nothing was due, and alarming there would fire on
every quiet window, which is exactly how the drift line earned its ceiling in §5.

`Coverage.complete` additionally requires `n_matched > 0`, because a comparison of nothing is
not a comparison that agreed, and every other clause is satisfied trivially by an empty
intersection. `Coverage.n_in_scope` — matched + within + after — is the denominator printed on
the head line, deliberately **not** `n_offline`: counting the legitimately-early rows would make
a healthy cold-started replay read as 8% covered.

**2. The aligner reports by default and raises only on request — and the *fused* call is the one
to reach for.**

`on_gap="report"` is the default. Three reasons, in ascending order of weight:

- ADR-0016 §7 already decided this for every other gate in the package: **gates return reports,
  not booleans**, and a raising aligner is a boolean with extra steps.
- A live monitor must alarm rather than die, so a raising aligner cannot be on its path at all.
- An aligner that raises on a bad alignment cannot be used to *demonstrate* one — which is
  precisely what the one-millisecond candle-stamp regression test does.

This was not a preference. `on_gap="raise"` was implemented first and shipped as the default;
it turned `feature_parity_gate`'s report for a deliberately half-blind serving path into an
exception and broke the test that asserts that path fails *as a report*. The default was flipped
on that evidence.

So the general close is not in the aligner's control flow. It is that **`aligned_feature_parity`
aligns and compares in one call**, and therefore always has the coverage to fold into the
report. The two-step form is two steps precisely because the first one can be thrown away: a
caller that keeps only the index arrays has discarded the record of what did not match, and what
did not match is the half of the verdict that fails silently. `FeatureParityReport.passed` is now
`every compared cell agreed AND every owed row was compared`, and a report built without an
alignment prints `coverage=unchecked` rather than letting the absence read as completeness.

**3. An empty intersection names itself.**

A candle's `ts_event` is `T + 1 ms` in both languages (Hyperliquid's `T` is the bar's last
millisecond, so a bar stamped `T` sorts equal to the trades inside it). While the two halves
were one millisecond apart, hourly bars shared no event time at all, `align_by_event_time`
intersected to empty, and the gate failed as *"a parity gate over an empty feature matrix proves
nothing"* — true, and a long way from the cause.

The alignment now computes the signed offset from each online stamp to its nearest offline
stamp. When every one of them is the same constant, that is not a data gap, it is two stamping
conventions, and the report says so — naming the 1 ms case explicitly, because it is the one
this codebase has already paid for. The note travels onto the *parity* report, not just the
alignment, because the alignment is the object a caller throws away and the report is the one
that reaches an operator.

This is also why zero overlap does not raise even under `on_gap="raise"`: it is already a hard
failure downstream and cannot be missed. What silently passes is the *partial* intersection, and
that is what the strict mode is for.

**4. The monitor is a state machine over windows, and silence is a verdict.**

`axon.parity.monitor` runs the existing gates — it reimplements no comparison, because a second
implementation of the thing whose single implementation is the entire point would be the
original sin `docs/03` names. What it adds is what a CI run does not need: state across windows,
and a verdict for a window that compared nothing.

`Level` is `OK < WARN < SILENT < ALARM`. `SILENT` sits above `WARN` because a window that
compared nothing is not evidence of health, and below `ALARM` because one quiet window is not
yet evidence of a dead feed. Persisting past `silence_after_ns` promotes it. `MonitorReport.passed`
additionally requires that *some* window compared *some* rows — a monitor that ran for an hour
and compared nothing has not proved a session correct, it has proved it was not looking.

**Silence is measured on an injected wall clock, and it is the only thing here that may be.**
Everything else orders on `ts_event`. But the absence of an event has no event time, and an
event-time deadline for "nothing has arrived" can never expire: the clock that would advance it
*is* the thing that stopped. Injecting it is also what keeps every test deterministic and
sleep-free.

**5. Feature divergence alarms; drift is capped at a warning, and that cap is measured.**

There is no acceptable *rate* of feature mismatch — `atol`/`rtol` already absorb the only
legitimate difference between the two paths — so one cell past tolerance is an `ALARM`. Drift is
the opposite, and its default ceiling is `WARN` for a reason that came out of running it:

> Over `perp_bar`'s own real history — 4,975 hourly BTC rows, in 256-row windows against the
> training sample, with **feature parity green on every window** — PSI passes the conventional
> 0.25 band on **18 of 20 windows** and peaks at **5.97** on `vol_24`. ETH: 14 of 20, peak 7.93.

Nothing is wrong. A 256-bar window of realized volatility simply does not look like a 3,000-bar
one. ADR-0016 §5 already records the bands as "the industry convention, not something derived
from these features"; this is the bill for that, in numbers. Drift at `ALARM` would fire on
nearly every window forever, and an operator taught to ignore `ALARM` by the drift line has been
taught to ignore the parity line — which has had no false positive at all. `drift_ceiling` is a
per-strategy knob to raise once its bands are calibrated on its own history.

A window under `min_drift_rows` reports **"drift not measured"** rather than a PSI. "Not
measured" and "stable" are opposite statements, and only one of them is true of a 20-row window.

**6. An alarm logs, and the reason it does nothing else is written down.**

The default `AlarmSink` writes one log line per non-OK window, carrying the whole verdict rather
than the level. Two arguments against anything stronger, and neither is squeamishness:

- **Two authorities that can independently stop trading, neither aware of the other, is the
  shape [ADR-0013](0013-runtime-supervision-and-safety-loop.md)'s "loop that must die last" was
  written to avoid.** The venue-side dead-man's switch and the runtime's intent source already
  own that decision, on the Rust side, synchronously. A Python monitor reaching for it would be
  a second, uncoordinated one.
- **This detector has never been run against a live session**, so its false-positive rate is
  unmeasured — and §5 above is direct evidence that one of its two halves has a large one on
  data that is entirely correct. Wiring an unmeasured detector to a kill switch converts every
  quirk of the feed into an outage the monitor caused.

The seam is a constructor argument, so escalation is a one-line change once there is a measured
rate to justify it, rather than a rewrite.

**7. Nothing in this package opens a connection.**

The monitor consumes `Window` objects, so the same code runs over a recording, a fixture
history, or a live feed a caller pumps into it. `run_monitor` is the offline driver; a live
driver is the same loop with a source that blocks plus a `heartbeat()` on its idle branch, and
it belongs to whatever owns the connection. `python -m axon.parity --perp-bar BTC` runs
the whole thing over the frozen candle fixtures through the real `PerpBar` serving path, and
`--blind-every N` withholds every *n*-th online row so an operator can watch it fail on demand:
a detector nobody has seen fire is a decoration, which is the argument ADR-0016 already makes
for its own leaky-feature test.

## The one seam this ADR specifies and does not build

Under [`MdWritePolicy::OnChange`](0012-market-data-ring-and-multi-record-contract.md) a slice is
written only when the state the record carries actually moved. That is correct, and it makes an
empty ring **ambiguous by design**: a quiet top of book and a dead publisher produce the same
nothing. The monitor therefore resolves silence with a timer, which is a guess with a deadline
on it rather than a fact. This is the same ambiguity `axon.live.liveness` exists to remove in
the Python → Rust direction; nothing removes it in the Rust → Python one.

The fix is a **market-data beacon**: a 64-byte memory-mapped sidecar beside the ring, written by
the core's pass loop rather than by the publisher's event handler — the whole point is that it
advances when *nothing* arrives, so it cannot hang off `on_event`.

- It belongs in **`axon-ipc`**, not `axon-runtime`. `#![deny(unsafe_code)]` holds everywhere but
  the two documented exceptions, and mapping a file *is* the unsafe `axon-ipc` already holds.
  Putting it in the runtime would be the third exception, which is an ADR rather than a
  judgement call.
- It is deliberately **not** a `contracts/schema.toml` change, for the reason
  `axon.live.liveness` already states about its own beacon: it is a second shared-memory object
  on the boundary, and folding it into the ring's bytes puts it where a future schema bump
  collides with it.
- Everything it needs to carry already exists: `MdStats` counts `published`, `dropped`,
  `coalesced`, `stale_quote` and the bar-ring equivalents. The beacon publishes those plus the
  core's event-time high-water mark, with the beat counter written **last** — the same publish
  discipline the ring itself uses.
- Off by default, like the ring and capture, and for the same first reason: this end creates and
  truncates a file. Validation must refuse a path equal to either ring's.

With it, `Level.SILENT` resolves into a fact instead of a deadline: the publisher's beat is
advancing and `coalesced` is climbing (quiet market, `WARN`), or it is not advancing at all
(dead publisher, `ALARM` immediately, with no 60-second wait). Without it the monitor still
works and still refuses to report `OK` — it just cannot say *which* kind of nothing it is
looking at.

## Consequences

- **+** The feature-parity green is now **earned rather than assumed**. `perp_bar` still reports
  `max_abs_diff` exactly 0 over 4,975 rows × 9 columns per coin — and now with
  `coverage=4975/4975` asserted beside it, where before the denominator was invisible.
- **+** The one-millisecond stamp disagreement diagnoses itself instead of presenting as an empty
  matrix. So does the general constant-offset case, which is the same bug with a different
  number.
- **+** Every existing call site of `align_by_event_time` is unchanged: the return value is still
  the pair of index arrays. The accounting rides along on it.
- **+** `docs/03`'s fourth feature-parity item exists, is tested, and has been run against the
  real serving path — 4,975 rows per coin, green, and red on demand.
- **−** **Nothing here has been run against a live session.** The monitor has seen a frozen
  fixture, a cached mainnet history, and synthetic windows. Its silence deadline (60 s) is a
  starting value, not a measurement, and its drift ceiling exists *because* the one measurement
  that was taken found a false-positive rate of ~90%.
- **+** `axon.strategies.training.feature_parity_gate` has since collapsed onto
  `aligned_feature_parity` (ADR-0029), deleting `ServedFeatureParityReport` and its private row
  count. There is now exactly one implementation of "how many rows were owed", and it is
  `Coverage`. That collapse is also what exposed §1a: with the reference trimmed to the owed
  rows, the inferred-span rule was excusing a late start that the deleted private count would
  have failed on — a smaller blind spot, but the wrong direction of travel, and closed rather
  than papered over.
- **−** The monitor cannot tell a quiet publisher from a dead one on its own. Under
  `MdWritePolicy::OnChange` a slice is written only when the state moved, so an empty ring is
  ambiguous by design and the monitor can only resolve it with a timer.
  **Amended 2026-07-26:** the Rust→Python market-data beacon specified above is **built**
  ([0034](0034-market-data-beacon-and-the-third-clock.md)) and `ParityMonitor` consumes it through
  a `beacon` probe — but it is **not wired into the runtime's pass loop**, because that agent owned
  `axon-ipc` and not the runtime. So in the tree as it stands the deadline is still the only
  evidence a live session offers. 0034 also *corrects* three details of the specification above:
  the counters do not fit at 8 bytes each, so five are `u32` and **wrap**; the path-collision rule
  is replaced by one that is unrepresentable rather than validated; and the two publisher states
  are seven, two of which are faults this ADR had no name for.
- **−** Three separate closures of one bug class in a single increment (§1b) is not a sign the
  design was right the first time. Each was found by the next component built on top rather than
  by the one below it, which is the pattern to expect from anything else that consumes a
  `Coverage`: the aligner cannot know which rows were owed, so every new caller has to be asked.
  `scope` is the place to ask, and a fourth level would extend it rather than reinvent it.
- **−** `Alignment` is a `tuple` subclass, which is an unusual shape. It was chosen so that
  adding the accounting could not break a caller that had not heard of it; the cost is that
  `Alignment` is not a dataclass and does not print like one.
- **−** The drift half is, for now, a warning light with a known false-positive rate rather than
  a gate. Making it a gate needs per-feature bands derived from each strategy's own history,
  which is a research task and not a monitoring one.

See [ADR-0016](0016-feature-spec-and-parity-gates.md) (the three gates this runs, and §4 whose
alignment decision this corrects), [ADR-0013](0013-runtime-supervision-and-safety-loop.md) (why
the alarm does not reach for the kill switch), [ADR-0012](0012-market-data-ring-and-multi-record-contract.md)
(the ring whose `on_change` policy makes silence ambiguous), and
[ADR-0022](0022-first-ml-strategy.md) (the strategy whose real history both the green and the
drift measurement above come from).
