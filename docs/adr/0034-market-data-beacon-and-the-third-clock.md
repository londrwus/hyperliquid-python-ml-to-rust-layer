# ADR-0034 — The market-data beacon: a third clock, and what 64 bytes could not hold

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) ends with a section
titled *"the one seam this ADR specifies and does not build"*. It names the ambiguity
precisely — under [`MdWritePolicy::OnChange`](0012-market-data-ring-and-multi-record-contract.md)
a slice is written only when the state it carries actually moved, so a flat top of book and
a dead publisher write the same empty ring — and it specifies the fix: a 64-byte
memory-mapped sidecar in `axon-ipc`, written by the core's **pass loop** rather than by its
event handler, carrying `MdStats`' counters plus the core's event-time high-water mark, with
the beat counter written last.

That specification is right and this ADR does not revisit it. What it records is the set of
questions that only appear once someone tries to lay the thing out, each of which has a
comfortable answer that ADR-0030 could not have known was wrong.

1. **Which clock does a liveness object read?** The house rule is event time everywhere and
   ADR-0030 asks only for the core's event-time high-water mark. That is the obvious answer
   and it is insufficient in a way the rule itself predicts.
2. **Do the counters fit?** ADR-0030 lists `published`, `dropped`, `coalesced`, `stale_quote`
   "and the bar-ring equivalents" plus a stamp. At eight bytes each that is more than 64
   bytes before the header, and nobody had added it up.
3. **What may a reader conclude from a read that races a write?** "Write the beat last" is
   the discipline; the reader's side of it is a separate claim, and the obvious one — *a
   snapshot is stale by at most one beat* — is false.
4. **How is a path collision with either ring prevented?** ADR-0030 says "validation must
   refuse a path equal to either ring's", which is a rule somebody has to remember to run.
5. **What does the monitor do with the answer?** ADR-0030 gives two cases: beat advancing
   with `coalesced` climbing is a quiet market (`WARN`), beat not advancing is a dead
   publisher (`ALARM`, immediately). Two is not enough, and one of the missing cases is a
   fault this project has already been bitten by.

## Decision

### 1. Both clocks are on the record, and the wall clock is a third named exception

`last_event_ns` is **event time** — the core's own `ts_event` high-water mark, the clock
every ordering decision in the system is made on. It is the field that says whether the
*market* is moving.

`last_beat_ns` is **wall clock (`CLOCK_REALTIME`)**, and it joins the dead-man's-switch
deadline (wall clock because the *venue* holds it) and the reconnect backoff (wall clock
because it is not ordering) as a named exception. The justification is the same shape as
both, and it is not a convenience:

> **The condition being detected is the absence of events, and the absence of an event has
> no event time.** An event-time-only beacon freezes at exactly the moment it is needed,
> because the clock that would advance it *is* the thing that stopped.

ADR-0030 §4 already makes that argument one process later, for the monitor's own silence
deadline. The beacon moves the measurement to the side that actually knows whether it is
alive, and the argument travels with it unchanged. What keeps it bounded is that **nothing
orders, ages or admits anything on `last_beat_ns`**: it is read only as a difference, and it
exists so that a *single* read can say how long ago the publisher last ran instead of
requiring two reads spaced by a poll interval.

`last_beat_ns == 0` is the sentinel for *this session had no wall clock at all*. An offline
session's core loop reads none (`CoreControl::wall_time` is false, deliberately, so a replay
does not depend on how fast the machine drained the bus), and a reader that treated the zero
as a timestamp would report every offline session as 56 years dead. `MdBeaconSnapshot.had_wall_clock`
is the property that says so, and an offline `axon` session is what exercises it.

The beat itself does not read a clock. `MdPublisher::beat` takes `wall_ns` as an argument,
because the core loop already takes exactly one wall-clock reading per iteration and shares
it with the mark cache's liveness clock; a second reading would be a second answer to the
same question.

### 2. Five counters are 32 bits, they wrap, and `stale_quote` beat `bars_dropped` on evidence

The arithmetic ADR-0030 did not do: `magic` (8) + `version` (4) + `pid` (4) + `beats` (8) +
`last_event_ns` (8) + `last_beat_ns` (8) + `flags` (4) is 44 bytes before a single counter.
Twenty bytes remain. So the layout is:

```text
 0  u64  magic          = "AXONMDBN"
 8  u32  version        = 1
12  u32  pid
16  u64  beats          monotonic; written last, with Release
24  i64  last_event_ns  EVENT time high-water mark  (0 = nothing seen yet)
32  u64  last_beat_ns   WALL clock at this beat     (0 = no wall clock in this session)
40  u32  published      44  u32  coalesced   48  u32  dropped
52  u32  bars_published 56  u32  stale_quote 60  u32  flags
```

Three things there are decisions.

**The first 40 bytes and the last 4 are `axon.live.liveness`'s layout, field for field.**
The two beacons are mirror images of each other across the same boundary — one says the
Python strategy is alive, one says the Rust publisher is — and an operator who has learned
one should not have to learn the other. Only `[40, 60)`, the direction-specific payload,
differs. The magic differs too, and it is doing real work: both files are 64 bytes with an
identical prefix, so nothing about a file's *shape* can tell them apart, and a reader pointed
at the wrong one would decode `signals` as `published` — a plausible number, silently wrong,
forever. This is ADR-0012 §3's argument for `record_kind` over `record_size` arriving in a
second place, and it has a test in each language.

**The counters truncate rather than saturate.** A wrapped counter still yields an exact
`wrapping_sub` delta for any two readings fewer than 2³² increments apart — five days at ten
thousand slices a second, against a monitor that polls per window. A *saturated* counter
yields a delta of zero forever once pinned, so a busy publisher would read as a quiet one,
which is the wrong answer to the only question this file is asked. The price is that the
absolute value stops meaning anything after a wrap, so **no reader may print it**; the totals
live on `MdStats` and the status line, which is the side with room for them.

**`stale_quote` earns the last four bytes over `bars_dropped`, and the reason is measured.**
A reconnect that failed to restore a `bbo` subscription put 45.8% of a 1 h 44 m soak under
stale marks with every other counter healthy — the *alive but broken* state a liveness object
should be able to name. A full bar ring, by contrast, needs nobody to have read one bar a
minute for hours, and the slice ring's `dropped` already says "the consumer fell behind".
`bars_forming`, `unrepresentable` and `bars_out_of_order` did not fit and are not liveness
questions.

### 3. A skewed read is safe by *direction*, and the "at most one beat" version of that claim is false

Every field is naturally aligned inside one cache line at offset 0 of a page, so on x86_64 —
the platform ADR-0006 already assumes for this boundary — no field can be torn. What remains
is cross-field skew, and the first draft of this ADR claimed a snapshot was stale by at most
one beat. **The test written to demonstrate that failed immediately**, with `beats 14` beside
`published 16`: a reader's ten loads are not one instruction, and against a pass loop
spinning at kilohertz the payload runs as far ahead as the reader is slow.

The claim that survives is about *direction*, and it is the one that matters:

- Every payload field is monotonically non-decreasing, so any mixture of beats is a state
  the publisher genuinely passed through. Nothing goes backwards between two reads.
- `beats` is stored **last with `Release`** and loaded **first with `Acquire`**, so a payload
  field is always *at least* as new as the beat count reported beside it, never older.
  Therefore **the reading that would turn a live publisher into a dead-looking one — the
  count moving while the counters appear frozen — cannot be produced by skew at all.** It can
  only be produced by a publisher that really did stop between its payload stores and its
  beat store and then stayed stopped, which is a publisher that is dead.
- The consequence to hold onto: `published - beats` is not a quantity and nothing computes
  one. Each counter is compared only against its own previous reading.

`a_read_that_races_a_beat_never_shows_the_count_ahead_of_the_payload` runs a reader against
200 000 beats and asserts exactly that; moving the beat store ahead of the payload reddens it.

A **seqlock was rejected**, and not for cost. It would make `beats` odd while a write is in
flight, so the field an operator watches would stop being a beat count; and it would put a
retry loop in the reader that can spin forever against a writer that died mid-write — trading
a staleness that is harmless by direction for an unbounded stall in the exact scenario the
beacon exists to detect.

Every access on both Rust sides is an atomic (`Relaxed` payload, `Release`/`Acquire` beat),
so there is no data race in the abstract machine either. On x86_64 a relaxed atomic store is
the same `mov`, so the hygiene is free. The Python reader instead copies all 64 bytes once
and parses the copy, which narrows its window to a memcpy; it is allowed to, because it is
not obliged to be a data-race-free Rust program.

### 4. The path is derived, not validated — and `create` must not truncate

`beacon_path` **appends** `.beacon` to the slice ring's path rather than substituting the
extension, and that is load-bearing rather than stylistic: a string cannot equal itself plus
a suffix, so the beacon can never name either ring's file however the ring's path was spelled
— including the adversarial `path = ".../axon-md.beacon"`, which is exactly what an
extension-substituting derivation would collide on. ADR-0030 asks for validation refusing a
collision; **this makes the collision unrepresentable, which is the stronger form, because
there is no check anybody can forget to call.** It is also ADR-0028 §5's argument for
deriving the bar ring's path reapplied: one switch cannot be half-turned, and an operator who
enabled slices and forgot the beacon would get a monitor that silently went back to guessing.

Consequently the beacon is **not independently configurable and not independently
switchable**. It exists exactly when `[md_ring] enabled` is true, and no config field was
added at all.

Less obvious, and found by writing the restart test: **`MdBeacon::create` reinitialises the
file, it does not `O_TRUNC` it.** Truncation takes the file to zero length before `set_len`
puts it back, and a reader with the page mapped takes a `SIGBUS` — a signal, not an error it
can handle — for touching a page past EOF during that window. The failure would be a
publisher killing its own monitor on restart, at the moment the monitor was most needed.
Every field is overwritten anyway, so nothing of the previous session survives; the length
simply never dips. This half is **argued rather than proven** — a sequential test cannot be
inside that instant, and swapping `create` back to `.truncate(true)` leaves the test green.
The test guards the observable consequence (the beat resets, the file is a full beacon
afterwards, a reader mapped before the restart still works) and says in its own comment which
half it does not cover.

### 5. Seven publisher states, because two of them are faults ADR-0030 did not have a name for

ADR-0030 gives the monitor two readings. Building it produced five more, and the two that
matter are not padding:

| state | what two readings showed | what the monitor does |
|---|---|---|
| `DEAD` | the beat did not move | **`ALARM`, immediately** — no deadline left to resolve |
| `STOPPED` | the beat did not move and the last one said so on purpose | `ALARM`, different words |
| `QUIET` | beating, event clock advancing, `coalesced` climbing, nothing published | **`WARN`, and the deadline is suspended** |
| `STARVED` | beating, and the **event clock has not moved at all** | stays `SILENT`; the deadline still runs, and the alarm it eventually raises names the cause |
| `PUBLISHING` | beating and writing records | stays `SILENT`; the deadline still runs |
| `RESTARTED` | the pid changed or the beat went backwards | `SILENT`; the baseline is void, not the publisher |
| `UNKNOWN` | no probe, no file yet, one reading only, or the read raised | exactly the pre-beacon behaviour |

**`STARVED` is a fault nothing else in this system can see.** The handoff records that a
Binance stream-name typo is accepted silently — no error frame, no close, socket healthy
forever — so a session can hold a permanently healthy, permanently silent connection that no
backoff and no duration check will ever notice. A live process whose pass loop is running and
whose event clock has not advanced is precisely that, and it is the *opposite* fault from a
quiet market while producing an identical empty ring. It keeps the deadline rather than
alarming at once because a genuinely quiet instrument can go minutes without an event; what
the beacon buys is that the alarm, when it comes, says *nothing is reaching the core* instead
of guessing "dead feed".

**`PUBLISHING` is the anti-excuse, and it is there deliberately.** The easy thing to write is
"the publisher is healthy, so the silence is fine". Market data crossed the boundary and the
comparison still ran over zero rows — that is the invisible denominator of ADR-0030 §1b
wearing a beacon, and the honest verdict names the consumer instead of the publisher.

Two rules hold the whole thing together. **The beacon may lower `SILENT` to `WARN` and may
never lower an `ALARM`** — a declared-scope window that owed rows and compared none is a
blind serving path whatever the publisher is doing, because the beacon says nothing about the
*consumer*. And **suspending the deadline for `QUIET` is the point, not a leniency**: a market
that does not move for an hour is not a fault, and an operator taught to ignore `ALARM` by a
quiet hour has been taught to ignore the parity line — which is ADR-0030 §5's `drift_ceiling`
argument, one floor down, reached for the same reason.

Everything degrades to `UNKNOWN`, including a probe that *raises*. A live monitor must alarm
rather than die (ADR-0030 §6), so an unreadable beacon becomes a sentence in the report and
the timer goes back to being the only evidence. It is deliberately not an alarm of its own:
escalating on a file that is not there yet would fire on every startup race, and the monitor
is wired before the session it watches begins.

The Python reader draws one more distinction for the same reason. A magic of **zero** is the
window between `ftruncate` and the header store, or a path nobody has written — a race, so it
reports "not ready". Any **other** wrong magic is a beacon of the wrong kind, which is a
wiring mistake that will not fix itself, so it raises.

## Consequences

- **+** ADR-0030's `Level.SILENT` resolves into a fact. A quiet market no longer escalates at
  all, and a dead publisher alarms with no sixty-second wait — both proven by tests named
  after the readings they prevent.
- **+** Two failure modes that had no detector now have one: a live process receiving nothing
  (`STARVED`), and a healthy publisher whose consumer compared nothing anyway (`PUBLISHING`).
- **+** `MdSlice` was not touched. It has had zero reserved bytes since
  [ADR-0028](0028-market-data-bars-and-the-ticker-tail.md), and a sidecar is what keeps the
  next thing that needs boundary bytes from spending a stride change on liveness.
- **+** No new `unsafe` crate. Mapping a file is the exception `axon-ipc` already holds, and
  `#![deny(unsafe_code)]` still stands in every crate that can hold it — including
  `axon-runtime`, which was where the obvious place to put a beacon would have been.
- **+** No new config field. One switch (`[md_ring] enabled`), three derived paths, and a
  startup banner that names all three.
- **− The beacon is not wired into the pass loop in the tree.** It is built, unit-tested in
  both languages, and cross-language round-tripped through `md_writer`; the one line in
  `axon_runtime::core::run` that drives it belongs to a crate this workstream did not own.
  The wiring *was* applied in a scratch copy of the tree, built clean, and produced a beacon
  a real `axon` session wrote and the Python reader parsed — so the diff is verified rather
  than proposed, but the tree does not contain it.
- **− Nothing here has run against a live venue session.** Every beacon read so far came from
  an offline session or the fixture writer, so `last_beat_ns` has never carried a real wall
  clock in anger and no reader has yet watched a beat advance over seconds of real time. The
  first live shadow session is what turns "the arithmetic says the beat outran the ring" into
  an observation.
- **− The counters are `u32` and their absolute values are meaningless after a wrap.** A
  future reader that prints one will be wrong roughly five days into a busy session, and
  nothing in the type system stops it. The docstrings say so in both languages; that is the
  whole of the guard.
- **− A publisher that unlinked and recreated its beacon rather than reinitialising it would
  leave a mapped reader on the dead inode, reading a file nobody writes.** `MdBeacon::create`
  does not do that, and there is no check that it never will.
- **− The reader's staleness is bounded by the reader's own speed, not by one beat.** Harmless
  by direction (§3), but it means the *deltas* a slow reader sees are attributed to a window
  they may slightly overlap. Nothing here counts anything precisely enough for that to matter,
  and if something ever does, it needs a snapshot primitive rather than a wider field.
- **− ADR-0030's consequence "Closing it needs a Rust→Python market-data beacon, specified
  above and **not built**" is now false on the first clause and still true on the last.** It
  should be amended to point here.

See [ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) (the monitor this
serves, and the section that specified this file),
[ADR-0012](0012-market-data-ring-and-multi-record-contract.md) (the ring whose `OnChange`
policy makes silence ambiguous, and §3's argument for a kind tag that this reuses for a
magic), [ADR-0028](0028-market-data-bars-and-the-ticker-tail.md) (why `MdSlice` has no bytes
left, and the derived-path argument), [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the
x86_64 assumption and the Release/Acquire discipline),
[ADR-0013](0013-runtime-supervision-and-safety-loop.md) (why a monitor that now *knows* the
publisher is dead still only logs), and `python/axon/live/liveness.py` (the beacon this one
is the mirror of).
