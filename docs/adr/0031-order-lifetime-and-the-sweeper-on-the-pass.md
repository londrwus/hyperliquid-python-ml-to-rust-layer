# ADR-0031 — An order lifetime, and the half of it a signal cannot enforce

**Status:** Accepted · **Date:** 2026-07-26

## Context

Until this decision the 64-byte `Signal` carried exactly one duration, `ttl_ms`, and everybody who
read the record assumed it was an order lifetime. It is not, and never was.
[ADR-0014](0014-signal-to-order-planning.md) §1 defines it precisely: `SignalReader` enforces
`min(ttl_ms, ReaderConfig::max_age_ms)` and **refuses the record** past that age, before
`Planner::plan` is ever called. It is a *signal admission* window. The operator's ceiling on it —
`intent.max_signal_age_ms` — is **2 000 ms**, so `perp_bar`'s `ttl_ms = 60_000` is clamped to two
seconds and buys an order already resting at the venue exactly nothing.

The confusion is not hypothetical. The runbook has had to carry the correction as a standing
"facts a new session will otherwise get wrong" bullet, `IntentConfig` has had to carry it as a doc
comment, and `PlannerConfig` calls it *"the standing mistake here"*. A field that three documents
have to warn about is a field the design is missing, not a field the readers are getting wrong.

Two changes then made the gap load-bearing rather than merely untidy. ADR-0014 §6's
leave-it-resting exception — *an order identical to the one we would place is left alone, because
replacing it forfeits queue position for nothing* — was **widened twice**:

- `TrackedOrder` gained a real `tif` and `reduce_only`, so an **adopted** order can now take the
  exception. [ADR-0020](0020-runtime-intent-source.md) §7 had explicitly relied on it never being
  able to. An order inherited across a restart — one nobody in this process ever decided to
  place — could therefore rest for as long as the strategy kept asking for the same target.
- `PlannerConfig::noop_band_bps` forgives a size difference, so "identical" became "close enough".

An exception with no bound on it preserves orders nobody currently intends. So the field, and the
planner's bound on it, landed on 2026-07-26. **This ADR is the record of why, and of the half that
did not land with it.**

Seven questions had to be answered, and each has a comfortable wrong answer.

1. **Does an order lifetime belong on the wire at all,** or is it purely the operator's?
2. **What does zero mean** — on the wire, in config, and when the two disagree?
3. **Who enforces it?** The planner is the obvious answer. It is also only half an answer, and the
   half it is missing is the one that matters: **the planner runs on a signal.** A strategy that
   has stopped emitting — crashed, stalled, still warming up, disconnected, restarted with a
   rewound `seq` — leaves its last quote resting at the venue, and the mechanism designed to bound
   that order's age never runs again.
4. **Which clock does a sweeper use?** Event time is the house rule. But a *silent* strategy is
   precisely the case where somebody will argue a clock has stopped.
5. **What about an order the session adopted rather than placed?** There is no `cloid` we minted.
6. **Does having a sweeper make the dead-man's switch less necessary?**
7. **What does the sweeper do when it cannot reach the venue?** Escalation, or silence?

## Decision

### 1. `max_order_age_ms` is a second duration on the wire, and it is not `ttl_ms`

The record's `schema_version` goes **1 → 2**: `max_order_age_ms` (u32, offset 52) is carved out of
`reserved`, with an explicit `pad0: [u8; 3]` at 49 so the u32 lands naturally aligned. The bump is
not optional — a v1 reader refuses non-zero reserved bytes (ADR-0014 §1), so it would reject a v2
producer's records wholesale, which is the drift-detection working rather than a problem to route
around.

|  | `ttl_ms` | `max_order_age_ms` |
|---|---|---|
| what it bounds | how old a **record** may be when we act on it | how long an **order** may keep its place at the venue |
| who consumes it | `SignalReader`, before the planner sees the record | `Planner`, and the sweeper (§4) |
| operator's ceiling | `intent.max_signal_age_ms`, **2 000 ms** | `intent.max_order_age_ms`, **60 000 ms** |
| what `0` means | the operator's ceiling | the operator's ceiling |
| what a large value buys a resting order | **nothing** | its lifetime |

It goes on the wire rather than staying purely operator config for the reason the urgency table
went on the wire: the strategy is the only party that knows what its quote is *for*. A
market-making strategy re-quoting every 200 ms and a slow allocator re-weighting hourly want
different answers, and the operator's single number cannot be right for both. What the wire field
may **not** do is lengthen anything — see §2.

It is deliberately not an argument to `Signal::target_position`. A tenth positional `u32` beside
`ttl_ms` and `model_version` is a transposition waiting to happen between two fields that are both
durations, both `u32`, both in milliseconds, and mean completely different things. It is a chained
`with_max_order_age_ms`, and a keyword-only argument in Python.

### 2. Zero means "the operator's ceiling" on both sides, and the binding lifetime is the shortest anyone actually expressed

This is [ADR-0020](0020-runtime-intent-source.md) §4's argument about `ttl_ms == 0`, transferred
without a change, because the premise transfers without a change: **zero is the value of a field
nobody wrote.** Refusing to emit it never stopped it arriving, so the consumer has to define it,
and the definition has to be the safe answer for a producer that never thought about the question.
*"Never expires"* gives that producer no protection at all; *"already expired"* silently stops a
strategy trading, which is indistinguishable from a quiet market.

The implementation matters more here than it did for `ttl_ms`, and this is the case that inverts if
it is written carelessly. The tempting one-liner is `requested.min(ceiling)` — and `min(0, 60_000)`
is `0`, so a strategy that never set the field (every strategy that exists today, and the Python
default) would have every order it placed cancelled on the very next pass. `Planner::order_lifetime_ns`
therefore **filters the zeros out before the comparison**, which makes the mistake unrepresentable
rather than merely avoided:

```rust
[requested_ms, self.cfg.max_order_age_ms].into_iter().filter(|ms| *ms != 0).min()
```

What is left is one sentence: **the binding lifetime is the shortest one anyone actually
expressed.** A strategy can ask for a shorter-lived quote and never a longer one — the same
asymmetry `ttl_ms` has, and for the same reason: the operator answers for the fills. And an
operator who set no ceiling has not thereby overridden a strategy that asked for a short-lived
quote; `0` there means *"I set no bound"*, not *"no bound may be set"*.

One migration consequence follows and it is worth stating, because it looks like a gap and is not.
**A v1 signal log is not upcastable.** `axon-replay` can carry a v1 *event* log forward; it refuses
a v1 signal log, because the value a naive upcast would write is zero, and zero is not "absent" —
it is *defer to the operator's ceiling*. An upcast log would replay with every order carrying a
lifetime the live session never gave it, and orders pulled on age where the session left them
resting. That is a changed decision wearing a migration's clothes, and there is no honest number to
write instead.

### 3. The planner owns half of this, and it can only ever *deny an exception*

`Planner::outlived` is consulted at exactly one place: the leave-it-resting exception. Past the
binding lifetime, the exception does not apply and the order is cancelled and replaced by whatever
the current signal decides. The planner never emits a cancel *on age alone*, and it does not need
to — every no-order path already emits the cancels for the symbol, including "already at target"
and "no usable quote" (ADR-0014 §6). So an over-age order goes on the next signal whatever that
signal says.

That is the honest boundary of what a pure function of `(signal, context)` can own. **It runs on a
signal.** The bound it enforces is therefore conditional on the producer still speaking, which is
the one condition under which a resting order is *least* likely to be a problem.

### 4. The other half is a sweeper on the pass loop, so it advances when nothing arrives

`axon_runtime::intent::sweep_overage_orders` runs inside `IntentSource::poll` — the same pass that
drains the ring — and cancels every open order that has outlived `intent.max_order_age_ms` with no
signal to supersede it. The whole point is that the pass is driven by the *core loop*, not by a
record arriving, so it keeps running when the producer has stopped. Six properties are decisions:

- **The trigger is a property of tracker state, not an event.** Level-triggered, not edge-triggered:
  an order the venue did not remove is still over-age on the next sweep, so a lost cancel retries
  itself with no separate retry queue to go wrong.
- **It enforces the operator's ceiling and nothing else.** The signal's own shorter request is not
  cached and re-applied, for two reasons. A cached request is a claim nobody is renewing — the
  producer that made it has stopped, which is the entire premise — and an *adopted* order has no
  signal behind it at all, so a per-order rule would need two rules. Reaching for the shorter number
  would also mean a second implementation of `Planner::order_lifetime_ns` in a crate that cannot
  call it, and two implementations of one rule is how the two come to disagree.
- **A separate, slower cadence** (`intent.sweep_interval_ms`, default 1 000 ms of event time). It is
  the only thing in the pass that reads order state when *nothing arrived*, so at the drain cadence
  it would take the tracker's read lock and walk the open orders once per millisecond of event time,
  on the deterministic core thread, to answer a question about a sixty-second bound.
- **It runs after the planning loop, and skips any symbol a signal just spoke for** or that has an
  intent still at the venue. Redundant in the first case (the planner's cancels are already in an
  intent of its own, and its bound is `min(signal, operator) ≤ operator`, so an order it chose to
  leave resting is inside the ceiling by construction) and *harmful* in the second: the in-flight
  claim is a per-symbol bit, so two claims against one release would gate that symbol for the rest
  of the session over an intent nobody is waiting on.
- **One intent per symbol, cancels sorted by `(symbol, cloid)`**, for the reason `project_working`
  is sorted: the tracker holds orders in a `HashMap` whose iteration order varies between runs, and
  an unsorted sweep would issue a different sequence of venue actions on each replay of one session.
- **`seq: 0` on a swept intent.** `SignalReader` counts from 1, so zero is not a sequence any record
  can carry: a venue action traced back to `seq 0` says *"the sweeper, not the strategy"* without a
  second field to carry it.

Turning the ceiling off (`intent.max_order_age_ms = 0`) turns the sweeper off, because it is the
same number and the same reading of zero. There is deliberately **no second switch** — two knobs
that can disagree about whether an order has a lifetime is exactly the shape this ADR exists to
remove.

### 5. The clock is event time, and the exception was considered and refused

The house rule is event time everywhere, and its real exceptions are named and few: a dead-man's
switch deadline is wall-clock because the *venue* holds it (ADR-0013 §3), a reconnect backoff is
wall-clock because it is not ordering, `MarkCache`'s liveness clock is wall-clock because a dead
feed emits no events and a frozen event clock calls every stale price fresh (ADR-0013 §2). That
last one is the tempting precedent here, and the argument is exactly the same shape: *a silent
strategy is the case where a clock may not be advancing, so measure it on a clock that always is.*

It does not survive three checks.

**The arithmetic.** `TrackedOrder::placed_ts` is an event-time stamp. A wall-clock `now` measured
against it is a subtraction between two different clocks, and a replay of yesterday's capture would
find every order a day old and sweep the lot. Using a wall clock would require a *second*,
wall-clock placement stamp — a second description of one fact, which is the shape
[ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) §1a rejected for the same
reason: two descriptions can disagree, and here they would disagree exactly when the machine's
clock and the venue's had drifted.

**The facts.** The clock a silent *strategy* stops is the signal ring's. The core's event clock is
fed by market data, from an entirely different producer, over a different transport. A strategy
that crashed, stalled, or is warming up does not touch it. So event time is not merely acceptable
here — it is *precisely* the clock that keeps advancing in the case the sweeper exists for.

**The structure.** What does stop event time is a dead *market-data* feed, and no clock choice here
could rescue that anyway: `IntentSource::poll`'s own schedule is on event time (ADR-0020 §2), so a
frozen event clock runs no pass for a sweeper to sit on. Putting a wall clock in the *measurement*
while the *schedule* stayed on the event clock would buy nothing and forfeit replay determinism for
every cancel the sweeper ever issues. Putting it in the schedule too would forfeit it for every
pass, which is the property the entire seam was arranged for.

A dead feed is a different failure with its own detectors — `STALE MARKS`, the risk gate refusing
every risk-increasing order, and the switch at the venue. It is named here rather than covered
here, and §7 says what does cover it.

### 6. An adopted order is cancelled by the venue's own id, and the sweeper needs that more than the planner does

The sweeper builds `CancelId::Cloid` and then runs the same `retarget_cancels` a planned cancel
gets: anything the tracker marks `adopted` is re-addressed to `CancelId::OrderId`, because an
adopted order's `cloid` may be one the *tracker* synthesized from a venue order id to key it into
its own map, and the venue has never seen it. A cancel sent under an id the venue does not
recognize fails, and the stale quote it was supposed to remove stays resting exactly where somebody
else's taker is looking for it (ADR-0020 §6).

This matters more here than on the planned path, and the reason is a fact about which orders a
sweep finds. **An order that outlives its producer is disproportionately one a previous incarnation
of this process left behind** — and this session minted none of their ids. There is a pleasing
corollary: `openOrders` carries the venue's own placement timestamp, so an order resting since
yesterday is adopted with a `placed_ts` from yesterday and is over-age on the **first sweep after a
restart**, before any signal has arrived. The sweeper is therefore also the thing that finally
clears an inheritance the leave-it-resting exception had been quietly preserving.

The residual case is stated rather than solved: an adopted order the venue reported with **no**
order id at all is cancelled under a synthesized `cloid` and that cancel will fail. It is then
re-asked on the schedule in §7. Nothing in the tracker can do better, because there is no id.

### 7. The sweeper is the client-side backstop; the dead-man's switch is the venue-side one, and neither excuses the other

They cover different failures and the overlap is small:

| | dead-man's switch | the sweeper |
|---|---|---|
| held by | **the venue** — `scheduleCancel` fires on the venue's clock | this process |
| covers | our process dying, freezing, losing the network, or being killed | our *producer* dying while this process is healthy |
| granularity | **everything**, all at once | one over-age order at a time |
| cost | one of ~10 venue triggers a day, and the account is unprotected for the rest of the day after ten | one metered `/exchange` cancel |
| runs when | this process runs no code at all | this process runs and reaches the venue |

Neither is a substitute. A DMS trigger flattens the whole account and is spent from a budget of ten
a day; using it to retire one stale quote would be spending the day's protection on a routine
event. And the sweeper cannot run in the case the switch exists for, by definition: a process that
is not running runs no passes — the narrow claim ADR-0013's amendment already makes about
`on_elapsed`, and it applies verbatim here.

So **the existence of the sweeper is not an argument for weakening the switch.** Config validation
still refuses `safety.dead_mans_switch = false` with `intent.enabled = true` on a live wiring
(ADR-0020 §9), and that stays refused. The sweeper makes the *routine* case cheap; it does nothing
about the case where there is nobody left to be routine.

### 8. When it cannot reach the venue: it re-asks on its own bound, counts, and does not reach for a kill switch

A level-triggered sweeper against a venue that is not answering would re-cancel the same order
every sweep interval, forever. That is not theoretical arithmetic: signed `/exchange` actions are
metered against `nRequestsUsed` (cap ≈ 10 203/day, and `/info` reads are free), so one stuck order
at a cancel a second would spend the day's budget and leave none for the orders that could still be
pulled.

**So an order is re-asked at most once per lifetime, not once per sweep.** The bound the sweeper
enforces is the bound it retries on, which is the only number in scope that already means "long
enough that this should have worked". The re-ask is counted separately as `resweeps`, and that
counter is the one piece of evidence available on this side that our cancels are not landing: a
cancel that *fails* is counted at the edge, but a cancel that succeeds and never removes the order —
a lost reply, a venue that acked and did nothing — is counted nowhere else, and from here it is the
same outcome. The exposure is still there and nobody asked it to go.

**Neither silence nor escalation, then: a count and a log line.** The sweeper does not halt the
session and does not fire the switch, for the reason ADR-0030 §6 already gives about the parity
monitor's alarm — *two authorities that can independently stop trading, neither aware of the other,
is the shape ADR-0013's "loop that must die last" was written to avoid.* The halt switch and the
dead-man's switch already own that decision, synchronously, on this side of the boundary. And this
detector has never been run against a live session, so wiring an unmeasured detector to a kill
switch converts every quirk of the venue into an outage the sweeper caused.

### 9. The sweeper's cancel is never risk-gated, and it depends on that rather than merely enjoying it

`GuardedClient::cancel` is ungated by construction ([ADR-0010](0010-execution-events-and-reconciliation.md)
§3) and `HaltableClient` passes cancels through while refusing placements, so a sweep reaches the
venue on precisely the sessions that can no longer place an order. That is not incidental — it is
the property the whole mechanism rests on, and the argument is the same one ADR-0010 makes for its
own exception to an otherwise unbypassable gate, in the case that makes it sharpest:

> **A cancel reduces exposure, so a gate that can refuse a cancel can pin an account into the
> position it is trying to leave.** Every input that would make the gate say no — no mark price,
> over the position limit, an instrument nothing can price — is a reason to want the quote *gone*.
> And the sweeper's whole subject is a session whose strategy has stopped speaking, so if the gate
> refuses, nobody is ever going to ask again.

A test pins it end to end rather than by inspection: a cancel is submitted through a `GuardedClient`
with an empty risk context, and the assertion is *both* that the cancel reached the venue **and**
that the placement beside it was refused — because a gate that lets everything through would
satisfy the first assertion on its own, and this check has to be able to fail.

## Amendment, 2026-07-26 — the sweeper has now cancelled at a venue, and the first sweep found a composition nobody had reasoned about

Two hours after this record was written, a live testnet session (`baseline_z`, m1, 1 h 35 m) swept
twice and the venue confirmed both: oid 57018922365 cancelled at **62 027 ms** and oid 57019306105
at **60 035 ms**, against `max_order_age_ms = 60000`. §4 is therefore observed rather than reasoned,
and the minus below that says "none of this has been run against a venue" is **no longer true** for
the sweep itself. It remains true for every other claim in it — a re-ask after an unreachable venue,
an escalation, an adopted order swept after a restart, and the once-per-lifetime suppression have
all still only met a spy client.

**The composition hazard, which is a real defect and is not fixed.** Both swept orders were
*exits*. A target position is idempotent, so the strategy — correctly — said nothing once its target
had not changed, and **nothing re-quoted them**: a short sat open with no working order for roughly
twelve minutes. The sweeper's rule is "cancel what no signal speaks for". The strategy's rule is
"say nothing when nothing has changed". **Each is right, and their composition is wrong**, because
the sweeper's trigger condition — a symbol no signal spoke for in this pass — is satisfied
permanently by a strategy that has already said everything it means to say. §4 assumed the silent
producer was the pathological case; a *deliberately* quiet producer holding a steady target is the
normal case, and it looks identical from inside the pass.

This is not repaired here, and the choice belongs with whoever fixes it rather than with the
observation. What must not happen is the comfortable repair: making the sweeper spare orders that
look like exits. `reduce_only` is a property of the order, not of the operator's intent, and a
sweeper that reads intent off an order flag is guessing at exactly the point this ADR spent §9
arguing it must not. The candidates worth weighing are re-quoting on the pass from the last live
target, having the sweep publish something the planner can act on, and making the strategy re-assert
rather than only emit changes — the third of which moves the problem into Python and out of a
component that cannot see whether a target is still meant.

Also learned, and owed as a separate fix: **`swept` and `resweeps` are not surfaced by any running
session.** Both counters moved in production and no operator could read them; the live figures above
are *inferred* from the venue and from `sent 6+2c`. §4 states the rule this violates — counted and
not reported is the same as not counted — and applies it to `busy` and `stalled` but not to these.

## Consequences

- **+** `ttl_ms` and an order lifetime are now two fields with two names, and the distinction is
  stated in the schema, on both language bindings, in the config docs and here. Three documents
  were carrying a warning about one field; they now describe two.
- **+** ADR-0014 §6's leave-it-resting exception is bounded in both directions. The planner denies
  it past `min(signal, operator)`; the sweeper pulls the order past `operator` even when the
  planner is never called again. Every resting order is eventually one that either a live decision
  produced or nothing at all did — and in the second case it goes.
- **+** A restarted session no longer inherits a stale quote indefinitely. Adopted orders carry the
  venue's own placement time, so anything older than the ceiling is swept on the first pass after
  market data arrives, before a single signal has been read.
- **+** A detached signal ring no longer ends the pass. "Python is not there" is the strongest
  possible statement that no signal is coming, and it was the one condition under which nothing
  downstream ran at all — so the case that most needs a sweeper was exactly the case that had none.
- **−** **None of this has been run against a venue.** The sweeper is unit-tested, offline, against
  a recording sink and a spy client. Zero cancels of any kind had been sent through the intent path
  at a venue as of the day it was written, so its first live exercise is also the cancel path's
  first live exercise. It does not get called proven.
- **−** The sweeper enforces the *operator's* ceiling, never the strategy's shorter request. A
  strategy that asked for a 5 s quote and then died has that quote pulled at 60 s, not at 5 s. That
  is the safe direction for a request nobody is renewing, and it is still a real gap between what
  the wire field says while the producer is alive and what survives it.
- **−** `swept` and `resweeps` are counted in `IntentStats` and are **not on the status line** —
  which this codebase's own rule (ADR-0020 §8) says is the same as uncounted. Each sweep does log a
  line naming the symbol and the reason, and a sweep is rare enough that a line per event is a
  better record than a counter; but the two numbers belong in `IntentLine` beside `busy` and
  `stalled`, and adding them is a `core.rs` change this increment did not own.
- **−** A frozen *event* clock runs no pass, so it runs no sweep. A session whose market-data feed
  has died entirely holds its resting orders until the feed returns or the switch fires. This is
  ADR-0020 §2's cost restated, not a new one, and it is why §7's table exists.
- **−** The sweep walks the tracker's open orders under a read lock once per second of event time,
  on the deterministic core thread. Negligible at the open-order counts this runtime has ever seen
  and not free; a Phase-8 core would want the oldest-order age maintained incrementally by the
  tracker rather than rescanned.
- **−** `swept_at` is a linear scan over a `Vec`, pruned to what is still open. Correct and
  allocation-free at a handful of orders; a session holding hundreds would want a map, and nothing
  refuses to run at that size — it just gets quadratic quietly.
- **−** The schema bump means a v1 signal log cannot be replayed by this build and cannot be
  honestly upcast (§2). Existing v1 captures re-observe with `--no-signals` and must be re-captured
  to re-decide.

See [ADR-0014](0014-signal-to-order-planning.md) §6 (the leave-it-resting exception this bounds, and
the cancel/replace rule it belongs to), [ADR-0020](0020-runtime-intent-source.md) §4 and §6 (the
reading of zero this transfers, and the `CancelId::OrderId` re-addressing it reuses),
[ADR-0010](0010-execution-events-and-reconciliation.md) §3 (the ungated cancel this depends on),
[ADR-0013](0013-runtime-supervision-and-safety-loop.md) §2–3 (the mark clock this declines to copy,
and the dead-man's switch this does not replace),
[ADR-0030](0030-live-parity-monitor-and-the-coverage-denominator.md) §6 (why a detector that has
never run live does not get a kill switch), [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the
record whose `reserved` bytes this spends).
