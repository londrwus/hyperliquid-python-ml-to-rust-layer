# ADR-0013 — Runtime supervision: composing a session, and the loop that must die last

**Status:** Accepted · **Date:** 2026-07-25

## Context

By the end of Phase 3 increment 3 every part a live session needs existed and **nothing composed
them**. `ExchangeClient` could sign and submit, `OrderTracker` could reconcile, `GuardedClient`
could refuse, `RateGovernor` could pace, `arm_dead_mans_switch` could arm, the WS client could
stream both market data and user channels, and `info.rs` could read the venue's own view of our
orders. `cargo run --bin axon` printed a config and exited.

Composition is not glue here — it is where the remaining safety properties live, and each of them
has a plausible wrong answer:

1. **Where does the deterministic core actually run?** [ADR-0008](0008-market-data-bus-and-ws-ingest.md)
   settled the *channel* between the async edge and the core. It did not settle the thread model,
   and "spawn the core as a blocking task" would put a tokio handle in scope on the one code path
   that must never await.
2. **What price does the risk gate measure exposure in, and when does that price stop counting?**
   [ADR-0010](0010-execution-events-and-reconciliation.md) made a *missing* mark fail closed and
   left `MarkCache` unpopulated by anyone.
3. **What happens when the dead-man's switch fails to re-arm?** Treating it as a log line is the
   obvious answer and the wrong one.
4. **How does a restarted process find what it left resting**, given that `orderUpdates` never
   snapshots?
5. **Where does the rate governor sit relative to the risk gate?**
6. **In what order does a session shut down?**

## Decision

**1. The core owns a thread; the supervisor owns the runtime.** `run_live` spawns `axon-core` as a
plain `std::thread` running a synchronous drain loop, and the main thread `block_on`s the
supervisor. Nothing on the deterministic path has a runtime handle in scope, so nothing on it *can*
await. The loop polls (`drain_available`, then park) rather than blocking on a receive, because it
also has to service the status line and notice a shutdown — and a session whose feed has died is
exactly the case where a blocking receive never returns. The loop exits only once the stop flag is
set **and** the bus is empty, so the cancel acknowledgements from the shutdown sweep are still
applied before the final status line claims what is resting.

One [`CoreHandler`] fans each event to the book, the mark cache and the order tracker in that order:
the mark updater reads the book it just updated, and the tracker — the only consumer behind a lock —
goes last so the lock is held for the shortest span. One handler because there is one ordering; three
independent consumers would let a fill be applied against a book that had not yet seen the trade
that caused it, and replay would stop matching live.

**2. Marks come from the venue's mark price, fall back to the mid, and expire.** Precedence is
`Ticker::mark_px` → book/BBO mid, and a mid can never displace a *live* venue mark. The venue's mark
is the number margin, liquidation and unrealized PnL are computed against; checking our notional
against anything else measures a different quantity than the one that can liquidate us. The **last
trade is never a mark source** — one print in a thin book is an outlier, and a quiet instrument's
last print can be minutes old while the book has moved.

The staleness question is settled explicitly, because a stale mark is worse than a missing one: a
missing mark fails the gate closed and costs one trade, while a five-minute-old mark sails through
the notional check and sizes a position against a price that no longer exists. **Entries expire**
(10 s by default) and an expired entry is indistinguishable from an absent one to the gate — stale
collapses into missing, which is the failure mode the gate already handles correctly.

That forces a second, less obvious decision: expiry needs a clock, and event time alone cannot
provide one. A dead feed emits no events, so event time freezes and a frozen clock calls every price
fresh — precisely the case we are trying to detect. So `MarkCache` keeps a high-water "now" advanced
by two sources: every price it is handed, and (live only) the core loop's wall clock. Backtests never
call `observe_now` and their staleness is measured purely in event time, so replay stays reproducible.

**3. A failed re-arm is the protection being gone, and is graded by protection remaining.** Not by
attempts: `DmsPolicy` is a pure state machine over an externally supplied `now_ms` that maps
`(deadline, now)` to one of *retry* (more than one interval left), *halt new orders* (one interval or
less left), or *unprotected* (none left → shut the session down). Halting **before** the deadline
rather than at it is the point — the orders placed in the final interval are the ones that get
stranded. A recovered re-arm clears the halt; a shutdown-issued halt is terminal and cannot be
cleared. A session with the switch enabled therefore **starts halted** and only begins accepting
orders once the first arm succeeds.

The re-arm interval is bounded from both sides, and both bounds are venue arithmetic rather than
taste. Below: 5 s, matching the venue's minimum lead, because a faster cadence buys no protection and
costs 17 280 address credits per UTC day — more than the entire 10 000-credit lifetime buffer of an
address that has not traded. Above: the lead must be at least **3×** the interval, so two consecutive
failures are survivable; a tighter ratio lets one HTTP 500 fire the switch, and the venue honours only
`SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY` (10) firings before the account is unprotected for the rest of
the day. The shipped default is a 60 s lead re-armed every 20 s: 4 320 actions/day, printed at startup
because on a fresh account the safety loop is the largest consumer of the address budget.

> **Amended 2026-07-26.** The table above is right and unchanged. The sentence that opens
> this section was not: *"a failed re-arm is the protection being gone"* named the only event
> the loop then consulted the table on, and protection also goes away without anything
> failing. A 1 h 44 m soak froze the process with `SIGSTOP` for 80 s against a 60 s lead
> ([07](../07-parity-and-testing.md)). Every arm before the gap succeeded and the arm after
> it succeeded, so `on_failure` was never called; the status line read `dms 0s` — the
> venue-side switch had actually fired — and then `dms 55s`, with **no `HALTED`, no
> `UNPROTECTED` and nothing on stderr**. A 48 s freeze reached `dms 9s`, deep inside the halt
> band, equally silently. A stalled process arms nothing and *fails* nothing, so a ladder
> consulted only on an error has no rungs on the one path the switch exists for.
>
> The loop now grades the same table twice per pass: once on the elapsed clock before it
> attempts anything (`DmsPolicy::on_elapsed`), and once on the outcome. Three things about
> that are decisions rather than mechanics.
>
> **It reads wall time, and here that is the correct clock.** §2 above established that
> staleness needs a two-source clock, because a dead feed emits no events and a frozen event
> clock calls every stale price fresh. Protection remaining is not that quantity: the
> deadline is a wall-clock instant **the venue holds**, and `scheduleCancel` fires when the
> venue's clock passes it — whether or not our feed is delivering, whether or not our event
> clock has moved, and whether or not this process is running. Aged in event time it would
> report infinite protection in exactly the case being detected. It is also why the check is
> not on a monotonic `Instant`: a machine that suspends stops `CLOCK_MONOTONIC` and does not
> stop the venue.
>
> **It notices; it does not protect.** A process that is not running runs no checks. What
> this delivers is that a **resumed** process escalates instead of continuing as if nothing
> happened. The venue-side deadline remains the only thing covering the gap itself — which is
> the entire reason the deadline exists — and the honest claim is the narrow one. Anyone
> reading `on_elapsed` as "freezes are now handled" has read it wrong.
>
> **It cannot fire on ordinary jitter, by arithmetic rather than by tuning.** The lead is at
> least 3× the re-arm interval, so a re-arm that lands on time leaves at least two intervals
> standing and reaching the halt band takes a **full missed beat**. The same arithmetic makes
> the cause unambiguous: with no failures recorded, `remaining ≤ interval` implies the loop
> was due to run an interval ago and did not, so the console says so rather than inventing a
> failed attempt to blame.
>
> A stall also earns its own counter — `late n` beside `dms` on the status line, and
> `DMS PROTECTION LAPSED n` in the warnings — because the re-arm that follows a stall
> *succeeds*, and in succeeding it repairs every other field: the deadline is pushed forward,
> `failures` is left at the zero it never left, and the halt clears microseconds after it is
> raised, long before the next status line samples it. Counted and not surfaced would be the
> same as uncounted.
>
> Verified against testnet, both rungs. A 42 s freeze on a 60 s lead resumed to
> `the re-arm loop did not run for 43 077 ms, only 15 767 ms of protection left - HALTING new
> orders`, then re-armed and traded on at `dms 58s (late 1)` — no trigger spent, because the
> deadline never passed. A 75 s freeze took the deadline with it and ended the session.

**4. Reconciliation publishes the venue's view onto the bus, and never writes an order off.** The
`/info` poll republishes every open order as an `OrderUpdate` — not only the unknown ones, since the
tracker is idempotent by construction and re-applying a known order repairs a partial fill whose WS
frame was lost. REST truth and stream truth take one path, in one event-time order, so replay sees
what live saw. The reverse divergence — an order we believe is live that the venue does not list — is
**reported, never synthesized into a cancellation**: inventing a terminal state we did not observe
removes exposure from our own risk view, and under-counting exposure is the direction that places the
order which breaches the limit. A grace window suppresses the read-your-writes race so a freshly
placed order is not mistaken for a lost one.

**5. The submit path is three nested `ExecutionClient`s: halt → risk → rate → venue.** Risk sits
above rate because a risk check is a free local computation while rate budget is a finite resource,
so an order risk will refuse must not spend any of it. Halt sits above both because when protection
is gone there is nothing left worth checking. The governor's structural invariant survives the
wrapping: `GovernedClient` may refuse a placement and **charges but never refuses a cancel**, which
is the same asymmetry the risk gate already has and for the same reason.

**6. Shutdown order: stop intents → sweep → decide the switch → drain.** `cancel_all` is a
read-then-write sweep (Hyperliquid has no native cancel-all), so any intent admitted after the read
survives it — intents must stop first, and the sweep runs more than once. The switch decision comes
*last* because its input is whether the sweep worked: a clean sweep **disarms**, because an armed
deadline outlives this process, burns one of the ten daily triggers, and fires into whatever session
is running when it expires — cancelling the *restarted* process's orders for reasons nobody will
connect to a shutdown minutes earlier. A failed sweep **leaves it armed**, because there may still be
exposure and there is about to be no process to remove it. The safety loop is the last task stopped;
a second interrupt exits immediately and deliberately leaves the deadline standing.

**7. Offline is the default, and it is a real session.** `cargo run --bin axon` builds the real bus,
handler, tracker and mark cache, pushes a canned six-event stream through them and exits — no socket,
no key, no tokio in the process. It exercises the fan-out, the mark precedence rule, order adoption
and fill accounting, so "it ran" means something. Live wiring needs an explicit `environment` in a
config file, and mainnet additionally needs `AXON_ALLOW_MAINNET=1`: a config file can be copied or
mistyped, and one file's worth of typo must not be sufficient to trade real money. Config validation
refuses the combinations that look fine and are not survivable (sandbox pointed at mainnet, a lead
below the venue minimum, a live session with no account address), and refuses outright any config
file containing a key-shaped field.

## Consequences

- **+** A session can now be left running: it re-arms its own protection, recovers its state after a
  restart or a dropped socket, paces itself inside the venue's budget, and unwinds in an order that
  cannot strand orders behind it.
- **+** The offline path is a genuine smoke test of the composition, so CI catches an unwired
  consumer instead of discovering it live.
- **+** Every escalation decision (`DmsPolicy`), every divergence decision (`reconcile::diff`) and the
  shutdown ordering are pure or trait-generic, so the branches that only run on the worst day are
  unit-tested offline against spies.
- **−** Expiring marks means a feed hiccup longer than `mark_max_age_ms` refuses risk-increasing
  orders until it recovers. That is the intended trade, but it makes the window a real tuning knob:
  set it too tight on a slow instrument and the session silently stops trading.
- **−** The mark cache's liveness clock mixes event time and wall time in a live session. They are
  the same epoch and the mixing is confined to staleness (never to ordering), but it is the one place
  in the runtime where wall time influences a trading decision.
- **−** The core loop polls, so it burns a timer wakeup per idle interval and adds up to
  `core_poll_us` of latency to an event that arrives just after a park. Fine for Phase 3; a
  thread-per-core Phase 8 wants a real timed receive.
- **−** The dead-man's switch is a standing cost against a cumulative, volume-gated budget. A session
  that trades little can spend most of its address credits staying protected.
- **−** `cancel_all` inside `GovernedClient` is charged as one request although the sweep sends many;
  the periodic `userRateLimit` read corrects the estimate, so the drift is bounded by one poll
  interval rather than eliminated.
- **−** There is still no source of order intents (Phase 4 brings the Python bridge), so the submit
  pipeline is fully built and exercised only by tests. The halt switch and risk gate are real code on
  a path nothing yet calls.

See [ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the bus and the sync/async split this extends),
[ADR-0009](0009-hyperliquid-signing.md) (env-var key, agent wallet),
[ADR-0010](0010-execution-events-and-reconciliation.md) (the tracker, the risk gate and the
`orderUpdates` snapshot gap this closes), `docs/01-architecture.md`,
`docs/research/hyperliquid-execution.md`.

[`CoreHandler`]: ../../crates/axon-runtime/src/handler.rs
