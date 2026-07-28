# ADR-0020 — The runtime's intent source: joining Python to the venue

**Status:** Accepted · **Date:** 2026-07-25

## Context

[ADR-0013](0013-runtime-supervision-and-safety-loop.md) ends with a consequence stated as a
weakness: *"There is still no source of order intents, so the submit pipeline is fully built and
exercised only by tests."* [ADR-0014](0014-signal-to-order-planning.md) built the other half —
`SignalReader` validates a record, `Planner` turns it into `OrderRequest`s — and ends without a
caller either. Both halves were finished. Nothing connected them, so the workspace could prove
everything about trading except that a strategy could trade.

Joining them is four lines of glue and about a dozen decisions, and the glue is the part that
does not matter. Each of these has an obvious answer that is wrong:

1. **Where does the sync/async seam go?** The submit pipeline is `async` and the reader and
   planner are not. The obvious move is to run the whole join on the tokio edge, where the client
   already lives — one task, no channel. That throws away the property the planner exists for:
   `Planner::plan` is pure so a replayed session re-plans to the same orders with the same
   `cloid`s, and it can only be pure if the position, book and working orders it reads are the
   deterministic core's, ordered by the same event clock. On the edge they would be whatever two
   tokio tasks happened to interleave.
2. **What state does a plan see, and as of when?** "Read the tracker, read the book" is two reads
   at two instants. A fill landing between them makes one of the two numbers describe a market the
   other one did not.
3. **What stops one target becoming two orders?** ADR-0014 is emphatic that the order is the
   *delta*. But the delta is computed against what the tracker knows, and an order we submitted
   200 µs ago is not in the tracker yet. Plan twice before the first submit lands and both plans
   compute the same delta — the exact compounding bug the delta rule was written to prevent,
   reintroduced by the plumbing underneath it.
4. **`ttl_ms == 0` meant two different things.** Rust read it as "unset — use the operator's
   ceiling". Python's `_check_ttl` raised `ValueError` on it, saying *"the contract gives 0 no
   meaning"*. Both cannot be the contract, and the obvious resolution — the producer is the
   authority on its own field, so keep refusing it — does not survive one question: zero is what
   the field contains when nobody wrote it, so the consumer has to decide *something* about zero
   whether or not the producer will emit it.
5. **`urgency` is a bare `u8` on the wire.** ADR-0014 §3 defines exactly what 0, 1, 2 and 3+ mean
   in venue terms. Python range-checks `0..=255` and documents "0 = most passive", and that is
   all. A researcher writing `urgency=2` because it "sounds about right" has picked *cross the
   spread* without being told.
6. **What happens when the ring file is not there?** Python may legitimately start after Rust.
   Refusing to start is wrong (it makes an ordinary race an outage) and so is ignoring it — a
   session that believes it is trading and is not is the worst outcome available here, worse than
   crashing, because it looks exactly like a strategy with no opinion.
7. **How does an operator know any of this is working?** A dead producer, a schema-drifted
   producer and a strategy that is simply flat today all produce the same output: nothing.

## Decision

### 1. The seam is where it already was: reader and planner on the core thread, a bounded queue to the edge

```text
  Python ─▶ signal ring ─▶ SignalReader ─▶ Planner ─▶ │queue│ ─▶ Haltable→Guarded→Governed→Exchange
            ────────────── axon-core thread ───────── │     │ ──── tokio edge ────
```

`SignalReader::drain` and `Planner::plan` take no clock, do no I/O and never `await`, so they run
inside the core loop's existing iteration, immediately after `drain_available`. Nothing on that
thread has a runtime handle in scope — the queue is the same bounded `crossbeam` channel the event
bus uses, in the other direction, for the same reason (ADR-0008).

The rejected alternative is worth naming precisely, because it is cheaper and looks fine: run the
reader and planner in a tokio task next to the client. It works, and it silently forfeits replay
determinism, which is not a feature of this component but the precondition for the entire Phase-5
parity harness. A backtest that cannot be diffed against a live run is a backtest nobody can
trust, and the diff is only possible if the same signals against the same state produce the same
orders — including the same `cloid`s, which are derived from the signal's identity precisely so
they can be compared.

The queue costs a poll: the submit task wakes every `intent.submit_poll_ms` (2 ms) rather than
being woken by a send. It polls rather than blocking on a receive for the same reason the core
loop does — a task blocked on a channel cannot also hear a shutdown, and shutdown is exactly when
intents must stop first.

### 2. One state, one clock, one pass

Everything a pass reads is read once, under one tracker lock, at one event time:

- the **signed position** from `OrderTracker::position` — filled quantity only, per ADR-0014 §6,
  because resting size is represented by the working-order list rather than folded into the
  position twice;
- the **top of book** from `MarketDataProcessor` (the `bbo` feed, falling back to the L2 book);
- the **working orders** projected from the tracker into `axon_strategy::WorkingOrder`.

The clock is `CoreHandler::last_ts()` — the event time of the newest event applied — and it is
*also* what the reader ages the signal against. That is the whole point of running the pass after
the drain: the state the planner sees is the state that event left behind, so a plan can never be
arithmetic about a market it had only half seen. A wall clock here would make a replayed session
call every signal infinitely stale.

**The pass *schedule* is on that clock as well**, and that is not a detail. `drain_interval_ms`
decides which records share a pass, and a record that shares a pass with a newer target for the
same symbol is `superseded` rather than planned (§3). Paced on `Instant::elapsed`, that grouping
becomes a property of how fast the machine drained the bus: replay a capture through any driver
that feeds events faster than one interval of *wall* time per event and two live passes collapse
into one, so the replay plans **one** order where the session placed two — a parity divergence
caused by replay speed rather than by logic. Comparing `now.saturating_sub(last_pass_ns)` instead
makes the schedule a pure function of the event stream, which is what the recorded `release_ts`
already assumes. The cost is that a session receiving no market data at all runs no passes, which
is the same thing the `last_ts == 0` rule above already says: with no clock and no book there is
nothing a pass could safely do.

Two consequences fall out. A session that has received **no market data at all** (`last_ts == 0`)
skips the pass entirely rather than draining: there is no clock to age a signal against and no
book to price it with, and consuming records anyway would advance the reader's `seq` baseline and
throw the strategy's first targets away with nothing to show for it. And a **poisoned tracker
lock** skips the pass too — planning a delta against a position we cannot read is guessing at
exposure, and guessing low is the direction that places the order which breaches the limit.

### 3. One batch in flight, and the newest target per symbol

The tracker learns about an order when its ack comes back, not when it is planned. So:

- **Within a pass**, only the newest accepted signal per symbol is planned. The older ones are
  read, validated, counted (`superseded`) and dropped. This is free rather than lossy: a
  target-position signal is self-contained, which is the entire reason ADR-0006 chose that shape,
  so an older target carries nothing the newer one lacks. Planning both would place two orders for
  one intent.
- **Between passes**, the core will not plan again while the edge still has an intent in flight.
  A batch is "in flight" from the moment it is queued to the moment the last `place_order` in it
  has returned and been recorded, and the reader does not even drain in the meantime — the ring is
  the queue, and a second queue behind it would age signals invisibly where nobody could see them
  expire.

The alternative was to register planned orders into the tracker at plan time rather than at ack
time. It closes the same hole and it invents venue state we have not observed: a submit that then
fails leaves a phantom order inflating `risk_position` until reconciliation notices, and ADR-0013
§4 already refuses to synthesize venue state in the *other* direction for the same reason.

### 4. `ttl_ms == 0` means "the operator's ceiling", on both sides

Rust's reading wins and Python now spells it: `axon.strategy.context.TTL_OPERATOR_CEILING`.
`_check_ttl` accepts `0` and still refuses anything negative, which is not a duration at all and
would wrap into a 49-day window on an unsigned wire field.

The deciding argument is not which side owns the field. It is that **zero is the value of a field
nobody wrote**, so refusing to emit it never stopped it arriving, and the consumer had to define
it regardless. Given that, the definition has to be the safe answer for a producer that never
thought about staleness at all:

- *"never expires"* — rejected. It gives that producer no protection whatsoever, which inverts
  the intent: the strategy that thought least about staleness would be the one with the longest
  licence to act on a stale target.
- *"already expired"* — rejected. A bug that leaves the field unset would then silently stop a
  strategy trading, and a strategy that stops trading for free is indistinguishable from a quiet
  market. Failing closed is right for a *price*; it is wrong for a field whose absence is
  ambiguous.
- *"the operator's ceiling"* — accepted. Every non-zero `ttl_ms` is capped by that ceiling
  anyway (`min(ttl_ms, max_signal_age_ms)`), so a strategy can only ever ask for a **shorter**
  window than the operator allows, never a longer one. Zero simply asks for nothing. The ceiling
  always binds, and the operator is who answers for the fills.

Python's default is unchanged at 500 ms, so nothing that was already explicit changes behaviour;
what changes is that "I have no opinion" is now sayable instead of being a rejected input that
arrived anyway.

### 5. `urgency` is named on both sides, and saturates on both sides

`URGENCY_POST_ONLY = 0`, `URGENCY_JOIN = 1`, `URGENCY_CROSS = 2`, `URGENCY_TAKE = 3` — ADR-0014
§3's table, in its own order, exported from `axon.strategy.context`. A strategy author now picks a
documented execution behaviour rather than a number that happens to sound about right.

Values above the table are **not** refused, matching the planner: `urgency = 255` means "as fast
as possible", and rejecting the record would drop precisely the signal you least want dropped.
Python range-checks `0..=255` because that is the wire type, and says nothing more.

The rejected alternative was an enum on the Python side. It reads better and it makes the wire
field un-writable by a strategy that needs a level a future venue defines — the escape hatch
matters more here than the tidiness, and the constants are as legible in practice.

### 6. Cancels are submitted before orders, and addressed by an id the venue recognizes

`Plan` already orders `cancels` before `orders` and ADR-0014 §6 requires the caller to keep that
order; `submit_intent` does, sequentially. Backwards, this is a window in which the superseded
order and its replacement are both live and we hold double the intended exposure.

One thing ADR-0014 could not know: the planner cancels by `cloid`, because that is the identity it
has — but an *adopted* order's `cloid` may be one the tracker synthesized from a venue order id to
key it into its own map. The venue has never seen that id. A cancel sent under it fails, and the
stale quote it was supposed to remove stays resting exactly where somebody else's taker is looking
for it. So the runtime re-addresses any cancel for an order it did not mint the `cloid` for to
`CancelId::OrderId`, which the venue cannot misunderstand.

**A failed cancel does not suppress the order.** The overwhelmingly common cause of a cancel
failing is that the order is already gone, and refusing to place on the venue's own success would
stall the strategy. The residual case — a venue erroring while the old order still rests — is
bounded by `OrderTracker::risk_position`, which counts that resting size as exposure the guarded
client must admit the new order on top of. The planner does not model it; the gate does.

### 7. A quote is usable only while the risk gate's own price is, and a working order is only trusted if we placed it

The planner fails closed on a missing, locked or crossed book. It cannot fail closed on a *stale*
one, because a stale book looks perfectly well-formed. So the runtime applies two freshness tests
before handing one over, and they catch different things:

- **`MarkCache::get`** — the same window and the same liveness clock the risk gate sizes positions
  against. It answers "has this instrument gone quiet?", and its clock keeps advancing on a dead
  feed where event time would freeze and call every stale price fresh (ADR-0013 §2). Reusing the
  window rather than adding a second one is deliberate: an instrument the gate has already
  declared too stale to size a position in is not one the planner should be quoting into either,
  and two windows would mean two answers to one question.
- **the quote's own `ts_event`** — because a live venue mark can arrive while the book behind it
  has not moved. The mark cannot stand in for this: `MarkCache` gives a venue `Ticker` *strict*
  precedence over book mids, so a live `activeAssetCtx` keeps `get` fresh for an instrument whose
  `l2Book` subscription a reconnect quietly failed to restore.

Which quote, and where its timestamp comes from, is `axon_runtime::quote::top_of_book` and it is
the **only** implementation: the market-data publisher (ADR-0012) calls the same function, because
features computed against one book while orders are priced against another is the divergence that
justified having a single venue connection in the first place. The `bbo` feed answers for as long
as it is still speaking and the L2 book takes over when it is not — which makes a half-restored
subscription self-correcting instead of self-perpetuating, and is *not* the same as taking the
freshest of the two on every event, which would alternate between two feeds that disagree
transiently about size and stop the publisher coalescing anything.

Ageing the book at all needs a per-symbol timestamp. `MarketDataProcessor::last_ts` is *global*,
so a frozen book on one instrument is invisible from it — the symbol keeps looking current for as
long as any other symbol is trading. That timestamp is `MarketDataProcessor::book_ts`, kept beside
the levels it describes. It briefly lived in `axon-runtime` as a `BookClock` fed by the same
fan-out; one book with two recorded ages is a disagreement waiting for the day the two are updated
from different places, so it was moved to the one structure that owns the levels.

Any of these failing yields no quote, so **stale collapses into missing**, which is the failure
the planner already handles correctly: it cancels what is working rather than leaving a limit out
in a market it can no longer see. That is the same trick ADR-0013 used for marks.

`WorkingOrder` gains one field, `placed_by_us`. A venue's order list carries no time-in-force and
no reduce-only flag, so projecting an adopted order means inventing both — and an invented
`Gtc`/not-reduce-only that happens to match the order we were about to place would trip ADR-0014
§6's leave-it-resting exception and leave a **reduce-only** quote in place of the one meant to
open the position. The strategy then sits flat behind an order that cannot ever move it. So the
exception now applies only to orders whose every field the caller can vouch for; everything else
is cancelled and replaced. The projection is also **sorted by `cloid`**, because the tracker holds
orders in a `HashMap` and an unsorted cancel sequence would differ between two runs of the same
session.

### 8. A missing ring is a degraded state with a name, and the join is on the status line

`LazyRing` opens the ring lazily, retries every `intent.attach_retry_ms`, logs the **first**
failure of each outage rather than one per retry, counts every failure, and reports
`SIGNAL RING DETACHED` on the status line for as long as it lasts. The session keeps running:
market data, reconciliation and the dead-man's switch are all still doing their jobs, and the one
thing that is missing is the one thing the operator is told about.

The status line gains `sig <accepted>/<rejected> sent <orders>+<cancels>c`, plus `noquote`, `held`
and `busy` when they are non-zero. The gap between "accepted" and "sent" is the whole diagnosis
when somebody asks why a strategy that is definitely emitting has not traded, and
`SessionHealth`'s new counters separate an order the venue refused from one **we** refused while
halted — a `ProviderError::Rejected` reads identically in the log either way.

Two of those counters are warnings rather than numbers, because they mean the path has stopped
working rather than that it is busy:

- **`INTENT STALLED`** — a batch has been in flight longer, *in event time*, than a signal stays
  worth acting on (`intent.max_signal_age_ms`, reused rather than reinvented: past that point
  nothing that arrived while the edge was stuck could still become an order). This is the one
  degradation that hides behind counters which **stop moving**: the core will not plan again until
  the edge answers, so it stops draining the ring, `accepted` freezes, and the line otherwise
  reads exactly like a strategy with nothing to say.
- **`POISONED TRACKER n`** — a panic left our own order state unreadable, so every pass is
  abandoned. On a quiet account no exec event arrives to raise `EXEC EVENTS DROPPED` either, and
  the session goes on printing `open 0 flat` while holding a position.

Counted-but-not-surfaced is the same as uncounted, which is what both of these were.

### 9. The intent source is on by default, and the config refuses the combinations that would trade nothing

Default **on**. A runtime whose intent source is off by default recreates exactly the gap this ADR
closes, and would report `OK` for a session that cannot place an order. Being on costs nothing
offline (the source is canned) and degrades cleanly live (no ring → detached). The default
`environment` is still `backtest`, so `cargo run --bin axon` still cannot reach a venue: what
decides whether a socket opens is unchanged, and the intent source reads a local file.

Validation refuses, in the spirit of ADR-0013 §7 — every one of these leaves a session looking
healthy while it could not possibly place an order, which is indistinguishable from a strategy
that had no opinion:

| refused | because |
|---|---|
| `max_per_drain = 0` | the reader takes nothing off the ring, forever |
| `max_signal_age_ms = 0` | every signal expires on arrival |
| `queue_capacity = 0` | every planned intent is dropped between core and venue |
| `drain_interval_ms`/`submit_poll_ms`/`attach_retry_ms` `= 0` | spins a thread at 100 % |
| empty `ipc.signal_ring_path` | the source has nowhere to read from |
| `ipc.capacity` not a power of two | the ring header refuses it, so the source runs permanently detached — a config error that presents as an idle strategy |
| `dead_mans_switch = false` **with** `intent.enabled = true` on a live wiring | there is now something in the runtime that places orders, so "read-only" is no longer a property a config can merely claim |

That last row is a behaviour change: before this ADR a live session could legitimately disable the
switch, because nothing could place an order. Turning the intent source off is now the only way to
say "read-only", and it has to be said.

### 10. The offline session proves the join, and stays free of tokio

`cargo run --bin axon` now replays canned **signals** as well as canned events: a stale record
that is refused and counted, a BTC target against a position the event stream moved and an order
the event stream left resting, and an ETH target for an instrument with no `bbo` feed so the L2
fallback is exercised. It prints the side, size, price and TIF of every order it planned, because
"the join produced two intents" is equally true of a build that sends the target instead of the
delta.

The offline sink **records** rather than submitting. Driving the async client would mean building
a tokio runtime inside `run_offline`, and that would cost the property that makes the offline run
worth having: ADR-0013 §7's claim that the deterministic path is provably runtime-free. What an
`Intent` becomes at a venue — cancels first, the halt suppressing placements, the ack reaching the
tracker — is proven separately against a spy `ExecutionClient` in `intent`'s own tests, which is
where an async assertion belongs.

## Consequences

- **+** Phase 4 is joined end to end. A `Signal` written by Python now becomes an `OrderRequest`
  at the venue through the halt switch, the risk gate and the rate governor, and the submit
  pipeline ADR-0013 shipped without a caller has one.
- **+** The whole path is deterministic and offline-testable. Two runs of the offline session plan
  byte-identical intents, cancels included, which is the property the Phase-5 parity harness needs
  and could not previously assert about anything but market data. Since the pass schedule is event
  time (§2), that now holds at whatever speed a log drains: the round-trip test used to buy it
  with a one-millisecond `sleep` per event, which does not scale to a real capture and was not
  available to `axon-replay`'s own driver at all.
- **+** Every field on the 64-byte record now changes an outcome, and the two languages agree on
  what `ttl_ms == 0` and `urgency = 2` mean. The remaining silent-drift surface on that boundary
  is `model_version`, which is audit metadata rather than behaviour.
- **+** "Are we trading on the signals Python thinks it sent?" is answerable from one status line
  without reading a log, and the answer distinguishes a dead producer from a quiet strategy.
- **−** The in-flight gate is **global, not per symbol.** A slow submit on BTC delays the next
  pass for ETH as well. Correct but coarse: the fix is per-symbol in-flight tracking, and at
  per-decision signal rates against a ~50 ms venue round trip it has not been worth the machinery.
- **−** If the submit task wedges, the core stops draining the ring entirely, the ring fills, and
  Python's `StrategyRunner` raises `BackpressureError`. That is the designed response to a dead
  consumer and it is loud, but it means one stuck HTTP call presents as a *Python-side* failure.
  The Rust side now says so too — `busy` on the line and `INTENT STALLED` past the ceiling (§8) —
  and it is no longer a **wedge**: `ExchangeClient` used to build its `reqwest::Client` with no
  request timeout, so a stalled TCP connection hung one `place_order` for as long as the kernel
  kept the socket. It now carries an explicit 10 s deadline, which turns a silent stall into a
  counted `intent_failures` the pump moves past.
- **−** Shutdown can now *abort* the submit task rather than only time out on it. Dropping a
  `JoinHandle` detaches the task, so the previous `let _ = timeout(...)` let an in-flight
  `place_order` complete after the sweep had read an empty book — the venue would rest that order
  behind a dead-man's switch the same shutdown had just disarmed. An abort is still not proof the
  placement never left, so `ShutdownOptions::submitter_stopped` tells `graceful_shutdown` to leave
  the switch **armed** in that case, exactly as a failed sweep does, and the closing status line
  says `SUBMITTER ABANDONED`. The cost is one of the venue's ten daily triggers, against an
  unmanaged position with no process left to close it.
- **−** A `bbo` feed that dies is only detected once the mark window (10 s) has passed, because
  that is the window §7 deliberately reuses. Until then the frozen quote is priced against and
  published; after it, the L2 book takes over. Shortening it would mean a second answer to "how
  old is too old", and every instrument on a genuinely quiet market would start failing closed.
- **+** `MarketDataProcessor` now keeps `book_ts` per symbol, so one structure describes one book.
  This ADR originally shipped a `BookClock` in `axon-runtime` because the processor kept no
  per-symbol book time; that was two structures for one book, kept in step only by both being fed
  the same fan-out, and it has been folded into `axon-marketdata` where the levels live.
- **−** `drain_interval_ms` (1 ms) and `submit_poll_ms` (2 ms) are pure added latency between
  Python emitting and the venue seeing an order. Immaterial against ~200 ms Hyperliquid blocks and
  exactly the kind of thing a Phase-8 core has to delete.
- **−** A restarted Python producer whose `seq` rewinds is refused by the reader until its sequence
  passes the old baseline — by design (ADR-0014 §1), but it makes "persist `first_seq`, or restart
  both sides together" an operational requirement that nothing in the runtime enforces or detects.
  The `stale_seq` counter is the only evidence, and it is not on the status line.
- **−** An adopted order can never take the leave-it-resting path, because we cannot vouch for its
  TIF. A restarted session therefore cancels and replaces every order its predecessor left, losing
  their queue positions. Correct, and it makes a restart more expensive than it needs to be; the
  real fix is a TIF on `TrackedOrder`, which is `axon-execution`'s to add.
- **−** A cancel that fails does not stop the replacement being placed, so a venue erroring on
  cancels while accepting orders can leave us doubly exposed for as long as that lasts. The risk
  gate bounds it; nothing prevents it.
- **−** The planner still has no no-op band, so a strategy emitting a slightly different target
  every tick will cancel/replace every tick. ADR-0014 named this as a weakness of a component
  nothing called; it is now reachable, and `min_order_qty` only blunts the case where the *delta*
  is dust, not the case where the delta is real but the target barely moved.
- **−** The default is on, which means an upgraded deployment starts reading
  `/dev/shm/axon-signal.ring` without being asked. Records left over from a previous run fail the
  TTL check, so the realistic accident is covered — but an unrelated live producer on that path
  would be obeyed, and nothing authenticates the ring's writer.
- **−** The offline run does not exercise the async submitter, so the code that turns an `Intent`
  into venue calls is covered only by spy tests. `run.sh` cannot catch a regression there; only
  `cargo test` can.
- **−** Each plan still allocates two `Vec`s and they are now moved through a channel as well.
  Per-decision rather than per-tick, so nowhere near the hot path, but ADR-0014's note about a
  Phase-8 core wanting this inline now applies to one more hop.

See [ADR-0014](0014-signal-to-order-planning.md) (the reader and planner this calls),
[ADR-0013](0013-runtime-supervision-and-safety-loop.md) (the session, the submit pipeline and the
mark window this extends), [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the record and the ring),
[ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the sync/async split this mirrors),
[06](../06-strategy-contract.md), [02](../02-python-rust-boundary.md).
