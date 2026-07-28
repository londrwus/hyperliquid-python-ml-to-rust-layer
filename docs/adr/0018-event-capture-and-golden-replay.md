# ADR-0018 — Event capture, deterministic replay, and what a replay refuses to claim

**Status:** Accepted · **Date:** 2026-07-25

## Context

`docs/07-parity-and-testing.md` opens with a ladder, and the bottom rung is this: *capture a
real event log, replay it through the exact production code path, assert outputs match a
stored reference within tolerance and that no discretized decision flips.* Everything above
that rung — the model-parity gate, the feature-parity gate, shadow-trading diffs, the live
parity monitor — compares two runs and alarms on the difference. None of those comparisons
means anything until two runs over one input are known to be identical.

Nothing captured and nothing replayed. The ingredients were all there — the bus (ADR-0008), one
event-time ordering key on every event, `ManualClock`, and `run_blocking_clocked`, which exists
for precisely this — but no code turned an event stream into a file, and no code turned a file
back into an event stream. The ladder had no bottom.

Five questions had to be answered, and each has an answer that reads as obviously correct until
it silently invalidates the gate it was supposed to support:

1. **What is the log format, and who owns its shape?** `MarketEvent` and `ExecEvent` already
   derive `serde`, so `serde_json::to_string(&event)` is right there.
2. **What order does replay publish in?** The architecture says the core is "keyed on event
   time, not processing time", so sorting by `ts_event` is the obvious reading.
3. **How does replay reach the handlers?** The natural shortcut is to call `handler.on_event`
   directly and skip the bus entirely — it is one function call, and the bus adds a channel and
   a thread for nothing.
4. **Does Python drive a Rust binary, or read the same JSONL itself?** Reading it in Python is
   simpler, faster to iterate on, and removes a toolchain dependency from `pip install axon`.
5. **What is a replay allowed to claim?** Once a log replays through the core, calling the
   result a "backtest" costs nothing and reads well in a README.

## Decision

**1. The log is JSONL — a versioned header line, then one record per event — and it owns its own
wire form.** `crates/axon-replay` defines `LoggedEvent`, a mirror of `axon_core::Event`, rather
than deriving `Serialize` on `Event` itself. Two reasons pull the same way. Persistence is
versioned on its own schedule, and binding the on-disk format to an in-memory enum's derive
makes every refactor of the core a silent format change. And the `From<&Event>` conversion is an
**exhaustive match**, so a new `Event` variant is a compile error in the capture path until it
is given a wire form — an untagged blob, or a catch-all arm, would instead drop a whole class of
event out of every capture with nothing to report it.

JSONL rather than a packed binary frame because a golden log's value is in being readable when a
replay diverges: the first question is always "which event differs", and `diff` answering that
in one command beats the bytes a binary format would save. The cost is named below.

Two properties of the format exist only to stop a log being *misread*, which is worse than being
unreadable:

- `LogHeader` pins `schema` and `schema_version`, and the reader **refuses** anything else. The
  event types are structurally decoded, so a field added with a default or a renamed variant
  would let an old log deserialize into a new shape that means something different — a gate
  passing against a reference it no longer understands. The version is a human promise, so a
  test pins one record's exact JSON: a vocabulary change breaks that test rather than the format.
- Each record carries `ts_event` **beside** the payload that already contains it, and the reader
  rejects the pair when they disagree. `Ticker::ts_event` is *derived* (venue time when there is
  one, receipt time otherwise, ADR-0011), so a change to that rule would re-key events a log had
  already ordered and the replay would interleave differently from the reference — a divergence
  that presents as a strategy bug, not a format bug.

**2. Capture is an `EventHandler`, and a failed write is latched rather than logged.** Recording
a session is adding a handler to the core loop, so what the log contains is what the core *saw*,
in the order it saw it — the only sequence a replay could reproduce. `EventHandler::on_event`
returns nothing, so the first write error is stored and surfaced by `Capture::finish`. A disk
filling mid-session would otherwise leave a short log that replays perfectly and proves nothing:
the harness would report success over the events it managed to keep, and the missing tail would
look like a session that simply ended.

**3. Replay goes through the bus and `run_blocking_clocked`, not straight into the handler.** A
scoped producer thread publishes onto the bounded bus and the core drains it exactly as it does
live. Skipping the bus would be faster and would remove a thread, and it would also mean the
replay was exercising a *different* dispatch path from the one production uses — which is the
one thing the harness may not do. The thread is a scheduling detail and not a source of
nondeterminism: one producer plus a FIFO channel means the consumer observes exactly the send
order, whatever the scheduler does in between.

`run_blocking_clocked` sets a `ManualClock` to each event's own `ts_event` before dispatch, so a
handler asking "what time is it?" gets the event's time. Without that, any TTL, staleness check
or timer would measure the age of the *replay run* and come out differently every time. The
golden test's digest deliberately includes each handler's clock reading, so a handler that
reaches for a wall clock breaks the test rather than quietly poisoning the gates above it.

**4. Event-time is the default order, capture order is available, and late arrivals are counted
rather than fixed.** `ReplayOrder::EventTime` runs the log through the core's own `TimedQueue`
— reusing the core primitive, not a local `sort_by_key`, so replay and any future event-time
scheduler cannot drift apart on tie-breaking. Ties break on capture sequence, because event time
alone does not order two events stamped the same nanosecond and a book snapshot must not swap
with the trade that produced it on one run in a thousand.

The subtle part is that the two orders are *both* honest and answer different questions.
Event-time order is the architecture's claim about the core. Capture order is what the live
session actually experienced, including its out-of-order arrivals, and it is the only order
whose output may legitimately be compared against a live-captured reference. They differ on
exactly the late events, so `ReplayReport::late_arrivals` reports how many there were. Silently
sorting a late feed into shape would let a replay certify an interleaving that never happened.

**5. Python shells out to the Rust replay binary. There is no Python fallback, by design.**
`axon.backtest.Backtester` runs `cargo`'s `replay_log` example, which replays through the real
`MarketDataProcessor` behind the real `ReplaySource`; when the binary is missing and cannot be
built, it raises `ReplayUnavailable` instead of degrading. A Python reader of the same JSONL
would be a second implementation of order-book replacement, the BBO-fallback mid, and the mark
staleness rule. Two implementations agree on the day they are written and drift afterwards, and
the drift lands inside the harness that exists to *detect* drift. `docs/07` is explicit that
parity comes from running the same code; a subprocess spawn per backtest is cheap next to
owning a second event engine forever.

**6. The golden comparison separates values from decisions as a type, not a parameter.** A
`TraceRow` carries `values` (continuous, compared within tolerance) and `decisions`
(discretized, compared exactly). Merging them behind a `decision_columns=` argument would make
"no discretized decision flips" something a caller can soften by passing a tolerance that
happens to cover it — and a flipped decision is a different order sent to a real venue, which no
tolerance makes acceptable. Two further rules fall out of the same instinct: shape is compared
before values (mismatched row identities are a *structural* failure, because diffing unrelated
events reports noise and hides the cause), and `None` against a number is always a divergence,
because a missing mark and a mark of zero are the difference between a risk gate failing closed
and a risk gate sizing against nothing.

**7. Replay does not reproduce the venue's response to orders the replay would place, and the
harness says so everywhere.** A log carries the fills and order updates the *captured* session
received, for the orders the *captured* session sent. Replay re-delivers those recordings
verbatim. If a strategy under replay decides to buy where the original did not, no fill ever
arrives for it, no book level moves because of it, and no queue position, partial-fill
sequencing, latency or rejection behaviour is modelled. The market in a log is a recording, not
a counterparty.

So replay answers *"does this code, on this input, still produce what it produced before?"* —
determinism, refactor safety, feature and model parity. It does **not** answer *"would this
strategy have made money?"* That needs a simulated venue that models fills against the book,
which is a separate adapter behind the provider port (ADR-0004) and deliberately not in this
crate. This is stated in the crate docs, in the module docs, in the Python package docstring,
and in a test named `replay_never_manufactures_the_venues_reply_to_an_order`, because a replay
harness that blurs the line manufactures confidence — and manufactured confidence is worse than
no harness, since it is acted on.

**8. Replay drives the production chain by *depending* on the runtime, as a dev-dependency.**
There is one fan-out and one strategy pass (ADR-0013 §1, ADR-0020); a copy in the harness would
drift from the live one silently, since both would keep passing their own tests, and parity would
become a claim about the harness rather than about the code. `axon-replay` therefore names
`axon-runtime` — but only under `[dev-dependencies]`, because the *production* edge has to run the
other way: a live session records itself by adding `Capture` to its own handler chain. Cargo
permits a cycle through a dev-dependency edge and forbids one through a normal edge, so nothing in
`src/` may name the runtime and the driver lives in `examples/chain/mod.rs`, shared with `tests/`
via `#[path]`. The cost is that `cargo test -p axon-replay` builds the runtime, and a break there
breaks this crate's tests.

**9. A live session records itself by holding a *tap*, and the tap is the head of the one fan-out.**
§2 said recording a session is adding a handler to the core loop. The runtime has exactly one
handler, because there is exactly one ordering (ADR-0013 §1), so a second consumer on the bus is
the thing that ADR forbids — `axon_runtime::capture` therefore puts the recording at the **head**
of `CoreHandler`'s fan-out, ahead of the book, the marks and the tracker. First rather than last
beside the market-data ring, because it is the only stop that records the fan-out's *input*: it
reads nothing the consumers below it produce, so nothing is gained by waiting, and the event a log
must never lack is the last one before a crash — exactly the one a tap placed after them would drop.

What sits there is **not** the `Capture` itself. §2's shape — serialize and `BufWriter` on the core
thread — is amortized cheap and is not *bounded*: a `write(2)` behind a full page cache, a foreign
`fsync`, or a filesystem that has gone away blocks for as long as it blocks, and it blocks the one
thread that must keep draining market data. A stall in the deterministic loop is a stale book, and a
stale book prices orders against a market that has moved. So the tap is a `try_send` onto a bounded
queue and a dedicated writer thread owns the `Capture`. Ordering survives the move because both taps
(events, signals) live on the core thread and the channel is FIFO.

Three rules follow, each the opposite of a plausible alternative:

- **A recording that cannot keep up stops; it does not drop.** The inverse of the market-data ring
  (ADR-0012 §2), and the difference is what each artifact is *for*: every `MdSlice` is full state, so
  the newest supersedes the old, while a log with a hole replays a session that never happened — and
  replays it *successfully*. A prefix is a truthful recording of a shorter session; a gap is a lie.
  Everything lost is counted, so `events + missed` is the whole session.
- **A log earns its real name only by closing cleanly.** The writer creates `<path>.partial` and
  renames it into place on a clean finish, and on no other path — not on a stop, not on a failed
  final flush, not on a kill. A truncated log that parses is the one artifact that fails silently,
  because the replay of it is green and narrower than it claims.
- **A hard size cap, and deliberately no rotation.** `EventLog` loads a log whole, so rotated
  segments are a set of files each of which replays a *different* session from the one recorded. The
  cap is set where the artifact stops being loadable rather than where the disk stops being
  writable, and reaching it stops the capture loudly rather than filling a volume the trading
  session also needs.

A *startup* failure is fatal to the session and a mid-session failure is not: nothing has been lost
at startup and the operator is watching, whereas mid-session there is a position on the book and
killing the process over a log file trades a recording for an unmanaged position.

## Consequences

- **+** The bottom rung of `docs/07` exists and is enforced offline in CI: the same log replayed
  twice through the same handler produces byte-identical output, and a fresh run is compared
  against a committed reference (`crates/axon-replay/testdata/session.golden.json`).
- **+** Backtest and live are the same program with a different producer attached, which is the
  parity property `docs/01` claims and had not yet demonstrated.
- **+** Determinism is *checkable*, not assumed. A handler that reads a wall clock, a summary
  built from a `HashMap`, or an event class missing from the capture all fail a test rather than
  becoming intermittent noise higher up the ladder.
- **+** Money survives the whole loop exactly: `Decimal` serializes as a decimal *string*, and
  the Python side parses it with `decimal.Decimal`. `docs/07`'s "same code, different numbers"
  caution is about float width, and there is no float anywhere on this path.
- **−** Capture serializes JSON on the core thread — microseconds per event. Acceptable for
  mid-frequency and for the sandbox/shadow sessions this is aimed at; a latency-critical path
  would need to tee raw events into a ring and serialize off-thread (Phase 8).
- **−** A log is loaded whole, because event-time ordering cannot begin until the last record has
  been seen (a feed that can be late can be late by any amount). Log size is therefore bounded by
  memory: a session slice, not a month of tape.
- **−** `SCHEMA_VERSION` is a promise a human has to keep. The pinned wire-format test makes
  forgetting it *noisy*, but it cannot make it impossible.
- **−** A Python backtest needs a Rust toolchain or a prebuilt binary. That is the price of
  refusing a second implementation, and it makes `axon.backtest` not pip-installable-and-go.
- **+** The trace covers the whole chain: `axon_runtime::CoreHandler` (book → marks → tracker) and
  `axon_runtime::IntentSource` (reader → planner), so a golden run compares the tracker's
  reconciled position *and* the orders and `cloid`s the planner emitted. `cloid`s are derived
  from the signal (ADR-0014 §5), which is what makes an order-level golden diff possible at all.
- **−** A session now needs *two* recordings. `crates/axon-replay/src/signals.rs` defines
  `axon.signallog`, a versioned JSONL of the records that crossed the ring, each with a
  `release_ts` the `Signal` itself does not carry — without it an expiry cannot be replayed. A live
  session now writes one (`axon --capture`). The committed fixture is still generated, because
  committing a real capture would carry a real account's fills into the repository.
- **+** A session records itself: `axon --capture <path>`, or `[capture]` in the config file, off by
  default. Both halves are written — the event log and `axon.signallog` — so a replay of a real
  session re-*decides* it rather than merely re-observing it. The whole round trip runs offline in
  CI (`a_captured_offline_session_replays_to_the_state_it_recorded`): capture, replay through the
  production chain, compare the tracker's reconciled position *and* the planner's orders and
  `cloid`s.
- **−** Recording costs one `Event::clone` per event on the core thread and a bounded queue's worth
  of memory. Both are the price of not writing from that thread, and a session that is not recording
  pays neither.
- **−** A recording is one file with a hard cap and no rotation, so a multi-hour soak will stop it.
  That is loud (`CAPTURE STOPPED (size cap)` on the status line, and the log keeps its `.partial`
  name) rather than silent, but a soak that needs more wants segments a replay can stitch, and that
  needs `EventLog` to stream.
- **−** The committed fixture is synthetic. A capture of real Hyperliquid traffic is the better
  fixture and is what this crate is for, but committing one would carry a real account's fills
  into the repository, and a fixture nobody can regenerate offline rots. It is generated by the
  real capture path (`--example make_fixture_log`), exercises every event variant, and contains
  a genuine out-of-order arrival.

See [ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the bus and vocabulary this records),
[ADR-0010](0010-execution-events-and-reconciliation.md) (one bus, so a fill orders against the
trade that caused it — the property that makes a single log sufficient),
[ADR-0011](0011-ticker-and-mark-price-feed.md) (the derived `Ticker` ordering key the log
cross-checks), [ADR-0004](0004-provider-abstraction-layer.md) (where a simulated venue would go).
