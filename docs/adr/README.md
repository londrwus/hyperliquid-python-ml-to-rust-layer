# Architecture Decision Records (ADRs)

An ADR captures a single significant, hard-to-reverse decision: its **context**, the
**decision**, and its **consequences**. They're immutable once accepted — if we change our
mind, we write a new ADR that supersedes the old one (and mark the old one `Superseded`).

Format: [Michael Nygard's template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-python-rust-boundary.md) | Python↔Rust boundary = shared memory (Boundary B → A) | Accepted |
| [0003](0003-model-serving-and-fidelity.md) | Model serving & fidelity strategy | Accepted |
| [0004](0004-provider-abstraction-layer.md) | Provider abstraction (venue-agnostic) | Accepted |
| [0005](0005-fp32-no-quantization.md) | Keep models FP32 — no quantization | Accepted |
| [0006](0006-signal-schema-and-spsc-ring.md) | Signal schema & the SPSC shared-memory ring | Accepted (schema v3 by [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)) |
| [0007](0007-linux-wsl2-dev-target.md) | Linux/WSL2 dev & deploy target; portable mmap | Superseded by [0024](0024-native-ubuntu-dev-target.md) |
| [0008](0008-market-data-bus-and-ws-ingest.md) | Market-data bus (crossbeam) + hand-rolled WS ingest | Accepted (amended 2026-07-26) |
| [0009](0009-hyperliquid-signing.md) | Hyperliquid signing: alloy crypto + hand-rolled msgpack encoding | Accepted |
| [0010](0010-execution-events-and-reconciliation.md) | Execution events on the core bus + an unbypassable risk gate | Accepted |
| [0011](0011-ticker-and-mark-price-feed.md) | A normalized ticker feed, and an event time the venue never sends | Accepted |
| [0012](0012-market-data-ring-and-multi-record-contract.md) | The market-data ring, and a contract with more than one record | Accepted |
| [0013](0013-runtime-supervision-and-safety-loop.md) | Runtime supervision: composing a session, and the loop that must die last | Accepted (amended 2026-07-26) |
| [0014](0014-signal-to-order-planning.md) | From `Signal` to order intents: validation, planning, cancel/replace | Accepted |
| [0015](0015-model-artifacts-and-registry.md) | Model artifacts: native for trees, ONNX for the rest, immutable + versioned | Accepted |
| [0016](0016-feature-spec-and-parity-gates.md) | The versioned feature spec, and parity as three gates that can fail | Accepted |
| [0017](0017-compute-offload-to-modal.md) | Offloading heavy compute to Modal through `hwsched` | Accepted (amended 2026-07-26) |
| [0018](0018-event-capture-and-golden-replay.md) | Event capture, deterministic replay, and what a replay refuses to claim | Accepted |
| [0019](0019-native-rust-inference-backends.md) | Native Rust inference: `tract` for ONNX, a hand-written XGBoost reader | Accepted |
| [0020](0020-runtime-intent-source.md) | The runtime's intent source: joining Python to the venue | Accepted |
| [0021](0021-rust-model-parity-gate.md) | The cross-language model-parity gate: a frozen Python question, answered in Rust | Accepted (amended 2026-07-26) |
| [0022](0022-first-ml-strategy.md) | The first ML strategy up the ladder, and what climbing it is allowed to claim | Accepted |
| [0023](0023-second-venue-adapter-binance.md) | The second venue: what adding Binance actually cost | Accepted |
| [0024](0024-native-ubuntu-dev-target.md) | Native Ubuntu 24.04 as the dev target; Windows kept as a CI claim | Accepted |
| [0025](0025-instrument-precision-and-rounding.md) | Instrument precision on the port: the planner quantizes, the encoder refuses | Accepted |
| [0026](0026-python-driven-execution-at-a-venue.md) | A Python decision at a venue: closing Phase 4, and what closing it does not mean | Accepted |
| [0027](0027-streaming-logs-and-the-replay-grid.md) | A log that outlives a soak, and a log that carries its own grid | Accepted |
| [0028](0028-market-data-bars-and-the-ticker-tail.md) | A bar record, and what the 48 reserved bytes were for | Accepted |
| [0029](0029-shadow-trading-and-the-continuous-diff.md) | Shadow trading `perp_bar`, and what a continuous diff can and cannot observe | Accepted (amended 2026-07-26) |
| [0030](0030-live-parity-monitor-and-the-coverage-denominator.md) | The live parity monitor, and the denominator the intersection hid | Accepted (amended 2026-07-26, 2026-07-27 by [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)) |
| [0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md) | An order lifetime, and the half of it a signal cannot enforce | Accepted (amended 2026-07-26, 2026-07-27 by [0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md) and [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)) |
| [0032](0032-the-model-zoo-and-what-actually-crosses-into-rust.md) | The model zoo, and what actually crosses into Rust | Accepted |
| [0033](0033-lightgbm-crosses-by-conversion-not-by-backend.md) | LightGBM crosses by conversion, not by a backend | Accepted |
| [0034](0034-market-data-beacon-and-the-third-clock.md) | The market-data beacon: a third clock, and what 64 bytes could not hold | Accepted |
| [0035](0035-rust-feature-runtime-and-the-bit-exact-gate.md) | The Rust feature runtime, and what bit-exactness actually cost | Accepted (amended 2026-07-27) |
| [0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md) | Watching a live session: the money, the clock, and the target nobody was working toward | Accepted (amended 2026-07-27 by [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)) |
| [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md) | A loss that halts, an exit that works, and the bar's own clock | Accepted |
| [0038](0038-many-strategies-one-account.md) | Many strategies, one account: netting, allocation, and the bounds that only exist across symbols | Accepted |

A gap in this sequence means a number was booked; a missing *row* for a file that exists means
somebody forgot this table.

## Keeping this table honest

This index is the one document that goes stale after every session, because the ADR it needs to
learn about is written by whoever was working elsewhere. Registering the row belongs to the same
change that adds the file — an ADR nobody can find is an ADR nobody reads, and the next author
then re-decides the question.

## When to write one

Write an ADR when a choice is (a) expensive to reverse, (b) shapes multiple components, or
(c) will make a future reader ask "why on earth did they do it this way?"

The Phase-0 list of "examples that will need ADRs later" — the shared-memory implementation, the
`Signal` schema, the async runtime, the money type, the dev target — has all been written. What
is still undecided and will need one:

- **strategy state persistence / warm restart**, which changes what a restart is allowed to
  assume. (Its other half, **portfolio-level risk across strategies and symbols**, is closed by
  [0038](0038-many-strategies-one-account.md) — and note what that leaves open rather than what
  it closes: the loss switch is still **per session**, so a book where one producer is losing and
  another is winning trips on the sum and nothing attributes the loss to a producer.)
- **how the two languages share a clock.** A `Signal`'s recorded `release_ts` is the core's
  event-time high-water mark at the pass that read it, not the producer's write time, so a replay
  judges a signal slightly fresher than the live session did. It looks like a schema gap and is
  not: Python has no access to the clock the core orders by, and a wall-clock stamp in the
  reserved bytes would close the gap in appearance while leaving the number incomparable.

**Closed since this list was last written:** **what a live session is watched by**
([0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md)) — a money view
that reports our accounting and the venue's side by side and never reconciles them, declared
latency budgets rather than by-product measurements, and the other half of ADR-0031's sweeper: a
bounded re-quote for a target the strategy still holds, and a named `UNQUOTED TARGET` when the
budget is spent. Its first live run found a defect in itself in fifteen seconds — the venue replays
a fill snapshot on subscribe, so a session that had placed no order opened with somebody else's
P&L — and the *obvious* fix, baselining at the first `clearinghouseState` reply, did not work
because arrival order is not a fact anyone controls.

Also closed: the **Rust feature runtime**
([0035](0035-rust-feature-runtime-and-the-bit-exact-gate.md)) — the Boundary-A half
[0021](0021-rust-model-parity-gate.md) could not gate, because nothing computed a feature in Rust
to compare against. `axon-features` is that implementation and it ships behind its own gate rather
than ahead of it: a feature bundle compares vectors *each language computed from the same market
data*, bit for bit. Read what it cost before writing a transform — NumPy does not sum a window left
to right, and the naive order is up to eight ULP away. Note what it does **not** close: nothing in
the live core calls the crate, so features are still computed in Python at Boundary B, and
promoting a strategy is a separate decision.

Also closed: bar and mark/funding records on the market-data ring
([0028](0028-market-data-bars-and-the-ticker-tail.md)) — the 48 reserved bytes are spent and the
missing candle kind is a second record, so shadow trading is no longer blocked on the contract.
And the **order lifetime** distinct from `ttl_ms`
([0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md)) — `max_order_age_ms` is on the wire,
the planner bounds ADR-0014 §6's leave-it-resting exception with it, and the half that did not
exist now does: a sweeper on the pass loop pulls a quote from a strategy that has gone silent, on
event time, with a cancel that is never risk-gated. **It has not been run at a venue.**

A note on what [0033](0033-lightgbm-crosses-by-conversion-not-by-backend.md) does *not* close.
[0019](0019-native-rust-inference-backends.md) leaves the native LightGBM backend unbuilt, and it
stays unbuilt — the artifact reaches Rust by conversion instead, so the gap was a routing question
rather than a missing crate. But the crossing is **objective-dependent**, and that is a new fact
rather than a closed one: `tract` 0.23.4 registers `TreeEnsembleClassifier` and not
`TreeEnsembleRegressor`, so a boosted-tree model crosses only if its converter emits a classifier.
Whether that ceiling is worth lifting — a `tract` operator, a tensor-only conversion path, or the
native backend after all — is undecided and will need its own record.

## What [0038](0038-many-strategies-one-account.md) closes, and what it deliberately leaves open

Phase 7's headline, and the piece the roadmap has carried since Phase 0. Until it, a session
had one signal ring, one producer and therefore one strategy on one instrument — and everything
downstream that *looked* multi-instrument (the per-symbol in-flight gate, the per-symbol held
target, the per-symbol sweep) had only ever been exercised at width one, because nothing
upstream could put two symbols on the wire.

Three things it decides. **One ring per producer**, because `seq` is per writer and two
producers interleaved on one ring is every record of the loser refused as `stale_seq`.
**Claims add** — the venue holds one position per instrument and a target-position signal is a
claim on part of it, so two strategies on one coin are two claims on one position and not two
positions; netting is opt-in anyway, because the overlap is far more often a copy-pasted config
than a decision. And **three bounds that no per-symbol limit can express**: gross notional, net
notional, and how many instruments may carry exposure at once.

Read the minus column before the plus one. A silent producer's exposure is **held** by default;
`intent.min_order_qty` is still one number for every instrument; the loss switch is still per
session; and one netted order carries one `urgency`, so the most impatient contributor decides
how the whole book is worked.

**And read what the measurement found**, because it is a negative result and it is in the
config: over 16 000 measured windows the net/gross ratio's median was **1.000**. The legs this
repo can populate today never offset — BTC, ETH and SOL move together and two of the three run
the same rule — so `max_net_notional` below `max_gross_notional` is not justifiable on any book
that exists here yet.

**None of it has run at a venue.**

## What [0037](0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md) closes, and the one thing it does not

Phase 6's own record ([0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md))
closed the *watching* box and left the acting on it undone. 0037 is that: a loss bound that
halts new exposure while every reducing order still gets out, an operator flatten that adopts
the venue's own position and ladders its urgency when a venue refuses a TIF, a dust floor
that no longer strands the position it was meant to protect, and the bar's own close on the
signal wire (schema **v3** — the `Signal` record is now fully named, so the next field has to
re-cut the layout rather than extend it).

It also makes the parity diff a component rather than a shape welded into the shadow
harness, which is what lets a session that is **trading** be watched at all — the gap
[0030](0030-live-parity-monitor-and-the-coverage-denominator.md) recorded and could not close,
for two reasons that turned out to be about *sessions* and not about parity.

**What it does not close is the soak.** Everything in 0037 is motivated by an outage that
happened inside an hour, and the longest session that has ever *traded* is 59 minutes. The
loss switch, the flatten ladder and the live parity alarm are each tested offline against the
run that motivated them and **none has fired at a venue.** That is hours of venue time rather
than code, and it is the largest untested thing in the system.
