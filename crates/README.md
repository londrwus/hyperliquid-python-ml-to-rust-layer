# `crates/` — the Rust execution plane

**Phases 2–5 landed in code; every crate carries real logic and all but two have a caller inside
the workspace.** All 14 compile as one workspace with 1 136 Rust tests green (14 `#[ignore]`d —
they are the live-venue ones) and `clippy -D warnings` clean. Nothing in the default gate touches
the network: live WS, live execution and the live signal path sit behind `#[ignore]`d tests and
explicit config switches.

The exceptions are `axon-model` and `axon-features`, and they are worth stating rather than
glossing: **no other crate depends on either.** They are the two halves of the Boundary-A future —
inference and features — and until a Rust strategy needs one, the only thing pointing at each is
its own cross-language parity gate. Those gates are genuine consumers of the *artifact pipeline*
(Python writes a frozen question, Rust answers it over the same bytes) and not workspace edges. A
crate nothing imports is a crate that can rot; these two are held in place by gates that fail, not
by dependencies.

Read that carefully rather than as a formality: **the live core computes no features in Rust and
serves no model in Rust.** Both capabilities are proven and neither is in use. Promoting a
strategy to Boundary A is a per-strategy decision with its own evidence to produce
([ADR-0035](../docs/adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)).

| Crate | What is actually in it today |
|-------|------------------------------|
| `axon-contracts` | All three wire records (`Signal` at schema v3, `MdSlice` at v2, `MdBar`) generated from `contracts/schema.toml`, with offsets asserted at **compile time**. None has a `reserved` block left, so the next field re-cuts a layout |
| `axon-core` | Domain types, `market` + `exec` vocabularies (incl. `Ticker`/`Funding`), the bus, the clock, `TimedQueue`, the deterministic loop |
| `axon-ipc` | `RingProducer<R>`/`RingConsumer<R>` — the SPSC mmap rings, generic over the record type |
| `axon-marketdata` | L2 book + `MarketDataProcessor` (books, BBO/trade/candle/ticker caches, and a per-symbol `book_ts` — the global `last_ts` cannot see one instrument's book freeze) |
| `axon-risk` | Pre-trade size checks on the hot path, plus `reduces_exposure` — the one definition of "does this order take exposure off", shared by the reduce-only rule and the loss switch — and `portfolio`, the bounds that only exist **across** symbols: gross notional, net notional and how many instruments may carry exposure at once ([ADR-0038](../docs/adr/0038-many-strategies-one-account.md)) |
| `axon-execution` | `OrderTracker` (reconciliation, and `adopt_position` for the operator flatten), `MarkCache` (expiring, venue-mark-first), the `Guarded`/`Governed`/`Haltable` client stack, and `LossLimiter` — the one gate here about **money** rather than size, which puts a session past its declared loss into de-risk-only rather than halting it ([ADR-0037](../docs/adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)) |
| `axon-replay` | Event capture to versioned JSONL, a `SignalLog` beside it, and deterministic golden replay through the **production** handler chain (a dev-dependency on `axon-runtime`, so the live edge can still run runtime→replay) |
| `axon-model` | Native FP32 inference: `tract` ONNX + a bit-exact XGBoost JSON reader, FP16 refused — plus `parity::ParityBundle`, the Rust half of the cross-language model gate, which needs no Python and no ML libraries |
| `axon-features` | The other half of Boundary A: seventeen transforms in NumPy's own summation order, the versioned `FeatureSpec` re-identified in Rust (the fingerprint is recomputed, never read), a bounded `FeatureStream` that refuses a spec it cannot serve, and `parity::FeatureBundle` — the feature gate, held to **bit** equality ([ADR-0035](../docs/adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)) |
| `axon-strategy` | `SignalReader` (validate) + `Planner` (`Signal` → order intents) + `TargetBook` — what several producers together want the account to hold, netted per instrument, with a single contributor passing through byte for byte so every session that has ever run plans exactly what it planned before ([ADR-0038](../docs/adr/0038-many-strategies-one-account.md)) |
| `axon-providers` | The outward port: traits, capabilities, error taxonomy |
| `axon-provider-binance` | The second venue adapter: `exchangeInfo` symbol/instrument resolution, the market-data decoders (partial-depth book, `aggTrade`, `bookTicker`, `markPriceUpdate`, `kline`), the `MarketData` port with reconnect, and order/cancel query-string encoding. **Offline-verified; it has never traded** ([ADR-0023](../docs/adr/0023-second-venue-adapter-binance.md)) |
| `axon-provider-hyperliquid` | `SymbolMap`, WS decoders + client, the signing core, `ExchangeClient`, `/info` reads (all bounded by an explicit request timeout), the rate `governor`, the `approve_agent` ceremony, and a funding-cadence probe that checks the interval we stamp against the one the venue actually funds at |
| `axon-runtime` | The supervised session, and now the only crate that closes the loop: bus + core thread, intent source, MD-ring publisher, capture tap, DMS loop, reconciliation poll, shutdown, health |

`axon-runtime` is where this session's work concentrated, so its modules, one line each:

| Module | What it is |
|--------|-----------|
| `session` | Composes a session — offline / sandbox / live — and owns the tokio edge |
| `core` | The core thread: the loop, the handler, and the control handle the edge reads state through |
| `handler` | `CoreHandler` — book → marks → tracker → publishers, in ADR-0013 §1's order |
| `intent` | `IntentSource`: N readers → the target book → allocation → planner, inside the core iteration, paced on **event time**, and the polling submit pump on the edge |
| `quote` | `top_of_book` — the one answer to "what is the top of book", shared by the planner and the publisher |
| `mdring` | `MdPublisher`: the core's event stream → `MdSlice`s, `on_change` or `every_update`, never a wall clock |
| `capture` | `SessionRecorder` + `CaptureTap`: a non-blocking hand-off to a writer thread; stops rather than drops |
| `dms` | The re-arming dead-man's switch and its graded escalation |
| `reconcile` | The `/info` poll, because `orderUpdates` never snapshots |
| `shutdown` | The ordered teardown, and what it leaves armed when it could not stop the submitter |
| `pnl` | The money view: **two** independent answers side by side (ours, the venue's) and a `drift` that is never used to correct either |
| `latency` | Four declared budgets — `bar` (the observation → the decision), `sig`, `ack`, `e2e` — measured whether or not a ceiling was declared |
| `daybook` | The one piece of session state that outlives the process: the UTC day's equity baseline, so a daily loss bound survives a crash-restart loop |
| `flatten` | `axon --flatten`: adopt the venue's own position and drive it to zero, every order a reduce-only close sized from a fresh read, urgency laddered when a venue refuses a TIF. **Not a session** — no socket, no DMS, no ring, no `cancel_all` |
| `health` | `StatusSnapshot` and `warnings()` — the line that distinguishes a dead producer from a quiet strategy |
| `config` | The TOML schema, and validation that refuses combinations which would trade nothing while reading `OK` |
| `selftest` | The canned event *and signal* stream the offline session replays |

See [architecture](../docs/01-architecture.md), the [roadmap](../docs/08-roadmap.md), and the
[ADRs](../docs/adr/README.md) — 0008/0009/0010 for the bus, signing and reconciliation, then
0011–0014 and 0018–0021 for what landed here most recently. [ADR-0020](../docs/adr/0020-runtime-intent-source.md)
is the one to read first if you are wondering how Python reaches the venue.

Design rules for everything here:
- **Single-threaded deterministic core** for trading logic; **async (Tokio) only at the edges**.
- **Message passing, not shared mutable state.**
- **Fixed-point money math** (`rust_decimal` / integer ticks), never `f64` for px/sz.
- **No allocations on the hot path.**

## Planned workspace

Kept **as written in Phase 0**, for comparison rather than as a description — the table above is
what is actually here, and nothing below is edited when a crate lands.

Two lines have since been overtaken. `axon-model`'s "+ feature runtime" is now a separate crate,
`axon-features`, because a feature runtime has nothing to do with inference and a strategy
computing features in Rust need not be serving a model at all — which closes the gap
[ADR-0021](../docs/adr/0021-rust-model-parity-gate.md) named in its own minus column
([ADR-0035](../docs/adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)). And the plan has
one provider where there are now two ([ADR-0023](../docs/adr/0023-second-venue-adapter-binance.md)).

```
crates/
├── axon-core/                 Domain types, message bus, clock, cache, deterministic loop.
│                              Knows nothing about venues or strategies.
├── axon-marketdata/           Order-book maintenance; normalized tick/bar/book events.
├── axon-execution/            Order manager (lifecycle, reconciliation) + execution engine.
├── axon-replay/               Event capture + deterministic golden replay (the bottom rung
│                              of the parity ladder in docs/07).
├── axon-risk/                 Pre-trade risk checks (hot path): limits, reduce-only, exposure.
├── axon-ipc/                  Shared-memory signal bus (Python↔Rust): SPSC ring, fixed records.
├── axon-providers/            Provider traits + registry + capability descriptors (outward port).
│   └── (trait defs: ExecutionClient, MarketData, AccountState, Signer)
├── axon-provider-hyperliquid/ Hyperliquid adapter: signing (2 schemes), nonces, encoding, WS/REST.
├── axon-model/                (Boundary-A) native inference (tree/ONNX/torch) + feature runtime.
├── axon-strategy/             Rust-side strategy plugin contract (inward port, for Rust strategies).
└── axon-runtime/              The binary: wires crates, loads config, process lifecycle,
                               dead-man's-switch, health/metrics.
```

## Dependency direction (rough)

```
axon-runtime ─▶ axon-strategy ─▶ axon-execution ─▶ axon-risk ─▶ axon-providers ◀─ axon-provider-hyperliquid
      │              │                 │                              ▲
      └──────────────┴─────────────────┴──────────▶ axon-core ◀───────┘
                                        axon-marketdata ─▶ axon-core
                                        axon-ipc ─▶ axon-core         axon-model ─▶ axon-core
```
`axon-core` depends on nothing internal. Adapters depend on `axon-providers` (the traits),
never the other way around.

One edge in that graph runs both ways, deliberately. `axon-runtime` depends on `axon-replay` for
real (a live session records itself), and `axon-replay` depends on `axon-runtime` as a
**dev**-dependency only (a replay drives the production handler chain rather than a copy of it).
Cargo permits that cycle only through a dev edge, so keeping the runtime out of `axon-replay`'s
`src/` is what keeps the recording direction legal. The shared driver lives in
`axon-replay/examples/chain/`, used verbatim by the `replay_log` binary and by the golden test.

## Candidate crates (evaluate in Phase 1+)
- Async/runtime: `tokio`. WS: `tokio-tungstenite` (or `fastwebsockets`). JSON: `simd-json`/`serde`.
- Money: `rust_decimal`. Alloc: `mimalloc`. Cache padding: `crossbeam-utils`.
- Shmem: `shmem-ipc` / `iceoryx2` / hand-rolled over `raw_sync` (see [ADR-0002](../docs/adr/0002-python-rust-boundary.md)).
- Inference: `ort` / `tract` / `tch` / `xgboost-ars` / `lightgbm3` (see [ADR-0003](../docs/adr/0003-model-serving-and-fidelity.md)).
- Signing (Hyperliquid): wrap a vetted SDK; watch `ethers`→`alloy` migration.
