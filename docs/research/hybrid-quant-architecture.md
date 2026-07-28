# Research — Hybrid quant architecture: Python research / Rust execution (2026)

**Organizing principle:** parity comes from a **single shared event-driven codepath, not two
parallel implementations.** Vectorized-Python-backtest + separate-live-engine is the classic
anti-pattern that guarantees divergence. Modern designs push environment differences (live
venue vs simulator, live feed vs replay) to the *edges* (adapters) and keep one core.

## 1. The research–production parity problem
Parity breaks at four layers; solve each separately:

| Layer | Failure mode | Technique |
|---|---|---|
| Model | Python model ≠ prod inference | Export to **ONNX**, run the bit-identical artifact in Rust (`ort`/`tract`) |
| Features | Offline ≠ online ("training–serving skew") | Compute features **once from the same event stream**; materialize both stores from that one transform; declare defs once |
| Execution semantics | Batch/vectorized backtest ≠ event-driven live | **Event loop in both**; identical matching/fill/time model |
| Numerics | Float/precision divergence | Golden/replay with epsilon tolerances; watch fixed-vs-float + platform width |

Transferable insights:
- **Event-time, not processing-time.** Features and replay must key off the event's own
  timestamp or windows corrupt and replay isn't deterministic. #1 subtle parity killer for
  time-series.
- Feature stores guarantee *schema* parity but **not freshness parity** (DoorDash measured up
  to 35.7% feature-value mismatch from staleness).
- **Numerical precision is a real cross-language trap:** NautilusTrader ships standard
  precision on Windows (MSVC lacks `__int128`) vs high precision elsewhere — same logic,
  different numbers. Pin precision and test it.
- **Validation ladder:** golden/replay (deterministic replay of a captured log through the
  exact prod path, compared within tolerance) → paper/sandbox → **shadow trading** (prod
  strategy on live data, no orders, diff signals) → live.

## 2. Where to split the boundary
Split on **latency-criticality and determinism**, not convenience.
- **Must be fast/compiled (Rust):** market-data ingestion & parsing, order-book maintenance,
  order management/execution & routing, **pre-trade risk (hot path)**, serialization, the
  event bus, the deterministic core loop.
- **Can tolerate Python:** model training & research, backtesting orchestration/sweeps,
  slow/low-frequency signal generation, config/monitoring/orchestration (control plane).
- **Pattern that makes it clean:** **hexagonal / ports-and-adapters.** The core never knows if
  it's talking to a simulated or real venue, historical or live data — they're interchangeable
  adapters. That's what lets one strategy binary run in backtest and prod unchanged.
- Caveat: the dual-language split's real cost is **build complexity + higher skill bar**. If
  you don't need low latency, pure Python may be right — don't take FFI complexity for a slow
  strategy.

## 3. Event-driven execution engine design in Rust
Convergent design across production Rust engines:
1. **Single-threaded deterministic core** for trading logic — no lock contention, reproducible
   event ordering (essential for parity & golden tests). LMAX Disruptor ring-buffer inspiration.
2. **Actor / message-passing** — components consume messages off a pub/sub **message bus**; no
   shared mutable state.
3. **Async runtime (Tokio) + separate threads only at the edges** — network I/O, persistence,
   adapters run off-core and hand results back via channels. Keep the hot loop off async.
4. **Kernel-bypass / fast I/O for the extreme end** — `io_uring`, `AF_XDP`, DPDK; zero-copy +
   SIMD parsing; fixed-size arrays for cache-friendly books; `mimalloc`.

Recurring cautionary tale: teams reach for `Mutex<Engine>`, then reads queue behind writes.
Fix = message-passing (`WebSocket → MPSC → single processor`) — *why* single-threaded-core +
actor model dominates.

Reference engines: **NautilusTrader** (canonical Python+Rust; single-threaded LMAX-style core,
message bus, hexagonal adapters; LGPLv3 — check licensing), **barter-rs** (modular pure-Rust,
trait-based `Strategy`+`RiskManager`; MIT, "educational"), **hftbacktest** (L2/L3 tick +
queue-position/latency modeling; same code backtest & live), **Botvana** (thread-per-core,
`io_uring`), **AKQuant** (Rust core + PyO3, zero-copy NumPy views).

## 4. Strategy-as-plugin ("a Layer")
Two orthogonal contracts:
- **Strategy contract (inward port):** strategies implement fixed event callbacks (`on_start`,
  `on_stop`, `on_bar`, `on_quote_tick`, `on_trade_tick`, `on_order_book_delta`, `on_save`/
  `on_load`). The engine calls you; you emit through provided facades (`order_factory`,
  `submit_order`, `cache`, `clock`) — never reach into internals.
- **Provider contract (outward port):** venues/feeds are adapters behind one interface; swap
  live ↔ simulator with zero strategy change.

Decisions worth copying:
- **Config-as-serializable-object** (Nautilus `StrategyConfig` serializes over the wire) →
  enables distributed backtests + remote live from one definition.
- **State hooks** (`on_save`/`on_load`) for warm restart.
- **Cross-language plugin surface:** Nautilus lets strategies be authored in **Python or Rust**
  against the *same* contract (Rust via a `nautilus_strategy!` macro; Python subclass). "Same
  contract, two languages" is exactly the layer property we want.

## 5. NautilusTrader — what to steal
- **Rust core** (`crates/`): `NautilusKernel` (orchestrator), `MessageBus`, `Cache`,
  `DataEngine`, `ExecutionEngine` (lifecycle/routing/reconciliation), `RiskEngine` (pre-trade,
  hot path). Single-threaded deterministic loop, `mimalloc`, Tokio only for I/O.
- **Python control plane:** strategy logic, config, orchestration, ML.
- **Binding layer (2026 inflection):** migrating Cython (v1) → **PyO3 (v2)**; Rust libs
  statically linked, shipped as binary wheels → end users need no Rust/Cython toolchain (a good
  distribution model to emulate).
- **Transferable lessons:** one kernel, three environments (backtest/sandbox/live) sharing the
  exact component impls → parity by construction; everything through the message bus →
  decoupled/deterministic/replayable; hexagonal adapters = provider-agnostic; actor/strategy
  contract = strategy-agnostic; pin & test numerics.

## One-line takeaway
Build a single Rust event-driven core (single-threaded deterministic loop + message bus),
expose an *inward* strategy-callback contract (Python via shmem now, Rust later — same
semantics) and an *outward* venue/data adapter contract (hexagonal), ship models as ONNX/native
and features from one stream, and gate every deploy through replay-golden-tests → shadow
trading → live. NautilusTrader is the closest turnkey embodiment; `ort`/`tract` cover the ONNX
leg it doesn't package.

## Sources
nautilustrader.io/docs/latest/concepts/{architecture,strategies} ·
github.com/nautechsystems/nautilus_trader · github.com/barter-rs/barter-rs ·
crates.io/crates/hftbacktest · github.com/featherenvy/botvana · akquant.akfamily.xyz ·
github.com/pykeio/ort · lib.rs/crates/tract-transformers ·
confluent.io/blog/eliminate-training-serving-skew-mlops ·
dev.to/synapcores/why-feature-stores-didnt-fix-training-serving-skew-fad ·
martinfowler.com/articles/lmax.html · arxiv.org/pdf/2309.04259
