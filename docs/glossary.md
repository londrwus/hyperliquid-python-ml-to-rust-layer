# Glossary

Terms of art used across the Axon docs.

## Project terms

- **Axon** — the working codename for this layer. Metaphor: an axon carries a signal from the
  neuron ("brain" = Python ML) to the muscle ("muscle" = Rust execution).
- **The Layer** — synonym for Axon: a strategy-agnostic + provider-agnostic execution layer.
- **Research plane** — the Python side: training, backtesting, features, signal generation.
- **Execution plane** — the Rust side: market data, book, risk, order lifecycle, venue I/O.
- **Boundary A / B / C** — the three Python↔Rust split options ([ADR-0002](adr/0002-python-rust-boundary.md)).
  **A** = Rust runs everything; **B** = Python signals → shmem → Rust (chosen); **C** = Rust
  embeds Python via PyO3.
- **Signal** — the fixed-layout record a strategy emits across the boundary (target position
  or explicit order intent). See [06](06-strategy-contract.md).
- **Inward / outward port** — hexagonal-architecture terms. Inward = strategy contract;
  outward = provider (venue) contract.

## Architecture / systems

- **Hexagonal (ports & adapters)** — architecture where a core is isolated behind ports;
  external systems (venues, data, strategies) are interchangeable adapters.
- **SPSC ring buffer** — single-producer, single-consumer lock-free queue; our shared-memory
  transport.
- **Shared memory (shmem)** — memory mapped into two processes for zero-syscall data transfer
  (`/dev/shm` on Linux; memory-mapped file on Windows).
- **PyO3 / maturin** — Rust↔Python binding library / its build+packaging tool.
- **Message bus** — the pub/sub + command/event backbone of the Rust core; components
  communicate through it, not by shared mutable state.
- **Deterministic (single-threaded) core** — trading logic on one thread keyed on event time,
  giving reproducible ordering (LMAX Disruptor inspiration).
- **False sharing** — two hot variables on the same cache line causing coherency traffic;
  avoided with cache-line padding.
- **GIL** — CPython's Global Interpreter Lock; serializes Python execution, which is why we
  keep Python off the hot loop.
- **Free-threaded Python** — no-GIL CPython (PEP 779, official in 3.14); promising but not
  prod-ready in 2026.

## ML / fidelity

- **Fidelity** — how closely Rust inference reproduces the Python model's output.
- **FP32 / FP16 / INT8** — floating-point-32 / -16 / integer-8 numeric precisions.
- **Quantization** — reducing model precision (e.g. FP32→INT8) for speed; **rejected here**
  because it changes predictions ([ADR-0005](adr/0005-fp32-no-quantization.md)).
- **ONNX** — Open Neural Network Exchange; a portable model format used as the Python→Rust
  bridge for NN/sklearn models.
- **Training–serving skew** — when features (or their freshness) differ between training and
  serving, silently degrading predictions. The #1 quality-leak risk.
- **Feature parity** — the property that online (serving) features equal offline (research)
  features. See [03](03-ml-fidelity-and-features.md).
- **Decision invariance** — the acceptance criterion that a small numeric delta must not flip
  any discretized trading decision vs. the reference.
- **Golden / replay test** — replaying a captured event log through the exact production code
  path and asserting outputs match a stored reference.
- **Shadow trading** — running the live strategy on live data without sending orders, diffing
  its signals against the reference.
- **Parity monitor** — a production process that continuously checks live vs. offline
  features/signals and alarms on drift.
- **Point-in-time-correct** — feature assembly that uses only data available at the event's
  timestamp (no lookahead leakage).

## Inference backends (Rust)

- **ort** — Rust bindings to Microsoft's ONNX Runtime (fastest general option).
- **tract** — pure-Rust ONNX/NNEF inference (portable, deterministic, no C++ deps).
- **tch-rs** — Rust bindings to libtorch (highest NN fidelity; heavy deploy).
- **candle** — HuggingFace pure-Rust ML (transformers/DL).
- **burn** — Rust DL framework; imports ONNX to native Rust (`burn-onnx`).
- **xgboost-ars / lightgbm3-rs** — native Rust tree-model inference (near-exact fidelity).

## Hyperliquid / trading

- **HyperCore / HyperBFT** — Hyperliquid's native on-chain matching engine / its consensus.
- **Agent (API) wallet** — a delegated key approved to sign L1 actions so the funded key
  stays cold.
- **Phantom-agent / EIP-712** — Hyperliquid's two signing schemes (L1 actions vs user-signed
  admin actions). Mixing them is the #1 "invalid signature" cause.
- **Nonce window** — Hyperliquid tracks the 100 highest nonces per address; nonces are ms
  timestamps, not sequential counters.
- **In-block priority** — Hyperliquid's per-block ordering: cancel → post-only → GTC → IOC.
- **TIF** — time-in-force: GTC (good-till-cancel), IOC (immediate-or-cancel), ALO/PostOnly,
  FOK (fill-or-kill).
- **`cloid`** — client order id (128-bit); makes orders idempotent and reconcilable.
- **`schedule_cancel`** — Hyperliquid's dead-man's-switch: auto-cancel/flatten if the process
  stops heartbeating.
- **reduce-only** — an order flag that can only shrink, never flip/grow, a position.
- **Colocation** — running near the venue's servers (Hyperliquid validators: AWS Tokyo
  `ap-northeast-1`) to cut network latency.
