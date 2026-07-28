# ADR-0002 — Python↔Rust boundary = shared memory (Boundary B, with a path to A)

**Status:** Accepted · **Date:** 2026-07-12

## Context

The core question of the project: where does Python stop and Rust start, and how do they hand
off? Strategies and ML must stay in Python (research velocity, ecosystem, and — critically —
*not rewriting the strategy* so it stays the same strategy). Execution must be fast,
deterministic, and jitter-controlled (Rust). Three options were analysed
([research/python-rust-ipc.md](../research/python-rust-ipc.md)):

- **A — Rust runs everything.** Models exported (ONNX/native), Python offline-only. Tightest,
  most deterministic tail; but requires re-implementing feature computation in Rust and
  loses live Python — high effort and high "is it still the same strategy?" risk.
- **B — Python generates signals in-process, hands to Rust via shared memory.** Python does
  features + inference off the hot path; Rust reads a `Signal` from a shared-memory ring and
  owns execution/networking. Full Python ML fidelity; ~100 ns–1 µs hop.
- **C — Rust embeds Python via PyO3 for inference.** Tiny per-call cost (~20–70 ns) but a hot
  loop **serializes on the GIL**, and it couples the Python runtime into the Rust binary.

Decisive fact: the Hyperliquid exchange round-trip is **milliseconds** (~0.2 s block commit),
i.e. **5–7 orders of magnitude larger** than any Python↔Rust hop. So the IPC mechanism is
*not* what determines execution speed — the choice should protect **strategy fidelity** and
**operational simplicity**.

## Decision

Adopt **Boundary B**: two processes (Python research plane, Rust execution plane) communicating
over **lock-free SPSC shared-memory rings**. Python computes features + runs inference and
writes a fixed-layout `Signal`; Rust reads it and owns order construction, risk, and venue
networking.

Keep **Boundary A as the documented migration target** for individual latency-critical
strategies: once a strategy's Rust feature+inference path is proven bit-equivalent
([ADR-0003](0003-model-serving-and-fidelity.md)), that strategy can move inference into Rust
(`axon-model`) without changing the architecture. **Reject Boundary C** for v1 (GIL
serialization + runtime coupling); keep PyO3 available for offline tooling.

## Consequences

- **+** Full, unchanged Python ML stack stays live → the strongest guarantee that the live
  strategy equals the researched one.
- **+** Rust owns exactly the part that benefits (execution, networking, book, risk, jitter).
- **+** IPC hop (~100 ns–1 µs) is negligible vs the ms wire → simplicity is nearly free.
- **+** Clean, incremental path to Boundary A per strategy — optimization, not rewrite.
- **−** Two processes to run, monitor, and crash-recover; we own ring synchronization, cache
  coherency, and peer-death detection (mitigated by heartbeats + Hyperliquid `schedule_cancel`
  dead-man's-switch).
- **−** A fixed-layout, versioned `Signal`/record contract must be maintained for both languages
  (lives in [`contracts/`](../../contracts/README.md)).
- **OPEN:** shared-memory implementation (hand-rolled SPSC vs `shmem-ipc` vs `iceoryx2`) and the
  Linux/Windows dev story — decided in Phase 1.

See [02 — Python↔Rust boundary](../02-python-rust-boundary.md) for the design detail.
