# 02 — The Python↔Rust Boundary

This is the heart of the layer. Everything hinges on *where* Python stops and Rust starts,
and *how* they hand off.

## The three options (and why we chose B)

| | Latency | Fidelity (keeps live Python ML) | Complexity | Verdict |
|---|---|---|---|---|
| **A. Rust runs everything** | Best, most deterministic tail | Lowest — no live Python; models exported to ONNX/native | Highest (port features + serve models in Rust) | **Later**, for the hot core |
| **B. Python signals → shmem → Rust executes** | Excellent — Python off the hot path; ~100ns–1µs hop | **High** — full Python ML stack stays live | Medium — two processes + a ring layout | **✅ Chosen** |
| **C. Rust embeds Python (PyO3) for inference** | Good call (~20–70ns) but the **GIL serializes the loop** | Highest — Python inside the Rust process | Medium/High — fight the GIL, couple runtimes | Rejected for v1 |

Full analysis and rationale: [ADR-0002](adr/0002-python-rust-boundary.md).

### Why B, in one breath

The exchange round-trip is **milliseconds**; the Python→Rust hop is **~100ns–1µs** — five to
seven orders of magnitude smaller. So paying a shared-memory hop to keep the **entire Python
ML ecosystem live and unchanged** is nearly free, and it's the design that best protects
strategy quality. Rust still owns the part that actually needs it (execution, networking,
book, risk, jitter).

### Why not C (embedding Python via PyO3)

The per-call overhead is tiny (~20–70ns), but a hot loop that repeatedly calls into Python
**serializes on the GIL** — you get no inference parallelism and you couple the Python
runtime into the Rust binary. Free-threaded (no-GIL) Python exists (PEP 779, official in
3.14) but the ecosystem consensus in 2026 is *test, don't deploy to prod yet*. We keep PyO3
in our back pocket, not on the hot path.

### Why A is the *destination*, not the *start*

Boundary A gives the tightest, most deterministic tail — but it requires exporting every
model to ONNX/native **and** re-implementing feature computation in Rust, which is exactly
the "is it still the same strategy?" risk we're trying to avoid. We earn our way there
per-strategy, once the Rust feature/inference path is proven bit-equivalent (see
[03](03-ml-fidelity-and-features.md)). The architecture makes this a swap, not a rewrite.

## The shared-memory design

Two independent, one-directional, lock-free **SPSC ring buffers** over shared memory
(`/dev/shm` on Linux; the Windows equivalent is a memory-mapped file):

```
  Python (Strategy)                         Rust (axon-ipc → strategy adapter)
  ┌───────────────┐   signal ring (Py→Rust)   ┌───────────────────────────────┐
  │ features +    │ ─────────────────────────▶ │ read signal → order intents   │
  │ inference     │                            │                               │
  │               │ ◀───────────────────────── │ publish market-data slice     │
  └───────────────┘   md ring (Rust→Py, opt.)  └───────────────────────────────┘
```

- **Signal ring (Py→Rust):** the strategy writes a fixed-layout `Signal` record per
  decision. Rust reads it, converts to order intents.
- **Market-data ring (Rust→Py, optional):** if the strategy needs live market data for
  feature computation, Rust publishes the needed slice. (Alternatively Python subscribes to
  the venue directly for data — a config choice; see `OPEN` below.)

### Design rules (from the IPC research)

- **One writer, one reader per ring (SPSC).** Lock-free, wait-free steady state, zero
  syscalls in the data path — just a memcpy.
- **Fixed-layout, versioned records.** No dynamic serialization on the hot path. A
  `schema_version` byte guards Python/Rust drift. The exact record layout lives in
  [`contracts/`](../contracts/README.md) as the single source of truth for *both* sides.
- **Cache-line padding.** Put the head and tail indices on separate 64-byte cache lines
  (`crossbeam_utils::CachePadded` on the Rust side) — sharing them cost a measured ~3×
  slowdown in benchmarks (false sharing).
- **Batch on the Python side.** Python's interpreter floor is ~50–350ns *per call*; poll the
  ring and read/write a batch per call, not one message per call.
- **Crash recovery is our job.** With shared memory we own synchronization and recovery. A
  sequence number + heartbeat lets each side detect a dead or stalled peer; Rust arms
  Hyperliquid's `schedule_cancel` dead-man's-switch so a Python crash flattens positions.

### Candidate implementations (decide in Phase 1)

- **Hand-rolled cache-padded SPSC ring** over `shared_memory`/`raw_sync` (Rust) +
  `multiprocessing.shared_memory` (Python) — minimum dependencies, maximum control.
- **`shmem-ipc`** (Rust) — purpose-built wait-free SPSC over shared memory with `eventfd`
  signaling; benchmarked ~3× faster than Unix sockets. *(Linux-only.)*
- **`iceoryx2`** — structured zero-copy pub/sub with **Python bindings**; great if we want
  multi-consumer fan-out (data → strategy + risk + logger) without hand-rolling. *Pre-1.0
  (v1.0 targeted end-2026); pin the version and benchmark.*
- **Apache Arrow C Data Interface / PyCapsule** — genuinely zero-copy for *columnar batches*
  of features **within one process**; consider it only if a Rust feature engine ever shares
  a process with Python. Not for per-tick messages.

> `OPEN:` cross-platform story. Primary target is Linux (`/dev/shm`, `shmem-ipc`,
> `iceoryx2`). Development happens on Windows — decide whether to (a) develop against a
> memory-mapped-file ring that works on both, or (b) develop in WSL2/Docker Linux. Leaning
> (b) for prod-parity, with a portable mmap fallback for local dev.

## What crosses the boundary — and what must not

**Crosses (Py→Rust), per decision:** a `Signal` = normalized order intent. Options for its
shape (finalize in [06](06-strategy-contract.md) + `contracts/`):
- **Target position** (`symbol, target_qty, urgency, price_band`) — Rust owns *how* to reach
  it (the execution algo). *Preferred* — keeps execution logic in Rust, robust to missed
  signals.
- **Explicit order intent** (`symbol, side, qty, type, tif, price?, reduce_only, cloid`) —
  Python decides the order; Rust just routes/risks/sends. More control, more coupling.

**Must not cross the hot boundary:** pandas DataFrames, Python objects, anything requiring
pickling/JSON on the per-tick path. Rich objects belong to the offline handoff (model
artifacts, config), not the ring.

## The offline handoff (Python→Rust, not on the hot path)

Separate from the ring, and just as important:
- **Model artifact:** exported model (ONNX for NN/sklearn; native JSON for XGBoost/LightGBM)
  + its version + input/output schema.
- **Feature spec:** the versioned description of what features feed the model, so the parity
  harness (and, later, a Rust feature runtime) can reproduce them.
- **Config:** strategy parameters, symbol universe, risk limits — a serializable object (see
  [06](06-strategy-contract.md)).

These are produced by `axon.models` / `axon.features` and consumed at Rust startup. See
[03](03-ml-fidelity-and-features.md).
