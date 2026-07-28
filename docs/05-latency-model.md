# 05 — Latency & Performance Model

The purpose of this doc is to keep us honest about **where time actually goes**, so we
optimize the right things and don't cargo-cult "Rust is fast."

## The latency ladder (small messages, same host, Linux)

| Mechanism | One-way latency | Cross-process? |
|---|---|---|
| Pure-Rust in-process call | ~3 ns (0 if inlined) | no |
| PyO3 boundary crossing | ~20–70 ns | no (same process) |
| Intra-process SPSC ring (Rust) | ~4–66 ns | no |
| **Shared-memory SPSC ring (mmap)** | **~66–135 ns** | **yes** |
| iceoryx2 zero-copy pub/sub | ~240 ns (graph) / ~600–700 ns (measured) | yes |
| POSIX message queue | ~2.7 µs | yes |
| Unix domain socket | ~0.7–4.5 µs | yes |
| TCP loopback | ~1.8–7 µs | yes |
| ZeroMQ (inproc / tcp-loopback) | ~8 µs / ~26 µs | yes |
| gRPC + protobuf (unary) | ~100–300 µs | yes |
| **Hyperliquid order round-trip** | **~200,000–900,000 ns (0.2–0.9 s)** | network |

Read the last row against the others. **The exchange is 5–7 orders of magnitude slower than
our IPC choice.** This single fact governs the whole design.

## Where time actually goes (Boundary B, one decision)

```
  market event → [Rust ingest+book: µs] → [to Python: ~100ns-1µs shmem]
    → [Python features+inference: 10µs–low ms] → [signal to Rust: ~100ns-1µs shmem]
    → [Rust risk+order build: µs] → [venue round-trip: 200–900 ms]  ◀── dominates
```

**Implications:**
1. **Optimizing the IPC hop from 700ns to 100ns is irrelevant to fill speed.** It matters
   only for internal jitter/fan-out. Don't over-engineer it in v1.
2. **Python inference time (10µs–low ms) is fine** as long as it's comfortably below the
   signal cadence and doesn't add tail jitter to the *next* market event's handling — which
   it won't, because it's off the Rust hot loop (that's the point of Boundary B).
3. **The wins are determinism and priority, not raw speed.** A predictable p99 and fast
   cancel/replace beat a fast median. See below.

## What we optimize (and what we don't) — v1

**Do:**
- **Jitter/tail control** in the Rust core: single-threaded deterministic loop, no
  allocations on the hot path, `mimalloc`, bounded channels, cache-friendly book layout.
- **Fast, correct market-data ingestion:** zero-copy / SIMD JSON (`simd-json`), fixed-point
  (`rust_decimal`, *not* `f64`) for px/sz, resilient reconnection.
- **Cancel/replace latency:** exploit Hyperliquid's in-block priority (cancel > post-only >
  GTC > IOC).
- **Batch orders/cancels** on a tick (Hyperliquid: up to 20/call) and keep a **cancel budget
  in reserve** (cancels get a higher rate-limit cap — you can always de-risk).

**Don't (yet):**
- Colocation, kernel bypass (`io_uring`/`AF_XDP`/DPDK), FPGA, thread-per-core pinning.
- Squeezing nanoseconds out of the IPC ring.
- Moving inference into Rust just for speed.

## HFT headroom — the migration path (if/when mid-freq → HFT)

The architecture reaches HFT by *optimization*, not *rewrite*. In rough priority order:

1. **Colocate in Tokyo (AWS `ap-northeast-1`).** This is the single biggest lever on
   Hyperliquid: a Tokyo box reaches validators in **2–3 ms** vs **>200 ms** from Europe/US.
   On a time-priority book that ~200 ms geographic gap is a *measured, real* edge and no
   speed-bump equalizes it. **Most of the "HFT" win is geography, not language.**
2. **Move the execution core to Boundary A** for the latency-critical strategies: export the
   model (ONNX/native), run inference + features in `axon-model` (Rust), remove Python from
   that strategy's live loop. Earn it per-strategy via the parity gate ([03](03-ml-fidelity-and-features.md)).
3. **Kernel-bypass I/O** (`io_uring`, `AF_XDP`) and **thread-per-core** pinning in the core.
4. **Private/high-throughput API gateway** (e.g. Tokyo-peered) for higher rate limits.

Note the ordering: **(1) colocation dwarfs (2) language.** Going Boundary-A without a Tokyo
box is optimizing the 1µs while ignoring the 200ms.

## Latency budget template (fill in per strategy/deployment)

| Segment | Target (mid-freq) | Target (HFT) |
|---|---|---|
| Market-data ingest → book updated | < 50 µs | < 10 µs |
| Book → signal available (incl. Python) | < few ms | n/a (Rust inline) |
| Signal → order on wire | < 100 µs | < 20 µs |
| Wire → venue commit | ~0.2–0.9 s (venue-bound) | ~0.2 s + geography |
| **Internal jitter (p99 − p50), core** | **< 200 µs** | **< 50 µs** |

The internal-jitter row is the one Rust genuinely owns and the one to hold the line on.

## Sources

Full figures, caveats (amortized-vs-single-message, vendor-favorable numbers), and links in
[research/python-rust-ipc.md](research/python-rust-ipc.md) and
[research/hyperliquid-execution.md](research/hyperliquid-execution.md).
