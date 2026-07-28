# Research — Python↔Rust low-latency IPC (2026)

**Bottom line:**
- In-process (PyO3) beats any cross-process IPC by 2–3 orders of magnitude, but the **real
  in-process enemy is the GIL**, not call overhead. Cross the boundary *rarely* (batch); keep
  hot math in Rust.
- Raw single-message hot path → **shared-memory SPSC ring**. Batches of features → **Apache
  Arrow** (zero-copy). Decoupled/multi-consumer → a message queue is fine.
- **Reality check:** talking to Hyperliquid is a **millisecond** round trip over the public
  internet — ~10,000–100,000× larger than any IPC choice here. **Pick the boundary for
  simplicity/correctness, not IPC nanoseconds.** The ns differences matter for internal
  fan-out and jitter, not for beating the wire.

## Latency ladder (small messages, same host, Linux; one-way)

| Mechanism | One-way latency | Cross-process? |
|---|---|---|
| Pure-Rust in-process call | ~3 ns (0 if inlined) | no |
| **PyO3 boundary crossing** | **~20–70 ns** | no (same process) |
| Intra-process SPSC/disruptor (Rust) | ~4–66 ns | no |
| **mmap shared-memory SPSC ring** | **~66–135 ns** | yes |
| **iceoryx2 pub-sub (zero-copy)** | ~240 ns (graph) / ~600–700 ns (measured) | yes |
| POSIX message queue | ~2.7 µs | yes |
| Unix domain socket | ~0.7–4.5 µs | yes |
| TCP loopback | ~1.8–7 µs | yes |
| ZeroMQ inproc / tcp-loopback | ~8 µs / ~26 µs | yes |
| gRPC + protobuf (unary) | ~100–300 µs | yes |
| **Exchange WS round trip (context)** | **milliseconds** | network |

## PyO3 + maturin (in-process)
- Per-call overhead ~20–70 ns; **data conversion usually dwarfs the call itself**.
- **GIL serializes a hot loop** — only one thread drives the interpreter; repeated calls can't
  run in parallel. Release the GIL around GIL-free Rust work only for chunks >~1 ms.
- Free-threaded (no-GIL) Python: PEP 779, officially supported (not default) from 3.14; PyO3
  supports it. **Ecosystem consensus: test now, don't deploy to prod yet.**
- Direction: *Rust-as-extension* (Python host imports a Rust module — Polars/Pydantic pattern)
  is the low-friction default; *embed Python in Rust* when the whole request path is
  latency/tail-critical. Same ~30–70 ns boundary cost either way.

## Shared memory (our pick)
- mmap `MAP_SHARED` / `/dev/shm` SPSC ring: **~100–135 ns one-way**; zero syscalls in the data
  path after setup — just a memcpy.
- **Crate reality check:** `rtrb`, `disruptor-rs`, `ringbuf` are **intra-process only** — do
  NOT use across the Python/Rust process boundary. Purpose-built cross-process option:
  **`shmem-ipc`** (Linux-only; wait-free bounded SPSC, `eventfd` signaling; ~3× faster than
  Unix sockets). `shared_memory`/`raw_sync` give you the raw region to build a ring on.
- **False sharing:** head/tail on the same cache line caused a ~3.1× slowdown — pad indices
  (`crossbeam_utils::CachePadded`, 64 B x86 / 128 B Apple Silicon) with acquire/release fences.
- **Python side:** `multiprocessing.shared_memory.SharedMemory` + `np.ndarray(buffer=shm.buf)`
  = zero-copy view. Python's interpreter floor is ~50–350 ns/call, so **poll the ring and read
  a batch per call**, not one message per call.
- Verdict: cross-process mmap SPSC ring ≈ 100 ns, ~10–50× faster than UDS. Best raw hot-path
  option — but you own synchronization, cache coherency, and crash recovery.

## Apache Arrow (zero-copy columnar)
- Two distinct mechanisms — don't conflate:
  - **C Data Interface / PyCapsule = same-process only, truly zero-copy** (pointer + release
    callback). Best fit: Python computes a feature batch → Rust consumes with no copy. Rust
    side via `arrow-pyarrow` or `pyo3-arrow` (avoids the ~130 MB pyarrow dep).
  - **Arrow IPC / Flight = cross-process/machine** (FlatBuffers/gRPC framing) — ~3 GB/s
    localhost but inherits gRPC's hundreds-of-µs floor.
- Deserialize is ~free (zero-copy); small batches are penalized. Sweet spot ~8192 rows / 256 KB.
- Vs raw shmem: Arrow wins for **batches + schema + cross-language interop**; raw ring wins for
  **single small fixed-size hot-path messages and tail latency**.

## iceoryx2 (Rust zero-copy IPC)
- Eclipse, zero-copy, lock-free, **broker-less** shared-memory middleware; pub-sub +
  request-response + event + blackboard. Latency "<1 µs," ~600–700 ns real-world.
- **v0.9.3 (2026-07), still pre-1.0; v1.0 targeted end-2026.** **Python bindings exist**
  (`pip install iceoryx2`) — directly usable for a Python-strategy / Rust-engine split with
  zero-copy cross-language pub-sub. Architecturally excellent for fan-out of tick data to many
  consumers. Caveat: no public finance case study; pin the version and benchmark.

## Message queues / RPC
- ZeroMQ: ~8 µs inproc / ~26 µs tcp-loopback, but **batches for throughput** so a lone message
  can be 5–10× slower. gRPC: ~100–300 µs unary even over Unix socket. Fine for control-plane,
  config, backtesting data pulls, cross-machine transport — **not** the hot path.

## Where should the boundary sit?

| Option | Latency | Fidelity (live Python ML) | Best when |
|---|---|---|---|
| **A. Rust runs everything** | Best/most deterministic tail | Lowest (no live Python) | Tightest tail; models export cleanly |
| **B. Python signals in-process → shmem → Rust executes** | Excellent (~100 ns hop, off hot path) | **High** (full Python stack live) | **Most trading systems — the best fit here** |
| **C. Rust embeds Python via PyO3** | Good call, but **GIL serializes the loop** | Highest | Inference must be inside the loop; can batch |

**Recommendation for this stack:** default to **B**. Python runs ML/signal generation and
writes signals into a shared-memory ring; Rust reads them and owns order construction +
Hyperliquid networking. Full Python ML fidelity; Rust owns the µs/jitter-sensitive path; the
IPC hop is negligible vs the ms exchange round trip. Use `shmem-ipc` or a hand-rolled
cache-padded SPSC ring for minimal deps, or `iceoryx2` for structured multi-consumer fan-out
(accept pre-1.0). Reserve **C** for inference-in-the-loop (batch to amortize the GIL). Choose
**A** later for the execution core. Skip gRPC/Flight on the hot path.

## Caveats
Many published ns figures are throughput-derived/amortized, not true single-message latency
(a "sub-10 ns C++ ring" latency claim was judged not credible). Aeron "0.25 µs RTT" is
vendor-favorable (credible figure ~18 µs). iceoryx2 ~240 ns is read off a graph; real-world
~600–700 ns, config-dependent. Benchmark on your own tick volumes.

## Sources
pyo3.rs/main/{performance,parallelism,free-threading} · github.com/PyO3/pyo3/issues/1607,3827 ·
victoranderssen.com/blog/linux-ipc-benchmark · lib.rs/crates/shmem-ipc · github.com/mgeier/rtrb ·
arrow.apache.org/docs/format/CDataInterface · github.com/eclipse-iceoryx/iceoryx2 ·
pypi.org/project/iceoryx2 · arxiv.org/html/2508.07934v1 (MQ benchmark) ·
mpi-hd.mpg.de/.../grpc-for-ipc · docs.python.org/3/library/multiprocessing.shared_memory.html
