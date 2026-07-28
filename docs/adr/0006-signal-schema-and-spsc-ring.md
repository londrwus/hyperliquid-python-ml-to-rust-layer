# ADR-0006 — Signal schema & the SPSC shared-memory ring

**Status:** Accepted · **Date:** 2026-07-12

## Context

Phase 1 must lock the seam every other component hangs off: the exact bytes that
cross the Python→Rust boundary, how both languages stay in sync, and the transport
that carries them. [ADR-0002](0002-python-rust-boundary.md) chose Boundary B (Python
signals → shared memory → Rust executes) and left three things `OPEN`: the `Signal`
shape, the contract-sync mechanism, and the shared-memory implementation. This ADR
closes them.

Constraints from the design: fixed-layout and versioned records (no per-tick
serialization); event-time everywhere; fixed-point, never float, for price/size;
cache-line-separated head/tail to avoid false sharing; and — critically — the two
language sides must not be able to drift silently. Development is on Windows;
production is Linux ([ADR-0007](0007-linux-wsl2-dev-target.md)).

## Decision

**1. Signal shape — target-position (default), 64-byte record.** A strategy emits a
`Signal` that declares the *position it wants*; the Rust engine decides *how* to
reach it. The record is exactly one cache line (64 B), little-endian, padding-free,
with fields ordered by descending alignment. A `kind` discriminant plus 15 reserved
bytes leave room for an explicit-order-intent variant later **without changing the
stride**. Fields: `seq, ts_event, target_qty, price_band, symbol_id, ttl_ms,
model_version, flags, schema_version, urgency, kind, reserved[15]`. `model_version`
+ `seq` make every live decision reproducible offline.

**2. One source of truth + codegen.** `contracts/schema.toml` is the single
definition. The Rust `axon-contracts` crate *generates* offset/size constants from
it at build time (`build.rs`) and the `#[repr(C)]` struct asserts every field offset
against those constants at **compile time** (`core::mem::offset_of!`) — a drift is a
build error. Python (`axon.contracts`) parses the *same* file at import with
`tomllib` and builds a matching NumPy dtype. A bidirectional cross-language
round-trip test is the final backstop.

**3. Transport — hand-rolled cache-padded SPSC ring over a memory-mapped file.**
`head` (producer) and `tail` (consumer) are monotonic `u64` counters on separate
64-byte cache lines; publish/consume use Release/Acquire. The backing store is a
plain memory-mapped file — the one primitive that is portable and coherent across
processes on *both* Linux (`/dev/shm` tmpfs) and Windows, and mappable zero-copy by
both `memmap2` (Rust) and Python's `mmap`. `shmem-ipc` (Linux-only) and `iceoryx2`
(pre-1.0) are **deferred**, kept behind the same `Producer`/`Consumer` API as future
optimizations.

**4. Platform assumption — x86_64, little-endian.** The Python producer relies on
x86-TSO store ordering (payload written before the index). ARM would additionally
need an explicit fence on the Python side; documented, not yet needed.

**5. Money type — `rust_decimal`** for all price/size/qty in the Rust core;
fixed-point integers (units of `10^-8`) on the wire. Rust edition 2021.

## Consequences

- **+** Drift between Python and Rust is structurally caught (compile-time asserts +
  runtime dtype + round-trip test), which is the entire point of `contracts/`.
- **+** The ring is dependency-light, portable, and already benchmarked-class
  (~100 ns), which is far below the ms wire — exactly the right amount of
  engineering for v1 ([ADR-0002](0002-python-rust-boundary.md)).
- **+** The 64-byte record + `kind`/`reserved` gives a clean extension path to
  explicit order intents without a new stride or a schema-breaking change.
- **−** We own the unsafe ring code (memory ordering, mmap lifetime, peer-death) —
  mitigated by an adversarial review and a threaded stress test under backpressure.
- **−** The x86_64/LE assumption is baked in; revisiting it (ARM/Graviton) is a new
  ADR.
- **Verified in Phase 1:** a byte round-trips Python → ring → Rust *and* Rust → ring
  → Python, byte-identical, on Windows (53 tests green).

Supersedes the `OPEN` items in [ADR-0002](0002-python-rust-boundary.md) §Consequences.
See [02](../02-python-rust-boundary.md), [06](../06-strategy-contract.md),
[`contracts/`](../../contracts/README.md).
