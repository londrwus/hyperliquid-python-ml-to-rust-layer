# `contracts/` — the language-neutral contract

**Implemented (Phase 1; the second record added in Phase 2, the third in Phase 6).**
[`schema.toml`](schema.toml) is the
**single source of truth** for everything that crosses the Python↔Rust boundary: Rust generates +
compile-time-asserts its `#[repr(C)]` structs against it (`axon-contracts`), Python builds a
matching NumPy dtype from it at import (`axon.contracts`), and a cross-language round-trip test is
the backstop — so the two sides can never silently drift. This is the linchpin of
[ADR-0002](../docs/adr/0002-python-rust-boundary.md) /
[ADR-0006](../docs/adr/0006-signal-schema-and-spsc-ring.md) /
[ADR-0012](../docs/adr/0012-market-data-ring-and-multi-record-contract.md).

> The section below documents the design; the authoritative layout now lives in `schema.toml`.

## What lives here

1. **The `Signal` record** — the fixed-layout, versioned struct a strategy emits across the
   shared-memory ring. Default shape (target-position; see [06](../docs/06-strategy-contract.md)),
   at **schema version 3**:
   ```
   Signal {
     schema_version:    u8,
     seq:               u64,   // monotonic; for reproducibility + gap detection
     ts_event:          i64,   // event-time ns: the moment the strategy DECIDED
     symbol_id:         u32,   // canonical symbol id (via SymbolMap)
     target_qty:        i64,   // fixed-point signed target position
     urgency:           u8,    // execution aggressiveness hint (URGENCY_TABLE)
     price_band:        i64,   // worst-acceptable price (fixed-point); 0 = none
     ttl_ms:            u32,   // signal ADMISSION window; 0 = the operator's ceiling
     model_version:     u32,   // which model produced this (audit/replay)
     flags:             u16,   // reduce_only, close
     max_order_age_ms:  u32,   // ORDER lifetime; 0 = the operator's ceiling  (v2, ADR-0031)
     ts_cause:          i64,   // event-time ns of the OBSERVATION this answers; 0 = unstated
   }                           //                                              (v3, ADR-0037)
   ```
   Fixed size, cache-friendly, no dynamic serialization on the hot path. A `schema_version`
   byte guards Python/Rust drift.

   **Three of those fields are routinely confused and each confusion has cost a run.**
   `ttl_ms` is how old a *record* may be when the reader admits it; `max_order_age_ms` is how
   long an *order* may keep its place at the venue — a large `ttl_ms` buys a resting order
   exactly nothing ([ADR-0031](../docs/adr/0031-order-lifetime-and-the-sweeper-on-the-pass.md)).
   And `ts_event` is the moment the strategy *decided*, while `ts_cause` is the moment of the
   thing it decided *about* — an m1 bar's own close. The gap between the two is the largest
   latency in the system (951 / 12 051 / **111 475** ms, measured over 57 live bars), and until
   v3 it was invisible from Rust
   ([ADR-0037](../docs/adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md) §4).

   **`ts_cause` spent the last of `reserved`.** The record is now fully named, so the next field
   has to re-cut the 64-byte layout rather than extend it.

2. **The ring layout** — header (capacity, head/tail indices on separate cache lines),
   record stride, the **record-kind tag**, and the framing/sequence protocol. The kind tag is
   what stops a reader mapping the wrong record type onto a matching stride; strides differing
   today is a coincidence of the field lists, not a guarantee (ADR-0012).

3. **The `MdSlice` record** (Rust→Python market-data ring) — the slice of market state a Python
   feature computation needs: `seq`, `ts_event`, fixed-point BBO (px+sz per side), the last trade
   print (px, sz, and its *own* event time), `symbol_id`, and a `kind` saying what caused the
   update — plus, since **schema version 2**, the venue's mark, index, funding rate, funding
   interval and the mark's two clocks, which is what the 48 reserved bytes were held for
   ([ADR-0028](../docs/adr/0028-market-data-bars-and-the-ticker-tail.md) §2). 128 bytes = two
   cache lines, and **none of them is reserved any more**. Every slice carries full state, not a
   delta, so a consumer that falls behind still sees something coherent. Read a **batch** per call
   from Python (`axon.marketdata`), never one record per call.

3b. **The `MdBar` record** (Rust→Python bar ring) — a *closed* candle: `open_time`, OHLCV,
   `interval_ms`, and flags for `gap_before` / `first_bar`. It is a separate record on a separate
   ring rather than an `MdSlice` kind, because a slice answers *what is true now* and a bar answers
   *what happened over a closed interval* — two consecutive identical bars are two facts, and the
   slice ring's `on_change` coalescing would delete the second one. It shares `MdSlice`'s 128-byte
   stride, which is exactly why the ring's `record_kind` tag is load-bearing: `record_size` cannot
   tell the two apart at all ([ADR-0028](../docs/adr/0028-market-data-bars-and-the-ticker-tail.md)).

4. **The config schema** — `StrategyConfig` (params, symbol universe, model ref, risk limits)
   and the model-artifact metadata (version, I/O schema, feature-spec ref, git SHA).

5. **The feature spec** — the versioned description of the model's input features (feeds the
   parity harness and any future Rust feature runtime).

## How it's kept in sync

Settled in Phase 1 (ADR-0006), and it is code generation from one definition, because that
makes drift structurally impossible — the entire point of this directory:

- **Rust:** `axon-contracts/build.rs` emits offset/size constants from `schema.toml`; each
  `#[repr(C)]` struct asserts every field offset against them with `core::mem::offset_of!` in a
  `const _` block. Drift is a **compile error**, and `#[derive(Pod)]` additionally rejects any
  implicit padding.
- **Python:** `axon.contracts` parses the same file with `tomllib` and builds NumPy structured
  dtypes with **explicit offsets** (never NumPy's own packing).
- **Backstop:** cross-language round-trip tests — `python/tests/test_roundtrip.py` (signals, both
  directions) and `python/tests/test_md_ring.py` (market data, Rust→Python).

## Rules
- **Event-time everywhere.** Timestamps are the event's own time, never wall-clock at receipt —
  required for deterministic replay ([07](../docs/07-parity-and-testing.md)).
- **Fixed-point, not float**, for any price/size/quantity.
- **Versioned + backward-compatible.** Bump the record's own `schema_version`; readers reject
  unknown versions loudly. Each record versions independently, so bumping one cannot change the
  version byte another stamps on the wire.
- **Changes here are ADR-worthy** — the record shapes are foundational, hard-to-reverse
  decisions.
