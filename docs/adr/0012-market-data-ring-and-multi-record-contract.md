# ADR-0012 — The market-data ring, and a contract with more than one record

**Status:** Accepted · **Date:** 2026-07-25

## Context

`docs/01-architecture.md` step 2 and `docs/02-python-rust-boundary.md` both describe a second
ring running the other way: the Rust core publishes the market-data slice Python needs for
feature computation. It could not exist. `contracts/schema.toml` defined exactly one record and
`axon-ipc`'s `Producer`/`Consumer` were hard-typed to `Signal` — the transport had no notion of
carrying anything else, and the schema had no vocabulary for a second layout.

The Phase-1 machinery is not the problem: the ring, the ordering contract and the codegen all
work and are proven byte-identical across the boundary ([ADR-0006](0006-signal-schema-and-spsc-ring.md)).
What was missing is *plurality*. Four questions had to be answered, and each has an answer that
looks reasonable right up until it corrupts something:

1. **Does Python need this ring at all?** `docs/02` left "Python subscribes to the venue
   directly" as a config choice, and it is the cheaper option — no new record, no new transport.
2. **What does one market-data record carry?** Top-of-book only, a depth-N snapshot, or deltas?
   Deltas are the smallest thing on the wire and the obvious answer for a book feed.
3. **How does a reader know which record a ring holds?** The ring header already carries
   `record_size`, and `Signal` (64) and the new record (128) differ, so the check appears free.
4. **What shape is the Python read API?** The existing `try_pop()` returns one record and already
   works; extending it to a second dtype is a one-line change.

## Decision

**1. Rust publishes; Python does not open its own venue connection.** The core already holds the
socket, maintains the book and stamps event time. A second connection would give Python a second
view of the market, and features would then be computed on a book the executing core never saw —
which is indistinguishable, from the outside, from a model change. The Phase-5 parity harness
compares a backtest against a live session event for event; it can only do that if there is one
event stream. The ring costs ~100 ns and removes an entire class of "why did live differ?"

**2. `MdSlice` is a 128-byte, full-state record.** Fields: `seq`, `ts_event`, fixed-point
`bid_px/bid_sz/ask_px/ask_sz`, `last_trade_px/sz/ts`, `symbol_id`, `flags`, `schema_version`,
`kind`, `reserved[48]`. Three choices inside that are worth naming:

- **Full state, not deltas.** Every slice carries the current BBO *and* the last print. A
  consumer that falls behind and skips records still sees something coherent; with deltas, one
  dropped record poisons every value derived after it, and the Python side is exactly the side
  that is allowed to fall behind (see 5).
- **The last print keeps its own timestamp.** `last_trade_ts` is separate from `ts_event`
  precisely because a quote-driven slice carries a *stale* print, and "how stale" is a feature
  input. Folding them into one field would make a 200 ms-old trade look simultaneous with the
  quote that triggered the update — the kind of leak that makes a backtest look better than
  live.
- **`kind` is the update's cause, not the record's type** (quote / trade / snapshot). Every slice
  has full state, so `kind` is the only thing that says *what just moved*, which is what
  event-driven features key on. The record *type* is a ring-level fact, not a per-record one —
  see 3.

128 bytes is two cache lines: nine 8-byte quantities are 72 bytes before the ids, so one line
was never available, and a non-power-of-two stride would put half the records across a line
boundary for no gain. The 48 spare bytes are the same bet as `Signal`'s `reserved[15]` — mark
price and funding (the Phase-2 ticker increment) land without a stride change, so adding them
cannot break a reader that only knows today's fields.

**3. The ring control block carries an explicit `record_kind`, and it is validated on open.**
`record_size` would distinguish the two records *today*, and that is exactly the problem: equal
strides are a coincidence of the current field lists, not a contract. `Signal`'s reserved bytes
exist so an order-intent variant can be added at the same 64-byte stride; an `MdSlice` v2 could
land on 64 or a `Signal` v2 on 128. A reader that clears a stride check and then maps the wrong
struct does not fail — it reports `target_qty` as a bid price and keeps going. That is the
silent corruption this whole directory exists to prevent, so the check is made intentional
rather than incidental. Kind `0` is left unassigned so a writer that never stamps the field is
rejected instead of defaulting into the signal ring, and `ring.version` went to **2**: a v1
reader is structurally unable to perform the check, so it should refuse the file rather than
half-validate it.

**4. One ring implementation, generic over a `Record` trait.** `Record` (in `axon-contracts`, so
the kind tag lives with the contract) is `bytemuck::Pod` plus `NAME`/`SIZE`/`SCHEMA_VERSION`/`KIND`.
`RingProducer<R>`/`RingConsumer<R>` replace the hard-typed pair, with
`Producer`/`Consumer`/`MdProducer`/`MdConsumer` aliases so call sites read as before. The
Release/Acquire sequence, the mutable-provenance base pointer, the `Send`-not-`Sync` opt-in and
every SAFETY comment were re-typed, not rewritten: one unsafe implementation with one set of
invariants to review is worth more than a specialized ring per direction. Python mirrors this
with a dtype→`RecordSpec` registry — `RingProducer`/`RingConsumer` take the record *dtype* and
look the kind and schema version up from it, so a ring's control block cannot end up describing
a different record than the one its dtype decodes.

Each record also owns its `schema_version`. A shared version would mean bumping `MdSlice`'s
layout changes the version byte `Signal` stamps, breaking a boundary that did not change.

**5. The Python API is batch-first, and the publisher never blocks.**
`MdRingConsumer.read_batch()` returns everything queued as one NumPy structured array — at most
two slice copies (a batch wraps the ring at most once) and a single `tail` publish. Python's
interpreter floor is ~50–350 ns *per call* (`docs/02`); a per-message API spends the entire
latency budget on call overhead at a few thousand updates a second, and hands the caller scalars
where `axon.features` wants vectors. The batch is a **copy**, not a view: `tail` releases those
slots to the producer, and a view into them would be overwritten mid-computation.

The other half of that trade: a full ring makes the Rust publisher drop the update rather than
wait. Stalling the execution core on a slow feature computation is the one thing this direction
must never do. `seq` is monotonic and gap-free at the source, so the span of a batch minus its
length is exactly what was lost — `MdRingConsumer.dropped` reports it, because a strategy needs
to know its feature history has a hole rather than compute confidently across it.

Two publisher-side rules follow, and a reader of this ADR alone would not know either. The
publisher spends a `seq` on every *attempted* push, so a drop leaves exactly the gap the consumer
infers loss from; an update coalesced away by the write policy spends none, so suppression is never
reported as loss. And `Ticker` and `Candle` never produce a slice: neither moves a field the record
has, and a venue that does not timestamp its ticker would make the published `ts_event` a receipt
clock, which does not replay.

## Consequences

- **+** The Rust→Python direction of `docs/01` exists, and both directions now run on one
  transport whose unsafe surface did *not* grow to accommodate the second record.
- **+** Mapping the wrong record onto a ring fails at `open()` with a named error, in both
  languages, whether or not the strides happen to differ.
- **+** `Signal` is byte-for-byte unchanged and keeps `schema_version = 1`; only the ring header
  grew, into previously unused bytes of control cache line 0.
- **+** Feature computation and execution see one event stream from one connection, which is what
  makes the Phase-5 parity comparison meaningful.
- **−** `ring.version = 2` means a stale binary refuses a new ring. Deliberate, but it does mean
  Rust and Python must be deployed together — there is no mixed-version window.
- **−** 48 of 128 bytes are reserved, so ~37% of md-ring bandwidth is padding today. Cheap
  (~1.3 MB/s at 10k updates/s) against the alternative of changing the stride later.
- **−** `MdSlice` is top-of-book only. Queue position, depth imbalance, or anything past L1 cannot
  be computed from it; that needs a second record type or Boundary A, and pretending otherwise
  would be worse than the limit.
- **−** Batch reads quantize *reaction* time by the poll interval, even though event time per
  record is exact. And Python busy-polls: there is no wake primitive on this ring, so a tight
  loop costs a core. An `eventfd`/futex wake is deferred, as it was for the signal ring.
- **−** Drop accounting trusts `seq` to be gap-free at the source. A future publisher that skips
  sequence numbers for its own reasons would report those skips as drops.
- **+** The Rust-side publisher exists: `axon_runtime::mdring::MdPublisher` writes the ring from
  `CoreHandler`'s fan-out, last, behind `[md_ring] enabled` and off by default.
  `crates/axon-ipc/examples/md_writer.rs` remains what the cross-language test drives, because a
  test that had to start a session would be testing the session.

See [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the ring and the codegen this extends),
[ADR-0002](0002-python-rust-boundary.md) (why Boundary B), [ADR-0008](0008-market-data-bus-and-ws-ingest.md)
(the in-process bus this ring is *not* — that one is Rust-to-Rust),
[`contracts/`](../../contracts/README.md), [02](../02-python-rust-boundary.md).
