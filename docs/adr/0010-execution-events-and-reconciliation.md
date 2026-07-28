# ADR-0010 — Execution events on the core bus, and an unbypassable risk gate

**Status:** Accepted · **Date:** 2026-07-25

## Context

Phase 3 increments 1–2 gave us signing and a REST `ExecutionClient`, and a live testnet
round-trip proved an order can be placed, rested and cancelled. What was still missing is
everything that happens *after* submit: an `OrderAck` says a submit succeeded, and nothing
more. It does not say the order is still there, how much of it filled, or whether the venue
cancelled it out from under us. Until the core consumes the venue's own lifecycle stream, our
position is a guess.

Four questions had to be answered, and each has a wrong answer that looks reasonable:

1. **Where does the execution vocabulary live, and does it ride the market-data bus?**
   The alternative is a second channel for order events.
2. **One "order status" type, or separate fill and order-update types?**
3. **How is `axon-risk` actually reached on the submit path?** It has existed and been pure
   since Phase 1, and nothing called it.
4. **How does reconciliation survive a reconnect?** Hyperliquid replays a fill snapshot on
   `userFills`, and its `orderUpdates` channel never snapshots at all.

## Decision

**1. The execution vocabulary lives in `axon-core` (`exec.rs`) and shares the one bus.**
`Event` gains exactly one variant, `Exec(ExecEvent)`, alongside `Market(MarketEvent)`, with
`ExecEvent` being `Fill | Order | Account`. Providers translate their wire format into these
types and publish with the same `EventSender`, exactly as ADR-0008 established for market data.
`OrderStatus` moved down from `axon-providers` into `axon-core`; `axon-providers` re-exports it.

One bus, not two, because **a fill and the book update that caused it must be orderable against
each other**. On separate channels the core could observe a fill before the trade that produced
it, and replay would stop matching live — which would silently invalidate the entire parity
harness that Phase 5 is built on. The cost is that every handler must ignore the variants it
does not care about; `MarketDataProcessor`'s infallible destructure became a `let … else`.

**2. `Fill` and `OrderUpdate` are separate types.** A venue can report a partial fill without
the order leaving the book, and can cancel an order that never filled. Collapsing both into one
status type discards the fill quantities that position math needs. `Fill` carries `trade_id`
purely as a dedup key, and `OrderUpdate` carries **absolute** `orig_qty`/`remaining_qty` rather
than deltas, because a post-reconnect snapshot is absolute and mixing the two representations is
how position drift starts.

**3. The risk gate is a type, not a convention.** `axon-execution`'s
`GuardedClient<C, X>` *implements* `ExecutionClient` and owns the venue client, so the strategy
side is handed a guarded client and the raw one is unreachable. A gate that callers are merely
expected to consult is one forgotten call site away from an unlimited position.

Three asymmetries fall out of taking that seriously:

- **Cancels are never gated.** Refusing a cancel cannot reduce risk; it converts a stale price
  feed into an un-exitable position.
- **Batches are checked cumulatively.** Twenty orders each under `max_position` can collectively
  exceed it, so each order is projected onto a running position as if it fills in full — the only
  safe assumption for a resting order.
- **A missing mark price fails closed** for anything that could add exposure, because an unpriced
  notional check is not a check. Reduce-only is exempt so flattening always remains possible.

What the gate sees comes from a `RiskContext`. `TrackerRiskContext` reports
`OrderTracker::risk_position` — filled position **plus** the worst case that every live order
fills — so resting exposure is counted before the next order is admitted. `StaticRiskContext`
reports filled position only and is for offline tests; using it in production would reintroduce
the aggregation hole.

**4. `OrderTracker` treats the venue as the authority, and is built around four named failure
modes** (each with a test named after it): duplicate fills after a reconnect (deduped on
`trade_id`), orders we never submitted (adopted, not ignored, or a cancel-all sweep misses them),
out-of-order updates (terminal is terminal), and resting-order exposure.

The subtlest decision here: **filled quantity is monotonic, not timestamp-gated.** Fills and
order updates arrive on different channels whose clocks need not agree, so the two sources
combine with `max` rather than "newest wins". A fill can only ever increase, so `max` is correct
regardless of arrival order — whereas trusting the newer timestamp discards a real fill whenever
the lifecycle stream lags the fill stream. Terminal statuses are likewise applied regardless of
timestamp, because believing an order is live when it is dead is the dangerous direction.

## Consequences

- **+** Replay determinism is preserved: fills and market data share one event-time ordering, so
  the Phase-5 parity harness can compare a backtest and a live session event for event.
- **+** Risk cannot be bypassed by construction, and the aggregation hole (N orders each passing,
  collectively breaching) is closed by the tracker-backed context rather than by discipline.
- **+** Reconnects are safe by default: snapshot replays are idempotent, and adopted orders mean a
  restarted process can still find and cancel what it left behind.
- **+** Position drift is *detectable*, not just avoided: `AccountSnapshot` gives the venue's own
  equity, and `orphan_fills` counts fills we could not attribute — the signal that our view and
  the venue's have diverged.
- **−** Every `EventHandler` must now ignore variants it does not handle. Cheap, but it is a
  `match` arm that will be forgotten at least once.
- **−** `OrderTracker` behind an `Arc<RwLock<…>>` puts a lock on the submit path. The critical
  section is a few map lookups and no `await` happens inside it, but it is not zero, and a
  thread-per-core Phase-8 core would want a different arrangement.
- **−** The dedup set is bounded (`MAX_SEEN_TRADES`), so a replay older than that window would be
  double-counted. Acceptable because venues only replay near a reconnect, but it is a real edge.
- **−** Hyperliquid's `orderUpdates` never snapshots, so open-order state after a restart *must*
  come from `POST /info`. Reconciliation is therefore not purely stream-driven; there is a REST
  dependency on the recovery path.

See [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the SPSC ring is for Python↔Rust signals,
not this bus), [ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the vocabulary/bus pattern this
extends), [ADR-0009](0009-hyperliquid-signing.md).
