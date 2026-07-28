# ADR-0008 — Market-data bus (crossbeam) + hand-rolled WS ingest

**Status:** Accepted · **Date:** 2026-07-24

## Context

Phase 2 (`docs/08-roadmap.md`) wires the market-data path: async venue I/O ingests
data and hands it to the single-threaded deterministic core (`docs/01-architecture.md`).
[ADR-0004](0004-provider-abstraction-layer.md) and `docs/04-provider-abstraction.md`
deliberately left the concrete **stream type** — how events cross from the async edge
to the core — marked `OPEN`, to be settled once the core loop existed. Three coupled
choices had to be made to land the first increment:

1. **The async→core seam.** The architecture mandates "async (tokio) only at the
   edges; the hot loop stays off the runtime." Options: a `tokio::mpsc` channel (drags
   the core consumer onto the runtime), reuse the `axon-ipc` SPSC mmap ring (shaped for
   the Python↔Rust fixed-record boundary, single-producer only), or a synchronous
   in-process channel.
2. **The Hyperliquid WS client.** Hand-roll `tokio-tungstenite`, or adopt a community/
   official SDK. The execution research (`docs/research/hyperliquid-execution.md`)
   already recommends wrapping any SDK behind our own trait and not adopting SDK types.
3. **Where the normalized market vocabulary lives.** It began in `axon-providers`, but
   the core bus must carry it, and `axon-core` cannot depend on `axon-providers`
   (that crate already depends on core).

## Decision

1. **Bounded `crossbeam-channel` as the seam.** `axon-core::bus` is a *synchronous*
   bounded MPSC channel: async adapters hold a cloneable `EventSender`; the core owns
   the single `EventReceiver` and drains it with `event_loop::{drain_available,
   run_blocking}` — never touching tokio. Bounded gives backpressure instead of
   unbounded memory growth. This is distinct from the `axon-ipc` SPSC ring, which stays
   dedicated to the cross-process Python↔Rust signal boundary
   ([ADR-0006](0006-signal-schema-and-spsc-ring.md)).
2. **Hand-roll the read-only WS.** `axon-provider-hyperliquid::ws` uses
   `tokio-tungstenite` (rustls) directly behind the `MarketData` port: pure decoders
   (`l2Book`/`trades`/`bbo` JSON → normalized events, fixed-point `Decimal`, ms→ns) plus
   a connect/subscribe/heartbeat/reconnect loop. No SDK dependency for market data; an
   SDK will be re-evaluated for **signing** in Phase 3, where the value is real.
3. **Normalized vocabulary lives in `axon-core::market`** (`Bbo`, `Trade`, `Level`,
   `BookSnapshot`, `MarketEvent`), wrapped by `axon-core::Event`. `axon-providers`
   **re-exports** it so the port and adapters still speak one set of types with no
   dependency cycle.

Testing follows the same split: decoders + bus + processor + loop are unit-tested
offline (deterministic, network-free in CI); the live WS connection sits behind an
`#[ignore]`d smoke test and the `live_book` example, run manually.

## Consequences

- **+** The core stays synchronous and deterministic — the property that lets backtest
  and live share one code path — with async strictly at the edge.
- **+** Minimal dependency surface for market data; full control over fixed-point
  parsing and reconnection. Adding a second venue is another decoder + client behind the
  same bus, not a new transport.
- **+** One source of truth for the event vocabulary; the `OPEN` stream-type question in
  `docs/04` is now closed.
- **+** CI is deterministic: no test depends on the network.
- **−** The bus is in-process only; a future out-of-process consumer would need a
  different transport (acceptable — that is what `axon-ipc` is for).
- **−** Hand-rolling means we own reconnection/heartbeat correctness rather than
  inheriting an SDK's. Mitigated by keeping the decoders pure and heavily tested, and by
  the live smoke test.

  > **Amended 2026-07-26.** That mitigation was insufficient, and it is worth recording
  > exactly how, because the shape recurs. Pure decoders and a smoke test cover the code that
  > runs *while a connection is up*; reconnection correctness lives in the code that runs when
  > one is **down**, and a smoke test never sees it. Two defects sat in that gap for a whole
  > phase and were found only by a 1 h 44 m soak through 36 induced outages
  > ([07](../07-parity-and-testing.md)). The backoff reset only on a clean WebSocket close,
  > which **no severed link produces** — a FIN with no Close frame arrives as
  > `Connection reset without closing handshake`, an RST as `Connection reset by peer`, both
  > the error path — so the reset was unreachable from any real network event and eight
  > disconnects pinned the wait at its 30 s cap for the rest of the session: a 0.4 s outage
  > cost a 30 s blackout, and 45.8 % of the soak ran under `STALE MARKS` with the risk gate
  > refusing every risk-increasing order. And the venue's heartbeat (`{"channel":"pong"}`,
  > which carries no `data` field) was logged as a decode error on every beat, while the test
  > asserting pongs were ignored passed against a frame the venue does not send — an invented
  > `{"channel":"pong","data":null}`. **A decoder test is only as good as the bytes it was
  > given**, which is why this crate holds decoders to captured real frames; that rule had
  > simply never been applied to the frames that are not market data. Both are closed, both
  > with real committed fixtures. The general lesson stands and is not closed: *nothing in the
  > default gate can exercise a network fault*, so this class of defect is reachable only by a
  > soak, and the soak is not run by CI.
- **−** The `MarketData::subscribe(feed)` signature predates multi-coin reality (it
  carries no symbol); this increment interprets it as "this feed for all configured
  coins." Finalizing the port signature is deferred to when execution lands (Phase 3).

See [ADR-0004](0004-provider-abstraction-layer.md), [ADR-0006](0006-signal-schema-and-spsc-ring.md),
`docs/01-architecture.md`, `docs/04-provider-abstraction.md`,
`docs/research/hyperliquid-execution.md`.
