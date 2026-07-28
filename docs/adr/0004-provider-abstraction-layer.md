# ADR-0004 — Provider abstraction (venue-agnostic execution)

**Status:** Accepted · **Date:** 2026-07-12

## Context

Hyperliquid is the first venue, but the project's stated goal is a **Layer** that can execute
against any provider (other DEXs, CEXs like Binance/Bybit). If Hyperliquid-specific details
(wallet signing, ms-timestamp nonces, synthetic market orders, volume-gated rate limits,
on-chain block timing) leak into the core or the strategy, we get a Hyperliquid bot, not a
layer. Research shows a clean line exists: order intent, order book, position/risk, and
lifecycle events generalize across CEX and DEX; the divergences are all absorbable at an
adapter edge ([research/hyperliquid-execution.md](../research/hyperliquid-execution.md)).

## Decision

Define venues behind an **outward port** of trait interfaces, with all venue quirks confined to
the adapter:

- `ExecutionClient` — `place_order` / `place_batch` / `cancel` / `modify` / `cancel_all` →
  venue-agnostic `OrderAck`.
- `MarketData` — `subscribe(Feed)` → normalized event stream; per-venue `SymbolMap`.
- `AccountState` — `positions` / `balances` / `open_orders` + `stream_account_events`.
- A **normalized order model** `{symbol, side, qty, price?, type, tif, reduce_only, trigger?,
  cloid}`, mapped per venue.
- A per-adapter **capability descriptor** (supported order types/TIFs, tick/lot, max batch,
  native-market?, rate-limit model) so the router rejects impossible requests early.
- Auth behind a `Credentials`/`Signer` enum (CEX HMAC vs DEX wallet/EIP-712).

We **wrap, not adopt**, any vendor SDK — Hyperliquid SDK types stay inside the Hyperliquid
adapter, behind our traits.

## Consequences

- **+** Adding a venue = writing one adapter; the core, strategies, and other adapters don't
  change. This is the concrete test of "it's a Layer."
- **+** The same traits let a **simulated/historical** venue be an adapter too → one core runs
  backtest, sandbox, and live ([01](../01-architecture.md), [07](../07-parity-and-testing.md)).
- **+** Capability descriptors let strategies stay generic and fail fast on unsupported combos.
- **−** The normalized model is a lowest-useful-denominator: venue-exclusive features (e.g.
  Hyperliquid TWAP action, HIP-3) need explicit extension points rather than the common path.
- **−** Up-front design cost before the abstraction pays off (only one venue exists today) —
  accepted because retrofitting an abstraction after coupling is far costlier.
- **OPEN:** exact trait signatures, error taxonomy, and the venue-extension mechanism —
  finalized in Phase 1/3.

See [04 — Provider abstraction](../04-provider-abstraction.md).
