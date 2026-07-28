# 04 — Provider Abstraction

The outward port of the hexagon. This is what makes Axon **provider-agnostic**: Hyperliquid,
Binance, Bybit, and other DEXs all sit behind one set of traits, with venue quirks pushed to
the *edges* (the adapters), never leaking into the core or the strategy.

## The core insight

Above a clean line, everything generalizes: **order intent, book state, position/risk,
lifecycle events.** Below that line, venues differ wildly — and that's the adapter's job to
absorb.

```
        ┌──────────────────── generalizes cleanly ────────────────────┐
        │  order intent · order book · positions/risk · fills/updates   │
        └───────────────────────────┬──────────────────────────────────┘
                                     │  (traits — the provider contract)
     ┌───────────────────────────────┼───────────────────────────────┐
     │  Hyperliquid adapter │  Binance adapter │  Bybit adapter │ …    │
     └───────────────────────────────┴───────────────────────────────┘
        ▲ venue quirks live *only* here ▲
```

## The traits (planned)

### `ExecutionClient` (async)
```
place_order(OrderRequest)        -> OrderAck
place_batch(Vec<OrderRequest>)   -> Vec<OrderAck>
cancel(OrderId | Cloid)          -> CancelAck
cancel_all()                     -> ...
modify(OrderId, ...)             -> ...
```
Returns venue-agnostic acks. A local `cloid` (client order id) makes every request
idempotent and reconcilable.

### `MarketData` (async)
```
subscribe(Feed) -> Stream<MarketEvent>
   Feed ∈ { L2Book, Trades, Bbo, Candles(interval), Ticker }
```
Emits **normalized** events. Symbols are normalized through a per-venue `SymbolMap`
(canonical `BTC-PERP` ↔ Hyperliquid asset-index ↔ Binance `BTCUSDT`).

> **Realized in Phase 2 ([ADR-0008](adr/0008-market-data-bus-and-ws-ingest.md)).** The
> `-> Stream<MarketEvent>` above is the *concept*; the transport was left `OPEN` and is now
> settled. Instead of returning a stream from `subscribe`, an adapter (running on tokio at
> the edge) **publishes** normalized `MarketEvent`s onto `axon-core`'s in-process **bus** — a
> bounded `crossbeam` channel — which the synchronous deterministic core drains. The
> `MarketData` port keeps `subscribe`/`unsubscribe` for feed registration; event delivery is
> wired at construction via an `EventSender`. `MarketEvent`/`Bbo`/`Trade`/`BookSnapshot` now
> live in `axon-core::market` and are re-exported here, so the port and every adapter share
> one vocabulary with no dependency cycle.

### `AccountState` (async)
```
positions()  ·  balances()  ·  open_orders()
stream_account_events() -> Stream<AccountEvent>   // fills, order updates, funding
```
Unifies Hyperliquid `userFills`/`orderUpdates` with CEX user-data streams.

### The normalized order model
```
Order {
  symbol, side, qty,
  price?: Decimal,
  type: Market | Limit,
  tif:  GTC | IOC | PostOnly | FOK,
  reduce_only: bool,
  trigger?: { price, kind: TP | SL, market: bool },
  cloid: u128,
}
```
Each adapter maps this to its venue. Unsupported combinations surface as a **typed
capability error**, early, before hitting the wire.

### Capability descriptor (per adapter)
Each adapter declares what it supports so the router can reject impossible requests up front
and the strategy layer stays generic:
```
Capabilities {
  order_types, tifs,
  max_batch,             // Hyperliquid = 20
  native_market_orders,  // Hyperliquid = false (synthetic IOC-at-slippage)
  rate_limit_model,      // weight/IP (CEX) vs volume-gated (Hyperliquid)
  ...
}
```

Tick size and lot size are **not** here, and that is a decision rather than an omission
([ADR-0025](adr/0025-instrument-precision-and-rounding.md)). They are per *instrument*, not per
venue — BTC and a $0.003 alt on the same venue have different grids — and they arrive over the
network at startup, so putting them here would stop `Capabilities` being `&'static` and
const-constructible. They live in an `InstrumentTable` each adapter populates from its own
metadata endpoint:
```
InstrumentSpec { symbol_id, price: PriceGrid, size: SizeGrid, min_notional }
PriceGrid      { increment, sig_figs }   // a CEX sets one field; Hyperliquid sets both
SizeGrid       { increment }             // szDecimals: n IS 10^-n
```
The planner rounds through it in the direction that preserves the order's intent; the encoder
**refuses** anything not already on the grid, so the one place that decides and the one place
that cannot be bypassed are different places.

### Auth abstraction — one `Signer`/`Credentials` seam
```
Credentials =
  | ApiKey { key, secret }          // CEX: HMAC
  | Wallet { signer }               // DEX: EIP-712 / agent-wallet signing (Hyperliquid)
```
Same `authenticate()` seam; the adapter picks the scheme. Signing sits behind a `Signer`
trait so we can support agent wallets / KMS / remote signers.

## CEX ↔ DEX divergences the adapter must absorb

These are the concrete things that must **not** leak upward (Hyperliquid examples):

| Concern | Hyperliquid (DEX) | Typical CEX | Absorbed by |
|---|---|---|---|
| Auth | Wallet signing, **no API keys**; agent wallets; EIP-712 + phantom-agent/msgpack | HMAC API key/secret | `Signer` / `Credentials` |
| Market orders | **Synthetic** — IOC limit at deep slippage | Native `MARKET` | Order mapper |
| Nonces | **ms-timestamp windows**, 100-nonce tracker per address | Sequential counter or none | Adapter nonce manager |
| Rate limits | **Volume-gated** (1 req / 1 USDC), cancels get a higher cap | Weight/IP limits | Per-venue rate governor |
| Settlement | **On-chain, block-timed (~0.2s)** | In-memory microseconds | Latency model / execution engine |
| Batch size | 20 orders/call | Varies | Capability descriptor |

Everything above that table — the order intent, the book, positions, lifecycle — is shared
code.

## Hyperliquid adapter specifics

Details in [research/hyperliquid-execution.md](research/hyperliquid-execution.md). Key
implementation notes:
- **Endpoints:** REST `POST /exchange` (trading), `POST /info` (state), WS `/ws` (data +
  user streams). Testnet mirror for sandbox.
- **Signing:** two schemes — L1 actions (phantom-agent, msgpack hash, chainId 1337) and
  user-signed admin actions (direct EIP-712). Mixing them is the #1 "invalid signature"
  cause. Isolate both behind the `Signer`.
- **Agent wallets:** approve a separate key to sign L1 actions so the hot key never holds
  funds. One agent wallet per trading process (shared signer ⇒ shared nonce tracker).
- **In-block priority:** cancel → post-only → GTC → IOC. The execution engine should exploit
  this (pull stale quotes before same-block takers).
- **Resilience:** WS heartbeat + auto-reconnect + resubscribe; REST resync on gaps;
  `schedule_cancel` dead-man's-switch to auto-flatten if the process dies.

### SDK stance
Don't adopt any SDK's types directly — **wrap behind our own trait**. Candidate reference
implementations for the Hyperliquid adapter: the official `hyperliquid_rust_sdk` (correctness
cross-check for signing), plus community SDKs `hypersdk` (docs-endorsed) or
`hyperliquid-sdk-rs` (perf, `fastwebsockets`). Note the official SDK still uses the deprecated
`ethers-rs` (successor is `alloy`) — a signing-only concern for a long-lived layer.

## Adding a venue = writing one adapter

The test of this design: onboarding Binance is implementing `ExecutionClient` +
`MarketData` + `AccountState` + `Capabilities` + a `SymbolMap` for Binance — and touching
**nothing** in `axon-core`, the strategy, or any other adapter.
