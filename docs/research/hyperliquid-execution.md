# Research — Hyperliquid execution surface (mid-2026)

For a provider-agnostic Rust execution layer with Hyperliquid as adapter #1. Verify signing
details against live docs before production.

## Official Rust SDK
- `hyperliquid-dex/hyperliquid-rust-sdk` (crate `hyperliquid_rust_sdk`, ~v0.6.0, Jun 2026),
  MIT. Reference-grade, not batteries-included.
- Shape: `ExchangeClient` (order/bulk_order/cancel/modify/market_open/market_close/
  schedule_cancel/leverage/transfers/agent approval), `InfoClient` (user_state/open_orders/
  l2_book/recent_trades/funding + WS `subscribe`), order builders (`ClientOrderRequest`,
  `ClientOrder`, `BulkOrder`).
- Deps: `ethers 2.0` (⚠️ deprecated upstream; successor `alloy`), `tokio`, `reqwest`,
  `tokio-tungstenite`.
- **Community SDKs** (often better ergonomics/perf): `infinitefield/hypersdk` (docs-endorsed;
  full HyperCore, EIP-712, tick rounding, HIP-3, HyperEVM), `lhermoso/hyperliquid-sdk-rs`
  (perf fork; `fastwebsockets` ~3–4× vs tungstenite; typed errors; MEV-builder),
  `elijahhampton/rhyperliquid` (correctness, `rust_decimal`), `quicknode-hyperliquid-sdk`
  (+gRPC streams).
- **Stance:** wrap behind our own trait; don't adopt SDK types directly. Use `hypersdk` or
  `hyperliquid-sdk-rs` as the initial adapter impl; keep the official SDK as a signing
  cross-check.

## API characteristics
- Endpoints: REST `POST /exchange` (trading), `POST /info` (state), WS `/ws` (data + user
  streams; can also send actions). Testnet: `api.hyperliquid-testnet.xyz`.
- Order types: limit with TIF **GTC / IOC / ALO (post-only)**. **Market = IOC limit at deep
  in-the-money price** (SDK default 5% slippage) — no distinct market type on the wire.
  Trigger orders (TP/SL) via `t.trigger` (`normalTpsl`/`positionTpsl` OCO grouping).
  `reduceOnly` flag. **TWAP** is a separate action (30s suborders, 3% max slippage, catch-up
  ≤3× size, returns `twapId`). Scale orders.
- Order encoding: compact keys `a`(asset idx), `b`(isBuy), `p`(px), `s`(sz), `r`(reduceOnly),
  `t`(type), `c`(cloid, optional 128-bit hex). Perp asset = index in `meta.universe`; spot =
  `10000 + spotMeta index`. **Batch up to 20 orders/call.**
- **In-block priority:** cancel → post-only → GTC → IOC. A cancel pulls a stale quote before
  same-block takers hit it — matters for makers.
- **Auth — no API keys.** Two schemes:
  - **L1 actions** (orders/cancels/leverage): "phantom agent", msgpack action hash, chainId
    1337.
  - **User-signed admin actions** (withdraw/transfers): direct EIP-712 on
    `HyperliquidSignTransaction`. **Mixing the two schemes is the #1 "invalid signature"
    cause.**
  - **Agent (API) wallets:** approve a separate key to sign L1 actions so the hot key never
    holds funds. Subaccounts/vaults: master signs with `vaultAddress` set.
- **Nonces:** per-signer, ms timestamps; **100 highest nonces tracked per address**; new nonce
  must exceed the smallest tracked and be within `(T-2d, T+1d)`. One API wallet per trading
  process (shared signer ⇒ shared nonce tracker).
- **Rate limits:** IP-based for Info/REST; **address-based for Exchange — 1 request per 1 USDC
  cumulative volume, initial buffer 10,000 requests.** When throttled: 1 action/10s, but
  **cancels get a higher cap** (`min(limit+100000, limit*2)`) so you can always de-risk.
- **Latency:** official co-located median ~0.2s, p99 ~0.9s (request→committed). Independent
  2026 Tokyo probes measured higher (~880 ms median order-to-fill) as the network scaled to
  24 validators / 200k+ orders/sec.

## Execution / latency model
- **HyperBFT** (HotStuff-family, pipelined 4-phase). Block finality ~0.07s optimistic, <0.2s
  typical, p99 <0.5s. Submit→settle typically <0.4s.
- **No mempool / no MEV:** matching is a native (non-EVM) deterministic HyperCore op, on-chain,
  **time-priority** book — latency directly converts to fill priority.
- **Physical location:** validators in **AWS Tokyo `ap-northeast-1`**; public API fronted by
  CloudFront.
- **Colocation reality:** Tokyo box → validators in **2–3 ms**; Seoul ~72 ms, Singapore
  ~100 ms, US ~120 ms+, Europe **>200 ms**. This ~200 ms geographic gap is a real, measured
  edge on a time-priority book — no speed-bump equalization.
- **Realistic tiers:** *Non-HFT* (directional/systematic/TWAP) — run anywhere, sub-second fills
  fine, design for correctness/resilience. *HFT/MM* — colocate Tokyo, agent wallets, batch
  orders/cancels every ~0.1s, exploit cancel priority.

## Market data (WebSocket)
`wss://api.hyperliquid.xyz/ws`; `{method:"subscribe", subscription:{type, ...}}`.
- Public: `allMids`, `l2Book` (per-coin, `{px,sz,n}` levels), `trades`, `bbo` (per block ≥0.5s),
  `candle` (OHLCV), `activeAssetCtx` (mark/funding/OI).
- User (need addr): `orderUpdates`, `userFills`, `userEvents` (fills/funding/liquidation/
  nonUserCancel), `webData2`, `notification`, `userFundings`, `activeAssetData`.
- **Ingest in Rust:** `tokio-tungstenite` (or `fastwebsockets`/`yawc`); `serde_json`/`simd-json`;
  **`rust_decimal`/fixed-point, not `f64`** (px/sz sent as strings); fan out per subscription to
  bounded `tokio::mpsc`; some feeds send no initial snapshot → seed from REST `l2Book` then apply
  WS; heartbeat + auto-reconnect + resubscribe mandatory.

## What the execution layer needs (Hyperliquid specifics)
- **Signing:** both phantom-agent/msgpack (L1) and EIP-712 (admin) behind a `Signer` trait
  (agent wallets / KMS / remote signing).
- **Nonce manager:** monotonic ms clock, per-signer, tolerant of the 100-nonce window; batch
  orders+cancels on a ~100 ms tick.
- **Order lifecycle:** local `cloid` → submit → reconcile via `orderUpdates`/`userFills`;
  handle partials, resting vs IOC-canceled, trigger activation, TWAP slices.
- **Position/risk:** from `webData2`/`clearinghouseState` + fills; margin, leverage,
  liquidation px; enforce reduce-only + pre-trade limits locally.
- **Resilience:** WS reconnect+resubscribe+backoff; REST resync on gaps; `schedule_cancel`
  dead-man's-switch; idempotency via `cloid`; keep a cancel budget in reserve.

## Provider abstraction (Hyperliquid as one adapter)
Common interface: `ExecutionClient` (place/cancel/modify/batch/cancel_all → venue-agnostic
`OrderAck`), `MarketData` (`subscribe(Feed)` → normalized stream, per-venue `SymbolMap`),
`AccountState` (positions/balances/open_orders + `stream_account_events`). Normalized order
model `{symbol, side, qty, price?, type, tif, reduce_only, trigger?, cloid}`; per-adapter
capability descriptor (order types, tick/lot, batch=20, rate-limit model, native-market?).
Auth via a `Credentials`/`Signer` enum (CEX HMAC vs DEX wallet). Keep at the adapter edge: no
API keys, volume-gated limits, synthetic market orders, ms-timestamp nonces, block-timed
settlement. Everything above — order intent, book, position/risk, lifecycle — generalizes.

## Sources
github.com/hyperliquid-dex/hyperliquid-rust-sdk · docs.rs/hyperliquid_rust_sdk ·
github.com/infinitefield/hypersdk · github.com/lhermoso/hyperliquid-sdk-rs ·
hyperliquid.gitbook.io/hyperliquid-docs (exchange-endpoint, order-types, nonces-and-api-wallets,
rate-limits, websocket/subscriptions, hypercore/overview) · hyperlatency.glassnode.com/hyperliquid ·
coindesk.com (2026-03-30, Tokyo 200ms edge) · blockhead.co (2025-06-05, architecture)

**Caveats:** ~~official SDK on deprecated `ethers-rs`~~ **Update (2026-07-24):** the official
`hyperliquid-rust-sdk` has **migrated to `alloy`** (`alloy::{primitives::keccak256, sol_types,
dyn_abi}`) — Axon uses `alloy` for signing crypto accordingly ([ADR-0009](../adr/0009-hyperliquid-signing.md)).
Official 0.2s latency predates the 24-validator scale-up (independent probes show higher real fill
latency); always test signing/nonce on testnet first.
