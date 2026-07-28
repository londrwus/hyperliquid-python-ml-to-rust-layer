# 01 — Architecture

Axon is a **hexagonal (ports-and-adapters)** system split into two planes that meet at a
**shared-memory boundary**.

## The two planes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH PLANE  (Python)                            │
│                                                                            │
│   data → features (SOURCE OF TRUTH) → model train → backtest → export      │
│                                                                            │
│   live: Strategy computes features + runs inference → emits SIGNAL         │
└───────────────────────────────┬────────────────────────────────────────────┘
                                 │
        OFFLINE  ┌───────────────┴───────────────┐  ONLINE
   model artifact (ONNX/native) │        signals via SHARED-MEMORY ring
   + feature spec + config      │        (fixed-layout, lock-free SPSC)
                                 │
┌───────────────────────────────▼────────────────────────────────────────────┐
│                        EXECUTION PLANE  (Rust core)                        │
│                                                                            │
│    ┌───────────┐        ┌───────────────── message bus ──────────────┐    │
│    │ marketdata │──tick─▶│                                            │    │
│    │ + orderbook│        │   ┌──────────┐   ┌──────────┐   ┌────────┐ │    │
│    └─────▲──────┘        │   │ strategy │──▶│  risk    │──▶│  exec  │ │    │
│          │               │   │ adapter  │   │  engine  │   │ engine │ │    │
│   ┌──────┴──────┐        │   └──────────┘   └──────────┘   └───┬────┘ │    │
│   │  provider   │◀───────┴──────────────────────────────────────┘      │    │
│   │  adapter    │  (subscribe market data / submit + cancel orders)     │    │
│   └──────┬──────┘                                                       │    │
└──────────┼─────────────────────────────────────────────────────────────────┘
           │
    ┌──────▼───────┐
    │   VENUE      │   Hyperliquid (adapter #1), then Binance / Bybit / …
    │ (WS + REST)  │
    └──────────────┘
```

- **Research plane (Python):** everything that benefits from the Python ecosystem and does
  *not* sit on the microsecond path — training, backtesting, feature definitions, and live
  signal generation (feature computation + model inference).
- **Execution plane (Rust):** the deterministic core that must be fast, jitter-controlled,
  and correct — market data, order book, risk, order lifecycle, venue I/O.
- **Boundary:** offline, Python hands Rust a **model artifact + feature spec + config**;
  online, Python streams **signals** through a **shared-memory ring**. See
  [02](02-python-rust-boundary.md).

## The two contracts (this is what makes it a "Layer")

A hexagonal core has ports on two sides:

- **Inward port — the Strategy contract.** Strategies implement a fixed set of event
  callbacks (`on_tick`, `on_bar`, `on_fill`, …) and emit orders/signals through provided
  facades. The engine calls the strategy; the strategy never reaches into the engine's
  internals. Strategies can be authored in **Python** (via the signal bus) or, later, in
  **Rust** (compiled plugin) against the *same* semantics. See [06](06-strategy-contract.md).
- **Outward port — the Provider contract.** Venues and data feeds are **adapters** behind
  one interface (`place_order`, `cancel`, `subscribe`, `account_state`). Swapping live venue
  ↔ simulator, or Hyperliquid ↔ Binance, is swapping an adapter — the core and the strategy
  don't change. See [04](04-provider-abstraction.md).

> **"A Layer" = strategy-agnostic (inward port) + provider-agnostic (outward port).** That's
> the entire definition.

## The Rust core (planned crates)

Modeled on the convergent design of production Rust engines (NautilusTrader, barter-rs) — a
**single-threaded deterministic core** driven by a **message bus**, with async I/O only at
the edges. See [research/hybrid-quant-architecture.md](research/hybrid-quant-architecture.md).

| Crate | Responsibility |
|-------|----------------|
| `axon-core` | Domain types (Order, Fill, Position, Instrument), the message bus, the clock, the in-memory cache, and the deterministic event loop. Knows nothing about venues or strategies. |
| `axon-marketdata` | Ingest market data (from a provider adapter), maintain order books, publish normalized tick/bar/book events onto the bus. |
| `axon-execution` | Order manager (lifecycle, reconciliation via fills/updates) + execution engine (routing to the provider). |
| `axon-risk` | Pre-trade risk checks on the hot path (position limits, reduce-only, exposure, rate/nonce budgets). |
| `axon-providers` | The provider trait(s) + a registry + capability descriptors. |
| `axon-provider-hyperliquid` | Hyperliquid adapter: signing, nonces, order encoding, WS/REST. See [05-hyperliquid](research/hyperliquid-execution.md). |
| `axon-ipc` | The shared-memory signal bus (Python↔Rust boundary): ring buffer, fixed-layout records, framing. |
| `axon-model` | *(Boundary-A path)* native model inference (tree/ONNX/torch) + feature runtime, for when inference moves into Rust. |
| `axon-strategy` | The Rust-side strategy plugin contract (for strategies dropped down to Rust). |
| `axon-runtime` | The binary. Wires the crates together, loads config, owns the process lifecycle, dead-man's-switch, health. |

Design rules for the core:
- **Single-threaded deterministic loop** for trading logic → reproducible event ordering,
  no lock contention, and the property that lets backtest and live share one code path.
  (LMAX Disruptor / ring-buffer inspiration.)
- **Message passing, not shared mutable state.** Components consume events off the bus; they
  don't hold locks on each other.
- **Async (Tokio) only at the edges** — network I/O, persistence, the venue adapters — which
  hand results to the core over channels. Keep the hot loop off the async runtime.

## The Python package (planned)

| Module | Responsibility |
|--------|----------------|
| `axon.features` | The **feature library** — the single source of truth for feature logic. Used by research *and* referenced by the parity harness. |
| `axon.strategy` | Strategy base class / contract; lifecycle + event handlers mirroring the Rust contract. |
| `axon.signals` | The signal publisher — writes fixed-layout signals into the shared-memory ring. |
| `axon.models` | Model training helpers + **export** (to ONNX / native tree format) + a model registry with versioning. |
| `axon.backtest` | Backtest harness that reuses the core's event semantics for parity. |
| `axon.parity` | Golden-test + parity-monitor tooling (see [07](07-parity-and-testing.md)). |

## Data & control flow (live, Boundary B)

1. **Market data in:** venue WS → `axon-provider-*` → `axon-marketdata` normalizes → book
   updates + tick/bar events on the bus.
2. **To Python (optional):** if the strategy needs raw market data for features, the core
   publishes the needed slice to Python via a **market-data shared-memory ring** (Rust→Py).
3. **Signal generation (Python):** `axon.features` computes features; the model runs
   inference; `axon.signals` writes a **signal** (target position / order intent) into the
   **signal ring** (Py→Rust).
4. **Strategy adapter (Rust):** reads the signal off the ring, turns it into concrete order
   intents, emits them on the bus.
5. **Risk:** `axon-risk` validates every intent (fast, on the hot path).
6. **Execution:** `axon-execution` constructs, signs (via the adapter), and submits orders;
   reconciles fills/updates back onto the bus (and back to Python for state).

## Environments — one core, three modes

The same core components run in three environments, differing only by which adapters are
plugged into the outward port:

- **Backtest:** historical-data adapter + simulated-venue adapter.
- **Sandbox / paper:** live-data adapter + Hyperliquid **testnet** adapter.
- **Live:** live-data adapter + Hyperliquid **mainnet** adapter.

Parity comes from this being *the same code*, not a reimplementation. See
[07](07-parity-and-testing.md).

## Key references

- Hexagonal core, message bus, single-threaded determinism, strategy-as-plugin, and the
  three-environment model are lifted from **NautilusTrader**'s proven design (Python + Rust
  via PyO3). See [research/hybrid-quant-architecture.md](research/hybrid-quant-architecture.md).
