# 06 — Strategy Contract

The inward port of the hexagon. This is how a strategy plugs into Axon — the seam that makes
the engine **strategy-agnostic**. A strategy is a *plugin against a fixed contract*, not code
that reaches into the engine.

## Principles (borrowed from NautilusTrader's proven design)

1. **The engine calls you; you never call the engine's internals.** Strategies implement
   event callbacks and emit intents through *provided facades* (an order factory, a submit
   function, a read-only cache, a clock). No direct access to the bus, the book internals, or
   the adapters.
2. **Same contract, two languages.** A strategy can be authored in **Python** (Boundary B,
   via the signal bus) or, later, in **Rust** (compiled plugin), against the *same*
   semantics. Researchers write Python; latency-critical strategies can drop to Rust without
   the engine noticing.
3. **Config is a serializable object, not constructor args.** This is what later enables
   distributed backtests and remote live runs from one definition.
4. **State is explicit and restorable.** `on_save`/`on_load` hooks let a strategy persist and
   restore state across restarts — essential for a stateful, long-running layer.

## The lifecycle + event callbacks (planned)

```
lifecycle:   on_start()   on_stop()   on_reset()   on_save() -> State   on_load(State)

data:        on_tick(Tick)        on_bar(Bar)          on_book(BookDelta)
             on_trade(Trade)      on_quote(Quote)

execution:   on_fill(Fill)        on_order_update(OrderUpdate)
             on_position(Position)

timers:      on_timer(TimerEvent)
```

A strategy overrides only what it needs. The engine guarantees ordering and delivers events
through the deterministic loop.

## How a Python strategy actually runs (Boundary B)

```
  ┌─────────────────────── Python (axon.strategy) ────────────────────────┐
  │  class MyStrategy(Strategy):                                            │
  │      def on_bar(self, bar):                                             │
  │          x = axon.features.compute(self.state, bar)  # SOURCE OF TRUTH  │
  │          y = self.model.predict(x)                   # FP32 inference   │
  │          self.emit(Signal(symbol=..., target_qty=..., urgency=...))     │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                      │  axon.signals → shared-memory ring
  ┌──────────────────────────────────▼─────────────────────────────────────┐
  │  Rust strategy adapter: read Signal → order intents → risk → execution   │
  └──────────────────────────────────────────────────────────────────────────┘
```

The Python strategy never talks to the venue, never manages orders, never touches nonces or
signing. It computes features, runs the model, and emits a **Signal**. That's the whole job —
and it's identical to what runs in the backtest.

## The `Signal` — what a strategy emits

Two candidate shapes (finalize in [`contracts/`](../contracts/README.md)):

- **Target-position signal** *(preferred default):*
  ```
  Signal { symbol, target_qty, urgency, price_band?, ttl, model_version, seq }
  ```
  The strategy declares *what position it wants*; the Rust execution engine decides *how* to
  get there (passive/aggressive, slicing, cancel/replace). Robust to a missed/late signal
  (the target is still valid), and keeps execution logic in Rust where it belongs.

- **Explicit order-intent signal** *(when the strategy must control the order):*
  ```
  Signal { symbol, side, qty, type, tif, price?, reduce_only, cloid, model_version, seq }
  ```
  More control for the strategy, more coupling, less resilience to missed signals.

Both carry `model_version` + `seq` so every live decision is **reproducible offline** (which
model, which sequence) — the foundation for shadow trading and audits ([07](07-parity-and-testing.md)).

### Several strategies, one account

`seq` is **per producer**, and that decides the shape of a multi-strategy session
([ADR-0038](adr/0038-many-strategies-one-account.md)): each strategy gets its own ring, because
an SPSC ring has one writer and two sequences interleaved into a stream the reader validates as
one means every record of the loser is refused as a replay. Which strategy a record came from is
therefore a property of the *ring*, declared in config, rather than a field on the 64-byte
record.

Their targets then **add**. The venue holds one position per instrument and a target-position
signal is a claim on part of it, so two strategies trading BTC are two claims on one position —
`axon_strategy::TargetBook` sums them and the engine plans the delta to the sum. The rule this
replaces, *newest signal per symbol wins*, is correct with one producer and silently wrong with
two: both strategies' counters climb, both believe they are positioned, and the account holds
whichever spoke last.

Netting is opt-in. Two producers pointed at one instrument is far more often a copy-pasted
config than a decision, so the default refuses the overlap at startup and `overlap = "net"` is
one declared word.

> `OPEN:` default to target-position signals, and support explicit intents as an opt-in for
> strategies that need order-level control. Confirm during Phase 1 contract design.

## Configuration

A strategy ships a serializable config (parameters, symbol universe, risk limits, model
reference). Example fields:
```
StrategyConfig {
  name, version,
  symbols: [...],
  model_ref: { registry_id, version },
  feature_spec_ref,
  risk: { max_position, max_notional, max_leverage, ... },
  params: { ... strategy-specific ... },
}
```
Risk limits in the config are enforced by `axon-risk` on the hot path — the strategy can't
bypass them.

## State & warm restart

`on_save()` returns a serializable state blob; `on_load(state)` restores it. The runtime
persists it so a restart resumes cleanly (open positions, rolling feature windows, etc.).

## The Rust-side contract (later)

When a strategy is dropped to Rust for latency, it implements the same callbacks as a trait
(a `nautilus_strategy!`-style macro can generate the boilerplate). Same semantics, same
config object, same `Signal` output — so a strategy can be promoted from Python to Rust
without the engine, the adapters, or the tests changing shape.

## The contract's job, restated

- **Strategy-agnostic engine:** the core only knows "a strategy consumes events and emits
  signals." It never knows *which* strategy.
- **Language-flexible:** Python for research velocity, Rust for latency — one contract.
- **Parity-friendly:** because a strategy is just `events → Signal`, the *same* strategy
  object runs in backtest, sandbox, and live.
