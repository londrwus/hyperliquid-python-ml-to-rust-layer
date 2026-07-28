# 00 — Vision & Scope

## The problem

Quant/ML strategies are best *researched* in Python — the data, ML, and backtesting
ecosystem is unmatched. But Python is a poor host for the *execution* path: the GIL, GC
pauses, interpreter overhead, and unpredictable tail latency make it hard to run a tight,
deterministic, jitter-controlled trading loop.

The usual "solution" — rewrite the strategy in Rust/C++ — is expensive, slow, and
dangerous: the rewritten strategy is a *different* strategy until proven otherwise, and
proving it is a research project in itself.

## The thesis

**Don't rewrite the strategy. Split the system.**

- Keep **research, ML, feature engineering, and signal generation in Python**. The strategy
  stays exactly what the researcher wrote and backtested.
- Give a **Rust core** the latency-, jitter-, and correctness-critical execution path:
  market data, order book, risk, order construction, venue networking.
- Connect them through **shared memory**, so the handoff costs ~100ns–1µs and preserves the
  strategy's behaviour bit-for-bit at the signal boundary.

This is the **Axon layer**: it carries the signal from the Python "brain" to the Rust
"muscle" without distorting it.

## Goals

1. **Zero strategy-quality loss.** The live strategy behaves identically to the researched
   one. Guaranteed by construction (FP32, no quantization; shared feature logic) and by test
   (golden replay, parity monitors). See [03](03-ml-fidelity-and-features.md).
2. **Provider-agnostic.** Hyperliquid first, but any CEX/DEX plugs in behind one contract.
   See [04](04-provider-abstraction.md).
3. **Strategy-agnostic.** Strategies are plugins against a fixed contract; the engine knows
   nothing about any specific strategy. See [06](06-strategy-contract.md).
4. **Fast where it counts, without special hardware.** No FPGAs, no kernel bypass required
   for v1. Use shared memory, a deterministic single-threaded core, and good engineering.
5. **Mixed-frequency, evolvable.** Correct and robust at mid-frequency (seconds–minutes)
   now; architected so the hot core can migrate toward HFT (sub-second, colocated) later
   *without a rewrite*.
6. **Backtest/live parity by construction.** The same Rust core semantics drive both.

## Non-goals (for now)

- **Not** an HFT/ultra-low-latency system on day one. We build the *headroom*, not the
  colocation. (See [05](05-latency-model.md) for what changes if/when we go there.)
- **Not** a Python-in-the-hot-loop design. Python generates signals *off* the critical path;
  it never sits inside the per-order Rust loop (that's Boundary C, which we rejected — see
  [ADR-0002](adr/0002-python-rust-boundary.md)).
- **Not** a model-quantization project. Quantization trades prediction quality for speed we
  don't need (the wire dominates). We stay FP32. See [ADR-0005](adr/0005-fp32-no-quantization.md).
- **Not** a strategy library. This is the *layer*; strategies live on top of it.
- **Not** a backtesting framework rewrite. We reuse the core's event semantics for
  backtests, but polished research tooling stays in Python.

## What "success" looks like

- A researcher writes a strategy + feature code in Python, exports a model, and the **same**
  strategy runs live through the Rust layer on Hyperliquid with signals that match the
  backtest within a defined numerical tolerance (and never flips a discretized trade
  decision vs. the reference).
- Adding a second venue (e.g. Binance) is writing **one adapter**, not touching the core or
  the strategy.
- The path from "mid-frequency, runs anywhere" to "colocated, sub-second" is a series of
  optimizations to the Rust core and IPC — not an architectural change.

## Audience & operating model

- **Researchers** work in Python: strategies, features, models, backtests. They rarely touch
  Rust.
- **Engineers** own the Rust core and the venue adapters.
- The **contract between them** (signal schema, feature spec, strategy callbacks) is
  explicit, versioned, and tested — this is the whole point of the layer.
