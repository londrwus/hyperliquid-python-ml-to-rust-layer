# Axon — Documentation

This is the design brain of the project. These documents were the whole deliverable in Phase 0;
there is now a great deal of code behind them, and they are maintained to keep saying what is
true of it. Read them in order, or jump to what you need.

One of them is load-bearing in a way the others are not. [08 — Roadmap](08-roadmap.md) is where
*proven live*, *built and offline-verified*, and *written* are kept apart — if you only read one
page before touching anything, read that one.

## Reading order

| # | Document | What it answers |
|---|----------|-----------------|
| 00 | [Vision & scope](00-vision-and-scope.md) | What we're building, why, and what we're deliberately *not* building. |
| 01 | [Architecture](01-architecture.md) | The two-plane / hexagonal design, components, and how they talk. |
| 02 | [Python↔Rust boundary](02-python-rust-boundary.md) | Where the languages meet, the shared-memory design, and the migration path. |
| 03 | [ML fidelity & feature parity](03-ml-fidelity-and-features.md) | How "no quality loss" is actually guaranteed. Model export paths per model type. |
| 04 | [Provider abstraction](04-provider-abstraction.md) | The venue-agnostic contract; Hyperliquid as one adapter. |
| 05 | [Latency & performance model](05-latency-model.md) | Where time really goes; latency budgets; HFT headroom. |
| 06 | [Strategy contract](06-strategy-contract.md) | How a strategy plugs in (Python or Rust), config, and state. |
| 07 | [Parity & testing](07-parity-and-testing.md) | The validation ladder: golden replay → shadow → live. |
| 08 | [Roadmap](08-roadmap.md) | Phased plan from scaffold to live trading. |
|  — | [Glossary](glossary.md) | Terms of art used throughout. |

## Reference material

- **[`research/`](research/)** — the grounding research briefings (2026) that the design is
  built on: ML inference in Rust, Python↔Rust IPC, Hyperliquid's execution surface, and
  hybrid quant architecture patterns. Each includes source links.
- **[`adr/`](adr/)** — Architecture Decision Records. Every significant, hard-to-reverse
  choice is recorded here with its context and consequences.
> The operational runbooks — the ones naming a particular machine, account and session — are
> kept out of this public tree. Everything they were written against is in the ADRs:
> [ADR-0037](adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md) is the one
> to read on getting an account flat, what the loss switch means when it has tripped, and why
> "the strategy asked to be flat" and "the account is flat" are different claims. It exists
> because on 2026-07-27 the documented exit path failed three different ways in one afternoon.

## Conventions

- **Codename:** the layer is called **Axon** throughout. It's provisional — rename freely.
- **"Research plane" = Python. "Execution plane" = Rust.** These terms are used everywhere.
- **Boundary A / B / C** refer to the three Python↔Rust split options analysed in
  [ADR-0002](adr/0002-python-rust-boundary.md). We chose **B**, with a path to **A**.
- Docs state **decisions**, not just options. Where something is still open, it's marked
  `OPEN:` so it's greppable.
