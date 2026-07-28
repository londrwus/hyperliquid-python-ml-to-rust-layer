# Research briefings

Grounding research (conducted mid-2026) behind the architecture. Each briefing is a
condensed, decision-oriented reference with source links. These are **inputs to the design**,
not the design itself — the design lives in the numbered docs and the [ADRs](../adr/README.md).

| Briefing | Feeds |
|----------|-------|
| [ML inference in Rust](ml-inference-in-rust.md) | [03 — ML fidelity](../03-ml-fidelity-and-features.md), [ADR-0003](../adr/0003-model-serving-and-fidelity.md), [ADR-0005](../adr/0005-fp32-no-quantization.md) |
| [Python↔Rust IPC](python-rust-ipc.md) | [02 — Boundary](../02-python-rust-boundary.md), [05 — Latency](../05-latency-model.md), [ADR-0002](../adr/0002-python-rust-boundary.md) |
| [Hyperliquid execution](hyperliquid-execution.md) | [04 — Providers](../04-provider-abstraction.md), [05 — Latency](../05-latency-model.md) |
| [Hybrid quant architecture](hybrid-quant-architecture.md) | [01 — Architecture](../01-architecture.md), [06 — Strategy](../06-strategy-contract.md), [07 — Parity](../07-parity-and-testing.md) |
| [Compute offload via hwsched/Modal](compute-offload-hwsched.md) | [08 — Roadmap](../08-roadmap.md) Phase 5, [07 — Parity](../07-parity-and-testing.md) |

> Facts are current as of mid-2026 and should be re-verified before implementation —
> crate versions, SDK details, and Hyperliquid's network characteristics evolve.
