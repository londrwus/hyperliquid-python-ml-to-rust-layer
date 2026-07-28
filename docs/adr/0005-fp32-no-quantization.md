# ADR-0005 — Keep models FP32; no quantization

**Status:** Accepted · **Date:** 2026-07-12

## Context

Quantization (FP32 → INT8/FP16) is a standard inference-speed technique and was raised as a
candidate for this project. The research is clear on the trade
([research/ml-inference-in-rust.md](../research/ml-inference-in-rust.md)):

- INT8 buys ~2–2.5× inference speed for ~0.5–2% accuracy loss.
- FP16 is *near*-lossless but can **silently downcast and flip a prediction across a decision
  threshold** (a documented ONNX Runtime / CoreML footgun) — that's real quality loss, not
  rounding.
- Meanwhile, inference is nowhere near the bottleneck: the Hyperliquid round-trip is
  **milliseconds**, and even FP32 tree/ONNX inference is **sub-millisecond**. The speedup
  quantization offers buys time we don't need.

The project's prime directive is **no strategy-quality loss**. Trading decisions are
discretized (enter/exit/size), so a prediction that changes near a threshold changes the
*trade*.

## Decision

**Serve all models in FP32, end-to-end. No INT8 quantization, no FP16, no automatic mixed
precision (AMP).** Additionally, disable aggressive graph optimizations (`opt_level=0`) when
structural numerical match to the Python model is required, and **round-trip test for silent
FP16 downcasting** in any export path.

## Consequences

- **+** Maximum achievable fidelity: trees stay exact; NNs stay within tight FP32 tolerance —
  supports the decision-invariance guarantee in [ADR-0003](0003-model-serving-and-fidelity.md).
- **+** One fewer variable when debugging live-vs-research divergence.
- **−** We forgo a 2–2.5× inference speedup — accepted, because inference isn't the bottleneck
  (the wire is) and quality is non-negotiable.
- **Revisit only if** a future strategy's inference genuinely becomes latency-critical *and*
  runs in a colocated Boundary-A path *and* a quantized variant passes the full model-parity +
  decision-invariance gate. Until all three hold, FP32 stands.
