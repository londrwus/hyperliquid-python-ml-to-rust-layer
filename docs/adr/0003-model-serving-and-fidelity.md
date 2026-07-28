# ADR-0003 — Model serving & fidelity strategy

**Status:** Accepted · **Date:** 2026-07-12

## Context

The layer must run tree-based, classical, *and* neural-net models (per the project scope)
without losing strategy quality. Fidelity differs sharply by family
([research/ml-inference-in-rust.md](../research/ml-inference-in-rust.md)):

- Tree ensembles (XGBoost/LightGBM) and classical models reproduce **numerically ~exactly** in
  Rust (deterministic threshold traversal / simple math).
- Neural nets **never** reproduce bit-for-bit across runtimes (ONNX doesn't encode op ordering;
  FP is non-associative) — best achievable is FP32 tolerance (~1e-5..1e-6).

Under Boundary B, inference runs in **Python** at serving time today, so live == research by
construction. This ADR governs (a) how a model becomes a versioned artifact, and (b) the Rust
serving path for the future Boundary-A migration and for classical models that may run natively.

## Decision

1. **Artifact export & registry.** `axon.models` exports every model to a portable, versioned
   artifact: **native JSON** for XGBoost/LightGBM; **ONNX (FP32, `opt_level=0`)** for
   sklearn/NN. Each artifact carries version, input/output schema, feature-spec reference, and
   git SHA. Artifacts are immutable; the running system records which model version produced
   which signal.
2. **Rust serving backends (Boundary A / native path), chosen per family:**
   - XGBoost → `xgboost-ars` (pure Rust, validated) or `ort`.
   - LightGBM → `lightgbm3-rs` (exact) or a `lleaves`-compiled lib.
   - sklearn/NN → ONNX via **`ort`** (perf) or **`tract`** (pure-Rust, portable, most
     deterministic); `tch-rs` only when a Torch model must match Python as closely as possible.
3. **Mandatory model-parity gate (CI).** Dump N-thousand research inputs; run Python vs Rust;
   assert **exact/near-exact for trees**, **max_abs_diff < ε (start 1e-5) for NN**, and
   **decision invariance** (no discretized trade flips). A model can't serve until it passes.
4. **Feature parity is treated separately and as the harder problem** — see
   [03 — ML fidelity & feature parity](../03-ml-fidelity-and-features.md).

## Consequences

- **+** "No quality loss" becomes a *tested gate*, not a hope. Trees get exactness; NNs get a
  defensible, bounded, decision-invariant equivalence.
- **+** Model family is abstracted behind the registry + a backend choice, so mixing tree/NN/
  classical strategies is routine.
- **+** Immutable versioned artifacts make every live decision reproducible offline (enables
  shadow trading + audits, [07](../07-parity-and-testing.md)).
- **−** Maintaining multiple Rust inference backends adds surface area (accepted: each is
  best-in-class for its family).
- **−** The parity gate needs representative input corpora per strategy (a real, ongoing effort).
- Depends on [ADR-0005](0005-fp32-no-quantization.md) (FP32) to keep NN drift within tolerance.
