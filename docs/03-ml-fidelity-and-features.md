# 03 — ML Fidelity & Feature Parity

The prime directive: **the live strategy must behave identically to the researched one.**
This document defines exactly how that's guaranteed. The surprising part: the *model* is the
easy bit; **feature computation** is where quality silently leaks.

## Two sources of divergence

```
   research signal  =  f_model( f_features( raw_data ) )
   live signal      =  g_model( g_features( raw_data ) )

   quality is preserved  ⟺  f_model ≈ g_model   AND   f_features ≈ g_features
                                   (model parity)        (FEATURE parity ← the hard one)
```

## Part 1 — Model parity (the tractable part)

How well a Python-trained model reproduces depends on the family:

| Model family | Rust inference path | Fidelity | Notes |
|---|---|---|---|
| **Tree ensembles** (XGBoost / LightGBM) | `xgboost-ars` / `lightgbm3-rs` (native), or ONNX via `ort` | **~Numerically exact** ✅ | Inference = deterministic threshold traversal; no float-reduction ambiguity. The cleanest "no quality loss" path. Match float32-vs-float64 threshold dtype. |
| **sklearn** (RF/GBM/linear) | `skl2onnx` → `ort` | Exact within ONNX reference | Watch float-vs-double thresholds. |
| **Neural nets, must match Python** | `tch-rs` (TorchScript, shares libtorch kernels) | **Closest possible** to Python | Heaviest deploy (ship libtorch). Not bit-exact across CPU/GPU (parallel reductions). |
| **Neural nets, light deploy** | ONNX → `ort` (fast) or `tract` (pure-Rust, portable, most deterministic) | **FP32-equivalent**, ~1e-5..1e-6, *not* bit-exact | ONNX doesn't encode op ordering; FP is non-associative. |
| **Classical / rule-based** | Native Rust | Exact (we control it) | Trivial. |

### The rules that keep model parity

1. **Stay FP32 end-to-end. No quantization, no AMP.** INT8 buys ~2–2.5× speed for ~0.5–2%
   accuracy loss — a bad trade when the wire dominates and quality is sacred. FP16 can
   silently downcast and *flip a prediction across a decision threshold* (real quality loss,
   not rounding). See [ADR-0005](adr/0005-fp32-no-quantization.md).
2. **Pin the runtime + hardware** for research and production (same ONNX Runtime execution
   provider, same CPU/GPU). Disable aggressive graph optimizations (`opt_level=0`) when you
   need structural match.
3. **ONNX is the neutral bridge** for NN/sklearn; **native format** (XGBoost/LightGBM JSON)
   for trees. `axon.models` owns export + versioning.
4. **Round-trip test for silent FP16 downcasting** — a documented ONNX Runtime / CoreML
   footgun.

### Acceptance gate (model)

Dump N thousand research inputs; run Python vs Rust; assert:
- **Trees:** exact / near-exact.
- **NN:** `max_abs_diff(logits) < ε` (start ε=1e-5).
- **Decision invariance:** *no* discretized trading decision flips vs. the reference. This
  is the check that actually protects P&L — a small logit delta that never crosses a
  threshold is harmless; one that does is a bug.

Baked into CI as a golden test (see [07](07-parity-and-testing.md)).

## Part 2 — Feature parity (the hard part — where quality actually leaks)

The model is maybe 20% of the strategy. If features are computed one way in Python research
and another way in Rust (or a different Python path) at serving time, you get
**training–serving skew** — the #1 silent-degradation failure mode in production ML.

The classic culprits: **timezone bugs, late/out-of-order events, rounding, windowing
off-by-one, NaN/null handling, staleness/freshness differences.**

### Strategy: single source of truth, then verify

**Design principle: never implement a feature twice.** Two independent implementations of
"the same" transform *is* the bug. Our approach, in order of strength:

1. **`axon.features` is the one implementation *on the serving path*.** In Boundary B, feature
   computation happens in **Python** at serving time (the strategy runs it), using the *exact
   same code* as research. So `f_features == g_features` by construction — there is only one
   function. This is a major reason B is safe, **and it is still how every live session
   runs**.
2. **When features migrate to Rust** (Boundary A, later, per-strategy): the Rust
   implementation must be validated *bit-equivalent* against `axon.features` before it's
   allowed to serve. Options to avoid a second hand-written impl:
   - Write core feature logic **once in Rust**, call it from Python via bindings for
     research (guarantees identical code path); or
   - Keep two impls but gate them behind a **mandatory parity test** on a large sample.

   > **Done, by the second route** ([ADR-0035](adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md),
   > 2026-07-26). `crates/axon-features` is a second implementation of all seventeen transforms,
   > and it exists only because the condition above is met: a *feature parity bundle* freezes a
   > spec, the inputs it reads and Python's own matrix over them, and the Rust runtime is held to
   > **bit** equality over five committed corpora of real venue data — 33 423 cells,
   > `max_abs_diff = 0e0`. The bindings route was rejected on cost, not principle: it would put a
   > compiled extension between research and every notebook.
   >
   > Read the scope before treating the prime directive as relaxed. **Nothing in the live core
   > calls that crate**; item 1 above still describes every session this repository has ever run.
   > What changed is that a strategy *could* now be promoted to Boundary A with evidence, not that
   > one has been. And bit-equivalence turned out to be achievable only because NumPy's summation
   > order was reproduced rather than approximated — the ADR is worth reading before anyone writes
   > transform number eighteen in either language.
3. **Point-in-time-correct feature assembly** for training (no lookahead leakage) — a
   research-side concern, but it defines what "correct" features even are.
4. **A live parity monitor** (the mandatory backstop): sample live feature vectors,
   recompute via the offline path, alarm on divergence. Catches the drift that slips past
   unit tests — especially timezone/late-event/staleness bugs.

> **Feature stores** (Feast/Tecton-style) guarantee *schema* parity and point-in-time
> correctness, but **not** transform parity or freshness parity if you still hand-code
> transforms twice. They're a storage/serving convenience, not a substitute for the single
> source of truth above. Adopt one later only if the operational need appears.

### Acceptance gate (features)

- Live-computed feature vectors, resampled and recomputed offline, match within tolerance.
- Drift monitored continuously (PSI/KL on feature distributions) with alarms.

## The `axon.models` export workflow (planned)

```
  train (Python)
     └─▶ export:  trees → native JSON  |  NN/sklearn → ONNX (FP32, opt_level=0)
            └─▶ attach: version, input/output schema, feature-spec ref, git SHA
                   └─▶ registry (versioned artifact)
                          └─▶ Rust loads artifact at startup + runs the model-parity gate
```

Every artifact is **immutable and versioned**; the running system records *which* model
version produced *which* signal, so any live decision is reproducible offline. This
reproducibility is what makes shadow trading and post-hoc audits possible
([07](07-parity-and-testing.md)).

## Summary

- Model parity: **solved** by FP32 + native/ONNX + a CI golden test. Trees are ~exact; NNs
  are FP32-equivalent with decision-invariance enforced.
- Feature parity: **the real work.** Solved on the serving path by *one* feature
  implementation (`axon.features`) plus a live parity monitor — which is still how every live
  session runs. The move of features into Rust now exists as a *capability*
  (`crates/axon-features`) and it passed the bit-equivalence gate this section demanded:
  33 423 cells of real venue data at `max_abs_diff = 0e0`
  ([ADR-0035](adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)). Nothing has been
  promoted to it. The gate is the precondition for promoting a strategy, not the promotion.
