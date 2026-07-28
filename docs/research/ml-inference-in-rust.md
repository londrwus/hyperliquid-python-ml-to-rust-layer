# Research — Running Python-trained ML models in Rust (2026)

**Bottom line:** For "no quality loss," the path depends on model family. **Tree ensembles
reproduce essentially exactly** in Rust (deterministic threshold traversal — no
float-reduction ambiguity). **Neural nets never reproduce bit-for-bit** across runtimes;
target FP32 tolerance (~1e-5..1e-6), not bit-identity. The **feature-engineering layer**, not
the model, is where most silent quality loss happens.

## Backends

### `ort` — ONNX Runtime bindings (Microsoft engine)
- Mid-2026: `2.0.0-rc.12`, wrapping ONNX Runtime 1.24; maintainers call it production-ready
  (only the API is unstable). Used by SurrealDB, Google Magika, Wasmtime.
- Loads anything exportable to ONNX (PyTorch, TF/Keras, sklearn, XGBoost/LightGBM via
  `skl2onnx`/`onnxmltools`).
- **Fidelity:** bit-identical to Python `onnxruntime` on the same ONNX model; vs the original
  PyTorch model expect ~1e-5..1e-6 FP32 deltas. Fastest general option.
- Trap: rc builds default to edge-tuned `--client_package_build`; re-enable op spinning for
  server throughput.

### `tract` — pure-Rust ONNX/NNEF inference (Sonos)
- `v0.23.3`, pure Rust, no C++ deps. Passes ~85% of ONNX conformance; all "real-life"
  integration models pass within tolerance. Gap: no Tensor Sequences / Optional Tensors
  (some dynamic/control-flow graphs won't load).
- **Most deterministic/reproducible** NN option (single-threaded-friendly, no vendor kernel
  surprises) → best for a research↔prod parity story. Slower than ORT, CPU-bound, but tiny
  static binaries and trivial cross-compile.

### `tch-rs` — libtorch/PyTorch bindings
- Requires an exact libtorch version match (~2.11.0 in mid-2026). Path: `torch.jit.trace/script`
  → `.pt` → `tch::CModule::load()`.
- **Highest NN fidelity** — runs the *same libtorch kernels* as Python. **Heaviest deploy**
  (hundreds of MB of C++ libs, version-pinned). TorchScript export breaks on some dynamic
  control flow / custom ops.

### `candle` — pure-Rust ML (Hugging Face)
- Mature for inference; loads safetensors/GGUF/`.pth` into **Rust-defined** architectures
  (not an arbitrary-graph importer). Strong for transformers/LLMs/ViT/Whisper. Own kernels →
  close but not identical; validate per model.

### `burn` — Rust DL framework
- `burn-onnx` transpiles ONNX → native Rust source (opset 1–24), stays trainable. **Best
  documented fidelity story of the pure-Rust options** — 26 real-world models validated
  against ONNX Runtime reference outputs. Operator coverage still expanding.

### Tree models (the quant workhorse)
Inference is deterministic tree traversal → **numerically reproducible**, so "no quality
loss" is genuinely achievable.

| Approach | Loads | Fidelity | Notes |
|---|---|---|---|
| `ort` + `skl2onnx`/`onnxmltools` | XGB, LGBM, sklearn | Matches ONNX ref | Most portable. Watch float-vs-double thresholds. |
| **`xgboost-ars`** (pure Rust) | XGBoost native JSON | Validated vs XGBoost 3.2.0 + ort; exact TreeSHAP | Zero C++ deps, cross-compiles. Best modern pure-Rust XGB pick. |
| `gbdt` (gbdt-rs) | XGBoost only | Small delta; **stale** (validated only to XGB 0.81/0.82) | Avoid for modern models. |
| **`lightgbm3-rs`** | LightGBM native | **Exact** (uses LightGBM C lib) | Current maintained LGBM binding. Ships C dep. |
| `lleaves` / `Treelite` | LGBM/XGB → LLVM/C | Exact | Python-side compilers → FFI a shared lib. Fastest CPU latency for LGBM. |

**Latency reference (256-feature/10-label bench):** XGBoost multithread 1.91 ms →
`inplace_predict` single-thread 0.70 ms → ONNX Runtime 0.14 ms. Compiled/ORT tree inference
is sub-millisecond.

## The hard truth on numerical fidelity
- **NN models are NOT bit-reproducible** across Python↔Rust: ONNX doesn't encode op ordering,
  and FP is non-associative (parallel reductions differ — one GPU matmul produced 6 distinct
  FP32 results).
- **Minimize drift:** stay **FP32** (no quant/AMP), disable graph optimizations (`opt_level=0`)
  for structural match, pin the **same execution provider + hardware**, and **round-trip test
  for silent FP16 downcasting** (a real ORT/CoreML footgun that can flip a prediction across a
  threshold — actual quality loss).
- **Validation gate:** dump N-thousand inputs, run Python + Rust, assert max abs diff < ε
  (1e-5 for NN logits; exact/near-exact for trees), *and* verify no discretized decision flips.
  Put it in CI.

## Feature-engineering parity — where quality actually leaks
The model is ~20% of the strategy; if research and execution compute features differently you
get **training–serving skew** (the #1 silent-degradation failure). Fixes, best→pragmatic:
1. **Single source of truth** — don't implement features twice. Shared execution engine, or
   compile one implementation to both languages.
2. **Feature store** with versioned defs (Feast/Tecton) — solves storage/point-in-time
   correctness, but **not** transform parity if you still hand-code twice.
3. **Point-in-time-correct joins** (no lookahead leakage).
4. **Parity monitor** (mandatory backstop): resample live feature vectors, recompute offline,
   alarm on divergence. Catches timezone/late-event/rounding/windowing/NaN/staleness bugs.

## Recommendation matrix

| Model type | Best Rust path | Fidelity |
|---|---|---|
| XGBoost / LightGBM | `xgboost-ars` / `lightgbm3-rs`, or `ort`+ONNX | Effectively exact ✅ |
| sklearn tabular | `skl2onnx` → `ort` | Exact within ONNX ref |
| PyTorch net, must match Python | `tch-rs` (TorchScript) | Closest possible |
| PyTorch/TF net, light deploy | ONNX → `ort` (perf) or `tract` (portable) | FP32-equivalent |
| Transformer/LLM | `candle` | Close; validate |
| No runtime dep + fine-tunable | `burn` via `burn-onnx` | Validated vs ORT (26 models) |

For a mostly-tree quant executor: **pure-Rust tree inference + a shared/compiled feature
library + a parity monitor** = genuine "no quality loss" at sub-ms latency, no C++ runtime.
Reserve `ort`/`tch-rs` for NN strategies (FP32-tolerance equivalence, not bit-identity).

## Sources
ort.pyke.io · github.com/sonos/tract · github.com/LaurentMazare/tch-rs ·
github.com/huggingface/candle · github.com/tracel-ai/burn-onnx · atomdrift.org/xgboost-ars ·
github.com/mesalock-linux/gbdt-rs · emmtrix.com/wiki/Numerical_Precision_in_ONNX_and_AI_Inference ·
arxiv.org/pdf/2506.09501 · featurestore.org · ncbi PMC12406228 (tree inference bench)
