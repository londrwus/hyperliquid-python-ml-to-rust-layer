# ADR-0019 — Native Rust inference backends: `tract` for ONNX, a hand-written XGBoost reader for trees

**Status:** Accepted · **Date:** 2026-07-25

## Context

[ADR-0003](0003-model-serving-and-fidelity.md) §2 names a Rust serving backend per model family
and §3 makes a **model-parity gate** mandatory: no model serves until Python and Rust are shown
to agree. Until now `axon-model` was a `Model` trait and a `LinearModel`, so the gate had nothing
to gate and Boundary A — inference in Rust, Python out of a strategy's live loop
([02](../02-python-rust-boundary.md)) — was a paragraph rather than a code path.

Four questions had to be answered, and each has a plausible wrong answer:

1. **Which ONNX runtime?** `ort` is faster and is what most projects reach for.
2. **How are trees served?** The cheap route is to convert them to ONNX and have one backend.
3. **What does "FP32 end-to-end" ([ADR-0005](0005-fp32-no-quantization.md)) actually mean at
   load time?** Checking that the model's inputs and outputs are FP32 is the obvious reading.
4. **Where does a model's version come from?** The caller passing one is the obvious answer.

## Decision

### 1. ONNX runs on `tract`, pinned in the crate, unoptimized

`tract-onnx` `=0.23.4`, added to `crates/axon-model/Cargo.toml` rather than to the workspace
manifest. `[workspace.dependencies]` is for versions more than one crate shares; listing a
single-consumer, forty-crate dependency tree there advertises it as common vocabulary and invites
the next crate to reach for it casually. The pin is *exact* because operator semantics **are** the
numbers `tests/parity.rs` asserts on — a silent patch bump could move a result inside the
tolerance with no code change to blame it on.

`tract` over `ort` because this path is chosen for **determinism, not throughput**:

- **Pure Rust.** No `libonnxruntime` to ship, to keep version-matched against research, or to
  accidentally pick up a different build of on the trading host. [ADR-0003](0003-model-serving-and-fidelity.md)'s
  "pin the runtime + hardware" rule is satisfied by the Cargo lockfile instead of by deployment
  discipline.
- **Single-threaded by default.** `tract`'s executor is `SingleThread` unless a non-default
  feature is enabled, so a matmul's reduction order is fixed and the same input produces the same
  bits on every run. A thread-pooled runtime gives no such guarantee, and a run-to-run difference
  would mean a replayed live session could not reproduce the decision it is replaying.
- **Speed is not the constraint.** The venue round-trip is 0.2–0.9 s ([05](../05-latency-model.md));
  the difference between the two runtimes is microseconds on the wrong side of five orders of
  magnitude.

The graph is built with `into_typed()`, **not** `into_optimized()`. Optimization fuses and
reassociates operators, which is precisely the `opt_level=0` that ADR-0005 requires when
structural numerical match to the Python model matters. The batch dimension is pinned to 1 at
load so every shape in the plan is concrete and nothing is resolved per tick.

The artifact must declare **exactly one input** of shape `[batch, features]` and **exactly one
FP32 output**. Which tensor feeds the strategy has to be a property of the artifact, not a
convention that the exporter and the loader each remember separately.

### 2. Trees are read from XGBoost's own JSON and evaluated directly

Not from `get_dump(dump_format="json")`, which omits `base_score`, the objective and
`num_feature` — you cannot reproduce `predict()` from it, only relative leaf sums. And not via
ONNX, whose `ai.onnx.ml` `TreeEnsembleRegressor` applies the *converter's* float handling: gating
that would gate a translation of the model rather than the model.

ADR-0003 calls this family "~numerically exact". `TreeModel` has to earn that, and three places
are where a hand-rolled reader silently diverges:

- **The missing-value default direction.** `NaN < threshold` is false in IEEE-754, so the natural
  `if v < t { left } else { right }` routes every missing value right — agreeing with XGBoost on
  exactly the nodes whose default happens to be right, and only on rows that actually have
  missing data. The NaN test comes first.
- **The threshold dtype.** Comparison is `f32` on both sides, as XGBoost does it. Widening either
  side moves a value sitting exactly on a threshold into a different leaf. (Thresholds are parsed
  decimal → `f64` → `f32` by `serde_json`; that is safe rather than lucky, because XGBoost writes
  the shortest decimal that round-trips an `f32` and `f64`'s 53 bits clear the 50 that the
  double-rounding bound needs.)
- **The intercept's link.** For a non-identity link, XGBoost stores `base_score` in *prediction*
  space and converts it with `-logf(1.0f/base_score - 1.0f)` in **single** precision. Reading it
  as a margin offsets every prediction by a constant (0.5 for a default-intercept classifier);
  computing the *more accurate* `f64` logit and rounding disagrees with XGBoost by one ULP for
  roughly half of all fitted intercepts. The `f32` expression is the specification.

**`TreeModel::predict` returns the raw margin and never applies the link** — the equivalent of
`Booster.predict(..., output_margin=True)`. XGBoost's links are monotone, so a decision threshold
on the probability is exactly a decision threshold on the margin: skipping the link costs nothing
under ADR-0003's decision-invariance criterion and removes `expf` from the serving path.

Everything the reader cannot reproduce **exactly** is refused by name at load, not approximated:
`dart` (its trees are dropout-weighted), categorical splits, multi-output/multi-class ensembles,
early-stopped artifacts (`predict` truncates the ensemble; we would serve the overfitted tail),
and any objective outside the identity and logit allow-lists. A load-time refusal is an operator
re-exporting a model; a silent approximation is a P&L question nobody thinks to ask.

Load also **proves the traversal terminates** — every child index must be greater than its
parent's and in range — because a cycle in a hand-edited artifact would park the single-threaded
core in an infinite descent, which is a hang, not a crash, and nothing would alert.

### 3. FP32 is enforced on the protobuf, before the graph is built

A graph whose *signature* is FP32 can still round every intermediate through FP16: a `Cast` in
the middle is invisible from the boundary, and that is the shape of the documented ONNX Runtime
downcast footgun ADR-0005 was written about. So the loader walks the `ModelProto` — initializers,
input/output/value-info types, node attributes (including `Cast`'s `to`), and subgraph bodies —
and refuses on any float type below FP32, plus on any quantization operator. It refuses; it does
not warn and load.

### 4. A model that cannot name its own version does not load

The version is stamped on every `Signal` a decision came from, which is what makes the decision
reproducible offline (ADR-0003, [07](../07-parity-and-testing.md)). Taking it from the caller
lets it drift from the bytes it claims to describe, so each backend reads it out of the artifact:
ONNX's first-class `model_version` field, and — because XGBoost's JSON has no equivalent slot —
the learner attribute `axon_model_version`, which `Booster.set_attr` round-trips through
save/load. The asymmetry is deliberate: inventing a metadata convention where the format already
has a field would be worse.

**Amended 2026-07-26.** This section described only the reading half, and the writing half did not
exist: `axon.models.export.export_artifact` stamped **neither** field, so for as long as both sides
have existed, *no artifact the registry's own export path produced could be loaded by either
backend* — every bundle the cross-language gate ever ran on had been hand-stamped by
`crates/axon-model/tests/bundles/generate.py`. The gate that certifies the boundary had never run
on the path it certifies. The exporter now writes the field the backends read, taken from the
`meta.version` it is already handed and written before the payload is hashed, which is recording
rather than inventing. Regenerating all three committed bundles through the stamped path produced
byte-identical artifacts, which is the strongest available evidence of that distinction. See
[0032](0032-the-model-zoo-and-what-actually-crosses-into-rust.md) §4d, which found it, and
[0033](0033-lightgbm-crosses-by-conversion-not-by-backend.md) §8, which settled it. LightGBM's
*native* kind is deliberately left unstamped: its text format has no version slot and no Rust
reader to read one, so a convention there would be one nothing consumes.

### 5. The decision-path entry point is `predict_into`

`Model::predict_into(features, out)` writes into a caller-owned buffer; `predict` is the
allocating wrapper for tests and startup checks. The core does not allocate per event
([05](../05-latency-model.md)). `Model` also reports `input_len`/`output_len` so a feature spec
that no longer matches the graph is caught once at wiring time rather than on the first tick.

### 6. The gate's fixtures are frozen, and the generator does not run in CI

`crates/axon-model/tests/fixtures/generate.py` trains the models, exports the artifacts and
records Python's own answers; the artifacts and answers are committed. Regenerating the reference
in the same breath as asserting it proves nothing — the point is to catch Rust drifting away from
a frozen Python answer. Tree margins are asserted **bit for bit** (stored as IEEE-754 bit
patterns, because a decimal round-trip would let the fixture absorb the one-ULP error the gate
exists to catch); the ONNX graph is asserted within **1e-5**, the ADR-0003 tolerance.

## Consequences

- **+** The model-parity gate now has a Rust side. Trees are held to bit equality against
  `Booster.predict` — including a logistic model whose fitted intercept exercises the link math —
  and the ONNX path to 1e-5, entirely offline and deterministic.
- **+** Boundary A is a swap rather than a project: a strategy that passes the gate can move its
  inference into the Rust core without touching the architecture ([05](../05-latency-model.md) §2).
- **+** Every failure mode the tree reader can hit is a named load-time error, so a bad artifact
  is caught by the person exporting it.
- **+** FP16 cannot be served by accident, including the hidden-`Cast` case that a
  boundary-only check would wave through.
- **−** `tract` is an unconditional dependency of `axon-model`, so a future tree-only consumer
  pays its (substantial) compile time. Feature-gating the ONNX backend is the obvious next move
  and was skipped rather than guessed at.
- **−** The objective allow-list is narrow. `count:poisson`, `reg:gamma`, `reg:tweedie`, the
  `multi:*` family and multi-target ensembles are refused, because their `base_score` link (or
  their tree-to-output-group mapping) has not been pinned against XGBoost the way logit has.
  Each is a small, testable addition; none is guesswork we should ship.
- **−** The logistic intercept is the one place tree exactness depends on the platform: `f32::ln`
  lowers to the system `logf`. It agrees with the reference on the Linux target
  ([ADR-0007](0007-linux-wsl2-dev-target.md)); a libm that rounds differently would shift every
  logistic margin by one ULP. Identity-link objectives carry no such dependency, and the fixture
  covers both.
- **−** The single-input/single-output rule rejects artifacts an exporter produces by default —
  an `skl2onnx` classifier emits a label *and* a probability tensor. Those must be re-exported
  with one output, which is friction at export time in exchange for no ambiguity at serve time.
- **−** `tract` does not implement every ONNX operator. An unsupported op fails at load, which is
  the right time to find out, but it is a real compatibility ceiling that `ort` would not have.
- **−** The fixtures are frozen against one Python stack (xgboost 3.3, onnx 1.22, skl2onnx 1.20).
  Regenerating on a different stack changes the committed bytes and the expected values; that is
  a deliberate, reviewable event, not a silent one.
- **−** Still unbuilt in this crate: the LightGBM backend (`lightgbm3-rs`), the `tch-rs` Torch
  path for models that must track Python as closely as possible, and the Rust feature runtime —
  which [03](../03-ml-fidelity-and-features.md) is explicit is the *harder* half of parity. This
  ADR closes the model half only.

See [ADR-0003](0003-model-serving-and-fidelity.md) (which family gets which backend, and the
parity gate this implements), [ADR-0005](0005-fp32-no-quantization.md) (FP32, and the FP16
threshold-flip this loader refuses), [ADR-0002](0002-python-rust-boundary.md) (why Boundary A is
the destination and not the start).
