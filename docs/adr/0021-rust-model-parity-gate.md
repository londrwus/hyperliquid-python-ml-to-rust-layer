# ADR-0021 — The cross-language model-parity gate: a frozen Python question, answered in Rust

**Status:** Accepted · **Date:** 2026-07-25

## Context

[ADR-0016](0016-feature-spec-and-parity-gates.md) built the model-parity gate and
[ADR-0019](0019-native-rust-inference-backends.md) built the Rust backends the gate exists to
protect. Between them they left one gap, and it is the gap the whole Boundary-A argument stands
on.

`axon.parity.model_parity` compares a reference set of predictions against a candidate set. In
every test that calls it today, **both sets are Python's.** That is a real gate — it catches a
retrain that moved, a refactor of the feature pipeline, a candidate that flips a trade — and it
is structurally incapable of failing on the thing Boundary A turns on: *the model Rust will
serve producing a different answer from the model Python researched with.* A Python-to-Python
comparison cannot see an f64 accumulator in the Rust tree loop, a widened threshold comparison,
a link applied on one side only, or a runtime that reassociates a reduction.

The Rust side had its own gate — `crates/axon-model/tests/parity.rs`, a table of numbers
`tests/fixtures/generate.py` recorded — and it proved the backends read *those artifacts*
correctly. It did not connect to the registry, it carried no feature-spec reference, it had no
trade thresholds and therefore no decision criterion, and its inputs crossed as JSON decimals.
Separately: `axon-model` was **imported by nothing in the workspace**. A crate with no consumer
is a crate whose contract is untested, and "bit-exact against XGBoost" was a claim asserted by
one test file about one fixture rather than a property of the serving path.

Four questions had to be answered, and each has an answer that looks obviously right:

1. **What is the gate's input?** A CSV or JSON of feature rows is the obvious answer, and it
   quietly changes the question being asked.
2. **What does the Rust side compare against?** A tolerance is the obvious answer. It is also
   the answer that ships a model which flips a trade.
3. **Where do the fixtures come from?** Generating them at test time is the obvious answer, and
   it produces a gate that can never fail.
4. **What holds the two languages to the same standard?** The manifest declaring one is the
   obvious answer, and it lets a red gate be fixed by editing the fixture.

## Decision

### 1. A parity bundle is a directory, and its inputs cross as raw IEEE-754 bits

`axon.parity.rust_gate.write_parity_bundle` takes a **registry artifact** (ADR-0015) and a
holdout matrix and writes:

```
<bundle>/manifest.json     what this is, and what it must be held to
        /model.json        the artifact's own bytes, straight off the registry
        /features.f32      the holdout matrix, raw little-endian IEEE-754
        /predictions.f32   Python's own answer over those exact bytes
        /decisions.i8      Python's own discretized decision per row
```

The matrices are **raw little-endian `f32`, never JSON numbers**, and the guarantee is
mechanical rather than careful:

- The holdout is cast to `float32` **once** and written with `dtype='<f4'`. There is no decimal
  anywhere on the path, so there is no re-parse that can land on the neighbouring float. That
  matters more for a tree than the size of the error suggests: a feature that moves one ULP
  across a split threshold does not move the prediction by one ULP, it moves the row to the
  other leaf.
- **The reference predictions are taken over the bytes read back off disk**, not over the array
  that produced them. The recorded answer is by construction the answer to the question the
  Rust side asks. Scoring the in-memory array and then serializing it would leave the file
  unverified in the one respect that decides whether the two languages are comparing like with
  like.
- The byte order is **pinned little-endian rather than native**, because the reader is
  `f32::from_le_bytes`. A native-order write on a big-endian host would hand Rust every feature
  byte-swapped, and a perfectly correct model would fail the gate spectacularly.
- `NaN` travels as its own bits, which JSON `null` cannot do. The tree backend's
  default-direction branch is gated on exactly those cells, and the Rust test refuses a tree
  bundle whose holdout has none — the corpus, not the assertion, is what puts that branch under
  the gate.

The manifest carries the model version, the artifact's `feature_spec_ref` (ADR-0016's
`name/vN#fingerprint`), the artifact's git SHA, and **the library versions that scored the
reference** — separately from the ones that exported the artifact, because the frozen answer
depends on the former and the reproduction instructions on the latter.

### 2. The gate is a tolerance check `and` a decision check, in both languages

The criteria are the ones already decided, applied across the boundary for the first time:
**bit equality** for XGBoost, because ADR-0019 claims `TreeModel` reproduces
`Booster.predict(output_margin=True)` exactly and a tolerance there would be the gate declining
to test its own claim; **`max_abs_diff <= 1e-5`** for ONNX (ADR-0003 §3); and for both, **no
discretized trading decision may flip.**

Note the Rust side's bit comparison is *stronger* than the Python gate's `TREE_EPS = 0.0`,
which is a numeric zero and so accepts `+0.0` against `-0.0`. The stronger check is the one
ADR-0019's text actually claims.

**Amended 2026-07-26 — a graph bundle now declares two ULP, not the family ceiling.**
ADR-0003 §3 set 1e-5 as a *starting* tolerance, and the three graphs this gate actually runs on
were measured against it rather than assumed to need it: `lgbm_binary` **1.1920929e-7** (one ULP
at 1.0, exactly), `zoo_logistic` **8.940697e-8**, `mlp_regressor` **0e0**. Declaring 1e-5 while
achieving 1e-7 leaves roughly forty-twofold of slack, and slack is where a regression passes
green — which is the same argument §6 of ADR-0019 makes for pinning `tract` to an exact version:
*a silent patch bump could move a result inside the tolerance with no code change to blame it
on.* Making that argument and then declaring the ceiling is declining to act on it.

So `ONNX_EPS` stays 1e-5 as the **ceiling a reader enforces**, and a new `ONNX_TIGHT_EPS =
2**-22` is what a **writer here declares** — mirrored in `axon.parity.rust_gate` and
`axon_model::parity`, with the committed bundles asserted against it in both languages so a
regeneration that fell back to the ceiling reddens. `Criterion::allows` needed no change: it was
already written to permit tightening and refuse loosening, and this is the first thing to use it.

Three things this deliberately did **not** do. It did not move the ceiling, so a bundle from
elsewhere declaring 1e-5 is still readable — it is *untightened*, not loosened, and no reader can
tell those apart; what keeps our own bundles honest is an assertion, not the criterion. It did not
fit the number to the measurement: a tolerance derived from what today's runtime produced is one
that ratchets, recording a regression as the new bar the next time it is regenerated, which is
precisely the failure `allows` exists to catch. And it did not promote `mlp_regressor`'s observed
`0e0` to `bit_exact`, because ONNX does not encode operator ordering and float addition is not
associative, so exact agreement on one machine is luck rather than a property, and a criterion
fitted to luck reddens on a CPU with different FMA behaviour for no defect at all.

One test had to be corrected rather than merely re-run, and the correction is the more useful
half. `a_bundle_cannot_buy_itself_a_looser_criterion` loosened its copy by
`.replace("1e-05", "0.01")` on the manifest text; the day the writer stopped emitting `1e-05`,
that pattern matched nothing, the "corrupted" bundle was a byte-identical copy, and the
assertion was asking its question of a bundle nobody had loosened. It now goes through a helper
that **asserts it changed the bytes**. A corruption helper that does not corrupt reads as
protection while testing nothing.

The decision half is an `and`, and `tests/cross_language_parity.rs` carries ADR-0016's paired
argument across the boundary rather than restating it: `the_same_delta_inside_the_flat_band_passes`
and `a_sub_tolerance_delta_that_crosses_the_trade_threshold_fails_the_gate` apply the *same*
sub-tolerance perturbation — a quarter of the bundle's own declared criterion, taken from that
criterion rather than written as a literal, since the literal it used to carry went stale the day
the criterion tightened — and come out opposite ways. In the
failing case `max_abs_diff` is comfortably green while a position goes from flat to long.

Two further traps are closed structurally:

- **A non-finite score is counted, never compared.** `nan <= eps` is false and a NaN decides
  *flat* on both sides, so a naive gate reports a model that started producing NaN as a perfect
  match with no decisions flipped.
- **A decision corpus that can only come out one way is refused.** The writer rejects a
  threshold pair that puts every row on the same side, and the Rust test asserts the committed
  bundles decide more than one way. An invariance check nobody has seen change its mind is a
  decoration.

### 3. The decision *rule* is gated, not only the numbers

`decisions.i8` records Python's own discretization, and `ParityBundle::open` re-derives it from
the recorded scores and the manifest's thresholds, refusing the bundle if the two disagree —
before a model is loaded. This closes a failure the score comparison structurally cannot see:
if the two languages implement the boundary differently (`>=` against `>`, or a threshold
rounded to a different float), **every prediction can match to the bit while the two systems
still trade differently.**

For the same reason the thresholds cross as bit patterns, and `Decision.__post_init__` rounds
them to `float32` on the Python side. A threshold held at float64 in Python and at its f32
neighbour in Rust puts the two decision boundaries a ULP apart, and every score in between
decides differently on each side — a disagreement manufactured by the gate itself.

### 4. The fixtures are committed, and the generator does not run in CI

`crates/axon-model/tests/bundles/generate.py` trains the models, exports them through the real
`export_artifact` → `ModelRegistry.save` path, and writes the bundles; the bundles are in git
(~33 kB for three). Regenerating the reference in the same breath as asserting it proves
nothing — it would agree with whatever Rust had just started doing. The point of the gate is to
catch Rust drifting away from a **frozen** Python answer, so the answer has to be frozen, and
the diff is the review.

The consequence is deliberate: the Rust gate runs in the default offline `cargo test`, on a
machine with no Python and no ML libraries at all, in half a second. A gate that needs a
research environment is a gate someone eventually marks as "run it before release."

The bundle carries the model bytes rather than a path into a registry, for the same reason: a
gate that reaches outside its own directory is a gate that passes on one machine.

Committing binary payloads into a repo whose root sets `* text=auto eol=lf` needs one guard,
and it is not hypothetical: git's text heuristic only looks for a NUL byte in the first 8 kB,
and `tree_identity/predictions.f32` genuinely has none. A `0x0D 0x0A` pair occurring *inside a
float* would be normalized away on checkout — silently shortening the matrix and rewriting the
numbers the gate compares, on a machine whose only sin was cloning the repo. A directory-local
`.gitattributes` marks `*.f32`, `*.i8` and `*.onnx` binary. This is the same class of bug as
`docs/07`'s cautionary tale about NautilusTrader's per-platform precision: same code, different
numbers, no error.

### 5. A bundle cannot buy itself a looser gate

The manifest declares its criterion, and **both readers check that declaration against what the
family allows** (`Criterion::required_for`). An ONNX bundle asking for 1e-2 is refused; a tree
bundle asking for any tolerance at all is refused, including one a thousand times tighter than
needed. Tightening below the required bar is allowed.

The failure this prevents is procedural rather than numerical: a bundle regenerated after a red
gate, with the tolerance nudged until it passed, would otherwise be indistinguishable in the
tree from one that never failed.

### 6. Python scores the way Rust serves, not the way research is convenient

Two asymmetries in `python_scores` are load-bearing:

- **XGBoost is scored as a margin** (`output_margin=True`), because `TreeModel` returns the raw
  margin and never applies the link. Recording probabilities would compare the *link* rather
  than the model, and would fail for a reason that has nothing to do with either. The reader
  refuses a tree bundle whose `score_space` is not `margin` rather than trusting the writer.
- **ONNX is scored one row at a time**, because the Rust plan pins its batch dimension to 1
  (ADR-0019 §1). A graph scored ninety-six rows at once may reassociate a reduction that a batch
  of one cannot, and the gate would then be measuring the batch shape. Trees go through in one
  call, because tree traversal is row-independent and the batch shape cannot change an answer.

### 7. The reader lives in `axon-model`, and it does not hash

`ParityBundle` is `pub` in the serving crate, not private test scaffolding, because the live
parity monitor `docs/07` asks for needs exactly this in Rust: load a frozen question, score it,
get back a report with `passed()`, `summary()` and the offending rows. It is the same shape
`axon.parity` returns for the same reason — the identical call has to serve a CI assertion and
a 03:00 alarm.

It deliberately **does not verify the SHA-256s the manifest records**, though the Python reader
does. A flipped bit in either matrix changes a prediction, so the gate itself catches it; a
crypto dependency in the crate that serves models on the decision path would buy a better error
message and nothing else. Python is the side that can afford to say *why*.

### 8. The failure output is the deliverable

A parity failure that says "assertion failed" is a parity failure nobody can act on. Every
divergent row reports the row index, both scores **as decimals and as bit patterns**, the
delta, both decisions and whether the decision flipped. The bit patterns are not decoration: at
one ULP the two decimals routinely print identically, and a failure message showing the same
number twice reads as a lie.

The gate was checked against a real regression before it was believed: replacing `TreeModel`'s
`f32` margin accumulator with an f64 one — the *more accurate* version, which is exactly why it
is wrong (ADR-0019 §2) — turns the gate red on 77 of 128 rows with a worst delta of 4.8e-7 and
**no decision flipped**, and the report names every one of them. That is the whole shape of the
problem in one line: a change nobody would call a bug, invisible to a 1e-5 tolerance, caught
only because trees are held to bits.

## Consequences

- **+** `axon-model` finally has a consumer, and the claim "bit-exact against XGBoost" is now a
  property of an artifact the registry produced rather than of a hand-made fixture. Two tree
  families (identity and logit link, the latter exercising the `f32` intercept math) and one
  ONNX graph are gated, offline, in the default `cargo test`.
- **+** The decision-invariance criterion, which ADR-0016 is emphatic about, now spans the
  language boundary — including the rule itself, not only the numbers it is applied to.
- **+** Boundary A gets a promotion checklist that is executable: write a bundle from the
  registry artifact, run `cargo test -p axon-model`, and the answer is a report rather than an
  opinion.
- **+** The bundle is portable and self-contained, so the same directory can be handed to a
  future live parity monitor, a second Rust backend, or a colleague debugging a divergence.
- **−** The committed corpora are synthetic gaussians, not market data. They gate *arithmetic*,
  and their `feature_spec_ref` names the probe recipe rather than borrowing a real
  `axon.features` spec id — a manifest claiming `PERP_CORE_V1` fed these would be a lie in
  exactly the field an audit trusts. The gate over a real strategy's holdout is N7's job and is
  not done here.
- **−** This gates the **model half only**. `docs/03` is explicit that features are the harder
  half of parity, and a Rust feature runtime does not exist; a bundle proves that identical
  feature *vectors* produce identical decisions, not that the two paths would compute identical
  vectors from the same market data.
- **−** Like ADR-0019's fixtures, the bundles are frozen against one Python stack (xgboost 3.3,
  onnx 1.22, skl2onnx 1.20, onnxruntime 1.28, numpy 2.5). An upgrade that moves the answers is
  now a *reviewable* event — `test_the_frozen_reference_is_still_what_this_python_stack_says`
  fails in Python, naming the library, instead of surfacing months later as an unexplained Rust
  failure — but it is still an event someone has to handle.
- **−** One score column only. A multi-output artifact cannot be bundled, because a decision
  threshold is defined on one number and picking a column here is the guess ADR-0015's
  `score_output` exists to prevent. Multi-class decision invariance (argmax stability) needs a
  second criterion kind and is not built.
- **−** LightGBM artifacts are legal in the registry and have no Rust backend, so they are
  refused by name at bundle-write time. That is the honest outcome of ADR-0019's documented gap,
  and it means a LightGBM strategy cannot use this gate at all.
- **−** The bundle's integrity story is a hash, and a hash is not a signature: anyone who can
  rewrite `predictions.f32` can rewrite the manifest beside it. In-tree that is what git review
  is for; for a bundle received from elsewhere it is not sufficient.
- **−** Nothing yet regenerates or *runs* this from `run.sh`, and CI gains it only implicitly
  through `cargo test --workspace`. A named subcommand and an explicit CI step would make a
  regression read as "the parity gate failed" rather than as "a test in axon-model failed."

See [ADR-0019](0019-native-rust-inference-backends.md) (the backends this gates and the
exactness claims it holds them to), [ADR-0016](0016-feature-spec-and-parity-gates.md) (the
Python gate this extends, and the decision-invariance argument it carries across),
[ADR-0015](0015-model-artifacts-and-registry.md) (the registry artifact a bundle is written
from), [ADR-0005](0005-fp32-no-quantization.md) (FP32 end to end, which is why a one-ULP
disagreement is worth a gate), [ADR-0003](0003-model-serving-and-fidelity.md) (§3, the
model-parity gate this finally completes), and
[07](../07-parity-and-testing.md) (the ladder this is the bottom rung of).
