# ADR-0033 — LightGBM crosses by conversion, not by a backend

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0019](0019-native-rust-inference-backends.md) closes with a list of what `axon-model` does
not have, and the LightGBM backend (`lightgbm3-rs`) is the first item on it. Downstream, that gap
is enforced by name: `axon.parity.rust_gate.SERVABLE_KINDS` is `("xgboost", "onnx")`, so a
`lightgbm` artifact — legal in the registry ([ADR-0015](0015-model-artifacts-and-registry.md)),
served in Python by `LightgbmPredictor`, usable in a backtest — cannot have a parity bundle
written for it at all. The handoff has carried "**LightGBM artifacts are refused by name at
bundle-write time**" as a standing fact for two sessions, and the obvious reading of it is that
the family needs a Rust reader.

That reading was never tested. `onnxmltools` converts a LightGBM booster to ONNX, ONNX is a kind
the gate already admits, and `tract` already serves it. If that works, the "missing backend" is a
**routing** question and the cheapest correct answer is no new Rust code at all.

Four things had to be measured rather than assumed, and three of them came out differently from
the way they read on paper:

1. **Does the conversion hold ADR-0003's 1e-5 tolerance for graphs?** A tree ensemble
   re-expressed as `ai.onnx.ml` operators is exactly the translation ADR-0019 §2 refuses to gate
   for XGBoost, on the grounds that it moves samples across splits.
2. **Which score space is each side in?** `SCORE_SPACE` exists because `TreeModel` returns
   XGBoost's raw margin and never applies the link, so a bundle holding probabilities would report
   the link as a parity failure. The same trap is available here in the opposite direction.
3. **Does `tract` load the result?** ADR-0019 already notes that `tract` does not implement every
   ONNX operator and calls that "a real compatibility ceiling that `ort` would not have".
4. **Is multi-class refused, or silently truncated?** `python_scores` raises when a graph emits
   more than one value per row and `read_parity_bundle` refuses `predictions.cols != 1`. Neither
   check had ever met a real multi-class graph, and a refusal that has never fired is a claim.

## Decision

**1. A LightGBM model that must reach Rust is converted, not re-kinded.
`SERVABLE_KINDS` does not change.**

`axon.models.export.lightgbm_to_onnx` converts the booster and the result is registered as an
ordinary `onnx` artifact. Adding `"lightgbm"` to `SERVABLE_KINDS` would be the tempting fix and
the wrong one: that kind is the booster's own *text* format, nothing in Rust parses it, and
widening the tuple would not make anything able to serve it — it would only move the failure from
bundle-write time to Rust **load** time, after the artifact has a version a `Signal` can refer to.
The refusal stays; what changes is that it now names the route out of itself, in all four places
it is raised, with a test pinning the text.

This does not repeal ADR-0003's "trees keep their own format" rule; it records that LightGBM is
the one family for which that rule has no destination. XGBoost's native JSON has a Rust reader and
is held to **bit** equality. LightGBM's does not, so its choice is a converted graph at 1e-5 or no
crossing at all. The native export stays the default, because a model that never leaves Python has
no reason to pay the conversion's cost.

**2. It holds the tolerance, with room to spare, and the numbers are these.**

Eight trees, seven leaves, five gaussian features, a 256-row holdout, `onnxmltools` 1.16.0 /
lightgbm 4.7.0 / onnx 1.22.0 / onnxruntime 1.28.0 / `tract` 0.23.4:

| comparison | max abs diff | verdict |
|---|---|---|
| `LightgbmPredictor` vs onnxruntime, binary objective | **6.63e-08** | inside `ONNX_EPS` 1e-5 |
| `LightgbmPredictor` vs onnxruntime, regression objective | **1.38e-07** | inside 1e-5 |
| `LightgbmPredictor` vs onnxruntime, 20% of cells NaN | **5.49e-08** | inside 1e-5 |
| Python's frozen reference vs **`tract`**, row by row | **1.19e-07** | inside 1e-5; **0 decision flips**, 91/256 rows bit-identical |

The NaN row is worth its own sentence: `onnxmltools` emits LightGBM's default direction as
`nodes_missing_value_tracks_true`, and `tract` reads that attribute, so missing values route the
same way on all three sides. The parity **bundle** still cannot carry a NaN into a graph —
`_holdout` refuses it, because a NaN into a neural graph comes back as a NaN score and `nan > eps`
is False on both sides, which would make the gate report a perfect match forever. That guard is
correct for the case it was written for and is now known to be conservative for tree graphs. It
was measured, not relaxed: loosening it needs a per-operator argument, not a per-family one.

**3. The score space is a probability on both sides, and that is a fact about the default rather
than a coincidence.**

`LightgbmPredictor` calls `Booster.predict` with the default `raw_score=False`, and the converted
graph carries LightGBM's own `post_transform=LOGISTIC`. So this family has no margin-vs-probability
mismatch — *unless someone takes the margin on one side*, which is one keyword away. Measured, on
the same holdout: the graph against `predict()` is **6.63e-08**; the graph against
`predict(raw_score=True)` is **1.41**. That is the sigmoid, reported as a model defect, at 140,000×
the tolerance. `SCORE_SPACE["onnx"]` is `"score"` and now says out loud what that means for a
converted booster, and there is a test named after the comparison that produces the 1.41.

**4. The converted graph is narrowed to one output and one column at export time, because the
alternative fails at the worst possible moment.**

`onnxmltools` emits a classifier as `(label, probabilities)` with the probability tensor `[n, 2]`.
`tract` requires exactly one output and one FP32 score column (ADR-0019 §1). A graph left in the
converter's shape passes the FP32 audit, passes the export round trip, registers cleanly — and
then fails at Rust **load**, which is after the artifact has a version, a hash and an audit trail.
ADR-0019 already names re-exporting with one output as the exporter's job; `narrow_to_score_column`
is that job, done once in code instead of by hand each time.

Three details are load-bearing. The kept column is the **last**, which is `P(positive class)` — the
same number `Booster.predict` returns, so the reference and the graph are the same quantity by
construction rather than by a convention two files remember separately. The graph is **pruned
to the nodes the kept output depends on**, so the `Cast` that only ever fed the discarded label
does not survive as a dead node that the FP32 audit and `tract`'s translator both still have to
make sense of. And a graph that **already** declares one float output of one column is returned
**untouched**: rebuilding it would change its bytes, and therefore its content hash, and therefore
every frozen reference taken over it, for no change in what it computes.

**4a. The same rewrite is what makes an sklearn *classifier* cross, and it is opt-in.**

`skl2onnx` emits `(label, probabilities[n, k])` for every classifier, ZipMap off included — the
identical shape, from a different converter. So `narrow_to_score_column` is not a LightGBM detail;
it is the general repair, and `export_artifact(..., narrow_score_output=True)` applies it to the
sklearn path. Measured on a bare `LogisticRegression`, 512-row holdout: the narrowed graph loads
in `tract` and reproduces Python's reference to **1.49e-07, 217/512 rows bit-identical, zero
decision flips**, where the un-narrowed export of the same model is refused with *"graph declares
2 outputs; exactly one is required"*.

It is **off by default**, and the reason is ADR-0015's, not caution: choosing a column is a
statement about which number a strategy trades on, and `score_output` exists precisely so that
nothing infers it. Two further costs make the default the right one. Narrowing changes what the
export round trip verifies — the reference must collapse to the same column, which removes the
`argmax` decision-invariance check `_compare` runs on multi-column outputs. And it is a
*deliberate* redefinition of the artifact, which a caller should have to write down.

Two things it cannot rescue, both measured and both surviving the narrowing untouched:

- A `StandardScaler` in the pipeline compiles to an `ai.onnx.ml` `Scaler` node, and `tract`
  0.23.4's registry for that domain is five operators — `CategoryMapper`, `LinearClassifier`,
  `LinearRegressor`, `Normalizer`, `TreeEnsembleClassifier`. A scaled model is refused at *parse*
  time however its outputs are arranged. (Dropping the scaler from a linear model cost nothing
  measurable: lbfgs converged in 8 iterations raw against 7 scaled, probabilities equal to three
  decimals.)
- `skl2onnx`'s `GradientBoostingClassifier` emits `base_values` of length 2 where `tract` expects
  one per class: `attribute 'base_values': expected length 1 (or undefined), got 2`. Deleting the
  attribute silently drops the intercept; padding it makes `tract` and onnxruntime disagree by
  0.18. Neither is a repair, so the boosted tree stays uncrossable through this route.

Both are recorded in the function's own doc comment, because they are the two afternoons this
route can cost and neither is visible from the error the caller first sees.

**5. Multi-class is refused three times, and each refusal now has a real graph behind it.**

There is no "positive" column among three, so any narrowing would be an invented trading rule
wearing a conversion's clothes. The refusals, in the order a model meets them:

- `lightgbm_to_onnx` refuses a booster with `num_class > 1` before converting anything.
- `python_scores` refuses a graph whose score output has more than one value per row. Verified
  against a genuine `[n, 3]` LightGBM graph, with the narrowing deliberately bypassed: it raises,
  and — checked explicitly — no `predictions.f32` and no `manifest.json` are left behind, so the
  half-written directory reads as an interrupted write rather than as a bundle.
- `read_parity_bundle` refuses `predictions.cols != 1`. This one is tested in its **strong** form:
  the forged bundle's column 0 is byte-for-byte the reference it already carries, so the hashes,
  the decisions and the counts all still agree. A reader that quietly took the first column would
  report that bundle as healthy; the test exists because that is the failure worth catching, not
  the ill-formed one.

On the Rust side the same artifact is refused as *"graph declares 2 outputs; exactly one is
required"*, measured against the committed fixture.

**6. The regression objective converts, scores correctly, and cannot be served — and the fixture
that proves it is committed.**

`tract` 0.23.4 registers `TreeEnsembleClassifier` and **does not register
`TreeEnsembleRegressor`** (`tract-onnx/src/ops/ml/mod.rs`). A LightGBM regression booster converts
cleanly, agrees with the native predictor to 1.38e-07, passes the FP32 audit and writes a valid
parity bundle — and then:

```
REFUSED parsing model artifact: Translating node #1 "LgbmRegressor"
Unimplemented(TreeEnsembleRegressor) ToTypedTranslator
```

That is a load-time refusal with a name on it, which ADR-0019 §"Consequences" already calls the
right time to find out, so this is **not** re-litigated as a Python-side guard. A Python refusal
would encode a fact about one `tract` version in the exporter, where it would go stale silently.
Instead the regressor bundle is committed as a fixture *expected to be refused*, so the claim is a
measurement anybody can re-run rather than a sentence in a document.

The consequence for strategy design is concrete and belongs in the open: **a LightGBM model that
has to cross into Rust should be fitted with a `binary` objective.** A regression objective is a
Python-only model until either `tract` grows the operator or the ensemble is re-expressed without
`ai.onnx.ml` — `onnxmltools`' `without_onnx_ml=True` does exactly that via `hummingbird-ml`, which
is not installed and was not evaluated.

**7. The fixtures live under `python/tests/fixtures/lightgbm-onnx`, and the reason is a
`.gitignore` line.**

`/data/` is ignored wholesale and `*.onnx` is ignored globally ("do NOT commit models or market
data"). A fixture the Rust side is pointed at that a fresh clone does not have is worse than no
fixture, so the bundles sit beside the tests and their `model.onnx` files are force-added — the
same way the committed bundles under `crates/axon-model/tests` already are. They are deliberately
**not** in `crates/axon-model/tests/bundles`, because everything there is asserted green by the
Rust gate on every `cargo test`, and one of these is a bundle `tract` is expected to refuse.

**8. The evidence above only became evidence once the export path stamped the version.**

This ADR's first draft measured a crossing that could not have happened through the registry's own
export path, and the ADR should say so rather than quietly acquire a corrected number.

Both Rust backends refuse a model that cannot name its own version and read it out of the artifact
— ONNX's `model_version`, and XGBoost's `axon_model_version` learner attribute (ADR-0019 §4).
**Neither is written by any library's own serializer, and `export_artifact` wrote neither.** So
every artifact the export path produced was unloadable by the core, and every bundle the
cross-language gate had ever run on — including the `lgbm_binary` this ADR rests on — had been
hand-stamped by its generator. The gate that certifies the boundary had never run on the path it
certifies. Found by the P2 zoo, which hit the refusal on a model with nothing else wrong with it.

`export_artifact` now writes `meta.version` into the payload before the payload is hashed. There
was a live disagreement about whether it should, and it is worth recording how it resolved,
because both positions were about the same word:

- The comment arguing *against* lived in `narrow_to_score_column`, which is a graph-shape rewrite
  with no version in scope. Stamping **there** would have been inventing one, and that comment was
  right about its own function. It has been rewritten to say the thing that is now true: the field
  is *carried across*, because `make_model` starts from an empty `ModelProto` and dropping it
  would make the rewrite that exists to render an artifact loadable the same rewrite that renders
  it unloadable.
- `export_artifact` is *handed* `meta.version`, which `meta.validate()` has already required to be
  ≥ 1 and which the registry will mint the artifact under. Writing it into the bytes is
  **recording**, not inventing, and it is what makes drift impossible rather than what risks it:
  the version in the payload, in the metadata and in the registry path become one fact with one
  source.

Two properties were checked rather than assumed. The XGBoost stamp goes on `booster.copy()`, so
the caller's live model is not mutated — an exporter that rewrites the object a notebook is still
predicting with is a surprise nobody budgets for. And **regenerating all three committed bundles
under `crates/axon-model/tests/bundles` produces byte-identical artifacts**, because their
generator already hand-stamped the same numbers: the strongest available demonstration that this
records rather than invents.

**9. A converter that names its graph with a UUID makes a content hash identify a conversion
rather than a model. Both of ours did it.**

`onnxmltools.convert_lightgbm` and `skl2onnx.convert_sklearn` both default `name` to
`uuid4().hex`, and it lands in `graph.name`. So two conversions of one fitted model on one stack
produce artifacts that differ in that field and in **nothing else** — same weights, same
predictions, same length, different bytes, different `content_sha256`.

That is worse than a churning fixture, and the wider harm is the reason this is a numbered section
rather than a footnote. **[ADR-0015](0015-model-artifacts-and-registry.md) opens by requiring that
a version resolve to one exact set of bytes, forever.** Under a UUID graph name, the content hash
identifies a *conversion event* instead of a model: two exports of one estimator are two artifacts
by identity. Nothing catches it, because `Artifact.verify()` compares the payload against the hash
that was taken *over that payload* and is satisfied either way. And downstream it makes every
frozen reference over such an artifact churn on regeneration, which is what a committed sklearn
parity bundle was blocked on.

Found twice, and the second time is the instructive one. The LightGBM call site was fixed on its
own evidence; the sklearn call site four screens away in the same file had the identical defect and
was found later, by another agent, on a bundle it could not commit. The two pins now carry the
same comment and cross-reference each other, because *two converters with one default* is one
decision, and a fix applied at one call site of a shared library idiom is a fix that has not been
made.

Measured after the pin, three separate interpreter invocations: identical `content_sha256`,
identical bundle files, byte for byte. Regenerating the three pre-existing committed bundles still
produces byte-identical artifacts, because none of them routes through either converter's
defaulting path — `generate.py` calls `skl2onnx.to_onnx`, which names its graph
`ONNX(MLPRegressor)` and was always deterministic. This was found by re-running a generator twice,
which is worth doing to anything called a frozen reference.

## Consequences

- **+** The "missing LightGBM backend" was a routing question. There is a working crossing today,
  end to end, with `tract` reproducing Python's frozen reference to **1.19e-07 over 256 rows and
  zero decision flips** — and no Rust code was written for it.
- **+** `lightgbm3-rs` is not worth arguing for on the evidence available. The one thing it would
  buy over this route is the regression objective, and the cheaper fixes for that (a `tract`
  operator, or a tensor-only conversion) have not been tried.
- **+** Three refusals that had never fired against a real graph have now fired against one, and
  the reading-side one is tested in the form where truncation would have *passed*.
- **+** The score-space trap is closed with a number rather than a warning: 6.63e-08 against the
  probability, 1.41 against the margin, both in the same test.
- **+** **Every artifact the export path produces is now loadable by the Rust backends**, which
  was not true of any of them before (§8). That is a larger fact than this ADR's own subject: it
  applies to XGBoost and to every sklearn export, not to LightGBM. **ADR-0019 §4 should be amended
  to say that the exporter writes the field the backends read** — its claim was true of the
  backends and unimplemented on the other side of the boundary for as long as both have existed.
- **+** `narrow_to_score_column` is public and general, so the linear sklearn family crosses too,
  at 1.49e-07 over 512 rows with zero decision flips (§4a).
- **+** **Every artifact the export path produces now has a reproducible `content_sha256`**, which
  is what ADR-0015 always claimed and what no skl2onnx export had (§9). A committed sklearn parity
  bundle is unblocked: a narrowed `LogisticRegression` over 348 rows reproduces in `tract` at
  1.79e-07 with zero decision flips, and its bundle is byte-identical across three separate runs.
- **−** `narrow_score_output` defaults to **off**, so nothing crosses until a caller asks. Turning
  it on for a family is a one-line change at the call site, and `axon.strategies.zoo` has not made
  it: `test_zoo.py` asserts by name that the sklearn families *cannot* cross, and its docstring
  says that test "will start failing the day somebody teaches the export path to slice the
  positive-class column, and that is the correct moment to read ADR-0032 again." That moment has
  arrived; the flip belongs with whoever owns that ADR and that test, not here.
- **−** Narrowing removes the export round trip's `argmax` decision-invariance check, because the
  reference collapses to one column with the graph. For a binary classifier that check is
  equivalent to a 0.5 threshold on the kept column, so little is lost — but "little" is not
  "nothing", and the full decision gate is the parity harness's, not the exporter's.
- **−** **The crossing is binary-objective only.** A regression booster is refused at Rust load,
  by design and with a name, but it is still a family a researcher can fit, gate green in Python,
  and only then discover cannot be served. Nothing warns earlier, deliberately (§6).
- **−** The route is 1e-5, not bit-exact. XGBoost crosses bit for bit; LightGBM crosses inside a
  tolerance, so the two families are held to genuinely different standards and a diff between them
  is not comparable. That is inherent in converting a tree ensemble, and it is the cost ADR-0003
  declined to pay for XGBoost precisely because XGBoost did not have to.
- **−** The measurements are on **eight trees and seven leaves**. `ai.onnx.ml`'s tree evaluation
  accumulates leaf contributions in float32, and `onnxmltools` ships a `split` parameter
  specifically for large ensembles whose sums drift; a 500-tree booster has not been measured and
  should not be assumed to land in the same place.
- **−** `onnxmltools` 1.16 tops out at target opset 15, so a LightGBM graph is pinned below the
  opset the sklearn exports use (17). The emitted default-domain opset is lower still. Nothing
  depends on it today; it is a ceiling that moves with a package, which is why
  `LIGHTGBM_TARGET_OPSET` is a named constant rather than a call to
  `get_maximum_opset_supported()`.
- **−** Neither `lightgbm_to_onnx` nor `narrow_to_score_column` is re-exported from
  `axon.models.__init__`, so callers import them from `axon.models.export`. That is a two-line
  follow-up, left undone because the package's `__init__` was owned by another workstream in the
  same session.
- **−** The parity bundle still cannot carry a NaN into a graph, although this route demonstrably
  routes missing values correctly on all three sides. The guard is right for neural graphs and
  conservative for tree graphs, and separating the two needs an argument per operator.

See [ADR-0019](0019-native-rust-inference-backends.md) (the unbuilt backend this answers, the
one-input/one-output rule, and `tract`'s operator ceiling),
[ADR-0003](0003-model-serving-and-fidelity.md) (the 1e-5 graph tolerance and "trees keep their own
format"), [ADR-0005](0005-fp32-no-quantization.md) (the FP32 audit the converted graph passes),
[ADR-0015](0015-model-artifacts-and-registry.md) (`score_output`, which is why nothing downstream
gets to guess a column), and [ADR-0021](0021-rust-model-parity-gate.md) (the bundle format these
fixtures are written in).
