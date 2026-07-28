# ADR-0032 — The model zoo, and what actually crosses into Rust

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0022](0022-first-ml-strategy.md) took one model — an XGBoost bar classifier — through the
whole ladder: a versioned feature spec, a purged walk-forward, an exported artifact, three green
gates, and an honest verdict that it should not trade. Everything this repo believes about "the
machinery" rests on that single case, and a single case cannot tell a property of the machinery
from a property of XGBoost.

The Phase-6 brief asked for the width, and it carried a matrix predicting the answer:

| Family | Registry | Python predict | Rust gate | Route |
|---|---|---|---|---|
| XGBoost | ✅ native | ✅ | ✅ bit-identical | proven by `perp_bar` |
| sklearn GradientBoosting | ✅ | ✅ | **✅ via ONNX** | untested end to end |
| sklearn LogisticRegression | ✅ | ✅ | **✅ via ONNX** | untested; the best canary |

Two of those three ticks were assumptions, and the reasoning behind them is sound as far as it
goes: `SERVABLE_KINDS` is `("xgboost", "onnx")`, `export_artifact` already routes any sklearn
estimator through `skl2onnx`, and the artifact that comes out is of kind `onnx`. Every step is
true. The conclusion is wrong, and it is wrong in a way no Python process can see.

There was also a second question the brief named as a trap rather than as an open item. The
serving path acts on a probability; a parity bundle for a tree records a **margin**
(`SCORE_SPACE = {"xgboost": "margin", "onnx": "score"}`), because Rust's `TreeModel` never
applies the link. A comparison across those two spaces reports the link function as a parity
failure of several hundred percent, which reads as a catastrophically broken model.

And one more, which nobody had named at all: **an artifact produced by this repo's own export
path cannot be loaded by either Rust backend.** More on that in §5.

Four questions had to be answered, and three of them had a comfortable wrong answer.

1. **Which bar interval?** The comfortable answer is "the one `perp_bar` uses". `perp_bar`'s
   longest window is 24 bars, which on hourly candles is *no opinion for the first 25 hours* —
   a session must outlive a day before the strategy says anything at all.
2. **Where does the shared spec live?** The convention (`PERP_CORE_V1`) says a strategy defines
   its own and pins it in its artifact. That convention produces two specs the day one copy is
   edited, and this one is shared by construction.
3. **What does "crosses into Rust" mean, and who is allowed to answer it?** The comfortable
   answer is the artifact `kind`, which is checkable in Python in one line.
4. **When may a transcript say a model is worth trading?** The comfortable answer is a positive
   number.

## Decision

**1. One shared spec, six columns, on m1 bars, with a 21-bar warmup — and the warmup is a test.**

`BAR_M1_V1` lives in `axon.features.spec` beside `PERP_CORE_V1`, not in a strategy module. Three
strategies are held to it — the zoo's three families — and a fourth workstream reads it; a spec
copied into two modules is two specs, and the fingerprint moves on one side only.

```
bar_m1/v1#c503688de24e863f
  ret_1      log_return          period=1        on close
  mom_5      momentum            window=5        on close
  z_20       rolling_zscore      window=20       on close
  vol_20     realized_volatility window=20       on close
  range_bps  relative_range                      on high/low/close
  clv        close_location                      on high/low/close
```

Every window is finite and the longest is 21 bars: `vol_20` is a 20-sample standard deviation of
one-step log returns, so its first window reaches back through the return's own extra bar and
the first finite row is index 20. **On m1 that is a 21-minute warmup. The same arithmetic on `1h`
would be 21 hours.** That single choice is the difference between a session that can be observed
in an afternoon and one that cannot, and it is the reason the interval is named in the module
rather than defaulted.

The number 21 is stated as `BAR_M1_WARMUP_BARS` and asserted two ways: against real venue bars in
`test_zoo.py`, and against `axon.strategies.baseline.warmup_bars`, which drives a synthetic
strictly-varying ramp through the spec so that any NaN in its answer is the window rather than
the data. The two agreeing is what makes 21 a property of the recipe. The failure this prevents
is silent by construction: a docstring saying "21 bars" goes stale the day a window widens, the
serving buffer is then one bar short, every row stays NaN, the strategy emits nothing forever,
and **nothing raises** — a strategy that never trades looks exactly like a strategy with no
opinion.

Nothing in the spec is an EMA or an expanding statistic, which is what lets a bounded serving
buffer reproduce the offline recompute *bit for bit* rather than within a tolerance. The
counter-example is asserted in the same test as the property, so the equality cannot be read as
a fact about the comparison.

**2. The serving path is `PerpBar`, and the criterion is derived from the artifact kind.**

`PerpBar` is already parameterized on its `FeatureSpec` and its `Predictor` and holds nothing
hourly; a second serving class for the zoo would be a second implementation of the exact
comparison the feature-parity gate exists to make. All three families are served through it,
with one entry band and one label horizon — a comparison in which two models were also given two
different bands measures the bands.

`Criterion.required_for(kind)` is the only source of the bar: `bit_exact` for a tree ensemble,
because [ADR-0019](0019-native-rust-inference-backends.md) claims `TreeModel` reproduces
`Booster.predict(output_margin=True)` exactly and a tolerance there would be the gate declining
to test its own claim; `max_abs_diff <= 1e-5` for a graph. A family cannot ask for a looser one.

**3. Whether a family crosses into Rust is decided by the ONNX *operator*, not by the artifact
kind — and this was measured against `tract` 0.23.4, not reasoned about.**

`tract` 0.23.4, the version `Cargo.lock` resolves, implements exactly five operators from the
`ai.onnx.ml` domain: `CategoryMapper`, `LinearClassifier`, `LinearRegressor`, `Normalizer`,
`TreeEnsembleClassifier`. Classical sklearn models compile to that domain. Neural nets do not —
they compile to core `ai.onnx` (`Gemm`, `Relu`, `MatMul`), which is why the ONNX parity bundle
that had been committed longest is an `MLPRegressor`, and why it proved less about the ONNX
route than it appeared to: it never touched `ai.onnx.ml` at all.

Measured, family by family, on the zoo's own artifacts, through `axon-model`'s own loaders:

| Family | Bundle writes? | Rust loader | Verdict |
|---|---|---|---|
| **XGBoost** (`gbtree`, binary) | ✅ bit-exact, 512 rows | `TreeModel` loads: 150 trees, `objective=binary:logistic`, `link=Logit` | **crosses** |
| **LogisticRegression** → `LinearClassifier` | ✅ narrowed | `OnnxModel` loads | **crosses** |
| **sklearn GBM** → `TreeEnsembleClassifier` | ✅ narrowed | ❌ `attribute 'base_values': expected length 1 (or undefined), got 2` | **does not cross** |
| sklearn GBM → `TreeEnsembleRegressor` | ✅ | ❌ `Unimplemented(TreeEnsembleRegressor)` | **does not cross** |
| LogisticRegression behind a `StandardScaler` | ✅ narrowed | ❌ `Unimplemented(Scaler)` | **unreachable** |

**A classifier's graph must be narrowed to one score column, and that is a declaration rather
than a conversion.** `skl2onnx` gives *every* classifier two outputs, `label` and
`probabilities`, even with ZipMap disabled, and both sides of the boundary refuse it —
`rust_gate.python_scores` with `the score output has 2 values per row; a parity bundle records
one score per row`, and `OnnxModel` with `unsupported model artifact: graph declares 2 outputs;
exactly one is required`. Neither is a bug: both are
[ADR-0015](0015-model-artifacts-and-registry.md)'s refusal to guess which number a strategy
trades on. `export_artifact(..., narrow_score_output=True)` is that guess turned into a statement
at the call site, and the zoo's families may make it **because they are binary by construction** —
a two-column classifier has exactly one positive column, so there is nothing to infer. A
multi-class model has no positive column and the rewrite refuses more than two outright.

Beyond the narrowing there are two refusals left, and they are not the same refusal:

- **An operator that is not in the registry.** The boosted tree's regressor form and any
  `StandardScaler` fail at *parse* time, and no change to the export path can reach them.
- **An attribute the converter and the runtime disagree about.** The boosted tree's classifier
  form, narrowed, is refused on `base_values` length. **`TreeEnsembleClassifier` is one of
  tract's five**, so this one is not an operator-coverage gap at all, and no reading of the
  operator list predicts it.

**A written bundle is not a crossing, and the transcript must print both.** This is the row this
workstream got wrong first. Before the narrowing, the boosted tree was refused early and loudly
by the bundle writer; after it, `write_parity_bundle` succeeds and `tract` still refuses the same
bytes at load. A gate table with one `rust=` column would have printed `bundle ok` for a model
the core cannot serve — the silent green this repo keeps finding. `FamilyResult.gate_row` now
carries `bundle=` (what this process measured) and `rust=` (what the loader did), and they
disagree for exactly one family.

**`Family.crosses` is declared out of band, and Python must not pretend to derive it.** It is
tempting to think it can — the refusal is an attribute check and the attribute is in the graph —
but `base_values` is length **1** in the bytes while tract reports `got 2`, so the number it
compares is one it derives internally. A predicate reimplementing that derivation would agree
today and be a guess wearing a check's clothes on the next `tract` bump. Instead the claim is
dated to `TRACT_VERSION`, and two tests keep it honest: one asserts the three preconditions
Python *can* check for every family declared to cross (one FP32 output of one column, a non-zero
`model_version`, no `ai.onnx.ml` operator outside the five) — those being the three that were
each silently violated by a different family — and one pins the exact `base_values`/`classlabels`
shape the refusal was measured against, so it reddens when the graph moves rather than when
somebody remembers to re-measure.

**Do not route around the `base_values` refusal.** Both obvious repairs were measured and both
are traps: deleting the attribute builds and agrees with onnxruntime to seven decimals — and it
is the model *without its intercept*, a green gate over the wrong model; padding it to two
entries builds and then `tract` says `0.32063872` where onnxruntime says `0.5`, a divergence
18 000× the tolerance. Only the parity gate catches either. It is not a conversion problem.

The general statement, which is not what anyone assumed: **a linear model crosses without
ceremony once its graph declares one column; a boosted tree crosses only as a
`TreeEnsembleClassifier` whose converter agrees with tract about `base_values`, and of the
converters in this tree only LightGBM's does** (see
[ADR-0033](0033-lightgbm-crosses-by-conversion-not-by-backend.md)).

Two consequences for the code:

- The zoo's linear family is a **bare** `LogisticRegression`. It shipped behind a
  `StandardScaler` first, which is the textbook move and legitimate in principle — a
  standardization fitted on training rows travels *inside* the artifact rather than becoming a
  seventh feature no `feature_spec_ref` describes. It was measured out anyway, because it costs
  the family its only route into Rust and it bought nothing: lbfgs converges in 8 iterations on
  the raw columns against 7 on the scaled ones, and the fitted probabilities have the same spread
  to three decimal places. **The conditioning argument is real in general and false here**, and
  only a measurement could say which.
- The zoo declares `narrow=True` for both sklearn families. For the linear model that is the
  whole difference: its bundle now goes end to end through `axon-model`'s own reader,

  ```
  cross-language model parity PASS: zoo_logistic@1 (onnx) n=512
  criterion=max_abs_diff <= 1e-5 max_abs_diff=5.9604645e-8 (row 310) over_criterion=0
  flips=0 non_finite=0
  ```

  — 168× inside the criterion, with no decision flipped. For the boosted tree the same flag buys
  a bundle and no crossing, which is why the two are recorded in separate columns.

**4. Two score spaces, and the conversion is deliberate in exactly one place.**

The strategy always sees a probability: `XgboostPredictor` applies the link, an ONNX classifier
graph emits both class columns, and `zoo.positive_class` is the single place either shape becomes
`P(up)`. Taking column zero instead would be `P(down)` — a strategy that trades the exact inverse
of its model, with no symptom other than losing money and nothing any gate can see, because both
columns are finite, in range and stable across the export.

The bundle is the other way round, on purpose. `zoo.margin_versus_probability` *measures* the gap
rather than describing it, and `test_zoo.py` pins the number, so the trap cannot quietly stop
being one.

**5. The registry's own bytes now carry the version both Rust backends read. They did not when
this workstream started, and that is why the cross-language gate had never run on the path it
certifies.**

`OnnxModel` reads ONNX's first-class `model_version` field; `TreeModel` reads an
`axon_model_version` learner attribute. `export_artifact` writes **neither**. Every artifact this
repo's own export path has ever produced is refused at load:

```
model artifact carries no version (axon_model_version); an unversioned model cannot be
reproduced offline, so it is refused
```

The committed parity bundles passed only because `crates/axon-model/tests/bundles/generate.py`
stamps both by hand before exporting, and says so in its own comments. **A registry path that has
to be re-implemented in a fixture generator before Rust will load it is not a registry path Rust
has ever certified**, and every ✅ this project has recorded for the cross-language gate was
therefore a statement about the generator.

The objection to fixing it was reasonable and turned out to be about a different function:
*"inventing one here would let it drift from the registry version the caller is about to mint."*
That holds inside a graph-shape rewrite, where no version is in scope. It does not hold in
`export_artifact`, which is **handed** `meta.version` — `meta.validate()` has already required
it to be ≥ 1, and it is the number the registry will mint the artifact under. Writing it into the
bytes *before* they are hashed is what makes drift impossible rather than what risks it: from
there on the version in the payload, in the metadata and in the registry path are one fact with
one source.

It is now stamped, in `export_artifact` and nowhere else — a stamp you must remember to call is
the same failure wearing a different name. Two things were checked rather than assumed: the
XGBoost stamp goes on a `booster.copy()`, so a notebook's live booster is not mutated by
exporting it; and regenerating all three previously-committed bundles through the stamped path
produced **byte-identical artifacts, 15/15 files**, which is what says the stamp reproduces what
the generator was doing by hand rather than merely resembling it.

The zoo's own XGBoost artifact, straight off the unassisted export path, now goes all the way:

```
cross-language model parity PASS: zoo_xgboost@1 (xgboost) n=512
criterion=bit-exact max_abs_diff=0e0 over_criterion=0 flips=0 non_finite=0
```

**6. A verdict clears the cheapest fee schedule, or it says SHOULD NOT TRADE. No AUC.**

`FamilyResult.verdict()` prices the model's edge over its own decision rows against a constant
short on *the same rows*, and holds the difference against the fee drag implied by the turnover
actually measured on the replay — counted from the signals the hysteresis emitted, not from
re-thresholding the scores, because the target depends on the previous target.

The bar is the maker schedule rather than zero, and that is not pedantry: all three families
came out with a **positive** selection and all three should not trade, because the drag is three
to twelve times larger. "Selection is positive" is precisely the sentence that would be quoted
out of this transcript. A ranking statistic appears nowhere in it —
[ADR-0022](0022-first-ml-strategy.md) exists because an AUC of 0.52 was once read as a working
strategy, and a ranking statistic cannot come out negative for a model that is merely riding a
drift. This subtraction can.

## The zoo's floor: a strategy with no model in it

*Reserved. `axon.strategies.baseline` (P4) supplies this section complete; the orchestrator
splices it here without renumbering the sections above or below.*

## The transcript

Three coins (BTC, ETH, SOL), 15 142 m1 bars of real mainnet `candleSnapshot` history with zero
gaps, 12 007 out-of-sample rows over four purged folds, a ten-bar label horizon, one entry band.

| Family | Exports | Model parity | Feature parity | Drift | Bundle | Crosses into Rust | Criterion |
|---|---|---|---|---|---|---|---|
| XGBoost | ✅ `xgboost`, 162 079 B | ✅ 0.0 | ✅ 0.0, 15 029/15 029 | ALARM | ✅ | ✅ **PASS** bit-exact, 512 rows, 0 flips | `bit_exact` |
| sklearn GBM | ✅ `onnx`, 66 279 B | ✅ 1.32e-7 | ✅ 0.0, 15 029/15 029 | ALARM | ✅ | ❌ `base_values: expected length 1 (or undefined), got 2` | `max_abs_diff` 1e-5 |
| LogisticRegression | ✅ `onnx`, 517 B | ✅ 5.96e-8 | ✅ 0.0, 15 029/15 029 | ALARM | ✅ | ✅ **PASS** 5.9604645e-8 (row 310), 0 over, 0 flips | `max_abs_diff` 1e-5 |

The **Bundle** and **Crosses** columns exist separately because they disagree for one family, and
a single column would have reported that family as servable.

Feature parity is **0.0** on every coin for every family, at complete coverage — the finite-lookback
rule doing exactly what it was chosen for. Drift alarms identically for all three, because it is a
property of the sample and not of the model: `vol_20` PSI 0.7726 and `range_bps` 0.4630 between the
first three folds and the last, inside a single 3.5-day window. That is the alarm working; a market
that moved is not a bug, which is why drift is not part of `passed`.

And the verdicts, which are the point:

```
verdict [xgboost]:     SHOULD NOT TRADE  selection +0.29bps  drag 1.02/3.05bps  turnover 3190/15142
verdict [sklearn_gbm]: SHOULD NOT TRADE  selection +0.09bps  drag 0.97/2.92bps  turnover 2995/15142
verdict [logistic]:    SHOULD NOT TRADE  selection +0.25bps  drag 0.61/1.82bps  turnover 1833/15142
```

Three families, every Python gate green, and not one of them worth an order. That is a
**successful** outcome for this phase, and it is the same shape `perp_bar` produced at a
different interval with a different model.

## Consequences

- **+** The three gates are now known to be properties of the machinery rather than of XGBoost.
  Feature parity is exactly 0.0 across two libraries and three model classes on the same spec,
  which is a much stronger statement than one family passing it.
- **+** **Two of the three families cross, both measured end to end through `axon-model`'s own
  loaders**, and the second one only because the boundary was pushed rather than described: the
  linear model's ✅ took a graph rewrite in the export path and the removal of a scaler nobody
  had suspected. The brief's matrix was right about it and right for the wrong reason.
- **+** The Phase-6 brief's matrix is corrected with measurements rather than with a caveat, and
  the correction is one row rather than two. `Unimplemented(TreeEnsembleRegressor)` and
  `Unimplemented(Scaler)` can never be fixed on the Python side; `base_values` is a converter
  disagreement about an operator `tract` *does* implement, which no reading of the operator list
  predicts.
- **+** The linear model was the best canary, as the brief guessed. It is not simple enough to be
  uninteresting; it is the classical family whose every operator `tract` implements, so it
  isolated the narrowing blocker from everything else — and once narrowed it crossed, which is
  what turned "sklearn cannot cross" from a conclusion into a much narrower claim about one
  converter's `base_values`.
- **+** A 21-minute warmup makes the whole zoo observable in one sitting. The m1 cache is 3.5 days
  and the venue serves ~5 000 candles per interval whatever start time is asked for, so the same
  row budget buys minutes of latency instead of days.
- **+** `SHOULD NOT TRADE` under a positive selection is now the default reading, and the rule is
  in code rather than in a reviewer's head.
- **−** `Family.crosses` is a **declaration**, not something the Python gate derives, and it will
  be wrong the day `tract` changes and nobody re-runs a Rust binary. Three preconditions and one
  characterization test narrow the window; they do not close it. The honest close is a committed
  zoo bundle in `cargo test`.
- **−** Both cross-language passes were run from a scratch binary linking `axon-model`, not from
  `cargo test`. Adding a bundle to the committed gate means editing
  `crates/axon-model/tests/cross_language_parity.rs`, whose bundle list is deliberately spelled
  out rather than globbed, and that file belonged to another workstream on this fan-out.
  **Nothing here has been run against a venue**, and the one thing that has been run against a
  venue in this whole area is `perp_bar`'s `MdBar` feed.
- **−** Narrowing moved the boosted tree's refusal *later* — from the bundle writer, where it was
  loud and early and this process could see it, to the Rust loader, where it cannot. That is the
  correct trade (the linear model crosses because of it) and it is still a loss: the gate table
  now depends on a column no Python test can compute, and the first version of that table said
  `bundle ok` for a model the core refuses.
- **−** `zoo.walk_forward` is a near-copy of `training.walk_forward_fit`, differing only in that
  it takes a fitter. Everything shareable is imported rather than restated, but the loop is
  duplicated and the honest fix is to give `walk_forward_fit` a fitter and delete the copy.
- **−** The drift alarm fires on all six columns for a reason that has nothing to do with the
  models, and it will fire on every run over this fixture. That makes it a poor smoke test for
  drift *regressions*, and the test that asserts it now carries a note saying so.
- **−** 3.5 days of m1 is not a sample anything can be concluded from about edge. It is the
  venue's cap per interval, not a choice, and the walk-forward's four folds are four windows of
  one market regime. The gates are what this run proves; the basis points are decoration on a
  verdict that was going to be negative at any sample size.

See [ADR-0022](0022-first-ml-strategy.md) (the first family, and the AUC reading this exists to
prevent), [ADR-0016](0016-feature-spec-and-parity-gates.md) (the three gates and the spec
fingerprint), [ADR-0021](0021-rust-model-parity-gate.md) (the bundle, the score spaces and the
criterion table), [ADR-0019](0019-native-rust-inference-backends.md) (the two backends and what
the tree reader does not cover), [ADR-0015](0015-model-artifacts-and-registry.md) (`score_output`,
whose refusal to guess is half of §3), and [ADR-0033](0033-lightgbm-crosses-by-conversion-not-by-backend.md)
(the same boundary approached from the other side).
