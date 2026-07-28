"""The model zoo: one feature recipe, three model families, the same four gates.

``perp_bar`` proved that *a* model can be fitted, exported, registered and gated.
It proved it once, for one family, and every claim the fidelity ladder makes about
"the machinery" rests on that single case. This module is the width: the **same**
:data:`~axon.features.BAR_M1_V1` matrix, the **same** serving path, the **same**
walk-forward, fitted by XGBoost, by sklearn's :class:`GradientBoostingClassifier`
and by a bare :class:`LogisticRegression`, each taken through model parity, feature
parity, drift, and the cross-language bundle of ADR-0021.

**The transcript is the deliverable, and the numbers are not.** A family that
exports, passes every gate and loses money has told us the plumbing is sound; a
family that cannot cross into Rust has told us where the boundary actually is. Both
are results. ADR-0022 exists because an AUC quoted as an outcome reads as success,
so no function here returns one on its own — :meth:`FamilyResult.verdict` prices the
selection in basis points against a benchmark that needs no model, which is the
comparison that can come out negative.

Four things here are not obvious, and each one is a mistake somebody would
otherwise make exactly once:

**The artifact kind does not decide whether a family crosses; the graph does.**
``SERVABLE_KINDS`` is ``("xgboost", "onnx")``, so an sklearn model looks servable the
moment ``skl2onnx`` accepts it — and the measured answer (ADR-0032) is that the ONNX
*kind* is necessary and nowhere near sufficient. Two of the three families cross:
XGBoost natively and bit-exact, and the linear model through ``skl2onnx`` once its
graph is narrowed to one score column. The boosted tree does not, and it fails
*third* — past the converter, past the bundle writer, at an attribute `tract` checks
on load. That is a fact about the serving runtime rather than about the model family,
which is why the ``route`` strings in :data:`FAMILIES` name operators and why
:attr:`Family.crosses` is dated to :data:`TRACT_VERSION`.

**The two families answer in different spaces, and only in the bundle.** The
strategy always sees a probability: ``XgboostPredictor`` applies the link and an
ONNX classifier graph emits both class columns, so :func:`positive_class` is the one
place either shape becomes ``P(up)``. The *bundle* is different on purpose —
``SCORE_SPACE`` is ``{"xgboost": "margin", "onnx": "score"}`` because ``TreeModel``
in Rust never applies the link. A margin compared against a probability reports the
**link function** as a parity failure, several hundred percent wide, and it looks
exactly like a broken model. :func:`margin_versus_probability` measures that gap
rather than describing it, so the test that pins it cannot go stale.

**Plain ``gbtree``, binary, numeric.** The native Rust reader does not cover ``dart``
boosters, categorical splits or multi-output ensembles (ADR-0019). Nothing here uses
any of them, and :data:`XGB_PARAMS` says so in the parameters rather than in a
comment nobody re-reads.

**The criterion is derived from the family, never chosen.** ``Criterion.declared_for``
is the only source: ``bit_exact`` for a tree ensemble, because ADR-0019 claims exact
reproduction and a tolerance there would be the gate declining to test its own claim;
``max_abs_diff <= 2**-22`` — two ULP — for a graph, because two runtimes never agree
bit for bit but these three agree to within one. A family cannot buy itself a looser
bar by asking for one, and the *ceiling* it may never exceed is still
``Criterion.required_for``'s 1e-5 (ADR-0021's 2026-07-26 amendment).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from axon.features import BAR_M1_V1, BAR_M1_WARMUP_BARS, FeatureSpec
from axon.models import ArtifactMeta, ModelRegistry, export_artifact
from axon.models.artifact import Artifact
from axon.models.inference import load_predictor
from axon.parity import (
    Criterion,
    DriftReport,
    FeatureParityReport,
    ModelParityReport,
    ParityBundle,
    model_parity,
    quantile_decision,
    read_parity_bundle,
    threshold_discretizer,
    write_parity_bundle,
)
from axon.strategies.data import INTERVAL_MS, Candles
from axon.strategies.labels import purged_walk_forward
from axon.strategies.perp_bar import PerpBar, PerpBarParams
from axon.strategies.training import (
    Dataset,
    Decomposition,
    Evaluation,
    TrainingError,
    WalkForward,
    build_dataset,
    decompose,
    drift_gate,
    feature_parity_gate,
    score_fold,
)

#: The `tract` build every crossing claim in :data:`FAMILIES` was measured against,
#: pinned in ``crates/axon-model/Cargo.toml`` with ``=`` for the reason ADR-0019 gives:
#: ONNX operator semantics *are* the numbers, so a patch bump can move a result with no
#: code change here to blame it on. It is named because the claims are dated to it —
#: "does not cross" is a statement about a runtime version, not about a model family.
TRACT_VERSION = "0.23.4"

#: The bar interval this zoo is fitted and served on. Named rather than defaulted:
#: the interval decides the warmup (:data:`~axon.features.BAR_M1_WARMUP_BARS` bars is
#: 21 minutes here and would be 21 *hours* on ``1h``), and a session that has to
#: outlive its own warmup before it says anything is a session nobody observes.
INTERVAL = "1m"

#: The forward horizon the label is taken over, in bars — ten m1 bars, so the question
#: is "does this close higher ten minutes from now". Stated because a strategy whose
#: holding period does not resemble its label horizon is trading a different question
#: from the one the model was asked, and the hysteresis band below is what makes the
#: holding period land in the same order of magnitude.
LABEL_HORIZON_BARS = 10

#: Bars the serving path holds. The binding constraint is
#: :data:`~axon.features.BAR_M1_WARMUP_BARS`; this is deliberately six times that,
#: because a buffer sized to the exact minimum silently becomes wrong the day someone
#: widens a window, and the symptom is a strategy that emits nothing, forever, without
#: an error. ``PerpBar`` probes for that at construction.
SERVING_BUFFER_BARS = 128

#: The strategy parameters every family in the zoo is served with. Identical across
#: families on purpose: a comparison in which two models were also given two different
#: entry bands measures the bands.
ZOO_PARAMS = PerpBarParams(
    symbol_id=0,
    max_position=Decimal("0.001"),
    entry_edge=0.02,
    exit_edge=0.005,
    urgency=0,
    ttl_ms=60_000,
)

#: XGBoost, deliberately unremarkable. ``booster="gbtree"`` and
#: ``objective="binary:logistic"`` are written out rather than left to the default
#: because they are the two settings the native Rust reader depends on: ``dart``
#: re-weights trees at prediction time and multi-class emits a score per class, and
#: neither is implemented on the Rust side (ADR-0019).
#:
#: ``n_jobs=1`` is not a performance setting. Histogram building parallelizes with a
#: non-deterministic reduction order, so a multi-threaded fit produces a slightly
#: different model on every run and no artifact hash is reproducible.
XGB_PARAMS: Mapping[str, Any] = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 60,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 7,
    "eval_metric": "logloss",
}

#: sklearn's own boosting, sized to match the XGBoost model rather than to win against
#: it. ``max_features=None`` and a fixed ``random_state`` make the fit a pure function
#: of the training rows; ``subsample=1.0`` removes the only other source of randomness,
#: so the two libraries differ in their arithmetic and not in what they were shown.
GBM_PARAMS: Mapping[str, Any] = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_samples_leaf": 60,
    "subsample": 1.0,
    "max_features": None,
    "random_state": 7,
}

#: The linear model, and **no scaler**, which was a deliberate reversal.
#:
#: A ``StandardScaler`` in front of it is the textbook move here — the columns are on
#: wildly different scales (``ret_1`` ~1e-4 against ``range_bps`` ~10) — and it is
#: legitimate in principle, because a standardization fitted on training rows travels
#: *inside* the artifact rather than being a seventh feature no ``feature_spec_ref``
#: describes. It was measured out anyway, for two reasons that only a measurement
#: gives you:
#:
#: * it costs the family its only route into Rust. ``skl2onnx`` compiles the scaler to
#:   an ``ai.onnx.ml`` ``Scaler`` node, and ``tract`` 0.23.4 implements exactly five
#:   operators from that domain — ``CategoryMapper``, ``LinearClassifier``,
#:   ``LinearRegressor``, ``Normalizer``, ``TreeEnsembleClassifier``. Loading the
#:   scaled graph fails with ``Unimplemented(Scaler)``, and no fix to the export path
#:   can reach that. Without it the graph is a single ``LinearClassifier`` node, which
#:   ``tract`` does run;
#: * it bought nothing. lbfgs converges in 8 iterations on the raw columns and in 7 on
#:   the scaled ones, and the fitted probabilities have the same spread to three
#:   decimal places. The conditioning argument is real in general and false here.
#:
#: ``max_iter`` stays generous so that a future spec with worse conditioning stops on
#: its tolerance rather than silently on its iteration cap — a linear model that hit
#: the cap is a different model from the one the numbers describe, and sklearn says so
#: only in a warning.
LOGIT_PARAMS: Mapping[str, Any] = {
    "max_iter": 2_000,
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": 7,
}


# ── the families ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Family:
    """One model family, and everything that differs about taking it up the ladder.

    ``kind`` is what :func:`~axon.models.export.export_artifact` will *detect*, not a
    request — it is recorded here so that a family which silently changed route (an
    sklearn model that stopped converting, a tree that arrived as a graph) fails as a
    mismatch rather than as a surprising tolerance somewhere downstream.
    """

    name: str
    kind: str
    #: A fresh, unfitted estimator. A callable rather than an instance because a
    #: shared estimator refitted per fold is one model wearing four folds' names.
    build: Callable[[], Any]
    #: Why this family is in the zoo at all, printed in the transcript.
    route: str
    #: Rewrite the exported graph down to its positive-class column.
    #:
    #: Declared per family and never inferred, which is the whole point of the flag:
    #: ADR-0015's ``score_output`` exists so that nothing *guesses* which number a
    #: strategy trades on, and ``export_artifact`` keeps the rewrite off by default
    #: for exactly that reason. Saying ``True`` here is the statement the principle
    #: asks for, and it is only available to say because these families are **binary
    #: by construction** — a two-column classifier has one positive column, so the
    #: choice is a declaration rather than an inference. A multi-class model would
    #: have no positive column, and ``narrow_to_score_column`` refuses more than two
    #: outright rather than letting a trading rule arrive dressed as a conversion.
    narrow: bool = False
    #: Whether ``axon-model``'s own loader accepted this family's artifact, measured
    #: against `tract` :data:`TRACT_VERSION` and **declared here rather than derived**.
    #:
    #: A Python process cannot answer this. It is tempting to think it can — the
    #: refusal that matters here is an attribute check, and the attribute is right
    #: there in the graph — but the boosted tree's ``base_values`` is length **1** in
    #: the bytes while `tract` reports ``got 2``, so the number it compares is one it
    #: derives internally. A predicate reimplementing that derivation would agree
    #: today and become a guess wearing a check's clothes on the next `tract` bump.
    crosses: bool = False
    #: The evidence for :attr:`crosses`, verbatim: what the loader printed.
    rust_loader: str = ""

    @property
    def criterion(self) -> Criterion:
        """The bar this family's artifact is held to, from ADR-0021's table only.

        ``declared_for`` rather than ``required_for``: the first is what a bundle
        written here stamps into its manifest, the second is the ceiling a reader
        refuses to see exceeded. Reporting the ceiling in the zoo's transcript while
        the manifest carried two ULP would put a number in the table that no artifact
        was actually held to.
        """
        return Criterion.declared_for(self.kind)

    @property
    def eps(self) -> float:
        """The numeric tolerance for the Python-side model-parity gate."""
        return self.criterion.numeric_eps


def _xgboost() -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(**XGB_PARAMS)


def _sklearn_gbm() -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(**GBM_PARAMS)


def _logistic() -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**LOGIT_PARAMS)


#: The zoo. Ordered so the transcript reads from the family that is already proven to
#: the one that has never been tried, and the ``route`` strings name the operators
#: rather than the library — because the operator is what decides whether a graph
#: crosses, and "sklearn → ONNX → tract" reads like one route when it is three.
FAMILIES: tuple[Family, ...] = (
    Family(
        "xgboost",
        "xgboost",
        _xgboost,
        "native JSON → axon-model TreeModel (bit-exact)",
        crosses=True,
        rust_loader=(
            "cross-language model parity PASS: zoo_xgboost@1 (xgboost) n=512 "
            "criterion=bit-exact max_abs_diff=0e0 over_criterion=0 flips=0 non_finite=0"
        ),
    ),
    Family(
        "sklearn_gbm",
        "onnx",
        _sklearn_gbm,
        "skl2onnx → ai.onnx.ml TreeEnsembleClassifier → tract",
        narrow=True,
        crosses=False,
        rust_loader=(
            "CHECK FAILED: loading the bundle's artifact: parsing model artifact: "
            "Building node TreeEnsembleClassifier: attribute 'base_values': "
            "expected length 1 (or undefined), got 2"
        ),
    ),
    Family(
        "logistic",
        "onnx",
        _logistic,
        "skl2onnx → ai.onnx.ml LinearClassifier → tract",
        narrow=True,
        crosses=True,
        rust_loader=(
            "cross-language model parity PASS: zoo_logistic@1 (onnx) n=512 "
            "criterion=max_abs_diff <= 2.3841858e-7 max_abs_diff=5.9604645e-8 (row 310) "
            "over_criterion=0 flips=0 non_finite=0"
        ),
    ),
)


def family(name: str) -> Family:
    for f in FAMILIES:
        if f.name == name:
            return f
    raise TrainingError(f"unknown family {name!r}; the zoo holds {[f.name for f in FAMILIES]}")


# ── scores ───────────────────────────────────────────────────────────────────


def positive_class(scores: Any) -> np.ndarray:
    """``P(up)`` out of whatever shape a backend returned, as a flat float64 vector.

    A tree booster hands back a flat vector of ``P(class 1)``; an ONNX classifier
    graph hands back both columns, label-ordered. Taking column zero of the second
    would be ``P(down)`` — a strategy that trades the exact inverse of its model, with
    no symptom other than losing money. This is the same rule
    ``PerpBar._probability`` applies one row at a time, spelled once for a matrix so
    the gate and the strategy cannot disagree about which column is the signal.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 2 and s.shape[1] == 2:
        return s[:, 1]
    return np.ravel(s)


def probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """The in-memory model's ``P(up)`` per row, at the width serving uses.

    float32 in, because that is what the artifact was verified at and what the Rust
    side will feed it (ADR-0005). Predicting at float64 measures a model that never
    runs, and the difference is not academic: a threshold sitting between the two
    widths decides differently on each.
    """
    x = np.ascontiguousarray(features, dtype=np.float32)
    return positive_class(model.predict_proba(x))


def margin_versus_probability(artifact: Artifact, features: np.ndarray) -> float:
    """How far apart a tree bundle's reference and the strategy's score actually are.

    Not a gate — a measurement, kept because the *shape* of this mistake is invisible.
    ``rust_gate.python_scores`` records an XGBoost margin and the strategy acts on a
    probability, and a gate that compared the two would report a difference of several
    units and read as a catastrophically broken model rather than as a missing
    ``sigmoid``. ``test_zoo.py`` pins the number so the trap cannot quietly stop being
    one; :data:`~axon.parity.rust_gate.SCORE_SPACE` is the fix, not a tolerance.
    """
    from axon.parity.rust_gate import python_scores

    margin = np.asarray(python_scores(artifact, np.ascontiguousarray(features, np.float32)))
    probability = positive_class(load_predictor(artifact).predict(features.astype(np.float32)))
    return float(np.max(np.abs(margin.reshape(-1) - probability)))


# ── the walk-forward, over any family ────────────────────────────────────────


def fit_family(fam: Family, features: np.ndarray, labels: np.ndarray) -> Any:
    """Fit one family on one window. Returns the estimator, never a bare booster.

    The wrapper is what the export path takes its reference prediction from, and an
    early-stopped wrapper predicts with fewer trees than the booster underneath it
    (ADR-0015). Handing a booster around here would make that mismatch invisible.
    """
    model = fam.build()
    model.fit(np.ascontiguousarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32))
    return model


def walk_forward(
    fam: Family,
    dataset: Dataset,
    *,
    folds: int = 4,
    embargo_bars: int = 0,
    entry_edge: float = ZOO_PARAMS.entry_edge,
) -> WalkForward:
    """A purged, expanding walk-forward for one family.

    Deliberately the same shape as ``training.walk_forward_fit`` and deliberately not
    a call to it: that function hardcodes the XGBoost ``fit``, and varying the family
    is the entire point of this module. The pieces that *are* shared — the split, the
    per-fold scoring, the report types — are imported rather than restated, so the two
    can only disagree about the fitter. If a later change lets
    ``walk_forward_fit`` take a fitter, this function should be deleted rather than
    kept in sync.

    The exported model is the **last fold's**, not a refit on everything, for the
    reason ADR-0022 gives: a refit is a model no reported number describes.
    """
    bar_ns = INTERVAL_MS[dataset.interval] * 1_000_000
    splits = purged_walk_forward(
        dataset.ts_event,
        folds=folds,
        horizon_ns=dataset.horizon_ns,
        embargo_ns=embargo_bars * bar_ns,
    )
    results, model = [], None
    scores = np.full(len(dataset), np.nan)
    for split in splits:
        x_train, y_train = dataset.rows(split.train)
        model = fit_family(fam, x_train, y_train)
        fold_scores = probabilities(model, dataset.features[split.test])
        scores[split.test] = fold_scores
        results.append(
            score_fold(
                split,
                dataset.label[split.test],
                dataset.forward_return[split.test],
                fold_scores,
                entry_edge=entry_edge,
            )
        )

    scored = np.isfinite(scores)
    decision = np.zeros(len(dataset), dtype=np.int64)
    decision[scored & (scores >= 0.5 + entry_edge)] = 1
    decision[scored & (scores <= 0.5 - entry_edge)] = -1
    taken = decision != 0
    signed = decision[taken] * dataset.forward_return[taken]
    evaluation = Evaluation(
        folds=tuple(results),
        auc=_pooled_auc(dataset.label[scored], scores[scored]),
        coverage=float(taken.sum() / max(1, int(scored.sum()))),
        hit_rate=float(np.mean(signed > 0.0)) if signed.size else float("nan"),
        gross_edge_bps=float(np.mean(signed) * 10_000.0) if signed.size else float("nan"),
        n_test=int(scored.sum()),
        entry_edge=float(entry_edge),
    )
    return WalkForward(
        evaluation=evaluation, folds=splits, model=model, oos_scores=scores, dataset=dataset
    )


def _pooled_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    # Imported at call time and never surfaced by this module's own API. The ranking
    # statistic is real and it is not a result: ADR-0022 was written because an AUC
    # of 0.52 was once read as a working strategy, so it stays inside `Evaluation`
    # next to the edge decomposition that can contradict it.
    from axon.strategies.training import roc_auc

    return roc_auc(labels, scores)


# ── export and the gates ─────────────────────────────────────────────────────


def export_family(
    fam: Family,
    run: WalkForward,
    registry: ModelRegistry,
    *,
    registry_id: str,
    spec: FeatureSpec = BAR_M1_V1,
    sample_rows: int = 512,
) -> Artifact:
    """Register the last fold's model, verified on rows from the block that judged it.

    The sample is drawn from that fold's **test** block, not from training data: the
    round trip is then evidence that the artifact reproduces the model on the inputs
    the reported numbers came from, and a sample of memorized training rows would
    exercise a narrower part of the tree than serving ever will.

    The kind is checked against the family's declaration rather than trusted, because
    the interesting failure is silent: an sklearn estimator that stopped converting
    would export as *something*, and the first thing to notice would be the Rust gate
    refusing a kind nobody chose.
    """
    test = run.final_fold.test
    sample = run.dataset.features[test][-sample_rows:]
    meta = ArtifactMeta(
        registry_id=registry_id,
        version=registry.next_version(registry_id),
        feature_spec_ref=spec.ref,
    )
    artifact = export_artifact(run.model, meta, sample, narrow_score_output=fam.narrow)
    if artifact.meta.kind != fam.kind:
        raise TrainingError(
            f"{fam.name}: exported as kind {artifact.meta.kind!r}, the family declares "
            f"{fam.kind!r}; the route changed and the criterion no longer describes it"
        )
    registry.save(artifact)
    return artifact


def model_parity_gate(
    fam: Family,
    run: WalkForward,
    artifact: Artifact,
    *,
    entry_edge: float = ZOO_PARAMS.entry_edge,
    rows: int = 2_000,
) -> ModelParityReport:
    """The registered artifact against the in-memory model, on the test block.

    Both sides are ``P(up)``, through :func:`positive_class`, which is the whole
    reason this exists instead of ``training.model_parity_gate``: that one compares a
    flat reference against whatever the predictor returned, and an ONNX classifier
    returns two columns. The mismatch surfaces as a *shape* error, so it is loud
    rather than wrong — but it is loud about brackets, and the question is about
    numbers.

    ``eps`` comes from the family: exact for the tree, because the same library
    reading back its own file has no excuse for a difference, and **two ULP** for a
    graph, because ONNX does not encode operator ordering and float addition is not
    associative — but the graphs here disagree by at most one ULP, so the 1e-5 family
    ceiling would leave forty-twofold of slack for a regression to hide in
    (ADR-0021, amended 2026-07-26). The discretizer is the strategy's own entry band, so the
    decision-invariance half asks whether *this* artifact would have taken a different
    position rather than a generic question about thresholds.
    """
    sample = run.dataset.features[run.final_fold.test][-rows:]
    reference = probabilities(run.model, sample)
    candidate = positive_class(load_predictor(artifact).predict(sample.astype(np.float32)))
    return model_parity(
        reference,
        candidate,
        discretizer=threshold_discretizer(long_at=0.5 + entry_edge, short_at=0.5 - entry_edge),
        eps=fam.eps,
    )


def serving_strategy(
    registry: ModelRegistry,
    registry_id: str,
    *,
    version: int | None = None,
    spec: FeatureSpec = BAR_M1_V1,
    params: PerpBarParams = ZOO_PARAMS,
    buffer_bars: int = SERVING_BUFFER_BARS,
) -> PerpBar:
    """The serving path for a zoo model — ``PerpBar``, pointed at this spec.

    Not a new class. ``PerpBar`` is parameterized on its :class:`FeatureSpec` and on
    its :class:`~axon.models.inference.Predictor`, and it holds no reference to an
    hourly bar anywhere: it buffers closed bars, recomputes the spec over the buffer,
    reads ``P(up)`` and applies a hysteresis band. A second serving class for the zoo
    would be a second implementation of the exact comparison the feature-parity gate
    is trying to make, which is the bug ``docs/03`` names.
    """
    return PerpBar.from_registry(
        registry, registry_id, params, version=version, spec=spec, buffer_bars=buffer_bars
    )


def live_strategy(*, symbol_id: int, max_position: Decimal | None, registry, model: str):
    """The factory ``--strategy axon.strategies.zoo:live_strategy`` resolves to.

    The live runner and the shadow runner share one factory table
    (``shadow._strategy_factory``) and it takes ``module:callable`` for exactly this —
    a model zoo lands one family at a time and neither runner should need editing per
    family. This is the zoo's entry in that protocol, and it is four lines of glue
    around :func:`serving_strategy` rather than a second serving path: a live run that
    built its own object would be trading something the feature-parity gate never
    measured.

    **It refuses a run with no registry, and that refusal is the point.**
    ``_perp_bar_factory`` falls back to a constant 0.5 predictor when no artifact is
    named, which is right for a shadow run whose continuous diff never looks at a score
    — but a *live* run under that fallback sits inside the hysteresis band forever,
    takes no position, and is indistinguishable from a warmed-up strategy with no
    opinion. Nothing raises, no counter moves, and the session reads as healthy for as
    long as anyone leaves it up.

    ``max_position`` is the size the strategy will hold and the only thing here that
    decides an order's size; :data:`ZOO_PARAMS` carries 0.001 BTC, which is ~$65 at
    2026 prices and above the Phase-6 brief's $50 ceiling, so a live run states its own.
    """
    from axon.strategies.shadow import ShadowError

    if registry is None:
        raise ShadowError(
            "the zoo serves a fitted artifact and has no meaningful behaviour without "
            "one: pass --registry and --model. A no-artifact fallback here would take "
            "no position and read exactly like a strategy with no opinion"
        )
    params = replace(
        ZOO_PARAMS,
        symbol_id=symbol_id,
        **({} if max_position is None else {"max_position": max_position}),
    )
    return serving_strategy(registry, model, params=params)


def rust_bundle(
    artifact: Artifact,
    features: np.ndarray,
    *,
    out_dir: str | Path,
    rows: int = 512,
) -> ParityBundle:
    """Write the cross-language question for ``artifact`` and read it back.

    Returns the bundle as :func:`~axon.parity.read_parity_bundle` sees it, having
    re-checked every content hash, the criterion, and that the recorded decisions
    follow from the recorded scores. That is the whole Python half of ADR-0021 and it
    is *not* the gate: the gate is a Rust process loading these bytes with no Python
    in the room. Writing here proves the question is well-formed and that this family
    can be asked at all; whether the answer matches is a `cargo test`.

    **A written bundle is not a crossing, and the boosted tree is the proof.** Once its
    graph is narrowed to one score column this call succeeds for it — and `tract` still
    refuses the same bytes at load, on an attribute. So :attr:`Family.crosses` is a
    separate, declared, out-of-band measurement, and the transcript prints both.

    The thresholds come from :func:`~axon.parity.quantile_decision` because the zoo's
    own entry band lives on a probability and an XGBoost bundle records a margin — a
    fixed 0.52 threshold on a margin corpus makes every row decide the same way, and a
    decision check that can only come out one way is a decoration.
    """
    directory = write_parity_bundle(
        artifact,
        np.ascontiguousarray(features[-rows:], dtype=np.float32),
        out_dir=out_dir,
        decision=quantile_decision(),
        overwrite=True,
    )
    return read_parity_bundle(directory)


# ── one family, end to end ───────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class FamilyResult:
    """Everything one family proved, and everything it did not."""

    family: Family
    run: WalkForward
    artifact: Artifact
    model_parity: ModelParityReport
    feature_parity: Mapping[str, FeatureParityReport]
    drift: DriftReport
    #: ``None`` when the family has no Rust backend at all, and the error text when
    #: writing the bundle refused. A refusal is a **result** — the boundary said where
    #: it is — so it is carried rather than raised.
    bundle: ParityBundle | None = None
    bundle_error: str | None = None
    #: Target changes over the whole replay, per coin. Turnover is half of whether an
    #: edge survives, and the fee schedule is the other half.
    target_changes: Mapping[str, int] = field(default_factory=dict)
    replay_bars: Mapping[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """The two gates that are pass/fail.

        Drift is an alarm, not a verdict: a market that moved is not a bug, and
        treating it as one would make the deploy gate red for the one thing it cannot
        fix. The bundle is not in here either — it is the *Rust* gate's input, and
        this process cannot run the Rust gate.
        """
        return self.model_parity.passed and all(r.passed for r in self.feature_parity.values())

    @property
    def decomposition(self) -> Decomposition:
        return decompose(self.run)

    @property
    def turnover(self) -> int:
        return int(sum(self.target_changes.values()))

    def verdict(self) -> str:
        """Should this trade? In basis points against a benchmark that needs no model.

        The only honest summary shape this repo has: the model's edge over its own
        decision rows, the same rows priced for a constant short, the difference
        between them, and the fee drag implied by the turnover actually measured on
        the replay. An AUC does not appear, on purpose (ADR-0022).
        """
        from axon.strategies.training import MAKER_FEE_BPS, TAKER_FEE_BPS

        d = self.decomposition
        bars = sum(self.replay_bars.values())
        changes = self.turnover
        per_change = (
            f"a mind changed every {bars / changes:.1f} bars" if changes else "no target ever set"
        )
        # A target change is one round trip's worth of intent at most; pricing it as a
        # full round trip is the conservative reading and the one an operator needs.
        maker = 2 * MAKER_FEE_BPS * changes / max(1, d.n)
        taker = 2 * TAKER_FEE_BPS * changes / max(1, d.n)
        # The bar is the **cheapest** fee schedule, not zero. A selection of +0.3 bps
        # under a 1.0 bps maker drag is a positive number and a losing strategy, and
        # "selection is positive" is exactly the sentence that would get quoted out of
        # this transcript — so the only way to read past SHOULD NOT TRADE is to clear
        # the drag, before spread, slippage and funding.
        should = (
            "clears the maker fee drag"
            if d.selection_bps > maker
            else "SHOULD NOT TRADE"
        )
        return "\n".join(
            [
                f"  verdict [{self.family.name}]: {should}",
                f"    selection {d.selection_bps:+.2f}bps "
                f"(model {d.edge_bps:+.2f} vs always-short {d.benchmark_edge_bps:+.2f}, "
                f"same {d.n} rows)",
                f"    turnover {changes} target changes over {bars} bars, {per_change}",
                f"    fee drag {maker:.2f}bps/decision at maker, {taker:.2f}bps at taker",
            ]
        )

    def gate_row(self) -> str:
        """One line of the table this whole module exists to produce."""
        fp = " ".join(
            f"{coin}:{'PASS' if r.passed else 'FAIL'}" for coin, r in self.feature_parity.items()
        )
        # Two columns, not one, and the split is the whole lesson of this workstream.
        # `bundle` is what *this* process measured: the question is well-formed. `rust`
        # is whether the loader on the other side of the boundary accepted it, which no
        # Python process can answer — and the boosted tree is exactly the family where
        # the two disagree. Collapsing them would print `rust=bundle ok` for a model
        # `tract` refuses at load, which is the silent green this repo keeps finding.
        bundle = (
            f"ok ({self.bundle.criterion})" if self.bundle is not None
            else f"REFUSED: {self.bundle_error}"
        )
        crosses = "YES" if self.family.crosses else "NO"
        return (
            f"  {self.family.name:<12} kind={self.artifact.meta.kind:<8} "
            f"model={'PASS' if self.model_parity.passed else 'FAIL'} "
            f"(max_abs_diff={self.model_parity.max_abs_diff:.3e}, eps={self.family.eps:.0e}) "
            f"features={fp} drift={'OK' if self.drift.passed else 'ALARM'} "
            f"bundle={bundle} rust={crosses}"
        )

    def summary(self) -> str:
        lines = [
            f"── {self.family.name} ── {self.family.route}",
            f"artifact: {self.artifact.ref} kind={self.artifact.meta.kind} "
            f"({self.artifact.meta.content_bytes} bytes, "
            f"roundtrip max_abs_diff={self.artifact.meta.roundtrip_max_abs_diff:g})",
            self.run.evaluation.summary(),
            self.decomposition.summary(),
            self.model_parity.summary(),
        ]
        lines += [f"{coin}: {r.summary()}" for coin, r in self.feature_parity.items()]
        lines.append(self.drift.summary())
        if self.bundle is not None:
            counts = self.bundle.manifest["decisions"]["counts"]
            lines.append(
                f"rust bundle: {self.bundle.kind} score_space="
                f"{self.bundle.manifest['score_space']} criterion={self.bundle.criterion} "
                f"rows={self.bundle.features.shape[0]} decisions={counts}"
            )
        else:
            lines.append(f"rust bundle: REFUSED — {self.bundle_error}")
        lines.append(
            f"rust loader (tract {TRACT_VERSION}, measured out of band): "
            f"{'crosses' if self.family.crosses else 'DOES NOT CROSS'} — "
            f"{self.family.rust_loader}"
        )
        lines.append(self.verdict())
        return "\n".join(lines)


def run_family(
    fam: Family,
    candles: Sequence[Candles],
    registry: ModelRegistry,
    *,
    spec: FeatureSpec = BAR_M1_V1,
    registry_id: str | None = None,
    folds: int = 4,
    embargo_bars: int = 0,
    params: PerpBarParams = ZOO_PARAMS,
    buffer_bars: int = SERVING_BUFFER_BARS,
    bundle_dir: str | Path | None = None,
    parity_bars: int | None = None,
) -> FamilyResult:
    """Fit, export, register and gate one family — the whole rung, once.

    Returns the reports rather than asserting on them. A gate that fails is a
    *result* (ADR-0016), and a pipeline that raised on one would be a pipeline whose
    failures never got looked at properly. That applies to the bundle too: a family
    with no Rust backend comes back with ``bundle_error`` set and everything else
    filled in, because "which families cannot cross, and why" is the deliverable.
    """
    dataset = build_dataset(candles, spec=spec, horizon=LABEL_HORIZON_BARS)
    run = walk_forward(
        fam, dataset, folds=folds, embargo_bars=embargo_bars, entry_edge=params.entry_edge
    )
    rid = registry_id or f"zoo_{fam.name}"
    artifact = export_family(fam, run, registry, registry_id=rid, spec=spec)

    served = serving_strategy(
        registry, rid, version=artifact.meta.version, spec=spec, params=params,
        buffer_bars=buffer_bars,
    )
    parity: dict[str, FeatureParityReport] = {}
    changes: dict[str, int] = {}
    bars: dict[str, int] = {}
    for c in candles:
        parity[c.coin] = feature_parity_gate(
            served, c, symbol_id=params.symbol_id, spec=spec, replay_bars_from=parity_bars
        )
        changes[c.coin], bars[c.coin] = _replay_turnover(served, c, params)

    bundle, bundle_error = None, None
    if bundle_dir is not None:
        try:
            bundle = rust_bundle(
                artifact, run.dataset.features[run.final_fold.test], out_dir=bundle_dir
            )
        except Exception as exc:  # noqa: BLE001 — the refusal text *is* the finding
            bundle_error = f"{type(exc).__name__}: {exc}"

    return FamilyResult(
        family=fam,
        run=run,
        artifact=artifact,
        model_parity=model_parity_gate(fam, run, artifact, entry_edge=params.entry_edge),
        feature_parity=parity,
        drift=drift_gate(dataset, run.final_fold),
        bundle=bundle,
        bundle_error=bundle_error,
        target_changes=changes,
        replay_bars=bars,
    )


def _replay_turnover(
    strategy: PerpBar, candles: Candles, params: PerpBarParams
) -> tuple[int, int]:
    """How many times the served strategy changed its mind over one history.

    Counted from the signals the serving path actually emitted, not from a
    re-thresholding of the offline scores: the hysteresis band means the target
    depends on the *previous* target, so a vectorized recompute of "which side would
    this row be on" is a different, larger number. ``perp_bar``'s measured 1 446
    changes over 4 999 bars came out this way, and the fee arithmetic that followed
    is only meaningful because of it.
    """
    from axon.strategies.training import replay_bars as _drive

    _, _, signals = _drive(strategy, candles, symbol_id=params.symbol_id)
    return len(signals), len(candles)


# ── the whole zoo ────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class ZooResult:
    """Every family, and the table."""

    spec: FeatureSpec
    results: tuple[FamilyResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def table(self) -> str:
        return "\n".join(
            [
                f"gate table — spec {self.spec.ref}, warmup {BAR_M1_WARMUP_BARS} bars "
                f"({BAR_M1_WARMUP_BARS} minutes on {INTERVAL})",
                *(r.gate_row() for r in self.results),
            ]
        )

    def summary(self) -> str:
        return "\n\n".join([*(r.summary() for r in self.results), self.table()])


def run_zoo(
    candles: Sequence[Candles],
    registry: ModelRegistry,
    *,
    families: Sequence[Family] = FAMILIES,
    spec: FeatureSpec = BAR_M1_V1,
    bundle_root: str | Path | None = None,
    **kwargs: Any,
) -> ZooResult:
    """Every family over the same candles, the same spec and the same parameters."""
    results = []
    for fam in families:
        directory = None if bundle_root is None else Path(bundle_root) / fam.name
        results.append(run_family(fam, candles, registry, spec=spec, bundle_dir=directory, **kwargs))
    return ZooResult(spec=spec, results=tuple(results))


__all__ = [
    "BAR_M1_V1",
    "FAMILIES",
    "GBM_PARAMS",
    "INTERVAL",
    "LABEL_HORIZON_BARS",
    "LOGIT_PARAMS",
    "SERVING_BUFFER_BARS",
    "TRACT_VERSION",
    "XGB_PARAMS",
    "ZOO_PARAMS",
    "Family",
    "FamilyResult",
    "ZooResult",
    "export_family",
    "family",
    "fit_family",
    "live_strategy",
    "margin_versus_probability",
    "model_parity_gate",
    "positive_class",
    "probabilities",
    "run_family",
    "run_zoo",
    "rust_bundle",
    "serving_strategy",
    "walk_forward",
]
