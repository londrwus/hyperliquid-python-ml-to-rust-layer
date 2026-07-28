"""The pipeline: candles → features → labels → walk-forward → artifact → gates.

This module is the ladder from ``docs/07`` for one strategy, written down so it can
be re-run rather than described. Everything in it is offline, deterministic and
importable without a network; the only heavy dependency is XGBoost, and it is
imported inside the function that needs it so the rest of the package stays usable
in a bare numpy environment.

Two choices here decide what the reported numbers are allowed to mean:

**The exported artifact is the last fold's model, not a refit on everything.** The
tidy thing to do after a walk-forward is to retrain on the full sample and ship
that — and then no number anyone reports describes the model that ships. The model
this pipeline registers is the one whose final test block was genuinely out of
sample for it. The cost is real and stated in ADR-0022: the shipped model has never
seen the most recent block.

**Costs are reported as a hurdle, not netted off.** Turning the fold metrics into a
P&L needs a turnover model, and a turnover model that lives here would be a second
implementation of the strategy's own hysteresis. So the evaluation reports the
gross edge per decision and the fee schedule beside it, and leaves the subtraction
to a reader who can see both numbers. A "net" figure computed from an invented
turnover assumption is worse than no figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from axon.features import FeatureSpec, finite_rows
from axon.models import ArtifactMeta, ModelRegistry, export_artifact
from axon.models.artifact import Artifact
from axon.parity import (
    TREE_EPS,
    Binning,
    DriftReport,
    FeatureParityReport,
    ModelParityReport,
    aligned_feature_parity,
    drift_report,
    model_parity,
    quantile_binning,
    threshold_discretizer,
)
from axon.strategies.data import INTERVAL_MS, Candles
from axon.strategies.labels import Fold, direction_label, forward_log_return, purged_walk_forward
from axon.strategies.perp_bar import (
    LABEL_HORIZON_BARS,
    PERP_BAR_V1,
    SERVING_BUFFER_BARS,
    PerpBar,
    PerpBarParams,
)
from axon.strategy.context import StrategyContext

#: The model. Small on purpose: nine features, a few thousand rows, and a signal
#: that is weak if it exists at all. Depth 3 with a large ``min_child_weight`` is
#: the shape that cannot memorize a fold — a deeper tree on this sample fits the
#: 2026 first quarter and reports it as skill.
#:
#: ``n_jobs=1`` is not a performance setting. Histogram building is parallelized
#: with a non-deterministic reduction order, so a multi-threaded fit produces a
#: slightly different model on every run and no artifact hash is reproducible.
XGB_PARAMS: Mapping[str, Any] = {
    "objective": "binary:logistic",
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 40,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 7,
    "eval_metric": "logloss",
}

#: Hyperliquid's published perp fee schedule at the base tier, in basis points of
#: notional, per side. Recorded as an assumption rather than a measurement: nothing
#: in this repo has yet been filled, so the only honest use of these numbers is as
#: the hurdle a gross edge has to clear, twice, to be worth trading.
TAKER_FEE_BPS = 4.5
MAKER_FEE_BPS = 1.5


class TrainingError(RuntimeError):
    """The pipeline refuses to produce a number that would not mean anything."""


# ── dataset ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class Dataset:
    """A pooled, point-in-time-correct training matrix over one or more coins.

    Coins are pooled into one model rather than fitted per symbol, which is only
    legitimate because every column in the spec is scale-free: a log return, a
    z-score, a ratio of means, a location within a range. A model fed raw prices
    would learn that BTC is a bigger number than ETH.

    Rows carry their coin so a per-symbol breakdown is still available, and the
    walk-forward split is defined on ``ts_event`` for exactly this reason — with
    two symbols interleaved, a row-index split holds one coin's future next to the
    other's past.
    """

    ts_event: np.ndarray
    features: np.ndarray
    label: np.ndarray
    forward_return: np.ndarray
    coin: np.ndarray
    columns: tuple[str, ...]
    spec_ref: str
    horizon: int
    interval: str

    def __len__(self) -> int:
        return int(self.ts_event.size)

    def __repr__(self) -> str:
        coins = "+".join(dict.fromkeys(self.coin.tolist()))
        return (
            f"Dataset({coins} {self.interval}, {len(self)} rows × "
            f"{len(self.columns)} features, horizon={self.horizon})"
        )

    @property
    def horizon_ns(self) -> int:
        return self.horizon * INTERVAL_MS[self.interval] * 1_000_000

    @property
    def base_rate(self) -> float:
        """Fraction of rows whose forward return was positive.

        Worth printing next to every accuracy: a sample that rose 54% of the time
        makes "54% accurate" the performance of a constant.
        """
        return float(np.mean(self.label))

    def rows(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.features[index], self.label[index]

    def select(self, index) -> "Dataset":
        """The same dataset restricted to ``index``, keeping its spec identity.

        The one use that matters: handing a hyper-parameter search *only* the rows
        a fold was allowed to train on, so the search cannot select on the holdout
        it will later be judged against.
        """
        idx = np.asarray(index)
        return Dataset(
            ts_event=self.ts_event[idx],
            features=self.features[idx],
            label=self.label[idx],
            forward_return=self.forward_return[idx],
            coin=self.coin[idx],
            columns=self.columns,
            spec_ref=self.spec_ref,
            horizon=self.horizon,
            interval=self.interval,
        )


def build_dataset(
    candles: Sequence[Candles],
    *,
    spec: FeatureSpec = PERP_BAR_V1,
    horizon: int = LABEL_HORIZON_BARS,
) -> Dataset:
    """Features and labels for a set of coins, keeping only usable rows.

    A row survives when every feature is finite *and* its forward return exists.
    Both halves matter: dropping the NaN warmup by hand from a window length is the
    off-by-one ``docs/03`` lists among the silent killers, and keeping the final
    ``horizon`` rows — whose outcome has not happened — would put fabricated labels
    in the most recent test block of every split.
    """
    if not candles:
        raise TrainingError("a dataset needs at least one coin's candles")
    intervals = {c.interval for c in candles}
    if len(intervals) != 1:
        # Two intervals pooled means a "4-bar horizon" is two different durations
        # and one label column describes two questions.
        raise TrainingError(f"cannot pool candles of different intervals: {sorted(intervals)}")

    ts_parts, x_parts, y_parts, r_parts, coin_parts = [], [], [], [], []
    for c in candles:
        inputs = c.feature_inputs()
        matrix = spec.compute(inputs)
        forward = forward_log_return(inputs["close"], horizon=horizon)
        usable = finite_rows(matrix) & np.isfinite(forward)
        if not usable.any():
            raise TrainingError(
                f"{c.coin}: no usable rows out of {len(c)} bars — the history is shorter "
                "than the spec's warmup plus the label horizon"
            )
        ts_parts.append(c.ts_event[usable])
        x_parts.append(matrix[usable])
        r_parts.append(forward[usable])
        y_parts.append(direction_label(forward[usable]))
        coin_parts.append(np.full(int(usable.sum()), c.coin, dtype=object))

    ts = np.concatenate(ts_parts)
    # Stable sort on event time: rows from different coins sharing a bar close stay
    # in coin order, so the dataset is a deterministic function of its inputs and
    # two runs produce byte-identical models.
    order = np.argsort(ts, kind="stable")
    return Dataset(
        ts_event=ts[order],
        features=np.concatenate(x_parts)[order],
        label=np.concatenate(y_parts)[order],
        forward_return=np.concatenate(r_parts)[order],
        coin=np.concatenate(coin_parts)[order],
        columns=spec.columns,
        spec_ref=spec.ref,
        horizon=int(horizon),
        interval=next(iter(intervals)),
    )


# ── metrics ──────────────────────────────────────────────────────────────────


def roc_auc(labels, scores) -> float:
    """Area under the ROC curve, by ranks, with ties averaged.

    Written out rather than imported so this module needs only numpy and XGBoost:
    the rank form *is* the definition (the probability that a randomly chosen up
    bar scores above a randomly chosen down bar), and a tie-blind implementation
    reports 0.5 as 1.0 for a model that outputs one constant.
    """
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise TrainingError(f"labels {y.shape} and scores {s.shape} describe different rows")
    positives = y > 0.5
    n_pos = int(positives.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        # One class only: every ranking is equally right, and 0.5 would read as a
        # measured coin flip rather than as "this fold cannot be scored".
        return float("nan")
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    average_rank = (starts + (counts - 1) / 2.0 + 1.0)[inverse]
    return float((average_rank[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass(frozen=True)
class FoldResult:
    """One walk-forward fold, scored the way a trader would ask about it."""

    index: int
    n_train: int
    n_test: int
    base_rate: float
    auc: float
    #: Fraction of test rows on which the discretized decision is not flat.
    coverage: float
    #: Fraction of non-flat decisions whose direction was right.
    hit_rate: float
    #: Mean signed forward return over the rows a position was taken on, in basis
    #: points, before any cost. The number the fee schedule has to be held against.
    gross_edge_bps: float
    test_start_ns: int
    test_end_ns: int

    def line(self) -> str:
        return (
            f"  fold {self.index}: train={self.n_train:>5} test={self.n_test:>5} "
            f"base={self.base_rate:.3f} auc={self.auc:.4f} "
            f"coverage={self.coverage:.3f} hit={self.hit_rate:.3f} "
            f"edge={self.gross_edge_bps:+.2f}bps"
        )


@dataclass(frozen=True)
class Evaluation:
    """Every fold, plus the pooled out-of-sample numbers."""

    folds: tuple[FoldResult, ...]
    auc: float
    coverage: float
    hit_rate: float
    gross_edge_bps: float
    n_test: int
    entry_edge: float

    def summary(self) -> str:
        head = (
            f"walk-forward: {len(self.folds)} folds, {self.n_test} out-of-sample rows\n"
            f"  pooled: auc={self.auc:.4f} coverage={self.coverage:.3f} "
            f"hit={self.hit_rate:.3f} gross_edge={self.gross_edge_bps:+.2f}bps/decision "
            f"(entry_edge={self.entry_edge})"
        )
        hurdle = (
            f"  hurdle: a round trip costs {2 * MAKER_FEE_BPS:.1f}bps at maker fees, "
            f"{2 * TAKER_FEE_BPS:.1f}bps at taker — before spread, slippage and funding"
        )
        return "\n".join([head, *(f.line() for f in self.folds), hurdle])


def score_fold(
    fold: Fold,
    labels: np.ndarray,
    forward: np.ndarray,
    scores: np.ndarray,
    *,
    entry_edge: float,
) -> FoldResult:
    """Score one fold's out-of-sample predictions the way a trader would ask.

    ``entry_edge`` is the strategy's own entry band, not a metric-specific
    threshold: a hit rate measured at a threshold nothing trades at describes a
    strategy nobody is running.
    """
    decision = np.zeros(scores.size, dtype=np.int64)
    decision[scores >= 0.5 + entry_edge] = 1
    decision[scores <= 0.5 - entry_edge] = -1
    taken = decision != 0
    signed = decision[taken] * forward[taken]
    return FoldResult(
        index=fold.index,
        n_train=int(fold.train.size),
        n_test=int(fold.test.size),
        base_rate=float(np.mean(labels)),
        auc=roc_auc(labels, scores),
        coverage=float(np.mean(taken)),
        hit_rate=float(np.mean(signed > 0.0)) if signed.size else float("nan"),
        gross_edge_bps=float(np.mean(signed) * 10_000.0) if signed.size else float("nan"),
        test_start_ns=fold.test_start_ns,
        test_end_ns=fold.test_end_ns,
    )



# ── fitting ──────────────────────────────────────────────────────────────────


def fit(features: np.ndarray, labels: np.ndarray, *, params: Mapping[str, Any] | None = None):
    """Fit one XGBoost classifier. Returns the *wrapper*, never the bare booster.

    The wrapper is what the export path takes its reference prediction from, and an
    early-stopped wrapper predicts with fewer trees than the booster underneath it
    (ADR-0015). Handing a booster around here would make that mismatch impossible to
    see later.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(**{**XGB_PARAMS, **(params or {})})
    model.fit(np.ascontiguousarray(features, dtype=np.float32), labels.astype(np.int32))
    return model


def _probabilities(model, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(np.ascontiguousarray(features, dtype=np.float32)))[:, 1]


def fit_fold(
    dataset: Dataset,
    fold: Fold,
    *,
    entry_edge: float = PerpBarParams.entry_edge,
    params: Mapping[str, Any] | None = None,
) -> tuple[Any, np.ndarray, FoldResult]:
    """Fit one window and score it — the unit of work a walk-forward fans out.

    Returns ``(model, test_scores, result)``. Kept separate from
    :func:`walk_forward_fit` because a remote fan-out gives each window its own
    task, and a task that re-fitted every window to report one of them would turn a
    parallel job back into a serial one wearing a parallel job's cost.
    """
    x_train, y_train = dataset.rows(fold.train)
    model = fit(x_train, y_train, params=params)
    scores = _probabilities(model, dataset.features[fold.test])
    result = score_fold(
        fold,
        dataset.label[fold.test],
        dataset.forward_return[fold.test],
        scores,
        entry_edge=entry_edge,
    )
    return model, scores, result


@dataclass(frozen=True, eq=False)
class WalkForward:
    """The result of a walk-forward run: the evidence, and the model it describes."""

    evaluation: Evaluation
    folds: tuple[Fold, ...]
    #: The model fitted for the *last* fold — the one this pipeline exports, whose
    #: final test block it never saw.
    model: Any
    #: Out-of-sample probability for every scored row, by dataset row index. NaN
    #: where a row was in no test block (the initial training span).
    oos_scores: np.ndarray
    dataset: Dataset = field(repr=False)

    @property
    def final_fold(self) -> Fold:
        return self.folds[-1]


def walk_forward_fit(
    dataset: Dataset,
    *,
    folds: int = 4,
    embargo_bars: int = 0,
    entry_edge: float = PerpBarParams.entry_edge,
    params: Mapping[str, Any] | None = None,
) -> WalkForward:
    """Fit and score the strategy's model on a purged, expanding walk-forward.

    ``embargo_bars`` is on top of the horizon purge. The purge removes the training
    rows whose *labels* reach into the test window; the embargo removes rows whose
    *features* still overlap it, which is a weaker but real dependence — the 24-bar
    windows either side of the cut share up to 23 bars.
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
        model, fold_scores, result = fit_fold(
            dataset, split, entry_edge=entry_edge, params=params
        )
        scores[split.test] = fold_scores
        results.append(result)

    scored = np.isfinite(scores)
    decision = np.zeros(len(dataset), dtype=np.int64)
    decision[scored & (scores >= 0.5 + entry_edge)] = 1
    decision[scored & (scores <= 0.5 - entry_edge)] = -1
    taken = decision != 0
    signed = decision[taken] * dataset.forward_return[taken]
    evaluation = Evaluation(
        folds=tuple(results),
        auc=roc_auc(dataset.label[scored], scores[scored]),
        coverage=float(taken.sum() / max(1, int(scored.sum()))),
        hit_rate=float(np.mean(signed > 0.0)) if signed.size else float("nan"),
        gross_edge_bps=float(np.mean(signed) * 10_000.0) if signed.size else float("nan"),
        n_test=int(scored.sum()),
        entry_edge=float(entry_edge),
    )
    return WalkForward(
        evaluation=evaluation,
        folds=splits,
        model=model,
        oos_scores=scores,
        dataset=dataset,
    )


@dataclass(frozen=True)
class Decomposition:
    """Where a gross edge came from: the model's two sides, and the free benchmark.

    A weak model in a trending sample earns its edge from the trend, and the only way
    to see that is to price the trend on **the same rows**. Hence one rule, which the
    field names are shaped around: every number here is measured over the rows the
    model actually took a position on.

    The comparison this exists to prevent is the plausible one — the model's short
    side against a constant short *on the rows it was short*. Those are the same
    position on the same rows, so they are equal by construction, always, for every
    model; subtracting one from the other and calling the difference selection
    credits a directional bias with skill it does not have. ADR-0022 published that
    subtraction once, and this type is why it cannot be published again by accident.
    """

    #: Rows a position was taken on. Every bps figure below is a mean over these.
    n: int
    n_long: int
    n_short: int
    hit_rate: float
    #: The model: mean signed forward return over its own decision rows.
    edge_bps: float
    #: Mean forward return on the rows it was long — its long side, signed as held.
    long_edge_bps: float
    #: Mean *negated* forward return on the rows it was short — likewise as held.
    short_edge_bps: float
    #: Mean forward return over the decision rows: the drift the model was swimming in.
    drift_bps: float
    #: The same over every scored row. Reported separately because it is a different
    #: number (the model does not take a position on every row it scores), and quoting
    #: one where the other belongs is how a benchmark ends up measured on the wrong set.
    scored_drift_bps: float
    n_scored: int

    @property
    def benchmark_edge_bps(self) -> float:
        """A constant short held on exactly the model's decision rows.

        Identically ``-drift_bps``: a benchmark that needs no model, no fit and no
        features, and the number the edge has to beat before anything else is worth
        discussing. A constant *long* is its negation, so one number covers both.
        """
        return -self.drift_bps

    @property
    def selection_bps(self) -> float:
        """What the model's choices added over the free benchmark. Signed, often not."""
        return self.edge_bps - self.benchmark_edge_bps

    def summary(self) -> str:
        return "\n".join(
            [
                f"  decomposition over the {self.n} rows a position was taken on "
                f"(of {self.n_scored} scored):",
                f"    long   n={self.n_long:>5}  edge={self.long_edge_bps:+.2f}bps",
                f"    short  n={self.n_short:>5}  edge={self.short_edge_bps:+.2f}bps",
                f"    model  n={self.n:>5}  edge={self.edge_bps:+.2f}bps "
                f"hit={self.hit_rate:.3f}",
                f"    always short, same rows: {self.benchmark_edge_bps:+.2f}bps "
                f"(drift {self.drift_bps:+.2f}bps here, {self.scored_drift_bps:+.2f}bps "
                "over every scored row)",
                f"    selection = model - benchmark = {self.selection_bps:+.2f}bps",
            ]
        )


def decompose(run: WalkForward, *, entry_edge: float | None = None) -> Decomposition:
    """Split a walk-forward's gross edge into its sides and price the free benchmark.

    ``entry_edge`` defaults to the band the run was scored at, so the decomposition
    describes the decisions that were reported rather than a different set of them.
    """
    edge = run.evaluation.entry_edge if entry_edge is None else float(entry_edge)
    dataset = run.dataset
    scores = run.oos_scores
    scored = np.isfinite(scores)
    decision = np.zeros(scores.size, dtype=np.int64)
    decision[scored & (scores >= 0.5 + edge)] = 1
    decision[scored & (scores <= 0.5 - edge)] = -1
    taken = decision != 0
    forward = dataset.forward_return
    signed = decision[taken] * forward[taken]
    longs = decision == 1
    shorts = decision == -1
    bps = 10_000.0

    def mean_bps(values: np.ndarray) -> float:
        return float(np.mean(values) * bps) if values.size else float("nan")

    return Decomposition(
        n=int(taken.sum()),
        n_long=int(longs.sum()),
        n_short=int(shorts.sum()),
        hit_rate=float(np.mean(signed > 0.0)) if signed.size else float("nan"),
        edge_bps=mean_bps(signed),
        long_edge_bps=mean_bps(forward[longs]),
        short_edge_bps=mean_bps(-forward[shorts]),
        drift_bps=mean_bps(forward[taken]),
        scored_drift_bps=mean_bps(forward[scored]),
        n_scored=int(scored.sum()),
    )


# ── export ───────────────────────────────────────────────────────────────────


def export_walk_forward(
    run: WalkForward,
    registry: ModelRegistry,
    *,
    registry_id: str,
    spec: FeatureSpec = PERP_BAR_V1,
    sample_rows: int = 512,
) -> Artifact:
    """Register the last fold's model, verified against the rows it was judged on.

    ``sample_input`` is drawn from that fold's **test** block rather than from
    training data: the round trip is evidence that the artifact reproduces the model
    on the inputs the reported numbers came from, and a sample of memorized training
    rows would exercise a narrower part of the tree than serving ever will.
    """
    test = run.final_fold.test
    sample = run.dataset.features[test][-sample_rows:]
    meta = ArtifactMeta(
        registry_id=registry_id,
        version=registry.next_version(registry_id),
        feature_spec_ref=spec.ref,
    )
    artifact = export_artifact(run.model, meta, sample)
    registry.save(artifact)
    return artifact


# ── the three gates ──────────────────────────────────────────────────────────


def replay_bars(
    strategy: PerpBar,
    candles: Candles,
    *,
    symbol_id: int,
    model_version: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Drive a strategy over a candle history exactly as the live runner would.

    Returns ``(ts_event, online_features, signals)``: one feature row per bar the
    strategy actually formed an opinion on, stamped with that bar's close time, plus
    every signal it emitted.

    This is *not* the golden replay of ADR-0018 — that republishes a captured event
    log through the real Rust core, and a candle history is not an event log. What
    it is: the Python serving path, driven over a recording, which is what the
    feature-parity gate needs both sides of.
    """
    from axon.strategy.events import Bar

    ctx = StrategyContext(model_version=model_version)
    times, rows, signals = [], [], []
    strategy.on_reset()
    for i in range(len(candles)):
        bar = Bar(
            symbol_id=symbol_id,
            ts_event=int(candles.ts_event[i]),
            open=int(candles.open[i]),
            high=int(candles.high[i]),
            low=int(candles.low[i]),
            close=int(candles.close[i]),
            volume=int(candles.volume[i]),
        )
        with ctx.event(bar.ts_event):
            strategy.on_bar(bar, ctx)
            row = strategy.feature_row()
        if row is not None:
            times.append(bar.ts_event)
            rows.append(row)
        signals.extend(ctx.take_pending())
    if not rows:
        raise TrainingError(
            f"{candles.coin}: the strategy formed no usable feature row over {len(candles)} "
            "bars; the history is shorter than the spec's warmup"
        )
    return np.array(times, dtype=np.int64), np.vstack(rows), signals


def owed_rows(
    candles: Candles, replayed: Candles, *, spec: FeatureSpec = PERP_BAR_V1
) -> np.ndarray:
    """Which rows of ``candles`` a serving path replayed over ``replayed`` owes a row for.

    A boolean mask over the full history, true exactly where a cold-started replay of
    ``replayed`` can produce a finite feature row. It is *not* ``finite_rows`` of the
    full-history recompute: a cold start legitimately produces nothing through its own
    warmup while the full recompute has real numbers there, and holding the replay to
    those would fire on every healthy run with ``replay_bars_from`` set — the reading
    that gets a guard deleted rather than fixed.

    Note what does **not** need checking: every span-finite row is also full-history
    finite, so this mask is always a subset of ``finite_rows(spec.compute(candles))``.
    That containment is a consequence of the finite-lookback rule — a window that fits
    inside the replayed span is the same arithmetic on the same numbers either way. An
    expanding transform would break it, and ADR-0022 §2 is why the spec has none.
    """
    mask = np.zeros(len(candles), dtype=bool)
    mask[len(candles) - len(replayed) :] = finite_rows(spec.compute(replayed.feature_inputs()))
    return mask


def feature_parity_gate(
    strategy: PerpBar,
    candles: Candles,
    *,
    symbol_id: int = 0,
    spec: FeatureSpec = PERP_BAR_V1,
    replay_bars_from: int | None = None,
) -> FeatureParityReport:
    """Online feature vectors against the offline recompute, aligned on event time.

    The gate that ``docs/03`` calls the hard one. It is only meaningful because the
    online side is genuinely the serving path — a bounded buffer, one bar at a time,
    through ``on_bar`` — while the offline side is one vectorized pass over the whole
    history. Every finite-lookback transform makes those two identical; an expanding
    one would not, which is why the spec has none.

    ``replay_bars_from`` starts the online side that many bars from the end, while
    the offline side still recomputes everything. That is a *stronger* comparison
    than it looks and it costs a fraction of the time: it is a strategy started cold
    in the middle of a history, matched against a recompute that saw all of it, and
    it lines up only because the alignment is on ``ts_event`` rather than on row
    position. A cold start that quietly produced different numbers is the failure
    this shape catches; the price is that the earliest rows go unchecked.

    **The denominator is part of the verdict, and there is exactly one of it.** The
    alignment is an intersection, so a serving path that produced no row at all on a
    bar is not a mismatch to it — it is simply absent, and the remaining rows still
    agree to the last bit. :class:`~axon.parity.Coverage` is what closes that, and
    :func:`~axon.parity.aligned_feature_parity` is the call that cannot be made
    without answering it. This gate used to carry a second, private row count for the
    same question (``ServedFeatureParityReport``); two implementations of "how many
    rows were owed" is the shape this codebase treats as a bug, so the reference
    handed to the aligner is trimmed to :func:`owed_rows` instead and ``Coverage``
    counts it. The values compared are still the *full-history* recompute's — that is
    the whole point of the cold-start shape.

    ``scope="declared"`` is what makes the trim above load-bearing rather than
    cosmetic. ``Coverage``'s default is to infer the owed span from the online side's
    own first stamp, which is right for a live monitor whose window opens mid-history
    — and wrong here, because a serving path blind through its opening rows has
    exactly the same first stamp as one that started on time, and would be excused as
    a late join. The reference handed over *is* the owed set, so saying so turns a
    late start back into the fault it is (ADR-0030 §1a).
    """
    replayed = candles if replay_bars_from is None else candles[-int(replay_bars_from) :]
    online_ts, online, _ = replay_bars(strategy, replayed, symbol_id=symbol_id)
    owed = owed_rows(candles, replayed, spec=spec)
    offline = spec.compute(candles.feature_inputs())
    return aligned_feature_parity(
        online,
        offline[owed],
        online_ts=online_ts,
        offline_ts=candles.ts_event[owed],
        columns=spec.columns,
        scope="declared",
    )


def model_parity_gate(
    run: WalkForward,
    artifact: Artifact,
    *,
    entry_edge: float,
    rows: int = 2_000,
) -> ModelParityReport:
    """The registered artifact against the in-memory model, on the test block.

    Exact (:data:`~axon.parity.TREE_EPS`) because it is a tree ensemble: inference is
    deterministic threshold traversal, so any difference at all means a float width
    changed somewhere between the fit and the file. The discretizer is the
    strategy's own entry mapping, so the decision-invariance half of the gate is
    asking the question that matters — would this artifact have taken a different
    position — rather than a generic one.
    """
    from axon.models.inference import load_predictor

    sample = run.dataset.features[run.final_fold.test][-rows:]
    reference = _probabilities(run.model, sample)
    candidate = np.asarray(load_predictor(artifact).predict(np.asarray(sample, dtype=np.float32)))
    return model_parity(
        reference,
        candidate,
        discretizer=threshold_discretizer(long_at=0.5 + entry_edge, short_at=0.5 - entry_edge),
        eps=TREE_EPS,
    )


def training_binnings(dataset: Dataset, fold: Fold, *, bins: int = 10) -> tuple[Binning, ...]:
    """Quantile bins frozen on the fold's training rows.

    Frozen, because recomputing them from each live window makes both histograms
    uniform by construction and PSI reads ~0 whatever happened (ADR-0016). These are
    what a live drift monitor would carry forward next to the artifact.
    """
    train = dataset.features[fold.train]
    return tuple(quantile_binning(train[:, j], bins=bins) for j in range(train.shape[1]))


def drift_gate(dataset: Dataset, fold: Fold, *, bins: int = 10) -> DriftReport:
    """How far the final test block moved from what the shipped model was fitted on."""
    return drift_report(
        dataset.features[fold.train],
        dataset.features[fold.test],
        columns=dataset.columns,
        binnings=training_binnings(dataset, fold, bins=bins),
    )


# ── the whole ladder, in one call ────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class LadderResult:
    """Everything one run of the pipeline proved, and everything it did not."""

    dataset: Dataset
    run: WalkForward
    artifact: Artifact
    model_parity: ModelParityReport
    feature_parity: Mapping[str, FeatureParityReport]
    drift: DriftReport

    @property
    def passed(self) -> bool:
        """The two gates that are pass/fail. Drift is an alarm, not a verdict —
        a market that moved is not a bug, and treating it as one would make the
        deploy gate red for the one thing it cannot fix."""
        return self.model_parity.passed and all(r.passed for r in self.feature_parity.values())

    def summary(self) -> str:
        lines = [
            repr(self.dataset),
            f"spec: {self.dataset.spec_ref}",
            f"artifact: {self.artifact.ref} ({self.artifact.meta.content_sha256[:19]}…, "
            f"{self.artifact.meta.content_bytes} bytes, "
            f"roundtrip max_abs_diff={self.artifact.meta.roundtrip_max_abs_diff:g})",
            self.run.evaluation.summary(),
            # Printed here, not left to a reader with a notebook: the decomposition is
            # the number that decides whether a gross edge is skill or drift, and the
            # one time it was worked out by hand beside the pipeline it was worked out
            # on the wrong rows and published (ADR-0022).
            decompose(self.run).summary(),
            self.model_parity.summary(),
        ]
        lines += [f"{coin}: {report.summary()}" for coin, report in self.feature_parity.items()]
        lines.append(self.drift.summary())
        return "\n".join(lines)


def climb(
    candles: Sequence[Candles],
    registry: ModelRegistry,
    *,
    registry_id: str = "perp_bar_xgb",
    spec: FeatureSpec = PERP_BAR_V1,
    folds: int = 4,
    embargo_bars: int = 0,
    params: PerpBarParams | None = None,
    buffer_bars: int = SERVING_BUFFER_BARS,
    xgb_params: Mapping[str, Any] | None = None,
    parity_bars: int | None = None,
) -> LadderResult:
    """Build, fit, export, register and gate — the whole rung, once.

    Returns the reports rather than asserting on them. A gate that fails is a
    *result* (ADR-0016), and a pipeline that raised on one would be a pipeline whose
    failures never got looked at properly.
    """
    strategy_params = params or PerpBarParams(symbol_id=0)
    dataset = build_dataset(candles, spec=spec, horizon=LABEL_HORIZON_BARS)
    run = walk_forward_fit(
        dataset,
        folds=folds,
        embargo_bars=embargo_bars,
        entry_edge=strategy_params.entry_edge,
        params=xgb_params,
    )
    artifact = export_walk_forward(run, registry, registry_id=registry_id, spec=spec)

    served = PerpBar.from_registry(
        registry,
        registry_id,
        strategy_params,
        version=artifact.meta.version,
        spec=spec,
        buffer_bars=buffer_bars,
    )
    parity = {
        c.coin: feature_parity_gate(
            served,
            c,
            symbol_id=strategy_params.symbol_id,
            spec=spec,
            replay_bars_from=parity_bars,
        )
        for c in candles
    }
    return LadderResult(
        dataset=dataset,
        run=run,
        artifact=artifact,
        model_parity=model_parity_gate(run, artifact, entry_edge=strategy_params.entry_edge),
        feature_parity=parity,
        drift=drift_gate(dataset, run.final_fold),
    )


__all__ = [
    "MAKER_FEE_BPS",
    "TAKER_FEE_BPS",
    "XGB_PARAMS",
    "Dataset",
    "Decomposition",
    "Evaluation",
    "FoldResult",
    "LadderResult",
    "TrainingError",
    "WalkForward",
    "build_dataset",
    "climb",
    "decompose",
    "drift_gate",
    "export_walk_forward",
    "feature_parity_gate",
    "fit",
    "fit_fold",
    "model_parity_gate",
    "owed_rows",
    "replay_bars",
    "roc_auc",
    "score_fold",
    "training_binnings",
    "walk_forward_fit",
]
