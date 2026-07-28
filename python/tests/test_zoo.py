"""``axon.strategies.zoo``: three model families held to one recipe and four gates.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic: the committed m1 fixture, fixed seeds, no clock, no network.

The heavy ML stack is optional (``pyproject`` puts it in the ``ml`` extra), so every
test that needs one of those libraries ``importorskip``s it. A bare numpy + pytest
environment still runs the whole first half of this file — the shared spec's warmup,
its bit-for-bit buffer property, the criterion table and the two score-space traps —
which is where the properties that outlive any particular model live.

Two tests carry the weight, and they are the two that came out *negative*.
``test_a_written_bundle_is_not_a_crossing`` pins the measured boundary, and it is the
row this workstream got wrong first: once every classifier graph is narrowed to one
score column, all three families write a parity bundle and only two of them cross —
the boosted tree is refused by ``tract`` at *load*, on an attribute, long after the
bundle is on disk. ``test_a_tree_bundles_margin_is_not_the_probability_the_strategy_acts_on``
pins the other trap, which fails in the opposite direction: it looks like a
catastrophically broken model and is a missing ``sigmoid``.
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.features import (
    BAR_M1_V1,
    BAR_M1_WARMUP_BARS,
    FeatureDef,
    FeatureSpec,
    finite_rows,
)
from axon.models import ModelRegistry
from axon.parity.rust_gate import (
    ONNX_EPS,
    ONNX_TIGHT_EPS,
    SCORE_SPACE,
    SERVABLE_KINDS,
    BundleError,
    Criterion,
)
from axon.strategies.baseline import warmup_bars, warmup_minutes
from axon.strategies.data import INTERVAL_MS, fixture_candles
from axon.strategies.zoo import (
    live_strategy,
    FAMILIES,
    INTERVAL,
    SERVING_BUFFER_BARS,
    ZOO_PARAMS,
    family,
    margin_versus_probability,
    positive_class,
    run_zoo,
)

#: Both committed m1 coins. Pooled rather than one, and not for statistical power:
#: with a single coin's 900 bars the first fold's model is close enough to constant
#: that the bundle writer correctly refuses it — "the holdout does not spread enough
#: to gate a decision on" — and the cross-language half of this file would then be
#: skipped by an accident of fixture size rather than by a finding.
COINS = ("BTC", "ETH")


@pytest.fixture(scope="module")
def candles():
    return fixture_candles(COINS[0], INTERVAL)


@pytest.fixture(scope="module")
def zoo_registry(tmp_path_factory):
    """The registry the module-scoped zoo mints into.

    Its own fixture because ``FamilyResult`` carries the artifact and not the registry
    it came from, and a test that reconstructed the path from a `tmp_path_factory` call
    of its own would get a different directory.
    """
    return ModelRegistry(tmp_path_factory.mktemp("zoo-shared") / "registry")


@pytest.fixture(scope="module")
def zoo(tmp_path_factory, zoo_registry):
    """The whole zoo, once, over the committed fixture.

    Module-scoped because it fits three walk-forwards, three exports and six replays
    into one run: fitting the same models per test would multiply a 10-second fixture
    by the number of questions asked of it, and the tests below are questions about
    *one* set of results, not about repeatability of the fit.
    """
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    root = tmp_path_factory.mktemp("zoo")
    return run_zoo(
        [fixture_candles(c, INTERVAL) for c in COINS],
        zoo_registry,
        bundle_root=root / "bundles",
        folds=2,
    )


def result(zoo, name):
    return next(r for r in zoo.results if r.family.name == name)


# ── the shared recipe ─────────────────────────────────────────────────────────


def test_the_shared_spec_warms_up_in_minutes_rather_than_in_days(candles):
    """A window chosen on bar *count* silently chooses a wall-clock blackout.

    ``perp_bar``'s longest window is 24 bars, which on hourly candles means the
    strategy has no opinion for its first 25 hours — a session has to outlive a day
    before it says anything at all, and the publisher only emits a closed bar once the
    venue starts the next one. The same arithmetic on m1 is 21 minutes. This asserts
    the number rather than the intent, because the intent is a docstring and the
    number is what an operator waits through.
    """
    matrix = BAR_M1_V1.compute(candles.feature_inputs())
    first_finite = int(np.argmax(finite_rows(matrix)))
    assert first_finite == BAR_M1_WARMUP_BARS - 1

    minutes = BAR_M1_WARMUP_BARS * INTERVAL_MS[INTERVAL] / 60_000
    assert minutes <= 30, (
        f"{BAR_M1_WARMUP_BARS} bars of {INTERVAL} is a {minutes:.0f}-minute blackout "
        "at the start of every session"
    )

    # The same claim measured a second way, from the other side of the package: the
    # baseline's probe drives a synthetic strictly-varying ramp through the spec, so
    # any NaN in its answer is the *window* rather than the data. The two agreeing is
    # what says 21 is a property of the recipe and not of this particular fixture —
    # and real bars can only be worse, because a bar with no range blanks `clv`.
    assert warmup_bars(BAR_M1_V1) == BAR_M1_WARMUP_BARS
    assert warmup_minutes(BAR_M1_V1, INTERVAL) == minutes


def test_a_bounded_buffer_reproduces_the_shared_specs_last_row_bit_for_bit(candles):
    """The property that makes feature parity *parity* rather than "within tolerance".

    Every window in the spec is finite, so a serving path holding the last
    ``SERVING_BUFFER_BARS`` bars computes the same arithmetic on the same numbers as a
    recompute that saw the whole history. An EMA does not: it never forgets its seed,
    so the two disagree — most right after a restart, which is the moment nobody is
    comparing feature values. The counter-example is in the same test on purpose, so
    the equality below cannot be read as a property of the comparison itself.
    """
    inputs = candles.feature_inputs()
    buffered = {k: v[-SERVING_BUFFER_BARS:] for k, v in inputs.items()}

    full_row = BAR_M1_V1.compute(inputs)[-1]
    buffered_row = BAR_M1_V1.compute(buffered)[-1]
    assert np.array_equal(full_row, buffered_row), "not equal to the last bit"

    expanding = FeatureSpec(
        name="counter_example",
        version=1,
        features=(
            FeatureDef(
                "ema_x", "ema_crossover", params={"fast": 8, "slow": 32}, inputs={"price": "close"}
            ),
        ),
    )
    assert expanding.compute(inputs)[-1, 0] != expanding.compute(buffered)[-1, 0]


def test_the_shared_spec_is_reachable_without_importing_the_model_zoo():
    """A no-model baseline must not have to depend on the models to read the features.

    ``BAR_M1_V1`` lives in :mod:`axon.features` rather than in a strategy module
    precisely because more than one strategy is held to it, and a statistical baseline
    that had to import the zoo to get at its own recipe would carry XGBoost's import
    graph for a rule that fits in ten lines. The identity check is the part that
    matters: two modules exporting two *equal* specs would still be two specs the day
    one of them is edited.
    """
    import axon.features as features
    import axon.strategies.zoo as zoo

    assert zoo.BAR_M1_V1 is features.BAR_M1_V1
    assert features.BAR_M1_V1.ref.startswith("bar_m1/v1#")

    source = (features.spec.__file__ or "").replace(".pyc", ".py")
    assert "axon.strategies" not in open(source, encoding="utf-8").read()


# ── the two score-space traps ─────────────────────────────────────────────────


def test_the_second_column_of_an_onnx_classifier_is_the_one_the_strategy_trades():
    """Column zero is ``P(down)``: a strategy that trades the exact inverse of itself.

    A tree booster hands back a flat vector of ``P(class 1)`` and an ONNX classifier
    graph hands back both columns. There is no symptom for getting this wrong other
    than losing money, and no gate can catch it — both columns are finite, in range,
    and stable across the export.
    """
    both = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert np.array_equal(positive_class(both), [0.3, 0.8])
    assert np.array_equal(positive_class(np.array([0.3, 0.8])), [0.3, 0.8])
    # And the wrong column is not a rounding difference from the right one.
    assert both[:, 0] == pytest.approx(1.0 - positive_class(both))


def test_a_tree_bundles_margin_is_not_the_probability_the_strategy_acts_on(zoo):
    """A margin compared against a probability reports the *link* as a model defect.

    ``SCORE_SPACE`` is ``{"xgboost": "margin", "onnx": "score"}`` because Rust's
    ``TreeModel`` never applies the link — it is monotone, so a threshold on the
    probability is a threshold on the margin, and skipping it keeps ``expf``, whose
    last bit is not portable, off the serving path. The gap measured here is hundreds
    of times any tolerance in the repo, which is exactly why it reads as a broken
    model rather than as a missing ``sigmoid``.
    """
    r = result(zoo, "xgboost")
    sample = r.run.dataset.features[r.run.final_fold.test][-256:]
    gap = margin_versus_probability(r.artifact, sample)
    assert gap > 0.5, gap
    assert SCORE_SPACE["xgboost"] == "margin"
    assert r.bundle is not None and r.bundle.manifest["score_space"] == "margin"


# ── the criterion table ───────────────────────────────────────────────────────


def test_a_family_takes_its_criterion_from_adr_0021_and_cannot_loosen_it():
    """A bundle regenerated after a red gate, with the tolerance nudged, must not pass.

    The criterion is derived from the artifact kind and never chosen by the family:
    exact for a tree ensemble, because ADR-0019 claims ``TreeModel`` reproduces
    ``Booster.predict(output_margin=True)`` and a tolerance there would be the gate
    declining to test its own claim.
    """
    assert family("xgboost").criterion == Criterion("bit_exact")
    assert family("xgboost").eps == 0.0
    for name in ("sklearn_gbm", "logistic"):
        # Two ULP, not the 1e-5 family ceiling: every graph gated in this repo sits
        # four to five orders of magnitude inside that ceiling, and the slack is where
        # a regression passes green (ADR-0021, amended 2026-07-26). The ceiling is
        # still what a bundle may never exceed, which is the assertion below.
        assert family(name).criterion == Criterion("max_abs_diff", ONNX_TIGHT_EPS)
        assert family(name).eps == ONNX_TIGHT_EPS
        assert Criterion.required_for("onnx").allows(family(name).criterion)
        assert ONNX_TIGHT_EPS < ONNX_EPS

    tight = Criterion("max_abs_diff", 1e-9)
    assert Criterion.required_for("onnx").allows(tight)
    assert not Criterion.required_for("onnx").allows(Criterion("max_abs_diff", 1e-3))
    assert not Criterion.required_for("xgboost").allows(Criterion("max_abs_diff", 1e-9))

    with pytest.raises(BundleError, match="no Rust backend serves"):
        Criterion.required_for("lightgbm")
    assert all(f.kind in SERVABLE_KINDS for f in FAMILIES)


# ── the gates, per family ─────────────────────────────────────────────────────


def test_every_family_exports_to_the_kind_its_criterion_was_written_for(zoo):
    """An estimator that stopped converting would export as *something*.

    The first thing to notice would be the Rust gate refusing a kind nobody chose, a
    long way from the fit that changed. ``export_family`` checks the detected kind
    against the family's declaration for that reason; this asserts the three routes
    the zoo claims are the three it took.
    """
    assert [r.artifact.meta.kind for r in zoo.results] == ["xgboost", "onnx", "onnx"]
    for r in zoo.results:
        assert r.artifact.meta.feature_spec_ref == BAR_M1_V1.ref
        assert r.artifact.meta.inputs[0].shape[-1] == len(BAR_M1_V1.columns)


def test_the_serving_path_reproduces_the_offline_matrix_for_every_family(zoo):
    """Training–serving skew, which is invisible in every other number a run prints.

    Zero, not "within tolerance": the online side is a bounded buffer driven one
    ``on_bar`` at a time and the offline side is one vectorized pass over the whole
    history, and every transform in the spec has a finite lookback. The coverage check
    is the other half — an intersection cannot disagree with a row that is not there,
    so a serving path emitting half its rows would otherwise report a flawless zero.
    """
    for r in zoo.results:
        for coin, report in r.feature_parity.items():
            report.raise_for_status()
            assert report.max_abs_diff == 0.0, (r.family.name, coin)
            assert report.coverage is not None and report.coverage.complete


def test_the_registered_artifact_takes_the_same_position_as_the_model_it_came_from(zoo):
    """A candidate inside tolerance everywhere can still flip a decision.

    Which is why the gate is an ``and`` of the numeric criterion and decision
    invariance under the strategy's own entry band, rather than a tolerance check with
    a decision check bolted on as advice.
    """
    for r in zoo.results:
        r.model_parity.raise_for_status()
        assert r.model_parity.n_flips == 0
        assert r.model_parity.max_abs_diff <= r.family.eps
        assert r.model_parity.eps == r.family.eps


def test_a_written_bundle_is_not_a_crossing(zoo):
    """The row this workstream got wrong first, and the reason the table has two columns.

    Narrowing every classifier graph to one score column — the fix ADR-0032 §3 asked
    for and P3 built — makes ``write_parity_bundle`` succeed for **all three**
    families. It does not make all three cross. The boosted tree's bytes are refused
    by ``tract`` at *load*, on an attribute, long after the bundle is on disk:

        attribute 'base_values': expected length 1 (or undefined), got 2

    So a transcript that printed one ``rust=`` column would say ``bundle ok`` for a
    model the core cannot serve. That is the silent green this repo keeps finding, and
    the only defence is to keep the two questions apart: what *this* process measured,
    and what the loader on the other side did.

    The earlier form of this test asserted the bundle was refused, and its docstring
    said it would start failing the day the export path learned to slice the
    positive-class column. That day came; this is the rewrite it asked for.
    """
    for r in zoo.results:
        assert r.bundle is not None, f"{r.family.name}: {r.bundle_error}"
        assert r.bundle.manifest["feature_spec_ref"] == BAR_M1_V1.ref
        assert r.passed

    assert result(zoo, "xgboost").bundle.criterion == Criterion("bit_exact")
    assert result(zoo, "logistic").bundle.criterion == Criterion("max_abs_diff", ONNX_TIGHT_EPS)

    # A bundle for every family, and a crossing for only two of them.
    assert [r.family.crosses for r in zoo.results] == [True, False, True]
    assert "base_values" in result(zoo, "sklearn_gbm").family.rust_loader


def test_a_family_declared_to_cross_carries_nothing_the_loader_refuses(zoo):
    """`crosses` is declared out of band, so something in the gate has to keep it honest.

    A Python process cannot run ``tract``, and it must not pretend to: the boosted
    tree's ``base_values`` is length **1** in the bytes while ``tract`` reports
    ``got 2``, so the number it compares is one it derives internally, and a predicate
    reimplementing that derivation would be a guess wearing a check's clothes.

    What Python *can* check is the loader's three structural preconditions — one FP32
    output of one column, a non-zero ``model_version``, and no ``ai.onnx.ml`` operator
    outside the five ``tract`` 0.23.4 implements. Those are the three that were
    silently violated before, each by a different family, so asserting them is what
    stops :attr:`Family.crosses` from going stale without anyone noticing.

    The ``Scaler`` case is the one this catches most cheaply. The zoo's first linear
    family sat behind a ``StandardScaler``, which compiles to an ``ai.onnx.ml``
    ``Scaler`` — not among the five — so it was refused at *parse* time however its
    outputs were arranged, for a conditioning benefit that measured out at nil.
    """
    onnx = pytest.importorskip("onnx")

    #: Measured against `tract` 0.23.4 with a Rust binary, not read from a changelog.
    reachable = {
        "CategoryMapper",
        "LinearClassifier",
        "LinearRegressor",
        "Normalizer",
        "TreeEnsembleClassifier",
    }
    for r in zoo.results:
        if r.artifact.meta.kind != "onnx":
            continue
        proto = onnx.load_from_string(r.artifact.payload)
        outputs = proto.graph.output
        assert len(outputs) == 1, [o.name for o in outputs]
        dims = outputs[0].type.tensor_type.shape.dim
        assert len(dims) == 2 and dims[1].dim_value == 1, r.family.name
        assert proto.model_version == r.artifact.meta.version

        ml_ops = {n.op_type for n in proto.graph.node if n.domain == "ai.onnx.ml"}
        assert ml_ops <= reachable, f"{r.family.name}: {ml_ops - reachable}"

    logistic = onnx.load_from_string(result(zoo, "logistic").artifact.payload)
    assert {n.op_type for n in logistic.graph.node if n.domain == "ai.onnx.ml"} == {
        "LinearClassifier"
    }


def test_the_boosted_trees_refusal_is_pinned_to_the_shape_that_was_measured(zoo):
    """A declared refusal has to redden when the thing it describes changes.

    `crosses=False` for the boosted tree is a claim about bytes measured once. If
    ``skl2onnx`` ever emits a different ``base_values``/``classlabels`` shape, the
    claim is about a graph that no longer exists — so this pins the exact combination
    that ``tract`` 0.23.4 rejected, and reddens when it moves rather than when someone
    remembers to re-measure.

    **Do not route around the refusal when it fires.** Both obvious repairs were
    measured and both are traps, recorded on ``narrow_to_score_column``: deleting
    ``base_values`` builds and agrees with onnxruntime to seven decimals while serving
    the model *without its intercept*, and padding it to two entries costs 0.18 against
    onnxruntime. Only the parity gate catches either.
    """
    onnx = pytest.importorskip("onnx")

    graph = onnx.load_from_string(result(zoo, "sklearn_gbm").artifact.payload).graph
    node = next(n for n in graph.node if n.op_type == "TreeEnsembleClassifier")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    assert len(attrs["base_values"]) == 1
    assert list(attrs["classlabels_int64s"]) == [0, 1]
    assert attrs["post_transform"] == b"LOGISTIC"
    assert not result(zoo, "sklearn_gbm").family.crosses


def test_drift_alarms_without_making_the_deploy_gate_red(zoo):
    """A market that moved is not a bug, and a gate red for it is a gate ignored.

    Drift is the failure the other two gates cannot see — the code is right and the
    world moved — so it is an alarm with a cause, never part of ``passed``. The m1
    fixture drifts hard inside a single day, which is what makes this assertion
    something other than a tautology here.
    """
    assert zoo.passed
    for r in zoo.results:
        assert not r.drift.passed, "the fixture stopped drifting; this test proves nothing now"
        assert r.drift.significant(), r.drift.summary()
        assert set(f.name for f in r.drift.features) == set(BAR_M1_V1.columns)


def test_the_verdict_prices_selection_against_a_free_benchmark_rather_than_an_auc(zoo):
    """ADR-0022 exists because an AUC of 0.52 was once read as a working strategy.

    The only honest summary shape this repo has is the model's edge over its own
    decision rows against a constant short on *the same rows*, with the turnover that
    would be paid to collect it. A ranking statistic cannot come out negative for a
    model that is merely riding a drift; this subtraction can.
    """
    for r in zoo.results:
        verdict = r.verdict()
        assert "auc" not in verdict.lower()
        assert "selection" in verdict and "bps" in verdict
        assert "fee drag" in verdict
        # The bar is the cheapest fee schedule, not zero: a positive selection under a
        # larger maker drag is a positive number and a losing strategy, and it is the
        # sentence a reader would quote out of the transcript.
        drag = 2 * 1.5 * r.turnover / r.decomposition.n
        assert ("SHOULD NOT TRADE" in verdict) == (r.decomposition.selection_bps <= drag)

        d = r.decomposition
        assert d.benchmark_edge_bps == pytest.approx(-d.drift_bps)
        assert d.selection_bps == pytest.approx(d.edge_bps - d.benchmark_edge_bps)
        # Turnover is counted from the signals the hysteresis actually emitted, not
        # from re-thresholding the scores: the target depends on the previous target,
        # so a vectorized recompute is a different, larger number.
        assert 0 < r.turnover < sum(r.replay_bars.values())


def test_the_gate_table_says_what_happened_in_every_cell(zoo):
    """A table with a blank cell is a table somebody will read as a pass.

    The transcript *is* the deliverable of this workstream, so the row for a family
    that could not cross has to carry the refusal text rather than an empty column.
    """
    table = zoo.table()
    assert BAR_M1_V1.ref in table
    for r in zoo.results:
        row = next(line for line in table.splitlines() if r.family.name in line)
        for column in ("model=", "features=", "drift=", "bundle=", "rust="):
            assert column in row, (r.family.name, column)
        # `bundle` and `rust` are two questions and the row must not merge them: the
        # boosted tree writes a bundle this process can verify and is refused by the
        # loader that would have to serve it.
        assert f"rust={'YES' if r.family.crosses else 'NO'}" in row
        assert "bundle=ok" in row

    # The one row where the two columns disagree, named rather than left to the loop:
    # a bundle this process verified, for a model the core refuses to load.
    gbm = next(line for line in table.splitlines() if "sklearn_gbm" in line)
    assert "bundle=ok" in gbm and "rust=NO" in gbm


def test_the_zoo_holds_every_family_to_one_recipe_and_one_entry_band(zoo):
    """Two models compared under two entry bands is a comparison of the bands.

    Cheap to assert and expensive to notice: a family given a wider band takes fewer,
    more confident positions, and every downstream number — coverage, hit rate, edge,
    turnover, fee drag — moves with it.
    """
    assert len({r.run.evaluation.entry_edge for r in zoo.results}) == 1
    assert all(r.run.evaluation.entry_edge == ZOO_PARAMS.entry_edge for r in zoo.results)
    assert len({r.run.dataset.spec_ref for r in zoo.results}) == 1
    assert len({tuple(r.run.dataset.columns) for r in zoo.results}) == 1
    assert len({r.run.dataset.horizon for r in zoo.results}) == 1


# ── the live factory (ADR-0036) ──────────────────────────────────────────────


def test_the_live_factory_serves_the_same_object_the_gates_measured(zoo, zoo_registry):
    """A live run that built its own strategy would trade something nothing gated.

    The live runner and the shadow runner share one factory table, and this is the
    zoo's entry in it. What the test pins is that the entry resolves to the *same*
    serving path — same spec, same entry band, same predictor kind — because a second
    construction of "the zoo's strategy" is exactly how a live session ends up serving
    a different object from the one feature parity was measured on.
    """
    from decimal import Decimal

    r = result(zoo, "xgboost")
    s = live_strategy(
        symbol_id=3,
        max_position=Decimal("0.0003"),
        registry=zoo_registry,
        model=r.artifact.meta.registry_id,
    )
    assert s.spec is BAR_M1_V1, "the zoo's spec, not PerpBar's hourly default"
    assert s.params.entry_edge == ZOO_PARAMS.entry_edge
    assert s.params.exit_edge == ZOO_PARAMS.exit_edge
    assert s.params.ttl_ms == ZOO_PARAMS.ttl_ms
    assert s.params.urgency == ZOO_PARAMS.urgency, "post-only; a taker fee eats the edge"
    assert s.params.symbol_id == 3
    assert s.params.max_position == Decimal("0.0003")


def test_the_live_factory_refuses_a_run_with_no_artifact_instead_of_serving_a_constant(zoo):
    """The shadow path's no-model fallback is a silent no-op on a live one.

    ``_perp_bar_factory`` falls back to a constant 0.5 predictor when no registry is
    named, which is right for a diff that never looks at a score. A *live* session
    under that fallback sits inside the hysteresis band forever, takes no position, and
    is indistinguishable from a warmed-up strategy with no opinion — nothing raises, no
    counter moves, and the session reads as healthy for as long as anyone leaves it up.
    """
    from axon.strategies.shadow import ShadowError

    with pytest.raises(ShadowError, match="no meaningful behaviour"):
        live_strategy(symbol_id=3, max_position=None, registry=None, model="zoo_xgboost")


def test_the_zoos_own_position_size_is_above_the_briefs_ceiling_and_the_factory_can_override_it(
    zoo, zoo_registry
):
    """The number that arrives in a live config by being copied.

    ``ZOO_PARAMS.max_position`` is 0.001 BTC — about $65 at 2026 testnet prices, over
    the Phase-6 brief's $50 ceiling — so a live run that took the default would breach
    it on its first order. Asserted rather than commented, because the failure is a
    config nobody re-read.
    """
    from decimal import Decimal

    assert ZOO_PARAMS.max_position == Decimal("0.001")
    r = result(zoo, "xgboost")
    default = live_strategy(
        symbol_id=3, max_position=None, registry=zoo_registry, model=r.artifact.meta.registry_id
    )
    assert default.params.max_position == ZOO_PARAMS.max_position, "no silent shrinking"
