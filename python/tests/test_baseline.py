"""Tests for the no-model baseline (:mod:`axon.strategies.baseline`).

Each name says what would break if the check were removed, because that is the
only reading of a test that survives the day it goes red at 2 a.m. Nothing here
imports an ML library, and one of the tests is about exactly that.
"""

from __future__ import annotations

import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from axon.contracts import FIXED_POINT_SCALE
from axon.features import FeatureDef, FeatureSpec, bar_inputs
from axon.parity import aligned_feature_parity
from axon.signals import RingConsumer
from axon.strategies.baseline import (
    BASELINE_Z_V1,
    NO_MODEL_VERSION,
    SERVING_BUFFER_BARS,
    VOL_COLUMN,
    Z_COLUMN,
    Baseline,
    BaselineParams,
    baseline_spec,
    evaluate_baseline,
    warmup_bars,
    warmup_minutes,
)
from axon.strategies.data import fixture_candles
from axon.strategy import Bar, NotInEventScope, StrategyContext

SYMBOL = 7


def params(**overrides) -> BaselineParams:
    return BaselineParams(symbol_id=SYMBOL, **overrides)


def bars_from(candles, *, symbol_id: int = SYMBOL):
    """The candle history as the ``Bar`` events a live session would deliver."""
    for i in range(len(candles)):
        yield Bar(
            symbol_id=symbol_id,
            ts_event=int(candles.ts_event[i]),
            open=int(candles.open[i]),
            high=int(candles.high[i]),
            low=int(candles.low[i]),
            close=int(candles.close[i]),
            volume=int(candles.volume[i]),
        )


FIRST_CLOSE_NS = 1_700_000_000_000_000_000


def synthetic_bars(close: np.ndarray, *, symbol_id: int = SYMBOL, start_ns: int = FIRST_CLOSE_NS):
    """Bars from a close series in fixed-point, one hour apart.

    ``ts_event`` is the bar's **close**, spaced by arithmetic rather than read off a
    clock — the same rule the venue's own candles follow (``open_time + interval``).
    """
    step = 3_600_000_000_000
    for i, c in enumerate(close):
        yield Bar(
            symbol_id=symbol_id,
            ts_event=start_ns + (i + 1) * step,
            open=int(c),
            high=int(c),
            low=int(c),
            close=int(c),
            volume=10**9,
        )


def drive(strategy: Baseline, bars, *, model_version: int = NO_MODEL_VERSION):
    """Run bars through a real context and hand back the records it produced."""
    ctx = StrategyContext(model_version=model_version)
    out = []
    for bar in bars:
        with ctx.event(bar.ts_event):
            strategy.on_bar(bar, ctx)
        out.extend(ctx.take_pending())
    return out


# ── the constraint the whole phase rests on ──────────────────────────────────


def test_a_bounded_serving_buffer_reproduces_the_offline_recompute_bit_for_bit():
    """Without this the feature-parity gate is measuring history length, not code.

    The serving path holds :data:`SERVING_BUFFER_BARS` bars; the offline recompute
    sees the whole download. A windowed transform forgets exactly that far back and
    the two agree to the *bit*; an exponentially weighted one never forgets its seed
    and the two disagree — largest right after a restart, which is the moment nobody
    is comparing feature values.
    """
    candles = fixture_candles("BTC")
    inputs = candles.feature_inputs()
    full = BASELINE_Z_V1.compute(inputs)

    compared = 0
    for end in range(200, len(candles) + 1):
        window = {k: v[max(0, end - SERVING_BUFFER_BARS) : end] for k, v in inputs.items()}
        served = BASELINE_Z_V1.compute(window)[-1]
        # `array_equal`, not `allclose`: "within tolerance" is the claim this spec
        # exists to be stronger than.
        assert np.array_equal(served, full[end - 1]), f"row {end - 1} differs off a buffer"
        compared += 1
    assert compared > 500  # a green run over three rows would prove nothing

    # The counter-example, in the same test so the claim cannot be read as vacuous.
    ema = FeatureSpec(
        name="baseline_ema_counterexample",
        version=1,
        features=(FeatureDef("e", "ema", params={"span": 20}, inputs={"x": "close"}),),
    )
    buffered = {k: v[-SERVING_BUFFER_BARS:] for k, v in inputs.items()}
    assert ema.compute(buffered)[-1, 0] != ema.compute(inputs)[-1, 0]


def test_every_row_the_strategy_served_survives_the_feature_parity_gate():
    """Without this the rule could be bit-exact in a unit test and skewed in serving.

    The gate the model strategies are held to, applied to a strategy with no model:
    the online vectors the running object produced, aligned on event time against the
    offline recompute of the same bars. ``scope="declared"`` because the offline side
    here *is* exactly the rows the serving path owed — the default would excuse a path
    that was blind through its own opening rows.
    """
    candles = fixture_candles("BTC")
    strategy = Baseline(params())

    online_ts, online_rows = [], []
    ctx = StrategyContext(model_version=NO_MODEL_VERSION)
    for bar in bars_from(candles):
        with ctx.event(bar.ts_event):
            strategy.on_bar(bar, ctx)
        ctx.take_pending()
        row = strategy.feature_row()
        if row is not None:
            online_ts.append(bar.ts_event)
            online_rows.append(np.asarray(row, dtype=np.float64))

    offline = BASELINE_Z_V1.compute(candles.feature_inputs())
    keep = np.isin(candles.ts_event, np.asarray(online_ts, dtype=np.int64))

    report = aligned_feature_parity(
        np.asarray(online_rows),
        offline[keep],
        online_ts=np.asarray(online_ts, dtype=np.int64),
        offline_ts=candles.ts_event[keep],
        columns=BASELINE_Z_V1.columns,
        scope="declared",
    )
    assert report.passed, report.summary()
    assert report.max_abs_diff == 0.0
    assert len(online_rows) == len(candles) - warmup_bars(BASELINE_Z_V1) + 1


# ── warmup, and the interval that follows from it ────────────────────────────


def test_the_warmup_is_measured_from_the_spec_rather_than_restated_in_a_comment():
    """Without this a widened window silently mutes the strategy forever.

    A buffer one bar short of the spec's warmup leaves the last row NaN on every bar.
    The strategy then emits nothing, nothing raises, and "never trades" is
    indistinguishable from "no opinion" to anyone who does not already suspect it.
    """
    assert warmup_bars(BASELINE_Z_V1) == 21  # 20 for the z-score, 21 for the volatility
    # The number moves with the spec, which is the whole point of measuring it.
    assert warmup_bars(baseline_spec(window=5)) == 6
    assert warmup_bars(baseline_spec(window=50)) == 51
    # Never warms up inside the probe → None, not a wrong number.
    assert warmup_bars(baseline_spec(window=50), probe_bars=30) is None


def test_the_warmup_in_wall_clock_is_what_decides_the_bar_interval():
    """Without this someone runs a 21-bar spec on hourly bars and observes nothing.

    The identical recipe is 21 minutes of silence on m1 and 21 *hours* of it on 1h.
    A session that does not outlive its own warmup never says anything at all, and
    that is a property of the interval, not of the rule.
    """
    assert warmup_minutes(BASELINE_Z_V1, "1m") == 21.0
    assert warmup_minutes(BASELINE_Z_V1, "5m") == 105.0
    assert warmup_minutes(BASELINE_Z_V1, "1h") == 21 * 60.0
    with pytest.raises(ValueError, match="unsupported interval"):
        warmup_minutes(BASELINE_Z_V1, "m1")  # the Rust config's spelling, not this one


def test_the_strategy_says_nothing_until_its_window_is_full():
    """Without this the first position is the largest one, taken on the least data.

    A partial window has a tiny standard deviation, so every z-score looks extreme and
    the strategy opens at full size on its worst information.
    """
    candles = fixture_candles("BTC")
    warmup = warmup_bars(BASELINE_Z_V1)
    strategy = Baseline(params())
    bars = list(bars_from(candles))

    assert drive(strategy, bars[: warmup - 1]) == []
    assert strategy.feature_row() is None
    # One more bar fills the window, and the rule has an opinion from here on.
    drive(strategy, bars[warmup - 1 : warmup])
    assert strategy.feature_row() is not None


def test_a_buffer_shorter_than_the_specs_warmup_is_refused_at_construction():
    """Without this the strategy emits nothing, forever, and nothing raises."""
    with pytest.raises(ValueError, match="emits nothing, forever"):
        Baseline(params(), buffer_bars=10)


# ── refusals ─────────────────────────────────────────────────────────────────


def test_a_frozen_tape_emits_nothing_rather_than_a_position_from_a_zero_denominator():
    """Without this a stalled feed produces a NaN or infinite z-score, not silence."""
    close = np.full(60, 50_000 * 10**8, dtype=np.int64)
    strategy = Baseline(params())
    assert drive(strategy, synthetic_bars(close)) == []
    assert strategy.feature_row() is None
    assert strategy.target == Decimal(0)


def test_a_constant_growth_tape_is_refused_even_though_its_z_score_is_finite():
    """Without the volatility floor the rule shorts a tape that has no volatility.

    A feed on a fixed geometric grid — every bar exactly twice the last — has 20
    identical log returns, so realized volatility is *exactly* zero while the
    z-score is a confident +3.78. Every price differs, so nothing upstream goes NaN:
    this is the one shape where a finite, extreme, meaningless z reaches the
    decision, and it is the case the ``vol`` column is in the spec for.
    """
    close = 10**8 * (2 ** np.arange(25, dtype=np.int64))
    inputs = bar_inputs(*(close for _ in range(4)), np.full(25, 10**9, dtype=np.int64))
    row = BASELINE_Z_V1.compute(inputs)[-1]
    assert np.isfinite(row).all()  # nothing upstream refused it
    assert row[BASELINE_Z_V1.columns.index(Z_COLUMN)] >= 1.5  # and it is past the entry band
    assert row[BASELINE_Z_V1.columns.index(VOL_COLUMN)] == 0.0

    strategy = Baseline(params())
    assert drive(strategy, synthetic_bars(close)) == []
    assert strategy.feature_row() is None


def test_another_instruments_bars_do_not_enter_this_instruments_buffer():
    """Without this a two-symbol session computes one z-score over two price series."""
    candles = fixture_candles("BTC")
    strategy = Baseline(params())
    foreign = list(bars_from(candles, symbol_id=SYMBOL + 1))
    assert drive(strategy, foreign) == []
    assert strategy.feature_row() is None


def test_the_decision_is_read_by_column_name_so_a_reordered_spec_still_trades_the_z():
    """Without this an inserted column silently makes the rule trade its volatility."""
    reordered = FeatureSpec(
        name="baseline_z",
        version=1,
        features=tuple(reversed(baseline_spec().features)),
    )
    assert reordered.columns == (VOL_COLUMN, Z_COLUMN)
    candles = fixture_candles("BTC")
    bars = list(bars_from(candles))

    forward = drive(Baseline(params()), bars)
    backward = drive(Baseline(params(), spec=reordered), bars)
    assert [int(r["target_qty"]) for r in forward] == [int(r["target_qty"]) for r in backward]
    assert forward  # and it actually traded, so the equality is not two empty lists


def test_a_spec_without_the_columns_the_rule_reads_is_refused_rather_than_indexed():
    """Without this the wrong spec is served and every decision is off some other column."""
    wrong = FeatureSpec(
        name="not_the_baseline",
        version=1,
        features=(FeatureDef("mom", "momentum", params={"window": 4}, inputs={"price": "close"}),),
    )
    with pytest.raises(ValueError, match="missing"):
        Baseline(params(), spec=wrong)


# ── the band ─────────────────────────────────────────────────────────────────


def test_the_target_holds_inside_the_band_so_a_wobbling_z_is_not_a_round_trip_a_bar():
    """Without hysteresis a z-score sitting on the threshold flips every bar, in fees."""
    strategy = Baseline(params())
    assert strategy.target_for(-2.0) == params().max_position  # cheap → long
    strategy._target = params().max_position  # noqa: SLF001 — the state under test
    assert strategy.target_for(1.0) == params().max_position  # inside the band: hold
    assert strategy.target_for(-1.0) == params().max_position
    assert strategy.target_for(0.4) == Decimal(0)  # through the exit band: flat
    assert strategy.target_for(2.0) == -params().max_position  # rich → short


def test_an_unchanged_target_is_not_re_emitted():
    """Without this every bar costs a ring slot and rate-limit budget to say nothing."""
    candles = fixture_candles("BTC")
    records = drive(Baseline(params()), bars_from(candles))
    targets = [int(r["target_qty"]) for r in records]
    assert targets, "the rule took no position at all on real data; nothing below is tested"
    assert all(a != b for a, b in zip(targets, targets[1:]))


# ── the wire ─────────────────────────────────────────────────────────────────


def test_the_wire_cannot_say_there_is_no_model_so_a_no_model_run_stamps_a_sentinel():
    """Without a distinct stamp, a no-model capture and a model-1 capture are identical.

    ``model_version`` is a mandatory ``u32`` on every signal and the context refuses
    ``0`` deliberately — zero is the value of a field nobody wrote. So a rule with no
    artifact must still name a model, and the only choice left is *which lie*: one
    that reads as "the first registered model" or one that reads as "look this up".
    """
    with pytest.raises(ValueError, match="model_version must be in 1"):
        StrategyContext(model_version=0)
    assert NO_MODEL_VERSION != 1
    assert 1 <= NO_MODEL_VERSION <= 2**32 - 1

    candles = fixture_candles("BTC")
    records = drive(Baseline(params()), bars_from(candles))
    assert records
    assert all(int(r["model_version"]) == NO_MODEL_VERSION for r in records)
    # And the strategy itself refuses to invent an artifact version.
    assert Baseline(params()).artifact_version is None


def test_a_signal_crosses_a_real_ring_carrying_the_bars_own_close_time(tmp_path):
    """Without this the rule is proven only as far as the context that queued it.

    ``features → decision → Signal → the signal ring`` is the path every ML strategy
    drives, and the ring is the half that can fail. This drives the real
    :class:`~axon.live.StrategyRunner` onto a real ring file and reads the records
    back off it, so what is asserted is what a Rust consumer would see.
    """
    from axon.live import StrategyRunner

    candles = fixture_candles("BTC")
    bars = list(bars_from(candles))
    ring = str(tmp_path / "baseline.ring")

    with StrategyRunner(
        Baseline(params()),
        ring_path=ring,
        capacity=4096,
        model_version=NO_MODEL_VERSION,
        liveness_path=str(tmp_path / "baseline.live"),
    ) as runner:
        runner.run(bars)
        stats = runner.stats
        consumer = RingConsumer(ring)
        batch = consumer.read_batch()
        consumer.close()

    assert stats.signals_emitted > 0
    assert stats.signals_pushed == stats.signals_emitted
    assert len(batch) == stats.signals_pushed

    stamps = {int(b.ts_event) for b in bars}
    unit = int(params().max_position * FIXED_POINT_SCALE)
    for i, rec in enumerate(batch):
        assert int(rec["seq"]) == i  # no gaps: the consumer's gap detector means something
        assert int(rec["ts_event"]) in stamps  # a bar close, never a wall clock
        assert int(rec["symbol_id"]) == SYMBOL
        assert int(rec["target_qty"]) in (-unit, 0, unit)
        assert int(rec["urgency"]) == 0  # post-only: the edge cannot pay a taker fee
        assert int(rec["model_version"]) == NO_MODEL_VERSION
    assert [int(r["ts_event"]) for r in batch] == sorted(int(r["ts_event"]) for r in batch)


def test_the_strategy_cannot_emit_outside_an_event_scope():
    """Without this a signal gets stamped from a wall clock instead of a bar close."""
    ctx = StrategyContext(model_version=NO_MODEL_VERSION)
    strategy = Baseline(params())
    bars = list(bars_from(fixture_candles("BTC")))
    with pytest.raises(NotInEventScope):
        for bar in bars:
            strategy.on_bar(bar, ctx)  # no `with ctx.event(...)` around it


# ── no artifact, and no library to load one with ─────────────────────────────


def test_deciding_and_emitting_needs_no_ml_library_at_all():
    """Without this the "no-model" baseline quietly depends on the ML stack it isolates.

    Run in a subprocess, because another test in this session may already have
    imported xgboost and the check would then pass for the wrong reason. The claim is
    the load-bearing one for the whole workstream: if this rule needed an ML library
    to reach the ring, it could not be the thing that runs when the ML side does not.
    """
    package_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from decimal import Decimal
import numpy as np
from axon.strategies.baseline import Baseline, BaselineParams, NO_MODEL_VERSION
from axon.strategies.data import fixture_candles
from axon.strategy import Bar, StrategyContext

c = fixture_candles("BTC")
s = Baseline(BaselineParams(symbol_id=1))
ctx = StrategyContext(model_version=NO_MODEL_VERSION)
n = 0
for i in range(len(c)):
    bar = Bar(1, int(c.ts_event[i]), int(c.open[i]), int(c.high[i]),
              int(c.low[i]), int(c.close[i]), int(c.volume[i]))
    with ctx.event(bar.ts_event):
        s.on_bar(bar, ctx)
    n += len(ctx.take_pending())
assert n > 0, "the rule emitted nothing, so this proves nothing"
heavy = sorted({m.split(".")[0] for m in sys.modules} & {
    "xgboost", "lightgbm", "sklearn", "onnx", "onnxruntime", "skl2onnx", "torch",
    "pandas", "scipy",
})
print(f"signals={n} heavy={heavy}")
assert not heavy, heavy
assert not hasattr(s, "predictor")
assert not hasattr(Baseline, "from_registry")
# The gates are not on the serving path. `axon.models` *is* imported, but only
# because `axon.strategies.__init__` re-exports `perp_bar` — no ML library comes
# with it, which is the claim above, and this line is the canary for the day one
# does.
assert "axon.parity" not in sys.modules
assert "axon.strategies.training" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(package_root))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "heavy=[]" in result.stdout


# ── the verdict ──────────────────────────────────────────────────────────────


def test_the_verdict_counts_the_turnover_it_emitted_rather_than_recomputing_the_rule():
    """Without this the reported fee drag is a model of the strategy, not the strategy.

    ``training.py`` refuses to net costs off an edge because netting needs a turnover
    model, and a turnover model beside the evaluator is a second implementation of the
    hysteresis. This evaluator does not model turnover: it reads it off the records the
    strategy actually emitted, which is what earns it the right to subtract.
    """
    candles = fixture_candles("BTC")
    verdict = evaluate_baseline(candles)

    assert verdict.bars == len(candles)
    assert verdict.warmup_bars == warmup_bars(BASELINE_Z_V1)
    assert verdict.signals > 0
    # Every emitted record moves the position by at least one unit; a flip moves two.
    assert verdict.sides >= verdict.signals
    assert 0 < verdict.held_bars < verdict.bars

    # The free benchmark is a constant short on exactly the rule's own rows.
    assert verdict.benchmark_edge_bps == pytest.approx(-verdict.drift_bps)
    assert verdict.selection_bps == pytest.approx(verdict.edge_bps - verdict.benchmark_edge_bps)

    # Fees are in the same units as the edge, so the subtraction needs no third number.
    assert verdict.drag_bps(verdict.maker_fee_bps) == pytest.approx(
        verdict.sides * verdict.maker_fee_bps / verdict.held_bars
    )
    assert verdict.net_bps(verdict.maker_fee_bps) < verdict.edge_bps
    assert verdict.net_bps(verdict.taker_fee_bps) < verdict.net_bps(verdict.maker_fee_bps)

    text = verdict.describe()
    assert "not a P&L" in text
    assert "selection" in text


def test_the_verdicts_turnover_matches_the_records_the_same_bars_put_on_a_ring(tmp_path):
    """Without this the verdict and a shadow run could disagree about what was emitted."""
    from axon.live import StrategyRunner

    candles = fixture_candles("BTC")
    verdict = evaluate_baseline(candles)
    ring = str(tmp_path / "verdict.ring")
    with StrategyRunner(
        Baseline(params()),
        ring_path=ring,
        capacity=4096,
        model_version=NO_MODEL_VERSION,
        liveness_path=str(tmp_path / "verdict.live"),
    ) as runner:
        runner.run(list(bars_from(candles)))
        consumer = RingConsumer(ring)
        batch = consumer.read_batch()
        consumer.close()

    assert len(batch) == verdict.signals
    held, sides = Decimal(0), Decimal(0)
    for rec in batch:
        target = Decimal(int(rec["target_qty"])) / Decimal(FIXED_POINT_SCALE)
        sides += abs(target - held) / params().max_position
        held = target
    assert float(sides) == pytest.approx(verdict.sides)


# ── restart ──────────────────────────────────────────────────────────────────


def test_a_warm_restart_resumes_on_the_size_it_left_off_on():
    """Without this the first post-restart signal is a real order for a rounding error."""
    candles = fixture_candles("BTC")
    bars = list(bars_from(candles))
    warm = Baseline(params())
    drive(warm, bars[:-1])
    saved = warm.on_save()
    assert isinstance(saved["target"], str)  # a Decimal through float comes back different

    restored = Baseline(params())
    restored.on_load(saved)
    assert restored.target == warm.target
    assert restored.feature_row() is not None
    assert np.array_equal(restored.feature_row(), warm.feature_row())

    # And the next bar produces the same decision on both, which is the actual claim:
    # a restart that resumes a hair off would trade the difference for no reason.
    assert [int(r["target_qty"]) for r in drive(restored, bars[-1:])] == [
        int(r["target_qty"]) for r in drive(warm, bars[-1:])
    ]
