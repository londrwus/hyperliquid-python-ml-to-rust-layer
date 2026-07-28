"""``axon.strategies``: one real strategy, and the ladder it climbs (ADR-0022).

Named after the failure modes they prevent, matching
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic: the market data is a **committed slice of real Hyperliquid candles**
rather than a generator, so a green run says the pipeline works on numbers the
venue actually printed — and it never dials out, because the fixture is a file.

Three tests carry most of the weight:

* ``test_a_bounded_serving_buffer_reproduces_the_offline_features_bit_for_bit`` —
  the feature-parity gate on real bars, with the online side genuinely driven one
  ``on_bar`` at a time. ``max_abs_diff == 0`` is the claim; anything else means the
  serving path and the research path are two implementations.
* ``test_training_labels_never_read_a_price_from_the_test_window`` — the purge. A
  strategy that climbs this ladder on a leaky label discredits the whole harness.
* ``test_the_registered_artifact_reproduces_the_model_exactly`` — the model-parity
  gate at zero tolerance, discretized by the strategy's own entry band.
* ``test_a_shadow_run_diffs_every_served_row_against_a_recompute_of_the_bars_it_was_shown``
  — rung 3's continuous diff (ADR-0029), with the denominator asserted beside the
  zero. The shadow tests are offline by construction: the bar ring they read is
  written by a harness, and none of them can say anything about the venue.

The XGBoost-dependent tests ``importorskip`` it, so a bare numpy + pytest
environment runs everything else.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from axon.compute import JOBSPEC_FIELDS
from axon.features import FeatureError, PERP_CORE_V1, finite_rows, registered_features
from axon.models import ArtifactMeta, ModelRegistry
from axon.strategies import (
    LABEL_HORIZON_BARS,
    PERP_BAR_V1,
    SERVING_BUFFER_BARS,
    Candles,
    DataError,
    PerpBar,
    PerpBarParams,
    direction_label,
    fixture_candles,
    fixture_coins,
    forward_log_return,
    purged_walk_forward,
)
from axon.strategies.data import (
    ALLOW_NETWORK_ENV,
    CLOSE_STAMP_OFFSET_MS,
    INTERVAL_MS,
    closed_rows,
)
from axon.strategies.jobs import SCHEMA_ENV, hyperparameter_sweep, walk_forward_job
from axon.strategy.context import StrategyContext
from axon.strategy.events import Bar

INTERVAL = "1h"
BAR_NS = INTERVAL_MS[INTERVAL] * 1_000_000
SYMBOL = 3


class ConstantPredictor:
    """Returns one probability for every row, so a decision test is about the
    decision rather than about a model."""

    def __init__(self, probability: float | np.ndarray = 0.5) -> None:
        self.probability = probability
        self.calls = 0

    def predict(self, x) -> np.ndarray:
        self.calls += 1
        rows = np.asarray(x).shape[0]
        if isinstance(self.probability, np.ndarray):
            return np.tile(self.probability, (rows, 1))
        return np.full(rows, float(self.probability))

    def declared_schema(self):
        return None


class ScriptedPredictor:
    """Reads one probability per call from a list — a path, not a level."""

    def __init__(self, probabilities) -> None:
        self.probabilities = list(probabilities)
        self.index = 0

    def predict(self, x) -> np.ndarray:
        value = self.probabilities[min(self.index, len(self.probabilities) - 1)]
        self.index += 1
        return np.full(np.asarray(x).shape[0], float(value))

    def declared_schema(self):
        return None


def strategy(predictor=None, **kwargs) -> PerpBar:
    params = kwargs.pop("params", PerpBarParams(symbol_id=SYMBOL))
    return PerpBar(params, predictor or ConstantPredictor(0.5), **kwargs)


def bar_at(index: int, candles: Candles, symbol_id: int = SYMBOL) -> Bar:
    return Bar(
        symbol_id=symbol_id,
        ts_event=int(candles.ts_event[index]),
        open=int(candles.open[index]),
        high=int(candles.high[index]),
        low=int(candles.low[index]),
        close=int(candles.close[index]),
        volume=int(candles.volume[index]),
    )


def drive(strat: PerpBar, candles: Candles, *, ctx: StrategyContext | None = None):
    """Feed a whole candle history through ``on_bar`` and return the signals."""
    context = ctx or StrategyContext(model_version=11)
    for i in range(len(candles)):
        bar = bar_at(i, candles)
        with context.event(bar.ts_event):
            strat.on_bar(bar, context)
    return context.take_pending()


# ── the data ──────────────────────────────────────────────────────────────────


def test_the_committed_fixture_is_real_venue_data_and_not_a_generator():
    candles = fixture_candles("BTC", INTERVAL)
    assert len(candles) == 800
    assert candles.gaps == 0
    # Real BTC, not a random walk around 100: a fixture that drifted into synthetic
    # data would still pass every mechanical check below it.
    closes = candles.close.astype(np.float64) / 1e8
    assert 10_000.0 < closes.min() and closes.max() < 1_000_000.0
    assert np.all(np.diff(candles.ts_event) == BAR_NS)
    assert set(fixture_coins(INTERVAL)) == {"BTC", "ETH"}


def test_the_cache_round_trips_through_its_own_decimal_form(tmp_path):
    # The cache is written as decimal strings and read back through Decimal. A
    # float on either side would move a price in the last bits, and every feature
    # downstream would be measuring the serializer.
    original = fixture_candles("ETH", INTERVAL)
    path = original.to_csv(tmp_path / "eth-1h.csv")
    reloaded = Candles.from_csv(path, coin="ETH", interval=INTERVAL)
    for name in ("ts_event", "open", "high", "low", "close", "volume"):
        np.testing.assert_array_equal(getattr(original, name), getattr(reloaded, name))


def test_the_unclosed_bar_is_dropped_rather_than_labelled_from_a_price_that_has_not_happened():
    # The venue returns the bar still forming. Its close is a mid-bar price stamped
    # with a close time in the future, and it is always the last row — which every
    # walk-forward puts in its most recent test window.
    rows = [{"T": 100}, {"T": 200}, {"T": 300}]
    assert [r["T"] for r in closed_rows(rows, now_ms=250)] == [100, 200]


def test_a_live_fetch_is_refused_unless_the_operator_opens_the_network(monkeypatch):
    from axon.strategies.data import fetch_candles

    monkeypatch.delenv(ALLOW_NETWORK_ENV, raising=False)
    with pytest.raises(DataError, match=ALLOW_NETWORK_ENV):
        fetch_candles("BTC", INTERVAL, start_ms=0, end_ms=1)


def test_the_bar_event_time_is_its_close_and_never_its_open():
    # A bar stamped with its open time is the textbook lookahead leak: the strategy
    # appears to have acted on the close a whole bar before it happened.
    row = {"T": BAR_NS // 1_000_000 - 1, "o": "1", "h": "2", "l": "1", "c": "2", "v": "1"}
    candles = Candles.from_rows([row], coin="BTC", interval=INTERVAL)
    assert int(candles.ts_event[0]) == BAR_NS


def test_the_bar_stamp_is_one_ms_past_the_venues_close_and_a_join_needs_both_sides_to_agree():
    """The extra millisecond, and what a one-millisecond disagreement costs.

    ``T`` is the bar's *last* millisecond, so a bar stamped ``T`` sorts equal to every
    trade printed inside it; ``T + 1 ms`` is the instant it is final for ordering,
    which is what ``axon_core::Candle::ts_event`` documents. The number is pinned here
    because the research history and the Rust bar feed are joined on ``ts_event`` and
    on nothing else: one millisecond of disagreement makes hourly bars 3.6e12 ns
    apart, the intersection empty, and the feature-parity gate fail as "an empty
    feature matrix proves nothing" — a long way from the cause. The Rust decoder now
    stamps ``T + 1`` too, and has its own test for it
    (``a_candle_is_stamped_one_ms_past_its_last_millisecond_...`` in ``ws/decode.rs``);
    this is the other end of that one contract.
    """
    from axon.parity import align_by_event_time

    assert CLOSE_STAMP_OFFSET_MS == 1
    candles = fixture_candles("BTC", INTERVAL)
    venue_close_ms = candles.ts_event // 1_000_000 - CLOSE_STAMP_OFFSET_MS
    assert np.all(venue_close_ms % (BAR_NS // 1_000_000) == BAR_NS // 1_000_000 - 1)

    stamped_at_t = candles.ts_event - CLOSE_STAMP_OFFSET_MS * 1_000_000
    left, _ = align_by_event_time(stamped_at_t, candles.ts_event)
    assert left.size == 0  # not one bar in 800 lines up with itself, one ms out


def test_pages_that_overlap_at_their_seam_are_deduplicated_not_doubled():
    # Paged requests repeat the boundary bar. Two rows for one close time would make
    # every rolling window one observation long in the wrong place.
    row = {"T": 3_599_999, "o": "1", "h": "2", "l": "1", "c": "2", "v": "1"}
    candles = Candles.from_rows([row, dict(row, c="3")], coin="BTC", interval=INTERVAL)
    assert len(candles) == 1
    assert int(candles.close[0]) == 3 * 10**8


def test_a_history_whose_times_go_backwards_is_refused_rather_than_reordered():
    good = fixture_candles("BTC", INTERVAL).head(4)
    with pytest.raises(DataError, match="do not increase"):
        Candles(
            coin="BTC",
            interval=INTERVAL,
            ts_event=good.ts_event[::-1].copy(),
            open=good.open,
            high=good.high,
            low=good.low,
            close=good.close,
            volume=good.volume,
        )


# ── the label and the split ───────────────────────────────────────────────────


def test_a_forward_return_leaves_the_unknowable_tail_as_nan():
    out = forward_log_return(np.array([1.0, np.e, np.e**2, 1.0]), horizon=2)
    assert out[0] == pytest.approx(2.0)
    assert np.isnan(out[-1]) and np.isnan(out[-2])


def test_the_forward_label_is_not_a_registered_feature():
    # Registration would let a FeatureSpec put the future in the feature matrix.
    # The point-in-time sweep would catch it — but a leak caught by a test is still
    # a leak that was written.
    assert not any("forward" in name for name in registered_features())


def test_an_exactly_flat_move_is_labelled_down_rather_than_coin_flipped():
    labels = direction_label(np.array([0.01, 0.0, -0.01, np.nan]))
    assert list(labels[:3]) == [1.0, 0.0, 0.0]
    assert np.isnan(labels[3])


def test_training_labels_never_read_a_price_from_the_test_window():
    # The purge. A training row within one horizon of the boundary is labelled with
    # prices from inside the test block, and the only symptom is a backtest that
    # looks good.
    ts = np.arange(400, dtype=np.int64) * BAR_NS
    horizon_ns = LABEL_HORIZON_BARS * BAR_NS
    for fold in purged_walk_forward(ts, folds=3, horizon_ns=horizon_ns):
        last_train = int(ts[fold.train].max())
        assert last_train + horizon_ns <= fold.test_start_ns
        assert int(ts[fold.test].min()) >= fold.test_start_ns


def test_the_embargo_cuts_further_back_than_the_label_overlap_alone():
    ts = np.arange(400, dtype=np.int64) * BAR_NS
    horizon_ns = LABEL_HORIZON_BARS * BAR_NS
    plain = purged_walk_forward(ts, folds=2, horizon_ns=horizon_ns)
    embargoed = purged_walk_forward(ts, folds=2, horizon_ns=horizon_ns, embargo_ns=20 * BAR_NS)
    assert embargoed[0].train.size == plain[0].train.size - 20
    assert embargoed[0].test.size == plain[0].test.size


def test_pooled_coins_are_split_on_time_so_one_coins_future_is_never_in_training():
    # Two symbols interleaved: an index range holds one coin's future next to the
    # other's past, and a positional split silently trains on both.
    hours = np.arange(200, dtype=np.int64) * BAR_NS
    ts = np.repeat(hours, 2)  # BTC and ETH share every bar close
    for fold in purged_walk_forward(ts, folds=2, horizon_ns=LABEL_HORIZON_BARS * BAR_NS):
        assert ts[fold.train].max() < ts[fold.test].min()
        assert fold.train.size % 2 == 0  # both coins cut at the same instant


def test_a_fold_with_no_training_rows_is_refused_rather_than_averaged_over():
    ts = np.arange(10, dtype=np.int64) * BAR_NS
    with pytest.raises(FeatureError, match="too short|cannot be cut"):
        purged_walk_forward(ts, folds=8, horizon_ns=LABEL_HORIZON_BARS * BAR_NS)


# ── the strategy ──────────────────────────────────────────────────────────────


def test_a_bounded_serving_buffer_reproduces_the_offline_features_bit_for_bit():
    # The feature-parity gate, on real bars, with the online side driven one
    # `on_bar` at a time through a 256-bar buffer and the offline side computed in
    # one vectorized pass over the whole history. Zero, not "within tolerance":
    # every transform in the spec has a finite lookback, so the two are the same
    # arithmetic on the same numbers. An expanding statistic (an EMA) would make
    # this test the place where a few basis points of quality quietly leaked.
    from axon.strategies.training import feature_parity_gate

    candles = fixture_candles("BTC", INTERVAL)
    report = feature_parity_gate(strategy(), candles, symbol_id=SYMBOL)
    report.raise_for_status()
    assert report.max_abs_diff == 0.0
    assert report.n_rows == int(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs())).sum())


class HalfBlindPerpBar(PerpBar):
    """A serving path whose feature row goes missing on every second bar.

    Nothing about it is broken in a way a comparison can see: the buffer is intact
    and every row it does produce is exactly right. This is a live NaN guard that is
    stricter than the research path's, a defensive clamp, or a Rust backend that
    returns NaN on inputs Python accepts — the whole family of skew that shows up as
    *absence*, which an intersection matches against nothing at all.
    """

    bars_seen = 0

    def on_reset(self) -> None:
        super().on_reset()
        self.bars_seen = 0

    def on_bar(self, bar, ctx) -> None:
        self.bars_seen += 1
        super().on_bar(bar, ctx)

    def feature_row(self):
        return None if self.bars_seen % 2 == 0 else super().feature_row()


@pytest.mark.parametrize("replay_bars_from", [None, 150])
def test_a_serving_path_that_skips_bars_fails_the_gate_instead_of_reporting_a_zero(
    replay_bars_from,
):
    # The gate aligns the two sides by intersecting their event times, and an
    # intersection cannot see a bar one side never produced. Without a row count in
    # the verdict, a serving path that emitted half the rows reports
    # `PASS ... max_abs_diff=0.000e+00` — identical to a healthy run in every field a
    # reader looks at, and green in the direction that gets deployed. `parity_bars`
    # is parametrized because it is what the ladder fixture and the fast path use,
    # and it is exactly where a check against the full history's row count would be
    # meaningless.
    from axon.strategies.training import feature_parity_gate

    candles = fixture_candles("BTC", INTERVAL)
    half_blind = HalfBlindPerpBar(PerpBarParams(symbol_id=SYMBOL), ConstantPredictor(0.5))
    report = feature_parity_gate(
        half_blind, candles, symbol_id=SYMBOL, replay_bars_from=replay_bars_from
    )

    assert report.max_abs_diff == 0.0  # every cell that was compared agreed exactly
    assert report.n_mismatched == 0 and report.n_nan_mismatched == 0
    assert not report.passed  # ...and the gate fails anyway, on the rows that are missing
    cover = report.coverage
    assert cover.n_online == report.n_rows < cover.n_in_scope
    assert cover.n_offline_within + cover.n_offline_after == cover.n_in_scope - report.n_rows
    # Deliberately no assertion that `n_offline_before` is 0: this path starts on time, so
    # it would be 0 under either scope and could never fail. The declaration is pinned by
    # `..._reddens_the_gate_rather_than_reading_as_out_of_scope`, where the pair actually
    # discriminates.
    assert report.summary().startswith("feature parity FAIL")
    assert "not wide enough" in report.summary()
    with pytest.raises(AssertionError, match="not wide enough"):
        report.raise_for_status()


def test_a_cold_started_replay_is_held_to_its_own_rows_and_not_to_the_whole_history():
    # The row count the online side owes is a recompute over exactly the bars it was
    # shown. Holding a cold start to the full history's finite rows would fire on
    # every healthy `parity_bars` run — the warmup at the start of the replay is
    # legitimately absent — and a guard that cries wolf gets deleted rather than
    # fixed.
    from axon.strategies.training import feature_parity_gate

    candles = fixture_candles("BTC", INTERVAL)
    report = feature_parity_gate(strategy(), candles, symbol_id=SYMBOL, replay_bars_from=150)
    report.raise_for_status()

    span = int(finite_rows(PERP_BAR_V1.compute(candles[-150:].feature_inputs())).sum())
    whole = int(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs())).sum())
    cover = report.coverage
    assert report.n_rows == cover.n_online == cover.n_in_scope == span
    assert span < 150 < whole  # the replay's own warmup is missing, and that is correct


def test_the_gate_has_exactly_one_answer_to_how_many_rows_were_owed():
    # It used to have two: `Coverage`, and a private `ServedFeatureParityReport` that
    # recomputed the same number. Two implementations of one question agree on the day
    # they are written and drift afterwards, and the drift lands in the harness that
    # exists to detect drift. `owed_rows` builds the mask; `Coverage` counts it; the
    # gate returns a plain report.
    from axon.parity import Coverage, FeatureParityReport
    from axon.strategies.training import feature_parity_gate, owed_rows

    candles = fixture_candles("BTC", INTERVAL)
    report = feature_parity_gate(strategy(), candles, symbol_id=SYMBOL, replay_bars_from=150)
    assert type(report) is FeatureParityReport
    assert isinstance(report.coverage, Coverage)
    assert report.coverage.n_in_scope == int(owed_rows(candles, candles[-150:]).sum())


def test_an_online_side_that_starts_late_reddens_the_gate_rather_than_reading_as_out_of_scope():
    """The failure ``scope="declared"`` exists to catch, and the one inference cannot.

    ``Coverage``'s default infers the owed span from the online side's own first stamp,
    which is right for a live monitor whose window opens mid-history — and blind here,
    because a serving path that produced nothing for its first *k* owed rows has
    **exactly the same first stamp** as one that started on time. The two are
    indistinguishable from the data, so excusing them is the honest default and a gate
    has to declare its way out. This gate does: the reference it hands over is already
    trimmed to the owed set, so it says ``scope="declared"`` and a late start is a gap
    like any other (ADR-0030 §1a).
    """
    from axon.strategies.training import feature_parity_gate

    late = 20
    candles = fixture_candles("BTC", INTERVAL)
    # The first bar the spec is finite on. Counted from the offline recompute rather
    # than from the spec's windows, so widening a window does not silently move what
    # this test suppresses.
    warm = int(np.argmax(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs()))))

    class LateStartPerpBar(PerpBar):
        """Healthy, except that its first ``late`` usable rows never appear.

        Suppressed on *bars seen*, not on rows returned: ``feature_row`` is called
        more than once per bar (``on_bar`` asks, and so does the replay driver), so a
        counter on the row would hide a different number of bars than it names.
        """

        bars_seen = 0

        def on_reset(self) -> None:
            super().on_reset()
            self.bars_seen = 0

        def on_bar(self, bar, ctx) -> None:
            self.bars_seen += 1
            super().on_bar(bar, ctx)

        def feature_row(self):
            return None if self.bars_seen <= warm + late else super().feature_row()

    report = feature_parity_gate(
        LateStartPerpBar(PerpBarParams(symbol_id=SYMBOL), ConstantPredictor(0.5)),
        candles,
        symbol_id=SYMBOL,
    )
    assert not report.passed  # the whole point: a late start is a fault, not a late join
    assert report.coverage.n_offline_before == 0  # nothing is out of scope when it was declared
    assert report.coverage.n_offline_within == late
    assert report.max_abs_diff == 0.0  # and every row that *was* compared still agrees


def test_a_buffer_too_short_for_the_spec_is_refused_instead_of_never_trading():
    # Otherwise the last row is NaN forever, the strategy emits nothing, and nothing
    # raises — a strategy that never trades looks exactly like one with no opinion.
    with pytest.raises(ValueError, match="never warms up"):
        strategy(buffer_bars=8)


def test_the_signal_takes_its_event_time_from_the_bar_and_never_a_wall_clock():
    before = time.time_ns()
    signals = drive(strategy(ConstantPredictor(0.9)), fixture_candles("BTC", INTERVAL))
    after = time.time_ns()
    assert signals
    stamped = {int(s["ts_event"]) for s in signals}
    assert stamped <= set(int(t) for t in fixture_candles("BTC", INTERVAL).ts_event)
    assert not any(before <= ts <= after for ts in stamped)


def test_the_deliberate_urgency_and_ttl_reach_the_wire():
    params = PerpBarParams(symbol_id=SYMBOL)
    signals = drive(strategy(ConstantPredictor(0.9), params=params), fixture_candles("BTC"))
    assert int(signals[0]["ttl_ms"]) == params.ttl_ms == 60_000
    assert int(signals[0]["urgency"]) == params.urgency == 0


def test_a_bar_for_another_symbol_is_ignored():
    strat = strategy(ConstantPredictor(0.9))
    candles = fixture_candles("BTC", INTERVAL)
    ctx = StrategyContext(model_version=1)
    for i in range(len(candles)):
        bar = bar_at(i, candles, symbol_id=SYMBOL + 1)
        with ctx.event(bar.ts_event):
            strat.on_bar(bar, ctx)
    assert ctx.take_pending() == []


def test_a_target_is_emitted_only_when_it_changes():
    # A target position is idempotent; re-sending it costs ring space and rate-limit
    # budget and tells the executor nothing new.
    signals = drive(strategy(ConstantPredictor(0.9)), fixture_candles("BTC", INTERVAL))
    assert len(signals) == 1
    assert int(signals[0]["target_qty"]) == 1_000_000  # +0.01 in fixed point


def test_a_broken_bar_emits_nothing_rather_than_flattening_the_position():
    # A bar with no range makes `clv` NaN. Flat is not a safe default: it is an
    # instruction to close a position that may be perfectly fine.
    candles = fixture_candles("BTC", INTERVAL)
    strat = strategy(ConstantPredictor(0.9))
    ctx = StrategyContext(model_version=1)
    signals = drive(strat, candles, ctx=ctx)
    assert len(signals) == 1

    frozen = bar_at(len(candles) - 1, candles)
    flat = Bar(
        symbol_id=frozen.symbol_id,
        ts_event=frozen.ts_event + BAR_NS,
        open=frozen.close,
        high=frozen.close,
        low=frozen.close,
        close=frozen.close,
        volume=frozen.volume,
    )
    with ctx.event(flat.ts_event):
        strat.on_bar(flat, ctx)
    assert ctx.take_pending() == []
    assert strat.target == Decimal("0.01")  # still long, not flattened by a bad bar


def test_the_hysteresis_band_holds_the_position_instead_of_thrashing():
    params = PerpBarParams(symbol_id=SYMBOL, entry_edge=0.05, exit_edge=0.01)
    strat = strategy(ConstantPredictor(0.5), params=params)
    assert strat.target_for(0.58) == params.max_position  # over the entry band
    strat._target = params.max_position
    assert strat.target_for(0.53) == params.max_position  # inside: hold, do not flip
    assert strat.target_for(0.505) == Decimal(0)  # back to even: exit
    assert strat.target_for(0.40) == -params.max_position


def test_the_probability_read_from_a_two_column_score_is_the_up_column():
    # An ONNX classifier returns both columns. Taking element zero would trade the
    # exact inverse of the model, with no symptom but losing money.
    strat = strategy(ConstantPredictor(np.array([0.95, 0.05])))
    assert strat.probability(np.zeros(len(PERP_BAR_V1.columns))) == pytest.approx(0.05)


def test_a_non_finite_probability_never_becomes_a_target():
    strat = strategy(ConstantPredictor(float("nan")))
    assert strat.probability(np.zeros(len(PERP_BAR_V1.columns))) is None


def test_a_warm_restart_resumes_on_exactly_the_size_it_left_off_on():
    strat = strategy(ScriptedPredictor([0.9]))
    drive(strat, fixture_candles("BTC", INTERVAL))
    state = strat.on_save()
    assert isinstance(state["target"], str)  # a float round trip would resume a hair off

    restored = strategy(ScriptedPredictor([0.9]))
    restored.on_load(state)
    assert restored.target == strat.target
    assert restored.feature_row() is not None
    np.testing.assert_array_equal(restored.feature_row(), strat.feature_row())


def test_the_strategy_reaches_the_signal_ring_through_the_live_runner(tmp_path):
    # The seam this whole package exists to feed: the same object, driven by the
    # real bridge, puts a real record on the ring. A strategy that only ever runs
    # inside its own test harness is a strategy nobody has wired to anything.
    from axon.contracts import SIGNAL_DTYPE
    from axon.live import StrategyRunner
    from axon.signals import RingConsumer

    ring = str(tmp_path / "signals.ring")
    candles = fixture_candles("BTC", INTERVAL)
    runner = StrategyRunner(
        strategy(ConstantPredictor(0.9)),
        ring_path=ring,
        model_version=9,
    )
    runner.start(int(candles.ts_event[0]))
    for i in range(len(candles)):
        runner.handle(bar_at(i, candles))
    runner.stop(int(candles.ts_event[-1]))

    with RingConsumer(ring) as consumer:
        records = []
        while (record := consumer.try_pop()) is not None:
            records.append(record)
    assert len(records) == 1
    assert records[0].dtype == SIGNAL_DTYPE
    assert int(records[0]["model_version"]) == 9
    assert int(records[0]["ts_event"]) in set(int(t) for t in candles.ts_event)


# ── shadow trading: rung 3, and what it can and cannot reach (ADR-0029) ──────


def shadow_run(candles, tmp_path, predictor=None, *, klass=PerpBar, **kwargs):
    """One shadow run over a recorded history, through the real runner and ring."""
    from axon.strategies.shadow import shadow_history

    return shadow_history(
        klass(PerpBarParams(symbol_id=SYMBOL), predictor or ConstantPredictor(0.5)),
        candles,
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        **kwargs,
    )


def test_a_shadow_run_diffs_every_served_row_against_a_recompute_of_the_bars_it_was_shown(
    tmp_path,
):
    # Rung 3's continuous diff, over real venue prints: the strategy is driven one
    # closed bar at a time through the real `StrategyRunner`, and every window is
    # compared against a recompute over exactly the bars that run was shown. The
    # denominator is the number that matters — `max_abs_diff = 0` over half the rows
    # reads identically to `max_abs_diff = 0` over all of them.
    candles = fixture_candles("BTC", INTERVAL)
    report = shadow_run(candles, tmp_path)
    report.raise_for_status()

    owed = int(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs())).sum())
    assert report.monitor.max_abs_diff == 0.0
    assert report.rows_compared == report.rows_owed == report.decisions == owed
    assert report.windows > 1  # continuous, not one verdict at the end
    assert report.bars == len(candles)


def test_a_shadow_run_that_goes_blind_fails_on_its_denominator_and_not_on_a_clean_zero(
    tmp_path,
):
    # The same failure `HalfBlindPerpBar` exposes in the offline gate, through the
    # live-shaped loop: every cell that was compared is exactly right, and the run
    # still fails — because half the rows the serving path owed were never produced.
    candles = fixture_candles("BTC", INTERVAL)
    report = shadow_run(candles, tmp_path, klass=HalfBlindPerpBar)

    assert report.monitor.max_abs_diff == 0.0
    assert not report.passed
    assert report.rows_compared < report.rows_owed
    assert "owed rows" in report.summary()


def test_a_shadow_run_blind_at_a_windows_opening_fails_instead_of_shrinking_its_denominator(
    tmp_path,
):
    """The same failure the gate closes with ``scope="declared"``, one level out.

    A shadow window's offline side is the owed set by construction. Under an
    *inferring* scope, rows the serving path missed at a window's **opening** are
    excused as a late join **and taken out of ``n_in_scope``** — so the run's own
    denominator shrinks to fit the damage and still prints a flawless ratio. Nothing
    in the data can catch that: a path blind through its opening rows has exactly the
    same first stamp as one that started on time. The trader therefore *declares* its
    scope, and this test is what stops that declaration going quiet — it asserts the
    blindness lands in ``n_offline_within`` (a gap) rather than ``n_offline_before``
    (a late join), which is only true because the wiring says ``scope="declared"``.
    """
    from axon.parity.monitor import Level, collecting_sink
    from axon.strategies.shadow import HistoryBarSource, ShadowTrader

    late = 20
    candles = fixture_candles("BTC", INTERVAL)
    warm = int(np.argmax(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs()))))

    class LateStartPerpBar(PerpBar):
        """Blind through its opening rows, and perfectly healthy after them."""

        bars_seen = 0

        def on_reset(self) -> None:
            super().on_reset()
            self.bars_seen = 0

        def on_bar(self, bar, ctx) -> None:
            self.bars_seen += 1
            super().on_bar(bar, ctx)

        def feature_row(self):
            return None if self.bars_seen <= warm + late else super().feature_row()

    verdicts: list = []
    source = HistoryBarSource(candles, symbol_id=SYMBOL)
    with ShadowTrader(
        LateStartPerpBar(PerpBarParams(symbol_id=SYMBOL), ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
    ) as trader:
        trader.monitor.sink = collecting_sink(verdicts)
        report = trader.run(source)

    # The blindness is one window's, and it is a gap in it rather than a late join to it.
    offending = [v for v in verdicts if v.coverage and v.coverage.n_matched]
    assert len(offending) == 1
    assert offending[0].coverage.n_offline_within == late
    assert offending[0].coverage.n_offline_before == 0  # declared: nothing is out of scope
    assert offending[0].level is Level.ALARM

    assert not report.passed
    assert report.monitor.max_abs_diff == 0.0  # every row that *was* compared agrees
    # And the denominator is every row that was owed, not what survived a classification.
    owed = int(finite_rows(PERP_BAR_V1.compute(candles.feature_inputs())).sum())
    assert report.rows_owed == owed
    assert report.rows_compared == owed - late
    assert "never produced" in report.summary()


def test_a_shadow_run_reads_its_bars_off_the_real_market_data_bar_ring(tmp_path):
    """The whole consumer side of ADR-0028, driven end to end without a venue.

    Real bar records, a real ring with its own `record_kind` header, the real
    `MdBarRingConsumer`, the real `StrategyRunner` — everything except the Rust
    publisher, which is the one half this cannot stand in for.
    """
    from axon.marketdata import MdBarRingConsumer, bar_ring_path
    from axon.strategies.shadow import RingBarSource, ShadowTrader, publish_bars

    candles = fixture_candles("BTC", INTERVAL)
    ring = bar_ring_path(str(tmp_path / "axon-md.ring"))
    assert publish_bars(candles, ring, symbol_id=SYMBOL, capacity=1024) == len(candles)

    with RingBarSource(MdBarRingConsumer(ring), symbol_id=SYMBOL, path=ring) as source:
        with ShadowTrader(
            strategy(ConstantPredictor(0.5)),
            symbol_id=SYMBOL,
            ring_path=str(tmp_path / "shadow.ring"),
            interval=INTERVAL,
        ) as trader:
            report = trader.run(source, idle_timeout_s=0.2)

    report.raise_for_status()
    assert report.monitor.max_abs_diff == 0.0
    assert report.rows_compared == report.rows_owed > 0
    assert report.feed.first_bars == 1  # the flag the publisher sets on a series opener
    assert report.feed.ring_dropped == 0


def test_a_shadow_run_never_calls_itself_live_because_the_ring_does_not_say_who_wrote_it(
    tmp_path,
):
    # An MdBar record carries no source marker, so a bar written by the Rust core and
    # one written by the harness above are byte-identical to this reader. Inferring
    # "live" from the source *type* would print the word on a run over a file, which
    # is the exact overclaim this module exists not to make.
    from axon.marketdata import MdBarRingConsumer, bar_ring_path
    from axon.strategies.shadow import RingBarSource, ShadowTrader, publish_bars

    candles = fixture_candles("BTC", INTERVAL).head(200)
    ring = bar_ring_path(str(tmp_path / "axon-md.ring"))
    publish_bars(candles, ring, symbol_id=SYMBOL, capacity=256)

    with RingBarSource(MdBarRingConsumer(ring), symbol_id=SYMBOL, path=ring) as source:
        assert "live" not in source.describe()
        with ShadowTrader(
            strategy(ConstantPredictor(0.5)),
            symbol_id=SYMBOL,
            ring_path=str(tmp_path / "shadow.ring"),
            interval=INTERVAL,
        ) as trader:
            report = trader.run(source, idle_timeout_s=0.2)

    assert report.venue_attested is False
    assert "NOT ATTESTED AGAINST THE VENUE" in report.summary()


class _StubBarConsumer:
    """A bar consumer whose two fault counters can be set independently."""

    path = "<stub>"

    def __init__(self, dropped=0, gaps=0, first_bars=0):
        self.dropped, self.gaps, self.first_bars = dropped, gaps, first_bars

    def read_bars(self, max_records=None):
        return []

    def close(self):
        pass


def test_a_ring_drop_and_a_venue_gap_are_never_reported_as_the_same_fault(tmp_path):
    """The two faults ADR-0028 is explicit a consumer must not conflate.

    A `seq` hole means the ring dropped a bar this reader was too slow for; a
    `gap_before` flag means the venue never printed one. They have one symptom and
    opposite meanings — and only the first one is invisible to the diff, because a bar
    the ring dropped reached neither side of it and both agree perfectly about a
    window that is one observation short.
    """
    from axon.strategies.shadow import RingBarSource, ShadowTrader

    source = RingBarSource(_StubBarConsumer(dropped=3, gaps=7, first_bars=1))
    source.poll()
    assert source.health.ring_dropped == 3
    assert source.health.feed_gaps == 7

    with ShadowTrader(
        strategy(ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
    ) as trader:
        report = trader.report(source)
    # A dropped bar corrupts the serving path's window where nothing else can see it,
    # so it fails the run; a venue gap is the market's and is reported, never repaired.
    assert not report.passed
    assert "reached NEITHER side of the diff" in report.summary()
    assert "the venue never printed" in report.summary()


def test_a_venue_gap_is_reported_without_reddening_the_diff_that_recomputes_over_it(tmp_path):
    # Both sides recompute over the same hole, so parity is still exactly zero — and
    # that is precisely why the gap has to be *said*: every windowed feature spanning
    # it covers more calendar time than its window claims.
    candles = fixture_candles("BTC", INTERVAL)
    holed = Candles(
        coin=candles.coin,
        interval=candles.interval,
        **{
            name: np.delete(getattr(candles, name), 400)
            for name in ("ts_event", "open", "high", "low", "close", "volume")
        },
    )
    assert holed.gaps == 1
    report = shadow_run(holed, tmp_path)

    assert report.feed.feed_gaps == 1
    assert report.feed.ring_dropped == 0
    assert report.monitor.max_abs_diff == 0.0
    assert report.passed  # a gap in the market is not a parity failure
    assert "the venue never printed" in report.summary()


def test_a_repeated_bar_close_is_refused_because_the_buffer_would_hold_it_twice(tmp_path):
    # Two bars for one instrument at one close time is a republished bar. Appending it
    # twice makes every window from there one observation wide in the wrong place —
    # on *both* sides of the diff, so the comparison stays green over a corrupted
    # recording. Refusing is the only reading that is not silently wrong.
    from axon.strategies.shadow import ShadowError, ShadowTrader

    candles = fixture_candles("BTC", INTERVAL)
    with ShadowTrader(
        strategy(ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
    ) as trader:
        trader.on_bar(bar_at(1, candles))
        with pytest.raises(ShadowError, match="one observation wide in the wrong place"):
            trader.on_bar(bar_at(1, candles))
        # And a bar that arrives strictly behind is the other half of the same rule:
        # a diff aligned on event time cannot align a series that goes backwards.
        with pytest.raises(ShadowError, match="cannot be aligned"):
            trader.on_bar(bar_at(0, candles))


def test_the_silence_deadline_is_bars_because_an_hourly_strategy_is_silent_for_an_hour(
    tmp_path,
):
    # `DEFAULT_SILENCE_AFTER_NS` is 60 s, which is right for a quote feed and absurd
    # here: a perfectly healthy `perp_bar` session says nothing for an hour at a time
    # and would alarm as a dead feed a minute in. The floor is two intervals, not one,
    # because ADR-0028's publisher emits a bar only once the venue starts the next —
    # so a bar legitimately arrives one frame after its own close.
    from axon.parity.monitor import DEFAULT_SILENCE_AFTER_NS
    from axon.strategies.shadow import SILENCE_AFTER_BARS, ShadowTrader

    assert SILENCE_AFTER_BARS >= 2
    with ShadowTrader(
        strategy(ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
    ) as trader:
        deadline = trader.monitor.config.silence_after_ns
    assert deadline == int(SILENCE_AFTER_BARS * BAR_NS)
    assert deadline > DEFAULT_SILENCE_AFTER_NS  # an hourly bar dwarfs the generic default


def test_a_shadow_run_reports_drift_as_unmeasured_rather_than_as_stable(tmp_path):
    # ADR-0030 measured the reason this is not on by default: over `perp_bar`'s own
    # history, with feature parity green on every window, PSI trips the conventional
    # 0.25 band on 18 of 20 windows and peaks at 5.97. A shadow monitor alarming on
    # that would alarm forever, and an operator who learns to ignore the drift line
    # has learned to ignore the parity line — which has no false positives at all.
    report = shadow_run(fixture_candles("BTC", INTERVAL), tmp_path)
    assert report.drift_measured is False
    assert "drift: not measured" in report.summary()
    assert "stable" not in report.summary()


def test_a_would_be_signal_crossed_a_real_ring_and_is_a_target_rather_than_an_order(tmp_path):
    # What `perp_bar` emits is a target position. Turning one into an order needs a
    # book to price against, a tick/lot grid to quantize onto and knowledge of what is
    # resting — none of which a shadow run has. It is read back *off the ring* rather
    # than out of the context, because crossing the ring is the part that can fail.
    report = shadow_run(fixture_candles("BTC", INTERVAL), tmp_path, ConstantPredictor(0.9))

    assert len(report.signals) == 1
    signal = report.signals[0]
    assert signal.target_qty == Decimal("0.01")
    assert signal.urgency == 0 and signal.ttl_ms == 60_000
    assert signal.ts_event in set(int(t) for t in fixture_candles("BTC", INTERVAL).ts_event)
    # One side of one position opened, priced at the published maker schedule as a
    # hurdle — a count of what was emitted, never a P&L. Nothing was filled.
    assert report.turnover.sides == Decimal(1)
    assert report.turnover.maker_bps == pytest.approx(1.5)
    assert "not a P&L" in report.turnover.describe()


def test_the_registered_artifact_shadow_trades_and_the_diff_still_reads_zero(ladder, tmp_path):
    # The whole rung with the real model in the loop: the artifact is loaded from the
    # registry through the spec check, driven bar by bar, and its served rows are
    # diffed against the recompute as the run goes. Bounded to the tail of the history
    # because a booster reloaded from its artifact costs milliseconds per single-row
    # prediction, not microseconds.
    from axon.strategies.shadow import shadow_history

    _, registry = ladder
    candles = fixture_candles("BTC", INTERVAL)[-200:]
    served = PerpBar.from_registry(registry, "perp_bar_xgb", PerpBarParams(symbol_id=SYMBOL))
    report = shadow_history(
        served, candles, symbol_id=SYMBOL, ring_path=str(tmp_path / "shadow.ring")
    )
    report.raise_for_status()

    assert report.monitor.max_abs_diff == 0.0
    assert report.rows_compared == report.rows_owed > 0
    assert report.spec_ref == PERP_BAR_V1.ref
    # The rung says the serving path matches the research path. It says nothing about
    # the market, and ADR-0022 answers that separately: this model should not trade.
    assert report.venue_attested is False


def test_a_model_trained_on_other_features_is_refused_at_load():
    class Stub:
        def load_meta(self, registry_id, version=None):
            return ArtifactMeta(
                registry_id=registry_id,
                version=1,
                feature_spec_ref=PERP_CORE_V1.ref,  # a different recipe entirely
            )

        def load_predictor(self, registry_id, version=None):  # pragma: no cover - never reached
            raise AssertionError("the spec check must run before the model is loaded")

    with pytest.raises(ValueError, match="never fitted on"):
        PerpBar.from_registry(Stub(), "perp_bar_xgb", PerpBarParams(symbol_id=SYMBOL))


def test_the_spec_fingerprint_is_stable_so_an_artifact_keeps_resolving():
    # Pinned like PERP_CORE_V1's: if this moves, every artifact that recorded it is
    # refused at load, which is correct — and must therefore be deliberate.
    assert PERP_BAR_V1.ref == "perp_bar/v1#a21328ed1532ecd4"
    assert len(PERP_BAR_V1.columns) == 9
    assert PERP_BAR_V1.required_inputs == ("close", "high", "low", "volume")


def test_the_serving_buffer_is_longer_than_the_longest_window_in_the_spec():
    assert SERVING_BUFFER_BARS > 24 + 1


# ── the artifact and the three gates ─────────────────────────────────────────


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """One climb of the whole ladder on the committed fixture. Shared: it fits four
    models and replays two histories, and every gate test below asks about the same
    run rather than a differently-seeded one."""
    pytest.importorskip("xgboost")
    from axon.strategies.training import climb

    registry = ModelRegistry(tmp_path_factory.mktemp("registry"))
    # `parity_bars` bounds the online replay: a booster reloaded from its artifact
    # costs ~20 ms for a single-row prediction (it defaults to every core, and the
    # OMP fan-out dwarfs nine features of threshold traversal), so a full-history
    # replay is a minute of test time to re-prove what the bit-for-bit test above
    # proves in one second with a constant predictor.
    return climb(
        [fixture_candles(coin, INTERVAL) for coin in fixture_coins(INTERVAL)],
        registry,
        folds=2,
        parity_bars=150,
    ), registry


def test_the_registered_artifact_reproduces_the_model_exactly(ladder):
    # Trees are exact or the format is lossy: inference is deterministic threshold
    # traversal, so any difference at all means a float width changed. The
    # discretizer is the strategy's own entry band, so "no decision flipped" is a
    # statement about positions, not about a generic threshold.
    result, _ = ladder
    result.model_parity.raise_for_status()
    assert result.model_parity.eps == 0.0
    assert result.model_parity.max_abs_diff == 0.0
    assert result.model_parity.n_flips == 0
    assert result.model_parity.non_finite == 0
    assert result.artifact.meta.roundtrip_max_abs_diff == 0.0


def test_the_online_and_offline_features_agree_on_every_coin(ladder):
    result, _ = ladder
    assert set(result.feature_parity) == set(fixture_coins(INTERVAL))
    for report in result.feature_parity.values():
        report.raise_for_status()
        assert report.max_abs_diff == 0.0
        # And every row the replay owed is in the comparison. The gate is an
        # intersection: on its own, a zero says the rows both sides produced agree,
        # not that the serving path produced them.
        assert report.n_rows == report.coverage.n_in_scope == report.coverage.n_online


def test_the_artifact_records_the_feature_spec_it_was_trained_on(ladder):
    # A model without the features that fed it is not reproducible, and the load
    # path refuses a mismatch — so this string is the whole audit trail.
    result, registry = ladder
    assert result.artifact.meta.feature_spec_ref == PERP_BAR_V1.ref
    served = PerpBar.from_registry(registry, "perp_bar_xgb", PerpBarParams(symbol_id=SYMBOL))
    assert served.artifact_version == result.artifact.meta.version


def test_the_exported_model_is_the_last_folds_and_not_a_refit_on_everything(ladder):
    # Refitting on the full sample ships a model no reported number describes.
    from axon.strategies.training import fit

    result, _ = ladder
    fold = result.run.final_fold
    holdout = result.dataset.features[fold.test]
    same = fit(*result.dataset.rows(fold.train))
    np.testing.assert_array_equal(
        same.predict_proba(holdout.astype(np.float32))[:, 1],
        result.run.model.predict_proba(holdout.astype(np.float32))[:, 1],
    )
    everything = fit(result.dataset.features, result.dataset.label)
    assert not np.array_equal(
        everything.predict_proba(holdout.astype(np.float32))[:, 1],
        result.run.model.predict_proba(holdout.astype(np.float32))[:, 1],
    )


def test_every_scored_row_is_out_of_sample_and_nothing_else_is(ladder):
    result, _ = ladder
    scored = np.flatnonzero(np.isfinite(result.run.oos_scores))
    expected = np.concatenate([f.test for f in result.run.folds])
    np.testing.assert_array_equal(scored, np.sort(expected))


def test_drift_reports_a_feature_that_leaves_the_range_it_was_fitted_on(ladder):
    # The gate has to be able to fire on this dataset, not only on a synthetic one.
    from axon.parity import drift_report
    from axon.strategies.training import training_binnings

    result, _ = ladder
    fold = result.run.final_fold
    binnings = training_binnings(result.dataset, fold)
    moved = result.dataset.features[fold.test].copy()
    column = result.dataset.columns.index("vol_24")
    moved[:, column] += 10.0 * np.nanstd(result.dataset.features[fold.train][:, column])

    report = drift_report(
        result.dataset.features[fold.train],
        moved,
        columns=result.dataset.columns,
        binnings=binnings,
    )
    assert report.ranked()[0].name == "vol_24"
    assert report.ranked()[0].band == "significant"
    assert not report.passed


def test_the_reported_numbers_are_a_result_and_not_an_assertion(ladder):
    # `climb` returns reports rather than raising on them: a gate that fails is a
    # result (ADR-0016), and a pipeline that raised would be one whose failures were
    # never read properly. AUC is asserted to exist and to be finite — deliberately
    # not asserted to be good, because it is not.
    result, _ = ladder
    assert np.isfinite(result.run.evaluation.auc)
    assert 0.0 <= result.run.evaluation.coverage <= 1.0
    assert len(result.run.evaluation.folds) == 2


def test_the_benchmark_is_priced_on_the_rows_the_model_traded_and_not_on_its_short_rows(ladder):
    # The subtraction ADR-0022 published once and had to retract. On the rows where
    # the model is short, a constant short *is* the model — the same position on the
    # same rows — so the two edges are equal by construction, for every model, on
    # every sample. Differencing them against the model's pooled edge measures two
    # different row sets and hands a directional bias a selection number it never
    # earned. The benchmark is therefore priced over every row a position was taken on.
    from axon.strategies.training import decompose

    result, _ = ladder
    d = decompose(result.run)

    assert d.n == d.n_long + d.n_short
    assert d.n <= d.n_scored == result.run.evaluation.n_test
    assert d.edge_bps == pytest.approx(result.run.evaluation.gross_edge_bps)
    assert d.hit_rate == pytest.approx(result.run.evaluation.hit_rate)
    # A constant short over the decision rows is exactly the drift, negated: free, and
    # what the model has to beat before anything else about it is worth discussing.
    assert d.benchmark_edge_bps == pytest.approx(-d.drift_bps)
    assert d.selection_bps == pytest.approx(d.edge_bps - d.benchmark_edge_bps)

    # And the vacuous comparison, spelled out so it cannot be quoted as evidence again.
    scores = result.run.oos_scores
    band = result.run.evaluation.entry_edge
    shorts = np.isfinite(scores) & (scores <= 0.5 - band)
    always_short_on_short_rows = -np.mean(result.dataset.forward_return[shorts]) * 10_000.0
    assert always_short_on_short_rows == pytest.approx(d.short_edge_bps)
    assert d.n_short == int(shorts.sum())


def test_roc_auc_averages_ties_so_a_constant_model_scores_a_half():
    from axon.strategies.training import roc_auc

    labels = np.array([0.0, 1.0, 0.0, 1.0])
    assert roc_auc(labels, np.full(4, 0.5)) == pytest.approx(0.5)
    assert roc_auc(labels, np.array([0.1, 0.9, 0.2, 0.8])) == pytest.approx(1.0)
    assert np.isnan(roc_auc(np.ones(4), np.arange(4.0)))


# ── the offloaded job (planned, never submitted) ─────────────────────────────


@pytest.mark.parametrize("job", [hyperparameter_sweep(), walk_forward_job()])
def test_the_emitted_spec_uses_only_keys_hwsched_accepts(job):
    # JobSpec is `extra="forbid"`: one unknown key and the spec is rejected after
    # the YAML is written and the subprocess spawned.
    assert set(job.to_mapping()) <= JOBSPEC_FIELDS


def test_sweep_constants_ride_as_grid_axes_because_a_fan_out_drops_kwargs():
    # hwsched materializes tasks from `params` and never merges the spec's `kwargs`
    # into them. A constant passed the obvious way never reaches the function, and
    # the container raises TypeError after the image has been built and paid for.
    job = hyperparameter_sweep()
    assert {"data", "coins", "interval", "folds"} <= set(job.params)
    assert job.kwargs == {}
    assert job.n_tasks == 36


def test_the_job_names_the_schema_path_the_container_cannot_derive():
    for job in (hyperparameter_sweep(), walk_forward_job()):
        assert job.env[SCHEMA_ENV].startswith("/vol/")
        assert job.env[SCHEMA_ENV].endswith("schema.toml")


def test_axon_cannot_be_imported_from_a_bare_package_mount_without_the_schema_path(tmp_path):
    """Why every offloaded job must carry ``AXON_SCHEMA_PATH``.

    hwsched mounts the entrypoint's top-level package with
    ``add_local_python_source("axon")`` — the repository around it, including
    ``contracts/schema.toml``, is not in the image. ``axon.contracts`` resolves that
    file relative to the repo root, so the import fails before a row is read. This
    reproduces the container's view by copying only the package.
    """
    shutil.copytree(Path(__file__).resolve().parents[1] / "axon", tmp_path / "axon")
    env = {k: v for k, v in os.environ.items() if k != "AXON_SCHEMA_PATH"}
    env["PYTHONPATH"] = str(tmp_path)

    bare = subprocess.run(
        [sys.executable, "-c", "import axon.contracts"],
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert bare.returncode != 0
    assert b"schema" in bare.stderr.lower()

    env["AXON_SCHEMA_PATH"] = str(
        Path(__file__).resolve().parents[2] / "contracts" / "schema.toml"
    )
    fixed = subprocess.run(
        [sys.executable, "-c", "import axon.contracts"],
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert fixed.returncode == 0, fixed.stderr.decode()


def test_a_fan_out_writes_one_receipt_per_task_rather_than_one_file(tmp_path):
    # Thirty-six tasks pointed at one file is thirty-six tasks overwriting each
    # other, and the surviving result is whichever container finished last.
    pytest.importorskip("xgboost")
    from axon.strategies.remote import sweep_point

    data = tmp_path / "candles"
    for coin in fixture_coins(INTERVAL):
        fixture_candles(coin, INTERVAL).to_csv(data / f"{coin.lower()}-{INTERVAL}.csv")
    out = tmp_path / "sweep"

    for depth in (2, 3):
        receipt = sweep_point(
            depth,
            data=str(data),
            coins=",".join(fixture_coins(INTERVAL)),
            interval=INTERVAL,
            folds=2,
            inner_folds=2,
            artifact=str(out),
            n_estimators=20,
        )
        assert receipt["spec_ref"] == PERP_BAR_V1.ref
        assert receipt["n_validation_rows"] < receipt["n_rows"]  # the holdout is withheld
    written = sorted(p.name for p in out.glob("*.json"))
    assert len(written) == 2, written


def test_the_remote_entrypoint_says_which_file_is_missing_from_the_volume(tmp_path):
    from axon.strategies.remote import read_candles

    with pytest.raises(FileNotFoundError, match="mounted volume"):
        read_candles(str(tmp_path), "BTC", INTERVAL)


# ── against the real hwsched CLI: free, offline, nothing submitted ───────────


def _hwsched_client(tmp_path):
    """A client on the real checkout, ``fake`` provider, throwaway ledger.

    The CLI runs for real; nothing reaches Modal and the actual run store is
    neither read nor written, so this cannot spend or hide a dollar.
    """
    from axon.compute import HwschedClient, HwschedError
    from axon.compute.client import DEFAULT_HOME, HOME_ENV

    home = Path(os.environ.get(HOME_ENV) or DEFAULT_HOME)
    if not (home / "hwsched" / "__init__.py").is_file():
        pytest.skip("no hwsched checkout; set AXON_HWSCHED_HOME to run the CLI tests")
    client = HwschedClient(home=home, provider="fake", store_path=tmp_path / "runs.db")
    try:
        client.budget_state()
    except (HwschedError, OSError) as exc:
        pytest.skip(f"hwsched CLI not runnable here: {exc}")
    return client


def test_real_hwsched_sizes_the_sweep_as_cpu_work_inside_its_cap(tmp_path):
    outcome = _hwsched_client(tmp_path).plan(hyperparameter_sweep())

    assert outcome.exit_code == 0
    assert outcome.plan["device"] == "cpu"  # trees, not a GPU job
    assert outcome.plan["n_tasks"] == 36
    assert 0 < outcome.cost.low <= outcome.cost.expected <= outcome.cost.high
    assert outcome.cost.high < 1.00
    assert not outcome.errors


def test_the_walk_forward_job_survives_plan_approve_run_against_the_fake_provider(tmp_path):
    # The dry-run-before-spend protocol end to end (ADR-0017): a plan, a ceiling at
    # or above its high estimate, a single-use approval bound to the job's digest.
    client = _hwsched_client(tmp_path)
    job = walk_forward_job()
    plan = client.plan(job)
    outcome = client.run(job, approval=plan.approve(max_usd=max(0.05, plan.cost.high)))

    assert outcome.submitted and outcome.succeeded
    assert outcome.correlation_id == job.resolved_correlation_id
