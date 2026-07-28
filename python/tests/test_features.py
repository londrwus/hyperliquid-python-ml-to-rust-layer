"""``axon.features``: the transforms, and the spec that pins what a model was fed.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic — fixed seeds, no clock, no network, numpy only.

The load-bearing test in this file is
``test_extending_the_series_with_future_data_does_not_change_history``. A feature
library without it is how a backtest comes out beautiful and live comes out flat,
and it runs over the whole registry so a feature added next month cannot opt out.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from axon.contracts import (
    MD_KIND_QUOTE,
    MD_KIND_TRADE,
    MD_SLICE_DTYPE,
    new_md_slice,
    to_fixed,
)
from axon.features import (
    FEATURES_VERSION,
    PERP_CORE_V1,
    FeatureDef,
    FeatureError,
    FeatureSpec,
    FeatureSpecMismatch,
    UnknownFeature,
    bar_inputs,
    book_imbalance,
    close_location,
    ema,
    ema_crossover,
    feature_info,
    finite_rows,
    log_return,
    md_slice_inputs,
    mid_price,
    momentum,
    registered_features,
    relative_range,
    relative_spread,
    rolling_std,
    sma_crossover,
    spread,
    trade_flow_imbalance,
)

#: One parameter set per registered feature, so the sweeps below actually exercise a
#: window rather than a default. The set-equality test right after this keeps it
#: complete: a new feature that is not listed here fails the suite instead of
#: quietly skipping the point-in-time check.
SAMPLE_PARAMS = {
    "book_imbalance": {},
    "close_location": {},
    "ema": {"span": 5},
    "ema_crossover": {"fast": 3, "slow": 7},
    "log_return": {"period": 2},
    "mid_price": {},
    "momentum": {"window": 4},
    "realized_volatility": {"window": 5},
    "relative_range": {},
    "relative_spread": {},
    "rolling_mean": {"window": 5},
    "rolling_std": {"window": 5},
    "rolling_sum": {"window": 5},
    "rolling_zscore": {"window": 5},
    "sma_crossover": {"fast": 3, "slow": 7},
    "spread": {},
    "trade_flow_imbalance": {"window": 5},
}


def synthetic_inputs(n: int = 64, *, seed: int = 11) -> dict[str, np.ndarray]:
    """One plausible array per input name the registry knows about."""
    rng = np.random.default_rng(seed)
    mid = 60_000.0 + np.cumsum(rng.normal(0.0, 4.0, n))
    half = rng.uniform(0.5, 2.0, n)
    reach = rng.uniform(1.0, 20.0, n)
    return {
        "x": mid,
        "price": mid,
        "last_px": mid,
        "bid_px": mid - half,
        "ask_px": mid + half,
        "bid_sz": rng.uniform(0.1, 9.0, n),
        "ask_sz": rng.uniform(0.1, 9.0, n),
        "trade_sz": rng.uniform(0.0, 2.0, n),
        "trade_sign": rng.choice([-1.0, 0.0, 1.0], n),
        # A bar's own extremes, wide enough around its close that no window is
        # degenerate: the point-in-time sweep has to exercise the transform, not
        # trip over a NaN guard.
        "close": mid,
        "high": mid + reach,
        "low": mid - reach,
    }


def call(name: str, inputs: dict[str, np.ndarray]) -> np.ndarray:
    info = feature_info(name)
    return info.fn(*(inputs[i] for i in info.inputs), **SAMPLE_PARAMS[name])


def assert_prefix_stable(compute, inputs: dict[str, np.ndarray], *, k: int) -> None:
    """Assert the first ``k`` values do not move when future data is appended."""
    full = compute(inputs)
    prefix = compute({name: arr[:k] for name, arr in inputs.items()})
    assert_array_equal(full[:k], prefix)


# ── the three rules every transform obeys ─────────────────────────────────────


def test_every_registered_feature_appears_in_the_point_in_time_sweep():
    assert set(SAMPLE_PARAMS) == set(registered_features())


@pytest.mark.parametrize("name", sorted(SAMPLE_PARAMS))
def test_extending_the_series_with_future_data_does_not_change_history(name):
    # Lookahead leakage does not announce itself: the feature still has the right
    # shape, the model still trains, and the backtest is simply better than reality.
    # Recomputing on a prefix is the only check that catches it.
    inputs = synthetic_inputs()
    assert_prefix_stable(lambda d: call(name, d), inputs, k=40)


@pytest.mark.parametrize("name", sorted(SAMPLE_PARAMS))
def test_every_feature_is_length_preserving_so_rows_never_shift_against_their_event(name):
    inputs = synthetic_inputs(n=40)
    assert call(name, inputs).shape == (40,)


@pytest.mark.parametrize("name", sorted(SAMPLE_PARAMS))
def test_warmup_is_nan_rather_than_a_plausible_zero(name):
    # Zero is a legal value for every feature here, so a zero-filled warmup is
    # indistinguishable from a real reading and the model learns from it.
    if not SAMPLE_PARAMS[name]:
        pytest.skip(f"{name} has no window to warm up")
    assert np.isnan(call(name, synthetic_inputs())[0])


def test_the_lookahead_check_fails_on_a_deliberately_leaky_transform():
    # The negative control: a gate that cannot fail is not a gate. A centred moving
    # average is the classic leak — it looks like a smoother and reads the future.
    def centered_mean(inputs, *, window=5):
        a = inputs["x"]
        out = np.full(a.size, np.nan)
        half = window // 2
        for i in range(half, a.size - half):
            out[i] = a[i - half : i + half + 1].mean()
        return out

    with pytest.raises(AssertionError):
        assert_prefix_stable(centered_mean, synthetic_inputs(), k=40)


# ── the transforms themselves ─────────────────────────────────────────────────


def test_log_return_matches_the_hand_computed_ratio():
    prices = np.array([100.0, 101.0, 99.0, 99.0])
    out = log_return(prices)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(np.log(101.0 / 100.0))
    assert out[2] == pytest.approx(np.log(99.0 / 101.0))
    assert out[3] == 0.0


def test_a_zero_price_becomes_nan_instead_of_negative_infinity():
    # A gap in the feed arrives as a zero on the wire. log(0) is -inf, and an -inf
    # feature is a position sized by a missing packet.
    out = log_return(np.array([100.0, 0.0, 100.0]))
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert np.isfinite(out[1:]).sum() == 0


def test_momentum_is_the_same_function_as_a_longer_log_return():
    prices = synthetic_inputs()["price"]
    assert_array_equal(momentum(prices, window=8), log_return(prices, period=8))


def test_rolling_std_uses_only_the_window_and_not_the_whole_history():
    x = np.concatenate([np.full(10, 1000.0), np.array([1.0, 2.0, 3.0, 4.0])])
    out = rolling_std(x, window=4)
    # The last window is [1,2,3,4]; the 1000s before it must not be in the answer.
    assert out[-1] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0]))
    assert out[9] == pytest.approx(0.0)


def test_ema_follows_the_recursion_and_not_a_closed_form():
    x = synthetic_inputs(n=32)["x"]
    span = 5
    alpha = 2.0 / (span + 1.0)
    level = x[0]
    expected = [level]
    for v in x[1:]:
        level = level + alpha * (v - level)
        expected.append(level)
    got = ema(x, span=span)
    assert_array_equal(got[span - 1 :], np.array(expected)[span - 1 :])


def test_an_ema_does_not_emit_its_own_seed_as_a_feature():
    # Before `span` observations the level is mostly "the first price we saw",
    # which is a number about the start of the session, not about the market.
    out = ema(np.arange(10.0), span=4)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()


def test_a_single_bad_tick_does_not_poison_the_ema_for_the_rest_of_the_session():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    out = ema(x, span=2)
    assert np.isnan(out[2])  # the bad observation itself is unknown, not invented
    assert np.isfinite(out[3:]).all()  # …and the level survives it


def test_a_missing_side_of_the_book_does_not_become_half_a_price():
    # An empty side is a zero on the wire, and (0 + ask)/2 is a price several
    # percent from anything tradable — which then feeds a return.
    out = mid_price(np.array([0.0, 99.0]), np.array([101.0, 101.0]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(100.0)


def test_a_crossed_book_reports_a_negative_spread_instead_of_looking_tight():
    out = spread(np.array([101.0]), np.array([100.0]))
    assert out[0] == pytest.approx(-1.0)


def test_relative_spread_is_comparable_across_price_levels():
    cheap = relative_spread(np.array([99.95]), np.array([100.05]))
    dear = relative_spread(np.array([59_970.0]), np.array([60_030.0]))
    assert cheap[0] == pytest.approx(10.0)  # 0.10 wide on a 100 mid = 10 bps
    assert dear[0] == pytest.approx(10.0)  # 60 wide on a 60,000 mid = the same


def test_book_imbalance_is_bounded_and_an_empty_book_is_nan():
    out = book_imbalance(np.array([3.0, 0.0, 1.0]), np.array([1.0, 0.0, 3.0]))
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])  # "no book" is not "balanced"
    assert out[2] == pytest.approx(-0.5)
    assert np.nanmax(np.abs(out)) <= 1.0


def test_trade_flow_imbalance_is_normalized_by_volume_and_a_quiet_window_is_nan():
    sizes = np.array([1.0, 3.0, 0.0, 0.0])
    signs = np.array([1.0, 1.0, 0.0, 0.0])
    out = trade_flow_imbalance(sizes, signs, window=2)
    assert out[1] == pytest.approx(1.0)  # both prints were buys
    assert out[2] == pytest.approx(1.0)  # window still holds the 3-lot buy
    assert np.isnan(out[3])  # no volume at all: not "balanced flow"


def test_relative_range_is_comparable_across_price_levels():
    cheap = relative_range(np.array([100.5]), np.array([99.5]), np.array([100.0]))
    dear = relative_range(np.array([60_300.0]), np.array([59_700.0]), np.array([60_000.0]))
    assert cheap[0] == pytest.approx(100.0)  # 1.0 wide on a 100 close = 100 bps
    assert dear[0] == pytest.approx(100.0)  # 600 wide on a 60,000 close = the same


def test_a_corrupt_bar_is_nan_rather_than_a_negative_range():
    # A high below its low is feed corruption, not a calm bar, and a negative width
    # would sit at the quiet end of the same distribution the model reads.
    out = relative_range(np.array([99.0, 101.0]), np.array([100.0, 99.0]), np.array([100.0, 0.0]))
    assert np.isnan(out[0])  # high < low
    assert np.isnan(out[1])  # a zero close is not a price


def test_close_location_is_bounded_and_a_flat_bar_is_nan():
    high = np.array([10.0, 10.0, 10.0, 10.0])
    low = np.array([0.0, 0.0, 0.0, 10.0])
    close = np.array([10.0, 0.0, 5.0, 10.0])
    out = close_location(high, low, close)
    assert out[0] == pytest.approx(1.0)  # closed on its high
    assert out[1] == pytest.approx(-1.0)  # gave the whole bar back
    assert out[2] == pytest.approx(0.0)
    assert np.isnan(out[3])  # no range at all is not "closed dead centre"
    assert np.nanmax(np.abs(out)) <= 1.0


def test_a_finite_window_crossover_survives_a_bounded_serving_buffer_where_an_ema_does_not():
    # The reason PERP_BAR_V1 has no EMA. The serving path holds the last `slow`
    # observations; research sees the whole series. A windowed mean forgets exactly
    # that far back and the two agree bit for bit; an EMA never forgets its seed, so
    # the same buffer gives a different number — largest right after a restart.
    inputs = synthetic_inputs(n=200)
    price, slow = inputs["price"], 7
    full_sma = sma_crossover(price, fast=3, slow=slow)
    buffered_sma = sma_crossover(price[-slow:], fast=3, slow=slow)
    assert buffered_sma[-1] == full_sma[-1]

    full_ema = ema_crossover(price, fast=3, slow=slow)
    buffered_ema = ema_crossover(price[-slow:], fast=3, slow=slow)
    assert buffered_ema[-1] != full_ema[-1]


def test_finite_rows_marks_the_warmup_rather_than_a_hand_counted_offset():
    matrix = PERP_CORE_V1.compute(spec_inputs(n=80))
    usable = finite_rows(matrix)
    assert not usable[:32].any()  # the 32-sample windows are still warming
    assert usable[-1]
    assert np.isfinite(matrix[usable]).all()


# ── the spec ──────────────────────────────────────────────────────────────────


def spec_inputs(n: int = 80, *, seed: int = 3) -> dict[str, np.ndarray]:
    all_inputs = synthetic_inputs(n, seed=seed)
    return {name: all_inputs[name] for name in PERP_CORE_V1.required_inputs}


def test_the_matrix_columns_follow_spec_order():
    # Permuting two columns leaves every name correct and every prediction wrong,
    # which is why order is part of the spec's identity rather than a convenience.
    matrix = PERP_CORE_V1.compute(spec_inputs())
    assert PERP_CORE_V1.columns[0] == "mid"
    assert matrix.shape == (80, len(PERP_CORE_V1.columns))
    inputs = spec_inputs()
    assert_array_equal(matrix[:, 0], mid_price(inputs["bid_px"], inputs["ask_px"]))


def test_a_column_can_read_an_earlier_column_instead_of_recomputing_it():
    inputs = spec_inputs()
    matrix = PERP_CORE_V1.compute(inputs)
    mid = matrix[:, PERP_CORE_V1.columns.index("mid")]
    ret = matrix[:, PERP_CORE_V1.columns.index("ret_1")]
    assert_array_equal(ret, log_return(mid, period=1))


def test_the_whole_spec_is_point_in_time_correct():
    assert_prefix_stable(PERP_CORE_V1.compute, spec_inputs(n=120), k=90)


def test_the_reference_spec_fingerprint_is_stable_across_processes():
    # Pinned deliberately. If this literal has to change, a feature's meaning or the
    # spec's contents changed, and the fix is to bump PERP_CORE_V1's version (or
    # FEATURES_VERSION) — not to edit the constant. Any artifact already carrying the
    # old id was trained on different numbers.
    assert PERP_CORE_V1.fingerprint == "868d3dbe95d4b386"
    assert PERP_CORE_V1.ref == "perp_core/v1#868d3dbe95d4b386"


def test_reordering_columns_changes_the_fingerprint():
    reordered = FeatureSpec(
        name=PERP_CORE_V1.name,
        version=PERP_CORE_V1.version,
        features=(PERP_CORE_V1.features[1], PERP_CORE_V1.features[0]) + PERP_CORE_V1.features[2:],
    )
    assert reordered.fingerprint != PERP_CORE_V1.fingerprint


def test_changing_a_parameter_changes_the_fingerprint():
    a = FeatureSpec("s", 1, (FeatureDef("v", "rolling_std", params={"window": 32}),))
    b = FeatureSpec("s", 1, (FeatureDef("v", "rolling_std", params={"window": 33}),))
    assert a.fingerprint != b.fingerprint


def test_a_round_tripped_spec_keeps_its_fingerprint_and_its_matrix():
    restored = FeatureSpec.from_json(PERP_CORE_V1.to_json())
    assert restored.fingerprint == PERP_CORE_V1.fingerprint
    assert restored.columns == PERP_CORE_V1.columns
    inputs = spec_inputs()
    assert_array_equal(restored.compute(inputs), PERP_CORE_V1.compute(inputs))


def test_parameter_order_does_not_change_the_identity():
    # The recipe is the same recipe however the training script spelled it; if dict
    # order leaked into the hash, re-exporting an unchanged model would mint a new id.
    a = FeatureDef("v", "ema_crossover", params={"fast": 3, "slow": 9})
    b = FeatureDef("v", "ema_crossover", params={"slow": 9, "fast": 3})
    assert FeatureSpec("s", 1, (a,)).fingerprint == FeatureSpec("s", 1, (b,)).fingerprint


def test_a_tampered_payload_is_refused():
    payload = json.loads(PERP_CORE_V1.to_json())
    payload["features"][5]["params"]["window"] = 64
    with pytest.raises(FeatureSpecMismatch, match="fingerprint"):
        FeatureSpec.from_dict(payload)


def test_a_spec_written_against_another_feature_library_is_refused():
    # The fingerprint pins the recipe; FEATURES_VERSION pins the kitchen. Without the
    # second check, rewriting rolling_std would leave every artifact id unchanged and
    # every model quietly fed different numbers.
    future = FeatureSpec(
        name="perp_core",
        version=1,
        features=PERP_CORE_V1.features,
        library_version=FEATURES_VERSION + 1,
    )
    with pytest.raises(FeatureSpecMismatch, match="axon.features"):
        FeatureSpec.from_dict(future.to_dict())
    # …and it is still inspectable, so tooling can report what the artifact wanted.
    assert FeatureSpec.from_dict(future.to_dict(), strict=False).columns == PERP_CORE_V1.columns


def test_a_misspelled_parameter_is_refused_rather_than_ignored():
    # Ignoring it means the spec claims a 32-sample volatility and ships the default.
    with pytest.raises(FeatureError, match="windwo"):
        FeatureDef("v", "rolling_std", params={"windwo": 32})


def test_an_unknown_feature_names_what_is_registered():
    with pytest.raises(UnknownFeature, match="rolling_median"):
        FeatureDef("v", "rolling_median")


def test_a_non_serializable_parameter_is_refused_at_construction():
    with pytest.raises(FeatureError, match="hashes and serializes"):
        FeatureDef("v", "rolling_std", params={"window": [32]})


def test_a_numpy_scalar_parameter_is_normalized_instead_of_hashing_differently():
    a = FeatureSpec("s", 1, (FeatureDef("v", "rolling_std", params={"window": np.int64(32)}),))
    b = FeatureSpec("s", 1, (FeatureDef("v", "rolling_std", params={"window": 32}),))
    assert a.fingerprint == b.fingerprint


def test_duplicate_columns_are_refused():
    with pytest.raises(FeatureError, match="duplicate"):
        FeatureSpec("s", 1, (FeatureDef("v", "mid_price"), FeatureDef("v", "spread")))


def test_a_column_that_shadows_an_input_is_refused():
    # Otherwise "which array did this feature read?" depends on evaluation order, and
    # the answer changes the day someone inserts a column above it.
    spec = FeatureSpec("s", 1, (FeatureDef("bid_px", "mid_price"),))
    with pytest.raises(FeatureError, match="collide"):
        spec.compute(spec_inputs())


def test_ragged_inputs_are_refused_rather_than_broadcast():
    inputs = spec_inputs()
    inputs["ask_px"] = inputs["ask_px"][:-1]
    with pytest.raises(FeatureError, match="not the same events"):
        PERP_CORE_V1.compute(inputs)


def test_an_unbound_input_says_what_it_was_looking_for():
    inputs = spec_inputs()
    del inputs["trade_sz"]
    with pytest.raises(FeatureError, match="trade_sz"):
        PERP_CORE_V1.compute(inputs)


def test_nanosecond_timestamps_are_refused_as_a_feature_input():
    # float64 holds 53 mantissa bits; a 2026 nanosecond stamp needs 61. Passed
    # through the matrix it rounds into ~256 ns buckets and reorders events.
    inputs = spec_inputs()
    inputs["ts_event"] = np.full(80, 1_700_000_000_123_456_789.0)
    with pytest.raises(FeatureError, match="2\\^53"):
        PERP_CORE_V1.compute(inputs)


def test_required_inputs_lists_only_what_the_caller_must_supply():
    # `mid` is produced by the spec, not supplied to it.
    assert "mid" not in PERP_CORE_V1.required_inputs
    assert set(PERP_CORE_V1.required_inputs) == {
        "ask_px",
        "ask_sz",
        "bid_px",
        "bid_sz",
        "trade_sign",
        "trade_sz",
    }


# ── the wire → inputs adapter ─────────────────────────────────────────────────


def md_batch(rows) -> np.ndarray:
    batch = np.zeros(len(rows), dtype=MD_SLICE_DTYPE)
    for i, kwargs in enumerate(rows):
        batch[i] = new_md_slice(**kwargs)[()]
    return batch


def quote(ts: int, **overrides):
    row = {
        "ts_event": ts,
        "bid_px": to_fixed(99.5),
        "bid_sz": to_fixed(2.0),
        "ask_px": to_fixed(100.5),
        "ask_sz": to_fixed(3.0),
        "kind": MD_KIND_QUOTE,
    }
    row.update(overrides)
    return row


def test_the_wire_fixed_point_is_scaled_exactly_once():
    inputs, _ = md_slice_inputs(md_batch([quote(1)]))
    assert inputs["bid_px"][0] == pytest.approx(99.5)
    assert inputs["ask_sz"][0] == pytest.approx(3.0)


def test_event_time_comes_back_as_int64_and_never_through_the_matrix():
    ts = 1_700_000_000_123_456_789
    inputs, ts_event = md_slice_inputs(md_batch([quote(ts)]))
    assert ts_event.dtype == np.int64
    assert int(ts_event[0]) == ts  # exact — a float64 round trip would not be
    assert "ts_event" not in inputs


def test_a_repeated_last_trade_is_not_counted_once_per_quote():
    # Every slice carries the last print whatever caused the update. Summing it on
    # each quote turns one 5-lot buy into a wall of one-sided flow.
    trade = quote(
        2,
        kind=MD_KIND_TRADE,
        last_trade_px=to_fixed(100.0),
        last_trade_sz=to_fixed(5.0),
        last_trade_ts=2,
    )
    echo = dict(trade, ts_event=3, kind=MD_KIND_QUOTE)
    inputs, _ = md_slice_inputs(md_batch([quote(1), trade, echo]))
    assert list(inputs["trade_sz"]) == [0.0, 5.0, 0.0]
    assert list(inputs["trade_sign"]) == [0.0, 1.0, 0.0]


def test_the_aggressor_flag_becomes_the_sign_of_the_flow():
    sell = quote(
        1,
        kind=MD_KIND_TRADE,
        last_trade_px=to_fixed(100.0),
        last_trade_sz=to_fixed(2.0),
        last_trade_sell=True,
    )
    inputs, _ = md_slice_inputs(md_batch([sell]))
    assert inputs["trade_sign"][0] == -1.0


def test_no_print_yet_is_nan_rather_than_a_zero_price():
    inputs, _ = md_slice_inputs(md_batch([quote(1)]))
    assert np.isnan(inputs["last_px"][0])


def test_an_out_of_order_batch_is_refused_rather_than_silently_recomputed():
    # Features are defined over the order the core observed. Recomputing them over a
    # reshuffled batch produces different numbers, and the parity gate would then
    # blame the transforms for a late-event bug in the capture.
    with pytest.raises(FeatureError, match="backwards"):
        md_slice_inputs(md_batch([quote(10), quote(9)]))
    inputs, _ = md_slice_inputs(md_batch([quote(10), quote(9)]), require_monotonic=False)
    assert inputs["bid_px"].size == 2


def test_a_foreign_dtype_is_refused_instead_of_decoded_as_prices():
    with pytest.raises(FeatureError, match="MdSlice"):
        md_slice_inputs(np.zeros(4, dtype=np.float64))


def test_the_ring_batch_feeds_the_reference_spec_end_to_end():
    rng = np.random.default_rng(5)
    rows = []
    mid = 60_000.0
    for i in range(200):
        mid += rng.normal(0.0, 3.0)
        is_trade = bool(rng.integers(0, 2))
        rows.append(
            quote(
                1_700_000_000_000_000_000 + i * 1_000_000,
                bid_px=to_fixed(mid - 0.5),
                ask_px=to_fixed(mid + 0.5),
                bid_sz=to_fixed(float(rng.uniform(0.1, 5.0))),
                ask_sz=to_fixed(float(rng.uniform(0.1, 5.0))),
                kind=MD_KIND_TRADE if is_trade else MD_KIND_QUOTE,
                last_trade_px=to_fixed(mid),
                last_trade_sz=to_fixed(float(rng.uniform(0.1, 2.0))),
                last_trade_sell=bool(rng.integers(0, 2)),
            )
        )
    inputs, ts_event = md_slice_inputs(md_batch(rows))
    matrix = PERP_CORE_V1.compute(inputs)
    assert matrix.shape == (200, len(PERP_CORE_V1.columns))
    assert ts_event.size == matrix.shape[0]
    assert finite_rows(matrix)[-1]


# ── the bar → inputs adapter ──────────────────────────────────────────────────


def bars(rows) -> tuple[np.ndarray, ...]:
    """``(open, high, low, close, volume)`` as fixed-point columns."""
    return tuple(
        np.array([to_fixed(row[j]) for row in rows], dtype=np.int64) for j in range(5)
    )


def test_the_bar_fixed_point_is_scaled_exactly_once():
    inputs = bar_inputs(*bars([(99.0, 101.0, 98.0, 100.0, 12.5)]))
    assert inputs["close"][0] == pytest.approx(100.0)
    assert inputs["volume"][0] == pytest.approx(12.5)


def test_a_float_bar_input_is_refused_rather_than_scaled_twice():
    # A caller who already divided by the scale would get prices at 1e-8 of
    # themselves — which still trains, still backtests, and is wrong by eight
    # orders of magnitude.
    columns = list(bars([(99.0, 101.0, 98.0, 100.0, 12.5)]))
    columns[3] = columns[3].astype(np.float64)
    with pytest.raises(FeatureError, match="fixed-point"):
        bar_inputs(*columns)


def test_a_zero_price_bar_is_nan_but_a_zero_volume_bar_is_zero():
    # Zero is what an unfilled bar looks like on the wire, and a zero close feeds a
    # -100% return into the first return feature that touches it. A bar in which
    # nothing traded, though, really did trade nothing.
    inputs = bar_inputs(*bars([(0.0, 0.0, 0.0, 0.0, 0.0), (99.0, 101.0, 98.0, 100.0, 0.0)]))
    assert np.isnan(inputs["close"][0])
    assert inputs["volume"][0] == 0.0
    assert inputs["volume"][1] == 0.0


def test_ragged_bar_columns_are_refused_rather_than_broadcast():
    o, h, low, c, v = bars([(99.0, 101.0, 98.0, 100.0, 1.0), (99.0, 101.0, 98.0, 100.0, 1.0)])
    with pytest.raises(FeatureError, match="same bars"):
        bar_inputs(o, h, low, c, v[:1])
