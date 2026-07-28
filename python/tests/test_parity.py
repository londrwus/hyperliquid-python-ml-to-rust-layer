"""``axon.parity``: the three gates from ``docs/03`` and ``docs/07``.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic — fixed seeds, no clock, no network, numpy only.

Two tests carry the weight of the model gate and belong together:
``test_a_tiny_delta_that_does_not_cross_a_threshold_passes`` and
``test_a_tiny_delta_that_crosses_a_threshold_fails_and_names_the_input``. Same
magnitude of numeric change, opposite verdicts — because the size of a logit delta
says nothing about whether it moved money. A gate that cannot fail is not a gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from axon.features import PERP_CORE_V1
from axon.parity import (
    FEATURE_ATOL,
    NN_EPS,
    PSI_MODERATE,
    PSI_STABLE,
    ParityError,
    TREE_EPS,
    align_by_event_time,
    aligned_feature_parity,
    decision_invariant,
    drift_report,
    feature_parity,
    kl_divergence,
    model_parity,
    psi,
    psi_band,
    quantile_binning,
    threshold_discretizer,
)

#: The discretizer under test throughout: a score below 0.45 is a short, above 0.55
#: a long, and anything between is flat.
DECIDE = threshold_discretizer(long_at=0.55, short_at=0.45)


def scores(n: int = 500, *, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


# ── model parity: the numeric criterion ───────────────────────────────────────


def test_identical_predictions_pass_the_exact_tree_gate():
    ref = scores()
    report = model_parity(ref, ref.copy(), discretizer=DECIDE, eps=TREE_EPS)
    assert report.passed
    assert report.max_abs_diff == 0.0
    assert "PASS" in report.summary()
    report.raise_for_status()


def test_a_tree_off_by_one_ulp_fails_the_exact_gate():
    # Tree inference is deterministic threshold traversal; a single ulp of drift
    # means the thresholds were cast to a different float width somewhere.
    ref = scores(64)
    cand = ref.copy()
    cand[7] = np.nextafter(cand[7], 1.0)
    report = model_parity(ref, cand, discretizer=DECIDE, eps=TREE_EPS)
    assert not report.passed
    assert report.max_abs_diff_index == 7


def test_a_neural_net_within_epsilon_passes():
    ref = scores()
    cand = ref + np.full(ref.shape, 1e-7)
    report = model_parity(ref, cand, discretizer=DECIDE, eps=NN_EPS)
    assert report.passed, report.summary()
    assert report.max_abs_diff < NN_EPS


def test_a_neural_net_beyond_epsilon_fails_and_names_the_worst_input():
    ref = scores(64)
    cand = ref.copy()
    cand[31] += 1e-3
    report = model_parity(ref, cand, discretizer=DECIDE, eps=NN_EPS)
    assert not report.passed
    assert report.max_abs_diff == pytest.approx(1e-3)
    assert report.max_abs_diff_index == 31
    with pytest.raises(ParityError, match="worst divergence at input 31"):
        report.raise_for_status()


# ── model parity: decision invariance, which is the one that protects P&L ─────


def test_a_tiny_delta_that_does_not_cross_a_threshold_passes():
    # A logit that wobbles in the middle of the flat band changes no position, and a
    # gate that fails here would be un-shippable noise.
    ref = np.array([0.10, 0.50, 0.90])
    cand = ref + 1e-6
    report = model_parity(ref, cand, discretizer=DECIDE, eps=NN_EPS)
    assert report.passed, report.summary()
    assert report.n_flips == 0


def test_a_tiny_delta_that_crosses_a_threshold_fails_and_names_the_input():
    # The same 1e-6, sat on the long threshold: flat becomes long, an order goes out,
    # and the numeric criterion alone would have signed off — max_abs_diff is 1000x
    # under eps. This asymmetry is the entire reason decision invariance exists.
    ref = np.array([0.10, 0.549_999_5, 0.90])
    cand = ref + 1e-6
    report = model_parity(ref, cand, discretizer=DECIDE, eps=NN_EPS)
    assert not report.passed
    assert report.max_abs_diff < report.eps  # the tolerance check would have passed
    assert report.n_flips == 1
    flip = report.flips[0]
    assert flip.index == 1
    assert (flip.reference_decision, flip.candidate_decision) == (0, 1)
    assert "flipped decision" in report.summary()


def test_decision_invariant_is_the_boolean_form_of_the_flip_list():
    ref = np.array([0.10, 0.549_999_5, 0.90])
    assert decision_invariant(ref, ref + 1e-6, discretizer=DECIDE) is False
    assert decision_invariant(ref, ref - 1e-6, discretizer=DECIDE) is True


def test_the_flip_list_is_capped_but_the_count_is_not():
    # A systematically broken candidate flips everything; the first handful debug it,
    # and keeping all of them turns a monitoring alarm into a memory problem.
    ref = np.full(200, 0.60)
    cand = np.full(200, 0.40)
    report = model_parity(ref, cand, discretizer=DECIDE, eps=1.0)
    assert report.n_flips == 200
    assert len(report.flips) == 20
    assert "and 180 more flip(s)" in report.summary()


def test_a_multiclass_argmax_decision_reports_the_row_that_moved():
    ref = np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]])
    cand = ref.copy()
    cand[1] = [0.6, 0.4]  # class 1 → class 0
    report = model_parity(ref, cand, discretizer=lambda s: np.argmax(s, axis=1), eps=1.0)
    assert report.n_flips == 1
    assert report.flips[0].index == 1


def test_threshold_discretizer_refuses_inverted_thresholds():
    with pytest.raises(ValueError, match="short_at < long_at"):
        threshold_discretizer(long_at=0.4, short_at=0.6)


# ── model parity: the ways a broken comparison looks like a pass ──────────────


def test_a_nan_prediction_fails_instead_of_comparing_false_against_epsilon():
    # `nan > 1e-5` is False, so a naive tolerance check calls a model that produced
    # NaN a perfect match. Non-finite predictions are counted, never compared.
    ref = scores(32)
    cand = ref.copy()
    cand[5] = np.nan
    report = model_parity(ref, cand, discretizer=DECIDE, eps=NN_EPS)
    assert not report.passed
    assert report.non_finite == 1
    assert "not finite" in report.summary()


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="same shape"):
        model_parity(scores(10), scores(9), discretizer=DECIDE)


def test_an_empty_comparison_is_refused_because_it_proves_nothing():
    with pytest.raises(ValueError, match="zero predictions"):
        model_parity(np.array([]), np.array([]), discretizer=DECIDE)


def test_a_discretizer_that_does_not_return_one_decision_per_input_is_refused():
    with pytest.raises(ValueError, match="decisions for"):
        model_parity(scores(10), scores(10), discretizer=lambda s: np.zeros(3), eps=1.0)


# ── feature parity ────────────────────────────────────────────────────────────


def matrices(n: int = 40, k: int = 3, *, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    offline = rng.normal(0.0, 1.0, (n, k))
    offline[:5, 1] = np.nan  # a warmup window, present in both paths
    return offline.copy(), offline


COLS = ("ret_1", "vol_32", "book_imb")


def test_matching_feature_matrices_pass():
    online, offline = matrices()
    report = feature_parity(online, offline, columns=COLS)
    assert report.passed, report.summary()
    report.raise_for_status()


def test_one_bad_cell_reports_its_row_and_column():
    # "The feature vectors differ" is not a debuggable statement about a matrix with
    # 40 columns and 100k rows.
    online, offline = matrices()
    online[17, 2] += 1e-6
    report = feature_parity(online, offline, columns=COLS)
    assert not report.passed
    assert report.n_mismatched == 1
    cell = report.mismatches[0]
    assert (cell.row, cell.column) == (17, "book_imb")
    assert cell.abs_diff == pytest.approx(1e-6)
    with pytest.raises(ParityError, match="row 17 col book_imb"):
        report.raise_for_status()


def test_a_wrongly_scaled_column_is_identified_as_the_worst_column():
    # The classic unit bug: bps against a fraction, or a size in lots against a size
    # in contracts. It breaks one column on every row, which is a different diagnosis
    # from one column being slightly worse on one row.
    online, offline = matrices()
    online[:, 0] = offline[:, 0] * 10_000.0
    online[3, 2] += 1e-3
    report = feature_parity(online, offline, columns=COLS)
    assert report.worst_column == "ret_1"
    assert report.per_column["ret_1"] == 40


def test_warmup_nan_on_both_sides_is_a_match():
    online, offline = matrices()
    assert np.isnan(offline[:5, 1]).all()
    assert feature_parity(online, offline, columns=COLS).passed


def test_nan_on_one_side_only_is_a_mismatch_and_not_silently_skipped():
    # A feature that goes NaN online and stays finite offline is the staleness bug
    # this gate exists to catch; `np.allclose` defaults would call it a match.
    online, offline = matrices()
    online[20, 0] = np.nan
    report = feature_parity(online, offline, columns=COLS)
    assert not report.passed
    assert report.n_nan_mismatched == 1
    assert report.mismatches[0].row == 20


def test_a_row_count_mismatch_is_refused_rather_than_broadcast():
    online, offline = matrices()
    with pytest.raises(ValueError, match="align the two sides on event time"):
        feature_parity(online[:-1], offline, columns=COLS)


def test_a_column_name_count_mismatch_is_refused():
    online, offline = matrices()
    with pytest.raises(ValueError, match="column names"):
        feature_parity(online, offline, columns=("a", "b"))


# ── feature parity: alignment on event time, not on row position ──────────────


def test_aligning_on_event_time_survives_a_dropped_online_sample():
    # One missing sample shifts every subsequent row. Compared positionally, a
    # perfectly correct feed reports 100% divergence.
    offline_ts = np.arange(10, dtype=np.int64) * 1_000_000 + 1_700_000_000_000_000_000
    online_ts = np.delete(offline_ts, 4)
    offline = np.arange(30, dtype=np.float64).reshape(10, 3)
    online = np.delete(offline, 4, axis=0)

    i_on, i_off = align_by_event_time(online_ts, offline_ts)
    assert i_on.size == 9
    assert_array_equal(offline_ts[i_off], online_ts[i_on])
    assert feature_parity(online[i_on], offline[i_off], columns=COLS).passed
    # …and the positional comparison it replaces would have been a wall of red.
    assert not feature_parity(online, offline[:9], columns=COLS).passed


def test_duplicate_event_times_are_refused_rather_than_matched_arbitrarily():
    ts = np.array([1, 2, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate event times"):
        align_by_event_time(ts, np.array([1, 2, 3, 4], dtype=np.int64))


# ── feature parity: the denominator the intersection used to hide ─────────────


def stamped(n: int = 10, *, step: int = 1_000_000) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * step + 1_700_000_000_000_000_000


def ramp(n: int = 10, k: int = 3) -> np.ndarray:
    return np.arange(n * k, dtype=np.float64).reshape(n, k)


def test_a_dropped_online_row_fails_the_gate_instead_of_leaving_the_denominator():
    # The blind spot this accounting exists to close. Every row the online path did
    # produce is exactly right, so every number a reader looks at reads healthy —
    # `max_abs_diff` is 0.0 here, which is the most reassuring value there is — and
    # the row that never arrived was compared against nothing at all. It is not a
    # mismatch, it is an absence, and an intersection cannot see one.
    offline_ts, offline = stamped(), ramp()
    online_ts, online = np.delete(offline_ts, 4), np.delete(offline, 4, axis=0)

    report = aligned_feature_parity(
        online, offline, online_ts=online_ts, offline_ts=offline_ts, columns=COLS
    )
    assert report.max_abs_diff == 0.0
    assert report.n_mismatched == 0 and report.n_nan_mismatched == 0
    assert not report.passed
    assert report.coverage.n_matched == 9 and report.coverage.n_offline_within == 1
    with pytest.raises(ParityError, match="not wide enough"):
        report.raise_for_status()


def test_an_online_path_that_stops_early_is_not_excused_as_out_of_span():
    # The other shape of the same absence, and the one an "inside the online side's
    # own span" rule would wave through on its own: a feed that produced perfect
    # rows and then died. Every remaining offline row is outside the online span,
    # which is exactly the excuse a cold start legitimately uses.
    offline_ts, offline = stamped(), ramp()
    report = aligned_feature_parity(
        offline[:5], offline, online_ts=offline_ts[:5], offline_ts=offline_ts, columns=COLS
    )
    assert not report.passed
    assert report.coverage.n_offline_after == 5
    assert "after the online side stopped" in report.summary()


def test_a_cold_started_online_side_is_held_only_to_the_rows_it_was_running_for():
    # The counter-case, and the reason the rule is not simply "every offline row".
    # A monitor window that opens mid-history, and a replay started cold, both leave
    # earlier offline rows unmatched — and a check that fired on those would fire on
    # every healthy run, which is how a real guard gets deleted rather than fixed.
    offline_ts, offline = stamped(), ramp()
    report = aligned_feature_parity(
        offline[5:], offline, online_ts=offline_ts[5:], offline_ts=offline_ts, columns=COLS
    )
    assert report.passed, report.summary()
    assert report.coverage.n_offline_before == 5
    assert report.coverage.n_in_scope == 5  # not 10: those five were never owed


def test_a_late_starting_online_side_is_excused_only_while_nobody_has_said_what_was_owed():
    # The residual the collapse onto `Coverage` exposed. Under the default the owed
    # span can only be inferred from the online side's own first stamp — and a path
    # that was blind through its opening rows produces exactly the same first stamp
    # as one that started on time, so the two are genuinely indistinguishable and
    # excusing them is the honest answer. A caller that trimmed the reference to the
    # owed rows has removed the ambiguity, and then the same data is a fault.
    offline_ts, offline = stamped(), ramp()
    late_ts, late = offline_ts[3:], offline[3:]

    observed = aligned_feature_parity(
        late, offline, online_ts=late_ts, offline_ts=offline_ts, columns=COLS
    )
    assert observed.passed
    assert observed.coverage.n_offline_before == 3

    declared = aligned_feature_parity(
        late, offline, online_ts=late_ts, offline_ts=offline_ts, columns=COLS, scope="declared"
    )
    assert not declared.passed
    assert declared.coverage.n_offline_before == 0
    assert declared.coverage.n_offline_within == 3
    assert declared.coverage.n_in_scope == 10  # every row given was owed


def test_a_reference_trimmed_too_narrowly_is_caught_by_the_rows_the_online_side_kept():
    # The guard that makes "declared" safe to be wrong. A caller that under-trims —
    # declaring fewer rows owed than really were — cannot quietly shrink the
    # denominator, because the online side produced rows at stamps the claim does not
    # contain and those land in `online_unmatched`. This is the direction a separate
    # `online_span=(lo, hi)` tuple could not guard at all: a narrow `lo` there simply
    # re-excuses the leading rows, which is the bug this mode exists to close.
    offline_ts, offline = stamped(), ramp()
    under_trimmed_ts, under_trimmed = offline_ts[4:], offline[4:]
    report = aligned_feature_parity(
        offline,
        under_trimmed,
        online_ts=offline_ts,
        offline_ts=under_trimmed_ts,
        columns=COLS,
        scope="declared",
    )
    assert not report.passed
    assert report.coverage.n_online_unmatched == 4


def test_a_reference_trimmed_too_widely_reddens_rather_than_being_absorbed():
    # The other direction, for completeness: rows declared owed that the online side
    # was never going to produce are a loud failure, not a quiet one.
    offline_ts, offline = stamped(), ramp()
    report = aligned_feature_parity(
        offline[:6], offline, online_ts=offline_ts[:6], offline_ts=offline_ts,
        columns=COLS, scope="declared",
    )
    assert not report.passed
    assert report.coverage.n_offline_within == 4


def test_an_unknown_scope_is_refused_rather_than_treated_as_the_lenient_one():
    # A typo must not silently select the mode that excuses rows.
    with pytest.raises(ValueError, match="scope must be"):
        align_by_event_time(stamped(), stamped(), scope="strict")


def test_an_online_row_the_reference_has_nothing_at_is_a_failure_and_not_a_bonus():
    # The offline recompute is the definition of correct, so a row the online path
    # produced at an event time the reference does not have means the two are
    # describing different events. Silently keeping the intersection would report a
    # perfect match over whatever happened to overlap.
    offline_ts, offline = stamped(), ramp()
    extra_ts = np.append(offline_ts, offline_ts[-1] + 500_000)
    extra = np.vstack([offline, offline[-1]])
    report = aligned_feature_parity(
        extra, offline, online_ts=extra_ts, offline_ts=offline_ts, columns=COLS
    )
    assert not report.passed
    assert report.coverage.n_online_unmatched == 1


def test_a_comparison_of_zero_rows_never_reports_a_pass():
    # "PASS, 0 rows compared" is this whole workstream's failure at its limit: every
    # tolerance satisfied because nothing was ever measured against one.
    offline_ts, offline = stamped(), ramp()
    report = aligned_feature_parity(
        offline[:0], offline, online_ts=offline_ts[:0], offline_ts=offline_ts, columns=COLS
    )
    assert report.n_rows == 0
    assert not report.passed
    blind = align_by_event_time(offline_ts[:0], offline_ts)
    assert "produced no rows at all" in blind.summary()


def test_a_one_millisecond_stamp_disagreement_names_itself_instead_of_an_empty_matrix():
    # The failure that cost a session: a candle's ts_event is T + 1 ms on both sides
    # (Hyperliquid's T is the bar's last millisecond), and while the two halves were
    # one millisecond apart the intersection was empty and the gate failed as "an
    # empty feature matrix proves nothing" — true, and a very long way from the
    # cause. Hourly bars one millisecond out share no stamp at all.
    hour = 3_600_000_000_000
    offline_ts, offline = stamped(8, step=hour), ramp(8)
    online_ts = offline_ts - 1_000_000  # stamped at T rather than T + 1 ms

    alignment = align_by_event_time(online_ts, offline_ts)
    assert alignment.disjoint
    assert "T is the bar's last millisecond" in alignment.summary()
    assert "+1000000 ns" in alignment.summary() or "-1000000 ns" in alignment.summary()

    report = aligned_feature_parity(
        offline, offline, online_ts=online_ts, offline_ts=offline_ts, columns=COLS
    )
    assert not report.passed
    assert "stamping conventions" in report.summary()


def test_an_alignment_still_unpacks_as_the_pair_of_index_arrays():
    # Every existing call site destructures the return value. The coverage rides
    # along on it rather than replacing it, so adding the accounting could not break
    # a caller that has not heard of it.
    offline_ts = stamped()
    alignment = align_by_event_time(offline_ts, offline_ts)
    left, right = alignment
    assert_array_equal(left, right)
    assert len(alignment) == 2 and alignment[0] is alignment.online
    assert alignment.passed and alignment.coverage.complete


def test_asking_the_aligner_to_raise_reports_the_gap_rather_than_the_symptom():
    # `on_gap="raise"` exists for a CI call site that wants the aligner itself to be
    # the assertion. The default is to report, because a live monitor must alarm
    # rather than die and a diagnostic must be able to observe a bad alignment.
    offline_ts, online_ts = stamped(), np.delete(stamped(), 4)
    assert align_by_event_time(online_ts, offline_ts).passed is False  # the default
    with pytest.raises(ParityError, match="never produced"):
        align_by_event_time(online_ts, offline_ts, on_gap="raise")


def test_a_pre_aligned_comparison_says_its_coverage_is_unchecked_rather_than_complete():
    # Two bare matrices carry no record of what was dropped upstream, and this gate
    # cannot invent one. What it must not do is let the absence read as completeness.
    online, offline = matrices()
    report = feature_parity(online, offline, columns=COLS)
    assert report.coverage is None
    assert "coverage=unchecked" in report.summary()


def test_the_online_and_the_offline_path_agree_on_the_reference_spec():
    # The end-to-end shape of the gate: "online" recomputes the whole spec on every
    # tick and keeps the last row, "offline" computes the batch in one pass. Under
    # Boundary B these are the same function, so the gate must be green — and it is
    # only green because every transform is causal.
    rng = np.random.default_rng(4)
    n = 120
    mid = 60_000.0 + np.cumsum(rng.normal(0.0, 3.0, n))
    half = rng.uniform(0.5, 1.5, n)
    inputs = {
        "bid_px": mid - half,
        "ask_px": mid + half,
        "bid_sz": rng.uniform(0.1, 5.0, n),
        "ask_sz": rng.uniform(0.1, 5.0, n),
        "trade_sz": rng.uniform(0.0, 2.0, n),
        "trade_sign": rng.choice([-1.0, 0.0, 1.0], n),
    }
    offline = PERP_CORE_V1.compute(inputs)
    online = np.vstack(
        [
            PERP_CORE_V1.compute({k: v[: i + 1] for k, v in inputs.items()})[-1]
            for i in range(n)
        ]
    )
    report = feature_parity(online, offline, columns=PERP_CORE_V1.columns, atol=FEATURE_ATOL)
    assert report.passed, report.summary()


# ── drift ─────────────────────────────────────────────────────────────────────


def normal(n: int, loc: float = 0.0, *, seed: int = 9) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc, 1.0, n)


def test_an_unchanged_distribution_is_stable():
    expected, actual = normal(5000, seed=1), normal(5000, seed=2)
    value = psi(expected, actual)
    assert value < PSI_STABLE
    assert psi_band(value) == "stable"


def test_a_shifted_distribution_is_significant():
    expected, actual = normal(5000, seed=1), normal(5000, 1.5, seed=2)
    value = psi(expected, actual)
    assert value > PSI_MODERATE
    assert psi_band(value) == "significant"


def test_the_bands_follow_the_conventional_thresholds():
    assert psi_band(0.0) == "stable"
    assert psi_band(PSI_STABLE - 1e-9) == "stable"
    assert psi_band(PSI_STABLE) == "moderate"
    assert psi_band(PSI_MODERATE) == "moderate"
    assert psi_band(PSI_MODERATE + 1e-9) == "significant"
    assert psi_band(float("inf")) == "significant"


def test_recomputing_bins_from_the_live_sample_would_hide_the_drift_frozen_bins_catch():
    # The mistake this design exists to prevent: quantile bins derived from each
    # sample separately make both histograms uniform by construction, so PSI reads
    # ~0 whatever happened and the monitor is structurally blind.
    expected, actual = normal(5000, seed=1), normal(5000, 1.5, seed=2)
    frozen = quantile_binning(expected, bins=10)
    assert psi(expected, actual, binning=frozen) > PSI_MODERATE

    p_e = quantile_binning(expected).counts(expected) / expected.size
    p_a = quantile_binning(actual).counts(actual) / actual.size
    blind = float(np.sum((p_a - p_e) * np.log(p_a / p_e)))
    assert blind < 0.01


def test_an_empty_bin_does_not_make_the_score_infinite():
    # Empty buckets are routine on a small live window; an infinite PSI turns every
    # alarm into the same alarm.
    rng = np.random.default_rng(3)
    expected = rng.uniform(0.0, 1.0, 1000)
    actual = rng.uniform(0.0, 0.2, 50)
    value = psi(expected, actual)
    assert np.isfinite(value) and value > PSI_MODERATE


def test_values_outside_the_training_range_are_counted_and_not_dropped():
    # "This feature now reaches values it never reached before" is the most
    # informative kind of drift; closed outer bins would discard exactly that.
    expected = np.random.default_rng(3).uniform(0.0, 1.0, 1000)
    actual = np.full(100, 5.0)
    binning = quantile_binning(expected)
    assert int(binning.counts(actual).sum()) == 100
    assert psi(expected, actual, binning=binning) > PSI_MODERATE


def test_a_constant_reference_still_detects_a_moved_feature():
    # A degenerate reference has no quantile structure at all; the honest split is
    # "equal to it" vs "anything else", not a single bin that can never move.
    expected = np.full(500, 1.0)
    assert psi(expected, np.full(500, 1.0)) == pytest.approx(0.0)
    assert psi_band(psi(expected, np.full(500, 2.0))) == "significant"


def test_kl_is_zero_for_an_identical_sample_and_positive_once_it_moves():
    expected = normal(2000, seed=1)
    assert kl_divergence(expected, expected) == pytest.approx(0.0)
    assert kl_divergence(expected, normal(2000, 1.0, seed=2)) > 0.0


def test_the_report_ranks_the_worst_feature_first():
    rng = np.random.default_rng(6)
    expected = rng.normal(0.0, 1.0, (4000, 3))
    actual = np.column_stack(
        [
            rng.normal(0.0, 1.0, 4000),
            rng.normal(0.3, 1.0, 4000),
            rng.normal(2.0, 1.0, 4000),
        ]
    )
    report = drift_report(expected, actual, columns=("calm", "nudged", "moved"))
    ranked = report.ranked()
    assert [f.name for f in ranked] == ["moved", "nudged", "calm"]
    assert ranked[0].band == "significant"
    assert ranked[-1].band == "stable"
    assert not report.passed
    assert report.significant()[0].name == "moved"
    with pytest.raises(ParityError, match="moved"):
        report.raise_for_status()


def test_a_feature_that_starts_emitting_nans_alarms_even_though_its_histogram_is_calm():
    # Drift a histogram of finite values cannot see: the same distribution, but a
    # fifth of the vectors now arrive empty. PSI stays stable and the model quietly
    # trades on whatever the caller substitutes for the hole.
    rng = np.random.default_rng(8)
    expected = rng.normal(0.0, 1.0, (4000, 1))
    actual = rng.normal(0.0, 1.0, (4000, 1))
    actual[rng.random(4000) < 0.2, 0] = np.nan

    report = drift_report(expected, actual, columns=("ret_1",))
    (feature,) = report.features
    assert feature.band == "stable"
    assert feature.nan_rate_delta > 0.15
    assert not report.passed
    assert report.nan_regressions()[0].name == "ret_1"
    assert "nan_rate" in report.summary()


def test_a_feature_that_has_gone_entirely_dark_is_not_reported_as_stable():
    expected = np.random.default_rng(2).normal(0.0, 1.0, (500, 1))
    actual = np.full((500, 1), np.nan)
    (feature,) = drift_report(expected, actual, columns=("ret_1",)).features
    assert feature.band == "significant"
    assert feature.n_actual == 0
