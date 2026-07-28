"""Tests for the loss-bound evidence job (:mod:`axon.strategies.loss_evidence`).

Only the pure halves are exercised here: the per-window accounting and the summary that
turns a fan-out's receipts into the numbers a bound is argued from. The remote entrypoint
needs the ML stack and a candle corpus, and nothing in the default gate may touch either.

Every name says what would go wrong if the check were removed, and two of them are about
an error that would make a **kill switch** wrong rather than a report wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.strategies.loss_evidence import (
    MAKER_FEE_BPS,
    TAKER_FEE_BPS,
    session_pnl,
    summarize,
)


def test_a_flat_window_costs_nothing_and_earns_nothing():
    """The zero case, pinned. A window where the strategy never took a position must
    report exactly zero — not a small number from a fee charged on a position it never
    held, which is the error that makes every quiet hour look like a loss."""
    out = session_pnl(
        targets=np.zeros(10),
        close=np.full(10, 65_000.0),
        notional=19.54,
        fee_bps=MAKER_FEE_BPS,
    )
    assert out == {
        "gross": 0.0,
        "fee": 0.0,
        "net": 0.0,
        "turnover_notional": 0.0,
        "trades": 0,
        "bars": 10,
    }


def test_the_fee_is_charged_on_what_traded_and_never_on_what_was_held():
    """The error that makes a low-turnover strategy look expensive and a high-turnover
    one look free — and it is the wrong direction for a bound in both cases.

    Two windows, the same *held* position throughout, different turnover: one goes long
    and stays, the other goes long, flat, long. The venue charges the second one more,
    because it traded more.
    """
    px = np.full(6, 65_000.0)
    steady = session_pnl(
        targets=np.array([1.0, 1, 1, 1, 1, 1]), close=px, notional=100.0, fee_bps=10.0
    )
    churny = session_pnl(
        targets=np.array([1.0, 0, 1, 0, 1, 1]), close=px, notional=100.0, fee_bps=10.0
    )
    # Steady: one entry of 100 and one exit of 100 = 200 of turnover at 10 bps = 0.20.
    assert steady["turnover_notional"] == pytest.approx(200.0)
    assert steady["fee"] == pytest.approx(0.20)
    # Churny: 100 in, out, in, out, in, then the closing exit = 600.
    assert churny["turnover_notional"] == pytest.approx(600.0)
    assert churny["fee"] == pytest.approx(0.60)
    assert churny["net"] < steady["net"], "trading more costs more, at the same P&L"


def test_a_window_is_always_closed_out_so_a_bound_is_not_argued_from_where_the_corpus_was_cut():
    """A window that ended holding something would report an unrealized figure this
    accounting has no mark for. Worse, the *bound* would then depend on where the corpus
    happened to be cut — a session that ended long in a rising tape would look free and
    one that ended long in a falling tape would look ruinous, both for the same reason.
    """
    px = np.array([100.0, 100.0, 100.0])
    out = session_pnl(
        targets=np.array([0.0, 1.0, 1.0]), close=px, notional=100.0, fee_bps=10.0
    )
    # One entry of 100 at bar 1, and the closing exit of 100 the last bar forces.
    assert out["turnover_notional"] == pytest.approx(200.0)
    assert out["trades"] == 2


def test_the_position_is_sized_once_the_way_a_live_session_sizes_it():
    """``--max-position`` is a quantity an operator states before the run, so the base-asset
    size is fixed for the window. Re-deriving it from each bar's close would make a window
    that spanned a large price move also span a changing position size, and the P&L would
    then include a re-sizing nobody asked for."""
    # Price doubles across the window while the target holds at +1.
    px = np.array([100.0, 200.0])
    out = session_pnl(
        targets=np.array([1.0, 1.0]), close=px, notional=100.0, fee_bps=0.0
    )
    # 1 unit bought at 100 (notional/first close), held through a 100-point rise.
    assert out["gross"] == pytest.approx(100.0)
    # And the turnover is the entry at 100 plus the forced exit at 200 = 300, on one unit.
    assert out["turnover_notional"] == pytest.approx(300.0)


def test_a_short_earns_when_the_price_falls():
    """The sign convention, pinned in the direction that is easy to get backwards. A
    bound that had this inverted would fire on the sessions that made money."""
    out = session_pnl(
        targets=np.array([-1.0, -1.0]),
        close=np.array([100.0, 90.0]),
        notional=100.0,
        fee_bps=0.0,
    )
    assert out["gross"] == pytest.approx(10.0)


def test_mismatched_or_empty_inputs_are_refused_rather_than_broadcast():
    """numpy would happily broadcast a scalar against a vector and produce a plausible
    number. A silent answer here is a window in a distribution that measured something
    else."""
    with pytest.raises(ValueError):
        session_pnl(targets=np.zeros(3), close=np.zeros(4), notional=1.0, fee_bps=0.0)
    with pytest.raises(ValueError):
        session_pnl(targets=np.zeros(0), close=np.zeros(0), notional=1.0, fee_bps=0.0)


def test_the_summary_reports_losses_as_magnitudes_because_that_is_what_the_config_takes():
    """``max_session_loss`` is a **magnitude** and the P&L it is compared against is
    signed. A summary that reported quantiles of the signed figure would have its p99 at
    the *best* session, and a bound copied from it would never fire."""
    results = [
        {"measurable": True, "net": -0.50},
        {"measurable": True, "net": -0.10},
        {"measurable": True, "net": +0.20},
        {"measurable": True, "net": -2.00},
        {"measurable": False, "reason": "still warming"},
    ]
    s = summarize(results)
    assert s["windows"] == 5
    assert s["measurable"] == 4, "the unmeasurable window is counted and not averaged"
    assert s["worst_loss"] == pytest.approx(2.00)
    assert s["best_gain"] == pytest.approx(0.20)
    assert s["mean_net"] == pytest.approx((-0.5 - 0.1 + 0.2 - 2.0) / 4)
    # The median loss is between the two middle losses (0.10 and 0.50 as magnitudes,
    # against -0.20 for the winning window): the point is that a *gain* enters as a
    # negative loss rather than being dropped.
    assert s["loss_quantiles"]["p50"] < s["loss_quantiles"]["p99"]
    assert s["suggested_ceiling"] == pytest.approx(2.00)


def test_a_fan_out_that_measured_nothing_says_so_rather_than_reporting_zeros():
    """Every window unmeasurable — a corpus shorter than the warmup, or a registry that
    served no model. Reporting a mean of zero and a worst loss of zero would be a
    distribution nobody could tell apart from a strategy that never traded."""
    s = summarize([{"measurable": False}, {"measurable": False}])
    assert s == {"windows": 2, "measurable": 0}
    assert "worst_loss" not in s


def test_the_two_fee_tiers_are_the_venues_and_the_maker_one_is_the_default():
    """The strategy serves at urgency 0 — a resting quote at the near touch — so a
    distribution priced at the taker fee would be about a session this one is not. Both
    are here because the flatten crosses, and a window that had to be crossed out of is
    exactly the expensive kind."""
    assert MAKER_FEE_BPS < TAKER_FEE_BPS
    px = np.full(4, 100.0)
    t = np.array([1.0, 1, 1, 1])
    maker = session_pnl(targets=t, close=px, notional=100.0, fee_bps=MAKER_FEE_BPS)
    taker = session_pnl(targets=t, close=px, notional=100.0, fee_bps=TAKER_FEE_BPS)
    assert maker["net"] > taker["net"]
    assert maker["gross"] == taker["gross"] == 0.0, "only the fee differs"


def test_the_cpu_fan_out_is_maximised_over_every_axis_that_changes_the_answer():
    """The four axes, and why fewer would answer a narrower question than the one asked.

    A loss bound has to be chosen for the duration a session will actually run, priced at the
    fee the session will actually pay, on the coin it will actually trade. The first version
    of this job was one coin, one session length, one fee tier and a fixed 240 windows — which
    is a distribution of one hour of BTC at the maker fee, and says nothing about the
    eight-hour soak at the taker fee that the exit will pay.
    """
    from axon.strategies.loss_evidence import (
        COIN_NOTIONAL,
        FEE_TIERS,
        SESSION_LENGTHS,
        session_loss_job,
    )

    job = session_loss_job()
    assert job.workload == "walk_forward", "a CPU workload in hwsched's own taxonomy"
    assert not job.resources, "nothing pinned: the planner's CPU sizing is the right one"

    tasks = job.tasks
    assert {t["coins"] for t in tasks} == {"BTC", "ETH", "SOL"}
    assert {t["session_bars"] for t in tasks} == set(SESSION_LENGTHS)
    assert {t["fee_bps"] for t in tasks} == set(FEE_TIERS)
    # An 8-hour window is on the list because that is what the soak runs, and a bound scaled
    # naively from one hour is too tight on the costs and too loose on the market.
    assert 480 in SESSION_LENGTHS

    # Each coin is priced at its own notional: `max_position` is a quantity and 0.0003 BTC is
    # not the same risk as 0.0003 SOL.
    for t in tasks:
        assert t["notional"] == pytest.approx(COIN_NOTIONAL[t["coins"]])

    # And within one (coin, length, fee) the windows are **non-overlapping**: overlapping ones
    # share bars, so a quantile over them understates the dispersion by exactly the overlap —
    # which gets worse as the window grows.
    for length in SESSION_LENGTHS:
        starts = sorted(
            t["start"]
            for t in tasks
            if t["coins"] == "BTC" and t["session_bars"] == length and t["fee_bps"] == FEE_TIERS[0]
        )
        assert len(set(starts)) == len(starts)
        assert all(b - a == length for a, b in zip(starts, starts[1:])), length


def test_a_corpus_too_short_for_one_window_is_refused_rather_than_planned_empty():
    """hwsched would happily plan zero tasks and report success, and an empty fan-out that
    "succeeded" is a distribution nobody would know they did not have."""
    from axon.strategies.loss_evidence import SpecTooSmall, session_loss_job

    with pytest.raises(SpecTooSmall):
        session_loss_job(corpus_bars=30, session_lengths=(60,))


def test_the_gpu_job_is_the_full_grid_and_the_calibration_is_the_one_that_fits_a_guard():
    """Two GPU jobs, and the pair is the correction to a mistake worth naming.

    hwsched prices a GPU tree fit at a flat 1800 s because it has no measurement for one, so
    the wide grid prices at ~$157 — pessimistic by roughly an order of magnitude. The first
    version of this module *cut the grid to four points* to fit a $3 cap. That sized the
    question against a guard: `hwsched.toml`'s per-job default describes one Modal Starter
    account, and there are fourteen profiles.

    So there are two jobs. The wide one declares a cap covering the pessimistic estimate; the
    calibration one is four points, fits a single account's default, and its product is the
    wall time that would let the wide one be priced honestly.
    """
    from axon.strategies.loss_evidence import gpu_calibration_job, gpu_fit_sweep_job

    wide = gpu_fit_sweep_job()
    assert wide.n_tasks == 256, "four axes of four"
    assert wide.resources["device"] == "gpu"
    assert wide.resources["gpu_type"] == "T4", "bandwidth-bound: a bigger card only spends faster"
    assert wide.max_usd >= 157, "the cap covers hwsched's pessimistic estimate"
    # Modal's own workspace cap on concurrent GPUs. Stated so the wall-time arithmetic is
    # legible rather than discovered: 256 / 10 = 26 waves.
    assert wide.max_parallel == 10

    cal = gpu_calibration_job()
    assert cal.n_tasks == 4
    assert cal.max_usd <= 5.0, "fits one account's default guard, so it can go first"
    assert cal.resources["device"] == "gpu", "same device, or it measures the wrong thing"


def test_sharding_across_profiles_keeps_shards_contiguous_so_a_loss_is_legible():
    """Capacity across the fourteen Modal profiles is an operator-level fan-out, because
    hwsched submits to one provider per run and Modal picks the account from the submitting
    process's ``MODAL_PROFILE``.

    The shards are contiguous, and that is the decision under test. A round-robin split would
    put adjacent windows of the same coin in different accounts, so one shard failing would
    punch evenly-spaced holes through every coin's distribution and the survivors would still
    look like a complete sample.
    """
    from axon.strategies.loss_evidence import session_loss_job, shard_by_profile

    job = session_loss_job()
    shards = shard_by_profile(job, 14)
    assert len(shards) <= 14
    assert sum(s.n_tasks for s in shards) == job.n_tasks, "no task lost, none duplicated"
    assert len({s.name for s in shards}) == len(shards), "distinct names or the store collides"

    # Contiguous: each shard's tasks are a consecutive slice of the parent's list.
    flat = [t for s in shards for t in s.tasks]
    assert flat == list(job.tasks)

    # Every shard keeps the parent's cap. Dividing it would make fourteen shards each a
    # fourteenth as able to run, which is the opposite of spreading capacity.
    assert all(s.max_usd == job.max_usd for s in shards)


def test_sharding_a_grid_job_is_refused_because_a_grid_has_no_task_list_to_split():
    """`param_sweep` materializes tasks from `params` inside hwsched, so there is nothing
    here to slice. Silently returning the whole job N times would run the same 256 points on
    fourteen accounts."""
    from axon.compute.spec import SpecError
    from axon.strategies.loss_evidence import gpu_fit_sweep_job, shard_by_profile

    with pytest.raises(SpecError):
        shard_by_profile(gpu_fit_sweep_job(), 4)
