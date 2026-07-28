"""The replay golden test, from the Python side (``docs/07``, rung 1).

Every test here drives the *Rust* replay binary over the committed session at
``crates/axon-replay/testdata/session.jsonl`` and the strategy output beside it at
``session.signals.jsonl``. Nothing in this file parses either, and nothing here
reimplements a market-data rule, a reconciliation rule or a planning rule — that is
the point of the harness, and a test that quietly did it in Python would be validating
the wrong program.

What the golden covers is the whole chain: the book, the mark cache, the order
tracker's reconciled position, and the orders and ``cloid``s the planner emitted. The
first three are what a shadow-trading diff compares; the last is what a live-versus-
backtest diff compares. A golden over market data alone was narrower than its name.

Regenerating the artifacts:

* the log and the signals — ``cargo run -p axon-replay --example make_fixture_log -- \\
  crates/axon-replay/testdata/session.jsonl \\
  --signals crates/axon-replay/testdata/session.signals.jsonl``
* the stored reference — re-run this file with ``AXON_UPDATE_GOLDEN=1``. Do that only
  when a *deliberate* change to the core's output is being accepted; the reference
  going stale on its own is the harness working.

Offline and deterministic: the only subprocess is the local replay binary, built from
the workspace if it is missing, and the tests skip cleanly when no Rust toolchain is
available.
"""

from __future__ import annotations

import dataclasses
import os
from decimal import Decimal
from pathlib import Path

import pytest

from axon.backtest import (
    BacktestResult,
    Backtester,
    ReplayUnavailable,
    TraceRow,
    compare_to_golden,
)
from axon.backtest.runner import PlannedOrder
from axon.backtest.golden import GoldenMismatch

FIXTURE_LOG = Path("crates/axon-replay/testdata/session.jsonl")
REFERENCE = Path("crates/axon-replay/testdata/session.golden.json")


@pytest.fixture(scope="module")
def log_path(repo_root) -> Path:
    path = repo_root / FIXTURE_LOG
    if not path.exists():  # pragma: no cover - the fixture is committed
        pytest.skip(f"{path} is missing")
    return path


@pytest.fixture(scope="module")
def backtester(repo_root) -> Backtester:
    try:
        return Backtester(repo_root=repo_root)
    except ReplayUnavailable as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def result(backtester, log_path) -> BacktestResult:
    return backtester.run(log_path)


def _first_row_with(result: BacktestResult, column: str) -> int:
    for i, row in enumerate(result.trace):
        if row.values.get(column) is not None:
            return i
    raise AssertionError(f"no trace row carries {column}")  # pragma: no cover


def _perturb(result: BacktestResult, index: int, column: str, value) -> BacktestResult:
    """A candidate identical to ``result`` except for one cell."""
    rows = list(result.trace)
    row = rows[index]
    rows[index] = dataclasses.replace(row, values={**row.values, column: value})
    return dataclasses.replace(result, trace=tuple(rows))


def _regime(row: TraceRow) -> str:
    """A toy discretized decision: which side of a threshold the mid sits on.

    The strategy's *real* discrete output now reaches the trace on its own — the
    ``plan`` decision column, and :attr:`BacktestResult.orders` — so this exists only
    to exercise :meth:`BacktestResult.with_decision`, which lets a caller derive a
    decision the core does not itself emit. It is a pure function of the row, which is
    the only property that method requires.
    """
    mid = row.values.get("mid")
    if mid is None:
        return "none"
    return "high" if mid > Decimal("50004") else "low"


def test_replaying_one_log_twice_produces_an_identical_result(backtester, log_path):
    """The property everything above this rung depends on.

    If two runs over one input can differ, model parity, feature parity and every
    shadow-trading diff are measuring noise rather than the thing they claim to.
    """
    first = backtester.run(log_path)
    second = backtester.run(log_path)
    compare_to_golden(first, second).assert_ok()
    assert first.to_dict() == second.to_dict()
    assert first.events == 59
    assert len(first.trace) == 59
    # Not just the same market data: the same orders, with the same client ids. A
    # `cloid` is derived from the signal rather than minted from a counter precisely so
    # this can be asserted (ADR-0014 §5) — without it a retried submit becomes a second
    # position and a live-versus-backtest diff has nothing to align on.
    assert [o.cloid for o in first.orders] == [o.cloid for o in second.orders]
    assert first.orders == second.orders
    assert first.cancels == second.cancels
    assert len(first.orders) == 4, "the fixture must actually plan something"


def test_a_run_matches_the_stored_reference_exactly(result, repo_root):
    """Rung 1 of ``docs/07``: outputs match a stored reference.

    Exact, not within tolerance — replay is deterministic on one platform, so a
    tolerance here would only be hiding something. Tolerances belong in the
    cross-implementation comparisons above this rung (Python model vs Rust
    inference), where they are a statement about float width, not about replay.
    """
    path = repo_root / REFERENCE
    if os.environ.get("AXON_UPDATE_GOLDEN"):
        result.save(path)
        pytest.skip(f"regenerated {path}")
    if not path.exists():  # pragma: no cover - the reference is committed
        pytest.skip(f"{path} is missing; regenerate with AXON_UPDATE_GOLDEN=1")

    comparison = compare_to_golden(BacktestResult.load(path), result)
    assert comparison.ok, comparison.report()


def test_a_stored_reference_round_trips_without_losing_precision(result, tmp_path):
    """A golden file is JSON, and JSON has no decimal type.

    Money crosses this boundary as a *string*, so ``50000.0`` stays ``50000.0`` and
    not ``50000.000000000001``. A float round trip would introduce a divergence that
    the comparison would then faithfully report — a failure manufactured by the
    harness itself.
    """
    path = tmp_path / "reference.json"
    result.save(path)
    reloaded = BacktestResult.load(path)
    compare_to_golden(result, reloaded).assert_ok()

    mid = reloaded.trace[_first_row_with(result, "mid")].values["mid"]
    assert isinstance(mid, Decimal)
    assert str(mid) == str(result.trace[_first_row_with(result, "mid")].values["mid"])


def test_money_arrives_as_decimal_never_float(result):
    """A price parsed as a float64 is a price that no longer compares equal.

    ``rust_decimal`` writes a decimal *string*; this asserts the Python side keeps it
    one, including the trailing zero that records the venue's scale.
    """
    row = result.trace[_first_row_with(result, "mid")]
    for column in ("mid", "best_bid", "best_ask"):
        assert isinstance(row.values[column], Decimal), column
    assert str(row.values["mid"]) == "50000.0", "scale survives the boundary"
    assert isinstance(row.values["mark_ts"], (int, type(None)))
    assert not isinstance(row.values["mid"], float)


def test_a_value_outside_the_tolerance_is_caught(result):
    """The comparison has to actually compare.

    A harness whose only test is "a run matches itself" would pass forever while
    detecting nothing, so this perturbs one cell and checks both directions of the
    tolerance.
    """
    i = _first_row_with(result, "mid")
    original = result.trace[i].values["mid"]
    candidate = _perturb(result, i, "mid", original + Decimal("0.5"))

    strict = compare_to_golden(result, candidate)
    assert not strict.ok
    assert strict.divergence_count == 1
    assert strict.max_abs_diff["mid"] == Decimal("0.5")

    assert compare_to_golden(result, candidate, tolerance=Decimal("0.5")).ok
    assert not compare_to_golden(
        result, candidate, tolerance={"best_bid": Decimal("1")}
    ).ok, "a tolerance on another column must not cover this one"


def test_a_decision_flip_is_never_forgiven_by_a_tolerance(result):
    """The rule that protects P&L: no discretized decision may flip.

    The perturbation here is small enough that a generous tolerance absorbs the
    *value* change, which is exactly the situation a merged check would wave through
    — and exactly when the strategy would have sent a different order.
    """
    i = _first_row_with(result, "mid")
    reference = result.with_decision("regime", _regime)
    assert reference.trace[i].decisions["regime"] == "low"

    nudged = _perturb(result, i, "mid", Decimal("50004.5")).with_decision(
        "regime", _regime
    )
    assert nudged.trace[i].decisions["regime"] == "high"

    comparison = compare_to_golden(reference, nudged, tolerance=Decimal("100"))
    assert not comparison.ok
    assert comparison.divergence_count == 0, "the value change is inside tolerance"
    assert comparison.flip_count == 1
    assert "regime" in comparison.report()
    with pytest.raises(GoldenMismatch):
        comparison.assert_ok()


def test_a_missing_value_is_not_a_small_value(result):
    """``None`` against a number is a divergence at any tolerance.

    A missing mark makes the risk gate fail closed; a mark of zero, or a mark treated
    as "near enough to absent", sizes a position against a price that does not exist.
    """
    i = _first_row_with(result, "mark_px")
    candidate = _perturb(result, i, "mark_px", None)
    comparison = compare_to_golden(result, candidate, tolerance=Decimal("1e9"))
    assert not comparison.ok
    assert comparison.divergence_count == 1


def test_a_reordered_row_is_a_structural_failure_not_a_numeric_one(result):
    """Rows must align before their columns mean anything.

    If two traces describe different events, diffing their columns reports noise and
    buries the real cause, so identity mismatches stop the row comparison instead of
    feeding it.
    """
    rows = list(result.trace)
    rows[3] = dataclasses.replace(rows[3], ts_event=rows[3].ts_event + 1)
    comparison = compare_to_golden(result, dataclasses.replace(result, trace=tuple(rows)))
    assert not comparison.ok
    assert any("identity" in s for s in comparison.structural)
    assert comparison.divergence_count == 0


def test_the_two_traversals_disagree_only_where_the_late_event_lands(backtester, log_path):
    """Event-time order and capture order are different questions.

    The fixture carries one genuinely late arrival. In event-time order it lands
    mid-session; in capture order it is the last thing the session saw, so it wins the
    "last trade" slot. Silently sorting a late feed into shape would erase that
    difference and let a replay certify an interleaving the live session never
    experienced.
    """
    event_time = backtester.run(log_path)
    as_captured = Backtester(binary=backtester.binary, order="as-captured").run(log_path)

    assert event_time.late_arrivals == 1
    assert as_captured.late_arrivals == 1, "the log's shape does not depend on the read"
    assert event_time.symbols[1]["last_trade_px"] == Decimal("50009.5")
    assert as_captured.symbols[1]["last_trade_px"] == Decimal("50004.2")

    comparison = compare_to_golden(event_time, as_captured)
    assert not comparison.ok
    assert any(s.startswith("order:") for s in comparison.structural)


# ── the widened chain ────────────────────────────────────────────────────────


def test_the_trace_carries_reconciled_state_and_not_only_market_data(result):
    """What N5 exists for.

    A golden over the book alone says nothing about the two things every rung above it
    compares. These columns are read straight out of ``OrderTracker`` — the position
    the log's fills produced, the resting size that is exposure the moment it is
    accepted, and the count of fills we could not attribute at all.
    """
    columns = set(result.columns())
    assert {"position_qty", "risk_qty", "resting_qty", "open_orders"} <= columns
    assert "orphan_fills" in columns
    final = result.symbols[1]
    assert final["position_qty"] == Decimal("0.02"), "the log's one fill"
    assert final["open_orders"] == 0
    # The adopted order rested with 0.03 unfilled before it was cancelled, so somewhere
    # in the middle of the session risk saw more exposure than the position did.
    peak = max(
        row.values["risk_qty"]
        for row in result.trace
        if row.symbol_id == 1 and row.values["risk_qty"] is not None
    )
    assert peak == Decimal("0.05")


def test_a_replayed_fill_is_attributed_to_the_order_that_caused_it(result):
    """A fill credited to the wrong order — or to none — leaves the position right and
    our own view of the venue wrong, so the next plan pulls an order that is already
    gone and leaves working the one it meant to cancel."""
    attributed = [r for r in result.trace if r.decisions.get("fill")]
    assert len(attributed) == 1
    assert attributed[0].decisions["fill"] == (
        "0x0000000000000000000000000000002a filled=0.02"
    )
    assert all(r.values["orphan_fills"] == 0 for r in result.trace)


def test_the_strategy_output_is_in_the_result_with_its_cloids(result):
    """The order-level half of the golden.

    Stated explicitly rather than only compared against itself: a change to the urgency
    table, the delta rule or the ``cloid`` layout must land here as a named difference,
    not as a reference somebody regenerated without reading.
    """
    assert result.signal_source == (
        "synthetic:two-symbol-session signals (make_fixture_log)"
    )
    assert result.signals["accepted"] == 4
    assert result.signals["expired"] == 1, "one record aged out in the ring"
    described = [
        (o.symbol_id, o.side, str(o.qty), str(o.price), o.tif) for o in result.orders
    ]
    assert described == [
        (1, "buy", "0.03000000", "49999.0", "post_only"),
        (2, "sell", "0.50000000", "2999.4", "gtc"),
        (1, "buy", "0.03000000", "50004.0", "gtc"),
        (1, "sell", "0.02", "49757.9600", "ioc"),
    ]
    assert result.orders[-1].reduce_only, "a flatten is reduce-only regardless"
    assert [c.target for c in result.cancels] == ["oid:56968034936"]
    assert all(o.cloid.startswith("0x") and len(o.cloid) == 34 for o in result.orders)
    assert len({o.cloid for o in result.orders}) == 4, "one signal, one id"


def test_a_replayed_order_is_never_given_a_fill(result):
    """The refusal, asserted rather than merely documented.

    The replay plans four orders. None reaches a venue, so none is acknowledged, none
    rests and none fills — the position moves exactly once, on the one fill the
    *captured* session actually received. A harness that closed this loop would be
    reporting a P&L the venue never agreed to, and it would read as a working backtest.
    """
    assert len(result.orders) == 4
    btc = [r for r in result.trace if r.symbol_id == 1]
    steps = sum(
        1
        for a, b in zip(btc, btc[1:])
        if a.values["position_qty"] != b.values["position_qty"]
    )
    assert steps == 1, "a position moved without a fill in the log"
    assert result.symbols[1]["position_qty"] == Decimal("0.02")
    # …and the orders really would have moved it, so this is not vacuous.
    assert sum(o.qty for o in result.orders if o.side == "buy") > Decimal("0.05")


def test_a_changed_order_is_a_flip_no_tolerance_forgives(result):
    """A size or a limit that moved is a different instruction at a real venue.

    This is the case a merged value/decision check waves through: the *value* change is
    tiny, and the order sent is not the order that was sent before.
    """
    nudged = dataclasses.replace(
        result,
        orders=(
            dataclasses.replace(
                result.orders[0], qty=result.orders[0].qty + Decimal("0.00000001")
            ),
            *result.orders[1:],
        ),
    )
    comparison = compare_to_golden(result, nudged, tolerance=Decimal("1000"))
    assert not comparison.ok
    assert comparison.divergence_count == 0
    assert comparison.flip_count == 1
    assert "orders[0].qty" in comparison.report()


def test_a_reused_cloid_is_caught_even_when_every_price_matches(result):
    """The id is the whole reason an order-level diff is possible.

    Two orders sharing a ``cloid`` is the collision that makes the venue de-duplicate a
    genuine second order, or makes reconciliation credit a fill to the wrong one.
    """
    swapped = dataclasses.replace(
        result,
        orders=(
            dataclasses.replace(result.orders[0], cloid="0x" + "0" * 32),
            *result.orders[1:],
        ),
    )
    comparison = compare_to_golden(result, swapped, tolerance=Decimal("1e9"))
    assert comparison.flip_count == 1
    assert "cloid" in comparison.report()


def test_an_order_that_appears_from_nowhere_is_structural_not_numeric(result):
    """Two plans of different lengths are not two versions of one plan. Pairing them by
    index would report every later order as changed and bury the one that appeared."""
    extra = dataclasses.replace(result, orders=(*result.orders, result.orders[0]))
    comparison = compare_to_golden(result, extra)
    assert not comparison.ok
    assert any(s.startswith("orders:") for s in comparison.structural)
    assert comparison.flip_count == 0


def test_a_lost_execution_event_is_a_structural_failure_not_an_absence(result):
    """A tracker the Rust core could not read is counted, never silent.

    Past the first drop the reconciled columns describe a tracker that stopped
    following the venue, and two runs that both stopped following it agree with each
    other perfectly — so the absences alone would keep a golden green. The counter is
    what makes the degradation a diff, and no tolerance may forgive it: this is not a
    quantity that drifted.
    """
    assert result.dropped_exec_events == 0, "the fixture session never lost one"
    degraded = dataclasses.replace(result, dropped_exec_events=1)
    comparison = compare_to_golden(result, degraded, tolerance=Decimal("1e9"))
    assert not comparison.ok
    assert any(s.startswith("dropped_exec_events:") for s in comparison.structural)


def test_a_run_with_no_signals_still_reconciles_and_plans_nothing(backtester, log_path):
    """A market-data-only capture is the ordinary case, not a degraded one.

    The fan-out, the marks and the tracker all still run; the adapter simply has
    nothing to admit. And the result says which kind of run it was, so a golden can
    never silently compare a session that had a strategy against one that did not.
    """
    bare = Backtester(binary=backtester.binary, signals=False).run(log_path)
    assert bare.signal_source is None
    assert bare.orders == ()
    assert bare.intent_passes == 0
    assert bare.symbols[1]["position_qty"] == Decimal("0.02")

    with_signals = backtester.run(log_path)
    comparison = compare_to_golden(with_signals, bare)
    assert not comparison.ok
    assert any(s.startswith("signal_source:") for s in comparison.structural)


def test_a_signal_counter_that_stops_moving_is_a_structural_failure(result):
    """A reader that silently began rejecting everything would otherwise show up only
    as an absence of orders, which looks exactly like a quiet strategy."""
    muted = dataclasses.replace(result, signals={**result.signals, "accepted": 0})
    comparison = compare_to_golden(result, muted, tolerance=Decimal("1e9"))
    assert not comparison.ok
    assert any(s.startswith("signals.accepted:") for s in comparison.structural)


def test_order_money_arrives_as_decimal_and_a_cloid_stays_a_string(result):
    """128 bits do not survive a JSON number, and a price through a float64 is a price
    that no longer compares equal."""
    order = result.orders[0]
    assert isinstance(order.qty, Decimal)
    assert isinstance(order.price, Decimal)
    assert isinstance(order.cloid, str)
    assert str(order.price) == "49999.0", "the venue's scale survives the boundary"
    assert isinstance(PlannedOrder.from_dict(order.to_dict()).price, Decimal)


def test_there_is_no_python_fallback_when_the_binary_is_missing(tmp_path):
    """A backtest that cannot run the Rust core must fail, not approximate it.

    The whole parity claim is "the same code"; a Python replay path would be a second
    implementation, and the first time the two disagreed the harness would be the last
    place anyone looked.
    """
    from axon.backtest import find_replay_binary

    with pytest.raises(ReplayUnavailable):
        find_replay_binary(tmp_path, build=False)


def test_an_unknown_traversal_is_refused_at_construction():
    with pytest.raises(ValueError):
        Backtester(binary="/nonexistent/replay_log", order="whatever-order")
