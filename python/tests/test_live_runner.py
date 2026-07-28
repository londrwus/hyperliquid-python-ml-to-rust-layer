"""Tests for the live producer (:mod:`axon.strategies.live_runner`).

Every name says what would break if the check were removed. Three of the four are
about a fault that has already happened to somebody: a stamp the reader refuses, a
target quietly dropped when the ring is full, and a flatten that skips in the one
case it exists for.

Nothing here touches a network or a venue. The signal ring is a real ring in
``tmp_path``, because "the record was built" and "the record crossed the ring" are
different claims and only the second one is what a live session depends on.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from axon.signals import RingConsumer
from axon.strategies.baseline import NO_MODEL_VERSION, Baseline, BaselineParams
from axon.strategies.live_runner import LiveRunError, LiveRunner
from axon.strategy import Bar

SYMBOL = 3
OTHER = 4

#: 2026-07-26T15:31:00Z in nanoseconds — a real m1 close time from the P6 run, so the
#: skew this file asserts on is the skew a live session actually has.
BAR_CLOSE_NS = 1_785_079_860_000_000_000


def _runner(tmp_path, **kwargs):
    strategy = Baseline(BaselineParams(symbol_id=SYMBOL, max_position=Decimal("0.0003")))
    return LiveRunner(
        strategy,
        symbol_id=SYMBOL,
        signal_ring=str(tmp_path / "sig.ring"),
        model_version=NO_MODEL_VERSION,
        **kwargs,
    )


def _bar(ts: int, close: int, symbol_id: int = SYMBOL) -> Bar:
    # Fixed-point wire integers, as the bar ring carries them.
    return Bar(
        symbol_id=symbol_id,
        ts_event=ts,
        open=close,
        high=close + 100,
        low=close - 100,
        close=close,
        volume=1_000,
    )


def _warm(runner, *, base: int = 6_400_000_000_000, step_ns: int = 60_000_000_000) -> int:
    """Feed enough bars to clear the 21-bar warmup, with a real range on each.

    A ramp rather than a constant: the spec's realized-volatility floor refuses a
    frozen tape, so a constant-price warmup would leave the strategy silent and every
    assertion below would pass vacuously.
    """
    ts = BAR_CLOSE_NS
    for i in range(runner.strategy.warmup_bars + 1):
        runner.on_bar(_bar(ts, base + i * 1_000_000))
        ts += step_ns
    return ts


def test_a_signal_is_stamped_with_the_producers_clock_and_not_the_bars_close(tmp_path):
    """The stamp the Rust reader judges admission on, and the run P1 lost to it.

    A record stamped with the bar's own close arrives at the reader with the feed's
    lag already spent — and on m1 that lag is measured against a close a whole
    interval old. The reader refuses it as ``expired`` before the planner sees it,
    and the symptom is a strategy that emits and never trades.
    """
    runner = _runner(tmp_path)
    before = time.time_ns()
    ts = _warm(runner)
    # Drive it hard enough in one direction to force a target off zero, so there is a
    # record to inspect at all.
    for i in range(8):
        runner.on_bar(_bar(ts + i * 60_000_000_000, 6_400_000_000_000 - (i + 1) * 8_000_000))
    after = time.time_ns()
    runner.flush()
    runner.close()

    records = RingConsumer(str(tmp_path / "sig.ring")).read_batch()
    assert records.size > 0, "the warmup ramp must produce at least one target change"
    for rec in records:
        stamp = int(rec["ts_event"])
        assert before <= stamp <= after, (
            "a signal must carry the moment the strategy decided, taken from the "
            "producer's own clock"
        )
        # And it must be nowhere near the bar's close, which is the wrong answer this
        # test exists to exclude. The synthetic bars are stamped in 2026-07-26's
        # 15:31 minute; a stamp within a bar interval of that is the bar's, not ours.
        assert abs(stamp - BAR_CLOSE_NS) > 60_000_000_000


def test_every_record_carries_the_close_of_the_bar_that_caused_it(tmp_path):
    """The other half of the stamp above, and the gap that had no ceiling for two days.

    ``ts_event`` is the producer's clock because the reader ages admission against it
    (the test above). That leaves the runtime unable to see the largest latency in the
    system: measured 951 / 12 051 / **111 475** ms from a closed m1 bar to the strategy
    acting on it, over 57 live bars on 2026-07-27, and visible only in this runner's own
    transcript. A record carrying only ``ts_event`` makes a decision one second after a
    bar and one two minutes after it identical from the Rust side, so the gap could be
    quoted afterwards and never budgeted.

    Schema version 3 put ``ts_cause`` on the wire and this is where it is written — by
    the runner rather than by the strategy, because the runner is the only thing holding
    both stamps, and asking every strategy author to remember a runtime invariant is the
    mistake ADR-0036 §5 names for the re-quote.
    """
    runner = _runner(tmp_path)
    ts = _warm(runner)
    closes = []
    for i in range(8):
        bar_ts = ts + i * 60_000_000_000
        closes.append(bar_ts)
        runner.on_bar(_bar(bar_ts, 6_400_000_000_000 - (i + 1) * 8_000_000))
    runner.flush()
    runner.close()

    records = RingConsumer(str(tmp_path / "sig.ring")).read_batch()
    assert records.size > 0
    for rec in records:
        cause = int(rec["ts_cause"])
        assert cause != 0, "a bar strategy must always say what it was deciding about"
        assert cause in set(closes) or cause == BAR_CLOSE_NS or cause < ts, (
            "the cause must be one of the bars actually fed, not a rounded or "
            f"reconstructed time: {cause}"
        )
        # And it must be the bar's own close rather than the decision: the whole point is
        # that these two are different numbers, and a producer that set them equal would
        # report a zero on the one stage that reads in seconds.
        assert cause < int(rec["ts_event"]), "the cause precedes the decision"
        assert int(rec["ts_event"]) - cause > 60_000_000_000, (
            "these synthetic bars are stamped in 2026-07-26 and decided now, so the gap "
            "is enormous - a producer that stamped ts_cause from its own clock would "
            "show a gap of microseconds here"
        )


def test_another_instruments_bar_never_reaches_the_strategy(tmp_path):
    """One runner drives one symbol, and the filter is the runner's job.

    The baseline filters internally, so a runner that dispatched everything would
    look correct against it and would trade another coin's tape the day a strategy
    without that guard is driven — which is the whole point of the ``module:callable``
    escape hatch this file's CLI offers.
    """
    runner = _runner(tmp_path)
    ts = _warm(runner)
    seen = runner.stats.bars_seen
    runner.on_bar(_bar(ts, 6_400_000_000_000, symbol_id=OTHER))
    runner.close()
    assert runner.stats.bars_seen == seen, "another instrument's bar was counted as ours"
    assert runner.stats.bars_other_symbol == 1


def test_a_full_ring_queues_the_target_rather_than_dropping_it(tmp_path):
    """Full is backpressure, never a licence to skip.

    "The next target supersedes it anyway" is wrong in the case that matters: if the
    consumer is dead there *is* no next one, and the venue is left holding the
    previous target. The records stay, in order, and the run raises only once the
    backlog outgrows a whole ring.
    """
    # Capacity 2 and no consumer, driven through `emit_target` rather than through
    # bars: how many targets a ramp happens to produce is a fact about the z-score,
    # and a backpressure test that depends on it stops testing backpressure the day
    # the entry threshold moves.
    runner = _runner(tmp_path, capacity=2)
    emitted = 0
    with pytest.raises(LiveRunError):
        for _ in range(16):
            runner.emit_target(Decimal("0.0003"))
            emitted += 1
    assert runner.stats.backpressure_waits > 0, "the ring must have filled for this to test"
    assert runner.stats.signals_pushed < emitted, "the ring cannot have taken them all"
    # Nothing was lost, **including on the raising path**: the call that raised had
    # already queued its own record before the overflow was detected, so the total is
    # one more than the calls that returned. That is the property that matters — a
    # run which reported backpressure and then quietly discarded the target it was
    # holding would satisfy every other assertion here.
    assert runner.pending() + runner.stats.signals_pushed == emitted + 1
    runner.close()


def test_the_flatten_request_is_emitted_even_when_the_strategy_believes_it_is_flat(tmp_path):
    """The strategy's target and the tracked position disagree exactly when it matters.

    A partial fill leaves a position the strategy never asked for. A flatten that
    skipped because ``strategy.target == 0`` would skip in the one situation it was
    written for, and the run would end holding it.
    """
    runner = _runner(tmp_path)
    assert runner.strategy.target == 0
    runner.emit_target(Decimal(0))
    runner.close()

    records = RingConsumer(str(tmp_path / "sig.ring")).read_batch()
    assert records.size == 1
    assert int(records["target_qty"][0]) == 0
    assert int(records["ttl_ms"][0]) == runner.strategy.params.ttl_ms


def test_a_trading_session_can_be_watched_by_a_diff_without_a_second_bar_reader(tmp_path):
    """The fan-out, and the two independent reasons it had to be a fan-out.

    Phase 6 could not watch parity on a session that was **trading**, for two reasons
    neither of which is about parity:

    1. **An SPSC bar ring has one consumer.** On a trading session the strategy is it.
       Measured on 2026-07-26: two drainers on one market-data ring each saw about half
       the records and each reported the other's reads as drops. A monitor attached
       beside the strategy takes bars away from the thing placing orders.
    2. **A second session's shutdown sweeps the account.** ``graceful_shutdown`` calls
       ``cancel_all``, which on Hyperliquid is account-wide, so a read-only watcher
       would cancel the trader's resting orders on its way out.

    So parity was computed afterwards from the capture, which keeps the comparison and
    gives up the alarm. This test is the alarm: one reader, dispatched to the strategy
    **and** to the diff, with the diff seeing exactly the rows the strategy served.
    """
    from axon.strategies.shadow import BarParityDiff

    strategy = Baseline(BaselineParams(symbol_id=SYMBOL, max_position=Decimal("0.0003")))
    diff = BarParityDiff(spec=strategy.spec, interval="1m", window_bars=8)
    runner = LiveRunner(
        strategy,
        symbol_id=SYMBOL,
        signal_ring=str(tmp_path / "sig.ring"),
        model_version=NO_MODEL_VERSION,
        parity=diff,
    )

    ts = BAR_CLOSE_NS
    for i in range(60):
        runner.on_bar(_bar(ts, 6_400_000_000_000 + i * 1_000_000))
        ts += 60_000_000_000
    runner.flush()
    runner.close()

    # The strategy got every bar — the diff took none away from it.
    assert runner.stats.bars_seen == 60
    assert diff.bars == 60, "the diff saw the same 60, from the same single read"

    # …and it compared them. A diff that ran and compared nothing is a diff that would
    # report a healthy session whatever the serving path did. 60 bars in 8-bar windows is
    # 7 flushes, and the ones entirely inside the spec's 21-bar warmup are counted apart:
    # a window where neither side has a finite row is not a window that agreed, and it is
    # not a window that disagreed either.
    assert diff.windows + diff.warmup_windows == 7, (
        f"{diff.windows} compared + {diff.warmup_windows} in warmup"
    )
    assert diff.windows >= 5, f"only {diff.windows} window(s) compared anything"
    assert diff.rows_compared > 0
    assert runner.parity_alarms == 0, runner.parity_last_reason

    # The serving path and an offline recompute over the same bars agree exactly, which
    # is the claim the whole harness exists to make. Not "approximately": the online
    # buffer and the offline matrix run the *same* spec over the *same* integers.
    worst_column, worst = diff.column_diff().worst(strategy.spec.columns)
    assert worst == 0.0, f"{worst_column}={worst:.3e}"


def test_a_broken_diff_never_takes_the_trading_session_down_with_it(tmp_path):
    """A watcher that can kill the thing it watches is a liability, not a safeguard.

    This process is holding a position and has a signal ring the Rust core is reading.
    Raising out of the diff would abandon both with no flatten — so a diff that fails is
    counted, named on stdout and in the transcript, and stepped over. What to do about a
    parity break is the operator's call; ``axon --flatten`` is how they act on it.
    """

    class Exploding:
        def observe(self, bar, row):
            raise RuntimeError("the recompute blew up")

    strategy = Baseline(BaselineParams(symbol_id=SYMBOL, max_position=Decimal("0.0003")))
    runner = LiveRunner(
        strategy,
        symbol_id=SYMBOL,
        signal_ring=str(tmp_path / "sig.ring"),
        model_version=NO_MODEL_VERSION,
        parity=Exploding(),
    )
    ts = _warm(runner)
    for i in range(4):
        runner.on_bar(_bar(ts + i * 60_000_000_000, 6_400_000_000_000 - (i + 1) * 8_000_000))
    runner.flush()
    runner.close()

    assert runner.stats.bars_seen > 0, "the strategy kept running"
    assert runner.parity_alarms > 0
    assert "the recompute blew up" in (runner.parity_last_reason or "")
    # And the trading half is untouched: records still reached the ring.
    assert RingConsumer(str(tmp_path / "sig.ring")).read_batch().size > 0
