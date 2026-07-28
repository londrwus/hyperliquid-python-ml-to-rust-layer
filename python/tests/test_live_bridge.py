"""The live bridge: strategy → ring, and how a dead peer is noticed.

Named after the failure modes they prevent, per
``crates/axon-execution/src/tracker.rs``. Offline and deterministic throughout:
the ring is a temp file, the consumer is in-process, and the only clock is an
injected counter so stall detection is not a race against the test runner.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from axon.contracts import SCHEMA_VERSION, SIGNAL_DTYPE
from axon.live import BackpressureError, LiveError, StrategyRunner, read_liveness
from axon.signals import RingConsumer
from axon.strategy import Bar, Side, Strategy, StrategyConfig, Tick, Trade

FIXED_TS = 1_700_000_000_123_456_789


class EmitEveryTick(Strategy):
    """Emits a distinct target on every tick, so nothing is coincidentally equal."""

    def __init__(self) -> None:
        self.n = 0

    def on_tick(self, tick: Tick, ctx) -> None:
        self.n += 1
        ctx.emit_target(tick.symbol_id, Decimal(self.n) / 1000)


class Silent(Strategy):
    """Never emits — the idle case a heartbeat has to distinguish from death."""

    def __init__(self) -> None:
        self.seen = 0

    def on_tick(self, tick: Tick, ctx) -> None:
        self.seen += 1


class RecordsEverything(Strategy):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_start(self, ctx) -> None:
        self.calls.append("start")

    def on_tick(self, tick, ctx) -> None:
        self.calls.append("tick")

    def on_trade(self, trade, ctx) -> None:
        self.calls.append("trade")

    def on_bar(self, bar, ctx) -> None:
        self.calls.append("bar")

    def on_stop(self, ctx) -> None:
        self.calls.append("stop")


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def advance_ms(self, ms: int) -> None:
        self.ns += ms * 1_000_000


def ticks(n: int, *, symbol_id: int = 1, start: int = FIXED_TS, step: int = 1_000_000):
    return [Tick(symbol_id, start + i * step, 60_000_00000000 + i) for i in range(n)]


def runner(tmp_path, strategy=None, **kwargs) -> StrategyRunner:
    kwargs.setdefault("capacity", 8)
    kwargs.setdefault("model_version", 3)
    return StrategyRunner(
        strategy if strategy is not None else EmitEveryTick(),
        ring_path=str(tmp_path / "signals.ring"),
        **kwargs,
    )


def drain(path: str) -> list:
    with RingConsumer(path) as cons:
        out = []
        while (rec := cons.try_pop()) is not None:
            out.append(rec)
        return out


# ── what lands on the ring ────────────────────────────────────────────────────


def test_records_reach_the_ring_byte_valid_and_in_order(tmp_path):
    r = runner(tmp_path, capacity=64)
    with r:
        r.run(ticks(20))
        path = str(tmp_path / "signals.ring")
        recs = drain(path)
    assert len(recs) == 20
    for i, rec in enumerate(recs):
        assert rec.dtype == SIGNAL_DTYPE
        assert len(rec.tobytes()) == 64
        assert int(rec["schema_version"]) == SCHEMA_VERSION
        assert int(rec["seq"]) == i
        assert int(rec["model_version"]) == 3
        assert int(rec["target_qty"]) == (i + 1) * 100_000  # 0.001 steps, fixed-point


def test_seq_is_gapless_on_the_ring_so_the_consumer_can_prove_nothing_was_lost(tmp_path):
    r = runner(tmp_path, capacity=256)
    with r:
        r.run(ticks(200))
        seqs = [int(rec["seq"]) for rec in drain(str(tmp_path / "signals.ring"))]
    assert seqs == list(range(200))


def test_ts_event_on_the_ring_is_the_events_own_time(tmp_path):
    r = runner(tmp_path, capacity=8)
    with r:
        r.run(ticks(3, start=FIXED_TS, step=7))
        recs = drain(str(tmp_path / "signals.ring"))
    assert [int(rec["ts_event"]) for rec in recs] == [FIXED_TS, FIXED_TS + 7, FIXED_TS + 14]


# ── backpressure ──────────────────────────────────────────────────────────────


def test_a_full_ring_queues_instead_of_dropping(tmp_path):
    # Capacity 4, nothing consuming: 4 land on the ring and the rest wait in the
    # outbox. Dropping them would leave the venue holding the *previous* target
    # with no correction ever coming.
    path = str(tmp_path / "signals.ring")
    r = runner(tmp_path, capacity=4, max_outbox=64)
    with r:
        r.run(ticks(6))
        assert r.ring_depth == 4
        assert r.pending_out == 2
        assert r.stats.backpressure_events > 0
        assert [int(x["seq"]) for x in drain(path)] == [0, 1, 2, 3]
        # Space freed: the queue empties in the order it was formed.
        assert r.flush() == 2
        rest = drain(path)
    assert [int(x["seq"]) for x in rest] == [4, 5]


def test_an_overflowing_outbox_raises_and_still_holds_every_record(tmp_path):
    path = str(tmp_path / "signals.ring")
    r = runner(tmp_path, capacity=4, max_outbox=4)
    with r:
        with pytest.raises(BackpressureError):
            r.run(ticks(20))
        emitted = r.stats.signals_emitted
        assert r.pending_out == emitted - r.stats.signals_pushed  # nothing vanished
        seen: list[int] = []
        while r.pending_out:
            seen += [int(rec["seq"]) for rec in drain(path)]
            r.flush()
        seen += [int(rec["seq"]) for rec in drain(path)]
    assert seen == list(range(emitted))


def test_backpressure_never_reorders_the_sequence(tmp_path):
    r = runner(tmp_path, capacity=2, max_outbox=1024)
    path = str(tmp_path / "signals.ring")
    seen: list[int] = []
    with r:
        r.start(FIXED_TS)
        for tick in ticks(50):
            r.handle(tick)
            seen += [int(rec["seq"]) for rec in drain(path)]
        r.flush()
        seen += [int(rec["seq"]) for rec in drain(path)]
    assert seen == list(range(50))


# ── the heartbeat ─────────────────────────────────────────────────────────────


def test_the_heartbeat_advances_even_when_nothing_is_emitted(tmp_path):
    # The whole reason the heartbeat exists: a target-position strategy is
    # silent by design, so an empty ring must not read as a dead process.
    strategy = Silent()
    r = runner(tmp_path, strategy)
    with r:
        r.run(ticks(5))
        snap = read_liveness(r.beacon_path)
    assert strategy.seen == 5
    assert snap.signals == 0
    assert snap.beats >= 5


def test_the_beacon_reports_the_last_event_time_in_event_time(tmp_path):
    r = runner(tmp_path, capacity=64)
    with r:
        r.run(ticks(4, start=FIXED_TS, step=1_000))
        snap = read_liveness(r.beacon_path)
    assert snap.last_event_ns == FIXED_TS + 3_000
    assert snap.pid > 0


def test_a_clean_stop_is_distinguishable_from_a_crash(tmp_path):
    r = runner(tmp_path, capacity=64)
    with r:
        r.run(ticks(3))
        stopped = read_liveness(r.beacon_path)
    assert stopped.stopped and not stopped.running

    crashed = runner(tmp_path, capacity=64)  # a fresh beacon, never stopped
    with crashed:
        crashed.run(ticks(3), stop=False)
        snap = read_liveness(crashed.beacon_path)
    assert snap.running and not snap.stopped


def test_the_beacon_counts_backpressure_so_a_supervisor_sees_it(tmp_path):
    r = runner(tmp_path, capacity=2, max_outbox=1024)
    with r:
        r.run(ticks(10))
        snap = read_liveness(r.beacon_path)
    assert snap.backpressure > 0
    assert snap.pending > 0


# ── detecting a dead consumer ─────────────────────────────────────────────────


def test_a_consumer_that_stops_draining_is_reported_stalled(tmp_path):
    clock = FakeClock()
    r = runner(tmp_path, capacity=8, peer_timeout_ms=2_000, monotonic_ns=clock)
    with r:
        r.run(ticks(4), stop=False)
        assert r.ring_depth == 4
        assert not r.peer_stalled()  # backlog, but not yet old enough
        clock.advance_ms(1_999)
        assert not r.peer_stalled()
        clock.advance_ms(1)
        assert r.peer_stalled()


def test_an_idle_consumer_with_an_empty_ring_is_not_reported_dead(tmp_path):
    # Reporting idle as dead would trip the flatten-everything response on a
    # quiet market, which is worse than the failure it is guarding against.
    clock = FakeClock()
    r = runner(tmp_path, Silent(), capacity=8, peer_timeout_ms=1, monotonic_ns=clock)
    with r:
        r.run(ticks(4))
        clock.advance_ms(10_000)
        assert r.ring_depth == 0
        assert not r.peer_stalled()


def test_a_consumer_that_is_draining_is_not_reported_stalled(tmp_path):
    # Backlog the whole time, so "not stalled" can only come from the tail moving.
    clock = FakeClock()
    r = runner(tmp_path, capacity=4, max_outbox=64, peer_timeout_ms=1_000, monotonic_ns=clock)
    path = str(tmp_path / "signals.ring")
    with r:
        r.run(ticks(6), stop=False)
        clock.advance_ms(900)
        drain(path)  # the peer consumed: tail moved
        r.flush()
        assert r.ring_depth > 0
        clock.advance_ms(900)
        assert not r.peer_stalled()


def test_the_consumer_tail_is_derived_without_reaching_into_the_ring(tmp_path):
    r = runner(tmp_path, capacity=8)
    path = str(tmp_path / "signals.ring")
    with r:
        r.run(ticks(3), stop=False)
        assert r.consumer_tail == 0
        drain(path)
        assert r.consumer_tail == 3


# ── dispatch + lifecycle ──────────────────────────────────────────────────────


def test_every_event_type_reaches_its_own_callback(tmp_path):
    strategy = RecordsEverything()
    r = runner(tmp_path, strategy, capacity=8)
    with r:
        r.start(FIXED_TS)
        r.handle(Tick(1, FIXED_TS, 1))
        r.handle(Trade(1, FIXED_TS + 1, 1, 1, Side.BUY))
        r.handle(Bar(1, FIXED_TS + 2, 1, 1, 1, 1, 1))
        r.stop(FIXED_TS + 3)
    assert strategy.calls == ["start", "tick", "trade", "bar", "stop"]


def test_an_unroutable_event_type_fails_loudly(tmp_path):
    class Mystery:
        ts_event = FIXED_TS

    r = runner(tmp_path)
    with r:
        r.start(FIXED_TS)
        with pytest.raises(TypeError, match="no strategy callback"):
            r.handle(Mystery())


def test_handling_before_start_is_refused(tmp_path):
    r = runner(tmp_path)
    with r:
        with pytest.raises(LiveError, match="start"):
            r.handle(Tick(1, FIXED_TS, 1))


def test_events_that_go_backwards_in_time_are_counted_not_hidden(tmp_path):
    r = runner(tmp_path, capacity=64)
    with r:
        r.start(FIXED_TS)
        r.handle(Tick(1, FIXED_TS + 10, 1))
        r.handle(Tick(1, FIXED_TS + 5, 1))  # late arrival
        assert r.stats.out_of_order_events == 1
        assert r.stats.events_handled == 2


def test_stats_account_for_every_emitted_signal(tmp_path):
    r = runner(tmp_path, capacity=4, max_outbox=1024)
    with r:
        stats = r.run(ticks(10))
    assert stats.events_handled == 10
    assert stats.signals_emitted == 10
    assert stats.signals_pushed == 4
    assert stats.signals_in_flight == 6
    assert stats.max_outbox_depth == 6


def test_the_config_supplies_the_stamping_so_call_sites_cannot_disagree(tmp_path):
    config = StrategyConfig(name="ref", model_version=99, symbols=(1,), default_ttl_ms=250)
    r = StrategyRunner.from_config(
        EmitEveryTick(), config, ring_path=str(tmp_path / "signals.ring"), capacity=8
    )
    with r:
        r.run(ticks(2))
        recs = drain(str(tmp_path / "signals.ring"))
    assert [int(rec["model_version"]) for rec in recs] == [99, 99]
    assert [int(rec["ttl_ms"]) for rec in recs] == [250, 250]
