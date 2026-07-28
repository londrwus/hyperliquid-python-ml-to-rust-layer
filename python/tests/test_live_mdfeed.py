"""The live bridge's missing half: the market-data ring as strategy events.

``test_live_bridge.py`` covers the runner over a canned event iterable, and
``test_md_ring.py`` covers the ring's bytes. Between them sat the join nothing
tested and nothing used — turning slices into events a strategy is driven by — which
is why the Python→Rust direction had only ever been exercised by a synthetic tick
generator whose timestamps no live core would admit.

Every test here is named after the failure it prevents, and each of those failures
is silent: a probe that never decides, a decision stamped from the wrong clock, a
target computed against a market at zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.contracts import (
    MD_KIND_QUOTE,
    MD_KIND_SNAPSHOT,
    MD_KIND_TRADE,
    MD_SLICE_DTYPE,
    new_md_slice,
    to_fixed,
)
from axon.live.mdfeed import MdFeedError, MdRingFeed
from axon.live.probe import DONE, HOLDING, WARMING, TargetProbe
from axon.live.runner import StrategyRunner
from axon.signals import RingProducer

MS = 1_000_000
#: A plausible 2026 venue timestamp. Deliberately not `time.time_ns()`: a test whose
#: expectations move with the wall clock is a test that passes for the wrong reason.
T0 = 1_785_000_000_000_000_000


def _slice(seq, ts, bid, ask, *, symbol_id=3, kind=MD_KIND_QUOTE):
    return new_md_slice(
        seq=seq,
        ts_event=ts,
        bid_px=to_fixed(bid),
        bid_sz=to_fixed(1),
        ask_px=to_fixed(ask),
        ask_sz=to_fixed(1),
        symbol_id=symbol_id,
        kind=kind,
    )


@pytest.fixture
def md_ring(tmp_path):
    """A market-data ring a test can write into, standing in for the Rust core."""
    return str(tmp_path / "md.ring")


def _publish(path, records, capacity=64):
    producer = RingProducer(path, capacity=capacity, dtype=MD_SLICE_DTYPE)
    for rec in records:
        assert producer.try_push(rec)
    return producer


def test_an_event_carries_the_venues_time_and_never_the_readers(md_ring):
    """The whole reason this module exists.

    ``StrategyContext`` stamps a signal with the event's time and the Rust
    ``SignalReader`` ages that stamp against the core's own event clock. An event
    timestamped when Python happened to read the ring is admitted or refused for
    reasons that have nothing to do with how stale the decision is.
    """
    producer = _publish(md_ring, [_slice(0, T0, 100, 101), _slice(1, T0 + 5 * MS, 100, 102)])
    with MdRingFeed(md_ring, symbol_id=3) as feed:
        events = list(feed.events(max_events=2))
    producer.close()

    assert [e.ts_event for e in events] == [T0, T0 + 5 * MS]
    assert [e.bid_px for e in events] == [to_fixed(100), to_fixed(100)]
    assert [e.ask_px for e in events] == [to_fixed(101), to_fixed(102)]


def test_a_slice_with_no_quote_is_skipped_rather_than_read_as_a_market_at_zero(md_ring):
    """The publisher zeroes all four quote fields rather than republish a top of book
    it would not price an order against. Passed on as a price, that is a mid of zero —
    and a strategy differencing against it emits a target the size of the whole book."""
    blank = new_md_slice(seq=1, ts_event=T0 + MS, symbol_id=3, kind=MD_KIND_QUOTE)
    producer = _publish(md_ring, [_slice(0, T0, 100, 101), blank, _slice(2, T0 + 2 * MS, 99, 100)])
    with MdRingFeed(md_ring, symbol_id=3) as feed:
        events = list(feed.events(max_events=2))
        assert feed.stats.no_quote == 1
    producer.close()

    assert [e.ts_event for e in events] == [T0, T0 + 2 * MS], "the blank one was not passed on"


def test_one_symbols_feed_never_interleaves_another_symbols_book(md_ring):
    """A single-symbol strategy handed two instruments through one set of callbacks
    computes every statistic across both books, and the number it emits is a target
    for whichever id happened to be last."""
    producer = _publish(
        md_ring,
        [
            _slice(0, T0, 100, 101, symbol_id=3),
            _slice(1, T0 + MS, 2_000, 2_001, symbol_id=4),
            _slice(2, T0 + 2 * MS, 100, 102, symbol_id=3),
        ],
    )
    with MdRingFeed(md_ring, symbol_id=3) as feed:
        events = list(feed.events(max_events=2))
        assert feed.stats.other_symbol == 1
    producer.close()

    assert all(e.symbol_id == 3 for e in events)


def test_a_trade_slice_is_not_republished_as_a_trade_event(md_ring):
    """``MdSlice`` carries the *last* print, not each print, so a quote move
    republishes the same execution on the next slice. One ``Trade`` per slice counts
    each execution as many times as the book moved afterwards."""
    producer = _publish(
        md_ring,
        [
            _slice(0, T0, 100, 101, kind=MD_KIND_TRADE),
            _slice(1, T0 + MS, 100, 101, kind=MD_KIND_SNAPSHOT),
        ],
    )
    with MdRingFeed(md_ring, symbol_id=3) as feed:
        events = list(feed.events(max_events=1))
        assert feed.stats.other_kind == 1, "the trade-kind slice yielded no event"
    producer.close()

    # The snapshot did, because a book refresh *is* a statement about the top of book.
    assert len(events) == 1 and events[0].ts_event == T0 + MS


def test_a_ring_that_never_appears_is_reported_rather_than_waited_on_forever(tmp_path):
    """A driver blocked forever on a ring the session was never configured to publish
    reads exactly like a quiet market — for as long as anybody is willing to watch."""
    with pytest.raises(MdFeedError, match="md_ring"):
        MdRingFeed.wait_for_ring(str(tmp_path / "never.ring"), timeout_s=0.2, poll_s=0.02)


def test_a_silent_ring_ends_the_generator_instead_of_hanging(md_ring):
    """A dead feed and a genuinely quiet market are the same silence. Unbounded, the
    caller never returns and never says why."""
    producer = _publish(md_ring, [_slice(0, T0, 100, 101)])
    with MdRingFeed(md_ring, symbol_id=3, poll_s=0.001) as feed:
        events = list(feed.events(idle_timeout_s=0.05))
    producer.close()
    assert len(events) == 1


# ── the probe ────────────────────────────────────────────────────────────────


def _quotes(n, *, start=T0, step_ns=MS, symbol_id=3):
    from axon.strategy.events import Bbo

    return [
        Bbo(
            symbol_id=symbol_id,
            ts_event=start + i * step_ns,
            bid_px=to_fixed(100),
            bid_sz=to_fixed(1),
            ask_px=to_fixed(101),
            ask_sz=to_fixed(1),
        )
        for i in range(n)
    ]


def _run_probe(tmp_path, probe, events):
    runner = StrategyRunner(probe, ring_path=str(tmp_path / "sig.ring"), capacity=64)
    try:
        runner.run(events)
        return runner.stats
    finally:
        runner.close()


def test_the_probe_emits_nothing_before_the_core_could_have_a_book(tmp_path):
    """A core that has received no market data skips the intent pass entirely, so a
    signal emitted into that window is consumed, aged and refused — the strategy's
    first target thrown away with nothing to show for it."""
    probe = TargetProbe(3, size="0.0002", hold_ms=10, warmup=5)
    stats = _run_probe(tmp_path, probe, _quotes(4))
    assert stats.signals_emitted == 0
    assert probe.state == WARMING
    assert probe.decisions == []


def test_the_probe_holds_for_event_time_and_not_for_elapsed_time(tmp_path):
    """Paced on a wall clock, the flatten would land at a moment that depends on how
    fast this machine drained the ring — so a replay of the same market would produce
    a different pair of orders, which is exactly the pair the parity harness diffs."""
    probe = TargetProbe(3, size="0.0002", hold_ms=8, warmup=2)
    # Twelve quotes a millisecond apart: the open at the 2nd, the close 8 ms later.
    _run_probe(tmp_path, probe, _quotes(12))

    assert [d["decision"] for d in probe.decisions] == ["open", "close"]
    open_d, close_d = probe.decisions
    assert close_d["ts_event"] - open_d["ts_event"] == 8 * MS
    assert probe.state == DONE


def test_the_probe_flattens_by_close_rather_than_by_a_zero_target(tmp_path):
    """A zero target is an opinion about the position, and one computed against a fill
    we have not been told about overshoots into the opposite side. ``FLAG_CLOSE``
    ignores ``target_qty`` and implies reduce-only (ADR-0014 §2)."""
    from axon.contracts import FLAG_CLOSE

    probe = TargetProbe(3, size="0.0002", hold_ms=2, warmup=1)
    _run_probe(tmp_path, probe, _quotes(6))

    assert probe.decisions[0]["flags"] == 0
    assert probe.decisions[0]["target_qty_fixed"] == 20_000
    assert probe.decisions[1]["flags"] == FLAG_CLOSE


def test_the_probe_emits_exactly_two_records_however_long_the_feed_runs(tmp_path):
    """One target and one flatten. A probe that kept emitting would be a strategy, and
    the account it left behind would not be this test's to reason about."""
    probe = TargetProbe(3, size="0.0002", hold_ms=2, warmup=1)
    stats = _run_probe(tmp_path, probe, _quotes(400))
    assert stats.signals_emitted == 2
    assert stats.signals_pushed == 2


def test_the_probe_takes_its_stamps_from_the_event_and_owns_no_clock(tmp_path):
    """The training/serving skew ``docs/03`` names as the #1 silent quality leak. The
    only way to rule it out is for the strategy to have no clock available at all —
    this asserts the decision times are event times to the nanosecond."""
    probe = TargetProbe(3, size="0.0002", hold_ms=3, warmup=2)
    events = _quotes(8)
    _run_probe(tmp_path, probe, events)
    stamped = {d["ts_event"] for d in probe.decisions}
    assert stamped <= {e.ts_event for e in events}


def test_a_warmup_of_zero_is_refused_because_it_would_decide_before_any_event(tmp_path):
    with pytest.raises(ValueError, match="warmup"):
        TargetProbe(3, warmup=0)


def test_the_cloid_the_probe_reports_is_the_one_the_rust_planner_will_mint(tmp_path):
    """The evidence chain's one joint. A `cloid` derived independently on both sides
    from the same record (ADR-0014 §5) is what turns "an order appeared at the venue"
    into "*this* Python decision became that order"; derived on one side and copied to
    the other it would prove only that copying works."""
    probe = TargetProbe(3, size="0.0002", hold_ms=2, warmup=1)
    _run_probe(tmp_path, probe, _quotes(6))
    d = probe.decisions[0]
    ts, seq, sym = d["ts_event"], d["seq"], d["symbol_id"]
    expected = (1 << 127) | ((ts & ((1 << 63) - 1)) << 64) | ((seq & 0xFFFFFFFF) << 32) | sym
    assert d["cloid"] == f"0x{expected:032x}"
    assert d["cloid"] != probe.decisions[1]["cloid"], "two decisions, two ids"


def test_the_probe_says_so_when_it_stopped_holding_an_unflattened_target(tmp_path):
    """The one outcome that must never be quiet: the open went out, the feed ended, and
    the position — if it filled — is no longer this process's to close."""
    probe = TargetProbe(3, size="0.0002", hold_ms=10_000, warmup=1)
    _run_probe(tmp_path, probe, _quotes(5))
    assert probe.state == HOLDING
    assert [d["decision"] for d in probe.decisions] == ["open"]


def test_the_feeds_slices_and_the_probes_decisions_join_end_to_end(md_ring, tmp_path):
    """The join itself: ring bytes in, signal-ring bytes out, and the record the Rust
    reader would see carries the venue's own event time."""
    from axon.signals import RingConsumer

    records = [_slice(i, T0 + i * MS, 100, 101) for i in range(12)]
    producer = _publish(md_ring, records)
    sig_path = str(tmp_path / "sig.ring")
    probe = TargetProbe(3, size="0.0002", hold_ms=5, warmup=3)
    runner = StrategyRunner(probe, ring_path=sig_path, capacity=64)
    try:
        with MdRingFeed(md_ring, symbol_id=3, poll_s=0.001) as feed:
            runner.run(feed.events(idle_timeout_s=0.05))
    finally:
        runner.close()
        producer.close()

    with RingConsumer(sig_path) as cons:
        out = []
        while (rec := cons.try_pop()) is not None:
            out.append(rec)
    assert len(out) == 2, "one target, one flatten"
    assert int(out[0]["ts_event"]) == T0 + 2 * MS, "the third slice's own time"
    assert int(out[0]["target_qty"]) == 20_000, "0.0002 BTC at the wire's 1e-8 scale"
    assert int(out[1]["ts_event"]) - int(out[0]["ts_event"]) == 5 * MS
    # `reserved` is gone — schema version 3 spent the last of it on `ts_cause` — so
    # `pad0` is the whole of the record's unwritten space, and a non-zero byte in it is
    # what the Rust reader refuses as an unversioned extension.
    assert np.all(out[0]["pad0"] == 0), "a non-zero pad byte is a rejection"
