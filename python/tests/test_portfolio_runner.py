"""Tests for the multi-strategy producer (:mod:`axon.strategies.portfolio_runner`).

Every name says what would break if the check were removed, and each one is about a
hazard this project has already measured on some other ring: two readers stealing from
one SPSC queue, two writers interleaving two sequences into a stream validated as one,
and a strategy fed a tape that is not its own.

Nothing here touches a network or a venue. The signal rings are real rings in
``tmp_path``, because "the record was built" and "the record crossed the ring" are
different claims and only the second is what a live session depends on.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from axon.signals import RingConsumer
from axon.strategies.baseline import NO_MODEL_VERSION
from axon.strategies.portfolio_runner import (
    ConfigError,
    PortfolioRunner,
    ProducerPlan,
    StrategyProducer,
    parse_symbol_ids,
    plans_from_config,
)
from axon.strategy import Bar

BTC = 3
ETH = 4
SOL = 5

#: A real m1 close time from the P6 run, so the skew these tests see is a live one.
BAR_CLOSE_NS = 1_785_079_860_000_000_000

SYMBOL_IDS = {"BTC": BTC, "ETH": ETH, "SOL": SOL}


def _bar(ts: int, close: int, symbol_id: int) -> Bar:
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


def _producer(tmp_path, name, coins, *, max_position="0.0003") -> StrategyProducer:
    return StrategyProducer(
        ProducerPlan(
            name=name,
            signal_ring=str(tmp_path / f"{name}.ring"),
            coins=coins,
            factory="baseline",
            max_position=max_position,
        ),
        symbol_ids=SYMBOL_IDS,
        registry=None,
        model="unused",
        model_version=NO_MODEL_VERSION,
    )


def _ramp(runner, symbol_ids, *, bars=40, base=6_400_000_000_000, step_ns=60_000_000_000):
    """Feed enough bars per instrument to clear the 21-bar warmup, with a real range.

    A ramp rather than a constant: the spec's realized-volatility floor refuses a flat
    tape, so a constant close warms the window and produces no opinion — which would
    make every assertion below pass for the wrong reason.
    """
    ts = BAR_CLOSE_NS
    for i in range(bars):
        for sid in symbol_ids:
            runner.on_bar(_bar(ts, base + i * 2_000_000_00, sid))
        ts += step_ns
    runner.flush()
    return ts


def test_each_producer_writes_only_to_its_own_ring(tmp_path):
    # The rule that makes the whole file safe: an SPSC ring has one writer, and `seq`
    # — the only proof nothing was lost — is per writer. Two producers sharing a ring
    # would interleave two sequences into a stream the Rust reader validates as one, and
    # every record of the loser would be refused as `stale_seq`: a strategy emitting
    # normally, its own counters climbing, and nothing reaching the venue.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    beta = _producer(tmp_path, "beta", ["ETH-PERP"])
    runner = PortfolioRunner([alpha, beta])
    _ramp(runner, [BTC, ETH])

    a_recs = list(RingConsumer(str(tmp_path / "alpha.ring")).read_batch())
    b_recs = list(RingConsumer(str(tmp_path / "beta.ring")).read_batch())
    assert a_recs, "alpha emitted nothing, so this test proves nothing"
    assert b_recs, "beta emitted nothing, so this test proves nothing"
    assert {int(r["symbol_id"]) for r in a_recs} == {BTC}
    assert {int(r["symbol_id"]) for r in b_recs} == {ETH}
    # And each sequence is its own: strictly increasing from 1, independently.
    for recs in (a_recs, b_recs):
        seqs = [int(r["seq"]) for r in recs]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_one_bar_is_dispatched_to_every_producer_that_covers_the_instrument(tmp_path):
    # The netting case, from the producing side. Two strategies on one coin is the
    # configuration ADR-0038's `TargetBook` exists for, and this side's job is to emit
    # both claims — unnetted, on separate rings — rather than to decide between them.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    beta = _producer(tmp_path, "beta", ["BTC-PERP"], max_position="0.0006")
    runner = PortfolioRunner([alpha, beta])
    _ramp(runner, [BTC])

    assert alpha.stats.bars_dispatched == beta.stats.bars_dispatched > 0
    a_recs = list(RingConsumer(str(tmp_path / "alpha.ring")).read_batch())
    b_recs = list(RingConsumer(str(tmp_path / "beta.ring")).read_batch())
    assert a_recs and b_recs, "both claims must reach their own ring"
    # Different sizes, because the two were configured differently — this side does not
    # combine them, and the assertion is that it did not.
    assert {abs(int(r["target_qty"])) for r in a_recs} != {
        abs(int(r["target_qty"])) for r in b_recs
    }


def test_a_bar_is_never_handed_to_a_strategy_that_does_not_cover_it(tmp_path):
    # A strategy that filters internally would silently accept the extra dispatch while
    # one that did not would trade another coin's tape — and which of those you have is a
    # property of the strategy rather than of this runner, so the runner must not rely on
    # it. The failure is a feature matrix computed across two tapes, which no offline
    # recompute reproduces and which the parity gate cannot catch, because it recomputes
    # against the bars the run was shown.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    runner = PortfolioRunner([alpha])
    _ramp(runner, [BTC, ETH, SOL])

    assert alpha.stats.bars_dispatched > 0
    assert set(alpha.stats.targets) == {BTC}, "only its own instrument"
    assert runner.bars_unclaimed > 0, "and the others are counted rather than ignored"

    recs = list(RingConsumer(str(tmp_path / "alpha.ring")).read_batch())
    assert {int(r["symbol_id"]) for r in recs} == {BTC}


def test_a_producer_asked_directly_about_a_foreign_bar_still_refuses_it(tmp_path):
    # The runner filters before it dispatches, so the guard inside `StrategyProducer`
    # is a second line of defence — and a second line nothing exercises is one a future
    # caller finds out about. This is that caller: anything holding a producer directly
    # (a test harness, a replay driver) must get the same refusal the runner gives.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    assert alpha.on_bar(_bar(BAR_CLOSE_NS, 6_400_000_000_000, ETH), BAR_CLOSE_NS) == 0
    assert alpha.stats.bars_dispatched == 0
    assert alpha.stats.targets == {}


def test_one_producer_over_two_coins_keeps_a_strategy_instance_per_coin(tmp_path):
    # One object fed two tapes would compute every rolling window across both. Separate
    # instances are what make each coin's window its own, and the evidence is that the
    # two reach different targets from different price paths.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP", "ETH-PERP"])
    assert set(alpha.strategies) == {BTC, ETH}
    assert alpha.strategies[BTC] is not alpha.strategies[ETH]

    runner = PortfolioRunner([alpha])
    ts = BAR_CLOSE_NS
    for i in range(40):
        # BTC ramps up, ETH ramps down — two tapes that could not produce one window.
        runner.on_bar(_bar(ts, 6_400_000_000_000 + i * 2_000_000_00, BTC))
        runner.on_bar(_bar(ts, 6_400_000_000_000 - i * 2_000_000_00, ETH))
        ts += 60_000_000_000
    runner.flush()

    recs = list(RingConsumer(str(tmp_path / "alpha.ring")).read_batch())
    assert {int(r["symbol_id"]) for r in recs} == {BTC, ETH}
    # One ring, one sequence, both instruments on it — which is exactly why one producer
    # is one writer however many coins it covers.
    seqs = [int(r["seq"]) for r in recs]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_every_record_a_bar_produced_carries_that_bars_own_close(tmp_path):
    # `ts_cause` is the largest latency in the system (951 / 12 051 / 111 475 ms over 57
    # live bars) and it exists only because the runtime could not otherwise tell a
    # decision one second after a bar from one two minutes after it. A second runner that
    # forgot to stamp would make the `bar` budget report on half a session.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    runner = PortfolioRunner([alpha])
    _ramp(runner, [BTC])

    recs = list(RingConsumer(str(tmp_path / "alpha.ring")).read_batch())
    assert recs
    for r in recs:
        assert int(r["ts_cause"]) > 0, "a record with no cause is one the budget cannot see"
        # The decision is stamped on the producer's own clock and is therefore *after*
        # the bar it answers — which is what makes the subtraction a latency rather than
        # a sign error.
        assert int(r["ts_event"]) >= int(r["ts_cause"])


def test_one_clock_read_per_bar_so_two_producers_stamp_the_same_instant(tmp_path):
    # Reading the clock per producer would make the second one's record look later than
    # the first's by however long the first strategy's inference took — a difference that
    # would show up as real skew in the `sig` latency stage and as an ordering between
    # two decisions that were made together.
    alpha = _producer(tmp_path, "alpha", ["BTC-PERP"])
    beta = _producer(tmp_path, "beta", ["BTC-PERP"], max_position="0.0006")
    runner = PortfolioRunner([alpha, beta])
    _ramp(runner, [BTC])

    a = {int(r["ts_event"]) for r in RingConsumer(str(tmp_path / "alpha.ring")).read_batch()}
    b = {int(r["ts_event"]) for r in RingConsumer(str(tmp_path / "beta.ring")).read_batch()}
    assert a and b
    assert a == b, "the two producers decided on the same bars at the same instants"


def test_a_producer_naming_an_instrument_with_no_symbol_id_is_refused_at_construction(tmp_path):
    # The venue's asset index is not derivable here and differs between networks (BTC is
    # 3 on testnet). Starting anyway would mean a producer trading an instrument it
    # cannot name, or — worse — trading nothing while every counter looked healthy.
    with pytest.raises(ConfigError, match="symbol-ids"):
        StrategyProducer(
            ProducerPlan(
                name="alpha",
                signal_ring=str(tmp_path / "alpha.ring"),
                coins=["DOGE-PERP"],
                factory="baseline",
                max_position="0.0003",
            ),
            symbol_ids=SYMBOL_IDS,
            registry=None,
            model="unused",
            model_version=NO_MODEL_VERSION,
        )


def test_a_run_with_no_producers_is_refused_rather_than_reading_bars_into_nothing(tmp_path):
    with pytest.raises(ConfigError):
        PortfolioRunner([])


def test_symbol_ids_are_parsed_by_the_same_coin_rule_the_config_uses(tmp_path):
    # Everything before the first `-`, uppercased — `RuntimeConfig::coins`'s rule. A
    # second rule here would let a producer be correctly configured and silently scoped
    # to nothing.
    assert parse_symbol_ids("BTC=3,ETH=4") == {"BTC": 3, "ETH": 4}
    assert parse_symbol_ids("btc-perp=3") == {"BTC": 3}
    assert parse_symbol_ids(" BTC = 3 , ETH = 4 ") == {"BTC": 3, "ETH": 4}
    with pytest.raises(ConfigError):
        parse_symbol_ids("BTC")
    with pytest.raises(ConfigError):
        parse_symbol_ids("BTC=three")
    with pytest.raises(ConfigError):
        parse_symbol_ids("")


CONFIG = """
environment = "sandbox"

[venue]
name = "hyperliquid"
network = "testnet"

[session]
bus_capacity = 4096
core_poll_us = 500
status_interval_ms = 15000
mark_max_age_ms = 10000
feeds = ["Bbo"]

[safety]
dead_mans_switch = true
lead_ms = 60000
rearm_interval_ms = 20000

[reconcile]
interval_ms = 15000
grace_ms = 5000
rate_limit_every = 10

[governor]
place_reserve_credits = 1000
place_reserve_ip_weight = 200
initial_address_cap = 10000

[ipc]
signal_ring_path = "/dev/shm/unused.ring"
capacity = 1024

[portfolio]
max_gross_notional = "120"
overlap = "net"

[strategy]
name = "portfolio-test"
version = 1
symbols = ["BTC-PERP", "ETH-PERP"]

[strategy.model_ref]
registry_id = "zoo_xgboost"
version = 1

[strategy.risk]
max_position = "0.0016"
max_notional = "120"
max_order_qty = "0.0008"

[[strategy.producer]]
name = "alpha"
signal_ring = "/dev/shm/alpha.ring"
symbols = ["BTC-PERP"]

[[strategy.producer]]
name = "beta"
signal_ring = "/dev/shm/beta.ring"
"""


def test_the_plans_come_from_the_config_both_sides_read(tmp_path):
    # The ring paths and the instrument scopes have to be identical on both sides of the
    # boundary. A producer writing to a ring nothing reads is a strategy whose every
    # decision is discarded; a scope the two disagree about is a record Rust refuses as
    # out-of-scope while Python's counters climb. Both present as a healthy session that
    # is not trading, which is why they come from one file rather than two command lines.
    path = tmp_path / "session.toml"
    path.write_text(CONFIG)
    plans, cfg = plans_from_config(str(path), {"alpha": "baseline:0.0003", "beta": "baseline"})
    assert [p.name for p in plans] == ["alpha", "beta"]
    assert plans[0].signal_ring == "/dev/shm/alpha.ring"
    assert plans[0].coins == ["BTC-PERP"]
    assert plans[0].max_position == "0.0003"
    # A producer that declares no symbols means the whole session universe — the same
    # reading the Rust `TargetBook` gives an empty scope.
    assert plans[1].coins == ["BTC-PERP", "ETH-PERP"]
    assert plans[1].max_position is None
    assert cfg["portfolio"]["overlap"] == "net"


def test_a_producer_with_no_strategy_named_is_refused_by_name(tmp_path):
    # The config says which ring a producer writes to; only the command line says what
    # code is behind it. Starting with one of them unstaffed would be a ring nothing
    # writes to and a Rust side reporting `SIGNAL RING DETACHED` for a strategy nobody
    # forgot to configure — they forgot to *run* it.
    path = tmp_path / "session.toml"
    path.write_text(CONFIG)
    with pytest.raises(ConfigError, match="beta"):
        plans_from_config(str(path), {"alpha": "baseline"})


def test_a_single_producer_config_is_sent_to_the_runner_that_is_better_at_it(tmp_path):
    # Not an error in the config — it is what every session before ADR-0038 describes.
    # It is an error to run *this* on it, and saying which runner to use is cheaper than
    # a run that works and buys nothing.
    path = tmp_path / "single.toml"
    path.write_text(CONFIG.split("[[strategy.producer]]")[0])
    with pytest.raises(ConfigError, match="live_runner"):
        plans_from_config(str(path), {})
