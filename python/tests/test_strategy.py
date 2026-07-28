"""The strategy contract: what a strategy author must not be able to get wrong.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic — no clock, no network, no ring.
"""

from __future__ import annotations

import time
from decimal import Decimal

import numpy as np
import pytest

from axon.contracts import (
    FLAG_CLOSE,
    FLAG_REDUCE_ONLY,
    KIND_TARGET_POSITION,
    SCHEMA_VERSION,
    SIGNAL_DTYPE,
    from_fixed,
    new_signal,
)
# Imported from the package, not from ``axon.strategy.context``: these tests are the
# check that a strategy author never has to know which submodule a boundary constant
# happens to live in today.
from axon.strategy import (
    DEFAULT_TTL_MS,
    TTL_OPERATOR_CEILING,
    URGENCY_CROSS,
    URGENCY_JOIN,
    URGENCY_MAX,
    URGENCY_POST_ONLY,
    URGENCY_TAKE,
    Bar,
    MeanReversion,
    MeanReversionParams,
    NotInEventScope,
    Side,
    Strategy,
    StrategyConfig,
    StrategyContext,
    StrategyError,
    Tick,
    Trade,
)

# A timestamp far from any plausible wall clock during the test run, so "did this
# come from the event?" is answerable by looking at the value alone.
FIXED_TS = 1_700_000_000_123_456_789


def ctx(**kwargs) -> StrategyContext:
    kwargs.setdefault("model_version", 7)
    return StrategyContext(**kwargs)


def emit_one(**emit_kwargs) -> np.ndarray:
    c = ctx()
    with c.event(FIXED_TS):
        c.emit_target(3, Decimal("0.25"), **emit_kwargs)
    return c.take_pending()[0]


# ── the record on the wire ────────────────────────────────────────────────────


def test_emitted_record_is_byte_valid_against_the_signal_dtype():
    rec = emit_one()
    assert rec.dtype == SIGNAL_DTYPE
    assert rec.nbytes == 64
    assert len(rec.tobytes()) == 64
    assert int(rec["schema_version"]) == SCHEMA_VERSION
    assert int(rec["kind"]) == KIND_TARGET_POSITION
    # A non-zero pad would be read as a field by a future schema. `reserved` is gone —
    # schema version 3 spent the last of it on `ts_cause`, so `pad0` is the whole of the
    # record's unwritten space now.
    # …and the named padding too: the Rust reader refuses a record with either dirty,
    # so a producer that left one would have every signal rejected at the far end.
    assert bytes(rec["pad0"]) == b"\x00" * 3


def test_an_order_lifetime_defaults_to_the_operators_ceiling_not_to_forever():
    # `max_order_age_ms` is not `ttl_ms` (ADR-0031): `ttl_ms` is a signal *admission*
    # window the Rust reader consumes before the planner sees the record, so a large
    # one buys a resting order nothing. Zero on either means "use the operator's
    # policy" — the reading both sides already agreed on for a field nobody wrote.
    assert int(emit_one()["max_order_age_ms"]) == 0
    rec = emit_one(max_order_age_ms=5_000)
    assert int(rec["max_order_age_ms"]) == 5_000
    assert int(rec["ttl_ms"]) == DEFAULT_TTL_MS, "the admission window is untouched"
    # The error has to name the field the caller passed, not the other duration of the
    # same shape sitting next to it.
    with pytest.raises(ValueError, match="max_order_age_ms"):
        emit_one(max_order_age_ms=-1)


def test_model_version_is_stamped_on_every_signal():
    assert int(emit_one()["model_version"]) == 7


def test_a_model_version_of_zero_is_refused_as_indistinguishable_from_unset():
    with pytest.raises(ValueError, match="model_version"):
        StrategyContext(model_version=0)


def test_ttl_defaults_instead_of_leaving_the_field_zero():
    assert int(emit_one()["ttl_ms"]) == DEFAULT_TTL_MS
    assert int(emit_one(ttl_ms=1500)["ttl_ms"]) == 1500


def test_a_zero_ttl_defers_to_the_operator_rather_than_meaning_two_things():
    # The two sides used to disagree about zero: Rust read it as "unset, use the
    # operator's ceiling" and Python refused to emit it at all. Refusing to emit it
    # never stopped it arriving — zero is the value of a field nobody wrote — so the
    # consumer's reading is the contract, and a producer that has no opinion about
    # staleness can now say exactly that (ADR-0020 §4).
    rec = emit_one(ttl_ms=TTL_OPERATOR_CEILING)
    assert int(rec["ttl_ms"]) == 0
    # And a whole run can defer, not just one signal.
    assert StrategyContext(model_version=1, default_ttl_ms=TTL_OPERATOR_CEILING).default_ttl_ms == 0


def test_a_negative_ttl_is_still_refused_because_it_is_not_a_duration():
    # Zero has a meaning; -1 has none, and the field is unsigned on the wire, so
    # letting it through would wrap into a 49-day validity window.
    with pytest.raises(ValueError, match="ttl_ms"):
        emit_one(ttl_ms=-1)
    with pytest.raises(ValueError, match="ttl_ms"):
        emit_one(ttl_ms=2**32)


def test_urgency_levels_are_named_so_both_sides_mean_the_same_thing():
    # A bare u8 is how a boundary drifts: an author picking "2 sounds about right"
    # cannot know that 2 crosses the spread. These names are the Rust planner's own
    # table (ADR-0014 §3), in the order it defines, so picking one is picking a
    # documented execution behaviour rather than a number.
    assert (URGENCY_POST_ONLY, URGENCY_JOIN, URGENCY_CROSS, URGENCY_TAKE) == (0, 1, 2, 3)
    assert URGENCY_MAX == URGENCY_TAKE
    assert int(emit_one(urgency=URGENCY_TAKE)["urgency"]) == 3
    # Above the table saturates at the far end rather than being refused: 255 means
    # "as fast as possible", and dropping it would drop the one signal that mattered.
    assert int(emit_one(urgency=255)["urgency"]) == 255


def test_reduce_only_and_close_reach_the_flags_field():
    assert int(emit_one(reduce_only=True)["flags"]) & FLAG_REDUCE_ONLY
    c = ctx()
    with c.event(FIXED_TS):
        c.emit_close(3)
    assert int(c.take_pending()[0]["flags"]) & FLAG_CLOSE


# ── event time, never the wall clock ──────────────────────────────────────────


def test_ts_event_comes_from_the_event_and_not_the_wall_clock():
    before = time.time_ns()
    rec = emit_one()
    after = time.time_ns()
    assert int(rec["ts_event"]) == FIXED_TS
    # If a wall clock had leaked in, the stamp would sit inside this window.
    assert not before <= int(rec["ts_event"]) <= after


def test_the_same_events_replay_to_byte_identical_signals():
    # The property the whole event-time rule exists for: two runs of the same
    # inputs must produce the same bytes, or a parity comparison is meaningless.
    runs = []
    for _ in range(2):
        c = ctx()
        for i, ts in enumerate((FIXED_TS, FIXED_TS + 5, FIXED_TS + 11)):
            with c.event(ts):
                c.emit_target(1, Decimal("0.5") * i)
        runs.append(b"".join(r.tobytes() for r in c.take_pending()))
    assert runs[0] == runs[1]


def test_emitting_outside_an_event_scope_is_refused():
    with pytest.raises(NotInEventScope):
        ctx().emit_target(1, 1.0)


def test_nested_event_scopes_are_refused_so_a_stale_time_cannot_leak():
    c = ctx()
    with c.event(FIXED_TS), pytest.raises(StrategyError, match="already open"):
        with c.event(FIXED_TS + 1):
            pass


def test_a_float_event_time_is_refused_because_float64_cannot_hold_nanoseconds():
    c = ctx()
    with pytest.raises(TypeError, match="ts_event"):
        with c.event(float(FIXED_TS)):
            pass


def test_the_raw_emit_escape_hatch_cannot_forge_time_or_sequence():
    c = ctx()
    forged = new_signal(seq=999, ts_event=1, model_version=42, target_qty=5)
    with c.event(FIXED_TS):
        rec = c.emit(forged)
    assert int(rec["ts_event"]) == FIXED_TS
    assert int(rec["seq"]) == 0
    assert int(rec["model_version"]) == 7
    assert int(rec["target_qty"]) == 5  # the payload itself is left alone
    assert int(forged["seq"]) == 999  # and the caller's array is not mutated


# ── sequence numbers ──────────────────────────────────────────────────────────


def test_seq_is_monotonic_and_gapless_across_events():
    c = ctx()
    for i in range(64):
        with c.event(FIXED_TS + i):
            c.emit_target(1, i)
    assert [int(r["seq"]) for r in c.take_pending()] == list(range(64))


def test_a_rejected_conversion_leaves_no_gap_in_the_sequence():
    # A gap would look to the consumer exactly like a lost signal.
    c = ctx()
    with c.event(FIXED_TS):
        c.emit_target(1, 1.0)
        with pytest.raises(ValueError):
            c.emit_target(1, float("nan"))
        c.emit_target(1, 2.0)
    assert [int(r["seq"]) for r in c.take_pending()] == [0, 1]


def test_seq_can_resume_after_a_restart():
    c = ctx(first_seq=1000)
    with c.event(FIXED_TS):
        c.emit_target(1, 1.0)
    assert int(c.take_pending()[0]["seq"]) == 1000
    assert c.next_seq == 1001


# ── real units in, fixed-point on the wire, converted once ────────────────────


def test_real_units_round_trip_through_the_fixed_point_conversion():
    for real, wire in (
        (Decimal("0.25"), 25_000_000),
        (Decimal("1.23456789"), 123_456_789),
        (Decimal("-2.5"), -250_000_000),
        (0.25, 25_000_000),
        (3, 300_000_000),
        ("0.00000001", 1),
    ):
        c = ctx()
        with c.event(FIXED_TS):
            rec = c.emit_target(1, real)
        assert int(rec["target_qty"]) == wire
        assert Decimal(str(from_fixed(int(rec["target_qty"])))) == Decimal(str(real))


def test_price_band_uses_the_same_scale_as_quantity():
    c = ctx()
    with c.event(FIXED_TS):
        rec = c.emit_target(1, Decimal("0.25"), price_band=Decimal("60123.5"))
    assert int(rec["price_band"]) == 6_012_350_000_000
    assert int(c.take_pending()[0]["price_band"]) == 6_012_350_000_000


def test_omitting_the_price_band_means_no_band_not_a_zero_price():
    assert int(emit_one()["price_band"]) == 0


def test_a_nan_prediction_never_becomes_a_target():
    with pytest.raises(ValueError, match="non-finite"):
        emit_one_qty(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        emit_one_qty(float("inf"))


def test_a_hand_scaled_quantity_is_refused_rather_than_silently_wrapped():
    # 1e12 "real" is what you get by pre-multiplying a 1e4 size by 1e8; scaled
    # again it overflows i64, and wrapping would flip the sign of the position.
    with pytest.raises(ValueError, match="fixed-point"):
        emit_one_qty(Decimal("10000000000000"))


def test_out_of_range_bookkeeping_fields_are_refused():
    c = ctx()
    with c.event(FIXED_TS):
        with pytest.raises(ValueError, match="symbol_id"):
            c.emit_target(2**32, 1.0)
        with pytest.raises(ValueError, match="urgency"):
            c.emit_target(1, 1.0, urgency=256)


def emit_one_qty(qty) -> np.ndarray:
    c = ctx()
    with c.event(FIXED_TS):
        return c.emit_target(3, qty)


# ── the base class + config ───────────────────────────────────────────────────


def test_the_base_class_ignores_every_event_it_does_not_override():
    s = Strategy()
    c = ctx()
    with c.event(FIXED_TS):
        s.on_start(c)
        s.on_tick(Tick(1, FIXED_TS, 1), c)
        s.on_trade(Trade(1, FIXED_TS, 1, 1, Side.BUY), c)
        s.on_bar(Bar(1, FIXED_TS, 1, 1, 1, 1, 1), c)
        s.on_stop(c)
    assert c.pending_len() == 0
    assert s.on_save() == {}


def test_an_unknown_config_key_is_refused_rather_than_ignored():
    d = StrategyConfig(name="x", symbols=(1, 2)).to_dict()
    assert StrategyConfig.from_dict(d).symbols == (1, 2)
    with pytest.raises(ValueError, match="unknown"):
        StrategyConfig.from_dict({**d, "windwo": 5})


# ── the reference strategy ────────────────────────────────────────────────────


def mr(**overrides) -> MeanReversion:
    params = {"symbol_id": 1, "window": 8, "entry_z": 1.5, "exit_z": 0.5}
    params.update(overrides)
    return MeanReversion(MeanReversionParams(**params))


def feed(strategy, prices, c, *, start=FIXED_TS, symbol_id=1):
    for i, px in enumerate(prices):
        with c.event(start + i):
            strategy.on_tick(Tick(symbol_id, start + i, int(px * 100_000_000)), c)


def test_nothing_is_emitted_before_the_window_is_warm():
    s, c = mr(window=8), ctx()
    feed(s, [100.0 + i for i in range(7)], c)
    assert c.pending_len() == 0


def test_a_zero_variance_window_emits_nothing_instead_of_a_nan_target():
    s, c = mr(window=4), ctx()
    feed(s, [100.0] * 12, c)
    assert c.pending_len() == 0


def test_a_cheap_price_goes_long_and_a_rich_one_goes_short():
    s, c = mr(window=8, max_position=Decimal("0.01")), ctx()
    feed(s, [100.0, 100.1, 99.9, 100.0, 100.1, 99.9, 100.0, 95.0], c)
    (rec,) = c.take_pending()
    assert int(rec["target_qty"]) == 1_000_000  # +0.01 in fixed point

    feed(s, [100.0, 100.1, 99.9, 100.0, 100.1, 99.9, 100.0, 105.0], c)
    assert int(c.take_pending()[-1]["target_qty"]) == -1_000_000


def test_a_target_that_has_not_changed_is_not_re_emitted():
    s, c = mr(window=4), ctx()
    feed(s, [100.0, 100.1, 99.9, 90.0], c)
    assert c.pending_len() == 1
    feed(s, [90.0], c)  # still long, nothing new to say
    assert c.pending_len() == 1


def test_the_hysteresis_band_holds_the_target_instead_of_thrashing():
    # Between exit_z and entry_z the target must be held: a single threshold
    # would flip the position on every tick that straddles it.
    s = mr(window=4, entry_z=1.5, exit_z=0.5)
    s._target = Decimal("0.01")
    assert s._next_target(1.0) == Decimal("0.01")
    assert s._next_target(-1.0) == Decimal("0.01")
    assert s._next_target(0.2) == Decimal(0)


def test_saved_state_restores_the_target_across_a_restart():
    s, c = mr(window=4), ctx()
    feed(s, [100.0, 100.1, 99.9, 90.0], c)
    assert c.pending_len() == 1
    state = s.on_save()

    restarted = mr(window=4)
    restarted.on_load(state)
    fresh = ctx()
    feed(restarted, [90.0], fresh)
    # It resumes knowing it is already long, so it does not re-open the position.
    assert fresh.pending_len() == 0
    assert restarted._target == Decimal("0.01")


def test_state_survives_json_without_becoming_a_float():
    import json

    s = mr()
    s._target = Decimal("0.00000001")
    restored = mr()
    restored.on_load(json.loads(json.dumps(s.on_save())))
    assert restored._target == Decimal("0.00000001")


def test_events_for_other_symbols_are_ignored():
    s, c = mr(window=4), ctx()
    feed(s, [100.0, 100.1, 99.9, 90.0], c, symbol_id=2)
    assert c.pending_len() == 0


def test_a_window_without_hysteresis_is_refused_at_construction():
    with pytest.raises(ValueError, match="hysteresis"):
        MeanReversionParams(symbol_id=1, entry_z=0.5, exit_z=1.5)
    with pytest.raises(ValueError, match="window"):
        MeanReversionParams(symbol_id=1, window=1)


def test_params_from_config_keep_the_size_out_of_float():
    cfg = StrategyConfig(
        name="ref", symbols=(4,), params={"window": 8, "max_position": "0.1", "exit_z": 0.5}
    )
    p = MeanReversionParams.from_config(cfg)
    assert p.symbol_id == 4
    assert p.max_position == Decimal("0.1")
