"""The Python contract must match the schema (and therefore the Rust struct)."""

import numpy as np
import pytest

from axon.contracts import (
    FLAG_REDUCE_ONLY,
    KIND_TARGET_POSITION,
    MD_BAR_DTYPE,
    MD_BAR_KIND_CLOSED,
    MD_BAR_SCHEMA_VERSION,
    MD_FLAG_LAST_TRADE_SELL,
    MD_KIND_QUOTE,
    MD_KIND_TRADE,
    MD_SLICE_DTYPE,
    MD_SLICE_SCHEMA_VERSION,
    MD_SLICE_SIZE,
    RING_HEADER_SIZE,
    RING_KIND_MD_BAR,
    RING_KIND_MD_SLICE,
    RING_KIND_SIGNAL,
    RING_MAGIC,
    SCHEMA_VERSION,
    SIGNAL_DTYPE,
    SIGNAL_SIZE,
    from_fixed,
    new_md_bar,
    new_md_slice,
    new_signal,
    record_spec_for,
    to_fixed,
)


def test_dtype_stride_is_one_cache_line():
    assert SIGNAL_DTYPE.itemsize == 64
    assert SIGNAL_SIZE == 64


def test_field_offsets_match_schema():
    # SIGNAL_DTYPE.fields maps name -> (dtype, offset).
    off = {name: spec[1] for name, spec in SIGNAL_DTYPE.fields.items()}
    assert off == {
        "seq": 0,
        "ts_event": 8,
        "target_qty": 16,
        "price_band": 24,
        "symbol_id": 32,
        "ttl_ms": 36,
        "model_version": 40,
        "flags": 44,
        "schema_version": 46,
        "urgency": 47,
        "kind": 48,
        # `pad0` is named rather than implicit: this file has no implicit padding, and
        # an unnamed hole is a byte no reader can be asked to validate. ADR-0031 carved
        # `max_order_age_ms` out of the reserved block, which is why it starts at 52, and
        # schema version 3 spent the last eight bytes on `ts_cause`. There is no
        # `reserved` field left: the record is fully named, so the next field to be added
        # has to re-cut this layout rather than extend it.
        "pad0": 49,
        "max_order_age_ms": 52,
        "ts_cause": 56,
    }


def test_ring_magic_reads_as_axonring():
    assert RING_MAGIC.to_bytes(8, "little") == b"AXONRING"
    assert RING_HEADER_SIZE == 192


def test_new_signal_stamps_version_and_kind():
    s = new_signal(seq=5, target_qty=100, flags=FLAG_REDUCE_ONLY)
    assert int(s["seq"]) == 5
    assert int(s["schema_version"]) == SCHEMA_VERSION
    assert int(s["kind"]) == KIND_TARGET_POSITION
    assert int(s["flags"]) & FLAG_REDUCE_ONLY
    assert bytes(s["pad0"]) == b"\x00" * 3
    assert int(s["max_order_age_ms"]) == 0, "0 = defer to the operator's ceiling"
    assert int(s["ts_cause"]) == 0, "0 = no cause stated, so the stage is not measured"


def test_an_order_lifetime_is_a_different_field_from_the_admission_window():
    # `ttl_ms` is a signal-*admission* window the Rust reader consumes before the
    # planner sees the record, clamped against the operator's `max_signal_age_ms`; it
    # has never applied to an order already resting at the venue. One number for both
    # would mean a strategy could not ask for a resting order without also asking to be
    # admitted for the same span (ADR-0031).
    s = new_signal(seq=1, ttl_ms=60_000, max_order_age_ms=5_000)
    assert int(s["ttl_ms"]) == 60_000
    assert int(s["max_order_age_ms"]) == 5_000
    assert s.tobytes().__len__() == 64, "the stride did not move"


def test_signal_serializes_to_64_bytes():
    assert new_signal(seq=1).tobytes().__len__() == 64


def test_fixed_point_helpers():
    assert to_fixed(1.5) == 150_000_000
    assert from_fixed(150_000_000) == 1.5


def test_md_slice_stride_is_two_cache_lines():
    assert MD_SLICE_DTYPE.itemsize == 128
    assert MD_SLICE_SIZE == 128


def test_md_slice_field_offsets_match_schema():
    off = {name: spec[1] for name, spec in MD_SLICE_DTYPE.fields.items()}
    assert off == {
        "seq": 0,
        "ts_event": 8,
        "bid_px": 16,
        "bid_sz": 24,
        "ask_px": 32,
        "ask_sz": 40,
        "last_trade_px": 48,
        "last_trade_sz": 56,
        "last_trade_ts": 64,
        "symbol_id": 72,
        "flags": 76,
        "schema_version": 78,
        "kind": 79,
        # The mark/funding tail (schema version 2, ADR-0028) spends what ADR-0012
        # reserved, at the stride it reserved it at. Everything above is unmoved: a
        # shifted offset here is the whole failure this table exists to catch.
        "mark_px": 80,
        "index_px": 88,
        "funding_rate": 96,
        "funding_interval_ns": 104,
        "mark_ts_venue": 112,
        "mark_ts_ingest": 120,
    }


def test_new_md_slice_stamps_version_and_flags():
    s = new_md_slice(seq=3, bid_px=10, ask_px=11, last_trade_sell=True, kind=MD_KIND_TRADE)
    assert int(s["seq"]) == 3
    assert int(s["schema_version"]) == MD_SLICE_SCHEMA_VERSION
    assert int(s["kind"]) == MD_KIND_TRADE
    assert int(s["flags"]) & MD_FLAG_LAST_TRADE_SELL
    assert s.tobytes().__len__() == 128
    assert not int(new_md_slice(kind=MD_KIND_QUOTE)["flags"]) & MD_FLAG_LAST_TRADE_SELL


def test_a_slice_without_a_ticker_says_so_on_the_clock_not_on_the_price():
    # "Has a ticker been seen?" must key on the receipt clock: a venue is free to mark
    # an instrument at zero, and a consumer that read `mark_px == 0` as absence would
    # discard a real mark. The whole tail is zero until a ticker arrives.
    s = new_md_slice(seq=1, bid_px=10)
    assert int(s["mark_ts_ingest"]) == 0
    assert int(s["funding_interval_ns"]) == 0
    marked_at_zero = new_md_slice(seq=2, mark_px=0, mark_ts_ingest=99)
    assert int(marked_at_zero["mark_ts_ingest"]) == 99
    # …and the venue clock stays distinct from ours, which is what says whether an age
    # derived from this mark reproduces on replay (ADR-0011).
    assert int(marked_at_zero["mark_ts_venue"]) == 0


def test_md_bar_field_offsets_match_schema():
    off = {name: spec[1] for name, spec in MD_BAR_DTYPE.fields.items()}
    assert off == {
        "seq": 0,
        "ts_event": 8,
        "open_time": 16,
        "open": 24,
        "high": 32,
        "low": 40,
        "close": 48,
        "volume": 56,
        "symbol_id": 64,
        "interval_ms": 68,
        "flags": 72,
        "schema_version": 74,
        "kind": 75,
        "reserved": 76,
    }


def test_new_md_bar_stamps_version_and_kind():
    b = new_md_bar(seq=3, ts_event=60_000_000_000, open_time=0, interval_ms=60_000, close=7)
    assert int(b["seq"]) == 3
    assert int(b["schema_version"]) == MD_BAR_SCHEMA_VERSION
    assert int(b["kind"]) == MD_BAR_KIND_CLOSED
    assert bytes(b["reserved"]) == b"\x00" * 52
    assert b.tobytes().__len__() == 128
    # The close, never the open. One field apart, and confusing them is a whole-bar
    # lookahead leak nothing downstream can see.
    assert int(b["ts_event"]) > int(b["open_time"])


def test_the_records_never_share_a_ring_kind():
    # Equal kinds would let a consumer open the wrong ring and decode another record's
    # bytes as prices; 0 is reserved for "the writer never stamped it". This matters
    # more than it used to: MdSlice and MdBar have the *same* 128-byte stride, so the
    # kind tag is now the only check that can tell those two rings apart at all.
    kinds = (RING_KIND_SIGNAL, RING_KIND_MD_SLICE, RING_KIND_MD_BAR)
    assert len(set(kinds)) == len(kinds)
    assert 0 not in kinds
    assert MD_BAR_DTYPE.itemsize == MD_SLICE_DTYPE.itemsize
    assert record_spec_for(SIGNAL_DTYPE).kind == RING_KIND_SIGNAL
    assert record_spec_for(MD_SLICE_DTYPE).kind == RING_KIND_MD_SLICE
    assert record_spec_for(MD_BAR_DTYPE).kind == RING_KIND_MD_BAR


def test_a_foreign_dtype_cannot_back_a_ring():
    # Anything not in the contract has no record kind to stamp, so a ring built on
    # it could never be validated by the other side.
    with pytest.raises(ValueError, match="not a contract record dtype"):
        record_spec_for(np.dtype([("px", "<i8")]))
