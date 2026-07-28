"""From the wire to the feature inputs — the one place fixed-point becomes float.

The Rust core publishes :data:`~axon.contracts.MD_SLICE_DTYPE` records into the
market-data ring (ADR-0012); :class:`~axon.marketdata.MdRingConsumer` hands them
over as a structured array. This module holds the adapters between what the core
publishes and the named arrays a :class:`~axon.features.spec.FeatureSpec` consumes,
so the online path and the offline recompute start from *identical* arrays — which
is what makes the feature-parity gate a comparison of transforms rather than of
decoders.

One adapter per shape of market data the core produces, and no more: a second way
to spell "this record becomes these arrays" is a second decoder, and the two agree
on the day they are written. :func:`md_slice_inputs` takes the ring's tick-level
record; :func:`bar_inputs` takes a closed OHLCV candle, which is what the venue's
``candle`` subscription publishes and what :class:`axon.strategy.events.Bar`
delivers to a strategy.
"""

from __future__ import annotations

import numpy as np

from axon.contracts import (
    FIXED_POINT_SCALE,
    MD_FLAG_LAST_TRADE_SELL,
    MD_KIND_TRADE,
    MD_SLICE_DTYPE,
)
from axon.features.registry import FeatureError

_SCALE = float(FIXED_POINT_SCALE)


def _real(field: np.ndarray) -> np.ndarray:
    """Fixed-point integers → float64 reals. The only ``/ 1e8`` in this package."""
    return field.astype(np.float64) / _SCALE


def md_slice_inputs(
    batch: np.ndarray, *, require_monotonic: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Split a batch of market-data slices into feature inputs and event times.

    Returns ``(inputs, ts_event)``. The timestamps come back **separately and as
    int64** rather than as another entry in ``inputs``: float64 carries 53 mantissa
    bits and a 2026 nanosecond timestamp needs 61, so a timestamp that went through
    the feature matrix would round into ~256 ns buckets and reorder events. This is
    the same refusal :class:`axon.strategy.context.StrategyContext` makes at the
    other end of the boundary, for the same reason.

    ``require_monotonic`` rejects a batch whose event times go backwards. Features
    are defined over the sequence the core saw; recomputing them over a reshuffled
    batch produces different numbers and the parity gate would blame the transforms
    for what is really a late-event bug in the capture.
    """
    arr = np.atleast_1d(np.asarray(batch))
    if arr.dtype != MD_SLICE_DTYPE:
        raise FeatureError(
            f"expected an MdSlice array ({MD_SLICE_DTYPE.names}), got dtype {arr.dtype}"
        )
    if arr.ndim != 1:
        raise FeatureError(f"expected a 1-D batch of slices, got shape {arr.shape}")

    ts_event = np.asarray(arr["ts_event"], dtype=np.int64)
    if require_monotonic and ts_event.size > 1 and np.any(np.diff(ts_event) < 0):
        first = int(np.argmax(np.diff(ts_event) < 0)) + 1
        raise FeatureError(
            f"event time goes backwards at row {first} "
            f"({int(ts_event[first - 1])} → {int(ts_event[first])}); features are "
            "defined over the order the core observed, so an out-of-order batch must "
            "be sorted (or dropped) before it is recomputed, not silently accepted"
        )

    # Every slice carries the *last* print, whatever caused the update, so a run of
    # quote updates repeats one trade over and over. Counting it once per quote turns
    # a single 5-lot buy into a wall of one-sided flow; only trade-kind slices carry
    # volume into the flow features.
    is_trade = arr["kind"] == MD_KIND_TRADE
    trade_sz = np.where(is_trade, _real(arr["last_trade_sz"]), 0.0)
    sell = (arr["flags"] & MD_FLAG_LAST_TRADE_SELL) != 0
    trade_sign = np.where(is_trade, np.where(sell, -1.0, 1.0), 0.0)

    last_px = _real(arr["last_trade_px"])
    inputs = {
        "bid_px": _real(arr["bid_px"]),
        "bid_sz": _real(arr["bid_sz"]),
        "ask_px": _real(arr["ask_px"]),
        "ask_sz": _real(arr["ask_sz"]),
        # 0 means "no print yet" on the wire, and zero is not a price. Left as 0 it
        # would show up as a -100% return the first time a trade arrives.
        "last_px": np.where(last_px > 0.0, last_px, np.nan),
        "trade_sz": trade_sz,
        "trade_sign": trade_sign,
    }
    return inputs, ts_event


#: The named arrays a bar-driven spec reads, in the order a caller supplies them.
BAR_INPUTS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def bar_inputs(open_px, high_px, low_px, close_px, volume) -> dict[str, np.ndarray]:
    """Closed OHLCV bars → the named arrays a bar-driven spec consumes.

    The other half of this module's job. :func:`md_slice_inputs` adapts the
    market-data ring; this adapts a **closed candle**, which is what the venue's
    ``candle`` subscription publishes and what
    :class:`axon.strategy.events.Bar` delivers to a strategy. Every argument is a
    fixed-point integer array — the wire encoding, not a real number — so the
    research path that reads a downloaded candle history and the serving path that
    reads a live ``Bar`` divide by the same scale in the same place, and can only
    disagree about the transforms the parity gate is actually comparing.

    ``ts_event`` is deliberately not returned and never an entry here: it is the
    bar's **close** time and it stays int64, for the reason
    :func:`md_slice_inputs` explains. A caller holds it alongside the matrix.

    A non-positive price is NaN rather than 0: zero is what an unfilled bar looks
    like on the wire, and a zero close feeds a -100% return into the first return
    feature that touches it. Volume is passed through, including a legitimate zero
    — a bar in which nothing traded is real information, not a gap.
    """
    arrays = {}
    n: int | None = None
    for name, raw in zip(BAR_INPUTS, (open_px, high_px, low_px, close_px, volume)):
        a = np.asarray(raw)
        if a.ndim != 1:
            raise FeatureError(f"bar input {name!r} must be 1-D, got shape {a.shape}")
        if n is None:
            n = a.size
        elif a.size != n:
            raise FeatureError(
                f"bar input {name!r} has {a.size} rows but earlier inputs have {n}; "
                "arrays of different lengths are not describing the same bars"
            )
        if a.dtype.kind not in "iu":
            # A float here means the caller already divided by the scale, and doing
            # it again puts every price at 1e-8 of itself — which still trains, still
            # backtests, and is wrong by eight orders of magnitude.
            raise FeatureError(
                f"bar input {name!r} is {a.dtype}, not a fixed-point integer; bars cross "
                "the wire as integers and the scaling happens here, exactly once"
            )
        real = _real(a)
        arrays[name] = real if name == "volume" else np.where(real > 0.0, real, np.nan)
    return arrays


__all__ = ["BAR_INPUTS", "bar_inputs", "md_slice_inputs"]
