"""Deterministic market-data fixtures shared with the Rust round-trip examples.

``make_md_slice(i)`` MUST match ``make_md_slice`` in
``crates/axon-ipc/examples/md_writer.rs`` byte-for-byte, and ``make_md_bar(i)``
``make_md_bar`` in ``crates/axon-ipc/examples/md_bar_writer.rs``, so the
cross-language round-trip test can generate on one side and verify on the other.
"""

from __future__ import annotations

import numpy as np

from axon.contracts import (
    MD_BAR_FLAG_FIRST_BAR,
    MD_BAR_FLAG_GAP_BEFORE,
    MD_KIND_QUOTE,
    MD_KIND_TRADE,
    new_md_bar,
    new_md_slice,
)

#: One minute, in the two units the bar record uses.
INTERVAL_MS = 60_000
INTERVAL_NS = 60_000_000_000
EPOCH = 1_700_000_000_000_000_000


def make_md_slice(i: int) -> np.ndarray:
    """The i-th deterministic market-data slice (mirrors the Rust fixture)."""
    base = 5_000_000_000_000 + i * 250_000  # 50,000.0 + i·0.0025, fixed-point
    # Every fifth record carries no ticker at all, so the round-trip covers a reader
    # that back-fills the sentinel as well as one that respects it. Half the rest are
    # venue-timed and half receipt-only, because a reader that collapsed the mark's
    # two clocks into one would pass a fixture that only ever exercised one of them.
    ticker = (
        {}
        if i % 5 == 4
        else {
            "mark_px": base + 125_000,
            "index_px": base - 1_000_000,
            "funding_rate": 1_250 + i,
            "funding_interval_ns": 3_600_000_000_000,
            "mark_ts_venue": 0 if i % 2 == 0 else EPOCH + i - 40,
            "mark_ts_ingest": EPOCH + i - 33,
        }
    )
    return new_md_slice(
        seq=i,
        ts_event=EPOCH + i,
        symbol_id=i % 7,
        bid_px=base,
        bid_sz=100_000_000 + i,
        ask_px=base + 500_000,
        ask_sz=250_000_000 - i * 2,
        last_trade_px=base + 250_000,
        last_trade_sz=50_000_000 + i * 3,
        last_trade_ts=EPOCH + i - 17,
        last_trade_sell=i % 2 == 0,
        kind=MD_KIND_TRADE if i % 3 == 0 else MD_KIND_QUOTE,
        **ticker,
    )


def make_md_bar(i: int) -> np.ndarray:
    """The i-th deterministic closed bar (mirrors the Rust fixture)."""
    open_time = EPOCH + i * INTERVAL_NS
    if i == 0:
        flags = MD_BAR_FLAG_FIRST_BAR
    elif i % 7 == 0:
        flags = MD_BAR_FLAG_GAP_BEFORE
    else:
        flags = 0
    base = 5_000_000_000_000 + i * 250_000  # 50,000.0 + i·0.0025, fixed-point
    return new_md_bar(
        seq=i,
        # The close, i.e. the venue's T + 1 ms — never `open_time`.
        ts_event=open_time + INTERVAL_NS,
        open_time=open_time,
        symbol_id=i % 7,
        interval_ms=INTERVAL_MS,
        open=base,
        high=base + 1_000_000,
        low=base - 750_000,
        close=base + 250_000,
        volume=50_000_000 + i * 3,
        flags=flags,
    )
