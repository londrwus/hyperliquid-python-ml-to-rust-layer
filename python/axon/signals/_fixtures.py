"""Deterministic signal fixtures shared with the Rust round-trip examples.

``make_signal(i, n)`` MUST match ``make_signal`` in
``crates/axon-ipc/examples/ipc_writer.rs`` byte-for-byte, so the cross-language
round-trip test can generate on one side and verify on the other.
"""

from __future__ import annotations

import numpy as np

from axon.contracts import FLAG_REDUCE_ONLY, new_signal


def make_signal(i: int, n: int) -> np.ndarray:
    """The i-th of n deterministic target-position signals (mirrors the Rust fixture)."""
    flags = FLAG_REDUCE_ONLY if i % 2 == 0 else 0
    return new_signal(
        seq=i,
        ts_event=1_700_000_000_000_000_000 + i,
        symbol_id=i % 7,
        target_qty=(i - n // 2) * 1_000_000,
        urgency=i % 4,
        price_band=0,
        ttl_ms=500,
        model_version=1,
        flags=flags,
    )
