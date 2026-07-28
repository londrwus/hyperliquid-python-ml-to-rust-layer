"""``axon.live`` — the runner that puts a strategy on the signal ring.

Boundary B in one module (``docs/02``): Python computes and emits, Rust executes.
:class:`~axon.live.runner.StrategyRunner` owns the ring producer, drives the
strategy's callbacks with each event's own time, drains the context after every
one, and pushes in order. A full ring is backpressure, never a dropped signal.

:mod:`axon.live.mdfeed` closes the other half of the loop: it turns the
market-data ring the Rust core publishes into the events a strategy is driven by,
so a live Python decision is taken against the same book the executing core saw and
is stamped with the same venue clock the core ages it against.
:mod:`axon.live.probe` is the smallest program that exercises the whole path.

:mod:`axon.live.liveness` adds the heartbeat the boundary needs in both
directions: the producer publishes a counter so the Rust side can tell an idle
strategy from a dead one, and the runner watches the ring's ``tail`` so Python
can tell a slow consumer from a dead core.

Run the reference strategy end to end with ``python -m axon.live --help``.
"""

from axon.live.liveness import (
    FLAG_RUNNING,
    FLAG_STOPPED,
    LivenessBeacon,
    LivenessSnapshot,
    read_liveness,
)
from axon.live.mdfeed import FeedStats, MdFeedError, MdRingFeed
from axon.live.probe import TargetProbe
from axon.live.runner import BackpressureError, LiveError, RunnerStats, StrategyRunner

__all__ = [
    "FLAG_RUNNING",
    "FLAG_STOPPED",
    "BackpressureError",
    "FeedStats",
    "LiveError",
    "MdFeedError",
    "MdRingFeed",
    "LivenessBeacon",
    "LivenessSnapshot",
    "RunnerStats",
    "StrategyRunner",
    "TargetProbe",
    "read_liveness",
]
