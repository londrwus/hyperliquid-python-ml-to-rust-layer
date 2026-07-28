"""The heartbeat ``docs/02`` requires: how each side detects a dead peer.

The ring alone cannot answer "is my peer alive?". A ``head`` index that stops
advancing means either *the strategy has nothing to say* or *the strategy is
dead*, and those demand opposite responses — hold the position, or flatten it.
A target-position strategy is idle by design (it re-sends nothing while its
target is unchanged), so the ambiguous case is the normal case.

So the producer publishes a counter that advances on a schedule rather than on
activity, into a 64-byte memory-mapped sidecar next to the ring. A reader
(the Rust runtime, or an operator) mmaps it and watches ``beats``; a counter
that stops while the process is supposed to be running is a dead peer, and the
Rust side answers that with Hyperliquid's ``scheduleCancel`` dead-man's switch
(``docs/02``, ADR-0010).

Layout — 64 bytes, one cache line, little-endian, every field naturally aligned::

     0  u64  magic          = "AXONBEAT"
     8  u32  version        = 1
    12  u32  pid            producer's process id (for the operator, not for logic)
    16  u64  beats          monotonic; the field a peer actually watches
    24  i64  last_event_ns  EVENT time of the last event handled
    32  u64  last_beat_ns   wall clock (CLOCK_REALTIME) of the last beat
    40  u64  signals        signals emitted since start
    48  u64  backpressure   times the ring was full when we tried to push
    56  u32  pending        signals queued in the runner's outbox right now
    60  u32  flags          bit0 running, bit1 stopped cleanly

``beats`` is written **last**, after every field it describes, so a reader that
sees a new count also sees the payload behind it — the same publish discipline
the ring itself uses. Per-field tearing is impossible on x86_64 (naturally
aligned 8-byte accesses are atomic), which is the platform ADR-0006 already
assumes; only cross-field skew is possible, and liveness does not care.

This file is *not* part of ``contracts/schema.toml``. It is a Python-side seam
that the Rust reader must be written against; folding it into the shared contract
is an ADR-sized decision (it adds a second shared-memory object to the boundary),
so it is deliberately kept out of the ring's own bytes where a future schema bump
would collide with it.
"""

from __future__ import annotations

import mmap
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

#: Reads as ASCII "AXONBEAT" in a little-endian hexdump, matching the ring's idiom.
BEACON_MAGIC = int.from_bytes(b"AXONBEAT", "little")
BEACON_VERSION = 1
BEACON_SIZE = 64

_OFF_MAGIC = 0
_OFF_VERSION = 8
_OFF_PID = 12
_OFF_BEATS = 16
_OFF_LAST_EVENT_NS = 24
_OFF_LAST_BEAT_NS = 32
_OFF_SIGNALS = 40
_OFF_BACKPRESSURE = 48
_OFF_PENDING = 56
_OFF_FLAGS = 60

#: The producer considers itself live.
FLAG_RUNNING = 1
#: The producer shut down on purpose. A peer can tell this from a crash and
#: alert differently — though it must still flatten, because a clean exit does
#: not mean the position is someone else's problem.
FLAG_STOPPED = 2


def _u64(buf, offset: int) -> np.ndarray:
    return np.ndarray((1,), dtype="<u8", buffer=buf, offset=offset)


def _i64(buf, offset: int) -> np.ndarray:
    return np.ndarray((1,), dtype="<i8", buffer=buf, offset=offset)


def _u32(buf, offset: int) -> np.ndarray:
    return np.ndarray((1,), dtype="<u4", buffer=buf, offset=offset)


@dataclass(frozen=True)
class LivenessSnapshot:
    """One read of a beacon file."""

    pid: int
    beats: int
    last_event_ns: int
    last_beat_ns: int
    signals: int
    backpressure: int
    pending: int
    flags: int

    @property
    def running(self) -> bool:
        return bool(self.flags & FLAG_RUNNING)

    @property
    def stopped(self) -> bool:
        return bool(self.flags & FLAG_STOPPED)


class LivenessBeacon:
    """The producer's end of the heartbeat. One per process; creates the file."""

    def __init__(
        self,
        path: str,
        *,
        pid: int | None = None,
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.path = path
        self._wall_ns = wall_ns
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, BEACON_SIZE)
            self._mm = mmap.mmap(fd, BEACON_SIZE)
        finally:
            os.close(fd)

        self._beats = _u64(self._mm, _OFF_BEATS)
        self._last_event = _i64(self._mm, _OFF_LAST_EVENT_NS)
        self._last_beat = _u64(self._mm, _OFF_LAST_BEAT_NS)
        self._signals = _u64(self._mm, _OFF_SIGNALS)
        self._backpressure = _u64(self._mm, _OFF_BACKPRESSURE)
        self._pending = _u32(self._mm, _OFF_PENDING)
        self._flags = _u32(self._mm, _OFF_FLAGS)

        self._beats[0] = 0
        self._last_event[0] = 0
        self._last_beat[0] = 0
        self._signals[0] = 0
        self._backpressure[0] = 0
        self._pending[0] = 0
        self._flags[0] = FLAG_RUNNING
        _u64(self._mm, _OFF_MAGIC)[0] = BEACON_MAGIC
        _u32(self._mm, _OFF_VERSION)[0] = BEACON_VERSION
        _u32(self._mm, _OFF_PID)[0] = os.getpid() if pid is None else pid
        self._mm.flush()

    def beat(
        self,
        *,
        last_event_ns: int,
        signals: int,
        backpressure: int,
        pending: int,
        flags: int = FLAG_RUNNING,
    ) -> int:
        """Publish one heartbeat. Returns the new beat count."""
        self._last_event[0] = last_event_ns
        self._last_beat[0] = self._wall_ns()
        self._signals[0] = signals
        self._backpressure[0] = backpressure
        self._pending[0] = min(pending, 2**32 - 1)
        self._flags[0] = flags
        beats = int(self._beats[0]) + 1
        self._beats[0] = beats  # published last: it is what the peer polls
        return beats

    @property
    def beats(self) -> int:
        return int(self._beats[0])

    def close(self) -> None:
        # Drop the views before closing, or mmap.close() raises BufferError
        # ("cannot close exported pointers exist") — same constraint as the ring.
        self._beats = None
        self._last_event = None
        self._last_beat = None
        self._signals = None
        self._backpressure = None
        self._pending = None
        self._flags = None
        if getattr(self, "_mm", None) is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None

    def __enter__(self) -> "LivenessBeacon":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_liveness(path: str) -> LivenessSnapshot:
    """Read a beacon file. The peer side of the heartbeat, and what tests assert on."""
    with open(path, "rb") as f:
        raw = f.read(BEACON_SIZE)
    if len(raw) < BEACON_SIZE:
        raise ValueError(f"file too small ({len(raw)} bytes) to be an Axon beacon")
    magic = int.from_bytes(raw[_OFF_MAGIC : _OFF_MAGIC + 8], "little")
    if magic != BEACON_MAGIC:
        raise ValueError(f"not an Axon beacon: bad magic {magic:#018x}")
    version = int.from_bytes(raw[_OFF_VERSION : _OFF_VERSION + 4], "little")
    if version != BEACON_VERSION:
        raise ValueError(f"unsupported beacon version {version}")
    return LivenessSnapshot(
        pid=int.from_bytes(raw[_OFF_PID : _OFF_PID + 4], "little"),
        beats=int.from_bytes(raw[_OFF_BEATS : _OFF_BEATS + 8], "little"),
        last_event_ns=int.from_bytes(
            raw[_OFF_LAST_EVENT_NS : _OFF_LAST_EVENT_NS + 8], "little", signed=True
        ),
        last_beat_ns=int.from_bytes(raw[_OFF_LAST_BEAT_NS : _OFF_LAST_BEAT_NS + 8], "little"),
        signals=int.from_bytes(raw[_OFF_SIGNALS : _OFF_SIGNALS + 8], "little"),
        backpressure=int.from_bytes(raw[_OFF_BACKPRESSURE : _OFF_BACKPRESSURE + 8], "little"),
        pending=int.from_bytes(raw[_OFF_PENDING : _OFF_PENDING + 4], "little"),
        flags=int.from_bytes(raw[_OFF_FLAGS : _OFF_FLAGS + 4], "little"),
    )


__all__ = [
    "BEACON_MAGIC",
    "BEACON_SIZE",
    "BEACON_VERSION",
    "FLAG_RUNNING",
    "FLAG_STOPPED",
    "LivenessBeacon",
    "LivenessSnapshot",
    "read_liveness",
]
