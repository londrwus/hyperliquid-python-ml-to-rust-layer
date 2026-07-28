"""The Rust→Python market-data beacon: telling a quiet publisher from a dead one.

``MdWritePolicy::OnChange`` writes a slice only when the state it carries actually
moved (ADR-0012). That is right, and it makes an empty ring **ambiguous by design**:
a flat top of book and a corpse produce the same nothing. :mod:`axon.parity.monitor`
can only resolve that with a timer — a guess with a deadline on it — which is why
ADR-0030 records the gap and specifies the fix rather than shipping one.

The fix is a 64-byte memory-mapped sidecar beside the ring, written by the Rust core's
**pass loop** rather than by its event handler. That distinction is the whole design: a
beacon hung off ``on_event`` carries exactly what the ring already carries, and the case
it must distinguish — *nothing arrived* — is precisely the case in which ``on_event``
does not run. The writer is :mod:`axon_ipc::beacon` (``crates/axon-ipc/src/beacon.rs``),
which is where it lives because mapping a file is the ``unsafe`` that crate already
holds; this module is the reader.

It is the mirror of :mod:`axon.live.liveness`, which removes the same ambiguity in the
Python→Rust direction, and **the first 40 bytes and the last 4 are laid out identically**
so that an operator who has learned one already knows the other::

     0  u64  magic          = "AXONMDBN"   (liveness says "AXONBEAT")
     8  u32  version        = 1
    12  u32  pid            publisher's process id
    16  u64  beats          monotonic; the field a reader actually watches
    24  i64  last_event_ns  EVENT time high-water mark of the core (0 = nothing yet)
    32  u64  last_beat_ns   WALL clock at that beat (0 = the session had none)
    40  u32  published      slices the slice ring accepted
    44  u32  coalesced      updates OnChange suppressed
    48  u32  dropped        slices the ring refused (this reader fell behind)
    52  u32  bars_published closed bars the bar ring accepted
    56  u32  stale_quote    slices published with no quote at all
    60  u32  flags          bit0 running, bit1 stopped cleanly

Two things a reader has to get right, both of them consequences of the 64-byte budget:

**The five counters are 32 bits and they wrap.** Ask them for a *delta* and never for a
total — :func:`publisher_state` does the modular arithmetic. The absolute value stops
being meaningful after 2\\ :sup:`32` increments (five days at ten thousand slices a
second) and nothing here prints one; ``MdStats`` on the Rust status line is where the
totals live, and it is the side with room for them.

**A read can mix beats.** ``beats`` is stored last by the writer and read first here,
which buys the one guarantee that matters: a counter is always *at least* as new as the
beat count beside it, never older. So the reading that would turn a live publisher into
a dead-looking one cannot be produced by skew at all. All 64 bytes are copied in one
slice and parsed from the copy, which narrows the window to a memcpy; ``published -
beats`` is still not a quantity, and nothing computes it.

``last_beat_ns`` is a **wall clock**, and it is the one field here that is. The house
rule is event time everywhere; the named exceptions are a dead-man's-switch deadline
(the venue holds it) and reconnect backoff (it is not ordering). This is a third, and
its reason is the same shape: the condition being detected is *the absence of events*,
and the absence of an event has no event time — an event-time-only beacon freezes at
exactly the moment it is needed, because the clock that would advance it is the thing
that stopped. Nothing orders or ages anything on it; it is read only as a difference, so
that a single read can say how long ago the publisher last ran. ``0`` means the
publishing session had no wall clock at all (an offline replay), which is reported as
unknown rather than as a beat in 1970.
"""

from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from enum import Enum

#: Reads as ASCII ``AXONMDBN`` in a little-endian hexdump.
MD_BEACON_MAGIC = int.from_bytes(b"AXONMDBN", "little")
MD_BEACON_VERSION = 1
MD_BEACON_SIZE = 64

_OFF_MAGIC = 0
_OFF_VERSION = 8
_OFF_PID = 12
_OFF_BEATS = 16
_OFF_LAST_EVENT_NS = 24
_OFF_LAST_BEAT_NS = 32
_OFF_PUBLISHED = 40
_OFF_COALESCED = 44
_OFF_DROPPED = 48
_OFF_BARS_PUBLISHED = 52
_OFF_STALE_QUOTE = 56
_OFF_FLAGS = 60

#: The publisher considers itself live.
MD_BEACON_FLAG_RUNNING = 1
#: The publisher shut down on purpose. A reader must still treat the feed as gone — a
#: clean exit does not put the market data back — but "stopped" and "died" want
#: different words at 03:00, and only the publisher can tell them apart.
MD_BEACON_FLAG_STOPPED = 2

_U32 = 1 << 32


class BeaconError(ValueError):
    """A file that is not a beacon this build can read.

    Subclasses :class:`ValueError` so it reads the same way as
    :func:`axon.live.liveness.read_liveness`'s refusals to a caller that never heard
    of it.
    """


def beacon_path(md_ring_path: str) -> str:
    """Where the beacon lives, given the slice ring's path.

    Mirrors ``axon_ipc::beacon::beacon_path``. The suffix is **appended** rather than
    substituted for the extension, and that is the load-bearing part: a string cannot
    equal itself plus a suffix, so the beacon can never name either ring's own file
    however the ring's path was spelled. ADR-0030 asks for validation refusing a
    collision; this makes one unrepresentable, which is the stronger form — there is no
    check anybody can forget to run.
    """
    return f"{md_ring_path}.beacon"


@dataclass(frozen=True)
class MdBeaconSnapshot:
    """One read of a beacon file.

    The counters are raw 32-bit values straight off the wire. **Do not read them as
    totals** — they wrap. :func:`publisher_state` is what turns two of these into an
    answer.
    """

    pid: int
    beats: int
    last_event_ns: int
    last_beat_ns: int
    published: int
    coalesced: int
    dropped: int
    bars_published: int
    stale_quote: int
    flags: int

    @property
    def running(self) -> bool:
        return bool(self.flags & MD_BEACON_FLAG_RUNNING)

    @property
    def stopped(self) -> bool:
        return bool(self.flags & MD_BEACON_FLAG_STOPPED)

    @property
    def had_wall_clock(self) -> bool:
        """Whether the publishing session read a wall clock at all.

        ``last_beat_ns == 0`` is the sentinel for "it did not" — an offline replay ages
        nothing against one. Reporting that as a beat at the epoch would make every
        offline session look 56 years dead.
        """
        return self.last_beat_ns != 0


class Publisher(Enum):
    """What two readings of a beacon prove about the process writing it.

    Deliberately not ordered and not a severity: each value maps to a *different*
    action, and collapsing them into a scale would lose the only thing the beacon added.
    """

    #: No beacon was configured, it is not there yet, or this is the first reading.
    #: Fall back to the silence deadline; that is the pre-ADR-0030 behaviour and it is
    #: still correct, just blind.
    UNKNOWN = "unknown"
    #: The beat count did not move. The pass loop is not running.
    DEAD = "dead"
    #: The beat count did not move and the last beat said so on purpose.
    STOPPED = "stopped"
    #: The beat count went backwards, or the pid changed: a new publisher owns this
    #: file, and every counter read before it is from a different session.
    RESTARTED = "restarted"
    #: Beating, and its event clock has not moved. The process is alive and **nothing is
    #: reaching it** — a socket that is healthy and permanently silent looks exactly
    #: like this, and nothing else in the system can see it.
    STARVED = "starved"
    #: Beating, events arriving, and nothing the ring can carry moved. A quiet market.
    QUIET = "quiet"
    #: Beating and writing records. Market data is flowing; whatever compared nothing is
    #: downstream of the ring.
    PUBLISHING = "publishing"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class PublisherState:
    """A verdict on the publisher, with the deltas it was reached from."""

    state: Publisher
    reason: str
    #: Beats since the previous reading. ``0`` is the whole point.
    beats: int = 0
    published: int = 0
    coalesced: int = 0
    dropped: int = 0
    bars_published: int = 0
    stale_quote: int = 0
    #: Whether the core's event-time high-water mark moved between the two readings.
    event_advanced: bool = False

    @property
    def alive(self) -> bool:
        return self.state in (Publisher.STARVED, Publisher.QUIET, Publisher.PUBLISHING)


UNKNOWN_PUBLISHER = PublisherState(
    Publisher.UNKNOWN, "no market-data beacon: silence can only be timed, not diagnosed"
)


def _delta32(prev: int, now: int) -> int:
    """Difference between two 32-bit counters that may have wrapped.

    Exact for any two readings fewer than 2\\ :sup:`32` increments apart, which against
    a monitor that polls per window is five days at ten thousand slices a second. The
    alternative the writer could have chosen — saturating — would yield a delta of zero
    forever once pinned, so a live publisher's slice stream would read as quiet: the
    wrong answer to the only question this file is asked.
    """
    return (now - prev) % _U32


def publisher_state(prev: MdBeaconSnapshot, now: MdBeaconSnapshot) -> PublisherState:
    """Turn two readings into the one thing the ring cannot say.

    The order of the tests is the design. ``beats`` is asked first because every other
    question presumes a running loop; ``published`` is asked before the event clock
    because a record on the ring is proof an event arrived whatever the clock says; and
    the event clock is asked before "quiet" because *no events at all* and *events that
    moved nothing* are opposite faults with the same empty ring.
    """
    if now.pid != prev.pid or now.beats < prev.beats:
        return PublisherState(
            Publisher.RESTARTED,
            f"the beacon restarted (pid {prev.pid}→{now.pid}, beats {prev.beats}→{now.beats}): "
            "a new publisher owns this file and every count before it belongs to another session",
        )

    beats = now.beats - prev.beats
    published = _delta32(prev.published, now.published)
    coalesced = _delta32(prev.coalesced, now.coalesced)
    dropped = _delta32(prev.dropped, now.dropped)
    bars = _delta32(prev.bars_published, now.bars_published)
    stale = _delta32(prev.stale_quote, now.stale_quote)
    advanced = now.last_event_ns > prev.last_event_ns
    common = dict(
        beats=beats,
        published=published,
        coalesced=coalesced,
        dropped=dropped,
        bars_published=bars,
        stale_quote=stale,
        event_advanced=advanced,
    )

    if beats == 0:
        if now.stopped:
            return PublisherState(
                Publisher.STOPPED,
                f"the publisher stopped on purpose at beat {now.beats} (pid {now.pid}); "
                "a clean exit still leaves this monitor watching nothing",
                **common,
            )
        return PublisherState(
            Publisher.DEAD,
            f"the publisher's beat has not moved from {now.beats} (pid {now.pid}): its pass "
            "loop is not running, so this is a dead publisher and not a quiet market",
            **common,
        )
    if published or bars:
        return PublisherState(
            Publisher.PUBLISHING,
            f"the publisher is alive and writing: +{beats} beat(s), +{published} slice(s), "
            f"+{bars} bar(s), +{dropped} dropped. Market data is flowing, so whatever "
            "compared nothing is downstream of the ring",
            **common,
        )
    if not advanced:
        return PublisherState(
            Publisher.STARVED,
            f"the publisher is alive (+{beats} beat(s)) and its event clock has not moved from "
            f"{now.last_event_ns}: nothing is reaching the core at all. A socket that is "
            "healthy and permanently silent looks exactly like this",
            **common,
        )
    return PublisherState(
        Publisher.QUIET,
        f"the publisher is alive and the market is quiet: +{beats} beat(s), +{coalesced} "
        f"update(s) coalesced, +{stale} stale quote(s), and nothing the ring carries moved",
        **common,
    )


def _parse(raw: bytes) -> MdBeaconSnapshot:
    def u(off: int, n: int, signed: bool = False) -> int:
        return int.from_bytes(raw[off : off + n], "little", signed=signed)

    return MdBeaconSnapshot(
        pid=u(_OFF_PID, 4),
        beats=u(_OFF_BEATS, 8),
        last_event_ns=u(_OFF_LAST_EVENT_NS, 8, signed=True),
        last_beat_ns=u(_OFF_LAST_BEAT_NS, 8),
        published=u(_OFF_PUBLISHED, 4),
        coalesced=u(_OFF_COALESCED, 4),
        dropped=u(_OFF_DROPPED, 4),
        bars_published=u(_OFF_BARS_PUBLISHED, 4),
        stale_quote=u(_OFF_STALE_QUOTE, 4),
        flags=u(_OFF_FLAGS, 4),
    )


def _check_header(raw: bytes) -> None:
    magic = int.from_bytes(raw[_OFF_MAGIC : _OFF_MAGIC + 8], "little")
    if magic != MD_BEACON_MAGIC:
        raise BeaconError(f"not an Axon market-data beacon: bad magic {magic:#018x}")
    version = int.from_bytes(raw[_OFF_VERSION : _OFF_VERSION + 4], "little")
    if version != MD_BEACON_VERSION:
        raise BeaconError(f"unsupported beacon version {version}")


class MdBeaconReader:
    """The monitor's end: maps the beacon once and re-reads it on demand.

    **Opening is lazy, and that is not laziness.** A monitor is wired before the session
    it watches has started, so "the file is not there yet" is the ordinary startup race
    and must not be a construction error — the same rule the runtime applies to the
    signal ring, which it *opens*, and not the one it applies to the md ring, which it
    *creates*. So :meth:`read` returns ``None`` until a publisher has created **and**
    stamped the file, and the two cases are separated deliberately:

    * a magic of ``0`` is the window between ``ftruncate`` and the header store, or a
      path nobody has written yet — a race, so it reports "not ready";
    * **any other** wrong magic is a beacon of the wrong kind (``axon.live.liveness``'s
      is also 64 bytes with the same first 40) and raises, because that is a wiring
      mistake and it will not fix itself.

    Known limit: a publisher that unlinked and recreated the file rather than truncating
    it in place would leave this reader on the old inode, reading a beacon nobody writes
    any more. ``MdBeacon::create`` truncates in place, so that does not happen today.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._mm: mmap.mmap | None = None

    def _ensure_open(self) -> bool:
        if self._mm is not None:
            return True
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return False  # not created yet; the session may not have started
        try:
            if os.fstat(fd).st_size < MD_BEACON_SIZE:
                return False  # created, not yet sized
            # `access=` rather than `prot=`: the latter is the POSIX spelling and does
            # not exist on Windows, where the import succeeds and the attribute lookup
            # is what fails. `ACCESS_READ` is defined on both and means the same thing —
            # a read-only mapping, which is the property this reader wants stated rather
            # than a platform's name for it.
            self._mm = mmap.mmap(fd, MD_BEACON_SIZE, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        return self._mm is not None

    def read(self) -> MdBeaconSnapshot | None:
        """One snapshot, or ``None`` if no publisher has stamped the file yet."""
        if not self._ensure_open():
            return None
        # All 64 bytes in one copy, then parsed from the copy: it does not make the read
        # atomic — nothing can — but it narrows the window in which the writer can move a
        # field under us to a single memcpy, and it guarantees every field in one
        # snapshot came from one pass over the line rather than from ten spread out over
        # whatever the interpreter was doing.
        raw = self._mm[0:MD_BEACON_SIZE]
        if int.from_bytes(raw[_OFF_MAGIC : _OFF_MAGIC + 8], "little") == 0:
            return None  # created and not yet stamped
        _check_header(raw)
        return _parse(raw)

    def __call__(self) -> MdBeaconSnapshot | None:
        """So a reader *is* the probe :class:`~axon.parity.monitor.ParityMonitor` takes."""
        return self.read()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None

    def __enter__(self) -> "MdBeaconReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_md_beacon(path: str) -> MdBeaconSnapshot:
    """Read a beacon once, strictly. For tests and operators.

    Strict where :class:`MdBeaconReader` is tolerant: a missing or unstamped file raises
    here rather than reporting "not ready", because a one-shot caller asked a question
    and "maybe later" is not an answer to it.
    """
    with open(path, "rb") as f:
        raw = f.read(MD_BEACON_SIZE)
    if len(raw) < MD_BEACON_SIZE:
        raise BeaconError(f"file too small ({len(raw)} bytes) to be an Axon beacon")
    _check_header(raw)
    return _parse(raw)


__all__ = [
    "MD_BEACON_FLAG_RUNNING",
    "MD_BEACON_FLAG_STOPPED",
    "MD_BEACON_MAGIC",
    "MD_BEACON_SIZE",
    "MD_BEACON_VERSION",
    "UNKNOWN_PUBLISHER",
    "BeaconError",
    "MdBeaconReader",
    "MdBeaconSnapshot",
    "Publisher",
    "PublisherState",
    "beacon_path",
    "publisher_state",
    "read_md_beacon",
]
