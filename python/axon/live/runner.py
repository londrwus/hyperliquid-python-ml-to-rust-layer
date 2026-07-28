"""The live bridge: a :class:`~axon.strategy.base.Strategy` wired to the signal ring.

This is the Python half of Phase 4's "a trivial Python strategy emits target
positions; Rust executes on testnet" (``docs/08``). The loop is deliberately
small — dispatch an event, bind its time, drain what the callback emitted, push —
because everything interesting is in the three things it refuses to do:

**It never stamps a signal with a wall clock.** The event's own ``ts_event`` is
bound around the callback and the context has no other source of time.

**It never drops a signal because the ring is full.** A full ring means the Rust
consumer is behind or dead, and the natural-looking response — skip this one,
the next target supersedes it anyway — is wrong in the case that matters: if the
consumer is dead, there *is* no next one, and the venue is left holding the
previous target. Full is backpressure. Records queue, in order, and if the queue
also overflows the runner raises rather than inventing a policy.

**It never lets silence be ambiguous.** A target-position strategy emits only
when its target changes, so an idle strategy and a crashed one produce the same
(empty) ring. The heartbeat in :mod:`axon.live.liveness` is what separates them.

Ordering note: the outbox is strictly FIFO. Pushing a newer record past a stuck
older one would make ``seq`` non-monotonic on the ring, and the consumer's gap
detector — the one mechanism that can prove nothing was lost — would be reading
noise.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import numpy as np

from axon.live.liveness import FLAG_RUNNING, FLAG_STOPPED, LivenessBeacon
from axon.signals import RingProducer
from axon.strategy.base import Strategy
from axon.strategy.config import StrategyConfig
from axon.strategy.context import DEFAULT_TTL_MS, StrategyContext
from axon.strategy.events import Bar, Bbo, Fill, OrderUpdate, Tick, Timer, Trade

#: Event type → callback name. Exact-type first, then the MRO, so a strategy can
#: subclass an event type without falling off the dispatch table.
_DISPATCH: dict[type, str] = {
    Tick: "on_tick",
    Trade: "on_trade",
    Bbo: "on_bbo",
    Bar: "on_bar",
    Fill: "on_fill",
    OrderUpdate: "on_order_update",
    Timer: "on_timer",
}


class LiveError(Exception):
    """Base for live-bridge failures."""


class BackpressureError(LiveError):
    """The ring stayed full long enough to overflow the outbox.

    Raised rather than resolved because there is no safe local answer: the
    consumer is gone or wedged, and the only correct responses (flatten via the
    venue-side dead-man's switch, restart the core, page someone) live above this
    layer. Nothing is lost when it raises — the outbox still holds every record,
    so a caller that clears the blockage can call :meth:`StrategyRunner.flush`
    and continue.
    """


@dataclass
class RunnerStats:
    """Counters a supervisor can scrape. Snapshot via :attr:`StrategyRunner.stats`."""

    events_handled: int = 0
    signals_emitted: int = 0
    signals_pushed: int = 0
    backpressure_events: int = 0
    max_outbox_depth: int = 0
    out_of_order_events: int = 0
    beats: int = 0

    @property
    def signals_in_flight(self) -> int:
        """Emitted but not yet on the ring. Non-zero means the consumer is behind."""
        return self.signals_emitted - self.signals_pushed


class StrategyRunner:
    """Drives one strategy onto one signal ring.

    Creating a runner **creates and re-initializes** the ring file: ``RingProducer``
    zeroes ``head`` and ``tail``. A consumer already attached to that path sees the
    indices rewind, so restart the Rust side alongside the Python side rather than
    reattaching a live runner to a running core.
    """

    def __init__(
        self,
        strategy: Strategy,
        *,
        ring_path: str,
        capacity: int = 1024,
        model_version: int = 1,
        default_ttl_ms: int = DEFAULT_TTL_MS,
        max_outbox: int | None = None,
        liveness_path: str | None = None,
        peer_timeout_ms: int = 2_000,
        first_seq: int = 0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        wall_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.strategy = strategy
        self._producer = RingProducer(ring_path, capacity=capacity)
        self._ctx = StrategyContext(
            model_version=model_version,
            default_ttl_ms=default_ttl_ms,
            first_seq=first_seq,
        )
        # One ring's worth of slack beyond the ring itself: enough to ride out a
        # consumer GC pause or a reconnect, small enough that a genuinely dead
        # consumer is reported in bounded time instead of being absorbed forever.
        self._max_outbox = capacity if max_outbox is None else max_outbox
        self._outbox: deque[np.ndarray] = deque()
        self._stats = RunnerStats()

        self._beacon = LivenessBeacon(
            liveness_path if liveness_path is not None else ring_path + ".live",
            wall_ns=wall_ns,
        )
        # Liveness is a wall-clock question by nature, so this is the one place in
        # the package that reads a clock — and it reads a *monotonic* one, because
        # an NTP step must not be mistaken for a peer coming back to life.
        self._monotonic_ns = monotonic_ns
        self._peer_timeout_ns = peer_timeout_ms * 1_000_000

        self._pushed = 0  # == the ring's head; we created it, so it started at 0
        self._last_tail = 0
        self._last_tail_move_ns = monotonic_ns()
        self._last_event_ns = 0
        self._started = False
        self._stopped = False

    @classmethod
    def from_config(
        cls, strategy: Strategy, config: StrategyConfig, *, ring_path: str, **kwargs: Any
    ) -> "StrategyRunner":
        """Build a runner whose stamping comes from the archived config, not call sites."""
        return cls(
            strategy,
            ring_path=ring_path,
            model_version=config.model_version,
            default_ttl_ms=config.default_ttl_ms,
            **kwargs,
        )

    # ── properties ──

    @property
    def ctx(self) -> StrategyContext:
        return self._ctx

    @property
    def stats(self) -> RunnerStats:
        return replace(self._stats)

    @property
    def beacon_path(self) -> str:
        """Where the heartbeat lives, so a supervisor can be pointed at it."""
        return self._beacon.path

    @property
    def started(self) -> bool:
        """Whether :meth:`start` has run.

        Public because a driver that dispatches events one at a time — rather than
        handing :meth:`run` a whole iterable — has to auto-start on the first
        event's own time, and reading the private flag to do it is how a caller
        ends up depending on this class's internals.
        """
        return self._started

    @property
    def last_event_ns(self) -> int:
        """Event time of the newest event handled. ``0`` before the first one.

        The only defensible argument to :meth:`stop`: there is no "now" in event
        time, and a stop stamped from a wall clock would be a signal scope with a
        timestamp no event in the session ever carried.
        """
        return self._last_event_ns

    @property
    def pending_out(self) -> int:
        """Records emitted but not yet accepted by the ring."""
        return len(self._outbox)

    @property
    def ring_depth(self) -> int:
        """Records on the ring the consumer has not taken yet."""
        return len(self._producer)

    @property
    def consumer_tail(self) -> int:
        """The consumer's index, derived as ``head - depth``.

        Derived rather than read out of ``RingProducer``'s internals: the ring's
        representation belongs to :mod:`axon.signals`, and reaching into it would
        make this module break the next time that file is optimized.
        """
        return self._pushed - len(self._producer)

    # ── lifecycle ──

    def start(self, ts_event: int) -> None:
        """Run ``on_start`` at the session's event time.

        The time is required, not defaulted to "now", because there is no "now" in
        event time — a session that starts by replaying a snapshot starts at the
        snapshot's timestamp, and only the caller knows what that is.
        """
        if self._started:
            raise LiveError("runner already started")
        with self._ctx.event(ts_event):  # validates the time before anything is recorded
            self.strategy.on_start(self._ctx)
        self._started = True
        self._last_event_ns = ts_event
        self._drain()
        self.beat()

    def handle(self, event: Any) -> None:
        """Dispatch one event, then drain and push whatever it produced."""
        if not self._started:
            raise LiveError("call start(ts_event) before handling events")
        if self._stopped:
            raise LiveError("runner is stopped")
        callback = self._callback_for(event)
        # Binding first is what validates the time; nothing is recorded from an
        # event whose timestamp the contract would have rejected.
        with self._ctx.event(event.ts_event) as ctx:
            ts_event = ctx.ts_event
            if ts_event < self._last_event_ns:
                # Counted, not refused: dropping a real event is worse than
                # handling a late one. But it is recorded, because a signal
                # stamped earlier than its predecessor breaks replay ordering,
                # and a silently wrong replay invalidates the parity harness
                # the whole of Phase 5 is built on (docs/07).
                self._stats.out_of_order_events += 1
            else:
                self._last_event_ns = ts_event
            callback(event, ctx)
        self._stats.events_handled += 1
        self._drain()
        self.beat()

    def run(self, events: Iterable[Any], *, stop: bool = True) -> RunnerStats:
        """Drive a whole event sequence.

        Auto-starts on the first event's time if :meth:`start` was not called, and
        stops on the last event's time — so a replay's start and stop are taken
        from the data, exactly as a backtest would.
        """
        for event in events:
            if not self._started:
                self.start(event.ts_event)
            self.handle(event)
        if stop and self._started and not self._stopped:
            self.stop(self._last_event_ns)
        return self.stats

    def stop(self, ts_event: int) -> None:
        """Run ``on_stop`` and mark the beacon cleanly stopped.

        A crash skips this, which is the point: the peer sees a stalled counter
        with no stop flag and treats it as death. Note that even a *clean* stop
        does not make the position safe — that is the venue-side dead-man's
        switch's job (``docs/02``).
        """
        if not self._started or self._stopped:
            return
        with self._ctx.event(ts_event):
            self.strategy.on_stop(self._ctx)
        self._drain()
        self._stopped = True
        self.beat()

    def close(self) -> None:
        """Release the ring and the beacon. Does not mark a clean stop by itself."""
        self._producer.close()
        self._beacon.close()

    def __enter__(self) -> "StrategyRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── the ring ──

    def flush(self) -> int:
        """Push as much of the outbox as the ring will take. Returns records pushed."""
        pushed = 0
        try:
            while self._outbox:
                if not self._producer.try_push(self._outbox[0]):
                    self._stats.backpressure_events += 1
                    if len(self._outbox) > self._max_outbox:
                        # Publish the state that explains the failure before
                        # unwinding: a supervisor scraping the beacon must see the
                        # overflow, not a heartbeat from before it happened.
                        self._stats.max_outbox_depth = max(
                            self._stats.max_outbox_depth, len(self._outbox)
                        )
                        self.beat()
                        raise BackpressureError(
                            f"signal ring full and outbox at {len(self._outbox)} "
                            f"(limit {self._max_outbox}): the consumer is not draining. "
                            "Nothing was dropped; clear the blockage and call flush() again."
                        )
                    break
                self._outbox.popleft()
                self._pushed += 1
                self._stats.signals_pushed += 1
                pushed += 1
        finally:
            # Counters stay truthful on the raising path too, or the numbers a
            # post-mortem is read from would be the ones the failure invalidated.
            self._stats.max_outbox_depth = max(self._stats.max_outbox_depth, len(self._outbox))
            self._observe_peer()
        return pushed

    def beat(self, *, flags: int | None = None) -> int:
        """Publish a heartbeat.

        An event-driven runner beats after every event; a runner blocked waiting
        for a quiet market must call this on a timer anyway, or the peer cannot
        tell a quiet market from a dead process.
        """
        if flags is None:
            flags = FLAG_STOPPED if self._stopped else FLAG_RUNNING
        beats = self._beacon.beat(
            last_event_ns=self._last_event_ns,
            signals=self._stats.signals_emitted,
            backpressure=self._stats.backpressure_events,
            pending=len(self._outbox),
            flags=flags,
        )
        self._stats.beats = beats
        return beats

    def peer_stalled(self) -> bool:
        """Whether the consumer looks dead: backlog present and ``tail`` frozen.

        Only meaningful when there is something to consume. An empty ring means
        the consumer has nothing to do, and reporting *idle* as *dead* would trip
        exactly the response (flatten everything) that a working system must not
        take on a quiet market.
        """
        self._observe_peer()
        if not self._outbox and len(self._producer) == 0:
            return False
        return self._monotonic_ns() - self._last_tail_move_ns >= self._peer_timeout_ns

    # ── internals ──

    def _callback_for(self, event: Any) -> Callable[[Any, StrategyContext], None]:
        for cls in type(event).__mro__:
            name = _DISPATCH.get(cls)
            if name is not None:
                return getattr(self.strategy, name)
        # An unroutable event type is a strategy that mysteriously never trades;
        # far better to fail at the first one than to discover it in a P&L review.
        raise TypeError(f"no strategy callback for event type {type(event).__name__}")

    def _drain(self) -> None:
        pending = self._ctx.take_pending()
        if pending:
            self._stats.signals_emitted += len(pending)
            self._outbox.extend(pending)
        self.flush()

    def _observe_peer(self) -> None:
        tail = self.consumer_tail
        if tail != self._last_tail:
            self._last_tail = tail
            self._last_tail_move_ns = self._monotonic_ns()


__all__ = ["BackpressureError", "LiveError", "RunnerStats", "StrategyRunner"]
