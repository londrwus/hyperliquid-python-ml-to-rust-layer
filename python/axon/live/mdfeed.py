"""The other half of the live bridge: the Rust core's market-data ring as strategy events.

:mod:`axon.live.runner` drives a strategy from *an iterable of events* and
:mod:`axon.marketdata` reads the ring the Rust core publishes into. Nothing joined
them, so the only thing that had ever driven a live :class:`~axon.live.runner.StrategyRunner`
was a synthetic generator — which is why the Python→Rust direction could be proved
offline and never at a venue. This module is that join, and it is four lines of loop
around three decisions that each have an obvious wrong answer.

**Every event's ``ts_event`` is the slice's, which is the venue's.** Not
``time.time_ns()`` at the moment Python read the ring. This is not a style
preference: :class:`~axon.strategy.context.StrategyContext` stamps an emitted
signal with the event's time, and the Rust ``SignalReader`` ages that stamp against
the *core's own event clock* with a ceiling of ``intent.max_signal_age_ms``. A
signal stamped from any other clock is admitted or refused for reasons that have
nothing to do with how stale the decision is — and on a machine whose clock leads
the venue's it is admitted while being arbitrarily old. The venue's own time is the
only stamp both sides agree about.

**A slice becomes a :class:`~axon.strategy.events.Bbo` and never a
:class:`~axon.strategy.events.Trade`.** ``MdSlice`` carries the *last* print, not
each print, and under the publisher's default ``on_change`` policy any quote move
republishes the same print on the next slice. Synthesizing a ``Trade`` per slice
would therefore count one execution several times, and a strategy keying on trade
arrival would read a quote-flicker storm as a burst of volume. The print is still
on the record (:attr:`MdRingFeed.last_slice`) for anything that wants it as state.

**A zero bid or ask is "no quote", never a price.** The publisher zeroes all four
quote fields rather than republishing a top of book it would not itself price an
order against (ADR-0012 / ``axon.marketdata``). Such a slice is skipped rather than
handed on as a market at zero — a strategy differencing against a zero mid emits a
target the size of the whole book.

The poll loop reads a wall clock, once, to decide how long to sleep between empty
reads. That is I/O pacing and never reaches a signal: nothing here stamps anything
with it, which is the same line ``LazyRing::ensure`` draws on the Rust side.

Typical use::

    with MdRingFeed("/dev/shm/axon-md.ring", symbol_id=0) as feed:
        runner.run(feed.events(max_events=500), stop=True)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from axon.contracts import MD_KIND_QUOTE, MD_KIND_SNAPSHOT
from axon.marketdata import MdRingConsumer
from axon.strategy.events import Bbo


class MdFeedError(Exception):
    """The market-data ring could not be attached to in time."""


@dataclass
class FeedStats:
    """What the feed made of the ring. A supervisor's denominator."""

    slices_read: int = 0
    events_yielded: int = 0
    #: Slices the publisher refused to put a quote on because every feed carrying
    #: one had gone quiet past the runtime's mark window. Not a drop and not an
    #: error — but a rising count means the strategy is being fed nothing.
    no_quote: int = 0
    #: Slices whose kind is neither a quote nor a book snapshot: they carry state
    #: but the top of book is not what moved.
    other_kind: int = 0
    #: Slices for an instrument this feed was not asked about.
    other_symbol: int = 0
    #: Records the publisher had to drop because this reader fell behind, taken
    #: from the consumer's own `seq` accounting.
    dropped: int = 0


class MdRingFeed:
    """Read one instrument's slices off the market-data ring as ``Bbo`` events.

    ``symbol_id`` of ``None`` yields every instrument on the ring, which is what a
    multi-symbol strategy wants and what a single-symbol one must not have: the
    events would interleave two books through one set of callbacks.
    """

    def __init__(
        self,
        path: str,
        *,
        symbol_id: int | None = None,
        poll_s: float = 0.001,
        batch: int = 256,
    ) -> None:
        self._consumer = MdRingConsumer(path)
        self.path = path
        self.symbol_id = symbol_id
        self._poll_s = poll_s
        self._batch = batch
        self.stats = FeedStats()
        #: The most recent slice seen, whatever its kind — the escape hatch for the
        #: trade print and (schema permitting) the mark, neither of which this feed
        #: turns into an event of its own.
        self.last_slice: np.ndarray | None = None

    @staticmethod
    def wait_for_ring(path: str, *, timeout_s: float = 30.0, poll_s: float = 0.05) -> "MdRingFeed":
        """Open the ring once the Rust core has created it, or fail saying so.

        The core creates this file; Python only reads it. Waiting rather than
        failing immediately is the same reasoning the Rust side applies to the
        *signal* ring in the other direction — one process legitimately starts
        before the other, and turning that race into an outage is worse than
        waiting for it. Bounded, because a ring that never appears means the
        session was started without ``md_ring.enabled``, and a driver that waits
        forever for that reads exactly like a quiet market.
        """
        import os

        deadline = time.monotonic() + timeout_s
        last: OSError | None = None
        while time.monotonic() < deadline:
            if os.path.exists(path):
                try:
                    return MdRingFeed(path)
                except (OSError, ValueError) as e:
                    # The core creates and sizes the file in two steps, so a read
                    # landing between them sees a short or header-less mapping.
                    # Retry rather than give up: the next attempt is milliseconds away.
                    last = e  # type: ignore[assignment]
            time.sleep(poll_s)
        raise MdFeedError(
            f"no market-data ring at {path} after {timeout_s:g}s"
            f"{f' (last error: {last})' if last else ''}. The Rust session creates it; "
            "check that its config has [md_ring] enabled and names this path."
        )

    def __enter__(self) -> "MdRingFeed":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._consumer.close()

    def events(
        self,
        *,
        max_events: int | None = None,
        until_ns: int | None = None,
        idle_timeout_s: float | None = None,
    ) -> Iterator[Bbo]:
        """Yield ``Bbo`` events until a stop condition is met.

        ``until_ns`` is compared against the *event* clock, so a caller can say
        "ten seconds of market" and mean the venue's ten seconds rather than this
        machine's. ``idle_timeout_s`` is the one wall-clock bound, and it exists
        because a feed that has genuinely stopped and a market that has genuinely
        gone quiet are the same silence — an unbounded generator over a dead ring
        never returns and never says why.
        """
        yielded = 0
        last_data = time.monotonic()
        while True:
            batch = self._consumer.read_batch(self._batch)
            self.stats.dropped = self._consumer.dropped
            if batch.size == 0:
                if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
                    return
                time.sleep(self._poll_s)
                continue
            last_data = time.monotonic()
            for rec in batch:
                self.stats.slices_read += 1
                self.last_slice = rec
                event = self._to_event(rec)
                if event is None:
                    continue
                if until_ns is not None and event.ts_event > until_ns:
                    return
                self.stats.events_yielded += 1
                yielded += 1
                yield event
                if max_events is not None and yielded >= max_events:
                    return

    # ── internals ──

    def _to_event(self, rec: np.ndarray) -> Bbo | None:
        if self.symbol_id is not None and int(rec["symbol_id"]) != self.symbol_id:
            self.stats.other_symbol += 1
            return None
        if int(rec["kind"]) not in (MD_KIND_QUOTE, MD_KIND_SNAPSHOT):
            self.stats.other_kind += 1
            return None
        bid_px, ask_px = int(rec["bid_px"]), int(rec["ask_px"])
        if bid_px <= 0 or ask_px <= 0:
            # The record's own sentinel for "nothing this reader should price
            # against". Passing it on as a market at zero is how a mid-differencing
            # strategy emits a target the size of the whole book.
            self.stats.no_quote += 1
            return None
        return Bbo(
            symbol_id=int(rec["symbol_id"]),
            ts_event=int(rec["ts_event"]),
            bid_px=bid_px,
            bid_sz=int(rec["bid_sz"]),
            ask_px=ask_px,
            ask_sz=int(rec["ask_sz"]),
        )


__all__ = ["FeedStats", "MdFeedError", "MdRingFeed"]
