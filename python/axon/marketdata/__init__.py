"""The Python consumer of the market-data ring (Rust → Python; ADR-0012).

The Rust core owns the venue connection, the order book and the event clock, and
publishes the slice of market state a feature computation needs (`docs/01` step 2).
This module is the reading end: Python never opens a second venue connection, so
features are always computed on the same book the executing core saw.

**Who writes it.** ``axon_runtime::mdring::MdPublisher``, from inside the core's own
event fan-out. Two properties of that publisher a consumer has to know:

* Under the default ``on_change`` write policy, a slice is written only when the
  state the record carries — the top of book and the last print — actually moved. So
  a quiet ring means *the top of book has not moved*, not that the feed is dead; a
  busy ``l2Book`` feed legitimately produces far fewer slices than updates. Nothing
  is lost by it: `seq` stays gap-free across coalesced updates, so a suppressed
  update is never mistaken for a drop by :attr:`MdRingConsumer.dropped`.
* A `seq` gap therefore means exactly one thing — the ring was full and the publisher
  dropped rather than blocked. It never waits for this reader (see below).
* **A zero bid or ask means "no quote", never a price.** The record carries one
  timestamp and it belongs to the event that *triggered* the slice, so nothing here
  can age the top of book for itself: a bid from five minutes ago and a live one look
  identical. The publisher therefore refuses to send one it would not price an order
  against — if every feed carrying a quote has gone quiet past the runtime's mark
  window, the four quote fields go out as zeros, which is the record's own sentinel
  for "nothing seen yet", and the Rust status line says ``MD QUOTE STALE``. Features
  must skip those rows rather than treat them as a market at zero; the trade fields on
  the same record are still good.

**Batch-first, deliberately.** Python's interpreter floor is ~50–350 ns *per call*
(`docs/02-python-rust-boundary.md`), which a per-message API pays on every update
— at a few thousand updates a second that is the whole latency budget, spent on
call overhead rather than on features. :meth:`MdRingConsumer.read_batch` returns
a NumPy structured array of everything queued, so the per-call cost is amortized
over the batch and the records arrive in a form :mod:`axon.features` can
vectorize over directly.

**The mark and funding ride the slice** (ADR-0028). ``mark_px``, ``index_px``,
``funding_rate``, ``funding_interval_ns`` and the mark's two clocks are on every
record. They are *not* a feed of their own, because Hyperliquid's ``activeAssetCtx``
carries no venue timestamp (ADR-0011) and a record ordered on our receipt clock does
not replay — so the ticker's values ride the next slice a venue-timed event triggers.
Two consequences for a consumer:

* **"Has a ticker been seen?" is ``mark_ts_ingest != 0``, never ``mark_px != 0``.**
  A venue is free to mark an instrument at zero, and the whole tail is zero until a
  ticker arrives.
* **``mark_ts_venue == 0`` means the mark has no venue time.** Its only clock is our
  receipt clock, so an age derived from it measures this machine and does not
  reproduce on replay. When it is non-zero it shares a clock with ``ts_event`` and
  ``ts_event - mark_ts_venue`` is a mark age a backtest can reproduce exactly.

**Bars are a second ring** (ADR-0028), because a bar is not a state snapshot:
:class:`MdBarRingConsumer` reads closed OHLCV bars from the sibling file
:func:`bar_ring_path` names. Only *closed* bars ever arrive there, and each carries
its own continuity flags — see that class.

Typical loop::

    from axon.marketdata import MdRingConsumer

    with MdRingConsumer("/dev/shm/axon-md.ring") as md:
        while running:
            batch = md.read_batch()
            if batch.size:
                update_features(batch)      # vectorized, one call per batch
            else:
                time.sleep(0)               # nothing queued; yield
"""

from __future__ import annotations

import os

import numpy as np

from axon.contracts import (
    MD_BAR_DTYPE,
    MD_BAR_FLAG_FIRST_BAR,
    MD_BAR_FLAG_GAP_BEFORE,
    MD_BAR_KIND_CLOSED,
    MD_FLAG_LAST_TRADE_SELL,
    MD_KIND_QUOTE,
    MD_KIND_SNAPSHOT,
    MD_KIND_TRADE,
    MD_SLICE_DTYPE,
)
from axon.signals import RingConsumer
from axon.strategy.events import Bar


def bar_ring_path(md_ring_path: str) -> str:
    """Where the bar ring lives, given the slice ring's path.

    Mirrors ``axon_runtime::mdring::bar_ring_path`` — ``/dev/shm/axon-md.ring``
    becomes ``/dev/shm/axon-md.bars.ring``. Derived rather than configured on either
    side, so an operator cannot enable the slice ring, forget the bar ring, and get a
    bar-driven strategy that starts cleanly and then simply never has an opinion. The
    runtime's startup banner prints both paths, so nothing about it is implicit at
    run time.
    """
    root, ext = os.path.splitext(md_ring_path)
    return f"{root}.bars{ext}" if ext else f"{md_ring_path}.bars"


class MdRingConsumer(RingConsumer):
    """Batch-first reader for the market-data ring.

    A :class:`~axon.signals.RingConsumer` pinned to :data:`MD_SLICE_DTYPE`, plus
    drop accounting. Opening a ring that carries anything else fails loudly rather
    than decoding foreign bytes as prices.
    """

    def __init__(self, path: str):
        super().__init__(path, dtype=MD_SLICE_DTYPE)
        #: Records the publisher had to drop because this reader fell behind.
        self.dropped = 0
        self._last_seq: int | None = None

    def read_batch(self, max_records: int | None = None) -> np.ndarray:
        """Read every queued slice (up to ``max_records``), oldest first.

        Also accounts for drops. The Rust publisher must never block on a slow
        Python — stalling the execution core to wait for a feature computation is
        the one failure this direction of the boundary cannot have — so a full ring
        makes it drop the update instead. `seq` is monotonic and gap-free at the
        source, so a span wider than the record count is exactly what was lost, and
        :attr:`dropped` is how a strategy learns its feature history has holes
        rather than silently computing on them.
        """
        batch = super().read_batch(max_records)
        if batch.size:
            first, last = int(batch["seq"][0]), int(batch["seq"][-1])
            self.dropped += (last - first + 1) - int(batch.size)
            if self._last_seq is not None:
                self.dropped += first - self._last_seq - 1
            self._last_seq = last
        return batch


def bar_from_record(rec) -> Bar:
    """One :data:`~axon.contracts.MD_BAR_DTYPE` record → a strategy-facing
    :class:`~axon.strategy.events.Bar`.

    ``ts_event`` crosses unchanged: the publisher already stamped the bar's **close**
    at the venue's ``T + 1 ms``, which is the same instant
    :data:`axon.strategies.data.CLOSE_STAMP_OFFSET_MS` puts on the same bar offline.
    Re-deriving it here — from ``open_time + interval_ms``, say — would be a second
    place for the two halves to drift one millisecond apart, and one millisecond is
    all it takes to make ``align_by_event_time`` intersect to nothing.

    ``int()`` on every field, deliberately: the record's fields are ``np.int64``, and
    a NumPy integer that later meets a float in a feature expression promotes the
    whole expression to ``float64`` — 53 mantissa bits against a 61-bit nanosecond
    timestamp, which silently rounds event times into ~256 ns buckets and makes two
    ordered events simultaneous.
    """
    return Bar(
        symbol_id=int(rec["symbol_id"]),
        ts_event=int(rec["ts_event"]),
        open=int(rec["open"]),
        high=int(rec["high"]),
        low=int(rec["low"]),
        close=int(rec["close"]),
        volume=int(rec["volume"]),
    )


class MdBarRingConsumer(RingConsumer):
    """Batch-first reader for the **bar** ring (ADR-0028).

    A :class:`~axon.signals.RingConsumer` pinned to
    :data:`~axon.contracts.MD_BAR_DTYPE`. ``MdBar`` and ``MdSlice`` have the *same*
    128-byte stride, so opening the wrong one of the two rings is caught by the
    header's ``record_kind`` and nothing else — without it this reader would decode a
    slice's ``bid_px`` as an open price and report plausible numbers forever.

    **Only closed bars arrive here.** The publisher emits a bar when the venue starts
    the next interval, because that is the only evidence of closure available that is
    not a wall clock. Two consequences a consumer has to know:

    * A bar arrives one venue frame *after* it closed — typically under a second, and
      its ``ts_event`` is still its own close time, so nothing downstream is skewed.
    * A session's **last** bar never arrives, because nothing ever proved it closed.
    * **What arrives is the venue's last *observed* frame, which is usually but not
      always the official bar.** Hyperliquid sends no closing frame at all — it
      republishes the bar it is filling and then simply starts the next one — so a
      trade printing in the final milliseconds is missing from ``volume``. Measured
      against ``candleSnapshot`` over seven consecutive BTC minutes (one small sample):
      six matched exactly, one was short by ``0.001`` in volume, its last frame having
      landed 35 ms before the close. So when a live-versus-offline diff shows a
      non-zero **volume** column, that is the venue, not a parity break — while a
      non-zero *price* column still is one, because OHLC only moves on a trade the
      final frame would have carried. See ADR-0028's consequences.

    **Continuity is reported, never repaired.** :attr:`gaps` counts bars flagged
    ``gap_before`` — a bar the venue should have printed and did not, or one the ring
    dropped. That is a different fault from :attr:`dropped`, which counts ``seq``
    holes, and both are different from an idle market. Nothing here interpolates a
    missing bar: an invented close is a price nothing traded at, and every return
    feature computed from it would be measuring our own arithmetic.
    """

    def __init__(self, path: str):
        super().__init__(path, dtype=MD_BAR_DTYPE)
        #: Bars the publisher had to drop because this reader fell behind.
        self.dropped = 0
        #: Bars whose predecessor is missing — a hole in the *feed*, not in the ring.
        self.gaps = 0
        #: Bars that opened a series, so continuity before them is unknown.
        self.first_bars = 0
        self._last_seq: int | None = None

    def read_batch(self, max_records: int | None = None) -> np.ndarray:
        """Read every queued bar (up to ``max_records``), oldest first.

        Returns the raw structured array, which is where ``flags``, ``open_time`` and
        ``interval_ms`` live — :meth:`read_bars` drops them, because
        :class:`~axon.strategy.events.Bar` has no field for them. Accounting for
        drops and gaps happens here, so it happens exactly once either way.
        """
        batch = super().read_batch(max_records)
        if batch.size:
            first, last = int(batch["seq"][0]), int(batch["seq"][-1])
            self.dropped += (last - first + 1) - int(batch.size)
            if self._last_seq is not None:
                self.dropped += first - self._last_seq - 1
            self._last_seq = last
            flags = batch["flags"]
            self.gaps += int(np.count_nonzero(flags & MD_BAR_FLAG_GAP_BEFORE))
            self.first_bars += int(np.count_nonzero(flags & MD_BAR_FLAG_FIRST_BAR))
        return batch

    def read_bars(self, max_records: int | None = None) -> list[Bar]:
        """Read every queued bar as :class:`~axon.strategy.events.Bar` events.

        The shape a live strategy is driven by::

            from axon.live import StrategyRunner
            from axon.marketdata import MdBarRingConsumer, bar_ring_path

            with MdBarRingConsumer(bar_ring_path("/dev/shm/axon-md.ring")) as feed:
                while running:
                    for bar in feed.read_bars():
                        runner.handle(bar)      # dispatches to Strategy.on_bar
                    if feed.gaps:
                        ...                     # the feature history has a hole

        A list rather than a generator: the batch is already materialized, and a lazy
        wrapper would let a caller hold the sequence past the next
        :meth:`read_batch`, whose ``tail`` publish releases the slots it was built
        from.
        """
        return [bar_from_record(rec) for rec in self.read_batch(max_records)]


__all__ = [
    "MdRingConsumer",
    "MdBarRingConsumer",
    "bar_from_record",
    "bar_ring_path",
    "MD_SLICE_DTYPE",
    "MD_KIND_QUOTE",
    "MD_KIND_TRADE",
    "MD_KIND_SNAPSHOT",
    "MD_FLAG_LAST_TRADE_SELL",
    "MD_BAR_DTYPE",
    "MD_BAR_KIND_CLOSED",
    "MD_BAR_FLAG_GAP_BEFORE",
    "MD_BAR_FLAG_FIRST_BAR",
]
