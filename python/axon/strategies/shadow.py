"""Shadow trading a bar strategy: the serving path run forward, diffed as it goes.

Rung 3 of the ladder in ``docs/07`` is *"shadow trade: live feed, would-be orders,
continuous diff vs offline"*. This module is that loop, and it is worth being precise
about which of those four words it can deliver and which it cannot, because a rung
claimed is worse than a rung honestly half-climbed.

* **The serving path is real.** A :class:`~axon.strategies.perp_bar.PerpBar` — or any
  object with the same ``on_bar``/``feature_row``/``target``/``spec``/``params``
  surface, which is what ``--strategy`` exists for — driven one closed bar at a time
  through the real :class:`~axon.live.StrategyRunner`, whose emitted records cross a
  real SPSC ring and are read back off it. Nothing here is a stand-in for the Python
  half. (:class:`ShadowTrader`'s annotation still says ``PerpBar`` and is narrower
  than the behaviour: agent P4 drove a ``baseline_z`` spec through it unmodified at
  ``780/780`` owed rows.)
* **The bars are real venue prints**, from whichever :class:`BarSource` the caller
  attaches: the Rust core's market-data bar ring (ADR-0028), or a cached
  ``candleSnapshot`` history the venue actually published.
* **The diff is continuous**, one window at a time, through
  :class:`~axon.parity.monitor.ParityMonitor` — not a single verdict at the end.
* **The would-be orders are not orders.** What ``perp_bar`` emits is a *target
  position* on the signal ring. The Rust planner turns a target into an order against
  the book, the tick/lot grid and whatever is already resting; none of that exists in
  a shadow run, so :class:`ShadowSignal` is named for what it is — the record the
  planner *would* have been handed. It reached no venue, was never acknowledged,
  never rested and never filled, which is the same discipline
  :class:`axon.backtest.PlannedOrder` holds itself to.

Eight decisions carry the design, and each has a comfortable wrong answer. **The last
three were written by the first live session** — decisions 6, 7 and 8 close checks that
fired on a perfectly healthy m1 feed and were unreachable offline, which is the whole
argument for running a harness rather than testing one.

**1. The offline reference is a recompute over exactly the bars the serving path was
shown — and that is what makes a dropped bar invisible to the diff.** Recomputing
from a separately downloaded history would compare two feeds as well as two code
paths, and every venue gap would read as a parity failure. So the shadow trader keeps
its own recording of every bar it handed the strategy and recomputes over that. The
consequence has to be stated rather than discovered: a bar the *ring* dropped never
reached either side, so both agree perfectly about a window that is silently one
observation short. :attr:`FeedHealth.ring_dropped` is the only thing in this module
that can see that fault, which is why it is part of the verdict and not decoration.

**2. Drift is not measured, deliberately.** The monitor is configured with no
reference sample, so it reports *"drift not measured"* rather than *"drift stable"* —
opposite statements, and only one of them would be true. ADR-0030 measured the reason:
over ``perp_bar``'s own history in 256-row windows, **with feature parity green on
every window**, PSI passes the conventional 0.25 band on 18 of 20 BTC windows and
peaks at 5.97. A shadow monitor that alarmed on that would alarm forever, and an
operator who learns to ignore the drift line has learned to ignore the parity line.

**3. The silence deadline is a multiple of the bar interval, never a constant.**
:data:`~axon.parity.monitor.DEFAULT_SILENCE_AFTER_NS` is 60 s, which is correct for a
quote feed and absurd for an hourly bar strategy: a perfectly healthy ``perp_bar``
session is silent for an hour at a time and would alarm as a dead feed sixty seconds
in. The deadline here is :data:`SILENCE_AFTER_BARS` intervals, and the floor is two,
because ADR-0028's publisher emits a bar only when the venue starts the next one — so
a bar legitimately arrives one frame *after* its close and a one-interval deadline
would fire on every healthy bar.

**4. A duplicate or out-of-order bar is refused, not repaired.** Two bars for one
instrument at the same close time would be appended to the serving buffer twice, and
every window from that point on is one observation wide in the wrong place — with no
symptom, because both sides of the diff would see the same corrupted recording.
:class:`ShadowError` says so at the first one.

**5. Each window declares what it owed, rather than letting the alignment guess.**
Every window's offline side is built as exactly the rows the serving path owed, so the
monitor is configured ``scope="declared"``. The default infers the owed span from the
online side's own first stamp — right for a monitor window that opens mid-history, and
blind here, because *a serving path that produced nothing for a window's opening rows
has exactly the same first stamp as one that started on time.* Under inference those
rows are excused as a late join **and removed from the denominator**, so the run's own
coverage ratio shrinks to fit the damage and still reads perfect. Declaring is the only
way out, because there is nothing in the data to infer it from.

**6. There is a denominator below the row denominator, and only the bars' own close
times can supply it.** Decision 5 makes the *row* count honest: how many feature rows
the serving path owed over the bars it was shown. It says nothing about how many bars
it *should have been shown*. A feed that delivered thirty of sixty minutes is compared
on those thirty, agrees to the last bit, and prints ``max_abs_diff=0.000e+00`` beside a
perfect ``coverage`` ratio — every field a reader looks at identical to a healthy run.
The headline number is at its most reassuring exactly when the feed is at its most
broken. :class:`BarCoverage` is the arithmetic that catches it: closed bars are a
*cadence*, so ``(last_close − first_close) / interval + 1`` is how many the interval
promised, against how many arrived. It is derived from the recording's own event times
and never from a wall clock — a wall-clock bar count would measure how long this
process happened to be attached.

It is reported and does **not** fail the run, for the same reason
:attr:`FeedHealth.feed_gaps` does not: a minute in which nothing traded is a minute the
venue may legitimately not print. What it must never be is invisible.

**7. The feed is called dead on the clock that would notice, which is the bar's and
not the window's.** The monitor's progress clock advances when *a window compares
something*, and a window is :attr:`ShadowTrader.window_bars` bars wide — so between two
healthy flushes the monitor is legitimately silent for far longer than a deadline
measured in 2.5 bar intervals, and answers ``SILENT`` to every question asked in
between. ``SILENT`` outranks ``WARN``. The first live m1 session ever driven through
this module logged **296 SILENT verdicts in 160 seconds** of a feed delivering a bar a
minute, and would have reported ``FAIL`` with every cell it compared exactly right.
:meth:`ShadowTrader.heartbeat` therefore asks only once a **bar** is overdue. Nothing
offline could ever have seen it: :class:`HistoryBarSource` is never idle, so the idle
branch is dead code in every test that predates a socket.

**8. A would-be order is born as old as its bar took to arrive, and this module can
only say so — it cannot refuse one.** :class:`~axon.live.StrategyRunner` stamps a
signal with the *event's own* ``ts_event``, so a bar strategy's signal carries the
bar's **close**. The live intent path then judges that stamp against
``CoreHandler::last_ts()`` under a 2 000 ms ceiling. A shadow run reaches none of that,
which means it will cheerfully report would-be orders a live session would have dropped
as ``Expired`` — so :class:`ArrivalLag` measures the age and prints the arithmetic
beside the count. A transcript of orders that could never have been placed is worse
than no transcript.

Everything here **orders** on ``ts_event``, and nothing here orders on a wall clock.
Three places read one, and each is a measurement of something an event clock cannot
express:

* the poll pacing, which is not ordering;
* the silence deadline, which is the only clock that can measure an *absence* — an
  event-time deadline for *"nothing has arrived"* can never expire, because the clock
  that would advance it is the thing that stopped;
* :class:`ArrivalLag`, which is the wall clock **minus** the event clock on purpose:
  the whole quantity is the distance between the two, and there is no way to measure
  it inside either one.

Typical use::

    from axon.strategies.shadow import RingBarSource, ShadowTrader

    with RingBarSource.attach("/dev/shm/axon-md.ring", symbol_id=3) as source:
        trader = ShadowTrader(strategy, symbol_id=3, ring_path="/dev/shm/axon-m5.ring")
        report = trader.run(source, idle_timeout_s=600.0)
    print(report.summary())

and the operator must have started the Rust session with the candle feed on —
``feeds = ["Bbo", "L2Book", "Ticker", { Candles = "m1" }]`` plus ``[md_ring] enabled
= true``. Neither is a default and neither needs a code change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Protocol, Sequence

import numpy as np

from axon.contracts import FIXED_POINT_SCALE
from axon.features import FeatureSpec, bar_inputs, finite_rows
from axon.parity.features import align_by_event_time
from axon.parity.gate import ParityError
from axon.parity.monitor import (
    Clock,
    MonitorConfig,
    MonitorReport,
    ParityMonitor,
    Verdict,
    Window,
)
from axon.strategies.data import INTERVAL_MS, Candles
from axon.strategies.perp_bar import PERP_BAR_V1, PerpBar
from axon.strategies.training import MAKER_FEE_BPS, TAKER_FEE_BPS
from axon.strategy.events import Bar

#: Bars per parity window. Small enough that a blind serving path is caught within
#: an hour of hourly bars rather than at the end of the run, large enough that the
#: alignment has something to align: a one-row window is a comparison whose
#: denominator is one, which is the invisible-denominator bug with extra steps.
DEFAULT_WINDOW_BARS = 64

#: Bar intervals of silence before the feed is called dead rather than quiet. The
#: floor is **two**, not one: ADR-0028's publisher emits a bar only when the venue
#: starts the next interval, so a bar arrives one frame after its own close and a
#: one-interval deadline would fire on every healthy bar in the session.
SILENCE_AFTER_BARS = 2.5

#: The signal reader's shipped admission ceiling (``intent.max_signal_age_ms``), in
#: milliseconds. A shadow run does **not** go through that reader — nothing here is
#: admitted or refused — so it is quoted rather than enforced, and it is quoted because
#: a would-be order a live session would have thrown away is not a would-be order.
DEFAULT_MAX_SIGNAL_AGE_MS = 2_000

#: How far the **core's event clock** runs behind wall time on this venue, measured
#: live on testnet BTC's ``bbo`` by agent P1: **1 564 ms**, in a range of 1.4–1.9 s.
#:
#: It belongs in this module because it is what turns a bar's close time into an age.
#: The signal reader judges a record against ``CoreHandler::last_ts()``, not against a
#: wall clock, so a signal stamped with an event time is already ``arrival_lag −
#: core_clock_lag`` old at the pass that reads it — which can be *negative*, and P1
#: measured that the reader admits a negative age and counts it as ``ahead_of_clock``.
MEASURED_CORE_CLOCK_LAG_MS = 1_564

#: How often the idle branch asks the monitor for a silence verdict, in seconds.
#: Rate-limited because the poll loop spins far faster than any bar cadence, and a
#: heartbeat per poll would bury the one verdict that matters under ten thousand
#: that do not.
HEARTBEAT_EVERY_S = 1.0


class ShadowError(RuntimeError):
    """A shadow run cannot be trusted to be diffing what it thinks it is."""


# ── which columns a bar field can move ───────────────────────────────────────


def column_sources(spec: FeatureSpec) -> Mapping[str, frozenset[str]]:
    """Every bar input each column of ``spec`` transitively reads.

    A :class:`~axon.features.FeatureDef` may bind an input to an **earlier column**
    of the same spec rather than to a bar field, so a column's real dependencies are
    the transitive closure and not ``def.sources``. Resolved in declaration order,
    which is the order the spec already guarantees.
    """
    resolved: dict[str, frozenset[str]] = {}
    for f in spec.features:
        acc: set[str] = set()
        for s in f.sources:
            acc |= resolved.get(s, frozenset({s}))
        resolved[f.column] = frozenset(acc)
    return resolved


def volume_derived_columns(spec: FeatureSpec) -> tuple[str, ...]:
    """The columns a bar's ``volume`` can move, in spec order.

    This split is the only way to read a **live-versus-venue** diff correctly, and
    ADR-0028 is why. A published bar is the venue's last *observed* frame — the
    venue sends no closing frame at all — so a trade printing in the final
    milliseconds is missing from ``v`` and from nothing else. A non-zero *volume*
    column is therefore the venue; a non-zero *price* column is a parity break,
    because OHLC only moves on a trade the final frame would have carried.

    Derived from the spec's own input bindings rather than from a list kept beside
    it, and the failure mode that prevents is concrete: the first volume-weighted
    price feature anybody adds reads *both*, and a hand-maintained list would file
    it under price and rate a real break as venue behaviour.
    """
    sources = column_sources(spec)
    return tuple(c for c in spec.columns if "volume" in sources[c])


# ── where bars come from ─────────────────────────────────────────────────────


@dataclass
class FeedHealth:
    """What the feed did, split by *which* fault it was.

    ``gap_before`` on a bar and a hole in ``seq`` are two different faults with one
    symptom, and ADR-0028 is explicit that a consumer must not conflate them:

    * :attr:`feed_gaps` — the **venue** never printed a bar. Nothing was lost between
      the core and here; the market data itself has a hole, and every windowed
      feature spanning it is computed over a wider stretch of calendar time than its
      window says. Reported, never repaired: an interpolated bar is a close nothing
      traded at.
    * :attr:`ring_dropped` — the **ring** dropped a bar because this reader fell
      behind. The bar existed and neither the strategy nor the diff ever saw it, so
      both sides of the comparison agree perfectly about a window that is one
      observation short. This counter is the only thing in a shadow run that can see
      that, which is why it fails the run.
    * :attr:`first_bars` — a bar that opened a series, so continuity before it is
      unknown rather than good.
    """

    bars: int = 0
    ring_dropped: int = 0
    feed_gaps: int = 0
    first_bars: int = 0

    def describe(self) -> str:
        return (
            f"{self.bars} bar(s), {self.ring_dropped} ring drop(s), "
            f"{self.feed_gaps} venue gap(s), {self.first_bars} first bar(s)"
        )


class BarSource(Protocol):
    """A source of closed bars, oldest first, plus what it knows about its own holes."""

    health: FeedHealth

    def poll(self) -> list[Bar]:
        """Every bar available right now. Empty is legitimate — see :attr:`exhausted`."""

    @property
    def exhausted(self) -> bool:
        """Whether an empty :meth:`poll` means *finished* rather than *quiet*."""

    def describe(self) -> str:
        """Provenance, for the report. A reader must never have to guess whether a
        number came from a live venue socket or from a file."""


class HistoryBarSource:
    """A cached ``candleSnapshot`` history, delivered as closed bars.

    Real prints from the real venue, and **not a live session** — :meth:`describe`
    says so in the string that reaches the report, because "shadow-traded 4,975 bars"
    reads identically for a recording and for a socket, and only one of those is
    rung 3.

    Bars are handed over in chunks rather than one list, so the shadow loop that
    drives a recording is byte-for-byte the loop that drives a ring: a driver that
    only ever sees the whole history at once would silently depend on that.
    """

    def __init__(self, candles: Candles, *, symbol_id: int, chunk: int = 16) -> None:
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        self.candles = candles
        self.symbol_id = int(symbol_id)
        self.chunk = int(chunk)
        self.health = FeedHealth()
        self._at = 0
        step = INTERVAL_MS[candles.interval] * 1_000_000
        # The venue's own holes, counted the way the bar ring flags them, so a
        # recording and a live feed report the same fault under the same name.
        self._gap_at: set[int] = set()
        if len(candles) > 1:
            self._gap_at = set(np.flatnonzero(np.diff(candles.ts_event) != step) + 1)

    @property
    def exhausted(self) -> bool:
        return self._at >= len(self.candles)

    def describe(self) -> str:
        return (
            f"recorded candle history: {self.candles.coin} {self.candles.interval}, "
            f"{len(self.candles)} closed bars — NOT a live session"
        )

    def poll(self) -> list[Bar]:
        c = self.candles
        stop = min(self._at + self.chunk, len(c))
        out = []
        for i in range(self._at, stop):
            if i == 0:
                self.health.first_bars += 1
            elif i in self._gap_at:
                self.health.feed_gaps += 1
            self.health.bars += 1
            out.append(
                Bar(
                    symbol_id=self.symbol_id,
                    ts_event=int(c.ts_event[i]),
                    open=int(c.open[i]),
                    high=int(c.high[i]),
                    low=int(c.low[i]),
                    close=int(c.close[i]),
                    volume=int(c.volume[i]),
                )
            )
        self._at = stop
        return out

    def close(self) -> None:  # pragma: no cover - symmetry with RingBarSource
        pass

    def __enter__(self) -> "HistoryBarSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RingBarSource:
    """The Rust core's bar ring (ADR-0028), delivered as closed bars.

    Wraps :class:`~axon.marketdata.MdBarRingConsumer` and does nothing to its
    numbers: the consumer already separates ``seq`` holes from ``gap_before`` flags,
    and re-deriving either here would be a second opinion about which fault occurred.

    Two properties of the publisher a shadow run has to design around, both from
    ADR-0028: **a bar arrives one venue frame after it closed** (its ``ts_event`` is
    still its own close, so nothing downstream is skewed — but a deadline measured in
    wall time must allow for it), and **a session's last bar never arrives**, because
    nothing ever proved it closed.
    """

    def __init__(self, consumer, *, symbol_id: int | None = None, path: str | None = None) -> None:
        self._consumer = consumer
        self.symbol_id = symbol_id
        self.path = path if path is not None else getattr(consumer, "path", "<ring>")
        self.health = FeedHealth()

    @classmethod
    def attach(cls, md_ring_path: str, *, symbol_id: int | None = None) -> "RingBarSource":
        """Open the bar ring that sits beside ``md_ring_path``.

        The path is *derived*, never configured twice — see
        :func:`axon.marketdata.bar_ring_path`. An operator who could name the two
        rings independently could enable one and forget the other, and a bar-driven
        strategy would then start cleanly and simply never have an opinion.
        """
        from axon.marketdata import MdBarRingConsumer, bar_ring_path

        path = bar_ring_path(md_ring_path)
        return cls(MdBarRingConsumer(path), symbol_id=symbol_id, path=path)

    @property
    def exhausted(self) -> bool:
        # A live ring is never finished; an empty read means the market is quiet or
        # the feed is dead, and only the silence deadline can tell those apart.
        return False

    def describe(self) -> str:
        # Never the word "live". A ring carries bars, not provenance: this reader
        # cannot tell the Rust core's publisher from the harness writer below, and a
        # source string that guessed would put "live" in a report about a file.
        which = "every instrument" if self.symbol_id is None else f"symbol {self.symbol_id}"
        return f"market-data bar ring {self.path} ({which}); the ring does not say who wrote it"

    def poll(self) -> list[Bar]:
        bars = self._consumer.read_bars()
        if self.symbol_id is not None:
            bars = [b for b in bars if b.symbol_id == self.symbol_id]
        self.health.bars += len(bars)
        # Taken from the consumer rather than recomputed: it owns the `seq`
        # accounting and the flag counting, and two counters that can disagree about
        # whether a bar was lost are worse than one.
        self.health.ring_dropped = int(self._consumer.dropped)
        self.health.feed_gaps = int(self._consumer.gaps)
        self.health.first_bars = int(self._consumer.first_bars)
        return bars

    def close(self) -> None:
        self._consumer.close()

    def __enter__(self) -> "RingBarSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def publish_bars(
    candles: Candles,
    path: str,
    *,
    symbol_id: int,
    capacity: int = 1024,
    first_seq: int = 0,
) -> int:
    """Write a candle history onto a real bar ring. **A harness writer, not the publisher.**

    This exists so the whole consumer side — :class:`~axon.marketdata.MdBarRingConsumer`,
    the flags, :class:`RingBarSource`, and the shadow loop over it — can be exercised
    end to end without a venue. It is deliberately *not* evidence about
    ``axon_runtime::mdring``: that publisher **derives** closure from the venue
    starting the next interval, which is the one interesting decision it makes, and
    this function is handed bars a loader already proved closed. Anything a run
    through this writer shows is a fact about the Python half only.

    Flags mirror the publisher's meaning so a recording and a live feed report the
    same fault under the same name: ``first_bar`` on the first record, ``gap_before``
    wherever the venue's own close times skip an interval.
    """
    from axon.contracts import (
        MD_BAR_DTYPE,
        MD_BAR_FLAG_FIRST_BAR,
        MD_BAR_FLAG_GAP_BEFORE,
        new_md_bar,
    )
    from axon.signals import RingProducer

    interval_ms = INTERVAL_MS[candles.interval]
    step = interval_ms * 1_000_000
    written = 0
    with RingProducer(path, capacity=capacity, dtype=MD_BAR_DTYPE) as producer:
        for i in range(len(candles)):
            if i == 0:
                flags = MD_BAR_FLAG_FIRST_BAR
            elif int(candles.ts_event[i]) - int(candles.ts_event[i - 1]) != step:
                flags = MD_BAR_FLAG_GAP_BEFORE
            else:
                flags = 0
            record = new_md_bar(
                seq=first_seq + i,
                # The close, i.e. the venue's T + 1 ms. The loader already stamped it;
                # re-deriving it from open_time + interval here would be a second place
                # for the two halves to drift the one millisecond that makes
                # `align_by_event_time` intersect to nothing.
                ts_event=int(candles.ts_event[i]),
                open_time=int(candles.ts_event[i]) - step,
                symbol_id=symbol_id,
                interval_ms=interval_ms,
                open=int(candles.open[i]),
                high=int(candles.high[i]),
                low=int(candles.low[i]),
                close=int(candles.close[i]),
                volume=int(candles.volume[i]),
                flags=flags,
            )
            if not producer.try_push(record):
                # Refusing rather than dropping: a harness that silently truncated
                # its own input would make the consumer's drop counter — the one
                # thing that can see a lost bar — read zero on a run that lost bars.
                raise ShadowError(
                    f"bar ring {path} full at record {i} of {len(candles)} "
                    f"(capacity {capacity}); drain it or size it for the history"
                )
            written += 1
    return written


# ── what the run produced ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BarCoverage:
    """How many bars the interval promised, against how many arrived.

    The denominator one level below :class:`~axon.parity.features.Coverage`. That one
    asks *how many feature rows did the serving path owe over the bars it was shown*;
    this one asks *how many bars should it have been shown*. Both can be perfect while
    the other is catastrophic, and only the second is visible to a diff whose reference
    is a recompute over the same recording (ADR-0029 §1).

    :attr:`expected` is arithmetic on the recording's **own** close times —
    ``(last − first) / interval + 1`` — and deliberately not a wall-clock count of
    minutes this process was attached for. A wall-clock denominator would measure the
    reader's uptime and would call a healthy late-attaching session broken.

    A shortfall is **reported and never fatal**, for :attr:`FeedHealth.feed_gaps`'
    reason: a minute in which nothing traded is a minute the venue may legitimately
    not print, and a run that failed on that would teach an operator to ignore it.
    """

    delivered: int
    expected: int
    interval_ms: int
    first_ts: int | None = None
    last_ts: int | None = None

    @property
    def missing(self) -> int:
        return max(0, self.expected - self.delivered)

    @property
    def complete(self) -> bool:
        return self.missing == 0

    def describe(self) -> str:
        if self.expected == 0:
            return "0/0 bars — the run was shown nothing, so nothing was measured"
        pct = 100.0 * self.delivered / self.expected
        tail = "" if self.complete else f"; {self.missing} bar(s) of the cadence never arrived"
        return (
            f"{self.delivered}/{self.expected} bars over the span "
            f"({pct:.1f}% of a {self.interval_ms / 1000:.0f}s cadence){tail}"
        )


@dataclass(frozen=True)
class ArrivalLag:
    """How old a bar already is when it reaches the strategy — and so is its signal.

    :class:`~axon.live.StrategyRunner` stamps a signal with **the event's own
    ``ts_event``** and never with a wall clock, which is the right rule and has a
    consequence a bar strategy has to know: the stamp is the bar's *close*, and the
    bar cannot arrive before it. So a would-be order is born with this much age
    already spent, and on an m1 feed the ceiling it is spending against is 2 000 ms.

    A shadow run never goes through the signal reader, so nothing here is admitted or
    refused. That is exactly why it has to be measured: **a harness that cannot refuse
    a signal will happily produce would-be orders the real intent path would drop**,
    and a transcript of them would be a transcript of orders that never could have
    been placed. Agent P1 hit the live form of this — `accepted: 1, expired: 1`, the
    second target never planned at all.

    The arithmetic the reader actually does, and the reason this is not just
    ``arrival_lag`` against the ceiling: it compares against ``CoreHandler::last_ts()``
    — the core's own event clock, which P1 measured running **1 564 ms behind wall
    time** on this venue. So the age a pass sees is ``arrival_lag − core_clock_lag``,
    which can be negative; the reader admits a negative age and counts it
    ``ahead_of_clock``. A bar feed's lag and the core clock's lag partially cancel,
    and whether the margin is comfortable is a measurement rather than a deduction.
    """

    lags_ms: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.lags_ms)

    @property
    def min_ms(self) -> float:
        return min(self.lags_ms) if self.lags_ms else 0.0

    @property
    def median_ms(self) -> float:
        return float(np.median(self.lags_ms)) if self.lags_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.lags_ms) if self.lags_ms else 0.0

    def ages_ms(self, *, core_clock_lag_ms: float = MEASURED_CORE_CLOCK_LAG_MS) -> tuple:
        """The age each signal would carry at the pass that read it."""
        return tuple(lag - core_clock_lag_ms for lag in self.lags_ms)

    def refused(
        self,
        *,
        ceiling_ms: float = DEFAULT_MAX_SIGNAL_AGE_MS,
        core_clock_lag_ms: float = MEASURED_CORE_CLOCK_LAG_MS,
    ) -> int:
        """How many would have been refused as ``Expired`` by a live signal reader."""
        ages = self.ages_ms(core_clock_lag_ms=core_clock_lag_ms)
        return sum(1 for age in ages if age > ceiling_ms)

    def describe(
        self,
        *,
        ceiling_ms: float = DEFAULT_MAX_SIGNAL_AGE_MS,
        core_clock_lag_ms: float = MEASURED_CORE_CLOCK_LAG_MS,
    ) -> str:
        if not self.lags_ms:
            return "no bar arrived, so no signal could be aged"
        ages = self.ages_ms(core_clock_lag_ms=core_clock_lag_ms)
        refused = self.refused(ceiling_ms=ceiling_ms, core_clock_lag_ms=core_clock_lag_ms)
        return (
            f"{self.n} bar(s) arrived {self.min_ms:.0f}/{self.median_ms:.0f}/"
            f"{self.max_ms:.0f} ms (min/median/max) after their own close. A signal is "
            f"stamped with that close, so against a core event clock "
            f"{core_clock_lag_ms:.0f} ms behind wall time its age at the pass would be "
            f"{min(ages):.0f}..{max(ages):.0f} ms and the shipped ceiling is "
            f"{ceiling_ms:.0f} ms — {refused} would be refused as Expired. Not enforced "
            "here: a shadow run never reaches the signal reader"
        )


@dataclass(frozen=True)
class ColumnDiff:
    """``max_abs_diff`` **per column**, and which side of the price/volume line it is on.

    The monitor reports one ``max_abs_diff`` for the whole matrix and the name of the
    worst column, which is the right summary for a gate and the wrong one for reading a
    live feed. ADR-0028's expectation is a statement about *columns*: a non-zero volume
    column is the venue's last-observed-frame behaviour, a non-zero price column is a
    parity break. A single scalar cannot be read that way — the two arrive as one
    number and the reassuring reading wins.

    ``inf`` in a column means a NaN disagreed with a number there. That is deliberately
    not folded into a max over the finite differences, where it would vanish entirely:
    a serving path that emitted NaN where the recompute has a value is the loudest
    possible break and must not be the quietest number in the table.
    """

    columns: tuple[str, ...]
    max_abs_diff: tuple[float, ...]
    volume_columns: frozenset[str]
    n_rows: int

    @property
    def price_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.columns if c not in self.volume_columns)

    def worst(self, columns: Sequence[str]) -> tuple[str | None, float]:
        pairs = [(c, d) for c, d in zip(self.columns, self.max_abs_diff) if c in set(columns)]
        if not pairs:
            return None, 0.0
        return max(pairs, key=lambda cd: cd[1])

    @property
    def worst_price(self) -> tuple[str | None, float]:
        return self.worst(self.price_columns)

    @property
    def worst_volume(self) -> tuple[str | None, float]:
        return self.worst(tuple(self.volume_columns))

    def describe(self) -> str:
        cells = ", ".join(
            f"{c}{'*' if c in self.volume_columns else ''}={d:.3e}"
            for c, d in zip(self.columns, self.max_abs_diff)
        )
        pc, pd = self.worst_price
        vc, vd = self.worst_volume
        return (
            f"{cells}  (* = volume-derived; {self.n_rows} row(s)); "
            f"worst price column {pc}={pd:.3e}, worst volume column {vc}={vd:.3e}"
        )


@dataclass(frozen=True)
class ShadowSignal:
    """One record the strategy put on the signal ring. **Not an order.**

    ``perp_bar`` emits a *target position*. Turning one into an order is the Rust
    planner's job and needs a book to price against, an instrument's tick and lot
    grid to quantize onto (ADR-0025), and knowledge of what is already resting — none
    of which a shadow run has. So this is the record the planner would have been
    handed, and nothing about what it would have become: no acknowledgement, no queue
    position, no fill.

    It is read back off the ring rather than out of the context, because crossing the
    ring is the part that can fail.
    """

    seq: int
    ts_event: int
    symbol_id: int
    target_qty: Decimal
    urgency: int
    ttl_ms: int
    model_version: int

    @classmethod
    def from_record(cls, rec) -> "ShadowSignal":
        return cls(
            seq=int(rec["seq"]),
            ts_event=int(rec["ts_event"]),
            symbol_id=int(rec["symbol_id"]),
            # Decimal, never float: a target size that went through float64 is a size
            # the venue rounds somewhere nobody chose.
            target_qty=Decimal(int(rec["target_qty"])) / Decimal(FIXED_POINT_SCALE),
            urgency=int(rec["urgency"]),
            ttl_ms=int(rec["ttl_ms"]),
            model_version=int(rec["model_version"]),
        )


@dataclass(frozen=True)
class Turnover:
    """What the observed target path would have cost in fees, and nothing else.

    A **count**, not a P&L. ADR-0022 refuses to net costs off the measured edge
    because that needs a turnover *model*, and a turnover model living beside the
    evaluator would be a second implementation of the strategy's own hysteresis. This
    is the other thing: the turnover a shadow run actually emitted, priced at the
    published fee schedule as a hurdle to be quoted next to the edge.

    ``sides`` counts position changes in units of ``max_position``, so flipping long
    to short is two and opening from flat is one. Fees are *per side*, which is why
    the count is in sides rather than round trips.
    """

    signals: int
    sides: Decimal
    max_position: Decimal

    @property
    def maker_bps(self) -> float:
        """Maker fees the emitted path would pay, in bps of one ``max_position`` notional."""
        return float(self.sides) * MAKER_FEE_BPS

    @property
    def taker_bps(self) -> float:
        return float(self.sides) * TAKER_FEE_BPS

    def describe(self) -> str:
        return (
            f"{self.signals} would-be signal(s), {self.sides} side(s) of "
            f"{self.max_position} = {self.maker_bps:.1f} bps maker "
            f"({self.taker_bps:.1f} bps taker) of one position's notional. "
            "A count of what was emitted, not a P&L: nothing was filled."
        )


@dataclass(frozen=True)
class ShadowReport:
    """One shadow run, and precisely what it does and does not license.

    :attr:`passed` is *the diff agreed, over every row it was owed, and no bar was
    lost on the way in*. It is deliberately **not** a statement about the strategy:
    a green shadow run says the serving path computes what the offline recompute
    computes, which is the same thing all three gates of ADR-0016 say. Whether the
    numbers are worth trading is answered by the numbers, and for ``perp_bar``
    ADR-0022 answers it: no.
    """

    source: str
    #: Whether the **operator** attested that a real Hyperliquid session fed this run.
    #: Defaults to ``False`` and is never inferred, because nothing in this process can
    #: infer it: an ``MdBar`` record carries no source marker, so a bar written by the
    #: Rust core and one written by :func:`publish_bars` are byte-identical here. A
    #: report that guessed would put the word *live* on a run over a file, which is the
    #: one claim this whole module exists not to make.
    venue_attested: bool
    spec_ref: str
    bars: int
    decisions: int
    feed: FeedHealth
    monitor: MonitorReport
    rows_compared: int
    rows_owed: int
    windows: int
    signals: tuple[ShadowSignal, ...]
    turnover: Turnover
    drift_measured: bool
    #: The denominator below the row denominator — see :class:`BarCoverage`. A field
    #: with a default so every existing construction of this report keeps working,
    #: and it is computed on every run rather than opted into: a coverage number an
    #: operator has to ask for is a coverage number nobody reads.
    bar_coverage: BarCoverage = field(
        default_factory=lambda: BarCoverage(delivered=0, expected=0, interval_ms=0)
    )
    #: ``max_abs_diff`` per column, split by whether volume can move it.
    columns: ColumnDiff | None = None
    #: How old each bar — and therefore each would-be signal — already was on arrival.
    arrival_lag: ArrivalLag | None = None
    #: Windows that fell entirely inside the spec's warmup, so they owed nothing and
    #: produced nothing. Counted rather than silently dropped: a run whose windows are
    #: narrower than its warmup spends its opening minutes comparing nothing, and that
    #: is a fact about the *configuration* an operator should see, not an absence.
    warmup_windows: int = 0

    @property
    def passed(self) -> bool:
        return self.monitor.passed and self.feed.ring_dropped == 0

    @property
    def coverage(self) -> str:
        """The number that means something: matched rows over rows owed."""
        return f"{self.rows_compared}/{self.rows_owed}"

    def summary(self) -> str:
        lines = [
            f"shadow {'OK' if self.passed else 'FAIL'} — {self.spec_ref}",
            f"  source: {self.source}",
            f"  bars: {self.feed.describe()}; {self.decisions} decision row(s)",
            f"  feature parity: max_abs_diff={self.monitor.max_abs_diff:.3e} over "
            f"{self.coverage} owed rows in {self.windows} window(s), worst={self.monitor.worst}",
            f"  bar coverage: {self.bar_coverage.describe()}",
            f"  turnover: {self.turnover.describe()}",
        ]
        if self.columns is not None:
            lines.append(f"  per column: {self.columns.describe()}")
        if self.arrival_lag is not None and self.arrival_lag.n:
            if self.venue_attested:
                lines.append(f"  signal age: {self.arrival_lag.describe()}")
            else:
                # Over a recording this subtraction is wall-clock-now minus a close
                # from last week, and printing it would put "900 would be refused as
                # Expired" on a flawless offline run — a false alarm of exactly the
                # kind the rest of this module exists to remove. Liveness is not
                # inferable here (§6), so neither is the meaning of this number, and
                # the honest report is the reason rather than the value.
                lines.append(
                    "  signal age: not measured — a bar's arrival lag is a *feed* "
                    "measurement and this run is not attested against a live venue, "
                    "so the subtraction below would be the age of a file"
                )
        if self.warmup_windows:
            lines.append(
                f"  {self.warmup_windows} window(s) fell entirely inside the spec's "
                "warmup and were not compared — they owed nothing and produced nothing, "
                "which is not the same statement as agreeing"
            )
        if not self.bar_coverage.complete:
            # Printed as a paragraph rather than left in the ratio, because the ratio
            # is the one line a reader skips: every *other* number on a run missing
            # half its bars is identical to a healthy run's.
            lines.append(
                f"  {self.bar_coverage.missing} bar(s) the interval promised over this "
                "span never reached the strategy. Both sides of the diff recompute over "
                "the bars that did arrive, so the parity number above is exactly as "
                "green as it would be on a complete feed — it is a statement about "
                "arithmetic, not about the feed"
            )
        if not self.drift_measured:
            # "Not measured" and "stable" are opposite statements, and a report that
            # prints neither invites a reader to assume the second one.
            lines.append(
                "  drift: not measured — no reference sample was supplied. ADR-0030 "
                "measured why this is not the default: with parity green on every "
                "window, PSI trips the conventional 0.25 band on 18 of 20 windows."
            )
        if self.feed.ring_dropped:
            lines.append(
                f"  {self.feed.ring_dropped} bar(s) were dropped by the ring and reached "
                "NEITHER side of the diff, so the parity number above is a statement "
                "about a feature history with a hole in it"
            )
        if self.feed.feed_gaps:
            lines.append(
                f"  {self.feed.feed_gaps} bar(s) the venue never printed. Not a parity "
                "failure — both sides recompute over the same hole — but every windowed "
                "feature spanning one covers more calendar time than its window says"
            )
        if not self.venue_attested:
            lines.append(
                "  NOT ATTESTED AGAINST THE VENUE: nothing here can prove a bar in this "
                "run crossed a Hyperliquid socket into the Rust core, so it is not "
                "claimed. The Python serving path, the ring and the continuous diff are "
                "exercised; the venue→core→ring half is not."
            )
        if not self.monitor.passed:
            lines.extend(f"  {r}" for r in self.monitor.reasons[:6])
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        if not self.passed:
            raise ParityError(self.summary())


# ── the diff, on its own, so a live order flow can carry one ─────────────────


@dataclass
class BarParityDiff:
    """Bars in, feature rows in, parity verdicts out — and **nothing else**.

    This is the piece extracted so that a *trading* session can be watched, and the
    extraction is the whole point rather than tidiness. Until now the diff lived inside
    :class:`ShadowTrader`, which also owns a signal ring, a
    :class:`~axon.live.StrategyRunner` and a decision transcript. None of that can be
    put beside a live order flow:

    * **An SPSC ring has one consumer.** On a trading session the *strategy* is the bar
      ring's consumer, and two readers do not share a ring, they steal from it —
      measured on 2026-07-26, when two drainers each saw about half the records and each
      reported the other's reads as drops. A monitor attached beside the strategy takes
      bars away from the thing placing orders.
    * **A second session's shutdown sweeps the account.** ``graceful_shutdown`` calls
      ``cancel_all``, which on Hyperliquid is account-wide, so a read-only watcher
      running beside a trader would cancel the trader's resting orders on its way out.

    Both are properties of *sessions*, and neither is a property of a diff. So the diff
    becomes a thing a session can own: one reader, dispatching each bar to the strategy
    and to this. That is the fan-out ADR-0030 wanted and could not have.

    **It is the same code the shadow path runs, and that is load-bearing.** The
    alternative — a cut-down comparison written beside the live runner — is how two
    answers to one question get into a tree, and they would disagree on exactly the
    window where it mattered. :class:`ShadowTrader` holds one of these; so does the live
    runner; there is one alignment rule, one window construction and one
    :class:`~axon.parity.monitor.ParityMonitor` configuration between them.

    What it deliberately does **not** own: the strategy, the ring, the clock that
    decides when a feed has gone silent (that is a session's), and any opinion about
    orders. It is handed a bar and the row the serving path computed for it, and it
    answers whether an offline recompute agrees.
    """

    spec: FeatureSpec = PERP_BAR_V1
    interval: str = "1h"
    window_bars: int = DEFAULT_WINDOW_BARS
    clock: Clock = time.time_ns

    def __post_init__(self) -> None:
        if self.window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {self.window_bars}")
        if self.interval not in INTERVAL_MS:
            raise ValueError(
                f"unsupported interval {self.interval!r}; known: {sorted(INTERVAL_MS)}"
            )
        interval_ns = INTERVAL_MS[self.interval] * 1_000_000
        self._monitor = ParityMonitor(
            MonitorConfig(
                columns=self.spec.columns,
                # No reference sample: drift is *not measured*, and the report says so
                # rather than reporting a stable session it never looked at.
                reference=None,
                # Each window's offline side is built as exactly the rows the serving
                # path owed, so say so. The default infers the owed span from the online
                # side's own first stamp, and a path blind through a window's *opening*
                # rows has the same first stamp as one that started on time — the two
                # are indistinguishable from the data, so an inferring scope excuses the
                # blindness and removes those rows from the denominator as well.
                scope="declared",
                silence_after_ns=int(SILENCE_AFTER_BARS * interval_ns),
            ),
            clock=self.clock,
        )
        self._bars: list[tuple[int, int, int, int, int, int]] = []
        self._online_ts: list[int] = []
        self._online_rows: list[np.ndarray] = []
        self._pending_from = 0
        self._rows_compared = 0
        self._rows_owed = 0
        self._windows = 0
        # Per-column magnitudes, accumulated across windows. The monitor keeps one
        # scalar and the worst column's *name*, which cannot answer ADR-0028's
        # question — whether the disagreement is in a column volume can move.
        self._col_max = np.zeros(len(self.spec.columns), dtype=np.float64)
        self._col_rows = 0
        self._volume_columns = frozenset(volume_derived_columns(self.spec))
        self._warmup_windows = 0

    # ── what a session hands over ────────────────────────────────────────────

    @property
    def monitor(self) -> ParityMonitor:
        return self._monitor

    @property
    def bars(self) -> int:
        return len(self._bars)

    @property
    def decisions(self) -> int:
        return len(self._online_rows)

    @property
    def windows(self) -> int:
        return self._windows

    @property
    def warmup_windows(self) -> int:
        return self._warmup_windows

    @property
    def rows_compared(self) -> int:
        return self._rows_compared

    @property
    def rows_owed(self) -> int:
        return self._rows_owed

    def observe(self, bar: Bar, row: np.ndarray | None) -> Verdict | None:
        """Record one bar of **this instrument** and the row the serving path made of it.

        The caller filters by symbol: a mixed series would be nine columns of nonsense,
        and a diff cannot tell one instrument's bar from another's without being told
        which one it is watching.

        ``row`` is ``None`` while the spec is still warming, which is not a fault and is
        why the two lists are kept separately rather than zipped — the offline side has a
        row for every bar and the online side does not.

        Returns a verdict on the flush that closes a window, and ``None`` otherwise.
        """
        self._check_orderly(bar)
        self._bars.append((bar.ts_event, bar.open, bar.high, bar.low, bar.close, bar.volume))
        if row is not None:
            self._online_ts.append(bar.ts_event)
            self._online_rows.append(np.asarray(row, dtype=np.float64))
        if len(self._bars) - self._pending_from >= self.window_bars:
            return self.flush_window()
        return None

    def _check_orderly(self, bar: Bar) -> None:
        if not self._bars:
            return
        last = self._bars[-1][0]
        if bar.ts_event == last:
            raise ShadowError(
                f"two bars at close time {bar.ts_event}: the serving buffer would hold "
                "the bar twice and every window from here is one observation wide in "
                "the wrong place, on both sides of the diff"
            )
        if bar.ts_event < last:
            raise ShadowError(
                f"bar at {bar.ts_event} arrived after {last}: a parity diff aligns on "
                "event time and a series that goes backwards cannot be aligned with "
                "anything"
            )

    def flush_window(self) -> Verdict | None:
        """Diff every bar since the last flush against the offline recompute.

        The recompute is over the **whole recording**, not over the window: every
        transform in the spec has a finite lookback, so a recompute that starts at the
        window boundary would be NaN through its own warmup and would disagree with a
        serving path whose buffer was already warm. That is the same property the
        bounded serving buffer relies on, applied to the reference instead.
        """
        if len(self._bars) <= self._pending_from:
            return None
        columns = np.asarray(self._bars, dtype=np.int64)
        ts_all = columns[:, 0]
        inputs = bar_inputs(*(columns[:, j] for j in range(1, 6)))
        offline = self.spec.compute(inputs)

        first_ts = int(ts_all[self._pending_from])
        in_window = np.zeros(ts_all.size, dtype=bool)
        in_window[self._pending_from :] = True
        owed = in_window & finite_rows(offline)

        online_ts = np.asarray(self._online_ts, dtype=np.int64)
        online_all = (
            np.vstack(self._online_rows)
            if self._online_rows
            else np.empty((0, len(self.spec.columns)), dtype=np.float64)
        )
        take = online_ts >= first_ts
        online = online_all[take]

        if not owed.any() and online.shape[0] == 0:
            # A window entirely inside the spec's warmup: nothing was owed and nothing
            # was produced. Offering it to the monitor grades it SILENT — "a window
            # that compared nothing is not a window that agreed" — which is true of a
            # dead feed and false of a warmup, and SILENT sits above WARN, so a healthy
            # run's first minutes fail it. Only visible live: `perp_bar` warms up in 25
            # bars and the shipped 64-bar window swallows that, so every offline run
            # ever done here had finite rows in its first window. **Both sides empty is
            # the whole condition** — an online side with rows the reference has none of
            # is `online_unmatched`, a real fault, and must still reach the monitor.
            self._pending_from = len(self._bars)
            self._warmup_windows += 1
            return None

        self._accumulate_columns(online, offline[owed], online_ts[take], ts_all[owed])
        verdict = self._monitor.observe(
            Window(
                online=online,
                offline=offline[owed],
                online_ts=online_ts[take],
                offline_ts=ts_all[owed],
            )
        )
        self._pending_from = len(self._bars)
        self._windows += 1
        if verdict.coverage is not None:
            self._rows_compared += verdict.coverage.n_matched
            # ``n_in_scope`` under a declared scope is every row handed over — nothing
            # can be out of scope when the caller has said what the scope is — so this
            # is the whole denominator and not a share of it. It is also the field that
            # stays correct if a window is ever built ``"observed"`` again, which
            # ``n_offline`` would not be.
            self._rows_owed += verdict.coverage.n_in_scope
        return verdict

    def _accumulate_columns(self, online, offline, online_ts, offline_ts) -> None:
        """Fold one window's per-column magnitudes into the run's.

        Aligned with the same :func:`~axon.parity.features.align_by_event_time` the
        monitor uses, under the same ``"declared"`` scope, rather than by zipping the
        two matrices: a private matching rule here would be a second implementation of
        the question ADR-0029 §2 spent an increment collapsing to one, and it would
        drift on exactly the window where it mattered. Calling the shared aligner twice
        costs an intersection; it cannot disagree with itself.
        """
        if online.size == 0 or offline.size == 0:
            return
        idx = align_by_event_time(online_ts, offline_ts, on_gap="report", scope="declared")
        if idx.online.size == 0:
            return
        diff = np.abs(
            np.asarray(online, dtype=np.float64)[idx.online]
            - np.asarray(offline, dtype=np.float64)[idx.offline]
        )
        # NaN against a number is the loudest possible disagreement, and `np.max`
        # over it would propagate while `np.nanmax` would erase it. `inf` keeps it
        # the largest number in the column, which is what it is.
        diff = np.where(np.isnan(diff), np.inf, diff)
        self._col_max = np.maximum(self._col_max, diff.max(axis=0))
        self._col_rows += int(idx.online.size)

    def column_diff(self) -> ColumnDiff:
        return ColumnDiff(
            columns=tuple(self.spec.columns),
            max_abs_diff=tuple(float(v) for v in self._col_max),
            volume_columns=self._volume_columns,
            n_rows=self._col_rows,
        )

    def bar_coverage(self) -> BarCoverage:
        """What the bar cadence promised over the span, against what arrived.

        Event time only: the span is the recording's own first and last close, so a
        session that attached late is measured over what it saw rather than over how
        long it happened to be running.
        """
        step_ms = INTERVAL_MS[self.interval]
        if not self._bars:
            return BarCoverage(delivered=0, expected=0, interval_ms=step_ms)
        first, last = self._bars[0][0], self._bars[-1][0]
        step_ns = step_ms * 1_000_000
        return BarCoverage(
            delivered=len(self._bars),
            expected=(last - first) // step_ns + 1,
            interval_ms=step_ms,
            first_ts=first,
            last_ts=last,
        )

    def recorded_bars(self) -> np.ndarray:
        """Exactly the bars this diff was shown, as an ``(n, 6)`` int64 array.

        Columns are ``ts_event, open, high, low, close, volume`` in the wire's
        fixed-point integers — the same units the ring carried, undivided, so a
        reconciliation against the venue compares what crossed rather than a float
        this module rounded on the way past.
        """
        return np.asarray(self._bars, dtype=np.int64).reshape(-1, 6)


# ── the loop ─────────────────────────────────────────────────────────────────


@dataclass
class ShadowTrader:
    """Drives one strategy over a bar source, diffing it against itself as it goes.

    Construct, then either hand it a :class:`BarSource` via :meth:`run` or pump bars
    into :meth:`on_bar` yourself. Either way the strategy runs under the real
    :class:`~axon.live.StrategyRunner`, so what reaches the ring is what a live
    session would put there.

    The run starts **cold**. That is not tidiness: the offline reference is a
    recompute over the bars *this run* was shown, so a strategy arriving with a warm
    buffer would produce feature rows the reference cannot have, and every one of them
    would be an unmatched online row rather than a checked one. A warm restart is a
    real thing to want and it needs a reference this module does not have.
    """

    strategy: PerpBar
    symbol_id: int
    ring_path: str
    spec: FeatureSpec = PERP_BAR_V1
    interval: str = "1h"
    window_bars: int = DEFAULT_WINDOW_BARS
    model_version: int = 1
    capacity: int = 1024
    clock: Clock = time.time_ns
    source_name: str = "unattached"
    #: The operator's claim that a real venue session fed this run, never a guess —
    #: see :attr:`ShadowReport.venue_attested`.
    venue_attested: bool = False

    def __post_init__(self) -> None:
        from axon.live import StrategyRunner
        from axon.signals import RingConsumer

        if self.window_bars < 1:
            raise ValueError(f"window_bars must be at least 1, got {self.window_bars}")
        if self.interval not in INTERVAL_MS:
            raise ValueError(
                f"unsupported interval {self.interval!r}; known: {sorted(INTERVAL_MS)}"
            )
        self.strategy.on_reset()
        self._runner = StrategyRunner(
            self.strategy,
            ring_path=self.ring_path,
            model_version=self.model_version,
            capacity=self.capacity,
        )
        # A consumer on the same ring, because a shadow run has no Rust core to drain
        # it. Reading the record back off the ring is the point: the context holding a
        # record and the ring carrying it are different claims, and only the second one
        # is what a live session depends on.
        self._consumer = RingConsumer(self.ring_path)
        interval_ns = INTERVAL_MS[self.interval] * 1_000_000
        # **The diff is a component, not this class's business.** It was inlined here
        # until 2026-07-27 and that is exactly why no live parity monitor could watch a
        # *trading* session: the comparison was welded to a signal ring, a
        # `StrategyRunner` and a second bar-ring consumer, none of which can sit beside a
        # live order flow. The alternative — a cut-down copy beside the live runner — is
        # how two answers to one question get into a tree, and they would disagree on
        # exactly the window where it mattered. So there is one implementation and both
        # callers hold one. See :class:`BarParityDiff`.
        self._diff = BarParityDiff(
            spec=self.spec,
            interval=self.interval,
            window_bars=self.window_bars,
            clock=self.clock,
        )
        self._signals: list[ShadowSignal] = []
        self._last_heartbeat_ns = 0
        self._silence_after_ns = int(SILENCE_AFTER_BARS * interval_ns)
        # **Wall clock, and named as the exception it is.** Everything else in this
        # module orders on ``ts_event``; this one measures an *absence*, and an
        # event-time deadline for "nothing has arrived" can never expire because the
        # clock that would advance it is the thing that stopped (ADR-0029 §4). It
        # starts at construction so a source that never delivers a first bar still
        # reaches the deadline. The same clock the monitor uses, because the two
        # numbers are compared.
        self._last_bar_ns = int(self.clock())
        self._arrival_lags_ms: list[float] = []
        self._started = False
        self._stopped = False

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def monitor(self) -> ParityMonitor:
        return self._diff.monitor

    @property
    def diff(self) -> BarParityDiff:
        """The comparison this run drives.

        Public because the *other* caller owns its own bar reader: a live trading session
        dispatches each bar to its strategy and to one of these, which is the fan-out
        that lets a session that is placing orders be watched at all.
        """
        return self._diff

    @property
    def signals(self) -> tuple[ShadowSignal, ...]:
        return tuple(self._signals)

    @property
    def bars_seen(self) -> int:
        """Bars for **this** instrument that reached the strategy."""
        return self._diff.bars

    # ── driving ──────────────────────────────────────────────────────────────

    def start(self, ts_event: int) -> None:
        self._runner.start(int(ts_event))
        self._started = True

    def on_bar(self, bar: Bar) -> None:
        """Dispatch one bar, record it, and keep the served row beside it.

        Every bar goes to the runner — including another instrument's, so the
        runner's event clock advances exactly as a live session's would — but only
        this instrument's is recorded, because the offline recompute is over one
        series and a mixed one would be nine columns of nonsense.
        """
        if not self._started:
            self.start(bar.ts_event)
        if self._stopped:
            raise ShadowError("shadow run is stopped")
        mine = bar.symbol_id == self.symbol_id
        # Armed by **any** bar, not only this instrument's, and the distinction is not
        # cosmetic. The deadline's question is *"has the feed stopped"*, and on a venue
        # that publishes no candle frame at all for a minute in which nothing traded —
        # Hyperliquid testnet omitted 6 of 64 BTC minutes in the first live run — one
        # instrument can be legitimately silent for several intervals while the session
        # is perfectly healthy. Another instrument's bar is proof of that; the
        # instrument's own silence is not evidence of anything.
        #
        # What this gives up is the ADR-0020 §7 shape: one instrument's subscription
        # coming back and another's not. That fault is still *visible* — it collapses
        # :class:`BarCoverage` for the stalled symbol — it just is not an alarm, which
        # is the right trade, because the alarm it replaces fires on a quiet market and
        # an alarm that fires on a healthy feed is the one that gets a check deleted.
        # A single-symbol run has no other instrument to appeal to and degenerates to
        # the old behaviour; that is an argument for shadowing two.
        self._last_bar_ns = int(self.clock())
        if mine:
            self._check_orderly(bar)
            # Wall minus event, the one place the two clocks are deliberately
            # subtracted. This is how old the signal this bar produces is *born*,
            # because the runner stamps it with the bar's own close and never with a
            # wall clock. Recorded rather than derived later: nothing downstream can
            # reconstruct when a record arrived.
            self._arrival_lags_ms.append((self._last_bar_ns - bar.ts_event) / 1e6)
        self._runner.handle(bar)
        if mine:
            # One dispatch, two consumers — the strategy above and the diff here. That
            # is the same shape a live trading session now uses, and it is why the diff
            # is a component: the alternative is a second reader on an SPSC ring, and a
            # second reader does not share a ring, it steals from it.
            self._diff.observe(bar, self.strategy.feature_row())
        self._drain_ring()

    def _check_orderly(self, bar: Bar) -> None:
        """The ordering guard, run **before** the diff so the message can name the symbol.

        :class:`BarParityDiff` refuses a backwards series too — it has to, because a diff
        nobody drove through this class is still a diff — but it is handed a bar and never
        told which instrument it is watching, so its message cannot say. Two checks, one
        rule, and the redundant one is the one that produces the better error.
        """
        bars = self._diff.recorded_bars()
        if bars.shape[0] == 0:
            return
        last = int(bars[-1, 0])
        if bar.ts_event == last:
            raise ShadowError(
                f"two bars for symbol {self.symbol_id} at close time {bar.ts_event}: the "
                "serving buffer would hold the bar twice and every window from here is "
                "one observation wide in the wrong place, on both sides of the diff"
            )
        if bar.ts_event < last:
            raise ShadowError(
                f"bar at {bar.ts_event} arrived after {last} for symbol {self.symbol_id}: "
                "a shadow diff aligns on event time and a series that goes backwards "
                "cannot be aligned with anything"
            )

    def _drain_ring(self) -> None:
        batch = self._consumer.read_batch()
        for rec in batch:
            self._signals.append(ShadowSignal.from_record(rec))

    def flush_window(self) -> Verdict | None:
        """Close the current window. See :meth:`BarParityDiff.flush_window`."""
        return self._diff.flush_window()

    def arrival_lag(self) -> ArrivalLag:
        return ArrivalLag(lags_ms=tuple(self._arrival_lags_ms))

    def column_diff(self) -> ColumnDiff:
        return self._diff.column_diff()

    def bar_coverage(self) -> BarCoverage:
        return self._diff.bar_coverage()

    def recorded_bars(self) -> np.ndarray:
        return self._diff.recorded_bars()

    def heartbeat(self, *, force: bool = False) -> Verdict | None:
        """Ask the monitor for a silence verdict — but only once a **bar** is overdue.

        Rate-limited here rather than in the monitor because the limit is a property
        of *this* loop's poll rate: a driver that spins at a millisecond would
        otherwise bury the verdict that matters under ten thousand that do not.

        The bar deadline is the second, larger gate, and it is not tidiness. The
        monitor's own clock advances when **a window compares something**, and a
        window is :attr:`window_bars` bars wide, so between two healthy flushes the
        monitor is legitimately silent for `window_bars` intervals — far past a
        deadline measured in 2.5 of them. Asked every second through that stretch it
        answers ``SILENT``, which sits *above* ``WARN``, so
        :attr:`~axon.parity.monitor.MonitorReport.passed` goes false and the run
        reports FAIL with every cell it compared exactly right. Measured on the first
        live m1 session ever run through this module: **296 SILENT verdicts in 160
        seconds of a perfectly healthy feed**, before a single window had closed.
        Nothing offline ever saw it, because :class:`HistoryBarSource` is never idle.

        So the question is asked only when the thing that would actually be missing
        is missing: a bar. What that costs is one deadline of latency and it is worth
        naming: the monitor's progress clock starts at the *first* heartbeat, so a
        dead feed reads ``SILENT`` 2.5 intervals after it stopped and promotes to
        ``ALARM`` 2.5 intervals after that. ``SILENT`` already outranks ``WARN`` and
        already fails the run, so nothing is missed in between — what is deferred is
        only the word, and the alternative was a run that shouted it every second of
        a healthy session.
        """
        now = int(self.clock())
        if not force:
            if now - self._last_heartbeat_ns < HEARTBEAT_EVERY_S * 1e9:
                return None
            if now - self._last_bar_ns < self._silence_after_ns:
                return None
        self._last_heartbeat_ns = now
        return self._diff.monitor.heartbeat()

    def stop(self, ts_event: int | None = None) -> None:
        if self._stopped or not self._started:
            self._stopped = True
            return
        self.flush_window()
        self._runner.stop(int(ts_event) if ts_event is not None else self._runner.last_event_ns)
        self._drain_ring()
        self._stopped = True

    def close(self) -> None:
        self._consumer.close()
        self._runner.close()

    def __enter__(self) -> "ShadowTrader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(
        self,
        source: BarSource,
        *,
        max_bars: int | None = None,
        idle_timeout_s: float | None = None,
        poll_s: float = 0.05,
    ) -> ShadowReport:
        """Drive the whole source and return what the run proved.

        One trader's case of :func:`drive`, and delegated rather than duplicated: a
        private copy of the poll loop here would be the second implementation of the
        silence deadline and the idle bound, and the two would drift on whichever one
        a test did not cover.
        """
        return drive(
            [self],
            source,
            max_bars=max_bars,
            idle_timeout_s=idle_timeout_s,
            poll_s=poll_s,
        )[0]

    def finish(self, source: BarSource | None = None) -> ShadowReport:
        """Flush the tail, stop the runner, and produce the report."""
        self.stop()
        return self.report(source)

    def report(self, source: BarSource | None = None) -> ShadowReport:
        health = source.health if source is not None else FeedHealth(bars=self._diff.bars)
        return ShadowReport(
            source=self.source_name if source is None else source.describe(),
            venue_attested=self.venue_attested,
            spec_ref=self.spec.ref,
            bars=self._diff.bars,
            decisions=self._diff.decisions,
            feed=health,
            monitor=self._diff.monitor.report(),
            rows_compared=self._diff.rows_compared,
            rows_owed=self._diff.rows_owed,
            windows=self._diff.windows,
            signals=tuple(self._signals),
            turnover=self._turnover(),
            drift_measured=self._diff.monitor.config.drift_enabled,
            bar_coverage=self.bar_coverage(),
            columns=self.column_diff(),
            arrival_lag=self.arrival_lag(),
            warmup_windows=self._diff.warmup_windows,
        )

    def _turnover(self) -> Turnover:
        held = Decimal(0)
        sides = Decimal(0)
        unit = self.strategy.params.max_position
        for signal in self._signals:
            sides += abs(signal.target_qty - held) / unit
            held = signal.target_qty
        return Turnover(signals=len(self._signals), sides=sides, max_position=unit)


def drive(
    traders: Sequence[ShadowTrader],
    source: BarSource,
    *,
    max_bars: int | None = None,
    idle_timeout_s: float | None = None,
    poll_s: float = 0.05,
) -> list[ShadowReport]:
    """Drive one **or several** strategies from a single bar source.

    Several traders and one source, never the reverse, and that is forced rather than
    stylistic: a ring's read cursor lives in the ring's own header, so two consumers
    attached to one bar ring each pop a share of the records and **neither can tell
    that from a quiet feed**. A session shadowing two instruments therefore has to fan
    out from one reader. The shape has a second benefit that ADR-0029 §5 asks for
    anyway: every trader sees every bar, so each one's event clock advances on the
    other instrument's prints exactly as a live session's would, while recording only
    its own.

    ``idle_timeout_s`` is the one bound a live source needs, and it is wall clock by
    necessity: a feed that has stopped and a market that has gone quiet are the same
    silence, and an event-time deadline for an absence can never expire.

    ``max_bars`` stops when the **fastest** series reaches it. A quiet instrument that
    prints half as often would otherwise hold the run open indefinitely, and the point
    of the bound is that it is a bound.
    """
    if not traders:
        raise ShadowError("drive() needs at least one trader; nothing would be diffed")
    for trader in traders:
        trader.source_name = source.describe()
    last_data = time.monotonic()
    while True:
        bars = source.poll()
        if not bars:
            if source.exhausted:
                break
            if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
                break
            for trader in traders:
                trader.heartbeat()
            time.sleep(poll_s)
            continue
        last_data = time.monotonic()
        for bar in bars:
            for trader in traders:
                trader.on_bar(bar)
            if max_bars is not None and max(t.bars_seen for t in traders) >= max_bars:
                return [t.finish(source) for t in traders]
    return [t.finish(source) for t in traders]


# ── the other diff: the published bar against the venue's own ────────────────

#: The OHLCV fields of a bar, in the order :func:`~axon.features.bar_inputs` takes them.
BAR_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class VenueBarDiff:
    """The **published** bar against the venue's own ``candleSnapshot``, field by field.

    A different question from the shadow diff, and the one ADR-0028 left open. The
    shadow diff compares a serving path against a recompute over *the same bars*, so
    it is blind to whether those bars were right; both sides would agree perfectly
    about a wrong bar. This compares the bar itself against the venue's answer for the
    same minute.

    Read it by **column**, never by magnitude. Hyperliquid sends no closing frame — it
    republishes the bar it is filling and then starts the next one — so a published bar
    is the last frame *observed* before the close, and a trade printing after it is
    missing from ``v``:

    * a non-zero **volume** field is the venue, and is expected;
    * a non-zero **price** field (o/h/l/c) is a **parity break**, because OHLC only
      moves on a trade the final frame would have carried, and a missing trade that
      moved the price would have moved volume too.

    Nothing here rates a break. :attr:`price_break` is a boolean and the fields that
    moved are named, because "small" is not a category ADR-0028 offers.

    It is also a **second view of the market**, which is exactly what ADR-0012 §1
    refuses as a *runtime input*. This is a measurement taken beside a run, never a
    source a session reads — the moment a reconciliation like this feeds back into what
    the strategy is shown, the two views have to be reconciled continuously and the
    ring stops being the single source of truth.
    """

    coin: str
    interval: str
    n_ring: int
    n_venue: int
    n_matched: int
    field_mismatched_bars: Mapping[str, int]
    field_max_abs: Mapping[str, int]
    price_mismatched_bars: int
    volume_only_mismatched_bars: int
    columns: ColumnDiff | None
    unmatched_ring_ts: tuple[int, ...]
    unmatched_venue_ts: tuple[int, ...]

    @property
    def price_fields(self) -> tuple[str, ...]:
        return tuple(f for f in BAR_FIELDS if f != "volume")

    @property
    def price_break(self) -> bool:
        """Any matched bar whose ``o``, ``h``, ``l`` or ``c`` disagreed. Not a rating."""
        return any(self.field_mismatched_bars.get(f, 0) for f in self.price_fields)

    def describe(self) -> str:
        fields = ", ".join(
            f"{f}={self.field_mismatched_bars.get(f, 0)} bar(s)"
            f"/max {self.field_max_abs.get(f, 0)}"
            for f in BAR_FIELDS
        )
        lines = [
            f"venue reconciliation {self.coin} {self.interval}: "
            f"{self.n_matched} matched of {self.n_ring} ring bar(s) and "
            f"{self.n_venue} candleSnapshot bar(s)",
            f"  fields (mismatched bars / max abs diff, fixed-point): {fields}",
            f"  {self.volume_only_mismatched_bars} bar(s) differed in volume ALONE — "
            "the venue's last-observed-frame behaviour, expected (ADR-0028)",
        ]
        if self.price_break:
            broken = ", ".join(
                f for f in self.price_fields if self.field_mismatched_bars.get(f, 0)
            )
            lines.append(
                f"  PARITY BREAK: {self.price_mismatched_bars} bar(s) differ in a PRICE "
                f"field ({broken}). OHLC only moves on a trade the final frame would "
                "have carried, so this is not the venue's missing tail"
            )
        else:
            lines.append("  no price field disagreed on any matched bar")
        if self.unmatched_ring_ts:
            lines.append(
                f"  {len(self.unmatched_ring_ts)} ring bar(s) the venue does not list at "
                "the same close time"
            )
        if self.unmatched_venue_ts:
            lines.append(
                f"  {len(self.unmatched_venue_ts)} bar(s) the venue lists that never "
                "reached the ring — the bar-level denominator, from the venue's side"
            )
        if self.columns is not None:
            lines.append(f"  the same disagreement, as features: {self.columns.describe()}")
        return "\n".join(lines)


def reconcile_against_venue(
    recorded: np.ndarray,
    *,
    coin: str,
    interval: str,
    spec: FeatureSpec = PERP_BAR_V1,
    url: str | None = None,
    now_ms: int | None = None,
) -> VenueBarDiff:
    """Fetch the venue's own bars for the recorded span and diff them field by field.

    ``recorded`` is :meth:`ShadowTrader.recorded_bars` — ``(n, 6)`` int64 of
    ``ts_event, o, h, l, c, v`` in the wire's fixed-point integers. Both sides stay in
    those integers: converting to float first would introduce a difference this
    function would then attribute to the venue.

    **Touches the network**, through :func:`axon.strategies.data.fetch_candles`, which
    refuses to run without ``AXON_ALLOW_NETWORK=1``. ``/info`` is unauthenticated and
    unmetered — it cannot place, cancel or modify anything — but the default gate is
    offline as a property of the gate, not of the politeness of the request.
    """
    from axon.strategies.data import CLOSE_STAMP_OFFSET_MS, Candles, fetch_candles

    rec = np.asarray(recorded, dtype=np.int64).reshape(-1, 6)
    if rec.shape[0] == 0:
        raise ShadowError("nothing was recorded, so there is nothing to reconcile")
    step_ms = INTERVAL_MS[interval]
    first_close_ms = int(rec[0, 0] // 1_000_000) - CLOSE_STAMP_OFFSET_MS
    last_close_ms = int(rec[-1, 0] // 1_000_000) - CLOSE_STAMP_OFFSET_MS
    rows = fetch_candles(
        coin,
        interval,
        # From the first bar's *open*, because the venue ranges on the candle and a
        # request starting at the first close would drop the first bar entirely.
        start_ms=first_close_ms + 1 - step_ms,
        end_ms=last_close_ms + 1,
        url=url,
        now_ms=now_ms,
    )
    venue = Candles.from_rows(rows, coin=coin, interval=interval)

    ring_ts = rec[:, 0]
    idx = align_by_event_time(ring_ts, venue.ts_event, on_gap="report", scope="declared")
    matched_ring, matched_venue = idx.online, idx.offline
    venue_cols = np.column_stack(
        [venue.open, venue.high, venue.low, venue.close, venue.volume]
    ).astype(np.int64)

    mismatched: dict[str, int] = {}
    max_abs: dict[str, int] = {}
    differs = np.zeros((matched_ring.size, len(BAR_FIELDS)), dtype=bool)
    for j, name in enumerate(BAR_FIELDS):
        a = rec[matched_ring, j + 1]
        b = venue_cols[matched_venue, j]
        d = np.abs(a - b)
        differs[:, j] = d != 0
        mismatched[name] = int(np.count_nonzero(d))
        max_abs[name] = int(d.max()) if d.size else 0

    price_cols = [j for j, n in enumerate(BAR_FIELDS) if n != "volume"]
    vol_col = BAR_FIELDS.index("volume")
    price_bad = differs[:, price_cols].any(axis=1) if differs.size else np.zeros(0, dtype=bool)
    vol_bad = differs[:, vol_col] if differs.size else np.zeros(0, dtype=bool)

    return VenueBarDiff(
        coin=coin,
        interval=interval,
        n_ring=int(rec.shape[0]),
        n_venue=int(len(venue)),
        n_matched=int(matched_ring.size),
        field_mismatched_bars=mismatched,
        field_max_abs=max_abs,
        price_mismatched_bars=int(np.count_nonzero(price_bad)),
        volume_only_mismatched_bars=int(np.count_nonzero(vol_bad & ~price_bad)),
        columns=_feature_level_diff(
            rec[matched_ring, 1:], venue_cols[matched_venue], spec=spec
        ),
        unmatched_ring_ts=tuple(int(t) for t in ring_ts[idx.online_unmatched]),
        unmatched_venue_ts=tuple(
            int(t)
            for t in np.concatenate(
                [venue.ts_event[idx.offline_within], venue.ts_event[idx.offline_after]]
            )
        ),
    )


def _feature_level_diff(ring_ohlcv, venue_ohlcv, *, spec: FeatureSpec) -> ColumnDiff | None:
    """The same two bar series, run through the spec, diffed per column.

    The field-level diff says *which bar field* moved; this says *what that costs a
    model*. They are not the same statement: a volume field short by one lot moves a
    24-bar volume z-score for the next 24 bars, and a reader who saw only "volume
    differed by 0.001" would not know that.

    Both series are the same close times in the same order, so the two matrices warm
    up identically and the comparison needs no alignment — which is why this one may
    zip where :meth:`ShadowTrader._accumulate_columns` may not.
    """
    a = np.asarray(ring_ohlcv, dtype=np.int64)
    b = np.asarray(venue_ohlcv, dtype=np.int64)
    if a.shape[0] == 0:
        return None
    fa = spec.compute(bar_inputs(*(a[:, j] for j in range(5))))
    fb = spec.compute(bar_inputs(*(b[:, j] for j in range(5))))
    usable = finite_rows(fa) & finite_rows(fb)
    if not usable.any():
        return None
    diff = np.abs(fa[usable] - fb[usable])
    diff = np.where(np.isnan(diff), np.inf, diff)
    return ColumnDiff(
        columns=tuple(spec.columns),
        max_abs_diff=tuple(float(v) for v in diff.max(axis=0)),
        volume_columns=frozenset(volume_derived_columns(spec)),
        n_rows=int(usable.sum()),
    )


# ── a way to actually run it ─────────────────────────────────────────────────


def shadow_history(
    strategy: PerpBar,
    candles: Candles,
    *,
    symbol_id: int,
    ring_path: str,
    window_bars: int = DEFAULT_WINDOW_BARS,
    chunk: int = 16,
    model_version: int = 1,
    spec: FeatureSpec = PERP_BAR_V1,
) -> ShadowReport:
    """Shadow-run a strategy over a recorded candle history, end to end.

    The convenience wrapper the tests and the CLI both use, so neither of them owns
    a private version of the loop.
    """
    source = HistoryBarSource(candles, symbol_id=symbol_id, chunk=chunk)
    with ShadowTrader(
        strategy,
        symbol_id=symbol_id,
        ring_path=ring_path,
        interval=candles.interval,
        window_bars=window_bars,
        model_version=model_version,
        spec=spec,
    ) as trader:
        return trader.run(source)


#: Built-in strategy factories for :func:`main`'s ``--strategy``. Every one takes the
#: same four keywords and returns something with ``on_bar``/``feature_row``/``target``/
#: ``spec``/``params`` — the surface :class:`ShadowTrader` actually drives. The type
#: annotation on that class still says :class:`PerpBar` and is narrower than the
#: behaviour; the annotation is the thing that is wrong, and agent P4 drove a
#: `baseline_z` spec through it unmodified at ``780/780`` owed rows before anybody
#: widened it.
def _perp_bar_factory(*, symbol_id, max_position, registry, model):
    from axon.strategies.perp_bar import PerpBarParams

    kwargs = {} if max_position is None else {"max_position": max_position}
    params = PerpBarParams(symbol_id=symbol_id, **kwargs)
    if registry is None:
        return PerpBar(params, ConstantHalf())
    return PerpBar.from_registry(registry, model, params)


def _baseline_factory(*, symbol_id, max_position, registry, model):
    # No `from_registry` and no predictor, deliberately (see `baseline.Baseline`): an
    # object that cannot be handed an artifact cannot be handed the wrong one. A
    # `--registry` passed alongside it is therefore an operator error, not an option
    # to ignore — a silently unused artifact is how a run reports on a model it never
    # loaded.
    from axon.strategies.baseline import Baseline, BaselineParams

    if registry is not None:
        raise ShadowError(
            "--strategy baseline takes no model: it is a statistical rule with no "
            "artifact at all, which is the whole point of it. Drop --registry"
        )
    kwargs = {} if max_position is None else {"max_position": max_position}
    return Baseline(BaselineParams(symbol_id=symbol_id, **kwargs))


BUILT_IN_STRATEGIES = {
    "perp_bar": _perp_bar_factory,
    "baseline": _baseline_factory,
}


def _strategy_factory(name: str):
    """A built-in name, or ``module:callable`` for anything this file does not know.

    The escape hatch is not decoration. A model zoo lands one family at a time and
    this module must not need editing for each one — a CLI that can only name the
    strategies its own author had heard of is a CLI that gets copied and forked.
    """
    if name in BUILT_IN_STRATEGIES:
        return BUILT_IN_STRATEGIES[name]
    if ":" not in name:
        raise ShadowError(
            f"unknown strategy {name!r}; built-ins are {sorted(BUILT_IN_STRATEGIES)}, "
            "or pass 'module:callable' taking (symbol_id, max_position, registry, model)"
        )
    import importlib

    module_name, _, attr = name.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise ShadowError(f"{module_name} has no callable {attr!r}")
    return factory


class ConstantHalf:
    """A predictor that always says 50/50, for a run that only measures the diff.

    The continuous diff compares feature vectors and never looks at a score, so a
    constant is the honest default when no artifact was named — and it takes no
    position, which is why such a run reports no would-be signals at all. That is a
    property of the predictor, not a finding about the strategy.
    """

    def predict(self, x):
        return np.full(len(np.asarray(x)), 0.5)

    def declared_schema(self):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Shadow-trade a bar strategy and print what the run proved.

    ``--bar-ring`` is the live path and needs a Rust session with the candle feed on
    (``scripts/sessions/shadow-testnet-m1.toml`` is a working one); it is never reached
    by the default gate, which has no ring to attach to. Every other mode reads a file.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m axon.strategies.shadow",
        description="Shadow-trade a bar strategy: would-be targets, and a continuous diff.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--coin", help="shadow-run over a recorded candle history")
    source.add_argument(
        "--bar-ring",
        metavar="MD_RING_PATH",
        help="attach to a live Rust session's market-data ring (the SLICE ring path; "
        "the bar ring beside it is derived)",
    )
    parser.add_argument("--interval", default="1h")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="use the full cached history rather than the committed 800-bar fixture",
    )
    parser.add_argument(
        "--symbol-id",
        type=int,
        action="append",
        help="repeatable on --bar-ring: one shadow trader per symbol, all fed from the "
        "SAME reader, because two consumers on one ring each pop a share of the bars "
        "and neither can tell that from a quiet feed. The first is the one --coin and "
        "--reconcile-venue apply to. Default 0",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_BARS)
    parser.add_argument("--ring", default="/dev/shm/axon-shadow.ring", help="the signal ring")
    parser.add_argument(
        "--strategy",
        default="perp_bar",
        help="which strategy to shadow: a built-in name (perp_bar, baseline) or a "
        "'module:callable' factory taking (symbol_id, max_position, registry, model) as "
        "keywords and returning an object with on_bar/feature_row/target/spec/params. "
        "The spec is read off the strategy, never assumed, so a family with its own "
        "feature set diffs against its own recompute",
    )
    parser.add_argument(
        "--max-position",
        action="append",
        help="position size per --symbol-id, in coin units, as a decimal STRING (never "
        "a float: 0.0003 has no exact float form and the venue rounds the difference "
        "somewhere nobody chose). PerpBarParams' 0.01 BTC default is ~$640, thirteen "
        "times the Phase-6 ceiling — a would-be notional reported at it does not "
        "transfer to a live run",
    )
    parser.add_argument(
        "--registry", help="a model registry directory; without one no target is ever taken"
    )
    parser.add_argument("--model", default="perp_bar_xgb")
    parser.add_argument("--idle-timeout", type=float, default=None)
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="stop after this many bars for THIS symbol. Event-driven rather than a "
        "wall-clock duration, so the run is bounded by what it observed; pair it with "
        "--idle-timeout, which is the only bound that can end a run on a dead feed",
    )
    parser.add_argument(
        "--attest-venue",
        action="store_true",
        help="the operator's claim that a real Hyperliquid session is feeding this ring. "
        "Never inferred: an MdBar record carries no source marker, so nothing in the "
        "process can tell the Rust publisher from a harness writer",
    )
    parser.add_argument(
        "--reconcile-venue",
        metavar="COIN",
        action="append",
        help="after the run, refetch this coin's bars with POST /info candleSnapshot and "
        "diff them against the ones the ring published, field by field. Repeat it once "
        "per --symbol-id, in the same order. NETWORK: needs AXON_ALLOW_NETWORK=1 (and "
        "AXON_HL_INFO_URL for testnet). A measurement beside a run, never a source a "
        "session reads",
    )
    parser.add_argument(
        "--bars-out",
        metavar="PATH",
        help="write the bars each trader was shown to PATH as CSV, in the wire's "
        "fixed-point integers (PATH.<symbol_id> for every symbol after the first). The "
        "recording is the only artefact that outlives a ring, and a ring is gone when "
        "the session is",
    )
    args = parser.parse_args(argv)

    from axon.strategies.data import fixture_candles, load_candles

    symbol_ids = args.symbol_id or [0]
    sizes = args.max_position or []
    if sizes and len(sizes) != len(symbol_ids):
        parser.error(
            f"--max-position was given {len(sizes)} time(s) for {len(symbol_ids)} "
            "symbol(s). One size for one symbol, in order: a size silently reused "
            "across instruments is a would-be notional that means nothing"
        )
    registry = None
    if args.registry:
        from axon.models import ModelRegistry

        registry = ModelRegistry(args.registry)

    def build(symbol_id: int, max_position: str | None):
        factory = _strategy_factory(args.strategy)
        return factory(
            symbol_id=symbol_id,
            max_position=None if max_position is None else Decimal(max_position),
            registry=registry,
            model=args.model,
        )

    recorded = None
    if args.bar_ring:
        # Attached **unfiltered**, and each trader does its own selecting. A source
        # that filtered would drop every other instrument's bar before the runners saw
        # it, and their event clocks would then advance only on their own symbol's
        # bars — so a shadow run on a quiet coin would order its passes differently
        # from the live session it is supposed to be reproducing. ADR-0029 §5: other
        # instruments are dispatched and recorded for none.
        with RingBarSource.attach(args.bar_ring, symbol_id=None) as src:
            built = [
                build(sid, sizes[i] if sizes else None) for i, sid in enumerate(symbol_ids)
            ]
            traders = [
                ShadowTrader(
                    strat,
                    symbol_id=sid,
                    # Off the strategy, never assumed. A trader diffing `perp_bar`'s
                    # nine columns against a strategy serving two would align on event
                    # time, find every stamp, and compare the wrong arrays.
                    spec=getattr(strat, "spec", PERP_BAR_V1),
                    # One signal ring per trader: two runners producing into one ring
                    # would interleave targets for different instruments under one
                    # sequence, and the turnover read back off it would be nonsense.
                    ring_path=args.ring if len(symbol_ids) == 1 else f"{args.ring}.{sid}",
                    interval=args.interval,
                    window_bars=args.window,
                    venue_attested=args.attest_venue,
                )
                for sid, strat in zip(symbol_ids, built)
            ]
            try:
                reports = drive(
                    traders, src, idle_timeout_s=args.idle_timeout, max_bars=args.max_bars
                )
                recorded = [t.recorded_bars() for t in traders]
            finally:
                for t in traders:
                    t.close()
    else:
        candles = (
            load_candles(args.coin, args.interval)
            if args.cache
            else fixture_candles(args.coin, args.interval)
        )
        strat = build(symbol_ids[0], sizes[0] if sizes else None)
        reports = [
            shadow_history(
                strat,
                candles,
                symbol_id=symbol_ids[0],
                ring_path=args.ring,
                window_bars=args.window,
                spec=getattr(strat, "spec", PERP_BAR_V1),
            )
        ]
    for sid, report in zip(symbol_ids, reports):
        print(f"── symbol {sid} ".ljust(78, "─"))
        print(report.summary())

    if recorded is not None and args.bars_out:
        for i, (sid, bars) in enumerate(zip(symbol_ids, recorded)):
            path = args.bars_out if i == 0 else f"{args.bars_out}.{sid}"
            np.savetxt(
                path,
                bars,
                fmt="%d",
                delimiter=",",
                header="ts_event,open,high,low,close,volume",
                comments="",
            )
            print(f"  symbol {sid}: {len(bars)} bar(s) written to {path}")

    ok = all(r.passed for r in reports)
    for i, coin in enumerate(args.reconcile_venue or []):
        if recorded is None:
            parser.error("--reconcile-venue needs --bar-ring: there is nothing to reconcile")
        venue = reconcile_against_venue(recorded[i], coin=coin, interval=args.interval)
        print(venue.describe())
        # A price break fails the process even when the shadow diff is green, because
        # the two say different things: the shadow diff proves the serving path
        # computed what a recompute over the same bars computes, and this proves the
        # bars were the venue's. A run can pass the first and fail the second.
        ok = ok and not venue.price_break
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "BAR_FIELDS",
    "BUILT_IN_STRATEGIES",
    "DEFAULT_MAX_SIGNAL_AGE_MS",
    "DEFAULT_WINDOW_BARS",
    "HEARTBEAT_EVERY_S",
    "MEASURED_CORE_CLOCK_LAG_MS",
    "SILENCE_AFTER_BARS",
    "ArrivalLag",
    "BarCoverage",
    "BarSource",
    "ColumnDiff",
    "ConstantHalf",
    "FeedHealth",
    "HistoryBarSource",
    "RingBarSource",
    "ShadowError",
    "ShadowReport",
    "ShadowSignal",
    "ShadowTrader",
    "Turnover",
    "VenueBarDiff",
    "column_sources",
    "drive",
    "main",
    "BarParityDiff",
    "publish_bars",
    "reconcile_against_venue",
    "shadow_history",
    "volume_derived_columns",
]
