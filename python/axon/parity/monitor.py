"""The live parity monitor: the gates, run forever, against a session in flight.

``docs/03`` lists it as the mandatory backstop — *"sample live feature vectors,
recompute via the offline path, alarm on divergence"* — and ``docs/08`` has carried
it as the last open Phase-5 line: the gates exist, and nothing runs them
continuously. This module is that loop. It is deliberately **not** a new gate; it
is a state machine over :mod:`axon.parity.features` and :mod:`axon.parity.drift`
verdicts, because a monitor that reimplements the comparison is a second
implementation of the thing whose single implementation is the entire point
(ADR-0016, ``docs/03`` "never implement a feature twice").

Five decisions carry the design, and each has a comfortable wrong answer.

**1. Silence is a verdict, not the absence of one.** A window that compared nothing
returns :attr:`Level.SILENT`, never :attr:`Level.OK`. "OK, 0 rows compared" is the
exact failure this workstream exists to close — an invisible denominator — one
level up from the alignment that produced it, and it is the reading a monitor
naturally produces when the feed it watches has died. :class:`Coverage` closes it
inside one window; this closes it across windows, where a *sequence* of empty
windows is the symptom. And the monitor has a denominator of its own, which is what
:attr:`MonitorConfig.scope` is for: a driver whose window reference is exactly the
owed rows must say so, or a serving path blind at a window's opening is excused as a
late join *and* those rows leave the run's own denominator with it. Three levels,
one bug; it is stated once here rather than re-derived in each driver.

**2. Feature divergence is alarmed; distribution drift is capped at a warning.**
There is no acceptable *rate* of feature mismatch: ``atol``/``rtol`` already absorb
the only legitimate difference (summation order between a vectorized recompute and
an incremental online one), so a cell past tolerance is a bug and one is enough.
Drift is the opposite — a moved market is not a bug — so it is banded, confirmed
across consecutive windows, **and held below the parity alarm by default**. That
last part is measured rather than cautious: over ``perp_bar``'s own real history,
4,975 hourly rows in 256-row windows with feature parity green on every one, PSI
passes the conventional 0.25 band on 18 of 20 windows and peaks at 5.97. ADR-0016
§5 already records those bands as the industry convention rather than something
derived from these features; this is the bill. An operator taught to ignore ALARM
by the drift line has been taught to ignore the parity line, which has never had a
false positive at all.

**3. An alarm logs, and nothing else — for now, and the reason is not squeamishness.**
Two authorities that can independently stop trading, neither aware of the other, is
precisely the shape ADR-0013's "the loop that must die last" was written to avoid;
the venue-side dead-man's switch and the runtime's intent source already own that
decision. And this detector has **never been run against a live session**, so its
false-positive rate is unmeasured — wiring an unmeasured detector to a kill switch
converts every quirk of the feed into an outage the monitor caused. The seam is
here (:class:`AlarmSink`) so escalation is a constructor argument once there is a
measured rate to justify it, rather than a rewrite.

**4. Silence is measured on a wall clock, and it is the only thing that may be.**
Everything else here orders on ``ts_event`` — a window's span, the alignment, the
drift sample. But the absence of an event has no event time, and an event-time
deadline for "nothing has arrived" can never expire: the clock that would advance
it *is* the thing that stopped. So the deadline reads an injected
:class:`Clock` (``time.time_ns`` in production, a counter in tests), which is also
what keeps every test in this file deterministic and offline.

**5. And a deadline is a guess, so where a fact is available the monitor takes it.**
The deadline above exists because under ``MdWritePolicy::OnChange`` an empty ring is
ambiguous *by design* — a flat top of book and a dead publisher write the same
nothing. :mod:`axon.parity.beacon` reads the sidecar the Rust publisher's **pass loop**
writes, which advances whether or not anything arrived, and that turns the guess into
an observation. Give :class:`ParityMonitor` a ``beacon`` probe and :attr:`Level.SILENT`
resolves: a beat that stopped is ``ALARM`` *immediately*, with nothing left for sixty
seconds to establish; a beat that is advancing over a market that did not move is
``WARN`` and the deadline is **suspended**, because alarming on a quiet market for an
hour is how an operator learns to ignore the parity line — the same argument decision 2
makes about drift, one floor down. Without a probe every behaviour above is exactly
what it was, which is the point: the beacon removes a blind spot and adds no
dependency.

**6. And the deadline itself is a guess until a run measures it, so every run
measures it.** :data:`DEFAULT_SILENCE_AFTER_NS` is sixty seconds because sixty seconds
is longer than a GC pause and shorter than a human notices — which is a sentence about
GC pauses and humans, not about this feed. :class:`SilenceEvidence` is what turns it
into a number that can be argued about: every run records the wall-clock gaps between
the windows that *compared* something (the quantity the deadline is actually judged
against), between consecutive *observations* of any kind (the resolution, without which
a gap of five minutes might only mean nobody asked for five minutes), and — when a
beacon is wired — between the readings in which the publisher's ring advanced. It is
recorded on every run and printed on every report because a number counted and not
surfaced is a number not counted, and because the one thing a monitor is uniquely
placed to measure is its own silence. It deliberately does **not** propose a
replacement: a deadline fitted to the tape that has been seen ratchets, and records
the next regression as the new bar.

Offline by construction: nothing in this module opens a socket or a ring. It
consumes :class:`Window` objects, so the same monitor runs over a recording, over a
fixture history, or over a live feed a caller pumps into it — and the default gate
never touches the network. The beacon is the one file this package will read, and it
still does not open it: the monitor takes a *callable* returning a snapshot, so a test
injects a fake and the caller that owns the session owns the mapping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np

from axon.parity.beacon import (
    UNKNOWN_PUBLISHER,
    MdBeaconSnapshot,
    Publisher,
    PublisherState,
    publisher_state,
)
from axon.parity.drift import DEFAULT_NAN_RATE_TOL, Binning, DriftReport, drift_report
from axon.parity.features import (
    FEATURE_ATOL,
    FEATURE_RTOL,
    Coverage,
    FeatureParityReport,
    aligned_feature_parity,
)
from axon.parity.gate import ParityError, _raise_unless

LOGGER = logging.getLogger("axon.parity.monitor")

#: Wall-clock nanoseconds. Injected so the silence deadline is testable without a
#: sleep — see decision 4 in the module docstring for why silence, and only
#: silence, is allowed to read one.
Clock = Callable[[], int]

#: One reading of the market-data beacon, or ``None`` when the publisher has not created
#: one yet. A callable rather than a path so this module still opens nothing:
#: :class:`axon.parity.beacon.MdBeaconReader` *is* one, and a test injects a list.
BeaconProbe = Callable[[], "MdBeaconSnapshot | None"]

#: Every verdict, including the ``OK`` ones — which is exactly what makes it a different
#: seam from :class:`AlarmSink` rather than a duplicate of it. The sink is fed only what
#: an operator has to *act* on, because a sink that receives every window is a log
#: nobody reads; a live harness watching a session in flight needs the opposite, one
#: line per window whatever it said, because a run that printed nothing for ten minutes
#: and a run that was not running look identical from the outside. Nothing here reads
#: the return value and nothing branches on it: a tap that could change a verdict would
#: be a second place levels are decided.
VerdictTap = Callable[["Verdict"], None]

#: How long a run of empty windows is tolerated before it stops being "the market is
#: quiet" and becomes "the feed is dead". Sixty seconds is a *starting* value chosen
#: to be longer than any plausible GC pause or reconnect and shorter than a human
#: notices; it has not been measured against a live session and is the first number
#: a soak should replace. A :data:`BeaconProbe` is what makes it stop being the only
#: evidence available — see decision 5.
DEFAULT_SILENCE_AFTER_NS = 60_000_000_000

#: Live rows below which PSI is noise rather than signal. A drift number over a
#: handful of rows is dominated by the sample, and reporting it would train an
#: operator to ignore the one gate that has to keep working for months.
DEFAULT_MIN_DRIFT_ROWS = 200

#: Consecutive windows a drift band must hold before it escalates. One window is a
#: sample; two in a row is a state.
DEFAULT_CONFIRM_WINDOWS = 2

#: How many wall-clock gap samples :class:`SilenceEvidence` retains, per series. A
#: driver that heartbeats once per poll rather than once per bar asks for a verdict
#: thousands of times a minute, and an unbounded sample would grow to a size nobody
#: chose on the runs that last longest — which are exactly the runs whose evidence is
#: worth keeping. Past this the recording stops and the evidence **says it stopped**:
#: quantiles taken over a silently truncated sample and presented as the run's are the
#: invisible-denominator bug wearing a histogram. 262 144 is ~2 MB per series, and three
#: days at one observation a second.
MAX_GAP_SAMPLES = 262_144


class Level(IntEnum):
    """Severity of one window's verdict, ordered so ``max()`` is "the worse of".

    :attr:`SILENT` sits **above** :attr:`WARN` and below :attr:`ALARM` on purpose. A
    window that compared nothing is strictly worse than one that compared everything
    and found a distribution wobble — it is not evidence of health at all — but a
    single quiet window is not yet evidence of a dead feed either. Persisting past
    the silence deadline is what promotes it to :attr:`ALARM`.

    What :attr:`SILENT` means precisely is *"I cannot tell whether there was anything
    to compare"*, and that is why it is a level rather than a synonym for `OK`. Under
    :attr:`MonitorConfig.scope` ``"declared"`` the question is already answered — the
    window's offline side is what the serving path owed — so a window that owed rows
    and compared none is :attr:`ALARM` immediately, with no deadline to wait out.
    """

    OK = 0
    WARN = 1
    SILENT = 2
    ALARM = 3

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass(frozen=True)
class Window:
    """One batch of online feature rows and the offline recompute of the same span.

    Both sides carry their own ``ts_event`` because the alignment is on event time
    and on nothing else; a row and its stamp are one observation, and re-pairing
    them by position after they have been separated is the bug the alignment exists
    to avoid.

    ``offline`` is the *reference*: the same :class:`~axon.features.FeatureSpec`,
    recomputed over the raw inputs the online path was shown. Under Boundary B that
    is literally the same function, which is why a nonzero ``max_abs_diff`` here
    means a different *input*, not different arithmetic.
    """

    online: np.ndarray
    offline: np.ndarray
    online_ts: np.ndarray
    offline_ts: np.ndarray

    def __post_init__(self) -> None:
        for name in ("online", "offline", "online_ts", "offline_ts"):
            object.__setattr__(self, name, np.asarray(getattr(self, name)))
        # An empty side arrives as shape ``(0,)`` from every natural slicing, and the
        # gate wants ``(0, n_features)``. Widening it here rather than refusing it is
        # the difference between a monitor that reports a blind window and one that
        # crashes on the window it exists to report.
        width = next(
            (m.shape[1] for m in (self.offline, self.online) if m.ndim == 2), None
        )
        if width is not None:
            for name in ("online", "offline"):
                m = getattr(self, name)
                if m.ndim != 2 and m.size == 0:
                    object.__setattr__(self, name, m.reshape(0, width))
        for matrix, stamps, name in (
            (self.online, self.online_ts, "online"),
            (self.offline, self.offline_ts, "offline"),
        ):
            if matrix.ndim != 2:
                raise ValueError(f"{name} must be a 2-D feature matrix, got {matrix.shape}")
            if matrix.shape[0] != stamps.size:
                raise ValueError(
                    f"{name}: {matrix.shape[0]} rows but {stamps.size} event times — a "
                    "row and its stamp are one observation"
                )


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds, and what each one is for.

    ``columns`` must be the spec's column tuple, in matrix order
    (:attr:`axon.features.FeatureSpec.columns`) — the gate reports by column name,
    and a name list out of order is a report that blames the wrong feature.
    """

    columns: tuple[str, ...]
    #: Feature tolerances. Defaults are the gate's own; a monitor that loosened them
    #: would be measuring something other than what CI measured, which makes the two
    #: incomparable exactly when they disagree.
    atol: float = FEATURE_ATOL
    rtol: float = FEATURE_RTOL
    #: The reference sample drift is measured against — the training rows. Without
    #: it the monitor runs the parity half only and *says so*; it does not quietly
    #: report a drift-free session it never measured.
    reference: np.ndarray | None = None
    #: Bins frozen at training time (:func:`axon.parity.quantile_binning`).
    #: Recomputing them per window makes both histograms uniform by construction and
    #: PSI reads ~0 whatever happened (ADR-0016 §5) — the monitor would be
    #: structurally blind, which is the same disease as the coverage bug.
    binnings: tuple[Binning, ...] | None = None
    #: What the ``offline`` side of each :class:`Window` means, passed straight to
    #: :func:`~axon.parity.align_by_event_time`. ``"observed"`` (the default) is right
    #: for a monitor whose window opens mid-history: the online side may legitimately
    #: not have been running for the whole reference. A driver that builds each
    #: window's reference as *exactly* the rows the serving path owed — as
    #: ``axon.strategies.shadow`` does — must pass ``"declared"``, or a path blind at
    #: a window's opening reads as a late join and those rows leave the denominator
    #: with it. That is ADR-0030's founding bug two levels out, and it is held here
    #: rather than in the driver so the monitor and a gate over the same window cannot
    #: return different verdicts.
    scope: str = "observed"
    min_drift_rows: int = DEFAULT_MIN_DRIFT_ROWS
    confirm_windows: int = DEFAULT_CONFIRM_WINDOWS
    silence_after_ns: int = DEFAULT_SILENCE_AFTER_NS
    nan_rate_tol: float = DEFAULT_NAN_RATE_TOL
    #: The loudest a drift finding may get. **WARN, and this is measured rather than
    #: cautious**: over ``perp_bar``'s own real history — 4,975 hourly BTC rows, in
    #: 256-row windows against the training sample, with feature parity green on
    #: every one of them — PSI exceeds the conventional 0.25 band on 18 of 20 windows
    #: and peaks at 5.97 on ``vol_24``. Nothing is wrong; a 256-bar window of realized
    #: volatility simply does not look like a 3,000-bar one. ADR-0016 §5 already flags
    #: the bands as "the industry convention, not something derived from these
    #: features", and this is what that costs. Drift at ALARM would fire on nearly
    #: every window forever, and an operator who learns to ignore ALARM has also
    #: learned to ignore the parity alarm — which has no false positives at all. Raise
    #: it per strategy once its bands are calibrated on its own history.
    drift_ceiling: Level = Level.WARN

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a monitor with no columns would compare nothing and pass")
        if self.scope not in ("observed", "declared"):
            # Checked here as well as in the aligner, and not because the aligner is
            # unreliable. A monitor is wired once and then runs; a scope the aligner
            # only rejects on the first `observe` is an exception inside the loop
            # rather than a refusal at wiring time — the same argument the runtime's
            # config validation makes for refusing an unsafe session before it starts
            # instead of on its first order.
            raise ValueError(f"scope must be 'observed' or 'declared', got {self.scope!r}")
        if self.confirm_windows < 1:
            raise ValueError("confirm_windows must be at least 1")
        if self.silence_after_ns <= 0:
            raise ValueError("silence_after_ns must be positive")
        if self.reference is not None:
            ref = np.asarray(self.reference)
            if ref.ndim != 2 or ref.shape[1] != len(self.columns):
                raise ValueError(
                    f"reference has shape {ref.shape}; expected (n, {len(self.columns)}) "
                    "to match the spec's columns"
                )
        if self.binnings is not None and len(self.binnings) != len(self.columns):
            raise ValueError(f"{len(self.binnings)} binnings for {len(self.columns)} columns")

    @property
    def drift_enabled(self) -> bool:
        return self.reference is not None


@dataclass(frozen=True)
class Verdict:
    """What one window proved, and what it could not.

    ``reasons`` is the operator-facing part: a level with no reason is a number
    nobody can act on at 03:00, which ``gate.py`` already refuses for the gates
    themselves.
    """

    seq: int
    level: Level
    reasons: tuple[str, ...]
    n_compared: int
    first_ts: int | None
    last_ts: int | None
    parity: FeatureParityReport | None = None
    drift: DriftReport | None = None
    coverage: Coverage | None = None
    silent_for_ns: int = 0
    #: What the market-data beacon said about the publisher over the interval this
    #: window covers. :attr:`~axon.parity.beacon.Publisher.UNKNOWN` when no probe was
    #: wired, which is also what every behaviour here degrades to.
    publisher: PublisherState = UNKNOWN_PUBLISHER

    @property
    def passed(self) -> bool:
        """``OK`` only. ``WARN`` is not a pass — it is a pass that is on notice."""
        return self.level is Level.OK

    def summary(self) -> str:
        span = "no rows"
        if self.first_ts is not None and self.last_ts is not None:
            span = f"ts {self.first_ts}..{self.last_ts}"
        head = f"[{self.level}] window {self.seq}: {self.n_compared} row(s) compared, {span}"
        if self.publisher.state is not Publisher.UNKNOWN:
            head += f", publisher={self.publisher.state}"
        return "\n".join([head] + [f"  {r}" for r in self.reasons])


def _secs(ns: int) -> str:
    """Nanoseconds as seconds, at millisecond resolution.

    Three decimals rather than two because the gaps this formats span six orders of
    magnitude — a poll cadence of a millisecond and a quiet hourly bar — and a format
    that rounds the small ones to ``0.00s`` would make the *resolution* series, whose
    only job is to say how finely the others were sampled, unreadable at exactly the
    cadence a live driver runs at.
    """
    return f"{ns / 1e9:.3f}s"


@dataclass(frozen=True)
class GapStats:
    """One series of wall-clock gaps, summarised without inventing a number.

    Percentiles are taken ``method="nearest"``, so every figure printed is a gap that
    was **observed** rather than an interpolation between two that were. A deadline
    argued from an interpolated p99 is a deadline argued from a measurement nobody
    made, which is the one thing this object exists to prevent.

    ``n == 0`` prints as *"no observation at all"* and never as a zero. "Zero seconds
    of silence" and "nothing was ever measured" are opposite statements and only one of
    them is ever true here — the same distinction the drift half draws between "stable"
    and "not measured".
    """

    n: int
    min_ns: int = 0
    p50_ns: int = 0
    p90_ns: int = 0
    p99_ns: int = 0
    max_ns: int = 0

    @classmethod
    def of(cls, samples: Sequence[int]) -> "GapStats":
        if len(samples) == 0:
            return cls(n=0)
        a = np.asarray(samples, dtype=np.int64)
        p50, p90, p99 = np.percentile(a, (50, 90, 99), method="nearest")
        return cls(
            n=int(a.size),
            min_ns=int(a.min()),
            p50_ns=int(p50),
            p90_ns=int(p90),
            p99_ns=int(p99),
            max_ns=int(a.max()),
        )

    def describe(self, what: str) -> str:
        if self.n == 0:
            return f"{what}: no observation at all — nothing here was measured"
        return (
            f"{what}: n={self.n} min={_secs(self.min_ns)} p50={_secs(self.p50_ns)} "
            f"p90={_secs(self.p90_ns)} p99={_secs(self.p99_ns)} max={_secs(self.max_ns)}"
        )


@dataclass(frozen=True)
class SilenceEvidence:
    """What a run measured about its own silence — the case for or against the deadline.

    :data:`DEFAULT_SILENCE_AFTER_NS` is sixty seconds because sixty seconds is longer
    than a plausible GC pause and shorter than a human notices. That is a claim about
    GC pauses and humans; it is not a claim about this feed, and ADR-0030 records it as
    *"a starting value, not a measurement"*. This is the record from which a measured
    one could be argued, and it is collected on **every** run — offline included — so
    that the number is not hostage to somebody remembering to switch a flag on for the
    one session that mattered.

    Three series, and the second is what keeps the first honest:

    * :attr:`between_compared_windows` — wall-clock nanoseconds between two windows
      that each compared at least one row. This is *exactly* the quantity
      ``silence_after_ns`` is judged against: the deadline is measured from the last
      window that made progress, so a deadline shorter than the ordinary gap between
      two healthy windows alarms on a healthy session. That is not hypothetical —
      ``axon.strategies.shadow`` had to override the default for precisely this reason,
      and a live m1 run logged 296 SILENT verdicts in 160 seconds before the first
      window closed.
    * :attr:`between_observations` — nanoseconds between consecutive
      :meth:`ParityMonitor.observe`/:meth:`ParityMonitor.heartbeat` calls, whatever
      they returned. The **resolution**. Without it, a five-minute gap in the first
      series is ambiguous between *"the feed was quiet for five minutes"* and *"nobody
      asked for five minutes"*, and those two want opposite responses.
    * :attr:`between_ring_advances` — nanoseconds between two beacon readings in which
      the publisher's slice or bar counters moved. Read it as an **upper bound
      quantised by the observation cadence**: it measures how long a *poll* went
      without finding the ring had moved, not how long the ring itself went without a
      record, because the beacon carries counters and not a per-record clock. It is
      informative in one direction only — a long gap really was a still ring; a short
      one may only be a fast poll.

    What it deliberately does not do is propose a replacement value. A tolerance fitted
    to what today's run produced ratchets to today's run, and records the next
    regression as the new bar — the argument ADR-0021 made for refusing to fit
    ``ONNX_TIGHT_EPS`` to a measurement, and it applies with more force here, because a
    deadline is a *detector* and a detector fitted to one healthy tape is a detector
    calibrated to never fire.
    """

    #: The deadline that was in force while this evidence was collected. Carried so the
    #: description can compare against it rather than re-reading a config that may have
    #: been constructed differently — two descriptions of one fact can disagree.
    silence_after_ns: int
    beacon_wired: bool
    between_compared_windows: GapStats
    between_observations: GapStats
    between_ring_advances: GapStats
    #: ``(publisher state name, times seen)``, most frequent first. A tuple rather than
    #: a mapping so the whole report stays hashable and comparable.
    publisher_states: tuple[tuple[str, int], ...] = ()
    truncated: bool = False

    @property
    def diagnosed(self) -> bool:
        """Whether the beacon ever said anything but ``unknown``.

        A wired probe that only ever returned ``unknown`` — no file, an unreadable
        file, or a session that ended after one reading — leaves silence exactly as
        uncategorised as no probe at all, and a report that printed "beacon: wired"
        without this distinction would be claiming a diagnosis nobody got.
        """
        return any(name != "unknown" and n for name, n in self.publisher_states)

    def describe(self) -> str:
        lines = [
            "silence evidence — measured on the injected wall clock, which is the one "
            "exception here because the absence of an event has no event time:",
            f"  {self.between_compared_windows.describe('between compared windows')}",
            f"  {self.between_observations.describe('between observations (resolution)')}",
            f"  {self.between_ring_advances.describe('between ring advances (beacon)')}",
        ]
        if self.publisher_states:
            seen = ", ".join(f"{name}x{n}" for name, n in self.publisher_states)
            lines.append(f"  publisher states seen: {seen}")

        deadline = _secs(self.silence_after_ns)
        widest = self.between_compared_windows
        if widest.n == 0:
            lines.append(
                f"  no two windows ever compared rows, so the {deadline} deadline was "
                "never tested against anything: this run is evidence about a monitor "
                "that saw nothing, not about a deadline"
            )
        elif widest.max_ns >= self.silence_after_ns:
            lines.append(
                f"  the widest gap between two compared windows was {_secs(widest.max_ns)}, "
                f"which REACHED the {deadline} deadline — every window in that gap was "
                "graded SILENT, and SILENT outranks WARN"
            )
        else:
            headroom = self.silence_after_ns / max(widest.max_ns, 1)
            lines.append(
                f"  the widest gap between two compared windows was {_secs(widest.max_ns)} "
                f"against a {deadline} deadline, {headroom:.2f}x of headroom"
            )

        if not self.beacon_wired:
            # Unmissable on purpose, and it is the whole difference decision 5 makes.
            # A report that quietly omitted this reads exactly like one from a session
            # whose publisher was proven alive.
            lines.append(
                "  !! NO BEACON WAS WIRED: every silence above is UNCATEGORISED. A quiet "
                "publisher and a dead one wrote the same nothing, and the deadline was "
                "the only evidence this run had"
            )
        elif not self.diagnosed:
            lines.append(
                "  !! A BEACON WAS WIRED AND NEVER SAID ANYTHING BUT 'unknown': no file, "
                "an unreadable one, or a run too short for the two readings a verdict "
                "about a counter needs. Silence here is as uncategorised as with no probe"
            )
        if self.truncated:
            lines.append(
                f"  the gap sample stopped at {MAX_GAP_SAMPLES} observations; the figures "
                "above describe that prefix and not the whole run"
            )
        lines.append(
            "  NOT a recommendation. These are the gaps of one run over one tape, and a "
            "deadline fitted to them is a detector calibrated to whatever that tape "
            "happened to do — the ratchet ADR-0021 refused when it declined to fit a "
            "tolerance to a measurement. What a run has to show before this may replace "
            "the stated 60 s is in docs/adr/0030-live-parity-monitor-and-the-coverage-denominator.md."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class MonitorReport:
    """Everything a monitor run saw, as a gate report like every other one here."""

    windows: int
    worst: Level
    n_compared: int
    n_ok: int
    n_alarm: int
    n_silent: int
    max_abs_diff: float
    worst_psi: float
    reasons: tuple[str, ...]
    #: What this run measured about its own silence (decision 6). Optional and last so
    #: that a caller who builds a report by hand — a test, a summary of summaries —
    #: still constructs, and ``None`` prints nothing rather than printing zeros: a
    #: report *without* the evidence and a report *of* no silence must not look alike.
    silence: SilenceEvidence | None = None

    @property
    def passed(self) -> bool:
        """No window worse than :attr:`Level.WARN` — **and at least one window that
        compared something**. A run of zero windows, or of windows that all compared
        zero rows, is the monitor's own version of the bug it exists to catch: every
        threshold satisfied because nothing was ever measured against them."""
        return self.worst <= Level.WARN and self.n_compared > 0 and self.windows > 0

    def summary(self) -> str:
        head = (
            f"parity monitor {'OK' if self.passed else 'ALARM'}: "
            f"{self.windows} window(s), {self.n_compared} row(s) compared, "
            f"worst={self.worst} (ok={self.n_ok} alarm={self.n_alarm} silent={self.n_silent}) "
            f"max_abs_diff={self.max_abs_diff:.3e} worst_psi={self.worst_psi:.4f}"
        )
        # The silence evidence rides along on every branch, including the two that
        # return early. It is *most* worth reading on a run that compared nothing —
        # that is the run where "how long was it quiet, and did the publisher beat
        # through it" is the whole question — and a summary that dropped it exactly
        # there would omit the number on the report that needed it.
        tail = [] if self.silence is None else [self.silence.describe()]
        if self.windows == 0:
            return "\n".join([head, "  no window was ever offered — the monitor never ran", *tail])
        if self.n_compared == 0:
            return "\n".join(
                [
                    head,
                    "  no window compared a single row: this is a report about a monitor "
                    "that saw nothing, not about a session that was correct",
                    *tail,
                ]
            )
        return "\n".join([head] + [f"  {r}" for r in self.reasons] + tail)

    def raise_for_status(self) -> None:
        _raise_unless(self.passed, self.summary())


class AlarmSink(Protocol):
    """Where a verdict goes when it is worse than :attr:`Level.OK`."""

    def __call__(self, verdict: Verdict) -> None: ...


def logging_sink(logger: logging.Logger | None = None) -> AlarmSink:
    """The default sink: one log line per non-OK window, at a level that matches.

    Logging and not flattening — see decision 3 in the module docstring. The whole
    verdict goes out, not the level, because "parity ALARM" without the column and
    the row is the bare boolean ``gate.py`` refuses to return.
    """
    log = logger if logger is not None else LOGGER

    def sink(verdict: Verdict) -> None:
        severity = {
            Level.OK: logging.DEBUG,
            Level.WARN: logging.WARNING,
            Level.SILENT: logging.WARNING,
            Level.ALARM: logging.ERROR,
        }[verdict.level]
        log.log(severity, "%s", verdict.summary())

    return sink


def collecting_sink(into: list) -> AlarmSink:
    """Append verdicts to a list — for tests, and for an operator UI that polls."""

    def sink(verdict: Verdict) -> None:
        into.append(verdict)

    return sink


@dataclass
class ParityMonitor:
    """The gates, held across windows, with the state a single window cannot have.

    Three things live here and nowhere else: the run of consecutive empty windows
    (a dead feed looks like one quiet window, repeatedly), the run of consecutive
    drift bands (a band that holds is a state, a band that flickers is a sample),
    and the totals that make :meth:`report` able to say "and nothing was ever
    compared".
    """

    config: MonitorConfig
    sink: AlarmSink | None = None
    clock: Clock = time.time_ns
    #: The market-data beacon, as a callable returning one reading (decision 5). ``None``
    #: keeps every behaviour exactly as it was before ADR-0030's seam was built: the
    #: silence deadline remains the only evidence, and the monitor still refuses to
    #: report ``OK`` — it simply cannot say *which* kind of nothing it is looking at.
    beacon: BeaconProbe | None = None
    #: Every verdict as it is produced, ``OK`` ones included — see :data:`VerdictTap`
    #: for why this is not :attr:`sink`. A live harness prints one line per window from
    #: here; nothing in this module reads what it returns, so a tap cannot move a level.
    tap: VerdictTap | None = None

    seq: int = field(default=0, init=False)
    n_compared: int = field(default=0, init=False)
    n_ok: int = field(default=0, init=False)
    n_alarm: int = field(default=0, init=False)
    n_silent: int = field(default=0, init=False)
    worst: Level = field(default=Level.OK, init=False)
    max_abs_diff: float = field(default=0.0, init=False)
    worst_psi: float = field(default=0.0, init=False)
    _reasons: list[str] = field(default_factory=list, init=False)
    _consecutive_drift: int = field(default=0, init=False)
    _last_progress_ns: int | None = field(default=None, init=False)
    _last_beacon: MdBeaconSnapshot | None = field(default=None, init=False)

    # ── the silence evidence (decision 6) ─────────────────────────────────────
    # Three series of wall-clock gaps and the two cursors they are differenced from.
    # Kept as plain lists and turned into a frozen :class:`SilenceEvidence` by
    # :meth:`report`, so nothing that reads a report can push a sample into it.
    #
    # `_compared_at_ns` is deliberately **not** `_last_progress_ns`: that one is seeded
    # to `now` on the very first call whether or not anything was compared, because it
    # is the deadline's origin and a monitor whose source never delivers must still
    # reach a deadline. Differencing against it would record a first "gap" of zero that
    # no silence ever had, and a zero in a min/p50 is a measurement, not a placeholder.
    _compared_at_ns: int | None = field(default=None, init=False)
    _observed_at_ns: int | None = field(default=None, init=False)
    _advanced_at_ns: int | None = field(default=None, init=False)
    _gaps_compared: list[int] = field(default_factory=list, init=False)
    _gaps_observed: list[int] = field(default_factory=list, init=False)
    _gaps_advanced: list[int] = field(default_factory=list, init=False)
    _publisher_seen: dict[str, int] = field(default_factory=dict, init=False)
    _gaps_truncated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.sink is None:
            self.sink = logging_sink()

    # ── one window ────────────────────────────────────────────────────────────

    def observe(self, window: Window) -> Verdict:
        """Run both gates over one window and update the run state."""
        now = int(self.clock())
        if self._last_progress_ns is None:
            self._last_progress_ns = now
        self._record_observation(now)
        self.seq += 1
        # Polled on **every** window, not only the silent ones. A verdict needs two
        # readings, and a beacon first read at the moment things go quiet has no
        # previous reading to compare against — the monitor would be blind for exactly
        # one window, which is the window it most needs to see.
        publisher = self._poll_beacon(now)

        if window.offline_ts.size == 0 and window.online_ts.size == 0:
            return self._emit(
                self._silence(
                    now, "the window carried no rows on either side", publisher=publisher
                )
            )

        parity = aligned_feature_parity(
            window.online,
            window.offline,
            online_ts=window.online_ts,
            offline_ts=window.offline_ts,
            columns=self.config.columns,
            scope=self.config.scope,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )
        if parity.n_rows == 0:
            # Rows arrived and none of them lined up. That is louder than an empty
            # window, not quieter: two sides that both produced data and share no
            # event time is a join that never happened, and `aligned_feature_parity`
            # has already named the constant-offset case if that is what it is.
            return self._emit(
                self._silence(
                    now,
                    "rows arrived on at least one side and not one pair aligned",
                    publisher=publisher,
                    parity=parity,
                )
            )

        # The gap this run will be judged on. Recorded here — at the *one* place the
        # progress clock advances — rather than reconstructed later, because the whole
        # question the deadline asks is "how long since the last window that compared
        # something", and any second answer to it could disagree with this one.
        if self._compared_at_ns is not None:
            self._record_gap(self._gaps_compared, now - self._compared_at_ns)
        self._compared_at_ns = now
        self._last_progress_ns = now
        self.n_compared += parity.n_rows
        self.max_abs_diff = max(self.max_abs_diff, parity.max_abs_diff)

        level, reasons = Level.OK, []
        if parity.n_mismatched or parity.n_nan_mismatched:
            level = Level.ALARM
            reasons.append(f"feature parity: {parity.summary().splitlines()[0]}")
            reasons.extend(f"  {line.strip()}" for line in parity.summary().splitlines()[1:4])
        if parity.coverage is not None and not parity.coverage.complete:
            level = Level.ALARM
            reasons.append(f"coverage: {parity.coverage.describe()}")

        drift = self._drift(window, reasons)
        if drift is not None:
            level = max(level, self._drift_level(drift, reasons))

        return self._emit(
            Verdict(
                seq=self.seq,
                level=level,
                reasons=tuple(reasons),
                n_compared=parity.n_rows,
                first_ts=int(window.offline_ts.min()),
                last_ts=int(window.offline_ts.max()),
                parity=parity,
                drift=drift,
                coverage=parity.coverage,
                publisher=publisher,
            )
        )

    def heartbeat(self) -> Verdict:
        """Ask for a verdict when no window arrived at all.

        A caller drives this from its own idle branch. Without it a monitor whose
        source has stopped emits nothing, and a monitor emitting nothing is
        indistinguishable from a monitor with nothing to report — which is the
        ambiguity :mod:`axon.live.liveness` exists to remove at the other end of the
        boundary, in the other direction.
        """
        now = int(self.clock())
        if self._last_progress_ns is None:
            self._last_progress_ns = now
        self._record_observation(now)
        self.seq += 1
        return self._emit(
            self._silence(now, "no window arrived", publisher=self._poll_beacon(now))
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _record_gap(self, series: list[int], gap_ns: int) -> None:
        """Append one wall-clock gap, up to :data:`MAX_GAP_SAMPLES`.

        The cap is checked per series rather than in total so that a run whose
        observation series filled up still records the far sparser compared-window
        gaps, which are the ones the deadline is actually argued from.
        """
        if len(series) >= MAX_GAP_SAMPLES:
            self._gaps_truncated = True
            return
        series.append(int(gap_ns))

    def _record_observation(self, now: int) -> None:
        """The resolution series: how often this monitor was asked anything at all.

        Called from :meth:`observe` and :meth:`heartbeat` alike, before either does
        any work, because what it measures is the *caller's* cadence and not the
        monitor's. Without it, a wide gap between two compared windows is ambiguous
        between a quiet feed and a driver that stopped asking, and those two want
        opposite responses from whoever reads the report.
        """
        if self._observed_at_ns is not None:
            self._record_gap(self._gaps_observed, now - self._observed_at_ns)
        self._observed_at_ns = now

    def _poll_beacon(self, now: int) -> PublisherState:
        """One reading of the market-data beacon, plus what it adds to the evidence.

        The states are counted only when a probe was actually wired. An unwired monitor
        returns :data:`~axon.parity.beacon.UNKNOWN_PUBLISHER` on every call, and tallying
        that would fill the report with a diagnosis nobody asked for and nobody got —
        :attr:`SilenceEvidence.beacon_wired` is the field that says so, once.
        """
        if self.beacon is None:
            return UNKNOWN_PUBLISHER
        state = self._read_publisher()
        self._publisher_seen[state.state.value] = self._publisher_seen.get(state.state.value, 0) + 1
        # A *record* crossed the boundary between the last reading and this one. The
        # gap is therefore an upper bound quantised by how often this was called — the
        # beacon carries counters, not a clock per record — which is why
        # `SilenceEvidence` prints the observation cadence beside it rather than
        # letting a reader assume the two are the same measurement.
        if state.published or state.bars_published:
            if self._advanced_at_ns is not None:
                self._record_gap(self._gaps_advanced, now - self._advanced_at_ns)
            self._advanced_at_ns = now
        return state

    def _read_publisher(self) -> PublisherState:
        """One reading of the market-data beacon, against the previous one.

        Everything here degrades to :attr:`Publisher.UNKNOWN`, which is the state in
        which the monitor behaves exactly as it did before the beacon existed. That
        includes the probe *raising*: a live monitor must alarm rather than die
        (decision 3), so a beacon that cannot be read becomes a sentence in the report
        rather than an exception out of the loop — and it is deliberately not an alarm
        of its own, because escalating on a file that is not there yet would fire on
        every startup race.
        """
        try:
            snapshot = self.beacon()
        except Exception as exc:  # noqa: BLE001 - see the docstring
            self._last_beacon = None
            return PublisherState(
                Publisher.UNKNOWN, f"the market-data beacon could not be read: {exc}"
            )
        if snapshot is None:
            self._last_beacon = None
            return PublisherState(
                Publisher.UNKNOWN,
                "no market-data beacon at the configured path yet: nothing has created one, "
                "so silence can only be timed",
            )
        previous, self._last_beacon = self._last_beacon, snapshot
        if previous is None:
            return PublisherState(
                Publisher.UNKNOWN,
                f"first reading of the market-data beacon (beat {snapshot.beats}, "
                f"pid {snapshot.pid}); a verdict about a counter needs two",
            )
        return publisher_state(previous, snapshot)

    def _silence(
        self,
        now: int,
        why: str,
        *,
        publisher: PublisherState = UNKNOWN_PUBLISHER,
        parity: FeatureParityReport | None = None,
    ) -> Verdict:
        quiet_for = now - (self._last_progress_ns or now)
        reasons = [
            f"{why}; nothing has been compared for {quiet_for / 1e9:.1f}s "
            f"(deadline {self.config.silence_after_ns / 1e9:.1f}s)",
            "a window that compared nothing is not a window that agreed",
        ]
        owed = 0 if parity is None or parity.coverage is None else parity.coverage.n_offline
        level = Level.SILENT
        if self.config.scope == "declared" and owed:
            # SILENT exists to express one specific thing: *I cannot tell whether
            # there was anything to compare.* A declared scope has already answered
            # that — the window's offline side is what the serving path owed — so this
            # is not silence, it is `owed` rows the path did not produce, and the
            # deadline has nothing left to resolve. Waiting sixty seconds to say so
            # would be waiting for information already in hand.
            level = Level.ALARM
            reasons.append(
                f"this window declared {owed} owed row(s) and compared none of them; "
                "under a declared scope that is a blind serving path, not a quiet one"
            )

        # ── what the publisher's own beacon says, if there is one (decision 5) ──
        # Reported whenever a probe is wired, including when it came back UNKNOWN: "the
        # beacon could not be read" and "no beacon at all" are different sentences and
        # an operator needs the first one. Only the no-probe singleton stays silent,
        # because a monitor nobody gave a beacon should not print a line about it.
        if publisher is not UNKNOWN_PUBLISHER:
            reasons.append(f"beacon: {publisher.reason}")
        if publisher.state in (Publisher.DEAD, Publisher.STOPPED):
            # No deadline. The thing sixty seconds was there to establish is already
            # established, and waiting would be waiting for information in hand — the
            # same sentence §1b of ADR-0030 applies to a declared scope.
            level = Level.ALARM
        elif publisher.state is Publisher.QUIET and level is not Level.ALARM:
            # The one downgrade in this module, and it is earned rather than lenient.
            # SILENT means precisely "I cannot tell whether there was anything to
            # compare"; a beat that is advancing over a market that did not move has
            # just told us. Left at SILENT, a quiet hour would escalate on the deadline
            # and teach an operator to ignore the level — which is exactly the argument
            # `drift_ceiling` is capped by, one floor down.
            level = Level.WARN

        # A proven-alive, proven-quiet publisher suspends the deadline; every other
        # state, including "no beacon at all", keeps it.
        if quiet_for >= self.config.silence_after_ns:
            if publisher.state is Publisher.QUIET:
                reasons.append(
                    f"the {self.config.silence_after_ns / 1e9:.1f}s silence deadline passed and "
                    "was not escalated: the beacon proves the publisher alive and the market "
                    "quiet, and there is no fault here to alarm on"
                )
            else:
                level = Level.ALARM
                reasons.append(
                    "past the silence deadline: treat this as a dead feed rather than a "
                    "quiet market until something says otherwise"
                )
        if parity is not None:
            reasons.extend(line.strip() for line in parity.summary().splitlines()[1:])
        return Verdict(
            seq=self.seq,
            level=level,
            reasons=tuple(reasons),
            n_compared=0,
            first_ts=None,
            last_ts=None,
            parity=parity,
            coverage=None if parity is None else parity.coverage,
            silent_for_ns=quiet_for,
            publisher=publisher,
        )

    def _drift(self, window: Window, reasons: list[str]) -> DriftReport | None:
        if not self.config.drift_enabled:
            return None
        live = np.asarray(window.online, dtype=np.float64)
        if live.shape[0] < self.config.min_drift_rows:
            # Reported, not silently skipped: "drift not measured" and "drift stable"
            # are opposite statements and only one of them is true here.
            reasons.append(
                f"drift not measured: {live.shape[0]} live row(s) is under the "
                f"{self.config.min_drift_rows}-row floor, and PSI below it is sample noise"
            )
            return None
        return drift_report(
            self.config.reference,
            live,
            columns=self.config.columns,
            nan_rate_tol=self.config.nan_rate_tol,
            binnings=self.config.binnings,
        )

    def _drift_level(self, drift: DriftReport, reasons: list[str]) -> Level:
        worst = max((f.psi for f in drift.features), default=0.0)
        if np.isfinite(worst):
            self.worst_psi = max(self.worst_psi, float(worst))
        else:
            self.worst_psi = float("inf")
        if drift.passed:
            self._consecutive_drift = 0
            return Level.OK
        self._consecutive_drift += 1
        offenders = ", ".join(
            f"{f.name} psi={f.psi:.4f} [{f.band}]" for f in drift.ranked()[:3] if f.band != "stable"
        )
        nan_regressions = ", ".join(f.name for f in drift.nan_regressions())
        if nan_regressions:
            offenders = f"{offenders}; started emitting NaN: {nan_regressions}".lstrip("; ")
        if self._consecutive_drift < self.config.confirm_windows:
            reasons.append(
                f"drift (window {self._consecutive_drift} of "
                f"{self.config.confirm_windows} before it counts): {offenders}"
            )
            return min(Level.WARN, self.config.drift_ceiling)
        reasons.append(
            f"drift confirmed over {self._consecutive_drift} consecutive window(s): {offenders}"
        )
        # Capped rather than reported at full volume: see `MonitorConfig.drift_ceiling`
        # for the measurement that says why. A moved market is not a bug, and the one
        # finding here that *is* a bug must stay the loudest thing on the page.
        return min(Level.ALARM, self.config.drift_ceiling)

    def _emit(self, verdict: Verdict) -> Verdict:
        self.worst = max(self.worst, verdict.level)
        if verdict.level is Level.OK:
            self.n_ok += 1
        if verdict.level is Level.ALARM:
            self.n_alarm += 1
        if verdict.level is Level.SILENT:
            self.n_silent += 1
        if verdict.level is not Level.OK:
            self._reasons.extend(f"window {verdict.seq}: {r}" for r in verdict.reasons[:2])
        # Before the sink, so an operator watching a live run sees the window's own line
        # above the alarm that window raised rather than under it.
        if self.tap is not None:
            self.tap(verdict)
        if self.sink is not None and verdict.level is not Level.OK:
            self.sink(verdict)
        return verdict

    def silence_evidence(self) -> SilenceEvidence:
        """The gap measurements so far, frozen.

        Separate from :meth:`report` so a long-running driver can print the evidence
        periodically without pretending the run has ended — and so that the lists stay
        private: a caller holding the report cannot append a sample to the record it
        describes.
        """
        return SilenceEvidence(
            silence_after_ns=self.config.silence_after_ns,
            beacon_wired=self.beacon is not None,
            between_compared_windows=GapStats.of(self._gaps_compared),
            between_observations=GapStats.of(self._gaps_observed),
            between_ring_advances=GapStats.of(self._gaps_advanced),
            publisher_states=tuple(
                sorted(self._publisher_seen.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            truncated=self._gaps_truncated,
        )

    def report(self) -> MonitorReport:
        """The run so far, as a gate report."""
        return MonitorReport(
            windows=self.seq,
            worst=self.worst,
            n_compared=self.n_compared,
            n_ok=self.n_ok,
            n_alarm=self.n_alarm,
            n_silent=self.n_silent,
            max_abs_diff=self.max_abs_diff,
            worst_psi=self.worst_psi,
            reasons=tuple(self._reasons[:20]),
            silence=self.silence_evidence(),
        )


def windows_from_matrices(
    online,
    offline,
    *,
    online_ts,
    offline_ts,
    size: int,
) -> Iterator[Window]:
    """Cut two whole matrices into event-time windows the monitor can eat.

    The cut is on ``offline_ts``, because the offline recompute is the reference and
    therefore the thing that defines a window's span. Cutting on the online side
    would move the window boundaries whenever the online side went blind, which is
    the one case the monitor has to be able to see — the window would shrink to fit
    the damage and report full coverage over it.

    Boundaries are half-open on the left and closed only at the very end, so **every
    online row lands in exactly one window**. An online row falling through the gap
    between two windows would be dropped by the windower itself, which is the same
    invisible-denominator bug one layer further out.
    """
    if size < 1:
        raise ValueError(f"window size must be at least 1, got {size}")
    off_ts = np.asarray(offline_ts, dtype=np.int64)
    on_ts = np.asarray(online_ts, dtype=np.int64)
    offline = np.asarray(offline)
    online = np.asarray(online)
    starts = list(range(0, off_ts.size, size))
    for i, start in enumerate(starts):
        stop = min(start + size, off_ts.size)
        first = int(off_ts[start]) if i else np.iinfo(np.int64).min
        last = int(off_ts[starts[i + 1]]) if i + 1 < len(starts) else None
        take = on_ts >= first
        if last is not None:
            take &= on_ts < last
        yield Window(
            online=online[take],
            offline=offline[start:stop],
            online_ts=on_ts[take],
            offline_ts=off_ts[start:stop],
        )


def run_monitor(
    windows: Iterable[Window],
    config: MonitorConfig,
    *,
    sink: AlarmSink | None = None,
    clock: Clock = time.time_ns,
    beacon: BeaconProbe | None = None,
    tap: VerdictTap | None = None,
) -> MonitorReport:
    """Drive the monitor over a finite source and return what it saw.

    Finite because this is the offline driver — a recording, a fixture history, a
    replayed session. A live driver is the same loop with a source that blocks, plus
    a :meth:`ParityMonitor.heartbeat` on its idle branch **and a ``beacon`` probe**;
    that loop belongs to whatever owns the connection, not here, because nothing in
    this module may open one. ``scripts/sessions/parity_live.py`` is that loop for a
    Hyperliquid session, and ``docs/adr/0030-live-parity-monitor-and-the-coverage-denominator.md``
    is what it is for.
    """
    monitor = ParityMonitor(config, sink=sink, clock=clock, beacon=beacon, tap=tap)
    for window in windows:
        monitor.observe(window)
    return monitor.report()


# ── a way to actually run it ─────────────────────────────────────────────────


def _perp_bar_windows(
    coin: str, interval: str, *, blind_every: int
) -> tuple[Iterable[Window], MonitorConfig]:
    """The frozen candle fixtures, driven through the real serving path.

    Imported lazily and from inside the function on purpose: this module must stay
    importable in a bare numpy environment (``axon.strategies`` reaches for an ML
    stack), and it must not grow a dependency on a package another workstream is
    editing just to be importable.

    ``blind_every`` drops every *n*-th online row. It exists so an operator can
    watch the monitor fail on demand: a detector nobody has seen fire is a
    decoration, which is the same argument ADR-0016 makes for its leaky-feature test.
    """
    from axon.features import finite_rows
    from axon.strategies import PERP_BAR_V1, PerpBar, PerpBarParams
    from axon.strategies.data import fixture_candles
    from axon.strategies.training import replay_bars

    class _Flat:
        """A predictor, because the serving path needs one. The monitor compares
        feature vectors and never looks at a score, so a constant is honest here."""

        def predict(self, x):
            return np.full(len(np.asarray(x)), 0.5)

    candles = fixture_candles(coin, interval)
    online_ts, online, _ = replay_bars(
        PerpBar(PerpBarParams(symbol_id=0), _Flat()), candles, symbol_id=0
    )
    offline = PERP_BAR_V1.compute(candles.feature_inputs())
    usable = finite_rows(offline)
    offline, offline_ts = offline[usable], candles.ts_event[usable]
    if blind_every > 1:
        keep = np.arange(online_ts.size) % blind_every != 0
        online_ts, online = online_ts[keep], online[keep]
    config = MonitorConfig(columns=PERP_BAR_V1.columns)
    windows = windows_from_matrices(
        online, offline, online_ts=online_ts, offline_ts=offline_ts, size=128
    )
    return windows, config


def main(argv: Sequence[str] | None = None) -> int:
    """Run the monitor over an offline source and print what it saw.

    Offline sources only, and deliberately: nothing here opens a socket. A live
    session drives :class:`ParityMonitor` from whatever owns its connection.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m axon.parity",
        description="Run the parity gates continuously over an offline source.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--npz",
        metavar="PATH",
        help="an .npz holding online, offline, online_ts, offline_ts",
    )
    source.add_argument(
        "--perp-bar",
        metavar="COIN",
        help="the frozen candle fixture for COIN, through the real serving path",
    )
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--window", type=int, default=128, help="rows per window")
    parser.add_argument(
        "--blind-every",
        type=int,
        default=0,
        help="drop every n-th online row, to watch the monitor fail on demand",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.perp_bar:
        windows, config = _perp_bar_windows(
            args.perp_bar, args.interval, blind_every=args.blind_every
        )
    else:
        data = np.load(args.npz)
        config = MonitorConfig(columns=tuple(str(c) for c in data["columns"]))
        windows = windows_from_matrices(
            data["online"],
            data["offline"],
            online_ts=data["online_ts"],
            offline_ts=data["offline_ts"],
            size=args.window,
        )
    report = run_monitor(windows, config)
    print(report.summary())
    return 0 if report.passed else 1


__all__ = [
    "AlarmSink",
    "BeaconProbe",
    "Clock",
    "DEFAULT_CONFIRM_WINDOWS",
    "DEFAULT_MIN_DRIFT_ROWS",
    "DEFAULT_SILENCE_AFTER_NS",
    "MAX_GAP_SAMPLES",
    "GapStats",
    "Level",
    "MonitorConfig",
    "MonitorReport",
    "ParityError",
    "ParityMonitor",
    "Publisher",
    "PublisherState",
    "SilenceEvidence",
    "Verdict",
    "VerdictTap",
    "Window",
    "collecting_sink",
    "logging_sink",
    "main",
    "run_monitor",
    "windows_from_matrices",
]

