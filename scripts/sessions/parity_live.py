#!/usr/bin/env python3
"""Point the live parity monitor at a session in flight, and measure its own deadline.

ADR-0030 §7 is explicit that ``axon.parity`` opens nothing: the monitor consumes
:class:`~axon.parity.monitor.Window` objects, and *"a live driver is the same loop with
a source that blocks plus a ``heartbeat()`` on its idle branch, and it belongs to
whatever owns the connection."* This file is that driver for a Hyperliquid session. It
lives under ``scripts/sessions/`` and not in the package for exactly that reason — the
package's promise that nothing in it can reach a socket is worth more than the
convenience of putting the loop next to the thing it drives.

``docs/08`` records the gap this closes, and it is worth quoting because it also bounds
what this script may claim: the monitor *"has been run against a frozen fixture, a
cached mainnet history and synthetic windows, and against no live session at all — its
60 s silence deadline is a stated starting value, not a measurement."* This is the
harness that points it at a real session. **It is not the observation.** Until an
operator has run ``--bar-ring`` against a live testnet session and pasted the transcript
into a runbook, nothing here has been run against a venue and nothing here may be called
proven.

Six decisions carry it, and each has a comfortable wrong answer.

**1. The comparison is not reimplemented, and neither is the loop that runs it.**
:class:`~axon.strategies.shadow.ShadowTrader` already drives the real serving path one
closed bar at a time, builds each window's reference as exactly the rows that path owed,
declares that scope to the monitor, and heartbeats on the bar clock rather than on the
poll clock. Re-deriving any of that here would be a second implementation of the thing
whose single implementation is the entire point — and a second one that would have to
relearn, live, the four false silence alarms the first one already paid for.

**2. The beacon is wired onto the monitor the shadow trader built, rather than into a
monitor of our own.** ``ParityMonitor.beacon`` is a public field precisely so the caller
that owns the session can supply the probe (ADR-0030: *"the monitor takes a callable
returning a snapshot… the caller that owns the session owns the mapping"*).
:class:`~axon.parity.beacon.MdBeaconReader` **is** such a callable, one reader is shared
by every trader, and nothing here parses a byte of the sidecar.

**3. Silence without a beacon is reported as uncategorised, loudly, in three places.**
At startup, on every silent window, and in the final summary. A run that degrades
quietly is worse than one that refuses to start: with a probe, ``SILENT`` resolves into
``STARVED`` (the process is beating and nothing is reaching it — a healthy, permanently
silent socket looks exactly like this and nothing else in the system can see it) or
``PUBLISHING`` (the ring is moving and whatever compared nothing is downstream of it).
Without one, every silence is the same shrug, and the deadline is the only evidence.

**4. The network is behind a flag and is never the default.** With no arguments this
runs the committed candle fixture through the same code path, and the only mode that
touches a ring is ``--bar-ring``. Nothing in the default gate reaches this file at all.

**5. A wrong ``--symbol-id`` and a dead feed are the same silence, so the arrivals are
tallied.** The bar ring carries the *venue's* asset index — on testnet BTC is **3**, not
0 — and a run pointed at the wrong one attaches cleanly, reads happily, hands its trader
nothing, and reports a monitor that compared nothing for an hour. Every ``symbol_id``
that arrives is counted and printed beside the ones traders were built for, so the
report names that mistake instead of presenting it as a dead publisher.

**6. Two wall clocks are read, and both are the named exception.** The silence deadline
(the absence of an event has no event time) and ``--duration`` (a bound on how long the
operator wants to be here, which orders nothing). Everything that *orders* — the
windows, the alignment, the bar cadence — is on ``ts_event``, inside the shadow trader
and the monitor, and this file does not touch it.

Typical use — see ``docs/adr/0030-live-parity-monitor-and-the-coverage-denominator.md``,
and prefer
``scripts/sessions/parity-live.sh`` which starts the session too::

    # offline, and the default: no session, no ring, no socket
    python scripts/sessions/parity_live.py --fixture BTC --interval 1m

    # live, against a running `axon` session with [md_ring] enabled
    python scripts/sessions/parity_live.py --bar-ring /dev/shm/parity-live-md.ring \\
        --symbol-id 3 --interval 1m --duration 2700 --attest-venue
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    # So the script runs as `python scripts/sessions/parity_live.py` from anywhere,
    # rather than only under a PYTHONPATH somebody remembered to export. A harness an
    # operator can invoke wrongly is a harness that gets invoked wrongly at 03:00.
    sys.path.insert(0, str(ROOT / "python"))

from axon.parity.beacon import MdBeaconReader, beacon_path  # noqa: E402
from axon.parity.monitor import Level, SilenceEvidence, Verdict  # noqa: E402
from axon.strategy.events import Bar  # noqa: E402

#: How often the poll loop asks the source for bars, in seconds. The shadow trader
#: rate-limits its own heartbeat to once a second *and* gates it on a bar being overdue,
#: so this is only how promptly a bar is picked up, never how often a verdict is asked
#: for — the distinction that stopped 296 SILENT verdicts in 160 seconds of a healthy m1
#: feed (ADR-0029 §7).
POLL_S = 0.05

#: Bars per parity window on the live path. Smaller than
#: :data:`axon.strategies.shadow.DEFAULT_WINDOW_BARS` (64) because a monitor is watched
#: while it runs: at one bar a minute, 64 bars is the first verdict an hour in, and an
#: operator who has seen nothing for an hour cannot tell a warming harness from a broken
#: one. Sixteen minutes to the first window is still long enough that the alignment has
#: something to align.
DEFAULT_WINDOW_BARS = 16


# ── printing ─────────────────────────────────────────────────────────────────


def banner(*lines: str) -> None:
    """A boxed, unmissable block. Used only where a run must not be misread.

    Reserved for the two facts that change what a transcript *means*: that no beacon
    was wired (so every silence in it is uncategorised), and that the run was not
    attested against a venue (so it is a rehearsal and not an observation). Everything
    else prints as an ordinary line, because a banner that appears on every run is
    furniture.
    """
    width = 78
    print("!" * width)
    for line in lines:
        print(f"!! {line}"[:width])
    print("!" * width)


def _hhmmss(ns: int) -> str:
    """A wall-clock time of day for the transcript, and nothing orders on it.

    The verdicts themselves carry ``ts_event``; this is here so that a transcript read
    hours later can be lined up against the session log and the venue's own charts,
    which is a *reading* concern and not a sequencing one.
    """
    return time.strftime("%H:%M:%S", time.localtime(ns / 1e9))


def print_verdict(symbol_id: int, verdict: Verdict) -> None:
    """One line per window, whatever it said — plus the reasons when it said something.

    Every window and not only the bad ones: a harness that printed only alarms is
    indistinguishable from a harness that has stopped, which is the same ambiguity the
    monitor exists to remove one level down. The reasons are held back on ``OK`` because
    a healthy m1 session prints one of these every sixteen minutes for hours, and a
    transcript nobody can scan is a transcript nobody reads.
    """
    head = verdict.summary().splitlines()[0]
    print(f"{_hhmmss(time.time_ns())} sym {symbol_id} {head}", flush=True)
    if verdict.level is not Level.OK:
        for reason in verdict.reasons:
            print(f"    {reason}", flush=True)


# ── sources ──────────────────────────────────────────────────────────────────


class BoundedSource:
    """Any :class:`~axon.strategies.shadow.BarSource`, bounded and counted.

    Two jobs, both of which have to sit *between* the reader and the traders:

    * **the wall-clock bound.** ``--duration`` is the operator's answer to "how long am
      I here for", which is not a property of the feed and cannot be expressed in event
      time — a dead feed's event clock never reaches any deadline stated in it. It is
      the second and last wall clock in this harness, and it orders nothing.
    * **the arrivals tally.** Every ``symbol_id`` that came off the source is counted,
      including the ones no trader was built for. A run pointed at the wrong asset index
      attaches, reads, and hands its trader nothing, which is *character for character*
      the transcript of a dead publisher; the tally is the only thing that tells them
      apart, and it costs one dict update per bar.

    ``exhausted`` is what ends the run rather than a ``break`` in a private loop, so the
    shared :func:`~axon.strategies.shadow.drive` still owns the shape of the loop —
    including the idle branch that heartbeats, which is the branch a live run spends
    almost all of its time in.
    """

    def __init__(self, inner, *, duration_s: float | None = None) -> None:
        self.inner = inner
        self.health = inner.health
        self.duration_s = duration_s
        self._deadline = None if duration_s is None else time.monotonic() + duration_s
        self.arrivals: dict[int, int] = {}
        self.expired = False

    @property
    def exhausted(self) -> bool:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self.expired = True
        return self.expired or self.inner.exhausted

    def describe(self) -> str:
        bound = "" if self.duration_s is None else f", bounded at {self.duration_s:.0f}s"
        return f"{self.inner.describe()}{bound}"

    def poll(self) -> list[Bar]:
        # The deadline is checked *here* and not only on the idle branch, so it holds
        # whatever the feed is doing. `drive()` consults `exhausted` only when a poll
        # came back empty, which on a quiet m1 tape is every poll and on a busy one is
        # none of them — a bound that only binds when the feed is quiet is not a bound.
        if self.exhausted:
            return []
        bars = self.inner.poll()
        for bar in bars:
            self.arrivals[bar.symbol_id] = self.arrivals.get(bar.symbol_id, 0) + 1
        return bars

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


class RecordedBarSource:
    """A CSV of the ring's own bar records, replayed as closed bars — offline.

    The format is the one :mod:`axon.strategies.shadow` writes with ``--bars-out``:
    ``ts_event,open,high,low,close,volume`` in the wire's fixed-point integers,
    undivided. Replaying *those* rather than a candle history is what makes this an
    honest rehearsal of a live run: the numbers are the bytes the ring carried, in the
    units it carried them in, so nothing between the venue and the serving path is
    re-derived here.

    What it still is not, and the report says so through
    :attr:`~axon.strategies.shadow.ShadowReport.venue_attested`: a recording. A file
    cannot go quiet, cannot drop a record, and cannot starve — which is precisely the
    set of faults a live run exists to expose.
    """

    def __init__(self, path: str, *, symbol_id: int, interval: str, chunk: int = 16) -> None:
        from axon.strategies.shadow import FeedHealth

        rows: list[tuple[int, ...]] = []
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            expected = ["ts_event", "open", "high", "low", "close", "volume"]
            if header != expected:
                raise SystemExit(f"{path}: header is {header}, expected {expected}")
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(tuple(int(v) for v in line.split(",")))
        if not rows:
            raise SystemExit(f"{path}: no rows — a source with nothing in it is not a rehearsal")
        self.path = path
        self.rows = rows
        self.symbol_id = int(symbol_id)
        self.interval = interval
        self.chunk = int(chunk)
        self.health = FeedHealth()
        self._at = 0
        # The venue's own holes, counted the way the bar ring flags them, so a recording
        # and a live feed report the same fault under the same name.
        from axon.strategies.data import INTERVAL_MS

        step = INTERVAL_MS[interval] * 1_000_000
        self._gap_at = {
            i for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][0] != step
        }

    @property
    def exhausted(self) -> bool:
        return self._at >= len(self.rows)

    def describe(self) -> str:
        return (
            f"recorded bar ring: {self.path}, {len(self.rows)} closed bars at "
            f"{self.interval} — a REPLAY, not a live session"
        )

    def poll(self) -> list[Bar]:
        stop = min(self._at + self.chunk, len(self.rows))
        out = []
        for i in range(self._at, stop):
            if i == 0:
                self.health.first_bars += 1
            elif i in self._gap_at:
                self.health.feed_gaps += 1
            self.health.bars += 1
            ts, o, h, low, c, v = self.rows[i]
            out.append(
                Bar(
                    symbol_id=self.symbol_id,
                    ts_event=ts,
                    open=o,
                    high=h,
                    low=low,
                    close=c,
                    volume=v,
                )
            )
        self._at = stop
        return out

    def close(self) -> None:
        pass


# ── the beacon ───────────────────────────────────────────────────────────────


def created_and_never_beaten(first, second) -> bool:
    """Whether two readings say *"this file was stamped and nothing ever beat it"*.

    A predicate rather than an inline condition so it can be tested without a file, a
    session or a sleep — and because it is the single decision in this harness that can
    manufacture a false alarm if it is wrong in either direction. Wrong one way and a
    healthy session alarms on every silent window; wrong the other and a genuinely dead
    publisher is reported as no beacon at all.

    ``beats == 0`` in **both** readings is the whole test, and the reason it is safe is
    the cadence on the other side: ``session.core_poll_us`` is 500, so a pass loop that
    is wired at all beats thousands of times in a settle window measured in seconds.
    ``stopped`` is excluded because a session that created its beacon and shut down
    before its first beat is a *stopped* publisher, and that is a diagnosis the probe
    should be allowed to make rather than a wiring fault to hide.
    """
    return (
        first is not None
        and second is not None
        and first.beats == 0
        and second.beats == 0
        and not second.stopped
    )


def wire_beacon(path: str | None, *, settle_s: float = 2.0):
    """Open a beacon probe, or say — unmissably — why silence will be uncategorised.

    The reader is deliberately **not** required to succeed. Its ``read`` returns ``None``
    until a publisher has created *and* stamped the file, because a monitor is wired
    before the session it watches starts and "not there yet" is the ordinary startup
    race rather than a fault. So a probe is returned even when nothing is at the path
    yet; what is decided here is only whether the operator is *told* that today's run
    may be blind, and that is decided by looking rather than by assuming.

    **And there is one configuration this must refuse to wire, because wiring it would
    manufacture an alarm.** ``MdBeacon::create`` stamps the header — magic, pid, flags
    ``RUNNING`` — and leaves ``beats`` at 0; the counter is advanced by the *core's pass
    loop*, which is a separate piece of wiring. A session whose rings are enabled and
    whose pass loop does not beat therefore leaves a perfectly readable beacon frozen at
    beat 0 for as long as it runs, and two readings of that are ``DEAD``: *"its pass
    loop is not running, so this is a dead publisher and not a quiet market"* — which is
    a true sentence about the beacon and a false one about the feed. Under it, every
    silent window would alarm immediately, with no deadline left to wait out, on a
    session delivering bars perfectly. That is the shape of false alarm that gets a real
    check deleted, and four of them were found on a healthy feed in one session already.

    So the probe is settled before it is trusted: read, wait, read again. A beat that is
    still **zero** after ``settle_s`` is a publisher that was never wired to this file —
    the core polls at 500 µs, so a live pass loop beats thousands of times in that
    window — and the honest report is "no beacon", not "dead publisher". A beat frozen
    at any value *above* zero is a different claim entirely: something beat this file
    and then stopped, which is exactly what the probe exists to detect, so that one is
    wired.

    Nothing in this function parses a beacon byte. That is :mod:`axon.parity.beacon`'s
    job and there is exactly one implementation of it.
    """
    if path is None:
        banner(
            "NO MARKET-DATA BEACON WAS WIRED (--no-beacon, or no ring path to derive",
            "one from). Every SILENT verdict in this run will be UNCATEGORISED: a",
            "quiet publisher and a dead one write the same nothing under OnChange,",
            "and the silence deadline will be the only evidence available.",
        )
        return None, "no probe was wired"

    reader = MdBeaconReader(path)
    try:
        first = reader.read()
    except Exception as exc:  # noqa: BLE001 - a bad beacon must not stop the harness
        banner(
            f"THE BEACON AT {path} COULD NOT BE READ: {exc}",
            "The probe stays wired in case it starts working, but treat every SILENT",
            "verdict in this run as UNCATEGORISED until the report says otherwise.",
        )
        return reader, f"unreadable at start: {exc}"
    if first is None:
        # Not a banner: this is the ordinary case when the harness is started before or
        # alongside the session, and a banner on every launch is a banner nobody reads.
        # The final report escalates it if the probe never *did* answer.
        print(
            f"beacon    : {path} — nothing has stamped it yet. That is the ordinary "
            "startup race; the probe stays wired and the report will say whether it "
            "ever answered.",
            flush=True,
        )
        return reader, "wired, unstamped at start"

    # The settle. A wall-clock sleep, and the third and last one in this harness: what
    # is being measured is whether a counter moves in real time, which no event clock
    # can express — the whole condition is that events may not be arriving at all.
    if settle_s > 0:
        time.sleep(settle_s)
        try:
            second = reader.read()
        except Exception:  # noqa: BLE001 - fall through to the wired path and report it
            second = None
        if created_and_never_beaten(first, second):
            banner(
                f"THE BEACON AT {path} EXISTS AND IS NOT BEING BEATEN (beat 0 after",
                f"{settle_s:.0f}s). Its header was stamped by MdBeacon::create and the",
                "core's pass loop is not advancing it — that is the runtime wiring, not",
                "a dead feed. The probe is NOT wired, deliberately: two readings of a",
                "frozen zero read as 'dead publisher' and would alarm on every silent",
                "window of a session delivering bars perfectly. Silence in this run is",
                "therefore UNCATEGORISED and the deadline is the only evidence.",
            )
            reader.close()
            return None, "refused: created and never beaten (beat 0)"
        if second is not None and second.beats == first.beats:
            # Frozen *above* zero: something beat this file and no longer does, which is
            # the fault the probe exists to name rather than a wiring mistake to hide.
            how = "stopped on purpose" if second.stopped else "frozen"
            print(
                f"beacon    : {path} — pid {second.pid}, beat {second.beats}, {how} "
                f"after {settle_s:.0f}s. Wired: something beat this file and no longer "
                "does, which is precisely what the probe is for.",
                flush=True,
            )
            return reader, f"wired, {how} at start (beat {second.beats})"
        if second is not None:
            print(
                f"beacon    : {path} — pid {second.pid}, beat {first.beats} -> "
                f"{second.beats} in {settle_s:.0f}s. Beating. Silence in this run "
                "resolves into a cause.",
                flush=True,
            )
            return reader, f"wired, beating (+{second.beats - first.beats} in {settle_s:.0f}s)"

    print(
        f"beacon    : {path} — pid {first.pid}, beat {first.beats}, unsettled.",
        flush=True,
    )
    return reader, f"wired, unsettled (pid {first.pid}, beat {first.beats})"


# ── the run ──────────────────────────────────────────────────────────────────


def evidence_json(evidence: SilenceEvidence) -> dict:
    """The gap measurements as data, so an argument about the deadline outlives a scroll.

    Quantiles and the maximum rather than the raw samples: the tail is the whole of what
    a deadline is argued from, and a file that reproduced every sample would still not
    let anyone else recompute anything, because the samples are of *this* run's cadence
    and mean nothing outside the report that names it.
    """
    out = asdict(evidence)
    out["diagnosed"] = evidence.diagnosed
    return out


def run(args: argparse.Namespace) -> int:
    from axon.strategies.shadow import (
        HistoryBarSource,
        RingBarSource,
        ShadowTrader,
        drive,
    )
    from axon.strategies.perp_bar import PERP_BAR_V1

    symbol_ids: list[int] = args.symbol_id or ([3] if args.bar_ring else [0])

    # ── the source ────────────────────────────────────────────────────────────
    if args.bar_ring:
        # Attached UNFILTERED and each trader selects its own, because a ring's read
        # cursor lives in the ring's own header: two consumers on one bar ring each pop
        # a share of the records and neither can tell that from a quiet feed.
        inner = RingBarSource.attach(args.bar_ring, symbol_id=None)
    elif args.bars:
        inner = RecordedBarSource(
            args.bars, symbol_id=symbol_ids[0], interval=args.interval
        )
    else:
        from axon.strategies.data import fixture_candles

        inner = HistoryBarSource(
            fixture_candles(args.fixture, args.interval), symbol_id=symbol_ids[0]
        )
    source = BoundedSource(inner, duration_s=args.duration)

    # ── the beacon ────────────────────────────────────────────────────────────
    bpath = args.beacon
    if bpath is None and args.bar_ring and not args.no_beacon:
        # Derived from the slice ring's path and never configured twice — a string
        # cannot equal itself plus a suffix, so the beacon can never name either ring's
        # own file (ADR-0034). `--bar-ring` takes the SLICE ring path for the same
        # reason the shadow CLI does.
        bpath = beacon_path(args.bar_ring)
    if args.no_beacon:
        bpath = None
    probe, beacon_note = wire_beacon(bpath, settle_s=args.beacon_settle)

    if not args.attest_venue:
        banner(
            "NOT ATTESTED AGAINST A VENUE. Nothing in this process can tell the Rust",
            "publisher from a harness writer — an MdBar record carries no source",
            "marker — so this run is a REHEARSAL of the harness and not an observation",
            "of a session. Pass --attest-venue only when a real testnet session is",
            "feeding this ring, and expect the report to repeat the claim back to you.",
        )

    print(f"source    : {source.describe()}", flush=True)
    print(
        f"traders   : symbol_id={symbol_ids} window={args.window} bars, interval "
        f"{args.interval}. On testnet BTC is asset index 3, not 0 — resolve it through "
        "the venue's own `meta` universe or off the ring, never by assumption.",
        flush=True,
    )

    # ── the traders ───────────────────────────────────────────────────────────
    traders = []
    for sid in symbol_ids:
        strategy = _build_strategy(sid, args)
        trader = ShadowTrader(
            strategy,
            symbol_id=sid,
            spec=getattr(strategy, "spec", PERP_BAR_V1),
            # One signal ring per trader: two runners producing into one ring would
            # interleave targets for different instruments under one sequence.
            ring_path=args.ring if len(symbol_ids) == 1 else f"{args.ring}.{sid}",
            interval=args.interval,
            window_bars=args.window,
            venue_attested=args.attest_venue,
        )
        # The two seams ADR-0030 left for the caller that owns the session, set on the
        # monitor the shadow trader built rather than on one of ours: `beacon` is the
        # probe decision 5 of `monitor.py` describes, and `tap` is every verdict as it
        # is produced. Both are public fields with defaults of `None`, so a trader that
        # is handed neither behaves exactly as it did before this file existed.
        trader.monitor.beacon = probe
        trader.monitor.tap = _tap_for(sid)
        # And the default logging sink is taken off, which is not the same as taking the
        # alarm off. The sink and the tap are the same verdict object; with both wired
        # to one terminal every non-OK window prints twice — once here and once through
        # the root logger — and a transcript in which every alarm appears twice is a
        # transcript an operator learns to skim. The tap prints the level, the reasons
        # and the publisher state, which is strictly more than `logging_sink` emits.
        trader.monitor.sink = None
        traders.append(trader)

    print("─" * 78, flush=True)
    started_ns = time.time_ns()
    try:
        reports = drive(
            traders,
            source,
            max_bars=args.max_bars,
            idle_timeout_s=args.idle_timeout,
            poll_s=POLL_S,
        )
    finally:
        for trader in traders:
            trader.close()
        source.close()

    # ── what it proved, and what it could not ─────────────────────────────────
    print("─" * 78)
    elapsed_s = (time.time_ns() - started_ns) / 1e9
    print(f"ran for {elapsed_s:.0f}s; beacon: {beacon_note}")
    _report_arrivals(source, symbol_ids)

    ok = True
    payload = {
        "started_ns": started_ns,
        "elapsed_s": elapsed_s,
        "source": source.describe(),
        "venue_attested": bool(args.attest_venue),
        "beacon": beacon_note,
        "beacon_path": bpath,
        "arrivals": {str(k): v for k, v in sorted(source.arrivals.items())},
        "symbols": {},
    }
    for sid, report in zip(symbol_ids, reports):
        print(f"── symbol {sid} ".ljust(78, "─"))
        print(report.summary())
        # Printed here rather than left inside `MonitorReport.summary()`, because
        # `ShadowReport` composes its own lines out of the monitor's *fields* and never
        # calls that method — so the one measurement this workstream exists to produce
        # would be collected on every run and shown on none of them. Counted and not
        # surfaced is the same as not counted.
        if report.monitor.silence is not None:
            print(report.monitor.silence.describe())
            if not args.attest_venue:
                print(
                    "  and over a recording these gaps are the speed of a for-loop, not "
                    "the speed of a feed: a file is never idle, so the numbers above "
                    "describe this process and say nothing whatever about a venue"
                )
        ok = ok and report.passed
        payload["symbols"][str(sid)] = {
            "passed": bool(report.passed),
            "bars": report.bars,
            "windows": report.windows,
            "rows_compared": report.rows_compared,
            "rows_owed": report.rows_owed,
            "max_abs_diff": float(report.monitor.max_abs_diff),
            "worst": str(report.monitor.worst),
            "silence": (
                None if report.monitor.silence is None else evidence_json(report.monitor.silence)
            ),
        }

    if args.evidence_out:
        Path(args.evidence_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.evidence_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nevidence written to {args.evidence_out}")

    if not args.attest_venue:
        banner(
            "REHEARSAL, NOT AN OBSERVATION. This run was not attested against a venue,",
            "so nothing above may be quoted as the live parity monitor having been run",
            "against a live session. Anything not run against the venue is not proven.",
        )
    return 0 if ok else 1


def _tap_for(symbol_id: int):
    def tap(verdict: Verdict) -> None:
        print_verdict(symbol_id, verdict)

    return tap


def _build_strategy(symbol_id: int, args: argparse.Namespace):
    """The serving path under test, with no model behind it by default.

    The monitor compares *feature vectors* and never looks at a score, so a constant
    predictor is the honest default: it makes the run a statement about the feature
    path, which is the only thing being diffed, and it takes no position — so a run with
    no would-be signals is a property of the predictor and not a finding about the
    strategy. ``--strategy`` and ``--registry`` reach the same factories the shadow CLI
    uses, so a family with its own feature set diffs against its own recompute.
    """
    from decimal import Decimal

    from axon.strategies.shadow import _strategy_factory

    registry = None
    if args.registry:
        from axon.models import ModelRegistry

        registry = ModelRegistry(args.registry)
    factory = _strategy_factory(args.strategy)
    return factory(
        symbol_id=symbol_id,
        max_position=None if args.max_position is None else Decimal(args.max_position),
        registry=registry,
        model=args.model,
    )


def _report_arrivals(source: BoundedSource, symbol_ids: Sequence[int]) -> None:
    """Which asset indices actually arrived, against the ones traders were built for.

    The failure this exists for has no other symptom: a run pointed at the wrong
    ``--symbol-id`` reads a healthy ring, hands its trader nothing, and produces a
    transcript identical to a dead publisher's. On testnet BTC is 3.
    """
    if not source.arrivals:
        print(
            "arrivals  : no bar arrived at all. That is a dead feed, a session with no "
            "candle subscription, or a ring nobody is publishing to — and note that on "
            "a venue that sends no frame for a minute with no trades, a short quiet "
            "run can also look like this."
        )
        return
    seen = ", ".join(f"symbol_id {sid}: {n} bar(s)" for sid, n in sorted(source.arrivals.items()))
    print(f"arrivals  : {seen}")
    unwatched = sorted(set(source.arrivals) - set(symbol_ids))
    if unwatched:
        print(
            f"            bars arrived for symbol_id {unwatched} that no trader was "
            "built for. If one of those is the instrument you meant, this run compared "
            "nothing for the reason above and not because the feed was quiet."
        )
    missing = sorted(set(symbol_ids) - set(source.arrivals))
    if missing:
        print(
            f"            !! NO BAR EVER ARRIVED for symbol_id {missing}, which a trader "
            "WAS built for. A wrong asset index and a dead feed are the same silence; "
            "this line is the only thing that tells them apart."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/sessions/parity_live.py",
        description=(
            "Drive the live parity monitor over a session's market-data bar ring, "
            "print a verdict per window, and record the gaps its silence deadline "
            "should be set from. Offline by default; --bar-ring is the only mode that "
            "touches a live session."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--bar-ring",
        metavar="MD_RING_PATH",
        help="THE LIVE PATH. The running session's SLICE ring path; the bar ring and "
        "the beacon beside it are both derived, never configured twice",
    )
    source.add_argument(
        "--fixture",
        metavar="COIN",
        default="BTC",
        help="offline, and the default: the committed candle fixture for COIN",
    )
    source.add_argument(
        "--bars",
        metavar="PATH",
        help="offline: a CSV of a previous run's bar-ring records "
        "(ts_event,open,high,low,close,volume), as written by shadow's --bars-out",
    )
    parser.add_argument("--interval", default="1m", help="the bar interval, Python spelling "
                        "(1m/1h). The venue's own spelling in the session TOML is m1")
    parser.add_argument(
        "--symbol-id",
        type=int,
        action="append",
        help="repeatable: one trader per symbol, all fed from the SAME reader. This is "
        "the VENUE's asset index, which the bar ring's records carry — on testnet BTC "
        "is 3 and not 0. Defaults to 3 on --bar-ring and 0 otherwise",
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_BARS)
    parser.add_argument("--ring", default="/dev/shm/axon-parity-live.ring",
                        help="the signal ring the serving path produces onto")
    parser.add_argument("--strategy", default="perp_bar")
    parser.add_argument("--registry", help="a model registry directory; without one no "
                        "target is ever taken, which the report states rather than hides")
    parser.add_argument("--model", default="perp_bar_xgb")
    parser.add_argument("--max-position", help="position size in coin units, as a decimal "
                        "STRING (never a float)")
    parser.add_argument(
        "--beacon",
        metavar="PATH",
        help="the market-data beacon. Derived from --bar-ring when omitted; this "
        "overrides that, and is how an offline rehearsal is pointed at a beacon written "
        "by `cargo run -p axon-ipc --example md_writer`",
    )
    parser.add_argument(
        "--beacon-settle",
        type=float,
        default=2.0,
        help="seconds to watch the beacon's beat before trusting it. A beat still at "
        "ZERO after this is a file the core's pass loop was never wired to, and wiring "
        "a probe to it would alarm 'dead publisher' on a healthy session; 0 disables "
        "the check, which is only right when something else has already established it",
    )
    parser.add_argument(
        "--no-beacon",
        action="store_true",
        help="run with no probe at all, to see exactly how the harness degrades. Every "
        "silence is then uncategorised and the report says so in three places",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="stop after this many seconds. A wall clock, and one of the two named "
        "exceptions here: it bounds the operator's afternoon and orders nothing",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        help="stop after this many bars for the fastest series. Event-driven, so the "
        "run is bounded by what it observed rather than by how long it sat there",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        help="stop after this many seconds with no bar at all. The only bound that can "
        "end a run on a dead feed, and therefore the one a live run must set",
    )
    parser.add_argument(
        "--attest-venue",
        action="store_true",
        help="the operator's claim that a real Hyperliquid session is feeding this "
        "ring. Never inferred: an MdBar record carries no source marker, so nothing in "
        "this process can tell the Rust publisher from a harness writer",
    )
    parser.add_argument(
        "--evidence-out",
        metavar="PATH",
        help="write the silence measurements and the run's shape to PATH as JSON. The "
        "transcript scrolls; the deadline argument should not have to",
    )
    args = parser.parse_args(argv)

    if args.attest_venue and not args.bar_ring:
        parser.error(
            "--attest-venue claims a live venue session is feeding this run, and "
            "nothing but --bar-ring can be fed by one. A recording attested as live is "
            "the one lie this harness must not be able to tell."
        )
    if args.bars and args.symbol_id and len(args.symbol_id) > 1:
        parser.error("--bars carries one series; give it one --symbol-id")
    return run(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
