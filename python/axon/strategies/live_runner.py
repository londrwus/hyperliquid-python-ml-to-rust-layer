"""Drive a strategy onto a **live** Rust session's signal ring, for real orders.

The shadow runner (:mod:`axon.strategies.shadow`) answers "would the serving path
have computed what an offline recompute computes". This module answers a different
and much smaller question: **can a Python strategy keep a real venue quote moving.**
Bars in from the market-data bar ring, targets out onto the runtime's *own* signal
ring, and from there through the planner to Hyperliquid.

It is deliberately not a second shadow trader. What this file owns is the three things
the shadow path does not do: it writes onto the ring the Rust core is actually reading,
it stamps the record so the reader will admit it, and it can be told to go flat and mean
it.

It *can* now carry a continuous diff, under ``--parity-diff``, and the distinction is
worth keeping straight: it does not carry a second one. The diff is
:class:`~axon.strategies.shadow.BarParityDiff` — the object the shadow path runs,
extracted so both callers hold the same alignment rule, the same window construction and
the same monitor configuration. Duplicating a cut-down comparison beside a live order
flow is still how two answers to one question get into a tree; the reason a diff can be
here at all is that this is not one.

Run it as::

    .venv/bin/python -m axon.strategies.live_runner \\
        --md-ring /dev/shm/p6-live-md.ring \\
        --signal-ring /dev/shm/p6-live-signal.ring \\
        --symbol-id 3 --strategy baseline --max-position 0.0003 \\
        --duration 3600 --transcript data/p6/live-run.jsonl

── the stamp, which is the whole reason this file exists ─────────────────────────

:class:`~axon.live.runner.StrategyRunner` binds **the event's own** ``ts_event``
around the callback and contains no clock at all. That is right for everything it
was built for and it is wrong here, and the difference is measured rather than
argued:

* The core's event clock runs **1.4–1.9 s behind wall time** on testnet BTC's
  ``bbo`` — it is a high-water mark over observed market data, so it is always as
  old as the last frame.
* The signal reader's admission window is ``min(ttl_ms, max_signal_age_ms)``, judged
  against that clock.
* An m1 bar's ``ts_event`` is its own **close**, and the bar arrives one venue frame
  after it. So a record stamped with the bar's close is already a second or more old
  before it is written, and everything downstream spends the rest of the budget.

Agent P1's first live run refused its own second target as ``expired`` and planned
nothing, from exactly this. So the stamp here is **the producer's own wall clock**:
the moment the strategy decided, which is what a decision's timestamp means. The
reader admits a record from slightly ahead of its clock and counts it under
``ahead_of_clock``; that asymmetry is deliberate on its side too, because a signal
from the future is a clock-skew observation while a signal from the past is a
decision about a market that has moved.

This is a **named exception** to event-time ordering, in the same class as the
dead-man's-switch deadline (wall clock because the venue holds it) and reconnect
backoff (wall clock because it is not ordering). Nothing about the *ordering* of
bars is taken from a wall clock: bars are consumed in ring order, the strategy sees
their real close times, and the transcript records both stamps side by side so the
skew is a number in the log rather than an assumption.

── what it refuses to do ─────────────────────────────────────────────────────────

**It never drops a signal because the ring is full.** A full ring means the Rust
consumer is behind or dead, and skipping "because the next target supersedes it
anyway" is wrong in the case that matters: if the consumer is dead there is no next
one, and the venue is left holding the previous target. Records queue in order and
the run fails loudly if the queue outgrows the ring.

**It never re-sends an unchanged target.** That is the strategy's own rule and this
file does not second-guess it. The consequence has to be stated because it is not
obvious: the sweeper (ADR-0031) cancels a tracked order older than
``intent.max_order_age_ms`` on a symbol *no signal spoke for*, so a strategy holding
a target lets its own quote be swept. That is the designed behaviour and this run
wants to observe it, not paper over it with a keepalive.

**It never decides the account is flat.** ``--flatten-on-exit`` emits a target of
zero and waits, which is a request to the planner, not an observation. Whether the
account is actually flat is a question for ``clearinghouseState``, asked more than
once, by the operator.

**And it is not the cleanup pass.** A target of zero is only an *exit* when the
tracker already knows the position, because the planner acts on the difference
between a target and what it believes it holds. That belief arrives from the
``userFills`` replay and not from the reconcile poll, so a fresh session can be flat
one status line and correct the next — and a flatten that lands in between places
nothing. On 2026-07-27 it landed in between. Use ``axon --flatten`` when the job is
"get out": it reads the venue's own position, sizes a reduce-only close from it, and
ladders the urgency when a venue refuses a TIF. ``--flatten-on-exit`` stays what it
is — a session's own last decision, through the same ring, reader and planner as
every other one, which is what keeps a capture of it replayable.
"""

from __future__ import annotations

import json
import signal as signalmod
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

import numpy as np

from axon.strategy.context import (
    URGENCY_CROSS,
    URGENCY_JOIN,
    URGENCY_POST_ONLY,
    URGENCY_TAKE,
    StrategyContext,
)

#: Poll cadence on the bar ring, in seconds. A bar arrives once a minute, so this is
#: nowhere near a latency question — it is chosen small enough that the wall-clock
#: stamp on a signal is within a poll of the bar that caused it, because the gap
#: between "the bar landed" and "the decision is stamped" is skew the reader pays for.
POLL_S = 0.1

#: How often the run prints a line when nothing has happened. A quiet strategy and a
#: dead one produce the same (empty) ring, and an hour of silence in a log is not
#: evidence of either.
HEARTBEAT_S = 60.0

#: The urgency table by name, for ``--flatten-urgency``. Named rather than numeric on
#: the command line because the numbers are a contract between two languages
#: (`axon_strategy::URGENCY_TABLE`) and "2 sounds about right" is exactly how an
#: operator crosses the spread by accident.
FLATTEN_URGENCIES = {
    "post_only": URGENCY_POST_ONLY,
    "join": URGENCY_JOIN,
    "cross": URGENCY_CROSS,
    "take": URGENCY_TAKE,
}


class LiveRunError(Exception):
    """A fault that must stop the run rather than be absorbed into a counter."""


def stamp_cause(records: list, bar: Any) -> None:
    """Put the bar's own close on every record it produced.

    **One line, and it is a function so that there is one of it.** The strategy does not
    do this, and asking it to would be asking every strategy author to remember a runtime
    invariant — the mistake ADR-0036 §5 names for the re-quote. The runner is the only
    thing holding both the bar and the records; ADR-0038 added a second runner, and a
    second copy of this line is how one of them would quietly stop stamping and the `bar`
    latency stage would report on half a session.

    See :class:`~axon.strategy.context.StrategyContext` and the module docstring for why
    ``ts_event`` is the producer's clock while this is the venue's.
    """
    for rec in records:
        rec["ts_cause"] = bar.ts_event


class SignalOutbox:
    """One producer's end of the boundary: one ring, one sequence space, one FIFO.

    Extracted from :class:`LiveRunner` when ADR-0038 gave a session more than one
    producer. What it owns is the part that must not be written twice: creating the ring,
    holding the ``seq`` the Rust reader validates, and the backpressure rule.

    **It never drops a signal because the ring is full.** A full ring means the Rust
    consumer is behind or dead, and skipping "because the next target supersedes it
    anyway" is wrong in the case that matters: if the consumer is dead there is no next
    one, and the venue is left holding the previous target. Records queue in order and the
    run fails loudly if the queue outgrows the ring.

    **One outbox per ring, always.** An SPSC ring has one writer; two outboxes on one path
    would interleave two sequences into a stream the reader validates as one, and every
    record of the loser would be refused as ``stale_seq`` — which reads on the Rust status
    line exactly like a strategy with nothing to say.
    """

    def __init__(
        self,
        signal_ring: str,
        *,
        model_version: int,
        capacity: int = 1024,
        first_seq: int = 0,
    ) -> None:
        from axon.signals import RingProducer

        # Creating the producer creates and zeroes the ring. The Rust intent source
        # retries its attach every `intent.attach_retry_ms`, which is what makes
        # "start the session first, attach the producer later" a supported order and
        # not a race — but it also means this must not be pointed at a ring a core is
        # already reading, because the indices would rewind under it.
        self._producer = RingProducer(signal_ring, capacity=capacity)
        self.path = signal_ring
        self.ctx = StrategyContext(model_version=model_version, first_seq=first_seq)
        self._outbox: list[np.ndarray] = []
        self._max_outbox = capacity
        self.pushed = 0
        self.backpressure_waits = 0

    def take_pending(self) -> list:
        """Whatever the strategy emitted inside the last :meth:`StrategyContext.event`."""
        return self.ctx.take_pending()

    def queue(self, records: list) -> None:
        self._outbox.extend(records)

    def flush(self) -> int:
        """Push as much of the outbox as the ring will take."""
        pushed = 0
        while self._outbox:
            if not self._producer.try_push(self._outbox[0]):
                self.backpressure_waits += 1
                if len(self._outbox) > self._max_outbox:
                    raise LiveRunError(
                        f"signal ring {self.path} full and outbox at {len(self._outbox)}: "
                        "the Rust consumer is not draining. Nothing was dropped"
                    )
                break
            self._outbox.pop(0)
            pushed += 1
            self.pushed += 1
        return pushed

    def depth(self) -> int:
        return len(self._producer)

    def pending(self) -> int:
        """Records queued and not yet on the ring.

        Public because it is the number that proves nothing was dropped: a run which
        reported backpressure and then quietly discarded the target it was holding
        satisfies every counter except this one.
        """
        return len(self._outbox)

    def close(self) -> None:
        self._producer.close()


@dataclass
class LiveStats:
    """What the run did, as counters a transcript can be checked against."""

    bars_seen: int = 0
    bars_other_symbol: int = 0
    rows_finite: int = 0
    targets_changed: int = 0
    signals_pushed: int = 0
    backpressure_waits: int = 0
    #: Per-bar skew between the producer's stamp and the bar's own close, in ms.
    #: Kept as extremes rather than a mean: the reader's admission is a threshold,
    #: and a mean cannot say whether the threshold was ever crossed.
    skew_ms_min: float | None = None
    skew_ms_max: float | None = None
    first_bar_ts: int | None = None
    last_bar_ts: int | None = None
    targets: list[str] = field(default_factory=list)

    def observe_skew(self, ms: float) -> None:
        self.skew_ms_min = ms if self.skew_ms_min is None else min(self.skew_ms_min, ms)
        self.skew_ms_max = ms if self.skew_ms_max is None else max(self.skew_ms_max, ms)

    def summary(self) -> str:
        span = ""
        if self.first_bar_ts is not None and self.last_bar_ts is not None:
            span = (
                f", span {(self.last_bar_ts - self.first_bar_ts) / 60e9:.1f} min "
                f"({self.first_bar_ts} → {self.last_bar_ts})"
            )
        skew = (
            "n/a"
            if self.skew_ms_min is None
            else f"{self.skew_ms_min:.0f}..{self.skew_ms_max:.0f} ms"
        )
        return (
            f"bars {self.bars_seen} (other symbols {self.bars_other_symbol}){span}\n"
            f"finite feature rows {self.rows_finite}\n"
            f"target changes {self.targets_changed}: {' → '.join(self.targets) or 'none'}\n"
            f"signals pushed {self.signals_pushed}, backpressure waits "
            f"{self.backpressure_waits}\n"
            f"producer-stamp minus bar-close skew {skew}"
        )


class Transcript:
    """A JSONL record of every bar, every decision and every push.

    Written as the run goes rather than at the end, because the interesting run is
    the one that dies: a summary printed on a clean exit is exactly the evidence a
    crashed session does not have.
    """

    def __init__(self, path: str | None) -> None:
        self._fh = open(path, "a", buffering=1) if path else None

    def write(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        rec = {"wall_ns": time.time_ns(), "kind": kind}
        rec.update(fields)
        self._fh.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class LiveRunner:
    """One strategy, one bar ring, one signal ring, one instrument.

    Single-instrument on purpose. Two traders on one bar ring each pop a share of
    the bars and neither can tell that from a quiet feed — the same hazard ADR-0029
    §5 names for the shadow path.

    ── the fan-out, and why it is here now ───────────────────────────────────────

    That hazard is also the reason no live parity monitor could watch a session that
    was **trading**. The monitor attaches to the bar ring; on a trading session the
    *strategy* is that ring's consumer; and an SPSC ring has one consumer, so two
    readers do not share it, they steal from it — measured on 2026-07-26, when two
    drainers each saw about half the records and each reported the other's reads as
    drops. The second way out, a read-only session beside the trading one, is worse:
    ``graceful_shutdown`` calls ``cancel_all``, which on Hyperliquid is account-wide,
    so the watcher's exit would cancel the trader's resting orders. Phase 6 therefore
    computed parity **afterwards, from the session's own capture**, which keeps the
    comparison and gives up the one thing a monitor is for: telling an operator
    mid-run that parity has broken.

    So this file now carries the fan-out: **one reader, dispatched to the strategy and
    to a diff.** What made that a bolt-on before was the risk of a second, cut-down
    comparison living beside a live order flow — two answers to one question, drifting
    on exactly the window where it mattered. It is not a second one:
    :class:`~axon.strategies.shadow.BarParityDiff` is the diff the shadow path runs,
    extracted so both callers hold the same object, with one alignment rule, one window
    construction and one monitor configuration between them.

    The diff is **optional and off by default**, and that is not timidity. It runs
    ``spec.compute`` over the whole recording on every window boundary, on the same
    thread that is about to stamp a decision — so a session that wants the alarm pays
    for it in the one place this file measures, and an operator who does not want to
    pay keeps the behaviour every previous run had.
    """

    def __init__(
        self,
        strategy: Any,
        *,
        symbol_id: int,
        signal_ring: str,
        model_version: int,
        capacity: int = 1024,
        first_seq: int = 0,
        transcript: Transcript | None = None,
        parity: Any | None = None,
    ) -> None:
        self.strategy = strategy
        self.symbol_id = int(symbol_id)
        self.stats = LiveStats()
        self._transcript = transcript or Transcript(None)
        # The ring, the sequence and the backpressure rule, in the object that owns them
        # for every runner in this package. See :class:`SignalOutbox`.
        self._out = SignalOutbox(
            signal_ring,
            model_version=model_version,
            capacity=capacity,
            first_seq=first_seq,
        )
        self._ctx = self._out.ctx
        self._last_target: Decimal | None = None
        #: The parity diff this run dispatches to beside the strategy, or ``None``.
        self._parity = parity
        #: Verdicts the diff raised that did not pass. Kept as a count and a last-reason
        #: rather than a list: a session that has broken parity has broken it, and an
        #: unbounded list of the same reason is a memory leak with a sad story.
        self.parity_alarms = 0
        self.parity_last_reason: str | None = None

    # ── the loop ─────────────────────────────────────────────────────────────

    def on_bar(self, bar: Any) -> None:
        """Dispatch one bar and push whatever it decided.

        Bars for other instruments are counted and dropped. They are *not* handed to
        the strategy: this runner drives one symbol, and a strategy that filters
        internally (the baseline does) would silently accept the extra dispatch while
        one that did not would trade another coin's tape.
        """
        if bar.symbol_id != self.symbol_id:
            self.stats.bars_other_symbol += 1
            return

        self.stats.bars_seen += 1
        if self.stats.first_bar_ts is None:
            self.stats.first_bar_ts = bar.ts_event
        self.stats.last_bar_ts = bar.ts_event

        # **The producer's clock, read once per bar**, so the stamp on every record
        # this bar produces is the same instant and the transcript's skew figure is
        # about the feed rather than about how long the callback took.
        decided_ns = time.time_ns()
        skew_ms = (decided_ns - bar.ts_event) / 1e6
        self.stats.observe_skew(skew_ms)

        before = self.strategy.target
        with self._ctx.event(decided_ns) as ctx:
            self.strategy.on_bar(bar, ctx)
        emitted = self._ctx.take_pending()
        after = self.strategy.target

        # **Stamp the bar's own close onto every record this bar produced.**
        #
        # The strategy does not do this, and asking it to would be asking every strategy
        # author to remember a runtime invariant — the mistake ADR-0036 §5 names for the
        # re-quote. This runner is the only thing that knows both stamps at once: it read
        # the producer's clock above and it is holding the bar.
        #
        # ``skew_ms`` below is the same quantity and it was for two days the *only* record
        # of it. It lived in this transcript, which means it could be quoted afterwards and
        # never budgeted: the runtime saw records carrying only ``ts_event``, so from its
        # side a decision one second after a bar and one two minutes after it were
        # identical. That gap was measured at 951 / 12 051 / **111 475** ms over 57 live
        # bars and was the largest number in the system. Schema version 3 put it on the
        # wire; this is where it is written.
        #
        # Set here rather than passed into ``emit_target`` so a strategy that emits
        # several records for one bar cannot label one of them and forget the rest.
        stamp_cause(emitted, bar)

        row = self.strategy.feature_row()
        if row is not None:
            self.stats.rows_finite += 1

        # **The fan-out.** The same bar, the same row the strategy just served, handed to
        # the diff — so parity is checked against the decision that is about to become an
        # order rather than reconstructed from a capture afterwards. One reader, two
        # consumers; a second reader on this ring would take bars away from the thing
        # placing orders.
        if self._parity is not None:
            self._observe_parity(bar, row)

        if after != before:
            self.stats.targets_changed += 1
            self.stats.targets.append(str(after))

        self._transcript.write(
            "bar",
            symbol_id=bar.symbol_id,
            bar_ts=bar.ts_event,
            decided_ns=decided_ns,
            skew_ms=round(skew_ms, 1),
            close=bar.close,
            volume=bar.volume,
            row=None if row is None else [float(v) for v in row],
            target_before=str(before),
            target_after=str(after),
            emitted=len(emitted),
        )
        if emitted:
            self._out.queue(emitted)
            self.flush()

    def _observe_parity(self, bar: Any, row: Any) -> None:
        """Hand one bar to the diff and shout if it disagrees.

        **A parity break must not stop the run, and that is a decision rather than an
        omission.** This process is holding a position and has a signal ring the Rust core
        is reading; raising here would abandon both with no flatten. The alarm's job is to
        tell an operator, in the log, at the moment it happened — which is the entire
        thing Phase 6's after-the-fact autopsy could not do. What to do about it is the
        operator's call, and ``axon --flatten`` is how they act on it.

        A diff that *itself* fails — a backwards bar series, a spec that raises — is
        caught for the same reason: the diff is a watcher, and a watcher that can take
        down the thing it watches is a liability rather than a safeguard.
        """
        try:
            verdict = self._parity.observe(bar, row)
        except Exception as exc:  # noqa: BLE001 - a watcher must not kill the trader
            self.parity_alarms += 1
            self.parity_last_reason = f"the diff itself failed: {exc!r}"
            print(f"  PARITY DIFF FAILED: {exc!r}", flush=True)
            self._transcript.write("parity_error", error=repr(exc))
            return
        if verdict is None:
            return
        passed = bool(getattr(verdict, "passed", True))
        self._transcript.write(
            "parity",
            passed=passed,
            level=str(getattr(verdict, "level", "")),
            max_abs_diff=getattr(verdict, "max_abs_diff", None),
            reasons=list(getattr(verdict, "reasons", ()) or ()),
        )
        if passed:
            return
        self.parity_alarms += 1
        reasons = list(getattr(verdict, "reasons", ()) or ())
        self.parity_last_reason = "; ".join(reasons) or str(verdict)
        # Uppercase and on its own line: this is the one thing on this run's stdout that
        # says the model being served is not the model that was gated.
        print(f"  PARITY ALARM ({self.parity_alarms}): {self.parity_last_reason}", flush=True)

    def emit_target(self, target: Decimal, *, urgency: int | None = None, **kwargs: Any) -> None:
        """Put a target on the ring **outside** a bar — the flatten path only.

        Separate from :meth:`on_bar` and named for what it is: a target that no
        market observation produced. Its stamp is a wall clock for the same reason
        every other stamp here is, and it carries the strategy's own ttl so the
        planner treats it exactly like a decided one.

        ``urgency`` overrides the strategy's, and the override earns its place from
        an observation rather than from symmetry. This run watched the **sweeper**
        cancel a resting exit at 62 s (`intent.max_order_age_ms`), which is correct
        and leaves a position behind: the strategy's target is already 0, a target
        position is idempotent, and so it will never re-emit the quote that would
        close it. A post-only flatten can therefore be swept in turn and the run ends
        holding the position it asked to be rid of. :data:`~axon.strategy.context.
        URGENCY_TAKE` is the operator's answer — an IOC through the far touch, which
        leaves no remainder resting, because an urgent exit that half-fills and rests
        is an unmanaged position.
        """
        p = self.strategy.params
        with self._ctx.event(time.time_ns()) as ctx:
            ctx.emit_target(
                self.symbol_id,
                target,
                urgency=p.urgency if urgency is None else int(urgency),
                ttl_ms=p.ttl_ms,
                **kwargs,
            )
        self._out.queue(self._ctx.take_pending())
        self._transcript.write(
            "operator_target",
            target=str(target),
            urgency=p.urgency if urgency is None else int(urgency),
        )
        self.flush()

    def flush(self) -> int:
        """Push as much of the outbox as the ring will take.

        Full is backpressure, never a licence to drop: the records stay in order and
        the run raises if the backlog outgrows a whole ring, because at that point
        the consumer is gone and there is no local answer that is not a lie about
        what the venue is holding. The rule itself lives in :class:`SignalOutbox`, so
        the multi-producer runner cannot end up with a second one.
        """
        pushed = self._out.flush()
        self.stats.signals_pushed = self._out.pushed
        self.stats.backpressure_waits = self._out.backpressure_waits
        if pushed:
            self._transcript.write("pushed", n=pushed, total=self.stats.signals_pushed)
        return pushed

    def ring_depth(self) -> int:
        return self._out.depth()

    def pending(self) -> int:
        """Records queued behind a full ring. See :meth:`SignalOutbox.pending`."""
        return self._out.pending()

    def close(self) -> None:
        self._out.close()


# ── strategy construction ────────────────────────────────────────────────────


def build_strategy(name: str, *, symbol_id: int, max_position: str | None, registry, model):
    """Reuse :mod:`axon.strategies.shadow`'s factory table, never a second one.

    A live runner with its own idea of what ``baseline`` means is a live runner that
    trades a different object from the one the shadow run measured, and the two would
    diverge on the day somebody changed a default. ``module:callable`` works here for
    the same reason it works there — a model zoo lands one family at a time.
    """
    from axon.strategies.shadow import _strategy_factory

    factory = _strategy_factory(name)
    return factory(
        symbol_id=symbol_id,
        max_position=None if max_position is None else Decimal(max_position),
        registry=registry,
        model=model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m axon.strategies.live_runner",
        description="Drive a strategy onto a live session's signal ring. Places REAL orders.",
    )
    parser.add_argument(
        "--md-ring",
        required=True,
        help="the Rust session's market-data SLICE ring path; the bar ring beside it "
        "is derived (ADR-0028 §5), never named twice",
    )
    parser.add_argument(
        "--signal-ring",
        required=True,
        help="the ring named in the session's [ipc] table. This process CREATES it and "
        "zeroes head/tail, so the Rust side must attach after — which it does, on its "
        "own retry timer",
    )
    parser.add_argument("--symbol-id", type=int, required=True)
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument(
        "--max-position",
        default=None,
        help="position size in coin units, as a decimal STRING (never a float: 0.0003 "
        "has no exact float form and the venue rounds the difference somewhere nobody "
        "chose). Omit to take the strategy's own default",
    )
    parser.add_argument("--registry", default=None)
    parser.add_argument("--model", default="perp_bar_xgb")
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="what goes in the record's model_version field. Defaults to the "
        "strategy's artifact version, or to baseline.NO_MODEL_VERSION (u32::MAX) when "
        "there is no artifact — a conspicuous value, so a capture of a no-model session "
        "is not byte-identical to one serving registry version 1",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds. WALL CLOCK, and named as the exception it "
        "is: a run's budget is an operator's afternoon, which no event time measures",
    )
    parser.add_argument(
        "--max-bars", type=int, default=None, help="stop after this many bars for this symbol"
    )
    parser.add_argument(
        "--flatten-on-exit",
        action="store_true",
        help="emit a target of zero before stopping and wait --flatten-wait seconds. A "
        "REQUEST to the planner, not a guarantee: whether the account is flat is a "
        "question for clearinghouseState, asked more than once",
    )
    parser.add_argument("--flatten-wait", type=float, default=45.0)
    parser.add_argument(
        "--flatten-urgency",
        choices=sorted(FLATTEN_URGENCIES),
        default=None,
        help="urgency for the flatten target only, overriding the strategy's. Use "
        "'take' (an IOC through the far touch) when the position must actually be "
        "gone: a post-only flatten can be swept at intent.max_order_age_ms, and a "
        "target position is idempotent, so nothing will re-emit it",
    )
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="skip the bar loop entirely: attach, emit one flatten target, wait, exit. "
        "**Prefer `axon --flatten` for a cleanup pass.** This path emits a target of "
        "zero onto the ring, and the planner subtracts a target from the *tracked* "
        "position -- so against a tracker that has not learned the position (reconcile "
        "reports drift and never writes it; only the userFills replay writes it) this is "
        "a no-op, which is what it was on 2026-07-27. `axon --flatten` reads the venue's "
        "own position, sizes a reduce-only close from it, and ladders the urgency when a "
        "venue refuses a TIF. Keep this flag for driving a target through the *same* "
        "ring, reader and planner a session's own decisions take -- which is what makes "
        "a capture of it replayable",
    )
    parser.add_argument(
        "--first-seq",
        type=int,
        default=0,
        help="the sequence number to start at. **A second producer against a session "
        "that is still running MUST set this above the first producer's last seq.** "
        "Creating a RingProducer zeroes the ring's head and tail, but the Rust reader "
        "keeps its own `seq` baseline and refuses everything at or below it as "
        "`stale_seq` — so a cleanup pass starting at 0 is silently discarded while "
        "both sides' counters look healthy",
    )
    parser.add_argument(
        "--parity-diff",
        action="store_true",
        help="watch feature parity AS THE SESSION TRADES, instead of computing it "
        "afterwards from the capture. One reader, dispatched to the strategy and to the "
        "diff -- an SPSC bar ring has one consumer, so a monitor attached beside the "
        "strategy would take bars away from the thing placing orders, and a second "
        "read-only session would sweep this one's resting orders on its way out "
        "(cancel_all is account-wide). Off by default because it recomputes the whole "
        "spec at every window boundary on this thread; a break raises an alarm in the "
        "log and never stops the run, because this process is holding a position",
    )
    parser.add_argument(
        "--parity-window",
        type=int,
        default=None,
        help="bars per parity window. Defaults to the shadow path's own constant, so the "
        "two report on the same granularity",
    )
    parser.add_argument("--transcript", default=None, help="append a JSONL record of the run")
    args = parser.parse_args(argv)

    from axon.marketdata import bar_ring_path
    from axon.strategies.baseline import NO_MODEL_VERSION
    from axon.strategies.shadow import RingBarSource

    registry = None
    if args.registry:
        from axon.models import ModelRegistry

        registry = ModelRegistry(args.registry)

    strategy = build_strategy(
        args.strategy,
        symbol_id=args.symbol_id,
        max_position=args.max_position,
        registry=registry,
        model=args.model,
    )
    model_version = args.model_version
    if model_version is None:
        model_version = getattr(strategy, "artifact_version", None) or NO_MODEL_VERSION

    warmup = getattr(strategy, "warmup_bars", None)
    print(f"strategy   : {args.strategy} on symbol {args.symbol_id}")
    print(f"size       : max_position={strategy.params.max_position} (coin units)")
    print(f"warmup     : {warmup} bars — measured, not restated")
    print("stamp      : the PRODUCER's wall clock (see module docstring)")
    print(f"bar ring   : {bar_ring_path(args.md_ring)}")
    print(f"signal ring: {args.signal_ring}  model_version={model_version}")

    parity = None
    if args.parity_diff:
        # Imported here, not at module scope: `axon.strategies.shadow` reaches for the ML
        # stack, and a live runner that could not start without it would be a live runner
        # that could not start on a box with no research extras.
        from axon.strategies.shadow import DEFAULT_WINDOW_BARS, BarParityDiff

        spec = getattr(strategy, "spec", None)
        if spec is None:
            raise SystemExit(
                "--parity-diff needs a strategy that exposes its feature `spec`; "
                f"{args.strategy!r} does not, so there is nothing to recompute against"
            )
        parity = BarParityDiff(
            spec=spec,
            interval="1m",
            window_bars=args.parity_window or DEFAULT_WINDOW_BARS,
        )
        print(
            f"parity     : ON, {parity.window_bars}-bar windows over {len(spec.columns)} "
            "columns (one reader, dispatched to the strategy AND the diff)"
        )
    else:
        print("parity     : OFF - it will have to be computed afterwards from the capture")

    transcript = Transcript(args.transcript)
    transcript.write(
        "start",
        strategy=args.strategy,
        symbol_id=args.symbol_id,
        max_position=str(strategy.params.max_position),
        ttl_ms=strategy.params.ttl_ms,
        urgency=strategy.params.urgency,
        warmup_bars=warmup,
        model_version=int(model_version),
        md_ring=args.md_ring,
        signal_ring=args.signal_ring,
    )

    stopping = {"now": False}

    def _stop(signum, _frame):
        # Flagged, not acted on: unwinding the ring and the transcript from inside a
        # signal handler is how a run's last records get lost.
        stopping["now"] = True

    signalmod.signal(signalmod.SIGTERM, _stop)
    signalmod.signal(signalmod.SIGINT, _stop)

    runner = LiveRunner(
        strategy,
        symbol_id=args.symbol_id,
        signal_ring=args.signal_ring,
        model_version=int(model_version),
        first_seq=args.first_seq,
        transcript=transcript,
        parity=parity,
    )
    # **Wall clock, named.** The deadline measures an operator's budget and an
    # absence; an event-time deadline for "nothing has arrived" can never expire,
    # because the clock that would advance it is the thing that stopped.
    started = time.monotonic()
    last_beat = started
    rc = 0
    try:
        with RingBarSource.attach(args.md_ring, symbol_id=None) as src:
            print(f"attached   : {src.describe()}", flush=True)
            while not args.flatten_only:
                if stopping["now"]:
                    print("stopping: signal received", flush=True)
                    break
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    print(f"stopping: --duration {args.duration}s reached", flush=True)
                    break
                if args.max_bars is not None and runner.stats.bars_seen >= args.max_bars:
                    print(f"stopping: --max-bars {args.max_bars} reached", flush=True)
                    break
                bars = src.poll()
                for bar in bars:
                    runner.on_bar(bar)
                    if bar.symbol_id == args.symbol_id:
                        row = strategy.feature_row()
                        z = "warming" if row is None else f"z={row[0]:+.3f}"
                        print(
                            f"  bar {runner.stats.bars_seen:>3} ts={bar.ts_event} "
                            f"close={bar.close / 1e8:.1f} {z} "
                            f"target={strategy.target} "
                            f"sig={runner.stats.signals_pushed}",
                            flush=True,
                        )
                runner.flush()
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_S:
                    last_beat = now
                    print(
                        f"  … {now - started:.0f}s  bars={runner.stats.bars_seen} "
                        f"target={strategy.target} "
                        f"signals={runner.stats.signals_pushed} "
                        f"ring_depth={runner.ring_depth()} "
                        f"feed(bars={src.health.bars} drops={src.health.ring_dropped} "
                        f"gaps={src.health.feed_gaps})",
                        flush=True,
                    )
                    transcript.write(
                        "heartbeat",
                        elapsed_s=round(now - started, 1),
                        bars=runner.stats.bars_seen,
                        target=str(strategy.target),
                        signals=runner.stats.signals_pushed,
                        feed_bars=src.health.bars,
                        feed_drops=src.health.ring_dropped,
                        feed_gaps=src.health.feed_gaps,
                    )
                time.sleep(POLL_S)

            if args.flatten_on_exit or args.flatten_only:
                # Emitted **unconditionally**, even when the strategy already believes
                # it is flat. The strategy's `target` is what it wants; the planner
                # acts on the difference between a target and the *tracked* position,
                # and a partial fill is exactly the case where those two disagree. A
                # flatten that skipped because the strategy said "0" would skip in the
                # one situation it was written for.
                urg = (
                    None if args.flatten_urgency is None
                    else FLATTEN_URGENCIES[args.flatten_urgency]
                )
                print(
                    f"flatten: emitting target 0 (strategy held {strategy.target}) "
                    f"urgency={args.flatten_urgency or 'strategy default'}",
                    flush=True,
                )
                runner.emit_target(Decimal(0), urgency=urg)
                deadline = time.monotonic() + args.flatten_wait
                while time.monotonic() < deadline:
                    # Keep draining the bar ring while waiting, so the transcript does
                    # not have a hole exactly where the unwind happened — and so the
                    # ring does not silently back up behind a paused reader.
                    for bar in src.poll():
                        if bar.symbol_id != args.symbol_id:
                            runner.stats.bars_other_symbol += 1
                    runner.flush()
                    time.sleep(POLL_S)
    except Exception as exc:  # noqa: BLE001 - the transcript must record the death
        transcript.write("error", error=repr(exc))
        print(f"FAILED: {exc!r}", flush=True)
        rc = 1
    finally:
        transcript.write("stats", **{
            "bars_seen": runner.stats.bars_seen,
            "bars_other_symbol": runner.stats.bars_other_symbol,
            "rows_finite": runner.stats.rows_finite,
            "targets_changed": runner.stats.targets_changed,
            "signals_pushed": runner.stats.signals_pushed,
            "backpressure_waits": runner.stats.backpressure_waits,
            "skew_ms_min": runner.stats.skew_ms_min,
            "skew_ms_max": runner.stats.skew_ms_max,
            "targets": runner.stats.targets,
            "parity_alarms": runner.parity_alarms,
            "parity_last_reason": runner.parity_last_reason,
        })
        print("── run summary ".ljust(72, "─"))
        print(runner.stats.summary())
        if parity is not None:
            # Printed whichever way it went. A parity run that reports nothing is
            # indistinguishable from one that was never asked for.
            print(
                f"parity: {parity.windows} window(s), {parity.rows_compared}/"
                f"{parity.rows_owed} rows compared, {runner.parity_alarms} alarm(s)"
            )
            print(f"  {parity.column_diff().describe()}")
            if runner.parity_last_reason:
                print(f"  last: {runner.parity_last_reason}")
        runner.close()
        transcript.close()
    return rc


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "FLATTEN_URGENCIES",
    "SignalOutbox",
    "HEARTBEAT_S",
    "POLL_S",
    "LiveRunError",
    "LiveRunner",
    "LiveStats",
    "Transcript",
    "build_strategy",
    "main",
    "stamp_cause",
]
