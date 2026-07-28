"""Phase 4's exit criterion, as a program: one Python decision, one order at the venue.

``docs/08``'s Phase-4 exit line is *"a trivial Python strategy emits target
positions; Rust executes on testnet"*. Every part of that path existed — the
context, the ring, the reader, the planner, the submit pipeline — and had only ever
been run against canned bytes and a spy client. What was missing was not code but a
**cause**: something that would put a real target position on a real ring, derived
from real market data, at a moment a live core would still admit it.

That is this module, and it is deliberately the smallest thing that can be traced
end to end:

.. code-block:: text

    Rust core ─▶ md ring ─▶ MdRingFeed ─▶ TargetProbe ─▶ StrategyContext ─▶ signal ring ─▶ Rust core ─▶ venue

:class:`TargetProbe` opens a position of a stated size, holds it for a stated span
of **event time**, and flattens. It is not a trading strategy and does not pretend
to be one: it has no view, no features and no model. What it has is the property a
trading strategy is useless without and which nothing in this repository had ever
demonstrated at a venue — that a target position formed in Python becomes an order
that a venue accepts.

Three things about it are decisions rather than convenience.

**It is driven by the venue's clock and holds by the venue's clock.** ``hold_ms``
is compared against ``ts_event``, never against ``time.monotonic()``. Wall-clock
pacing would make the probe emit its flatten at a moment that depends on how fast
this machine drained the ring, so replaying the same market would produce a
different pair of orders — and the parity harness the whole of Phase 5 rests on
diffs exactly that pair.

**It waits for the book before it decides.** A core that has received no market data
skips the intent pass entirely (ADR-0020 §2), and a signal emitted into that window
is consumed, aged, and refused with nothing to show for it. ``warmup`` quotes is not
a settling period for the probe — the probe needs no settling — it is proof that the
core on the other end has a clock and a book.

**It flattens by ``FLAG_CLOSE``, not by a zero target.** A zero target is an opinion
about the position, and an opinion computed against a fill we have not yet been told
about overshoots into the opposite side. Close says *get out* regardless of what the
quantity field holds, and it implies reduce-only on the Rust side for that reason
(ADR-0014 §2).

Run it against a live sandbox session::

    python -m axon.live.probe \\
        --md-ring /dev/shm/axon-md.ring --ring /dev/shm/axon-signal.ring \\
        --symbol-id 3 --size 0.0002 --hold-ms 8000 --urgency 3

Nothing here opens a socket, holds a key, or knows a venue exists. It writes 64-byte
records into a file; whether any of them becomes an order is the Rust session's
decision, and every gate that decision passes through is on the other side of the
ring by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from typing import Any

from axon.contracts import FIXED_POINT_DECIMALS
from axon.live.mdfeed import MdFeedError, MdRingFeed
from axon.live.runner import BackpressureError, StrategyRunner
from axon.strategy.base import Strategy
from axon.strategy.context import (
    TTL_OPERATOR_CEILING,
    URGENCY_TAKE,
    StrategyContext,
)
from axon.strategy.events import Bbo

#: The probe's three states, in the order it moves through them. Named because
#: "which decision has already been taken" is the only state it has, and a bare
#: integer in a log line does not say whether the flatten went out.
WARMING = "warming"
HOLDING = "holding"
DONE = "done"


class TargetProbe(Strategy):
    """Open a position of ``size``, hold it ``hold_ms`` of event time, flatten, stop.

    ``size`` is in **real units** (``0.0002`` BTC), because
    :meth:`~axon.strategy.context.StrategyContext.emit_target` is the one place the
    ``10**8`` wire scaling happens and a strategy that pre-scales is a strategy
    whose position is 100 000 000× wrong.
    """

    def __init__(
        self,
        symbol_id: int,
        *,
        size: str = "0.0002",
        hold_ms: int = 8_000,
        warmup: int = 10,
        urgency: int = URGENCY_TAKE,
        ttl_ms: int = TTL_OPERATOR_CEILING,
    ) -> None:
        if warmup < 1:
            # Zero would emit before any event has been handled, which is exactly
            # the window in which the core has no clock to age the signal against.
            raise ValueError(f"warmup must be at least 1 quote, got {warmup}")
        self.symbol_id = symbol_id
        self.size = size
        self.hold_ns = int(hold_ms) * 1_000_000
        self.warmup = warmup
        self.urgency = urgency
        self.ttl_ms = ttl_ms
        self.on_reset()

    def on_reset(self) -> None:
        self.state = WARMING
        self.quotes = 0
        self.opened_ns: int | None = None
        #: Every decision, in order, as plain data — the evidence chain's Python end.
        self.decisions: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        """Whether the flatten has been emitted. The driver's stop condition."""
        return self.state == DONE

    def on_bbo(self, bbo: Bbo, ctx: StrategyContext) -> None:
        self.quotes += 1
        if self.state == WARMING:
            if self.quotes < self.warmup:
                return
            rec = ctx.emit_target(
                self.symbol_id,
                self.size,
                urgency=self.urgency,
                ttl_ms=self.ttl_ms,
            )
            self.opened_ns = bbo.ts_event
            self.state = HOLDING
            self._record("open", rec, bbo)
            return
        if self.state == HOLDING:
            assert self.opened_ns is not None
            # Event time, not elapsed time: a replay of this market must reach the
            # same verdict at the same event, whatever speed it runs at.
            if bbo.ts_event - self.opened_ns < self.hold_ns:
                return
            rec = ctx.emit_close(self.symbol_id, urgency=self.urgency, ttl_ms=self.ttl_ms)
            self.state = DONE
            self._record("close", rec, bbo)

    # ── internals ──

    def _record(self, what: str, rec, bbo: Bbo) -> None:
        self.decisions.append(
            {
                "decision": what,
                "seq": int(rec["seq"]),
                "ts_event": int(rec["ts_event"]),
                "symbol_id": int(rec["symbol_id"]),
                "target_qty": _real(int(rec["target_qty"])),
                # The wire integer as well as its decimal rendering. This line is the
                # Python end of an evidence chain, and the two ends have to be
                # comparable byte for byte: `str(float)` renders 0.00001 as "1e-05",
                # which no log on the other side of the ring will ever say.
                "target_qty_fixed": int(rec["target_qty"]),
                "urgency": int(rec["urgency"]),
                "ttl_ms": int(rec["ttl_ms"]),
                "flags": int(rec["flags"]),
                "model_version": int(rec["model_version"]),
                # The `cloid` the Rust planner will derive from this record (ADR-0014
                # §5). Printed here rather than looked up afterwards because it is the
                # one identifier that ties a Python decision to a venue fill, and
                # deriving it on both sides independently is what makes the match
                # evidence rather than assertion.
                "cloid": f"0x{_cloid_for(rec):032x}",
                # The book the decision was taken against, so the venue's fill price
                # can be judged against what Python could see rather than against a
                # number looked up afterwards.
                "bid_px": _real(bbo.bid_px),
                "ask_px": _real(bbo.ask_px),
                "quotes_seen": self.quotes,
            }
        )


def _real(fixed: int) -> str:
    """A wire integer as an exact decimal string, never through a float.

    ``from_fixed`` returns a ``float`` because features are statistics and float is
    correct there. A *quantity* rendered through one comes out as ``1e-05``, which
    is unreadable next to a venue's ``0.00001`` and, at sizes a real account trades,
    silently wrong in the last digit.
    """
    return str(Decimal(fixed).scaleb(-FIXED_POINT_DECIMALS))


def _cloid_for(rec) -> int:
    """The client order id ``axon_strategy::cloid_for`` will mint for this record.

    A second implementation of a layout, which is normally the thing to avoid — but
    the whole point here is to derive the same id from the same fields *without*
    reading the Rust code's answer, so that the two agreeing means something. Kept
    honest by ``docs/adr/0014`` §5, which is the specification both read.
    """
    tag = 1 << 127
    ts = (int(rec["ts_event"]) & ((1 << 63) - 1)) << 64
    seq = (int(rec["seq"]) & 0xFFFF_FFFF) << 32
    return tag | ts | seq | int(rec["symbol_id"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m axon.live.probe",
        description=__doc__.splitlines()[0],
    )
    p.add_argument("--md-ring", required=True, help="the ring the Rust core publishes into")
    p.add_argument("--ring", required=True, help="the signal ring the Rust core reads")
    p.add_argument("--capacity", type=int, default=1024, help="signal-ring slots; power of two")
    p.add_argument("--symbol-id", type=int, required=True, help="canonical id, via the SymbolMap")
    p.add_argument("--size", default="0.0002", help="target position in real units")
    p.add_argument("--hold-ms", type=int, default=8_000, help="event-time span to hold it for")
    p.add_argument("--warmup", type=int, default=10, help="quotes to see before deciding")
    p.add_argument("--urgency", type=int, default=URGENCY_TAKE, help="0 post-only .. 3 IOC")
    p.add_argument(
        "--ttl-ms",
        type=int,
        default=TTL_OPERATOR_CEILING,
        help="0 = the operator's ceiling (intent.max_signal_age_ms)",
    )
    p.add_argument("--model-version", type=int, default=1)
    p.add_argument("--wait-s", type=float, default=60.0, help="how long to wait for the md ring")
    p.add_argument(
        "--idle-timeout-s",
        type=float,
        default=90.0,
        help="give up if the ring goes silent this long (a dead feed and a quiet "
        "market are the same silence)",
    )
    p.add_argument(
        "--linger-ms",
        type=int,
        default=4_000,
        help="event-time span to keep reading after the flatten, so the fill that "
        "answers it is still observed",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    strategy = TargetProbe(
        args.symbol_id,
        size=args.size,
        hold_ms=args.hold_ms,
        warmup=args.warmup,
        urgency=args.urgency,
        ttl_ms=args.ttl_ms,
    )

    # The signal ring is created FIRST and the market-data ring is waited for
    # second, and the order matters. `RingProducer` zeroes `head` and `tail` on the
    # file it opens, so creating it under a core that has already attached rewinds
    # the indices the consumer is reading against — the consumer then sees its own
    # future as its past. Creating it before the core can attach makes that
    # unreachable rather than unlikely.
    try:
        runner = StrategyRunner(
            strategy,
            ring_path=args.ring,
            capacity=args.capacity,
            model_version=args.model_version,
        )
    except (OSError, ValueError) as e:
        print(f"probe: cannot create the signal ring {args.ring}: {e}", file=sys.stderr)
        return 1

    try:
        feed = MdRingFeed.wait_for_ring(args.md_ring, timeout_s=args.wait_s)
    except MdFeedError as e:
        print(f"probe: {e}", file=sys.stderr)
        runner.close()
        return 1
    feed.symbol_id = args.symbol_id

    print(f"probe: signal ring {args.ring} (capacity {args.capacity})")
    print(f"probe: md ring     {args.md_ring} symbol_id={args.symbol_id}")
    print(f"probe: beacon      {runner.beacon_path}")
    print(
        f"probe: target {args.size} for {args.hold_ms} ms of event time after "
        f"{args.warmup} quotes, urgency {args.urgency}, ttl_ms {args.ttl_ms}"
    )

    stop_after_ns: int | None = None
    exit_code = 0
    try:
        for event in feed.events(idle_timeout_s=args.idle_timeout_s):
            if not runner.started:
                # Start at the first event's own time. There is no "now" in event
                # time, and only the data knows when this session began.
                runner.start(event.ts_event)
            runner.handle(event)
            if strategy.finished:
                # Keep reading past the flatten so the fill it caused is inside the
                # window the operator sees. Bounded on the event clock like
                # everything else here.
                if stop_after_ns is None:
                    stop_after_ns = event.ts_event + args.linger_ms * 1_000_000
                elif event.ts_event >= stop_after_ns:
                    break
    except BackpressureError as e:
        # The Rust consumer stopped draining. Nothing was lost — the outbox still
        # holds every record — but the only fix is on the other side of the ring.
        print(f"probe: backpressure: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        # `stop` runs `on_stop` inside an event scope and beats the beacon cleanly;
        # the probe emits nothing there, so this cannot add an order after the
        # flatten. Skipped when the runner never started, which is what an md ring
        # that produced no usable quote looks like.
        if runner.started:
            runner.stop(runner.last_event_ns)
        stats = runner.stats
        feed.close()
        runner.close()

    print(
        f"probe: slices {feed.stats.slices_read} events {feed.stats.events_yielded} "
        f"no_quote {feed.stats.no_quote} dropped {feed.stats.dropped}"
    )
    print(
        f"probe: emitted {stats.signals_emitted} pushed {stats.signals_pushed} "
        f"backpressure {stats.backpressure_events} in_flight {stats.signals_in_flight}"
    )
    for d in strategy.decisions:
        print(f"probe: decision {json.dumps(d, sort_keys=True)}")
    if not strategy.decisions:
        # The one outcome that is indistinguishable from a healthy run if it is not
        # said out loud: the probe never got far enough to have an opinion.
        print(
            f"probe: NO DECISION - the probe saw {strategy.quotes} usable quotes and "
            f"needed {strategy.warmup}. Nothing was emitted, so nothing could trade.",
            file=sys.stderr,
        )
        exit_code = exit_code or 1
    elif strategy.state != DONE:
        print(
            f"probe: OPEN TARGET LEFT - the probe emitted its open and stopped in "
            f"state {strategy.state!r} without flattening. The venue-side position, "
            "if it filled, is not this process's to close any more.",
            file=sys.stderr,
        )
        exit_code = exit_code or 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
