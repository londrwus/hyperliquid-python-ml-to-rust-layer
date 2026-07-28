"""Run the reference strategy onto a signal ring: ``python -m axon.live``.

A **loopback demo**, and with ``--drain`` a self-contained one that proves the bytes
survive the round trip without a venue, a network, or a second process.

**It cannot drive a live session, and that is not a bug in the Rust side.** The
synthetic feed stamps events from ``--start-ns``, which defaults to a fixed instant
in November 2023. ``StrategyContext`` stamps an emitted signal with the event's own
time and the Rust ``SignalReader`` ages that stamp against the core's event clock,
capped at ``intent.max_signal_age_ms`` (2 000 ms) — so every record this writes is
about two and a half years stale the moment a live core reads it, and the session
correctly refuses all of them while reporting a perfectly healthy status line. Pass a
``--start-ns`` near the venue's own clock if you want to watch that path move, and
use :mod:`axon.live.probe` if you want a strategy driven by the *venue's* clock,
which is the only stamp both sides agree about (ADR-0026).

Both event sources timestamp events *themselves*; nothing here reads a wall clock
to stamp anything. The synthetic feed advances its own event time by ``--step-ms``
per tick, so two runs with the same ``--seed`` emit byte-identical signals — which
is the property that makes a replay comparison meaningful at all.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import numpy as np

from axon.contracts import from_fixed, to_fixed
from axon.live.liveness import read_liveness
from axon.live.runner import BackpressureError, StrategyRunner
from axon.signals import RingConsumer
from axon.strategy import MeanReversion, MeanReversionParams, StrategyConfig, Tick

#: tmpfs on Linux — RAM-backed, so the ring never touches a disk (ADR-0006).
_RING_DIR = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
_DEFAULT_RING = str(_RING_DIR / "axon-signals.ring")


def synthetic_ticks(
    *,
    symbol_id: int,
    count: int,
    start_ns: int,
    step_ns: int,
    start_px: float,
    seed: int,
) -> Iterator[Tick]:
    """A mean-reverting (Ornstein–Uhlenbeck) price path with self-generated event times.

    Mean-reverting on purpose: the reference strategy is a mean-reversion toggle,
    and a pure random walk would leave it flat forever, which demos nothing.
    """
    rng = np.random.default_rng(seed)
    theta, sigma = 0.05, start_px * 5e-4
    px = start_px
    for i in range(count):
        px += theta * (start_px - px) + sigma * float(rng.standard_normal())
        # The one place a real price becomes wire fixed-point, as at any feed edge.
        yield Tick(symbol_id=symbol_id, ts_event=start_ns + i * step_ns, px=to_fixed(px))


def stdin_ticks() -> Iterator[Tick]:
    """Ticks as ``ts_event_ns,symbol_id,px_fixed`` CSV on stdin — pipe a real feed in."""
    for lineno, line in enumerate(sys.stdin, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) != 3:
            raise SystemExit(f"stdin line {lineno}: expected ts_event_ns,symbol_id,px_fixed")
        ts, sym, px = (int(p) for p in parts)
        yield Tick(symbol_id=sym, ts_event=ts, px=px)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m axon.live", description=__doc__.splitlines()[0])
    p.add_argument("--ring", default=_DEFAULT_RING, help=f"ring path (default: {_DEFAULT_RING})")
    p.add_argument("--capacity", type=int, default=1024, help="ring slots; power of two")
    p.add_argument("--symbol-id", type=int, default=0)
    p.add_argument("--source", choices=("synthetic", "stdin"), default="synthetic")
    p.add_argument("--ticks", type=int, default=500, help="synthetic tick count")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start-px", type=float, default=60_000.0)
    p.add_argument(
        "--start-ns",
        type=int,
        default=1_700_000_000_000_000_000,
        help="event time of the first synthetic tick. The default is a FIXED instant in "
        "2023, which keeps the demo reproducible and makes every signal it writes far "
        "too stale for a live core to admit — set it near the venue's clock, or use "
        "axon.live.probe, to drive a real session",
    )
    p.add_argument("--step-ms", type=int, default=100)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--entry-z", type=float, default=1.5)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--max-position", default="0.01", help="real units, e.g. 0.01 BTC")
    p.add_argument("--model-version", type=int, default=1)
    p.add_argument("--ttl-ms", type=int, default=500)
    p.add_argument(
        "--drain",
        action="store_true",
        help="consume the ring in-process (loopback demo when no Rust core is attached)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config = StrategyConfig(
        name="reference-mean-reversion",
        model_version=args.model_version,
        symbols=(args.symbol_id,),
        default_ttl_ms=args.ttl_ms,
        params={
            "window": args.window,
            "entry_z": args.entry_z,
            "exit_z": args.exit_z,
            "max_position": args.max_position,
        },
    )
    strategy = MeanReversion(MeanReversionParams.from_config(config))

    events = (
        stdin_ticks()
        if args.source == "stdin"
        else synthetic_ticks(
            symbol_id=args.symbol_id,
            count=args.ticks,
            start_ns=args.start_ns,
            step_ns=args.step_ms * 1_000_000,
            start_px=args.start_px,
            seed=args.seed,
        )
    )

    runner = StrategyRunner.from_config(
        strategy, config, ring_path=args.ring, capacity=args.capacity
    )
    beacon_path = runner.beacon_path
    try:
        stats = runner.run(events)
    except BackpressureError as e:
        # Not a crash: every record is still queued. Surfaced loudly because the
        # only real fix is on the consumer side.
        print(f"backpressure: {e}", file=sys.stderr)
        return 1
    finally:
        runner.close()

    # Said after the run rather than before, because the numbers above are what an
    # operator is looking at when they wonder why the Rust side counted nothing.
    if args.source == "synthetic" and not args.drain:
        stale_s = (time.time_ns() - args.start_ns) / 1e9
        if abs(stale_s) > 60:
            days = abs(stale_s) / 86400
            when = "in the past" if stale_s > 0 else "in the future"
            print(
                f"warning: the synthetic feed stamped its events {days:.0f} days {when}, "
                "so a live Rust core will refuse every one of these signals as expired "
                "and still report a healthy status line. Pass --start-ns near the venue's "
                "clock, or use `python -m axon.live.probe`, which takes its clock from "
                "the market-data ring the core publishes.",
                file=sys.stderr,
            )

    print(f"ring     {args.ring} (capacity {args.capacity})")
    print(f"beacon   {beacon_path}")
    print(
        f"events   {stats.events_handled}  signals {stats.signals_emitted}  "
        f"pushed {stats.signals_pushed}  backpressure {stats.backpressure_events}  "
        f"beats {stats.beats}"
    )
    snap = read_liveness(beacon_path)
    print(
        f"liveness beats={snap.beats} last_event_ns={snap.last_event_ns} "
        f"running={snap.running} stopped={snap.stopped}"
    )

    if args.drain:
        with RingConsumer(args.ring) as cons:
            first = last = None
            n = 0
            while (rec := cons.try_pop()) is not None:
                first = rec if first is None else first
                last, n = rec, n + 1
            print(f"drained  {n} signal(s)")
            for label, rec in (("first", first), ("last", last)):
                if rec is not None:
                    print(
                        f"  {label}: seq={int(rec['seq'])} ts_event={int(rec['ts_event'])} "
                        f"symbol={int(rec['symbol_id'])} "
                        f"target={Decimal(str(from_fixed(int(rec['target_qty']))))} "
                        f"ttl_ms={int(rec['ttl_ms'])} model_version={int(rec['model_version'])}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
