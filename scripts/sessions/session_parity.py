#!/usr/bin/env python
"""Feature parity for the session that **traded**, computed from its own capture.

The live parity monitor (ADR-0030) attaches to a session's bar ring, and that is the
right shape for a read-only watcher. It cannot be used on a trading session, and the
reason is a property of the ring rather than a limitation of the monitor: **an SPSC
ring has one consumer.** Two readers do not share it, they steal from it — measured on
2026-07-26, when two drainers on one market-data ring each saw about half the records
and each reported the other's reads as drops. The strategy *is* the bar ring's consumer
on a live run, so a monitor attached beside it would take bars away from the thing
placing orders.

So the diff is computed afterwards, from the session's own capture log, over exactly the
bars the session saw. What that gives up against the live monitor is the alarm: it
cannot tell an operator mid-run that parity has broken. What it keeps is the comparison
itself, on the *trading* session rather than on a second one running beside it.

Two things it deliberately does not do:

* **It does not re-derive bar closure.** A bar is closed when a frame carrying a later
  ``open_time`` arrives — the same rule ``crates/axon-runtime/src/mdring.rs`` applies to
  publish onto the bar ring and ``axon.strategies.data.closed_rows`` applies offline. A
  third implementation here would be a third answer to one question.
* **It does not claim the venue printed these bars.** The capture holds what *arrived*.
  Hyperliquid sends no frame at all for a minute in which nothing traded, so a missing
  minute is invisible from this side; reconciling against ``candleSnapshot`` is a
  separate question and a separate read.

Usage::

    .venv/bin/python scripts/sessions/session_parity.py \\
        --capture data/captures/p6-ml-m1.jsonl --symbol-id 3 \\
        --registry data/models-p6-live --model zoo_xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

#: The wire scale every fixed-point field on the ring and in the log uses.
FIXED = Decimal("1e8")


def closed_bars(path: Path, symbol_id: int, interval: str) -> list[dict]:
    """Every candle frame in the log, reduced to the bars that finished.

    Frames repeat: the venue republishes the bar it is still filling, so one bar can
    arrive dozens of times. The last frame for an ``open_time`` is the one that closed
    it, and a bar is only known to be closed once a frame for a *later* ``open_time``
    has been seen — which is why the newest ``open_time`` is dropped.
    """
    latest: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"Candle"' not in line:
                continue
            rec = json.loads(line)
            c = rec.get("Event", {}).get("event", {}).get("Market", {}).get("Candle")
            if c is None or c["symbol_id"] != symbol_id or c["interval"] != interval:
                continue
            latest[c["open_time"]] = c
    if not latest:
        return []
    newest = max(latest)
    return [latest[t] for t in sorted(latest) if t != newest]


#: The venue's own interval name on the wire (`m1`) against the one
#: `axon.strategies.data` keys its table by (`1m`). Two vocabularies for one interval,
#: and the translation is here rather than in either of them: the capture records what
#: Hyperliquid sent, and `Candles` is venue-agnostic.
WIRE_TO_DATA = {"m1": "1m", "m5": "5m", "m15": "15m", "h1": "1h", "h4": "4h", "d1": "1d"}


def to_candles(bars: list[dict], coin: str, interval: str):
    import numpy as np

    from axon.strategies.data import Candles

    def fixed(values: list[str]) -> "np.ndarray":
        # `Decimal` and not `float`: these are prices on the wire scale, and a float
        # round-trip is exactly the quiet difference a parity gate exists to find.
        return np.array([int(Decimal(v) * FIXED) for v in values], dtype=np.int64)

    return Candles(
        coin=coin,
        interval=WIRE_TO_DATA.get(interval, interval),
        ts_event=np.array([b["ts_event"] for b in bars], dtype=np.int64),
        open=fixed([b["open"] for b in bars]),
        high=fixed([b["high"] for b in bars]),
        low=fixed([b["low"] for b in bars]),
        close=fixed([b["close"] for b in bars]),
        volume=fixed([b["volume"] for b in bars]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="session_parity.py", description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--symbol-id", type=int, required=True)
    parser.add_argument("--interval", default="m1")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-position", default="0.0003")
    args = parser.parse_args(argv)

    from axon.models import ModelRegistry
    from axon.strategies.training import feature_parity_gate
    from axon.strategies.zoo import BAR_M1_V1, live_strategy

    bars = closed_bars(Path(args.capture), args.symbol_id, args.interval)
    if not bars:
        print("no closed bars in the capture — nothing to compare")
        return 1
    candles = to_candles(bars, args.coin, args.interval)
    print(f"closed bars in the capture: {len(candles)}  gaps: {candles.gaps}")
    print(f"  first close {int(candles.ts_event[0])}  last close {int(candles.ts_event[-1])}")

    strategy = live_strategy(
        symbol_id=args.symbol_id,
        max_position=Decimal(args.max_position),
        registry=ModelRegistry(Path(args.registry)),
        model=args.model,
    )
    report = feature_parity_gate(
        strategy,
        candles,
        symbol_id=args.symbol_id,
        # Off the strategy, never assumed: a gate diffing the wrong spec's columns
        # would align on event time, find every stamp, and compare the wrong arrays.
        spec=getattr(strategy, "spec", BAR_M1_V1),
    )
    print(report.summary() if hasattr(report, "summary") else report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
