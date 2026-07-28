"""``python -m axon.strategies`` — climb the ladder and print what happened.

The numbers in ADR-0022 come from this command; it exists so that a reader can
disagree with them by re-running it rather than by taking them on trust. It is
offline by default (the committed fixture), prints every gate report whether it
passed or failed, and exits non-zero only when a gate that can fail did.

Drift is deliberately not an exit condition. A market that moved is not a bug, and
a runner that failed on it would go red for the one thing nobody can fix by editing
code — the alarm belongs to the live monitor, not to a deploy gate.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from axon.models import ModelRegistry
from axon.strategies.data import Candles, fixture_candles, fixture_coins, load_candles
from axon.strategies.perp_bar import PerpBarParams
from axon.strategies.training import climb


def _load(coins: list[str], *, interval: str, cache: bool) -> list[Candles]:
    if cache:
        return [load_candles(coin, interval) for coin in coins]
    return [fixture_candles(coin, interval) for coin in coins]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m axon.strategies", description=__doc__)
    parser.add_argument("--coins", nargs="+", default=None, help="default: every fixture coin")
    parser.add_argument("--interval", default="1h")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="use the downloaded history in data/candles instead of the committed fixture",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--entry-edge", type=float, default=PerpBarParams.entry_edge)
    parser.add_argument(
        "--parity-bars",
        type=int,
        default=None,
        help="start the online feature-parity replay this many bars from the end "
        "(default: the whole history, which is the number worth reporting)",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="model registry root; a throwaway directory unless given, so a report run "
        "cannot quietly mint a version some other run then serves",
    )
    args = parser.parse_args(argv)

    coins = args.coins or list(fixture_coins(args.interval))
    candles = _load(coins, interval=args.interval, cache=args.cache)
    params = PerpBarParams(symbol_id=0, entry_edge=args.entry_edge)

    with tempfile.TemporaryDirectory(prefix="axon-ladder-") as tmp:
        registry = ModelRegistry(Path(args.registry) if args.registry else Path(tmp))
        result = climb(
            candles,
            registry,
            params=params,
            folds=args.folds,
            embargo_bars=args.embargo_bars,
            parity_bars=args.parity_bars,
        )
        print(result.summary())
        if args.registry:
            print(f"registered: {registry.artifact_path(result.artifact.meta.registry_id)}")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised by hand and by the ADR
    sys.exit(main())
