#!/usr/bin/env python
"""Fit the zoo and leave one family in a **persistent** registry, for a live run.

``python -m axon.strategies`` mints its artifacts into a temporary directory unless
``--registry`` is given, and the model zoo (ADR-0032) ran entirely inside
``pytest``'s ``tmp_path``. Both are right for a report: a run that can quietly mint a
version some other run then *serves* is how a live session ends up trading a model
nobody chose. This script is the deliberate opposite — it exists to mint exactly
that version, names the directory on the command line, and prints the four gates so
the artifact that reaches the venue arrives with its transcript attached.

It is **not** a second recipe. Every number comes from :mod:`axon.strategies.zoo`;
this file chooses coins, a registry root, and which families to run. A second
definition of ``BAR_M1_V1``, of the walk-forward or of the entry band would be a
second answer to the question the parity gates exist to ask.

Usage::

    .venv/bin/python scripts/sessions/fit_zoo_live.py \\
        --registry data/models-p6-live --coins BTC ETH SOL --refresh

``--refresh`` re-fetches the m1 cache from the venue's ``candleSnapshot``. It is off
by default for the reason :func:`axon.strategies.data.load_candles` gives — a
research run that silently extends its own sample moves every number in the ADR —
but a model about to trade *today* wants today's tape, so a live fit passes it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fit_zoo_live.py", description=__doc__)
    parser.add_argument("--registry", required=True, help="registry root to mint into")
    parser.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL"])
    parser.add_argument(
        "--families",
        nargs="+",
        default=["xgboost"],
        help="zoo family names. The default is the one family that crosses into Rust "
        "bit-exact (ADR-0032), which is the only reason to prefer it here",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--refresh", action="store_true", help="re-fetch the m1 cache")
    parser.add_argument(
        "--cache-root",
        default=None,
        help="where the m1 CSVs live. Give a separate root when fetching from a "
        "network other than the one data/candles was filled from — a testnet fetch "
        "written over a mainnet cache changes the corpus of every other run in the "
        "tree, silently, and nothing in a CSV records which venue it came from",
    )
    parser.add_argument(
        "--bundles",
        default=None,
        help="write the ADR-0021 cross-language bundle for each family here",
    )
    parser.add_argument("--json-out", default=None, help="append a machine-readable summary")
    args = parser.parse_args(argv)

    from axon.models import ModelRegistry
    from axon.strategies.data import load_candles
    from axon.strategies.zoo import INTERVAL, family, run_family

    root = None if args.cache_root is None else Path(args.cache_root)
    candles = [load_candles(c, INTERVAL, refresh=args.refresh, root=root) for c in args.coins]
    rows = sum(len(c) for c in candles)
    print(f"candles: {len(candles)} coins, {rows} {INTERVAL} bars, root={root or 'default'}")
    for c in candles:
        # The last close time and the gap count, because a model fitted on a stale
        # cache and served on a live socket is the failure this print prevents for
        # free — and `gaps` is the one property of the corpus the walk-forward cannot
        # see, since a missing bar is simply a row that is not there.
        last = int(c.ts_event[-1]) // 1_000_000
        print(f"  {c.coin:5s} {len(c):6d} bars, {c.gaps} gaps, last close {last} ms")

    registry = ModelRegistry(Path(args.registry))
    summaries = []
    for name in args.families:
        fam = family(name)
        t0 = time.time()
        result = run_family(
            fam,
            candles,
            registry,
            folds=args.folds,
            bundle_dir=None if args.bundles is None else Path(args.bundles) / name,
        )
        print(f"\n{'=' * 78}\n{name}  ({time.time() - t0:.1f}s)\n{'=' * 78}")
        print(result.summary())
        meta = result.artifact.meta
        path = registry.artifact_path(meta.registry_id, meta.version)
        print(f"registered: {meta.registry_id} v{meta.version} -> {path}")
        summaries.append(
            {
                "family": name,
                "registry_id": meta.registry_id,
                "version": meta.version,
                "kind": meta.kind,
                "path": str(path),
                "coins": args.coins,
                "bars": rows,
                "bundle_error": result.bundle_error,
            }
        )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "a", encoding="utf-8") as fh:
            for s in summaries:
                fh.write(json.dumps(s) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
