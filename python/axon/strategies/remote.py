"""The functions hwsched runs *inside* the container. Not stubs — the real work.

Two entrypoints, one per shape of offloaded job (:mod:`axon.strategies.jobs`
builds the specs that name them):

* :func:`sweep_point` — one hyper-parameter point, scored by an inner walk-forward.
* :func:`fit_window` — one outer walk-forward window, fitted and scored.

Three constraints shape both, and each one is a way a Modal job fails at the far
end of a five-minute image build:

**The candles arrive on a Volume, not in the payload.** hwsched materializes one
task per point and pickles the return value back through Modal's control plane, so
anything of size travels through a mounted volume (ADR-0017) and what comes back is
a receipt. The job declares ``inputs={"candles": volume://…}``; this code reads
ordinary paths under the mount that :func:`axon.compute.path_for` resolves.

**A fan-out drops ``kwargs``.** hwsched materializes tasks from ``params``/``tasks``
and never merges the spec's ``kwargs`` into them, so a constant every point needs —
which coins, which volume — has to arrive as a **single-element grid axis** or as a
default here. It cannot ride in ``kwargs``, and the symptom of getting that wrong is
a ``TypeError`` in the container after the image has been built and paid for.

**Only the ``axon`` package is mounted.** hwsched calls
``Image.add_local_python_source("axon")``, so the repository around the package —
including ``contracts/schema.toml``, which :mod:`axon.contracts` resolves relative
to the repo root — is not there. Every job that reaches this module must therefore
set ``AXON_SCHEMA_PATH`` to a copy of the schema on the volume; without it the
import fails before a single row is read. ``test_strategies.py`` proves that by
importing the package from a tree with nothing but ``axon`` in it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from axon.compute.artifacts import VOLUME_SCHEME, path_for
from axon.strategies.data import Candles, cache_path
from axon.strategies.perp_bar import LABEL_HORIZON_BARS

#: Where the schema has to be on the volume for :mod:`axon.contracts` to find it.
#: Named here so the job spec and the container agree on one string.
SCHEMA_ON_VOLUME = "schema.toml"


def _resolve(location: str) -> Path:
    """A ``volume://`` URI or a plain directory, resolved to a path.

    Both, deliberately: an entrypoint that can only be exercised inside a Modal
    container is an entrypoint nobody runs before paying to find out it is wrong.
    Locally it points at ``data/candles``; remotely at the mount hwsched made.
    """
    return Path(path_for(location)) if location.startswith(VOLUME_SCHEME) else Path(location)


def read_candles(data_uri: str, coins: str | Sequence[str], interval: str) -> list[Candles]:
    """Load the cached CSVs for ``coins`` from a mounted volume directory.

    ``coins`` is accepted as a comma-separated string because a grid axis whose
    values are lists is legal but reads like a mistake, and one spelling of "which
    coins" beats two.
    """
    names = coins.split(",") if isinstance(coins, str) else list(coins)
    root = _resolve(data_uri)
    out = []
    for coin in (c.strip().upper() for c in names if c.strip()):
        path = cache_path(coin, interval, root=root)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is not on the mounted volume; upload the candle cache before "
                "submitting, or the job burns its image build to discover this"
            )
        out.append(Candles.from_csv(path, coin=coin, interval=interval))
    if not out:
        raise ValueError(f"no coins parsed from {coins!r}")
    return out


def _write_receipt(artifact: str | None, name: str, payload: Mapping[str, Any]) -> None:
    """Write one task's result under ``artifact``, which is always a *directory*.

    A fan-out of thirty-six tasks pointed at one file is thirty-six tasks
    overwriting each other, and the surviving result is whichever container
    finished last — a sweep that reports one point and looks like it worked.
    """
    if not artifact:
        return
    path = _resolve(artifact) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def sweep_point(
    max_depth: int = 3,
    learning_rate: float = 0.05,
    min_child_weight: int = 40,
    n_estimators: int = 200,
    *,
    data: str = "",
    coins: str = "BTC,ETH",
    interval: str = "1h",
    folds: int = 4,
    inner_folds: int = 3,
    embargo_bars: int = 0,
    artifact: str | None = None,
) -> dict[str, Any]:
    """Score one hyper-parameter point on an inner walk-forward. Never the holdout.

    The point of doing this remotely is the grid, and the point of doing it *this*
    way is that a sweep is a selection procedure: choosing the best point by a
    number computed on the final holdout is fitting the holdout, one comparison at
    a time. So the search sees only the rows the final outer fold was allowed to
    train on — the outer purge boundary is the wall — and the outer holdout stays
    untouched for the run that reports a result.
    """
    from dataclasses import asdict

    from axon.strategies.data import INTERVAL_MS
    from axon.strategies.labels import purged_walk_forward
    from axon.strategies.training import build_dataset, walk_forward_fit

    dataset = build_dataset(read_candles(data, coins, interval), horizon=LABEL_HORIZON_BARS)
    bar_ns = INTERVAL_MS[interval] * 1_000_000
    outer = purged_walk_forward(
        dataset.ts_event,
        folds=folds,
        horizon_ns=dataset.horizon_ns,
        embargo_ns=embargo_bars * bar_ns,
    )
    inner = walk_forward_fit(
        dataset.select(outer[-1].train),
        folds=inner_folds,
        embargo_bars=embargo_bars,
        params={
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_child_weight": int(min_child_weight),
            "n_estimators": int(n_estimators),
        },
    )
    receipt = {
        "point": {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "min_child_weight": int(min_child_weight),
            "n_estimators": int(n_estimators),
        },
        "coins": coins,
        "interval": interval,
        "n_rows": len(dataset),
        "n_validation_rows": int(outer[-1].train.size),
        "auc": inner.evaluation.auc,
        "hit_rate": inner.evaluation.hit_rate,
        "gross_edge_bps": inner.evaluation.gross_edge_bps,
        "coverage": inner.evaluation.coverage,
        "folds": [asdict(f) for f in inner.evaluation.folds],
        "correlation_id": os.environ.get("AXON_CORRELATION_ID"),
        "spec_ref": dataset.spec_ref,
    }
    _write_receipt(
        artifact,
        f"point-{max_depth}-{learning_rate}-{min_child_weight}-{n_estimators}.json",
        receipt,
    )
    return receipt


def fit_window(
    *,
    fold: int = 0,
    folds: int = 4,
    data: str = "",
    coins: str = "BTC,ETH",
    interval: str = "1h",
    embargo_bars: int = 0,
    artifact: str | None = None,
) -> dict[str, Any]:
    """Fit and score one outer walk-forward window.

    The fan-out shape for a longer history than this box can hold: each task owns
    one window, so the wall time is the slowest window rather than their sum. The
    windows are an explicit task list and not a grid, because a Cartesian product of
    starts and ends enumerates windows that run backwards.
    """
    from dataclasses import asdict

    from axon.strategies.data import INTERVAL_MS
    from axon.strategies.labels import purged_walk_forward
    from axon.strategies.training import build_dataset, fit_fold

    dataset = build_dataset(read_candles(data, coins, interval), horizon=LABEL_HORIZON_BARS)
    splits = purged_walk_forward(
        dataset.ts_event,
        folds=folds,
        horizon_ns=dataset.horizon_ns,
        embargo_ns=embargo_bars * INTERVAL_MS[interval] * 1_000_000,
    )
    if not 0 <= fold < len(splits):
        raise IndexError(f"fold {fold} is outside the {len(splits)} windows")
    _, _, result = fit_fold(dataset, splits[fold])
    receipt = {
        "fold": int(fold),
        "coins": coins,
        "interval": interval,
        "spec_ref": dataset.spec_ref,
        "correlation_id": os.environ.get("AXON_CORRELATION_ID"),
        **asdict(result),
    }
    _write_receipt(artifact, f"fold-{fold}.json", receipt)
    return receipt


__all__ = ["SCHEMA_ON_VOLUME", "fit_window", "read_candles", "sweep_point"]
