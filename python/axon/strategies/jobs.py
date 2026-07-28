"""The heavy half of the pipeline, expressed as :mod:`axon.compute` jobs.

The walk-forward in :mod:`axon.strategies.training` fits four models over ten
thousand rows and takes under a minute on this box. A *search* does not: forty-eight
hyper-parameter points, each running its own inner walk-forward, is forty-eight
times that — and the version of the question worth asking (more coins, a finer
interval, several horizons) is another order of magnitude on top. That is exactly
the shape ADR-0017 built ``axon.compute`` for: batch, embarrassingly parallel,
nowhere near the latency path.

Nothing here submits anything. These are builders; ``plan()`` is free and always
callable, ``run()`` needs an :class:`~axon.compute.Approval` that only a plan can
mint, and the decision to spend belongs to the operator (ADR-0022 records the
planned cost, not a receipt).

Two things a caller has to do before a real submission, both of which fail late and
confusingly if skipped:

1. **Put the candle cache and ``contracts/schema.toml`` on the volume.** Only the
   ``axon`` package is mounted into the image, so :mod:`axon.contracts` cannot find
   the schema at its usual repo-relative path and every import of Axon fails in the
   container. :data:`SCHEMA_ENV` points it at the copy on the volume.
2. **Check the grid width against the ledger.** A sweep is priced per task, and the
   Cartesian product grows faster than anyone's intuition — :meth:`ComputeJob.n_tasks`
   is worth reading out loud before planning.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from axon.compute import ComputeJob, param_sweep, volume_uri, walk_forward
from axon.compute.artifacts import Artifact
from axon.strategies.remote import SCHEMA_ON_VOLUME

#: The volume holding the inputs: the candle cache and the contract schema. Kept
#: apart from the artifact volume so a job that may only read data cannot be given
#: write access to the place results land.
DATA_VOLUME = "axon-market-data"
ARTIFACT_VOLUME = "axon-artifacts"

#: :mod:`axon.contracts` resolves ``contracts/schema.toml`` relative to the repo
#: root, which does not exist in a container that mounted only the package.
SCHEMA_ENV = "AXON_SCHEMA_PATH"

#: What the image needs on top of the mounted ``axon`` package. Deliberately short:
#: the research plane's heavy extras (onnx, skl2onnx, sklearn) are export-time
#: dependencies and a sweep never exports.
SWEEP_PIP: tuple[str, ...] = ("numpy>=1.24", "xgboost>=2.0")


def _data_uri(interval: str) -> str:
    return volume_uri(DATA_VOLUME, f"candles/{interval}")


def _schema_path() -> str:
    return Artifact(DATA_VOLUME, SCHEMA_ON_VOLUME).path


def _common(interval: str, out_subpath: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "framework": "xgboost",
        "pip": list(SWEEP_PIP),
        "env": {SCHEMA_ENV: _schema_path()},
        "inputs": {"candles": _data_uri(interval)},
        "outputs": {"metrics": volume_uri(ARTIFACT_VOLUME, out_subpath)},
        **kwargs,
    }


def hyperparameter_sweep(
    *,
    coins: Sequence[str] = ("BTC", "ETH"),
    interval: str = "1h",
    grid: Mapping[str, Sequence[Any]] | None = None,
    folds: int = 4,
    inner_folds: int = 3,
    name: str = "axon-perp-bar-sweep",
    max_usd: float = 1.00,
    **kwargs: Any,
) -> ComputeJob:
    """A hyper-parameter search for ``perp_bar``, scored on an inner walk-forward.

    The constants every point needs — which coins, which volume, how many folds —
    ride as **single-element grid axes**, not in ``kwargs``. hwsched materializes
    tasks from ``params`` and drops the spec's ``kwargs`` for a fan-out, so a
    constant passed the obvious way never reaches the function and the container
    raises ``TypeError`` after the image is built.

    The search never sees the outer holdout: :func:`axon.strategies.remote.sweep_point`
    truncates the dataset at the final fold's purge boundary. A sweep judged on the
    block that will later be reported as out-of-sample is not a search, it is a slow
    way of fitting the test set.
    """
    axes: dict[str, Sequence[Any]] = dict(
        grid
        or {
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.02, 0.05, 0.1],
            "min_child_weight": [20, 40, 80],
        }
    )
    axes.update(
        {
            "data": [_data_uri(interval)],
            "coins": [",".join(coins)],
            "interval": [interval],
            "folds": [folds],
            "inner_folds": [inner_folds],
            "artifact": [volume_uri(ARTIFACT_VOLUME, f"perp_bar/sweep/{interval}/")],
        }
    )
    return param_sweep(
        name,
        "axon.strategies.remote:sweep_point",
        axes,
        max_usd=max_usd,
        **_common(interval, f"perp_bar/sweep/{interval}/", **kwargs),
    )


def walk_forward_job(
    *,
    coins: Sequence[str] = ("BTC", "ETH"),
    interval: str = "1h",
    folds: int = 4,
    name: str = "axon-perp-bar-walk-forward",
    max_usd: float = 0.50,
    **kwargs: Any,
) -> ComputeJob:
    """One task per outer walk-forward window.

    An explicit task list rather than a grid, because windows are not a Cartesian
    product: a product of starts and ends enumerates windows that run backwards.
    Each task fits only its own window, so the wall time is the slowest window
    instead of their sum.
    """
    windows = [
        {
            "fold": index,
            "folds": folds,
            "data": _data_uri(interval),
            "coins": ",".join(coins),
            "interval": interval,
            "artifact": volume_uri(ARTIFACT_VOLUME, f"perp_bar/walk_forward/{interval}/"),
        }
        for index in range(folds)
    ]
    return walk_forward(
        name,
        "axon.strategies.remote:fit_window",
        windows,
        max_usd=max_usd,
        **_common(interval, f"perp_bar/walk_forward/{interval}/", **kwargs),
    )


__all__ = [
    "ARTIFACT_VOLUME",
    "DATA_VOLUME",
    "SCHEMA_ENV",
    "SWEEP_PIP",
    "hyperparameter_sweep",
    "walk_forward_job",
]
