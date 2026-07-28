"""``axon.compute`` — offloading Axon's heavy compute to Modal via ``hwsched``.

Phase 5 needs compute this box does not have: hyper-parameter sweeps, walk-forward
backtests over many windows, model training, feature-matrix builds. None of it is
on the latency path — it is batch work that should run wide and then hand back
artifacts. The sibling ``hwsched`` project already decides *where* such work runs
(GPU or CPU, which GPU, how many workers, how to chunk), prices it, checks it
against a budget and submits it to Modal. This package is Axon's consumer of that,
and nothing more (ADR-0017).

The boundary is strict and worth stating twice: **hwsched decides where compute
runs, never whether a model is good enough.** Pass/fail belongs to
:mod:`axon.parity` (``docs/07``). A scheduler that could fail a model would be a
second, invisible gate on production decisions.

Typical use::

    from axon.compute import HwschedClient, param_sweep, volume_uri

    job = param_sweep(
        "sma-sweep",
        "axon.backtest.sweep:run_one",
        {"fast": [5, 10, 20], "slow": [50, 100, 200]},
        framework="vectorbt",
        data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/metrics.parquet")},
        max_usd=0.50,
    )

    client = HwschedClient()
    plan = client.plan(job)          # free, no spend, always callable
    print(plan.summary())            # cost.high + the planner's rationale
    outcome = client.run(job, approval=plan.approve(max_usd=0.50))

Submodules
----------
- :mod:`axon.compute.spec`      — the JobSpec emitter and the four Axon workloads.
- :mod:`axon.compute.client`    — the typed CLI client, JSON contract and exit codes.
- :mod:`axon.compute.artifacts` — Modal Volume URIs and their mount points.
- :mod:`axon.compute.entry`     — the entrypoints that run remotely (CPU and GPU),
  stdlib-only *at import time* so the whole package can be mounted into a bare image.

Everything here is stdlib at import time (``pyyaml`` is imported inside
:meth:`~axon.compute.spec.ComputeJob.to_yaml`) so this package can be mounted into
a bare Modal image without dragging Axon's research dependencies along.
"""

from __future__ import annotations

from axon.compute.artifacts import (
    MOUNT_ROOT,
    VOLUME_SCHEME,
    Artifact,
    ArtifactError,
    mount_path,
    path_for,
    volume_uri,
)
from axon.compute.client import (
    DEFAULT_HOME,
    HOME_ENV,
    PROVIDER_ENV,
    PYTHON_ENV,
    Approval,
    ApprovalError,
    Budget,
    BudgetRefused,
    Cost,
    HwschedClient,
    HwschedError,
    HwschedUnavailable,
    PlanOutcome,
    ProviderFailed,
    RunOutcome,
    SpecRejected,
    Violation,
)
from axon.compute.spec import (
    CORRELATION_ENV,
    JOBSPEC_FIELDS,
    WORKLOADS,
    ComputeJob,
    SpecError,
    feature_etl,
    param_sweep,
    train_model,
    walk_forward,
)

__all__ = [
    "CORRELATION_ENV",
    "DEFAULT_HOME",
    "HOME_ENV",
    "JOBSPEC_FIELDS",
    "MOUNT_ROOT",
    "PROVIDER_ENV",
    "PYTHON_ENV",
    "VOLUME_SCHEME",
    "WORKLOADS",
    "Approval",
    "ApprovalError",
    "Artifact",
    "ArtifactError",
    "Budget",
    "BudgetRefused",
    "ComputeJob",
    "Cost",
    "HwschedClient",
    "HwschedError",
    "HwschedUnavailable",
    "PlanOutcome",
    "ProviderFailed",
    "RunOutcome",
    "SpecError",
    "Violation",
    "feature_etl",
    "mount_path",
    "param_sweep",
    "path_for",
    "train_model",
    "volume_uri",
    "walk_forward",
]
