"""Turning an Axon workload into an hwsched ``JobSpec``.

hwsched's ``JobSpec`` is a pydantic model with ``extra="forbid"``: one unknown key
and the whole spec is rejected at parse time, after the YAML has been written and
the subprocess spawned. :data:`JOBSPEC_FIELDS` mirrors that model so a bad key is
caught in-process, in a unit test, instead of as an exit-3 from a CLI — and
``test_compute.py`` asks the real model for its field set so the mirror cannot
drift silently.

The four Axon shapes that are worth offloading each map onto a workload hwsched's
planner already knows how to size (``hwsched/docs/07-workload-hardware-map.md``):

===========================  ===================  ==================================
Axon workload                hwsched ``workload``  planner behaviour
===========================  ===================  ==================================
hyper-parameter sweep        ``param_sweep``      CPU fan-out over the Cartesian grid
walk-forward backtest        ``walk_forward``     CPU fan-out over explicit windows
neural-net training          ``ml_train_dnn``     forced GPU
tree training                ``ml_train_trees``   CPU unless pinned
feature-matrix build         ``etl``              big-memory CPU
===========================  ===================  ==================================

**Declare ``workload`` and ``framework``; never rely on hwsched inferring them.**
Its profiler infers device from the entrypoint module's imports, but it resolves
that module relative to *its own* working directory — and we run it from the
hwsched checkout, not from Axon's. So the static-introspection stage always misses
Axon's source and the declared signals are the only ones the planner ever sees. A
training job that forgets ``workload: ml_train_dnn`` silently plans as CPU.

## Correlation

``JobSpec`` has no ``metadata``/``tags`` field and forbids extras, so an Axon
correlation id has to ride in ``args``, ``kwargs``, ``env`` or ``idempotency_key``.
It rides in ``env`` (:data:`CORRELATION_ENV`), because:

* ``kwargs`` is passed straight through to the user's function — and for a fan-out
  it is *dropped entirely* (hwsched materializes tasks from ``params``/``tasks``),
  so the id would reach a single-task job and vanish from a sweep;
* ``args`` is never read by the Modal adapter at all;
* ``idempotency_key`` is load-bearing for dedup — overwriting it with a run id
  would disable "don't re-run an identical completed job" outright;
* ``env`` reaches the container as a real environment variable *and* is folded
  into hwsched's derived idempotency key.

That last point is the trap. Because ``env`` feeds the dedup hash, a per-call
UUID would give every submission a fresh key and quietly turn the idempotency
cache off. So the id is **content-addressed**: derived from exactly the fields
hwsched itself hashes, which means it changes when and only when hwsched's own key
would have changed. Pass ``correlation_id=`` explicitly when you *want* a distinct
run of identical work.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon.compute.artifacts import try_parse

#: Environment variable carrying the Axon correlation id into the container and
#: into hwsched's persisted spec, where a parity report can join against it.
CORRELATION_ENV = "AXON_CORRELATION_ID"

#: Every field ``hwsched.spec.JobSpec`` accepts. Anything else is an exit-3.
JOBSPEC_FIELDS: frozenset[str] = frozenset(
    {
        "name", "entrypoint", "args", "kwargs", "image", "pip", "env", "secrets",
        "workload", "framework", "precision",
        "params", "tasks", "n_tasks", "chunk_size", "max_parallel",
        "data", "resources",
        "objective", "deadline", "budget", "priority", "preemptible", "retries",
        "timeout", "region", "provider",
        "schedule", "depends_on", "outputs", "inputs", "on_failure", "idempotency_key",
    }
)

#: hwsched's ``WorkloadType`` enum.
WORKLOADS: frozenset[str] = frozenset(
    {
        "vectorized_backtest", "event_backtest", "param_sweep", "walk_forward",
        "monte_carlo", "portfolio_opt", "ml_train_trees", "ml_train_dnn",
        "ml_train_rl", "feature_eng", "etl", "inference", "generic",
    }
)

#: Keys accepted inside ``resources`` (hwsched's ``ResourceOverrides``). These are
#: hard pins that override the planner, so a typo must not be silently ignored.
RESOURCE_KEYS: frozenset[str] = frozenset(
    {"device", "gpu_type", "gpu_count", "cpu", "memory_gb", "disk_gb"}
)

#: Frameworks whose training runs belong on a GPU, and those that do not. Used
#: only to pick a default workload for :func:`train_model`; an explicit
#: ``workload=`` always wins.
_GPU_FRAMEWORKS = frozenset({"torch", "pytorch", "tensorflow", "jax", "keras", "transformers"})
_TREE_FRAMEWORKS = frozenset({"xgboost", "lightgbm", "catboost", "sklearn", "scikit-learn"})

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][\w.]*(?:\.py)?:[A-Za-z_]\w*$")


class SpecError(ValueError):
    """A job that hwsched would reject, caught before the subprocess."""


@dataclass(frozen=True)
class ComputeJob:
    """One Axon unit of offloaded work, in hwsched's vocabulary.

    Construct via :func:`param_sweep`, :func:`walk_forward`, :func:`train_model`
    or :func:`feature_etl` unless you need a workload none of them cover.
    """

    name: str
    entrypoint: str
    workload: str

    # fan-out: a Cartesian grid, or an explicit task list — never both
    params: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    tasks: Sequence[Any] | None = None
    chunk_size: int | None = None
    max_parallel: int | None = None

    # what the container needs
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    pip: Sequence[str] = ()
    secrets: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    image: str | None = None

    # sizing signals
    framework: str | None = None
    data_size_gb: float = 0.0
    data_rows: int | None = None
    data_source: str | None = None
    resources: Mapping[str, Any] = field(default_factory=dict)

    # artifacts — both sides of the Volume hand-off
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)

    # policy
    objective: str | None = None
    deadline: str | None = None
    timeout: str | None = None
    max_usd: float | None = None
    priority: int = 0
    retries: int = 2
    region: str | None = None
    preemptible: bool | None = None

    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", self.name):
            raise SpecError(
                f"job name {self.name!r} must be alphanumeric with . _ -; hwsched derives "
                "the Modal app name from it and sanitizes anything else away, which would "
                "merge two different jobs into one app"
            )
        if not _ENTRYPOINT.match(self.entrypoint):
            raise SpecError(
                f"entrypoint {self.entrypoint!r} must be 'module.path:function' — the "
                "container resolves it with importlib, and its top-level package is what "
                "gets mounted into the image"
            )
        if self.workload not in WORKLOADS:
            raise SpecError(
                f"unknown workload {self.workload!r}; hwsched accepts {sorted(WORKLOADS)}"
            )
        if self.params and self.tasks is not None:
            raise SpecError("specify either params (a grid) or tasks (an explicit list), not both")
        for key in self.resources:
            if key not in RESOURCE_KEYS:
                raise SpecError(
                    f"unknown resource pin {key!r}; hwsched accepts {sorted(RESOURCE_KEYS)}"
                )
        for key in self.env:
            if not _ENV_NAME.match(key):
                raise SpecError(f"invalid environment variable name {key!r}")
        if CORRELATION_ENV in self.env:
            raise SpecError(
                f"{CORRELATION_ENV} is set by axon.compute; pass correlation_id= instead of "
                "writing it into env, or the id and the digest it is derived from disagree"
            )
        for label, mapping in (("inputs", self.inputs), ("outputs", self.outputs)):
            for name, uri in mapping.items():
                if try_parse(uri) is None:
                    raise SpecError(
                        f"{label}[{name!r}] = {uri!r} is not a volume:// URI. Artifacts move "
                        "through Modal Volumes, never through return values: the CLI path "
                        "never collects results, so anything not on a volume is discarded "
                        "when the container exits"
                    )
        if self.data_source is not None and "://" not in self.data_source:
            raise SpecError(
                f"data_source {self.data_source!r} needs a scheme (volume://, s3://, "
                "local://); hwsched infers data locality from it and defaults to 'auto' "
                "otherwise, which mis-sizes the upload allowance"
            )
        if self.max_usd is not None and self.max_usd <= 0:
            raise SpecError("max_usd must be positive")

    # ── identity ──

    @property
    def digest(self) -> str:
        """Content address of the work: what is computed, not where or how fast.

        Covers exactly the fields hwsched folds into its own idempotency key, so
        the correlation id this feeds changes if and only if hwsched's key would
        have changed anyway — i.e. carrying the id costs nothing in dedup.
        ``outputs`` is the one addition: hwsched omits it, so two runs writing
        different artifacts hash identically and the second is skipped as a
        duplicate. Including it here closes that, since ``env`` feeds hwsched's key.
        """
        payload = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "workload": self.workload,
            "framework": self.framework,
            "args": {},
            "kwargs": dict(self.kwargs),
            "params": {k: list(v) for k, v in self.params.items()},
            "tasks": list(self.tasks) if self.tasks is not None else None,
            "image": self.image,
            "pip": sorted(self.pip),
            "env": dict(self.env),
            "secrets": sorted(self.secrets),
            "data": {
                "size_gb": self.data_size_gb,
                "rows": self.data_rows,
                "source": self.data_source,
            },
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def resolved_correlation_id(self) -> str:
        """The id that rides in ``env``. Explicit if given, else content-addressed."""
        return self.correlation_id or f"axon.{self.name}.{self.digest}"

    @property
    def n_tasks(self) -> int:
        """Fan-out width, mirroring hwsched's own derivation."""
        if self.tasks is not None:
            return len(self.tasks)
        n = 1
        for values in self.params.values():
            n *= max(1, len(values))
        return n

    # ── emission ──

    def to_mapping(self) -> dict[str, Any]:
        """The JobSpec as a plain mapping, with defaults omitted.

        Only keys hwsched declares are emitted, and the result is asserted against
        :data:`JOBSPEC_FIELDS` — under ``extra="forbid"`` a stray key is not a
        warning, it is a rejected spec.
        """
        spec: dict[str, Any] = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "workload": self.workload,
        }
        if self.framework:
            spec["framework"] = self.framework
        if self.params:
            spec["params"] = {k: list(v) for k, v in self.params.items()}
        if self.tasks is not None:
            spec["tasks"] = list(self.tasks)
        if self.kwargs:
            spec["kwargs"] = dict(self.kwargs)
        if self.chunk_size is not None:
            spec["chunk_size"] = self.chunk_size
        if self.max_parallel is not None:
            spec["max_parallel"] = self.max_parallel
        if self.pip:
            spec["pip"] = list(self.pip)
        if self.secrets:
            spec["secrets"] = list(self.secrets)
        if self.image:
            spec["image"] = self.image

        env = dict(self.env)
        env[CORRELATION_ENV] = self.resolved_correlation_id
        spec["env"] = env

        data: dict[str, Any] = {}
        if self.data_size_gb:
            data["size_gb"] = float(self.data_size_gb)
        if self.data_rows is not None:
            data["rows"] = self.data_rows
        if self.data_source:
            data["source"] = self.data_source
        if data:
            spec["data"] = data

        if self.resources:
            spec["resources"] = dict(self.resources)
        if self.inputs:
            spec["inputs"] = dict(self.inputs)
        if self.outputs:
            spec["outputs"] = dict(self.outputs)

        if self.objective:
            spec["objective"] = self.objective
        if self.deadline:
            spec["deadline"] = self.deadline
        if self.timeout:
            spec["timeout"] = self.timeout
        if self.max_usd is not None:
            spec["budget"] = {"max_usd": float(self.max_usd)}
        if self.priority:
            spec["priority"] = self.priority
        if self.retries != 2:
            spec["retries"] = self.retries
        if self.region:
            spec["region"] = self.region
        if self.preemptible is not None:
            spec["preemptible"] = self.preemptible

        unknown = set(spec) - JOBSPEC_FIELDS
        if unknown:  # pragma: no cover - a guard against future edits to this method
            raise SpecError(f"emitted keys hwsched forbids: {sorted(unknown)}")
        return spec

    def to_yaml(self) -> str:
        """The JobSpec as YAML. ``pyyaml`` is imported here, not at module scope,
        so importing :mod:`axon.compute` inside a Modal container — which has no
        Axon dependencies installed — still works."""
        import yaml

        return yaml.safe_dump(self.to_mapping(), sort_keys=False, default_flow_style=False)

    def write(self, path: str | Path) -> Path:
        """Write the spec YAML to *path* and return it."""
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


# --------------------------------------------------------------------------- #
# the four Axon shapes
# --------------------------------------------------------------------------- #
def param_sweep(
    name: str, entrypoint: str, grid: Mapping[str, Sequence[Any]], **kwargs: Any
) -> ComputeJob:
    """A hyper-parameter sweep: one task per point of the Cartesian grid.

    The grid keys become the entrypoint function's arguments — hwsched calls
    ``fn(**point)`` — so they must match its signature exactly.
    """
    if not grid:
        raise SpecError("a param sweep needs a non-empty grid")
    return ComputeJob(
        name=name, entrypoint=entrypoint, workload="param_sweep", params=grid, **kwargs
    )


def walk_forward(
    name: str, entrypoint: str, windows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> ComputeJob:
    """A walk-forward backtest: one task per train/test window.

    Windows are an explicit task list rather than a grid because they are not a
    Cartesian product — a product of starts and ends would enumerate windows that
    run backwards.
    """
    if not windows:
        raise SpecError("a walk-forward run needs at least one window")
    return ComputeJob(
        name=name, entrypoint=entrypoint, workload="walk_forward",
        tasks=list(windows), **kwargs,
    )


def train_model(
    name: str, entrypoint: str, *, framework: str, workload: str | None = None, **kwargs: Any
) -> ComputeJob:
    """A model-training run, sized by framework unless *workload* overrides it.

    Trees stay on CPU (``ml_train_trees``) and nets go to a GPU
    (``ml_train_dnn``); an unrecognized framework is left to the caller rather
    than guessed, because guessing GPU is the expensive direction to be wrong in.
    """
    if workload is None:
        key = framework.strip().lower()
        if key in _TREE_FRAMEWORKS:
            workload = "ml_train_trees"
        elif key in _GPU_FRAMEWORKS:
            workload = "ml_train_dnn"
        else:
            raise SpecError(
                f"cannot infer a workload for framework {framework!r}; pass workload= "
                "explicitly (ml_train_trees stays on CPU, ml_train_dnn forces a GPU)"
            )
    return ComputeJob(
        name=name, entrypoint=entrypoint, workload=workload, framework=framework, **kwargs
    )


def feature_etl(
    name: str, entrypoint: str, *, workload: str = "etl", **kwargs: Any
) -> ComputeJob:
    """A feature-matrix build: big-memory CPU, output written to a Volume.

    Use ``workload="feature_eng"`` for a transform-heavy build; ``etl`` is the
    IO/memory-dominated default.
    """
    return ComputeJob(name=name, entrypoint=entrypoint, workload=workload, **kwargs)


__all__ = [
    "CORRELATION_ENV",
    "JOBSPEC_FIELDS",
    "RESOURCE_KEYS",
    "WORKLOADS",
    "ComputeJob",
    "SpecError",
    "feature_etl",
    "param_sweep",
    "train_model",
    "walk_forward",
]
