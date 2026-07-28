"""A typed client for the hwsched CLI.

hwsched exposes no Rust binding, no HTTP surface and no gRPC — it is a Python
library plus a CLI, and it is **not pip-installed** on this box. So the integration
is a subprocess: ``<python> -m hwsched <cmd> --json`` run with the hwsched checkout
on ``PYTHONPATH`` and as the working directory. Both matter. ``PYTHONPATH`` is what
makes ``-m hwsched`` resolve at all; the working directory is what makes it find
``hwsched.toml`` and its run store, because :func:`hwsched.config.load_config`
searches ``Path.cwd()``. Point it somewhere else and the job runs against default
config and an empty ledger — i.e. against a budget that has forgotten every dollar
already committed.

## Dry run before spend

hwsched's own first principle, and the one this module enforces mechanically
rather than by convention:

* :meth:`HwschedClient.plan` is free, never submits, and is always callable —
  including when the budget guard refuses, which is a *result*, not an error.
* :meth:`HwschedClient.run` will not run without an :class:`Approval`, and an
  ``Approval`` can only be minted by :meth:`PlanOutcome.approve`, which requires
  the caller to name a ceiling at or above the plan's own ``cost.high``. Stating
  that number is the mechanical form of "the estimate was surfaced".
* The approval is bound to the job's content digest and is single-use, so a plan
  for a 16-point CPU sweep cannot authorize the A100 job someone edited it into,
  and a retry loop cannot re-spend one approval N times.

## Budget refusals are configuration, not catastrophe

Exit code 2 means the guard, *as configured for the currently active Modal
profile*, would not admit the plan. With fourteen profiles in ``~/.modal.toml``
that is a statement about one ledger, not about whether the work is affordable.
So a refusal is surfaced with everything needed to act on it — ``decision``,
``remaining``, ``cap``, the guard's message and the plan's ``cost.high`` — and
never routed around. The one sanctioned lever is
``HwschedClient(monthly_budget_usd=…)``, which sets hwsched's own
``HWSCHED_MONTHLY_USD`` override for the subprocess: it *raises the cap the guard
enforces*, it does not disable the guard, it is opt-in per client, and it is
echoed on every outcome so it can never be in force unnoticed. ``allow_overage``
is deliberately not exposed, and this module never writes to ``hwsched.toml``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axon.compute.spec import ComputeJob

log = logging.getLogger("axon.compute")

#: Where hwsched lives when nothing says otherwise. Overridable so a checkout
#: elsewhere (or a CI image) does not need a code change.
DEFAULT_HOME = str(Path.home() / "hardware-scheduler")

HOME_ENV = "AXON_HWSCHED_HOME"
PYTHON_ENV = "AXON_HWSCHED_PYTHON"
PROVIDER_ENV = "AXON_HWSCHED_PROVIDER"

#: hwsched's documented exit-code taxonomy (``hwsched/cli/main.py``). Note that 3
#: is broader than "the spec is malformed": it is also returned for an unknown
#: queue id and for a command that exists in ``--help`` but is not implemented
#: yet (``reproduce``). Treat it as "hwsched declined to act", not "your YAML is bad".
EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_VALIDATION = 3
EXIT_PROVIDER = 4

_PLAN_TIMEOUT_S = 300.0


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class HwschedError(RuntimeError):
    """Base for every failure of the hwsched integration."""


class HwschedUnavailable(HwschedError):
    """No hwsched checkout at the configured home, or no interpreter to run it."""


class SpecRejected(HwschedError):
    """hwsched declined to act on the spec (exit 3), or the plan has hard violations."""


class ProviderFailed(HwschedError):
    """The provider path failed before producing a parseable result (exit 4)."""


class BudgetRefused(HwschedError):
    """The budget guard refused the plan (exit 2 / ``decision == "refuse"``).

    Carries the numbers needed to decide what to do about it. Recoverable by
    shrinking the job, raising the configured cap, or switching Modal profile —
    never by bypassing the guard.
    """

    def __init__(self, message: str, *, budget: "Budget", cost: "Cost") -> None:
        super().__init__(message)
        self.budget = budget
        self.cost = cost


class ApprovalError(HwschedError):
    """The dry-run-before-spend protocol was not satisfied."""


# --------------------------------------------------------------------------- #
# the JSON contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cost:
    """``cost`` from the hwsched JSON contract. The guard compares ``high``."""

    low: float = 0.0
    expected: float = 0.0
    high: float = 0.0
    duration_s: float = 0.0
    breakdown: Mapping[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "Cost":
        d = payload or {}
        return cls(
            low=float(d.get("low") or 0.0),
            expected=float(d.get("expected") or 0.0),
            high=float(d.get("high") or 0.0),
            duration_s=float(d.get("duration_s") or 0.0),
            breakdown={k: float(v) for k, v in (d.get("breakdown") or {}).items()
                       if isinstance(v, (int, float))},
            assumptions=tuple(d.get("assumptions") or ()),
        )

    def one_line(self) -> str:
        return f"${self.low:.4f} / ${self.expected:.4f} / ${self.high:.4f} (low/expected/high)"


@dataclass(frozen=True)
class Budget:
    """``budget`` from the hwsched JSON contract."""

    decision: str = "unknown"
    remaining: float | None = None
    cap: float | None = None
    conservative_mode: bool = False
    message: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "Budget":
        d = payload or {}
        return cls(
            decision=str(d.get("decision") or "unknown"),
            remaining=_opt_float(d.get("remaining")),
            cap=_opt_float(d.get("cap")),
            conservative_mode=bool(d.get("conservative_mode")),
            message=str(d.get("message") or ""),
        )

    @property
    def refused(self) -> bool:
        return self.decision == "refuse"

    def one_line(self) -> str:
        remaining = "?" if self.remaining is None else f"${self.remaining:.2f}"
        cap = "none" if self.cap is None else f"${self.cap:.2f}"
        return f"{self.decision.upper()} (remaining {remaining}, per-job cap {cap})"


@dataclass(frozen=True)
class Violation:
    """One entry from ``violations``. ``is_error`` distinguishes hard from advisory."""

    code: str
    message: str
    is_error: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Violation":
        return cls(
            code=str(payload.get("code") or "?"),
            message=str(payload.get("message") or ""),
            is_error=bool(payload.get("is_error")),
        )


@dataclass(frozen=True)
class PlanOutcome:
    """A completed dry run: what it would take, what it would cost, and why."""

    job_name: str
    correlation_id: str
    spec_digest: str
    exit_code: int
    plan: Mapping[str, Any]
    cost: Cost
    budget: Budget
    violations: tuple[Violation, ...]
    rationale: tuple[str, ...]
    confidence: float
    spec: Mapping[str, Any] = field(default_factory=dict)
    budget_override_usd: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.budget.refused or self.exit_code == EXIT_REFUSED

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.is_error)

    def plan_line(self) -> str:
        p = self.plan
        device = p.get("device", "?")
        gpu = p.get("gpu_type")
        hw = f"{device}:{gpu}x{p.get('gpu_count', 0)}" if gpu else str(device)
        return (
            f"{hw} cpu={p.get('cpu_cores')} mem={p.get('memory_mib')}MiB "
            f"workers={p.get('n_workers')} chunk={p.get('chunk_size')} "
            f"tasks={p.get('n_tasks')} timeout={p.get('timeout_s')}s"
        )

    def summary(self) -> str:
        """The human-readable dry run. Logged by :meth:`HwschedClient.plan`, so the
        cost and the rationale reach an operator even if the caller drops the
        return value on the floor."""
        lines = [
            f"job:         {self.job_name}  [{self.correlation_id}]",
            f"plan:        {self.plan_line()}",
            f"cost:        {self.cost.one_line()}",
            f"budget:      {self.budget.one_line()}",
        ]
        if self.budget.message:
            lines.append(f"             {self.budget.message}")
        if self.budget_override_usd is not None:
            lines.append(
                f"             monthly cap raised to ${self.budget_override_usd:.2f} for "
                "this client (HWSCHED_MONTHLY_USD)"
            )
        lines.append(f"confidence:  {self.confidence:.2f}")
        for v in self.violations:
            lines.append(f"  {'ERROR' if v.is_error else 'warn '}   {v.code}: {v.message}")
        lines.append("why:")
        lines.extend(f"  - {line}" for line in self.rationale)
        return "\n".join(lines)

    def approve(self, max_usd: float) -> "Approval":
        """Mint the token :meth:`HwschedClient.run` demands.

        *max_usd* is the ceiling the caller accepts, and it must cover the plan's
        ``cost.high`` — the same figure the guard uses. Requiring it here is what
        makes "the estimate was surfaced" checkable: you cannot pass a number you
        have not looked at the estimate to choose. It is also forwarded to
        ``--max-spend`` so the guard re-enforces it at submit, against a ledger
        that may have moved since the dry run.
        """
        if self.refused:
            raise BudgetRefused(
                f"cannot approve a refused plan for {self.job_name!r}: "
                f"{self.budget.message or self.budget.one_line()}",
                budget=self.budget,
                cost=self.cost,
            )
        if self.errors:
            codes = ", ".join(f"{v.code}: {v.message}" for v in self.errors)
            raise SpecRejected(f"plan for {self.job_name!r} has hard violations — {codes}")
        if max_usd < self.cost.high:
            raise ApprovalError(
                f"approval ceiling ${max_usd:.4f} is below the plan's high estimate "
                f"${self.cost.high:.4f} for {self.job_name!r}. Approve at or above the "
                "high estimate, or shrink the job — the guard prices the worst case, "
                "and so should you"
            )
        return Approval(
            correlation_id=self.correlation_id,
            spec_digest=self.spec_digest,
            cost_high=self.cost.high,
            max_usd=float(max_usd),
            rationale=self.rationale,
            plan_line=self.plan_line(),
        )


@dataclass
class Approval:
    """Proof that a dry run happened and its worst case was accepted.

    Mutable and single-use on purpose: :meth:`HwschedClient.run` consumes it. A
    reusable approval inside a retry loop is how one reviewed $0.40 estimate turns
    into forty unreviewed submissions.
    """

    correlation_id: str
    spec_digest: str
    cost_high: float
    max_usd: float
    rationale: tuple[str, ...]
    plan_line: str
    used: bool = False

    def consume(self, job: ComputeJob) -> None:
        """Bind this approval to *job* and spend it, or explain why it does not apply."""
        if self.used:
            raise ApprovalError(
                f"approval for {self.correlation_id!r} was already used; plan again before "
                "submitting again"
            )
        if job.digest != self.spec_digest:
            raise ApprovalError(
                f"approval was minted for spec digest {self.spec_digest} but this job "
                f"hashes to {job.digest}: the job changed after it was planned, so the "
                "approved cost no longer describes it"
            )
        self.used = True


@dataclass(frozen=True)
class RunOutcome:
    """The result of an actual submission."""

    job_name: str
    correlation_id: str
    exit_code: int
    submitted: bool
    status: str
    budget_decision: str
    handle: Mapping[str, Any] | None
    run_id: str | None
    actual_cost_usd: float | None
    cost: Cost
    plan: Mapping[str, Any]
    message: str
    warnings: tuple[str, ...] = ()
    alerts: tuple[str, ...] = ()
    approved_max_usd: float | None = None
    budget_override_usd: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def summary(self) -> str:
        cost = (
            f"${self.actual_cost_usd:.4f} actual"
            if self.actual_cost_usd is not None
            else f"~${self.cost.expected:.4f} expected (Modal reports no synchronous cost)"
        )
        lines = [
            f"job:      {self.job_name}  [{self.correlation_id}]",
            f"status:   {self.status}  (submitted={self.submitted}, exit={self.exit_code})",
            f"budget:   {self.budget_decision}",
            f"cost:     {cost}",
        ]
        if self.message:
            lines.append(f"note:     {self.message}")
        lines.extend(f"  warn: {w}" for w in self.warnings)
        lines.extend(f"  {a}" for a in self.alerts)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# subprocess seam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CliResult:
    """What one CLI invocation produced. The seam tests replace."""

    returncode: int
    stdout: str
    stderr: str


CliRunner = Callable[[list[str], str, dict[str, str], float | None], CliResult]


def _subprocess_runner(
    argv: list[str], cwd: str, env: dict[str, str], timeout_s: float | None
) -> CliResult:
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built here, never from user text
            argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HwschedError(
            f"hwsched did not finish within {timeout_s}s: {' '.join(argv)}. The Modal job "
            "may still be running — killing this client does not cancel it. Check "
            "`hwsched queue` / the Modal dashboard before resubmitting"
        ) from exc
    except FileNotFoundError as exc:
        raise HwschedUnavailable(f"cannot execute {argv[0]!r}: {exc}") from exc
    return CliResult(proc.returncode, proc.stdout, proc.stderr)


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #
class HwschedClient:
    """Shells out to hwsched and parses its documented JSON contract."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        python: str | None = None,
        provider: str | None = None,
        config: str | Path | None = None,
        store_path: str | Path | None = None,
        monthly_budget_usd: float | None = None,
        extra_pythonpath: Sequence[str | Path] = (),
        runner: CliRunner | None = None,
    ) -> None:
        self.home = Path(home or os.environ.get(HOME_ENV) or DEFAULT_HOME).expanduser()
        self.python = python or os.environ.get(PYTHON_ENV) or "python3"
        self.provider = provider if provider is not None else os.environ.get(PROVIDER_ENV)
        self.config = Path(config) if config else None
        self.store_path = Path(store_path) if store_path else None
        self.monthly_budget_usd = monthly_budget_usd
        self._extra_pythonpath = [str(p) for p in extra_pythonpath]
        self._runner: CliRunner = runner or _subprocess_runner

    # ── availability ──

    @property
    def available(self) -> bool:
        """Whether the configured home actually holds an hwsched package."""
        return (self.home / "hwsched" / "__init__.py").is_file()

    def require_available(self) -> None:
        if not self.available:
            raise HwschedUnavailable(
                f"no hwsched checkout at {self.home} (set ${HOME_ENV}). hwsched is not "
                "pip-installed; `python3 -m hwsched` only resolves with that directory "
                "on sys.path"
            )
        if shutil.which(self.python) is None and not Path(self.python).is_file():
            raise HwschedUnavailable(f"interpreter {self.python!r} not found (set ${PYTHON_ENV})")

    # ── commands ──

    def plan(
        self,
        job: ComputeJob,
        *,
        objective: str | None = None,
        max_spend: float | None = None,
        timeout_s: float | None = _PLAN_TIMEOUT_S,
    ) -> PlanOutcome:
        """Dry run: size, cost and rationale, with no spend.

        Always returns. A budget refusal (exit 2) and a capability violation
        (exit 4) are reported on the outcome rather than raised, because the whole
        point of a dry run is to *see* them — a planner that throws on a refusal
        cannot tell you how far over you are.
        """
        argv = ["plan", "{spec}", "--json"]
        if objective:
            argv += ["--objective", objective]
        if max_spend is not None:
            argv += ["--max-spend", repr(float(max_spend))]
        result, payload = self._invoke(job, argv, timeout_s=timeout_s)

        if result.returncode == EXIT_VALIDATION:
            raise SpecRejected(
                f"hwsched rejected the spec for {job.name!r} (exit 3): "
                f"{_first_line(result.stderr) or _first_line(result.stdout) or 'no output'}"
            )
        if payload is None:
            # A dry run that produced no report is not a $0 plan. Approving on the
            # back of an empty parse is how an uncosted job gets submitted.
            raise HwschedError(
                f"hwsched produced no plan JSON for {job.name!r} "
                f"(exit {result.returncode}): {_first_line(result.stderr) or 'no output'}"
            )

        outcome = PlanOutcome(
            job_name=job.name,
            correlation_id=job.resolved_correlation_id,
            spec_digest=job.digest,
            exit_code=result.returncode,
            plan=payload.get("plan") or {},
            cost=Cost.from_payload(payload.get("cost")),
            budget=Budget.from_payload(payload.get("budget")),
            violations=tuple(Violation.from_payload(v) for v in payload.get("violations") or ()),
            rationale=tuple(payload.get("rationale") or ()),
            confidence=float(payload.get("confidence") or 0.0),
            spec=payload.get("spec") or {},
            budget_override_usd=self.monthly_budget_usd,
            raw=payload,
        )
        # Logged, not just returned: this is the "surface the estimate" step, and it
        # has to happen even for a caller that ignores the value it got back.
        log.info("hwsched dry run\n%s", outcome.summary())
        return outcome

    def run(
        self,
        job: ComputeJob,
        *,
        approval: Approval | None,
        force: bool = False,
        timeout_s: float | None = None,
    ) -> RunOutcome:
        """Plan, guard and submit — only against a live :class:`Approval`.

        Returns an outcome for a job that ran and failed (exit 4 with a parseable
        result): "the compute failed" is data the caller must inspect, not an
        exception. Raises when nothing ran — no approval, a refusal, a rejected
        spec, or output that could not be parsed.

        *timeout_s* defaults to no limit because hwsched blocks until the job is
        terminal, and Modal allows a 24 h container. A timeout here kills the
        client, not the job.
        """
        if approval is None:
            raise ApprovalError(
                f"refusing to submit {job.name!r} without a plan. Call plan(), read "
                "cost.high and the rationale, then approve(max_usd=…) — dry run before "
                "spend is not optional"
            )
        approval.consume(job)

        argv = ["run", "{spec}", "--json", "--max-spend", repr(approval.max_usd)]
        if force:
            argv.append("--force")
        log.warning(
            "hwsched submit %s [%s]: %s, approved ceiling $%.4f (plan high $%.4f)",
            job.name, job.resolved_correlation_id, approval.plan_line,
            approval.max_usd, approval.cost_high,
        )
        result, payload = self._invoke(job, argv, timeout_s=timeout_s)

        if result.returncode == EXIT_REFUSED:
            budget = Budget.from_payload(
                {"decision": "refuse", "message": _budget_message(payload, result)}
            )
            raise BudgetRefused(
                f"the budget guard refused {job.name!r} at submit: {budget.message}. "
                "Nothing was spent. Shrink the job, raise the configured monthly cap for "
                "this profile, or switch profile — do not bypass the guard",
                budget=budget,
                cost=Cost.from_payload((payload or {}).get("cost")),
            )
        if result.returncode == EXIT_VALIDATION:
            raise SpecRejected(
                f"hwsched declined to run {job.name!r} (exit 3): "
                f"{_first_line(result.stderr) or 'no output'}"
            )
        if payload is None:
            raise ProviderFailed(
                f"hwsched produced no parseable JSON for {job.name!r} "
                f"(exit {result.returncode}): {_first_line(result.stderr) or 'no output'}"
            )

        return RunOutcome(
            job_name=job.name,
            correlation_id=job.resolved_correlation_id,
            exit_code=result.returncode,
            submitted=bool(payload.get("submitted")),
            status=str(payload.get("status") or "unknown"),
            budget_decision=str(payload.get("budget_decision") or "unknown"),
            handle=payload.get("handle"),
            run_id=payload.get("run_id"),
            actual_cost_usd=_opt_float(payload.get("actual_cost_usd")),
            cost=Cost.from_payload(payload.get("cost")),
            plan=payload.get("plan") or {},
            message=str(payload.get("message") or ""),
            warnings=tuple(payload.get("warnings") or ()),
            alerts=tuple(payload.get("alerts") or ()),
            approved_max_usd=approval.max_usd,
            budget_override_usd=self.monthly_budget_usd,
            raw=payload,
        )

    def validate(self, job: ComputeJob) -> dict[str, Any]:
        """Ask hwsched to parse the spec and echo it back, normalized."""
        result, payload = self._invoke(job, ["validate", "{spec}", "--json"], timeout_s=60.0)
        if result.returncode != EXIT_OK or payload is None:
            raise SpecRejected(
                f"hwsched rejected {job.name!r}: {_first_line(result.stderr) or 'no output'}"
            )
        return payload

    def budget_state(self) -> dict[str, Any]:
        """The ledger: spent, committed, remaining for the current billing window."""
        result, payload = self._invoke(None, ["budget", "--json"], timeout_s=60.0)
        if payload is None:
            raise HwschedError(
                f"could not read the hwsched ledger (exit {result.returncode}): "
                f"{_first_line(result.stderr) or 'no output'}"
            )
        return payload

    # ── plumbing ──

    def _invoke(
        self, job: ComputeJob | None, argv: Sequence[str], *, timeout_s: float | None
    ) -> tuple[CliResult, dict[str, Any] | None]:
        """Emit the spec (if any), run the CLI, and parse stdout as JSON.

        The spec is written to a temp file and passed as an **absolute** path: the
        subprocess runs with the hwsched checkout as its cwd, so a relative path
        would resolve against the wrong repository.
        """
        self.require_available()
        env = self._env()
        # ``--provider``/``--config`` are attached to each *subcommand*, not to the
        # top-level parser, so they must follow the command word. Put them first and
        # argparse exits 2 with a usage message — indistinguishable by exit code from
        # a budget refusal, which is exactly the confusion _check_usage() catches.
        command, *rest = argv
        flags: list[str] = []
        if self.provider:
            flags += ["--provider", self.provider]
        if self.config:
            flags += ["--config", str(self.config)]
        base = [self.python, "-m", "hwsched", command, *flags]

        if job is None:
            result = self._runner(base + rest, str(self.home), env, timeout_s)
            return result, _check_usage(result, _parse_json(result.stdout))

        with tempfile.TemporaryDirectory(prefix="axon-compute-") as tmp:
            spec_path = job.write(Path(tmp) / f"{job.name}.yaml")
            filled = [str(spec_path) if a == "{spec}" else a for a in rest]
            result = self._runner(base + filled, str(self.home), env, timeout_s)
        return result, _check_usage(result, _parse_json(result.stdout))

    def _env(self) -> dict[str, str]:
        """Environment for the subprocess.

        ``PYTHONPATH`` gets the hwsched checkout (so ``-m hwsched`` resolves) and the
        directory holding the ``axon`` package. The second is not cosmetic: hwsched's
        Modal adapter calls ``Image.add_local_python_source(<top package of the
        entrypoint>)``, which resolves that package on the *client's* ``sys.path``. An
        ``axon.*`` entrypoint that is not importable here fails at submit, after the
        app has been deployed.
        """
        env = dict(os.environ)
        parts = [str(self.home), str(_axon_source_root()), *self._extra_pythonpath]
        existing = env.get("PYTHONPATH")
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))

        # hwsched stops a finished run's Modal app by shelling out to the `modal`
        # executable, found with shutil.which. Driven from Axon the interpreter is
        # usually a venv one whose bin/ is not on PATH, so that lookup fails, the
        # best-effort cleanup silently no-ops, and every run leaves a lingering
        # `deployed` app behind. Putting the interpreter's own bin/ first makes the
        # cleanup hwsched already wrote actually fire.
        bin_dir = _interpreter_bin(self.python)
        if bin_dir is not None:
            env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")]).rstrip(os.pathsep)
        if self.store_path is not None:
            env["HWSCHED_STORE_PATH"] = str(self.store_path)
        if self.monthly_budget_usd is not None:
            # hwsched's own override knob. It moves the cap the guard enforces; it
            # does not disable the comparison, and HWSCHED_ALLOW_OVERAGE is never set.
            env["HWSCHED_MONTHLY_USD"] = repr(float(self.monthly_budget_usd))
        return env


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _axon_source_root() -> Path:
    """The directory containing the ``axon`` package (``…/python``)."""
    return Path(__file__).resolve().parents[2]


def _interpreter_bin(python: str) -> Path | None:
    """The ``bin/`` directory the configured interpreter *sits in*.

    Deliberately ``parent.resolve()`` and not ``resolve().parent``: a venv's
    ``bin/python`` is a symlink chain ending at the system interpreter, so
    resolving the file first yields ``/usr/bin`` — the one directory guaranteed
    not to hold the venv's ``modal`` entry point.
    """
    candidate = Path(python)
    if not candidate.is_file():
        found = shutil.which(python)
        if found is None:
            return None
        candidate = Path(found)
    return candidate.parent.resolve()


def _opt_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _check_usage(result: CliResult, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fail loudly on an argparse usage error masquerading as a budget refusal.

    argparse exits 2 on a bad command line and hwsched uses 2 for "refused by the
    budget guard". Same code, opposite meanings: one says the job is too expensive,
    the other says this client built a command hwsched does not understand. They
    are told apart by the payload — a real refusal still prints its report JSON on
    stdout, a usage error prints only a usage banner on stderr.
    """
    if payload is None and result.returncode == EXIT_REFUSED and "usage:" in result.stderr:
        raise HwschedError(
            "hwsched rejected the command line, not the budget: "
            f"{_first_line(result.stderr)}. This is an axon.compute bug, not a cost problem"
        )
    return payload


def _budget_message(payload: Mapping[str, Any] | None, result: CliResult) -> str:
    if payload:
        for key in ("message", "budget_decision"):
            value = payload.get(key)
            if value:
                return str(value)
    return _first_line(result.stderr) or "refused"


def _parse_json(stdout: str) -> dict[str, Any] | None:
    """Parse the CLI's ``--json`` payload, tolerating leading noise.

    Returns ``None`` rather than raising: a non-zero exit with no JSON is a real
    and expected shape (validation errors print only to stderr), and the caller
    decides what that means for the command it ran.
    """
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            parsed = json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "DEFAULT_HOME",
    "EXIT_OK",
    "EXIT_PROVIDER",
    "EXIT_REFUSED",
    "EXIT_VALIDATION",
    "HOME_ENV",
    "PROVIDER_ENV",
    "PYTHON_ENV",
    "Approval",
    "ApprovalError",
    "Budget",
    "BudgetRefused",
    "CliResult",
    "Cost",
    "HwschedClient",
    "HwschedError",
    "HwschedUnavailable",
    "PlanOutcome",
    "ProviderFailed",
    "RunOutcome",
    "SpecRejected",
    "Violation",
]
