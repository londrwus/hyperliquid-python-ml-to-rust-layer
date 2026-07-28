"""Offloading compute to Modal through hwsched.

Named after the failure modes they prevent, per
``crates/axon-execution/src/tracker.rs``. Two layers, and the split is what keeps
the default gate offline and deterministic:

* Everything about the emitted spec and the parsed JSON runs against **recorded
  payloads** — real output captured from ``python3 -m hwsched … --json`` on
  2026-07-25 — through an injected CLI runner. No subprocess, no network.
* The handful of tests that genuinely shell out are guarded by
  :func:`_real_client`, which skips when the hwsched checkout is absent or its
  interpreter cannot import it. Those still never touch the network: ``plan`` is
  a pure sizing/costing pass and they run against the ``fake`` provider with an
  isolated run store, so they cannot see or move the real ledger.
"""

from __future__ import annotations

import json
import os
import pickle
import pickletools
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from axon.compute import (
    CORRELATION_ENV,
    JOBSPEC_FIELDS,
    Approval,
    ApprovalError,
    ArtifactError,
    BudgetRefused,
    ComputeJob,
    HwschedClient,
    HwschedError,
    HwschedUnavailable,
    SpecError,
    feature_etl,
    mount_path,
    param_sweep,
    train_model,
    volume_uri,
    walk_forward,
)
from axon.compute.client import CliResult
from axon.compute.entry import gpu_probe, probe

# --------------------------------------------------------------------------- #
# recorded hwsched output (captured 2026-07-25, trimmed to the contract)
# --------------------------------------------------------------------------- #
PLAN_APPROVED = {
    "spec": {"name": "axon-sma-sweep", "env": {CORRELATION_ENV: "axon.axon-sma-sweep.1ca020c0"}},
    "plan": {
        "provider": "modal", "device": "cpu", "gpu_type": None, "gpu_count": 0,
        "cpu_cores": 2.0, "memory_mib": 8704, "n_workers": 5, "chunk_size": 4,
        "n_tasks": 18, "waves": 1, "timeout_s": 300, "preemptible": True,
        "est_cost_usd": 0.01365621, "confidence": 0.6,
    },
    "cost": {
        "low": 0.0120174648, "expected": 0.01365621, "high": 0.019118694,
        "breakdown": {"gpu": 0.0, "cpu": 0.0075456, "mem": 0.00543456,
                      "overhead": 0.00067605},
        "duration_s": 67.0,
        "assumptions": ["safety multiplier 1.2x as a floor on high"],
    },
    "budget": {
        "decision": "approve", "remaining": 27.0, "cap": 1.0, "conservative_mode": False,
        "message": "approved: est high $0.02 <= cap $1.00 (remaining $27.00)",
    },
    "violations": [],
    "rationale": [
        "param_sweep: CPU, no GPU framework imported.",
        "chunk_size=4 targets ~64s/container (objective balanced).",
    ],
    "confidence": 0.6,
}

PLAN_REFUSED = {
    "spec": {"name": "axon-lstm-train"},
    "plan": {
        "provider": "modal", "device": "gpu", "gpu_type": "L40S", "gpu_count": 1,
        "cpu_cores": 4.0, "memory_mib": 40960, "n_workers": 1, "chunk_size": 1,
        "n_tasks": 1, "timeout_s": 21600, "confidence": 0.6,
    },
    "cost": {
        "low": 1.0972192, "expected": 1.24684, "high": 1.745576,
        "breakdown": {"gpu": 0.9756, "cpu": 0.09432, "mem": 0.15984, "overhead": 0.01708},
        "duration_s": 1825.0, "assumptions": [],
    },
    "budget": {
        "decision": "refuse", "remaining": 27.0, "cap": 0.05, "conservative_mode": False,
        "message": (
            "REFUSED: cheapest feasible plan still $1.75 > cap $0.05 (remaining $27.00, "
            "job cap $0.05). Options: switch to CPU if the workload allows (planner "
            "GPU->CPU); raise this job's budget.max_usd above $0.05."
        ),
    },
    "violations": [],
    "rationale": ["ml_train_dnn is GPU-bound.", "smallest GPU covering ~28 GB: L40S (48 GB)."],
    "confidence": 0.6,
}

RUN_SUCCEEDED = {
    "spec_name": "axon-modal-probe",
    "plan": {"device": "cpu", "cpu_cores": 1.0, "memory_mib": 1024, "n_workers": 1,
             "chunk_size": 1, "n_tasks": 1, "timeout_s": 600},
    "cost": {"low": 0.0004132953, "expected": 0.00050556, "high": 0.000770979,
             "breakdown": {}, "duration_s": 33.0, "assumptions": []},
    "handle": {
        "id": "axon-modal-probe", "provider": "modal",
        "native_ref": '{"app": "hwsched-axon-modal-probe", "app_id": "ap-JK0Zprt2CkG1XS6"}',
        "n_workers": 1, "gpu_count": 0,
    },
    "run_id": "0bc2f5209cb540dba7a67ef3ce8b9506",
    "status": "succeeded",
    "actual_cost_usd": None,
    "outputs": {"receipt": "volume://axon-artifacts/probe/receipt.json"},
    "submitted": True,
    "dry_run": False,
    "budget_decision": "approve",
    "message": None,
    "results": None,
    "alerts": [],
    "warnings": [],
}

RUN_FAILED = dict(
    RUN_SUCCEEDED,
    status="failed",
    message="task 0: ModuleNotFoundError: No module named 'pandas'",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class Recorder:
    """A stand-in for the CLI: records the invocation, replays a canned result."""

    def __init__(self, *results: CliResult) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    def __call__(self, argv, cwd, env, timeout_s):  # noqa: ANN001 - the CliRunner shape
        self.calls.append({"argv": argv, "cwd": cwd, "env": env, "timeout_s": timeout_s})
        return self._results.pop(0) if self._results else CliResult(0, "{}", "")

    @property
    def argv(self) -> list[str]:
        return self.calls[-1]["argv"]

    @property
    def env(self) -> dict:
        return self.calls[-1]["env"]


def ok(payload: dict, code: int = 0, stderr: str = "") -> CliResult:
    return CliResult(code, json.dumps(payload), stderr)


def _make_home(root: Path) -> Path:
    """Make *root* look like an hwsched checkout, so availability passes."""
    pkg = root / "hwsched"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def fake_home(tmp_path) -> Path:
    return _make_home(tmp_path)


@pytest.fixture
def sweep() -> ComputeJob:
    return param_sweep(
        "axon-sma-sweep",
        "axon.backtest.sweep:run_one",
        {"fast": [5, 10, 20], "slow": [50, 100, 200], "symbol": ["BTC", "ETH"]},
        framework="vectorbt",
        data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/metrics.json")},
        max_usd=0.50,
    )


def client(home, runner, **kwargs) -> HwschedClient:
    return HwschedClient(home=home, python=sys.executable, runner=runner, **kwargs)


def approved(job: ComputeJob, max_usd: float = 1.0) -> Approval:
    """An approval minted the only way one can be: from a plan."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    with tempfile.TemporaryDirectory() as tmp:
        home = _make_home(Path(tmp))
        return client(home, Recorder(ok(PLAN_APPROVED))).plan(job).approve(max_usd)


# --------------------------------------------------------------------------- #
# the emitter: a spec hwsched will actually accept
# --------------------------------------------------------------------------- #
def test_emitted_spec_uses_only_fields_hwsched_accepts(sweep):
    """JobSpec forbids extras, so one stray key rejects the whole spec at exit 3."""
    jobs = [
        sweep,
        walk_forward("wf", "axon.backtest.wf:run", [{"start": 0, "end": 1}]),
        train_model("t", "axon.models.train:fit", framework="xgboost"),
        feature_etl("etl", "axon.features.build:run", data_size_gb=12.0),
    ]
    for job in jobs:
        assert set(job.to_mapping()) <= JOBSPEC_FIELDS, job.name


def test_field_allowlist_tracks_the_real_jobspec_model():
    """A field added to hwsched's model must not silently stay unreachable here.

    Asks the installed model for its own field set rather than trusting the copy,
    because the copy is exactly the thing that rots.
    """
    home = _hwsched_home()
    if home is None:
        pytest.skip("no hwsched checkout to compare against")
    code = (
        "import json; from hwsched.spec import JobSpec; "
        "print(json.dumps(sorted(JobSpec.model_fields)))"
    )
    env = dict(os.environ, PYTHONPATH=str(home))
    proc = subprocess.run(
        [_hwsched_python(), "-c", code], cwd=str(home), env=env,
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        pytest.skip(f"cannot import hwsched.spec with {_hwsched_python()}: {last} — "
                    f"{_INTERPRETER_HINT}")
    assert set(json.loads(proc.stdout)) == set(JOBSPEC_FIELDS)


def test_correlation_id_never_reaches_the_task_arguments(sweep):
    """The id rides in env, not kwargs.

    hwsched calls ``fn(**task)`` for every grid point, so an id smuggled into
    ``params`` becomes an argument the entrypoint does not accept — and one
    smuggled into ``kwargs`` is dropped outright for a fan-out, because tasks are
    materialized from the grid.
    """
    spec = sweep.to_mapping()
    assert spec["env"][CORRELATION_ENV] == sweep.resolved_correlation_id
    assert set(spec["params"]) == {"fast", "slow", "symbol"}
    assert "kwargs" not in spec
    assert sweep.n_tasks == 18


def test_correlation_id_is_content_addressed_so_dedup_still_works(sweep):
    """A fresh id per call would disable hwsched's idempotency cache.

    ``env`` feeds hwsched's derived idempotency key, so a UUID here would make
    every submission look like new work and re-run completed jobs forever.
    """
    assert sweep.resolved_correlation_id == param_sweep(
        "axon-sma-sweep", "axon.backtest.sweep:run_one",
        {"fast": [5, 10, 20], "slow": [50, 100, 200], "symbol": ["BTC", "ETH"]},
        framework="vectorbt", data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/metrics.json")},
        max_usd=0.50,
    ).resolved_correlation_id

    wider = param_sweep(
        "axon-sma-sweep", "axon.backtest.sweep:run_one",
        {"fast": [5, 10, 20, 30], "slow": [50, 100, 200], "symbol": ["BTC", "ETH"]},
        framework="vectorbt", data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/metrics.json")},
        max_usd=0.50,
    )
    assert wider.resolved_correlation_id != sweep.resolved_correlation_id


def test_repointed_outputs_are_not_deduped_as_the_same_job(sweep):
    """hwsched omits ``outputs`` from its key; two runs writing different artifacts
    would hash identically and the second would be skipped as a duplicate."""
    elsewhere = param_sweep(
        "axon-sma-sweep", "axon.backtest.sweep:run_one",
        {"fast": [5, 10, 20], "slow": [50, 100, 200], "symbol": ["BTC", "ETH"]},
        framework="vectorbt", data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/v2.json")},
        max_usd=0.50,
    )
    assert elsewhere.digest != sweep.digest


def test_explicit_correlation_id_is_honoured():
    job = param_sweep("j", "m:f", {"a": [1]}, correlation_id="parity-2026-07-25-window-3")
    assert job.to_mapping()["env"][CORRELATION_ENV] == "parity-2026-07-25-window-3"


def test_hand_written_correlation_env_is_rejected():
    """Two sources for the id means the digest and the id can disagree."""
    with pytest.raises(SpecError, match=CORRELATION_ENV):
        param_sweep("j", "m:f", {"a": [1]}, env={CORRELATION_ENV: "sneaky"})


def test_grid_and_task_list_together_are_rejected_before_the_subprocess():
    with pytest.raises(SpecError, match="not both"):
        ComputeJob(name="j", entrypoint="m:f", workload="param_sweep",
                   params={"a": [1]}, tasks=[{"a": 1}])


def test_unknown_workload_is_rejected_before_the_subprocess():
    with pytest.raises(SpecError, match="unknown workload"):
        ComputeJob(name="j", entrypoint="m:f", workload="backtest_but_faster")


def test_mistyped_resource_pin_is_rejected_rather_than_ignored():
    """``resources`` entries are hard pins; a typo would be dropped by ``extra=forbid``
    at hwsched's boundary — or worse, silently let the planner pick for itself."""
    with pytest.raises(SpecError, match="unknown resource pin"):
        ComputeJob(name="j", entrypoint="m:f", workload="ml_train_dnn",
                   resources={"gpu": "A100"})


def test_entrypoint_must_name_a_module_and_a_function():
    """The container resolves it with importlib; a bare module fails after deploy."""
    with pytest.raises(SpecError, match="module.path:function"):
        ComputeJob(name="j", entrypoint="axon.models.train", workload="generic")


def test_net_training_is_not_planned_as_a_cpu_job():
    """``ml_train_dnn`` is what forces a GPU. hwsched's static profiler cannot see
    Axon's source from its own working directory, so the declared workload is the
    only signal it gets."""
    assert train_model("t", "m:f", framework="torch").workload == "ml_train_dnn"
    assert train_model("t", "m:f", framework="xgboost").workload == "ml_train_trees"
    with pytest.raises(SpecError, match="cannot infer a workload"):
        train_model("t", "m:f", framework="some-new-thing")


def test_walk_forward_windows_are_a_task_list_not_a_grid():
    """A Cartesian product of starts and ends enumerates windows that run backwards."""
    job = walk_forward("wf", "m:f", [{"start": 0, "end": 10}, {"start": 10, "end": 20}])
    spec = job.to_mapping()
    assert spec["tasks"] == [{"start": 0, "end": 10}, {"start": 10, "end": 20}]
    assert "params" not in spec
    assert job.n_tasks == 2


def test_spec_yaml_round_trips_through_a_yaml_parser(sweep, tmp_path):
    yaml = pytest.importorskip("yaml")
    path = sweep.write(tmp_path / "job.yaml")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == sweep.to_mapping()


# --------------------------------------------------------------------------- #
# artifacts: the Volume hand-off
# --------------------------------------------------------------------------- #
def test_volume_subpath_cannot_escape_the_mount():
    """A write outside /vol does not fail — it lands on ephemeral container disk and
    vanishes on exit, so the artifact is simply gone with no error anywhere."""
    for bad in ("../secrets", "a/../../b", "\\windows\\path"):
        with pytest.raises(ArtifactError):
            volume_uri("axon-artifacts", bad)


def test_uri_and_mount_path_describe_one_location():
    assert volume_uri("axon-artifacts", "models/v1.onnx") == (
        "volume://axon-artifacts/models/v1.onnx"
    )
    assert mount_path("axon-artifacts", "models/v1.onnx") == (
        "/vol/axon-artifacts/models/v1.onnx"
    )


def test_invalid_volume_name_is_caught_before_submit():
    with pytest.raises(ArtifactError, match="volume name"):
        volume_uri("Axon Artifacts", "x")


def test_artifacts_must_be_volume_uris_not_return_values():
    """Return values are pickled through Modal's control plane and the CLI path never
    collects them at all, so a non-volume output is discarded silently."""
    with pytest.raises(SpecError, match="volume://"):
        feature_etl("etl", "m:f", outputs={"matrix": "/tmp/features.parquet"})
    with pytest.raises(SpecError, match="volume://"):
        feature_etl("etl", "m:f", inputs={"raw": "s3://bucket/raw"})


# --------------------------------------------------------------------------- #
# the JSON contract and the exit-code taxonomy
# --------------------------------------------------------------------------- #
def test_plan_contract_is_parsed_into_typed_fields(fake_home, sweep):
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    outcome = client(fake_home, runner).plan(sweep)

    assert outcome.exit_code == 0
    assert not outcome.refused
    assert outcome.cost.high == pytest.approx(0.019118694)
    assert outcome.cost.breakdown["cpu"] == pytest.approx(0.0075456)
    assert outcome.budget.decision == "approve"
    assert outcome.budget.remaining == 27.0
    assert outcome.budget.cap == 1.0
    assert outcome.confidence == 0.6
    assert outcome.rationale[0].startswith("param_sweep:")
    assert outcome.plan["device"] == "cpu"
    assert "cost:" in outcome.summary() and "why:" in outcome.summary()


def test_budget_refusal_is_returned_by_plan_not_raised(fake_home):
    """Exit 2 is a per-profile guard decision, not a crash: a dry run that throws on
    a refusal cannot tell you how far over the cap you are."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    job = train_model("axon-lstm-train", "axon.models.train_dnn:fit",
                      framework="torch", data_size_gb=40.0, max_usd=0.05)
    runner = Recorder(ok(PLAN_REFUSED, code=2, stderr="budget guard REFUSED this plan"))
    outcome = client(fake_home, runner).plan(job)

    assert outcome.exit_code == 2
    assert outcome.refused
    assert outcome.budget.cap == 0.05
    assert outcome.cost.high == pytest.approx(1.745576)
    assert "raise this job's budget.max_usd" in outcome.budget.message


def test_validation_exit_is_raised_rather_than_read_as_an_empty_plan(fake_home, sweep):
    """Exit 3 prints only to stderr; treating the empty stdout as a $0 plan would
    approve a job that was never costed."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(CliResult(3, "", "validation error: 1 validation error for JobSpec"))
    with pytest.raises(HwschedError, match="rejected the spec"):
        client(fake_home, runner).plan(sweep)


def test_argparse_usage_error_is_not_mistaken_for_a_budget_refusal(fake_home, sweep):
    """argparse also exits 2. The two are told apart by the payload: a real refusal
    still prints its report on stdout, a usage error prints only a banner."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(CliResult(2, "", "usage: hwsched [-h]\nhwsched: error: unrecognized"))
    with pytest.raises(HwschedError, match="not the budget"):
        client(fake_home, runner).plan(sweep)


def test_run_contract_is_parsed_into_typed_fields(fake_home, sweep):
    runner = Recorder(ok(RUN_SUCCEEDED))
    outcome = client(fake_home, runner).run(sweep, approval=approved(sweep))

    assert outcome.submitted and outcome.succeeded
    assert outcome.run_id == "0bc2f5209cb540dba7a67ef3ce8b9506"
    assert outcome.handle["provider"] == "modal"
    assert outcome.budget_decision == "approve"
    # Modal reports no synchronous cost; a client that defaulted this to 0.0 would
    # make every run look free in whatever report consumed it.
    assert outcome.actual_cost_usd is None
    assert outcome.cost.expected == pytest.approx(0.00050556)


def test_failed_compute_is_returned_as_data_not_raised(fake_home, sweep):
    """"The job ran and failed" is an outcome the caller must inspect; raising would
    throw away the handle and the plan needed to diagnose it."""
    runner = Recorder(ok(RUN_FAILED, code=4))
    outcome = client(fake_home, runner).run(sweep, approval=approved(sweep))

    assert outcome.exit_code == 4
    assert not outcome.succeeded
    assert "ModuleNotFoundError" in outcome.message


def test_provider_failure_with_no_output_is_raised(fake_home, sweep):
    runner = Recorder(CliResult(4, "", "provider error: could not reach Modal"))
    with pytest.raises(HwschedError, match="no parseable JSON"):
        client(fake_home, runner).run(sweep, approval=approved(sweep))


def test_refusal_at_submit_reports_that_nothing_was_spent(fake_home, sweep):
    """The ledger can move between the dry run and the submit."""
    runner = Recorder(ok({"budget_decision": "refuse", "message": "REFUSED: over cap"}, code=2))
    with pytest.raises(BudgetRefused, match="Nothing was spent"):
        client(fake_home, runner).run(sweep, approval=approved(sweep))


# --------------------------------------------------------------------------- #
# dry run before spend
# --------------------------------------------------------------------------- #
def test_submit_without_a_plan_never_reaches_the_cli(fake_home, sweep):
    """The whole point: an accidental A100 sweep has to be impossible to launch
    without someone having read what it costs."""
    runner = Recorder(ok(RUN_SUCCEEDED))
    with pytest.raises(ApprovalError, match="without a plan"):
        client(fake_home, runner).run(sweep, approval=None)
    assert runner.calls == []


def test_approval_does_not_carry_over_to_a_mutated_job(fake_home, sweep):
    """Plan a 16-point CPU sweep, widen the grid, submit: the approved cost no longer
    describes the work."""
    approval = approved(sweep)
    wider = param_sweep(
        "axon-sma-sweep", "axon.backtest.sweep:run_one",
        {"fast": list(range(5, 60, 5)), "slow": list(range(50, 400, 25)),
         "symbol": ["BTC", "ETH"]},
        framework="vectorbt", data_size_gb=2.0,
        outputs={"metrics": volume_uri("axon-artifacts", "sweeps/sma/metrics.json")},
        max_usd=0.50,
    )
    runner = Recorder(ok(RUN_SUCCEEDED))
    with pytest.raises(ApprovalError, match="changed after it was planned"):
        client(fake_home, runner).run(wider, approval=approval)
    assert runner.calls == []


def test_approval_is_single_use(fake_home, sweep):
    """A reusable approval in a retry loop turns one reviewed estimate into N
    unreviewed submissions."""
    approval = approved(sweep)
    runner = Recorder(ok(RUN_SUCCEEDED), ok(RUN_SUCCEEDED))
    api = client(fake_home, runner)
    api.run(sweep, approval=approval)
    with pytest.raises(ApprovalError, match="already used"):
        api.run(sweep, approval=approval)
    assert len(runner.calls) == 1


def test_approval_below_the_high_estimate_is_refused(fake_home, sweep):
    """The guard prices the worst case; approving only the expected case would admit
    a job whose bad path is over budget."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    outcome = client(fake_home, Recorder(ok(PLAN_APPROVED))).plan(sweep)
    with pytest.raises(ApprovalError, match="below the plan's high estimate"):
        outcome.approve(max_usd=0.015)


def test_a_refused_plan_cannot_be_approved(fake_home):
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    job = train_model("axon-lstm-train", "m:f", framework="torch", max_usd=0.05)
    outcome = client(fake_home, Recorder(ok(PLAN_REFUSED, code=2))).plan(job)
    with pytest.raises(BudgetRefused):
        outcome.approve(max_usd=100.0)


def test_the_approved_ceiling_is_handed_back_to_the_guard(fake_home, sweep):
    """Our ceiling is not advisory: ``--max-spend`` makes hwsched re-enforce it at
    submit, against a ledger that may have moved since the dry run."""
    runner = Recorder(ok(RUN_SUCCEEDED))
    client(fake_home, runner).run(sweep, approval=approved(sweep, max_usd=0.42))
    assert "--max-spend" in runner.argv
    assert runner.argv[runner.argv.index("--max-spend") + 1] == "0.42"


def test_global_flags_follow_the_subcommand(fake_home, sweep):
    """hwsched attaches ``--provider``/``--config`` to each subcommand, not to the
    top-level parser; putting them first is an argparse usage error that exits 2 and
    reads like a budget refusal."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    client(fake_home, runner, provider="fake").plan(sweep)
    argv = runner.argv
    assert argv[1:3] == ["-m", "hwsched"]
    assert argv[3] == "plan"
    assert argv[4:6] == ["--provider", "fake"]
    assert argv[-1] == "--json"


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_missing_hwsched_home_points_at_the_env_var(tmp_path, sweep):
    api = HwschedClient(home=tmp_path / "nowhere", python=sys.executable,
                        runner=Recorder(ok(PLAN_APPROVED)))
    assert not api.available
    with pytest.raises(HwschedUnavailable, match="AXON_HWSCHED_HOME"):
        api.plan(sweep)


def test_axon_source_root_is_on_the_subprocess_pythonpath(fake_home, sweep):
    """hwsched's Modal adapter resolves the entrypoint's top-level package on the
    *client's* sys.path to mount it into the image; an ``axon.*`` entrypoint that is
    not importable here fails at submit, after the app has been deployed."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    client(fake_home, runner).plan(sweep)
    entries = runner.env["PYTHONPATH"].split(os.pathsep)
    assert str(fake_home) in entries
    import axon

    assert str(os.path.dirname(os.path.dirname(os.path.abspath(axon.__file__)))) in entries


def test_the_spec_path_is_absolute(fake_home, sweep):
    """The subprocess runs with the hwsched checkout as its cwd, so a relative spec
    path would resolve against the wrong repository."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    client(fake_home, runner).plan(sweep)
    spec_arg = next(a for a in runner.argv if a.endswith(".yaml"))
    assert os.path.isabs(spec_arg)


def test_monthly_override_raises_the_cap_and_never_disables_the_guard(fake_home, sweep):
    """The one sanctioned lever for a bigger-balance profile. It moves the number the
    guard enforces; ``allow_overage`` — which would switch the comparison off — is
    never set, and hwsched.toml is never written."""
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    outcome = client(fake_home, runner, monthly_budget_usd=250.0).plan(sweep)
    assert runner.env["HWSCHED_MONTHLY_USD"] == "250.0"
    assert "HWSCHED_ALLOW_OVERAGE" not in runner.env
    assert outcome.budget_override_usd == 250.0
    assert "monthly cap raised to $250.00" in outcome.summary()


def test_an_isolated_store_keeps_a_test_run_out_of_the_real_ledger(fake_home, sweep, tmp_path):
    pytest.importorskip("yaml")  # emitting the JobSpec needs the `compute` extra
    runner = Recorder(ok(PLAN_APPROVED))
    client(fake_home, runner, store_path=tmp_path / "runs.db").plan(sweep)
    assert runner.env["HWSCHED_STORE_PATH"] == str(tmp_path / "runs.db")


# --------------------------------------------------------------------------- #
# the remote entrypoints: what the container imports, and what comes back
# --------------------------------------------------------------------------- #
class TorchVersion(str):
    """Stand-in for torch's own version type, which is a ``str`` **subclass**.

    The subclassing is the entire hazard, so the fake reproduces it exactly: a value
    of this class pickles as a *reference* to ``torch.TorchVersion``, which means
    unpickling it imports torch wherever the value lands — including on a client
    that has no torch and no reason to.
    """


# pickle records a class by module and qualname, so the fake has to claim to live
# in `torch` for the round trip to break the way the real one did.
TorchVersion.__module__ = "torch"


def fake_torch() -> types.ModuleType:
    """A torch stand-in covering the surface :func:`gpu_probe` touches before it computes.

    A CUDA *build* with no visible device — precisely the case where a GPU job
    silently becomes a CPU job, and precisely as far as a fake can go: the matmul
    itself is real work on real hardware and is verified against Modal, not here.
    What is testable here is the shape of what comes back, which is what actually
    broke a paid run.
    """
    module = types.ModuleType("torch")
    module.TorchVersion = TorchVersion
    module.__version__ = TorchVersion("2.13.0+cu130")
    module.version = types.SimpleNamespace(cuda="13.0")
    module.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    return module


def _axon_source_root() -> Path:
    """The directory holding the ``axon`` package — what a subprocess needs on PYTHONPATH."""
    import axon

    return Path(axon.__file__).resolve().parents[1]


def _pickle_globals(value) -> list[str]:
    """Every class reference a pickle of *value* would make the client resolve."""
    names, stack = [], []
    for op, arg, _pos in pickletools.genops(pickle.dumps(value)):
        if op.name == "STACK_GLOBAL":
            names.append(" ".join(stack[-2:]))
        elif op.name == "GLOBAL":
            names.append(str(arg).replace("\n", " "))
        elif isinstance(arg, str):
            stack.append(arg)
    return names


def test_importing_the_remote_entrypoint_pulls_in_nothing_heavy():
    """hwsched mounts the entrypoint's whole top-level package and the container
    imports it, so a module-scope ``import torch`` would put a 2 GB wheel on the
    critical path of every CPU job — and a bare ``debian-slim`` probe would stop
    starting at all. Checked in a subprocess because this session has long since
    imported numpy for other tests."""
    code = (
        "import json, sys; import axon.compute.entry; "
        "print(json.dumps(sorted(m for m in "
        "('torch', 'numpy', 'pandas', 'yaml', 'modal', 'pydantic') if m in sys.modules)))"
    )
    env = dict(os.environ, PYTHONPATH=str(_axon_source_root()))
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []


def test_a_container_without_torch_reports_it_instead_of_raising(monkeypatch):
    """A spec that forgot ``pip=["torch"]`` builds and runs fine and only fails here.
    A raised task reaches nobody — the CLI collects no return values — so the miss
    has to come back as a receipt."""
    monkeypatch.setitem(sys.modules, "torch", None)  # any `import torch` now fails
    receipt = gpu_probe(seed=1)

    assert receipt["cuda_available"] is False
    assert "torch is not installed" in receipt["error"]
    # Still identified: a chunk of eight tasks has to say *which* one missed.
    assert receipt["seed"] == 1


def test_a_gpu_job_that_landed_on_a_cpu_says_so(monkeypatch):
    """The failure this probe exists for: the GPU request does not land, torch falls
    back to CPU, the job succeeds, it costs GPU money and proves nothing."""
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    receipt = gpu_probe(seed=2)

    assert receipt["cuda_available"] is False
    assert receipt["device_count"] == 0
    assert "the GPU request did not land" in receipt["error"]
    assert "checksum" not in receipt


def test_the_gpu_receipt_carries_no_type_the_client_would_have_to_import(monkeypatch):
    """Modal pickles a task's return value back and unpickles it on the *client*.

    ``torch.__version__`` is a ``TorchVersion`` — a ``str`` subclass — so returning
    it unwrapped fails the run with "Deserialization failed because the 'torch'
    module is not available in the local environment" *after* the GPU work has
    completed and been paid for. Plain-typed, not merely small, is the rule.
    """
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    receipt = gpu_probe(seed=3)

    assert receipt["torch"] == "2.13.0+cu130"
    for key, value in receipt.items():
        assert type(value) in (str, int, float, bool, type(None)), f"{key} is a {type(value)}"

    # The mechanical form of the same statement: a pickle with no GLOBAL opcode asks
    # the client to import nothing at all.
    assert _pickle_globals(receipt) == []
    # …and the negative control, so the assertion above cannot pass vacuously: the
    # unwrapped attribute really does drag torch across the wire.
    assert _pickle_globals(torch.__version__) == ["torch TorchVersion"]

    monkeypatch.delitem(sys.modules, "torch")
    assert pickle.loads(pickle.dumps(receipt)) == receipt


def test_every_exit_path_leaves_its_receipt_on_the_volume(monkeypatch, tmp_path):
    """hwsched's CLI never passes ``collect_results``, so a receipt that is only
    returned is discarded when the container exits — including the one explaining
    why a GPU job found no GPU."""
    monkeypatch.setattr("axon.compute.artifacts.MOUNT_ROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", fake_torch())

    receipt = gpu_probe(seed=4, artifact=volume_uri("axon-verify", "gpu/failed.json"))
    written = json.loads((tmp_path / "axon-verify" / "gpu" / "failed.json").read_text())

    assert "the GPU request did not land" in written["error"]
    assert receipt["artifact_bytes"] == len(json.dumps(written, sort_keys=True).encode())
    # The file describes the run, not itself.
    assert "artifact" not in written

    cpu = probe(seed=4, rounds=16, artifact=volume_uri("axon-verify", "cpu/receipt.json"))
    assert json.loads(
        (tmp_path / "axon-verify" / "cpu" / "receipt.json").read_text()
    )["digest"] == cpu["digest"]


# --------------------------------------------------------------------------- #
# against the real CLI — offline (plan spends nothing), skipped when absent
# --------------------------------------------------------------------------- #
#: Why these tests skip under ``./run.sh`` but pass under a bare ``python -m pytest``
#: — a five-test discrepancy (388 passed / 11 skipped versus 393 / 6) that nothing
#: else in the tree explains, and that has already confused one person counting.
#:
#: ``axon.compute`` falls back to bare ``python3`` when ``$AXON_HWSCHED_PYTHON`` is
#: unset. ``./run.sh`` activates ``.venv``, whose ``python3`` has no ``pydantic``, so
#: ``-m hwsched`` dies on import; ``/usr/bin/python3`` *does* have pydantic, so a
#: bare run finds it and the same five tests execute. Nothing is broken either way,
#: and the skip is the honest outcome — but it is an unset variable, not a missing
#: dependency, and the message has to say so or the next person installs the wrong
#: thing into the wrong interpreter.
_INTERPRETER_HINT = (
    "set $AXON_HWSCHED_PYTHON to an interpreter that can import hwsched — it needs "
    "pydantic, and on this box ~/hardware-scheduler/dashboard/backend/"
    ".venv/bin/python is the one with both pydantic and modal. Unset, axon.compute "
    "falls back to bare `python3`, which under ./run.sh is .venv's: that, and only "
    "that, is why the gate reports five more skips than a bare pytest run"
)


def _hwsched_home():
    from axon.compute.client import DEFAULT_HOME, HOME_ENV

    home = Path(os.environ.get(HOME_ENV) or DEFAULT_HOME)
    return home if (home / "hwsched" / "__init__.py").is_file() else None


def _hwsched_python() -> str:
    """The interpreter ``axon.compute`` would use — resolved the same way it resolves it.

    Deliberately not "whatever interpreter happens to work": a test that quietly
    picked a better one than the library does would go green over a default that
    cannot run hwsched at all.
    """
    from axon.compute.client import PYTHON_ENV

    return os.environ.get(PYTHON_ENV) or "python3"


def _real_client(tmp_path) -> HwschedClient:
    """A client pointed at the real checkout, or a skip.

    ``fake`` provider and a throwaway run store: the CLI is exercised for real, but
    nothing reaches Modal and the actual ledger is neither read nor written.
    """
    home = _hwsched_home()
    if home is None:
        pytest.skip("no hwsched checkout; set AXON_HWSCHED_HOME to run the CLI tests")
    api = HwschedClient(home=home, provider="fake", store_path=tmp_path / "runs.db")
    try:
        api.budget_state()
    except (HwschedError, OSError) as exc:
        pytest.skip(f"hwsched CLI not runnable with {api.python}: "
                    f"{str(exc).splitlines()[0]} — {_INTERPRETER_HINT}")
    return api


def test_real_hwsched_plans_the_emitted_spec(tmp_path, sweep):
    """End of the line for the emitter: the real parser accepts it, the real planner
    sizes it, and the correlation id survives the round trip."""
    outcome = _real_client(tmp_path).plan(sweep)

    assert outcome.exit_code == 0
    assert outcome.spec["env"][CORRELATION_ENV] == sweep.resolved_correlation_id
    assert 0 < outcome.cost.low <= outcome.cost.expected <= outcome.cost.high
    assert outcome.plan["device"] == "cpu"
    assert outcome.plan["n_tasks"] == 18
    assert outcome.rationale
    assert not outcome.errors


def test_real_hwsched_refuses_an_unaffordable_plan_without_raising(tmp_path):
    """The refusal path, end to end, for free: a GPU training job under a 5-cent cap."""
    job = train_model("axon-lstm-train", "axon.models.train_dnn:fit", framework="torch",
                      data_size_gb=40.0, timeout="6h", max_usd=0.05)
    outcome = _real_client(tmp_path).plan(job)

    assert outcome.exit_code == 2
    assert outcome.refused
    assert outcome.plan["device"] == "gpu"
    assert outcome.cost.high > 0.05
    with pytest.raises(BudgetRefused):
        outcome.approve(max_usd=100.0)
