# Research — offloading Axon's heavy compute to `hwsched` / Modal

**Question:** Axon will need serious compute for Phase 5 (model training, HPO sweeps,
walk-forward backtests) and possibly for long-running market-data capture. A sibling
project, `hwsched` (`C:\Users\Lenovo\Documents\hardware-scheduler`), already plans and
schedules Modal jobs against a budget. Should Axon use it, and for what?

**Verdict up front:** yes — but for **Phase 5, not now**, and **not** for market-data
capture. Wiring it in today would be building a consumer for a producer that doesn't
exist yet. The concrete thing to do in Phase 5 is shell out to the `hwsched` CLI with
`--json`; nothing needs to change in either repo before then.

*Surveyed 2026-07-25 against `hwsched` at "Phases 0–4 functionally complete", 360 tests
passing, live Modal path verified 2026-07-24.*

## What hwsched actually is

A Python library + CLI that turns a declarative `JobSpec` (YAML) into a `ResourcePlan`
(GPU type / CPU cores / memory / worker count / chunking), prices it, checks it against
a monthly budget, and submits it to Modal. It has a rule-based planner with a
human-readable rationale, a persisted priority queue with admission control, failure
classification with retry/escalation ladders, a recurring (cron) scheduler, and a DAG
runner. There is also a read-only React dashboard fed by live Modal billing.

Its design principles are strict about the parts that matter here: dry-run before
spend, budget as a hard ceiling that can downgrade or refuse but never be bypassed, and
provider-agnosticism (nothing outside `providers/` imports the Modal SDK).

## The budget reality

Modal Starter: **$30/month of compute credits, monthly reset, no roll-over, hard stop**
(workloads stop rather than silently charge). hwsched is configured with
`monthly_usd = 30.0`, `reserve_usd = 3.0`, `per_job_max_usd = 5.0`,
`allow_overage = false`, and a `1.2×` safety multiplier applied as a floor on the
*high* estimate — the guard always compares `est.high`, never `expected`.

What $30 buys at base (preemptible, region-agnostic) rates: **≈50 h T4, ≈12 h
A100-80GB, ≈7.6 h H100**. CPU is $0.0000131/core/s (min request 0.125 cores) and
memory $0.00000222/GiB/s. Non-preemptible is **3×**; pinning a region is 1.5–1.75×;
they stack. Volume storage is $0.09/GiB-month with the first 1 TiB free — effectively
free for our artifact sizes.

*Rates as-of 2026-07-12 per `hwsched/docs/reference/modal-facts.md`; that file is the
single source of truth and is the thing to re-check, not memory.*

## Why NOT market-data capture

This was the intuitive idea — park a container on Modal recording Hyperliquid WS
frames for later training and replay tests. It's the wrong tool, for four reasons:

1. **hwsched is batch-only.** Every path is submit → poll to terminal → collect
   telemetry. There is no streaming, daemon, or long-running job model, and no
   keep-warm. `Provider.results`/`collect_telemetry` are defined only for a terminal
   job.
2. **24 h is a hard container ceiling**, enforced by Modal *and* by hwsched in two
   places (planner clamp + a blocking `timeout_high` validation error). Continuous
   capture would need decomposing into ≤24 h checkpointed segments chained by the
   recurring scheduler — and hwsched provides no segment-chaining or checkpoint/resume
   helper.
3. **The economics are bad.** A minimal 0.125-core / 512 MiB container running 24/7 for
   a month costs roughly `(0.125·1.31e-5 + 0.5·2.22e-6) · 2.6e6 s ≈ $7` — nearly a
   quarter of the entire monthly budget, before a single training run. It also holds one
   of the 100 workspace container slots for its whole life.
4. **The budget guard can't see it.** Estimates are driven by `est_task_time_s`, so a
   job with indefinite runtime is mis-estimated by construction — precisely the "plan
   that silently risks overspend" that hwsched treats as a bug.

Capture belongs where the socket is: a local process or a cheap always-on VPS writing
to disk, near the venue. That is a latency-and-uptime problem, not a scheduling one.

## Where it genuinely fits (Phase 5)

hwsched already has first-class workload types matching what Axon Phase 5 needs:

| Axon Phase 5 need | hwsched `workload` | Planner behaviour |
| --- | --- | --- |
| Hyperparameter sweeps | `param_sweep` | CPU fan-out, Cartesian `params` grid, chunked to ~60 s/container |
| Walk-forward backtests | `walk_forward` | CPU fan-out, same chunking |
| NN model training | `ml_train_dnn` | forced GPU; GPU-framework imports also force GPU |
| Feature-matrix builds | `etl` / `feature_eng` | big-memory CPU |
| Tree models (XGBoost/LightGBM) | — | stay CPU unless explicitly pinned |

This is a good match for the **parity harness** ([07](../07-parity-and-testing.md)) too:
a fan-out of replay/parity runs across many windows is exactly `param_sweep` shaped.

## How Axon would call it

There is **no Rust binding, no HTTP submit API, and no gRPC surface** — hwsched is a
Python library plus a CLI. Three integration shapes exist; for a Rust caller the CLI is
the robust one:

- **CLI + `--json`.** Every subcommand accepts `--json`, and the plan/run JSON is a
  documented stable contract: `{spec, plan, cost:{low,expected,high,breakdown,...},
  budget:{decision,remaining,cap,...}, violations, rationale, confidence}`. Exit codes
  are machine-parseable: `0` ok, `2` budget-refused, `3` validation, `4` provider error.
  The flow is: write a YAML `JobSpec` → `hwsched plan f.yaml --json` (cost preview, no
  spend) → `hwsched run f.yaml --json`, or `submit` + `dispatch --once --json`.
- **Artifact hand-off via Modal Volumes** — `volume://<name>/<subpath>` in
  `JobSpec.inputs`/`outputs`, mounted at `/vol/<name>`. This is the right channel for
  feature matrices and model files; DAG stages wire them automatically.
- **Observability** — the SQLite run store (`.hwsched_runs/runs.db`) is directly
  queryable from Rust with any sqlite driver, so a parity report can join Axon's own
  results against run outcomes and plans.

Note `JobSpec` uses `extra="forbid"` and has no `metadata`/`tags` field, so correlation
IDs must ride in `args`, `kwargs`, `env`, or `idempotency_key`.

## Constraints to design around when we get there

- **Nothing drives itself.** `dispatch`, `schedule tick`, and DAG progress only advance
  when something calls them. Axon (or cron/systemd) must own the polling loop.
- **`hw.run()` blocks** the calling thread up to 24 h and the public API doesn't expose
  the wait bound; use `submit()` + `dispatch()` for non-blocking behaviour.
- **Actual cost and utilization never come back from Modal synchronously** —
  `actual_cost_usd`, `cpu_util`, `gpu_util` are all `None`. The ledger reconciles the
  *estimate*; real spend has to be read from Modal billing separately (billing is
  rate-limited and the current day lags).
- **Caps are structural, per-workspace**: 10 concurrent GPUs, 100 concurrent
  containers, `.map()` ≤1000 concurrent, 25,000 total inputs, 5 deployed crons.
- **Pipelines cannot be queued** (`submit` rejects a `PipelineSpec`), and there is no
  `reproduce` command — re-running a past run with a pinned image/seed/spec would have
  to be built if the parity harness needs audit replay.

## Decision

Record no ADR yet — there is nothing to decide until Phase 5 produces a training or
backtest job worth offloading. When it does:

1. Add an `axon.compute` Python helper that emits a `JobSpec` YAML and shells out to
   `hwsched plan --json` first (dry-run, no spend), surfacing `rationale` and
   `cost.high` before anything is submitted.
2. Route feature matrices and model artifacts through a Modal Volume, not through job
   return values (payloads must stay small and picklable).
3. Keep the parity gate's own pass/fail logic in Axon; hwsched decides *where* the
   compute runs, never *whether the model is good enough*.

Until then the useful takeaway is the budget arithmetic above: Phase 5's compute plan
has to fit in ≈$27/month usable, which favours CPU fan-out and T4/L4-class GPUs over
anything A100-and-up.
