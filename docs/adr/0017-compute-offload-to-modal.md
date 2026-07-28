# ADR-0017 — Offloading heavy compute to Modal through `hwsched`

**Status:** Accepted · **Date:** 2026-07-25 · **Amended:** 2026-07-26

> The 2026-07-26 amendment adds to **Verification** what the GPU submission path did
> when it was finally run, and corrects the *argument* in decision 4 — not its
> conclusion, which stands. Both additions are marked with their date inline. The
> Decision section is otherwise unchanged: this ADR is Accepted and its decisions
> were not revisited.

## Context

Phase 5 is where Axon starts needing compute this box does not have: hyper-parameter
sweeps, walk-forward backtests across many windows, model training, feature-matrix
builds. None of it is on the latency path, all of it is embarrassingly parallel or
GPU-bound, and all of it wants to be somewhere else.

A sibling project, `hwsched`, already solves the part that is not Axon's problem: given
a declarative job, decide *where* it runs (GPU or CPU, which GPU, how many workers, how
to chunk), price it, check it against a budget, submit it to Modal and record the
outcome. [`research/compute-offload-hwsched.md`](../research/compute-offload-hwsched.md)
surveyed it and concluded *yes, but in Phase 5, and record no ADR yet — there is nothing
to decide until Phase 5 produces a job worth offloading*.

**This ADR supersedes that position.** Not because the survey was wrong about the
sequencing, but because of what "the first job worth offloading" means in practice: the
first expensive job is precisely the one that must not be an accident. Building the
consumer *before* there is anything expensive to run is what makes the dry-run-before-
spend rule cheap to enforce; building it after means enforcing it retroactively on code
that already works without it.

Three of the survey's facts were stale, and each was re-verified against the checkout on
2026-07-25 before a line of integration was written:

**1. Where hwsched lives, and how it is reached.** Not the Windows path the survey gives.
It is at `~/hardware-scheduler` and it is **not pip-installed**:
`python3 -m hwsched` resolves only with that directory on `sys.path`. The working
directory matters twice more, and both are silent when wrong — `load_config` searches
`Path.cwd()` for `hwsched.toml`, and the SQLite run store path (`.hwsched_runs/runs.db`)
is relative. Run it from the wrong directory and the job is planned against default
config and an empty ledger: a budget that has forgotten every dollar already committed.

**2. `reproduce` exists in `--help` and does nothing.** `python3 -m hwsched --help` now
lists it, but invoking it prints *"planned for a later phase"* and exits **3**. The
surface changed; the capability did not, and audit replay of a past run still has to be
built if the parity harness needs it. What *has* genuinely landed since the survey:
`JobSpec.pip` (extra requirements layered on the image), `run(collect_results=True)` plus
a `Provider.results` contract, `run --allow-scope-cut` and `--llm`, `Provider.cleanup`
(stop a finished run's deployed app), pipeline schedules, native-cron slot tracking,
per-task partial-sweep retry, and a dashboard that merges live Modal billing with the
run store. The live Modal path is now verified with dependency-bearing fan-outs, not
just a dep-free smoke test.

**3. The $30 arithmetic is no longer the constraint.** The user states budget is not a
constraint, and `~/.modal.toml` holds **fourteen** profiles (active profile is whichever `modal profile current` reports).
hwsched still has one `hwsched.toml` with `monthly_usd = 30.0`, and its ledger is
per-checkout, not per-profile — so an exit-2 refusal is a statement about one configured
number against one ledger, not about whether the work is affordable. It is worse than
that: Modal returns no synchronous cost, so the ledger reconciles the *estimate*, and
after three real Modal runs `hwsched budget` still reports `spent_this_month: 0.0`. The
guard is a deliberate speed bump, not ground truth, and Axon must treat it as such —
without ever routing around it.

## Decision

**1. `axon.compute` consumes hwsched through its CLI, as a subprocess, and owns nothing
of its internals.** Every call is `<python> -m hwsched <cmd> --json` with the checkout as
both a `PYTHONPATH` entry and the working directory. The home, the interpreter and the
provider are configurable (`AXON_HWSCHED_HOME`, `AXON_HWSCHED_PYTHON`,
`AXON_HWSCHED_PROVIDER`), defaulting to the paths above.

Not an import, even though both sides are Python. Importing would pull pydantic and
hwsched's whole dependency tree into Axon's research plane and turn a documented JSON
contract plus a four-value exit-code taxonomy into an import graph — the same reasoning
that keeps venues behind an adapter in [ADR-0004](0004-provider-abstraction-layer.md).
The cost is real (an interpreter start per call, errors arriving as text on stderr) and
worth paying.

Two details of that subprocess are load-bearing and neither is obvious. `PYTHONPATH`
carries the directory holding the `axon` package as well as the checkout, because
hwsched's Modal adapter calls `Image.add_local_python_source(<top package of the
entrypoint>)` and resolves that package on the **client's** `sys.path` — an `axon.*`
entrypoint that is not importable in the subprocess fails at submit, after the app has
been deployed. And `PATH` is prefixed with the interpreter's own `bin/`, because
hwsched's post-run app cleanup shells out to a `modal` executable found with
`shutil.which`; driven from Axon with a venv interpreter, that lookup fails, the
best-effort cleanup silently no-ops, and every run leaves a lingering `deployed` app.
Both were found by running it, not by reading it.

**2. The Axon correlation id rides in `env`, and it is content-addressed.** `JobSpec`
uses `extra="forbid"` and has no `metadata`/`tags` field, so the id has to travel in
`args`, `kwargs`, `env` or `idempotency_key`. `args` is never read by the Modal adapter;
`kwargs` is passed to the user function and is *dropped entirely* for a fan-out, because
tasks are materialized from `params`/`tasks`; `idempotency_key` is load-bearing, and
overwriting it with a run id would disable "don't re-run an identical completed job"
outright. `env` reaches the container as a real environment variable and is folded into
hwsched's derived idempotency key.

That last property is the trap, and it decides the shape of the id. Because `env` feeds
the dedup hash, a fresh UUID per submission would give every call a new key and turn the
idempotency cache off by accident. So the id is a **digest of exactly the fields hwsched
itself hashes** — it changes when and only when hwsched's own key would have changed, and
carrying it costs nothing. One field is added on purpose: `outputs`, which hwsched omits,
so two runs writing different artifacts would otherwise hash identically and the second
would be skipped as a duplicate.

**3. Dry run before spend is a type, not a convention.** `plan()` is free, never submits,
and always returns — a budget refusal is a *result*, because a planner that throws on a
refusal cannot tell you how far over the cap you are. `run()` will not execute without an
`Approval`, and an `Approval` can only be minted by `PlanOutcome.approve(max_usd)`, which
requires a ceiling at or above the plan's own `cost.high`. Naming that number is the
mechanical form of "the estimate was surfaced"; the summary (cost range, budget decision,
rationale, violations) is also logged at INFO, so it reaches an operator even when the
caller drops the return value.

The approval is bound to the job's content digest and is single-use. Both guard a
specific accident: a plan for a 16-point CPU sweep must not authorize the A100 job
someone edited it into, and one reviewed estimate inside a retry loop must not become
forty unreviewed submissions. The accepted ceiling is forwarded to `--max-spend`, so the
guard re-enforces it at submit against a ledger that may have moved since the dry run.
This is the same move as [ADR-0010](0010-execution-events-and-reconciliation.md)'s
`GuardedClient`: a gate callers are merely expected to consult is one forgotten call site
away from the thing it was protecting against.

**4. Artifacts move on Modal Volumes; return values are receipts.** `inputs` and
`outputs` must be `volume://<name>/<subpath>` — validated in Axon, which is stricter than
hwsched — and the adapter mounts each at `/vol/<name>`. The return path is structurally
unable to carry a feature matrix: the CLI never passes `collect_results`, so per-task
values are never fetched at all. Subpaths are validated against `..` and absolute paths
because that failure is silent: a write outside the mount succeeds against ephemeral
container disk, and the artifact is simply gone when the container exits.

> **Corrected 2026-07-26, after the GPU path ran.** The conclusion above holds —
> artifacts move on Volumes — but the sentence *"per-task values are never fetched at
> all"* is wrong, and the constraint it implies is the wrong constraint.
>
> **Return values are fetched, just never delivered.** `Provider.results` is indeed
> never called by the CLI. But `ModalProvider.status()` polls every spawned
> `FunctionCall` with `fc.get(timeout=0)`, and `get` **deserializes**; any exception
> it raises is caught and counted as a *failed task*. A return value the client cannot
> unpickle therefore does not vanish quietly — it fails a run whose compute already
> succeeded, and reports the failure as if the task had crashed.
>
> **And the binding property is the value's type, not its size.** `torch.__version__`
> is a `TorchVersion`: a `str` **subclass**. Pickling one emits a `STACK_GLOBAL`
> naming `torch`, so unpickling it imports torch *on the client* — the box that
> deliberately has no torch, which is the entire reason the work was offloaded. The
> observed failure was `Deserialization failed because the 'torch' module is not
> available in the local environment`, raised **after** 8/8 GPU tasks had completed
> and been paid for. The Volume artifacts survived it; the return value did not.
>
> So the rule an entrypoint is held to is **plain-typed, not merely small**: every
> field of a receipt is a `str`/`int`/`float`/`bool`/`None`, and every exit path
> writes the receipt to its Volume *before* returning it, because the channel that
> can fail is not the channel that carries the evidence.

**5. A refusal is surfaced, never bypassed.** `plan()` reports it; `run()` raises a typed
`BudgetRefused` carrying `decision`, `remaining`, `cap`, the guard's message and
`cost.high`, and says plainly that nothing was spent. The one sanctioned lever is
`HwschedClient(monthly_budget_usd=…)`, which sets hwsched's own `HWSCHED_MONTHLY_USD`
override for the subprocess: it moves the cap the guard enforces, it does not switch the
comparison off, it is opt-in per client, and it is echoed on every outcome so it can
never be in force unnoticed. `HWSCHED_ALLOW_OVERAGE` — which *would* disable the guard —
is deliberately not exposed, and `axon.compute` never writes to `hwsched.toml`.

**6. hwsched decides where compute runs, never whether a model is good enough.** The
parity gate stays in `axon.parity` ([07](../07-parity-and-testing.md)). A scheduler that
could fail a model would be a second, invisible gate on production decisions.

**7. Workload and framework are declared, never inferred.** hwsched's profiler infers
device from the entrypoint module's imports, but resolves that module relative to *its
own* working directory — which is the hwsched checkout, not Axon's. Static introspection
therefore always misses Axon's source and the declared signals are the only ones the
planner ever sees. A training job that forgets `workload: ml_train_dnn` silently plans as
CPU, so `train_model()` refuses to guess a workload for an unrecognized framework:
guessing GPU is the expensive direction to be wrong in, and guessing CPU is the wrong one.

## Verification

### 2026-07-25 — the CPU path

Run for real, in order, on 2026-07-25:

- **Plan (free).** The emitted sweep spec through `hwsched plan --json`: 18 tasks, CPU,
  5 workers, `$0.0120 / $0.0137 / $0.0191` low/expected/high, `approve` against a $0.50
  per-job cap. A GPU training spec under a $0.05 cap returned exit **2** with
  `decision: refuse`, `cost.high: $1.746`, and the guard's own remediation text — surfaced
  by `plan()`, not raised.
- **Fake provider.** A four-task sweep submitted through `axon.compute` end to end:
  `status: succeeded`, `budget_decision: approve`, isolated run store.
- **Real Modal.** Three genuinely small CPU jobs (`axon.compute.entry:probe`, ~2.4 CPU-s
  each), driven from Axon on the active Modal profile, all `succeeded` —
  e.g. app `ap-JK0Zprt2CkG1XS6aem0jpk`, call `fc-01KYD5C2F7BVRWMXJH67BFMH8Y`. The container
  received the correlation id as an environment variable and wrote its receipt to the
  Modal Volume, retrieved afterwards:
  `{"correlation_id": "axon.axon-modal-probe.72f8c517f5bbfe90", "digest":
  "e1262587c3549c4c432ef7b68819367b", "elapsed_s": 2.44, "machine": "x86_64", "python":
  "3.12.10", "rounds": 8000000, "seed": 7}` — a digest identical to the local run, so the
  remote result is bit-for-bit reproducible. `actual_cost_usd` came back `None`, as
  expected.

The `PATH` fix in decision 1 was found here: the first two runs left `deployed` apps
behind; with the interpreter's `bin/` on `PATH` the third was stopped automatically. That
also live-verifies hwsched's own B7 cleanup, which its changelog still lists as pending.

### Added 2026-07-26 — the GPU path, which until now had never executed

Everything above is CPU. The only GPU exercise this ADR originally recorded was a *plan*
the budget guard deliberately refused, so every step between "the planner says `device:
gpu`" and "a container with a device attached returned a result" was reasoned about
rather than seen — the same distinction the roadmap draws between *offline verified* and
*proven live*, and it applied to this path too.

It has now run four times: twice at width from a throwaway package outside the repo, and
twice from the entrypoint that now lives in it.

- **8/8 tasks on Tesla T4s, twice, bit-identical.** Two independent submissions of an
  8-task fan-out (2048×2048 fp32 matmul, 200 iterations, one seed per task, `workload:
  inference`, `resources.gpu_type: T4`) returned **8/8 succeeded** and **8/8 matching
  checksums** across the two runs — e.g. seed 0 `43054397.976621315`, seed 1
  `78255972.45740408`. The second: planned `$0.0093 / $0.0103 / $0.0139`, `approve`, app
  `ap-B0EuCoFDZ0Vu4NJ6QmBpML`, call `fc-01KYDQW02N1YTMNFJ111MHJXZM`, run
  `915b08184eeb49f882535b7fee105e2b`. Containers reported `Tesla T4`, compute capability
  `7.5`, 14.56 GiB, torch `2.13.0+cu130` on CUDA `13.0`, Python `3.12.10`, ~4.2 TFLOP/s
  fp32 at a 0.117 GiB peak. Reproducibility is not free here: **TF32 is disabled
  explicitly**, because it silently drops matmul mantissa bits on Ampere and later, and a
  checksum that changes with whichever GPU class Modal happened to schedule verifies
  nothing. Both ran from a throwaway package *outside* the repo — which is what made the
  next bullet necessary rather than decorative.
- **The same work, from `axon.compute.entry:gpu_probe` in this tree.** 2 tasks, planned
  `$0.0066 / $0.0073 / $0.0099` low/expected/high, `approve`, submitted against a
  `$0.0119` approved ceiling — run `aef943ccf8634176811294820e842385`, app
  `ap-6HhTQ9Us9bxF080QRD5t1s`, call `fc-01KYEKJBGZG0AK19SCFFVKBG1E`, `succeeded`
  (and once before it at `ap-dPUmCY4OHSosymOsOFLfwn`, re-run so that the source in the
  tree is exactly the source that ran). Seeds 0 and 1 returned checksums **identical to
  both out-of-tree runs**, so moving the probe in-tree changed nothing about the
  arithmetic. That is not the interesting part. The mounted package is now `axon` itself,
  so `add_local_python_source("axon")` executed `axon/__init__.py`,
  `axon/compute/__init__.py` and `entry.py` inside a `debian-slim + torch` image — the
  property decision 1's import discipline claims, previously exercised only by a CPU job,
  and the reason `import torch` sits *inside* `gpu_probe` rather than at module scope.
  `actual_cost_usd` came back `None`, again, and both apps show `stopped`, so the `PATH`
  cleanup above still fires when the client is driven with `AXON_HWSCHED_PYTHON` set.

Three things were learned that are not deducible from the code, and each cost an hour:

**1. hwsched's `workload` label *is* the price, not just the placement.** The label alone
selects the **duration** model. The `resources` pins set which hardware is billed; they
say nothing about for how long, and the how-long term is the one that moves. The same GPU
probe, same pins, same two tasks, one word different:

| declared `workload` | `est_task_time_s` | cost low/expected/high |
|---|---|---|
| `inference` | 2 s | `$0.0066 / $0.0073 / $0.0099` |
| `ml_train_dnn` | **1800 s** | `$0.8322 / $0.9195 / $1.2413` |

That is **900× on the assumed runtime** and a 125× spread on the bill — the gap is
narrower than the ratio only because a fixed $0.0063 cold start dominates the cheap plan.
At the eight-task width first observed the spread was **350×**: `ml_train_dnn` planned at
**$4.91** and was refused outright by the guard, while `inference` planned at **$0.0139**
and ran. A 4-second probe declared as training is not a mispricing hwsched could have
caught — it is doing exactly what it was told. Decision 7 already says *declare, never
infer*; this is what declaring wrong costs, and the default leans expensive, because
`train_model(framework="torch")` picks `ml_train_dnn`. **The timeout does not enter the
estimate at all** — identical at 240 s and 1200 s — so a generous timeout is free, and it
is the wrong knob to reach for when a plan looks expensive.

**2. A return value must be plain-typed, not merely small.** See the correction under
decision 4: the failure is a `str` subclass, not a large object, and it lands after the
compute has been paid for.

**3. `sys.path` is necessary but not sufficient — the interpreter is the other half.**
`axon.compute` defaults to bare `python3`, and `python3` is whatever the shell resolves.
On this box the project's own `.venv/bin/python3` has **no `pydantic`**, so `-m hwsched`
dies on import before doing anything, and the traceback reads like a broken hwsched
checkout rather than a wrong interpreter. `~/modal_venv` fails the same way;
`/usr/bin/python3` has pydantic but no `modal`. The one interpreter here with both is
`~/hardware-scheduler/dashboard/backend/.venv/bin/python`, and
`AXON_HWSCHED_PYTHON` (decision 1) is exactly the knob for it. This is also why the gate
reports five more skipped tests than a bare `pytest` run: `./run.sh` activates `.venv`,
which shadows the `/usr/bin/python3` a bare run would have found. The five skips are an
unset environment variable, not a missing dependency, and the skip messages in
`python/tests/test_compute.py` now say so — with the variable set, all five pass.

The two properties the probe depends on are held by the **offline gate**, not by this
document. `test_importing_the_remote_entrypoint_pulls_in_nothing_heavy` imports
`axon.compute.entry` in a subprocess and fails if `torch`, `numpy`, `pandas`, `yaml`,
`modal` or `pydantic` turns up in `sys.modules`.
`test_the_gpu_receipt_carries_no_type_the_client_would_have_to_import` builds a receipt
against a fake torch whose `__version__` is a `str` subclass, and fails unless a pickle of
that receipt contains **no `GLOBAL` opcode at all** — the mechanical form of "the client
imports nothing to read this" — with the unwrapped attribute asserted to *fail* the same
check, so the test cannot pass vacuously. Both were confirmed to redden before they were
believed: a module-scope `import numpy` breaks the first, dropping one `str()` breaks the
second. Neither needs torch, a GPU or a network, so verifying the GPU path cost the gate
nothing.

The ledger, meanwhile, still reports `spent_this_month: 0.0` after every run above. It
reconciles estimates, and Modal returns no synchronous cost; treat it as the speed bump
decision 5 describes, never as ground truth.

## Consequences

- **+** An accidental A100 sweep cannot be launched. Not by policy — there is no code path
  from a `ComputeJob` to a submission that does not pass through a costed, approved plan.
- **+** The correlation id survives into the container *and* into hwsched's persisted spec
  and run record, so a parity report can join Axon's results against plans and outcomes.
- **+** The ledger stays shared. Axon's commitments are visible to every other hwsched
  caller and vice versa, which a private store would have broken.
- **+** Artifacts have exactly one channel, and both ends of it are derived from one
  parsed value, so a job cannot declare one path and write another.
- **−** `hwsched run` blocks until the job is terminal, up to Modal's 24 h container
  ceiling. `submit`/`dispatch` exist for non-blocking use and Axon does not drive them
  yet, so a long training run holds the calling process. Killing the client does not
  cancel the job.
- **−** Real spend is invisible from here. The ledger reconciles estimates; actual cost has
  to be read from Modal billing separately, which is rate-limited and lags the current day.
- **−** Per-task return values are unreachable over the CLI. A fan-out that wants them must
  write to a Volume and read it back — correct for anything large, ceremony for a scalar.
  *(Corrected 2026-07-26: unreachable, but **not** unevaluated. `status()` deserializes
  every chunk's result to decide whether it completed, so a value the client cannot
  unpickle fails the run anyway. Unreadable and harmless are different things.)*
- **−** The exit-code taxonomy is lossy in two places. Exit 3 also covers "unknown queue id"
  and "command not implemented", and argparse's own usage error is exit **2** — the same
  code as a budget refusal. They are told apart by the payload (a real refusal still prints
  its report on stdout), which is a heuristic, not a contract.
- **−** `JOBSPEC_FIELDS` duplicates hwsched's model. A test compares the two against the
  real model, but only when the checkout is present; on a machine without it, drift is
  invisible until a spec is rejected.
- **−** Modal's structural caps are unchanged and unmodelled here: 10 concurrent GPUs, 100
  containers, `.map` ≤1000, 25,000 total inputs, 5 deployed crons.
- **−** *(2026-07-26)* The `workload` string is the largest single lever on price and
  nothing in `axon.compute` validates it against what the job actually does — it cannot,
  because only the caller knows. `train_model(framework="torch")` defaults to
  `ml_train_dnn` on purpose (decision 7: guessing CPU is the wrong direction), so a short
  GPU job routed through it prices ~350× high and may be refused for being what it is not.
  Declaring `inference` for short GPU work is the fix, and it has to be a human choice.
- **−** *(2026-07-26)* The default interpreter is `python3`, which on this box cannot run
  hwsched at all. Every caller must set `AXON_HWSCHED_PYTHON` or supply `python=`, and
  nothing fails until the subprocess does. A default that is wrong in the common case is
  worse than no default; changing it is a one-line edit to `DEFAULT_HOME`'s neighbour in
  `axon/compute/client.py` and was deliberately not made here, because hard-coding one
  box's venv path into the library trades a loud failure for a silent, unportable one.
- **−** Market-data capture still does not belong on Modal, for the reasons the survey
  gives: batch-only, a 24 h hard ceiling, ~$7/month for a minimal always-on container, and
  an indefinite runtime the estimator cannot price. Capture belongs next to the socket.

See [research/compute-offload-hwsched.md](../research/compute-offload-hwsched.md) (the
survey this implements and whose "no ADR yet" it supersedes),
[ADR-0004](0004-provider-abstraction-layer.md) (adapters behind one interface),
[ADR-0010](0010-execution-events-and-reconciliation.md) (the "gate as a type" pattern
reused here), [07](../07-parity-and-testing.md) (where pass/fail stays).
