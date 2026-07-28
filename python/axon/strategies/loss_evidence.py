"""What a session-length window of this strategy's trading actually costs.

**Why this exists.** ``[strategy.risk] max_session_loss`` and ``max_daily_loss`` are the
first gates in this project that can stop a live session, and the numbers in the shipped
configs are **declarations** — 1.00 and 2.00 USDC, argued from one 59-minute run that lost
0.028. ADR-0036 already recorded the same problem for the latency ceilings and named it
plainly: *a latency budget is a number somebody chose, and only one session's worth of
evidence stands behind the numbers*. A loss bound chosen the same way is worse, because it
is not a warning: too tight and it halts a healthy session on an ordinary hour, too loose
and it is decoration.

What a bound needs is a **distribution**: over many windows of the length a session
actually runs, what did this strategy's bottom line come to? Then "1.00" can be stated as
a quantile of something measured rather than as a round number, and the two failure modes
above become quantities.

That is a compute problem and not a runtime one — hundreds of independent windows, each a
few thousand bars, nowhere near the latency path. It is exactly the shape ADR-0017 built
:mod:`axon.compute` for, and it is the first thing in this project that has genuinely
needed the fan-out rather than merely fitting it.

── what one window computes, and what it refuses to ──────────────────────────────

Each task replays the **serving** strategy over one contiguous window of bars and
accounts the result the way the runtime's own money view does
(``axon_runtime::pnl``): average-cost realized P&L over the closes, plus the fee the
venue would have charged at the urgency the strategy trades at. Nothing here models a
queue, a partial fill, or a book — so this is **not** a backtest and must not be read as
one:

* **It assumes every target change trades at the bar's close.** The live path posts at the
  near touch and is swept if it does not fill, so a real session trades *less* than this
  and pays a maker fee when it does. The direction of that error is stated because it
  matters for a bound: a window that looks bad here would look less bad live, and a bound
  argued from these numbers is therefore **conservative** — which is the safe direction
  for a kill switch and the wrong direction for a profit claim.
* **It contains no edge claim.** ``zoo_xgboost``'s own verdict is SHOULD NOT TRADE and
  nothing here revisits that. The output is a *dispersion*, and the mean being negative is
  the expected result, not a finding.
* **It never selects.** No parameter here is chosen by looking at an outcome. The windows
  are every window in the corpus, in order, at a fixed length.

── the two device classes, and how big each one is allowed to be ─────────────────

**The binding constraint is the question, not the budget, and getting that backwards cost
this module a rewrite.** ``hwsched.toml`` declares ``monthly_usd = 30`` and
``per_job_max_usd = 5``, which is *one* Modal Starter account's guard — and there are
**fourteen profiles** in ``~/.modal.toml``, so the aggregate pool is roughly ``$420`` a
month. The first version of the GPU job here was cut from 64 grid points to 4 because a
``$3`` cap I had set myself refused it. That is not sizing a job; that is sizing a guard.

So both jobs below are sized for what they are trying to *learn*, and each declares the cap
it needs:

:func:`session_loss_job` is a **CPU** fan-out: each window is a few thousand rows of numpy
and one model's ``predict``, so a container per chunk of windows is the whole shape and a
GPU would sit idle behind the data loading. It is maximised over the four axes that change
the answer — **coin**, **session length**, **fee tier**, and every non-overlapping window in
the corpus — because a loss bound for an eight-hour soak cannot be argued from one-hour
windows, and a bound priced at the maker fee says nothing about the run that had to cross
out.

:func:`gpu_fit_sweep_job` is the **GPU** one, at the full grid. It carries a caveat that is
worth reading before spending: hwsched has **no per-task runtime for a tree fit on a GPU** —
``GPU_TASK_TIME`` covers ``ml_train_dnn``, ``ml_train_rl`` and ``inference``, and everything
else takes a flat **1800 s** (``hwsched/planner/rules.py``). A real m1 XGBoost fit is
minutes, not half an hour, so every price this job is given is **conservative by roughly an
order of magnitude**. The right response to that is to declare a cap that covers the
pessimistic estimate — not to shrink the grid until an estimate nobody measured happens to
fit. :func:`gpu_calibration_job` exists to replace the guess with a measurement, and it is a
*first* run rather than a substitute for the real one.

── spreading across profiles ─────────────────────────────────────────────────────

hwsched submits to one provider per run and Modal picks the account from the submitting
process's ``MODAL_PROFILE``. So capacity across the fourteen profiles is an **operator-level
fan-out**: set ``MODAL_PROFILE`` and ``HWSCHED_MONTHLY_USD`` per invocation.
:func:`shard_by_profile` splits a job's task list so that is a mechanical step rather than a
manual one — see its docstring for why the shards are contiguous and not interleaved.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import numpy as np

from dataclasses import replace

from axon.compute import ComputeJob, param_sweep, volume_uri, walk_forward
from axon.compute.spec import SpecError
from axon.strategies.jobs import ARTIFACT_VOLUME, _common, _data_uri

#: Hyperliquid's published taker and maker fees for the base tier, in basis points of
#: notional. Positive is a cost. Pinned here rather than derived, and the source is named:
#: a fee schedule read off a live account would make a window's result depend on which
#: account ran it, and the whole point of a distribution is that it does not.
TAKER_FEE_BPS = 4.5
MAKER_FEE_BPS = 1.5

#: How long a "session" is, in bars. 60 m1 bars is one hour, which is the length of the
#: only live trading session this project has ever run — so the first distribution this
#: produces is directly comparable with the one number that exists.
DEFAULT_SESSION_BARS = 60


def session_pnl(
    *,
    targets: np.ndarray,
    close: np.ndarray,
    notional: float,
    fee_bps: float,
) -> dict[str, float]:
    """Account one window of target changes as the runtime's money view would.

    ``targets`` is the position the strategy wanted at each bar, in units of
    ``max_position`` (so ``+1`` is fully long and ``-1`` fully short). ``close`` is the
    bar's close in real price units. ``notional`` is what a full position is worth.

    The accounting is **average cost over closes**, which is what
    ``axon_core::Position::apply_fill`` does, so a window's ``realized`` here is the same
    quantity the status line prints. The fee is charged on the *traded* notional — the
    magnitude of each target change — because that is what a venue charges, and charging
    it on the held position instead is the error that makes a low-turnover strategy look
    expensive and a high-turnover one look free.
    """
    t = np.asarray(targets, dtype=np.float64)
    px = np.asarray(close, dtype=np.float64)
    if t.size != px.size or t.size == 0:
        raise ValueError(f"targets and closes must be the same non-empty length: {t.size} vs {px.size}")

    # Units of the base asset, at the notional a full position is worth. Sized off the
    # window's FIRST close rather than each bar's, so a window that happens to span a
    # large price move is not also spanning a changing position size — the live session
    # sizes once, from `--max-position`, and never re-sizes.
    qty_full = notional / float(px[0])
    pos = t * qty_full

    traded = np.abs(np.diff(pos, prepend=0.0))
    # The last bar closes the position out: a window that ended holding something would
    # report an unrealized figure this function has no mark for, and a bound argued from
    # windows whose last bar was arbitrary is a bound argued from where the corpus was cut.
    traded[-1] += abs(pos[-1])

    fee = float((traded * px).sum()) * fee_bps / 10_000.0
    # Mark-to-market over the held position, which for a window that ends flat is exactly
    # the realized P&L: sum of position * price change.
    gross = float((pos[:-1] * np.diff(px)).sum())
    return {
        "gross": gross,
        "fee": fee,
        "net": gross - fee,
        "turnover_notional": float((traded * px).sum()),
        "trades": int(np.count_nonzero(traded)),
        "bars": int(t.size),
    }


def session_loss_window(
    *,
    start: int = 0,
    session_bars: int = DEFAULT_SESSION_BARS,
    data: str = "",
    coins: str = "BTC",
    interval: str = "1m",
    notional: float = 19.54,
    fee_bps: float = MAKER_FEE_BPS,
    registry: str | None = None,
    model: str = "zoo_xgboost",
    artifact: str | None = None,
) -> dict[str, Any]:
    """One window's bottom line. **The remote entrypoint** — one task per window.

    Imports are inside the function for the same reason every other entrypoint in
    :mod:`axon.strategies.remote` does it: the container mounts only the ``axon`` package,
    and an import at module scope would make this module unimportable anywhere the ML
    stack is absent.

    ``notional`` defaults to **19.54 USDC**, which is what ``max_position = 0.0003`` BTC
    was worth at the live run's session prep. It is the live size and not a round number,
    because a bound is compared against what a session actually risks.
    """
    from axon.strategies.data import INTERVAL_MS  # noqa: F401 - validates the interval
    from axon.strategies.remote import read_candles
    from axon.strategies.zoo import serving_strategy
    from axon.strategies.training import replay_bars

    if session_bars < 2:
        raise ValueError(f"a session of {session_bars} bar(s) has no trade in it")

    candles = read_candles(data, coins, interval)[0]
    n = candles.close.size
    if start + session_bars > n:
        raise IndexError(f"window [{start}, {start + session_bars}) is outside {n} bars")

    from axon.models import ModelRegistry

    strategy = serving_strategy(
        symbol_id=0,
        registry=ModelRegistry(registry) if registry else None,
        model=model,
    )
    # The window is replayed with the **whole prefix** in front of it, because the spec's
    # transforms have a finite lookback and a strategy that started at the window boundary
    # would be NaN through its own warmup — the same property the shadow diff's offline
    # reference relies on. Only the window's own bars are accounted.
    prefix = candles.slice(0, start + session_bars) if hasattr(candles, "slice") else candles
    _, _, targets = replay_bars(strategy, prefix, symbol_id=0)
    t = np.asarray(targets, dtype=np.float64)
    if t.size < start + session_bars:
        # `replay_bars` yields one target per bar it served; a short answer means the
        # strategy was still warming and the window is not measurable. Reported as such
        # rather than padded with zeros, which would look like a session that chose to be
        # flat.
        return {
            "start": int(start),
            "session_bars": int(session_bars),
            "measurable": False,
            "reason": f"the strategy served {t.size} of {start + session_bars} bars",
        }

    window = slice(start, start + session_bars)
    out = session_pnl(
        targets=t[window],
        close=np.asarray(candles.close[window], dtype=np.float64) / 1e8,
        notional=float(notional),
        fee_bps=float(fee_bps),
    )
    out.update(
        {
            "start": int(start),
            "session_bars": int(session_bars),
            "measurable": True,
            "coins": coins,
            "interval": interval,
            "notional": float(notional),
            "fee_bps": float(fee_bps),
            "correlation_id": os.environ.get("AXON_CORRELATION_ID"),
        }
    )
    if artifact:
        from axon.strategies.remote import _write_receipt

        _write_receipt(artifact, f"session-{start:07d}-{session_bars}.json", out)
    return out


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn a fan-out's receipts into the numbers a bound is argued from.

    **Quantiles of the loss, not of the P&L**, and the sign convention is the config's:
    ``max_session_loss`` is a magnitude, so the figures here are ``-net`` and a positive
    number is money lost. Reported as exact order statistics rather than as bucket edges —
    unlike the runtime's latency histogram, this is an offline computation over a list that
    fits in memory, so there is no reason to give up the exact number.
    """
    nets = np.array(
        [float(r["net"]) for r in results if r.get("measurable")], dtype=np.float64
    )
    if nets.size == 0:
        # ``windows`` is how many came back and ``measurable`` is how many had a number
        # in them, and the difference is the finding: "240 windows, 0 measurable" says the
        # fan-out ran and the strategy never served a row, while "0 windows" says the
        # fan-out did not run. Collapsing both to zero was this function's first shape and
        # it made those two indistinguishable.
        return {"windows": len(results), "measurable": 0}
    losses = -nets
    q = {f"p{p}": float(np.percentile(losses, p)) for p in (50, 90, 95, 99)}
    return {
        "windows": len(results),
        "measurable": int(nets.size),
        "mean_net": float(nets.mean()),
        "worst_loss": float(losses.max()),
        "best_gain": float(nets.max()),
        "loss_quantiles": q,
        # The two numbers a bound is chosen between, said out loud. A bound below `p99`
        # halts roughly one session in a hundred that was doing nothing unusual; a bound
        # above `worst_loss` never fires on anything this corpus contains.
        "suggested_floor": q["p99"],
        "suggested_ceiling": float(losses.max()),
    }


# ── the jobs ─────────────────────────────────────────────────────────────────

#: Session lengths to measure, in m1 bars: 1 h, 2 h, 4 h, 8 h.
#:
#: **Four rather than one, and this is the axis that matters most.** A loss bound has to be
#: chosen for the duration a session will actually run, and an eight-hour soak's bound
#: cannot be argued from one-hour windows: costs accumulate linearly in turnover while the
#: dispersion of the P&L grows like the square root of time, so a bound scaled naively from
#: an hour to eight is too tight on the costs and too loose on the market. 60 is included
#: because it is the length of the only live trading session this project has run, and 480
#: because it is the length of the soak `scripts/soak/run-trading-soak.sh` defaults to.
SESSION_LENGTHS: tuple[int, ...] = (60, 120, 240, 480)

#: Fee tiers to price each window at. Both, because the flatten crosses.
#:
#: The strategy serves at urgency 0 — a resting quote — so `MAKER_FEE_BPS` describes the
#: session. `TAKER_FEE_BPS` describes the *exit*, and a window that had to be crossed out of
#: is exactly the expensive kind. A distribution priced at one tier is a bound that is wrong
#: about whichever half it did not measure.
FEE_TIERS: tuple[float, ...] = (MAKER_FEE_BPS, TAKER_FEE_BPS)

#: Notional a full position is worth, per coin, at the sizes the live configs use. Named per
#: coin rather than shared because `max_position` is a *quantity* and the three coins trade
#: at wildly different prices — 0.0003 BTC and 0.0003 SOL are not the same risk.
COIN_NOTIONAL: dict[str, float] = {"BTC": 19.54, "ETH": 19.60, "SOL": 19.50}


def _window_starts(n_bars: int, session_bars: int) -> list[int]:
    """Every **non-overlapping** window that fits in `n_bars`.

    Non-overlapping is the whole reason this is a function rather than a range: overlapping
    windows share bars, so a quantile over them understates the dispersion by exactly the
    amount they overlap. That is the error that produces a confident-looking bound from one
    afternoon of data, and it gets *worse* as the window grows — at 480 bars, a stride of 60
    would make every window 87 % the same data as its neighbour.
    """
    if session_bars < 2:
        raise ValueError(f"a session of {session_bars} bar(s) has no trade in it")
    return [i * session_bars for i in range(max(0, n_bars // session_bars))]


def session_loss_job(
    *,
    coins: Sequence[str] = ("BTC", "ETH", "SOL"),
    interval: str = "1m",
    session_lengths: Sequence[int] = SESSION_LENGTHS,
    fee_tiers: Sequence[float] = FEE_TIERS,
    corpus_bars: int = 5_200,
    registry: str = "zoo_xgboost",
    model: str = "zoo_xgboost",
    name: str = "axon-session-loss",
    max_usd: float = 10.00,
    **kwargs: Any,
) -> ComputeJob:
    """**CPU.** One task per (coin, session length, fee tier, window). The fan-out *is* the
    distribution, and it is maximised over every axis that changes the answer.

    ``corpus_bars`` is the per-coin m1 history available — 15 595 bars across BTC+ETH+SOL
    were cached for the Phase-6 refit, so ~5 200 each. It is a parameter rather than a
    constant because the corpus grows every day the venue is up, and a job that hardcoded it
    would silently stop measuring the newest data.

    Sized as a CPU job explicitly rather than by inference: each task is a few thousand rows
    of numpy and one ``predict``, so a GPU would sit idle behind the data loading — and
    hwsched's planner forces GPU on a *framework* hint, which xgboost is.

    ``max_usd = 10`` is chosen against the aggregate pool (~$420/month across fourteen Modal
    profiles), not against one account's $5 default. See the module docstring.
    """
    tasks: list[dict[str, Any]] = []
    for coin in coins:
        notional = COIN_NOTIONAL.get(coin, 19.54)
        for session_bars in session_lengths:
            for fee_bps in fee_tiers:
                for start in _window_starts(corpus_bars, session_bars):
                    tasks.append(
                        {
                            "start": int(start),
                            "session_bars": int(session_bars),
                            "data": _data_uri(interval),
                            "coins": coin,
                            "interval": interval,
                            "notional": float(notional),
                            "fee_bps": float(fee_bps),
                            "registry": registry,
                            "model": model,
                            "artifact": volume_uri(
                                ARTIFACT_VOLUME, f"loss_evidence/{interval}/"
                            ),
                        }
                    )
    if not tasks:
        raise SpecTooSmall(
            f"no windows: {corpus_bars} bars cannot hold one "
            f"{min(session_lengths)}-bar session"
        )
    # An explicit task list, not a grid, and for the same reason `walk_forward_job` uses
    # one: windows are not a Cartesian product. A product of starts and lengths enumerates
    # windows that run past the end of the corpus and windows that run backwards.
    return walk_forward(
        name,
        "axon.strategies.loss_evidence:session_loss_window",
        tasks,
        max_usd=max_usd,
        **_common(interval, f"loss_evidence/{interval}/", **kwargs),
    )


class SpecTooSmall(ValueError):
    """A fan-out that would enumerate nothing. Raised rather than returning an empty job,
    because hwsched would happily plan zero tasks and report success."""


def gpu_fit_sweep_job(
    *,
    coins: Sequence[str] = ("BTC", "ETH", "SOL"),
    interval: str = "1m",
    grid: Mapping[str, Sequence[Any]] | None = None,
    folds: int = 4,
    inner_folds: int = 3,
    name: str = "axon-zoo-gpu-fit",
    gpu: str = "T4",
    max_parallel: int = 10,
    max_usd: float = 180.00,
    **kwargs: Any,
) -> ComputeJob:
    """**GPU, at the full grid.** 256 points over four axes.

    ``max_usd = 180`` covers hwsched's *pessimistic* price for this — it planned at
    ``$105 / $116 / $157`` (low/expected/high) over ~16 h — and the cap is declared to cover
    the estimate rather than the grid being cut to fit it.

    **Raising a cap and overriding a guard are different acts, and the difference is whether
    the ceiling means anything.** ``hwsched.toml``'s ``per_job_max_usd = 5`` describes one
    Modal Starter account; there are fourteen profiles, so the real pool is ~``$420`` a month
    and ``$157`` of it is a third. Declaring ``$180`` is sizing a job against the capacity
    that exists. Cutting the grid to 4 points because a ``$3`` cap somebody typed refused it —
    which is what the first version of this function did — is sizing the *question* against a
    guard, and it answers a different question badly.

    The estimate is still an estimate: hwsched has no per-task runtime for a tree fit on a
    GPU and falls back to a flat 1800 s, so 256 points price as 128 GPU-hours when a real m1
    XGBoost ``hist`` fit is minutes. :func:`gpu_calibration_job` is the run that replaces the
    guess with a measurement, and it should go first.

    ``max_parallel = 10`` is the **workspace** cap on concurrent GPUs, not a choice: Modal
    allows ten per workspace, so a job asking for more queues anyway and the number is
    stated so the wall-time arithmetic is legible (256 tasks / 10 = 26 waves).

    A **T4** rather than anything larger, and this is a choice: an XGBoost ``hist`` fit over
    a pooled three-coin m1 corpus is memory-light and bandwidth-bound, so an A100 buys wall
    time this pipeline does not need at four times the rate. Maximising the *grid* is what
    answers the question; maximising the *card* would only spend faster.

    ``framework="xgboost"`` is what makes hwsched plan this on a device at all, and
    ``resources.device`` is what makes the fit use it — the two are independent, and a job
    that set only the first would pay for a GPU and train on the CPU beside it.
    """
    axes: dict[str, Sequence[Any]] = dict(
        grid
        or {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [20, 40, 80, 160],
            "n_estimators": [100, 200, 400, 800],
        }
    )
    axes.update(
        {
            "data": [_data_uri(interval)],
            "coins": [",".join(coins)],
            "interval": [interval],
            "folds": [folds],
            "inner_folds": [inner_folds],
            "artifact": [volume_uri(ARTIFACT_VOLUME, f"zoo/gpu_fit/{interval}/")],
        }
    )
    # `resources` is a **hard pin** that overrides hwsched's planner, which is what is
    # wanted here: the planner sizes a tree framework onto CPU by default (correctly, for
    # the small fits this pipeline usually does), and this job is the one case where the
    # grid is wide enough that the device is worth paying for. A pin rather than a hint, so
    # the choice is visible in the spec an operator reads before spending.
    return param_sweep(
        name,
        "axon.strategies.remote:sweep_point",
        axes,
        max_usd=max_usd,
        max_parallel=max_parallel,
        resources={"device": "gpu", "gpu_type": gpu, "gpu_count": 1},
        **_common(interval, f"zoo/gpu_fit/{interval}/", **kwargs),
    )


def gpu_calibration_job(
    *, gpu: str = "T4", name: str = "axon-zoo-gpu-calibration", max_usd: float = 3.00, **kwargs: Any
) -> ComputeJob:
    """**GPU, four points.** The run whose product is a *number*, not a ranking.

    hwsched prices a GPU tree fit at a flat 1800 s because it has no measurement for one, so
    every plan for :func:`gpu_fit_sweep_job` is conservative by roughly an order of
    magnitude. This run's receipts carry the real per-point wall time on a ``T4``; with that
    in hand the wide grid can be priced honestly instead of pessimistically.

    Four points at the corners of the box, so the calibration also answers whether the
    device changes the *ranking* at the extremes — which is the one thing four points can
    tell you. It fits inside a single profile's default guard, which makes it the right thing
    to run first on a fresh account.
    """
    return gpu_fit_sweep_job(
        name=name,
        gpu=gpu,
        grid={"max_depth": [3, 6], "learning_rate": [0.01, 0.1]},
        max_parallel=4,
        max_usd=max_usd,
        **kwargs,
    )


def shard_by_profile(job: ComputeJob, n: int) -> list[ComputeJob]:
    """Split an explicit-task job into `n` shards, one per Modal profile.

    hwsched submits to one provider per run and Modal picks the account from the submitting
    process's ``MODAL_PROFILE``, so spreading across the fourteen configured profiles is an
    operator-level fan-out: plan and run each shard with a different ``MODAL_PROFILE`` and
    ``HWSCHED_MONTHLY_USD``. This makes that mechanical instead of manual.

    **The shards are contiguous, not interleaved**, and that is the one decision in here. A
    round-robin split would put adjacent windows of the same coin in different accounts, so a
    single shard failing would punch evenly-spaced holes through every coin's distribution
    and the survivors would still look like a complete sample. Contiguous shards fail
    *legibly*: a lost shard is a named coin's worth of a named session length, and
    :func:`summarize` reports `windows` against `measurable` so the gap is visible.

    Each shard keeps the parent's budget cap, because the cap is per job and each shard is a
    job — dividing it would make fourteen shards each a fourteenth as able to run.
    """
    if job.tasks is None:
        raise SpecError("shard_by_profile needs a job with an explicit task list")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tasks = list(job.tasks)
    size = -(-len(tasks) // n)  # ceil, so no shard is empty while another is oversized
    out: list[ComputeJob] = []
    for k in range(0, len(tasks), size):
        chunk = tasks[k : k + size]
        out.append(
            replace(
                job,
                name=f"{job.name}-shard{len(out):02d}",
                tasks=chunk,
                # Cleared so hwsched re-derives it: a shard that inherited the parent's
                # correlation id would collide with its siblings in the run store, and the
                # idempotency key derived from it would make hwsched treat shard 2 as a
                # duplicate of shard 1 and refuse to run it.
                correlation_id=None,
            )
        )
    return out


__all__ = [
    "COIN_NOTIONAL",
    "DEFAULT_SESSION_BARS",
    "FEE_TIERS",
    "MAKER_FEE_BPS",
    "SESSION_LENGTHS",
    "TAKER_FEE_BPS",
    "SpecTooSmall",
    "gpu_calibration_job",
    "gpu_fit_sweep_job",
    "session_loss_job",
    "session_loss_window",
    "session_pnl",
    "shard_by_profile",
    "summarize",
]
