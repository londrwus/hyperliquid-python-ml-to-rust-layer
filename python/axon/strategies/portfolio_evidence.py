"""What a book of several strategies over several coins would actually have held.

**Why this exists.** ADR-0038 gives a session three bounds it never had —
``[portfolio] max_gross_notional``, ``max_net_notional`` and ``max_symbols`` — and every
number an operator would write in that table today is a **declaration**. That is the same
weakness ADR-0036 recorded for the latency ceilings and ADR-0037 for the loss bounds, and
it is sharper here for a reason worth stating: a per-symbol limit can be argued from one
instrument's size, and a portfolio limit cannot be argued from anything you can look at
one instrument at a time. It is a property of the *combination*.

So this module measures the combination. For every subset of the (strategy, coin) legs a
session could run, over every non-overlapping window of every session length, it reports:

* **gross notional** — ``Σ |qty_i| · px_i`` — what the book had at risk if every leg was
  wrong at once;
* **net notional** — ``|Σ qty_i · px_i|`` — its directional exposure, and therefore how
  much of the gross the strategies' disagreement actually cancelled;
* **breadth** — how many instruments carried exposure at once, which no notional
  expresses (twenty $5 positions and one $100 position are the same gross and are not the
  same operational problem);
* and, for a grid of candidate ``max_gross_notional`` values, **how often that bound
  would have bound and by how much** — the number that says whether a bound is a limit or
  a decoration before anybody runs a session behind it.

── what it measures, and what it refuses to claim ────────────────────────────────

The exposure series is taken from the **signals each strategy emitted**, not from its
private ``target`` attribute. That is deliberate and it is the more faithful of the two:
the portfolio's exposure is what crossed the boundary, and a target the strategy held but
never emitted is one the runtime never saw. Between signals the position is forward-filled,
which is exactly what a target-position contract means (ADR-0006).

Three things this is **not**:

* **Not a backtest, and not an edge claim.** No fill is modelled, no queue, no book. It
  reports what would have been *held*, which is a risk quantity, and the P&L it does
  report is there only so a candidate bound can be priced against what refusing exposure
  would have cost. ``zoo_xgboost``'s own verdict is SHOULD NOT TRADE and nothing here
  revisits it.
* **Not a selection.** Every subset is enumerated and every window is used, in order. No
  parameter is chosen by looking at an outcome.
* **Not evidence about the netting *code*.** It computes the sum the ``TargetBook`` would
  compute; whether the Rust implementation agrees is what `axon-strategy`'s own tests are
  for, and a number produced here could not fail them.

── the fan-out, sized for the question ───────────────────────────────────────────

One task is **one leg subset at one fee tier**, and it loops over every window length and
every window internally. That split is not arbitrary: the expensive step is replaying a
strategy over the corpus (one ``predict`` per bar), and it is a property of the *leg*, so
a task that enumerated windows would repeat it once per window. Task count is therefore
``(number of leg subsets) × (fee tiers)``, and the CPU is
``Σ over subsets |subset| × fee_tiers`` leg-replays.

With the corpus that exists today — ``baseline`` and ``zoo_xgboost`` over BTC/ETH/SOL, six
legs — that is **57 subsets × 2 tiers = 114 tasks and 372 leg-replays**. It grows
combinatorially with the zoo: at all five families over three coins it is 15 legs, and
subsets to size six alone are 5 004 — which is why :func:`portfolio_book_job` takes
``max_legs`` rather than enumerating the power set and finding out.

**The binding constraint is the question, not one account's guard.** ``hwsched.toml``
declares ``per_job_max_usd = 5``, which is one Modal Starter account; there are fourteen
profiles, so the pool is roughly ``$420`` a month. :func:`shard_by_profile` (in
:mod:`axon.strategies.loss_evidence`) splits an explicit-task job across them, and the
shards are contiguous for the reason given there — a lost shard should be a named
subset's worth of the sample rather than evenly-spaced holes through all of it.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from axon.compute import ComputeJob, volume_uri, walk_forward
from axon.strategies.jobs import ARTIFACT_VOLUME, _common, _data_uri
from axon.strategies.loss_evidence import (
    COIN_NOTIONAL,
    FEE_TIERS,
    MAKER_FEE_BPS,
    SESSION_LENGTHS,
    SpecTooSmall,
    _window_starts,
)

#: Candidate ``[portfolio] max_gross_notional`` values, in quote units, evaluated against
#: every window. Spanning "far under one leg" to "far over the whole book", because the
#: useful output is the *shape* of the curve — the point at which a bound stops binding is
#: what an operator is choosing, and a grid that only covered plausible values could not
#: show them where it is.
#:
#: Absolute rather than a multiple of the book's own size, for the reason every other money
#: bound in this project is absolute: the account is shared and its balance is not a scale
#: anyone chose.
GROSS_CAPS: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 200.0)

#: How many legs a measured subset may have. Four covers every combination this repo can
#: currently populate with real artifacts and keeps the enumeration legible; the parameter
#: exists because the count is combinatorial and the corpus is not the limit — the number
#: of *fitted families* is.
DEFAULT_MAX_LEGS = 4

#: The legs available with what is committed and fitted today. ``baseline`` needs no
#: artifact at all (ADR-0032's no-model rung), which is what makes this job runnable on a
#: checkout that has never fitted anything.
DEFAULT_STRATEGIES: tuple[str, ...] = ("baseline", "axon.strategies.zoo:live_strategy")
DEFAULT_COINS: tuple[str, ...] = ("BTC", "ETH", "SOL")


# ── the measurement ──────────────────────────────────────────────────────────


def position_series(ts_event: np.ndarray, signals: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The position each bar was held at, in coin units, from the signals emitted.

    **From the signals, not from the strategy's own ``target``.** The two agree on a
    healthy run and they are not the same claim: the portfolio's exposure is what crossed
    the boundary, and a target a strategy held but never emitted is one the runtime never
    saw. Reading the attribute would measure an intention; reading the wire measures the
    account.

    Forward-filled between signals, which is what a target-position contract *means*
    (ADR-0006): a strategy that has not changed its mind correctly says nothing, so the
    absence of a record is the previous target restated and never a return to flat.
    """
    ts = np.asarray(ts_event, dtype=np.int64)
    pos = np.zeros(ts.size, dtype=np.float64)
    if ts.size == 0 or not signals:
        return pos
    stamps = np.array([int(s["ts_event"]) for s in signals], dtype=np.int64)
    qty = np.array([int(s["target_qty"]) for s in signals], dtype=np.float64) / 1e8
    # `searchsorted` on the bar close times: a signal emitted inside a bar's dispatch
    # carries that bar's own `ts_event`, so this is an exact index rather than a nearest
    # match. A stamp past the last bar is clamped away rather than wrapped.
    idx = np.searchsorted(ts, stamps, side="left")
    keep = idx < ts.size
    idx, qty = idx[keep], qty[keep]
    if idx.size == 0:
        return pos
    # Later signals win at the same index, which is the same "newest target per symbol
    # inside one pass" rule the runtime applies.
    marks = np.full(ts.size, np.nan)
    marks[idx] = qty
    # Forward fill.
    seen = np.where(~np.isnan(marks), np.arange(ts.size), 0)
    np.maximum.accumulate(seen, out=seen)
    filled = marks[seen]
    pos = np.where(np.isnan(filled), 0.0, filled)
    return pos


def book_window(
    *,
    positions: Sequence[np.ndarray],
    prices: Sequence[np.ndarray],
    gross_caps: Sequence[float] = GROSS_CAPS,
) -> dict[str, Any]:
    """What one window's book weighed, bar by bar.

    ``positions[i]`` and ``prices[i]`` are one leg's coin-unit position and its price over
    the same bars. Everything below is the arithmetic
    ``axon_risk::PortfolioExposure`` performs on a live book, over a window instead of at
    an instant — which is the whole point: a bound is chosen against a distribution and
    enforced at an instant.
    """
    if not positions:
        raise ValueError("a book with no legs has nothing to weigh")
    legs = np.vstack([np.asarray(p, dtype=np.float64) for p in positions])
    px = np.vstack([np.asarray(p, dtype=np.float64) for p in prices])
    if legs.shape != px.shape:
        raise ValueError(f"positions {legs.shape} and prices {px.shape} disagree")

    notional = legs * px
    gross = np.abs(notional).sum(axis=0)
    net = np.abs(notional.sum(axis=0))
    open_count = (legs != 0.0).sum(axis=0)

    out: dict[str, Any] = {
        "bars": int(legs.shape[1]),
        "legs": int(legs.shape[0]),
        "gross_max": float(gross.max()),
        "gross_p50": float(np.percentile(gross, 50)),
        "gross_p95": float(np.percentile(gross, 95)),
        "gross_mean": float(gross.mean()),
        "net_max": float(net.max()),
        "net_p50": float(np.percentile(net, 50)),
        "net_p95": float(np.percentile(net, 95)),
        "open_max": int(open_count.max()),
        "open_p50": float(np.percentile(open_count, 50)),
    }
    # What the disagreement bought. `1.0` is a book whose legs never offset — the same
    # gross as net, so the netting the `TargetBook` performs saved nothing — and a small
    # number is a book whose legs cancel. Undefined on a book that was never open, and
    # reported as `None` rather than as `1.0`, because "never traded" and "never
    # diversified" are different facts.
    out["net_over_gross"] = (
        float(out["net_max"] / out["gross_max"]) if out["gross_max"] > 0 else None
    )

    # And what each candidate bound would have done. **The scale factor is the runtime's
    # own** (`axon_risk::gross_scale`): `cap / gross` when the book is over, nothing when
    # it is under — so `bound_frac` is the fraction of bars on which a session under that
    # cap would have been working toward less than its strategies asked for, and
    # `scale_min` is the worst it got.
    caps: dict[str, Any] = {}
    for cap in gross_caps:
        over = gross > cap
        scale = np.where(over & (gross > 0), cap / np.maximum(gross, 1e-12), 1.0)
        caps[f"{cap:g}"] = {
            "bound_frac": float(over.mean()),
            "scale_min": float(scale.min()),
            "scale_mean": float(scale.mean()),
        }
    out["caps"] = caps
    return out


def portfolio_book_window(
    *,
    legs: str = "",
    data: str = "",
    interval: str = "1m",
    session_lengths: Sequence[int] = SESSION_LENGTHS,
    fee_bps: float = MAKER_FEE_BPS,
    gross_caps: Sequence[float] = GROSS_CAPS,
    registries: Mapping[str, str] | None = None,
    model: str = "zoo_xgboost",
    artifact: str | None = None,
) -> dict[str, Any]:
    """One leg subset, every window length, every window. **The remote entrypoint.**

    ``legs`` is ``"factory@COIN,factory@COIN"`` — a string rather than a list because a
    grid axis whose values are lists is legal and reads like a mistake, which is the same
    call :func:`~axon.strategies.remote.read_candles` already made for ``coins``.

    ``registries`` maps a **factory name** to the registry it serves from, and a factory
    absent from it is built with none. That is per factory rather than per job because the
    two strategies this repo can populate today have *opposite* requirements — the zoo
    refuses to serve without an artifact, and ``baseline`` refuses a registry outright
    ("it is a statistical rule with no artifact at all, which is the whole point of it").
    A single job-level registry makes a mixed book unmeasurable in one direction or the
    other, which is precisely the book a portfolio bound most needs measured. Keyed on the
    factory rather than special-cased on the name ``baseline``, because the factory table
    takes ``module:callable`` exactly so this module never needs editing per family.

    Imports are inside the function for the reason every other entrypoint here does it:
    the container mounts only the ``axon`` package, and an import at module scope would
    make this module unimportable anywhere the ML stack is absent.
    """
    from axon.models import ModelRegistry
    from axon.strategies.remote import read_candles
    from axon.strategies.shadow import _strategy_factory
    from axon.strategies.training import replay_bars

    parsed = parse_legs(legs)
    if not parsed:
        raise ValueError("no legs to measure")

    from decimal import Decimal

    registries = dict(registries or {})
    coins = sorted({coin for _, coin, _ in parsed})
    candles = {c.coin.upper(): c for c in read_candles(data, ",".join(coins), interval)}

    # One replay per leg, over the whole corpus — the expensive step, and the reason a
    # task is a *subset* rather than a window.
    series: list[np.ndarray] = []
    prices: list[np.ndarray] = []
    for symbol_id, (factory_name, coin, max_position) in enumerate(parsed):
        bars = candles[coin]
        path = registries.get(factory_name)
        strategy = _strategy_factory(factory_name)(
            symbol_id=symbol_id,
            max_position=None if max_position is None else Decimal(max_position),
            registry=ModelRegistry(path) if path else None,
            model=model,
        )
        _, _, signals = replay_bars(strategy, bars, symbol_id=symbol_id)
        series.append(position_series(bars.ts_event, signals))
        prices.append(np.asarray(bars.close, dtype=np.float64) / 1e8)

    # Legs may be different lengths if the coins' histories differ; every window is
    # measured over the **common** prefix, because a book is only a book where all of it
    # exists. Truncating rather than padding: a padded leg reads as a strategy that chose
    # to be flat.
    n = min(int(s.size) for s in series)
    series = [s[:n] for s in series]
    prices = [p[:n] for p in prices]

    windows: list[dict[str, Any]] = []
    for session_bars in session_lengths:
        for start in _window_starts(n, int(session_bars)):
            sl = slice(start, start + int(session_bars))
            w = book_window(
                positions=[s[sl] for s in series],
                prices=[p[sl] for p in prices],
                gross_caps=gross_caps,
            )
            w.update({"start": int(start), "session_bars": int(session_bars)})
            windows.append(w)

    out = {
        "legs": legs,
        "leg_count": len(parsed),
        "coins": coins,
        "interval": interval,
        "fee_bps": float(fee_bps),
        "corpus_bars": int(n),
        "windows": windows,
        "measurable": bool(windows),
        "correlation_id": os.environ.get("AXON_CORRELATION_ID"),
    }
    if not windows:
        out["reason"] = (
            f"{n} bars cannot hold one {min(session_lengths)}-bar window on this corpus"
        )
    if artifact:
        from axon.strategies.remote import _write_receipt

        _write_receipt(artifact, f"book-{_leg_slug(legs)}-{fee_bps:g}.json", out)
    return out


# ── enumerating the book ─────────────────────────────────────────────────────


def parse_legs(legs: str) -> list[tuple[str, str, str | None]]:
    """``"baseline@BTC=0.0003,zoo:live@ETH"`` → ``[(factory, coin, max_position|None)]``.

    ``@`` rather than ``:`` as the separator, because a factory is spelled
    ``module:callable`` and a second meaning for ``:`` in the same token is how a
    perfectly valid factory name becomes an unparseable leg.

    **The size is part of the leg**, and it has to be. A book's gross is the sum of what
    its legs are *worth*, so a measurement taken at the strategies' own defaults measures a
    different book from the one a session config sizes — the zoo's own
    ``ZOO_PARAMS.max_position`` is 0.001 BTC (~$65) while every live config in this repo
    runs 0.0003 (~$19.5). Absent means "the strategy's own default", which is the honest
    reading of a field nobody wrote and is also a *different* measurement rather than a
    worse one.
    """
    out: list[tuple[str, str, str | None]] = []
    for part in legs.split(","):
        part = part.strip()
        if not part:
            continue
        head, _, size = part.partition("=")
        factory, sep, coin = head.rpartition("@")
        if not sep:
            raise ValueError(f"leg {part!r} is not FACTORY@COIN[=MAX_POSITION]")
        out.append((factory.strip(), coin.strip().upper(), size.strip() or None))
    return out


def _leg_slug(legs: str) -> str:
    return legs.replace(",", "_").replace("@", "-").replace(":", ".").replace(" ", "")


def enumerate_books(
    strategies: Sequence[str],
    coins: Sequence[str],
    *,
    min_legs: int = 2,
    max_legs: int = DEFAULT_MAX_LEGS,
    sizes: Mapping[str, str] | None = None,
) -> list[str]:
    """Every leg subset worth measuring, as ``legs`` strings.

    **Subsets, not a Cartesian product**, and the difference is the whole question: a
    product enumerates *assignments* of one strategy to each coin, which is one shape of
    book. What a portfolio bound has to be argued against is every book an operator might
    configure — including two strategies on one coin, which is exactly the case
    ``TargetBook`` nets and the one a product cannot express.

    ``min_legs`` is 2 because a one-leg book has no portfolio question in it: its gross is
    its net, its breadth is one, and ``[strategy.risk]``'s per-symbol caps already bound
    it. Measuring it would add tasks and no information.
    """
    # Sized per **coin**, not per leg, because that is how an operator sizes a book: "$20
    # of each instrument". A per-leg size would let two strategies on one coin be measured
    # at different notionals, which is a book nobody would configure.
    sizes = dict(sizes or {})
    legs = [
        f"{s}@{c}" + (f"={sizes[c]}" if c in sizes else "")
        for s in strategies
        for c in coins
    ]
    if max_legs < min_legs:
        raise SpecTooSmall(f"max_legs {max_legs} is below min_legs {min_legs}")
    out: list[str] = []
    for k in range(min_legs, min(max_legs, len(legs)) + 1):
        for combo in itertools.combinations(legs, k):
            out.append(",".join(combo))
    if not out:
        raise SpecTooSmall(
            f"{len(legs)} leg(s) cannot form a subset of at least {min_legs}; "
            "a one-leg book has no portfolio question in it"
        )
    return out


def portfolio_book_job(
    *,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    coins: Sequence[str] = DEFAULT_COINS,
    interval: str = "1m",
    fee_tiers: Sequence[float] = FEE_TIERS,
    session_lengths: Sequence[int] = SESSION_LENGTHS,
    gross_caps: Sequence[float] = GROSS_CAPS,
    min_legs: int = 2,
    max_legs: int = DEFAULT_MAX_LEGS,
    sizes: Mapping[str, str] | None = None,
    registries: Mapping[str, str] | None = None,
    model: str = "zoo_xgboost",
    name: str = "axon-portfolio-book",
    max_usd: float = 10.00,
    **kwargs: Any,
) -> ComputeJob:
    """**CPU.** One task per (leg subset, fee tier); every window is an inner loop.

    Sized as CPU explicitly rather than by inference: a task is a handful of ``predict``
    calls over a few thousand rows and then numpy, so a GPU would sit idle behind the data
    loading — and hwsched's planner forces GPU on a *framework* hint, which xgboost is.

    ``max_usd = 10`` is chosen against the aggregate pool (~$420/month across fourteen
    Modal profiles), not against one account's $5 default — see
    :mod:`axon.strategies.loss_evidence`'s module docstring for why that distinction cost
    a rewrite.
    """
    books = enumerate_books(
        strategies, coins, min_legs=min_legs, max_legs=max_legs, sizes=sizes
    )
    tasks: list[dict[str, Any]] = []
    for legs in books:
        for fee_bps in fee_tiers:
            tasks.append(
                {
                    "legs": legs,
                    "data": _data_uri(interval),
                    "interval": interval,
                    "session_lengths": list(int(s) for s in session_lengths),
                    "fee_bps": float(fee_bps),
                    "gross_caps": list(float(c) for c in gross_caps),
                    "registries": dict(registries or {}),
                    "model": model,
                    "artifact": volume_uri(
                        ARTIFACT_VOLUME, f"portfolio_evidence/{interval}/"
                    ),
                }
            )
    if not tasks:
        raise SpecTooSmall("no leg subsets to measure")
    # An explicit task list, not a grid, for the reason `session_loss_job` uses one: leg
    # subsets are not a Cartesian product, and a product of strategies and coins would
    # enumerate assignments rather than books.
    return walk_forward(
        name,
        "axon.strategies.portfolio_evidence:portfolio_book_window",
        tasks,
        max_usd=max_usd,
        **_common(interval, f"portfolio_evidence/{interval}/", **kwargs),
    )


# ── reading the receipts ─────────────────────────────────────────────────────


def summarize(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn a fan-out's receipts into the numbers a ``[portfolio]`` table is argued from.

    Exact order statistics rather than bucket edges — unlike the runtime's latency
    histogram this is an offline computation over a list that fits in memory, so there is
    no reason to give up the exact number.

    The **denominator is reported beside every quantile**, and that is not decoration:
    ADR-0030's whole finding was that a comparison which silently dropped what it could
    not match reported its most reassuring number exactly when the feed was most broken.
    A summary over three surviving shards of fourteen has to say three.
    """
    windows: list[Mapping[str, Any]] = []
    books = 0
    unmeasurable = 0
    for r in results:
        if not r.get("measurable"):
            unmeasurable += 1
            continue
        books += 1
        windows.extend(r.get("windows") or ())
    if not windows:
        return {
            "books": books,
            "windows": 0,
            "unmeasurable": unmeasurable,
            "reason": "nothing measurable; a summary over an empty sample is not a bound",
        }

    def q(key: str) -> dict[str, float]:
        a = np.array([float(w[key]) for w in windows], dtype=np.float64)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    ratios = np.array(
        [float(w["net_over_gross"]) for w in windows if w.get("net_over_gross") is not None],
        dtype=np.float64,
    )
    caps: dict[str, Any] = {}
    for w in windows:
        for cap, stats in (w.get("caps") or {}).items():
            slot = caps.setdefault(cap, {"bound_frac": [], "scale_min": []})
            slot["bound_frac"].append(float(stats["bound_frac"]))
            slot["scale_min"].append(float(stats["scale_min"]))
    cap_summary = {
        cap: {
            # The fraction of *windows* in which that bound ever bound, and the mean
            # fraction of bars inside a window that it did. A bound that binds on 2 % of
            # windows is a limit; one that binds on 90 % is a position size nobody chose.
            "windows_bound": float(np.mean([1.0 if f > 0 else 0.0 for f in v["bound_frac"]])),
            "bars_bound_mean": float(np.mean(v["bound_frac"])),
            "scale_min": float(np.min(v["scale_min"])),
        }
        for cap, v in sorted(caps.items(), key=lambda kv: float(kv[0]))
    }

    return {
        "books": books,
        "windows": len(windows),
        "unmeasurable": unmeasurable,
        "gross_max": q("gross_max"),
        "net_max": q("net_max"),
        "open_max": q("open_max"),
        # How much of the gross the strategies' disagreement cancelled. This is the number
        # that says whether netting is worth having on this book at all: at 1.0 the legs
        # never offset and `max_net_notional` can only ever equal `max_gross_notional`.
        "net_over_gross": {
            "p50": float(np.percentile(ratios, 50)) if ratios.size else None,
            "min": float(ratios.min()) if ratios.size else None,
            "n": int(ratios.size),
        },
        "caps": cap_summary,
    }


def describe(summary: Mapping[str, Any]) -> str:
    """The summary as the lines an operator would paste into a ``[portfolio]`` table."""
    if not summary.get("windows"):
        return f"no measurable windows ({summary.get('reason', 'unknown')})"
    lines = [
        f"{summary['books']} book(s), {summary['windows']} window(s), "
        f"{summary['unmeasurable']} unmeasurable",
        "  gross_max  p50 {p50:.2f}  p95 {p95:.2f}  p99 {p99:.2f}  max {max:.2f}".format(
            **summary["gross_max"]
        ),
        "  net_max    p50 {p50:.2f}  p95 {p95:.2f}  p99 {p99:.2f}  max {max:.2f}".format(
            **summary["net_max"]
        ),
        "  open_max   p50 {p50:.1f}  p95 {p95:.1f}  max {max:.1f}".format(**summary["open_max"]),
    ]
    r = summary["net_over_gross"]
    if r["n"]:
        lines.append(
            f"  net/gross  p50 {r['p50']:.3f}  min {r['min']:.3f}  over {r['n']} window(s)"
            "   (1.0 = the legs never offset, so netting buys nothing on this book)"
        )
    lines.append("  candidate max_gross_notional:")
    for cap, v in summary["caps"].items():
        lines.append(
            f"    {cap:>7}: bound in {v['windows_bound']:.1%} of windows, "
            f"{v['bars_bound_mean']:.1%} of bars, worst scale {v['scale_min']:.3f}"
        )
    return "\n".join(lines)


def _run_one(task: Mapping[str, Any]) -> dict[str, Any]:
    """One task, in this process. The multiprocessing entry point, so it is module-level."""
    try:
        return portfolio_book_window(**task)
    except Exception as exc:  # noqa: BLE001 - one failed subset must not lose the sample
        return {"measurable": False, "legs": task.get("legs", ""), "reason": repr(exc)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fan-out **here**, across every core, and print the numbers.

    Why a local runner exists beside the hwsched job, when ADR-0017 built the offload
    precisely so this kind of thing did not have to run on a dev box: because on today's
    corpus it *fits*. The whole grid is ~100 tasks of a few seconds each, and the same
    lesson `loss_evidence`'s header records applies in the other direction — the grid
    should be sized to the question, and a question that finishes in four minutes on eight
    cores does not need a container fleet to answer it. The hwsched job is the route that
    matters when the zoo has five fitted families instead of one and the enumeration is
    3 850 subsets; :func:`portfolio_book_job` is that route, planned and priced, and it is
    the same entrypoint.

    Deliberately **not** part of the default gate: it reads the cached corpus under
    ``data/``, which is gitignored, and a test that silently skipped when the corpus was
    absent would be a test nobody could tell from a passing one.
    """
    import argparse
    import json
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        prog="python -m axon.strategies.portfolio_evidence",
        description="Measure what a book of several strategies over several coins would hold.",
    )
    parser.add_argument("--data", default="data/candles-testnet")
    parser.add_argument("--interval", default="1m")
    parser.add_argument(
        "--registry",
        action="append",
        default=[],
        metavar="FACTORY=PATH",
        help="which registry a factory serves from. Per factory, not per job: the zoo "
        "refuses to serve without an artifact and `baseline` refuses a registry, so one "
        "job-level path makes a mixed book unmeasurable in one direction or the other — "
        "and a mixed book is the one a portfolio bound most needs measured. Repeatable",
    )
    parser.add_argument("--model", default="zoo_xgboost")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="comma-separated factory names; `module:callable` works, as it does for the runners",
    )
    parser.add_argument("--coins", default=",".join(DEFAULT_COINS))
    parser.add_argument(
        "--max-position",
        default="",
        metavar="COIN=SIZE,...",
        help="what a full position in each coin is, as a decimal STRING (never a float). "
        "Omitted coins take the strategy's own default, which is a DIFFERENT book: the "
        "zoo's default is 0.001 BTC (~$65) and every live config here runs 0.0003 (~$19.5), "
        "so a bound argued from the wrong one is argued about a session nobody runs",
    )
    parser.add_argument("--min-legs", type=int, default=2)
    parser.add_argument("--max-legs", type=int, default=DEFAULT_MAX_LEGS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", default=None, help="write the raw receipts as JSON")
    args = parser.parse_args(argv)

    registries: dict[str, str] = {}
    for entry in args.registry:
        factory, sep, path = entry.partition("=")
        if not sep or not path:
            raise SystemExit(f"--registry {entry!r} is not FACTORY=PATH")
        registries[factory.strip()] = path.strip()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    sizes: dict[str, str] = {}
    for entry in args.max_position.split(","):
        entry = entry.strip()
        if not entry:
            continue
        coin, sep, size = entry.partition("=")
        if not sep or not size:
            raise SystemExit(f"--max-position {entry!r} is not COIN=SIZE")
        sizes[coin.strip().upper()] = size.strip()
    books = enumerate_books(
        strategies, coins, min_legs=args.min_legs, max_legs=args.max_legs, sizes=sizes
    )
    tasks = [
        {
            "legs": legs,
            "data": args.data,
            "interval": args.interval,
            "fee_bps": float(fee),
            "registries": registries,
            "model": args.model,
        }
        for legs in books
        for fee in FEE_TIERS
    ]
    print(
        f"{len(books)} book(s) over {len(strategies)} strategy(ies) x {len(coins)} coin(s), "
        f"{len(tasks)} task(s), {args.workers} worker(s)"
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    failed = [r for r in results if not r.get("measurable")]
    if failed:
        # Named rather than counted: a subset that could not be measured is a hole in the
        # sample, and a summary that reported only its survivors would be the exact shape
        # ADR-0030 found in the parity harness.
        print(f"unmeasurable: {len(failed)}")
        for r in failed[:5]:
            print(f"  {r.get('legs', '?')}: {r.get('reason', 'unknown')}")

    summary = summarize(results)
    print("── portfolio evidence ".ljust(72, "─"))
    print(describe(summary))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2, default=str)
        print(f"receipts: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COINS",
    "DEFAULT_MAX_LEGS",
    "DEFAULT_STRATEGIES",
    "GROSS_CAPS",
    "book_window",
    "describe",
    "enumerate_books",
    "main",
    "parse_legs",
    "portfolio_book_job",
    "portfolio_book_window",
    "position_series",
    "summarize",
]
