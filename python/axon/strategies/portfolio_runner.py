"""Drive **several** strategies over **several** instruments onto one live session.

:mod:`axon.strategies.live_runner` answers "can a Python strategy keep a real venue
quote moving". This answers the question ADR-0038 opened underneath it: *can several
of them, over several instruments, share one account without fighting over it.*

::

    bar ring ──▶ one reader ──┬──▶ alpha ──▶ /dev/shm/alpha.ring ──┐
                              ├──▶ beta  ──▶ /dev/shm/beta.ring  ──┼──▶ Rust TargetBook
                              └──▶ diff                            ┘

Three structural facts shape the whole file, and each of them is a hazard this project
has already measured.

**One reader on the bar ring.** An SPSC ring has one consumer, and two readers do not
share it — they steal from it, each seeing about half the records with no way to tell
that from a quiet feed (measured 2026-07-26, ADR-0029 §5). So there is exactly one
:class:`~axon.strategies.shadow.RingBarSource` here and every strategy is *dispatched*
from it, which is the same fan-out ``--parity-diff`` already uses one level down.

**One ring per producer, never one ring per strategy-symbol.** The ring is SPSC on the
writing side too, and ``seq`` — the only proof that nothing was lost — is per writer.
Two writers on one ring interleave two sequences into a stream the Rust reader validates
as one, and every record of the loser is refused as ``stale_seq``: a producer emitting
normally, its own counters climbing, and nothing reaching the venue. A producer that
covers three coins writes all three onto **its** ring.

**One process, so the strategies cannot disagree about what a bar was.** Running N
copies of ``live_runner`` would mean N readers on the bar ring, which is the first
hazard again.

── what this file does NOT decide ───────────────────────────────────────────────

It does not net, allocate, or bound anything. Every one of those is the Rust side's, and
deliberately: netting two claims needs the position the venue actually holds, an
allocation needs the marks the risk gate uses, and a bound that Python could compute is
a bound Python could get wrong. This process emits each strategy's own target on each
strategy's own ring, unchanged and unaware of the others — and
``axon_strategy::TargetBook`` adds them up against the account
(`docs/adr/0038-many-strategies-one-account.md`).

That separation is what makes the failure modes legible. If two strategies want opposite
things, the Rust status line says ``net 1``; if the portfolio is binding, it says
``alloc 5000bp``; if one of these producers dies, it says ``STRATEGY SILENT <name>``.
None of those numbers could be produced here.

── running it ───────────────────────────────────────────────────────────────────

The producers, their rings and their instruments come from the **session's own TOML** —
the same file the Rust side reads — so the two cannot drift about which ring belongs to
whom. What the config does not know is which *strategy code* sits behind a ring (Rust
never needs to), so that comes from the command line::

    .venv/bin/python -m axon.strategies.portfolio_runner \\
        --config scripts/sessions/portfolio-testnet-m1.toml \\
        --symbol-ids BTC=3,ETH=4 \\
        --strategy alpha=zoo_xgboost:0.0003 \\
        --strategy beta=baseline:0.0003 \\
        --registry data/models-p6-live --model zoo_xgboost \\
        --duration 3600 --transcript data/p7/portfolio-run.jsonl

**Symbol ids are supplied, not guessed.** They are the venue's own asset indices and
they differ between mainnet and testnet (BTC is 3 on testnet). Resolving them here would
mean a second `meta` fetch beside the session's, and two answers to "what is BTC".
"""

from __future__ import annotations

import signal as signalmod
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

from axon.strategies.live_runner import (
    FLATTEN_URGENCIES,
    HEARTBEAT_S,
    POLL_S,
    LiveRunError,
    SignalOutbox,
    Transcript,
    build_strategy,
    stamp_cause,
)


class ConfigError(Exception):
    """A session description this runner refuses to start on."""


@dataclass
class ProducerPlan:
    """One producer, as the TOML declared it and the command line completed it."""

    name: str
    signal_ring: str
    #: Venue coin names, from the config. Empty means every configured instrument.
    coins: list[str]
    #: The strategy factory behind this ring, from ``--strategy``.
    factory: str
    max_position: str | None = None
    first_seq: int = 0


@dataclass
class ProducerStats:
    bars_dispatched: int = 0
    targets_changed: int = 0
    signals_pushed: int = 0
    backpressure_waits: int = 0
    #: Per symbol, the last target each strategy stated. Strings, because a `Decimal`
    #: in a JSONL transcript is a float somewhere downstream.
    targets: dict[int, str] = field(default_factory=dict)


class StrategyProducer:
    """One ring, one sequence space, one strategy instance per instrument.

    **Per instrument, not per producer.** A strategy object carries a rolling feature
    window, and one object fed two coins' bars would compute every window across both
    tapes — a feature matrix that is neither coin's and that no offline recompute can
    reproduce. The parity gate would not catch it either, because it recomputes against
    the bars the run was *shown*. So each instrument gets its own instance of the same
    factory, and the only thing they share is the ring.
    """

    def __init__(
        self,
        plan: ProducerPlan,
        *,
        symbol_ids: dict[str, int],
        registry: Any,
        model: str,
        model_version: int | None = None,
        capacity: int = 1024,
    ) -> None:
        from axon.strategies.baseline import NO_MODEL_VERSION

        self.name = plan.name
        self.plan = plan
        self.stats = ProducerStats()
        # **Strategies first, then the ring.** The ring's `StrategyContext` fixes
        # `model_version` for the whole run and will not take a new one afterwards — by
        # design, since a record that changed model mid-run could not be replayed against
        # the artifact it claims. So the version has to be known before the outbox
        # exists, and the only place it is knowable is the artifacts the strategies
        # loaded.
        self.strategies: dict[int, Any] = {}
        for coin in plan.coins:
            sid = symbol_ids.get(_coin_key(coin))
            if sid is None:
                raise ConfigError(
                    f"producer {plan.name!r} declares {coin!r} and --symbol-ids has no "
                    f"entry for it. The venue's asset index is not derivable here and "
                    f"differs between networks; supply it rather than let this process "
                    f"trade an instrument it cannot name"
                )
            self.strategies[sid] = build_strategy(
                plan.factory,
                symbol_id=sid,
                max_position=plan.max_position,
                registry=registry,
                model=model,
            )

        if model_version is None:
            versions = {
                getattr(s, "artifact_version", None) for s in self.strategies.values()
            }
            versions.discard(None)
            if len(versions) > 1:
                # One producer, one ring, one `model_version` on every record it writes.
                # Two artifacts behind one ring would put a version on the wire that is
                # right for one instrument and wrong for the other, and the record is the
                # only thing an audit has to find the model with.
                raise ConfigError(
                    f"producer {plan.name!r} loaded {len(versions)} different artifact "
                    f"versions {sorted(versions)} across its instruments; one ring "
                    f"carries one model_version, so split them into one producer each or "
                    f"state --model-version explicitly"
                )
            # `NO_MODEL_VERSION` is u32::MAX and conspicuous on purpose: a capture of a
            # no-model session must not be byte-identical to one serving registry
            # version 1.
            model_version = int(versions.pop()) if versions else NO_MODEL_VERSION

        self.model_version = int(model_version)
        self.out = SignalOutbox(
            plan.signal_ring,
            model_version=self.model_version,
            capacity=capacity,
            first_seq=plan.first_seq,
        )

    @property
    def symbols(self) -> list[int]:
        return sorted(self.strategies)

    def owns(self, symbol_id: int) -> bool:
        return symbol_id in self.strategies

    def on_bar(self, bar: Any, decided_ns: int) -> int:
        """Dispatch one bar to this producer's strategy for that instrument.

        Returns the number of records it emitted. A bar for an instrument this producer
        does not cover is **not** handed to any strategy: a strategy that filters
        internally would silently accept the extra dispatch while one that did not would
        trade another coin's tape, and which of those you have is a property of the
        strategy rather than of this runner.
        """
        strategy = self.strategies.get(bar.symbol_id)
        if strategy is None:
            return 0
        self.stats.bars_dispatched += 1
        before = strategy.target
        with self.out.ctx.event(decided_ns) as ctx:
            strategy.on_bar(bar, ctx)
        emitted = self.out.take_pending()
        # The venue's own clock for the observation, beside the producer's for the
        # decision. One function, shared with `live_runner`, so the `bar` latency stage
        # cannot end up measuring one of the two runners and not the other.
        stamp_cause(emitted, bar)
        after = strategy.target
        if after != before:
            self.stats.targets_changed += 1
        self.stats.targets[bar.symbol_id] = str(after)
        if emitted:
            self.out.queue(emitted)
        return len(emitted)

    def emit_target(self, symbol_id: int, target: Decimal, *, urgency: int | None = None) -> None:
        """Put a target on the ring outside a bar — the flatten path only.

        Its stamp is a wall clock for the same reason every other stamp here is, and it
        carries the strategy's own ttl so the planner treats it exactly like a decided
        one. See :meth:`~axon.strategies.live_runner.LiveRunner.emit_target` for why an
        urgent exit usually wants ``take`` rather than the strategy's default.
        """
        strategy = self.strategies.get(symbol_id)
        if strategy is None:
            return
        p = strategy.params
        with self.out.ctx.event(time.time_ns()) as ctx:
            ctx.emit_target(
                symbol_id,
                target,
                urgency=p.urgency if urgency is None else int(urgency),
                ttl_ms=p.ttl_ms,
            )
        self.out.queue(self.out.take_pending())

    def flush(self) -> int:
        pushed = self.out.flush()
        self.stats.signals_pushed = self.out.pushed
        self.stats.backpressure_waits = self.out.backpressure_waits
        return pushed

    def close(self) -> None:
        self.out.close()


class PortfolioRunner:
    """One bar-ring reader, dispatched to every producer that covers the instrument."""

    def __init__(
        self,
        producers: list[StrategyProducer],
        *,
        transcript: Transcript | None = None,
    ) -> None:
        if not producers:
            raise ConfigError("a portfolio run with no producers would read bars and emit nothing")
        self.producers = producers
        self._transcript = transcript or Transcript(None)
        self.bars_seen = 0
        self.bars_unclaimed = 0

    def on_bar(self, bar: Any) -> None:
        """One bar, one clock read, every producer that owns the instrument.

        **The clock is read once per bar and shared**, so two producers acting on one bar
        stamp the same instant. Reading it per producer would make the second one's
        record look later than the first's by however long the first strategy's inference
        took — a difference that would then show up as real skew in the `sig` latency
        stage and as an ordering between two decisions that were made together.
        """
        self.bars_seen += 1
        decided_ns = time.time_ns()
        claimed = False
        for p in self.producers:
            if not p.owns(bar.symbol_id):
                continue
            claimed = True
            p.on_bar(bar, decided_ns)
        if not claimed:
            # Counted rather than ignored: a session subscribing an instrument no
            # producer covers is paying for a feed nothing reads, and on a multi-coin
            # session that is indistinguishable from a strategy that has stopped.
            self.bars_unclaimed += 1
        self._transcript.write(
            "bar",
            symbol_id=bar.symbol_id,
            bar_ts=bar.ts_event,
            decided_ns=decided_ns,
            skew_ms=round((decided_ns - bar.ts_event) / 1e6, 1),
            close=bar.close,
            claimed=claimed,
            targets={p.name: dict(p.stats.targets) for p in self.producers},
        )

    def flush(self) -> int:
        return sum(p.flush() for p in self.producers)

    def close(self) -> None:
        for p in self.producers:
            p.close()

    def summary(self) -> str:
        lines = [
            f"bars {self.bars_seen} (unclaimed {self.bars_unclaimed}) "
            f"over {len(self.producers)} producer(s)"
        ]
        for p in self.producers:
            lines.append(
                f"  {p.name:>10}: symbols={p.symbols} bars={p.stats.bars_dispatched} "
                f"changes={p.stats.targets_changed} signals={p.stats.signals_pushed} "
                f"backpressure={p.stats.backpressure_waits} targets={p.stats.targets}"
            )
        return "\n".join(lines)


# ── the session description, read from the config both sides read ────────────


def _coin_key(name: str) -> str:
    """The venue coin a config instrument name refers to.

    The same rule ``RuntimeConfig::coins`` applies in Rust: everything before the first
    ``-``, uppercased, so ``"BTC-PERP"``, ``"btc"`` and ``"BTC"`` are one instrument. A
    second rule here would let a producer be correctly configured and silently scoped to
    nothing.
    """
    return name.split("-")[0].strip().upper()


def plans_from_config(path: str, strategies: dict[str, str]) -> tuple[list[ProducerPlan], dict]:
    """Read the session TOML and return one plan per declared producer.

    ``strategies`` maps a producer's declared name to ``factory`` or
    ``factory:max_position``.

    **The TOML is the shared truth and this reads it rather than restating it.** The ring
    paths and the instrument scopes have to be identical on both sides of the boundary —
    a producer writing to a ring nothing reads is a strategy whose every decision is
    discarded, and a scope Python and Rust disagree about is a record Rust refuses as
    out-of-scope while Python's counters climb. Both failures present as a healthy
    session that is not trading.
    """
    import tomllib

    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)

    strategy = cfg.get("strategy") or {}
    universe = [str(s) for s in strategy.get("symbols", [])]
    declared = strategy.get("producer") or []
    if not declared:
        # A single-producer session is legal and is what every config before ADR-0038
        # describes; it just has nothing for *this* runner to do that `live_runner`
        # does not do better, so say so rather than start something worse.
        raise ConfigError(
            f"{path} declares no [[strategy.producer]] tables, so it is a "
            "single-producer session. Use `python -m axon.strategies.live_runner` for "
            "that — this runner exists for the case one process must drive several "
            "rings without two readers on the bar ring"
        )

    plans: list[ProducerPlan] = []
    for i, p in enumerate(declared):
        name = str(p.get("name", "")).strip()
        if not name:
            raise ConfigError(f"{path}: [[strategy.producer]] #{i} has no name")
        if name not in strategies:
            raise ConfigError(
                f"producer {name!r} is declared in {path} and has no --strategy. "
                f"The config says which ring it writes to; only the command line says "
                f"what code is behind it. Known: {sorted(strategies) or 'none'}"
            )
        ring = str(p.get("signal_ring", "")).strip()
        if not ring:
            raise ConfigError(f"producer {name!r} has no signal_ring in {path}")
        coins = [str(s) for s in (p.get("symbols") or universe)]
        factory, _, max_pos = strategies[name].partition(":")
        plans.append(
            ProducerPlan(
                name=name,
                signal_ring=ring,
                coins=coins,
                factory=factory,
                max_position=max_pos or None,
            )
        )
    return plans, cfg


def parse_symbol_ids(raw: str) -> dict[str, int]:
    """``"BTC=3,ETH=4"`` → ``{"BTC": 3, "ETH": 4}``."""
    out: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        if not sep:
            raise ConfigError(f"--symbol-ids entry {part!r} is not NAME=ID")
        try:
            out[_coin_key(name)] = int(value)
        except ValueError as exc:
            raise ConfigError(f"--symbol-ids entry {part!r} has a non-integer id") from exc
    if not out:
        raise ConfigError("--symbol-ids is empty")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m axon.strategies.portfolio_runner",
        description=(
            "Drive several strategies over several instruments onto one live session's "
            "producer rings. Places REAL orders."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="the session TOML the Rust side reads. The producer names, their rings and "
        "their instruments come from here so the two cannot drift",
    )
    parser.add_argument(
        "--md-ring",
        required=True,
        help="the session's market-data SLICE ring; the bar ring beside it is derived "
        "(ADR-0028 §5), never named twice",
    )
    parser.add_argument(
        "--symbol-ids",
        required=True,
        help="NAME=ID pairs, e.g. BTC=3,ETH=4. The venue's own asset indices, which "
        "differ between networks — supplied rather than fetched, because a second `meta` "
        "read beside the session's is a second answer to 'what is BTC'",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="PRODUCER=FACTORY[:MAX_POSITION]",
        help="which strategy code sits behind a declared producer, and optionally its "
        "position size as a decimal STRING. Repeat once per producer",
    )
    parser.add_argument("--registry", default=None)
    parser.add_argument("--model", default="zoo_xgboost")
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="what goes in every record's model_version field. Defaults to "
        "baseline.NO_MODEL_VERSION when no artifact is loaded",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="stop after this many seconds. WALL CLOCK, and named as the exception it "
        "is: a run's budget is an operator's afternoon, which no event time measures",
    )
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument(
        "--flatten-on-exit",
        action="store_true",
        help="emit a target of zero for every instrument every producer covers, then "
        "wait. A REQUEST to the planner, not a guarantee — and NOT the cleanup pass: "
        "use `axon --flatten`, which reads the venue's own position",
    )
    parser.add_argument("--flatten-wait", type=float, default=45.0)
    parser.add_argument(
        "--flatten-urgency",
        choices=sorted(FLATTEN_URGENCIES),
        default="take",
        help="urgency for the flatten targets. 'take' by default here rather than the "
        "strategy's own, because a post-only flatten can be swept at "
        "intent.max_order_age_ms and a target position is idempotent, so nothing "
        "re-emits it",
    )
    parser.add_argument("--transcript", default=None)
    args = parser.parse_args(argv)

    from axon.marketdata import bar_ring_path
    from axon.strategies.shadow import RingBarSource

    strategies: dict[str, str] = {}
    for entry in args.strategy:
        name, sep, spec = entry.partition("=")
        if not sep or not spec:
            raise SystemExit(f"--strategy {entry!r} is not PRODUCER=FACTORY[:MAX_POSITION]")
        strategies[name.strip()] = spec.strip()

    try:
        plans, cfg = plans_from_config(args.config, strategies)
        symbol_ids = parse_symbol_ids(args.symbol_ids)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    registry = None
    if args.registry:
        from axon.models import ModelRegistry

        registry = ModelRegistry(args.registry)

    capacity = int((cfg.get("ipc") or {}).get("capacity", 1024))
    portfolio = cfg.get("portfolio") or {}

    producers: list[StrategyProducer] = []
    try:
        for plan in plans:
            producers.append(
                StrategyProducer(
                    plan,
                    symbol_ids=symbol_ids,
                    registry=registry,
                    model=args.model,
                    model_version=args.model_version,
                    capacity=capacity,
                )
            )
    except ConfigError as exc:
        # Every ring opened so far is closed before the exit: a `RingProducer` left open
        # on a failed start is a file a retry would then create *under* a mapping.
        for p in producers:
            p.close()
        raise SystemExit(str(exc)) from exc

    print(f"config     : {args.config}")
    print(f"bar ring   : {bar_ring_path(args.md_ring)}  (ONE reader, dispatched)")
    print(f"producers  : {len(producers)}")
    for p in producers:
        print(
            f"  {p.name:>10} : {p.plan.signal_ring} -> symbols {p.symbols} "
            f"({p.plan.factory}, max_position="
            f"{next(iter(p.strategies.values())).params.max_position}, "
            f"model_version={p.model_version})"
        )
    print(
        "portfolio  : "
        + (
            "gross<={} net<={} symbols<={} overlap={}".format(
                portfolio.get("max_gross_notional", 0),
                portfolio.get("max_net_notional", 0),
                portfolio.get("max_symbols", 0),
                portfolio.get("overlap", "exclusive"),
            )
            if portfolio
            else "no bound declared in the config - the Rust side gates nothing across symbols"
        )
    )
    print("stamp      : the PRODUCER's wall clock, read ONCE per bar and shared")

    transcript = Transcript(args.transcript)
    transcript.write(
        "start",
        config=args.config,
        md_ring=args.md_ring,
        producers=[
            {
                "name": p.name,
                "ring": p.plan.signal_ring,
                "symbols": p.symbols,
                "factory": p.plan.factory,
            }
            for p in producers
        ],
    )

    runner = PortfolioRunner(producers, transcript=transcript)
    stopping = {"now": False}

    def _stop(signum, _frame):
        # Flagged, not acted on: unwinding the rings and the transcript from inside a
        # signal handler is how a run's last records get lost.
        stopping["now"] = True

    signalmod.signal(signalmod.SIGTERM, _stop)
    signalmod.signal(signalmod.SIGINT, _stop)

    started = time.monotonic()
    last_beat = started
    rc = 0
    try:
        with RingBarSource.attach(args.md_ring, symbol_id=None) as src:
            print(f"attached   : {src.describe()}", flush=True)
            while True:
                if stopping["now"]:
                    print("stopping: signal received", flush=True)
                    break
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    print(f"stopping: --duration {args.duration}s reached", flush=True)
                    break
                if args.max_bars is not None and runner.bars_seen >= args.max_bars:
                    print(f"stopping: --max-bars {args.max_bars} reached", flush=True)
                    break
                for bar in src.poll():
                    runner.on_bar(bar)
                    print(
                        f"  bar {runner.bars_seen:>3} sym={bar.symbol_id} "
                        f"ts={bar.ts_event} close={bar.close / 1e8:.2f} "
                        + " ".join(
                            f"{p.name}={p.stats.targets.get(bar.symbol_id, '-')}"
                            for p in producers
                            if p.owns(bar.symbol_id)
                        ),
                        flush=True,
                    )
                runner.flush()
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_S:
                    last_beat = now
                    print(
                        f"  … {now - started:.0f}s  bars={runner.bars_seen} "
                        + " ".join(
                            f"{p.name}(sig={p.stats.signals_pushed},q={p.out.depth()})"
                            for p in producers
                        )
                        + f" feed(bars={src.health.bars} drops={src.health.ring_dropped} "
                        f"gaps={src.health.feed_gaps})",
                        flush=True,
                    )
                    transcript.write(
                        "heartbeat",
                        elapsed_s=round(now - started, 1),
                        bars=runner.bars_seen,
                        signals={p.name: p.stats.signals_pushed for p in producers},
                        feed_bars=src.health.bars,
                        feed_drops=src.health.ring_dropped,
                        feed_gaps=src.health.feed_gaps,
                    )
                time.sleep(POLL_S)

            if args.flatten_on_exit:
                # Emitted **unconditionally**, for every instrument every producer
                # covers, even where the strategy already believes it is flat: the
                # planner acts on the difference between a target and the *tracked*
                # position, and a partial fill is exactly the case where those disagree.
                urg = FLATTEN_URGENCIES[args.flatten_urgency]
                for p in producers:
                    for sid in p.symbols:
                        print(f"flatten: {p.name} target 0 on {sid}", flush=True)
                        p.emit_target(sid, Decimal(0), urgency=urg)
                runner.flush()
                deadline = time.monotonic() + args.flatten_wait
                while time.monotonic() < deadline:
                    # Keep draining while waiting, so the ring does not silently back up
                    # behind a paused reader and the transcript has no hole exactly where
                    # the unwind happened.
                    for bar in src.poll():
                        runner.on_bar(bar)
                    runner.flush()
                    time.sleep(POLL_S)
    except (LiveRunError, Exception) as exc:  # noqa: BLE001 - the transcript records the death
        transcript.write("error", error=repr(exc))
        print(f"FAILED: {exc!r}", flush=True)
        rc = 1
    finally:
        transcript.write(
            "stats",
            bars_seen=runner.bars_seen,
            bars_unclaimed=runner.bars_unclaimed,
            producers={
                p.name: {
                    "bars": p.stats.bars_dispatched,
                    "targets_changed": p.stats.targets_changed,
                    "signals_pushed": p.stats.signals_pushed,
                    "backpressure_waits": p.stats.backpressure_waits,
                    "targets": p.stats.targets,
                }
                for p in producers
            },
        )
        print("── run summary ".ljust(72, "─"))
        print(runner.summary())
        runner.close()
        transcript.close()
    return rc


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "ConfigError",
    "PortfolioRunner",
    "ProducerPlan",
    "ProducerStats",
    "StrategyProducer",
    "main",
    "parse_symbol_ids",
    "plans_from_config",
]
