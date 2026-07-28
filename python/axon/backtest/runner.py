"""Driving the Rust replay path from Python, and the result object it produces.

The one design decision in this module is what :class:`Backtester` *is*: a thin
driver around the ``replay_log`` example binary in ``crates/axon-replay``. It shells
out. It does not read the JSONL event log itself, and there is deliberately **no
pure-Python fallback** for when the binary is missing — :class:`ReplayUnavailable`
is raised instead.

That refusal is the whole point, and it got stronger with the chain it now drives.
``docs/07-parity-and-testing.md`` says parity comes from running *the same code*, and
the binary runs the production fan-out (book, then mark cache, then order tracker) and
the production strategy adapter (signal reader, then planner). A Python replay would
be a second implementation not only of order-book replacement and the mark staleness
rule but of fill attribution, resting exposure, the urgency table and ``cloid``
derivation. Two implementations agree on the day they are written and drift silently
afterwards, and the drift lands exactly where it is least visible: in the harness that
is supposed to be *detecting* drift. Paying a subprocess spawn per backtest is cheap
next to that. The cost of the choice is real and worth stating: a backtest needs a
Rust toolchain or a prebuilt binary, so this is not a pip-installable-and-go module.

**A replayed order is not a filled order.** :attr:`BacktestResult.orders` is what the
strategy *asked for* at each instant, against the state the log actually produced. It
reached no venue, so it was never acknowledged, never rested, and never filled — the
positions in the result move only on the fills the *captured* session received. The
replay does not even write a planned order into the tracker, because that would mean
inventing an acknowledgement the venue never gave (ADR-0018 §7).

**A column that arrives as ``None`` is a reading nobody could take.** The order tracker
sits behind a lock in the Rust core, and a panic on any other path that holds it leaves
every reconciled column unknowable for the rest of the session — the core keeps running
and counts the execution events it had to drop (:attr:`BacktestResult.dropped_exec_events`).
Those columns cross as ``None``, never as zeros. Zeros would say *flat, nothing resting,
nothing unattributed*, which renders a session that lost track of its own orders
identical to a quiet one — and both sides of a golden would agree on it.

Money stays exact on this side of the boundary too. ``rust_decimal`` serializes a
``Decimal`` as a *string*, and this module parses those strings with
:class:`decimal.Decimal`, never :class:`float`. Passing a price through a float64
would round it to something that no longer compares equal to the reference — a
divergence manufactured by the comparison itself. A ``cloid`` crosses as a hex
*string* for the same family of reason: it is 128 bits and a JSON number is not.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

#: The result contract the ``replay_log`` binary emits. Checked, not assumed: a
#: reshaped trace compared against an old reference would produce a meaningless
#: result rather than a failed one.
RESULT_SCHEMA = "axon.backtest"

#: ``1`` was market-data state only. ``2`` is the whole chain — tracker and planner
#: columns, and a ``values``/``decisions`` split emitted by the binary itself. ``3``
#: reports a tracker that could not be read as ``None`` instead of as zeros, and carries
#: the ``dropped_exec_events`` counter that says what that cost.
RESULT_SCHEMA_VERSION = 3

_EXAMPLE = "replay_log"
_CRATE = "axon-replay"

#: Sibling convention: ``session.jsonl`` is replayed with ``session.signals.jsonl``
#: beside it, when one exists. The binary applies it, not this module — one place, so
#: a run from the shell and a run from Python cannot disagree about what was replayed.
SIGNALS_SUFFIX = ".signals.jsonl"


class BacktestError(RuntimeError):
    """The replay binary ran and failed, or produced something unparseable."""


class ReplayUnavailable(BacktestError):
    """No replay binary, and no toolchain to build one.

    Deliberately fatal rather than a fallback path — see the module docstring.
    """


def _repo_root() -> Path:
    # python/axon/backtest/runner.py → parents[3] is the repo root.
    return Path(__file__).resolve().parents[3]


def _target_dir(repo_root: Path) -> Path:
    """Cargo's target dir, honoring ``CARGO_TARGET_DIR`` — the same rule the pytest
    fixtures use, so a WSL2 checkout with a Linux-native target dir still finds it."""
    env = os.environ.get("CARGO_TARGET_DIR")
    return Path(env) if env else repo_root / "target"


def find_replay_binary(repo_root: Path | None = None, *, build: bool = True) -> Path:
    """Locate ``replay_log``, building it with cargo if it is missing.

    Raises :class:`ReplayUnavailable` rather than degrading to a Python replay.
    """
    root = repo_root or _repo_root()
    suffix = ".exe" if os.name == "nt" else ""
    path = _target_dir(root) / "debug" / "examples" / f"{_EXAMPLE}{suffix}"
    if path.exists():
        return path
    if not build:
        raise ReplayUnavailable(f"{path} not built; run cargo build -p {_CRATE} --examples")

    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / f"cargo{suffix}")
    if not Path(cargo).exists():
        raise ReplayUnavailable(
            f"no cargo on PATH and {path} is not built; a backtest runs the Rust core, "
            "so there is nothing to fall back to"
        )
    try:
        subprocess.run(
            [cargo, "build", "-q", "-p", _CRATE, "--examples"], cwd=root, check=True
        )
    except subprocess.CalledProcessError as e:  # pragma: no cover - toolchain failure
        raise ReplayUnavailable(f"failed to build {_CRATE} examples: {e}") from e
    if not path.exists():  # pragma: no cover - cargo layout change
        raise ReplayUnavailable(f"{path} still missing after a successful cargo build")
    return path


def _scalar(v: object) -> Decimal | int | None:
    """Decode one trace value.

    A JSON **string** is a fixed-point quantity — that is how ``rust_decimal``
    serializes, and keeping it a :class:`Decimal` is what stops a price from being
    rounded on the way into a comparison. A JSON number here is an event time or a
    count, which stays an ``int``.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return Decimal(v)
    if isinstance(v, bool):  # bool is an int subclass; nothing emits one, but be exact
        raise BacktestError(f"unexpected boolean in a trace value: {v!r}")
    if isinstance(v, int):
        return v
    raise BacktestError(f"unexpected trace value {v!r} of type {type(v).__name__}")


def _encode(v: Decimal | int | None) -> object:
    return str(v) if isinstance(v, Decimal) else v


def _opt_str(v: object) -> str | None:
    return None if v is None else str(v)


def _rows(obj: Mapping[str, object], name: str) -> list[Mapping[str, object]]:
    raw = obj.get(name, [])
    if not isinstance(raw, list):
        raise BacktestError(f"{name} must be a list, got {type(raw).__name__}")
    return raw


def _counters(raw: object) -> dict[str, int]:
    """Read the signal counters, refusing a partial block rather than defaulting it.

    A missing counter silently read as zero would report "the strategy sent nothing"
    for a run whose producer was simply not being counted, and that is the one reading
    an operator must never be handed.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise BacktestError(f"signals must be an object, got {type(raw).__name__}")
    missing = [c for c in SIGNAL_COUNTERS if c not in raw]
    if missing:
        raise BacktestError(f"signal counters missing: {missing}")
    return {c: int(raw[c]) for c in SIGNAL_COUNTERS}  # type: ignore[arg-type]


def _dropped_exec_events(obj: Mapping[str, object], where: str) -> int:
    """Read the poisoned-tracker counter, refusing a missing one rather than zeroing it.

    Zero is a strong claim — *every execution event reached the tracker* — and it is
    precisely the claim a defaulted absence would manufacture, in the one field that
    says whether the reconciled columns beside it describe the venue at all.
    """
    if "dropped_exec_events" not in obj:
        raise BacktestError(f"{where} carries no dropped_exec_events counter")
    return int(obj["dropped_exec_events"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PlannedOrder:
    """One order the replayed strategy asked for.

    **It reached no venue.** No acknowledgement, no queue position, no fill. Every
    field here is compared *exactly* by :func:`~axon.backtest.golden.compare_to_golden`
    and a difference is reported as a decision flip, never as a value inside a
    tolerance: a size or a limit that moved is a different instruction sent to a real
    venue, and ``cloid`` is the identity the venue de-duplicates a retry on.
    """

    signal_seq: int
    ts_event: int
    symbol_id: int
    cloid: str
    side: str
    qty: Decimal
    price: Decimal | None
    tif: str
    reduce_only: bool

    @property
    def identity(self) -> tuple[int, int, int]:
        return (self.signal_seq, self.ts_event, self.symbol_id)

    def to_dict(self) -> dict:
        return {
            "signal_seq": self.signal_seq,
            "ts_event": self.ts_event,
            "symbol_id": self.symbol_id,
            "cloid": self.cloid,
            "side": self.side,
            "qty": str(self.qty),
            "price": None if self.price is None else str(self.price),
            "tif": self.tif,
            "reduce_only": self.reduce_only,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> PlannedOrder:
        price = obj.get("price")
        return cls(
            signal_seq=int(obj["signal_seq"]),  # type: ignore[arg-type]
            ts_event=int(obj["ts_event"]),  # type: ignore[arg-type]
            symbol_id=int(obj["symbol_id"]),  # type: ignore[arg-type]
            # Never `int(..., 16)`: a cloid is an identity, not a quantity, and two
            # renderings of one number would compare unequal for no reason.
            cloid=str(obj["cloid"]),
            side=str(obj["side"]),
            qty=Decimal(str(obj["qty"])),
            price=None if price is None else Decimal(str(price)),
            tif=str(obj["tif"]),
            reduce_only=bool(obj["reduce_only"]),
        )


@dataclass(frozen=True, slots=True)
class PlannedCancel:
    """One cancel the replayed strategy asked for.

    ``target`` says *how* the cancel addresses the order — ``cloid:0x…`` or ``oid:…``
    — and which one it is matters: an adopted order's ``cloid`` may be one the tracker
    synthesized from a venue id that the venue has never seen, so a cancel sent under
    it fails silently and the stale quote stays resting.
    """

    signal_seq: int
    ts_event: int
    symbol_id: int
    target: str

    def to_dict(self) -> dict:
        return {
            "signal_seq": self.signal_seq,
            "ts_event": self.ts_event,
            "symbol_id": self.symbol_id,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> PlannedCancel:
        return cls(
            signal_seq=int(obj["signal_seq"]),  # type: ignore[arg-type]
            ts_event=int(obj["ts_event"]),  # type: ignore[arg-type]
            symbol_id=int(obj["symbol_id"]),  # type: ignore[arg-type]
            target=str(obj["target"]),
        )


#: What the strategy adapter made of the session, straight off the summary.
#:
#: In the golden because a refactor that started silently rejecting every record would
#: otherwise show up only as an absence of orders — which looks exactly like a quiet
#: strategy, and is the one failure a signal path must never present as normal.
SIGNAL_COUNTERS = (
    "records",
    "accepted",
    "rejected",
    "expired",
    "superseded",
    "planned",
    "no_quote",
)


@dataclass(frozen=True, slots=True)
class TraceRow:
    """The core's state as of one replayed event.

    ``values`` are continuous quantities, compared with a tolerance.
    ``decisions`` are discretized outputs, compared **exactly** — the split is a type
    distinction rather than a parameter, because "no discretized decision flips"
    (``docs/07``) is not a thing a caller should be able to soften by passing a
    tolerance that happens to cover it.
    """

    seq: int
    ts_event: int
    clock_ns: int
    symbol_id: int | None
    kind: str
    values: Mapping[str, Decimal | int | None]
    decisions: Mapping[str, object]

    @property
    def identity(self) -> tuple[int, int, int | None, str]:
        """What must match before two rows are even comparable."""
        return (self.seq, self.ts_event, self.symbol_id, self.kind)

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts_event": self.ts_event,
            "clock_ns": self.clock_ns,
            "symbol_id": self.symbol_id,
            "kind": self.kind,
            "values": {k: _encode(v) for k, v in self.values.items()},
            "decisions": dict(self.decisions),
        }

    @classmethod
    def from_trace_line(cls, obj: Mapping[str, object]) -> TraceRow:
        """Parse one line the Rust binary emitted.

        The same shape a stored reference uses, deliberately: one parser, so a fresh
        run and the reference it is compared against cannot mean different things by
        the same bytes. Under schema 1 the binary emitted a flat row and only the
        stored form was nested, which left two readers to keep in agreement.
        """
        return cls.from_dict(obj)

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> TraceRow:
        """Parse one row of a stored golden reference (nested form)."""
        values = obj.get("values", {})
        assert isinstance(values, Mapping)
        decisions = obj.get("decisions", {})
        assert isinstance(decisions, Mapping)
        return cls(
            seq=int(obj["seq"]),  # type: ignore[arg-type]
            ts_event=int(obj["ts_event"]),  # type: ignore[arg-type]
            clock_ns=int(obj["clock_ns"]),  # type: ignore[arg-type]
            symbol_id=None if obj["symbol_id"] is None else int(obj["symbol_id"]),  # type: ignore[arg-type]
            kind=str(obj["kind"]),
            values={k: _scalar(v) for k, v in values.items()},
            decisions=dict(decisions),
        )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """One replay pass, in a form two of which can be compared.

    This is a *deterministic re-drive of the whole chain over a captured log* — book,
    mark cache, order tracker, signal reader, planner — not a profit-and-loss
    simulation. The log's fills are the ones the captured session received; an order
    in :attr:`orders` got no fill, moved no price, and was never even written into the
    tracker. See ``crates/axon-replay/src/chain.rs`` and ADR-0018 §7.
    """

    source: str
    order: str
    events: int
    first_ts: int | None
    last_ts: int | None
    late_arrivals: int
    symbols: Mapping[int, Mapping[str, Decimal | int | None]]
    trace: tuple[TraceRow, ...]
    #: The signal log's own provenance string, or ``None`` when no strategy was
    #: attached. Never a path: a path differs between checkouts and would turn a golden
    #: comparison into a comparison of working directories.
    signal_source: str | None = None
    #: Execution events the Rust core could not apply, because a panic elsewhere left
    #: the order tracker's lock poisoned. Non-zero means the reconciled columns stopped
    #: following the venue partway through, so every position after the first drop is
    #: the last one anybody could read rather than the one the session held. Compared
    #: structurally — a run that lost events is not a slightly different run.
    dropped_exec_events: int = 0
    #: How many strategy passes ran. Part of the result because the pass schedule
    #: decides *when* the strategy got to look, and two runs that looked at different
    #: moments are not two runs of one experiment.
    intent_passes: int = 0
    signals: Mapping[str, int] = field(default_factory=dict)
    orders: tuple[PlannedOrder, ...] = ()
    cancels: tuple[PlannedCancel, ...] = ()
    schema_version: int = RESULT_SCHEMA_VERSION

    @classmethod
    def from_run(cls, summary: Mapping[str, object], trace_lines: Iterable[str]) -> BacktestResult:
        schema = summary.get("schema")
        version = summary.get("schema_version")
        if schema != RESULT_SCHEMA or version != RESULT_SCHEMA_VERSION:
            raise BacktestError(
                f"replay produced {schema!r} v{version}, this build reads "
                f"{RESULT_SCHEMA!r} v{RESULT_SCHEMA_VERSION}"
            )
        symbols_raw = summary.get("symbols", {})
        assert isinstance(symbols_raw, Mapping)
        return cls(
            source=str(summary["source"]),
            order=str(summary["order"]),
            events=int(summary["events"]),  # type: ignore[arg-type]
            first_ts=summary["first_ts"],  # type: ignore[arg-type]
            last_ts=summary["last_ts"],  # type: ignore[arg-type]
            late_arrivals=int(summary["late_arrivals"]),  # type: ignore[arg-type]
            symbols={
                int(k): {c: _scalar(v) for c, v in state.items()}
                for k, state in symbols_raw.items()
            },
            trace=tuple(
                TraceRow.from_trace_line(json.loads(line))
                for line in trace_lines
                if line.strip()
            ),
            signal_source=_opt_str(summary.get("signal_source")),
            dropped_exec_events=_dropped_exec_events(summary, "the replay summary"),
            intent_passes=int(summary.get("intent_passes", 0)),  # type: ignore[arg-type]
            signals=_counters(summary.get("signals")),
            orders=tuple(PlannedOrder.from_dict(o) for o in _rows(summary, "orders")),
            cancels=tuple(PlannedCancel.from_dict(c) for c in _rows(summary, "cancels")),
        )

    def to_dict(self) -> dict:
        return {
            "schema": RESULT_SCHEMA,
            "schema_version": self.schema_version,
            "source": self.source,
            "order": self.order,
            "events": self.events,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "late_arrivals": self.late_arrivals,
            "dropped_exec_events": self.dropped_exec_events,
            "signal_source": self.signal_source,
            "intent_passes": self.intent_passes,
            "signals": dict(self.signals),
            "orders": [o.to_dict() for o in self.orders],
            "cancels": [c.to_dict() for c in self.cancels],
            "symbols": {
                str(sym): {c: _encode(v) for c, v in state.items()}
                for sym, state in sorted(self.symbols.items())
            },
            "trace": [r.to_dict() for r in self.trace],
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> BacktestResult:
        if obj.get("schema") != RESULT_SCHEMA:
            raise BacktestError(f"not an {RESULT_SCHEMA} reference: {obj.get('schema')!r}")
        symbols_raw = obj.get("symbols", {})
        assert isinstance(symbols_raw, Mapping)
        trace_raw = obj.get("trace", [])
        assert isinstance(trace_raw, list)
        return cls(
            source=str(obj["source"]),
            order=str(obj["order"]),
            events=int(obj["events"]),  # type: ignore[arg-type]
            first_ts=obj["first_ts"],  # type: ignore[arg-type]
            last_ts=obj["last_ts"],  # type: ignore[arg-type]
            late_arrivals=int(obj["late_arrivals"]),  # type: ignore[arg-type]
            symbols={
                int(k): {c: _scalar(v) for c, v in state.items()}
                for k, state in symbols_raw.items()
            },
            trace=tuple(TraceRow.from_dict(r) for r in trace_raw),
            signal_source=_opt_str(obj.get("signal_source")),
            dropped_exec_events=_dropped_exec_events(obj, "the stored reference"),
            intent_passes=int(obj.get("intent_passes", 0)),  # type: ignore[arg-type]
            signals=_counters(obj.get("signals")),
            orders=tuple(PlannedOrder.from_dict(o) for o in _rows(obj, "orders")),
            cancels=tuple(PlannedCancel.from_dict(c) for c in _rows(obj, "cancels")),
            schema_version=int(obj.get("schema_version", RESULT_SCHEMA_VERSION)),  # type: ignore[arg-type]
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a stored reference.

        Sorted keys and a trailing newline so two references diff cleanly in review —
        a golden file nobody can read a diff of stops being reviewed.
        """
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        Path(path).write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BacktestResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def columns(self) -> tuple[str, ...]:
        """Continuous column names present in the trace, in first-seen order."""
        seen: dict[str, None] = {}
        for row in self.trace:
            for name in row.values:
                seen.setdefault(name, None)
        return tuple(seen)

    def with_decision(
        self, name: str, fn: Callable[[TraceRow], object]
    ) -> BacktestResult:
        """Attach a discretized decision column derived from each row.

        ``fn`` must be a pure function of the row: it is applied identically to the
        reference and to the candidate, so anything it reads from outside the row
        (a clock, a random seed, a global) reintroduces exactly the nondeterminism
        the harness exists to rule out.
        """
        rows = tuple(
            TraceRow(
                seq=r.seq,
                ts_event=r.ts_event,
                clock_ns=r.clock_ns,
                symbol_id=r.symbol_id,
                kind=r.kind,
                values=r.values,
                decisions={**r.decisions, name: fn(r)},
            )
            for r in self.trace
        )
        return dataclasses.replace(self, trace=rows)


class Backtester:
    """Replays a captured session through the production Rust chain.

    ``order`` selects how the log is traversed and is passed straight through:
    ``"event-time"`` (the default) sorts by each event's own timestamp,
    ``"as-captured"`` reproduces the bus order the live session saw. They differ only
    on out-of-order arrivals, and :attr:`BacktestResult.late_arrivals` says whether
    this log has any.

    ``signals`` names the recorded strategy output to replay alongside the events. The
    default is the sibling file — ``<log>`` with its extension replaced by
    ``.signals.jsonl`` — which the *binary* resolves, so a run from the shell and a run
    from here cannot disagree about what was replayed. Pass ``signals=False`` to replay
    market data alone; that is a legitimate run, not a degraded one, and
    :attr:`BacktestResult.signal_source` says which it was.
    """

    def __init__(
        self,
        *,
        binary: str | os.PathLike[str] | None = None,
        repo_root: str | os.PathLike[str] | None = None,
        order: str = "event-time",
        signals: str | os.PathLike[str] | bool = True,
        build: bool = True,
        timeout: float = 300.0,
    ) -> None:
        if order not in ("event-time", "as-captured"):
            raise ValueError(f"unknown replay order {order!r}")
        self.order = order
        self.signals = signals
        self.timeout = timeout
        self._binary = (
            Path(binary)
            if binary is not None
            else find_replay_binary(Path(repo_root) if repo_root else None, build=build)
        )

    @property
    def binary(self) -> Path:
        return self._binary

    def _signal_args(self) -> list[str]:
        if self.signals is True:
            return []  # the binary applies the sibling convention
        if self.signals is False:
            return ["--no-signals"]
        return ["--signals", str(self.signals)]

    def run(self, log_path: str | os.PathLike[str]) -> BacktestResult:
        """Replay ``log_path`` and collect the summary plus the per-event trace."""
        with tempfile.TemporaryDirectory(prefix="axon-backtest-") as tmp:
            trace = Path(tmp) / "trace.jsonl"
            proc = subprocess.run(
                [
                    str(self._binary),
                    str(log_path),
                    "--trace",
                    str(trace),
                    "--order",
                    self.order,
                    *self._signal_args(),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                raise BacktestError(
                    f"replay_log exited {proc.returncode}: {proc.stderr.strip()}"
                )
            try:
                summary = json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                raise BacktestError(f"replay_log produced no summary: {proc.stdout!r}") from e
            lines = trace.read_text(encoding="utf-8").splitlines()
        result = BacktestResult.from_run(summary, lines)
        # The binary counts the rows it wrote; if the file disagrees, the trace was
        # truncated between write and read and every downstream comparison would be
        # against a silently shortened run.
        expected = int(summary.get("trace_rows", len(result.trace)))  # type: ignore[arg-type]
        if len(result.trace) != expected:
            raise BacktestError(
                f"trace has {len(result.trace)} rows, replay reported {expected}"
            )
        return result
