"""The golden comparison ``docs/07-parity-and-testing.md`` describes: outputs match a
stored reference **within tolerance**, and **no discretized decision flips**.

Those are two different questions and this module keeps them apart, because merging
them is how a parity gate quietly stops protecting anything. A continuous quantity may
legitimately move by a rounding unit after a refactor; a *decision* may not move at
all, since a flipped decision is a different order sent to a real venue and no
tolerance makes that acceptable. So :attr:`GoldenComparison.divergences` (tolerance
breaches) and :attr:`GoldenComparison.flips` (decision changes) are separate, and a
tolerance can never soften a flip — see :class:`~axon.backtest.runner.TraceRow`, where
the split is a field, not an argument.

Four further rules, each preventing a way a comparison can be true but meaningless:

* **Shape before values.** Row counts, row identities (`seq`/`ts_event`/`symbol_id`),
  and the column set are compared exactly and reported as *structural* problems. If
  two runs produced different rows there is nothing to compare within tolerance; the
  numbers would just be diffing unrelated events.
* **Absence is not a small number.** ``None`` against a value is always a divergence,
  whatever the tolerance. A missing mark and a mark of zero are the difference between
  a risk gate failing closed and a risk gate sizing against nothing. The Rust side
  reports every column a poisoned order tracker made unreadable as ``None`` for the
  same reason, so a session that lost track of its own orders cannot compare equal to
  a flat one — and ``dropped_exec_events`` says how much it lost.
* **A tolerance is a statement about money math.** It applies only where both sides
  are :class:`~decimal.Decimal`. Event times and counts compare exactly — a timestamp
  is not a quantity you are allowed to be nearly right about.
* **An order is a decision, not a value.** Every field of a planned order and of a
  planned cancel — side, size, limit, time-in-force, reduce-only, ``cloid``, the way a
  cancel addresses its order — is compared exactly and any difference is reported as a
  *flip*. A tolerance may never reach them. A size that moved by a rounding unit is
  still a different instruction sent to a real venue, and a ``cloid`` that moved is a
  second position rather than a de-duplicated retry (ADR-0014 §5).

**What a comparison here does not tell you.** Two runs matching means the code still
produces what it produced, over reconciliation and strategy output as well as market
data. It does not mean the orders would have filled: nothing in a replay reaches a
venue, so no order in the result was ever acknowledged (ADR-0018 §7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal

from axon.backtest.runner import SIGNAL_COUNTERS, BacktestResult

#: How many examples of each problem a comparison keeps. The counts are exact; only
#: the listed examples are truncated, so a report stays readable when a refactor
#: moves every row.
MAX_EXAMPLES = 10


class GoldenMismatch(AssertionError):
    """A candidate run did not match its stored reference."""


@dataclass(frozen=True, slots=True)
class Divergence:
    """One column, in one place, that did not match."""

    where: str
    column: str
    reference: object
    candidate: object
    delta: Decimal | None = None

    def __str__(self) -> str:
        gap = "" if self.delta is None else f" (Δ {self.delta})"
        return f"{self.where}.{self.column}: {self.reference!r} → {self.candidate!r}{gap}"


@dataclass(frozen=True, slots=True)
class GoldenComparison:
    """The verdict, and enough detail to act on it."""

    rows: int
    structural: tuple[str, ...] = ()
    divergences: tuple[Divergence, ...] = ()
    flips: tuple[Divergence, ...] = ()
    divergence_count: int = 0
    flip_count: int = 0
    max_abs_diff: Mapping[str, Decimal] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.structural and self.divergence_count == 0 and self.flip_count == 0

    def __bool__(self) -> bool:
        return self.ok

    def report(self) -> str:
        if self.ok:
            worst = (
                ", ".join(f"{c}≤{d}" for c, d in sorted(self.max_abs_diff.items()) if d)
                or "exact"
            )
            return f"golden match over {self.rows} rows ({worst})"
        lines = [f"golden mismatch over {self.rows} rows:"]
        for msg in self.structural[:MAX_EXAMPLES]:
            lines.append(f"  structural: {msg}")
        if self.flip_count:
            lines.append(f"  {self.flip_count} decision flip(s):")
            lines += [f"    {d}" for d in self.flips]
        if self.divergence_count:
            lines.append(f"  {self.divergence_count} value(s) outside tolerance:")
            lines += [f"    {d}" for d in self.divergences]
        return "\n".join(lines)

    def assert_ok(self) -> None:
        if not self.ok:
            raise GoldenMismatch(self.report())


def _tolerance_for(column: str, tolerance: Decimal | Mapping[str, Decimal]) -> Decimal:
    if isinstance(tolerance, Mapping):
        return Decimal(tolerance.get(column, 0))
    return Decimal(tolerance)


def _compare_values(
    where: str,
    ref: Mapping[str, Decimal | int | None],
    cand: Mapping[str, Decimal | int | None],
    tolerance: Decimal | Mapping[str, Decimal],
    structural: list[str],
    divergences: list[Divergence],
    worst: dict[str, Decimal],
) -> None:
    if ref.keys() != cand.keys():
        structural.append(
            f"{where}: columns {sorted(ref.keys())} vs {sorted(cand.keys())}"
        )
        return
    for column, a in ref.items():
        b = cand[column]
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            delta = abs(a - b)
            if delta > worst.get(column, Decimal(0)):
                worst[column] = delta
            if delta > _tolerance_for(column, tolerance):
                divergences.append(Divergence(where, column, a, b, delta))
        elif a != b:
            # Covers None-vs-value, int-vs-int and any type change. None of these is
            # a rounding question, so the tolerance does not apply.
            divergences.append(Divergence(where, column, a, b))


#: Order fields compared exactly. Not a subset of anything — every field of a planned
#: order changes what the venue would be told to do.
_ORDER_FIELDS = (
    "signal_seq",
    "ts_event",
    "symbol_id",
    "cloid",
    "side",
    "qty",
    "price",
    "tif",
    "reduce_only",
)

_CANCEL_FIELDS = ("signal_seq", "ts_event", "symbol_id", "target")


def _compare_orders(
    reference: BacktestResult,
    candidate: BacktestResult,
    structural: list[str],
    flips: list[Divergence],
) -> None:
    """Diff the planner's output, field by field, as decisions.

    A count mismatch is *structural* for the same reason a row-count mismatch is: two
    plans of different lengths are not two versions of one plan, and pairing them by
    index would report every later order as changed and bury the one that appeared.
    """
    for label, ref, cand, fields in (
        ("orders", reference.orders, candidate.orders, _ORDER_FIELDS),
        ("cancels", reference.cancels, candidate.cancels, _CANCEL_FIELDS),
    ):
        if len(ref) != len(cand):
            structural.append(f"{label}: {len(ref)} vs {len(cand)}")
            continue
        for i, (a, b) in enumerate(zip(ref, cand)):
            for name in fields:
                x, y = getattr(a, name), getattr(b, name)
                if x != y:
                    flips.append(Divergence(f"{label}[{i}]", name, x, y))


def compare_to_golden(
    reference: BacktestResult,
    candidate: BacktestResult,
    *,
    tolerance: Decimal | Mapping[str, Decimal] = Decimal(0),
    max_examples: int = MAX_EXAMPLES,
) -> GoldenComparison:
    """Compare a fresh run against a stored reference.

    ``tolerance`` is either one :class:`~decimal.Decimal` for every column or a
    per-column mapping (columns absent from the mapping must match exactly). The
    default is exact: replay is deterministic, so anything else should be a
    deliberate allowance for a specific known source of rounding, not a blanket
    loosening applied because a test went red.
    """
    structural: list[str] = []
    divergences: list[Divergence] = []
    flips: list[Divergence] = []
    worst: dict[str, Decimal] = {}

    # ``dropped_exec_events`` sits here rather than among the values because it is not a
    # quantity that drifted: one event the core could not apply is the point past which
    # the reconciled columns stop describing the venue at all.
    for name in ("schema_version", "source", "signal_source", "order", "events",
                 "first_ts", "last_ts", "late_arrivals", "dropped_exec_events",
                 "intent_passes"):
        a, b = getattr(reference, name), getattr(candidate, name)
        if a != b:
            structural.append(f"{name}: {a!r} vs {b!r}")

    # Counters are structural, not numeric: "the reader accepted 3 records instead of
    # 4" is not a quantity that drifted, it is a record that stopped being acted on.
    for name in SIGNAL_COUNTERS:
        a, b = reference.signals.get(name), candidate.signals.get(name)
        if a != b:
            structural.append(f"signals.{name}: {a!r} vs {b!r}")

    _compare_orders(reference, candidate, structural, flips)

    if reference.symbols.keys() != candidate.symbols.keys():
        structural.append(
            f"symbols {sorted(reference.symbols)} vs {sorted(candidate.symbols)}"
        )
    else:
        for sym in sorted(reference.symbols):
            _compare_values(
                f"symbols[{sym}]",
                reference.symbols[sym],
                candidate.symbols[sym],
                tolerance,
                structural,
                divergences,
                worst,
            )

    if len(reference.trace) != len(candidate.trace):
        structural.append(
            f"trace rows: {len(reference.trace)} vs {len(candidate.trace)}"
        )
    else:
        for i, (a_row, b_row) in enumerate(zip(reference.trace, candidate.trace)):
            if a_row.identity != b_row.identity:
                structural.append(
                    f"trace[{i}] identity: {a_row.identity} vs {b_row.identity}"
                )
                # Past this point the two traces describe different events; comparing
                # their columns would report noise and hide the real cause.
                break
            where = f"trace[{i}]"
            _compare_values(
                where, a_row.values, b_row.values, tolerance, structural, divergences, worst
            )
            if a_row.decisions.keys() != b_row.decisions.keys():
                structural.append(
                    f"{where}: decision columns {sorted(a_row.decisions)} vs "
                    f"{sorted(b_row.decisions)}"
                )
                continue
            for column, a in a_row.decisions.items():
                b = b_row.decisions[column]
                if a != b:
                    flips.append(Divergence(where, column, a, b))

    return GoldenComparison(
        rows=min(len(reference.trace), len(candidate.trace)),
        structural=tuple(structural[:max_examples]),
        divergences=tuple(divergences[:max_examples]),
        flips=tuple(flips[:max_examples]),
        divergence_count=len(divergences),
        flip_count=len(flips),
        max_abs_diff=worst,
    )
