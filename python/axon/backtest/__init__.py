"""``axon.backtest`` — the bottom rung of the validation ladder, from Python.

``docs/07-parity-and-testing.md`` puts one thing under everything else: *capture a
real event log, replay it through the exact production code path, assert outputs
match a stored reference within tolerance and that no discretized decision flips.*
This package is the Python half of that rung. :class:`Backtester` drives the Rust
replay path and returns a :class:`BacktestResult`; :func:`compare_to_golden` performs
the comparison.

**It shells out to the Rust ``replay_log`` binary and never reimplements the event
semantics in Python.** That is the load-bearing decision, and there is no fallback:
if the binary cannot be found or built, :class:`ReplayUnavailable` is raised. A
Python reader of the same log would be a second implementation of order-book
replacement, mid fallback and mark staleness — two implementations that agree on the
day they are written and drift silently afterwards, with the drift landing inside the
harness meant to detect drift. See :mod:`axon.backtest.runner` for the full argument.

**What is under the golden.** The whole production chain, not just market data:
`MarketDataProcessor` -> `MarkCache` -> `OrderTracker` -> `SignalReader` -> `Planner`,
driven by the same `CoreHandler` a live session installs. So a run compares the
tracker's reconciled position *and* the orders and `cloid`s the planner emitted, which
are the two things every rung above this one exists to compare.

**What this is not.** A run here is a deterministic re-drive of that chain over a
captured session, not a profit-and-loss simulation. The log carries the fills the
*captured* session received; an order a replay would place gets no fill, moves no
price, takes no queue position, and is not even written into the tracker — doing so
would mean inventing an acknowledgement the venue never gave. Answering "would this
have made money?" needs a simulated venue behind the provider port, which is a
separate adapter and deliberately not this package (ADR-0018 §7).
"""

from axon.backtest.golden import (
    Divergence,
    GoldenComparison,
    GoldenMismatch,
    compare_to_golden,
)
from axon.backtest.runner import (
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    SIGNAL_COUNTERS,
    SIGNALS_SUFFIX,
    BacktestError,
    Backtester,
    BacktestResult,
    PlannedCancel,
    PlannedOrder,
    ReplayUnavailable,
    TraceRow,
    find_replay_binary,
)

__all__ = [
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "SIGNALS_SUFFIX",
    "SIGNAL_COUNTERS",
    "BacktestError",
    "BacktestResult",
    "Backtester",
    "Divergence",
    "GoldenComparison",
    "GoldenMismatch",
    "PlannedCancel",
    "PlannedOrder",
    "ReplayUnavailable",
    "TraceRow",
    "compare_to_golden",
    "find_replay_binary",
]
