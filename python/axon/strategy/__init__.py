"""``axon.strategy`` — the inward port, Python side (``docs/06-strategy-contract.md``).

A strategy implements event callbacks and emits target positions through the
provided :class:`StrategyContext`. It never talks to the venue, manages orders,
or touches nonces/signing — that is the Rust core's job. The same object runs in
backtest, sandbox and live, which is where parity comes from: it is the same
code, not a reimplementation (``docs/07``).

Two invariants are enforced structurally rather than documented and hoped for,
because both fail silently:

* **event time comes from the event.** :class:`StrategyContext` holds no clock;
  ``ts_event`` is bound by the runner around each callback. A wall-clock stamp at
  emit time is the staleness / training-serving-skew leak ``docs/03`` calls the
  number-one silent quality killer.
* **real units in, fixed-point on the wire, converted once.** ``emit_target``
  takes ``0.25``, not ``25_000_000``. Every hand-written ``* 1e8`` is a chance to
  be 100× wrong in the direction of a real position.

Wiring a strategy to the ring is :mod:`axon.live`.
"""

from axon.strategy.base import Strategy
from axon.strategy.config import StrategyConfig
from axon.strategy.context import (
    DEFAULT_TTL_MS,
    TTL_OPERATOR_CEILING,
    URGENCY_CROSS,
    URGENCY_JOIN,
    URGENCY_MAX,
    URGENCY_POST_ONLY,
    URGENCY_TAKE,
    NotInEventScope,
    StrategyContext,
    StrategyError,
)
from axon.strategy.events import Bar, Bbo, Fill, OrderUpdate, Side, Tick, Timer, Trade
from axon.strategy.reference import MeanReversion, MeanReversionParams

#: The urgency levels and the TTL sentinel are part of the *wire* contract, not of one
#: module's internals: a strategy author choosing `URGENCY_POST_ONLY` is choosing a TIF
#: the Rust planner reads (ADR-0014 §3). Re-exported here so nobody reaches into
#: `axon.strategy.context` for them — an import path that names a submodule is one that
#: keeps working after the submodule stops being the contract.
__all__ = [
    "DEFAULT_TTL_MS",
    "TTL_OPERATOR_CEILING",
    "URGENCY_CROSS",
    "URGENCY_JOIN",
    "URGENCY_MAX",
    "URGENCY_POST_ONLY",
    "URGENCY_TAKE",
    "Bar",
    "Bbo",
    "Fill",
    "MeanReversion",
    "MeanReversionParams",
    "NotInEventScope",
    "OrderUpdate",
    "Side",
    "Strategy",
    "StrategyConfig",
    "StrategyContext",
    "StrategyError",
    "Tick",
    "Timer",
    "Trade",
]
