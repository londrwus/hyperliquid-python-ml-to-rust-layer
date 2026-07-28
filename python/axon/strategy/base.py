"""The strategy base class — the inward port, Python side (``docs/06-strategy-contract.md``).

The engine calls you; you never call the engine. A strategy implements the
callbacks it cares about and emits through the provided
:class:`~axon.strategy.context.StrategyContext`. It has no reference to the ring,
the venue, the order book, or a clock — which is what makes the engine
strategy-agnostic and the strategy transplantable between backtest, sandbox and
live without changing a line.

This mirrors the Rust ``axon-strategy`` trait so a strategy can be promoted from
Python (Boundary B) to Rust (Boundary A) without the engine, the adapters, or the
tests changing shape (``docs/02``).
"""

from __future__ import annotations

from typing import Any

from axon.strategy.context import StrategyContext
from axon.strategy.events import Bar, Bbo, Fill, OrderUpdate, Tick, Timer, Trade


class Strategy:
    """Override only the callbacks you need; the defaults do nothing.

    Lifecycle, in the order the runner drives it:

    1. ``on_load(state)`` — only on a warm restart, before anything else, with
       whatever the previous run's ``on_save()`` returned.
    2. ``on_start(ctx)`` — once, before the first event. It runs inside an event
       scope stamped with the session's start time, so it *may* emit (typically
       to declare an initial flat target).
    3. the data / execution / timer callbacks — many, in event-time order.
    4. ``on_stop(ctx)`` — once, inside an event scope stamped with the session's
       last event time. Emitting a flatten here is legitimate; relying on it is
       not, since a crash skips it (that is what the Rust-side dead-man's switch
       is for — ``docs/02``).
    5. ``on_save()`` — returns a JSON-serializable blob the runtime persists.

    ``on_reset()`` returns the object to its as-constructed state; the parity
    harness uses it to run the same instance over several replays without
    carrying state between them.

    Ordering guarantee: callbacks are delivered in non-decreasing ``ts_event``
    order. Anything a callback emits is stamped with *that event's* time, so the
    same event sequence always produces the same signal bytes.
    """

    # ── lifecycle ──

    def on_start(self, ctx: StrategyContext) -> None:
        """Called once before the first event. Subscribe-equivalent setup goes here."""

    def on_stop(self, ctx: StrategyContext) -> None:
        """Called once after the last event, on a clean shutdown only."""

    def on_reset(self) -> None:
        """Discard all accumulated state, as if freshly constructed."""

    # ── data ──

    def on_tick(self, tick: Tick, ctx: StrategyContext) -> None:
        """A last-price update from a feed that does not separate trades from quotes."""

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        """A trade printed on the tape."""

    def on_bbo(self, bbo: Bbo, ctx: StrategyContext) -> None:
        """Top-of-book change. Named to match the Rust trait's ``on_bbo``."""

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        """A bar that has *closed* at ``bar.ts_event``."""

    # ── execution ──

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        """One of our orders executed. The core owns position accounting; react, don't tally."""

    def on_order_update(self, update: OrderUpdate, ctx: StrategyContext) -> None:
        """A lifecycle transition on one of our orders."""

    # ── timers ──

    def on_timer(self, timer: Timer, ctx: StrategyContext) -> None:
        """A scheduled wake-up, stamped with the time it was scheduled for."""

    # ── state (warm restart) ──

    def on_save(self) -> dict[str, Any]:
        """Return JSON-serializable state.

        Keep money as strings, not floats: a Decimal size round-tripped through a
        float comes back subtly different, and a warm restart that resumes with a
        *slightly* different target immediately trades the difference.
        """
        return {}

    def on_load(self, state: dict[str, Any]) -> None:
        """Restore what ``on_save`` produced. Called before ``on_start``."""


__all__ = ["Strategy"]
