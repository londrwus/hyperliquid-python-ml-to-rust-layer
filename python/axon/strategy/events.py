"""The events a Python strategy receives.

Every event carries its **own** ``ts_event`` in nanoseconds — the instant the
venue (or the replay) says the thing happened, not the instant Python saw it.
That field is the only clock in this package: :class:`~axon.strategy.context.StrategyContext`
stamps outgoing signals from the event being handled, so a replay of the same
events produces byte-identical signals (``docs/07-parity-and-testing.md``).

Money is fixed-point, never float: every price/size field here is a signed
integer in units of ``10^-FIXED_POINT_DECIMALS``, the same encoding the wire uses
(``contracts/schema.toml``). :func:`axon.contracts.from_fixed` is the one
sanctioned way to get a real number out of one, and it is only for *feature*
math — a feature is a statistic, not money, so float is correct there. Anything
that ends up as a position size goes back out through
:meth:`StrategyContext.emit_target`, which converts once.

``ts_event`` is an ``int`` and must stay one. A float64 carries 53 bits of
mantissa; a 2026 nanosecond timestamp needs 61, so passing event times around as
floats silently rounds them to ~256 ns buckets and two events that were ordered
become simultaneous.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from axon.contracts import from_fixed


class Side(str, Enum):
    """Aggressor side of a trade, or the side of one of our own fills."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Tick:
    """A single last-price update from a feed that does not distinguish trades
    from quotes. The cheapest event a strategy can be driven by."""

    symbol_id: int
    ts_event: int
    px: int

    @property
    def px_real(self) -> float:
        return from_fixed(self.px)


@dataclass(frozen=True, slots=True)
class Trade:
    """A trade printed on the tape, with the aggressor's side."""

    symbol_id: int
    ts_event: int
    px: int
    sz: int
    side: Side

    @property
    def px_real(self) -> float:
        return from_fixed(self.px)


@dataclass(frozen=True, slots=True)
class Bbo:
    """Top of book. ``mid`` is computed in fixed-point on purpose — a mid taken
    through float and converted back is off by up to half a wire unit, which is
    enough to move a rounded target size by one lot."""

    symbol_id: int
    ts_event: int
    bid_px: int
    bid_sz: int
    ask_px: int
    ask_sz: int

    @property
    def mid(self) -> int:
        return (self.bid_px + self.ask_px) // 2

    @property
    def mid_real(self) -> float:
        return from_fixed(self.mid)


@dataclass(frozen=True, slots=True)
class Bar:
    """An OHLCV bar closed at ``ts_event`` — the *close* time, never the open
    time. A bar stamped with its open time is the textbook lookahead leak: the
    strategy would appear to have acted on the bar's close a whole bar early,
    and the backtest would print returns live can never reproduce."""

    symbol_id: int
    ts_event: int
    open: int
    high: int
    low: int
    close: int
    volume: int

    @property
    def close_real(self) -> float:
        return from_fixed(self.close)


@dataclass(frozen=True, slots=True)
class Fill:
    """One of our own executions, fed back so a strategy can track what it
    actually holds. Position *accounting* is the Rust core's job; this is here so
    a strategy can react (e.g. stop adding once it is filled)."""

    symbol_id: int
    ts_event: int
    px: int
    qty: int
    side: Side

    @property
    def signed_qty(self) -> int:
        """Fill quantity signed by side — the form position math actually wants."""
        return self.qty if self.side is Side.BUY else -self.qty


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    """A lifecycle transition on one of our orders, normalized by the core."""

    symbol_id: int
    ts_event: int
    order_id: int
    status: str
    remaining_qty: int


@dataclass(frozen=True, slots=True)
class Timer:
    """A scheduled wake-up. ``ts_event`` is the time the timer was *scheduled
    for*, not when it fired, so a timer-driven strategy replays identically even
    when the live process was late servicing it."""

    ts_event: int
    name: str


__all__ = ["Bar", "Bbo", "Fill", "OrderUpdate", "Side", "Tick", "Timer", "Trade"]
