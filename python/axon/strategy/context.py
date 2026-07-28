"""The facade a strategy emits through — and the bookkeeping it is not trusted with.

A strategy's job is ``events → features → inference → Signal``
(``docs/06-strategy-contract.md``). Everything *around* that — the sequence
number, the model version, the validity window, the event time, the fixed-point
encoding — is bookkeeping that is easy to get subtly wrong and impossible to
notice afterwards. So none of it is the author's to supply:

* ``seq`` is issued here, monotonically and without gaps, so the Rust consumer's
  gap detector means something.
* ``model_version`` is fixed for the run, so every live decision can be replayed
  against the model that made it (``docs/03``).
* ``ttl_ms`` defaults, because a signal with no stated validity window is a
  signal the executor has to guess about — and when a strategy genuinely has no
  opinion it says so with :data:`TTL_OPERATOR_CEILING` rather than by leaving a
  field unset and hoping.
* ``ts_event`` **comes from the event being handled**. Not ``time.time()``. This
  module contains no clock at all, which is the point: a wall-clock stamp at emit
  time is the staleness / training-serving-skew leak ``docs/03`` names as the
  number-one silent quality killer, and the only way to make it not happen is to
  make it unavailable. A strategy that wants to know "now" has to look at the
  event, which is the correct answer anyway.
* real units are converted to fixed-point **once, here**, via
  :func:`axon.contracts.to_fixed`. Every place a human writes ``* 1e8`` is a
  place a position ends up 100× wrong; there is exactly one such place in Axon
  and it is not in strategy code.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator, Union

import numpy as np

from axon.contracts import (
    FLAG_CLOSE,
    FLAG_REDUCE_ONLY,
    SCHEMA_VERSION,
    SIGNAL_DTYPE,
    new_signal,
    to_fixed,
)

#: Default validity window for an emitted signal. At crypto-perp speeds a target
#: position formed half a second ago is an opinion about a book that no longer
#: exists; the executor should be told to stop chasing it rather than left to
#: assume the target is eternal.
DEFAULT_TTL_MS = 500

#: ``ttl_ms = 0`` — "I have no opinion about staleness; use the operator's ceiling."
#:
#: The one reading of zero that is safe, and it is not the obvious one. Zero is also
#: the value of a field nobody wrote, so whatever it means has to be the *right*
#: answer for a producer that never thought about the question — which rules out
#: "never expires" immediately. It could have been "already expired", and that was
#: rejected: a bug that leaves the field unset would then silently stop a strategy
#: trading, and a strategy that stops trading for free is indistinguishable from a
#: quiet market. So zero defers to ``intent.max_signal_age_ms``, the operator's hard
#: ceiling, which every non-zero ``ttl_ms`` is *also* capped by (ADR-0014 §1,
#: ADR-0020 §4). A strategy can only ever ask for a **shorter** window than the
#: operator allows, never a longer one, and the ceiling always binds.
TTL_OPERATOR_CEILING = 0

# ── urgency: the same four numbers mean the same four things on both sides ──
#
# ``urgency`` is a bare ``u8`` on the wire, and a bare number is exactly how the two
# halves of a boundary drift apart: an author picking "2 sounds about right" has no
# way to know that 2 crosses the spread. The Rust planner's table is the contract
# (ADR-0014 §3, ``axon_strategy::URGENCY_TABLE``); these names are that table, so a
# strategy says what it wants rather than what it guesses.
#
# What increases is not price aggression — 0 and 1 price identically — but **what you
# are willing to give up to get the position on**.

#: Post-only at the near touch (buy at the bid). Gives up fill certainty; cannot pay
#: a taker fee, and is first in Hyperliquid's in-block priority.
URGENCY_POST_ONLY = 0

#: GTC at the near touch. Same price as :data:`URGENCY_POST_ONLY`, but it cannot be
#: *rejected* for crossing if the book moves under us. Buys certainty of placement,
#: not of fill.
URGENCY_JOIN = 1

#: GTC at the far touch (buy at the ask). Crosses the spread but stays a limit, so a
#: partial fill rests at a price you chose and slippage is bounded by one spread.
URGENCY_CROSS = 2

#: IOC through the far touch, plus the operator's slippage allowance. Hyperliquid has
#: no native market order — a market order there *is* this. Leaves no remainder
#: resting, which is the point: an urgent exit that half-fills and rests is an
#: unmanaged position.
URGENCY_TAKE = 3

#: The most aggressive level the table defines. Anything above it **saturates** into
#: it rather than being refused, so ``urgency=255`` means "as fast as possible" and
#: not "dropped record". Refusing it would drop precisely the signal you least want
#: dropped.
URGENCY_MAX = URGENCY_TAKE

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_U32_MAX = 2**32 - 1
_U8_MAX = 2**8 - 1

#: What ``emit_target`` accepts for a quantity or a price, in **real** units.
#: ``str`` is accepted and routed through :class:`~decimal.Decimal` because that
#: is the only lossless way to spell a size that came from a config file.
Real = Union[int, float, Decimal, str]


class StrategyError(Exception):
    """Base for the errors this contract raises at a strategy author."""


class NotInEventScope(StrategyError):
    """Raised when a strategy emits outside a callback.

    There is no event to take the time from, and the alternative — falling back
    to a wall clock — is precisely the substitution this class exists to prevent.
    Emitting from a constructor or a background thread is a bug, so it fails
    loudly instead of producing a signal stamped with a plausible-looking lie.
    """


def _check_event_time(ts_event: int) -> int:
    # bool is an int subclass and would sail through; a float is the real hazard.
    if isinstance(ts_event, bool) or not isinstance(ts_event, int):
        raise TypeError(
            f"ts_event must be an int of nanoseconds, got {type(ts_event).__name__}. "
            "float64 holds 53 mantissa bits and a 2026 nanosecond timestamp needs 61, "
            "so a float event time silently rounds into ~256 ns buckets and reorders events."
        )
    if ts_event < 0:
        raise ValueError(f"ts_event must be non-negative nanoseconds, got {ts_event}")
    if ts_event > _I64_MAX:
        raise ValueError(f"ts_event {ts_event} does not fit the contract's i64 field")
    return ts_event


def _to_wire(value: Real, *, what: str) -> int:
    """Convert a real quantity/price to the contract's fixed-point integer.

    The single conversion point. Everything it rejects, it rejects because the
    alternative is a silently wrong position rather than an error.
    """
    if isinstance(value, str):
        value = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            # A NaN prediction means a feature window went bad (a zero-variance
            # denominator, an empty rolling mean). Letting it through converts a
            # broken model into a garbage or accidentally-flat target.
            raise ValueError(
                f"{what} is {value!r}; a non-finite prediction must never become a target"
            )
    elif not isinstance(value, (int, Decimal)):
        raise TypeError(
            f"{what} must be int/float/Decimal/str in real units, got {type(value).__name__}"
        )

    fixed = to_fixed(value)
    if not _I64_MIN <= fixed <= _I64_MAX:
        # numpy would raise on assignment anyway, but the message would name a
        # C type rather than the mistake, which is almost always a value that was
        # already scaled by hand and is now scaled twice.
        raise ValueError(
            f"{what}={value!r} is {fixed} in fixed-point, outside the contract's i64 range. "
            "Pass real units — the 10^8 scaling happens here, exactly once."
        )
    return fixed


class StrategyContext:
    """The only channel out of a strategy. The runner drains it after each callback.

    A strategy never sees the ring, the venue, or a clock; it sees this object.
    Construct one per run and hand it to a :class:`~axon.live.runner.StrategyRunner`.
    """

    __slots__ = ("_pending", "_model_version", "_default_ttl_ms", "_seq", "_ts_event")

    def __init__(
        self,
        *,
        model_version: int,
        default_ttl_ms: int = DEFAULT_TTL_MS,
        first_seq: int = 0,
    ) -> None:
        if not 1 <= model_version <= _U32_MAX:
            # Zero is the zero value of an unwritten record, so allowing it makes
            # "nobody stamped this" indistinguishable from "model 0 stamped this"
            # in an audit — and an unreproducible live decision is not auditable.
            raise ValueError(f"model_version must be in 1..{_U32_MAX}, got {model_version}")
        self._model_version = int(model_version)
        self._default_ttl_ms = _check_ttl(default_ttl_ms)
        if first_seq < 0:
            raise ValueError(f"first_seq must be non-negative, got {first_seq}")
        self._seq = int(first_seq)
        self._ts_event: int | None = None
        self._pending: list[np.ndarray] = []

    # ── properties ──

    @property
    def model_version(self) -> int:
        return self._model_version

    @property
    def default_ttl_ms(self) -> int:
        return self._default_ttl_ms

    @property
    def next_seq(self) -> int:
        """The sequence number the next emitted signal will carry.

        Persist this across a restart if the consumer's gap detector must survive
        one; restarting at 0 reads as a rewind.
        """
        return self._seq

    @property
    def ts_event(self) -> int | None:
        """Event time of the callback currently running, or ``None`` outside one."""
        return self._ts_event

    # ── engine-side plumbing ──

    @contextmanager
    def event(self, ts_event: int) -> Iterator["StrategyContext"]:
        """Bind the context to one event for the duration of a callback.

        The runner owns this; a strategy has no reason to call it. Nesting is
        refused because the inner scope's exit would restore a *stale* time that
        the outer callback would then keep emitting under — the exact bug this
        whole arrangement exists to rule out.
        """
        if self._ts_event is not None:
            raise StrategyError(
                f"event scope already open at ts_event={self._ts_event}; "
                "nesting would leave a stale event time bound after the inner scope exits"
            )
        self._ts_event = _check_event_time(ts_event)
        try:
            yield self
        finally:
            self._ts_event = None

    def take_pending(self) -> list[np.ndarray]:
        """The runner takes what the callback emitted. Leaves the context empty."""
        out, self._pending = self._pending, []
        return out

    def pending_len(self) -> int:
        return len(self._pending)

    # ── the strategy-facing API ──

    def emit_target(
        self,
        symbol_id: int,
        target_qty: Real,
        *,
        urgency: int = 0,
        price_band: Real | None = None,
        ttl_ms: int | None = None,
        reduce_only: bool = False,
        max_order_age_ms: int = 0,
        ts_cause: int | None = None,
    ) -> np.ndarray:
        """Declare the position you want, in **real units** (e.g. ``0.25`` BTC).

        The Rust core decides *how* to get there (ADR-0006) — that is why a
        missed signal is survivable and why execution mechanics are not a
        strategy's business.

        ``price_band`` is the worst price you will accept, also in real units;
        omit it for no band. Returns the record for inspection/testing; it has
        already been queued.

        ``max_order_age_ms`` is how long an order this signal places may keep its
        place at the venue, and it is **not** ``ttl_ms`` (ADR-0031). ``ttl_ms`` is a
        signal *admission* window: the Rust reader consumes it before the planner
        ever sees the record, clamped against the operator's ``max_signal_age_ms``,
        so a large ``ttl_ms`` buys a resting order exactly nothing. ``0`` on either
        defers to the operator's ceiling — never "forever".

        ``ts_cause`` is the event time of the **observation this decision answers** —
        for a bar strategy, the bar's own ``ts_event``, which is its close. Pass it and
        the runtime can give the bar-close-to-decision gap a ceiling; omit it and that
        stage is simply not measured for this record.

        It is a separate argument from the event scope on purpose, and the reason is
        measured rather than stylistic. ``StrategyContext.event(ts)`` binds the stamp the
        *reader* ages the record against, and on a live bar session that has to be the
        producer's wall clock: an m1 bar's own close is already seconds old when the bar
        arrives, so a record stamped with it is refused as ``expired`` before it can
        become an order — which is exactly what happened to the first live run. So the
        two stamps have to be different values, and this is the one that says what the
        decision was about.
        """
        rec = new_signal(
            seq=self._seq,
            ts_event=self._require_scope(),
            ts_cause=0 if ts_cause is None else _check_event_time(ts_cause),
            symbol_id=_check_u32(symbol_id, "symbol_id"),
            target_qty=_to_wire(target_qty, what="target_qty"),
            price_band=0 if price_band is None else _to_wire(price_band, what="price_band"),
            urgency=_check_urgency(urgency),
            ttl_ms=self._default_ttl_ms if ttl_ms is None else _check_ttl(ttl_ms),
            model_version=self._model_version,
            flags=FLAG_REDUCE_ONLY if reduce_only else 0,
            max_order_age_ms=_check_ttl(max_order_age_ms, "max_order_age_ms"),
        )
        return self._queue(rec)

    def emit_close(
        self,
        symbol_id: int,
        *,
        urgency: int = 0,
        ttl_ms: int | None = None,
        ts_cause: int | None = None,
    ) -> np.ndarray:
        """Flatten ``symbol_id`` unconditionally (``FLAG_CLOSE``; ``target_qty`` ignored).

        Distinct from ``emit_target(symbol_id, 0)``: a zero target is an opinion
        that can be netted against other intents, whereas close says *get out*
        regardless of what the record's quantity field happens to hold.
        """
        rec = new_signal(
            seq=self._seq,
            ts_event=self._require_scope(),
            symbol_id=_check_u32(symbol_id, "symbol_id"),
            urgency=_check_urgency(urgency),
            ttl_ms=self._default_ttl_ms if ttl_ms is None else _check_ttl(ttl_ms),
            model_version=self._model_version,
            flags=FLAG_CLOSE,
            ts_cause=0 if ts_cause is None else _check_event_time(ts_cause),
        )
        return self._queue(rec)

    def emit(self, signal: np.ndarray) -> np.ndarray:
        """Emit a pre-built record (escape hatch for a strategy that needs a field
        this API does not expose yet).

        The bookkeeping fields — ``seq``, ``ts_event``, ``model_version``,
        ``schema_version`` — are **overwritten**, not trusted. An author who fills
        in ``ts_event`` by hand is the failure mode this class exists to prevent,
        so the escape hatch is not allowed to lie about it either. The record is
        copied first, so the caller's array is left alone.
        """
        if signal.dtype != SIGNAL_DTYPE:
            raise TypeError(f"signal must have the contract dtype, got {signal.dtype}")
        rec = signal.copy()
        rec["seq"] = self._seq
        rec["ts_event"] = self._require_scope()
        rec["model_version"] = self._model_version
        rec["schema_version"] = SCHEMA_VERSION
        return self._queue(rec)

    # ── internals ──

    def _require_scope(self) -> int:
        if self._ts_event is None:
            raise NotInEventScope(
                "emit() called outside an event scope: there is no event to take ts_event from. "
                "Emit from a strategy callback; the runner binds the event's own time around it."
            )
        return self._ts_event

    def _queue(self, rec: np.ndarray) -> np.ndarray:
        # seq advances only once the record is fully built, so a rejected
        # conversion (a NaN target, an out-of-range size) leaves no hole in the
        # sequence for the consumer's gap detector to alarm on.
        self._seq += 1
        self._pending.append(rec)
        return rec


def _check_u32(value: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    if not 0 <= value <= _U32_MAX:
        raise ValueError(f"{what} must be in 0..{_U32_MAX}, got {value}")
    return value


def _check_urgency(urgency: int) -> int:
    if isinstance(urgency, bool) or not isinstance(urgency, int):
        raise TypeError(f"urgency must be an int, got {type(urgency).__name__}")
    if not 0 <= urgency <= _U8_MAX:
        raise ValueError(f"urgency must be in 0..{_U8_MAX} (0 = most passive), got {urgency}")
    # Values above URGENCY_MAX are *not* refused: the executor saturates them into the
    # most aggressive row, and a strategy writing 255 means "as fast as possible".
    # Rejecting it would drop precisely the signal you least want dropped.
    return urgency


def _check_ttl(ttl_ms: int, what: str = "ttl_ms") -> int:
    """Validate a millisecond duration destined for a ``u32`` on the wire.

    ``what`` names the field in the error, because ``ttl_ms`` and
    ``max_order_age_ms`` are two different durations with the same shape and the
    same failure modes, and an error message naming the wrong one sends the reader
    to the wrong field (ADR-0031).
    """
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int):
        raise TypeError(f"{what} must be an int of milliseconds, got {type(ttl_ms).__name__}")
    if ttl_ms < 0:
        # Not a duration. Unlike 0 there is no reading of it at all, and the field is
        # unsigned on the wire, so letting it through would wrap into ~49 days.
        raise ValueError(f"{what} must be non-negative milliseconds, got {ttl_ms}")
    if ttl_ms > _U32_MAX:
        raise ValueError(f"{what} must fit u32 (<= {_U32_MAX}), got {ttl_ms}")
    # 0 is deliberately allowed: it is TTL_OPERATOR_CEILING, the only way a producer
    # can say "use the operator's policy". It used to be refused here, on the grounds
    # that the contract gave zero no meaning — but the consumer had to decide
    # *something* about a field that is zero whenever nobody wrote it, and refusing to
    # emit zero did not stop zero arriving. One side defining it and the other
    # forbidding it is how a boundary ends up with two contracts.
    return ttl_ms


__all__ = [
    "DEFAULT_TTL_MS",
    "TTL_OPERATOR_CEILING",
    "URGENCY_CROSS",
    "URGENCY_JOIN",
    "URGENCY_MAX",
    "URGENCY_POST_ONLY",
    "URGENCY_TAKE",
    "NotInEventScope",
    "Real",
    "StrategyContext",
    "StrategyError",
]
