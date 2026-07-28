"""A reference strategy that exists to be run end to end.

Phase 4's exit criterion is "a trivial Python strategy emits target positions;
Rust executes on testnet" (``docs/08``). This is that strategy. It is
deliberately boring — a rolling z-score toggling between long, flat and short —
because its job is to exercise the *seam*, not to make money. What it does
demonstrate is the handful of things every real strategy has to get right:

* prices arrive as fixed-point and become floats only for the statistic;
* the target goes out in **real units** and is scaled exactly once, by the context;
* nothing is emitted until the window is warm;
* a degenerate window emits nothing rather than a NaN;
* the target has hysteresis, so it does not thrash across a threshold;
* it emits only on *change*, and relies on the runner's heartbeat to prove it is
  alive while it has nothing to say.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from axon.contracts import from_fixed
from axon.strategy.base import Strategy
from axon.strategy.config import StrategyConfig
from axon.strategy.context import StrategyContext
from axon.strategy.events import Bar, Tick, Trade


@dataclass(frozen=True)
class MeanReversionParams:
    """Parameters for :class:`MeanReversion`, in real units.

    ``entry_z``/``exit_z`` form a band, not a single threshold: with one
    threshold, a price sitting on it flips the target on every tick, and each
    flip is a round trip paid in fees and rate-limit budget.
    """

    symbol_id: int
    window: int = 32
    entry_z: float = 1.5
    exit_z: float = 0.5
    max_position: Decimal = Decimal("0.01")
    urgency: int = 0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2 to have a standard deviation, got {self.window}")
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError(
                "need 0 <= exit_z < entry_z for hysteresis, got "
                f"exit_z={self.exit_z} entry_z={self.entry_z}"
            )
        if self.max_position <= 0:
            raise ValueError(f"max_position must be positive, got {self.max_position}")

    @classmethod
    def from_config(cls, config: StrategyConfig) -> "MeanReversionParams":
        params = dict(config.params)
        if "max_position" in params:
            # Through Decimal(str(...)), never float(...): 0.01 has no exact float
            # representation, and a size that is 0.010000000000000000208 wide is a
            # size the venue will round somewhere we did not choose.
            params["max_position"] = Decimal(str(params["max_position"]))
        if not config.symbols:
            raise ValueError("StrategyConfig.symbols must name the symbol this strategy trades")
        return cls(symbol_id=config.symbols[0], **params)


class MeanReversion(Strategy):
    """Long when the price is cheap against its rolling mean, short when rich, flat between."""

    def __init__(self, params: MeanReversionParams) -> None:
        self.params = params
        self._prices: deque[float] = deque(maxlen=params.window)
        self._target = Decimal(0)

    # ── data callbacks: three feeds, one decision path ──

    def on_tick(self, tick: Tick, ctx: StrategyContext) -> None:
        self._observe(tick.symbol_id, tick.px, ctx)

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        self._observe(trade.symbol_id, trade.px, ctx)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        self._observe(bar.symbol_id, bar.close, ctx)

    # ── lifecycle ──

    def on_reset(self) -> None:
        self._prices.clear()
        self._target = Decimal(0)

    def on_save(self) -> dict[str, Any]:
        return {"prices": list(self._prices), "target": str(self._target)}

    def on_load(self, state: dict[str, Any]) -> None:
        self._prices = deque(state.get("prices", []), maxlen=self.params.window)
        # str → Decimal, so a restart resumes on exactly the size it left off on.
        # Resuming a hair off would make the first post-restart signal a real
        # order for the difference, for no reason anyone could later explain.
        self._target = Decimal(str(state.get("target", "0")))

    # ── the decision ──

    def _observe(self, symbol_id: int, px_fixed: int, ctx: StrategyContext) -> None:
        if symbol_id != self.params.symbol_id:
            return
        # Fixed-point in, float only for the statistic. A z-score is a statistic,
        # not money; the money is `max_position`, which never leaves Decimal. The
        # conversion goes through the contract helper rather than a literal 1e8 —
        # the scale is the contract's to change, not this file's to hardcode.
        self._prices.append(from_fixed(px_fixed))
        if len(self._prices) < self.params.window:
            # Acting on a partial window is the classic warmup bug: the first few
            # samples give a tiny standard deviation, every z-score looks extreme,
            # and the strategy opens its largest position on its worst information.
            return

        window = np.fromiter(self._prices, dtype=np.float64, count=len(self._prices))
        sigma = float(window.std())  # population std: the window IS the population
        if sigma <= 0.0:
            # A frozen or stepped feed gives zero variance; dividing by it yields
            # inf/NaN, and a NaN target is either rejected downstream or, worse,
            # coerced to something plausible.
            return

        z = (window[-1] - float(window.mean())) / sigma
        target = self._next_target(z)
        if target == self._target:
            # A target position is idempotent — re-sending it costs ring space and
            # rate-limit budget and tells the executor nothing new. Silence is
            # safe here only because the runner heartbeats independently; without
            # that, "no signal" and "dead process" would look identical.
            return
        self._target = target
        ctx.emit_target(symbol_id, target, urgency=self.params.urgency)

    def _next_target(self, z: float) -> Decimal:
        p = self.params
        if z <= -p.entry_z:
            return p.max_position  # cheap versus the mean → long
        if z >= p.entry_z:
            return -p.max_position
        if abs(z) <= p.exit_z:
            return Decimal(0)
        return self._target  # inside the band: hold, do not thrash


__all__ = ["MeanReversion", "MeanReversionParams"]
