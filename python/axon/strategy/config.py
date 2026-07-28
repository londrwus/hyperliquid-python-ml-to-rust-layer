"""Strategy configuration as a serializable object, not constructor arguments.

``docs/06`` principle 3: config is data. A run is then reproducible from a file —
which is what later allows the same definition to drive a distributed backtest, a
shadow session, and a live process without three ways of spelling the same
parameters.

The runner builds the :class:`~axon.strategy.context.StrategyContext` from this,
so ``model_version`` and ``default_ttl_ms`` are declared once, in the artifact
that gets archived next to the run, rather than passed by hand at each call site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from axon.strategy.context import DEFAULT_TTL_MS


@dataclass(frozen=True)
class StrategyConfig:
    """Everything the runtime needs to instantiate and stamp a strategy run.

    ``params`` is the strategy-specific bag; keep it JSON-serializable so the
    whole config can be archived alongside the signals it produced. Risk limits
    deliberately do **not** live here as anything the strategy can read: they are
    enforced by ``axon-risk`` on the Rust hot path, where a strategy cannot
    negotiate with them (``docs/06``).
    """

    name: str
    version: int = 1
    model_version: int = 1
    symbols: tuple[int, ...] = ()
    default_ttl_ms: int = DEFAULT_TTL_MS
    feature_spec_ref: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["symbols"] = list(self.symbols)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategyConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # A silently ignored key is a parameter the operator believes is in
            # effect and is not — the config equivalent of a dropped signal.
            raise ValueError(f"unknown StrategyConfig keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "symbols" in kwargs:
            kwargs["symbols"] = tuple(kwargs["symbols"])
        return cls(**kwargs)


__all__ = ["StrategyConfig"]
