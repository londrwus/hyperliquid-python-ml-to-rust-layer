"""``perp_bar`` — one real ML strategy: a bar classifier on a Hyperliquid perp.

What it does, in one line: on every **closed** hourly bar it recomputes a
nine-column feature matrix over its own buffer, asks an XGBoost classifier for the
probability that the next four hours close higher, and turns that probability into
a target position with a hysteresis band.

The design decisions that are not obvious, and what each one buys:

**Every feature has a finite lookback.** No EMA, no expanding statistic. An EMA
never forgets its seed, so a serving path that holds a bounded buffer computes a
different number from the research path that saw the whole history — biggest right
after a restart, which is the moment nobody is comparing feature values. With only
windowed transforms in the spec, a buffer of :data:`SERVING_BUFFER_BARS` bars
reproduces the offline matrix *bit for bit*, and the feature-parity gate compares
transforms rather than histories. That is the difference between "parity within
tolerance" and parity.

**The whole buffer is recomputed every bar, not updated incrementally.** An
incremental update would be a second implementation of every transform in
:mod:`axon.features` — the exact thing ``docs/03`` names as the bug. At a one-hour
cadence, recomputing nine columns over a few hundred rows costs microseconds we do
not have to account for; if a strategy ever needs the incremental form it goes to
Rust behind the bit-equivalence gate (Boundary A), not into a second Python path.

**It trades on the bar's close time, and holds no clock.** ``Bar.ts_event`` is the
close, the context binds it around the callback, and this module imports nothing
that can tell the time. A bar stamped with its open would let the strategy act on
a close an hour before it happened.

**It refuses rather than guesses.** A warmup that is not full, a non-finite feature
(a bar with no range, a zero price in the feed), a non-finite probability — each
emits nothing at all. The alternative is a target position sized by a broken
window, and a flat target is not a safe default either: it is an instruction to
close a position that may be perfectly fine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from axon.features import BAR_INPUTS, FeatureDef, FeatureSpec, bar_inputs
from axon.models.inference import Predictor
from axon.strategy.base import Strategy
from axon.strategy.config import StrategyConfig
from axon.strategy.context import StrategyContext
from axon.strategy.events import Bar

#: The feature recipe this strategy is served with, and the one it was trained on.
#: Nine columns over a closed OHLCV bar — three views of the return, two of the
#: range, one of volume — chosen so that every one of them is computable from what
#: the venue's ``candle`` subscription actually publishes. A feature that needs the
#: book or the tape would be a research artifact this strategy could never serve
#: (ADR-0022 names the live source of each).
#:
#: ``version=1`` and the fingerprint that follows from it are what a model artifact
#: records; changing any window here is a new version, which is what stops a model
#: fitted on the old numbers from being served the new ones.
PERP_BAR_V1 = FeatureSpec(
    name="perp_bar",
    version=1,
    features=(
        # Return, at three horizons: the last bar, the last quarter-day, the trend.
        FeatureDef("ret_1", "log_return", params={"period": 1}, inputs={"price": "close"}),
        FeatureDef("mom_4", "momentum", params={"window": 4}, inputs={"price": "close"}),
        FeatureDef("mom_24", "momentum", params={"window": 24}, inputs={"price": "close"}),
        FeatureDef(
            "sma_x_6_24",
            "sma_crossover",
            params={"fast": 6, "slow": 24},
            inputs={"price": "close"},
        ),
        # Where the last close sits in its own recent distribution.
        FeatureDef("z_24", "rolling_zscore", params={"window": 24}, inputs={"x": "close"}),
        # Volatility, from closes and from the bar extremes. The two disagree exactly
        # when a bar round-trips, which is the state worth knowing about.
        FeatureDef(
            "vol_24", "realized_volatility", params={"window": 24}, inputs={"price": "close"}
        ),
        FeatureDef("range_bps", "relative_range"),
        # The closest thing to flow a candle feed can honestly supply.
        FeatureDef("clv", "close_location"),
        FeatureDef("vol_z_24", "rolling_zscore", params={"window": 24}, inputs={"x": "volume"}),
    ),
)

#: How many closed bars the serving path must hold for its last row to equal the
#: offline recompute exactly. The binding constraint is the longest window in the
#: spec (24) plus the one extra bar ``log_return`` consumes before it — 25 would be
#: arithmetically sufficient, and this is deliberately far more, because a buffer
#: sized to the exact minimum silently becomes wrong the day someone widens a
#: window. ``test_strategies.py`` asserts the equality rather than the arithmetic.
SERVING_BUFFER_BARS = 256

#: The forward horizon the model is fitted on, in bars: four hourly bars, so the
#: label is "does this close higher four hours from now". Stated here because a
#: strategy whose holding period does not resemble its label horizon is trading a
#: different question from the one the model was asked.
LABEL_HORIZON_BARS = 4


@dataclass(frozen=True)
class PerpBarParams:
    """Parameters for :class:`PerpBar`, in real units.

    ``entry_edge``/``exit_edge`` are distances from an even 50/50 probability, and
    they form a band rather than a single threshold for the reason
    :class:`~axon.strategy.reference.MeanReversionParams` gives: a probability
    sitting on one threshold flips the target on every bar, and each flip is a
    round trip paid in fees and rate-limit budget.
    """

    symbol_id: int
    max_position: Decimal = Decimal("0.01")
    entry_edge: float = 0.02
    exit_edge: float = 0.005
    #: 0 is the most passive. A four-hour opinion worth a handful of basis points
    #: cannot afford to cross the spread and pay the taker fee to express itself:
    #: at Hyperliquid's fee schedule the round trip costs more than the edge this
    #: model measures. A passive fill or no fill is the correct trade here.
    urgency: int = 0
    #: A signal-**admission** window, not an order lifetime. Nothing cancels an order
    #: on age: the runtime admits this record only while the core's event clock is
    #: within ``min(ttl_ms, intent.max_signal_age_ms)`` of its ``ts_event``
    #: (``SignalReader::effective_ttl_ns``), and ``Planner::plan`` never reads the
    #: field at all.
    #:
    #: One minute is the *opinion's* useful life — the price it was formed at goes
    #: stale long before the four-hour half-life does — but with the shipped ceiling
    #: (``IntentConfig::max_signal_age_ms`` = 2_000) the effective window is **two
    #: seconds**. Deploying ``perp_bar`` unmodified therefore drops any bar whose
    #: signal reaches the ring more than 2 s after the close: counted as
    #: ``SignalReject::Expired``, and not retried for an hour. An operator running this
    #: strategy must raise ``intent.max_signal_age_ms`` to at least this value.
    #:
    #: A real order lifetime would need a separate field the planner reads. ``ttl_ms``
    #: cannot serve, because the reader consumes it before the planner ever sees the
    #: record.
    ttl_ms: int = 60_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit_edge < self.entry_edge:
            raise ValueError(
                "need 0 <= exit_edge < entry_edge for hysteresis, got "
                f"exit_edge={self.exit_edge} entry_edge={self.entry_edge}"
            )
        if not 0.0 < self.entry_edge < 0.5:
            raise ValueError(
                f"entry_edge is a distance from an even 50/50 probability and must be in "
                f"(0, 0.5), got {self.entry_edge}"
            )
        if self.max_position <= 0:
            raise ValueError(f"max_position must be positive, got {self.max_position}")

    @property
    def long_at(self) -> float:
        """The probability at or above which the strategy is long."""
        return 0.5 + self.entry_edge

    @property
    def short_at(self) -> float:
        return 0.5 - self.entry_edge

    @classmethod
    def from_config(cls, config: StrategyConfig) -> "PerpBarParams":
        params = dict(config.params)
        if "max_position" in params:
            # Decimal(str(...)), never float(...): 0.01 has no exact float form, and
            # a size 0.010000000000000000208 wide is a size the venue rounds somewhere
            # we did not choose.
            params["max_position"] = Decimal(str(params["max_position"]))
        if not config.symbols:
            raise ValueError("StrategyConfig.symbols must name the symbol this strategy trades")
        return cls(symbol_id=config.symbols[0], **params)


class PerpBar(Strategy):
    """Long when the model says the next four hours close higher, short when lower.

    Construct it with a :class:`~axon.models.inference.Predictor` — anything that
    turns a float32 feature matrix into scores. In research that is the in-memory
    booster; in serving it is an artifact loaded from the registry through
    :meth:`from_registry`, which refuses a model whose recorded feature spec is not
    the one this build computes.
    """

    def __init__(
        self,
        params: PerpBarParams,
        predictor: Predictor,
        *,
        spec: FeatureSpec = PERP_BAR_V1,
        buffer_bars: int = SERVING_BUFFER_BARS,
        artifact_version: int | None = None,
    ) -> None:
        if buffer_bars < 2:
            raise ValueError(f"buffer_bars must hold at least two bars, got {buffer_bars}")
        _check_buffer_fits_spec(spec, buffer_bars)
        self.params = params
        self.predictor = predictor
        self.spec = spec
        self.buffer_bars = int(buffer_bars)
        #: The registry version of the model being served, for the caller to stamp
        #: on the run. The strategy never writes it onto a signal itself — the
        #: context owns that, so every emitted record carries the same one.
        self.artifact_version = artifact_version
        self._bars: deque[tuple[int, int, int, int, int]] = deque(maxlen=self.buffer_bars)
        self._target = Decimal(0)

    # ── construction from an artifact ────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        registry_id: str,
        params: PerpBarParams,
        *,
        version: int | None = None,
        spec: FeatureSpec = PERP_BAR_V1,
        **kwargs: Any,
    ) -> "PerpBar":
        """Load a model from a :class:`~axon.models.ModelRegistry` and bind it here.

        The spec check is the load-bearing line. An artifact records the exact
        recipe its model was fed (``name/vN#fingerprint``, ADR-0016); if that string
        is not the one this build computes, the columns have been reordered, a
        window has changed, or the transforms themselves have — and serving it
        anyway feeds the model numbers it has never seen, in a way that produces
        plausible probabilities and no error at all.
        """
        meta = registry.load_meta(registry_id, version)
        if meta.feature_spec_ref != spec.ref:
            raise ValueError(
                f"{registry_id}@{meta.version} was trained on features "
                f"{meta.feature_spec_ref!r} but this build computes {spec.ref!r}; the model "
                "would be served numbers it was never fitted on"
            )
        width = len(spec.columns)
        declared = meta.inputs[0].shape[-1] if meta.inputs and meta.inputs[0].shape else None
        if declared is not None and declared != width:
            raise ValueError(
                f"{registry_id}@{meta.version} takes {declared} features, the spec produces "
                f"{width}; the artifact and the spec disagree about the matrix"
            )
        return cls(
            params,
            registry.load_predictor(registry_id, meta.version),
            spec=spec,
            artifact_version=meta.version,
            **kwargs,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_reset(self) -> None:
        self._bars.clear()
        self._target = Decimal(0)

    def on_save(self) -> dict[str, Any]:
        # Bars are wire integers and stay integers; the target is a size and goes
        # out as a string, so a warm restart resumes on exactly the position it left
        # off on rather than trading the rounding difference.
        return {"bars": [list(b) for b in self._bars], "target": str(self._target)}

    def on_load(self, state: dict[str, Any]) -> None:
        self._bars = deque(
            (tuple(int(v) for v in bar) for bar in state.get("bars", ())),
            maxlen=self.buffer_bars,
        )
        self._target = Decimal(str(state.get("target", "0")))

    # ── the decision ─────────────────────────────────────────────────────────

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        if bar.symbol_id != self.params.symbol_id:
            return
        # Fixed-point in, and it stays fixed-point until `bar_inputs` converts it —
        # the same call the research path makes on a downloaded candle history.
        self._bars.append((bar.open, bar.high, bar.low, bar.close, bar.volume))

        row = self.feature_row()
        if row is None:
            return
        probability = self.probability(row)
        if probability is None:
            return

        target = self._next_target(probability)
        if target == self._target:
            # A target position is idempotent: re-sending it costs ring space and
            # rate-limit budget and tells the executor nothing new. Silence is only
            # safe because the runner heartbeats separately — without that, "no
            # opinion" and "dead process" would look identical.
            return
        self._target = target
        ctx.emit_target(
            bar.symbol_id,
            target,
            urgency=self.params.urgency,
            ttl_ms=self.params.ttl_ms,
        )

    # ── the pieces, exposed so the parity harness can drive them ─────────────

    def feature_row(self) -> np.ndarray | None:
        """The latest feature vector, or ``None`` while the window is unusable.

        Recomputes the spec over the whole buffer and returns its last row. That is
        the single-source-of-truth rule made concrete: the online vector is produced
        by the *same call* the offline matrix is, so the feature-parity gate is
        comparing the two paths and not two implementations.

        There is deliberately no warmup counter here. "How many bars before this is
        usable" is a property of the spec's windows, not a number this file should
        restate and get wrong when a window changes; the NaN warmup that every
        transform in :mod:`axon.features` guarantees answers it exactly.
        """
        if not self._bars:
            return None
        columns = np.asarray(self._bars, dtype=np.int64)
        inputs = bar_inputs(*(columns[:, j] for j in range(columns.shape[1])))
        row = self.spec.compute(inputs)[-1]
        if not np.isfinite(row).all():
            # A bar with no range, a zero price in the feed, a frozen window — each
            # makes at least one column NaN. Emitting nothing is right; emitting flat
            # would be an instruction to close a position for a feed glitch.
            return None
        return row

    def probability(self, row: np.ndarray) -> float | None:
        """``P(the next horizon closes higher)`` for one feature row.

        float32 in, because that is what the artifact was verified at and what a
        Rust backend will feed it (ADR-0005). Predicting at float64 measures a model
        that never runs.
        """
        scores = np.asarray(
            self.predictor.predict(np.asarray(row, dtype=np.float32).reshape(1, -1))
        )
        value = _probability(scores)
        if not np.isfinite(value):
            # A non-finite score is a broken model, not a 50/50 opinion, and
            # `emit_target` would refuse it one layer later with less context.
            return None
        return value

    def target_for(self, probability: float) -> Decimal:
        """The target this probability implies, given the position currently held."""
        return self._next_target(probability)

    def _next_target(self, probability: float) -> Decimal:
        p = self.params
        edge = probability - 0.5
        if edge >= p.entry_edge:
            return p.max_position
        if edge <= -p.entry_edge:
            return -p.max_position
        if abs(edge) <= p.exit_edge:
            return Decimal(0)
        return self._target  # inside the band: hold, do not thrash

    @property
    def target(self) -> Decimal:
        """The position the strategy currently wants, in real units."""
        return self._target


def _check_buffer_fits_spec(spec: FeatureSpec, buffer_bars: int) -> None:
    """Refuse a buffer too short for the spec's longest window.

    Worth a synthetic probe at construction because of how this fails otherwise: a
    buffer shorter than the longest window leaves the last row NaN *forever*, so
    :meth:`PerpBar.feature_row` returns ``None`` on every bar, the strategy emits
    nothing, and nothing anywhere raises. A strategy that never trades looks exactly
    like a strategy with no opinion, and the difference is only visible to someone
    who already suspects it.

    The probe is a strictly-varying ramp — every price positive, every bar with a
    real range, volume that moves — so any NaN in its last row is the window, not
    the data.
    """
    i = np.arange(int(buffer_bars), dtype=np.int64)
    close = 10**12 + (i % 7) * 10**9 + i * 10**8
    ramp = {
        "open": close - 10**8,
        "high": close + 2 * 10**8,
        "low": close - 2 * 10**8,
        "close": close,
        "volume": 10**9 + (i % 5) * 10**8,
    }
    row = spec.compute(bar_inputs(*(ramp[name] for name in BAR_INPUTS)))[-1]
    if not np.isfinite(row).all():
        blank = [c for c, v in zip(spec.columns, row) if not np.isfinite(v)]
        raise ValueError(
            f"a {buffer_bars}-bar buffer never warms up column(s) {blank} of spec "
            f"{spec.ref}; the strategy would emit nothing, forever, without an error"
        )


def _probability(scores: np.ndarray) -> float:
    """The positive-class probability out of whatever shape a backend returned.

    A tree booster hands back a flat vector of ``P(class 1)``; an ONNX classifier
    graph hands back both columns. Taking element zero of the second would be
    ``P(down)`` — a strategy that trades the exact inverse of its model, with no
    symptom other than losing money.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 2 and s.shape[1] == 2:
        return float(s[0, 1])
    return float(np.ravel(s)[0])


__all__ = [
    "LABEL_HORIZON_BARS",
    "PERP_BAR_V1",
    "SERVING_BUFFER_BARS",
    "PerpBar",
    "PerpBarParams",
]
