"""``baseline`` — the member of the zoo with no model in it at all.

Every other strategy in :mod:`axon.strategies` is a *model* plus a serving path.
This one is the serving path with the model removed: on every **closed** bar it
recomputes a two-column feature matrix over its own buffer, reads a rolling
z-score straight off it, and turns the sign of that number into a target position
with a hysteresis band. There is no artifact, no registry entry, no predictor, no
parity bundle and no export. Nothing here can be re-fitted, because nothing here
was fitted.

That absence is the whole point, and it buys three things nothing else in the zoo
can buy:

**It separates the bridge from the model.** ``features → decision → Signal → the
ring`` is one path with two halves, and when a live session says nothing there is
no way to tell which half is silent. Run this beside an ML strategy on the same
feed: if this one emits and the model one does not, the fault is the model. If
neither emits, the fault is the bridge. The distinction costs one file and no
training run, and it is the only way to get it without a debugger attached to a
live process.

**It measures what the pipeline actually requires.** Serving is *supposed* to
accept any object that satisfies :class:`~axon.strategy.base.Strategy`. Whether it
really does is an empirical question, and the only way to ask it is to hand the
pipeline something with nothing to load. Two places said no, and both are recorded
here rather than worked around silently: :data:`NO_MODEL_VERSION` for the wire, and
the module note below for the Rust session config.

**It is the one strategy that can be right when every other one is broken.** A
model that fails to export, fails a gate, or returns a non-finite score takes its
strategy off the air. This rule has no step that can fail that way — a bar with no
range and a frozen tape are refusals it already makes, and there is nothing else in
it to break.

The rest of the design is deliberately *not* novel, because a novel baseline is not
a baseline:

**Finite lookback, and only finite lookback.** No EMA, no expanding statistic. An
EMA never forgets its seed, so a serving path holding a bounded buffer computes a
different number from the research path that saw the whole history —
``test_features.py``'s
``test_a_finite_window_crossover_survives_a_bounded_serving_buffer_where_an_ema_does_not``
is that claim as arithmetic. With only windowed transforms, a buffer of
:data:`SERVING_BUFFER_BARS` reproduces the offline matrix *bit for bit*, and the
feature-parity gate compares transforms rather than histories. It applies to a
statistical rule exactly as it applies to a model: this file has no artifact to
protect, but the recompute it is diffed against is the same recompute.

**Twenty bars, and the interval is the operator's choice.** The spec's warmup is
**21 bars** — 20 for the z-score, 21 for the volatility column that needs a return
before it needs a window. That is *measured* by :func:`warmup_bars` rather than
restated in a comment, because a restated number goes stale the day a window
changes and nothing notices. Twenty-one bars is **21 minutes on m1** and **21 hours
on 1h**, which is the difference between a session that has an opinion before lunch
and one that does not have an opinion at all. Use m1 or m5.

**The direction was not chosen by trying both.** The rule reverts — long when the
close is cheap against its own rolling mean — because that is the direction
:class:`~axon.strategy.reference.MeanReversion` already picked when Phase 4 needed
a strategy to exercise the seam. Testing both signs on the sample and shipping the
better one is a one-bit fit over the whole history, and it is exactly the thing a
baseline exists not to do. The thresholds (1.5 in, 0.5 out) are that class's
defaults, carried across unchanged.

**The point is not to make money.** :func:`evaluate_baseline` reports the rule's
gross edge next to the free benchmark and next to the fees its own emitted target
path would have paid, and the subtraction is left visible. See ADR-0022 for why a
green gate over a losing rule is a *success* here, and :meth:`BaselineVerdict.describe`
for what this one actually did.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from axon.contracts import FIXED_POINT_SCALE
from axon.features import BAR_INPUTS, FeatureDef, FeatureSpec, bar_inputs
from axon.strategies.data import INTERVAL_MS, Candles
from axon.strategies.labels import forward_log_return
from axon.strategy.base import Strategy
from axon.strategy.config import StrategyConfig
from axon.strategy.context import URGENCY_POST_ONLY, StrategyContext
from axon.strategy.events import Bar

#: The column the decision is read out of, and the column that says whether the
#: decision meant anything. Named by *role* rather than by window length — the
#: window lives in the spec's parameters and in its fingerprint, which is the only
#: place it can go stale in exactly one direction. A column called ``z_20`` in a
#: five-bar spec is a label that lies, and nothing checks a label.
Z_COLUMN = "z"
VOL_COLUMN = "vol"

#: The window every column in the spec is measured over, in bars.
#:
#: One number rather than two, because a baseline with two tunables is a baseline
#: someone will tune. Twenty is long enough that a rolling standard deviation is not
#: dominated by a single print and short enough that :func:`warmup_bars` stays inside
#: half an hour on m1 — see the module docstring for the arithmetic that matters.
BASELINE_WINDOW = 20


def baseline_spec(window: int = BASELINE_WINDOW, *, version: int = 1) -> FeatureSpec:
    """The rule's feature recipe, at a given window length.

    A constructor rather than a single frozen constant because the window is the only
    thing about this spec anyone would sensibly vary, and it belongs *in* the spec:
    :attr:`FeatureSpec.ref` folds it into the fingerprint, so a 20-bar rule and a
    5-bar rule are different recipes with different identities rather than the same
    recipe run with a different argument nobody recorded.

    Two columns, and both are read:

    :data:`Z_COLUMN` is the opinion — how many trailing standard deviations the last
    close sits from its own rolling mean.

    :data:`VOL_COLUMN` is the refusal. A z-score is a ratio, and a ratio computed on a
    tape that barely moved is a large number about nothing: the classic warmup
    failure, where the strategy takes its largest position on its worst information.
    The z-score itself already goes NaN on a *perfectly* flat window
    (:func:`~axon.features.rolling_zscore` refuses a zero denominator), but a window
    that stepped once and then froze has a non-zero price deviation and no realized
    volatility at all, and that window still produces a finite, confident, meaningless
    z. Requiring realized volatility to be strictly positive is the check that covers
    it, and it is a floor rather than a threshold — there is no number in it to fit.
    """
    return FeatureSpec(
        name="baseline_z",
        version=version,
        features=(
            FeatureDef(
                Z_COLUMN, "rolling_zscore", params={"window": window}, inputs={"x": "close"}
            ),
            FeatureDef(
                VOL_COLUMN,
                "realized_volatility",
                params={"window": window},
                inputs={"price": "close"},
            ),
        ),
    )


#: The shipped recipe: a 20-bar z-score of the close, and the 20-bar realized
#: volatility that says whether the denominator meant anything.
BASELINE_Z_V1 = baseline_spec()

#: How many closed bars the serving path holds. The binding constraint is the
#: warmup (:func:`warmup_bars` measures 21 for :data:`BASELINE_Z_V1`); this is
#: deliberately several times that, because a buffer sized to the exact minimum
#: silently becomes wrong the day someone widens a window, and the symptom is a
#: strategy that emits nothing forever with no error anywhere.
SERVING_BUFFER_BARS = 128

#: What a no-model run stamps in the signal's ``model_version`` field.
#:
#: **The wire has no way to say "there is no model".** ``model_version`` is a
#: mandatory ``u32`` on every :data:`~axon.contracts.SIGNAL_DTYPE` record, and
#: :class:`~axon.strategy.context.StrategyContext` refuses ``0`` for a good reason
#: it states itself: zero is the value of a field nobody wrote, so allowing it makes
#: "unstamped" and "model 0" indistinguishable in an audit. That argument is right,
#: and it leaves a rule with no artifact obliged to write *some* version number for a
#: model that does not exist.
#:
#: Stamping ``1`` would be the quiet choice, and it is the wrong one: a capture of a
#: no-model session and a capture of a session serving the first registered model
#: would then be byte-identical in the one field that is supposed to say which model
#: made the decision. ``u32::MAX`` is conspicuous, cannot collide with a registry
#: version anyone will reach, and is a value a reader who has never seen this file
#: will look up rather than assume. Nothing in the Rust runtime reads the field, so
#: this costs nothing and buys a legible capture.
#:
#: The honest fix is a flag in the record that says "no artifact", and that is a
#: schema change with an ADR behind it — not something one strategy file may do.
NO_MODEL_VERSION = 0xFFFF_FFFF

# ── what the Rust session config still demands, and what to put there ────────
#
# `RuntimeConfig` refuses to load a session whose `[strategy]` table has no
# `model_ref`: `axon --check` on a config with the section deleted exits 1 with
# "missing field `model_ref` in `strategy`". The field is a `ModelRef { registry_id,
# version }` with no `#[serde(default)]` and no `Option`, and — measured by grep over
# `crates/` — *nothing in the runtime ever reads it*. So a no-model session starts by
# naming a registry entry that does not exist and never will, and the runtime is
# perfectly happy. An empty `registry_id = ""` is accepted too, which is at least not
# a lie; neither spelling is checked against anything.
#
# This is not worked around here because it cannot be: it is a Rust config schema, and
# a Python strategy has no say in it. It is written down so the next operator does not
# spend an afternoon looking for the registry the config appears to require.


@dataclass(frozen=True)
class BaselineParams:
    """Parameters for :class:`Baseline`, in real units.

    Every default here is either inherited from an existing strategy in this repo or
    derived from a venue limit. None of them was chosen by looking at a result — see
    the module docstring.
    """

    symbol_id: int
    #: A position size, so ``Decimal`` and never ``float``.
    #:
    #: **0.0003 BTC is about $19 at the $64k the cached history ends on**, which sits
    #: inside Hyperliquid's $10 minimum notional and the Phase 6 brief's $50 ceiling,
    #: and lands on the 0.00001 lot (``szDecimals`` 5 on testnet BTC). The band holds
    #: for any BTC price between roughly $33k and $167k, which is wide enough not to
    #: need re-deriving every week and narrow enough that it *does* need re-deriving.
    #:
    #: Not :class:`~axon.strategies.perp_bar.PerpBarParams`'s 0.01, deliberately: that
    #: is ~$640 of BTC at the same price, thirteen times the brief's ceiling. A default
    #: that breaches the session limit is a default that gets copied into a live config
    #: by someone who trusted it.
    max_position: Decimal = Decimal("0.0003")
    #: Distance from the mean, in trailing standard deviations, at which a position
    #: goes on. 1.5/0.5 are :class:`~axon.strategy.reference.MeanReversionParams`'
    #: defaults, unchanged.
    entry_z: float = 1.5
    #: …and the distance at which it comes off. A *band* rather than one threshold:
    #: with a single level, a z-score sitting on it flips the target on every bar, and
    #: every flip is a round trip paid in fees and rate-limit budget.
    exit_z: float = 0.5
    #: 0 is the most passive. A mean-reversion opinion worth a few basis points cannot
    #: afford to cross the spread and pay the taker fee to express itself; the
    #: arithmetic is in :meth:`BaselineVerdict.describe` and it is not close.
    urgency: int = URGENCY_POST_ONLY
    #: A signal **admission** window, not an order lifetime — the reader consumes it
    #: before the planner sees the record, clamped to
    #: ``min(ttl_ms, intent.max_signal_age_ms)`` with the shipped ceiling at 2 000 ms.
    #: :class:`~axon.strategies.perp_bar.PerpBarParams` states the trap in full; the
    #: same warning applies here verbatim. On m1 bars a minute is exactly one bar,
    #: which is the longest this opinion is worth anything anyway.
    ttl_ms: int = 60_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit_z < self.entry_z:
            raise ValueError(
                "need 0 <= exit_z < entry_z for hysteresis, got "
                f"exit_z={self.exit_z} entry_z={self.entry_z}"
            )
        if self.max_position <= 0:
            raise ValueError(f"max_position must be positive, got {self.max_position}")

    @classmethod
    def from_config(cls, config: StrategyConfig) -> "BaselineParams":
        params = dict(config.params)
        if "max_position" in params:
            # Decimal(str(...)), never float(...): 0.0003 has no exact float form, and
            # a size 0.00030000000000000003 wide is a size the venue rounds somewhere
            # nobody chose.
            params["max_position"] = Decimal(str(params["max_position"]))
        if not config.symbols:
            raise ValueError("StrategyConfig.symbols must name the symbol this strategy trades")
        return cls(symbol_id=config.symbols[0], **params)


class Baseline(Strategy):
    """Long when the close is cheap against its own rolling mean, short when rich.

    Constructed with parameters and nothing else. There is no ``from_registry``
    here and no ``predictor`` attribute, and their absence is load-bearing: an
    object that cannot be handed an artifact cannot be silently handed the *wrong*
    artifact, which is the failure
    :meth:`~axon.strategies.perp_bar.PerpBar.from_registry` spends fifteen lines
    guarding against.

    The surface is otherwise the same as :class:`~axon.strategies.perp_bar.PerpBar`
    — ``on_bar``, ``feature_row``, ``target``, ``spec``, ``params`` — so the same
    harnesses drive it.
    """

    def __init__(
        self,
        params: BaselineParams,
        *,
        spec: FeatureSpec = BASELINE_Z_V1,
        buffer_bars: int = SERVING_BUFFER_BARS,
    ) -> None:
        missing = [c for c in (Z_COLUMN, VOL_COLUMN) if c not in spec.columns]
        if missing:
            raise ValueError(
                f"spec {spec.ref} has columns {spec.columns} and is missing {missing}; "
                "this strategy reads its decision by column name, which is the whole "
                "reason it can refuse a spec instead of trading the wrong column"
            )
        # By name, resolved once. Reading column 0 by position would keep working, and
        # keep meaning something else, the day a column is inserted in front of it.
        self._z = spec.columns.index(Z_COLUMN)
        self._vol = spec.columns.index(VOL_COLUMN)
        warmup = warmup_bars(spec, probe_bars=buffer_bars)
        if warmup is None:
            raise ValueError(
                f"a {buffer_bars}-bar buffer never warms spec {spec.ref} up: the last row "
                "stays NaN forever, so the strategy emits nothing, forever, and nothing "
                "raises. A strategy that never trades looks exactly like a strategy with "
                "no opinion"
            )
        self.params = params
        self.spec = spec
        self.buffer_bars = int(buffer_bars)
        #: Bars before the first usable feature row. Measured at construction over a
        #: synthetic ramp, not restated from the window lengths — see :func:`warmup_bars`.
        self.warmup_bars = warmup
        #: There is no artifact, so there is no version of one. ``None`` rather than a
        #: placeholder integer: a caller stamping a run from this field must be made to
        #: notice that there is nothing to stamp (see :data:`NO_MODEL_VERSION`).
        self.artifact_version: int | None = None
        self._bars: deque[tuple[int, int, int, int, int]] = deque(maxlen=self.buffer_bars)
        self._target = Decimal(0)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_reset(self) -> None:
        self._bars.clear()
        self._target = Decimal(0)

    def on_save(self) -> dict[str, Any]:
        # Bars are wire integers and stay integers; the target is a size and goes out
        # as a string, so a warm restart resumes on exactly the position it left off
        # on rather than immediately trading the rounding difference.
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

        target = self._next_target(float(row[self._z]))
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

    # ── the pieces, exposed so a parity harness can drive them ───────────────

    def feature_row(self) -> np.ndarray | None:
        """The latest feature vector, or ``None`` while the window is unusable.

        Recomputes the spec over the whole buffer and returns its last row — the
        single-source-of-truth rule made concrete. The online vector is produced by
        the *same call* the offline matrix is, so a feature-parity gate over this
        strategy compares two paths rather than two implementations, exactly as it
        does for a model.

        Recomputing the whole buffer every bar rather than updating incrementally is
        the same trade :class:`~axon.strategies.perp_bar.PerpBar` makes: an
        incremental update would be a second implementation of every transform in
        :mod:`axon.features`, and two columns over 128 rows costs microseconds
        nobody has to account for.

        There is deliberately no warmup counter consulted here. "How many bars before
        this is usable" is a property of the spec's windows, and the NaN warmup every
        transform in :mod:`axon.features` guarantees answers it exactly.
        """
        if not self._bars:
            return None
        columns = np.asarray(self._bars, dtype=np.int64)
        inputs = bar_inputs(*(columns[:, j] for j in range(columns.shape[1])))
        row = self.spec.compute(inputs)[-1]
        if not np.isfinite(row).all():
            # Still warming, or a bar with no range, or a zero price in the feed.
            # Emitting nothing is right; emitting flat would be an instruction to
            # close a position because of a feed glitch.
            return None
        if row[self._vol] <= 0.0:
            # Finite but zero realized volatility: a tape that stepped once and then
            # froze. The z-score is finite and large here and means nothing at all —
            # see `baseline_spec`.
            return None
        return row

    def target_for(self, z: float) -> Decimal:
        """The target this z-score implies, given the position currently held."""
        return self._next_target(z)

    def _next_target(self, z: float) -> Decimal:
        p = self.params
        if z <= -p.entry_z:
            return p.max_position  # cheap versus the mean → long
        if z >= p.entry_z:
            return -p.max_position
        if abs(z) <= p.exit_z:
            return Decimal(0)
        return self._target  # inside the band: hold, do not thrash

    @property
    def target(self) -> Decimal:
        """The position the strategy currently wants, in real units."""
        return self._target


# ── warmup, measured ─────────────────────────────────────────────────────────


def warmup_bars(spec: FeatureSpec, *, probe_bars: int = 4096) -> int | None:
    """Bars before ``spec``'s last row is finite, or ``None`` if it never is.

    Measured, not restated. "This spec needs 21 bars" written in a comment is a
    number that stops being true the moment a window changes, and the failure is
    silent: the buffer is one bar short, every row stays NaN, the strategy emits
    nothing, and nothing anywhere raises.

    The probe is a strictly-varying ramp — every price positive, every bar with a
    real range, volume that moves — so any NaN in a row is the window and not the
    data. It also means the answer is a property of the *spec*, which is what the
    caller is actually asking about.
    """
    n = int(probe_bars)
    if n < 1:
        raise ValueError(f"probe_bars must be at least one bar, got {probe_bars}")
    i = np.arange(n, dtype=np.int64)
    close = 10**12 + (i % 7) * 10**9 + i * 10**8
    ramp = {
        "open": close - 10**8,
        "high": close + 2 * 10**8,
        "low": close - 2 * 10**8,
        "close": close,
        "volume": 10**9 + (i % 5) * 10**8,
    }
    matrix = spec.compute(bar_inputs(*(ramp[name] for name in BAR_INPUTS)))
    usable = np.isfinite(matrix).all(axis=1)
    if not usable.any():
        return None
    return int(np.argmax(usable)) + 1


def warmup_minutes(spec: FeatureSpec = BASELINE_Z_V1, interval: str = "1m") -> float:
    """The same warmup in wall-clock minutes, which is what decides an interval.

    The arithmetic the Phase 6 brief insists on doing *before* choosing a bar length:
    the identical spec is 21 minutes of silence on ``1m`` and 21 hours of it on
    ``1h``, and a session that does not outlive its own warmup never says anything at
    all. Wall clock is legitimate here for the same narrow reason a dead-man's-switch
    deadline is — this is a duration, not an ordering.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval!r}; known: {sorted(INTERVAL_MS)}")
    bars = warmup_bars(spec)
    if bars is None:
        raise ValueError(f"spec {spec.ref} never warms up; there is no wall-clock answer")
    return bars * INTERVAL_MS[interval] / 60_000.0


# ── the honest verdict ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BaselineVerdict:
    """What the rule did over a recorded history, and what that is worth.

    A run of :func:`evaluate_baseline`, which drives the **real strategy object**
    over real bars through a real :class:`~axon.strategy.context.StrategyContext`.
    Every number below is read off the records that came out of that context, not
    re-derived from the rule — which is what lets this report do the one thing
    :mod:`axon.strategies.training` deliberately refuses to do.

    ``training.py`` reports costs as a hurdle and never nets them off, because
    netting needs a turnover *model* and a turnover model living beside the evaluator
    would be a second implementation of the strategy's own hysteresis. That objection
    does not apply here: the turnover is not modelled, it is **counted**, off the
    emitted signal records. So the subtraction is made, and it is made in the open.
    """

    coin: str
    interval: str
    bars: int
    warmup_bars: int
    #: Bars entered holding a non-zero target, with a next bar to earn a return in.
    #: Every bps figure below is a mean over exactly these.
    held_bars: int
    #: Records the strategy actually emitted. Each is a target change.
    signals: int
    #: Position changes in units of ``max_position``: flat→long is one side,
    #: long→short is two. Fees are charged per side, which is why sides is the unit.
    sides: float
    #: Mean signed next-bar log return over the held bars, in basis points.
    edge_bps: float
    #: Mean *unsigned* next-bar return over the same rows: the market's drift under
    #: the rule's own feet.
    drift_bps: float
    maker_fee_bps: float
    taker_fee_bps: float

    @property
    def benchmark_edge_bps(self) -> float:
        """A constant short held on exactly the rule's own decision rows.

        Identically ``-drift_bps``, and identical in construction to
        :attr:`axon.strategies.training.Decomposition.benchmark_edge_bps`: a
        benchmark that needs no model, no fit, no features and no code, and the
        number an edge has to beat before anything else about it is interesting.
        """
        return -self.drift_bps

    @property
    def selection_bps(self) -> float:
        """Edge minus the free benchmark. What the rule's *choices* were worth."""
        return self.edge_bps - self.benchmark_edge_bps

    @property
    def bars_per_side(self) -> float:
        """Held bars per position change — how often the rule changes its mind."""
        return self.held_bars / self.sides if self.sides else float("inf")

    def drag_bps(self, fee_bps: float) -> float:
        """Fees the emitted path would have paid, spread over the held bars.

        In the same units as :attr:`edge_bps` — basis points per held bar of one
        ``max_position`` notional — so the two can be subtracted without a second
        assumption creeping in between them.
        """
        if not self.held_bars:
            return float("nan")
        return self.sides * fee_bps / self.held_bars

    def net_bps(self, fee_bps: float) -> float:
        return self.edge_bps - self.drag_bps(fee_bps)

    @property
    def notional_turned_over(self) -> float:
        """Total fees as a fraction of one position's notional, at maker rates.

        The form the handoff quotes ``perp_bar``'s turnover in ("36 % of notional in
        maker fees over 208 days"), so the two are comparable at a glance.
        """
        return self.sides * self.maker_fee_bps / 10_000.0

    def describe(self) -> str:
        span_days = self.bars * INTERVAL_MS[self.interval] / 86_400_000.0
        return "\n".join(
            [
                f"baseline_z {self.coin} {self.interval}: {self.bars} bars "
                f"(~{span_days:.0f} days), warmup {self.warmup_bars}, "
                f"{self.held_bars} held bar(s)",
                f"  edge      = {self.edge_bps:+.2f} bps / held bar",
                f"  always short, same rows: {self.benchmark_edge_bps:+.2f} bps "
                f"(drift {self.drift_bps:+.2f} bps)",
                f"  selection = edge - benchmark = {self.selection_bps:+.2f} bps",
                f"  turnover  = {self.signals} signal(s), {self.sides:.0f} side(s), "
                f"one mind change every {self.bars_per_side:.1f} held bars",
                f"  fees      = {self.drag_bps(self.maker_fee_bps):.2f} bps/bar maker, "
                f"{self.drag_bps(self.taker_fee_bps):.2f} bps/bar taker "
                f"({100 * self.notional_turned_over:.1f} % of notional in maker fees "
                f"over the sample)",
                f"  net       = {self.net_bps(self.maker_fee_bps):+.2f} bps/bar maker, "
                f"{self.net_bps(self.taker_fee_bps):+.2f} bps/bar taker "
                "— before spread, slippage and funding",
                "  This is a rule, not a model: nothing in it was fitted on this sample, "
                "so there is no in-sample number to discount. It is also not a P&L — "
                "nothing was filled, and a post-only target that never gets a fill earns "
                "neither the edge nor the fee.",
            ]
        )


def evaluate_baseline(
    candles: Candles,
    params: BaselineParams | None = None,
    *,
    spec: FeatureSpec = BASELINE_Z_V1,
    symbol_id: int = 1,
) -> BaselineVerdict:
    """Drive the real strategy over a real candle history and price what it did.

    The loop is the serving loop: a :class:`~axon.strategy.events.Bar` per closed
    candle, dispatched inside an event scope bound to that bar's own close time,
    with the emitted records taken off the context afterwards. Nothing here
    reimplements the rule, so the turnover this reports is the turnover the ring
    would have carried.

    ``model_version`` on the context is :data:`NO_MODEL_VERSION`, because even an
    offline evaluation has to answer the question the wire asks.
    """
    # Function-local, so the serving path does not import the research pipeline to
    # start a session — but imported rather than re-declared, because a second copy
    # of a venue's fee schedule is a schedule that goes stale in one of two places.
    from axon.strategies.training import MAKER_FEE_BPS, TAKER_FEE_BPS

    p = params if params is not None else BaselineParams(symbol_id=symbol_id)
    strategy = Baseline(p, spec=spec)
    ctx = StrategyContext(model_version=NO_MODEL_VERSION)

    held: list[Decimal] = []
    records: list[np.ndarray] = []
    for i in range(len(candles)):
        bar = Bar(
            symbol_id=p.symbol_id,
            ts_event=int(candles.ts_event[i]),
            open=int(candles.open[i]),
            high=int(candles.high[i]),
            low=int(candles.low[i]),
            close=int(candles.close[i]),
            volume=int(candles.volume[i]),
        )
        with ctx.event(bar.ts_event):
            strategy.on_bar(bar, ctx)
        records.extend(ctx.take_pending())
        held.append(strategy.target)

    # The return a position taken at bar t's close earns by bar t+1's close. The
    # repo's own forward return at horizon 1, not a second `np.diff(np.log(...))`.
    close = candles.feature_inputs()["close"]
    forward = forward_log_return(close, horizon=1)
    direction = np.array([0 if t == 0 else (1 if t > 0 else -1) for t in held], dtype=np.float64)
    taken = (direction != 0) & np.isfinite(forward)

    # Sides off the emitted records, exactly as `shadow.Turnover` counts them: the
    # wire record is the artifact of record, and a count taken from `held` would be
    # counting an intention rather than an instruction.
    position = Decimal(0)
    sides = Decimal(0)
    for rec in records:
        target = Decimal(int(rec["target_qty"])) / Decimal(FIXED_POINT_SCALE)
        sides += abs(target - position) / p.max_position
        position = target

    signed = direction[taken] * forward[taken]
    return BaselineVerdict(
        coin=candles.coin,
        interval=candles.interval,
        bars=len(candles),
        warmup_bars=strategy.warmup_bars,
        held_bars=int(taken.sum()),
        signals=len(records),
        sides=float(sides),
        edge_bps=float(np.mean(signed) * 10_000.0) if signed.size else float("nan"),
        drift_bps=float(np.mean(forward[taken]) * 10_000.0) if signed.size else float("nan"),
        maker_fee_bps=MAKER_FEE_BPS,
        taker_fee_bps=TAKER_FEE_BPS,
    )


__all__ = [
    "BASELINE_WINDOW",
    "BASELINE_Z_V1",
    "NO_MODEL_VERSION",
    "SERVING_BUFFER_BARS",
    "Baseline",
    "BaselineParams",
    "BaselineVerdict",
    "baseline_spec",
    "evaluate_baseline",
    "warmup_bars",
    "warmup_minutes",
]
