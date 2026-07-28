"""Distribution drift: PSI and KL divergence per feature, for the live monitor.

The parity gates answer "is the live path computing what research computed?". Drift
answers the question that survives a green gate: **the code is right and the world
moved.** A model fed features from a regime it never saw is not broken in any way a
comparison can detect, which is why ``docs/07`` puts PSI/KL next to the parity
monitor rather than inside it.

Population Stability Index, with the conventional bands:

===============  ==============================================================
``psi < 0.10``   stable — no action
``0.10–0.25``    moderate — investigate, watch the model's realized edge
``psi > 0.25``   significant — the input distribution has moved; retrain/disable
===============  ==============================================================

Three implementation choices that decide whether the number means anything:

* **Bins are quantiles of the reference sample, and they are frozen.** Equal-width
  bins over a fat-tailed return distribution put 99% of the mass in one bucket and
  PSI stops responding. Worse, *recomputing* quantile bins from the live sample makes
  both histograms uniform by construction, so PSI reads ~0 no matter what happened —
  the monitor would be structurally blind. :func:`quantile_binning` is therefore
  computed once at training time and carried forward.
* **The outer edges are infinite.** Live values outside the training range land in the
  end bins instead of being dropped, because "the feature now reaches values it never
  reached before" is the single most informative kind of drift.
* **Empty bins are floored, not skipped.** An empty bucket makes the log term infinite,
  which on small samples happens constantly and turns every alarm into ``inf``.

NaNs are excluded from the histograms and tracked separately: a feature that starts
emitting NaNs has drifted in a way no histogram of its finite values can show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from axon.parity.gate import ParityError, _raise_unless

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

#: Probability floor for an empty bin. Small enough not to move a real PSI, large
#: enough to keep the log finite.
_BIN_FLOOR = 1e-6

#: A NaN rate that jumps by more than this is reported even when PSI is calm.
DEFAULT_NAN_RATE_TOL = 0.01


def psi_band(value: float) -> str:
    """``"stable"`` / ``"moderate"`` / ``"significant"`` for a PSI value."""
    if not np.isfinite(value):
        return "significant"
    if value < PSI_STABLE:
        return "stable"
    if value <= PSI_MODERATE:
        return "moderate"
    return "significant"


@dataclass(frozen=True)
class Binning:
    """Frozen bin boundaries derived from a reference sample.

    ``cuts`` are the interior boundaries; a value falls in bin
    ``searchsorted(cuts, v, "right")``, so there are ``len(cuts) + 1`` bins with
    open ends. ``constant`` marks the degenerate case where the reference took a
    single value and has no quantile structure at all: bins are then "equal to it"
    and "anything else", which is the only honest split available.
    """

    cuts: np.ndarray
    constant: float | None = None

    @property
    def n_bins(self) -> int:
        return 2 if self.constant is not None else int(self.cuts.size) + 1

    def counts(self, values) -> np.ndarray:
        """Histogram ``values`` over these bins, ignoring non-finite entries."""
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if self.constant is not None:
            same = int(np.count_nonzero(x == self.constant))
            return np.array([same, x.size - same], dtype=np.int64)
        idx = np.searchsorted(self.cuts, x, side="right")
        return np.bincount(idx, minlength=self.cuts.size + 1).astype(np.int64)


def quantile_binning(reference, *, bins: int = 10) -> Binning:
    """Bin edges at the quantiles of ``reference`` — computed once, then reused."""
    if bins < 2:
        raise ValueError(f"need at least 2 bins, got {bins}")
    x = np.asarray(reference, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("cannot bin a reference sample with no finite values")
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return Binning(cuts=np.empty(0), constant=float(edges[0]))
    # Drop the outer quantiles: the end bins are open, so a live value below the
    # training minimum is counted rather than discarded.
    cuts = edges[1:-1]
    if cuts.size == 0:
        # Only two distinct quantile levels (a near-binary feature). Splitting at the
        # lower one keeps two bins; taking the empty interior would collapse to a
        # single bin, and a single-bin PSI is 0 no matter what the live data does.
        cuts = edges[:1]
    return Binning(cuts=cuts)


def _probabilities(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        raise ValueError("cannot form a distribution from an empty sample")
    return np.maximum(counts / total, _BIN_FLOOR)


def psi(expected, actual, *, bins: int = 10, binning: Binning | None = None) -> float:
    """Population Stability Index of ``actual`` against ``expected``.

    Symmetric by construction (``sum((p_a - p_e) * ln(p_a / p_e))``), which is why
    it is the industry's drift number rather than a KL in one direction.
    """
    b = binning if binning is not None else quantile_binning(expected, bins=bins)
    p_e = _probabilities(b.counts(expected))
    p_a = _probabilities(b.counts(actual))
    return float(np.sum((p_a - p_e) * np.log(p_a / p_e)))


def kl_divergence(expected, actual, *, bins: int = 10, binning: Binning | None = None) -> float:
    """``KL(actual ‖ expected)`` in nats, over the same bins as :func:`psi`.

    The direction is deliberate and worth remembering, because KL is asymmetric:
    this is the surprise of *today's* data under the distribution the model was
    trained on. The reverse direction answers a question nobody is asking at 03:00.
    """
    b = binning if binning is not None else quantile_binning(expected, bins=bins)
    p_e = _probabilities(b.counts(expected))
    p_a = _probabilities(b.counts(actual))
    return float(np.sum(p_a * np.log(p_a / p_e)))


@dataclass(frozen=True)
class FeatureDrift:
    """Drift of one feature between a reference window and a live window."""

    name: str
    psi: float
    band: str
    kl: float
    n_expected: int
    n_actual: int
    nan_rate_expected: float
    nan_rate_actual: float

    @property
    def nan_rate_delta(self) -> float:
        return self.nan_rate_actual - self.nan_rate_expected


@dataclass(frozen=True)
class DriftReport:
    """Per-feature drift across a feature matrix."""

    features: tuple[FeatureDrift, ...]
    nan_rate_tol: float

    def ranked(self) -> tuple[FeatureDrift, ...]:
        """Worst first, by PSI."""
        return tuple(sorted(self.features, key=lambda f: -f.psi))

    def significant(self) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features if f.band == "significant")

    def nan_regressions(self) -> tuple[FeatureDrift, ...]:
        """Features that started emitting NaNs — drift no histogram would show."""
        return tuple(f for f in self.features if f.nan_rate_delta > self.nan_rate_tol)

    @property
    def passed(self) -> bool:
        """No feature in the significant band and no new NaNs.

        Deliberately *not* the deploy gate — a moved market is not a bug. This is the
        alarm condition for the live monitor: something changed enough that the
        model's inputs are outside what it was fitted on.
        """
        return not self.significant() and not self.nan_regressions()

    def summary(self) -> str:
        lines = [f"drift {'OK' if self.passed else 'ALARM'}: {len(self.features)} feature(s)"]
        for f in self.ranked():
            note = ""
            if f.nan_rate_delta > self.nan_rate_tol:
                note = f"  nan_rate {f.nan_rate_expected:.3f} → {f.nan_rate_actual:.3f}"
            lines.append(
                f"  {f.name}: psi={f.psi:.4f} [{f.band}] kl={f.kl:.4f} "
                f"n={f.n_expected}/{f.n_actual}{note}"
            )
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`~axon.parity.gate.ParityError` unless nothing alarmed."""
        _raise_unless(self.passed, self.summary())


def drift_report(
    expected,
    actual,
    *,
    columns: Sequence[str],
    bins: int = 10,
    nan_rate_tol: float = DEFAULT_NAN_RATE_TOL,
    binnings: Sequence[Binning] | None = None,
) -> DriftReport:
    """PSI + KL per column of a feature matrix.

    ``expected`` is the reference window (typically the training sample) and
    ``actual`` the live one; they need not have the same number of rows. Pass
    ``binnings`` — the bins frozen at training time — whenever they exist; deriving
    them from ``expected`` here is only correct when ``expected`` *is* that sample.
    """
    ref = np.asarray(expected, dtype=np.float64)
    live = np.asarray(actual, dtype=np.float64)
    if ref.ndim != 2 or live.ndim != 2:
        raise ValueError(f"expected 2-D feature matrices, got {ref.shape} and {live.shape}")
    cols = tuple(columns)
    if ref.shape[1] != live.shape[1] or len(cols) != ref.shape[1]:
        raise ValueError(
            f"column mismatch: reference has {ref.shape[1]}, live has {live.shape[1]}, "
            f"{len(cols)} names given"
        )
    if binnings is not None and len(binnings) != len(cols):
        raise ValueError(f"{len(binnings)} binnings for {len(cols)} columns")

    drifts = []
    for j, name in enumerate(cols):
        ref_col, live_col = ref[:, j], live[:, j]
        ref_finite, live_finite = np.isfinite(ref_col), np.isfinite(live_col)
        try:
            b = binnings[j] if binnings is not None else quantile_binning(ref_col, bins=bins)
        except ValueError as e:
            raise ValueError(f"column {name!r}: {e}") from None
        if live_finite.any():
            value = psi(ref_col, live_col, binning=b)
            kl = kl_divergence(ref_col, live_col, binning=b)
        else:
            # Every live value is NaN: there is no distribution to compare, and
            # reporting 0.0 would say "stable" about a feature that has gone dark.
            value, kl = float("inf"), float("inf")
        drifts.append(
            FeatureDrift(
                name=name,
                psi=value,
                band=psi_band(value),
                kl=kl,
                n_expected=int(ref_finite.sum()),
                n_actual=int(live_finite.sum()),
                nan_rate_expected=float(1.0 - ref_finite.mean()) if ref_col.size else 0.0,
                nan_rate_actual=float(1.0 - live_finite.mean()) if live_col.size else 0.0,
            )
        )
    return DriftReport(features=tuple(drifts), nan_rate_tol=float(nan_rate_tol))


__all__ = [
    "Binning",
    "DEFAULT_NAN_RATE_TOL",
    "DriftReport",
    "FeatureDrift",
    "PSI_MODERATE",
    "PSI_STABLE",
    "ParityError",
    "drift_report",
    "kl_divergence",
    "psi",
    "psi_band",
    "quantile_binning",
]
