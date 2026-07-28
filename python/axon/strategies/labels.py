"""Labels and splits — the half of a research pipeline that decides what is true.

Features get the attention; the label and the split decide whether any number the
model produces means anything. Both of the classic ways to be wrong are silent:

**A label that peeks.** The target of row ``i`` is a forward return, so it is
computed from prices row ``i`` had not seen. That is correct and necessary — but it
means the label of a row near the end of a training block is a function of prices
that sit inside the *test* block. Train on it and the model has read its own exam
paper, and the only symptom is that the backtest looks good.

**A split that pretends rows are independent.** Overlapping forward returns are
autocorrelated by construction: with a four-bar horizon, four consecutive rows
share three quarters of their outcome. A shuffled K-fold puts near-copies of a test
row in the training set and reports an accuracy that has nothing to do with
tomorrow. Only a *forward* split with a purge tells you anything.

Nothing here is registered as a feature. Registration would make it available to a
:class:`~axon.features.FeatureSpec`, and the point-in-time sweep in
``test_features.py`` would (rightly) fail it — but a leak that is caught by a test
is still a leak that was written, and this is the one transform in the codebase
that is *supposed* to read the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axon.features import FeatureError


def forward_log_return(close, *, horizon: int) -> np.ndarray:
    """``log(c_{t+horizon} / c_t)`` — the outcome row ``t`` is asked to predict.

    The last ``horizon`` rows are NaN: their outcome has not happened. Filling them
    with zero would teach the model that the end of every sample is a flat market,
    and — worse — every walk-forward split puts the most recent rows in its most
    recent test window, so the fabricated flat period would be exactly the part
    reported as out-of-sample performance.
    """
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1:
        raise FeatureError(f"close must be 1-D, got shape {c.shape}")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise FeatureError(f"horizon must be an int >= 1 bar, got {horizon!r}")
    horizon = int(horizon)
    out = np.full(c.size, np.nan)
    if c.size > horizon:
        future, now = c[horizon:], c[:-horizon]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:-horizon] = np.where(
                (future > 0.0) & (now > 0.0), np.log(future / now), np.nan
            )
    return out


def direction_label(forward_return) -> np.ndarray:
    """``1`` where the forward return is positive, ``0`` where it is not, NaN kept.

    A sign label, not a cost-aware one. It says nothing about whether the move was
    big enough to pay for the round trip — that hurdle lives in the strategy's
    probability band, and keeping the two separate is a real weakness, stated in
    ADR-0022: the model is not trained on the objective it is traded on.

    Exactly zero counts as "not up". A perp print is discrete, so an exactly flat
    four-hour return is rare but not impossible, and a coin flip on the boundary
    would put irreproducible noise in the training set.
    """
    r = np.asarray(forward_return, dtype=np.float64)
    return np.where(np.isnan(r), np.nan, (r > 0.0).astype(np.float64))


@dataclass(frozen=True, eq=False)
class Fold:
    """One walk-forward fold: what may be trained on, and what it is judged against.

    The boundaries are carried in nanoseconds as well as row indices because the
    rows are pooled across coins — an index range is meaningless once two symbols
    are interleaved, and every leakage question is a question about *time*.
    """

    index: int
    train: np.ndarray
    test: np.ndarray
    train_end_ns: int
    test_start_ns: int
    test_end_ns: int

    def __len__(self) -> int:
        return int(self.test.size)

    def __repr__(self) -> str:
        return (
            f"Fold({self.index}: train={self.train.size} rows < {self.train_end_ns}, "
            f"test={self.test.size} rows in [{self.test_start_ns}, {self.test_end_ns}))"
        )


def purged_walk_forward(
    ts_event,
    *,
    folds: int = 4,
    horizon_ns: int,
    embargo_ns: int = 0,
) -> tuple[Fold, ...]:
    """Expanding-window walk-forward over event time, with the label overlap purged.

    The test blocks partition the tail of the sample into ``folds`` equal spans of
    *distinct event times* (equal spans of rows would give the fold covering a
    busier symbol set more calendar time than its neighbour). Fold ``k`` trains on
    everything strictly before ``test_start - horizon - embargo``.

    The purge is the whole point. A row's label reads ``horizon`` into the future,
    so a training row within one horizon of the test boundary is labelled with
    prices from inside the test window; without the subtraction the model is
    trained on part of the answer and the out-of-sample number is fiction. The
    embargo takes out a further span on top, because the features either side of
    the cut are computed from overlapping windows and stay correlated after the
    labels no longer overlap.

    Rows are pooled across symbols, so the split is defined on timestamps and never
    on row position: with two coins interleaved, an index range holds one coin's
    future and the other's past.
    """
    ts = np.asarray(ts_event, dtype=np.int64)
    if ts.ndim != 1 or ts.size == 0:
        raise FeatureError(f"ts_event must be a non-empty 1-D int64 array, got shape {ts.shape}")
    if folds < 1:
        raise FeatureError(f"need at least one fold, got {folds}")
    if horizon_ns < 0 or embargo_ns < 0:
        raise FeatureError("horizon_ns and embargo_ns are durations and cannot be negative")

    times = np.unique(ts)
    if times.size < folds + 1:
        raise FeatureError(
            f"{times.size} distinct event times cannot be cut into {folds} test blocks "
            "plus an initial training block"
        )
    # The first block is training-only; the remaining `folds` blocks are tested in
    # turn. Anything else would test on the very first bars, which no model has yet
    # had a chance to be fitted on.
    edges = np.linspace(0, times.size, folds + 2).astype(int)[1:]

    out: list[Fold] = []
    for k in range(folds):
        start_ns = int(times[edges[k]])
        end_ns = int(times[edges[k + 1] - 1]) + 1
        train_end = start_ns - int(horizon_ns) - int(embargo_ns)
        train = np.flatnonzero(ts < train_end)
        test = np.flatnonzero((ts >= start_ns) & (ts < end_ns))
        if train.size == 0 or test.size == 0:
            # A fold with an empty side is not a weak fold, it is a missing one, and
            # averaging over it silently would report a mean of fewer folds than the
            # caller asked for.
            raise FeatureError(
                f"fold {k} has {train.size} training and {test.size} test rows; the sample "
                "is too short for this many folds at this horizon"
            )
        out.append(
            Fold(
                index=k,
                train=train,
                test=test,
                train_end_ns=train_end,
                test_start_ns=start_ns,
                test_end_ns=end_ns,
            )
        )
    return tuple(out)


__all__ = ["Fold", "direction_label", "forward_log_return", "purged_walk_forward"]
