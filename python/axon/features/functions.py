"""The transforms themselves — plain functions over 1-D arrays.

Three rules hold for **every** function here, and the point-in-time test in
``python/tests/test_features.py`` enforces them across the whole registry rather
than function by function, so a feature added later cannot quietly opt out.

1. **Length-preserving.** ``f(x)`` has ``len(x)`` elements and element ``i`` lines
   up with observation ``i``. A transform that returns a shorter array forces the
   caller to re-align it against timestamps, and every place that alignment is
   done by hand is a place a feature ends up one row ahead of its own event — the
   "windowing off-by-one" that ``docs/03`` lists among the silent killers.
2. **Causal.** Element ``i`` is a function of ``x[:i + 1]`` and nothing else.
   Appending future observations must not change a single historical value. This
   is the difference between a backtest that is optimistic and one that is real.
3. **Warmup is NaN.** Not zero, not the seed value, not back-filled. Zero is a
   legal value for every feature here, so a zero warmup is indistinguishable from
   a genuine reading and the model learns from it.

**Float, deliberately.** Prices arrive as fixed-point integers and money math stays
in ``Decimal`` (see ``axon.strategy.context``), but a z-score is a statistic, not
money: it has no venue precision to respect and it is fed to a model that works in
float32/64 anyway. The conversion from the wire happens exactly once, in
:mod:`axon.features.inputs`, and never flows back into an order size.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from axon.features.registry import FeatureError, register

#: Bumped by hand whenever the *numerical meaning* of any transform here changes.
#: It is folded into every :class:`~axon.features.spec.FeatureSpec` fingerprint, so
#: an artifact trained before the change refuses to load against the library after
#: it. Without this the fingerprint would pin the recipe and say nothing about the
#: kitchen: rewriting ``rolling_std`` under a spec's feet leaves every id unchanged.
FEATURES_VERSION = 1


def _series(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise FeatureError(f"{name} must be a 1-D array, got shape {a.shape}")
    return a


def _aligned(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.size != b.size:
        raise FeatureError(
            f"{name_a} and {name_b} must be the same length, got {a.size} and {b.size}; "
            "two market-data columns of different lengths are not describing the same events"
        )


def _check_window(window: int, *, name: str = "window", minimum: int = 1) -> int:
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise FeatureError(f"{name} must be an int, got {type(window).__name__}")
    window = int(window)
    if window < minimum:
        raise FeatureError(f"{name} must be >= {minimum}, got {window}")
    return window


def _windows(a: np.ndarray, window: int) -> np.ndarray | None:
    """Rolling windows ending at each position, or ``None`` if the series is shorter."""
    if a.size < window:
        return None
    return sliding_window_view(a, window)


# ── rolling primitives ────────────────────────────────────────────────────────
# All three reduce over an explicit window view rather than differencing a running
# cumulative sum. The cumsum trick is O(n) instead of O(n·w), but on a series of
# perp prices near 60,000 it subtracts two nearly-equal large numbers and loses
# several digits — and the parity gate would then be measuring our own arithmetic
# rather than the divergence it exists to find. Windows here are tens of samples;
# the constant factor is not worth the precision.


@register("rolling_mean", inputs=("x",))
def rolling_mean(x, *, window: int) -> np.ndarray:
    """Mean of the trailing ``window`` observations, ending at each position."""
    a = _series(x, "x")
    window = _check_window(window)
    out = np.full(a.size, np.nan)
    view = _windows(a, window)
    if view is not None:
        out[window - 1 :] = view.mean(axis=1)
    return out


@register("rolling_std", inputs=("x",))
def rolling_std(x, *, window: int, ddof: int = 0) -> np.ndarray:
    """Standard deviation of the trailing ``window`` observations.

    ``ddof=0`` by default: the window *is* the population being described, and the
    sample correction would make the feature depend on the window length in a way
    the model then has to unlearn.
    """
    a = _series(x, "x")
    ddof = _check_window(ddof, name="ddof", minimum=0)
    window = _check_window(window, minimum=ddof + 1)
    out = np.full(a.size, np.nan)
    view = _windows(a, window)
    if view is not None:
        out[window - 1 :] = view.std(axis=1, ddof=ddof)
    return out


@register("rolling_sum", inputs=("x",))
def rolling_sum(x, *, window: int) -> np.ndarray:
    """Sum of the trailing ``window`` observations."""
    a = _series(x, "x")
    window = _check_window(window)
    out = np.full(a.size, np.nan)
    view = _windows(a, window)
    if view is not None:
        out[window - 1 :] = view.sum(axis=1)
    return out


@register("rolling_zscore", inputs=("x",))
def rolling_zscore(x, *, window: int, ddof: int = 0) -> np.ndarray:
    """How many trailing standard deviations the current value sits from its mean."""
    a = _series(x, "x")
    mu = rolling_mean(a, window=window)
    sd = rolling_std(a, window=window, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (a - mu) / sd
    # A frozen or stepped feed gives a zero-width window. The quotient is then ±inf
    # or NaN, and an inf that reaches a model is a position sized by a broken feed.
    return np.where(sd > 0.0, z, np.nan)


# ── returns and momentum ──────────────────────────────────────────────────────


@register("log_return", inputs=("price",))
def log_return(price, *, period: int = 1) -> np.ndarray:
    """``log(p_t / p_{t-period})``, NaN for the first ``period`` positions.

    Non-positive prices yield NaN rather than ``-inf``/NaN from the logarithm: a
    zero price is a gap in the feed, not a return of negative infinity, and the
    two must not be summarised the same way by whatever consumes this.
    """
    p = _series(price, "price")
    period = _check_window(period, name="period")
    out = np.full(p.size, np.nan)
    if p.size > period:
        num, den = p[period:], p[:-period]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = np.where((num > 0.0) & (den > 0.0), np.log(num / den), np.nan)
    return out


@register("momentum", inputs=("price",))
def momentum(price, *, window: int) -> np.ndarray:
    """Trailing ``window``-observation log price change.

    Deliberately *is* :func:`log_return` over a longer horizon rather than a second
    implementation of the same arithmetic. Two spellings of one transform is the
    bug ``docs/03`` warns about, in miniature.
    """
    return log_return(price, period=_check_window(window))


@register("realized_volatility", inputs=("price",))
def realized_volatility(price, *, window: int) -> np.ndarray:
    """Standard deviation of the trailing ``window`` one-step log returns.

    Unannualized: the sampling interval of a tick-driven series is not constant, so
    a √T scaling would be a fiction. Scale it downstream if a strategy needs it.
    """
    return rolling_std(log_return(price, period=1), window=window)


@register("ema", inputs=("x",))
def ema(x, *, span: int) -> np.ndarray:
    """Exponentially weighted mean with ``alpha = 2 / (span + 1)``.

    Computed by the recursion itself, one observation at a time, because the
    recursion *is* the definition — a closed form over powers of ``(1 - alpha)``
    underflows on long series and would put the online and offline paths on
    different arithmetic, which is exactly what the parity gate exists to catch.

    The level is seeded on the first finite observation, and the output stays NaN
    until ``span`` observations have been folded in; emitting the seed itself would
    hand the model a number that is really just "the first price we saw". A
    non-finite observation mid-series leaves the level untouched and blanks that
    one output, rather than poisoning the recursion for the rest of the session.
    """
    a = _series(x, "x")
    span = _check_window(span, name="span")
    alpha = 2.0 / (span + 1.0)
    out = np.full(a.size, np.nan)
    level = np.nan
    seen = 0
    for i in range(a.size):
        v = a[i]
        if not np.isfinite(v):
            continue
        level = v if seen == 0 else level + alpha * (v - level)
        seen += 1
        if seen >= span:
            out[i] = level
    return out


@register("sma_crossover", inputs=("price",))
def sma_crossover(price, *, fast: int, slow: int) -> np.ndarray:
    """Fast-minus-slow rolling mean as a fraction of the slow mean.

    The finite-lookback counterpart of :func:`ema_crossover`, and the reason to
    prefer it in a served spec: an EMA never forgets its seed, so a serving path
    holding a bounded buffer computes a *different* number from the research path
    that saw the whole history — and the gap is widest right after a restart, which
    is the moment nobody is watching the feature values. Every window here is
    finite, so a buffer of ``slow`` observations reproduces the offline value bit
    for bit and the feature-parity gate compares transforms rather than histories.

    Normalized by the slow mean for the same reason :func:`ema_crossover` is: the
    raw difference teaches a model that BTC moves in bigger numbers than SOL.
    """
    fast = _check_window(fast, name="fast")
    slow = _check_window(slow, name="slow")
    if fast >= slow:
        raise FeatureError(f"fast window must be shorter than slow, got fast={fast} slow={slow}")
    f, s = rolling_mean(price, window=fast), rolling_mean(price, window=slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (f - s) / s
    return np.where(np.abs(s) > 0.0, out, np.nan)


@register("ema_crossover", inputs=("price",))
def ema_crossover(price, *, fast: int, slow: int) -> np.ndarray:
    """Fast-minus-slow EMA as a fraction of the slow EMA.

    Normalizing by the slow EMA is what makes the feature comparable across price
    levels; the raw difference would train a model on "BTC moves in bigger numbers
    than SOL" and then mis-size every position when it is pointed at a new symbol.
    """
    fast = _check_window(fast, name="fast")
    slow = _check_window(slow, name="slow")
    if fast >= slow:
        raise FeatureError(f"fast span must be shorter than slow, got fast={fast} slow={slow}")
    f, s = ema(price, span=fast), ema(price, span=slow)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (f - s) / s
    return np.where(np.abs(s) > 0.0, out, np.nan)


# ── microstructure ────────────────────────────────────────────────────────────


@register("mid_price", inputs=("bid_px", "ask_px"))
def mid_price(bid_px, ask_px) -> np.ndarray:
    """``(bid + ask) / 2``, NaN whenever either side is missing.

    A missing side arrives as zero on the wire, and ``(0 + ask) / 2`` is half the
    ask: a price that never existed, several percent from anything tradable, fed
    straight into a return. NaN says "no book" and propagates honestly.
    """
    b, a = _series(bid_px, "bid_px"), _series(ask_px, "ask_px")
    _aligned("bid_px", b, "ask_px", a)
    return np.where((b > 0.0) & (a > 0.0), (b + a) / 2.0, np.nan)


@register("spread", inputs=("bid_px", "ask_px"))
def spread(bid_px, ask_px) -> np.ndarray:
    """``ask - bid`` in price units, NaN whenever either side is missing.

    A crossed book (negative result) is passed through rather than clamped: it is
    real information about a venue in trouble, and clamping it to zero makes the
    worst moment of the day look like the tightest market of the day.
    """
    b, a = _series(bid_px, "bid_px"), _series(ask_px, "ask_px")
    _aligned("bid_px", b, "ask_px", a)
    return np.where((b > 0.0) & (a > 0.0), a - b, np.nan)


@register("relative_spread", inputs=("bid_px", "ask_px"))
def relative_spread(bid_px, ask_px) -> np.ndarray:
    """Spread in **basis points of the mid** — the cross-symbol comparable form."""
    b, a = _series(bid_px, "bid_px"), _series(ask_px, "ask_px")
    m = mid_price(b, a)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / m * 10_000.0
    return np.where(m > 0.0, out, np.nan)


@register("book_imbalance", inputs=("bid_sz", "ask_sz"))
def book_imbalance(bid_sz, ask_sz) -> np.ndarray:
    """``(bid_sz - ask_sz) / (bid_sz + ask_sz)`` at the touch, in ``[-1, 1]``.

    An empty book on both sides is NaN, not 0. Zero here means "balanced", which is
    a strong claim about a book that is telling us nothing at all.
    """
    b, a = _series(bid_sz, "bid_sz"), _series(ask_sz, "ask_sz")
    _aligned("bid_sz", b, "ask_sz", a)
    total = b + a
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (b - a) / total
    return np.where((b >= 0.0) & (a >= 0.0) & (total > 0.0), out, np.nan)


@register("trade_flow_imbalance", inputs=("trade_sz", "trade_sign"))
def trade_flow_imbalance(trade_sz, trade_sign, *, window: int) -> np.ndarray:
    """Signed traded volume over the trailing ``window``, as a fraction of volume.

    ``trade_sign`` is ``+1`` for a buyer-initiated print, ``-1`` for a
    seller-initiated one, and ``0`` for an observation that carried no print (see
    :func:`axon.features.inputs.md_slice_inputs` — every market-data slice repeats
    the last trade, so a feed of quote updates would otherwise count one trade once
    per quote and read as a wall of one-sided flow).

    A window with no volume is NaN, for the same reason an empty book is: "no trades
    happened" is not "buys and sells balanced".
    """
    sz = _series(trade_sz, "trade_sz")
    sg = _series(trade_sign, "trade_sign")
    _aligned("trade_sz", sz, "trade_sign", sg)
    volume = np.abs(sz)
    signed = np.sign(sg) * volume
    signed_total = rolling_sum(signed, window=window)
    volume_total = rolling_sum(volume, window=window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = signed_total / volume_total
    return np.where(volume_total > 0.0, out, np.nan)


# ── bars ─────────────────────────────────────────────────────────────────────
# A closed OHLCV bar is the coarsest event the venue publishes and the only one a
# candle feed delivers. Both transforms below read the *extremes* the bar reached,
# which is the whole reason to take a bar over its own close price: a close-to-close
# series cannot tell a quiet 10 bps drift from a bar that traded 80 bps in each
# direction and came back, and those are different markets to put an order into.


@register("relative_range", inputs=("high", "low", "close"))
def relative_range(high, low, close) -> np.ndarray:
    """``(high - low) / close`` in **basis points of the close** — the ground covered.

    Basis points rather than price units for the same reason
    :func:`relative_spread` is: a 60-point range means one thing on BTC and another
    on a $2 perp, and a model fed the raw width learns the ticker rather than the
    volatility.

    A non-positive close, or a bar whose high is below its low, is NaN. Both are
    feed corruption rather than a quiet market, and a zero would put them in the
    same bucket as the calmest bar of the session.
    """
    h, low_, c = _series(high, "high"), _series(low, "low"), _series(close, "close")
    _aligned("high", h, "low", low_)
    _aligned("high", h, "close", c)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (h - low_) / c * 10_000.0
    return np.where((c > 0.0) & (h >= low_), out, np.nan)


@register("close_location", inputs=("high", "low", "close"))
def close_location(high, low, close) -> np.ndarray:
    """``(2c - h - l) / (h - l)`` in ``[-1, 1]``: where in its own range the bar closed.

    The bar-level analogue of :func:`book_imbalance`, and the closest thing to an
    order-flow read that a candle feed can honestly supply: a bar that ran up and
    gave it all back closes near its low, and that is different information from the
    same close reached by drifting up all bar.

    A bar with no range (``high == low``) is NaN, not 0. Zero here means "closed
    dead centre", which is a claim about a bar that never moved.
    """
    h, low_, c = _series(high, "high"), _series(low, "low"), _series(close, "close")
    _aligned("high", h, "low", low_)
    _aligned("high", h, "close", c)
    width = h - low_
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (2.0 * c - h - low_) / width
    return np.where(width > 0.0, out, np.nan)


def finite_rows(matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of the rows of a feature matrix that are usable.

    Warmup is NaN by construction, and different features warm up at different
    lengths, so "where does the usable data start" is a property of the matrix, not
    something a caller should compute from window sizes by hand and get wrong.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise FeatureError(f"expected a 2-D feature matrix, got shape {m.shape}")
    return np.isfinite(m).all(axis=1)


__all__ = [
    "FEATURES_VERSION",
    "book_imbalance",
    "close_location",
    "ema",
    "ema_crossover",
    "finite_rows",
    "log_return",
    "mid_price",
    "momentum",
    "realized_volatility",
    "relative_range",
    "relative_spread",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_zscore",
    "sma_crossover",
    "spread",
    "trade_flow_imbalance",
]
