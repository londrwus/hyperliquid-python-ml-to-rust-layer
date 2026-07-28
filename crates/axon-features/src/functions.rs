//! The transforms themselves — plain functions over `&[f64]`.
//!
//! A line-for-line counterpart of `axon.features.functions`, and the counterpart
//! relationship is the point: this file is a **second implementation** of transforms
//! `docs/03` says should never be implemented twice. It is allowed to exist only
//! because [`crate::parity`] holds it to bit equality with the first one, over real
//! market data, in CI. Read `docs/adr/0035-*.md` before changing a line of it — a
//! "tidier" formula here is a silent training–serving skew, and the gate is the only
//! thing between the two.
//!
//! ## The three rules, unchanged from Python
//!
//! 1. **Length-preserving.** `f(x)` writes `x.len()` elements and element `i` lines
//!    up with observation `i`. A transform returning a shorter series forces the
//!    caller to re-align it against timestamps, and every hand alignment is a place
//!    a feature ends up one row ahead of its own event — the windowing off-by-one
//!    `docs/03` lists among the silent killers.
//! 2. **Causal.** Element `i` depends on `x[..=i]` and nothing else. Appending future
//!    observations must not change a single historical value. This is the difference
//!    between a backtest that is optimistic and one that is real.
//! 3. **Warmup is NaN.** Not zero, not the seed, not back-filled. Zero is a legal
//!    value for every feature here, so a zero warmup is indistinguishable from a
//!    genuine reading and the model learns from it.
//!
//! ## Two arithmetic facts this file is built on
//!
//! - **Reductions go through [`crate::numeric`]**, never through `iter().sum()`.
//!   NumPy sums a window pairwise, and the naive loop disagrees by up to eight ULP.
//!   That module is where the reasoning lives.
//! - **`ln` is the one operation whose agreement is measured rather than
//!   guaranteed.** IEEE-754 requires `+ - * /` and `sqrt` to be correctly rounded,
//!   so those agree between NumPy and Rust on any conforming platform, full stop.
//!   `log` is not in that list: neither NumPy nor libm promises correct rounding,
//!   and they agree here because they are both computing it well, not because they
//!   must. Measured against NumPy 2.5.1 at **0 ULP** — over the 32 perp-close ratios
//!   `tests/fixtures/cross_language.json` pins, and over a 200 000-sample sweep across
//!   26 decades that `scripts/modal_libm_probe.py` re-runs on demand, on glibc 2.39
//!   here and on glibc 2.36 and musl in a container.
//!   `tests/cross_language.rs::rusts_natural_log_agrees_with_numpys_bit_for_bit_on_this_platform`
//!   pins it against NumPy's own answers, and the feature bundle's
//!   manifest lists which columns depend on it — so if this gate ever reddens on a
//!   different libm, the first question is whether it reddened *only* there.

use crate::numeric::{mean, pairwise_sum, std_in};
use crate::registry::{Param, Params};
use crate::FeatureError;

// ── shape helpers ─────────────────────────────────────────────────────────────

fn require_len(feature: &str, inputs: &[&[f64]], out: &[f64]) -> Result<usize, FeatureError> {
    let n = out.len();
    for (i, arr) in inputs.iter().enumerate() {
        if arr.len() != n {
            // Two market-data columns of different lengths are not describing the
            // same events, and a matrix built from them is misaligned in a way no
            // later check can detect.
            return Err(FeatureError::Inputs(format!(
                "{feature}: input {i} has {} rows but the column is {n}; arrays of different \
                 lengths are not the same events",
                arr.len()
            )));
        }
    }
    Ok(n)
}

fn fill_nan(out: &mut [f64]) {
    out.fill(f64::NAN);
}

/// `np.where(cond, value, nan)` in one place.
///
/// Spelled as a helper because the NaN branch is the *interesting* one in every
/// transform below, and writing it out seventeen times invites one of them to become
/// a zero. Zero is a legal value for every feature here; NaN is the only way to say
/// "no reading", and the two must never be confused.
#[inline]
fn or_nan(cond: bool, value: f64) -> f64 {
    if cond {
        value
    } else {
        f64::NAN
    }
}

// ── rolling primitives ────────────────────────────────────────────────────────
//
// All three reduce over an explicit window rather than differencing a running
// cumulative sum. The cumsum trick is O(n) instead of O(n·w), but on a series of
// perp prices near 60 000 it subtracts two nearly-equal large numbers and loses
// several digits — and the parity gate would then be measuring our own arithmetic
// rather than the divergence it exists to find. Windows here are tens of samples.
// This mirrors the identical choice, for the identical reason, in Python.

/// Mean of the trailing `window` observations, ending at each position.
pub fn rolling_mean(x: &[f64], window: usize, out: &mut [f64]) {
    fill_nan(out);
    if x.len() < window || window == 0 {
        return;
    }
    for i in (window - 1)..x.len() {
        out[i] = mean(&x[i + 1 - window..=i]);
    }
}

/// Standard deviation of the trailing `window` observations.
///
/// `ddof = 0` by default on the Python side: the window *is* the population being
/// described, and the sample correction would make the feature depend on the window
/// length in a way the model then has to unlearn.
pub fn rolling_std(x: &[f64], window: usize, ddof: usize, out: &mut [f64]) {
    fill_nan(out);
    if x.len() < window || window == 0 {
        return;
    }
    let mut scratch = vec![0.0f64; window];
    for i in (window - 1)..x.len() {
        let w = &x[i + 1 - window..=i];
        // The mean is taken here and handed down, so the two-pass structure is
        // visible at the call site rather than hidden: NumPy's `_var` computes the
        // whole window's mean first and only then the squared deviations.
        let mu = pairwise_sum(w) / window as f64;
        out[i] = std_in(w, ddof, mu, &mut scratch);
    }
}

/// Sum of the trailing `window` observations.
pub fn rolling_sum(x: &[f64], window: usize, out: &mut [f64]) {
    fill_nan(out);
    if x.len() < window || window == 0 {
        return;
    }
    for i in (window - 1)..x.len() {
        out[i] = pairwise_sum(&x[i + 1 - window..=i]);
    }
}

/// How many trailing standard deviations the current value sits from its mean.
pub fn rolling_zscore(x: &[f64], window: usize, ddof: usize, out: &mut [f64]) {
    let n = x.len();
    let mut mu = vec![0.0f64; n];
    let mut sd = vec![0.0f64; n];
    rolling_mean(x, window, &mut mu);
    rolling_std(x, window, ddof, &mut sd);
    for i in 0..n {
        // A frozen or stepped feed gives a zero-width window. The quotient is then
        // ±inf or NaN, and an inf that reaches a model is a position sized by a
        // broken feed. `sd > 0.0` is false for NaN too, which is what collapses the
        // warmup rows to NaN rather than to an infinity.
        out[i] = or_nan(sd[i] > 0.0, (x[i] - mu[i]) / sd[i]);
    }
}

// ── returns and momentum ──────────────────────────────────────────────────────

/// `log(p_t / p_{t-period})`, NaN for the first `period` positions.
///
/// Non-positive prices yield NaN rather than `-inf`/NaN from the logarithm: a zero
/// price is a gap in the feed, not a return of negative infinity, and the two must
/// not be summarised the same way by whatever consumes this.
pub fn log_return(price: &[f64], period: usize, out: &mut [f64]) {
    fill_nan(out);
    if price.len() <= period || period == 0 {
        return;
    }
    for i in period..price.len() {
        let (num, den) = (price[i], price[i - period]);
        out[i] = or_nan(num > 0.0 && den > 0.0, (num / den).ln());
    }
}

/// Trailing `window`-observation log price change.
///
/// Deliberately *is* [`log_return`] over a longer horizon rather than a second
/// implementation of the same arithmetic. Two spellings of one transform is the bug
/// `docs/03` warns about, in miniature — and the fact that this file exists at all
/// is that bug held at bay by a gate, so committing it again *inside* the file would
/// be a poor joke.
pub fn momentum(price: &[f64], window: usize, out: &mut [f64]) {
    log_return(price, window, out)
}

/// Standard deviation of the trailing `window` one-step log returns.
///
/// Unannualized: the sampling interval of a tick-driven series is not constant, so a
/// √T scaling would be a fiction. Scale it downstream if a strategy needs it.
pub fn realized_volatility(price: &[f64], window: usize, out: &mut [f64]) {
    let mut returns = vec![0.0f64; price.len()];
    log_return(price, 1, &mut returns);
    rolling_std(&returns, window, 0, out);
}

/// Exponentially weighted mean with `alpha = 2 / (span + 1)`.
///
/// Computed by the recursion itself, one observation at a time, because the
/// recursion *is* the definition — a closed form over powers of `(1 - alpha)`
/// underflows on long series and would put the online and offline paths on different
/// arithmetic, which is exactly what the parity gate exists to catch.
///
/// The level is seeded on the first finite observation, and the output stays NaN
/// until `span` observations have been folded in; emitting the seed itself would hand
/// the model a number that is really just "the first price we saw". A non-finite
/// observation mid-series leaves the level untouched and blanks that one output,
/// rather than poisoning the recursion for the rest of the session.
///
/// **This is the transform [`crate::streaming::FeatureStream`] refuses.** An EMA
/// never forgets its seed, so a bounded serving buffer computes a *different* number
/// from the research path that saw the whole history — and the gap is widest right
/// after a restart, which is the moment nobody is watching the feature values. It is
/// implemented here anyway, and faithfully, because the batch path is also the
/// research path and refusing to compute a legitimate research feature would just
/// move the second implementation somewhere with no gate on it.
pub fn ema(x: &[f64], span: usize, out: &mut [f64]) {
    fill_nan(out);
    if span == 0 {
        return;
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut level = f64::NAN;
    let mut seen: usize = 0;
    for i in 0..x.len() {
        let v = x[i];
        if !v.is_finite() {
            continue;
        }
        level = if seen == 0 {
            v
        } else {
            level + alpha * (v - level)
        };
        seen += 1;
        if seen >= span {
            out[i] = level;
        }
    }
}

/// Fast-minus-slow rolling mean as a fraction of the slow mean.
///
/// The finite-lookback counterpart of [`ema_crossover`], and the reason to prefer it
/// in a served spec: every window here is finite, so a buffer of `slow` observations
/// reproduces the offline value bit for bit and the parity gate compares transforms
/// rather than histories.
///
/// Normalized by the slow mean for the same reason [`ema_crossover`] is: the raw
/// difference teaches a model that BTC moves in bigger numbers than SOL.
pub fn sma_crossover(price: &[f64], fast: usize, slow: usize, out: &mut [f64]) {
    let n = price.len();
    let (mut f, mut s) = (vec![0.0f64; n], vec![0.0f64; n]);
    rolling_mean(price, fast, &mut f);
    rolling_mean(price, slow, &mut s);
    for i in 0..n {
        out[i] = or_nan(s[i].abs() > 0.0, (f[i] - s[i]) / s[i]);
    }
}

/// Fast-minus-slow EMA as a fraction of the slow EMA.
///
/// Normalizing by the slow EMA is what makes the feature comparable across price
/// levels; the raw difference would train a model on "BTC moves in bigger numbers
/// than SOL" and then mis-size every position when it is pointed at a new symbol.
pub fn ema_crossover(price: &[f64], fast: usize, slow: usize, out: &mut [f64]) {
    let n = price.len();
    let (mut f, mut s) = (vec![0.0f64; n], vec![0.0f64; n]);
    ema(price, fast, &mut f);
    ema(price, slow, &mut s);
    for i in 0..n {
        out[i] = or_nan(s[i].abs() > 0.0, (f[i] - s[i]) / s[i]);
    }
}

// ── microstructure ────────────────────────────────────────────────────────────

/// `(bid + ask) / 2`, NaN whenever either side is missing.
///
/// A missing side arrives as zero on the wire, and `(0 + ask) / 2` is half the ask: a
/// price that never existed, several percent from anything tradable, fed straight
/// into a return. NaN says "no book" and propagates honestly.
pub fn mid_price(bid_px: &[f64], ask_px: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (b, a) = (bid_px[i], ask_px[i]);
        out[i] = or_nan(b > 0.0 && a > 0.0, (b + a) / 2.0);
    }
}

/// `ask - bid` in price units, NaN whenever either side is missing.
///
/// A crossed book (negative result) is passed through rather than clamped: it is real
/// information about a venue in trouble, and clamping it to zero makes the worst
/// moment of the day look like the tightest market of the day.
pub fn spread(bid_px: &[f64], ask_px: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (b, a) = (bid_px[i], ask_px[i]);
        out[i] = or_nan(b > 0.0 && a > 0.0, a - b);
    }
}

/// Spread in **basis points of the mid** — the cross-symbol comparable form.
pub fn relative_spread(bid_px: &[f64], ask_px: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (b, a) = (bid_px[i], ask_px[i]);
        let m = or_nan(b > 0.0 && a > 0.0, (b + a) / 2.0);
        // `(a - b) / m * 10_000.0`, in that association. Multiplying the basis-point
        // factor in first would be a different float.
        out[i] = or_nan(m > 0.0, (a - b) / m * 10_000.0);
    }
}

/// `(bid_sz - ask_sz) / (bid_sz + ask_sz)` at the touch, in `[-1, 1]`.
///
/// An empty book on both sides is NaN, not 0. Zero here means "balanced", which is a
/// strong claim about a book that is telling us nothing at all.
pub fn book_imbalance(bid_sz: &[f64], ask_sz: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (b, a) = (bid_sz[i], ask_sz[i]);
        let total = b + a;
        out[i] = or_nan(b >= 0.0 && a >= 0.0 && total > 0.0, (b - a) / total);
    }
}

/// Signed traded volume over the trailing `window`, as a fraction of volume.
///
/// `trade_sign` is `+1` for a buyer-initiated print, `-1` for a seller-initiated one,
/// and `0` for an observation that carried no print — every market-data slice repeats
/// the last trade, so a feed of quote updates would otherwise count one trade once per
/// quote and read as a wall of one-sided flow.
///
/// A window with no volume is NaN, for the same reason an empty book is: "no trades
/// happened" is not "buys and sells balanced".
pub fn trade_flow_imbalance(trade_sz: &[f64], trade_sign: &[f64], window: usize, out: &mut [f64]) {
    let n = out.len();
    let volume: Vec<f64> = trade_sz.iter().map(|v| v.abs()).collect();
    // `np.sign` is NaN-preserving and zero-preserving; `signum` is not — it maps
    // 0.0 to 1.0 and -0.0 to -1.0, which would turn every quote-only observation
    // into a one-lot buy. That is the whole failure this feature's `0` sign exists
    // to prevent, so the mapping is spelled out rather than delegated.
    let signed: Vec<f64> = volume
        .iter()
        .zip(trade_sign.iter())
        .map(|(vol, sg)| np_sign(*sg) * vol)
        .collect();
    let (mut signed_total, mut volume_total) = (vec![0.0f64; n], vec![0.0f64; n]);
    rolling_sum(&signed, window, &mut signed_total);
    rolling_sum(&volume, window, &mut volume_total);
    for i in 0..n {
        out[i] = or_nan(volume_total[i] > 0.0, signed_total[i] / volume_total[i]);
    }
}

/// `np.sign` semantics: `-1`, `0`, `+1`, and NaN for NaN.
#[inline]
fn np_sign(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ── bars ─────────────────────────────────────────────────────────────────────
//
// A closed OHLCV bar is the coarsest event the venue publishes and the only one a
// candle feed delivers. Both transforms below read the *extremes* the bar reached,
// which is the whole reason to take a bar over its own close price: a close-to-close
// series cannot tell a quiet 10 bps drift from a bar that traded 80 bps in each
// direction and came back, and those are different markets to put an order into.

/// `(high - low) / close` in **basis points of the close** — the ground covered.
///
/// A non-positive close, or a bar whose high is below its low, is NaN. Both are feed
/// corruption rather than a quiet market, and a zero would put them in the same
/// bucket as the calmest bar of the session.
pub fn relative_range(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (h, l, c) = (high[i], low[i], close[i]);
        out[i] = or_nan(c > 0.0 && h >= l, (h - l) / c * 10_000.0);
    }
}

/// `(2c - h - l) / (h - l)` in `[-1, 1]`: where in its own range the bar closed.
///
/// The bar-level analogue of [`book_imbalance`], and the closest thing to an
/// order-flow read a candle feed can honestly supply: a bar that ran up and gave it
/// all back closes near its low, and that is different information from the same
/// close reached by drifting up all bar.
///
/// A bar with no range (`high == low`) is NaN, not 0. Zero here means "closed dead
/// centre", which is a claim about a bar that never moved — and it is not rare:
/// **6 of 58 BTC and 5 of 62 ETH live m1 bars** hit it in the ADR-0029 shadow run,
/// against **0 of the 800-bar committed hourly fixtures**. Only the BTC half is
/// checkable from this tree — those 58 bars are the `bar_m1_testnet_live` parity
/// bundle, and `test_committed_feature_bundles.py` asserts the NaNs are still there;
/// the ETH count is quoted from that run's own report and no artifact carries it.
/// A quiet strategy on m1 may be NaN rather than
/// undecided, and this is where that comes from.
pub fn close_location(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    for i in 0..out.len() {
        let (h, l, c) = (high[i], low[i], close[i]);
        let width = h - l;
        // `(2.0 * c - h - l)`, left to right: `((2c - h) - l)`.
        out[i] = or_nan(width > 0.0, (2.0 * c - h - l) / width);
    }
}

/// Whether every cell of a feature row is finite, and therefore usable.
///
/// Warmup is NaN by construction and different features warm up at different
/// lengths, so "where does the usable data start" is a property of the matrix, not
/// something a caller should compute from window sizes by hand and get wrong.
pub fn row_is_finite(row: &[f64]) -> bool {
    row.iter().all(|v| v.is_finite())
}

// ── registry adapters ─────────────────────────────────────────────────────────
//
// One `eval_*` and one `lookback_*` per registered name, matching the `EvalFn` and
// `LookbackFn` signatures in `registry.rs`. They are mechanical on purpose: every
// decision worth making is in the transform above, and an adapter that did anything
// clever would be a place the registry and the function could disagree about which
// array is which.

macro_rules! inputs {
    ($feature:expr, $inputs:expr, $out:expr, $n:expr) => {{
        require_len($feature, $inputs, $out)?;
        if $inputs.len() != $n {
            return Err(FeatureError::Inputs(format!(
                "{}: expected {} input array(s), got {}",
                $feature,
                $n,
                $inputs.len()
            )));
        }
    }};
}

pub fn eval_rolling_mean(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("rolling_mean", inputs, out, 1);
    let window = params.required_window("rolling_mean", "window", 1)?;
    rolling_mean(inputs[0], window, out);
    Ok(())
}

pub fn eval_rolling_std(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("rolling_std", inputs, out, 1);
    let ddof = params.window("rolling_std", "ddof", 0, 0)?;
    // The minimum is `ddof + 1`, not 1: a window equal to the delta degrees of
    // freedom divides by zero, and the resulting inf/NaN would read as a broken feed
    // rather than as a spec that cannot be honoured.
    let window = params.required_window("rolling_std", "window", ddof as i64 + 1)?;
    rolling_std(inputs[0], window, ddof, out);
    Ok(())
}

pub fn eval_rolling_sum(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("rolling_sum", inputs, out, 1);
    let window = params.required_window("rolling_sum", "window", 1)?;
    rolling_sum(inputs[0], window, out);
    Ok(())
}

pub fn eval_rolling_zscore(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("rolling_zscore", inputs, out, 1);
    let ddof = params.window("rolling_zscore", "ddof", 0, 0)?;
    let window = params.required_window("rolling_zscore", "window", ddof as i64 + 1)?;
    rolling_zscore(inputs[0], window, ddof, out);
    Ok(())
}

pub fn eval_log_return(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("log_return", inputs, out, 1);
    // `period` is the one window-shaped parameter with a real default on the Python
    // side (`period: int = 1`), so it is the one that may be omitted.
    let period = params.window("log_return", "period", 1, 1)?;
    log_return(inputs[0], period, out);
    Ok(())
}

pub fn eval_momentum(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("momentum", inputs, out, 1);
    let window = params.required_window("momentum", "window", 1)?;
    momentum(inputs[0], window, out);
    Ok(())
}

pub fn eval_realized_volatility(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("realized_volatility", inputs, out, 1);
    let window = params.required_window("realized_volatility", "window", 1)?;
    realized_volatility(inputs[0], window, out);
    Ok(())
}

pub fn eval_ema(inputs: &[&[f64]], params: &Params, out: &mut [f64]) -> Result<(), FeatureError> {
    inputs!("ema", inputs, out, 1);
    let span = params.required_window("ema", "span", 1)?;
    ema(inputs[0], span, out);
    Ok(())
}

pub fn eval_sma_crossover(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("sma_crossover", inputs, out, 1);
    let (fast, slow) = crossover_spans("sma_crossover", params, "window")?;
    sma_crossover(inputs[0], fast, slow, out);
    Ok(())
}

pub fn eval_ema_crossover(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("ema_crossover", inputs, out, 1);
    let (fast, slow) = crossover_spans("ema_crossover", params, "span")?;
    ema_crossover(inputs[0], fast, slow, out);
    Ok(())
}

/// `fast` and `slow`, with the ordering rule both crossovers enforce.
///
/// `noun` is "window" or "span" only so the message reads the way Python's does; the
/// check is the same one, and it is a real check rather than a formality: a `fast`
/// that is not shorter than `slow` inverts the sign of the whole feature, which
/// trains and backtests perfectly well while meaning the opposite of its name.
fn crossover_spans(
    feature: &str,
    params: &Params,
    noun: &str,
) -> Result<(usize, usize), FeatureError> {
    let fast = params.required_window(feature, "fast", 1)?;
    let slow = params.required_window(feature, "slow", 1)?;
    if fast >= slow {
        return Err(FeatureError::Param {
            feature: feature.to_string(),
            message: format!("fast {noun} must be shorter than slow, got fast={fast} slow={slow}"),
        });
    }
    Ok((fast, slow))
}

pub fn eval_mid_price(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("mid_price", inputs, out, 2);
    mid_price(inputs[0], inputs[1], out);
    Ok(())
}

pub fn eval_spread(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("spread", inputs, out, 2);
    spread(inputs[0], inputs[1], out);
    Ok(())
}

pub fn eval_relative_spread(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("relative_spread", inputs, out, 2);
    relative_spread(inputs[0], inputs[1], out);
    Ok(())
}

pub fn eval_book_imbalance(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("book_imbalance", inputs, out, 2);
    book_imbalance(inputs[0], inputs[1], out);
    Ok(())
}

pub fn eval_trade_flow_imbalance(
    inputs: &[&[f64]],
    params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("trade_flow_imbalance", inputs, out, 2);
    let window = params.required_window("trade_flow_imbalance", "window", 1)?;
    trade_flow_imbalance(inputs[0], inputs[1], window, out);
    Ok(())
}

pub fn eval_relative_range(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("relative_range", inputs, out, 3);
    relative_range(inputs[0], inputs[1], inputs[2], out);
    Ok(())
}

pub fn eval_close_location(
    inputs: &[&[f64]],
    _params: &Params,
    out: &mut [f64],
) -> Result<(), FeatureError> {
    inputs!("close_location", inputs, out, 3);
    close_location(inputs[0], inputs[1], inputs[2], out);
    Ok(())
}

// ── lookbacks ─────────────────────────────────────────────────────────────────
//
// How many trailing observations of the *input* a transform needs to produce its
// last value. `None` means "all of them", which is not an unknown — it is the honest
// answer for an EMA, and it is what `FeatureStream` refuses a spec on.
//
// Every one of these is derived from the transform's own arithmetic rather than
// restated from a docstring, because "measure a warmup, do not restate it" is a house
// rule with a specific failure behind it: a docstring saying "21 bars" goes stale the
// day a window widens, the buffer is then one bar short, every row stays NaN, the
// strategy emits nothing forever, and *nothing raises*.

/// One observation: the transform reads only the current row.
pub fn lookback_pointwise(_params: &Params) -> Result<Option<usize>, FeatureError> {
    Ok(Some(1))
}

/// Unbounded: the value depends on every observation ever seen.
pub fn lookback_unbounded(_params: &Params) -> Result<Option<usize>, FeatureError> {
    Ok(None)
}

/// `window` observations, for the plain rolling reductions.
pub fn lookback_window(params: &Params) -> Result<Option<usize>, FeatureError> {
    // The feature name is only used in the error text here, and every caller of this
    // one takes a required `window`; naming it "rolling" keeps the message honest
    // without pretending to know which of the five it was.
    Ok(Some(params.required_window("rolling", "window", 1)?))
}

/// `period + 1`: a return needs the observation it is measured against.
pub fn lookback_log_return(params: &Params) -> Result<Option<usize>, FeatureError> {
    Ok(Some(params.window("log_return", "period", 1, 1)? + 1))
}

/// `window + 1`, for the same reason [`lookback_log_return`] is `period + 1`.
pub fn lookback_momentum(params: &Params) -> Result<Option<usize>, FeatureError> {
    Ok(Some(params.required_window("momentum", "window", 1)? + 1))
}

/// `window + 1`: a `window`-sample deviation *of one-step returns* reaches back
/// through the return's own extra observation.
///
/// This is the arithmetic behind `BAR_M1_WARMUP_BARS = 21` for a 20-sample
/// `vol_20` — the first finite row is index 20, the 21st bar — and getting it wrong
/// by one is the difference between a serving buffer that reproduces the offline
/// value and one that is permanently NaN.
pub fn lookback_realized_volatility(params: &Params) -> Result<Option<usize>, FeatureError> {
    Ok(Some(
        params.required_window("realized_volatility", "window", 1)? + 1,
    ))
}

/// `slow`: the longer of the two means is what gates the first finite value.
pub fn lookback_sma_crossover(params: &Params) -> Result<Option<usize>, FeatureError> {
    let (_, slow) = crossover_spans("sma_crossover", params, "window")?;
    Ok(Some(slow))
}

/// A `Param::Int`, for building params in tests and in tooling.
impl From<i64> for Param {
    fn from(v: i64) -> Self {
        Param::Int(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: impl Fn(&mut [f64]), n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n];
        f(&mut out);
        out
    }

    fn params(pairs: &[(&str, i64)]) -> Params {
        let mut p = Params::new();
        for (k, v) in pairs {
            p.insert(*k, Param::Int(*v));
        }
        p
    }

    // ── the three rules ──

    #[test]
    fn every_transform_is_length_preserving_so_no_row_shifts_against_its_event() {
        // The rule that removes hand re-alignment against timestamps. A transform
        // writing a prefix would leave the tail holding whatever the buffer had,
        // which reads as data rather than as a bug.
        let x: Vec<f64> = (1..=40).map(|i| 100.0 + i as f64).collect();
        let two: Vec<f64> = x.iter().map(|v| v + 1.0).collect();
        let cases: Vec<(&str, Vec<&[f64]>, Params)> = vec![
            ("rolling_mean", vec![&x], params(&[("window", 5)])),
            ("rolling_std", vec![&x], params(&[("window", 5)])),
            ("rolling_sum", vec![&x], params(&[("window", 5)])),
            ("rolling_zscore", vec![&x], params(&[("window", 5)])),
            ("log_return", vec![&x], params(&[("period", 2)])),
            ("momentum", vec![&x], params(&[("window", 3)])),
            ("realized_volatility", vec![&x], params(&[("window", 4)])),
            ("ema", vec![&x], params(&[("span", 4)])),
            (
                "sma_crossover",
                vec![&x],
                params(&[("fast", 2), ("slow", 6)]),
            ),
            (
                "ema_crossover",
                vec![&x],
                params(&[("fast", 2), ("slow", 6)]),
            ),
            ("mid_price", vec![&x, &two], Params::new()),
            ("spread", vec![&x, &two], Params::new()),
            ("relative_spread", vec![&x, &two], Params::new()),
            ("book_imbalance", vec![&x, &two], Params::new()),
            (
                "trade_flow_imbalance",
                vec![&x, &two],
                params(&[("window", 4)]),
            ),
            ("relative_range", vec![&two, &x, &x], Params::new()),
            ("close_location", vec![&two, &x, &x], Params::new()),
        ];
        assert_eq!(
            cases.len(),
            crate::registry::registered_features().len(),
            "the sweep does not cover every registered transform; a feature added \
             later must not be able to opt out of the three rules by being forgotten"
        );
        // The buffer is pre-filled with a sentinel no transform can produce, and the
        // assertion is that **none survives**. Asserting `out.len() == x.len()` would
        // be a tautology — `EvalFn` takes `&mut [f64]` and a callee cannot change the
        // caller's length — so the first version of this test could not fail, and in
        // particular could not see the failure it is named for: a transform that
        // writes a prefix and leaves the tail holding whatever the buffer had, which
        // reads as data rather than as a bug.
        const UNWRITTEN: f64 = -12_345.678_901_234_5;
        for (name, inputs, p) in cases {
            let info = crate::registry::feature_info(name).unwrap();
            let mut out = vec![UNWRITTEN; x.len()];
            (info.eval)(&inputs, &p, &mut out).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(out.len(), x.len(), "{name} is not length-preserving");
            if let Some(i) = out.iter().position(|v| v.to_bits() == UNWRITTEN.to_bits()) {
                panic!("{name} left row {i} unwritten; the tail holds stale buffer, not data");
            }
        }
    }

    #[test]
    fn extending_a_series_with_future_data_changes_no_historical_value() {
        // Causality, enforced across the registry rather than function by function —
        // the same shape as Python's point-in-time test. A feature added next month
        // cannot opt out by forgetting to write a test, because the sweep above
        // asserts it covers every registered name.
        let full: Vec<f64> = (1..=60)
            .map(|i| 100.0 + ((i * 7 % 23) as f64) * 0.5)
            .collect();
        let prefix = &full[..40];
        for name in crate::registry::registered_features() {
            let info = crate::registry::feature_info(name).unwrap();
            let p = match *name {
                "ema" => params(&[("span", 4)]),
                "sma_crossover" | "ema_crossover" => params(&[("fast", 2), ("slow", 6)]),
                "log_return" => params(&[("period", 2)]),
                _ => params(&[("window", 5)]),
            };
            let n_in = info.inputs.len();
            let long_in: Vec<&[f64]> = (0..n_in).map(|_| &full[..]).collect();
            let short_in: Vec<&[f64]> = (0..n_in).map(|_| prefix).collect();
            let mut long_out = vec![0.0; full.len()];
            let mut short_out = vec![0.0; prefix.len()];
            (info.eval)(&long_in, &p, &mut long_out).unwrap();
            (info.eval)(&short_in, &p, &mut short_out).unwrap();
            for i in 0..prefix.len() {
                let (a, b) = (long_out[i], short_out[i]);
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                    "{name} is not causal: row {i} reads {a} with the future attached \
                     and {b} without it"
                );
            }
        }
    }

    #[test]
    fn a_leaky_centred_mean_fails_the_causality_check_that_the_real_ones_pass() {
        // A leakage detector nobody has seen fail is a decoration. This is the
        // deliberately-leaky counter-example: a centred window reads x[i+1], and the
        // check above must be able to catch it.
        let full: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let centred = |x: &[f64]| -> Vec<f64> {
            (0..x.len())
                .map(|i| {
                    if i == 0 || i + 1 >= x.len() {
                        f64::NAN
                    } else {
                        (x[i - 1] + x[i] + x[i + 1]) / 3.0
                    }
                })
                .collect()
        };
        let long = centred(&full);
        let short = centred(&full[..30]);
        let leaked = (0..30).any(|i| {
            let (a, b) = (long[i], short[i]);
            !(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()))
        });
        assert!(
            leaked,
            "the causality check cannot see a known-leaky transform"
        );
    }

    #[test]
    fn warmup_is_nan_and_never_zero_because_zero_is_a_legal_reading() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let out = run(|o| rolling_mean(&x, 4, o), x.len());
        assert!(out[..3].iter().all(|v| v.is_nan()), "warmup was not NaN");
        assert_eq!(out[3], 2.5);
        // A series shorter than its own window is entirely warmup, not an error.
        let short = run(|o| rolling_mean(&x[..2], 4, o), 2);
        assert!(short.iter().all(|v| v.is_nan()));
    }

    // ── the NaN contract, per transform ──

    #[test]
    fn a_missing_book_side_is_nan_rather_than_half_the_ask() {
        // A missing side arrives as zero on the wire. `(0 + ask) / 2` is a price
        // that never existed, several percent from anything tradable.
        let out = run(|o| mid_price(&[0.0, 100.0], &[102.0, 102.0], o), 2);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 101.0);
    }

    #[test]
    fn a_crossed_book_passes_through_rather_than_being_clamped_to_zero() {
        // Clamping makes the worst moment of the day look like the tightest market
        // of the day.
        let out = run(|o| spread(&[103.0], &[102.0], o), 1);
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn an_empty_book_is_nan_not_a_balanced_one() {
        let out = run(|o| book_imbalance(&[0.0, 3.0], &[0.0, 1.0], o), 2);
        assert!(out[0].is_nan(), "an empty book read as balanced");
        assert_eq!(out[1], 0.5);
    }

    #[test]
    fn a_bar_with_no_range_is_nan_not_a_close_dead_centre() {
        // The live m1 case: 6 of 58 BTC bars in a measured session traded at one
        // price for the whole minute. Zero would say "closed dead centre", which is a
        // claim about a bar that never moved.
        let out = run(
            |o| close_location(&[50.0, 60.0], &[50.0, 40.0], &[50.0, 55.0], o),
            2,
        );
        assert!(out[0].is_nan());
        assert_eq!(out[1], 0.5);
    }

    #[test]
    fn a_high_below_its_low_is_nan_because_it_is_corruption_not_a_quiet_market() {
        let out = run(|o| relative_range(&[40.0], &[60.0], &[50.0], o), 1);
        assert!(out[0].is_nan());
    }

    #[test]
    fn a_non_positive_price_yields_nan_rather_than_negative_infinity() {
        // A zero price is a gap in the feed, not a return of negative infinity, and
        // the two must not be summarised the same way.
        let out = run(|o| log_return(&[100.0, 0.0, 100.0], 1, o), 3);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan(), "log of zero leaked through as {}", out[1]);
        assert!(out[2].is_nan());
    }

    #[test]
    fn a_frozen_feed_gives_nan_rather_than_an_infinite_zscore() {
        // A zero-width window makes the quotient ±inf, and an inf that reaches a
        // model is a position sized by a broken feed.
        let flat = [7.0; 10];
        let out = run(|o| rolling_zscore(&flat, 4, 0, o), 10);
        assert!(
            out[4..].iter().all(|v| v.is_nan()),
            "a frozen feed scored finite"
        );
    }

    #[test]
    fn a_window_with_no_volume_is_nan_not_balanced_flow() {
        let sz = [0.0; 6];
        let sign = [0.0; 6];
        let out = run(|o| trade_flow_imbalance(&sz, &sign, 3, o), 6);
        assert!(out[2..].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn a_quote_only_observation_contributes_no_flow_because_sign_zero_stays_zero() {
        // `f64::signum` maps 0.0 to 1.0, which would turn every quote update into a
        // one-lot buy and read as a wall of one-sided flow. `np_sign` is the reason
        // this feature can be computed off a slice stream at all.
        assert_eq!(np_sign(0.0), 0.0);
        assert_eq!(np_sign(-0.0), 0.0);
        assert_eq!(0.0f64.signum(), 1.0, "the trap this guards against moved");
        let sz = [5.0, 0.0, 0.0, 5.0];
        let sign = [1.0, 0.0, 0.0, -1.0];
        let out = run(|o| trade_flow_imbalance(&sz, &sign, 4, o), 4);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn an_ema_never_forgets_its_seed_which_is_why_streaming_refuses_it() {
        // The measurement behind the "no EMA in a served spec" rule: the same span
        // over the same tail gives a different number depending on how much history
        // preceded it. `sma_crossover` is the finite-lookback counterpart and does
        // not have this property — asserted beside it so the contrast is the test.
        let long: Vec<f64> = (1..=200).map(|i| 100.0 + (i % 5) as f64).collect();
        let tail = &long[150..];
        let full = run(|o| ema(&long, 8, o), long.len());
        let bounded = run(|o| ema(tail, 8, o), tail.len());
        assert_ne!(
            full[long.len() - 1].to_bits(),
            bounded[tail.len() - 1].to_bits(),
            "an EMA over a bounded buffer matched the full history; the refusal in \
             FeatureStream would then be guarding nothing"
        );

        let full_sma = run(|o| sma_crossover(&long, 2, 6, o), long.len());
        let bounded_sma = run(|o| sma_crossover(tail, 2, 6, o), tail.len());
        assert_eq!(
            full_sma[long.len() - 1].to_bits(),
            bounded_sma[tail.len() - 1].to_bits(),
            "a finite-lookback crossover disagreed across a bounded buffer"
        );
    }

    #[test]
    fn a_non_finite_observation_blanks_one_ema_output_rather_than_poisoning_the_rest() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        let out = run(|o| ema(&x, 2, o), 6);
        assert!(out[2].is_nan(), "the NaN row should be blank");
        assert!(
            out[3].is_finite(),
            "the recursion was poisoned past the NaN"
        );
    }

    // ── parameter refusals ──

    #[test]
    fn a_crossover_with_fast_not_shorter_than_slow_is_refused_not_silently_inverted() {
        // An inverted pair flips the sign of the whole feature, which trains and
        // backtests perfectly well while meaning the opposite of its name.
        let p = params(&[("fast", 6), ("slow", 6)]);
        let x = [1.0; 10];
        let mut out = vec![0.0; 10];
        assert!(matches!(
            eval_sma_crossover(&[&x], &p, &mut out),
            Err(FeatureError::Param { .. })
        ));
    }

    #[test]
    fn a_rolling_std_window_equal_to_its_ddof_is_refused_not_divided_by_zero() {
        let p = params(&[("window", 2), ("ddof", 2)]);
        let x = [1.0; 10];
        let mut out = vec![0.0; 10];
        assert!(matches!(
            eval_rolling_std(&[&x], &p, &mut out),
            Err(FeatureError::Param { .. })
        ));
    }

    #[test]
    fn a_missing_required_window_is_refused_rather_than_defaulted_to_something_plausible() {
        let x = [1.0; 10];
        let mut out = vec![0.0; 10];
        assert!(matches!(
            eval_rolling_mean(&[&x], &Params::new(), &mut out),
            Err(FeatureError::Param { .. })
        ));
        // `period` is the exception: it genuinely defaults to 1 on both sides.
        assert!(eval_log_return(&[&x], &Params::new(), &mut out).is_ok());
    }

    #[test]
    fn ragged_inputs_are_refused_because_they_are_not_describing_the_same_events() {
        let a = [1.0; 10];
        let b = [1.0; 9];
        let mut out = vec![0.0; 10];
        assert!(matches!(
            eval_mid_price(&[&a, &b], &Params::new(), &mut out),
            Err(FeatureError::Inputs(_))
        ));
    }

    // ── lookbacks ──

    #[test]
    fn the_realized_volatility_lookback_reaches_through_the_returns_own_extra_bar() {
        // 20 samples of one-step returns need 21 observations. Off by one here and a
        // serving buffer is permanently one bar short: every row stays NaN, the
        // strategy emits nothing forever, and nothing raises.
        let p = params(&[("window", 20)]);
        assert_eq!(lookback_realized_volatility(&p).unwrap(), Some(21));

        // And the measurement, rather than the claim: the first finite row of a
        // 20-sample realized volatility over 21 bars is index 20.
        let price: Vec<f64> = (0..40).map(|i| 100.0 + (i % 7) as f64).collect();
        let out = run(|o| realized_volatility(&price, 20, o), price.len());
        assert!(out[19].is_nan());
        assert!(out[20].is_finite());
    }

    #[test]
    fn an_ema_reports_an_unbounded_lookback_rather_than_a_large_one() {
        // `None` is the honest answer, and it is what FeatureStream refuses on. A
        // large finite number here would let a bounded buffer be built that is
        // wrong by an amount nobody can bound.
        assert_eq!(lookback_unbounded(&Params::new()).unwrap(), None);
        let info = crate::registry::feature_info("ema").unwrap();
        assert_eq!((info.lookback)(&params(&[("span", 8)])).unwrap(), None);
        let info = crate::registry::feature_info("ema_crossover").unwrap();
        assert_eq!(
            (info.lookback)(&params(&[("fast", 8), ("slow", 32)])).unwrap(),
            None
        );
    }

    #[test]
    fn every_registered_transform_answers_the_lookback_question() {
        // A transform whose lookback function panicked or errored would take down
        // the streaming runtime at spec-load time, which is a worse failure than
        // being unbounded.
        for name in crate::registry::registered_features() {
            let info = crate::registry::feature_info(name).unwrap();
            let p = match *name {
                "ema" => params(&[("span", 4)]),
                "sma_crossover" | "ema_crossover" => params(&[("fast", 2), ("slow", 6)]),
                "log_return" => params(&[("period", 2)]),
                _ => params(&[("window", 5)]),
            };
            (info.lookback)(&p).unwrap_or_else(|e| panic!("{name} lookback: {e}"));
        }
    }

    // ── the arithmetic that has to match NumPy ──

    #[test]
    fn a_rolling_mean_goes_through_the_pairwise_reduction_not_a_running_total() {
        // The whole crate's bit-exactness rests on this. A running total over the
        // window (or an incremental add/subtract) would be a different number, and
        // the difference is invisible to every check except the cross-language gate.
        let x = crate::numeric::perp_series(40);
        let out = run(|o| rolling_mean(&x, 20, o), x.len());
        let w = &x[20..40];
        assert_eq!(out[39].to_bits(), (pairwise_sum(w) / 20.0).to_bits());

        let mut naive = 0.0;
        for v in w {
            naive += v;
        }
        assert_ne!(
            out[39].to_bits(),
            (naive / 20.0).to_bits(),
            "the pairwise and naive means agreed on a window built to separate them"
        );
    }
}
