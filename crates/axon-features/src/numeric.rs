//! The reductions, in NumPy's own summation order.
//!
//! This module is the reason the feature gate can be held to *bits* instead of a
//! tolerance, and it is the least obvious file in the crate. Everything else here
//! is arithmetic anyone would write; this is arithmetic written to agree with
//! somebody else's.
//!
//! # Why a plain `iter().sum()` is wrong
//!
//! `axon.features` reduces a rolling window with `np.ndarray.sum`, and NumPy does
//! **not** accumulate left to right. It uses pairwise summation: eight independent
//! accumulators for a window of eight or more, combined in a fixed tree, and a
//! recursive split above 128. The result is a different number from the obvious
//! loop — not a worse one, a *different* one.
//!
//! **How different depends on the data**, and that is worth understanding before
//! quoting a figure. Two summation orders round apart only when the summands' low
//! bits make them; on a series of whole-dollar BTC closes they can agree everywhere.
//! `tests/fixtures/generate.py` therefore records the separation per window, measured
//! by NumPy 2.5.1 over its own 160-sample perp-close series, and the fixture carries
//! a `naive_separates` flag beside each:
//!
//! | window | separates? | naive left-to-right vs `np.sum` |
//! |---|---|---|
//! | 5, 7 | no  | 0 ULP (below the unroll threshold, NumPy loops too) |
//! | 8    | yes | 1 ULP |
//! | 20   | **no** on this series | 0 ULP |
//! | 32   | yes | 2 ULP |
//! | 128  | yes | 3 ULP |
//! | 129  | yes | 4 ULP |
//!
//! Note the `20` row: a window `BAR_M1_V1` actually uses, on which the two orders
//! happen to agree here. So "the naive loop is wrong on every rolling column" would
//! be false, and falsifiable from this crate's own committed fixture.
//!
//! The magnitude that matters is not a ULP count on one window. Swapping this
//! transcription for a naive fold and recomputing the four committed parity bundles
//! moves **2 689 cells** — 1 398 in `all_transforms`, 584 and 684 in the two
//! mainnet bar corpora, 23 in the 58-bar live tape — concentrated in exactly the
//! columns that reduce. One ULP is nothing to a model and everything to a gate: the
//! failure would read as "the Rust transforms are wrong" when the transforms were
//! fine and the *reduction order* differed. Worse is the other outcome — widen the
//! criterion to absorb it, and the gate that was built to catch a windowing
//! off-by-one is now blind to a systematic per-element error.
//!
//! So the order is reproduced rather than tolerated. [`pairwise_sum`] is a
//! transcription of NumPy's `DOUBLE_pairwise_sum` (`loops.c.src`), and
//! `tests/cross_language.rs` pins it against real NumPy output rather than
//! against this description of it — a comment claiming to match NumPy is a comment,
//! and the day NumPy changes its unroll factor is a day this crate has to find out
//! from a red test rather than from a parity report on a live feed.
//!
//! # What this does not claim
//!
//! Nothing here says NumPy's order is *better*. Pairwise summation has lower error
//! growth than the naive loop, but the point is agreement, not accuracy: the whole
//! premise of ADR-0035 is that the offline recompute is the definition and the Rust
//! path has to reproduce it. If Python's arithmetic were the worse of the two, this
//! file would still transcribe it, and the fix would be to change Python — where
//! the change moves `FEATURES_VERSION` and invalidates every artifact trained on
//! the old numbers, which is exactly the loud failure ADR-0016 §2 designed.

/// NumPy's `PW_BLOCKSIZE`: above this many elements the reduction splits rather
/// than unrolling. Named rather than inlined because it is *their* constant — if
/// it ever moves, the mismatch shows up as a parity failure on long windows only,
/// and the first thing anyone reading this file will want is the number to compare.
const PW_BLOCKSIZE: usize = 128;

/// Sum `xs` in NumPy's pairwise order.
///
/// Bit-identical to `np.asarray(xs).sum()` for float64 input, which is what makes
/// every rolling feature in this crate bit-identical to `axon.features`.
///
/// # `np.sum` is not `DOUBLE_pairwise_sum`
///
/// It is **`identity + DOUBLE_pairwise_sum(...)`**. `np.add.reduce` seeds its
/// accumulator with the ufunc's identity, `0.0`, and adds the pairwise result to it.
/// That outer add is exact for every finite value — with exactly one exception:
///
/// ```text
/// 0.0 + (-0.0) == +0.0
/// ```
///
/// So a window that sums to negative zero comes back from NumPy as **positive**
/// zero, and a faithful transcription of the inner function alone returns `-0.0`.
/// One bit, on one value, on a corpus nobody has fed this yet — and a bit-exact gate
/// is precisely the thing that cannot shrug at one bit. Found by differential fuzzing
/// against NumPy over ~1.06 M cells; it was the *only* divergence class in the whole
/// sweep, which is worth saying because it is also the only one this comment can
/// claim to have looked for.
///
/// The identity is added **once, at the top**, and never inside the recursion —
/// NumPy adds it once per reduction, not once per split. Hence the split between this
/// function and [`pairwise_inner`].
///
/// Written iteratively over slices where NumPy writes pointer arithmetic; the
/// grouping — which is the only thing that affects the result — is the same.
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    // The empty sum is the identity, which is what `np.sum([])` returns.
    0.0 + pairwise_inner(xs)
}

/// NumPy's `DOUBLE_pairwise_sum`, transcribed — the recursive half, without the
/// identity. Private because on its own it is not `np.sum`; see [`pairwise_sum`].
fn pairwise_inner(xs: &[f64]) -> f64 {
    let n = xs.len();

    // Below the unroll threshold NumPy loops, so we loop. A "tidier" tree here
    // would disagree with NumPy on exactly the short windows a feature spec is
    // most likely to use.
    if n < 8 {
        let mut res = 0.0;
        for &v in xs {
            res += v;
        }
        return res;
    }

    if n <= PW_BLOCKSIZE {
        // Eight independent accumulators, seeded from the first eight elements —
        // *not* from zero, because that is how NumPy writes it. Seeding each chain
        // from zero would fold a `0.0 +` into all eight rather than into the total
        // once, which is a different answer for a window summing to negative zero.
        // (The identity NumPy *does* add is applied once, in `pairwise_sum`.)
        let mut r = [xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7]];
        let mut i = 8;
        let unrolled_end = n - (n % 8);
        while i < unrolled_end {
            for k in 0..8 {
                r[k] += xs[i + k];
            }
            i += 8;
        }

        // The combining tree is fixed and the parenthesisation is load-bearing:
        // float addition is not associative, so `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))`
        // and `r0+r1+r2+...` are different numbers.
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));

        // The tail — up to seven elements — is folded in one at a time, after the
        // tree, which is where NumPy folds it.
        while i < n {
            res += xs[i];
            i += 1;
        }
        return res;
    }

    // Divide and conquer, with the split forced onto a multiple of the unroll
    // factor so the left half still takes the unrolled path.
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_inner(&xs[..n2]) + pairwise_inner(&xs[n2..])
}

/// Mean of `xs` in NumPy's order: [`pairwise_sum`] then one division.
///
/// Bit-identical to `np.ndarray.mean` for float64. `mean` is not a separate
/// algorithm in NumPy — it is `umr_sum(...) / n` — so a compensated or streaming
/// mean here would be a second implementation of a transform, which is the exact
/// thing `docs/03` forbids, in the one place where the two languages have to agree.
///
/// Returns NaN on an empty slice rather than panicking: an empty window is a
/// warmup row, and warmup is NaN by construction across this whole crate.
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(xs) / xs.len() as f64
}

/// Population/sample standard deviation in NumPy's order.
///
/// Transcribes `numpy._core._methods._var` and then takes the square root, because
/// that is what `np.ndarray.std` is. The structure matters more than it looks:
///
/// 1. the mean is `pairwise_sum(xs) / n` — the **whole** window's mean, computed
///    first, so this is a genuine two-pass algorithm and not a `E[x²] − E[x]²`
///    shortcut. That shortcut subtracts two nearly-equal large numbers on a series
///    of perp prices and loses most of the significant digits, which is the same
///    argument ADR-0016 makes against the cumulative-sum trick;
/// 2. the deviations are materialised and squared elementwise;
/// 3. the squares are reduced with [`pairwise_sum`] again — a second pairwise
///    reduction, not a running total;
/// 4. the divisor is `n - ddof`, and only then `sqrt`.
///
/// `sqrt` is safe to take with the hardware instruction: IEEE-754 requires it to be
/// correctly rounded, so it is one of the few transcendental-looking operations
/// where "the same answer everywhere" is a guarantee rather than an observation.
/// (`ln` is not required to be correctly rounded, which is why [`crate::functions`]
/// pins that agreement with a test instead of asserting it here.)
///
/// A window shorter than `ddof + 1` yields NaN rather than dividing by zero or by a
/// negative count: a negative variance would come back from `sqrt` as NaN anyway,
/// but by a route that reads as an arithmetic accident instead of as "this window
/// is too short to describe".
pub fn std(xs: &[f64], ddof: usize) -> f64 {
    let n = xs.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    let mu = pairwise_sum(xs) / n as f64;
    // Allocation-free: the squared deviations are reduced in place through a scratch
    // buffer the caller owns in the hot path. Here the window is a slice we do not
    // own, so the deviations go through a small stack-friendly Vec — this function
    // is used by the batch path and by the streaming path's own scratch buffer, and
    // `docs/05`'s no-allocation rule is honoured by `RollingStd` in `streaming.rs`,
    // which calls `std_in` below with a buffer it allocated once.
    let mut scratch = vec![0.0f64; n];
    std_in(xs, ddof, mu, &mut scratch)
}

/// [`std()`] with the mean already computed and a caller-owned scratch buffer.
///
/// The form the streaming runtime calls, so a per-bar feature evaluation allocates
/// nothing (`docs/05`: the core does not allocate per event). `scratch` must be at
/// least `xs.len()` long; the extra tail is untouched.
pub fn std_in(xs: &[f64], ddof: usize, mean_of_xs: f64, scratch: &mut [f64]) -> f64 {
    let n = xs.len();
    if n == 0 || n <= ddof {
        return f64::NAN;
    }
    debug_assert!(scratch.len() >= n, "scratch buffer shorter than the window");
    for i in 0..n {
        let d = xs[i] - mean_of_xs;
        scratch[i] = d * d;
    }
    let ret = pairwise_sum(&scratch[..n]) / (n - ddof) as f64;
    ret.sqrt()
}

/// A deterministic series shaped like real perp closes, for the tests that have to
/// show pairwise and naive summation *disagreeing*.
///
/// Not decoration and not a random seed: a series has to have full-mantissa low bits
/// for the two orders to round differently at all. The obvious "readable" test data —
/// `60_000.0 + i * 0.1` — has such regular low bits that the naive loop and the
/// pairwise tree agree to the bit, and a separation test written on it passes for the
/// wrong reason and then never fails again. That happened once here, which is why
/// this exists. Verified against NumPy 2.5.1: over 20, 32 and 40 elements the two
/// orders differ by exactly **1 ULP**, and `np.sum` matches the pairwise side.
#[cfg(test)]
pub(crate) fn perp_series(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            60_000.0 + u * 600.0 - 300.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_test_series_is_the_one_numpy_was_measured_on() {
        // This LCG is transcribed a second time, in Python, by
        // `tests/fixtures/generate.py::perp_series` — the two have to produce the same
        // series or every reduction pinned in `cross_language.json` describes different
        // numbers from the ones tested here. Neither copy can check the other, so the
        // first three values are pinned instead: if either drifts, the separation tests
        // below go on passing while measuring something else entirely.
        let xs = perp_series(3);
        assert_eq!(xs[0], 60091.3676044132);
        assert_eq!(xs[1], 59911.49873478495);
        assert_eq!(xs[2], 59897.87109493555);
    }

    #[test]
    fn a_short_window_sums_left_to_right_because_numpy_does() {
        // Below eight elements NumPy runs the plain loop, so the "clever" tree must
        // not kick in early. If it did, every 5-bar momentum window in `BAR_M1_V1`
        // would disagree with Python by a ULP or two — and only the short windows
        // would, which is the hardest kind of parity failure to read.
        let xs = [1e16, 1.0, -1e16, 1.0, 1.0];
        let mut expect = 0.0;
        for v in xs {
            expect += v;
        }
        assert_eq!(pairwise_sum(&xs).to_bits(), expect.to_bits());
    }

    #[test]
    fn eight_or_more_does_not_sum_left_to_right() {
        // The negative of the test above: this is the case where the naive loop and
        // NumPy genuinely differ, so a transcription that quietly fell back to the
        // loop would still pass every "is it approximately right" check.
        //
        // The delta is asserted as an exact ULP count rather than as "not equal",
        // because "they differ" stays true if the transcription breaks in some third
        // way. One ULP is what NumPy 2.5.1 was measured at on this series.
        let xs = perp_series(20);
        let mut naive = 0.0;
        for &v in &xs {
            naive += v;
        }
        let pw = pairwise_sum(&xs);
        assert_ne!(
            pw.to_bits(),
            naive.to_bits(),
            "pairwise and naive agreed on a series built to separate them; the \
             transcription is not exercising the unrolled path"
        );
        let ulps = (pw.to_bits() as i64 - naive.to_bits() as i64).abs();
        assert_eq!(ulps, 1, "the measured separation moved: {pw} vs {naive}");
    }

    #[test]
    fn the_split_above_the_blocksize_lands_on_a_multiple_of_the_unroll_factor() {
        // A split at an arbitrary midpoint gives a different (still plausible)
        // answer, so this test pins the shape: the two halves must be the ones NumPy
        // would take.
        //
        // It runs on `perp_series` and not on readable data, and that is the whole
        // point of the test. Its first version used `((i % 7) as f64) * 1.5 - 3.0`,
        // whose every partial sum is an exact multiple of 0.5 far below 2^53 — so
        // *every* grouping gives identical bits, including no split at all, and the
        // test passed with `n2 -= n2 % 8` deleted. It was guarding the one branch of
        // `pairwise_inner` that no committed bundle reaches (every shipped window is
        // ≤ 20) and it could not fail. Exactly the trap `perp_series`' own docstring
        // warns about, committed anyway.
        let xs = perp_series(300);
        let n = xs.len();
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        let expect = pairwise_inner(&xs[..n2]) + pairwise_inner(&xs[n2..]);
        assert_eq!(pairwise_sum(&xs).to_bits(), (0.0 + expect).to_bits());

        // …and a mis-aligned split has to be *observable*, or the assertion above is a
        // restatement rather than a check. This is where the first version of this
        // test went wrong twice over, and both are worth recording.
        //
        // Perp closes will not do it. Pairwise summation over 300 values of similar
        // magnitude is so well conditioned that **every** split — 144, 148, 150, 151 —
        // lands on identical bits, so a test built on `perp_series` alone passes with
        // the alignment deleted. The grouping only becomes visible when the summands
        // span decades, which is what this series is for and the only thing it is for.
        // Alternating ±1e8, offset by a value far below the sum's resolution: the
        // classic cancellation shape, and the one where *which* elements are grouped
        // together decides how much of the offset survives. Chosen — measured, not
        // guessed — so that the split a naive `n / 2` would take is one of the ones
        // that disagrees, because `n / 2` is precisely the mutation this test exists
        // to catch.
        let wide: Vec<f64> = (0..300)
            .map(|i| if i % 2 == 1 { 1e8 } else { -1e8 + 1e-6 })
            .collect();
        let m = wide.len();
        let mut m2 = m / 2;
        m2 -= m2 % 8;
        let aligned = pairwise_inner(&wide[..m2]) + pairwise_inner(&wide[m2..]);
        assert_eq!(pairwise_sum(&wide).to_bits(), (0.0 + aligned).to_bits());
        let observable = [m2 - 4, m2 + 4, m2 + 1, m / 2]
            .into_iter()
            .filter(|w| *w != m2 && *w > 0 && *w < m)
            .any(|w| {
                (pairwise_inner(&wide[..w]) + pairwise_inner(&wide[w..])).to_bits()
                    != aligned.to_bits()
            });
        assert!(
            observable,
            "no mis-aligned split produced different bits on a series built to cancel, \
             so the alignment is unobservable and this test is asleep"
        );
        // And specifically the split a naive implementation takes, since that is the
        // mutation. Asserted separately from `observable` so a failure says which.
        let naive_half = m / 2;
        assert_ne!(naive_half, m2, "the two splits coincide at this length");
        assert_ne!(
            (pairwise_inner(&wide[..naive_half]) + pairwise_inner(&wide[naive_half..])).to_bits(),
            aligned.to_bits(),
            "splitting at n/2 instead of the unroll-aligned {m2} gave identical bits"
        );
    }

    #[test]
    fn a_window_holding_nan_reduces_to_nan_rather_than_skipping_it() {
        // `realized_volatility` reduces a return series whose first element is NaN
        // by construction. A reduction that skipped non-finite values would emit a
        // finite volatility for a window that does not have one, and the warmup
        // rows would silently become real readings.
        let xs = [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!(pairwise_sum(&xs).is_nan());
        assert!(mean(&xs).is_nan());
        assert!(std(&xs, 0).is_nan());
    }

    #[test]
    fn a_window_too_short_for_its_ddof_is_nan_not_a_division_by_zero() {
        assert!(std(&[1.0], 1).is_nan());
        assert!(std(&[], 0).is_nan());
        // ddof = 0 on a single sample is a legitimate zero, not a warmup.
        assert_eq!(std(&[4.0], 0), 0.0);
    }

    #[test]
    fn std_is_two_pass_so_a_large_offset_does_not_cancel_the_signal() {
        // The `E[x²] - E[x]²` shortcut returns 0.0 here — the mean square and the
        // square of the mean agree to every bit float64 has left at 1e8. Two-pass
        // keeps the answer. This is the failure the docstring's step 1 describes,
        // and it is why the mean is computed over the whole window first.
        let xs: Vec<f64> = (0..16).map(|i| 1e8 + (i % 2) as f64).collect();
        let s = std(&xs, 0);
        assert!((s - 0.5).abs() < 1e-9, "two-pass std lost the signal: {s}");
    }

    #[test]
    fn std_in_agrees_with_std_so_the_streaming_path_is_not_a_second_formula() {
        let xs: Vec<f64> = (0..37).map(|i| ((i * 13 % 29) as f64) * 0.375).collect();
        let mu = pairwise_sum(&xs) / xs.len() as f64;
        let mut scratch = vec![0.0; xs.len()];
        assert_eq!(
            std(&xs, 0).to_bits(),
            std_in(&xs, 0, mu, &mut scratch).to_bits()
        );
    }
}
