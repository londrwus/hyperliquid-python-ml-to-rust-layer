//! The claim the streaming runtime exists to make: **a bounded buffer reproduces the
//! batch path bit for bit, and only because every window is finite.**
//!
//! [ADR-0022] makes that claim for `perp_bar` and [ADR-0032] for `BAR_M1_V1` — "nothing
//! here is an EMA or an expanding statistic, which is what lets a bounded serving buffer
//! reproduce the offline recompute *bit for bit* rather than within a tolerance" — and
//! until this crate existed nothing in Rust could check it. `spec.rs` proves the two
//! paths agree about the *recipe*; `tests/cross_language.rs` proves Rust and Python agree
//! about the recipe and the arithmetic. This file proves the research path and the
//! serving path agree about the **numbers**, which is the only one of the three a live
//! core actually depends on.
//!
//! Four things are measured here, in the order they matter:
//!
//! 1. **Every cell agrees.** 400 bars of real-shaped OHLC through
//!    `FeatureSpec::compute`, then the same 400 pushed one at a time through
//!    `FeatureStream`, compared cell by cell with the count asserted — a comparison
//!    that silently compared nothing is the easiest way to pass a test like this.
//! 2. **The bound is tight.** 21 is derived, not decorative: a buffer of 20 is refused,
//!    and the refusal is worth having because a 20-deep window is *silently* dead
//!    rather than wrong — its last row is NaN forever and nothing raises.
//! 3. **The refusal is real.** `PERP_CORE_V1`, the repo's own reference perp spec, is
//!    unservable from a bounded Rust buffer, and the measurement of how wrong serving
//!    it anyway would be is here rather than in a docstring.
//! 4. **Warmup is NaN**, per column, returned rather than withheld.
//!
//! ## The NaN rule, spelled out
//!
//! `f64::NAN == f64::NAN` is false, and two NaNs may have different `to_bits()` — a NaN
//! carries a sign bit and a 51-bit payload, and `0.0/0.0` and `-(0.0/0.0)` are both NaN
//! with different patterns. So every comparison below checks `is_nan()` **first** and
//! only then bits. The split is ADR-0016 §4's, and it is deliberate in both directions:
//! **both NaN is a match** (warmup is legitimately NaN on both paths and would otherwise
//! fail every run) and **one NaN is a mismatch** (a feature that goes NaN on the serving
//! path and finite offline is precisely the staleness bug the gate exists to catch).
//!
//! [ADR-0022]: ../../../docs/adr/0022-perp-bar-strategy.md
//! [ADR-0032]: ../../../docs/adr/0032-model-zoo.md

use std::collections::BTreeMap;

use axon_features::{FeatureError, FeatureMatrix, FeatureSpec, FeatureStream};

// ── the specs, as literals ────────────────────────────────────────────────────
//
// Both carry the fingerprint Python wrote, so `from_json` re-identifies them rather
// than taking the transcription on trust: if these bytes were not the spec Python
// means by `BAR_M1_V1`, the load fails here instead of this file quietly measuring a
// recipe of its own.

/// `axon.features.spec.BAR_M1_V1.to_json()`.
const BAR_M1_JSON: &str = r#"{"features":[{"column":"ret_1","feature":"log_return","inputs":{"price":"close"},"params":{"period":1}},{"column":"mom_5","feature":"momentum","inputs":{"price":"close"},"params":{"window":5}},{"column":"z_20","feature":"rolling_zscore","inputs":{"x":"close"},"params":{"window":20}},{"column":"vol_20","feature":"realized_volatility","inputs":{"price":"close"},"params":{"window":20}},{"column":"range_bps","feature":"relative_range","inputs":{},"params":{}},{"column":"clv","feature":"close_location","inputs":{},"params":{}}],"fingerprint":"c503688de24e863f","library_version":1,"spec":"bar_m1","version":1}"#;

/// `axon.features.spec.PERP_CORE_V1.to_json()`.
const PERP_CORE_JSON: &str = r#"{"features":[{"column":"mid","feature":"mid_price","inputs":{},"params":{}},{"column":"spread_bps","feature":"relative_spread","inputs":{},"params":{}},{"column":"book_imb","feature":"book_imbalance","inputs":{},"params":{}},{"column":"ret_1","feature":"log_return","inputs":{"price":"mid"},"params":{"period":1}},{"column":"mom_32","feature":"momentum","inputs":{"price":"mid"},"params":{"window":32}},{"column":"vol_32","feature":"realized_volatility","inputs":{"price":"mid"},"params":{"window":32}},{"column":"ema_x_8_32","feature":"ema_crossover","inputs":{"price":"mid"},"params":{"fast":8,"slow":32}},{"column":"z_32","feature":"rolling_zscore","inputs":{"x":"mid"},"params":{"window":32}},{"column":"tfi_32","feature":"trade_flow_imbalance","inputs":{},"params":{"window":32}}],"fingerprint":"868d3dbe95d4b386","library_version":1,"spec":"perp_core","version":1}"#;

fn bar_spec() -> FeatureSpec {
    FeatureSpec::from_json(BAR_M1_JSON).expect("BAR_M1_V1 must load and re-identify")
}

// ── the data ──────────────────────────────────────────────────────────────────

/// Bars over which the two paths are compared.
const BARS: usize = 400;

/// Every 47th bar from index 30 trades at one price for the whole minute.
///
/// Not decoration. `close_location` is NaN for a bar with no range, and **6 of 58 BTC
/// and 5 of 62 ETH live m1 bars** hit that in a measured testnet session, so a
/// comparison over a series that never goes flat would leave the entire NaN half of the
/// comparison rule — the half that says both-NaN is a match — untested outside the
/// warmup block, where every column is NaN together and a bug cannot show.
const FLAT_EVERY: usize = 47;
const FIRST_FLAT: usize = 30;

/// Deterministic OHLC bars shaped like real perp candles.
///
/// The generator is the LCG `numeric.rs` uses, and for the reason that file gives: a
/// "readable" series like `60_000.0 + i * 0.1` has such regular low bits that two
/// different reduction *groupings* agree to the bit, and a comparison built on it would
/// pass while measuring nothing. This series has full-mantissa low bits, so a streaming
/// path that summed its window in a different order — which is exactly what a ring
/// buffer or an incremental running total would do — separates from the batch path here.
fn bars(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut close, mut high, mut low) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let c = 60_000.0 + next() * 600.0 - 300.0;
        let (up, down) = (next() * 30.0, next() * 30.0);
        let flat = i >= FIRST_FLAT && (i - FIRST_FLAT) % FLAT_EVERY == 0;
        close.push(c);
        high.push(if flat { c } else { c + up });
        low.push(if flat { c } else { c - down });
    }
    (close, high, low)
}

fn columns(close: &[f64], high: &[f64], low: &[f64]) -> BTreeMap<String, Vec<f64>> {
    BTreeMap::from([
        ("close".to_string(), close.to_vec()),
        ("high".to_string(), high.to_vec()),
        ("low".to_string(), low.to_vec()),
    ])
}

fn observation(close: f64, high: f64, low: f64) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("close".to_string(), close),
        ("high".to_string(), high),
        ("low".to_string(), low),
    ])
}

/// Push a whole series through a stream and keep every row, row-major.
fn stream_all(stream: &mut FeatureStream, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(close.len() * stream.columns().len());
    for i in 0..close.len() {
        let row = stream
            .push(&observation(close[i], high[i], low[i]))
            .unwrap_or_else(|e| panic!("push {i}: {e}"));
        out.extend_from_slice(row);
    }
    out
}

/// The result of a cell-by-cell comparison, counted rather than asserted away.
struct Tally {
    compared: usize,
    agreed: usize,
    /// Cells that were NaN on both sides — a match under ADR-0016 §4, and counted
    /// separately because a comparison where *every* cell is a NaN pair has proved
    /// nothing about the arithmetic.
    nan_pairs: usize,
    /// Cells that were finite and bit-identical.
    exact_pairs: usize,
    mismatches: Vec<String>,
}

fn compare(batch: &FeatureMatrix, streamed: &[f64]) -> Tally {
    let width = batch.cols();
    assert_eq!(
        streamed.len(),
        batch.rows() * width,
        "the streamed matrix is a different shape from the batch one"
    );
    let mut t = Tally {
        compared: 0,
        agreed: 0,
        nan_pairs: 0,
        exact_pairs: 0,
        mismatches: Vec::new(),
    };
    for i in 0..batch.rows() {
        for j in 0..width {
            let b = batch.get(i, j);
            let s = streamed[i * width + j];
            t.compared += 1;
            // `is_nan()` first: `NAN == NAN` is false and two NaNs need not share a bit
            // pattern, so a bits-only comparison would report every warmup cell as a
            // mismatch, and an equality-only comparison would report them as one too.
            if b.is_nan() || s.is_nan() {
                if b.is_nan() && s.is_nan() {
                    t.agreed += 1;
                    t.nan_pairs += 1;
                } else {
                    t.mismatches.push(format!(
                        "row {i} column {:?}: batch {b} streamed {s} (one side NaN)",
                        batch.columns()[j]
                    ));
                }
            } else if b.to_bits() == s.to_bits() {
                t.agreed += 1;
                t.exact_pairs += 1;
            } else {
                t.mismatches.push(format!(
                    "row {i} column {:?}: batch {b} (0x{:016x}) streamed {s} (0x{:016x}), {} ULP",
                    batch.columns()[j],
                    b.to_bits(),
                    s.to_bits(),
                    (b.to_bits() as i64 - s.to_bits() as i64).abs()
                ));
            }
        }
    }
    t
}

// ── 1. the bit-for-bit comparison ─────────────────────────────────────────────

#[test]
fn every_cell_of_the_streamed_matrix_matches_the_batch_matrix_to_the_bit() {
    let spec = bar_spec();
    let (close, high, low) = bars(BARS);
    let batch = spec.compute(&columns(&close, &high, &low)).unwrap();
    let mut stream = FeatureStream::new(&spec).unwrap();
    let streamed = stream_all(&mut stream, &close, &high, &low);

    let t = compare(&batch, &streamed);
    assert!(
        t.mismatches.is_empty(),
        "{} of {} cells disagree; first three:\n  {}",
        t.mismatches.len(),
        t.compared,
        t.mismatches[..t.mismatches.len().min(3)].join("\n  ")
    );

    // The counts, asserted as numbers rather than as "no mismatches" — a comparison
    // that ran over an empty matrix, or over a matrix of nothing but NaN, would satisfy
    // the assertion above and prove nothing at all.
    assert_eq!(t.compared, BARS * 6, "400 bars × 6 columns");
    assert_eq!(t.compared, 2_400);
    assert_eq!(t.agreed, 2_400);

    // 53 NaN cells, and every one of them is accounted for: ret_1 is NaN on row 0 (1),
    // mom_5 through row 4 (5), z_20 through row 18 (19), vol_20 through row 19 (20 — a
    // 20-sample deviation *of one-step returns* reaches through the return's own extra
    // bar), and clv on each of the 8 flat bars. range_bps is finite from row 0: a bar
    // with no range has a range of exactly 0.0 bps, which is a reading rather than an
    // absence.
    assert_eq!(t.nan_pairs, 1 + 5 + 19 + 20 + 8);
    assert_eq!(t.nan_pairs, 53);
    assert_eq!(t.exact_pairs, 2_400 - 53);
    assert_eq!(batch.nan_cells(), t.nan_pairs);

    // …and the NaN half of the rule is exercised *outside* the warmup block, where a
    // single column goes NaN on its own while its five neighbours stay finite. Inside
    // warmup every column is NaN together, and a comparison that only ever saw that
    // could not tell a correct NaN from a row the streaming path failed to compute.
    let late_nans = (21..BARS)
        .flat_map(|i| (0..6).map(move |j| (i, j)))
        .filter(|(i, j)| batch.get(*i, *j).is_nan())
        .count();
    assert_eq!(late_nans, 8, "the flat bars did not reach the comparison");

    // The stream saw every bar and ends warm, so the agreement above is not the
    // agreement of two paths that both stopped early.
    assert_eq!(stream.observed(), BARS);
    assert!(stream.is_warm());
    assert_eq!(batch.finite_rows(), BARS - 20 - 8);
}

#[test]
fn the_streamed_row_is_the_batch_row_at_that_index_and_not_the_previous_one() {
    // An off-by-one in the buffer — emitting the row for the observation *before* the
    // one just pushed — passes every count in the test above and is the single most
    // damaging bug this module could have: it is a one-bar lookahead in the other
    // direction, and a strategy fed it would trade on stale features while the parity
    // gate's row counts stayed perfect. The check is that the diagonal agrees and the
    // off-diagonal does not.
    let spec = bar_spec();
    let (close, high, low) = bars(60);
    let batch = spec.compute(&columns(&close, &high, &low)).unwrap();
    let mut stream = FeatureStream::new(&spec).unwrap();
    let streamed = stream_all(&mut stream, &close, &high, &low);

    let ret_1 = 0; // the column that moves most between adjacent bars
    let shifted = (25..60)
        .filter(|i| {
            let now = streamed[i * 6 + ret_1];
            let before = batch.get(i - 1, ret_1);
            now.to_bits() == before.to_bits()
        })
        .count();
    assert_eq!(
        shifted, 0,
        "{shifted} streamed rows equal the *previous* batch row; the buffer is one \
         observation behind its own event"
    );
}

// ── 2. the bound is tight ─────────────────────────────────────────────────────

#[test]
fn a_buffer_one_short_of_the_derived_lookback_is_refused_rather_than_never_warming() {
    let spec = bar_spec();
    let derived = spec.max_lookback().unwrap().unwrap();
    assert_eq!(derived, 21, "the derivation moved");

    match FeatureStream::with_lookback(&spec, derived - 1) {
        Err(FeatureError::Spec(msg)) => {
            assert!(msg.contains("20"), "the message must name the depth: {msg}");
            assert!(
                msg.contains("21"),
                "the message must name the derivation: {msg}"
            );
        }
        other => panic!("a 20-observation buffer was accepted: {other:?}"),
    }

    // Why that refusal is worth having, measured rather than asserted: a 20-deep buffer
    // is not *wrong*, it is silently dead. The batch path over the trailing 20
    // observations produces a last row that is not finite — `vol_20`'s window never
    // fills — and it would go on doing that for every bar of the session while the
    // process looked healthy, the rows kept arriving, and nothing raised.
    let (close, high, low) = bars(BARS);
    let short = spec
        .compute(&columns(
            &close[BARS - 20..],
            &high[BARS - 20..],
            &low[BARS - 20..],
        ))
        .unwrap();
    assert!(
        !axon_features::functions::row_is_finite(short.row(19)),
        "a 20-observation window produced a finite last row; the bound is not tight"
    );
    assert!(
        short.row(19)[3].is_nan(),
        "vol_20 was expected to be the cell that starves"
    );
    assert_eq!(short.finite_rows(), 0);

    // One more observation and the same window is not merely finite but *exactly* the
    // batch value at that index — which is the whole claim, at its tightest point.
    let full = spec.compute(&columns(&close, &high, &low)).unwrap();
    let exactly = spec
        .compute(&columns(
            &close[BARS - 21..],
            &high[BARS - 21..],
            &low[BARS - 21..],
        ))
        .unwrap();
    assert!(axon_features::functions::row_is_finite(exactly.row(20)));
    for j in 0..6 {
        assert_eq!(
            exactly.get(20, j).to_bits(),
            full.get(BARS - 1, j).to_bits(),
            "column {:?} differs between a 21-observation window and the full history",
            full.columns()[j]
        );
    }
}

#[test]
fn a_deeper_buffer_than_the_derivation_changes_no_value_which_is_what_bounded_means() {
    // The other side of tightness. If a deeper buffer changed an answer, "bounded by
    // 21" would be false and the whole bit-for-bit claim would be an artefact of the
    // depth that happened to be chosen. 21 and 64 must agree on every cell.
    let spec = bar_spec();
    let (close, high, low) = bars(BARS);
    let batch = spec.compute(&columns(&close, &high, &low)).unwrap();

    let mut deep = FeatureStream::with_lookback(&spec, 64).unwrap();
    assert_eq!(deep.lookback(), 64);
    let streamed = stream_all(&mut deep, &close, &high, &low);

    let t = compare(&batch, &streamed);
    assert!(
        t.mismatches.is_empty(),
        "{:?}",
        &t.mismatches[..1.min(t.mismatches.len())]
    );
    assert_eq!(t.compared, 2_400);
    assert_eq!(t.agreed, 2_400);
    assert_eq!(t.nan_pairs, 53);
}

// ── 3. the refusal, and what it is protecting against ─────────────────────────

#[test]
fn the_reference_perp_spec_is_unservable_from_a_bounded_buffer_and_names_the_column() {
    // A finding about the repo's own reference spec, not an edge case. `PERP_CORE_V1`
    // is nine columns, eight of them bounded, and the ninth costs the whole spec its
    // place on a Rust serving path. Recorded here so that the day somebody swaps
    // `ema_crossover` for `sma_crossover`, the change in what is servable is a visible
    // diff rather than an incidental one.
    let perp = FeatureSpec::from_json(PERP_CORE_JSON).expect("PERP_CORE_V1 must re-identify");
    assert_eq!(perp.reference(), "perp_core/v1#868d3dbe95d4b386");

    match FeatureStream::new(&perp) {
        Err(FeatureError::Spec(msg)) => {
            assert!(msg.contains("ema_x_8_32"), "the column is not named: {msg}");
            assert!(msg.contains("perp_core"), "the spec is not named: {msg}");
        }
        other => panic!("the reference perp spec was accepted for streaming: {other:?}"),
    }

    // The batch path computes it faithfully, which is the point of refusing it *here*
    // rather than deleting the transform: `PERP_CORE_V1` remains a legitimate research
    // spec, and the same nine columns can be computed offline all day.
    let n = 64;
    let (close, _, _) = bars(n);
    let inputs = BTreeMap::from([
        (
            "bid_px".to_string(),
            close.iter().map(|c| c - 0.5).collect(),
        ),
        (
            "ask_px".to_string(),
            close.iter().map(|c| c + 0.5).collect(),
        ),
        ("bid_sz".to_string(), vec![3.0; n]),
        ("ask_sz".to_string(), vec![1.0; n]),
        ("trade_sz".to_string(), vec![0.5; n]),
        ("trade_sign".to_string(), vec![1.0; n]),
    ]);
    let matrix = perp.compute(&inputs).unwrap();
    assert_eq!(matrix.rows(), n);
    assert!(
        matrix.finite_rows() > 0,
        "the batch path must still serve research"
    );
}

#[test]
fn a_bounded_ema_still_carries_its_seed_after_the_buffer_has_turned_over_many_times() {
    // `functions.rs::an_ema_never_forgets_its_seed_which_is_why_streaming_refuses_it`
    // established that the two answers *differ*. What it did not say is by how much, and
    // "the bits differ" is a weak argument for refusing a spec — a one-ULP gap would be
    // a rounding curiosity. This measures the gap a 21-deep buffer would actually have
    // produced on 400 bars of perp closes, which is the number the refusal is worth.
    let ema_json = r#"{"features":[{"column":"e8","feature":"ema","inputs":{"x":"close"},"params":{"span":8}}],"library_version":1,"spec":"t","version":1}"#;
    let spec = FeatureSpec::from_json(ema_json).unwrap();
    assert_eq!(spec.max_lookback().unwrap(), None);

    let (close, _, _) = bars(BARS);
    let full = spec
        .compute(&BTreeMap::from([("close".to_string(), close.clone())]))
        .unwrap();
    // What a 21-deep buffer would have computed for that same bar: 379 observations of
    // history discarded, 18 full turnovers of the buffer. The weight an 8-span EMA still
    // puts on its seed after 21 observations is `(1 - 2/9)^20 ≈ 0.66%` — geometric, so it
    // shrinks forever and reaches zero never, which is what "unbounded lookback" means
    // arithmetically rather than as a label.
    let bounded = spec
        .compute(&BTreeMap::from([(
            "close".to_string(),
            close[BARS - 21..].to_vec(),
        )]))
        .unwrap();

    let (a, b) = (full.get(BARS - 1, 0), bounded.get(20, 0));
    assert!(a.is_finite() && b.is_finite());
    assert_ne!(
        a.to_bits(),
        b.to_bits(),
        "the bounded EMA matched the full history"
    );

    // Measured on this series: the full-history EMA reads 60083.8878 and the bounded one
    // 60082.7640 — **1.12 price units, −0.1870 bps, 1.5445e11 ULP** apart. That is what
    // the refusal is worth, and it is why "the bits differ" was not a good enough
    // argument on its own: a one-ULP gap would be a rounding curiosity, while a fifth of
    // a basis point is the same order as the m1 bar-to-bar return this feature is meant
    // to be reading. It is also *systematic* — it does not average out, it is widest
    // right after a restart, and a parity gate meeting it on healthy data would report it
    // as "the Rust transforms are wrong".
    let gap_bps = (b - a) / a * 10_000.0;
    assert!(
        (gap_bps + 0.187_035).abs() < 1e-6,
        "the measured gap moved to {gap_bps} bps; the number in this comment is stale"
    );
    // The exact ULP count, pinned rather than bounded: an EMA is `+ - *` only and the
    // series comes out of integer arithmetic, so this is reproducible to the bit on any
    // IEEE-754 platform. A change here is a change in the arithmetic, not the weather.
    let ulps = (a.to_bits() as i64 - b.to_bits() as i64).abs();
    assert_eq!(ulps, 154_450_955_866);
}

// ── 4. warmup ─────────────────────────────────────────────────────────────────

#[test]
fn a_warmup_row_is_returned_with_nan_cells_rather_than_withheld_or_zero_filled() {
    // Zero is a legal value for every column here — `clv` is exactly 0.0 for a bar that
    // closed dead centre in its range and `range_bps` is exactly 0.0 for a bar that did
    // not move — so a zero-filled warmup is indistinguishable from twenty real
    // readings, and the model learns from it. And the row is *returned*: warmup is per
    // column, so bar 1 already carries two usable cells, and a runtime that withheld
    // the row until every column was finite would throw those away and leave a caller
    // unable to tell "warming up" from "the core is wedged".
    let spec = bar_spec();
    let mut stream = FeatureStream::new(&spec).unwrap();
    let (close, high, low) = bars(BARS);

    let first = stream
        .push(&observation(close[0], high[0], low[0]))
        .expect("a warmup row is returned, not refused")
        .to_vec();
    assert_eq!(first.len(), 6);
    assert_eq!(
        first.iter().filter(|v| v.is_nan()).count(),
        4,
        "expected ret_1, mom_5, z_20 and vol_20 to be NaN on the first bar: {first:?}"
    );
    assert!(
        first[4].is_finite(),
        "range_bps is pointwise and finite on bar 1"
    );
    assert!(first[5].is_finite(), "clv is pointwise and finite on bar 1");
    assert!(!stream.is_warm());

    // Not one zero anywhere in the warmup block, in any column that has not warmed.
    // This is the assertion that reddens if a buffer is ever pre-filled with zeros
    // instead of NaN.
    let mut warmup_cells = 0;
    let mut warmup_nans = 0;
    for i in 1..21 {
        let row = stream
            .push(&observation(close[i], high[i], low[i]))
            .unwrap();
        for (j, v) in row.iter().enumerate() {
            if j < 4 {
                warmup_cells += 1;
                if v.is_nan() {
                    warmup_nans += 1;
                } else {
                    assert_ne!(*v, 0.0, "row {i} column {j} zero-filled its warmup");
                }
            }
        }
    }
    assert_eq!(warmup_cells, 80, "20 rows × the 4 windowed columns");
    // 1 + 5 + 19 + 20 NaN cells in total, less the 4 on row 0 which is not in this loop.
    assert_eq!(warmup_nans, 45 - 4);

    // The 21st bar is the first fully finite row — the derived warmup, reached by
    // pushing rather than by computing.
    assert_eq!(stream.observed(), 21);
    assert!(
        stream.is_warm(),
        "the stream was not warm after its derived lookback of {} observations",
        stream.lookback()
    );
}
