//! The cross-language feature-parity gate, from the Rust side.
//!
//! `src/parity.rs` reads a *feature parity bundle* — a frozen question written by
//! `axon.parity.feature_bundle` out of a `FeatureSpec`, a set of input arrays and
//! Python's own matrix over exactly those bytes — and answers it with no Python, no
//! numpy, no network and no clock. This file is what holds that reader to its claims.
//!
//! The tests come in two halves, and they prove different things.
//!
//! **The committed bundles under `tests/bundles/` are the cross-language claim.** They
//! were written by `axon.parity.feature_bundle` from real Hyperliquid market data — two
//! 900-bar mainnet bar corpora, 58 bars that crossed a live testnet socket, 675
//! market-data slices off a *recorded order book and tape*, and one 18-column spec
//! exercising every registered transform — and they are in git, because regenerating a
//! reference in the same breath as asserting it proves nothing. The matrices in them are
//! NumPy's. Nothing in this crate could have produced them, and
//! `rust_reproduces_pythons_matrix_for_every_cell_of_every_committed_bundle` asserts
//! this build reproduces all **33 423 cells to the bit**. That is the half of Boundary A
//! ADR-0021 could not close.
//!
//! **The bundles built in `CARGO_TARGET_TMPDIR` are how the gate is shown to fail.** A
//! frozen fixture cannot be perturbed in the tree, so the failure paths — one ULP, a NaN
//! on one side, a loosened criterion, a truncated matrix, an edited spec — are exercised
//! against bundles this file writes itself. Those are Rust answering Rust, which is
//! exactly wrong for proving agreement and exactly right for proving the gate can go
//! red. The two halves need each other: the first would pass forever if the comparison
//! were `assert!(true)`, and the second cannot see across the language boundary at all.
//!
//! The most important test in the file is
//! `one_ulp_of_drift_in_pythons_own_matrix_reddens_the_gate`. A gate nobody has seen
//! fail is a decoration: it moves one cell of a *committed* NumPy matrix by one ULP —
//! 2.220446049250313e-16 on a z-score near 1 — and the verdict flips. That is the size
//! of the disagreement an incremental rolling mean, a `E[x²] − E[x]²` variance or a naive
//! summation order would introduce (`src/numeric.rs` measures the last one per window,
//! and the same mutation moves 2 689 cells of the committed corpora), and none of them is
//! visible to any check but this one.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use axon_features::parity::{BundleError, Criterion, FeatureBundle};
use axon_features::{registered_features, FeatureMatrix, FeatureSpec};

/// `BAR_M1_V1`, canonically serialized — byte-identical to what `FeatureSpec.to_json()`
/// writes, which is what a real bundle carries in `spec.json`.
///
/// Committed as a literal so this file needs no fixture. `tests/cross_language.rs` is
/// what checks the transcription against Python's own bytes; here it only has to be
/// something `FeatureSpec::from_json` re-identifies, since the fingerprint is recomputed
/// on every load and a wrong literal would fail immediately rather than quietly.
const BAR_M1_JSON: &str = r#"{"features":[{"column":"ret_1","feature":"log_return","inputs":{"price":"close"},"params":{"period":1}},{"column":"mom_5","feature":"momentum","inputs":{"price":"close"},"params":{"window":5}},{"column":"z_20","feature":"rolling_zscore","inputs":{"x":"close"},"params":{"window":20}},{"column":"vol_20","feature":"realized_volatility","inputs":{"price":"close"},"params":{"window":20}},{"column":"range_bps","feature":"relative_range","inputs":{},"params":{}},{"column":"clv","feature":"close_location","inputs":{},"params":{}}],"fingerprint":"c503688de24e863f","library_version":1,"spec":"bar_m1","version":1}"#;

/// The columns of `BAR_M1_V1` whose value passes through `log`, as
/// `axon.parity.feature_bundle.libm_columns` derives them: `log_return`, `momentum`
/// (which *is* a log return over a longer horizon) and `realized_volatility` (a rolling
/// deviation *of* log returns). `z_20` is not one of them — it reads `close` directly.
const LIBM_COLUMNS: [&str; 3] = ["ret_1", "mom_5", "vol_20"];

const ROWS: usize = 300;
const COLS: usize = 6;

/// Warmup, counted rather than restated: `ret_1` 1 + `mom_5` 5 + `z_20` 19 + `vol_20`
/// 20. `range_bps` and `clv` are pointwise and finite on every row of this corpus.
const NAN_CELLS: usize = 45;

/// The first fully finite row of `BAR_M1_V1` is index 20 — the derived warmup of 21 bars
/// less one — so 280 of 300 rows are usable.
const FINITE_ROWS: usize = ROWS - 20;

// ── building a bundle ─────────────────────────────────────────────────────────

/// A deterministic series shaped like real perp closes.
///
/// The same LCG `numeric::perp_series` uses, transcribed rather than imported because
/// that one is `#[cfg(test)] pub(crate)` and an integration test links the library as an
/// outside consumer does. It matters that the low bits are full: a "readable" series
/// like `60_000.0 + i * 0.1` has such regular mantissas that a one-ULP perturbation of a
/// z-score computed from it can land on a value the other side also produces.
fn perp_series(n: usize) -> Vec<f64> {
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

fn bar_inputs(n: usize) -> BTreeMap<String, Vec<f64>> {
    let close = perp_series(n);
    let high: Vec<f64> = close.iter().map(|c| c + 5.0).collect();
    let low: Vec<f64> = close.iter().map(|c| c - 5.0).collect();
    BTreeMap::from([
        ("close".to_string(), close),
        ("high".to_string(), high),
        ("low".to_string(), low),
    ])
}

fn bar_spec() -> FeatureSpec {
    FeatureSpec::from_json(BAR_M1_JSON).expect("the committed spec literal re-identifies")
}

/// `axon.models.artifact.content_hash`'s spelling — `sha256:` and 64 hex digits.
///
/// Computed for real even though [`FeatureBundle::open`] never reads it, so that a
/// fixture built here is a bundle in every field the format defines rather than in the
/// subset this reader happens to consume.
/// `the_reader_does_not_verify_the_hashes_the_manifest_records` is what turns that into
/// a checked property instead of a convention.
fn content_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::from("sha256:");
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn le_bytes(values: impl IntoIterator<Item = f64>) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn json_list(items: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let quoted: Vec<String> = items
        .into_iter()
        .map(|s| format!("{:?}", s.as_ref()))
        .collect();
    format!("[{}]", quoted.join(","))
}

/// Write a bundle the pinned format describes, into `CARGO_TARGET_TMPDIR`.
///
/// The manifest is *derived* from the matrices it describes rather than passed in, so a
/// test that perturbs a cell gets a self-consistent bundle for free and the perturbation
/// surfaces as a parity failure rather than as a malformed fixture. The tests that need
/// an inconsistent manifest break it afterwards, through [`rewrite`], which asserts it
/// changed the bytes.
///
/// Under `CARGO_TARGET_TMPDIR` for the reason the model gate's scratch bundles are: a
/// failed run leaves the evidence in `target/` rather than in the source tree, where the
/// next `cargo test` would gate it.
fn write_bundle(
    scratch: &str,
    spec_json: &str,
    inputs: &BTreeMap<String, Vec<f64>>,
    features: &FeatureMatrix,
) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(scratch);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");

    let names: Vec<&str> = inputs.keys().map(String::as_str).collect();
    let rows = features.rows();
    // Row-major, little-endian: element `(i, j)` of the inputs matrix is input `j` at
    // observation `i`, which is what the reader de-interleaves back into named arrays.
    let inputs_bytes = le_bytes((0..rows).flat_map(|i| names.iter().map(move |n| inputs[*n][i])));
    let features_bytes = le_bytes(features.as_slice().iter().copied());
    let spec_bytes = spec_json.as_bytes();

    fs::write(dir.join("inputs.f64"), &inputs_bytes).unwrap();
    fs::write(dir.join("features.f64"), &features_bytes).unwrap();
    fs::write(dir.join("spec.json"), spec_bytes).unwrap();

    let spec = FeatureSpec::from_json(spec_json).expect("the fixture spec re-identifies");
    let input_nans = inputs
        .values()
        .flat_map(|c| c.iter())
        .filter(|v| v.is_nan())
        .count();
    let manifest = format!(
        r#"{{
  "bundle_schema": 1,
  "note": "Written by crates/axon-features/tests/feature_parity.rs; not market data.",
  "spec_ref": "{spec_ref}",
  "library_version": {library_version},
  "source": {{
    "description": "{rows} synthetic m1 bars generated in tests/feature_parity.rs",
    "instrument": "NONE",
    "interval": "1m",
    "venue": "none — this corpus never touched a venue"
  }},
  "spec": {{"file": "spec.json", "sha256": "{spec_hash}", "bytes": {spec_bytes_len}}},
  "inputs": {{"file": "inputs.f64", "sha256": "{inputs_hash}", "bytes": {inputs_bytes_len},
              "rows": {rows}, "cols": {input_cols}, "names": {input_names},
              "nan_cells": {input_nans}}},
  "features": {{"file": "features.f64", "sha256": "{features_hash}",
                "bytes": {features_bytes_len}, "rows": {rows}, "cols": {feature_cols},
                "names": {feature_names}, "nan_cells": {feature_nans},
                "finite_rows": {finite_rows}}},
  "criterion": {{"kind": "bit_exact"}},
  "libm_columns": {libm_columns},
  "producer": {{"rust": "tests/feature_parity.rs", "numpy": "none"}}
}}
"#,
        spec_ref = spec.reference(),
        library_version = spec.library_version(),
        spec_hash = content_hash(spec_bytes),
        spec_bytes_len = spec_bytes.len(),
        inputs_hash = content_hash(&inputs_bytes),
        inputs_bytes_len = inputs_bytes.len(),
        rows = rows,
        input_cols = names.len(),
        input_names = json_list(&names),
        input_nans = input_nans,
        features_hash = content_hash(&features_bytes),
        features_bytes_len = features_bytes.len(),
        feature_cols = features.cols(),
        feature_names = json_list(features.columns()),
        feature_nans = features.nan_cells(),
        finite_rows = features.finite_rows(),
        libm_columns = json_list(
            LIBM_COLUMNS
                .iter()
                .filter(|c| features.columns().iter().any(|name| name == *c))
        ),
    );
    fs::write(dir.join("manifest.json"), manifest).unwrap();
    dir
}

/// A healthy bundle over `BAR_M1_V1`, with `edits` applied to the **reference** matrix.
///
/// Perturbing the reference rather than the candidate is the shape that matters: the
/// candidate is whatever this build computes, and the question the gate asks is whether
/// this build reproduces a frozen answer. An edit here is a stand-in for Python having
/// computed something else.
fn bar_bundle(scratch: &str, edits: &[(usize, usize, f64)]) -> FeatureBundle {
    let spec = bar_spec();
    let inputs = bar_inputs(ROWS);
    let matrix = spec.compute(&inputs).expect("BAR_M1_V1 computes");
    let reference = with_cells(&matrix, edits);
    let dir = write_bundle(scratch, BAR_M1_JSON, &inputs, &reference);
    FeatureBundle::open(&dir).unwrap_or_else(|e| panic!("{scratch}: {e}"))
}

/// `matrix` with individual cells replaced, asserting each replacement changed the bits.
///
/// The assertion is the whole point of routing edits through here. Last session a test
/// loosened a manifest with `.replace("1e-05", "0.01")`, the pattern stopped matching,
/// the "corrupted" bundle was a byte-identical copy, and the assertion was asking its
/// question of a bundle nobody had loosened. A perturbation that does not perturb reads
/// as protection while testing nothing — and here it would be worse than nothing, since
/// the gate is *supposed* to pass on an unperturbed matrix.
fn with_cells(matrix: &FeatureMatrix, edits: &[(usize, usize, f64)]) -> FeatureMatrix {
    let mut data = matrix.as_slice().to_vec();
    for (row, col, value) in edits {
        let at = row * matrix.cols() + col;
        assert_ne!(
            data[at].to_bits(),
            value.to_bits(),
            "the perturbation of row {row} column {col} changed nothing"
        );
        data[at] = *value;
    }
    FeatureMatrix::new(matrix.columns().to_vec(), data).expect("a matrix of the same shape")
}

/// The next representable double away from zero — one ULP, the smallest disagreement
/// two implementations of the same arithmetic can have.
fn one_ulp_away(value: f64) -> f64 {
    assert!(value.is_finite(), "a ULP step off {value} is not a number");
    f64::from_bits(value.to_bits() + 1)
}

/// Edit a file in a bundle, failing loudly if the pattern matched nothing.
///
/// See [`with_cells`] for the failure this shape exists to prevent; it is the same one,
/// and it happened to a manifest rather than to a matrix.
fn rewrite(path: &Path, from: &str, to: &str) {
    let text = fs::read_to_string(path).expect("the file to corrupt");
    let patched = text.replace(from, to);
    assert_ne!(
        patched,
        text,
        "the corruption pattern {from:?} matched nothing in {}; the bundle was left \
         byte-identical and the assertion would be asking its question of a healthy fixture",
        path.display()
    );
    fs::write(path, patched).expect("write the corrupted file");
}

/// A healthy bundle directory, with one file edited afterwards. Returns the path rather
/// than an opened bundle, because most callers are asserting that opening it fails.
fn broken_bundle(scratch: &str, file: &str, from: &str, to: &str) -> PathBuf {
    let spec = bar_spec();
    let inputs = bar_inputs(ROWS);
    let matrix = spec.compute(&inputs).expect("BAR_M1_V1 computes");
    let dir = write_bundle(scratch, BAR_M1_JSON, &inputs, &matrix);
    rewrite(&dir.join(file), from, to);
    dir
}

fn column(matrix: &FeatureMatrix, name: &str) -> usize {
    matrix.column_index(name).expect("a column of BAR_M1_V1")
}

// ── the committed bundles: the cross-language claim ───────────────────────────

/// Every committed bundle, with the shape it must still have.
///
/// A literal list rather than a directory glob, so a bundle that vanishes is a failing
/// test rather than a silently smaller gate; `every_committed_bundle_is_in_the_gate`
/// closes the other direction. The shapes are asserted for the same reason
/// `cells_compared` is: how much was compared is the number a gate loses quietly, and a
/// regeneration that halved a corpus would otherwise look exactly like one that did not.
/// Regenerating these bundles is a deliberate, reviewable event, and updating these
/// numbers is part of it.
const COMMITTED: [(&str, usize, usize); 5] = [
    ("all_transforms", 900, 18),
    ("bar_m1_btc", 900, 6),
    ("bar_m1_eth", 900, 6),
    ("bar_m1_testnet_live", 58, 6),
    // `PERP_CORE_V1` over a **recorded** order book and tape — market-data slices from a
    // live read-only Hyperliquid testnet session, through the Rust core and publisher
    // onto the `MdSlice` ring. `all_transforms` covers the same microstructure
    // transforms over columns *derived* from bars and says so in its manifest; this is
    // the one where the book is the venue's.
    ("perp_core_live", 675, 9),
];

fn bundles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bundles")
}

fn committed(name: &str) -> FeatureBundle {
    FeatureBundle::open(bundles_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A writable copy of a committed bundle, for the tests that have to break one.
///
/// Under `CARGO_TARGET_TMPDIR` so a failed run leaves the evidence in `target/` rather
/// than in the source tree, where the next `cargo test` would gate it — and so that no
/// test in this file can edit a frozen fixture, which is the one thing that would turn
/// the cross-language claim back into a Rust-to-Rust one.
fn corrupted_copy(source: &str, scratch: &str, edit: impl FnOnce(&Path)) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(scratch);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    for entry in fs::read_dir(bundles_dir().join(source)).expect("bundle dir") {
        let entry = entry.expect("dir entry");
        fs::copy(entry.path(), dir.join(entry.file_name())).expect("copy");
    }
    edit(&dir);
    dir
}

/// Whether this build runs against the libm the committed matrices were taken against.
///
/// The bundles are NumPy's answers on glibc. ADR-0035 measured `log` agreeing to 0 ULP
/// over a 200 000-sample sweep *there*, and states plainly that this does not make the
/// gate portable: `log` is not in IEEE-754's correctly-rounded list, so another libm may
/// round the last bit differently with nothing wrong on either side.
const REFERENCE_LIBM: bool = cfg!(target_env = "gnu");

/// Distance in ULPs between two finite doubles, via the usual monotonic re-keying of the
/// bit pattern. Non-finite or opposite-infinity inputs come back saturated so a caller
/// bounding this can never read them as "close".
fn ulps_between(a: f64, b: f64) -> u128 {
    if a == b {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() {
        return u128::MAX;
    }
    fn ordered(x: f64) -> i128 {
        let bits = x.to_bits();
        if bits >> 63 == 1 {
            -((bits & !(1u64 << 63)) as i128)
        } else {
            bits as i128
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

#[test]
fn rust_reproduces_pythons_matrix_for_every_cell_of_every_committed_bundle() {
    // The claim ADR-0021 said a model bundle could never make. These matrices are
    // NumPy's, computed by `axon.features` over bytes read back off disk, and nothing in
    // this crate could have produced them: no test here writes them, and `cargo test`
    // runs on a machine with no Python at all. Every cell, to the bit.
    let mut cells = 0;
    for (name, rows, cols) in COMMITTED {
        let bundle = committed(name);
        assert_eq!(bundle.rows(), rows, "{name} changed shape");
        assert_eq!(bundle.cols(), cols, "{name} changed shape");
        // Not the manifest's opinion: a feature bundle has one criterion and no arm to
        // fall back to, so this is what `open` refused to let it be anything else.
        assert_eq!(bundle.criterion(), Criterion::BitExact, "{name}");

        let report = bundle.check().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(report.cells_compared(), rows * cols, "{name}");
        // A window that filled on one side and not the other is a structural defect, not
        // arithmetic, so it is refused on every platform.
        assert_eq!(report.nan_disagreements(), 0, "{}", report.summary());

        if REFERENCE_LIBM {
            assert_eq!(report.bit_mismatches(), 0, "{}", report.summary());
            assert_eq!(report.worst_column(), None, "{}", report.summary());
            // Zero, not "inside a tolerance" — there is no tolerance here to be inside of.
            assert_eq!(
                report.max_abs_diff().to_bits(),
                0.0f64.to_bits(),
                "{}",
                report.summary()
            );
            report.assert_passed();
        } else {
            // ADR-0035 §"`libm_columns` is a signpost, not a tolerance": these matrices are
            // NumPy's, taken against glibc's `log`, which is not in IEEE-754's
            // correctly-rounded list — so bit equality is a property of *that* libm pair and
            // the ADR says so outright ("it does not make the gate portable"). Claiming it
            // here would be claiming something never measured. What is still owed off the
            // reference libm is the ADR's own first question — "did it redden *only* on
            // those columns?" — so that is what is asserted, and it is not a tolerance: a
            // divergence anywhere else, or of more than a last bit, still fails.
            for d in report.divergences() {
                assert!(
                    report.libm_columns().iter().any(|c| c == &d.column),
                    "{name}: column {:?} diverged and does not reach libm — this is not the \
                     documented libm difference. {}",
                    d.column,
                    report.summary()
                );
                let ulps = ulps_between(d.reference, d.candidate);
                assert!(
                    ulps <= 1,
                    "{name}: column {:?} diverged by {ulps} ULP, not the last bit a differing \
                     libm explains. {}",
                    d.column,
                    report.summary()
                );
            }
        }
        cells += report.cells_compared();
    }
    // The denominator of the whole claim, asserted rather than described. A corpus that
    // silently emptied would satisfy every per-bundle assertion above by vacuity.
    assert_eq!(cells, 33_423);
}

#[test]
fn every_committed_bundle_is_in_the_gate() {
    // The other direction from `COMMITTED` being a literal: a bundle added to the tree
    // and forgotten here would sit in git looking like coverage while gating nothing.
    let mut found: Vec<String> = fs::read_dir(bundles_dir())
        .expect("bundles dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("manifest.json").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    let listed: Vec<String> = COMMITTED.iter().map(|(n, _, _)| n.to_string()).collect();
    assert_eq!(found, listed, "tests/bundles/ and COMMITTED disagree");
}

#[test]
fn the_committed_corpora_exercise_both_halves_of_the_nan_rule_on_real_bars() {
    // Both halves of ADR-0016 §4 have to be *reached* by the corpus, or the rule is
    // asserted only by the unit tests that were written to reach it. Warmup supplies the
    // first half everywhere; the second is the interesting one, and it comes from real
    // market data: a minute that traded at a single price has `high == low`, so `clv` is
    // NaN by the rule in `functions.rs` — 5 such bars in the ETH corpus and 6 in the 58
    // live testnet bars, which is the ~10% rate that file documents. Those are NaN cells
    // *past* the warmup block, where NumPy said NaN and this build said NaN, and the
    // gate matched them rather than counting them.
    for (name, _, _) in COMMITTED {
        let bundle = committed(name);
        assert!(
            bundle.reference().nan_cells() > 0,
            "{name} has no NaN cells, so the both-NaN half of the rule is untested by it"
        );
        assert!(
            bundle.reference().finite_rows() > 0,
            "{name} never finishes warming up"
        );
    }

    let btc = committed("bar_m1_btc");
    assert_eq!(btc.reference().nan_cells(), 45);
    assert_eq!(btc.reference().finite_rows(), 880);

    // A NaN `clv` past the derived warmup of 21 bars: a bar that never moved.
    let late_clv_nans = |bundle: &FeatureBundle| -> usize {
        let matrix = bundle.reference();
        let col = column(matrix, "clv");
        let warmup = bundle
            .spec()
            .max_lookback()
            .unwrap()
            .expect("a bounded spec");
        (warmup - 1..matrix.rows())
            .filter(|row| matrix.get(*row, col).is_nan())
            .count()
    };
    let eth = committed("bar_m1_eth");
    assert_eq!(eth.reference().nan_cells(), 50);
    assert_eq!(eth.reference().finite_rows(), 875);
    assert_eq!(late_clv_nans(&eth), 5);

    let live = committed("bar_m1_testnet_live");
    assert_eq!(live.reference().nan_cells(), 51);
    assert_eq!(live.reference().finite_rows(), 33);
    assert_eq!(late_clv_nans(&live), 5);
}

#[test]
fn the_coverage_bundle_gates_every_registered_transform_and_not_the_subset_a_spec_uses() {
    // `bar_m1` uses six of the seventeen transforms, so a gate over shipped specs alone
    // would leave eleven of them unmeasured against Python — and the two that are hardest
    // to reproduce (`ema`, and every rolling reduction that goes through NumPy's pairwise
    // summation) are among them. This asserts the coverage bundle is what its own
    // description claims, in Rust, rather than trusting the generator's comment.
    let bundle = committed("all_transforms");
    let mut used: Vec<&str> = bundle
        .spec()
        .features()
        .iter()
        .map(|definition| definition.feature())
        .collect();
    used.sort_unstable();
    used.dedup();
    assert_eq!(
        used,
        registered_features(),
        "the coverage bundle does not exercise every registered transform"
    );
    // Unbounded, because it contains an EMA — the shape `FeatureStream` refuses. The
    // batch path is also the research path, so the gate covers it anyway.
    assert_eq!(bundle.spec().max_lookback().unwrap(), None);
    bundle.check().expect("the gate runs").assert_passed();
}

#[test]
fn one_ulp_of_drift_in_pythons_own_matrix_reddens_the_gate() {
    // The most important test in the file. Everything above shows the gate agreeing;
    // this shows it disagreeing, on the smallest disagreement float64 can carry, against
    // a matrix NumPy actually wrote. One ULP is what an incremental rolling mean, a
    // `E[x²] − E[x]²` variance or a naive summation order each cost — `numeric.rs`
    // measures that last one at up to eight ULP over a 128-sample window, so this is the
    // conservative end of what this gate exists to catch.
    //
    // It is also the argument for not hashing, made concrete: the manifest's recorded
    // SHA-256 no longer describes this file, and the gate catches the change anyway —
    // naming the row and the column, which a hash cannot do.
    let (row, cols) = (500usize, 6usize);
    let col = column(committed("bar_m1_btc").reference(), "z_20");
    let mut moved = (0.0, 0.0);
    let dir = corrupted_copy("bar_m1_btc", "committed-one-ulp", |dir| {
        moved = nudge_feature_cell(dir, row, col, cols);
    });
    let (before, after) = moved;

    let bundle = FeatureBundle::open(&dir).expect("a bundle that is still well-formed");
    assert_eq!(
        bundle.reference().get(row, col).to_bits(),
        after.to_bits(),
        "the perturbation did not survive the round trip"
    );
    let report = bundle.check().expect("the gate runs");

    assert!(!report.passed(), "{}", report.summary());
    assert_eq!(report.cells_compared(), 900 * 6, "{}", report.summary());
    assert_eq!(report.bit_mismatches(), 1, "{}", report.summary());
    assert_eq!(report.nan_disagreements(), 0, "{}", report.summary());
    assert_eq!(report.worst_column(), Some("z_20"));
    // The measured delta, as the exact number rather than a bound: one ULP at this
    // z-score is 2.220446049250313e-16, and "less than 1e-15" would stay true if the
    // perturbation grew a thousandfold.
    assert_eq!(
        report.max_abs_diff().to_bits(),
        (after - before).abs().to_bits(),
        "{}",
        report.summary()
    );

    let divergence = report.divergences().first().expect("a divergent cell");
    assert_eq!(divergence.row, row);
    assert_eq!(divergence.column, "z_20");
    assert!(!divergence.nan_disagreement);

    let summary = report.summary();
    assert!(summary.contains("FAIL"), "{summary}");
    assert!(summary.contains("row 500 column \"z_20\""), "{summary}");
    // The bit patterns have to be in the message: at one ULP the two decimals print
    // identically, and a failure showing the same number twice reads as a lie.
    assert!(
        summary.contains(&format!("{:#018x}", before.to_bits()))
            && summary.contains(&format!("{:#018x}", after.to_bits())),
        "{summary}"
    );
    // `z_20` reads `close` directly and never reaches `log`, so the report must say that
    // a libm difference cannot explain this rather than leaving the question open.
    assert!(summary.contains("do not pass through log"), "{summary}");
}

/// Move one cell of a bundle's `features.f64` by one ULP, in place, and return the two
/// values. Asserts the file's bytes actually changed.
fn nudge_feature_cell(dir: &Path, row: usize, col: usize, cols: usize) -> (f64, f64) {
    let path = dir.join("features.f64");
    let original = fs::read(&path).expect("features.f64");
    let at = (row * cols + col) * 8;
    let mut bytes = original.clone();
    let before = f64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    let after = one_ulp_away(before);
    bytes[at..at + 8].copy_from_slice(&after.to_le_bytes());
    assert_ne!(
        bytes, original,
        "the perturbation of row {row} column {col} changed no bytes"
    );
    fs::write(&path, bytes).expect("write the perturbed matrix");
    (before, after)
}

// ── the self-written bundles: how the gate is shown to fail ───────────────────

#[test]
fn a_bundle_written_from_this_build_reproduces_its_own_matrix_cell_for_cell() {
    // The baseline every failing test below is measured against, and the proof that the
    // harness writes a bundle the reader accepts — without it, a refusal below could be
    // the reader objecting to the *fixture* while the test read it as the refusal it was
    // looking for. The counts are asserted as numbers rather than as "greater than
    // zero", because a corpus that quietly shrank would still satisfy a bound.
    let bundle = bar_bundle("bar-healthy", &[]);
    assert_eq!(bundle.spec_ref(), "bar_m1/v1#c503688de24e863f");
    assert_eq!(bundle.rows(), ROWS);
    assert_eq!(bundle.cols(), COLS);
    assert_eq!(bundle.input_names(), ["close", "high", "low"]);
    assert_eq!(bundle.criterion(), Criterion::BitExact);
    assert_eq!(bundle.libm_columns(), LIBM_COLUMNS);
    assert_eq!(bundle.reference().nan_cells(), NAN_CELLS);
    assert_eq!(bundle.reference().finite_rows(), FINITE_ROWS);

    let report = bundle.check().expect("the gate runs");
    // The denominator, asserted rather than printed. "PASS, 0 cells compared" is the
    // same invisible-denominator failure ADR-0030 spent an increment on one level up:
    // a gate whose corpus emptied is green in exactly the way a gate that ran is.
    assert_eq!(report.cells_compared(), ROWS * COLS);
    assert_eq!(report.rows(), ROWS);
    assert_eq!(report.bit_mismatches(), 0, "{}", report.summary());
    assert_eq!(report.nan_disagreements(), 0, "{}", report.summary());
    assert_eq!(report.worst_column(), None);
    // Zero, to the bit, over 1800 cells — not "within a tolerance", because there is no
    // tolerance here to be within.
    assert_eq!(report.max_abs_diff().to_bits(), 0.0f64.to_bits());
    report.assert_passed();
    assert!(report.summary().contains("PASS"), "{}", report.summary());
    assert!(
        report.summary().contains("cells=1800"),
        "{}",
        report.summary()
    );
}

#[test]
fn one_ulp_in_a_self_written_bundle_reddens_it_too_which_is_what_makes_the_harness_honest() {
    // `one_ulp_of_drift_in_pythons_own_matrix_reddens_the_gate` is the version of this
    // that carries the cross-language claim. This one is about the *harness*: every
    // refusal below is asserted against a bundle this file wrote, so if `with_cells` or
    // `write_bundle` silently dropped an edit, those tests would be asking their
    // questions of a healthy fixture and passing for the wrong reason. Here the edit is
    // one ULP and the verdict has to flip, which it cannot do unless the edit reached
    // the comparison.
    let healthy = bar_bundle("bar-ulp-reference", &[]);
    let (row, col) = (100, column(healthy.reference(), "z_20"));
    let before = healthy.reference().get(row, col);
    let after = one_ulp_away(before);

    let bundle = bar_bundle("bar-ulp", &[(row, col, after)]);
    let report = bundle.check().expect("the gate runs");

    assert!(!report.passed(), "{}", report.summary());
    assert_eq!(report.cells_compared(), ROWS * COLS);
    assert_eq!(report.bit_mismatches(), 1, "{}", report.summary());
    assert_eq!(report.nan_disagreements(), 0, "{}", report.summary());
    assert_eq!(report.worst_column(), Some("z_20"));
    // The measured delta, asserted as the exact number rather than as a bound. On a
    // z-score near 1 one ULP is 2.220446049250313e-16, and the bound "less than 1e-15"
    // would stay true if the perturbation grew a thousandfold.
    assert_eq!(
        report.max_abs_diff().to_bits(),
        (after - before).abs().to_bits()
    );

    let divergence = report.divergences().first().expect("a divergent cell");
    assert_eq!(divergence.row, row);
    assert_eq!(divergence.column, "z_20");
    assert!(!divergence.nan_disagreement);

    let summary = report.summary();
    assert!(summary.contains("FAIL"), "{summary}");
    assert!(summary.contains("row 100 column \"z_20\""), "{summary}");
    // The bit patterns have to be in the message: at one ULP the two decimals print
    // identically, and a failure showing the same number twice reads as a lie.
    assert!(
        summary.contains(&format!("{:#018x}", before.to_bits()))
            && summary.contains(&format!("{:#018x}", after.to_bits())),
        "{summary}"
    );
}

#[test]
fn a_feature_that_goes_nan_on_one_side_only_is_a_mismatch_and_is_counted_apart() {
    // The staleness bug `docs/03` names: a window that fills on one path and not on the
    // other. Under `np.allclose` defaults this is indistinguishable from a matching
    // pair, which is why neither language uses it. It is counted separately from a
    // numeric mismatch because the diagnosis differs — a window that did not fill is not
    // arithmetic that drifted.
    let bundle = bar_bundle("bar-nan-one-side", &[(150, 3, f64::NAN)]);
    let report = bundle.check().expect("the gate runs");

    assert!(!report.passed(), "{}", report.summary());
    assert_eq!(report.nan_disagreements(), 1, "{}", report.summary());
    assert_eq!(report.bit_mismatches(), 0, "{}", report.summary());
    assert_eq!(report.worst_column(), Some("vol_20"));
    // An unmeasurable difference is never summarized as a small one: the cell where one
    // side is NaN contributes NaN to the delta, and NaN never wins a `>` comparison, so
    // the reported maximum stays at the largest *measurable* difference — which here is
    // zero, because every other cell matches.
    assert_eq!(report.max_abs_diff().to_bits(), 0.0f64.to_bits());

    let divergence = report.divergences().first().expect("a divergent cell");
    assert!(divergence.nan_disagreement);
    assert!(divergence.abs_diff.is_nan());
    assert!(
        report.summary().contains("NAN ON ONE SIDE"),
        "{}",
        report.summary()
    );
}

#[test]
fn warmup_nan_meeting_warmup_nan_is_the_match_that_makes_the_gate_runnable_at_all() {
    // The other half of ADR-0016 §4's split, and the reason it is a split. Warmup is NaN
    // by construction on both sides and 45 of this corpus's 1800 cells are warmup, so a
    // comparison that failed on a NaN would fail every run — hardest on a bundle that is
    // perfectly correct. The pass in the baseline test is only meaningful because these
    // cells exist, so the count is asserted here rather than assumed.
    let bundle = bar_bundle("bar-warmup", &[]);
    let reference = bundle.reference();
    assert_eq!(reference.nan_cells(), NAN_CELLS);
    assert_eq!(reference.rows() - reference.finite_rows(), 20);
    // Row 19 is still warming up (`vol_20` needs 21 bars) and row 20 is the first usable
    // one. Both are compared; neither is skipped.
    assert!(reference.get(19, 3).is_nan());
    assert!(reference.get(20, 3).is_finite());
    bundle.check().expect("the gate runs").assert_passed();
}

#[test]
fn every_nan_this_crate_emits_is_the_canonical_pattern_the_python_writer_accepts() {
    // A writer invariant this side has to keep meeting. `axon.parity.feature_bundle`
    // refuses a matrix carrying any NaN but `0x7ff8000000000000`, because its own reader
    // compares NaN cells as bits; on x86 `0.0 / 0.0` yields `0xfff8000000000000` — the
    // same value with the sign bit set. Every NaN in this crate comes from `or_nan` or
    // `fill_nan` masking a guarded cell, never from an invalid operation, and this is
    // what asserts a future transform cannot quietly start dividing first and masking
    // afterwards.
    assert_eq!(f64::NAN.to_bits(), 0x7ff8_0000_0000_0000);
    let matrix = bar_spec()
        .compute(&bar_inputs(ROWS))
        .expect("BAR_M1_V1 computes");
    let mut seen = 0;
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            let value = matrix.get(row, col);
            if value.is_nan() {
                seen += 1;
                assert_eq!(
                    value.to_bits(),
                    0x7ff8_0000_0000_0000,
                    "row {row} column {:?} carries a NaN the writer would refuse",
                    matrix.columns()[col]
                );
            }
        }
    }
    assert_eq!(seen, NAN_CELLS);
}

#[test]
fn a_nan_payload_difference_is_not_a_divergence_because_neither_language_promises_one() {
    // The deliberate asymmetry with the Python reader, and the reason it is safe. Python
    // compares NaN cells as raw bits and keeps that honest by refusing a non-canonical
    // NaN at write time; this side is blind to the payload instead. Blindness is the
    // safer half: a Rust transform that divided first and masked afterwards would be
    // arithmetically perfect and would redden a payload comparison on every guarded
    // cell, which is a defect in the gate rather than in the runtime.
    let x86_nan = f64::from_bits(0xfff8_0000_0000_0000);
    assert!(x86_nan.is_nan());
    assert_ne!(x86_nan.to_bits(), f64::NAN.to_bits());

    // A warmup cell, recorded with the other sign bit. Every count the manifest carries
    // is unchanged — it is still a NaN — so the bundle is well-formed and the only
    // question left is what the comparison makes of it.
    let bundle = bar_bundle("bar-nan-payload", &[(0, 0, x86_nan)]);
    assert_eq!(bundle.reference().nan_cells(), NAN_CELLS);
    assert_eq!(
        bundle.reference().get(0, 0).to_bits(),
        0xfff8_0000_0000_0000
    );
    let report = bundle.check().expect("the gate runs");
    report.assert_passed();
    assert_eq!(report.bit_mismatches(), 0, "{}", report.summary());
}

#[test]
fn the_worst_column_is_the_one_with_the_most_divergent_cells_not_the_largest_delta() {
    // ADR-0016 §4's ordering, and it is not a preference. A unit error or an off-by-one
    // window breaks *one column on every row*; a genuine arithmetic difference is a
    // handful of cells that are slightly worse. Ranking by magnitude would name the
    // second while the first was destroying the matrix — which is the diagnosis that
    // costs the afternoon.
    let spec = bar_spec();
    let inputs = bar_inputs(ROWS);
    let matrix = spec.compute(&inputs).expect("BAR_M1_V1 computes");
    let clv = column(&matrix, "clv");
    let range = column(&matrix, "range_bps");

    let mut edits: Vec<(usize, usize, f64)> = (0..ROWS)
        .map(|row| (row, clv, one_ulp_away(matrix.get(row, clv))))
        .collect();
    // One cell, wrong by a factor of a hundred — the largest delta in the matrix by
    // fourteen orders of magnitude, and the *less* urgent finding.
    edits.push((7, range, matrix.get(7, range) * 100.0));

    let bundle = bar_bundle("bar-worst-column", &edits);
    let report = bundle.check().expect("the gate runs");
    assert_eq!(report.bit_mismatches(), ROWS + 1, "{}", report.summary());
    assert_eq!(report.worst_column(), Some("clv"));
    // The big delta is still reported — it is the diagnosis, not the ranking.
    assert!(report.max_abs_diff() > 1.0, "{}", report.summary());
    let summary = report.summary();
    assert!(summary.contains("worst column \"clv\""), "{summary}");
    assert!(summary.contains("300 of 300 cells diverge"), "{summary}");
    // The divergence list is capped and the counts are not, so the verdict never depends
    // on how many rows were kept for printing.
    assert_eq!(report.divergences().len(), 20);
    assert!(summary.contains("showing the first 20"), "{summary}");
}

#[test]
fn the_libm_signpost_is_reported_in_both_directions_and_excuses_nothing() {
    // `libm_columns` names the columns whose value passes through `log`, the one
    // operation in this library IEEE-754 does not require to be correctly rounded. It is
    // a signpost: the first question when this gate reddens on a platform nobody has
    // measured is whether it reddened *only* there. Both answers are printed, and the
    // negative one is the more useful — it closes off the explanation that would
    // otherwise absorb the afternoon.
    let healthy = bar_bundle("bar-libm-reference", &[]);
    let log_col = column(healthy.reference(), "ret_1");
    let plain_col = column(healthy.reference(), "clv");

    let on_log = bar_bundle(
        "bar-libm-log",
        &[(
            50,
            log_col,
            one_ulp_away(healthy.reference().get(50, log_col)),
        )],
    );
    let summary = on_log.check().expect("the gate runs").summary();
    assert!(
        summary.contains("every divergent column passes through log"),
        "{summary}"
    );
    // Named, and not forgiven: the report is still a failure.
    assert!(!on_log.check().unwrap().passed(), "{summary}");

    let off_log = bar_bundle(
        "bar-libm-plain",
        &[(
            50,
            plain_col,
            one_ulp_away(healthy.reference().get(50, plain_col)),
        )],
    );
    let summary = off_log.check().expect("the gate runs").summary();
    assert!(
        summary.contains("do not pass through log"),
        "the failure must say that libm cannot explain this: {summary}"
    );
}

#[test]
fn a_candidate_of_another_shape_is_refused_rather_than_compared_pairwise() {
    // A short candidate means the two runs did not see the same events, and zipping
    // them would compare the rows that happen to line up and call it parity. Permuted
    // columns are worse: same width, every cell comparable, every value attributed to
    // the wrong feature.
    let bundle = bar_bundle("bar-shape", &[]);
    let reference = bundle.reference();

    let short = FeatureMatrix::new(
        reference.columns().to_vec(),
        reference.as_slice()[..(ROWS - 1) * COLS].to_vec(),
    )
    .unwrap();
    assert!(matches!(
        bundle.compare(&short),
        Err(BundleError::Mismatch(_))
    ));

    let mut renamed = reference.columns().to_vec();
    renamed.swap(0, 1);
    let permuted = FeatureMatrix::new(renamed, reference.as_slice().to_vec()).unwrap();
    assert!(matches!(
        bundle.compare(&permuted),
        Err(BundleError::Mismatch(_))
    ));
}

// ── refusals: a broken fixture must not read as a parity failure ──────────────

#[test]
fn a_bundle_cannot_buy_itself_a_looser_criterion() {
    // The failure is procedural rather than numerical: a bundle regenerated after a red
    // gate, with a tolerance added until it passed, is otherwise indistinguishable in
    // the tree from one that never failed. Even 1e-12 is refused — there is no tolerance
    // arm to fall back to, because every transform here is built from operations
    // IEEE-754 requires to be correctly rounded, plus one whose agreement is measured
    // and named in `libm_columns` rather than paid for with slack.
    let dir = broken_bundle(
        "bar-loosened",
        "manifest.json",
        r#""criterion": {"kind": "bit_exact"}"#,
        r#""criterion": {"kind": "max_abs_diff", "eps": 1e-12}"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(err, BundleError::Weakened(_)),
        "expected a Weakened refusal, got {err}"
    );
}

#[test]
fn a_spec_edited_after_its_fingerprint_was_taken_is_refused_at_the_bundle() {
    // Somebody widening a window in a committed `spec.json` to make a gate pass. The
    // edit keeps the file the same length, so the manifest's `bytes` still agrees and
    // the *recomputed* fingerprint is the only thing that can catch it — which is why
    // `FeatureSpec::from_json` recomputes rather than reads.
    let dir = broken_bundle(
        "bar-edited-spec",
        "spec.json",
        r#""params":{"window":20}}"#,
        r#""params":{"window":21}}"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("fingerprint")),
        "expected a fingerprint refusal, got {err}"
    );
}

#[test]
fn a_spec_serialized_in_another_key_order_is_refused_even_though_it_parses_the_same() {
    // `spec_ref` names a hash of *these bytes*. A file that parses to the same spec and
    // serializes differently is the one failure only a re-serialization can see: the two
    // languages disagreeing about how the recipe is written, which produces a
    // fingerprint that reproduces nowhere else.
    //
    // The two top-level keys are **swapped rather than reformatted**, and the length is
    // therefore unchanged, because that is the only edit this check is the sole owner of.
    // Written first as `{"features":[` -> `{ "features": [`, it grew the file by two
    // bytes and the manifest's `spec.bytes` refused it first — the test passed, the
    // canonical comparison was never reached, and deleting that comparison left the test
    // green. A corruption that is caught by the wrong check is a test of the wrong check.
    let dir = broken_bundle(
        "bar-reordered-spec",
        "spec.json",
        r#""spec":"bar_m1","version":1}"#,
        r#""version":1,"spec":"bar_m1"}"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(_)),
        "expected a canonical-form refusal, got {err}"
    );
}

#[test]
fn a_manifest_naming_a_spec_the_directory_does_not_hold_is_refused() {
    // The manifest and the spec beside it are two halves of one claim. If `spec_ref`
    // were believed rather than checked against the spec on disk, a bundle could carry a
    // recipe nobody trained on under an id somebody did.
    let dir = broken_bundle(
        "bar-wrong-ref",
        "manifest.json",
        "bar_m1/v1#c503688de24e863f",
        "bar_m1/v1#0000000000000000",
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("re-identifies")),
        "expected a spec_ref refusal, got {err}"
    );
}

#[test]
fn inputs_named_something_the_spec_does_not_read_are_refused_before_a_column_is_computed() {
    // The reader pairs arrays with names positionally from `inputs.names`, so a mismatch
    // here mislabels every column at once — and the result is not obviously wrong. A
    // swapped `high` and `low` produces a plausible number in every cell.
    let dir = broken_bundle(
        "bar-wrong-inputs",
        "manifest.json",
        r#""names": ["close","high","low"]"#,
        r#""names": ["close","hi","low"]"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("mislabels")),
        "expected an input-name refusal, got {err}"
    );
}

#[test]
fn a_manifest_and_a_spec_that_disagree_about_the_library_build_are_refused() {
    // Two halves of one claim. `FeatureSpec::from_json` already refuses a spec written
    // against another build of `axon.features`; this is the manifest saying something
    // different from the spec beside it, which means one of the two was edited.
    let dir = broken_bundle(
        "bar-library-version",
        "manifest.json",
        r#""library_version": 1"#,
        r#""library_version": 2"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("which build of axon.features")),
        "expected a library-version refusal, got {err}"
    );
}

#[test]
fn a_libm_signpost_that_disagrees_with_rusts_own_derivation_is_refused() {
    // The signpost is the first question asked when this gate reddens on a platform
    // nobody has measured, and a wrong one answers that question with silence: every
    // divergent column would look like it was not a log column, which is the answer
    // that closes off the investigation.
    //
    // This used to check only that the names were *real columns*, which let a
    // manifest name a perfectly real column that does not touch `log` and be
    // believed. It is now re-derived on both sides — `LIBM_FEATURES` in Python,
    // `FeatureInfo::reaches_libm` here, each walked through the spec's own binding
    // graph — so the field is a checked cross-language property rather than one Rust
    // reads and trusts. Both failures below are refused; only the first was before.
    for (label, from, to) in [
        // A real column that is not a log column. The old check waved this through.
        (
            "bar-libm-plausible",
            r#""libm_columns": ["ret_1","mom_5","vol_20"]"#,
            r#""libm_columns": ["clv","mom_5","vol_20"]"#,
        ),
        // A name that is not a column at all.
        (
            "bar-libm-nonexistent",
            r#""libm_columns": ["ret_1""#,
            r#""libm_columns": ["ret_9""#,
        ),
        // Dropping one entirely — the quietest of the three, and the one that would
        // make a genuine libm failure read as "not the log columns, so not libm".
        (
            "bar-libm-truncated",
            r#""libm_columns": ["ret_1","mom_5","vol_20"]"#,
            r#""libm_columns": ["ret_1","mom_5"]"#,
        ),
    ] {
        let dir = broken_bundle(label, "manifest.json", from, to);
        let err = FeatureBundle::open(&dir).unwrap_err();
        assert!(
            matches!(&err, BundleError::Malformed(m) if m.contains("libm signpost")),
            "{label}: expected a libm-signpost refusal, got {err}"
        );
    }
}

#[test]
fn the_two_languages_derive_the_same_libm_columns_for_every_committed_bundle() {
    // The positive half, and the one that makes the refusal above mean something: the
    // committed manifests were written from Python's table and this build re-derives
    // them from its own. A test that only ever saw corrupted signposts would pass
    // just as happily if the two derivations had never agreed at all.
    let mut checked = 0;
    for (name, ..) in COMMITTED {
        let bundle = committed(name);
        let derived = bundle.spec().libm_columns().expect("the spec derives");
        assert_eq!(
            derived,
            bundle.libm_columns(),
            "{name}: Rust derives a different log signpost from the one Python wrote"
        );
        checked += derived.len();
    }
    assert!(
        checked > 0,
        "no committed bundle has a log column, so this comparison proved nothing"
    );
}

#[test]
fn feature_columns_named_something_the_spec_does_not_produce_are_refused() {
    // The counterpart of the input-name check. Column order and naming are inside the
    // spec fingerprint precisely because permuting two leaves every name correct and
    // every prediction wrong; a manifest that renames one is the same failure arriving
    // from the other side.
    let dir = broken_bundle(
        "bar-wrong-columns",
        "manifest.json",
        r#""ret_1""#,
        r#""ret_9""#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("matrix columns are named")),
        "expected a column-name refusal, got {err}"
    );
}

#[test]
fn a_manifest_that_contradicts_itself_is_refused_before_a_cell_is_compared() {
    // Every one of these describes a bundle that would otherwise be *read*, with the
    // disagreement showing up later as a divergence — which is the one diagnosis that
    // sends somebody looking at `numeric.rs` for a defect that is in a manifest.
    for (scratch, from, to, needle) in [
        // A zero-row corpus: the two languages agree about the empty matrix and disagree
        // about everything else.
        (
            "bar-zero-rows",
            r#""rows": 300, "cols": 3"#,
            r#""rows": 0, "cols": 3"#,
            "zero rows proves nothing",
        ),
        // Input arrays and feature rows describing different events.
        (
            "bar-row-disagreement",
            r#""rows": 300, "cols": 6"#,
            r#""rows": 299, "cols": 6"#,
            "not describing the same events",
        ),
        (
            "bar-input-width",
            r#""cols": 3"#,
            r#""cols": 4"#,
            "input columns for",
        ),
        (
            "bar-feature-width",
            r#""cols": 6"#,
            r#""cols": 7"#,
            "feature columns for",
        ),
        // A declared byte count that does not follow from the declared shape. The file is
        // untouched, so only the manifest's own arithmetic can catch this.
        (
            "bar-byte-count",
            r#""bytes": 14400"#,
            r#""bytes": 14408"#,
            "f64 =",
        ),
        (
            "bar-input-nan-count",
            r#""nan_cells": 0"#,
            r#""nan_cells": 1"#,
            "NaN input cells",
        ),
    ] {
        let dir = broken_bundle(scratch, "manifest.json", from, to);
        let err = FeatureBundle::open(&dir).unwrap_err();
        assert!(
            matches!(&err, BundleError::Malformed(m) if m.contains(needle)),
            "{scratch}: expected a refusal mentioning {needle:?}, got {err}"
        );
    }
}

#[test]
fn a_nan_count_that_does_not_match_the_matrix_is_a_broken_fixture_not_a_parity_failure() {
    // Warmup is NaN by construction and its extent is part of what a bundle asserts: a
    // matrix whose warmup silently became zeros has the same shape, a different meaning,
    // and nothing else in the format would notice. Refused as malformed rather than
    // surfacing later as a divergence, because "the fixture is wrong" and "the two
    // languages disagree" send a reader to different files.
    let dir = broken_bundle(
        "bar-wrong-nan-count",
        "manifest.json",
        r#""nan_cells": 45"#,
        r#""nan_cells": 44"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("NaN feature cells")),
        "expected a NaN-count refusal, got {err}"
    );
}

#[test]
fn a_finite_row_count_that_does_not_match_the_matrix_is_refused() {
    let dir = broken_bundle(
        "bar-wrong-finite-rows",
        "manifest.json",
        r#""finite_rows": 280"#,
        r#""finite_rows": 279"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("fully finite rows")),
        "expected a finite-row refusal, got {err}"
    );
}

#[test]
fn a_corpus_that_never_finishes_warming_up_is_refused_rather_than_passing_vacuously() {
    // An all-NaN reference matches an all-NaN candidate under the NaN rule, so this
    // bundle would go green for a runtime that computes nothing at all — the same
    // invisible denominator `cells_compared` guards, arriving by a different route. The
    // Python writer refuses to write one; this reader refuses to read one, because a
    // bundle can also be hand-made or truncated into that state.
    let spec = bar_spec();
    let inputs = bar_inputs(ROWS);
    let matrix = spec.compute(&inputs).expect("BAR_M1_V1 computes");
    let all_nan =
        FeatureMatrix::new(matrix.columns().to_vec(), vec![f64::NAN; ROWS * COLS]).unwrap();
    assert_eq!(all_nan.finite_rows(), 0);

    let dir = write_bundle("bar-all-nan", BAR_M1_JSON, &inputs, &all_nan);
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("never finishes warming up")),
        "expected an all-NaN refusal, got {err}"
    );
}

#[test]
fn a_truncated_matrix_is_refused_rather_than_read_as_a_shorter_corpus() {
    // Silently reading the rows that survived would leave a gate that passes on the half
    // of the corpus that made it to disk.
    let spec = bar_spec();
    let inputs = bar_inputs(ROWS);
    let matrix = spec.compute(&inputs).expect("BAR_M1_V1 computes");
    let dir = write_bundle("bar-truncated", BAR_M1_JSON, &inputs, &matrix);
    let path = dir.join("features.f64");
    let mut bytes = fs::read(&path).unwrap();
    let before = bytes.len();
    bytes.truncate(before - 8);
    assert_ne!(bytes.len(), before, "the truncation removed nothing");
    fs::write(&path, bytes).unwrap();

    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("bytes")),
        "expected a length refusal, got {err}"
    );
}

#[test]
fn a_bundle_from_a_newer_schema_is_refused_rather_than_read_with_the_unknown_half_absent() {
    let dir = broken_bundle(
        "bar-newer-schema",
        "manifest.json",
        r#""bundle_schema": 1"#,
        r#""bundle_schema": 2"#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("schema")),
        "expected a schema refusal, got {err}"
    );
}

#[test]
fn a_bundle_that_cannot_say_what_market_data_it_froze_is_refused() {
    // The first thing a red cross-language gate has to answer is what it ran on. A
    // bundle nobody can identify is a bundle nobody can regenerate, and the label in
    // every report is built from this field.
    let dir = broken_bundle(
        "bar-anonymous",
        "manifest.json",
        r#""description": "300 synthetic m1 bars generated in tests/feature_parity.rs""#,
        r#""description": "   ""#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("description")),
        "expected a provenance refusal, got {err}"
    );
}

#[test]
fn a_manifest_filename_that_escapes_the_bundle_directory_is_refused() {
    // A manifest is data, and a bundle received from elsewhere is somebody else's bytes.
    // Reading one must not become a file read of whoever wrote it choosing.
    let dir = broken_bundle(
        "bar-escaping-file",
        "manifest.json",
        r#""file": "inputs.f64""#,
        r#""file": "../inputs.f64""#,
    );
    let err = FeatureBundle::open(&dir).unwrap_err();
    assert!(
        matches!(&err, BundleError::Malformed(m) if m.contains("bare filename")),
        "expected a path refusal, got {err}"
    );
}

#[test]
fn the_reader_does_not_verify_the_hashes_the_manifest_records() {
    // A decision (ADR-0021 §7), pinned so it reads as one rather than as an omission: a
    // flipped bit in either matrix changes a value, so the gate itself catches it, and
    // the Python reader — which can afford to say *why* — is where the hashes are
    // checked. `sha2` is already in this crate for the spec fingerprint, so nothing but
    // the argument was stopping this reader from hashing too.
    //
    // The corrupted hash is not cosmetic: the same bundle is refused by
    // `axon.parity.feature_bundle.read_feature_bundle`, and that division of labour is
    // the thing being asserted.
    let dir = broken_bundle(
        "bar-bad-hash",
        "manifest.json",
        "\"sha256\": \"sha256:",
        "\"sha256\": \"sha256:0",
    );
    let bundle = FeatureBundle::open(&dir).expect("a wrong hash is not this reader's business");
    bundle.check().expect("the gate runs").assert_passed();
}

#[test]
fn a_matrix_whose_values_moved_is_caught_by_the_comparison_rather_than_by_a_hash() {
    // The other half of the argument above. Hashing would have caught this too, and so
    // does the gate — cell by cell, naming the row and the column, which is what a hash
    // cannot do.
    let healthy = bar_bundle("bar-flipped-bit-reference", &[]);
    let value = healthy.reference().get(42, 4);
    let bundle = bar_bundle("bar-flipped-bit", &[(42, 4, one_ulp_away(value))]);
    let report = bundle.check().expect("the gate runs");
    assert!(!report.passed(), "{}", report.summary());
    assert_eq!(report.bit_mismatches(), 1, "{}", report.summary());
    assert_eq!(report.worst_column(), Some("range_bps"));
}
