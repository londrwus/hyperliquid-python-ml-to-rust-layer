//! What the two languages have to agree on *before* a matrix comparison means
//! anything.
//!
//! `tests/feature_parity.rs` compares whole matrices, and that is the gate. This file
//! is the layer underneath it, and it exists because a matrix comparison can only
//! fail in one direction: it says the two sides disagree, never which of four
//! separate things went wrong. Each claim below can break on its own, and each one
//! makes the matrix comparison mean something different when it does.
//!
//! 1. **The registries describe the same seventeen transforms.** A drifted Rust
//!    table would bind `price` to one column while the transform read another — a
//!    silent feature swap, and both sides would still be computing *something* for
//!    every row.
//! 2. **The two languages agree on what a spec is**, byte for byte, and therefore on
//!    its fingerprint. Rust already recomputes the fingerprint on every load; what
//!    this adds is the bytes, so a disagreement points at the serialization rather
//!    than at the recipe.
//! 3. **NumPy's reduction order.** The claim the whole crate rests on, pinned against
//!    NumPy itself rather than against `numeric.rs`'s description of it. A comment
//!    claiming to match NumPy is a comment; this is the thing that reddens the day
//!    NumPy changes its unroll factor.
//! 4. **`log` agrees with `np.log`.** IEEE-754 requires `+ - * /` and `sqrt` to be
//!    correctly rounded and says nothing at all about `log`, so this one is a
//!    *measurement* on this platform's libm and not a guarantee. It is separated from
//!    the rest deliberately: if this file ever reddens on a different machine, this
//!    is the assertion to look at first.
//!
//! The fixture is written by `tests/fixtures/generate.py`. Regenerating it rewrites a
//! frozen reference and **the git diff is the review** — the same rule
//! `./run.sh parity-bundles` follows for the model bundles.

use std::collections::BTreeMap;
use std::path::PathBuf;

use axon_features::registry::{feature_info, registered_features};
use axon_features::{FeatureSpec, FEATURES_VERSION};
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cross_language.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nregenerate with: .venv/bin/python crates/axon-features/tests/fixtures/generate.py",
            path.display()
        )
    });
    serde_json::from_str(&raw).expect("the fixture is valid JSON")
}

/// A float64 back out of the fixture's hex bit pattern.
///
/// Bits rather than decimals throughout: a float written as a decimal and re-parsed
/// can land on the neighbouring value, and every test here would then be measuring
/// its own serialization instead of the thing it names.
fn from_bits(hex: &str) -> f64 {
    let digits = hex.trim().trim_start_matches("0x");
    f64::from_bits(u64::from_str_radix(digits, 16).expect("a 64-bit pattern"))
}

fn series(fixture: &Value) -> Vec<f64> {
    fixture["series"]
        .as_array()
        .expect("series")
        .iter()
        .map(|v| from_bits(v.as_str().unwrap()))
        .collect()
}

// ── 1. the registries ─────────────────────────────────────────────────────────

#[test]
fn both_languages_register_the_same_transforms_with_the_same_call_shape() {
    let fixture = fixture();
    let python = fixture["registry"].as_object().expect("registry");

    let rust: Vec<&str> = registered_features().to_vec();
    let mut py_names: Vec<&str> = python.keys().map(String::as_str).collect();
    py_names.sort_unstable();
    assert_eq!(
        rust, py_names,
        "the two registries do not hold the same transforms; a name in one and not \
         the other means a spec that loads in Python is unservable in Rust, or worse, \
         that Rust silently has a transform nobody gated"
    );

    for name in &rust {
        let entry = &python[*name];
        let info = feature_info(name).unwrap();

        let py_inputs: Vec<&str> = entry["inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // Order matters and is not sorted away: `inputs` is the *positional* call
        // shape, and swapping two of them binds every spec's arrays to the wrong
        // slots while every name stays correct.
        assert_eq!(
            info.inputs.to_vec(),
            py_inputs,
            "{name}: input order differs between the languages"
        );

        let py_params: Vec<&str> = entry["params"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        // Params *are* compared sorted, because the two sides legitimately declare
        // them in different orders: Python's is the function signature's, Rust's is
        // sorted so it matches the order a spec hashes them in. The fixture writes
        // Python's sorted for the same reason.
        assert_eq!(
            info.params.to_vec(),
            py_params,
            "{name}: parameter sets differ between the languages"
        );
    }
}

#[test]
fn the_two_libraries_bump_their_version_in_lockstep() {
    // A version only one side bumps is worse than no version: every fingerprint
    // disagrees at once, and the diagnosis points at the recipe instead of at the
    // mismatch. `FEATURES_VERSION` is folded into every spec hash, which is exactly
    // why it cannot be allowed to drift.
    let fixture = fixture();
    assert_eq!(
        u64::from(FEATURES_VERSION),
        fixture["features_version"].as_u64().unwrap()
    );
}

// ── 2. what a spec is ─────────────────────────────────────────────────────────

#[test]
fn a_python_spec_reads_in_rust_and_re_identifies_to_the_same_fingerprint() {
    // The load-bearing cross-language claim. Rust recomputes the fingerprint from
    // its *own* canonical serialization of the recipe, so agreement means the two
    // languages hold the same object, not merely the same label for it.
    let fixture = fixture();
    for name in ["bar_m1", "perp_core"] {
        let entry = &fixture["specs"][name];
        let spec = FeatureSpec::from_json(entry["json"].as_str().unwrap())
            .unwrap_or_else(|e| panic!("{name} did not load in Rust: {e}"));
        assert_eq!(
            spec.fingerprint(),
            entry["fingerprint"].as_str().unwrap(),
            "{name}: Rust recomputed a different fingerprint from the same recipe"
        );
        assert_eq!(spec.reference(), entry["ref"].as_str().unwrap());
    }
}

#[test]
fn rusts_canonical_serialization_is_byte_identical_to_pythons() {
    // Stronger than the fingerprint check and worth having beside it: two different
    // byte strings could in principle hash the same, and — far more likely — a
    // difference here is readable, while a fingerprint mismatch is sixteen hex
    // characters with no clue attached.
    let fixture = fixture();
    for name in ["bar_m1", "perp_core"] {
        let python = fixture["specs"][name]["json"].as_str().unwrap();
        let spec = FeatureSpec::from_json(python).unwrap();
        assert_eq!(
            spec.canonical_json(),
            python,
            "{name}: Rust re-serializes the spec differently from Python"
        );
    }
}

#[test]
fn the_columns_and_required_inputs_agree_across_the_boundary() {
    let fixture = fixture();
    for name in ["bar_m1", "perp_core"] {
        let entry = &fixture["specs"][name];
        let spec = FeatureSpec::from_json(entry["json"].as_str().unwrap()).unwrap();
        let py_columns: Vec<&str> = entry["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(spec.columns(), py_columns, "{name}: column order differs");

        let py_inputs: Vec<String> = entry["required_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            spec.required_inputs().unwrap(),
            py_inputs,
            "{name}: the two sides disagree about which arrays a caller must supply"
        );
    }
}

#[test]
fn the_bar_spec_warmup_rust_derives_is_the_one_python_declares() {
    // `BAR_M1_WARMUP_BARS = 21` is a constant on the Python side and a *derivation*
    // on the Rust side. Pinning them against each other is what catches the day
    // somebody widens a window: Python's constant goes stale silently, Rust's
    // derivation moves, and this reddens.
    let fixture = fixture();
    let spec =
        FeatureSpec::from_json(fixture["specs"]["bar_m1"]["json"].as_str().unwrap()).unwrap();
    assert_eq!(
        spec.max_lookback().unwrap(),
        Some(fixture["bar_m1_warmup_bars"].as_u64().unwrap() as usize)
    );
}

#[test]
fn the_reference_perp_spec_cannot_be_served_from_a_bounded_buffer() {
    // Not an edge case — a finding about the repo's own reference spec.
    // `PERP_CORE_V1` carries `ema_crossover`, and an EMA never forgets its seed, so
    // no bounded Rust buffer reproduces the offline recompute. The batch path
    // computes it faithfully; the streaming path refuses it. Recorded as a test so
    // that the day somebody swaps that column for `sma_crossover`, the change in
    // what is servable is visible rather than incidental.
    let fixture = fixture();
    let perp =
        FeatureSpec::from_json(fixture["specs"]["perp_core"]["json"].as_str().unwrap()).unwrap();
    assert_eq!(perp.max_lookback().unwrap(), None);
    assert_eq!(perp.column_lookback("ema_x_8_32").unwrap(), None);
    // Every other column in it is bounded; it is one column that costs the spec.
    for column in [
        "mid",
        "spread_bps",
        "book_imb",
        "ret_1",
        "mom_32",
        "vol_32",
        "z_32",
        "tfi_32",
    ] {
        assert!(
            perp.column_lookback(column).unwrap().is_some(),
            "{column} was unbounded too; the finding is that exactly one column is"
        );
    }

    let bar = FeatureSpec::from_json(fixture["specs"]["bar_m1"]["json"].as_str().unwrap()).unwrap();
    assert_eq!(bar.max_lookback().unwrap(), Some(21));
}

// ── 3. NumPy's reduction order ────────────────────────────────────────────────

#[test]
fn the_pairwise_transcription_reproduces_numpy_to_the_bit() {
    // The claim the crate rests on. `numeric.rs` transcribes NumPy's
    // `DOUBLE_pairwise_sum` from its source; this pins the transcription against
    // NumPy's actual output, which is the only thing that can catch the day NumPy
    // changes its unroll factor or its block size.
    let fixture = fixture();
    let xs = series(&fixture);
    let cases = fixture["reductions"].as_array().unwrap();
    assert!(!cases.is_empty(), "the fixture recorded no reductions");

    for case in cases {
        let window = case["window"].as_u64().unwrap() as usize;
        let w = &xs[xs.len() - window..];

        let sum = axon_features::numeric::pairwise_sum(w);
        assert_eq!(
            sum.to_bits(),
            from_bits(case["sum"].as_str().unwrap()).to_bits(),
            "window {window}: pairwise_sum does not reproduce np.sum"
        );

        let mean = axon_features::numeric::mean(w);
        assert_eq!(
            mean.to_bits(),
            from_bits(case["mean"].as_str().unwrap()).to_bits(),
            "window {window}: mean does not reproduce np.mean"
        );

        for (ddof, key) in [(0usize, "std_ddof0"), (1usize, "std_ddof1")] {
            let std = axon_features::numeric::std(w, ddof);
            assert_eq!(
                std.to_bits(),
                from_bits(case[key].as_str().unwrap()).to_bits(),
                "window {window}, ddof {ddof}: std does not reproduce np.std"
            );
        }
    }
}

#[test]
fn a_naive_summation_would_have_failed_this_gate() {
    // The negative that makes the positive mean something. If left-to-right
    // accumulation happened to agree with NumPy everywhere, the transcription in
    // `numeric.rs` would be unnecessary and the test above would be passing for free.
    // The fixture records, per window, whether the two orders actually separate —
    // measured by NumPy, not asserted here — and this checks that at least one
    // window separates and that `numeric` lands on NumPy's side of every one that
    // does.
    let fixture = fixture();
    let xs = series(&fixture);
    let mut separating = 0;

    for case in fixture["reductions"].as_array().unwrap() {
        if !case["naive_separates"].as_bool().unwrap() {
            continue;
        }
        separating += 1;
        let window = case["window"].as_u64().unwrap() as usize;
        let w = &xs[xs.len() - window..];
        let naive = w.iter().fold(0.0f64, |acc, v| acc + v);
        let recorded_naive = from_bits(case["naive_sum"].as_str().unwrap());
        assert_eq!(
            naive.to_bits(),
            recorded_naive.to_bits(),
            "window {window}: Rust's naive fold does not match Python's, so this test \
             is not comparing the two orders it thinks it is"
        );
        assert_ne!(
            axon_features::numeric::pairwise_sum(w).to_bits(),
            naive.to_bits(),
            "window {window}: the fixture says NumPy and naive summation differ here, \
             but the Rust pairwise sum agreed with naive — the transcription has \
             fallen back to the loop"
        );
    }

    assert!(
        separating >= 3,
        "only {separating} of the fixture's windows separate pairwise from naive \
         summation; the fixture no longer exercises the difference this crate exists \
         to reproduce"
    );
}

#[test]
fn a_window_summing_to_negative_zero_comes_back_positive_because_numpy_adds_its_identity() {
    // A real defect in this crate, caught by differential fuzzing and pinned here.
    //
    // `np.sum` is not `DOUBLE_pairwise_sum` — it is `identity + DOUBLE_pairwise_sum(...)`,
    // because `np.add.reduce` seeds its accumulator with the ufunc identity `0.0`. That
    // outer add is exact for every finite value with exactly one exception:
    // `0.0 + (-0.0)` is `+0.0`. A faithful transcription of the inner function alone
    // returns `-0.0`, one bit off, on the one value a bit-exact gate cannot shrug at.
    //
    // The `n` values straddle all three branches. Note `n < 8`: NumPy's plain loop
    // already starts from `+0.0`, so that branch never had the bug — which is why a
    // test on short windows alone would have passed throughout.
    let fixture = fixture();
    let cases = fixture["negative_zero"].as_array().expect("negative_zero");
    assert!(
        !cases.is_empty(),
        "the fixture recorded no negative-zero cases"
    );
    for case in cases {
        let n = case["n"].as_u64().unwrap() as usize;
        let all_negzero = vec![-0.0f64; n];
        for (key, got) in [
            (
                "all_negzero_sum",
                axon_features::numeric::pairwise_sum(&all_negzero),
            ),
            (
                "all_negzero_mean",
                axon_features::numeric::mean(&all_negzero),
            ),
            (
                "all_negzero_std",
                axon_features::numeric::std(&all_negzero, 0),
            ),
        ] {
            let want = from_bits(case[key].as_str().unwrap());
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "n={n} {key}: NumPy says {want} ({:#018x}), this build says {got} ({:#018x}) \
                 — the sign of zero is the whole finding",
                want.to_bits(),
                got.to_bits()
            );
        }
        let mut mixed = vec![-0.0f64; n];
        if n >= 2 {
            mixed[n / 2] = 0.0;
        }
        let want = from_bits(case["mixed_zero_sum"].as_str().unwrap());
        assert_eq!(
            axon_features::numeric::pairwise_sum(&mixed).to_bits(),
            want.to_bits(),
            "n={n}: a single +0.0 among negative zeros"
        );
    }

    // And the negative half, so the test is not passing because everything is +0.0
    // everywhere: the *inner* transcription really does produce -0.0, which is what
    // makes the identity load-bearing rather than decorative. Checked through the one
    // window size where the plain-loop branch cannot mask it.
    let neg = vec![-0.0f64; 20];
    assert!(
        !axon_features::numeric::pairwise_sum(&neg).is_sign_negative(),
        "the identity was not applied"
    );
    assert!(
        (-0.0f64 + -0.0f64).is_sign_negative(),
        "the platform stopped preserving the sign of zero under addition, which is \
         what made this a bug in the first place"
    );
}

// ── 4. the one operation that is measured rather than guaranteed ──────────────

#[test]
fn rusts_natural_log_agrees_with_numpys_bit_for_bit_on_this_platform() {
    // IEEE-754 requires `+ - * /` and `sqrt` to be correctly rounded; it says
    // nothing about `log`. NumPy and this platform's libm agree here because both
    // compute it well, not because either promises to — measured at 0 ULP over
    // the 32 ratios below, and over a 200 000-sample sweep across 26 decades that
    // `scripts/modal_libm_probe.py` re-runs on demand — here, and in glibc-2.36 and
    // musl containers.
    //
    // **If this file ever reddens on a different machine, start here.** A feature
    // bundle's manifest carries a `libm_columns` list for exactly this reason: it
    // names the columns whose value passes through `log`, so a platform-specific
    // failure can be recognised as one instead of read as a broken transform.
    let fixture = fixture();
    let cases = fixture["logs"].as_array().unwrap();
    assert!(!cases.is_empty(), "the fixture recorded no logarithms");
    for case in cases {
        let ratio = from_bits(case["ratio"].as_str().unwrap());
        let expected = from_bits(case["log"].as_str().unwrap());
        assert_eq!(
            ratio.ln().to_bits(),
            expected.to_bits(),
            "ln({ratio}) disagrees with np.log; this is the one operation neither \
             language guarantees, and the feature gate's bit-exactness depends on it"
        );
    }
}

#[test]
fn every_test_file_this_crate_names_in_prose_actually_exists() {
    // This crate's doc comments carry their weight by naming the test that asserts
    // each claim — "asserted against it in `tests/…`" rather than "we check this".
    // That is only worth anything if the pointer resolves, and four of them did not:
    // `cross_language_registry.rs` and `cross_language_spec.rs` were named in
    // `registry.rs`, `lib.rs` and twice in `spec.rs`, and neither file has ever
    // existed in this repository's history. They were written while the tests were
    // still planned as separate files, and the consolidation into *this* file never
    // reached the prose.
    //
    // Two of the four ship to docs.rs as public API documentation, so a reader
    // following them finds nothing and reasonably concludes the cross-language
    // registry gate was never written. Nothing else can catch this: it compiles, the
    // assertions genuinely run, and every other test passes.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0;
    let mut missing = Vec::new();
    for entry in walk(&root.join("src"))
        .into_iter()
        .chain(walk(&root.join("tests")))
    {
        let text = std::fs::read_to_string(&entry).expect("a source file this crate owns");
        // Deliberately narrow: a backtick-quoted path under tests/ ending in .rs, which
        // is the one shape this crate uses to point at a gate. A looser regex would start
        // matching prose about files in other crates and produce false alarms — and a
        // false alarm here is the failure that gets the check deleted.
        for (line, cited) in text.lines().enumerate().flat_map(|(i, l)| {
            l.match_indices("`tests/").map(move |(at, _)| {
                let rest = &l[at + 1..];
                let end = rest.find('`').unwrap_or(rest.len());
                (i + 1, rest[..end].to_string())
            })
        }) {
            // A citation may name a specific test with `::`; only the leading path is
            // a file, and the split has to happen *before* the `.rs` test — checking
            // the suffix first silently skips every citation that names a test, which
            // is five of this crate's eleven and the more useful half of them.
            let path = cited.split("::").next().unwrap_or(&cited);
            if !path.ends_with(".rs") {
                continue;
            }
            checked += 1;
            if !root.join(path).exists() {
                missing.push(format!("{}:{line} cites {path}", entry.display()));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "doc comments name test files that do not exist:\n  {}",
        missing.join("\n  ")
    );
    // A floor, measured rather than guessed: this crate carries eleven such citations
    // today. The assertion is not that the number is eleven — prose is allowed to grow
    // and shrink — but that the sweep is still *finding* prose. A parser that quietly
    // stopped matching would report zero dangling pointers, which is the same green a
    // perfect repository produces. That is not hypothetical: the first version of this
    // test tested the `.rs` suffix before splitting on `::` and silently skipped the
    // five citations that name a specific test.
    assert!(
        checked >= 11,
        "only {checked} of this crate's citations were seen; the sweep has stopped \
         reading the prose it exists to police, which is green for the same reason an \
         empty gate is"
    );
}

/// Every `.rs` under `dir`, recursively. `walkdir` is not a dependency of this crate
/// and is not worth becoming one for a single test.
fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    out
}

// ── the whole path, end to end ────────────────────────────────────────────────

#[test]
fn the_bar_spec_computes_in_rust_from_python_arrays_with_the_derived_warmup() {
    // Not a parity check — that is `feature_parity.rs`, over real market data with
    // Python's own answers. This one only asserts the pieces fit together: a spec
    // that crossed as JSON, computed by the Rust runtime, warms up exactly where the
    // derivation says it should.
    let fixture = fixture();
    let spec =
        FeatureSpec::from_json(fixture["specs"]["bar_m1"]["json"].as_str().unwrap()).unwrap();
    let close = series(&fixture);
    let high: Vec<f64> = close.iter().map(|c| c + 4.5).collect();
    let low: Vec<f64> = close.iter().map(|c| c - 4.5).collect();
    let inputs = BTreeMap::from([
        ("close".to_string(), close.clone()),
        ("high".to_string(), high),
        ("low".to_string(), low),
    ]);

    let matrix = spec.compute(&inputs).unwrap();
    assert_eq!(matrix.rows(), close.len());
    assert_eq!(matrix.cols(), 6);

    let warmup = spec.max_lookback().unwrap().unwrap();
    assert_eq!(matrix.finite_rows(), close.len() - (warmup - 1));
}
