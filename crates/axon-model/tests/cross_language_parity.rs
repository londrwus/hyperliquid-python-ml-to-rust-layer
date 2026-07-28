//! The cross-language model-parity gate (ADR-0021), from the Rust side.
//!
//! `tests/parity.rs` asserts this crate reproduces a frozen table of numbers.
//! This file asserts something stricter and more useful: that the *artifact a
//! registry would hand production* (ADR-0015), scored through the Rust backend
//! that will serve it, produces the same **decisions** as the Python that
//! researched it. The bundles under `tests/bundles/` were written by
//! `axon.parity.rust_gate` from real registry artifacts and are committed —
//! regenerating a reference in the same breath as asserting it proves nothing.
//!
//! Everything here is offline: no network, no Python, no ML libraries, no clock.
//! The default `cargo test` gate runs it on a machine that could not load
//! xgboost if it wanted to, which is the point — the gate has to be cheap enough
//! that nobody is tempted to skip it.
//!
//! Two tests carry the weight and belong together:
//! `a_sub_tolerance_delta_that_crosses_the_trade_threshold_fails_the_gate` and
//! `the_same_delta_inside_the_flat_band_passes`. Same magnitude, opposite
//! verdicts — because the size of a delta says nothing about whether it moved
//! money. That is ADR-0016 §3's argument, carried across the language boundary,
//! which is the only place it had not yet been made.

use std::fs;
use std::path::{Path, PathBuf};

use axon_model::parity::{
    BundleError, Criterion, Decision, ModelKind, ParityBundle, ONNX_EPS, ONNX_TIGHT_EPS,
};

/// Every committed bundle, in sorted order — `every_committed_bundle_is_in_the_gate`
/// compares against a sorted directory listing. Named explicitly rather than
/// globbed so that a directory that vanishes is a failing test rather than a
/// silently smaller gate; that test closes the other direction.
const BUNDLES: &[&str] = &[
    "lgbm_binary",
    "mlp_regressor",
    "tree_identity",
    "tree_logistic",
    "zoo_logistic",
    "zoo_xgboost",
];

fn bundles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bundles")
}

fn bundle(name: &str) -> ParityBundle {
    ParityBundle::open(bundles_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// A writable copy of a committed bundle, for the tests that have to break one.
/// Under `CARGO_TARGET_TMPDIR` so a failed run leaves the evidence in `target/`
/// rather than in the source tree, where the next `cargo test` would gate it.
fn corrupted(source: &str, scratch: &str, edit: impl FnOnce(&Path)) -> PathBuf {
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

/// The row whose score sits furthest from either threshold, so a perturbation
/// there fails the numeric criterion *without* flipping a decision. Isolating
/// the two halves of the gate is the whole reason they are an `and`.
fn deepest_inside_a_band(bundle: &ParityBundle) -> usize {
    let decision = bundle.decision();
    let margin = |score: f32| {
        (score - decision.long_at())
            .abs()
            .min((score - decision.short_at()).abs())
    };
    let mut best = 0;
    for (row, score) in bundle.reference().iter().enumerate() {
        if margin(*score) > margin(bundle.reference()[best]) {
            best = row;
        }
    }
    best
}

#[test]
fn rust_reproduces_pythons_answer_for_every_row_of_every_bundle() {
    for name in BUNDLES {
        let bundle = bundle(name);
        let model = bundle
            .load_model()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        // The version is read out of the artifact, never handed in, so this
        // equality is what ties the frozen answers to these exact bytes.
        assert_eq!(model.version(), bundle.model_version(), "{name}");
        assert_eq!(model.input_len(), bundle.feature_width(), "{name}");
        assert!(!bundle.feature_spec_ref().is_empty(), "{name}");

        let candidate = bundle
            .candidate(model.as_ref())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let report = bundle.compare(&candidate).expect("compare");
        // Spelled out rather than left to `passed()`, because the two criteria
        // are an `and` and a reader of this test should see both of them.
        assert_eq!(report.non_finite(), 0, "{}", report.summary());
        assert_eq!(report.flips(), 0, "{}", report.summary());
        report.assert_passed();
    }
}

#[test]
fn tree_bundles_are_gated_at_bit_equality_and_graphs_at_the_documented_tolerance() {
    // The criterion is not the manifest's opinion: it is what the family allows
    // (ADR-0019 claims *bits* for trees, so a tolerance there would be the gate
    // declining to test its own claim).
    assert_eq!(bundle("tree_identity").criterion(), Criterion::BitExact);
    assert_eq!(bundle("tree_logistic").criterion(), Criterion::BitExact);
    assert_eq!(
        bundle("mlp_regressor").criterion(),
        Criterion::MaxAbsDiff(ONNX_TIGHT_EPS)
    );
}

#[test]
fn tree_bundles_carry_missing_values_so_the_default_direction_is_gated() {
    // Without this the bit-equality above could pass while the missing-value
    // branch went entirely unexercised — `NaN < threshold` is false, so a reader
    // that compares before testing for NaN agrees with XGBoost on exactly the
    // nodes whose default happens to be right. The corpus, not the assertion, is
    // what makes that branch part of the gate.
    for name in ["tree_identity", "tree_logistic"] {
        assert!(
            bundle(name).missing_cells() > 0,
            "{name} holdout has no missing features"
        );
    }
}

#[test]
fn every_bundle_decides_more_than_one_way() {
    // A corpus where every row is flat would pass the decision half of the gate
    // no matter what the model did to the numbers. A check that can only come
    // out one way is a decoration.
    for name in BUNDLES {
        let bundle = bundle(name);
        let sides = bundle.reference_decisions();
        assert!(
            sides.iter().any(|s| *s != sides[0]),
            "{name}: every row decides {}",
            sides[0]
        );
    }
}

#[test]
fn every_committed_bundle_is_in_the_gate() {
    // The other direction from `BUNDLES` being a literal list: a bundle added to
    // the tree and forgotten here would sit in git looking like coverage while
    // gating nothing.
    let mut found: Vec<String> = fs::read_dir(bundles_dir())
        .expect("bundles dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("manifest.json").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    assert_eq!(found, BUNDLES, "tests/bundles/ and BUNDLES disagree");
}

#[test]
fn the_zoos_own_artifact_crosses_and_it_was_never_hand_stamped() {
    // Every other bundle in this directory was written by a generator that stamped
    // the model version itself — `booster.set_attr(axon_model_version=...)`, right
    // beside the assertion. So the gate that certifies Boundary A had never once run
    // on the export path it certifies: `export_artifact` learned to write the version
    // into the payload before hashing it, and nothing in `cargo test` had ever loaded
    // an artifact that came out of it untouched. This one did. It is
    // `axon.strategies.zoo`'s XGBoost family, fitted on the committed 900-bar m1
    // fixtures, exported through `export_family` -> `export_artifact` -> the registry,
    // and frozen with no hand afterwards (`tests/bundles/generate_zoo.py`).
    //
    // The version assertion is the load-bearing one. `TreeModel` reads
    // `axon_model_version` out of the learner attributes and refuses an artifact that
    // has none (ADR-0019 §4), so a regression that dropped the stamp would fail here
    // as a *load* error — which is the failure mode, since an unstamped artifact
    // cannot be tied back to the signals it produced.
    let bundle = bundle("zoo_xgboost");
    assert_eq!(bundle.kind(), ModelKind::Xgboost);
    assert_eq!(bundle.model_version(), 1);
    assert_eq!(bundle.registry_id(), "zoo_xgboost");
    // The recipe, not a probe: these rows came off `BAR_M1_V1` over real candles,
    // unlike the synthetic gaussians the other tree bundles gate.
    assert!(
        bundle.feature_spec_ref().starts_with("bar_m1/v1#"),
        "feature_spec_ref is {}",
        bundle.feature_spec_ref()
    );
    assert_eq!(bundle.rows(), 348);
    assert_eq!(bundle.feature_width(), 6);
    assert_eq!(bundle.criterion(), Criterion::BitExact);

    let model = bundle.load_model().expect("zoo_xgboost loads");
    assert_eq!(model.version(), 1, "the version came out of the payload");

    let report = bundle.check().expect("zoo_xgboost");
    assert_eq!(report.rows(), 348, "{}", report.summary());
    assert_eq!(report.over_criterion(), 0, "{}", report.summary());
    assert_eq!(report.flips(), 0, "{}", report.summary());
    assert_eq!(report.non_finite(), 0, "{}", report.summary());
    assert!(report.passed(), "{}", report.summary());

    // Not "within a tolerance" — zero, to the bit, over all 348 rows. ADR-0019 claims
    // `TreeModel` reproduces `Booster.predict(output_margin=True)` exactly, and this
    // is the first time that claim has been made against an artifact a strategy's own
    // export path produced rather than against a fixture built to be reproduced.
    assert_eq!(
        report.max_abs_diff().to_bits(),
        0.0f32.to_bits(),
        "{}",
        report.summary()
    );
}

#[test]
fn the_onnx_half_of_the_unassisted_version_stamp_is_gated_too() {
    // `zoo_xgboost` beside this proves the export path stamps a version without a
    // hand — but only through `axon_model_version`, an XGBoost learner attribute.
    // `export_artifact` writes ONNX's first-class `model_version` field from an
    // entirely different branch, and until this bundle landed nothing in `cargo test`
    // had loaded an artifact off it. The two fail differently and both are silent
    // until a load: an unstamped graph leaves `model_version` at 0, which reads as
    // "unset" and is refused (ADR-0019 §4).
    //
    // This is also the zoo's ONNX *route*, which `zoo_xgboost` cannot exercise at all:
    // skl2onnx to an `ai.onnx.ml` LinearClassifier, narrowed to one score column, run
    // by `tract`. Three third-party things have to keep holding for the zoo's linear
    // family to reach Rust, and none of them is ours.
    let bundle = bundle("zoo_logistic");
    assert_eq!(bundle.kind(), ModelKind::Onnx);
    assert_eq!(bundle.model_version(), 1);
    assert_eq!(bundle.registry_id(), "zoo_logistic");
    assert!(
        bundle.feature_spec_ref().starts_with("bar_m1/v1#"),
        "feature_spec_ref is {}",
        bundle.feature_spec_ref()
    );
    assert_eq!(bundle.rows(), 348);
    assert_eq!(bundle.feature_width(), 6);
    assert_eq!(bundle.criterion(), Criterion::MaxAbsDiff(ONNX_TIGHT_EPS));

    let model = bundle.load_model().expect("zoo_logistic loads");
    assert_eq!(model.version(), 1, "the version came out of the graph");
    // One value per row. An skl2onnx classifier emits two class probabilities, and a
    // graph that reached here un-narrowed would be refused at load rather than served
    // with a column picked for it — so this is the narrowing having happened, seen
    // from the far side of the boundary.
    assert_eq!(model.output_len(), 1);

    // The reference is in probability space, which is what `score_space: "score"`
    // claims and what the strategy trades on. A margin would sit outside [0, 1] and
    // would mean the graph lost its LOGISTIC post-transform on the way out — the
    // link-versus-margin trap that costs an afternoon every time it is met.
    //
    // Note what this does *not* prove: both languages agree on whichever column was
    // narrowed to, so an export that took `P(down)` would pass here with every number
    // still in [0, 1]. That trap is pinned where it can be — `positive_class` in the
    // zoo, and `logistic_probabilities.onnx` in tests/refusals/, which asserts an
    // un-narrowed two-column graph is refused rather than silently read at column 0.
    for (row, score) in bundle.reference().iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(score),
            "row {row} scores {score}, which is not a probability"
        );
    }

    let report = bundle.check().expect("zoo_logistic");
    assert_eq!(report.rows(), 348, "{}", report.summary());
    assert_eq!(report.over_criterion(), 0, "{}", report.summary());
    assert_eq!(report.flips(), 0, "{}", report.summary());
    assert_eq!(report.non_finite(), 0, "{}", report.summary());
    assert!(report.passed(), "{}", report.summary());

    // Measured on this bundle, not inherited from anyone else's holdout. Pinned
    // exactly for the reason ADR-0019 §1 pins `tract-onnx` exactly: a patch bump
    // could move a result inside the tolerance with no code change to blame.
    assert_eq!(
        report.max_abs_diff().to_bits(),
        8.940697e-8f32.to_bits(),
        "max_abs_diff moved to {:e}; {}",
        report.max_abs_diff(),
        report.summary()
    );
}

#[test]
fn a_tract_upgrade_that_moved_lightgbms_numbers_would_redden_this_gate() {
    // LightGBM is the one family in the zoo whose crossing rests entirely on
    // third-party arithmetic this crate does not own: `tract` implements
    // `ai.onnx.ml.TreeEnsembleClassifier` and does *not* implement
    // `TreeEnsembleRegressor`, so a LightGBM booster reaches Rust only when its
    // converter emits the classifier form. Nothing about that is ours to keep
    // true, and until this bundle was promoted here the claim was provable only
    // by pytest — so a `tract` bump that moved the numbers would have gone
    // unnoticed by `cargo test` on a machine with no Python.
    //
    // The exact figures are asserted rather than a tolerance, and that follows
    // from ADR-0019 §1's reason for pinning `tract-onnx = "=0.23.4"` exactly:
    // "a silent patch bump could move a result inside the tolerance with no code
    // change to blame it on." Pinning the crate and then accepting any number
    // under 1e-5 would be making that argument and declining to act on it.
    let bundle = bundle("lgbm_binary");
    assert_eq!(bundle.kind(), ModelKind::Onnx);
    assert_eq!(bundle.rows(), 256);
    assert_eq!(bundle.criterion(), Criterion::MaxAbsDiff(ONNX_TIGHT_EPS));

    let report = bundle.check().expect("lgbm_binary");
    assert_eq!(report.rows(), 256, "{}", report.summary());
    assert_eq!(report.over_criterion(), 0, "{}", report.summary());
    assert_eq!(report.flips(), 0, "{}", report.summary());
    assert_eq!(report.non_finite(), 0, "{}", report.summary());
    assert!(report.passed(), "{}", report.summary());

    // 1.1920929e-7 is 2^-23 — one ULP at 1.0, the smallest disagreement a
    // probability in [0, 1] can show. If this moves, the cause is a `tract`
    // version change or a regenerated bundle, and both are events someone must
    // look at rather than absorb.
    assert_eq!(
        report.max_abs_diff().to_bits(),
        1.1920929e-7f32.to_bits(),
        "max_abs_diff moved to {:e}; {}",
        report.max_abs_diff(),
        report.summary()
    );
}

#[test]
fn one_ulp_of_drift_fails_the_tree_gate_and_names_the_row() {
    // The gate has to be seen to fail, and this is the failure ADR-0019's tree
    // reader exists to prevent: one ULP is what a widened threshold comparison,
    // an f64 accumulator or an f64 logit intercept each cost.
    let bundle = bundle("tree_identity");
    let row = deepest_inside_a_band(&bundle);
    let mut candidate = bundle.reference().to_vec();
    candidate[row] = f32::from_bits(candidate[row].to_bits() + 1);

    let report = bundle.compare(&candidate).expect("compare");
    assert!(!report.passed(), "{}", report.summary());
    // Numeric criterion only: a ULP deep inside a band moves no decision, which
    // is why bit equality — not a tolerance — is what caught it.
    assert_eq!(report.flips(), 0, "{}", report.summary());
    assert!(report.max_abs_diff() < 1e-5);

    let divergence = report.divergences().first().expect("a divergent row");
    assert_eq!(divergence.row, row);
    assert!(divergence.over_criterion && !divergence.flipped);
    let summary = report.summary();
    assert!(summary.contains(&format!("row {row}")), "{summary}");
    // The bit patterns have to be in the message: at one ULP the two decimals
    // routinely print identically and the failure would read as a lie.
    assert!(
        summary.contains(&format!("{:#010x}", candidate[row].to_bits())),
        "{summary}"
    );
}

#[test]
fn a_committed_graph_bundle_declares_two_ulp_and_the_ceiling_still_only_refuses_looser() {
    // What tightening did and did not do, pinned so neither half is mistaken for
    // the other.
    //
    // It DID: every graph bundle committed here now declares two ULP, so the ~42x
    // of slack between what these graphs achieve and what ADR-0003 §3 permits is no
    // longer a place a regression can hide. `mlp_regressor` had been declaring 1e-5
    // while measuring 0e0 — five orders of magnitude of room.
    // A `const` block, so the relationship between the two constants is checked when
    // this file compiles rather than when it runs: a build in which the declared
    // criterion had drifted up to the ceiling should not get as far as producing a
    // test binary that could be filtered out.
    const {
        assert!(
            ONNX_TIGHT_EPS < ONNX_EPS,
            "the declared criterion must be tighter than the family ceiling"
        )
    };
    for name in ["lgbm_binary", "mlp_regressor", "zoo_logistic"] {
        assert_eq!(
            bundle(name).criterion(),
            Criterion::MaxAbsDiff(ONNX_TIGHT_EPS),
            "{name} declares the family ceiling instead of what graphs here achieve"
        );
    }

    // It did NOT move the ceiling, and that is the deliberate half. `required_for`
    // is what a *reader* enforces; two ULP is what a *writer* here declares. So a
    // bundle from anywhere else that declares the family's 1e-5 is still perfectly
    // readable -- it is not loosened, it is untightened, and the reader has no way
    // to tell those apart. What keeps our own bundles honest is the assertion
    // above, not the ceiling.
    assert!(
        Criterion::required_for(ModelKind::Onnx).allows(Criterion::MaxAbsDiff(ONNX_EPS)),
        "the ceiling still admits a bundle declaring exactly the ceiling"
    );
    assert!(
        Criterion::required_for(ModelKind::Onnx).allows(Criterion::MaxAbsDiff(ONNX_TIGHT_EPS)),
        "and it admits the tighter one, which is why tightening needed no reader change"
    );
}

/// A delta strictly inside `bundle`'s own declared criterion.
///
/// Derived rather than written down, because a literal here is a literal that
/// goes stale the day the declared criterion tightens — and it did: the two
/// callers below were written against `1e-6` when a graph bundle declared the
/// 1e-5 family ceiling, and both broke on the move to two ULP for a reason
/// unrelated to the property they test.
fn sub_tolerance_delta(bundle: &ParityBundle) -> f32 {
    match bundle.criterion() {
        Criterion::MaxAbsDiff(eps) => eps / 4.0,
        Criterion::BitExact => {
            panic!("a bit-exact bundle has no sub-tolerance delta; every delta is over it")
        }
    }
}

/// Rewrite a manifest's `criterion.eps` in place, failing loudly if nothing moved.
///
/// The failure this prevents is one this file already suffered. The corruption
/// below was written as `.replace("1e-05", "0.01")` against a manifest declaring
/// the family ceiling; the day the writer began declaring two ULP instead, the
/// pattern matched nothing, the "corrupted" bundle was a byte-identical copy, and
/// the assertion about refusing a loosened gate was asking a question of a bundle
/// nobody had loosened. A corruption helper that does not corrupt reads as
/// protection while testing nothing, so this one asserts it changed the bytes.
fn set_manifest_eps(dir: &Path, eps: &str) {
    let path = dir.join("manifest.json");
    let text = fs::read_to_string(&path).unwrap();
    let key = "\"eps\": ";
    let at = text.find(key).expect("the manifest declares an eps") + key.len();
    let end = at + text[at..].find([',', '\n']).expect("the eps value ends");
    let patched = format!("{}{}{}", &text[..at], eps, &text[end..]);
    assert_ne!(patched, text, "the manifest was not actually changed");
    fs::write(path, patched).unwrap();
}

#[test]
fn a_sub_tolerance_delta_that_crosses_the_trade_threshold_fails_the_gate() {
    // ADR-0016 §3, across the boundary: `max_abs_diff` is well *under* the bundle's
    // own tolerance and the gate is still red, because the row went from flat to
    // long. The numeric criterion alone would have signed this deploy off.
    let bundle = bundle("mlp_regressor");
    let reference = bundle.reference()[0];
    // Taken from the bundle's own criterion rather than written as a literal. The
    // literal was `1e-6`, chosen when a graph bundle declared the 1e-5 family
    // ceiling; the day the writer began declaring two ULP instead, `1e-6` stopped
    // being sub-tolerance and this test failed for a reason that had nothing to do
    // with what it tests. A quarter of the criterion is four ULP at this row's
    // magnitude, so it still moves the score to a distinct float.
    let delta = sub_tolerance_delta(&bundle);
    let mut candidate = bundle.reference().to_vec();
    candidate[0] = reference + delta;

    let long_at = reference + delta / 2.0;
    let decision = Decision::new(long_at, long_at - 1.0).expect("thresholds");
    let report = bundle
        .compare_with(&candidate, bundle.criterion(), decision)
        .expect("compare");

    assert!(
        report.max_abs_diff() < ONNX_TIGHT_EPS,
        "{}",
        report.summary()
    );
    assert_eq!(report.flips(), 1, "{}", report.summary());
    assert!(!report.passed(), "{}", report.summary());
    let summary = report.summary();
    assert!(summary.contains("FLIPPED"), "{summary}");
    assert!(summary.contains("row 0"), "{summary}");
}

#[test]
fn the_same_delta_inside_the_flat_band_passes() {
    // The other half of the pair. Identical magnitude, thresholds far from the
    // score: nothing moved, so nothing fails. Without this, the test above would
    // only prove the gate is strict, not that it is discriminating.
    let bundle = bundle("mlp_regressor");
    let reference = bundle.reference()[0];
    let mut candidate = bundle.reference().to_vec();
    candidate[0] = reference + sub_tolerance_delta(&bundle);

    let decision = Decision::new(reference + 1.0, reference - 1.0).expect("thresholds");
    let report = bundle
        .compare_with(&candidate, bundle.criterion(), decision)
        .expect("compare");
    assert!(report.passed(), "{}", report.summary());
    assert!(report.max_abs_diff() > 0.0);
}

#[test]
fn a_bundle_cannot_buy_itself_a_looser_criterion() {
    // The failure this prevents is procedural, not numerical: a bundle
    // regenerated after a red gate with the tolerance nudged until it passed
    // would otherwise be indistinguishable from one that never failed.
    let loosened = corrupted("mlp_regressor", "loosened-eps", |dir| {
        set_manifest_eps(dir, "0.01");
    });
    assert!(
        matches!(ParityBundle::open(&loosened), Err(BundleError::Weakened(_))),
        "an ONNX bundle asking for 1e-2 must be refused"
    );

    let toleranced = corrupted("tree_identity", "tolerance-on-a-tree", |dir| {
        let path = dir.join("manifest.json");
        let text = fs::read_to_string(&path).unwrap().replace(
            "\"kind\": \"bit_exact\"",
            "\"kind\": \"max_abs_diff\", \"eps\": 1e-12",
        );
        fs::write(path, text).unwrap();
    });
    assert!(
        matches!(
            ParityBundle::open(&toleranced),
            Err(BundleError::Weakened(_))
        ),
        "even a 1e-12 tolerance is a tolerance, and trees are claimed exact"
    );
}

#[test]
fn a_truncated_matrix_is_refused_rather_than_read_as_a_shorter_corpus() {
    // Silently reading the surviving rows would leave a gate that passes on the
    // half of the holdout that made it to disk.
    let dir = corrupted("tree_logistic", "truncated-features", |dir| {
        let path = dir.join("features.f32");
        let mut bytes = fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 4);
        fs::write(path, bytes).unwrap();
    });
    assert!(matches!(
        ParityBundle::open(&dir),
        Err(BundleError::Malformed(_))
    ));
}

#[test]
fn decisions_that_do_not_follow_from_the_scores_are_refused_before_a_model_loads() {
    // The two languages must agree about the *rule*, not only the numbers. If
    // Rust discretizes Python's own scores differently — `>=` against `>`, a
    // threshold rounded to another float — every prediction could match to the
    // bit and the two systems would still trade differently.
    let dir = corrupted("tree_identity", "edited-decisions", |dir| {
        let path = dir.join("decisions.i8");
        let mut bytes = fs::read(&path).unwrap();
        let row = bytes
            .iter()
            .position(|b| *b as i8 == 1)
            .expect("a long row");
        bytes[row] = 0;
        fs::write(path, bytes).unwrap();
    });
    let err = ParityBundle::open(&dir).unwrap_err();
    assert!(
        matches!(err, BundleError::Malformed(_)) && err.to_string().contains("decision rule"),
        "expected a decision-rule refusal, got {err}"
    );
}

#[test]
fn a_model_of_another_version_cannot_answer_this_bundle() {
    // Both tree bundles take five features, so nothing about the shapes would
    // catch this: the version read out of the artifact is the only thing that
    // says the frozen answers came from these bytes.
    let question = bundle("tree_identity");
    let other = bundle("tree_logistic").load_model().expect("model");
    assert_eq!(question.kind(), ModelKind::Xgboost);
    let err = question.candidate(other.as_ref()).unwrap_err();
    assert!(
        matches!(err, BundleError::Mismatch(_)),
        "expected a version mismatch, got {err}"
    );
}

#[test]
fn a_candidate_of_the_wrong_length_is_refused_rather_than_compared_pairwise() {
    // A short candidate means the two runs did not see the same corpus. Zipping
    // them would compare the rows that happen to line up and call it parity.
    let bundle = bundle("tree_identity");
    let short = &bundle.reference()[..bundle.rows() - 1];
    assert!(matches!(
        bundle.compare(short),
        Err(BundleError::Mismatch(_))
    ));
}
