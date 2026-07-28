//! What this crate does with the model families it cannot serve (ADR-0019 §2).
//!
//! `tests/parity.rs` asks whether the backends reproduce Python on the families
//! they *do* serve. This file asks the opposite question, and it is the one with
//! the expensive wrong answer: when an artifact is outside the reader, does it
//! come back as a **named refusal**, or as a number nobody can tell is wrong?
//!
//! Every refusal here was already written and every one of them already had a
//! unit test — against a JSON blob edited by hand. That proves the reader
//! refuses the shape somebody imagined. It says nothing about the bytes
//! xgboost, skl2onnx and onnxmltools actually emit, and the difference was not
//! academic: **xgboost 3.3.0 writes `"name": "gbtree"` for a dart booster**, so
//! the name check that was supposed to catch dart had silently stopped
//! catching it, and a dropout-weighted ensemble loaded and served margins
//! roughly 1.6–1.9× too large. `tests/refusals/generate.py` produced these
//! artifacts through the real libraries; they are committed and the generator
//! does not run in CI, for the same reason the parity fixtures are frozen.
//!
//! Offline and deterministic: no network, no Python, no retraining.

use std::path::PathBuf;

use axon_model::{LoadError, Model, OnnxModel, TreeModel};
use serde::Deserialize;

fn refusal_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/refusals")
        .join(name)
}

/// ADR-0003 §3's tolerance for graphs, the bar an `onnx` parity bundle declares.
const ONNX_TOLERANCE: f32 = 1e-5;

#[derive(Deserialize)]
struct Reference {
    dart: DartCase,
    linear_control: LinearControl,
    refused_trees: Vec<String>,
}

#[derive(Deserialize)]
struct LinearControl {
    artifact: String,
    inputs: Vec<Vec<f32>>,
    /// onnxruntime's own answers, taken one row at a time with fusion disabled.
    onnxruntime_scores: Vec<f32>,
}

#[derive(Deserialize)]
struct DartCase {
    artifact: String,
    /// The per-tree dropout scale factors XGBoost recorded. If these were all
    /// 1.0 the unweighted sum would be right by coincidence and this fixture
    /// would prove nothing.
    weight_drop: Vec<f32>,
    /// What `gradient_booster.name` actually says on a dart artifact. Recorded
    /// rather than assumed, because assuming it is what went wrong.
    booster_name_in_json: String,
    /// `Booster.predict(output_margin=True)`, as IEEE-754 bit patterns.
    xgboost_margin_bits: Vec<String>,
    /// What a reader that ignored `weight_drop` would return instead.
    unweighted_sum_bits: Vec<String>,
}

fn reference() -> Reference {
    let raw = std::fs::read(refusal_fixture("reference.json")).expect("refusal reference fixture");
    serde_json::from_slice(&raw).expect("refusal reference fixture parses")
}

fn bits(hex: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(hex.trim_start_matches("0x"), 16).expect("bit pattern"))
}

/// The refusal must name what it could not represent. A load error that says
/// only "unsupported" sends the operator to the source instead of to the export
/// script.
fn assert_unsupported_mentioning(err: &LoadError, needle: &str, what: &str) {
    match err {
        LoadError::Unsupported(message) => assert!(
            message.contains(needle),
            "{what}: refusal does not name '{needle}': {message}"
        ),
        other => panic!("{what}: expected an Unsupported refusal, got {other}"),
    }
}

// ── XGBoost ──

#[test]
fn a_real_dart_artifact_is_refused_rather_than_summed_unweighted() {
    let case = reference().dart;
    let err = TreeModel::from_path(refusal_fixture(&case.artifact))
        .expect_err("a dart artifact must not load");
    assert_unsupported_mentioning(&err, "weight_drop", "dart");
}

#[test]
fn the_dart_fixture_would_still_be_a_wrong_answer_if_it_loaded() {
    // The point of the refusal, stated in numbers. XGBoost scales tree `i` by
    // `weight_drop[i]` at predict time; this backend's loop cannot, so if the
    // guard above ever comes off again the margins do not go slightly wrong,
    // they go wrong by a factor — and every one of them is a finite, plausible,
    // correctly-signed float. Without this assertion the refusal test would
    // still pass against a fixture whose dropout weights had drifted to 1.0,
    // which is exactly the fixture that proves nothing.
    let case = reference().dart;
    assert_eq!(
        case.booster_name_in_json, "gbtree",
        "xgboost now names the dart booster; the name check may be live again, but do not \
         remove the weight_drop check on the strength of one version"
    );
    assert!(
        case.weight_drop.iter().any(|w| *w != 1.0),
        "every dropout weight is 1.0, so an unweighted sum would be right by coincidence; \
         refit the fixture with one_drop"
    );
    let mut worst = 0.0f32;
    for (truth, naive) in case
        .xgboost_margin_bits
        .iter()
        .zip(case.unweighted_sum_bits.iter())
    {
        let (truth, naive) = (bits(truth), bits(naive));
        assert_ne!(truth.to_bits(), naive.to_bits());
        worst = worst.max((truth - naive).abs());
    }
    // Orders of magnitude past the 1e-5 an ONNX bundle is allowed, and trees
    // are held to the bit — but only if the artifact reaches the gate at all,
    // which is why this has to be caught at load.
    assert!(worst > 0.1, "the wrong answer is only {worst} away");
}

#[test]
fn a_real_categorical_split_is_refused_rather_than_read_as_a_threshold() {
    // XGBoost still writes a `split_conditions` entry on a categorical node —
    // a denormal holding a segment offset, not a threshold. Comparing a feature
    // against 1e-45 routes almost every row right and produces a complete,
    // wrong ensemble rather than an error.
    let err = TreeModel::from_path(refusal_fixture("categorical.json"))
        .expect_err("a categorical artifact must not load");
    assert_unsupported_mentioning(&err, "categorical split", "categorical");
}

#[test]
fn a_real_multiclass_ensemble_is_refused_rather_than_collapsed_to_one_margin() {
    // `multi:softmax` fits one tree per class per round and `tree_info` fans
    // them across output groups. Summing them all yields a margin that belongs
    // to no class at all.
    let err = TreeModel::from_path(refusal_fixture("multiclass.json"))
        .expect_err("a multi-class artifact must not load");
    assert_unsupported_mentioning(&err, "multi-output ensemble", "multi:softmax");
}

#[test]
fn both_multi_output_strategies_are_refused_not_only_the_vector_leaf_one() {
    // XGBoost has two ways to fit several targets and they fail differently:
    // `multi_output_tree` puts a vector in each leaf, `one_output_per_tree`
    // keeps scalar leaves and splits the ensemble across output groups. A
    // reader that only knew the first would serve the second as a single margin
    // built from half the trees — the shape of wrong answer that looks right.
    for artifact in ["multi_output_tree.json", "multi_output_one_per_tree.json"] {
        let err = TreeModel::from_path(refusal_fixture(artifact))
            .err()
            .unwrap_or_else(|| panic!("{artifact} loaded; a two-target ensemble must not"));
        assert_unsupported_mentioning(&err, "multi-output ensemble", artifact);
    }
}

#[test]
fn every_refused_tree_fixture_is_actually_refused() {
    // The list is in the reference, so adding a fixture to the generator
    // without a test here still fails: a fixture nobody asks about is a fixture
    // that stops covering anything the day the reader changes.
    for artifact in reference().refused_trees {
        let err = TreeModel::from_path(refusal_fixture(&artifact))
            .err()
            .unwrap_or_else(|| panic!("{artifact} loaded; it is in the refusal corpus"));
        assert!(
            matches!(err, LoadError::Unsupported(_)),
            "{artifact}: expected a named refusal, got {err}"
        );
    }
}

// ── ONNX ──

#[test]
fn a_graph_with_two_values_per_row_is_refused_rather_than_serving_column_zero() {
    // One output tensor, two class probabilities. This satisfied the old
    // "exactly one output" rule, loaded, and reported `output_len() == 2`.
    // Column 0 is P(class 0), so the first value is the model's answer with the
    // sign reversed — in range, finite, and backwards.
    let err = OnnxModel::from_path(refusal_fixture("logistic_probabilities.onnx"))
        .expect_err("a two-value graph must not load");
    assert_unsupported_mentioning(&err, "2 values per row", "logistic probabilities");

    let err = OnnxModel::from_path(refusal_fixture("three_values_per_row.onnx"))
        .expect_err("a three-value graph must not load");
    assert_unsupported_mentioning(&err, "3 values per row", "hand-built three-wide graph");
}

#[test]
fn the_skl2onnx_classifier_default_export_is_refused_for_its_second_tensor() {
    // What `to_onnx` gives you if you do nothing: an int64 `label` tensor
    // alongside the probabilities. ADR-0019 §1 refuses it because which tensor
    // feeds the strategy must be a property of the artifact.
    let err = OnnxModel::from_path(refusal_fixture("logistic_two_outputs.onnx"))
        .expect_err("a two-tensor graph must not load");
    assert_unsupported_mentioning(&err, "2 outputs", "skl2onnx default classifier export");
}

#[test]
fn a_converted_sklearn_linear_model_does_cross_into_rust() {
    // The control, and the reason the refusals above can be read as findings
    // about specific constructs rather than as "tract will not take anything
    // skl2onnx produces". `LinearRegression` converts to a single
    // `ai.onnx.ml.LinearRegressor` node with one value per row, `tract`
    // implements it, and the answers hold ADR-0003's graph tolerance against
    // onnxruntime — which is the bar an `onnx` parity bundle declares.
    let case = reference().linear_control;
    let model = OnnxModel::from_path(refusal_fixture(&case.artifact)).expect(&case.artifact);
    assert_eq!(model.output_len(), 1);

    let mut worst = 0.0f32;
    for (row, expected) in case.inputs.iter().zip(case.onnxruntime_scores.iter()) {
        let got = model.predict(row).expect("predict")[0];
        worst = worst.max((got - expected).abs());
    }
    assert!(
        worst <= ONNX_TOLERANCE,
        "{}: max_abs_diff {worst} exceeds {ONNX_TOLERANCE}",
        case.artifact
    );
}

#[test]
fn tract_refuses_a_tree_ensemble_graph_at_load_instead_of_serving_it_wrong() {
    // `tract` 0.23.4 does not implement `ai.onnx.ml.TreeEnsembleRegressor`,
    // which is the single operator both `skl2onnx` and `onnxmltools` emit for a
    // boosted-tree model. So the "convert the tree family to ONNX and serve it
    // with the existing backend" route — the cheap answer for LightGBM and for
    // sklearn's GradientBoosting — does not exist today. It fails loudly at
    // load, naming the operator, which is the right time and the right message;
    // this test exists so the ceiling is a recorded fact rather than something
    // the next session rediscovers by hand.
    for artifact in ["sklearn_gbm.onnx", "lightgbm.onnx"] {
        let err = OnnxModel::from_path(refusal_fixture(artifact))
            .err()
            .unwrap_or_else(|| panic!("{artifact} loaded; tract grew a TreeEnsemble operator"));
        let message = err.to_string();
        assert!(
            message.contains("TreeEnsembleRegressor"),
            "{artifact}: the refusal must name the operator so the fix is obvious; got {message}"
        );
        // Note the variant: this arrives as `Parse`, not `Unsupported`, because
        // it is tract's translator refusing rather than one of this crate's own
        // checks. Recorded here so nobody reads it as a malformed file.
        assert!(
            matches!(err, LoadError::Parse(_)),
            "{artifact}: expected tract's own load failure, got {err}"
        );
    }
}

#[test]
fn the_sklearn_classifier_route_around_that_ceiling_is_refused_on_base_values() {
    // `TreeEnsembleClassifier` *is* in tract's registry, so converting a
    // boosted-tree model as a classifier is the obvious way round the test
    // above — it is how LightGBM's binary graph crosses. It does not work for
    // sklearn: tract wants `base_values` to carry one entry per class and
    // skl2onnx writes sklearn's single scalar intercept, so the graph fails to
    // build even though it satisfies every rule this crate imposes.
    //
    // The two obvious repairs are both traps, measured against onnxruntime with
    // fusion off: deleting `base_values` builds and agrees, but it is the model
    // *without its intercept*; padding it to two entries builds and then tract
    // and onnxruntime disagree by 0.18, five orders past the 1e-5 tolerance.
    // Neither is a re-export anyone should make without a parity bundle.
    let err = OnnxModel::from_path(refusal_fixture("sklearn_gbm_classifier.onnx"))
        .expect_err("sklearn's TreeEnsembleClassifier must not build");
    let message = err.to_string();
    assert!(
        message.contains("base_values"),
        "the refusal must name the attribute, not just the node — that is the whole \
         actionable content, and `anyhow`'s plain Display drops it. Got: {message}"
    );
}
