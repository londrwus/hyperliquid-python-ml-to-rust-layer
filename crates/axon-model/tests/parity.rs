//! The model-parity gate, seen from the Rust side (docs/03 part 1, ADR-0003 §3).
//!
//! Python trained the models, exported the artifacts, and recorded its own
//! answers; `tests/fixtures/generate.py` did all of that once and the results
//! are committed. These tests assert that the Rust backends reproduce those
//! frozen answers — **bit for bit** for the tree ensembles, within 1e-5 for the
//! ONNX graph, which is the split ADR-0003 draws between "~numerically exact"
//! and "FP32-equivalent".
//!
//! Everything here is offline and deterministic: no network, no Python, no
//! retraining. The reference is frozen precisely so that a failure means Rust
//! moved, not that the fixture moved with it.

use std::path::PathBuf;

use axon_model::{LoadError, Model, OnnxModel, TreeModel};
use serde::Deserialize;

/// ADR-0003 §3: neural nets are gated at max_abs_diff < 1e-5, not at equality.
const ONNX_TOLERANCE: f32 = 1e-5;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[derive(Deserialize)]
struct Reference {
    trees: Vec<TreeCase>,
    onnx: Vec<OnnxCase>,
}

#[derive(Deserialize)]
struct TreeCase {
    artifact: String,
    objective: String,
    version: u32,
    num_feature: usize,
    /// A `null` is a missing feature — JSON has no NaN literal, and writing one
    /// anyway is what makes a fixture unparseable by strict readers.
    inputs: Vec<Vec<Option<f32>>>,
    /// Expected margins as IEEE-754 bit patterns. A decimal here would let the
    /// fixture absorb the one-ULP error the gate exists to catch.
    expected_bits: Vec<String>,
}

#[derive(Deserialize)]
struct OnnxCase {
    artifact: String,
    version: u32,
    num_feature: usize,
    inputs: Vec<Vec<f32>>,
    expected: Vec<Vec<f32>>,
}

fn reference() -> Reference {
    let raw = std::fs::read(fixture("reference.json")).expect("reference fixture");
    serde_json::from_slice(&raw).expect("reference fixture parses")
}

fn row(values: &[Option<f32>]) -> Vec<f32> {
    values.iter().map(|v| v.unwrap_or(f32::NAN)).collect()
}

fn bits(hex: &str) -> u32 {
    u32::from_str_radix(hex.trim_start_matches("0x"), 16).expect("hex bit pattern")
}

#[test]
fn tree_margins_are_bit_identical_to_xgboost() {
    let reference = reference();
    assert!(!reference.trees.is_empty(), "reference has no tree cases");

    for case in &reference.trees {
        let model = TreeModel::from_path(fixture(&case.artifact)).expect(&case.artifact);
        assert_eq!(model.version(), case.version, "{}", case.artifact);
        assert_eq!(model.input_len(), case.num_feature, "{}", case.artifact);
        assert_eq!(model.objective(), case.objective, "{}", case.artifact);
        assert_eq!(
            case.inputs.len(),
            case.expected_bits.len(),
            "{} has a ragged reference",
            case.artifact
        );

        for (i, input) in case.inputs.iter().enumerate() {
            let got = model.predict(&row(input)).expect("predict");
            assert_eq!(got.len(), 1);
            let expected = bits(&case.expected_bits[i]);
            assert_eq!(
                got[0].to_bits(),
                expected,
                "{} row {i}: got {} ({:#010x}), xgboost said {} ({expected:#010x})",
                case.artifact,
                got[0],
                got[0].to_bits(),
                f32::from_bits(expected),
            );
        }
    }
}

#[test]
fn tree_fixtures_actually_contain_missing_values() {
    // Without this the bit-exactness test above could pass while the
    // default-direction branch went entirely unexercised — the corpus, not the
    // assertion, is what makes that branch part of the gate.
    let reference = reference();
    for case in &reference.trees {
        let missing = case.inputs.iter().flatten().filter(|v| v.is_none()).count();
        assert!(
            missing > 0,
            "{} holdout has no missing features",
            case.artifact
        );
    }
}

#[test]
fn onnx_outputs_match_onnxruntime_within_tolerance() {
    let reference = reference();
    assert!(!reference.onnx.is_empty(), "reference has no onnx cases");

    for case in &reference.onnx {
        let model = OnnxModel::from_path(fixture(&case.artifact)).expect(&case.artifact);
        assert_eq!(model.version(), case.version, "{}", case.artifact);
        assert_eq!(model.input_len(), case.num_feature, "{}", case.artifact);
        assert_eq!(
            model.output_len(),
            case.expected[0].len(),
            "{}",
            case.artifact
        );

        let mut worst = 0.0f32;
        for (i, input) in case.inputs.iter().enumerate() {
            let got = model.predict(input).expect("predict");
            let want = &case.expected[i];
            assert_eq!(got.len(), want.len(), "{} row {i}", case.artifact);
            for (g, w) in got.iter().zip(want.iter()) {
                worst = worst.max((g - w).abs());
            }
        }
        assert!(
            worst < ONNX_TOLERANCE,
            "{}: max_abs_diff {worst} exceeds {ONNX_TOLERANCE}",
            case.artifact
        );
    }
}

#[test]
fn onnx_inference_is_reproducible_across_calls() {
    // Determinism is the reason tract was chosen over a thread-pooled runtime
    // (ADR-0019). A run-to-run difference here would mean a replay of a live
    // session could not reproduce the decision it is replaying.
    let case = &reference().onnx[0];
    let model = OnnxModel::from_path(fixture(&case.artifact)).unwrap();
    let input = &case.inputs[0];
    let first = model.predict(input).unwrap();
    for _ in 0..16 {
        let again = model.predict(input).unwrap();
        let same = first
            .iter()
            .zip(again.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "tract produced different bits for the same input");
    }
}

#[test]
fn fp16_graph_is_refused_at_its_boundary() {
    let err = OnnxModel::from_path(fixture("fp16_boundary.onnx")).unwrap_err();
    assert!(
        matches!(err, LoadError::ReducedPrecision { .. }),
        "expected an FP32 refusal, got {err}"
    );
}

#[test]
fn fp16_hidden_behind_an_fp32_signature_is_still_refused() {
    // The dangerous shape: FP32 in, FP32 out, half precision in the middle.
    // Checking only the graph's inputs and outputs would load this happily and
    // serve predictions rounded to 10 mantissa bits (ADR-0005).
    let err = OnnxModel::from_path(fixture("fp16_hidden_cast.onnx")).unwrap_err();
    assert!(
        matches!(err, LoadError::ReducedPrecision { .. }),
        "expected an FP32 refusal, got {err}"
    );
}

#[test]
fn onnx_artifact_without_a_model_version_is_refused() {
    // A valid FP32 graph that nobody stamped. Serving it would put a signal on
    // the bus that no artifact can be matched back to, which is the one thing
    // the version field exists to prevent.
    let err = OnnxModel::from_path(fixture("unversioned.onnx")).unwrap_err();
    assert!(
        matches!(err, LoadError::MissingVersion { .. }),
        "expected a version refusal, got {err}"
    );
}

#[test]
fn onnx_rejects_a_feature_vector_of_the_wrong_length() {
    let model = OnnxModel::from_path(fixture("mlp_regressor.onnx")).unwrap();
    let short = vec![0.0f32; model.input_len() - 1];
    assert!(model.predict(&short).is_err());
}
