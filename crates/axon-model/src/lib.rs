//! # axon-model
//!
//! The Boundary-A native inference path (`docs/02-python-rust-boundary.md`,
//! ADR-0003, ADR-0019): once a strategy's Rust feature+inference path is proven
//! equivalent to `axon.features` / the Python model, inference moves here and
//! Python leaves that strategy's live loop.
//!
//! Two backends ship, one per model family, matching ADR-0003 §2:
//!
//! - [`TreeModel`] — XGBoost's own JSON artifact, evaluated by threshold
//!   traversal. Trees are the family ADR-0003 calls "~numerically exact", so
//!   this backend is held to **bit equality** with `Booster.predict`.
//! - [`OnnxModel`] — ONNX via `tract`, pure Rust with no C++ toolchain and the
//!   most deterministic of the options in the glossary. Neural nets cannot be
//!   bit-exact across runtimes (ONNX does not encode op ordering and FP addition
//!   is not associative), so this backend is held to 1e-5.
//!
//! **FP32 end-to-end, no quantization** (ADR-0005). Both loaders *refuse*
//! reduced-precision artifacts rather than serving them, because an FP16
//! downcast does not show up as an error — it shows up as a prediction that
//! crossed a decision threshold it should not have.
//!
//! [`parity`] is the gate that lets any of that be believed: a *parity bundle*
//! written by `axon.parity.rust_gate` carries a registry artifact, a holdout
//! matrix and Python's own answers over it, and [`parity::ParityBundle::check`]
//! asserts this crate reproduces them — bit for bit for trees, inside 1e-5 for
//! graphs, and with no discretized trading decision flipped either way
//! (ADR-0021). A backend nobody has held to Python's numbers is a backend whose
//! claim is a comment.
//!
//! Everything here is synchronous and allocation-light; this crate is part of
//! the deterministic core and never touches tokio.

#![deny(unsafe_code)]

mod onnx;
pub mod parity;
mod tree;

pub use onnx::OnnxModel;
pub use tree::{TreeLink, TreeModel};

use std::io;
use std::path::PathBuf;

/// A versioned FP32 model.
///
/// [`Model::version`] is recorded on every emitted `Signal`: a decision is only
/// reproducible offline if the artifact that produced it can be named, so a
/// backend that cannot read a version out of its artifact refuses to load
/// rather than serving an anonymous model.
///
/// [`Model::input_len`] and [`Model::output_len`] exist so a wiring mistake —
/// a feature spec that no longer matches the graph — is caught once at startup
/// instead of on the first tick.
pub trait Model: Send + Sync {
    /// The artifact version, read from the artifact itself.
    fn version(&self) -> u32;

    /// Number of features this model consumes, in feature-spec order.
    fn input_len(&self) -> usize;

    /// Number of values a prediction produces.
    fn output_len(&self) -> usize;

    /// Run inference over an FP32 feature vector, writing [`Model::output_len`]
    /// values into `out`.
    ///
    /// This, not [`Model::predict`], is the decision-path entry point: the core
    /// does not allocate per event (docs/05), so the caller owns the output
    /// buffer and reuses it tick after tick.
    fn predict_into(&self, features: &[f32], out: &mut [f32]) -> Result<(), InferenceError>;

    /// Allocating convenience wrapper over [`Model::predict_into`], for tests,
    /// tooling and startup checks.
    fn predict(&self, features: &[f32]) -> Result<Vec<f32>, InferenceError> {
        let mut out = vec![0.0f32; self.output_len()];
        self.predict_into(features, &mut out)?;
        Ok(out)
    }
}

/// Why an artifact could not be turned into a servable model.
///
/// Loading is a startup-time operation, so these are deliberately verbose: the
/// operator reading this message is deciding whether to re-export the model.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("reading model artifact {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("parsing model artifact: {0}")]
    Parse(String),

    /// The artifact carries no version, so any signal it produced would be
    /// impossible to tie back to a specific model. See [`Model::version`].
    #[error(
        "model artifact carries no version ({slot}); an unversioned model cannot be \
         reproduced offline, so it is refused"
    )]
    MissingVersion { slot: &'static str },

    /// ADR-0005: FP32 end-to-end. A reduced-precision tensor anywhere in the
    /// graph can silently move a prediction across a decision threshold.
    #[error("artifact is not FP32: {what} uses {dtype}; FP32 end-to-end is required (ADR-0005)")]
    ReducedPrecision { what: String, dtype: &'static str },

    /// The artifact is well-formed but uses a construct this backend will not
    /// serve, because serving it would silently disagree with Python.
    #[error("unsupported model artifact: {0}")]
    Unsupported(String),

    /// The artifact is internally inconsistent (bad indices, mismatched array
    /// lengths). Caught at load so the hot path cannot panic or spin.
    #[error("malformed model artifact: {0}")]
    Malformed(String),
}

/// Why a prediction could not be produced.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("model expects {expected} features, got {got}")]
    FeatureCount { expected: usize, got: usize },

    #[error("model produces {expected} outputs, the buffer holds {got}")]
    OutputLen { expected: usize, got: usize },

    #[error("inference backend failed: {0}")]
    Backend(String),
}

/// Shared precondition for every backend's `predict_into`.
fn check_shapes(
    features: &[f32],
    out: &[f32],
    input_len: usize,
    output_len: usize,
) -> Result<(), InferenceError> {
    if features.len() != input_len {
        return Err(InferenceError::FeatureCount {
            expected: input_len,
            got: features.len(),
        });
    }
    if out.len() != output_len {
        return Err(InferenceError::OutputLen {
            expected: output_len,
            got: out.len(),
        });
    }
    Ok(())
}

/// A reference linear model (`wᵀx + b`). Exercises the trait and is a useful
/// parity baseline: it is the one model whose Rust and Python forms are
/// trivially inspectable when a parity failure needs bisecting.
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub version: u32,
}

impl Model for LinearModel {
    fn version(&self) -> u32 {
        self.version
    }

    fn input_len(&self) -> usize {
        self.weights.len()
    }

    fn output_len(&self) -> usize {
        1
    }

    fn predict_into(&self, features: &[f32], out: &mut [f32]) -> Result<(), InferenceError> {
        // Zipping weights against features would silently truncate to the
        // shorter of the two, so a feature spec that gained or lost a column
        // would keep producing plausible numbers from the wrong inputs.
        check_shapes(features, out, self.weights.len(), 1)?;
        let dot: f32 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum();
        out[0] = dot + self.bias;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_model_predicts() {
        let m = LinearModel {
            weights: vec![2.0, -1.0, 0.5],
            bias: 1.0,
            version: 7,
        };
        assert_eq!(m.version(), 7);
        // 2*1 + (-1)*2 + 0.5*4 + 1 = 2 - 2 + 2 + 1 = 3
        assert_eq!(m.predict(&[1.0, 2.0, 4.0]).unwrap(), vec![3.0]);
    }

    #[test]
    fn linear_model_rejects_a_feature_vector_of_the_wrong_length() {
        let m = LinearModel {
            weights: vec![2.0, -1.0, 0.5],
            bias: 1.0,
            version: 7,
        };
        assert!(matches!(
            m.predict(&[1.0, 2.0]),
            Err(InferenceError::FeatureCount {
                expected: 3,
                got: 2
            })
        ));
        assert!(m.predict(&[1.0, 2.0, 4.0, 8.0]).is_err());
    }

    #[test]
    fn predict_into_refuses_a_buffer_that_is_not_output_len() {
        // The decision path reuses one buffer for the life of the process. If a
        // model swap changed the output width, a silent partial write would
        // leave stale values from the previous model in the tail.
        let m = LinearModel {
            weights: vec![1.0],
            bias: 0.0,
            version: 1,
        };
        let mut two = [0.0f32; 2];
        assert!(matches!(
            m.predict_into(&[1.0], &mut two),
            Err(InferenceError::OutputLen {
                expected: 1,
                got: 2
            })
        ));
        let mut one = [0.0f32; 1];
        m.predict_into(&[3.0], &mut one).unwrap();
        assert_eq!(one, [3.0]);
    }
}
