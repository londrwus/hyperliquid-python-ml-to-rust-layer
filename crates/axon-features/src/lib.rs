//! # axon-features
//!
//! Feature computation in Rust, held to **bit** equality with `axon.features`.
//!
//! `docs/03-ml-fidelity-and-features.md` splits the fidelity problem in two and
//! calls one half the easy one:
//!
//! ```text
//!    research signal  =  f_model( f_features( raw_data ) )
//!    live signal      =  g_model( g_features( raw_data ) )
//!
//!    quality preserved  ⟺  f_model ≈ g_model   AND   f_features ≈ g_features
//!                              (ADR-0021)             (this crate, ADR-0035)
//! ```
//!
//! [ADR-0021](../../../docs/adr/0021-rust-model-parity-gate.md) closed the left
//! side: a parity bundle carries a model, a holdout matrix and Python's own scores
//! over it, and `axon-model` proves this build reproduces them. It could not close
//! the right side, and said so — a model bundle proves that *identical feature
//! vectors* produce identical decisions, never that the two languages would compute
//! identical vectors from the same market data. Nothing in Rust computed a feature,
//! so there was nothing to compare.
//!
//! This crate is that missing side. It is the one item Phase 5 carried unchecked.
//!
//! ## The prime directive, and how a second implementation survives it
//!
//! `docs/03` says **never implement a feature twice**, and this crate is a second
//! implementation of seventeen of them. That is not a licence quietly taken; it is
//! the exception `docs/03` §2 names, and it comes with the condition attached: a
//! Rust implementation "must be validated *bit-equivalent* against `axon.features`
//! before it's allowed to serve." So the crate ships with the gate rather than
//! ahead of it — [`parity::FeatureBundle`] reads a frozen question written by
//! `axon.parity.feature_bundle` and answers it with no Python in the process.
//!
//! Bit-equivalence is achievable here and it was not free. See [`numeric`]: NumPy
//! does not sum a window left to right, and a Rust crate that did would disagree
//! with Python by up to eight ULP on every rolling column — enough to fail the gate
//! forever, and small enough that widening the tolerance to absorb it would look
//! like the reasonable fix.
//!
//! ## What is here
//!
//! - [`numeric`] — NumPy's reduction order, transcribed. The load-bearing file.
//! - [`functions`] — the transforms, one per registered name, over `&[f64]`.
//! - [`registry`] — name → transform, with the call shape pinned, mirroring
//!   `axon.features.registry`.
//! - [`spec`] — a [`FeatureSpec`] read from the canonical JSON Python writes, with
//!   its fingerprint **recomputed** rather than trusted, and evaluated in
//!   declaration order with column composition.
//! - [`streaming`] — the incremental runtime a live core would actually use: a
//!   bounded buffer per input, one feature row per event. It **refuses** a spec it
//!   cannot serve from a bounded buffer, which is what turns the house rule "finite
//!   lookback on every feature, no EMA" from a convention into a type error.
//! - [`parity`] — the cross-language gate.
//!
//! Synchronous, allocation-light, no tokio, no network, no clock — this crate is
//! part of the deterministic core.
//!
//! ## Float64, deliberately
//!
//! Features are `f64` and money is not, exactly as
//! [ADR-0016](../../../docs/adr/0016-feature-spec-and-parity-gates.md) §6 has it on
//! the Python side: prices cross the wire as fixed-point integers and order sizes
//! stay in `rust_decimal::Decimal`, but a z-score is a statistic. The fixed-point →
//! float conversion happens at the edge (`axon.features.inputs` and its Rust
//! counterpart in the runtime), never inside a transform, and never flows back
//! toward an order size. A timestamp is never a feature: float64 carries 53 mantissa
//! bits and a 2026 nanosecond stamp needs 61, so routing one through this matrix
//! would round events into ~256 ns buckets and reorder them.

#![deny(unsafe_code)]

pub mod functions;
pub mod numeric;
pub mod parity;
pub mod registry;
pub mod spec;
pub mod streaming;

pub use parity::{FeatureBundle, FeatureParityReport};
pub use registry::{feature_info, registered_features, FeatureInfo, Param, Params};
pub use spec::{FeatureDef, FeatureMatrix, FeatureSpec};
pub use streaming::FeatureStream;

/// The build of the transform library this crate implements.
///
/// Mirrors `axon.features.functions.FEATURES_VERSION`, and it is not decoration:
/// the value is folded into every spec fingerprint, so a spec written against a
/// different build fails [`FeatureSpec::from_json`]'s identity check rather than
/// being computed with transforms that have quietly changed meaning. The two
/// languages bumping this in lockstep is asserted by
/// `tests/cross_language.rs::the_two_libraries_bump_their_version_in_lockstep`,
/// because a version that only one side bumps
/// is worse than no version at all — it makes every fingerprint disagree and the
/// diagnosis point at the recipe instead of at the mismatch.
pub const FEATURES_VERSION: u32 = 1;

/// Anything that stops a feature matrix being computed, or being trusted.
///
/// One error type across the crate, mirroring `axon.features.registry.FeatureError`,
/// because every one of these is a startup-shaped failure a human reads: a spec that
/// does not describe what this build computes, an input that is the wrong shape, a
/// window that cannot be honoured. None of them happen per row on a healthy path.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FeatureError {
    /// A spec names a transform this build does not have.
    #[error("unknown feature {name:?}; this build registers {available:?}")]
    UnknownFeature {
        name: String,
        available: Vec<&'static str>,
    },

    /// The spec is malformed — a duplicate column, an unbound input, a parameter
    /// the transform does not take.
    #[error("invalid feature spec: {0}")]
    Spec(String),

    /// The serialized spec does not describe what this library would compute:
    /// either the payload was edited after its fingerprint was taken, or it was
    /// written by a different build of `axon.features`.
    ///
    /// Separate from [`FeatureError::Spec`] because the response differs. A
    /// malformed spec is a bug in whoever wrote it; a mismatch means the artifact's
    /// model was trained on features this process cannot reproduce, which is the
    /// training–serving skew the fingerprint exists to make impossible.
    #[error("feature spec mismatch: {0}")]
    Mismatch(String),

    /// The supplied inputs are the wrong shape — ragged, missing, or too large to
    /// hold exactly in float64.
    #[error("feature inputs: {0}")]
    Inputs(String),

    /// A parameter is out of range for its transform (a window below one, a `fast`
    /// span not shorter than `slow`).
    #[error("feature {feature:?}: {message}")]
    Param { feature: String, message: String },
}

/// Above 2^53 a float64 stops representing consecutive integers, so anything larger
/// arriving as a feature input has already lost its low bits.
///
/// Mirrors `axon.features.spec._EXACT_FLOAT_LIMIT`. In practice the check fires on
/// exactly one mistake — feeding a nanosecond `ts_event` in as a feature — which is
/// why timestamps travel beside the matrix and never inside it.
pub const EXACT_FLOAT_LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53
