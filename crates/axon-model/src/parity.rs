//! The Rust side of the model-parity gate: scoring a frozen Python question.
//!
//! `axon.parity.model` compares one set of Python predictions against another.
//! That is a real gate, and it is not the one Boundary A turns on. The claim
//! that decides whether inference may move into the Rust core (docs/02,
//! ADR-0019) is cross-language: **the model Rust will serve produces the same
//! decisions as the model Python researched with.** No Python-to-Python
//! comparison can fail on the way those two differ.
//!
//! A *parity bundle* is that question frozen into a directory, written by
//! `axon.parity.rust_gate` from a registry artifact (ADR-0015) and read here
//! with no Python, no ML libraries, no network and no clock:
//!
//! ```text
//! <bundle>/manifest.json     what this is, and what it must be held to
//!         /model.json        the artifact's own bytes, straight off the registry
//!         /features.f32      the holdout matrix, raw little-endian IEEE-754
//!         /predictions.f32   Python's own answer over those exact bytes
//!         /decisions.i8      Python's own discretized decision per row
//! ```
//!
//! Three properties of that layout are load-bearing:
//!
//! - **The matrices are raw f32, never JSON numbers.** A feature written as a
//!   decimal and re-parsed can land on the neighbouring float, which sends a row
//!   down the other side of a split; the gate would then be comparing two
//!   different questions and reporting the difference as a model defect. Raw
//!   little-endian bits cross the boundary unrounded, and [`f32::from_le_bytes`]
//!   is why the writer pins the byte order instead of using the host's.
//! - **`decisions.i8` is recorded, not derived.** Both sides discretize the
//!   score into `{-1 short, 0 flat, +1 long}`, and if the two languages disagree
//!   about the *rule* — `>=` against `>`, a threshold rounded differently — the
//!   numbers can match perfectly while the trades do not. [`ParityBundle::open`]
//!   re-derives Python's recorded decisions from Python's recorded scores and
//!   refuses the bundle if they disagree, so the rule is gated before the model
//!   is even loaded.
//! - **The criterion is declared and then checked against what the family
//!   allows.** A bundle cannot buy itself a looser gate: a manifest asking for
//!   1e-2 on an ONNX graph, or any tolerance at all on a tree, is refused
//!   ([`Criterion::allows`]).
//!
//! Decision invariance is an `and`, not advice (ADR-0016). Both criteria are
//! reported, and a candidate that is numerically well inside tolerance still
//! fails if one row lands on the other side of the trade threshold — that is the
//! failure the numeric check alone has waved through every time it has been
//! trusted on its own.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{InferenceError, LoadError, Model, OnnxModel, TreeModel};

/// Bundle layout this build understands. A bundle stamped higher is refused
/// rather than read with the unknown half treated as absent — the same rule
/// `ArtifactMeta.meta_schema` follows on the Python side.
pub const BUNDLE_SCHEMA: u32 = 1;

/// ADR-0003 §3's starting tolerance for graphs. Also the ceiling: a bundle may
/// declare something tighter, never looser.
pub const ONNX_EPS: f32 = 1e-5;

/// What a graph bundle written by this repo actually declares: **two ULP at 1.0**.
///
/// Mirrors `axon.parity.rust_gate.ONNX_TIGHT_EPS`, and the two must agree — the
/// writer stamps it into the manifest and the committed bundles are asserted
/// against it here, so a regeneration that quietly fell back to [`ONNX_EPS`]
/// reddens instead of passing with a hundredfold of slack restored.
///
/// It is not [`ONNX_EPS`] because every graph gated here sits four to five orders
/// of magnitude inside that ceiling — `lgbm_binary` 1.1920929e-7 (one ULP at 1.0,
/// exactly), `zoo_logistic` 8.940697e-8, `mlp_regressor` 0e0 — and the slack is
/// where a regression passes green. That is the same argument [`ONNX_EPS`]'s own
/// pinning of `tract` makes: a silent patch bump could move a result inside the
/// tolerance with no code change to blame it on.
pub const ONNX_TIGHT_EPS: f32 = 2.384_185_8e-7;

/// How many divergent rows a report keeps. A systematically broken candidate
/// diverges on every row and the first handful are enough to diagnose it;
/// keeping them all turns a failing test into a wall nobody reads. Matches
/// `axon.parity.model.DEFAULT_LIMIT`.
const DIVERGENCE_LIMIT: usize = 20;

const MANIFEST: &str = "manifest.json";

/// Why a bundle could not be read, or could not be asked of a model.
///
/// These are offline, startup-shaped failures, so they are verbose on purpose:
/// whoever reads one is deciding whether to regenerate the bundle or to go
/// looking for a bug in the serving path.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("reading parity bundle file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("parsing {path}: {message}")]
    Parse { path: PathBuf, message: String },

    /// The bundle is internally inconsistent — a length that contradicts a
    /// declared shape, a decision that does not follow from the score beside it.
    #[error("malformed parity bundle: {0}")]
    Malformed(String),

    /// The manifest asks to be held to a weaker standard than its model family
    /// allows. Refused, because a gate that can lower its own bar is not a gate.
    #[error("parity bundle weakens its own criterion: {0}")]
    Weakened(String),

    /// The model does not answer the question the bundle asks — a different
    /// version, a different feature width, more than one output.
    #[error("model does not match the parity bundle: {0}")]
    Mismatch(String),

    #[error("loading the bundle's artifact: {0}")]
    Model(#[from] LoadError),

    #[error("scoring the bundle's holdout: {0}")]
    Inference(#[from] InferenceError),
}

/// Which backend serves the bundle's artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// XGBoost's native JSON, served by [`TreeModel`].
    Xgboost,
    /// An ONNX graph, served by [`OnnxModel`].
    Onnx,
}

impl ModelKind {
    fn parse(raw: &str) -> Result<Self, BundleError> {
        match raw {
            "xgboost" => Ok(ModelKind::Xgboost),
            "onnx" => Ok(ModelKind::Onnx),
            // LightGBM artifacts exist in the registry and have no Rust backend
            // yet (ADR-0019's documented gap). Naming it beats "unknown kind".
            other => Err(BundleError::Malformed(format!(
                "artifact kind '{other}' has no Rust backend; this crate serves xgboost and onnx"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ModelKind::Xgboost => "xgboost",
            ModelKind::Onnx => "onnx",
        }
    }
}

/// The bar a bundle's candidate predictions are held to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criterion {
    /// Every bit. ADR-0019 claims `TreeModel` reproduces
    /// `Booster.predict(output_margin=True)` exactly, so the gate asserts
    /// exactly that: a tolerance on a tree would be the gate declining to test
    /// the claim it exists to test. Note this is *stronger* than the Python
    /// gate's `TREE_EPS = 0.0`, which is a numeric zero and so accepts `+0.0`
    /// against `-0.0`.
    BitExact,
    /// `max_abs_diff <= eps`. Two runtimes never agree bit for bit — ONNX does
    /// not encode operator ordering and float addition is not associative — so
    /// the graph family is gated at ADR-0003's 1e-5 (see [`ONNX_EPS`]).
    MaxAbsDiff(f32),
}

impl Criterion {
    /// What the family demands, regardless of what the manifest asked for.
    pub fn required_for(kind: ModelKind) -> Self {
        match kind {
            ModelKind::Xgboost => Criterion::BitExact,
            ModelKind::Onnx => Criterion::MaxAbsDiff(ONNX_EPS),
        }
    }

    /// Whether `declared` is at least as strict as `self`.
    ///
    /// Tightening is allowed — a model whose ONNX path happens to be exact may
    /// say so and be held to it. Loosening is the failure mode this exists for:
    /// a bundle regenerated after a red gate, with the tolerance nudged until it
    /// passed, would otherwise look identical to one that never failed.
    pub fn allows(self, declared: Criterion) -> bool {
        match (self, declared) {
            (Criterion::BitExact, Criterion::BitExact) => true,
            (Criterion::BitExact, Criterion::MaxAbsDiff(_)) => false,
            (Criterion::MaxAbsDiff(_), Criterion::BitExact) => true,
            (Criterion::MaxAbsDiff(required), Criterion::MaxAbsDiff(asked)) => asked <= required,
        }
    }

    /// Whether one row's reference and candidate agree under this criterion.
    /// Non-finite values are never "within": they are counted separately, since
    /// `nan <= eps` is false and would fail for the wrong stated reason.
    fn holds(self, reference: f32, candidate: f32) -> bool {
        if !(reference.is_finite() && candidate.is_finite()) {
            return false;
        }
        match self {
            Criterion::BitExact => reference.to_bits() == candidate.to_bits(),
            Criterion::MaxAbsDiff(eps) => (reference - candidate).abs() <= eps,
        }
    }
}

impl fmt::Display for Criterion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Criterion::BitExact => write!(f, "bit-exact"),
            Criterion::MaxAbsDiff(eps) => write!(f, "max_abs_diff <= {eps:e}"),
        }
    }
}

/// The strategy's discretized trading decision, as two thresholds on the score.
///
/// The same shape as `axon.parity.threshold_discretizer`, and deliberately the
/// same comparisons: `>= long_at` is long, `<= short_at` is short, anything
/// between is flat, and a NaN decides flat because every comparison against it
/// is false. That last case is why non-finite scores are failed by *count*
/// rather than left to the decision check — two NaNs agree on flat and would
/// otherwise pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    long_at: f32,
    short_at: f32,
}

impl Decision {
    /// Thresholds must be strictly ordered; an inverted pair would decide every
    /// row twice over. Spelled through `partial_cmp` rather than as a negated
    /// `<` so that a NaN threshold — which compares false against everything and
    /// would silently make every row flat — is refused rather than accepted.
    pub fn new(long_at: f32, short_at: f32) -> Result<Self, BundleError> {
        if short_at.partial_cmp(&long_at) != Some(std::cmp::Ordering::Less) {
            return Err(BundleError::Malformed(format!(
                "decision thresholds must satisfy short_at < long_at, got short_at={short_at} \
                 long_at={long_at}"
            )));
        }
        Ok(Self { long_at, short_at })
    }

    pub fn long_at(&self) -> f32 {
        self.long_at
    }

    pub fn short_at(&self) -> f32 {
        self.short_at
    }

    /// `-1` short, `0` flat, `+1` long.
    pub fn side(&self, score: f32) -> i8 {
        if score >= self.long_at {
            1
        } else if score <= self.short_at {
            -1
        } else {
            0
        }
    }
}

/// One row the gate has something to say about.
#[derive(Debug, Clone, Copy)]
pub struct Divergence {
    pub row: usize,
    /// Python's answer, from the bundle.
    pub reference: f32,
    /// What this build of Rust produced.
    pub candidate: f32,
    /// `NaN` when either side is non-finite: an unmeasurable difference is not
    /// a small one.
    pub abs_diff: f32,
    pub reference_side: i8,
    pub candidate_side: i8,
    /// Whether the row failed the numeric criterion.
    pub over_criterion: bool,
    /// Whether the discretized trading decision changed. This is the one that
    /// costs money.
    pub flipped: bool,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The bit patterns are printed next to the decimals because at one ULP
        // the two decimals routinely render identically, and a failure message
        // showing the same number twice reads as a lie.
        write!(
            f,
            "row {}: python {} ({:#010x}) -> rust {} ({:#010x})  delta {:e}  decision {:+} -> {:+}{}",
            self.row,
            self.reference,
            self.reference.to_bits(),
            self.candidate,
            self.candidate.to_bits(),
            self.abs_diff,
            self.reference_side,
            self.candidate_side,
            if self.flipped { "  FLIPPED" } else { "" },
        )
    }
}

/// The outcome of one cross-language parity run.
///
/// A report rather than a bool for the same reason `axon.parity` returns one:
/// the identical call has to serve a CI assertion and a live parity monitor
/// (docs/07), and "parity failed" at 03:00 tells an operator nothing about
/// whether to flatten the book.
#[derive(Debug, Clone)]
pub struct ParityReport {
    label: String,
    rows: usize,
    criterion: Criterion,
    max_abs_diff: f32,
    max_abs_diff_row: Option<usize>,
    non_finite: usize,
    over_criterion: usize,
    flips: usize,
    divergences: Vec<Divergence>,
}

impl ParityReport {
    /// Both criteria, as an `and`. A candidate inside tolerance everywhere can
    /// still move a position, so the decision check is not advice attached to a
    /// numeric verdict (ADR-0016 §3).
    pub fn passed(&self) -> bool {
        self.non_finite == 0 && self.over_criterion == 0 && self.flips == 0
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn max_abs_diff(&self) -> f32 {
        self.max_abs_diff
    }

    pub fn flips(&self) -> usize {
        self.flips
    }

    pub fn non_finite(&self) -> usize {
        self.non_finite
    }

    /// Rows that failed the numeric criterion. Exposed alongside [`Self::flips`]
    /// and [`Self::non_finite`] because `passed()` is an `and` over all three,
    /// and a caller that can only read two of them has to infer the third from
    /// a string — which is what `summary()` is for humans, not for assertions.
    pub fn over_criterion(&self) -> usize {
        self.over_criterion
    }

    /// Every divergent row the report kept, worst-first is *not* the order:
    /// they are in row order, because a systematic break is recognised by its
    /// shape and re-sorting hides it.
    pub fn divergences(&self) -> &[Divergence] {
        &self.divergences
    }

    pub fn summary(&self) -> String {
        let head = format!(
            "cross-language model parity {}: {} n={} criterion={} max_abs_diff={:e}{} \
             over_criterion={} flips={} non_finite={}",
            if self.passed() { "PASS" } else { "FAIL" },
            self.label,
            self.rows,
            self.criterion,
            self.max_abs_diff,
            match self.max_abs_diff_row {
                Some(row) => format!(" (row {row})"),
                None => String::new(),
            },
            self.over_criterion,
            self.flips,
            self.non_finite,
        );
        if self.passed() {
            return head;
        }
        let mut lines = vec![head];
        if self.non_finite > 0 {
            lines.push(format!(
                "  {} row(s) were not finite on one or both sides; a non-finite score is \
                 counted, never compared",
                self.non_finite
            ));
        }
        for divergence in &self.divergences {
            lines.push(format!("  {divergence}"));
        }
        let shown = self.divergences.len();
        let total = self.over_criterion.max(self.flips);
        if total > shown {
            lines.push(format!("  ... and more (showing the first {shown})"));
        }
        lines.join("\n")
    }

    /// Panic with [`ParityReport::summary`] unless the gate passed. The test
    /// form of `raise_for_status()`.
    pub fn assert_passed(&self) {
        assert!(self.passed(), "{}", self.summary());
    }
}

/// A frozen cross-language parity question, loaded from disk.
#[derive(Debug, Clone)]
pub struct ParityBundle {
    dir: PathBuf,
    manifest: Manifest,
    features: Vec<f32>,
    expected: Vec<f32>,
    decisions: Vec<i8>,
    criterion: Criterion,
    decision: Decision,
    kind: ModelKind,
}

impl ParityBundle {
    /// Read and validate a bundle directory.
    ///
    /// Everything checkable without a model is checked here, because a bundle
    /// that describes itself wrongly would otherwise surface as a parity
    /// failure — the one diagnosis that sends someone looking at the serving
    /// path instead of at the fixture.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, BundleError> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join(MANIFEST);
        let raw = read_file(&manifest_path)?;
        let manifest: Manifest = serde_json::from_slice(&raw).map_err(|e| BundleError::Parse {
            path: manifest_path.clone(),
            message: e.to_string(),
        })?;

        if manifest.bundle_schema > BUNDLE_SCHEMA {
            return Err(BundleError::Malformed(format!(
                "bundle is schema {}, this build understands {BUNDLE_SCHEMA}; refusing rather \
                 than reading unknown fields as absent",
                manifest.bundle_schema
            )));
        }
        if manifest.feature_spec_ref.is_empty() {
            // A model without the features that fed it is not reproducible, and
            // the harder half of parity is the features (docs/03).
            return Err(BundleError::Malformed(
                "bundle carries no feature_spec_ref; the corpus cannot be tied to a recipe"
                    .to_string(),
            ));
        }

        let kind = ModelKind::parse(&manifest.kind)?;
        if kind == ModelKind::Xgboost && manifest.score_space != "margin" {
            // TreeModel returns the raw margin and never applies the link
            // (ADR-0019). A bundle of probabilities would compare a probability
            // against a margin and report the *link* as a parity failure.
            return Err(BundleError::Malformed(format!(
                "xgboost bundle records '{}' scores; TreeModel serves the raw margin, so the \
                 reference must be Booster.predict(output_margin=True)",
                manifest.score_space
            )));
        }

        let rows = manifest.features.rows;
        let cols = manifest.features.cols;
        if rows == 0 || cols == 0 {
            return Err(BundleError::Malformed(
                "a parity gate over zero rows or zero features proves nothing".to_string(),
            ));
        }
        if manifest.predictions.rows != rows {
            return Err(BundleError::Malformed(format!(
                "{rows} feature rows but {} prediction rows; the two sides did not see the same \
                 corpus",
                manifest.predictions.rows
            )));
        }
        if manifest.predictions.cols != 1 {
            // Both Rust backends serve exactly one score (OnnxModel refuses a
            // multi-output graph outright), and a decision threshold is defined
            // on one number. A wider reference would have to pick a column here,
            // which is the guess ADR-0015's `score_output` exists to prevent.
            return Err(BundleError::Malformed(format!(
                "bundle records {} score columns; the gate discretizes one score per row",
                manifest.predictions.cols
            )));
        }

        let features = read_f32(&dir.join(&manifest.features.file), rows, cols)?;
        let expected = read_f32(&dir.join(&manifest.predictions.file), rows, 1)?;
        let decisions = read_i8(&dir.join(&manifest.decisions.file), rows)?;

        let declared = manifest.criterion.resolve();
        let required = Criterion::required_for(kind);
        if !required.allows(declared) {
            return Err(BundleError::Weakened(format!(
                "manifest asks for {declared} on an {} artifact, which is held to {required}",
                kind.as_str()
            )));
        }

        let decision = Decision::new(
            parse_bits(&manifest.decision.long_at_bits, "long_at_bits")?,
            parse_bits(&manifest.decision.short_at_bits, "short_at_bits")?,
        )?;

        let missing = features.iter().filter(|v| v.is_nan()).count();
        if missing != manifest.features.missing_cells {
            return Err(BundleError::Malformed(format!(
                "manifest claims {} missing feature cells, the matrix holds {missing}",
                manifest.features.missing_cells
            )));
        }

        // The rule, not just the numbers. If Rust discretizes Python's own
        // scores into different decisions, then every prediction could match to
        // the bit and the two systems would still trade differently — and that
        // disagreement would be invisible in a comparison of scores.
        for (row, (score, recorded)) in expected.iter().zip(decisions.iter()).enumerate() {
            let derived = decision.side(*score);
            if derived != *recorded {
                return Err(BundleError::Malformed(format!(
                    "row {row}: python recorded decision {recorded:+} for score {score} \
                     ({:#010x}), rust derives {derived:+} from the same score and the same \
                     thresholds — the two sides disagree about the decision rule, not the number",
                    score.to_bits()
                )));
            }
            if !score.is_finite() {
                return Err(BundleError::Malformed(format!(
                    "row {row}: the recorded reference is {score}; a non-finite reference makes \
                     every comparison against it vacuous"
                )));
            }
        }

        let counts = manifest.decisions.counts;
        let tally = |want: i8| decisions.iter().filter(|d| **d == want).count();
        if (tally(-1), tally(0), tally(1)) != (counts.short, counts.flat, counts.long) {
            return Err(BundleError::Malformed(format!(
                "manifest decision counts (short {}, flat {}, long {}) do not match the recorded \
                 decisions (short {}, flat {}, long {})",
                counts.short,
                counts.flat,
                counts.long,
                tally(-1),
                tally(0),
                tally(1),
            )));
        }

        Ok(Self {
            dir,
            manifest,
            features,
            expected,
            decisions,
            criterion: declared,
            decision,
            kind,
        })
    }

    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn model_version(&self) -> u32 {
        self.manifest.model_version
    }

    pub fn registry_id(&self) -> &str {
        &self.manifest.registry_id
    }

    /// `name/vN#fingerprint` — the recipe the artifact was trained on
    /// (ADR-0016). Carried so a bundle cannot be paired with a corpus computed
    /// by different transforms.
    pub fn feature_spec_ref(&self) -> &str {
        &self.manifest.feature_spec_ref
    }

    pub fn rows(&self) -> usize {
        self.manifest.features.rows
    }

    pub fn feature_width(&self) -> usize {
        self.manifest.features.cols
    }

    /// How many feature cells are missing (`NaN`). Zero means the tree
    /// backend's default-direction branch is not exercised by this corpus.
    pub fn missing_cells(&self) -> usize {
        self.manifest.features.missing_cells
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    pub fn decision(&self) -> Decision {
        self.decision
    }

    /// One holdout row, in feature-spec order.
    pub fn row(&self, i: usize) -> &[f32] {
        let width = self.feature_width();
        &self.features[i * width..(i + 1) * width]
    }

    /// Python's own scores, in row order.
    pub fn reference(&self) -> &[f32] {
        &self.expected
    }

    /// Python's own decisions, in row order.
    pub fn reference_decisions(&self) -> &[i8] {
        &self.decisions
    }

    /// Load the bundle's artifact through the backend its kind names.
    ///
    /// The bundle carries the model bytes rather than a path into a registry:
    /// a gate that reaches outside its own directory is a gate that passes on
    /// one machine.
    pub fn load_model(&self) -> Result<Box<dyn Model>, BundleError> {
        let path = self.dir.join(&self.manifest.artifact.file);
        Ok(match self.kind {
            ModelKind::Xgboost => Box::new(TreeModel::from_path(path)?),
            ModelKind::Onnx => Box::new(OnnxModel::from_path(path)?),
        })
    }

    /// Score every holdout row through `model`.
    ///
    /// One row at a time through [`Model::predict_into`] with a reused buffer,
    /// which is both the decision-path entry point (docs/05: the core does not
    /// allocate per event) and the batch shape the reference was taken at — a
    /// graph scored 128 rows at once can reassociate a reduction that a batch of
    /// one cannot.
    pub fn candidate(&self, model: &dyn Model) -> Result<Vec<f32>, BundleError> {
        if model.version() != self.model_version() {
            return Err(BundleError::Mismatch(format!(
                "bundle records model_version {}, the loaded artifact says {}; the reference was \
                 taken from different bytes",
                self.model_version(),
                model.version()
            )));
        }
        if model.input_len() != self.feature_width() {
            return Err(BundleError::Mismatch(format!(
                "bundle holds {}-wide feature rows, the model consumes {}",
                self.feature_width(),
                model.input_len()
            )));
        }
        if model.output_len() != 1 {
            return Err(BundleError::Mismatch(format!(
                "model produces {} outputs; the gate discretizes one score per row",
                model.output_len()
            )));
        }

        let mut out = [0.0f32; 1];
        let mut scores = Vec::with_capacity(self.rows());
        for i in 0..self.rows() {
            model.predict_into(self.row(i), &mut out)?;
            scores.push(out[0]);
        }
        Ok(scores)
    }

    /// Compare candidate scores against the frozen reference, under the
    /// bundle's own criterion and thresholds.
    pub fn compare(&self, candidate: &[f32]) -> Result<ParityReport, BundleError> {
        self.compare_with(candidate, self.criterion, self.decision)
    }

    /// [`ParityBundle::compare`] with the criterion and thresholds supplied.
    ///
    /// Exposed because a strategy's own thresholds are not the fixture's, and
    /// because it is how a test can put a decision boundary exactly where a
    /// sub-tolerance delta will cross it — which is the only way to see the
    /// decision half of the gate actually fail.
    pub fn compare_with(
        &self,
        candidate: &[f32],
        criterion: Criterion,
        decision: Decision,
    ) -> Result<ParityReport, BundleError> {
        if candidate.len() != self.rows() {
            return Err(BundleError::Mismatch(format!(
                "{} candidate scores against {} reference rows; a length mismatch means the two \
                 runs did not see the same inputs",
                candidate.len(),
                self.rows()
            )));
        }

        let mut report = ParityReport {
            label: format!(
                "{}@{} ({}, {})",
                self.registry_id(),
                self.model_version(),
                self.kind.as_str(),
                self.dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            rows: self.rows(),
            criterion,
            max_abs_diff: 0.0,
            max_abs_diff_row: None,
            non_finite: 0,
            over_criterion: 0,
            flips: 0,
            divergences: Vec::new(),
        };

        for (row, (reference, candidate)) in self.expected.iter().zip(candidate.iter()).enumerate()
        {
            let (reference, candidate) = (*reference, *candidate);
            let finite = reference.is_finite() && candidate.is_finite();
            let abs_diff = if finite {
                (reference - candidate).abs()
            } else {
                // Not zero: an unmeasurable difference must never be summarized
                // as a small one.
                f32::NAN
            };
            if finite && abs_diff > report.max_abs_diff {
                report.max_abs_diff = abs_diff;
                report.max_abs_diff_row = Some(row);
            }
            if !finite {
                report.non_finite += 1;
            }

            let over_criterion = !criterion.holds(reference, candidate);
            let reference_side = decision.side(reference);
            let candidate_side = decision.side(candidate);
            let flipped = reference_side != candidate_side;
            if over_criterion {
                report.over_criterion += 1;
            }
            if flipped {
                report.flips += 1;
            }
            if (over_criterion || flipped) && report.divergences.len() < DIVERGENCE_LIMIT {
                report.divergences.push(Divergence {
                    row,
                    reference,
                    candidate,
                    abs_diff,
                    reference_side,
                    candidate_side,
                    over_criterion,
                    flipped,
                });
            }
        }
        Ok(report)
    }

    /// Load the artifact, score the holdout, and compare — the whole gate.
    pub fn check(&self) -> Result<ParityReport, BundleError> {
        let model = self.load_model()?;
        let candidate = self.candidate(model.as_ref())?;
        self.compare(&candidate)
    }
}

// ── the manifest ──
//
// Unknown keys are tolerated and `bundle_schema` is the guard, so adding a
// descriptive field to the writer does not strand every committed bundle. A key
// that changes the *question* has to bump the schema.

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    bundle_schema: u32,
    registry_id: String,
    model_version: u32,
    kind: String,
    score_space: String,
    feature_spec_ref: String,
    artifact: FileEntry,
    features: MatrixEntry,
    predictions: MatrixEntry,
    decisions: DecisionsEntry,
    criterion: CriterionEntry,
    decision: DecisionEntry,
}

#[derive(Debug, Clone, Deserialize)]
struct FileEntry {
    file: String,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixEntry {
    file: String,
    rows: usize,
    #[serde(default = "one")]
    cols: usize,
    #[serde(default)]
    missing_cells: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct DecisionsEntry {
    file: String,
    counts: DecisionCounts,
}

#[derive(Debug, Clone, Copy, Deserialize)]
struct DecisionCounts {
    short: usize,
    flat: usize,
    long: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CriterionEntry {
    BitExact,
    MaxAbsDiff { eps: f32 },
}

impl CriterionEntry {
    fn resolve(self) -> Criterion {
        match self {
            CriterionEntry::BitExact => Criterion::BitExact,
            CriterionEntry::MaxAbsDiff { eps } => Criterion::MaxAbsDiff(eps),
        }
    }
}

/// Thresholds cross as IEEE-754 bit patterns. A decimal would be re-parsed on
/// this side and could land one ULP away, which moves the decision boundary
/// between the two languages — precisely the disagreement the gate is meant to
/// detect, injected by the gate itself. The `_approx` fields in the manifest are
/// for humans and are deliberately not read here.
#[derive(Debug, Clone, Deserialize)]
struct DecisionEntry {
    long_at_bits: String,
    short_at_bits: String,
}

fn one() -> usize {
    1
}

fn parse_bits(hex: &str, what: &str) -> Result<f32, BundleError> {
    let digits = hex.trim().trim_start_matches("0x");
    u32::from_str_radix(digits, 16)
        .map(f32::from_bits)
        .map_err(|e| BundleError::Malformed(format!("{what} '{hex}' is not a 32-bit pattern: {e}")))
}

fn read_file(path: &Path) -> Result<Vec<u8>, BundleError> {
    fs::read(path).map_err(|source| BundleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_f32(path: &Path, rows: usize, cols: usize) -> Result<Vec<f32>, BundleError> {
    let bytes = read_file(path)?;
    let want = rows * cols * 4;
    if bytes.len() != want {
        // A truncated matrix would otherwise be read as a shorter corpus with a
        // plausible shape, and the gate would pass on the rows that survived.
        return Err(BundleError::Malformed(format!(
            "{}: holds {} bytes; the manifest describes {rows}x{cols} f32 = {want} bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i8(path: &Path, rows: usize) -> Result<Vec<i8>, BundleError> {
    let bytes = read_file(path)?;
    if bytes.len() != rows {
        return Err(BundleError::Malformed(format!(
            "{}: holds {} decisions, the manifest describes {rows} rows",
            path.display(),
            bytes.len()
        )));
    }
    bytes
        .iter()
        .enumerate()
        .map(|(row, b)| match *b as i8 {
            side @ -1..=1 => Ok(side),
            other => Err(BundleError::Malformed(format!(
                "{}: row {row} records decision {other}; the alphabet is -1, 0, +1",
                path.display()
            ))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision() -> Decision {
        Decision::new(0.5, -0.5).unwrap()
    }

    #[test]
    fn inverted_thresholds_are_refused_rather_than_deciding_every_row_twice() {
        assert!(matches!(
            Decision::new(-0.5, 0.5),
            Err(BundleError::Malformed(_))
        ));
        assert!(matches!(
            Decision::new(0.5, 0.5),
            Err(BundleError::Malformed(_))
        ));
    }

    #[test]
    fn a_score_exactly_on_a_threshold_decides_the_same_way_numpy_does() {
        // `threshold_discretizer` uses `>=` and `<=`, so the boundary itself is
        // a position, not the flat band. Getting this backwards would flip every
        // row that lands exactly on a threshold — rare in a logit, routine in a
        // score rounded for display.
        let d = decision();
        assert_eq!(d.side(0.5), 1);
        assert_eq!(d.side(-0.5), -1);
        assert_eq!(d.side(0.0), 0);
        assert_eq!(d.side(f32::from_bits(0.5f32.to_bits() - 1)), 0);
    }

    #[test]
    fn a_nan_score_decides_flat_which_is_why_non_finite_is_counted_separately() {
        // Both sides NaN agree on "flat", so the decision check cannot see a
        // model that started producing NaN. The count is the only thing that
        // catches it.
        assert_eq!(decision().side(f32::NAN), 0);
        assert!(!Criterion::BitExact.holds(f32::NAN, f32::NAN));
        assert!(!Criterion::MaxAbsDiff(1e-5).holds(f32::NAN, f32::NAN));
    }

    #[test]
    fn a_tree_bundle_cannot_ask_for_a_tolerance_and_an_onnx_one_cannot_ask_for_more() {
        let trees = Criterion::required_for(ModelKind::Xgboost);
        assert!(trees.allows(Criterion::BitExact));
        assert!(!trees.allows(Criterion::MaxAbsDiff(1e-12)));

        let graphs = Criterion::required_for(ModelKind::Onnx);
        assert!(graphs.allows(Criterion::MaxAbsDiff(ONNX_EPS)));
        assert!(graphs.allows(Criterion::MaxAbsDiff(1e-7)));
        assert!(graphs.allows(Criterion::BitExact));
        assert!(!graphs.allows(Criterion::MaxAbsDiff(1e-4)));
    }

    #[test]
    fn bit_exactness_refuses_the_signed_zero_the_python_gate_accepts() {
        // TREE_EPS = 0.0 compares numerically, and `0.0 - -0.0` is 0.0. ADR-0019
        // claims the bits, so the Rust side asserts the bits.
        assert!(!Criterion::BitExact.holds(0.0, -0.0));
        assert!(Criterion::MaxAbsDiff(0.0).holds(0.0, -0.0));
    }
}
