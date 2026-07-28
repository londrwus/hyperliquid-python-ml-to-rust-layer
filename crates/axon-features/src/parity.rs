//! The cross-language **feature** gate: a frozen Python question about vectors,
//! recomputed here.
//!
//! [ADR-0021](../../../docs/adr/0021-rust-model-parity-gate.md) built this module's
//! sibling one level up and named its own limit in the same breath. A *model* parity
//! bundle carries a matrix, an artifact and Python's own scores, and it proves that
//! **identical feature vectors produce identical decisions**. It hands both sides the
//! same matrix, so it can never prove the two languages would compute identical vectors
//! from the same market data — and `docs/03` is explicit that the second half is the
//! hard one, the one where quality actually leaks. A rolling window that ends one
//! sample early, a standard deviation carrying the sample correction on one side, a NaN
//! warmup back-filled with a seed, a reduction reassociated by a different loop: every
//! one of those leaves the model gate green.
//!
//! A *feature parity bundle* is that harder question, written by
//! `axon.parity.feature_bundle` out of a [`FeatureSpec`] and a set of input arrays, and
//! answered here with no Python, no numpy, no network and no clock:
//!
//! ```text
//! <bundle>/manifest.json   what this is, and what it must be held to
//!         /spec.json       the canonical FeatureSpec JSON, re-identified on load
//!         /inputs.f64      the input arrays, raw little-endian IEEE-754, ROW-MAJOR
//!         /features.f64    Python's own matrix over exactly those bytes, same encoding
//! ```
//!
//! Four properties of that layout are load-bearing, and three of them are the model
//! bundle's arguments repeated at f64 width because they were right the first time.
//!
//! - **The matrices are raw little-endian f64, never JSON numbers.** A value written as
//!   a decimal and re-parsed can land on the neighbouring float, and this gate is built
//!   to see exactly one ULP; a serialization that can move one would be the gate
//!   manufacturing its own failures. Row-major, because both sides write and read a
//!   flat buffer — a column-major write would hand this one a transposed matrix, which
//!   for a square corpus fails *quietly*.
//! - **f64, not the model bundle's f32.** Features are float64 (ADR-0016 §6): a z-score
//!   is a statistic, not money. Reusing the model bundle's width would round every
//!   reference value before the comparison, and two implementations differing in the
//!   last three bits of a double would then agree perfectly — which is precisely the
//!   divergence this exists to catch.
//! - **NaN travels as its own bits**, which a JSON `null` cannot do. Warmup is NaN by
//!   construction on both sides (ADR-0016 §1 — zero is a legal value for every feature
//!   in this library, so a zero warmup is indistinguishable from a reading), so the NaN
//!   cells *are* the first rows, and the gate's whole warmup claim depends on them
//!   surviving the trip.
//! - **`inputs.names` is the spec's own `required_inputs`, sorted**, and this side
//!   rebuilds its input mapping from that list positionally. A free-form list would let
//!   the two sides pair arrays with the wrong names, and every column would then be
//!   wrong in a way that reads exactly like a transform bug: a plausible number, off by
//!   a shape nobody can trace back to a swapped `high` and `low`.
//!
//! ## Bit equality is the only criterion a feature bundle may declare
//!
//! The model gate has a tolerance arm because ONNX does not encode operator ordering
//! and float addition is not associative, so two graph runtimes never agree to the bit.
//! Nothing of the sort is true here, and the claim is about the library rather than
//! optimism: every transform in `axon.features` is built from `+ - * /`, `sqrt`,
//! comparison and `log`. IEEE-754 *requires* the first five to be correctly rounded, so
//! they agree on any conforming machine by definition. `log` is not in that list —
//! neither NumPy nor libm promises correct rounding — and its agreement is a measurement
//! rather than a guarantee: **0 ULP** over the 32 ratios `tests/fixtures` pins and over
//! a 200 000-sample, 26-decade sweep that `scripts/modal_libm_probe.py` re-runs on
//! demand — on glibc 2.39 here, glibc 2.36 and musl in a container (see
//! [`crate::functions`]).
//!
//! So [`Criterion::allows`] has the same shape it has in `axon_model::parity` and one
//! fewer thing to permit. A manifest asking for a tolerance is refused
//! ([`BundleError::Weakened`]) even when the tolerance is absurdly tight, for the
//! procedural reason ADR-0021 §5 gives: a bundle regenerated after a red gate, with the
//! bar nudged until it passed, is otherwise indistinguishable in the tree from one that
//! never failed. A gate that can lower its own bar is not a gate.
//!
//! ## The NaN rule, which is not a tolerance
//!
//! Bit equality is the wrong comparison for a NaN cell and the right one for everything
//! else, so the two are split exactly as ADR-0016 §4 splits them:
//!
//! - **NaN on both sides is a match.** Warmup is legitimately NaN in both paths, and a
//!   strict bit comparison of every cell would fail every run — hardest on a bundle
//!   that is perfectly correct.
//! - **NaN on one side only is a mismatch.** A feature that goes NaN on one side and
//!   finite on the other is precisely the staleness bug this gate exists to catch. It
//!   is counted separately from a numeric mismatch because the diagnosis differs: one
//!   is a window that did not fill, the other is arithmetic that drifted.
//! - **Two finite values must be bit-equal**, `to_bits()` against `to_bits()`, which
//!   also refuses `+0.0` against `-0.0` — a sign that survives into a z-score is a sign
//!   the two implementations genuinely disagree about.
//!
//! What this deliberately does **not** do is compare NaN *payloads*. A quiet NaN's
//! payload is not part of what either language promises: `axon.features` masks every
//! quotient, so its NaNs are the literal `np.nan` (`0x7ff8000000000000`), while x86's
//! own invalid-operation result is `0xfff8000000000000` — the same value, the other
//! sign bit. A Rust transform that divided first and masked afterwards would be
//! arithmetically perfect and would redden a payload comparison on every guarded cell.
//! The Python writer refuses a non-canonical NaN at write time, which is what keeps its
//! own reader's stricter view honest; this reader does not restate that check, because
//! a comparison blind to the payload cannot be broken by one.
//!
//! Note the masking claim holds for finite inputs and not universally: an `inf` reaching
//! `realized_volatility`, `close_location` or `trade_flow_imbalance` makes NumPy's own
//! `_var` compute `inf - inf` before any mask, so **Python** produces the negative quiet
//! NaN too. The writer refuses such a corpus, which is right — but the fault is the
//! infinity in the feed, not the NaN spelling, and the message says the latter.
//!
//! Under `np.allclose` defaults the first two cases above are indistinguishable, which
//! is the whole reason neither language uses it.
//!
//! ## `libm_columns` is a signpost, not a tolerance
//!
//! The manifest names the columns whose value passes through `log`. Nothing widens for
//! them. They exist so that a gate reddening on a platform this repo has not measured
//! has a first question: did it redden *only* there? If it did, the argument is about
//! one libm and not about the runtime. [`FeatureParityReport::summary`] answers that
//! question in the failure text rather than leaving it to be asked, and answers it in
//! both directions — "these columns do not touch `log`" is the more useful half.
//!
//! **Both languages derive the list and the two are compared**, rather than Rust
//! reading Python's answer. Each carries a transform-level flag — `LIBM_FEATURES` in
//! `axon.parity.feature_bundle`, [`axon_features::registry::FeatureInfo::reaches_libm`]
//! here — and each walks the spec's own binding graph, because the dependency is
//! inherited: a z-score *of* a log return never calls `log` and is nonetheless a
//! function of one. A field that only one side could derive would be a field the other
//! side believes, and the two tables silently drifting apart is precisely the failure
//! the signpost would then hide — since its entire job is to be trusted on a platform
//! nobody here has measured. Two copies that agree because they are compared is a
//! different thing from two copies that agree because nobody has looked.
//!
//! ## The report names the column and the row
//!
//! "The feature matrices differ" is not a debuggable statement about nine hundred rows
//! of six columns. The worst column is named **by cell count first and magnitude
//! second** (ADR-0016 §4), because a unit error breaks one column on every row and that
//! is a different diagnosis from one column being slightly worse on one row. Every
//! divergent cell reports both values as decimals *and* as bit patterns, because at one
//! ULP the two decimals print identically and a failure message showing the same number
//! twice reads as a lie.
//!
//! ## What is checked at load, and what is deliberately not
//!
//! Everything checkable without recomputing the matrix is checked in
//! [`FeatureBundle::open`]: the schema ceiling, the criterion, the spec's **recomputed**
//! fingerprint against `spec_ref`, the canonical serialization byte for byte, the input
//! and column names against the spec's own, the declared shapes against the actual byte
//! lengths, the NaN counts on both matrices, and `finite_rows`. A bundle that describes
//! itself wrongly must not surface as a parity failure: that is the one diagnosis that
//! sends somebody looking at the feature path instead of at the fixture.
//!
//! The SHA-256s the manifest records are verified by the Python reader and **not here**,
//! for the reason ADR-0021 §7 gives one level up: a flipped bit in either matrix changes
//! a value, so the gate itself catches it, and hashing would buy a better error message
//! and nothing else. `sha2` is already a dependency of this crate, so this is a choice
//! rather than a constraint — and the distinction is worth keeping straight. The spec
//! fingerprint is a hash of a *recipe*, which nothing else in the bundle can re-derive
//! and which is therefore recomputed rather than trusted; a matrix hash is a hash of the
//! answer, and the comparison below already checks that answer cell by cell.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::spec::{FeatureMatrix, FeatureSpec};
use crate::FeatureError;

/// Bundle layout this build understands. A bundle stamped higher is refused rather than
/// read with the unknown half treated as absent — the same rule
/// `axon_model::parity::BUNDLE_SCHEMA` and `ArtifactMeta.meta_schema` follow.
///
/// Deliberately a *separate* counter from the model bundle's, mirroring
/// `axon.parity.feature_bundle.BUNDLE_SCHEMA`: the two formats carry different files and
/// move for different reasons, and sharing a number would mean every model-side layout
/// change invalidated every feature bundle.
pub const BUNDLE_SCHEMA: u32 = 1;

/// How many divergent cells a report keeps. A systematically broken column diverges on
/// every row, and the first handful are enough to diagnose it; keeping them all turns a
/// failing test into a wall nobody reads. The *counts* are never truncated and the
/// per-column tally is taken over every cell, so the verdict never depends on this
/// number.
const DIVERGENCE_LIMIT: usize = 20;

const MANIFEST: &str = "manifest.json";

/// Why a bundle could not be read, or could not be asked of this build.
///
/// The same arms as `axon_model::parity::BundleError`, for the same reason: these are
/// offline, startup-shaped failures, and whoever reads one is deciding whether to
/// regenerate the bundle or to go looking for a bug in the feature path. Keeping "the
/// fixture is wrong" ([`Malformed`](Self::Malformed)) apart from "the two languages
/// disagree" (a [`FeatureParityReport`] that did not pass) is the whole point of the
/// split — a malformed bundle reported as a parity failure sends somebody reading
/// `numeric.rs` for a defect that is in a manifest.
#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("reading feature parity bundle file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("parsing {path}: {message}")]
    Parse { path: PathBuf, message: String },

    /// The bundle is internally inconsistent — a length that contradicts a declared
    /// shape, a column name the spec does not produce, a NaN count that does not match
    /// the matrix beside it.
    #[error("malformed feature parity bundle: {0}")]
    Malformed(String),

    /// The manifest asks to be held to a weaker standard than a feature bundle allows.
    /// Refused, because a gate that can lower its own bar is not a gate.
    #[error("feature parity bundle weakens its own criterion: {0}")]
    Weakened(String),

    /// A candidate does not answer the question the bundle asks — a different row count,
    /// a different width, differently named columns.
    #[error("candidate does not match the feature parity bundle: {0}")]
    Mismatch(String),

    /// The bundle's own spec could not be computed over the bundle's own inputs. This is
    /// a failure of *this build*, not a verdict on the fixture: Python computed the same
    /// spec over the same bytes in order to write the reference in the first place.
    #[error("computing the bundle's feature matrix: {0}")]
    Feature(#[from] FeatureError),
}

// ── the criterion ─────────────────────────────────────────────────────────────

/// The bar a bundle's candidate matrix is held to.
///
/// [`Criterion::MaxAbsDiff`] exists so that a manifest asking for one can be *named* in
/// the refusal rather than failing to parse. It is unreachable through
/// [`FeatureBundle::open`], which is the point: the arm is here to be refused, and a
/// reader that could not represent the thing it refuses would have to report a
/// deliberately loosened bundle as a syntax error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criterion {
    /// Every bit, with the NaN rule in the module docstring. The only criterion a
    /// feature bundle may declare.
    BitExact,
    /// `max_abs_diff <= eps`. Legal in a *model* bundle for the graph family; never
    /// legal here.
    MaxAbsDiff(f64),
}

impl Criterion {
    /// What a feature bundle is held to, regardless of what its manifest asked for.
    ///
    /// A function rather than a bare constant so it reads like
    /// `axon_model::parity::Criterion::required_for`, which takes the model family as an
    /// argument because trees and graphs differ. Here there is one family and one
    /// answer, and spelling it the same way keeps the two gates comparable at a glance.
    pub const fn required() -> Self {
        Criterion::BitExact
    }

    /// Whether `declared` is at least as strict as `self`.
    ///
    /// Tightening is allowed, and there is nothing here to tighten *from* — which makes
    /// this look redundant until the failure it prevents is named: a bundle regenerated
    /// after a red gate, with a tolerance added until it passed, is otherwise
    /// indistinguishable in the tree from one that never failed. The check is
    /// procedural, not numerical.
    pub fn allows(self, declared: Criterion) -> bool {
        match (self, declared) {
            (Criterion::BitExact, Criterion::BitExact) => true,
            (Criterion::BitExact, Criterion::MaxAbsDiff(_)) => false,
            (Criterion::MaxAbsDiff(_), Criterion::BitExact) => true,
            (Criterion::MaxAbsDiff(required), Criterion::MaxAbsDiff(asked)) => asked <= required,
        }
    }

    /// Whether two **finite** values agree under this criterion.
    ///
    /// The NaN cases are decided before this is reached (see [`verdict`]), because "both
    /// sides are still warming up" is a property of the matrices rather than of the bar
    /// they are held to, and folding it in here would make a NaN's verdict depend on a
    /// tolerance nobody applied to it.
    fn holds(self, reference: f64, candidate: f64) -> bool {
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

/// What one cell's pair of values amounts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Both finite and equal under the criterion, or both NaN — warmup agreeing with
    /// warmup.
    Match,
    /// Both non-NaN, and not equal under the criterion.
    Mismatch,
    /// NaN on exactly one side: a window that filled on one path and not the other.
    NanDisagreement,
}

/// The comparison, in one place, with ADR-0016 §4's split written out.
fn verdict(criterion: Criterion, reference: f64, candidate: f64) -> Verdict {
    match (reference.is_nan(), candidate.is_nan()) {
        // Warmup is NaN by construction on both sides and different columns warm up at
        // different lengths, so a bit comparison that included NaN cells would fail
        // every run.
        (true, true) => Verdict::Match,
        // The staleness bug, and the reason `np.allclose` is not the comparison: under
        // its defaults this case is indistinguishable from the one above.
        (true, false) | (false, true) => Verdict::NanDisagreement,
        (false, false) => {
            if criterion.holds(reference, candidate) {
                Verdict::Match
            } else {
                Verdict::Mismatch
            }
        }
    }
}

// ── the report ────────────────────────────────────────────────────────────────

/// One cell the gate has something to say about.
#[derive(Debug, Clone)]
pub struct Divergence {
    pub row: usize,
    /// The column *name*. "Column 3" is not a debuggable statement, and the index moves
    /// the day a spec grows a column while the name does not.
    pub column: String,
    /// Python's value, from the bundle.
    pub reference: f64,
    /// What this build of Rust computed.
    pub candidate: f64,
    /// `NaN` when either side is NaN: an unmeasurable difference is not a small one.
    pub abs_diff: f64,
    /// Whether exactly one side was NaN, which is a different diagnosis from a numeric
    /// disagreement — a window that did not fill, rather than arithmetic that drifted.
    pub nan_disagreement: bool,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The bit patterns sit beside the decimals because at one ULP — the size of the
        // disagreement this gate exists to find — the two decimals render identically,
        // and a failure message showing the same number twice reads as a lie.
        write!(
            f,
            "row {} column {:?}: python {} ({:#018x}) -> rust {} ({:#018x})  delta {:e}{}",
            self.row,
            self.column,
            self.reference,
            self.reference.to_bits(),
            self.candidate,
            self.candidate.to_bits(),
            self.abs_diff,
            if self.nan_disagreement {
                "  NAN ON ONE SIDE"
            } else {
                ""
            },
        )
    }
}

/// Per-column totals, which is how a systematic break is told from a local one.
#[derive(Debug, Clone)]
struct ColumnTally {
    name: String,
    cells: usize,
    nan_disagreements: usize,
    max_abs_diff: f64,
}

/// The outcome of one cross-language feature-parity run.
///
/// A report rather than a bool, for the same reason `axon.parity` returns one: the
/// identical call has to serve a CI assertion and a live parity monitor (`docs/07`), and
/// "feature parity failed" at 03:00 tells an operator nothing about whether the book is
/// being traded on numbers nobody has checked.
#[derive(Debug, Clone)]
pub struct FeatureParityReport {
    label: String,
    criterion: Criterion,
    rows: usize,
    cols: usize,
    cells_compared: usize,
    bit_mismatches: usize,
    nan_disagreements: usize,
    max_abs_diff: f64,
    max_abs_diff_at: Option<(usize, String)>,
    columns: Vec<ColumnTally>,
    libm_columns: Vec<String>,
    divergences: Vec<Divergence>,
}

impl FeatureParityReport {
    /// Whether this build reproduced Python's matrix.
    ///
    /// The `cells_compared > 0` conjunct is not defensive padding. "PASS, 0 cells
    /// compared" is the invisible denominator ADR-0030 spent a whole increment on one
    /// level up: a gate whose corpus quietly emptied reports the same green as a gate
    /// that ran, and the two are indistinguishable in a CI log. A report over nothing is
    /// not a passing report, it is an absent one.
    pub fn passed(&self) -> bool {
        self.cells_compared > 0 && self.bit_mismatches == 0 && self.nan_disagreements == 0
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// How many cells this report is a verdict over. Meant to be asserted rather than
    /// merely printed — see [`Self::passed`].
    pub fn cells_compared(&self) -> usize {
        self.cells_compared
    }

    /// Cells where both sides were non-NaN and the bits differed.
    pub fn bit_mismatches(&self) -> usize {
        self.bit_mismatches
    }

    /// Cells where exactly one side was NaN.
    pub fn nan_disagreements(&self) -> usize {
        self.nan_disagreements
    }

    /// The largest absolute difference over the cells where **both** sides were non-NaN.
    ///
    /// Reported for diagnosis and deliberately **not** the criterion: it is how a reader
    /// tells one ULP of drift from a unit error, and it is the number a tolerance would
    /// have been compared against had this gate had one. [`Self::passed`] never consults
    /// it.
    pub fn max_abs_diff(&self) -> f64 {
        self.max_abs_diff
    }

    /// The column with the most divergent cells, ties broken by magnitude.
    ///
    /// That order is ADR-0016 §4's, and it is the one that matches how the two failures
    /// are diagnosed: a unit error or an off-by-one window breaks *one column on every
    /// row*, while a genuine arithmetic difference is a handful of cells that are
    /// slightly worse. Ranking by magnitude first would name the second while the first
    /// was destroying the matrix.
    pub fn worst_column(&self) -> Option<&str> {
        self.columns
            .iter()
            .filter(|c| c.cells > 0)
            .max_by(|a, b| {
                a.cells.cmp(&b.cells).then_with(|| {
                    a.max_abs_diff
                        .partial_cmp(&b.max_abs_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|c| c.name.as_str())
    }

    /// The columns whose value passes through `log`, as the manifest recorded them.
    pub fn libm_columns(&self) -> &[String] {
        &self.libm_columns
    }

    /// Every divergent cell the report kept, in row-major order — *not* worst-first,
    /// because a systematic break is recognised by its shape and re-sorting hides it.
    pub fn divergences(&self) -> &[Divergence] {
        &self.divergences
    }

    pub fn summary(&self) -> String {
        let head = format!(
            "cross-language feature parity {}: {} rows={} cols={} cells={} criterion={} \
             bit_mismatches={} nan_disagreements={} max_abs_diff={:e}{}",
            if self.passed() { "PASS" } else { "FAIL" },
            self.label,
            self.rows,
            self.cols,
            self.cells_compared,
            self.criterion,
            self.bit_mismatches,
            self.nan_disagreements,
            self.max_abs_diff,
            match &self.max_abs_diff_at {
                Some((row, column)) => format!(" (row {row} column {column:?})"),
                None => String::new(),
            },
        );
        if self.passed() {
            return head;
        }

        let mut lines = vec![head];
        if self.cells_compared == 0 {
            lines.push(
                "  nothing was compared; a report over zero cells is an absent gate rather than \
                 a passing one"
                    .to_string(),
            );
        }
        if let Some(worst) = self.worst_column() {
            let tally = self
                .columns
                .iter()
                .find(|c| c.name == worst)
                .expect("the worst column is one of the tallied columns");
            lines.push(format!(
                "  worst column {:?}: {} of {} cells diverge ({} of them NaN on one side only), \
                 worst delta {:e}",
                worst, tally.cells, self.rows, tally.nan_disagreements, tally.max_abs_diff,
            ));
            lines.push(self.libm_verdict());
        }
        for divergence in &self.divergences {
            lines.push(format!("  {divergence}"));
        }
        let shown = self.divergences.len();
        let total: usize = self.columns.iter().map(|c| c.cells).sum();
        if total > shown {
            lines.push(format!("  ... and more (showing the first {shown})"));
        }
        lines.join("\n")
    }

    /// The `log` question, answered in the failure text rather than left to be asked.
    ///
    /// Both directions are printed, and the negative one is the more useful: knowing
    /// that the divergent columns *do not* touch `log` closes off the one explanation
    /// that would otherwise absorb an afternoon, and it is the answer in every case
    /// except a genuinely different libm.
    fn libm_verdict(&self) -> String {
        if self.libm_columns.is_empty() {
            return "  no column in this spec reaches log, so every operation compared here is \
                    one IEEE-754 requires to be correctly rounded; this is a real divergence and \
                    not a libm difference"
                .to_string();
        }
        let libm: Vec<&str> = self.libm_columns.iter().map(String::as_str).collect();
        let outside: Vec<&str> = self
            .columns
            .iter()
            .filter(|c| c.cells > 0 && !libm.contains(&c.name.as_str()))
            .map(|c| c.name.as_str())
            .collect();
        if outside.is_empty() {
            format!(
                "  every divergent column passes through log ({}); on a libm this repo has not \
                 measured, that is the first thing to check — and it is not a licence to widen \
                 anything, since log agreed to 0 ULP everywhere it has been measured",
                libm.join(", ")
            )
        } else {
            format!(
                "  column(s) {} do not pass through log (the log columns are {}), so a libm \
                 difference does not explain this",
                outside.join(", "),
                libm.join(", ")
            )
        }
    }

    /// Panic with [`Self::summary`] unless the gate passed. The test form of
    /// `raise_for_status()`.
    pub fn assert_passed(&self) {
        assert!(self.passed(), "{}", self.summary());
    }
}

// ── the bundle ────────────────────────────────────────────────────────────────

/// A frozen cross-language feature-parity question, loaded from disk.
#[derive(Debug, Clone)]
pub struct FeatureBundle {
    dir: PathBuf,
    spec: FeatureSpec,
    spec_ref: String,
    description: String,
    source: BTreeMap<String, Value>,
    input_names: Vec<String>,
    inputs: BTreeMap<String, Vec<f64>>,
    reference: FeatureMatrix,
    criterion: Criterion,
    libm_columns: Vec<String>,
}

impl FeatureBundle {
    /// Read and validate a bundle directory.
    ///
    /// Everything checkable without recomputing the matrix is checked here. The order is
    /// the narrative one — what this is, what it must be held to, which recipe, which
    /// shape, which bytes — so the first refusal a broken bundle produces is the most
    /// general true statement about it.
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

        // Before anything expensive, and before anything else the bundle could fail on:
        // a loosened bundle has to be refused *as loosened*, not as whatever it happens
        // to trip over next.
        let declared = manifest.criterion.resolve();
        let required = Criterion::required();
        if !required.allows(declared) {
            return Err(BundleError::Weakened(format!(
                "manifest asks for {declared}; a feature bundle is held to {required} and has no \
                 looser arm to fall back to. Every transform in axon.features is + - * / sqrt, \
                 comparison and log, and IEEE-754 makes all but the last correctly rounded — a \
                 tolerance would buy nothing for the five exact operations and would hide the one \
                 inexact one behind slack granted to every column that never touches it"
            )));
        }

        // A bundle nobody can identify is a bundle nobody can regenerate, and the first
        // thing a red cross-language gate has to answer is what market data it ran on.
        let description = manifest
            .source
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if description.is_empty() {
            return Err(BundleError::Malformed(
                "the source section carries no description; a bundle that cannot say what market \
                 data it froze cannot be regenerated, and a red gate over it names nothing"
                    .to_string(),
            ));
        }

        // ── the recipe ──
        let spec_path = bundle_file(&dir, &manifest.spec.file)?;
        let spec_text = read_to_string(&spec_path)?;
        if spec_text.len() != manifest.spec.bytes {
            return Err(BundleError::Malformed(format!(
                "{}: holds {} bytes, the manifest records {}",
                spec_path.display(),
                spec_text.len(),
                manifest.spec.bytes
            )));
        }
        // `from_json` **recomputes** the fingerprint rather than reading it, so this one
        // call refuses a spec edited after it was written, a spec from a different build
        // of the library, and — the one only Rust can catch — the two languages
        // disagreeing about what the spec *is*. See [`crate::spec`].
        let spec = FeatureSpec::from_json(&spec_text)
            .map_err(|e| BundleError::Malformed(format!("{}: {e}", spec_path.display())))?;
        // Byte for byte, and deliberately without a trim: `spec.json` is exactly
        // `FeatureSpec.to_json()` with no trailing newline, while `manifest.json` ends in
        // one. `spec_ref` names a hash of these bytes, so a pretty-printed file that
        // parses to the same spec is no longer the thing the reference names, and a
        // reader that trimmed whitespace would mask precisely the difference only a
        // re-serialization can see — the two languages disagreeing about how the recipe
        // is *written*, which is a fingerprint that will not reproduce anywhere else.
        if spec_text != spec.canonical_json() {
            return Err(BundleError::Malformed(format!(
                "{} is not the canonical serialization of the spec it parses to; the fingerprint \
                 identifies a recipe, and these are not the bytes that recipe writes",
                spec_path.display()
            )));
        }
        if spec.reference() != manifest.spec_ref {
            return Err(BundleError::Malformed(format!(
                "manifest names {}, the spec on disk re-identifies as {}",
                manifest.spec_ref,
                spec.reference()
            )));
        }
        if spec.library_version() != manifest.library_version {
            return Err(BundleError::Malformed(format!(
                "manifest records library_version {}, the spec beside it says {}; the two halves \
                 of the bundle disagree about which build of axon.features wrote it",
                manifest.library_version,
                spec.library_version()
            )));
        }

        // ── the names ──
        let required_inputs = spec
            .required_inputs()
            .map_err(|e| BundleError::Malformed(format!("{}: {e}", manifest.spec_ref)))?;
        if manifest.inputs.names != required_inputs {
            return Err(BundleError::Malformed(format!(
                "inputs are named {:?}, {} reads {:?}; this side pairs arrays with names \
                 positionally from that list, so a mismatch here mislabels every column at once",
                manifest.inputs.names, manifest.spec_ref, required_inputs
            )));
        }
        let columns: Vec<String> = spec.columns().into_iter().map(str::to_string).collect();
        if manifest.features.names != columns {
            return Err(BundleError::Malformed(format!(
                "matrix columns are named {:?}, {} produces {:?}",
                manifest.features.names, manifest.spec_ref, columns
            )));
        }
        // The signpost is **re-derived**, not read. Both languages carry a
        // transform-level "does this reach `log`" flag — `LIBM_FEATURES` in
        // `axon.parity.feature_bundle`, `FeatureInfo::reaches_libm` here — and both
        // walk the spec's own binding graph to turn it into a column list, because
        // the dependency is inherited: a z-score of a log return never calls `log`
        // and is nonetheless a function of one.
        //
        // Checking it rather than believing it is the same argument that makes the
        // fingerprint recomputed. A field Rust reads and trusts is a field that can
        // be right about a spec nobody is running: the two tables drifting apart is
        // exactly the failure the signpost would then hide, since the whole reason
        // it exists is to answer "did this gate redden *only* on the log columns?" on
        // a platform nobody here has measured. Two copies that agree because they are
        // compared is a different thing from two copies that agree because nobody has
        // looked.
        let derived = spec.libm_columns()?;
        if derived != manifest.libm_columns {
            return Err(BundleError::Malformed(format!(
                "the manifest's libm signpost is {:?}, this build derives {derived:?} from {}; \
                 the two languages disagree about which columns pass through log, so the one \
                 question a platform failure would be asked has two answers",
                manifest.libm_columns, manifest.spec_ref
            )));
        }

        // ── the shapes ──
        let rows = manifest.inputs.rows;
        if rows == 0 {
            return Err(BundleError::Malformed(
                "a feature-parity gate over zero rows proves nothing; the two languages agree \
                 about the empty matrix and disagree about everything else"
                    .to_string(),
            ));
        }
        if manifest.inputs.cols != manifest.inputs.names.len() {
            return Err(BundleError::Malformed(format!(
                "{} input columns for {} names",
                manifest.inputs.cols,
                manifest.inputs.names.len()
            )));
        }
        if manifest.features.rows != rows {
            return Err(BundleError::Malformed(format!(
                "{rows} input rows against {} feature rows; the two matrices are not describing \
                 the same events",
                manifest.features.rows
            )));
        }
        if manifest.features.cols != columns.len() {
            return Err(BundleError::Malformed(format!(
                "{} feature columns for {} names",
                manifest.features.cols,
                columns.len()
            )));
        }
        for entry in [&manifest.inputs, &manifest.features] {
            let want = entry.rows * entry.cols * 8;
            if entry.bytes != want {
                return Err(BundleError::Malformed(format!(
                    "{:?}: the manifest declares {} bytes and describes {}x{} f64 = {want} bytes",
                    entry.file, entry.bytes, entry.rows, entry.cols
                )));
            }
        }

        // ── the bytes ──
        let inputs_flat = read_f64(
            &bundle_file(&dir, &manifest.inputs.file)?,
            rows,
            manifest.inputs.cols,
        )?;
        let features_flat = read_f64(
            &bundle_file(&dir, &manifest.features.file)?,
            rows,
            manifest.features.cols,
        )?;

        let input_nans = inputs_flat.iter().filter(|v| v.is_nan()).count();
        if input_nans != manifest.inputs.nan_cells {
            return Err(BundleError::Malformed(format!(
                "manifest claims {} NaN input cells, the matrix holds {input_nans}",
                manifest.inputs.nan_cells
            )));
        }
        let reference = FeatureMatrix::new(columns, features_flat)
            .map_err(|e| BundleError::Malformed(format!("the recorded matrix: {e}")))?;
        let feature_nans = reference.nan_cells();
        if feature_nans != manifest.features.nan_cells {
            // Warmup is NaN by construction and its extent is part of what this bundle
            // asserts: a matrix whose warmup silently became zeros has the same shape, a
            // different meaning, and nothing else in the format would notice.
            return Err(BundleError::Malformed(format!(
                "manifest claims {} NaN feature cells, the matrix holds {feature_nans}",
                manifest.features.nan_cells
            )));
        }
        let declared_finite = manifest.features.finite_rows.ok_or_else(|| {
            BundleError::Malformed(
                "the features section records no finite_rows; how much of a corpus is a real \
                 comparison is a number rather than a shape, and it belongs in the record"
                    .to_string(),
            )
        })?;
        let finite_rows = reference.finite_rows();
        if finite_rows != declared_finite {
            return Err(BundleError::Malformed(format!(
                "manifest claims {declared_finite} fully finite rows, the matrix holds \
                 {finite_rows}"
            )));
        }
        if finite_rows == 0 {
            // This gate's version of the model bundle's "every row decides the same way":
            // an all-NaN reference matches an all-NaN candidate under the NaN rule, so
            // such a bundle would pass for a runtime that computes nothing at all.
            return Err(BundleError::Malformed(format!(
                "not one of {rows} rows has every column finite — the corpus never finishes \
                 warming up, and NaN matching NaN would make this bundle pass for a runtime that \
                 computes nothing at all"
            )));
        }

        // Row-major, de-interleaved into the named arrays `FeatureSpec::compute` takes.
        // Sliced out of the matrix that was actually on disk, so recomputing from these
        // is recomputing from the bytes the reference was taken over — the only version
        // of the question worth asking.
        let mut inputs = BTreeMap::new();
        for (j, name) in manifest.inputs.names.iter().enumerate() {
            let column: Vec<f64> = (0..rows)
                .map(|i| inputs_flat[i * manifest.inputs.cols + j])
                .collect();
            inputs.insert(name.clone(), column);
        }

        Ok(Self {
            dir,
            spec,
            spec_ref: manifest.spec_ref,
            description,
            source: manifest.source,
            input_names: manifest.inputs.names,
            inputs,
            reference,
            criterion: declared,
            libm_columns: manifest.libm_columns,
        })
    }

    /// The recipe, re-identified from `spec.json` rather than believed.
    pub fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    /// `name/vN#fingerprint`, as the manifest records it and the spec on disk confirms.
    pub fn spec_ref(&self) -> &str {
        &self.spec_ref
    }

    pub fn rows(&self) -> usize {
        self.reference.rows()
    }

    /// The width of the **feature** matrix. The input width is `input_names().len()`,
    /// and the two are different numbers for every real spec.
    pub fn cols(&self) -> usize {
        self.reference.cols()
    }

    /// The input array names, sorted — the spec's own `required_inputs`.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn inputs(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.inputs
    }

    /// Python's own matrix, as recorded.
    pub fn reference(&self) -> &FeatureMatrix {
        &self.reference
    }

    pub fn criterion(&self) -> Criterion {
        self.criterion
    }

    /// The columns whose value passes through `log`. A signpost, never a tolerance.
    pub fn libm_columns(&self) -> &[String] {
        &self.libm_columns
    }

    /// What market data this froze — instrument, interval, venue, description.
    pub fn source(&self) -> &BTreeMap<String, Value> {
        &self.source
    }

    /// What this build of Rust computes from the bundle's own inputs.
    pub fn candidate(&self) -> Result<FeatureMatrix, FeatureError> {
        self.spec.compute(&self.inputs)
    }

    /// Compare a candidate matrix against the frozen reference, under the bundle's own
    /// criterion.
    pub fn compare(&self, candidate: &FeatureMatrix) -> Result<FeatureParityReport, BundleError> {
        if candidate.rows() != self.rows() {
            return Err(BundleError::Mismatch(format!(
                "{} candidate rows against {} reference rows; a length mismatch means the two \
                 runs did not see the same events, and zipping them would compare the rows that \
                 happen to line up and call it parity",
                candidate.rows(),
                self.rows()
            )));
        }
        if candidate.columns() != self.reference.columns() {
            // Not merely a shape check: two matrices of the same width whose columns are
            // in a different order compare cell for cell perfectly happily, and every
            // value is then attributed to the wrong feature.
            return Err(BundleError::Mismatch(format!(
                "candidate columns {:?} against reference columns {:?}",
                candidate.columns(),
                self.reference.columns()
            )));
        }

        let mut report = FeatureParityReport {
            label: format!(
                "{} over {} ({})",
                self.spec_ref,
                self.description,
                self.dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            criterion: self.criterion,
            rows: self.rows(),
            cols: self.cols(),
            cells_compared: 0,
            bit_mismatches: 0,
            nan_disagreements: 0,
            max_abs_diff: 0.0,
            max_abs_diff_at: None,
            columns: self
                .reference
                .columns()
                .iter()
                .map(|name| ColumnTally {
                    name: name.clone(),
                    cells: 0,
                    nan_disagreements: 0,
                    max_abs_diff: 0.0,
                })
                .collect(),
            libm_columns: self.libm_columns.clone(),
            divergences: Vec::new(),
        };

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                let reference = self.reference.get(row, col);
                let candidate = candidate.get(row, col);
                report.cells_compared += 1;

                // Measured even where the pair matches, because "how far apart were they
                // at worst" is the first thing read off a *passing* report too: a gate
                // that only measures its failures cannot show that it is nowhere near
                // failing. NaN on either side leaves it NaN, and a NaN never wins a `>`
                // comparison, so an unmeasurable difference is never summarized as a
                // small one.
                let abs_diff = if reference.is_nan() || candidate.is_nan() {
                    f64::NAN
                } else {
                    (reference - candidate).abs()
                };
                if abs_diff > report.max_abs_diff {
                    report.max_abs_diff = abs_diff;
                    report.max_abs_diff_at = Some((row, report.columns[col].name.clone()));
                }
                if abs_diff > report.columns[col].max_abs_diff {
                    report.columns[col].max_abs_diff = abs_diff;
                }

                match verdict(self.criterion, reference, candidate) {
                    Verdict::Match => continue,
                    Verdict::Mismatch => report.bit_mismatches += 1,
                    Verdict::NanDisagreement => {
                        report.nan_disagreements += 1;
                        report.columns[col].nan_disagreements += 1;
                    }
                }
                report.columns[col].cells += 1;
                if report.divergences.len() < DIVERGENCE_LIMIT {
                    report.divergences.push(Divergence {
                        row,
                        column: report.columns[col].name.clone(),
                        reference,
                        candidate,
                        abs_diff,
                        nan_disagreement: reference.is_nan() != candidate.is_nan(),
                    });
                }
            }
        }
        Ok(report)
    }

    /// Compute and compare — the whole gate.
    pub fn check(&self) -> Result<FeatureParityReport, BundleError> {
        let candidate = self.candidate()?;
        self.compare(&candidate)
    }
}

// ── the manifest ──
//
// Unknown keys are tolerated and `bundle_schema` is the guard, so adding a descriptive
// field to the writer does not strand every committed bundle. A key that changes the
// *question* has to bump the schema. The recorded `sha256`s are absent from these
// structs rather than read and ignored: see the module docstring.

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    bundle_schema: u32,
    spec_ref: String,
    library_version: u32,
    source: BTreeMap<String, Value>,
    spec: FileEntry,
    inputs: MatrixEntry,
    features: MatrixEntry,
    criterion: CriterionEntry,
    libm_columns: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FileEntry {
    file: String,
    bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct MatrixEntry {
    file: String,
    bytes: usize,
    rows: usize,
    cols: usize,
    names: Vec<String>,
    nan_cells: usize,
    /// Only the feature matrix records this. `open` requires it there by name, so the
    /// refusal reads as "the features section is incomplete" rather than as a parse
    /// error about the inputs.
    #[serde(default)]
    finite_rows: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CriterionEntry {
    BitExact,
    MaxAbsDiff { eps: f64 },
}

impl CriterionEntry {
    fn resolve(self) -> Criterion {
        match self {
            CriterionEntry::BitExact => Criterion::BitExact,
            CriterionEntry::MaxAbsDiff { eps } => Criterion::MaxAbsDiff(eps),
        }
    }
}

// ── bytes ─────────────────────────────────────────────────────────────────────

/// A manifest-named file, resolved inside the bundle directory and nowhere else.
///
/// A manifest is data, and a bundle received from elsewhere is data somebody else wrote.
/// A path in it that escapes the bundle turns reading a fixture into a file read of
/// whoever wrote it choosing. The Python reader enforces the same rule — one rule,
/// checked twice, because the two readers are handed the same untrusted bytes.
fn bundle_file(dir: &Path, name: &str) -> Result<PathBuf, BundleError> {
    if Path::new(name)
        .file_name()
        .map(|n| n != name)
        .unwrap_or(true)
    {
        return Err(BundleError::Malformed(format!(
            "{name:?} is not a bare filename; a bundle names its own files and nothing else"
        )));
    }
    Ok(dir.join(name))
}

fn read_file(path: &Path) -> Result<Vec<u8>, BundleError> {
    fs::read(path).map_err(|source| BundleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_to_string(path: &Path) -> Result<String, BundleError> {
    fs::read_to_string(path).map_err(|source| BundleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_f64(path: &Path, rows: usize, cols: usize) -> Result<Vec<f64>, BundleError> {
    let bytes = read_file(path)?;
    let want = rows * cols * 8;
    if bytes.len() != want {
        // A truncated matrix would otherwise be read as a shorter corpus with a
        // plausible shape, and the gate would pass on the rows that survived.
        return Err(BundleError::Malformed(format!(
            "{}: holds {} bytes; the manifest describes {rows}x{cols} f64 = {want} bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_that_compared_nothing_does_not_pass() {
        // The invisible denominator, one level down from ADR-0030. Every counter is zero
        // and there is nothing to complain about, which is exactly how an empty corpus
        // looks in a CI log — indistinguishable from a gate that ran. `open` refuses a
        // zero-row bundle, so this report cannot be produced through the public path;
        // the invariant is asserted here rather than left to that refusal, because
        // `passed()` is what callers read and it has to be true on its own.
        let empty = FeatureParityReport {
            label: "nothing/v1#0000000000000000 over nothing (nowhere)".to_string(),
            criterion: Criterion::BitExact,
            rows: 0,
            cols: 0,
            cells_compared: 0,
            bit_mismatches: 0,
            nan_disagreements: 0,
            max_abs_diff: 0.0,
            max_abs_diff_at: None,
            columns: Vec::new(),
            libm_columns: Vec::new(),
            divergences: Vec::new(),
        };
        assert!(!empty.passed());
        let summary = empty.summary();
        assert!(summary.contains("FAIL"), "{summary}");
        assert!(summary.contains("nothing was compared"), "{summary}");
    }

    #[test]
    fn both_nan_matches_one_nan_does_not_and_two_finites_are_compared_by_bits() {
        // ADR-0016 §4's split, which `np.allclose` defaults cannot express: under them
        // the first two cases here are indistinguishable.
        let c = Criterion::BitExact;
        assert_eq!(verdict(c, f64::NAN, f64::NAN), Verdict::Match);
        assert_eq!(verdict(c, f64::NAN, 0.0), Verdict::NanDisagreement);
        assert_eq!(verdict(c, 0.0, f64::NAN), Verdict::NanDisagreement);
        assert_eq!(verdict(c, 1.5, 1.5), Verdict::Match);
        let one_ulp = f64::from_bits(1.5f64.to_bits() + 1);
        assert_eq!(verdict(c, 1.5, one_ulp), Verdict::Mismatch);
    }

    #[test]
    fn a_nan_payload_is_not_compared_because_neither_language_promises_one() {
        // `axon.features` masks every quotient, so its NaNs are the literal `np.nan`
        // (0x7ff8000000000000). x86's own invalid-operation result is the same value
        // with the sign bit set. A payload comparison would redden a Rust transform that
        // divided first and masked afterwards while computing the right answer — a
        // defect in the gate rather than in the runtime.
        let python_nan = f64::from_bits(0x7ff8_0000_0000_0000);
        let x86_nan = f64::from_bits(0xfff8_0000_0000_0000);
        assert_ne!(python_nan.to_bits(), x86_nan.to_bits());
        assert_eq!(
            verdict(Criterion::BitExact, python_nan, x86_nan),
            Verdict::Match
        );
    }

    #[test]
    fn signed_zero_is_a_mismatch_because_a_sign_that_reaches_a_zscore_is_a_disagreement() {
        // The stronger half of bit equality, and the same stance `axon-model` takes: a
        // numeric `0.0` comparison accepts `+0.0` against `-0.0`, and the bits do not.
        assert_eq!(verdict(Criterion::BitExact, 0.0, -0.0), Verdict::Mismatch);
    }

    #[test]
    fn a_feature_bundle_may_not_declare_a_tolerance_however_tight() {
        // Not a numerical judgement but a procedural one: a bundle regenerated after a
        // red gate, with a tolerance added until it passed, is otherwise
        // indistinguishable in the tree from one that never failed.
        let required = Criterion::required();
        assert!(required.allows(Criterion::BitExact));
        assert!(!required.allows(Criterion::MaxAbsDiff(1e-12)));
        assert!(!required.allows(Criterion::MaxAbsDiff(f64::MIN_POSITIVE)));
    }

    #[test]
    fn a_manifest_filename_that_escapes_the_bundle_is_refused() {
        // A manifest is data. `../../.ssh/id_ed25519` is a perfectly good relative path,
        // and a bundle from elsewhere is somebody else's bytes.
        let dir = Path::new("/tmp/bundle");
        assert!(bundle_file(dir, "inputs.f64").is_ok());
        for bad in ["../inputs.f64", "sub/inputs.f64", "/etc/passwd", ".."] {
            assert!(
                matches!(bundle_file(dir, bad), Err(BundleError::Malformed(_))),
                "{bad:?} was accepted as a bundle-local filename"
            );
        }
    }
}
