//! The versioned feature spec, read back in Rust — and re-identified, not trusted.
//!
//! A [`FeatureSpec`] is what turns a pile of transforms into a library: an ordered,
//! named, hash-identified list of `(feature, params)`. It is the same object
//! `axon.features.spec.FeatureSpec` is, and it crosses as the canonical JSON that
//! object writes.
//!
//! ## Why the fingerprint is recomputed here
//!
//! The cheap version of this module would read the recorded `fingerprint` field and
//! carry it around as a label. That would make the identity a *claim* rather than a
//! check, and identity is the whole mechanism: `docs/03`'s prime directive is
//! enforced by a hash that moves when the recipe moves. Recomputing it in Rust buys
//! three specific refusals that a label cannot:
//!
//! - **A payload edited after its fingerprint was taken** is refused. Somebody
//!   widening a window in a committed `spec.json` to make a gate pass is exactly the
//!   shape ADR-0021 built `Criterion::allows` against, one level up.
//! - **A spec written against a different build of `axon.features`** is refused by
//!   name, because [`crate::FEATURES_VERSION`] is folded into the hash. Without it
//!   the fingerprint would pin the recipe and say nothing about the kitchen:
//!   rewriting `rolling_std` under a spec's feet leaves every artifact id unchanged
//!   while quietly feeding every model different numbers.
//! - **The two languages disagreeing about what the spec *is*** is refused — and this
//!   is the one only Rust can catch. If this crate parsed the JSON into a structure
//!   that serialized back differently (a dropped empty `params`, an unsorted key, an
//!   int read as a float), the recomputed hash would not match and the load fails
//!   loudly. A Rust runtime that merely *believed* the recorded id could be computing
//!   a different recipe under the right name, and nothing downstream would ever say
//!   so. The fingerprint check is therefore a cross-language conformance test that
//!   runs on every load, not only in CI.
//!
//! ## Canonical JSON, and the one place it can be wrong
//!
//! Python writes `json.dumps(payload, sort_keys=True, separators=(",", ":"),
//! allow_nan=False)`. [`FeatureSpec::canonical_json`] reproduces that: sorted keys at
//! every level, no whitespace, `ensure_ascii` escaping. The `features` *list* keeps
//! its order, which is the point — column order is part of what is being identified,
//! because permuting two columns leaves every name correct and every prediction
//! wrong.
//!
//! Float parameters are the soft spot: Python renders a float with `repr`, and no two
//! languages agree on shortest-round-trip formatting in every corner. It is a soft
//! spot rather than a hazard because of how it fails — a formatting disagreement
//! makes the recomputed hash differ, which is a loud [`FeatureError::Mismatch`] at
//! load time, never a spec that quietly computes something else. Every spec in this
//! repo uses integer parameters only.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::registry::{feature_info, Param, Params};
use crate::{FeatureError, EXACT_FLOAT_LIMIT, FEATURES_VERSION};

/// One column of a feature matrix: a registered transform plus its bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDef {
    column: String,
    feature: String,
    params: Params,
    /// Overrides for where each declared input comes from — either a key of the
    /// caller's inputs mapping or an **earlier** column of the same spec. Unbound
    /// inputs read the source of the same name.
    bindings: BTreeMap<String, String>,
}

impl FeatureDef {
    pub fn new(
        column: impl Into<String>,
        feature: impl Into<String>,
        params: Params,
        bindings: BTreeMap<String, String>,
    ) -> Result<Self, FeatureError> {
        let column = column.into();
        let feature = feature.into();
        if column.trim().is_empty() {
            return Err(FeatureError::Spec(format!(
                "column must be a non-empty string, got {column:?}"
            )));
        }
        let info = feature_info(&feature)?;

        // Silently ignoring a misspelled window is how a spec claims a 32-sample
        // volatility and delivers the default one, with nothing to show for it.
        let unknown: Vec<&str> = params.keys().filter(|k| !info.params.contains(k)).collect();
        if !unknown.is_empty() {
            return Err(FeatureError::Spec(format!(
                "{feature} does not take parameter(s) {unknown:?}; it accepts {:?}",
                info.params
            )));
        }
        let unbound: Vec<&String> = bindings
            .keys()
            .filter(|k| !info.inputs.contains(&k.as_str()))
            .collect();
        if !unbound.is_empty() {
            return Err(FeatureError::Spec(format!(
                "{feature} has no input(s) {unbound:?}; it reads {:?}",
                info.inputs
            )));
        }
        for (key, source) in &bindings {
            if source.trim().is_empty() {
                return Err(FeatureError::Spec(format!(
                    "{feature}.{key} must bind to a name, got {source:?}"
                )));
            }
        }
        Ok(Self {
            column,
            feature,
            params,
            bindings,
        })
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    pub fn feature(&self) -> &str {
        &self.feature
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn bindings(&self) -> &BTreeMap<String, String> {
        &self.bindings
    }

    /// Where each positional input actually comes from, in call order.
    pub fn sources(&self) -> Result<Vec<String>, FeatureError> {
        let info = feature_info(&self.feature)?;
        Ok(info
            .inputs
            .iter()
            .map(|name| {
                self.bindings
                    .get(*name)
                    .cloned()
                    .unwrap_or_else(|| (*name).to_string())
            })
            .collect())
    }

    fn to_value(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("column".into(), Value::String(self.column.clone()));
        obj.insert("feature".into(), Value::String(self.feature.clone()));
        // `params` and `inputs` are always present even when empty: an omitted key
        // and an empty mapping hash differently while meaning the same thing, and
        // Python emits both unconditionally.
        obj.insert("inputs".into(), {
            let mut m = serde_json::Map::new();
            for (k, v) in &self.bindings {
                m.insert(k.clone(), Value::String(v.clone()));
            }
            Value::Object(m)
        });
        obj.insert("params".into(), {
            let mut m = serde_json::Map::new();
            for (k, v) in self.params.iter() {
                m.insert(k.to_string(), param_to_value(v));
            }
            Value::Object(m)
        });
        Value::Object(obj)
    }

    fn from_value(value: &Value) -> Result<Self, FeatureError> {
        let obj = value
            .as_object()
            .ok_or_else(|| FeatureError::Spec("a feature definition must be an object".into()))?;
        let unknown: Vec<&String> = obj
            .keys()
            .filter(|k| !matches!(k.as_str(), "column" | "feature" | "params" | "inputs"))
            .collect();
        if !unknown.is_empty() {
            return Err(FeatureError::Spec(format!(
                "unknown feature-definition key(s) {unknown:?}"
            )));
        }
        let column = string_field(obj, "column")?;
        let feature = string_field(obj, "feature")?;

        let mut params = Params::new();
        if let Some(p) = obj.get("params") {
            let map = p
                .as_object()
                .ok_or_else(|| FeatureError::Spec(format!("{column}: params must be an object")))?;
            for (k, v) in map {
                params.insert(k.clone(), value_to_param(&feature, k, v)?);
            }
        }

        let mut bindings = BTreeMap::new();
        if let Some(i) = obj.get("inputs") {
            let map = i
                .as_object()
                .ok_or_else(|| FeatureError::Spec(format!("{column}: inputs must be an object")))?;
            for (k, v) in map {
                let source = v.as_str().ok_or_else(|| {
                    FeatureError::Spec(format!("{column}.{k} must bind to a string name"))
                })?;
                bindings.insert(k.clone(), source.to_string());
            }
        }
        FeatureDef::new(column, feature, params, bindings)
    }
}

/// An ordered, named, hash-identified list of `(feature, params)`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSpec {
    name: String,
    version: u32,
    library_version: u32,
    features: Vec<FeatureDef>,
}

impl FeatureSpec {
    pub fn new(
        name: impl Into<String>,
        version: u32,
        library_version: u32,
        features: Vec<FeatureDef>,
    ) -> Result<Self, FeatureError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(FeatureError::Spec(format!(
                "spec name must be a non-empty string, got {name:?}"
            )));
        }
        if version < 1 {
            return Err(FeatureError::Spec(format!(
                "spec version must be >= 1, got {version}"
            )));
        }
        if features.is_empty() {
            return Err(FeatureError::Spec(
                "a spec with no features would produce an empty matrix".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for def in &features {
            if !seen.insert(def.column.clone()) {
                // Two columns with one name: the second silently wins as a binding
                // source and the matrix carries both, so the model and the spec
                // disagree about which column is which.
                return Err(FeatureError::Spec(format!(
                    "duplicate column name {:?}",
                    def.column
                )));
            }
        }
        Ok(Self {
            name,
            version,
            library_version,
            features,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// The build of `axon.features` this spec was written against.
    pub fn library_version(&self) -> u32 {
        self.library_version
    }

    pub fn features(&self) -> &[FeatureDef] {
        &self.features
    }

    /// The matrix column names, in matrix order.
    pub fn columns(&self) -> Vec<&str> {
        self.features.iter().map(|d| d.column.as_str()).collect()
    }

    /// The input arrays a caller must supply, sorted.
    ///
    /// Anything a feature reads that is not produced by an earlier column in the same
    /// spec. Sorted because the feature bundle pins `inputs.names` to exactly this
    /// list, and the two sides pairing arrays with different names would make every
    /// column wrong in a way that reads as a transform bug.
    pub fn required_inputs(&self) -> Result<Vec<String>, FeatureError> {
        let mut produced: BTreeSet<String> = BTreeSet::new();
        let mut needed: BTreeSet<String> = BTreeSet::new();
        for def in &self.features {
            for source in def.sources()? {
                if !produced.contains(&source) {
                    needed.insert(source);
                }
            }
            produced.insert(def.column.clone());
        }
        Ok(needed.into_iter().collect())
    }

    // ── identity ──────────────────────────────────────────────────────────────

    /// The canonical JSON the fingerprint is taken over — the *body*, without the
    /// fingerprint field itself.
    pub fn canonical_body(&self) -> String {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "features".into(),
            Value::Array(self.features.iter().map(|d| d.to_value()).collect()),
        );
        obj.insert("library_version".into(), Value::from(self.library_version));
        obj.insert("spec".into(), Value::String(self.name.clone()));
        obj.insert("version".into(), Value::from(self.version));
        canonical_json(&Value::Object(obj))
    }

    /// The serialized spec with its fingerprint attached — byte-identical to
    /// `FeatureSpec.to_json()` on the Python side.
    pub fn canonical_json(&self) -> String {
        let body = self.canonical_body();
        // Re-parsing rather than string-splicing: the fingerprint key has to land in
        // sorted position ("features", "fingerprint", "library_version", "spec",
        // "version"), and splicing it in would put it wherever the author guessed.
        let mut value: Value = serde_json::from_str(&body).expect("canonical body is valid JSON");
        if let Some(obj) = value.as_object_mut() {
            obj.insert("fingerprint".into(), Value::String(fingerprint_of(&body)));
        }
        canonical_json(&value)
    }

    /// Content hash of the whole recipe — the id an artifact records.
    pub fn fingerprint(&self) -> String {
        fingerprint_of(&self.canonical_body())
    }

    /// `name/vN#fingerprint` — what goes in a model artifact's spec reference.
    pub fn reference(&self) -> String {
        format!("{}/v{}#{}", self.name, self.version, self.fingerprint())
    }

    // ── serialization ─────────────────────────────────────────────────────────

    /// Read a spec and verify it still describes what this library would compute.
    ///
    /// This is the load path. It performs both identity checks; see the module
    /// docstring for what each one buys.
    pub fn from_json(text: &str) -> Result<Self, FeatureError> {
        let (spec, recorded) = Self::parse(text)?;
        if let Some(recorded) = recorded {
            let actual = spec.fingerprint();
            if recorded != actual {
                return Err(FeatureError::Mismatch(format!(
                    "spec fingerprint {recorded} does not match the recomputed {actual}; either \
                     the payload was altered after it was written, or this build serializes the \
                     recipe differently from the one that wrote it"
                )));
            }
        }
        if spec.library_version != FEATURES_VERSION {
            return Err(FeatureError::Mismatch(format!(
                "spec {:?} was written against axon.features v{}, this build is v{}; the \
                 transforms changed meaning, so the model would be fed different numbers",
                spec.name, spec.library_version, FEATURES_VERSION
            )));
        }
        Ok(spec)
    }

    /// Read a spec **without** the identity checks.
    ///
    /// For tooling that needs to *inspect* an incompatible spec in order to report
    /// what it wanted. It must never be used on a serving path: the whole point of
    /// the fingerprint is that a model trained on other features refuses to run.
    pub fn from_json_lax(text: &str) -> Result<Self, FeatureError> {
        Ok(Self::parse(text)?.0)
    }

    fn parse(text: &str) -> Result<(Self, Option<String>), FeatureError> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| FeatureError::Spec(format!("spec is not valid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| FeatureError::Spec("a spec payload must be an object".into()))?;

        const KEYS: [&str; 5] = [
            "spec",
            "version",
            "library_version",
            "features",
            "fingerprint",
        ];
        let unknown: Vec<&String> = obj.keys().filter(|k| !KEYS.contains(&k.as_str())).collect();
        if !unknown.is_empty() {
            return Err(FeatureError::Spec(format!(
                "unknown spec key(s) {unknown:?}"
            )));
        }
        let missing: Vec<&str> = ["spec", "version", "features"]
            .into_iter()
            .filter(|k| !obj.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            return Err(FeatureError::Spec(format!(
                "spec payload is missing key(s) {missing:?}"
            )));
        }

        let name = string_field(obj, "spec")?;
        let version = u32_field(obj, "version")?;
        let library_version = match obj.get("library_version") {
            Some(v) => u32_of(v, "library_version")?,
            None => FEATURES_VERSION,
        };
        let features = obj["features"]
            .as_array()
            .ok_or_else(|| FeatureError::Spec("features must be a list".into()))?
            .iter()
            .map(FeatureDef::from_value)
            .collect::<Result<Vec<_>, _>>()?;

        let recorded = obj
            .get("fingerprint")
            .and_then(Value::as_str)
            .map(str::to_string);
        Ok((
            Self::new(name, version, library_version, features)?,
            recorded,
        ))
    }

    /// The columns whose value passes through a **logarithm**, in matrix order.
    ///
    /// Mirrors `axon.parity.feature_bundle.libm_columns`, and the mirroring is the
    /// point: a feature bundle records this list, and if only one language could
    /// derive it the other would be reading a field and believing it. Recomputing it
    /// here makes the signpost a *checked* cross-language property, on the same
    /// argument that makes the fingerprint recomputed rather than read.
    ///
    /// Derived by walking the dependency graph rather than listed per transform,
    /// because the dependency is inherited through a **binding**: a z-score of a log
    /// return never calls `log` and is nonetheless a function of one. A hand-written
    /// list would be right for the spec it was written against and silently wrong
    /// for the next one.
    ///
    /// It is not a tolerance and nothing widens for the columns it names. It is the
    /// first question to ask when a bit-exact feature gate reddens on a platform
    /// this repo has not measured: did it redden *only* here?
    pub fn libm_columns(&self) -> Result<Vec<String>, FeatureError> {
        let mut tainted: BTreeSet<&str> = BTreeSet::new();
        for def in &self.features {
            let info = feature_info(&def.feature)?;
            let inherited = def.sources()?.iter().any(|s| tainted.contains(s.as_str()));
            if info.reaches_libm || inherited {
                tainted.insert(def.column.as_str());
            }
        }
        // Matrix order, not sorted: the list is read beside a column list and a
        // different order in the two places invites somebody to compare them by eye
        // and conclude they disagree.
        Ok(self
            .columns()
            .into_iter()
            .filter(|c| tainted.contains(c))
            .map(str::to_string)
            .collect())
    }

    // ── lookback ──────────────────────────────────────────────────────────────

    /// How many trailing raw observations the whole spec needs before every column
    /// can be finite — or `None` if any column is unbounded.
    ///
    /// This is the number [`crate::streaming::FeatureStream`] sizes its buffer from,
    /// and it is *derived* rather than declared. That distinction has a specific
    /// failure behind it: a constant restating the warmup goes stale the day a window
    /// widens, the buffer is then one observation short, every row stays NaN, the
    /// strategy emits nothing forever, and nothing raises. A strategy that never
    /// trades looks exactly like a strategy with no opinion.
    ///
    /// For `BAR_M1_V1` this comes out at **21**, which is
    /// `axon.features.spec.BAR_M1_WARMUP_BARS` — asserted against it in
    /// `tests/cross_language.rs::the_bar_spec_warmup_rust_derives_is_the_one_python_declares`
    /// rather than restated here.
    pub fn max_lookback(&self) -> Result<Option<usize>, FeatureError> {
        let mut best = 1usize;
        for def in &self.features {
            match self.column_lookback(&def.column)? {
                None => return Ok(None),
                Some(d) => best = best.max(d),
            }
        }
        Ok(Some(best))
    }

    /// The lookback of one column, in raw observations.
    ///
    /// Composed through bindings: a column reading an *earlier column* needs its own
    /// window plus whatever that column needed, less the one observation they share.
    /// Getting the overlap wrong by one is the same off-by-one as above, so the
    /// arithmetic is written out rather than approximated upward — a buffer that is
    /// generously too long hides a spec whose warmup nobody has measured.
    pub fn column_lookback(&self, column: &str) -> Result<Option<usize>, FeatureError> {
        // `usize::MAX` is the in-loop spelling of "unbounded", so a column can carry
        // the state forward to whatever reads it. It never escapes this function.
        const UNBOUNDED: usize = usize::MAX;
        let mut depth: BTreeMap<&str, usize> = BTreeMap::new();
        for def in &self.features {
            let info = feature_info(&def.feature)?;
            let own = (info.lookback)(&def.params)?;
            let mut deepest_source = 1usize;
            for source in def.sources()? {
                let d = *depth.get(source.as_str()).unwrap_or(&1);
                deepest_source = deepest_source.max(d);
            }
            let total = match own {
                None => UNBOUNDED,
                Some(_) if deepest_source == UNBOUNDED => UNBOUNDED,
                Some(own) => own + deepest_source - 1,
            };
            depth.insert(def.column.as_str(), total);
            if def.column == column {
                return Ok(if total == UNBOUNDED {
                    None
                } else {
                    Some(total)
                });
            }
        }
        Err(FeatureError::Spec(format!(
            "spec {:?} has no column {column:?}",
            self.name
        )))
    }

    // ── computation ───────────────────────────────────────────────────────────

    /// Build the `(rows, columns)` matrix, in spec column order.
    ///
    /// Every input array must be the same length: ragged inputs mean two columns are
    /// describing different events, and a matrix built from them is misaligned in a
    /// way no later check can detect.
    pub fn compute(
        &self,
        inputs: &BTreeMap<String, Vec<f64>>,
    ) -> Result<FeatureMatrix, FeatureError> {
        let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut n: Option<usize> = None;
        for (key, arr) in inputs {
            match n {
                None => n = Some(arr.len()),
                Some(len) if arr.len() != len => {
                    return Err(FeatureError::Inputs(format!(
                        "input {key:?} has {} rows but earlier inputs have {len}; inputs of \
                         different lengths are not the same events",
                        arr.len()
                    )))
                }
                _ => {}
            }
            if let Some(bad) = arr
                .iter()
                .find(|v| v.is_finite() && v.abs() >= EXACT_FLOAT_LIMIT)
            {
                return Err(FeatureError::Inputs(format!(
                    "input {key:?} holds {bad}, which exceeds 2^53 and cannot be held exactly in \
                     float64; nanosecond timestamps are not features — pass them alongside the \
                     matrix"
                )));
            }
            values.insert(key.clone(), arr.clone());
        }
        let n = match n {
            Some(n) => n,
            None => {
                return Err(FeatureError::Inputs(format!(
                    "spec {:?} needs inputs {:?}",
                    self.name,
                    self.required_inputs()?
                )))
            }
        };

        // If a column could shadow a supplied input, "which one did feature X read?"
        // depends on evaluation order — and the answer changes when a column is
        // inserted, silently re-pointing a downstream feature.
        let collisions: Vec<&str> = self
            .columns()
            .into_iter()
            .filter(|c| values.contains_key(*c))
            .collect();
        if !collisions.is_empty() {
            return Err(FeatureError::Spec(format!(
                "column name(s) {collisions:?} collide with supplied input names"
            )));
        }

        let width = self.features.len();
        let mut data = vec![0.0f64; n * width];
        let mut column = vec![0.0f64; n];
        for (j, def) in self.features.iter().enumerate() {
            let info = feature_info(&def.feature)?;
            let sources = def.sources()?;
            let mut args: Vec<&[f64]> = Vec::with_capacity(sources.len());
            for source in &sources {
                let arr = values.get(source).ok_or_else(|| {
                    FeatureError::Spec(format!(
                        "column {:?} reads {source:?}, which is neither a supplied input {:?} nor \
                         an earlier column",
                        def.column,
                        values.keys().collect::<Vec<_>>()
                    ))
                })?;
                args.push(arr.as_slice());
            }
            (info.eval)(&args, &def.params, &mut column)?;
            for i in 0..n {
                data[i * width + j] = column[i];
            }
            values.insert(def.column.clone(), column.clone());
        }
        FeatureMatrix::new(
            self.columns().into_iter().map(str::to_string).collect(),
            data,
        )
    }
}

/// A computed feature matrix, row-major.
///
/// Row-major because every consumer reads a *row*: a model scores one feature vector,
/// the streaming runtime emits one, and the parity gate compares them row by row. A
/// column-major layout would make the one access pattern anybody has a strided one.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    rows: usize,
    columns: Vec<String>,
    data: Vec<f64>,
}

impl FeatureMatrix {
    pub fn new(columns: Vec<String>, data: Vec<f64>) -> Result<Self, FeatureError> {
        let width = columns.len();
        if width == 0 {
            return Err(FeatureError::Spec(
                "a matrix with no columns describes nothing".into(),
            ));
        }
        if data.len() % width != 0 {
            return Err(FeatureError::Inputs(format!(
                "{} values do not divide into {width} columns",
                data.len()
            )));
        }
        Ok(Self {
            rows: data.len() / width,
            columns,
            data,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn row(&self, i: usize) -> &[f64] {
        let w = self.cols();
        &self.data[i * w..(i + 1) * w]
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols() + col]
    }

    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Rows where every cell is finite, and therefore usable.
    pub fn finite_rows(&self) -> usize {
        (0..self.rows)
            .filter(|i| crate::functions::row_is_finite(self.row(*i)))
            .count()
    }

    /// How many cells are NaN. The bundle manifest records this and both readers
    /// check it: a matrix whose warmup silently became zeros has the same shape and a
    /// different meaning, and nothing else in the format would notice.
    pub fn nan_cells(&self) -> usize {
        self.data.iter().filter(|v| v.is_nan()).count()
    }
}

// ── canonical JSON ────────────────────────────────────────────────────────────

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))`, in Rust.
///
/// Written out rather than configured on `serde_json`, because the fingerprint is
/// taken over these exact bytes and a dependency's default changing under it would
/// move every spec id in the repo — silently on a regeneration, loudly on a load, and
/// in both cases for a reason nobody would find quickly.
fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sorted, because Python sorts. A `serde_json::Map` is a BTreeMap by
            // default, but collecting and sorting the keys makes the guarantee local
            // rather than a property of a feature flag in a sibling crate.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_canonical(&map[*key], out);
            }
            out.push('}');
        }
    }
}

/// A JSON string literal with Python's `ensure_ascii=True` escaping.
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => out.push(c),
            // Non-ASCII travels as an escape, which is what `ensure_ascii=True` does,
            // as a surrogate pair for anything above the BMP — same as Python.
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// 64 bits of SHA-256 over the canonical body — long enough that an accidental
/// collision between two specs in one registry is not a thing that happens, short
/// enough to read in a log line and to sit inside an artifact filename.
fn fingerprint_of(canonical_body: &str) -> String {
    let digest = Sha256::digest(canonical_body.as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn param_to_value(param: &Param) -> Value {
    match param {
        Param::Bool(b) => Value::Bool(*b),
        Param::Int(i) => Value::from(*i),
        Param::Float(f) => Value::from(*f),
        Param::Str(s) => Value::String(s.clone()),
    }
}

fn value_to_param(feature: &str, key: &str, value: &Value) -> Result<Param, FeatureError> {
    match value {
        Value::Bool(b) => Ok(Param::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Param::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Param::Float(f))
            } else {
                Err(FeatureError::Spec(format!(
                    "{feature}.{key} is a number this build cannot represent"
                )))
            }
        }
        Value::String(s) => Ok(Param::Str(s.clone())),
        other => Err(FeatureError::Spec(format!(
            "{feature}.{key} is {other}; spec parameters must be bool/int/float/str so the spec \
             hashes and serializes identically everywhere"
        ))),
    }
}

fn string_field(obj: &serde_json::Map<String, Value>, key: &str) -> Result<String, FeatureError> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| FeatureError::Spec(format!("{key} must be a string")))
}

fn u32_field(obj: &serde_json::Map<String, Value>, key: &str) -> Result<u32, FeatureError> {
    let value = obj
        .get(key)
        .ok_or_else(|| FeatureError::Spec(format!("{key} must be a non-negative integer")))?;
    u32_of(value, key)
}

/// A JSON integer as a `u32`, **refusing** anything that does not fit.
///
/// `as u32` truncates, and truncation here is not a cosmetic bug: `library_version`
/// is what makes a spec from a different build of `axon.features` refuse to load, and
/// `4294967297 as u32` is `1`. A payload carrying that would be refused by Python —
/// which has no fixed integer width — and accepted by Rust as the current build, so
/// the two languages would disagree about whether a spec is loadable at all. The
/// fingerprint check catches it whenever a fingerprint is recorded; `from_json`
/// deliberately permits a payload without one, and that is the hole this closes.
fn u32_of(value: &Value, key: &str) -> Result<u32, FeatureError> {
    let raw = value
        .as_u64()
        .ok_or_else(|| FeatureError::Spec(format!("{key} must be a non-negative integer")))?;
    u32::try_from(raw).map_err(|_| {
        FeatureError::Spec(format!(
            "{key} is {raw}, which does not fit in 32 bits; truncating it would silently \
             read as {}",
            raw as u32
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BAR_M1_V1`, transcribed. Committed as a literal rather than read from a file
    /// so the unit tests need no fixture.
    ///
    /// **It is not byte-identical to what Python writes, and must not be described as
    /// if it were.** `FeatureSpec.to_json()` emits a `fingerprint` field; this literal
    /// is the *body* without it, 577 bytes against Python's 610. What validates the
    /// transcription is therefore indirect and stronger than a byte comparison would
    /// be: `from_json` recomputes the fingerprint from this crate's own canonical
    /// serialization of the parsed recipe, and
    /// `the_fingerprint_matches_the_one_python_computed` below pins the result to
    /// `c503688de24e863f` — the id `tests/fixtures/cross_language.json` carries from
    /// Python. A character wrong anywhere in this literal moves that hash and the test
    /// fails immediately. The byte-for-byte comparison against Python's own bytes is a
    /// different assertion on a different literal:
    /// `tests/cross_language.rs::rusts_canonical_serialization_is_byte_identical_to_pythons`,
    /// over the copy in `tests/feature_parity.rs`, which does carry the fingerprint
    /// field.
    const BAR_M1_JSON: &str = r#"{"features":[{"column":"ret_1","feature":"log_return","inputs":{"price":"close"},"params":{"period":1}},{"column":"mom_5","feature":"momentum","inputs":{"price":"close"},"params":{"window":5}},{"column":"z_20","feature":"rolling_zscore","inputs":{"x":"close"},"params":{"window":20}},{"column":"vol_20","feature":"realized_volatility","inputs":{"price":"close"},"params":{"window":20}},{"column":"range_bps","feature":"relative_range","inputs":{},"params":{}},{"column":"clv","feature":"close_location","inputs":{},"params":{}}],"library_version":1,"spec":"bar_m1","version":1}"#;

    #[test]
    fn the_fingerprint_matches_the_one_python_computed() {
        // The single most load-bearing assertion in this crate. If Rust and Python
        // disagree about what the recipe *is*, every downstream comparison is
        // between two different questions — and nothing else would ever say so.
        let spec = FeatureSpec::from_json(BAR_M1_JSON).unwrap();
        assert_eq!(spec.fingerprint(), "c503688de24e863f");
        assert_eq!(spec.reference(), "bar_m1/v1#c503688de24e863f");
    }

    #[test]
    fn a_spec_edited_after_its_fingerprint_was_taken_is_refused() {
        // Somebody widening a window in a committed spec.json to make a gate pass.
        let with_id = r#"{"features":[{"column":"ret_1","feature":"log_return","inputs":{"price":"close"},"params":{"period":2}}],"fingerprint":"c503688de24e863f","library_version":1,"spec":"bar_m1","version":1}"#;
        assert!(matches!(
            FeatureSpec::from_json(with_id),
            Err(FeatureError::Mismatch(_))
        ));
        // …and the lax reader still gets to report what it wanted.
        assert!(FeatureSpec::from_json_lax(with_id).is_ok());
    }

    #[test]
    fn a_spec_from_a_different_library_build_is_refused_by_name() {
        let other = BAR_M1_JSON.replace(r#""library_version":1"#, r#""library_version":2"#);
        assert_ne!(other, BAR_M1_JSON, "the corruption pattern matched nothing");
        match FeatureSpec::from_json(&other) {
            Err(FeatureError::Mismatch(msg)) => {
                assert!(msg.contains("v2"), "the message must name the build: {msg}")
            }
            other => panic!("expected a Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn the_library_version_is_inside_the_hash_not_beside_it() {
        // Without this the fingerprint would pin the recipe and say nothing about
        // the kitchen: rewriting `rolling_std` leaves every artifact id unchanged.
        let a = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let raw = BAR_M1_JSON.replace(r#""library_version":1"#, r#""library_version":9"#);
        assert_ne!(raw, BAR_M1_JSON, "the corruption pattern matched nothing");
        let b = FeatureSpec::from_json_lax(&raw).unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn column_order_is_inside_the_hash_because_permuting_two_leaves_every_name_right() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let mut swapped = spec.features.clone();
        swapped.swap(0, 1);
        let other = FeatureSpec::new("bar_m1", 1, 1, swapped).unwrap();
        assert_ne!(spec.fingerprint(), other.fingerprint());
    }

    #[test]
    fn an_empty_params_map_is_emitted_rather_than_omitted() {
        // An omitted key and an empty mapping hash differently while meaning the
        // same thing, and Python emits both unconditionally. Dropping them here
        // would make every fingerprint disagree — loudly, but for a reason nobody
        // would find quickly.
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let body = spec.canonical_body();
        assert!(
            body.contains(r#"{"column":"clv","feature":"close_location","inputs":{},"params":{}}"#)
        );
    }

    #[test]
    fn the_canonical_form_round_trips_through_its_own_reader() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let again = FeatureSpec::from_json(&spec.canonical_json()).unwrap();
        assert_eq!(spec, again);
        assert_eq!(spec.fingerprint(), again.fingerprint());
    }

    #[test]
    fn the_fingerprint_key_lands_in_sorted_position_not_appended() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let json = spec.canonical_json();
        let f = json.find(r#""fingerprint""#).unwrap();
        let l = json.find(r#""library_version""#).unwrap();
        let s = json.find(r#""spec""#).unwrap();
        assert!(f < l && l < s, "keys are not in sorted order: {json}");
    }

    #[test]
    fn a_version_too_large_for_thirty_two_bits_is_refused_rather_than_truncated() {
        // `4294967297 as u32` is `1`. Python has no fixed integer width, so it refuses
        // such a payload as "written against a different build"; a truncating Rust
        // would accept it as *this* build. The two languages would then disagree about
        // whether a spec is loadable, which is the class of divergence this whole crate
        // exists to make impossible — and `from_json` permits a payload with no
        // recorded fingerprint, so the identity check does not cover it.
        for key in ["library_version", "version"] {
            let json =
                BAR_M1_JSON.replace(&format!("\"{key}\":1"), &format!("\"{key}\":4294967297"));
            assert_ne!(json, BAR_M1_JSON, "the {key} pattern matched nothing");
            match FeatureSpec::from_json_lax(&json) {
                Err(FeatureError::Spec(msg)) => {
                    assert!(msg.contains("32 bits"), "{key}: {msg}")
                }
                other => panic!("{key}: expected a refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_duplicate_column_is_refused_rather_than_letting_the_second_win() {
        let dup = BAR_M1_JSON.replace(r#""column":"mom_5""#, r#""column":"ret_1""#);
        assert_ne!(dup, BAR_M1_JSON, "the corruption pattern matched nothing");
        assert!(matches!(
            FeatureSpec::from_json_lax(&dup),
            Err(FeatureError::Spec(_))
        ));
    }

    #[test]
    fn a_misspelled_parameter_is_refused_rather_than_silently_defaulted() {
        let typo = BAR_M1_JSON.replace(r#""params":{"window":20}"#, r#""params":{"windwo":20}"#);
        assert_ne!(typo, BAR_M1_JSON, "the corruption pattern matched nothing");
        assert!(matches!(
            FeatureSpec::from_json_lax(&typo),
            Err(FeatureError::Spec(_))
        ));
    }

    #[test]
    fn required_inputs_are_what_no_earlier_column_produces() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        assert_eq!(
            spec.required_inputs().unwrap(),
            vec!["close".to_string(), "high".to_string(), "low".to_string()]
        );
    }

    #[test]
    fn the_bar_spec_warmup_is_derived_as_twenty_one_and_not_restated() {
        // 20 samples of one-step returns need 21 bars. This is
        // `BAR_M1_WARMUP_BARS`, and it is *computed* from the spec here — the whole
        // reason `column_lookback` composes rather than approximating upward.
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        assert_eq!(spec.max_lookback().unwrap(), Some(21));
        assert_eq!(spec.column_lookback("vol_20").unwrap(), Some(21));
        assert_eq!(spec.column_lookback("z_20").unwrap(), Some(20));
        assert_eq!(spec.column_lookback("mom_5").unwrap(), Some(6));
        assert_eq!(spec.column_lookback("ret_1").unwrap(), Some(2));
        assert_eq!(spec.column_lookback("clv").unwrap(), Some(1));
    }

    #[test]
    fn a_lookback_composes_through_a_binding_to_an_earlier_column() {
        // `mom_32` reading `mid` needs 33 observations of `mid`, and `mid` is
        // pointwise, so 33 raw rows.
        let json = r#"{"features":[{"column":"mid","feature":"mid_price","inputs":{},"params":{}},{"column":"mom_32","feature":"momentum","inputs":{"price":"mid"},"params":{"window":32}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(json).unwrap();
        assert_eq!(spec.column_lookback("mom_32").unwrap(), Some(33));

        // And through two rolling stages, where the overlap actually bites: a
        // 5-sample mean of a 4-sample mean spans 8 rows, not 9.
        let two = r#"{"features":[{"column":"a","feature":"rolling_mean","inputs":{"x":"px"},"params":{"window":4}},{"column":"b","feature":"rolling_mean","inputs":{"x":"a"},"params":{"window":5}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(two).unwrap();
        assert_eq!(spec.column_lookback("b").unwrap(), Some(8));
    }

    #[test]
    fn an_ema_makes_the_whole_spec_unbounded_which_is_what_streaming_refuses() {
        // PERP_CORE_V1's shape. This is a finding rather than an edge case: the
        // reference perp spec cannot be served from a bounded Rust buffer.
        let json = r#"{"features":[{"column":"mid","feature":"mid_price","inputs":{},"params":{}},{"column":"ema_x_8_32","feature":"ema_crossover","inputs":{"price":"mid"},"params":{"fast":8,"slow":32}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(json).unwrap();
        assert_eq!(spec.max_lookback().unwrap(), None);
        assert_eq!(spec.column_lookback("ema_x_8_32").unwrap(), None);
        // The bounded column beside it is still bounded; only the spec as a whole
        // is refused.
        assert_eq!(spec.column_lookback("mid").unwrap(), Some(1));
    }

    #[test]
    fn unboundedness_propagates_to_anything_that_reads_it() {
        let json = r#"{"features":[{"column":"e","feature":"ema","inputs":{"x":"px"},"params":{"span":4}},{"column":"m","feature":"rolling_mean","inputs":{"x":"e"},"params":{"window":3}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(json).unwrap();
        assert_eq!(spec.column_lookback("m").unwrap(), None);
    }

    // ── compute ──

    fn bars(n: usize) -> BTreeMap<String, Vec<f64>> {
        let close = crate::numeric::perp_series(n);
        let high: Vec<f64> = close.iter().map(|c| c + 5.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 5.0).collect();
        BTreeMap::from([
            ("close".to_string(), close),
            ("high".to_string(), high),
            ("low".to_string(), low),
        ])
    }

    #[test]
    fn the_matrix_is_in_spec_column_order_not_alphabetical() {
        // Column order is part of the identity; a matrix built in sorted order would
        // leave every name correct and every prediction wrong.
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let m = spec.compute(&bars(40)).unwrap();
        assert_eq!(
            m.columns(),
            ["ret_1", "mom_5", "z_20", "vol_20", "range_bps", "clv"]
        );
        assert_eq!(m.rows(), 40);
        assert_eq!(m.cols(), 6);
    }

    #[test]
    fn the_first_fully_finite_row_is_the_derived_warmup_minus_one() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let m = spec.compute(&bars(40)).unwrap();
        let warmup = spec.max_lookback().unwrap().unwrap();
        for i in 0..warmup - 1 {
            assert!(
                !crate::functions::row_is_finite(m.row(i)),
                "row {i} was finite before the derived warmup of {warmup}"
            );
        }
        assert!(
            crate::functions::row_is_finite(m.row(warmup - 1)),
            "row {} was not finite at the derived warmup",
            warmup - 1
        );
    }

    #[test]
    fn a_column_that_shadows_an_input_is_refused_because_order_would_decide_it() {
        let json = r#"{"features":[{"column":"close","feature":"rolling_mean","inputs":{"x":"close"},"params":{"window":3}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(json).unwrap();
        assert!(matches!(
            spec.compute(&bars(10)),
            Err(FeatureError::Spec(_))
        ));
    }

    #[test]
    fn a_nanosecond_timestamp_fed_in_as_a_feature_is_refused() {
        // The one mistake this check exists for. A 2026 nanosecond stamp needs 61
        // mantissa bits; routed through float64 it rounds into ~256 ns buckets and
        // reorders events.
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let mut inputs = bars(30);
        inputs.insert("close".into(), vec![1.7e18; 30]);
        assert!(matches!(
            spec.compute(&inputs),
            Err(FeatureError::Inputs(_))
        ));
    }

    #[test]
    fn ragged_inputs_are_refused_before_a_single_column_is_computed() {
        let spec = FeatureSpec::from_json_lax(BAR_M1_JSON).unwrap();
        let mut inputs = bars(30);
        inputs.get_mut("high").unwrap().pop();
        assert!(matches!(
            spec.compute(&inputs),
            Err(FeatureError::Inputs(_))
        ));
    }

    #[test]
    fn a_column_reading_a_name_nobody_supplies_says_what_it_looked_for() {
        let json = r#"{"features":[{"column":"a","feature":"rolling_mean","inputs":{"x":"nope"},"params":{"window":3}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json_lax(json).unwrap();
        match spec.compute(&bars(10)) {
            Err(FeatureError::Spec(msg)) => assert!(msg.contains("nope"), "{msg}"),
            other => panic!("expected a Spec error naming the source, got {other:?}"),
        }
    }
}
