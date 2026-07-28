//! [`TreeModel`] — XGBoost served natively, by deterministic threshold traversal.
//!
//! ADR-0003 calls tree ensembles "~numerically exact" in Rust. That is a claim
//! this module has to *earn*: exactness is the deliverable, and every design
//! choice below exists because the obvious alternative is off by a hair.
//!
//! The three places a hand-rolled reader silently diverges from XGBoost:
//!
//! 1. **The missing-value branch.** A split has a *default direction* taken when
//!    the feature is absent. `NaN < threshold` is false in IEEE-754, so the
//!    natural `if v < t { left } else { right }` sends every missing value right
//!    — correct for half the nodes by luck, wrong for the other half, and wrong
//!    only on the rows that actually have missing data. [`Node::step`] tests for
//!    NaN *before* comparing.
//! 2. **The threshold dtype.** XGBoost compares in `f32`. Widening either side
//!    to `f64` moves a value that sits exactly on a threshold to the other
//!    branch, which is a different leaf and a different trade.
//! 3. **The intercept's link.** For a non-identity link XGBoost stores
//!    `base_score` in *prediction* space, not margin space, and converts with
//!    `f32` arithmetic. Reading it as a margin shifts every prediction by a
//!    constant; converting it in `f64` shifts every prediction by one ULP. See
//!    [`TreeLink::to_margin`].
//!
//! Everything the reader cannot reproduce exactly is refused **by name** at
//! load rather than approximated, and the refusals are asked of real artifacts
//! in `tests/model_family_refusals.rs` rather than of hand-edited JSON. That
//! distinction has already paid for itself once: xgboost 3.3.0 writes
//! `"name": "gbtree"` on a **dart** booster and hangs the dropout factors off a
//! sibling `weight_drop` array, so the check that read the name had stopped
//! catching dart while its unit test — which edited the name by hand — went on
//! passing. See the `weight_drop` refusal in [`TreeModel::from_bytes`].
//!
//! **This backend returns the raw margin** — the equivalent of
//! `Booster.predict(..., output_margin=True)` — and never applies the link. The
//! link functions XGBoost uses (logistic, softmax) are monotone, so a decision
//! threshold on the probability is exactly a decision threshold on the margin:
//! skipping the link costs no decision fidelity and removes `expf` (a libm call
//! whose last bit is not portable) from the serving path entirely.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{InferenceError, LoadError, Model};

/// Learner attribute the artifact version is read from. XGBoost's JSON has no
/// first-class version field, but `Booster.set_attr` round-trips through
/// save/load, so `axon.models` stamps the version there on export.
const VERSION_ATTR: &str = "axon_model_version";

/// Objectives whose `base_score` is already a margin. Anything outside this
/// list and [`LOGIT_OBJECTIVES`] is refused by name rather than guessed at,
/// because guessing the link wrong is a constant offset on every prediction
/// that no amount of downstream testing would attribute to the model loader.
const IDENTITY_OBJECTIVES: &[&str] = &[
    "reg:squarederror",
    "reg:squaredlogerror",
    "reg:linear",
    "reg:absoluteerror",
    "reg:pseudohubererror",
    "reg:quantileerror",
    "binary:logitraw",
    "rank:pairwise",
    "rank:ndcg",
    "rank:map",
];

/// Objectives whose `base_score` is a probability and must be pushed through
/// `logit` to become a margin.
const LOGIT_OBJECTIVES: &[&str] = &["binary:logistic", "reg:logistic"];

/// How XGBoost's stored `base_score` relates to the margin the trees add to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeLink {
    /// `base_score` is already in margin space.
    Identity,
    /// `base_score` is a probability; the margin is its logit.
    Logit,
}

impl TreeLink {
    /// Convert a stored `base_score` into the margin-space intercept.
    ///
    /// The `f32` arithmetic is not sloppiness, it is the specification: XGBoost
    /// computes `-logf(1.0f / base_score - 1.0f)` in single precision, and the
    /// *more accurate* `f64` form disagrees with it by one ULP for roughly half
    /// of all `base_score` values — a one-ULP shift applied to every prediction
    /// the model will ever make. `tests/parity.rs` pins this against a fitted
    /// intercept rather than the trivial 0.5 case, which would hide the bug.
    ///
    /// This is also the one place the tree path's exactness leans on the
    /// platform: `f32::ln` lowers to the system `logf`. It agrees with the
    /// reference on the Linux CI target (ADR-0019); a platform whose `logf`
    /// rounds differently would move every logistic margin by one ULP.
    fn to_margin(self, base_score: f32) -> Result<f32, LoadError> {
        match self {
            TreeLink::Identity => Ok(base_score),
            TreeLink::Logit => {
                if !(base_score > 0.0 && base_score < 1.0) {
                    return Err(LoadError::Malformed(format!(
                        "logit-link base_score must lie in (0, 1), got {base_score}"
                    )));
                }
                Ok(-(1.0f32 / base_score - 1.0f32).ln())
            }
        }
    }
}

/// One node of one tree, flattened. XGBoost's JSON stores the fields as parallel
/// arrays; they are interleaved here so a traversal step touches one cache line
/// instead of four.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// Child indices, or `-1` on both when this is a leaf.
    left: i32,
    right: i32,
    feature: u32,
    /// The split threshold on an internal node; the leaf weight on a leaf.
    /// XGBoost overloads one array for both, and so does this.
    value: f32,
    /// Direction taken when the feature is missing.
    default_left: bool,
}

impl Node {
    /// The child to descend to. Returns `None` on a leaf.
    #[inline]
    fn step(&self, features: &[f32]) -> Option<usize> {
        if self.left < 0 {
            return None;
        }
        let v = features[self.feature as usize];
        // NaN first: `NaN < threshold` is false, so folding this into the
        // comparison below would silently route every missing value right and
        // ignore `default_left` on half the nodes.
        let next = if v.is_nan() {
            if self.default_left {
                self.left
            } else {
                self.right
            }
        } else if v < self.value {
            self.left
        } else {
            self.right
        };
        Some(next as usize)
    }
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// The leaf weight this feature vector lands on.
    ///
    /// The loop is unbounded by construction; [`TreeModel::from_bytes`] proves
    /// termination by checking that every child index is greater than its
    /// parent's, so a hand-edited artifact cannot park the core thread in an
    /// infinite descent.
    #[inline]
    fn leaf(&self, features: &[f32]) -> f32 {
        let mut node = &self.nodes[0];
        while let Some(next) = node.step(features) {
            node = &self.nodes[next];
        }
        node.value
    }
}

/// An XGBoost gradient-boosted tree ensemble, served from its native JSON.
///
/// Load with [`TreeModel::from_path`]. [`Model::predict`] returns a single
/// margin; see the module docs for why it is not a probability.
#[derive(Debug, Clone)]
pub struct TreeModel {
    version: u32,
    num_feature: usize,
    /// `base_score` already converted into margin space.
    intercept: f32,
    objective: String,
    link: TreeLink,
    trees: Vec<Tree>,
}

impl TreeModel {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| LoadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        // serde_json parses a JSON number into f64 and narrows to f32. That is
        // safe here rather than lucky: XGBoost writes the shortest decimal that
        // round-trips an f32, and f64 carries 53 bits against the 50 that the
        // double-rounding bound needs for f32, so decimal → f64 → f32 lands on
        // the original bits. A writer that emitted long decimals instead would
        // make this the first place exactness leaked.
        let artifact: Artifact =
            serde_json::from_slice(bytes).map_err(|e| LoadError::Parse(e.to_string()))?;
        let learner = artifact.learner;

        if learner.gradient_booster.name != "gbtree" {
            // `gblinear` still names itself here. `dart` no longer does — see
            // the `weight_drop` check below, which is the one that catches it.
            return Err(LoadError::Unsupported(format!(
                "booster '{}'; only 'gbtree' is served natively",
                learner.gradient_booster.name
            )));
        }
        if !learner.gradient_booster.weight_drop.is_empty() {
            // A dart booster does not identify itself by name: xgboost 3.3.0
            // writes `"name": "gbtree"` and hangs the dropout factors off a
            // sibling `weight_drop` array, and 3.3.0 additionally deprecates
            // `booster="dart"` in favour of passing `rate_drop` to a plain
            // gbtree — so a dart ensemble is reachable without anyone typing
            // the word. The name check above therefore waved it straight
            // through, this loop summed the leaves unweighted, and the answer
            // came back plausible and wrong by the dropout factor (measured on
            // the committed fixture: 0.390 became 0.739). `weight_drop` is the
            // only field that is *present* on a dart artifact and absent on
            // every other, which is why the presence, not the values, is the
            // test: an ensemble whose weights all happen to be 1.0 serves
            // correctly by coincidence, and a coincidence is not a contract.
            return Err(LoadError::Unsupported(format!(
                "booster carries {} dropout weights (weight_drop), so its trees are scaled at \
                 predict time; this backend sums them unweighted and would be wrong by that \
                 factor. Re-export without dropout (no rate_drop/one_drop, no booster='dart')",
                learner.gradient_booster.weight_drop.len()
            )));
        }
        if let Some(best) = learner.attributes.get("best_iteration") {
            // With early stopping, `Booster.predict` truncates the ensemble at
            // `best_iteration`; this backend sums every tree, so serving such an
            // artifact would quietly include the overfitted tail.
            return Err(LoadError::Unsupported(format!(
                "artifact was early-stopped at iteration {best}; re-export the truncated \
                 ensemble so the served trees are the evaluated ones"
            )));
        }

        let version = learner
            .attributes
            .get(VERSION_ATTR)
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or(LoadError::MissingVersion { slot: VERSION_ATTR })?;

        let param = &learner.learner_model_param;
        let num_feature = parse_param::<usize>(&param.num_feature, "num_feature")?;
        let num_class = parse_param::<u32>(&param.num_class, "num_class")?;
        let num_target = match &param.num_target {
            Some(t) => parse_param::<u32>(t, "num_target")?,
            None => 1,
        };
        if num_class > 1 || num_target > 1 {
            return Err(LoadError::Unsupported(format!(
                "multi-output ensemble (num_class={num_class}, num_target={num_target}); this \
                 backend serves a single margin"
            )));
        }

        let objective = learner.objective.name;
        let link = if IDENTITY_OBJECTIVES.contains(&objective.as_str()) {
            TreeLink::Identity
        } else if LOGIT_OBJECTIVES.contains(&objective.as_str()) {
            TreeLink::Logit
        } else {
            return Err(LoadError::Unsupported(format!(
                "objective '{objective}': its base_score→margin link is not one this backend \
                 can reproduce exactly"
            )));
        };
        let intercept = link.to_margin(parse_base_score(&param.base_score)?)?;

        let model = learner.gradient_booster.model.ok_or_else(|| {
            LoadError::Malformed("gbtree booster has no 'model' object".to_string())
        })?;
        if model.tree_info.iter().any(|g| *g != 0) {
            return Err(LoadError::Unsupported(
                "trees are assigned to more than one output group".to_string(),
            ));
        }
        let trees = model
            .trees
            .iter()
            .enumerate()
            .map(|(i, t)| build_tree(i, t, num_feature))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            version,
            num_feature,
            intercept,
            objective,
            link,
            trees,
        })
    }

    /// The XGBoost objective the artifact was trained with. Exposed so a caller
    /// can assert the model it loaded is the family it expected.
    pub fn objective(&self) -> &str {
        &self.objective
    }

    /// How `base_score` was interpreted. See [`TreeLink`].
    pub fn link(&self) -> TreeLink {
        self.link
    }

    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }
}

impl Model for TreeModel {
    fn version(&self) -> u32 {
        self.version
    }

    fn input_len(&self) -> usize {
        self.num_feature
    }

    fn output_len(&self) -> usize {
        1
    }

    fn predict_into(&self, features: &[f32], out: &mut [f32]) -> Result<(), InferenceError> {
        crate::check_shapes(features, out, self.num_feature, 1)?;
        // f32 accumulator, in tree order, one leaf at a time — mirroring
        // XGBoost's own `out_preds[i] += leaf` loop. An f64 accumulator would be
        // *more* accurate and therefore *wrong*: the deliverable is the same
        // bits Python produced, not the best available answer. Rust never
        // reassociates float arithmetic, so source order fixes the rounding.
        let mut margin = self.intercept;
        for tree in &self.trees {
            margin += tree.leaf(features);
        }
        out[0] = margin;
        Ok(())
    }
}

fn build_tree(index: usize, t: &TreeJson, num_feature: usize) -> Result<Tree, LoadError> {
    let n = t.left_children.len();
    let malformed = |what: &str| LoadError::Malformed(format!("tree {index}: {what}"));
    if t.right_children.len() != n
        || t.split_indices.len() != n
        || t.split_conditions.len() != n
        || t.default_left.len() != n
    {
        return Err(malformed("node arrays have inconsistent lengths"));
    }
    if t.split_type.iter().any(|k| *k != 0) {
        // A categorical split matches against a bitset, not a threshold. There
        // is no correct threshold answer to fall back on, so refuse.
        return Err(LoadError::Unsupported(format!(
            "tree {index} contains a categorical split; only numerical splits are served"
        )));
    }
    if let Some(tp) = &t.tree_param {
        let size_leaf_vector = parse_param::<u32>(&tp.size_leaf_vector, "size_leaf_vector")?;
        if size_leaf_vector > 1 {
            return Err(LoadError::Unsupported(format!(
                "tree {index} has vector leaves (size_leaf_vector={size_leaf_vector})"
            )));
        }
    }

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let (left, right) = (t.left_children[i], t.right_children[i]);
        let is_leaf = left < 0 && right < 0;
        if !is_leaf {
            // Children strictly after their parent is how XGBoost writes the
            // arrays, and checking it here is what makes `Tree::leaf`'s
            // unbounded descent provably terminating.
            let ok = |c: i32| c > i as i32 && (c as usize) < n;
            if !ok(left) || !ok(right) {
                return Err(malformed(&format!(
                    "node {i} has children ({left}, {right}) that are not later, in-range nodes"
                )));
            }
            if t.split_indices[i] as usize >= num_feature {
                return Err(malformed(&format!(
                    "node {i} splits on feature {} but the model declares {num_feature}",
                    t.split_indices[i]
                )));
            }
        }
        nodes.push(Node {
            left: if is_leaf { -1 } else { left },
            right: if is_leaf { -1 } else { right },
            feature: t.split_indices[i],
            value: t.split_conditions[i],
            default_left: t.default_left[i] != 0,
        });
    }
    if nodes.is_empty() {
        return Err(malformed("tree has no nodes"));
    }
    Ok(Tree { nodes })
}

/// XGBoost writes `learner_model_param` values as strings, and `base_score` as
/// a one-element vector literal (`"[1.4226291E-2]"`) since 2.0 — older exports
/// wrote a bare scalar. Both are accepted; a genuinely multi-valued intercept
/// is a multi-output model and is rejected upstream.
fn parse_base_score(raw: &str) -> Result<f32, LoadError> {
    let trimmed = raw.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.contains(',') {
        return Err(LoadError::Unsupported(format!(
            "base_score '{raw}' has more than one component (multi-output model)"
        )));
    }
    trimmed
        .trim()
        .parse::<f32>()
        .map_err(|e| LoadError::Malformed(format!("base_score '{raw}': {e}")))
}

fn parse_param<T: std::str::FromStr>(raw: &str, what: &str) -> Result<T, LoadError>
where
    T::Err: std::fmt::Display,
{
    raw.trim()
        .parse::<T>()
        .map_err(|e| LoadError::Malformed(format!("{what} '{raw}': {e}")))
}

// ── the subset of XGBoost's JSON this backend reads ──
//
// Unknown fields are ignored on purpose: the format grows between XGBoost
// releases, and the fields that matter are validated explicitly above. Failing
// on an unrecognised key would break on an upgrade that changed nothing we use.

#[derive(Deserialize)]
struct Artifact {
    learner: Learner,
}

#[derive(Deserialize)]
struct Learner {
    #[serde(default)]
    attributes: BTreeMap<String, String>,
    gradient_booster: GradientBooster,
    learner_model_param: LearnerModelParam,
    objective: Objective,
}

#[derive(Deserialize)]
struct GradientBooster {
    name: String,
    #[serde(default)]
    model: Option<BoosterModel>,
    /// Per-tree dropout scale factors. Present only on a dart booster, and the
    /// only thing in the artifact that says so — see the refusal in
    /// [`TreeModel::from_bytes`].
    #[serde(default)]
    weight_drop: Vec<f32>,
}

#[derive(Deserialize)]
struct BoosterModel {
    trees: Vec<TreeJson>,
    #[serde(default)]
    tree_info: Vec<i32>,
}

#[derive(Deserialize)]
struct LearnerModelParam {
    base_score: String,
    num_feature: String,
    num_class: String,
    #[serde(default)]
    num_target: Option<String>,
}

#[derive(Deserialize)]
struct Objective {
    name: String,
}

#[derive(Deserialize)]
struct TreeJson {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<u32>,
    split_conditions: Vec<f32>,
    default_left: Vec<u8>,
    #[serde(default)]
    split_type: Vec<u8>,
    #[serde(default)]
    tree_param: Option<TreeParam>,
}

#[derive(Deserialize)]
struct TreeParam {
    size_leaf_vector: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-split stump: `f0 < 1.5` → -1.0, else +1.0, with the default
    /// direction under the caller's control.
    fn stump(default_left: bool, objective: &str, base_score: &str) -> String {
        format!(
            r#"{{
              "learner": {{
                "attributes": {{ "axon_model_version": "42" }},
                "gradient_booster": {{
                  "name": "gbtree",
                  "model": {{
                    "tree_info": [0],
                    "trees": [{{
                      "left_children": [1, -1, -1],
                      "right_children": [2, -1, -1],
                      "split_indices": [0, 0, 0],
                      "split_conditions": [1.5, -1.0, 1.0],
                      "default_left": [{}, 0, 0],
                      "split_type": [0, 0, 0],
                      "tree_param": {{ "size_leaf_vector": "1" }}
                    }}]
                  }}
                }},
                "learner_model_param": {{
                  "base_score": "{base_score}", "num_feature": "1",
                  "num_class": "0", "num_target": "1"
                }},
                "objective": {{ "name": "{objective}" }}
              }}
            }}"#,
            u8::from(default_left)
        )
    }

    fn identity_stump(default_left: bool) -> TreeModel {
        TreeModel::from_bytes(stump(default_left, "reg:squarederror", "[0]").as_bytes()).unwrap()
    }

    #[test]
    fn missing_feature_follows_the_default_direction_not_the_comparison() {
        // The whole point: `NaN < 1.5` is false, so a reader that compares first
        // sends every missing value right and agrees with XGBoost only on the
        // nodes whose default happens to be right.
        assert_eq!(
            identity_stump(true).predict(&[f32::NAN]).unwrap(),
            vec![-1.0]
        );
        assert_eq!(
            identity_stump(false).predict(&[f32::NAN]).unwrap(),
            vec![1.0]
        );
        // A present value ignores the default direction entirely.
        assert_eq!(identity_stump(true).predict(&[9.0]).unwrap(), vec![1.0]);
        assert_eq!(identity_stump(false).predict(&[0.0]).unwrap(), vec![-1.0]);
    }

    #[test]
    fn a_value_exactly_on_the_threshold_goes_right() {
        // XGBoost's test is `value < threshold`, not `<=`. Getting this backwards
        // only shows up on the rows that sit exactly on a split, which is rare in
        // continuous features and common in rounded ones (tick sizes, counts).
        assert_eq!(identity_stump(true).predict(&[1.5]).unwrap(), vec![1.0]);
        let just_below = f32::from_bits(1.5f32.to_bits() - 1);
        assert_eq!(
            identity_stump(true).predict(&[just_below]).unwrap(),
            vec![-1.0]
        );
    }

    #[test]
    fn logit_link_intercept_uses_float32_arithmetic() {
        // base_score = 0.515 is what XGBoost fits for a mildly imbalanced binary
        // problem. The f64 route rounds to a different f32, so this pins the
        // single-precision expression rather than "the most accurate logit".
        let m =
            TreeModel::from_bytes(stump(true, "binary:logistic", "[5.15E-1]").as_bytes()).unwrap();
        assert_eq!(m.link(), TreeLink::Logit);
        let expected_f32 = -(1.0f32 / 0.515f32 - 1.0f32).ln();
        let via_f64 = (0.515f64 / (1.0 - 0.515f64)).ln() as f32;
        assert_ne!(
            expected_f32.to_bits(),
            via_f64.to_bits(),
            "fixture no longer distinguishes the f32 and f64 links; pick another base_score"
        );
        // Leaf +1.0 on the right branch, so the output is intercept + 1.
        let got = m.predict(&[9.0]).unwrap()[0];
        assert_eq!(got.to_bits(), (expected_f32 + 1.0f32).to_bits());
    }

    #[test]
    fn identity_link_leaves_base_score_alone() {
        let m = TreeModel::from_bytes(stump(true, "reg:squarederror", "[2.5]").as_bytes()).unwrap();
        assert_eq!(m.link(), TreeLink::Identity);
        assert_eq!(m.predict(&[9.0]).unwrap(), vec![3.5]);
    }

    #[test]
    fn artifact_without_a_version_is_refused() {
        let json = stump(true, "reg:squarederror", "[0]").replace(VERSION_ATTR, "unrelated_attr");
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::MissingVersion { .. })
        ));
    }

    #[test]
    fn unreproducible_objective_is_refused_rather_than_guessed() {
        // count:poisson has a log link whose f32 expression we have not pinned.
        // Serving it would offset every prediction by a constant.
        let json = stump(true, "count:poisson", "[1.5]");
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::Unsupported(_))
        ));
    }

    #[test]
    fn a_booster_that_names_itself_something_other_than_gbtree_is_refused() {
        // Still live for `gblinear`, which does name itself. It is *not* what
        // catches dart: xgboost 3.3.0 writes `"name": "gbtree"` on a dart
        // artifact, so this assertion passed for a year against a spelling the
        // library had stopped emitting. See the test below and
        // `tests/model_family_refusals.rs`, which asks a real dart artifact.
        for name in ["\"dart\"", "\"gblinear\""] {
            let json = stump(true, "reg:squarederror", "[0]").replace("\"gbtree\"", name);
            assert!(matches!(
                TreeModel::from_bytes(json.as_bytes()),
                Err(LoadError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn dropout_weights_are_refused_even_though_the_booster_calls_itself_gbtree() {
        // The shape a real dart export has: name "gbtree", trees that look
        // ordinary, and a `weight_drop` array beside them that `Booster.predict`
        // multiplies each tree by. Summing the leaves unweighted returns a
        // number in the right units and the wrong magnitude — nothing
        // downstream can tell it apart from a model that simply disagrees.
        let json = stump(true, "reg:squarederror", "[0]").replace(
            "\"name\": \"gbtree\",",
            "\"name\": \"gbtree\", \"weight_drop\": [0.5],",
        );
        let err = TreeModel::from_bytes(json.as_bytes()).unwrap_err();
        assert!(
            matches!(&err, LoadError::Unsupported(m) if m.contains("weight_drop")),
            "expected a dropout refusal, got {err}"
        );
    }

    #[test]
    fn early_stopped_artifact_is_refused_because_we_would_serve_the_overfit_tail() {
        let json = stump(true, "reg:squarederror", "[0]").replace(
            "\"axon_model_version\": \"42\"",
            "\"axon_model_version\": \"42\", \"best_iteration\": \"3\"",
        );
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::Unsupported(_))
        ));
    }

    #[test]
    fn cyclic_child_indices_are_rejected_at_load_not_hit_at_predict() {
        // A child pointing back at its parent would spin `Tree::leaf` forever on
        // the core thread — a hang, not a crash, so nothing would ever alert.
        let json = stump(true, "reg:squarederror", "[0]")
            .replace("\"left_children\": [1,", "\"left_children\": [0,");
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::Malformed(_))
        ));
    }

    #[test]
    fn split_on_an_out_of_range_feature_is_rejected_at_load() {
        let json = stump(true, "reg:squarederror", "[0]")
            .replace("\"split_indices\": [0,", "\"split_indices\": [3,");
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::Malformed(_))
        ));
    }

    #[test]
    fn categorical_split_is_refused_rather_than_treated_as_a_threshold() {
        let json = stump(true, "reg:squarederror", "[0]")
            .replace("\"split_type\": [0,", "\"split_type\": [1,");
        assert!(matches!(
            TreeModel::from_bytes(json.as_bytes()),
            Err(LoadError::Unsupported(_))
        ));
    }

    #[test]
    fn wrong_feature_count_is_an_error_not_a_silent_truncation() {
        let m = identity_stump(true);
        assert!(matches!(
            m.predict(&[]),
            Err(InferenceError::FeatureCount {
                expected: 1,
                got: 0
            })
        ));
    }
}
