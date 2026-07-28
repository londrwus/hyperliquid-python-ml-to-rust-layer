//! Name → transform, with the call shape pinned.
//!
//! Mirrors `axon.features.registry`, and exists for the same reason: a spec that
//! lives inside a model artifact refers to transforms by **name**, not by function
//! pointer, so a process that has never seen the training script can resolve it
//! months later.
//!
//! The registration pins the *shape* of the call as well as the name — which arrays
//! a feature consumes, in order, and which knobs it accepts. Python derives both
//! from the function signature by introspection; Rust has no introspection at
//! runtime, so they are declared here and
//! `tests/cross_language.rs::both_languages_register_the_same_transforms_with_the_same_call_shape`
//! asserts the declaration matches Python's, table against table. A registry that can
//! disagree with the code it indexes is worse than no registry: a spec would
//! validate against the declaration and then call something else, binding `price` to
//! one column while the transform reads another — a silent feature swap no
//! downstream test can tell from a bad model.

use std::collections::BTreeMap;

use crate::functions;
use crate::FeatureError;

/// A spec parameter, restricted to the JSON scalars a fingerprint can hold.
///
/// The same alphabet `axon.features.spec._canonical_param` enforces: bool, int,
/// float, str and nothing else. Anything richer would serialize differently on
/// different machines and the spec's identity would stop being stable — which is
/// the one property the fingerprint exists to provide.
///
/// `Int` and `Float` are separate arms rather than one `f64`, because they
/// serialize differently in canonical JSON (`20` against `20.0`) and the
/// fingerprint is taken over those bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

/// A transform's parameters, in sorted key order.
///
/// `BTreeMap` rather than `HashMap` so iteration order is the canonical one — the
/// order the fingerprint is taken in. A hash map here would make the spec's identity
/// depend on the process's hash seed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params(BTreeMap<String, Param>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Param) {
        self.0.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&Param> {
        self.0.get(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Param)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// A window-shaped parameter as a `usize`, with the transform's own default.
    ///
    /// Refuses a bool outright, exactly as `_check_window` does: `True` is an `int`
    /// in Python and would read as a window of one, which is a legal window and a
    /// silent misconfiguration. Refuses a float for the matching reason on this
    /// side — a spec that says `window: 20.0` fingerprints differently from one that
    /// says `20`, so accepting both would let two distinct specs compute the same
    /// numbers under different identities.
    pub fn window(
        &self,
        feature: &str,
        key: &str,
        default: i64,
        minimum: i64,
    ) -> Result<usize, FeatureError> {
        let raw = match self.0.get(key) {
            None => default,
            Some(Param::Int(v)) => *v,
            Some(other) => {
                return Err(FeatureError::Param {
                    feature: feature.to_string(),
                    message: format!(
                        "{key} must be an int, got {other:?}; a spec parameter that is not the \
                         type the transform reads fingerprints as a different recipe"
                    ),
                })
            }
        };
        if raw < minimum {
            return Err(FeatureError::Param {
                feature: feature.to_string(),
                message: format!("{key} must be >= {minimum}, got {raw}"),
            });
        }
        Ok(raw as usize)
    }

    /// A window-shaped parameter with **no** default.
    ///
    /// `window`, `span`, `fast` and `slow` are keyword-only arguments with no
    /// default on the Python side, so omitting one is a `TypeError` at call time
    /// rather than a quiet fallback. Reproduced here rather than defaulted to
    /// something plausible: a spec that forgot to say `window` and got 1 would
    /// compute a rolling mean of the current value — a column that is numerically
    /// fine, trains fine, and is not the feature anybody asked for.
    pub fn required_window(
        &self,
        feature: &str,
        key: &str,
        minimum: i64,
    ) -> Result<usize, FeatureError> {
        if !self.0.contains_key(key) {
            return Err(FeatureError::Param {
                feature: feature.to_string(),
                message: format!(
                    "{key} is required and the spec does not set it; it has no default, because \
                     a plausible one would silently compute a different feature"
                ),
            });
        }
        self.window(feature, key, minimum, minimum)
    }
}

impl FromIterator<(String, Param)> for Params {
    fn from_iter<T: IntoIterator<Item = (String, Param)>>(iter: T) -> Self {
        Params(iter.into_iter().collect())
    }
}

/// Compute one column into `out`.
///
/// `inputs` are the transform's positional arrays in declaration order, already
/// resolved through the spec's bindings. `out` is exactly as long as every input and
/// must be written in full: **length-preserving** is the first of the three rules
/// (`docs/03`), and a transform that wrote a prefix would leave the tail holding
/// whatever the buffer had, which reads as data rather than as a bug.
pub type EvalFn =
    fn(inputs: &[&[f64]], params: &Params, out: &mut [f64]) -> Result<(), FeatureError>;

/// How many trailing observations of each input this transform needs to produce its
/// last value — or `None` when the answer is "all of them".
///
/// `None` is not a failure and not an unknown: it is the honest answer for an EMA,
/// whose level depends on every observation it has ever seen. It is what
/// [`crate::streaming::FeatureStream`] refuses a spec on, because a bounded serving
/// buffer cannot reproduce an unbounded statistic and the gap is widest right after
/// a restart — the moment nobody is watching the feature values.
pub type LookbackFn = fn(params: &Params) -> Result<Option<usize>, FeatureError>;

/// One registered transform: its name, its call shape, and how to run it.
pub struct FeatureInfo {
    pub name: &'static str,
    /// Positional array arguments, in order. These are the binding names a spec uses.
    pub inputs: &'static [&'static str],
    /// Keyword-only tunables, sorted. A spec naming anything outside this list is
    /// refused rather than having the extra key ignored — silently dropping a
    /// misspelled window is how a spec claims a 32-sample volatility and delivers
    /// the default one, with nothing to show for it.
    pub params: &'static [&'static str],
    pub eval: EvalFn,
    pub lookback: LookbackFn,
    /// Whether this transform's value passes through a **logarithm**.
    ///
    /// The one operation in this library whose cross-language agreement is a
    /// *measurement* rather than a guarantee: IEEE-754 requires `+ - * /` and `sqrt`
    /// to be correctly rounded and says nothing about `log`, so NumPy and this
    /// host's libm agree because both compute it well, not because either must
    /// (0 ULP over the 32 ratios the cross-language fixture pins, and over the
    /// 200 000-sample sweep `scripts/modal_libm_probe.py` re-runs on demand).
    ///
    /// It is a flag on the transform rather than a list of columns because the
    /// dependency is inherited through a **binding**: a z-score of a log return
    /// never calls `log` and is nonetheless a function of one, so the answer for a
    /// *column* has to be walked out of the spec graph
    /// ([`crate::spec::FeatureSpec::libm_columns`]) and the answer for a
    /// *transform* is the only part that can be stated here.
    ///
    /// This exists so the feature bundle's `libm_columns` is a **checked**
    /// cross-language property instead of a field Rust reads and believes. Python
    /// derives it from its own table and refuses a manifest that disagrees; without
    /// this, Rust could only confirm the names were real columns — which is exactly
    /// the shape of trust the fingerprint check exists to refuse everywhere else.
    pub reaches_libm: bool,
}

impl std::fmt::Debug for FeatureInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureInfo")
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

/// The registry, in name order.
///
/// A `static` array rather than a lazily-built map: seventeen linear comparisons at
/// spec-load time cost nothing, and a table that cannot be mutated at runtime cannot
/// be extended by one call site in a way the cross-language test never sees.
static REGISTRY: &[FeatureInfo] = &[
    FeatureInfo {
        name: "book_imbalance",
        inputs: &["bid_sz", "ask_sz"],
        params: &[],
        eval: functions::eval_book_imbalance,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "close_location",
        inputs: &["high", "low", "close"],
        params: &[],
        eval: functions::eval_close_location,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "ema",
        inputs: &["x"],
        params: &["span"],
        eval: functions::eval_ema,
        lookback: functions::lookback_unbounded,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "ema_crossover",
        inputs: &["price"],
        params: &["fast", "slow"],
        eval: functions::eval_ema_crossover,
        lookback: functions::lookback_unbounded,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "log_return",
        inputs: &["price"],
        params: &["period"],
        eval: functions::eval_log_return,
        lookback: functions::lookback_log_return,
        reaches_libm: true,
    },
    FeatureInfo {
        name: "mid_price",
        inputs: &["bid_px", "ask_px"],
        params: &[],
        eval: functions::eval_mid_price,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "momentum",
        inputs: &["price"],
        params: &["window"],
        eval: functions::eval_momentum,
        lookback: functions::lookback_momentum,
        reaches_libm: true,
    },
    FeatureInfo {
        name: "realized_volatility",
        inputs: &["price"],
        params: &["window"],
        eval: functions::eval_realized_volatility,
        lookback: functions::lookback_realized_volatility,
        reaches_libm: true,
    },
    FeatureInfo {
        name: "relative_range",
        inputs: &["high", "low", "close"],
        params: &[],
        eval: functions::eval_relative_range,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "relative_spread",
        inputs: &["bid_px", "ask_px"],
        params: &[],
        eval: functions::eval_relative_spread,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "rolling_mean",
        inputs: &["x"],
        params: &["window"],
        eval: functions::eval_rolling_mean,
        lookback: functions::lookback_window,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "rolling_std",
        inputs: &["x"],
        params: &["ddof", "window"],
        eval: functions::eval_rolling_std,
        lookback: functions::lookback_window,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "rolling_sum",
        inputs: &["x"],
        params: &["window"],
        eval: functions::eval_rolling_sum,
        lookback: functions::lookback_window,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "rolling_zscore",
        inputs: &["x"],
        params: &["ddof", "window"],
        eval: functions::eval_rolling_zscore,
        lookback: functions::lookback_window,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "sma_crossover",
        inputs: &["price"],
        params: &["fast", "slow"],
        eval: functions::eval_sma_crossover,
        lookback: functions::lookback_sma_crossover,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "spread",
        inputs: &["bid_px", "ask_px"],
        params: &[],
        eval: functions::eval_spread,
        lookback: functions::lookback_pointwise,
        reaches_libm: false,
    },
    FeatureInfo {
        name: "trade_flow_imbalance",
        inputs: &["trade_sz", "trade_sign"],
        params: &["window"],
        eval: functions::eval_trade_flow_imbalance,
        lookback: functions::lookback_window,
        reaches_libm: false,
    },
];

/// Resolve a name, or refuse it while naming what exists.
pub fn feature_info(name: &str) -> Result<&'static FeatureInfo, FeatureError> {
    REGISTRY
        .iter()
        .find(|info| info.name == name)
        .ok_or_else(|| FeatureError::UnknownFeature {
            name: name.to_string(),
            available: registered_features().to_vec(),
        })
}

/// Every registered transform name, sorted.
pub fn registered_features() -> &'static [&'static str] {
    static NAMES: &[&str] = &[
        "book_imbalance",
        "close_location",
        "ema",
        "ema_crossover",
        "log_return",
        "mid_price",
        "momentum",
        "realized_volatility",
        "relative_range",
        "relative_spread",
        "rolling_mean",
        "rolling_std",
        "rolling_sum",
        "rolling_zscore",
        "sma_crossover",
        "spread",
        "trade_flow_imbalance",
    ];
    NAMES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_list_and_the_table_cannot_drift_apart() {
        // Two lists of the same thing is one list too many, and the second one is
        // the one that goes stale. It exists because `registered_features` has to
        // return a `&'static [&'static str]` for the error path; this test is the
        // price of that, and it is the whole price.
        let from_table: Vec<&str> = REGISTRY.iter().map(|i| i.name).collect();
        assert_eq!(from_table, registered_features());
    }

    #[test]
    fn the_registry_is_sorted_so_a_refusal_reads_alphabetically() {
        let mut sorted = registered_features().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, registered_features());
    }

    #[test]
    fn declared_params_are_sorted_because_a_spec_hashes_them_sorted() {
        for info in REGISTRY {
            let mut sorted = info.params.to_vec();
            sorted.sort_unstable();
            assert_eq!(
                sorted, info.params,
                "{} declares unsorted params",
                info.name
            );
        }
    }

    #[test]
    fn an_unknown_name_names_what_exists_rather_than_saying_no() {
        // At 03:00 "unknown feature" is not actionable and "unknown feature, here
        // are the seventeen that are not" is.
        let err = feature_info("rolling_meen").unwrap_err();
        match err {
            FeatureError::UnknownFeature { name, available } => {
                assert_eq!(name, "rolling_meen");
                assert!(available.contains(&"rolling_mean"));
            }
            other => panic!("expected UnknownFeature, got {other:?}"),
        }
    }

    #[test]
    fn a_bool_window_is_refused_rather_than_read_as_one() {
        // `isinstance(True, int)` is True in Python and `_check_window` refuses it
        // explicitly. A window of `True` is a window of 1: a legal window, and a
        // configuration nobody wrote.
        let mut params = Params::new();
        params.insert("window", Param::Bool(true));
        assert!(matches!(
            params.window("rolling_mean", "window", 1, 1),
            Err(FeatureError::Param { .. })
        ));
    }

    #[test]
    fn a_float_window_is_refused_because_it_fingerprints_as_a_different_recipe() {
        let mut params = Params::new();
        params.insert("window", Param::Float(20.0));
        assert!(matches!(
            params.window("rolling_mean", "window", 1, 1),
            Err(FeatureError::Param { .. })
        ));
    }

    #[test]
    fn a_missing_window_falls_back_to_the_transforms_own_default() {
        let params = Params::new();
        assert_eq!(params.window("log_return", "period", 1, 1).unwrap(), 1);
    }
}
