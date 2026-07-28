//! The incremental runtime: one observation in, one feature row out.
//!
//! [`FeatureSpec::compute`](crate::spec::FeatureSpec::compute) is the research path
//! and the gate's path — a whole matrix from whole arrays, which is what a backtest
//! has and what [`crate::parity`] compares. This module is the *serving* path: the
//! shape a live Rust core actually has, where the array does not exist yet because
//! the next observation has not happened.
//!
//! The two must agree to the bit or the migration to Boundary A buys nothing. That
//! they *can* agree is not a hope; it is a consequence of one property this module
//! enforces rather than assumes.
//!
//! # Why a bounded buffer reproduces the batch path exactly
//!
//! Every transform in [`crate::functions`] obeys `docs/03`'s three rules, and two of
//! them do the work here:
//!
//! - **Causal** — element `i` depends on `x[..=i]` and nothing later. So the last row
//!   of a computation over a *prefix* is the batch value at that index.
//! - **Finite lookback** — element `i` depends on `x[i + 1 - L ..= i]` and nothing
//!   earlier, for an `L` the transform declares. So the last row of a computation over
//!   the trailing `L` observations is *also* the batch value at that index.
//!
//! Put together: hold the last `max_lookback()` observations, recompute the column
//! over that window on every event, keep the last cell. Not "close enough to" the
//! batch value — the same arithmetic over the same window in the same order, so the
//! same bits. `tests/streaming_matches_batch.rs` measures it: **400 bars × 6 columns
//! = 2 400 cells, 2 400 agreeing.**
//!
//! The second rule is the one that has to be enforced, because a spec author can break
//! it by writing one word. [`FeatureStream::new`] **refuses** a spec whose lookback is
//! `None`, naming the column — which is what turns the house rule *finite lookback on
//! every feature, no EMA, no expanding statistic* from a convention an author has to
//! remember into an error the constructor produces. And the refusal bites on the
//! repo's own reference perp spec: `PERP_CORE_V1` carries `ema_crossover`, so **it is
//! unservable from a bounded Rust buffer**. That is a finding, not a formality — see
//! `the_reference_perp_spec_is_unservable_from_a_bounded_buffer_and_names_the_column`.
//!
//! # Why the whole column is recomputed, and not just its last cell
//!
//! Recomputing 21 rows to keep 1 looks wasteful, and a hand-written "last value only"
//! variant of each transform would be O(1) per event where this is O(L). It would also
//! be a **third** implementation of seventeen transforms `docs/03` says should have
//! one, sitting on the serving path, with the parity gate comparing the *other* two.
//! The whole argument of this crate is that a second implementation is tolerable only
//! because something holds it to the first; a third one, held to nothing, is the
//! training–serving skew arriving by the door marked "optimization".
//!
//! So [`push`](FeatureStream::push) calls exactly the `EvalFn` the batch path calls,
//! over a window of `lookback` observations, and throws away all but the last cell.
//! For `BAR_M1_V1` that is 6 columns over 21 rows per bar — on the order of a thousand
//! flops, against a venue round trip of 200–900 ms (`docs/05`). The day this is
//! measured to matter, the fix is a fourth implementation *with a gate on it*, not a
//! quiet one here.
//!
//! # No timestamps, deliberately
//!
//! Nothing in this module sees an event time, and that is not an omission. A feature is
//! defined over the *sequence* the core observed — the `i`-th row is the `i`-th push —
//! while event time governs which observations reach the core and in what order
//! (ADR-0006), one layer out. A stamp routed *into* the matrix would be worse than
//! redundant: float64 carries 53 mantissa bits, a 2026 nanosecond stamp needs 61, and
//! the rounding puts events into ~256 ns buckets and reorders them (ADR-0016 §6).
//! [`push`](FeatureStream::push) refuses a value above 2^53 for exactly that reason, as
//! [`FeatureSpec::compute`](crate::spec::FeatureSpec::compute) does, because the check
//! fires on precisely one mistake and it is this one.
//!
//! # Allocation
//!
//! `docs/05` asks the core not to allocate per event. Every buffer this module owns is
//! allocated in [`FeatureStream::with_lookback`] and reused: for `BAR_M1_V1`, 9 slots
//! (3 inputs + 6 columns) × 21 × 8 B, plus a 21-element scratch column and a 6-element
//! row — **1 728 bytes of buffer**, inside a construction that takes 178 allocations in
//! total (column and input names, the params clones, the spec traversals
//! `max_lookback` makes) and all of it before the first event.
//! [`FeatureStream::push`] itself allocates nothing: the per-column argument list is a
//! fixed `[&[f64]; 3]` on the stack, and the window is compacted with `copy_within`
//! rather than reallocated.
//!
//! It is **not** zero-allocation, and the residue is measured rather than estimated: the
//! transforms allocate their own intermediates. Counted with a global allocator over 400
//! pushes of `BAR_M1_V1`, steady state, the figure is **exactly 5 heap allocations and
//! 824 bytes per push** — 3 and 496 B for `z_20` (`rolling_zscore` takes two `Vec`s of
//! `lookback` and the `rolling_std` beneath it one of `window`: 2 × 21 × 8 + 20 × 8) and
//! 2 and 328 B for `vol_20` (`realized_volatility` takes one of `lookback`, plus the
//! same `rolling_std` scratch: 21 × 8 + 20 × 8). The other four columns allocate
//! nothing, and neither does this file. During warmup it is 3 and rises with the buffer,
//! because `rolling_std` returns before allocating when the window cannot be filled;
//! push 20 is where it becomes 5 (800 B; 824 B from push 21, once `vol_16`'s inner
//! window fills). Removing the last five means changing
//! [`crate::functions`], which is the gated file, so they stay until there is a reason
//! to spend that risk — and until then the number is stated rather than the property
//! claimed.
//!
//! Synchronous throughout. No `async`, no tokio, no clock, no `unsafe`.

use std::collections::BTreeMap;

use crate::functions::row_is_finite;
use crate::registry::{feature_info, EvalFn, Params};
use crate::spec::FeatureSpec;
use crate::{FeatureError, EXACT_FLOAT_LIMIT};

/// The widest call shape in the registry — `relative_range` and `close_location` read
/// three arrays each.
///
/// It exists so [`FeatureStream::push`] can hold its argument list in a stack array
/// instead of a `Vec`, which is the difference between one allocation per column per
/// event and none. `no_registered_transform_takes_more_inputs_than_the_argument_array_holds`
/// pins it against the registry, so a four-input transform added later reddens a test
/// here rather than being refused at runtime by a process that is already serving.
const MAX_ARITY: usize = 3;

/// One column, resolved once at construction.
///
/// Resolution — name → transform, binding → slot index — happens in
/// [`FeatureStream::with_lookback`] and never again. A per-event string lookup would
/// put a map probe on the hot path for an answer that cannot change, and the answer is
/// the one `FeatureSpec::compute` reaches by the same rules.
#[derive(Debug)]
struct Step {
    eval: EvalFn,
    params: Params,
    /// Slot indices of this transform's positional inputs, in call order.
    args: [usize; MAX_ARITY],
    arity: usize,
    /// Slot this column's values are written back into, so a later column can bind to
    /// it. Written even when nothing reads it: which columns get read is a property of
    /// the spec, and a `Step` that sometimes skipped the write would be a second thing
    /// to keep in step with the binding table.
    out: usize,
}

/// The incremental feature runtime — a bounded buffer per input, one row per event.
///
/// Construct it from a spec whose lookback is finite, push one observation at a time,
/// and read the feature row back. The row is the row
/// [`FeatureSpec::compute`](crate::spec::FeatureSpec::compute) would have produced at
/// that index, bit for bit, which is the whole point.
#[derive(Debug)]
pub struct FeatureStream {
    /// Matrix column names in spec order — *not* sorted. Column order is inside the
    /// spec fingerprint (ADR-0016 §2) because permuting two leaves every name correct
    /// and every prediction wrong.
    columns: Vec<String>,
    /// The input names [`FeatureStream::push`] expects, sorted: exactly
    /// `FeatureSpec::required_inputs()`.
    inputs: Vec<String>,
    plan: Vec<Step>,
    lookback: usize,
    /// One `lookback`-long buffer per input, then one per column. Contiguous within a
    /// slot, because every reduction in [`crate::numeric`] takes a single `&[f64]` and
    /// sums it in NumPy's pairwise order — a ring buffer would hand a transform two
    /// slices, and a window summed in two pieces is a different float.
    slots: Vec<Vec<f64>>,
    /// One transform's output over the buffered window.
    scratch: Vec<f64>,
    /// The row [`FeatureStream::push`] hands back. Owned by the stream and overwritten
    /// in place, so a caller that wants to keep a row copies it — the same contract
    /// `FeatureMatrix::row` has.
    row: Vec<f64>,
    /// Observations currently in the buffer: `min(observed, lookback)`.
    len: usize,
    observed: usize,
    warm: bool,
}

impl FeatureStream {
    /// Build a stream for `spec`, sized from the spec's own derived lookback.
    ///
    /// Refuses a spec that cannot be served from a bounded buffer, naming the offending
    /// column. See [`FeatureStream::with_lookback`] for why the depth is derived and
    /// never a constant.
    pub fn new(spec: &FeatureSpec) -> Result<Self, FeatureError> {
        // The `0` is never used as a depth: [`FeatureStream::with_lookback`] re-derives
        // the lookback and refuses an unbounded spec before it reads the argument at
        // all. Written this way so that the refusal exists in exactly **one** place — the
        // first draft of this file refused here as well, and a perturbation that deleted
        // the refusal in `with_lookback` left every test green, because this copy was
        // still guarding. A duplicated check is a check that can be removed without
        // anything noticing.
        let derived = spec.max_lookback()?.unwrap_or(0);
        Self::with_lookback(spec, derived)
    }

    /// Build a stream with an explicitly chosen buffer depth.
    ///
    /// The depth may be **deeper** than the spec's derivation — a caller holding more
    /// history than the recipe needs gets bit-identical rows, which is precisely what
    /// "bounded" means and is asserted as such in
    /// `a_deeper_buffer_than_the_derivation_changes_no_value_which_is_what_bounded_means`.
    /// It may never be shallower, and that refusal is why this constructor is public at
    /// all.
    ///
    /// A buffer one observation short does not fail loudly on its own: the deepest
    /// window never fills, `vol_20`'s cell stays NaN, every row is therefore incomplete,
    /// a strategy gated on a finite row emits nothing for the rest of the session, and
    /// **nothing raises** — a strategy that never trades looks exactly like a strategy
    /// with no opinion. That is the failure this check turns into a startup error, and
    /// it is why the depth comes from [`FeatureSpec::max_lookback`] rather than from a
    /// constant: a constant saying "21 bars" is correct until the day somebody widens a
    /// window, and wrong silently thereafter.
    pub fn with_lookback(spec: &FeatureSpec, lookback: usize) -> Result<Self, FeatureError> {
        let derived = match spec.max_lookback()? {
            Some(d) => d,
            None => return Err(unbounded(spec)),
        };
        if lookback < derived {
            return Err(FeatureError::Spec(format!(
                "a buffer of {lookback} observations is shorter than spec {:?}'s derived lookback \
                 of {derived}; the deepest window would never fill, every row would stay NaN, a \
                 strategy gated on a finite row would emit nothing for the rest of the session, \
                 and nothing would raise",
                spec.name()
            )));
        }

        let inputs = spec.required_inputs()?;
        let columns: Vec<String> = spec.columns().into_iter().map(str::to_string).collect();

        // The same refusal `FeatureSpec::compute` makes, for the same reason and at the
        // same strength: if a column could shadow a supplied input, "which one did
        // feature X read?" is decided by evaluation order, and inserting a column
        // silently re-points a downstream feature. The two paths must accept exactly
        // the same specs — a spec that streams but does not compute has no gate.
        let collisions: Vec<&String> = columns.iter().filter(|c| inputs.contains(c)).collect();
        if !collisions.is_empty() {
            return Err(FeatureError::Spec(format!(
                "column name(s) {collisions:?} collide with supplied input names"
            )));
        }

        let mut plan = Vec::with_capacity(columns.len());
        for (j, def) in spec.features().iter().enumerate() {
            let info = feature_info(def.feature())?;
            let sources = def.sources()?;
            if sources.len() > MAX_ARITY {
                return Err(FeatureError::Spec(format!(
                    "{} reads {} input arrays and the streaming runtime holds at most \
                     {MAX_ARITY}; raise MAX_ARITY in streaming.rs",
                    def.feature(),
                    sources.len()
                )));
            }
            let mut args = [0usize; MAX_ARITY];
            for (k, source) in sources.iter().enumerate() {
                // Earlier columns first, then the supplied inputs. The collision check
                // above is what makes that order irrelevant rather than a rule to
                // remember — no name can be both.
                args[k] = if let Some(c) = columns[..j].iter().position(|c| c == source) {
                    inputs.len() + c
                } else if let Some(i) = inputs.iter().position(|n| n == source) {
                    i
                } else {
                    return Err(FeatureError::Spec(format!(
                        "column {:?} reads {source:?}, which is neither a supplied input \
                         {inputs:?} nor an earlier column",
                        def.column()
                    )));
                };
            }
            plan.push(Step {
                eval: info.eval,
                params: def.params().clone(),
                args,
                arity: sources.len(),
                out: inputs.len() + j,
            });
        }

        // Every buffer the stream will ever use, allocated here and never again. Seeded
        // with NaN rather than zero so that a bug reading past `len` would produce a
        // NaN column — visible — instead of a plausible zero.
        let slots = vec![vec![f64::NAN; lookback]; inputs.len() + columns.len()];
        Ok(Self {
            scratch: vec![f64::NAN; lookback],
            row: vec![f64::NAN; columns.len()],
            columns,
            inputs,
            plan,
            lookback,
            slots,
            len: 0,
            observed: 0,
            warm: false,
        })
    }

    /// The derived buffer depth, in raw observations.
    ///
    /// For `BAR_M1_V1` this is **21** — the same number as
    /// `axon.features.spec.BAR_M1_WARMUP_BARS` — and it is computed from the spec
    /// rather than declared. [`FeatureSpec::max_lookback`] has the composition rule.
    pub fn lookback(&self) -> usize {
        self.lookback
    }

    /// The matrix column names, in spec order.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// The input names each [`push`](FeatureStream::push) must supply, sorted.
    pub fn required_inputs(&self) -> &[String] {
        &self.inputs
    }

    /// How many observations have been pushed.
    ///
    /// Counts every observation the stream consumed, warmup included and regardless of
    /// whether the row it produced was finite: it is a count of events seen, not of
    /// opinions formed. [`is_warm`](FeatureStream::is_warm) answers the second question,
    /// and the two are genuinely different — as that method explains.
    pub fn observed(&self) -> usize {
        self.observed
    }

    /// Push one observation and get the feature row for it.
    ///
    /// `inputs` is keyed by [`FeatureSpec::required_inputs`]. A missing key or an extra
    /// key is an error rather than a default: "the venue published no low" and "the
    /// caller forgot to wire `low`" are different failures, and a NaN default would make
    /// the second — a wiring mistake that is permanent and total — arrive looking like
    /// the first, which is a quiet market and clears up on its own.
    ///
    /// During warmup the row is returned with NaN cells. Never withheld, because a
    /// caller that gets no row cannot tell "warming up" from "the core is wedged", and
    /// because warmup is per *column*: `clv` and `range_bps` are pointwise and finite on
    /// the very first bar while `vol_20` is NaN until the 21st, so a runtime that
    /// withheld the row would hide twenty bars of perfectly good readings. Never
    /// zero-filled, because zero is a legal value for every feature here — `clv` is
    /// exactly 0.0 for a bar that closed dead centre in its range, and a zero-filled
    /// warmup is indistinguishable from twenty of those.
    ///
    /// The returned slice is the stream's own row buffer and is overwritten by the next
    /// push.
    pub fn push(&mut self, inputs: &BTreeMap<String, f64>) -> Result<&[f64], FeatureError> {
        // Validated in full before a single buffer is touched: a push that is going to
        // be refused must not leave half an observation in the window, or the next row
        // is computed over a series that never happened.
        for key in inputs.keys() {
            if !self.inputs.iter().any(|n| n == key) {
                return Err(FeatureError::Inputs(format!(
                    "unexpected input {key:?}; this spec reads {:?}, and an unrecognised key is a \
                     wiring mistake rather than spare data — whatever name it was meant to be is \
                     getting nothing",
                    self.inputs
                )));
            }
        }
        for name in &self.inputs {
            let v = match inputs.get(name) {
                Some(v) => *v,
                None => {
                    return Err(FeatureError::Inputs(format!(
                        "missing input {name:?}; this spec reads {:?}",
                        self.inputs
                    )))
                }
            };
            if v.is_finite() && v.abs() >= EXACT_FLOAT_LIMIT {
                return Err(FeatureError::Inputs(format!(
                    "input {name:?} holds {v}, which exceeds 2^53 and cannot be held exactly in \
                     float64; nanosecond timestamps are not features — they order the events this \
                     stream is fed, and never enter a row of it"
                )));
            }
        }

        // Append. Once the window is full it is compacted with a `lookback`-element
        // memmove — 21 f64, 168 bytes, per input per bar for `BAR_M1_V1` — rather than
        // being kept in a ring: a ring's window wraps, and every reduction in `numeric`
        // sums one contiguous slice in NumPy's pairwise order, so a window handed over
        // in two pieces would be grouped differently and come back a different float.
        // That is a parity failure which appears only once the buffer first wraps,
        // which on m1 bars is 21 minutes after the process starts.
        let (len, cap) = (self.len, self.lookback);
        for (slot_ix, name) in self.inputs.iter().enumerate() {
            let v = inputs.get(name).copied().unwrap_or(f64::NAN);
            let slot = &mut self.slots[slot_ix];
            if len < cap {
                slot[len] = v;
            } else {
                slot.copy_within(1.., 0);
                slot[cap - 1] = v;
            }
        }
        self.len = (len + 1).min(cap);
        self.observed += 1;
        let len = self.len;

        // Evaluate in declaration order, exactly as `FeatureSpec::compute` does, so a
        // column binding to an earlier one reads values that are already there.
        for (j, step) in self.plan.iter().enumerate() {
            let mut args: [&[f64]; MAX_ARITY] = [&[]; MAX_ARITY];
            for (arg, &slot) in args.iter_mut().zip(step.args.iter()).take(step.arity) {
                *arg = &self.slots[slot][..len];
            }
            if let Err(e) = (step.eval)(&args[..step.arity], &step.params, &mut self.scratch[..len])
            {
                // The observation stays in the buffer — it genuinely was observed — but
                // the row is blanked rather than left holding the previous event's
                // values, which would be a stale row indistinguishable from a fresh one.
                // Unreachable in practice after construction: every parameter a
                // transform reads was validated before the first buffer was allocated.
                self.row.fill(f64::NAN);
                self.warm = false;
                return Err(e);
            }
            self.slots[step.out][..len].copy_from_slice(&self.scratch[..len]);
            self.row[j] = self.scratch[len - 1];
        }

        self.warm = row_is_finite(&self.row);
        Ok(&self.row)
    }

    /// Whether the last row returned was fully finite.
    ///
    /// A property of the **row**, not of the observation count, and the difference is
    /// not theoretical: `clv` is NaN for a bar whose high equals its low, and **6 of 58
    /// BTC and 5 of 62 ETH live m1 bars** hit that in a measured testnet session (see
    /// [`crate::functions::close_location`]). A stream that has been warm for hours goes
    /// cold for one bar when the market stops moving, and a caller gating on
    /// `observed() >= lookback()` instead would hand a model a NaN it never trained on.
    ///
    /// False before the first push.
    pub fn is_warm(&self) -> bool {
        self.warm
    }
}

/// The refusal, naming the column that caused it.
///
/// Naming it is most of the value: "this spec is unbounded" sends a reader to a
/// nine-column recipe at 03:00, and "`ema_x_8_32` (`ema_crossover`) is unbounded" sends
/// them to the line. The message also names the finite counterpart, because the fix is
/// usually `sma_crossover` and the reader would otherwise have to discover for
/// themselves that the registry has one.
fn unbounded(spec: &FeatureSpec) -> FeatureError {
    for def in spec.features() {
        match spec.column_lookback(def.column()) {
            Ok(Some(_)) => {}
            Ok(None) => {
                return FeatureError::Spec(format!(
                    "spec {:?} cannot be served from a bounded buffer: column {:?} ({}) has an \
                     unbounded lookback — its value depends on every observation the process has \
                     ever seen, so a buffer of any finite depth computes a different number from \
                     the offline recompute, and the gap is widest right after a restart, which is \
                     the moment nobody is watching the feature values. Replace it with a \
                     finite-lookback transform (`sma_crossover` is the counterpart of \
                     `ema_crossover`), or serve this spec from the batch path",
                    spec.name(),
                    def.column(),
                    def.feature()
                ))
            }
            Err(e) => return e,
        }
    }
    // `max_lookback` said the spec was unbounded, so some column is. Reaching here means
    // the two disagree, which is a bug in this crate rather than in the spec.
    FeatureError::Spec(format!(
        "spec {:?} reports an unbounded lookback but no column claims one; max_lookback and \
         column_lookback disagree",
        spec.name()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{registered_features, Param};

    /// `BAR_M1_V1`, transcribed — the same literal `spec.rs` carries, byte-identical to
    /// what `axon.features.spec.BAR_M1_V1.to_json()` writes minus the fingerprint.
    const BAR_M1_JSON: &str = r#"{"features":[{"column":"ret_1","feature":"log_return","inputs":{"price":"close"},"params":{"period":1}},{"column":"mom_5","feature":"momentum","inputs":{"price":"close"},"params":{"window":5}},{"column":"z_20","feature":"rolling_zscore","inputs":{"x":"close"},"params":{"window":20}},{"column":"vol_20","feature":"realized_volatility","inputs":{"price":"close"},"params":{"window":20}},{"column":"range_bps","feature":"relative_range","inputs":{},"params":{}},{"column":"clv","feature":"close_location","inputs":{},"params":{}}],"library_version":1,"spec":"bar_m1","version":1}"#;

    fn bar_spec() -> FeatureSpec {
        FeatureSpec::from_json(BAR_M1_JSON).expect("the transcribed bar spec must load")
    }

    fn bar(close: f64, high: f64, low: f64) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("close".to_string(), close),
            ("high".to_string(), high),
            ("low".to_string(), low),
        ])
    }

    #[test]
    fn the_buffer_depth_is_the_specs_own_derivation_and_not_a_constant() {
        // The number 21 exists in this crate in exactly one place that computes it. A
        // stream that restated a literal `max_lookback` did not produce would go one
        // short the day somebody widens a window, and stay silent about it.
        let spec = bar_spec();
        let stream = FeatureStream::new(&spec).unwrap();
        assert_eq!(stream.lookback(), spec.max_lookback().unwrap().unwrap());
        assert_eq!(stream.lookback(), 21);
        assert_eq!(
            stream.columns(),
            ["ret_1", "mom_5", "z_20", "vol_20", "range_bps", "clv"]
        );
        assert_eq!(stream.required_inputs(), ["close", "high", "low"]);
        assert_eq!(stream.observed(), 0);
        assert!(
            !stream.is_warm(),
            "a stream is not warm before its first push"
        );
    }

    #[test]
    fn a_spec_with_an_unbounded_column_is_refused_by_name_rather_than_served_wrong() {
        // The type-error half of "finite lookback on every feature": an EMA never
        // forgets its seed, so no finite buffer reproduces the offline number, and the
        // refusal has to name the column or the reader is left auditing the recipe.
        let json = r#"{"features":[{"column":"mid","feature":"mid_price","inputs":{},"params":{}},{"column":"ema_x_8_32","feature":"ema_crossover","inputs":{"price":"mid"},"params":{"fast":8,"slow":32}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        match FeatureStream::new(&spec) {
            Err(FeatureError::Spec(msg)) => {
                assert!(msg.contains("ema_x_8_32"), "the column is not named: {msg}");
                assert!(
                    msg.contains("ema_crossover"),
                    "the transform is not named: {msg}"
                );
                assert!(
                    msg.contains("sma_crossover"),
                    "the message does not point at the finite counterpart: {msg}"
                );
            }
            other => panic!("expected a refusal naming the column, got {other:?}"),
        }
    }

    #[test]
    fn a_plain_ema_column_is_refused_too_and_not_only_the_crossover() {
        // `ema` and `ema_crossover` are two registered names with one unbounded lookback
        // between them; a refusal keyed on the crossover alone would let the simpler
        // spelling straight through onto a serving path.
        let json = r#"{"features":[{"column":"e8","feature":"ema","inputs":{"x":"close"},"params":{"span":8}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        match FeatureStream::new(&spec) {
            Err(FeatureError::Spec(msg)) => assert!(msg.contains("e8"), "{msg}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn unboundedness_reaching_a_column_through_a_binding_is_refused_at_that_column() {
        // `m` is a 3-sample mean *of an EMA*, so it inherits the unboundedness. The
        // message must name a column that is genuinely unbounded rather than whichever
        // one it happened to check first.
        let json = r#"{"features":[{"column":"e","feature":"ema","inputs":{"x":"close"},"params":{"span":4}},{"column":"m","feature":"rolling_mean","inputs":{"x":"e"},"params":{"window":3}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        match FeatureStream::new(&spec) {
            Err(FeatureError::Spec(msg)) => assert!(
                msg.contains("\"e\""),
                "the first unbounded column is not named: {msg}"
            ),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_input_key_is_refused_rather_than_defaulted_to_a_nan_row() {
        // A wiring mistake is permanent and total; a NaN default would dress it up as a
        // quiet market, which clears up on its own and therefore nobody investigates.
        let spec = bar_spec();
        let mut stream = FeatureStream::new(&spec).unwrap();
        let mut partial = bar(100.0, 101.0, 99.0);
        partial.remove("low");
        match stream.push(&partial) {
            Err(FeatureError::Inputs(msg)) => assert!(msg.contains("low"), "{msg}"),
            other => panic!("expected a refusal naming the input, got {other:?}"),
        }
        // …and the refused observation left nothing behind in the buffer.
        assert_eq!(stream.observed(), 0);
    }

    #[test]
    fn an_extra_input_key_is_refused_because_it_is_a_wiring_mistake_not_spare_data() {
        // The realistic shape: a caller misspells `close` as `closes` and supplies both.
        // Ignoring the extra key lets that through with `close` reading whatever was
        // left over.
        let spec = bar_spec();
        let mut stream = FeatureStream::new(&spec).unwrap();
        let mut extra = bar(100.0, 101.0, 99.0);
        extra.insert("volume".to_string(), 12.0);
        match stream.push(&extra) {
            Err(FeatureError::Inputs(msg)) => assert!(msg.contains("volume"), "{msg}"),
            other => panic!("expected a refusal naming the extra key, got {other:?}"),
        }
        assert_eq!(stream.observed(), 0);
    }

    #[test]
    fn a_nanosecond_timestamp_pushed_as_a_feature_is_refused_on_the_serving_path_too() {
        // `FeatureSpec::compute` refuses this, and the serving path has to as well or
        // the one mistake the check exists for gets caught in research and waved through
        // live. 1.7e18 is a 2026 stamp; float64 rounds it into ~256 ns buckets.
        let spec = bar_spec();
        let mut stream = FeatureStream::new(&spec).unwrap();
        match stream.push(&bar(1.7e18, 1.7e18, 1.7e18)) {
            Err(FeatureError::Inputs(msg)) => assert!(msg.contains("2^53"), "{msg}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(stream.observed(), 0);
        // A NaN is not what this check is for and passes it: a non-finite input is a
        // legitimate "no reading", which the transforms' own NaN rules already handle.
        assert!(stream.push(&bar(f64::NAN, f64::NAN, f64::NAN)).is_ok());
        assert_eq!(stream.observed(), 1);
    }

    #[test]
    fn observed_counts_every_push_including_the_ones_that_produced_nothing_usable() {
        let spec = bar_spec();
        let mut stream = FeatureStream::new(&spec).unwrap();
        for i in 0..5 {
            let c = 100.0 + i as f64;
            stream.push(&bar(c, c + 1.0, c - 1.0)).unwrap();
        }
        assert_eq!(stream.observed(), 5);
        assert!(!stream.is_warm(), "five bars cannot warm a 21-bar spec");
    }

    #[test]
    fn is_warm_is_a_property_of_the_row_not_of_the_observation_count() {
        // The measured live case: a bar that traded at one price all minute makes `clv`
        // NaN, and a stream warm for hours goes cold for exactly that bar. A caller
        // gating on `observed() >= lookback()` would hand a model a NaN it never
        // trained on.
        let spec = bar_spec();
        let mut stream = FeatureStream::new(&spec).unwrap();
        for i in 0..40 {
            let c = 100.0 + ((i * 7 % 13) as f64) * 0.25;
            stream.push(&bar(c, c + 0.5, c - 0.5)).unwrap();
        }
        assert!(stream.is_warm(), "40 bars should have warmed a 21-bar spec");
        let row = stream.push(&bar(101.0, 101.0, 101.0)).unwrap().to_vec();
        assert!(
            row[5].is_nan(),
            "a bar with no range scored a clv of {}",
            row[5]
        );
        assert!(!stream.is_warm(), "a flat bar left the stream reading warm");
        assert!(stream.observed() > stream.lookback());
        // And it comes back on the next bar that moves: the coldness is one row, not a
        // latch that has to be reset.
        stream.push(&bar(101.5, 102.0, 101.0)).unwrap();
        assert!(stream.is_warm());
    }

    #[test]
    fn no_registered_transform_takes_more_inputs_than_the_argument_array_holds() {
        // `MAX_ARITY` is what keeps `push` off the heap. A four-input transform added
        // later must redden this rather than being refused at runtime by a process that
        // is already serving.
        let widest = registered_features()
            .iter()
            .map(|name| feature_info(name).unwrap().inputs.len())
            .max()
            .unwrap();
        assert_eq!(widest, 3, "the widest call shape in the registry moved");
        assert_eq!(MAX_ARITY, widest);
    }

    #[test]
    fn a_column_that_shadows_an_input_is_refused_here_too_so_both_paths_take_the_same_specs() {
        // `FeatureSpec::compute` refuses this. If the streaming path accepted it, a spec
        // would exist that serves and cannot be gated — which is worse than either path
        // refusing it, because the gate is the only reason this crate is allowed to be a
        // second implementation at all.
        let json = r#"{"features":[{"column":"close","feature":"rolling_mean","inputs":{"x":"close"},"params":{"window":3}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        assert!(matches!(
            FeatureStream::new(&spec),
            Err(FeatureError::Spec(_))
        ));
    }

    #[test]
    fn a_column_bound_to_an_earlier_column_reads_that_columns_buffer_not_a_stale_input() {
        // Composition is the whole reason `column_lookback` is not `max(window)`. A
        // 5-sample mean of a 4-sample mean spans 8 observations, and the streamed value
        // is the batch value at that index only if the intermediate column's buffer is
        // written back before the second column reads it.
        let json = r#"{"features":[{"column":"a","feature":"rolling_mean","inputs":{"x":"px"},"params":{"window":4}},{"column":"b","feature":"rolling_mean","inputs":{"x":"a"},"params":{"window":5}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        let mut stream = FeatureStream::new(&spec).unwrap();
        assert_eq!(stream.lookback(), 8);

        let px: Vec<f64> = (0..20)
            .map(|i| 10.0 + ((i * 5 % 9) as f64) * 0.125)
            .collect();
        let batch = spec
            .compute(&BTreeMap::from([("px".to_string(), px.clone())]))
            .unwrap();
        for (i, v) in px.iter().enumerate() {
            let row = stream
                .push(&BTreeMap::from([("px".to_string(), *v)]))
                .unwrap();
            let (s, b) = (row[1], batch.get(i, 1));
            assert!(
                s.to_bits() == b.to_bits() || (s.is_nan() && b.is_nan()),
                "row {i}: streamed {s} against batch {b}"
            );
        }
    }

    #[test]
    fn a_params_map_the_transform_would_reject_is_refused_at_construction_not_mid_session() {
        // A `fast` that is not shorter than `slow` inverts the sign of the whole
        // feature. Catching it on the first push would mean a process that started clean
        // and failed at the first market event, which reads as a data problem.
        //
        // Nothing in this module performs that check: it falls out of the
        // `max_lookback()` call that sizes the buffer, because a lookback function has
        // to read the same parameters the transform does. Asserted here anyway, because
        // "construction validates the parameters" is the property a caller relies on and
        // it would be quietly lost if the depth ever came from somewhere cheaper.
        let json = r#"{"features":[{"column":"x","feature":"sma_crossover","inputs":{"price":"close"},"params":{"fast":6,"slow":6}}],"library_version":1,"spec":"t","version":1}"#;
        let spec = FeatureSpec::from_json(json).unwrap();
        assert!(matches!(
            FeatureStream::new(&spec),
            Err(FeatureError::Param { .. })
        ));
    }

    #[test]
    fn the_params_a_step_carries_are_the_specs_own_and_not_the_transform_defaults() {
        // A `Step` that dropped its params would compute `log_return` with period 1
        // under a column named `mom_5` — numerically fine, trains fine, and not the
        // feature anybody asked for.
        let spec = bar_spec();
        let stream = FeatureStream::new(&spec).unwrap();
        assert_eq!(stream.plan.len(), 6);
        assert_eq!(stream.plan[1].params.get("window"), Some(&Param::Int(5)));
        assert_eq!(stream.plan[2].params.get("window"), Some(&Param::Int(20)));
        assert_eq!(stream.plan[4].params.len(), 0);
    }
}
