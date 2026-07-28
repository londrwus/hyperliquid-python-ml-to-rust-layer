# ADR-0016 — The versioned feature spec, and parity as three gates that can fail

**Status:** Accepted · **Date:** 2026-07-25

## Context

[03](../03-ml-fidelity-and-features.md) calls feature parity "the hard part — where
quality actually leaks," and [07](../07-parity-and-testing.md) turns "no quality loss" from a
claim into something every deploy has to pass. Until this increment, both documents described
work that did not exist: `axon.features` held one reference `log_return` and `axon.parity` held
a `NotImplementedError`. The prime directive — *never implement a feature twice* — was a
sentence in a docstring, not a property of anything.

Four questions had to be answered, and each has a wrong answer that looks reasonable:

1. **What is a "feature"?** A function over arrays is the obvious answer, and it is not enough:
   a model artifact that records "trained on features" reproduces nothing. What has to be
   versioned is the *call* — which transforms, with which parameters, in which order.
2. **What does the model gate assert?** A tolerance on the predictions is the obvious answer.
   It is also the answer that passes every model that ever silently moved a position.
3. **What does the feature gate compare, and how are the two sides lined up?** Row position is
   the obvious answer, and it is wrong the first time the online sampler misses a tick.
4. **How is drift measured so the number means anything?** PSI is standard; the three
   implementation details that decide whether it responds at all are not.

## Decision

**1. A feature is a length-preserving, causal, NaN-warmup function over 1-D arrays — and the
rules are enforced across the registry, not per function.**

Every transform in `axon.features.functions` returns an array as long as its input, where
element `i` depends only on `x[:i+1]`, and where warmup is NaN. Each rule buys a specific
failure:

- *Length-preserving* removes hand re-alignment against timestamps, which is where the
  windowing off-by-one in `docs/03`'s list of silent killers actually comes from.
- *Causal* is the difference between a backtest that is optimistic and one that is real.
- *NaN warmup* matters because zero is a legal value for every feature here, so a zero-filled
  warmup is indistinguishable from a genuine reading and the model trains on it.

The enforcement is the interesting part. The test named after extending a series with future
data recomputes each feature on a prefix of its input and asserts the historical values are
bit-identical, and it is **parametrized over the whole registry**, with a companion test
asserting the sweep covers every registered name. A feature added next month cannot opt
out by forgetting to write a test. A deliberately leaky centred moving average is checked to
*fail* that same assertion, because a leakage detector nobody has seen fail is a decoration.

**2. The unit of reproducibility is the `FeatureSpec`, not the function.**

A spec is an ordered, named, hash-identified list of `(feature, params)`. `fingerprint` is a
64-bit SHA-256 of its canonical JSON and `ref` is `name/vN#fingerprint` — what a model artifact
stores in `feature_spec_ref` (ADR-0003).

Three properties are deliberate:

- **Order is inside the hash.** Permuting two columns leaves every name correct and every
  prediction wrong, so column order is part of the identity rather than a convenience.
- **Columns compose.** A `FeatureDef` may bind an input to an *earlier column* of the same
  spec, so `mid_price` is computed once and the return, the volatility and the z-score all read
  it. Recomputing the mid inside each of those would be three implementations of one transform,
  which is the exact thing this package exists to prevent.
- **The hash covers the library, not just the recipe.** `FEATURES_VERSION` is folded into the
  fingerprint, and `FeatureSpec.from_dict` additionally refuses a spec whose recorded library
  version is not the running one. Without that second check the fingerprint would pin the
  recipe and say nothing about the kitchen: rewriting `rolling_std` under a spec's feet leaves
  every artifact id unchanged while quietly feeding every model different numbers.

Parameters are restricted to JSON scalars and normalized (sorted keys, `np.int64` → `int`) so
the fingerprint is stable across processes and machines; a golden test pins `PERP_CORE_V1`'s
literal id. Changing it is allowed — by bumping the spec version, which is what tells the
registry that artifacts carrying the old id were trained on different numbers.

**3. Model parity is a tolerance check `and` a decision check, not a tolerance check with
advice attached.**

The criteria are `docs/03`'s: exact for trees (`TREE_EPS = 0.0`, since inference is
deterministic threshold traversal and any drift means a float width changed), `max_abs_diff <
eps` starting at `1e-5` for neural nets, and — for both — **no discretized trading decision may
flip**. The gate takes a caller-supplied discretizer and reports *which inputs* flipped, with
their before/after scores and decisions, because "decision invariance violated" is not
actionable at 03:00.

The reason this is an `and` and not a fallback is in the paired tests: the same `1e-6` delta
passes when it lands in the middle of the flat band and fails when it sits on the long
threshold. In the failing case `max_abs_diff` is a thousand times *under* eps — the numeric
criterion alone would have signed the deploy off while a position went from flat to long.

One further trap is closed structurally: **a non-finite prediction is counted, never compared.**
`nan > 1e-5` is `False`, so a naive tolerance check reports a model that produced NaN as a
perfect match.

**4. Feature parity aligns on event time and reports the offending column and row.**

`align_by_event_time` matches the two sides on `ts_event` nanoseconds, because the online path
samples and the offline path recomputes a contiguous window: one dropped sample shifts every
subsequent row, and a positional comparison then reports total divergence for a feed that is
perfectly correct. Duplicate timestamps are refused rather than matched arbitrarily.

NaN handling is split deliberately: **NaN on both sides is a match** (warmup is legitimately
NaN in both paths and would otherwise fail every run), **NaN on one side only is a mismatch**
(a feature that goes NaN online and finite offline is precisely the staleness bug this gate
exists to catch). Under `np.allclose` defaults those two cases are indistinguishable.

The report names the worst column by cell count first and magnitude second, because a unit
error breaks one column on every row and that is a different diagnosis from one column being
slightly worse on one row.

**5. Drift is PSI + KL over bins frozen at training time.**

The conventional bands (`<0.1` stable, `0.1–0.25` moderate, `>0.25` significant) are only
meaningful if the binning is right, and three details decide that:

- **Quantile bins, frozen.** Equal-width bins over a fat-tailed return distribution put 99% of
  the mass in one bucket and PSI stops responding. Worse, *recomputing* quantile bins from each
  sample separately makes both histograms uniform by construction and PSI reads ~0 whatever
  happened — the monitor would be structurally blind. There is a test that measures exactly
  that blindness against the frozen-bin result.
- **Infinite outer edges**, so live values outside the training range are counted rather than
  dropped. "This feature now reaches values it never reached before" is the most informative
  drift there is.
- **Floored empty bins**, because an empty bucket makes the log term infinite, which on a small
  live window happens constantly and turns every alarm into the same alarm.

KL is reported as `KL(actual ‖ expected)` in nats — the surprise of today's data under the
distribution the model was fitted on. The direction is stated because KL is asymmetric and the
other direction answers a question nobody is asking. NaN rates are tracked *outside* the
histogram: a feature that starts emitting NaNs has drifted in a way no histogram of its finite
values can show, and PSI will happily call it stable.

**6. Features are float64; money is not.** Prices cross the wire as fixed-point integers and
order sizes stay in `Decimal` (`axon.strategy.context`), but a z-score is a statistic, not
money. The fixed-point → float conversion happens exactly once, in `axon.features.inputs`, and
never flows back toward an order size. `ts_event` is deliberately **not** an entry in the inputs
mapping and comes back as a separate int64 array: float64 carries 53 mantissa bits and a 2026
nanosecond timestamp needs 61, so a timestamp routed through the feature matrix would round
into ~256 ns buckets and reorder events. `FeatureSpec.compute` refuses any input above 2^53 for
the same reason — in practice that check fires on exactly one mistake, and it is this one.

**7. Gates return reports, not booleans.** The same call is a CI assertion
(`raise_for_status()`) and a live-monitor alarm (`passed` + `summary()`), which is what
`docs/07` needs when it asks for the parity monitor to keep running in production.

## Consequences

- **+** "No quality loss" is now a gate that can fail, and two tests prove it can: a tiny delta
  that does not flip a decision passes, the same delta on a threshold fails.
- **+** Every artifact can record exactly what its model was fed, and a change to the library's
  arithmetic invalidates that record loudly instead of silently.
- **+** Point-in-time correctness is a property of the library enforced across the registry,
  rather than of each feature author's care on the day.
- **+** The market-data ring (ADR-0012) feeds the spec directly: `md_slice_inputs` produces
  exactly the named arrays `PERP_CORE_V1` consumes, so the online and offline paths start from
  identical arrays and the gate compares transforms rather than decoders.
- **−** `FEATURES_VERSION` is a manual bump. Nothing detects a semantic change to a transform
  whose author forgets it; the only backstop is that such a change usually breaks a unit test
  first. Hashing the source would be worse — every comment edit would invalidate every artifact.
- **−** Rolling reductions run over explicit window views (O(n·w)) rather than differencing a
  cumulative sum (O(n)). The cumsum trick subtracts two nearly-equal large numbers on a series
  of perp prices and loses digits, which would leave the parity gate measuring our own
  arithmetic — but the constant factor will need revisiting for multi-year tick research.
- **−** The EMA is a Python loop, because the recursion *is* the definition and a closed form
  over powers of `(1 - alpha)` underflows on long series. It is the slowest thing here.
- **−** A spec is evaluated in declaration order, so a column must appear after everything it
  reads. This is a linear pass, not a scheduler; reordering is a spec edit and therefore a new
  fingerprint, which is correct but means the ordering is not free to change.
- **−** The PSI bands are the industry convention, not something derived from these features.
  They are a starting point per feature, and a feature that is naturally bimodal will trip them.
- **−** Feature parity compares two materialized matrices. A genuinely incremental Rust
  implementation (Boundary A) still has to emit vectors to be compared, so the gate imposes a
  small cost on the path it is protecting.
- **−** The artifact seam is only half-built here: this ADR defines the payload and the `ref`
  format, and `axon.models` owns writing them into an artifact. Until it does, a spec's identity
  is recorded by convention rather than by the registry.

See [ADR-0003](0003-model-serving-and-fidelity.md) (the artifact/registry and the model-parity
gate this implements), [ADR-0005](0005-fp32-no-quantization.md) (FP32 is what keeps NN drift
inside `eps` in the first place), [ADR-0012](0012-market-data-ring-and-multi-record-contract.md)
(the `MdSlice` record `md_slice_inputs` decodes), and
[ADR-0006](0006-signal-schema-and-spsc-ring.md) (event-time discipline at the other end of the
boundary).
