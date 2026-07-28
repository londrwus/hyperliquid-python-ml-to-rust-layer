# ADR-0035 — The Rust feature runtime, and what bit-exactness actually cost

**Status:** Accepted (amended 2026-07-27 — see [Amendment: `log` is not portable to the bit, and the reason is not the libc](#amendment-2026-07-27--log-is-not-portable-to-the-bit-and-the-reason-is-not-the-libc)) · **Date:** 2026-07-26

## Context

[03](../03-ml-fidelity-and-features.md) splits fidelity in two and is blunt about which half
matters:

```
   research signal  =  f_model( f_features( raw_data ) )
   live signal      =  g_model( g_features( raw_data ) )

   quality preserved  ⟺  f_model ≈ g_model   AND   f_features ≈ g_features
                             (model parity)        (FEATURE parity ← the hard one)
```

[ADR-0021](0021-rust-model-parity-gate.md) closed the left side and said, in its own words, that
it could not close the right one. A *parity bundle* carries a model, a holdout matrix and Python's
scores over it, and proves this build of Rust reproduces them. What it cannot prove — what nothing
in this repository could prove — is that the two languages would compute **the same feature
vectors from the same market data**. The bundle hands Rust a matrix Python already computed. The
comparison starts one step after the interesting question.

That gap was the last unchecked box in Phase 5, and it was unchecked for a structural reason
rather than an unfinished one: **there was no Rust implementation of any feature**, so there was
nothing to gate. `docs/adr/README.md` has carried "a **Rust feature runtime**" on its list of
undecided questions since the list was written.

Four questions had to be answered, and three of them have an answer that looks reasonable and is
wrong.

1. **Is a second implementation allowed at all?** `docs/03`'s prime directive is *never implement a
   feature twice*, and two independent implementations of "the same" transform **is** the bug.
2. **What is the criterion?** A tolerance is the obvious answer. It is also the answer that makes
   the gate unable to see the specific failure it exists for.
3. **Can Rust and NumPy agree to the bit at all?** "Obviously yes, it is just arithmetic" is the
   obvious answer, and it is false in a way that would have been discovered as a permanently red
   gate with no visible cause.
4. **What does a bounded serving buffer do to all this?** "Keep the last N and recompute" is the
   obvious answer, and it is right — for exactly the specs where it is right, which is a property
   nothing was checking.

## Decision

### 1. A second implementation is allowed, and it ships *behind* its own gate rather than ahead of it

`crates/axon-features` is a second implementation of seventeen transforms. That is the exception
`docs/03` Part 2's "single source of truth" list names explicitly at item 2 — "when features migrate to Rust (Boundary A, later, per-strategy):
the Rust implementation must be validated *bit-equivalent* against `axon.features` before it's
allowed to serve" — and the condition is the whole permission. So the crate does not exist without
the gate: `axon.parity.feature_bundle` writes a frozen question and
`axon_features::parity::FeatureBundle` answers it with no Python, no NumPy, no network and no
clock.

`docs/03` offered a second route — write the logic once in Rust and call it from Python through
bindings — and it is rejected here on cost rather than on principle. It would put a compiled
extension between the research path and every notebook, make `pip install` a toolchain problem,
and buy an identity that the gate already establishes empirically over real market data. If the
gate were ever hard to keep green, that trade would be worth revisiting; it has not been.

The crate is its own crate rather than a module inside `axon-model`, which the Phase-0 plan had
assumed. A feature runtime has nothing to do with inference: a strategy computing features in Rust
need not be serving a model at all, `baseline_z` being the existing proof that a signal can reach
a venue with no artifact anywhere in it.

### 2. The criterion is **bit equality**, and there is no tolerance arm to fall back to

The model gate has two criteria because its two families genuinely differ: trees are deterministic
threshold traversal and are held to bits, while ONNX does not encode operator ordering and float
addition is not associative, so graphs get a tolerance. Features have no such split. Every
transform in `axon.features` is built from `+ - * /`, `sqrt`, comparison, and `log`. IEEE-754
**requires** the first five to be correctly rounded, so they agree on any conforming platform as a
matter of the standard rather than of luck.

So `Criterion::BitExact` is both what a bundle declares and what a reader requires, and a manifest
asking for anything looser is refused — the same `allows` shape ADR-0021 uses, for the same
reason: a gate that can lower its own bar is not a gate.

The NaN rule is inherited from [ADR-0016](0016-feature-spec-and-parity-gates.md) §4 unchanged,
because it is the rule that makes the criterion usable: **NaN on both sides is a match, NaN on one
side only is a mismatch, and two finite values must be bit-equal.** Warmup is legitimately NaN in
both paths and would otherwise fail every run; a feature that goes NaN online and finite offline
is precisely the staleness bug the gate exists to catch. NaN *payloads* are deliberately not
compared — a quiet NaN's payload is not something either language promises, and two correct
runtimes can differ there.

### 3. `log` is the one operation whose agreement is measured, not guaranteed — and it is named on the wire

`log` is **not** in IEEE-754's correctly-rounded list. NumPy and glibc's libm agree here because
both compute it well, not because either must. That is a real distinction and it is recorded as
one rather than glossed: measured against NumPy 2.5.1 on this host, `f64::ln` and `np.log` agree
at **0 ULP**: over the 32 perp-close ratios the fixture pins in the tree, and over a
**200 000-sample sweep across 26 decades** that `scripts/modal_libm_probe.py` re-runs on demand
(the number is reproducible by a committed script rather than remembered from a console).

The bundle manifest therefore carries a `libm_columns` list: the columns whose value passes
through a logarithm, computed by walking the spec's dependency graph rather than hardcoded. It is
**a signpost, not a tolerance** — it can never excuse a mismatch. Its only job is that if this gate
ever reddens on a different libm, the first question ("did it redden *only* on those columns?")
has an answer sitting in the file.

### 4. NumPy does not sum a window left to right, and this is the finding that decided the design

This is the part that would not have been guessed, and it is why the criterion could have ended up
being a tolerance for no good reason.

`axon.features` reduces every rolling window with `np.ndarray.sum`. NumPy accumulates **pairwise**:
a plain loop below eight elements, eight independent accumulators combined in a fixed tree from
eight to 128, and a recursive split above that with the split forced onto a multiple of the unroll
factor. A Rust crate writing `iter().sum()` — the obvious, correct-looking thing — produces a
*different* number.

**How different is data-dependent, and that is the first thing to understand about it.** Whether
two summation orders round apart at all depends on the low bits of the summands, so the honest
form of this measurement is per-window and pinned. `tests/fixtures/generate.py` records exactly
that, measured by NumPy 2.5.1 over the fixture's own 160-sample perp-close series:

| window | separates? | naive vs `np.sum` |
|---|---|---|
| 5, 7 | no | 0 ULP — below the unroll threshold, NumPy loops too |
| 8   | yes | 1 ULP |
| 20  | **no** on this series | 0 ULP |
| 32  | yes | 2 ULP |
| 128 | yes | 3 ULP |
| 129 | yes | 4 ULP |

Read the `20` row: a spec window that the shipped `BAR_M1_V1` uses, on which these two orders
happen to agree. That is why the fixture stores a `naive_separates` flag per window rather than a
blanket claim, and why the test below asserts that *some* windows separate rather than that all of
them do. A sentence saying "the naive loop is wrong on every rolling column" would be false, and
falsifiable from this repo's own committed numbers.

**The magnitude that matters is not a ULP count on one window — it is what the mutation does to
the gate.** Replacing `pairwise_sum` with a left-to-right fold and recomputing the four committed
bundles moves **1 398 cells** in `all_transforms`, **584** in `bar_m1_btc`, **684** in
`bar_m1_eth` and **23** in `bar_m1_testnet_live`, concentrated in exactly the columns that reduce
(`tfi_16` 612, `z_20` 341/434, `vol_20` 243/250, …). One or two ULP is nothing to a model and
everything to a gate, and the consequences of not knowing it are both quiet:

- **Take it as a tolerance.** The gate is widened to absorb a systematic per-element error, and it
  is then blind to the class of bug it was built for — a windowing off-by-one, a NaN handled
  differently, a stale reading — every one of which lives comfortably inside such a tolerance.
- **Take it as a defect.** Somebody spends a day looking for a bug in transforms that are correct,
  because a red cross-language gate reads as "the Rust transforms are wrong."

So the order is **reproduced rather than tolerated**. `numeric::pairwise_sum` transcribes NumPy's
`DOUBLE_pairwise_sum`, and `mean` and `std` transcribe `_methods._var`'s two-pass structure —
whole-window mean first, then materialised squared deviations, then a *second* pairwise reduction,
then `sqrt`. With that, `rolling_mean`, `rolling_std`, `rolling_sum`, `rolling_zscore` and
`realized_volatility` are bit-identical to NumPy.

Two things follow, and both are load-bearing:

**The transcription is pinned against NumPy itself, not against this description of it.**
`tests/fixtures/generate.py` writes NumPy's own answers as raw bit patterns over a pinned series,
and `tests/cross_language.rs` checks them. A comment claiming to match NumPy is a comment; the day
NumPy changes its unroll factor or its block size, this repo has to find out from a red test rather
than from a parity report on a live feed. The fixture covers the windows the shipped specs use
(5 and 20 in `BAR_M1_V1`, 32 in `PERP_CORE_V1`) **and both of NumPy's own boundaries** — 7/8
straddle the unroll threshold and 128/129 straddle the recursive split — because a transcription
that got a boundary wrong would still pass on the shipped windows alone.

**There is a test that a naive summation would fail.** The positive result means nothing without
it: if left-to-right accumulation happened to agree everywhere, the transcription would be
unnecessary and the gate would be passing for free. `a_naive_summation_would_have_failed_this_gate`
walks the fixture's per-window `naive_separates` flags, checks that the Rust fold reproduces
Python's naive answer (so it is comparing the two orders it thinks it is), and asserts at least
three windows separate — a floor rather than "all of them", because the `20` row above shows why
"all" would be false.

Verified by mutation: replacing `pairwise_sum` with the naive loop reddens that test, the
transcription test beside it, **and the whole committed-bundle gate** — 2 689 cells across the four
bundles. That last part is the reassuring half rather than an inconvenience: a defect in the
reduction order is not quarantined to the two tests that name it.

**And `np.sum` is not `DOUBLE_pairwise_sum` either — it is `identity + DOUBLE_pairwise_sum`.**
This was a real defect in the first version of this crate, found after the gate was green, by
differential fuzzing against NumPy over roughly a million cells. `np.add.reduce` seeds its
accumulator with the ufunc identity `0.0` and adds the pairwise result to it. That outer add is
exact for every finite value with exactly one exception:

```text
0.0 + (-0.0) == +0.0
```

So a window summing to negative zero comes back from NumPy as **positive** zero, and a faithful
transcription of the inner function alone returns `-0.0`. One bit, on one value — and a bit-exact
gate is precisely the thing that cannot shrug at one bit. A legally-written parity bundle over a
signed-flow column pinned at `-0.0` reddened the shipped gate with 42 mismatched cells and
`max_abs_diff = 0e0`, which is the correct report and a startling one to read.

Three things about it are worth keeping:

- **The comment justifying the original code was backwards.** It said seeding the eight
  accumulators from the first eight elements rather than from zero *avoided* turning a leading
  `-0.0` into `+0.0`. NumPy turns it into `+0.0` anyway, one level up. The code was faithful to the
  function it was transcribed from and unfaithful to the function being claimed, and the comment
  argued for the wrong one confidently.
- **No committed bundle and no live feed could reach it.** `axon.features.inputs` divides
  fixed-point integers by `1e8`, and `0 / 1e8` is `+0.0`, so the whole existing corpus is immune.
  It was reachable only through the public `FeatureSpec::compute` on a hand-supplied array — which
  is to say, through research code that nobody had written yet.
- **The gate did not find it and could not have.** Four bundles of real venue data at
  `max_abs_diff = 0e0` say nothing about a value none of them contains. What found it was
  adversarial fuzzing *against NumPy directly*, and the lesson is the one ADR-0021 already learned
  about its own gate: a green corpus is evidence about the corpus. The `-0.0` family is now pinned
  in `cross_language.json` at seven lengths straddling all three branches, so the defect cannot
  come back — and note which lengths would have hidden it: below eight, NumPy's plain loop already
  starts from `+0.0`, so a test on short windows alone would have passed throughout.

**And the gate catches the failure it was actually built for.** The reduction-order finding is the
reason bit-exactness is *achievable*; the reason it is *worth having* is the class of bug
[03](../03-ml-fidelity-and-features.md) calls a silent killer. Measured by mutation against the
committed bundles: shifting one rolling mean's window by a single sample — the windowing
off-by-one, written the way somebody would actually write it by accident — turns the gate red with
**2 289 bit mismatches and 48 NaN disagreements over `all_transforms`' 16 200 cells** (the
widest of the four bundles; the gate's full corpus is 33 423), and names
`sma_x_4_12` as the worst column at **788 of 900 cells** rather than `mean_12`, which carries the
largest single delta. That ordering is ADR-0016 §4's rule doing its job: a unit error breaks one
column on every row, and that is a different diagnosis from one column being slightly worse on one
row. Restoring the shift returns the gate to `0e0` over all five bundles.

Nothing here claims NumPy's order is *better*. Pairwise summation has lower error growth, but the
point is agreement: the offline recompute is the definition and the Rust path reproduces it. If
Python's arithmetic were the worse of the two, this file would still transcribe it, and the fix
would be to change Python — where the change moves `FEATURES_VERSION` and invalidates every
artifact trained on the old numbers, which is the loud failure ADR-0016 §2 designed.

### 5. The spec's fingerprint is **recomputed** in Rust, never read

A `FeatureSpec` crosses as the canonical JSON Python writes. The cheap version of the reader takes
the recorded `fingerprint` field and carries it as a label; this one recomputes it — canonical
JSON, sorted keys at every level, `ensure_ascii` escaping, SHA-256, first 64 bits — and refuses a
mismatch.

That buys three refusals a label cannot, and the third is the one only Rust can make:

- a payload **edited after its fingerprint was taken** is refused, which is the same shape
  `Criterion::allows` guards one level up;
- a spec written against a **different build of `axon.features`** is refused by name, because
  `FEATURES_VERSION` is inside the hash — without it the fingerprint would pin the recipe and say
  nothing about the kitchen;
- **the two languages disagreeing about what the spec is** is refused. If Rust parsed the JSON into
  a structure that serialized back differently — a dropped empty `params`, an unsorted key, an int
  read as a float — the recomputed hash would not match. A runtime that merely *believed* the
  recorded id could be computing a different recipe under the right name, and nothing downstream
  would ever say so. This makes the fingerprint check a cross-language conformance test that runs
  on **every load**, not only in CI.

Both shipped specs re-identify in Rust: `bar_m1/v1#c503688de24e863f` and
`perp_core/v1#868d3dbe95d4b386`, with Rust's canonical serialization byte-identical to Python's.

Float parameters are the one soft spot — Python renders a float with `repr` and no two languages
agree on shortest-round-trip formatting in every corner. It is a soft spot rather than a hazard
because of *how* it fails: a formatting disagreement makes the hash differ, which is a loud refusal
at load time, never a spec that quietly computes something else. Every spec in this repo uses
integer parameters only.

### 6. The streaming runtime refuses a spec it cannot serve, which turns a house rule into a type error

`FeatureSpec::compute` is the batch path: the research path, and the gate's path. `FeatureStream`
is the serving path — a bounded buffer, one observation in, one feature row out — and it is what a
live Rust core would actually run.

It is bounded by a **derived** lookback, not a declared one. `max_lookback` composes through column
bindings: a column reading an earlier column needs its own window plus whatever that column needed,
less the one observation they share. For `BAR_M1_V1` that comes out at **21**, which is
`axon.features.spec.BAR_M1_WARMUP_BARS` — and the two are asserted against each other rather than
one restating the other. The house rule is *measure a warmup, do not restate it*, and the failure
behind it is specific: a constant goes stale the day a window widens, the buffer is then one
observation short, every row stays NaN, the strategy emits nothing forever, and **nothing raises**.
A strategy that never trades looks exactly like a strategy with no opinion.

`max_lookback` returns `Option`, and `None` — unbounded — is an honest answer rather than an
unknown. It is what an EMA is: the level depends on every observation ever seen. `FeatureStream`
**refuses** such a spec, naming the offending column. That is the house rule *"finite lookback on
every feature, no EMA, no expanding statistic"* stopping being a convention an author has to
remember.

And it immediately said something about this repository rather than about a hypothetical:
**`PERP_CORE_V1`, the reference perp spec, cannot be served from a bounded Rust buffer.** It
carries `ema_crossover`, exactly one of its nine columns is unbounded, and that one column costs
the whole spec. `BAR_M1_V1` — the model zoo's spec, and the one whose live m1 bars this gate runs on
— is bounded at 21 and serves. The batch path computes both faithfully; only the serving path
refuses, which is the right place for the refusal, since the research path has every right to an
EMA.

## Consequences

- **+** The half of Boundary A that ADR-0021 named in its own minus column now has a gate. A feature
  bundle compares vectors *the two languages each computed from the same market data*, which is the
  comparison `docs/03` calls the hard one and the one no Python-to-Python check can fail on.
- **+** The gate is bit equality rather than a tolerance, so it can still see the failures it was
  built for. Nothing was widened to make it pass.
- **+** A spec's identity is checked across the boundary on every load, not asserted. Two languages
  now have to agree on what a recipe *is* before they are allowed to disagree about a number.
- **+** "Finite lookback, no EMA" is enforced by the runtime that would serve it, and the enforcement
  found that the repo's own reference spec is unservable — a fact that was true before this ADR and
  that nothing could state.
- **+** The `libm` dependency is named on the wire rather than discovered on a new machine.
- **−** **There are now two implementations of seventeen transforms**, and no amount of gating makes
  that free. Every future feature has to be written twice or the two libraries drift, and the gate
  catches the drift only for specs a committed bundle actually covers. A transform registered in
  Python and missing in Rust is caught (the registries are compared); a transform whose Rust body
  is wrong in a way no committed bundle exercises is **not**.
- **−** **Bit-exactness is pinned to NumPy's implementation, not to a specification.** NumPy is free
  to change its summation order in a patch release. When it does, this repo's gate reddens
  everywhere at once and the fix is to re-transcribe — the failure is loud and the diagnosis is
  one test away, but it is maintenance that a tolerance would not have.
- **−** **`log` agreement is a measurement, and the measurement now covers three libcs — but not
  a Rust binary on any of them but one.** Read the chain carefully, because the obvious summary
  ("it's portable") is stronger than what was actually run.
  - **The compiler.** All **33 423 cells reproduce at `0e0` under `--release`** as well as under the
    default `dev` profile (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`). That is what Rust's refusal
    to enable fast-math predicts, and it rules out a bit-exact claim that holds only at
    `opt-level = 0`. "Expected" and "measured" are different words.
  - **The libc.** `scripts/modal_libm_probe.py` recomputes the `cross_language.json` fixture's own
    numbers on Modal (ADR-0017's offload path, seconds of CPU) in two containers: **glibc 2.36 on
    gVisor** and **musl on Alpine**, against this box's **glibc 2.39 on an AMD Ryzen 9 5950X**. On
    both, every one of the seven windows' `sum`/`mean`/`std` at both `ddof` agreed to the bit and
    all 32 recorded logarithms agreed to the bit. Separately, on each container, NumPy's `log` was
    measured against **that platform's own `math.log`** — which is the libm `f64::ln` calls — over
    **200 000 fresh samples across 26 decades, at 0 ULP**.
  - **What that chain does and does not establish.** It establishes that NumPy's pairwise
    summation order is not a property of one build, and that glibc 2.36, glibc 2.39 and musl
    produce the same `log` as the frozen fixture on every value tested. It does **not** establish
    that the *Rust gate* passes on musl: `rustup target list --installed` on this box has exactly
    one entry, so no `axon-features` binary has ever been built or run against a non-glibc libm.
    The arithmetic was ported; the artifact was not. macOS and non-x86 are untouched entirely.
    `libm_columns` names the columns to look at first; it does not make the gate portable, and
    this ADR does not claim it is.
- **−** `FeatureStream` recomputes over its bounded window rather than maintaining incremental
  state. That is what makes it bit-identical to the batch path — an incremental sum is a different
  summation order and would put the two paths on different arithmetic, which is the whole thing
  this ADR is about — but it is O(w) per observation per column rather than O(1). Phase 8 is
  deferred by decision and nothing here may be justified by a latency argument, so this is recorded
  as a known cost rather than optimized.
- **−** **Nothing in the live core calls this crate.** The runtime still computes no features in
  Rust; Python does, at Boundary B, exactly as before. This ADR builds the thing a Boundary-A
  promotion would need and gates it. Promoting a strategy is a separate decision and a separate
  change, and calling this "Boundary A" would be calling a proven capability a shipped one.
- **−** The gate covers the specs that have committed bundles. Five exist: two 900-bar mainnet
  corpora, 58 bars off a live testnet socket, 675 slices off a **recorded order book and tape**,
  and one 18-column spec naming every registered transform. That last one is the only reason
  coverage is 17 of 17, and the microstructure half is now gated twice — once over derived
  columns that say so, once over a venue's own book. **`perp_core_live` cannot be regenerated by
  a script**: it needs a live socket, and a generator that quietly substituted a synthetic
  stand-in under the same name would be a fixture lying about its provenance.
- **−** One committed bundle is a spec the serving path refuses. `all_transforms` carries an `ema` and an `ema_crossover` deliberately —
  it exists to gate the *library* rather than the subset a shipped strategy uses — so its
  `max_lookback()` is `None` and `FeatureStream` will not serve it. It is gated through the **batch**
  path, which is also the research path, and that is the honest coverage: those two columns are
  proven identical across the languages and are proven unservable from a bounded buffer, which are
  two different facts and both worth having. Nothing streams them, and nothing should.

See [ADR-0021](0021-rust-model-parity-gate.md) (the model half, and the gap this closes),
[ADR-0016](0016-feature-spec-and-parity-gates.md) (the `FeatureSpec`, the fingerprint, and the NaN
rule inherited unchanged), [ADR-0019](0019-native-rust-inference-backends.md) (the pin-the-runtime
argument, applied here to NumPy rather than to `tract`),
[ADR-0032](0032-the-model-zoo-and-what-actually-crosses-into-rust.md) (`BAR_M1_V1`, the spec this
gate runs on), and [ADR-0022](0022-first-ml-strategy.md) (the finite-lookback claim that
`FeatureStream` now enforces rather than assumes).


---

## Amendment (2026-07-27) — `log` is not portable to the bit, and the reason is not the libc

This ADR's own consequences said `log` "agrees by measurement", named two glibcs that agreed, and
put the plausible disagreement on **musl and macOS**. `scripts/modal_libm_probe.py` ran on both
libcs today and the result was neither of the two expected answers.

| Where | libc | NumPy SIMD | `np.log` vs the committed fixture | `math.log` vs it | wide sweep, `np.log` vs `math.log` |
|---|---|---|---|---|---|
| dev box (Ryzen 9 5950X) | glibc 2.39 | AVX2 | **32/32** | 32/32 | — |
| Modal, `debian_slim` | glibc 2.36 | AVX2 | **32/32** | 32/32 | 0 ULP |
| Modal, `debian_slim` | glibc 2.36 | **AVX-512 (SKX)** | **30/32** | 32/32 | **1 ULP** |
| Modal, `python:3.12-alpine` | musl | AVX2 | **32/32** | 32/32 | 0 ULP |
| Modal, `python:3.12-alpine` | musl | **AVX-512 (SKX)** | **30/32** | 32/32 | **1 ULP** |

Four container runs, two libcs × two CPU classes, and the correlation is **perfect with the CPU and
zero with the libc**. The first two runs of the probe looked like a libc difference and were not:
an earlier pass had `debian_slim` at 30/32 and Alpine at 32/32, and a later pass had it exactly the
other way round — **the same image, a different answer, on a different worker**. NumPy chooses its
`log` kernel from the CPU's feature flags at import time, so two runs of one image on two workers
are two different programs, and its AVX-512 kernel is 1 ULP away from its AVX2 kernel on two of the
32 ratios this repo pins. (That the probe could *say* which is why it now reports `cpu` and `avx`
on every run; before that field existed the same four numbers were unreadable.)

**The decisive column is `math.log`.** The platform's scalar libm agrees with the fixture on
**every** worker, glibc and musl alike. So the operation this ADR worried about — the one
IEEE-754 does not require to be correctly rounded — is *not* where the portability goes. It is
NumPy's own vectorized kernel, and this box, which has no AVX-512, cannot produce the disagreement
at any price.

**And Rust is on the libm side of it.** `f64::ln` on the dev box returns
`0x3f7477bbd3d9376d` and `0x3f62beb8b9d45ee7` for the two disputed ratios — the fixture's own bits,
0 ULP, matching `math.log` on both Modal workers and NumPy's AVX2 kernel.

The consequence is precise, it is asymmetric, and it is the one this ADR did not have:

> **On a host where NumPy dispatches to AVX-512, `axon.features` and `axon-features` disagree by
> one ULP on `log`-bearing cells — and neither is wrong.** The two gates then fail in *opposite*
> directions, which is worth stating separately because it decides who gets blamed.
>
> - **The Python side reddens.** `test_a_committed_bundle_still_describes_what_this_build_computes`
>   recomputes each bundle's matrix from its own inputs and re-hashes it against the committed
>   bytes. All five committed bundles name `ret_1` and a `mom_*`/`vol_*` pair in `libm_columns`, so
>   all five would move.
> - **The Rust side stays green.** `axon-features` computes `ln` through the platform's libm, which
>   is CPU-independent and matches the AVX2 answer the committed bundles were written with. Rust is
>   the *portable* half here, which is the reverse of what a reader of this ADR would assume.
> - **And regenerating the bundles on such a host inverts it**: the committed matrix moves to the
>   AVX-512 answer, and from then on the **Rust** gate reddens everywhere, attributing to the Rust
>   runtime a change that came out of NumPy's dispatch table.

So `libm_columns` stops being a signpost and becomes **load-bearing**: it is the list of columns
whose bit-exactness is a property of the machine that computed them, and the practical rule that
falls out is *do not regenerate a feature bundle on a machine whose `np._core._multiarray_umath.__cpu_features__`
you have not looked at.* Two follow-ups this amendment
deliberately does *not* decide, because each is a real trade rather than a fix:

1. whether the gate should pin the AVX2 answer and set `NPY_DISABLE_CPU_FEATURES` where it runs,
   which buys reproducibility by making the gate measure a NumPy nobody deploys;
2. whether `log`-bearing columns should be held to 1 ULP rather than to bits, which is the honest
   criterion for an operation two of NumPy's own kernels disagree about — and which gives up
   exactly the resolution the rest of the gate was built to keep.

**What is still unmeasured:** macOS, ARM (Apple silicon and Graviton both dispatch differently
again), and the Rust side on any of them. The probe now reports `cpu` and `avx` on every run, so a
future disagreement is attributable instead of mysterious — which is the change that mattered most,
because the first two runs of this experiment were read as a libc result and were not one.
