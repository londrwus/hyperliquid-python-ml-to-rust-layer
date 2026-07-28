# ADR-0015 — Model artifacts: native for trees, ONNX for the rest, immutable in a versioned registry

**Status:** Accepted · **Date:** 2026-07-25

## Context

[ADR-0003](0003-model-serving-and-fidelity.md) decided the policy — native format for
XGBoost/LightGBM, ONNX at FP32 with `opt_level=0` for sklearn/NN, immutable versioned artifacts
carrying version, I/O schema, feature-spec reference and git SHA — and
[ADR-0005](0005-fp32-no-quantization.md) added the demand that every export path be *round-trip
tested for silent FP16 downcasting*. Neither existed in code: `axon.models.export` raised
`NotImplementedError`, and `ArtifactMeta` named five fields that nothing filled in. This ADR closes
the Phase-5 roadmap item "`axon.models`: export (ONNX / native tree) + versioned registry".

The reason this matters more than it looks: the signal record carries a `model_version` and
**nothing else about the model** ([ADR-0006](0006-signal-schema-and-spsc-ring.md)). Every claim in
[07](../07-parity-and-testing.md) — shadow trading, post-hoc audits, replaying a live session
offline — reduces to the assumption that this one integer resolves, years later, to one exact set
of bytes. Whether that assumption holds is decided entirely here.

Five questions had to be answered, and each has a reasonable-sounding wrong answer:

1. **One format for everything, or one per family?** "Convert everything to ONNX" is the tidy
   answer and it is the one that costs money.
2. **What does "immutable" mean operationally?** Re-saving a version could be an overwrite, an
   error, or a silent no-op when the bytes match.
3. **What actually proves an export did not change the model?** A structural check looks
   sufficient. It is not.
4. **What must the metadata bind, and who fills it in?** A field the caller types by hand is a
   field that records what the caller believed.
5. **What is "latest"?** The newest file is the intuitive answer and it is wrong.

## Decision

### 1. Trees keep their own serializer; everything else goes to ONNX at FP32

| family | artifact | how |
|---|---|---|
| XGBoost | `model.json` | `Booster.save_raw(raw_format="json")` |
| LightGBM | `model.txt` | `Booster.model_to_string()` |
| sklearn (incl. `MLP*`) | `model.onnx` | `skl2onnx`, one `float32` input named `input`, ZipMap off |
| Torch / anything else | `model.onnx` | caller supplies the `ModelProto` **and** a reference callable |

**Trees are not converted, because conversion is not free.** A tree ensemble re-expressed as an
ONNX graph is a different program with its own float handling, and a 1e-7 disagreement at a split
threshold does not shift a prediction by 1e-7 — it sends the sample down the other branch. The
library reading back its own file is bit-exact, and the exporter *asserts* bit-exactness (§2)
rather than assuming it.

**LightGBM's artifact is its text model, not JSON — this amends ADR-0003's wording.** LightGBM's
`dump_model()` produces JSON, but the library provides no loader for it: the JSON is a one-way
inspection format. A JSON LightGBM artifact would be an artifact that cannot be served, which is a
worse outcome than an artifact that is not JSON. ADR-0003's intent was *the library's own exact
format*; for LightGBM that is the text string.

**ZipMap is disabled for classifiers.** Left on, skl2onnx returns probabilities as a *list of
dicts*. That is not a tensor: it cannot be compared numerically here, and `ort` on the Rust side
cannot consume it at all — a Boundary-A migration would discover this after the model had been in
production for months.

**We never import torch.** A Torch strategy exports its own graph with `torch.onnx.export` and
hands us the proto; the registry stays free of a multi-gigabyte dependency that only one family
needs.

### 2. Verification is part of the export, not a step someone remembers

`export_artifact(model, meta, sample_input)` **requires** `sample_input`, and that argument does
two jobs: it is where the input schema comes from, and it is the evidence. An export nobody re-ran
is an export nobody checked, so the registry refuses an artifact whose metadata does not record a
round trip.

The round trip is literal: serialize, re-load **from the serialized bytes**, re-run, compare.

- **Trees are compared at zero tolerance,** and the `tolerance` argument cannot loosen that. The
  same library reading back its own file has no excuse for a difference; if it produces one, the
  format is lossy and the artifact is not the model.
- **ONNX is compared at 1e-5** (ADR-0003's starting ε) plus a class-flip check: no row may change
  its `argmax`. The measured deviation is **recorded** in the metadata as a number, not collapsed
  to a boolean, because "verified" without a magnitude tells a future reader nothing.
- **The reference comes from the wrapper, never from the booster underneath it.** An early-stopped
  `XGBRegressor` predicts with `best_iteration` trees while a bare `Booster` uses all of them.
  Comparing booster to booster would pronounce that artifact identical while it silently serves a
  model nobody validated.
- **The sample is cast to float32 first**, because that is what serving feeds. Verifying against a
  float64 reference measures a model that never runs and buries a real downcast in the noise.

**FP16 is checked three times, because the three checks catch different things.** A structural
audit of the *written bytes* walks initializers, value types, node attributes and **subgraphs** for
any reduced-precision tensor, any `Cast` into one, and any quantization operator — subgraphs
because an `If` branch is exactly where a conversion tool puts the half-precision path, and an
audit that stops at the top level calls that graph clean. Then the runtime's output dtype is
checked, because the graph can be FP32 and an execution provider can still hand back half precision
(ADR-0005's CoreML footgun) — no structural check can see that. Then the numbers themselves are
compared. The set of reduced-precision dtypes is *derived from the installed onnx* rather than
hardcoded, so the next low-precision type the standard adds (the FP8 and FP4 families arrived this
way) is refused on the day it appears rather than the day someone updates a list.

**Scope, stated plainly:** this proves the artifact reproduces the model. It is *not* the
model-parity gate of ADR-0003 §3, which compares Python against Rust over a research corpus and
enforces decision invariance against the strategy's own thresholds. The class-flip check here is
the only form of decision invariance available at export time, because the export does not know
what the strategy will do with the score.

### 3. Immutability is enforced by the filesystem, not by convention

```
<root>/<registry_id>/v0000000007/model.onnx   ← the bytes
                                /meta.json     ← the record
```

Writing an existing `(registry_id, version)` **raises**. It is not an overwrite, and it is not a
no-op even when the payload is byte-identical: "identical" is only knowable after comparing, the
metadata beside it (feature-spec ref, git SHA) may well differ, and a silent no-op teaches every
caller that re-saving is safe. Version numbers are free; a mutable version makes every decision it
ever produced unreproducible.

The write is staged in a sibling directory and moved in with a single `rename`. That is what makes
the guarantee real rather than best-effort: a crash mid-write leaves a staging directory that
`list` never sees, instead of a half-written version that can never be rewritten, and the rename —
not the existence check before it — is what makes two concurrent exporters unable to both win.

Two constraints come from outside this module. `version` must fit `1..=u32::MAX`, because the wire
field is a `u32` and a version the signal cannot carry is a version no fill can be traced back to;
`0` is refused as indistinguishable from unset. `registry_id` is restricted to lowercase
`[a-z0-9._-]`, because it becomes a directory name — a `/` or `..` turns `load()` into a file read
of the caller's choosing, and two ids differing only in case silently merge on a case-insensitive
filesystem.

### 4. The metadata is a record of what happened, and the loader enforces it

Beyond ADR-0003's five fields, each artifact records its input/output `TensorSpec`s, which output
carries the score, the producing library versions and opset, the payload's SHA-256 and length, the
measured round-trip deviation, and a creation timestamp. Everything except the caller's five
declarations is filled in by the exporter from the artifact it actually produced.

- **`load()` re-hashes the payload and refuses a mismatch**, and there is deliberately no
  `verify=False`: a loader that can be asked to skip the check will be asked to skip it, by someone
  debugging in a hurry.
- **The path is a claim; the metadata is the record.** If they disagree the load fails. `cp -r v1
  v2` is how one model comes to answer to two version numbers, and after that a replay of v2 quietly
  runs v1.
- **`feature_spec_ref` is mandatory.** A model without the features that fed it is not
  reproducible, and feature parity is the harder half of the problem ([03](../03-ml-fidelity-and-features.md)).
  The convention is `axon.features.FeatureSpec.ref` — `name/vN#fingerprint`, which pins the
  transforms and not merely their names.
- **`git_sha` is filled automatically and carries a `-dirty` suffix** when the tree is not clean. A
  clean-looking SHA on a model trained from uncommitted code is a reproduction instruction that
  rebuilds a *different* model. Outside a checkout it records `"unknown"`, which is honest; refusing
  to export from a notebook would only push people around the registry.
- **`score_output` names the output a strategy acts on.** An skl2onnx classifier returns `label`
  and `probabilities`; a consumer that takes the first output trades on a class index.
- **`meta_schema` is versioned and unknown keys are refused** (the same rule as
  `StrategyConfig.from_dict`), so an older Axon rejects a newer artifact instead of reading half of
  it and treating the rest as absent.

### 5. Latest is the highest version, never the newest file

A registry restored from backup has fresh mtimes and unchanged versions, so ordering by time hands
out whatever was copied last. Version directories are zero-padded to ten digits — the width of a
`u32` — so that `ls` agrees with the numeric sort a human is already assuming, and so a version
cannot be spelled two ways. A directory carrying no `meta.json` is an interrupted write and is not
a version: it is excluded from listings and named explicitly when someone asks for it, because
"why is my export refused?" and "delete this directory" need to be the same sentence.

### 6. Inference settings are chosen for reproducibility, not throughput

Sessions run with **graph optimization disabled** (a fused op is better, and different, so an
optimized graph is no longer the graph the parity gate compared), **one thread** (FP addition is
not associative, so parallel reductions let a session disagree with itself between runs), and the
**CPU execution provider pinned** (the silent-downcast footgun lives in the accelerated providers).
Inference here is offline and off the microsecond path ([05](../05-latency-model.md)), so nothing is
being traded away.

## Consequences

- **+** A `model_version` on a signal now resolves to one exact set of bytes, with the feature spec
  and commit that produced them. That is the precondition the entire Phase-5/6 ladder assumes.
- **+** FP16 and quantization cannot enter by accident, including through a subgraph, and the
  failure is an exception at export rather than a slow drift in fill quality.
- **+** The registry is a directory tree: it rsyncs, it inspects with `ls`, it needs no daemon, and
  the Rust startup path loads an artifact by opening a file.
- **+** Export refuses instead of warning. A tree that does not round-trip exactly, a sample with a
  NaN in it, a mislabeled `kind`, a graph with two inputs — all are errors at the only moment when
  someone is still looking.
- **−** A hash is not a signature. It catches truncation, bit rot and accidental edits; an attacker
  who can rewrite `model.onnx` can rewrite `meta.json` beside it. Tamper-evidence needs signing,
  which we do not have.
- **−** Verification is pinned to the exporting machine's library versions. An artifact re-loaded in
  two years under a newer XGBoost is not re-verified; `producer` records what to reproduce, but no
  re-verification harness exists yet.
- **−** `sample_input` being mandatory means a model shipped as a bare file cannot be registered
  without a reference callable. That is the intended friction, and it is still friction.
- **−** No retention policy. Versions accumulate forever by design, and nothing prunes them; a
  strategy retrained nightly will need a garbage-collection decision this ADR does not make.
- **−** Atomicity rests on `rename` within one filesystem. On a network filesystem that does not
  provide it, two concurrent exporters could both believe they won.
- **−** One input tensor only. A multi-input graph is refused rather than fed by a guessed argument
  order, but a model that genuinely needs two inputs has no path today.
- **−** `score_output` is chosen by skl2onnx's naming convention, falling back to the first float
  output. A converter that names things differently will pick the right tensor by luck.

See [ADR-0003](0003-model-serving-and-fidelity.md) (the policy this implements, whose "native JSON"
wording §1 amends for LightGBM), [ADR-0005](0005-fp32-no-quantization.md) (FP32 and the round-trip
demand), [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the `u32 model_version` that bounds a
version), [03](../03-ml-fidelity-and-features.md), [07](../07-parity-and-testing.md).
