"""The Rust half of the model-parity gate: a frozen question Rust can answer offline.

:mod:`axon.parity.model` compares one set of Python predictions against another. That
is a real gate and it is not the one Boundary A turns on. The claim that decides
whether inference may move into the Rust core (``docs/02``, ADR-0019) is
cross-language: **the model Rust will serve produces the same decisions as the model
Python researched with.** No Python-to-Python comparison can fail on the way those two
differ — a different float width at a split threshold, a link applied on one side
only, a runtime that reassociates a reduction.

A *parity bundle* is that question, written out of a registry artifact (ADR-0015) into
a directory a Rust test reads with no Python, no ML libraries, no network and no clock::

    <bundle>/manifest.json     what this is, and what it must be held to
            /model.json        the artifact's own bytes, straight off the registry
            /features.f32      the holdout matrix, raw little-endian IEEE-754
            /predictions.f32   Python's own answer over those exact bytes
            /decisions.i8      Python's own discretized decision per row

**Why the matrices are raw f32 rather than JSON numbers.** The gate only means
anything if both languages score the same inputs, and "the same" has to mean the same
*bits*: a feature written as a decimal and re-parsed can land on the neighbouring
float, which sends a row down the other side of a split, and the gate then reports the
serialization as a model defect. So the matrix crosses as raw little-endian f32 — cast
once, written once — and, decisively, :func:`write_parity_bundle` takes the reference
predictions **over the bytes it read back from the file**, not over the array that
produced them. The recorded answer is by construction the answer to the question the
Rust side asks.

Byte order is pinned (``<f4``) rather than native, because the reader is
``f32::from_le_bytes``: a native-order write on a big-endian host would hand Rust every
feature byte-swapped, and a perfectly correct model would fail the gate spectacularly.
NaN travels as its own bits too, which is what a JSON ``null`` cannot do — the tree
backend's missing-value branch is gated on exactly those rows.

**Why the decisions are recorded and not just derived.** Both sides discretize the
score into ``{-1 short, 0 flat, +1 long}``. If the two languages disagree about the
*rule* — ``>=`` against ``>``, a threshold rounded to a different float — every
prediction can match to the bit while the two systems still trade differently. The
bundle carries Python's decisions so the Rust reader can re-derive them and refuse the
bundle before a model is even loaded. For the same reason the thresholds cross as bit
patterns, not decimals.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from axon.models.artifact import Artifact, ArtifactMeta, content_hash
from axon.parity.model import Discretizer, ModelParityReport, model_parity

#: Bumped when the on-disk layout changes. A bundle stamped higher than this is
#: refused rather than read with the unknown half treated as absent — the same rule
#: ``ArtifactMeta.meta_schema`` follows, and the Rust reader enforces it too.
BUNDLE_SCHEMA = 1

MANIFEST_FILENAME = "manifest.json"
FEATURES_FILENAME = "features.f32"
PREDICTIONS_FILENAME = "predictions.f32"
DECISIONS_FILENAME = "decisions.i8"

#: ADR-0003 §3's starting tolerance for graphs, and the **ceiling** a bundle may declare.
#: This is what :meth:`Criterion.required_for` demands and what both readers refuse to see
#: exceeded. It is deliberately *not* what a bundle written here declares — see
#: :data:`ONNX_TIGHT_EPS`.
ONNX_EPS = 1e-5

#: What a graph bundle written by this repo actually declares: **two ULP at 1.0**, which is
#: ``2**-22``.
#:
#: ``ONNX_EPS`` is a ceiling for the *family*, and every graph this repo has ever gated
#: sits between four and five orders of magnitude inside it — measured, not assumed:
#: ``lgbm_binary`` 1.1920929e-7 (one ULP at 1.0, exactly), ``zoo_logistic`` 8.940697e-8,
#: ``mlp_regressor`` 0e0. A bundle declaring 1e-5 while achieving 1e-7 is a gate with a
#: hundredfold of slack in it, and slack is where a regression passes green: the whole
#: argument for pinning ``tract`` in ADR-0019 §1 is that *a silent patch bump could move a
#: result inside the tolerance with no code change to blame it on*. Declaring the ceiling
#: after making that argument is declining to act on it.
#:
#: **Two ULP, and not the measurement.** The value is anchored to float32's own resolution
#: rather than fitted to what today's runtime produced, because a tolerance derived from a
#: measurement is a tolerance that ratchets: regenerate after a regression and it records
#: the regression as the new bar. That is precisely the failure :meth:`Criterion.allows`
#: exists to catch, and auto-fitting would institutionalize it one level up. Two ULP is
#: the smallest round number that clears every graph measured here with margin, and a
#: graph that cannot meet it is telling you something worth reading rather than something
#: to widen the constant for.
#:
#: Not applied to ``mlp_regressor``'s observed ``0e0`` as ``bit_exact``: ONNX does not
#: encode operator ordering and float addition is not associative, so one machine's exact
#: agreement is luck rather than a property, and a criterion fitted to luck reddens on a
#: CPU with different FMA behaviour for no defect at all.
ONNX_TIGHT_EPS = 2.0**-22

#: What each family's reference is *in*. ``TreeModel`` returns the raw margin and never
#: applies the link (ADR-0019), so a tree bundle holding probabilities would compare a
#: probability against a margin and report the link as a parity failure.
#:
#: ``"score"`` means *whatever the graph's own score output emits*, and for a graph the
#: only honest answer is that: an MLP regressor emits a value, a converted LightGBM
#: binary classifier emits a probability because it carries LightGBM's own
#: ``post_transform=LOGISTIC``. That last case has no margin/probability trap only
#: because neither side skips the link — ``LightgbmPredictor`` calls ``Booster.predict``
#: with the default ``raw_score=False``. Compare one of those graphs against
#: ``raw_score=True`` and the sigmoid is reported as a model defect (ADR-0033).
SCORE_SPACE = {"xgboost": "margin", "onnx": "score"}

#: Artifact kinds a Rust backend serves. The ``lightgbm`` kind is the booster's own
#: *text* format, and no Rust reader parses it — ADR-0019's documented gap, named here
#: rather than reported as an unknown kind. It is deliberately still absent after
#: ADR-0033: the fix for a LightGBM model that must reach Rust is to convert it with
#: ``axon.models.export.lightgbm_to_onnx`` and register it as an ``onnx`` artifact,
#: which this tuple already admits. Adding ``"lightgbm"`` here would not make anything
#: able to serve it; it would only move the failure from bundle-write time to Rust
#: load time, after the artifact has a version a signal can refer to.
SERVABLE_KINDS = ("xgboost", "onnx")

#: Appended to every "no Rust backend serves this" refusal, because the refusal is now
#: a signpost rather than a dead end and a caller who only reads the first clause will
#: reach for a native backend that does not need building.
_LIGHTGBM_ROUTE = (
    "a LightGBM booster crosses by conversion, not by kind: see "
    "axon.models.export.lightgbm_to_onnx (ADR-0033)"
)


class BundleError(Exception):
    """A parity bundle could not be written, or could not be trusted once read."""


# ── the criterion ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Criterion:
    """The bar a candidate's predictions are held to.

    ``kind="bit_exact"`` for trees, because ADR-0019 claims `TreeModel` reproduces
    ``Booster.predict(output_margin=True)`` exactly and a tolerance there would be the
    gate declining to test its own claim. ``kind="max_abs_diff"`` for graphs, because
    two runtimes never agree bit for bit — ONNX does not encode operator ordering and
    float addition is not associative.
    """

    kind: str
    eps: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("bit_exact", "max_abs_diff"):
            raise BundleError(f"unknown parity criterion {self.kind!r}")
        if self.kind == "max_abs_diff" and not self.eps > 0.0:
            raise BundleError(f"max_abs_diff needs a positive eps, got {self.eps}")

    @classmethod
    def required_for(cls, kind: str) -> "Criterion":
        """The **ceiling** the family imposes, whatever a manifest asks for.

        This is the bar a *reader* enforces, and it is not the bar a writer declares —
        see :meth:`declared_for` and :data:`ONNX_TIGHT_EPS`. Keeping the two apart is what
        lets a bundle be held to a tighter number than the family requires without any
        bundle being able to buy itself a looser one.
        """
        if kind == "xgboost":
            return cls("bit_exact")
        if kind == "onnx":
            return cls("max_abs_diff", ONNX_EPS)
        raise BundleError(
            f"no Rust backend serves {kind!r} artifacts; this gate covers {SERVABLE_KINDS} — "
            f"{_LIGHTGBM_ROUTE}"
        )

    @classmethod
    def declared_for(cls, kind: str) -> "Criterion":
        """What a bundle written *here* declares — as strict as this repo can hold.

        Trees are bit-exact because ADR-0019 claims `TreeModel` reproduces
        ``Booster.predict(output_margin=True)`` exactly and a tolerance would be the gate
        declining to test its own claim. Graphs get :data:`ONNX_TIGHT_EPS` rather than the
        family ceiling, because every graph gated here sits four to five orders of
        magnitude inside that ceiling and the slack is where a silent regression lives.
        """
        if kind == "onnx":
            return cls("max_abs_diff", ONNX_TIGHT_EPS)
        return cls.required_for(kind)

    def allows(self, declared: "Criterion") -> bool:
        """Whether ``declared`` is at least as strict as this one.

        Tightening is allowed. Loosening is the failure this exists for: a bundle
        regenerated after a red gate with the tolerance nudged until it passed would
        otherwise be indistinguishable from one that never failed.
        """
        if declared.kind == "bit_exact":
            return True
        if self.kind == "bit_exact":
            return False
        return declared.eps <= self.eps

    @property
    def numeric_eps(self) -> float:
        """The tolerance to hand :func:`axon.parity.model_parity`.

        ``0.0`` for the exact families, which is ``TREE_EPS`` — the numeric shadow of
        the Rust side's bit comparison. It is deliberately weaker: ``0.0`` accepts
        ``+0.0`` against ``-0.0`` and the bit check does not.
        """
        return 0.0 if self.kind == "bit_exact" else float(self.eps)

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "bit_exact":
            return {"kind": self.kind}
        return {"kind": self.kind, "eps": self.eps}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Criterion":
        kind = str(data.get("kind", ""))
        return cls(kind, float(data["eps"])) if kind == "max_abs_diff" else cls(kind)

    def __str__(self) -> str:
        return "bit-exact" if self.kind == "bit_exact" else f"max_abs_diff <= {self.eps:g}"


# ── the decision rule ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    """A discretized trading decision as two thresholds on the score.

    The same rule as :func:`axon.parity.threshold_discretizer`: ``>= long_at`` is long,
    ``<= short_at`` is short, between is flat, and a NaN decides flat because every
    comparison against it is false. That last case is why a non-finite reference is
    refused outright rather than left to the decision check — two NaNs agree on "flat"
    and would pass.

    Both thresholds are **rounded to float32 on construction**, because they cross to
    Rust as 32-bit patterns. Discretizing here on a float64 threshold and there on its
    f32 neighbour puts the two languages' decision boundaries a ULP apart, and every
    score in between decides differently on each side — a disagreement manufactured by
    the gate itself.
    """

    long_at: float
    short_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "long_at", float(np.float32(self.long_at)))
        object.__setattr__(self, "short_at", float(np.float32(self.short_at)))
        if not np.isfinite([self.long_at, self.short_at]).all():
            raise BundleError(
                f"decision thresholds must be finite, got short_at={self.short_at} "
                f"long_at={self.long_at}"
            )
        if not self.short_at < self.long_at:
            raise BundleError(
                f"need short_at < long_at, got short_at={self.short_at} long_at={self.long_at}"
                " — an inverted pair decides every row twice over"
            )

    def sides(self, scores: Any) -> np.ndarray:
        """Scores → ``int8`` decisions in ``{-1, 0, +1}``."""
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        out = np.zeros(s.shape, dtype=np.int8)
        out[s >= self.long_at] = 1
        out[s <= self.short_at] = -1
        return out

    def discretizer(self) -> Discretizer:
        """This rule as something :func:`axon.parity.model_parity` accepts."""
        return self.sides

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_at_bits": _f32_bits(self.long_at),
            "short_at_bits": _f32_bits(self.short_at),
            # For humans reading the manifest and for nothing else: both readers
            # take the bits, so a hand-edited decimal here changes no decision.
            "long_at_approx": self.long_at,
            "short_at_approx": self.short_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Decision":
        return cls(
            long_at=_from_f32_bits(str(data["long_at_bits"])),
            short_at=_from_f32_bits(str(data["short_at_bits"])),
        )


def quantile_decision(lower: float = 0.4, upper: float = 0.6) -> Callable[[np.ndarray], Decision]:
    """Thresholds placed at two quantiles of the reference scores.

    Handed to :func:`write_parity_bundle` when the caller has no strategy thresholds to
    impose. Quantiles rather than fixed numbers because the decision half of the gate
    is only a test if the corpus actually decides more than one way: a threshold above
    every score in the holdout makes every row flat, and a check that can only come out
    one way is a decoration.
    """
    if not 0.0 < lower < upper < 1.0:
        raise BundleError(f"need 0 < lower < upper < 1, got lower={lower} upper={upper}")

    def choose(scores: np.ndarray) -> Decision:
        ref = np.asarray(scores, dtype=np.float64)
        short_at, long_at = np.quantile(ref, [lower, upper])
        if not np.float32(short_at) < np.float32(long_at):
            raise BundleError(
                f"quantiles {lower} and {upper} of the reference collapse to the same float32 "
                f"({short_at}); the holdout does not spread enough to gate a decision on"
            )
        long_at = _off_the_score_grid(long_at, ref, downward=True)
        short_at = _off_the_score_grid(short_at, ref, downward=False)
        if not np.float32(short_at) < np.float32(long_at):
            raise BundleError(
                f"separating the thresholds from the score grid collapsed them "
                f"(short_at={short_at} long_at={long_at}); the holdout is too dense to gate on"
            )
        return Decision(long_at=float(long_at), short_at=float(short_at))

    return choose


def _off_the_score_grid(threshold: float, scores: np.ndarray, *, downward: bool) -> float:
    """Move *threshold* off any score it lands exactly on, without moving a decision.

    A quantile of the reference is, by construction, often *equal* to one of the scores
    it was taken over — and with duplicates in the holdout, equal to many of them. That
    is a threshold fitted to luck in the sense ADR-0021 refuses for the criterion: the
    rule is ``>= long_at``, so a row sitting exactly on it decides long only while the
    candidate reproduces the reference's last bit. The ONNX backend does not promise
    that bit (this module's own note on ``expf``, and tract selects a 16-wide sigmoid
    kernel on an AVX-512 host and an 8-wide one elsewhere), so such a threshold reddens
    the gate on a different CPU for no defect at all — which is exactly what it did.

    The fix is to place the threshold in the *gap* rather than on the grid: halfway to
    the nearest distinct score on the side the rule does not include. Every decision is
    preserved — no achievable score lies between the old threshold and the new one — and
    the margin becomes the width of that gap instead of zero, so a last-bit difference
    can no longer cross it.
    """
    grid = np.unique(np.asarray(scores, dtype=np.float32))
    here = np.float32(threshold)
    if not np.any(grid == here):
        return float(here)  # already in a gap
    neighbours = grid[grid < here] if downward else grid[grid > here]
    if neighbours.size == 0:
        # No score on that side: step one ULP away, which is all the room there is.
        return float(np.nextafter(here, np.float32(-np.inf if downward else np.inf)))
    nearest = neighbours.max() if downward else neighbours.min()
    return float(np.float32((np.float64(nearest) + np.float64(here)) / 2.0))


# ── the reference ─────────────────────────────────────────────────────────────


def python_scores(artifact: Artifact, features: np.ndarray) -> np.ndarray:
    """Python's own score per row, in the space the Rust backend serves.

    Two deliberate asymmetries between the families:

    * **XGBoost is scored as a margin**, via ``output_margin=True``, because that is
      what ``TreeModel`` returns: it never applies the link (the link is monotone, so a
      decision threshold on the probability is a decision threshold on the margin, and
      skipping it keeps ``expf`` — whose last bit is not portable — off the serving
      path). Scoring the probability here would compare the *link* rather than the
      model. The whole matrix goes through at once: tree traversal is row-independent,
      so the batch shape cannot change an answer.
    * **ONNX is scored one row at a time**, because the Rust plan pins its batch
      dimension to 1. A graph scored 128 rows at once may reassociate a reduction that
      a batch of one cannot, and the gate would then be measuring the batch shape.
    """
    kind = artifact.meta.kind
    if kind == "xgboost":
        import xgboost

        booster = xgboost.Booster()
        # From the bytes, not from a path: the artifact under test is the payload the
        # registry verified, never a file someone re-saved beside it.
        booster.load_model(bytearray(artifact.payload))
        matrix = xgboost.DMatrix(features, missing=np.nan)
        margin = booster.predict(matrix, output_margin=True)
        return np.asarray(margin, dtype=np.float32).reshape(-1)

    if kind == "onnx":
        from axon.models.inference import load_predictor

        predictor = load_predictor(artifact)
        rows = []
        for row in features:
            out = np.asarray(predictor.predict(row.reshape(1, -1))).reshape(-1)
            if out.size != 1:
                # Both Rust backends serve exactly one score, and a threshold is
                # defined on one number. Picking a column here is the guess
                # ADR-0015's `score_output` exists to prevent.
                raise BundleError(
                    f"{artifact.ref}: the score output has {out.size} values per row; a parity "
                    "bundle records one score per row"
                )
            rows.append(out[0])
        return np.asarray(rows, dtype=np.float32)

    raise BundleError(
        f"{artifact.ref}: no Rust backend serves {kind!r} artifacts "
        f"(ADR-0019 leaves LightGBM and Torch unbuilt); this gate covers {SERVABLE_KINDS} — "
        f"{_LIGHTGBM_ROUTE}"
    )


# ── writing ───────────────────────────────────────────────────────────────────


def write_parity_bundle(
    artifact: Artifact,
    features: Any,
    *,
    out_dir: str | os.PathLike[str],
    decision: Decision | Callable[[np.ndarray], Decision],
    criterion: Criterion | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the cross-language question for ``artifact`` over ``features``.

    ``features`` is a 2-D holdout matrix in feature-spec order. ``decision`` is the
    strategy's discretization, or a callable picking it from the reference scores (see
    :func:`quantile_decision`); it is **required**, because the numeric criterion alone
    has passed every model that ever silently moved a position (ADR-0016 §3).

    ``criterion`` defaults to :meth:`Criterion.declared_for`, which for a graph is
    :data:`ONNX_TIGHT_EPS` — two ULP — rather than the family ceiling. Passing one
    explicitly may only ever **tighten**: anything looser than
    :meth:`Criterion.required_for` is refused here, so the writer still has no way to emit
    a loosened gate and a bundle asking for one was edited by hand. Note what that does
    *not* prevent, and read it before widening anything: an explicit ``criterion`` at a
    call site can walk a bundle back from two ULP to the family's 1e-5 and the writer will
    accept it, because that is inside the family. Change :data:`ONNX_TIGHT_EPS` instead, in
    one reviewable line, so the loosening is a diff somebody sees rather than an argument
    somebody passed.

    Returns the bundle directory. The manifest is written **last**, so an interrupted
    run leaves a directory the reader reports as an interrupted write rather than a
    bundle whose recorded hashes describe files that are no longer there.
    """
    artifact.verify()
    kind = artifact.meta.kind
    if kind not in SERVABLE_KINDS:
        raise BundleError(
            f"{artifact.ref}: no Rust backend serves {kind!r} artifacts; this gate covers "
            f"{SERVABLE_KINDS} — {_LIGHTGBM_ROUTE}"
        )

    # Resolved before a byte is written: a bundle whose matrices exist and whose criterion
    # was then refused is the interrupted-write state the manifest-last rule exists to make
    # legible, and there is no reason to manufacture one over an argument.
    declared = Criterion.declared_for(kind) if criterion is None else criterion
    if not isinstance(declared, Criterion):
        raise BundleError(f"criterion must be a Criterion, got {type(declared).__name__}")
    required = Criterion.required_for(kind)
    if not required.allows(declared):
        raise BundleError(
            f"{artifact.ref}: asked to declare {declared} on a {kind} artifact, which is held to "
            f"{required}; a bundle may tighten its criterion and may never loosen it"
        )

    x = _holdout(features, kind=kind)
    directory = Path(out_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise BundleError(
            f"{manifest_path} already exists; regenerating a committed bundle is a deliberate, "
            "reviewable event — pass overwrite=True to say so"
        )
    directory.mkdir(parents=True, exist_ok=True)
    # Off first: a stale manifest beside fresh matrices describes bytes that no longer
    # exist, and its recorded hashes would be the only thing to notice.
    manifest_path.unlink(missing_ok=True)

    features_path = directory / FEATURES_FILENAME
    features_bytes = _f32_bytes(x)
    features_path.write_bytes(features_bytes)

    # The one line that makes the reference answer *this* question: score the matrix
    # read back off disk, so the recorded predictions are the predictions over exactly
    # the bytes Rust will read. Casting and scoring the in-memory array instead would
    # leave the file unverified in the one respect that matters.
    scored = _read_f32(features_path, x.shape[0], x.shape[1])
    if scored.tobytes() != features_bytes:
        raise BundleError(f"{features_path}: the matrix did not survive its own round trip")

    predictions = np.asarray(python_scores(artifact, scored), dtype=np.float32).reshape(-1)
    if predictions.shape[0] != x.shape[0]:
        raise BundleError(
            f"{artifact.ref}: scored {predictions.shape[0]} rows of a {x.shape[0]}-row holdout"
        )
    if not np.isfinite(predictions).all():
        # A non-finite reference is never compared, on either side: `nan > eps` is
        # False, so a naive gate would call it a perfect match forever.
        bad = int(np.flatnonzero(~np.isfinite(predictions))[0])
        raise BundleError(
            f"{artifact.ref}: row {bad} scores {predictions[bad]}; a non-finite reference makes "
            "every comparison against it vacuous"
        )

    rule = decision(predictions) if callable(decision) else decision
    if not isinstance(rule, Decision):
        raise BundleError(f"decision must be a Decision, got {type(rule).__name__}")
    sides = rule.sides(predictions)
    if len(set(sides.tolist())) < 2:
        raise BundleError(
            f"{artifact.ref}: every row decides {int(sides[0]):+d} under long_at={rule.long_at} "
            f"short_at={rule.short_at}; a decision-invariance check that can only come out one "
            "way is a decoration, not a gate"
        )

    predictions_path = directory / PREDICTIONS_FILENAME
    predictions_bytes = _f32_bytes(predictions.reshape(-1, 1))
    predictions_path.write_bytes(predictions_bytes)
    decisions_path = directory / DECISIONS_FILENAME
    decisions_bytes = sides.astype(np.int8).tobytes()
    decisions_path.write_bytes(decisions_bytes)

    artifact_path = directory / artifact.meta.artifact_filename
    artifact_path.write_bytes(artifact.payload)

    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "note": "Written by axon.parity.rust_gate; regenerate rather than hand-edit.",
        "registry_id": artifact.meta.registry_id,
        "model_version": artifact.meta.version,
        "kind": kind,
        "score_space": SCORE_SPACE[kind],
        "feature_spec_ref": artifact.meta.feature_spec_ref,
        "git_sha": artifact.meta.git_sha,
        "artifact": {
            "file": artifact.meta.artifact_filename,
            "sha256": artifact.meta.content_sha256,
            "bytes": len(artifact.payload),
        },
        "features": {
            "file": FEATURES_FILENAME,
            "sha256": content_hash(features_bytes),
            "bytes": len(features_bytes),
            "rows": int(x.shape[0]),
            "cols": int(x.shape[1]),
            "missing_cells": int(np.count_nonzero(np.isnan(x))),
        },
        "predictions": {
            "file": PREDICTIONS_FILENAME,
            "sha256": content_hash(predictions_bytes),
            "bytes": len(predictions_bytes),
            "rows": int(predictions.shape[0]),
            "cols": 1,
        },
        "decisions": {
            "file": DECISIONS_FILENAME,
            "sha256": content_hash(decisions_bytes),
            "bytes": len(decisions_bytes),
            "rows": int(sides.shape[0]),
            "counts": {
                "short": int(np.count_nonzero(sides == -1)),
                "flat": int(np.count_nonzero(sides == 0)),
                "long": int(np.count_nonzero(sides == 1)),
            },
        },
        # As strict as this repo can hold, never looser than the family allows: the
        # writer refuses a loosened criterion above, so a bundle asking for one was
        # edited by hand and both readers say so.
        "criterion": declared.to_dict(),
        "decision": rule.to_dict(),
        # What *scored the reference*, which is what a frozen answer depends on. The
        # artifact's own `producer` records what exported it, and the two can differ.
        "producer": _producer(kind),
        "exported_by": dict(artifact.meta.producer),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )
    return directory


def write_bundle_from_registry(
    registry: Any,
    registry_id: str,
    version: int | None = None,
    *,
    features: Any,
    out_dir: str | os.PathLike[str],
    decision: Decision | Callable[[np.ndarray], Decision],
    criterion: Criterion | None = None,
    overwrite: bool = False,
) -> Path:
    """:func:`write_parity_bundle` over ``registry.load(registry_id, version)``.

    The bundle is taken from the registry rather than from the in-memory model on
    purpose: the question is about the bytes that will be served, and the registry is
    the only thing that can say which bytes a ``model_version`` names (ADR-0015).
    """
    return write_parity_bundle(
        registry.load(registry_id, version),
        features,
        out_dir=out_dir,
        decision=decision,
        criterion=criterion,
        overwrite=overwrite,
    )


# ── reading ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParityBundle:
    """A parity bundle read back and checked against itself."""

    path: Path
    manifest: Mapping[str, Any]
    features: np.ndarray
    predictions: np.ndarray
    decisions: np.ndarray
    criterion: Criterion
    decision: Decision

    @property
    def kind(self) -> str:
        return str(self.manifest["kind"])

    @property
    def registry_id(self) -> str:
        return str(self.manifest["registry_id"])

    @property
    def model_version(self) -> int:
        return int(self.manifest["model_version"])

    @property
    def feature_spec_ref(self) -> str:
        return str(self.manifest["feature_spec_ref"])

    @property
    def artifact_path(self) -> Path:
        return self.path / str(self.manifest["artifact"]["file"])

    def artifact(self) -> Artifact:
        """The bundle's model, back in the shape :mod:`axon.models` speaks.

        Enough of the record to load and score it, not a substitute for the
        registry entry: a bundle is a frozen question, and the artifact it carries
        is a copy taken at the moment the question was asked.
        """
        entry = self.manifest["artifact"]
        meta = ArtifactMeta(
            registry_id=self.registry_id,
            version=self.model_version,
            kind=self.kind,
            feature_spec_ref=self.feature_spec_ref,
            git_sha=str(self.manifest.get("git_sha", "")),
            artifact_filename=str(entry["file"]),
            content_sha256=str(entry["sha256"]),
            content_bytes=int(entry["bytes"]),
            producer=dict(self.manifest.get("exported_by", {})),
        )
        return Artifact(meta=meta, payload=self.artifact_path.read_bytes())

    def compare(self, candidate: Any) -> ModelParityReport:
        """The bundle's own criterion and thresholds, as a model-parity report.

        The same report type the Python-to-Python gate returns, so a live parity
        monitor (``docs/07``) can hold a Rust-scored candidate to the identical
        standard it holds a Python one to.
        """
        return model_parity(
            self.predictions,
            candidate,
            discretizer=self.decision.discretizer(),
            eps=self.criterion.numeric_eps,
        )


def read_parity_bundle(path: str | os.PathLike[str]) -> ParityBundle:
    """Read a bundle, verifying everything checkable without a model.

    Content hashes are checked here because Python is the side that can cheaply say
    *why* a bundle is wrong. The Rust reader deliberately does not hash: a flipped bit
    in either matrix changes a prediction, so the gate itself catches it — a crypto
    dependency in the serving crate would buy a better error message and nothing else.
    """
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        if directory.is_dir():
            raise BundleError(
                f"{directory} has no {MANIFEST_FILENAME} — an interrupted write, since the "
                "manifest is written last; regenerate it deliberately"
            )
        raise BundleError(f"{directory}: no such parity bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = int(manifest.get("bundle_schema", 0))
    if schema > BUNDLE_SCHEMA:
        raise BundleError(
            f"{directory} is bundle schema {schema}, this Axon understands {BUNDLE_SCHEMA}; "
            "refusing rather than reading unknown fields as absent"
        )
    kind = str(manifest.get("kind", ""))
    if kind not in SERVABLE_KINDS:
        raise BundleError(
            f"{directory}: no Rust backend serves {kind!r} artifacts — {_LIGHTGBM_ROUTE}"
        )
    if manifest.get("score_space") != SCORE_SPACE[kind]:
        raise BundleError(
            f"{directory}: records {manifest.get('score_space')!r} scores, but a {kind} bundle "
            f"must record {SCORE_SPACE[kind]!r} — see python_scores"
        )
    if not str(manifest.get("feature_spec_ref", "")):
        raise BundleError(
            f"{directory}: carries no feature_spec_ref, so the corpus names no recipe"
        )

    for entry in ("artifact", "features", "predictions", "decisions", "criterion", "decision"):
        # Every failure out of this module is a BundleError, including "the manifest
        # is missing a section": a KeyError from here would read as a bug in the reader
        # rather than as a verdict on the bundle.
        if entry not in manifest:
            raise BundleError(f"{directory}: manifest has no {entry!r} section")
    for entry in ("artifact", "features", "predictions", "decisions"):
        _verify_file(directory, manifest[entry])

    rows = int(manifest["features"]["rows"])
    cols = int(manifest["features"]["cols"])
    if rows < 1 or cols < 1:
        raise BundleError(f"{directory}: a parity gate over {rows}x{cols} inputs proves nothing")
    if int(manifest["predictions"]["rows"]) != rows:
        raise BundleError(
            f"{directory}: {rows} feature rows against "
            f"{manifest['predictions']['rows']} prediction rows; the two sides did not see the "
            "same corpus"
        )
    if int(manifest["predictions"].get("cols", 1)) != 1:
        # Refused rather than sliced. A multi-class reference's first column is a
        # perfectly plausible number, and every other check here — the hashes, the
        # decisions, the counts — still passes against it, so a reader that quietly
        # took it would report a gate running on one class as healthy.
        raise BundleError(
            f"{directory}: the reference has {manifest['predictions']['cols']} columns; the gate "
            "discretizes one score per row, and picking one here would be an invented rule"
        )

    features = _read_f32(directory / manifest["features"]["file"], rows, cols)
    predictions = _read_f32(directory / manifest["predictions"]["file"], rows, 1).reshape(-1)
    decisions = np.frombuffer(
        (directory / manifest["decisions"]["file"]).read_bytes(), dtype=np.int8
    )
    if decisions.shape[0] != rows:
        raise BundleError(f"{directory}: {decisions.shape[0]} decisions for {rows} rows")

    missing = int(np.count_nonzero(np.isnan(features)))
    if missing != int(manifest["features"].get("missing_cells", -1)):
        raise BundleError(
            f"{directory}: manifest claims {manifest['features'].get('missing_cells')} missing "
            f"feature cells, the matrix holds {missing}"
        )
    if not np.isfinite(predictions).all():
        raise BundleError(f"{directory}: the recorded reference holds non-finite scores")

    criterion = Criterion.from_dict(manifest["criterion"])
    required = Criterion.required_for(kind)
    if not required.allows(criterion):
        raise BundleError(
            f"{directory}: manifest asks for {criterion} on a {kind} artifact, which is held to "
            f"{required}"
        )

    decision = Decision.from_dict(manifest["decision"])
    derived = decision.sides(predictions)
    disagree = np.flatnonzero(derived != decisions)
    if disagree.size:
        row = int(disagree[0])
        raise BundleError(
            f"{directory}: row {row} records decision {int(decisions[row]):+d} for score "
            f"{predictions[row]!r}, which discretizes to {int(derived[row]):+d} under the "
            "manifest's own thresholds — the decisions do not follow from the scores"
        )
    counts = manifest["decisions"]["counts"]
    actual = {
        "short": int(np.count_nonzero(decisions == -1)),
        "flat": int(np.count_nonzero(decisions == 0)),
        "long": int(np.count_nonzero(decisions == 1)),
    }
    if {k: int(v) for k, v in counts.items()} != actual:
        raise BundleError(f"{directory}: manifest decision counts {dict(counts)} != {actual}")

    return ParityBundle(
        path=directory,
        manifest=manifest,
        features=features,
        predictions=predictions,
        decisions=decisions,
        criterion=criterion,
        decision=decision,
    )


# ── bytes ─────────────────────────────────────────────────────────────────────


def _f32_bytes(matrix: np.ndarray) -> bytes:
    """A C-contiguous, little-endian float32 matrix as raw bytes."""
    return np.ascontiguousarray(np.asarray(matrix, dtype="<f4")).tobytes()


def _read_f32(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = path.read_bytes()
    want = rows * cols * 4
    if len(raw) != want:
        # A truncated matrix would otherwise read as a shorter corpus with a plausible
        # shape, and the gate would pass on the rows that survived.
        raise BundleError(
            f"{path}: holds {len(raw)} bytes; the manifest describes {rows}x{cols} f32 = {want}"
        )
    # `<f4` then `astype(float32)`: the values are unchanged (the cast is a byte-order
    # move, never a rounding), and downstream code gets a native, writable array.
    return np.frombuffer(raw, dtype="<f4").reshape(rows, cols).astype(np.float32)


def _verify_file(directory: Path, entry: Mapping[str, Any]) -> None:
    path = directory / str(entry["file"])
    if Path(entry["file"]).name != str(entry["file"]):
        # A manifest is data; a path in it that escapes the bundle directory turns
        # reading one into a file read of whoever wrote it choosing.
        raise BundleError(f"{directory}: {entry['file']!r} is not a bare filename")
    if not path.is_file():
        raise BundleError(f"{path}: named by the manifest and not present")
    payload = path.read_bytes()
    if len(payload) != int(entry["bytes"]):
        raise BundleError(
            f"{path}: {len(payload)} bytes, the manifest records {entry['bytes']} — truncated "
            "or replaced"
        )
    actual = content_hash(payload)
    if actual != str(entry["sha256"]):
        raise BundleError(
            f"{path}: content hash {actual} does not match the recorded {entry['sha256']}; these "
            "are not the bytes Python scored"
        )


def _holdout(features: Any, *, kind: str) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise BundleError(f"the holdout must be a non-empty 2-D matrix, got shape {x.shape}")
    if np.isinf(x).any():
        # NaN is a missing value; an infinity is not, and it propagates into a score
        # that neither side can compare.
        raise BundleError("the holdout contains infinities, which no backend treats as missing")
    if kind != "xgboost" and np.isnan(x).any():
        # Only the tree backend has a missing-value branch. A NaN into a graph comes
        # back as a NaN score, and a comparison of two NaNs is vacuous.
        raise BundleError(f"a {kind} holdout may not contain NaN; only trees route missing values")
    return x


def _f32_bits(value: float) -> str:
    return f"0x{int(np.float32(value).view(np.uint32)):08x}"


def _from_f32_bits(hex_bits: str) -> float:
    return float(np.uint32(int(hex_bits, 16)).view(np.float32))


def _producer(kind: str) -> dict[str, str]:
    """The libraries whose arithmetic the frozen reference depends on."""
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:  # pragma: no cover - the import is what is being recorded
        if kind == "xgboost":
            import xgboost

            versions["xgboost"] = xgboost.__version__
        else:
            import onnx
            import onnxruntime

            versions["onnx"] = onnx.__version__
            versions["onnxruntime"] = onnxruntime.__version__
    except ImportError as exc:  # pragma: no cover - unreachable once scoring succeeded
        raise BundleError(f"cannot record the library that scored the reference: {exc}") from exc
    return versions


def bundle_dirs(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Every parity bundle under ``root``, in name order."""
    directory = Path(root)
    if not directory.is_dir():
        return ()
    return tuple(sorted(p for p in directory.iterdir() if (p / MANIFEST_FILENAME).is_file()))


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleError",
    "Criterion",
    "DECISIONS_FILENAME",
    "Decision",
    "FEATURES_FILENAME",
    "MANIFEST_FILENAME",
    "ONNX_EPS",
    "ONNX_TIGHT_EPS",
    "PREDICTIONS_FILENAME",
    "ParityBundle",
    "SCORE_SPACE",
    "SERVABLE_KINDS",
    "bundle_dirs",
    "python_scores",
    "quantile_decision",
    "read_parity_bundle",
    "write_bundle_from_registry",
    "write_parity_bundle",
]
