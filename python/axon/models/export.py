"""Export a trained model to a verified, self-describing artifact.

The rules come from ADR-0003 and ADR-0005, and each one is here because the
obvious alternative loses money quietly:

**Trees keep their own format.** XGBoost and LightGBM are exported through the
library's own serializer, not through ONNX. Converting a tree ensemble to ONNX
re-expresses exact threshold traversal as a different graph with its own float
handling, so a "conversion" that is 1e-7 off is a conversion that moves a sample
to the other side of a split. The library reading its own file is bit-exact, and
this module *asserts* bit-exactness rather than hoping for it.

**LightGBM is the one family where that rule has no destination**, and
:func:`lightgbm_to_onnx` is the exception it earns. XGBoost's native JSON has a
Rust reader (ADR-0019 §2); LightGBM's native text does not, and
``SERVABLE_KINDS`` in :mod:`axon.parity.rust_gate` refuses the kind by name for
exactly that reason. So a LightGBM model that must cross into Rust converts to
ONNX and becomes an ``onnx`` artifact — which costs the bit-exactness above and
buys the only crossing there is. The cost is *measured*, not assumed: the
conversion goes through the same round trip as every other export, against the
native booster as the reference, at ``DEFAULT_TOLERANCE``. The native export
stays the default, because an artifact that never leaves Python has no reason to
pay for the conversion.

**Everything else goes to ONNX at FP32 with graph optimization off.** No
quantization, no FP16, no AMP (ADR-0005). FP16 is not a rounding error: it
silently downcasts and flips a prediction across a decision threshold, which
changes the trade rather than the fourth decimal place. So every export is
round-tripped — serialized, re-loaded, re-run — and audited structurally for any
reduced-precision tensor, cast or quantization operator.

**Nothing is exported without evidence.** ``sample_input`` is required: it is
both how the input schema is learned and how the round trip is measured. An
artifact whose export was never re-run is an artifact nobody has checked.

The sample is a positional ``float32`` matrix because that is exactly what the
serving path feeds — the Rust side hands over a feature vector, not a frame with
column names, and verifying at float64 would measure a model that never runs.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from axon.models.artifact import (
    ARTIFACT_FILENAMES,
    Artifact,
    ArtifactMeta,
    ModelError,
    TensorSpec,
    content_hash,
    current_git_sha,
)
from axon.models.inference import load_predictor

#: FP32 tolerance for the ONNX path. ADR-0003's starting point: NNs never
#: reproduce bit-for-bit across runtimes (ONNX does not encode op ordering and FP
#: addition is not associative), so 1e-5 is the bound, not zero.
DEFAULT_TOLERANCE = 1e-5

#: Trees are exact or the format is lossy. The tolerance argument cannot loosen
#: this: the same library reading back its own file has no excuse for a
#: difference, and if it produces one the artifact is not the model.
_EXACT_KINDS = frozenset({"xgboost", "lightgbm"})

#: What ``onnxmltools`` 1.16 will accept for a LightGBM booster
#: (``get_maximum_opset_supported()``). Asking for more is a hard error, not a
#: downgrade, so the number is pinned here rather than discovered per call: a
#: converter that silently moved with the installed package would change the
#: committed graph bytes — and therefore the frozen parity reference — on an
#: unrelated upgrade, which is the event ADR-0019 §6 says must be visible.
LIGHTGBM_TARGET_OPSET = 15


class ExportError(ModelError):
    """Base for export failures."""


class PrecisionError(ExportError):
    """The exported graph is not FP32 (ADR-0005)."""


class FidelityError(ExportError):
    """The re-loaded artifact does not reproduce the in-memory model."""


def export_artifact(
    model: Any,
    meta: ArtifactMeta,
    sample_input: Any,
    *,
    reference: Callable[[np.ndarray], np.ndarray] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    opset: int | None = None,
    narrow_score_output: bool = False,
) -> Artifact:
    """Serialize ``model``, re-load it, and return the verified :class:`Artifact`.

    ``sample_input`` is a 2-D array of representative feature rows. ``reference``
    overrides how the in-memory prediction is taken; it is *required* for a bare
    ``onnx.ModelProto``, because a graph handed to us has no in-memory counterpart
    to disagree with and recording a fidelity number from comparing a graph to
    itself would put a lie in the audit trail.

    ``narrow_score_output`` rewrites a classifier graph down to its positive-class
    column before anything is verified — see :func:`narrow_to_score_column` for
    what that costs and why it is not the default.

    **``meta.version`` is written into the payload, not merely recorded beside
    it.** Both Rust backends refuse a model that cannot name its own version
    (ADR-0019 §4) and read it out of the artifact: ONNX's first-class
    ``model_version`` field, and — because XGBoost's JSON has no equivalent slot —
    the learner attribute ``axon_model_version``. Neither is stamped by any
    library's own serializer, so before this every artifact the export path
    produced was unloadable by the core, and every bundle the cross-language gate
    ever ran on had been hand-stamped by its generator. The gate that certifies
    the boundary had never run on the path it certifies.

    This is not the exporter inventing a version. ``meta.validate()`` above has
    already required ``version >= 1``, and it is the number the registry will mint
    the artifact under, so writing it into the bytes *before* they are hashed is
    what makes drift impossible rather than what risks it: from here on the
    version in the payload, the version in the metadata and the version in the
    registry path are one fact with one source.

    Raises :class:`PrecisionError` on any reduced-precision or quantized tensor,
    and :class:`FidelityError` if the round trip moves a prediction.
    """
    meta.validate()
    x = _as_matrix(sample_input)
    kind = _detect_kind(model)
    if meta.kind and meta.kind != kind:
        # A mislabeled artifact is loaded by the wrong backend at startup, which
        # fails late and far away from here.
        raise ExportError(f"meta declares kind={meta.kind!r} but {_describe(model)} is {kind!r}")

    if kind == "xgboost":
        payload, producer = _export_xgboost(model, meta.version)
    elif kind == "lightgbm":
        # No stamp: LightGBM's text format has no version slot and no Rust reader
        # to read one, so inventing a convention here would be a convention
        # nothing consumes. A LightGBM model reaches Rust as an `onnx` artifact
        # via `lightgbm_to_onnx`, and gets ONNX's own field on the way.
        payload, producer = _export_lightgbm(model)
    else:
        payload, producer = _export_onnx(
            model, x, opset=opset, version=meta.version, narrow=narrow_score_output
        )
        # Audited on the bytes that were written rather than on the object we
        # built: that is the round trip ADR-0005 asks for.
        audit_fp32(payload)

    if reference is not None:

        def model_scores(rows: np.ndarray) -> np.ndarray:
            return np.asarray(reference(rows))

    else:
        if kind == "onnx" and not _is_estimator(model):
            raise ExportError(
                "exporting a pre-built ONNX graph needs reference=<callable> — the eager "
                "model it came from is the only thing that can prove the graph matches it"
            )

        def model_scores(rows: np.ndarray) -> np.ndarray:
            # A narrowed graph emits the positive-class column only, so the
            # reference has to be taken in that shape too. Comparing a 2-column
            # `predict_proba` against a 1-column graph would fail as a shape
            # change — the export reporting its own narrowing as a model defect.
            return _estimator_scores(
                model,
                rows,
                collapse_binary=narrow_score_output or kind in _EXACT_KINDS,
            )

    provisional = Artifact(
        meta=replace(
            meta,
            kind=kind,
            git_sha=meta.git_sha or current_git_sha(),
            artifact_filename=ARTIFACT_FILENAMES[kind],
            content_sha256=content_hash(payload),
            content_bytes=len(payload),
            producer=producer,
            created_ns=time.time_ns(),
        ),
        payload=payload,
    )

    # The round trip: from here on we are talking to the artifact, not the model.
    predictor = load_predictor(provisional)
    actual = np.asarray(predictor.predict(x))
    expected = np.asarray(model_scores(x))
    tol = 0.0 if kind in _EXACT_KINDS else float(tolerance)
    max_abs_diff = _compare(expected, actual, tolerance=tol, ref=provisional.ref)

    declared = predictor.declared_schema()
    if declared is None:
        # A tree booster's file carries no signature of its own, so the schema is
        # what serving actually passed and what it actually got back.
        inputs = (TensorSpec("input", "float32", (None, int(x.shape[1]))),)
        outputs = (TensorSpec("score", str(actual.dtype), (None, *actual.shape[1:])),)
        score_output = "score"
    else:
        inputs, outputs, score_output = declared

    final = replace(
        provisional.meta,
        inputs=inputs,
        outputs=outputs,
        score_output=score_output,
        roundtrip_max_abs_diff=max_abs_diff,
        roundtrip_rows=int(x.shape[0]),
    )
    final.validate_complete()
    return Artifact(meta=final, payload=payload)


def audit_fp32(model: Any) -> None:
    """Refuse a graph that is not FP32 end to end (ADR-0005).

    Walks initializers, value types, node attributes *and subgraphs* for any
    reduced-precision tensor, any cast into one, and any quantization operator.
    Subgraphs matter: an ``If`` branch is where a converter tool would put the
    half-precision path, and a walk that stops at the top level would call it
    clean.

    Accepts serialized bytes or an ``onnx.ModelProto``. Bytes are the interesting
    case — that is the round trip ADR-0005 asks for, auditing what was written
    rather than what we think we built.
    """
    import onnx

    proto = onnx.load_from_string(model) if isinstance(model, (bytes, bytearray)) else model
    problems = _scan_graph(proto.graph, _reduced_precision_types(), path="graph")
    if problems:
        # Every finding, not just the first: a half-precision conversion produces
        # dozens, and discovering them one export at a time is how people give up
        # and reach for a tolerance instead.
        shown = "; ".join(problems[:5])
        more = f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        raise PrecisionError(f"graph is not FP32 (ADR-0005): {shown}{more}")


# ── the LightGBM crossing ────────────────────────────────────────────────────


def lightgbm_to_onnx(
    model: Any,
    *,
    target_opset: int = LIGHTGBM_TARGET_OPSET,
) -> Any:
    """A LightGBM booster as a **single-output, single-column FP32** ONNX graph.

    This is the whole of the LightGBM → Rust route, and it needs no Rust code:
    the result is an ordinary ``onnx`` artifact, which ``SERVABLE_KINDS`` already
    admits and `tract` already serves. Hand the returned ``ModelProto`` to
    :func:`export_artifact` with ``reference=booster.predict`` — a converted graph
    has no in-memory counterpart of its own, and verifying it against itself
    would put a fidelity number in the audit trail that measured nothing.

    Three shape decisions, each preventing a failure that would otherwise surface
    a long way from here:

    * **The ``label`` output is dropped.** ``onnxmltools`` emits a classifier as
      ``(label, probabilities)``. `tract` requires exactly one output (ADR-0019
      §1), so a two-output graph passes every Python gate and then fails at Rust
      *load* — after the artifact is registered, versioned and referenced by a
      signal. ADR-0019 already names re-exporting with one output as the
      exporter's job; this is that job, done here instead of by hand.
    * **A two-column probability tensor is narrowed to its last column**, which
      is ``P(positive class)`` — the same number ``Booster.predict`` returns.
      Leaving both columns would make :func:`axon.parity.rust_gate.python_scores`
      refuse the artifact for having no single score, and a consumer that picked
      a column itself would be guessing at the thing ``score_output`` exists to
      state.
    * **Multi-class is refused here rather than narrowed.** There is no
      "positive" column among three, so any choice would be an invented trading
      rule. See ADR-0033.

    **The score space is a probability on both sides, and that is not luck.**
    ``LightgbmPredictor`` calls ``Booster.predict`` with the default
    ``raw_score=False``, and the converted graph carries LightGBM's own
    ``post_transform=LOGISTIC``. The margin-vs-probability mismatch that
    ``SCORE_SPACE`` exists for (``TreeModel`` never applies XGBoost's link) has no
    analogue here — but only because neither side skips the link. Comparing this
    graph against ``predict(raw_score=True)`` reports the sigmoid as a parity
    failure, ~1.4 absolute on a 256-row holdout, and there is a test named after
    it.
    """
    import lightgbm
    import onnx

    booster = model.booster_ if hasattr(model, "booster_") else model
    if not isinstance(booster, lightgbm.Booster):
        raise ExportError(f"{_describe(model)} is not a LightGBM model")

    dumped = booster.dump_model()
    n_class = int(dumped.get("num_class", 1))
    if n_class > 1:
        raise ExportError(
            f"a {n_class}-class LightGBM booster has no single score column, and both the "
            "parity bundle and the Rust ONNX backend serve exactly one score per row "
            "(ADR-0019 §1); fit a binary or regression objective, or keep it in Python"
        )

    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    width = int(booster.num_feature())
    # zipmap=False because ZipMap turns the probability tensor into a list of
    # dicts — not a tensor, not comparable numerically, and not loadable by any
    # Rust backend. Same reason `_export_onnx` passes it for sklearn classifiers.
    proto = onnxmltools.convert_lightgbm(
        booster,
        # A stable graph name, because `convert_lightgbm` defaults it to
        # `uuid4().hex`. That lands in `graph.name`, so two conversions of the
        # same booster on the same stack produce artifacts with different content
        # hashes — and a frozen reference that churns on every regeneration is a
        # reference whose diff nobody can review, which is the one property
        # ADR-0019 §6 asks of these fixtures. Measured: two consecutive runs
        # differed in exactly this field and nothing else.
        #
        # `skl2onnx`'s `convert_sklearn` has the identical default and needs the
        # identical pin — see `_export_onnx`. Two converters, one decision.
        name="axon_lightgbm",
        initial_types=[("input", FloatTensorType([None, width]))],
        target_opset=target_opset,
        zipmap=False,
    )
    proto = narrow_to_score_column(proto)
    # Stamped into the graph rather than into ArtifactMeta.producer, because the
    # doc string is part of the bytes the content hash covers: which converter
    # built this graph is then a property of the artifact itself and cannot be
    # separated from it by a re-registration.
    proto.doc_string = (
        f"axon lightgbm_to_onnx: lightgbm {lightgbm.__version__}, "
        f"onnxmltools {onnxmltools.__version__}, target_opset ceiling {target_opset}"
    )
    onnx.checker.check_model(proto)
    return proto


def narrow_to_score_column(proto: Any, column: int | None = None) -> Any:
    """Rewrite a graph to declare exactly one FP32 output of exactly one column.

    Every classifier converter this project uses — ``skl2onnx`` and
    ``onnxmltools`` alike, ZipMap off — emits ``(label, probabilities[n, k])``.
    `tract` requires exactly one output and one FP32 score column (ADR-0019 §1),
    and :func:`axon.parity.rust_gate.python_scores` records one score per row, so
    such a graph is refused by both halves of the boundary. This is the one
    rewrite that fixes both, and it is deliberately explicit rather than
    automatic: choosing a column is a statement about which number a strategy
    trades on, which is exactly what ADR-0015's ``score_output`` exists to stop
    anything from guessing.

    ``column`` is which column of a multi-column score tensor is the score.
    ``None`` means the **last**, which for a two-column classifier is
    ``P(positive class)``. More than two columns is refused outright: there is no
    positive column among three, so any choice would be an invented trading rule
    wearing a conversion's clothes.

    **A graph that already declares one float output of one column is returned
    untouched.** Rebuilding it would change its bytes — and therefore its content
    hash, and therefore every frozen reference taken over it — for no change in
    what it computes.

    Two things this cannot rescue, both measured on `tract` 0.23.4 and both
    surviving the narrowing untouched:

    * A ``StandardScaler`` anywhere in the pipeline compiles to an ``ai.onnx.ml``
      ``Scaler`` node. `tract` implements five operators from that domain —
      ``CategoryMapper``, ``LinearClassifier``, ``LinearRegressor``,
      ``Normalizer``, ``TreeEnsembleClassifier`` — and ``Scaler`` is not one, so
      a scaled model is refused at *parse* time however its outputs are arranged.
      Fit on raw features if the model has to cross.
    * ``skl2onnx``'s ``GradientBoostingClassifier`` emits ``base_values`` of
      length 2 where `tract` expects one per class, and fails with
      ``attribute 'base_values': expected length 1 (or undefined), got 2``.
      Deleting the attribute silently drops the intercept and padding it makes
      the two runtimes disagree by 0.18, so neither is a repair — the boosted
      tree stays uncrossable through this route.

    Prunes to the nodes the kept output actually depends on, so the ``Cast`` that
    only ever fed the discarded ``label`` does not survive as a dead node the FP32
    audit and `tract`'s translator both still have to make sense of.
    """
    from onnx import TensorProto, helper

    graph = proto.graph
    floats = [o for o in graph.output if o.type.tensor_type.elem_type == TensorProto.FLOAT]
    if len(floats) != 1:
        raise ExportError(
            f"the graph declares {len(floats)} float outputs "
            f"({[o.name for o in graph.output]}); serving reads exactly one score tensor"
        )
    source = floats[0]
    dims = source.type.tensor_type.shape.dim
    columns = int(dims[1].dim_value) if len(dims) == 2 and dims[1].HasField("dim_value") else 0

    if columns == 1 and len(graph.output) == 1:
        return proto
    if columns > 2:
        raise ExportError(
            f"the score output has {columns} columns; a parity bundle records one score per "
            "row and the Rust backend serves one, and there is no positive column among "
            f"{columns} (ADR-0033)"
        )
    if column is None:
        column = columns - 1
    if not 0 <= column < max(columns, 1):
        raise ExportError(f"score column {column} is outside a {columns}-column score tensor")

    nodes = list(graph.node)
    initializers = list(graph.initializer)
    if columns == 1:
        # One column already, but the graph declares a second output (the label).
        # Dropping that output is the whole of the narrowing here.
        keep = source.name
    else:
        # Gather with a length-1 index vector keeps the result 2-D ([batch, 1]);
        # a scalar index would collapse it to [batch], and the serving contract
        # is a column.
        index = helper.make_tensor("axon_score_column", TensorProto.INT64, [1], [column])
        initializers.append(index)
        keep = "score"
        nodes.append(
            helper.make_node(
                "Gather", [source.name, index.name], [keep], name="axon_score_column_gather", axis=1
            )
        )

    kept_nodes = _reachable(nodes, keep)
    live = {name for node in kept_nodes for name in node.input}
    output = helper.make_tensor_value_info(keep, TensorProto.FLOAT, [None, 1])
    pruned = helper.make_graph(
        kept_nodes,
        graph.name,
        list(graph.input),
        [output],
        initializer=[i for i in initializers if i.name in live],
        value_info=[v for v in graph.value_info if v.name in live],
    )
    out = helper.make_model(pruned, opset_imports=list(proto.opset_import))
    out.ir_version = proto.ir_version
    out.producer_name = proto.producer_name
    out.producer_version = proto.producer_version
    # Carried across explicitly, because `make_model` starts from an empty
    # `ModelProto`: dropping `model_version` here would silently undo the stamp
    # `export_artifact` applies before narrowing, and the artifact would be
    # refused at Rust load for having no version — a failure introduced by the
    # rewrite that exists to make it loadable.
    out.model_version = proto.model_version
    out.doc_string = proto.doc_string
    return out


def _reachable(nodes: list[Any], output: str) -> list[Any]:
    """The nodes ``output`` depends on, in their original topological order."""
    producer = {name: node for node in nodes for name in node.output}
    wanted: set[int] = set()
    frontier = [output]
    while frontier:
        name = frontier.pop()
        node = producer.get(name)
        if node is None or id(node) in wanted:
            continue
        wanted.add(id(node))
        frontier.extend(node.input)
    return [node for node in nodes if id(node) in wanted]


# ── per-family serialization ─────────────────────────────────────────────────


def _export_xgboost(model: Any, version: int) -> tuple[bytes, dict[str, str]]:
    import xgboost

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    if not isinstance(booster, xgboost.Booster):
        raise ExportError(f"{_describe(model)} is not an XGBoost model")
    # Stamped on a *copy*. XGBoost's JSON has no version field, so the version
    # rides in the learner attribute map (ADR-0019 §4) — but `set_attr` mutates
    # the booster in place, and an exporter that silently rewrites the caller's
    # live model is one that changes what the next `predict` in a notebook is
    # running against. `Booster.copy()` round-trips through the same serializer
    # this function is about to call, so the copy is exact by construction.
    stamped = booster.copy()
    stamped.set_attr(axon_model_version=str(version))
    # save_raw over save_model: the JSON never touches the filesystem, so a
    # half-written temp file can never become an artifact.
    return bytes(stamped.save_raw(raw_format="json")), {"xgboost": xgboost.__version__}


def _export_lightgbm(model: Any) -> tuple[bytes, dict[str, str]]:
    import lightgbm

    booster = model.booster_ if hasattr(model, "booster_") else model
    if not isinstance(booster, lightgbm.Booster):
        raise ExportError(f"{_describe(model)} is not a LightGBM model")
    # LightGBM's native format is text, not JSON, and this is deliberate: its
    # `dump_model()` JSON is one-way — the library provides no loader for it — so
    # a JSON artifact would be an artifact that cannot be served. ADR-0003 asks
    # for the library's *own exact* format; for LightGBM that is this string.
    return booster.model_to_string().encode("utf-8"), {"lightgbm": lightgbm.__version__}


def _export_onnx(
    model: Any, x: np.ndarray, *, opset: int | None, version: int, narrow: bool
) -> tuple[bytes, dict[str, str]]:
    import onnx

    if isinstance(model, onnx.ModelProto):
        # Copied before stamping: a `ModelProto` handed in by a caller is theirs,
        # and `model_version` is a field they may be relying on. protobuf's own
        # `CopyFrom` is exact, so the only difference between the copy and the
        # original is the field this function is here to set.
        proto = onnx.ModelProto()
        proto.CopyFrom(model)
        proto.model_version = version
        if narrow:
            proto = narrow_to_score_column(proto)
        producer = {"onnx": onnx.__version__, "opset": str(_default_opset(proto))}
        return proto.SerializeToString(), producer

    from skl2onnx import __version__ as skl2onnx_version
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    import sklearn

    initial_types = [("input", FloatTensorType([None, int(x.shape[1])]))]
    options = None
    if sklearn.base.is_classifier(model):
        # ZipMap turns the probability tensor into a list of dicts. That is not a
        # tensor, cannot be compared numerically here, and cannot be consumed by
        # `ort` on the Rust side at all.
        #
        # ZipMap off still leaves *two* graph outputs, `label` and
        # `probabilities`, which is one refusal short of servable — see
        # `narrow_score_output` on `export_artifact`.
        options = {id(model): {"zipmap": False}}
    proto = convert_sklearn(
        model,
        # The same pin, for the same reason, as `lightgbm_to_onnx` puts on
        # `convert_lightgbm`: `convert_sklearn` defaults `name` to `uuid4().hex`
        # and it lands in `graph.name`, so two exports of one fitted estimator
        # produce different bytes and therefore different `content_sha256`. That
        # makes ADR-0015's content hash identify a *conversion event* rather than
        # a model, which is not what the registry's immutability claim means —
        # and it makes any frozen reference over the artifact churn on every
        # regeneration. Measured: two consecutive conversions of one fitted
        # `LogisticRegression`, 415 bytes each, differing in exactly this field
        # and nothing else.
        name="axon_sklearn",
        initial_types=initial_types,
        target_opset=opset,
        options=options,
    )
    # Ours, freshly built, so it is stamped in place. skl2onnx leaves
    # `model_version` at 0, which reads as unset and is refused by the Rust
    # loader — the one field that turns a correct graph into an unloadable one.
    proto.model_version = version
    if narrow:
        proto = narrow_to_score_column(proto)
    producer = {
        "sklearn": sklearn.__version__,
        "skl2onnx": skl2onnx_version,
        "onnx": onnx.__version__,
        # Recorded rather than pinned: the opset determines op semantics, and
        # "whatever was installed" is not an answer an audit can act on.
        "opset": str(_default_opset(proto)),
    }
    return proto.SerializeToString(), producer


def _default_opset(proto: Any) -> int:
    versions = [op.version for op in proto.opset_import if op.domain in ("", "ai.onnx")]
    return max(versions, default=0)


# ── reference predictions ────────────────────────────────────────────────────


def _estimator_scores(model: Any, x: np.ndarray, *, collapse_binary: bool) -> np.ndarray:
    """The in-memory model's score for ``x`` — probability for a classifier.

    Taken from the *wrapper* (``XGBRegressor``, ``LGBMClassifier``, an sklearn
    estimator), never from the booster underneath it, because an early-stopped
    wrapper predicts with ``best_iteration`` trees while a bare booster uses all
    of them. Comparing booster to booster would call that artifact identical while
    it silently serves a model nobody validated.

    ``collapse_binary`` picks the convention of the artifact being verified: a
    tree booster emits ``P(class 1)`` as a flat vector while an ONNX classifier
    graph emits both columns. Taking the reference in the artifact's own shape
    keeps a convention difference from being reported as a numerical one.
    """
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(x))
        if collapse_binary and p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        return p
    return np.asarray(model.predict(x))


def _compare(expected: np.ndarray, actual: np.ndarray, *, tolerance: float, ref: str) -> float:
    if actual.dtype == np.float16:
        # The structural audit cannot see this one: the graph can be FP32 and the
        # execution provider still hand back half precision (ADR-0005's CoreML
        # footgun). The only way to catch it is to look at what came out.
        raise PrecisionError(f"{ref}: the runtime returned float16 outputs (ADR-0005)")
    if not np.isfinite(actual).all():
        raise FidelityError(f"{ref}: the re-loaded artifact produced non-finite scores")
    a, b = _align(expected, actual)
    if a.shape != b.shape:
        raise FidelityError(
            f"{ref}: the artifact returns {actual.shape} where the model returns "
            f"{expected.shape} — the export changed the model's output, not just its bits"
        )
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    worst = float(diff.max()) if diff.size else 0.0
    if worst > tolerance:
        row = int(np.unravel_index(int(diff.argmax()), diff.shape)[0])
        raise FidelityError(
            f"{ref}: max_abs_diff {worst:.3e} exceeds {tolerance:.3e} (worst at row {row}: "
            f"model {a.reshape(len(a), -1)[row]} vs artifact {b.reshape(len(b), -1)[row]})"
        )
    if a.ndim == 2 and a.shape[1] > 1:
        # Decision invariance, in the one form that is unambiguous at export time:
        # the predicted class must not move. A full decision-invariance gate needs
        # the strategy's own thresholds and belongs to the parity harness
        # (ADR-0003 §3) — this proves the artifact matches the model, not that the
        # strategy is unchanged.
        flips = int(np.count_nonzero(a.argmax(axis=1) != b.argmax(axis=1)))
        if flips:
            raise FidelityError(
                f"{ref}: {flips}/{len(a)} rows change predicted class after export — "
                "a decision flip, not a rounding difference"
            )
    return worst


def _align(expected: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconcile ``(n,)`` against ``(n, 1)``.

    skl2onnx emits a column vector where a regressor returns a flat one. That is a
    shape convention, not a numerical difference, and refusing it would fail every
    regressor export for a disagreement about brackets.
    """
    if expected.shape != actual.shape and expected.size == actual.size:
        return expected.reshape(-1), actual.reshape(-1)
    return expected, actual


# ── the FP32 audit ───────────────────────────────────────────────────────────

#: Operators that only exist to run a model at reduced precision. ADR-0005 forbids
#: quantization outright, so their presence is a rejection regardless of dtype.
_QUANTIZED_OPS = frozenset(
    {
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "DynamicQuantizeMatMul",
        "MatMulInteger",
        "MatMulInteger16",
        "ConvInteger",
        "QGemm",
        "QAttention",
    }
)


def _reduced_precision_types() -> frozenset[int]:
    """Every ONNX dtype narrower than FP32, derived from the installed onnx.

    Derived rather than hardcoded so that the next low-precision type the standard
    adds (the FP8 and FP4 families arrived this way) is refused on the day it
    appears, instead of on the day someone remembers to update a list. INT8 and
    UINT8 are in the set for the same reason the quantization operators are: an
    FP32 graph has no legitimate use for them.
    """
    from onnx import TensorProto

    narrow = ("FLOAT16", "BFLOAT16", "FLOAT8", "INT4", "FLOAT4", "INT8")
    return frozenset(v for k, v in TensorProto.DataType.items() if any(n in k for n in narrow))


def _scan_graph(graph: Any, reduced: frozenset[int], *, path: str) -> list[str]:
    from onnx import AttributeProto, TensorProto

    problems: list[str] = []
    names = TensorProto.DataType.Name

    for init in graph.initializer:
        if init.data_type in reduced:
            problems.append(f"{path}: initializer {init.name!r} is {names(init.data_type)}")
    for sparse in graph.sparse_initializer:
        if sparse.values.data_type in reduced:
            problems.append(f"{path}: sparse initializer is {names(sparse.values.data_type)}")
    groups = (("input", graph.input), ("output", graph.output), ("value", graph.value_info))
    for group, values in groups:
        for value in values:
            for dtype in _value_types(value.type):
                if dtype in reduced:
                    problems.append(f"{path}: {group} {value.name!r} is {names(dtype)}")

    for node in graph.node:
        label = node.name or node.op_type
        if node.op_type in _QUANTIZED_OPS or node.op_type.startswith("QLinear"):
            problems.append(f"{path}: node {label!r} is quantized ({node.op_type})")
        for attr in node.attribute:
            if node.op_type in ("Cast", "CastLike") and attr.name == "to" and attr.i in reduced:
                problems.append(f"{path}: node {label!r} casts to {names(attr.i)}")
            if attr.type == AttributeProto.TENSOR and attr.t.data_type in reduced:
                problems.append(f"{path}: node {label!r} holds a {names(attr.t.data_type)} tensor")
            if attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    if tensor.data_type in reduced:
                        problems.append(
                            f"{path}: node {label!r} holds a {names(tensor.data_type)} tensor"
                        )
            if attr.type == AttributeProto.GRAPH:
                problems += _scan_graph(attr.g, reduced, path=f"{path}/{label}:{attr.name}")
            if attr.type == AttributeProto.GRAPHS:
                for i, sub in enumerate(attr.graphs):
                    problems += _scan_graph(sub, reduced, path=f"{path}/{label}:{attr.name}[{i}]")
    return problems


def _value_types(type_proto: Any) -> list[int]:
    """Element types reachable from a value's type, through sequences and maps."""
    which = type_proto.WhichOneof("value")
    if which == "tensor_type":
        return [type_proto.tensor_type.elem_type]
    if which == "sequence_type":
        return _value_types(type_proto.sequence_type.elem_type)
    if which == "optional_type":
        return _value_types(type_proto.optional_type.elem_type)
    if which == "map_type":
        return _value_types(type_proto.map_type.value_type)
    return []


# ── input handling ───────────────────────────────────────────────────────────


def _as_matrix(sample_input: Any) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(sample_input, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ExportError(f"sample_input must be a non-empty 2-D matrix, got shape {x.shape}")
    if not np.isfinite(x).all():
        # A NaN row makes every comparison below vacuous, so the export would
        # "verify" without having compared anything.
        raise ExportError("sample_input contains non-finite values; verification would be vacuous")
    return x


def _detect_kind(model: Any) -> str:
    """Which artifact format this model serializes to, from its own module."""
    root = type(model).__module__.split(".")[0]
    if root in ("xgboost", "lightgbm"):
        return root
    if root == "onnx" or root.startswith("sklearn") or _is_estimator(model):
        return "onnx"
    raise ExportError(
        f"no export path for {_describe(model)}; supported: xgboost, lightgbm, sklearn, "
        "and a pre-built onnx.ModelProto (export a Torch model with torch.onnx.export first)"
    )


def _is_estimator(model: Any) -> bool:
    return hasattr(model, "fit") and (hasattr(model, "predict") or hasattr(model, "predict_proba"))


def _describe(model: Any) -> str:
    return f"{type(model).__module__}.{type(model).__name__}"


__all__ = [
    "DEFAULT_TOLERANCE",
    "LIGHTGBM_TARGET_OPSET",
    "ExportError",
    "FidelityError",
    "PrecisionError",
    "audit_fp32",
    "export_artifact",
    "lightgbm_to_onnx",
    "narrow_to_score_column",
]
