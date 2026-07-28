"""LightGBM's only crossing into Rust, and the two things it is refused for.

ADR-0019 left the native LightGBM backend unbuilt, so
``axon.parity.rust_gate.SERVABLE_KINDS`` refuses the ``lightgbm`` kind by name and
the family has no Rust side at all. ADR-0033 answers that with a **route rather
than a backend**: ``onnxmltools`` converts the booster, the result is an ordinary
``onnx`` artifact, and `tract` already serves those. This file is the evidence for
that claim and for its two boundaries.

Each test is named after the failure mode it prevents, matching
``crates/axon-execution/src/tracker.rs``. Everything is offline and deterministic:
fixed seeds, no clock, no network. The heavy ML stack is optional (``pyproject``
puts it in the ``ml`` extra), so every test that needs lightgbm, onnxmltools or
onnxruntime ``importorskip``s it *inside* the test — a bare numpy + pytest
environment still runs the whole bundle-format half, which is where the refusals
live.

The committed fixtures under ``data/lightgbm-onnx`` are the Rust side's inputs;
``crates/axon-model`` points `tract` at them. Regenerate them deliberately with

    PYTHONPATH=python .venv/bin/python python/tests/test_lightgbm_onnx.py

and review the diff, for the same reason
``crates/axon-model/tests/bundles/generate.py`` is not run by CI: a reference
regenerated in the same breath as the assertion agrees with whatever the code has
just started doing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile

import numpy as np
import pytest

from axon.models import Artifact, ArtifactMeta, content_hash
from axon.parity.rust_gate import (
    ONNX_EPS,
    BundleError,
    bundle_dirs,
    python_scores,
    quantile_decision,
    read_parity_bundle,
    write_parity_bundle,
)

#: Where the committed LightGBM→ONNX fixtures live, relative to the repo root.
#: Deliberately **not** under ``crates/axon-model/tests/bundles``: everything in
#: that directory is asserted by the Rust gate on every ``cargo test``, and the
#: regressor bundle here is one `tract` is expected to refuse.
#:
#: Not under ``data/`` either, which ``.gitignore`` excludes wholesale — a fixture
#: the Rust side is told to point at, that a fresh clone does not have, is worse
#: than no fixture. ``*.onnx`` is ignored too, so ``model.onnx`` needs ``git add
#: -f``; the committed bundles under ``crates/axon-model/tests`` were added the
#: same way.
FIXTURES = "python/tests/fixtures/lightgbm-onnx"

#: The bundle the whole route rests on: a binary booster whose graph `tract`
#: loads and runs. Named here so a rename shows up as a test failure rather than
#: as a silently empty fixture sweep.
BINARY_BUNDLE = "lgbm_binary"

#: Written on purpose and expected to fail on the Rust side: a regression
#: objective converts to ``TreeEnsembleRegressor``, which `tract` 0.23.4 has no
#: reader for. Keeping it as a fixture is what makes that a measurement someone
#: can re-run rather than a sentence in an ADR.
REGRESSOR_BUNDLE = "lgbm_regressor"

DIM = 5
COEF = np.array([1.5, -2.0, 0.3, 0.0, 0.7], dtype=np.float32)


def fixtures(repo_root) -> tuple:
    dirs = bundle_dirs(repo_root / FIXTURES)
    assert dirs, f"no LightGBM parity bundles under {FIXTURES}; the route has no evidence"
    return dirs


def probe_ref(recipe: dict) -> str:
    """A ``FeatureSpec.ref``-shaped id for a synthetic probe, as ``generate.py`` does."""
    body = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return f"lgbm_probe/v1#{hashlib.sha256(body).hexdigest()[:16]}"


def gaussian(seed: int, rows: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, DIM)).astype(np.float32)


def booster(kind: str, seed: int = 4201):
    """A tiny seeded LightGBM booster: ``binary``, ``regression`` or ``multiclass``.

    ``n_jobs=1`` and ``deterministic=True`` because these bytes are committed —
    a booster whose split order depends on how many cores the box has would make
    every regeneration a diff.
    """
    lightgbm = pytest.importorskip("lightgbm")
    x = gaussian(seed, 600)
    linear = x @ COEF
    common = dict(
        n_estimators=8,
        num_leaves=7,
        random_state=0,
        verbose=-1,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
    )
    if kind == "regression":
        est = lightgbm.LGBMRegressor(**common).fit(x, linear)
    elif kind == "binary":
        est = lightgbm.LGBMClassifier(**common).fit(x, (linear > 0.0).astype(np.int32))
    elif kind == "multiclass":
        est = lightgbm.LGBMClassifier(**common).fit(x, np.digitize(linear, [-1.0, 1.0]))
    else:  # pragma: no cover - a typo in a test, not a runtime path
        raise AssertionError(f"unknown probe objective {kind!r}")
    return est.booster_, x


def unnarrowed_graph(native_booster):
    """``convert_lightgbm`` with the narrowing deliberately skipped.

    The multi-class refusals are only worth anything if they have met a real
    multi-class graph, and ``lightgbm_to_onnx`` refuses to build one — so this is
    the converter called directly. ``name`` is passed for the same reason
    ``lightgbm_to_onnx`` passes it: the default is ``uuid4().hex`` and lands in
    ``graph.name``, so a fixture built without it has a different content hash on
    every regeneration.
    """
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    from axon.models.export import LIGHTGBM_TARGET_OPSET

    return onnxmltools.convert_lightgbm(
        native_booster,
        name="axon_lightgbm_multiclass",
        initial_types=[("input", FloatTensorType([None, DIM]))],
        target_opset=LIGHTGBM_TARGET_OPSET,
        zipmap=False,
    )


def as_artifact(proto, *, registry_id: str, version: int, reference, sample) -> Artifact:
    """``proto`` through the real export path, so the round trip is the real one."""
    from axon.models import export_artifact

    proto.model_version = version
    meta = ArtifactMeta(
        registry_id=registry_id,
        version=version,
        feature_spec_ref=probe_ref({"kind": "gaussian", "seed": 4201, "dim": DIM}),
    )
    return export_artifact(proto, meta, sample, reference=reference)


# ── the route itself (needs the ML stack) ─────────────────────────────────────


def test_a_binary_booster_and_its_converted_graph_score_the_same_probability():
    # The claim ADR-0033 rests on. If this drifts past ONNX_EPS the route is not a
    # route, and the argument for a native `lightgbm3-rs` backend becomes a
    # measurement instead of an assumption.
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxruntime")
    from axon.models.export import lightgbm_to_onnx
    from axon.models.inference import LightgbmPredictor, OnnxPredictor

    native_booster, train = booster("binary")
    proto = lightgbm_to_onnx(native_booster)
    proto.model_version = 33

    holdout = gaussian(7717, 256)
    native = np.asarray(LightgbmPredictor(native_booster.model_to_string().encode())
                        .predict(holdout)).reshape(-1)
    graph = OnnxPredictor(proto.SerializeToString())
    # One row at a time, because `python_scores` scores an ONNX bundle that way:
    # the Rust plan pins its batch dimension to 1, and a graph scored 256 rows at
    # once may reassociate a reduction that a batch of one cannot.
    converted = np.array(
        [float(np.asarray(graph.predict(row.reshape(1, -1))).reshape(-1)[0]) for row in holdout]
    )
    worst = float(np.abs(native - converted).max())
    assert worst <= ONNX_EPS, f"max_abs_diff {worst:.3e} over {len(holdout)} rows"
    assert worst > 0.0, (
        "the two runtimes agreed to the last bit, which no two tree implementations do; "
        "this test is almost certainly comparing something against itself"
    )
    assert len(train) == 600


def test_comparing_the_graph_against_a_raw_score_reports_the_link_as_a_parity_failure():
    # The score-space trap, in numbers rather than in prose. `Booster.predict`
    # defaults to `raw_score=False` and the converted graph carries LightGBM's own
    # `post_transform=LOGISTIC`, so both sides are probabilities and the families
    # agree. Take the *margin* on one side and the gate reports the sigmoid as a
    # model defect — the same shape as SCORE_SPACE's margin-vs-probability rule for
    # XGBoost, and the reason ADR-0033 states the space rather than leaving it to
    # whoever writes the next comparison.
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxruntime")
    from axon.models.export import lightgbm_to_onnx
    from axon.models.inference import OnnxPredictor

    native_booster, _ = booster("binary")
    proto = lightgbm_to_onnx(native_booster)
    proto.model_version = 33
    holdout = gaussian(7717, 256)
    converted = np.asarray(
        OnnxPredictor(proto.SerializeToString()).predict(holdout)
    ).reshape(-1).astype(np.float64)

    probability = np.asarray(native_booster.predict(holdout)).reshape(-1)
    margin = np.asarray(native_booster.predict(holdout, raw_score=True)).reshape(-1)
    assert float(np.abs(probability - converted).max()) <= ONNX_EPS
    assert float(np.abs(margin - converted).max()) > 1.0


def test_the_converted_graph_declares_one_output_because_the_rust_backend_serves_one():
    # `onnxmltools` emits a classifier as (label, probabilities). `tract` requires
    # exactly one output and one FP32 score column (ADR-0019 §1), so a graph left
    # in that shape passes every Python gate and then fails at Rust *load* — after
    # the artifact has been registered, versioned and referenced by a signal.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxmltools")
    from axon.models.export import lightgbm_to_onnx

    proto = lightgbm_to_onnx(booster("binary")[0])
    assert len(proto.graph.output) == 1
    (out,) = proto.graph.output
    assert out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert [d.dim_value for d in out.type.tensor_type.shape.dim][1:] == [1]
    assert "label" not in {o.name for o in proto.graph.output}
    # The Cast that only ever fed the discarded label must be pruned, not merely
    # orphaned: a dead node is still a node the FP32 audit and `tract`'s translator
    # both have to make sense of.
    assert "Cast" not in {n.op_type for n in proto.graph.node}


def test_a_multiclass_booster_is_refused_at_conversion_rather_than_narrowed_to_one_class():
    # There is no "positive" column among three, so narrowing would be an invented
    # trading rule wearing a conversion's clothes.
    pytest.importorskip("onnxmltools")
    from axon.models.export import ExportError, lightgbm_to_onnx

    with pytest.raises(ExportError, match="3-class"):
        lightgbm_to_onnx(booster("multiclass")[0])


def test_a_multiclass_graph_is_refused_by_the_bundle_writer_rather_than_scored_on_column_zero(
    tmp_path,
):
    # The refusal in `python_scores` has to meet a real multi-class graph at least
    # once, or it is a claim rather than a check. This is that graph: converted by
    # `onnxmltools` with the narrowing deliberately bypassed, so the score output is
    # a genuine (n, 3) probability tensor.
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxruntime")
    from axon.models.inference import OnnxPredictor

    native_booster, train = booster("multiclass")
    artifact = as_artifact(
        unnarrowed_graph(native_booster),
        registry_id="lgbm_multiclass",
        version=34,
        reference=native_booster.predict,
        sample=train[:32],
    )
    # Column 0 is a perfectly plausible number — that is exactly the danger. A
    # writer that silently took it would produce a bundle nothing downstream could
    # tell apart from a real one.
    column_zero = np.asarray(
        OnnxPredictor(artifact.payload).predict(train[:48])
    )[:, 0]
    assert np.isfinite(column_zero).all()

    out = tmp_path / "bundle"
    with pytest.raises(BundleError, match="3 values per row"):
        write_parity_bundle(
            artifact, train[:48], out_dir=out, decision=quantile_decision()
        )
    # Refused, not truncated: nothing that records an answer was written. The
    # manifest is written last, so a directory with features and no manifest reads
    # as an interrupted write rather than as a bundle.
    assert not (out / "predictions.f32").exists()
    assert not (out / "manifest.json").exists()
    with pytest.raises(BundleError, match="interrupted write"):
        read_parity_bundle(out)


def test_a_native_lightgbm_artifact_is_still_refused_because_no_rust_backend_reads_the_text():
    # Widening SERVABLE_KINDS would be the wrong fix and the tempting one: the
    # `lightgbm` kind is the booster's own text format, and nothing in Rust parses
    # it. The route converts the model instead of widening the gate.
    from axon.parity.rust_gate import SERVABLE_KINDS

    assert "lightgbm" not in SERVABLE_KINDS
    payload = b"tree\nversion=v3\n"
    artifact = Artifact(
        meta=ArtifactMeta(
            registry_id="lgbm_native",
            version=1,
            kind="lightgbm",
            feature_spec_ref=probe_ref({"kind": "gaussian", "seed": 4201, "dim": DIM}),
            content_sha256=content_hash(payload),
            content_bytes=len(payload),
            artifact_filename="model.txt",
        ),
        payload=payload,
    )
    with pytest.raises(BundleError, match="lightgbm_to_onnx"):
        python_scores(artifact, np.ones((2, DIM), dtype=np.float32))


def test_converting_the_same_booster_twice_produces_the_same_bytes():
    # `convert_lightgbm` defaults its `name` argument to `uuid4().hex` and puts it
    # in `graph.name`, so two conversions of one booster on one stack differ in
    # their content hash and in nothing else. That makes a committed reference
    # churn on every regeneration, which destroys the only property ADR-0019 §6
    # asks of these fixtures: that the diff is the review. Measured before the
    # fix — two consecutive runs, identical nodes, different sha256.
    pytest.importorskip("onnxmltools")
    from axon.models.export import lightgbm_to_onnx

    native_booster, _ = booster("binary")
    first = lightgbm_to_onnx(native_booster).SerializeToString()
    second = lightgbm_to_onnx(native_booster).SerializeToString()
    assert first == second
    # The fixture generator's other converter call has to be pinned too, or the
    # multi-class artifact churns on its own.
    assert unnarrowed_graph(native_booster).SerializeToString() == (
        unnarrowed_graph(native_booster).SerializeToString()
    )


def test_the_committed_fixtures_name_the_version_the_rust_loader_demands(repo_root):
    # ADR-0019 §4: `model_version` 0 reads as unset and the graph is refused
    # before it is planned. These fixtures are the Rust side's inputs, so a zero
    # here is a fixture nothing can run — and it is now the export path that
    # writes the field, not the generator by hand.
    onnx = pytest.importorskip("onnx")
    for directory in fixtures(repo_root):
        bundle = read_parity_bundle(directory)
        stamped = onnx.load_from_string(bundle.artifact().payload).model_version
        assert stamped == bundle.model_version != 0, directory


# ── the committed fixtures ────────────────────────────────────────────────────


def test_every_committed_lightgbm_bundle_verifies_against_its_own_manifest(repo_root):
    # Bare numpy: the fixture the Rust side reads has to be checkable on a machine
    # that cannot load a model at all.
    names = set()
    for directory in fixtures(repo_root):
        bundle = read_parity_bundle(directory)
        names.add(directory.name)
        assert bundle.kind == "onnx", directory
        assert bundle.manifest["score_space"] == "score", directory
        assert bundle.features.shape[0] == bundle.predictions.shape[0] > 0
        assert np.isfinite(bundle.predictions).all()
        assert len(set(bundle.decisions.tolist())) > 1, directory
    assert {BINARY_BUNDLE, REGRESSOR_BUNDLE} <= names


def test_the_frozen_lightgbm_reference_is_still_what_this_python_stack_says(repo_root):
    # Same guarantee as `test_the_frozen_reference_is_still_what_this_python_stack_says`
    # for the committed XGBoost/MLP bundles: a library upgrade that moves these
    # answers has to fail here, naming the library, rather than surfacing months
    # later as an unexplained Rust parity failure.
    pytest.importorskip("onnxruntime")
    for directory in fixtures(repo_root):
        bundle = read_parity_bundle(directory)
        bundle.compare(python_scores(bundle.artifact(), bundle.features)).raise_for_status()


def test_a_prediction_matrix_with_more_than_one_column_is_refused_rather_than_read_as_column_zero(
    tmp_path, repo_root
):
    # The strong form of the multi-class refusal on the *reading* side. The forged
    # bundle's column 0 is byte-for-byte the reference it already carries, so every
    # other check — the hashes, the decisions, the counts — still passes. A reader
    # that quietly took the first column would therefore report this bundle as
    # healthy, and the gate would be running against one third of a model.
    source = repo_root / FIXTURES / BINARY_BUNDLE
    directory = tmp_path / BINARY_BUNDLE
    shutil.copytree(source, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = int(manifest["predictions"]["rows"])
    original = np.frombuffer((directory / "predictions.f32").read_bytes(), dtype="<f4")
    widened = np.column_stack([original, original * 0.5, original * 0.25]).astype("<f4")
    payload = np.ascontiguousarray(widened).tobytes()
    (directory / "predictions.f32").write_bytes(payload)
    manifest["predictions"].update(
        cols=3, bytes=len(payload), sha256=content_hash(payload), rows=rows
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BundleError, match="one score per row"):
        read_parity_bundle(directory)


# ── regeneration ──────────────────────────────────────────────────────────────


def _regenerate() -> None:  # pragma: no cover - run by hand, never by the gate
    """Rewrite ``data/lightgbm-onnx``. See this module's docstring."""
    import pathlib

    from axon.models import ModelRegistry
    from axon.models.export import lightgbm_to_onnx

    root = pathlib.Path(__file__).resolve().parents[2] / FIXTURES
    root.mkdir(parents=True, exist_ok=True)
    holdout = gaussian(7717, 256)

    for name, objective, version in (
        (BINARY_BUNDLE, "binary", 33),
        (REGRESSOR_BUNDLE, "regression", 34),
    ):
        native_booster, train = booster(objective)
        artifact = as_artifact(
            lightgbm_to_onnx(native_booster),
            registry_id=f"parity_{name}",
            version=version,
            reference=native_booster.predict,
            sample=train[:32],
        )
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(tmp)
            registry.save(artifact)
            write_parity_bundle(
                registry.load(artifact.meta.registry_id, version),
                holdout,
                out_dir=root / name,
                decision=quantile_decision(lower=0.4, upper=0.6),
                overwrite=True,
            )
        print(f"wrote {root / name}")

    # No bundle for the multi-class case — `python_scores` refuses to record one,
    # which is the point. The bare graph is kept so the Rust side can measure its
    # own refusal of a two-output, three-column artifact against a real one.
    multi, _ = booster("multiclass")
    proto = unnarrowed_graph(multi)
    proto.model_version = 35
    unservable = root / "lgbm_multiclass_unservable"
    unservable.mkdir(parents=True, exist_ok=True)
    (unservable / "model.onnx").write_bytes(proto.SerializeToString())
    print(f"wrote {unservable / 'model.onnx'}")


if __name__ == "__main__":  # pragma: no cover
    _regenerate()
