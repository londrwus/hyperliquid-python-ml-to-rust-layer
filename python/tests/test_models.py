"""Model artifacts and the registry: what must never be possible.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic — no network, no clock-dependent assertions.

The heavy ML stack is optional (``pyproject`` puts it in the ``ml`` extra), so
every test that needs one of those libraries ``importorskip``s it *inside* the
test. A bare numpy + pytest environment still runs the whole artifact/registry
half of this file, which is where the correctness properties live.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from axon.models import (
    Artifact,
    ArtifactExistsError,
    ArtifactMeta,
    ArtifactNotFoundError,
    ExportError,
    FidelityError,
    IntegrityError,
    ModelRegistry,
    PrecisionError,
    TensorSpec,
    audit_fp32,
    content_hash,
    current_git_sha,
    export,
    export_artifact,
    load_predictor,
    session_options,
)

# The shape of an `axon.features.FeatureSpec.ref`: name, version, and a
# fingerprint of the transforms themselves.
FEATURE_SPEC = "btc_micro/v3#0f1e2d3c4b5a6978"


def meta(registry_id: str = "btc_1m", version: int = 1, **overrides) -> ArtifactMeta:
    """A declaration: what a caller fills in before an export runs."""
    fields = {
        "registry_id": registry_id,
        "version": version,
        "feature_spec_ref": FEATURE_SPEC,
        "git_sha": "0" * 40,
    }
    fields.update(overrides)
    return ArtifactMeta(**fields)


def fake_artifact(registry_id: str = "btc_1m", version: int = 1, payload: bytes = b'{"a":1}\n'):
    """A complete artifact record with a payload no ML library needs to parse.

    Lets the registry's invariants — immutability, integrity, ordering — be tested
    in a bare numpy environment, which is where they matter most: they are the
    part that has to hold on a machine that cannot even load the model.
    """
    return Artifact(
        meta=meta(
            registry_id,
            version,
            kind="xgboost",
            inputs=(TensorSpec("input", "float32", (None, 3)),),
            outputs=(TensorSpec("score", "float32", (None,)),),
            score_output="score",
            producer={"xgboost": "3.3.0"},
            content_sha256=content_hash(payload),
            content_bytes=len(payload),
            artifact_filename="model.json",
            roundtrip_max_abs_diff=0.0,
            roundtrip_rows=32,
            created_ns=1_700_000_000_000_000_000,
        ),
        payload=payload,
    )


def _xy(rows: int = 160, cols: int = 4, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    y = 2.0 * x[:, 0] - x[:, 1] + 0.25 * x[:, 2] * x[:, 3]
    return x, y.astype(np.float64)


# ── immutability ──────────────────────────────────────────────────────────────


def test_writing_an_existing_version_is_refused_instead_of_overwriting(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(fake_artifact(payload=b"first"))
    with pytest.raises(ArtifactExistsError, match="immutable"):
        registry.save(fake_artifact(payload=b"second"))
    # The point of the refusal: the bytes version 1 named are still the bytes
    # version 1 names, so a signal stamped model_version=1 still resolves.
    assert registry.load("btc_1m", 1).payload == b"first"


def test_a_refused_write_leaves_no_staging_directory_behind(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(fake_artifact())
    with pytest.raises(ArtifactExistsError):
        registry.save(fake_artifact(payload=b"other"))
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".staging")] == []


def test_next_version_never_hands_out_a_number_already_taken(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.next_version("btc_1m") == 1
    registry.save(fake_artifact(version=1))
    registry.save(fake_artifact(version=2))
    assert registry.next_version("btc_1m") == 3
    registry.save(fake_artifact(version=registry.next_version("btc_1m")))
    assert registry.list_versions("btc_1m") == (1, 2, 3)


# ── integrity ─────────────────────────────────────────────────────────────────


def test_an_artifact_whose_bytes_were_edited_is_refused_by_the_loader(tmp_path):
    registry = ModelRegistry(tmp_path)
    directory = registry.save(fake_artifact(payload=b'{"threshold": 0.5}'))
    (directory / "model.json").write_bytes(b'{"threshold": 0.6}')
    with pytest.raises(IntegrityError, match="content hash"):
        registry.load("btc_1m", 1)


def test_a_truncated_artifact_is_refused_by_the_loader(tmp_path):
    registry = ModelRegistry(tmp_path)
    directory = registry.save(fake_artifact(payload=b"0123456789"))
    (directory / "model.json").write_bytes(b"01234")
    with pytest.raises(IntegrityError, match="truncated"):
        registry.load("btc_1m", 1)


def test_metadata_that_disagrees_with_its_directory_is_refused(tmp_path):
    # `cp -r v1 v2` is how one model comes to answer to two version numbers, and
    # after that a replay of version 2 silently runs version 1.
    registry = ModelRegistry(tmp_path)
    directory = registry.save(fake_artifact(version=1))
    shutil.copytree(directory, directory.parent / "v0000000002")
    with pytest.raises(IntegrityError, match="edited by hand"):
        registry.load("btc_1m", 2)


def test_a_missing_model_file_is_reported_against_the_name_the_metadata_gave(tmp_path):
    registry = ModelRegistry(tmp_path)
    directory = registry.save(fake_artifact())
    (directory / "model.json").unlink()
    with pytest.raises(ArtifactNotFoundError, match="model.json"):
        registry.load("btc_1m", 1)


def test_a_version_directory_without_metadata_is_named_rather_than_silently_skipped(tmp_path):
    registry = ModelRegistry(tmp_path)
    (tmp_path / "btc_1m" / "v0000000004").mkdir(parents=True)
    assert registry.list_versions("btc_1m") == ()
    with pytest.raises(ArtifactNotFoundError, match="interrupted write"):
        registry.load("btc_1m", 4)


# ── resolution and listing ────────────────────────────────────────────────────


def test_latest_is_the_highest_version_not_the_most_recently_written(tmp_path):
    registry = ModelRegistry(tmp_path)
    for version in (10, 2, 3):
        registry.save(fake_artifact(version=version))
    # Written last, and still not the latest: a restored backup has fresh mtimes
    # and unchanged versions, and lexical order would put v10 before v2.
    assert registry.resolve_latest("btc_1m") == 10
    assert registry.list_versions("btc_1m") == (2, 3, 10)
    assert registry.load("btc_1m").meta.version == 10


def test_listing_an_empty_registry_is_empty_rather_than_an_error(tmp_path):
    registry = ModelRegistry(tmp_path / "does-not-exist")
    assert registry.list_ids() == ()
    assert registry.list_versions("btc_1m") == ()
    assert registry.list_artifacts() == ()
    with pytest.raises(ArtifactNotFoundError):
        registry.resolve_latest("btc_1m")


def test_listing_covers_every_model_and_orders_it_by_id_then_version(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.save(fake_artifact("eth_5m", 2))
    registry.save(fake_artifact("btc_1m", 1))
    registry.save(fake_artifact("eth_5m", 1))
    assert registry.list_ids() == ("btc_1m", "eth_5m")
    assert [(m.registry_id, m.version) for m in registry.list_artifacts()] == [
        ("btc_1m", 1),
        ("eth_5m", 1),
        ("eth_5m", 2),
    ]


def test_the_metadata_sits_beside_the_artifact_where_a_human_can_read_it(tmp_path):
    registry = ModelRegistry(tmp_path)
    directory = registry.save(fake_artifact())
    assert directory == tmp_path / "btc_1m" / "v0000000001"
    assert json.loads((directory / "meta.json").read_text())["feature_spec_ref"] == FEATURE_SPEC
    assert registry.artifact_path("btc_1m") == directory / "model.json"


# ── the metadata record ───────────────────────────────────────────────────────


def test_a_registry_id_that_escapes_the_root_is_refused(tmp_path):
    registry = ModelRegistry(tmp_path)
    for bad in ("../../etc", "a/b", "", ".hidden", "BTC", "x" * 65):
        with pytest.raises(ValueError, match="registry_id"):
            registry.list_versions(bad)


def test_version_zero_is_refused_as_indistinguishable_from_unset():
    with pytest.raises(ValueError, match="unset"):
        meta(version=0).validate()
    with pytest.raises(TypeError, match="int"):
        meta(version=1.0).validate()


def test_a_version_the_signal_record_cannot_carry_is_refused():
    # model_version is a u32 on the wire; a version that does not fit is a version
    # no fill can be traced back to.
    meta(version=2**32 - 1).validate()
    with pytest.raises(ValueError, match="u32"):
        meta(version=2**32).validate()


def test_an_artifact_without_a_feature_spec_ref_is_refused():
    with pytest.raises(ValueError, match="feature_spec_ref"):
        meta(feature_spec_ref="").validate()


def test_an_artifact_that_was_never_round_tripped_cannot_be_registered(tmp_path):
    # The registry is the last place that can insist on evidence, so an artifact
    # assembled by hand — no export, no verification — is refused here.
    verified = fake_artifact()
    unverified = Artifact(
        meta=ArtifactMeta.from_dict({**verified.meta.to_dict(), "roundtrip_rows": 0}),
        payload=verified.payload,
    )
    with pytest.raises(ValueError, match="never round-trip verified"):
        ModelRegistry(tmp_path).save(unverified)


def test_a_score_output_that_names_nothing_is_refused():
    bad = ArtifactMeta.from_dict({**fake_artifact().meta.to_dict(), "score_output": "label"})
    with pytest.raises(ValueError, match="score_output"):
        bad.validate_complete()


def test_metadata_round_trips_through_json_without_losing_the_schema():
    original = fake_artifact().meta
    restored = ArtifactMeta.from_json(original.to_json())
    assert restored == original
    assert restored.inputs[0].shape == (None, 3)
    # Canonical: the same record must serialize to the same bytes, or two
    # registries holding the same model look different to a diff.
    assert restored.to_json() == original.to_json()


def test_an_unknown_metadata_key_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown"):
        ArtifactMeta.from_dict({**fake_artifact().meta.to_dict(), "quantized": True})


def test_metadata_from_a_newer_axon_is_refused_rather_than_half_read():
    future = {**fake_artifact().meta.to_dict(), "meta_schema": 99}
    with pytest.raises(ValueError, match="schema 99"):
        ArtifactMeta.from_dict(future)


def test_the_content_hash_names_its_algorithm_and_covers_every_byte():
    assert content_hash(b"abc").startswith("sha256:")
    assert content_hash(b"abc") != content_hash(b"abd")
    assert len(content_hash(b"")) == len("sha256:") + 64


def test_a_git_sha_is_recorded_even_outside_a_checkout(tmp_path):
    # An artifact built outside a repo is less reproducible; saying "unknown" is
    # honest, while a blank field reads as "nobody looked".
    assert current_git_sha(tmp_path) == "unknown"
    assert current_git_sha()


# ── the FP32 audit (ADR-0005) ─────────────────────────────────────────────────


def _linear_model(dtype=np.float32, opset: int = 17):
    """A one-node ``MatMul`` graph, the smallest thing that can carry a dtype."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    weights = np.array([[1.5], [-0.5]], dtype=dtype)
    elem = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "w"], ["score"])],
        "linear",
        [helper.make_tensor_value_info("input", elem, [None, 2])],
        [helper.make_tensor_value_info("score", elem, [None, 1])],
        initializer=[numpy_helper.from_array(weights, name="w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, weights


def test_an_fp32_graph_passes_the_audit():
    model, _ = _linear_model()
    audit_fp32(model)
    audit_fp32(model.SerializeToString())  # the round trip ADR-0005 asks for


def test_an_fp16_initializer_is_refused_by_the_audit():
    model, _ = _linear_model(np.float16)
    with pytest.raises(PrecisionError, match="FLOAT16"):
        audit_fp32(model.SerializeToString())


def test_a_cast_into_fp16_is_refused_by_the_audit():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(np.array([[1.5], [-0.5]], dtype=np.float32), name="w")
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["half"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["half"], ["back"], to=TensorProto.FLOAT),
            helper.make_node("MatMul", ["back", "w"], ["score"]),
        ],
        "downcast",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    # This is the ADR-0005 footgun exactly: FP32 in, FP32 out, and a prediction
    # that has been through half precision on the way. Nothing about the
    # signature gives it away.
    with pytest.raises(PrecisionError, match="casts to FLOAT16"):
        audit_fp32(model.SerializeToString())


def test_a_quantized_operator_is_refused_by_the_audit():
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["input", "scale"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "scale"], ["score"]),
        ],
        "quantized",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    with pytest.raises(PrecisionError, match="quantized"):
        audit_fp32(model.SerializeToString())


def test_fp16_hidden_inside_a_subgraph_is_still_refused():
    # An `If` branch is where a conversion tool puts the half-precision path, and
    # an audit that stops at the top level calls the graph clean.
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    half = numpy_helper.from_array(np.array([1.0], dtype=np.float16), name="half")
    branch = helper.make_graph(
        [helper.make_node("Identity", ["half"], ["out"])],
        "then",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1])],
        initializer=[half],
    )
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["score"], then_branch=branch, else_branch=branch)],
        "outer",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT16, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    with pytest.raises(PrecisionError, match="FLOAT16"):
        audit_fp32(model)


def test_the_serving_session_disables_optimization_and_extra_threads():
    ort = pytest.importorskip("onnxruntime")
    options = session_options()
    # opt_level=0 (ADR-0003/0005): a fused graph is a different graph, and the
    # parity gate compared the unfused one.
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # One thread: FP addition is not associative, so parallel reductions let a
    # session disagree with itself between runs.
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1


# ── export round trips ────────────────────────────────────────────────────────


def test_an_xgboost_artifact_reloads_to_bit_identical_predictions(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBRegressor(n_estimators=8, max_depth=3, n_jobs=1, random_state=0).fit(x, y)

    artifact = export_artifact(model, meta("btc_1m_xgb"), x)
    assert artifact.meta.kind == "xgboost"
    assert artifact.meta.roundtrip_max_abs_diff == 0.0  # trees are exact or the format is lossy

    reloaded = load_predictor(artifact).predict(x)
    assert np.std(reloaded) > 0  # a constant model would pass every check below
    assert np.array_equal(reloaded, model.predict(x))


def test_a_lightgbm_artifact_reloads_to_bit_identical_predictions(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    x, y = _xy()
    model = lgb.LGBMRegressor(
        n_estimators=8, num_leaves=7, min_child_samples=5, n_jobs=1, random_state=0, verbose=-1
    ).fit(x, y)

    artifact = export_artifact(model, meta("btc_1m_lgb"), x)
    assert artifact.meta.kind == "lightgbm"
    reloaded = load_predictor(artifact).predict(x)
    assert np.std(reloaded) > 0
    assert np.array_equal(reloaded, model.predict(x))


def test_a_tree_classifier_serves_the_probability_and_not_a_baked_in_label():
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, n_jobs=1, random_state=0).fit(
        x, (y > 0).astype(int)
    )

    artifact = export_artifact(model, meta("btc_xgb_clf"), x)
    scores = load_predictor(artifact).predict(x)
    # Where to put the threshold is the strategy's decision (and the thing the
    # parity gate checks for invariance); an artifact that returns a label has
    # already made it, out of sight of both.
    assert np.array_equal(scores, model.predict_proba(x)[:, 1])
    assert artifact.meta.score_output == "score"


def test_trees_are_exported_native_and_never_routed_through_onnx():
    # ADR-0003: a tree ensemble converted to ONNX is re-expressed as a different
    # graph with its own float handling, and a 1e-7 difference at a split moves a
    # sample to the other branch.
    xgb = pytest.importorskip("xgboost")
    lgb = pytest.importorskip("lightgbm")
    x, y = _xy()

    booster = export_artifact(
        xgb.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=0).fit(x, y),
        meta("btc_xgb"),
        x,
    )
    assert booster.meta.artifact_filename == "model.json"
    assert json.loads(booster.payload)["learner"]  # xgboost's own JSON, not a proto

    leaves = export_artifact(
        lgb.LGBMRegressor(
            n_estimators=4, num_leaves=7, min_child_samples=5, n_jobs=1, random_state=0, verbose=-1
        ).fit(x, y),
        meta("btc_lgb"),
        x,
    )
    assert leaves.meta.artifact_filename == "model.txt"
    assert leaves.payload.startswith(b"tree")


def test_an_sklearn_classifier_survives_onnx_export_without_flipping_a_label():
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    from sklearn.linear_model import LogisticRegression

    x, y = _xy()
    labels = (y > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(x, labels)

    artifact = export_artifact(model, meta("btc_logit"), x)
    assert artifact.meta.kind == "onnx"
    assert artifact.meta.roundtrip_max_abs_diff < 1e-5
    # The label output exists, but the strategy acts on the probability; naming
    # the wrong one hands it a class index and calls it a score.
    assert artifact.meta.score_output == "probabilities"
    assert [s.name for s in artifact.meta.outputs] == ["label", "probabilities"]

    predictor = load_predictor(artifact)
    assert np.array_equal(
        predictor.predict(x).argmax(axis=1), model.predict_proba(x).argmax(axis=1)
    )
    assert np.array_equal(predictor.run(x)["label"], model.predict(x))


def test_a_neural_net_export_stays_within_fp32_tolerance():
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    from sklearn.neural_network import MLPRegressor

    x, y = _xy()
    model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=4000, random_state=0).fit(x, y)

    artifact = export_artifact(model, meta("btc_mlp"), x)
    assert 0.0 < artifact.meta.roundtrip_max_abs_diff < 1e-5  # never bit-exact, always bounded
    reloaded = load_predictor(artifact).predict(x).reshape(-1)
    assert np.max(np.abs(reloaded - model.predict(x))) < 1e-5


def test_the_recorded_schema_is_the_one_the_artifact_actually_accepts():
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    from sklearn.linear_model import LinearRegression

    x, y = _xy(cols=4)
    artifact = export_artifact(LinearRegression().fit(x, y), meta("btc_lin"), x)
    (spec,) = artifact.meta.inputs
    assert spec.dtype == "float32" and spec.shape == (None, 4)
    assert artifact.meta.producer["opset"].isdigit()

    with pytest.raises(Exception):  # the artifact enforces its own width
        load_predictor(artifact).predict(np.zeros((2, 5), dtype=np.float32))


def test_a_pre_built_onnx_graph_cannot_be_verified_without_its_eager_model():
    model, _ = _linear_model()
    with pytest.raises(ExportError, match="reference"):
        export_artifact(model, meta("hand_built"), np.ones((4, 2), dtype=np.float32))


def test_a_pre_built_onnx_graph_is_verified_against_the_reference_it_came_from():
    pytest.importorskip("onnxruntime")
    model, weights = _linear_model()
    x = np.array([[1.0, 2.0], [3.0, -4.0], [0.5, 0.25]], dtype=np.float32)

    artifact = export_artifact(model, meta("hand_built"), x, reference=lambda rows: rows @ weights)
    assert artifact.meta.kind == "onnx"
    assert np.allclose(load_predictor(artifact).predict(x), x @ weights)


def test_an_export_that_stops_matching_its_reference_is_refused():
    pytest.importorskip("onnxruntime")
    model, _ = _linear_model()
    x = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    with pytest.raises(FidelityError, match="max_abs_diff"):
        export_artifact(model, meta("hand_built"), x, reference=lambda rows: np.zeros(len(rows)))


def test_an_fp16_graph_never_becomes_an_artifact():
    pytest.importorskip("onnx")
    model, weights = _linear_model(np.float16)
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(PrecisionError):
        export_artifact(model, meta("half"), x, reference=lambda rows: rows @ weights)


def test_a_mislabeled_kind_is_refused_so_the_wrong_backend_cannot_load_it():
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=0).fit(x, y)
    with pytest.raises(ExportError, match="kind"):
        export_artifact(model, meta("btc_xgb", kind="onnx"), x)


def test_a_non_finite_sample_makes_verification_vacuous_and_is_refused():
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=0).fit(x, y)
    bad = x.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ExportError, match="non-finite"):
        export_artifact(model, meta("btc_xgb"), bad)


def test_an_unsupported_model_family_is_named_rather_than_half_exported():
    class HomeGrown:
        def predict(self, x):
            return np.zeros(len(x))

    with pytest.raises(ExportError, match="no export path"):
        export_artifact(HomeGrown(), meta("diy"), np.ones((2, 2), dtype=np.float32))


# ── end to end ────────────────────────────────────────────────────────────────


def test_a_registered_model_predicts_identically_after_a_reload(tmp_path):
    """The property the whole module exists for: a version resolves to the model.

    Train, export, register, forget the model, load the version back, predict —
    and get the same numbers. That is what makes a live decision reproducible
    offline, and everything else here is in service of it.
    """
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBRegressor(n_estimators=12, max_depth=3, n_jobs=1, random_state=0).fit(x, y)
    expected = model.predict(x)

    registry = ModelRegistry(tmp_path)
    version = registry.next_version("btc_1m_xgb")
    registry.save(export_artifact(model, meta("btc_1m_xgb", version), x))
    del model

    loaded = registry.load("btc_1m_xgb")
    assert loaded.meta.version == version
    assert loaded.meta.feature_spec_ref == FEATURE_SPEC
    assert loaded.meta.git_sha
    assert np.array_equal(registry.load_predictor("btc_1m_xgb").predict(x), expected)


def test_the_one_call_export_writes_a_loadable_artifact(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y = _xy()
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=0).fit(x, y)

    path = export(model, meta("btc_1m_xgb"), str(tmp_path), sample_input=x)
    assert path.endswith("model.json")
    assert ModelRegistry(tmp_path).load("btc_1m_xgb", 1).meta.kind == "xgboost"
