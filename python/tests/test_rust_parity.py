"""``axon.parity.rust_gate``: the bundle that carries a question across the boundary.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and deterministic:
fixed seeds, no clock, no network.

The Rust half of this gate lives in ``crates/axon-model/tests/cross_language_parity.rs``
and runs in the default ``cargo test`` gate with no Python installed at all. This file
covers the side Rust cannot: that the bundle a research machine *writes* records the
answer Python actually gave, over exactly the bytes it wrote down.

Two tests are the point of the file.
``test_recorded_predictions_are_the_scores_of_the_bytes_that_were_written`` is the
serialization guarantee — if the reference were taken over the in-memory array instead
of over the file, the two languages would be answering slightly different questions and
the gate would report the difference as a model defect.
``test_the_frozen_reference_is_still_what_this_python_stack_says`` is the other
direction: a library upgrade that moves the committed answers has to be a visible,
reviewable event rather than a silent one.

The heavy ML stack is optional (``pyproject`` puts it in the ``ml`` extra), so every
test that needs one of those libraries ``importorskip``s it *inside* the test. A bare
numpy + pytest environment still runs the whole bundle-format half, which is where the
correctness properties live.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from axon.models import Artifact, ArtifactMeta, TensorSpec, content_hash
from axon.parity.rust_gate import (
    BUNDLE_SCHEMA,
    ONNX_EPS,
    ONNX_TIGHT_EPS,
    BundleError,
    Criterion,
    Decision,
    ParityBundle,
    bundle_dirs,
    python_scores,
    quantile_decision,
    read_parity_bundle,
    write_parity_bundle,
)
from axon.parity.rust_gate import _f32_bytes, _read_f32

#: The committed bundles the Rust gate runs on, relative to the repo root. Read here
#: as well so that a bundle which no longer describes itself fails on both sides.
BUNDLES = "crates/axon-model/tests/bundles"

FEATURE_SPEC = "synthetic_probe/v1#0f1e2d3c4b5a6978"


def committed(repo_root) -> tuple:
    dirs = bundle_dirs(repo_root / BUNDLES)
    assert dirs, f"no parity bundles under {BUNDLES}; the Rust gate has nothing to run"
    return dirs


def copy_bundle(source, tmp_path):
    """A writable copy of a committed bundle, for the tests that must break one."""
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    return destination


def edit_manifest(directory, mutate) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fake_artifact(kind: str, payload: bytes = b'{"not":"a model"}') -> Artifact:
    """An artifact whose payload no ML library ever parses.

    Lets the checks that run *before* scoring — the kind, the holdout's own validity —
    be tested in a bare numpy environment, which is where they matter most: they are
    the part that has to hold on a machine that cannot load the model.
    """
    return Artifact(
        meta=ArtifactMeta(
            registry_id="probe",
            version=3,
            kind=kind,
            feature_spec_ref=FEATURE_SPEC,
            git_sha="0" * 40,
            inputs=(TensorSpec("input", "float32", (None, 2)),),
            outputs=(TensorSpec("score", "float32", (None,)),),
            score_output="score",
            content_sha256=content_hash(payload),
            content_bytes=len(payload),
            artifact_filename="model.json" if kind != "onnx" else "model.onnx",
            roundtrip_rows=8,
        ),
        payload=payload,
    )


# ── the serialization guarantee ───────────────────────────────────────────────


def test_a_feature_matrix_survives_the_boundary_as_the_exact_bits_it_was_written_with(tmp_path):
    # Written and read through the private primitives on purpose: these are values a
    # trained model would rarely produce and the format has to carry anyway. A decimal
    # round trip loses the NaN payload, collapses -0.0 into 0.0, and can move a
    # denormal — and any of those sends a row down the other side of a split.
    awkward = np.array(
        [
            [np.nan, -0.0, 0.0, 5e-324],
            [np.float32(1.0000001), -3.4028235e38, 1.1754944e-38, 7.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "features.f32"
    path.write_bytes(_f32_bytes(awkward))
    back = _read_f32(path, 2, 4)
    assert back.dtype == np.float32
    assert np.array_equal(back.view(np.uint32), awkward.view(np.uint32))


def test_a_matrix_file_that_lost_bytes_is_refused_rather_than_read_as_a_smaller_one(tmp_path):
    path = tmp_path / "features.f32"
    path.write_bytes(_f32_bytes(np.zeros((4, 3), dtype=np.float32)))
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(BundleError, match="bytes"):
        _read_f32(path, 4, 3)


# ── the decision rule ─────────────────────────────────────────────────────────


def test_thresholds_are_rounded_to_float32_so_both_languages_share_one_boundary():
    # The thresholds cross to Rust as 32-bit patterns. Discretizing here on the float64
    # value and there on its f32 neighbour puts the two decision boundaries a ULP
    # apart, and every score in between decides differently on each side — a
    # disagreement manufactured by the gate itself.
    rule = Decision(long_at=0.1, short_at=-0.3)
    assert rule.long_at == float(np.float32(0.1)) != 0.1
    assert rule.short_at == float(np.float32(-0.3))


def test_inverted_thresholds_are_refused_rather_than_deciding_every_row_twice():
    with pytest.raises(BundleError, match="short_at < long_at"):
        Decision(long_at=-1.0, short_at=1.0)
    with pytest.raises(BundleError, match="short_at < long_at"):
        Decision(long_at=0.5, short_at=0.5)


def test_a_score_exactly_on_a_threshold_takes_the_position_not_the_flat_band():
    # `>=` and `<=`, the same rule as `axon.parity.threshold_discretizer` and the same
    # rule the Rust reader implements. Getting the boundary backwards flips every row
    # that lands exactly on a threshold — rare in a logit, routine in a rounded score.
    rule = Decision(long_at=0.5, short_at=-0.5)
    assert rule.sides([0.5, -0.5, 0.0, np.nan]).tolist() == [1, -1, 0, 0]


def test_quantiles_that_collapse_to_one_float32_are_refused():
    # A holdout with no spread gives a threshold pair Rust cannot even represent
    # apart, and every row would decide the same way.
    with pytest.raises(BundleError, match="spread"):
        quantile_decision()(np.full(64, 0.25))


# ── what a bundle refuses to be written from ──────────────────────────────────


def test_a_graph_holdout_containing_nan_is_refused_because_the_comparison_is_vacuous(tmp_path):
    # Only the tree backend routes missing values. A NaN into a graph comes back as a
    # NaN score, and `nan > eps` is False on both sides — the gate would report a
    # perfect match forever.
    holdout = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(BundleError, match="NaN"):
        write_parity_bundle(
            fake_artifact("onnx"),
            holdout,
            out_dir=tmp_path / "b",
            decision=Decision(long_at=1.0, short_at=-1.0),
        )


def test_an_infinite_feature_is_refused_because_no_backend_treats_it_as_missing(tmp_path):
    holdout = np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32)
    with pytest.raises(BundleError, match="infinit"):
        write_parity_bundle(
            fake_artifact("xgboost"),
            holdout,
            out_dir=tmp_path / "b",
            decision=Decision(long_at=1.0, short_at=-1.0),
        )


def test_a_family_with_no_rust_backend_is_refused_by_name(tmp_path):
    # LightGBM artifacts are legal in the registry and have no Rust reader (ADR-0019's
    # documented gap). Saying so beats writing a bundle nothing can ever run.
    with pytest.raises(BundleError, match="lightgbm"):
        write_parity_bundle(
            fake_artifact("lightgbm"),
            np.ones((2, 2), dtype=np.float32),
            out_dir=tmp_path / "b",
            decision=Decision(long_at=1.0, short_at=-1.0),
        )


def test_the_unservable_kind_refusal_names_the_route_out_of_it(tmp_path):
    # ADR-0033: the refusal above is now a signpost, not a dead end. A reader who
    # stops at "no Rust backend serves lightgbm" reaches for a native backend that
    # does not need building — the model crosses by conversion, as an `onnx`
    # artifact, and the message has to say so where the refusal is read.
    with pytest.raises(BundleError, match="lightgbm_to_onnx"):
        write_parity_bundle(
            fake_artifact("lightgbm"),
            np.ones((2, 2), dtype=np.float32),
            out_dir=tmp_path / "b",
            decision=Decision(long_at=1.0, short_at=-1.0),
        )
    with pytest.raises(BundleError, match="lightgbm_to_onnx"):
        Criterion.required_for("lightgbm")


def test_writing_over_a_committed_bundle_needs_saying_so(tmp_path, repo_root):
    source = copy_bundle(committed(repo_root)[0], tmp_path)
    with pytest.raises(BundleError, match="overwrite"):
        write_parity_bundle(
            fake_artifact("xgboost"),
            np.ones((2, 2), dtype=np.float32),
            out_dir=source,
            decision=Decision(long_at=1.0, short_at=-1.0),
        )


# ── what a bundle refuses to be read as ───────────────────────────────────────


def test_every_committed_bundle_verifies_against_its_own_manifest(repo_root):
    for directory in committed(repo_root):
        bundle = read_parity_bundle(directory)
        assert isinstance(bundle, ParityBundle)
        assert bundle.model_version >= 1
        assert bundle.feature_spec_ref
        assert bundle.features.shape[0] == bundle.predictions.shape[0] > 0
        assert np.isfinite(bundle.predictions).all()
        # The decision half of the gate is only a test if the corpus decides more than
        # one way; a holdout that is flat everywhere would pass it unconditionally.
        assert len(set(bundle.decisions.tolist())) > 1, directory


def test_committed_tree_bundles_carry_missing_values(repo_root):
    # The default-direction branch is gated by the corpus, not by the assertion.
    # ADR-0019 §2: `NaN < threshold` is false in IEEE-754, so a reader that
    # compares before testing for NaN agrees with XGBoost on exactly the nodes
    # whose default happens to be right, and only on rows that actually have
    # missing data. A NaN-free corpus gates none of that.
    #
    # `any`, not `all`, and the difference is a real bundle rather than a
    # convenience: a tree bundle scored over a *real* feature matrix with a finite
    # lookback legitimately has no missing cells, and demanding NaN from every
    # tree bundle would mean bending that corpus to fit this test. What has to
    # hold is that the branch is covered somewhere, and the covering bundles are
    # named on failure so losing the last one cannot read as a pass.
    trees = {d.name: read_parity_bundle(d).features for d in committed(repo_root)
             if read_parity_bundle(d).kind == "xgboost"}
    assert trees, "no tree bundle is committed; the exact half of the gate is unmanned"
    carrying = sorted(name for name, x in trees.items() if np.isnan(x).any())
    assert carrying, (
        f"no committed tree bundle carries a missing value ({sorted(trees)}); XGBoost's "
        "default-direction branch is unexercised on both sides of the gate"
    )


def test_a_bundle_whose_features_were_edited_after_the_fact_is_refused(tmp_path, repo_root):
    directory = copy_bundle(committed(repo_root)[0], tmp_path)
    path = directory / "features.f32"
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(BundleError, match="content hash"):
        read_parity_bundle(directory)


def test_a_bundle_cannot_declare_a_looser_tolerance_than_its_family_allows(tmp_path, repo_root):
    # The manifest is not hashed, so this edit leaves every file consistent: it has to
    # be caught on meaning. A bundle regenerated after a red gate with the tolerance
    # nudged until it passed would otherwise look exactly like one that never failed.
    for directory in committed(repo_root):
        copy = copy_bundle(directory, tmp_path)
        edit_manifest(copy, lambda m: m.update(criterion={"kind": "max_abs_diff", "eps": 0.01}))
        with pytest.raises(BundleError, match="held to"):
            read_parity_bundle(copy)


def test_a_graph_bundle_declares_what_graphs_here_achieve_not_the_family_ceiling():
    # ADR-0003 §3's 1e-5 is a ceiling for the *family*; every graph gated in this repo
    # sits four to five orders of magnitude inside it. Declaring the ceiling anyway
    # leaves ~42x of slack, and slack is where a regression passes green — which is the
    # same argument ADR-0019 §1 makes for pinning `tract` to an exact version.
    assert Criterion.declared_for("onnx") == Criterion("max_abs_diff", ONNX_TIGHT_EPS)
    assert ONNX_TIGHT_EPS < ONNX_EPS
    # Two ULP at 1.0, anchored to float32's own resolution rather than fitted to a
    # measurement. A tolerance derived from what today's runtime produced is one that
    # ratchets: regenerate after a regression and it records the regression as the bar.
    assert ONNX_TIGHT_EPS == 2.0**-22

    # Trees are untouched: a tolerance there would be the gate declining to test
    # ADR-0019's own exactness claim.
    assert Criterion.declared_for("xgboost") == Criterion("bit_exact")

    # And tightening needed no reader change on either side, because `allows` was
    # already written to permit it.
    assert Criterion.required_for("onnx").allows(Criterion.declared_for("onnx"))


def test_every_committed_graph_bundle_declares_the_tightened_criterion(repo_root):
    # The bundles are the thing the Rust gate actually runs on, so a regeneration that
    # quietly fell back to the family ceiling would restore the slack with nothing to
    # notice.
    #
    # `mlp_regressor` is the documented exception, and it is named here rather than
    # skipped so that a *second* bundle drifting to the ceiling still reddens.
    # ONNX_TIGHT_EPS is an absolute bound meaning "two ULP at 1.0", so what it demands
    # depends on the scores it meets: four ULP on a probability in [0, 1], but exactly
    # *one* on this net, whose scores reach 3.83. It had been tightened off 1e-5 because
    # this machine measured 0e0 — and that is the reading ADR-0021 refuses, since a
    # runner whose `tract` picks a different matmul kernel measures two ULP and reddens
    # with nothing wrong. A neural net is the one family this repo never claimed bits
    # for, so it is held to ADR-0003's 1e-5.
    ceiling = {"mlp_regressor": Criterion("max_abs_diff", ONNX_EPS)}
    graphs = [d for d in committed(repo_root) if read_parity_bundle(d).kind == "onnx"]
    assert graphs, "no committed graph bundles; this check would pass by vacuity"
    for directory in graphs:
        bundle = read_parity_bundle(directory)
        expected = ceiling.get(directory.name, Criterion("max_abs_diff", ONNX_TIGHT_EPS))
        assert bundle.criterion == expected, (
            f"{directory.name} declares {bundle.criterion}, not {expected}"
        )


def test_the_writer_refuses_a_criterion_looser_than_the_family_ceiling(tmp_path):
    # The writer may be asked to tighten and may never be asked to loosen, so a bundle
    # asking for a looser gate was edited by hand and both readers say so. Refused
    # before a byte is written, so a rejected argument cannot leave the half-written
    # directory that the manifest-last rule exists to make legible.
    with pytest.raises(BundleError, match="may tighten its criterion and may never loosen it"):
        write_parity_bundle(
            fake_artifact("onnx"),
            np.zeros((4, 2), dtype=np.float32),
            out_dir=tmp_path / "loose",
            decision=Decision(long_at=0.6, short_at=0.4),
            criterion=Criterion("max_abs_diff", 1e-2),
        )
    assert not (tmp_path / "loose").exists(), "a refused criterion left a directory behind"


def test_decisions_that_do_not_follow_from_the_scores_are_refused(tmp_path, repo_root):
    # Hashes kept honest on purpose: the interesting tamper is the one that keeps the
    # bundle self-consistent. If the two languages disagree about the decision *rule*
    # rather than the number, every prediction can match to the bit while the two
    # systems still trade differently.
    directory = copy_bundle(committed(repo_root)[0], tmp_path)
    path = directory / "decisions.i8"
    decisions = np.frombuffer(path.read_bytes(), dtype=np.int8).copy()
    row = int(np.flatnonzero(decisions != 0)[0])
    decisions[row] = 0
    path.write_bytes(decisions.tobytes())
    counts = {
        "short": int(np.count_nonzero(decisions == -1)),
        "flat": int(np.count_nonzero(decisions == 0)),
        "long": int(np.count_nonzero(decisions == 1)),
    }

    def relabel(manifest):
        manifest["decisions"]["sha256"] = content_hash(decisions.tobytes())
        manifest["decisions"]["counts"] = counts

    edit_manifest(directory, relabel)
    with pytest.raises(BundleError, match="do not follow from the scores"):
        read_parity_bundle(directory)


def test_a_bundle_without_a_manifest_reads_as_an_interrupted_write(tmp_path, repo_root):
    # The manifest is written last, so its absence is diagnostic: "an interrupted
    # write" and "delete this directory" have to be the same sentence.
    directory = copy_bundle(committed(repo_root)[0], tmp_path)
    (directory / "manifest.json").unlink()
    with pytest.raises(BundleError, match="interrupted write"):
        read_parity_bundle(directory)


def test_a_newer_bundle_schema_is_refused_rather_than_half_understood(tmp_path, repo_root):
    directory = copy_bundle(committed(repo_root)[0], tmp_path)
    edit_manifest(directory, lambda m: m.update(bundle_schema=BUNDLE_SCHEMA + 1))
    with pytest.raises(BundleError, match="refusing"):
        read_parity_bundle(directory)


def test_a_manifest_naming_a_file_outside_the_bundle_is_refused(tmp_path, repo_root):
    # A manifest is data. A path in it that escapes the directory turns reading a
    # bundle into a file read of whoever wrote it choosing.
    directory = copy_bundle(committed(repo_root)[0], tmp_path)
    edit_manifest(directory, lambda m: m["features"].update(file="../features.f32"))
    with pytest.raises(BundleError, match="bare filename"):
        read_parity_bundle(directory)


def test_the_criterion_a_family_demands_is_not_the_manifests_opinion():
    assert Criterion.required_for("xgboost") == Criterion("bit_exact")
    assert Criterion.required_for("onnx") == Criterion("max_abs_diff", 1e-5)
    # Tightening is allowed, loosening is the whole point.
    assert Criterion.required_for("onnx").allows(Criterion("max_abs_diff", 1e-7))
    assert not Criterion.required_for("onnx").allows(Criterion("max_abs_diff", 1e-4))
    assert not Criterion.required_for("xgboost").allows(Criterion("max_abs_diff", 1e-12))


# ── the version the Rust loaders read out of the artifact ─────────────────────


def test_an_exported_xgboost_artifact_names_its_own_version_without_being_hand_stamped(tmp_path):
    # ADR-0019 §4: both backends refuse a model that cannot name its own version,
    # and read it out of the artifact rather than from a caller. Nothing in
    # XGBoost's own serializer writes that attribute, so before the exporter did
    # it, *no artifact the export path produced could be loaded by the core* —
    # every bundle the cross-language gate ever ran on had been stamped by hand in
    # its generator, and the gate that certifies the boundary had never run on the
    # path it certifies.
    xgb = pytest.importorskip("xgboost")
    from axon.models import export_artifact

    est, train = _tiny_booster()
    est.get_booster().set_attr(axon_model_version=None)  # undo the helper's stamp
    meta = ArtifactMeta(registry_id="probe", version=4097, feature_spec_ref=FEATURE_SPEC)
    artifact = export_artifact(est, meta, np.nan_to_num(train[:16]))

    booster = xgb.Booster()
    booster.load_model(bytearray(artifact.payload))
    assert booster.attributes().get("axon_model_version") == "4097"


def test_exporting_does_not_stamp_the_callers_live_booster(tmp_path):
    # The stamp goes on a copy. `set_attr` mutates in place, and an exporter that
    # silently rewrites the caller's model changes what the next `predict` in a
    # notebook is running against — and makes a second export of the same object
    # with a different version look like a version that drifted.
    pytest.importorskip("xgboost")
    from axon.models import export_artifact

    est, train = _tiny_booster()
    est.get_booster().set_attr(axon_model_version=None)
    meta = ArtifactMeta(registry_id="probe", version=4097, feature_spec_ref=FEATURE_SPEC)
    export_artifact(est, meta, np.nan_to_num(train[:16]))
    assert est.get_booster().attributes().get("axon_model_version") is None


def test_an_exported_onnx_graph_carries_the_registry_version_not_skl2onnxs_zero(tmp_path):
    # skl2onnx leaves `model_version` at 0, which reads as unset and is refused by
    # the Rust loader — one field between a correct graph and an unloadable one.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LinearRegression

    from axon.models import export_artifact

    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 3)).astype(np.float32)
    model = LinearRegression().fit(x, x @ np.array([1.0, -2.0, 0.5], dtype=np.float32))
    meta = ArtifactMeta(registry_id="probe", version=77, feature_spec_ref=FEATURE_SPEC)
    artifact = export_artifact(model, meta, x[:8])
    assert onnx.load_from_string(artifact.payload).model_version == 77


def test_narrowing_a_classifier_keeps_the_version_stamp_it_was_given(tmp_path):
    # `narrow_to_score_column` rebuilds the graph from an empty `ModelProto`, so
    # every field it does not carry across is silently dropped. Dropping
    # `model_version` would mean the rewrite that exists to make an artifact
    # loadable is the same rewrite that makes it unloadable.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression

    from axon.models import export_artifact

    rng = np.random.default_rng(5)
    x = rng.normal(size=(128, 3)).astype(np.float32)
    y = (x @ np.array([1.0, -1.0, 0.5], dtype=np.float32) > 0).astype(np.int32)
    model = LogisticRegression(max_iter=500).fit(x, y)
    meta = ArtifactMeta(registry_id="probe", version=78, feature_spec_ref=FEATURE_SPEC)
    artifact = export_artifact(model, meta, x[:16], narrow_score_output=True)

    graph = onnx.load_from_string(artifact.payload)
    assert graph.model_version == 78
    assert [o.name for o in graph.graph.output] == [artifact.meta.score_output]
    assert len(graph.graph.output) == 1
    # And the bundle writer now accepts it, where the two-output form is refused
    # for having no single score.
    directory = write_parity_bundle(
        artifact, x, out_dir=tmp_path / "bundle", decision=quantile_decision()
    )
    assert read_parity_bundle(directory).predictions.shape == (128,)


def test_exporting_one_fitted_estimator_twice_produces_one_artifact():
    # `convert_sklearn` defaults `name` to `uuid4().hex` and it lands in
    # `graph.name`, so without a pin two exports of one fitted model are two
    # different artifacts by identity: same weights, same predictions, different
    # `content_sha256`. That makes ADR-0015's content hash name a *conversion
    # event* rather than a model — `Artifact.verify()` still passes, because the
    # hash does match the bytes it was taken over, so nothing catches it — and it
    # makes any frozen reference over the artifact churn on every regeneration,
    # which is what blocked a committed sklearn parity bundle.
    #
    # The twin of `test_converting_the_same_booster_twice_produces_the_same_bytes`
    # in test_lightgbm_onnx.py: two converters, one defect, one decision.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("skl2onnx")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from axon.models import export_artifact

    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 3)).astype(np.float32)
    y = (x @ np.array([1.0, -1.0, 0.5], dtype=np.float32) > 0).astype(np.int32)
    meta = ArtifactMeta(registry_id="probe", version=80, feature_spec_ref=FEATURE_SPEC)

    # Both families, because both route through `convert_sklearn`: a pin on one
    # call site is not a property of the export path.
    for model in (
        LogisticRegression(max_iter=500),
        GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0),
    ):
        model.fit(x, y)
        first = export_artifact(model, meta, x[:16])
        second = export_artifact(model, meta, x[:16])
        names = (
            onnx.load_from_string(first.payload).graph.name,
            onnx.load_from_string(second.payload).graph.name,
        )
        assert first.payload == second.payload, (
            f"{type(model).__name__}: two exports of one fitted model differ; graph names {names}"
        )
        assert first.meta.content_sha256 == second.meta.content_sha256


def test_a_three_class_sklearn_graph_is_refused_rather_than_narrowed_to_one_of_three():
    # Same rule as LightGBM: there is no positive column among three, so choosing
    # one would be an invented trading rule wearing a conversion's clothes.
    pytest.importorskip("skl2onnx")
    from sklearn.linear_model import LogisticRegression

    from axon.models import export_artifact
    from axon.models.export import ExportError

    rng = np.random.default_rng(6)
    x = rng.normal(size=(150, 3)).astype(np.float32)
    y = np.digitize(x @ np.array([1.0, -1.0, 0.5], dtype=np.float32), [-1.0, 1.0])
    model = LogisticRegression(max_iter=500).fit(x, y)
    meta = ArtifactMeta(registry_id="probe", version=79, feature_spec_ref=FEATURE_SPEC)
    with pytest.raises(ExportError, match="no positive column among 3"):
        export_artifact(model, meta, x[:16], narrow_score_output=True)


# ── the reference itself (needs the ML stack) ─────────────────────────────────


def _tiny_booster(seed: int = 5, rows: int = 200, dim: int = 4):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    x[rng.random((rows, dim)) < 0.1] = np.nan
    y = np.nan_to_num(x) @ np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    est = xgb.XGBRegressor(n_estimators=4, max_depth=3, random_state=0)
    est.fit(x, y)
    est.get_booster().set_attr(axon_model_version="3")
    return est, x


def test_recorded_predictions_are_the_scores_of_the_bytes_that_were_written(tmp_path):
    # The serialization guarantee, end to end: the bundle's reference must be the
    # answer to the question the Rust side asks, which is the matrix *on disk*. Scoring
    # the in-memory array instead would leave the file unverified in the one respect
    # that decides whether the gate is comparing like with like.
    xgb = pytest.importorskip("xgboost")
    from axon.models import ModelRegistry, export_artifact

    est, train = _tiny_booster()
    meta = ArtifactMeta(registry_id="probe", version=3, feature_spec_ref=FEATURE_SPEC)
    artifact = export_artifact(est, meta, np.nan_to_num(train[:16]))
    registry = ModelRegistry(tmp_path / "registry")
    registry.save(artifact)

    holdout = np.random.default_rng(11).normal(size=(48, 4)).astype(np.float32)
    holdout[np.random.default_rng(12).random((48, 4)) < 0.2] = np.nan
    directory = write_parity_bundle(
        registry.load("probe", 3),
        holdout,
        out_dir=tmp_path / "bundle",
        decision=quantile_decision(),
    )

    bundle = read_parity_bundle(directory)
    booster = xgb.Booster()
    booster.load_model(bytearray(bundle.artifact().payload))
    margin = booster.predict(xgb.DMatrix(bundle.features, missing=np.nan), output_margin=True)
    recorded = bundle.predictions
    assert np.array_equal(
        recorded.view(np.uint32), np.asarray(margin, dtype=np.float32).view(np.uint32)
    ), "the recorded reference is not XGBoost's own margin over the bytes in the bundle"
    # And the margin, not the probability: TreeModel never applies the link.
    assert bundle.manifest["score_space"] == "margin"
    assert bundle.model_version == 3


def test_a_decision_rule_no_row_can_cross_is_refused_as_a_decoration(tmp_path):
    pytest.importorskip("xgboost")
    from axon.models import ModelRegistry, export_artifact

    est, train = _tiny_booster()
    meta = ArtifactMeta(registry_id="probe", version=3, feature_spec_ref=FEATURE_SPEC)
    registry = ModelRegistry(tmp_path / "registry")
    registry.save(export_artifact(est, meta, np.nan_to_num(train[:16])))
    with pytest.raises(BundleError, match="decoration"):
        write_parity_bundle(
            registry.load("probe", 3),
            train[:32],
            out_dir=tmp_path / "bundle",
            # Far above anything the model can score: every row is short, so the
            # decision half of the gate could never come out any other way.
            decision=Decision(long_at=1e9, short_at=1e8),
        )


def test_the_frozen_reference_is_still_what_this_python_stack_says(repo_root):
    # The committed answers are pinned to one Python stack (ADR-0019 says so out
    # loud). This test is what makes an upgrade that moves them a *reviewable event*
    # rather than a silent one: it fails here, in Python, naming the library — instead
    # of surfacing as an unexplained Rust parity failure months later.
    for directory in committed(repo_root):
        bundle = read_parity_bundle(directory)
        if bundle.kind == "xgboost":
            pytest.importorskip("xgboost")
        else:
            pytest.importorskip("onnxruntime")
        rescored = python_scores(bundle.artifact(), bundle.features)
        report = bundle.compare(rescored)
        report.raise_for_status()
        if bundle.kind == "xgboost":
            # `TREE_EPS = 0.0` is a numeric zero; the Rust side asserts the bits, so
            # this side has to as well or the two gates are not the same gate.
            assert np.array_equal(
                rescored.view(np.uint32), bundle.predictions.view(np.uint32)
            ), directory
