"""Regenerate the committed cross-language parity bundles (ADR-0021).

This is the Python half of the *cross-language* model-parity gate. It trains tiny
models, exports them through the real registry path (`axon.models.export_artifact`
→ `ModelRegistry.save`), and writes each one out as a parity bundle: the artifact's
own bytes, a holdout matrix, Python's own scores over it and Python's own discretized
decisions (`axon.parity.rust_gate`). `tests/cross_language_parity.rs` then asserts
that this crate reproduces those answers.

It is deliberately not run by CI, and neither the bundles nor this script are on the
gate's path. The whole point is to catch **Rust drifting away from a frozen Python
answer**, and a reference regenerated in the same breath as the assertion proves
nothing — it would agree with whatever Rust had just started doing. Regenerating is a
deliberate, reviewable event; the diff is the review.

Run from the repo root with the project venv:

    PYTHONPATH=python .venv/bin/python crates/axon-model/tests/bundles/generate.py

Every number here comes from a named seed, so a re-run on the same library stack
reproduces the same bytes and produces an empty diff. A *different* stack (xgboost,
onnx, sklearn, skl2onnx) will change the committed answers, which is exactly the
event ADR-0019 says must be visible rather than silent — the manifest records the
versions that scored each reference for that reason.

These models are trained on seeded gaussians, not on market data, so their
`feature_spec_ref` names the probe recipe that generated the columns rather than
borrowing a real `axon.features` spec id. A manifest that claimed `PERP_CORE_V1` fed
these would be a lie in exactly the field an audit trusts.

**`lgbm_binary/` is not written by this script.** It is a byte-identical copy of
`python/tests/fixtures/lightgbm-onnx/lgbm_binary/`, written by P3's LightGBM
generator, promoted here so the one crossing this phase turns on is provable by
`cargo test` on a machine with no Python at all. Do not be alarmed when a re-run of
this script writes three bundles into a directory holding four — regenerate that one
from its own generator, and re-copy. The duplication is deliberate: ADR-0021 §4 says
a gate that reaches outside its own directory is a gate that passes on one machine,
and twelve kilobytes is the correct price for not reaching.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile

import numpy as np

from axon.models import ArtifactMeta, ModelRegistry, export_artifact
from axon.parity.rust_gate import (
    ONNX_EPS,
    Criterion,
    quantile_decision,
    write_bundle_from_registry,
)

HERE = pathlib.Path(__file__).resolve().parent

# Distinct per bundle, and asserted on the Rust side against the version read out of
# the artifact itself: a bundle whose reference came from other bytes than the ones it
# ships is the failure the version field exists to catch (ADR-0019 §4).
VERSIONS = {"tree_identity": 21, "tree_logistic": 22, "mlp_regressor": 23}

# skl2onnx's default opset moves with the installed package; pinning keeps a
# regeneration on the same stack byte-identical.
TARGET_OPSET = 17

# The decision boundary sits at the 40th/60th percentile of Python's own scores, so
# roughly 40% of the holdout is short, 20% flat and 40% long. A threshold outside the
# score range would make every row decide the same way, and a decision-invariance
# check that can only come out one way is a decoration.
DECISION = quantile_decision(lower=0.4, upper=0.6)


def probe_ref(name: str, recipe: dict) -> str:
    """A `FeatureSpec.ref`-shaped id for a synthetic probe: `name/vN#fingerprint`.

    Same shape as `axon.features.FeatureSpec.ref` (16 hex = 64 bits of SHA-256 over
    the canonical recipe) so the field parses the same way everywhere, and same
    property: change how the columns are generated and the id moves.
    """
    body = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    return f"{name}/v1#{hashlib.sha256(body).hexdigest()[:16]}"


def gaussian(seed: int, rows: int, dim: int, missing: float = 0.0) -> np.ndarray:
    """Seeded feature rows, with `missing` of the cells knocked out to NaN.

    The knocked-out cells are what exercise the tree backend's default-direction
    branch: `NaN < threshold` is false, so a reader that compares before testing for
    NaN agrees with XGBoost only on the nodes whose default happens to be right, and
    only on the rows that have missing data. A NaN-free holdout gates none of that.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    if missing > 0.0:
        x[rng.random((rows, dim)) < missing] = np.nan
    return x


def tree_bundle(name: str, objective: str, seed: int) -> None:
    import xgboost as xgb

    dim = 5
    version = VERSIONS[name]
    train = gaussian(seed, 500, dim, missing=0.15)
    linear = np.nan_to_num(train) @ np.array([1.5, -2.0, 0.3, 0.0, 0.7], dtype=np.float32)

    if objective.startswith("binary:"):
        y = (linear > 0.0).astype(np.int32)
        est = xgb.XGBClassifier(n_estimators=8, max_depth=3, random_state=0, objective=objective)
        # Left to estimate its own intercept, XGBClassifier fits a base_score that is
        # not 0.5 — which is what makes the logit link non-trivial and puts the f32
        # `-logf(1/p - 1)` expression under the gate. A default 0.5 intercept would
        # test nothing about the link.
        est.set_params(base_score=None)
    else:
        y = linear
        est = xgb.XGBRegressor(n_estimators=8, max_depth=3, random_state=0, objective=objective)
    est.fit(train, y)

    booster = est.get_booster()
    # XGBoost's JSON has no first-class version slot, so the artifact version rides in
    # the learner attribute map, which round-trips through save/load. `save_raw` keeps
    # it, and the Rust loader refuses an artifact without it.
    booster.set_attr(axon_model_version=str(version))

    recipe = {"kind": "gaussian", "seed": seed, "dim": dim, "missing": 0.15}
    meta = ArtifactMeta(
        registry_id=f"parity_{name}",
        version=version,
        feature_spec_ref=probe_ref("synthetic_probe", recipe),
    )
    # The verification sample must be finite (a NaN row makes the round-trip
    # comparison vacuous), so it is drawn without missing values; the *holdout* the
    # gate scores is the one that carries them.
    artifact = export_artifact(est, meta, gaussian(seed + 1, 32, dim))

    with tempfile.TemporaryDirectory() as root:
        registry = ModelRegistry(root)
        registry.save(artifact)
        write_bundle_from_registry(
            registry,
            meta.registry_id,
            version,
            features=gaussian(seed + 2, 128, dim, missing=0.2),
            out_dir=HERE / name,
            decision=DECISION,
            overwrite=True,
        )


def onnx_bundle(name: str, seed: int) -> None:
    import onnx  # noqa: F401  (imported so a missing onnx fails here, not deeper)
    from skl2onnx import to_onnx
    from sklearn.neural_network import MLPRegressor

    dim = 4
    version = VERSIONS[name]
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(600, dim)).astype(np.float32)
    y = (np.sin(train[:, 0]) + 0.5 * train[:, 1] - train[:, 2] * train[:, 3]).astype(np.float32)
    net = MLPRegressor(hidden_layer_sizes=(8, 6), activation="relu", max_iter=900, random_state=0)
    net.fit(train, y)

    proto = to_onnx(net, train[:1], target_opset=TARGET_OPSET)
    # ONNX has a first-class version field, unlike XGBoost. skl2onnx leaves it at 0,
    # which reads as "unset" and is refused by the Rust loader, so it is stamped here.
    proto.model_version = version
    proto.doc_string = "axon cross-language parity bundle"

    recipe = {"kind": "gaussian", "seed": seed, "dim": dim, "missing": 0.0}
    meta = ArtifactMeta(
        registry_id=f"parity_{name}",
        version=version,
        feature_spec_ref=probe_ref("synthetic_probe", recipe),
    )
    # A pre-built graph has no in-memory counterpart of its own, so the export needs
    # the eager model as the reference — otherwise it would verify the graph against
    # itself and record that as evidence.
    artifact = export_artifact(proto, meta, train[:32], reference=net.predict)

    with tempfile.TemporaryDirectory() as root:
        registry = ModelRegistry(root)
        registry.save(artifact)
        write_bundle_from_registry(
            registry,
            meta.registry_id,
            version,
            features=rng.normal(size=(96, dim)).astype(np.float32),
            out_dir=HERE / name,
            decision=DECISION,
            # The family ceiling, declared rather than defaulted. ONNX_TIGHT_EPS is an
            # absolute bound meaning "two ULP at 1.0"; this net's scores reach 3.83,
            # where the same constant is *one* ULP — so defaulting would ask a neural
            # net for bit equality on a runtime that reassociates matmuls. See the note
            # in `cross_language_parity.rs`.
            criterion=Criterion("max_abs_diff", ONNX_EPS),
            overwrite=True,
        )


def main() -> None:
    tree_bundle("tree_identity", "reg:squarederror", seed=1101)
    tree_bundle("tree_logistic", "binary:logistic", seed=2203)
    onnx_bundle("mlp_regressor", seed=3307)


if __name__ == "__main__":
    main()
