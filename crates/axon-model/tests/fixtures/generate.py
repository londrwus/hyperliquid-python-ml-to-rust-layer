"""Regenerate the axon-model parity fixtures.

This is the Python half of the model-parity gate (docs/03, ADR-0003): it trains
tiny models, exports them the way `axon.models` will, and records the *Python*
answer so the Rust test can assert it reproduces it. It is deliberately not run
by CI — the artifacts and the reference outputs are committed, because the point
of the gate is to catch Rust drifting away from a frozen Python answer, and a
reference regenerated in the same breath as the assertion proves nothing.

Run from the repo root with the project venv:

    PYTHONHASHSEED=0 .venv/bin/python \\
        crates/axon-model/tests/fixtures/generate.py

Reference outputs are stored as IEEE-754 bit patterns, not decimals. The tree
gate asserts *bit* equality, and a decimal round-trip through JSON is exactly
the kind of silent one-ULP loss the gate exists to detect — writing the
expectation in decimal would let the fixture absorb the very error it is
supposed to catch. Missing feature values are written as JSON `null` rather
than `NaN`, which is not valid JSON and which serde_json rightly refuses.
"""

from __future__ import annotations

import json
import math
import pathlib
import struct

import numpy as np
import onnx
import xgboost as xgb
from onnx import TensorProto, helper, numpy_helper
from sklearn.neural_network import MLPRegressor
from skl2onnx import to_onnx

HERE = pathlib.Path(__file__).resolve().parent

# Stamped into every artifact and asserted on the Rust side. A model that cannot
# name its own version cannot be tied back to the signals it produced, so the
# loaders refuse an artifact without one (ADR-0019).
MODEL_VERSION = 7

# skl2onnx defaults move with the installed opset; pinning keeps the committed
# graph stable across regenerations so a re-run produces a reviewable diff.
TARGET_OPSET = 17


def f32_bits(values) -> list[str]:
    """IEEE-754 bit patterns of an f32 array, as 0x-prefixed hex."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [f"0x{struct.unpack('<I', struct.pack('<f', float(v)))[0]:08x}" for v in arr]


def rows_with_nulls(x: np.ndarray) -> list[list[float | None]]:
    """Feature rows as JSON, with NaN (a missing value) written as `null`."""
    return [[None if math.isnan(float(v)) else float(v) for v in row] for row in x]


def make_features(rng: np.random.Generator, n: int, dim: int, missing: float) -> np.ndarray:
    x = rng.normal(size=(n, dim)).astype(np.float32)
    # Missing values are the whole point of the default-direction branch, so the
    # holdout rows must contain them — a NaN-free corpus exercises none of it.
    x[rng.random((n, dim)) < missing] = np.nan
    return x


def export_tree(name: str, objective: str, rng: np.random.Generator) -> dict:
    """Train a small XGBoost model, save its native JSON, record the margins."""
    dim = 5
    x = make_features(rng, 400, dim, missing=0.15)
    linear = np.nan_to_num(x) @ np.array([1.5, -2.0, 0.3, 0.0, 0.7], dtype=np.float32)

    if objective.startswith("binary:"):
        y = (linear > 0.0).astype(np.int32)
        est = xgb.XGBClassifier(
            n_estimators=6, max_depth=3, random_state=0, objective=objective
        )
        # XGBClassifier only fits a non-trivial intercept when it is left to
        # estimate one, and a 0.5 intercept would make the logit link a no-op —
        # i.e. it would not test the link handling at all.
        est.set_params(base_score=None)
    else:
        y = linear
        est = xgb.XGBRegressor(n_estimators=6, max_depth=3, random_state=0, objective=objective)

    est.fit(x, y)
    booster = est.get_booster()
    # XGBoost's JSON has no first-class version field, so the artifact version
    # rides in the learner attribute map, which round-trips through save/load.
    booster.set_attr(axon_model_version=str(MODEL_VERSION))
    booster.save_model(str(HERE / name))

    holdout = make_features(np.random.default_rng(99), 64, dim, missing=0.2)
    margin = booster.predict(xgb.DMatrix(holdout, missing=np.nan), output_margin=True)
    return {
        "artifact": name,
        "objective": objective,
        "version": MODEL_VERSION,
        "num_feature": dim,
        "inputs": rows_with_nulls(holdout),
        "expected_bits": f32_bits(margin),
    }


def export_onnx(name: str, rng: np.random.Generator) -> dict:
    """Train a tiny MLP, export FP32 ONNX, record the onnxruntime outputs."""
    import onnxruntime as ort

    dim = 4
    x = rng.normal(size=(500, dim)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] - x[:, 2] * x[:, 3]).astype(np.float32)
    net = MLPRegressor(
        hidden_layer_sizes=(8, 6), activation="relu", max_iter=800, random_state=0
    )
    net.fit(x, y)

    model = to_onnx(net, x[:1], target_opset=TARGET_OPSET)
    model.model_version = MODEL_VERSION
    model.doc_string = "axon-model parity fixture"
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)

    holdout = rng.normal(size=(32, dim)).astype(np.float32)
    # opt_level=0: the reference must be the graph as exported, not whatever a
    # runtime's fusion pass turns it into (ADR-0005).
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(HERE / name), opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = [
        sess.run(None, {input_name: row.reshape(1, dim)})[0].reshape(-1) for row in holdout
    ]
    return {
        "artifact": name,
        "version": MODEL_VERSION,
        "num_feature": dim,
        "inputs": [[float(v) for v in row] for row in holdout],
        "expected": [[float(v) for v in out] for out in outputs],
    }


def export_unversioned(name: str) -> None:
    """A perfectly valid FP32 graph that simply never had a version stamped.

    `model_version` defaults to 0, which is what an exporter that forgot leaves
    behind — indistinguishable from "version zero" and therefore refused.
    """
    w = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node],
        "unversioned",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", TARGET_OPSET)])
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)


def export_fp16_boundary(name: str) -> None:
    """A graph that is FP16 at its own input and output — the obvious case."""
    w = numpy_helper.from_array(np.eye(2, dtype=np.float16), name="w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"])
    graph = helper.make_graph(
        [node],
        "fp16_boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", TARGET_OPSET)])
    model.model_version = MODEL_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)


def export_fp16_hidden(name: str) -> None:
    """FP32 in, FP32 out, FP16 in the middle — the case that is actually dangerous.

    Nothing about this graph's signature says it will round every intermediate
    to half precision. This is the shape of the documented ONNX Runtime silent
    downcast (ADR-0005), and it is the one the loader has to catch by inspecting
    the body rather than the boundary.
    """
    w = numpy_helper.from_array(np.eye(2, dtype=np.float16), name="w")
    nodes = [
        helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["h", "w"], ["h2"]),
        helper.make_node("Cast", ["h2"], ["y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "fp16_hidden_cast",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", TARGET_OPSET)])
    model.model_version = MODEL_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, HERE / name)


def main() -> None:
    reference = {
        "note": "Generated by generate.py; see its docstring. Do not hand-edit.",
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "xgboost": xgb.__version__,
        },
        "trees": [
            export_tree("xgb_identity.json", "reg:squarederror", np.random.default_rng(11)),
            export_tree("xgb_logistic.json", "binary:logistic", np.random.default_rng(23)),
        ],
        "onnx": [export_onnx("mlp_regressor.onnx", np.random.default_rng(31))],
    }
    export_fp16_boundary("fp16_boundary.onnx")
    export_fp16_hidden("fp16_hidden_cast.onnx")
    export_unversioned("unversioned.onnx")

    with (HERE / "reference.json").open("w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")


if __name__ == "__main__":
    main()
