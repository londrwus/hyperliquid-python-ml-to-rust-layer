"""Regenerate the axon-model *refusal* fixtures.

`tests/fixtures/` holds artifacts this crate must serve correctly. This
directory holds the opposite: artifacts it must **refuse**, produced by the same
converters a strategy would actually reach for. The distinction matters because
every refusal in `tree.rs` and `onnx.rs` was written against a *hand-edited*
JSON blob, and a hand-edited blob only proves the reader refuses the shape
somebody imagined. It proves nothing about the bytes xgboost, skl2onnx and
onnxmltools really emit — and one of those refusals had already stopped firing:

    xgboost 3.3.0 writes ``gradient_booster.name == "gbtree"`` for a **dart**
    booster. The dropout weights ride in a sibling ``weight_drop`` array. The
    name check that was supposed to catch dart therefore waves it straight
    through, and the reader sums the trees unweighted and returns a plausible
    number that is wrong by the dropout factor.

So these fixtures are committed, and `tests/model_family_refusals.rs` asserts a
*named* refusal against each. Like the other two generators here, this one is
deliberately not run by CI: an artifact regenerated in the same breath as the
assertion would agree with whatever the reader had just started doing.

One fixture here is **not** refused, and it is load-bearing:
`linear_regression.onnx` is a real `skl2onnx` export that loads and scores. A
corpus of nothing but refusals is consistent with a loader that refuses
everything a converter emits, and "sklearn cannot cross into Rust" is a much
bigger claim than the true one — which is that `tract` has no
`TreeEnsembleRegressor` and this crate has no way to guess a score column.

Run from the repo root with the project venv:

    PYTHONHASHSEED=0 .venv/bin/python \\
        crates/axon-model/tests/refusals/generate.py

Everything is kept tiny on purpose — a three-tree stump ensemble proves a
refusal exactly as well as a five-hundred-tree one, and these files live in git.
"""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np
import onnx
import pandas as pd
import xgboost as xgb
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).resolve().parent

# The loaders refuse an unversioned artifact before they get anywhere near the
# construct under test (ADR-0019 §4). Every fixture here is stamped so the
# refusal a test observes is the one it is named after, not a missing version.
MODEL_VERSION = 7

# onnxmltools' LightGBM converter caps out below skl2onnx's default, and a
# mismatched opset is a conversion error rather than a serving question. 15 is
# the highest all three converters here agree on.
TARGET_OPSET = 15

N_ROWS = 200
N_FEATURES = 3
WEIGHTS = np.array([1.0, -2.0, 0.5], dtype=np.float32)


def f32_bits(values) -> list[str]:
    """IEEE-754 bit patterns of an f32 array, as 0x-prefixed hex."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [f"0x{struct.unpack('<I', struct.pack('<f', float(v)))[0]:08x}" for v in arr]


def training_set(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    return x, (x @ WEIGHTS).astype(np.float32)


def holdout() -> np.ndarray:
    return np.random.default_rng(4242).normal(size=(8, N_FEATURES)).astype(np.float32)


def save(name: str, booster: xgb.Booster) -> dict:
    booster.set_attr(axon_model_version=str(MODEL_VERSION))
    booster.save_model(str(HERE / name))
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def unweighted_leaf_sum(artifact: dict, rows: np.ndarray) -> np.ndarray:
    """What a reader that ignores ``weight_drop`` would return.

    This is not a reimplementation of the tree backend for its own sake — it is
    the *wrong* answer, recorded so the Rust test can assert it differs from
    XGBoost's. A refusal test whose two numbers happen to coincide is a test
    that would still pass with the refusal removed.
    """
    learner = artifact["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base = np.float32(learner["learner_model_param"]["base_score"].strip("[]"))

    def leaf(tree: dict, row: np.ndarray) -> np.float32:
        i = 0
        while tree["left_children"][i] >= 0:
            value = row[tree["split_indices"][i]]
            threshold = np.float32(tree["split_conditions"][i])
            i = tree["left_children"][i] if value < threshold else tree["right_children"][i]
        return np.float32(tree["split_conditions"][i])

    out = []
    for row in rows:
        margin = base
        for tree in trees:
            margin = np.float32(margin + leaf(tree, row))
        out.append(margin)
    return np.asarray(out, dtype=np.float32)


def export_dart() -> dict:
    """A dart booster: dropout-weighted trees, under xgboost 3.3.0's spelling.

    `one_drop` guarantees at least one tree is dropped per round, which is what
    drives `weight_drop` away from 1.0 — a dart ensemble whose weights all
    happen to be 1.0 serves correctly by coincidence and would make this fixture
    vacuous. Note the fixture is fitted through `booster="dart"`, but 3.3.0
    deprecates that spelling in favour of passing `rate_drop` to a plain
    `gbtree`: both produce a byte-identical artifact, so a dart ensemble is now
    reachable without anyone typing the word.
    """
    x, y = training_set(0)
    est = xgb.XGBRegressor(
        n_estimators=4,
        max_depth=2,
        booster="dart",
        rate_drop=0.5,
        one_drop=True,
        skip_drop=0.0,
        random_state=0,
    )
    est.fit(x, y)
    booster = est.get_booster()
    artifact = save("dart.json", booster)

    rows = holdout()
    truth = booster.predict(xgb.DMatrix(rows, missing=np.nan), output_margin=True)
    naive = unweighted_leaf_sum(artifact, rows)
    assert not np.array_equal(truth, naive), "weight_drop is all 1.0; refit with one_drop"
    return {
        "artifact": "dart.json",
        "weight_drop": artifact["learner"]["gradient_booster"]["weight_drop"],
        "booster_name_in_json": artifact["learner"]["gradient_booster"]["name"],
        "inputs": [[float(v) for v in row] for row in rows],
        # What `Booster.predict` actually returns, and what a reader that summed
        # the trees unweighted would return instead. The gap between them is the
        # size of the silent wrong answer.
        "xgboost_margin_bits": f32_bits(truth),
        "unweighted_sum_bits": f32_bits(naive),
    }


def export_categorical() -> str:
    """Partition-based categorical splits: a bitset test, not a threshold.

    `max_cat_to_onehot=1` forces the partition form. XGBoost still writes a
    `split_conditions` entry for those nodes — a denormal (1e-45) holding the
    category-segment offset — so a reader that skipped `split_type` would
    compare a feature against 1e-45 and route almost every row right.
    """
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=N_ROWS).astype(np.float32),
            "cat": pd.Categorical(rng.integers(0, 4, size=N_ROWS)),
        }
    )
    y = (frame["num"].to_numpy() + frame["cat"].cat.codes.to_numpy() * 1.5).astype(np.float32)
    est = xgb.XGBRegressor(
        n_estimators=3,
        max_depth=3,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        random_state=0,
    )
    est.fit(frame, y)
    artifact = save("categorical.json", est.get_booster())
    assert any(
        1 in tree["split_type"]
        for tree in artifact["learner"]["gradient_booster"]["model"]["trees"]
    ), "no categorical split survived training; the fixture proves nothing"
    return "categorical.json"


def export_multiclass() -> str:
    """`multi:softmax` over three classes: one tree per class per round."""
    rng = np.random.default_rng(6)
    x = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 3, size=N_ROWS)
    est = xgb.XGBClassifier(
        n_estimators=2, max_depth=2, objective="multi:softmax", num_class=3, random_state=0
    )
    est.fit(x, y)
    save("multiclass.json", est.get_booster())
    return "multiclass.json"


def export_multi_output(strategy: str, name: str) -> str:
    """Two regression targets, under both of XGBoost's multi-output strategies.

    They fail differently and both have to be covered: `multi_output_tree` puts
    a *vector* in every leaf (`size_leaf_vector=2`), while `one_output_per_tree`
    keeps scalar leaves and fans the ensemble out across output groups via
    `tree_info`. A reader that only knew about one of them would serve the other
    as a single margin built from half the trees.
    """
    x, y = training_set(0)
    targets = np.stack([y, -0.5 * y], axis=1)
    est = xgb.XGBRegressor(
        n_estimators=3,
        max_depth=2,
        multi_strategy=strategy,
        tree_method="hist",
        random_state=0,
    )
    est.fit(x, targets)
    save(name, est.get_booster())
    return name


def export_onnx_families() -> None:
    """The graphs the model zoo's converters actually emit.

    Nothing in this tree had ever fed `tract` a graph from `skl2onnx` or
    `onnxmltools`. These four are the shapes those converters produce by
    default, committed so the answer is a test rather than a memory.
    """
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import to_onnx
    import lightgbm as lgb

    x, y = training_set(1)
    labels = (y > 0).astype(np.int64)

    def stamp(model, name: str) -> None:
        model.model_version = MODEL_VERSION
        onnx.save(model, HERE / name)

    gbm = GradientBoostingRegressor(n_estimators=4, max_depth=2, random_state=0).fit(x, y)
    stamp(to_onnx(gbm, x[:1], target_opset=TARGET_OPSET), "sklearn_gbm.onnx")

    booster = lgb.LGBMRegressor(n_estimators=4, max_depth=2, verbose=-1).fit(x, y)
    lightgbm_graph = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, N_FEATURES]))],
        target_opset=TARGET_OPSET,
    )
    # onnxmltools names the graph with a fresh uuid4 on every conversion, so a
    # re-run of this script would otherwise rewrite the committed bytes with a
    # diff that means nothing — and a diff that always appears is a diff nobody
    # reads. Pinning the name keeps a regeneration reviewable.
    lightgbm_graph.graph.name = "lightgbm_regressor"
    stamp(lightgbm_graph, "lightgbm.onnx")

    # skl2onnx' default classifier export: an int64 `label` tensor *and* a
    # probability tensor. ADR-0019 §1 refuses it for the ambiguity, and this is
    # the artifact that ambiguity is about.
    logistic = LogisticRegression(max_iter=200).fit(x, labels)
    two_outputs = to_onnx(
        logistic,
        x[:1],
        target_opset=TARGET_OPSET,
        options={id(logistic): {"zipmap": False}},
    )
    stamp(two_outputs, "logistic_two_outputs.onnx")

    # The same graph with the label output removed — the obvious "fix" for the
    # refusal above, and still wrong: one tensor, two *values* per row, and
    # column 0 is P(class 0). A caller that took the first column would trade
    # the model backwards.
    probabilities = to_onnx(
        logistic,
        x[:1],
        target_opset=TARGET_OPSET,
        options={id(logistic): {"zipmap": False}},
    )
    del probabilities.graph.output[0]
    stamp(probabilities, "logistic_probabilities.onnx")

    assert len(two_outputs.graph.output) == 2, "skl2onnx stopped emitting a label tensor"
    assert len(probabilities.graph.output) == 1, "the label output was not removed"

    # sklearn's GradientBoosting *classifier*, narrowed to one score column the
    # way P3 narrowed LightGBM's. It clears every rule this crate imposes — one
    # input, one tensor, one FP32 value per row — and still does not build,
    # because `tract`'s TreeEnsembleClassifier wants `base_values` to carry one
    # entry per class and skl2onnx writes sklearn's single scalar intercept.
    # Committed so the ceiling is attributed to the attribute rather than to
    # "sklearn does not work".
    classifier = GradientBoostingClassifier(n_estimators=4, max_depth=2, random_state=0)
    classifier.fit(x, labels)
    narrowed = to_onnx(
        classifier,
        x[:1],
        target_opset=TARGET_OPSET,
        options={id(classifier): {"zipmap": False}},
    )
    narrowed.graph.initializer.append(
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="positive_class")
    )
    narrowed.graph.node.append(
        helper.make_node("Gather", ["probabilities", "positive_class"], ["score"], axis=1)
    )
    del narrowed.graph.output[:]
    narrowed.graph.output.append(
        helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])
    )
    onnx.checker.check_model(narrowed)
    stamp(narrowed, "sklearn_gbm_classifier.onnx")


def export_linear_control() -> dict:
    """The control: a converted sklearn model that *does* cross into Rust.

    `LinearRegression` converts to a single `ai.onnx.ml.LinearRegressor` node
    with one output holding one value, which is the whole contract. Its answers
    are recorded from onnxruntime with fusion disabled, so the Rust test can
    hold `tract` to the ADR-0003 tolerance over a graph nobody hand-built —
    the same bar a parity bundle would set, minus the bundle.
    """
    import onnxruntime as ort
    from sklearn.linear_model import LinearRegression
    from skl2onnx import to_onnx

    x, y = training_set(2)
    model = to_onnx(LinearRegression().fit(x, y), x[:1], target_opset=TARGET_OPSET)
    model.model_version = MODEL_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, HERE / "linear_regression.onnx")
    assert [n.op_type for n in model.graph.node] == ["LinearRegressor"]

    # opt_level=0: the reference is the graph as exported, not what a runtime's
    # fusion pass turns it into (ADR-0005). And one row at a time, because the
    # Rust plan pins its batch dimension to 1 (ADR-0019 §1) — a reference taken
    # over a whole matrix would be measuring the batch shape.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(HERE / "linear_regression.onnx"), options, providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    rows = holdout()
    scores = [
        float(session.run(None, {name: row.reshape(1, N_FEATURES)})[0].reshape(-1)[0])
        for row in rows
    ]
    return {
        "artifact": "linear_regression.onnx",
        "inputs": [[float(v) for v in row] for row in rows],
        "onnxruntime_scores": scores,
    }


def export_three_wide_graph() -> str:
    """A hand-built FP32 graph with one output tensor holding three values.

    `logistic_probabilities.onnx` is the realistic version of this; this one
    isolates the property with no converter in the way, so a change in
    skl2onnx cannot quietly stop covering it.
    """
    w = numpy_helper.from_array(
        np.random.default_rng(9).normal(size=(N_FEATURES, 3)).astype(np.float32), name="w"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "three_values_per_row",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, N_FEATURES])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", TARGET_OPSET)])
    model.model_version = MODEL_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, HERE / "three_values_per_row.onnx")
    return "three_values_per_row.onnx"


def main() -> None:
    reference = {
        "note": "Generated by generate.py; see its docstring. Do not hand-edit.",
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "xgboost": xgb.__version__,
        },
        "dart": export_dart(),
        "linear_control": export_linear_control(),
        "refused_trees": [
            export_categorical(),
            export_multiclass(),
            export_multi_output("multi_output_tree", "multi_output_tree.json"),
            export_multi_output("one_output_per_tree", "multi_output_one_per_tree.json"),
        ],
    }
    export_onnx_families()
    export_three_wide_graph()

    with (HERE / "reference.json").open("w", encoding="utf-8") as fh:
        json.dump(reference, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")


if __name__ == "__main__":
    main()
