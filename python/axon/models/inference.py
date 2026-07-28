"""Load an artifact back into something that predicts.

This is the *verification* side of the export (and the offline half of the parity
gate): whatever runs here is what a Rust backend will have to reproduce, so the
settings are chosen for reproducibility rather than throughput.

Three of them are load-bearing, all from ADR-0005:

- **Graph optimization off** (``opt_level=0``). Fusion rewrites the arithmetic —
  the fused op is *better*, and different — so a graph that has been optimized is
  no longer structurally the graph the parity gate compared.
- **One thread.** Parallel reductions sum in whatever order the threads finish,
  and FP addition is not associative, so a multi-threaded session can disagree
  with itself between runs. A backtest that cannot reproduce its own numbers
  cannot be diffed against a live session.
- **CPU execution provider, pinned.** The documented silent-FP16-downcast footgun
  lives in the accelerated providers (CoreML, and GPU paths generally). Pinning
  the provider is what makes "we serve FP32" a fact rather than a preference.

Inference here is *not* on the hot path — it runs at export time and in the
research plane. The live signal path is Python-side too under Boundary B, but the
microsecond budget lives in Rust (``docs/05``), so clarity wins over speed.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from axon.models.artifact import Artifact, TensorSpec

#: ONNX Runtime type strings → the dtype names recorded in artifact metadata.
_ORT_DTYPES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(bool)": "bool",
    "tensor(string)": "string",
}

Schema = tuple[tuple[TensorSpec, ...], tuple[TensorSpec, ...], str]


class Predictor(Protocol):
    """What every artifact format reduces to once loaded."""

    def predict(self, x: Any) -> np.ndarray:
        """Scores for a 2-D float32 matrix of feature rows."""

    def declared_schema(self) -> Schema | None:
        """The artifact's own I/O signature, or ``None`` if the format has none."""


def load_predictor(artifact: Artifact) -> Predictor:
    """Load ``artifact`` for inference, after checking its bytes against its metadata.

    There is deliberately no ``verify=False``: a loader that can be asked to skip
    the check is a loader that will be asked to skip the check, on the day someone
    is debugging in a hurry.
    """
    artifact.verify()
    kind = artifact.meta.kind
    if kind == "xgboost":
        return XgboostPredictor(artifact.payload)
    if kind == "lightgbm":
        return LightgbmPredictor(artifact.payload)
    if kind == "onnx":
        return OnnxPredictor(artifact.payload)
    raise ValueError(f"no inference backend for artifact kind {kind!r}")


def session_options() -> Any:
    """ONNX Runtime options: optimization off, single-threaded, sequential."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


class XgboostPredictor:
    """An XGBoost booster loaded from its own JSON."""

    def __init__(self, payload: bytes) -> None:
        import xgboost

        booster = xgboost.Booster()
        # bytearray, not a temp file: an artifact that never lands on disk cannot
        # be served half-written or left behind.
        booster.load_model(bytearray(payload))
        # Pinned to one thread for the same reason `session_options` pins the ONNX
        # session, and this was the one backend that did not. A booster reloaded from
        # an artifact defaults to every core, and serving is one row at a time: the
        # OpenMP fan-out then dwarfs the tree traversal it is parallelizing — 5.4 ms
        # per single-row prediction against 0.22 ms pinned, measured on an 8-core box.
        # That is a 25x latency tax paid on the decision path, and it grows with the
        # machine, so it is worst exactly where the system is deployed. Verified
        # bit-identical either way (max_abs_diff 0.0 over 5,000 rows), so nothing about
        # the answer changes — only how long it takes to arrive.
        booster.set_param({"nthread": 1})
        self._booster = booster

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self._booster.inplace_predict(_matrix(x)))

    def declared_schema(self) -> Schema | None:
        return None


class LightgbmPredictor:
    """A LightGBM booster loaded from its native text model."""

    def __init__(self, payload: bytes) -> None:
        import lightgbm

        self._booster = lightgbm.Booster(model_str=payload.decode("utf-8"))

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self._booster.predict(_matrix(x)))

    def declared_schema(self) -> Schema | None:
        return None


class OnnxPredictor:
    """An ONNX graph in a deterministic, FP32-pinned onnxruntime session."""

    def __init__(self, payload: bytes) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            payload, sess_options=session_options(), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            # The serving path hands over one feature matrix. A multi-input graph
            # needs a feeding convention that does not exist yet, and inventing one
            # here would bind the argument order silently.
            raise ValueError(
                f"artifact declares {len(inputs)} inputs "
                f"({[i.name for i in inputs]}); serving feeds exactly one feature matrix"
            )
        self._input = inputs[0].name
        self._score = _score_output(self._session)

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self._session.run([self._score], {self._input: _matrix(x)})[0])

    def run(self, x: Any) -> dict[str, np.ndarray]:
        """Every output by name — the label tensor as well as the probabilities."""
        names = [o.name for o in self._session.get_outputs()]
        return dict(zip(names, self._session.run(names, {self._input: _matrix(x)})))

    def declared_schema(self) -> Schema | None:
        inputs = tuple(_spec(v) for v in self._session.get_inputs())
        outputs = tuple(_spec(v) for v in self._session.get_outputs())
        return inputs, outputs, self._score


def _score_output(session: Any) -> str:
    """Which output carries the number a strategy acts on.

    skl2onnx gives a classifier two outputs, ``label`` and ``probabilities``, in
    that order. Taking the first would hand the strategy a class index and call it
    a score.
    """
    outputs = session.get_outputs()
    by_name = {o.name: o for o in outputs}
    if "probabilities" in by_name:
        return "probabilities"
    for out in outputs:
        if out.type == "tensor(float)":
            return out.name
    return outputs[0].name


def _spec(value: Any) -> TensorSpec:
    # ORT reports a symbolic dimension as a string ("batch"); the schema records
    # every non-fixed dimension the same way, as None.
    shape = tuple(d if isinstance(d, int) else None for d in value.shape)
    return TensorSpec(value.name, _ORT_DTYPES.get(value.type, value.type), shape)


def _matrix(x: Any) -> np.ndarray:
    """Feature rows as a C-contiguous float32 matrix — what every backend expects."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {a.shape}")
    return a


__all__ = [
    "LightgbmPredictor",
    "OnnxPredictor",
    "Predictor",
    "XgboostPredictor",
    "load_predictor",
    "session_options",
]
