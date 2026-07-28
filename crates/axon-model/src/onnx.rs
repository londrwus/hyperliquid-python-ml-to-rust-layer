//! [`OnnxModel`] — ONNX graphs served by `tract`.
//!
//! `tract` rather than `ort` (ADR-0003 §2 allows either) because this path is
//! chosen for *determinism*, not throughput. `tract` is pure Rust: there is no
//! `libonnxruntime` to ship, to keep version-matched with research, or to
//! accidentally pick up a different build of on the trading host. Its default
//! executor is single-threaded, so a matmul's reduction order is fixed and the
//! same input yields the same bits on every run — a thread-pooled runtime gives
//! no such guarantee. Inference is not the bottleneck anyway: the venue
//! round-trip is five orders of magnitude larger (docs/05).
//!
//! Two loading rules follow from ADR-0005 and are enforced before anything is
//! executed:
//!
//! - **The graph is inspected in protobuf form first.** A model whose *boundary*
//!   is FP32 can still round every intermediate through FP16 — a `Cast` in the
//!   middle is invisible from the signature, and it is the documented ONNX
//!   Runtime downcast footgun. The scan walks initializers, value info, node
//!   attributes and subgraphs, and refuses on any reduced-precision float or
//!   any quantization operator.
//! - **The graph is not optimized.** `into_typed()` resolves shapes and stops;
//!   `into_optimized()` would fuse and reassociate operators, which is exactly
//!   the `opt_level=0` that ADR-0005 requires when structural numerical match to
//!   the Python model matters.
//!
//! Neural nets are never bit-exact across runtimes (ONNX does not encode op
//! ordering and float addition is not associative), so `tests/parity.rs` holds
//! this backend to the ADR-0003 tolerance of 1e-5 rather than to bit equality.

use std::path::Path;
use std::sync::Arc;

use tract_onnx::pb::{tensor_proto::DataType, type_proto, AttributeProto, GraphProto, ModelProto};
use tract_onnx::prelude::*;

use crate::{InferenceError, LoadError, Model};

/// Operators that only exist to carry quantized tensors. Their presence means
/// the artifact was quantized, which ADR-0005 forbids outright.
const QUANTIZATION_OPS: &[&str] = &[
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "MatMulInteger",
    "ConvInteger",
];

const FLOAT16: i32 = DataType::Float16 as i32;
const BFLOAT16: i32 = DataType::Bfloat16 as i32;
const FLOAT8E4M3FN: i32 = DataType::Float8e4m3fn as i32;
const FLOAT8E4M3FNUZ: i32 = DataType::Float8e4m3fnuz as i32;
const FLOAT8E5M2: i32 = DataType::Float8e5m2 as i32;
const FLOAT8E5M2FNUZ: i32 = DataType::Float8e5m2fnuz as i32;
const FLOAT4E2M1: i32 = DataType::Float4e2m1 as i32;
const FLOAT: i32 = DataType::Float as i32;

/// An FP32 ONNX graph, loaded and planned once at startup.
///
/// The artifact must declare exactly one input of shape `[batch, features]` and
/// produce exactly one FP32 **value** per row — one output tensor holding one
/// number, not merely one tensor. Both restrictions are deliberate: which
/// number feeds the strategy must be a property of the artifact, not a
/// convention the loader and the exporter each remember separately. An
/// `skl2onnx` classifier fails the first rule (a label tensor beside the
/// probabilities) and, once the label is stripped, the second (two class
/// probabilities in one tensor). Both must be re-exported with a single score.
#[derive(Clone)]
pub struct OnnxModel {
    version: u32,
    input_len: usize,
    plan: Arc<TypedRunnableModel>,
}

impl std::fmt::Debug for OnnxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The plan holds the whole graph; printing it would dump every weight
        // into a log line.
        f.debug_struct("OnnxModel")
            .field("version", &self.version)
            .field("input_len", &self.input_len)
            .finish_non_exhaustive()
    }
}

impl OnnxModel {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| LoadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let onnx = tract_onnx::onnx();
        let proto = onnx
            .proto_model_for_read(&mut std::io::Cursor::new(bytes))
            .map_err(|e| LoadError::Parse(tract_cause(&e)))?;

        // Version first: an artifact we cannot name has no business being
        // planned, let alone served. ONNX has a first-class `model_version`
        // field, so unlike the XGBoost path there is no convention to invent.
        let version = u32::try_from(proto.model_version)
            .ok()
            .filter(|v| *v != 0)
            .ok_or(LoadError::MissingVersion {
                slot: "model_version",
            })?;

        reject_reduced_precision(&proto)?;

        let graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| LoadError::Malformed("ONNX model has no graph".to_string()))?;
        let input_len = single_fp32_input_width(graph)?;
        let output = only(&graph.output, "output")?;
        if elem_type(output) != Some(FLOAT) {
            return Err(LoadError::Unsupported(
                "graph output is not FP32; the strategy consumes f32 scores".to_string(),
            ));
        }

        let typed = onnx
            .model_for_proto_model(&proto)
            .map_err(|e| LoadError::Parse(tract_cause(&e)))?
            // Pinning the batch dimension to 1 turns every symbolic shape in the
            // graph concrete, so the plan is fully static and no shape is
            // resolved per tick. Inference here is one decision at a time; a
            // batched research path is a different artifact.
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, input_len)),
            )
            .and_then(|m| m.into_typed())
            .map_err(|e| LoadError::Parse(tract_cause(&e)))?;

        let fact = typed
            .output_fact(0)
            .map_err(|e| LoadError::Parse(tract_cause(&e)))?;
        if fact.datum_type != f32::datum_type() {
            return Err(LoadError::Unsupported(format!(
                "graph output resolved to {:?}, not f32",
                fact.datum_type
            )));
        }
        let values_per_row = fact
            .shape
            .as_concrete()
            .map(|dims| dims.iter().product::<usize>())
            .ok_or_else(|| {
                LoadError::Unsupported(
                    "graph output shape is not concrete once the batch is fixed to 1".to_string(),
                )
            })?;
        if values_per_row != 1 {
            // ADR-0019 §1 asks for exactly one FP32 output and this loader used
            // to read that as one output *tensor* — which a two-class
            // probability vector satisfies. `skl2onnx` with `zipmap=False`
            // emits exactly that, and column 0 is P(class 0): a caller that
            // took the first value would trade the model backwards, with every
            // number in [0, 1] and nothing to look wrong. Which column is the
            // score has to be a property of the artifact (ADR-0015's
            // `score_output`), never a guess made here.
            return Err(LoadError::Unsupported(format!(
                "graph produces {values_per_row} values per row; a trading decision is a \
                 threshold on one number and this loader cannot know which column is the \
                 score. Re-export with a single score output"
            )));
        }

        let plan = typed
            .into_runnable()
            .map_err(|e| LoadError::Parse(tract_cause(&e)))?;

        Ok(Self {
            version,
            input_len,
            plan,
        })
    }
}

impl Model for OnnxModel {
    fn version(&self) -> u32 {
        self.version
    }

    fn input_len(&self) -> usize {
        self.input_len
    }

    /// Always 1: a graph that resolved to more values per row was refused at
    /// load rather than served with a column picked for it.
    fn output_len(&self) -> usize {
        1
    }

    /// Unlike [`TreeModel`](crate::TreeModel), this cannot be allocation-free:
    /// `tract` allocates its intermediate tensors per run. That is one more
    /// reason the tree family is the first to earn a Boundary-A promotion.
    fn predict_into(&self, features: &[f32], out: &mut [f32]) -> Result<(), InferenceError> {
        crate::check_shapes(features, out, self.input_len, 1)?;
        let input = Tensor::from_shape(&[1, self.input_len], features)
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
        let outputs = self
            .plan
            .run(tvec!(input.into()))
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
        let produced = outputs
            .first()
            .ok_or_else(|| InferenceError::Backend("graph produced no output".to_string()))?;
        let slice = produced
            .view()
            .as_slice::<f32>()
            .map_err(|e| InferenceError::Backend(e.to_string()))?;
        if slice.len() != out.len() {
            // The planned output fact said otherwise; a graph whose runtime
            // shape disagrees with its static shape must not quietly truncate.
            return Err(InferenceError::OutputLen {
                expected: slice.len(),
                got: out.len(),
            });
        }
        out.copy_from_slice(slice);
        Ok(())
    }
}

/// A tract failure, *with* its cause chain.
///
/// `TractError` is an `anyhow::Error`, whose plain `Display` prints only the
/// outermost context. For a graph tract cannot build, that outermost line is
/// `Building node TreeEnsembleClassifier (TreeEnsembleClassifier)` — which
/// names the operator and withholds the only part an operator can act on. The
/// cause underneath it reads `attribute 'base_values': expected length 1 (or
/// undefined), got 2`, i.e. re-export the model with a conformant attribute.
/// The alternate format walks the chain. A load error is read once, by someone
/// deciding whether to change the export script (ADR-0021 §8).
fn tract_cause(error: &TractError) -> String {
    format!("{error:#}")
}

/// The declared width of the single FP32 feature input.
fn single_fp32_input_width(graph: &GraphProto) -> Result<usize, LoadError> {
    // Pre-IR-4 exporters list initializers among the graph inputs, so weights
    // would otherwise be counted as feature inputs.
    let inputs: Vec<_> = graph
        .input
        .iter()
        .filter(|vi| !graph.initializer.iter().any(|t| t.name == vi.name))
        .collect();
    let input = match inputs.as_slice() {
        [one] => *one,
        other => {
            return Err(LoadError::Unsupported(format!(
                "graph declares {} feature inputs; exactly one is required",
                other.len()
            )))
        }
    };
    if elem_type(input) != Some(FLOAT) {
        return Err(LoadError::Unsupported(
            "graph input is not FP32; features cross the boundary as f32".to_string(),
        ));
    }
    let dims = tensor_shape(input)
        .ok_or_else(|| LoadError::Unsupported("graph input has no declared shape".to_string()))?;
    // `[batch, features]`. The batch dim may be symbolic — it is pinned to 1 at
    // load — but the feature count must be concrete, because it is what the
    // feature spec is checked against.
    match dims.as_slice() {
        [_, Some(features)] if *features > 0 => Ok(*features as usize),
        _ => Err(LoadError::Unsupported(format!(
            "graph input shape {dims:?} is not [batch, features] with a concrete feature count"
        ))),
    }
}

fn only<'a, T>(items: &'a [T], what: &str) -> Result<&'a T, LoadError> {
    match items {
        [one] => Ok(one),
        other => Err(LoadError::Unsupported(format!(
            "graph declares {} {what}s; exactly one is required",
            other.len()
        ))),
    }
}

fn tensor_type(vi: &tract_onnx::pb::ValueInfoProto) -> Option<&type_proto::Tensor> {
    let type_proto::Value::TensorType(t) = vi.r#type.as_ref()?.value.as_ref()?;
    Some(t)
}

fn elem_type(vi: &tract_onnx::pb::ValueInfoProto) -> Option<i32> {
    Some(tensor_type(vi)?.elem_type)
}

/// Declared dims, `None` for a symbolic one.
fn tensor_shape(vi: &tract_onnx::pb::ValueInfoProto) -> Option<Vec<Option<i64>>> {
    Some(
        tensor_type(vi)?
            .shape
            .as_ref()?
            .dim
            .iter()
            .map(|d| match d.value {
                Some(tract_onnx::pb::tensor_shape_proto::dimension::Value::DimValue(v)) => Some(v),
                _ => None,
            })
            .collect(),
    )
}

/// Name of a reduced-precision float type, or `None` if the type is acceptable.
fn reduced_precision(dtype: i32) -> Option<&'static str> {
    match dtype {
        FLOAT16 => Some("float16"),
        BFLOAT16 => Some("bfloat16"),
        FLOAT8E4M3FN => Some("float8e4m3fn"),
        FLOAT8E4M3FNUZ => Some("float8e4m3fnuz"),
        FLOAT8E5M2 => Some("float8e5m2"),
        FLOAT8E5M2FNUZ => Some("float8e5m2fnuz"),
        FLOAT4E2M1 => Some("float4e2m1"),
        _ => None,
    }
}

/// Refuse the artifact if anything anywhere in it is below FP32 (ADR-0005).
fn reject_reduced_precision(proto: &ModelProto) -> Result<(), LoadError> {
    let Some(graph) = proto.graph.as_ref() else {
        return Ok(());
    };
    scan_graph(graph, "graph")
}

fn scan_graph(graph: &GraphProto, path: &str) -> Result<(), LoadError> {
    let refuse = |what: String, dtype: &'static str| LoadError::ReducedPrecision { what, dtype };

    for t in graph.initializer.iter() {
        if let Some(dtype) = reduced_precision(t.data_type) {
            return Err(refuse(format!("{path} initializer '{}'", t.name), dtype));
        }
    }
    for vi in graph
        .input
        .iter()
        .chain(graph.output.iter())
        .chain(graph.value_info.iter())
    {
        if let Some(dtype) = elem_type(vi).and_then(reduced_precision) {
            return Err(refuse(format!("{path} tensor '{}'", vi.name), dtype));
        }
    }
    for node in graph.node.iter() {
        if QUANTIZATION_OPS.contains(&node.op_type.as_str()) {
            return Err(LoadError::Unsupported(format!(
                "{path} node '{}' is a {} operator; quantized artifacts are refused (ADR-0005)",
                node.name, node.op_type
            )));
        }
        for attr in node.attribute.iter() {
            scan_attribute(attr, &format!("{path} node '{}'", node.name))?;
        }
    }
    Ok(())
}

fn scan_attribute(attr: &AttributeProto, path: &str) -> Result<(), LoadError> {
    // `Cast`'s target type lives in an int attribute, which is how a graph with
    // an all-FP32 signature ends up computing in half precision internally.
    if matches!(attr.name.as_str(), "to" | "dtype" | "output_dtype") {
        if let Some(dtype) = reduced_precision(attr.i as i32) {
            return Err(LoadError::ReducedPrecision {
                what: format!("{path} attribute '{}'", attr.name),
                dtype,
            });
        }
    }
    for t in attr.t.iter().chain(attr.tensors.iter()) {
        if let Some(dtype) = reduced_precision(t.data_type) {
            return Err(LoadError::ReducedPrecision {
                what: format!("{path} attribute '{}' tensor", attr.name),
                dtype,
            });
        }
    }
    // If/Loop/Scan bodies are graphs of their own; a downcast hidden in a branch
    // is still a downcast.
    for g in attr.g.iter().chain(attr.graphs.iter()) {
        scan_graph(g, &format!("{path} subgraph '{}'", attr.name))?;
    }
    Ok(())
}
