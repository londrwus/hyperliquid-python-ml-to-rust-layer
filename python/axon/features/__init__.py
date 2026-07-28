"""``axon.features`` — THE feature library (single source of truth).

The prime directive (``docs/03-ml-fidelity-and-features.md``, ADR-0003, ADR-0016):
**never implement a feature twice.** Under Boundary B the *same* code here runs at
research time and at serving time, so online == offline by construction. If a
feature ever moves to Rust (Boundary A), the Rust version must pass a
bit-equivalence gate against this module before it may serve.

Three layers:

* :mod:`axon.features.functions` — the transforms, as plain functions over 1-D
  arrays. Length-preserving, causal, NaN during warmup; see that module for why
  each of those three is non-negotiable.
* :mod:`axon.features.spec` — :class:`FeatureSpec`, the ordered, named,
  hash-identified recipe that both produces a feature matrix and serializes into a
  model artifact, so the parity harness knows exactly what a model was fed.
* :mod:`axon.features.inputs` — the adapters from a wire record to the named arrays
  a spec consumes: the market-data ring's ``MdSlice`` and a closed OHLCV bar. The
  only fixed-point→float conversion here.

Typical use::

    from axon.features import PERP_CORE_V1, md_slice_inputs, finite_rows

    inputs, ts_event = md_slice_inputs(md.read_batch())
    matrix = PERP_CORE_V1.compute(inputs)          # (n_rows, n_features), NaN warmup
    usable = finite_rows(matrix)
    artifact_ref = PERP_CORE_V1.ref                # "perp_core/v1#<fingerprint>"
"""

from __future__ import annotations

from axon.features.functions import (
    FEATURES_VERSION,
    book_imbalance,
    close_location,
    ema,
    ema_crossover,
    finite_rows,
    log_return,
    mid_price,
    momentum,
    realized_volatility,
    relative_range,
    relative_spread,
    rolling_mean,
    rolling_std,
    rolling_sum,
    rolling_zscore,
    sma_crossover,
    spread,
    trade_flow_imbalance,
)
from axon.features.inputs import BAR_INPUTS, bar_inputs, md_slice_inputs
from axon.features.registry import (
    FeatureError,
    FeatureInfo,
    UnknownFeature,
    feature_info,
    register,
    registered_features,
)
from axon.features.spec import (
    BAR_M1_V1,
    BAR_M1_WARMUP_BARS,
    PERP_CORE_V1,
    FeatureDef,
    FeatureSpec,
    FeatureSpecMismatch,
    spec_from_defs,
)

__all__ = [
    "BAR_INPUTS",
    "BAR_M1_V1",
    "BAR_M1_WARMUP_BARS",
    "FEATURES_VERSION",
    "PERP_CORE_V1",
    "FeatureDef",
    "FeatureError",
    "FeatureInfo",
    "FeatureSpec",
    "FeatureSpecMismatch",
    "UnknownFeature",
    "bar_inputs",
    "book_imbalance",
    "close_location",
    "ema",
    "ema_crossover",
    "feature_info",
    "finite_rows",
    "log_return",
    "md_slice_inputs",
    "mid_price",
    "momentum",
    "realized_volatility",
    "register",
    "registered_features",
    "relative_range",
    "relative_spread",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_zscore",
    "sma_crossover",
    "spec_from_defs",
    "spread",
    "trade_flow_imbalance",
]
