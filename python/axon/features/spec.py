"""The versioned feature spec — what turns a pile of functions into a library.

A model is only reproducible if you know **exactly** what it was fed: which
transforms, with which parameters, in which column order, from which build of this
library. A :class:`FeatureSpec` is that record. It is:

* **ordered** — column order is part of the contract, because permuting two columns
  of a feature matrix leaves every name correct and every prediction wrong;
* **named** — a spec is bound to transforms by name, so an artifact read back in six
  months resolves against :mod:`axon.features.registry` rather than against whatever
  function object happened to be in scope at training time;
* **hash-identified** — :attr:`FeatureSpec.fingerprint` is a content hash of the whole
  recipe *including* :data:`~axon.features.functions.FEATURES_VERSION`, so a spec
  cannot be edited, or the library's arithmetic changed underneath it, without the
  identity moving;
* **serializable** — :meth:`FeatureSpec.to_dict` is what a model artifact stores as
  its ``feature_spec_ref`` payload, which is how the parity harness knows what the
  candidate model was trained on.

Specs also *compose*: a :class:`FeatureDef` may bind an input to an earlier column
in the same spec, so ``mid_price`` is computed once and the return, the volatility
and the z-score all read it. Recomputing the mid inside each of those would be three
implementations of one transform.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from axon.features import functions as _functions  # noqa: F401  (registration side effect)
from axon.features.functions import FEATURES_VERSION
from axon.features.registry import FeatureError, feature_info

#: Above 2^53 a float64 stops representing consecutive integers, so anything larger
#: arriving as a feature input has already lost its low bits. In practice this fires
#: on exactly one mistake — feeding nanosecond ``ts_event`` in as a feature — which
#: is why :func:`axon.features.inputs.md_slice_inputs` hands timestamps back
#: separately instead of putting them in the inputs mapping.
_EXACT_FLOAT_LIMIT = 2.0**53

_SPEC_KEYS = frozenset({"spec", "version", "library_version", "features", "fingerprint"})


class FeatureSpecMismatch(FeatureError):
    """A serialized spec does not describe what this library would compute.

    Either the payload was edited after its fingerprint was taken, or it was written
    by a different build of :mod:`axon.features`. Both mean the artifact's model was
    trained on features this process cannot reproduce, which is precisely the
    training–serving skew the spec exists to make impossible.
    """


def _canonical_param(feature: str, key: str, value: Any) -> Any:
    """Coerce a parameter to a JSON scalar, or refuse it.

    Parameters end up inside a content hash, so they have to serialize the same way
    on every machine and in every process. A ``np.int64`` window or a callable would
    hash differently (or not at all) and the spec's identity would stop being stable.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FeatureError(f"{feature}.{key} is {value!r}; a spec parameter must be finite")
        return value
    raise FeatureError(
        f"{feature}.{key} is a {type(value).__name__}; spec parameters must be "
        "bool/int/float/str so the spec hashes and serializes identically everywhere"
    )


@dataclass(frozen=True)
class FeatureDef:
    """One column of a feature matrix: a registered transform plus its bindings."""

    #: The column name in the produced matrix (and in every parity report).
    column: str
    #: A name from :func:`~axon.features.registry.registered_features`.
    feature: str
    #: Keyword-only tunables for the transform (window lengths, spans, …).
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Overrides for where each declared input comes from — either a key of the
    #: caller's inputs mapping or an **earlier** column of the same spec. Unbound
    #: inputs read the source of the same name.
    inputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise FeatureError(f"column must be a non-empty string, got {self.column!r}")
        info = feature_info(self.feature)  # raises UnknownFeature naming what exists

        unknown = sorted(set(self.params) - set(info.params))
        if unknown:
            # Silently ignoring a misspelled window is how a spec claims a 32-sample
            # volatility and delivers the default one, with nothing to show for it.
            raise FeatureError(
                f"{self.feature} does not take parameter(s) {unknown}; "
                f"it accepts {list(info.params)}"
            )
        params = {k: _canonical_param(self.feature, k, v) for k, v in sorted(self.params.items())}

        unbound = sorted(set(self.inputs) - set(info.inputs))
        if unbound:
            raise FeatureError(
                f"{self.feature} has no input(s) {unbound}; it reads {list(info.inputs)}"
            )
        for key, source in self.inputs.items():
            if not isinstance(source, str) or not source.strip():
                raise FeatureError(f"{self.feature}.{key} must bind to a name, got {source!r}")
        object.__setattr__(self, "params", MappingProxyType(params))
        object.__setattr__(self, "inputs", MappingProxyType(dict(sorted(self.inputs.items()))))

    @property
    def sources(self) -> tuple[str, ...]:
        """Where each positional input actually comes from, in call order."""
        info = feature_info(self.feature)
        return tuple(self.inputs.get(name, name) for name in info.inputs)

    def to_dict(self) -> dict[str, Any]:
        """The canonical serialized form.

        ``params`` and ``inputs`` are always present even when empty: an omitted key
        and an empty mapping would hash differently while meaning the same thing.
        """
        return {
            "column": self.column,
            "feature": self.feature,
            "params": dict(self.params),
            "inputs": dict(self.inputs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureDef":
        unknown = sorted(set(payload) - {"column", "feature", "params", "inputs"})
        if unknown:
            raise FeatureError(f"unknown feature-definition key(s) {unknown}")
        return cls(
            column=payload["column"],
            feature=payload["feature"],
            params=payload.get("params") or {},
            inputs=payload.get("inputs") or {},
        )


@dataclass(frozen=True)
class FeatureSpec:
    """An ordered, named, hash-identified list of ``(feature, params)``."""

    name: str
    version: int
    features: tuple[FeatureDef, ...]
    #: The build of :mod:`axon.features` this spec was written against. Defaults to
    #: the current one; a loaded artifact carries the one it was trained with.
    library_version: int = FEATURES_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise FeatureError(f"spec name must be a non-empty string, got {self.name!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise FeatureError(f"spec version must be an int >= 1, got {self.version!r}")
        features = tuple(self.features)
        if not features:
            raise FeatureError("a spec with no features would produce an empty matrix")
        if any(not isinstance(d, FeatureDef) for d in features):
            raise FeatureError("every entry of `features` must be a FeatureDef")
        columns = [d.column for d in features]
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        if duplicates:
            # Two columns with one name: the second silently wins as a binding source
            # and the matrix carries both, so the model and the spec disagree.
            raise FeatureError(f"duplicate column name(s) {duplicates}")
        object.__setattr__(self, "features", features)

    @property
    def columns(self) -> tuple[str, ...]:
        """The matrix column names, in matrix order."""
        return tuple(d.column for d in self.features)

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The input arrays a caller must supply, sorted.

        Anything a feature reads that is not produced by an earlier column in the
        same spec.
        """
        produced: set[str] = set()
        needed: set[str] = set()
        for d in self.features:
            needed.update(s for s in d.sources if s not in produced)
            produced.add(d.column)
        return tuple(sorted(needed))

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def fingerprint(self) -> str:
        """Content hash of the whole recipe — the id an artifact records."""
        return _fingerprint(self._body())

    @property
    def ref(self) -> str:
        """``name/vN#fingerprint`` — what goes in a model artifact's spec reference."""
        return f"{self.name}/v{self.version}#{self.fingerprint}"

    def _body(self) -> dict[str, Any]:
        return {
            "spec": self.name,
            "version": self.version,
            "library_version": self.library_version,
            "features": [d.to_dict() for d in self.features],
        }

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """The serialized spec, with its fingerprint attached."""
        body = self._body()
        body["fingerprint"] = _fingerprint(body)
        return body

    def to_json(self) -> str:
        """Canonical JSON — stable key order, no whitespace, byte-comparable."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, strict: bool = True) -> "FeatureSpec":
        """Rebuild a spec, verifying it still describes what this library computes.

        ``strict=False`` skips the two identity checks and exists for tooling that
        needs to *inspect* an incompatible artifact (to report what it wanted). It
        must never be used on the serving path: the whole point of the fingerprint
        is that a model trained on other features refuses to run.
        """
        unknown = sorted(set(payload) - _SPEC_KEYS)
        if unknown:
            raise FeatureError(f"unknown spec key(s) {unknown}")
        missing = sorted({"spec", "version", "features"} - set(payload))
        if missing:
            raise FeatureError(f"spec payload is missing key(s) {missing}")

        spec = cls(
            name=payload["spec"],
            version=payload["version"],
            features=tuple(FeatureDef.from_dict(d) for d in payload["features"]),
            library_version=int(payload.get("library_version", FEATURES_VERSION)),
        )
        if not strict:
            return spec

        recorded = payload.get("fingerprint")
        if recorded is not None and recorded != spec.fingerprint:
            raise FeatureSpecMismatch(
                f"spec fingerprint {recorded} does not match the recomputed "
                f"{spec.fingerprint}; the payload was altered after it was written"
            )
        if spec.library_version != FEATURES_VERSION:
            raise FeatureSpecMismatch(
                f"spec {spec.name!r} was written against axon.features v"
                f"{spec.library_version}, this build is v{FEATURES_VERSION}; the "
                "transforms changed meaning, so the model would be fed different numbers"
            )
        return spec

    @classmethod
    def from_json(cls, text: str, *, strict: bool = True) -> "FeatureSpec":
        return cls.from_dict(json.loads(text), strict=strict)

    # ── computation ───────────────────────────────────────────────────────────

    def compute(self, inputs: Mapping[str, Any]) -> np.ndarray:
        """Build the ``(n_rows, n_features)`` matrix, in spec column order.

        Every input array must be 1-D and the same length: ragged inputs mean two
        columns are describing different events, and a matrix built from them is
        misaligned in a way no later check can detect.
        """
        values: dict[str, np.ndarray] = {}
        n: int | None = None
        for key, arr in inputs.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 1:
                raise FeatureError(f"input {key!r} must be 1-D, got shape {a.shape}")
            if n is None:
                n = a.size
            elif a.size != n:
                raise FeatureError(
                    f"input {key!r} has {a.size} rows but earlier inputs have {n}; "
                    "inputs of different lengths are not the same events"
                )
            finite = a[np.isfinite(a)]
            if finite.size and np.max(np.abs(finite)) >= _EXACT_FLOAT_LIMIT:
                raise FeatureError(
                    f"input {key!r} exceeds 2^53 and cannot be held exactly in float64; "
                    "nanosecond timestamps are not features — pass them alongside the matrix"
                )
            values[key] = a
        if n is None:
            raise FeatureError(f"spec {self.name!r} needs inputs {list(self.required_inputs)}")

        collisions = sorted(set(self.columns) & set(values))
        if collisions:
            # If a column could shadow a supplied input, "which one did feature X
            # read?" depends on evaluation order — and the answer changes when a
            # column is inserted, silently re-pointing a downstream feature.
            raise FeatureError(f"column name(s) {collisions} collide with supplied input names")

        out = np.empty((n, len(self.features)), dtype=np.float64)
        for j, d in enumerate(self.features):
            info = feature_info(d.feature)
            args = []
            for source in d.sources:
                try:
                    args.append(values[source])
                except KeyError:
                    raise FeatureError(
                        f"column {d.column!r} reads {source!r}, which is neither a "
                        f"supplied input {sorted(values)} nor an earlier column"
                    ) from None
            column = np.asarray(info.fn(*args, **dict(d.params)), dtype=np.float64)
            if column.shape != (n,):
                raise FeatureError(
                    f"feature {d.feature!r} returned {column.shape} for {n} rows; "
                    "features must be length-preserving or every row shifts against its event"
                )
            values[d.column] = column
            out[:, j] = column
        return out


def _canonical_json(payload: Mapping[str, Any]) -> str:
    # sort_keys makes dict order irrelevant; the `features` *list* keeps its order,
    # which is the point — column order is part of what is being identified.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(body: Mapping[str, Any]) -> str:
    # 64 bits of SHA-256. Long enough that an accidental collision between two specs
    # in one registry is not a thing that happens; short enough to read in a log line
    # and to sit inside an artifact filename.
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:16]


def spec_from_defs(name: str, version: int, defs: Iterable[FeatureDef]) -> FeatureSpec:
    """Convenience constructor for building a spec from an iterable."""
    return FeatureSpec(name=name, version=version, features=tuple(defs))


#: The reference spec for a crypto perp strategy: book state, a mid-based return
#: ladder, and trade flow. It exists to be a real, runnable example of the shape —
#: a strategy is expected to define its own and pin it in its artifact.
PERP_CORE_V1 = FeatureSpec(
    name="perp_core",
    version=1,
    features=(
        FeatureDef("mid", "mid_price"),
        FeatureDef("spread_bps", "relative_spread"),
        FeatureDef("book_imb", "book_imbalance"),
        FeatureDef("ret_1", "log_return", params={"period": 1}, inputs={"price": "mid"}),
        FeatureDef("mom_32", "momentum", params={"window": 32}, inputs={"price": "mid"}),
        FeatureDef(
            "vol_32", "realized_volatility", params={"window": 32}, inputs={"price": "mid"}
        ),
        FeatureDef(
            "ema_x_8_32", "ema_crossover", params={"fast": 8, "slow": 32}, inputs={"price": "mid"}
        ),
        FeatureDef("z_32", "rolling_zscore", params={"window": 32}, inputs={"x": "mid"}),
        FeatureDef("tfi_32", "trade_flow_imbalance", params={"window": 32}),
    ),
)


#: The shared bar spec for the model zoo (ADR-0032): six columns over a closed m1
#: OHLCV candle, deliberately the smallest recipe that still says something about a
#: bar. It lives **here** rather than inside a strategy module — the convention
#: :data:`PERP_CORE_V1` describes — because more than one strategy is held to it on
#: purpose: the zoo's three model families and the no-model statistical baseline all
#: read the same columns, and that is the only reason their results are comparable at
#: all. A spec copied into two modules is two specs the day one of them is edited, and
#: the fingerprint would move on one side only.
#:
#: **Every window is finite and the longest is 21 bars.** ``vol_20`` is a 20-sample
#: standard deviation of one-step log returns, so its first window reaches back through
#: the return's own extra bar: the first finite row is index 20, the 21st bar. On m1
#: that is a **21-minute** warmup, which is the whole argument for the interval —
#: ``perp_bar``'s 24-bar window on hourly bars means no opinion for 25 hours, so a
#: session has to outlive a day before the strategy says anything at all. Nothing here
#: is an EMA or an expanding statistic, which is what lets a bounded serving buffer
#: reproduce the offline recompute *bit for bit* rather than within a tolerance.
#:
#: Every column is computable from what a ``candle`` subscription actually publishes.
#: A feature reading the book or the tape would be a research artifact no strategy on
#: this spec could ever serve.
BAR_M1_V1 = FeatureSpec(
    name="bar_m1",
    version=1,
    features=(
        # Return, at two horizons: the last bar and the last five minutes.
        FeatureDef("ret_1", "log_return", params={"period": 1}, inputs={"price": "close"}),
        FeatureDef("mom_5", "momentum", params={"window": 5}, inputs={"price": "close"}),
        # Where the last close sits in its own recent distribution.
        FeatureDef("z_20", "rolling_zscore", params={"window": 20}, inputs={"x": "close"}),
        # Volatility from closes, and the ground the bar itself covered. The two
        # disagree exactly when a bar round-trips, which is the state worth knowing.
        FeatureDef(
            "vol_20", "realized_volatility", params={"window": 20}, inputs={"price": "close"}
        ),
        FeatureDef("range_bps", "relative_range"),
        # The closest thing to flow a candle feed can honestly supply.
        FeatureDef("clv", "close_location"),
    ),
)

#: The first row index of :data:`BAR_M1_V1` that can be finite, and therefore the
#: number of bars a serving path must have seen before it has any opinion at all.
#: Stated as a constant so the warmup claim is a *test* rather than a sentence in a
#: docstring — ``test_zoo.py`` asserts the spec agrees with it, which is what catches
#: the day somebody widens a window and quietly puts the strategy back to sleep for an
#: extra quarter of an hour.
BAR_M1_WARMUP_BARS = 21


__all__ = [
    "BAR_M1_V1",
    "BAR_M1_WARMUP_BARS",
    "PERP_CORE_V1",
    "FeatureDef",
    "FeatureSpec",
    "FeatureSpecMismatch",
    "spec_from_defs",
]
