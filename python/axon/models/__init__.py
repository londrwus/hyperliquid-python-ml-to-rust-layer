"""``axon.models`` — model export and a versioned, immutable artifact registry.

The export workflow (``docs/03-ml-fidelity-and-features.md``, ADR-0003):

```
  train (Python)
     └─▶ export:  trees → the library's own format  |  sklearn/NN → ONNX FP32, opt_level=0
            └─▶ attach: version, I/O schema, feature-spec ref, git SHA, content hash
                   └─▶ registry (immutable versioned artifact)
                          └─▶ Rust loads it at startup and runs the model-parity gate
```

Why any of this is worth the code: the signal record carries a ``model_version``
and nothing else about the model. If that number cannot be resolved, years later,
to one exact set of bytes, then no live decision can be replayed offline — and
shadow trading, post-hoc audits and the whole parity ladder in ``docs/07`` rest on
being able to replay them. Immutability is therefore not tidiness; it is the
property that makes the audit trail true.

**FP32 end to end, no quantization** (ADR-0005). Every export is round-tripped and
audited before it is allowed to become an artifact — see :mod:`axon.models.export`.

Typical use::

    from axon.models import ArtifactMeta, ModelRegistry, export_artifact

    registry = ModelRegistry("artifacts")
    meta = ArtifactMeta(
        registry_id="btc_1m_xgb",
        version=registry.next_version("btc_1m_xgb"),
        feature_spec_ref=spec.ref,  # axon.features: "name/vN#fingerprint"
    )
    registry.save(export_artifact(model, meta, X_valid))
    ...
    predictor = registry.load_predictor("btc_1m_xgb")  # latest, hash-verified
"""

from __future__ import annotations

from typing import Any

from axon.models.artifact import (
    ARTIFACT_FILENAMES,
    KINDS,
    MAX_VERSION,
    META_FILENAME,
    META_SCHEMA,
    Artifact,
    ArtifactMeta,
    IntegrityError,
    ModelError,
    TensorSpec,
    content_hash,
    current_git_sha,
)
from axon.models.export import (
    DEFAULT_TOLERANCE,
    ExportError,
    FidelityError,
    PrecisionError,
    audit_fp32,
    export_artifact,
)
from axon.models.inference import Predictor, load_predictor, session_options
from axon.models.registry import (
    ArtifactExistsError,
    ArtifactNotFoundError,
    ModelRegistry,
    RegistryError,
)


def export(
    model: Any, meta: ArtifactMeta, out_dir: str, *, sample_input: Any, **kwargs: Any
) -> str:
    """Export ``model`` into the registry rooted at ``out_dir``; return the artifact path.

    A one-call convenience over :func:`~axon.models.export.export_artifact` plus
    :meth:`~axon.models.registry.ModelRegistry.save`. ``sample_input`` is required
    for the reason it is required everywhere here: an export nobody re-ran is an
    export nobody checked (ADR-0005).
    """
    artifact = export_artifact(model, meta, sample_input, **kwargs)
    registry = ModelRegistry(out_dir)
    registry.save(artifact)
    return str(registry.artifact_path(artifact.meta.registry_id, artifact.meta.version))


__all__ = [
    "ARTIFACT_FILENAMES",
    "DEFAULT_TOLERANCE",
    "KINDS",
    "MAX_VERSION",
    "META_FILENAME",
    "META_SCHEMA",
    "Artifact",
    "ArtifactExistsError",
    "ArtifactMeta",
    "ArtifactNotFoundError",
    "ExportError",
    "FidelityError",
    "IntegrityError",
    "ModelError",
    "ModelRegistry",
    "PrecisionError",
    "Predictor",
    "RegistryError",
    "TensorSpec",
    "audit_fp32",
    "content_hash",
    "current_git_sha",
    "export",
    "export_artifact",
    "load_predictor",
    "session_options",
]
