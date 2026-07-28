"""What an exported model *is*: bytes, plus the record that makes them reproducible.

``docs/02`` calls the model artifact an **offline handoff** — Python hands the
execution plane a file, a schema and a version, and from then on every signal
refers to the model only by that version (``model_version`` on the wire record).
Everything here exists to make that reference resolve to one exact set of bytes,
forever. The moment a version can name two different models, no past decision is
reproducible, and the shadow-trading and audit story in ``docs/07`` is fiction.

The metadata is deliberately a *record of what happened*, not a description of
intent: the content hash, the I/O schema, the library versions and the measured
export-time deviation are all filled in by the exporter from the artifact it
actually produced (:mod:`axon.models.export`), never asserted by the caller.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Bumped when the on-disk ``meta.json`` layout changes. A metadata file with no
#: version of its own is the file that breaks silently the first time a field is
#: added; this lets an older Axon *refuse* a newer artifact instead of loading
#: half of it.
META_SCHEMA = 1

#: The signal record's ``model_version`` is a ``u32`` (``contracts/schema.toml``),
#: and that field is the only thing tying a fill back to the model that caused it.
#: A version the wire cannot carry is a version no audit can resolve, so the
#: registry refuses to mint one.
MAX_VERSION = 2**32 - 1

#: Artifact *formats*, not model families — the loader dispatches on format, and
#: an sklearn model and a Torch model both arrive as ``onnx``. Which family it came
#: from is recorded in ``producer``.
KINDS: tuple[str, ...] = ("xgboost", "lightgbm", "onnx")

#: One filename per format so a human (and a Rust startup path) can find the model
#: without parsing the metadata first.
ARTIFACT_FILENAMES: Mapping[str, str] = {
    "xgboost": "model.json",
    "lightgbm": "model.txt",
    "onnx": "model.onnx",
}

META_FILENAME = "meta.json"

# A registry id becomes a directory name. Anything outside this alphabet — a
# slash, a `..`, a NUL, a drive letter — turns `load(id)` into a file read of the
# caller's choosing, and an id that differs only by case collides on Windows.
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ModelError(Exception):
    """Base for every failure in :mod:`axon.models`."""


class IntegrityError(ModelError):
    """Stored bytes do not match the metadata that describes them."""


@dataclass(frozen=True)
class TensorSpec:
    """One model input or output: the shape and dtype the Rust side must feed it.

    ``None`` in ``shape`` is the batch dimension. Recorded per-tensor rather than
    as a bare feature count because a graph with two inputs, or one that returns
    ``(label, probabilities)``, is normal and a single number cannot describe it.
    """

    name: str
    dtype: str
    shape: tuple[int | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSpec":
        unknown = set(data) - {"name", "dtype", "shape"}
        if unknown:
            raise ValueError(f"unknown TensorSpec keys: {sorted(unknown)}")
        shape = tuple(None if d is None else int(d) for d in data.get("shape", ()))
        return cls(name=str(data["name"]), dtype=str(data["dtype"]), shape=shape)


@dataclass(frozen=True)
class ArtifactMeta:
    """Immutable metadata attached to every exported model artifact.

    The first five fields are what ADR-0003 requires of every artifact. The rest
    are filled in by the exporter: leave them alone when *declaring* an export and
    read them back off the artifact it returns.

    ``created_ns`` is a wall clock, and is deliberately *not* what orders the
    registry — a restored backup has fresh timestamps and unchanged versions, so
    "latest" is always the highest version, never the newest file.
    """

    registry_id: str
    version: int
    kind: str = ""
    feature_spec_ref: str = ""
    git_sha: str = ""
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()
    score_output: str = ""
    producer: Mapping[str, str] = field(default_factory=dict)
    content_sha256: str = ""
    content_bytes: int = 0
    artifact_filename: str = ""
    roundtrip_max_abs_diff: float = 0.0
    roundtrip_rows: int = 0
    created_ns: int = 0

    # ── validation ──

    def validate(self) -> None:
        """Check what the *caller* declares. Called before an export runs."""
        validate_registry_id(self.registry_id)
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError(f"version must be an int, got {type(self.version).__name__}")
        if self.version < 1:
            # Zero is what an unset field looks like, and the wire cannot tell the
            # difference between "no model" and "model 0".
            raise ValueError(f"version must be >= 1 (0 reads as unset), got {self.version}")
        if self.version > MAX_VERSION:
            raise ValueError(
                f"version {self.version} exceeds the u32 model_version field on the signal "
                "record; a version the wire cannot carry cannot be audited"
            )
        if self.kind and self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not one of {KINDS}")
        if not self.feature_spec_ref:
            # A model without the features that fed it is not reproducible, and an
            # artifact that is not reproducible is the one thing this module exists
            # to prevent (docs/03: feature parity is the harder half). The string is
            # opaque here, but the convention is `axon.features.FeatureSpec.ref` —
            # `name/vN#fingerprint`, which pins the transforms as well as their names.
            raise ValueError("feature_spec_ref is required; see docs/03 and axon.features")

    def validate_complete(self) -> None:
        """Check the full record. Called before anything is written to a registry."""
        self.validate()
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not one of {KINDS}")
        if not self.git_sha:
            raise ValueError("git_sha is required (axon.models.current_git_sha fills it)")
        if not _SHA_RE.match(self.content_sha256 or ""):
            raise ValueError(f"content_sha256 {self.content_sha256!r} is not 'sha256:<64 hex>'")
        if self.content_bytes <= 0:
            raise ValueError("content_bytes must be > 0")
        filename = self.artifact_filename
        if not filename or Path(filename).name != filename:
            raise ValueError(f"artifact_filename {filename!r} must be a bare name")
        if not self.inputs:
            raise ValueError("an artifact with no declared inputs cannot be fed by the Rust side")
        for spec in self.inputs:
            if spec.dtype != "float32":
                # ADR-0005: FP32 end to end. FP16 flips predictions across decision
                # thresholds; FP64 means the graph computes something the f32 Rust
                # path structurally cannot reproduce.
                raise ValueError(f"input {spec.name!r} is {spec.dtype}, not float32 (ADR-0005)")
        if not self.outputs:
            raise ValueError("an artifact with no declared outputs cannot be consumed")
        if self.score_output not in {s.name for s in self.outputs}:
            # A classifier graph returns (label, probabilities) and a consumer that
            # guesses which one is the score has a 50% chance of trading on a class
            # index. The artifact names it once, here.
            raise ValueError(
                f"score_output {self.score_output!r} is not one of "
                f"{sorted(s.name for s in self.outputs)}"
            )
        if self.roundtrip_rows <= 0:
            # An artifact whose export was never round-tripped has no evidence it
            # still is the model that was trained (ADR-0005).
            raise ValueError("artifact was never round-trip verified; export it via axon.models")

    # ── serialization ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_schema": META_SCHEMA,
            "registry_id": self.registry_id,
            "version": self.version,
            "kind": self.kind,
            "feature_spec_ref": self.feature_spec_ref,
            "git_sha": self.git_sha,
            "inputs": [s.to_dict() for s in self.inputs],
            "outputs": [s.to_dict() for s in self.outputs],
            "score_output": self.score_output,
            "producer": dict(self.producer),
            "content_sha256": self.content_sha256,
            "content_bytes": self.content_bytes,
            "artifact_filename": self.artifact_filename,
            "roundtrip_max_abs_diff": self.roundtrip_max_abs_diff,
            "roundtrip_rows": self.roundtrip_rows,
            "created_ns": self.created_ns,
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, two-space indent, trailing newline.

        Canonical so that two exports of the same model produce byte-identical
        metadata and a diff between two registries is readable rather than a
        reordering.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactMeta":
        data = dict(data)
        schema = int(data.pop("meta_schema", 0))
        if schema > META_SCHEMA:
            raise ValueError(
                f"artifact metadata is schema {schema}, this Axon understands {META_SCHEMA}; "
                "refusing rather than reading unknown fields as absent"
            )
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            # Same rule as StrategyConfig.from_dict: a silently ignored key is a
            # property the operator believes is recorded and is not.
            raise ValueError(f"unknown ArtifactMeta keys: {sorted(unknown)}")
        data["inputs"] = tuple(TensorSpec.from_dict(s) for s in data.get("inputs", ()))
        data["outputs"] = tuple(TensorSpec.from_dict(s) for s in data.get("outputs", ()))
        data["producer"] = dict(data.get("producer", {}))
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "ArtifactMeta":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class Artifact:
    """An exported model in memory: the exact bytes plus their completed record."""

    meta: ArtifactMeta
    payload: bytes

    def verify(self) -> None:
        """Raise :class:`IntegrityError` unless the payload is what the metadata says.

        Checks length first: a truncated file is the common failure (a full disk, a
        killed copy), and reporting the size is far more actionable than reporting
        two hashes that differ.
        """
        if len(self.payload) != self.meta.content_bytes:
            raise IntegrityError(
                f"{self.ref}: artifact is {len(self.payload)} bytes, metadata says "
                f"{self.meta.content_bytes} — the file is truncated or was replaced"
            )
        actual = content_hash(self.payload)
        if actual != self.meta.content_sha256:
            raise IntegrityError(
                f"{self.ref}: content hash {actual} does not match the recorded "
                f"{self.meta.content_sha256} — these are not the bytes that were verified"
            )

    @property
    def ref(self) -> str:
        """``id@version`` — how an artifact is named in logs and error messages."""
        return f"{self.meta.registry_id}@{self.meta.version}"


def validate_registry_id(registry_id: Any) -> str:
    """Refuse an id that cannot safely be a directory name. Returns it unchanged."""
    if not isinstance(registry_id, str) or not _ID_RE.match(registry_id):
        raise ValueError(
            f"registry_id {registry_id!r} must match {_ID_RE.pattern} — it becomes a directory "
            "name, and a path separator or '..' in it makes load() read files of the caller's "
            "choosing; lowercase-only because a case-insensitive filesystem would silently "
            "merge two ids into one"
        )
    return registry_id


def content_hash(payload: bytes) -> str:
    """SHA-256 of the artifact bytes, prefixed with the algorithm that produced it."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def current_git_sha(repo: str | Path | None = None) -> str:
    """The commit an export was built from, suffixed ``-dirty`` if the tree is not clean.

    Returns ``"unknown"`` rather than raising when git is unavailable (a notebook,
    an unpacked tarball, a CI stage without the history). An artifact born outside
    a checkout *is* less reproducible, and recording that honestly beats both
    refusing to export and stamping a sha that reproduces nothing.

    The ``-dirty`` suffix matters more than it looks: a clean-looking sha on a
    model trained from uncommitted code is a reproduction instruction that silently
    rebuilds a different model.
    """
    cwd = str(repo) if repo is not None else str(Path(__file__).resolve().parents[3])
    sha = _git(["rev-parse", "HEAD"], cwd)
    if sha is None:
        return "unknown"
    dirty = _git(["status", "--porcelain"], cwd)
    return sha + ("-dirty" if dirty else "")


def _git(args: Sequence[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8", "replace").strip()


__all__ = [
    "ARTIFACT_FILENAMES",
    "Artifact",
    "ArtifactMeta",
    "IntegrityError",
    "KINDS",
    "MAX_VERSION",
    "META_FILENAME",
    "META_SCHEMA",
    "ModelError",
    "TensorSpec",
    "content_hash",
    "current_git_sha",
    "validate_registry_id",
]
