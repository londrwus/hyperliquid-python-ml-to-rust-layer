"""A filesystem model registry: immutable versions, verified on the way out.

```
<root>/<registry_id>/v0000000007/model.onnx   ← the artifact, byte-for-byte
                                /meta.json     ← the record that describes it
```

Three properties, each of which exists because of a specific way a model registry
goes wrong:

**Immutable.** Writing an ``(registry_id, version)`` that already exists is an
error, never an overwrite. A mutable version means the ``model_version`` stamped
on a signal three weeks ago no longer identifies the model that produced it, and
every audit and shadow-trading comparison built on it (``docs/07``) is quietly
answering a different question than it was asked.

**Ordered by version, never by time.** "Latest" is the highest version number. A
registry restored from backup has fresh mtimes and unchanged versions, and a
registry sorted lexically puts ``v10`` before ``v2``.

**Verified on load.** The stored bytes are hashed and compared to the metadata
beside them, so a truncated copy, a hand-edited file or a swapped model is caught
at load rather than at the first divergent fill.

A directory tree is the whole implementation on purpose. It is inspectable with
``ls``, it rsyncs, it needs no daemon, and the Rust side can load an artifact at
startup by opening a path — a database here would buy nothing the trading system
actually needs.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import uuid
from pathlib import Path

from axon.models.artifact import (
    META_FILENAME,
    Artifact,
    ArtifactMeta,
    IntegrityError,
    ModelError,
    validate_registry_id,
)
from axon.models.inference import Predictor, load_predictor

#: Zero-padded to the width of a u32 so that a plain ``ls`` sorts the way a human
#: reads it, and so a version directory can never be spelled two ways.
_VERSION_FMT = "v{:010d}"
_VERSION_RE = re.compile(r"v(\d{10})\Z")
_STAGING_PREFIX = ".staging-"


class RegistryError(ModelError):
    """Base for registry failures."""


class ArtifactExistsError(RegistryError):
    """That ``(registry_id, version)`` is already taken. Versions are immutable."""


class ArtifactNotFoundError(RegistryError):
    """No such artifact in this registry."""


class ModelRegistry:
    """Versioned artifacts under one root directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"ModelRegistry({str(self.root)!r})"

    # ── paths ──

    def version_dir(self, registry_id: str, version: int) -> Path:
        return self.root / validate_registry_id(registry_id) / _VERSION_FMT.format(version)

    def artifact_path(self, registry_id: str, version: int | None = None) -> Path:
        """Where the model file itself lives — what a Rust startup path is handed."""
        meta = self.load_meta(registry_id, version)
        return self.version_dir(registry_id, meta.version) / meta.artifact_filename

    # ── writing ──

    def save(self, artifact: Artifact) -> Path:
        """Write ``artifact`` and return its directory. Refuses an existing version.

        The write is staged in a sibling directory and moved into place with a
        single rename, so a crash mid-write leaves a staging directory that
        ``list`` never sees rather than a half-written version that can never be
        rewritten. The rename is also what makes the existence check race-free:
        two processes exporting the same version cannot both win, whatever they
        each saw a moment earlier.
        """
        artifact.meta.validate_complete()
        artifact.verify()
        dest = self.version_dir(artifact.meta.registry_id, artifact.meta.version)
        if dest.exists():
            raise ArtifactExistsError(
                f"{artifact.ref} already exists at {dest}: registry versions are immutable. "
                "Export a new version — overwriting one makes every decision it produced "
                "unreproducible."
            )

        staging = self.root / f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
        try:
            staging.mkdir(parents=True)
            (staging / artifact.meta.artifact_filename).write_bytes(artifact.payload)
            # newline="" so the canonical JSON is byte-identical on Windows too;
            # otherwise the same export produces two different files by platform.
            (staging / META_FILENAME).write_text(
                artifact.meta.to_json(), encoding="utf-8", newline=""
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(staging, dest)
            except OSError as exc:
                if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    raise ArtifactExistsError(
                        f"{artifact.ref} was created at {dest} by another writer while this "
                        "export was staging; registry versions are immutable"
                    ) from exc
                raise
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return dest

    # ── reading ──

    def load(self, registry_id: str, version: int | None = None) -> Artifact:
        """Load an artifact (``version=None`` → latest), verifying its content hash."""
        meta = self.load_meta(registry_id, version)
        path = self.version_dir(registry_id, meta.version) / meta.artifact_filename
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"{registry_id}@{meta.version}: metadata names {meta.artifact_filename}, "
                f"which is not in {path.parent}"
            )
        artifact = Artifact(meta=meta, payload=path.read_bytes())
        artifact.verify()
        return artifact

    def load_meta(self, registry_id: str, version: int | None = None) -> ArtifactMeta:
        """Read one artifact's metadata without reading the model itself."""
        resolved = self.resolve_latest(registry_id) if version is None else int(version)
        path = self.version_dir(registry_id, resolved) / META_FILENAME
        if not path.is_file():
            if path.parent.is_dir():
                # A directory with no metadata is an interrupted write, and saying
                # so is the difference between "delete this" and "why is my export
                # refused?".
                raise ArtifactNotFoundError(
                    f"{registry_id}@{resolved}: {path.parent} exists but has no "
                    f"{META_FILENAME} — an interrupted write; remove it deliberately"
                )
            raise ArtifactNotFoundError(
                f"{registry_id}@{resolved}: no such artifact under {self.root}"
            )
        meta = ArtifactMeta.from_json(path.read_text(encoding="utf-8"))
        if (meta.registry_id, meta.version) != (registry_id, resolved):
            # The path is a claim; the metadata is the record. A copied directory
            # is how one model comes to answer to two version numbers.
            raise IntegrityError(
                f"{path} holds metadata for {meta.registry_id}@{meta.version}; the directory "
                f"says {registry_id}@{resolved}. The tree was edited by hand."
            )
        return meta

    def load_predictor(self, registry_id: str, version: int | None = None) -> Predictor:
        """Load an artifact and return something that predicts (see ADR-0005 settings)."""
        return load_predictor(self.load(registry_id, version))

    # ── listing and resolution ──

    def list_ids(self) -> tuple[str, ...]:
        """Every registry id that has at least one complete version."""
        if not self.root.is_dir():
            return ()
        ids = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                validate_registry_id(entry.name)
            except ValueError:
                continue
            if self.list_versions(entry.name):
                ids.append(entry.name)
        return tuple(sorted(ids))

    def list_versions(self, registry_id: str) -> tuple[int, ...]:
        """Complete versions of one model, ascending.

        Numeric order, and only directories that actually carry metadata: an
        interrupted write is not a version, and returning one would let
        ``resolve_latest`` hand out a model that was never finished.
        """
        directory = self.root / validate_registry_id(registry_id)
        if not directory.is_dir():
            return ()
        versions = []
        for entry in directory.iterdir():
            match = _VERSION_RE.match(entry.name)
            if match and (entry / META_FILENAME).is_file():
                versions.append(int(match.group(1)))
        return tuple(sorted(versions))

    def list_artifacts(self, registry_id: str | None = None) -> tuple[ArtifactMeta, ...]:
        """Metadata for every artifact (optionally one model's), ordered id then version."""
        ids = (registry_id,) if registry_id is not None else self.list_ids()
        return tuple(
            self.load_meta(model_id, version)
            for model_id in ids
            for version in self.list_versions(model_id)
        )

    def resolve_latest(self, registry_id: str) -> int:
        """The highest complete version. Raises if the model has none."""
        versions = self.list_versions(registry_id)
        if not versions:
            raise ArtifactNotFoundError(f"no versions of {registry_id!r} under {self.root}")
        return versions[-1]

    def next_version(self, registry_id: str) -> int:
        """The version a new export should claim: one past the highest that exists.

        Derived from what is on disk rather than from a counter, because a counter
        that drifts hands out a number that is already taken — and the registry
        would then (correctly) refuse the export at the very end of a training run.
        """
        versions = self.list_versions(registry_id)
        return versions[-1] + 1 if versions else 1


__all__ = [
    "ArtifactExistsError",
    "ArtifactNotFoundError",
    "ModelRegistry",
    "RegistryError",
]
