"""Modal Volume artifacts: the only channel Axon uses to move bytes off a job.

A Modal task's return value is pickled back through the control plane, so it has
to stay small; a feature matrix or a model file sent that way either fails or
turns a cheap fan-out into a serialization benchmark. Small is not sufficient,
either — it must also be **plain-typed**, because the unpickling happens on the
client and a value whose class lives in a third-party module makes the client
import that module (ADR-0017 §4). Everything of size therefore travels on a
**Volume**: the spec names it as ``volume://<name>/<sub>``
in ``inputs``/``outputs``, hwsched's Modal adapter mounts it at ``/vol/<name>``,
and the remote code reads and writes ordinary paths under that mount.

The URI and the mount point are two views of one location, which is exactly the
kind of pair that drifts. :func:`volume_uri` and :func:`mount_path` are derived
from the same parsed :class:`Artifact` so a job cannot declare one path and write
another — a mismatch there does not fail, it silently writes into the container's
ephemeral disk and the artifact is gone when the container exits.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

#: The scheme hwsched's Modal adapter recognizes (``providers/modal.py``).
VOLUME_SCHEME = "volume://"

#: Where the adapter mounts every referenced volume. Convention, not config —
#: hwsched hard-codes ``/vol/<name>``, so hard-code it here too rather than
#: inventing a knob whose two ends could disagree.
MOUNT_ROOT = "/vol"

# Modal volume names are DNS-ish labels. Rejecting a bad one here beats
# discovering it at submit time, after a plan was already approved.
_VOLUME_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


class ArtifactError(ValueError):
    """A volume reference that would not resolve to the location it claims."""


@dataclass(frozen=True)
class Artifact:
    """One location inside a Modal Volume, addressable as a URI or a path."""

    volume: str
    subpath: str = ""

    def __post_init__(self) -> None:
        if not _VOLUME_NAME.match(self.volume):
            raise ArtifactError(
                f"invalid Modal volume name {self.volume!r}: expected a lowercase "
                "DNS-style label ([a-z0-9][a-z0-9._-]*)"
            )
        object.__setattr__(self, "subpath", _clean_subpath(self.subpath))

    @property
    def uri(self) -> str:
        """``volume://<name>/<subpath>`` — what goes in the JobSpec."""
        return f"{VOLUME_SCHEME}{self.volume}/{self.subpath}" if self.subpath else (
            f"{VOLUME_SCHEME}{self.volume}"
        )

    @property
    def path(self) -> str:
        """``/vol/<name>/<subpath>`` — what the remote code opens."""
        return posixpath.join(MOUNT_ROOT, self.volume, self.subpath) if self.subpath else (
            posixpath.join(MOUNT_ROOT, self.volume)
        )

    def child(self, subpath: str) -> "Artifact":
        """A location below this one, validated the same way."""
        return Artifact(self.volume, posixpath.join(self.subpath, subpath) if self.subpath
                        else subpath)

    @classmethod
    def parse(cls, uri: str) -> "Artifact":
        """Parse ``volume://name/sub``; raises :class:`ArtifactError` on anything else."""
        parsed = try_parse(uri)
        if parsed is None:
            raise ArtifactError(f"not a Modal volume URI: {uri!r} (expected {VOLUME_SCHEME}…)")
        return parsed


def _clean_subpath(subpath: str) -> str:
    """Normalize a relative subpath, refusing anything that leaves the mount.

    ``..`` and a leading ``/`` are the two ways a write lands outside the volume.
    Neither raises at runtime — the write succeeds against container-local disk
    and the bytes vanish on exit — so they have to be refused here, where the
    caller is still watching.
    """
    cleaned = subpath.strip().strip("/")
    if not cleaned:
        return ""
    if "\\" in cleaned:
        raise ArtifactError(
            f"volume subpath {subpath!r} contains a backslash; the mount is POSIX"
        )
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ArtifactError(
            f"volume subpath {subpath!r} escapes the mount with '..'; artifacts written "
            "outside /vol are lost when the container exits"
        )
    return "/".join(parts)


def try_parse(uri: str) -> Artifact | None:
    """:class:`Artifact` for a volume URI, ``None`` for anything else."""
    if not isinstance(uri, str) or not uri.startswith(VOLUME_SCHEME):
        return None
    rest = uri[len(VOLUME_SCHEME):]
    name, _, sub = rest.partition("/")
    if not name:
        return None
    return Artifact(name, sub)


def volume_uri(volume: str, subpath: str = "") -> str:
    """The ``volume://`` URI to put in a JobSpec's ``inputs``/``outputs``."""
    return Artifact(volume, subpath).uri


def mount_path(volume: str, subpath: str = "") -> str:
    """The in-container path the same artifact is readable/writable at."""
    return Artifact(volume, subpath).path


def path_for(uri: str) -> str:
    """Mount path for an already-formed ``volume://`` URI.

    Remote code is handed URIs (they are what the spec carries) but has to open
    paths; going through here keeps the translation in one place.
    """
    return Artifact.parse(uri).path


__all__ = [
    "MOUNT_ROOT",
    "VOLUME_SCHEME",
    "Artifact",
    "ArtifactError",
    "mount_path",
    "path_for",
    "try_parse",
    "volume_uri",
]
