"""Shared pytest fixtures: repo location and the built ``axon-ipc`` example binaries.

Every cross-language test drives one of those examples, and every one of them needs
the same build-if-missing / skip-if-unbuildable dance. It lives here once so a test
file cannot end up with a private copy that quietly stops matching.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# python/tests/conftest.py → parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def _target_dir() -> Path:
    """Cargo's target dir — honors CARGO_TARGET_DIR (e.g. a fast Linux-native path
    when building the /mnt/c-mounted repo under WSL2)."""
    env = os.environ.get("CARGO_TARGET_DIR")
    return Path(env) if env else REPO_ROOT / "target"


def _example_path(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return _target_dir() / "debug" / "examples" / f"{name}{suffix}"


def _find_cargo() -> str | None:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    candidate = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    return str(candidate) if candidate.exists() else None


#: Every ``axon-ipc`` example a cross-language test drives, keyed by the name tests
#: ask for. One list in one place: they come out of a single cargo invocation, so a
#: test that carried its own copy of the build-and-skip dance would drift the day a
#: fourth example landed — and that drift shows up as a test that silently skips
#: forever, which reads exactly like a passing suite.
_IPC_EXAMPLES = {
    "reader": "ipc_reader",
    "writer": "ipc_writer",
    "md_writer": "md_writer",
    "md_bar_writer": "md_bar_writer",
}


@pytest.fixture(scope="session")
def _ipc_examples() -> tuple[dict[str, Path], str | None]:
    """Build every ``axon-ipc`` example once if *any* is missing.

    Building on "any missing" rather than "all missing" is what covers a target dir
    that predates an example — the older binaries are there and the new one is not,
    and only a rebuild tells them apart.

    Returns the paths plus the reason the build could not run, if it could not. The
    reason travels instead of skipping here on purpose: which binaries a test needs
    is the test's own business, and skipping the whole session over an example nobody
    asked for would turn one stale target dir into a suite-wide skip.
    """
    built = {key: _example_path(name) for key, name in _IPC_EXAMPLES.items()}
    if all(p.exists() for p in built.values()):
        return built, None
    cargo = _find_cargo()
    if cargo is None:
        return built, "cargo not found; cannot build the Rust IPC examples"
    try:
        subprocess.run(
            [cargo, "build", "-q", "-p", "axon-ipc", "--examples"],
            cwd=REPO_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        return built, f"failed to build the Rust IPC examples: {e}"
    return built, None


def _require_examples(ipc_examples, *keys: str) -> dict[str, str]:
    """Resolve `keys` to built binaries, or skip naming the ones that are absent."""
    built, reason = ipc_examples
    missing = [_IPC_EXAMPLES[k] for k in keys if not built[k].exists()]
    if missing:
        pytest.skip(reason or f"Rust IPC example(s) not built: {', '.join(missing)}")
    return {k: str(built[k]) for k in keys}


@pytest.fixture(scope="session")
def rust_examples(_ipc_examples) -> dict[str, str]:
    """The signal-ring round-trip pair: ``ipc_reader`` and ``ipc_writer``.

    Skips cleanly when the toolchain or the binaries are unavailable (a CI stage
    without Rust), so a Python-only environment still passes.
    """
    return _require_examples(_ipc_examples, "reader", "writer")


@pytest.fixture(scope="session")
def md_writer(_ipc_examples) -> str:
    """The ``md_writer`` example — the only producer of the market-data ring today.

    Independent of :func:`rust_examples` so a target dir holding one pair but not the
    other skips only the tests that actually need the missing binary.
    """
    return _require_examples(_ipc_examples, "md_writer")["md_writer"]


@pytest.fixture(scope="session")
def md_bar_writer(_ipc_examples) -> str:
    """The ``md_bar_writer`` example — the fixture producer of the **bar** ring.

    Separate from :func:`md_writer` for the same reason that one is separate from
    :func:`rust_examples`: the two rings carry different records, and a target dir
    that predates one should skip only the tests that need it.
    """
    return _require_examples(_ipc_examples, "md_bar_writer")["md_bar_writer"]


@pytest.fixture
def ring_path(tmp_path) -> str:
    return str(tmp_path / "axon-test.ring")
