#!/usr/bin/env bash
# Full local CI gate — mirrors .github/workflows/ci.yml.
# Works in Git Bash (Windows) and in WSL2/Linux. Run from anywhere:
#   bash scripts/check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Make cargo available even if it isn't on PATH yet (fresh rustup install).
export PATH="$HOME/.cargo/bin:$PATH"
# A globally-installed pytest plugin must never break our run.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="python"

echo "==> rustfmt --check"
cargo fmt --all --check

echo "==> clippy (deny warnings)"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cargo test --workspace"
cargo test --workspace

echo "==> build IPC examples (for the cross-language round-trip test)"
cargo build -p axon-ipc --examples

echo "==> pytest (incl. Python<->Rust round-trip)"
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
"$PY" -m pytest python/tests -q

echo ""
echo "All checks passed."
