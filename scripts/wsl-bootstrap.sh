#!/usr/bin/env bash
# One-time setup INSIDE a fresh WSL2 Ubuntu, then run the full gate.
# From Windows, after `wsl --install` + reboot, open Ubuntu and run:
#   cd /mnt/c/Users/<you>/Documents/hyperliquid-ml-rust-layer
#   bash scripts/wsl-bootstrap.sh
# (sudo will prompt for your password.)
#
# Notes:
# - Python is pinned to 3.12 via `uv` (NOT the distro's system Python): the ML
#   stack (torch/onnx/xgboost/lightgbm) lags the newest Python, and quality is
#   sacred here — we don't want source-built wheels. See docs/DEVELOPMENT.md.
# - Cargo builds into a Linux-native target dir ($HOME/axon-target) so the
#   /mnt/c-mounted repo isn't slow to build and the Windows target/ isn't clobbered.
set -euo pipefail

echo "==> apt: build tools"
sudo apt-get update
sudo apt-get install -y build-essential pkg-config curl

echo "==> rustup (stable)"
if [ ! -x "$HOME/.cargo/bin/cargo" ] && ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version && cargo --version

echo "==> uv + pinned Python 3.12 venv"
if [ ! -x "$HOME/.local/bin/uv" ] && ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv venv --python 3.12 "$HOME/axon-venv"
uv pip install --python "$HOME/axon-venv/bin/python" numpy pytest

echo "==> running the full gate (3.12 venv, Linux-native target dir)"
export CARGO_TARGET_DIR="$HOME/axon-target"
# shellcheck disable=SC1090
source "$HOME/axon-venv/bin/activate"
bash "$(dirname "${BASH_SOURCE[0]}")/check.sh"

echo ""
echo "WSL2 ready. Activate the venv in new shells with:"
echo "  source ~/axon-venv/bin/activate && export PATH=\"\$HOME/.cargo/bin:\$PATH\" CARGO_TARGET_DIR=\"\$HOME/axon-target\""
