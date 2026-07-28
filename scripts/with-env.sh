#!/usr/bin/env bash
# Run a command with the repo's `.env` exported — the only sanctioned way to get a
# signing key into a process. Keeps secrets off argv (visible in `ps`), out of
# shell history, and out of CI logs.
#
#   bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid --example wallet_info
#   bash scripts/with-env.sh cargo test -p axon-provider-hyperliquid -- --ignored
#
# `.env` is gitignored; copy `.env.example` to start. Missing `.env` is fatal
# rather than silent, so a live test never runs against the wrong identity.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "scripts/with-env.sh: no .env at $ROOT (copy .env.example and fill it in)" >&2
  exit 1
fi

# `set -a` exports everything sourced; comments and blank lines are fine.
set -a
# shellcheck disable=SC1091
source .env
set +a

export PATH="$HOME/.cargo/bin:$PATH"

if [[ $# -eq 0 ]]; then
  echo "scripts/with-env.sh: nothing to run (pass a command)" >&2
  exit 2
fi

exec "$@"
