#!/usr/bin/env bash
#
# run.sh - the single entry point for working with Axon on Linux.
#
# It puts cargo and the project venv on PATH itself, so it works from any shell
# with no prior `source` or exported variables. If the environment is missing it
# says exactly what to run instead of failing obscurely.
#
# Usage:
#   ./run.sh                 # the full gate: rustfmt, clippy, Rust tests, Python tests
#   ./run.sh check           #   (same thing, named)
#   ./run.sh test            # Rust + Python tests only, skipping the lints
#   ./run.sh build           # debug build of the whole workspace
#   ./run.sh release         # optimized build (lto, codegen-units=1, panic=abort)
#   ./run.sh runtime         # the `axon` binary: an offline session (no network, no key)
#   ./run.sh book [COIN]     # live Hyperliquid order book (default BTC). Network, no key.
#   ./run.sh wallet          # read-only key/account pre-flight. Needs .env.
#   ./run.sh agent           # approveAgent ceremony, DRY RUN by default. Needs .env.
#   ./run.sh replay [LOG]    # replay a captured event log through the real core
#   ./run.sh capture [LOG]   # record an offline session, then replay the log it wrote
#   ./run.sh parity          # the cross-language model-parity gate (offline, no ML deps)
#   ./run.sh parity-bundles  # REGENERATE the committed parity fixtures (needs the ml extra)
#   ./run.sh feature-parity  # the cross-language FEATURE-parity gate (offline, numpy only)
#   ./run.sh feature-bundles # REGENERATE the committed feature fixtures (numpy only)
#   ./run.sh strategy        # the reference Python strategy onto a signal ring
#   ./run.sh portfolio-evidence  # what a multi-strategy book would have held (minutes of CPU)
#   ./run.sh live            # the #[ignore]d live testnet tests. PLACES REAL ORDERS.
#   ./run.sh flatten CONFIG  # PLACES REAL ORDERS. Adopt the venue's own position in
#                            #   every symbol the config names and drive it to zero.
#                            #   The operator cleanup pass: no session, no dead-man's
#                            #   switch, no ring. Every order is reduce-only and sized
#                            #   from a fresh venue read, and the urgency is a ladder
#                            #   (IOC -> crossing GTC -> post-only) because a venue that
#                            #   refuses one TIF is exactly when this is needed.
#   ./run.sh fmt             # format the workspace in place
#   ./run.sh clean           # remove build artifacts (keeps .venv and .env)
#   ./run.sh doctor          # report what is installed and what is missing
#   ./run.sh help
#
# First-time setup on Ubuntu:  bash scripts/install-ubuntu.sh
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
say()  { printf '%s\n' "${CYAN}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s%s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '    %s%s%s\n' "$YELLOW" "$*" "$RESET"; }
die()  { printf '%s\n' "${RED}ERROR: $*${RESET}" >&2; exit 1; }

VENV="$ROOT/.venv"

# --------------------------------------------------------------------------- #
# environment: make this script work from a bare shell
# --------------------------------------------------------------------------- #
# rustup and uv install here; a login shell may not have them yet.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
# A globally-installed pytest plugin must never break our run (see docs/DEVELOPMENT.md).
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$ROOT/python"

if [[ -x "$VENV/bin/python" ]]; then
  # Activating rather than just calling .venv/bin/python so `python` resolves for
  # scripts/check.sh, which prefers `python` when it exists.
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

need_cargo() {
  command -v cargo >/dev/null 2>&1 || die "cargo not found.
    Run the installer first:  bash scripts/install-ubuntu.sh
    Or, if Rust is already installed, open a new shell / run: source \"\$HOME/.cargo/env\""
}

need_venv() {
  [[ -x "$VENV/bin/python" ]] || die "no Python venv at .venv
    Run the installer first:  bash scripts/install-ubuntu.sh"
}

need_env_file() {
  [[ -f "$ROOT/.env" ]] || die "no .env in the repo root.
    Copy the template and fill it in:  cp .env.example .env && chmod 600 .env
    See .env.example"
  # A placeholder key produces a confusing signature error at the venue rather than
  # an obvious local one, so catch it here.
  if grep -qE '^AXON_HL_SECRET_KEY=(0x)?0{64}\s*$' "$ROOT/.env"; then
    die "AXON_HL_SECRET_KEY in .env is still the all-zeros placeholder.
    Generate one locally (never paste a key anywhere):
      printf '0x%s\n' \"\$(openssl rand -hex 32)\"
    See .env.example"
  fi
}

usage() {
  # Print this file's leading comment block as the help text, so `run.sh help` can
  # never drift from the documentation at the top of the file. Derived from the
  # comment structure rather than a hardcoded line range, which silently goes stale
  # the first time anyone adds a line up there.
  awk 'NR == 1 { next }
       /^#/    { sub(/^# ?/, ""); print; next }
                { exit }' "${BASH_SOURCE[0]}"
}

# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
cmd_check() {
  need_cargo; need_venv
  say "Full gate (rustfmt, clippy -D warnings, cargo test, pytest)"
  bash "$ROOT/scripts/check.sh"
}

cmd_test() {
  need_cargo; need_venv
  say "Rust tests"
  cargo test --workspace || return 1
  say "Building the IPC examples (the cross-language round-trip needs them)"
  cargo build -p axon-ipc --examples || return 1
  say "Python tests"
  python -m pytest python/tests -q
}

cmd_build()   { need_cargo; say "Debug build";   cargo build --workspace; }
cmd_release() { need_cargo; say "Release build"; cargo build --workspace --release; }
cmd_fmt()     { need_cargo; say "Formatting";    cargo fmt --all; ok "formatted"; }

cmd_runtime() {
  need_cargo
  say "axon runtime (contract + config smoke check)"
  cargo run --quiet --bin axon
}

cmd_book() {
  need_cargo
  local coin="${1:-BTC}"
  say "Live Hyperliquid order book: $coin"
  info "public market data over WebSocket - no key required. Ctrl+C to stop."
  cargo run --quiet -p axon-provider-hyperliquid --example live_book -- "$coin"
}

cmd_wallet() {
  need_cargo; need_env_file
  say "Wallet pre-flight (read-only)"
  bash "$ROOT/scripts/with-env.sh" \
    cargo run --quiet -p axon-provider-hyperliquid --example wallet_info
}

cmd_agent() {
  need_cargo; need_env_file
  say "approveAgent ceremony"
  info "DRY RUN unless you pass --submit. Generates a FRESH agent key locally;"
  info "the secret is written to secrets/ at mode 0600, never printed by default."
  bash "$ROOT/scripts/with-env.sh" \
    cargo run --quiet -p axon-provider-hyperliquid --example approve_agent -- "$@"
}

cmd_replay() {
  need_cargo
  local log="${1:-crates/axon-replay/testdata/session.jsonl}"
  say "Deterministic replay: $log"
  info "republishes a captured log onto the real bus, through the real CoreHandler fan-out"
  info "(book -> marks -> tracker) and the real signal reader + planner. Signals come from"
  info "<log>.signals.jsonl beside the log when one exists; --no-signals replays data alone."
  cargo run --quiet -p axon-replay --example replay_log -- "$log"
}

cmd_capture() {
  need_cargo
  local log="${1:-data/captures/axon-session.jsonl}"
  say "Recorded session -> $log"
  info "runs the offline session with capture on, then replays the log it wrote through"
  info "the same chain. Signals go beside it as <log>.signals.jsonl. The log only gets"
  info "that name if the recording closed cleanly; otherwise it is left as <log>.partial."
  cargo run --quiet --bin axon -- --capture "$log" || return 1
  cmd_replay "$log"
}

cmd_parity() {
  need_cargo
  say "Cross-language model-parity gate (ADR-0021)"
  info "frozen Python answers in crates/axon-model/tests/bundles, scored by the Rust backends."
  cargo test -p axon-model --test cross_language_parity
}

cmd_parity_bundles() {
  need_venv
  say "Regenerating the committed parity bundles"
  warn "This rewrites frozen references. The git diff IS the review (ADR-0021)."
  python crates/axon-model/tests/bundles/generate.py
}

cmd_feature_parity() {
  need_cargo
  say "Cross-language FEATURE-parity gate (ADR-0035)"
  info "the other half of Boundary A: whether the two languages compute the same"
  info "feature vectors from the same market data, not whether the same vectors score alike."
  info "Held to bit equality — every transform is + - * / sqrt, comparison and log."
  cargo test -p axon-features || return 1
  # And then print what it measured. The tests assert; an assertion that passes says
  # nothing, and a gate whose numbers nobody can see is a gate nobody can tell from
  # one that quietly stopped running on anything.
  say "What the gate actually compared"
  cargo run -q -p axon-features --example feature_gate
}

cmd_feature_bundles() {
  need_venv
  say "Regenerating the committed feature-parity bundles"
  warn "This rewrites frozen references. The git diff IS the review (ADR-0035)."
  python crates/axon-features/tests/bundles/generate.py
  # The numeric fixture is regenerated in the same breath, because it pins NumPy's
  # own summation order and the two would otherwise be refreshed against different
  # builds of NumPy — which is precisely the drift both of them exist to detect.
  python crates/axon-features/tests/fixtures/generate.py
}

cmd_strategy() {
  need_venv
  local ring="${AXON_RING:-/dev/shm/axon-demo.ring}"
  say "Reference Python strategy -> signal ring"
  info "ring: $ring  (Rust side: the planner in axon-strategy consumes these)"
  python -m axon.live --ring "$ring" "$@"
}

cmd_portfolio_evidence() {
  need_venv
  # What a book of several strategies over several coins would actually have held, so
  # `[portfolio]`'s bounds are quantiles rather than declarations (ADR-0038 §6).
  #
  # **Not part of the gate**, and the reason is the same one `parity-bundles` gives for
  # being separate: it reads the cached corpus under `data/`, which is gitignored, so a
  # test that silently skipped when the corpus was absent would be indistinguishable from
  # a passing one. It is also minutes of CPU, and the gate has to stay fast.
  local registry="${AXON_ZOO_REGISTRY:-data/models-p6-live}"
  say "Portfolio evidence -> what a multi-strategy book would have held"
  info "corpus: data/candles-testnet   registry: $registry"
  info "sizes are the live config's, not the strategies' defaults - a bound argued from"
  info "the wrong one is argued about a session nobody runs"
  python -m axon.strategies.portfolio_evidence \
    --registry "axon.strategies.zoo:live_strategy=$registry" \
    --max-position "BTC=0.0003,ETH=0.006,SOL=0.06" \
    "$@"
}

cmd_live() {
  need_cargo; need_env_file
  say "Live testnet tests"
  warn "This PLACES AND CANCELS REAL ORDERS on Hyperliquid testnet using the key in .env."
  warn "Testnet is play money, but it is a real venue and a real signed order."
  if [[ "${AXON_YES:-}" != "1" ]]; then
    read -r -p "    Type 'yes' to continue: " reply
    [[ "$reply" == "yes" ]] || { info "aborted"; return 1; }
  fi
  bash "$ROOT/scripts/with-env.sh" \
    cargo test -p axon-provider-hyperliquid -- --ignored --nocapture
}

cmd_flatten() {
  need_cargo; need_env_file
  local cfg="${1:-}"
  [[ -n "$cfg" ]] || die "flatten needs a session config.
    It reads [venue].account and [strategy].symbols from it, so the pass cannot guess
    which account or which instruments an operator means:
      ./run.sh flatten scripts/sessions/live-ml-testnet-m1.toml"
  [[ -f "$cfg" ]] || die "no such config: $cfg"
  say "Flatten: adopt the venue's position and go flat"
  warn "This PLACES REAL ORDERS on the venue named in $cfg."
  warn "Every one is reduce-only and sized from the venue's own position read, so it"
  warn "cannot overshoot - but it is a real signed order and it closes real exposure."
  warn "It does NOT cancel resting orders: cancel-all on Hyperliquid is account-wide and"
  warn "would sweep a running session's quotes. Stop the session first."
  if [[ "${AXON_YES:-}" != "1" ]]; then
    read -r -p "    Type 'flatten' to continue: " reply
    [[ "$reply" == "flatten" ]] || { info "aborted"; return 1; }
  fi
  cargo build --release -p axon-runtime || return 1
  # Exit code is the answer: 1 means at least one symbol is not flat on the venue's own
  # word, or the venue could not be read. An unknown position is never reported as flat.
  bash "$ROOT/scripts/with-env.sh" "$ROOT/target/release/axon" --config "$cfg" --flatten
}

cmd_clean() {
  need_cargo
  say "Cleaning build artifacts"
  cargo clean
  ok "removed target/ (.venv and .env kept)"
}

cmd_doctor() {
  say "Environment report"
  local missing=0
  report() { # label, command, version-args
    if command -v "$2" >/dev/null 2>&1; then
      ok "$(printf '%-10s %s' "$1" "$("$2" "${@:3}" 2>&1 | head -1)")"
    else
      warn "$(printf '%-10s %s' "$1" "MISSING")"; missing=1
    fi
  }
  report rustc  rustc  --version
  report cargo  cargo  --version
  report rustup rustup --version
  report uv     uv     --version
  report git    git    --version

  if [[ -x "$VENV/bin/python" ]]; then
    ok "$(printf '%-10s %s' "venv" "$("$VENV/bin/python" --version 2>&1)")"
    for pkg in numpy pytest; do
      if "$VENV/bin/python" -c "import $pkg" 2>/dev/null; then
        ok "$(printf '%-10s %s' "  $pkg" "importable")"
      else
        warn "$(printf '%-10s %s' "  $pkg" "MISSING")"; missing=1
      fi
    done
  else
    warn "$(printf '%-10s %s' "venv" "MISSING (.venv)")"; missing=1
  fi

  if [[ -f "$ROOT/.env" ]]; then
    if grep -qE '^AXON_HL_SECRET_KEY=(0x)?0{64}\s*$' "$ROOT/.env"; then
      warn "$(printf '%-10s %s' ".env" "present, key is still the placeholder")"
    else
      ok "$(printf '%-10s %s' ".env" "present, key looks set")"
    fi
    # Never print the key itself. Mode matters: 600 keeps it off other accounts.
    local mode
    mode="$(stat -c '%a' "$ROOT/.env" 2>/dev/null || echo '?')"
    if [[ "$mode" == "600" ]]; then
      ok "$(printf '%-10s %s' "" "mode $mode")"
    else
      # 777 is normal on a /mnt/c NTFS mount, where chmod cannot take effect. On a
      # native filesystem it means the key is world-readable.
      warn "$(printf '%-10s %s' "" "mode $mode - expected 600. Fix: chmod 600 .env")"
    fi
  else
    warn "$(printf '%-10s %s' ".env" "absent (only needed for wallet/live)")"
  fi

  echo
  if [[ $missing -eq 1 ]]; then
    warn "something is missing - run: bash scripts/install-ubuntu.sh"
    return 1
  fi
  ok "environment looks complete"
}

# --------------------------------------------------------------------------- #
main() {
  local cmd="${1:-check}"
  shift || true
  case "$cmd" in
    check)          cmd_check ;;
    test)           cmd_test ;;
    build)          cmd_build ;;
    release)        cmd_release ;;
    fmt)            cmd_fmt ;;
    runtime)        cmd_runtime ;;
    book)           cmd_book "$@" ;;
    wallet)         cmd_wallet ;;
    agent)          cmd_agent "$@" ;;
    replay)         cmd_replay "$@" ;;
    capture)        cmd_capture "$@" ;;
    parity)         cmd_parity ;;
    parity-bundles) cmd_parity_bundles ;;
    feature-parity) cmd_feature_parity ;;
    feature-bundles) cmd_feature_bundles ;;
    strategy)       cmd_strategy "$@" ;;
    portfolio-evidence) cmd_portfolio_evidence "$@" ;;
    live)           cmd_live ;;
    flatten)        cmd_flatten "$@" ;;
    clean)          cmd_clean ;;
    doctor)         cmd_doctor ;;
    help|-h|--help) usage ;;
    *)              printf '%s\n\n' "${RED}unknown command: $cmd${RESET}" >&2; usage; exit 2 ;;
  esac
}

main "$@"
