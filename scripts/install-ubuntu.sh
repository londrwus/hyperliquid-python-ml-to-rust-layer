#!/usr/bin/env bash
#
# install-ubuntu.sh - one-shot setup for Axon on a native Ubuntu box (24.04 LTS
# target; 22.04 and newer work too). After this, `./run.sh` works.
#
# Usage:
#   bash scripts/install-ubuntu.sh              # install everything, then run the gate
#   bash scripts/install-ubuntu.sh --no-gate    # install only, skip the test run
#   bash scripts/install-ubuntu.sh --dry-run    # print what would happen, change nothing
#
# Safe to re-run: every step checks first and skips work already done.
#
# Why each piece is here (Ubuntu 24.04 specifics that bite otherwise):
#   - Ubuntu's apt `rustc` is far older than this workspace's 1.85 minimum, and if it
#     is installed it sits on PATH ahead of rustup. We install via rustup and then
#     *verify the version actually on PATH*, because "cargo exists" is not the same
#     as "cargo is new enough".
#   - 24.04 enforces PEP 668, so `pip install` into the system Python fails with
#     externally-managed-environment. Everything Python goes in a venv.
#   - `python3 -m venv` needs the separate python3-venv package on Debian/Ubuntu.
#   - No libssl-dev: this workspace uses rustls end to end (reqwest, tokio-tungstenite
#     and alloy's crypto are all pure Rust), so there is no OpenSSL build dependency.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
say()  { printf '%s\n' "${CYAN}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s%s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '    %s%s%s\n' "$YELLOW" "$*" "$RESET"; }
skip() { printf '    %s%s%s\n' "$DIM" "$*" "$RESET"; }
die()  { printf '%s\n' "${RED}ERROR: $*${RESET}" >&2; exit 1; }
# Past-tense confirmations must stay silent under --dry-run: printing "installed"
# for something that was only described is exactly the kind of false report that
# makes a dry run useless.
done_msg() { [[ $DRY_RUN -eq 0 ]] && ok "$*"; return 0; }

DRY_RUN=0
RUN_GATE=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-gate) RUN_GATE=0 ;;
    -h|--help)
      # Derived from the leading comment block, not a line range, so help cannot
      # drift from the documentation above.
      awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
        "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) die "unknown option: $arg (try --help)" ;;
  esac
done

# Run a command, or just describe it under --dry-run.
run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '    %s$ %s%s\n' "$DIM" "$*" "$RESET"
  else
    "$@"
  fi
}

MIN_RUST_MINOR=85   # workspace.package.rust-version = 1.85

# --------------------------------------------------------------------------- #
# 0. sanity: not root, right OS, in the right directory
# --------------------------------------------------------------------------- #
say "Checking the environment"

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  die "do not run this as root or with sudo.
    It would leave root-owned files in your home and repo (~/.cargo, ~/.local, .venv,
    target/) that your normal user cannot write, and the next plain \`./run.sh\` would
    fail with permission errors. Run it as yourself; it will call sudo only for apt."
fi

[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/crates" ]] \
  || die "this does not look like the Axon repo root (no Cargo.toml + crates/): $ROOT"

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  info "OS: ${PRETTY_NAME:-unknown}"
  if [[ "${ID:-}" != "ubuntu" ]]; then
    warn "built and tested for Ubuntu; '${ID:-unknown}' may work but is unverified"
  elif [[ "${VERSION_ID:-}" != "24.04" ]]; then
    warn "targeted at Ubuntu 24.04; you are on ${VERSION_ID:-unknown} (should still work)"
  else
    ok "Ubuntu 24.04 - the target platform"
  fi
else
  warn "no /etc/os-release; assuming a Debian-family system"
fi

ARCH="$(uname -m)"
info "arch: $ARCH"
if [[ "$ARCH" != "x86_64" ]]; then
  warn "the IPC ring assumes x86_64 little-endian (ADR-0006); '$ARCH' is untested"
fi

# --------------------------------------------------------------------------- #
# 1. apt packages
# --------------------------------------------------------------------------- #
say "System packages"

APT_WANTED=(build-essential pkg-config curl git ca-certificates python3-venv)
APT_MISSING=()
for pkg in "${APT_WANTED[@]}"; do
  if dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    skip "$pkg already installed"
  else
    APT_MISSING+=("$pkg")
  fi
done

if [[ ${#APT_MISSING[@]} -eq 0 ]]; then
  ok "all system packages present"
else
  info "need: ${APT_MISSING[*]}"
  if ! command -v sudo >/dev/null 2>&1; then
    die "sudo not found. Install these as root, then re-run:
    apt-get update && apt-get install -y ${APT_MISSING[*]}"
  fi
  # A single sudo prompt up front reads better than one per command.
  if [[ $DRY_RUN -eq 0 ]] && ! sudo -n true 2>/dev/null; then
    info "sudo password needed for apt (this is the only privileged step)"
  fi
  run sudo apt-get update
  run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_MISSING[@]}"
  done_msg "installed: ${APT_MISSING[*]}"
fi

# apt's rustc, if present, shadows rustup and is too old for this workspace.
if dpkg-query -W -f='${Status}' rustc 2>/dev/null | grep -q "install ok installed"; then
  warn "the apt 'rustc' package is installed. Ubuntu ships a version older than the"
  warn "1.$MIN_RUST_MINOR this workspace requires, and /usr/bin/cargo may shadow rustup's."
  warn "If the version check below fails, remove it: sudo apt-get remove rustc cargo"
fi

# --------------------------------------------------------------------------- #
# 2. Rust toolchain (rustup)
# --------------------------------------------------------------------------- #
say "Rust toolchain"

export PATH="$HOME/.cargo/bin:$PATH"

if command -v rustup >/dev/null 2>&1; then
  skip "rustup already installed"
else
  info "installing rustup (stable, non-interactive)"
  # rust-toolchain.toml pins the channel and the rustfmt/clippy components, so
  # rustup materializes the right toolchain on the first cargo call in this repo.
  run bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal"
  done_msg "rustup installed"
fi

if [[ $DRY_RUN -eq 0 ]]; then
  command -v cargo >/dev/null 2>&1 \
    || die "cargo still not on PATH after installing rustup.
    Open a new shell, or run: source \"\$HOME/.cargo/env\""

  # The pinned toolchain + components. Idempotent; a no-op once present.
  rustup toolchain install stable --component rustfmt clippy --profile minimal >/dev/null 2>&1 || true

  RUSTC_V="$(rustc --version | awk '{print $2}')"
  RUSTC_MINOR="${RUSTC_V#*.}"; RUSTC_MINOR="${RUSTC_MINOR%%.*}"
  RUSTC_MAJOR="${RUSTC_V%%.*}"
  info "rustc $RUSTC_V  ($(command -v rustc))"
  if (( RUSTC_MAJOR < 1 || (RUSTC_MAJOR == 1 && RUSTC_MINOR < MIN_RUST_MINOR) )); then
    die "rustc $RUSTC_V is older than the required 1.$MIN_RUST_MINOR.
    This is almost always apt's rustc shadowing rustup's. Fix with:
      sudo apt-get remove rustc cargo
      export PATH=\"\$HOME/.cargo/bin:\$PATH\"
    then re-run this installer."
  fi
  ok "rustc $RUSTC_V satisfies the 1.$MIN_RUST_MINOR minimum"
fi

# --------------------------------------------------------------------------- #
# 3. Python 3.12 venv
# --------------------------------------------------------------------------- #
say "Python environment"

VENV="$ROOT/.venv"   # repo-local and gitignored, so run.sh needs no env vars
export PATH="$HOME/.local/bin:$PATH"

# uv is the project's documented way to pin Python (docs/DEVELOPMENT.md). Ubuntu
# 24.04 happens to ship exactly the 3.12 we want, so system python3 is a perfectly
# good fallback when uv cannot be fetched - no reason to hard-fail on a download.
PY_TOOL=""
if command -v uv >/dev/null 2>&1; then
  skip "uv already installed"
  PY_TOOL="uv"
else
  info "installing uv"
  if [[ $DRY_RUN -eq 1 ]]; then
    run bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    PY_TOOL="uv"
  elif curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 && PY_TOOL="uv" || PY_TOOL=""
    [[ -n "$PY_TOOL" ]] && ok "uv installed"
  fi
  if [[ -z "$PY_TOOL" ]]; then
    warn "could not install uv; falling back to the system python3"
    PY_TOOL="system"
  fi
fi

if [[ $DRY_RUN -eq 0 && "$PY_TOOL" == "system" ]]; then
  command -v python3 >/dev/null 2>&1 || die "no python3 and no uv - cannot build a venv"
  SYS_PY="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  info "system python3 is $SYS_PY"
  # pyproject: requires-python = ">=3.11,<3.14"
  python3 - <<'EOF' || die "system python3 is outside the supported range >=3.11,<3.14.
    Install uv (https://astral.sh/uv) or a 3.12 from deadsnakes, then re-run."
import sys
raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)
EOF
  ok "python $SYS_PY is within >=3.11,<3.14"
fi

if [[ -x "$VENV/bin/python" ]]; then
  skip "venv already exists at .venv"
else
  info "creating .venv"
  if [[ "$PY_TOOL" == "uv" ]]; then
    run uv python install 3.12
    run uv venv --python 3.12 "$VENV"
  else
    run python3 -m venv "$VENV"
  fi
  done_msg "venv created"
fi

info "installing numpy + pytest into .venv"
if [[ "$PY_TOOL" == "uv" ]]; then
  run uv pip install --python "$VENV/bin/python" numpy pytest
else
  run "$VENV/bin/python" -m pip install --upgrade pip
  run "$VENV/bin/python" -m pip install numpy pytest
fi
done_msg "Python dependencies installed"

# --------------------------------------------------------------------------- #
# 4. secrets scaffold
# --------------------------------------------------------------------------- #
say "Secrets"

if [[ -f "$ROOT/.env" ]]; then
  skip ".env already exists (left untouched)"
else
  info "creating .env from .env.example"
  run cp "$ROOT/.env.example" "$ROOT/.env"
  # Placeholders only - no real key is ever written by this script.
  run chmod 600 "$ROOT/.env"
  done_msg ".env created with placeholders, mode 600"
  warn "fill in AXON_HL_SECRET_KEY before using ./run.sh wallet or ./run.sh live."
  warn "Generate a key locally and never paste it anywhere:  printf '0x%s\\n' \"\$(openssl rand -hex 32)\""
  warn "See .env.example"
fi

# --------------------------------------------------------------------------- #
# 5. prove it works
# --------------------------------------------------------------------------- #
if [[ $RUN_GATE -eq 1 && $DRY_RUN -eq 0 ]]; then
  say "Running the full gate (rustfmt, clippy, Rust tests, Python tests)"
  info "first build compiles every dependency - expect several minutes"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if bash "$ROOT/scripts/check.sh"; then
    ok "gate passed"
  else
    die "the gate failed. The install itself may still be fine - re-run just the
    tests with ./run.sh check to see the failure on its own."
  fi
else
  say "Skipping the gate"
  skip "run it yourself with: ./run.sh check"
fi

# --------------------------------------------------------------------------- #
say "Done"
if [[ $DRY_RUN -eq 1 ]]; then
  info "dry run only - nothing was changed."
  exit 0
fi
cat <<EOF

    ${BOLD}Axon is installed.${RESET} Next:

      ./run.sh              ${DIM}# the full gate (default) - proves everything works${RESET}
      ./run.sh runtime      ${DIM}# the axon binary: contract + config smoke check${RESET}
      ./run.sh book BTC     ${DIM}# live Hyperliquid order book (network, no key needed)${RESET}
      ./run.sh wallet       ${DIM}# read-only key/account pre-flight (needs .env)${RESET}
      ./run.sh help         ${DIM}# everything else${RESET}

    ${DIM}run.sh puts cargo and the venv on PATH itself, so no shell setup is needed.
    For interactive cargo use in a fresh shell: source "\$HOME/.cargo/env"${RESET}
EOF
