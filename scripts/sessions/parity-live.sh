#!/usr/bin/env bash
#
# parity-live.sh - point the live parity monitor at a real Hyperliquid testnet session.
#
# ADR-0030 shipped the monitor and recorded, as a `−` consequence, that "nothing here
# has been run against a live session. The monitor has seen a frozen fixture, a cached
# mainnet history, and synthetic windows. Its silence deadline (60 s) is a starting
# value, not a measurement." This script is the harness that closes the first half of
# that sentence and measures the second. It is NOT the observation: until somebody has
# run `--live` and pasted the transcript into a runbook, nothing here is proven.
#
# Usage:
#   scripts/sessions/parity-live.sh                 # OFFLINE rehearsal. No session, no
#                                                   # socket. This is the default.
#   scripts/sessions/parity-live.sh --rehearse-silence
#                                                   # …plus a ~3 min stage that drives a
#                                                   # real Rust-written beacon past the
#                                                   # silence deadline with no bars
#   scripts/sessions/parity-live.sh --live [MINUTES]
#                                                   # THE LIVE PATH. Starts an `axon`
#                                                   # session against Hyperliquid
#                                                   # TESTNET with the market-data ring
#                                                   # on and --capture on, runs the
#                                                   # monitor against its bar ring, and
#                                                   # stops both. Default 45 minutes.
#
# Why the default is the offline one, and not a convenience: the live path starts a
# process that dials a venue. A harness whose no-argument invocation opens a socket is a
# harness that opens a socket by accident, and this repository's rule is that nothing in
# the default gate touches the network. `--live` is the only word that changes that, and
# it is checked in exactly one place below.
#
# TESTNET ONLY. The key in .env is an approved *agent* wallet and it is public — it was
# pasted into a chat. This session sends nothing signed at all (`intent.enabled = false`,
# dead-man's switch off), so it spends no metered request and places no order; `/info`
# reads are free. Mainnet is refused by the runtime unless AXON_ALLOW_MAINNET=1, and
# nothing here sets it.
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
say()  { printf '%s\n' "${CYAN}${BOLD}==>${RESET} ${BOLD}$*${RESET}"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s%s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '    %s%s%s\n' "$YELLOW" "$*" "$RESET"; }
die()  { printf '%s\n' "${RED}ERROR: $*${RESET}" >&2; exit 1; }

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/python"
# A globally installed pytest/ipython plugin must never change what this prints.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

PY="$ROOT/.venv/bin/python"
CONFIG="$ROOT/scripts/sessions/parity-live-testnet.toml"
DRIVER="$ROOT/scripts/sessions/parity_live.py"

# Every path below is *derived* from the ring path in the TOML, exactly as the runtime
# derives them, so this script cannot drift from the session it starts. Read out of the
# config rather than restated here for the same reason: two descriptions of one fact can
# disagree, and the one that disagrees silently is the one in a shell script.
MD_RING="$(awk -F'"' '/^path = / { print $2; exit }' "$CONFIG")"
BAR_RING="${MD_RING%.ring}.bars.ring"
BEACON="$MD_RING.beacon"
CAPTURE="$(awk -F'"' '/^path = "data/ { print $2; exit }' "$CONFIG")"
RUN=/dev/shm/parity-live
EVIDENCE="$ROOT/data/captures/parity-live-evidence.json"

need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. Run: bash scripts/install-ubuntu.sh"; }

# --------------------------------------------------------------------------- #
# the offline rehearsal — the whole live code path, with no session and no socket
# --------------------------------------------------------------------------- #
rehearse() {
  [[ -x "$PY" ]] || die "no Python venv at .venv (bash scripts/install-ubuntu.sh)"
  say "Offline rehearsal — the same driver, the same monitor, no session and no socket"

  # Stage 1: the committed candle fixture through the real serving path, with a beacon
  # probe pointed at nothing. This is what the harness looks like when Q1's pass-loop
  # wiring is absent: it runs, it compares, and it says in a box that every silence it
  # reports is uncategorised.
  say "1/3  the committed fixture, with NO beacon (the degraded path)"
  "$PY" "$DRIVER" --fixture BTC --interval 1m --window 16 --max-bars 240 \
      --no-beacon --ring /dev/shm/parity-rehearsal.ring || return 1

  # Stage 2: the same run with a beacon **written by the real Rust writer**. `md_writer`
  # is `axon_ipc::MdBeacon` — the same code the runtime's pass loop calls — so this
  # exercises the probe against bytes Rust produced rather than against a Python fake.
  # It ends with a deliberate `stop()`, so the publisher reads as STOPPED rather than
  # DEAD, which is the distinction only the publisher can make.
  say "2/3  the same run against a beacon the REAL Rust writer produced"
  if cargo build --quiet -p axon-ipc --examples 2>/dev/null \
     && [[ -x "$ROOT/target/debug/examples/md_writer" ]]; then
    rm -f /dev/shm/parity-rehearsal-md.ring* /dev/shm/parity-rehearsal-md.bars.ring
    "$ROOT/target/debug/examples/md_writer" /dev/shm/parity-rehearsal-md.ring 8 16 300 \
      | sed 's/^/    /'
    "$PY" "$DRIVER" --fixture BTC --interval 1m --window 16 --max-bars 240 \
        --beacon /dev/shm/parity-rehearsal-md.ring.beacon \
        --ring /dev/shm/parity-rehearsal.ring || return 1
  else
    warn "could not build the axon-ipc examples, so the beacon-wired path was NOT"
    warn "rehearsed. That is a gap in this rehearsal, not a fact about the harness."
    warn "A brand-new crate mid-write elsewhere in the workspace can cause it;"
    warn "'cargo build -p axon-ipc --examples' on its own is the thing to retry."
  fi

  # Stage 3 (opt-in, ~3 minutes): the silence path itself. A bar ring nobody is writing
  # to, past the shadow trader's bar deadline, with a real beacon beside it — which is
  # the only offline arrangement in which the monitor is genuinely idle. It is opt-in
  # because the deadline for m1 is 2.5 intervals and there is no shorter interval to
  # borrow: the wait is the measurement.
  if [[ "$REHEARSE_SILENCE" == "1" ]]; then
    say "3/3  the SILENCE path: an idle bar ring, a real beacon, past the deadline (~3 min)"
    if [[ -x "$ROOT/target/debug/examples/md_bar_writer" ]]; then
      rm -f /dev/shm/parity-rehearsal-md.ring* /dev/shm/parity-rehearsal-md.bars.ring
      "$ROOT/target/debug/examples/md_writer" /dev/shm/parity-rehearsal-md.ring 8 16 300 >/dev/null
      "$ROOT/target/debug/examples/md_bar_writer" /dev/shm/parity-rehearsal-md.bars.ring 4 16 >/dev/null
      info "the deadline is 2.5 x 60 s; nothing will print for about 150 seconds, and"
      info "that silence is the thing being rehearsed rather than a hung script."
      "$PY" "$DRIVER" --bar-ring /dev/shm/parity-rehearsal-md.ring --symbol-id 0 \
          --interval 1m --window 4 --idle-timeout 175 \
          --ring /dev/shm/parity-rehearsal.ring
    else
      warn "no md_bar_writer example built; the silence stage needs one."
    fi
  else
    say "3/3  skipped. Add --rehearse-silence for the ~3-minute idle-ring stage."
  fi

  echo
  ok "Rehearsal done. NOTHING above was run against a venue, so nothing above is"
  ok "evidence about a live session — it is evidence that the harness runs."
  info "Next: scripts/sessions/parity-live.sh --live 45"
}

# --------------------------------------------------------------------------- #
# the live path
# --------------------------------------------------------------------------- #
live() {
  local minutes="${1:-45}"
  need cargo
  [[ -x "$PY" ]] || die "no Python venv at .venv (bash scripts/install-ubuntu.sh)"
  [[ -f "$ROOT/.env" ]] || die "no .env in the repo root (see .env.example)"

  say "LIVE: Hyperliquid TESTNET, read-only, ${minutes} minutes"
  info "config  : $CONFIG"
  info "rings   : $MD_RING  ->  bars $BAR_RING  ->  beacon $BEACON"
  info "capture : $CAPTURE"
  warn "This session sends NOTHING signed: intents off, dead-man's switch off, and"
  warn "/info reads are free. It places no order and spends no metered request."
  if [[ "${AXON_YES:-}" != "1" ]]; then
    read -r -p "    Type 'yes' to start a live testnet session: " reply
    [[ "$reply" == "yes" ]] || { info "aborted"; return 1; }
  fi

  # Built before the session is started rather than by `cargo run`, so the wait loop
  # below is waiting for a ring and not for a compiler. A three-minute first build with
  # a 60-second ring timeout on the other side of it is a harness that reports a dead
  # publisher on its first ever run.
  say "Building the runtime first (so the wait below is waiting for a ring, not rustc)"
  cargo build --quiet --bin axon || die "the runtime did not build. If it failed inside a
    crate you did not expect, say so rather than working around it."

  mkdir -p "$ROOT/data/captures"
  # A stale ring from a previous run is worse than no ring: the harness would attach to
  # it, read whatever it still held, and report bars that this session never published.
  rm -f "$MD_RING" "$BAR_RING" "$BEACON" "$CAPTURE" "$CAPTURE.partial" "$RUN.log"

  # The Rust session FIRST. It creates all three files, and nothing advances the bar
  # ring's tail until a consumer attaches — so a monitor started a minute later replays
  # every bar since startup and warms its 25-bar window out of history.
  say "Starting the session"
  ( bash "$ROOT/scripts/with-env.sh" "$ROOT/target/debug/axon" \
      --config "$CONFIG" --capture "$CAPTURE" >> "$RUN.log" 2>&1 & echo $! > "$RUN.pid" )
  local pid; pid="$(cat "$RUN.pid")"
  # shellcheck disable=SC2064
  trap "stop_session $pid" EXIT INT TERM

  local waited=0
  while [[ ! -e "$BAR_RING" ]]; do
    kill -0 "$pid" 2>/dev/null || { tail -30 "$RUN.log"; die "the session died at startup"; }
    sleep 1; waited=$((waited + 1))
    [[ $waited -lt 60 ]] || { tail -30 "$RUN.log"; die "no bar ring after 60 s"; }
  done
  ok "session pid $pid, bar ring up after ${waited}s. Log: $RUN.log"
  if [[ -e "$BEACON" ]]; then
    ok "beacon present at $BEACON — silence in this run will resolve into a cause"
  else
    warn "NO BEACON at $BEACON. The runtime created the rings and is not beating the"
    warn "sidecar, so every SILENT verdict in this run will be UNCATEGORISED. That is"
    warn "workstream Q1 (the pass-loop wiring); the run is still worth doing."
  fi

  # --symbol-id is the VENUE's asset index and the bar ring's records carry it. **On
  # testnet BTC is 3, not 0**, and that is the one index this repository has established;
  # ETH's is deliberately not guessed here. It does not need to be: the session
  # subscribes to both coins, `RingBarSource` is attached unfiltered, and a shadow
  # trader arms its bar deadline on ANY instrument's bar — which is the whole reason two
  # coins are configured, since Hyperliquid sends no candle frame at all for a minute
  # with no trades and a single-coin run cannot tell an empty minute from a stall. The
  # run's `arrivals` line then *reports* every id that actually came off the ring, so
  # ETH's index is learned from the venue rather than assumed here.
  say "Running the monitor for ${minutes} minutes"
  "$PY" -u "$DRIVER" \
      --bar-ring "$MD_RING" \
      --symbol-id 3 \
      --interval 1m --window 16 \
      --duration "$((minutes * 60))" \
      --idle-timeout 600 \
      --attest-venue \
      --evidence-out "$EVIDENCE" \
      --ring /dev/shm/parity-live-shadow.ring \
      2>&1 | tee "$RUN.monitor.log"
  local rc=${PIPESTATUS[0]}

  echo
  say "Artefacts"
  info "session log : $RUN.log"
  info "monitor log : $RUN.monitor.log"
  info "capture     : $CAPTURE  (replay: ./run.sh replay $CAPTURE)"
  info "evidence    : $EVIDENCE"
  info "See docs/adr/0030-live-parity-monitor-and-the-coverage-denominator.md for what this run can and cannot say."
  return $rc
}

stop_session() {
  local pid="$1"
  trap - EXIT INT TERM
  if kill -0 "$pid" 2>/dev/null; then
    # SIGINT and not SIGKILL: the runtime's shutdown path is what writes the beacon's
    # final beat with the STOPPED flag, which is the difference between "the session
    # ended" and "the session died" for anyone reading the sidecar afterwards. It is
    # also what closes the capture cleanly — an interrupted one is left as `.partial`.
    say "Stopping the session (SIGINT, so it can mark its beacon stopped)"
    kill -INT "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { warn "still up after 20 s; SIGTERM"; kill -TERM "$pid"; }
  fi
}

# --------------------------------------------------------------------------- #
REHEARSE_SILENCE=0
case "${1:-}" in
  --live)             shift; live "$@" ;;
  --rehearse-silence) REHEARSE_SILENCE=1; rehearse ;;
  ""|--rehearse)      rehearse ;;
  -h|--help)          awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                        "${BASH_SOURCE[0]}" ;;
  *)                  die "unknown argument ${1:?}. Try --help." ;;
esac
