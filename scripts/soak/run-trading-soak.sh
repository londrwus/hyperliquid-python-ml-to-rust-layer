#!/usr/bin/env bash
# Start the M8 soak: a session that **trades**, through induced outages, holding a position.
#
# PLACES REAL ORDERS on Hyperliquid testnet, for hours, unattended. That is the point and
# it is also the reason this is a separate script from `run-soak.sh` rather than a flag on
# it: the read-only soak's own header says a session that cannot place an order cannot
# leave one resting after hours, and this one can. Everything that stands between that and
# an abandoned position is named in `soak-testnet-trading.toml`.
#
# Five processes, and the order is not arbitrary:
#
#   1. the loopback **relay**, because the session dials it and it has to be listening;
#   2. the Rust **session**, which creates the md/bar rings and the beacon;
#   3. the Python **producer**, which creates the signal ring — the Rust side retries its
#      attach, so producer-after-session is the supported order and not a race;
#   4. the resource **monitor**;
#   5. the **outage plan**, last and after a lead-in.
#
# The lead-in is load-bearing twice over. The session needs its cold start — the
# `userFills` snapshot, the startup reconcile, the first dead-man's-switch arm — or the
# first break lands on a session that had not finished coming up. And the *strategy* needs
# its warmup: `zoo_xgboost` on m1 warms in 21 bars, so a plan that starts before that is a
# plan whose early breaks land on a session holding nothing, which is the read-only soak's
# subject and not this one's. **Default 1 800 s** (30 min) for exactly that reason —
# `run-soak.sh` defaults to 180 s and would be wrong here.
#
#   scripts/soak/run-trading-soak.sh [tag] [lead_in_seconds] [duration_seconds]
#
# Stop it with `scripts/soak/stop-trading-soak.sh <tag>`, which flattens. Killing the
# processes by hand leaves the position to the dead-man's switch, which is protection and
# not a plan.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-m8-soak}"
LEAD_IN="${2:-1800}"
DURATION="${3:-28800}"   # 8 hours
CONFIG=scripts/soak/soak-testnet-trading.toml
RUN=/dev/shm/$TAG
CAPTURE="data/captures/$TAG-trading.jsonl"
MD_RING=/dev/shm/m8-soak-md.ring
SIGNAL_RING=/dev/shm/m8-soak-signal.ring

BOLD=$'\033[1m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
die() { printf '%sERROR: %s%s\n' "$RED" "$1" "$RESET" >&2; exit 1; }
warn() { printf '%s%s%s\n' "$YELLOW" "$1" "$RESET"; }

[[ -f .env ]] || die "no .env — this run signs real orders. See .env.example"
[[ -f "$CONFIG" ]] || die "no $CONFIG"
[[ -x ./target/release/axon ]] || die "build first: cargo build --release -p axon-runtime"

# The account must be flat before a soak starts, and this is not a formality: a session
# that inherits a position books the whole realized P&L of closing it, including the part
# that accrued before it existed (ADR-0036, consequences) — so the money view's drift
# figure and the loss bound would both be measuring somebody else's trade for hours.
warn "This soak PLACES REAL ORDERS on testnet for ${DURATION}s, unattended."
warn "The account must be FLAT before it starts: ./run.sh flatten $CONFIG"
warn "Outages begin after ${LEAD_IN}s, which is the session's cold start plus the"
warn "strategy's 21-bar warmup. Breaks before that land on a session holding nothing."
if [[ "${AXON_YES:-}" != "1" ]]; then
  read -r -p "    Type 'soak' to continue: " reply
  [[ "$reply" == "soak" ]] || { echo "aborted"; exit 1; }
fi

mkdir -p data/captures data/soak
rm -f "$CAPTURE" "$CAPTURE.partial" "${CAPTURE%.jsonl}.signals.jsonl" \
      "${CAPTURE%.jsonl}.signals.jsonl.partial" "$SIGNAL_RING"

bash scripts/soak/relay-start.sh 8765 || exit 1

# ── the session ──────────────────────────────────────────────────────────────
: > "$RUN.log"
( bash scripts/with-env.sh ./target/release/axon \
    --config "$CONFIG" --capture "$CAPTURE" \
    >> "$RUN.log" 2>&1 & echo $! > "$RUN.pid" )
sleep 5
PID="$(cat "$RUN.pid")"
kill -0 "$PID" 2>/dev/null || { echo "session died at startup:"; tail -30 "$RUN.log"; exit 1; }
date -u +'session started %Y-%m-%dT%H:%M:%SZ' | tee "$RUN.started"

# ── the producer ─────────────────────────────────────────────────────────────
# Started after the session so the bar ring exists and has history in it: nothing advances
# the ring's tail until a consumer attaches, so the strategy warms its 21-bar window from
# whatever accumulated in the gap — which on m1 is 21 minutes it does not have to wait.
#
# `--parity-diff` is the reason this soak can claim something the 59-minute run could not:
# one bar-ring reader, dispatched to the strategy *and* to the diff, so parity is watched
# **as it trades** rather than reconstructed from the capture afterwards (ADR-0037 §5).
: > "$RUN.producer.log"
( PYTHONPATH="$ROOT/python" nohup "$ROOT/.venv/bin/python" -m axon.strategies.live_runner \
    --md-ring "$MD_RING" --signal-ring "$SIGNAL_RING" \
    --symbol-id 3 --strategy axon.strategies.zoo:live_strategy \
    --registry data/models-p6-live --model zoo_xgboost \
    --max-position 0.0003 --duration "$DURATION" \
    --parity-diff \
    --flatten-on-exit --flatten-urgency take --flatten-wait 60 \
    --transcript "$RUN.transcript.jsonl" \
    >> "$RUN.producer.log" 2>&1 & echo $! > "$RUN.producer.pid" )
sleep 5
PPID_="$(cat "$RUN.producer.pid")"
kill -0 "$PPID_" 2>/dev/null || {
  echo "producer died at startup:"; tail -30 "$RUN.producer.log"
  echo "stopping the session so it does not run with nothing driving it"
  kill -TERM "$PID"; exit 1
}

# ── the monitor and the outage plan ──────────────────────────────────────────
# Both pids are watched: a producer that dies silently leaves a session holding whatever it
# last asked for, and the sweeper plus the bounded re-quote are what bound that — but an
# eight-hour run whose strategy died in hour two is eight hours of soaking the wrong thing.
nohup .venv/bin/python scripts/soak/monitor.py --pid "$PID" --out "$RUN.monitor.csv" \
  --every 15 --watch "$CAPTURE.partial" \
  --watch "${CAPTURE%.jsonl}.signals.jsonl.partial" \
  > /dev/null 2>&1 &
echo $! > "$RUN.monitor.pid"

nohup bash -c "sleep $LEAD_IN; exec .venv/bin/python scripts/soak/outages.py \
  --control /dev/shm/m7-relay.ctl --plan scripts/soak/plan-long.txt \
  --journal $RUN.outages.jsonl" > /dev/null 2>&1 &
echo $! > "$RUN.outages.pid"

cat <<EOF

${BOLD}M8 trading soak started${RESET}
  session   pid $PID        log $RUN.log
  producer  pid $PPID_      log $RUN.producer.log
  capture   $CAPTURE
  transcript $RUN.transcript.jsonl
  outages   begin in ${LEAD_IN}s, journal $RUN.outages.jsonl

  watch:  tail -f $RUN.log | grep -E 'LOSS LIMIT|UNQUOTED|STRANDED|PARITY|UNPROTECTED|HALTED'
  stop:   scripts/soak/stop-trading-soak.sh $TAG
EOF
