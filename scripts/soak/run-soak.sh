#!/usr/bin/env bash
# Start the M7 soak: relay, session, resource monitor, and the scripted outage plan.
#
# Everything is started as a detached background process with its pid recorded, so the
# run survives the shell that launched it and can be observed, frozen (SIGSTOP) and
# stopped (SIGTERM) by anyone with the pid file. A soak whose only handle is a foreground
# terminal is a soak that ends when the terminal does.
#
# The order is not arbitrary: the relay has to be listening before the session dials it,
# and the outage plan must not start until the session has had its cold start — the
# `userFills` snapshot, the startup reconcile and the first dead-man's-switch arm — or
# the first break would land on a session that had not finished coming up, and every
# later observation would be measured against a baseline that never existed.
#
#   scripts/soak/run-soak.sh [tag] [lead_in_seconds]
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-m7-soak}"
LEAD_IN="${2:-180}"
RUN=/dev/shm/$TAG
CAPTURE="data/captures/$TAG.jsonl"

mkdir -p data/captures
rm -f "$CAPTURE" "$CAPTURE.partial" "${CAPTURE%.jsonl}.signals.jsonl" \
      "${CAPTURE%.jsonl}.signals.jsonl.partial"

bash scripts/soak/relay-start.sh 8765 || exit 1

: > "$RUN.log"
( bash scripts/with-env.sh ./target/debug/axon \
    --config scripts/soak/soak-testnet.toml --capture "$CAPTURE" \
    >> "$RUN.log" 2>&1 & echo $! > "$RUN.pid" )
sleep 3
PID="$(cat "$RUN.pid")"
kill -0 "$PID" 2>/dev/null || { echo "session died at startup:"; tail -20 "$RUN.log"; exit 1; }
date -u +'session started %Y-%m-%dT%H:%M:%SZ' | tee "$RUN.started"
echo "$PID" > "$RUN.pid"

nohup .venv/bin/python scripts/soak/monitor.py --pid "$PID" --out "$RUN.monitor.csv" \
  --every 15 --watch "$CAPTURE.partial" \
  --watch "${CAPTURE%.jsonl}.signals.jsonl.partial" \
  > /dev/null 2>&1 &
echo $! > "$RUN.monitor.pid"

nohup bash -c "sleep $LEAD_IN; exec .venv/bin/python scripts/soak/outages.py \
  --control /dev/shm/m7-relay.ctl --plan scripts/soak/plan-long.txt \
  --journal $RUN.outages.jsonl" > /dev/null 2>&1 &
echo $! > "$RUN.outages.pid"

echo "session pid $PID  log $RUN.log  capture $CAPTURE"
echo "outage plan starts in ${LEAD_IN}s"
