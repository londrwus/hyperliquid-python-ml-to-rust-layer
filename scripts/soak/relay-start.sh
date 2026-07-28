#!/usr/bin/env bash
# Start (or restart) the soak's loopback WS relay, recording its pid.
#
# By pid file rather than `pkill -f`: the pattern that matches the relay also matches
# the shell that is running the restart, so `pkill -f ws-relay.py` kills the caller.
# Learned the hard way, mid-soak.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PIDFILE=/dev/shm/m7-relay.pid
CTL=/dev/shm/m7-relay.ctl
JOURNAL=/dev/shm/m7-relay.jsonl

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" 2>/dev/null
  sleep 1
fi

echo open > "$CTL"
: > "$JOURNAL"
nohup .venv/bin/python scripts/soak/ws-relay.py \
  --port "${1:-8765}" \
  --upstream api.hyperliquid-testnet.xyz:443 \
  --control "$CTL" --journal "$JOURNAL" > /dev/shm/m7-relay.out 2>&1 &
echo $! > "$PIDFILE"
sleep 1
echo "relay pid $(cat "$PIDFILE") on port ${1:-8765}"
