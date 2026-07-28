#!/usr/bin/env bash
# Stop the M8 trading soak, and leave the account flat.
#
# The order is the whole content of this script, and it is the reverse of the start order
# for a reason at every step:
#
#   1. **The outage plan first.** A break landing while the producer is trying to flatten
#      is a flatten with no market data, and the planner correctly refuses to price an
#      order against a book it cannot see. The relay is restored to passthrough, not
#      killed, because the flatten below needs the socket the session is already on.
#   2. **Then the producer**, with SIGTERM rather than SIGKILL. `live_runner` flags the
#      signal and unwinds on its own loop — `--flatten-on-exit` emits its target of zero
#      and waits — because unwinding a ring and a transcript from inside a signal handler
#      is how a run's last records get lost.
#   3. **Then the session**, also SIGTERM, so `graceful_shutdown` runs: stop intents,
#      sweep, and only then decide whether the venue-side deadline should stand.
#   4. **Then `axon --flatten`, unconditionally**, and this is the step that matters.
#      Steps 2 and 3 are *requests*: the producer asks for a target of zero, the session
#      asks the venue to cancel. Whether the account is flat is an observation, and on
#      2026-07-27 the requested version failed in three separate ways. This is the one
#      that reads the venue's own position and sizes a reduce-only close from it.
#   5. **Then read `clearinghouseState` twice**, because that is what the Phase-6 runbook
#      did and it is why it could claim the account was left as it was found.
#
# The monitor is killed last of the background processes: it is the only thing recording
# what the shutdown itself cost.
#
#   scripts/soak/stop-trading-soak.sh [tag]
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="${1:-m8-soak}"
RUN=/dev/shm/$TAG
CONFIG=scripts/soak/soak-testnet-trading.toml

say() { printf '\033[36m==>\033[0m %s\n' "$1"; }

stop_pid() {
  local what="$1" file="$2" grace="${3:-20}"
  [[ -f "$file" ]] || { say "$what: no pid file"; return 0; }
  local pid; pid="$(cat "$file")"
  if ! kill -0 "$pid" 2>/dev/null; then say "$what ($pid): already gone"; return 0; fi
  say "$what ($pid): SIGTERM, waiting up to ${grace}s"
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq "$grace"); do kill -0 "$pid" 2>/dev/null || { say "$what: exited"; return 0; }; sleep 1; done
  # Reported rather than escalated to SIGKILL. A process that ignored SIGTERM for its whole
  # grace window is a process whose state nobody here understands, and killing it is how a
  # capture becomes a prefix and a position becomes the dead-man's switch's problem.
  say "$what: STILL RUNNING after ${grace}s — investigate before killing it"
  return 1
}

# The control fifo is the *relay's* and it is named for the soak that introduced it
# (`m7-relay.ctl`), not for this one. One relay, one control path: two names would be two
# relays, and the second one would not be the one the session is dialled into.
say "restoring the relay to passthrough so the flatten has market data"
if [[ -p /dev/shm/m7-relay.ctl ]] || [[ -e /dev/shm/m7-relay.ctl ]]; then
  # `open` is the relay's own word for passthrough (`ws-relay.py`'s default mode), and
  # writing anything else would leave it in a mode the flatten below cannot see through.
  printf 'open\n' > /dev/shm/m7-relay.ctl 2>/dev/null || true
fi
stop_pid "outage plan" "$RUN.outages.pid" 5

# 60 s: the producer's own `--flatten-wait` plus room for the exchange round trip. Cutting
# this short is cutting short the flatten it is waiting for.
stop_pid "producer" "$RUN.producer.pid" 90
stop_pid "session"  "$RUN.pid" 60

say "flattening from the venue's own position — a request is not an observation"
AXON_YES=1 ./run.sh flatten "$CONFIG"
FLAT=$?

stop_pid "monitor" "$RUN.monitor.pid" 5

say "reading the account twice, because a flat account is an observation"
for i in 1 2; do
  bash scripts/with-env.sh .venv/bin/python - <<'PY'
import json, os, urllib.request
url = os.environ.get("AXON_HL_INFO_URL", "https://api.hyperliquid-testnet.xyz/info")
acct = os.environ["AXON_HL_ACCOUNT_ADDRESS"]
for kind in ("clearinghouseState", "openOrders"):
    req = urllib.request.Request(
        url, data=json.dumps({"type": kind, "user": acct}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    if kind == "clearinghouseState":
        print(f"  accountValue {body['marginSummary']['accountValue']}  "
              f"positions {body.get('assetPositions')}")
    else:
        print(f"  openOrders {body}")
PY
  [[ $i -eq 1 ]] && sleep 10
done

if [[ $FLAT -ne 0 ]]; then
  printf '\033[31m%s\033[0m\n' "flatten reported NOT FLAT — do not walk away from this run"
  exit 1
fi
say "soak stopped. Analyse it: .venv/bin/python scripts/soak/analyse.py --run $RUN"
