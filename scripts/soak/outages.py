#!/usr/bin/env python3
"""Drive a scripted sequence of induced outages against `ws-relay.py`.

A soak that induces its outages by hand produces a story; one that runs a written plan
produces evidence. The plan is the deliverable here — the lengths, the spacing and the
*kind* of each break are what the report has to be able to state exactly, and a script
is the only way that survives an hour of watching.

Three properties are deliberate:

- **Varied length.** Sub-second, seconds, tens of seconds, and one long enough that the
  venue's own state has moved underneath the session (minutes). A reconnect loop that
  only ever sees short breaks is never asked to back off, and a `userFills` snapshot
  only re-teaches the session something after a gap long enough for a fill to have
  happened in it.
- **Awkward timing.** The plan spaces breaks so they fall at different phases of the
  20 s dead-man's-switch re-arm and the 15 s reconciliation poll rather than in step
  with either. A break that always lands between beats never tests what happens to one
  that is in flight.
- **Back off between breaks.** Every gap is at least 45 s of recovery. Hammering
  reconnects in a loop is abusive to the venue and would also measure nothing: what is
  being tested is whether a session *recovers*, and a session that is never left alone
  is never given the chance to fail to.

Plan lines (one per outage), whitespace separated:

    <at_seconds_from_start> <kind> <duration_seconds> [life_ms]

`kind` is `cut` (port dies, ECONNREFUSED, the error path) or `churn` (connections are
relayed normally and closed cleanly after `life_ms` — the path `reconnect_forever`
treats as a clean close and does not back off).
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--journal", required=True)
    args = ap.parse_args()

    steps = []
    for line in open(args.plan):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        p = line.split()
        steps.append((float(p[0]), p[1], float(p[2]), int(p[3]) if len(p) > 3 else 0))
    steps.sort(key=lambda s: s[0])

    j = open(args.journal, "a", buffering=1)

    def note(**kw):
        kw["wall"] = time.time()
        j.write(json.dumps(kw) + "\n")

    def send(cmd: str):
        # Written whole and re-read by mtime on the relay side; a partial write would
        # be read as a malformed mode and ignored, which is the safe direction.
        with open(args.control, "w") as f:
            f.write(cmd + "\n")

    t0 = time.time()
    note(ev="plan_start", steps=len(steps))
    for at, kind, dur, life in steps:
        wait = t0 + at - time.time()
        if wait > 0:
            time.sleep(wait)
        until = time.time() + dur
        send(f"{kind} {until:.3f} {life}")
        note(ev="outage_begin", kind=kind, at=round(time.time() - t0, 1),
             duration_s=dur, life_ms=life)
        time.sleep(dur)
        send("open")
        note(ev="outage_end", kind=kind, at=round(time.time() - t0, 1))
    note(ev="plan_done", elapsed_s=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
