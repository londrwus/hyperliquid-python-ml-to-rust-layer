#!/usr/bin/env python3
"""Sample a soaking session's resource footprint and its growing artifacts.

The soak's whole claim is duration, so the numbers that matter are the ones with a
slope: resident memory, thread count, open file descriptors, and the size of the two
logs the session is writing. A single reading at the end proves nothing — a leak and a
steady state look identical from one sample.

Everything is read from `/proc`, so the measurement costs the session nothing and does
not need the session to cooperate. RSS in particular is taken from `/proc/<pid>/status`
(`VmRSS`) rather than `ps`, because `ps` rounds and this run is looking for a slope of a
few hundred KiB an hour.

Usage:
    monitor.py --pid 1234 --out /tmp/m7-monitor.csv --every 15 \
               --watch data/captures/m7-soak.jsonl.partial \
               --watch data/captures/m7-soak.signals.jsonl.partial
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def proc_status(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "VmSize", "Threads", "voluntary_ctxt_switches"):
                    out[k] = int(v.split()[0])
    except FileNotFoundError:
        return {}
    try:
        out["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        out["fds"] = -1
    return out


def size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=15.0)
    ap.add_argument("--watch", action="append", default=[])
    args = ap.parse_args()

    started = time.time()
    with open(args.out, "w", buffering=1) as f:
        cols = ["wall", "elapsed_s", "rss_kb", "hwm_kb", "vsz_kb", "threads", "fds"]
        cols += [f"bytes:{os.path.basename(w)}" for w in args.watch]
        f.write(",".join(cols) + "\n")
        while True:
            st = proc_status(args.pid)
            if not st:
                # The process is gone. Stopping rather than writing zeros: a row of
                # zeros in a growth series reads as a collapse, not as an exit.
                return 0
            row = [
                f"{time.time():.3f}",
                f"{time.time() - started:.1f}",
                st.get("VmRSS", -1),
                st.get("VmHWM", -1),
                st.get("VmSize", -1),
                st.get("Threads", -1),
                st.get("fds", -1),
            ]
            row += [size(w) for w in args.watch]
            f.write(",".join(str(c) for c in row) + "\n")
            time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
