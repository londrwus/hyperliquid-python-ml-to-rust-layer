#!/usr/bin/env python3
"""Read a soak's three artifacts together and answer the questions the soak was for.

The session log says what the session *believed*, the relay journal says what actually
happened to the socket, and the capture says what arrived. Each alone is a story; the
value is in the join — in particular, "did every subscription come back?" is a question
only the capture can answer and only the relay journal can ask, because the session
itself has no idea a subscription is missing. ADR-0020 §7 is explicit about the shape:
a live `activeAssetCtx` keeps `MarkCache::get` fresh for an instrument whose `l2Book`
a reconnect quietly failed to restore, so the status line reads `marks 3/3` over a book
that stopped updating.

Reported:

- every induced outage, its kind, and how long the session took to get a socket back;
- every connection, and which of the nine (symbol × feed) streams delivered an event
  during it — the half-restored-subscription check;
- the gap between a connection closing and the next one opening, split by *how* it
  closed, because a clean close and a refused connect are handled by different branches
  of `reconnect_forever` and only one of them backs off;
- `ORPHAN FILLS` over time, which is what says whether the `userFills` snapshot's dedup
  on `trade_id` holds across many reconnects rather than only the first.

Usage:
    analyse.py --journal /dev/shm/m7-relay.jsonl --capture data/captures/m7-soak.jsonl \
               --log /dev/shm/m7-soak.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

NS = 1_000_000_000


def read_journal(path: str):
    conns, outages, events = {}, [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(r)
        ev = r.get("ev")
        if ev == "conn_open":
            conns[r["cid"]] = {"open": r["wall"], "close": None, "reason": None}
        elif ev == "churn_close":
            if r["cid"] in conns:
                conns[r["cid"]]["reason"] = "clean-close"
        elif ev == "conn_closed":
            if r["cid"] in conns:
                conns[r["cid"]]["close"] = r["wall"]
                conns[r["cid"]].setdefault("reason", None)
        elif ev == "conns_killed":
            for c in conns.values():
                if c["close"] is None:
                    c["reason"] = "rst"
        elif ev == "mode":
            outages.append((r["wall"], r["frm"], r["to"]))
    return conns, outages, events


def read_capture(path: str):
    """(kind, symbol_id) -> sorted list of event wall-seconds.

    The last line is skipped when it does not parse: a capture read while the writer is
    still appending ends mid-record, and treating that as corruption would make every
    mid-run reading a false alarm.
    """
    streams: dict[tuple[str, int], list[float]] = defaultdict(list)
    n = fills = 0
    with open(path) as f:
        for line in f:
            if not line.startswith('{"Event"'):
                continue
            try:
                r = json.loads(line)["Event"]
            except json.JSONDecodeError:
                continue
            n += 1
            ev = r["event"]
            group = next(iter(ev))
            inner = ev[group]
            kind = next(iter(inner))
            body = inner[kind]
            if kind == "Fill":
                fills += 1
            sid = body.get("symbol_id") if isinstance(body, dict) else None
            if sid is not None:
                streams[(kind, sid)].append(r["ts_event"] / NS)
    for v in streams.values():
        v.sort()
    return streams, n, fills


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--log")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds after a connect before a stream is expected back")
    args = ap.parse_args()

    conns, outages, _ = read_journal(args.journal)
    streams, n_events, n_fills = read_capture(args.capture)
    # Only the *market* feeds are expected in every connection window. `Fill` arrives
    # once as the `userFills` snapshot and then only when something trades, so a
    # read-only soak would report every window after the first as a missing stream —
    # a false alarm that would drown the real one this check exists to find.
    kinds = [k for k in sorted({k for k, _ in streams}) if k in ("Bbo", "Book", "Ticker")]
    syms = sorted({s for _, s in streams})

    print(f"capture: {n_events} events, {n_fills} fills, "
          f"{len(kinds)} kinds x {len(syms)} symbols")
    print(f"relay:   {len(conns)} connections, "
          f"{sum(1 for _, f, t in outages if t != 'open')} outages induced")

    # -- reconnect latency, split by how the previous connection ended -----------
    print("\n== connections ==")
    print(f"{'cid':>4} {'open(rel)':>10} {'secs':>8} {'closed by':>11} {'gap':>7}  streams seen")
    ordered = sorted(conns.items())
    t0 = ordered[0][1]["open"] if ordered else 0
    gaps = defaultdict(list)
    missing_total = Counter()
    for i, (cid, c) in enumerate(ordered):
        end = c["close"]
        dur = (end - c["open"]) if end else float("nan")
        gap = ""
        if i and ordered[i - 1][1]["close"]:
            g = c["open"] - ordered[i - 1][1]["close"]
            prev_reason = ordered[i - 1][1]["reason"] or "?"
            gaps[prev_reason].append(g)
            gap = f"{g:6.2f}s"
        seen, missing = [], []
        lo, hi = c["open"] + args.settle, end or 1e18
        for k in kinds:
            for s in syms:
                ts = streams.get((k, s), [])
                hit = any(lo <= t <= hi for t in ts)
                (seen if hit else missing).append(f"{k}:{s}")
        for m in missing:
            missing_total[m] += 1
        print(f"{cid:>4} {c['open'] - t0:9.1f}s {dur:7.1f}s {c['reason'] or '-':>11} "
              f"{gap:>7}  {len(seen)}/{len(kinds) * len(syms)}"
              + (f"  MISSING {','.join(missing)}" if missing else ""))

    print("\n== reconnect gap by how the previous socket ended ==")
    for reason, gs in sorted(gaps.items()):
        gs = sorted(gs)
        print(f"  {reason:>11}: n={len(gs)} min={gs[0]:.3f}s "
              f"median={gs[len(gs) // 2]:.3f}s max={gs[-1]:.3f}s")

    if missing_total:
        print("\n== streams absent from a connection window (candidate half-restore) ==")
        for k, v in missing_total.most_common():
            print(f"  {k}: absent in {v} connection window(s)")

    # -- what the session itself said -------------------------------------------
    if args.log:
        orphan = Counter()
        warn = Counter()
        errs = Counter()
        status = 0
        for line in open(args.log):
            if line.startswith("axon "):
                status += 1
                m = re.search(r"ORPHAN FILLS (\d+)", line)
                if m:
                    orphan[int(m.group(1))] += 1
                for w in ("STALE MARKS", "HALTED", "CAPTURE STOPPED", "DMS", "DEGRADED",
                          "RECONCILE"):
                    if w in line:
                        warn[w] += 1
            elif "error" in line.lower() or "WS " in line:
                errs[re.sub(r"\d+", "N", line.strip())[:110]] += 1
        print(f"\n== session log ==\n  {status} status lines")
        print(f"  ORPHAN FILLS values seen: {dict(orphan)}")
        print(f"  status-line warnings: {dict(warn)}")
        print("  repeated log lines:")
        for k, v in errs.most_common(12):
            print(f"    {v:>5}x {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
