"""Measure the real one-way Python→Rust latency across the signal ring.

``docs/05-latency-model.md`` puts the shared-memory hop at ~66-135 ns and that row is
a literature ladder — nothing here had ever timed it. The runtime's ``sig`` stage does
not time it either: it spans the producer's wall clock to the core's *event* clock, so
the transport sits under a feed lag three to five orders of magnitude larger and is
clamped at zero.

This driver and ``crates/axon-ipc/examples/ring_latency_probe.rs`` are the two ends of
one measurement. Three stamps, so the number decomposes instead of arriving as a single
figure nobody can attribute:

===========  ==============================================================
``t1 - t0``  Python's own write path — stamp to ``head`` published.
``t2 - t1``  **The wire** — publish to the Rust process observing it.
``t2 - t0``  What a production ``ts_event``-based stage would report.
===========  ==============================================================

``t0`` rides on the wire in ``ts_event``; ``t1`` stays here and is joined on ``seq``.

Three things are decisions rather than convenience.

**The stamp is written straight into the ring slot, last.** ``RingProducer.try_push``
copies a whole record, so a stamp taken before it would carry ~1 µs of interpreter into
a number that is supposed to be nanoseconds. The hot loop writes the payload, then the
stamp, then publishes — so ``t1 - t0`` is the *floor* of what Python costs, not a
measurement of ``try_push``.

**The reader spins by default.** A sleeping reader measures its own poll cadence and
nothing else. ``--poll-us`` reproduces the production core's ``core_poll_us`` sleep
(500 µs by default, ``config.rs``) so the two can be quoted side by side — the spin
number is the transport, the poll number is what a live session actually pays.

**Sends are paced.** Pushing in a tight loop fills the ring and measures throughput:
every record after the first would time queueing rather than transport. ``--gap-us``
keeps the reader idle-spinning when each record lands.

Run it::

    cargo build --release --example ring_latency_probe -p axon-ipc
    python scripts/ipc_latency.py --count 20000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from axon.contracts import SIGNAL_DTYPE, new_signal
from axon.signals import RingProducer


def clock_floor_ns(n: int = 100_000) -> tuple[int, int]:
    """Two back-to-back clock reads: the floor under every number below.

    Quoted rather than subtracted out. A measurement that corrects itself against its
    own noise floor is one nobody can check.
    """
    d = np.empty(n, dtype=np.int64)
    for i in range(n):
        a = time.time_ns()
        d[i] = time.time_ns() - a
    return int(d.min()), int(np.median(d))


def _steal_ticks(cpu: int) -> int:
    """This vCPU's cumulative steal, in USER_HZ. The host taking the core away is the
    one explanation for a tail that a spinning reader cannot otherwise produce."""
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith(f"cpu{cpu} "):
                return int(line.split()[8])
    return 0


def summarize(name: str, x: np.ndarray, unit: str = "ns") -> str:
    if x.size == 0:
        return f"{name:<34} (no samples)"
    q = np.percentile(x, [50, 90, 99, 99.9])
    return (
        f"{name:<34} n={x.size:<7} min={x.min():>9,} p50={q[0]:>10,.0f} "
        f"p90={q[1]:>10,.0f} p99={q[2]:>10,.0f} p999={q[3]:>11,.0f} "
        f"max={x.max():>12,} mean={x.mean():>10,.0f} {unit}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ring", default="/dev/shm/axon-latency.ring")
    ap.add_argument("--count", type=int, default=20_000, help="measured records")
    ap.add_argument("--warmup", type=int, default=2_000, help="discarded records")
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--gap-us", type=float, default=200.0, help="pause between sends")
    ap.add_argument("--poll-us", type=int, default=0, help="reader sleep; 0 spins")
    ap.add_argument(
        "--control",
        type=int,
        default=0,
        help="time the reader's empty spin iterations, to attribute the tail. Costs a "
        "clock read per spin and so inflates the median it is run beside.",
    )
    ap.add_argument("--producer-cpu", type=int, default=2)
    ap.add_argument("--consumer-cpu", type=int, default=4)
    ap.add_argument(
        "--probe",
        default="target/release/examples/ring_latency_probe",
        help="the built Rust reader",
    )
    ap.add_argument("--rust-csv", default="/tmp/ring_latency_rust.csv")
    args = ap.parse_args()

    probe = Path(args.probe).resolve()
    if not probe.exists():
        print(
            f"probe not built: {probe}\n"
            "  cargo build --release --example ring_latency_probe -p axon-ipc",
            file=sys.stderr,
        )
        return 2

    # Pin. The hop is a cache-line transfer between two cores, so an unpinned run
    # measures the scheduler's choices as much as the ring's.
    try:
        os.sched_setaffinity(0, {args.producer_cpu})
    except OSError as e:  # pragma: no cover - platform dependent
        print(f"warning: could not pin producer: {e}", file=sys.stderr)

    py_floor_min, py_floor_p50 = clock_floor_ns()

    total = args.warmup + args.count
    if os.path.exists(args.ring):
        os.unlink(args.ring)

    with RingProducer(args.ring, args.capacity, SIGNAL_DTYPE) as prod:
        child = subprocess.Popen(
            [
                "taskset", "-c", str(args.consumer_cpu), str(probe),
                "--ring", args.ring,
                "--count", str(total),
                "--poll-us", str(args.poll_us),
                "--control", str(args.control),
                "--out", args.rust_csv,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = child.stdout.readline().strip()
        if not ready.startswith("READY"):
            child.kill()
            print(f"probe did not come up: {ready!r}", file=sys.stderr)
            return 1
        rust_floor = ready.split(maxsplit=1)[1]
        steal_before = _steal_ticks(args.consumer_cpu)

        # Everything but the two stamps, written once. Field views into the mapping,
        # so the stamp is a single strided store into shared memory rather than a
        # record copy.
        template = new_signal(symbol_id=3, target_qty=1_000, ttl_ms=500, model_version=1)
        records = prod._records
        seq_view = records["seq"]
        ts_view = records["ts_event"]
        head_view = prod._head
        mask = prod.mask
        capacity = prod.capacity
        tail_view = prod._tail

        t0s = np.zeros(total, dtype=np.int64)
        t1s = np.zeros(total, dtype=np.int64)
        dropped = 0

        gap_ns = int(args.gap_us * 1000)
        perf = time.perf_counter_ns
        stamp = time.time_ns
        next_send = perf()

        for i in range(total):
            # Spin, not sleep: sleep granularity is ~50 µs and lands the send at a
            # moment the scheduler chose rather than one we did.
            while perf() < next_send:
                pass
            next_send += gap_ns

            head = int(head_view[0])
            if head - int(tail_view[0]) >= capacity:
                dropped += 1
                continue
            slot = head & mask
            records[slot] = template
            seq_view[slot] = i
            t0 = stamp()
            ts_view[slot] = t0  # the stamp, written last…
            head_view[0] = head + 1  # …then publish
            t1 = stamp()
            t0s[i] = t0
            t1s[i] = t1

        done = child.stdout.readline().strip()
        child.wait(timeout=60)
        steal_after = _steal_ticks(args.consumer_cpu)

    rust = np.genfromtxt(args.rust_csv, delimiter=",", names=True, dtype=np.int64)
    if rust.size == 0:
        print("probe read nothing", file=sys.stderr)
        return 1

    # Join on seq, and drop the warmup: the first records pay for page faults on a
    # fresh mapping and for the branch predictors on both sides being cold.
    seq = np.atleast_1d(rust["seq"])
    t2 = np.atleast_1d(rust["t2_ns"])
    keep = seq >= args.warmup
    seq, t2 = seq[keep], t2[keep]
    t0 = t0s[seq]
    t1 = t1s[seq]

    write_path = t1 - t0
    wire = t2 - t1
    end_to_end = t2 - t0

    print()
    print("=" * 118)
    print(
        f"Python→Rust signal ring · {seq.size:,} records · gap {args.gap_us:g} µs · "
        f"reader {'spinning' if args.poll_us == 0 else f'polling {args.poll_us} µs'} · "
        f"cpu {args.producer_cpu}→{args.consumer_cpu}"
    )
    print("=" * 118)
    print(summarize("t1-t0  python write path", write_path))
    print(summarize("t2-t1  THE WIRE (publish→observed)", wire))
    print(summarize("t2-t0  end-to-end (ts_event→read)", end_to_end))
    print("-" * 118)
    print(f"clock floor: python min={py_floor_min} p50={py_floor_p50} ns · rust {rust_floor}")
    print(f"dropped (ring full): {dropped} · negative wire samples: {int((wire < 0).sum())}")
    print(f"preemption control: {done}")
    print(f"consumer cpu{args.consumer_cpu} steal during run: {steal_after - steal_before} ticks (10ms each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
