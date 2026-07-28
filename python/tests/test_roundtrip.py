"""The Phase-1 exit criterion: a byte round-trips Python → ring → Rust (and back).

Both directions drive the *same* ring file with the *same* deterministic fixtures,
proving the Python NumPy dtype and the Rust ``#[repr(C)]`` struct are byte-identical
across the process boundary.
"""

import subprocess

from axon.signals import RingConsumer, RingProducer
from axon.signals._fixtures import make_signal

_FIELDS = (
    "seq",
    "ts_event",
    "symbol_id",
    "target_qty",
    "price_band",
    "urgency",
    "ttl_ms",
    "model_version",
    "flags",
    "schema_version",
    "kind",
)


def test_python_writes_rust_reads(ring_path, rust_examples):
    """Python producer → ring → Rust ``ipc_reader`` prints; we compare its output."""
    n = 500
    prod = RingProducer(ring_path, capacity=512)
    try:
        for i in range(n):
            assert prod.try_push(make_signal(i, n))
    finally:
        prod.close()  # flush + unmap so the Rust process reads a settled file

    res = subprocess.run(
        [rust_examples["reader"], ring_path, str(n)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    lines = res.stdout.strip().splitlines()
    assert len(lines) == n

    for i, line in enumerate(lines):
        got = dict(zip(_FIELDS, map(int, line.split())))
        exp = make_signal(i, n)
        for field in _FIELDS:
            assert got[field] == int(exp[field]), f"field {field} at record {i}"


def test_rust_writes_python_reads(ring_path, rust_examples):
    """Rust ``ipc_writer`` → ring → Python consumer; we compare every byte."""
    n = 500
    res = subprocess.run(
        [rust_examples["writer"], ring_path, str(n)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr

    with RingConsumer(ring_path) as cons:
        for i in range(n):
            rec = cons.try_pop()
            assert rec is not None, f"missing record {i}"
            assert rec.tobytes() == make_signal(i, n).tobytes(), f"mismatch at record {i}"
        assert cons.try_pop() is None
