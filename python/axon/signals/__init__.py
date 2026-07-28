"""The Python client for Axon's shared-memory SPSC ring.

Mirrors the Rust ``axon-ipc`` ``RingProducer``/``RingConsumer`` byte-for-byte (same
header layout, same record stride, same head/tail placement), so a ring created on
one side can be driven from the other. In Boundary B, Python is normally the
*producer* on the signal ring (Py→Rust); the consumer is here too for the
market-data ring (Rust→Py, see :mod:`axon.marketdata`) and for tests.

Both ends are parameterized by the record *dtype* and default to the signal
record, so existing signal call sites read unchanged. The dtype is the only knob:
the ring's control block (record kind + schema version) is looked up from it via
:func:`axon.contracts.record_spec_for`, so a ring cannot be stamped as carrying
one record while its reader decodes another (ADR-0012).

Concurrency & ordering (x86_64 assumption, per ``axon-ipc`` docs): ``head`` and
``tail`` are 8-byte aligned counters on separate cache lines. The producer writes
the record payload *before* publishing the new ``head``; x86-TSO store ordering
guarantees a reader that observes the new ``head`` also observes the payload.

Views over the mapping use ``np.ndarray(buffer=...)`` (not ``np.frombuffer``) so
they are writable zero-copy views — the same idiom used with
``multiprocessing.shared_memory``.
"""

from __future__ import annotations

import mmap
import os

import numpy as np

from axon.contracts import (
    RING_HEADER_SIZE,
    RING_MAGIC,
    RING_OFF_CAPACITY,
    RING_OFF_HEAD,
    RING_OFF_MAGIC,
    RING_OFF_RECORD_KIND,
    RING_OFF_RECORD_SCHEMA_VERSION,
    RING_OFF_RECORD_SIZE,
    RING_OFF_RING_VERSION,
    RING_OFF_TAIL,
    RING_VERSION,
    SIGNAL_DTYPE,
    record_spec_for,
)


def _u64_view(buf, offset: int) -> np.ndarray:
    """A writable length-1 little-endian u64 view into ``buf`` at ``offset``."""
    return np.ndarray((1,), dtype="<u8", buffer=buf, offset=offset)


def _u32_view(buf, offset: int) -> np.ndarray:
    """A writable length-1 little-endian u32 view into ``buf`` at ``offset``."""
    return np.ndarray((1,), dtype="<u4", buffer=buf, offset=offset)


def _total_len(capacity: int, record_size: int) -> int:
    return RING_HEADER_SIZE + capacity * record_size


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class RingProducer:
    """The writer end. Exactly one per ring. Creates/sizes the backing file."""

    def __init__(self, path: str, capacity: int, dtype: np.dtype = SIGNAL_DTYPE):
        if not _is_pow2(capacity):
            raise ValueError(f"capacity must be a power of two > 0, got {capacity}")
        self.record = record_spec_for(dtype)
        self.dtype = self.record.dtype
        self.path = path
        self.capacity = capacity
        self.mask = capacity - 1
        total = _total_len(capacity, self.record.size)

        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)

        # Writable zero-copy views into the mapping.
        self._head = _u64_view(self._mm, RING_OFF_HEAD)
        self._tail = _u64_view(self._mm, RING_OFF_TAIL)
        self._records = np.ndarray(
            (capacity,), dtype=self.dtype, buffer=self._mm, offset=RING_HEADER_SIZE
        )

        # Write the control header, then zero the indices.
        _u64_view(self._mm, RING_OFF_MAGIC)[0] = RING_MAGIC
        _u32_view(self._mm, RING_OFF_RING_VERSION)[0] = RING_VERSION
        _u32_view(self._mm, RING_OFF_RECORD_SIZE)[0] = self.record.size
        _u32_view(self._mm, RING_OFF_CAPACITY)[0] = capacity
        _u32_view(self._mm, RING_OFF_RECORD_SCHEMA_VERSION)[0] = self.record.schema_version
        _u32_view(self._mm, RING_OFF_RECORD_KIND)[0] = self.record.kind
        self._head[0] = 0
        self._tail[0] = 0
        self._mm.flush()

    def try_push(self, record: np.ndarray) -> bool:
        """Push one record. Returns ``False`` if the ring is full."""
        head = int(self._head[0])
        tail = int(self._tail[0])
        if head - tail >= self.capacity:
            return False
        slot = head & self.mask
        self._records[slot] = record  # payload write (before publish)
        self._head[0] = head + 1  # publish
        return True

    def __len__(self) -> int:
        return int(self._head[0]) - int(self._tail[0])

    def flush(self) -> None:
        self._mm.flush()

    def close(self) -> None:
        # Drop the array views before closing the mmap, or mmap.close() raises
        # BufferError ("cannot close exported pointers exist").
        self._head = None
        self._tail = None
        self._records = None
        if getattr(self, "_mm", None) is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None

    def __enter__(self) -> "RingProducer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RingConsumer:
    """The reader end. Exactly one per ring. Opens and validates an existing file."""

    def __init__(self, path: str, dtype: np.dtype = SIGNAL_DTYPE):
        self.record = record_spec_for(dtype)
        self.dtype = self.record.dtype
        self.path = path
        fd = os.open(path, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            if size < RING_HEADER_SIZE:
                raise ValueError(f"file too small ({size} bytes) to be an Axon ring")
            self._mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)

        magic = int(_u64_view(self._mm, RING_OFF_MAGIC)[0])
        if magic != RING_MAGIC:
            raise ValueError(f"not an Axon ring: bad magic {magic:#018x}")
        ring_version = int(_u32_view(self._mm, RING_OFF_RING_VERSION)[0])
        if ring_version != RING_VERSION:
            raise ValueError(f"unsupported ring version {ring_version}")
        record_size = int(_u32_view(self._mm, RING_OFF_RECORD_SIZE)[0])
        if record_size != self.record.size:
            raise ValueError(
                f"record size mismatch: header {record_size}, "
                f"{self.record.name} is {self.record.size}"
            )
        # Checked in addition to the stride: two records of equal size are byte-
        # compatible and mutually meaningless, so the stride alone cannot say which
        # one is in the data region (ADR-0012).
        record_kind = int(_u32_view(self._mm, RING_OFF_RECORD_KIND)[0])
        if record_kind != self.record.kind:
            raise ValueError(
                f"ring carries record kind {record_kind}, this reader reads "
                f"kind {self.record.kind} ({self.record.name})"
            )
        schema = int(_u32_view(self._mm, RING_OFF_RECORD_SCHEMA_VERSION)[0])
        if schema != self.record.schema_version:
            raise ValueError(
                f"{self.record.name} schema version mismatch: header {schema}, "
                f"this build {self.record.schema_version}"
            )

        self.capacity = int(_u32_view(self._mm, RING_OFF_CAPACITY)[0])
        if not _is_pow2(self.capacity):
            raise ValueError(f"capacity must be a power of two, got {self.capacity}")
        if size < _total_len(self.capacity, self.record.size):
            raise ValueError("mapped file too small for declared capacity")
        self.mask = self.capacity - 1

        self._head = _u64_view(self._mm, RING_OFF_HEAD)
        self._tail = _u64_view(self._mm, RING_OFF_TAIL)
        self._records = np.ndarray(
            (self.capacity,), dtype=self.dtype, buffer=self._mm, offset=RING_HEADER_SIZE
        )

    def try_pop(self):
        """Pop one record as a self-owning 0-d record, or ``None`` if empty."""
        tail = int(self._tail[0])
        head = int(self._head[0])
        if tail == head:
            return None
        slot = tail & self.mask
        out = np.empty((), dtype=self.dtype)
        out[()] = self._records[slot]  # copy out before releasing the slot
        self._tail[0] = tail + 1
        return out

    def read_batch(self, max_records: int | None = None) -> np.ndarray:
        """Pop up to ``max_records`` records at once, oldest first.

        One interpreter call per *batch* rather than per record: Python's floor is
        ~50–350 ns per call (``docs/02-python-rust-boundary.md``), which a
        per-record API pays on every single update. Everything below is bulk NumPy
        — at most two slice copies (the ring wraps at most once per batch) and a
        single ``tail`` publish.

        Returns a **copy**: the slots are released the moment ``tail`` advances, so
        a view into them could be overwritten by the producer mid-computation.
        """
        tail = int(self._tail[0])
        head = int(self._head[0])
        available = head - tail
        if max_records is not None:
            available = min(available, max(max_records, 0))
        if available <= 0:
            return np.empty((0,), dtype=self.dtype)

        out = np.empty((available,), dtype=self.dtype)
        start = tail & self.mask
        first = min(available, self.capacity - start)
        out[:first] = self._records[start : start + first]
        if first < available:
            out[first:] = self._records[: available - first]
        self._tail[0] = tail + available  # release the slots, once
        return out

    def __len__(self) -> int:
        return int(self._head[0]) - int(self._tail[0])

    def close(self) -> None:
        self._head = None
        self._tail = None
        self._records = None
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None

    def __enter__(self) -> "RingConsumer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["RingProducer", "RingConsumer"]
