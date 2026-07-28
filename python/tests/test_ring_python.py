"""The Python ring in isolation (same-process producer↔consumer)."""

import struct

import pytest

from axon.contracts import MD_SLICE_DTYPE, MD_SLICE_SIZE, RING_OFF_RECORD_SIZE
from axon.marketdata import MdRingConsumer
from axon.marketdata._fixtures import make_md_slice
from axon.signals import RingConsumer, RingProducer
from axon.signals._fixtures import make_signal


def test_roundtrip_preserves_every_byte(ring_path):
    n = 1000
    with RingProducer(ring_path, capacity=1024) as prod:
        for i in range(n):
            assert prod.try_push(make_signal(i, n))
        with RingConsumer(ring_path) as cons:
            for i in range(n):
                rec = cons.try_pop()
                assert rec is not None
                assert rec.tobytes() == make_signal(i, n).tobytes()
            assert cons.try_pop() is None


def test_full_ring_reports_false(ring_path):
    with RingProducer(ring_path, capacity=2) as prod:
        assert prod.try_push(make_signal(0, 2))
        assert prod.try_push(make_signal(1, 2))
        assert not prod.try_push(make_signal(2, 2))  # full


def test_consumer_rejects_non_ring_file(tmp_path):
    junk = tmp_path / "junk.ring"
    junk.write_bytes(b"\x00" * 4096)
    try:
        RingConsumer(str(junk))
    except ValueError as e:
        assert "magic" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for a non-ring file")


def test_md_ring_roundtrips_in_python(ring_path):
    n = 300
    with RingProducer(ring_path, capacity=512, dtype=MD_SLICE_DTYPE) as prod:
        for i in range(n):
            assert prod.try_push(make_md_slice(i))
        with MdRingConsumer(ring_path) as cons:
            batch = cons.read_batch()
            assert len(batch) == n
            for i in range(n):
                assert batch[i].tobytes() == make_md_slice(i).tobytes()
            assert cons.read_batch().size == 0


def test_a_batch_that_wraps_the_ring_stays_in_order(ring_path):
    # The batch read copies at most two contiguous runs; if the wrap split is wrong
    # the records come back rotated, which no per-record test would catch.
    with RingProducer(ring_path, capacity=8, dtype=MD_SLICE_DTYPE) as prod:
        with MdRingConsumer(ring_path) as cons:
            for i in range(5):
                assert prod.try_push(make_md_slice(i))
            assert len(cons.read_batch(3)) == 3  # tail is now mid-ring
            for i in range(5, 11):
                assert prod.try_push(make_md_slice(i))
            batch = cons.read_batch()
            assert [int(s) for s in batch["seq"]] == list(range(3, 11))
            assert cons.dropped == 0


def test_a_publisher_drop_is_visible_as_a_seq_gap(ring_path):
    # The Rust core never blocks on a slow Python: a full ring drops the update.
    # Python must be able to tell that its feature history has a hole.
    with RingProducer(ring_path, capacity=8, dtype=MD_SLICE_DTYPE) as prod:
        with MdRingConsumer(ring_path) as cons:
            for i in (0, 1, 5, 6):  # 2..4 were dropped by the publisher
                assert prod.try_push(make_md_slice(i))
            assert len(cons.read_batch()) == 4
            assert cons.dropped == 3
            for i in (9, 10):  # a further gap, this time across batches
                assert prod.try_push(make_md_slice(i))
            assert len(cons.read_batch()) == 2
            assert cons.dropped == 5


def test_an_md_reader_refuses_a_signal_ring(ring_path):
    with RingProducer(ring_path, capacity=4) as prod:
        assert prod.try_push(make_signal(0, 1))
    with pytest.raises(ValueError, match="record size mismatch"):
        MdRingConsumer(ring_path)


def test_a_matching_stride_does_not_admit_the_wrong_record(ring_path):
    # The strides differ *today*. Forge a signal ring that claims the MdSlice stride
    # to prove the type check does not rest on that coincidence: an MdSlice reader
    # over Signal bytes would report `target_qty` as a bid price.
    with RingProducer(ring_path, capacity=4) as prod:
        assert prod.try_push(make_signal(0, 1))
    with open(ring_path, "r+b") as f:
        f.seek(RING_OFF_RECORD_SIZE)
        f.write(struct.pack("<I", MD_SLICE_SIZE))

    with pytest.raises(ValueError, match="record kind"):
        MdRingConsumer(ring_path)
