"""``axon.parity.beacon``: telling a quiet market-data publisher from a dead one.

Each test is named after the failure mode it prevents. The two that carry the weight
are ``test_a_beat_that_did_not_move_is_a_dead_publisher_and_not_a_quiet_market`` and
``test_a_real_publisher_beat_past_the_last_slice_it_wrote``: under
``MdWritePolicy::OnChange`` those two situations write the same empty ring, and every
other test here exists to make sure the one thing that separates them keeps working.

The cross-language half drives the **real Rust writer** — ``md_writer``, the same
example the md-ring round trip uses — so the layout is checked against bytes a Rust
binary actually produced rather than against a Python restatement of the same table.
What it does *not* do is watch a live session's beacon advance in real time; see
``test_a_real_publisher_beat_past_the_last_slice_it_wrote`` for the arithmetic that
stands in for it, and ADR-0034 for why that is the honest limit today.
"""

from __future__ import annotations

import subprocess

import pytest

from axon.marketdata import bar_ring_path
from axon.parity.beacon import (
    MD_BEACON_FLAG_RUNNING,
    MD_BEACON_FLAG_STOPPED,
    MD_BEACON_MAGIC,
    MD_BEACON_SIZE,
    MD_BEACON_VERSION,
    BeaconError,
    MdBeaconReader,
    MdBeaconSnapshot,
    Publisher,
    beacon_path,
    publisher_state,
    read_md_beacon,
)

U32 = 1 << 32


def snap(**kw) -> MdBeaconSnapshot:
    """A beacon reading, with everything the test does not care about at rest."""
    base = dict(
        pid=4242,
        beats=100,
        last_event_ns=1_700_000_000_000_000_000,
        last_beat_ns=1_700_000_000_000_000_000,
        published=10,
        coalesced=20,
        dropped=0,
        bars_published=3,
        stale_quote=0,
        flags=MD_BEACON_FLAG_RUNNING,
    )
    base.update(kw)
    return MdBeaconSnapshot(**base)


# ── what two readings prove ──────────────────────────────────────────────────


def test_a_beat_that_did_not_move_is_a_dead_publisher_and_not_a_quiet_market():
    # The whole reason the file exists. Both situations leave the ring empty; only a
    # counter driven by the pass loop can separate them, and a monitor that cannot
    # separate them has to guess with a sixty-second timer.
    state = publisher_state(snap(), snap())
    assert state.state is Publisher.DEAD
    assert not state.alive
    assert "not a quiet market" in state.reason


def test_a_publisher_that_stopped_on_purpose_is_not_reported_as_a_crash():
    # It still means the monitor is watching nothing — but "the session ended" and
    # "the session died" want different words from whoever is woken up by it.
    state = publisher_state(snap(), snap(flags=MD_BEACON_FLAG_STOPPED))
    assert state.state is Publisher.STOPPED
    assert "on purpose" in state.reason


def test_a_beating_publisher_over_a_still_market_is_quiet_rather_than_dead():
    # ADR-0030's named case: the beat advances and `coalesced` climbs, so the feed is
    # busy and the top of book is not.
    state = publisher_state(snap(), snap(beats=104, coalesced=57, last_event_ns=1_700_000_000_000_000_500))
    assert state.state is Publisher.QUIET
    assert state.alive
    assert (state.beats, state.coalesced, state.published) == (4, 37, 0)


def test_a_beating_publisher_whose_event_clock_is_frozen_is_starved_rather_than_quiet():
    # The failure a duration-based health check cannot see: a Binance stream-name typo
    # is accepted silently, leaving a socket that is healthy and permanently silent
    # forever. The process is alive, the pass loop runs, and nothing arrives — which is
    # the opposite fault from a quiet market and produces the same empty ring.
    state = publisher_state(snap(), snap(beats=104))
    assert state.state is Publisher.STARVED
    assert not state.event_advanced
    assert "nothing is reaching the core" in state.reason


def test_a_publisher_that_is_writing_records_says_the_fault_is_downstream_of_the_ring():
    # Market data crossed the boundary and the monitor still compared nothing. That is
    # not the publisher's fault, and excusing the window because the publisher is
    # healthy would be the invisible-denominator bug wearing a beacon.
    state = publisher_state(snap(), snap(beats=104, published=25, last_event_ns=1_700_000_000_000_000_900))
    assert state.state is Publisher.PUBLISHING
    assert state.published == 15
    assert "downstream of the ring" in state.reason


def test_a_wrapped_counter_is_read_as_a_delta_rather_than_as_a_number_going_backwards():
    # 32-bit counters are what 64 bytes bought; the price is that the absolute value
    # stops meaning anything after five days at ten thousand slices a second. Read as a
    # total, a wrap looks like a publisher that un-published a billion records.
    state = publisher_state(
        snap(published=U32 - 3), snap(beats=101, published=2, last_event_ns=1_700_000_000_000_000_001)
    )
    assert state.published == 5
    assert state.state is Publisher.PUBLISHING


def test_a_restarted_publisher_is_not_read_as_a_beat_count_going_backwards():
    # A new session reinitialises the file, so its beat count starts again. Compared
    # naively that is a decrease, and a reader that clamped it at zero would call a
    # freshly started publisher dead.
    state = publisher_state(snap(beats=900), snap(pid=99, beats=3))
    assert state.state is Publisher.RESTARTED
    assert "another session" in state.reason


def test_a_session_with_no_wall_clock_is_reported_as_unknown_rather_than_as_1970():
    # An offline replay ages nothing against a wall clock, so it writes 0. Read as a
    # timestamp that is a beat 56 years ago, which is a very confident wrong answer.
    assert not snap(last_beat_ns=0).had_wall_clock
    assert snap().had_wall_clock


# ── the file itself ──────────────────────────────────────────────────────────


def test_a_path_nothing_has_written_is_not_ready_rather_than_a_dead_publisher(tmp_path):
    # A monitor is wired before the session it watches starts, so "no file yet" is the
    # ordinary startup race. Reported as a dead publisher it would alarm on every
    # launch, and an alarm that fires on every launch is one nobody reads.
    reader = MdBeaconReader(str(tmp_path / "nothing.beacon"))
    assert reader.read() is None
    reader.close()


def test_a_created_but_unstamped_file_is_not_ready_rather_than_a_beacon_at_beat_zero(tmp_path):
    # The window between `ftruncate` and the header store. "Beats 0 and never moving"
    # is a *conclusion*; the truthful answer is that nobody has written this yet.
    path = tmp_path / "unstamped.beacon"
    path.write_bytes(bytes(MD_BEACON_SIZE))
    with MdBeaconReader(str(path)) as reader:
        assert reader.read() is None


def test_a_liveness_beacon_is_refused_rather_than_decoded_as_market_data(tmp_path):
    # The two beacons are both 64 bytes with an identical first 40 by design, so
    # nothing about the file's shape can tell them apart — the magic is the only thing
    # that can. Read the wrong one and `signals` silently becomes `published`.
    from axon.live.liveness import LivenessBeacon

    path = tmp_path / "wrong.beacon"
    with LivenessBeacon(str(path)) as lb:
        lb.beat(last_event_ns=7, signals=3, backpressure=0, pending=0)
    with pytest.raises(BeaconError, match="bad magic"):
        read_md_beacon(str(path))
    # And the tolerant reader raises too: an unwritten file is a race that fixes
    # itself, a beacon of the wrong kind is a wiring mistake that does not.
    with MdBeaconReader(str(path)) as reader:
        with pytest.raises(BeaconError, match="bad magic"):
            reader.read()


def test_a_short_file_is_refused_rather_than_read_past_its_end(tmp_path):
    path = tmp_path / "short.beacon"
    path.write_bytes(b"AXONMDBN")
    with pytest.raises(BeaconError, match="too small"):
        read_md_beacon(str(path))


def test_a_version_this_build_does_not_know_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "future.beacon"
    raw = bytearray(MD_BEACON_SIZE)
    raw[0:8] = MD_BEACON_MAGIC.to_bytes(8, "little")
    raw[8:12] = (MD_BEACON_VERSION + 1).to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    with pytest.raises(BeaconError, match="version"):
        read_md_beacon(str(path))


def test_the_beacon_can_never_name_the_same_file_as_either_ring():
    # ADR-0030 asks for validation refusing a beacon path equal to either ring's.
    # Appending a suffix makes the collision unrepresentable instead, which is the
    # stronger form: there is no check anybody can forget to call. The adversarial
    # entry is a ring whose own path already ends in `.beacon`.
    for ring in (
        "/dev/shm/axon-md.ring",
        "/dev/shm/axon-md",
        "/dev/shm/axon-md.beacon",
        "/dev/shm/axon-md.bars.ring",
    ):
        beacon = beacon_path(ring)
        assert beacon != ring
        assert beacon != bar_ring_path(ring)


# ── against bytes a Rust binary actually wrote ───────────────────────────────


def _publish(md_writer: str, path: str, count: int, capacity: int, beats: int) -> None:
    res = subprocess.run(
        [md_writer, path, str(count), str(capacity), str(beats)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr


def test_a_beacon_written_by_rust_is_read_by_python_field_for_field(ring_path, md_writer):
    # The layout, across the boundary, against real bytes. Five counters at five
    # distinct four-byte offsets is exactly the shape a transposition survives: each
    # field checked alone passes, and only checking them together fails.
    _publish(md_writer, ring_path, count=8, capacity=16, beats=40)
    s = read_md_beacon(beacon_path(ring_path))
    assert s.beats == 40
    assert s.published == 8, "one beat per push while the ring was being filled"
    assert s.coalesced == 31, "and the idle passes after it, less the shutdown beat"
    assert (s.dropped, s.bars_published, s.stale_quote) == (0, 0, 0)
    assert s.pid > 0
    assert s.stopped and not s.running
    assert s.had_wall_clock, "the fixture writes a synthetic non-zero wall clock"
    assert s.last_event_ns > 1_700_000_000_000_000_000


def test_a_real_publisher_beat_past_the_last_slice_it_wrote(ring_path, md_writer):
    # The property the whole workstream exists for, established from a real Rust
    # session's bytes rather than argued: the beat count ran on after the ring stopped
    # gaining records. A beacon written from the event handler could not produce this
    # file at all — `beats` would equal `published` and a reader would be exactly as
    # blind as it is with no beacon.
    #
    # It is arithmetic rather than an observation over time: this reads the finished
    # file once. Watching a live publisher's beat advance from Python needs a session
    # that outlives the read, which needs the runtime wiring this workstream did not own.
    _publish(md_writer, ring_path, count=4, capacity=8, beats=500)
    s = read_md_beacon(beacon_path(ring_path))
    assert s.published == 4
    assert s.beats == 500
    assert s.beats > s.published


def test_a_second_publisher_on_the_same_path_reads_as_a_restart_and_not_as_progress(
    ring_path, md_writer
):
    # Two real Rust processes, two real readings. The second session reinitialises the
    # file and its beat count starts again — compared naively that is a count going
    # backwards, and a reader that clamped it would call the *new* publisher dead.
    _publish(md_writer, ring_path, count=8, capacity=16, beats=400)
    first = read_md_beacon(beacon_path(ring_path))
    _publish(md_writer, ring_path, count=8, capacity=16, beats=40)
    second = read_md_beacon(beacon_path(ring_path))
    assert second.beats < first.beats, "the premise: the new session counts from zero"
    state = publisher_state(first, second)
    assert state.state is Publisher.RESTARTED
