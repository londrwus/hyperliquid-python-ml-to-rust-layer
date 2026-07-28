"""The Rust→Python direction: the Rust core publishes, Python reads a batch.

Mirrors ``test_roundtrip.py`` for the market-data ring (ADR-0012). Only one
direction exists here by design — the md ring is SPSC with Rust as the sole
producer — so the check is that ``md_writer``'s records arrive byte-identical to
the fixture Python builds independently from the *same* ``contracts/schema.toml``.

The second half of this file drives the **real runtime** rather than the fixture
writer. ``md_writer`` proves the record layout crosses the boundary; it cannot prove
that anything in production ever writes one, and for the whole of ADR-0012's life
nothing did. ``axon``'s offline session runs the real ``MarketDataProcessor``, the
real fan-out and the real ``MdPublisher`` over its canned event stream, so these
tests fail if the publisher is unwired, mistimed, or writing the wrong field —
none of which the fixture writer can see.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from axon.contracts import MD_KIND_SNAPSHOT, new_md_slice, to_fixed
from axon.marketdata import (
    MD_BAR_FLAG_FIRST_BAR,
    MD_BAR_FLAG_GAP_BEFORE,
    MD_KIND_QUOTE,
    MD_KIND_TRADE,
    MdBarRingConsumer,
    MdRingConsumer,
    bar_ring_path,
)
from axon.marketdata._fixtures import make_md_bar, make_md_slice

# The ``md_writer`` fixture lives in conftest.py alongside the signal-ring examples:
# they are one cargo build with one skip policy, and a second copy here drifted from
# it silently the moment either side changed.


def _publish(md_writer: str, path: str, n: int, capacity: int | None = None) -> None:
    cmd = [md_writer, path, str(n)] + ([str(capacity)] if capacity else [])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr


def test_rust_publishes_python_reads_a_batch(ring_path, md_writer):
    """Rust ``md_writer`` → md ring → one batched Python read; every byte compared."""
    n = 500
    _publish(md_writer, ring_path, n)

    with MdRingConsumer(ring_path) as md:
        batch = md.read_batch()
        assert len(batch) == n, "one call must return the whole queue, not one record"
        for i in range(n):
            assert batch[i].tobytes() == make_md_slice(i).tobytes(), f"mismatch at record {i}"
        assert md.dropped == 0
        assert md.read_batch().size == 0


def test_both_update_kinds_and_the_aggressor_flag_survive_the_crossing(ring_path, md_writer):
    """The fields most likely to be mis-mapped are the small ones after the i64s."""
    n = 12
    _publish(md_writer, ring_path, n)

    with MdRingConsumer(ring_path) as md:
        batch = md.read_batch()

    kinds = [int(k) for k in batch["kind"]]
    assert kinds == [MD_KIND_TRADE if i % 3 == 0 else MD_KIND_QUOTE for i in range(n)]
    assert [bool(f & 1) for f in batch["flags"]] == [i % 2 == 0 for i in range(n)]
    # The print's own time is deliberately *older* than the slice's — a reader that
    # confused the two would see it equal.
    assert all(int(s["last_trade_ts"]) < int(s["ts_event"]) for s in batch)


def test_the_marks_two_clocks_survive_the_crossing_separately(ring_path, md_writer):
    """The pair ADR-0011 refuses to collapse, checked on the wire.

    A reader that folded ``mark_ts_venue`` and ``mark_ts_ingest`` into one field could
    not tell "the venue said when" from "we noticed when" — and a mark age derived
    from the second measures this machine rather than the market, so it does not
    reproduce on replay.
    """
    n = 15
    _publish(md_writer, ring_path, n)

    with MdRingConsumer(ring_path) as md:
        batch = md.read_batch()

    # Every fifth record carries no ticker at all: the whole tail stays at its zero
    # sentinel, and "has a ticker" is the receipt clock, never the price.
    assert [int(s["mark_ts_ingest"]) == 0 for s in batch] == [i % 5 == 4 for i in range(n)]
    assert all(int(s["mark_px"]) == 0 for s in batch if int(s["mark_ts_ingest"]) == 0)
    # Half the rest are venue-stamped and half are not, which is the distinction.
    venue_timed = [i for i in range(n) if int(batch[i]["mark_ts_venue"]) != 0]
    assert venue_timed == [i for i in range(n) if i % 5 != 4 and i % 2 == 1]
    # And the rate never arrives without the interval it is charged over.
    assert all(
        int(s["funding_interval_ns"]) == 3_600_000_000_000
        for s in batch
        if int(s["funding_rate"]) != 0
    )


def test_the_bar_ring_crosses_the_boundary_byte_for_byte(ring_path, md_bar_writer):
    """Rust ``md_bar_writer`` → bar ring → one batched Python read.

    ``MdBar`` and ``MdSlice`` are both 128 bytes, so nothing about the stride can
    catch a field mapped to the wrong offset here — only comparing every byte against
    a fixture Python builds from the *same* ``contracts/schema.toml`` can.
    """
    n = 40
    _publish(md_bar_writer, ring_path, n)

    with MdBarRingConsumer(ring_path) as bars:
        batch = bars.read_batch()
        assert len(batch) == n
        for i in range(n):
            assert batch[i].tobytes() == make_md_bar(i).tobytes(), f"mismatch at bar {i}"
        assert bars.dropped == 0
        # The continuity flags are the only thing on this record that can say a feature
        # window spans a hole, so a reader that dropped `flags` must fail here.
        assert bars.first_bars == 1
        assert bars.gaps == len([i for i in range(1, n) if i % 7 == 0])


def test_a_bar_reaches_a_strategy_facing_event_with_its_close_time_intact(
    ring_path, md_bar_writer
):
    """The whole point of the ring: ``MdBar`` → :class:`axon.strategy.events.Bar`.

    ``ts_event`` must cross unchanged as the bar's **close**. A bar handed to a
    strategy under its open time is the textbook lookahead leak, and re-deriving the
    close here would be a second place for the two languages to drift the one
    millisecond that makes ``align_by_event_time`` intersect to nothing.
    """
    _publish(md_bar_writer, ring_path, 5)

    with MdBarRingConsumer(ring_path) as feed:
        events = feed.read_bars()

    assert len(events) == 5
    assert all(isinstance(b.ts_event, int) for b in events), "float64 buckets nanoseconds"
    for i, bar in enumerate(events):
        rec = make_md_bar(i)
        assert bar.ts_event == int(rec["ts_event"])
        assert bar.ts_event > int(rec["open_time"]), "the close, never the open"
        assert bar.close == int(rec["close"])
        assert bar.symbol_id == int(rec["symbol_id"])
    # Strictly increasing, which is what an event-time strategy loop requires.
    assert all(b.ts_event < c.ts_event for b, c in zip(events, events[1:]))


def test_a_bar_off_the_ring_dispatches_to_a_strategys_on_bar(ring_path, md_bar_writer, tmp_path):
    """The end of the chain: Rust candle feed → bar ring → ``Strategy.on_bar``.

    ``StrategyRunner`` routes on the event's exact type first, so a ``Bar``-lookalike
    would fall off the dispatch table and the strategy would simply never be called —
    a strategy that mysteriously never trades, which is the failure the runner raises
    ``TypeError`` for rather than absorbing. This pins the type that actually crosses.
    """
    from axon.live import StrategyRunner
    from axon.strategy.base import Strategy

    class Recorder(Strategy):
        def __init__(self):
            self.seen = []

        def on_bar(self, bar, ctx):
            self.seen.append(bar)

    _publish(md_bar_writer, ring_path, 3)
    strategy = Recorder()
    with StrategyRunner(strategy, ring_path=str(tmp_path / "signals.ring"), capacity=8) as runner:
        with MdBarRingConsumer(ring_path) as feed:
            bars = feed.read_bars()
        runner.run(bars)

    assert [b.ts_event for b in strategy.seen] == [int(make_md_bar(i)["ts_event"]) for i in range(3)]
    assert [b.close for b in strategy.seen] == [int(make_md_bar(i)["close"]) for i in range(3)]


def test_a_slice_reader_refuses_a_bar_ring_despite_the_matching_stride(
    ring_path, md_bar_writer
):
    """The check that has to hold now that two records share a size.

    ``record_size`` agrees on both, so without the header's ``record_kind`` an
    ``MdRingConsumer`` over a bar ring would report ``open`` as a bid price and
    ``high`` as a bid size — plausible numbers, silently wrong, with nothing
    downstream able to notice.
    """
    _publish(md_bar_writer, ring_path, 4)
    with pytest.raises(ValueError, match="record kind"):
        MdRingConsumer(ring_path)


def test_a_partial_read_leaves_the_rest_queued(ring_path, md_writer):
    """`max_records` must release exactly what it returned, not the whole ring."""
    n = 64
    _publish(md_writer, ring_path, n)

    with MdRingConsumer(ring_path) as md:
        first = md.read_batch(10)
        assert [int(s) for s in first["seq"]] == list(range(10))
        rest = md.read_batch()
        assert [int(s) for s in rest["seq"]] == list(range(10, n))
        assert md.dropped == 0


# ── the real publisher: the runtime's own session writes the ring ────────────────
#
# Deliberately not folded into conftest's ``_ipc_examples``: that fixture is one
# ``cargo build -p axon-ipc --examples`` invocation, and the runtime is a different
# package and a different binary. Sharing the fixture would make every signal-ring
# test pay to build the whole runtime before it could run.


def _find_cargo() -> str | None:
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    candidate = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    return str(candidate) if candidate.exists() else None


@pytest.fixture(scope="session")
def axon_binary(repo_root) -> str:
    """The ``axon`` runtime binary, rebuilt before it is trusted.

    Built unconditionally rather than "if missing", unlike conftest's examples: this
    test asserts on the *publisher's* behaviour, and a target dir holding yesterday's
    binary would report a green suite for a publisher that no longer exists. Cargo is
    a no-op when nothing changed, so the cost is a fraction of a second.

    Skips rather than fails when there is no toolchain, so a Python-only CI stage
    still passes — the same policy conftest applies to the IPC examples.
    """
    env = os.environ.get("CARGO_TARGET_DIR")
    target = Path(env) if env else repo_root / "target"
    binary = target / "debug" / ("axon.exe" if os.name == "nt" else "axon")
    cargo = _find_cargo()
    if cargo is None:
        pytest.skip("cargo not found; cannot build the axon runtime")
    try:
        subprocess.run(
            [cargo, "build", "-q", "-p", "axon-runtime", "--bin", "axon"],
            cwd=repo_root,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"failed to build the axon runtime: {e}")
    if not binary.exists():  # pragma: no cover
        pytest.skip(f"axon runtime not at {binary}")
    return str(binary)


#: The whole ``[md_ring]`` table, up to the next table or end of file. Replacing the
#: block rather than individual keys keeps this working if the section gains a field.
_MD_SECTION = re.compile(r"(?ms)^\[md_ring\]\n.*?(?=^\[|\Z)")


def _run_axon(
    axon_binary: str, tmp_path, ring: str, enabled: bool = True, capacity: int = 16
) -> str:
    """Run one offline session pointed at ``ring``. Returns stdout.

    The config comes from the binary's own ``--dump-config`` so this test cannot
    drift from the schema: a renamed key breaks the substitution loudly instead of
    silently running with a default. ``ring`` is set even when publishing is off, so
    "no file appeared" is an assertion about a path this test owns rather than about
    whatever happens to be in ``/dev/shm``.
    """
    dump = subprocess.run(
        [axon_binary, "--dump-config"], capture_output=True, text=True, timeout=120, check=True
    ).stdout
    block = (
        "[md_ring]\n"
        f"enabled = {'true' if enabled else 'false'}\n"
        f'path = "{ring}"\n'
        f"capacity = {capacity}\n"
        'policy = "on_change"\n\n'
    )
    dump, n = _MD_SECTION.subn(block, dump)
    assert n == 1, "the dumped config no longer has an [md_ring] section"

    cfg = tmp_path / "axon.toml"
    cfg.write_text(dump)
    res = subprocess.run(
        [axon_binary, "--config", str(cfg)], capture_output=True, text=True, timeout=120
    )
    assert res.returncode == 0, res.stderr
    return res.stdout


def test_the_runtime_publishes_slices_python_can_read(axon_binary, tmp_path):
    """The real session's fan-out → md ring → one batched Python read.

    Every byte is compared against a fixture Python builds from the *same*
    ``contracts/schema.toml``, so a publisher that wrote the right numbers into the
    wrong fields would fail here rather than reaching a feature computation.
    """
    ring = str(tmp_path / "axon-md.ring")
    _run_axon(axon_binary, tmp_path, ring)

    with MdRingConsumer(ring) as md:
        batch = md.read_batch()
        assert md.dropped == 0, "a gap in seq means the publisher dropped a slice"
        assert md.read_batch().size == 0

    # The canned session quotes BTC (symbol 0) and snapshots ETH's book (symbol 1).
    # Its ticker still triggers no record of its own — it is the one event a live venue
    # would not timestamp (ADR-0011) — but its mark and index now ride BTC's next
    # quote, which is the only way they cross at all (ADR-0028). ETH has no ticker
    # feed, so its slice keeps the whole tail at the "nothing seen yet" sentinel.
    expected = [
        new_md_slice(
            seq=0,
            ts_event=2_000_000_000,
            symbol_id=0,
            bid_px=to_fixed(49_999),
            bid_sz=to_fixed(1),
            ask_px=to_fixed(50_001),
            ask_sz=to_fixed(1),
            kind=MD_KIND_QUOTE,
            mark_px=to_fixed(50_000),
            index_px=to_fixed(49_995),
            mark_ts_venue=1_000_000_000,
            mark_ts_ingest=1_000_000_000,
        ),
        new_md_slice(
            seq=1,
            ts_event=3_000_000_000,
            symbol_id=1,
            bid_px=to_fixed(2_999),
            bid_sz=to_fixed(10),
            ask_px=to_fixed(3_001),
            ask_sz=to_fixed(10),
            kind=MD_KIND_SNAPSHOT,
        ),
    ]
    assert len(batch) == len(expected), f"got {batch['kind']!r}"
    for i, exp in enumerate(expected):
        assert batch[i].tobytes() == exp.tobytes(), f"mismatch at record {i}"

    # No print has crossed in this session, so the last-trade fields keep the record's
    # own "nothing yet" sentinel rather than being back-filled from the quote.
    assert all(int(s["last_trade_ts"]) == 0 for s in batch)
    # Neither is the mark back-filled from the mid on the instrument that has no
    # ticker: a fabricated mark is one the venue would not margin against, and the
    # risk gate and the feature computation would then be reasoning about different
    # prices.
    assert int(batch[1]["mark_ts_ingest"]) == 0
    assert int(batch[1]["mark_px"]) == 0


def test_the_runtime_always_creates_a_bar_ring_a_consumer_can_open(axon_binary, tmp_path):
    """One switch, two rings — and the empty one is still openable.

    The offline session subscribes to no candle feed, so no bar can be published. The
    file must exist anyway: a consumer that cannot open it has no way to tell "no bars
    yet" from "wrong path", and the second reads exactly like a strategy with nothing
    to say.
    """
    ring = str(tmp_path / "axon-md.ring")
    out = _run_axon(axon_binary, tmp_path, ring)
    bars = bar_ring_path(ring)

    assert Path(bars).exists(), f"no bar ring beside the slice ring:\n{out}"
    with MdBarRingConsumer(bars) as feed:
        assert feed.read_bars() == []
        assert feed.dropped == 0
        assert feed.gaps == 0


def test_the_runtimes_own_count_agrees_with_what_crossed_the_boundary(axon_binary, tmp_path):
    """The operator's status line and Python's batch must be the same number.

    They are measured on opposite sides of the mapping. If they ever disagree, one of
    them is lying about whether the feed is healthy — which is the single thing the md
    counters exist to answer.
    """
    ring = str(tmp_path / "axon-md.ring")
    out = _run_axon(axon_binary, tmp_path, ring)

    m = re.search(r"\bmd (\d+) q (\d+)/(\d+)\b", out)
    assert m, f"the status line reported no md counters:\n{out}"
    published, queued, capacity = (int(g) for g in m.groups())
    assert capacity == 16

    with MdRingConsumer(ring) as md:
        batch = md.read_batch()
    assert queued == len(batch), "the ring depth Rust reported is what Python found"
    assert published == len(batch)


def test_the_default_runtime_publishes_no_ring_at_all(axon_binary, tmp_path):
    """An unconfigured session must create no file.

    The publisher creates and *truncates* its ring, so a default-on setting would let
    a bare ``cargo run --bin axon`` zero a ring another process was reading.
    """
    ring = str(tmp_path / "axon-md.ring")
    out = _run_axon(axon_binary, tmp_path, ring, enabled=False)
    # The status line's md counters, matched exactly rather than by the substring
    # "md " — the startup banner now names the ring on every run, publishing or not,
    # which is the entire point of the banner line.
    assert not re.search(
        r"\bmd \d+ q \d+/\d+", out
    ), f"the status line reported a publisher nobody asked for:\n{out}"
    assert "md ring    : OFF" in out, f"the banner must say so before the session:\n{out}"
    assert not Path(ring).exists(), "an unconfigured session created a ring file"
    assert not Path(bar_ring_path(ring)).exists(), "…nor a bar ring"
