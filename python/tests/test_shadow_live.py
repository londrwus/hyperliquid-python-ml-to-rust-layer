"""What a *live* shadow run does that a run over a file never could (ADR-0029).

Every test in ``test_strategies.py``'s shadow section drives
:class:`~axon.strategies.shadow.HistoryBarSource`, which is never idle, always warm
inside its first window, and always complete. Those three accidents hid three defects
until the first live m1 session was actually run, and each one of them presented as a
**failing report on a perfectly healthy feed** — the worst shape a false alarm can
take, because the next person turns the check off.

The tests here are named after those failure modes and they are all offline: the
faults are reproduced with a controllable clock, a narrow window and a stubbed venue
fetch, not with a socket. Nothing in this file touches the network — the live run that
found them is recorded in ADR-0029, not repeated in the gate.

The XGBoost-dependent path is not exercised here at all: the continuous diff compares
feature vectors and never looks at a score, so a constant predictor is the honest
stand-in and this file needs no ML extras.
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.features import FeatureDef, FeatureSpec, finite_rows
from axon.parity.monitor import Level
from axon.strategies import PERP_BAR_V1, Candles, PerpBar, PerpBarParams, fixture_candles
from axon.strategies.data import INTERVAL_MS
from axon.strategies.shadow import (
    HEARTBEAT_EVERY_S,
    SILENCE_AFTER_BARS,
    BarCoverage,
    ColumnDiff,
    HistoryBarSource,
    RingBarSource,
    ShadowTrader,
    VenueBarDiff,
    drive,
    publish_bars,
    reconcile_against_venue,
    volume_derived_columns,
)
from axon.strategy.events import Bar

INTERVAL = "1h"
BAR_NS = INTERVAL_MS[INTERVAL] * 1_000_000
SYMBOL = 3
OTHER = 4


class ConstantPredictor:
    """One probability for every row: the diff is about features, not scores."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def predict(self, x) -> np.ndarray:
        return np.full(np.asarray(x).shape[0], float(self.probability))

    def declared_schema(self):
        return None


class FakeClock:
    """A clock the test moves by hand.

    The silence deadline is the one thing in this module allowed to read a wall clock
    (ADR-0029 §4), so the only way to test it without sleeping through a deadline is
    to own the clock.
    """

    def __init__(self, now_ns: int = 1_000_000_000_000) -> None:
        self.now = int(now_ns)

    def __call__(self) -> int:
        return self.now

    def advance_s(self, seconds: float) -> None:
        self.now += int(seconds * 1e9)


def perp_bar(symbol_id: int = SYMBOL, probability: float = 0.5) -> PerpBar:
    return PerpBar(PerpBarParams(symbol_id=symbol_id), ConstantPredictor(probability))


def bar_at(index: int, candles: Candles, symbol_id: int = SYMBOL) -> Bar:
    return Bar(
        symbol_id=symbol_id,
        ts_event=int(candles.ts_event[index]),
        open=int(candles.open[index]),
        high=int(candles.high[index]),
        low=int(candles.low[index]),
        close=int(candles.close[index]),
        volume=int(candles.volume[index]),
    )


def warmup_bars(candles: Candles, spec: FeatureSpec = PERP_BAR_V1) -> int:
    """The index of the first bar the spec can produce a finite row for."""
    return int(np.argmax(finite_rows(spec.compute(candles.feature_inputs()))))


# ── 1. the silence deadline, against the clock that actually advances ────────


def test_a_quiet_gap_between_windows_is_not_graded_as_a_feed_that_died(tmp_path):
    """The monitor's clock advances per **window**; the deadline is in **bars**.

    A window is ``window_bars`` bars wide, so between two healthy flushes the monitor
    is legitimately silent for that many intervals — far past a deadline of 2.5 of
    them. Asked every second through that stretch it answers ``SILENT``, which sits
    *above* ``WARN``, so ``MonitorReport.passed`` goes false and the run reports FAIL
    with every cell it compared exactly right. Measured on the first live m1 session
    ever driven through this module: **296 SILENT verdicts in 160 s** of a feed
    delivering a bar a minute, before one window had closed.

    Nothing offline could see it: ``HistoryBarSource`` hands over its whole history
    without ever going idle, so the idle branch is dead code in every other test.
    """
    candles = fixture_candles("BTC", INTERVAL)
    clock = FakeClock()
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=64,
        clock=clock,
    ) as trader:
        trader.on_bar(bar_at(0, candles))

        # Most of a window's worth of wall time with no bar due yet: the poll loop
        # asks on every idle pass and must be told nothing, because there is nothing
        # to say. One question per second for the whole stretch is the shipped rate.
        asked = 0
        for _ in range(int(SILENCE_AFTER_BARS * INTERVAL_MS[INTERVAL] / 1000 / 2)):
            clock.advance_s(HEARTBEAT_EVERY_S)
            asked += trader.heartbeat() is not None
        assert asked == 0, "a quiet minute between bars is not a dead feed"
        assert trader.monitor.report().worst is Level.OK

        # And past the deadline the question does get asked, and the answer already
        # fails the run: SILENT outranks WARN. The monitor's own progress clock starts
        # at this first heartbeat, so the *word* ALARM costs one more deadline — which
        # is the price of not shouting it every second of a healthy session.
        clock.advance_s(SILENCE_AFTER_BARS * INTERVAL_MS[INTERVAL] / 1000)
        first = trader.heartbeat()
        assert first is not None and first.level is Level.SILENT
        assert not trader.monitor.report().passed

        clock.advance_s(SILENCE_AFTER_BARS * INTERVAL_MS[INTERVAL] / 1000 + 1)
        verdict = trader.heartbeat()
        assert verdict is not None and verdict.level is Level.ALARM
        assert "dead feed" in " ".join(verdict.reasons)


def test_an_instrument_with_nothing_to_print_is_not_a_dead_feed(tmp_path):
    """A venue that omits an empty minute makes a quiet instrument look like a dead one.

    Hyperliquid's ``candle`` subscription sends **no frame at all** for a minute in
    which nothing traded — 6 of 64 BTC minutes and 2 of 64 ETH minutes on the first
    live m1 run — and the publisher can only emit a bar when a frame for a later
    ``open_time`` arrives. So one instrument can go several intervals without a bar
    while the session is perfectly healthy, and a deadline armed only by *its own*
    bars alarms on a quiet market. It did: a live pass logged 15 consecutive ``SILENT``
    verdicts on a session that was delivering the other instrument's bars throughout.

    Another instrument's bar is proof the feed is alive; this instrument's silence is
    not evidence of anything. What that gives up — a single instrument's subscription
    stalling while another's flows — stays visible as a collapsed
    :class:`~axon.strategies.shadow.BarCoverage` rather than as an alarm.
    """
    candles = fixture_candles("BTC", INTERVAL)
    clock = FakeClock()
    deadline_s = SILENCE_AFTER_BARS * INTERVAL_MS[INTERVAL] / 1000
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        clock=clock,
    ) as trader:
        trader.on_bar(bar_at(0, candles))
        # Well past the deadline for *this* symbol, and the ring keeps delivering the
        # other one's bars the whole way.
        for i in range(1, 5):
            clock.advance_s(deadline_s / 2)
            trader.on_bar(bar_at(i, candles, symbol_id=OTHER))
            assert trader.heartbeat() is None

        assert trader.bars_seen == 1, "another instrument's bar must not be recorded"
        assert trader.monitor.report().worst is Level.OK

        # And when the whole feed stops, the deadline still fires.
        clock.advance_s(deadline_s + 1)
        assert trader.heartbeat() is not None


def test_a_source_that_never_delivers_a_first_bar_still_reaches_the_deadline(tmp_path):
    """The bar deadline runs from construction, not from the first bar.

    The obvious way to write the guard above is ``if a bar has arrived and it is
    recent, stay quiet`` — which is silent forever on the one feed that is
    unambiguously broken: the one that never delivered anything. A shadow run against
    a ring nobody is publishing to would then sit at ``OK`` until its idle timeout.
    """
    clock = FakeClock()
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        clock=clock,
    ) as trader:
        clock.advance_s(SILENCE_AFTER_BARS * INTERVAL_MS[INTERVAL] / 1000 + 1)
        verdict = trader.heartbeat()
    assert verdict is not None and verdict.level is not Level.OK


# ── 2. a window inside the warmup owes nothing ───────────────────────────────


def test_a_window_inside_the_spec_warmup_is_not_graded_as_a_window_that_saw_nothing(
    tmp_path,
):
    """``perp_bar`` warms up in 25 bars; a 12-bar window closes twice before that.

    Both of those windows owe nothing and produce nothing, and offering one to the
    monitor grades it ``SILENT`` — *"a window that compared nothing is not a window
    that agreed"*, which is true of a dead feed and false of a warmup. ``SILENT``
    outranks ``WARN``, so a healthy run fails on its own opening minutes.

    Invisible offline for an arithmetic reason: the shipped 64-bar window is wider
    than the 25-bar warmup, so every window in every history run has finite rows in
    it. It is only reachable once someone picks a window narrow enough to see a live
    session flush more than twice — which is exactly what an m1 run has to do.
    """
    candles = fixture_candles("BTC", INTERVAL)
    warm = warmup_bars(candles)
    window = 12
    assert window * 2 <= warm, "the fixture must warm up slower than two windows"

    source = HistoryBarSource(candles.head(window * 4), symbol_id=SYMBOL)
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=window,
    ) as trader:
        report = trader.run(source)

    assert report.warmup_windows == warm // window
    assert report.monitor.worst is Level.OK
    assert report.passed
    assert report.rows_compared == report.rows_owed > 0
    assert "fell entirely inside the spec's warmup" in report.summary()


def test_a_serving_path_emitting_rows_the_reference_lacks_still_reaches_the_monitor(
    tmp_path,
):
    """The warmup skip is ``both sides empty``, and the second half is load-bearing.

    Skipping on ``the reference owed nothing`` alone would throw away the one window
    where a serving path inventing rows out of its warmup is visible: the reference
    has nothing at those stamps, so every row lands in ``online_unmatched``, which is
    never legitimate — it means the two are looking at different events.
    """
    candles = fixture_candles("BTC", INTERVAL)
    window = 12

    class EagerPerpBar(PerpBar):
        """Produces a row from its very first bar, warm or not."""

        def feature_row(self):
            row = super().feature_row()
            return np.zeros(len(PERP_BAR_V1.columns)) if row is None else row

    from axon.parity.monitor import collecting_sink

    verdicts: list = []
    source = HistoryBarSource(candles.head(window), symbol_id=SYMBOL)
    with ShadowTrader(
        EagerPerpBar(PerpBarParams(symbol_id=SYMBOL), ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=window,
    ) as trader:
        trader.monitor.sink = collecting_sink(verdicts)
        report = trader.run(source)

    assert report.warmup_windows == 0, "a window with online rows was skipped"
    assert not report.passed
    assert report.monitor.windows == 1, "the window never reached the monitor"
    assert report.monitor.worst is not Level.OK
    assert verdicts[0].coverage.n_online_unmatched == window
    assert "the reference has nothing at" in verdicts[0].summary()


# ── 3. the denominator below the row denominator ─────────────────────────────


def test_bars_that_never_arrived_are_counted_against_the_cadence_not_the_arrivals(
    tmp_path,
):
    """``coverage=n/n`` and ``max_abs_diff=0`` on a feed missing half its bars.

    This is the invisible-denominator bug one level below the one ADR-0030 closed.
    The offline reference is a recompute over *the bars the serving path was shown*,
    so a bar that never arrived reaches neither side: both agree perfectly, the row
    ratio is 1.0, and every field a reader looks at is identical to a healthy run.
    The only witness is the bars' own close times — a cadence with holes in it.
    """
    candles = fixture_candles("BTC", INTERVAL).head(200)
    kept = np.ones(len(candles), dtype=bool)
    kept[100:140] = False  # forty hours the strategy is simply never told about
    holed = Candles(
        coin=candles.coin,
        interval=candles.interval,
        **{
            name: getattr(candles, name)[kept]
            for name in ("ts_event", "open", "high", "low", "close", "volume")
        },
    )

    source = HistoryBarSource(holed, symbol_id=SYMBOL)
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=32,
    ) as trader:
        report = trader.run(source)

    # Everything a reader normally looks at says the run was flawless.
    assert report.monitor.max_abs_diff == 0.0
    assert report.rows_compared == report.rows_owed
    assert report.passed

    # And the bar cadence says forty of them are missing.
    assert report.bar_coverage.delivered == 160
    assert report.bar_coverage.expected == 200
    assert report.bar_coverage.missing == 40
    assert not report.bar_coverage.complete
    assert "never reached the strategy" in report.summary()


def test_a_complete_cadence_reports_its_denominator_rather_than_staying_quiet(tmp_path):
    # The other half of the same rule: a number that only appears when it is bad is a
    # number nobody has calibrated, so the healthy case prints it too.
    candles = fixture_candles("BTC", INTERVAL).head(120)
    source = HistoryBarSource(candles, symbol_id=SYMBOL)
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=32,
    ) as trader:
        report = trader.run(source)
    assert report.bar_coverage == BarCoverage(
        delivered=120,
        expected=120,
        interval_ms=INTERVAL_MS[INTERVAL],
        first_ts=int(candles.ts_event[0]),
        last_ts=int(candles.ts_event[-1]),
    )
    assert "bar coverage: 120/120" in report.summary()


# ── 4. which columns volume can move ─────────────────────────────────────────


def test_a_volume_column_is_identified_from_the_spec_and_never_from_a_kept_list():
    """ADR-0028's whole reading depends on this split being right.

    A published bar is the venue's last *observed* frame, so a trade printing in the
    final milliseconds is missing from ``v`` and from nothing else — a non-zero volume
    column is the venue, a non-zero price column is a parity break. A list of volume
    columns maintained beside the spec gets that backwards the first time somebody
    adds a feature reading **both**: it would be filed under price, and a real break
    in it would be rated as venue behaviour.
    """
    assert volume_derived_columns(PERP_BAR_V1) == ("vol_z_24",)

    # A column bound to an *earlier column* rather than to a bar field: the binding is
    # transitive and a one-level scan would call this one a price column.
    chained = FeatureSpec(
        name="chained",
        version=1,
        features=(
            PERP_BAR_V1.features[-1],  # vol_z_24, reads volume
            FeatureDef("mom_of_vol_z", "momentum", params={"window": 3},
                       inputs={"price": "vol_z_24"}),
            FeatureDef("ret_1", "log_return", params={"period": 1}, inputs={"price": "close"}),
        ),
    )
    assert volume_derived_columns(chained) == ("vol_z_24", "mom_of_vol_z")


def test_a_nan_against_a_number_is_the_largest_diff_in_its_column_and_not_the_smallest(
    tmp_path,
):
    """A serving path emitting NaN where the recompute has a value is the loudest
    break there is, and the two obvious reductions both hide it: ``np.nanmax`` erases
    it completely and reports the column's largest *finite* difference, while
    ``np.max`` propagates one NaN into a number that sorts below every real one.
    ``inf`` keeps it the largest value in its own column and in no other.
    """
    candles = fixture_candles("BTC", INTERVAL).head(80)

    class NanInOneColumn(PerpBar):
        """Healthy in eight columns and NaN in the ninth — a defensive clamp, a
        stricter live guard, a backend returning NaN on inputs Python accepts."""

        def feature_row(self):
            row = super().feature_row()
            if row is None:
                return None
            row = np.asarray(row, dtype=np.float64).copy()
            row[0] = np.nan
            return row

    source = HistoryBarSource(candles, symbol_id=SYMBOL)
    with ShadowTrader(
        NanInOneColumn(PerpBarParams(symbol_id=SYMBOL), ConstantPredictor(0.5)),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        window_bars=32,
    ) as trader:
        report = trader.run(source)

    assert not report.passed
    per_column = dict(zip(report.columns.columns, report.columns.max_abs_diff))
    assert per_column[PERP_BAR_V1.columns[0]] == float("inf")
    assert all(v == 0.0 for c, v in per_column.items() if c != PERP_BAR_V1.columns[0])
    assert report.columns.worst_price == (PERP_BAR_V1.columns[0], float("inf"))
    assert "=inf" in report.summary()


def test_the_price_and_volume_halves_of_a_column_diff_are_read_off_the_spec():
    # The split a reader acts on, kept honest independently of how it is computed:
    # `worst_volume` must never report a price column and vice versa, whatever the
    # magnitudes are.
    diff = ColumnDiff(
        columns=("a", "b"),
        max_abs_diff=(1e-9, float("inf")),
        volume_columns=frozenset({"b"}),
        n_rows=10,
    )
    assert diff.price_columns == ("a",)
    assert diff.worst_price == ("a", 1e-9)
    assert diff.worst_volume == ("b", float("inf"))
    assert "b*=inf" in diff.describe()


# ── 5. the published bar against the venue's own ─────────────────────────────


def _venue_rows(candles: Candles, **edits) -> list[dict]:
    """``candleSnapshot`` rows echoing a candle history, with named fields edited."""
    from axon.strategies.data import CLOSE_STAMP_OFFSET_MS
    from axon.contracts import FIXED_POINT_SCALE

    def unfix(v: int) -> str:
        return f"{v / FIXED_POINT_SCALE:.8f}"

    rows = []
    for i in range(len(candles)):
        row = {
            "T": int(candles.ts_event[i] // 1_000_000) - CLOSE_STAMP_OFFSET_MS,
            "o": unfix(int(candles.open[i])),
            "h": unfix(int(candles.high[i])),
            "l": unfix(int(candles.low[i])),
            "c": unfix(int(candles.close[i])),
            "v": unfix(int(candles.volume[i])),
        }
        for key, per_index in edits.items():
            if i in per_index:
                row[key] = unfix(per_index[i])
        rows.append(row)
    return rows


def _recorded(candles: Candles) -> np.ndarray:
    return np.column_stack(
        [candles.ts_event, candles.open, candles.high, candles.low, candles.close, candles.volume]
    ).astype(np.int64)


def test_a_bar_short_on_volume_is_the_venue_and_a_bar_with_a_moved_price_is_a_break(
    monkeypatch,
):
    """The one classification ADR-0028 asks a live run to make, and both directions.

    Hyperliquid sends no closing frame, so the published bar is the last frame
    observed before the close and a trade printing after it is missing from ``v``.
    That is expected. ``o/h/l/c`` cannot move for the same reason — a missing trade
    that moved the price would have moved volume too — so a price field that
    disagrees is a parity break and this asserts it is *named* one rather than rated.
    """
    from axon.strategies import data as data_module

    candles = fixture_candles("BTC", INTERVAL).head(60)

    # The venue's answer: two bars carry a little more volume than the ring published,
    # and nothing else differs. That is the shape M12 measured live.
    fatter = {7: int(candles.volume[7]) + 1000, 23: int(candles.volume[23]) + 5}
    monkeypatch.setattr(
        data_module,
        "fetch_candles",
        lambda *a, **k: _venue_rows(candles, v=fatter),
    )
    diff = reconcile_against_venue(_recorded(candles), coin="BTC", interval=INTERVAL)

    assert diff.n_matched == 60
    assert diff.price_break is False
    assert diff.volume_only_mismatched_bars == 2
    assert diff.field_mismatched_bars["volume"] == 2
    assert diff.field_max_abs["volume"] == 1000
    assert all(diff.field_mismatched_bars[f] == 0 for f in diff.price_fields)
    assert "no price field disagreed" in diff.describe()
    # And the cost of two short volumes is a *volume* column, over the whole window
    # each one sits in — which is the thing a bar-level "short by 0.001" does not say.
    assert diff.columns is not None
    assert diff.columns.worst_price[1] == 0.0
    assert diff.columns.worst_volume[0] == "vol_z_24"
    assert diff.columns.worst_volume[1] > 0.0

    # Now move a close. Nothing about the venue's missing tail can produce this.
    monkeypatch.setattr(
        data_module,
        "fetch_candles",
        lambda *a, **k: _venue_rows(candles, c={30: int(candles.close[30]) + 100}),
    )
    broken = reconcile_against_venue(_recorded(candles), coin="BTC", interval=INTERVAL)
    assert broken.price_break is True
    assert broken.price_mismatched_bars == 1
    assert broken.volume_only_mismatched_bars == 0
    assert "PARITY BREAK" in broken.describe()
    assert broken.columns.worst_price[1] > 0.0


def test_a_bar_the_venue_lists_and_the_ring_never_published_is_counted(monkeypatch):
    # The bar-level denominator from the other side. The ring's own `gap_before` flag
    # marks *one* bar after a hole however wide the hole is, so a run that lost twenty
    # minutes of feed reports one gap; the venue's own listing is the only place the
    # width of it is written down.
    from axon.strategies import data as data_module

    candles = fixture_candles("BTC", INTERVAL).head(40)
    kept = np.ones(len(candles), dtype=bool)
    kept[10:15] = False
    ring = _recorded(candles)[kept]
    monkeypatch.setattr(data_module, "fetch_candles", lambda *a, **k: _venue_rows(candles))

    diff = reconcile_against_venue(ring, coin="BTC", interval=INTERVAL)
    assert diff.n_ring == 35 and diff.n_venue == 40 and diff.n_matched == 35
    assert len(diff.unmatched_venue_ts) == 5
    assert diff.price_break is False
    assert "never reached the ring" in diff.describe()


def test_a_venue_reconciliation_with_nothing_recorded_refuses_rather_than_agreeing():
    # `0 matched of 0` would satisfy every clause in `price_break` and print a clean
    # bill of health for a run that observed nothing.
    from axon.strategies.shadow import ShadowError

    with pytest.raises(ShadowError, match="nothing to reconcile"):
        reconcile_against_venue(np.empty((0, 6), dtype=np.int64), coin="BTC", interval=INTERVAL)


# ── 6. one reader, several strategies ────────────────────────────────────────


def test_two_traders_share_one_reader_because_two_consumers_would_split_the_bars(tmp_path):
    """A ring's read cursor lives in its own header.

    Two :class:`~axon.marketdata.MdBarRingConsumer` instances on one bar ring each pop
    a share of the records, and **neither can tell that from a quiet feed** — so a
    session shadowing two instruments by running two readers silently diffs half a
    tape against a recompute over that half, and reports a flawless zero on both.
    :func:`~axon.strategies.shadow.drive` fans out from one reader instead.
    """
    from axon.marketdata import bar_ring_path

    candles = fixture_candles("BTC", INTERVAL).head(40)
    ring = bar_ring_path(str(tmp_path / "axon-md.ring"))
    publish_bars(candles, ring, symbol_id=SYMBOL, capacity=128)

    traders = [
        ShadowTrader(
            perp_bar(sid),
            symbol_id=sid,
            ring_path=str(tmp_path / f"shadow-{sid}.ring"),
            interval=INTERVAL,
            window_bars=16,
        )
        for sid in (SYMBOL, OTHER)
    ]
    try:
        with RingBarSource(_consumer(ring), symbol_id=None) as source:
            # `max_bars`, because a ring is never `exhausted` — a live one cannot be,
            # and this one is a real ring. Without a bound this test is the hang it is
            # asserting against.
            reports = drive(traders, source, max_bars=len(candles), idle_timeout_s=5.0)
    finally:
        for t in traders:
            t.close()

    # Every bar reached both traders; only the one whose symbol it is recorded it.
    assert reports[0].bars == 40
    assert reports[1].bars == 0
    assert traders[0].bars_seen == 40 and traders[1].bars_seen == 0
    # And the second trader's event clock advanced on the first's bars, which is the
    # property ADR-0029 §5 asks for: a live session's runner sees every instrument.
    assert traders[1]._runner.last_event_ns == int(candles.ts_event[-1])


def _consumer(path: str):
    from axon.marketdata import MdBarRingConsumer

    return MdBarRingConsumer(path)


# ── 6b. a would-be order the real intent path would have thrown away ─────────


def test_a_bar_stamped_signal_is_born_as_old_as_the_bar_took_to_arrive(tmp_path):
    """A shadow run cannot refuse a signal, so it must at least say how old one is.

    ``StrategyRunner`` stamps a signal with the **event's own** ``ts_event`` and never
    with a wall clock, so a bar strategy's signal carries the bar's *close* — and the
    bar cannot arrive before it. The signal reader then judges that stamp against
    ``CoreHandler::last_ts()``, which agent P1 measured running 1 564 ms behind wall
    time, with a 2 000 ms admission ceiling above it. P1's first live run refused its
    own second target: ``accepted: 1, expired: 1``.

    Nothing in a shadow run is admitted or refused, which is precisely the gap: the
    harness will happily report would-be orders a live session would have dropped, and
    a transcript of those is a transcript of orders that never could have been placed.
    """
    candles = fixture_candles("BTC", INTERVAL).head(4)
    clock = FakeClock(now_ns=int(candles.ts_event[0]))
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
        clock=clock,
    ) as trader:
        # Bar 0 lands 300 ms after its own close — a healthy publisher, one venue
        # frame late. Bar 1 lands four seconds late.
        clock.now = int(candles.ts_event[0]) + 300_000_000
        trader.on_bar(bar_at(0, candles))
        clock.now = int(candles.ts_event[1]) + 4_000_000_000
        trader.on_bar(bar_at(1, candles))
        lag = trader.arrival_lag()

    assert lag.n == 2
    assert lag.min_ms == pytest.approx(300.0)
    assert lag.max_ms == pytest.approx(4000.0)

    # The healthy one is *ahead* of the core's clock — the feed's lag and the core's
    # partially cancel — and the reader admits a negative age as `ahead_of_clock`.
    ages = lag.ages_ms(core_clock_lag_ms=1564)
    assert ages[0] < 0
    # The late one is not, and it is over the ceiling by the margin that matters.
    assert ages[1] == pytest.approx(4000 - 1564)
    assert lag.refused(ceiling_ms=2000, core_clock_lag_ms=1564) == 1
    assert "would be refused as Expired" in lag.describe()
    assert "never reaches the signal reader" in lag.describe()


def test_a_would_be_signals_stamp_is_a_bar_close_and_not_the_moment_it_was_made(tmp_path):
    """The link the whole admission argument rests on, asserted rather than assumed.

    If the runner stamped a signal with the moment it was emitted, its age at the pass
    would be the pass delay and nothing else, and a bar interval would not enter the
    arithmetic at all. It does not: ``StrategyRunner`` stamps the **event's own**
    ``ts_event`` (``runner.py``: *"It never stamps a signal with a wall clock"*), so a
    bar strategy's signal carries a bar's close, and :class:`ArrivalLag` is measuring
    the right quantity.
    """
    from axon.strategies.shadow import shadow_history

    candles = fixture_candles("BTC", INTERVAL).head(120)
    report = shadow_history(
        perp_bar(probability=0.9),
        candles,
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        window_bars=32,
    )
    closes = set(int(t) for t in candles.ts_event)
    assert report.signals
    assert all(s.ts_event in closes for s in report.signals)


def test_an_unattested_run_reports_no_signal_age_instead_of_the_age_of_a_file(tmp_path):
    """An arrival lag over a recording is wall-clock-now minus a close from last week.

    Printed, it says *"876 would be refused as Expired"* on a flawless offline run —
    a false alarm of precisely the kind this module keeps finding and removing.
    Liveness is not inferable here (ADR-0029 §6), so neither is the meaning of the
    subtraction, and the report gives the reason rather than the number.
    """
    from axon.strategies.shadow import shadow_history

    candles = fixture_candles("BTC", INTERVAL).head(120)
    report = shadow_history(
        perp_bar(),
        candles,
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        window_bars=32,
    )
    assert not report.venue_attested
    assert report.arrival_lag.n == 120  # measured, and deliberately not printed
    assert "signal age: not measured" in report.summary()
    assert "Expired" not in report.summary()


def test_a_run_with_no_bars_ages_no_signal_rather_than_reporting_a_zero_lag(tmp_path):
    # `min/median/max = 0 ms` on a run that saw nothing reads as a perfect feed.
    with ShadowTrader(
        perp_bar(),
        symbol_id=SYMBOL,
        ring_path=str(tmp_path / "shadow.ring"),
        interval=INTERVAL,
    ) as trader:
        lag = trader.arrival_lag()
    assert lag.n == 0
    assert "no bar arrived" in lag.describe()


# ── 7. the CLI is not hardwired to one family ────────────────────────────────


def test_a_strategy_family_is_diffed_against_its_own_spec_and_not_perp_bars(
    tmp_path, capsys
):
    """``--strategy`` exists so a zoo lands without editing this module, and the spec
    has to travel with the strategy.

    A trader diffing ``perp_bar``'s nine columns against a family serving two would
    align on event time, find every stamp present on both sides, and compare the wrong
    arrays — a shape mismatch at best and a silent column-by-column mismatch at worst.
    Reading ``spec`` off the strategy is what makes the escape hatch safe.
    """
    pytest.importorskip("axon.strategies.baseline")
    from axon.strategies.shadow import main

    code = main(
        [
            "--coin", "BTC", "--interval", INTERVAL,
            "--strategy", "baseline",
            "--symbol-id", str(SYMBOL),
            "--max-position", "0.0003",
            "--ring", str(tmp_path / "shadow.ring"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "baseline_z" in out and "perp_bar/v1" not in out
    # And the spec's own columns, not perp_bar's nine.
    assert "vol_z_24" not in out


def test_a_registry_handed_to_a_strategy_with_no_artifact_is_refused_not_ignored(
    tmp_path,
):
    # A `--registry` silently unused is a run that reports on a model it never loaded,
    # which is indistinguishable in the transcript from one that did.
    pytest.importorskip("axon.strategies.baseline")
    from axon.strategies.shadow import ShadowError, main

    with pytest.raises(ShadowError, match="takes no model"):
        main(
            [
                "--coin", "BTC", "--interval", INTERVAL,
                "--strategy", "baseline",
                "--registry", str(tmp_path),
                "--ring", str(tmp_path / "shadow.ring"),
            ]
        )


def test_one_max_position_per_symbol_because_a_reused_size_is_a_meaningless_notional(
    tmp_path,
):
    from axon.strategies.shadow import main

    with pytest.raises(SystemExit):
        main(
            [
                "--bar-ring", str(tmp_path / "md.ring"),
                "--symbol-id", "3", "--symbol-id", "4",
                "--max-position", "0.0003",
                "--ring", str(tmp_path / "shadow.ring"),
            ]
        )


def test_an_unknown_strategy_name_says_what_exists_instead_of_importing_nothing():
    from axon.strategies.shadow import ShadowError, _strategy_factory

    with pytest.raises(ShadowError, match="built-ins are"):
        _strategy_factory("perpbar")  # a plausible typo, not a dotted path


def test_drive_refuses_an_empty_trader_list_rather_than_looping_over_nothing(tmp_path):
    # A driver with no traders polls a live source forever and reports nothing, which
    # reads as a session that is running.
    from axon.strategies.shadow import ShadowError

    candles = fixture_candles("BTC", INTERVAL).head(4)
    with pytest.raises(ShadowError, match="at least one trader"):
        drive([], HistoryBarSource(candles, symbol_id=SYMBOL))
