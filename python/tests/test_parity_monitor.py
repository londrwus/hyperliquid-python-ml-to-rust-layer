"""``axon.parity.monitor``: the gates run forever, and what "forever" adds.

Each test is named after the failure mode it prevents, matching the convention in
``crates/axon-execution/src/tracker.rs``. Everything here is offline and
deterministic — a fake clock, fixed seeds, no sleep, no socket. The clock is fake
rather than patched because the silence deadline is the one thing in this module
that reads a wall clock at all (the absence of an event has no event time), and a
test that reached for ``time.sleep`` to exercise it would be both slow and flaky.

The two tests that carry the weight are
``test_a_window_that_compared_nothing_is_never_reported_as_ok`` and
``test_a_run_of_windows_that_compared_nothing_fails_the_whole_report``: the same
blindness the feature gate closes inside one comparison, at the level of the loop
that runs it. A monitor whose feed died and reported "OK, 0 rows compared" every
minute for a week is the failure this module exists to make impossible.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from axon.parity import (
    ParityError,
    aligned_feature_parity,
    quantile_binning,
)
from axon.parity.beacon import (
    MD_BEACON_FLAG_RUNNING,
    MD_BEACON_FLAG_STOPPED,
    MdBeaconSnapshot,
    Publisher,
)
from axon.parity.monitor import (
    DEFAULT_SILENCE_AFTER_NS,
    MAX_GAP_SAMPLES,
    GapStats,
    Level,
    MonitorConfig,
    MonitorReport,
    ParityMonitor,
    Verdict,
    Window,
    collecting_sink,
    logging_sink,
    run_monitor,
    windows_from_matrices,
)

COLS = ("ret_1", "vol_32", "book_imb")
BAR_NS = 3_600_000_000_000


class FakeClock:
    """A wall clock that only moves when a test says so."""

    def __init__(self, start: int = 1_700_000_000_000_000_000):
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, ns: int) -> None:
        self.now += ns


def stamps(n: int, *, start: int = 1_700_000_000_000_000_000) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * BAR_NS + start


def features(n: int, k: int = 3, *, seed: int = 7, loc: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc, 1.0, (n, k))


def healthy(n: int = 64, **kw) -> Window:
    ts, rows = stamps(n), features(n, **kw)
    return Window(online=rows.copy(), offline=rows, online_ts=ts, offline_ts=ts.copy())


def config(**kw) -> MonitorConfig:
    return MonitorConfig(columns=COLS, **kw)


# ── the window that compared nothing ─────────────────────────────────────────


def test_a_window_that_compared_nothing_is_never_reported_as_ok():
    # The whole reason this module exists, one level up from the alignment. A feed
    # that has gone dark produces windows that satisfy every threshold trivially,
    # and "OK, 0 rows compared" is the single most dangerous line a monitor can
    # print: it is indistinguishable from a healthy session in the one field an
    # operator reads.
    clock = FakeClock()
    monitor = ParityMonitor(config(), sink=None, clock=clock)
    verdict = monitor.observe(Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0)))
    assert verdict.level is Level.SILENT
    assert not verdict.passed
    assert "not a window that agreed" in verdict.summary()


def test_an_online_side_that_produced_nothing_is_silent_rather_than_perfect():
    # The offline recompute has rows and the serving path produced none. There is
    # nothing here that could have disagreed, which is not the same as agreeing.
    clock = FakeClock()
    monitor = ParityMonitor(config(), sink=None, clock=clock)
    ts, rows = stamps(32), features(32)
    verdict = monitor.observe(Window(np.empty((0, 3)), rows, stamps(0), ts))
    assert verdict.level is Level.SILENT
    assert verdict.n_compared == 0
    assert not monitor.report().passed


def test_silence_past_the_deadline_escalates_from_quiet_to_dead():
    # A quiet market and a dead feed produce the same empty window; only elapsed
    # time separates them, and only a wall clock measures elapsed time when the
    # event clock is the thing that stopped.
    clock = FakeClock()
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock)
    monitor.observe(healthy())
    assert monitor.heartbeat().level is Level.SILENT
    clock.advance(11_000_000_000)
    late = monitor.heartbeat()
    assert late.level is Level.ALARM
    assert "dead feed" in late.summary()


def test_a_run_of_windows_that_compared_nothing_fails_the_whole_report():
    # A monitor that ran for an hour and compared nothing has not proved a session
    # correct; it has proved that it was not looking. `passed` says so.
    report = run_monitor(
        [Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0))] * 5,
        config(),
        sink=None,
        clock=FakeClock(),
    )
    assert report.n_compared == 0
    assert not report.passed
    assert "saw nothing" in report.summary()
    with pytest.raises(ParityError, match="saw nothing"):
        report.raise_for_status()


def test_a_monitor_that_was_never_offered_a_window_does_not_report_a_pass():
    report = run_monitor([], config(), sink=None, clock=FakeClock())
    assert not report.passed
    assert "never ran" in report.summary()


# ── the parity half ──────────────────────────────────────────────────────────


def test_a_matching_session_reports_ok_and_says_how_many_rows_it_compared():
    report = run_monitor([healthy(), healthy(seed=8)], config(), sink=None, clock=FakeClock())
    assert report.passed
    assert report.n_compared == 128
    assert report.max_abs_diff == 0.0


def test_a_single_diverged_cell_alarms_rather_than_being_rated():
    # There is no acceptable *rate* of feature divergence: atol/rtol already absorb
    # the only legitimate difference between the two paths (summation order), so a
    # cell past tolerance is a bug and one of them is enough. Banding it the way
    # drift is banded would give a real skew a whole window to look normal in.
    window = healthy()
    online = window.online.copy()
    online[13, 2] += 1e-3
    diverged = Window(online, window.offline, window.online_ts, window.offline_ts)

    seen: list[Verdict] = []
    report = run_monitor([diverged], config(), sink=collecting_sink(seen), clock=FakeClock())
    assert not report.passed
    assert seen[0].level is Level.ALARM
    assert "book_imb" in seen[0].summary()


def test_a_half_blind_feed_alarms_even_though_every_compared_cell_agreed():
    # The headline number stays at its most reassuring value while the feed is at
    # its most broken. `max_abs_diff` is exactly 0.0 here — over half the rows.
    ts, rows = stamps(64), features(64)
    keep = np.arange(64) % 2 == 1
    window = Window(rows[keep], rows, ts[keep], ts)

    seen: list[Verdict] = []
    report = run_monitor([window], config(), sink=collecting_sink(seen), clock=FakeClock())
    assert report.max_abs_diff == 0.0
    assert not report.passed
    assert seen[0].level is Level.ALARM
    assert "owed row(s) the online side never produced" in seen[0].summary()


def test_two_sides_a_millisecond_apart_are_reported_as_a_stamp_bug_not_as_a_quiet_feed():
    # Hourly bars stamped `T` against hourly bars stamped `T + 1 ms` share no event
    # time at all, so the intersection is empty and every naive reading of that is
    # "nothing happened". The monitor says which of the two it is.
    ts, rows = stamps(32), features(32)
    window = Window(rows, rows, ts - 1_000_000, ts)

    seen: list[Verdict] = []
    run_monitor([window], config(), sink=collecting_sink(seen), clock=FakeClock())
    assert seen[0].level is Level.SILENT
    assert "T is the bar's last millisecond" in seen[0].summary()


# ── the monitor's own denominator ────────────────────────────────────────────


def test_a_run_blind_at_a_windows_opening_fails_instead_of_shrinking_its_denominator():
    # ADR-0030's founding bug, two levels out and inside the component built to
    # prevent it. A driver whose window reference *is* the owed rows hands the
    # monitor a window whose leading gap looks exactly like a late join — so the
    # rows are excused, and they leave `n_in_scope` on the way out. The run's own
    # denominator shrinks to fit the damage, which is the same shape as the gate
    # reporting a perfect zero over a fraction of the rows.
    ts, rows = stamps(64), features(64)
    blind_opening = Window(rows[20:], rows, ts[20:], ts)

    lenient = ParityMonitor(config(), sink=None, clock=FakeClock()).observe(blind_opening)
    assert lenient.level is Level.OK  # the default is right for a monitor window
    assert lenient.coverage.n_offline_before == 20
    assert lenient.coverage.n_in_scope == 44  # …and the 20 left the denominator

    declared = ParityMonitor(
        config(scope="declared"), sink=None, clock=FakeClock()
    ).observe(blind_opening)
    assert declared.level is Level.ALARM
    assert declared.coverage.n_offline_before == 0
    assert declared.coverage.n_offline_within == 20
    assert declared.coverage.n_in_scope == 64  # every row the driver said was owed


def test_the_monitor_and_a_gate_over_the_same_window_cannot_return_different_verdicts():
    # The reason to hold the scope on the config rather than let each driver
    # reinterpret the counts it gets back: two answers to one question is the shape
    # this codebase treats as a bug, and it is the shape that got a duplicate row
    # count deleted from the strategy gate in the first place.
    ts, rows = stamps(64), features(64)
    window = Window(rows[20:], rows, ts[20:], ts)
    for scope in ("observed", "declared"):
        monitor = ParityMonitor(config(scope=scope), sink=None, clock=FakeClock())
        verdict = monitor.observe(window)
        gate = aligned_feature_parity(
            window.online,
            window.offline,
            online_ts=window.online_ts,
            offline_ts=window.offline_ts,
            columns=COLS,
            scope=scope,
        )
        assert verdict.passed == gate.passed
        assert verdict.coverage == gate.coverage


def test_a_declared_window_that_compared_nothing_alarms_rather_than_waiting_out_a_deadline():
    # SILENT means "I cannot tell whether there was anything to compare". A declared
    # scope has already answered that, so waiting sixty seconds to say so would be
    # waiting for information already in hand.
    ts, rows = stamps(32), features(32)
    owed_nothing_produced = Window(np.empty((0, 3)), rows, stamps(0), ts)

    lenient = ParityMonitor(config(), sink=None, clock=FakeClock())
    lenient = lenient.observe(owed_nothing_produced)
    assert lenient.level is Level.SILENT  # it may simply not have been running yet

    declared = ParityMonitor(
        config(scope="declared"), sink=None, clock=FakeClock()
    ).observe(owed_nothing_produced)
    assert declared.level is Level.ALARM
    assert "declared 32 owed row(s) and compared none" in declared.summary()


def test_a_declared_window_that_owed_nothing_is_still_silence_and_not_an_alarm():
    # The counter-case, and the reason this is not just "declared means louder". A
    # window in which no bar closed owes nothing, and a serving path that produced
    # nothing for it is correct. Alarming would fire on every quiet window, which is
    # how an alarm gets muted — the failure mode `drift_ceiling` exists for.
    empty = Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0))
    monitor = ParityMonitor(config(scope="declared"), sink=None, clock=FakeClock())
    verdict = monitor.observe(empty)
    assert verdict.level is Level.SILENT
    assert not verdict.passed  # …but still never OK: nothing was proved either way


def test_a_typo_in_the_scope_is_refused_at_wiring_time_and_not_on_the_first_window():
    # A monitor is wired once and then runs. A scope the aligner only rejects inside
    # `observe` is an exception in the loop rather than a refusal before the session
    # starts — and the lenient mode is the one a typo must never select.
    with pytest.raises(ValueError, match="scope must be"):
        MonitorConfig(columns=COLS, scope="strict")


# ── the drift half ───────────────────────────────────────────────────────────


def training_sample(n: int = 4_000, *, loc: float = 0.0, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc, 1.0, (n, 3))


def drift_config(**kw) -> MonitorConfig:
    ref = training_sample()
    return MonitorConfig(
        columns=COLS,
        reference=ref,
        binnings=tuple(quantile_binning(ref[:, j]) for j in range(3)),
        min_drift_rows=100,
        **kw,
    )


def test_a_drift_band_must_hold_for_two_windows_before_it_alarms():
    # PSI over a few hundred live rows is noisy, and an alarm that fires on noise
    # gets muted — which is strictly worse than not having it, because a muted alarm
    # looks like a working one on every dashboard.
    moved = Window(
        online=training_sample(400, loc=3.0, seed=11),
        offline=training_sample(400, loc=3.0, seed=11),
        online_ts=stamps(400),
        offline_ts=stamps(400),
    )
    seen: list[Verdict] = []
    monitor = ParityMonitor(
        drift_config(confirm_windows=2, drift_ceiling=Level.ALARM),
        sink=collecting_sink(seen),
        clock=FakeClock(),
    )
    first = monitor.observe(moved)
    second = monitor.observe(moved)
    assert first.level is Level.WARN
    assert "1 of 2 before it counts" in first.summary()
    assert second.level is Level.ALARM
    assert "confirmed over 2 consecutive" in second.summary()


def test_drift_cannot_shout_over_the_parity_alarm_unless_it_is_asked_to():
    # Measured, not cautious: over `perp_bar`'s own 4,975-row history in 256-row
    # windows, PSI passes 0.25 on 18 of 20 windows while feature parity is green on
    # every one of them. Drift at ALARM by default would fire forever, and an
    # operator who learns to ignore ALARM has also learned to ignore the one alarm
    # that has never had a false positive.
    moved = Window(
        online=training_sample(400, loc=3.0, seed=11),
        offline=training_sample(400, loc=3.0, seed=11),
        online_ts=stamps(400),
        offline_ts=stamps(400),
    )
    monitor = ParityMonitor(drift_config(confirm_windows=1), sink=None, clock=FakeClock())
    assert monitor.observe(moved).level is Level.WARN  # not ALARM: the default ceiling
    assert monitor.observe(moved).level is Level.WARN
    assert monitor.report().passed  # WARN is a pass on notice, and drift is not a bug


def test_a_short_window_reports_that_drift_was_not_measured_rather_than_stable():
    # "Not measured" and "stable" are opposite statements, and only one of them is
    # true of a 20-row window. Reporting the wrong one teaches an operator that the
    # drift line is always green, which is how the line stops being read.
    small = Window(
        online=training_sample(20, loc=9.0, seed=5),
        offline=training_sample(20, loc=9.0, seed=5),
        online_ts=stamps(20),
        offline_ts=stamps(20),
    )
    monitor = ParityMonitor(drift_config(), sink=None, clock=FakeClock())
    verdict = monitor.observe(small)
    assert verdict.drift is None
    assert "drift not measured" in verdict.summary()
    assert verdict.level is Level.OK  # the parity half genuinely passed


def test_a_monitor_with_no_reference_sample_runs_parity_only_and_does_not_claim_drift():
    monitor = ParityMonitor(config(), sink=None, clock=FakeClock())
    verdict = monitor.observe(healthy(300))
    assert verdict.drift is None
    assert monitor.report().worst_psi == 0.0


def test_recomputed_bins_are_refused_by_construction_because_the_config_carries_frozen_ones():
    # ADR-0016 §5: quantile bins recomputed from each live window make both
    # histograms uniform and PSI reads ~0 whatever happened. The monitor cannot make
    # that mistake because it has nowhere to put a per-window binning — but a wrong
    # *number* of frozen bins is a real mistake, and it is refused.
    with pytest.raises(ValueError, match="binnings for"):
        MonitorConfig(columns=COLS, binnings=(quantile_binning(training_sample()[:, 0]),))


# ── the shape of the loop ────────────────────────────────────────────────────


def test_windowing_never_drops_an_online_row_between_two_windows():
    # A row that falls through the gap between two windows is dropped by the
    # windower itself — the same invisible denominator, one layer further out, and
    # invisible in exactly the same way.
    ts, rows = stamps(50), features(50)
    windows = list(windows_from_matrices(rows, rows, online_ts=ts, offline_ts=ts, size=7))
    assert sum(int(w.online_ts.size) for w in windows) == 50
    assert sum(int(w.offline_ts.size) for w in windows) == 50


def test_an_alarm_reaches_the_sink_and_an_ok_window_does_not():
    # The sink is the escalation seam. It is fed only what an operator has to act
    # on, because a sink that receives every window is a log nobody reads.
    seen: list[Verdict] = []
    monitor = ParityMonitor(config(), sink=collecting_sink(seen), clock=FakeClock())
    monitor.observe(healthy())
    assert seen == []
    monitor.observe(Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0)))
    assert len(seen) == 1


def test_the_default_alarm_logs_and_does_nothing_else():
    # Deliberate, and stated so it cannot be mistaken for an oversight: two
    # independent authorities that can stop trading, neither aware of the other, is
    # the shape ADR-0013's supervision loop exists to avoid — and this detector has
    # never seen a live session, so its false-positive rate is unmeasured.
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    log = logging.getLogger("axon.parity.monitor.test")
    log.addHandler(Capture())
    log.setLevel(logging.DEBUG)
    try:
        monitor = ParityMonitor(config(), sink=logging_sink(log), clock=FakeClock())
        monitor.observe(Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0)))
    finally:
        log.handlers.clear()
    assert [r.levelno for r in records] == [logging.WARNING]


def test_a_row_without_its_stamp_is_refused_rather_than_paired_by_position():
    # A row and its event time are one observation. Re-pairing them by position
    # after they have been separated is precisely the bug the alignment exists to
    # avoid, so a window that has lost the pairing cannot be built at all.
    with pytest.raises(ValueError, match="one observation"):
        Window(features(10), features(10), stamps(9), stamps(10))


# ── against something real ───────────────────────────────────────────────────


def perp_bar_windows(*, blind_every: int = 0):
    """The frozen candle fixture through the real ``PerpBar`` serving path.

    ``importorskip`` on the strategy package, not on numpy: the ladder's own
    dependencies are heavier than this module's, and a bare numpy environment must
    still run every test above.
    """
    pytest.importorskip("axon.strategies.training")
    from axon.parity.monitor import _perp_bar_windows

    return _perp_bar_windows("BTC", "1h", blind_every=blind_every)


def test_the_monitor_is_green_over_the_real_serving_path_it_will_watch():
    # Not a synthetic matrix: `PerpBar` driven one `on_bar` at a time through its
    # bounded buffer, against one vectorized recompute of the same history. This is
    # the closest thing to a live session that exists offline, and it is the run
    # that says the monitor's own machinery does not manufacture an alarm.
    windows, cfg = perp_bar_windows()
    report = run_monitor(windows, cfg, sink=None, clock=FakeClock())
    assert report.passed, report.summary()
    assert report.n_compared == 776
    assert report.max_abs_diff == 0.0


def test_the_monitor_reddens_over_the_same_path_when_the_serving_side_goes_half_blind():
    # A detector nobody has seen fire is a decoration. Same history, same code,
    # every third online row withheld — and `max_abs_diff` stays at exactly 0.0
    # while the verdict goes to ALARM, which is the entire point.
    windows, cfg = perp_bar_windows(blind_every=3)
    report = run_monitor(windows, cfg, sink=None, clock=FakeClock())
    assert report.max_abs_diff == 0.0
    assert not report.passed
    assert report.n_alarm == 7


def test_the_silence_deadline_default_is_stated_rather_than_derived():
    # It has not been measured against a live session and the docstring says so.
    # Pinned here so a change to it is a decision somebody made on purpose.
    assert DEFAULT_SILENCE_AFTER_NS == 60_000_000_000


# ── the beacon: what turns the deadline into an observation ──────────────────
#
# ADR-0030 §"the one seam this ADR specifies and does not build" names the ambiguity
# these close: under `MdWritePolicy::OnChange` a quiet top of book and a dead publisher
# write the same empty ring, and until there was a beacon the monitor could only
# resolve that with a timer. Every test below is a window that compared nothing — the
# question is only *what the monitor is entitled to say about it*.


class FakeBeacon:
    """A market-data beacon under the test's control.

    Deliberately not a file. The reader that maps one has its own tests
    (``test_beacon.py``, including against bytes ``md_writer`` really wrote); what is
    being checked here is the monitor's *response*, and that must not depend on a
    publisher happening to be running.
    """

    def __init__(self, beats: int = 100, **kw):
        self.snapshot = self._make(beats, **kw)
        self.error: Exception | None = None
        self.missing = False

    @staticmethod
    def _make(beats: int, **kw) -> MdBeaconSnapshot:
        base = dict(
            pid=777,
            beats=beats,
            last_event_ns=1_700_000_000_000_000_000,
            last_beat_ns=1_700_000_000_000_000_000,
            published=50,
            coalesced=50,
            dropped=0,
            bars_published=5,
            stale_quote=0,
            flags=MD_BEACON_FLAG_RUNNING,
        )
        base.update(kw)
        return MdBeaconSnapshot(**base)

    def advance(self, **kw) -> None:
        """Move the beacon on. Anything not named here stays exactly where it was —
        which is what makes 'the beat moved and nothing else did' expressible."""
        fields = dict(self.snapshot.__dict__)
        fields["beats"] = self.snapshot.beats + kw.pop("beats", 1)
        fields.update(kw)
        self.snapshot = MdBeaconSnapshot(**fields)

    def __call__(self) -> MdBeaconSnapshot | None:
        if self.error is not None:
            raise self.error
        return None if self.missing else self.snapshot


def blind_window() -> Window:
    return Window(np.empty((0, 3)), np.empty((0, 3)), stamps(0), stamps(0))


def test_a_dead_publisher_alarms_immediately_instead_of_waiting_out_the_deadline():
    # Sixty seconds exists to establish something the beacon has already established.
    # Waiting it out would be waiting for information in hand — the same sentence
    # ADR-0030 §1b applies to a declared scope that owed rows and compared none.
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(), sink=None, clock=clock, beacon=beacon)
    assert monitor.observe(blind_window()).level is Level.SILENT  # first reading only
    dead = monitor.observe(blind_window())  # the beat did not move
    assert dead.level is Level.ALARM
    assert dead.publisher.state is Publisher.DEAD
    assert clock.now == 1_700_000_000_000_000_000, "and no time had to pass for it"


def test_a_publisher_that_stopped_cleanly_alarms_with_different_words_than_a_crash():
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(), sink=None, clock=clock, beacon=beacon)
    monitor.observe(blind_window())
    beacon.snapshot = FakeBeacon._make(beacon.snapshot.beats, flags=MD_BEACON_FLAG_STOPPED)
    stopped = monitor.observe(blind_window())
    assert stopped.level is Level.ALARM
    assert stopped.publisher.state is Publisher.STOPPED
    assert "on purpose" in stopped.summary()


def test_a_quiet_publisher_does_not_escalate_however_long_the_market_stays_still():
    # The false positive the beacon exists to remove. Without it, a market that does
    # not move for a minute is indistinguishable from a corpse and the monitor alarms;
    # an operator taught to ignore ALARM by a quiet hour has been taught to ignore the
    # parity line, which is the same argument `drift_ceiling` is capped by.
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock, beacon=beacon)
    monitor.heartbeat()  # the first reading; a verdict about a counter needs two
    for _ in range(5):
        clock.advance(30_000_000_000)  # three times the deadline, every window
        beacon.advance(
            beats=250,
            coalesced=beacon.snapshot.coalesced + 40,
            last_event_ns=beacon.snapshot.last_event_ns + 1_000_000,
        )
        verdict = monitor.heartbeat()
    assert verdict.publisher.state is Publisher.QUIET
    assert verdict.level is Level.WARN, "quiet is a warning, and never becomes an alarm"
    assert "no fault here to alarm on" in verdict.summary()
    assert monitor.n_alarm == 0, "150 seconds against a 10 second deadline, and no alarm"
    # And the run still does not pass. The beacon downgraded the *level*; it says
    # nothing about whether anything was compared, and `passed` asks that separately.
    assert monitor.report().n_compared == 0
    assert not monitor.report().passed


def test_a_starved_publisher_still_escalates_but_the_alarm_names_the_cause():
    # Alive, beating, and its event clock frozen: nothing is reaching the core at all.
    # That is a fault, so the deadline still applies — the beacon's contribution is
    # that the alarm says *why* instead of guessing "dead feed".
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock, beacon=beacon)
    monitor.observe(healthy())
    beacon.advance(beats=5)
    early = monitor.heartbeat()
    assert early.publisher.state is Publisher.STARVED
    assert early.level is Level.SILENT, "one starved window is not yet a verdict"
    clock.advance(11_000_000_000)
    beacon.advance(beats=5)
    late = monitor.heartbeat()
    assert late.level is Level.ALARM
    assert "dead feed" in late.summary()
    assert "nothing is reaching the core" in late.summary()


def test_a_publisher_that_is_writing_records_does_not_excuse_a_window_that_compared_nothing():
    # The excuse that would have been easy to write: "the publisher is healthy, so the
    # silence is fine". Market data crossed the boundary and the comparison still
    # happened over zero rows, which is the invisible denominator wearing a beacon.
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock, beacon=beacon)
    monitor.observe(healthy())
    beacon.advance(beats=9, published=beacon.snapshot.published + 300, last_event_ns=beacon.snapshot.last_event_ns + 5_000_000)
    busy = monitor.heartbeat()
    assert busy.publisher.state is Publisher.PUBLISHING
    assert busy.level is Level.SILENT
    clock.advance(11_000_000_000)
    beacon.advance(beats=9, published=beacon.snapshot.published + 300, last_event_ns=beacon.snapshot.last_event_ns + 5_000_000)
    assert monitor.heartbeat().level is Level.ALARM, "the deadline still runs"


def test_the_beacons_verdict_never_lowers_an_alarm_a_declared_scope_already_raised():
    # Two independent reasons to alarm, and the quieter one must not win. A declared
    # window that owed rows and compared none is a blind serving path whatever the
    # publisher is doing — the beacon says nothing about the *consumer*.
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(config(scope="declared"), sink=None, clock=clock, beacon=beacon)
    ts, rows = stamps(32), features(32)
    monitor.observe(Window(np.empty((0, 3)), rows, stamps(0), ts))  # first beacon reading
    beacon.advance(beats=40, coalesced=beacon.snapshot.coalesced + 9, last_event_ns=beacon.snapshot.last_event_ns + 1)
    verdict = monitor.observe(Window(np.empty((0, 3)), rows, stamps(0), ts))
    assert verdict.publisher.state is Publisher.QUIET
    assert verdict.level is Level.ALARM, "owed rows nobody produced outrank a quiet market"


def test_a_beacon_that_cannot_be_read_is_reported_and_does_not_kill_the_monitor():
    # A live monitor must alarm rather than die. A beacon that raises — a stale mmap, a
    # permissions change, a path that turned into somebody else's file — becomes a
    # sentence in the report, and the deadline goes back to being the only evidence.
    clock = FakeClock()
    beacon = FakeBeacon()
    beacon.error = OSError("the sidecar went away")
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock, beacon=beacon)
    verdict = monitor.observe(blind_window())
    assert verdict.level is Level.SILENT
    assert verdict.publisher.state is Publisher.UNKNOWN
    assert "could not be read" in verdict.summary()
    clock.advance(11_000_000_000)
    assert monitor.heartbeat().level is Level.ALARM, "and the timer is back in charge"


def test_a_beacon_that_does_not_exist_yet_is_a_startup_race_and_not_an_alarm():
    # The monitor is wired before the session it watches starts. Escalating on "no file
    # there" would fire on every launch, and an alarm that fires on every launch has
    # taught its operator to close it.
    clock = FakeClock()
    beacon = FakeBeacon()
    beacon.missing = True
    monitor = ParityMonitor(config(), sink=None, clock=clock, beacon=beacon)
    verdict = monitor.observe(blind_window())
    assert verdict.level is Level.SILENT
    assert "nothing has created one" in verdict.summary()


def test_a_restarted_publisher_is_not_reported_as_a_beat_count_going_backwards():
    # A new session reinitialises the file and counts from zero. Read as a delta that
    # is a decrease, and the honest thing is to say the baseline is void rather than to
    # call the new publisher dead.
    clock = FakeClock()
    beacon = FakeBeacon(beats=900)
    monitor = ParityMonitor(config(), sink=None, clock=clock, beacon=beacon)
    monitor.observe(blind_window())
    beacon.snapshot = FakeBeacon._make(2, pid=778)
    verdict = monitor.observe(blind_window())
    assert verdict.publisher.state is Publisher.RESTARTED
    assert verdict.level is Level.SILENT, "a restart is not by itself a dead publisher"


# ── the deadline, measured rather than stated ────────────────────────────────
#
# ADR-0030 shipped `silence_after_ns` at sixty seconds and recorded it as "a starting
# value, not a measurement". Everything below is the evidence a measured one would have
# to be argued from, and — just as importantly — the guard that collecting it cannot
# move a verdict. Four false silence alarms were found and fixed on a perfectly healthy
# feed last session; a fifth introduced by the machinery that measures the deadline
# would be the worst possible one, because it would arrive wearing the authority of a
# measurement.


def test_the_silence_deadline_is_measured_by_every_run_rather_than_restated_by_one():
    # The quantity the deadline is actually judged against is the wall-clock gap
    # between two windows that COMPARED something, because that is what the progress
    # clock advances on. Pinned as numbers rather than as "some gaps were recorded":
    # a series that silently recorded the wrong quantity would satisfy every bound a
    # test could state about it, which is the same disease as an invisible denominator.
    clock = FakeClock()
    monitor = ParityMonitor(config(silence_after_ns=10_000_000_000), sink=None, clock=clock)
    for i, gap in enumerate((3_000_000_000, 9_000_000_000, 4_000_000_000)):
        monitor.observe(healthy(seed=i))
        clock.advance(gap)
    monitor.observe(healthy(seed=99))

    gaps = monitor.report().silence.between_compared_windows
    assert (gaps.n, gaps.min_ns, gaps.p50_ns, gaps.max_ns) == (
        3,
        3_000_000_000,
        4_000_000_000,
        9_000_000_000,
    )
    # And under the deadline this run actually ran with, not under the module default:
    # a description that quoted the default would compare the measurement against a
    # number nobody was using.
    assert monitor.report().silence.silence_after_ns == 10_000_000_000
    assert "10.000s deadline" in monitor.report().silence.describe()


def test_the_first_compared_window_does_not_contribute_a_gap_nobody_was_silent_for():
    # The progress clock is seeded on the first call whether or not anything was
    # compared, because a monitor whose source never delivers must still reach a
    # deadline. Differencing against that seed would enter a zero-second "silence" that
    # never happened, and a zero in a min or a p50 is a measurement rather than a
    # placeholder — it would drag the very quantiles a deadline gets argued from.
    monitor = ParityMonitor(config(), sink=None, clock=FakeClock())
    monitor.observe(healthy())
    assert monitor.report().silence.between_compared_windows.n == 0
    assert "no observation at all" in monitor.report().silence.describe()


def test_a_gap_between_windows_is_never_reported_without_the_cadence_that_bounds_it():
    # Six seconds between two compared windows means "the feed was quiet for six
    # seconds" only if somebody was asking throughout. Asked once at each end it means
    # nothing at all, and the two want opposite responses from whoever reads it. The
    # resolution series is what separates them, so it is collected on the same runs and
    # printed on the same line.
    clock = FakeClock()
    monitor = ParityMonitor(config(silence_after_ns=600_000_000_000), sink=None, clock=clock)
    monitor.observe(healthy())
    for _ in range(5):
        clock.advance(1_000_000_000)
        monitor.heartbeat()
    clock.advance(1_000_000_000)
    monitor.observe(healthy(seed=8))

    evidence = monitor.report().silence
    assert evidence.between_compared_windows.n == 1
    assert evidence.between_compared_windows.max_ns == 6_000_000_000
    assert evidence.between_observations.n == 6, "every ask, not only the ones that compared"
    assert evidence.between_observations.max_ns == 1_000_000_000
    text = evidence.describe()
    assert "between compared windows" in text
    assert "resolution" in text


def test_the_ring_advance_series_counts_only_the_readings_in_which_a_record_crossed():
    # The inter-record half, and the only one the beacon can supply: `published` and
    # `bars_published` moving between two readings is proof a record crossed the
    # boundary. A poll that found nothing moved is not an advance, and counting it as
    # one would make a still ring indistinguishable from a busy one.
    clock = FakeClock()
    beacon = FakeBeacon()
    monitor = ParityMonitor(
        config(silence_after_ns=600_000_000_000), sink=None, clock=clock, beacon=beacon
    )
    monitor.heartbeat()  # the first reading; a verdict about a counter needs two
    for published in (10, 0, 0, 7):
        clock.advance(1_000_000_000)
        beacon.advance(
            beats=20,
            published=beacon.snapshot.published + published,
            last_event_ns=beacon.snapshot.last_event_ns + 1_000_000,
        )
        monitor.heartbeat()

    evidence = monitor.report().silence
    # Two readings saw a record cross, so there is exactly one gap between them — and
    # it spans the two polls that found the ring still.
    assert evidence.between_ring_advances.n == 1
    assert evidence.between_ring_advances.max_ns == 3_000_000_000
    assert dict(evidence.publisher_states) == {"publishing": 2, "quiet": 2, "unknown": 1}
    assert evidence.diagnosed


def test_a_run_with_no_beacon_says_its_silence_was_uncategorised_rather_than_omitting_it():
    # The degradation has to be loud. A report that simply left the beacon line out
    # reads exactly like a report from a session whose publisher was proven alive, and
    # the whole contribution of ADR-0034 is the difference between those two.
    report = run_monitor([healthy()], config(), sink=None, clock=FakeClock())
    assert not report.silence.beacon_wired
    assert not report.silence.diagnosed
    assert "NO BEACON WAS WIRED" in report.summary()
    assert "UNCATEGORISED" in report.summary()


def test_a_beacon_that_only_ever_said_unknown_is_not_reported_as_a_diagnosis():
    # "A probe was wired" and "the probe answered" are different claims, and a run too
    # short for the two readings a counter needs makes the first true and the second
    # false. Printing only the first would credit the run with a diagnosis nobody got.
    beacon = FakeBeacon()
    beacon.missing = True
    monitor = ParityMonitor(config(), sink=None, clock=FakeClock(), beacon=beacon)
    monitor.observe(blind_window())
    monitor.observe(blind_window())

    evidence = monitor.report().silence
    assert evidence.beacon_wired
    assert not evidence.diagnosed
    assert "NEVER SAID ANYTHING BUT 'unknown'" in evidence.describe()


def test_recording_the_silence_evidence_cannot_move_a_single_verdict_it_records():
    # The guard that matters most. The evidence reads the same injected clock the
    # deadline reads and touches the same poll of the same beacon, so a bug here would
    # land as a level that changed — and a false alarm is the failure that gets a real
    # check deleted. Four were found on a healthy feed last session; a fifth arriving
    # from the machinery that measures the deadline would carry the authority of a
    # measurement with it. Pinned as a whole sequence, because the failure is a level
    # that moves in one place and nowhere else.
    clock = FakeClock()
    beacon = FakeBeacon()
    seen: list[Verdict] = []
    monitor = ParityMonitor(
        config(silence_after_ns=10_000_000_000),
        sink=None,
        clock=clock,
        beacon=beacon,
        tap=seen.append,
    )
    monitor.observe(healthy())  # OK, and the first beacon reading
    clock.advance(1_000_000_000)
    beacon.advance(beats=5)  # beating, event clock frozen: starved
    monitor.heartbeat()
    clock.advance(11_000_000_000)
    beacon.advance(beats=5)
    monitor.heartbeat()  # still starved, now past the deadline
    clock.advance(1_000_000_000)
    beacon.advance(
        beats=5,
        coalesced=beacon.snapshot.coalesced + 3,
        last_event_ns=beacon.snapshot.last_event_ns + 1,
    )
    monitor.heartbeat()  # alive and quiet: the one downgrade, and it is earned
    monitor.observe(healthy(seed=4))

    assert [v.level for v in seen] == [
        Level.OK,
        Level.SILENT,
        Level.ALARM,
        Level.WARN,
        Level.OK,
    ]
    # …and the run still measured itself while every one of those was decided.
    assert monitor.report().silence.between_compared_windows.n == 1
    assert monitor.report().silence.between_observations.n == 4


def test_a_tap_sees_every_window_and_the_sink_only_the_ones_worth_waking_someone_for():
    # Two seams, deliberately not one. A sink fed every window is a log nobody reads,
    # which is why `_emit` gates it — but a live harness printing only the bad windows
    # is indistinguishable from a harness that has stopped, which is the ambiguity the
    # whole module exists to remove. So the tap is fed everything and the sink is not.
    tapped: list[Verdict] = []
    alarmed: list[Verdict] = []
    monitor = ParityMonitor(
        config(), sink=collecting_sink(alarmed), clock=FakeClock(), tap=tapped.append
    )
    monitor.observe(healthy())
    monitor.observe(blind_window())
    assert [v.level for v in tapped] == [Level.OK, Level.SILENT]
    assert [v.level for v in alarmed] == [Level.SILENT]


def test_the_evidence_refuses_to_propose_the_deadline_it_just_measured():
    # A tolerance fitted to what today's run produced ratchets to today's run and
    # records the next regression as the new bar — the argument ADR-0021 made about
    # ONNX_TIGHT_EPS, and it is worse here, because a deadline is a detector and a
    # detector fitted to one healthy tape is calibrated never to fire. So the evidence
    # describes and does not suggest, and the constant does not move because a run
    # looked at it.
    clock = FakeClock()
    monitor = ParityMonitor(config(), sink=None, clock=clock)
    for i in range(4):
        monitor.observe(healthy(seed=i))
        clock.advance(3_000_000_000)

    text = monitor.report().silence.describe()
    assert "NOT a recommendation" in text
    assert "0030-live-parity-monitor-and-the-coverage-denominator.md" in text
    assert monitor.config.silence_after_ns == DEFAULT_SILENCE_AFTER_NS == 60_000_000_000


def test_a_percentile_is_an_observed_gap_and_never_an_interpolation_between_two():
    # A deadline argued from an interpolated p99 is a deadline argued from a
    # measurement nobody made. `method="nearest"` is what keeps every printed figure a
    # gap that actually occurred; the default linear interpolation would return 9.7
    # here, which is a duration this series never contained.
    stats = GapStats.of([1, 2, 3, 10])
    assert stats.p99_ns == 10
    assert stats.p50_ns in (2, 3)
    assert GapStats.of([]).n == 0
    assert "no observation at all" in GapStats.of([]).describe("nothing")


def test_a_gap_sample_that_stopped_recording_says_so_rather_than_quantifying_a_prefix():
    # Quantiles over a silently truncated sample, presented as the run's, are the
    # invisible-denominator bug wearing a histogram. Driven by pre-filling the series
    # rather than by MAX_GAP_SAMPLES real heartbeats: the behaviour under test is what
    # happens *at* the cap, and a test that spent ten seconds reaching it honestly is a
    # test that stops being run.
    clock = FakeClock()
    monitor = ParityMonitor(config(), sink=None, clock=clock)
    monitor._gaps_observed.extend([1] * MAX_GAP_SAMPLES)
    monitor.heartbeat()
    clock.advance(1_000_000_000)
    monitor.heartbeat()

    evidence = monitor.report().silence
    assert evidence.truncated
    assert evidence.between_observations.n == MAX_GAP_SAMPLES
    assert "stopped at" in evidence.describe()


def test_the_evidence_reaches_the_summary_of_a_run_that_compared_nothing():
    # `summary()` returns early on both degenerate runs, and those are exactly the runs
    # whose silence measurements are the whole story: how long was it quiet, and did the
    # publisher beat through it. A summary that dropped the evidence on the early
    # branches would drop it precisely where it was needed.
    saw_nothing = run_monitor([blind_window()] * 3, config(), sink=None, clock=FakeClock())
    assert "saw nothing" in saw_nothing.summary()
    assert "silence evidence" in saw_nothing.summary()

    never_ran = run_monitor([], config(), sink=None, clock=FakeClock())
    assert "never ran" in never_ran.summary()
    assert "silence evidence" in never_ran.summary()


def test_a_report_built_without_the_evidence_prints_nothing_about_silence_rather_than_zeros():
    # `MonitorReport` is constructed by hand in places that summarize other reports. A
    # missing measurement must not print as a measurement of zero — "no evidence" and
    # "no silence" are opposite statements, the same distinction the drift half draws
    # between "not measured" and "stable".
    bare = MonitorReport(
        windows=1,
        worst=Level.OK,
        n_compared=10,
        n_ok=1,
        n_alarm=0,
        n_silent=0,
        max_abs_diff=0.0,
        worst_psi=0.0,
        reasons=(),
    )
    assert bare.silence is None
    assert "silence evidence" not in bare.summary()


# ── the beacon a publisher created and never beat ────────────────────────────
#
# `MdBeacon::create` stamps the header — magic, pid, flags RUNNING — and leaves `beats`
# at zero; the counter is advanced by the *core's pass loop*, which is separate wiring.
# So a session with `[md_ring] enabled = true` and no pass-loop beat leaves a perfectly
# readable beacon frozen at beat 0 for as long as it runs, and that configuration
# existed in this tree while these tests were written. The two tests below pin what the
# monitor does with it and what the live harness does about it, in that order, because
# the first is correct and the second is what stops it from being a false alarm.


def test_a_beacon_frozen_at_beat_zero_alarms_and_that_is_why_the_harness_refuses_it():
    # The monitor is right: two readings of an unmoving beat ARE a pass loop that is not
    # running, and `publisher_state` says so in those words. What makes it dangerous is
    # the sentence an operator reads next — "dead publisher and not a quiet market" — on
    # a session whose bars are arriving perfectly, with no deadline left to wait out
    # because the beacon has supposedly already settled the question. Nothing in the
    # monitor can tell "never wired" from "died at once"; only something that watched
    # the counter before trusting it can, which is why `scripts/sessions/parity_live.py`
    # settles the probe rather than the monitor loosening this.
    clock = FakeClock()
    beacon = FakeBeacon(beats=0, published=0, bars_published=0, coalesced=0)
    monitor = ParityMonitor(config(), sink=None, clock=clock, beacon=beacon)
    monitor.observe(blind_window())  # the first reading; a counter needs two
    verdict = monitor.observe(blind_window())
    assert verdict.publisher.state is Publisher.DEAD
    assert verdict.level is Level.ALARM
    assert clock.now == 1_700_000_000_000_000_000, "and it did not wait for the deadline"


def test_the_live_harness_refuses_to_wire_a_beacon_that_was_created_and_never_beaten():
    # The guard, as a predicate over two readings, so it needs no file and no sleep.
    # Both directions are pinned because both are false alarms: refusing too eagerly
    # reports a genuinely dead publisher as "no beacon at all" and hands the run back to
    # a sixty-second timer that has just been told the answer.
    harness = _live_harness()
    running = FakeBeacon._make(0)
    assert harness.created_and_never_beaten(running, running), "beat 0 twice: never wired"

    beating = FakeBeacon._make(4_000)
    assert not harness.created_and_never_beaten(running, beating), "it started beating"
    assert not harness.created_and_never_beaten(beating, beating), (
        "frozen ABOVE zero is a publisher that beat and stopped — the one thing the "
        "probe exists to detect, and it must still be wired for it"
    )
    stopped = FakeBeacon._make(0, flags=MD_BEACON_FLAG_STOPPED)
    assert not harness.created_and_never_beaten(running, stopped), (
        "a session that created its beacon and exited before its first beat is STOPPED, "
        "which is a diagnosis and not a wiring fault to hide"
    )
    assert not harness.created_and_never_beaten(None, running)


def _live_harness():
    """``scripts/sessions/parity_live.py``, imported by path.

    Tested from here rather than from a test beside the script because the failure it
    prevents is a *monitor* false alarm, and this is the file a future reader of
    ``ParityMonitor(beacon=…)`` opens. The script is not importable as a package — it is
    deliberately outside ``python/`` so that ADR-0030 §7's promise that nothing in
    ``axon.parity`` can open a connection stays a property of the package rather than a
    habit.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "sessions" / "parity_live.py"
    if not path.is_file():  # pragma: no cover - the tree without the harness
        pytest.skip(f"no live parity harness at {path}")
    spec = importlib.util.spec_from_file_location("axon_parity_live_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
