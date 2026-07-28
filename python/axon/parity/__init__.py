"""``axon.parity`` — the three gates that turn "no quality loss" into a test.

``docs/07-parity-and-testing.md`` says it plainly: no quality loss is a *claim*
until it is a *gate*. This package is the gate, in three parts (ADR-0016):

* :mod:`axon.parity.model` — **model parity.** Reference vs. candidate predictions:
  exact for trees, ``max_abs_diff < eps`` for neural nets, and — the check that
  actually protects P&L — no discretized trading decision may flip, with the
  offending inputs named.
* :mod:`axon.parity.features` — **feature parity.** Online feature vectors against
  the offline recompute, reporting the offending column and row. ``docs/03`` calls
  this the hard one: it is where training–serving skew hides.
* :mod:`axon.parity.drift` — **drift.** PSI and KL per feature with the conventional
  bands, for the monitor that runs forever in production and catches the failure the
  other two cannot: the code is right and the market moved.

And the loop that runs them rather than the gates themselves:

* :mod:`axon.parity.monitor` — **the live parity monitor** (ADR-0030). The same
  three gates over a session in flight, with the two things a CI run does not need:
  state across windows, and a verdict for a window that compared *nothing*.
* :mod:`axon.parity.beacon` — **the market-data beacon's reader.** The monitor can
  see that nothing was compared; only this can say whether that is a quiet market or
  a dead publisher, because under ``MdWritePolicy::OnChange`` an empty ring is
  ambiguous by design. The Rust core's pass loop writes a 64-byte sidecar beside the
  ring (``crates/axon-ipc/src/beacon.rs``) that advances whether or not anything
  arrived; this reads it and turns the monitor's deadline into an observation.

Every gate returns a report with ``passed``, ``summary()`` and
``raise_for_status()``, so the same call is a CI assertion and a monitoring alarm.

::

    from axon.features import PERP_CORE_V1
    from axon.parity import aligned_feature_parity, model_parity, threshold_discretizer

    decide = threshold_discretizer(long_at=0.55, short_at=0.45)
    model_parity(reference_scores, rust_scores, discretizer=decide).raise_for_status()
    aligned_feature_parity(
        online, offline,
        online_ts=online_ts, offline_ts=offline_ts,
        columns=PERP_CORE_V1.columns,
    ).raise_for_status()

Prefer :func:`~axon.parity.aligned_feature_parity` over ``align_by_event_time`` +
``feature_parity``: the two-step form lets a caller keep the matched rows and throw
away the record of what did *not* match, and what did not match is the half of the
verdict that fails silently and green (ADR-0030).
"""

from __future__ import annotations

from axon.parity.beacon import (
    MD_BEACON_SIZE,
    BeaconError,
    MdBeaconReader,
    MdBeaconSnapshot,
    Publisher,
    PublisherState,
    beacon_path,
    publisher_state,
    read_md_beacon,
)
from axon.parity.drift import (
    DEFAULT_NAN_RATE_TOL,
    PSI_MODERATE,
    PSI_STABLE,
    Binning,
    DriftReport,
    FeatureDrift,
    drift_report,
    kl_divergence,
    psi,
    psi_band,
    quantile_binning,
)
from axon.parity.features import (
    FEATURE_ATOL,
    FEATURE_RTOL,
    Alignment,
    Cell,
    Coverage,
    FeatureParityReport,
    align_by_event_time,
    aligned_feature_parity,
    feature_parity,
)
from axon.parity.gate import GateReport, ParityError

# The monitor is the loop, not a fourth gate; it lives beside them because
# ``docs/03`` calls it the mandatory backstop rather than an optional extra. Nothing
# it imports at module scope is heavier than numpy.
from axon.parity.monitor import (
    DEFAULT_SILENCE_AFTER_NS,
    AlarmSink,
    BeaconProbe,
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
from axon.parity.model import (
    NN_EPS,
    TREE_EPS,
    Discretizer,
    Flip,
    ModelParityReport,
    decision_flips,
    decision_invariant,
    model_parity,
    threshold_discretizer,
)

# The other half of that boundary: the same freeze applied to the *feature* matrix
# rather than to a model. A model bundle proves that identical feature vectors produce
# identical decisions; this one asks whether the two languages compute identical vectors
# from the same market data, which ``docs/03`` calls the harder half and which no
# Python-to-Python gate can fail on. Neither writing nor reading it needs an ML library —
# numpy and the standard library are the whole dependency, deliberately, so the harder
# half can be regenerated on the machine that is failing.
from axon.parity.feature_bundle import (
    FeatureBundle,
    libm_columns,
    read_feature_bundle,
    write_feature_bundle,
)

# The fourth gate is the same three questions asked across the language boundary
# (ADR-0021), so it belongs beside them rather than one import deeper. Nothing here
# pulls an ML library in: writing a bundle needs one, reading one needs numpy.
from axon.parity.rust_gate import (
    BundleError,
    Criterion,
    Decision,
    ParityBundle,
    quantile_decision,
    read_parity_bundle,
    write_bundle_from_registry,
    write_parity_bundle,
)

__all__ = [
    "AlarmSink",
    "Alignment",
    "BeaconError",
    "BeaconProbe",
    "Binning",
    "BundleError",
    "Cell",
    "Coverage",
    "Criterion",
    "DEFAULT_NAN_RATE_TOL",
    "DEFAULT_SILENCE_AFTER_NS",
    "Decision",
    "Discretizer",
    "DriftReport",
    "FEATURE_ATOL",
    "FEATURE_RTOL",
    "FeatureBundle",
    "FeatureDrift",
    "FeatureParityReport",
    "Flip",
    "GateReport",
    "Level",
    "MD_BEACON_SIZE",
    "MdBeaconReader",
    "MdBeaconSnapshot",
    "ModelParityReport",
    "MonitorConfig",
    "MonitorReport",
    "NN_EPS",
    "PSI_MODERATE",
    "PSI_STABLE",
    "ParityBundle",
    "ParityError",
    "ParityMonitor",
    "Publisher",
    "PublisherState",
    "TREE_EPS",
    "Verdict",
    "Window",
    "align_by_event_time",
    "aligned_feature_parity",
    "beacon_path",
    "collecting_sink",
    "decision_flips",
    "decision_invariant",
    "drift_report",
    "feature_parity",
    "kl_divergence",
    "libm_columns",
    "logging_sink",
    "model_parity",
    "psi",
    "psi_band",
    "publisher_state",
    "quantile_binning",
    "quantile_decision",
    "read_feature_bundle",
    "read_md_beacon",
    "read_parity_bundle",
    "run_monitor",
    "threshold_discretizer",
    "windows_from_matrices",
    "write_bundle_from_registry",
    "write_feature_bundle",
    "write_parity_bundle",
]
