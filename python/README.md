# `python/` — the Python research plane

**Implemented through Phase 5's harness, and real strategies now use it (658 tests green,
11 skipped for optional deps).** Every module below has real behaviour behind it. Tests live in
`python/tests` (run via `./run.sh` / `scripts/check.sh`); the heavy ML dependencies are optional
extras (`pip install -e 'python[ml]'`) and every test that needs one `importorskip`s it, so the
default gate passes on bare numpy + pytest.

The 11 skips are environment-dependent, not structural failures, and the count moves: six are
`test_features.py` parametrizations with no warmup window, and five are `hwsched` integration
tests that skip only because the project `.venv` has no `pydantic`. Outside the venv the same
suite reports 663 passed / 6 skipped. **`./run.sh`'s number is the one to quote** — it is the
gate that actually runs.

| Module | What is in it today |
|--------|---------------------|
| `axon.contracts` | `SIGNAL_DTYPE` (schema v3, carrying `ts_cause`) + `MD_SLICE_DTYPE` + `MD_BAR_DTYPE` and the ring layout, all parsed from `contracts/schema.toml` at import |
| `axon.signals` | The mmap SPSC ring client (writer + reader), parameterized by record dtype |
| `axon.marketdata` | The Rust→Python market-data ring consumer, batch reads with drop accounting, and the sentinel convention for a stale quote — all four quote fields go out as **zeros**, never a dead bid under a fresh `ts_event`, because the record carries one timestamp and it belongs to the event that triggered the slice |
| `axon.strategy` | The base class, the event types, and a `StrategyContext` that **imports no clock** — a strategy cannot stamp wall-clock time; also the named `URGENCY_*` levels and `TTL_OPERATOR_CEILING`, so the wire's two ambiguous fields mean the same thing in both languages |
| `axon.live` | The runner: drives a strategy onto the signal ring, FIFO backpressure, liveness beacon |
| `axon.features` | The feature library + the versioned, fingerprinted `FeatureSpec` |
| `axon.models` | Export (native trees / ONNX FP32) + the immutable versioned registry |
| `axon.parity` | The model-, feature- and drift-parity gates, incl. decision invariance — plus `rust_gate`, which writes the **parity bundle** Rust reads back (model bytes, a holdout matrix as raw f32, and Python's own scores *over the bytes it read off disk*) |
| `axon.backtest` | Deterministic replay driven through the Rust core, including the counter that says a tracker went unreadable rather than reporting it flat |
| `axon.compute` | Offloading sweeps/training to Modal via the `hwsched` CLI, dry-run-before-spend enforced as a type |
| `axon.strategies` | Real strategies, and the research plumbing they need: `perp_bar` (the first one up the ladder) and the `zoo` (five families, one of which crosses into Rust bit-exact), venue candle pulls with an on-disk cache, purged event-time walk-forward splits, the ladder runner behind `python -m axon.strategies`, `live_runner` (drives a strategy onto a **live** session's ring, and can carry a parity diff beside it), `shadow.BarParityDiff` (the diff both the shadow harness and the live runner hold — one alignment rule between them), `loss_evidence` (the fan-out that turns a loss bound from a declaration into a quantile), `portfolio_runner` (one bar-ring reader dispatched to **several** producers over several instruments, each writing its own ring — ADR-0038) and `portfolio_evidence` (what a book of several legs would actually have held, so `[portfolio]`'s bounds are quantiles too) |

See [architecture](../docs/01-architecture.md) and the
[strategy contract](../docs/06-strategy-contract.md).

Guiding rule: **Python owns research and signal generation, never the hot execution loop.**
A strategy's job is `events → features → inference → Signal`. Everything below the signal
boundary (orders, nonces, signing, networking) belongs to Rust.

## Planned package

The Phase-0 plan, kept for comparison rather than as a description — the table above is what is
actually here. `live/`, `marketdata/`, `compute/` and `strategies/` were not foreseen; nothing
in the plan was abandoned.

```
python/
└── axon/
    ├── features/     THE feature library — single source of truth for feature logic.
    │                 Used by research AND referenced by the parity harness. (see ADR-0003)
    ├── strategy/     Strategy base class + lifecycle/event handlers mirroring the Rust contract.
    ├── signals/      Signal publisher: writes fixed-layout Signals into the shared-memory ring.
    ├── models/       Training helpers + export (ONNX / native tree) + versioned registry.
    ├── backtest/     Backtest harness reusing the core event semantics (for parity).
    ├── parity/       Golden-test + live parity-monitor tooling (see docs/07).
    └── contracts/    Generated/loaded bindings for the shared contract in /contracts.
```

## The critical constraint: feature parity

`axon.features` must be the **only** implementation of any given feature. Under Boundary B,
the same `axon.features` code runs at research time and at serving time, so online == offline
by construction. If a feature is ever moved to Rust (`axon-model`, Boundary A), the Rust
version must pass a bit-equivalence gate against `axon.features` before it may serve. See
[03 — ML fidelity & feature parity](../docs/03-ml-fidelity-and-features.md).

Read the second sentence carefully before quoting the parity numbers. `axon.strategies.perp_bar`
reports feature parity at `max_abs_diff` **exactly 0**, and that number is a Python path measured
against a Python path — it is the Boundary-B guarantee stated as a measurement, which is *why*
it is zero rather than evidence that it would be. The cross-language gate that would mean
something stronger exists for **models** only
([ADR-0021](../docs/adr/0021-rust-model-parity-gate.md)); there is no Rust feature runtime yet,
so nothing gates a feature across the boundary because nothing computes one there.

## Packaging / tooling (decide in Phase 1)
- Build: prefer a modern toolchain (`uv` / `hatch` / `poetry`) — TBD.
- Shared-memory client: `multiprocessing.shared_memory` + `numpy` views, or `iceoryx2` Python
  bindings (mirrors the Rust choice in [ADR-0002](../docs/adr/0002-python-rust-boundary.md)).
- Export deps: `onnx`, `skl2onnx`/`onnxmltools`, plus native XGBoost/LightGBM.
- If we ever embed Rust into Python (or vice-versa): `maturin` + `pyo3` — not needed for
  Boundary B, kept in reserve.
