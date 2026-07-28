"""Axon — the Python research plane.

Python owns research and signal generation, never the hot execution loop
(``docs/00-vision-and-scope.md``). A strategy's job is
``events → features → inference → Signal``; everything below the signal boundary
(orders, nonces, signing, networking) belongs to the Rust core.

Submodules
----------
- :mod:`axon.contracts`  — the shared boundary contract (``Signal`` and ``MdSlice``
  dtypes + ring layout), parsed from the *same* ``contracts/schema.toml`` the Rust
  side uses, so the two languages cannot silently drift.
- :mod:`axon.signals`    — the shared-memory SPSC ring client (writer + reader).
- :mod:`axon.marketdata` — the Rust→Python market-data ring consumer (batch reads).
- :mod:`axon.strategy`   — the strategy base class (mirrors the Rust contract).
- :mod:`axon.live`       — the runner that drives a strategy onto the signal ring.
- :mod:`axon.features`   — THE feature library + the versioned ``FeatureSpec``.
- :mod:`axon.models`     — model export (native trees / ONNX FP32) + the registry.
- :mod:`axon.parity`     — the model-, feature- and drift-parity gates.
- :mod:`axon.backtest`   — deterministic replay through the Rust core.
- :mod:`axon.compute`    — offloading heavy jobs to Modal via the ``hwsched`` CLI.
"""

__version__ = "0.0.1"
