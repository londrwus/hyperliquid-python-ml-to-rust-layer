"""``python -m axon.parity`` — run the live parity monitor over an offline source.

A ``__main__`` module rather than an ``if __name__ == "__main__"`` inside
:mod:`axon.parity.monitor`, matching :mod:`axon.live` and :mod:`axon.strategies`.
The package's ``__init__`` already imports the monitor, so ``-m`` on the module
itself would execute it a second time under a different name — ``runpy`` warns
about exactly that, and two copies of a module holding a logger and an enum is the
kind of thing that is confusing precisely when something is already wrong.

::

    python -m axon.parity --perp-bar BTC                # the real serving path
    python -m axon.parity --perp-bar BTC --blind-every 3  # …and watch it fail

**Offline sources only, and that is a property rather than a limitation.** ADR-0030 §7
holds that nothing in this package opens a connection, so the live driver lives outside
it: ``scripts/sessions/parity_live.py`` drives the same monitor over a running session's
market-data bar ring, and ``scripts/sessions/parity-live.sh`` starts the session too.
``docs/adr/0030-live-parity-monitor-and-the-coverage-denominator.md``
is what such a run can say and — the longer half — what it still cannot.

Every run here, live or not, now records the wall-clock gaps its silence deadline should
be argued from; they print under ``silence evidence`` at the foot of the report.
"""

from __future__ import annotations

from axon.parity.monitor import main

raise SystemExit(main())
