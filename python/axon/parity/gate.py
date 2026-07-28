"""Shared vocabulary for the three gates.

A gate returns a **report**, not a boolean. ``docs/07`` asks these checks to run in
CI *and* forever in production as the live parity monitor, and in both settings the
useful output is "which input, which column, by how much" — a bare ``False`` at
03:00 tells an operator nothing about whether to flatten the book.

``raise_for_status()`` is how a report becomes a failing test or a failing deploy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ParityError(AssertionError):
    """A parity gate failed.

    Deriving from :class:`AssertionError` so a gate called straight from a test
    reads as the assertion it is, while still being catchable by name in the live
    monitor (which alarms rather than dies).
    """


@runtime_checkable
class GateReport(Protocol):
    """What every gate report provides."""

    @property
    def passed(self) -> bool: ...

    def summary(self) -> str: ...

    def raise_for_status(self) -> None: ...


def _raise_unless(passed: bool, summary: str) -> None:
    if not passed:
        raise ParityError(summary)


__all__ = ["GateReport", "ParityError"]
