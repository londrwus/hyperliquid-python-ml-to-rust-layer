"""The feature-parity gate: online vectors vs. the offline recompute.

``docs/03`` calls this the hard one, and the reason is that both sides are *supposed*
to be the same code. When they diverge it is never the arithmetic — it is a late
event, a stale book, a NaN handled differently, a window that starts one sample
early. So the report names the **column and the row**, because "the feature vectors
differ" is not a debuggable statement about a matrix with 40 columns and 100k rows.

Two decisions worth stating:

* **NaN on one side only is a mismatch; NaN on both sides is a match.** Warmup is
  legitimately NaN in both paths and would otherwise fail every run, while a feature
  that goes NaN online and finite offline is exactly the staleness bug this gate
  exists to catch. Left to ``==`` or ``np.allclose`` defaults, both cases read the
  same way.
* **Alignment is on event time, not row position** (:func:`align_by_event_time`).
  The online path samples; the offline path recomputes a contiguous window. One
  missing sample shifts every subsequent row, and a positional comparison then
  reports 100% divergence for a feed that is perfectly correct.
* **An intersection has a denominator, and it is part of the verdict**
  (:class:`Alignment`, :class:`Coverage`). This is the correction to the second
  point, and it is the reason this module was revisited: matching on event time
  makes a row the online path never produced *absent* rather than wrong, and an
  absent row is compared against nothing. A feed that emits half its rows is then
  compared on that half, agrees with it to the last bit, and reports
  ``PASS … max_abs_diff=0.000e+00`` — every field a reader looks at identical to a
  healthy run. So the alignment now carries what it dropped, with its own
  ``passed``/``raise_for_status()`` over it, and :func:`aligned_feature_parity`
  folds that count into the parity report — which is why it, and not
  ``align_by_event_time`` + ``feature_parity``, is the call to reach for: the
  two-step form lets a caller keep the matched rows and discard the record of what
  did not match (ADR-0030).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from axon.parity.gate import ParityError, _raise_unless

#: Default tolerances. Tight enough that a different formula or a different window
#: shows up immediately, loose enough to absorb a different summation order between
#: two runs of the same formula (which is all that separates a vectorized recompute
#: from an incremental online one).
FEATURE_ATOL = 1e-9
FEATURE_RTOL = 1e-9

#: Mismatching cells kept in a report; see ``model.DEFAULT_LIMIT`` for the reasoning.
DEFAULT_LIMIT = 20


class Cell(NamedTuple):
    """One disagreeing element of the feature matrix."""

    row: int
    column: str
    online: float
    offline: float
    abs_diff: float


class Coverage(NamedTuple):
    """How much of each side an alignment actually managed to compare.

    All counts, no arrays, so a report holding one stays hashable and comparable.
    The three ``offline_*`` buckets split the unmatched offline rows by whether they
    were ever the online side's to produce, because only one of the three is
    legitimate. Which rows those are depends on ``scope``
    (:func:`align_by_event_time`): inferred from the online side's own event-time
    span under ``"observed"``, and *all of them* under ``"declared"``, where the
    caller has already trimmed the reference to what was owed.

    * ``offline_before`` — the offline recompute has rows from before the online
      side's first event. A cold-started replay and a monitor window that opens
      mid-history both produce exactly this, and holding them to the whole history
      would fire on every healthy run. Reported, never a failure. **This bucket is
      empty by construction under** ``scope="declared"``: a caller that has trimmed
      the reference to the owed rows has said there is no such thing as too early,
      and a serving path blind through its opening rows is then a fault rather than
      a late join. See :func:`align_by_event_time`.
    * ``offline_within`` — a gap *inside* the span the online path was running for.
      This is the blind spot: the online path was there and produced nothing.
    * ``offline_after`` — the offline side keeps going past the online side's last
      event. That is a feed that stopped, which is the loudest form of the same bug
      and the one an in-span rule alone would excuse.

    ``online_unmatched`` is separate again: a row the online path produced at an
    event time the *reference* has nothing at. The offline recompute is the
    definition of correct, so this direction is never legitimate — it means the two
    are looking at different events, not that one is missing some.
    """

    n_online: int
    n_offline: int
    n_matched: int
    n_online_unmatched: int
    n_offline_before: int
    n_offline_within: int
    n_offline_after: int

    @property
    def complete(self) -> bool:
        """Every row either matched or was never the online side's to produce.

        ``n_matched > 0`` is a condition and not a formality: a comparison of nothing
        is not a comparison that agreed. "PASS, 0 rows compared" is this class's own
        failure mode at its limit, and it would otherwise satisfy every other clause
        here trivially.
        """
        return (
            self.n_matched > 0
            and self.n_online_unmatched == 0
            and self.n_offline_within == 0
            and self.n_offline_after == 0
        )

    @property
    def n_in_scope(self) -> int:
        """Offline rows the online side owed — the denominator that means something.

        Not ``n_offline``: under ``scope="observed"`` rows before the online side's
        first event were never its to produce, and counting them would make a
        cold-started replay read as 8% covered when it is in fact complete — the
        reading that gets a real check called noisy and deleted. Under
        ``scope="declared"`` the two are equal, because the caller has already said
        every row it handed over was owed.
        """
        return self.n_matched + self.n_offline_within + self.n_offline_after

    @property
    def fraction(self) -> float:
        """Matched rows over rows owed — the denominator, made visible.

        Reported for a human; the verdict is :attr:`complete`, which is a set of
        exact counts rather than a ratio. A ratio invites a threshold, and there is
        no fraction of a feature history it is acceptable to be blind to.
        """
        return float(self.n_matched) / self.n_in_scope if self.n_in_scope else 0.0

    def describe(self) -> str:
        parts = [f"{self.n_matched}/{self.n_in_scope} owed rows compared"]
        if self.n_online_unmatched:
            parts.append(f"{self.n_online_unmatched} online row(s) the reference has nothing at")
        if self.n_offline_within:
            # Phrased as the fault rather than the geometry, because the geometry
            # differs by scope: "inside the online side's own span" under "observed",
            # "inside the set the caller declared owed" under "declared". The fault
            # is the same sentence either way.
            parts.append(f"{self.n_offline_within} owed row(s) the online side never produced")
        if self.n_offline_after:
            parts.append(f"{self.n_offline_after} offline row(s) after the online side stopped")
        if self.n_offline_before:
            parts.append(f"{self.n_offline_before} before it started (out of scope, not a fault)")
        return "; ".join(parts)


def _nearest_offsets(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed nanoseconds from each ``a`` stamp to the nearest ``b`` stamp."""
    ordered = np.sort(b)
    pos = np.searchsorted(ordered, a)
    left = ordered[np.clip(pos - 1, 0, ordered.size - 1)]
    right = ordered[np.clip(pos, 0, ordered.size - 1)]
    to_left, to_right = a - left, a - right
    return np.where(np.abs(to_left) <= np.abs(to_right), to_left, to_right)


def _offset_note(a: np.ndarray, b: np.ndarray) -> str | None:
    """Name a stamping-convention disagreement, when that is what an empty intersection is.

    An empty intersection has two very different causes and one symptom. Either the
    two sides genuinely describe different spans of time, or they describe the *same*
    events under two stamping conventions — and the second one has bitten this
    codebase already: a candle's ``ts_event`` is ``T + 1 ms`` in both languages
    (Hyperliquid's ``T`` is the bar's last millisecond, so a bar stamped ``T`` sorts
    equal to the trades inside it). While the two halves were one millisecond apart
    the intersection was empty and the gate failed as "an empty matrix proves
    nothing", which is a long way from the cause. This is that distance, removed.
    """
    if a.size == 0 or b.size == 0:
        return None
    offsets = _nearest_offsets(a, b)
    distinct = np.unique(offsets)
    if distinct.size == 1:
        k = int(distinct[0])
        note = (
            f"every online stamp is exactly {k:+d} ns from its nearest offline stamp, "
            "so these are the same events under two stamping conventions rather than "
            "two different spans of time"
        )
        if abs(k) == 1_000_000:
            note += (
                " — and 1 ms is the known one: a candle's ts_event is T + 1 ms on both "
                "sides (T is the bar's last millisecond), so one of these is stamping "
                "bars at T. Fix the stamp, not the alignment"
            )
        return note
    lo, hi = int(offsets.min()), int(offsets.max())
    return (
        f"nearest-stamp offsets range {lo:+d}..{hi:+d} ns (median "
        f"{int(np.median(offsets)):+d}), so the two sides are not merely offset by a "
        "constant — they are describing different events"
    )


class Alignment(tuple):
    """The matched row indices, *and* what matching them threw away.

    It **is** the ``(online_idx, offline_idx)`` pair — ``left, right =
    align_by_event_time(...)`` and ``idx[0]`` both still work — so this is a
    strictly additive change to every existing call site. What is new is that the
    pair now arrives attached to its own denominator, because an intersection that
    silently forgets what it did not match is how a half-blind feed reports a
    perfect zero.

    Like every other gate in this package it is a **report, not a boolean**
    (ADR-0016 §7): :attr:`passed`, :meth:`summary` and :meth:`raise_for_status`.
    """

    online: np.ndarray
    offline: np.ndarray
    coverage: Coverage
    online_unmatched: np.ndarray
    offline_before: np.ndarray
    offline_within: np.ndarray
    offline_after: np.ndarray
    offset_note: str | None

    def __new__(
        cls,
        online_idx: np.ndarray,
        offline_idx: np.ndarray,
        *,
        coverage: Coverage,
        online_unmatched: np.ndarray,
        offline_before: np.ndarray,
        offline_within: np.ndarray,
        offline_after: np.ndarray,
        offset_note: str | None,
    ) -> "Alignment":
        self = super().__new__(cls, (online_idx, offline_idx))
        self.online = online_idx
        self.offline = offline_idx
        self.coverage = coverage
        self.online_unmatched = online_unmatched
        self.offline_before = offline_before
        self.offline_within = offline_within
        self.offline_after = offline_after
        self.offset_note = offset_note
        return self

    @property
    def n_matched(self) -> int:
        return self.coverage.n_matched

    @property
    def disjoint(self) -> bool:
        """Both sides have rows and not one of them lines up."""
        return (
            self.coverage.n_matched == 0
            and self.coverage.n_online > 0
            and self.coverage.n_offline > 0
        )

    @property
    def passed(self) -> bool:
        return self.coverage.complete

    def summary(self) -> str:
        head = (
            f"alignment {'OK' if self.passed else 'INCOMPLETE'}: "
            f"{self.coverage.describe()} "
            f"(coverage {self.coverage.fraction:.3f}; {self.coverage.n_online} online rows, "
            f"{self.coverage.n_offline} offline)"
        )
        lines = [head]
        if self.disjoint:
            lines.append(
                "  the two sides share no event time at all — an empty intersection is "
                "not a small disagreement, it is a join that never happened"
            )
        if self.coverage.n_online == 0 and self.coverage.n_offline:
            lines.append(
                "  the online path produced no rows at all; there is nothing here that "
                "could have disagreed, which is not the same as agreeing"
            )
        if self.offset_note:
            lines.append(f"  {self.offset_note}")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`~axon.parity.gate.ParityError` unless coverage is complete."""
        _raise_unless(self.passed, self.summary())


def align_by_event_time(
    online_ts, offline_ts, *, on_gap: str = "report", scope: str = "observed"
) -> Alignment:
    """Row indices of the events both sides observed, in event-time order.

    Timestamps must be int64 nanoseconds (``ts_event``, never a receipt clock) and
    unique within each side: duplicates make "which row is this?" ambiguous, and
    picking one arbitrarily is how a comparison passes against the wrong row.

    The rows that did **not** match are no longer thrown away: they come back on the
    returned :class:`Alignment`, classified by :class:`Coverage`, with
    :attr:`Alignment.passed` and :meth:`Alignment.raise_for_status` over them.

    ``scope`` says **what the ``offline_ts`` argument means**, which is the one thing
    about the comparison that this function cannot see and the caller always knows:

    ``"observed"`` (default)
        ``offline_ts`` is a *reference* recompute, and the online side may
        legitimately not have been running for all of it. The owed span is therefore
        inferred from the online side's own first stamp onward. Right for a live
        monitor, whose window opens mid-history, and for a cold-started replay
        matched against a full-history recompute.
    ``"declared"``
        ``offline_ts`` is exactly the set of rows the online side owed — the caller
        has already trimmed it. Nothing is out of scope, so a serving path that was
        blind through its opening rows is a fault rather than a late join.

    The distinction is real and it is not "monitor versus gate": it is a property of
    the array passed in. Under ``"observed"`` the only evidence available is the
    online side's first stamp, and a path that produced nothing for its first *k*
    owed rows has the same first stamp as one that started on time — so the two are
    genuinely indistinguishable and excusing them is the honest default. Under
    ``"declared"`` there is nothing to infer.

    A ``"declared"`` claim is about the array the caller has just built, so it cannot
    disagree with a *separate* description of the same fact — which is why this is a
    mode and not, say, an ``online_span=(lo, hi)`` tuple. A span is a second
    description, and a second description can be narrower than the data and silently
    re-excuse the very rows this closes. Trimming the reference makes the two
    descriptions one, and mis-trimming it is caught in **both** directions: too
    narrow and the online side's extra rows land in ``online_unmatched``; too wide
    and the surplus lands in ``offline_within``. Both are failures with names.

    ``on_gap`` decides whether a *partial* intersection raises here or is left for
    the caller to judge:

    ``"report"`` (default)
        Return the alignment; the verdict is on the object. This is the default for
        the reason ADR-0016 §7 gives for every other gate in this package —
        **gates return reports, not booleans** — and for two concrete ones. A live
        monitor must alarm rather than die, so an aligner that raises cannot be on
        its path at all. And an aligner that raises on a bad alignment cannot be
        used to *demonstrate* one, which is exactly what the one-millisecond
        candle-stamp regression test does.
    ``"raise"``
        Raise :class:`~axon.parity.gate.ParityError` on a partial intersection, for
        a CI call site that wants the aligner itself to be the assertion.

    **Zero overlap deliberately does not raise in either mode.** An empty
    intersection is already a hard failure downstream — :func:`feature_parity`
    refuses an empty matrix, :func:`aligned_feature_parity` returns a failing report
    — so it is the one case that cannot be missed. What silently passes, and what
    this whole accounting exists for, is the *partial* intersection. Zero overlap
    gets a better *name* instead: see :func:`_offset_note`.
    """
    if on_gap not in ("raise", "report"):
        raise ValueError(f"on_gap must be 'raise' or 'report', got {on_gap!r}")
    if scope not in ("observed", "declared"):
        raise ValueError(f"scope must be 'observed' or 'declared', got {scope!r}")
    a = np.asarray(online_ts, dtype=np.int64)
    b = np.asarray(offline_ts, dtype=np.int64)
    for name, arr in (("online_ts", a), ("offline_ts", b)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
        if np.unique(arr).size != arr.size:
            raise ValueError(
                f"{name} contains duplicate event times; two feature rows stamped with "
                "the same nanosecond cannot be matched to each other unambiguously"
            )
    _, idx_a, idx_b = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    order = np.argsort(idx_a, kind="stable")
    idx_a, idx_b = idx_a[order], idx_b[order]

    online_unmatched = np.setdiff1d(np.arange(a.size), idx_a, assume_unique=True)
    offline_unmatched = np.setdiff1d(np.arange(b.size), idx_b, assume_unique=True)
    if scope == "declared":
        # The caller states that every offline row it handed over was owed, so there
        # is nothing here that could be out of scope and a late start is a fault like
        # any other gap. Nothing is inferred from the online side at all, which is the
        # point: under "observed" the only evidence available is the online side's own
        # first stamp, and a serving path that was blind through its opening rows
        # produces exactly the same first stamp as one that started on time.
        before = after = np.empty(0, dtype=np.intp)
        within = offline_unmatched
    elif a.size:
        stamps = b[offline_unmatched]
        first, last = int(a.min()), int(a.max())
        before = offline_unmatched[stamps < first]
        within = offline_unmatched[(stamps >= first) & (stamps <= last)]
        after = offline_unmatched[stamps > last]
    else:
        # No online rows at all: there is no span to be out of. Calling every offline
        # row "out of scope" would make a serving path that never started look exactly
        # like one that had nothing to do, which is the ambiguity this whole module
        # exists to remove.
        before = after = np.empty(0, dtype=np.intp)
        within = offline_unmatched

    coverage = Coverage(
        n_online=int(a.size),
        n_offline=int(b.size),
        n_matched=int(idx_a.size),
        n_online_unmatched=int(online_unmatched.size),
        n_offline_before=int(before.size),
        n_offline_within=int(within.size),
        n_offline_after=int(after.size),
    )
    alignment = Alignment(
        idx_a,
        idx_b,
        coverage=coverage,
        online_unmatched=online_unmatched,
        offline_before=before,
        offline_within=within,
        offline_after=after,
        offset_note=_offset_note(a, b) if idx_a.size == 0 else None,
    )
    if on_gap == "raise" and not alignment.passed and not alignment.disjoint:
        alignment.raise_for_status()
    return alignment


@dataclass(frozen=True)
class FeatureParityReport:
    """The outcome of one feature-parity run."""

    n_rows: int
    columns: tuple[str, ...]
    atol: float
    rtol: float
    max_abs_diff: float
    n_mismatched: int
    n_nan_mismatched: int
    worst_column: str | None
    per_column: Mapping[str, int]
    mismatches: tuple[Cell, ...]
    #: What the alignment compared and what it could not, when the caller supplied
    #: one. ``None`` means the two matrices arrived already row-for-row and this gate
    #: has no way to know what was dropped upstream — which is a real state and is
    #: printed as ``coverage=unchecked`` rather than left to look like completeness.
    coverage: Coverage | None
    #: The alignment's own diagnosis, when it had one. Carried onto the *gate's*
    #: report rather than left on the alignment, because the alignment is the object
    #: a caller throws away and the report is the one that reaches an operator. This
    #: is what turns "an empty feature matrix proves nothing" into "these are the
    #: same events, one millisecond apart".
    alignment_note: str | None

    @property
    def passed(self) -> bool:
        """Every compared cell agreed **and** every row that should have been
        compared was. The second half is not a formality: an intersection cannot
        disagree with a row that is not there, so without it a serving path emitting
        half its rows reports a flawless zero over the half it did emit."""
        agreed = self.n_mismatched == 0 and self.n_nan_mismatched == 0
        return agreed and (self.coverage is None or self.coverage.complete)

    def summary(self) -> str:
        cover = "unchecked" if self.coverage is None else (
            f"{self.coverage.n_matched}/{self.coverage.n_in_scope}"
        )
        head = (
            f"feature parity {'PASS' if self.passed else 'FAIL'}: "
            f"rows={self.n_rows} cols={len(self.columns)} "
            f"max_abs_diff={self.max_abs_diff:.3e} "
            f"(atol={self.atol:.1e} rtol={self.rtol:.1e}) "
            f"mismatched={self.n_mismatched} nan_mismatched={self.n_nan_mismatched} "
            f"coverage={cover}"
        )
        if self.passed:
            return head
        lines = [head]
        if self.coverage is not None and not self.coverage.complete:
            lines.append(f"  coverage: {self.coverage.describe()}")
            lines.append(
                "  a gate is only as wide as the rows both sides produced, and this "
                "one was not wide enough"
            )
        if self.alignment_note:
            lines.append(f"  {self.alignment_note}")
        if self.n_mismatched or self.n_nan_mismatched:
            lines.append(f"  worst column: {self.worst_column}")
            offenders = sorted(self.per_column.items(), key=lambda kv: -kv[1])
            lines.append("  cells per column: " + ", ".join(f"{c}={n}" for c, n in offenders if n))
            for cell in self.mismatches:
                lines.append(
                    f"  row {cell.row} col {cell.column}: online={cell.online:.12g} "
                    f"offline={cell.offline:.12g} |diff|={cell.abs_diff:.3e}"
                )
            if self.n_mismatched + self.n_nan_mismatched > len(self.mismatches):
                remaining = self.n_mismatched + self.n_nan_mismatched - len(self.mismatches)
                lines.append(f"  … and {remaining} more cell(s)")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`~axon.parity.gate.ParityError` unless the gate passed."""
        _raise_unless(self.passed, self.summary())


def feature_parity(
    online,
    offline,
    *,
    columns: Sequence[str],
    alignment: Alignment | None = None,
    atol: float = FEATURE_ATOL,
    rtol: float = FEATURE_RTOL,
    limit: int = DEFAULT_LIMIT,
) -> FeatureParityReport:
    """Compare an online feature matrix against the offline recompute.

    Both matrices are ``(n_rows, n_features)`` and must already be aligned — use
    :func:`align_by_event_time` first if the online path sampled. ``columns`` names
    the features, in matrix order; :attr:`axon.features.FeatureSpec.columns` is
    exactly that tuple.

    Pass the ``alignment`` that produced these rows and its :class:`Coverage`
    becomes part of the verdict, so rows the online path never produced fail the
    gate instead of vanishing from the denominator. Prefer
    :func:`aligned_feature_parity`, which does both steps and therefore cannot be
    called without answering that question.
    """
    on = np.asarray(online, dtype=np.float64)
    off = np.asarray(offline, dtype=np.float64)
    if on.ndim != 2 or off.ndim != 2:
        raise ValueError(f"expected 2-D feature matrices, got {on.shape} and {off.shape}")
    if on.shape != off.shape:
        raise ValueError(
            f"online {on.shape} and offline {off.shape} differ; align the two sides on "
            "event time before comparing, or the rows are not the same events"
        )
    cols = tuple(columns)
    if len(cols) != on.shape[1]:
        raise ValueError(f"{len(cols)} column names for a matrix with {on.shape[1]} columns")
    if on.size == 0:
        if alignment is None:
            # Nothing compared and nothing that could explain why. Refusing is the
            # only honest answer available; with an alignment in hand there is a
            # better one, below.
            raise ValueError("a parity gate over an empty feature matrix proves nothing")
        # An alignment turns "an empty matrix proves nothing" into a diagnosis: which
        # side had rows, how many, and whether the two are merely a constant stamp
        # offset apart. Returning a failing report rather than raising is what lets a
        # live monitor alarm on it instead of dying on it.
        return FeatureParityReport(
            n_rows=0,
            columns=cols,
            atol=float(atol),
            rtol=float(rtol),
            max_abs_diff=0.0,
            n_mismatched=0,
            n_nan_mismatched=0,
            worst_column=None,
            per_column={c: 0 for c in cols},
            mismatches=(),
            coverage=alignment.coverage,
            alignment_note=alignment.offset_note,
        )

    # Non-finite cells are never subtracted: NaN - NaN and inf - inf are both NaN,
    # and a NaN difference silently compares False against every tolerance. They are
    # matched by identity instead — same NaN-ness, or the same infinity.
    comparable = np.isfinite(on) & np.isfinite(off)
    same_non_finite = (np.isnan(on) & np.isnan(off)) | (~comparable & (on == off))
    nan_mismatch = ~comparable & ~same_non_finite

    diff = np.zeros_like(on)
    np.subtract(on, off, out=diff, where=comparable)
    np.abs(diff, out=diff)
    tolerance = atol + rtol * np.abs(np.where(comparable, off, 0.0))
    over = comparable & (diff > tolerance)
    max_abs_diff = float(diff[comparable].max()) if comparable.any() else 0.0

    bad = over | nan_mismatch
    # An incomparable cell has no meaningful magnitude, so it ranks above every
    # numeric divergence: "this column went NaN online" outranks "this one is 3e-9 off".
    rank = np.where(nan_mismatch, np.inf, diff)
    per_column = {c: int(bad[:, j].sum()) for j, c in enumerate(cols)}
    worst = None
    if bad.any():
        # Rank by cell count first, magnitude second. A unit or scale error breaks one
        # column on every row, which is a very different diagnosis from one column
        # being slightly worse on one row.
        worst = max(
            (j for j in range(len(cols)) if per_column[cols[j]]),
            key=lambda j: (per_column[cols[j]], float(rank[:, j].max())),
        )
        worst = cols[worst]

    rows, col_idx = np.nonzero(bad)
    order = np.argsort(-rank[rows, col_idx], kind="stable")
    mismatches = tuple(
        Cell(
            row=int(rows[k]),
            column=cols[int(col_idx[k])],
            online=float(on[rows[k], col_idx[k]]),
            offline=float(off[rows[k], col_idx[k]]),
            abs_diff=float(rank[rows[k], col_idx[k]]),
        )
        for k in order[:limit]
    )
    return FeatureParityReport(
        n_rows=int(on.shape[0]),
        columns=cols,
        atol=float(atol),
        rtol=float(rtol),
        max_abs_diff=max_abs_diff,
        n_mismatched=int(over.sum()),
        n_nan_mismatched=int(nan_mismatch.sum()),
        worst_column=worst,
        per_column=per_column,
        mismatches=mismatches,
        coverage=None if alignment is None else alignment.coverage,
        alignment_note=None if alignment is None else alignment.offset_note,
    )


def aligned_feature_parity(
    online,
    offline,
    *,
    online_ts,
    offline_ts,
    columns: Sequence[str],
    scope: str = "observed",
    atol: float = FEATURE_ATOL,
    rtol: float = FEATURE_RTOL,
    limit: int = DEFAULT_LIMIT,
) -> FeatureParityReport:
    """Align on event time and compare, in one call that cannot forget the denominator.

    This is the entry point every new caller and the live monitor should use.
    :func:`align_by_event_time` followed by :func:`feature_parity` is the same two
    steps, and it is two steps precisely because the first one can be thrown away:
    a caller that keeps only the index arrays has discarded the record of what did
    not match, and what did not match is the half of the verdict that fails
    silently. Here the alignment travels with the matrices it selected, so the
    report always carries its :class:`Coverage`.

    Pass ``scope="declared"`` when ``offline`` has already been trimmed to exactly
    the rows the online side owed; see :func:`align_by_event_time`. A gate that knows
    its span should — otherwise an online side blind through its opening rows is
    excused as a late join, which is the same absence this call exists to catch.

    Never raises for a bad comparison — it returns a failing report, including for
    an empty intersection. ``raise_for_status()`` is how it becomes a CI failure;
    a monitor reads ``passed`` and alarms.
    """
    alignment = align_by_event_time(online_ts, offline_ts, on_gap="report", scope=scope)
    on = np.asarray(online, dtype=np.float64)
    off = np.asarray(offline, dtype=np.float64)
    for name, matrix, stamps in (
        ("online", on, np.asarray(online_ts)),
        ("offline", off, np.asarray(offline_ts)),
    ):
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D {name} feature matrix, got shape {matrix.shape}")
        if matrix.shape[0] != stamps.size:
            raise ValueError(
                f"{name}: {matrix.shape[0]} rows but {stamps.size} event times — a row "
                "and its stamp are one observation, and pairing them by position after "
                "they have been separated is the bug this gate aligns to avoid"
            )
    return feature_parity(
        on[alignment.online],
        off[alignment.offline],
        columns=columns,
        alignment=alignment,
        atol=atol,
        rtol=rtol,
        limit=limit,
    )


__all__ = [
    "Alignment",
    "Cell",
    "Coverage",
    "DEFAULT_LIMIT",
    "FEATURE_ATOL",
    "FEATURE_RTOL",
    "FeatureParityReport",
    "ParityError",
    "align_by_event_time",
    "aligned_feature_parity",
    "feature_parity",
]
