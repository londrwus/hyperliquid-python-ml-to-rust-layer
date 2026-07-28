"""The model-parity gate: reference predictions vs. candidate predictions.

The criteria are ``docs/03``'s, verbatim:

* **Trees** reproduce by deterministic threshold traversal, so the bar is exact
  (:data:`TREE_EPS`). Anything else means the thresholds were cast to a different
  width somewhere.
* **Neural nets** never reproduce bit-for-bit across runtimes — ONNX does not encode
  op ordering and floating-point addition is not associative — so the bar is
  ``max_abs_diff < eps``, starting at :data:`NN_EPS`.
* **Decision invariance**, for both. This is the check that actually protects P&L: a
  1e-9 logit wobble that never crosses a threshold changes nothing, and the same
  wobble sitting on a threshold changes a position from short to long. The size of
  the number tells you nothing about which of those happened, so the gate discretizes
  both sides and compares the *decisions*, and reports **which inputs flipped**.

The two failure modes worth stating out loud, because both look like a pass:

* a non-finite score never compares greater than ``eps`` — ``nan > 1e-5`` is
  ``False`` — so non-finite values are counted and failed on, never compared;
* a candidate that is within tolerance everywhere can still flip a decision, which
  is why ``passed`` is an ``and`` of both conditions and not a tolerance check with
  a decision check bolted on as advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np

from axon.parity.gate import ParityError, _raise_unless

#: Tree ensembles and classical models: exact, per ADR-0003.
TREE_EPS = 0.0

#: Neural nets / ONNX: the starting tolerance from ``docs/03``. Tighten per model,
#: never loosen it to make a red gate green — that is the whole gate.
NN_EPS = 1e-5

Discretizer = Callable[[np.ndarray], np.ndarray]

#: How many flips a report keeps. A systematically broken candidate flips thousands
#: of decisions and the first handful are enough to debug it; keeping them all turns
#: a monitoring alarm into a memory problem.
DEFAULT_LIMIT = 20


class Flip(NamedTuple):
    """One input whose discretized trading decision changed."""

    index: int
    reference: float
    candidate: float
    reference_decision: int
    candidate_decision: int


def threshold_discretizer(*, long_at: float, short_at: float) -> Discretizer:
    """Scores → ``{-1 short, 0 flat, +1 long}`` on two thresholds.

    The usual shape of a discretized trading decision, and the default thing to hand
    :func:`model_parity`. A strategy with its own mapping (argmax over classes,
    size buckets, a hysteresis band) should pass that instead — the gate cares that
    *its* decisions are stable, not that they match this convention.
    """
    if not short_at < long_at:
        raise ValueError(f"need short_at < long_at, got short_at={short_at} long_at={long_at}")

    def discretize(scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=np.float64)
        out = np.zeros(s.shape, dtype=np.int64)
        out[s >= long_at] = 1
        out[s <= short_at] = -1
        return out

    return discretize


def _pair(reference, candidate) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(
            f"reference {ref.shape} and candidate {cand.shape} must have the same shape; "
            "a length mismatch means the two runs did not see the same inputs"
        )
    if ref.ndim not in (1, 2):
        raise ValueError(f"predictions must be 1-D or 2-D, got shape {ref.shape}")
    if ref.size == 0:
        raise ValueError("a parity gate over zero predictions proves nothing")
    return ref, cand


def decision_flips(
    reference,
    candidate,
    *,
    discretizer: Discretizer,
    limit: int = DEFAULT_LIMIT,
) -> tuple[tuple[Flip, ...], int]:
    """Every input whose decision changed, capped at ``limit``, plus the total count."""
    ref, cand = _pair(reference, candidate)
    ref_dec = np.asarray(discretizer(ref))
    cand_dec = np.asarray(discretizer(cand))
    if ref_dec.shape != cand_dec.shape:
        raise ValueError(
            f"the discretizer returned {ref_dec.shape} for the reference and "
            f"{cand_dec.shape} for the candidate; it must be a pure function of scores"
        )
    if ref_dec.shape[0] != ref.shape[0]:
        raise ValueError(
            f"the discretizer returned {ref_dec.shape[0]} decisions for {ref.shape[0]} inputs"
        )

    differing = ref_dec != cand_dec
    rows = np.flatnonzero(differing if differing.ndim == 1 else differing.any(axis=1))
    flips = tuple(
        Flip(
            index=int(i),
            reference=float(np.ravel(ref[i])[0]),
            candidate=float(np.ravel(cand[i])[0]),
            reference_decision=int(np.ravel(ref_dec[i])[0]),
            candidate_decision=int(np.ravel(cand_dec[i])[0]),
        )
        for i in rows[:limit]
    )
    return flips, int(rows.size)


def decision_invariant(reference, candidate, *, discretizer: Discretizer) -> bool:
    """Whether no discretized trading decision flips between two prediction sets.

    The boolean form of :func:`decision_flips`, kept because ``docs/03`` names the
    property. Use the report from :func:`model_parity` when you need to know *which*
    inputs moved — which is every time a gate is red.
    """
    _, total = decision_flips(reference, candidate, discretizer=discretizer, limit=0)
    return total == 0


@dataclass(frozen=True)
class ModelParityReport:
    """The outcome of one model-parity run."""

    n: int
    eps: float
    max_abs_diff: float
    max_abs_diff_index: int
    non_finite: int
    n_flips: int
    flips: tuple[Flip, ...]

    @property
    def passed(self) -> bool:
        # Ordered so the non-finite check runs first: `nan <= eps` is False, which
        # would fail for the right reason by accident, but `max_abs_diff` is NaN in
        # that case and the message would be about tolerance rather than about a
        # model that produced a NaN.
        return self.non_finite == 0 and self.max_abs_diff <= self.eps and self.n_flips == 0

    def summary(self) -> str:
        head = (
            f"model parity {'PASS' if self.passed else 'FAIL'}: n={self.n} "
            f"max_abs_diff={self.max_abs_diff:.3e} (eps={self.eps:.3e}) "
            f"flips={self.n_flips} non_finite={self.non_finite}"
        )
        if self.passed:
            return head
        lines = [head]
        if self.non_finite:
            lines.append(f"  {self.non_finite} prediction(s) were not finite on one or both sides")
        if self.max_abs_diff > self.eps:
            lines.append(f"  worst divergence at input {self.max_abs_diff_index}")
        for f in self.flips:
            lines.append(
                f"  input {f.index}: {f.reference:.9g} → {f.candidate:.9g} "
                f"flipped decision {f.reference_decision:+d} → {f.candidate_decision:+d}"
            )
        if self.n_flips > len(self.flips):
            lines.append(f"  … and {self.n_flips - len(self.flips)} more flip(s)")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        """Raise :class:`~axon.parity.gate.ParityError` unless the gate passed."""
        _raise_unless(self.passed, self.summary())


def model_parity(
    reference,
    candidate,
    *,
    discretizer: Discretizer,
    eps: float = NN_EPS,
    limit: int = DEFAULT_LIMIT,
) -> ModelParityReport:
    """Compare candidate predictions against the reference on both criteria.

    ``eps`` is the numeric tolerance (:data:`TREE_EPS` for trees and classical
    models, :data:`NN_EPS` to start for neural nets). ``discretizer`` maps scores to
    trading decisions and is **required**: the numeric check alone has passed every
    model that ever silently changed a position at a threshold.
    """
    ref, cand = _pair(reference, candidate)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    finite = np.isfinite(ref) & np.isfinite(cand)
    non_finite = int(ref.size - np.count_nonzero(finite))
    # `inf - inf` is NaN *and* a warning; the non-finite positions are accounted for
    # by `non_finite` and contribute nothing to the magnitude.
    diff = np.zeros_like(ref)
    np.subtract(ref, cand, out=diff, where=finite)
    np.abs(diff, out=diff)
    if non_finite == ref.size:
        max_abs_diff, worst = float("nan"), -1
    else:
        flat = int(np.argmax(diff))
        max_abs_diff = float(diff.reshape(-1)[flat])
        worst = int(np.unravel_index(flat, diff.shape)[0])

    flips, n_flips = decision_flips(ref, cand, discretizer=discretizer, limit=limit)
    return ModelParityReport(
        n=int(ref.shape[0]),
        eps=float(eps),
        max_abs_diff=max_abs_diff,
        max_abs_diff_index=worst,
        non_finite=non_finite,
        n_flips=n_flips,
        flips=flips,
    )


__all__ = [
    "DEFAULT_LIMIT",
    "Discretizer",
    "Flip",
    "ModelParityReport",
    "NN_EPS",
    "ParityError",
    "TREE_EPS",
    "decision_flips",
    "decision_invariant",
    "model_parity",
    "threshold_discretizer",
]
