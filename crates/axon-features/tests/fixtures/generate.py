"""Write the cross-language conformance fixture that `axon-features` is pinned to.

`crates/axon-features` is a **second implementation** of transforms `docs/03` says
should never be implemented twice. The feature-parity bundle
(:mod:`axon.parity.feature_bundle`) is the gate that makes that survivable at the
level of whole matrices. This fixture is the layer *underneath* it, and it exists
because a matrix comparison can only fail in one direction: it tells you the two
sides disagree, never *which* of the four things that could mean actually happened.

Three separate claims are pinned here, each of which can break on its own:

1. **The two registries describe the same seventeen transforms** — same names, same
   positional inputs in the same order, same parameter sets. A Rust table that had
   drifted would bind ``price`` to one column while the transform read another: a
   silent feature swap that no matrix comparison can distinguish from a bad model,
   because both sides would still be computing *something* for every row.

2. **The two languages agree on what a spec IS** — the canonical JSON of
   ``BAR_M1_V1`` and ``PERP_CORE_V1``, byte for byte, and therefore their
   fingerprints. Rust recomputes the fingerprint on every load rather than reading
   the recorded one, so this is a conformance check that also runs in production;
   what the fixture adds is the *bytes*, so a disagreement points at the
   serialization rather than at the recipe.

3. **NumPy's reduction order**, as raw bit patterns over a pinned series. This is
   the one that motivated the whole crate: NumPy does not sum a window left to
   right, and a Rust implementation that did would be wrong by up to eight ULP on
   every rolling column. `numeric.rs` transcribes NumPy's pairwise algorithm from
   its source; this fixture pins the transcription against NumPy *itself*, which is
   the only thing that can catch the day NumPy changes its unroll factor. A comment
   claiming to match NumPy is a comment.

**Everything numeric crosses as a hex bit pattern, never as a decimal.** A float
written as a decimal and re-parsed can land on the neighbouring value, and the test
would then be measuring its own serialization — the same rule, for the same reason,
that makes `axon.parity.rust_gate` write raw little-endian matrices.

Regenerating this file rewrites a frozen reference. **The git diff is the review.**
"""

from __future__ import annotations

import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from axon.features.functions import FEATURES_VERSION  # noqa: E402
from axon.features.registry import feature_info, registered_features  # noqa: E402
from axon.features.spec import BAR_M1_V1, BAR_M1_WARMUP_BARS, PERP_CORE_V1  # noqa: E402

HERE = Path(__file__).resolve().parent


def bits(value: float) -> str:
    """A float64 as its IEEE-754 bit pattern, big-endian hex."""
    return f"0x{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def perp_series(n: int) -> list[float]:
    """The series `numeric.rs::perp_series` generates, transcribed.

    Deliberately an LCG rather than ``np.random``: the Rust side has to produce the
    identical series with no NumPy, and a generator simple enough to transcribe in
    six lines is the only kind where "identical" is checkable by reading. The
    values are shaped like real perp closes and — this is the part that matters —
    carry full-mantissa low bits, which is what makes the pairwise and naive
    summation orders round *differently* at all. A readable series like
    ``60_000 + i * 0.1`` has low bits so regular that the two orders agree to the
    bit, and a test written on one passes for the wrong reason and then never fails.
    """
    state = 0x243F6A8885A308D3
    out = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        u = (state >> 11) / float(1 << 53)
        out.append(60000.0 + u * 600.0 - 300.0)
    return out


def naive_sum(xs) -> float:
    """Left-to-right accumulation — what a Rust crate would write by default."""
    res = 0.0
    for v in xs:
        res += v
    return res


def main() -> None:
    series = perp_series(160)
    a = np.asarray(series, dtype=np.float64)

    # The windows the shipped specs actually use (5 and 20), plus the boundaries of
    # NumPy's own algorithm: 7 and 8 straddle the unroll threshold, 128 and 129
    # straddle the recursive split. A transcription that got either boundary wrong
    # would still pass on 5 and 20 alone.
    reductions = []
    for window in (5, 7, 8, 20, 32, 128, 129):
        view = sliding_window_view(a, window)
        last = view[-1]
        reductions.append(
            {
                "window": window,
                "sum": bits(view.sum(axis=1)[-1]),
                "mean": bits(view.mean(axis=1)[-1]),
                "std_ddof0": bits(view.std(axis=1, ddof=0)[-1]),
                "std_ddof1": bits(view.std(axis=1, ddof=1)[-1]),
                # Recorded so the Rust test can assert it is *separated* from the
                # NumPy answer rather than merely equal to it. A transcription that
                # quietly fell back to the naive loop would otherwise look fine on
                # any window where the two happen to agree.
                "naive_sum": bits(naive_sum(last)),
                "naive_separates": bool(view.sum(axis=1)[-1] != naive_sum(last)),
            }
        )

    # `log` is the one operation whose cross-language agreement is measured rather
    # than guaranteed: IEEE-754 requires `+ - * / sqrt` to be correctly rounded and
    # says nothing about `log`. These are the exact ratios a `log_return` over this
    # series produces, with NumPy's answers pinned to the bit.
    ratios = (a[1:] / a[:-1])[:32]
    logs = [
        {"ratio": bits(r), "log": bits(v)} for r, v in zip(ratios, np.log(ratios))
    ]

    # The `-0.0` family. `np.sum` is `identity + DOUBLE_pairwise_sum(...)`, and that
    # outer `0.0 +` is exact for every finite value except one: it turns a total of
    # negative zero into positive zero. A transcription of the inner function alone
    # returns `-0.0`, which is one bit, on one value — and a bit-exact gate is exactly
    # the thing that cannot shrug at one bit. This was a real defect in this crate,
    # found by differential fuzzing, and it is pinned here so it cannot come back.
    #
    # `n` straddles all three branches: the plain loop (< 8, which already starts from
    # +0.0 and so never had the bug), the unrolled tree, and the recursive split.
    negative_zero = []
    for n in (1, 7, 8, 9, 20, 129, 300):
        z = np.full(n, -0.0)
        mixed = np.array([-0.0] * n)
        if n >= 2:
            mixed[n // 2] = 0.0
        negative_zero.append(
            {
                "n": n,
                "all_negzero_sum": bits(z.sum()),
                "all_negzero_mean": bits(z.mean()),
                "all_negzero_std": bits(z.std()),
                "mixed_zero_sum": bits(mixed.sum()),
            }
        )

    payload = {
        "note": (
            "Written by crates/axon-features/tests/fixtures/generate.py. "
            "Regenerating rewrites a frozen reference; the git diff IS the review."
        ),
        "features_version": int(FEATURES_VERSION),
        "bar_m1_warmup_bars": int(BAR_M1_WARMUP_BARS),
        "registry": {
            name: {
                "inputs": list(feature_info(name).inputs),
                # Sorted, because the Rust table declares them sorted: a spec hashes
                # its params sorted, so the Rust registry keeps the same order and
                # the two declarations are only comparable as sets. Python's own
                # order is the function signature's, which is a different fact.
                "params": sorted(feature_info(name).params),
            }
            for name in registered_features()
        },
        "specs": {
            "bar_m1": {
                "json": BAR_M1_V1.to_json(),
                "ref": BAR_M1_V1.ref,
                "fingerprint": BAR_M1_V1.fingerprint,
                "columns": list(BAR_M1_V1.columns),
                "required_inputs": list(BAR_M1_V1.required_inputs),
            },
            "perp_core": {
                "json": PERP_CORE_V1.to_json(),
                "ref": PERP_CORE_V1.ref,
                "fingerprint": PERP_CORE_V1.fingerprint,
                "columns": list(PERP_CORE_V1.columns),
                "required_inputs": list(PERP_CORE_V1.required_inputs),
            },
        },
        "series": [bits(v) for v in series],
        "reductions": reductions,
        "negative_zero": negative_zero,
        "logs": logs,
        "producer": {"python": platform.python_version(), "numpy": np.__version__},
    }

    out = HERE / "cross_language.json"
    out.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )
    separated = sum(1 for r in reductions if r["naive_separates"])
    print(f"wrote {out}")
    print(f"  {len(payload['registry'])} transforms, {len(series)} series values")
    print(f"  {separated}/{len(reductions)} windows separate pairwise from naive summation")
    print(f"  bar_m1 {payload['specs']['bar_m1']['ref']}")
    print(f"  perp_core {payload['specs']['perp_core']['ref']}")


if __name__ == "__main__":
    main()
