"""Ask a *different machine* whether the feature gate's arithmetic is portable.

ADR-0035 holds the Rust feature runtime to bit equality with `axon.features`, and
names one gap it cannot close from the dev box: `+ - * / sqrt` are correctly rounded
by IEEE-754 and agree everywhere, while `log` is not required to be and agrees here
only by measurement. Everything in the committed fixture
`crates/axon-features/tests/fixtures/cross_language.json` was produced by *this*
host's NumPy and *this* host's libm.

This job recomputes the fixture's own numbers somewhere else and diffs the bits.
Two separate questions, and they fail differently:

  1. **NumPy's pairwise summation order** — sum/mean/std over the fixture's series.
     A difference here means the *transcription* in `numeric.rs` is pinned to one
     NumPy build, and the gate would redden on any host with another.
  2. **`log`** — the one operation neither language promises. A difference here means
     `libm_columns` stops being a signpost and starts being load-bearing.

Nothing is written and nothing is trained: it reads a JSON fixture, does arithmetic,
and returns a verdict. Bits cross as hex, never as decimals, for the same reason the
fixture stores them that way. It is a CPU container that runs for seconds — this is
about as small as an ADR-0017 offload gets, and it buys a fact the dev box cannot
produce at any price.

**Run it from the repository root**, with the one interpreter that has both `modal`
and `pydantic` (the trap is `.venv`, not `python3` — see the hwsched notes)::

    MODAL_PROFILE=<profile> \\
      ~/hardware-scheduler/dashboard/backend/.venv/bin/python \\
      -m modal run scripts/modal_libm_probe.py

Recorded result, 2026-07-27: glibc **2.36** on gVisor against this box's glibc
**2.39** on an AMD Ryzen 9 5950X — all seven windows' `sum`/`mean`/`std` bit-identical
at both `ddof`, all 32 recorded logarithms bit-identical, and **0 ULP over 200 000
fresh samples across 26 decades**. What that does *not* cover is anything that is not
glibc: musl and macOS are the plausible disagreements and neither has been run.
"""
import json, modal

# Read locally only. The module is imported inside the container too, where the
# repository does not exist — a module-level read is the classic Modal trap and it
# fails at import time, before any of the work this job exists for.
FIXTURE = "crates/axon-features/tests/fixtures/cross_language.json"

# Two libcs, and that is the whole point of the trip. glibc is what the dev box runs,
# so a Debian container only varies the *version* and the microarchitecture; musl is a
# genuinely different `log`, written by different people to a standard that does not
# require either of them to be correctly rounded. If the two disagree, `libm_columns`
# stops being a signpost and becomes load-bearing, and this repo needs to know that
# from a container rather than from a red gate on somebody's laptop.
GLIBC = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.1")
MUSL = modal.Image.from_registry("python:3.12-alpine").pip_install("numpy==2.5.1")
app = modal.App("axon-libm-probe")


@app.function(image=GLIBC, timeout=600)
def probe_glibc(fixture: dict) -> dict:
    return _probe(fixture)


@app.function(image=MUSL, timeout=600)
def probe_musl(fixture: dict) -> dict:
    return _probe(fixture)


def _probe(fixture: dict) -> dict:
    import platform, struct
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    def bits(v):
        return f"0x{struct.unpack('<Q', struct.pack('<d', float(v)))[0]:016x}"

    def unbits(h):
        return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]

    series = [unbits(h) for h in fixture["series"]]
    a = np.asarray(series, dtype=np.float64)

    reductions = []
    for case in fixture["reductions"]:
        w = case["window"]
        view = sliding_window_view(a, w)
        got = {
            "window": w,
            "sum": bits(view.sum(axis=1)[-1]),
            "mean": bits(view.mean(axis=1)[-1]),
            "std_ddof0": bits(view.std(axis=1, ddof=0)[-1]),
            "std_ddof1": bits(view.std(axis=1, ddof=1)[-1]),
        }
        got["agrees"] = all(got[k] == case[k] for k in ("sum", "mean", "std_ddof0", "std_ddof1"))
        reductions.append(got)

    # **Two logs, not one, and the difference is the whole diagnosis.** `np.log` on an
    # array dispatches to NumPy's own SIMD kernel, chosen at import time from the CPU's
    # feature flags; `math.log` on a scalar calls the platform's libm. If the fixture
    # disagrees with the first and agrees with the second, the cause is NumPy's dispatch
    # and not the libc — which is a completely different fact about portability, and the
    # one this probe was originally built on the assumption of being unable to happen.
    import math

    logs_agree, scalar_agree, log_diffs = 0, 0, []
    for case in fixture["logs"]:
        r = unbits(case["ratio"])
        here = bits(np.log(np.asarray([r], dtype=np.float64))[0])
        scalar = bits(math.log(r))
        if here == case["log"]:
            logs_agree += 1
        if scalar == case["log"]:
            scalar_agree += 1
        if here != case["log"] or scalar != case["log"]:
            log_diffs.append(
                {
                    "ratio": case["ratio"],
                    "fixture": case["log"],
                    "np_log": here,
                    "math_log": scalar,
                }
            )

    # And a much wider `log` sweep than the fixture carries, since this is the whole
    # reason for the trip: 26 decades, the same shape the dev box was measured on.
    rng = np.random.default_rng(20260726)
    wide = np.exp(rng.uniform(-30, 30, size=200_000))
    ours = np.log(wide)
    libm = np.array([math.log(float(v)) for v in wide])
    wide_ulp = int(np.max(np.abs(ours.view(np.int64) - libm.view(np.int64))))

    # What the container actually ran on. Without it a differing answer is a mystery
    # rather than a measurement: NumPy picks its `log` kernel from these flags at import
    # time, so two runs of the *same image* on two workers are two different programs.
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        feats = np._core._multiarray_umath.__cpu_features__
        simd = sorted(k for k, v in feats.items() if v and k.startswith("AVX"))
    except Exception:  # noqa: BLE001 — a NumPy without the private attribute is a result
        simd = []

    return {
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu_model,
        "avx": simd,
        "reductions": reductions,
        "reductions_all_agree": all(r["agrees"] for r in reductions),
        "logs_checked": len(fixture["logs"]),
        "logs_agree": logs_agree,
        "logs_agree_scalar_libm": scalar_agree,
        "log_diffs": log_diffs[:5],
        "wide_log_vs_libm_max_ulp": wide_ulp,
    }


@app.local_entrypoint()
def main():
    payload = json.loads(open(FIXTURE).read())
    for label, fn in (("glibc", probe_glibc), ("musl", probe_musl)):
        try:
            out = fn.remote(payload)
        except Exception as exc:  # a missing musl wheel is a result, not a crash
            print(f"=== {label}: COULD NOT RUN — {type(exc).__name__}: {exc}")
            continue
        print(f"=== {label} ===")
        print(json.dumps(out, indent=2)[:3000])
