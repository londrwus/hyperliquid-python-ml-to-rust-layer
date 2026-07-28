"""The entrypoints hwsched runs *inside* the container, for end-to-end proof.

Not placeholders — these are the jobs that verify the whole chain from Axon
actually works on real Modal: spec emitted here, planned and guarded by hwsched,
image built with the ``axon`` package mounted, function imported by name in the
container, correlation id arriving as an environment variable, and an artifact
landing on a Volume rather than in the return value. :func:`probe` does it on CPU,
:func:`gpu_probe` on a real device.

Three constraints shape them, and all three are load-bearing for anything else
Axon sends to Modal:

**Stdlib only at import time.** hwsched mounts the entrypoint's *top-level
package* into the image with ``add_local_python_source("axon")``, so importing
``axon.compute.entry`` also executes ``axon/__init__.py`` and
``axon/compute/__init__.py``. If any of them reached for numpy, a job would need
``pip=["numpy"]`` just to start. Keeping this path import-clean is what lets a
probe run on a bare ``debian-slim`` — and it is why :func:`gpu_probe` imports
``torch`` *inside itself*: at module scope it would put a 2 GB wheel on the
critical path of every CPU job in the package.

**A plain-typed return value, which is a stricter rule than a small one.** The
result is pickled back through Modal's control plane and unpickled on the
*client*, so any value whose class lives in a third-party module makes the client
import that module. ``torch.__version__`` is a ``TorchVersion`` — a ``str``
subclass — and returning it unwrapped fails a finished GPU run with
"Deserialization failed because the 'torch' module is not available in the local
environment", after the work has been done and paid for. Every receipt field here
is a plain ``str``/``int``/``float``/``bool``/``None``.

**Every exit path writes its receipt.** The CLI never collects per-task return
values at all (ADR-0017 §4), so a diagnosis that is only returned is a diagnosis
nobody reads: a job that found no GPU would come back green with its explanation
discarded. The Volume is the only channel that survives.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from axon.compute.artifacts import path_for
from axon.compute.spec import CORRELATION_ENV


def _persist(receipt: dict[str, Any], artifact: str | None) -> dict[str, Any]:
    """Write *receipt* to its Volume location, and note on it that it went there.

    Called on every exit path, including the failing ones: hwsched's CLI path never
    fetches a task's return value, so a receipt that was only returned is lost when
    the container exits. What lands on disk is the receipt as it stood *before* the
    two keys below, so the file describes the run rather than itself.

    The path comes from :func:`axon.compute.artifacts.path_for` rather than being
    rebuilt here, because a mount path that disagrees with the declared URI does not
    raise — the write succeeds against ephemeral container disk and the bytes are
    simply gone afterwards.
    """
    if not artifact:
        return receipt
    path = Path(path_for(artifact))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, sort_keys=True).encode()
    path.write_bytes(payload)
    receipt["artifact"] = artifact
    receipt["artifact_bytes"] = len(payload)
    return receipt


def probe(seed: int = 0, rounds: int = 400_000, artifact: str | None = None) -> dict[str, Any]:
    """Burn a few CPU-seconds deterministically and report what the container saw.

    *rounds* iterations of a SHA-256 chain: pure CPU, no allocation growth, and the
    same digest on every machine — so a re-run that disagrees means the image or
    the inputs changed, not that the hardware is noisy.

    *artifact* is an optional ``volume://<name>/<subpath>`` to write the receipt
    to. Bytes written outside a mounted Volume disappear when the container exits,
    so the caller must have declared the same volume in the job's ``outputs`` for
    this to persist — that is the pairing :mod:`axon.compute.artifacts` exists to
    keep honest.
    """
    started = time.monotonic()
    digest = hashlib.sha256(str(int(seed)).encode()).digest()
    for _ in range(int(rounds)):
        digest = hashlib.sha256(digest).digest()

    receipt: dict[str, Any] = {
        "seed": int(seed),
        "rounds": int(rounds),
        "digest": digest.hex()[:32],
        "correlation_id": os.environ.get(CORRELATION_ENV),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "machine": platform.machine(),
        "elapsed_s": round(time.monotonic() - started, 3),
    }

    return _persist(receipt, artifact)


def gpu_probe(
    seed: int = 0, size: int = 2048, iters: int = 50, artifact: str | None = None
) -> dict[str, Any]:
    """Prove a GPU container really ran on a real device, and report what it was.

    Deliberately not training anything: *iters* fixed-seed float32 matmuls of a
    *size*×*size* matrix are enough to show the device was allocated and did work,
    and the **receipt** is the point — device name, compute capability, CUDA
    version, a checksum. Nothing here decides whether a model is good enough; that
    stays in :mod:`axon.parity` (ADR-0017 §6).

    TF32 is turned off explicitly. It silently drops matmul mantissa bits on Ampere
    and later, so leaving it on would make the checksum depend on which GPU class
    Modal happened to schedule that day — and a verification number that changes
    with the hardware verifies nothing.

    ``torch`` is imported *inside* this function, not at module scope: mounting the
    package executes the module, so a top-level GPU import would make every CPU job
    in ``axon.compute`` depend on a 2 GB wheel it never uses, and a ``debian-slim``
    probe would stop starting at all.
    """
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "seed": int(seed),
        "size": int(size),
        "iters": int(iters),
        "correlation_id": os.environ.get(CORRELATION_ENV),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "machine": platform.machine(),
    }

    def done(**fields: Any) -> dict[str, Any]:
        receipt.update(fields)
        receipt["elapsed_s"] = round(time.monotonic() - started, 3)
        return _persist(receipt, artifact)

    try:
        import torch
    except ImportError as exc:
        # A spec that forgot pip=["torch"] builds and runs perfectly well; it only
        # fails here. Returning that as a receipt beats a traceback, because the
        # traceback reaches nobody: the CLI does not collect return values and a
        # raised task shows up as an opaque failed chunk.
        return done(error=f"torch is not installed in this container: {exc}",
                    cuda_available=False)

    # str(), never the attribute itself. `torch.__version__` is a TorchVersion — a
    # str *subclass* — and Modal pickles a task's return value back to the client,
    # so a value whose class lives in torch makes the *client* import torch to
    # unpickle it. On a box that deliberately has no torch that turns a finished GPU
    # run into "Deserialization failed because the 'torch' module is not available in
    # the local environment", after the GPU work has completed and been paid for. The
    # Volume artifact survives that; the return value does not. Every field below is
    # a plain str/int/float/bool/None for the same reason.
    receipt["torch"] = str(torch.__version__)
    receipt["cuda_available"] = bool(torch.cuda.is_available())
    receipt["cuda_version"] = None if torch.version.cuda is None else str(torch.version.cuda)

    if not receipt["cuda_available"]:
        # A GPU job that quietly fell back to CPU is the failure this probe exists to
        # make visible: it succeeds, it costs GPU money, and it proves nothing.
        return done(device_count=0,
                    error="no CUDA device visible - the GPU request did not land")

    receipt["device_count"] = int(torch.cuda.device_count())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    receipt["device_name"] = str(torch.cuda.get_device_name(0))
    receipt["capability"] = ".".join(str(int(x)) for x in torch.cuda.get_device_capability(0))
    receipt["total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)

    torch.manual_seed(int(seed))
    device = torch.device("cuda:0")
    a = torch.randn(int(size), int(size), device=device, dtype=torch.float32)
    b = torch.randn(int(size), int(size), device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    matmul_started = time.monotonic()
    c = a
    for _ in range(int(iters)):
        c = torch.matmul(c, b) / float(size) ** 0.5
    torch.cuda.synchronize()
    gpu_s = time.monotonic() - matmul_started

    # .item() off the float64 reduction, never the tensor: a torch scalar pickles as
    # a torch type, which is the same trap `torch.__version__` set above.
    return done(
        checksum=float(c.double().abs().sum().item()),
        gpu_seconds=round(gpu_s, 3),
        gflops=round(2 * int(size) ** 3 * int(iters) / gpu_s / 1e9, 1),
        peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
    )


__all__ = ["gpu_probe", "probe"]
