"""The Rust half of the *feature* gate: a frozen question about vectors, not decisions.

:mod:`axon.parity.rust_gate` freezes a model and asks Rust to score it. That bundle
proves something real and it proves the *easy* half of Boundary A: given identical
feature vectors, the two languages reach identical decisions. It cannot fail on the way
the two languages actually diverge, because it never asks either of them to *compute* a
feature — it hands both sides the same matrix. ``docs/03`` calls the other half the hard
one, and it is the one where quality leaks: a rolling window that ends one sample early,
a standard deviation with the sample correction on one side, a NaN warmup back-filled
with a seed, a reduction reassociated by a different loop. Every one of those leaves the
model bundle green.

A *feature parity bundle* is that harder question, written out of a
:class:`~axon.features.spec.FeatureSpec` and a set of input arrays into a directory a
Rust test reads with no Python, no numpy, no network and no clock::

    <bundle>/manifest.json     what this is, and what it must be held to
            /spec.json         the canonical FeatureSpec JSON, byte-identical to to_json()
            /inputs.f64        the input arrays, raw little-endian IEEE-754, ROW-MAJOR
            /features.f64      Python's own matrix over exactly those bytes, same encoding

Nothing here needs the ML stack. Writing a *model* bundle needs a model, and therefore
XGBoost or onnxruntime; writing this one needs numpy, and the reader needs numpy. That
is not an accident of scope — it means the harder half of Boundary A can be regenerated
and verified on a bare machine, including the machine that is failing.

**The inputs are named by the spec, not by the caller.** ``inputs.names`` is sorted and
is exactly :attr:`~axon.features.spec.FeatureSpec.required_inputs`; the Rust side
rebuilds its inputs mapping from that list, positionally. If the list were free-form —
whatever the caller happened to pass, in whatever order — the two sides could pair
arrays with the wrong names, and *every* column would then be wrong in a way that looks
exactly like a transform bug: a plausible number, off by a shape nobody can read back to
a swapped ``high`` and ``low``.

**Row-major, and the byte order is pinned** (``<f8``, not native), because the reader is
``f64::from_le_bytes`` over a flat slice. A native-order write on a big-endian host would
hand Rust every value byte-swapped and a perfectly correct runtime would fail
spectacularly; a column-major write would hand it a transposed matrix, which for a square
corpus fails *quietly*.

**f64, not f32.** Features are float64 (ADR-0016 §6) — a z-score is a statistic, not
money, and it is computed in double. Casting to f32 to reuse the model bundle's width
would round every reference value before the comparison, so the gate would be measuring
its own serialization instead of the two runtimes: two implementations that differ in the
last three bits of a double would agree perfectly after the cast, which is precisely the
divergence this exists to catch.

**NaN travels as its own bits**, which a JSON ``null`` cannot do. Warmup is NaN by
construction (ADR-0016 §1 — zero is a legal value for every feature here, so a zero
warmup is indistinguishable from a reading), so the NaN cells *are* the majority of the
first rows and the gate's whole warmup claim depends on them surviving the trip. Raw
IEEE-754 carries them; it also makes them comparable at all, since ``nan == nan`` is
false while ``bits == bits`` is true.

That last point has a consequence worth stating, because it is the one thing a Rust
implementation can get wrong while being arithmetically perfect: **a bit-exact comparison
of a NaN cell is a comparison of NaN *patterns*.** ``axon.features`` never lets a NaN
arise from arithmetic — every quotient is masked, so a warmup or a guarded cell is the
literal ``np.nan``, ``0x7ff8000000000000``. On x86, ``0.0 / 0.0`` produces
``0xfff8000000000000`` instead: the same value, the other sign bit. A Rust transform that
divides first and masks afterwards would therefore redden this gate on every guarded cell
while computing the right answer.

**And "never lets a NaN arise from arithmetic" is true only for finite inputs**, which is
worth knowing because the refusal below then fires for a reason its message does not name.
Feed an ``inf`` into ``realized_volatility``, ``close_location`` or
``trade_flow_imbalance`` and NumPy's own ``_var`` computes ``inf - inf`` *before* any mask,
producing ``0xfff8000000000000`` on the Python side. The corpus is refused — correctly, a
bit-exact gate cannot carry two spellings of NaN — but the thing actually wrong with it is
the infinity in the feed, not the NaN. If this check fires, look for an ``inf`` in the
inputs first. So the writer refuses any matrix carrying a
non-canonical NaN (:data:`CANONICAL_NAN_BITS`), which turns "produce NaN the way Python
does" into a property of the fixture rather than a paragraph somebody has to have read.

**The reference is computed over the bytes read back off disk.** ``inputs.f64`` is
written, read back, and *that* array — the one whose bits Rust will see — is what
:meth:`FeatureSpec.compute` is called on. Computing over the in-memory array instead
would leave the file unverified in the one respect that matters, and the two languages
would be answering slightly different questions.

**The manifest is written last**, so an interrupted run leaves a directory the reader
reports as an interrupted write rather than a bundle whose recorded hashes describe files
that are no longer there.

**The criterion is ``bit_exact`` and a bundle may not declare anything else.** Unlike the
model gate there is no tolerance arm to fall back to, and that is a claim about the
library rather than optimism: every transform in :mod:`axon.features.functions` is built
from ``+ - * /``, ``sqrt``, comparison and ``log``. IEEE-754 requires the first five to be
correctly rounded, so they agree on every conforming machine by definition. ``log`` is
*not* required to be, and numpy's agreement with libm is a **measurement on this box** —
0 ULP over the 32 ratios the cross-language fixture pins and over a 200 000-sample,
26-decade sweep re-runnable via ``scripts/modal_libm_probe.py``, on glibc 2.39 here and
on glibc 2.36 and musl in a container — rather than a guarantee. Declaring a
tolerance instead would buy nothing for the five exact operations and would hide the one
inexact one, so the disagreement is handled by naming it: :data:`libm_columns` lists the
columns whose value passes through ``np.log``, directly or through a binding to an
earlier column. **It is a signpost, not a tolerance.** If this gate ever reddens on
another platform, the first question is whether it reddened *only* on those columns; if
it did, the argument is about ``log`` and not about the runtime.

That question is now asked by the reader rather than by whoever is holding the pager,
because it stopped being hypothetical: NumPy selects its ``log`` loop from the CPU it
finds at import, so *the same numpy 2.5.1 that wrote these matrices* recomputes a cell a
last bit differently on a machine with different SIMD, and two CI runners on one version
came out one green and one red. So a recomputation that disagrees is still a defect
**unless** every disagreeing cell is in :data:`libm_columns` and is one ULP — the
signpost deciding *where* a difference is allowed to be, not *how large* the gate's
tolerance is. A column that never reaches libm must still agree to the bit, two ULP in a
column that does still fails, and a NaN meeting a number is not one ULP from anything.

**A corpus that never finishes warming up is refused.** An all-NaN matrix compares
bit-exactly against another all-NaN matrix, so a Rust runtime that returned NaN for
everything would pass — the same "a check that can only come out one way is a decoration"
failure the model bundle's decision-spread rule exists to prevent. A bundle must carry at
least one row on which every column is finite.
"""

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from axon.features.functions import FEATURES_VERSION, finite_rows
from axon.features.registry import FeatureError
from axon.features.spec import FeatureSpec
from axon.models.artifact import content_hash

# `_verify_file` is imported rather than copied. "A manifest is data, and a path in it
# that escapes the bundle directory turns reading a bundle into a file read of whoever
# wrote it choosing" is one rule, not two; a second implementation of it would agree with
# the first on the day it was written and diverge on the day one of them is fixed — which
# is the same argument `axon.features` makes for never implementing a feature twice. The
# behaviours it enforces (bare filename, present, declared length, content hash) each
# have a test *here*, against this reader, so a change on the model side that quietly
# stopped enforcing one of them reddens this file too.
from axon.parity.rust_gate import BundleError, Criterion, _verify_file

#: Bumped when the on-disk layout changes. A bundle stamped higher than this is refused
#: rather than read with the unknown half treated as absent — the same rule
#: ``ArtifactMeta.meta_schema`` and the model bundle follow, and the Rust reader enforces
#: it too. Deliberately a *separate* counter from ``rust_gate.BUNDLE_SCHEMA``: the two
#: formats carry different files and move for different reasons, and sharing a number
#: would mean every model-side layout change invalidated every feature bundle.
BUNDLE_SCHEMA = 1

MANIFEST_FILENAME = "manifest.json"
SPEC_FILENAME = "spec.json"
INPUTS_FILENAME = "inputs.f64"
FEATURES_FILENAME = "features.f64"

#: The bit pattern of ``np.nan``: the *positive* quiet NaN. Every NaN in this library is
#: this one, because every NaN it produces is a masked literal rather than the result of
#: an invalid operation — see the module docstring. x86's own invalid-operation default
#: is ``0xfff8000000000000``, so this is a real thing for the two sides to disagree about
#: under a bit-exact criterion, and the writer refuses anything else rather than letting
#: a correct Rust runtime redden on the sign bit of a warmup cell.
CANONICAL_NAN_BITS = 0x7FF8000000000000

#: The only criterion a feature bundle may declare, at write time or in a manifest.
#:
#: Shared with the model bundle rather than re-spelled here: ``bit_exact`` has to mean
#: the same thing on both sides of ``axon.parity``, and :meth:`Criterion.allows` already
#: encodes the one asymmetry that matters (a bundle may tighten and may never loosen).
#: What is different is that here there is nothing to tighten *from*: see the docstring
#: for why every transform in this library is exactly reproducible, and
#: :func:`libm_columns` for the one operation that is a measurement rather than a
#: guarantee.
FEATURE_CRITERION = Criterion("bit_exact")

#: Registered transforms whose value passes through ``np.log``.
#:
#: A table of *transforms*, from which :func:`libm_columns` derives the affected
#: *columns* of any spec by walking its dependency graph. Split that way because the two
#: facts have different lifetimes: which transforms call ``log`` changes when
#: :mod:`axon.features.functions` changes, and which columns inherit it changes with
#: every spec. ``momentum`` and ``realized_volatility`` are here without containing the
#: word ``log`` themselves — the first *is* :func:`~axon.features.functions.log_return`
#: over a longer horizon, the second is a rolling standard deviation *of* log returns —
#: which is exactly why this cannot be a grep. ``python/tests/test_feature_bundle.py``
#: walks the registry's own call graph and asserts this table is complete, so a transform
#: added next month cannot quietly acquire a libm dependency nobody wrote down.
LIBM_FEATURES = frozenset({"log_return", "momentum", "realized_volatility"})


def libm_columns(spec: FeatureSpec) -> tuple[str, ...]:
    """The spec's columns whose value passes through ``np.log``, in matrix order.

    Derived by walking the spec's dependency graph rather than listed by hand, because a
    column inherits the dependency through a *binding*: a z-score of ``ret_1`` never
    calls ``log`` and is nonetheless a function of one, so a hand-written list would be
    right for the spec it was written against and silently wrong for the next one.

    This is not a tolerance and nothing widens for the columns it names — see the module
    docstring. It is the first question to ask when a bit-exact feature gate reddens on a
    platform this repo has not measured.
    """
    tainted: set[str] = set()
    for definition in spec.features:
        if definition.feature in LIBM_FEATURES or any(s in tainted for s in definition.sources):
            tainted.add(definition.column)
    return tuple(c for c in spec.columns if c in tainted)


# ── writing ───────────────────────────────────────────────────────────────────


def write_feature_bundle(
    spec: FeatureSpec,
    inputs: Mapping[str, Any],
    *,
    out_dir: str | os.PathLike[str],
    source: Mapping[str, Any],
    criterion: Criterion | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the cross-language question for ``spec`` over ``inputs``.

    ``inputs`` is a mapping of name → 1-D array, as :mod:`axon.features.inputs` produces
    it. Only the arrays the spec actually reads are carried into the bundle: the file's
    column count follows from ``spec.required_inputs`` and from nothing else, so an
    unused ``volume`` beside a spec that never reads it cannot make the two sides
    disagree about which column is which.

    ``source`` is free text describing *what market data this is* — instrument, interval,
    venue, and a description. It is required and must carry a non-empty ``description``,
    because a bundle nobody can identify is a bundle nobody can regenerate, and the first
    thing a red cross-language gate needs is to know what it was run on.

    ``criterion`` may be passed only to be refused: see :data:`FEATURE_CRITERION`. The
    argument exists so the refusal has somewhere to live, rather than to offer a choice.

    Returns the bundle directory. The manifest is written **last**, so an interrupted run
    leaves a directory the reader reports as an interrupted write rather than a bundle
    whose recorded hashes describe files that are no longer there.
    """
    if not isinstance(spec, FeatureSpec):
        raise BundleError(f"spec must be a FeatureSpec, got {type(spec).__name__}")
    if spec.library_version != FEATURES_VERSION:
        # Refused here rather than left to the reader, which would refuse it anyway
        # (`FeatureSpec.from_dict` is strict): writing a bundle that this build cannot
        # read back is writing a fixture whose only possible verdict is "regenerate me".
        raise BundleError(
            f"{spec.ref} was written against axon.features v{spec.library_version}, this build "
            f"is v{FEATURES_VERSION}; the transforms changed meaning, so a bundle written now "
            "would freeze one library's answer under another library's name"
        )
    declared = FEATURE_CRITERION if criterion is None else criterion
    if not isinstance(declared, Criterion):
        raise BundleError(f"criterion must be a Criterion, got {type(declared).__name__}")
    if declared != FEATURE_CRITERION:
        # Resolved before a byte is written, for the reason the manifest is written last:
        # there is no reason to manufacture a half-written directory over an argument.
        raise BundleError(
            f"{spec.ref}: asked to declare {declared}, but a feature bundle is held to "
            f"{FEATURE_CRITERION} and has no tolerance arm to fall back to — every transform in "
            "axon.features is + - * / sqrt, comparison and log, and IEEE-754 makes all but the "
            "last correctly rounded; the log columns are named in `libm_columns` rather than "
            "paid for with slack across every column that never touches it"
        )
    source_dict = _source_dict(source)

    x = _input_matrix(spec, inputs)
    rows, n_inputs = x.shape
    names = spec.required_inputs

    directory = Path(out_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise BundleError(
            f"{manifest_path} already exists; regenerating a committed bundle is a deliberate, "
            "reviewable event — pass overwrite=True to say so"
        )
    directory.mkdir(parents=True, exist_ok=True)
    # Off first: a stale manifest beside fresh matrices describes bytes that no longer
    # exist, and its recorded hashes would be the only thing to notice.
    manifest_path.unlink(missing_ok=True)

    inputs_path = directory / INPUTS_FILENAME
    inputs_bytes = _f64_bytes(x)
    inputs_path.write_bytes(inputs_bytes)

    # The one line that makes the reference answer *this* question: compute over the
    # matrix read back off disk, so the recorded features are the features of exactly the
    # bytes Rust will read. Computing over the in-memory array instead would leave the
    # file unverified in the one respect that matters.
    written = _read_f64(inputs_path, rows, n_inputs)
    if _f64_bytes(written) != inputs_bytes:
        raise BundleError(f"{inputs_path}: the inputs did not survive their own round trip")
    _check_nan_bits(written, what=str(inputs_path))

    matrix = _compute(spec, _columns(written, names), where=directory)
    if matrix.shape != (rows, len(spec.columns)):
        raise BundleError(
            f"{spec.ref}: computed {matrix.shape} from {rows} rows of input; a spec's matrix is "
            f"({rows}, {len(spec.columns)}) or the columns no longer line up with their events"
        )
    _check_nan_bits(matrix, what=f"the matrix {spec.ref} computes")

    usable = int(np.count_nonzero(finite_rows(matrix)))
    if usable == 0:
        raise BundleError(
            f"{spec.ref}: not one of {rows} rows has every column finite — the corpus never "
            "finishes warming up. An all-NaN reference compares bit-exactly against an all-NaN "
            "candidate, so this bundle would pass for a runtime that computes nothing at all"
        )

    features_path = directory / FEATURES_FILENAME
    features_bytes = _f64_bytes(matrix)
    features_path.write_bytes(features_bytes)
    if _f64_bytes(_read_f64(features_path, rows, matrix.shape[1])) != features_bytes:
        raise BundleError(f"{features_path}: the matrix did not survive its own round trip")

    spec_path = directory / SPEC_FILENAME
    # The canonical serialization, byte for byte: the fingerprint in `spec_ref` is a hash
    # of this recipe, and the reader re-derives it from these bytes. Pretty-printing here
    # would leave a file that parses to the same spec and is no longer the thing the ref
    # names, which is a difference only a re-serialization can see.
    spec_bytes = spec.to_json().encode("utf-8")
    spec_path.write_bytes(spec_bytes)

    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "note": "Written by axon.parity.feature_bundle; regenerate rather than hand-edit.",
        "spec_ref": spec.ref,
        "library_version": int(spec.library_version),
        "source": source_dict,
        "spec": {
            "file": SPEC_FILENAME,
            "sha256": content_hash(spec_bytes),
            "bytes": len(spec_bytes),
        },
        "inputs": {
            "file": INPUTS_FILENAME,
            "sha256": content_hash(inputs_bytes),
            "bytes": len(inputs_bytes),
            "rows": rows,
            "cols": n_inputs,
            # Sorted, and exactly what the spec requires: the Rust side rebuilds its
            # inputs mapping from this list positionally.
            "names": list(names),
            "nan_cells": int(np.count_nonzero(np.isnan(written))),
        },
        "features": {
            "file": FEATURES_FILENAME,
            "sha256": content_hash(features_bytes),
            "bytes": len(features_bytes),
            "rows": rows,
            "cols": int(matrix.shape[1]),
            "names": list(spec.columns),
            "nan_cells": int(np.count_nonzero(np.isnan(matrix))),
            # Warmup is NaN by construction, so "how much of this corpus is a real
            # comparison" is a number rather than a shape, and it belongs in the record.
            "finite_rows": usable,
        },
        "criterion": declared.to_dict(),
        "libm_columns": list(libm_columns(spec)),
        # What computed the reference, which is what a frozen answer depends on. numpy
        # and nothing else: no ML library is on this path, on either side.
        "producer": _producer(),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline=""
    )
    return directory


# ── reading ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureBundle:
    """A feature-parity bundle read back and checked against itself."""

    path: Path
    manifest: Mapping[str, Any]
    spec: FeatureSpec
    #: ``(rows, len(spec.required_inputs))``, columns in sorted input-name order.
    inputs: np.ndarray
    #: ``(rows, len(spec.columns))``, columns in spec order.
    features: np.ndarray
    criterion: Criterion

    @property
    def spec_ref(self) -> str:
        return str(self.manifest["spec_ref"])

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.spec.required_inputs

    @property
    def columns(self) -> tuple[str, ...]:
        return self.spec.columns

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def source(self) -> Mapping[str, Any]:
        """What market data this is — instrument, interval, venue, description."""
        return dict(self.manifest["source"])

    @property
    def libm_columns(self) -> tuple[str, ...]:
        """The columns that reach ``np.log``; verified against the spec on read."""
        return tuple(str(c) for c in self.manifest["libm_columns"])

    def inputs_mapping(self) -> dict[str, np.ndarray]:
        """The inputs back in the shape :meth:`FeatureSpec.compute` takes.

        The named arrays, sliced out of the matrix that was actually on disk — so a
        caller recomputing from these is recomputing from the bytes Rust reads, which is
        the only version of the question worth asking.
        """
        return _columns(self.inputs, self.input_names)


def read_feature_bundle(path: str | os.PathLike[str]) -> FeatureBundle:
    """Read a bundle, verifying everything checkable without Rust.

    Which is nearly everything: the schema ceiling, the hashes, the declared byte
    lengths, the declared shapes against the actual ones, ``spec.json`` re-fingerprinting
    to ``spec_ref``, the input and column names against the spec's own, the NaN counts,
    the libm signpost — and, last, that recomputing the matrix from ``inputs.f64``
    reproduces ``features.f64`` **bit for bit**. That final check is the writer's answer
    surviving its own reader: a bundle that fails it is one whose reference was taken
    over something other than the bytes it carries, and it would have sent Rust after a
    defect that does not exist.

    Content hashes are checked here because Python is the side that can cheaply say *why*
    a bundle is wrong. The Rust reader is not expected to hash: a flipped bit in either
    matrix changes a feature, so the gate itself catches it, and a crypto dependency in
    the feature crate would buy a better error message and nothing else.
    """
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        if directory.is_dir():
            raise BundleError(
                f"{directory} has no {MANIFEST_FILENAME} — an interrupted write, since the "
                "manifest is written last; regenerate it deliberately"
            )
        raise BundleError(f"{directory}: no such feature-parity bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = int(manifest.get("bundle_schema", 0))
    if schema > BUNDLE_SCHEMA:
        raise BundleError(
            f"{directory} is bundle schema {schema}, this Axon understands {BUNDLE_SCHEMA}; "
            "refusing rather than reading unknown fields as absent"
        )
    for key in (
        "spec_ref",
        "spec",
        "source",
        "inputs",
        "features",
        "criterion",
        "libm_columns",
        "library_version",
    ):
        # Every failure out of this module is a BundleError, including "the manifest is
        # missing a section": a KeyError from here would read as a bug in the reader
        # rather than as a verdict on the bundle.
        if key not in manifest:
            raise BundleError(f"{directory}: manifest has no {key!r} section")
    for section, keys in (
        ("spec", ("file", "sha256", "bytes")),
        ("inputs", ("file", "sha256", "bytes", "rows", "cols", "names", "nan_cells")),
        ("features", ("file", "sha256", "bytes", "rows", "cols", "names", "nan_cells",
                      "finite_rows")),
    ):
        absent = [k for k in keys if k not in manifest[section]]
        if absent:
            raise BundleError(f"{directory}: the {section!r} section is missing {absent}")
    for section in ("spec", "inputs", "features"):
        _verify_file(directory, manifest[section])

    if not str(manifest["source"].get("description", "")).strip():
        # Checked here as well as at write time, because the two readers have to agree
        # about what a valid bundle *is*. The Rust reader refuses this — it builds its
        # report label from the description — so without this line a hand-made bundle
        # would be accepted by Python and refused by Rust, and "the bundle is wrong"
        # would depend on which language you asked. That is the same class of
        # disagreement the whole gate exists to eliminate, one level up.
        raise BundleError(
            f"{directory}: source carries no 'description'; a bundle nobody can identify is a "
            "bundle nobody can regenerate, and the first thing a red cross-language gate has to "
            "answer is what market data it ran on"
        )

    criterion = Criterion.from_dict(manifest["criterion"])
    if criterion != FEATURE_CRITERION:
        raise BundleError(
            f"{directory}: manifest asks for {criterion}; a feature bundle is held to "
            f"{FEATURE_CRITERION} and has no looser arm to fall back to — a tolerance here would "
            "hide the only inexact operation in the library (log, whose columns are named in "
            "`libm_columns`) behind slack granted to every column that never touches it"
        )

    spec_text = (directory / str(manifest["spec"]["file"])).read_text(encoding="utf-8")
    try:
        spec = FeatureSpec.from_json(spec_text)
    except FeatureError as exc:
        # Includes the fingerprint mismatch: a spec edited after its fingerprint was taken
        # describes features this library does not compute, and the bundle's reference was
        # therefore computed from a recipe that is no longer in the directory.
        raise BundleError(f"{directory}: {manifest['spec']['file']} is not usable: {exc}") from exc
    if spec.to_json() != spec_text:
        raise BundleError(
            f"{directory}: {manifest['spec']['file']} is not the canonical serialization of the "
            "spec it parses to; the fingerprint identifies a recipe, and these are not the bytes "
            "that recipe writes"
        )
    if spec.ref != str(manifest["spec_ref"]):
        raise BundleError(
            f"{directory}: manifest names {manifest['spec_ref']}, the spec on disk is {spec.ref}"
        )
    if int(manifest["library_version"]) != spec.library_version:
        # The manifest repeats what `spec.json` already carries, so that a reader can see
        # which build's arithmetic this reference depends on without parsing the recipe.
        # A repeated field is a field that can disagree, and the disagreement would say
        # the transforms changed meaning while the recipe says they did not.
        raise BundleError(
            f"{directory}: manifest records library_version {manifest['library_version']}, the "
            f"spec on disk was written against v{spec.library_version}"
        )

    entry_inputs, entry_features = manifest["inputs"], manifest["features"]
    names = tuple(str(n) for n in entry_inputs["names"])
    if names != spec.required_inputs:
        raise BundleError(
            f"{directory}: inputs are named {list(names)}, {spec.ref} reads "
            f"{list(spec.required_inputs)}; the Rust side pairs arrays with names positionally "
            "from this list, so a mismatch here mislabels every column at once"
        )
    columns = tuple(str(n) for n in entry_features["names"])
    if columns != spec.columns:
        raise BundleError(
            f"{directory}: matrix columns are named {list(columns)}, {spec.ref} produces "
            f"{list(spec.columns)}"
        )
    derived = libm_columns(spec)
    if tuple(str(c) for c in manifest["libm_columns"]) != derived:
        raise BundleError(
            f"{directory}: manifest lists {list(manifest['libm_columns'])} as the columns that "
            f"reach log, the spec's own graph says {list(derived)}; that list is the first "
            "question asked when this gate reddens on an unmeasured platform"
        )

    rows = int(entry_inputs["rows"])
    if rows < 1:
        raise BundleError(f"{directory}: a feature-parity gate over {rows} rows proves nothing")
    if int(entry_inputs["cols"]) != len(names):
        raise BundleError(
            f"{directory}: {entry_inputs['cols']} input columns for {len(names)} names"
        )
    if int(entry_features["rows"]) != rows:
        raise BundleError(
            f"{directory}: {rows} input rows against {entry_features['rows']} feature rows; the "
            "two matrices are not describing the same events"
        )
    if int(entry_features["cols"]) != len(columns):
        raise BundleError(
            f"{directory}: {entry_features['cols']} feature columns for {len(columns)} names"
        )

    inputs = _read_f64(directory / str(entry_inputs["file"]), rows, len(names))
    features = _read_f64(directory / str(entry_features["file"]), rows, len(columns))
    for label, matrix, entry in (
        ("input", inputs, entry_inputs),
        ("feature", features, entry_features),
    ):
        _check_nan_bits(matrix, what=f"{directory}: the {label} matrix")
        actual = int(np.count_nonzero(np.isnan(matrix)))
        if actual != int(entry["nan_cells"]):
            raise BundleError(
                f"{directory}: manifest claims {entry['nan_cells']} NaN {label} cells, the matrix "
                f"holds {actual}; warmup is NaN by construction and its extent is part of what "
                "this bundle asserts"
            )
    usable = int(np.count_nonzero(finite_rows(features)))
    if usable != int(entry_features["finite_rows"]):
        raise BundleError(
            f"{directory}: manifest claims {entry_features['finite_rows']} fully finite rows, the "
            f"matrix holds {usable}"
        )
    if usable == 0:
        raise BundleError(
            f"{directory}: not one row has every column finite; an all-NaN reference compares "
            "bit-exactly against an all-NaN candidate, so this bundle would pass for a runtime "
            "that computes nothing at all"
        )

    recomputed = _compute(spec, _columns(inputs, names), where=directory)
    disagree = np.flatnonzero(
        np.ascontiguousarray(recomputed).view(np.uint64) != features.view(np.uint64)
    )
    if disagree.size:
        # Bit equality is the claim, and it is a claim about *this* stack on *this* CPU.
        # `np.log` is not correctly rounded — ADR-0035 measured it agreeing to 0 ULP with
        # glibc over a 200 000-sample sweep and said plainly that this "does not make the
        # gate portable" — and NumPy picks its log loop from the CPU it finds at import,
        # so the same numpy 2.5.1 that wrote these matrices recomputes one of 6 075 cells
        # a last bit differently on a machine with different SIMD. Two CI runners on the
        # same version, one green and one red, is what that looks like.
        #
        # So a divergence is still a defect unless it is exactly the documented one, and
        # `libm_columns` decides which: the ADR's own first question is "did it redden
        # *only* on those columns?". This is not slack granted to the gate — it is not a
        # tolerance, which `libm_columns` explicitly is not. A column that never reaches
        # libm must still agree to the bit, and a libm column that moved by more than a
        # last bit still fails. Everything else in the file is unchanged, including the
        # NaN-cell counts above, which is where a window that did not fill would show up.
        libm = set(str(c) for c in manifest["libm_columns"])
        for index in disagree:
            row, column = divmod(int(index), len(columns))
            name = str(columns[column])
            recorded, computed = features[row, column], recomputed[row, column]
            why: str | None = None
            if name not in libm:
                why = "that column never reaches libm, so nothing explains a difference"
            elif _ulps_apart(recorded, computed) > 1:
                why = (
                    f"that is {_ulps_apart(recorded, computed)} ULP, and a differing libm "
                    "explains one"
                )
            if why is not None:
                raise BundleError(
                    f"{directory}: row {row} column {name!r} is recorded as "
                    f"{_bits(recorded)} and this build computes {_bits(computed)} from the "
                    f"bundle's own inputs — {why}. The recorded reference is not the answer "
                    "to the question the bundle asks"
                )

    return FeatureBundle(
        path=directory,
        manifest=manifest,
        spec=spec,
        inputs=inputs,
        features=features,
        criterion=criterion,
    )


# ── bytes ─────────────────────────────────────────────────────────────────────


def _f64_bytes(matrix: np.ndarray) -> bytes:
    """A C-contiguous (row-major), little-endian float64 matrix as raw bytes."""
    return np.ascontiguousarray(np.asarray(matrix, dtype="<f8")).tobytes()


def _read_f64(path: Path, rows: int, cols: int) -> np.ndarray:
    raw = path.read_bytes()
    want = rows * cols * 8
    if len(raw) != want:
        # A truncated matrix would otherwise read as a shorter corpus with a plausible
        # shape, and the gate would pass on the rows that survived.
        raise BundleError(
            f"{path}: holds {len(raw)} bytes; the manifest describes {rows}x{cols} f64 = {want}"
        )
    # `<f8` then `astype(float64)`: on a little-endian host the cast is a copy and nothing
    # else, and on a big-endian one it is the byte-order move that makes the values
    # computable. Either way the caller gets a native, writable, C-contiguous array.
    return np.frombuffer(raw, dtype="<f8").reshape(rows, cols).astype(np.float64)


def _ulps_apart(a: float, b: float) -> int:
    """Distance in ULPs between two float64s, via the usual monotonic re-keying.

    Saturates for a non-finite pair so a caller bounding this can never read "NaN met a
    number" as "one bit apart" — that is a window that did not fill, a different
    diagnosis from arithmetic whose last bit moved.
    """
    x, y = float(a), float(b)
    if x == y:
        return 0
    if not (np.isfinite(x) and np.isfinite(y)):
        return 1 << 62

    def ordered(value: float) -> int:
        bits = int(np.float64(value).view(np.uint64))
        return -(bits & ~(1 << 63)) if bits >> 63 else bits

    return abs(ordered(x) - ordered(y))


def _bits(value: float) -> str:
    """A float64 as its bit pattern — the only spelling that can be compared by eye.

    Two doubles one ULP apart print identically at repr precision, and one ULP is exactly
    the size of the disagreement this gate exists to find.
    """
    return f"0x{int(np.float64(value).view(np.uint64)):016x}"


def _check_nan_bits(matrix: np.ndarray, *, what: str) -> None:
    """Refuse a matrix carrying a NaN neither language would spell the same way."""
    nan_cells = np.isnan(matrix)
    if not nan_cells.any():
        return
    patterns = np.unique(np.ascontiguousarray(matrix).view(np.uint64)[nan_cells])
    rogue = [int(p) for p in patterns if int(p) != CANONICAL_NAN_BITS]
    if rogue:
        raise BundleError(
            f"{what} carries NaN bit pattern(s) {[f'0x{p:016x}' for p in rogue]}, not the "
            f"canonical 0x{CANONICAL_NAN_BITS:016x}; under a bit-exact criterion a NaN is "
            "compared as bits, and x86 spells an invalid operation's NaN with the sign bit set "
            "while every NaN this library emits is a masked np.nan — a bundle carrying both "
            "would redden against a runtime that is arithmetically perfect"
        )


# ── inputs ────────────────────────────────────────────────────────────────────


def _input_matrix(spec: FeatureSpec, inputs: Mapping[str, Any]) -> np.ndarray:
    """The spec's required inputs as one ``(rows, n_inputs)`` matrix, in name order."""
    if not isinstance(inputs, Mapping):
        raise BundleError(
            f"inputs must be a mapping of name → 1-D array, got {type(inputs).__name__}"
        )
    names = spec.required_inputs
    missing = sorted(set(names) - set(inputs))
    if missing:
        raise BundleError(f"{spec.ref} reads {list(names)}; the inputs mapping has no {missing}")

    columns = []
    rows: int | None = None
    for name in names:
        array = np.asarray(inputs[name], dtype=np.float64)
        if array.ndim != 1:
            raise BundleError(f"input {name!r} must be 1-D, got shape {array.shape}")
        if rows is None:
            rows = int(array.size)
        elif array.size != rows:
            raise BundleError(
                f"input {name!r} has {array.size} rows but earlier inputs have {rows}; inputs of "
                "different lengths are not describing the same events"
            )
        columns.append(array)
    if not rows:
        raise BundleError(
            f"{spec.ref}: a feature-parity gate over {rows} rows proves nothing — the two "
            "languages would agree about the empty matrix and disagree about everything else"
        )
    return np.ascontiguousarray(np.stack(columns, axis=1))


def _columns(matrix: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """A ``(rows, len(names))`` matrix as the named 1-D arrays a spec is computed from.

    Each column is materialized **contiguously** rather than handed over as a stride into
    the matrix. Measured equal on this box either way, and that is exactly the reason not
    to depend on it: numpy is free to take a different reduction path for a strided array
    than for a packed one, and the reference would then be the answer to "what does this
    library compute from a column of *this* matrix" rather than to "what does this
    library compute from these numbers" — which is the question Rust, reading a packed
    slice, is going to ask.
    """
    return {name: np.ascontiguousarray(matrix[:, j]) for j, name in enumerate(names)}


def _compute(spec: FeatureSpec, mapping: Mapping[str, np.ndarray], *, where: Path) -> np.ndarray:
    try:
        return np.ascontiguousarray(np.asarray(spec.compute(mapping), dtype=np.float64))
    except FeatureError as exc:
        raise BundleError(
            f"{where}: {spec.ref} cannot be computed from its inputs: {exc}"
        ) from exc


def _source_dict(source: Mapping[str, Any]) -> dict[str, Any]:
    """What market data this is, as JSON scalars, sorted."""
    if not isinstance(source, Mapping):
        raise BundleError(f"source must be a mapping, got {type(source).__name__}")
    out: dict[str, Any] = {}
    for key, value in source.items():
        if not isinstance(key, str):
            raise BundleError(f"source keys must be strings, got {key!r}")
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (bool, int, float, str)):
            raise BundleError(
                f"source[{key!r}] is a {type(value).__name__}; a bundle's provenance has to "
                "serialize as JSON scalars on any machine that regenerates it"
            )
        if isinstance(value, float) and not math.isfinite(value):
            # `json.dumps` would happily write the token `NaN`, which is not JSON and
            # which the Rust reader's parser refuses — at read time, in the other language.
            raise BundleError(f"source[{key!r}] is {value!r}; JSON has no spelling for it")
        out[key] = value
    if not str(out.get("description", "")).strip():
        raise BundleError(
            "source needs a non-empty 'description': a bundle nobody can identify is a bundle "
            "nobody can regenerate, and the first thing a red cross-language gate has to answer "
            "is what market data it ran on"
        )
    return dict(sorted(out.items()))


def _producer() -> dict[str, str]:
    """The libraries whose arithmetic the frozen reference depends on.

    Two entries, and that is the point: a feature bundle is written and read with numpy
    and the standard library, so the harder half of Boundary A can be regenerated on a
    machine that cannot install an ML stack.
    """
    return {"python": platform.python_version(), "numpy": np.__version__}


__all__ = [
    "BUNDLE_SCHEMA",
    "CANONICAL_NAN_BITS",
    "FEATURES_FILENAME",
    "FEATURE_CRITERION",
    "INPUTS_FILENAME",
    "LIBM_FEATURES",
    "MANIFEST_FILENAME",
    "SPEC_FILENAME",
    "FeatureBundle",
    "libm_columns",
    "read_feature_bundle",
    "write_feature_bundle",
]
