"""``axon.parity.feature_bundle``: the frozen question about feature *vectors*.

The sibling file ``test_rust_parity.py`` covers the model bundle, which proves that
identical feature vectors produce identical decisions. This one covers the half
``docs/03`` calls harder and ADR-0021 could not gate: that the two languages compute
identical vectors *from the same market data*. The Rust half lives in the
``axon-features`` crate and reads these directories with no Python in the process; this
file covers what Rust cannot — that the bundle a research machine writes records the
answer Python actually gave, over exactly the bytes it wrote down.

Two tests are the point of the file.
``test_the_recorded_matrix_is_what_the_spec_computes_from_the_bytes_that_were_written``
is the serialization guarantee: if the reference were taken over the in-memory arrays
instead of over ``inputs.f64``, the two languages would be answering slightly different
questions and the gate would report the difference as a runtime defect.
``test_a_corpus_that_never_finishes_warming_up_is_refused_because_all_nan_passes_bit_exactly``
is the one that keeps the gate from being a decoration: an all-NaN reference compares
bit-exactly against an all-NaN candidate.

Every test is named after the failure mode it prevents. Nothing here touches the network,
a clock, or an ML library: writing a *model* bundle needs XGBoost or onnxruntime, and
writing this one needs numpy — which is what lets the harder half of Boundary A be
regenerated on the machine that is failing.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re

import numpy as np
import pytest

from axon.features import BAR_M1_V1, PERP_CORE_V1, FeatureDef, FeatureSpec
from axon.features.functions import FEATURES_VERSION, finite_rows
from axon.features.inputs import bar_inputs
from axon.features.registry import feature_info, registered_features
from axon.features.spec import BAR_M1_WARMUP_BARS
from axon.models.artifact import content_hash
from axon.parity.feature_bundle import (
    BUNDLE_SCHEMA,
    CANONICAL_NAN_BITS,
    FEATURE_CRITERION,
    LIBM_FEATURES,
    BundleError,
    Criterion,
    FeatureBundle,
    libm_columns,
    read_feature_bundle,
    write_feature_bundle,
)

#: Long enough that the corpus outlives ``BAR_M1_V1``'s warmup by an order of magnitude
#: and short enough that the whole file runs in a second. Every count asserted below is
#: derived from this and from :data:`BAR_M1_WARMUP_BARS`, never typed.
ROWS = 240

SOURCE = {
    "description": "synthetic geometric random walk, closed m1 bars, seeded",
    "instrument": "BTC",
    "interval": "1m",
    "venue": "synthetic",
}


def bars(rows: int = ROWS, *, seed: int = 7) -> dict[str, np.ndarray]:
    """A deterministic run of closed OHLCV bars, as fixed-point integers off the wire.

    Through :func:`axon.features.inputs.bar_inputs` rather than as hand-made floats,
    because that adapter is the one place fixed-point becomes float and a bundle written
    from floats invented by a test would freeze a corpus no feed can produce.

    A geometric random walk with a half-range around each close: every bar satisfies
    ``high >= close >= low > 0``, so the corpus exercises the finite path of every column
    in ``BAR_M1_V1`` rather than its NaN guards.
    """
    rng = np.random.default_rng(seed)
    close = 60_000.0 * np.exp(np.cumsum(rng.normal(0.0, 2e-4, rows)))
    half = np.abs(rng.normal(0.0, 8e-4, rows)) * close + 0.5
    open_px = np.concatenate(([close[0]], close[:-1]))
    volume = rng.uniform(1.0, 50.0, rows)
    fixed = lambda a: np.round(a * 100_000_000).astype(np.int64)  # noqa: E731
    return bar_inputs(
        fixed(open_px), fixed(close + half), fixed(close - half), fixed(close), fixed(volume)
    )


def written(tmp_path, *, spec=BAR_M1_V1, inputs=None, name="bundle", **kwargs):
    """A bundle on disk, with the arguments every test would otherwise repeat."""
    return write_feature_bundle(
        spec,
        bars() if inputs is None else inputs,
        out_dir=tmp_path / name,
        source=SOURCE,
        **kwargs,
    )


def manifest_of(directory) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def corrupt(path, mutate):
    """Rewrite ``path`` through ``mutate``, **asserting the bytes actually moved**.

    The handoff records why this is a helper rather than three inline edits: a
    ``.replace()`` whose pattern stopped matching left one of the model bundle's refusal
    tests corrupting nothing at all, and the assertion then asked its question of a
    byte-identical copy — a test that passed for a year while testing nothing.
    """
    before = path.read_bytes()
    after = mutate(before)
    assert after != before, (
        f"{path.name}: this edit changed nothing, so the refusal below would be asked of an "
        "untouched bundle and would pass by accident"
    )
    path.write_bytes(after)


def edit_manifest(directory, mutate):
    """Apply ``mutate`` to the manifest, asserting the serialized text moved."""
    path = directory / "manifest.json"
    before = path.read_text(encoding="utf-8")
    manifest = json.loads(before)
    mutate(manifest)
    after = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    assert after != before, "this manifest edit changed nothing; the refusal below is vacuous"
    path.write_text(after, encoding="utf-8", newline="")


def restamp(directory, section):
    """A manifest edit that re-records a file's length and hash.

    The interesting tamper is the one that leaves the bundle self-consistent: with the
    hash restamped, only the *meaning* of the bytes can catch the edit.
    """

    def apply(manifest):
        payload = (directory / manifest[section]["file"]).read_bytes()
        manifest[section].update(sha256=content_hash(payload), bytes=len(payload))

    return apply


# ── the serialization guarantee ───────────────────────────────────────────────


def test_the_recorded_matrix_is_what_the_spec_computes_from_the_bytes_that_were_written(tmp_path):
    # The whole bundle in one assertion: the features on disk are what this library
    # computes from the *inputs on disk*, bit for bit. Taken over the in-memory arrays
    # instead, the file would be unverified in the one respect that matters — the bits
    # Rust reads — and any divergence introduced by the write would be reported as a Rust
    # defect months later.
    directory = written(tmp_path)
    bundle = read_feature_bundle(directory)
    assert isinstance(bundle, FeatureBundle)

    raw = (directory / "inputs.f64").read_bytes()
    from_disk = np.frombuffer(raw, dtype="<f8").reshape(ROWS, len(BAR_M1_V1.required_inputs))
    recomputed = BAR_M1_V1.compute(
        {name: from_disk[:, j] for j, name in enumerate(BAR_M1_V1.required_inputs)}
    )
    recorded = np.frombuffer((directory / "features.f64").read_bytes(), dtype="<f8")
    assert np.array_equal(recomputed.reshape(-1).view(np.uint64), recorded.view(np.uint64))
    assert bundle.spec_ref == BAR_M1_V1.ref == "bar_m1/v1#c503688de24e863f"
    assert bundle.criterion == FEATURE_CRITERION == Criterion("bit_exact")
    assert bundle.rows == ROWS


def test_the_matrices_cross_row_major_as_pinned_little_endian_float64(tmp_path):
    # `<f8`, not native: the reader is `f64::from_le_bytes`, so a native-order write on a
    # big-endian host hands Rust every value byte-swapped and a perfectly correct runtime
    # fails spectacularly. Row-major, because a column-major write is a transposed matrix
    # — which for a square corpus fails *quietly*.
    directory = written(tmp_path)
    manifest = manifest_of(directory)
    inputs = bars()
    rows, cols = manifest["inputs"]["rows"], manifest["inputs"]["cols"]
    assert (rows, cols) == (ROWS, len(BAR_M1_V1.required_inputs))
    assert manifest["inputs"]["bytes"] == rows * cols * 8

    raw = (directory / "inputs.f64").read_bytes()
    first_row = np.frombuffer(raw[: cols * 8], dtype="<f8")
    # Row-major means the first `cols` values are row 0 of the matrix — one value per
    # named input — and not the first `cols` samples of the first input.
    assert first_row.tolist() == [inputs[name][0] for name in BAR_M1_V1.required_inputs]
    assert np.frombuffer(raw, dtype="<f8")[cols] == inputs[BAR_M1_V1.required_inputs[0]][1]


def test_a_reference_narrowed_to_float32_would_lose_the_divergence_this_gate_looks_for(tmp_path):
    # Why the width is f64 and not the model bundle's f32 (ADR-0016 §6). Demonstrated on
    # the bundle's own values rather than argued: build the candidate a bit-exact gate
    # exists to catch — one differing in the last bit of every finite cell, which is what
    # two runtimes whose rolling reduction associates differently produce — and watch the
    # whole divergence vanish at the narrower width.
    bundle = read_feature_bundle(written(tmp_path))
    finite = bundle.features[np.isfinite(bundle.features)]
    nudged = (finite.view(np.uint64) + np.uint64(1)).view(np.float64)
    assert not np.array_equal(nudged.view(np.uint64), finite.view(np.uint64))
    assert np.array_equal(nudged.astype(np.float32), finite.astype(np.float32)), (
        "some cell survives the narrowing as a distinguishable float32"
    )
    # Every cell, not a majority of them: at f32 the gate would be measuring its own
    # serialization rather than the two runtimes.
    assert int(np.count_nonzero(nudged.astype(np.float32) == finite.astype(np.float32))) == (
        finite.size
    )


def test_nan_travels_as_its_own_bits_because_a_json_null_cannot(tmp_path):
    # Warmup is NaN by construction (ADR-0016 §1) and a JSON `null` cannot carry a bit
    # pattern, so the matrices cross as raw IEEE-754. It is also what makes a NaN cell
    # comparable at all: `nan == nan` is false, `bits == bits` is true.
    directory = written(tmp_path)
    features = np.frombuffer((directory / "features.f64").read_bytes(), dtype="<f8")
    patterns = np.unique(features.view(np.uint64)[np.isnan(features)])
    assert patterns.size == 1 and int(patterns[0]) == CANONICAL_NAN_BITS
    assert manifest_of(directory)["features"]["nan_cells"] == int(np.isnan(features).sum()) > 0


def test_warmup_is_the_specs_own_warmup_and_the_manifest_counts_it(tmp_path):
    # `BAR_M1_WARMUP_BARS` is the spec's claim about itself; the bundle has to agree with
    # it or the Rust side is being asked to reproduce a warmup that is not the documented
    # one. Derived from the constant on purpose: a count typed here would keep passing on
    # the day somebody widens a window and puts the strategy back to sleep for a quarter
    # of an hour.
    bundle = read_feature_bundle(written(tmp_path))
    usable = finite_rows(bundle.features)
    assert not usable[: BAR_M1_WARMUP_BARS - 1].any()
    assert usable[BAR_M1_WARMUP_BARS - 1 :].all()
    assert bundle.manifest["features"]["finite_rows"] == ROWS - (BAR_M1_WARMUP_BARS - 1)


def test_the_inputs_are_named_exactly_what_the_spec_requires_and_carry_nothing_else(tmp_path):
    # The Rust side rebuilds its inputs mapping from this list, positionally. Free-form
    # names would let the two sides pair arrays with the wrong ones, and every column
    # would then be wrong in a way that looks exactly like a transform bug.
    directory = written(tmp_path)
    manifest = manifest_of(directory)
    assert manifest["inputs"]["names"] == sorted(manifest["inputs"]["names"])
    names = tuple(manifest["inputs"]["names"])
    assert names == BAR_M1_V1.required_inputs == ("close", "high", "low")
    assert tuple(manifest["features"]["names"]) == BAR_M1_V1.columns
    # `bar_inputs` supplies five arrays and this spec reads three. The two the spec never
    # reads are not carried: the file's column count follows from `required_inputs` and
    # from nothing else.
    assert set(bars()) - set(manifest["inputs"]["names"]) == {"open", "volume"}
    assert manifest["inputs"]["cols"] == len(BAR_M1_V1.required_inputs)


# ── the libm signpost ─────────────────────────────────────────────────────────


def test_the_libm_signpost_names_the_columns_that_reach_log_including_through_a_binding(tmp_path):
    # Not a tolerance — nothing widens for these columns. `+ - * / sqrt` are correctly
    # rounded by IEEE-754 and agree everywhere; `log` is not required to be, and numpy's
    # agreement with libm is a measurement on this box. If this gate ever reddens on
    # another platform, the first question is whether it reddened only here.
    assert libm_columns(BAR_M1_V1) == ("ret_1", "mom_5", "vol_20")
    assert manifest_of(written(tmp_path))["libm_columns"] == list(libm_columns(BAR_M1_V1))
    # `mom_32` never spells `log` and *is* a log return over a longer horizon; `vol_32` is
    # a standard deviation of them. Neither is grep-able from the spec.
    assert libm_columns(PERP_CORE_V1) == ("ret_1", "mom_32", "vol_32")

    # And the graph is walked rather than the transforms listed: a z-score of a log
    # return calls no logarithm and is entirely a function of one, while the same z-score
    # over a raw price is not.
    probe = FeatureSpec(
        name="libm_probe",
        version=1,
        features=(
            FeatureDef("ret_1", "log_return", params={"period": 1}, inputs={"price": "close"}),
            FeatureDef("z_ret", "rolling_zscore", params={"window": 5}, inputs={"x": "ret_1"}),
            FeatureDef("z_close", "rolling_zscore", params={"window": 5}, inputs={"x": "close"}),
        ),
    )
    assert libm_columns(probe) == ("ret_1", "z_ret")


def test_every_registered_transform_that_reaches_log_is_named_in_the_libm_table():
    # The table is per-transform and hand-written, so it is the thing that goes stale.
    # Walking the registry's own call graph is what catches the transform added next
    # month that acquires a libm dependency nobody wrote down — at which point every
    # bundle's signpost would be pointing away from the column that actually reddened.
    sources = {name: inspect.getsource(feature_info(name).fn) for name in registered_features()}
    by_python_name = {feature_info(name).fn.__name__: name for name in sources}

    def reaches_log(name: str, seen: frozenset[str] = frozenset()) -> bool:
        source = sources[name]
        if re.search(r"\bnp\.log\(", source):
            return True
        seen = seen | {name}
        return any(
            callee not in seen
            and re.search(rf"\b{re.escape(python_name)}\(", source)
            and reaches_log(callee, seen)
            for python_name, callee in by_python_name.items()
        )

    assert {name for name in sources if reaches_log(name)} == set(LIBM_FEATURES)


def test_a_manifest_whose_libm_signpost_disagrees_with_the_spec_is_refused(tmp_path):
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m.update(libm_columns=[]))
    with pytest.raises(BundleError, match="reach log"):
        read_feature_bundle(directory)


# ── what a bundle refuses to be written from ──────────────────────────────────


def test_a_spec_with_zero_rows_is_refused_because_a_gate_over_nothing_proves_nothing(tmp_path):
    # `FeatureSpec.compute` is perfectly happy to return a (0, 6) matrix, and every hash,
    # count and shape in the manifest would agree with it. Two languages agree about the
    # empty matrix and disagree about everything else.
    empty = {name: np.empty(0, dtype=np.float64) for name in BAR_M1_V1.required_inputs}
    with pytest.raises(BundleError, match="proves nothing"):
        written(tmp_path, inputs=empty)
    assert not (tmp_path / "bundle" / "inputs.f64").exists()


def test_a_corpus_that_never_finishes_warming_up_is_refused_because_all_nan_passes_bit_exactly(
    tmp_path,
):
    # The decoration check. An all-NaN reference compares bit-exactly against an all-NaN
    # candidate, so a Rust runtime that computed nothing at all would pass this gate —
    # the feature-side twin of the model bundle's "every row decides the same way".
    # One bar short of the spec's own warmup is exactly the boundary.
    with pytest.raises(BundleError, match="never finishes warming up"):
        written(tmp_path, inputs=bars(BAR_M1_WARMUP_BARS - 1), name="short")
    # And the refusal happens after `inputs.f64` is on disk, which is the state the
    # manifest-last rule exists to make legible rather than to prevent.
    with pytest.raises(BundleError, match="interrupted write"):
        read_feature_bundle(tmp_path / "short")

    directory = written(tmp_path, inputs=bars(BAR_M1_WARMUP_BARS), name="exact")
    assert manifest_of(directory)["features"]["finite_rows"] == 1


def test_a_bundle_cannot_buy_itself_a_looser_criterion_at_write_time(tmp_path):
    # Unlike the model gate there is no tolerance arm to fall back to, so `criterion`
    # exists to be refused rather than to offer a choice. Refused before a byte is
    # written, so a rejected argument cannot leave a half-written directory behind.
    with pytest.raises(BundleError, match="no tolerance arm"):
        written(tmp_path, name="loose", criterion=Criterion("max_abs_diff", 1e-9))
    assert not (tmp_path / "loose").exists(), "a refused criterion left a directory behind"


def test_a_nan_pattern_the_two_languages_spell_differently_is_refused(tmp_path):
    # Every NaN this library emits is a masked `np.nan`. x86's own default for an invalid
    # operation is the same value with the sign bit set, so a Rust transform that divides
    # first and masks afterwards would redden a bit-exact gate on every guarded cell
    # while computing the right answer. Refusing here makes "spell NaN the way Python
    # does" a property of the fixture rather than a paragraph somebody has to have read.
    rogue = np.uint64(CANONICAL_NAN_BITS | (1 << 63)).view(np.float64)
    assert np.isnan(rogue) and rogue.view(np.uint64) != CANONICAL_NAN_BITS

    inputs = bars()
    inputs["close"] = inputs["close"].copy()
    inputs["close"][3] = rogue
    with pytest.raises(BundleError, match="canonical"):
        written(tmp_path, inputs=inputs, name="rogue")


def test_a_spec_from_another_build_of_the_library_is_refused_rather_than_written_unreadable(
    tmp_path,
):
    # `FeatureSpec.from_dict` is strict about `library_version`, so this bundle's only
    # possible verdict on being read back is "regenerate me". Writing a fixture nobody
    # can read is worse than refusing to write one.
    stale = dataclasses.replace(BAR_M1_V1, library_version=FEATURES_VERSION + 1)
    with pytest.raises(BundleError, match="changed meaning"):
        written(tmp_path, spec=stale, name="stale")


def test_writing_over_an_existing_bundle_needs_saying_so(tmp_path):
    directory = written(tmp_path)
    with pytest.raises(BundleError, match="overwrite"):
        written(tmp_path)
    assert write_feature_bundle(
        BAR_M1_V1, bars(), out_dir=directory, source=SOURCE, overwrite=True
    ) == directory


def test_a_bundle_that_cannot_say_what_market_data_it_froze_is_refused(tmp_path):
    # The first thing a red cross-language gate has to answer is what it ran on.
    with pytest.raises(BundleError, match="description"):
        write_feature_bundle(
            BAR_M1_V1, bars(), out_dir=tmp_path / "anon", source={"instrument": "BTC"}
        )
    assert not (tmp_path / "anon").exists()


# ── what a bundle refuses to be read as ───────────────────────────────────────


def test_a_spec_edited_after_its_fingerprint_was_taken_is_refused(tmp_path):
    # The recipe and its identity travel together, and the reference was computed from
    # the recipe. A spec.json edited afterwards describes features this library does not
    # compute — the training–serving skew the fingerprint exists to make impossible.
    directory = written(tmp_path)
    corrupt(
        directory / "spec.json",
        lambda raw: raw.replace(b'"window":20', b'"window":19'),
    )
    edit_manifest(directory, restamp(directory, "spec"))
    with pytest.raises(BundleError, match="fingerprint"):
        read_feature_bundle(directory)


def test_a_matrix_edited_after_its_hash_was_taken_is_refused(tmp_path):
    directory = written(tmp_path)
    corrupt(directory / "features.f64", lambda raw: bytes([raw[0] ^ 0x01]) + raw[1:])
    with pytest.raises(BundleError, match="content hash"):
        read_feature_bundle(directory)


def test_a_matrix_that_lost_a_row_is_refused_rather_than_read_as_a_shorter_corpus(tmp_path):
    # With the hash restamped the truncation is invisible to every integrity check; only
    # the declared shape catches it, and a reader that did not check would gate on the
    # rows that happened to survive.
    directory = written(tmp_path)
    width = len(BAR_M1_V1.columns) * 8
    corrupt(directory / "features.f64", lambda raw: raw[:-width])
    edit_manifest(directory, restamp(directory, "features"))
    with pytest.raises(BundleError, match="f64 ="):
        read_feature_bundle(directory)


def test_a_reference_that_is_not_what_the_spec_computes_is_refused_though_every_hash_agrees(
    tmp_path,
):
    # One ULP on one finite cell: the length is unchanged, the NaN count is unchanged, the
    # finite-row count is unchanged, and the hash is restamped. Nothing but recomputing
    # the matrix from the bundle's own inputs can see it — and one ULP is exactly the size
    # of the disagreement this gate exists to find.
    directory = written(tmp_path)
    cell = ROWS * len(BAR_M1_V1.columns) - 1

    def nudge(raw: bytes) -> bytes:
        values = np.frombuffer(raw, dtype="<f8").copy()
        assert np.isfinite(values[cell]), "nudging a NaN would move the counts as well"
        bits = values.view(np.uint64)
        bits[cell] += np.uint64(1)
        return bits.view("<f8").tobytes()

    corrupt(directory / "features.f64", nudge)
    edit_manifest(directory, restamp(directory, "features"))
    with pytest.raises(BundleError, match="not the answer to the question"):
        read_feature_bundle(directory)


def test_a_manifest_claiming_the_wrong_nan_count_is_refused(tmp_path):
    # Warmup is NaN by construction and its extent is part of what the bundle asserts: a
    # count nobody checks is a count that drifts the first time a window moves.
    for section in ("inputs", "features"):
        directory = written(tmp_path, name=f"count-{section}")
        edit_manifest(
            directory, lambda m, s=section: m[s].update(nan_cells=m[s]["nan_cells"] + 1)
        )
        with pytest.raises(BundleError, match="NaN"):
            read_feature_bundle(directory)


def test_a_bundle_asking_for_a_criterion_other_than_bit_exact_is_refused_when_read(tmp_path):
    # The manifest is not hashed, so this edit leaves every file consistent and has to be
    # caught on meaning. A tolerance here would hide the library's one inexact operation
    # behind slack granted to every column that never touches it.
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m.update(criterion={"kind": "max_abs_diff", "eps": 1e-9}))
    with pytest.raises(BundleError, match="held to"):
        read_feature_bundle(directory)


def test_a_manifest_naming_a_file_outside_the_bundle_is_refused(tmp_path):
    # A manifest is data. A path in it that escapes the directory turns reading a bundle
    # into a file read of whoever wrote it choosing.
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m["inputs"].update(file="../inputs.f64"))
    with pytest.raises(BundleError, match="bare filename"):
        read_feature_bundle(directory)


def test_input_names_that_do_not_match_the_spec_are_refused_rather_than_paired_positionally(
    tmp_path,
):
    # The names are the whole contract: swap two and every column is computed from the
    # wrong array, with a plausible number in every cell.
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m["inputs"].update(names=["close", "low", "high"]))
    with pytest.raises(BundleError, match="positionally"):
        read_feature_bundle(directory)


def test_a_manifest_that_disagrees_with_its_own_spec_about_the_library_is_refused(tmp_path):
    # `library_version` is in the manifest so a reader can see which build's arithmetic
    # the reference depends on without parsing the recipe. A repeated field is a field
    # that can disagree, and this one would claim the transforms changed meaning while
    # spec.json says they did not.
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m.update(library_version=FEATURES_VERSION + 1))
    with pytest.raises(BundleError, match="library_version"):
        read_feature_bundle(directory)


def test_a_newer_bundle_schema_is_refused_rather_than_half_understood(tmp_path):
    directory = written(tmp_path)
    edit_manifest(directory, lambda m: m.update(bundle_schema=BUNDLE_SCHEMA + 1))
    with pytest.raises(BundleError, match="refusing"):
        read_feature_bundle(directory)


def test_a_bundle_without_a_manifest_reads_as_an_interrupted_write(tmp_path):
    # The manifest is written last, so its absence is diagnostic: "an interrupted write"
    # and "regenerate this directory" have to be the same sentence.
    directory = written(tmp_path)
    (directory / "manifest.json").unlink()
    with pytest.raises(BundleError, match="interrupted write"):
        read_feature_bundle(directory)
    with pytest.raises(BundleError, match="no such feature-parity bundle"):
        read_feature_bundle(tmp_path / "never-written")
