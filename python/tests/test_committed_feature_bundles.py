"""The committed feature-parity bundles, checked from the side that wrote them.

`crates/axon-features/tests/bundles/` holds the frozen cross-language questions
ADR-0035 defines, and the Rust gate is what answers them. This file is the other
end: it asserts the bundles are still *well-formed* and still describe what this
build of `axon.features` computes, from Python, with no Rust in the process.

Both ends are needed and they fail differently, which is the whole reason to have
two. If a bundle and the Rust runtime disagree, the Rust gate reddens and says the
two languages differ — but that verdict is only trustworthy if the bundle is known
to still match *Python*. Without this file, an `axon.features` change that moved a
transform would redden the Rust gate and read as "Rust is wrong", when what actually
happened is that Python moved and the frozen reference did not. The bundles are
committed fixtures, and a committed fixture nobody re-derives is a fixture that
quietly stops describing anything.

**A committed fixture is a cross-language object.** These files live under `crates/`
and are read by a Python test, so a regeneration reddens on both sides of the tree
at once — which is the point. `./run.sh feature-bundles` rewrites them, and the git
diff is the review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from axon.features.registry import registered_features
from axon.features.spec import BAR_M1_V1, PERP_CORE_V1
from axon.parity.feature_bundle import read_feature_bundle

BUNDLE_ROOT = Path(__file__).resolve().parents[2] / "crates/axon-features/tests/bundles"


def committed_bundles() -> list[Path]:
    if not BUNDLE_ROOT.is_dir():
        return []
    return sorted(p for p in BUNDLE_ROOT.iterdir() if (p / "manifest.json").is_file())


def test_the_committed_set_is_not_empty_so_this_sweep_cannot_pass_by_finding_nothing():
    # A parametrized sweep over an empty list is a green test that checked nothing —
    # the same invisible denominator ADR-0030 spent an increment on one level up. If
    # the bundles ever move, this fails rather than the sweep silently emptying.
    found = committed_bundles()
    assert found, (
        f"no committed feature-parity bundles under {BUNDLE_ROOT}; regenerate with "
        "./run.sh feature-bundles"
    )


@pytest.mark.parametrize("path", committed_bundles(), ids=lambda p: p.name)
def test_a_committed_bundle_still_describes_what_this_build_computes(path: Path):
    # `read_feature_bundle` re-fingerprints the spec, re-hashes every file, and
    # recomputes the matrix from the bundle's own inputs — so this single call is
    # the whole claim. What it buys over the Rust gate is attribution: if Python
    # moved, this reddens and the Rust gate's verdict can be read as "the fixture is
    # stale" rather than "the Rust runtime is broken".
    bundle = read_feature_bundle(path)
    assert bundle.rows > 0
    assert int(bundle.manifest["features"]["finite_rows"]) >= 1


@pytest.mark.parametrize("path", committed_bundles(), ids=lambda p: p.name)
def test_a_committed_bundles_warmup_is_the_one_its_own_spec_implies(path: Path):
    # Not a restatement of the manifest: `finite_rows` is recomputed here from the
    # matrix, and compared against the count the manifest recorded. A widened window
    # moves both together in a regeneration and moves neither in a stale fixture, so
    # the day they disagree is the day somebody edited one of them by hand.
    bundle = read_feature_bundle(path)
    finite = int(np.isfinite(bundle.features).all(axis=1).sum())
    assert finite == int(bundle.manifest["features"]["finite_rows"])
    assert int(np.isnan(bundle.features).sum()) == int(bundle.manifest["features"]["nan_cells"])


def test_the_committed_set_covers_every_registered_transform():
    # The gate is only as broad as its corpus. `BAR_M1_V1` exercises six of the
    # seventeen registered transforms, and a gate that ran on the shipped specs alone
    # would leave `spread`, `sma_crossover`, both EMAs and the microstructure columns
    # ungated in Rust forever — present, plausible, and never once compared against
    # Python. `all_transforms` exists to close that, and this asserts it did rather
    # than trusting the docstring that says so.
    covered: set[str] = set()
    for path in committed_bundles():
        covered |= {d.feature for d in read_feature_bundle(path).spec.features}
    missing = sorted(set(registered_features()) - covered)
    assert not missing, (
        f"no committed bundle exercises {missing}; the Rust implementation of each is "
        "ungated across the language boundary"
    )


def test_the_live_provenance_bundle_carries_the_pathology_only_a_venue_produces():
    # The 58 bars in `bar_m1_testnet_live` crossed a real socket, and the reason to
    # keep so small a corpus is that it holds something no offline history does:
    # `clv = (c-l)/(h-l)` is NaN whenever every trade in a minute happened at one
    # price. Measured at 6 of 58 BTC bars in that session against 0 of 4,999 hourly.
    # If a regeneration ever silences that column, this fixture has stopped being the
    # thing it was committed for and the test says so instead of passing quietly.
    path = BUNDLE_ROOT / "bar_m1_testnet_live"
    if not path.is_dir():
        pytest.skip("the live-provenance bundle is not committed in this tree")
    bundle = read_feature_bundle(path)
    clv = bundle.features[:, bundle.columns.index("clv")]
    flat_minutes = int(np.isnan(clv).sum())
    assert flat_minutes > 0, (
        "no NaN clv in the live tape; a single-price minute is the live pathology "
        "this bundle exists to carry across the boundary"
    )
    # And the offline history genuinely does not have it, which is what makes the
    # live bundle worth its 20 KB. Asserted rather than asserted-about.
    offline = read_feature_bundle(BUNDLE_ROOT / "bar_m1_btc")
    warm = int(BAR_M1_V1.features[3].params["window"]) + 1
    offline_clv = offline.features[warm:, offline.columns.index("clv")]
    assert not np.isnan(offline_clv).any(), (
        "the committed mainnet m1 history now has a flat minute too; the contrast "
        "this test draws is no longer true and the comment above needs correcting"
    )


def test_the_shipped_specs_are_the_ones_the_bundles_name():
    # A bundle carries a `spec_ref`, and a spec's identity is the whole mechanism
    # (ADR-0016 §2). If `BAR_M1_V1` is edited without regenerating, its fingerprint
    # moves and every bundle claiming to be it is describing a recipe that no longer
    # exists — which the reader catches, but this names the cause in one line.
    refs = {read_feature_bundle(p).spec_ref for p in committed_bundles()}
    assert BAR_M1_V1.ref in refs
    # `PERP_CORE_V1` needs a real order book and a real tape, which is why it went
    # without a bundle until one was recorded: `perp_core_live` is 675 market-data
    # slices off a live testnet session, through the Rust core and publisher onto the
    # `MdSlice` ring. `all_transforms` covers the same microstructure transforms over
    # columns *derived* from bars and says so in its own manifest; this one's book is
    # the venue's, which is a different and stronger claim about the same code.
    assert PERP_CORE_V1.ref in refs
