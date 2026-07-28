"""Tests for the portfolio-bound measurement (:mod:`axon.strategies.portfolio_evidence`).

Nothing here reads the cached corpus or fits anything: every test drives the arithmetic
over arrays it constructs, because what is being checked is the *definition* of gross, net
and breadth, and a test that needed a model would be one that skipped on a fresh checkout.

The end-to-end run over real bars is a command an operator types
(``python -m axon.strategies.portfolio_evidence``), deliberately outside the gate: it reads
``data/``, which is gitignored, and a test that silently skipped when the corpus was absent
would be indistinguishable from a passing one.
"""

from __future__ import annotations

import numpy as np
import pytest

from axon.strategies.loss_evidence import SpecTooSmall
from axon.strategies.portfolio_evidence import (
    book_window,
    describe,
    enumerate_books,
    parse_legs,
    position_series,
    summarize,
)


def test_a_position_is_forward_filled_between_signals():
    # What a target-position contract *means* (ADR-0006): a strategy that has not changed
    # its mind correctly says nothing, so the absence of a record is the previous target
    # restated. Reading a gap as flat would report a book that repeatedly went to zero and
    # back — and would make every gross quantile smaller than the book ever was.
    ts = np.arange(6, dtype=np.int64) * 60
    sigs = [
        {"ts_event": 120, "target_qty": 30_000},
        {"ts_event": 240, "target_qty": -10_000},
    ]
    got = position_series(ts, sigs)
    assert list(got) == pytest.approx([0.0, 0.0, 0.0003, 0.0003, -0.0001, -0.0001])


def test_a_signal_stamped_past_the_last_bar_is_dropped_rather_than_wrapped():
    # `searchsorted` returns `size` for a stamp past the end, and an unguarded index would
    # wrap to bar 0 — putting a decision at the start of the window that was made after it.
    ts = np.arange(3, dtype=np.int64) * 60
    got = position_series(ts, [{"ts_event": 10_000, "target_qty": 50_000}])
    assert list(got) == [0.0, 0.0, 0.0]


def test_no_signals_is_a_flat_book_rather_than_an_error():
    # A strategy still in its warmup emits nothing, and every leg subset containing it must
    # still be measurable — otherwise the sample loses exactly the windows where one leg
    # was quiet, which is a selection nobody chose.
    assert list(position_series(np.arange(4, dtype=np.int64), [])) == [0.0] * 4


def test_gross_counts_both_sides_and_net_cancels_them():
    # The distinction the two bounds exist for, and the reason a book of strategies that
    # disagree is worth running: same gross, very different net.
    opposed = book_window(
        positions=[np.array([1.0, 1.0]), np.array([-1.0, -1.0])],
        prices=[np.array([100.0, 100.0]), np.array([40.0, 40.0])],
        gross_caps=(),
    )
    assert opposed["gross_max"] == pytest.approx(140.0)
    assert opposed["net_max"] == pytest.approx(60.0)
    assert opposed["net_over_gross"] == pytest.approx(60.0 / 140.0)

    aligned = book_window(
        positions=[np.array([1.0, 1.0]), np.array([1.0, 1.0])],
        prices=[np.array([100.0, 100.0]), np.array([40.0, 40.0])],
        gross_caps=(),
    )
    assert aligned["gross_max"] == pytest.approx(140.0), "gross cannot tell them apart"
    assert aligned["net_max"] == pytest.approx(140.0), "net can"
    assert aligned["net_over_gross"] == pytest.approx(1.0)


def test_a_book_that_never_opened_reports_no_ratio_rather_than_a_perfect_one():
    # "never traded" and "never diversified" are different facts, and `1.0` would be the
    # more reassuring reading of the first — a book of warming strategies would look like
    # one whose legs never offset.
    flat = book_window(
        positions=[np.zeros(3), np.zeros(3)],
        prices=[np.full(3, 100.0), np.full(3, 40.0)],
        gross_caps=(),
    )
    assert flat["gross_max"] == 0.0
    assert flat["net_over_gross"] is None
    assert flat["open_max"] == 0


def test_breadth_counts_instruments_carrying_exposure_and_not_legs_declared():
    # Twenty $5 positions and one $100 position are the same gross and are not the same
    # operational problem — each instrument is a feed that can go stale and a position
    # somebody has to close. A leg sitting flat is not one of them.
    w = book_window(
        positions=[np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.zeros(2)],
        prices=[np.full(2, 100.0), np.full(2, 40.0), np.full(2, 10.0)],
        gross_caps=(),
    )
    assert w["open_max"] == 2
    assert w["legs"] == 3


def test_a_candidate_cap_reports_the_scale_the_runtime_would_have_applied():
    # The same factor `axon_risk::gross_scale` computes — `cap / gross` when the book is
    # over and nothing when it is under. `bound_frac` is therefore the fraction of bars a
    # session under that cap would have been working toward less than its strategies
    # asked for, which is the number that says whether a bound is a limit or a position
    # size nobody chose.
    w = book_window(
        positions=[np.array([1.0, 2.0])],
        prices=[np.array([100.0, 100.0])],
        gross_caps=(150.0, 500.0),
    )
    assert w["caps"]["150"]["bound_frac"] == pytest.approx(0.5)
    assert w["caps"]["150"]["scale_min"] == pytest.approx(150.0 / 200.0)
    assert w["caps"]["500"]["bound_frac"] == 0.0
    assert w["caps"]["500"]["scale_min"] == 1.0


def test_positions_and_prices_of_different_shapes_are_refused():
    with pytest.raises(ValueError):
        book_window(positions=[np.zeros(3)], prices=[np.zeros(4)], gross_caps=())
    with pytest.raises(ValueError):
        book_window(positions=[], prices=[], gross_caps=())


def test_legs_are_parsed_with_the_size_and_a_factory_that_contains_a_colon():
    # The factory is spelled `module:callable`, so `@` is the separator and `:` must keep
    # its one meaning — otherwise a perfectly valid factory name becomes an unparseable leg.
    assert parse_legs("axon.strategies.zoo:live_strategy@BTC=0.0003") == [
        ("axon.strategies.zoo:live_strategy", "BTC", "0.0003")
    ]
    assert parse_legs("baseline@eth") == [("baseline", "ETH", None)]
    assert parse_legs("a@BTC, b@ETH ") == [("a", "BTC", None), ("b", "ETH", None)]
    with pytest.raises(ValueError):
        parse_legs("baseline")


def test_books_are_subsets_and_not_a_product_of_strategies_and_coins():
    # A product enumerates *assignments* of one strategy to each coin, which is one shape
    # of book. What a portfolio bound has to be argued against is every book an operator
    # might configure — including two strategies on one coin, which is exactly the case
    # `TargetBook` nets and the one a product cannot express.
    books = enumerate_books(("a", "b"), ("BTC", "ETH"), min_legs=2, max_legs=2)
    assert "a@BTC,b@BTC" in books, "two strategies on one coin is a book"
    assert "a@BTC,a@ETH" in books
    # C(4, 2) = 6 subsets of the four legs.
    assert len(books) == 6

    # And the sizes ride on the leg, per coin.
    sized = enumerate_books(("a",), ("BTC", "ETH"), min_legs=2, sizes={"BTC": "0.0003"})
    assert sized == ["a@BTC=0.0003,a@ETH"]


def test_a_one_leg_book_is_refused_because_it_has_no_portfolio_question_in_it():
    # Its gross is its net, its breadth is one, and `[strategy.risk]`'s per-symbol caps
    # already bound it. Measuring it would add tasks and no information.
    with pytest.raises(SpecTooSmall):
        enumerate_books(("a",), ("BTC",), min_legs=2, max_legs=4)
    with pytest.raises(SpecTooSmall):
        enumerate_books(("a", "b"), ("BTC",), min_legs=3, max_legs=2)


def test_a_summary_reports_its_denominator_beside_every_quantile():
    # ADR-0030's finding, applied here: a comparison that silently dropped what it could
    # not match reported its most reassuring number exactly when the feed was most broken.
    # A summary over three surviving shards of fourteen has to say three.
    good = {
        "measurable": True,
        "windows": [
            {
                "gross_max": 100.0,
                "net_max": 60.0,
                "open_max": 2,
                "net_over_gross": 0.6,
                "caps": {"50": {"bound_frac": 1.0, "scale_min": 0.5}},
            }
        ],
    }
    bad = {"measurable": False, "legs": "a@BTC,b@ETH", "reason": "boom"}
    s = summarize([good, bad, good])
    assert s["books"] == 2
    assert s["windows"] == 2
    assert s["unmeasurable"] == 1
    assert s["gross_max"]["max"] == pytest.approx(100.0)
    assert s["net_over_gross"]["n"] == 2
    assert s["caps"]["50"]["windows_bound"] == pytest.approx(1.0)
    assert "2 book(s), 2 window(s), 1 unmeasurable" in describe(s)


def test_an_empty_sample_says_so_rather_than_summarizing_nothing():
    # A quantile over an empty array is not a bound, and `describe` printing a table of
    # zeros would be the most confident-looking possible report of having measured nothing.
    s = summarize([{"measurable": False, "reason": "no corpus"}])
    assert s["windows"] == 0
    assert "not a bound" in s["reason"]
    assert "no measurable windows" in describe(s)
