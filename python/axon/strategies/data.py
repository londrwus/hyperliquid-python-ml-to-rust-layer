"""Real Hyperliquid candles: fetched once, cached, and never synthetic.

A strategy that climbs the validation ladder on generated data proves that the
ladder runs, not that the strategy works. So the only market data this package
accepts is what the venue actually printed, pulled from the public read-only
``POST /info {"type": "candleSnapshot"}`` endpoint — no key, no order, no spend.

Four rules hold here, and each one is a lookahead leak or a skew bug that this
module refuses to let downstream:

**The default gate never reaches the network.** :func:`fetch_candles` refuses
unless ``AXON_ALLOW_NETWORK=1`` is set. Tests read :func:`fixture_candles`, a
small verbatim slice committed next to this file, so the suite is offline and
deterministic; the full history lives in a gitignored cache directory.

**The unclosed bar is dropped.** The venue happily returns the bar that is still
forming. Its ``c`` is a mid-bar price stamped with a close time in the future, and
training on it is the purest form of lookahead available: the label of the last
row would be computed from a close that had not happened. :func:`closed_rows`
drops it, and it is the one place a wall clock is legitimate — deciding which
*recording* is complete is not the same as ordering events by receipt time.

**Event time is the bar's close.** Hyperliquid's ``t`` is the open, ``T`` the last
millisecond of the bar; a bar stamped with its open time is the textbook leak
:class:`axon.strategy.events.Bar` already warns about, so ``ts_event`` here is
``T + 1 ms`` in nanoseconds — the instant the bar was final. See
:data:`CLOSE_STAMP_OFFSET_MS` for why the extra millisecond is load-bearing and
for the one place in this repo that does not yet carry it.

**Prices go through fixed-point, exactly as the wire carries them.** The venue
sends decimal strings; a ``float(s)`` here would round differently from the
``i64`` a live ``Bar`` carries, and the research and serving paths would then
disagree in the last bits before a single feature had been computed. Every value
is parsed with :class:`~decimal.Decimal` into the contract's fixed-point integer,
and becomes a float exactly once, in :func:`axon.features.bar_inputs`.
"""

from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from axon.contracts import to_fixed
from axon.features import bar_inputs

#: Mainnet, deliberately, even though execution is on testnet. Testnet candles are
#: thin and largely self-dealt; a model fitted on them learns the testnet market
#: maker, and every number this package reports would describe a market that does
#: not exist. Reading mainnet ``/info`` needs no key and places nothing.
MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"
TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"

INFO_URL_ENV = "AXON_HL_INFO_URL"
#: The switch that keeps the default gate offline. Nothing here dials out without it.
ALLOW_NETWORK_ENV = "AXON_ALLOW_NETWORK"
CACHE_DIR_ENV = "AXON_CANDLE_CACHE"

#: The venue caps one ``candleSnapshot`` response; history is paged, not requested
#: in one call. Discovered by asking for more and counting what came back.
MAX_ROWS_PER_REQUEST = 5_000

#: Bar length in milliseconds, for the intervals this package uses. Not derived
#: from the string at runtime: an interval the venue does not serve should fail
#: here, with the list, rather than at the first empty response.
INTERVAL_MS: Mapping[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

#: Milliseconds between the venue's ``T`` and the stamp this module puts on the bar.
#:
#: ``T`` is the bar's *last* millisecond, not the instant after it: a bar stamped
#: ``T`` sorts equal to every trade printed inside its own final millisecond, and an
#: event-time sort may then hand a strategy the closed bar before the tick that
#: closed it. ``T + 1 ms`` is the instant the bar is final for ordering, which is
#: what ``axon_core::Candle::ts_event`` documents ("the point at which it is final
#: for ordering").
#:
#: **The Rust decoder agrees.** ``decode_candle``
#: (``axon-provider-hyperliquid/src/ws/decode.rs``) stamps ``(T + 1) * MS_TO_NS`` and
#: is tested for it. That agreement is the whole reason this constant is named rather
#: than a magic ``+ 1`` on each side: were the two halves one millisecond apart, an
#: ``align_by_event_time`` between an online bar feature and its offline recompute
#: would intersect to *nothing* — the stamps are 1e6 ns apart on a grid 3.6e12 ns
#: wide, so no pair ever collides — and the feature-parity gate would fail as "an
#: empty feature matrix proves nothing", a long way from the cause.
CLOSE_STAMP_OFFSET_MS = 1

_CSV_HEADER = ("T", "o", "h", "l", "c", "v")
_FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"


class DataError(RuntimeError):
    """Candle data that cannot be trusted to train on."""


@dataclass(frozen=True, eq=False)
class Candles:
    """A contiguous run of closed OHLCV bars for one coin, in fixed-point.

    ``eq=False`` because the fields are arrays: a generated ``__eq__`` would build
    an array of booleans and raise the moment anything compared two of these.
    """

    coin: str
    interval: str
    #: Bar **close** time in nanoseconds, strictly increasing.
    ts_event: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        n = self.ts_event.size
        for name in ("open", "high", "low", "close", "volume"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise DataError(f"{self.coin} {name} has {arr.shape}, expected ({n},)")
            if arr.dtype != np.int64:
                raise DataError(f"{self.coin} {name} is {arr.dtype}, expected fixed-point int64")
        if n and np.any(np.diff(self.ts_event) <= 0):
            # Duplicate or reversed bars mean the pages were stitched wrong. Features
            # are defined over the order the venue printed, and a resorted-in-place
            # series produces numbers no live path will ever reproduce.
            first = int(np.argmax(np.diff(self.ts_event) <= 0)) + 1
            raise DataError(
                f"{self.coin} {self.interval}: bar close times do not increase at row "
                f"{first} ({int(self.ts_event[first - 1])} → {int(self.ts_event[first])})"
            )

    def __len__(self) -> int:
        return int(self.ts_event.size)

    def __repr__(self) -> str:
        return f"Candles({self.coin} {self.interval}, {len(self)} bars)"

    @property
    def gaps(self) -> int:
        """Bars the venue never printed — a halted feed, not a quiet market.

        Reported rather than filled. Interpolating a missing bar invents a close
        that nothing traded at, and every return feature downstream would then be
        measuring our own arithmetic.
        """
        if len(self) < 2:
            return 0
        step = INTERVAL_MS[self.interval] * 1_000_000
        return int(np.count_nonzero(np.diff(self.ts_event) != step))

    def head(self, n: int) -> "Candles":
        return self[:n]

    def __getitem__(self, key: slice) -> "Candles":
        if not isinstance(key, slice):
            raise TypeError("Candles slices by range only; a single bar is not a Candles")
        return Candles(
            coin=self.coin,
            interval=self.interval,
            ts_event=self.ts_event[key],
            open=self.open[key],
            high=self.high[key],
            low=self.low[key],
            close=self.close[key],
            volume=self.volume[key],
        )

    def feature_inputs(self) -> dict[str, np.ndarray]:
        """The named float arrays a :class:`~axon.features.FeatureSpec` consumes.

        Straight through :func:`axon.features.bar_inputs` — the same call the
        serving strategy makes on its own buffer of live ``Bar`` events, which is
        what makes the feature-parity gate a comparison of transforms rather than
        of two decoders.
        """
        return bar_inputs(self.open, self.high, self.low, self.close, self.volume)

    # ── serialization ────────────────────────────────────────────────────────

    def to_csv(self, path: str | os.PathLike[str]) -> Path:
        """Write the venue's own field names and decimal form, not the integers.

        Each value is rendered as the decimal string that parses back to exactly
        this fixed-point integer, so the file is readable, diffable, and re-decoded
        by the same path a fresh download takes — a cache written in our internal
        representation would be a cache that no longer tests the decoder.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CSV_HEADER)
            for i in range(len(self)):
                writer.writerow(
                    [
                        int(self.ts_event[i] // 1_000_000) - CLOSE_STAMP_OFFSET_MS,
                        _unfix(self.open[i]),
                        _unfix(self.high[i]),
                        _unfix(self.low[i]),
                        _unfix(self.close[i]),
                        _unfix(self.volume[i]),
                    ]
                )
        return out

    @classmethod
    def from_csv(cls, path: str | os.PathLike[str], *, coin: str, interval: str) -> "Candles":
        with Path(path).open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if tuple(reader.fieldnames or ()) != _CSV_HEADER:
                raise DataError(
                    f"{path}: header is {reader.fieldnames}, expected {list(_CSV_HEADER)}"
                )
            rows = [dict(row) for row in reader]
        return cls.from_rows(rows, coin=coin, interval=interval)

    @classmethod
    def from_rows(
        cls, rows: Sequence[Mapping[str, Any]], *, coin: str, interval: str
    ) -> "Candles":
        """Build from raw ``candleSnapshot`` rows (or their CSV echo).

        Rows are de-duplicated on close time and sorted, because paged requests
        overlap at their seams by one bar and the venue is under no obligation to
        return them in order.
        """
        if interval not in INTERVAL_MS:
            raise DataError(f"unsupported interval {interval!r}; known: {sorted(INTERVAL_MS)}")
        by_close: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            try:
                close_ms = int(row["T"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError(f"candle row missing a usable close time: {row!r}") from exc
            by_close[close_ms] = row
        ordered = [by_close[k] for k in sorted(by_close)]

        ts = np.array(
            [(int(r["T"]) + CLOSE_STAMP_OFFSET_MS) * 1_000_000 for r in ordered],
            dtype=np.int64,
        )
        fields = {
            name: np.array([_fix(r[key], key, coin) for r in ordered], dtype=np.int64)
            for name, key in (
                ("open", "o"),
                ("high", "h"),
                ("low", "l"),
                ("close", "c"),
                ("volume", "v"),
            )
        }
        return cls(coin=coin, interval=interval, ts_event=ts, **fields)


def _fix(value: Any, key: str, coin: str) -> int:
    """Decimal string → the contract's fixed-point integer, exactly."""
    try:
        return to_fixed(Decimal(str(value)))
    except (InvalidOperation, ArithmeticError, TypeError) as exc:
        raise DataError(f"{coin}: candle field {key}={value!r} is not a number") from exc


def _unfix(fixed: np.int64) -> str:
    """Fixed-point integer → the shortest decimal string that reads back identically."""
    text = format(Decimal(int(fixed)) / Decimal(10**8), "f")
    return text


# ── the network edge ─────────────────────────────────────────────────────────


def network_allowed() -> bool:
    """Whether a live fetch is permitted. Off unless the operator says otherwise."""
    return os.environ.get(ALLOW_NETWORK_ENV, "") not in ("", "0", "false", "no")


def info_url() -> str:
    return os.environ.get(INFO_URL_ENV) or MAINNET_INFO_URL


def closed_rows(rows: Iterable[Mapping[str, Any]], *, now_ms: int) -> list[Mapping[str, Any]]:
    """Only the bars that had finished by ``now_ms``.

    The venue returns the bar currently forming. Its ``c`` is a mid-bar price
    carrying a close time in the future, so a label computed from it is a label
    read off a price that has not happened yet — and it is the *last* row, which is
    the one every walk-forward split puts in its most recent test window.
    """
    return [r for r in rows if int(r["T"]) < int(now_ms)]


def fetch_candles(
    coin: str,
    interval: str,
    *,
    start_ms: int,
    end_ms: int,
    url: str | None = None,
    timeout_s: float = 30.0,
    pause_s: float = 0.25,
    now_ms: int | None = None,
) -> list[Mapping[str, Any]]:
    """Page ``candleSnapshot`` over ``[start_ms, end_ms)`` and return closed bars.

    Read-only and unauthenticated: this endpoint cannot place, cancel or modify
    anything. It still refuses to run unless ``AXON_ALLOW_NETWORK`` is set, because
    "the default gate touches no network" is a property of the gate, not of the
    politeness of the request.

    Paging is forward from ``start_ms`` in windows of :data:`MAX_ROWS_PER_REQUEST`
    bars, advancing past the last close actually received rather than by the
    nominal window: a halted feed returns fewer rows than asked for, and advancing
    by the request width would skip the bars that resumed after the halt.
    """
    if not network_allowed():
        raise DataError(
            f"a live candle fetch needs {ALLOW_NETWORK_ENV}=1. The default test gate is "
            "offline by construction; use fixture_candles() or a populated cache"
        )
    if interval not in INTERVAL_MS:
        raise DataError(f"unsupported interval {interval!r}; known: {sorted(INTERVAL_MS)}")

    step_ms = INTERVAL_MS[interval]
    endpoint = url or info_url()
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    horizon = min(int(end_ms), now)

    collected: dict[int, Mapping[str, Any]] = {}
    cursor = int(start_ms)
    while cursor < horizon:
        window_end = min(cursor + step_ms * MAX_ROWS_PER_REQUEST, horizon)
        page = _post(
            endpoint,
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": window_end,
                },
            },
            timeout_s=timeout_s,
        )
        if not page:
            # An empty page inside the requested range is a listing gap, not the end
            # of history: step over one window and keep going rather than truncating
            # every later bar out of the dataset.
            cursor = window_end
            continue
        for row in page:
            collected[int(row["T"])] = row
        last_close = max(int(row["T"]) for row in page)
        cursor = max(last_close + 1, cursor + step_ms)
        if pause_s:
            time.sleep(pause_s)

    return closed_rows((collected[k] for k in sorted(collected)), now_ms=now)


def _post(url: str, payload: Mapping[str, Any], *, timeout_s: float) -> list[Mapping[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        body = json.load(response)
    if not isinstance(body, list):
        raise DataError(
            f"{url} returned {type(body).__name__}, expected a list of candles: {body}"
        )
    return body


# ── cache and fixture ────────────────────────────────────────────────────────


def cache_dir() -> Path:
    """Where downloaded history lives. Gitignored: market data is not source."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data" / "candles"


def cache_path(coin: str, interval: str, *, root: Path | None = None) -> Path:
    return (root or cache_dir()) / f"{coin.lower()}-{interval}.csv"


def load_candles(
    coin: str,
    interval: str = "1h",
    *,
    days: int = 365,
    root: Path | None = None,
    refresh: bool = False,
    now_ms: int | None = None,
) -> Candles:
    """The cached history for one coin, fetching it if the cache is cold.

    ``refresh=False`` never dials out when a cache file exists, so a research run
    is reproducible: the same file yields the same model, and re-running tomorrow
    does not silently extend the sample and move every number in the ADR.

    ``days`` is an upper bound, not a promise. The venue keeps roughly
    :data:`MAX_ROWS_PER_REQUEST` candles *per interval* and serves the most recent
    ones whatever start time is asked for — verified by requesting a window a year
    back and getting an empty list. That is why ``1h`` is the default here: at a
    fixed row cap, a longer bar buys strictly more calendar coverage for the same
    number of training rows, and coverage is what a walk-forward is short of.
    """
    path = cache_path(coin, interval, root=root)
    if path.is_file() and not refresh:
        return Candles.from_csv(path, coin=coin, interval=interval)

    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows = fetch_candles(
        coin,
        interval,
        start_ms=now - days * 86_400_000,
        end_ms=now,
        now_ms=now,
    )
    candles = Candles.from_rows(rows, coin=coin, interval=interval)
    candles.to_csv(path)
    return candles


def fixture_path(coin: str, interval: str = "1h") -> Path:
    return _FIXTURE_DIR / f"{coin.lower()}-{interval}.csv"


def fixture_coins(interval: str = "1h") -> tuple[str, ...]:
    """Which coins have a committed slice. Sorted, so a sweep over them is stable."""
    suffix = f"-{interval}.csv"
    return tuple(sorted(p.name[: -len(suffix)].upper() for p in _FIXTURE_DIR.glob(f"*{suffix}")))


def fixture_candles(coin: str = "BTC", interval: str = "1h") -> Candles:
    """A small, real, committed slice — what the offline tests train on.

    Verbatim rows from ``candleSnapshot``, not a resampling or a reconstruction, so
    a test that passes here is a test against numbers the venue actually printed.
    It is far too short to say anything about a strategy's edge; it exists so the
    pipeline that says such things can be run without a network.
    """
    path = fixture_path(coin, interval)
    if not path.is_file():
        raise DataError(
            f"no committed fixture for {coin} {interval} at {path}; "
            f"available: {list(fixture_coins(interval))}"
        )
    return Candles.from_csv(path, coin=coin, interval=interval)


def iter_fixtures(interval: str = "1h") -> Iterator[Candles]:
    for coin in fixture_coins(interval):
        yield fixture_candles(coin, interval)


__all__ = [
    "ALLOW_NETWORK_ENV",
    "CACHE_DIR_ENV",
    "CLOSE_STAMP_OFFSET_MS",
    "INFO_URL_ENV",
    "INTERVAL_MS",
    "MAINNET_INFO_URL",
    "MAX_ROWS_PER_REQUEST",
    "TESTNET_INFO_URL",
    "Candles",
    "DataError",
    "cache_dir",
    "cache_path",
    "closed_rows",
    "fetch_candles",
    "fixture_candles",
    "fixture_coins",
    "fixture_path",
    "info_url",
    "iter_fixtures",
    "load_candles",
    "network_allowed",
]
