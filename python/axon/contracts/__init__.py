"""The Python half of Axon's boundary contract.

Everything here is derived at import time from ``contracts/schema.toml`` — the
*same* single source of truth the Rust ``axon-contracts`` crate is generated
from. Because both sides read one file, they cannot silently drift; a
cross-language round-trip test is the final backstop.

Exposes:
- :data:`SIGNAL_DTYPE` — a NumPy structured dtype byte-identical to the Rust
  ``#[repr(C)] Signal`` (little-endian, 64-byte stride, no padding).
- :data:`MD_SLICE_DTYPE` — the same for ``MdSlice``, the Rust→Python market-data
  record (128-byte stride; ADR-0012, extended with a mark/funding tail by ADR-0028).
- :data:`MD_BAR_DTYPE` — the same for ``MdBar``, the closed-OHLCV record on the
  second Rust→Python ring (ADR-0028). Note it is **also** 128 bytes: the two are
  told apart by the ring's ``record_kind``, never by their stride.
- Layout/flag/kind constants and the ring-header offsets.
- :func:`new_signal` / :func:`new_md_slice` / :func:`new_md_bar` — build one record.
- :data:`RECORDS` / :func:`record_spec_for` — the dtype→(kind, schema version)
  registry the ring clients use to stamp and validate a ring's control block.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Locate and load the single source of truth ───────────────────────────────
# python/axon/contracts/__init__.py → parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = Path(os.environ.get("AXON_SCHEMA_PATH", _REPO_ROOT / "contracts" / "schema.toml"))

if not _SCHEMA_PATH.is_file():
    raise FileNotFoundError(
        f"Axon contract schema not found at {_SCHEMA_PATH}. Set AXON_SCHEMA_PATH "
        "to point at contracts/schema.toml."
    )

with open(_SCHEMA_PATH, "rb") as _f:
    _SCHEMA = tomllib.load(_f)

SCHEMA_PATH = str(_SCHEMA_PATH)

# ── Top-level + fixed point ──────────────────────────────────────────────────
SCHEMA_VERSION: int = int(_SCHEMA["schema_version"])
ENDIANNESS: str = str(_SCHEMA["endianness"])
FIXED_POINT_DECIMALS: int = int(_SCHEMA["fixed_point"]["decimals"])
FIXED_POINT_SCALE: int = int(_SCHEMA["fixed_point"]["scale"])

if ENDIANNESS != "little":
    raise ValueError(f"axon.contracts only supports little-endian, schema says {ENDIANNESS!r}")

# ── Signal record ────────────────────────────────────────────────────────────
_SIGNAL = _SCHEMA["signal"]
SIGNAL_SIZE: int = int(_SIGNAL["size"])
SIGNAL_ALIGN: int = int(_SIGNAL["align"])
KIND_TARGET_POSITION: int = int(_SIGNAL["kinds"]["target_position"])
FLAG_REDUCE_ONLY: int = int(_SIGNAL["flags"]["reduce_only"])
FLAG_CLOSE: int = int(_SIGNAL["flags"]["close"])

# Map contract scalar/array types → little-endian NumPy formats.
_NP_SCALAR = {
    "u64": "<u8",
    "i64": "<i8",
    "u32": "<u4",
    "u16": "<u2",
    "u8": "u1",
}


def _np_format(type_str: str):
    """Translate a schema type (e.g. ``u64`` or ``u8[15]``) to a NumPy format."""
    if type_str in _NP_SCALAR:
        return _NP_SCALAR[type_str]
    if type_str.endswith("]") and "[" in type_str:
        elem, count = type_str[:-1].split("[")
        return (_NP_SCALAR[elem], int(count))
    raise ValueError(f"unsupported contract type {type_str!r}")


def _dtype_for(section: dict) -> np.dtype:
    """Build the structured dtype for one record section of the schema.

    Offsets come from the file rather than from NumPy's own packing rules: NumPy
    would otherwise be free to lay the fields out its way, and "both sides agree"
    would quietly mean "both sides agree with themselves".
    """
    fields = section["fields"]
    itemsize = int(section["size"])
    dtype = np.dtype(
        {
            "names": [f["name"] for f in fields],
            "formats": [_np_format(f["type"]) for f in fields],
            "offsets": [int(f["offset"]) for f in fields],
            "itemsize": itemsize,
        }
    )
    assert dtype.itemsize == itemsize, "dtype itemsize must equal the contract stride"
    return dtype


#: A NumPy structured dtype identical to the Rust ``Signal`` on the wire.
SIGNAL_DTYPE = _dtype_for(_SIGNAL)

# ── MdSlice record (market data, Rust→Python; ADR-0012) ──────────────────────
_MD_SLICE = _SCHEMA["md_slice"]
MD_SLICE_SIZE: int = int(_MD_SLICE["size"])
MD_SLICE_ALIGN: int = int(_MD_SLICE["align"])
#: MdSlice's layout version — independent of :data:`SCHEMA_VERSION`, so bumping one
#: record's layout never changes the version byte the other stamps on the wire.
MD_SLICE_SCHEMA_VERSION: int = int(_MD_SLICE["schema_version"])
MD_KIND_QUOTE: int = int(_MD_SLICE["kinds"]["quote"])
MD_KIND_TRADE: int = int(_MD_SLICE["kinds"]["trade"])
MD_KIND_SNAPSHOT: int = int(_MD_SLICE["kinds"]["snapshot"])
MD_FLAG_LAST_TRADE_SELL: int = int(_MD_SLICE["flags"]["last_trade_sell"])

#: A NumPy structured dtype identical to the Rust ``MdSlice`` on the wire.
MD_SLICE_DTYPE = _dtype_for(_MD_SLICE)

# ── MdBar record (closed OHLCV bars, Rust→Python; ADR-0028) ──────────────────
_MD_BAR = _SCHEMA["md_bar"]
MD_BAR_SIZE: int = int(_MD_BAR["size"])
MD_BAR_ALIGN: int = int(_MD_BAR["align"])
#: MdBar's own layout version — independent of every other record's.
MD_BAR_SCHEMA_VERSION: int = int(_MD_BAR["schema_version"])
MD_BAR_KIND_CLOSED: int = int(_MD_BAR["kinds"]["closed"])
#: The venue should have printed a bar between this one and the previous one for the
#: same instrument and interval, and did not. A rolling feature computed across it is
#: wrong; a ``seq`` gap would mean something else entirely (the ring dropped).
MD_BAR_FLAG_GAP_BEFORE: int = int(_MD_BAR["flags"]["gap_before"])
#: No previous bar for this instrument and interval in this session, so continuity is
#: *unknown* rather than broken.
MD_BAR_FLAG_FIRST_BAR: int = int(_MD_BAR["flags"]["first_bar"])

#: A NumPy structured dtype identical to the Rust ``MdBar`` on the wire.
MD_BAR_DTYPE = _dtype_for(_MD_BAR)

# ── Ring header layout ───────────────────────────────────────────────────────
_RING = _SCHEMA["ring"]
RING_MAGIC: int = int(_RING["magic"])
RING_VERSION: int = int(_RING["version"])
RING_HEADER_SIZE: int = int(_RING["header_size"])
RING_CACHE_LINE: int = int(_RING["cache_line"])

RING_KIND_SIGNAL: int = int(_RING["record_kinds"]["signal"])
RING_KIND_MD_SLICE: int = int(_RING["record_kinds"]["md_slice"])
RING_KIND_MD_BAR: int = int(_RING["record_kinds"]["md_bar"])

_ring_off = {f["name"]: int(f["offset"]) for f in _RING["header_fields"]}
RING_OFF_MAGIC: int = _ring_off["magic"]
RING_OFF_RING_VERSION: int = _ring_off["ring_version"]
RING_OFF_RECORD_SIZE: int = _ring_off["record_size"]
RING_OFF_CAPACITY: int = _ring_off["capacity"]
RING_OFF_RECORD_SCHEMA_VERSION: int = _ring_off["record_schema_version"]
RING_OFF_RECORD_KIND: int = _ring_off["record_kind"]
RING_OFF_HEAD: int = _ring_off["head"]
RING_OFF_TAIL: int = _ring_off["tail"]


class RecordSpec(NamedTuple):
    """Everything a ring endpoint must stamp or check for one record type.

    The Python mirror of the Rust ``Record`` trait. Ring clients take a *dtype* and
    look the rest up here, so a ring's control block cannot end up describing a
    different record than the one its dtype decodes — the Python equivalent of
    Rust's ``RingConsumer<R>`` being one type parameter.
    """

    name: str
    dtype: np.dtype
    size: int
    kind: int
    schema_version: int


SIGNAL_RECORD = RecordSpec("Signal", SIGNAL_DTYPE, SIGNAL_SIZE, RING_KIND_SIGNAL, SCHEMA_VERSION)
MD_SLICE_RECORD = RecordSpec(
    "MdSlice", MD_SLICE_DTYPE, MD_SLICE_SIZE, RING_KIND_MD_SLICE, MD_SLICE_SCHEMA_VERSION
)
MD_BAR_RECORD = RecordSpec(
    "MdBar", MD_BAR_DTYPE, MD_BAR_SIZE, RING_KIND_MD_BAR, MD_BAR_SCHEMA_VERSION
)

#: Every record type a ring may carry, keyed by its dtype.
#:
#: ``MdSlice`` and ``MdBar`` have the *same* 128-byte stride, so this mapping — and
#: the ``record_kind`` it feeds into the ring's control block — is the only thing
#: standing between a reader and decoding a bar's ``open`` as a bid price.
RECORDS: dict[np.dtype, RecordSpec] = {
    SIGNAL_DTYPE: SIGNAL_RECORD,
    MD_SLICE_DTYPE: MD_SLICE_RECORD,
    MD_BAR_DTYPE: MD_BAR_RECORD,
}


def record_spec_for(dtype: np.dtype) -> RecordSpec:
    """The :class:`RecordSpec` for ``dtype``, or ``ValueError`` if it is not one
    of the contract's records."""
    try:
        return RECORDS[dtype]
    except KeyError:
        raise ValueError(
            f"{dtype} is not a contract record dtype; a ring must carry one of "
            f"{[spec.name for spec in RECORDS.values()]}"
        ) from None


def new_signal(
    *,
    seq: int = 0,
    ts_event: int = 0,
    symbol_id: int = 0,
    target_qty: int = 0,
    price_band: int = 0,
    urgency: int = 0,
    ttl_ms: int = 0,
    model_version: int = 0,
    flags: int = 0,
    kind: int = KIND_TARGET_POSITION,
    max_order_age_ms: int = 0,
    ts_cause: int = 0,
) -> np.ndarray:
    """Build a target-position signal as a self-owning 0-d structured array.

    ``schema_version`` and ``kind`` are stamped automatically; ``pad0`` stays zero —
    the reader *refuses* a record whose padding is not, because that is a producer
    writing a field it does not know about. The result can be pushed onto a ring or
    compared field-by-field.

    ``max_order_age_ms`` is **not** ``ttl_ms`` (ADR-0031). ``ttl_ms`` is a signal
    *admission* window the reader consumes before the planner ever sees the record,
    clamped against the operator's ``max_signal_age_ms``; it has never applied to an
    order already resting at the venue. ``0`` defers to the operator's ceiling on
    both, so an unset field never means "never expires".

    ``ts_cause`` is the event time of the **observation this decision answers** — an
    m1 bar's own close, not its arrival and not the moment the strategy got round to
    it. ``0`` means the producer stated none, and the runtime then does not measure
    the stage at all: not every strategy is driven by a timed observation, and
    inventing a cause for a tick-driven one would report a latency describing nothing.

    It exists because the largest latency in this system was invisible from the Rust
    side. A closed m1 bar reached the strategy 951 / 12 051 / **111 475** ms after its
    own close on 2026-07-27, and the record carried only ``ts_event`` — the moment the
    producer *decided* — so a decision one second after a bar and one two minutes
    after it were the same record. Schema version 3 put the cause on the wire.
    """
    rec = np.zeros((), dtype=SIGNAL_DTYPE)
    rec["seq"] = seq
    rec["ts_event"] = ts_event
    rec["target_qty"] = target_qty
    rec["price_band"] = price_band
    rec["symbol_id"] = symbol_id
    rec["ttl_ms"] = ttl_ms
    rec["model_version"] = model_version
    rec["flags"] = flags
    rec["schema_version"] = SCHEMA_VERSION
    rec["urgency"] = urgency
    rec["kind"] = kind
    rec["max_order_age_ms"] = max_order_age_ms
    rec["ts_cause"] = ts_cause
    return rec


def new_md_slice(
    *,
    seq: int = 0,
    ts_event: int = 0,
    symbol_id: int = 0,
    bid_px: int = 0,
    bid_sz: int = 0,
    ask_px: int = 0,
    ask_sz: int = 0,
    last_trade_px: int = 0,
    last_trade_sz: int = 0,
    last_trade_ts: int = 0,
    last_trade_sell: bool = False,
    kind: int = MD_KIND_QUOTE,
    mark_px: int = 0,
    index_px: int = 0,
    funding_rate: int = 0,
    funding_interval_ns: int = 0,
    mark_ts_venue: int = 0,
    mark_ts_ingest: int = 0,
) -> np.ndarray:
    """Build a market-data slice as a self-owning 0-d structured array.

    ``schema_version`` is stamped automatically. Prices and sizes are fixed-point
    integers (see :func:`to_fixed`) — passing floats here would silently truncate on
    assignment into the integer fields.

    The mark/funding tail defaults to all zeros, which is the record's own "no
    ticker seen yet" sentinel. A consumer reads that from ``mark_ts_ingest == 0``,
    never from ``mark_px == 0``: a venue is free to mark an instrument at zero.
    """
    rec = np.zeros((), dtype=MD_SLICE_DTYPE)
    rec["seq"] = seq
    rec["ts_event"] = ts_event
    rec["bid_px"] = bid_px
    rec["bid_sz"] = bid_sz
    rec["ask_px"] = ask_px
    rec["ask_sz"] = ask_sz
    rec["last_trade_px"] = last_trade_px
    rec["last_trade_sz"] = last_trade_sz
    rec["last_trade_ts"] = last_trade_ts
    rec["symbol_id"] = symbol_id
    rec["flags"] = MD_FLAG_LAST_TRADE_SELL if last_trade_sell else 0
    rec["schema_version"] = MD_SLICE_SCHEMA_VERSION
    rec["kind"] = kind
    rec["mark_px"] = mark_px
    rec["index_px"] = index_px
    rec["funding_rate"] = funding_rate
    rec["funding_interval_ns"] = funding_interval_ns
    rec["mark_ts_venue"] = mark_ts_venue
    rec["mark_ts_ingest"] = mark_ts_ingest
    return rec


def new_md_bar(
    *,
    seq: int = 0,
    ts_event: int = 0,
    open_time: int = 0,
    symbol_id: int = 0,
    interval_ms: int = 0,
    open: int = 0,
    high: int = 0,
    low: int = 0,
    close: int = 0,
    volume: int = 0,
    flags: int = 0,
    kind: int = MD_BAR_KIND_CLOSED,
) -> np.ndarray:
    """Build a closed OHLCV bar as a self-owning 0-d structured array.

    ``ts_event`` is the bar's **close** — the venue's ``T + 1 ms`` in nanoseconds,
    never its open time. A bar stamped with its open is the textbook lookahead leak
    (:class:`axon.strategy.events.Bar` says so too), and the two arguments are
    adjacent here on purpose so a transposition is visible at the call site.
    """
    rec = np.zeros((), dtype=MD_BAR_DTYPE)
    rec["seq"] = seq
    rec["ts_event"] = ts_event
    rec["open_time"] = open_time
    rec["open"] = open
    rec["high"] = high
    rec["low"] = low
    rec["close"] = close
    rec["volume"] = volume
    rec["symbol_id"] = symbol_id
    rec["interval_ms"] = interval_ms
    rec["flags"] = flags
    rec["schema_version"] = MD_BAR_SCHEMA_VERSION
    rec["kind"] = kind
    return rec


def to_fixed(real: float) -> int:
    """Convert a real value to the fixed-point wire integer (tooling convenience)."""
    return int(round(real * FIXED_POINT_SCALE))


def from_fixed(fixed: int) -> float:
    """Inverse of :func:`to_fixed` (lossy; for inspection only)."""
    return fixed / FIXED_POINT_SCALE


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "ENDIANNESS",
    "FIXED_POINT_DECIMALS",
    "FIXED_POINT_SCALE",
    "SIGNAL_SIZE",
    "SIGNAL_ALIGN",
    "SIGNAL_DTYPE",
    "KIND_TARGET_POSITION",
    "FLAG_REDUCE_ONLY",
    "FLAG_CLOSE",
    "MD_SLICE_SIZE",
    "MD_SLICE_ALIGN",
    "MD_SLICE_DTYPE",
    "MD_SLICE_SCHEMA_VERSION",
    "MD_KIND_QUOTE",
    "MD_KIND_TRADE",
    "MD_KIND_SNAPSHOT",
    "MD_FLAG_LAST_TRADE_SELL",
    "MD_BAR_SIZE",
    "MD_BAR_ALIGN",
    "MD_BAR_DTYPE",
    "MD_BAR_SCHEMA_VERSION",
    "MD_BAR_KIND_CLOSED",
    "MD_BAR_FLAG_GAP_BEFORE",
    "MD_BAR_FLAG_FIRST_BAR",
    "RING_MAGIC",
    "RING_VERSION",
    "RING_HEADER_SIZE",
    "RING_CACHE_LINE",
    "RING_KIND_SIGNAL",
    "RING_KIND_MD_SLICE",
    "RING_KIND_MD_BAR",
    "RING_OFF_MAGIC",
    "RING_OFF_RING_VERSION",
    "RING_OFF_RECORD_SIZE",
    "RING_OFF_CAPACITY",
    "RING_OFF_RECORD_SCHEMA_VERSION",
    "RING_OFF_RECORD_KIND",
    "RING_OFF_HEAD",
    "RING_OFF_TAIL",
    "RecordSpec",
    "RECORDS",
    "SIGNAL_RECORD",
    "MD_SLICE_RECORD",
    "MD_BAR_RECORD",
    "record_spec_for",
    "new_signal",
    "new_md_slice",
    "new_md_bar",
    "to_fixed",
    "from_fixed",
]
