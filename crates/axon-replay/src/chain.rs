//! The **chain trace**: what a replay of the whole production chain emitted, in the
//! shape a golden comparison diffs.
//!
//! Until this existed the trace carried market-data state only, and ADR-0018 said so
//! in its own consequences: *"a golden run says nothing about reconciliation or
//! strategy output"*. Reconciliation and strategy output are the two things every rung
//! above the bottom one compares, so a golden that omitted them was narrower than its
//! name. These are the columns that close that gap — the order tracker's reconciled
//! position and the planner's emitted orders, alongside the book and the marks.
//!
//! Two shapes, and the split between them is a type rather than a parameter:
//!
//! - [`ChainRow::values`] are **continuous** — prices, sizes, counts. A comparison may
//!   allow a tolerance on them, because a rounding unit moving after a refactor is a
//!   legitimate thing to accept.
//! - [`ChainRow::decisions`] are **discretized** — which order the tracker credited a
//!   fill to, what the planner decided to send. These compare exactly and no tolerance
//!   may soften them, because a flipped decision is a different order at a real venue
//!   (`docs/07-parity-and-testing.md`). Making it a field is what stops a caller
//!   softening it by passing a tolerance that happens to cover the value underneath.
//!
//! [`PlannedOrder`] carries the `cloid` as a **string**, and that is not cosmetic. A
//! `cloid` is 128 bits, JSON numbers are not, and the whole reason an order-level
//! golden is possible at all is that `cloid`s are derived from the signal rather than
//! minted from a counter (ADR-0014 §5) — a 128-bit id truncated through a float would
//! make two identical runs look identical while comparing something else.
//!
//! ## A reading nobody could take is not a reading of zero
//!
//! The order tracker sits behind a lock, and a panic anywhere else that holds it leaves
//! that lock poisoned; the fan-out survives (`CoreHandler` drops the execution events it
//! can no longer apply, and counts them) but every tracker column becomes unknowable. So
//! those columns are `Option` all the way out to the JSON and land as [`Cell::Absent`],
//! and [`ChainSummary::dropped_exec_events`] carries the count. Filling them with zeros
//! instead would write *"we hold nothing, nothing is resting, nothing went unattributed"*
//! into the golden — a degraded session rendered byte-identical to a flat one, agreed on
//! by both sides of the comparison, in the one harness whose job is to notice.
//!
//! ## What a chain trace still refuses to claim
//!
//! An order in [`ChainSummary::orders`] is what the strategy **asked for** at that
//! instant, against the state the log actually produced. It was never sent, never
//! acknowledged, and never filled: the replay does not write it into the tracker,
//! because doing so would invent an `OrderAck` the venue never gave. So the position
//! columns move only on the fills the *captured* session received. See
//! [`crate::replay`] and ADR-0018 §7 — a harness that blurred this would be reporting
//! a P&L nobody agreed to.

use std::collections::BTreeMap;

use axon_core::{Decimal, Nanos, Side, Tif};
use serde::Serialize;

/// The result contract the replay binary emits and `axon.backtest` reads.
pub const RESULT_SCHEMA: &str = "axon.backtest";

/// **Bump whenever the summary or trace shape changes.** `axon.backtest` compares a
/// fresh run against a stored reference; a silently reshaped trace would make that
/// comparison meaningless rather than failed.
///
/// - `1` — market-data state only, flat rows.
/// - `2` — the whole chain: tracker and planner columns, `values`/`decisions` split.
/// - `3` — a tracker that cannot be read reports [`Cell::Absent`] instead of zeros, and
///   [`ChainSummary::dropped_exec_events`] counts what that cost.
pub const RESULT_SCHEMA_VERSION: u32 = 3;

/// One cell of a trace row.
///
/// Three cases and no fourth, because the fourth is the bug: `Absent` is not a small
/// number. A missing mark makes the risk gate fail closed and a mark of zero sizes a
/// position against a price that does not exist, so the two must never collapse into
/// each other on the way through JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Cell {
    /// A quantity of money or size. Serialized as a decimal **string**, never a float
    /// — a price through a float64 is a price that no longer compares equal.
    Money(Decimal),
    /// A count or a timestamp. Compared exactly: a timestamp is not something a
    /// comparison is allowed to be nearly right about.
    Count(i64),
    /// The column does not apply to this row, or the state it reads has never been
    /// set. Serialized as `null`.
    Absent,
}

impl Cell {
    pub fn money(v: Option<Decimal>) -> Self {
        v.map_or(Cell::Absent, Cell::Money)
    }

    pub fn count(v: Option<i64>) -> Self {
        v.map_or(Cell::Absent, Cell::Count)
    }
}

impl From<Decimal> for Cell {
    fn from(v: Decimal) -> Self {
        Cell::Money(v)
    }
}

impl From<u64> for Cell {
    fn from(v: u64) -> Self {
        Cell::Count(v as i64)
    }
}

impl From<usize> for Cell {
    fn from(v: usize) -> Self {
        Cell::Count(v as i64)
    }
}

/// The chain's state as of one replayed event.
///
/// Every column is *read back* from production state, never computed here. The moment
/// a probe derives something itself it becomes a second implementation, and the parity
/// claim quietly stops being true.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChainRow {
    pub seq: u64,
    pub ts_event: Nanos,
    /// The clock as the handler saw it. Present so a run whose handlers reach for
    /// wall-clock time shows up as a diff in the golden file rather than as an
    /// intermittent failure much further up the ladder.
    pub clock_ns: Nanos,
    /// The **event's** instrument, and therefore the one every per-symbol column in
    /// [`values`](Self::values) describes. `None` for account-level execution events,
    /// which are not per-symbol.
    pub symbol_id: Option<u32>,
    pub kind: &'static str,
    /// A `BTreeMap`, not a `HashMap`: `HashMap` iteration order is randomized per
    /// process, and a harness that exists to detect nondeterminism must not be the
    /// thing manufacturing it.
    pub values: BTreeMap<&'static str, Cell>,
    /// What the chain *decided* at this event.
    ///
    /// A strategy pass runs after the event, whatever instrument that event belonged
    /// to, so a `plan` here may name a different symbol from
    /// [`symbol_id`](Self::symbol_id). That is not a mismatch to be tidied away: the
    /// row records *when* the decision was taken, and the pass saw a book that the
    /// other instrument's event had just advanced the clock past.
    /// [`ChainSummary::orders`] carries each order's own symbol.
    pub decisions: BTreeMap<&'static str, String>,
}

/// One order the planner emitted during the replay.
///
/// It reached no venue. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlannedOrder {
    /// The `seq` of the signal that produced it, so an order traces back to the
    /// record that caused it without a side channel.
    pub signal_seq: u64,
    /// Event time of the pass that planned it — the core's clock, not the signal's
    /// decision time.
    pub ts_event: Nanos,
    pub symbol_id: u32,
    /// Hex, `0x`-prefixed. 128 bits do not survive a JSON number.
    pub cloid: String,
    pub side: Side,
    pub qty: Decimal,
    pub price: Option<Decimal>,
    pub tif: Tif,
    pub reduce_only: bool,
}

/// One cancel the planner emitted during the replay.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlannedCancel {
    pub signal_seq: u64,
    pub ts_event: Nanos,
    pub symbol_id: u32,
    /// How the cancel addresses the order: `cloid:0x…` or `oid:…`.
    ///
    /// Which one is a *decision*, not a detail. An adopted order's `cloid` may be one
    /// the tracker synthesized from a venue id and the venue has never seen; a cancel
    /// sent under it fails silently and the stale quote stays resting (ADR-0020).
    pub target: String,
}

/// Final per-symbol state, the summary's own view of where the session ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SymbolState {
    pub events: u64,
    pub mid: Option<Decimal>,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub last_trade_px: Option<Decimal>,
    pub mark_px: Option<Decimal>,
    pub mark_ts: Option<Nanos>,
    /// Filled position — what the *captured* session's fills left behind.
    ///
    /// `None` when the tracker could not be read: our order state is unknown, and
    /// unknown is not flat. See the module docs.
    pub position_qty: Option<Decimal>,
    /// Position plus the worst case that every live order fills, which is the number
    /// the pre-trade gate checks against. `None` for the reason above.
    pub risk_qty: Option<Decimal>,
    pub open_orders: Option<usize>,
}

/// What the strategy adapter made of the session.
///
/// The counters are the operator's real question — *are we acting on the records the
/// producer thinks it sent?* — and they are in the golden because a refactor that
/// started silently rejecting every signal would otherwise show up only as an absence
/// of orders, which looks exactly like a quiet strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct SignalCounters {
    pub records: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub expired: u64,
    pub superseded: u64,
    /// Passes that produced at least one order.
    pub planned: u64,
    /// Passes that produced none because there was no usable top of book.
    pub no_quote: u64,
}

/// One replay pass, in the form two of which can be compared.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChainSummary {
    pub schema: &'static str,
    pub schema_version: u32,
    /// The event log's own provenance string. A *path* would differ between machines
    /// and turn a golden comparison into a comparison of checkouts.
    pub source: String,
    /// The signal log's provenance, or `None` when no strategy was attached.
    pub signal_source: Option<String>,
    pub order: &'static str,
    pub events: u64,
    pub first_ts: Option<Nanos>,
    pub last_ts: Option<Nanos>,
    pub late_arrivals: u64,
    /// Execution events the fan-out could not apply, because a panic elsewhere left the
    /// tracker's lock poisoned.
    ///
    /// In the golden, and not merely in a log line, because this is the number that
    /// tells a green comparison apart from a meaningless one: past the first drop the
    /// reconciled columns describe a tracker that stopped following the venue, and two
    /// runs that both stopped following it agree with each other perfectly.
    pub dropped_exec_events: u64,
    pub trace_rows: u64,
    /// How many intent passes ran. Part of the golden because the pass schedule is
    /// what decides *when* the strategy got to look, and two runs that looked at
    /// different moments are not two runs of one experiment.
    pub intent_passes: u64,
    pub signals: SignalCounters,
    pub orders: Vec<PlannedOrder>,
    pub cancels: Vec<PlannedCancel>,
    /// Keyed by symbol id, in a `BTreeMap` for the reason [`ChainRow::values`] is one.
    pub symbols: BTreeMap<u32, SymbolState>,
}

/// Render `rows` as the JSONL a golden comparison diffs.
///
/// One line per row, which is what makes `diff` answer "which event differs" in one
/// command — the property that justifies JSONL over a packed frame in the first place
/// (ADR-0018 §1). Also the byte string the Rust golden test compares two runs on.
pub fn digest(rows: &[ChainRow]) -> Vec<u8> {
    let mut out = Vec::new();
    for row in rows {
        // Every field of every row is `Serialize` and none is a float, so this cannot
        // fail on well-formed data; a panic here would mean the trace type itself is
        // unserializable, which is a build-time mistake, not a runtime condition.
        let line = serde_json::to_vec(row).expect("a ChainRow is always serializable");
        out.extend_from_slice(&line);
        out.push(b'\n');
    }
    out
}

/// The `cloid` as it appears in a trace: `0x` + 32 hex digits, fixed width.
///
/// Fixed width so a lexical sort of two golden files is also a numeric one, and so a
/// leading-zero id cannot be mistaken for a shorter one in a diff.
pub fn cloid_hex(cloid: u128) -> String {
    format!("0x{cloid:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn row() -> ChainRow {
        ChainRow {
            seq: 0,
            ts_event: 10,
            clock_ns: 10,
            symbol_id: Some(1),
            kind: "market",
            values: BTreeMap::from([
                ("mid", Cell::Money(dec!(100.5))),
                ("mark_px", Cell::Absent),
                ("open_orders", Cell::Count(0)),
            ]),
            decisions: BTreeMap::from([("plan", String::new())]),
        }
    }

    #[test]
    fn a_missing_column_serializes_as_null_not_as_zero() {
        // The difference between a risk gate failing closed and a risk gate sizing
        // against a price that does not exist. A `Cell::Absent` that rendered as `0`
        // would make the second look like the first in every golden file.
        let json = serde_json::to_string(&row()).unwrap();
        assert!(json.contains(r#""mark_px":null"#), "{json}");
        assert!(json.contains(r#""mid":"100.5""#), "money is a string");
        assert!(json.contains(r#""open_orders":0"#), "a count is a number");
    }

    #[test]
    fn a_trace_digest_is_one_line_per_row_in_row_order() {
        let rows = vec![row(), row()];
        let bytes = digest(&rows);
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 2);
        assert_eq!(digest(&rows), bytes, "the digest is a pure function");
    }

    #[test]
    fn a_cloid_keeps_all_128_bits_and_a_fixed_width() {
        // The planner tags bit 127, so a cloid is routinely wider than a u64 and
        // always has a leading `8`. A truncating or variable-width rendering would
        // make two different orders compare equal in a golden diff.
        let tagged = 1u128 << 127 | 0x2a;
        let hex = cloid_hex(tagged);
        assert_eq!(hex, "0x8000000000000000000000000000002a");
        assert_eq!(hex.len(), 34);
        assert_eq!(
            cloid_hex(0).len(),
            hex.len(),
            "width does not depend on value"
        );
    }
}
