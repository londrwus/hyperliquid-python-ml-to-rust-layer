//! # axon-replay
//!
//! Capture and deterministic replay of the core [`Event`](axon_core::Event) stream —
//! the bottom rung of the validation ladder in `docs/07-parity-and-testing.md`
//! ("capture a real event log, replay it through the exact production code path,
//! assert outputs match a stored reference"). Everything above that rung — model
//! parity, feature parity, shadow-trading diffs — compares two runs, and none of
//! those comparisons mean anything until two runs over one input are known to be
//! identical.
//!
//! Five pieces, and they compose in the obvious direction:
//!
//! - [`Capture`] is an [`EventHandler`](axon_core::EventHandler), so recording a
//!   session is adding a handler to the core loop. It writes JSONL: a versioned
//!   [`LogHeader`], then the [`InstrumentSet`] the session planned against, then one
//!   [`LogRecord`] per event in the order the bus delivered them.
//! - [`LogReader`] reads one back a record at a time, refusing a log this build cannot
//!   interpret rather than deserializing it into whatever the current types happen to
//!   accept. [`EventLog`] is that reader drained into memory, for logs small enough to
//!   hold; a session-sized log is streamed and never held.
//! - [`SignalLog`] is the other half of a recorded session: the records the *strategy*
//!   sent. Without it a replay can re-observe a session but cannot re-decide it, and
//!   the reconciliation and strategy halves of the chain would be replayed against a
//!   producer that never spoke.
//! - [`ReplaySource`] republishes the events onto an
//!   [`EventSender`](axon_core::EventSender) and drives the core's own
//!   `run_blocking_clocked` against a [`ManualClock`](axon_core::ManualClock), so
//!   handlers observe **event time**.
//! - [`chain`] is the vocabulary a golden comparison diffs: the book, the marks, the
//!   tracker's reconciled position and the planner's emitted orders, per event.
//!
//! The whole design rests on one property, and [`replay`] holds the test that states
//! it: replaying a log twice through the same handler produces byte-identical output.
//!
//! ## One fan-out, not two
//!
//! The chain a replay drives is the *production* chain — `axon_runtime::CoreHandler`
//! for the book → marks → tracker fan-out (ADR-0013 §1) and
//! `axon_runtime::IntentSource` for the strategy pass — not a copy of either. A second
//! fan-out here would drift from the live one, and parity would become a claim about
//! this harness rather than about the code.
//!
//! That is why the runtime is a **dev**-dependency and nothing in `src/` names it: the
//! production edge runs the other way, since a live session records itself by adding
//! [`Capture`] to its own handler chain. Cargo allows a cycle through a dev-dependency
//! and forbids one through a normal dependency, so the driver lives in
//! `examples/chain/mod.rs`, shared by the replay binary and this crate's tests.
//!
//! ## The grid a session planned on
//!
//! A live session plans on the venue's own price and lot grids and **rounds**
//! (ADR-0025). A log used to carry no table, so a replay of it planned *unconstrained*
//! and produced prices the session could not have sent — reported by
//! `python/axon/backtest/golden.py` as a strategy flip on every order the grid moved,
//! inside the harness whose one job is to tell a strategy change from a harness change.
//! `SCHEMA_VERSION` 2 puts the table in the log ([`instruments`], ADR-0027), and a log
//! written before that is **refused by name** rather than replayed loosely.
//!
//! ## What replay does not reproduce
//!
//! A log is a recording, not a counterparty. It contains the fills the *captured*
//! session received for the orders the *captured* session sent — so an order a replay
//! would place gets no fill, moves no price, and takes no queue position. The replay
//! does not even write it into the tracker, because that would mean inventing an
//! `OrderAck` the venue never gave. Replay therefore answers "does this code still
//! produce what it produced before?", not "would this strategy have made money?". The
//! second question needs a simulated venue behind the provider port, which is
//! deliberately not in this crate. See [`replay`], [`chain`] and ADR-0018 — a harness
//! that blurs this manufactures confidence, which is worse than having no harness.

#![deny(unsafe_code)]

pub mod capture;
pub mod chain;
pub mod error;
pub mod instruments;
pub mod log;
pub mod replay;
pub mod signals;
pub mod upcast;

#[cfg(test)]
mod test_support;

pub use capture::{Capture, CaptureStats};
pub use chain::{
    Cell, ChainRow, ChainSummary, PlannedCancel, PlannedOrder, SignalCounters, SymbolState,
    RESULT_SCHEMA, RESULT_SCHEMA_VERSION,
};
pub use error::ReplayError;
pub use instruments::{InstrumentSet, LoggedInstrument};
pub use log::{
    EventLog, LogHeader, LogLine, LogReader, LogRecord, LoggedEvent, SCHEMA, SCHEMA_VERSION,
};
pub use replay::{ReplayOrder, ReplayReport, ReplaySource};
pub use signals::{LoggedSignal, SignalLog, SignalRecord, SIGNAL_SCHEMA, SIGNAL_SCHEMA_VERSION};
pub use upcast::{upcast_v1, UpcastReport};
