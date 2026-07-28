//! # axon-runtime
//!
//! The process: config, the session supervisor, and the safety loop that lets a
//! session be left running.
//!
//! Every part a live session needs already existed elsewhere in the workspace — the
//! WS ingest and user channels, the signing execution client, the order tracker, the
//! risk gate, the rate governor, the `scheduleCancel` dead-man's switch, the `/info`
//! reads. This crate is what composes them, and composition is where the safety
//! properties actually live:
//!
//! - [`session`] wires the async venue edges to the synchronous deterministic core,
//!   which runs on its own thread and never touches tokio (ADR-0008).
//! - [`handler`] fans every event to the book, the mark cache and the order tracker in
//!   one ordering, so a fill is applied against the market state that caused it.
//! - [`quote`] is the one answer to "what is the top of book", shared by the planner and
//!   the publisher so features are never computed against a different book from the one
//!   orders are priced against — and aged, so a frozen feed reads as no quote at all.
//! - [`intent`] is the Py→Rust join: it drains the signal ring on the core thread,
//!   plans against the session's own position and book, and hands the result across a
//!   queue to the submit pipeline. Before it, the pipeline was fully built and nothing
//!   called it (ADR-0020).
//! - [`mdring`] is the Rust→Python half of the boundary: the core's own event stream
//!   published as `MdSlice`s so Python computes features on the same book the executing
//!   core saw. Before it, ADR-0012's ring had exactly one producer and it was an
//!   example program (ADR-0012, consequences).
//! - [`capture`] records a session — the events the core saw and the signals the
//!   strategy adapter read — so that it can be replayed later through the same code that
//!   produced it. Before it, `axon_replay::Capture` was installed by nothing but its own
//!   tests, so no real session had ever been recorded and the golden harness had only
//!   ever seen synthetic logs (ADR-0018).
//! - [`dms`] keeps the venue-side dead-man's switch armed, and treats a failed re-arm
//!   as the protection being gone rather than as a log line.
//! - [`reconcile`] polls `POST /info`, because Hyperliquid's `orderUpdates` channel
//!   never snapshots and a restarted process has no other way to find the orders it
//!   left resting.
//! - [`shutdown`] stops intents, sweeps, and only then decides whether the venue-side
//!   deadline should stand — in that order, for reasons documented there.
//! - [`pnl`] and [`latency`] are the two things a *trading* session has to be watched
//!   by that a read-only one does not: what it has done to the account, and how it
//!   fared against ceilings somebody declared in advance (ADR-0036).
//! - [`daybook`] is the one piece of session state that outlives the process, and it
//!   holds exactly one number: the venue's `accountValue` at the first reading of the
//!   UTC day. A daily loss bound that a restart clears is not a daily bound, and a
//!   crash-restart loop is precisely how a losing session restarts.
//! - [`health`] is the single line an operator watches the whole thing by.
//!
//! The default configuration is **offline**: `cargo run --bin axon` builds the real
//! core, runs a canned event stream ([`selftest`]) through it and exits, with no
//! socket, no key and no tokio in the process. Live wiring requires both an explicit
//! `environment` in a config file and, for mainnet, a second environment-variable
//! gate. See `docs/adr/0013-runtime-supervision-and-safety-loop.md`.

#![deny(unsafe_code)]

pub mod capture;
pub mod config;
pub mod core;
pub mod daybook;
pub mod dms;
pub mod flatten;
pub mod handler;
pub mod health;
pub mod intent;
pub mod latency;
pub mod mdring;
pub mod pnl;
pub mod quote;
pub mod reconcile;
pub mod selftest;
pub mod session;
pub mod shutdown;

pub use capture::{
    CaptureOutcome, CaptureProgress, CaptureStop, CaptureTap, CapturedSignals, RecordingSource,
    SessionRecorder,
};
pub use config::{
    CaptureConfig, Environment, IntentConfig, LatencyConfig, MdRingConfig, Network, PnlConfig,
    RuntimeConfig,
};
pub use daybook::{DayBook, DayBookFault, DayState};
pub use handler::CoreHandler;
pub use health::{CaptureLine, IntentLine, MdLine, SessionHealth, StatusSnapshot};
pub use intent::{Intent, IntentSource, IntentStats, RingIntentSource};
pub use latency::{LatencyBook, LatencySnapshot, Stage, StageReport};
pub use mdring::{MdPublisher, MdStats, MdWritePolicy};
pub use pnl::PnlSnapshot;
pub use quote::{top_of_book, TopOfBook, TopState};
pub use session::{run, run_live, run_offline, RuntimeError, SessionSummary};
