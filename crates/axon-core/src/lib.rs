//! # axon-core
//!
//! The venue- and strategy-agnostic heart of the execution plane. Per
//! `docs/01-architecture.md`, this crate "knows nothing about venues or
//! strategies": it holds the domain vocabulary (ids, enums, orders, positions,
//! normalized [`market`] and [`exec`] events), a deterministic **event-time** clock,
//! the [`TimedQueue`], the in-process [`bus`], and the single-threaded deterministic
//! [`event_loop`] (LMAX-Disruptor style ordering).
//!
//! Design rules honored here (see `crates/README.md`):
//! - **Fixed-point money math** ([`rust_decimal::Decimal`]), never `f64` for px/sz.
//! - **Event-time everywhere** — ordering keys on the event's own time, never
//!   wall-clock at receipt, so replay is reproducible.
//! - **Sync, no I/O.** The [`bus`] is a *synchronous* crossbeam channel: async
//!   edges publish onto it, the core drains it off the tokio runtime (ADR-0008).

#![deny(unsafe_code)]

pub mod bus;
pub mod clock;
pub mod enums;
pub mod error;
pub mod event;
pub mod event_loop;
pub mod exec;
pub mod ids;
pub mod market;
pub mod position;

pub use bus::{bus, EventReceiver, EventSender, SendError};
pub use clock::{Clock, ManualClock, Nanos, SystemClock};
pub use enums::{OrderType, Side, Tif};
pub use error::CoreError;
pub use event::{Event, TimedQueue};
pub use event_loop::{drain_available, run_blocking, run_blocking_clocked, EventHandler};
pub use exec::{
    AccountSnapshot, CancelReason, ExecEvent, Fill, Liquidity, OrderStatus, OrderUpdate,
};
pub use ids::{Cloid, OrderId, SymbolId};
pub use market::{
    Bbo, BookSnapshot, Candle, CandleInterval, Funding, Level, MarketEvent, Ticker, Trade,
};
pub use position::Position;

pub use rust_decimal::Decimal;
