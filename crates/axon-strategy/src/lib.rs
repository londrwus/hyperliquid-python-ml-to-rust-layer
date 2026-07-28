//! # axon-strategy
//!
//! The **inward port** (`docs/06-strategy-contract.md`) and the adapter behind it.
//!
//! Two halves that meet in the middle:
//!
//! - **The contract.** A strategy is a plugin: the engine calls its callbacks and
//!   it emits [`Signal`]s through the provided [`StrategyContext`] facade and
//!   nothing else. This mirrors the Python `axon.strategy` contract so a strategy
//!   can be promoted from Python (Boundary B) to Rust (Boundary A) without the
//!   engine, adapters, or tests changing shape.
//! - **The signal-to-order adapter.** [`SignalReader`] drains the shared-memory
//!   ring and admits only records this build can act on; [`Planner`] turns an
//!   admitted target-position [`Signal`] plus the position we hold into concrete
//!   [`OrderRequest`](axon_providers::OrderRequest)s. This is where "the strategy
//!   declares what position it WANTS; the Rust execution engine decides HOW to get
//!   there" actually happens (ADR-0014).
//!
//! Everything here is **synchronous and pure**: the deterministic core never
//! touches tokio, and the planner takes no clock, so a replayed session produces
//! the same orders — down to the same `cloid` — as the live one did.

#![deny(unsafe_code)]

pub mod book;
pub mod fixed;
pub mod planner;
pub mod reader;

pub use book::{
    BookReject, BookStats, NetTarget, OnSilence, Overlap, StrategyId, StrategyPolicy, TargetBook,
    NET_SEQ_TAG,
};
pub use fixed::{decimal_to_fixed, fixed_to_decimal, scale_fixed};
pub use planner::{
    cloid_for, urgency_rule, Anchor, NoOrder, Plan, PlanContext, Planner, PlannerConfig, Quote,
    UrgencyRule, WorkingOrder, CLOID_PLANNER_TAG, URGENCY_TABLE,
};
// Re-exported beside the planner types the runtime already imports from here: a caller
// building a `PlanContext` must name a `Precision`, and having to reach into
// `axon-providers` for the word would make the required field feel optional.
pub use axon_providers::Precision;
pub use reader::{
    DrainReport, ReaderConfig, ReplaySource, RingSignalReader, SignalReader, SignalReject,
    SignalSource, SignalStats,
};

use axon_contracts::Signal;
use axon_providers::{Bbo, Trade};

/// The facade a strategy uses to emit signals. The engine drains it after each
/// callback and forwards the signals to risk + execution. A strategy cannot touch
/// the bus, the book, or the adapters directly.
#[derive(Debug, Default)]
pub struct StrategyContext {
    pending: Vec<Signal>,
}

impl StrategyContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit a signal (target position). Order-level mechanics are the engine's job.
    pub fn emit(&mut self, signal: Signal) {
        self.pending.push(signal);
    }

    /// The engine takes emitted signals after each callback.
    pub fn take_pending(&mut self) -> Vec<Signal> {
        std::mem::take(&mut self.pending)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

/// The strategy contract. Override only the callbacks you need; the engine
/// guarantees ordering and delivers events through the deterministic loop.
pub trait Strategy: Send {
    fn on_start(&mut self, _ctx: &mut StrategyContext) {}
    fn on_stop(&mut self, _ctx: &mut StrategyContext) {}
    fn on_bbo(&mut self, _bbo: &Bbo, _ctx: &mut StrategyContext) {}
    fn on_trade(&mut self, _trade: &Trade, _ctx: &mut StrategyContext) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::SymbolId;

    /// A trivial strategy that flattens on every trade — exercises the contract.
    struct FlattenOnTrade;
    impl Strategy for FlattenOnTrade {
        fn on_trade(&mut self, trade: &Trade, ctx: &mut StrategyContext) {
            let mut s = Signal::zeroed();
            s.symbol_id = trade.symbol_id.get();
            s.ts_event = trade.ts_event;
            s.flags = axon_contracts::FLAG_CLOSE;
            ctx.emit(s);
        }
    }

    #[test]
    fn strategy_emits_through_context_only() {
        let mut strat: Box<dyn Strategy> = Box::new(FlattenOnTrade);
        let mut ctx = StrategyContext::new();
        let trade = Trade {
            symbol_id: SymbolId::new(3),
            px: axon_core::Decimal::from(100),
            sz: axon_core::Decimal::from(1),
            side: axon_core::Side::Buy,
            ts_event: 42,
        };
        strat.on_trade(&trade, &mut ctx);
        let out = ctx.take_pending();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].symbol_id, 3);
        assert!(out[0].is_close());
        assert_eq!(ctx.pending_len(), 0);
    }

    /// The whole path in one place: Python's bytes → validated record → orders.
    ///
    /// The failure mode it guards is integration drift — each half can be correct
    /// while the seam between them is not, and this is the seam `docs/01` step 4
    /// describes.
    #[test]
    fn a_signal_off_the_ring_becomes_an_order_for_the_delta() {
        use axon_contracts::SCHEMA_VERSION;
        use axon_core::{Side, Tif};
        use rust_decimal_macros::dec;

        let mut src = ReplaySource::default();
        // Emitted at t=1s with a 500 ms window, asking to be long 1.5.
        src.push(Signal::target_position(
            41,
            1_000_000_000,
            7,
            150_000_000,
            0,
            0,
            500,
            3,
            0,
        ));
        // The next decision, but it sat in the ring too long to act on.
        src.push(Signal::target_position(
            42,
            1_000_000_000,
            7,
            900_000_000,
            0,
            0,
            10,
            3,
            0,
        ));
        let mut reader = SignalReader::new(src);
        let planner = Planner::default();

        let mut admitted = Vec::new();
        let report = reader.drain(1_200_000_000, 8, |s| admitted.push(s));
        assert_eq!(report.accepted, 1);
        assert_eq!(report.rejected, 1, "the 10 ms signal is 200 ms late");
        assert_eq!(admitted[0].schema_version, SCHEMA_VERSION);

        let ctx = PlanContext::new(dec!(0.5), Quote::new(dec!(100), dec!(101)));
        let plan = planner.plan(&admitted[0], &ctx);
        assert_eq!(plan.orders.len(), 1);
        let o = &plan.orders[0];
        assert_eq!(o.side, Side::Buy);
        assert_eq!(o.qty, dec!(1), "1.5 target minus 0.5 held");
        assert_eq!(o.price, Some(dec!(100)));
        assert_eq!(o.tif, Tif::PostOnly);
        assert_eq!(o.cloid, cloid_for(&admitted[0]));
        assert!(!o.reduce_only);
    }
}
