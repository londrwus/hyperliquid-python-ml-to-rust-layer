//! The deterministic, event-time-ordered queue that will feed the single-threaded
//! core loop. It is generic over the payload `T` so it can carry ticks, bars, book
//! deltas, fills, and timers alike as those types land in later phases.
//!
//! Ordering is **(event-time, insertion-order)**: events are delivered oldest
//! event-time first, and ties break by the order they were pushed. That second
//! key is what makes replay reproducible — two events stamped the same nanosecond
//! always come out in the same order.

use crate::clock::Nanos;
use crate::exec::ExecEvent;
use crate::market::MarketEvent;
use core::cmp::Ordering;
use std::collections::BinaryHeap;

/// The single event type that flows on the core [`bus`](crate::bus). Everything the
/// deterministic loop reacts to is a variant here: public market data, and what the
/// venue told us about our own orders. Timer variants land with the scheduler.
///
/// Both kinds share one bus deliberately. A fill and the book update that caused it
/// must be ordered against each other by event time — if they arrived on separate
/// channels the core could see a fill before the trade that produced it, and replay
/// would stop matching live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Market(MarketEvent),
    Exec(ExecEvent),
}

impl Event {
    /// The event's own timestamp — the key the deterministic core orders on.
    pub fn ts_event(&self) -> Nanos {
        match self {
            Event::Market(m) => m.ts_event(),
            Event::Exec(e) => e.ts_event(),
        }
    }
}

impl From<MarketEvent> for Event {
    fn from(m: MarketEvent) -> Self {
        Event::Market(m)
    }
}

impl From<ExecEvent> for Event {
    fn from(e: ExecEvent) -> Self {
        Event::Exec(e)
    }
}

/// An internal heap entry pairing a payload with its ordering keys.
struct Entry<T> {
    ts_event: Nanos,
    /// Monotonic tie-breaker assigned at push time (stable ordering).
    seq: u64,
    item: T,
}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ts_event == other.ts_event && self.seq == other.seq
    }
}
impl<T> Eq for Entry<T> {}

impl<T> Ord for Entry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap; invert so the *smallest* (ts, seq) is "max"
        // and therefore popped first.
        other
            .ts_event
            .cmp(&self.ts_event)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A min-heap on event time with a stable insertion tie-breaker.
pub struct TimedQueue<T> {
    heap: BinaryHeap<Entry<T>>,
    counter: u64,
}

impl<T> Default for TimedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TimedQueue<T> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            counter: 0,
        }
    }

    /// Push an event stamped with its own event time. Out-of-order pushes are
    /// fine — the queue reorders them by `ts_event`.
    pub fn push(&mut self, ts_event: Nanos, item: T) {
        let seq = self.counter;
        self.counter += 1;
        self.heap.push(Entry {
            ts_event,
            seq,
            item,
        });
    }

    /// Pop the earliest event (oldest `ts_event`, then earliest-pushed).
    pub fn pop(&mut self) -> Option<(Nanos, T)> {
        self.heap.pop().map(|e| (e.ts_event, e.item))
    }

    /// Peek the event time of the next event without removing it.
    pub fn peek_ts(&self) -> Option<Nanos> {
        self.heap.peek().map(|e| e.ts_event)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_event_time_order() {
        let mut q = TimedQueue::new();
        q.push(30, "c");
        q.push(10, "a");
        q.push(20, "b");
        assert_eq!(q.pop(), Some((10, "a")));
        assert_eq!(q.pop(), Some((20, "b")));
        assert_eq!(q.pop(), Some((30, "c")));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn ties_break_by_insertion_order() {
        let mut q = TimedQueue::new();
        q.push(100, "first");
        q.push(100, "second");
        q.push(100, "third");
        // Same timestamp → stable, deterministic delivery order.
        assert_eq!(q.pop(), Some((100, "first")));
        assert_eq!(q.pop(), Some((100, "second")));
        assert_eq!(q.pop(), Some((100, "third")));
    }

    #[test]
    fn peek_ts_reports_next() {
        let mut q = TimedQueue::new();
        assert_eq!(q.peek_ts(), None);
        q.push(5, 1);
        q.push(3, 2);
        assert_eq!(q.peek_ts(), Some(3));
        assert_eq!(q.len(), 2);
    }

    fn a_fill(ts: Nanos) -> ExecEvent {
        use crate::exec::{Fill, Liquidity};
        use crate::ids::{Cloid, OrderId, SymbolId};
        ExecEvent::Fill(Fill {
            symbol_id: SymbolId::new(1),
            order_id: OrderId::new(1),
            cloid: Some(Cloid::new(1)),
            side: crate::enums::Side::Buy,
            qty: rust_decimal::Decimal::ONE,
            price: rust_decimal::Decimal::ONE_HUNDRED,
            fee: rust_decimal::Decimal::ZERO,
            closed_pnl: rust_decimal::Decimal::ZERO,
            liquidity: Liquidity::Taker,
            trade_id: 1,
            ts_event: ts,
        })
    }

    fn a_trade(ts: Nanos) -> MarketEvent {
        use crate::ids::SymbolId;
        use crate::market::Trade;
        MarketEvent::Trade(Trade {
            symbol_id: SymbolId::new(1),
            px: rust_decimal::Decimal::ONE_HUNDRED,
            sz: rust_decimal::Decimal::ONE,
            side: crate::enums::Side::Buy,
            ts_event: ts,
        })
    }

    #[test]
    fn exec_events_ride_the_same_bus_as_market_data() {
        // `EventSender::send` takes `impl Into<Event>`, so an adapter can publish an
        // ExecEvent with no more ceremony than a MarketEvent.
        let (tx, rx) = crate::bus::bus(8);
        tx.send(a_fill(4_200)).unwrap();
        let got = rx.recv().unwrap();
        assert!(matches!(got, Event::Exec(ExecEvent::Fill(_))));
        assert_eq!(
            got.ts_event(),
            4_200,
            "Event::ts_event delegates to the fill"
        );
    }

    #[test]
    fn timed_queue_interleaves_market_and_exec_by_event_time() {
        // The reason both kinds share one bus: a fill and the trade that caused it must
        // be orderable against each other, or replay stops matching live.
        let mut q = TimedQueue::new();
        q.push(20, Event::from(a_trade(20)));
        q.push(10, Event::from(a_fill(10)));
        assert!(matches!(q.pop(), Some((10, Event::Exec(_)))));
        assert!(matches!(q.pop(), Some((20, Event::Market(_)))));
    }
}
