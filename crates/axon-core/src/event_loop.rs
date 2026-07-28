//! The single-threaded deterministic loop driver.
//!
//! The core reacts to events **one at a time, in delivery order**, on one thread.
//! That is what makes ordering reproducible, removes lock contention, and lets
//! backtest and live share a code path (`docs/01-architecture.md`). This module
//! is the mechanism: it drains the [`bus`](crate::bus) and dispatches each
//! [`Event`] to an [`EventHandler`]. Handlers own the mutable state (books,
//! positions) and must never block or do I/O — the async edges already did that.

use crate::bus::EventReceiver;
use crate::clock::ManualClock;
use crate::event::Event;
use crate::Nanos;

/// A consumer of core events. Invoked serially, in order, on the core thread.
pub trait EventHandler {
    /// React to one event. `ts_event` is the event's own timestamp (the ordering
    /// key), passed alongside so handlers need not re-extract it.
    fn on_event(&mut self, ts_event: Nanos, event: &Event);
}

/// Process every event currently buffered on the bus, then return — without
/// blocking for more. Returns how many events were dispatched. Use this to
/// interleave event processing with other periodic work on the core thread.
pub fn drain_available(rx: &EventReceiver, handler: &mut impl EventHandler) -> usize {
    let mut n = 0;
    while let Some(ev) = rx.try_recv() {
        handler.on_event(ev.ts_event(), &ev);
        n += 1;
    }
    n
}

/// Run the core loop: block for each event and dispatch it, until every sender
/// has been dropped (shutdown). This is the live main loop.
pub fn run_blocking(rx: &EventReceiver, handler: &mut impl EventHandler) {
    while let Some(ev) = rx.recv() {
        handler.on_event(ev.ts_event(), &ev);
    }
}

/// Like [`run_blocking`], but advances `clock` to each event's time *before*
/// dispatch, so a handler that reads the clock sees the current event's event
/// time (not wall-clock). This is the property replay relies on.
pub fn run_blocking_clocked(
    rx: &EventReceiver,
    clock: &ManualClock,
    handler: &mut impl EventHandler,
) {
    while let Some(ev) = rx.recv() {
        let ts = ev.ts_event();
        clock.set(ts);
        handler.on_event(ts, &ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::bus;
    use crate::clock::Clock;
    use crate::market::{MarketEvent, Trade};
    use crate::{Side, SymbolId};
    use rust_decimal_macros::dec;

    #[derive(Default)]
    struct Spy {
        seen: Vec<Nanos>,
    }
    impl EventHandler for Spy {
        fn on_event(&mut self, ts_event: Nanos, _event: &Event) {
            self.seen.push(ts_event);
        }
    }

    fn trade(ts: i64) -> MarketEvent {
        MarketEvent::Trade(Trade {
            symbol_id: SymbolId::new(1),
            px: dec!(100),
            sz: dec!(1),
            side: Side::Buy,
            ts_event: ts,
        })
    }

    #[test]
    fn drains_buffered_events_in_order() {
        let (tx, rx) = bus(16);
        tx.send(trade(10)).unwrap();
        tx.send(trade(20)).unwrap();
        tx.send(trade(30)).unwrap();
        let mut spy = Spy::default();
        let n = drain_available(&rx, &mut spy);
        assert_eq!(n, 3);
        assert_eq!(spy.seen, vec![10, 20, 30]);
        // Nothing left to drain.
        assert_eq!(drain_available(&rx, &mut spy), 0);
    }

    #[test]
    fn run_blocking_exits_when_senders_drop() {
        let (tx, rx) = bus(16);
        tx.send(trade(1)).unwrap();
        tx.send(trade(2)).unwrap();
        drop(tx); // no more producers → loop must terminate
        let mut spy = Spy::default();
        run_blocking(&rx, &mut spy);
        assert_eq!(spy.seen, vec![1, 2]);
    }

    #[test]
    fn clocked_run_advances_clock_to_event_time() {
        let (tx, rx) = bus(16);
        tx.send(trade(1_000)).unwrap();
        tx.send(trade(2_500)).unwrap();
        drop(tx);
        let clock = ManualClock::new(0);

        struct ClockSpy<'a> {
            clock: &'a ManualClock,
            at: Vec<Nanos>,
        }
        impl EventHandler for ClockSpy<'_> {
            fn on_event(&mut self, _ts: Nanos, _ev: &Event) {
                self.at.push(self.clock.now_ns());
            }
        }
        let mut spy = ClockSpy {
            clock: &clock,
            at: vec![],
        };
        run_blocking_clocked(&rx, &clock, &mut spy);
        assert_eq!(spy.at, vec![1_000, 2_500]);
        assert_eq!(clock.now_ns(), 2_500);
    }
}
