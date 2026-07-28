//! The in-process event bus: the seam where async I/O edges hand normalized
//! [`Event`]s to the single-threaded deterministic core (ADR-0008,
//! `docs/04-provider-abstraction.md`).
//!
//! It is a **bounded** multi-producer / single-consumer channel over
//! `crossbeam-channel`. Bounded on purpose: if the core falls behind, producers
//! feel backpressure instead of the process growing without limit. Crucially the
//! consumer side is **synchronous** — the core drains it with plain blocking/`try`
//! calls and never touches the tokio runtime, honoring "async only at the edges."
//!
//! Producers (venue adapters running on tokio) hold a cloneable [`EventSender`];
//! the core owns the one [`EventReceiver`].

use crate::event::Event;
use crossbeam_channel::{bounded, Receiver, Sender};

/// A cloneable handle used by async edges to publish events onto the bus.
#[derive(Clone)]
pub struct EventSender {
    tx: Sender<Event>,
}

/// The single consumer handle, owned by the deterministic core loop.
pub struct EventReceiver {
    rx: Receiver<Event>,
}

/// Create a bounded event bus with room for `capacity` in-flight events.
///
/// Returns the producer/consumer pair. Clone the [`EventSender`] once per
/// producer (feed/adapter); keep the single [`EventReceiver`] on the core thread.
pub fn bus(capacity: usize) -> (EventSender, EventReceiver) {
    let (tx, rx) = bounded(capacity);
    (EventSender { tx }, EventReceiver { rx })
}

impl EventSender {
    /// Publish an event, blocking if the bus is full (backpressure). Fails only
    /// once the core has dropped the receiver (shutdown).
    pub fn send(&self, ev: impl Into<Event>) -> Result<(), SendError> {
        self.tx.send(ev.into()).map_err(|_| SendError::Disconnected)
    }

    /// Publish without blocking. Returns [`SendError::Full`] if the bus is at
    /// capacity — a caller on the hot edge can then shed load rather than stall.
    pub fn try_send(&self, ev: impl Into<Event>) -> Result<(), SendError> {
        use crossbeam_channel::TrySendError;
        self.tx.try_send(ev.into()).map_err(|e| match e {
            TrySendError::Full(_) => SendError::Full,
            TrySendError::Disconnected(_) => SendError::Disconnected,
        })
    }
}

impl EventReceiver {
    /// Block until the next event is available. `None` once every sender is gone.
    pub fn recv(&self) -> Option<Event> {
        self.rx.recv().ok()
    }

    /// Take the next event if one is ready, without blocking. `None` when the bus
    /// is empty or every sender has disconnected.
    pub fn try_recv(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Why a publish failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// The bus is at capacity (only from [`EventSender::try_send`]).
    Full,
    /// The core dropped the receiver — the process is shutting down.
    Disconnected,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Full => write!(f, "event bus is full"),
            SendError::Disconnected => write!(f, "event bus receiver disconnected"),
        }
    }
}

impl std::error::Error for SendError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::{MarketEvent, Trade};
    use crate::{Side, SymbolId};
    use rust_decimal_macros::dec;

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
    fn delivers_in_fifo_order() {
        let (tx, rx) = bus(8);
        tx.send(trade(1)).unwrap();
        tx.send(trade(2)).unwrap();
        assert_eq!(rx.recv().unwrap().ts_event(), 1);
        assert_eq!(rx.recv().unwrap().ts_event(), 2);
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn bounded_try_send_reports_full() {
        let (tx, _rx) = bus(1);
        tx.try_send(trade(1)).unwrap();
        assert_eq!(tx.try_send(trade(2)), Err(SendError::Full));
    }

    #[test]
    fn send_fails_once_receiver_dropped() {
        let (tx, rx) = bus(4);
        drop(rx);
        assert_eq!(tx.send(trade(1)), Err(SendError::Disconnected));
    }

    #[test]
    fn multiple_producers_share_the_bus() {
        let (tx, rx) = bus(16);
        let tx2 = tx.clone();
        tx.send(trade(1)).unwrap();
        tx2.send(trade(2)).unwrap();
        assert_eq!(rx.len(), 2);
    }
}
