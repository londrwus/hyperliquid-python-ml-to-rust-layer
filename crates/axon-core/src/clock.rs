//! Clocks. The deterministic core keys ordering on **event time**, so a
//! [`ManualClock`] can replay a captured log bit-reproducibly; [`SystemClock`]
//! is for the async edges only, never for ordering decisions.

use core::sync::atomic::{AtomicI64, Ordering};

/// Nanoseconds since the Unix epoch. Signed so deltas are representable and to
/// match the `i64 ts_event` on the wire (`contracts/schema.toml`).
pub type Nanos = i64;

/// A source of "now" in event-time nanoseconds.
pub trait Clock: Send + Sync {
    fn now_ns(&self) -> Nanos;
}

/// Wall-clock time. Use at the I/O edges (timestamps on receipt, logging) — but
/// never to order events inside the deterministic loop.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ns(&self) -> Nanos {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0)
    }
}

/// A clock that advances only when told. Backtests and replay tests set it to the
/// event's own timestamp, making the whole core reproducible.
#[derive(Debug, Default)]
pub struct ManualClock {
    ns: AtomicI64,
}

impl ManualClock {
    pub fn new(start: Nanos) -> Self {
        Self {
            ns: AtomicI64::new(start),
        }
    }

    /// Jump to an absolute event time (must not move backwards in normal replay,
    /// but that invariant is enforced by the driver, not here).
    pub fn set(&self, ns: Nanos) {
        self.ns.store(ns, Ordering::SeqCst);
    }

    /// Advance by a delta.
    pub fn advance(&self, by: Nanos) {
        self.ns.fetch_add(by, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_ns(&self) -> Nanos {
        self.ns.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_is_deterministic() {
        let c = ManualClock::new(1_000);
        assert_eq!(c.now_ns(), 1_000);
        c.advance(500);
        assert_eq!(c.now_ns(), 1_500);
        c.set(42);
        assert_eq!(c.now_ns(), 42);
    }

    #[test]
    fn system_clock_is_monotonicish() {
        let c = SystemClock;
        // Just assert it produces a plausible post-2020 nanosecond timestamp.
        assert!(c.now_ns() > 1_577_836_800_000_000_000);
    }
}
