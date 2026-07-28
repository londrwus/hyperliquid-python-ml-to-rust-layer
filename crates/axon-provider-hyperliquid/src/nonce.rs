//! The nonce manager (`docs/research/hyperliquid-execution.md`).
//!
//! Hyperliquid nonces are ms timestamps that must **strictly increase** — the
//! venue tracks the 100 highest per address and rejects anything not above the
//! smallest tracked, within `(T-2days, T+1day)`. One agent wallet per process ⇒
//! one tracker. We take the wall clock in ms but force strict monotonicity, so a
//! burst within one millisecond, or a clock that stalls or steps backward, still
//! yields increasing nonces.

use std::sync::atomic::{AtomicU64, Ordering};

/// Strictly-monotonic ms-timestamp nonce source (thread-safe, lock-free).
#[derive(Debug)]
pub struct NonceManager {
    last: AtomicU64,
}

impl NonceManager {
    pub fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
        }
    }

    /// The next nonce given the current wall-clock time in ms. Always strictly
    /// greater than the previous nonce; equal to `now_ms` once the clock has
    /// advanced past the last issued value.
    pub fn next(&self, now_ms: u64) -> u64 {
        let mut last = self.last.load(Ordering::Relaxed);
        loop {
            let candidate = now_ms.max(last + 1);
            match self.last.compare_exchange_weak(
                last,
                candidate,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return candidate,
                Err(observed) => last = observed, // lost the race — retry with the new value
            }
        }
    }
}

impl Default for NonceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn increments_within_the_same_millisecond() {
        let n = NonceManager::new();
        assert_eq!(n.next(1_000), 1_000);
        assert_eq!(n.next(1_000), 1_001);
        assert_eq!(n.next(1_000), 1_002);
    }

    #[test]
    fn survives_a_backward_clock() {
        let n = NonceManager::new();
        assert_eq!(n.next(1_000), 1_000);
        assert_eq!(n.next(500), 1_001); // clock went backward → still increases
        assert_eq!(n.next(999), 1_002);
    }

    #[test]
    fn tracks_a_forward_jump() {
        let n = NonceManager::new();
        assert_eq!(n.next(1_000), 1_000);
        assert_eq!(n.next(5_000), 5_000); // clock advanced → follows it
    }

    #[test]
    fn concurrent_next_calls_are_all_unique() {
        let n = Arc::new(NonceManager::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let n = Arc::clone(&n);
            handles.push(std::thread::spawn(move || {
                (0..1000).map(|_| n.next(1_000)).collect::<Vec<_>>()
            }));
        }
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        let total = all.len();
        all.dedup();
        assert_eq!(all.len(), total, "nonces must never collide across threads");
    }
}
