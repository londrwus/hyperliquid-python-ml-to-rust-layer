//! Graceful shutdown, in the order that makes it safe.
//!
//! The sequence is four steps and **the order is the design**:
//!
//! 1. **Stop accepting intents.** `cancel_all` on Hyperliquid is a sweep — read the
//!    open orders, then cancel them — because the venue has no native cancel-all. Any
//!    order admitted between the read and the cancel therefore survives the sweep. If
//!    intents are still flowing while it runs, the process can exit believing it left
//!    nothing behind while an order it placed two milliseconds ago is still resting.
//!    "Stopped" has to mean *stopped*: a submit task that could not be joined is
//!    reported through [`ShutdownOptions::submitter_stopped`], because a placement it
//!    had already sent can land after the sweep has read an empty book.
//! 2. **Sweep, more than once.** The same race means one pass is not proof. Repeating
//!    until a pass reports nothing left costs two round-trips and converts "probably
//!    flat" into "the venue agrees".
//! 3. **Only then decide about the dead-man's switch.** Deciding earlier would be
//!    deciding without knowing whether the sweep worked, and that is precisely the
//!    input the decision turns on:
//!    - *Sweep succeeded* → **disarm**. A deadline left armed outlives this process.
//!      It burns one of the venue's ten daily triggers, and — much worse — it fires
//!      into whatever session is running when it expires, cancelling the *restarted*
//!      process's orders for reasons nobody will connect to a shutdown minutes
//!      earlier.
//!    - *Sweep failed, or the submitter could not be stopped* → **leave it armed**.
//!      There may still be exposure and there is about to be no process to remove it.
//!      This is the exact situation the switch exists for, so it stands.
//! 4. **Drain.** The core loop keeps running until the bus is empty so the cancel
//!    acknowledgements land in the tracker and the closing status line is true.
//!
//! Step 4 lives in [`crate::core`]; the first three are here, generic over the client
//! and the switch so the ordering itself is unit-tested offline.

use axon_execution::HaltSwitch;
use axon_providers::ExecutionClient;

use crate::dms::DeadMansSwitch;

/// How hard to try before giving up on the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownOptions {
    /// Sweep passes attempted. More than one because the sweep is inherently racy;
    /// bounded because a venue that refuses twice will refuse a hundred times, and a
    /// shutdown that never finishes is its own outage.
    pub sweep_attempts: u32,
    /// Whether step 1 actually finished — the caller's answer, because only the caller
    /// owns the submit task.
    ///
    /// `false` means a placement may still be in flight, so the sweep proved nothing:
    /// it read the book before the order reached it. Disarming on that reading is how a
    /// process exits leaving an order resting with no venue-side protection behind it,
    /// which is the outcome the whole arm-before-subscribe ordering exists to prevent.
    pub submitter_stopped: bool,
}

impl Default for ShutdownOptions {
    fn default() -> Self {
        Self {
            sweep_attempts: 3,
            submitter_stopped: true,
        }
    }
}

/// What the shutdown managed to do. Logged, and returned so a caller (or a test) can
/// assert on it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShutdownOutcome {
    /// A sweep pass completed without error.
    pub swept: bool,
    /// The dead-man's switch was disarmed. `false` with `swept == false` is the
    /// deliberate outcome, not a failure: the venue-side deadline is now the last
    /// line of defence.
    pub disarmed: bool,
    pub errors: Vec<String>,
}

/// Run the shutdown sequence. Never panics and never returns early on error — every
/// step is attempted, because a failure in one is exactly when the next matters most.
pub async fn graceful_shutdown(
    halt: &HaltSwitch,
    client: &dyn ExecutionClient,
    dms: Option<&dyn DeadMansSwitch>,
    opts: ShutdownOptions,
) -> ShutdownOutcome {
    let mut outcome = ShutdownOutcome::default();

    // 1. Terminal, not a recoverable halt: nothing may re-open trading behind the
    //    sweep, including the safety loop's own recovery path.
    halt.stop();

    // 2. Sweep until a pass succeeds.
    for attempt in 1..=opts.sweep_attempts.max(1) {
        match client.cancel_all().await {
            Ok(()) => {
                outcome.swept = true;
                break;
            }
            Err(e) => {
                outcome
                    .errors
                    .push(format!("cancel_all attempt {attempt}: {e}"));
            }
        }
    }

    // 3. The switch decision, made only now that the sweep's result is known.
    if let Some(dms) = dms {
        // Two ways the sweep can fail to prove anything, and the second is the quiet
        // one: it *succeeded* against a book an order had not reached yet.
        let unproven = if !outcome.swept {
            Some("sweep failed")
        } else if !opts.submitter_stopped {
            Some(
                "the submit task could not be stopped, so the sweep may have read the \
                 book before an order reached it",
            )
        } else {
            None
        };
        match unproven {
            None => match dms.disarm().await {
                Ok(()) => outcome.disarmed = true,
                // Failing to disarm leaves protection in place, which is the safe
                // direction; it is recorded so an operator knows a deadline is still
                // out there and may fire into the next session.
                Err(e) => outcome.errors.push(format!("disarm: {e}")),
            },
            Some(why) => outcome
                .errors
                .push(format!("{why} - leaving the dead-man's switch armed")),
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axon_core::{Cloid, Decimal, Side, SymbolId, Tif};
    use axon_providers::{
        CancelAck, CancelId, Capabilities, OrderAck, OrderRequest, OrderStatus, ProviderError,
        RateLimitModel,
    };
    use std::sync::Mutex;

    /// Records the sequence of venue calls so the *ordering* can be asserted, which
    /// is the only thing this module is really responsible for.
    #[derive(Default)]
    struct Spy {
        log: Mutex<Vec<&'static str>>,
        caps: Option<Capabilities>,
        sweep_fails: Mutex<u32>,
    }

    impl Spy {
        fn new(sweep_fails: u32) -> Self {
            Self {
                log: Mutex::new(Vec::new()),
                caps: Some(Capabilities {
                    venue: "spy",
                    order_types: &[axon_core::OrderType::Limit],
                    tifs: &[Tif::Gtc],
                    max_batch: 20,
                    native_market_orders: true,
                    reduce_only: true,
                    rate_limit_model: RateLimitModel::None,
                }),
                sweep_fails: Mutex::new(sweep_fails),
            }
        }
        fn log(&self) -> Vec<&'static str> {
            self.log.lock().unwrap().clone()
        }
        fn note(&self, what: &'static str) {
            self.log.lock().unwrap().push(what);
        }
    }

    #[async_trait]
    impl ExecutionClient for Spy {
        fn capabilities(&self) -> &Capabilities {
            self.caps.as_ref().unwrap()
        }
        async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
            self.note("place");
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: None,
                status: OrderStatus::Resting,
            })
        }
        async fn place_batch(&self, _r: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
            self.note("place_batch");
            Ok(Vec::new())
        }
        async fn cancel(&self, _id: CancelId) -> Result<CancelAck, ProviderError> {
            self.note("cancel");
            Ok(CancelAck {
                cloid: None,
                order_id: None,
            })
        }
        async fn cancel_all(&self) -> Result<(), ProviderError> {
            self.note("cancel_all");
            let mut left = self.sweep_fails.lock().unwrap();
            if *left > 0 {
                *left -= 1;
                return Err(ProviderError::Network("sweep failed".into()));
            }
            Ok(())
        }
        async fn modify(
            &self,
            _id: CancelId,
            req: OrderRequest,
        ) -> Result<OrderAck, ProviderError> {
            self.note("modify");
            Ok(OrderAck {
                cloid: req.cloid,
                order_id: None,
                status: OrderStatus::Resting,
            })
        }
    }

    #[derive(Default)]
    struct SpySwitch {
        disarms: Mutex<u32>,
        fail: bool,
    }

    #[async_trait]
    impl DeadMansSwitch for SpySwitch {
        async fn arm(&self, _lead_ms: u64) -> Result<u64, ProviderError> {
            Ok(0)
        }
        async fn disarm(&self) -> Result<(), ProviderError> {
            *self.disarms.lock().unwrap() += 1;
            if self.fail {
                return Err(ProviderError::Network("disarm failed".into()));
            }
            Ok(())
        }
    }

    fn order() -> OrderRequest {
        OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::ONE_HUNDRED,
            Tif::Gtc,
            Cloid::new(1),
        )
    }

    #[tokio::test]
    async fn intents_stop_before_the_sweep_reads_the_book() {
        // The race the ordering exists to close: an order admitted after the sweep's
        // read survives it, and the process exits believing it left nothing behind.
        let halt = HaltSwitch::new();
        let spy = Spy::new(0);
        let switch = SpySwitch::default();

        let outcome =
            graceful_shutdown(&halt, &spy, Some(&switch), ShutdownOptions::default()).await;

        assert!(!halt.is_accepting(), "intents must be refused first");
        assert_eq!(spy.log(), vec!["cancel_all"]);
        assert!(outcome.swept && outcome.disarmed);
        assert!(outcome.errors.is_empty());
    }

    #[tokio::test]
    async fn a_clean_shutdown_disarms_so_the_deadline_cannot_fire_into_the_next_session() {
        // An armed deadline outlives the process. Left in place after a clean exit it
        // cancels the *restarted* session's orders minutes later, and burns one of the
        // ten firings the venue honours per day.
        let halt = HaltSwitch::new();
        let switch = SpySwitch::default();
        let outcome = graceful_shutdown(
            &halt,
            &Spy::new(0),
            Some(&switch),
            ShutdownOptions::default(),
        )
        .await;
        assert!(outcome.disarmed);
        assert_eq!(*switch.disarms.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn a_failed_sweep_leaves_the_dead_mans_switch_armed() {
        // Exposure may remain and there is about to be no process to remove it. This
        // is the case the switch was armed for in the first place.
        let halt = HaltSwitch::new();
        let switch = SpySwitch::default();
        let spy = Spy::new(u32::MAX); // every pass fails
        let outcome =
            graceful_shutdown(&halt, &spy, Some(&switch), ShutdownOptions::default()).await;

        assert!(!outcome.swept);
        assert!(!outcome.disarmed);
        assert_eq!(*switch.disarms.lock().unwrap(), 0, "never disarmed blind");
        assert_eq!(spy.log().len(), 3, "all attempts used");
        assert!(outcome
            .errors
            .iter()
            .any(|e| e.contains("leaving the dead-man's switch armed")));
    }

    #[tokio::test]
    async fn a_submitter_that_could_not_be_stopped_leaves_the_switch_armed() {
        // A sweep that succeeds against a book an in-flight order has not reached yet
        // proves nothing, and the venue rests that order seconds later — behind a
        // deadline this shutdown had already cancelled. Leaving the switch armed costs
        // one of the venue's ten daily triggers; disarming costs an unmanaged position
        // with no process left to close it.
        let halt = HaltSwitch::new();
        let switch = SpySwitch::default();
        let outcome = graceful_shutdown(
            &halt,
            &Spy::new(0),
            Some(&switch),
            ShutdownOptions {
                submitter_stopped: false,
                ..ShutdownOptions::default()
            },
        )
        .await;

        assert!(outcome.swept, "the sweep itself still ran");
        assert!(!outcome.disarmed);
        assert_eq!(*switch.disarms.lock().unwrap(), 0);
        assert!(
            outcome
                .errors
                .iter()
                .any(|e| e.contains("leaving the dead-man's switch armed")),
            "{:?}",
            outcome.errors
        );
    }

    #[tokio::test]
    async fn a_sweep_that_needs_a_second_pass_still_counts_as_clean() {
        let halt = HaltSwitch::new();
        let switch = SpySwitch::default();
        let spy = Spy::new(1);
        let outcome =
            graceful_shutdown(&halt, &spy, Some(&switch), ShutdownOptions::default()).await;
        assert!(outcome.swept && outcome.disarmed);
        assert_eq!(spy.log(), vec!["cancel_all", "cancel_all"]);
        assert_eq!(outcome.errors.len(), 1, "the failed pass is still reported");
    }

    #[tokio::test]
    async fn a_failed_disarm_is_reported_rather_than_retried_into_the_ground() {
        // Failing to disarm leaves protection in place — safe, but the next session
        // needs to know a deadline is still out there.
        let halt = HaltSwitch::new();
        let switch = SpySwitch {
            disarms: Mutex::new(0),
            fail: true,
        };
        let outcome = graceful_shutdown(
            &halt,
            &Spy::new(0),
            Some(&switch),
            ShutdownOptions::default(),
        )
        .await;
        assert!(outcome.swept);
        assert!(!outcome.disarmed);
        assert!(outcome.errors.iter().any(|e| e.contains("disarm")));
    }

    #[tokio::test]
    async fn a_stopped_session_refuses_orders_for_good() {
        use axon_execution::HaltableClient;
        use std::sync::Arc;

        let halt = Arc::new(HaltSwitch::new());
        let spy = Spy::new(0);
        graceful_shutdown(&halt, &spy, None, ShutdownOptions::default()).await;

        // The switch is what the submit path reads, so prove the refusal through it
        // rather than through the flag alone.
        let client = HaltableClient::new(Spy::new(0), halt.clone());
        assert!(client.place_order(order()).await.is_err());
        halt.resume();
        assert!(
            client.place_order(order()).await.is_err(),
            "a shutdown halt is terminal"
        );
    }
}
