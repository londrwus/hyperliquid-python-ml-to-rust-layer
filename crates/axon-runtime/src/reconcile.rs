//! The `POST /info` reconciliation poll — how a restarted session learns what it
//! already has at the venue.
//!
//! Hyperliquid's `orderUpdates` channel **never snapshots** (ADR-0010). A process
//! that restarts, or a socket that drops for thirty seconds, therefore has no
//! stream-side way to discover the orders still resting under its own account, and an
//! order we do not know about is exposure the risk gate does not count and a
//! cancel-all sweep does not reach. REST is the only source of that truth, so it is
//! polled on an interval rather than only at startup: a mid-session disconnect has
//! exactly the same effect as a restart.
//!
//! Two shapes of divergence come out of a poll, and they are treated very differently:
//!
//! - **The venue has an order we do not.** Adopted, by publishing the venue's own row
//!   onto the bus as an [`OrderUpdate`]. It goes through the bus rather than straight
//!   into the tracker so that REST truth and WS truth take one path, in one event-time
//!   order, and replay sees what live saw.
//! - **We have an order the venue does not list.** *Reported, never written off.*
//!   Synthesizing a cancellation we did not observe would silently drop exposure from
//!   our own risk view — and under-counting exposure is the direction that places the
//!   order that breaches the limit. The honest resolution is a targeted `orderStatus`
//!   read, which is left for when something needs it; until then the count is on the
//!   status line, where a human can see that our view and the venue's disagree.
//!
//! The diff itself ([`diff`]) is a pure function of a tracker snapshot and a venue
//! snapshot, so every case is unit-testable with no network.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use axon_core::{
    Clock as _, Decimal, Event, EventSender, ExecEvent, Nanos, OrderId, Position, SendError,
    SymbolId,
};
use axon_execution::OrderTracker;
use axon_provider_hyperliquid::info::{Decoded, OpenOrder, UserState};
use axon_provider_hyperliquid::{RateGovernor, SymbolMap};

use crate::health::SessionHealth;

/// One symbol where our position and the venue's disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionDrift {
    pub symbol_id: SymbolId,
    /// What our fill accounting says we hold.
    pub ours: Decimal,
    /// What the venue's `clearinghouseState` says.
    pub venue: Decimal,
}

/// What one poll found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReconcileReport {
    /// Orders the venue reports as live.
    pub venue_open: usize,
    /// Of those, the ones the tracker had no record of.
    pub unknown_to_tracker: Vec<OrderId>,
    /// Orders the tracker believes are live that the venue did not list, ignoring
    /// anything younger than the grace window.
    pub missing_at_venue: Vec<OrderId>,
    pub position_drift: Vec<PositionDrift>,
    /// True when the venue returned rows on instruments outside our symbol map (HIP-3
    /// perps, spot). The view is then knowingly partial — worth saying out loud rather
    /// than concluding we have seen everything.
    pub incomplete: bool,
}

impl ReconcileReport {
    pub fn is_clean(&self) -> bool {
        self.missing_at_venue.is_empty() && self.position_drift.is_empty() && !self.incomplete
    }
}

/// Compare our view against the venue's.
///
/// `symbols` is the configured universe: position drift is checked per configured
/// symbol rather than per venue-reported position, because a position we hold and the
/// venue reports as flat is exactly as much of a divergence as the reverse, and the
/// venue only lists non-flat ones.
///
/// `grace_ns` suppresses the read-your-writes race: `frontendOpenOrders` and a submit
/// are independent round-trips, so an order accepted moments ago can legitimately be
/// absent from a snapshot already in flight. Without the grace window every placement
/// would raise a false divergence for one poll.
pub fn diff(
    tracker: &OrderTracker,
    venue_open: &Decoded<OpenOrder>,
    venue_positions: &[Position],
    symbols: &[SymbolId],
    now_ns: Nanos,
    grace_ns: Nanos,
) -> ReconcileReport {
    let venue_ids: Vec<OrderId> = venue_open.items.iter().map(|o| o.order_id).collect();

    let unknown_to_tracker = venue_open
        .items
        .iter()
        .filter(|o| tracker.order_by_id(o.order_id).is_none())
        .map(|o| o.order_id)
        .collect();

    let missing_at_venue = tracker
        .open_orders()
        .filter_map(|o| o.order_id)
        .filter(|id| !venue_ids.contains(id))
        .filter(|id| {
            tracker
                .order_by_id(*id)
                .is_some_and(|o| now_ns.saturating_sub(o.last_update) > grace_ns)
        })
        .collect();

    let position_drift = symbols
        .iter()
        .filter_map(|s| {
            let ours = tracker.position(*s).qty;
            let venue = venue_positions
                .iter()
                .find(|p| p.symbol_id == *s)
                .map(|p| p.qty)
                .unwrap_or(Decimal::ZERO);
            (ours != venue).then_some(PositionDrift {
                symbol_id: *s,
                ours,
                venue,
            })
        })
        .collect();

    ReconcileReport {
        venue_open: venue_open.items.len(),
        unknown_to_tracker,
        missing_at_venue,
        position_drift,
        incomplete: !venue_open.is_complete(),
    }
}

/// The async driver: read `/info`, publish what it found onto the bus, record the
/// divergences.
pub struct Reconciler {
    info_url: String,
    account: String,
    symbol_map: SymbolMap,
    symbols: Vec<SymbolId>,
    tracker: Arc<RwLock<OrderTracker>>,
    governor: Arc<RateGovernor>,
    bus: EventSender,
    health: Arc<SessionHealth>,
    grace_ns: Nanos,
    /// Read `userRateLimit` once every N cycles.
    rate_limit_every: u32,
    cycles: u32,
}

impl Reconciler {
    #[allow(clippy::too_many_arguments)] // a composition root wires many things; bundling them into a struct only moves the list
    pub fn new(
        info_url: impl Into<String>,
        account: impl Into<String>,
        symbol_map: SymbolMap,
        symbols: Vec<SymbolId>,
        tracker: Arc<RwLock<OrderTracker>>,
        governor: Arc<RateGovernor>,
        bus: EventSender,
        health: Arc<SessionHealth>,
        grace_ns: Nanos,
        rate_limit_every: u32,
    ) -> Self {
        Self {
            info_url: info_url.into(),
            account: account.into(),
            symbol_map,
            symbols,
            tracker,
            governor,
            bus,
            health,
            grace_ns,
            rate_limit_every: rate_limit_every.max(1),
            cycles: 0,
        }
    }

    /// One poll. Returns `None` when the rate governor declined to spend the weight,
    /// which is not a failure: info traffic must never be the reason a cancel cannot
    /// get out, so a poll that does not fit is simply skipped and the status line's
    /// "last reconciled" age grows.
    pub async fn cycle(&mut self) -> Option<Result<ReconcileReport, String>> {
        let now_ms = crate::dms::now_ms();
        if !self
            .governor
            .try_info("frontendOpenOrders", 0, now_ms)
            .is_admitted()
        {
            return None;
        }
        let open = match axon_provider_hyperliquid::fetch_frontend_open_orders(
            &self.info_url,
            &self.account,
            None,
            &self.symbol_map,
        )
        .await
        {
            Ok(o) => o,
            Err(e) => return Some(Err(format!("frontendOpenOrders: {e}"))),
        };

        // The account snapshot is a separate, cheap (weight 2) read; a refusal here
        // leaves the order side of the poll usable rather than discarding both.
        let state = if self
            .governor
            .try_info("clearinghouseState", 0, now_ms)
            .is_admitted()
        {
            match axon_provider_hyperliquid::fetch_user_state(
                &self.info_url,
                &self.account,
                &self.symbol_map,
            )
            .await
            {
                Ok(s) => Some(s),
                Err(e) => return Some(Err(format!("clearinghouseState: {e}"))),
            }
        } else {
            None
        };

        self.cycles = self.cycles.wrapping_add(1);
        if self.cycles % self.rate_limit_every == 0 {
            self.refresh_rate_budget(now_ms).await;
        }

        let report = {
            // Short read lock, no await inside: the core thread is writing to this
            // tracker on every event and must not be made to wait on a REST call.
            let Ok(t) = self.tracker.read() else {
                return Some(Err("order tracker lock poisoned".into()));
            };
            let positions = state
                .as_ref()
                .map(|s| s.positions.as_slice())
                .unwrap_or(&[]);
            // Wall clock, not event time: the grace window asks "how long ago did we
            // last hear about this order", which is a liveness question. Both clocks
            // are epoch nanoseconds — the venue stamps its own — so the subtraction is
            // meaningful, and the answer never feeds an ordering decision.
            diff(
                &t,
                &open,
                positions,
                &self.symbols,
                axon_core::SystemClock.now_ns(),
                self.grace_ns,
            )
        };

        self.publish(&open, state.as_ref());
        Some(Ok(report))
    }

    /// Publish the venue's view onto the bus.
    ///
    /// Every open order is republished, not only the unknown ones: the tracker is
    /// idempotent by construction (quantities combine with `max`, terminal stays
    /// terminal), so re-applying a known order costs a map lookup and repairs a
    /// partial fill whose WS frame was lost. Getting that repair for free is worth
    /// more than the handful of events it puts on the bus.
    fn publish(&self, open: &Decoded<OpenOrder>, state: Option<&UserState>) {
        for o in &open.items {
            self.send(Event::Exec(ExecEvent::Order(o.to_order_update())));
        }
        if let Some(s) = state {
            self.send(Event::Exec(ExecEvent::Account(s.account.clone())));
        }
    }

    fn send(&self, ev: Event) {
        // `try_send`, never `send`: this runs on the tokio runtime, and blocking a
        // runtime worker on a full bus would stall the WS reader that is trying to
        // drain the very backlog causing it. A dropped snapshot row is recovered by
        // the next poll.
        if let Err(SendError::Full) = self.bus.try_send(ev) {
            self.health.note_bus_full_drop();
        }
    }

    /// Fold the venue's own accounting into the governor, so the local estimate
    /// cannot drift into a 429 (or into refusing orders we could have sent).
    async fn refresh_rate_budget(&self, now_ms: u64) {
        if !self
            .governor
            .try_info("userRateLimit", 0, now_ms)
            .is_admitted()
        {
            return;
        }
        match axon_provider_hyperliquid::fetch_user_rate_limit(&self.info_url, &self.account).await
        {
            Ok(s) => self.governor.observe_rate_limit_status(
                s.n_requests_used,
                s.n_requests_cap,
                s.n_requests_surplus,
                now_ms,
            ),
            Err(e) => eprintln!("reconcile: userRateLimit read failed: {e}"),
        }
    }

    /// Poll until `stop` is signalled.
    pub async fn run(&mut self, interval: Duration, stop: &tokio::sync::Notify) {
        loop {
            match self.cycle().await {
                Some(Ok(report)) => {
                    self.health.note_reconcile_ok(crate::dms::now_ms());
                    self.health
                        .note_adopted(report.unknown_to_tracker.len() as u64);
                    self.health
                        .set_venue_missing(report.missing_at_venue.len() as u64);
                    self.health
                        .set_position_drift(report.position_drift.len() as u64);
                    if !report.is_clean() {
                        eprintln!("reconcile: {report:?}");
                    }
                }
                Some(Err(e)) => {
                    self.health.note_reconcile_failure();
                    eprintln!("reconcile: {e}");
                }
                None => {}
            }
            tokio::select! {
                biased;
                _ = stop.notified() => return,
                _ = tokio::time::sleep(interval) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, OrderStatus, Side, Tif};
    use axon_provider_hyperliquid::info::decode_open_orders;
    use axon_providers::{OrderAck, OrderRequest};
    use rust_decimal_macros::dec;

    const BTC: SymbolId = SymbolId::new(0);
    const SEC: Nanos = 1_000_000_000;

    fn symbols() -> SymbolMap {
        SymbolMap::from_perps(["BTC", "ETH"])
    }

    /// One live venue order, oid 555, placed at t=1 s (the venue stamps ms).
    fn venue_open(json: &str) -> Decoded<OpenOrder> {
        decode_open_orders(json, &symbols()).expect("decode")
    }

    const ONE_ORDER: &str = r#"[{"coin":"BTC","side":"B","limitPx":"50000.0","sz":"1.0",
        "oid":555,"timestamp":1000,"origSz":"1.0","reduceOnly":false}]"#;
    const NO_ORDERS: &str = "[]";
    const UNKNOWN_COIN: &str = r#"[{"coin":"xyz:SP500","side":"B","limitPx":"1.0","sz":"1.0",
        "oid":777,"timestamp":1000,"origSz":"1.0","reduceOnly":false}]"#;

    fn tracked(tracker: &mut OrderTracker, oid: u64, qty: Decimal, ts: Nanos) {
        let req = OrderRequest::limit(
            BTC,
            Side::Buy,
            qty,
            dec!(50_000),
            Tif::Gtc,
            Cloid::new(oid as u128),
        );
        let ack = OrderAck {
            cloid: Cloid::new(oid as u128),
            order_id: Some(OrderId::new(oid)),
            status: OrderStatus::Resting,
        };
        tracker.on_ack(&req, &ack, ts);
    }

    #[test]
    fn an_order_the_tracker_never_saw_is_flagged_for_adoption() {
        // The restart case: our process is new, the orders are not.
        let t = OrderTracker::new();
        let r = diff(&t, &venue_open(ONE_ORDER), &[], &[BTC], 10 * SEC, 5 * SEC);
        assert_eq!(r.venue_open, 1);
        assert_eq!(r.unknown_to_tracker, vec![OrderId::new(555)]);
        assert!(r.missing_at_venue.is_empty());
    }

    #[test]
    fn a_freshly_placed_order_is_not_written_off_before_the_grace_window() {
        // The read and the submit race. Treating that race as a divergence would make
        // every placement look like a lost order for one poll.
        let mut t = OrderTracker::new();
        tracked(&mut t, 555, dec!(1), 10 * SEC);
        let now = 12 * SEC; // 2 s old, grace is 5 s
        let r = diff(&t, &venue_open(NO_ORDERS), &[], &[BTC], now, 5 * SEC);
        assert!(r.missing_at_venue.is_empty(), "too young to judge");

        // Once it is older than the grace window, the disagreement is real.
        let r = diff(&t, &venue_open(NO_ORDERS), &[], &[BTC], 20 * SEC, 5 * SEC);
        assert_eq!(r.missing_at_venue, vec![OrderId::new(555)]);
        assert!(!r.is_clean());
    }

    #[test]
    fn a_missing_order_is_reported_and_never_written_off() {
        // Explicitly: the report says "we disagree", it does not cancel our own record.
        // Dropping the order would remove exposure from the risk view that the venue
        // may still hold, and under-counting exposure is what places the order that
        // breaches the limit.
        let mut t = OrderTracker::new();
        tracked(&mut t, 555, dec!(1), 0);
        let r = diff(&t, &venue_open(NO_ORDERS), &[], &[BTC], 60 * SEC, 5 * SEC);
        assert_eq!(r.missing_at_venue.len(), 1);
        assert_eq!(t.open_count(), 1, "our own record is untouched");
        assert_eq!(t.risk_position(BTC).qty, dec!(1), "exposure still counted");
    }

    #[test]
    fn position_drift_is_detected_in_both_directions() {
        let mut t = OrderTracker::new();
        // We think we are flat; the venue says we are long 2 (a fill we never saw).
        let venue = vec![Position {
            symbol_id: BTC,
            qty: dec!(2),
            avg_px: dec!(50_000),
            realized_pnl: Decimal::ZERO,
        }];
        let r = diff(&t, &venue_open(NO_ORDERS), &venue, &[BTC], 0, 0);
        assert_eq!(
            r.position_drift,
            vec![PositionDrift {
                symbol_id: BTC,
                ours: Decimal::ZERO,
                venue: dec!(2)
            }]
        );

        // And the reverse: we think we hold something the venue does not list at all.
        // The venue omits flat positions, so this only surfaces because the diff walks
        // the configured universe rather than the venue's rows.
        tracked(&mut t, 1, dec!(1), 0);
        let ev = Event::Exec(ExecEvent::Fill(axon_core::Fill {
            symbol_id: BTC,
            order_id: OrderId::new(1),
            cloid: Some(Cloid::new(1)),
            side: Side::Buy,
            qty: dec!(1),
            price: dec!(50_000),
            fee: Decimal::ZERO,
            closed_pnl: Decimal::ZERO,
            liquidity: axon_core::Liquidity::Taker,
            trade_id: 9,
            ts_event: 1,
        }));
        axon_core::EventHandler::on_event(&mut t, 1, &ev);
        let r = diff(&t, &venue_open(NO_ORDERS), &[], &[BTC], 0, 0);
        assert_eq!(r.position_drift[0].ours, dec!(1));
        assert_eq!(r.position_drift[0].venue, Decimal::ZERO);
    }

    #[test]
    fn an_order_on_an_untracked_instrument_makes_the_view_incomplete() {
        // A HIP-3 or spot order we cannot decode is still an order. Reporting "clean"
        // while one survives would misinform a caller about its own exposure.
        let t = OrderTracker::new();
        let r = diff(&t, &venue_open(UNKNOWN_COIN), &[], &[BTC], 0, 0);
        assert!(r.incomplete);
        assert!(!r.is_clean());
        assert_eq!(r.venue_open, 0, "the row itself could not be decoded");
    }

    #[test]
    fn agreement_reports_clean() {
        let mut t = OrderTracker::new();
        tracked(&mut t, 555, dec!(1), 0);
        let r = diff(&t, &venue_open(ONE_ORDER), &[], &[BTC], 60 * SEC, 5 * SEC);
        assert!(r.is_clean());
        assert!(r.unknown_to_tracker.is_empty());
        assert_eq!(r.venue_open, 1);
    }
}
