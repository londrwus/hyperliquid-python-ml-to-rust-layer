//! The rate-limit governor (`docs/research/hyperliquid-execution.md`).
//!
//! Being rate-limited while holding open orders you cannot cancel is the worst
//! state a trading system can reach: exposure you are unable to remove.
//! Hyperliquid deliberately makes that state avoidable — cancels get a strictly
//! larger address-based allowance than places, `min(limit + 100_000, limit * 2)`,
//! and the venue's own wording is *"this way, hitting the address-based rate
//! limit still allows open orders to be canceled."* That only holds if the client
//! never spends cancel headroom on places. Enforcing it is this file's purpose,
//! and the enforcement is **structural** (separate methods with separate
//! ceilings), not a policy a caller can opt out of.
//!
//! Two budgets, never collapsed into one counter, because a batch is priced
//! oppositely against them:
//!
//! - **Per-IP:** a sliding 1200-weight / 60 s window over *all* REST traffic.
//!   An exchange action costs `1 + batch_len / 40`, so a batch of 20 costs **1**.
//! - **Address-based (actions only, not info):** one credit per *order*, so the
//!   same batch of 20 costs **20**. The cap is `10_000 + floor(cumVlm)` plus any
//!   `reserveRequestWeight` surplus.
//!
//! Like [`NonceManager`](crate::nonce::NonceManager) this is a pure state machine
//! over an externally supplied `now_ms`: it never reads the clock, does no I/O and
//! knows nothing about tokio, which is what makes every branch deterministically
//! testable. `&self` everywhere with a `Mutex` inside, so the async execution
//! client can hold one behind an `Arc`.
//!
//! **Modelled:** the per-IP REST weight window, action weight, the info-weight
//! tiers (2 / 20 / 60) with the per-20-items surcharge, address-based action
//! credits with cap/used/surplus reconciliation, the cancel allowance, the
//! place-side reserves, the 5x stale-`expiresAfter` surcharge, and the throttled
//! (one action / 10 s) state.
//!
//! **Deliberately not modelled** — they belong to other components:
//! the WebSocket caps (10 connections, 30 new/min, 1000 subscriptions, 10 unique
//! users, 2000 msgs/min, 100 in-flight posts) are a property of the WS client;
//! the open-order cap (1000, +1 per 5M USDC, hard cap 5000) is position/OMS state,
//! not a rate; the `reserveRequestWeight` action itself lives in `/exchange` — we
//! only consume its result as [`observe_rate_limit_status`]'s `surplus` argument.
//! The venue's congestion advice (*"do not resend cancels whose results have
//! already been returned"*) is a caller-side de-duplication concern: the governor
//! cannot tell a resend from a first attempt.
//!
//! [`observe_rate_limit_status`]: RateGovernor::observe_rate_limit_status

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

use axon_providers::ProviderError;

/// Aggregated per-IP REST weight allowed per [`IP_WINDOW_MS`].
pub const IP_WEIGHT_LIMIT: u32 = 1200;

/// Length of the per-IP weight window.
pub const IP_WINDOW_MS: u64 = 60_000;

/// Every 40 orders in a batch add 1 to the action's per-IP weight.
pub const BATCH_WEIGHT_STEP: u32 = 40;

/// Address-based credits a brand-new address starts with, before any volume.
pub const INITIAL_ADDRESS_BUFFER: u64 = 10_000;

/// Additive part of the cancel allowance: `min(limit + this, limit * 2)`.
pub const CANCEL_BONUS_REQUESTS: u64 = 100_000;

/// Once the venue rate-limits an address it permits one action per this interval.
pub const THROTTLE_INTERVAL_MS: u64 = 10_000;

/// A stale `expiresAfter` multiplies the *address-based* cost (only) by this.
pub const STALE_EXPIRES_AFTER_MULTIPLIER: u64 = 5;

/// Paginated info endpoints charge 1 extra weight per this many items returned.
pub const INFO_ITEMS_PER_EXTRA_WEIGHT: u32 = 20;

/// Info endpoints in the cheap tier (weight 2).
const INFO_WEIGHT_2: [&str; 6] = [
    "l2Book",
    "allMids",
    "clearinghouseState",
    "orderStatus",
    "spotClearinghouseState",
    "exchangeStatus",
];

/// `userRole` is uniquely expensive at 60 — 30x a book snapshot. Polling it in a
/// loop is a self-inflicted IP ban, hence the dedicated tier.
const INFO_WEIGHT_60: [&str; 1] = ["userRole"];

/// Everything not otherwise classified.
const INFO_WEIGHT_DEFAULT: u32 = 20;

/// Endpoints whose weight grows with the number of items returned. A full
/// 2000-fill `userFills` therefore costs 20 + 100 = 120, not 20.
const INFO_PAGINATED: [&str; 5] = [
    "historicalOrders",
    "userFills",
    "userFillsByTime",
    "recentTrades",
    "fundingHistory",
];

/// Per-IP weight of one `/exchange` action: `1 + batch_len / 40`.
///
/// Note the asymmetry this creates — batching is nearly free against the IP
/// budget (79 orders cost 2) but linear against the address budget. An empty
/// batch still costs 1: the request itself is what is weighed.
pub fn action_weight(batch_len: u32) -> u32 {
    1 + batch_len / BATCH_WEIGHT_STEP
}

/// Address-based credit cost of one `/exchange` action: one credit per order.
///
/// `stale_expires_after` prices the venue's 5x penalty for an action carrying an
/// `expiresAfter` the venue considers stale. It must be priced *before* the
/// request goes out — discovering it afterwards means the local estimate has
/// already drifted 5x low, which is exactly how a client walks into a 429.
pub fn address_credits(batch_len: u32, stale_expires_after: bool) -> u64 {
    // A zero-length action is still one request against the address limit.
    let base = u64::from(batch_len.max(1));
    if stale_expires_after {
        base.saturating_mul(STALE_EXPIRES_AFTER_MULTIPLIER)
    } else {
        base
    }
}

/// Per-IP weight of a `POST /info` request.
///
/// `items_returned` is the *expected* item count for the paginated endpoints; it
/// is ignored for the others. Callers that cannot predict it should pass their
/// page limit, since over-estimating costs nothing but a little headroom while
/// under-estimating silently overspends the window.
pub fn info_weight(endpoint: &str, items_returned: u32) -> u32 {
    let base = if INFO_WEIGHT_2.contains(&endpoint) {
        2
    } else if INFO_WEIGHT_60.contains(&endpoint) {
        60
    } else {
        INFO_WEIGHT_DEFAULT
    };
    if INFO_PAGINATED.contains(&endpoint) {
        base + items_returned / INFO_ITEMS_PER_EXTRA_WEIGHT
    } else {
        base
    }
}

/// The cancel-side address ceiling for a given default action limit:
/// `min(limit + 100_000, limit * 2)`.
///
/// A fresh address has 10 000 action credits and therefore 20 000 cancel
/// credits; a high-volume address gets a flat +100 000 instead, because doubling
/// a multi-million cap would be absurd.
pub fn cancel_allowance(action_limit: u64) -> u64 {
    action_limit
        .saturating_add(CANCEL_BONUS_REQUESTS)
        .min(action_limit.saturating_mul(2))
}

/// Which side of the ledger a request is on. The distinction is load-bearing:
/// [`Cancel`](ActionKind::Cancel) draws on the larger allowance and is exempt
/// from the place-side reserves and the throttle drip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// A new order, modify, or anything else that *adds* exposure.
    Place,
    /// A cancel — the one operation that always removes exposure.
    Cancel,
}

/// The budget that refused a request. Named separately from the raw limits so an
/// operator woken at 3am learns which knob to turn, not merely "rate limited".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// The per-IP 1200 / 60 s weight window is genuinely full.
    IpWeight,
    /// IP weight remains, but only inside the slice held back for cancels.
    IpPlaceReserve,
    /// Address-based action credits are exhausted; only cancels remain possible.
    AddressActions,
    /// Address credits remain, but only inside the place-side reserve.
    AddressPlaceReserve,
    /// Even the larger `min(limit + 100_000, limit * 2)` cancel ceiling is spent.
    /// The only genuinely alarming refusal in this enum.
    CancelAllowance,
    /// The venue has rate-limited us; one action per 10 s until it recovers.
    Throttled,
    /// A panic poisoned the internal state, so places fail closed.
    PoisonedState,
}

impl fmt::Display for Budget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Budget::IpWeight => "per-IP REST weight window (1200 / 60s)",
            Budget::IpPlaceReserve => "per-IP REST weight place-side reserve",
            Budget::AddressActions => "address-based action credits",
            Budget::AddressPlaceReserve => "address-based place-side reserve",
            Budget::CancelAllowance => "address-based cancel allowance",
            Budget::Throttled => "venue throttle (one action / 10s)",
            Budget::PoisonedState => "governor state poisoned by a panic",
        };
        f.write_str(s)
    }
}

/// What an admitted request was charged, for logging and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cost {
    /// Charged against the per-IP sliding window.
    pub ip_weight: u32,
    /// Charged against the address-based action budget (0 for info requests).
    pub address_credits: u64,
}

/// Why a request was refused, and when an identical one could succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refusal {
    /// The binding constraint.
    pub budget: Budget,
    /// Earliest `now_ms` at which the same request would be admitted.
    ///
    /// `None` when recovery is not a function of time: the address budget only
    /// grows with cumulative traded volume or a `reserveRequestWeight` purchase,
    /// and a poisoned lock never heals. Waiting will not help, so we say so
    /// rather than inventing a retry time the caller would spin on.
    pub retry_at_ms: Option<u64>,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.retry_at_ms {
            Some(t) => write!(f, "{} exhausted; next allowed at {t} ms", self.budget),
            None => write!(
                f,
                "{} exhausted; not time-recoverable (needs cumulative volume or reserved weight)",
                self.budget
            ),
        }
    }
}

impl From<Refusal> for ProviderError {
    fn from(r: Refusal) -> Self {
        ProviderError::RateLimited(r.to_string())
    }
}

/// The governor's verdict. `#[must_use]` because dropping it and sending anyway
/// defeats the entire component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Decision {
    /// Send it. The budgets have already been charged.
    Admitted(Cost),
    /// Do not send it.
    Refused(Refusal),
}

impl Decision {
    pub fn is_admitted(&self) -> bool {
        matches!(self, Decision::Admitted(_))
    }

    /// The refusal detail, or `None` if admitted.
    pub fn refusal(&self) -> Option<Refusal> {
        match self {
            Decision::Refused(r) => Some(*r),
            Decision::Admitted(_) => None,
        }
    }

    /// The charge, or `None` if refused (a refusal never charges anything).
    pub fn cost(&self) -> Option<Cost> {
        match self {
            Decision::Admitted(c) => Some(*c),
            Decision::Refused(_) => None,
        }
    }

    /// Collapse into the crate's error type for `?` in the execution client.
    pub fn into_result(self) -> Result<Cost, ProviderError> {
        match self {
            Decision::Admitted(c) => Ok(c),
            Decision::Refused(r) => Err(r.into()),
        }
    }
}

/// Tunables. The reserves are the whole point; the defaults are chosen to make an
/// unwind affordable without ever touching the cancel bonus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GovernorConfig {
    /// Address credits withheld from placements.
    ///
    /// Default 1000 = one credit per open order at Hyperliquid's default
    /// 1000-order cap, so even the worst case — unwinding every open order with
    /// unbatched cancels — fits inside the reserve, *before* drawing a single
    /// credit from the +100 000 cancel bonus. Belt and braces: the bonus is the
    /// venue's promise, the reserve is ours, and we do not want to find out which
    /// one is wrong while carrying risk.
    pub place_reserve_credits: u64,
    /// Per-IP weight withheld from non-cancel traffic (places *and* info).
    ///
    /// A cancel with plenty of address credits but no IP weight is just as stuck
    /// as one with no credits, so the same reserve logic applies to the IP
    /// window. Default 200 of 1200 (~17%) buys 200 unbatched or 4000 batched
    /// cancels per minute, far more than the open-order cap can require.
    pub place_reserve_ip_weight: u32,
    /// Address action cap assumed before the first `userRateLimit` reading.
    ///
    /// Defaults to the fresh-address buffer, i.e. the most pessimistic value a
    /// real account can have. A high-volume address corrects upward on its first
    /// [`RateGovernor::observe_rate_limit_status`] call; starting optimistic
    /// would instead over-spend until the first correction.
    pub initial_address_cap: u64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            place_reserve_credits: 1_000,
            place_reserve_ip_weight: 200,
            initial_address_cap: INITIAL_ADDRESS_BUFFER,
        }
    }
}

/// A point-in-time view of both budgets, for logging and health endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GovernorSnapshot {
    /// Weight charged inside the trailing [`IP_WINDOW_MS`].
    pub ip_weight_used: u32,
    /// Always [`IP_WEIGHT_LIMIT`]; included so dashboards need no constants.
    pub ip_weight_limit: u32,
    /// Ceiling non-cancel traffic stops at (limit minus the IP reserve).
    pub ip_weight_place_ceiling: u32,
    /// Address-based credits spent — the local mirror of `nRequestsUsed`.
    pub address_used: u64,
    /// Local mirror of `nRequestsCap` (`10_000 + floor(cumVlm)`).
    pub address_cap: u64,
    /// Local mirror of `nRequestsSurplus` (purchased via `reserveRequestWeight`).
    pub address_surplus: u64,
    /// `address_cap + address_surplus`: the venue's effective action limit.
    pub address_limit: u64,
    /// Where placements stop: `address_limit - place_reserve_credits`.
    pub place_ceiling: u64,
    /// Where cancels stop: `min(limit + 100_000, limit * 2)`.
    pub cancel_ceiling: u64,
    /// Whether the venue throttle is currently armed.
    pub throttled: bool,
    /// While throttled, the next instant a place may go out.
    pub throttle_next_ms: Option<u64>,
    /// False if a panic poisoned the state; every counter above is then stale.
    pub healthy: bool,
}

/// Sliding per-IP weight window.
#[derive(Debug, Default)]
struct IpWindow {
    /// `(charged_at_ms, weight)`, oldest first. The 1200-weight cap bounds this
    /// to ~1200 entries — and same-millisecond charges coalesce — so the linear
    /// scans below are cheaper than any index would be.
    entries: VecDeque<(u64, u32)>,
    used: u32,
}

impl IpWindow {
    /// Drop charges that have aged out. A charge at `ts` is in-window while
    /// `now_ms < ts + IP_WINDOW_MS`, which is the same boundary
    /// [`Self::frees_at`] reports, so a refusal's `retry_at_ms` is exact rather
    /// than optimistic by a millisecond.
    fn prune(&mut self, now_ms: u64) {
        while let Some(&(ts, w)) = self.entries.front() {
            if now_ms >= ts.saturating_add(IP_WINDOW_MS) {
                self.used = self.used.saturating_sub(w);
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// When `deficit` weight will have aged out, or `None` if the window can
    /// never free that much (the request is larger than the whole budget).
    fn frees_at(&self, deficit: u32) -> Option<u64> {
        let mut freed = 0u32;
        for &(ts, w) in &self.entries {
            freed = freed.saturating_add(w);
            if freed >= deficit {
                return Some(ts.saturating_add(IP_WINDOW_MS));
            }
        }
        None
    }

    fn charge(&mut self, now_ms: u64, weight: u32) {
        self.used = self.used.saturating_add(weight);
        // Fold into the newest entry when it shares this millisecond — bursts
        // within one ms are the norm. Folding also when `now_ms` moved *backward*
        // keeps `entries` sorted despite a stepping clock; the cost then expires
        // marginally later than strictly necessary, which errs the safe way.
        match self.entries.back_mut() {
            Some((ts, w)) if *ts >= now_ms => *w = w.saturating_add(weight),
            _ => self.entries.push_back((now_ms, weight)),
        }
    }
}

#[derive(Debug)]
struct State {
    ip: IpWindow,
    /// Local mirror of `nRequestsUsed`. Both places and cancels increment it —
    /// the venue has one counter and merely checks it against a higher ceiling
    /// for cancels, so mirroring that shape is what keeps the two ceilings
    /// consistent with each other.
    used: u64,
    cap: u64,
    surplus: u64,
    /// `Some(t)` while throttled: the next instant a place may go out.
    throttle_next_ms: Option<u64>,
}

impl State {
    fn limit(&self) -> u64 {
        self.cap.saturating_add(self.surplus)
    }
}

/// Keeps outbound Hyperliquid traffic inside both budgets while guaranteeing
/// cancel capacity. Share it as `Arc<RateGovernor>`; all methods take `&self`.
#[derive(Debug)]
pub struct RateGovernor {
    config: GovernorConfig,
    state: Mutex<State>,
}

impl RateGovernor {
    /// A governor for a fresh address (10 000 credits) with default reserves.
    pub fn new() -> Self {
        Self::with_config(GovernorConfig::default())
    }

    pub fn with_config(config: GovernorConfig) -> Self {
        Self {
            state: Mutex::new(State {
                ip: IpWindow::default(),
                used: 0,
                cap: config.initial_address_cap,
                surplus: 0,
                throttle_next_ms: None,
            }),
            config,
        }
    }

    /// Guard against being handed nanoseconds where milliseconds are expected.
    ///
    /// This is the one input mistake that turns the governor into a silent
    /// admit-everything no-op rather than a visible error: with a `Nanos` value the IP
    /// window ages out every charge on the first call (`now_ms >= ts + 60_000` always
    /// holds) and the throttle is always satisfied. The crate's ambient timestamp type
    /// *is* `axon_core::Nanos` (i64 nanoseconds) and it is a bare alias, so the compiler
    /// cannot catch the mix-up — hence a runtime assertion in debug builds.
    ///
    /// The bound is deliberately loose: anything past year ~5138 in ms is far likelier
    /// to be ns than a real timestamp.
    fn debug_assert_millis(now_ms: u64) {
        debug_assert!(
            now_ms < 100_000_000_000_000,
            "now_ms={now_ms} looks like nanoseconds, not milliseconds - the governor \
             would silently admit everything"
        );
    }

    /// Ask permission to send an order-adding action of `batch_len` orders.
    ///
    /// Charges both budgets on success. This path can never reach above the
    /// place ceiling, which sits strictly below the venue's action limit, which
    /// in turn sits strictly below the cancel ceiling — that chain of
    /// inequalities *is* the cancel guarantee.
    pub fn try_place(&self, batch_len: u32, now_ms: u64) -> Decision {
        self.try_action(ActionKind::Place, batch_len, false, now_ms)
    }

    /// Ask permission to send a cancel of `batch_len` orders.
    ///
    /// Admitted whenever the cancel allowance and the IP window permit, even
    /// with the place budget at zero and the venue throttle armed.
    pub fn try_cancel(&self, batch_len: u32, now_ms: u64) -> Decision {
        self.try_action(ActionKind::Cancel, batch_len, false, now_ms)
    }

    /// The general form. Prefer [`Self::try_place`] / [`Self::try_cancel`]; reach
    /// for this one only to price a stale `expiresAfter` (5x address cost).
    pub fn try_action(
        &self,
        kind: ActionKind,
        batch_len: u32,
        stale_expires_after: bool,
        now_ms: u64,
    ) -> Decision {
        Self::debug_assert_millis(now_ms);
        let cost = Cost {
            ip_weight: action_weight(batch_len),
            address_credits: address_credits(batch_len, stale_expires_after),
        };

        let mut st = match self.state.lock() {
            Ok(g) => g,
            // A poisoned lock means a panic left the counters unknown. Places
            // fail CLOSED and cancels fail OPEN, because the two failure modes
            // are not remotely symmetric: a wrongly refused place costs us one
            // missed trade, while a wrongly refused cancel strands live exposure
            // we cannot remove. Over-spending the cancel allowance at worst
            // earns a 429 on requests the venue would mostly have honoured
            // anyway — an acceptable price for never being unable to flatten.
            Err(_) => {
                return match kind {
                    ActionKind::Place => Decision::Refused(Refusal {
                        budget: Budget::PoisonedState,
                        retry_at_ms: None,
                    }),
                    ActionKind::Cancel => Decision::Admitted(cost),
                }
            }
        };
        st.ip.prune(now_ms);

        // 1. Venue throttle. It gates places only. Applying the 10 s drip to
        //    cancels would hand back exactly the failure the cancel allowance
        //    exists to prevent, so cancels neither wait on the drip nor re-arm
        //    it (a re-arming cancel storm would starve places indefinitely).
        if kind == ActionKind::Place {
            if let Some(next) = st.throttle_next_ms {
                if now_ms < next {
                    return Decision::Refused(Refusal {
                        budget: Budget::Throttled,
                        retry_at_ms: Some(next),
                    });
                }
            }
        }

        // 2. Address-based credits, checked before IP weight: it is the budget
        //    whose exhaustion is not self-healing, so it is the more useful
        //    thing to name in a refusal.
        let limit = st.limit();
        let address_ceiling = match kind {
            ActionKind::Place => limit.saturating_sub(self.config.place_reserve_credits),
            ActionKind::Cancel => cancel_allowance(limit),
        };
        let after = st.used.saturating_add(cost.address_credits);
        if after > address_ceiling {
            let budget = match kind {
                ActionKind::Cancel => Budget::CancelAllowance,
                ActionKind::Place if after > limit => Budget::AddressActions,
                ActionKind::Place => Budget::AddressPlaceReserve,
            };
            return Decision::Refused(Refusal {
                budget,
                retry_at_ms: None,
            });
        }

        // 3. Per-IP window. Cancels get the full 1200; everything else stops at
        //    the reserve.
        let ip_ceiling = match kind {
            ActionKind::Place => {
                IP_WEIGHT_LIMIT.saturating_sub(self.config.place_reserve_ip_weight)
            }
            ActionKind::Cancel => IP_WEIGHT_LIMIT,
        };
        if let Some(refusal) = check_ip(&st.ip, cost.ip_weight, ip_ceiling) {
            return Decision::Refused(refusal);
        }

        st.ip.charge(now_ms, cost.ip_weight);
        st.used = st.used.saturating_add(cost.address_credits);
        // Each place that slips through the drip re-arms it for another 10 s.
        if kind == ActionKind::Place && st.throttle_next_ms.is_some() {
            st.throttle_next_ms = Some(now_ms.saturating_add(THROTTLE_INTERVAL_MS));
        }
        Decision::Admitted(cost)
    }

    /// Ask permission to send a `POST /info` request.
    ///
    /// Info costs IP weight only — the address-based limit applies to actions,
    /// not queries — but it is still non-cancel traffic, so it respects the IP
    /// place-side reserve. A `userFills` poll must not be the reason a cancel
    /// cannot get out. `items_returned` is the expected item count and matters
    /// only for the paginated endpoints (see [`info_weight`]).
    pub fn try_info(&self, endpoint: &str, items_returned: u32, now_ms: u64) -> Decision {
        let cost = Cost {
            ip_weight: info_weight(endpoint, items_returned),
            address_credits: 0,
        };
        let mut st = match self.state.lock() {
            Ok(g) => g,
            // Info is never the thing that removes exposure: fail closed.
            Err(_) => {
                return Decision::Refused(Refusal {
                    budget: Budget::PoisonedState,
                    retry_at_ms: None,
                })
            }
        };
        st.ip.prune(now_ms);
        let ceiling = IP_WEIGHT_LIMIT.saturating_sub(self.config.place_reserve_ip_weight);
        if let Some(refusal) = check_ip(&st.ip, cost.ip_weight, ceiling) {
            return Decision::Refused(refusal);
        }
        st.ip.charge(now_ms, cost.ip_weight);
        Decision::Admitted(cost)
    }

    /// Fold in a `userRateLimit` reading — `{"cumVlm", "nRequestsUsed",
    /// "nRequestsCap", "nRequestsSurplus"}` — so the address budget tracks the
    /// venue's accounting instead of drifting on local estimates.
    ///
    /// The reading wins in **both** directions. Drifting low (another process on
    /// the same address, or actions we never priced) walks us into a 429; drifting
    /// high needlessly stops trading. Note that the reading lags anything still
    /// in flight, so prefer to poll it when the order pipe is quiet.
    ///
    /// It is also how the throttle clears: a reading with headroom lifts it, a
    /// reading at or above the limit (re-)arms it. Absent any reading the throttle
    /// persists, which is the safe direction — cancels flow regardless.
    pub fn observe_rate_limit_status(&self, used: u64, cap: u64, surplus: u64, now_ms: u64) {
        // A poisoned lock leaves the estimate as-is; nothing here is worth a panic.
        if let Ok(mut st) = self.state.lock() {
            st.used = used;
            st.cap = cap;
            st.surplus = surplus;
            st.throttle_next_ms = if used < st.limit() {
                None
            } else {
                Some(now_ms.saturating_add(THROTTLE_INTERVAL_MS))
            };
        }
    }

    /// Record that the venue just rate-limited us (HTTP 429 or the equivalent
    /// error body). Arms the one-action-per-10 s drip; the 429 itself is treated
    /// as having consumed the current slot, so the next place waits a full
    /// interval rather than firing immediately into a closed door.
    pub fn observe_rate_limited(&self, now_ms: u64) {
        if let Ok(mut st) = self.state.lock() {
            st.throttle_next_ms = Some(now_ms.saturating_add(THROTTLE_INTERVAL_MS));
        }
    }

    /// Both budgets as of `now_ms` (prunes the IP window as a side effect).
    pub fn snapshot(&self, now_ms: u64) -> GovernorSnapshot {
        let ip_place_ceiling = IP_WEIGHT_LIMIT.saturating_sub(self.config.place_reserve_ip_weight);
        let mut st = match self.state.lock() {
            Ok(g) => g,
            // Report the poisoning rather than inventing numbers: a dashboard
            // showing "exhausted" would send an operator hunting the wrong bug.
            Err(_) => {
                return GovernorSnapshot {
                    ip_weight_used: 0,
                    ip_weight_limit: IP_WEIGHT_LIMIT,
                    ip_weight_place_ceiling: ip_place_ceiling,
                    address_used: 0,
                    address_cap: 0,
                    address_surplus: 0,
                    address_limit: 0,
                    place_ceiling: 0,
                    cancel_ceiling: 0,
                    throttled: false,
                    throttle_next_ms: None,
                    healthy: false,
                }
            }
        };
        st.ip.prune(now_ms);
        let limit = st.limit();
        GovernorSnapshot {
            ip_weight_used: st.ip.used,
            ip_weight_limit: IP_WEIGHT_LIMIT,
            ip_weight_place_ceiling: ip_place_ceiling,
            address_used: st.used,
            address_cap: st.cap,
            address_surplus: st.surplus,
            address_limit: limit,
            place_ceiling: limit.saturating_sub(self.config.place_reserve_credits),
            cancel_ceiling: cancel_allowance(limit),
            throttled: st.throttle_next_ms.is_some(),
            throttle_next_ms: st.throttle_next_ms,
            healthy: true,
        }
    }

    pub fn config(&self) -> &GovernorConfig {
        &self.config
    }
}

impl Default for RateGovernor {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared IP-window admission check. `None` means "fits".
fn check_ip(ip: &IpWindow, weight: u32, ceiling: u32) -> Option<Refusal> {
    let after = ip.used.saturating_add(weight);
    if after <= ceiling {
        return None;
    }
    // Distinguish "the window is full" from "only the cancel reserve is left":
    // the first is a pacing problem, the second means we are already unwinding.
    let budget = if after > IP_WEIGHT_LIMIT {
        Budget::IpWeight
    } else {
        Budget::IpPlaceReserve
    };
    Some(Refusal {
        budget,
        retry_at_ms: ip.frees_at(after - ceiling),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn gov() -> RateGovernor {
        RateGovernor::new()
    }

    #[test]
    fn action_weight_steps_every_forty_orders() {
        assert_eq!(action_weight(1), 1);
        assert_eq!(action_weight(39), 1);
        assert_eq!(action_weight(40), 2);
        assert_eq!(action_weight(79), 2);
        assert_eq!(action_weight(80), 3);
    }

    #[test]
    fn info_weight_covers_all_three_tiers_and_the_surcharge() {
        assert_eq!(info_weight("l2Book", 0), 2);
        assert_eq!(info_weight("clearinghouseState", 0), 2);
        assert_eq!(info_weight("userRole", 0), 60);
        assert_eq!(info_weight("candleSnapshot", 0), 20); // unlisted -> default tier
        assert_eq!(info_weight("userFills", 0), 20);
        assert_eq!(info_weight("userFills", 2000), 120); // 20 + 2000/20
        assert_eq!(info_weight("recentTrades", 100), 25);
        // The surcharge applies only to the paginated endpoints.
        assert_eq!(info_weight("l2Book", 2000), 2);
    }

    #[test]
    fn ip_window_is_genuinely_sliding() {
        let g = gov();
        let t0 = 1_000;
        // Unbatched cancels cost 1 weight each and get the whole 1200.
        for _ in 0..IP_WEIGHT_LIMIT {
            assert!(g.try_cancel(1, t0).is_admitted());
        }
        assert_eq!(g.snapshot(t0).ip_weight_used, IP_WEIGHT_LIMIT);

        let r = g.try_cancel(1, t0).refusal().expect("window is full");
        assert_eq!(r.budget, Budget::IpWeight);
        assert_eq!(r.retry_at_ms, Some(t0 + IP_WINDOW_MS));

        // One ms before the oldest charge ages out: still refused.
        assert!(g.try_cancel(1, t0 + IP_WINDOW_MS - 1).refusal().is_some());
        // The instant it does: recovered.
        assert!(g.try_cancel(1, t0 + IP_WINDOW_MS).is_admitted());
        assert_eq!(g.snapshot(t0 + IP_WINDOW_MS).ip_weight_used, 1);
    }

    #[test]
    fn a_batch_is_cheap_for_ip_and_expensive_for_the_address() {
        let g = gov();
        let cost = g.try_place(20, 5_000).cost().expect("admitted");
        assert_eq!(cost.ip_weight, 1); // 1 + 20/40
        assert_eq!(cost.address_credits, 20); // one credit per order
        let s = g.snapshot(5_000);
        assert_eq!(s.ip_weight_used, 1);
        assert_eq!(s.address_used, 20);
    }

    /// The reason this file exists.
    #[test]
    fn cancels_survive_a_fully_exhausted_place_budget() {
        let g = gov();
        let t = 1_000;
        // Drain the place side with batches of 20 so the *address* budget binds
        // (450 batches = 9000 credits but only 450 IP weight).
        let mut placed = 0u64;
        while g.try_place(20, t).is_admitted() {
            placed += 20;
        }
        let s = g.snapshot(t);
        assert_eq!(placed, s.address_used);
        assert_eq!(s.address_used, s.place_ceiling);
        assert_eq!(
            g.try_place(1, t)
                .refusal()
                .expect("place budget spent")
                .budget,
            Budget::AddressPlaceReserve
        );

        // The invariant: cancels are unaffected.
        assert!(g.try_cancel(20, t).is_admitted());
        assert!(g.try_cancel(1, t).is_admitted());
    }

    #[test]
    fn cancel_capacity_runs_all_the_way_to_the_venue_ceiling() {
        let g = gov();
        let t = 1_000;
        while g.try_place(20, t).is_admitted() {}
        while g.try_cancel(20, t).is_admitted() {}
        let s = g.snapshot(t);
        assert_eq!(s.cancel_ceiling, 20_000); // min(10k + 100k, 10k * 2)
        assert_eq!(s.address_used, s.cancel_ceiling);
        assert!(
            s.ip_weight_used < IP_WEIGHT_LIMIT,
            "IP was not the binding budget"
        );
        let r = g
            .try_cancel(20, t)
            .refusal()
            .expect("cancel ceiling reached");
        assert_eq!(r.budget, Budget::CancelAllowance);
    }

    #[test]
    fn the_place_reserve_stops_placements_before_the_address_limit() {
        let g = gov();
        let t = 1_000;
        while g.try_place(20, t).is_admitted() {}
        let s = g.snapshot(t);
        assert!(
            s.address_used < s.address_cap,
            "placements must stop short of the venue limit"
        );
        assert_eq!(
            s.address_cap - s.address_used,
            GovernorConfig::default().place_reserve_credits
        );
    }

    #[test]
    fn observe_rate_limit_status_corrects_drift_in_both_directions() {
        let g = gov();
        let t = 1_000;
        assert!(g.try_place(20, t).is_admitted());
        assert_eq!(g.snapshot(t).address_used, 20);

        // Upward: the venue says we have spent far more than we thought. Use the
        // real `userRateLimit` body from the docs.
        g.observe_rate_limit_status(3_053, 5_770_209, 0, t);
        let s = g.snapshot(t);
        assert_eq!(s.address_used, 3_053);
        assert_eq!(s.address_cap, 5_770_209); // 10_000 + floor(5_760_209.98)
        assert_eq!(s.cancel_ceiling, 5_870_209); // flat +100k, not doubled

        // Downward: a fresher reading with fewer used requests must relax again,
        // and purchased surplus must lift the effective limit.
        g.observe_rate_limit_status(10, 5_770_209, 2_000, t);
        let s = g.snapshot(t);
        assert_eq!(s.address_used, 10);
        assert_eq!(s.address_surplus, 2_000);
        assert_eq!(s.address_limit, 5_772_209);
    }

    #[test]
    fn the_throttled_state_admits_exactly_one_action_per_ten_seconds() {
        let g = gov();
        g.observe_rate_limited(0);
        // The 429 consumed the current slot.
        let r = g.try_place(1, 0).refusal().expect("throttled");
        assert_eq!(r.budget, Budget::Throttled);
        assert_eq!(r.retry_at_ms, Some(THROTTLE_INTERVAL_MS));
        assert!(g.try_place(1, 9_999).refusal().is_some());

        assert!(g.try_place(1, 10_000).is_admitted());
        // …and exactly one: the slot is immediately re-armed.
        let r = g.try_place(1, 10_001).refusal().expect("still throttled");
        assert_eq!(r.budget, Budget::Throttled);
        assert_eq!(r.retry_at_ms, Some(20_000));
        assert!(g.try_place(1, 19_999).refusal().is_some());
        assert!(g.try_place(1, 20_000).is_admitted());
        assert_eq!(g.snapshot(20_000).address_used, 2); // exactly two got out
    }

    #[test]
    fn a_throttle_never_gates_a_cancel() {
        let g = gov();
        g.observe_rate_limited(0);
        assert!(g.try_place(1, 0).refusal().is_some());
        assert!(g.try_cancel(1, 0).is_admitted());
        assert!(g.try_cancel(1, 0).is_admitted());
        // Cancels do not re-arm the drip either.
        assert_eq!(g.snapshot(0).throttle_next_ms, Some(THROTTLE_INTERVAL_MS));
    }

    #[test]
    fn a_status_reading_lifts_or_re_arms_the_throttle() {
        let g = gov();
        g.observe_rate_limited(0);
        assert!(g.try_place(1, 0).refusal().is_some());

        // Headroom -> lifted.
        g.observe_rate_limit_status(10, INITIAL_ADDRESS_BUFFER, 0, 0);
        assert!(!g.snapshot(0).throttled);
        assert!(g.try_place(1, 0).is_admitted());

        // At the limit -> re-armed, and places are refused for the throttle
        // rather than for the credit budget.
        g.observe_rate_limit_status(INITIAL_ADDRESS_BUFFER, INITIAL_ADDRESS_BUFFER, 0, 5_000);
        let s = g.snapshot(5_000);
        assert!(s.throttled);
        assert_eq!(s.throttle_next_ms, Some(15_000));
        assert_eq!(
            g.try_place(1, 5_000).refusal().expect("throttled").budget,
            Budget::Throttled
        );
        // And even at the venue's action limit, a cancel still goes out.
        assert!(g.try_cancel(1, 5_000).is_admitted());
    }

    #[test]
    fn a_refusal_names_the_binding_budget_and_a_next_allowed_time() {
        let g = gov();
        let t = 500;
        for _ in 0..IP_WEIGHT_LIMIT {
            assert!(g.try_cancel(1, t).is_admitted());
        }
        let r = g.try_cancel(1, t).refusal().expect("window is full");
        assert_eq!(r.budget, Budget::IpWeight);
        assert_eq!(r.retry_at_ms, Some(60_500));
        let msg = r.to_string();
        assert!(msg.contains("per-IP"), "{msg}");
        assert!(msg.contains("60500"), "{msg}");

        // And the non-time-recoverable case says so instead of guessing.
        let g = gov();
        while g.try_place(20, t).is_admitted() {}
        let r = g.try_place(20, t).refusal().expect("reserve reached");
        assert_eq!(r.retry_at_ms, None);
        assert!(r.to_string().contains("not time-recoverable"), "{r}");

        // Refusals convert into the crate's error type with the detail intact.
        let err: ProviderError = r.into();
        assert!(matches!(err, ProviderError::RateLimited(_)));
    }

    #[test]
    fn info_traffic_cannot_eat_the_cancel_reserve() {
        let g = gov();
        let t = 1_000;
        // `userRole` costs 60; 16 fit under the 1000 place ceiling, the 17th
        // (1020) would dip into the reserve.
        let mut spent = 0;
        while g.try_info("userRole", 0, t).is_admitted() {
            spent += 60;
        }
        assert_eq!(spent, 960);
        let r = g
            .try_info("userRole", 0, t)
            .refusal()
            .expect("reserve reached");
        assert_eq!(r.budget, Budget::IpPlaceReserve);
        assert_eq!(r.retry_at_ms, Some(t + IP_WINDOW_MS));
        // Info spends no address credits.
        assert_eq!(g.snapshot(t).address_used, 0);
        // The reserve did its job.
        assert!(g.try_cancel(20, t).is_admitted());
    }

    #[test]
    fn a_stale_expires_after_costs_five_times_the_address_amount() {
        assert_eq!(address_credits(4, false), 4);
        assert_eq!(address_credits(4, true), 20);
        let g = gov();
        let t = 1_000;
        let cost = g
            .try_action(ActionKind::Place, 4, true, t)
            .cost()
            .expect("admitted");
        assert_eq!(cost.address_credits, 20);
        assert_eq!(cost.ip_weight, 1); // the IP price is unchanged
        assert_eq!(g.snapshot(t).address_used, 20);
    }

    #[test]
    fn a_refusal_charges_nothing() {
        let g = gov();
        let t = 1_000;
        g.observe_rate_limited(t);
        assert!(g.try_place(20, t).refusal().is_some());
        let s = g.snapshot(t);
        assert_eq!(s.ip_weight_used, 0);
        assert_eq!(s.address_used, 0);
    }

    #[test]
    fn a_poisoned_lock_fails_closed_for_places_and_open_for_cancels() {
        let g = Arc::new(gov());
        let g2 = Arc::clone(&g);
        // Panic while holding the guard. The message is expected test noise.
        let _ = std::thread::spawn(move || {
            let _guard = g2.state.lock().expect("lock is fresh");
            panic!("deliberately poisoning the governor");
        })
        .join();

        assert_eq!(
            g.try_place(1, 0)
                .refusal()
                .expect("places fail closed")
                .budget,
            Budget::PoisonedState
        );
        assert!(g.try_info("l2Book", 0, 0).refusal().is_some());
        // Exposure must still be removable even with the counters unknown.
        assert!(g.try_cancel(1, 0).is_admitted());
        assert!(!g.snapshot(0).healthy);
    }
}
