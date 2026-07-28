//! The runtime configuration schema (`docs/06-strategy-contract.md`) — everything a
//! session needs to start, and nothing it must not have.
//!
//! Three properties this file is responsible for:
//!
//! - **No secrets, ever.** There is no field here that can hold a private key, and
//!   [`RuntimeConfig::load`] *refuses* a file containing one (see
//!   [`ConfigError::SecretInConfig`]). Config files get committed, copied into
//!   tickets and pasted into chat; the signing key comes from `AXON_HL_SECRET_KEY`
//!   and nowhere else (ADR-0009). An account **address** is public data and does
//!   live here — it is the one thing the venue cannot derive for us under the
//!   agent-wallet model.
//! - **Refuse an unsafe combination rather than run it.** [`RuntimeConfig::validate`]
//!   is not a formality: a sandbox session pointed at mainnet, or a dead-man's switch
//!   with a lead shorter than its own re-arm interval, are configurations that look
//!   fine and are not survivable.
//! - **Round-trippable.** The default config serializes to TOML and parses back
//!   identically, so `--dump-config` produces a file that is valid input.

use std::path::Path;

use axon_core::Decimal;
use axon_provider_hyperliquid::governor::{GovernorConfig, IP_WEIGHT_LIMIT};
use axon_provider_hyperliquid::{
    ExchangeClient, HyperliquidMarketData, SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY,
    SCHEDULE_CANCEL_MIN_LEAD_MS,
};
use axon_providers::Feed;
use axon_risk::RiskLimits;
use serde::{Deserialize, Serialize};

use crate::mdring::MdWritePolicy;

/// Env var pointing at a config file, so a deployment can set it once instead of
/// threading `--config` through every wrapper script.
pub const ENV_CONFIG_PATH: &str = "AXON_CONFIG";

/// Second gate in front of real money. A config file alone is not enough to reach
/// mainnet: a typo in `network` must not be able to trade.
pub const ENV_ALLOW_MAINNET: &str = "AXON_ALLOW_MAINNET";

/// Milliseconds in a UTC day — the window the venue counts `scheduleCancel`
/// triggers and the address action budget over.
const DAY_MS: u64 = 24 * 60 * 60 * 1_000;

/// Floor on the dead-man's-switch re-arm interval.
///
/// Matched to the venue's own 5 s minimum *lead* because a re-arm cadence faster than
/// the shortest deadline the venue will accept buys nothing — and costs plenty: each
/// re-arm is a rate-limited action, so 5 s spends 17 280 address credits per UTC day,
/// more than the entire 10 000-credit lifetime buffer of an address that has not
/// traded (see [`RuntimeConfig::dms_actions_per_day`]).
pub const MIN_REARM_INTERVAL_MS: u64 = SCHEDULE_CANCEL_MIN_LEAD_MS;

/// A re-arm interval must fit into the lead at least this many times.
///
/// Three, so two consecutive failed re-arms are survivable: one missed beat is a
/// transient the retry absorbs, and the third attempt still lands with a margin. A
/// tighter ratio means a single 500 from the venue can let protection lapse, and a
/// switch that *fires* costs one of the venue's
/// [`SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY`] daily triggers — spend them and the
/// account is unprotected for the rest of the UTC day.
pub const MIN_LEAD_TO_REARM_RATIO: u64 = 3;

/// One core, three modes — differ only by which adapters are plugged in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    /// Offline. No sockets, no keys, no tokio: the deterministic core over a
    /// canned event stream. This is the default, so a bare `cargo run --bin axon`
    /// can never reach a venue.
    Backtest,
    /// Live market data + testnet execution.
    Sandbox,
    /// Live market data + mainnet execution. Real money.
    Live,
}

impl Environment {
    /// Whether this mode opens sockets and needs a key.
    pub fn is_live_wiring(&self) -> bool {
        matches!(self, Environment::Sandbox | Environment::Live)
    }
}

/// Which of a venue's two deployments to talk to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Network {
    Testnet,
    Mainnet,
}

/// The venue endpoint set. URLs are overridable so a session can be pointed at a
/// proxy or a capture harness without a rebuild; `None` means "the network default".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VenueConfig {
    pub name: String,
    pub network: Network,
    /// The **master account** address, lower-case `0x` hex. Public data.
    ///
    /// Required for a live session and not derivable: under the agent-wallet model
    /// the signing key belongs to the agent, and querying `/info` with an agent
    /// address returns empty results and no error — a mistake that looks exactly
    /// like an idle account.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ws_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchange_url: Option<String>,
}

/// How the core loop and its bus are sized and paced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    /// In-flight events the bounded bus holds before producers feel backpressure.
    pub bus_capacity: usize,
    /// How long the core thread parks when the bus is empty.
    ///
    /// It is a poll rather than a blocking receive because the loop also has to
    /// service the status line and notice a shutdown; a blocking receive on a silent
    /// feed would do neither. Sub-millisecond latency is a Phase-8 concern, and by
    /// then the core wants a proper timed receive anyway.
    pub core_poll_us: u64,
    /// How often the status line is printed.
    pub status_interval_ms: u64,
    /// Age at which a mark price stops being usable by the risk gate. See
    /// [`axon_execution::marks`] for why this is not optional.
    pub mark_max_age_ms: u64,
    /// Feeds subscribed for every configured symbol.
    ///
    /// `Ticker` earns its place by carrying the venue's own mark price, which is the
    /// number margin and liquidation are computed against; without it the risk gate
    /// falls back to the book mid, which is a different quantity.
    pub feeds: Vec<Feed>,
}

/// The venue-side dead-man's switch — the only protection that survives this process
/// dying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Turning this off is only defensible for a read-only session that will never
    /// place an order. A trading session without it has no protection against its own
    /// process being killed.
    pub dead_mans_switch: bool,
    /// How far ahead each `scheduleCancel` deadline is set.
    pub lead_ms: u64,
    /// How often the deadline is pushed forward. Must divide into the lead at least
    /// [`MIN_LEAD_TO_REARM_RATIO`] times.
    pub rearm_interval_ms: u64,
}

/// The `POST /info` reconciliation poll — the only way to learn about resting orders
/// after a restart, because Hyperliquid's `orderUpdates` channel never snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileConfig {
    pub interval_ms: u64,
    /// How long an order we submitted is allowed to be absent from the venue's
    /// snapshot before that absence is treated as meaningful.
    ///
    /// Without it every freshly placed order looks like a discrepancy: the REST read
    /// and the submit race, and read-your-writes is not guaranteed.
    pub grace_ms: u64,
    /// Read `userRateLimit` every N cycles to correct the governor's local estimate.
    /// Not every cycle: at weight 20 it is ten times a book snapshot.
    pub rate_limit_every: u32,
}

/// What the session's money view alarms on ([`crate::pnl`]).
///
/// Both thresholds default to `0`, which reads as **no bound declared** — the same
/// reading `intent.max_order_age_ms` gives zero, and for the same reason: a default
/// that fires is a default nobody chose. Neither is a risk control. A breach here
/// raises a warning on the status line and changes nothing about what the session
/// will place; a limit that pulls the trading switch is a gate, belongs with the other
/// gates in `axon-execution`, and needs its own argument.
///
/// The two are in *quote* units (USDC here) rather than in percent, because the
/// account this project runs against is shared and its balance is not a scale anyone
/// chose — a percentage of it would mean something different every week.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PnlConfig {
    /// Warn once the session's own bottom line is worse than `-max_session_loss`.
    ///
    /// Measured against [`PnlSnapshot::net`](crate::pnl::PnlSnapshot::net), so it is
    /// silent while any held position is unpriced — which is correct and worth saying
    /// out loud: the unpriced position raises its own, louder warning, and a loss
    /// alarm computed off a partial book would be the more comforting of the two.
    #[serde(default)]
    pub max_session_loss: Decimal,
    /// Warn once our accounting and the venue's `accountValue` disagree by more than
    /// this about what the session did.
    ///
    /// Expected to be non-zero on any perp session: funding is not a fill, and no
    /// fill-derived accounting can see it. So this is a *drift rate* detector, not a
    /// correctness check — set it above an hour or two of funding on the size being
    /// traded, and it catches the thing that matters instead, which is a fill nobody
    /// here ever saw.
    #[serde(default)]
    pub equity_drift_alarm: Decimal,
}

/// Declared latency ceilings, in milliseconds ([`crate::latency`]).
///
/// Every field defaults to `0` = **no ceiling declared**, and an undeclared stage is
/// still measured. That asymmetry is the point: the first thing anyone setting a
/// budget needs is what the session actually does, and a book that went dark without
/// a config would make declaring a ceiling require guessing one first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LatencyConfig {
    /// Ceiling on the gap between the observation a decision answers — an m1 bar's own
    /// close — and the decision itself. `Signal::ts_cause` → `Signal::ts_event`.
    ///
    /// **The largest number in this system, and it could not have a ceiling before schema
    /// version 3 because it was not on the wire.** Measured 951 / 12 051 / 111 475 ms over
    /// 57 live m1 bars on 2026-07-27, visible only in the producer's own transcript. A
    /// record carrying only the moment the producer decided makes a decision one second
    /// after a bar and one two minutes after it indistinguishable — and a number nobody
    /// can measure cannot regress.
    ///
    /// A producer that states no cause is not measured here, so this is silent on any
    /// strategy that is not driven by a timed observation.
    #[serde(default)]
    pub cause_to_decision_ms: u64,
    /// How old a decision may be when the core plans it.
    ///
    /// Distinct from [`IntentConfig::max_signal_age_ms`], which *refuses* a record.
    /// This one only reports. Setting it below that ceiling is the useful
    /// configuration — it makes the session say "these would have been refused if the
    /// window were tighter" before anyone tightens it.
    #[serde(default)]
    pub signal_age_ms: u64,
    /// The venue round trip on one placement.
    #[serde(default)]
    pub submit_ack_ms: u64,
    /// Decision to order-at-the-venue, end to end. The number an operator means when
    /// they ask how fast the loop is.
    #[serde(default)]
    pub decision_to_ack_ms: u64,
    /// Percentage of samples that may breach before the status line says so. `0`
    /// silences the warning and keeps every measurement.
    #[serde(default)]
    pub breach_warn_pct: u64,
}

/// Rate-governor reserves. The defaults are the governor's own; they are surfaced
/// here because the right size depends on how many orders a strategy keeps resting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GovernorBudgets {
    pub place_reserve_credits: u64,
    pub place_reserve_ip_weight: u32,
    pub initial_address_cap: u64,
}

impl Default for GovernorBudgets {
    /// Mirrors the governor's own defaults rather than restating numbers: the reserve
    /// sizes are reasoned about in `axon-provider-hyperliquid`, and a second copy here
    /// would be the one that goes stale.
    fn default() -> Self {
        let g = GovernorConfig::default();
        Self {
            place_reserve_credits: g.place_reserve_credits,
            place_reserve_ip_weight: g.place_reserve_ip_weight,
            initial_address_cap: g.initial_address_cap,
        }
    }
}

/// Shared-memory boundary config (the Py→Rust signal ring).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IpcConfig {
    /// Ring file path. On Linux, put this under `/dev/shm` for tmpfs (RAM-backed).
    pub signal_ring_path: String,
    /// Ring capacity in records (power of two).
    pub capacity: u64,
}

/// The Rust→Python market-data ring (ADR-0012): what the core publishes so Python can
/// compute features on the same book the executing core saw.
///
/// Off by default, which is the opposite of [`IntentConfig::enabled`] and for a reason
/// that is specific to this direction: this end **creates** the ring file, and
/// `MdProducer::create` truncates whatever is already there. A publisher that started
/// unasked would silently truncate a ring some other process is mid-read on, which is
/// the single failure an SPSC transport has no defence against. Publishing is therefore
/// something an operator turns on, and the status line reports it once it is on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MdRingConfig {
    pub enabled: bool,
    /// Ring file path. `/dev/shm` on Linux, for the same tmpfs reason as the signal
    /// ring. Must not be the signal ring's path — see [`RuntimeConfig::validate`].
    pub path: String,
    /// Ring capacity in records (power of two). At 128 bytes a slice, the default
    /// 4096 is 512 KiB and roughly 0.4 s of headroom at ADR-0012's 10k updates/s:
    /// enough to absorb a Python GC pause, and short enough that a genuinely stalled
    /// consumer shows up as drops rather than as latency nobody measures.
    pub capacity: u64,
    /// When a slice is written. See [`MdWritePolicy`].
    pub policy: MdWritePolicy,
}

impl Default for MdRingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: default_md_ring_path(),
            capacity: 4096,
            policy: MdWritePolicy::default(),
        }
    }
}

/// Session capture (ADR-0018): the event log and signal log a session writes so that
/// it can be replayed later through the same code that produced it.
///
/// Off by default, like [`MdRingConfig`] and for the same first reason — this end
/// *creates* files — plus a second one that is specific to it: a recording is unbounded
/// work on the deterministic path's behalf, and a session should only ever be doing
/// that because somebody asked. Turning it on is what makes a soak run, a shadow
/// session or a parity reference produce an artifact instead of only a status line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub enabled: bool,
    /// Where the event log goes. The signal log is its sibling, with `.signals.jsonl`
    /// in place of the extension — the convention `axon-replay`'s `replay_log` looks
    /// for, so a captured session replays by naming one path.
    ///
    /// A real filesystem, not `/dev/shm`: the two rings are RAM-backed because they are
    /// transports, and a capture is the one thing here that is supposed to outlive the
    /// process.
    pub path: String,
    /// Whether to record the signals the strategy adapter read.
    ///
    /// On by default, because without it a replay can re-observe the session but not
    /// re-decide it: the reader would admit nothing and every strategy column in a
    /// golden run would be zero. Off is for a market-data-only capture, which is a
    /// legitimate thing to want and a bad thing to get by accident.
    pub signals: bool,
    /// Depth of the core→writer hand-off queue, in records.
    ///
    /// A hiccup buffer, not a backlog: it exists to absorb one page-cache flush, and
    /// sizing it for a genuinely slow disk would only mean the recording dies later
    /// with more memory held. Filling it stops the recording (see [`crate::capture`]).
    pub queue_capacity: usize,
    /// Hard ceiling on the recording, in bytes. `0` disables it.
    ///
    /// **This number used to mean something else, and the old reason is the trap.** It
    /// was set where the *artifact* stopped being usable, because `EventLog` loaded a
    /// log whole — event-time ordering cannot begin until the last record has been seen
    /// (ADR-0018) — so a log too large to hold in memory was a log nobody could replay.
    /// ADR-0027 made the reader streaming, and `--order as-captured` is now constant in
    /// memory over a log of any size. A cap chosen against the old constraint would
    /// stop a soak at a few hours for a reason that no longer exists, and the tape ends
    /// exactly where the interesting part of a long run begins.
    ///
    /// What it guards now is the **disk**: an unbounded recording on a box whose volume
    /// also holds the venue's logs takes the session down with it. So it is set where a
    /// filesystem starts to care rather than where a reader does, and an operator who
    /// has measured their disk can say `0`. There is deliberately no rotation — see
    /// [`crate::capture`] for why segments would each replay a different session.
    pub max_bytes: u64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            path: "data/captures/axon-session.jsonl".to_string(),
            signals: true,
            queue_capacity: 16_384,
            // 8 GiB of JSONL is tens of millions of events: a multi-day soak, which is
            // the shape of run this cap now has to survive rather than truncate. The
            // number is a disk guard, not a reader limit (see the field's docs), and it
            // is deliberately still finite — a recording that fills a volume takes the
            // trading session with it, and that trade is never worth an extra hour of
            // tape.
            max_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

/// The intent source: how the runtime drains the signal ring, plans against it, and
/// hands the result to the submit pipeline (ADR-0020).
///
/// Every default here is chosen so that turning the source *on* — which it is by
/// default — cannot make an unconfigured session trade by accident. A ring that does
/// not exist degrades; a record older than [`IntentConfig::max_signal_age_ms`] is
/// refused; and an instrument whose price the risk gate has already expired is one
/// the planner will not price an order against either.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentConfig {
    /// Whether the runtime reads signals at all.
    ///
    /// On by default. A runtime whose intent source is off looks *exactly* like a
    /// healthy session that nobody is sending signals to — and a session that
    /// believes it is trading and is not is the worst outcome available here.
    pub enabled: bool,
    /// How often the core drains the ring.
    ///
    /// A cadence rather than every loop iteration because the core thread's first
    /// job is market data: on a busy feed the loop spins with no park at all, and a
    /// ring check on every pass competes with the fan-out for the same thread. One
    /// millisecond is two orders of magnitude inside a Hyperliquid block (~0.2 s), so
    /// it costs no realizable edge.
    pub drain_interval_ms: u64,
    /// Records taken off the ring per pass, so a producer burst cannot starve
    /// market-data processing in the same thread.
    pub max_per_drain: usize,
    /// The operator's hard ceiling on a signal's age, in milliseconds
    /// ([`axon_strategy::ReaderConfig::max_age_ms`]). A record's own `ttl_ms` can
    /// only ever be *shorter* than this: the operator answers for the fills.
    pub max_signal_age_ms: u32,
    /// Depth of the core→edge intent queue.
    ///
    /// Only ever holds one pass's worth, because the core will not plan a second
    /// batch while the first is unsubmitted; the capacity is headroom for a
    /// many-symbol pass, not a buffer.
    pub queue_capacity: usize,
    /// How often the edge drains that queue. It polls rather than blocking so the
    /// task can also hear a shutdown.
    pub submit_poll_ms: u64,
    /// How often to retry opening a signal ring that is not there yet.
    ///
    /// Python may legitimately start after Rust. Retrying on a cadence is what turns
    /// "the file does not exist" into a degraded state the session recovers from
    /// instead of a startup failure that needs an operator.
    pub attach_retry_ms: u64,
    /// [`axon_strategy::PlannerConfig::taker_slippage_bps`].
    pub taker_slippage_bps: u32,
    /// [`axon_strategy::PlannerConfig::min_order_qty`] — the dust floor that stops a
    /// target differing by a rounding residue from re-sending an order every signal.
    pub min_order_qty: Decimal,
    /// [`axon_strategy::PlannerConfig::noop_band_bps`] — how far the order we would
    /// place may differ in size from the one already resting before we pay queue
    /// position to replace it.
    ///
    /// `min_order_qty` only reaches the case where the *delta* is dust; this reaches
    /// the case where the delta is real and the target barely moved. Off by default:
    /// the band is a bounded, deliberate position error, and one that appeared in a
    /// deployment because a default changed would be an error nobody chose.
    ///
    /// Defaulted on deserialize, like [`RuntimeConfig::md_ring`] and
    /// [`RuntimeConfig::capture`] and for the same reason one step down: a field added
    /// to a table that already exists in every deployed config file makes **every one
    /// of those files stop parsing**, so upgrading the binary takes the session down at
    /// startup with `missing field` — a message about TOML, for a change about queue
    /// position. `--dump-config` still emits it, so a round trip is still complete.
    #[serde(default = "default_noop_band_bps")]
    pub noop_band_bps: u32,
    /// [`axon_strategy::PlannerConfig::max_order_age_ms`] — how long a resting order
    /// may keep its place before it must be re-derived from a current decision.
    ///
    /// **Not [`Self::max_signal_age_ms`], and this is the standing confusion here.**
    /// That one refuses a stale *record* before the planner sees it. This one refuses
    /// to keep a stale *order*. Nothing in the runtime pulled a resting order on age
    /// before it existed, and `ttl_ms` on the wire never did either.
    ///
    /// Defaulted on deserialize for the same reason as [`Self::noop_band_bps`], and the
    /// consequence is different and worth stating: this default is **on**, so an
    /// existing config file that never mentioned it starts pulling orders older than a
    /// minute. That is the intended direction — it is the only thing bounding the two
    /// changes that widened the leave-it-resting exception — but it is a behaviour
    /// change an operator gets without editing anything, and the startup banner is
    /// where they will see it.
    #[serde(default = "default_max_order_age_ms")]
    pub max_order_age_ms: u32,
    /// How often the pass looks for resting orders that have outlived
    /// [`Self::max_order_age_ms`] with no signal to supersede them (ADR-0031), in
    /// milliseconds of **event time**.
    ///
    /// A cadence of its own rather than every pass, and the reason is the tracker's
    /// lock: the sweeper is the only thing in the pass that reads order state when
    /// *nothing arrived*, so at `drain_interval_ms` it would take that lock and walk
    /// the open orders once per millisecond of event time, on the deterministic core
    /// thread, to answer a question about a sixty-second bound. A second of slack on
    /// that bound is 1.7% of it and three orders of magnitude above a Hyperliquid
    /// block.
    ///
    /// **This does not turn the sweeper off, and nothing here should.** Setting
    /// [`Self::max_order_age_ms`] to `0` does — it is the operator saying they set no
    /// bound, which is the same reading the planner gives it. One number, one meaning,
    /// rather than two switches that can disagree about whether an order has a lifetime.
    #[serde(default = "default_sweep_interval_ms")]
    pub sweep_interval_ms: u64,
    /// How many times the pass may re-quote **one** target the strategy still holds
    /// but has no working order for (ADR-0031 amended, ADR-0036).
    ///
    /// A target position is idempotent, so a strategy that has not changed its mind
    /// correctly says nothing; the sweeper cancels orders no signal speaks for. The
    /// composition leaves a position open with no working order and nothing that will
    /// ever speak for it — observed live for about twelve minutes on 2026-07-26.
    ///
    /// The budget is what keeps the repair from undoing the sweeper. The sweeper's
    /// subject is a producer that has *stopped*, and an unbounded re-quote would hand a
    /// dead strategy an immortal quote. After this many, the session stops placing and
    /// starts saying `UNQUOTED` instead — which is the one thing the live run could not
    /// do.
    ///
    /// **`0` disables the re-quote**, which is exactly the behaviour of every session
    /// before this field existed. The default is not zero: a defect that is fixed only
    /// for operators who read the release notes is not fixed.
    #[serde(default = "default_max_requotes")]
    pub max_requotes: u32,
}

/// The three `serde` defaults above, taken from [`IntentConfig::default`] rather than
/// restated: a second copy of a number is the copy that goes stale, and these decide
/// whether a resting order survives and how long after its ceiling it does.
fn default_noop_band_bps() -> u32 {
    IntentConfig::default().noop_band_bps
}

fn default_max_order_age_ms() -> u32 {
    IntentConfig::default().max_order_age_ms
}

fn default_max_requotes() -> u32 {
    IntentConfig::default().max_requotes
}

fn default_sweep_interval_ms() -> u64 {
    IntentConfig::default().sweep_interval_ms
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            drain_interval_ms: 1,
            max_per_drain: 32,
            // The reader's own default, restated here because this is where an
            // operator changes it (ADR-0014 §1: ~0.2 s blocks, so two seconds is
            // several blocks of slack and still far short of "the market has moved").
            max_signal_age_ms: 2_000,
            queue_capacity: 64,
            submit_poll_ms: 2,
            attach_retry_ms: 1_000,
            taker_slippage_bps: 50,
            min_order_qty: Decimal::ZERO,
            noop_band_bps: 0,
            // A minute: three orders of magnitude above a Hyperliquid block (~0.2 s),
            // so a maker quote keeps its queue position through any realistic re-quote
            // cycle — and bounded, so nothing a previous incarnation of this process
            // left resting outlives a minute of trading.
            max_order_age_ms: 60_000,
            // A second. The sweeper is a backstop for a producer that has stopped
            // speaking, and it cannot become relevant until a signal has been absent
            // for `max_order_age_ms` anyway — so a second of granularity on a
            // sixty-second bound costs nothing and keeps the tracker's lock off the
            // core thread's millisecond cadence.
            sweep_interval_ms: 1_000,
            // Three. Not zero, because a defect fixed only for operators who read the
            // release notes is not fixed; and not large, because every re-quote past
            // the first is a session insisting on a target from a producer that has
            // gone quiet. At `max_order_age_ms` = 60 s, three re-quotes is at most
            // three minutes of a dead strategy's last opinion still being worked
            // toward, after which the position is *reported* unquoted rather than
            // silently held.
            max_requotes: 3,
        }
    }
}

impl IntentConfig {
    pub fn reader_config(&self) -> axon_strategy::ReaderConfig {
        axon_strategy::ReaderConfig {
            max_age_ms: self.max_signal_age_ms,
        }
    }

    pub fn planner_config(&self) -> axon_strategy::PlannerConfig {
        axon_strategy::PlannerConfig {
            taker_slippage_bps: self.taker_slippage_bps,
            min_order_qty: self.min_order_qty,
            noop_band_bps: self.noop_band_bps,
            max_order_age_ms: self.max_order_age_ms,
        }
    }
}

// ── many strategies, one account (ADR-0038) ──────────────────────────────────

/// What to do with a producer's standing target once it has stopped speaking.
///
/// A local serde mirror of [`axon_strategy::OnSilence`] rather than `serde` derives on
/// that type, for the reason [`RiskConfig`] is a mirror of `RiskLimits`: the wire format
/// of a config file is this crate's problem, and the strategy crate stays free of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OnSilenceMode {
    /// Keep it. The default and the behaviour of every session before this existed.
    #[default]
    Hold,
    /// Drive that producer's share to zero. A trading decision — see
    /// [`axon_strategy::OnSilence::Flat`].
    Flat,
}

impl From<OnSilenceMode> for axon_strategy::OnSilence {
    fn from(m: OnSilenceMode) -> Self {
        match m {
            OnSilenceMode::Hold => axon_strategy::OnSilence::Hold,
            OnSilenceMode::Flat => axon_strategy::OnSilence::Flat,
        }
    }
}

/// What happens when two producers claim one instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OverlapMode {
    /// Refuse the second claim, and refuse the *config* that declares the overlap.
    ///
    /// The default because two producers on one instrument is far more often a
    /// copy-pasted config than a decision, and netting them silently composes two
    /// strategies' risk into a position neither author sized.
    #[default]
    Exclusive,
    /// Add the claims and work toward the sum. One declared word, so the deliberate case
    /// is possible and the accidental one is not.
    Net,
}

impl From<OverlapMode> for axon_strategy::Overlap {
    fn from(m: OverlapMode) -> Self {
        match m {
            OverlapMode::Exclusive => axon_strategy::Overlap::Exclusive,
            OverlapMode::Net => axon_strategy::Overlap::Net,
        }
    }
}

/// One signal producer: a strategy process with a ring of its own.
///
/// **One ring per producer, rather than one ring with a producer id on the record.** The
/// ring is SPSC by construction (ADR-0006) and `seq` is what proves nothing was lost, so
/// interleaving two producers on one ring would make gap detection meaningless and put
/// two writers on a structure that has exactly one. It also means a producer can crash
/// and restart without touching another's sequence, and that "which strategy" is a
/// property of config rather than a field that would have to re-cut a full 64-byte
/// record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerConfig {
    /// What the status line calls it. Distinct across producers, because this is the
    /// name an operator reads at 03:00 and two of them would be worse than none.
    pub name: String,
    /// The ring this producer writes into.
    pub signal_ring: String,
    /// The instruments this producer is allowed to speak for.
    ///
    /// Empty means the whole session universe. Declared rather than inferred because it
    /// is what lets [`RuntimeConfig::validate`] catch an overlap at startup instead of
    /// letting the runtime discover it on the first pass that trades.
    #[serde(default)]
    pub symbols: Vec<String>,
    #[serde(default)]
    pub on_silence: OnSilenceMode,
    /// How long this producer may say nothing before it is called silent, in
    /// milliseconds of event time. `0` is "never", which is the only safe reading of a
    /// field nobody wrote — the same rule `ttl_ms` and `max_order_age_ms` turn on.
    #[serde(default)]
    pub silence_ms: u64,
    /// The most gross notional this producer's own claims may add up to. `0` is
    /// unbounded.
    ///
    /// An allocation, not a risk gate: a claim past it is **scaled down** rather than
    /// refused, so a strategy that asks for more than its share converges to its share
    /// instead of emitting a target that is refused forever. The gate that cannot be
    /// talked past is `[portfolio]`, enforced in `axon_execution::GuardedClient`.
    #[serde(default)]
    pub max_gross_notional: Decimal,
}

/// Bounds and policy across every instrument and every strategy at once
/// ([`axon_risk::portfolio`]).
///
/// Defaulted, and the default declares nothing — so a config written before this table
/// existed keeps running exactly as it did, and upgrading the binary cannot stop a
/// session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortfolioConfig {
    /// Ceiling on `Σ |qty_i| · mark_i`. `0` declares no bound.
    #[serde(default)]
    pub max_gross_notional: Decimal,
    /// Ceiling on `|Σ qty_i · mark_i|`. `0` declares no bound.
    #[serde(default)]
    pub max_net_notional: Decimal,
    /// How many instruments may carry exposure at once. `0` is unbounded.
    #[serde(default)]
    pub max_symbols: u32,
    #[serde(default)]
    pub overlap: OverlapMode,
    /// Whether the pass scales its targets to fit [`Self::max_gross_notional`].
    ///
    /// On by default, and that default cannot fire on its own: it does nothing at all
    /// unless a gross bound is declared. When one is, the alternative is worse than it
    /// looks — an unscaled session emits targets the guard refuses on every pass, which
    /// presents as orders that keep failing rather than as a bound that is binding, and
    /// leaves a position permanently short of a target it can never reach. Scaling makes
    /// the session converge to the largest book it is allowed to hold. The guard still
    /// enforces the bound, so this is convergence rather than the guarantee.
    #[serde(default = "default_scale_to_fit")]
    pub scale_to_fit: bool,
}

fn default_scale_to_fit() -> bool {
    true
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            max_gross_notional: Decimal::ZERO,
            max_net_notional: Decimal::ZERO,
            max_symbols: 0,
            overlap: OverlapMode::Exclusive,
            scale_to_fit: true,
        }
    }
}

impl PortfolioConfig {
    pub fn to_limits(&self) -> axon_risk::PortfolioLimits {
        axon_risk::PortfolioLimits {
            max_gross_notional: self.max_gross_notional,
            max_net_notional: self.max_net_notional,
            max_symbols: self.max_symbols,
        }
    }

    pub fn is_declared(&self) -> bool {
        self.to_limits().is_declared()
    }
}

/// One producer, with every default filled in and an id assigned.
///
/// The runtime never reads [`ProducerConfig`] directly: a session with no
/// `[[strategy.producer]]` entries has exactly one producer — the `[ipc]` ring, named
/// after the strategy — and making that the *same* type as a declared one is what keeps
/// the single-producer path from being a second code path
/// ([`RuntimeConfig::producers`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProducer {
    pub id: axon_strategy::StrategyId,
    pub name: String,
    pub ring_path: String,
    /// The instruments this producer may speak for; empty means the session universe.
    pub symbols: Vec<String>,
    pub on_silence: OnSilenceMode,
    pub silence_ms: u64,
    pub max_gross_notional: Decimal,
}

impl ResolvedProducer {
    pub fn policy(&self) -> axon_strategy::StrategyPolicy {
        axon_strategy::StrategyPolicy {
            id: self.id,
            on_silence: self.on_silence.into(),
            silence_ns: (self.silence_ms as i64).saturating_mul(1_000_000),
        }
    }
}

/// Whether a venue coin name and a config instrument name mean the same instrument.
///
/// One rule, shared: [`RuntimeConfig::coins`] takes everything before the first `-`,
/// uppercased, so `"BTC-PERP"`, `"btc"` and `"BTC"` are one instrument. A producer's
/// declared universe is matched against the resolved symbol list with the *same* rule
/// rather than a second one — two answers to "is this the same coin" is how a producer
/// ends up correctly configured and silently scoped to nothing.
pub fn coin_matches(coin: &str, config_name: &str) -> bool {
    let want = config_name
        .split('-')
        .next()
        .unwrap_or(config_name)
        .trim()
        .to_uppercase();
    coin.trim().to_uppercase() == want
}

/// Reference to a versioned model artifact in the registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub registry_id: String,
    pub version: u32,
}

/// Risk limits as configured; converted to [`RiskLimits`] for the hot path.
///
/// The first three bound a **size** and the last three bound a **loss**, and until
/// 2026-07-27 only the first three existed. That gap is the one Phase 6 left open on
/// purpose and the one this table now closes: `max_position`, `max_notional` and
/// `max_order_qty` all answer "how big may this get" and none of them answers "how much
/// may this cost", so a strategy that is quietly wrong stays inside every one of them
/// for as long as anyone lets it run.
///
/// The loss bounds are **not** `[pnl]`. That table warns and never halts, deliberately
/// ([ADR-0036] §3), and the two must stay distinguishable: an operator raising an alarm
/// threshold to stop a noisy status line must not be able to widen a kill switch by
/// accident. These are in `[strategy.risk]` because that is where the gates are.
///
/// [ADR-0036]: https://docs.rs/axon
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position: Decimal,
    pub max_notional: Decimal,
    pub max_order_qty: Decimal,
    /// Halt **new exposure** once this session's own bottom line is worse than
    /// `-max_session_loss`. A magnitude in quote units; `0` is no bound declared.
    ///
    /// "Halt new exposure" is precise and is the whole design: orders that strictly
    /// reduce the position keep going out, because a session that has lost more than it
    /// was allowed to wants to *get out*, and getting out is an order. See
    /// [`axon_execution::LossLimiter`].
    ///
    /// Defaulted on deserialize so every config written before this existed keeps
    /// parsing — and defaults to *off*, so upgrading the binary cannot stop a session.
    #[serde(default)]
    pub max_session_loss: Decimal,
    /// The same halt, on the **UTC day's** change in the venue's own `accountValue`.
    ///
    /// Independent of the session bound and computed from the other accounting, for the
    /// reason [ADR-0036] keeps the two apart: a crash-restart loop resets ours and
    /// cannot reset the venue's, so a session bound alone is spent once per restart.
    #[serde(default)]
    pub max_daily_loss: Decimal,
    /// Where the day's equity baseline is persisted ([`crate::daybook`]).
    ///
    /// Required whenever `max_daily_loss` is set, and [`RuntimeConfig::validate`]
    /// refuses the combination without it. A daily bound whose baseline lives only in
    /// the process is a session bound wearing a day's name — every number on the status
    /// line would be right and the guarantee would not be the one the operator
    /// configured, which is the quiet kind of wrong.
    #[serde(default)]
    pub daily_state_path: Option<String>,
}

impl RiskConfig {
    pub fn to_limits(&self) -> RiskLimits {
        RiskLimits {
            max_position: self.max_position,
            max_notional: self.max_notional,
            max_order_qty: self.max_order_qty,
        }
    }

    /// The money bounds, for [`axon_execution::LossLimiter`].
    pub fn to_loss_limits(&self) -> axon_execution::LossLimits {
        axon_execution::LossLimits {
            session: self.max_session_loss,
            day: self.max_daily_loss,
        }
    }
}

/// A strategy's serializable configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub name: String,
    pub version: u32,
    pub symbols: Vec<String>,
    pub model_ref: ModelRef,
    pub risk: RiskConfig,
    /// The signal producers this session reads, as `[[strategy.producer]]` tables.
    ///
    /// **Empty is not "no producers"** — it is the single-producer session every config
    /// written before ADR-0038 describes, whose one ring is `[ipc].signal_ring_path` and
    /// whose name is [`Self::name`]. See [`RuntimeConfig::producers`], which is the only
    /// thing that should read this field: the runtime is written against the resolved
    /// list so there is no branch anywhere between one strategy and several.
    ///
    /// Last in the struct because it is an array of tables, and TOML serialization emits
    /// fields in declaration order — a table array written before `[strategy.risk]`
    /// would produce a file that round-trips differently from the one it was written
    /// from.
    #[serde(default, rename = "producer", skip_serializing_if = "Vec::is_empty")]
    pub producers: Vec<ProducerConfig>,
}

/// Top-level runtime config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub environment: Environment,
    pub venue: VenueConfig,
    pub session: SessionConfig,
    pub safety: SafetyConfig,
    pub reconcile: ReconcileConfig,
    pub governor: GovernorBudgets,
    pub ipc: IpcConfig,
    #[serde(default)]
    pub intent: IntentConfig,
    /// Defaulted so a config file written before the publisher existed still loads —
    /// and defaults to *off*, so loading one cannot start writing a ring file.
    #[serde(default)]
    pub md_ring: MdRingConfig,
    /// Defaulted for the same two reasons as `md_ring`: an older file keeps loading, and
    /// upgrading the binary cannot start writing a log nobody asked for.
    #[serde(default)]
    pub capture: CaptureConfig,
    /// Defaulted, and the default declares no alarm at all — so an existing config
    /// gains the money view on the status line and gains nothing that can fire.
    #[serde(default)]
    pub pnl: PnlConfig,
    /// Defaulted, and the default declares no ceiling — measured everywhere, breached
    /// nowhere, for the reason [`LatencyConfig`] gives.
    #[serde(default)]
    pub latency: LatencyConfig,
    /// Bounds across every instrument and strategy at once (ADR-0038). Defaulted, and
    /// the default declares nothing, so an existing config gains no gate it did not ask
    /// for.
    #[serde(default)]
    pub portfolio: PortfolioConfig,
    pub strategy: StrategyConfig,
}

/// Why a config was rejected.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse config {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error(
        "config key {key:?} looks like a secret. Keys belong in the {env} environment \
         variable, never in a file that gets committed or pasted"
    )]
    SecretInConfig { key: String, env: &'static str },
    #[error("invalid config: {0}")]
    Invalid(String),
}

/// Default ring path: `/dev/shm` (tmpfs) on Linux — the prod target (ADR-0007) —
/// and a temp-dir path on Windows, which has no `/dev/shm`.
fn default_ring_path() -> String {
    ring_path("axon-signal.ring")
}

/// The market-data ring's default path. A *different* file from the signal ring: SPSC
/// means one writer and one reader per ring, and a bidirectional ring would be neither.
fn default_md_ring_path() -> String {
    ring_path("axon-md.ring")
}

fn ring_path(name: &str) -> String {
    if cfg!(windows) {
        std::env::temp_dir()
            .join(name)
            .to_string_lossy()
            .into_owned()
    } else {
        format!("/dev/shm/{name}")
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            environment: Environment::Backtest,
            venue: VenueConfig {
                name: axon_provider_hyperliquid::VENUE.to_string(),
                network: Network::Testnet,
                account: None,
                ws_url: None,
                info_url: None,
                exchange_url: None,
            },
            session: SessionConfig {
                bus_capacity: 4096,
                core_poll_us: 500,
                status_interval_ms: 5_000,
                mark_max_age_ms: 10_000,
                feeds: vec![Feed::Bbo, Feed::L2Book, Feed::Ticker],
            },
            safety: SafetyConfig {
                dead_mans_switch: true,
                // 60 s of protection re-armed every 20 s: two consecutive failures are
                // survivable, and the switch spends 4 320 address credits a day rather
                // than the 17 280 a 5 s cadence would.
                lead_ms: 60_000,
                rearm_interval_ms: 20_000,
            },
            reconcile: ReconcileConfig {
                interval_ms: 15_000,
                grace_ms: 5_000,
                rate_limit_every: 10,
            },
            governor: GovernorBudgets::default(),
            ipc: IpcConfig {
                signal_ring_path: default_ring_path(),
                capacity: 1024,
            },
            intent: IntentConfig::default(),
            md_ring: MdRingConfig::default(),
            capture: CaptureConfig::default(),
            pnl: PnlConfig::default(),
            latency: LatencyConfig::default(),
            portfolio: PortfolioConfig::default(),
            strategy: StrategyConfig {
                name: "example-mean-reversion".to_string(),
                version: 1,
                symbols: vec!["BTC-PERP".to_string(), "ETH-PERP".to_string()],
                model_ref: ModelRef {
                    registry_id: "example-mean-reversion".to_string(),
                    version: 1,
                },
                risk: RiskConfig {
                    max_position: Decimal::from(10),
                    max_notional: Decimal::from(100_000),
                    max_order_qty: Decimal::from(5),
                    // No bound declared, for the reason every other default here gives:
                    // a kill switch nobody chose is one that takes a session down on an
                    // upgrade. The size caps have defaults because a session with no
                    // size cap is unbounded in a way nothing else catches; a session
                    // with no loss cap is exactly what every session before this was.
                    max_session_loss: Decimal::ZERO,
                    max_daily_loss: Decimal::ZERO,
                    daily_state_path: None,
                },
                // No `[[strategy.producer]]` tables: the default session is the
                // single-producer one, and `producers()` resolves that to the `[ipc]`
                // ring. Writing one here would put a second ring path in every dumped
                // config for a session that has one.
                producers: Vec::new(),
            },
        }
    }
}

impl RuntimeConfig {
    /// Load and validate a config file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let name = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: name.clone(),
            source,
        })?;
        Self::from_toml(&text, &name)
    }

    /// Parse and validate TOML text. `origin` names the source in error messages.
    pub fn from_toml(text: &str, origin: &str) -> Result<Self, ConfigError> {
        let value: toml::Value = toml::from_str(text).map_err(|source| ConfigError::Parse {
            path: origin.to_string(),
            source,
        })?;
        reject_secrets(&value)?;
        let cfg: RuntimeConfig =
            value
                .try_into()
                .map_err(|source: toml::de::Error| ConfigError::Parse {
                    path: origin.to_string(),
                    source,
                })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// The config a session should run with: an explicit path, else
    /// [`ENV_CONFIG_PATH`], else the default (which is offline).
    ///
    /// A *missing* explicit path is an error, never a silent fallback: being handed
    /// the offline default when you asked for a live config is the kind of surprise
    /// that is only noticed after an hour of "why is nothing trading".
    pub fn resolve(explicit: Option<&Path>) -> Result<Self, ConfigError> {
        if let Some(p) = explicit {
            return Self::load(p);
        }
        match std::env::var(ENV_CONFIG_PATH) {
            Ok(p) if !p.trim().is_empty() => Self::load(p.trim()),
            _ => {
                let cfg = Self::default();
                cfg.validate()?;
                Ok(cfg)
            }
        }
    }

    /// Every signal producer this session reads, with the single-producer case resolved
    /// into the same shape as a declared one.
    ///
    /// **This is the only function that knows a session might have one producer rather
    /// than several**, and that is the whole point of it. Everything downstream — the
    /// intent source, the target book, the status line, the pre-start banner — is
    /// written against a list, so there is no `if single { … } else { … }` anywhere for
    /// the two paths to drift apart in. The single-producer case is the one that has run
    /// at a venue and the multi-producer case is the one that has not, and a branch would
    /// mean the tested path is not the shipped one.
    ///
    /// Ids are the list index, so they are stable across a run and printable without an
    /// allocation. They are *not* stable across a config edit, which is why the status
    /// line prints [`ResolvedProducer::name`] and never the id.
    pub fn producers(&self) -> Vec<ResolvedProducer> {
        if self.strategy.producers.is_empty() {
            return vec![ResolvedProducer {
                id: axon_strategy::StrategyId::new(0),
                name: self.strategy.name.clone(),
                ring_path: self.ipc.signal_ring_path.clone(),
                symbols: Vec::new(),
                on_silence: OnSilenceMode::Hold,
                silence_ms: 0,
                max_gross_notional: Decimal::ZERO,
            }];
        }
        self.strategy
            .producers
            .iter()
            .enumerate()
            .map(|(i, p)| ResolvedProducer {
                id: axon_strategy::StrategyId::new(i as u16),
                name: p.name.clone(),
                ring_path: p.signal_ring.clone(),
                symbols: p.symbols.clone(),
                on_silence: p.on_silence,
                silence_ms: p.silence_ms,
                max_gross_notional: p.max_gross_notional,
            })
            .collect()
    }

    /// Reject configurations that are internally inconsistent or unsafe.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: String| Err(ConfigError::Invalid(m));

        if self.session.bus_capacity == 0 {
            return bad("session.bus_capacity must be > 0".into());
        }
        if self.session.core_poll_us == 0 {
            return bad("session.core_poll_us must be > 0 (0 spins a core at 100%)".into());
        }
        if self.session.status_interval_ms == 0 {
            return bad("session.status_interval_ms must be > 0".into());
        }
        if self.session.mark_max_age_ms == 0 {
            return bad(
                "session.mark_max_age_ms must be > 0; a zero window expires every price \
                 instantly and the risk gate then refuses every order"
                    .into(),
            );
        }
        if self.session.feeds.is_empty() {
            return bad("session.feeds is empty - the session would receive no market data".into());
        }
        if self.strategy.symbols.is_empty() {
            return bad("strategy.symbols is empty - nothing to subscribe to".into());
        }

        // Both P&L alarms are *magnitudes*. A negative `max_session_loss` is the one
        // typo here that fails silently in the dangerous direction: the threshold is
        // compared against a net figure that is itself usually negative, so `-5`
        // written where `5` was meant is an alarm that is on from the first tick and
        // is therefore ignored by the time it means something.
        if self.pnl.max_session_loss < Decimal::ZERO {
            return bad(format!(
                "pnl.max_session_loss {} is negative; it is a magnitude - the session \
                 warns once its own net is worse than MINUS this. 0 declares no bound",
                self.pnl.max_session_loss
            ));
        }
        if self.pnl.equity_drift_alarm < Decimal::ZERO {
            return bad(format!(
                "pnl.equity_drift_alarm {} is negative; it is a magnitude, and drift is \
                 compared on its absolute value because either side may be the larger",
                self.pnl.equity_drift_alarm
            ));
        }
        if self.latency.breach_warn_pct > 100 {
            return bad(format!(
                "latency.breach_warn_pct {} is above 100; it is a percentage of samples, \
                 and a threshold no run can reach is a warning that never fires",
                self.latency.breach_warn_pct
            ));
        }

        // The loss bounds. Same sign trap as the alarms above and a worse consequence:
        // a negative one here is not a warning that is always on, it is a session that
        // never places an order — and the symptom is a strategy that emits and never
        // trades, which reads exactly like a warmup that has not finished.
        if let Err(e) = self.strategy.risk.to_loss_limits().validate() {
            return bad(format!("strategy.risk.{e}"));
        }
        // A daily bound with nowhere to persist its baseline is a session bound wearing
        // a day's name. Refused rather than degraded, because the degradation is
        // invisible: every number on the status line is right and the guarantee is not
        // the one that was configured. The failure it exists to stop is a crash-restart
        // loop, and that is exactly when a process-local baseline resets.
        if self.strategy.risk.max_daily_loss > Decimal::ZERO
            && self
                .strategy
                .risk
                .daily_state_path
                .as_deref()
                .is_none_or(str::is_empty)
        {
            return bad(
                "strategy.risk.max_daily_loss is set with no strategy.risk.daily_state_path; \
                 a daily bound whose baseline dies with the process is spent once per \
                 restart, and a crash-restart loop is how a losing session restarts"
                    .into(),
            );
        }

        // The dead-man's switch: both bounds are load-bearing.
        if self.safety.lead_ms < SCHEDULE_CANCEL_MIN_LEAD_MS {
            return bad(format!(
                "safety.lead_ms {} is below the venue minimum {SCHEDULE_CANCEL_MIN_LEAD_MS}; \
                 the venue would reject every re-arm and the session would run unprotected",
                self.safety.lead_ms
            ));
        }
        if self.safety.rearm_interval_ms < MIN_REARM_INTERVAL_MS {
            return bad(format!(
                "safety.rearm_interval_ms {} is below {MIN_REARM_INTERVAL_MS}; a faster \
                 cadence buys no protection and spends {} address credits a day",
                self.safety.rearm_interval_ms,
                DAY_MS / self.safety.rearm_interval_ms.max(1)
            ));
        }
        if self.safety.rearm_interval_ms * MIN_LEAD_TO_REARM_RATIO > self.safety.lead_ms {
            return bad(format!(
                "safety.rearm_interval_ms {} must fit into safety.lead_ms {} at least \
                 {MIN_LEAD_TO_REARM_RATIO} times, or a single failed re-arm lets the \
                 protection lapse and fires the switch (only \
                 {SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY} firings are honoured per day)",
                self.safety.rearm_interval_ms, self.safety.lead_ms
            ));
        }

        if self.reconcile.interval_ms < 1_000 {
            return bad(
                "reconcile.interval_ms below 1000 would spend the per-IP weight window on \
                 polling that cancels may need"
                    .into(),
            );
        }
        if self.reconcile.rate_limit_every == 0 {
            return bad("reconcile.rate_limit_every must be > 0".into());
        }
        if self.governor.place_reserve_ip_weight >= IP_WEIGHT_LIMIT {
            return bad(format!(
                "governor.place_reserve_ip_weight {} must be below the venue's {IP_WEIGHT_LIMIT} \
                 per-IP limit, or no placement can ever be admitted",
                self.governor.place_reserve_ip_weight
            ));
        }

        // The intent source. Each of these is a value that would leave the session
        // *looking* healthy while it could not possibly place an order — the one
        // failure this component exists to make impossible.
        if self.intent.enabled {
            if self.intent.drain_interval_ms == 0 {
                return bad(
                    "intent.drain_interval_ms must be > 0 (0 drains the ring on \
                            every core iteration and starves the market-data fan-out)"
                        .into(),
                );
            }
            if self.intent.max_per_drain == 0 {
                return bad(
                    "intent.max_per_drain must be > 0; at 0 the reader takes nothing off \
                     the ring, forever, and the session reports OK while Python fills it"
                        .into(),
                );
            }
            if self.intent.max_signal_age_ms == 0 {
                return bad(
                    "intent.max_signal_age_ms must be > 0; a zero ceiling expires every \
                     signal on arrival and the session silently trades nothing"
                        .into(),
                );
            }
            if self.intent.queue_capacity == 0 {
                return bad(
                    "intent.queue_capacity must be > 0; every planned intent would be \
                     dropped between the core and the venue"
                        .into(),
                );
            }
            if self.intent.submit_poll_ms == 0 || self.intent.attach_retry_ms == 0 {
                return bad(
                    "intent.submit_poll_ms and intent.attach_retry_ms must be > 0 (0 spins \
                     an edge task at 100%)"
                        .into(),
                );
            }
            if self.intent.sweep_interval_ms == 0 {
                return bad(
                    "intent.sweep_interval_ms must be > 0 (0 takes the tracker's read \
                     lock and walks every open order on every core iteration); to run \
                     no sweeper at all, set intent.max_order_age_ms = 0, which is the \
                     one number that says an order has no lifetime"
                        .into(),
                );
            }
            if self.intent.min_order_qty < Decimal::ZERO {
                return bad("intent.min_order_qty must not be negative".into());
            }
            // The §9 shape: a config that leaves a session looking healthy while it
            // cannot act. A band of 100% forgives every size difference, so once one
            // order is resting the strategy's target can never reach the venue again —
            // and every pass reports a perfectly ordinary "already working".
            if self.intent.noop_band_bps >= 10_000 {
                return bad(
                    "intent.noop_band_bps must be < 10000 (100%); at or above it every \
                     size difference is forgiven and a resting order is never replaced"
                        .into(),
                );
            }
            if self.ipc.signal_ring_path.trim().is_empty() {
                return bad(
                    "ipc.signal_ring_path is empty but intent.enabled is true - the intent \
                     source would have nowhere to read from"
                        .into(),
                );
            }
            // Checked here rather than discovered at `Consumer::open`: a ring the
            // runtime cannot open degrades into "no signals", which reads as an idle
            // strategy rather than as the config error it is.
            if self.ipc.capacity == 0 || !self.ipc.capacity.is_power_of_two() {
                return bad(format!(
                    "ipc.capacity {} must be a power of two; the ring header refuses \
                     anything else and the intent source would run permanently detached",
                    self.ipc.capacity
                ));
            }
            // The producers, and the four ways a multi-strategy config is wrong in a way
            // that leaves the session looking healthy (ADR-0038).
            let producers = self.producers();
            for (i, p) in producers.iter().enumerate() {
                if p.name.trim().is_empty() {
                    return bad(format!(
                        "strategy.producer[{i}] has no name; the name is what the status \
                         line calls it, and an unnamed producer is one nobody can be told \
                         has gone silent"
                    ));
                }
                if p.ring_path.trim().is_empty() {
                    return bad(format!(
                        "strategy.producer[{i}] ({}) has an empty signal_ring - it would \
                         run permanently detached while the session reported OK",
                        p.name
                    ));
                }
                if let Some(other) = producers[..i].iter().find(|q| q.name == p.name) {
                    return bad(format!(
                        "two producers are both named {:?}; the status line, the silence \
                         warning and the allocation are all keyed on the name",
                        other.name
                    ));
                }
                // The worst of the four. Two producers on one ring is two consumers of an
                // SPSC queue reading from one writer: each takes a share of the records
                // and neither can tell that from a producer with nothing to say — the same
                // hazard ADR-0029 §5 names for the bar ring, arriving on the signal side.
                if let Some(other) = producers[..i]
                    .iter()
                    .find(|q| q.ring_path.trim() == p.ring_path.trim())
                {
                    return bad(format!(
                        "producers {:?} and {:?} both read {:?}. An SPSC ring has one \
                         consumer: two readers do not share it, they steal from it, and \
                         each would see a share of the records with no way to tell that \
                         from a quiet strategy",
                        other.name, p.name, p.ring_path
                    ));
                }
                for s in &p.symbols {
                    if !self.strategy.symbols.iter().any(|u| u == s) {
                        return bad(format!(
                            "producer {:?} declares {s:?}, which is not in strategy.symbols; \
                             the session subscribes to no market data for it, so every \
                             target it sent would be planned against no book",
                            p.name
                        ));
                    }
                }
                if p.max_gross_notional < Decimal::ZERO {
                    return bad(format!(
                        "producer {:?} has a negative max_gross_notional; it is a magnitude \
                         bounding a sum of magnitudes, so a negative value allocates nothing",
                        p.name
                    ));
                }
                if p.on_silence == OnSilenceMode::Flat && p.silence_ms == 0 {
                    return bad(format!(
                        "producer {:?} sets on_silence = \"flat\" with silence_ms = 0. \
                         Zero means 'never call it silent' on this field, as it does on \
                         ttl_ms and max_order_age_ms - so the policy would never fire and \
                         the operator who asked for it would not find out",
                        p.name
                    ));
                }
            }
            // An overlap declared in the config is caught here rather than on the first
            // pass that trades. Netting is correct and is opt-in: two producers pointed at
            // one instrument is far more often a copy-paste than a decision, and the
            // accident composes two strategies' risk into a position neither author sized.
            if self.portfolio.overlap == OverlapMode::Exclusive {
                for (i, p) in producers.iter().enumerate() {
                    for q in &producers[..i] {
                        if let Some(s) = p.symbols.iter().find(|s| q.symbols.contains(s)) {
                            return bad(format!(
                                "producers {:?} and {:?} both declare {s:?}, and \
                                 portfolio.overlap is \"exclusive\" - so the second \
                                 claim would be refused on every pass. Set \
                                 portfolio.overlap = \"net\" to have the two targets \
                                 added, which is a decision about combined risk and not \
                                 a formality",
                                q.name, p.name
                            ));
                        }
                    }
                }
                // Two producers that each declare *nothing* both mean "the whole
                // universe", which is the same overlap written a shorter way.
                let open: Vec<&ResolvedProducer> =
                    producers.iter().filter(|p| p.symbols.is_empty()).collect();
                if producers.len() > 1 && open.len() > 1 {
                    return bad(format!(
                        "producers {:?} and {:?} declare no symbols, which means the whole \
                         of strategy.symbols for each of them - the same overlap as naming \
                         the instruments twice. Give each producer its own symbols, or set \
                         portfolio.overlap = \"net\"",
                        open[0].name, open[1].name
                    ));
                }
            }

            // Before Phase 4 a session with no dead-man's switch was defensible because
            // nothing in the runtime could place an order. That is no longer true, and
            // "read-only" is not a property a config can claim any more — it has to be
            // arranged by turning the intent source off.
            if self.environment.is_live_wiring() && !self.safety.dead_mans_switch {
                return bad(
                    "safety.dead_mans_switch = false with intent.enabled = true would place \
                     orders that outlive this process with nothing at the venue to remove \
                     them. Turn the intent source off for a read-only session"
                        .into(),
                );
            }
        }

        // The market-data ring. Every check here catches a configuration that publishes
        // nowhere, or publishes over something else.
        if self.md_ring.enabled {
            if self.md_ring.path.trim().is_empty() {
                return bad(
                    "md_ring.path is empty but md_ring.enabled is true - the publisher \
                     would have nowhere to write"
                        .into(),
                );
            }
            if self.md_ring.capacity == 0 || !self.md_ring.capacity.is_power_of_two() {
                return bad(format!(
                    "md_ring.capacity {} must be a power of two; the ring header refuses \
                     anything else and the session would fail to start",
                    self.md_ring.capacity
                ));
            }
            // The two rings run in opposite directions and carry different records, so
            // one file cannot be both. Worse than useless: the publisher *creates* its
            // ring, so pointing it at the signal ring truncates the file Python is
            // producing into — and that shows up as "the strategy stopped signalling",
            // nowhere near its cause.
            //
            // Checked against **every** producer's ring, not just `[ipc]`: with
            // `[[strategy.producer]]` declared, `ipc.signal_ring_path` is not read at
            // all, so a check that only looked there would pass a config whose publisher
            // truncates the ring a strategy is producing into.
            for p in self.producers() {
                if self.md_ring.path.trim() == p.ring_path.trim() {
                    return bad(format!(
                        "md_ring.path and producer {:?}'s signal_ring are both {:?}. \
                         Creating the market-data ring truncates that file, destroying \
                         the signal ring Python writes into",
                        p.name, self.md_ring.path
                    ));
                }
            }
        }

        // The portfolio bounds. Two sign traps and one that can never bind; all three
        // are configurations an operator would read as protection and would not get.
        if let Err(e) = self.portfolio.to_limits().validate() {
            return bad(e);
        }

        // Session capture. Each of these produces a session that believes it is
        // recording and is not, or one that records over something else.
        if self.capture.enabled {
            if self.capture.path.trim().is_empty() {
                return bad(
                    "capture.path is empty but capture.enabled is true - the recording \
                     would have nowhere to go"
                        .into(),
                );
            }
            if self.capture.queue_capacity == 0 {
                return bad(
                    "capture.queue_capacity must be > 0; a zero-capacity channel is a \
                     rendezvous, so every hand-off would block the core thread until the \
                     writer took it - which is exactly the stall the writer thread exists \
                     to prevent"
                        .into(),
                );
            }
            // The capture *creates and truncates* both of its files, and both rings are
            // files too. Pointing one at another is not a wasted setting: it destroys the
            // transport, and the symptom (the strategy going quiet) is nowhere near the
            // cause.
            // The bar ring's and the beacon's paths are *derived* from `md_ring.path`
            // rather than configured, so an operator cannot see either in their own file
            // — which is exactly why they have to be checked here rather than left to
            // them. ADR-0034 §4's "unrepresentable" argument covers the beacon colliding
            // with the two rings, because one switch derives all three from one name; it
            // says nothing about a *capture* aimed at the beacon, which is a path an
            // operator types. The failure is the worst-shaped one in this list: the
            // capture truncates, a monitor with the page already mapped takes a SIGBUS
            // for the instant the file is zero-length, and the session that killed it
            // goes on printing OK.
            let bars = crate::mdring::bar_ring_path(&self.md_ring.path)
                .to_string_lossy()
                .into_owned();
            let beacon = axon_ipc::beacon_path(&self.md_ring.path)
                .to_string_lossy()
                .into_owned();
            for (name, other) in [
                ("ipc.signal_ring_path", &self.ipc.signal_ring_path),
                ("md_ring.path", &self.md_ring.path),
                ("the bar ring derived from md_ring.path", &bars),
                ("the beacon derived from md_ring.path", &beacon),
            ] {
                if self.capture.path.trim() == other.trim() {
                    return bad(format!(
                        "capture.path and {name} are both {:?}. The capture truncates what \
                         it creates, which would destroy the ring",
                        self.capture.path
                    ));
                }
            }
        }

        // Environment ↔ network. Getting this pair wrong is the expensive mistake:
        // "sandbox" that reaches mainnet spends real money on a test.
        match (self.environment, self.venue.network) {
            (Environment::Sandbox, Network::Mainnet) => {
                return bad(
                    "environment = \"sandbox\" with venue.network = \"mainnet\" would place \
                     real-money orders from a test session"
                        .into(),
                )
            }
            (Environment::Live, Network::Testnet) => {
                return bad(
                    "environment = \"live\" with venue.network = \"testnet\" - say \
                     \"sandbox\" if testnet is what you meant"
                        .into(),
                )
            }
            _ => {}
        }

        if self.environment.is_live_wiring() {
            match self.venue.account.as_deref() {
                None => {
                    return bad(
                        "a live session needs venue.account (the master account address). It \
                         cannot be derived: the signing key belongs to the agent wallet, and \
                         the venue answers an agent address with empty results and no error"
                            .into(),
                    )
                }
                Some(a) if !is_address(a) => {
                    return bad(format!(
                        "venue.account {a:?} is not an 0x-prefixed 20-byte hex address"
                    ))
                }
                Some(_) => {}
            }
            if self.venue.name != axon_provider_hyperliquid::VENUE {
                return bad(format!(
                    "venue.name {:?} has no adapter; the only live venue is {:?}",
                    self.venue.name,
                    axon_provider_hyperliquid::VENUE
                ));
            }
        }
        Ok(())
    }

    /// Whether the session may reach mainnet, given config *and* the environment gate.
    ///
    /// Two independent switches on purpose: a config file can be copied, edited or
    /// generated by mistake, and one file's worth of typo should not be sufficient to
    /// trade real money.
    pub fn mainnet_allowed(&self) -> Result<(), ConfigError> {
        if self.venue.network == Network::Mainnet
            && std::env::var(ENV_ALLOW_MAINNET).as_deref() != Ok("1")
        {
            return Err(ConfigError::Invalid(format!(
                "refusing to run against mainnet without {ENV_ALLOW_MAINNET}=1"
            )));
        }
        Ok(())
    }

    pub fn is_mainnet(&self) -> bool {
        self.venue.network == Network::Mainnet
    }

    /// The venue coin names for the configured instruments.
    ///
    /// The config names instruments venue-neutrally (`"BTC-PERP"`) so one strategy
    /// config can address more than one venue; Hyperliquid names its perps by bare
    /// coin. Anything before the first `-` is the coin, so a plain `"BTC"` also works.
    /// Duplicates collapse — subscribing twice to the same coin doubles the frames.
    pub fn coins(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::with_capacity(self.strategy.symbols.len());
        for s in &self.strategy.symbols {
            let coin = s.split('-').next().unwrap_or(s).trim().to_uppercase();
            if !coin.is_empty() && !out.contains(&coin) {
                out.push(coin);
            }
        }
        out
    }

    pub fn ws_url(&self) -> &str {
        self.venue
            .ws_url
            .as_deref()
            .unwrap_or(match self.venue.network {
                Network::Mainnet => HyperliquidMarketData::MAINNET_WS,
                Network::Testnet => HyperliquidMarketData::TESTNET_WS,
            })
    }

    pub fn info_url(&self) -> &str {
        self.venue
            .info_url
            .as_deref()
            .unwrap_or(match self.venue.network {
                Network::Mainnet => axon_provider_hyperliquid::ws::MAINNET_INFO,
                Network::Testnet => axon_provider_hyperliquid::ws::TESTNET_INFO,
            })
    }

    pub fn exchange_url(&self) -> &str {
        self.venue
            .exchange_url
            .as_deref()
            .unwrap_or(match self.venue.network {
                Network::Mainnet => ExchangeClient::MAINNET,
                Network::Testnet => ExchangeClient::TESTNET,
            })
    }

    pub fn governor_config(&self) -> GovernorConfig {
        GovernorConfig {
            place_reserve_credits: self.governor.place_reserve_credits,
            place_reserve_ip_weight: self.governor.place_reserve_ip_weight,
            initial_address_cap: self.governor.initial_address_cap,
        }
    }

    /// Address credits the dead-man's switch alone will spend per UTC day.
    ///
    /// Printed at startup because it is the least obvious running cost in the system:
    /// the address budget is cumulative and volume-gated, so on a fresh account the
    /// safety loop can be the largest consumer of it.
    pub fn dms_actions_per_day(&self) -> u64 {
        if !self.safety.dead_mans_switch {
            return 0;
        }
        DAY_MS / self.safety.rearm_interval_ms.max(1)
    }

    pub fn mark_max_age_ns(&self) -> axon_core::Nanos {
        (self.session.mark_max_age_ms as axon_core::Nanos).saturating_mul(1_000_000)
    }
}

/// `0x` + 40 hex characters. Deliberately strict: a truncated or checksum-mangled
/// address reads as an idle account rather than as an error.
fn is_address(s: &str) -> bool {
    s.len() == 42 && s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
}

/// Key fragments that must never appear in a config file.
const SECRET_KEY_FRAGMENTS: [&str; 6] = [
    "secret",
    "private_key",
    "privatekey",
    "privkey",
    "mnemonic",
    "passphrase",
];

/// Walk the parsed TOML and refuse anything that looks like a credential.
///
/// This is a tripwire, not a security boundary — it cannot stop a key pasted into a
/// field called `note`. What it does stop is the realistic accident: someone adds
/// `secret_key = "0x…"` next to `account`, it works, and the key is in git forever.
fn reject_secrets(value: &toml::Value) -> Result<(), ConfigError> {
    match value {
        toml::Value::Table(map) => {
            for (k, v) in map {
                let lower = k.to_ascii_lowercase();
                if SECRET_KEY_FRAGMENTS.iter().any(|f| lower.contains(f)) {
                    return Err(ConfigError::SecretInConfig {
                        key: k.clone(),
                        env: axon_provider_hyperliquid::HlSigner::ENV_KEY,
                    });
                }
                reject_secrets(v)?;
            }
            Ok(())
        }
        toml::Value::Array(items) => items.iter().try_for_each(reject_secrets),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(name: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, body).expect("write temp config");
        p
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = RuntimeConfig::default();
        let text = toml::to_string_pretty(&cfg).expect("serialize");
        let back: RuntimeConfig = toml::from_str(&text).expect("deserialize");
        assert_eq!(cfg, back);
    }

    // ── many producers, one account (ADR-0038) ───────────────────────────────

    /// A session with `n` declared producers over BTC/ETH/SOL, otherwise the default.
    fn multi(n: usize) -> RuntimeConfig {
        let coins = ["BTC-PERP", "ETH-PERP", "SOL-PERP"];
        let mut cfg = RuntimeConfig::default();
        cfg.strategy.symbols = coins.iter().map(|s| s.to_string()).collect();
        cfg.strategy.producers = (0..n)
            .map(|i| ProducerConfig {
                name: format!("p{i}"),
                signal_ring: format!("/dev/shm/axon-test-{i}.ring"),
                symbols: vec![coins[i % coins.len()].to_string()],
                on_silence: OnSilenceMode::Hold,
                silence_ms: 0,
                max_gross_notional: Decimal::ZERO,
            })
            .collect();
        cfg
    }

    fn refusal(cfg: &RuntimeConfig) -> String {
        match cfg.validate() {
            Err(ConfigError::Invalid(m)) => m,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_session_with_no_declared_producers_resolves_to_exactly_the_one_it_always_had() {
        // The property that lets this land: every config written before ADR-0038 has to
        // describe the same session it described yesterday. Empty is not "no producers",
        // it is the single-producer session whose ring is `[ipc]` and whose name is the
        // strategy's — and it must come out of `producers()` in the *same shape* as a
        // declared one, so nothing downstream has a branch for it.
        let cfg = RuntimeConfig::default();
        let ps = cfg.producers();
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].id, axon_strategy::StrategyId::new(0));
        assert_eq!(ps[0].name, cfg.strategy.name);
        assert_eq!(ps[0].ring_path, cfg.ipc.signal_ring_path);
        assert!(ps[0].symbols.is_empty(), "the whole session universe");
        assert_eq!(ps[0].on_silence, OnSilenceMode::Hold);
        assert_eq!(ps[0].silence_ms, 0, "never called silent");
    }

    #[test]
    fn declared_producers_get_stable_ids_in_declaration_order() {
        let cfg = multi(3);
        let ps = cfg.producers();
        assert_eq!(ps.len(), 3);
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(p.id.get(), i as u16);
            assert_eq!(p.name, format!("p{i}"));
        }
        cfg.validate().expect("three disjoint producers are fine");
    }

    #[test]
    fn two_producers_on_one_ring_are_refused_because_an_spsc_ring_has_one_consumer() {
        // The worst of the multi-producer misconfigurations, and the least visible: two
        // readers of one SPSC ring do not share it, they steal from it, so each sees a
        // share of the records and neither can tell that from a producer with nothing to
        // say. Measured on the *bar* ring on 2026-07-26 (ADR-0029 §5); this is the same
        // hazard on the signal side, and it is caught at startup rather than in a run.
        let mut cfg = multi(2);
        cfg.strategy.producers[1].signal_ring = cfg.strategy.producers[0].signal_ring.clone();
        let m = refusal(&cfg);
        assert!(m.contains("one consumer"), "{m}");
        assert!(m.contains("steal"), "{m}");
    }

    #[test]
    fn two_producers_with_one_name_are_refused_because_the_status_line_is_keyed_on_it() {
        let mut cfg = multi(2);
        cfg.strategy.producers[1].name = cfg.strategy.producers[0].name.clone();
        assert!(refusal(&cfg).contains("both named"));
    }

    #[test]
    fn a_producer_claiming_a_symbol_the_session_does_not_subscribe_is_refused() {
        // The session subscribes market data for `strategy.symbols` and nothing else, so
        // a target for anything outside it would be planned against no book, refused as
        // `NoQuote` on every pass, and read on the status line as a strategy with no
        // opinion.
        let mut cfg = multi(1);
        cfg.strategy.producers[0].symbols = vec!["DOGE-PERP".into()];
        assert!(refusal(&cfg).contains("not in strategy.symbols"));
    }

    #[test]
    fn an_overlap_is_refused_at_startup_unless_the_operator_asked_for_netting() {
        // Netting is correct — two claims on one position add — and it is opt-in,
        // because two producers pointed at one instrument is far more often a
        // copy-pasted config than a decision, and the accident composes two strategies'
        // risk into a position neither author sized.
        let mut cfg = multi(2);
        cfg.strategy.producers[1].symbols = cfg.strategy.producers[0].symbols.clone();
        let m = refusal(&cfg);
        assert!(m.contains("overlap"), "{m}");
        assert!(m.contains("\"net\""), "the fix is named: {m}");

        cfg.portfolio.overlap = OverlapMode::Net;
        cfg.validate().expect("declared netting is a legal session");
    }

    #[test]
    fn two_producers_that_declare_no_symbols_are_the_same_overlap_written_shorter() {
        // Empty means "the whole universe", so two empty producers claim every
        // instrument twice. Caught by name, because the error an operator would
        // otherwise get is nothing at all until the first pass that trades.
        let mut cfg = multi(2);
        cfg.strategy.producers[0].symbols.clear();
        cfg.strategy.producers[1].symbols.clear();
        assert!(refusal(&cfg).contains("declare no symbols"));

        // One open producer beside one that names its instruments is legal: the open one
        // is the catch-all, and an overlap between them is a runtime question the book
        // answers per record rather than a config error.
        cfg.strategy.producers[1].symbols = vec!["BTC-PERP".into()];
        cfg.validate().expect("one open producer is fine");
    }

    #[test]
    fn flat_on_silence_with_no_window_is_refused_rather_than_never_firing() {
        // Zero means "never call it silent" on this field, exactly as it does on
        // `ttl_ms` and `max_order_age_ms`. So the combination is a policy that can never
        // fire, and the operator who asked for it would find out by not being protected.
        let mut cfg = multi(1);
        cfg.strategy.producers[0].on_silence = OnSilenceMode::Flat;
        let m = refusal(&cfg);
        assert!(m.contains("never call it silent"), "{m}");
        cfg.strategy.producers[0].silence_ms = 120_000;
        cfg.validate()
            .expect("a flat policy with a window is legal");
    }

    #[test]
    fn the_market_data_publisher_may_not_truncate_any_producers_ring() {
        // With `[[strategy.producer]]` declared, `ipc.signal_ring_path` is not read at
        // all — so the pre-existing check, which only looked there, would have passed a
        // config whose publisher creates (and therefore truncates) the ring a strategy
        // is producing into. The symptom is "the strategy stopped signalling", nowhere
        // near its cause.
        let mut cfg = multi(2);
        cfg.md_ring.enabled = true;
        cfg.md_ring.path = cfg.strategy.producers[1].signal_ring.clone();
        let m = refusal(&cfg);
        assert!(m.contains("truncates"), "{m}");
        assert!(m.contains("p1"), "the producer is named: {m}");
    }

    #[test]
    fn a_portfolio_bound_that_can_never_bind_is_refused() {
        // |sum| <= sum of |.| for every book, so a net ceiling above the gross one is a
        // directional limit an operator believes they have and does not.
        let mut cfg = RuntimeConfig::default();
        cfg.portfolio.max_gross_notional = Decimal::from(100);
        cfg.portfolio.max_net_notional = Decimal::from(200);
        assert!(refusal(&cfg).contains("can never bind"));

        cfg.portfolio.max_net_notional = Decimal::from(60);
        cfg.validate()
            .expect("a net bound inside the gross one is fine");
    }

    #[test]
    fn the_default_portfolio_declares_nothing_and_therefore_gates_nothing() {
        // An existing config gains the table and gains nothing that can refuse an order.
        let cfg = RuntimeConfig::default();
        assert!(!cfg.portfolio.is_declared());
        assert!(!cfg.portfolio.to_limits().is_declared());
        assert_eq!(cfg.portfolio.overlap, OverlapMode::Exclusive);
        assert!(
            cfg.portfolio.scale_to_fit,
            "on, and inert while no gross bound is declared"
        );
    }

    #[test]
    fn a_multi_producer_config_round_trips_through_toml() {
        // `[[strategy.producer]]` is an array of tables and is declared last in the
        // struct for exactly this reason: TOML serializes in field order, and a table
        // array emitted before `[strategy.risk]` produces a file that parses back
        // differently from the one it was written from.
        let mut cfg = multi(2);
        cfg.portfolio = PortfolioConfig {
            max_gross_notional: Decimal::from(150),
            max_net_notional: Decimal::from(100),
            max_symbols: 3,
            overlap: OverlapMode::Net,
            scale_to_fit: true,
        };
        cfg.strategy.producers[0].on_silence = OnSilenceMode::Flat;
        cfg.strategy.producers[0].silence_ms = 300_000;
        cfg.strategy.producers[0].max_gross_notional = Decimal::from(60);

        let text = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(text.contains("[[strategy.producer]]"), "{text}");
        let back: RuntimeConfig = toml::from_str(&text).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn risk_config_maps_to_limits() {
        let cfg = RuntimeConfig::default();
        let limits = cfg.strategy.risk.to_limits();
        assert_eq!(limits.max_position, Decimal::from(10));
    }

    #[test]
    fn the_default_config_is_offline_and_valid() {
        // `cargo run --bin axon` with no arguments must not be able to reach a venue.
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.environment, Environment::Backtest);
        assert!(!cfg.environment.is_live_wiring());
        assert!(cfg.venue.account.is_none());
        cfg.validate().expect("the shipped default must be valid");
    }

    #[test]
    fn a_dumped_default_reloads_as_itself() {
        let cfg = RuntimeConfig::default();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let path = write_temp("axon-cfg-roundtrip.toml", &text);
        let back = RuntimeConfig::load(&path).expect("reload");
        assert_eq!(cfg, back);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_missing_config_path_is_an_error_not_a_silent_default() {
        // Falling back to the offline default when a live config was requested is how
        // a session runs for an hour before anyone notices it never connected.
        let err = RuntimeConfig::load("/nonexistent/axon.toml").unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
    }

    #[test]
    fn a_config_carrying_a_secret_is_refused() {
        // The realistic accident: someone adds the key next to the account address,
        // it works, and the key is in git forever. Refusing to start is the only
        // moment at which that is still recoverable.
        let text = toml::to_string_pretty(&RuntimeConfig::default())
            .unwrap()
            .replace("[venue]", "[venue]\nsecret_key = \"0xdeadbeef\"");
        let err = RuntimeConfig::from_toml(&text, "test").unwrap_err();
        match err {
            ConfigError::SecretInConfig { key, env } => {
                assert_eq!(key, "secret_key");
                assert_eq!(env, "AXON_HL_SECRET_KEY");
            }
            other => panic!("expected SecretInConfig, got {other}"),
        }

        // …and a nested one is caught too, since the walk is recursive.
        let text = toml::to_string_pretty(&RuntimeConfig::default())
            .unwrap()
            .replace("[strategy]", "[strategy]\napi_privatekey = \"0x00\"");
        assert!(matches!(
            RuntimeConfig::from_toml(&text, "test"),
            Err(ConfigError::SecretInConfig { .. })
        ));
    }

    #[test]
    fn a_dead_mans_switch_lead_below_the_venue_minimum_is_refused() {
        let mut cfg = RuntimeConfig::default();
        cfg.safety.lead_ms = 4_000;
        cfg.safety.rearm_interval_ms = MIN_REARM_INTERVAL_MS;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("below the venue minimum"));
    }

    #[test]
    fn a_rearm_interval_that_leaves_no_room_for_a_missed_beat_is_refused() {
        // lead 30 s re-armed every 20 s: one failed re-arm and the deadline is 10 s
        // away, i.e. the switch fires on the next hiccup and burns a daily trigger.
        let mut cfg = RuntimeConfig::default();
        cfg.safety.lead_ms = 30_000;
        cfg.safety.rearm_interval_ms = 20_000;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("at least 3 times"), "{err}");

        cfg.safety.lead_ms = 60_000;
        cfg.validate().expect("3:1 is acceptable");
    }

    #[test]
    fn a_sandbox_session_cannot_be_pointed_at_mainnet() {
        // The single most expensive config typo available: a test session spending
        // real money.
        let mut cfg = RuntimeConfig {
            environment: Environment::Sandbox,
            ..RuntimeConfig::default()
        };
        cfg.venue.network = Network::Mainnet;
        cfg.venue.account = Some("0x".to_string() + &"a".repeat(40));
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("real-money"), "{err}");
    }

    #[test]
    fn a_live_session_without_an_account_address_is_refused() {
        // The agent wallet's own address would query as an empty account and look
        // exactly like an idle one, so guessing is not an option.
        let mut cfg = RuntimeConfig {
            environment: Environment::Sandbox,
            ..RuntimeConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("venue.account"), "{err}");

        cfg.venue.account = Some("0xnothex".into());
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("hex address"), "{err}");

        cfg.venue.account = Some("0x".to_string() + &"1".repeat(40));
        cfg.validate().expect("a well-formed address is accepted");
    }

    #[test]
    fn symbols_resolve_to_venue_coins_without_duplicates() {
        let mut cfg = RuntimeConfig::default();
        cfg.strategy.symbols = vec![
            "BTC-PERP".into(),
            "btc".into(),
            "ETH-PERP".into(),
            "SOL".into(),
        ];
        assert_eq!(cfg.coins(), vec!["BTC", "ETH", "SOL"]);
    }

    #[test]
    fn urls_follow_the_network_unless_overridden() {
        let mut cfg = RuntimeConfig::default();
        assert!(cfg.ws_url().contains("testnet"));
        assert!(cfg.info_url().contains("testnet"));
        assert!(cfg.exchange_url().contains("testnet"));

        cfg.venue.network = Network::Mainnet;
        cfg.environment = Environment::Live;
        cfg.venue.account = Some("0x".to_string() + &"b".repeat(40));
        cfg.validate().unwrap();
        assert!(!cfg.ws_url().contains("testnet"));

        cfg.venue.ws_url = Some("ws://127.0.0.1:9/ws".into());
        assert_eq!(cfg.ws_url(), "ws://127.0.0.1:9/ws");
    }

    #[test]
    fn the_daily_cost_of_the_safety_loop_is_reported() {
        // The address budget is cumulative and volume-gated, so on a fresh account the
        // re-arm cadence can be the largest consumer of it. 20 s = 4 320/day against a
        // 10 000-credit starting buffer.
        let mut cfg = RuntimeConfig::default();
        assert_eq!(cfg.dms_actions_per_day(), 4_320);
        cfg.safety.rearm_interval_ms = 5_000;
        cfg.safety.lead_ms = 60_000;
        assert_eq!(cfg.dms_actions_per_day(), 17_280);
        cfg.safety.dead_mans_switch = false;
        assert_eq!(cfg.dms_actions_per_day(), 0);
    }

    #[test]
    fn the_default_intent_source_is_on_and_still_offline() {
        // On, because a runtime whose intent source is off looks exactly like a healthy
        // session nobody is signalling. Offline, because `environment` is what decides
        // whether anything opens a socket — the intent source reads a local file.
        let cfg = RuntimeConfig::default();
        assert!(cfg.intent.enabled);
        assert!(!cfg.environment.is_live_wiring());
        assert_eq!(cfg.intent.reader_config().max_age_ms, 2_000);
        assert_eq!(cfg.intent.planner_config().taker_slippage_bps, 50);
        cfg.validate().expect("the shipped default must be valid");
    }

    #[test]
    fn an_intent_setting_that_would_silently_trade_nothing_is_refused() {
        // Each of these leaves the session reporting OK while it cannot place an order,
        // which is indistinguishable from a strategy that simply had no opinion.
        for (name, mutate) in [
            (
                "max_per_drain",
                Box::new(|c: &mut RuntimeConfig| c.intent.max_per_drain = 0)
                    as Box<dyn Fn(&mut RuntimeConfig)>,
            ),
            (
                "max_signal_age_ms",
                Box::new(|c: &mut RuntimeConfig| c.intent.max_signal_age_ms = 0),
            ),
            (
                "queue_capacity",
                Box::new(|c: &mut RuntimeConfig| c.intent.queue_capacity = 0),
            ),
            (
                "drain_interval_ms",
                Box::new(|c: &mut RuntimeConfig| c.intent.drain_interval_ms = 0),
            ),
            (
                "sweep_interval_ms",
                Box::new(|c: &mut RuntimeConfig| c.intent.sweep_interval_ms = 0),
            ),
            (
                "signal_ring_path",
                Box::new(|c: &mut RuntimeConfig| c.ipc.signal_ring_path = "  ".into()),
            ),
            (
                "capacity",
                Box::new(|c: &mut RuntimeConfig| c.ipc.capacity = 1000),
            ),
        ] {
            let mut cfg = RuntimeConfig::default();
            mutate(&mut cfg);
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn a_trading_session_cannot_turn_off_the_dead_mans_switch() {
        // Before there was an intent source, `dead_mans_switch = false` was defensible
        // for a session that could not place an order. It can now, so the only way to
        // be read-only is to say so.
        let mut cfg = RuntimeConfig {
            environment: Environment::Sandbox,
            ..RuntimeConfig::default()
        };
        cfg.venue.account = Some("0x".to_string() + &"c".repeat(40));
        cfg.safety.dead_mans_switch = false;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("dead_mans_switch"), "{err}");

        cfg.intent.enabled = false;
        cfg.validate()
            .expect("a genuinely read-only session is allowed");
    }

    #[test]
    fn a_config_written_before_the_money_view_existed_still_loads_and_alarms_at_nothing() {
        // Two properties in one: `[pnl]` and `[latency]` are `#[serde(default)]`, so an
        // existing session file keeps loading — and their defaults declare *no* bound,
        // so upgrading the binary gains an operator the numbers and nothing that can
        // fire at 03:00 against a threshold they never chose.
        let toml = r#"
            environment = "backtest"
            [venue]
            name = "hyperliquid"
            network = "testnet"
            [session]
            bus_capacity = 1024
            core_poll_us = 500
            status_interval_ms = 5000
            mark_max_age_ms = 10000
            feeds = ["Bbo"]
            [safety]
            dead_mans_switch = false
            lead_ms = 60000
            rearm_interval_ms = 20000
            [reconcile]
            interval_ms = 15000
            grace_ms = 5000
            rate_limit_every = 10
            [governor]
            place_reserve_credits = 1000
            place_reserve_ip_weight = 200
            initial_address_cap = 10000
            [ipc]
            signal_ring_path = "/dev/shm/x.ring"
            capacity = 1024
            [strategy]
            name = "s"
            version = 1
            symbols = ["BTC-PERP"]
            [strategy.model_ref]
            registry_id = "s"
            version = 1
            [strategy.risk]
            max_position = "1"
            max_notional = "100"
            max_order_qty = "1"
        "#;
        let cfg: RuntimeConfig = toml::from_str(toml).expect("an older file still parses");
        assert_eq!(cfg.pnl, PnlConfig::default());
        assert_eq!(cfg.latency, LatencyConfig::default());
        assert_eq!(cfg.pnl.max_session_loss, Decimal::ZERO, "no bound declared");
        assert_eq!(cfg.latency.signal_age_ms, 0, "no ceiling declared");
        // The loss bounds arrived after every deployed `[strategy.risk]` table was
        // written, and they arrive **off**: a kill switch nobody chose is one that takes
        // a session down on an upgrade, which is the one way a safety feature can be a
        // net loss of safety.
        assert!(!cfg.strategy.risk.to_loss_limits().is_declared());
        assert_eq!(cfg.strategy.risk.daily_state_path, None);
        cfg.validate().expect("and it is valid");
    }

    #[test]
    fn a_daily_loss_bound_with_nowhere_to_persist_its_baseline_is_refused() {
        // A daily bound whose baseline dies with the process is a session bound wearing
        // a day's name, and the failure it exists to stop — a strategy that loses money
        // and takes the process down with it — is exactly the one that restarts it. The
        // degradation is invisible from the status line, so it is refused at load rather
        // than warned about at runtime.
        let mut cfg = RuntimeConfig::default();
        cfg.strategy.risk.max_daily_loss = Decimal::from(5);
        let err = cfg.validate().expect_err("refused");
        assert!(err.to_string().contains("daily_state_path"), "{err}");

        cfg.strategy.risk.daily_state_path = Some(String::new());
        assert!(cfg.validate().is_err(), "an empty path is not a path");

        cfg.strategy.risk.daily_state_path = Some("data/day.json".into());
        cfg.validate().expect("now it can remember the day");

        // A session bound needs no file: a session is a process, so its own accounting
        // is the right scope for it.
        let mut cfg = RuntimeConfig::default();
        cfg.strategy.risk.max_session_loss = Decimal::from(5);
        cfg.validate().expect("no file needed");
    }

    #[test]
    fn a_negative_loss_gate_is_refused_because_it_would_stop_the_session_before_its_first_order() {
        // The same sign trap as the P&L alarms and a worse consequence. An alarm that is
        // always on is ignored; a *gate* that is always on is a session that emits and
        // never trades, which reads exactly like a warmup that has not finished.
        let mut cfg = RuntimeConfig::default();
        cfg.strategy.risk.max_session_loss = Decimal::from(-5);
        let err = cfg.validate().expect_err("refused");
        assert!(err.to_string().contains("max_session_loss"), "{err}");

        let mut cfg = RuntimeConfig::default();
        cfg.strategy.risk.max_daily_loss = Decimal::from(-1);
        cfg.strategy.risk.daily_state_path = Some("data/day.json".into());
        let err = cfg.validate().expect_err("refused");
        assert!(err.to_string().contains("max_day_loss"), "{err}");
    }

    #[test]
    fn a_negative_loss_bound_is_refused_because_it_would_alarm_from_the_first_tick() {
        // `max_session_loss` is a magnitude compared against a signed net figure. `-5`
        // written where `5` was meant makes the warning permanently on, which is the
        // same as permanently off by the second hour.
        let mut cfg = RuntimeConfig::default();
        cfg.pnl.max_session_loss = Decimal::from(-5);
        let err = cfg.validate().expect_err("refused");
        assert!(err.to_string().contains("magnitude"), "{err}");

        let mut cfg = RuntimeConfig::default();
        cfg.pnl.equity_drift_alarm = Decimal::from(-1);
        assert!(cfg.validate().is_err());

        let mut cfg = RuntimeConfig::default();
        cfg.latency.breach_warn_pct = 101;
        let err = cfg.validate().expect_err("a threshold no run can reach");
        assert!(err.to_string().contains("percentage"), "{err}");
    }

    #[test]
    fn the_re_quote_is_on_by_default_and_can_be_turned_off_by_a_number_not_a_flag() {
        // A defect fixed only for operators who read the release notes is not fixed, so
        // the default is not zero. One number rather than an `enabled` beside a count,
        // for the same reason `max_order_age_ms` alone turns the sweeper off: two
        // switches can disagree about whether the mechanism exists.
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.intent.max_requotes, 3);
        cfg.validate().expect("the shipped default must be valid");
    }

    #[test]
    fn the_market_data_ring_is_off_until_someone_asks_for_it() {
        // The publisher creates and truncates its ring file. A default-on setting would
        // make `cargo run --bin axon` — and every test that builds a default config —
        // clobber whatever happens to sit at that path.
        let cfg = RuntimeConfig::default();
        assert!(!cfg.md_ring.enabled);
        assert_eq!(cfg.md_ring.policy, MdWritePolicy::OnChange);
        assert_ne!(
            cfg.md_ring.path, cfg.ipc.signal_ring_path,
            "the two directions are two files"
        );
        cfg.validate().expect("the shipped default must be valid");
    }

    #[test]
    fn an_md_ring_setting_that_would_publish_nowhere_is_refused() {
        // Each of these leaves a session that believes it is feeding Python and is not,
        // which is indistinguishable from a strategy with nothing to say.
        for (name, mutate) in [
            (
                "md_ring.path",
                Box::new(|c: &mut RuntimeConfig| c.md_ring.path = "  ".into())
                    as Box<dyn Fn(&mut RuntimeConfig)>,
            ),
            (
                "md_ring.capacity",
                Box::new(|c: &mut RuntimeConfig| c.md_ring.capacity = 3_000),
            ),
        ] {
            let mut cfg = RuntimeConfig::default();
            cfg.md_ring.enabled = true;
            mutate(&mut cfg);
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn the_md_ring_cannot_be_pointed_at_the_signal_ring() {
        // Creating it truncates the file, so Python's producer would keep writing into a
        // mapping the runtime had just zeroed. The symptom is the strategy going quiet;
        // the cause is two lines apart in a config file.
        let mut cfg = RuntimeConfig::default();
        cfg.md_ring.enabled = true;
        cfg.md_ring.path = cfg.ipc.signal_ring_path.clone();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("truncates"), "{err}");
    }

    #[test]
    fn a_config_written_before_the_publisher_existed_still_loads() {
        // `md_ring` is `#[serde(default)]`, and the default is off — so an older file
        // keeps working *and* cannot start writing a ring by upgrading the binary.
        let text = toml::to_string_pretty(&RuntimeConfig::default()).unwrap();
        // Drop the whole `[md_ring]` block, header and keys: dropping only the header
        // would silently reassign its keys to the preceding table.
        let mut without = String::new();
        let mut in_md = false;
        for line in text.lines() {
            if line.starts_with('[') {
                in_md = line.starts_with("[md_ring]");
            }
            if !in_md {
                without.push_str(line);
                without.push('\n');
            }
        }
        assert!(!without.contains("md_ring"), "{without}");
        let cfg = RuntimeConfig::from_toml(&without, "test").expect("an older config loads");
        assert!(!cfg.md_ring.enabled);
    }

    #[test]
    fn a_config_written_before_the_planners_newest_knobs_still_loads() {
        // The failure mode a `#[serde(default)]` on a *field* prevents, as opposed to on
        // a table: `[intent]` already exists in every deployed config file, so adding a
        // required key to it makes every one of those files stop parsing. The operator's
        // upgrade then dies at startup with `missing field noop_band_bps` — a message
        // about TOML syntax, for a change about queue position — and the session that
        // was trading a minute ago will not start.
        //
        // The second assertion is the one worth stating out loud: the order-lifetime
        // default is **on**, so a file that never mentioned it comes back with a minute's
        // ceiling on a resting order. That is deliberate (it is the only bound on the
        // two changes that widened ADR-0014 §6's leave-it-resting exception) and it is a
        // behaviour change nobody typed.
        let text = toml::to_string_pretty(&RuntimeConfig::default()).unwrap();
        let without: String = text
            .lines()
            .filter(|l| {
                !l.starts_with("noop_band_bps")
                    && !l.starts_with("max_order_age_ms")
                    && !l.starts_with("sweep_interval_ms")
            })
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(!without.contains("noop_band_bps"), "{without}");
        assert!(!without.contains("max_order_age_ms"), "{without}");
        assert!(!without.contains("sweep_interval_ms"), "{without}");

        let cfg = RuntimeConfig::from_toml(&without, "test").expect("an older config loads");
        assert_eq!(
            cfg.intent.noop_band_bps, 0,
            "off, as a bounded position error"
        );
        assert_eq!(
            cfg.intent.max_order_age_ms, 60_000,
            "on, and it bounds the rest"
        );
        assert_eq!(
            cfg.intent.sweep_interval_ms, 1_000,
            "and the sweeper that enforces it runs, on a config that never heard of it"
        );
        assert_eq!(cfg.intent, IntentConfig::default());
    }

    #[test]
    fn session_capture_is_off_until_someone_asks_for_it() {
        // A recording creates and truncates files and does unbounded work on the
        // deterministic path's behalf. Neither is something a binary upgrade should
        // start doing to an existing deployment.
        let cfg = RuntimeConfig::default();
        assert!(!cfg.capture.enabled);
        assert!(cfg.capture.signals, "a replay needs both halves");
        assert_ne!(cfg.capture.path, cfg.ipc.signal_ring_path);
        assert_ne!(cfg.capture.path, cfg.md_ring.path);
        cfg.validate().expect("the shipped default must be valid");
    }

    #[test]
    fn a_capture_setting_that_would_stall_or_record_nothing_is_refused() {
        // Each of these is a session that looks like it is recording. The queue one is
        // the sharp case: crossbeam's zero-capacity channel is a *rendezvous*, so it
        // would turn every hand-off into the blocking write the writer thread exists to
        // avoid.
        for (name, mutate) in [
            (
                "capture.path",
                Box::new(|c: &mut RuntimeConfig| c.capture.path = "  ".into())
                    as Box<dyn Fn(&mut RuntimeConfig)>,
            ),
            (
                "capture.queue_capacity",
                Box::new(|c: &mut RuntimeConfig| c.capture.queue_capacity = 0),
            ),
        ] {
            let mut cfg = RuntimeConfig::default();
            cfg.capture.enabled = true;
            mutate(&mut cfg);
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn a_capture_cap_of_zero_means_no_cap_because_that_is_what_the_writer_does() {
        // `capture.max_bytes = 0` was refused here, on the grounds that a zero cap stops
        // the recording before its first record. That was true when it was written and
        // stopped being true when the writer learned to read `0` as *no limit* — and the
        // two files then disagreed with validation winning, so the escape hatch an
        // operator is told about in `capture`'s own docs could not be reached at all.
        //
        // The general shape is worth more than this instance: a validation rule is an
        // assertion about another module's behaviour, and nothing links the two. When
        // that module changes its mind, the rule keeps refusing a config that would now
        // work, and the message it refuses with describes behaviour that no longer
        // exists.
        let mut cfg = RuntimeConfig::default();
        cfg.capture.enabled = true;
        cfg.capture.max_bytes = 0;
        cfg.validate()
            .expect("0 is the documented way to say 'this volume is mine'");
    }

    #[test]
    fn a_capture_cannot_be_pointed_at_either_ring() {
        // It truncates what it creates. Aimed at the signal ring it destroys the file
        // Python is producing into; aimed at the market-data ring it destroys the one
        // Python is consuming — and both present as "the strategy went quiet".
        for path in [
            RuntimeConfig::default().ipc.signal_ring_path,
            RuntimeConfig::default().md_ring.path,
        ] {
            let mut cfg = RuntimeConfig::default();
            cfg.capture.enabled = true;
            cfg.capture.path = path;
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains("truncates"), "{err}");
        }
    }

    #[test]
    fn a_capture_cannot_be_pointed_at_the_derived_bar_ring_or_beacon_either() {
        // Both are derived from `md_ring.path` rather than configured, so an operator
        // cannot see either in their own file and cannot be expected to avoid them.
        // The beacon is the worse of the two: the capture truncates, a monitor with the
        // page already mapped takes a SIGBUS for the instant the file is zero-length,
        // and the session that killed it goes on printing OK. ADR-0034 §4's
        // "unrepresentable" argument covers the beacon colliding with the *rings* —
        // one switch derives all three from one name — and says nothing about this.
        let base = RuntimeConfig::default().md_ring.path;
        let derived = [
            crate::mdring::bar_ring_path(&base)
                .to_string_lossy()
                .into_owned(),
            axon_ipc::beacon_path(&base).to_string_lossy().into_owned(),
        ];
        assert_ne!(derived[0], derived[1], "the two derived paths collide");
        for path in derived {
            assert_ne!(
                path, base,
                "a derived path shadows the ring it is derived from"
            );
            let mut cfg = RuntimeConfig::default();
            cfg.capture.enabled = true;
            cfg.capture.path = path.clone();
            let err = cfg
                .validate()
                .expect_err(&format!("{path} was accepted as a capture target"));
            assert!(err.to_string().contains("truncates"), "{err}");
        }
    }

    #[test]
    fn a_config_written_before_capture_existed_still_loads_with_it_off() {
        // `capture` is `#[serde(default)]` and the default is off, so an older file keeps
        // working *and* upgrading the binary cannot start writing logs into a deployment
        // that never asked for them.
        let text = toml::to_string_pretty(&RuntimeConfig::default()).unwrap();
        let mut without = String::new();
        let mut in_capture = false;
        for line in text.lines() {
            if line.starts_with('[') {
                in_capture = line.starts_with("[capture]");
            }
            if !in_capture {
                without.push_str(line);
                without.push('\n');
            }
        }
        assert!(!without.contains("capture"), "{without}");
        let cfg = RuntimeConfig::from_toml(&without, "test").expect("an older config loads");
        assert!(!cfg.capture.enabled);
    }

    #[test]
    fn feed_lists_survive_a_toml_round_trip() {
        let mut cfg = RuntimeConfig::default();
        cfg.session.feeds = vec![Feed::Bbo, Feed::L2Book, Feed::Trades, Feed::Ticker];
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: RuntimeConfig = toml::from_str(&text).unwrap();
        assert_eq!(cfg.session.feeds, back.session.feeds);
    }

    #[test]
    fn the_candle_feed_is_nameable_in_a_config_file_and_not_only_in_rust() {
        // The only [`Feed`] carrying a payload, and the only one an operator cannot
        // write as a bare string. Every other entry in `session.feeds` is `"Bbo"`; this
        // one is an inline table, and the natural guess — `"Candles"` — is a parse
        // error at *startup*, which is the worst moment to discover a config syntax.
        // Candles are not in the default feed set, so nothing else exercises the shape.
        let toml_text = r#"feeds = ["Bbo", { Candles = "m1" }, "Ticker"]"#;
        #[derive(serde::Deserialize)]
        struct Feeds {
            feeds: Vec<Feed>,
        }
        let parsed: Feeds = toml::from_str(toml_text).expect("the documented shape parses");
        assert_eq!(
            parsed.feeds,
            vec![
                Feed::Bbo,
                Feed::Candles(axon_providers::CandleInterval::M1),
                Feed::Ticker
            ]
        );

        // …and it survives a full round trip through the real config, so a session that
        // rewrites its own configuration cannot drop the interval.
        let mut cfg = RuntimeConfig::default();
        cfg.session.feeds = vec![Feed::Bbo, Feed::Candles(axon_providers::CandleInterval::M1)];
        let back: RuntimeConfig =
            toml::from_str(&toml::to_string_pretty(&cfg).unwrap()).expect("round trip");
        assert_eq!(cfg.session.feeds, back.session.feeds);
    }
}
