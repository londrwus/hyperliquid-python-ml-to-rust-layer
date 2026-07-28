//! The composition root: turns a [`RuntimeConfig`] into a running session.
//!
//! Two shapes, one core. The offline session ([`run_offline`]) builds the bus, the
//! handler and the state, pushes a canned event stream through them and exits — no
//! socket, no key, no tokio anywhere in the process. The live session ([`run_live`])
//! builds the same core and adds the async edges around it:
//!
//! ```text
//!   tokio runtime (main thread)                 axon-core thread (no tokio)
//!   ───────────────────────────                 ───────────────────────────
//!   WS: market data + user channels ─┐
//!   /info reconciliation poll ───────┼─ bus ──▶ CoreHandler
//!   dead-man's-switch re-arm         │            ├─ MarketDataProcessor
//!   signal handler ──────────────────┘            ├─ MarkCache
//!                                                 ├─ OrderTracker
//!                                                 └─ MdPublisher ──▶ md ring
//!   submit pipeline (halt→risk→rate→venue)        IntentSource
//!            ▲                                    ├─ SignalReader ◀── signal ring
//!            └────────── intent queue ────────────┴─ Planner
//! ```
//!
//! Both rings are driven from the core thread and neither can block it: the signal ring
//! is polled, the market-data ring is written with a `try_push` that drops rather than
//! waits. That is what lets the two cross-process boundaries sit inside a deterministic
//! loop at all.
//!
//! The core runs on a **dedicated OS thread** rather than a blocking task, so no code
//! on the deterministic path has a runtime handle in scope to accidentally `await`
//! into. The supervisor never touches the handler's state directly; everything it
//! needs to publish goes on the bus, and everything it needs to read is behind the
//! same shared handles the core writes to.
//!
//! Startup order is a safety property, not a convenience:
//! **arm before subscribing, reconcile before trading.** The session begins *halted*
//! when the dead-man's switch is enabled and only starts accepting orders once the
//! first re-arm succeeds — the alternative is a window where orders can rest with no
//! venue-side protection behind them.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use axon_core::{bus, EventSender, SymbolId};
use axon_execution::{
    GovernedClient, GuardedClient, HaltSwitch, HaltableClient, MarkCache, OrderTracker,
    PortfolioEngine, RateLimiter, TrackerRiskContext,
};
use axon_provider_hyperliquid::{
    ExchangeClient, HlSigner, HyperliquidMarketData, RateGovernor, SymbolMap,
};
use axon_providers::{
    CancelAck, CancelId, Capabilities, ExecutionClient, InstrumentTable, OrderAck, OrderRequest,
    ProviderError,
};
use axon_risk::RiskEngine;

use crate::capture::{CaptureOutcome, RecordingSource, SessionRecorder};
use crate::config::{ConfigError, RuntimeConfig};
use crate::core::{self, CoreControl};
use crate::dms::{self, now_ms, DeadMansSwitch, DmsPolicy, DmsState};
use crate::handler::CoreHandler;
use crate::health::{CaptureLine, SessionHealth, StatusSnapshot};
use crate::intent::{
    Intent, IntentPoll, IntentQueue, IntentSink, IntentSource, LazyRing, NamedSource, PumpConfig,
    RingIntentSource,
};
use crate::mdring::MdPublisher;
use crate::reconcile::Reconciler;
use crate::selftest;
use crate::shutdown::{graceful_shutdown, ShutdownOptions, ShutdownOutcome};

/// Why a session could not start, or could not finish cleanly.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("venue: {0}")]
    Venue(String),
    #[error(
        "symbol {0:?} is not in the venue's perp universe - subscribing to it would \
         silently receive nothing"
    )]
    UnknownSymbol(String),
    #[error(
        "{0} has no instrument precision at this venue - its tick and lot are unknown, \
         so every order that would add exposure to it is refused"
    )]
    NoInstrumentSpec(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("market-data ring: {0}")]
    MdRing(#[from] axon_ipc::RingError),
    /// Only ever a *startup* failure. Once the session is running, a capture that
    /// cannot write degrades and says so; killing a session with a position on the book
    /// over a log file would trade a recording for an unmanaged position.
    #[error("session capture: {0}")]
    Capture(#[from] axon_replay::ReplayError),
    #[error("the core thread panicked")]
    CorePanicked,
    #[error("event bus: {0}")]
    Bus(String),
}

/// What a finished session leaves behind.
#[derive(Debug)]
pub struct SessionSummary {
    pub status: StatusSnapshot,
    /// `None` for an offline run, which has nothing to unwind.
    pub shutdown: Option<ShutdownOutcome>,
    /// Every intent the offline run planned, in the order it planned them.
    ///
    /// Offline only, and deliberately so: a live session hands its intents to the
    /// submit pipeline and keeping a copy of every one would be an unbounded log
    /// nobody reads. Offline this is the assertion surface — it is what makes
    /// `cargo run --bin axon` proof that the join works rather than proof that it
    /// compiles.
    pub planned: Vec<Intent>,
    /// What the session recording left on disk, or `None` when nobody asked for one.
    ///
    /// Taken *after* the writer thread has drained, so it is the truth about the file
    /// rather than a reading of the queue: the last events of a session are the sweep's
    /// cancel acknowledgements, and a summary that reported before they landed would
    /// under-count the log it is describing.
    pub capture: Option<CaptureOutcome>,
}

/// Run whatever the config describes.
pub fn run(cfg: &RuntimeConfig) -> Result<SessionSummary, RuntimeError> {
    cfg.validate()?;
    if cfg.environment.is_live_wiring() {
        run_live(cfg)
    } else {
        run_offline(cfg)
    }
}

/// Adopt the venue's positions and drive them to zero — the operator cleanup pass.
///
/// **Not a session.** No core thread, no market-data socket, no dead-man's switch, no
/// signal ring, no capture. It reads `/info`, places closes through a *bare* signing
/// client, and reads `/info` again. Each of those absences is a decision:
///
/// - **No dead-man's switch**, because arming one takes a signed action and 60 s of lead
///   to protect a pass that is over in seconds — and the whole point of this pass is that
///   it runs when a session's protection has already lapsed. What it does instead is
///   stronger: it never leaves a marketable order working, and the resting rung is the
///   last one it tries.
/// - **No `HaltableClient`**, because there is nothing to halt: this process places
///   nothing but closes.
/// - **No `GuardedClient`.** The size caps are about *adding* exposure, and every order
///   here is `reduce_only` against a size the venue itself reported. Putting the gate in
///   would reproduce the trap ADR-0031 names — an unpriced instrument or a breached limit
///   is a reason to want a position **gone**, and a gate would refuse exactly then. The
///   rate governor is left out for the same reason `charge_cancel` never refuses: the
///   exit path must not be the thing that runs out of budget.
/// - **No cancel-all**, and this one is the surprise. `cancel_all` on Hyperliquid is
///   account-wide, so sweeping first would pull the resting orders of any session still
///   running — the same hazard that stops the live parity monitor watching a trading
///   session. Working orders are left alone; they are visible in `openOrders` and are an
///   operator's decision, not this pass's.
pub fn run_flatten(
    cfg: &RuntimeConfig,
    policy: crate::flatten::FlattenPolicy,
) -> Result<Vec<crate::flatten::FlattenReport>, RuntimeError> {
    cfg.validate()?;
    cfg.mainnet_allowed()?;
    let account = cfg
        .venue
        .account
        .clone()
        .ok_or_else(|| RuntimeError::Venue("no account address".into()))?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("axon-flatten")
        .build()?;

    // One `meta` read for the asset indices *and* the grids, exactly as a live session
    // does and for the same reason: two reads could disagree after a listing, and an
    // order signed across that seam trades one coin at a size computed for another.
    let universe = rt
        .block_on(axon_provider_hyperliquid::ws::fetch_universe(
            cfg.info_url(),
        ))
        .map_err(|e| RuntimeError::Venue(format!("meta: {e}")))?;
    let map = universe.symbols;
    let instruments = Arc::new(universe.instruments);
    let symbols = resolve_symbols(cfg, &map, &instruments)?;

    let signer = HlSigner::from_env(cfg.is_mainnet())
        .map_err(|e| RuntimeError::Venue(format!("{} ({e})", HlSigner::ENV_KEY)))?;
    let client = ExchangeClient::new(cfg.exchange_url(), signer)
        .with_account(account.as_str(), map.clone())
        // The same grid the planner rounds to. Without it the encoder's own precision
        // check would refuse every order the planner produced, and the refusal would read
        // in the log exactly like a venue rejection.
        .with_instruments(instruments.clone());

    let venue = crate::flatten::HlFlattenVenue {
        info_url: cfg.info_url().to_string(),
        account,
        symbols: map,
    };

    // Event time does not exist here: there is no event stream. The planner needs a `now`
    // to age working orders against and this pass hands it none, so the only thing the
    // stamp does is make each attempt's `cloid` distinct — a **named** wall-clock
    // exception, and the narrowest one in the runtime.
    let now = axon_core::Clock::now_ns(&axon_core::SystemClock);
    let planner_cfg = cfg.intent.planner_config();

    let mut reports = Vec::new();
    for (id, coin) in &symbols {
        println!("── flatten {coin} ({id}) ───────────────────────────────────");
        let report = rt.block_on(crate::flatten::adopt_and_flatten(
            *id,
            &venue,
            &client,
            // No tracker: this process has none, and the adoption exists for a *session*
            // planning behind the pass. `None` keeps the report honest about that — it
            // reports `tracked_before: None` rather than a zero nobody measured.
            None,
            &instruments,
            planner_cfg,
            policy,
            now,
        ));
        println!("{report}");
        reports.push(report);
    }
    Ok(reports)
}

/// The core state every session shares, whatever its adapters.
struct CoreState {
    tracker: Arc<RwLock<OrderTracker>>,
    marks: Arc<MarkCache>,
    health: Arc<SessionHealth>,
    halt: Arc<HaltSwitch>,
    dms_state: Arc<DmsState>,
    /// The declared latency budgets and what has been measured against them.
    ///
    /// Here rather than beside the config because it is *written* from two threads —
    /// `SignalAge` on the deterministic core, the two wall-clock stages on the submit
    /// task — which is the same reason `health` is here (ADR-0036).
    latency: Arc<crate::latency::LatencyBook>,
    /// The loss-based kill switch. Here for the same reason `halt` is: it is read on the
    /// submit path (inside the risk gate) and written on the core thread, so the one
    /// place both can reach it is the state every session shares.
    loss: Arc<axon_execution::LossLimiter>,
    /// The UTC day's equity baseline. `None` unless a daily bound was declared —
    /// absence rather than an in-memory book, so nothing can quietly downgrade a daily
    /// bound into a session one.
    daybook: Option<Arc<crate::daybook::DayBook>>,
}

impl CoreState {
    fn new(cfg: &RuntimeConfig, expire_marks: bool) -> Self {
        // `expire_marks` is the live/offline switch everywhere else in this
        // constructor, and it is the right one here too: a live session inherits a
        // venue's replayed fill history and must not report it as its own P&L, while a
        // replay has no earlier session to inherit from and wants every fill in the log
        // counted (ADR-0036). The clock read is wall time and it is **named**: "did
        // this trade happen before I existed" is a question about the world, and a
        // session's own birth has no event time.
        let mut tracker = OrderTracker::new();
        if expire_marks {
            tracker.set_session_start(axon_core::Clock::now_ns(&axon_core::SystemClock));
        }
        Self {
            tracker: Arc::new(RwLock::new(tracker)),
            marks: Arc::new(if expire_marks {
                MarkCache::with_max_age(cfg.mark_max_age_ns())
            } else {
                // A backtest has no wall clock to age against, and expiring prices by
                // event time over a canned log would make the run's outcome depend on
                // the gaps in the capture rather than on the logic under test.
                MarkCache::never_expires()
            }),
            health: Arc::new(SessionHealth::new(now_ms())),
            halt: Arc::new(HaltSwitch::new()),
            dms_state: Arc::new(DmsState::default()),
            latency: Arc::new(crate::latency::LatencyBook::new(
                [
                    cfg.latency.cause_to_decision_ms,
                    cfg.latency.signal_age_ms,
                    cfg.latency.submit_ack_ms,
                    cfg.latency.decision_to_ack_ms,
                ],
                cfg.latency.breach_warn_pct,
            )),
            loss: Arc::new(axon_execution::LossLimiter::new(
                cfg.strategy.risk.to_loss_limits(),
            )),
            // Only when a daily bound exists. `RuntimeConfig::validate` has already
            // refused a daily bound with no path, so a `Some` bound here always has a
            // path to go with it — and `None` means the day figure is never computed at
            // all rather than computed against a baseline nobody would keep.
            daybook: (cfg.strategy.risk.max_daily_loss > rust_decimal::Decimal::ZERO)
                .then(|| {
                    cfg.strategy
                        .risk
                        .daily_state_path
                        .as_deref()
                        .filter(|p| !p.is_empty())
                        .map(|p| Arc::new(crate::daybook::DayBook::load(p)))
                })
                .flatten(),
        }
    }

    fn handler(&self) -> CoreHandler {
        CoreHandler::new(self.tracker.clone(), self.marks.clone())
    }
}

/// The core handler, with a market-data publisher attached if the config asks for one.
///
/// A failure to create the ring aborts the session rather than degrading it. The signal
/// ring's absence is survivable because that end only *opens* a file Python may not have
/// created yet; this end creates one, so a failure is a path or a permission — nothing
/// a retry fixes, and a session running on with no feature feed is one whose strategy
/// will silently never have an opinion.
fn build_handler(
    cfg: &RuntimeConfig,
    state: &CoreState,
    recorder: Option<&SessionRecorder>,
) -> Result<CoreHandler, RuntimeError> {
    let handler = state.handler();
    let handler = match recorder {
        Some(r) => handler.with_capture(r.tap()),
        None => handler,
    };
    Ok(
        match MdPublisher::open(&cfg.md_ring, cfg.mark_max_age_ns())? {
            Some(md) => handler.with_md_publisher(md),
            None => handler,
        },
    )
}

/// The provenance string stamped into both recorded logs.
///
/// Derived from the config and nothing else — never a path or a hostname. It ends up in
/// `ChainSummary::source`, which a golden comparison diffs, and anything that varies
/// between machines would turn that comparison into a comparison of checkouts.
fn capture_source(cfg: &RuntimeConfig) -> String {
    format!(
        "{:?}/{:?} {} {}",
        cfg.environment,
        cfg.venue.network,
        cfg.venue.name,
        cfg.coins().join(","),
    )
    .to_lowercase()
}

/// Fold a finished recording back into the closing status line.
///
/// The snapshot the core loop returns is taken while the writer thread may still be
/// draining, so the last line an operator sees would otherwise under-report the log by
/// however much was in flight — and under-report a *stop* that happened during the
/// drain, which is the number that actually matters.
fn restamp(status: &mut StatusSnapshot, outcome: &CaptureOutcome, capacity: u64) {
    status.capture = Some(CaptureLine {
        events: outcome.events,
        signals: outcome.signals,
        bytes: outcome.bytes,
        missed: outcome.missed,
        queued: 0,
        capacity,
        stopped: outcome.stopped,
    });
}

/// Resolve the configured instruments against a symbol map, refusing any the venue
/// does not know.
///
/// A coin missing from the universe is not a warning: subscribing to it yields no
/// frames and no error, so the session would look healthy while trading nothing.
///
/// A coin the venue knows but whose **grid** we do not is refused for the same reason
/// one step further on: every order that would add exposure to it is refused at the
/// wire, so the session runs, reconciles, prints OK and never trades that instrument.
/// The check is free — the map and the table come out of one `meta` response — and it
/// moves the failure from the first signal to startup, where an operator is watching.
fn resolve_symbols(
    cfg: &RuntimeConfig,
    map: &SymbolMap,
    instruments: &InstrumentTable,
) -> Result<Vec<(SymbolId, String)>, RuntimeError> {
    cfg.coins()
        .into_iter()
        .map(|coin| {
            let id = map
                .id(&coin)
                .ok_or_else(|| RuntimeError::UnknownSymbol(coin.clone()))?;
            if !instruments.contains(id) {
                return Err(RuntimeError::NoInstrumentSpec(coin));
            }
            Ok((id, coin))
        })
        .collect()
}

// ── offline ──────────────────────────────────────────────────────────────────

/// The offline session: the real core over a canned stream, with no I/O at all.
pub fn run_offline(cfg: &RuntimeConfig) -> Result<SessionSummary, RuntimeError> {
    cfg.validate()?;
    let coins = cfg.coins();
    let map = SymbolMap::from_perps(coins.iter().cloned());
    // Built from the map and *before* the resolve, because the resolve now also refuses
    // a coin with no declared grid. Keyed on `SymbolId` for the same reason everything
    // above the adapter is: nothing up here speaks coin names.
    //
    // Every configured coin, not the two the canned stream touches: this mode has no
    // venue, no socket and no key, so a missing-spec refusal here would be a fiction
    // refusing a config the live path would accept — and it would say so in a sentence
    // that sends an operator to a venue nothing here contacts.
    let instruments = Arc::new(selftest::instruments(
        coins.iter().filter_map(|c| map.id(c)),
    ));
    let symbols = resolve_symbols(cfg, &map, &instruments)?;
    let state = CoreState::new(cfg, false);
    // The *same* table the intent source plans against, declared into the log. A
    // recording that names no grid replays under `Precision::Unconstrained` while the
    // session that wrote it planned under `Precision::Known`, so a golden diff of it
    // shows unrounded prices and disagrees with the run it is reproducing (ADR-0025,
    // ADR-0027). Handing over a *different* table would be worse than handing over
    // none — the log would claim a grid nothing ever planned on.
    let recorder = SessionRecorder::start_with(&cfg.capture, capture_source(cfg), &instruments)?;
    let mut handler = build_handler(cfg, &state, recorder.as_ref())?;

    let (tx, rx) = bus(cfg.session.bus_capacity);
    let first = symbols[0].0;
    let second = symbols.get(1).map(|(s, _)| *s).unwrap_or(first);
    for ev in selftest::events(first, second) {
        tx.send(ev).map_err(|e| RuntimeError::Bus(e.to_string()))?;
    }
    // Dropping the producer and pre-setting the stop flag makes this a bounded,
    // deterministic run: the loop drains exactly what was queued and returns.
    drop(tx);

    let ctl = CoreControl {
        stop: Arc::new(AtomicBool::new(true)),
        poll: Duration::from_micros(cfg.session.core_poll_us),
        status_every: Duration::from_millis(cfg.session.status_interval_ms),
        wall_time: false,
        mode: format!("{:?}/offline", cfg.environment).to_lowercase(),
        symbols,
        health: state.health.clone(),
        halt: state.halt.clone(),
        dms: state.dms_state.clone(),
        governor: None,
        dms_expected: false,
        capture: recorder.as_ref().map(|r| r.tap()),
        // Offline: there is no account, so there is no P&L. The positions this run
        // holds are arithmetic over a canned event stream.
        pnl_expected: false,
        pnl_limits: cfg.pnl,
        // Wired unconditionally, exactly like `latency`: an offline run gets a limiter
        // that can never fire (there is no `pnl` for it to judge), and the code path a
        // live session takes is the one this run takes.
        loss: state.loss.clone(),
        daybook: state.daybook.clone(),
        // Measured, and reported only if something is sampled. Nothing offline
        // reaches the submit edge, so the two wall-clock stages stay empty and the
        // block does not print — which is the honest shape for a run with no venue.
        latency: state.latency.clone(),
    };

    // The join, offline: canned signals through the real reader and the real planner.
    // The sink records instead of submitting, because building a tokio runtime here to
    // drive an async client would cost the property that makes this run worth having —
    // that the deterministic path is provably runtime-free. What reaches a venue from
    // an `Intent` is proven separately, against a spy client, in `intent`'s own tests.
    let mut source = cfg.intent.enabled.then(|| {
        IntentSource::new(
            // The same recording wrapper the live session uses, so what a capture of
            // this run contains is what a capture of a live one would.
            RecordingSource::new(
                axon_strategy::ReplaySource::new(selftest::signals(first, second)),
                recorder.as_ref().map(|r| r.tap()),
            ),
            &cfg.intent,
            instruments.clone(),
            cfg.mark_max_age_ns(),
            state.halt.clone(),
            IntentSink::Record(Vec::new()),
        )
    });
    let mut status = core::run(
        &rx,
        &mut handler,
        &ctl,
        source.as_mut().map(|s| s as &mut dyn IntentPoll),
    );
    let planned = source
        .as_mut()
        .map(|s| s.take_recorded())
        .unwrap_or_default();

    // Closed before the summary is built, so the recording described is the file on
    // disk and not the queue: `finish` drains the writer and renames the log into place.
    let capture = recorder.map(|r| {
        let capacity = r.stats().capacity;
        let outcome = r.finish();
        restamp(&mut status, &outcome, capacity);
        println!(
            "capture: {} events + {} signals -> {}{}",
            outcome.events,
            outcome.signals,
            outcome.log_path.display(),
            match outcome.stopped {
                Some(reason) => format!(" (STOPPED: {reason}, {} missed)", outcome.missed),
                None => String::new(),
            }
        );
        outcome
    });
    println!("{status}");
    // Printed by side and size rather than counted, because "the join produced two
    // intents" is true of a build that sends the target instead of the delta.
    for intent in &planned {
        let coin = ctl
            .symbols
            .iter()
            .find(|(s, _)| *s == intent.symbol_id)
            .map(|(_, n)| n.as_str())
            .unwrap_or("?");
        for id in &intent.plan.cancels {
            println!("intent: seq {} {coin} cancel {id:?}", intent.seq);
        }
        for o in &intent.plan.orders {
            println!(
                "intent: seq {} {coin} {:?} {} @ {:?} {:?}",
                intent.seq, o.side, o.qty, o.price, o.tif
            );
        }
    }
    Ok(SessionSummary {
        status,
        shutdown: None,
        planned,
        capture,
    })
}

// ── live ─────────────────────────────────────────────────────────────────────

/// `Arc<ExchangeClient>` as an [`ExecutionClient`].
///
/// The venue client is needed in two places at once: inside the submit pipeline,
/// which owns what it wraps, and by the dead-man's-switch loop, which calls
/// `scheduleCancel` — an action that is deliberately not part of the venue-agnostic
/// port. Sharing it is cheaper than splitting the client in two.
struct SharedExchange(Arc<ExchangeClient>);

#[async_trait]
impl ExecutionClient for SharedExchange {
    fn capabilities(&self) -> &Capabilities {
        self.0.capabilities()
    }
    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.0.place_order(req).await
    }
    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
        self.0.place_batch(reqs).await
    }
    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
        self.0.cancel(id).await
    }
    async fn cancel_all(&self) -> Result<(), ProviderError> {
        self.0.cancel_all().await
    }
    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        self.0.modify(id, req).await
    }
}

/// Hyperliquid's rate governor as the venue-agnostic submit-path limiter.
///
/// A newtype because the trait belongs to `axon-execution` and the governor to the
/// venue crate — the composition root is the only place entitled to know both. It
/// also owns the clock read: rate windows are wall-clock, and the governor asserts in
/// debug builds that it was not handed the core's nanosecond event time by mistake.
struct HlGovernor(Arc<RateGovernor>);

impl RateLimiter for HlGovernor {
    fn admit_place(&self, orders: u32) -> Result<(), ProviderError> {
        self.0.try_place(orders, now_ms()).into_result().map(|_| ())
    }

    /// Charged, never refused. The venue's cancel allowance is strictly larger than
    /// the place ceiling precisely so an unwind is always possible; declining one here
    /// would hand back the failure that design exists to prevent.
    fn charge_cancel(&self, orders: u32) -> bool {
        let admitted = self.0.try_cancel(orders, now_ms()).is_admitted();
        if !admitted {
            eprintln!("rate: cancel exceeded the local cancel allowance - sending anyway");
        }
        admitted
    }
}

/// One [`NamedSource`] per declared producer, with its ring opened lazily and the
/// session recording teed off it.
///
/// The instrument names in a producer's declared universe are resolved to ids here,
/// against the same `(SymbolId, coin)` list the subscriptions were built from — so a
/// producer's scope and the session's feeds cannot disagree about what a name means.
/// `RuntimeConfig::validate` has already refused any name that is not in
/// `strategy.symbols`, so an unresolvable one at this point is impossible rather than
/// merely unlikely; it is dropped instead of panicking, and the producer is then scoped
/// to *fewer* instruments than it asked for, which is the direction with no exposure in
/// it.
fn producer_sources(
    cfg: &RuntimeConfig,
    symbols: &[(SymbolId, String)],
    recorder: Option<&SessionRecorder>,
) -> Vec<NamedSource<RecordingSource<LazyRing>>> {
    cfg.producers()
        .into_iter()
        .map(|p| NamedSource {
            symbols: p
                .symbols
                .iter()
                .filter_map(|name| {
                    symbols
                        .iter()
                        .find(|(_, coin)| crate::config::coin_matches(coin, name))
                        .map(|(id, _)| id.get())
                })
                .collect(),
            policy: p.policy(),
            max_gross_notional: p.max_gross_notional,
            source: RecordingSource::new(
                LazyRing::new(&p.ring_path, cfg.intent.attach_retry_ms),
                recorder.map(|r| r.tap()),
            ),
            name: p.name,
        })
        .collect()
}

/// The full submit path, in the order the wrappers must nest.
type SubmitPipeline =
    HaltableClient<GuardedClient<GovernedClient<SharedExchange, HlGovernor>, TrackerRiskContext>>;

fn build_pipeline(
    cfg: &RuntimeConfig,
    exchange: Arc<ExchangeClient>,
    governor: Arc<RateGovernor>,
    state: &CoreState,
) -> SubmitPipeline {
    // Innermost first: rate below risk, because a risk check is free and rate budget
    // is not, so an order risk will refuse must not spend any of it. Halt outermost,
    // because when protection is gone there is nothing left worth checking.
    let governed = GovernedClient::new(SharedExchange(exchange), HlGovernor(governor));
    let guarded = GuardedClient::new(
        governed,
        RiskEngine::new(cfg.strategy.risk.to_limits()),
        TrackerRiskContext::new(state.tracker.clone(), state.marks.clone()),
    )
    // The money bound, inside the same wrapper as the size bounds and for the same
    // structural reason the whole pipeline is types rather than call-site discipline: a
    // gate a caller is merely expected to consult is one forgotten call site away from
    // the thing it exists to prevent. It sits *here* rather than as a fourth wrapper
    // because it needs the position to tell an order that adds exposure from one that
    // removes it, and refusing the second is the trap ADR-0031 named for cancels.
    .with_loss_limiter(state.loss.clone())
    // The across-symbol bound, in the same wrapper and on the same argument (ADR-0038).
    // The intent source also *scales* targets to fit it, and that is convergence rather
    // than a guarantee: a scaler is arithmetic a call site can get wrong or bypass, and
    // this is the type that cannot be gone around. Ten instruments each inside their own
    // `max_notional` are inside every limit `RiskEngine` can see.
    .with_portfolio(PortfolioEngine::new(cfg.portfolio.to_limits()));
    HaltableClient::new(guarded, state.halt.clone())
}

/// The live session.
pub fn run_live(cfg: &RuntimeConfig) -> Result<SessionSummary, RuntimeError> {
    cfg.validate()?;
    cfg.mainnet_allowed()?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("axon-edge")
        .build()?;

    // Symbol resolution is a network read but not a trading action, so it happens
    // before any key is touched: a typo'd symbol should fail here, not after we have
    // authenticated and armed a switch.
    // One `meta` read for both halves — the asset indices and each instrument's tick
    // and lot. Two reads could disagree after a listing, and an order signed across
    // that seam trades one coin at a size computed for another.
    let universe = rt
        .block_on(axon_provider_hyperliquid::ws::fetch_universe(
            cfg.info_url(),
        ))
        .map_err(|e| RuntimeError::Venue(format!("meta: {e}")))?;
    let map = universe.symbols;
    // Built once, here, and handed to *both* the planner and the venue client. Two
    // tables that can drift apart would have the planner rounding to one grid while the
    // encoder refuses against another — a session-wide outage that reads in the log
    // like a venue rejection.
    let instruments = Arc::new(universe.instruments);
    let symbols = resolve_symbols(cfg, &map, &instruments)?;

    let state = CoreState::new(cfg, true);
    // The venue's own grid, from the one `meta` read above, declared into the log for
    // the same reason as offline — and it matters more here, because this is the only
    // path that produces a recording of a market nobody can re-fetch.
    let recorder = SessionRecorder::start_with(&cfg.capture, capture_source(cfg), &instruments)?;
    let governor = Arc::new(RateGovernor::with_config(cfg.governor_config()));
    let (tx, rx) = bus(cfg.session.bus_capacity);

    // Start halted: until the first re-arm succeeds there is no venue-side protection,
    // and an order placed in that window can outlive the process that placed it.
    if cfg.safety.dead_mans_switch {
        state.halt.halt();
    }

    let ctl = Arc::new(CoreControl {
        stop: Arc::new(AtomicBool::new(false)),
        poll: Duration::from_micros(cfg.session.core_poll_us),
        status_every: Duration::from_millis(cfg.session.status_interval_ms),
        wall_time: true,
        mode: format!(
            "{}/{}",
            format!("{:?}", cfg.environment).to_lowercase(),
            format!("{:?}", cfg.venue.network).to_lowercase()
        ),
        symbols: symbols.clone(),
        health: state.health.clone(),
        halt: state.halt.clone(),
        dms: state.dms_state.clone(),
        governor: Some(governor.clone()),
        dms_expected: cfg.safety.dead_mans_switch,
        capture: recorder.as_ref().map(|r| r.tap()),
        // There is a venue, so there is an account and a P&L — whether or not this
        // session is allowed to place an order. A read-only session reporting a flat
        // zero is a *result*, and it is the one that proves the monitor is wired.
        pnl_expected: true,
        pnl_limits: cfg.pnl,
        loss: state.loss.clone(),
        daybook: state.daybook.clone(),
        latency: state.latency.clone(),
    });

    // The intent source lives on the core thread, because that is where the reader and
    // the planner have to run for a replayed session to produce the same orders. Only
    // the queue crosses to the edge.
    let queue = IntentQueue::new(cfg.intent.queue_capacity);
    let mut source: Option<RingIntentSource> = cfg.intent.enabled.then(|| {
        IntentSource::multi(
            // One reader per declared producer, and exactly one for a session that
            // declared none — `RuntimeConfig::producers` resolves both into the same
            // shape, so there is no branch here between one strategy and several
            // (ADR-0038).
            //
            // The recording tees off each ring itself, not off the accepted records: a
            // log missing the expired and out-of-order ones would replay with every
            // refusal counter at zero.
            producer_sources(cfg, &symbols, recorder.as_ref()),
            &cfg.intent,
            cfg.portfolio.to_limits(),
            cfg.portfolio.overlap.into(),
            cfg.portfolio.scale_to_fit,
            instruments.clone(),
            cfg.mark_max_age_ns(),
            state.halt.clone(),
            queue.sink(),
        )
        // The session's own book, so the pass's signal ages and the edge's round trips
        // land in one place. Without this the core would fill a private book nobody
        // reads and the status line would report an empty stage on a busy session.
        .with_latency(state.latency.clone())
    });

    let mut handler = build_handler(cfg, &state, recorder.as_ref())?;
    let core_ctl = ctl.clone();
    let core_thread = std::thread::Builder::new()
        .name("axon-core".into())
        .spawn(move || {
            core::run(
                &rx,
                &mut handler,
                &core_ctl,
                source.as_mut().map(|s| s as &mut dyn IntentPoll),
            )
        })?;

    let outcome = rt.block_on(supervise(
        cfg,
        &map,
        instruments,
        &symbols,
        tx,
        &state,
        governor,
        &queue,
    ));

    // Stop the core only after the supervisor has finished unwinding, so the cancel
    // acknowledgements from the sweep are still applied before the loop exits.
    ctl.stop();
    let mut status = core_thread.join().map_err(|_| RuntimeError::CorePanicked)?;

    // Only now: the core thread has exited, so the handler and the intent source that
    // held taps are gone and nothing else will hand a record over. Closing earlier would
    // truncate the log at the sweep, which is the part of a session a post-mortem always
    // wants.
    let capture = recorder.map(|r| {
        let capacity = r.stats().capacity;
        let outcome = r.finish();
        restamp(&mut status, &outcome, capacity);
        match outcome.stopped {
            None => println!(
                "capture: {} events + {} signals -> {}",
                outcome.events,
                outcome.signals,
                outcome.log_path.display()
            ),
            Some(reason) => eprintln!(
                "capture: STOPPED ({reason}) after {} events, {} missed - {} is a prefix \
                 of this session",
                outcome.events,
                outcome.missed,
                outcome.log_path.display()
            ),
        }
        outcome
    });
    println!("{status}");

    Ok(SessionSummary {
        status,
        shutdown: Some(outcome?),
        planned: Vec::new(),
        capture,
    })
}

/// The async edge: spawn the feeds, the safety loop and the reconciliation poll, wait
/// for a reason to stop, then unwind in order.
#[allow(clippy::too_many_arguments)]
async fn supervise(
    cfg: &RuntimeConfig,
    map: &SymbolMap,
    instruments: Arc<InstrumentTable>,
    symbols: &[(SymbolId, String)],
    tx: EventSender,
    state: &CoreState,
    governor: Arc<RateGovernor>,
    queue: &IntentQueue,
) -> Result<ShutdownOutcome, RuntimeError> {
    let account = cfg
        .venue
        .account
        .clone()
        .ok_or_else(|| RuntimeError::Venue("no account address".into()))?;
    let coins: Vec<String> = symbols.iter().map(|(_, c)| c.clone()).collect();

    // The key never leaves this scope as anything but a signer (ADR-0009).
    let signer = HlSigner::from_env(cfg.is_mainnet())
        .map_err(|e| RuntimeError::Venue(format!("{} ({e})", HlSigner::ENV_KEY)))?;
    let exchange = Arc::new(
        ExchangeClient::new(cfg.exchange_url(), signer)
            .with_account(account.as_str(), map.clone())
            // The same `Arc` the intent source got: the encoder's refusal is only a
            // tripwire while it is checking against the grid the planner rounded to.
            .with_instruments(instruments),
    );
    let pipeline: Arc<dyn ExecutionClient> = Arc::new(build_pipeline(
        cfg,
        exchange.clone(),
        governor.clone(),
        state,
    ));

    // Market data and the account-scoped user channels share one socket, so a fill and
    // the book update that caused it stay orderable against each other (ADR-0008).
    let md = Arc::new(HyperliquidMarketData::new(
        cfg.ws_url(),
        map.clone(),
        coins.clone(),
        tx.clone(),
    ));
    for feed in &cfg.session.feeds {
        for coin in &coins {
            md.subscribe_coin(*feed, coin);
        }
    }
    md.subscribe_user_channels(account.as_str());

    let stop_dms = Arc::new(tokio::sync::Notify::new());
    let mut dms_handle = tokio::spawn({
        let exchange = exchange.clone();
        let halt = state.halt.clone();
        let health = state.health.clone();
        let dms_state = state.dms_state.clone();
        let stop = stop_dms.clone();
        let enabled = cfg.safety.dead_mans_switch;
        let mut policy = DmsPolicy::new(cfg.safety.lead_ms, cfg.safety.rearm_interval_ms);
        async move {
            if !enabled {
                // A read-only session still needs this task to exist so the shutdown
                // path has one shape, but it must not spend actions arming a switch
                // for orders that will never be placed.
                stop.notified().await;
                return;
            }
            dms::run(&*exchange, &mut policy, &dms_state, &halt, &health, &stop).await;
        }
    });

    let mut reconciler = Reconciler::new(
        cfg.info_url(),
        account.as_str(),
        map.clone(),
        symbols.iter().map(|(s, _)| *s).collect(),
        state.tracker.clone(),
        governor,
        tx,
        state.health.clone(),
        (cfg.reconcile.grace_ms as i64).saturating_mul(1_000_000),
        cfg.reconcile.rate_limit_every,
    );

    // One reconciliation before the feeds start: the orders a previous process left
    // resting are exposure this one is already carrying, and the risk gate has to
    // count them before anything new is admitted.
    match reconciler.cycle().await {
        Some(Ok(r)) => {
            state.health.note_reconcile_ok(now_ms());
            state.health.note_adopted(r.unknown_to_tracker.len() as u64);
            println!(
                "startup reconcile: {} open at the venue, {} adopted",
                r.venue_open,
                r.unknown_to_tracker.len()
            );
        }
        Some(Err(e)) => {
            state.health.note_reconcile_failure();
            eprintln!("startup reconcile failed: {e}");
        }
        None => eprintln!("startup reconcile skipped by the rate governor"),
    }

    // The submit task starts only after the startup reconcile, so the first plan is
    // computed against a tracker that already knows what a previous process left
    // resting. Starting it earlier would let the session's first order be sized
    // against exposure it had not yet counted.
    let stop_pump = Arc::new(tokio::sync::Notify::new());
    let pump_task = cfg.intent.enabled.then(|| {
        let pipeline = pipeline.clone();
        let stop = stop_pump.clone();
        let pcfg = PumpConfig {
            rx: queue.receiver(),
            inflight: queue.inflight(),
            halt: state.halt.clone(),
            tracker: state.tracker.clone(),
            health: state.health.clone(),
            latency: state.latency.clone(),
            poll: Duration::from_millis(cfg.intent.submit_poll_ms),
        };
        tokio::spawn(async move { crate::intent::pump(pipeline, pcfg, &stop).await })
    });

    let ws = tokio::spawn({
        let md = md.clone();
        async move { md.run_forever().await }
    });
    let stop_reconcile = Arc::new(tokio::sync::Notify::new());
    let reconcile_task = tokio::spawn({
        let stop = stop_reconcile.clone();
        let interval = Duration::from_millis(cfg.reconcile.interval_ms);
        async move { reconciler.run(interval, &stop).await }
    });

    println!(
        "session up: {} coins, dms {} (lead {} ms, re-arm {} ms, {} actions/day), reconcile every {} ms, intents {}",
        coins.len(),
        if cfg.safety.dead_mans_switch { "on" } else { "OFF" },
        cfg.safety.lead_ms,
        cfg.safety.rearm_interval_ms,
        cfg.dms_actions_per_day(),
        cfg.reconcile.interval_ms,
        if cfg.intent.enabled {
            &cfg.ipc.signal_ring_path
        } else {
            "OFF (read-only session)"
        },
    );

    // Two ways out: an operator asks us to stop, or the safety loop reports that
    // protection is gone and returns.
    let mut dms_dead = false;
    tokio::select! {
        _ = wait_for_signal() => println!("shutdown: signal received"),
        _ = &mut dms_handle => {
            dms_dead = true;
            eprintln!("shutdown: the dead-man's-switch loop gave up - unwinding");
        }
    }

    // Step 1 of the shutdown sequence, and it has to happen before the sweep rather
    // than inside it: `cancel_all` is a read-then-write, so an order admitted after the
    // read survives it. Stopping the submitter first is what makes "we left nothing
    // behind" true rather than probable (see [`crate::shutdown`]).
    let mut submitter_stopped = true;
    if let Some(mut task) = pump_task {
        stop_pump.notify_one();
        // The notify is only heard between intents; a `place_order` on a stalled
        // connection is inside `submit_intent` and hears nothing. Dropping the handle
        // *detaches* the task rather than cancelling it, so that placement would go on
        // to complete after the sweep, rest at the venue, and outlive a process whose
        // dead-man's switch had already been disarmed. Aborting is the only thing that
        // stops it, and it is still not proof — hence the flag.
        if tokio::time::timeout(Duration::from_secs(5), &mut task)
            .await
            .is_err()
        {
            task.abort();
            submitter_stopped = false;
            state.health.note_submitter_abandoned();
            eprintln!(
                "shutdown: the submit task did not stop within 5 s and was aborted - an \
                 order it had already sent may still reach the venue, so the dead-man's \
                 switch stays armed"
            );
        }
    }

    let outcome = graceful_shutdown(
        &state.halt,
        pipeline.as_ref(),
        cfg.safety
            .dead_mans_switch
            .then(|| &*exchange as &dyn DeadMansSwitch),
        ShutdownOptions {
            submitter_stopped,
            ..ShutdownOptions::default()
        },
    )
    .await;
    println!("shutdown: {outcome:?}");

    // The safety loop dies last, and only once the sweep has decided whether the
    // venue-side deadline should stand. Anything else can be aborted at will; this one
    // is the protection itself.
    if !dms_dead {
        stop_dms.notify_one();
        let _ = tokio::time::timeout(Duration::from_secs(10), dms_handle).await;
    }
    stop_reconcile.notify_one();
    let _ = tokio::time::timeout(Duration::from_secs(10), reconcile_task).await;
    // The socket is kept alive through the sweep so its cancel confirmations reach the
    // tracker; only now is there nothing left to hear.
    ws.abort();

    Ok(outcome)
}

/// Wait for SIGINT or SIGTERM.
///
/// A second Ctrl-C exits immediately: an operator who asks twice has decided the
/// orderly path is not working, and the venue-side switch is still armed behind us —
/// which is exactly the situation it was armed for.
#[cfg(unix)]
async fn wait_for_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
    tokio::spawn(async {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("second interrupt - exiting now, leaving the dead-man's switch armed");
        std::process::exit(130);
    });
}

/// SIGTERM is Unix-only; elsewhere Ctrl-C is the whole story.
#[cfg(not(unix))]
async fn wait_for_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Environment, Network};
    use axon_core::Decimal;
    use rust_decimal_macros::dec;

    fn offline_cfg() -> RuntimeConfig {
        RuntimeConfig::default()
    }

    /// A per-test ring path. Tests in one binary run in parallel and the publisher
    /// *truncates* what it creates, so a shared path would have one test zeroing
    /// another's ring mid-run.
    fn md_temp_path(tag: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir()
            .join(format!(
                "axon-session-md-{tag}-{}-{}.ring",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn the_offline_session_runs_end_to_end_with_no_network_and_no_key() {
        // The hard constraint on this whole crate: `cargo run --bin axon` and
        // `./run.sh runtime` must work on a machine with no key and no route to the
        // venue, and must exit rather than hang.
        std::env::remove_var(axon_provider_hyperliquid::HlSigner::ENV_KEY);
        let summary = run(&offline_cfg()).expect("the offline session must run");
        assert_eq!(summary.status.events, 7);
        assert!(summary.shutdown.is_none());
        assert!(
            summary.status.warnings().is_empty(),
            "a clean offline run reports nothing wrong: {}",
            summary.status
        );
    }

    #[test]
    fn the_offline_run_proves_the_whole_fan_out_not_just_the_config() {
        // Each assertion here is a different consumer of the same event stream: the
        // book, the mark cache's precedence rule, and the tracker's adoption + fill
        // accounting. If any of them were unwired, the run would still "succeed".
        let summary = run_offline(&offline_cfg()).expect("offline");
        let s = &summary.status;
        assert_eq!(s.marks_fresh, 2, "both symbols priced: {s}");
        assert!(s.stale_marks.is_empty());
        assert_eq!(
            s.positions,
            vec![("BTC".to_string(), dec!(0.1))],
            "the adopted order's fill moved the position"
        );
        assert_eq!(
            s.open_orders, 1,
            "the filled order closed out; the second adopted one is still resting"
        );
        assert_eq!(s.orphan_fills, 0, "the fill was attributed, not orphaned");
    }

    #[test]
    fn the_offline_run_proves_the_intent_source_not_only_the_fan_out() {
        // The Phase-4 join, offline: canned signals through the real reader, the real
        // planner and the session's own tracker and book. Before this existed the
        // submit pipeline was fully built and nothing in the runtime called it, so a
        // green test suite proved everything except that Python could trade.
        let summary = run_offline(&offline_cfg()).expect("offline");
        assert_eq!(summary.planned.len(), 2, "one intent per instrument");

        // BTC: we hold 0.1 and the target is 0.3 — the order is the *delta*. Anything
        // that sends the target instead ends up long 0.4 and compounds from there.
        let btc = &summary.planned[0];
        assert_eq!(btc.seq, 2, "the stale first signal never got here");
        let o = &btc.plan.orders[0];
        assert_eq!(o.qty, dec!(0.2));
        assert_eq!(o.side, axon_core::Side::Buy);
        assert_eq!(o.price, Some(dec!(49999)), "urgency 0 joins the bid");
        assert_eq!(o.tif, axon_core::Tif::PostOnly);
        assert_eq!(
            btc.plan.cancels.len(),
            1,
            "the order resting against the superseded target is pulled first"
        );

        // ETH has no BBO in the canned stream, so this only exists if the fallback to
        // the L2 book works.
        let eth = &summary.planned[1];
        let o = &eth.plan.orders[0];
        assert_eq!(o.qty, dec!(1));
        assert_eq!(o.side, axon_core::Side::Sell);
        assert_eq!(o.price, Some(dec!(2999)), "urgency 2 crosses to the bid");

        let line = summary.status.intent.expect("counters are surfaced");
        assert_eq!(line.accepted, 2);
        assert_eq!(line.rejected, 1, "the 500 ms signal was five seconds late");
        assert_eq!(line.expired, 1);
        assert!(line.attached);
    }

    #[test]
    fn a_read_only_session_plans_nothing_at_all() {
        // The only supported way to run a session that cannot place an order. It has to
        // be visible as absence, not as a healthy-looking zero.
        let mut cfg = offline_cfg();
        cfg.intent.enabled = false;
        let summary = run_offline(&cfg).expect("offline");
        assert!(summary.planned.is_empty());
        assert!(summary.status.intent.is_none());
    }

    #[test]
    fn the_offline_session_is_deterministic() {
        // Same input, same output, twice — the property the parity harness will lean
        // on, asserted at the level of a whole session rather than one handler.
        let a = run_offline(&offline_cfg()).unwrap();
        let b = run_offline(&offline_cfg()).unwrap();
        assert_eq!(a.status.events, b.status.events);
        assert_eq!(a.status.positions, b.status.positions);
        assert_eq!(a.status.marks_fresh, b.status.marks_fresh);
        assert_eq!(a.status.open_orders, b.status.open_orders);
        // Including the join: same orders, same cancels, same cloids. Without this the
        // parity harness could only ever compare market data.
        assert_eq!(
            a.planned, b.planned,
            "a replayed session must plan byte-identical intents"
        );
    }

    #[test]
    fn the_offline_session_writes_no_market_data_ring_unless_asked() {
        // The hard constraint on the default path: `cargo run --bin axon` opens no
        // socket, needs no key — and creates no file. A publisher that started unasked
        // would truncate whatever sits at the configured path.
        let cfg = offline_cfg();
        assert!(!cfg.md_ring.enabled);
        let _ = std::fs::remove_file(&cfg.md_ring.path);
        let summary = run_offline(&cfg).expect("offline");
        assert!(summary.status.md.is_none(), "absence, not zeros");
        assert!(
            !std::path::Path::new(&cfg.md_ring.path).exists(),
            "the default run created {}",
            cfg.md_ring.path
        );
    }

    #[test]
    fn an_offline_session_that_is_asked_publishes_what_its_events_carried() {
        // The Rust→Python direction, end to end through the real session: the canned
        // stream's quote and book snapshot come out as slices Python can read. Before
        // this the md ring's only producer was an example program (ADR-0012).
        let mut cfg = offline_cfg();
        cfg.md_ring.enabled = true;
        cfg.md_ring.path = md_temp_path("offline");
        cfg.md_ring.capacity = 16;
        let summary = run_offline(&cfg).expect("offline");

        let line = summary.status.md.expect("the publisher is reported");
        assert_eq!(line.dropped, 0);
        assert_eq!(line.unrepresentable, 0);
        assert_eq!(line.capacity, 16);

        let c = axon_ipc::MdConsumer::open(&cfg.md_ring.path).expect("open the published ring");
        let mut got = Vec::new();
        while let Some(s) = c.try_pop() {
            got.push(s);
        }
        drop(c);
        let _ = std::fs::remove_file(&cfg.md_ring.path);

        assert_eq!(got.len() as u64, line.published);
        // The BTC quote (event 2) and the ETH book snapshot (event 3). The BTC ticker
        // publishes nothing: it moves no field the record has, and it is the one event
        // in the stream that a live venue would not timestamp.
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].seq, 0);
        assert_eq!(got[0].symbol_id, 0);
        assert_eq!(got[0].kind, axon_contracts::MD_KIND_QUOTE);
        assert_eq!(got[0].ts_event, 2_000_000_000, "the event's own time");
        assert_eq!(
            got[0].bid_px,
            axon_strategy::decimal_to_fixed(dec!(49_999)).unwrap()
        );
        assert_eq!(got[1].seq, 1, "gap-free: nothing was dropped");
        assert_eq!(got[1].symbol_id, 1);
        assert_eq!(got[1].kind, axon_contracts::MD_KIND_SNAPSHOT);
        assert_eq!(
            got[1].ask_px,
            axon_strategy::decimal_to_fixed(dec!(3_001)).unwrap()
        );
    }

    #[test]
    fn a_published_session_is_still_deterministic_byte_for_byte() {
        // The parity harness compares a replay against a live capture record by record.
        // If the published ring depended on anything but the event stream, the two would
        // differ for reasons nobody could attribute.
        fn run_to_bytes(tag: &str) -> Vec<u8> {
            let mut cfg = offline_cfg();
            cfg.md_ring.enabled = true;
            cfg.md_ring.path = md_temp_path(tag);
            cfg.md_ring.capacity = 16;
            run_offline(&cfg).expect("offline");
            let bytes = std::fs::read(&cfg.md_ring.path).expect("read the ring");
            let _ = std::fs::remove_file(&cfg.md_ring.path);
            bytes
        }
        assert_eq!(run_to_bytes("det-a"), run_to_bytes("det-b"));
    }

    /// A path nothing can create, on any platform: its parent is an existing *file*.
    ///
    /// `/nonexistent-dir-axon/...` only refuses where the filesystem root is unwritable.
    /// On Windows it resolves to `C:\nonexistent-dir-axon\...`, which the build user can
    /// create — and `capture.rs` does exactly that, via `create_dir_all`. So the capture
    /// test below was manufacturing the very directory this one needs to be absent, and
    /// the two failed together on Windows and nowhere else. Anchoring under a file is
    /// portable (a file is not a directory anywhere) and cannot be conjured by a test
    /// running beside this one.
    fn blocked_path(tag: &str, leaf: &str) -> (std::path::PathBuf, String) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let blocker = std::env::temp_dir().join(format!(
            "axon-not-a-dir-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&blocker, b"not a directory").expect("write the blocking file");
        // Component by component, so the caller's `/` does not travel into a path on a
        // platform that does not spell separators that way.
        let mut inside = blocker.clone();
        for part in leaf.split('/') {
            inside.push(part);
        }
        (blocker, inside.to_string_lossy().into_owned())
    }

    #[test]
    fn a_market_data_ring_that_cannot_be_created_refuses_to_start_the_session() {
        // Degrading here would hand back the exact gap the publisher exists to close: a
        // session that reports OK while Python is being fed nothing.
        let mut cfg = offline_cfg();
        cfg.md_ring.enabled = true;
        let (blocker, path) = blocked_path("mdring", "md.ring");
        cfg.md_ring.path = path;
        let err = run_offline(&cfg).unwrap_err();
        let _ = std::fs::remove_file(&blocker);
        assert!(matches!(err, RuntimeError::MdRing(_)), "{err:?}");
    }

    /// A per-test capture path, for the same reason the ring tests need one: the
    /// recorder creates and truncates what it names, and tests in one binary run in
    /// parallel.
    fn capture_temp_path(tag: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir()
            .join(format!(
                "axon-session-cap-{tag}-{}-{}.jsonl",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ))
            .to_string_lossy()
            .into_owned()
    }

    fn capturing_cfg(tag: &str) -> RuntimeConfig {
        let mut cfg = offline_cfg();
        cfg.capture.enabled = true;
        cfg.capture.path = capture_temp_path(tag);
        cfg
    }

    fn remove_capture(out: &crate::capture::CaptureOutcome) {
        let _ = std::fs::remove_file(&out.log_path);
        if let Some(p) = &out.signal_path {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Replay a captured session through the production chain: the same `CoreHandler`
    /// fan-out and the same `IntentSource` the session itself ran, driven by
    /// `axon-replay`'s bus.
    ///
    /// It reimplements neither. What it owns is the *schedule* — a pass after every
    /// event, exactly where the core loop runs one — and that is the only reason this is
    /// not simply `run_offline` again. It does **not** sleep: `IntentSource` paces its
    /// passes on the event clock, so the round trip holds at whatever speed the log
    /// drains. It used to park a millisecond per event, and that was not a detail of the
    /// harness — it was the only thing making the property true, and it is not available
    /// to anything replaying a real capture.
    struct ReplayProbe {
        core: CoreHandler,
        intent: IntentSource<crate::capture::CapturedSignals>,
        planned: Vec<Intent>,
    }

    impl axon_core::EventHandler for ReplayProbe {
        fn on_event(&mut self, ts_event: axon_core::Nanos, event: &axon_core::Event) {
            self.core.on_event(ts_event, event);
            // After the event has been applied, so the position, the book and the
            // working orders the plan is computed against are the state that event left
            // behind.
            self.intent.poll(self.core.last_ts(), 0, &self.core);
            self.planned.extend(self.intent.take_recorded());
        }
    }

    fn replay_captured(cfg: &RuntimeConfig, out: &crate::capture::CaptureOutcome) -> ReplayProbe {
        let signals = axon_replay::SignalLog::open(out.signal_path.as_ref().unwrap())
            .expect("the recorded signal log must parse");
        let mut probe = ReplayProbe {
            core: CoreHandler::new(
                Arc::new(RwLock::new(OrderTracker::new())),
                // The offline session's own cache, so the replay ages marks the way the
                // run being reproduced did.
                Arc::new(axon_execution::MarkCache::never_expires()),
            ),
            intent: IntentSource::new(
                crate::capture::CapturedSignals::from_log(&signals),
                &cfg.intent,
                // The session being replayed declared these, so the replay has to as
                // well: a replay rounding against a different grid would plan different
                // orders and the divergence would be the harness's, not the code's.
                Arc::new(selftest::instruments([SymbolId::new(0), SymbolId::new(1)])),
                cfg.mark_max_age_ns(),
                Arc::new(axon_execution::HaltSwitch::new()),
                IntentSink::Record(Vec::new()),
            ),
            planned: Vec::new(),
        };
        let src = axon_replay::ReplaySource::open(&out.log_path)
            .expect("the recorded event log must parse")
            // The order the session actually experienced, including any late arrival.
            // Event-time order would answer a counterfactual (ADR-0018 §4).
            .with_order(axon_replay::ReplayOrder::AsCaptured);
        src.run(&axon_core::ManualClock::new(0), &mut probe);
        probe
    }

    #[test]
    fn a_captured_offline_session_replays_to_the_state_it_recorded() {
        // The round trip, closed. Before this, capture had only ever been driven by its
        // own crate's tests: no session had recorded itself, so no session had ever been
        // replayed and the golden harness had only seen logs written by a generator.
        //
        // Both halves are asserted, because either alone would pass over a broken
        // recording: the market-data and reconciliation state proves the *event* log,
        // and the planned orders — same sides, same sizes, same limits, same cloids —
        // prove the *signal* log, which is the half that lets a replay re-decide rather
        // than merely re-observe.
        let cfg = capturing_cfg("roundtrip");
        let summary = run_offline(&cfg).expect("offline");
        let out = summary
            .capture
            .as_ref()
            .expect("the session recorded itself");
        assert!(out.complete(), "{out:?}");
        assert_eq!(
            out.events, summary.status.events,
            "every event was recorded"
        );
        assert_eq!(out.signals, 3, "including the one the reader then expired");

        let probe = replay_captured(&cfg, out);
        assert_eq!(
            probe.core.events(),
            summary.status.events,
            "the replay saw the whole session"
        );
        assert_eq!(
            probe.core.positions(&[SymbolId::new(0), SymbolId::new(1)]),
            vec![(SymbolId::new(0), dec!(0.1))],
            "the captured fill moved the replayed position"
        );
        assert_eq!(
            probe.core.tracker().read().unwrap().open_count(),
            summary.status.open_orders
        );
        assert_eq!(probe.core.marks().get(SymbolId::new(0)), Some(dec!(50000)));
        assert_eq!(probe.core.marks().get(SymbolId::new(1)), Some(dec!(3000)));
        assert_eq!(
            probe.planned, summary.planned,
            "a replay of a recorded session must plan byte-identical intents"
        );
        let stats = probe.intent.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(
            stats.expired, 1,
            "the stale signal expired again - the release times are what make that \
             reproducible at all"
        );
        remove_capture(out);
    }

    #[test]
    fn a_recording_declares_the_grid_the_session_it_records_planned_against() {
        // The half of ADR-0025's precision hole that lives at the composition root, and
        // it is the kind that is green in every test until somebody diffs a golden. A
        // log that names no grid replays under `Precision::Unconstrained` while the
        // session that wrote it planned under `Precision::Known`, so the replay's prices
        // are *unrounded* and its diff against the run it is reproducing shows the
        // rounding as a strategy change. Nothing about the session looks wrong; the
        // artifact is simply describing a market with no ticks in it.
        let cfg = capturing_cfg("grid");
        let summary = run_offline(&cfg).expect("offline");
        let out = summary.capture.as_ref().expect("recorded");
        let log = axon_replay::EventLog::open(&out.log_path).expect("parse");
        let declared = log.instruments();
        assert!(
            declared.is_declared(),
            "a session knows its grid; a log that shrugs is the whole bug: {declared:?}"
        );
        // The same table the intent source planned against, not a second one built here:
        // a recording that claims a grid the session never used is worse than one that
        // claims none, because the diff then looks correct and is not.
        assert_eq!(declared.len(), cfg.coins().len());
        remove_capture(out);
    }

    #[test]
    fn a_session_that_was_not_asked_to_record_writes_nothing_at_all() {
        // The default path stays a process that touches nothing: no socket, no key, no
        // ring — and no log. A recorder that started unasked would also truncate whatever
        // sits at the configured path.
        let cfg = offline_cfg();
        assert!(!cfg.capture.enabled);
        let log = std::path::Path::new(&cfg.capture.path);
        let _ = std::fs::remove_file(log);
        let summary = run_offline(&cfg).expect("offline");
        assert!(summary.capture.is_none(), "absence, not an empty recording");
        assert!(summary.status.capture.is_none(), "absence, not zeros");
        assert!(!log.exists(), "the default run created {}", log.display());
        assert!(
            !std::path::Path::new(&format!("{}.partial", cfg.capture.path)).exists(),
            "not even a partial"
        );
    }

    #[test]
    fn a_capture_that_stopped_leaves_no_log_at_the_path_a_harness_would_replay() {
        // The worst artifact available here is a truncated log that parses: the replay of
        // it is green over a session that ended early, and nothing downstream can tell.
        // The rename is what makes that impossible — and the status line says so, so the
        // operator does not have to notice a missing file.
        let mut cfg = capturing_cfg("capped");
        cfg.capture.max_bytes = 1;
        let summary = run_offline(&cfg).expect("a stopped recording never stops the session");
        let out = summary.capture.as_ref().unwrap();

        assert_eq!(out.stopped, Some(crate::capture::CaptureStop::SizeCap));
        assert_eq!(
            summary.status.events, 7,
            "the session itself ran to the end regardless"
        );
        assert!(!std::path::Path::new(&cfg.capture.path).exists());
        let warnings = summary.status.warnings();
        assert!(
            warnings.iter().any(|w| w.starts_with("CAPTURE STOPPED")),
            "a recording that stopped has to reach the operator: {warnings:?}"
        );
        assert!(
            axon_replay::ReplaySource::open(&cfg.capture.path).is_err(),
            "and a replay pointed at the path it would have had finds nothing"
        );
        remove_capture(out);
    }

    #[test]
    fn a_capture_that_cannot_be_created_refuses_to_start_the_session() {
        // Startup only. Nothing has been lost yet and the operator is watching, so a bad
        // path is a two-second fix now rather than a soak with no artifact at the end.
        let mut cfg = offline_cfg();
        cfg.capture.enabled = true;
        // Not a merely-absent directory: the recorder creates what it names, parents and
        // all, so "absent" is only unwritable where the root is. See `blocked_path`.
        let (blocker, path) = blocked_path("capture", "deep/session.jsonl");
        cfg.capture.path = path;
        let err = run_offline(&cfg).unwrap_err();
        let _ = std::fs::remove_file(&blocker);
        assert!(matches!(err, RuntimeError::Capture(_)), "{err:?}");
    }

    #[test]
    fn recording_a_session_does_not_change_the_session() {
        // The observer effect, ruled out. A recording that perturbed the run would make
        // every artifact it produced a recording of a *different* session — and the one
        // place it could plausibly do so is the strategy pass, since the tap sits on the
        // signal source the reader drains.
        let plain = run_offline(&offline_cfg()).expect("offline");
        let cfg = capturing_cfg("noninterference");
        let recorded = run_offline(&cfg).expect("offline");

        assert_eq!(recorded.planned, plain.planned, "same orders, same cloids");
        assert_eq!(recorded.status.events, plain.status.events);
        assert_eq!(recorded.status.positions, plain.status.positions);
        assert_eq!(recorded.status.open_orders, plain.status.open_orders);
        assert_eq!(recorded.status.intent, plain.status.intent);
        remove_capture(recorded.capture.as_ref().unwrap());
    }

    #[test]
    fn two_captures_of_the_same_session_record_the_same_records() {
        // The property the parity harness rests on, asserted through the recording
        // rather than through the run: same events in, same records out. The *bytes*
        // differ by one field — the header's wall-clock `created_ns`, which is
        // provenance and never an ordering key — so this compares what a replay reads.
        fn records(tag: &str) -> Vec<axon_replay::LogRecord> {
            let cfg = capturing_cfg(tag);
            let summary = run_offline(&cfg).expect("offline");
            let out = summary.capture.unwrap();
            let log = axon_replay::EventLog::open(&out.log_path).unwrap();
            let recs = log.records().to_vec();
            remove_capture(&out);
            recs
        }
        assert_eq!(records("det-a"), records("det-b"));
    }

    #[test]
    fn a_single_symbol_universe_still_runs() {
        let mut cfg = offline_cfg();
        cfg.strategy.symbols = vec!["BTC-PERP".into()];
        let s = run_offline(&cfg).expect("one symbol is enough").status;
        assert_eq!(s.events, 7);
        assert_eq!(s.positions.len(), 1);
    }

    #[test]
    fn a_third_configured_symbol_does_not_refuse_a_session_that_contacts_no_venue() {
        // `resolve_symbols` demands a grid for every configured coin, and the offline
        // table used to declare two. So a three-symbol config — a shape `coins()` is
        // itself tested to support — could not start `cargo run --bin axon` at all, and
        // said so by naming a venue this mode has no socket, no key and no universe for.
        // The default gate refusing a valid config is the worst kind of red: it sends an
        // operator to look at the venue.
        let mut cfg = offline_cfg();
        cfg.strategy.symbols = vec!["BTC-PERP".into(), "ETH-PERP".into(), "SOL-PERP".into()];
        assert_eq!(cfg.coins(), vec!["BTC", "ETH", "SOL"]);
        let s = run_offline(&cfg)
            .expect("the offline fiction declares a grid for every coin it is given")
            .status;
        assert_eq!(
            s.events, 7,
            "the canned stream still only touches the first two"
        );
    }

    #[test]
    fn an_unknown_symbol_is_refused_rather_than_silently_ignored() {
        // Live, this is the difference between "no signal today" and "we subscribed to
        // a coin the venue has never heard of".
        let cfg = offline_cfg();
        let map = SymbolMap::from_perps(["BTC"]);
        let all = selftest::instruments([SymbolId::new(0), SymbolId::new(1)]);
        let err = resolve_symbols(&cfg, &map, &all).unwrap_err();
        assert!(matches!(err, RuntimeError::UnknownSymbol(ref c) if c == "ETH"));
    }

    #[test]
    fn a_configured_coin_with_no_instrument_spec_is_refused_at_startup() {
        // The venue knows the coin, so the subscription works, the reconcile works and
        // the status line says OK — and every order that would add exposure to it is
        // refused at the wire, one signal at a time, forever. The check costs nothing
        // (the map and the table come out of one `meta` response) and it moves that
        // failure to the one moment an operator is watching.
        let cfg = offline_cfg();
        let map = SymbolMap::from_perps(["BTC", "ETH"]);
        let mut only_btc = axon_providers::InstrumentTable::new();
        only_btc.insert(
            *selftest::instruments([SymbolId::new(0), SymbolId::new(1)])
                .get(SymbolId::new(0))
                .unwrap(),
        );
        let err = resolve_symbols(&cfg, &map, &only_btc).unwrap_err();
        assert!(
            matches!(err, RuntimeError::NoInstrumentSpec(ref c) if c == "ETH"),
            "got {err:?}"
        );
        // And it is a different sentence from "the venue never heard of it", because it
        // sends an operator to a different place.
        assert!(err.to_string().contains("tick and lot"), "{err}");
    }

    #[test]
    fn a_live_config_without_the_mainnet_gate_is_refused_before_anything_connects() {
        // Two independent switches guard real money: the config *and* the environment
        // variable. This asserts the second one, without opening a socket.
        std::env::remove_var(crate::config::ENV_ALLOW_MAINNET);
        let mut cfg = RuntimeConfig {
            environment: Environment::Live,
            ..offline_cfg()
        };
        cfg.venue.network = Network::Mainnet;
        cfg.venue.account = Some("0x".to_string() + &"a".repeat(40));
        cfg.validate().expect("the config itself is coherent");
        let err = cfg.mainnet_allowed().unwrap_err();
        assert!(err.to_string().contains("AXON_ALLOW_MAINNET"));
    }

    #[test]
    fn the_submit_pipeline_refuses_before_it_reaches_the_venue_when_halted() {
        // Built with a throwaway key so nothing can be signed for a real account; the
        // point is that the outermost wrapper answers without any network at all.
        let cfg = offline_cfg();
        let signer = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            false,
        )
        .unwrap();
        let exchange = Arc::new(ExchangeClient::new(ExchangeClient::TESTNET, signer));
        let state = CoreState::new(&cfg, true);
        state.halt.halt();
        let pipeline = build_pipeline(&cfg, exchange, Arc::new(RateGovernor::new()), &state);

        let req = axon_providers::OrderRequest::limit(
            SymbolId::new(0),
            axon_core::Side::Buy,
            Decimal::ONE,
            dec!(100),
            axon_core::Tif::Gtc,
            axon_core::Cloid::new(1),
        );
        let err = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(pipeline.place_order(req))
            .unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(ref m) if m.contains("halted")));
    }

    /// The account address a live test trades for.
    ///
    /// `AXON_HL_ACCOUNT_ADDRESS` is the name `.env`, `.env.example` and
    /// `scripts/with-env.sh` actually define; `AXON_HL_ACCOUNT` is kept as an explicit
    /// override for a one-off run against a different account. It used to be the only
    /// name read, which is a variable defined nowhere in this repository — so the test
    /// panicked before it opened a socket, and because **cargo abandons a run at the
    /// first failing test binary**, that panic could stop an entirely different live
    /// test from ever executing. A missing account is a real failure; a missing account
    /// that stops something else running is a failure nobody attributes correctly.
    fn live_account() -> String {
        std::env::var("AXON_HL_ACCOUNT")
            .or_else(|_| std::env::var("AXON_HL_ACCOUNT_ADDRESS"))
            .expect("AXON_HL_ACCOUNT_ADDRESS (or AXON_HL_ACCOUNT) must name the master account")
    }

    /// The shape both live tests start from: sandbox on testnet, one instrument, a
    /// short dead-man's-switch lead so a run of this length re-arms more than once.
    fn live_cfg() -> RuntimeConfig {
        let mut cfg = RuntimeConfig {
            environment: Environment::Sandbox,
            ..RuntimeConfig::default()
        };
        cfg.venue.network = Network::Testnet;
        cfg.venue.account = Some(live_account());
        cfg.strategy.symbols = vec!["BTC-PERP".into()];
        cfg.session.status_interval_ms = 2_000;
        cfg.safety.lead_ms = 30_000;
        cfg.safety.rearm_interval_ms = 10_000;
        cfg.reconcile.interval_ms = 5_000;
        cfg
    }

    /// One `/info` read from a synchronous test.
    ///
    /// A current-thread runtime built and dropped around a single request, rather than
    /// a `#[tokio::test]`: `run_live` builds its **own** multi-thread runtime and
    /// `block_on`s it, and doing that from inside another runtime's worker panics. The
    /// read is free — `nRequestsUsed` counts only signed `/exchange` actions.
    fn rt_block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for one /info read")
            .block_on(f)
    }

    /// Stop the process from the outside the way an operator would — a real signal, so
    /// the shutdown path under test is the one that runs in production.
    fn interrupt_after(secs: u64) {
        let pid = std::process::id();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(secs));
            let _ = std::process::Command::new("kill")
                .arg("-INT")
                .arg(pid.to_string())
                .status();
        });
    }

    /// A short **read-only** live session against testnet. Ignored by default so the
    /// gate stays offline; run it with a testnet key:
    /// `AXON_HL_SECRET_KEY=0x… AXON_HL_ACCOUNT_ADDRESS=0x… cargo test -p axon-runtime --
    /// --ignored live_sandbox_session`.
    ///
    /// It asserts only what a read-only session can prove: that the session starts,
    /// resolves symbols, reconciles, arms the switch, streams prices into the mark
    /// cache and shuts down cleanly. It never places an order, so it never depends on
    /// a fill being observable — and `intent.enabled = false` is what makes that true
    /// rather than merely likely. The intent source is **on by default** (ADR-0020 §9),
    /// and this test used to leave it on while claiming to place nothing: it would have
    /// obeyed whatever an unrelated producer had left on the default ring path. Turning
    /// it off is the only way to say "read-only", and it has to be said.
    #[test]
    #[ignore = "hits live Hyperliquid testnet; needs AXON_HL_SECRET_KEY and AXON_HL_ACCOUNT_ADDRESS"]
    fn live_sandbox_session() {
        let mut cfg = live_cfg();
        cfg.intent.enabled = false;
        cfg.validate().expect("config");

        interrupt_after(25);
        let summary = run_live(&cfg).expect("live session");
        assert!(summary.status.events > 0, "no market data arrived");
        assert!(
            summary.status.marks_fresh > 0,
            "the mark cache stayed empty"
        );
        assert!(
            summary.status.intent.is_none(),
            "a read-only session reports the absence of an intent source, not zeros"
        );
        let shutdown = summary.shutdown.expect("shutdown outcome");
        assert!(shutdown.swept, "the sweep must succeed on an empty account");
    }

    /// The Phase-4 exit criterion, as a test: a target position written onto the signal
    /// ring becomes an order at the venue, and the flatten that follows it leaves the
    /// account as it found it.
    ///
    /// **This places real orders on Hyperliquid testnet.** Run it deliberately:
    /// `bash scripts/with-env.sh cargo test -p axon-runtime --
    /// --ignored --nocapture a_signal_on_the_ring_becomes_an_order_at_the_venue`
    ///
    /// It exists because everything between a `Signal` and a venue was proven against a
    /// spy `ExecutionClient` and an offline sink, and neither of those can fail the way
    /// a venue does. `run_offline`'s sink records rather than submits (ADR-0020 §10),
    /// so nothing in the default gate exercises `pump` → `submit_intent` →
    /// `Haltable→Guarded→Governed→Exchange` at all; `live_sandbox_session` above runs
    /// the session and deliberately reads only. This is the one test in the workspace
    /// whose green means an order reached a venue *because Python asked for one*.
    ///
    /// Two things about the harness are deliberate and would be bugs in production code:
    ///
    /// - It writes the records itself, with `axon_ipc::Producer`, instead of running
    ///   `axon.live.probe`. A Rust test that shells out to Python proves the two
    ///   processes can be started in the right order, which is a launch-script property
    ///   and not this crate's; what this crate owes an assertion about is the bytes.
    /// - It stamps `ts_event` from the **wall clock**, which nothing in the runtime may
    ///   do. Here it is standing in for a producer whose event source is the live venue,
    ///   and the two clocks are the same epoch to within the feed's latency. In the
    ///   runtime the same substitution would make a replayed session call every signal
    ///   infinitely stale, which is why `SignalReader` takes the core's event clock and
    ///   `StrategyContext` has no clock at all.
    #[test]
    #[ignore = "PLACES REAL ORDERS on Hyperliquid testnet; needs AXON_HL_SECRET_KEY and AXON_HL_ACCOUNT_ADDRESS"]
    fn a_signal_on_the_ring_becomes_an_order_at_the_venue() {
        use axon_contracts::{Signal, FLAG_CLOSE};

        // BTC's testnet lot is 1e-5 and the venue's minimum notional is $10, so this is
        // the smallest target that is both expressible on the grid and large enough to
        // be accepted — about $13 at the price this was written against. The risk limits
        // below are the structural ceiling: `max_order_qty` refuses anything larger
        // before it can reach a wire, so a mistyped size cannot become a real trade.
        const TARGET_QTY: i64 = 20_000; // 0.0002 BTC in 1e-8 fixed point
        let ring = std::env::temp_dir().join(format!(
            "axon-live-intent-{}-{}.ring",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&ring);

        let mut cfg = live_cfg();
        cfg.ipc.signal_ring_path = ring.to_string_lossy().into_owned();
        cfg.ipc.capacity = 16;
        cfg.intent.enabled = true;
        // Two whole seconds of admission window, and the producer below writes its
        // records at the moment it wants them acted on — so a signal that expires here
        // means the core was not draining, never that the record was written early.
        cfg.intent.max_signal_age_ms = 2_000;
        cfg.strategy.risk.max_position = dec!(0.0005);
        cfg.strategy.risk.max_order_qty = dec!(0.0005);
        cfg.strategy.risk.max_notional = dec!(50);
        cfg.validate().expect("config");

        // Resolved through the venue's own `meta` universe rather than written down.
        // Asset indices are **not stable across networks** — BTC is index 0 on mainnet
        // and 3 on testnet — and a hard-coded id does not fail: it trades a different
        // instrument, successfully and silently, at a size computed for this one.
        let btc = rt_block_on(axon_provider_hyperliquid::ws::fetch_universe(
            cfg.info_url(),
        ))
        .expect("meta")
        .symbols
        .id("BTC")
        .expect("BTC must be in the testnet perp universe");

        // The ring is created *before* the session, because `Producer::create` zeroes
        // the indices: creating it under a core that had already attached would rewind
        // the sequence the consumer reads against.
        let producer =
            axon_ipc::Producer::create(&ring, cfg.ipc.capacity).expect("create the signal ring");
        let writer = std::thread::spawn(move || {
            let now_ns = || {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as i64
            };
            // Long enough for the socket to connect, the first quotes to arrive and the
            // dead-man's switch to arm: the session *starts halted* and refuses orders
            // until the first re-arm succeeds, so a signal written before that is
            // admitted, planned, and then correctly refused by the halt.
            std::thread::sleep(Duration::from_secs(12));
            // urgency 3 is an IOC through the far touch: it leaves no remainder resting,
            // so what the venue holds afterwards is a position and never an order.
            // `ttl_ms = 0` is the operator's ceiling (ADR-0020 §4), which is what a
            // producer with no opinion about staleness emits.
            let open = Signal::target_position(1, now_ns(), btc.get(), TARGET_QTY, 3, 0, 0, 1, 0);
            assert!(producer.try_push(&open), "the ring must take the open");

            // Held long enough for the IOC's fill to reach the tracker through
            // `userFills`, so the flatten's delta is computed against a position the
            // session has actually observed rather than against the one it hoped for.
            std::thread::sleep(Duration::from_secs(10));
            // `FLAG_CLOSE` rather than a zero target: a zero target is an opinion about
            // the position, and one computed against a fill we have not been told about
            // overshoots into the opposite side. Close ignores `target_qty` entirely and
            // implies reduce-only (ADR-0014 §2).
            let close = Signal::target_position(2, now_ns(), btc.get(), 0, 3, 0, 0, 1, FLAG_CLOSE);
            assert!(producer.try_push(&close), "the ring must take the close");
            std::thread::sleep(Duration::from_secs(8));
            producer
        });

        interrupt_after(45);
        let summary = run_live(&cfg).expect("live session");
        let producer = writer.join().expect("the signal writer must not panic");
        drop(producer);
        let _ = std::fs::remove_file(&ring);

        let line = summary
            .status
            .intent
            .clone()
            .expect("the intent source is reported");
        assert!(
            line.attached,
            "the core never attached to the ring: {line:?}"
        );
        assert_eq!(
            line.accepted, 2,
            "both records must be read and admitted: {line:?}"
        );
        assert_eq!(line.expired, 0, "a record expired on the ring: {line:?}");
        assert_eq!(line.planned, 2, "both targets must plan an order: {line:?}");
        // The assertion the whole test exists for. `planned` is a core-thread number and
        // says only that a `Plan` was built; `orders` is incremented after the venue
        // answered, so it is the first counter in this workspace that cannot be true
        // without a venue.
        assert!(
            line.orders >= 2,
            "no order reached the venue - the gap between accepted and sent is the \
             diagnosis: {line:?}"
        );
        assert_eq!(line.failures, 0, "the venue refused an order: {line:?}");
        assert_eq!(
            line.halted, 0,
            "we refused our own order while halted: {line:?}"
        );
        assert!(!line.stalled, "the submit path stopped answering: {line:?}");

        // And the account is left as it was found. `graceful_shutdown` sweeps whatever
        // is still resting, but an IOC leaves nothing to sweep — so a non-flat position
        // here means the flatten never became an order, which is the one outcome of this
        // test that must never pass quietly.
        let shutdown = summary.shutdown.expect("shutdown outcome");
        assert!(shutdown.swept, "the closing sweep failed: {shutdown:?}");
        assert_eq!(
            summary.status.open_orders, 0,
            "an order was left resting at the venue: {}",
            summary.status
        );
        assert!(
            summary
                .status
                .positions
                .iter()
                .all(|(_, qty)| qty.is_zero()),
            "the session ended holding a position: {:?}",
            summary.status.positions
        );
    }
}
