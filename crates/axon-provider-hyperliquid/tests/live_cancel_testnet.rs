//! Live **cancel/replace** verification on Hyperliquid testnet — the untouched half of
//! [ADR-0014] §6.
//!
//! ## What this proves that nothing else does
//!
//! Every live order this project has ever placed was a marketable IOC that filled
//! outright, so **no cancel has ever been sent through the intent path at a venue**.
//! Two existing tests look as though they cover it and do not:
//!
//! - `place_then_cancel_on_testnet` (in `src/exchange/mod.rs`) rests a post-only 20 %
//!   under the bid and cancels it. It proves signing, the nonce, the `/exchange`
//!   envelope and response parsing — and, by construction, nothing about the planner,
//!   the re-addressing, or the decision *not* to cancel. It hand-builds one
//!   `CancelId::OrderId` and posts it.
//! - `live_fill_testnet.rs` proves a fill and its reconciliation. An IOC that fills
//!   outright never rests, so it never becomes something to cancel.
//!
//! What has therefore never met a venue is the whole cancel *decision*: the planner
//! choosing to supersede a working order, the runtime re-addressing an adopted order's
//! cancel from a synthesized `cloid` to the venue's own oid, the submitter putting
//! cancels on the wire before their replacements, and — the one a test that cancels
//! unconditionally would pass without noticing — the planner deciding an order already
//! resting still satisfies the target and correctly cancelling **nothing**.
//!
//! So this file drives the **production intent path** end to end:
//!
//! ```text
//!   Signal ─▶ SignalReader ─▶ Planner ─▶ retarget_cancels ─▶ submit_intent ─▶ venue
//!   ───────── axon-strategy ──────────┤──── axon-runtime ────┤─── this crate ───
//! ```
//!
//! Nothing here rounds its own price, picks its own `cloid`, or hand-builds a
//! `CancelId`. That is not tidiness: the live test this one is modelled on used to
//! round its own prices, and it passed green over an encoder that could not have
//! rounded them correctly — a test of itself. Every order and every cancel below is
//! whatever `axon_runtime::intent` decided, asserted and then submitted unchanged.
//!
//! `axon-runtime` is a **dev**-dependency of this crate for exactly that reason; see
//! `Cargo.toml`, and note that nothing in `src/` here may ever name it.
//!
//! ## The five claims, and which phase settles each
//!
//! | # | Claim | Phase |
//! |---|-------|-------|
//! | 1 | A resting post-only reaches the venue **through the intent path** | A |
//! | 2 | Moving the target makes the planner cancel, the cancel go out, and the replacement land | B |
//! | 3 | An **adopted** order — one this session never placed — is cancelled by venue **oid**, because the `cloid` keying it was synthesized and the venue has never seen it | D |
//! | 4 | An order that still satisfies the target is **left resting**, and no cancel is sent | C |
//! | 5 | What the venue actually emits for a cancel: the `orderUpdates` frame, the REST body, and the answer to a cancel of an oid that is already gone | throughout, and E |
//!
//! Phase C is the one that cannot be faked. A cancel path that cancels unconditionally
//! passes every "did the cancel go out" check and is wrong, so the evidence for it is
//! negative: zero intents produced, and the venue still reporting the *same* oid at the
//! same price afterwards.
//!
//! ## What it costs
//!
//! Testnet play money and, at most, **nine signed `/exchange` actions**: two places, two
//! intent-path cancels, one replacement place, two deliberate probe cancels that are
//! expected to be *refused*, and the cleanup. `/info` reads are free. It arms **no**
//! dead-man's switch, so it spends none of the venue's ten daily triggers.
//!
//! Every order rests **20 % below the best bid**, priced there by the planner through
//! the signal's own `price_band`. It cannot fill, so no position is ever opened and no
//! fee is ever paid — the balance moves by nothing. The cleanup still reads
//! `clearinghouseState` twice and sweeps, because "cannot fill" is an argument and a
//! flat account is an observation.
//!
//! ## Running it
//!
//! ```text
//! ./run.sh wallet     # first: agent approved+valid, account flat and order-free
//! bash scripts/with-env.sh cargo test -p axon-provider-hyperliquid \
//!     --test live_cancel_testnet -- --ignored --nocapture
//! ```
//!
//! `--nocapture` because the running commentary *is* the evidence: the plan each pass
//! produced, the oid each cancel addressed, and every `orderUpdates` frame the venue
//! sent, verbatim. Secrets arrive through `.env` only, never argv.
//!
//! It refuses to start unless `AXON_HL_NETWORK` is exactly `testnet`, before the key is
//! read and before anything is signed, and it refuses to start on an account that
//! already holds a position or a resting order in this coin — the sweep at the end
//! cancels everything in the coin, and it must not cancel somebody else's quote.
//!
//! [ADR-0014]: ../../../docs/adr/0014-signal-to-order-planning.md

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axon_contracts::Signal;
use axon_core::{
    bus, drain_available, Clock, Cloid, Decimal, Event, EventHandler, EventReceiver, ExecEvent,
    Fill, Nanos, OrderId, OrderStatus, OrderUpdate, Side, SymbolId, SystemClock, Tif,
};
use axon_execution::{HaltSwitch, MarkCache, OrderTracker};
use axon_provider_hyperliquid::exchange::response::{parse_cancel_outcomes, CancelOutcome};
use axon_provider_hyperliquid::ws::{
    decode_universe, subscribe_user_msg, UserChannel, TESTNET_INFO,
};
use axon_provider_hyperliquid::{
    fetch_frontend_open_orders, fetch_open_orders, fetch_user_state, ExchangeClient, HlSigner,
    HyperliquidMarketData, OpenOrder, SymbolMap,
};
use axon_providers::{
    CancelId, ExecutionClient, Feed, InstrumentSpec, InstrumentTable, OrderRequest, PriceIntent,
};
use axon_runtime::intent::{
    submit_intent, Attachable, Intent, IntentPoll, IntentQueue, IntentSource,
};
use axon_runtime::latency::LatencyBook;
use axon_runtime::{CoreHandler, RuntimeConfig, SessionHealth};
use axon_strategy::{cloid_for, decimal_to_fixed, SignalSource};
use rust_decimal::RoundingStrategy;
use rust_decimal_macros::dec;

// ── what this test trades, and the bounds it refuses to leave ────────────────

/// The instrument. BTC because its testnet lot (`szDecimals: 5`) is fine enough that a
/// $12 target and a $20 one are two *different* sizes — which is the whole mechanism
/// phase B uses to move the target.
const COIN: &str = "BTC";

/// The first target's value in USDC, and the second's. They must round to different
/// sizes on the lot, or "move the target" moves nothing and phase B proves nothing.
const TARGET_A: Decimal = dec!(12);
const TARGET_B: Decimal = dec!(20);

/// Hyperliquid's minimum order notional. Under it the venue answers
/// `minTradeNtlRejected`, and a rejection proves nothing.
const MIN_NOTIONAL: Decimal = dec!(10);

/// The most this test will ever put on the wire. Not a risk limit — a tripwire for
/// **lot rounding**: on an asset with `szDecimals: 0` a $12 target rounds up to one
/// whole unit, and one whole unit of a $1,000 asset is a $1,000 order.
const MAX_NOTIONAL: Decimal = dec!(50);

/// Where the resting quote sits, as a fraction of the best bid.
///
/// Twenty per cent under the market. Far enough that the order cannot be filled by any
/// plausible move during the run — this test is about cancels, and a fill would open a
/// position it is not built to reason about — and still inside the venue's limit-vs-
/// oracle band, which refuses prices ~80 % away.
const REST_UNDER: Decimal = dec!(0.80);

// ── timings (wall-clock deadlines, not ordering keys) ────────────────────────

const POLL: Duration = Duration::from_millis(200);
/// Long enough for a TLS handshake, the subscribe frames and the first BBO on a bad
/// link; short enough that a dead socket fails rather than hangs an operator.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// The `userFills` snapshot replays this account's history the instant we subscribe.
/// Consume it before anything is measured.
const SNAPSHOT_SETTLE: Duration = Duration::from_secs(2);
/// How long the venue gets to publish an order-lifecycle transition on `orderUpdates`.
/// A cancel is acknowledged over REST in well under a second; 20 s means "it did not
/// happen", which is itself a finding worth recording rather than hanging on.
const FRAME_TIMEOUT: Duration = Duration::from_secs(20);
/// Time for `clearinghouseState` / `openOrders` to reflect an action. Both trail the
/// `/exchange` reply by a round trip.
const VENUE_SETTLE: Duration = Duration::from_secs(2);
/// A book older than this is not a price, it is a memory.
const MAX_BOOK_AGE: Duration = Duration::from_secs(60);

/// Deep enough that nothing has to drain the bus during the sweep: the bus blocks its
/// producer when full, and the blocked producer would be the WS task delivering the
/// very frames the sweep is waiting on.
const BUS_CAPACITY: usize = 16_384;

/// A check that yields a message instead of a panic.
///
/// Everything between "an order was sent" and "the account is order-free again" uses
/// this rather than `assert!`: a panic there unwinds straight past the sweep, and a test
/// that leaves an order resting when it fails is a test nobody dares run twice.
macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err(format!($($arg)*));
        }
    };
}

// ── pure helpers (offline-tested at the bottom of this file) ─────────────────

/// Refuse anything but testnet, before the key is read and before anything is signed.
///
/// An unrecognized value is refused rather than defaulted. This file places real orders
/// and sends real cancels; "I assumed you meant testnet" is not a sentence worth writing
/// about live order flow.
fn require_testnet(network: Option<&str>) -> Result<(), String> {
    let rule = "!".repeat(78);
    match network {
        Some("testnet") => Ok(()),
        other => Err(format!(
            "{rule}\n\
             !! REFUSING TO RUN: AXON_HL_NETWORK is {other:?}, not \"testnet\".\n\
             !!\n\
             !! This test places real orders and sends real cancels at a real venue. On\n\
             !! mainnet that is real money, and nothing about the code below would stop\n\
             !! it. Set AXON_HL_NETWORK=testnet in .env, confirm with `./run.sh wallet`,\n\
             !! and run it again.\n\
             {rule}"
        )),
    }
}

/// The size that puts `notional` on the wire at `px`, on the asset's lot, with the
/// result checked against both bounds.
///
/// Rounded **away from zero** so lot rounding can only ever push the notional up. Down
/// is the cheap direction and the wrong one: a $9.99 order is refused by the planner's
/// own minimum-notional check and the pass produces nothing, which reads from outside
/// exactly like the leave-it-resting decision phase C is trying to prove.
fn size_for_notional(px: Decimal, sz_decimals: u32, notional: Decimal) -> Result<Decimal, String> {
    ensure!(px > Decimal::ZERO, "price must be positive, got {px}");
    let qty = (notional / px).round_dp_with_strategy(sz_decimals, RoundingStrategy::AwayFromZero);
    ensure!(
        qty > Decimal::ZERO,
        "lot rounding produced a zero size at {px}"
    );
    let value = qty * px;
    ensure!(
        value >= MIN_NOTIONAL,
        "notional ${value} (= {qty} @ {px}) is under the venue's ${MIN_NOTIONAL} minimum"
    );
    ensure!(
        value <= MAX_NOTIONAL,
        "notional ${value} (= {qty} @ {px}) exceeds this test's ${MAX_NOTIONAL} ceiling - \
         the lot size ({sz_decimals} decimals) rounded a ${notional} test into something \
         far larger. Pick a coin with a finer lot."
    );
    Ok(qty)
}

/// One asset as the venue's own `meta.universe` describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AssetSpec {
    /// Position in `universe`, which **is** the on-wire asset index.
    index: u32,
    sz_decimals: u32,
    delisted: bool,
}

/// Pull one coin's spec out of a `meta` body.
///
/// The index has to come from the live universe and nowhere else: it is positional,
/// testnet's differs from mainnet's (BTC is index 3 on testnet, not 0), and it changes
/// as assets are listed. A stale index does not fail — it trades **a different coin**.
fn asset_spec(meta_json: &str, coin: &str) -> Result<AssetSpec, String> {
    let v: serde_json::Value =
        serde_json::from_str(meta_json).map_err(|e| format!("meta is not JSON: {e}"))?;
    let universe = v
        .get("universe")
        .and_then(|u| u.as_array())
        .ok_or_else(|| "meta has no universe array".to_string())?;
    for (i, asset) in universe.iter().enumerate() {
        if asset.get("name").and_then(|n| n.as_str()) != Some(coin) {
            continue;
        }
        let sz_decimals = asset
            .get("szDecimals")
            .and_then(|d| d.as_u64())
            .ok_or_else(|| format!("{coin} has no szDecimals"))?;
        return Ok(AssetSpec {
            index: i as u32,
            sz_decimals: sz_decimals as u32,
            // Absent means listed. 53 of testnet's 210 perps carry this flag and they
            // keep their index, so resolving a name proves nothing about tradability.
            delisted: asset
                .get("isDelisted")
                .and_then(|d| d.as_bool())
                .unwrap_or(false),
        });
    }
    Err(format!("{coin} is not in the venue's perp universe"))
}

/// A target-position signal, shaped the way Python's producer shapes one.
///
/// `price_band` is what puts the quote 20 % under the market: the planner reads it as
/// the *worst* price acceptable, which for a buy is a ceiling, so it can only ever move
/// the limit away from the touch. That is the production knob for "rest here", and using
/// it rather than a bespoke price is what keeps this test on the intent path.
///
/// `ttl_ms = 0` deliberately: it means "the operator's ceiling" on both sides
/// (ADR-0020 §4), which is the value every strategy that has never thought about
/// staleness emits, and therefore the one worth exercising.
fn target_signal(
    seq: u64,
    ts_event: Nanos,
    symbol: SymbolId,
    qty: Decimal,
    band: Decimal,
) -> Signal {
    Signal::target_position(
        seq,
        ts_event,
        symbol.get(),
        decimal_to_fixed(qty).expect("target is representable at the contract's 8 dp"),
        // Urgency 0 ⇒ post-only at the near touch. Post-only because a resting quote is
        // the only order shape there is anything to cancel about, and because the venue
        // rejects a post-only that would cross — which is a second, free assertion that
        // the band really did move the price out of the market.
        0,
        decimal_to_fixed(band).expect("band is representable at the contract's 8 dp"),
        0,
        1,
        0,
    )
}

// ── the core-side handler chain ──────────────────────────────────────────────

/// What the core sees during the run: the production [`CoreHandler`] fan-out, plus a
/// verbatim log of the execution events so a failure can be read instead of guessed at.
struct LiveSession {
    core: CoreHandler,
    updates: Vec<OrderUpdate>,
    fills: Vec<Fill>,
}

impl LiveSession {
    fn new() -> Self {
        Self {
            // The real cache with its real 10 s expiry, not `never_expires()`: the
            // planner refuses to quote an instrument with no live mark, and a test that
            // disabled the expiry would not be running the rule a session runs.
            core: CoreHandler::new(
                Arc::new(RwLock::new(OrderTracker::new())),
                Arc::new(MarkCache::new()),
            ),
            updates: Vec::new(),
            fills: Vec::new(),
        }
    }

    /// Throw away everything this process knew about its own orders, as a restart does.
    ///
    /// Phase D needs an order the session did **not** place, and the honest way to get
    /// one is not to place it differently — it is to forget having placed it. A fresh
    /// tracker then meets the venue's own order list exactly as a restarted process
    /// does, and adopts what it finds.
    fn restart(&mut self) {
        self.core = CoreHandler::new(
            Arc::new(RwLock::new(OrderTracker::new())),
            Arc::new(MarkCache::new()),
        );
    }

    fn open_for(&self, sym: SymbolId) -> Vec<(Cloid, OrderId, bool, Option<Tif>, Decimal)> {
        let Ok(t) = self.core.tracker().read() else {
            return Vec::new();
        };
        t.open_orders()
            .filter(|o| o.symbol_id == sym)
            .map(|o| {
                (
                    o.cloid,
                    o.order_id.unwrap_or(OrderId::new(0)),
                    o.adopted,
                    o.tif,
                    o.remaining_qty(),
                )
            })
            .collect()
    }

    fn status_of(&self, cloid: Cloid) -> Option<OrderStatus> {
        self.core
            .tracker()
            .read()
            .ok()?
            .order(cloid)
            .map(|o| o.status)
    }
}

impl EventHandler for LiveSession {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        self.core.on_event(ts_event, event);
        match event {
            Event::Exec(ExecEvent::Fill(f)) => self.fills.push(f.clone()),
            Event::Exec(ExecEvent::Order(u)) => self.updates.push(u.clone()),
            _ => {}
        }
    }
}

// ── the signal source ────────────────────────────────────────────────────────

/// A [`SignalSource`] the test can append to between passes.
///
/// The production source is a memory-mapped ring another process writes into.
/// `ReplaySource` is the offline stand-in, but an `IntentSource` owns its source and
/// exposes no way to push into it, and building a fresh `IntentSource` per pass would
/// reset the reader's `seq` baseline — the one piece of reader state a live sequence of
/// targets is supposed to carry. So this is a queue with the handle kept outside.
#[derive(Debug, Clone, Default)]
struct PushSource {
    queue: Arc<Mutex<VecDeque<Signal>>>,
}

impl PushSource {
    fn push(&self, sig: Signal) {
        self.queue.lock().expect("signal queue lock").push_back(sig);
    }
}

impl SignalSource for PushSource {
    fn next_signal(&mut self) -> Option<Signal> {
        self.queue.lock().ok()?.pop_front()
    }
}

impl Attachable for PushSource {
    /// Always there — an in-process queue cannot be missing. The retry machinery this
    /// defaults away is `LazyRing`'s and is tested in the runtime.
    fn ensure(&mut self, _now_ms: u64) -> bool {
        true
    }
}

/// Everything one intent pass needs, assembled the way the session assembles it.
struct IntentRig {
    src: IntentSource<PushSource>,
    signals: PushSource,
    queue: IntentQueue,
    halt: Arc<HaltSwitch>,
    health: Arc<SessionHealth>,
    /// Declared nowhere and measured anyway (ADR-0036): this harness is where the
    /// venue round trip is actually taken, so the numbers are free and a live run's
    /// `ack` distribution is one of the few things only a live run can produce.
    latency: Arc<LatencyBook>,
    seq: u64,
}

impl IntentRig {
    fn new(instruments: Arc<InstrumentTable>, health: Arc<SessionHealth>) -> Self {
        let cfg = RuntimeConfig::default();
        let signals = PushSource::default();
        let queue = IntentQueue::new(cfg.intent.queue_capacity);
        let halt = Arc::new(HaltSwitch::new());
        let latency = Arc::new(LatencyBook::undeclared());
        Self {
            src: IntentSource::new(
                signals.clone(),
                &cfg.intent,
                instruments,
                cfg.mark_max_age_ns(),
                halt.clone(),
                queue.sink(),
            )
            .with_latency(latency.clone()),
            signals,
            queue,
            halt,
            health,
            latency,
            seq: 0,
        }
    }

    /// A restarted process gets a fresh reader with a fresh `seq` baseline. Keeping the
    /// old one would be modelling a restart that remembered something a restart forgets.
    fn restart(&mut self, instruments: Arc<InstrumentTable>) {
        let health = self.health.clone();
        *self = Self::new(instruments, health);
    }

    /// Write one target onto the "ring", stamped with the **producer's own clock**.
    ///
    /// Not with `CoreHandler::last_ts()`, and the first live run is why. A signal's
    /// `ts_event` is the moment the strategy decided, and a strategy deciding now stamps
    /// now — which is what the Python producer does. Stamping it with the core's event
    /// clock instead back-dates the decision by however long it has been since the last
    /// market frame, and testnet BTC's `bbo` is quiet enough that the gap exceeds the
    /// reader's 2 s ceiling: the record is written, one BBO lands, `now` jumps past the
    /// stamp, and the reader refuses the record as **expired** before the planner ever
    /// sees it. Observed exactly so — `accepted: 1, expired: 1` with the second target
    /// never planned.
    ///
    /// The consequence is a record stamped slightly *ahead* of the core clock, which the
    /// reader counts as `ahead_of_clock` and admits. That asymmetry is deliberate on its
    /// side too: a signal from the future is a clock-skew observation, a signal from the
    /// past is a decision about a market that has moved.
    fn emit(&mut self, symbol: SymbolId, qty: Decimal, band: Decimal) -> Signal {
        self.seq += 1;
        let sig = target_signal(self.seq, SystemClock.now_ns(), symbol, qty, band);
        self.signals.push(sig);
        sig
    }

    /// One pass of the deterministic core loop, with the same three steps in the same
    /// order `axon_runtime::core::run` uses: drain the bus, advance the mark cache's
    /// *liveness* clock (wall time — it has to keep moving when the feed does not), then
    /// poll the intent source at **event** time.
    fn pass(&mut self, rx: &EventReceiver, session: &mut LiveSession) -> Vec<Intent> {
        drain_available(rx, session);
        let wall_ns = SystemClock.now_ns();
        session.core.marks().observe_now(wall_ns);
        self.src.poll(
            session.core.last_ts(),
            (wall_ns / 1_000_000) as u64,
            &session.core,
        );
        let rx_intent = self.queue.receiver();
        let mut out = Vec::new();
        while let Ok(i) = rx_intent.try_recv() {
            out.push(i);
        }
        out
    }

    /// Nothing in this run may have been cancelled by the **sweeper**.
    ///
    /// ADR-0031's resting-order sweeper lives in the same `poll` this test drives: after
    /// the planning loop it cancels any tracked order older than
    /// `intent.max_order_age_ms` on a symbol no signal spoke for. That makes it a live
    /// alternative explanation for every cancel below, and "the cancel we observed was
    /// ours" stops being an observation the moment a second thing in the process can
    /// produce one. Two facts rule it out here and this asserts the second: the run is
    /// tens of seconds against a 60 s ceiling, and the sweeper's own counter is zero.
    ///
    /// Deliberately *not* disabled by setting `max_order_age_ms = 0`. The point of the
    /// run is that the intent path a live session runs sends a cancel, and a session
    /// runs with the sweeper armed; turning it off would prove something about a
    /// configuration nobody deploys.
    fn no_sweep_yet(&self, phase: &str) -> Result<(), String> {
        let s = self.src.stats();
        if s.swept != 0 || s.resweeps != 0 {
            return Err(format!(
                "{phase}: the ADR-0031 sweeper fired ({} swept, {} reswept). Every cancel \
                 attributed to the planner above is now ambiguous, because the sweeper \
                 cancels through the same sink.",
                s.swept, s.resweeps
            ));
        }
        Ok(())
    }

    /// Hand one plan to the venue, exactly as `axon_runtime::intent::pump` does:
    /// cancels first, then orders, then release the symbol's in-flight slot.
    async fn submit(&self, client: &dyn ExecutionClient, session: &LiveSession, intent: &Intent) {
        submit_intent(
            client,
            intent,
            &self.halt,
            session.core.tracker(),
            &self.health,
            &self.latency,
        )
        .await;
        self.queue.inflight().release(intent.symbol_id);
    }
}

// ── the raw frame tap ────────────────────────────────────────────────────────

/// Every `orderUpdates` frame the venue sent, as text, in arrival order.
type Frames = Arc<Mutex<Vec<String>>>;

/// A second socket that subscribes to `orderUpdates` and records the bytes, untouched.
///
/// The decoded events already reach the core through `HyperliquidMarketData`, so this
/// looks redundant and is not. The claim this test has to settle is *what the venue
/// emits for a cancel*, and a decoded `OrderUpdate` cannot answer it: a decoder is only
/// as good as the bytes it was given, and this codebase has already shipped a unit test
/// asserting `{"channel":"pong","data":null}` was ignored — a payload the venue has
/// never sent, next to a real `{"channel":"pong"}` that logged a decode error on every
/// heartbeat. The only way not to make that mistake twice is to write the venue's own
/// bytes into the transcript.
async fn tap_order_updates(account: &str) -> Result<(tokio::task::JoinHandle<()>, Frames), String> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    let (ws, _resp) = connect_async(HyperliquidMarketData::TESTNET_WS)
        .await
        .map_err(|e| format!("frame tap connect: {e}"))?;
    let (mut write, mut read) = ws.split();
    write
        .send(Message::text(subscribe_user_msg(
            UserChannel::OrderUpdates,
            account,
        )))
        .await
        .map_err(|e| format!("frame tap subscribe: {e}"))?;

    let frames: Frames = Arc::new(Mutex::new(Vec::new()));
    let sink = frames.clone();
    let handle = tokio::spawn(async move {
        // A ping well inside the venue's ~60 s idle timeout, so the tap outlives a quiet
        // stretch between phases rather than being closed under us.
        let mut heartbeat = tokio::time::interval(Duration::from_secs(50));
        heartbeat.tick().await;
        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    if write.send(Message::text(r#"{"method":"ping"}"#)).await.is_err() {
                        return;
                    }
                }
                item = read.next() => {
                    let Some(Ok(msg)) = item else { return };
                    if let Message::Text(txt) = msg {
                        if let Ok(mut f) = sink.lock() {
                            f.push(txt.to_string());
                        }
                    }
                }
            }
        }
    });
    Ok((handle, frames))
}

/// Frames mentioning `oid`, verbatim. The substring match is deliberate: parsing here
/// would put this file's opinion between the venue and the transcript.
fn frames_for(frames: &Frames, oid: OrderId) -> Vec<String> {
    let needle = format!("\"oid\":{}", oid.get());
    frames
        .lock()
        .map(|f| {
            f.iter()
                .filter(|t| t.contains(&needle))
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

// ── async helpers (the edges) ────────────────────────────────────────────────

async fn post_testnet_info(body: serde_json::Value) -> Result<String, String> {
    reqwest::Client::new()
        .post(TESTNET_INFO)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("POST /info: {e}"))?
        .error_for_status()
        .map_err(|e| format!("POST /info: {e}"))?
        .text()
        .await
        .map_err(|e| format!("POST /info body: {e}"))
}

/// The venue's own live orders for `sym`, from plain `openOrders`.
///
/// **Plain, not `frontendOpenOrders`, and that is the point of phase D.** `openOrders`
/// reports no `cloid` at all, so an order recovered through it is adopted under a `cloid`
/// the tracker synthesized from the venue's oid — an id the venue has never seen. That is
/// the case `CancelId::OrderId` re-addressing exists for, and it is not a corner: a
/// restarted process, or an order placed from the venue's own UI, arrives exactly so.
async fn venue_open_orders(
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
) -> Result<Vec<OpenOrder>, String> {
    let open = fetch_open_orders(TESTNET_INFO, account, None, symbols)
        .await
        .map_err(|e| format!("openOrders: {e}"))?;
    Ok(open
        .items
        .into_iter()
        .filter(|o| o.symbol_id == sym)
        .collect())
}

/// The venue's signed position in `sym`, from `clearinghouseState`.
async fn venue_position(
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
) -> Result<Decimal, String> {
    let state = fetch_user_state(TESTNET_INFO, account, symbols)
        .await
        .map_err(|e| format!("clearinghouseState: {e}"))?;
    Ok(state
        .positions
        .iter()
        .find(|p| p.symbol_id == sym)
        .map(|p| p.qty)
        .unwrap_or(Decimal::ZERO))
}

/// Poll the venue's order list until `f` accepts it, draining the bus meanwhile.
///
/// Draining is what keeps the bounded bus from backing up into the WS task while we
/// wait — a blocked producer would stall the very frames we are waiting for.
async fn wait_for_venue(
    rx: &EventReceiver,
    session: &mut LiveSession,
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    budget: Duration,
    mut f: impl FnMut(&[OpenOrder]) -> bool,
) -> Result<Vec<OpenOrder>, String> {
    let deadline = Instant::now() + budget;
    loop {
        drain_available(rx, session);
        let open = venue_open_orders(account, symbols, sym).await?;
        if f(&open) {
            return Ok(open);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "the venue's order list never reached the expected state within {budget:?}; \
                 it currently holds {}",
                describe(&open)
            ));
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Drain the bus for `budget`, or until `f` is satisfied. Used to wait on the *tracker*
/// rather than on the venue, which is a different question with a different answer.
async fn wait_for_session(
    rx: &EventReceiver,
    session: &mut LiveSession,
    budget: Duration,
    mut f: impl FnMut(&LiveSession) -> bool,
) -> bool {
    let deadline = Instant::now() + budget;
    loop {
        drain_available(rx, session);
        if f(session) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL).await;
    }
}

fn describe(open: &[OpenOrder]) -> String {
    if open.is_empty() {
        return "nothing".into();
    }
    open.iter()
        .map(|o| {
            format!(
                "oid={} {:?} {} @ {}",
                o.order_id.get(),
                o.side,
                o.sz,
                o.limit_px
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Send one cancel *outside* the intent path and record the venue's own bytes.
///
/// Used only for the two deliberate probes and the final sweep, each of which is asking
/// a question about the venue rather than about the planner. Every cancel that is part
/// of a *claim* goes through [`submit_intent`].
async fn probe_cancel(client: &ExchangeClient, id: CancelId, why: &str) -> Option<CancelOutcome> {
    match client.cancel_raw(id).await {
        Ok(raw) => {
            eprintln!("probe [{why}]: {id:?}\n  venue body: {raw}");
            let outcome = parse_cancel_outcomes(&raw)
                .map(|v| v.into_iter().next())
                .unwrap_or(None);
            eprintln!("  parsed as: {outcome:?}");
            outcome
        }
        Err(e) => {
            // A transport failure, not a venue refusal — `cancel_raw` returns the body
            // for both `success` and `error` items, so an `Err` here means the POST
            // itself did not complete.
            eprintln!("probe [{why}]: {id:?} POST failed: {e}");
            None
        }
    }
}

// ── the traded section ───────────────────────────────────────────────────────

/// What the run observed, printed at the end whether or not an assertion failed.
#[derive(Debug, Default)]
struct Transcript {
    lines: Vec<String>,
}

impl Transcript {
    fn note(&mut self, line: impl Into<String>) {
        let line = line.into();
        eprintln!("  · {line}");
        self.lines.push(line);
    }
}

/// Everything between "place a resting order" and "the last replacement is at the
/// venue". Returns `Err` instead of asserting so that the sweep always runs.
#[allow(clippy::too_many_arguments)]
async fn cancel_and_replace(
    client: &ExchangeClient,
    rx: &EventReceiver,
    session: &mut LiveSession,
    rig: &mut IntentRig,
    frames: &Frames,
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    spec: &AssetSpec,
    grid: &InstrumentSpec,
    instruments: Arc<InstrumentTable>,
    t: &mut Transcript,
) -> Result<(), String> {
    // ── phase A: a resting post-only, through the intent path ────────────────
    eprintln!("\n── phase A — rest a post-only through the intent path ──");
    let bbo = session
        .core
        .market()
        .bbo(sym)
        .cloned()
        .ok_or_else(|| "no live BBO to price against".to_string())?;
    let age_ns = SystemClock.now_ns() - bbo.ts_event;
    ensure!(
        age_ns < MAX_BOOK_AGE.as_nanos() as i64,
        "the BBO is {}s old; a band computed from a stale book is not 20% under \
         anything in particular",
        age_ns / 1_000_000_000
    );
    // Through the **production** grid, not a helper this file owns. `Passive` floors a
    // buy, so rounding can only move the band further from the touch.
    let band = grid
        .price
        .quantize(bbo.bid_px * REST_UNDER, Side::Buy, PriceIntent::Passive);
    ensure!(
        band < bbo.bid_px,
        "band {band} is not under the best bid {} - the order would be marketable, and \
         a post-only that crosses is rejected outright",
        bbo.bid_px
    );
    let qty_a = size_for_notional(band, spec.sz_decimals, TARGET_A)?;
    let qty_b = size_for_notional(band, spec.sz_decimals, TARGET_B)?;
    ensure!(
        qty_a != qty_b,
        "the two targets round to the same size {qty_a} on this lot, so 'move the \
         target' would move nothing and phase B would prove nothing"
    );
    eprintln!(
        "book: bid {} / ask {} | band {band} ({}% under) | targets {qty_a} (${}) then {qty_b} (${})",
        bbo.bid_px,
        bbo.ask_px,
        (Decimal::ONE - REST_UNDER) * dec!(100),
        qty_a * band,
        qty_b * band
    );
    // How far behind wall time the core's event clock is. Recorded because it is the
    // number that decides whether a signal is admissible at all: a producer that stamps
    // a record with `last_ts` rather than its own clock starts the record this many
    // milliseconds old, and the reader's ceiling is 2 000.
    t.note(format!(
        "feed: the core's event clock is {} ms behind wall time",
        (SystemClock.now_ns() - session.core.last_ts()) / 1_000_000
    ));
    // Stated up front because it is what makes every "the planner cancelled this"
    // claim below readable: the ADR-0031 sweeper is armed, in the configuration a live
    // session runs, and `IntentRig::no_sweep_yet` checks after each phase that it did
    // not fire.
    let deployed = RuntimeConfig::default();
    t.note(format!(
        "sweeper: armed at max_order_age_ms={}, sweep_interval_ms={} (the default a live \
         session runs); this run is far shorter than the ceiling, and the counter is \
         checked after every phase",
        deployed.intent.max_order_age_ms, deployed.intent.sweep_interval_ms
    ));

    let sig_a = rig.emit(sym, qty_a, band);
    let mut plans = rig.pass(rx, session);
    ensure!(
        plans.len() == 1,
        "the pass produced {} intents, not one: stats {:?}",
        plans.len(),
        rig.src.stats()
    );
    let intent_a = plans.remove(0);
    ensure!(
        intent_a.plan.cancels.is_empty(),
        "the first plan wants to cancel {:?} on an account verified order-free",
        intent_a.plan.cancels
    );
    ensure!(intent_a.plan.orders.len() == 1, "expected one order");
    let req_a: OrderRequest = intent_a.plan.orders[0].clone();
    ensure!(
        req_a.tif == Tif::PostOnly && req_a.side == Side::Buy,
        "urgency 0 must plan a post-only buy, got {:?} {:?}",
        req_a.tif,
        req_a.side
    );
    ensure!(
        req_a.price == Some(band),
        "the planner priced {:?}, not the band {band} - the price_band is what puts this \
         order out of the market",
        req_a.price
    );
    ensure!(req_a.qty == qty_a, "planned {} not {qty_a}", req_a.qty);
    ensure!(
        req_a.cloid == cloid_for(&sig_a),
        "the cloid is not the one derived from the signal"
    );
    t.note(format!(
        "phase A plan: 0 cancels, 1 post-only buy {} @ {} cloid {:#034x}",
        req_a.qty,
        band,
        req_a.cloid.get()
    ));

    rig.submit(client, session, &intent_a).await;
    ensure!(
        rig.health.intent_orders() == 1 && rig.health.intent_failures() == 0,
        "the submit did not place: orders {} failures {}",
        rig.health.intent_orders(),
        rig.health.intent_failures()
    );
    let oid_a = {
        let tracker = session.core.tracker().read().map_err(|_| "poisoned")?;
        let o = tracker
            .order(req_a.cloid)
            .ok_or_else(|| "the tracker lost the order it acked".to_string())?;
        ensure!(
            o.status == OrderStatus::Resting,
            "the ack came back {:?}, not Resting - a post-only 20% under the bid must rest",
            o.status
        );
        o.order_id
            .ok_or_else(|| "the resting ack carried no oid".to_string())?
    };
    t.note(format!("phase A: oid {} resting at {band}", oid_a.get()));

    let open = wait_for_venue(rx, session, account, symbols, sym, FRAME_TIMEOUT, |o| {
        o.iter().any(|x| x.order_id == oid_a)
    })
    .await?;
    let at_venue = open
        .iter()
        .find(|o| o.order_id == oid_a)
        .expect("just matched");
    ensure!(
        at_venue.limit_px == band && at_venue.sz == qty_a,
        "the venue holds oid={} at {} x {}, not {band} x {qty_a}",
        oid_a.get(),
        at_venue.limit_px,
        at_venue.sz
    );
    t.note(format!(
        "phase A at the venue: openOrders reports oid={} {} @ {} (tif {:?}, cloid {:?})",
        at_venue.order_id.get(),
        at_venue.sz,
        at_venue.limit_px,
        at_venue.tif,
        at_venue.cloid
    ));
    for f in frames_for(frames, oid_a) {
        t.note(format!("phase A orderUpdates frame: {f}"));
    }
    rig.no_sweep_yet("phase A")?;

    // ── phase B: move the target ─────────────────────────────────────────────
    eprintln!("\n── phase B — move the target: cancel, then replace ──");
    let sig_b = rig.emit(sym, qty_b, band);
    let mut plans = rig.pass(rx, session);
    ensure!(
        plans.len() == 1,
        "moving the target produced {} intents, not one: stats {:?}",
        plans.len(),
        rig.src.stats()
    );
    let intent_b = plans.remove(0);
    ensure!(
        intent_b.plan.cancels.len() == 1,
        "the planner wanted {} cancels, not one: {:?}",
        intent_b.plan.cancels.len(),
        intent_b.plan.cancels
    );
    // The **negative** branch of the re-addressing rule, and it is what keeps phase D's
    // assertion from being vacuous: this order was acked by *us*, so the tracker knows
    // its cloid is one the venue has seen, and `retarget_cancels` must leave it alone.
    ensure!(
        intent_b.plan.cancels[0]
            == CancelId::Cloid {
                symbol: sym,
                cloid: req_a.cloid
            },
        "an order this session placed must be cancelled by the cloid it minted, got {:?}",
        intent_b.plan.cancels[0]
    );
    ensure!(intent_b.plan.orders.len() == 1, "expected one replacement");
    let req_b = intent_b.plan.orders[0].clone();
    ensure!(
        req_b.qty == qty_b && req_b.price == Some(band) && req_b.tif == Tif::PostOnly,
        "the replacement is {} @ {:?} {:?}, not {qty_b} @ {band} post-only",
        req_b.qty,
        req_b.price,
        req_b.tif
    );
    ensure!(
        req_b.cloid == cloid_for(&sig_b) && req_b.cloid != req_a.cloid,
        "the replacement must carry its own signal's cloid"
    );
    t.note(format!(
        "phase B plan: cancel Cloid({:#034x}) then place {} @ {} cloid {:#034x}",
        req_a.cloid.get(),
        req_b.qty,
        band,
        req_b.cloid.get()
    ));

    let cancels_before = rig.health.intent_cancels();
    rig.submit(client, session, &intent_b).await;
    ensure!(
        rig.health.intent_cancels() == cancels_before + 1,
        "the cancel did not go out: {} cancel failure(s)",
        rig.health.intent_cancel_failures()
    );
    ensure!(
        rig.health.intent_cancel_failures() == 0,
        "a cancel through the intent path failed"
    );
    let oid_b = {
        let tracker = session.core.tracker().read().map_err(|_| "poisoned")?;
        tracker
            .order(req_b.cloid)
            .and_then(|o| o.order_id)
            .ok_or_else(|| "the replacement ack carried no oid".to_string())?
    };
    t.note(format!(
        "phase B: cancel of oid={} acked over REST; replacement rests as oid={}",
        oid_a.get(),
        oid_b.get()
    ));

    let open = wait_for_venue(rx, session, account, symbols, sym, FRAME_TIMEOUT, |o| {
        o.iter().any(|x| x.order_id == oid_b) && !o.iter().any(|x| x.order_id == oid_a)
    })
    .await?;
    t.note(format!(
        "phase B at the venue: {} — the cancelled oid is gone and the replacement is there",
        describe(&open)
    ));
    for f in frames_for(frames, oid_a) {
        t.note(format!(
            "phase B orderUpdates frame for the cancelled oid: {f}"
        ));
    }
    for f in frames_for(frames, oid_b) {
        t.note(format!(
            "phase B orderUpdates frame for the replacement: {f}"
        ));
    }

    rig.no_sweep_yet("phase B")?;

    // Whether the *session* learned the order is gone is a separate question from
    // whether the venue cancelled it, and the answer decides phase C: the leave-it-
    // resting exception needs exactly one working order for the symbol. If the venue
    // publishes no terminal frame, the tracker carries a ghost forever and every later
    // pass cancels it again.
    let learned = wait_for_session(rx, session, FRAME_TIMEOUT, |s| {
        s.status_of(req_a.cloid).is_some_and(|st| st.is_terminal())
    })
    .await;
    t.note(format!(
        "phase B: the tracker {} that oid={} is gone (status {:?}); it now holds {} working \
         order(s) for {COIN}",
        if learned {
            "learned from orderUpdates"
        } else {
            "NEVER LEARNED"
        },
        oid_a.get(),
        session.status_of(req_a.cloid),
        session.open_for(sym).len()
    ));
    ensure!(
        learned,
        "the venue published no terminal frame for the cancelled oid={} within \
         {FRAME_TIMEOUT:?}. That is a real defect and not a flake: the tracker keeps the \
         dead order as working, so the planner cancels a ghost on every later pass and \
         the leave-it-resting exception (which needs exactly one working order) can never \
         fire again.",
        oid_a.get()
    );

    // ── phase C: leave it resting ────────────────────────────────────────────
    eprintln!("\n── phase C — the same target again: nothing must be cancelled ──");
    let working = session.open_for(sym);
    ensure!(
        working.len() == 1,
        "phase C needs exactly one working order and the tracker holds {}: {working:?}",
        working.len()
    );
    let before = rig.src.stats();
    // Deliberately the *same* target and the same band as phase B, on a new signal: a
    // strategy that has not changed its mind. The planner must recognize the resting
    // order as the order it would place and keep it, because a replaced post-only goes
    // to the back of its price level and that is a strict loss.
    let _sig_c = rig.emit(sym, qty_b, band);
    let plans = rig.pass(rx, session);
    let after = rig.src.stats();
    ensure!(
        plans.is_empty(),
        "the planner produced {} intent(s) for a target already resting: {:?}",
        plans.len(),
        plans.iter().map(|p| &p.plan).collect::<Vec<_>>()
    );
    ensure!(
        after.accepted == before.accepted + 1,
        "the record was not even read: accepted {} → {}",
        before.accepted,
        after.accepted
    );
    ensure!(
        after.no_order == before.no_order + 1 && after.planned == before.planned,
        "expected exactly one no-order outcome, got no_order {} → {} planned {} → {}",
        before.no_order,
        after.no_order,
        before.planned,
        after.planned
    );
    // Pinning the *reason* without a counter for it. `AlreadyAtTarget` is unreachable
    // (we are flat and the target is positive), `WithinNoOpBand` needs `noop_band_bps`
    // and the default is 0, and the three counters below are the only other no-order
    // outcomes that get one. What is left is `AlreadyWorking`.
    ensure!(
        after.no_quote == before.no_quote
            && after.precision_refusals == before.precision_refusals
            && after.unknown_precision == before.unknown_precision,
        "the pass produced no order for the wrong reason: {after:?}"
    );
    rig.no_sweep_yet("phase C")?;
    t.note("phase C: the pass read the record, produced no order and no cancel".to_string());

    tokio::time::sleep(VENUE_SETTLE).await;
    let open = venue_open_orders(account, symbols, sym).await?;
    ensure!(
        open.len() == 1 && open[0].order_id == oid_b,
        "the order was disturbed: expected only oid={} still resting, found {}",
        oid_b.get(),
        describe(&open)
    );
    ensure!(
        open[0].limit_px == band && open[0].sz == qty_b,
        "oid={} changed under us: {} @ {}",
        oid_b.get(),
        open[0].sz,
        open[0].limit_px
    );
    t.note(format!(
        "phase C at the venue: oid={} still resting, unchanged, at {} x {} — it kept its \
         place in the queue",
        oid_b.get(),
        open[0].limit_px,
        open[0].sz
    ));

    // ── phase D: an adopted order is cancelled by venue oid ──────────────────
    eprintln!("\n── phase D — restart, adopt the resting order, re-address its cancel ──");
    session.restart();
    rig.restart(instruments);
    // Exactly what `reconcile::publish` puts on the bus, from exactly the same kind of
    // read — the venue's own order list, republished as an `OrderUpdate` for a tracker
    // that has never heard of it.
    let adopted_from = venue_open_orders(account, symbols, sym).await?;
    ensure!(
        adopted_from.len() == 1 && adopted_from[0].order_id == oid_b,
        "the venue no longer holds the single order phase D adopts: {}",
        describe(&adopted_from)
    );
    // Recorded, not asserted, and the first live run is why: `info.rs` documents that
    // "plain `openOrders` does not expose [a cloid] at all", and the venue echoed one.
    // Which id the tracker ends up keying an adopted order by is therefore a fact about
    // the venue on the day, and the claim under test does not depend on it —
    // `retarget_cancels` keys on `adopted`, not on where the cloid came from.
    t.note(format!(
        "phase D: plain openOrders reports cloid {:?} for oid={} (the code's own doc says \
         it reports none — the venue disagrees) and tif {:?}",
        adopted_from[0].cloid,
        oid_b.get(),
        adopted_from[0].tif
    ));
    for o in &adopted_from {
        let ev = Event::Exec(ExecEvent::Order(o.to_order_update()));
        session.on_event(ev.ts_event(), &ev);
    }
    // An id no process ever minted: a bare oid widened to 128 bits, which is what the
    // tracker synthesizes when the venue reports no cloid. The planner's own ids always
    // have bit 127 set (`CLOID_PLANNER_TAG`), precisely so the two spaces cannot collide,
    // so this is guaranteed to be an address the venue has never seen — which is what
    // makes the probe below a real question rather than a coin flip.
    let synthesized = Cloid::new(oid_b.get() as u128);
    let adopted_cloid = {
        let tracker = session.core.tracker().read().map_err(|_| "poisoned")?;
        let o = tracker
            .order_by_id(oid_b)
            .ok_or_else(|| "the restarted tracker did not adopt the resting order".to_string())?;
        ensure!(o.adopted, "the order was not marked adopted");
        ensure!(
            o.tif.is_none() && o.reduce_only.is_none(),
            "an adopted order must report an unknown tif/reduce-only, not a plausible one"
        );
        ensure!(o.order_id == Some(oid_b), "the adopted order lost its oid");
        // The adopted order keeps the venue's **own** placement time, not the moment we
        // met it. `openOrders` carries `timestamp` and `to_order_update` passes it
        // straight through, so an order resting since before a restart is adopted
        // already aged — which is what lets ADR-0031's sweeper judge it on the first
        // pass instead of granting every inherited order a fresh lifetime. Asserted
        // rather than assumed, because the alternative (stamping the adoption) is the
        // obvious implementation and is silently wrong in the direction that keeps a
        // stale quote alive.
        ensure!(
            o.placed_ts == adopted_from[0].ts_event,
            "the adopted order was stamped with the adoption time ({}) instead of the \
             venue's own placement time ({})",
            o.placed_ts,
            adopted_from[0].ts_event
        );
        ensure!(
            o.placed_ts < SystemClock.now_ns(),
            "an order the venue says was placed in the future is not an order we adopted"
        );
        o.cloid
    };
    t.note(format!(
        "phase D: the adopted order carries the venue's own placement time ({} ms ago), \
         not the moment this process met it",
        (SystemClock.now_ns() - adopted_from[0].ts_event) / 1_000_000
    ));
    t.note(format!(
        "phase D: a fresh tracker adopted oid={} under cloid {:#034x} — {} — with the \
         time-in-force unknown, as an order nobody in this process placed must be",
        oid_b.get(),
        adopted_cloid.get(),
        if adopted_cloid == synthesized {
            "synthesized from the oid, an id the venue has never seen"
        } else {
            "the one the venue echoed back"
        }
    ));

    // The restarted session has no book yet; the planner refuses to quote without one.
    let quoted = wait_for_session(rx, session, CONNECT_TIMEOUT, |s| {
        s.core.market().bbo(sym).is_some() && s.core.marks().get(sym).is_some()
    })
    .await;
    ensure!(quoted, "the restarted session never got a book back");

    let qty_c = qty_a;
    let sig_d = rig.emit(sym, qty_c, band);
    let mut plans = rig.pass(rx, session);
    ensure!(
        plans.len() == 1,
        "the restarted pass produced {} intents, not one: stats {:?}",
        plans.len(),
        rig.src.stats()
    );
    let intent_d = plans.remove(0);
    ensure!(
        intent_d.plan.cancels.len() == 1,
        "expected one cancel for the adopted order, got {:?}",
        intent_d.plan.cancels
    );
    // **The claim.** The planner emits `CancelId::Cloid` because a cloid is the only
    // identity it knows; the runtime re-addresses it because the tracker says this order
    // was adopted, and an adopted order's cloid may be one nobody at the venue has ever
    // seen. Sent under that id the cancel fails, and the stale quote it was meant to
    // remove stays resting exactly where somebody else's taker is looking for it.
    ensure!(
        intent_d.plan.cancels[0]
            == CancelId::OrderId {
                symbol: sym,
                order_id: oid_b
            },
        "an adopted order must be cancelled by venue oid, got {:?}",
        intent_d.plan.cancels[0]
    );
    let req_d = intent_d.plan.orders[0].clone();
    t.note(format!(
        "phase D plan: cancel OrderId({}) — re-addressed from Cloid({:#034x}) because the \
         tracker adopted this order — then place {} @ {}",
        oid_b.get(),
        adopted_cloid.get(),
        req_d.qty,
        band
    ));

    // Prove the re-addressing is load-bearing rather than decorative: address a cancel
    // to a **synthesized** cloid and let the venue refuse it. One signed action, and it
    // is the difference between "we changed the id" and "we had to". Deliberately *not*
    // the cloid this order was adopted under — the venue echoed a real one today, so
    // cancelling by that would succeed and would only cancel our own order early. This
    // asks the question the re-addressing exists for: is an id the tracker made up an id
    // the venue will accept?
    let refused = probe_cancel(
        client,
        CancelId::Cloid {
            symbol: sym,
            cloid: synthesized,
        },
        "a synthesized cloid, the kind the tracker mints when the venue reports none",
    )
    .await;
    match refused {
        Some(CancelOutcome::Rejected { ref reason }) => t.note(format!(
            "phase D probe: a cancel addressed to the synthesized cloid {:#034x} was REFUSED \
             by the venue — \"{reason}\". An adopted order keyed by such an id is \
             un-cancellable by cloid, which is why `retarget_cancels` exists.",
            synthesized.get()
        )),
        other => t.note(format!(
            "phase D probe: the venue answered {other:?} to a cancel by the synthesized \
             cloid. If that is a success the order is now gone and the re-addressing was \
             not load-bearing on this venue — read the assertions below with that in mind."
        )),
    }

    let cancels_before = rig.health.intent_cancels();
    let failures_before = rig.health.intent_cancel_failures();
    rig.submit(client, session, &intent_d).await;
    ensure!(
        rig.health.intent_cancels() == cancels_before + 1
            && rig.health.intent_cancel_failures() == failures_before,
        "the re-addressed cancel failed at the venue: cancels {} → {}, failures {} → {}",
        cancels_before,
        rig.health.intent_cancels(),
        failures_before,
        rig.health.intent_cancel_failures()
    );
    let oid_c = {
        let tracker = session.core.tracker().read().map_err(|_| "poisoned")?;
        tracker
            .order(cloid_for(&sig_d))
            .and_then(|o| o.order_id)
            .ok_or_else(|| "the phase D replacement carried no oid".to_string())?
    };
    let open = wait_for_venue(rx, session, account, symbols, sym, FRAME_TIMEOUT, |o| {
        o.iter().any(|x| x.order_id == oid_c) && !o.iter().any(|x| x.order_id == oid_b)
    })
    .await?;
    t.note(format!(
        "phase D at the venue: the adopted oid={} is cancelled and oid={} rests in its \
         place — {}",
        oid_b.get(),
        oid_c.get(),
        describe(&open)
    ));
    for f in frames_for(frames, oid_b) {
        t.note(format!(
            "phase D orderUpdates frame for the adopted oid: {f}"
        ));
    }
    rig.no_sweep_yet("phase D")?;

    // ── phase E: cancelling an oid that is already gone ──────────────────────
    eprintln!("\n── phase E — what the venue says about a cancel it cannot honour ──");
    let gone = probe_cancel(
        client,
        CancelId::OrderId {
            symbol: sym,
            order_id: oid_b,
        },
        "an oid that was cancelled a moment ago",
    )
    .await;
    match gone {
        Some(CancelOutcome::Rejected { ref reason }) => t.note(format!(
            "phase E: cancelling the already-gone oid={} is refused per-item, inside an \
             HTTP 200 — \"{reason}\". `submit_intent` treats this as non-fatal and still \
             places the replacement, which is the right call: the overwhelmingly common \
             cause is that the order is already gone.",
            oid_b.get()
        )),
        other => t.note(format!(
            "phase E: the venue answered {other:?} to a cancel of an already-gone oid"
        )),
    }

    Ok(())
}

// ── the sweep ────────────────────────────────────────────────────────────────

/// Leave the account order-free and flat in `sym`, and prove it by reading the venue
/// back more than once.
///
/// Orders first, position second, and never the other way round: a resting buy that
/// filled while we were closing would put the position straight back, and a flatten that
/// races itself never terminates. In this test nothing can fill — every order rests 20 %
/// under the bid — but "cannot fill" is an argument and a flat read is an observation,
/// so the reads happen anyway.
async fn sweep(
    client: &ExchangeClient,
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    t: &mut Transcript,
) -> Result<(), String> {
    for round in 1..=3u32 {
        let open = match fetch_frontend_open_orders(TESTNET_INFO, account, None, symbols).await {
            Ok(o) => o,
            Err(e) => {
                t.note(format!(
                    "sweep round {round}: could not read open orders ({e})"
                ));
                tokio::time::sleep(VENUE_SETTLE).await;
                continue;
            }
        };
        let ours: Vec<_> = open.items.iter().filter(|o| o.symbol_id == sym).collect();
        if ours.is_empty() {
            t.note(format!("sweep round {round}: no {COIN} order is resting"));
            break;
        }
        for o in ours {
            // Through `cancel_raw` rather than `cancel`, so the *successful* body is in
            // the transcript too. Everything else in this file records what a refusal
            // looks like; a claim about the reply shape needs the other half.
            let outcome = probe_cancel(client, o.cancel_id(), "sweep").await;
            t.note(format!(
                "sweep: cancel of oid={} → {outcome:?}",
                o.order_id.get()
            ));
        }
        tokio::time::sleep(VENUE_SETTLE).await;
    }

    // Two reads, VENUE_SETTLE apart. One read taken a round trip ahead of the venue's
    // own bookkeeping reports whatever was true before the last action landed.
    let mut flats = 0u32;
    let mut last = Decimal::ZERO;
    for round in 1..=3u32 {
        match venue_position(account, symbols, sym).await {
            Ok(q) => {
                last = q;
                t.note(format!(
                    "sweep: clearinghouseState read {round} — {COIN} position {q}"
                ));
                if q.is_zero() {
                    flats += 1;
                    if flats >= 2 {
                        break;
                    }
                } else {
                    flats = 0;
                }
            }
            Err(e) => {
                // "We could not check" is not "there is nothing there".
                flats = 0;
                t.note(format!(
                    "sweep: clearinghouseState read {round} failed ({e})"
                ));
            }
        }
        tokio::time::sleep(VENUE_SETTLE).await;
    }

    let open = venue_open_orders(account, symbols, sym).await?;
    if flats >= 2 && open.is_empty() {
        t.note("sweep: the account is flat and order-free, confirmed twice".to_string());
        return Ok(());
    }
    let rule = "!".repeat(78);
    Err(format!(
        "{rule}\n\
         !! THE ACCOUNT IS NOT CLEAN. {COIN} position {last}, still resting: {}.\n\
         !!\n\
         !! This is exposure at a venue, not a test failure. Deal with it first:\n\
         !!     ./run.sh wallet\n\
         !! then cancel/close from the testnet UI. Do not re-run this test until the\n\
         !! account is flat and order-free: it refuses to start otherwise, and for good\n\
         !! reason — the sweep cancels every {COIN} order it finds.\n\
         {rule}",
        describe(&open)
    ))
}

// ── the live test ────────────────────────────────────────────────────────────

/// Place a resting post-only through the intent path, move the target, and watch the
/// cancel and its replacement reach the venue. See the module docs before running it.
#[tokio::test]
#[ignore = "places REAL orders and sends REAL cancels on Hyperliquid testnet; needs a funded .env"]
async fn a_cancel_reaches_the_venue_through_the_intent_path_on_testnet() {
    // 1. Network guard, before the key is read and before anything is signed.
    if let Err(why) = require_testnet(std::env::var("AXON_HL_NETWORK").ok().as_deref()) {
        panic!("{why}");
    }
    // The **master** address: user channels and `/info` keyed on the agent that signs
    // return empty results forever, with no error to say why.
    let account = std::env::var("AXON_HL_ACCOUNT_ADDRESS")
        .expect("AXON_HL_ACCOUNT_ADDRESS (the master account, not the agent wallet)");
    let signer = HlSigner::from_env(false).expect("AXON_HL_SECRET_KEY (testnet)");
    assert!(!signer.is_mainnet(), "the signer must sign testnet actions");
    eprintln!(
        "network: testnet | account {account} | signer {}",
        signer.address()
    );

    // 2. Resolve the asset from the venue's own universe. One `meta` read serves the
    //    SymbolMap, the instrument grids and the index/delisted facts.
    let meta = post_testnet_info(serde_json::json!({ "type": "meta" }))
        .await
        .expect("meta");
    let universe = decode_universe(&meta).expect("meta decodes to a universe");
    let symbols = universe.symbols.clone();
    let spec = asset_spec(&meta, COIN).expect("coin in the perp universe");
    let sym = symbols.id(COIN).expect("coin in the symbol map");
    let grid = *universe
        .instruments
        .get(sym)
        .expect("the universe declares this coin's grid");
    // **One** table, shared by the planner and the encoder. Two that can drift apart is
    // a planner rounding to one grid while the encoder refuses against another, and
    // every refusal then reads in the log exactly like a venue rejection.
    let instruments = Arc::new(universe.instruments);
    let client = ExchangeClient::testnet(signer)
        .expect("testnet ExchangeClient")
        .with_instruments(instruments.clone())
        .with_account(account.clone(), symbols.clone());
    assert_eq!(
        sym,
        SymbolId::new(spec.index),
        "the symbol map and the universe disagree about {COIN}'s asset index; an order \
         signed with the wrong index trades a different coin"
    );
    assert!(
        !spec.delisted,
        "{COIN} is delisted: it keeps its index in the universe and rejects every order"
    );
    eprintln!(
        "asset: {COIN} index {} szDecimals {} lot {} tick@60k {}",
        spec.index,
        spec.sz_decimals,
        grid.size.increment(),
        grid.price.tick_at(Decimal::from(60_000))
    );

    // 3. Preconditions. The sweep at the end cancels *everything* in this coin, so
    //    starting on top of somebody else's resting order would cancel their quote.
    let state = fetch_user_state(TESTNET_INFO, &account, &symbols)
        .await
        .expect("clearinghouseState");
    eprintln!(
        "account: equity {} withdrawable {} positions {}",
        state.account.equity,
        state.account.withdrawable,
        state.positions.len()
    );
    assert!(
        state.account.equity >= MAX_NOTIONAL,
        "equity {} is too small to margin a ${MAX_NOTIONAL} order",
        state.account.equity
    );
    assert!(
        state
            .positions
            .iter()
            .all(|p| p.symbol_id != sym || p.is_flat()),
        "the account already holds a {COIN} position; flatten it first"
    );
    let open = fetch_frontend_open_orders(TESTNET_INFO, &account, None, &symbols)
        .await
        .expect("frontendOpenOrders");
    assert!(
        !open.items.iter().any(|o| o.symbol_id == sym),
        "the account already has a resting {COIN} order; cancel it first"
    );

    // 4. Subscribe BEFORE anything is placed. `orderUpdates` never snapshots, so a
    //    transition that happens in the gap between placing and subscribing is gone.
    let (tx, rx) = bus(BUS_CAPACITY);
    let md = HyperliquidMarketData::testnet(symbols.clone(), vec![COIN.into()], tx);
    md.subscribe_user_channels(account.as_str());
    md.subscribe_coin(Feed::Bbo, COIN);
    let ws = tokio::spawn(async move { md.run_forever().await });
    let (tap, frames) = tap_order_updates(&account)
        .await
        .expect("a raw orderUpdates tap");

    let mut session = LiveSession::new();
    let health = Arc::new(SessionHealth::new(0));
    let mut rig = IntentRig::new(instruments.clone(), health.clone());

    let connected = wait_for_session(&rx, &mut session, CONNECT_TIMEOUT, |s| {
        s.core.market().bbo(sym).is_some()
    })
    .await;
    assert!(
        connected,
        "no BBO within {CONNECT_TIMEOUT:?}: the socket is not delivering, so nothing \
         below could observe a cancel either"
    );
    tokio::time::sleep(SNAPSHOT_SETTLE).await;
    drain_available(&rx, &mut session);
    eprintln!(
        "subscribed: userFills + orderUpdates for {account}; snapshot replayed {} fill(s)",
        session.fills.len()
    );

    // 5. The traded section. From here to the sweep, failures are values, never panics.
    let mut t = Transcript::default();
    let traded = cancel_and_replace(
        &client,
        &rx,
        &mut session,
        &mut rig,
        &frames,
        &account,
        &symbols,
        sym,
        &spec,
        &grid,
        instruments.clone(),
        &mut t,
    )
    .await;

    // 6. Sweep, whatever happened above.
    let swept = sweep(&client, &account, &symbols, sym, &mut t).await;
    tap.abort();
    ws.abort();

    // 7. Report. The transcript prints whichever way the run went — a failed assertion
    //    with no record of what the venue actually said is a run that has to be repeated
    //    to learn anything, and every repeat costs actions.
    eprintln!("\n──────── venue transcript ────────");
    for line in &t.lines {
        eprintln!("{line}");
    }
    eprintln!("\n──────── every orderUpdates frame, verbatim ────────");
    for f in frames.lock().map(|f| f.clone()).unwrap_or_default() {
        eprintln!("{f}");
    }
    eprintln!(
        "\nintent counters: orders {} cancels {} place-failures {} cancel-failures {}",
        health.intent_orders(),
        health.intent_cancels(),
        health.intent_failures(),
        health.intent_cancel_failures()
    );
    eprintln!("intent stats: {:?}", rig.src.stats());

    // An unclean account outranks a failed assertion: one is a broken test, the other is
    // an order sitting at a venue.
    if let Err(why) = swept {
        panic!("{why}\n\nthe traded section reported: {traded:?}");
    }
    if let Err(why) = traded {
        panic!("live cancel verification failed: {why}");
    }
    eprintln!(
        "\nPASS: a cancel was planned, re-addressed, sent and observed at Hyperliquid \
         testnet; the account is flat and order-free."
    );
}

// ── offline tests: the rules this file bets on, checked without a network ────

/// A slice of the live testnet `meta` (captured 2026-07-25): SOL at index 0, a delisted
/// asset in the middle, and BTC at index 3 — which is what makes it worth pinning,
/// because BTC is index 0 on mainnet.
const META_SAMPLE: &str = r#"{"universe":[
    {"szDecimals":2,"name":"SOL","maxLeverage":10,"marginTableId":10},
    {"szDecimals":1,"name":"MATIC","maxLeverage":50,"marginTableId":50,"isDelisted":true},
    {"szDecimals":2,"name":"ATOM","maxLeverage":10,"marginTableId":55},
    {"szDecimals":5,"name":"BTC","maxLeverage":40,"marginTableId":54},
    {"szDecimals":4,"name":"ETH","maxLeverage":25,"marginTableId":53}]}"#;

#[test]
fn a_non_testnet_network_is_refused_before_anything_is_signed() {
    assert!(require_testnet(Some("testnet")).is_ok());
    for hostile in [
        Some("mainnet"),
        Some("Testnet"),
        Some("prod"),
        Some(""),
        None,
    ] {
        let err = require_testnet(hostile).expect_err(&format!("{hostile:?} must be refused"));
        assert!(
            err.contains("REFUSING TO RUN"),
            "the refusal must be loud: {err}"
        );
        assert!(err.contains("AXON_HL_NETWORK"), "and must say what to fix");
    }
}

#[test]
fn the_asset_index_comes_from_the_live_universe_not_from_memory() {
    let btc = asset_spec(META_SAMPLE, "BTC").unwrap();
    assert_eq!(btc.index, 3);
    assert_eq!(btc.sz_decimals, 5);
    assert!(!btc.delisted);
    assert!(asset_spec(META_SAMPLE, "MATIC").unwrap().delisted);
    assert!(asset_spec(META_SAMPLE, "NOTACOIN").is_err());
    assert!(asset_spec("not json", "BTC").is_err());
}

#[test]
fn the_two_targets_must_round_to_different_sizes_or_phase_b_moves_nothing() {
    // The failure this guards: pick two notionals close enough that the lot collapses
    // them, and "move the target" produces the *same* order. The planner then takes the
    // leave-it-resting branch, no cancel is sent, and the run passes phase B by doing
    // exactly what phase C exists to test — a green run over an untested cancel path.
    let px = dec!(51500); // testnet BTC ~64 400, 20% under
    let a = size_for_notional(px, 5, TARGET_A).unwrap();
    let b = size_for_notional(px, 5, TARGET_B).unwrap();
    assert_ne!(
        a, b,
        "$12 and $20 must be two different sizes on a 1e-5 lot"
    );
    assert!(a * px >= MIN_NOTIONAL && b * px >= MIN_NOTIONAL);
    assert!(a * px <= MAX_NOTIONAL && b * px <= MAX_NOTIONAL);
    // On a whole-coin lot they collapse, and the helper's ceiling catches it first.
    let err = size_for_notional(dec!(1000), 0, TARGET_A).unwrap_err();
    assert!(err.contains("ceiling"), "{err}");
    assert!(err.contains("finer lot"), "and it says what to do: {err}");
}

#[test]
fn a_size_that_lot_rounding_puts_under_the_venue_minimum_is_refused_not_sent() {
    // Rounding away from zero is what makes this reachable only through a bad price, and
    // the check is here because a $9.99 order is not merely rejected — it is rejected in
    // a way that reads from outside exactly like the leave-it-resting decision.
    assert!(size_for_notional(Decimal::ZERO, 5, TARGET_A).is_err());
    assert!(size_for_notional(dec!(-1), 5, TARGET_A).is_err());
    let ok = size_for_notional(dec!(51500), 5, TARGET_A).unwrap();
    assert_eq!(ok, dec!(0.00024), "rounded up, never down");
}

#[test]
fn the_signal_this_file_emits_is_the_one_the_planner_prices_out_of_the_market() {
    // The band is the whole mechanism: without it urgency 0 joins the bid, and an order
    // at the touch on a live book is an order that can fill — which would open a position
    // this test has no machinery to reason about. Asserted on the record itself, because
    // a transposed argument in a nine-positional constructor is exactly the mistake
    // `Signal::target_position`'s own docs warn about.
    let sym = SymbolId::new(3);
    let sig = target_signal(7, 1_234_000_000_000, sym, dec!(0.00024), dec!(51500));
    assert_eq!(sig.seq, 7);
    assert_eq!(sig.ts_event, 1_234_000_000_000);
    assert_eq!(sig.symbol_id, 3);
    assert_eq!(sig.urgency, 0, "post-only at the near touch");
    assert_eq!(sig.target_qty, 24_000, "0.00024 at the contract's 8 dp");
    assert_eq!(sig.price_band, 5_150_000_000_000, "51500 at the same scale");
    assert_eq!(sig.ttl_ms, 0, "the operator's ceiling, not 'never expires'");
    assert!(!sig.is_reduce_only() && !sig.is_close());
    assert_eq!(sig.max_order_age_ms, 0, "the operator's ceiling here too");
}

#[test]
fn the_push_source_hands_records_out_once_and_in_order() {
    // The production source is a ring another process writes; this stands in for it, so
    // a bug here would show up as the reader refusing records for a reason that has
    // nothing to do with the venue. Re-reading a record would trip the `stale_seq` check
    // and the whole run would go quiet.
    let sym = SymbolId::new(3);
    let src = PushSource::default();
    let mut reader = src.clone();
    assert!(reader.next_signal().is_none(), "empty is not an error");
    src.push(target_signal(1, 10, sym, dec!(1), dec!(2)));
    src.push(target_signal(2, 20, sym, dec!(3), dec!(4)));
    assert_eq!(reader.next_signal().map(|s| s.seq), Some(1));
    assert_eq!(reader.next_signal().map(|s| s.seq), Some(2));
    assert!(reader.next_signal().is_none());
    assert!(reader.ensure(0), "an in-process queue is always attached");
}

// ── offline rehearsals: the same intent path, without a venue ────────────────
//
// These are not a substitute for the live run and are not claimed as one. What they
// are is the difference between discovering a sequencing mistake for free and
// discovering it three signed actions into a testnet session — and, for the two claims
// whose whole content is *which* `CancelId` came out of the planner, they pin both
// branches at once, which is the only way an assertion about a re-addressing cannot
// pass by accident.

/// The BTC-shaped table these rehearsals plan against, built through the **production**
/// decoder. A hand-rolled grid here would let a rehearsal agree with a planner that the
/// live venue disagrees with.
fn btc_table() -> (SymbolId, Arc<InstrumentTable>, InstrumentSpec) {
    let u = decode_universe(r#"{"universe":[{"name":"BTC","szDecimals":5}]}"#)
        .expect("the decoder builds a grid from szDecimals");
    let sym = u.symbols.id("BTC").expect("BTC at index 0");
    let spec = *u.instruments.get(sym).expect("index 0");
    (sym, Arc::new(u.instruments), spec)
}

/// Feed one synthetic top of book into a session.
fn quote(session: &mut LiveSession, sym: SymbolId, ts: Nanos, bid: Decimal, ask: Decimal) {
    use axon_core::{Bbo, MarketEvent};
    let ev = Event::Market(MarketEvent::Bbo(Bbo {
        symbol_id: sym,
        bid_px: bid,
        bid_sz: dec!(1),
        ask_px: ask,
        ask_sz: dec!(1),
        ts_event: ts,
    }));
    session.on_event(ts, &ev);
}

/// A venue that records what it was asked to do, in the order it was asked.
///
/// The ordering is the point: Hyperliquid processes `cancel > post-only > GTC > IOC`
/// inside one block, so a cancel and its replacement submitted together cannot both be
/// live — but only if the cancel is submitted *first*. Reversed, there is a window in
/// which we hold double the intended exposure, and it is invisible from any counter.
struct RecordingClient {
    calls: Mutex<Vec<String>>,
    next_oid: Mutex<u64>,
    caps: axon_providers::Capabilities,
}

impl RecordingClient {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            next_oid: Mutex::new(1_000),
            caps: axon_provider_hyperliquid::capabilities(),
        }
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }
}

#[async_trait::async_trait]
impl ExecutionClient for RecordingClient {
    fn capabilities(&self) -> &axon_providers::Capabilities {
        &self.caps
    }

    async fn place_order(
        &self,
        req: OrderRequest,
    ) -> Result<axon_providers::OrderAck, axon_providers::ProviderError> {
        let mut n = self.next_oid.lock().expect("oid lock");
        *n += 1;
        let oid = OrderId::new(*n);
        // `normalize()` because `Decimal` keeps the scale its arithmetic produced —
        // `0.00024000` and `0.00024` are equal numbers with different `Display`s, and a
        // transcript that compared their text would fail on a difference nobody made.
        self.calls.lock().expect("calls lock").push(format!(
            "place {} @ {} → oid {}",
            req.qty.normalize(),
            req.price
                .map(|p| p.normalize().to_string())
                .unwrap_or_default(),
            oid.get()
        ));
        Ok(axon_providers::OrderAck {
            cloid: req.cloid,
            order_id: Some(oid),
            status: OrderStatus::Resting,
        })
    }

    async fn place_batch(
        &self,
        _reqs: Vec<OrderRequest>,
    ) -> Result<Vec<axon_providers::OrderAck>, axon_providers::ProviderError> {
        unreachable!("the intent path submits one order at a time")
    }

    async fn cancel(
        &self,
        id: CancelId,
    ) -> Result<axon_providers::CancelAck, axon_providers::ProviderError> {
        let (line, ack) = match id {
            CancelId::OrderId { order_id, .. } => (
                format!("cancel oid {}", order_id.get()),
                axon_providers::CancelAck {
                    cloid: None,
                    order_id: Some(order_id),
                },
            ),
            CancelId::Cloid { cloid, .. } => (
                format!("cancel cloid {:#034x}", cloid.get()),
                axon_providers::CancelAck {
                    cloid: Some(cloid),
                    order_id: None,
                },
            ),
        };
        self.calls.lock().expect("calls lock").push(line);
        Ok(ack)
    }

    async fn cancel_all(&self) -> Result<(), axon_providers::ProviderError> {
        unreachable!("the intent path never sweeps")
    }

    async fn modify(
        &self,
        _id: CancelId,
        _req: OrderRequest,
    ) -> Result<axon_providers::OrderAck, axon_providers::ProviderError> {
        unreachable!("the planner emits cancel+place, never a modify")
    }
}

/// Event times are derived from the wall clock on purpose.
///
/// [`MarkCache`] ages a price against its own *liveness* clock, and [`IntentRig::pass`]
/// advances that clock with `SystemClock::now_ns()` exactly as `axon_runtime::core::run`
/// does — a dead feed is precisely the case where event time would freeze and call every
/// stale price fresh. Synthetic timestamps near zero would therefore be expired by
/// eighteen digits the first time a pass looked at them, and every rehearsal below would
/// report `no_quote`.
fn rehearsal_clock() -> Nanos {
    SystemClock.now_ns()
}

#[tokio::test]
async fn a_target_that_has_not_moved_leaves_the_order_resting_and_sends_nothing() {
    // The leave-it-resting exception, offline. A cancel path that cancels
    // unconditionally passes every "did the cancel go out" test and is wrong: it pays
    // queue position on every signal to re-place an order it already has. Post-only, the
    // replacement goes to the back of its price level, so the cost is real and recurring.
    let (sym, table, grid) = btc_table();
    let (_tx, rx) = bus(64);
    let mut session = LiveSession::new();
    let health = Arc::new(SessionHealth::new(0));
    let mut rig = IntentRig::new(table, health);
    let client = RecordingClient::new();

    let t0 = rehearsal_clock();
    quote(&mut session, sym, t0, dec!(64400), dec!(64410));
    let band = grid
        .price
        .quantize(dec!(64400) * REST_UNDER, Side::Buy, PriceIntent::Passive);
    let qty = size_for_notional(band, 5, TARGET_A).expect("a legal size");

    rig.emit(sym, qty, band);
    let mut plans = rig.pass(&rx, &mut session);
    assert_eq!(plans.len(), 1, "the first target must produce an order");
    let intent = plans.remove(0);
    assert!(intent.plan.cancels.is_empty(), "nothing was working yet");
    assert_eq!(
        intent.plan.orders[0].price,
        Some(band),
        "priced at the band"
    );
    assert_eq!(intent.plan.orders[0].tif, Tif::PostOnly);
    rig.submit(&client, &session, &intent).await;
    assert_eq!(client.calls().len(), 1, "one placement reached the venue");

    // Half a second later the market has moved and the strategy has not changed its mind.
    quote(
        &mut session,
        sym,
        t0 + 500_000_000,
        dec!(64405),
        dec!(64415),
    );
    let before = rig.src.stats();
    rig.emit(sym, qty, band);
    let plans = rig.pass(&rx, &mut session);
    let after = rig.src.stats();

    assert!(
        plans.is_empty(),
        "the resting order already is the order we would place: {:?}",
        plans.iter().map(|p| &p.plan).collect::<Vec<_>>()
    );
    assert_eq!(after.accepted, before.accepted + 1, "the record was read");
    assert_eq!(after.no_order, before.no_order + 1);
    assert_eq!(after.planned, before.planned);
    assert_eq!(after.no_quote, before.no_quote, "the book was fine");
    assert_eq!(after.precision_refusals, before.precision_refusals);
    assert_eq!(
        client.calls().len(),
        1,
        "nothing further reached the venue - no cancel, no replacement"
    );
}

#[tokio::test]
async fn a_moved_target_cancels_before_it_replaces_and_re_addresses_only_what_it_adopted() {
    // Both branches of the re-addressing rule, in one test, because either alone can
    // pass by accident: assert only the adopted case and a `retarget_cancels` that
    // rewrote *every* cancel would look correct; assert only our own and one that
    // rewrote none would.
    let (sym, table, grid) = btc_table();
    let (_tx, rx) = bus(64);
    let mut session = LiveSession::new();
    let health = Arc::new(SessionHealth::new(0));
    let mut rig = IntentRig::new(table.clone(), health);
    let client = RecordingClient::new();

    let t0 = rehearsal_clock();
    quote(&mut session, sym, t0, dec!(64400), dec!(64410));
    let band = grid
        .price
        .quantize(dec!(64400) * REST_UNDER, Side::Buy, PriceIntent::Passive);
    let qty_a = size_for_notional(band, 5, TARGET_A).expect("a legal size");
    let qty_b = size_for_notional(band, 5, TARGET_B).expect("a legal size");

    let sig_a = rig.emit(sym, qty_a, band);
    let mut plans = rig.pass(&rx, &mut session);
    let first = plans.remove(0);
    rig.submit(&client, &session, &first).await;
    let oid = session
        .core
        .tracker()
        .read()
        .unwrap()
        .order(cloid_for(&sig_a))
        .and_then(|o| o.order_id)
        .expect("the ack carried an oid");

    // ── our own order: cancelled by the cloid we minted ──
    quote(
        &mut session,
        sym,
        t0 + 500_000_000,
        dec!(64400),
        dec!(64410),
    );
    rig.emit(sym, qty_b, band);
    let mut plans = rig.pass(&rx, &mut session);
    assert_eq!(plans.len(), 1);
    let moved = plans.remove(0);
    assert_eq!(
        moved.plan.cancels,
        vec![CancelId::Cloid {
            symbol: sym,
            cloid: cloid_for(&sig_a)
        }],
        "the venue has seen this cloid; re-addressing it would throw away the one \
         identity that survives a partial submit"
    );
    rig.submit(&client, &session, &moved).await;
    assert_eq!(
        client.calls(),
        vec![
            format!(
                "place {} @ {} → oid {}",
                qty_a.normalize(),
                band.normalize(),
                oid.get()
            ),
            format!("cancel cloid {:#034x}", cloid_for(&sig_a).get()),
            format!(
                "place {} @ {} → oid {}",
                qty_b.normalize(),
                band.normalize(),
                oid.get() + 1
            ),
        ],
        "the cancel must be on the wire before its replacement: reversed, both orders \
         are live in the same block and we hold double the intended exposure"
    );

    // ── an adopted order: cancelled by the venue's oid ──
    // The restart is the honest way to get an order this session did not place: forget
    // having placed it, then meet the venue's own order list. Plain `openOrders` reports
    // no cloid, so the tracker synthesizes one from the oid — an id the venue has never
    // seen, and a cancel sent under it fails while the stale quote stays resting.
    let live_oid = OrderId::new(oid.get() + 1);
    session.restart();
    rig.restart(table);
    let adopted = OrderUpdate {
        symbol_id: sym,
        order_id: live_oid,
        cloid: None,
        side: Side::Buy,
        status: OrderStatus::Resting,
        price: Some(band),
        orig_qty: qty_b,
        remaining_qty: qty_b,
        cancel_reason: None,
        ts_event: t0 + 600_000_000,
    };
    let ev = Event::Exec(ExecEvent::Order(adopted));
    session.on_event(ev.ts_event(), &ev);
    let synthesized = Cloid::new(live_oid.get() as u128);
    {
        let t = session.core.tracker().read().unwrap();
        let o = t.order(synthesized).expect("the restart adopted it");
        assert!(o.adopted);
        assert_eq!(o.tif, None, "the venue reports no time-in-force");
    }

    quote(
        &mut session,
        sym,
        t0 + 700_000_000,
        dec!(64400),
        dec!(64410),
    );
    rig.emit(sym, qty_a, band);
    let mut plans = rig.pass(&rx, &mut session);
    assert_eq!(plans.len(), 1);
    let after_restart = plans.remove(0);
    assert_eq!(
        after_restart.plan.cancels,
        vec![CancelId::OrderId {
            symbol: sym,
            order_id: live_oid
        }],
        "an adopted order's cloid is one we synthesized; the venue would not recognize it"
    );
}

#[test]
fn the_frame_filter_picks_out_one_orders_frames_and_no_others() {
    // The transcript is the deliverable, so a filter that quietly matched the wrong oid
    // would put another order's lifecycle under ours. `"oid":N` and not `N`: a bare
    // number matches a timestamp, a size, or the tail of a longer oid.
    let frames: Frames = Arc::new(Mutex::new(vec![
        r#"{"channel":"orderUpdates","data":[{"order":{"coin":"BTC","oid":56968034936},"status":"open","statusTimestamp":1}]}"#.into(),
        r#"{"channel":"orderUpdates","data":[{"order":{"coin":"BTC","oid":56968034937},"status":"canceled","statusTimestamp":2}]}"#.into(),
        r#"{"channel":"subscriptionResponse","data":{}}"#.into(),
    ]));
    let ours = frames_for(&frames, OrderId::new(56968034936));
    assert_eq!(ours.len(), 1);
    assert!(ours[0].contains("\"status\":\"open\""));
    assert!(frames_for(&frames, OrderId::new(3493)).is_empty());
    assert_eq!(frames_for(&frames, OrderId::new(56968034937)).len(), 1);
}
