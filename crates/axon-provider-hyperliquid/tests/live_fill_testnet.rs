//! Live **fill** verification on Hyperliquid testnet — the last open Phase-3 item.
//!
//! ## What this proves that `place_then_cancel_on_testnet` does not
//!
//! That test (in `src/exchange/mod.rs`) rests a post-only buy 20% under the bid and
//! cancels it. It proves signing, the nonce, the REST envelope and response parsing —
//! and, by construction, **nothing about an execution**. An order priced where it can
//! never trade is never margin-checked, never fills, and never produces a `userFills`
//! frame, an `orderUpdates` transition, or a position. Everything downstream of a
//! fill has therefore only ever met captured frames:
//!
//! - the `userFills` / `orderUpdates` decoders against *this account's* real payloads;
//! - that a subscription made with the **master** address actually delivers (an agent
//!   address returns empty forever, with no error — indistinguishable from an idle
//!   account, which is why an idle-looking silence here is not evidence of anything);
//! - `OrderTracker` attribution: that the venue echoes our `cloid` on the fill, that
//!   the fill lands on the order the ack created rather than counting as an orphan;
//! - that the tracker's fill-derived position equals the venue's own `clearinghouseState`
//!   `szi` — the reconciliation claim ADR-0010 is built on, checked against the venue
//!   rather than against a fixture;
//! - that a `reduce_only` close actually flattens.
//!
//! This test crosses the spread on purpose, watches the fill arrive on the account
//! channels it subscribed to *before* sending, reconciles it, and then flattens.
//!
//! ## What it costs
//!
//! Testnet play money: one taker round trip of ~$12 notional. The spread crossed twice
//! plus taker fees is a few cents of the account's 999 mock USDC. It spends about six
//! actions of the address's rate budget (base cap 10,000/day) and one WS connection,
//! and takes well under a minute. It places **real orders at a real venue** — that is
//! the point, and it is why the decision to run it belongs to an operator.
//!
//! ## What it leaves behind
//!
//! The account **flat**, verified by reading `clearinghouseState` back — the flatten
//! runs even when an assertion above it has already failed, which is why the traded
//! section returns errors instead of asserting. What does survive: two filled orders
//! and their fills in the account's permanent history, and a balance a few cents lower.
//! If the flatten cannot finish it fails loudly and prints what is still open; do not
//! ignore that, and do not re-run until `./run.sh wallet` shows the account flat again.
//!
//! "Verified" is a narrower word here than it looks, and [`FlattenPlan`] is what keeps
//! it honest. `clearinghouseState` trails a fill by a round trip, which is the whole
//! reason [`FLATTEN_SETTLE`] exists — and it trails an *open* by exactly as much. On
//! the one path the flatten was built for, a placement whose reply never came back, a
//! read taken milliseconds later can report flat about an account that is long. So a
//! flat read is only taken as proof once the venue has been *seen* holding something;
//! when it never is, the report says "a position was never seen open" rather than "the
//! account was flattened", because those are not the same sentence.
//!
//! ## Running it
//!
//! ```text
//! ./run.sh wallet     # first: agent approved+valid, accountValue well over $10
//! bash scripts/with-env.sh cargo test -p axon-provider-hyperliquid \
//!     --test live_fill_testnet -- --ignored --nocapture
//! ```
//!
//! `--nocapture` because the running commentary — resolved asset, book, size, oid,
//! every fill, the flatten — is the evidence; without it a pass prints nothing.
//! Secrets arrive through `.env` only, never argv (`ps` is world-readable).
//!
//! It refuses to start unless `AXON_HL_NETWORK` is exactly `testnet`, before the key
//! is read and before anything is signed, and it refuses to start on an account that
//! already holds a position or a resting order in this coin — the flatten at the end
//! would close exposure that is not ours, and the reconciliation assertions would be
//! measuring someone else's position.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axon_core::{
    bus, drain_available, Clock, Cloid, Decimal, Event, EventHandler, EventReceiver, ExecEvent,
    Fill, Liquidity, Nanos, OrderId, OrderStatus, OrderUpdate, Position, Side, SymbolId,
    SystemClock, Tif,
};
use axon_execution::OrderTracker;
use axon_marketdata::MarketDataProcessor;
use axon_provider_hyperliquid::ws::{decode_universe, fetch_l2_snapshot, TESTNET_INFO};
use axon_provider_hyperliquid::{
    fetch_frontend_open_orders, fetch_user_state, ExchangeClient, HlSigner, HyperliquidMarketData,
    SymbolMap,
};
use axon_providers::{ExecutionClient, Feed, InstrumentSpec, OrderRequest, PriceIntent};
use rust_decimal::RoundingStrategy;
use rust_decimal_macros::dec;
use std::sync::Arc;

// ── what this test trades, and the bounds it refuses to leave ────────────────

/// The instrument. BTC because testnet's BTC book is the deepest one there: a thin
/// book turns a 50 bp cross into a partial fill, and a partial proves the same thing
/// far less legibly.
const COIN: &str = "BTC";

/// Target order value in USDC. Comfortably over the venue's minimum so a tick of
/// price movement between reading the book and the order landing cannot drop it under.
const TARGET_NOTIONAL: Decimal = dec!(12);

/// Hyperliquid's minimum order notional. Under it the venue answers
/// `minTradeNtlRejected` and the run proves nothing at all.
const MIN_NOTIONAL: Decimal = dec!(10);

/// The most this test will ever send. Not a risk limit — a tripwire for **lot-size
/// rounding**: on an asset with `szDecimals: 0` a $12 target rounds up to one whole
/// unit, and one whole unit of a $1,000 asset is a $1,000 order. Refusing here beats
/// discovering it from a filled order.
const MAX_NOTIONAL: Decimal = dec!(50);

/// How far through the book to price, as a fraction. Deep enough to sweep several
/// levels so the order is genuinely marketable, shallow enough to stay far inside the
/// venue's limit-vs-oracle band (which rejects anything ~80% away).
const CROSS: Decimal = dec!(0.005);

// ── timings (wall-clock deadlines, not ordering keys) ────────────────────────

const POLL: Duration = Duration::from_millis(250);
/// Long enough for a TLS handshake, the subscribe frames and the first BBO on a bad
/// link; short enough that a dead socket fails rather than hangs a CI-less operator.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// The `userFills` snapshot replays this account's fill history the instant we
/// subscribe. It has to be consumed **before** the baselines are taken, or an earlier
/// session's fills land inside this one's measurement window.
const SNAPSHOT_SETTLE: Duration = Duration::from_secs(2);
/// A taker fill reports back in well under a second; 30 s means "it did not happen".
const FILL_TIMEOUT: Duration = Duration::from_secs(30);
/// Reduce-only closes the flatten will send before giving up and shouting.
const FLATTEN_ATTEMPTS: u32 = 3;
/// Time for a close to be reflected in `clearinghouseState` — and, symmetrically, for
/// an open to be. The endpoint lags both, which is the entire premise of [`FlattenPlan`].
const FLATTEN_SETTLE: Duration = Duration::from_secs(2);
/// Reads the flatten takes before its final, arbitrating one. Deliberately one more
/// than the closes: a flat read that arrives before the venue's bookkeeping has caught
/// up buys itself a second look, and that look must not be paid for out of the budget
/// that actually closes positions.
const FLATTEN_READS: u32 = FLATTEN_ATTEMPTS + 1;
/// Consecutive flat reads, [`FLATTEN_SETTLE`] apart, that stand in for having *seen* a
/// position when the entry may never have executed at all. One is not enough: the first
/// read can be issued a round trip ahead of the venue recording a fill we never heard
/// about, and it reads flat about an account that is long.
const FLAT_CONFIRMATIONS: u32 = 2;
/// A book older than this is not a price, it is a memory.
const MAX_BOOK_AGE: Duration = Duration::from_secs(60);

/// Deep enough that nobody has to drain the bus during the flatten, which does no bus
/// work at all: the bus blocks its producer when it fills, and a blocked producer is
/// the WS task that would otherwise still be delivering the fills we are waiting on.
const BUS_CAPACITY: usize = 16_384;

/// `cloid` tags, so a fill's client id says which leg produced it.
const TAG_ENTRY: u8 = 0x01;
const TAG_FLATTEN: u8 = 0x02;

/// A check that yields a message instead of a panic.
///
/// Everything between "an order was sent" and "the account is flat again" uses this
/// rather than `assert!`: a panic there unwinds straight past the flatten, and a test
/// that leaves a position open when it fails is a test nobody dares run twice.
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
/// An unrecognized value is refused rather than defaulted: `wallet_info` may read an
/// absent `AXON_HL_NETWORK` as testnet because it only ever *reads*, but this file
/// sends marketable orders, and "I assumed you meant testnet" is not a sentence worth
/// writing about live order flow.
fn require_testnet(network: Option<&str>) -> Result<(), String> {
    let rule = "!".repeat(78);
    match network {
        Some("testnet") => Ok(()),
        other => Err(format!(
            "{rule}\n\
             !! REFUSING TO RUN: AXON_HL_NETWORK is {other:?}, not \"testnet\".\n\
             !!\n\
             !! This test crosses the spread with a real, marketable order. On mainnet\n\
             !! that is real money at a real venue, and nothing about the code below\n\
             !! would stop it. Set AXON_HL_NETWORK=testnet in .env, confirm with\n\
             !! `./run.sh wallet`, and run it again.\n\
             {rule}"
        )),
    }
}

/// The size that puts [`TARGET_NOTIONAL`] on the wire at `px`, rounded to the asset's
/// lot size, with the resulting notional checked against both bounds.
///
/// Rounded **away from zero** so lot rounding can only ever push the notional up. Down
/// would be the cheap direction and the wrong one: a $9.99 order is rejected outright,
/// and a rejection proves nothing.
fn size_for_notional(px: Decimal, sz_decimals: u32) -> Result<Decimal, String> {
    ensure!(px > Decimal::ZERO, "price must be positive, got {px}");
    let qty =
        (TARGET_NOTIONAL / px).round_dp_with_strategy(sz_decimals, RoundingStrategy::AwayFromZero);
    ensure!(
        qty > Decimal::ZERO,
        "lot rounding produced a zero size at {px}"
    );
    let notional = qty * px;
    ensure!(
        notional >= MIN_NOTIONAL,
        "notional ${notional} (= {qty} @ {px}) is under the venue's ${MIN_NOTIONAL} \
         minimum; the venue would answer minTradeNtlRejected"
    );
    ensure!(
        notional <= MAX_NOTIONAL,
        "notional ${notional} (= {qty} @ {px}) exceeds this test's ${MAX_NOTIONAL} \
         ceiling - the lot size ({sz_decimals} decimals) rounded a ${TARGET_NOTIONAL} \
         test into something far larger. Pick a coin with a finer lot."
    );
    Ok(qty)
}

/// A client id unique per run and per leg.
///
/// Time-derived on purpose: a fixed `cloid` collides with the previous run's order at
/// the venue, and the venue's refusal of a duplicate client id reads exactly like a
/// signing failure from in here.
fn cloid_for(tag: u8, now_ms: u64) -> Cloid {
    Cloid::new(((now_ms as u128) << 8) | tag as u128)
}

/// One asset as the venue's `meta.universe` describes it.
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
            // keep their index in the universe, so resolving a name proves nothing
            // about whether it can be traded.
            delisted: asset
                .get("isDelisted")
                .and_then(|d| d.as_bool())
                .unwrap_or(false),
        });
    }
    Err(format!("{coin} is not in the venue's perp universe"))
}

// ── the core-side handler chain ──────────────────────────────────────────────

/// What the core sees during the run: the book the order is priced from, the tracker
/// under test, and a verbatim log of the execution events so a failure can be read
/// instead of guessed at.
struct LiveSession {
    book: MarketDataProcessor,
    tracker: OrderTracker,
    fills: Vec<Fill>,
    updates: Vec<OrderUpdate>,
}

impl LiveSession {
    fn new() -> Self {
        Self {
            book: MarketDataProcessor::new(),
            tracker: OrderTracker::new(),
            fills: Vec::new(),
            updates: Vec::new(),
        }
    }

    /// Fills for one venue order, deduped on `trade_id`.
    ///
    /// Deduped because the log records every frame, and `userFills` replays its
    /// snapshot on every reconnect — the tracker survives that by design, and an
    /// assertion counting raw frames would not.
    fn fills_for(&self, oid: OrderId) -> Vec<&Fill> {
        let mut seen = std::collections::HashSet::new();
        self.fills
            .iter()
            .filter(|f| f.order_id == oid)
            .filter(|f| seen.insert(f.trade_id))
            .collect()
    }
}

impl EventHandler for LiveSession {
    fn on_event(&mut self, ts_event: Nanos, event: &Event) {
        self.book.on_event(ts_event, event);
        self.tracker.on_event(ts_event, event);
        match event {
            Event::Exec(ExecEvent::Fill(f)) => self.fills.push(f.clone()),
            Event::Exec(ExecEvent::Order(u)) => self.updates.push(u.clone()),
            _ => {}
        }
    }
}

// ── async helpers (the edges) ────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// POST one `/info` body against **testnet**, returning the raw text.
///
/// The endpoint is a hard-coded constant rather than a parameter: a test that can be
/// pointed at mainnet by a variable is a test that will be.
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

/// Top of book from a REST `l2Book` snapshot, decoded through the same processor the
/// core uses.
///
/// REST rather than the live socket, and deliberately so: this is what the flatten
/// prices against, and the flatten has to work in exactly the case where the socket is
/// what died.
async fn rest_top_of_book(
    symbols: &SymbolMap,
    sym: SymbolId,
) -> Result<(Decimal, Decimal), String> {
    let snapshot = fetch_l2_snapshot(TESTNET_INFO, COIN, symbols)
        .await
        .map_err(|e| format!("l2Book snapshot: {e}"))?;
    let mut book = MarketDataProcessor::new();
    let event = Event::from(snapshot);
    book.on_event(event.ts_event(), &event);
    let b = book
        .book(sym)
        .ok_or_else(|| "l2Book snapshot carried no book".to_string())?;
    match (b.best_bid(), b.best_ask()) {
        (Some((bid, _)), Some((ask, _))) => Ok((bid, ask)),
        _ => Err("one-sided book: nothing to cross into".to_string()),
    }
}

/// The venue's own signed position for `sym`, from `clearinghouseState`.
async fn venue_position(
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
) -> Result<Position, String> {
    let state = fetch_user_state(TESTNET_INFO, account, symbols)
        .await
        .map_err(|e| format!("clearinghouseState: {e}"))?;
    Ok(state
        .positions
        .iter()
        .find(|p| p.symbol_id == sym)
        .cloned()
        .unwrap_or_else(|| Position::flat(sym)))
}

/// Drain the bus, then answer `f`, until it says yes or `budget` runs out.
///
/// Draining is what keeps the bounded bus from backing up into the WS task while we
/// wait — a blocked producer would stall the very frames we are waiting for.
async fn wait_until(
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

// ── the flatten ──────────────────────────────────────────────────────────────

/// What the traded section put on the wire, which is what decides how much a flat
/// `clearinghouseState` read is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    /// Nothing was signed: the run gave up before `place_order` — a stale BBO, a size
    /// outside its bounds. There is no fill for the venue's bookkeeping to be behind
    /// on, so the first flat read is simply true.
    NeverSent,
    /// An order reached the transport. Whether it executed is not knowable from in
    /// here: a failure on the place is ambiguous by construction, and it comes back
    /// long before `clearinghouseState` would show the position it may have opened.
    MaybeExecuted,
}

/// How the flatten ended, in the only terms an operator can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flatness {
    /// No order ever reached the venue, so there was never anything to close.
    NothingSent,
    /// The venue was seen holding a position and then seen flat. This is the only
    /// outcome that has verified anything.
    Verified,
    /// An order went out and every read came back flat without the venue once showing
    /// a position. Most likely it never executed — but that is a different sentence
    /// from "we watched it close", and the report must not print the second when it
    /// means the first.
    NeverSeenOpen,
}

impl Flatness {
    /// The clause the operator reads. It has to survive being skimmed.
    fn report(self) -> &'static str {
        match self {
            Flatness::NothingSent => "no order ever reached the venue",
            Flatness::Verified => "the account was flattened, confirmed by clearinghouseState",
            Flatness::NeverSeenOpen => {
                "the venue reads flat, but a position was never seen open - that is NOT a \
                 confirmation that something which may have filled was closed; check \
                 `./run.sh wallet` by hand"
            }
        }
    }
}

/// What the flatten should do about one `clearinghouseState` read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Flat, and the read is worth believing. Stop.
    Stop(Flatness),
    /// Flat, but too early to believe. Wait and read again; send nothing.
    Recheck,
    /// The venue holds this much, signed. Send a reduce-only close and read again.
    Close(Decimal),
    /// Still holding, with the close budget spent. The final read decides.
    Spent(Decimal),
    /// The read itself failed. "We could not check" is not "there is nothing there",
    /// so it never counts towards a confirmation and never ends the flatten.
    Unreadable,
}

/// The rule the flatten applies to the reads it gets, kept pure and out of the driver
/// because it is the one part that cannot be tested against a live venue.
///
/// It exists for a single failure. `place_order` times out *after* the venue accepted
/// and filled the marketable IOC; the traded section returns an error within
/// milliseconds; the flatten's first `clearinghouseState` read is therefore issued a
/// round trip ahead of the venue recording the position, comes back flat, and the run
/// tells the operator the account was flattened while a live position sits at the
/// venue. A single flat read only ever means "nothing is there *yet*".
#[derive(Debug)]
struct FlattenPlan {
    entry: Entry,
    /// Set the first time the venue is *seen* holding something. Once it has been, the
    /// venue's bookkeeping has demonstrably caught up with whatever we sent, and a
    /// later flat read is a close we watched happen rather than a state we outran.
    seen_open: bool,
    /// Flat reads since the last read that was not flat. Reset by a position and by an
    /// unreadable answer alike.
    consecutive_flat: u32,
    /// Closes sent, so an exhausted budget stops rather than streams orders at a venue.
    closes: u32,
}

impl FlattenPlan {
    fn new(entry: Entry) -> Self {
        Self {
            entry,
            seen_open: false,
            consecutive_flat: 0,
            closes: 0,
        }
    }

    /// Fold in one read — `None` when the read itself failed — and say what to do next.
    fn observe(&mut self, qty: Option<Decimal>) -> Step {
        let Some(qty) = qty else {
            self.consecutive_flat = 0;
            return Step::Unreadable;
        };
        if !qty.is_zero() {
            self.seen_open = true;
            self.consecutive_flat = 0;
            if self.closes >= FLATTEN_ATTEMPTS {
                return Step::Spent(qty);
            }
            self.closes += 1;
            return Step::Close(qty);
        }
        self.consecutive_flat += 1;
        // Order matters: having actually seen the position outranks everything else,
        // including a run that believed it never sent anything.
        if self.seen_open {
            Step::Stop(Flatness::Verified)
        } else {
            match self.entry {
                Entry::NeverSent => Step::Stop(Flatness::NothingSent),
                Entry::MaybeExecuted if self.consecutive_flat >= FLAT_CONFIRMATIONS => {
                    Step::Stop(Flatness::NeverSeenOpen)
                }
                Entry::MaybeExecuted => Step::Recheck,
            }
        }
    }
}

/// One `clearinghouseState` read reduced to the signed quantity, `None` when the read
/// itself failed — a distinction that must never collapse into zero.
async fn read_position(
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    round: u32,
) -> Option<Decimal> {
    match venue_position(account, symbols, sym).await {
        Ok(p) => Some(p.qty),
        Err(why) => {
            eprintln!("flatten: round {round} could not read clearinghouseState: {why}");
            None
        }
    }
}

/// Leave the account flat in `sym`, and prove it by reading `clearinghouseState` back.
///
/// Order of operations is load-bearing. Resting orders are cancelled **first**: a
/// resting buy that fills while we are closing puts the position straight back, and a
/// flatten that races itself never terminates. Then the position is closed with a
/// `reduce_only` IOC — reduce-only so it can only ever shrink exposure, IOC so nothing
/// it fails to fill is left resting behind us.
///
/// It re-reads the venue between rounds rather than trusting its own arithmetic,
/// because a partial close is exactly the case where our number and the venue's differ.
/// What it will not do is believe the *first* flat read when an order went out and no
/// position has been seen yet — that read races the venue by a round trip, and
/// [`FlattenPlan`] is what refuses it. One edge is documented rather than solved: a
/// close whose remainder falls under the venue's minimum notional can be refused, and
/// the error below says so.
async fn flatten(
    client: &ExchangeClient,
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    grid: &InstrumentSpec,
    entry: Entry,
) -> Result<Flatness, String> {
    match fetch_frontend_open_orders(TESTNET_INFO, account, None, symbols).await {
        Ok(open) => {
            for order in open.items.iter().filter(|o| o.symbol_id == sym) {
                // A cancel that fails because the order just filled is not a problem —
                // the position read below is what decides. Only silence would be.
                match client.cancel(order.cancel_id()).await {
                    Ok(_) => eprintln!("flatten: cancelled resting oid={}", order.order_id.get()),
                    Err(e) => eprintln!(
                        "flatten: cancel of oid={} failed: {e}",
                        order.order_id.get()
                    ),
                }
            }
        }
        Err(e) => eprintln!("flatten: could not read open orders ({e}); closing anyway"),
    }

    // Every read and every close is allowed to fail on its own. A `?` here would turn
    // one dropped REST call into an abandoned position, which is the exact outcome this
    // function exists to prevent — so failures are logged and the next round re-reads.
    let mut plan = FlattenPlan::new(entry);
    for round in 1..=FLATTEN_READS {
        match plan.observe(read_position(account, symbols, sym, round).await) {
            Step::Stop(how) => {
                eprintln!("flatten: round {round} - {}", how.report());
                return Ok(how);
            }
            Step::Recheck => eprintln!(
                "flatten: round {round} reads flat, but nothing has been seen open and \
                 clearinghouseState trails a fill by a round trip - re-reading rather than \
                 calling the account clean"
            ),
            Step::Close(qty) => {
                if let Err(why) = close_once(client, symbols, sym, grid, qty, round).await {
                    eprintln!("flatten: round {round} could not close: {why}");
                }
            }
            Step::Spent(qty) => eprintln!(
                "flatten: round {round} still holds {qty} with the {FLATTEN_ATTEMPTS}-close \
                 budget spent; the final read decides"
            ),
            Step::Unreadable => {}
        }
        tokio::time::sleep(FLATTEN_SETTLE).await;
    }

    match plan.observe(read_position(account, symbols, sym, FLATTEN_READS + 1).await) {
        Step::Stop(how) => Ok(how),
        // An unreadable final state is reported as unknown, never as flat: "we could not
        // check" and "there is nothing there" are not the same sentence. The reason is
        // on stderr, from the read itself — this test is run with `--nocapture` or not
        // at all.
        Step::Unreadable => Err(not_flat(
            "unknown - the final clearinghouseState read failed; its reason is logged above",
        )),
        // Flat, and still not worth believing. Announcing a flatten here is precisely the
        // lie the plan exists to refuse: an order went out, and a fill the venue has not
        // published yet looks exactly like an order that never filled.
        Step::Recheck => Err(not_flat(
            "unknown - the venue reads flat, but an order went out and a position was \
             never once seen",
        )),
        Step::Close(qty) | Step::Spent(qty) => Err(not_flat(&qty.to_string())),
    }
}

/// Send one reduce-only IOC against `qty`, signed exactly as the venue reports it.
///
/// It does not read the position itself: deciding *whether* to close is [`FlattenPlan`]'s
/// job, and a close that re-read the venue would be re-deriving the very judgement the
/// plan was built to make.
async fn close_once(
    client: &ExchangeClient,
    symbols: &SymbolMap,
    sym: SymbolId,
    grid: &InstrumentSpec,
    qty: Decimal,
    round: u32,
) -> Result<(), String> {
    let (bid, ask) = rest_top_of_book(symbols, sym).await?;
    // Close in the direction that reduces: a long is sold into the bid, a short bought
    // back from the ask. Both are quantized **toward** the market, through the same
    // production grid the planner uses — rounding a close away from the market is how a
    // flatten turns into a resting quote and leaves the position it was sent to remove.
    let (side, px) = if qty.is_sign_positive() {
        (Side::Sell, bid * (Decimal::ONE - CROSS))
    } else {
        (Side::Buy, ask * (Decimal::ONE + CROSS))
    };
    let px = grid.price.quantize(px, side, PriceIntent::Marketable);
    // The venue's own size, unrounded: it is the sum of our fills and therefore already
    // a whole number of lots. Re-rounding it could only leave dust behind or overshoot
    // the position into a `reduceOnlyRejected`.
    let qty = qty.abs();
    let mut req = OrderRequest::limit(
        sym,
        side,
        qty,
        px,
        Tif::Ioc,
        cloid_for(TAG_FLATTEN, now_ms()),
    );
    req.reduce_only = true;
    eprintln!("flatten: round {round} closing {qty} @ {px} ({side:?}, reduce-only)");
    match client.place_order(req).await {
        Ok(ack) => eprintln!("flatten: ack {:?} oid={:?}", ack.status, ack.order_id),
        // Not fatal: the next round re-reads the position and tries again. A rejection
        // here is usually a stale price or a remainder under the minimum notional.
        Err(e) => eprintln!("flatten: round {round} rejected: {e}"),
    }
    Ok(())
}

/// The message an operator must not be able to skim past.
///
/// "up to" and not "after {FLATTEN_ATTEMPTS}": the flatten also ends here when it never
/// saw a position to close, and a message that overstates what it tried is a message
/// that will be argued with instead of acted on.
fn not_flat(remaining: &str) -> String {
    let rule = "!".repeat(78);
    format!(
        "{rule}\n\
         !! THE ACCOUNT IS NOT FLAT. {COIN} position {remaining}, after up to \
         {FLATTEN_ATTEMPTS} reduce-only closes.\n\
         !!\n\
         !! This is exposure, not a test failure. Deal with it before anything else:\n\
         !!     ./run.sh wallet          # the position, and any order still resting\n\
         !! then close it from the testnet UI. The usual reason a close does not\n\
         !! complete is a remainder worth less than the venue's $10 minimum notional,\n\
         !! which it refuses outright.\n\
         !!\n\
         !! Do not re-run this test until the account is flat again: it refuses to start\n\
         !! on an existing position, and for good reason.\n\
         {rule}"
    )
}

// ── the traded section ───────────────────────────────────────────────────────

/// Everything between "send the order" and "it is reconciled", returning the quantity
/// actually executed. Returns `Err` instead of asserting so that the flatten always
/// runs; see the [`ensure!`] macro.
///
/// `entry` is an out-parameter rather than part of the return value because the flatten
/// needs it on the error paths, which is where it decides anything: an `Err` alone
/// cannot say whether an order reached the venue, and a flatten that cannot tell those
/// apart has to guess about a live position.
async fn trade_and_reconcile(
    client: &ExchangeClient,
    rx: &EventReceiver,
    session: &mut LiveSession,
    sym: SymbolId,
    spec: &AssetSpec,
    grid: &InstrumentSpec,
    entry: &mut Entry,
) -> Result<Decimal, String> {
    // Price off the live BBO the socket has been delivering — the same socket the user
    // channels are on, so a book we are receiving is a socket whose subscriptions the
    // venue has already processed.
    let bbo = session
        .book
        .bbo(sym)
        .cloned()
        .ok_or_else(|| "no live BBO to price against".to_string())?;
    let age_ns = SystemClock.now_ns() - bbo.ts_event;
    ensure!(
        age_ns < MAX_BOOK_AGE.as_nanos() as i64,
        "the BBO is {}s old - pricing a marketable order off a stale book is how a \
         50bp cross becomes a 5% one",
        age_ns / 1_000_000_000
    );

    // Through the **production** grid, not a helper this file owns. The whole point of
    // the test is that the path it exercises is the path a live session takes: a test
    // that rounded its own price would pass over a broken production encoder and
    // manufacture confidence in it.
    let px = grid.price.quantize(
        bbo.ask_px * (Decimal::ONE + CROSS),
        Side::Buy,
        PriceIntent::Marketable,
    );
    ensure!(
        px > bbo.ask_px,
        "priced {px} at or under the best ask {} - that rests, it does not fill",
        bbo.ask_px
    );
    let qty = size_for_notional(px, spec.sz_decimals)?;
    let notional = px * qty;
    // The venue's rule, asserted before we sign rather than learned from a rejection.
    ensure!(
        notional >= MIN_NOTIONAL,
        "notional ${notional} is under the venue's ${MIN_NOTIONAL} minimum"
    );
    ensure!(
        notional <= MAX_NOTIONAL,
        "notional ${notional} exceeds this test's ${MAX_NOTIONAL} ceiling"
    );
    eprintln!(
        "book: bid {} x {} / ask {} x {} | crossing with {qty} @ {px} (${notional})",
        bbo.bid_px, bbo.bid_sz, bbo.ask_px, bbo.ask_sz
    );
    if bbo.ask_sz < qty {
        // Not fatal — the cross reaches deeper levels — but a partial fill has a
        // different shape, and knowing which one happened matters when reading a
        // failure below.
        eprintln!(
            "note: top of book ({}) is thinner than the order; expect the fill to sweep levels",
            bbo.ask_sz
        );
    }

    // Baselines, taken after the `userFills` snapshot has been consumed. Everything
    // below is asserted as a *delta*: this account's history legitimately contains
    // fills from earlier sessions that no tracker in this process ever placed, so an
    // absolute `orphan_fills() == 0` would be asserting something untrue.
    let orphans_before = session.tracker.orphan_fills();
    let pos_before = session.tracker.position(sym).qty;
    if pos_before != Decimal::ZERO {
        // The venue said flat before this run started, so a non-zero number here means
        // the replayed history is not the whole history — `userFills` is capped, and a
        // truncated replay reconstructs a partial position. Worth saying out loud: it
        // is the one thing that would make the deltas below the only honest measure.
        eprintln!("note: replayed history opens at {pos_before}, while the venue reports flat");
    }

    // IOC, not GTC: whatever fails to fill is cancelled by the venue rather than left
    // resting behind the test.
    let cloid = cloid_for(TAG_ENTRY, now_ms());
    let req = OrderRequest::limit(sym, Side::Buy, qty, px, Tif::Ioc, cloid);
    // A transport failure here is *ambiguous*: the order may well have reached the
    // venue and filled, and we simply never heard. That is precisely why the flatten
    // reads `clearinghouseState` instead of trusting anything this function knows —
    // returning early from here is safe only because the flatten still runs.
    //
    // Flipped **before** the await, not after: on that exact path the reply never comes
    // back, so anything set afterwards is never set at all, and the flatten would go on
    // believing nothing was sent — and then believe its first flat read.
    *entry = Entry::MaybeExecuted;
    let ack = client
        .place_order(req.clone())
        .await
        .map_err(|e| format!("place: {e}"))?;
    // An ack carries no venue timestamp, so its event time is when we learned of it —
    // the same honest answer `Ticker` gives for `activeAssetCtx`. Safe here because
    // the tracker's filled quantities are monotonic and its terminal transitions are
    // applied whatever their timestamp, so an ack stamped slightly ahead of the venue's
    // own clock cannot lose a fill or resurrect a dead order.
    session.tracker.on_ack(&req, &ack, SystemClock.now_ns());
    let oid = ack
        .order_id
        .ok_or_else(|| format!("no oid on the ack: {ack:?}"))?;
    eprintln!(
        "placed: oid={} status={:?} cloid={cloid:?}",
        oid.get(),
        ack.status
    );

    // Wait for the venue's own account channels, not for the ack. The ack says the
    // submit was accepted; only `userFills` says an execution happened, and only
    // `orderUpdates` says the order is done. **Both** are required before anything is
    // asserted, because the two channels are independent: asserting on the lifecycle
    // the moment a fill lands would fail whenever `orderUpdates` happened to be second.
    let observed = wait_until(rx, session, FILL_TIMEOUT, |s| {
        let traded = s
            .tracker
            .order(cloid)
            .is_some_and(|o| o.filled_qty > Decimal::ZERO && o.status.is_terminal());
        traded && s.updates.iter().any(|u| u.order_id == oid)
    })
    .await;
    let tracked = session
        .tracker
        .order(cloid)
        .cloned()
        .ok_or_else(|| "the tracker lost the order it acked".to_string())?;
    ensure!(
        observed,
        "the venue's account channels went quiet for {FILL_TIMEOUT:?}. Tracker: status \
         {:?}, filled {} of {}; {} orderUpdates frame(s) for oid={}. Either the order \
         did not execute (check the venue), or a channel is not delivering — a \
         subscription made with the agent address instead of the master address returns \
         nothing forever, with no error to say so.",
        tracked.status,
        tracked.filled_qty,
        tracked.orig_qty,
        session.updates.iter().filter(|u| u.order_id == oid).count(),
        oid.get()
    );

    // ── what the tracker made of it ──
    let fills = session.fills_for(oid);
    ensure!(
        !fills.is_empty(),
        "the tracker filled but logged no fill for oid={}",
        oid.get()
    );
    let mut executed = Decimal::ZERO;
    for f in &fills {
        eprintln!(
            "fill: tid={} {} @ {} side={:?} liquidity={:?} fee={} cloid={:?}",
            f.trade_id, f.qty, f.price, f.side, f.liquidity, f.fee, f.cloid
        );
        ensure!(
            f.symbol_id == sym,
            "a fill for another instrument was attributed here"
        );
        ensure!(
            f.side == Side::Buy,
            "fill side {:?} is not the side we sent",
            f.side
        );
        ensure!(
            f.liquidity == Liquidity::Taker,
            "fill reported as {:?}: a marketable order that rested and was filled later \
             proves resting, which is already proven",
            f.liquidity
        );
        // A cloid the venue echoed must be *ours*; one it omitted is not a failure. The
        // tracker keeps both maps precisely because this field is optional on the wire,
        // and it falls back to the oid — but the fallback is the weaker path (it cannot
        // match a fill that overtakes its own ack), so an absent id is worth saying.
        match f.cloid {
            Some(c) => ensure!(
                c == cloid,
                "a fill carrying someone else's cloid {c:?} was attributed to our order"
            ),
            None => eprintln!(
                "note: no cloid echoed on tid={}; attribution fell back to the oid",
                f.trade_id
            ),
        }
        executed += f.qty;
    }

    let orphans_after = session.tracker.orphan_fills();
    ensure!(
        orphans_after == orphans_before,
        "{} fill(s) were orphaned during the window: the tracker moved a position it \
         could not attribute to any order it knows",
        orphans_after - orphans_before
    );
    ensure!(
        tracked.filled_qty == executed,
        "the tracker attributed {} to the order but {executed} was executed",
        tracked.filled_qty
    );
    let pos_delta = session.tracker.position(sym).qty - pos_before;
    ensure!(
        pos_delta == executed,
        "the tracker's position moved by {pos_delta}, but {executed} was executed"
    );
    // Nothing may be left working: an IOC's unfilled remainder is cancelled by the
    // venue, so risk-position and filled position must have converged again.
    ensure!(
        session.tracker.resting_exposure(sym) == Decimal::ZERO,
        "{} is still resting after an IOC",
        session.tracker.resting_exposure(sym)
    );
    ensure!(
        session.tracker.risk_position(sym).qty == session.tracker.position(sym).qty,
        "risk position and filled position disagree with nothing resting"
    );
    for u in session.updates.iter().filter(|u| u.order_id == oid) {
        eprintln!(
            "update: status={:?} filled {} of {} reason={:?}",
            u.status,
            u.filled_qty(),
            u.orig_qty,
            u.cancel_reason
        );
    }
    eprintln!(
        "tracker: order {:?} filled {} of {}, position {} (was {pos_before}), orphans {orphans_after}",
        tracked.status,
        tracked.filled_qty,
        tracked.orig_qty,
        session.tracker.position(sym).qty
    );
    Ok(executed)
}

/// The claim the whole file exists to check: our fill-derived position is the venue's.
///
/// Compared against what this run *executed*, not against the tracker's absolute
/// position: the account was verified flat in this coin before the order went out, so
/// the venue's number must now be exactly the quantity we saw fill. The tracker's own
/// total may legitimately carry a replayed history on top of that.
async fn assert_agrees_with_the_venue(
    account: &str,
    symbols: &SymbolMap,
    sym: SymbolId,
    executed: Decimal,
) -> Result<(), String> {
    let venue = venue_position(account, symbols, sym).await?;
    ensure!(
        venue.qty == executed,
        "reconciliation failed: we observed {executed} fill, the venue's \
         clearinghouseState reports a position of {}",
        venue.qty
    );
    eprintln!(
        "reconciled: venue szi {} == the {executed} we saw fill",
        venue.qty
    );
    Ok(())
}

// ── the live test ────────────────────────────────────────────────────────────

/// Place a marketable order on testnet, observe the fill on the account channels,
/// reconcile it, then flatten. See the module docs before running it.
#[tokio::test]
#[ignore = "places a REAL marketable order on Hyperliquid testnet; needs a funded .env"]
async fn fill_is_observed_reconciled_and_flattened_on_testnet() {
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
    let signing = signer.address().to_string();
    eprintln!("network: testnet | account {account} | signer {signing}");

    // 2. Resolve the asset from the venue's own universe. One `meta` read serves the
    //    SymbolMap, the instrument grids and the index/delisted facts — three reads
    //    could disagree.
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
    // The same table the encoder validates against, so a price this test rounds and a
    // price the wire accepts cannot be two different opinions.
    let client = ExchangeClient::testnet(signer)
        .expect("testnet ExchangeClient")
        .with_instruments(Arc::new(universe.instruments));
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
    assert_eq!(
        grid.size.increment(),
        Decimal::new(1, spec.sz_decimals),
        "the decoded grid and the raw universe must agree about {COIN}'s lot"
    );
    eprintln!(
        "asset: {COIN} index {} szDecimals {} lot {} tick@bid {}",
        spec.index,
        spec.sz_decimals,
        grid.size.increment(),
        grid.price.tick_at(Decimal::from(60_000))
    );

    // 3. Preconditions on the account itself. Both exist because the flatten at the end
    //    closes *everything* in this coin: starting on top of someone else's position or
    //    resting order would close their exposure and measure their fills as ours.
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
        state.account.equity >= MIN_NOTIONAL * Decimal::TWO,
        "equity {} is too small to margin a ${TARGET_NOTIONAL} order - fund the account \
         (fund it at the venue's testnet faucet)",
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

    // 4. Subscribe to the account channels BEFORE anything is placed. This ordering is
    //    the point of the test: `orderUpdates` never snapshots, so a fill that happens
    //    in the gap between placing and subscribing is simply gone, and the run would
    //    fail for a reason that has nothing to do with the code under test.
    let (tx, rx) = bus(BUS_CAPACITY);
    let md = HyperliquidMarketData::testnet(symbols.clone(), vec![COIN.into()], tx);
    md.subscribe_user_channels(account.as_str());
    // The BBO rides the same socket, which is what makes it evidence: `run_once` writes
    // every subscribe frame before it reads the first reply, and the venue processes one
    // connection's frames in order — so a BBO in hand means the user subscriptions
    // landed at the venue no later than the one delivering it.
    md.subscribe_coin(Feed::Bbo, COIN);
    let ws = tokio::spawn(async move { md.run_forever().await });

    let mut session = LiveSession::new();
    let connected = wait_until(&rx, &mut session, CONNECT_TIMEOUT, |s| {
        s.book.bbo(sym).is_some()
    })
    .await;
    assert!(
        connected,
        "no BBO within {CONNECT_TIMEOUT:?}: the socket is not delivering, so nothing \
         below could observe a fill either"
    );
    // Let the `userFills` snapshot land and be applied before any baseline is taken.
    tokio::time::sleep(SNAPSHOT_SETTLE).await;
    drain_available(&rx, &mut session);
    eprintln!(
        "subscribed: userFills + orderUpdates for {account}; snapshot replayed {} fill(s)",
        session.fills.len()
    );

    // 5. Trade. From here to the flatten, failures are values, never panics.
    let mut entry = Entry::NeverSent;
    let traded =
        trade_and_reconcile(&client, &rx, &mut session, sym, &spec, &grid, &mut entry).await;
    let reconciled = match &traded {
        Ok(executed) => assert_agrees_with_the_venue(&account, &symbols, sym, *executed).await,
        // Skipped, not silently passed: comparing against the venue means nothing when
        // we do not know what we traded.
        Err(_) => Ok(()),
    };

    // 6. Flatten, whatever happened above. It is told whether an order reached the wire,
    //    because that is what decides whether a flat read means anything.
    let flattened = flatten(&client, &account, &symbols, sym, &grid, entry).await;
    ws.abort();

    // 7. Report, worst first. An open position outranks a failed assertion: one is a
    //    broken test, the other is money sitting at a venue. And what the flatten
    //    actually established is quoted rather than assumed — "we watched it close" and
    //    "we never saw it open" are both `Ok`, and only one of them is reassuring.
    let left = match flattened {
        Ok(how) => how,
        Err(why) => panic!("{why}\n\nthe traded section reported: {traded:?}"),
    };
    if let Err(why) = traded {
        panic!("live fill verification failed ({}): {why}", left.report());
    }
    if let Err(why) = reconciled {
        panic!("{why}");
    }
    eprintln!(
        "PASS: fill observed, reconciled against the venue - {}.",
        left.report()
    );
}

// ── offline tests: the rules this file bets on, checked without a network ────

/// A slice of the live testnet `meta` (captured 2026-07-25), kept small but honest:
/// SOL at index 0, a delisted asset in the middle, and BTC at index 3 — which is what
/// makes it worth pinning, because BTC is index 0 on mainnet.
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
    // Positional and venue-specific: BTC is 3 on testnet and 0 on mainnet. An index
    // taken from anywhere but the response in hand does not fail — it trades a
    // different coin, at a size computed for this one.
    let btc = asset_spec(META_SAMPLE, "BTC").unwrap();
    assert_eq!(btc.index, 3);
    assert_eq!(btc.sz_decimals, 5);
    assert!(!btc.delisted);
    assert_eq!(asset_spec(META_SAMPLE, "SOL").unwrap().index, 0);
    assert_eq!(asset_spec(META_SAMPLE, "ETH").unwrap().sz_decimals, 4);
    assert!(asset_spec(META_SAMPLE, "NOTACOIN").is_err());
    assert!(asset_spec("not json", "BTC").is_err());
    assert!(asset_spec(r#"{"nope":[]}"#, "BTC").is_err());
}

#[test]
fn a_delisted_asset_still_resolves_and_must_not_be_traded() {
    // 53 of testnet's 210 perps carry `isDelisted`. They keep their index, so name
    // resolution succeeds and every order they receive is rejected.
    let dead = asset_spec(META_SAMPLE, "MATIC").unwrap();
    assert!(dead.delisted);
    assert_eq!(dead.index, 1, "it still occupies its slot in the universe");
    assert!(!asset_spec(META_SAMPLE, "ATOM").unwrap().delisted);
}

/// The grid the decoder builds for an asset with `sz_decimals`, reached through the
/// production decoder rather than reconstructed here.
///
/// This file used to carry its own `round_price`, and that was the dangerous version of
/// the bug: the live test rounded its own price, passed, and proved nothing whatever
/// about the production encoder it was meant to validate — a green test over a broken
/// path, which is worse than a red one.
fn grid_for(sz_decimals: u32) -> InstrumentSpec {
    let body = format!(r#"{{"universe":[{{"name":"X","szDecimals":{sz_decimals}}}]}}"#);
    *decode_universe(&body)
        .expect("the decoder builds a grid from szDecimals")
        .instruments
        .get(SymbolId::new(0))
        .expect("index 0")
}

#[test]
fn a_price_that_breaks_the_venues_tick_rules_is_rounded_before_it_is_sent() {
    // 5 significant figures, and at most `6 - szDecimals` decimals — whichever binds
    // first. Either one broken comes back `tickRejected`, which from inside the process
    // is indistinguishable from a signing bug.
    //
    // The direction is now the *order's*, not "nearest": a marketable buy ceils so it
    // still crosses, a passive buy floors so it cannot be rejected as crossing. The old
    // helper rounded to nearest, which is the direction that silently turns a taker into
    // a resting quote.
    let up = |px, sz| {
        grid_for(sz)
            .price
            .quantize(px, Side::Buy, PriceIntent::Marketable)
    };
    let down = |px, sz| {
        grid_for(sz)
            .price
            .quantize(px, Side::Buy, PriceIntent::Passive)
    };

    assert_eq!(up(dec!(64714.965), 5), dec!(64715), "5 digits ⇒ integer");
    assert_eq!(down(dec!(64714.965), 5), dec!(64714));
    assert_eq!(up(dec!(3456.789), 4), dec!(3456.8), "4 digits ⇒ 1 decimal");
    assert_eq!(up(dec!(1.234567), 2), dec!(1.2346), "1 digit ⇒ 4 decimals");
    // The lot rule binds first here: 6 - 5 = 1 decimal, even though 5 significant
    // figures would have allowed 4.
    assert_eq!(down(dec!(1.234567), 5), dec!(1.2));
    assert_eq!(up(dec!(1.234567), 5), dec!(1.3));

    // Sub-$1 prices, and the one value this change moves. The old helper computed
    // "integer digits" and got 0 for every price under a dollar, so it capped at five
    // decimal places where the venue permits six — 0.00343 instead of 0.003429. It was
    // throwing a digit away on every cheap asset, 3 bp, which on a passive quote is the
    // whole edge. Both are legal prices; the new ones are strictly finer, so this is a
    // tightening.
    assert_eq!(up(dec!(0.0034285714), 0), dec!(0.003429));
    assert_eq!(down(dec!(0.0034285714), 0), dec!(0.003428));
    assert!(grid_for(0).price.is_valid(dec!(0.003429)));
    assert!(!grid_for(0).price.is_valid(dec!(0.0034285714)));

    // Above 5 digits the price lands on an integer, and integers are exempt from the
    // significant-figure rule entirely.
    assert_eq!(up(dec!(118234.5), 5), dec!(118235));
    assert!(grid_for(5).price.is_valid(dec!(118235)));
    // Leading zeros are not significant figures; trailing zeros are not digits.
    assert_eq!(grid_for(0).price.tick_at(dec!(0.5)), dec!(0.00001));
    assert_eq!(grid_for(5).price.tick_at(dec!(64393.0)), dec!(1));
}

#[test]
fn lot_rounding_can_never_put_the_order_under_the_ten_dollar_minimum() {
    // Testnet BTC on 2026-07-25: ~64,393 with a 5-decimal lot.
    let qty = size_for_notional(dec!(64715), 5).unwrap();
    assert_eq!(qty, dec!(0.00019), "rounded up, never down");
    assert!(qty * dec!(64715) >= MIN_NOTIONAL);
    assert!(qty * dec!(64715) <= MAX_NOTIONAL);
    // ETH-like: a coarser lot, a smaller price, still inside both bounds.
    let qty = size_for_notional(dec!(3456.8), 4).unwrap();
    assert!(qty * dec!(3456.8) >= MIN_NOTIONAL);
    assert!(qty * dec!(3456.8) <= MAX_NOTIONAL);
}

#[test]
fn lot_rounding_cannot_turn_a_twelve_dollar_test_into_a_thousand_dollar_one() {
    // `szDecimals: 0` means the lot is one whole unit. A $12 target on a $1,000 asset
    // rounds up to one unit — an order eighty times the size intended, which is not a
    // thing to discover from a filled order.
    let err = size_for_notional(dec!(1000), 0).unwrap_err();
    assert!(err.contains("ceiling"), "{err}");
    assert!(
        err.contains("finer lot"),
        "and it must say what to do: {err}"
    );
    // A price of zero is a decode failure upstream, not a free order.
    assert!(size_for_notional(Decimal::ZERO, 5).is_err());
    assert!(size_for_notional(dec!(-1), 5).is_err());
}

#[test]
fn a_client_id_reused_across_runs_would_be_rejected_as_a_duplicate() {
    // The venue keys orders on the client id. A constant would collide with the
    // previous run's order and be refused in a way that reads exactly like a bad
    // signature, so the id carries the clock and the leg.
    let t = 1_785_003_324_667u64;
    assert_ne!(cloid_for(TAG_ENTRY, t), cloid_for(TAG_FLATTEN, t));
    assert_ne!(cloid_for(TAG_ENTRY, t), cloid_for(TAG_ENTRY, t + 1));
    // The tag occupies the low byte and the clock everything above it, so neither can
    // silently overwrite the other.
    assert_eq!(cloid_for(TAG_ENTRY, t).get() >> 8, t as u128);
    assert_eq!(cloid_for(TAG_FLATTEN, t).get() & 0xff, TAG_FLATTEN as u128);
}

#[test]
fn a_flat_read_taken_a_round_trip_after_an_ambiguous_place_is_not_believed() {
    // The failure the plan exists for. `place_order` times out *after* the venue
    // accepted and filled the IOC; the traded section errors within milliseconds; the
    // flatten's first `clearinghouseState` read is issued before the venue has recorded
    // the position. Believing it abandons a live position and prints "the account was
    // flattened" — the one sentence an operator is entitled to trust.
    let mut plan = FlattenPlan::new(Entry::MaybeExecuted);
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Recheck,
        "the first flat read after an order went out is a race, not evidence"
    );
    // FLATTEN_SETTLE later, the venue has caught up and the position is there.
    assert_eq!(
        plan.observe(Some(dec!(0.00019))),
        Step::Close(dec!(0.00019))
    );
    // Having seen it open, a flat read now means the close landed: one read is enough,
    // because the endpoint has demonstrably caught up with what we sent.
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Stop(Flatness::Verified)
    );
}

#[test]
fn a_flatten_that_never_saw_a_position_is_not_reported_as_a_flattened_account() {
    // The honest outcome for a run that sent an order and then read flat every time: it
    // probably never executed, but nothing here watched a position close, and those two
    // must not print the same words.
    let mut plan = FlattenPlan::new(Entry::MaybeExecuted);
    assert_eq!(plan.observe(Some(Decimal::ZERO)), Step::Recheck);
    let Step::Stop(how) = plan.observe(Some(Decimal::ZERO)) else {
        panic!("{FLAT_CONFIRMATIONS} flat reads must end the flatten rather than loop it");
    };
    assert_eq!(how, Flatness::NeverSeenOpen);
    assert_ne!(
        how.report(),
        Flatness::Verified.report(),
        "an unconfirmed flatten that reads like a confirmed one is the whole bug"
    );
    assert!(
        how.report().contains("NOT a confirmation"),
        "the operator has to be told which of the two happened: {}",
        how.report()
    );
}

#[test]
fn a_run_that_signed_nothing_does_not_wait_on_a_position_that_cannot_exist() {
    // The other side of the same coin. A run that gave up on a stale BBO never reached
    // `place_order`, so there is no venue bookkeeping to be behind on, and spending a
    // second round trip to confirm nothing would just be superstition.
    let mut plan = FlattenPlan::new(Entry::NeverSent);
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Stop(Flatness::NothingSent)
    );
    // A position that is nonetheless there is still closed — and closing it is what
    // makes the later flat read worth something, whatever this run thought it sent.
    let mut plan = FlattenPlan::new(Entry::NeverSent);
    assert_eq!(
        plan.observe(Some(dec!(-0.0002))),
        Step::Close(dec!(-0.0002))
    );
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Stop(Flatness::Verified)
    );
}

#[test]
fn an_unreadable_read_never_counts_as_one_of_the_flat_confirmations() {
    // "We could not check" collapsing into "there is nothing there" is how a dropped
    // REST call becomes a false all-clear. It resets the count instead of adding to it.
    let mut plan = FlattenPlan::new(Entry::MaybeExecuted);
    assert_eq!(plan.observe(Some(Decimal::ZERO)), Step::Recheck);
    assert_eq!(plan.observe(None), Step::Unreadable);
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Recheck,
        "the failed read wiped the one confirmation we had; it did not stand in for it"
    );
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Stop(Flatness::NeverSeenOpen)
    );
}

#[test]
fn the_flatten_stops_sending_closes_once_its_budget_is_spent() {
    // A venue that keeps reporting the position must not turn a failing test into an
    // unbounded stream of orders. The budget is the same number `not_flat` quotes back.
    let mut plan = FlattenPlan::new(Entry::MaybeExecuted);
    for _ in 0..FLATTEN_ATTEMPTS {
        assert_eq!(
            plan.observe(Some(dec!(0.00019))),
            Step::Close(dec!(0.00019))
        );
    }
    assert_eq!(
        plan.observe(Some(dec!(0.00019))),
        Step::Spent(dec!(0.00019))
    );
    // Spent is not given up: a position seen open still makes the eventual flat read
    // real, which is what lets the driver keep reading after the last close.
    assert_eq!(
        plan.observe(Some(Decimal::ZERO)),
        Step::Stop(Flatness::Verified)
    );
}

#[test]
fn the_session_dedups_replayed_fills_before_asserting_on_them() {
    // `userFills` replays its snapshot on every reconnect, so the raw log can hold the
    // same execution twice. Counting frames instead of executions would report a
    // doubled fill for a tracker that correctly applied it once.
    let mut session = LiveSession::new();
    let sym = SymbolId::new(3);
    let oid = OrderId::new(56968034936);
    let fill = Fill {
        symbol_id: sym,
        order_id: oid,
        cloid: Some(Cloid::new(7)),
        side: Side::Buy,
        qty: dec!(0.00019),
        price: dec!(64715),
        fee: dec!(0.0055),
        closed_pnl: Decimal::ZERO,
        liquidity: Liquidity::Taker,
        trade_id: 351400627290190,
        ts_event: 1_785_003_324_667 * 1_000_000,
    };
    let other = Fill {
        order_id: OrderId::new(1),
        trade_id: 42,
        ..fill.clone()
    };
    for f in [fill.clone(), fill.clone(), other] {
        let ev = Event::Exec(ExecEvent::Fill(f));
        session.on_event(ev.ts_event(), &ev);
    }
    assert_eq!(session.fills.len(), 3, "the log keeps every frame");
    let ours = session.fills_for(oid);
    assert_eq!(ours.len(), 1, "deduped on trade_id");
    assert_eq!(ours[0].trade_id, 351400627290190);
    // And the tracker underneath applied the replayed execution exactly once.
    assert_eq!(
        session.tracker.position(sym).qty,
        dec!(0.00019) + dec!(0.00019)
    );
    assert_eq!(
        session.tracker.orphan_fills(),
        2,
        "neither order was acked here, so both fills are orphans - which is exactly \
         why the live test asserts orphans as a delta, not as zero"
    );
}

#[test]
fn the_session_feeds_both_the_book_and_the_tracker_from_one_bus() {
    // ADR-0010's one-bus rule, at the seam this test relies on: a market event must
    // reach the book and a fill must reach the tracker, from the same drain.
    use axon_core::{Bbo, MarketEvent};
    let sym = SymbolId::new(3);
    let mut session = LiveSession::new();
    let ev = Event::from(MarketEvent::Bbo(Bbo {
        symbol_id: sym,
        bid_px: dec!(64384),
        bid_sz: dec!(0.01129),
        ask_px: dec!(64393),
        ask_sz: dec!(0.01597),
        ts_event: 1_785_003_324_667 * 1_000_000,
    }));
    session.on_event(ev.ts_event(), &ev);
    assert_eq!(session.book.bbo(sym).map(|b| b.ask_px), Some(dec!(64393)));

    let update = OrderUpdate {
        symbol_id: sym,
        order_id: OrderId::new(9),
        cloid: None,
        side: Side::Buy,
        status: OrderStatus::Resting,
        price: Some(dec!(64715)),
        orig_qty: dec!(0.00019),
        remaining_qty: dec!(0.00019),
        cancel_reason: None,
        ts_event: 1_785_003_324_667 * 1_000_000,
    };
    let ev = Event::Exec(ExecEvent::Order(update));
    session.on_event(ev.ts_event(), &ev);
    assert_eq!(session.updates.len(), 1);
    assert_eq!(
        session.tracker.open_count(),
        1,
        "adopted, as ADR-0010 requires"
    );
}
