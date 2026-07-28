//! The live Hyperliquid WS client — the async *edge*.
//!
//! It connects with `tokio-tungstenite` (rustls), sends subscribe frames,
//! decodes each text frame via [`decode_ws_message`], and publishes the resulting
//! normalized events onto the core [`bus`](axon_core::bus). It handles heartbeats
//! and reconnect/resubscribe with capped backoff. This is the only place tokio
//! runs — the core stays synchronous and off the runtime.
//!
//! One connection carries both public market feeds and the account-scoped *user*
//! channels ([`UserChannel`]), and both are replayed identically on reconnect. That
//! is deliberate: fills and the book updates that caused them must share a socket
//! and a bus to stay orderable against each other.
//!
//! The decoding logic is unit-tested offline (see [`super::decode`] and
//! [`super::user`]); the *live* connection is exercised only by the `#[ignore]`d
//! smoke tests at the bottom, so CI stays deterministic and network-free. Run them
//! manually with: `cargo test -p axon-provider-hyperliquid -- --ignored`.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axon_core::{Clock, EventSender, SystemClock};
use axon_providers::{Capabilities, Feed, MarketData, ProviderError};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::decode::{decode_ws_message, venue_error, DecodeError, MS_TO_NS};
use super::funding::{FundingCadence, CADENCE_WINDOW_MS};
use super::rest::{fetch_funding_cadence, MAINNET_INFO, TESTNET_INFO};
use super::sub::{ping_msg, subscribe_msg, subscribe_user_msg, UserChannel};
use crate::symbol_map::SymbolMap;

/// Failure talking to Hyperliquid over WS/REST. Adapters map these to
/// [`ProviderError`] at the port boundary.
#[derive(Debug, thiserror::Error)]
pub enum HlError {
    #[error("websocket: {0}")]
    Ws(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("decode: {0}")]
    Decode(#[from] DecodeError),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
}

impl From<HlError> for ProviderError {
    fn from(e: HlError) -> Self {
        ProviderError::Network(e.to_string())
    }
}

// ── the reconnect policy ─────────────────────────────────────────────────────
//
// Both types below time things with [`Instant`] — monotonic wall time, and
// deliberately *not* event time. This is the one corner of the crate where that is
// the right clock: the backoff is a rate limit on our own connect attempts, so it
// asks "how long ago did this happen to me", which is not a question about the
// market. Nothing measured here reaches the bus, stamps a `ts_event` or orders an
// event against another.

/// What one connection achieved — the input to the backoff's reset decision.
///
/// Recorded *during* the connection rather than read off its outcome, because the
/// outcome cannot carry it: a severed link ends in `Err` whether the socket had been
/// up for two hours or two milliseconds, and those two must not be backed off the
/// same way.
#[derive(Default)]
struct ConnectionHealth {
    /// When the socket was established; `None` if it never was.
    ///
    /// Timed from the *established* socket, not from the start of the attempt, so a
    /// venue that takes twenty seconds to accept and then drops us cannot bank those
    /// twenty seconds as evidence of a healthy link.
    opened: Option<Instant>,
    /// Whether the venue ever sent a frame on this connection.
    heard_from_venue: bool,
    /// The core dropped the bus receiver: a shutdown, not a network event.
    bus_closed: bool,
}

impl ConnectionHealth {
    fn opened_now(&mut self) {
        self.opened = Some(Instant::now());
    }

    /// How long the socket was up; zero if it never opened.
    fn lived(&self) -> Duration {
        self.opened.map(|t| t.elapsed()).unwrap_or_default()
    }
}

/// Capped exponential backoff between connection attempts.
///
/// A value with its own tests rather than three lines inside the connect loop,
/// because the defect it replaces was invisible offline. The old loop reset its wait
/// only when `run_once` returned `Ok(())` — a *clean* WebSocket close, which no real
/// network failure produces (an RST arrives as `Connection reset by peer`, a FIN with
/// no Close frame as `Connection reset without closing handshake`; both are `Err`).
/// So the wait only ever doubled: eight disconnects pinned it at the 30 s cap for the
/// rest of the process, however healthy the link was in between. A 1 h 44 m soak
/// measured a **0.4 s** outage costing **30.0 s** of blackout in a session whose
/// first 0.8 s outage cost 0.8 s, and with `mark_max_age_ms = 10 000` that is a risk
/// gate refusing every risk-increasing order for 45.8 % of the session
/// (`docs/07-parity-and-testing.md`).
struct Backoff {
    wait: Duration,
}

impl Backoff {
    /// The floor: long enough not to be a retry storm, short enough to disappear
    /// beside a mark-staleness budget measured in seconds.
    const MIN: Duration = Duration::from_millis(250);
    /// The ceiling on a single wait.
    const MAX: Duration = Duration::from_secs(30);
    /// How long a connection must survive before it counts as evidence that the link
    /// is healthy.
    ///
    /// Equal to [`MAX`](Self::MAX), and the equality *is* the argument: whatever the
    /// venue does, it bounds us to about one connection attempt per `MAX`. A
    /// connection that dies sooner never resets the wait, so the wait keeps doubling
    /// to the cap; a connection that outlives the cap has already spent longer up
    /// than the longest wait we would have imposed, so resetting after it cannot make
    /// the attempt rate worse. Reset on anything cheaper — "the socket opened", "the
    /// subscribe went out", "a frame arrived" — and a venue that accepts connections
    /// and drops them a second later is hammered every 250 ms forever, which is
    /// exactly the abuse a backoff exists to prevent.
    const HEALTHY: Duration = Self::MAX;

    fn new() -> Self {
        Self { wait: Self::MIN }
    }

    /// Judge the connection that just ended — it was up for `lived` and
    /// `heard_from_venue` says whether the venue ever sent a frame on it — and return
    /// how long to wait before the next attempt.
    ///
    /// Both conditions, not either. Duration alone would reset on a socket the venue
    /// accepted and then blackholed, the partial-outage case
    /// [`HyperliquidMarketData::run_forever`] already calls strictly worse than a
    /// connect failure — zero events with zero errors reads as a healthy session.
    /// Hearing from the venue alone would reset on a venue that flaps every second,
    /// because Hyperliquid answers a subscribe with a snapshot immediately, so the
    /// first frame arrives within milliseconds of every connect however sick the
    /// endpoint is.
    fn after(&mut self, lived: Duration, heard_from_venue: bool) -> Duration {
        if heard_from_venue && lived >= Self::HEALTHY {
            self.wait = Self::MIN;
        }
        let wait = self.wait;
        self.wait = (self.wait * 2).min(Self::MAX);
        wait
    }
}

/// Hyperliquid WS ingest adapter. Construct it, register the feeds you want (via
/// [`MarketData::subscribe`] or [`Self::subscribe_coin`]) and optionally the user
/// channels for an account ([`Self::subscribe_user_channels`]), then drive it with
/// [`Self::run_forever`].
///
/// It remains read-*only* in the sense that it never sends orders — order entry is
/// the [`ExchangeClient`](crate::ExchangeClient)'s job over REST. What it adds here
/// is the venue's report of what happened to those orders.
pub struct HyperliquidMarketData {
    url: String,
    symbols: SymbolMap,
    sender: EventSender,
    caps: Capabilities,
    /// Coins this adapter knows about (targets for a coin-less `subscribe(feed)`).
    coins: Vec<String>,
    /// (feed, coin) pairs to (re)subscribe on every connect.
    desired: Mutex<Vec<(Feed, String)>>,
    /// (user channel, account address) pairs to (re)subscribe on every connect.
    /// Separate from `desired` only because the venue keys them on an address
    /// instead of a coin; they are replayed the same way.
    desired_user: Mutex<Vec<(UserChannel, String)>>,
}

impl HyperliquidMarketData {
    pub const MAINNET_WS: &'static str = "wss://api.hyperliquid.xyz/ws";
    pub const TESTNET_WS: &'static str = "wss://api.hyperliquid-testnet.xyz/ws";

    pub fn new(
        url: impl Into<String>,
        symbols: SymbolMap,
        coins: Vec<String>,
        sender: EventSender,
    ) -> Self {
        Self {
            url: url.into(),
            symbols,
            sender,
            caps: crate::capabilities(),
            coins,
            desired: Mutex::new(Vec::new()),
            desired_user: Mutex::new(Vec::new()),
        }
    }

    /// Adapter pointed at Hyperliquid mainnet.
    pub fn mainnet(symbols: SymbolMap, coins: Vec<String>, sender: EventSender) -> Self {
        Self::new(Self::MAINNET_WS, symbols, coins, sender)
    }

    /// Adapter pointed at Hyperliquid testnet (always test here before mainnet).
    pub fn testnet(symbols: SymbolMap, coins: Vec<String>, sender: EventSender) -> Self {
        Self::new(Self::TESTNET_WS, symbols, coins, sender)
    }

    /// Register interest in one `feed` for one `coin` (idempotent).
    pub fn subscribe_coin(&self, feed: Feed, coin: impl Into<String>) {
        let coin = coin.into();
        let mut d = self.desired.lock().expect("desired subs mutex");
        if !d.iter().any(|(f, c)| *f == feed && *c == coin) {
            d.push((feed, coin));
        }
    }

    fn desired_subs(&self) -> Vec<(Feed, String)> {
        self.desired.lock().expect("desired subs mutex").clone()
    }

    /// Register interest in one account-scoped `channel` for account `user`
    /// (idempotent). Takes effect on the next connect, like [`Self::subscribe_coin`].
    ///
    /// `user` is the address the orders belong to — the **master** or a sub-account —
    /// and *not* the agent/API-wallet address that signs them. The venue answers an
    /// agent address with empty results and no error, so the mistake looks exactly
    /// like an idle account. There is no authentication on these channels: no
    /// signature, no token, just the address.
    pub fn subscribe_user(&self, channel: UserChannel, user: impl Into<String>) {
        let user = user.into();
        let mut d = self.desired_user.lock().expect("desired user subs mutex");
        if !d.iter().any(|(c, u)| *c == channel && *u == user) {
            d.push((channel, user));
        }
    }

    /// Register the user channels needed to reconcile order flow for `user`:
    /// `orderUpdates` for lifecycle and `userFills` for fills.
    ///
    /// Deliberately **not** all three. `userEvents` also carries fills, so subscribing
    /// to it alongside `userFills` publishes every fill onto the bus twice. The
    /// duplicates are harmless downstream — `OrderTracker` dedups on `Fill::trade_id`
    /// precisely because the venue replays fills — but doubling bus traffic to rely on
    /// that is backwards. `userEvents`' unique content (funding, liquidation,
    /// `nonUserCancel`) is not yet part of the normalized vocabulary and is discarded by
    /// the decoder, so it currently contributes nothing but duplicates. Subscribe to it
    /// explicitly via [`subscribe_user`](Self::subscribe_user) if you want the raw
    /// stream; revisit this when funding/liquidation become `ExecEvent`s.
    pub fn subscribe_user_channels(&self, user: impl Into<String>) {
        let user = user.into();
        for channel in [UserChannel::OrderUpdates, UserChannel::UserFills] {
            self.subscribe_user(channel, user.clone());
        }
    }

    fn desired_user_subs(&self) -> Vec<(UserChannel, String)> {
        self.desired_user
            .lock()
            .expect("desired user subs mutex")
            .clone()
    }

    /// The `POST /info` endpoint that pairs with this adapter's WS URL.
    ///
    /// `None` for anything else — a test double or a proxy — because guessing an
    /// info URL from an arbitrary WS URL would send a live request somewhere nobody
    /// asked us to talk to.
    fn info_url(&self) -> Option<&'static str> {
        match self.url.as_str() {
            Self::MAINNET_WS => Some(MAINNET_INFO),
            Self::TESTNET_WS => Some(TESTNET_INFO),
            _ => None,
        }
    }

    /// Cross-check the funding interval this adapter stamps on every
    /// [`Ticker`](axon_core::Ticker) against the cadence the venue actually charges
    /// on, and log loudly if they disagree.
    ///
    /// `activeAssetCtx` carries a funding *rate* and no period, so the period comes
    /// from [`ASSUMED_FUNDING_INTERVAL_NS`](super::funding::ASSUMED_FUNDING_INTERVAL_NS).
    /// A venue that moved to a four- or eight-hourly charge would not change the shape
    /// of a single frame — every carry number would just quietly scale, so the only
    /// way to notice is to go and ask. This is that ask, once per session.
    ///
    /// Returns `None`, and sends nothing, when there is no ticker subscription to be
    /// wrong about or the endpoint is not one of Hyperliquid's. Never fatal: a failed
    /// probe is a diagnostic that did not run, not a reason to refuse market data.
    pub async fn verify_funding_cadence(&self) -> Option<FundingCadence> {
        // Only pay for it when a ticker feed is actually registered — the constant is
        // harmless until something stamps it onto an event.
        let coin = self
            .desired_subs()
            .into_iter()
            .find(|(f, _)| *f == Feed::Ticker)
            .map(|(_, c)| c)?;
        let info_url = self.info_url()?;

        // The venue keys `startTime` in ms; the one ms↔ns factor in the crate.
        let start_ms = SystemClock.now_ns() / MS_TO_NS - CADENCE_WINDOW_MS;
        match fetch_funding_cadence(info_url, &coin, start_ms).await {
            Ok(cadence) => {
                if cadence.differs() {
                    eprintln!(
                        "hyperliquid funding cadence CHANGED: {cadence:?} — every Ticker is \
                         stamped with the assumed interval, so funding carry is now wrong by \
                         the ratio. See crates/axon-provider-hyperliquid/src/ws/funding.rs."
                    );
                } else {
                    // One line on success too. A check that only speaks when it fails
                    // is indistinguishable, in a log, from a check that never ran —
                    // and this one is skipped for several ordinary reasons.
                    eprintln!("hyperliquid funding cadence checked ({coin}): {cadence:?}");
                }
                Some(cadence)
            }
            Err(e) => {
                eprintln!("hyperliquid funding cadence check skipped ({coin}): {e}");
                None
            }
        }
    }

    /// One connection attempt: connect, subscribe, then pump frames until the
    /// socket closes (`Ok`) or a transport error occurs (`Err`). Returns `Ok`
    /// early if the core dropped the bus receiver (shutdown).
    ///
    /// `Ok` is rarer than it reads, and [`Self::reconnect_forever`] depends on
    /// knowing it: reaching it needs the venue to send a WebSocket **Close frame**. No
    /// severed link does that — an RST surfaces as an IO error and a FIN with no
    /// Close as `ResetWithoutClosingHandshake`, both `Err`. So which arm this returns
    /// through says almost nothing about whether the connection was healthy, and
    /// [`Backoff`] judges it on [`ConnectionHealth`] instead.
    pub async fn run_once(&self) -> Result<(), HlError> {
        self.run_once_recording(&mut ConnectionHealth::default())
            .await
    }

    /// [`Self::run_once`], recording what the connection achieved into `health`.
    ///
    /// Split out only because that record has to survive the `?` early-returns: the
    /// backoff needs it precisely on the paths that end in an error, which in
    /// practice is all of them.
    async fn run_once_recording(&self, health: &mut ConnectionHealth) -> Result<(), HlError> {
        let (ws, _resp) = connect_async(&self.url).await?;
        health.opened_now();
        let (mut write, mut read) = ws.split();

        for (feed, coin) in self.desired_subs() {
            write
                .send(Message::text(subscribe_msg(feed, &coin)))
                .await?;
        }
        // Replayed on every connect exactly like the market subscriptions. This is
        // not optional politeness: `orderUpdates` and `userEvents` never snapshot, so
        // a reconnect that forgot to resubscribe would go quiet and look like an idle
        // account. (Resting-order state missed *during* the gap still has to come
        // from `POST /info` — the socket cannot replay it.)
        for (channel, user) in self.desired_user_subs() {
            write
                .send(Message::text(subscribe_user_msg(channel, &user)))
                .await?;
        }

        // Hyperliquid drops idle sockets (~60s); ping well inside that window.
        let mut heartbeat = tokio::time::interval(Duration::from_secs(50));
        heartbeat.tick().await; // the first tick is immediate — skip it

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    write.send(Message::text(ping_msg())).await?;
                }
                item = read.next() => {
                    let Some(msg) = item else { return Ok(()) }; // stream ended
                    let msg = msg?;
                    // The venue spoke on this socket. That the *subscribe* went out
                    // proves nothing — `write.send` succeeds into a socket buffer
                    // with nobody at the far end — so this is the only evidence the
                    // link carries traffic in the direction we need, and the backoff
                    // will not treat a connection without it as a healthy one.
                    health.heard_from_venue = true;
                    match msg {
                        Message::Text(txt) => {
                            // Read before parsing, not after: this is the ordering key
                            // for `activeAssetCtx`, the one feed Hyperliquid ships
                            // with no timestamp of its own, and the honest answer
                            // there is when the frame landed — not when we happened to
                            // finish decoding it. Every other feed ignores it in
                            // favour of the venue's own time.
                            let ts_ingest = SystemClock.now_ns();
                            match venue_error(txt.as_str()) {
                                // A rejected subscription arrives as an ordinary frame
                                // on a healthy socket. Log it and keep the connection: a
                                // teardown would only reconnect and resend the same bad
                                // subscription forever, while dropping the *good* ones.
                                Some(msg) => eprintln!("hyperliquid WS error frame: {msg}"),
                                None => {
                                    match decode_ws_message(txt.as_str(), &self.symbols, ts_ingest) {
                                        Ok(events) => {
                                            for ev in events {
                                                // A dropped receiver means the core is
                                                // shutting down.
                                                if self.sender.send(ev).is_err() {
                                                    health.bus_closed = true;
                                                    return Ok(());
                                                }
                                            }
                                        }
                                        // A decode failure is a *data* problem, not a
                                        // transport one, and must not close the socket.
                                        // `userFills` redelivers its whole snapshot on
                                        // every reconnect, so propagating here would
                                        // mean: one malformed entry tears down the
                                        // connection, the reconnect replays the same
                                        // entry, and we loop forever while every good
                                        // feed stays dark. Log and drop the frame; the
                                        // tracker recovers missed state from
                                        // `POST /info` regardless. A
                                        // `DecodeError::UnsupportedChannel` (a spot
                                        // asset context) lands here too, deliberately
                                        // loud, because a silently ignored one is a
                                        // subscription that never yields a price.
                                        Err(e) => eprintln!(
                                            "hyperliquid WS decode error (frame dropped): {e}"
                                        ),
                                    }
                                }
                            }
                        }
                        Message::Ping(payload) => {
                            write.send(Message::Pong(payload)).await?;
                        }
                        Message::Close(_) => return Ok(()),
                        _ => {} // Binary / Pong / Frame: nothing to do
                    }
                }
            }
        }
    }

    /// Run forever: reconnect and resubscribe on any disconnect, with capped
    /// exponential backoff. Returns only when the core drops the bus receiver.
    ///
    /// The once-per-session funding probe runs *beside* the connect loop, not in front
    /// of it. Awaited first, it put a `POST /info` on the critical path of the market
    /// data socket: an endpoint that accepts the connection and then answers nothing
    /// (partial outage, an LB blackhole, an inline proxy) would hold the socket down
    /// forever, while the session above it has already printed `session up` and armed
    /// the dead-man's switch. Zero events with zero errors reads as a healthy session —
    /// strictly worse than a connect failure, which at least logs and backs off. The
    /// request is bounded as well ([`INFO_TIMEOUT`](super::rest::INFO_TIMEOUT)); both,
    /// because the socket's liveness must not depend on a diagnostic being well behaved.
    ///
    /// Still once per session and still not inside `run_once`: a flapping socket would
    /// otherwise re-probe on every reconnect, spending venue budget on a number that
    /// changes about never.
    pub async fn run_forever(&self) {
        let (_cadence, ()) = tokio::join!(self.verify_funding_cadence(), self.reconnect_forever());
    }

    /// Connect, pump, and reconnect with capped exponential backoff ([`Backoff`]).
    ///
    /// Every ended connection is judged the same way and waited out the same way,
    /// whichever arm it returned through. The old loop treated `Ok` and `Err` as two
    /// different policies — reset-and-retry-instantly against double-the-wait — and
    /// both halves were wrong for the same reason: the *outcome* of a connection says
    /// nothing about whether it was worth having. A severed link is always `Err`
    /// however long it had been healthy, and a venue that accepts a connection and
    /// immediately closes it politely is always `Ok`, which the old loop answered with
    /// an unthrottled reconnect storm.
    ///
    /// Returns only when the core drops the bus receiver.
    async fn reconnect_forever(&self) {
        let mut backoff = Backoff::new();
        loop {
            let mut health = ConnectionHealth::default();
            let outcome = self.run_once_recording(&mut health).await;
            if health.bus_closed {
                // The core is shutting down. Reconnecting into a closed bus would
                // spin a connect loop against the venue for as long as the process
                // takes to exit — every connection succeeding, delivering one frame,
                // and dying on the same closed channel.
                return;
            }
            let lived = health.lived();
            let wait = backoff.after(lived, health.heard_from_venue);
            match outcome {
                // Logged even though it is not an error: a socket that closed and
                // came back with no line in the log is indistinguishable, afterwards,
                // from one that never closed.
                Ok(()) => {
                    eprintln!("hyperliquid WS closed after {lived:.1?}; reconnecting in {wait:?}")
                }
                // A connect that never established is a different event from a
                // connection that died, and sharing one line makes it read as
                // nonsense — "error after 0.0ns" — on exactly the retries an
                // operator is scanning to tell an outage from a bad endpoint.
                Err(e) if health.opened.is_none() => {
                    eprintln!("hyperliquid WS connect failed: {e}; retrying in {wait:?}")
                }
                Err(e) => eprintln!(
                    "hyperliquid WS error after {lived:.1?}: {e}; reconnecting in {wait:?}"
                ),
            }
            tokio::time::sleep(wait).await;
        }
    }
}

#[async_trait]
impl MarketData for HyperliquidMarketData {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Register `feed` for every configured coin. Takes effect on the next
    /// connect (the current Phase-2 model; a live control channel comes later).
    async fn subscribe(&self, feed: Feed) -> Result<(), ProviderError> {
        for coin in self.coins.clone() {
            self.subscribe_coin(feed, coin);
        }
        Ok(())
    }

    async fn unsubscribe(&self, feed: Feed) -> Result<(), ProviderError> {
        self.desired
            .lock()
            .expect("desired subs mutex")
            .retain(|(f, _)| *f != feed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── the reconnect policy, offline ────────────────────────────────────────

    /// A connection long and healthy enough to be worth having.
    const HEALTHY_RUN: Duration = Duration::from_secs(120);
    /// A connection that died before it proved anything.
    const SHORT_RUN: Duration = Duration::from_secs(1);

    /// Drive the backoff to its cap the way a flapping link does. Eight doublings
    /// reach it; twelve is deliberately well past, because the state that mattered in
    /// the soak was the one a session sits in for its remaining hour.
    fn pinned_at_the_cap() -> Backoff {
        let mut b = Backoff::new();
        for _ in 0..12 {
            b.after(Duration::ZERO, false);
        }
        assert_eq!(b.after(Duration::ZERO, false), Backoff::MAX);
        b
    }

    #[test]
    fn a_healthy_connection_resets_the_backoff_so_a_late_outage_costs_what_an_early_one_did() {
        // The soak's finding, as arithmetic. Before this, the wait was reset only by
        // `run_once` returning `Ok` — a clean WebSocket close, which a severed link
        // never produces — so eight disconnects pinned it at 30 s for the rest of the
        // process. A 0.4 s outage then cost 30.0 s of blackout, 75× the outage, and
        // every mark in the risk gate expired while it waited.
        let mut b = pinned_at_the_cap();
        assert_eq!(
            b.after(HEALTHY_RUN, true),
            Backoff::MIN,
            "two healthy minutes between two outages is the link telling us it is fine"
        );
        // And it climbs again from the floor, rather than snapping back to the cap.
        assert_eq!(b.after(SHORT_RUN, true), Duration::from_millis(500));

        // The threshold is inclusive: a connection that lasted exactly as long as the
        // longest wait we would impose has earned the reset.
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(Backoff::HEALTHY, true), Backoff::MIN);
    }

    #[test]
    fn a_venue_that_accepts_connections_and_drops_them_is_never_hot_looped() {
        // The failure mode the backoff exists for, and the one a careless fix
        // reintroduces. Every connection here *delivers data* — Hyperliquid answers a
        // subscribe with a snapshot within milliseconds — so "we got a frame" is not
        // evidence of anything, and treating it as evidence would hammer a sick
        // endpoint at 250 ms forever.
        let mut b = Backoff::new();
        let mut total = Duration::ZERO;
        for _ in 0..20 {
            total += b.after(SHORT_RUN, true);
        }
        assert_eq!(b.after(SHORT_RUN, true), Backoff::MAX);
        assert!(
            total >= Duration::from_secs(200),
            "twenty one-second connections should have been throttled, not retried: {total:?}"
        );
    }

    #[test]
    fn a_socket_the_venue_accepted_and_then_blackholed_is_not_a_healthy_connection() {
        // The partial outage `run_forever` warns about: an endpoint that accepts and
        // then says nothing. It can hold a socket open for hours, so duration alone
        // would call it healthy and reset the wait every time — and a session that
        // reconnects promptly into silence is exactly as dark as one that cannot
        // connect at all, minus the error that would say so.
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(Duration::from_secs(3600), false), Backoff::MAX);
    }

    #[test]
    fn a_dead_venue_still_waits_the_full_cap() {
        // A connect that never completes leaves `lived` at zero, so nothing resets:
        // the point of the fix is a faster return from a *recovered* link, never a
        // faster retry against one that is simply down.
        let mut b = pinned_at_the_cap();
        for _ in 0..10 {
            assert_eq!(b.after(Duration::ZERO, false), Backoff::MAX);
        }
    }

    #[test]
    fn the_wait_is_bounded_below_by_the_floor_and_above_by_the_cap() {
        // Whatever the sequence of outcomes, a wait outside the band is one of the two
        // failures this policy exists to prevent: below the floor is a retry storm at
        // the venue, above the cap is a session that stays dark long after the link
        // came back.
        let mut b = Backoff::new();
        for (lived, heard) in [
            (HEALTHY_RUN, true),
            (SHORT_RUN, false),
            (Duration::ZERO, false),
            (HEALTHY_RUN, false),
            (HEALTHY_RUN, true),
        ] {
            let w = b.after(lived, heard);
            assert!((Backoff::MIN..=Backoff::MAX).contains(&w), "{w:?}");
        }
    }

    #[test]
    fn subscribe_coin_is_idempotent() {
        let (tx, _rx) = axon_core::bus(16);
        let md =
            HyperliquidMarketData::mainnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_coin(Feed::Bbo, "BTC");
        md.subscribe_coin(Feed::Bbo, "BTC"); // duplicate ignored
        md.subscribe_coin(Feed::L2Book, "BTC");
        assert_eq!(md.desired_subs().len(), 2);
    }

    #[tokio::test]
    async fn ticker_needs_no_special_case_anywhere_in_the_loop() {
        // The mark-price feed is carried by the *existing* machinery or not at all.
        // `run_once` replays `desired` verbatim on every connect, so a feed that lands
        // in `desired` the same way the others do is a feed that survives a reconnect;
        // a `Feed::Ticker` arm added to the pump instead would be one more place for a
        // resubscribe to be forgotten, and a silently unsubscribed mark feed reads as a
        // risk gate that fails closed for no visible reason.
        let (tx, _rx) = axon_core::bus(16);
        let md = HyperliquidMarketData::mainnet(
            SymbolMap::from_perps(["BTC", "ETH"]),
            vec!["BTC".into(), "ETH".into()],
            tx,
        );
        md.subscribe(Feed::Ticker).await.unwrap();
        let subs = md.desired_subs();
        assert_eq!(subs.len(), 2, "one per configured coin, like every feed");
        assert!(subs.iter().all(|(f, _)| *f == Feed::Ticker));
        md.subscribe_coin(Feed::Ticker, "BTC"); // still idempotent
        assert_eq!(md.desired_subs().len(), 2);
        md.unsubscribe(Feed::Ticker).await.unwrap();
        assert!(md.desired_subs().is_empty());
    }

    #[tokio::test]
    async fn coinless_subscribe_expands_to_all_coins() {
        let (tx, _rx) = axon_core::bus(16);
        let md = HyperliquidMarketData::mainnet(
            SymbolMap::from_perps(["BTC", "ETH"]),
            vec!["BTC".into(), "ETH".into()],
            tx,
        );
        md.subscribe(Feed::Trades).await.unwrap();
        assert_eq!(md.desired_subs().len(), 2); // one per coin
        md.unsubscribe(Feed::Trades).await.unwrap();
        assert_eq!(md.desired_subs().len(), 0);
    }

    #[test]
    fn user_subscriptions_are_idempotent_and_independent_of_market_subs() {
        const ACCOUNT: &str = "0x9be9c0f9c1e4e4a1b0d0f2e3d4c5b6a7f8091a2b";
        let (tx, _rx) = axon_core::bus(16);
        let md =
            HyperliquidMarketData::mainnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_coin(Feed::Bbo, "BTC");
        md.subscribe_user(UserChannel::OrderUpdates, ACCOUNT);
        md.subscribe_user(UserChannel::OrderUpdates, ACCOUNT); // duplicate ignored
        assert_eq!(md.desired_user_subs().len(), 1);

        md.subscribe_user_channels(ACCOUNT);
        let subs = md.desired_user_subs();
        assert_eq!(
            subs.len(),
            2,
            "the already-registered channel is not duplicated"
        );
        // Only orderUpdates + userFills. userEvents is excluded on purpose: it also
        // carries fills, so registering it here would publish every fill twice.
        assert!(subs.iter().any(|(c, _)| *c == UserChannel::OrderUpdates));
        assert!(subs.iter().any(|(c, _)| *c == UserChannel::UserFills));
        assert!(
            !subs.iter().any(|(c, _)| *c == UserChannel::UserEvents),
            "userEvents would duplicate every fill from userFills"
        );
        // Market and user registries do not interfere.
        assert_eq!(md.desired_subs().len(), 1);
    }

    #[tokio::test]
    async fn a_session_with_no_ticker_feed_is_not_probed_for_funding_history() {
        // The funding interval is only a hazard once something stamps it onto an
        // event. A book-only session that spent a venue request on it would be paying
        // rate-limit budget for a number it never uses — and, worse, would make the
        // offline unit suite depend on the network.
        let (tx, _rx) = axon_core::bus(16);
        let md =
            HyperliquidMarketData::mainnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_coin(Feed::L2Book, "BTC");
        assert_eq!(md.verify_funding_cadence().await, None);
    }

    #[tokio::test]
    async fn an_unknown_endpoint_is_never_guessed_into_an_info_url() {
        // A test double or a proxy is not Hyperliquid. Deriving an info URL from an
        // arbitrary WS URL would send a live POST somewhere nobody configured, which
        // is how an "offline" test suite quietly starts talking to the internet.
        let (tx, _rx) = axon_core::bus(16);
        let md = HyperliquidMarketData::new(
            "ws://127.0.0.1:1/ws",
            SymbolMap::from_perps(["BTC"]),
            vec!["BTC".into()],
            tx,
        );
        md.subscribe_coin(Feed::Ticker, "BTC");
        assert_eq!(md.info_url(), None);
        assert_eq!(md.verify_funding_cadence().await, None);
    }

    #[test]
    fn a_second_account_gets_its_own_subscriptions() {
        let (tx, _rx) = axon_core::bus(16);
        let md =
            HyperliquidMarketData::mainnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_user(UserChannel::UserFills, "0xaaa");
        md.subscribe_user(UserChannel::UserFills, "0xbbb");
        assert_eq!(md.desired_user_subs().len(), 2, "keyed on (channel, user)");
    }

    /// The committed fixture, re-checked against the venue that produced it.
    ///
    /// `testdata/non-data-frames.jsonl` exists because the *hand-written* version of
    /// it was wrong for a whole phase: the venue's heartbeat reply is
    /// `{"channel":"pong"}` with no `data` field, while the unit test asserted
    /// `{"channel":"pong","data":null}` — a frame Hyperliquid does not send. The test
    /// passed the entire time every real pong was being logged as
    /// `WS decode error (frame dropped)`. A fixture nobody re-checks decays back into
    /// an assumption, so this is the re-check, and it also proves the frames decode
    /// silently rather than merely parsing.
    #[tokio::test]
    #[ignore = "hits the live Hyperliquid WS"]
    async fn the_venues_non_data_frames_still_match_the_fixture() {
        use crate::ws::decode::captured_frame;

        let (ws, _resp) = connect_async(HyperliquidMarketData::TESTNET_WS)
            .await
            .expect("connect to Hyperliquid testnet");
        let (mut write, mut read) = ws.split();
        // The same subscription the fixture was captured through — the venue echoes it
        // back inside the `subscriptionResponse`, so a different one is a different
        // frame.
        write
            .send(Message::text(subscribe_msg(Feed::Bbo, "BTC")))
            .await
            .expect("subscribe");
        write
            .send(Message::text(ping_msg()))
            .await
            .expect("heartbeat");

        let mut pong = None;
        let mut sub_response = None;
        let deadline = Instant::now() + Duration::from_secs(20);
        while (pong.is_none() || sub_response.is_none()) && Instant::now() < deadline {
            let Ok(Some(Ok(msg))) = tokio::time::timeout(Duration::from_secs(5), read.next()).await
            else {
                break;
            };
            let Message::Text(txt) = msg else { continue };
            let txt = txt.as_str().to_string();
            if txt.contains(r#""channel":"pong""#) {
                pong.get_or_insert(txt);
            } else if txt.contains(r#""channel":"subscriptionResponse""#) {
                sub_response.get_or_insert(txt);
            }
        }
        let pong = pong.expect("no pong within 20s of a ping");
        let sub_response = sub_response.expect("no subscriptionResponse within 20s");

        for (live, fixture) in [
            (&pong, captured_frame("pong")),
            (&sub_response, captured_frame("subscriptionResponse")),
        ] {
            assert_eq!(
                live, fixture,
                "the venue's frame has changed; re-capture testdata/non-data-frames.jsonl \
                 rather than relaxing this"
            );
            assert!(
                decode_ws_message(live, &SymbolMap::from_perps(["BTC"]), 0)
                    .expect("a non-data frame must not be a decode error")
                    .is_empty(),
                "{live} produced events"
            );
        }
    }

    /// Live smoke test — hits the real Hyperliquid WS. Ignored by default so CI
    /// stays deterministic and offline. Run with:
    /// `cargo test -p axon-provider-hyperliquid -- --ignored`.
    #[tokio::test]
    #[ignore = "hits the live Hyperliquid WS"]
    async fn live_bbo_smoke() {
        let (tx, rx) = axon_core::bus(1024);
        // Fake indices are fine for a smoke test — we only assert an event arrives.
        let md =
            HyperliquidMarketData::mainnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_coin(Feed::Bbo, "BTC");
        md.subscribe_coin(Feed::Trades, "BTC");

        let stream = tokio::spawn(async move {
            let _ = md.run_once().await;
        });
        tokio::time::sleep(Duration::from_secs(8)).await;
        stream.abort();

        assert!(
            rx.try_recv().is_some(),
            "expected at least one live market event within 8s"
        );
    }

    // ── the relay-driven reconnect reproducer ────────────────────────────────
    //
    // These need the soak harness's loopback relay, which severs the TCP connection
    // *underneath* the client's socket — the only honest way to produce the failure
    // the backoff exists for (`scripts/soak/ws-relay.py` explains why simulating one
    // in the client tests the simulator). Start it with
    // `scripts/soak/relay-start.sh 8765`, then:
    // `cargo test -p axon-provider-hyperliquid -- --ignored --nocapture reconnect`.
    //
    // The relay's retries land on this box's own loopback during an outage, so an
    // outage costs Hyperliquid nothing; the only traffic that reaches the venue is
    // one WS connection per successful reconnect.

    const RELAY_WS: &str = "ws://127.0.0.1:8765/ws";
    const RELAY_CTL: &str = "/dev/shm/m7-relay.ctl";

    fn relay_ws() -> String {
        std::env::var("AXON_SOAK_RELAY_WS").unwrap_or_else(|_| RELAY_WS.to_string())
    }

    fn relay_ctl() -> String {
        std::env::var("AXON_SOAK_RELAY_CTL").unwrap_or_else(|_| RELAY_CTL.to_string())
    }

    fn relay_deadline(secs: f64) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before the epoch")
            .as_secs_f64()
            + secs
    }

    fn relay_control(line: String) {
        std::fs::write(relay_ctl(), line)
            .expect("relay control file — is scripts/soak/relay-start.sh running?");
    }

    /// Tell the relay to sever every connection and refuse new ones for `secs`.
    fn relay_cut(secs: f64) {
        relay_control(format!("cut {}\n", relay_deadline(secs)));
    }

    /// Tell the relay to relay normally but end each connection `life_ms` after it
    /// opens — a FIN in both directions, with no WebSocket Close frame.
    fn relay_churn(secs: f64, life_ms: u64) {
        relay_control(format!("churn {} {life_ms}\n", relay_deadline(secs)));
    }

    /// The claim the whole reconnect policy rests on, checked against a real socket:
    /// a connection the peer severs **never** reports a clean close.
    ///
    /// `churn` is the gentlest severance available — FIN in both directions, long
    /// after the handshake and the subscribe frames went through, which is as close to
    /// a graceful close as a network fault gets. It still surfaces as `Err`, because a
    /// WebSocket close is a *frame* and a FIN is not one. So the old
    /// `Ok(()) => reset the backoff` branch was dead to every network event, and that
    /// is why the backoff never reset once in 1 h 44 m and 36 outages: the reset
    /// condition was not merely too strict, it was unreachable.
    #[tokio::test]
    #[ignore = "needs the loopback soak relay: scripts/soak/relay-start.sh 8765"]
    async fn a_severed_link_never_reports_a_clean_close() {
        let (tx, _rx) = axon_core::bus(4096);
        let md = HyperliquidMarketData::new(
            relay_ws(),
            SymbolMap::from_perps(["BTC"]),
            vec!["BTC".into()],
            tx,
        );
        md.subscribe_coin(Feed::L2Book, "BTC");
        relay_churn(20.0, 3000);
        // The relay stamps a connection's life when it *accepts* it, and polls its
        // control file every 50 ms. Connecting inside that window is accepted in
        // `open` mode and never churned — the connection then outlives the test and
        // the failure looks like "a severed link was never severed".
        tokio::time::sleep(Duration::from_millis(400)).await;
        let outcome = tokio::time::timeout(Duration::from_secs(30), md.run_once())
            .await
            .expect("run_once never returned from a connection with a 3s life");
        relay_control("open\n".to_string());
        let err = outcome.expect_err(
            "a FIN with no Close frame was reported as a clean close — if the venue or \
             tungstenite really has started closing gracefully, the backoff's reset \
             condition needs revisiting, not this assertion",
        );
        eprintln!("a severed link surfaces as: {err}");
    }

    /// Induce an outage of `secs` and return the resulting data blackout: the time
    /// from the moment the outage was ordered to the first event that arrives after
    /// it. Always at least `secs`; the interesting part is by how much it exceeds it.
    async fn blackout_after_a_cut(rx: &axon_core::EventReceiver, secs: f64) -> Option<Duration> {
        while rx.try_recv().is_some() {} // pre-outage events answer the wrong question
        let ordered = std::time::Instant::now();
        relay_cut(secs);
        // The relay polls its control file every 50 ms, so frames sent in that window
        // are still from before the outage. Drain them rather than let one of them
        // report a blackout of nothing.
        tokio::time::sleep(Duration::from_millis(150)).await;
        while rx.try_recv().is_some() {}
        loop {
            if rx.try_recv().is_some() {
                return Some(ordered.elapsed());
            }
            if ordered.elapsed() > Duration::from_secs(90) {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// The soak's finding, as a test: **the same short outage must cost the same
    /// whether it is the session's first or its twentieth.**
    ///
    /// Before the fix, `reconnect_forever` reset its backoff only when `run_once`
    /// returned `Ok` — a clean WebSocket close, which a severed link never produces —
    /// so eight disconnects pinned the wait at the 30 s cap for the rest of the
    /// process. Measured over a 1 h 44 m soak: a 0.8 s outage cost 0.8 s of blackout
    /// when it came first, and a **0.4 s** outage cost **30.0 s** two hours later
    /// (docs/07-parity-and-testing.md). With `mark_max_age_ms = 10 000` every one of
    /// those windows expires every mark, so the risk gate refuses every
    /// risk-increasing order for 30 s at a time.
    ///
    /// The ladder in the middle is what makes this a regression test rather than a
    /// connectivity check: it drives the backoff to its cap first, so a run that
    /// passes cannot have passed by never leaving the floor.
    #[tokio::test]
    #[ignore = "needs the loopback soak relay: scripts/soak/relay-start.sh 8765"]
    async fn a_short_outage_late_in_a_session_costs_what_the_same_outage_cost_first() {
        let (tx, rx) = axon_core::bus(4096);
        let md = HyperliquidMarketData::new(
            relay_ws(),
            SymbolMap::from_perps(["BTC", "ETH", "SOL"]),
            vec!["BTC".into(), "ETH".into(), "SOL".into()],
            tx,
        );
        for coin in ["BTC", "ETH", "SOL"] {
            md.subscribe_coin(Feed::L2Book, coin);
            md.subscribe_coin(Feed::Bbo, coin);
        }
        let session = tokio::spawn(async move { md.run_forever().await });

        // Wait for the session to be live before measuring anything.
        let mut up = false;
        for _ in 0..300 {
            if rx.try_recv().is_some() {
                up = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(up, "no market data through the relay within 30s");

        let first = blackout_after_a_cut(&rx, 0.8)
            .await
            .expect("no recovery from the session's first outage");
        eprintln!("first 0.8s outage → blackout {first:?}");

        // Climb the ladder: three outages with too little recovery between them to
        // count as a healthy link. This is what pins the old code at the cap.
        for i in 0..3 {
            let b = blackout_after_a_cut(&rx, 3.0).await;
            eprintln!("ladder outage {i} (3.0s) → blackout {b:?}");
            tokio::time::sleep(Duration::from_secs(6)).await;
        }

        // Then a stretch of *healthy* link — longer than any wait this loop would
        // ever impose — which is the whole difference between the two behaviours.
        tokio::time::sleep(Duration::from_secs(45)).await;

        let last = blackout_after_a_cut(&rx, 0.8)
            .await
            .expect("no recovery from the session's last outage");
        eprintln!("last 0.8s outage → blackout {last:?} (first was {first:?})");
        session.abort();

        assert!(
            last <= first + Duration::from_secs(3),
            "an identical outage cost {last:?} late in the session against {first:?} at \
             the start — the backoff is not resetting on a healthy connection"
        );
    }

    /// Live user-channel smoke test. Asserts only that subscribing does not tear the
    /// connection down — it *cannot* assert that events arrive, because `userEvents`
    /// and `orderUpdates` never snapshot and an idle account is legitimately silent.
    /// The master address (not the agent wallet) comes from `AXON_HL_ACCOUNT_ADDRESS`,
    /// which is the name `.env.example` and `with-env.sh` actually define; the older
    /// `AXON_HL_ACCOUNT` is still honoured as an override.
    ///
    /// It reads the canonical name because `AXON_HL_ACCOUNT` is defined nowhere in this
    /// repo, so this test panicked on every `./run.sh live` — and cargo abandons a run
    /// at the first failing test *binary*, which meant this panic also stopped
    /// `live_fill_testnet` from ever executing. The failure an operator saw was never
    /// the one that mattered.
    #[tokio::test]
    #[ignore = "hits the live Hyperliquid WS and needs AXON_HL_ACCOUNT_ADDRESS"]
    async fn live_user_channel_subscribe_smoke() {
        let Ok(account) =
            std::env::var("AXON_HL_ACCOUNT").or_else(|_| std::env::var("AXON_HL_ACCOUNT_ADDRESS"))
        else {
            panic!("set AXON_HL_ACCOUNT_ADDRESS to the master account address");
        };
        let (tx, _rx) = axon_core::bus(1024);
        let md =
            HyperliquidMarketData::testnet(SymbolMap::from_perps(["BTC"]), vec!["BTC".into()], tx);
        md.subscribe_user_channels(account.as_str());
        // A bad subscription would be logged as an error frame, not raised — so what
        // this proves is that the socket survives all three subscriptions.
        let stream = tokio::spawn(async move { md.run_once().await });
        tokio::time::sleep(Duration::from_secs(8)).await;
        assert!(!stream.is_finished(), "the connection closed early");
        stream.abort();
    }
}
