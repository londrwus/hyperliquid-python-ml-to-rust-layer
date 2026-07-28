//! The live Binance USD-M WS client — the async *edge*, and the only place tokio
//! runs in this crate.
//!
//! It connects with `tokio-tungstenite` (rustls), decodes each text frame via
//! [`decode_ws_message`], and publishes normalized events onto the core
//! core bus. The core stays synchronous and off the runtime: nothing
//! this module exports carries a runtime handle, and the only thing that crosses the
//! seam is an [`EventSender`], which is a crossbeam channel.
//!
//! The decoding is unit-tested offline against captured frames ([`super::decode`]);
//! the live connection is exercised only by the `#[ignore]`d smoke test at the
//! bottom. **It has never been run.**
//!
//! ## One difference from the Hyperliquid client, and it is structural
//!
//! Hyperliquid subscribes *after* connecting, so its `desired` set can be replayed on
//! every reconnect by sending frames. Binance takes the stream list in the **connect
//! URL** (`/stream?streams=a/b/c`), which means the subscription is re-established by
//! the act of reconnecting and cannot be forgotten — but also that a subscription
//! added mid-session takes effect on the next connect. That is the same limitation
//! the Hyperliquid adapter documents for its own `subscribe`, arrived at from the
//! opposite direction.
//!
//! The URL form is used because it is the one these fixtures were captured through
//! and is therefore the one shape of this client anybody has watched work. The
//! `SUBSCRIBE`/`UNSUBSCRIBE` builders in [`super::stream`] are tested and unused for
//! the same reason ADR-0011 keeps `is_venue_timed`: the day a live control channel
//! lands, the frame format should not also be new.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use axon_core::{Clock, EventSender, SymbolId, SystemClock};
use axon_providers::{Capabilities, Feed, MarketData, ProviderError};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::decode::{decode_ws_message, venue_error, DecodeError};
use super::stream::{stream_name, BookDepth, UpdateSpeed};
use crate::exchange_info::ExchangeInfoError;
use crate::symbols::SymbolTable;

// `Instant` and not the event clock, deliberately. The backoff is a rate limit on our
// own connect attempts — "how long ago did this happen to me" — which is not a question
// about the market. Nothing measured here reaches the bus, stamps a `ts_event` or
// orders one event against another.

/// What one connection achieved — the input to the backoff's reset decision.
///
/// Recorded *during* the connection rather than read off its outcome, because the
/// outcome cannot carry it: a severed link ends in `Err` whether the socket had been up
/// for two hours or two milliseconds.
#[derive(Default)]
struct ConnectionHealth {
    /// When the socket was established; `None` if it never was.
    ///
    /// Timed from the *established* socket rather than the start of the attempt, so a
    /// venue that takes twenty seconds to accept and then drops us cannot bank those
    /// twenty seconds as evidence of a healthy link.
    opened: Option<Instant>,
    /// Whether the venue ever sent a frame — of any kind — on this connection.
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
/// Identical in shape to the Hyperliquid adapter's, and that is a conclusion rather
/// than a copy — the two venues' close semantics differ enough that it had to be
/// re-measured. What the old loop here did was reset the wait only when [`run_once`]
/// returned `Ok(())` and double it only on `Err`, with **no sleep at all on the `Ok`
/// path**. Three separate measurements say why that is wrong, and the third is the one
/// that differs from Hyperliquid:
///
/// 1. **A severed link never reaches `Ok`.** That is a property of `tungstenite`, not
///    of a venue: a FIN with no Close frame surfaces as `ResetWithoutClosingHandshake`
///    and an RST as an IO error, both `Err`. Proven against *this* client over loopback
///    by `a_severed_socket_ends_in_err_so_a_reset_keyed_on_ok_can_never_fire`, so the
///    old reset branch was unreachable from any real network event and the wait only
///    ever climbed. The Hyperliquid soak measured what that costs: a 0.4 s outage
///    producing a 30.0 s blackout, and 45.8 % of a session under `STALE MARKS`.
/// 2. **`Ok` *is* reachable here, unlike on Hyperliquid** — the venue closes every
///    connection at the 24 h mark, and a polite Close frame returns `Ok` (proven over
///    loopback by `a_polite_close_frame_ends_in_ok_which_this_venue_does_at_24h`). That
///    makes the old no-sleep reset an unthrottled connect storm on a venue whose
///    *normal* lifecycle includes a clean close. It does **not** justify a separate
///    `Ok` branch: a connection that lived 24 h clears [`HEALTHY`](Self::HEALTHY) on
///    duration alone, so the health-based rule already reconnects it promptly. Judging
///    the connection instead of its exit code is what makes one policy correct on both
///    venues.
/// 3. **`heard_from_venue` is load-bearing *here* in a way it is not on Hyperliquid.**
///    A stream name this venue does not recognise is not refused — the socket is
///    accepted and stays silent, with no error frame and no close (measured:
///    `/stream?streams=btcusdt@nosuchstream` held open for 6 s, zero frames). A
///    duration-only reset would therefore treat a connection subscribed to nothing as
///    evidence of health. Hyperliquid's argument for the same condition was the
///    opposite one — that it answers a subscribe with a snapshot in milliseconds, so
///    hearing a frame is too cheap to mean anything. Two venues, opposite reasoning,
///    same condition.
struct Backoff {
    wait: Duration,
}

impl Backoff {
    /// The floor: long enough not to be a retry storm, short enough to disappear beside
    /// a mark-staleness budget measured in seconds.
    const MIN: Duration = Duration::from_millis(250);
    /// The ceiling on a single wait.
    const MAX: Duration = Duration::from_secs(30);
    /// How long a connection must survive before it counts as evidence of a healthy
    /// link.
    ///
    /// Equal to [`MAX`](Self::MAX), and the equality *is* the argument: whatever the
    /// venue does, it bounds us to about one connection attempt per `MAX`. A connection
    /// that dies sooner never resets the wait, so the wait keeps doubling to the cap; a
    /// connection that outlives the cap has already spent longer up than the longest
    /// wait we would have imposed, so resetting after it cannot make the attempt rate
    /// worse. It also comfortably admits this venue's 24 h close, which is the only
    /// clean close it is documented to perform.
    const HEALTHY: Duration = Self::MAX;

    fn new() -> Self {
        Self { wait: Self::MIN }
    }

    /// Judge the connection that just ended — it was up for `lived`, and
    /// `heard_from_venue` says whether the venue ever sent a frame on it — and return
    /// how long to wait before the next attempt.
    ///
    /// Both conditions, not either. See the type's note: duration alone resets on a
    /// socket subscribed to a stream that does not exist, which on this venue is a
    /// silent, indefinitely open connection rather than an error.
    fn after(&mut self, lived: Duration, heard_from_venue: bool) -> Duration {
        if heard_from_venue && lived >= Self::HEALTHY {
            self.wait = Self::MIN;
        }
        let wait = self.wait;
        self.wait = (self.wait * 2).min(Self::MAX);
        wait
    }
}

/// Failure talking to Binance over WS/REST. Mapped to [`ProviderError`] at the port
/// boundary, so nothing above the adapter ever sees a Binance error code.
#[derive(Debug, thiserror::Error)]
pub enum BnError {
    #[error("websocket: {0}")]
    Ws(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("decode: {0}")]
    Decode(#[from] DecodeError),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("exchangeInfo: {0}")]
    ExchangeInfo(#[from] ExchangeInfoError),
    #[error("malformed response: {0}")]
    Malformed(String),
    /// Nothing was subscribed, so there is no URL to connect to.
    ///
    /// An error rather than an idle connection: Binance refuses `/stream` with an
    /// empty `streams` parameter, and a client that connected anyway would sit on a
    /// healthy socket producing nothing — which reads exactly like a quiet market.
    #[error("no feeds registered; nothing to subscribe to")]
    NoSubscriptions,
}

impl From<BnError> for ProviderError {
    fn from(e: BnError) -> Self {
        ProviderError::Network(e.to_string())
    }
}

/// Binance USD-M WS ingest adapter.
///
/// Construct it, register the feeds you want (via [`MarketData::subscribe`] or
/// [`Self::subscribe_symbol`]), then drive it with [`Self::run_forever`].
///
/// It is read-only in the strongest available sense: this crate contains no signed
/// request path at all, so there is nothing here that could place an order even by
/// mistake.
pub struct BinanceMarketData {
    base_url: String,
    symbols: SymbolTable,
    sender: EventSender,
    caps: Capabilities,
    /// Instruments a feed-wide `subscribe(feed)` expands over.
    instruments: Vec<SymbolId>,
    /// `(feed, symbol)` pairs, rebuilt into the connect URL on every attempt.
    desired: Mutex<Vec<(Feed, SymbolId)>>,
    book_depth: BookDepth,
    book_speed: UpdateSpeed,
}

impl BinanceMarketData {
    pub const MAINNET_WS: &'static str = "wss://fstream.binance.com";
    pub const TESTNET_WS: &'static str = "wss://fstream.binancefuture.com";

    pub fn new(
        base_url: impl Into<String>,
        symbols: SymbolTable,
        instruments: Vec<SymbolId>,
        sender: EventSender,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            symbols,
            sender,
            caps: crate::capabilities(),
            instruments,
            desired: Mutex::new(Vec::new()),
            // 20 levels at 100 ms: the deepest and fastest partial-depth stream the
            // venue offers. Depth costs nothing extra on the wire budget and the
            // shallower streams are a subset, so anything less is a book that is
            // missing levels a strategy might have wanted with no way to notice.
            book_depth: BookDepth::Levels20,
            book_speed: UpdateSpeed::Ms100,
        }
    }

    pub fn mainnet(symbols: SymbolTable, instruments: Vec<SymbolId>, sender: EventSender) -> Self {
        Self::new(Self::MAINNET_WS, symbols, instruments, sender)
    }

    /// Adapter pointed at the USD-M futures **testnet**.
    pub fn testnet(symbols: SymbolTable, instruments: Vec<SymbolId>, sender: EventSender) -> Self {
        Self::new(Self::TESTNET_WS, symbols, instruments, sender)
    }

    /// Choose the partial-depth stream's level count and cadence.
    ///
    /// A knob rather than a constant because the venue's three level counts are three
    /// different streams and the choice is a latency/bandwidth trade a caller owns.
    /// It never selects the *diff* stream: [`stream_name`] cannot produce that name.
    pub fn with_book_depth(mut self, depth: BookDepth, speed: UpdateSpeed) -> Self {
        self.book_depth = depth;
        self.book_speed = speed;
        self
    }

    /// Register interest in one `feed` for one instrument (idempotent).
    pub fn subscribe_symbol(&self, feed: Feed, symbol: SymbolId) {
        let mut d = self.desired.lock().expect("desired subs mutex");
        if !d.iter().any(|(f, s)| *f == feed && *s == symbol) {
            d.push((feed, symbol));
        }
    }

    fn desired_subs(&self) -> Vec<(Feed, SymbolId)> {
        self.desired.lock().expect("desired subs mutex").clone()
    }

    /// The stream names this adapter currently wants, lower-cased for the venue.
    ///
    /// A symbol that is not in the table is **skipped**, not guessed. There is no
    /// string to build a stream name from, and inventing one subscribes to something
    /// the venue will refuse — which arrives as an error frame that names a stream
    /// nobody in this process asked for.
    pub fn stream_names(&self) -> Vec<String> {
        self.desired_subs()
            .into_iter()
            .filter_map(|(feed, id)| {
                let symbol = self.symbols.stream_symbol(id)?;
                Some(stream_name(feed, &symbol, self.book_depth, self.book_speed))
            })
            .collect()
    }

    /// The combined-stream URL for the current subscription set.
    ///
    /// Always `/stream?streams=`, never `/ws/`. The raw endpoint sends bare payloads,
    /// and a bare `depthUpdate` cannot be told from an incremental diff — the decoder
    /// refuses one outright ([`DecodeError::Unenveloped`]) rather than trusting a URL
    /// nobody can see from inside the decode.
    pub fn connect_url(&self) -> Result<String, BnError> {
        let streams = self.stream_names();
        if streams.is_empty() {
            return Err(BnError::NoSubscriptions);
        }
        Ok(format!(
            "{}/stream?streams={}",
            self.base_url,
            streams.join("/")
        ))
    }

    /// One connection attempt: connect, then pump frames until the socket closes
    /// (`Ok`) or a transport error occurs (`Err`). Returns `Ok` early if the core
    /// dropped the bus receiver (shutdown).
    ///
    /// Which arm this returns through says much less about the connection than it
    /// looks like it does. `Ok` needs the venue to send a WebSocket **Close frame** or
    /// end the stream; no severed link does either — an RST surfaces as an IO error and
    /// a FIN with no Close as `ResetWithoutClosingHandshake`, both `Err`. On this venue
    /// `Ok` is nevertheless reachable, because it closes every connection at 24 hours.
    /// The connect loop therefore judges a connection on what it achieved — how long it
    /// stayed up, and whether the venue ever spoke on it — rather than on its exit code.
    pub async fn run_once(&self) -> Result<(), BnError> {
        self.run_once_recording(&mut ConnectionHealth::default())
            .await
    }

    /// [`Self::run_once`], recording what the connection achieved into `health`.
    ///
    /// Split out only because that record has to survive the `?` early-returns: the
    /// backoff needs it precisely on the paths that end in an error, which in practice
    /// is all of them.
    async fn run_once_recording(&self, health: &mut ConnectionHealth) -> Result<(), BnError> {
        let url = self.connect_url()?;
        let (ws, _resp) = connect_async(&url).await?;
        health.opened_now();
        let (mut write, mut read) = ws.split();

        // Binance pings every three minutes and closes a connection whose pong does not
        // arrive within ten. An unsolicited pong is accepted as a keepalive, so this
        // sends one well inside the window rather than waiting to be asked: a client
        // that only ever answers is one missed server ping away from a disconnect it
        // could have prevented.
        let mut heartbeat = tokio::time::interval(Duration::from_secs(150));
        heartbeat.tick().await; // the first tick is immediate — skip it

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    write.send(Message::Pong(Vec::new().into())).await?;
                }
                item = read.next() => {
                    let Some(msg) = item else { return Ok(()) }; // stream ended
                    let msg = msg?;
                    // The venue spoke on this socket. That our *connect* succeeded
                    // proves nothing about the subscription — this venue accepts a
                    // stream name it does not recognise and then says nothing at all,
                    // with no error frame and no close — so this is the only evidence
                    // the link carries traffic in the direction we need, and the
                    // backoff will not treat a connection without it as healthy.
                    //
                    // Every frame counts, including the server's own Ping. A Ping is
                    // genuine evidence the far end is alive and talking, which is the
                    // question the backoff is asking; whether the *subscription* is
                    // right is a data-staleness question answered above the adapter.
                    // It matters here because this venue pings on its own schedule,
                    // where Hyperliquid expects the client to.
                    health.heard_from_venue = true;
                    match msg {
                        Message::Text(txt) => {
                            // Stamped before parsing, not after, so decode cost does not
                            // leak into the timestamp. Only the mark-price feed reads it,
                            // and there it is the ingest half of a `Ticker` whose venue
                            // time is genuinely present — unlike Hyperliquid, where the
                            // same field is the entire ordering key.
                            let ts_ingest = SystemClock.now_ns();
                            match venue_error(txt.as_str()) {
                                // A refused subscription is an ordinary frame on a healthy
                                // socket. Log it and keep the connection: tearing down would
                                // reconnect, resend the same bad stream name forever, and
                                // drop every good feed on each pass.
                                Some(msg) => eprintln!("binance WS error frame: {msg}"),
                                None => match decode_ws_message(txt.as_str(), &self.symbols, ts_ingest) {
                                    Ok(events) => {
                                        for ev in events {
                                            // A dropped receiver means the core is shutting down.
                                            if self.sender.send(ev).is_err() {
                                                health.bus_closed = true;
                                                return Ok(());
                                            }
                                        }
                                    }
                                    // A decode failure is a *data* problem, not a transport
                                    // one, and must not close the socket — a frame the
                                    // venue will resend would otherwise loop us forever
                                    // while every healthy feed stays dark. Both deliberate
                                    // refusals land here and are meant to be loud:
                                    // `UnsupportedStream` (an incremental depth stream that
                                    // would be read as a snapshot) and `Unenveloped` (the
                                    // raw endpoint, where that distinction is invisible).
                                    Err(e) => eprintln!("binance WS decode error (frame dropped): {e}"),
                                },
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
    /// exponential backoff. Returns only when the core drops the bus receiver, or when
    /// there is nothing subscribed to connect for.
    ///
    /// Resubscription is not a separate step here — the stream list is in the URL, so
    /// reconnecting *is* resubscribing and there is no path on which it can be
    /// forgotten. This venue also closes every connection after 24 hours regardless of
    /// health, so the loop is load-bearing on a long session rather than only on a
    /// fault, and it is the reason the `Ok` arm below is a real case here and a dead
    /// one on Hyperliquid.
    ///
    /// **Every ended connection is judged the same way and waited out the same way,
    /// whichever arm it returned through.** The old loop treated `Ok` and `Err` as two
    /// policies — reset-and-retry-instantly against double-the-wait — and both halves
    /// were wrong for one reason: the *outcome* of a connection says nothing about
    /// whether it was worth having. A severed link is always `Err` however long it had
    /// been healthy, and this venue's routine 24 h close is always `Ok`, which the old
    /// loop answered with an unthrottled reconnect.
    pub async fn run_forever(&self) {
        let mut backoff = Backoff::new();
        loop {
            let mut health = ConnectionHealth::default();
            let outcome = self.run_once_recording(&mut health).await;
            if health.bus_closed {
                // The core is shutting down. Reconnecting into a closed bus would spin a
                // connect loop against the venue for as long as the process takes to
                // exit — every connection succeeding, delivering one frame, and dying on
                // the same closed channel. Under the old loop that spin had no sleep in
                // it at all.
                return;
            }
            if let Err(e @ BnError::NoSubscriptions) = &outcome {
                // Not a network event, and no amount of waiting adds a subscription.
                // Judged before the backoff so an empty set is never waited out.
                eprintln!("binance WS not started: {e}");
                return;
            }
            let lived = health.lived();
            let wait = backoff.after(lived, health.heard_from_venue);
            match outcome {
                // Logged even though it is not an error: a socket that closed and came
                // back with no line in the log is indistinguishable, afterwards, from
                // one that never closed. On this venue it is also the expected shape of
                // a 24 h rollover, which an operator should be able to see happen.
                Ok(()) => {
                    eprintln!("binance WS closed after {lived:.1?}; reconnecting in {wait:?}")
                }
                // A connect that never established is a different event from a
                // connection that died, and sharing one line makes it read as nonsense —
                // "error after 0.0ns" — on exactly the retries an operator is scanning to
                // tell an outage from a bad endpoint.
                Err(e) if health.opened.is_none() => {
                    eprintln!("binance WS connect failed: {e}; retrying in {wait:?}")
                }
                Err(e) => {
                    eprintln!("binance WS error after {lived:.1?}: {e}; reconnecting in {wait:?}")
                }
            }
            tokio::time::sleep(wait).await;
        }
    }
}

#[async_trait]
impl MarketData for BinanceMarketData {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    /// Register `feed` for every configured instrument. Takes effect on the next
    /// connect, exactly as the Hyperliquid adapter's does.
    async fn subscribe(&self, feed: Feed) -> Result<(), ProviderError> {
        for id in self.instruments.clone() {
            self.subscribe_symbol(feed, id);
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
    use axon_providers::CandleInterval;
    use tokio::net::TcpListener;

    fn md() -> (BinanceMarketData, axon_core::EventReceiver) {
        let (tx, rx) = axon_core::bus(16);
        let symbols = SymbolTable::from_ordered(["BTCUSDT", "ETHUSDT"]);
        let ids = vec![SymbolId::new(0), SymbolId::new(1)];
        (BinanceMarketData::testnet(symbols, ids, tx), rx)
    }

    // ── loopback: how a connection *ends*, measured rather than assumed ──────
    //
    // These stand a WebSocket server on 127.0.0.1 and drive the real `run_once`
    // against it. Nothing leaves the box and no venue is contacted, which is the
    // point: the claim being checked — "no severed link returns `Ok`" — is a
    // property of `tungstenite`, so it can and should be settled locally instead
    // of inferred from a venue's documentation.

    /// How the loopback server ends the connection.
    #[derive(Clone, Copy)]
    enum Ending {
        /// Drop the TCP socket with no closing handshake — a FIN with no Close
        /// frame, which is what a severed link looks like from this side.
        Abrupt,
        /// Send a WebSocket Close frame first, the way this venue is documented
        /// to at the 24 h mark.
        PoliteClose,
    }

    /// Serve one connection: accept, optionally send `frames`, then end it.
    /// Returns the bound address.
    async fn serve_once(frames: Vec<String>, ending: Ending) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr").to_string();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let mut ws = tokio_tungstenite::accept_async(stream)
                .await
                .expect("ws handshake");
            for f in frames {
                ws.send(Message::text(f)).await.expect("send frame");
            }
            match ending {
                // Dropping the stream closes the TCP socket without a handshake.
                Ending::Abrupt => drop(ws),
                Ending::PoliteClose => {
                    let _ = ws.send(Message::Close(None)).await;
                    let _ = ws.flush().await;
                    // Hold briefly so the Close is read before the socket dies.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        });
        addr
    }

    /// A client pointed at a loopback server, subscribed to one real feed.
    fn loopback(addr: &str) -> (BinanceMarketData, axon_core::EventReceiver) {
        let (tx, rx) = axon_core::bus(64);
        let md = BinanceMarketData::new(
            format!("ws://{addr}"),
            SymbolTable::from_ordered(["BTCUSDT"]),
            vec![SymbolId::new(0)],
            tx,
        );
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(0));
        (md, rx)
    }

    /// One real captured frame, so the server sends something a decoder accepts.
    const BOOK_TICKER_FRAME: &str = r#"{"stream":"btcusdt@bookTicker","data":{"e":"bookTicker","u":359220650272,"s":"BTCUSDT","ps":"BTCUSDT","b":"64336.00","B":"73.3346","a":"64346.70","A":"589.8026","T":1785048483771,"E":1785048483771,"st":1}}"#;

    #[tokio::test]
    async fn a_severed_socket_ends_in_err_so_a_reset_keyed_on_ok_can_never_fire() {
        // **The defect, measured.** The old loop reset its backoff only on `Ok(())`.
        // This is a socket dying the way a severed link dies — a FIN with no closing
        // handshake — and it comes back `Err`. So the reset branch was unreachable
        // from any real network event and the wait only ever doubled, which is what
        // turned a 0.4 s outage into a 30 s blackout on the other adapter.
        let addr = serve_once(vec![BOOK_TICKER_FRAME.to_string()], Ending::Abrupt).await;
        let (md, _rx) = loopback(&addr);
        let mut health = ConnectionHealth::default();
        let out = md.run_once_recording(&mut health).await;
        // Named, not merely `is_err()`: the whole argument is that this specific
        // shape — the transport noticing the peer vanished — lands on the error arm.
        // A test that only asserted "some error" would still pass if the client had
        // failed the handshake instead, which proves nothing about a severed link.
        assert!(
            matches!(
                out,
                Err(BnError::Ws(
                    tokio_tungstenite::tungstenite::Error::Protocol(
                        tokio_tungstenite::tungstenite::error::ProtocolError::ResetWithoutClosingHandshake
                    )
                ))
            ),
            "a severed link must not look like a clean close: {out:?}"
        );
        assert!(health.opened.is_some(), "the socket did establish");
        assert!(
            health.heard_from_venue,
            "and the venue spoke on it before it died"
        );
    }

    #[tokio::test]
    async fn a_polite_close_frame_ends_in_ok_which_this_venue_does_at_24h() {
        // The other half, and the reason this adapter's `Ok` path is *not* dead code
        // the way Hyperliquid's is: this venue closes every connection at the 24 h
        // mark. Under the old loop that was a reset with no sleep — an unthrottled
        // reconnect on a venue whose normal lifecycle includes a clean close.
        let addr = serve_once(vec![BOOK_TICKER_FRAME.to_string()], Ending::PoliteClose).await;
        let (md, _rx) = loopback(&addr);
        let mut health = ConnectionHealth::default();
        let out = md.run_once_recording(&mut health).await;
        assert!(out.is_ok(), "a Close frame is a clean close: {out:?}");
        assert!(health.heard_from_venue);
    }

    #[tokio::test]
    async fn a_connection_that_never_opened_is_told_apart_from_one_that_died() {
        // Different events, different operator response — a bad endpoint versus an
        // outage — and `lived` must not report a duration for a socket that never
        // existed, or the backoff would judge a failed connect as a zero-length
        // healthy one.
        let (tx, _rx) = axon_core::bus(16);
        let md = BinanceMarketData::new(
            "ws://127.0.0.1:1",
            SymbolTable::from_ordered(["BTCUSDT"]),
            vec![SymbolId::new(0)],
            tx,
        );
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(0));
        let mut health = ConnectionHealth::default();
        assert!(md.run_once_recording(&mut health).await.is_err());
        assert!(health.opened.is_none(), "it never established");
        assert_eq!(health.lived(), Duration::ZERO);
        assert!(!health.heard_from_venue);
    }

    #[tokio::test]
    async fn a_connection_nobody_answered_on_never_counts_as_having_heard_the_venue() {
        // This venue accepts a stream name it does not recognise and then says
        // nothing — no error frame, no close (measured against the testnet endpoint:
        // `/stream?streams=btcusdt@nosuchstream` held open with zero frames). A
        // duration-only health test would read that as a healthy link. The loopback
        // stand-in is a server that accepts and sends nothing at all.
        let addr = serve_once(Vec::new(), Ending::Abrupt).await;
        let (md, _rx) = loopback(&addr);
        let mut health = ConnectionHealth::default();
        let _ = md.run_once_recording(&mut health).await;
        assert!(health.opened.is_some(), "the socket established");
        assert!(
            !health.heard_from_venue,
            "and the venue never spoke, which is the whole distinction"
        );
        // So even a long-lived one of these must not reset the wait.
        let mut b = Backoff::new();
        b.after(Duration::ZERO, false);
        assert_eq!(
            b.after(Duration::from_secs(3600), false),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn subscribe_symbol_is_idempotent() {
        let (md, _rx) = md();
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(0));
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(0)); // duplicate ignored
        md.subscribe_symbol(Feed::L2Book, SymbolId::new(0));
        assert_eq!(md.desired_subs().len(), 2);
    }

    #[tokio::test]
    async fn a_coinless_subscribe_expands_to_every_configured_instrument() {
        let (md, _rx) = md();
        md.subscribe(Feed::Trades).await.unwrap();
        assert_eq!(md.desired_subs().len(), 2);
        assert_eq!(
            md.stream_names(),
            vec!["btcusdt@aggTrade", "ethusdt@aggTrade"]
        );
        md.unsubscribe(Feed::Trades).await.unwrap();
        assert!(md.desired_subs().is_empty());
    }

    #[tokio::test]
    async fn the_connect_url_is_the_combined_endpoint_and_never_the_raw_one() {
        // The raw endpoint delivers bare payloads, and a bare `depthUpdate` is
        // indistinguishable from an incremental diff. Connecting there would produce a
        // book that is wrong and complete-looking; the decoder refuses an unenveloped
        // frame for that reason, and this is the other half of the same guarantee.
        let (md, _rx) = md();
        md.subscribe(Feed::L2Book).await.unwrap();
        md.subscribe(Feed::Ticker).await.unwrap();
        let url = md.connect_url().unwrap();
        assert!(url.starts_with("wss://fstream.binancefuture.com/stream?streams="));
        assert!(!url.contains("/ws/"), "{url}");
        assert!(url.contains("btcusdt@depth20@100ms"));
        assert!(url.contains("btcusdt@markPrice@1s"));
        assert!(url.contains("ethusdt@depth20@100ms"));
    }

    // ── the backoff policy ───────────────────────────────────────────────────

    /// A connection long enough to be evidence of health under any threshold.
    const HEALTHY_RUN: Duration = Duration::from_secs(120);

    /// A backoff already driven to its ceiling by repeated short failures — the state
    /// the old loop reached after ~8 disconnects and never left.
    fn pinned_at_the_cap() -> Backoff {
        let mut b = Backoff::new();
        for _ in 0..10 {
            b.after(Duration::ZERO, false);
        }
        assert_eq!(b.after(Duration::ZERO, false), Backoff::MAX);
        b
    }

    #[test]
    fn a_healthy_connection_resets_the_wait_so_one_bad_hour_does_not_cost_the_session() {
        // The defect's actual cost: on the other adapter the wait pinned at 30 s and
        // stayed there, so a 0.4 s outage late in a session produced a 30.0 s blackout
        // where the session's first 0.4 s outage had produced 0.4 s. Recovery has to be
        // earned by a connection that worked, not by the exit code it happened to
        // return through.
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(HEALTHY_RUN, true), Backoff::MIN);
        assert_eq!(
            b.after(Duration::ZERO, false),
            Backoff::MIN * 2,
            "and it climbs again from the floor"
        );
    }

    #[test]
    fn a_twenty_four_hour_close_resets_without_needing_an_ok_branch_of_its_own() {
        // The Binance-specific case, and the reason re-measuring did not change the
        // policy. This venue closes every connection at 24 h; that connection clears
        // `HEALTHY` on duration alone, so the health-based rule reconnects it promptly
        // with no special case for a clean close. Judging the connection rather than
        // its exit code is what makes one policy right on a venue where `Ok` is
        // routine and on one where it is unreachable.
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(Duration::from_secs(24 * 3600), true), Backoff::MIN);
    }

    #[test]
    fn a_connection_that_dies_young_never_resets_however_often_it_reconnects() {
        // The abuse a backoff exists to prevent: a venue that accepts a connection and
        // drops it a second later would be hammered at the floor forever if "the socket
        // opened" or "a frame arrived" were enough to count as healthy.
        const SHORT_RUN: Duration = Duration::from_secs(5);
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(SHORT_RUN, true), Backoff::MAX);
        assert_eq!(b.after(SHORT_RUN, true), Backoff::MAX);
    }

    #[test]
    fn a_long_silent_connection_never_resets_because_this_venue_accepts_bad_streams_quietly() {
        // Measured against the venue: `/stream?streams=btcusdt@nosuchstream` is
        // accepted and stays open with zero frames — no error frame, no close. That is
        // a connection which is long-lived and useless, and duration alone would read
        // it as proof the endpoint is healthy.
        let mut b = pinned_at_the_cap();
        assert_eq!(b.after(Duration::from_secs(3600), false), Backoff::MAX);
    }

    #[test]
    fn the_wait_is_bounded_at_both_ends_for_every_sequence_of_connection_outcomes() {
        // The property that makes the whole thing safe to leave running: whatever the
        // venue does, we never retry faster than `MIN` and never wait longer than
        // `MAX`, so the attempt rate is bounded at roughly one connect per `MAX` in the
        // worst case and one per `MIN` in the best.
        let mut b = Backoff::new();
        for (lived, heard) in [
            (HEALTHY_RUN, true),
            (Duration::ZERO, false),
            (Duration::from_secs(5), true),
            (HEALTHY_RUN, false),
            (Duration::from_secs(24 * 3600), true),
            (Duration::ZERO, false),
        ] {
            let w = b.after(lived, heard);
            assert!((Backoff::MIN..=Backoff::MAX).contains(&w), "{w:?}");
        }
        assert_eq!(
            Backoff::HEALTHY,
            Backoff::MAX,
            "the equality is the safety argument, not a coincidence"
        );
    }

    #[tokio::test]
    async fn a_client_with_nothing_subscribed_refuses_to_connect_rather_than_going_quiet() {
        // Binance refuses `/stream?streams=` with an empty list, and a client that
        // connected anyway would hold a healthy socket that produces nothing — which in
        // a log is indistinguishable from a quiet market. `run_forever` returns instead
        // of spinning, because reconnecting cannot fix an empty subscription set.
        let (md, _rx) = md();
        assert!(matches!(md.connect_url(), Err(BnError::NoSubscriptions)));
        md.run_forever().await; // returns immediately rather than looping
    }

    #[tokio::test]
    async fn the_book_depth_knob_can_never_select_the_incremental_stream() {
        // The one configuration mistake that would be silent. `stream_name` cannot
        // produce `@depth@100ms`, so no combination of these knobs reaches the diff
        // stream — the decoder's refusal is a backstop, not the only guard.
        for depth in [BookDepth::Levels5, BookDepth::Levels10, BookDepth::Levels20] {
            let (tx, _rx) = axon_core::bus(16);
            let md = BinanceMarketData::testnet(
                SymbolTable::from_ordered(["BTCUSDT"]),
                vec![SymbolId::new(0)],
                tx,
            )
            .with_book_depth(depth, UpdateSpeed::Ms500);
            md.subscribe(Feed::L2Book).await.unwrap();
            let names = md.stream_names();
            assert_eq!(names.len(), 1);
            assert!(
                names[0].starts_with(&format!("btcusdt@depth{}", depth.levels())),
                "got {}",
                names[0]
            );
            assert!(!names[0].starts_with("btcusdt@depth@"));
        }
    }

    #[tokio::test]
    async fn a_symbol_with_no_entry_in_the_table_is_skipped_rather_than_guessed() {
        // There is no string to build a stream name from, and inventing one subscribes
        // to something the venue refuses — which comes back as an error frame naming a
        // stream nobody in this process asked for.
        let (md, _rx) = md();
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(0));
        md.subscribe_symbol(Feed::Bbo, SymbolId::new(99));
        assert_eq!(md.stream_names(), vec!["btcusdt@bookTicker"]);
    }

    #[tokio::test]
    async fn every_feed_the_port_defines_reaches_the_url_without_a_special_case() {
        // The market-data half of the Layer claim: five normalized feeds, one loop, no
        // per-feed arm anywhere in the client. A feed that needed one would be a feed
        // that could be forgotten on reconnect — and on this venue the subscription
        // lives in the URL, so forgetting it is forgetting the instrument entirely.
        let (md, _rx) = md();
        for feed in [
            Feed::L2Book,
            Feed::Trades,
            Feed::Bbo,
            Feed::Candles(CandleInterval::M1),
            Feed::Ticker,
        ] {
            md.subscribe(feed).await.unwrap();
        }
        let names = md.stream_names();
        assert_eq!(names.len(), 10, "five feeds x two instruments");
        let url = md.connect_url().unwrap();
        for n in &names {
            assert!(url.contains(n.as_str()), "{n} missing from {url}");
        }
    }

    /// Live smoke test — hits the real Binance USD-M **testnet** WS. Ignored by
    /// default so the gate stays deterministic and offline, and **it has never been
    /// run**: nothing in this crate has been observed against the venue.
    ///
    /// Run with `cargo test -p axon-provider-binance -- --ignored`. It subscribes and
    /// reads; there is no credential anywhere in this crate and no code path that
    /// could place an order.
    #[tokio::test]
    #[ignore = "hits the live Binance USD-M futures testnet WS"]
    async fn live_market_data_smoke() {
        let (tx, rx) = axon_core::bus(4096);
        let symbols = SymbolTable::from_ordered(["BTCUSDT"]);
        let md = BinanceMarketData::testnet(symbols, vec![SymbolId::new(0)], tx);
        md.subscribe(Feed::Bbo).await.unwrap();
        md.subscribe(Feed::Trades).await.unwrap();
        md.subscribe(Feed::L2Book).await.unwrap();
        md.subscribe(Feed::Ticker).await.unwrap();

        let stream = tokio::spawn(async move {
            let _ = md.run_once().await;
        });
        tokio::time::sleep(Duration::from_secs(8)).await;
        stream.abort();

        let mut saw_ticker = false;
        let mut n = 0;
        while let Some(ev) = rx.try_recv() {
            n += 1;
            if let axon_core::Event::Market(axon_core::MarketEvent::Ticker(t)) = &ev {
                // The claim that only a live run can settle: this venue stamps its
                // ticker, so the one feed Hyperliquid cannot replay deterministically is
                // replayable here.
                assert!(t.is_venue_timed(), "markPriceUpdate must carry E");
                saw_ticker = true;
            }
        }
        assert!(n > 0, "expected at least one live market event within 8s");
        assert!(saw_ticker, "expected at least one markPriceUpdate in 8s");
    }
}
