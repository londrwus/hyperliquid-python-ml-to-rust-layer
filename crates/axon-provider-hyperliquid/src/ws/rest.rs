//! REST snapshot seeding via `POST /info`.
//!
//! Some feeds send no initial snapshot, so a book is seeded from REST `l2Book`
//! and then WS updates are applied on top (`docs/research/hyperliquid-execution.md`).
//! Hyperliquid's `l2Book` WS feed *does* send full snapshots, so this is optional
//! for it, but the seam is here for feeds that need it. The response decode is the
//! same pure, tested [`decode_rest_l2`] used elsewhere; only the fetch is live.
//!
//! Every fetch here is bounded by [`INFO_TIMEOUT`]. `reqwest` waits forever by default,
//! and one of these calls runs alongside a live session — see [`post_info`].

use std::time::Duration;

use axon_core::MarketEvent;

use super::client::HlError;
use super::decode::{decode_rest_l2, decode_universe, Universe};
use super::funding::{decode_funding_cadence, FundingCadence};
use crate::symbol_map::SymbolMap;

pub const MAINNET_INFO: &str = "https://api.hyperliquid.xyz/info";
pub const TESTNET_INFO: &str = "https://api.hyperliquid-testnet.xyz/info";

/// How long any one `POST /info` may take, end to end.
///
/// `reqwest` has **no** default timeout, and the failure that matters here is not a
/// refused connection — that returns an error the caller can log and move past. It is
/// the endpoint that completes the TCP/TLS handshake and then answers nothing: a
/// partial venue outage, a load balancer blackholing a backend, an inline proxy. Against
/// that, an unbounded `send().await` never returns, and whatever the caller was going to
/// do next never happens. `verify_funding_cadence` is the caller that makes this fatal —
/// it is a once-per-session *diagnostic*, and a diagnostic must never be able to outlive
/// the market-data socket it sits next to.
///
/// Ten seconds is far beyond any healthy `/info` round trip (tens of ms) and comfortably
/// inside the WS reconnect ceiling, so a bounded failure looks like one slow probe
/// rather than a stalled session.
pub const INFO_TIMEOUT: Duration = Duration::from_secs(10);

/// One `POST /info` round trip, bounded end to end, returning the raw body.
///
/// Every `/info` fetch in this module goes through here so the bound cannot be forgotten
/// by the next one added. `timeout` covers connect *and* response — `connect_timeout`
/// alone would leave the blackhole case unbounded, which is the case that hangs.
/// The client is built per call, as it always was: these are once-per-session requests
/// and a pooled client would only hide which of them is slow.
async fn post_info(
    info_url: &str,
    body: &serde_json::Value,
    timeout: Duration,
) -> Result<String, HlError> {
    let text = reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .build()?
        .post(info_url)
        .json(body)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    Ok(text)
}

/// Fetch `meta` and build **both** halves of the perp universe: the [`SymbolMap`]
/// (real asset indices) and the [`InstrumentTable`](axon_providers::InstrumentTable)
/// (each instrument's tick and lot).
///
/// One request, one decode. A session that fetched the two separately could hold an
/// asset index from before a listing and a lot from after it, and an order signed
/// across that seam trades a different coin at a size computed for this one.
pub async fn fetch_universe(info_url: &str) -> Result<Universe, HlError> {
    let body = serde_json::json!({ "type": "meta" });
    let text = post_info(info_url, &body, INFO_TIMEOUT).await?;
    Ok(decode_universe(&text)?)
}

/// The symbol half alone, for the callers that only address instruments.
pub async fn fetch_meta(info_url: &str) -> Result<SymbolMap, HlError> {
    Ok(fetch_universe(info_url).await?.symbols)
}

/// Fetch an `l2Book` snapshot for `coin` and decode it to a [`MarketEvent::Book`].
/// `info_url` is [`MAINNET_INFO`] or [`TESTNET_INFO`].
pub async fn fetch_l2_snapshot(
    info_url: &str,
    coin: &str,
    symbols: &SymbolMap,
) -> Result<MarketEvent, HlError> {
    let body = serde_json::json!({ "type": "l2Book", "coin": coin });
    let text = post_info(info_url, &body, INFO_TIMEOUT).await?;
    Ok(decode_rest_l2(&text, symbols)?)
}

/// Measure the venue's funding period for `coin` over `[start_ms, now]`.
///
/// The funding *interval* is the one thing `activeAssetCtx` needs and never carries,
/// so the adapter stamps a constant onto every [`Ticker`](axon_core::Ticker); this is
/// the endpoint that can contradict it. See
/// [`funding`](super::funding) for why the measurement comes from market-wide history
/// rather than from the account's own funding charges.
///
/// Bounded by [`INFO_TIMEOUT`], and that bound is load-bearing rather than tidy: this is
/// the one `/info` call a *running* session makes, and it runs beside the WS connect
/// loop. A hang here with no bound is a session whose market-data socket never comes up.
pub async fn fetch_funding_cadence(
    info_url: &str,
    coin: &str,
    start_ms: i64,
) -> Result<FundingCadence, HlError> {
    let body = serde_json::json!({ "type": "fundingHistory", "coin": coin, "startTime": start_ms });
    let text = post_info(info_url, &body, INFO_TIMEOUT).await?;
    Ok(decode_funding_cadence(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_blackholed_info_endpoint_gives_up_instead_of_hanging_the_session() {
        // The failure a refused connection does *not* cover: the peer completes the
        // handshake and then never answers. An unaccepted listener does exactly that —
        // the kernel finishes the TCP handshake from the backlog, so the POST is sent
        // and no response ever comes. Loopback only; nothing here leaves the box.
        //
        // Unbounded, `send().await` never returns, and the caller
        // (`verify_funding_cadence`) never returns either. When that caller was awaited
        // ahead of the WS connect, the whole session sat behind this: no events, no
        // reconnect, no log line, and a status line that still printed OK. Bounded, the
        // worst case is one skipped diagnostic.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let url = format!("http://{addr}/info");

        let started = std::time::Instant::now();
        let out = post_info(
            &url,
            &serde_json::json!({ "type": "fundingHistory", "coin": "BTC" }),
            Duration::from_millis(250),
        )
        .await;
        let elapsed = started.elapsed();

        // Specifically a *timeout*, not any old error. A connect failure would prove
        // nothing about the hang this test exists for — the peer here is reachable and
        // silent, which is the whole point.
        assert!(
            matches!(out, Err(HlError::Http(ref e)) if e.is_timeout()),
            "expected a timeout from a silent peer, got {out:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the request outlived its 250 ms budget by {elapsed:?} — it is not bounded"
        );
        // The shipped bound has to be small enough that a stalled probe is invisible
        // next to the WS reconnect backoff ceiling (30 s in `client.rs`); a bound of
        // "eventually" is the same bug with extra steps.
        assert!(
            INFO_TIMEOUT <= Duration::from_secs(30),
            "INFO_TIMEOUT must stay inside the reconnect ceiling"
        );
    }
}
