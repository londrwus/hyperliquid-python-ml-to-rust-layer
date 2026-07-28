//! The public REST reads: `exchangeInfo`, `fundingInfo`, `depth`, `openInterest`.
//!
//! All four are unauthenticated `GET`s. Nothing in this module signs anything, sends
//! anything, or can reach `/fapi/v1/order` — see [`crate::sign`] for what is
//! deliberately absent and why.
//!
//! Every request is bounded by [`REST_TIMEOUT`]. `reqwest` has **no** default request
//! timeout, and the failure that matters is not a refused connection — that returns an
//! error a caller can log and move past. It is the endpoint that completes the TCP and
//! TLS handshake and then answers nothing: a partial outage, a load balancer
//! blackholing a backend, an inline proxy. Against that an unbounded `send().await`
//! never returns and whatever the caller was going to do next never happens. The
//! Hyperliquid adapter paid for that lesson in a session that printed `OK` while its
//! market-data socket never came up; this one starts with the bound.

use std::time::Duration;

use axon_core::{Decimal, MarketEvent, Nanos, SymbolId};
use std::str::FromStr;

use super::client::BnError;
use super::decode::decode_rest_depth;
use crate::exchange_info::{decode_exchange_info, decode_funding_info, Universe};
use crate::MS_TO_NS;

/// Binance USD-M futures, production.
pub const MAINNET_REST: &str = "https://fapi.binance.com";

/// Binance USD-M futures **testnet** — the only Binance these fixtures could be
/// captured from, because production answers HTTP 451 from this host.
pub const TESTNET_REST: &str = "https://testnet.binancefuture.com";

/// How long any one REST read may take, end to end. See the module note.
pub const REST_TIMEOUT: Duration = Duration::from_secs(10);

/// One bounded `GET`, returning the raw body.
///
/// Every read in this module goes through here so the bound cannot be forgotten by
/// the next one added. `timeout` covers connect *and* response; `connect_timeout`
/// alone would leave the blackhole case unbounded, which is the case that hangs.
async fn get(url: &str, timeout: Duration) -> Result<String, BnError> {
    let text = reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .build()?
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    Ok(text)
}

/// Fetch `exchangeInfo` and build the [`Universe`] — ids, grids and the symbols this
/// adapter declines, with a reason each.
pub async fn fetch_exchange_info(base_url: &str) -> Result<Universe, BnError> {
    let body = get(&format!("{base_url}/fapi/v1/exchangeInfo"), REST_TIMEOUT).await?;
    Ok(decode_exchange_info(&body)?)
}

/// Fetch `exchangeInfo` **and** `fundingInfo`, so every symbol's funding period is
/// the venue's own where the venue publishes one.
///
/// Two requests, and they are two failures on purpose. A `fundingInfo` that times out
/// leaves a working universe behind on the documented eight-hour default rather than
/// taking the instrument table down with it — the interval is the one field where an
/// assumption is survivable, because it scales carry rather than invalidating a
/// price. What it costs is the seam ADR-0025 closed for Hyperliquid by insisting on a
/// single `meta` read: these two can disagree across a listing. Here the venue offers
/// no single endpoint that carries both.
///
/// A failed funding fetch is reported on stderr and swallowed, in the shape the
/// Hyperliquid cadence probe already established: a diagnostic that did not run is
/// not a reason to refuse market data, and a check that only speaks when it fails is
/// indistinguishable in a log from a check that never ran.
pub async fn fetch_universe(base_url: &str) -> Result<Universe, BnError> {
    let mut universe = fetch_exchange_info(base_url).await?;
    match get(&format!("{base_url}/fapi/v1/fundingInfo"), REST_TIMEOUT).await {
        Ok(body) => match decode_funding_info(&body, &mut universe.symbols) {
            Ok(applied) => eprintln!(
                "binance funding intervals: {applied} of {} symbols published, the rest assumed at \
                 {}h",
                universe.symbols.len(),
                crate::DEFAULT_FUNDING_INTERVAL_HOURS
            ),
            Err(e) => {
                eprintln!("binance fundingInfo undecodable ({e}); assuming 8h for every symbol")
            }
        },
        Err(e) => eprintln!("binance fundingInfo unavailable ({e}); assuming 8h for every symbol"),
    }
    Ok(universe)
}

/// Fetch a book snapshot for `symbol` and decode it against `symbol_id`.
///
/// `limit` must be one the venue offers (5, 10, 20, 50, 100, 500, 1000); anything
/// else is refused with `-1100`. The id is an argument because the response does not
/// echo the symbol — see [`decode_rest_depth`].
pub async fn fetch_depth_snapshot(
    base_url: &str,
    symbol: &str,
    limit: u32,
    symbol_id: SymbolId,
) -> Result<MarketEvent, BnError> {
    let url = format!("{base_url}/fapi/v1/depth?symbol={symbol}&limit={limit}");
    let body = get(&url, REST_TIMEOUT).await?;
    Ok(decode_rest_depth(&body, symbol_id)?)
}

/// Fetch current open interest for `symbol`, in **base** units, with the venue's own
/// timestamp.
///
/// Returned as `(value, ts)` rather than a bare number for ADR-0011 §6's reason: a
/// *stale* reference is more dangerous than a missing one, and handing out the value
/// alone lets every call site drop the age on the floor. There is no stream for this
/// number — it is one weight per symbol per call — which is why
/// [`Ticker::open_interest`](axon_core::Ticker) is `None` on every event this adapter
/// produces rather than being back-filled from a poll and stamped with a frame's time.
pub async fn fetch_open_interest(
    base_url: &str,
    symbol: &str,
) -> Result<(Decimal, Nanos), BnError> {
    #[derive(serde::Deserialize)]
    struct OpenInterest {
        #[serde(rename = "openInterest")]
        open_interest: String,
        time: i64,
    }
    let url = format!("{base_url}/fapi/v1/openInterest?symbol={symbol}");
    let body = get(&url, REST_TIMEOUT).await?;
    let oi: OpenInterest =
        serde_json::from_str(&body).map_err(|e| BnError::Malformed(e.to_string()))?;
    let value = Decimal::from_str(&oi.open_interest).map_err(|_| {
        BnError::Malformed(format!(
            "openInterest {:?} is not a number",
            oi.open_interest
        ))
    })?;
    Ok((value, oi.time * MS_TO_NS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_blackholed_endpoint_gives_up_instead_of_hanging_the_session() {
        // The failure a refused connection does not cover: the peer completes the
        // handshake and then never answers. An unaccepted listener does exactly that —
        // the kernel finishes the TCP handshake out of the backlog, so the GET is sent
        // and no response ever comes. Loopback only; nothing here leaves the box.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let url = format!("http://{addr}/fapi/v1/exchangeInfo");

        let started = std::time::Instant::now();
        let out = get(&url, Duration::from_millis(250)).await;
        let elapsed = started.elapsed();

        assert!(
            matches!(out, Err(BnError::Http(ref e)) if e.is_timeout()),
            "expected a timeout from a silent peer, got {out:?}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the request outlived its 250 ms budget by {elapsed:?} - it is not bounded"
        );
        // The shipped bound has to be small enough that a stalled read is invisible
        // beside the WS reconnect ceiling; a bound of "eventually" is the same bug with
        // extra steps.
        assert!(REST_TIMEOUT <= Duration::from_secs(30));
    }

    #[test]
    fn open_interest_is_parsed_as_a_decimal_and_carries_the_venues_time() {
        // Captured shape (`GET /fapi/v1/openInterest?symbol=BTCUSDT`). The value is
        // large enough that an f64 would start losing the fractional part, which is
        // exactly why it arrives as a string.
        let body = r#"{"symbol":"BTCUSDT","openInterest":"342584392.0158","time":1785048456235}"#;
        #[derive(serde::Deserialize)]
        struct Oi {
            #[serde(rename = "openInterest")]
            open_interest: String,
            time: i64,
        }
        let oi: Oi = serde_json::from_str(body).unwrap();
        assert_eq!(
            Decimal::from_str(&oi.open_interest).unwrap(),
            Decimal::from_str("342584392.0158").unwrap()
        );
        assert_eq!(oi.time * MS_TO_NS, 1_785_048_456_235 * 1_000_000);
    }

    /// Live read — hits the real Binance USD-M **testnet** REST. Ignored by default so
    /// the gate stays deterministic and offline. It asserts what a captured fixture
    /// cannot: that the shape this crate decodes is still the shape the venue sends.
    ///
    /// Run with `cargo test -p axon-provider-binance -- --ignored`. It places nothing
    /// and needs no credentials; there are none in this repo and this workstream is
    /// not authorized to trade on Binance.
    #[tokio::test]
    #[ignore = "hits the live Binance USD-M futures testnet REST"]
    async fn live_exchange_info_still_decodes() {
        let u = fetch_universe(TESTNET_REST).await.expect("exchangeInfo");
        assert!(u.symbols.len() > 100, "got {} symbols", u.symbols.len());
        let btc = u.symbols.id("BTCUSDT").expect("BTCUSDT is listed");
        let spec = u.instruments.get(btc).expect("BTCUSDT has a grid");
        assert!(spec.min_notional.is_some(), "MIN_NOTIONAL is published");
        assert!(spec.price.tick_at(Decimal::from(64000)) > Decimal::ZERO);
        // The bound `SymbolTable` cannot enforce and `InFlight` silently degrades past.
        assert_eq!(
            u.symbols.exceeds_inflight_bound(),
            0,
            "the universe outgrew InFlight::CAPACITY; the per-symbol gate is now global"
        );
    }
}
