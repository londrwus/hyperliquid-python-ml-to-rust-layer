//! The live Hyperliquid execution client — the async *edge* for order flow.
//!
//! `ExchangeClient` signs an action (increment 1's [`sign`](crate::sign)), wraps it
//! in the `/exchange` request envelope, POSTs it, and parses the response
//! ([`response`]). It implements the venue-agnostic
//! [`ExecutionClient`](axon_providers::ExecutionClient) port, so the core/strategy
//! never see Hyperliquid specifics. Testnet-first (ADR-0009).
//!
//! Response decoding is unit-tested offline; the live round-trip is the
//! `#[ignore]`d `place_then_cancel_on_testnet` test, run manually with a testnet
//! key in `AXON_HL_SECRET_KEY`.

pub mod response;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axon_core::Cloid;
use axon_providers::{
    CancelAck, CancelId, Capabilities, ExecutionClient, InstrumentTable, OrderAck, OrderRequest,
    OrderStatus, ProviderError,
};
use serde::Serialize;

use crate::encode::{
    cancel_action, cancel_by_cloid_action, modify_action, order_action, schedule_cancel_action,
    schedule_cancel_deadline, EncodeError,
};
use crate::nonce::NonceManager;
use crate::sign::{HlSigner, RpcSignature};
use crate::symbol_map::SymbolMap;
use crate::ws::{MAINNET_INFO, TESTNET_INFO};
use response::{parse_cancel_outcomes, parse_order_outcomes, CancelOutcome, OrderOutcome};

/// The `POST /exchange` request envelope. The signature is over the action's
/// msgpack hash, not this JSON, so field order here is irrelevant.
#[derive(Serialize)]
struct ExchangeRequest<'a, A: Serialize> {
    action: &'a A,
    nonce: u64,
    signature: &'a RpcSignature,
    #[serde(rename = "vaultAddress", skip_serializing_if = "Option::is_none")]
    vault_address: Option<String>,
    #[serde(rename = "expiresAfter", skip_serializing_if = "Option::is_none")]
    expires_after: Option<u64>,
}

/// Signs and submits orders/cancels to Hyperliquid's REST `/exchange`.
pub struct ExchangeClient {
    signer: HlSigner,
    nonce: NonceManager,
    http: reqwest::Client,
    base_url: String,
    caps: Capabilities,
    /// The account this client trades for, plus the coin↔id map needed to read its
    /// orders back.
    ///
    /// Needed because some operations are not expressible as a single signed action:
    /// Hyperliquid has no native cancel-all, so [`cancel_all`](Self::cancel_all) must
    /// *read* the open orders first, which means resolving the venue's coin names.
    /// Under the agent-wallet model the signer's own address is not the account —
    /// querying `/info` with an agent address returns empty — so the account cannot be
    /// derived and has to be supplied.
    account: Option<AccountContext>,
    /// Each instrument's tick and lot, shared with the planner (ADR-0025).
    ///
    /// See [`with_instruments`](Self::with_instruments) for why a client nobody hands
    /// one to refuses every order that would add exposure — and still flattens.
    instruments: Arc<InstrumentTable>,
}

/// Map an encoding failure onto the port's taxonomy.
///
/// A precision failure gets its own variant rather than `Unsupported`, because it is
/// categorically not "the venue cannot express this": it means our table or our planner
/// is wrong, and the fix is on this side of the wire.
fn encode_error(e: EncodeError) -> ProviderError {
    match e {
        EncodeError::Precision(_) | EncodeError::PrecisionUnknown { .. } => {
            ProviderError::Precision(e.to_string())
        }
        other => ProviderError::Unsupported {
            venue: crate::VENUE,
            what: other.to_string(),
        },
    }
}

/// What a read-then-write operation needs beyond a signature.
struct AccountContext {
    address: String,
    symbols: SymbolMap,
}

impl ExchangeClient {
    pub const MAINNET: &'static str = "https://api.hyperliquid.xyz/exchange";
    pub const TESTNET: &'static str = "https://api.hyperliquid-testnet.xyz/exchange";

    pub fn new(base_url: impl Into<String>, signer: HlSigner) -> Self {
        Self {
            signer,
            nonce: NonceManager::new(),
            // `reqwest`'s default is *no* request timeout, so a stalled TCP connection
            // hangs one `place_order` for as long as the kernel keeps the socket — and
            // the intent pump is a single task awaiting that call. The ring stops being
            // drained, no order is ever refused, and the session reports OK the whole
            // time: a wedge that looks exactly like a quiet market. A deadline turns it
            // into a counted `intent_failures` the operator can see.
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("the default reqwest client must build"),
            base_url: base_url.into(),
            caps: crate::capabilities(),
            account: None,
            // Empty, so every lookup is `Precision::Unknown`. A client nobody handed a
            // universe to refuses everything that adds exposure and still flattens —
            // the right direction to fail, and a loud one: an error per order, counted
            // as `intent_failures` rather than absorbed.
            instruments: Arc::new(InstrumentTable::new()),
        }
    }

    /// Set the account whose orders this client acts on, and the symbol map used to
    /// read them back.
    ///
    /// Required for [`cancel_all`](Self::cancel_all). Pass the **master account**
    /// address, not the agent-wallet address. The two arguments travel together because
    /// neither is useful alone: an address with no symbol map cannot have its orders
    /// decoded, and a symbol map with no address has nothing to read.
    pub fn with_account(mut self, account: impl Into<String>, symbols: SymbolMap) -> Self {
        self.account = Some(AccountContext {
            address: account.into(),
            symbols,
        });
        self
    }

    /// Give this client the instrument grids it validates against.
    ///
    /// Hand it the **same** `Arc` the planner was given. Two tables that can drift
    /// apart turns the encoder's tripwire into a session-wide outage: the planner
    /// rounds to one grid, the encoder refuses against another, and every refusal reads
    /// in the log like a venue rejection.
    pub fn with_instruments(mut self, instruments: Arc<InstrumentTable>) -> Self {
        self.instruments = instruments;
        self
    }

    /// The signing address. Under the agent-wallet model this is the agent, not the
    /// account — see [`with_account`](Self::with_account).
    pub fn signer_address(&self) -> alloy_primitives::Address {
        self.signer.address()
    }

    /// The `/info` endpoint matching this client's network.
    fn info_url(&self) -> &'static str {
        if self.signer.is_mainnet() {
            MAINNET_INFO
        } else {
            TESTNET_INFO
        }
    }

    fn require_account(&self) -> Result<&AccountContext, ProviderError> {
        self.account
            .as_ref()
            .ok_or_else(|| ProviderError::Unsupported {
                venue: crate::VENUE,
                what: "cancel_all needs the account address and symbol map - build the \
                   client with `.with_account(<master address>, symbols)`"
                    .into(),
            })
    }

    /// Testnet client. The signer must be a testnet signer (signs `source:"b"`).
    pub fn testnet(signer: HlSigner) -> Result<Self, ProviderError> {
        if signer.is_mainnet() {
            return Err(ProviderError::Auth(
                "testnet ExchangeClient requires a testnet signer".into(),
            ));
        }
        Ok(Self::new(Self::TESTNET, signer))
    }

    /// Mainnet client. The signer must be a mainnet signer (signs `source:"a"`).
    pub fn mainnet(signer: HlSigner) -> Result<Self, ProviderError> {
        if !signer.is_mainnet() {
            return Err(ProviderError::Auth(
                "mainnet ExchangeClient requires a mainnet signer".into(),
            ));
        }
        Ok(Self::new(Self::MAINNET, signer))
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Sign `action`, POST it, and return the parsed JSON response.
    async fn post_action(
        &self,
        action: &(impl Serialize + Sync),
    ) -> Result<serde_json::Value, ProviderError> {
        let nonce = self.nonce.next(Self::now_ms());
        let signature = self
            .signer
            .sign_l1_action(action, nonce, None, None)
            .map_err(|e| ProviderError::Auth(e.to_string()))?;
        let body = ExchangeRequest {
            action,
            nonce,
            signature: &signature,
            vault_address: None,
            expires_after: None,
        };
        let resp = self
            .http
            .post(&self.base_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Network(format!("HTTP {status}: {text}")));
        }
        serde_json::from_str(&text)
            .map_err(|e| ProviderError::Rejected(format!("malformed /exchange body: {e}")))
    }

    /// Arm the **dead-man's switch**: tell the venue to cancel every open order at
    /// `deadline_ms` (absolute UTC epoch ms) unless we move the deadline first.
    ///
    /// This is the only safety net that survives *this process* dying. A client-side
    /// cancel cannot run if we have crashed, wedged, or lost the network; a scheduled
    /// cancel is held by the venue and fires regardless. Re-arm it on a timer well
    /// inside the lead time — the canonical pattern is a lead of a few tens of
    /// seconds re-armed every few seconds, so a single missed beat is survivable but a
    /// dead process is not.
    ///
    /// Costs: the venue honours at most
    /// [`SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY`](crate::encode::SCHEDULE_CANCEL_MAX_TRIGGERS_PER_DAY)
    /// firings per UTC day, and each re-arm is an ordinary rate-limited action — so
    /// re-arming every 5 s spends 12 actions/minute of the address budget.
    pub async fn schedule_cancel(&self, deadline_ms: u64) -> Result<(), ProviderError> {
        let now = Self::now_ms();
        let lead = deadline_ms.saturating_sub(now);
        // Validate locally rather than paying a round-trip to be told no.
        schedule_cancel_deadline(now, lead).map_err(|e| ProviderError::Rejected(e.to_string()))?;
        self.post_action(&schedule_cancel_action(Some(deadline_ms)))
            .await
            .map(|_| ())
    }

    /// Arm the dead-man's switch `lead_ms` from now, returning the deadline it set.
    /// This is the call the re-arming loop makes.
    ///
    /// It posts directly rather than delegating to
    /// [`schedule_cancel`](Self::schedule_cancel) on purpose: that method re-reads the
    /// clock, so a lead of exactly the 5 s minimum would have shrunk by the elapsed
    /// microseconds and been rejected as too soon. Validating once against a single
    /// `now` is what makes `arm_dead_mans_switch(5_000)` behave the same every call
    /// instead of failing whenever the two clock reads straddle a millisecond.
    pub async fn arm_dead_mans_switch(&self, lead_ms: u64) -> Result<u64, ProviderError> {
        let deadline = schedule_cancel_deadline(Self::now_ms(), lead_ms)
            .map_err(|e| ProviderError::Rejected(e.to_string()))?;
        self.post_action(&schedule_cancel_action(Some(deadline)))
            .await?;
        Ok(deadline)
    }

    /// Disarm the dead-man's switch.
    ///
    /// Deliberately separate from [`schedule_cancel`](Self::schedule_cancel) rather
    /// than an `Option` parameter: disarming removes a safety net, and that should read
    /// as its own decision at the call site, not as passing `None`.
    pub async fn cancel_scheduled_cancel(&self) -> Result<(), ProviderError> {
        self.post_action(&schedule_cancel_action(None))
            .await
            .map(|_| ())
    }

    /// Authorize an agent (API) wallet to trade for an account.
    ///
    /// **`master` must be the account's own key, not this client's signer.** That is
    /// why it is an explicit parameter rather than reusing `self.signer`: the whole
    /// point of an agent wallet is that the trading key *cannot* grant authority, so a
    /// method that quietly signed this with the hot key would defeat the model it
    /// exists to support. Under a correct setup `self.signer` is the agent being
    /// approved, and it could not sign this action even if it tried.
    ///
    /// This is a **user-signed** action (EIP-712 over the action itself, domain
    /// `HyperliquidSignTransaction`), not an L1 phantom-agent action — mixing the two
    /// is the canonical "invalid signature" mistake, which is why the signing path is a
    /// different method on [`HlSigner`] and `expiresAfter`/`vaultAddress` are
    /// structurally absent from the request body.
    ///
    /// Approving is also a **rotation** primitive: approving a new unnamed agent, or a
    /// named one reusing an existing name, deregisters the old one. Generate a fresh
    /// key per approval — the venue's docs warn that a pruned agent address can have
    /// its previously signed actions replayed.
    pub async fn approve_agent(
        &self,
        master: &HlSigner,
        action: crate::sign::ApproveAgent,
    ) -> Result<(), ProviderError> {
        let body = master
            .user_signed_request(action)
            .map_err(|e| ProviderError::Auth(e.to_string()))?;
        let resp = self
            .http
            .post(&self.base_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Network(format!("HTTP {status}: {text}")));
        }
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ProviderError::Rejected(format!("malformed /exchange body: {e}")))?;
        // Success is `{"status":"ok","response":{"type":"default"}}`; anything else
        // carries the reason in `response`.
        if v.get("status").and_then(|s| s.as_str()) == Some("ok") {
            Ok(())
        } else {
            Err(ProviderError::Rejected(format!(
                "approveAgent rejected: {}",
                v.get("response").unwrap_or(&v)
            )))
        }
    }

    async fn submit_orders(
        &self,
        reqs: &[OrderRequest],
    ) -> Result<Vec<OrderOutcome>, ProviderError> {
        let action = order_action(reqs, &self.instruments).map_err(encode_error)?;
        parse_order_outcomes(&self.post_action(&action).await?)
    }

    /// Sign and POST the cancel for `id`, and hand back the venue's body **verbatim**.
    ///
    /// [`ExecutionClient::cancel`] is the production call and delegates to this, then
    /// throws the body away — nothing in a running session can act on it, and a
    /// `CancelAck` is the honest reduction. What *cannot* obtain the body any other way
    /// is a live test, and the shape of a cancel reply is exactly the kind of claim that
    /// rots unobserved: `{"channel":"pong"}` carries no `data` field, and the unit test
    /// asserting pongs were ignored passed for months against an invented
    /// `{"channel":"pong","data":null}` the venue has never sent. A decoder is only as
    /// good as the bytes it was given, so the one caller that can get real bytes gets
    /// them instead of a summary — including for a *rejected* cancel, where the reply is
    /// an HTTP 200 whose per-item `error` is the only thing that says what the venue
    /// thought, and where the reduction to [`ProviderError::Rejected`] keeps the message
    /// but loses the envelope it arrived in.
    ///
    /// An `Err` here therefore means the POST itself failed (network, auth, or a body
    /// that is not JSON) — never that the venue refused the cancel. Parse the value with
    /// [`parse_cancel_outcomes`](response::parse_cancel_outcomes) for that.
    pub async fn cancel_raw(&self, id: CancelId) -> Result<serde_json::Value, ProviderError> {
        // HL keys cancels on (asset, id) — the symbol rides along in `CancelId`.
        match id {
            CancelId::OrderId { symbol, order_id } => {
                self.post_action(&cancel_action(&[(symbol, order_id)]))
                    .await
            }
            CancelId::Cloid { symbol, cloid } => {
                self.post_action(&cancel_by_cloid_action(&[(symbol, cloid)]))
                    .await
            }
        }
    }
}

fn ack_from_outcome(cloid: Cloid, outcome: OrderOutcome) -> OrderAck {
    match outcome {
        OrderOutcome::Resting { oid } => OrderAck {
            cloid,
            order_id: Some(oid),
            status: OrderStatus::Resting,
        },
        OrderOutcome::Filled { oid } => OrderAck {
            cloid,
            order_id: Some(oid),
            status: OrderStatus::Filled,
        },
        OrderOutcome::Rejected { .. } => OrderAck {
            cloid,
            order_id: None,
            status: OrderStatus::Rejected,
        },
    }
}

#[async_trait]
impl ExecutionClient for ExchangeClient {
    fn capabilities(&self) -> &Capabilities {
        &self.caps
    }

    async fn place_order(&self, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        let cloid = req.cloid;
        let outcome = self
            .submit_orders(std::slice::from_ref(&req))
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Rejected("venue returned no order status".into()))?;
        // A single-order rejection surfaces its reason as an error.
        match outcome {
            OrderOutcome::Rejected { reason } => Err(ProviderError::Rejected(reason)),
            ok => Ok(ack_from_outcome(cloid, ok)),
        }
    }

    async fn place_batch(&self, reqs: Vec<OrderRequest>) -> Result<Vec<OrderAck>, ProviderError> {
        let cloids: Vec<Cloid> = reqs.iter().map(|r| r.cloid).collect();
        let outcomes = self.submit_orders(&reqs).await?;
        if outcomes.len() != cloids.len() {
            return Err(ProviderError::Rejected(format!(
                "venue returned {} statuses for {} orders",
                outcomes.len(),
                cloids.len()
            )));
        }
        Ok(cloids
            .into_iter()
            .zip(outcomes)
            .map(|(cloid, o)| ack_from_outcome(cloid, o))
            .collect())
    }

    async fn cancel(&self, id: CancelId) -> Result<CancelAck, ProviderError> {
        let resp = self.cancel_raw(id).await?;
        // The ack echoes back the identity the caller addressed the cancel with, and
        // only that one: the venue's reply carries neither, so claiming the other would
        // be inventing it.
        let ack = match id {
            CancelId::OrderId { order_id, .. } => CancelAck {
                cloid: None,
                order_id: Some(order_id),
            },
            CancelId::Cloid { cloid, .. } => CancelAck {
                cloid: Some(cloid),
                order_id: None,
            },
        };
        match parse_cancel_outcomes(&resp)?.into_iter().next() {
            Some(CancelOutcome::Success) => Ok(ack),
            Some(CancelOutcome::Rejected { reason }) => Err(ProviderError::Rejected(reason)),
            None => Err(ProviderError::Rejected(
                "venue returned no cancel status".into(),
            )),
        }
    }

    /// Cancel every open order, by sweeping: read them from `/info`, then batch-cancel.
    ///
    /// Hyperliquid has no native cancel-all action, so this is inherently
    /// read-then-write and therefore inherently racy — an order that rests between the
    /// read and the cancel survives. That is why it is **not** the primary safety
    /// mechanism: [`arm_dead_mans_switch`](Self::arm_dead_mans_switch) is, because the
    /// venue evaluates it atomically and it still fires if this process is gone.
    /// `cancel_all` is the orderly-shutdown path, and callers who need certainty should
    /// re-run it until it reports nothing left.
    ///
    /// Uses `frontendOpenOrders`, not `openOrders`: same rate-limit weight, but it is
    /// the only sweep source that exposes `cloid` and trigger metadata, so a cancel can
    /// be attributed back to the order that requested it.
    ///
    /// Batched at [`MAX_BATCH`](crate::MAX_BATCH). Cancels draw on a strictly larger
    /// rate-limit allowance than places (`min(limit + 100_000, limit * 2)`), which is
    /// exactly what makes an unwind possible while rate-limited on placement.
    async fn cancel_all(&self) -> Result<(), ProviderError> {
        let ctx = self.require_account()?;
        let open = crate::info::fetch_frontend_open_orders(
            self.info_url(),
            &ctx.address,
            None,
            &ctx.symbols,
        )
        .await
        .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !open.is_complete() {
            // An order on an instrument we do not track is still an order. Surfacing
            // this matters: a caller told "cancelled everything" while a HIP-3 or spot
            // order survives has been misled about its own exposure.
            return Err(ProviderError::Unsupported {
                venue: crate::VENUE,
                what: format!(
                    "{} open order(s) are on untracked instruments {:?} and cannot be \
                     cancelled through the symbol map",
                    open.skipped_coins.len(),
                    open.skipped_coins
                ),
            });
        }

        let ids: Vec<(axon_core::SymbolId, axon_core::OrderId)> = open
            .items
            .iter()
            .filter_map(|o| match o.cancel_id() {
                CancelId::OrderId { symbol, order_id } => Some((symbol, order_id)),
                // `frontendOpenOrders` always carries an oid, so this is unreachable in
                // practice; skipping beats fabricating an id.
                CancelId::Cloid { .. } => None,
            })
            .collect();

        for chunk in ids.chunks(crate::MAX_BATCH as usize) {
            let resp = self.post_action(&cancel_action(chunk)).await?;
            // Per-order failures are expected and benign here: an order that filled or
            // was already cancelled between the read and this call reports an error, and
            // the goal (it is not resting) is satisfied. Only report if none succeeded,
            // which instead suggests the whole call was wrong.
            let outcomes = parse_cancel_outcomes(&resp)?;
            if !outcomes.is_empty()
                && outcomes
                    .iter()
                    .all(|o| matches!(o, CancelOutcome::Rejected { .. }))
            {
                let reason = outcomes
                    .iter()
                    .find_map(|o| match o {
                        CancelOutcome::Rejected { reason } => Some(reason.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                return Err(ProviderError::Rejected(format!(
                    "cancel_all: every cancel in a batch of {} failed: {reason}",
                    chunk.len()
                )));
            }
        }
        Ok(())
    }

    async fn modify(&self, id: CancelId, req: OrderRequest) -> Result<OrderAck, ProviderError> {
        let cloid = req.cloid;
        let action = modify_action(id, &req, &self.instruments).map_err(encode_error)?;
        let outcome = parse_order_outcomes(&self.post_action(&action).await?)?
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Rejected("venue returned no modify status".into()))?;
        match outcome {
            OrderOutcome::Rejected { reason } => Err(ProviderError::Rejected(reason)),
            ok => Ok(ack_from_outcome(cloid, ok)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Side, SymbolId, Tif};
    use rust_decimal::Decimal;
    use std::str::FromStr;

    #[test]
    fn exchange_request_serializes_with_expected_keys() {
        let sig = RpcSignature {
            r: "0x01".into(),
            s: "0x02".into(),
            v: 27,
        };
        let action = order_action(
            &[OrderRequest::limit(
                SymbolId::new(0),
                Side::Buy,
                Decimal::from_str("0.001").unwrap(),
                Decimal::from_str("10000").unwrap(),
                Tif::Gtc,
                Cloid::new(1),
            )],
            // The envelope's keys are what this test is about, not the grid.
            &InstrumentTable::unconstrained(),
        )
        .unwrap();
        let body = ExchangeRequest {
            action: &action,
            nonce: 1_700_000_000_000,
            signature: &sig,
            vault_address: None,
            expires_after: None,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["nonce"], 1_700_000_000_000u64);
        assert_eq!(v["signature"]["v"], 27);
        assert_eq!(v["action"]["type"], "order");
        assert!(v.get("vaultAddress").is_none()); // skipped when None
    }

    fn testnet_client() -> ExchangeClient {
        // Hardhat account #0 — a public test vector, never a real key.
        let signer = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            false,
        )
        .unwrap();
        ExchangeClient::testnet(signer).unwrap()
    }

    #[tokio::test]
    async fn cancel_all_refuses_without_an_account_rather_than_guessing() {
        // Under the agent-wallet model the signer's address is NOT the account, so there
        // is nothing to fall back on. Silently sweeping the wrong address, or the agent's
        // (which returns empty and would report success), is worse than an error.
        let err = testnet_client().cancel_all().await.unwrap_err();
        match err {
            ProviderError::Unsupported { what, .. } => {
                assert!(what.contains("with_account"), "actionable message: {what}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn info_url_follows_the_signers_network() {
        use crate::ws::{MAINNET_INFO, TESTNET_INFO};
        assert_eq!(testnet_client().info_url(), TESTNET_INFO);

        let mainnet_signer = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            true,
        )
        .unwrap();
        let c = ExchangeClient::mainnet(mainnet_signer).unwrap();
        assert_eq!(c.info_url(), MAINNET_INFO);
    }

    #[test]
    fn approve_agent_body_carries_only_the_user_signed_keys() {
        // A user-signed action must never carry `expiresAfter` or `vaultAddress` — the
        // venue rejects them, and the request type is what enforces it.
        let master = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            false,
        )
        .unwrap();
        // Mixed case on purpose: the venue requires a lowercased agentAddress.
        let agent: alloy_primitives::Address = "0x1234567890ABCDEF1234567890abcdef12345678"
            .parse()
            .unwrap();
        let action = crate::sign::ApproveAgent::unnamed(agent, 1_784_976_929_583, false);
        let body = master.user_signed_request(action).unwrap();
        let v = serde_json::to_value(&body).unwrap();

        assert_eq!(v["action"]["type"], "approveAgent");
        assert_eq!(v["action"]["hyperliquidChain"], "Testnet");
        assert_eq!(v["nonce"], 1_784_976_929_583u64);
        assert_eq!(
            v["action"]["nonce"], v["nonce"],
            "inner and outer nonce must agree"
        );
        assert!(
            v["action"]["agentAddress"]
                .as_str()
                .unwrap()
                .chars()
                .all(|c| !c.is_ascii_uppercase()),
            "agentAddress must be lowercased"
        );
        assert!(
            v["action"].get("agentName").is_none(),
            "an unnamed agent omits the key entirely (it still hashes as \"\")"
        );
        assert!(v.get("expiresAfter").is_none());
        assert!(v.get("vaultAddress").is_none());
    }

    #[test]
    fn env_mismatch_is_rejected() {
        // A mainnet signer must not build a testnet client.
        let signer = HlSigner::from_hex(
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            true,
        )
        .unwrap();
        assert!(ExchangeClient::testnet(signer).is_err());
    }

    /// Live testnet round-trip — hits the real venue. Ignored by default; run with
    /// a funded testnet key: `AXON_HL_SECRET_KEY=0x... cargo test -p
    /// axon-provider-hyperliquid -- --ignored place_then_cancel_on_testnet`.
    #[tokio::test]
    #[ignore = "hits live Hyperliquid testnet; needs AXON_HL_SECRET_KEY"]
    async fn place_then_cancel_on_testnet() {
        use crate::ws::{fetch_universe, TESTNET_INFO};
        use crate::HyperliquidMarketData;

        use axon_providers::PriceIntent;

        let coin = "BTC";
        let signer = HlSigner::from_env(false).expect("AXON_HL_SECRET_KEY (testnet)");

        // Resolve the real asset index, the real grid and a safe resting price from the
        // book — all from one `meta` read, because two can disagree after a listing.
        let universe = fetch_universe(TESTNET_INFO).await.expect("meta");
        let symbols = universe.symbols.clone();
        let sym = symbols.id(coin).expect("coin in universe");
        // Taken *before* the table moves into the client, and deliberately the same
        // table the encoder validates against: a price this test rounds and a price the
        // wire accepts must not be two different opinions.
        let grid = *universe
            .instruments
            .get(sym)
            .expect("the universe declares this coin's grid");
        let client = ExchangeClient::testnet(signer)
            .unwrap()
            .with_instruments(Arc::new(universe.instruments));

        // Best bid via a one-shot L2 snapshot (reuse the market-data path).
        let (tx, rx) = axon_core::bus(64);
        let md = HyperliquidMarketData::testnet(symbols, vec![coin.into()], tx);
        md.subscribe_coin(axon_providers::Feed::L2Book, coin);
        let stream = tokio::spawn(async move { md.run_once().await });
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        stream.abort();
        let mut proc = axon_marketdata::MarketDataProcessor::new();
        axon_core::drain_available(&rx, &mut proc);
        let best_bid = proc.book(sym).and_then(|b| b.best_bid()).expect("a bid").0;

        // Post-only buy ~20% below best bid, sized ~$15 notional so it clears the
        // venue's $10 minimum. It rests without ever crossing.
        //
        // Both numbers go through the **production** grid rather than a `round_dp` this
        // test owns, and that is the whole point rather than a tidy-up. The previous
        // version rounded the price to an integer and the size to a hardcoded 4 decimal
        // places, and both were on-grid purely by luck: integers are exempt from the
        // venue's 5-significant-figure rule, and 4 decimals is *coarser* than BTC's
        // 5-decimal lot. Point this at an asset with `szDecimals < 4` and the size is
        // off-grid, the encoder refuses it (ADR-0025), and the refusal reads exactly
        // like a signing failure. A live test that does its own venue arithmetic can go
        // green over a broken encoder — which is the one thing it exists to catch.
        //
        // `Passive` floors a buy, so rounding can only ever move the price further from
        // the touch; this order cannot be rejected as crossing.
        let px = grid.price.quantize(
            best_bid * Decimal::from_str("0.8").unwrap(),
            Side::Buy,
            PriceIntent::Passive,
        );
        // The lot truncates toward zero, so the notional is asserted rather than
        // assumed: on a coarse-lot asset a $15 target can truncate under the venue's
        // $10 floor, and `minTradeNtlRejected` is another rejection that reads like a
        // signing bug from in here.
        let qty = grid.size.quantize(Decimal::from(15) / px);
        assert!(
            qty > Decimal::ZERO && qty * px >= Decimal::from(10),
            "lot rounding left {qty} @ {px} (${}) under the venue's $10 minimum",
            qty * px
        );
        // A client id unique per run. A constant one is a bet that the venue frees a
        // cloid once its order reaches a terminal state — it does, observed on testnet
        // across two consecutive runs of this test — but the bet is only safe while the
        // previous run actually got to its cancel. A run killed between place and cancel
        // leaves the id live, and the next run's collision is refused in a way that is
        // indistinguishable from a bad signature.
        let cloid = Cloid::new(
            (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
                << 8)
                | 0xA1,
        );
        let req = OrderRequest::limit(sym, Side::Buy, qty, px, Tif::PostOnly, cloid);

        let ack = client.place_order(req).await.expect("place");
        let oid = ack.order_id.expect("resting oid");
        assert_eq!(ack.status, OrderStatus::Resting);
        eprintln!("rested oid={oid} px={px} qty={qty}");

        let cancelled = client
            .cancel(CancelId::OrderId {
                symbol: sym,
                order_id: oid,
            })
            .await
            .expect("cancel");
        assert_eq!(cancelled.order_id, Some(oid));
        eprintln!("cancelled oid={oid}");
    }
}
