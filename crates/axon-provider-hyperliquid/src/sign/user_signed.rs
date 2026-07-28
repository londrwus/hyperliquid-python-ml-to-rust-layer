//! Hyperliquid **user-signed** actions — the second signing scheme (ADR-0009).
//!
//! Orders and cancels are L1 actions: msgpack-hashed and signed through a phantom
//! agent ([`super::action`]). Admin actions — agent approval, withdrawals,
//! transfers — are *user-signed*: the action's own fields are signed directly as
//! EIP-712 typed data under the `HyperliquidSignTransaction` domain
//! ([`super::eip712`]). Everything the L1 path appends (the msgpack
//! `connectionId`, the vault prefix, `expiresAfter`) is absent, and the venue
//! rejects `expiresAfter` on this scheme outright.
//!
//! This increment implements `approveAgent`, which is what makes a leaked hot key
//! survivable: it authorizes a separate **agent (API) wallet** to trade the
//! account while leaving it unable to withdraw. Note the asymmetry — the
//! `approveAgent` action itself must be signed by the **master account** key (the
//! funded one), so approval is a deliberate one-off ceremony, never something the
//! trading loop does with its own key.

use alloy_primitives::{hex, Address};
use alloy_sol_types::sol;
use serde::{Serialize, Serializer};

use super::eip712::UserSignedPayload;
use super::signer::SignError;

/// How far in the future an agent's expiry may be set: 180 days, the venue's cap.
/// Omitting an expiry entirely gets the venue's own default, which is also ~180
/// days.
pub const MAX_AGENT_VALIDITY_MS: u64 = 180 * 24 * 60 * 60 * 1_000;

/// Headroom [`AgentName::max_validity`] leaves under [`MAX_AGENT_VALIDITY_MS`].
///
/// The venue checks the deadline against its own clock at receipt, so requesting the
/// exact cap makes the request's validity depend on clock skew — and a rejection there
/// costs a spent nonce. One minute of slack removes the whole class of failure.
pub const MAX_VALIDITY_MARGIN_MS: u64 = 60 * 1_000;

/// The token the venue parses an agent's expiry out of its name with.
const VALID_UNTIL: &str = "valid_until";

const MAINNET: &str = "Mainnet";
const TESTNET: &str = "Testnet";

/// The user-signed network discriminator — this scheme's analogue of the L1
/// phantom agent's `source` (`"a"`/`"b"`). It is a *signed* field, so a `"Mainnet"`
/// action sent to testnet is not merely misrouted, it is invalid.
pub const fn hyperliquid_chain(is_mainnet: bool) -> &'static str {
    if is_mainnet {
        MAINNET
    } else {
        TESTNET
    }
}

/// The `signatureChainId`: one value with two encodings — a hex **string** in the
/// action JSON, and the same number as the EIP-712 domain's `chainId`.
///
/// They are bundled into one type because a disagreement between them fails
/// silently: the venue rebuilds the domain from the string it received, recovers a
/// different signer, and reports an invalid signature with no hint as to why.
///
/// The value itself does not affect validity — nothing on-chain verifies these
/// signatures, so it only changes what a hardware wallet *displays*. We default to
/// Arbitrum One, the chain Hyperliquid's own bridge lives on and the value in the
/// venue's docs; the official Python SDK hardcodes Arbitrum Sepolia for both
/// networks and is equally accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignatureChainId(u64);

impl SignatureChainId {
    /// Arbitrum One (`0xa4b1`) — the default.
    pub const ARBITRUM_ONE: Self = Self(42_161);
    /// Arbitrum Sepolia (`0x66eee`) — what the official Python SDK sends.
    pub const ARBITRUM_SEPOLIA: Self = Self(421_614);

    /// Declare an arbitrary chain id (see the type docs: any value is accepted).
    pub const fn new(chain_id: u64) -> Self {
        Self(chain_id)
    }

    /// The integer form, for the EIP-712 domain's `chainId`.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The lowercase `0x`-prefixed hex form that goes into the action JSON.
    pub fn to_hex(self) -> String {
        format!("{:#x}", self.0)
    }
}

impl Default for SignatureChainId {
    fn default() -> Self {
        Self::ARBITRUM_ONE
    }
}

impl Serialize for SignatureChainId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

/// A validated `approveAgent` name.
///
/// Hyperliquid has no expiry field: an agent's deadline rides *inside its name*,
/// as the suffix `valid_until {epoch_millis}`, which the venue parses back out.
/// Going through this type is what enforces the 180-day ceiling locally, so an
/// over-long expiry fails here instead of burning a nonce on a round trip.
///
/// The name is also the agent's **identity**: approving a name that already exists
/// deregisters the old agent. That is why a name is never silently rewritten here
/// — an untrimmed or already-suffixed name is rejected rather than normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentName(String);

impl AgentName {
    /// A named agent with no explicit expiry; the venue applies its own default
    /// (~180 days).
    pub fn new(name: &str) -> Result<Self, SignError> {
        Ok(Self(Self::check_base(name)?.to_string()))
    }

    /// A named agent expiring at `valid_until_ms` (epoch **milliseconds**),
    /// validated against the current time `now_ms`.
    pub fn valid_until(base: &str, valid_until_ms: u64, now_ms: u64) -> Result<Self, SignError> {
        let base = Self::check_base(base)?;
        if valid_until_ms <= now_ms {
            return Err(SignError::AgentExpiryNotFuture {
                valid_until_ms,
                now_ms,
            });
        }
        if valid_until_ms - now_ms > MAX_AGENT_VALIDITY_MS {
            return Err(SignError::AgentExpiryTooFar {
                valid_until_ms,
                now_ms,
            });
        }
        Ok(Self(format!("{base} {VALID_UNTIL} {valid_until_ms}")))
    }

    /// A named agent with the longest lifetime that is *safely* within the venue's cap:
    /// `now_ms + 180 days − ` [`MAX_VALIDITY_MARGIN_MS`].
    ///
    /// The margin is not timidity. The venue re-evaluates the deadline against **its**
    /// clock when the request arrives, so asking for exactly 180 days means any clock
    /// skew or network delay in our favour puts the request over the cap — and it is
    /// rejected *after* the nonce has been spent, which is the very outcome the local
    /// validation exists to avoid. A minute of headroom costs nothing measurable
    /// against a six-month lifetime.
    pub fn max_validity(base: &str, now_ms: u64) -> Result<Self, SignError> {
        let lifetime = MAX_AGENT_VALIDITY_MS.saturating_sub(MAX_VALIDITY_MARGIN_MS);
        Self::valid_until(base, now_ms.saturating_add(lifetime), now_ms)
    }

    /// The name exactly as it will be hashed and sent.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Reject names the venue would misread: empty ones, ones carrying stray
    /// whitespace (which would double up around the `valid_until` separator), and
    /// ones that already contain the expiry token.
    fn check_base(base: &str) -> Result<&str, SignError> {
        if base.is_empty() || base.trim() != base || base.contains(VALID_UNTIL) {
            return Err(SignError::BadAgentName);
        }
        Ok(base)
    }
}

impl Serialize for AgentName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

sol! {
    /// The EIP-712 struct hashed for `approveAgent` — the venue's field order and
    /// types, verbatim. `sol!` supplies `encodeData`; the `typeHash` comes from
    /// [`APPROVE_AGENT_ENCODE_TYPE`] because the venue's type name contains a `:`
    /// (see [`UserSignedPayload`]).
    struct ApproveAgentPayload {
        string hyperliquidChain;
        address agentAddress;
        string agentName;
        uint64 nonce;
    }
}

/// The EIP-712 `primaryType` for `approveAgent`.
pub const APPROVE_AGENT_PRIMARY_TYPE: &str = "HyperliquidTransaction:ApproveAgent";

/// The byte-exact EIP-712 `encodeType` string for `approveAgent`, verified against
/// the official Python SDK and two Rust SDKs. One byte off — a space after a
/// comma, a reordered field — changes the `typeHash` and with it every signature.
pub const APPROVE_AGENT_ENCODE_TYPE: &str = "HyperliquidTransaction:ApproveAgent(string hyperliquidChain,address agentAddress,string agentName,uint64 nonce)";

impl UserSignedPayload for ApproveAgentPayload {
    const PRIMARY_TYPE: &'static str = APPROVE_AGENT_PRIMARY_TYPE;
    const ENCODE_TYPE: &'static str = APPROVE_AGENT_ENCODE_TYPE;
}

/// A Hyperliquid user-signed action: its JSON wire shape, plus the EIP-712 struct
/// that is actually hashed.
///
/// The two are deliberately separate. They legitimately differ (`approveAgent`
/// hashes `agentName` as `""` but omits the key from the JSON), and the envelope
/// `nonce` and domain chain id are *read off* the action instead of being passed
/// alongside it — which is what makes an inner/outer nonce mismatch
/// unrepresentable rather than merely discouraged.
pub trait UserSignedAction: Serialize {
    /// The typed-data struct this action is hashed as.
    type Payload: UserSignedPayload;

    /// The typed-data view of this action's current field values.
    fn eip712_payload(&self) -> Self::Payload;

    /// The action's `nonce`, which is also the envelope's `nonce`.
    fn nonce(&self) -> u64;

    /// The declared `signatureChainId`, whose integer form is the domain chainId.
    fn signature_chain_id(&self) -> SignatureChainId;

    /// Whether the action declares `hyperliquidChain: "Mainnet"`.
    fn is_mainnet(&self) -> bool;
}

/// The `approveAgent` action, exactly as it is POSTed to `/exchange`.
///
/// Approving an agent authorizes it to trade — never to withdraw or transfer —
/// which is the whole point: the trading process holds a key that cannot move
/// funds. An account may hold **one unnamed** agent plus up to **three named**
/// ones.
///
/// # Deregistration, and why agent addresses must not be reused
///
/// An agent is deregistered when a new unnamed agent is approved, when one with a
/// **matching name** is approved, when its `valid_until` passes, or when the
/// account is emptied. The venue warns explicitly against reusing an agent
/// address afterwards: once the agent has been pruned, actions it previously
/// signed can be **replayed** against a fresh approval of the same address.
/// Always approve a freshly generated key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApproveAgent {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(rename = "hyperliquidChain")]
    hyperliquid_chain: &'static str,
    #[serde(rename = "signatureChainId")]
    signature_chain_id: SignatureChainId,
    #[serde(
        rename = "agentAddress",
        serialize_with = "serialize_lowercase_address"
    )]
    agent_address: Address,
    /// `None` for the account's single unnamed agent: the key is then omitted
    /// from the JSON entirely, yet still hashed as `""`. See
    /// [`eip712_payload`](UserSignedAction::eip712_payload).
    #[serde(rename = "agentName", skip_serializing_if = "Option::is_none")]
    agent_name: Option<AgentName>,
    nonce: u64,
}

impl ApproveAgent {
    /// Approve a **named** agent. Up to three may coexist; approving a name that
    /// already exists replaces — and thereby deregisters — that agent.
    ///
    /// `nonce` becomes both the action's `nonce` field and the request envelope's,
    /// which cannot diverge (see [`UserSignedAction`]). Take it from the
    /// [`NonceManager`](crate::NonceManager).
    pub fn named(agent_address: Address, name: AgentName, nonce: u64, is_mainnet: bool) -> Self {
        Self {
            kind: "approveAgent",
            hyperliquid_chain: hyperliquid_chain(is_mainnet),
            signature_chain_id: SignatureChainId::default(),
            agent_address,
            agent_name: Some(name),
            nonce,
        }
    }

    /// Approve the account's single **unnamed** agent. Approving another unnamed
    /// agent later deregisters this one.
    pub fn unnamed(agent_address: Address, nonce: u64, is_mainnet: bool) -> Self {
        Self {
            kind: "approveAgent",
            hyperliquid_chain: hyperliquid_chain(is_mainnet),
            signature_chain_id: SignatureChainId::default(),
            agent_address,
            agent_name: None,
            nonce,
        }
    }

    /// Declare a different `signatureChainId` (e.g. to match what a hardware
    /// wallet should display). Any value is accepted by the venue; the domain
    /// chainId follows automatically.
    pub fn with_signature_chain_id(mut self, chain_id: SignatureChainId) -> Self {
        self.signature_chain_id = chain_id;
        self
    }

    /// The agent's name, or `None` for the unnamed agent.
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_ref().map(AgentName::as_str)
    }
}

impl UserSignedAction for ApproveAgent {
    type Payload = ApproveAgentPayload;

    /// Build the typed-data struct that is actually hashed.
    ///
    /// **The `agentName` trap:** an unnamed agent still hashes `agentName` as the
    /// empty string — the field is part of the type, so dropping it would change
    /// the `typeHash` — but the JSON must omit the key entirely. The Python SDK
    /// makes the asymmetry explicit: it fills the name in, signs, then deletes the
    /// key. Sending `"agentName": ""` instead earns a "does not exist" error.
    fn eip712_payload(&self) -> ApproveAgentPayload {
        ApproveAgentPayload {
            hyperliquidChain: self.hyperliquid_chain.to_string(),
            // An `address` field is hashed as a canonical 32-byte word, so the
            // hash is case-blind; only the JSON has to be lowercased.
            agentAddress: self.agent_address,
            agentName: self.agent_name().unwrap_or_default().to_string(),
            nonce: self.nonce,
        }
    }

    fn nonce(&self) -> u64 {
        self.nonce
    }

    fn signature_chain_id(&self) -> SignatureChainId {
        self.signature_chain_id
    }

    fn is_mainnet(&self) -> bool {
        self.hyperliquid_chain == MAINNET
    }
}

/// Serialize an address as lowercase `0x…` hex.
///
/// Spelled out rather than left to `{}` because `alloy`'s `Display for Address`
/// emits the EIP-55 **checksummed** (mixed-case) form, and the venue documents a
/// non-lowercased `agentAddress` as a common cause of rejection.
fn serialize_lowercase_address<S: Serializer>(
    address: &Address,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&hex::encode_prefixed(address.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_sol_types::SolStruct;

    /// The address from the docs' `approveAgent` example, mixed-case so the
    /// lowercasing is actually exercised.
    const AGENT: &str = "0x1234567890ABCDEF1234567890abcdef12345678";
    const AGENT_LOWER: &str = "0x1234567890abcdef1234567890abcdef12345678";

    fn agent() -> Address {
        AGENT.parse().expect("literal address")
    }

    #[test]
    fn encode_type_is_byte_exact() {
        assert_eq!(
            APPROVE_AGENT_ENCODE_TYPE,
            "HyperliquidTransaction:ApproveAgent(string hyperliquidChain,address agentAddress,string agentName,uint64 nonce)"
        );
        assert!(APPROVE_AGENT_ENCODE_TYPE.starts_with(APPROVE_AGENT_PRIMARY_TYPE));
    }

    #[test]
    fn encode_type_matches_the_payload_struct_fields() {
        // Cross-check the hand-written string against `sol!`'s own view of the
        // struct: they must differ *only* in the leading type name. This is what
        // catches a field added to the payload but not to the string (or a type
        // or order changed on either side).
        let derived = ApproveAgentPayload::eip712_encode_type();
        assert_eq!(
            APPROVE_AGENT_ENCODE_TYPE,
            derived.replacen("ApproveAgentPayload", APPROVE_AGENT_PRIMARY_TYPE, 1)
        );
    }

    #[test]
    fn named_action_matches_the_documented_json() {
        // The literal example body from the venue's docs.
        let nonce = 1_784_976_929_583u64;
        let name = AgentName::valid_until("axon-live", 1_800_000_000_000, nonce).unwrap();
        let action = ApproveAgent::named(agent(), name, nonce, true);
        assert_eq!(
            serde_json::to_value(&action).unwrap(),
            serde_json::json!({
                "type": "approveAgent",
                "hyperliquidChain": "Mainnet",
                "signatureChainId": "0xa4b1",
                "agentAddress": AGENT_LOWER,
                "agentName": "axon-live valid_until 1800000000000",
                "nonce": 1_784_976_929_583u64,
            })
        );
    }

    #[test]
    fn named_agent_keeps_the_name_in_json_and_in_the_hash() {
        let action = ApproveAgent::named(agent(), AgentName::new("axon").unwrap(), 7, false);
        let json = serde_json::to_value(&action).unwrap();
        assert_eq!(json["agentName"], "axon");
        assert_eq!(action.agent_name(), Some("axon"));
        assert_eq!(action.eip712_payload().agentName, "axon");
    }

    #[test]
    fn unnamed_agent_omits_the_json_key_but_hashes_an_empty_name() {
        let action = ApproveAgent::unnamed(agent(), 7, false);
        let json = serde_json::to_value(&action).unwrap();
        // Both halves of the trap, asserted together: absent from the JSON...
        assert!(json.get("agentName").is_none());
        // ...but present as "" in the struct that gets hashed.
        assert_eq!(action.eip712_payload().agentName, "");

        let explicit_empty = ApproveAgentPayload {
            hyperliquidChain: TESTNET.to_string(),
            agentAddress: agent(),
            agentName: String::new(),
            nonce: 7,
        };
        assert_eq!(
            action.eip712_payload().hash_struct(),
            explicit_empty.hash_struct()
        );
        // And an unnamed agent is a different signature from any named one.
        let named = ApproveAgent::named(agent(), AgentName::new("axon").unwrap(), 7, false);
        assert_ne!(
            action.eip712_payload().hash_struct(),
            named.eip712_payload().hash_struct()
        );
    }

    #[test]
    fn agent_address_is_lowercased_in_json_and_case_blind_in_the_hash() {
        let mixed = ApproveAgent::unnamed(agent(), 7, false);
        let lower = ApproveAgent::unnamed(AGENT_LOWER.parse().unwrap(), 7, false);
        assert_eq!(
            serde_json::to_value(&mixed).unwrap()["agentAddress"],
            AGENT_LOWER
        );
        assert_eq!(
            mixed.eip712_payload().hash_struct(),
            lower.eip712_payload().hash_struct()
        );
    }

    #[test]
    fn inner_nonce_is_the_actions_only_nonce() {
        let action = ApproveAgent::unnamed(agent(), 1_784_976_929_583, false);
        let json = serde_json::to_value(&action).unwrap();
        assert_eq!(json["nonce"], 1_784_976_929_583u64);
        assert_eq!(action.nonce(), json["nonce"].as_u64().unwrap());
        assert_eq!(action.eip712_payload().nonce, action.nonce());
    }

    #[test]
    fn network_flag_drives_hyperliquid_chain_and_the_hash() {
        let mainnet = ApproveAgent::unnamed(agent(), 7, true);
        let testnet = ApproveAgent::unnamed(agent(), 7, false);
        assert_eq!(
            serde_json::to_value(&mainnet).unwrap()["hyperliquidChain"],
            "Mainnet"
        );
        assert_eq!(
            serde_json::to_value(&testnet).unwrap()["hyperliquidChain"],
            "Testnet"
        );
        assert!(mainnet.is_mainnet() && !testnet.is_mainnet());
        assert_ne!(
            mainnet.eip712_payload().hash_struct(),
            testnet.eip712_payload().hash_struct()
        );
    }

    #[test]
    fn signature_chain_id_hex_and_int_agree() {
        assert_eq!(SignatureChainId::default(), SignatureChainId::ARBITRUM_ONE);
        assert_eq!(SignatureChainId::ARBITRUM_ONE.to_hex(), "0xa4b1");
        assert_eq!(SignatureChainId::ARBITRUM_ONE.get(), 0xa4b1);
        assert_eq!(SignatureChainId::ARBITRUM_SEPOLIA.to_hex(), "0x66eee");
        assert_eq!(SignatureChainId::ARBITRUM_SEPOLIA.get(), 0x6_6eee);

        let action = ApproveAgent::unnamed(agent(), 7, false)
            .with_signature_chain_id(SignatureChainId::ARBITRUM_SEPOLIA);
        let json = serde_json::to_value(&action).unwrap();
        assert_eq!(json["signatureChainId"], "0x66eee");
        // The domain chainId is derived from the same value, never passed apart.
        assert_eq!(action.signature_chain_id().get(), 421_614);
    }

    #[test]
    fn expiry_beyond_180_days_is_rejected_locally() {
        let now = 1_784_976_929_583u64;
        // Exactly 180 days is the boundary and is allowed.
        let ok = AgentName::valid_until("axon", now + MAX_AGENT_VALIDITY_MS, now).unwrap();
        assert_eq!(
            ok.as_str(),
            format!("axon valid_until {}", now + MAX_AGENT_VALIDITY_MS)
        );
        // But `max_validity` deliberately stops short of the boundary: the venue
        // re-checks the deadline against its own clock, so requesting the exact cap
        // makes acceptance depend on clock skew — and the rejection lands after the
        // nonce is spent.
        let safe = AgentName::max_validity("axon", now).unwrap();
        assert_ne!(safe, ok, "max_validity must not sit exactly on the cap");
        assert_eq!(
            safe.as_str(),
            format!(
                "axon valid_until {}",
                now + MAX_AGENT_VALIDITY_MS - MAX_VALIDITY_MARGIN_MS
            )
        );

        // One millisecond past it is not.
        assert!(matches!(
            AgentName::valid_until("axon", now + MAX_AGENT_VALIDITY_MS + 1, now),
            Err(SignError::AgentExpiryTooFar { .. })
        ));
        assert!(matches!(
            AgentName::valid_until("axon", now, now),
            Err(SignError::AgentExpiryNotFuture { .. })
        ));
    }

    #[test]
    fn malformed_agent_names_are_rejected() {
        // Empty, untrimmed, or already carrying the expiry token: all would make
        // the venue read a different name than the caller intended.
        for bad in ["", " ", "axon ", " axon", "axon valid_until 123"] {
            assert!(
                matches!(AgentName::new(bad), Err(SignError::BadAgentName)),
                "expected rejection of {bad:?}"
            );
            assert!(matches!(
                AgentName::valid_until(bad, 2, 1),
                Err(SignError::BadAgentName)
            ));
        }
        assert_eq!(AgentName::new("axon-live").unwrap().as_str(), "axon-live");
    }
}
