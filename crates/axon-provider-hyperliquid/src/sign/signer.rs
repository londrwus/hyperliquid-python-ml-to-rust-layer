//! `HlSigner` — the concrete Hyperliquid signer over an `alloy` local wallet.
//!
//! The private key is normally the **agent (API) wallet** key (so the hot key
//! never holds funds), sourced from an env var — never the repo, config files, or
//! logs. `HlSigner` deliberately does **not** derive `Debug`, and the key is
//! zeroized on drop by `alloy-signer-local` (the `zeroize` feature).
//!
//! One entry point per signing scheme — [`sign_l1_action`](HlSigner::sign_l1_action)
//! and [`sign_user_signed_action`](HlSigner::sign_user_signed_action) — so the two
//! cannot be confused at the call site. The only action that needs the funded
//! **master** key is `approveAgent`, which authorizes the agent wallet in the
//! first place; that is a deliberate one-off ceremony with its own signer.

use alloy_primitives::{Address, Signature};
use alloy_signer::SignerSync;
use alloy_signer_local::PrivateKeySigner;
use serde::Serialize;

use super::action::l1_action_hash;
use super::eip712::{l1_signing_hash, user_signed_hash};
use super::user_signed::{hyperliquid_chain, UserSignedAction};

/// A signature in Hyperliquid's request shape (`{"r","s","v"}`, `v` ∈ {27, 28}).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RpcSignature {
    pub r: String,
    pub s: String,
    pub v: u64,
}

impl From<Signature> for RpcSignature {
    fn from(sig: Signature) -> Self {
        Self {
            r: format!("0x{:064x}", sig.r()),
            s: format!("0x{:064x}", sig.s()),
            // Recovery parity (0/1) → Ethereum `v` of 27/28.
            v: 27 + sig.v() as u64,
        }
    }
}

/// A signed **user-signed** action, ready to POST to `/exchange`.
///
/// `nonce` is copied out of the action itself, so the envelope's `nonce` and the
/// action's own `nonce` field are the same value by construction — the venue
/// rejects the request when they differ, and nothing in the response says so.
/// This shape also carries *only* the three keys the scheme permits: `vaultAddress`
/// and `expiresAfter` are not part of it, and `expiresAfter` in particular is
/// rejected on user-signed actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SignedUserAction<A> {
    pub action: A,
    pub nonce: u64,
    pub signature: RpcSignature,
}

/// Why signing failed.
#[derive(Debug, thiserror::Error)]
pub enum SignError {
    #[error("invalid private key")]
    BadKey,
    #[error("missing env var {0}")]
    MissingEnv(&'static str),
    #[error("action encode failed: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("signing failed: {0}")]
    Sign(#[from] alloy_signer::Error),
    /// The action declares one network and the signer is configured for the other.
    /// Caught here because `hyperliquidChain` is a *signed* field: the mismatch
    /// would only surface as an opaque venue rejection.
    #[error("action is for {action_chain} but the signer is configured for {signer_chain}")]
    NetworkMismatch {
        action_chain: &'static str,
        signer_chain: &'static str,
    },
    /// See [`AgentName`](super::user_signed::AgentName).
    #[error("agent name must be non-empty, trimmed, and must not itself contain 'valid_until'")]
    BadAgentName,
    /// Rejected locally rather than at the venue: the 180-day cap is documented,
    /// and learning it from a round trip costs a nonce.
    #[error("agent expiry {valid_until_ms} is more than 180 days after {now_ms}")]
    AgentExpiryTooFar { valid_until_ms: u64, now_ms: u64 },
    #[error("agent expiry {valid_until_ms} is not after {now_ms}")]
    AgentExpiryNotFuture { valid_until_ms: u64, now_ms: u64 },
}

/// Signs Hyperliquid actions — L1 (orders/cancels) and user-signed (admin).
pub struct HlSigner {
    wallet: PrivateKeySigner,
    is_mainnet: bool,
}

impl HlSigner {
    /// Env var the agent-wallet private key is read from.
    pub const ENV_KEY: &'static str = "AXON_HL_SECRET_KEY";

    /// Build from a hex private key (`0x`-prefixed or bare).
    pub fn from_hex(private_key: &str, is_mainnet: bool) -> Result<Self, SignError> {
        let wallet: PrivateKeySigner = private_key.trim().parse().map_err(|_| SignError::BadKey)?;
        Ok(Self { wallet, is_mainnet })
    }

    /// Build from the [`ENV_KEY`](Self::ENV_KEY) environment variable.
    pub fn from_env(is_mainnet: bool) -> Result<Self, SignError> {
        let key = std::env::var(Self::ENV_KEY).map_err(|_| SignError::MissingEnv(Self::ENV_KEY))?;
        Self::from_hex(&key, is_mainnet)
    }

    /// The agent wallet's address (used as `agent`/nonce owner, not the account).
    pub fn address(&self) -> Address {
        self.wallet.address()
    }

    pub fn is_mainnet(&self) -> bool {
        self.is_mainnet
    }

    /// Sign an L1 action, returning the Hyperliquid `{r,s,v}` signature.
    pub fn sign_l1_action(
        &self,
        action: &impl Serialize,
        nonce: u64,
        vault: Option<Address>,
        expires_after: Option<u64>,
    ) -> Result<RpcSignature, SignError> {
        let action_hash = l1_action_hash(action, nonce, vault, expires_after)?;
        let hash = l1_signing_hash(action_hash, self.is_mainnet);
        Ok(self.wallet.sign_hash_sync(&hash)?.into())
    }

    /// Sign a **user-signed** action (`approveAgent`, withdrawals, transfers).
    ///
    /// The digest is EIP-712 typed data over the action's own fields: no msgpack,
    /// no `connectionId`, no vault prefix, and no `expiresAfter` — that field is
    /// not part of this scheme. Signing an action under the other scheme's domain
    /// is the venue's #1 "invalid signature" cause, hence a separate entry point
    /// rather than a flag on [`sign_l1_action`](Self::sign_l1_action).
    ///
    /// The envelope nonce and the domain chain id are read off `action`, so the
    /// caller cannot desynchronize them from what was signed.
    pub fn sign_user_signed_action<A: UserSignedAction>(
        &self,
        action: &A,
    ) -> Result<RpcSignature, SignError> {
        if action.is_mainnet() != self.is_mainnet {
            return Err(SignError::NetworkMismatch {
                action_chain: hyperliquid_chain(action.is_mainnet()),
                signer_chain: hyperliquid_chain(self.is_mainnet),
            });
        }
        let hash = user_signed_hash(&action.eip712_payload(), action.signature_chain_id().get());
        Ok(self.wallet.sign_hash_sync(&hash)?.into())
    }

    /// Sign `action` and wrap it in the `/exchange` envelope this scheme uses,
    /// with the envelope nonce taken from the action itself.
    pub fn user_signed_request<A: UserSignedAction>(
        &self,
        action: A,
    ) -> Result<SignedUserAction<A>, SignError> {
        let signature = self.sign_user_signed_action(&action)?;
        Ok(SignedUserAction {
            nonce: action.nonce(),
            action,
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::action::l1_action_hash;
    use crate::sign::eip712::l1_signing_hash;
    use crate::sign::user_signed::{AgentName, ApproveAgent, SignatureChainId};

    // Hardhat/anvil well-known account #0 — key ↔ address is a public vector.
    const KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const ADDR: &str = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266";
    const AGENT: &str = "0x1234567890abcdef1234567890abcdef12345678";

    #[derive(Serialize)]
    struct Noop {
        #[serde(rename = "type")]
        kind: &'static str,
    }

    fn agent() -> Address {
        AGENT.parse().expect("literal address")
    }

    #[test]
    fn derives_the_known_address_from_the_key() {
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        assert_eq!(signer.address(), ADDR.parse::<Address>().unwrap());
    }

    #[test]
    fn signature_recovers_to_the_signer_address() {
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        let action = Noop { kind: "noop" };
        let nonce = 1_700_000_000_000;

        // Re-derive the exact digest that was signed.
        let hash = l1_signing_hash(l1_action_hash(&action, nonce, None, None).unwrap(), false);
        let sig = signer.wallet.sign_hash_sync(&hash).unwrap();
        let recovered = sig.recover_address_from_prehash(&hash).unwrap();
        assert_eq!(recovered, signer.address());

        // And the public API produces a well-formed {r,s,v}.
        let rpc = signer.sign_l1_action(&action, nonce, None, None).unwrap();
        assert!(rpc.r.starts_with("0x") && rpc.r.len() == 66);
        assert!(rpc.v == 27 || rpc.v == 28);
    }

    #[test]
    fn bad_key_is_rejected() {
        assert!(matches!(
            HlSigner::from_hex("not-a-key", false),
            Err(SignError::BadKey)
        ));
    }

    #[test]
    fn user_signed_signature_recovers_to_the_signer_address() {
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        let action = ApproveAgent::named(agent(), AgentName::new("axon").unwrap(), 7, false);

        // Re-derive the exact digest and prove the recovered signer is us — the
        // same proof technique as the L1 test above.
        let hash = user_signed_hash(&action.eip712_payload(), action.signature_chain_id().get());
        let sig = signer.wallet.sign_hash_sync(&hash).unwrap();
        assert_eq!(
            sig.recover_address_from_prehash(&hash).unwrap(),
            signer.address()
        );

        let rpc = signer.sign_user_signed_action(&action).unwrap();
        assert_eq!(rpc, RpcSignature::from(sig));
        assert!(rpc.v == 27 || rpc.v == 28);
    }

    #[test]
    fn the_two_schemes_produce_different_signatures() {
        // The L1 `Exchange`/1337 domain and the user-signed
        // `HyperliquidSignTransaction` domain must never coincide.
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        let action = ApproveAgent::unnamed(agent(), 7, false);
        let user = signer.sign_user_signed_action(&action).unwrap();
        let l1 = signer
            .sign_l1_action(&action, action.nonce(), None, None)
            .unwrap();
        assert_ne!(user, l1);
    }

    #[test]
    fn mainnet_and_testnet_actions_sign_differently() {
        let mainnet = HlSigner::from_hex(KEY, true).unwrap();
        let testnet = HlSigner::from_hex(KEY, false).unwrap();
        let a = mainnet
            .sign_user_signed_action(&ApproveAgent::unnamed(agent(), 7, true))
            .unwrap();
        let b = testnet
            .sign_user_signed_action(&ApproveAgent::unnamed(agent(), 7, false))
            .unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn signature_chain_id_changes_the_signature() {
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        let base = ApproveAgent::unnamed(agent(), 7, false);
        let other = base
            .clone()
            .with_signature_chain_id(SignatureChainId::ARBITRUM_SEPOLIA);
        assert_ne!(
            signer.sign_user_signed_action(&base).unwrap(),
            signer.sign_user_signed_action(&other).unwrap()
        );
    }

    #[test]
    fn a_network_mismatch_is_refused_before_signing() {
        let testnet = HlSigner::from_hex(KEY, false).unwrap();
        assert!(matches!(
            testnet.sign_user_signed_action(&ApproveAgent::unnamed(agent(), 7, true)),
            Err(SignError::NetworkMismatch { .. })
        ));
    }

    #[test]
    fn request_envelope_reuses_the_actions_nonce_and_omits_l1_fields() {
        let signer = HlSigner::from_hex(KEY, false).unwrap();
        let nonce = 1_784_976_929_583;
        let req = signer
            .user_signed_request(ApproveAgent::unnamed(agent(), nonce, false))
            .unwrap();
        assert_eq!(req.nonce, nonce);

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["nonce"], json["action"]["nonce"]);
        assert_eq!(json["action"]["type"], "approveAgent");
        assert!(json["signature"]["r"].as_str().unwrap().starts_with("0x"));
        // `expiresAfter` is not supported by this scheme; `vaultAddress` is not
        // part of it either.
        assert!(json.get("expiresAfter").is_none());
        assert!(json.get("vaultAddress").is_none());
    }
}
