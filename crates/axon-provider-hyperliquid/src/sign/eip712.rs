//! EIP-712 hashing for Hyperliquid's **two** signing schemes.
//!
//! - **L1 actions** (orders/cancels): the action hash (`connectionId`) is wrapped
//!   in a phantom `Agent` and hashed under the fixed `Exchange` domain (chainId
//!   1337, verifyingContract 0x0), per the official SDK / `signing.py`. The
//!   `source` field selects mainnet (`"a"`) vs testnet (`"b"`).
//! - **User-signed actions** (`approveAgent`, withdrawals, transfers): the
//!   action's own fields *are* the message, hashed directly under the
//!   `HyperliquidSignTransaction` domain whose chainId is the real EVM chain id
//!   the action declares in `signatureChainId`. There is no msgpack hash, no
//!   phantom agent, and the network rides in the message's `hyperliquidChain`
//!   field rather than in `source`.
//!
//! The resulting 32-byte digest is what the wallet signs. Hashing an action under
//! the wrong domain is the venue's #1 "invalid signature" cause — it produces a
//! perfectly well-formed signature that recovers to the wrong address — so the
//! two domains live side by side here with a test asserting their separators
//! differ.

use alloy_primitives::{keccak256, Address, Keccak256, B256};
use alloy_sol_types::{eip712_domain, sol, Eip712Domain, SolStruct};

sol! {
    /// The phantom-agent struct signed for every L1 action.
    struct Agent {
        string source;
        bytes32 connectionId;
    }
}

/// The domain every L1 action is signed under. `chainId` 1337 is a venue
/// sentinel, not a real network — the mainnet/testnet split is carried by the
/// phantom agent's `source` instead.
fn l1_domain() -> Eip712Domain {
    eip712_domain! {
        name: "Exchange",
        version: "1",
        chain_id: 1337u64,
        verifying_contract: Address::ZERO,
    }
}

/// The 32-byte EIP-712 signing hash for an L1 action, given its action hash
/// (`connectionId`). `is_mainnet` picks the phantom-agent `source` (`"a"`/`"b"`).
pub fn l1_signing_hash(action_hash: B256, is_mainnet: bool) -> B256 {
    let agent = Agent {
        source: if is_mainnet { "a" } else { "b" }.to_string(),
        connectionId: action_hash,
    };
    agent.eip712_signing_hash(&l1_domain())
}

/// The domain `name` shared by *every* user-signed action.
pub const USER_SIGNED_DOMAIN_NAME: &str = "HyperliquidSignTransaction";

/// The domain a user-signed action is signed under.
///
/// `chain_id` must be `int(action.signatureChainId, 16)`: the venue rebuilds this
/// domain from the hex string it received in the action, so if the two disagree
/// it recovers a different signer and rejects the request. Which chain id is used
/// does not otherwise matter — nothing on-chain verifies these signatures; it
/// only changes what a hardware wallet displays.
pub fn user_signed_domain(chain_id: u64) -> Eip712Domain {
    eip712_domain! {
        name: USER_SIGNED_DOMAIN_NAME,
        version: "1",
        chain_id: chain_id,
        verifying_contract: Address::ZERO,
    }
}

/// The EIP-712 struct of a user-signed action, plus the venue's own type names.
///
/// `sol!` cannot supply those names: Hyperliquid's `primaryType` is
/// `"HyperliquidTransaction:ApproveAgent"`, and `:` is not legal in a Rust
/// identifier — so the macro-derived `encodeType` carries the wrong struct name
/// and therefore the wrong `typeHash`. Implementors keep `sol!` for `encodeData`
/// (which handles the string→keccak and address/uint word padding) and hand over
/// the type string themselves, asserted byte-for-byte in tests.
pub trait UserSignedPayload: SolStruct {
    /// The EIP-712 `primaryType`, e.g. `"HyperliquidTransaction:ApproveAgent"`.
    const PRIMARY_TYPE: &'static str;
    /// The byte-exact `encodeType` string: [`PRIMARY_TYPE`](Self::PRIMARY_TYPE)
    /// followed by `(type name,...)` — no spaces after the commas.
    const ENCODE_TYPE: &'static str;

    /// `hashStruct` = `keccak256(typeHash ‖ encodeData)`.
    ///
    /// Identical to [`SolStruct::eip712_hash_struct`] except that the `typeHash`
    /// comes from [`ENCODE_TYPE`](Self::ENCODE_TYPE) rather than from the Rust
    /// struct's name.
    fn hash_struct(&self) -> B256 {
        let mut hasher = Keccak256::new();
        hasher.update(keccak256(Self::ENCODE_TYPE.as_bytes()));
        hasher.update(self.eip712_encode_data());
        hasher.finalize()
    }
}

/// The 32-byte EIP-712 signing hash for a user-signed action:
/// `keccak256(0x1901 ‖ domainSeparator ‖ hashStruct(payload))`.
///
/// Note what is *absent* compared with [`l1_signing_hash`]: no msgpack, no
/// action hash, no vault prefix, and no `expiresAfter` — that field is not part
/// of this scheme at all.
pub fn user_signed_hash<P: UserSignedPayload>(payload: &P, chain_id: u64) -> B256 {
    let mut digest_input = [0u8; 2 + 32 + 32];
    digest_input[0] = 0x19;
    digest_input[1] = 0x01;
    digest_input[2..34].copy_from_slice(user_signed_domain(chain_id).separator().as_slice());
    digest_input[34..66].copy_from_slice(payload.hash_struct().as_slice());
    keccak256(digest_input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;

    /// Arbitrum One — the chain id the docs' `approveAgent` example declares.
    const CHAIN_ID: u64 = 42_161;

    sol! {
        struct Probe {
            string a;
        }
    }

    impl UserSignedPayload for Probe {
        const PRIMARY_TYPE: &'static str = "HyperliquidTransaction:Probe";
        const ENCODE_TYPE: &'static str = "HyperliquidTransaction:Probe(string a)";
    }

    #[test]
    fn signing_hash_is_deterministic_and_source_sensitive() {
        let action = B256::with_last_byte(0x42);
        let mainnet = l1_signing_hash(action, true);
        let mainnet_again = l1_signing_hash(action, true);
        let testnet = l1_signing_hash(action, false);
        assert_eq!(mainnet, mainnet_again);
        // Mainnet ("a") and testnet ("b") must produce different digests.
        assert_ne!(mainnet, testnet);
    }

    #[test]
    fn user_signed_domain_fields_are_the_venue_values() {
        let domain = user_signed_domain(CHAIN_ID);
        assert_eq!(domain.name.as_deref(), Some("HyperliquidSignTransaction"));
        assert_eq!(domain.version.as_deref(), Some("1"));
        // chainId is the *real* EVM chain id, i.e. int(signatureChainId, 16).
        assert_eq!(domain.chain_id, Some(U256::from(0xa4b1u64)));
        assert_eq!(domain.verifying_contract, Some(Address::ZERO));
    }

    #[test]
    fn user_signed_domain_differs_from_the_l1_domain() {
        // The whole reason this module keeps both: an action hashed under the
        // wrong domain still signs, it just recovers to the wrong address.
        assert_ne!(
            l1_domain().separator(),
            user_signed_domain(CHAIN_ID).separator()
        );
        assert_ne!(l1_domain().name, user_signed_domain(CHAIN_ID).name);
        // ...and 1337 is never a user-signed chain id.
        assert_ne!(
            l1_domain().separator(),
            user_signed_domain(1337).separator(),
            "same chainId, different name — separators must still differ"
        );
    }

    #[test]
    fn user_signed_domain_is_chain_id_sensitive() {
        // Arbitrum One vs Arbitrum Sepolia: mismatching the declared
        // signatureChainId changes the digest, hence the recovered signer.
        assert_ne!(
            user_signed_domain(42_161).separator(),
            user_signed_domain(421_614).separator()
        );
    }

    #[test]
    fn hash_struct_uses_the_venue_type_name_not_the_rust_one() {
        let probe = Probe { a: "x".to_string() };
        // `sol!` would hash typeHash("Probe(string a)"); the venue wants
        // typeHash("HyperliquidTransaction:Probe(string a)").
        assert_ne!(probe.hash_struct(), probe.eip712_hash_struct());
        let mut expected = Keccak256::new();
        expected.update(keccak256(Probe::ENCODE_TYPE.as_bytes()));
        expected.update(probe.eip712_encode_data());
        assert_eq!(probe.hash_struct(), expected.finalize());
    }

    #[test]
    fn user_signed_hash_is_the_1901_digest() {
        let probe = Probe { a: "x".to_string() };
        let mut expected = [0u8; 66];
        expected[0] = 0x19;
        expected[1] = 0x01;
        expected[2..34].copy_from_slice(user_signed_domain(CHAIN_ID).separator().as_slice());
        expected[34..66].copy_from_slice(probe.hash_struct().as_slice());
        assert_eq!(user_signed_hash(&probe, CHAIN_ID), keccak256(expected));
        assert_ne!(
            user_signed_hash(&probe, CHAIN_ID),
            user_signed_hash(&probe, 421_614)
        );
    }
}
