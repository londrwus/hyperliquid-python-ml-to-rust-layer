//! The Hyperliquid **L1 action hash** — the `connectionId` fed into the phantom
//! agent (`docs/research/hyperliquid-execution.md`, ADR-0009).
//!
//! Ground truth (official Python SDK `signing.py`):
//! `keccak256( msgpack(action) ++ nonce_be8 ++ vault_prefix ++ expires )`, where
//! - `msgpack(action)` is a **named map**, field order significant (so our action
//!   structs must declare fields in the venue's order);
//! - `nonce_be8` is the nonce as 8 big-endian bytes;
//! - `vault_prefix` is `0x00` (no vault) or `0x01 ++ 20-byte address`;
//! - `expires` is absent, or `0x00 ++ expires_after` as 8 big-endian bytes.

use alloy_primitives::{keccak256, Address, B256};
use serde::Serialize;

/// Build the exact byte string that gets hashed. Split out from [`l1_action_hash`]
/// so tests can assert the layout byte-for-byte (the msgpack encoding + appends
/// are the fiddly, easy-to-get-wrong part).
pub(crate) fn l1_action_payload(
    action: &impl Serialize,
    nonce: u64,
    vault: Option<Address>,
    expires_after: Option<u64>,
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    // `to_vec_named` → msgpack **map** with string keys (positional `to_vec` would
    // encode arrays, which the venue does not expect).
    let mut bytes = rmp_serde::to_vec_named(action)?;
    bytes.extend_from_slice(&nonce.to_be_bytes());
    match vault {
        None => bytes.push(0x00),
        Some(addr) => {
            bytes.push(0x01);
            bytes.extend_from_slice(addr.as_slice());
        }
    }
    if let Some(exp) = expires_after {
        bytes.push(0x00);
        bytes.extend_from_slice(&exp.to_be_bytes());
    }
    Ok(bytes)
}

/// The L1 action hash (`connectionId`). `vault` is the sub-account/vault address
/// if trading on its behalf (else `None`); `expires_after` is an optional
/// ms-timestamp deadline.
pub fn l1_action_hash(
    action: &impl Serialize,
    nonce: u64,
    vault: Option<Address>,
    expires_after: Option<u64>,
) -> Result<B256, rmp_serde::encode::Error> {
    Ok(keccak256(l1_action_payload(
        action,
        nonce,
        vault,
        expires_after,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Noop {
        #[serde(rename = "type")]
        kind: &'static str,
    }

    #[test]
    fn payload_layout_is_byte_exact() {
        // msgpack of {"type":"noop"} is a fixmap(1): 0x81, key fixstr(4) "type",
        // value fixstr(4) "noop"; then nonce=1 big-endian, then the 0x00 no-vault
        // marker. Locking these bytes pins the whole encoding.
        let payload = l1_action_payload(&Noop { kind: "noop" }, 1, None, None).unwrap();
        let expected: &[u8] = &[
            0x81, // fixmap, 1 pair
            0xa4, b't', b'y', b'p', b'e', // "type"
            0xa4, b'n', b'o', b'o', b'p', // "noop"
            0, 0, 0, 0, 0, 0, 0, 1,    // nonce = 1, big-endian
            0x00, // no vault
        ];
        assert_eq!(payload, expected);
    }

    #[test]
    fn vault_prefix_and_expires_are_appended() {
        let vault = Address::with_last_byte(0xAB);
        let no_vault = l1_action_payload(&Noop { kind: "noop" }, 1, None, None).unwrap();
        let with_vault = l1_action_payload(&Noop { kind: "noop" }, 1, Some(vault), None).unwrap();
        // vault adds a 0x01 marker + 20 address bytes in place of the single 0x00.
        assert_eq!(with_vault.len(), no_vault.len() - 1 + 1 + 20);
        assert_eq!(with_vault[no_vault.len() - 1], 0x01);
        assert_eq!(&with_vault[no_vault.len()..], vault.as_slice());

        let with_exp = l1_action_payload(&Noop { kind: "noop" }, 1, None, Some(7)).unwrap();
        assert_eq!(with_exp.len(), no_vault.len() + 1 + 8);
        assert_eq!(with_exp[no_vault.len()], 0x00);
        assert_eq!(&with_exp[no_vault.len() + 1..], &7u64.to_be_bytes());
    }

    #[test]
    fn hash_is_deterministic_and_nonce_sensitive() {
        let h1 = l1_action_hash(&Noop { kind: "noop" }, 1, None, None).unwrap();
        let h1b = l1_action_hash(&Noop { kind: "noop" }, 1, None, None).unwrap();
        let h2 = l1_action_hash(&Noop { kind: "noop" }, 2, None, None).unwrap();
        assert_eq!(h1, h1b);
        assert_ne!(h1, h2);
    }
}
