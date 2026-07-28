# ADR-0009 — Hyperliquid signing: `alloy` crypto + hand-rolled msgpack action encoding

**Status:** Accepted · **Date:** 2026-07-24

## Context

Phase 3 (`docs/08-roadmap.md`) is the execution path, and its foundation is **signing** —
Hyperliquid authenticates trading with wallet signatures, not API keys. Getting it subtly wrong
is a money-critical, silent failure (`invalid signature`, or worse, a mis-signed order), so this
is the highest-value part to get provably right, offline, first.

The mechanics (verified against the official Python SDK `signing.py`): an **L1 action** (order,
cancel) is MessagePack-serialized, then `nonce` (8-byte big-endian), a vault prefix, and an
optional expiry are appended and `keccak256`'d into a `connectionId`. That hash is wrapped in a
phantom `Agent(string source, bytes32 connectionId)` and signed as EIP-712 typed data under a
fixed `Exchange` domain (chainId 1337, verifyingContract 0x0); `source` is `"a"` (mainnet) or
`"b"` (testnet). Two open choices had to be made:

1. **Crypto stack.** The research doc (`docs/research/hyperliquid-execution.md`) noted the official
   Rust SDK was on the deprecated `ethers`. That is now stale: the **official SDK has migrated to
   `alloy`** (`alloy::{primitives::keccak256, sol_types, dyn_abi}`). Options: adopt `alloy`,
   hand-roll with `k256` + `sha3`, or depend on a HL SDK.
2. **Key source / wallet model.** Where the private key lives, and whether we sign with the main
   account key or an agent (API) wallet.

## Decision

1. **Use `alloy` for the cryptography** — `keccak256`, EIP-712 domain/struct hashing
   (`alloy-sol-types` `sol!` + `eip712_domain!`), and the local secp256k1 wallet
   (`alloy-signer-local`, `zeroize` on). It is the audited, ecosystem-standard stack the official
   SDK itself uses, minimizing crypto risk on a money path. We **hand-roll only the HL-specific
   msgpack action encoding** (`rmp-serde`, `to_vec_named` — field order matters), which is glue,
   not cryptography. All of it is wrapped behind our own `HlSigner`; no SDK types leak upward.
2. **Env-var key + agent (API) wallet.** The signing key is the **agent wallet** key (so the hot
   key never holds funds) and is read from the `AXON_HL_SECRET_KEY` env var — never the repo,
   config files, or logs. `HlSigner` does not derive `Debug`, and `alloy-signer-local`'s `zeroize`
   wipes the key on drop. **Testnet before mainnet, always.**

Landed offline-first: `sign/` (action hash, EIP-712, `HlSigner`), `nonce.rs`
(strictly-monotonic ms nonces), and `encode.rs` (order/cancel → wire action) are unit-tested with
zero network — including a Hardhat known key↔address vector, a byte-exact msgpack layout test, and
a sign→recover-address round-trip. The live testnet `ExecutionClient` comes next increment.

## Consequences

- **+** Audited, standard crypto; the EIP-712 + ECDSA path is proven self-consistent offline
  (address recovery) before any network call, so live bring-up is a formality.
- **+** Minimal HL-specific surface (just the msgpack encoding) is ours to own and test; the rest
  is `alloy`. Key never holds funds (agent wallet) and never touches disk/logs.
- **+** One vocabulary boundary: `HlSigner` behind the adapter; strategies/core never see keys.
- **−** `alloy` is a large dependency tree — accepted, confined to the execution edge crate.
- **−** The existing `axon_providers::Signer::sign(&[u8]) -> Vec<u8>` seam is too low-level for
  HL's structured L1 flow; `HlSigner` is a concrete richer type for now. Reconciling/extending the
  port `Signer` trait is deferred to when the async `ExecutionClient` lands (same pattern as
  `MarketData::subscribe` in [ADR-0008](0008-market-data-bus-and-ws-ingest.md)).
- **−** True end-to-end signature acceptance is only *proven* by a live testnet order (next
  increment); offline confidence is maximized to de-risk that.

See [ADR-0004](0004-provider-abstraction-layer.md), [ADR-0005](0005-fp32-no-quantization.md),
[ADR-0008](0008-market-data-bus-and-ws-ingest.md), `docs/research/hyperliquid-execution.md`.
