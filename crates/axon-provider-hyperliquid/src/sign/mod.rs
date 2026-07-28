//! Hyperliquid signing (ADR-0009).
//!
//! Two mutually incompatible EIP-712 schemes, deliberately kept apart:
//!
//! - **L1 actions** — orders/cancels. [`action`] builds the msgpack action hash
//!   (`connectionId`); [`eip712`] wraps it in a phantom `Agent` under the
//!   `Exchange` domain.
//! - **User-signed actions** — admin operations (agent approval, withdrawals,
//!   transfers). [`user_signed`] defines the action shapes — `approveAgent` so
//!   far — which [`eip712`] hashes directly as typed data under the
//!   `HyperliquidSignTransaction` domain.
//!
//! [`signer`] holds [`HlSigner`], the concrete key handling, with one entry point
//! per scheme so the two cannot be confused at the call site.

pub mod action;
pub mod eip712;
pub mod signer;
pub mod user_signed;

pub use action::l1_action_hash;
pub use eip712::{
    l1_signing_hash, user_signed_domain, user_signed_hash, UserSignedPayload,
    USER_SIGNED_DOMAIN_NAME,
};
pub use signer::{HlSigner, RpcSignature, SignError, SignedUserAction};
pub use user_signed::{
    hyperliquid_chain, AgentName, ApproveAgent, ApproveAgentPayload, SignatureChainId,
    UserSignedAction, APPROVE_AGENT_ENCODE_TYPE, APPROVE_AGENT_PRIMARY_TYPE, MAX_AGENT_VALIDITY_MS,
};
