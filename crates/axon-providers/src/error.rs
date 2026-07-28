//! The provider error taxonomy. Adapters map venue-specific failures into these
//! so the core and strategy never see venue error codes.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    /// The venue cannot express this request (bad TIF, unsupported order type,
    /// batch too large). Surfaced *before* hitting the wire.
    #[error("{venue} does not support: {what}")]
    Unsupported { venue: &'static str, what: String },

    /// Authentication / signing failed.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// The venue rate-limited us.
    #[error("rate limited: {0}")]
    RateLimited(String),

    /// The venue rejected the order (risk, price band, insufficient margin, …).
    #[error("order rejected: {0}")]
    Rejected(String),

    /// Transport/network error talking to the venue.
    #[error("network error: {0}")]
    Network(String),

    /// Hyperliquid's 100-nonce window is exhausted (adapter must widen/refresh).
    #[error("nonce budget exhausted")]
    NonceExhausted,

    /// The order breaks the instrument's own number format, or we do not know what
    /// that format is (ADR-0025).
    ///
    /// Its own variant rather than [`Rejected`](Self::Rejected), because it is
    /// categorically not the venue saying no: it means *our* instrument table or *our*
    /// planner is wrong, and the fix is on this side of the wire. Folded into
    /// `Rejected` it would read in a log exactly like a margin refusal, and an operator
    /// would go looking at the account.
    #[error("precision: {0}")]
    Precision(String),

    /// A path that exists in the trait but is not implemented in this phase.
    #[error("not implemented yet: {0}")]
    NotImplemented(&'static str),
}
