//! Core error taxonomy. Kept deliberately small; venue/IPC/risk crates define
//! their own richer errors and convert into or out of these where they meet the core.

use thiserror::Error;

/// Errors originating in the deterministic core.
#[derive(Debug, Error)]
pub enum CoreError {
    /// An invariant the core relies on was violated (a bug, surfaced loudly).
    #[error("core invariant violated: {0}")]
    Invariant(String),

    /// A value crossing into the core failed validation.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
