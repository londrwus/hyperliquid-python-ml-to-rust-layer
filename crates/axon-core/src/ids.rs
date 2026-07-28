//! Newtype identifiers. Wrapping raw integers stops us from accidentally passing
//! a symbol id where an order id belongs, and documents intent at call sites.

use core::fmt;
use serde::{Deserialize, Serialize};

/// Canonical, venue-independent symbol id. Resolved from a venue's native symbol
/// via that adapter's `SymbolMap`. Matches `Signal.symbol_id` on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SymbolId(pub u32);

impl SymbolId {
    #[inline]
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sym#{}", self.0)
    }
}

/// Client order id (128-bit) — makes every request idempotent and reconcilable
/// (see `docs/04-provider-abstraction.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cloid(pub u128);

impl Cloid {
    #[inline]
    pub const fn new(v: u128) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl fmt::Display for Cloid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cloid:{:#034x}", self.0)
    }
}

/// Venue-assigned order id (Hyperliquid `oid`, CEX order id, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl OrderId {
    #[inline]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "oid#{}", self.0)
    }
}
