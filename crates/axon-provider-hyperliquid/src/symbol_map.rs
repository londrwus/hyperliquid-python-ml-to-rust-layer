//! `SymbolMap` — the venue-specific translation between Hyperliquid coin names
//! (`"BTC"`, `"ETH"`, …) and Axon's venue-independent [`SymbolId`].
//!
//! Per the execution research (`docs/research/hyperliquid-execution.md`), a perp's
//! asset index is its position in `meta.universe`; spot assets are addressed as
//! `10000 + spotMeta index`. We mirror that so a `SymbolId` built from the perp
//! universe lines up with the on-wire asset index, and spot ids never collide
//! with perp ids. This mapping is the *only* place coin strings appear — above it
//! everything speaks `SymbolId`.

use std::collections::HashMap;

use axon_core::SymbolId;

/// Spot assets are numbered from this offset to avoid colliding with perp indices.
pub const SPOT_OFFSET: u32 = 10_000;

/// Bidirectional coin ↔ `SymbolId` map for one venue.
#[derive(Debug, Clone, Default)]
pub struct SymbolMap {
    by_coin: HashMap<String, SymbolId>,
    by_id: HashMap<SymbolId, String>,
}

impl SymbolMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a coin ↔ id pair (last write wins for either direction).
    pub fn insert(&mut self, coin: impl Into<String>, id: SymbolId) {
        let coin = coin.into();
        self.by_coin.insert(coin.clone(), id);
        self.by_id.insert(id, coin);
    }

    /// Register a spot asset by its `spotMeta` index (stored as `SPOT_OFFSET + idx`).
    pub fn insert_spot(&mut self, coin: impl Into<String>, spot_index: u32) {
        self.insert(coin, SymbolId::new(SPOT_OFFSET + spot_index));
    }

    /// Build a map from a perp universe: the coin at position `i` gets `SymbolId(i)`,
    /// matching Hyperliquid's `meta.universe` asset indexing.
    pub fn from_perps<I, S>(coins: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut m = Self::new();
        for (i, c) in coins.into_iter().enumerate() {
            m.insert(c, SymbolId::new(i as u32));
        }
        m
    }

    /// Resolve a coin name to its `SymbolId`.
    pub fn id(&self, coin: &str) -> Option<SymbolId> {
        self.by_coin.get(coin).copied()
    }

    /// Resolve a `SymbolId` back to its coin name.
    pub fn coin(&self, id: SymbolId) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_coin.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_coin.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perp_universe_indices_match_positions_and_round_trip() {
        let m = SymbolMap::from_perps(["BTC", "ETH", "SOL"]);
        assert_eq!(m.id("BTC"), Some(SymbolId::new(0)));
        assert_eq!(m.id("ETH"), Some(SymbolId::new(1)));
        assert_eq!(m.id("SOL"), Some(SymbolId::new(2)));
        assert_eq!(m.coin(SymbolId::new(2)), Some("SOL"));
        assert_eq!(m.id("DOGE"), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn spot_assets_are_offset_and_do_not_collide_with_perps() {
        let mut m = SymbolMap::from_perps(["BTC"]); // perp BTC → 0
        m.insert_spot("PURR", 0); // spot 0 → 10000
        assert_eq!(m.id("BTC"), Some(SymbolId::new(0)));
        assert_eq!(m.id("PURR"), Some(SymbolId::new(SPOT_OFFSET)));
        assert_ne!(m.id("BTC"), m.id("PURR"));
        assert_eq!(m.coin(SymbolId::new(SPOT_OFFSET)), Some("PURR"));
    }
}
