//! # axon-marketdata
//!
//! Order-book maintenance and normalized market-data consumption. Prices and
//! sizes are fixed-point (`Decimal`), never `f64`. This crate ships a correct L2
//! [`OrderBook`] plus the [`MarketDataProcessor`] — the core-side
//! [`EventHandler`](axon_core::EventHandler) that maintains books and a
//! latest-value cache from the normalized event stream. The provider-side WS
//! ingest that *produces* those events lives in `axon-provider-hyperliquid`.

#![deny(unsafe_code)]

pub mod processor;
pub use processor::MarketDataProcessor;

use axon_core::{Decimal, SymbolId};
use std::collections::BTreeMap;

/// A price-aggregated L2 order book for one instrument.
///
/// Levels are kept in `BTreeMap`s keyed by price, so best bid is the max key and
/// best ask is the min key. Setting a level to size `0` removes it.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol_id: SymbolId,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
}

impl OrderBook {
    pub fn new(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    /// Set (or, with `size == 0`, remove) a bid level.
    pub fn set_bid(&mut self, price: Decimal, size: Decimal) {
        if size.is_zero() {
            self.bids.remove(&price);
        } else {
            self.bids.insert(price, size);
        }
    }

    /// Set (or, with `size == 0`, remove) an ask level.
    pub fn set_ask(&mut self, price: Decimal, size: Decimal) {
        if size.is_zero() {
            self.asks.remove(&price);
        } else {
            self.asks.insert(price, size);
        }
    }

    /// Best (highest) bid as `(price, size)`.
    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.iter().next_back().map(|(p, s)| (*p, *s))
    }

    /// Best (lowest) ask as `(price, size)`.
    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.iter().next().map(|(p, s)| (*p, *s))
    }

    /// Mid price, if both sides are populated.
    pub fn mid(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some((b + a) / Decimal::TWO),
            _ => None,
        }
    }

    /// Ask − bid spread, if both sides are populated.
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some(a - b),
            _ => None,
        }
    }

    pub fn bid_levels(&self) -> usize {
        self.bids.len()
    }

    pub fn ask_levels(&self) -> usize {
        self.asks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn tracks_best_levels_mid_and_spread() {
        let mut b = OrderBook::new(SymbolId::new(1));
        b.set_bid(dec!(100), dec!(2));
        b.set_bid(dec!(99), dec!(5));
        b.set_ask(dec!(101), dec!(3));
        b.set_ask(dec!(102), dec!(1));

        assert_eq!(b.best_bid(), Some((dec!(100), dec!(2))));
        assert_eq!(b.best_ask(), Some((dec!(101), dec!(3))));
        assert_eq!(b.mid(), Some(dec!(100.5)));
        assert_eq!(b.spread(), Some(dec!(1)));
    }

    #[test]
    fn zero_size_removes_level() {
        let mut b = OrderBook::new(SymbolId::new(1));
        b.set_bid(dec!(100), dec!(2));
        assert_eq!(b.bid_levels(), 1);
        b.set_bid(dec!(100), dec!(0));
        assert_eq!(b.bid_levels(), 0);
        assert_eq!(b.best_bid(), None);
    }

    #[test]
    fn empty_book_has_no_mid() {
        let b = OrderBook::new(SymbolId::new(1));
        assert_eq!(b.mid(), None);
        assert_eq!(b.spread(), None);
    }
}
