//! Normalized order enums. Venue-specific encodings (e.g. Hyperliquid's synthetic
//! market order = IOC-at-slippage) are the adapter's job to map — these stay generic.

use serde::{Deserialize, Serialize};

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// `+1` for buy, `-1` for sell — handy for signed position math.
    #[inline]
    pub const fn sign(self) -> i8 {
        match self {
            Side::Buy => 1,
            Side::Sell => -1,
        }
    }

    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

/// Normalized order type. Adapters map these to venue reality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
}

/// Time-in-force. `PostOnly` (a.k.a. ALO) rejects if it would take liquidity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tif {
    /// Good-till-cancel.
    Gtc,
    /// Immediate-or-cancel.
    Ioc,
    /// Post-only / add-liquidity-only.
    PostOnly,
    /// Fill-or-kill.
    Fok,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_sign_and_opposite() {
        assert_eq!(Side::Buy.sign(), 1);
        assert_eq!(Side::Sell.sign(), -1);
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
    }

    #[test]
    fn tif_serde_roundtrip() {
        let j = serde_json::to_string(&Tif::PostOnly).unwrap();
        assert_eq!(j, "\"post_only\"");
        let back: Tif = serde_json::from_str(&j).unwrap();
        assert_eq!(back, Tif::PostOnly);
    }
}
