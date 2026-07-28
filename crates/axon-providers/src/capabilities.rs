//! Per-adapter capability descriptor. The router consults this to reject
//! impossible requests *before* the wire, so strategies stay generic and fail fast.
//!
//! **Venue-level facts only.** Tick size and lot size do not belong here even though
//! they look like they should: they are per *instrument*, they differ between two
//! symbols on the same venue, and they arrive over the network at startup. Putting
//! them here would force [`Capabilities`] to stop being `&'static` and
//! const-constructible — which is what makes a capability check free on the submit
//! path and lets an adapter declare one as a `const`. Per-instrument precision lives
//! in [`instrument`](crate::instrument) with its own table (ADR-0025).

use crate::error::ProviderError;
use crate::order::OrderRequest;
use axon_core::{OrderType, Tif};

/// How a venue governs request rate — informs the per-venue rate governor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitModel {
    /// CEX-style: weighted requests per IP.
    WeightPerIp,
    /// Hyperliquid-style: volume-gated (≈1 request per 1 USDC traded), with a
    /// higher cap for cancels.
    VolumeGated,
    /// No documented limit (simulator / backtest).
    None,
}

/// What an adapter supports. `'static` slices keep this cheap and const-friendly.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub venue: &'static str,
    pub order_types: &'static [OrderType],
    pub tifs: &'static [Tif],
    /// Max orders per batch call (Hyperliquid = 20).
    pub max_batch: u32,
    /// Does the venue have a native market order? (Hyperliquid = false — it
    /// synthesizes one as an IOC limit at deep slippage.)
    pub native_market_orders: bool,
    pub reduce_only: bool,
    pub rate_limit_model: RateLimitModel,
}

impl Capabilities {
    #[inline]
    pub fn supports_order_type(&self, t: OrderType) -> bool {
        self.order_types.contains(&t)
    }

    #[inline]
    pub fn supports_tif(&self, tif: Tif) -> bool {
        self.tifs.contains(&tif)
    }

    /// Validate a request against declared capabilities. Returns a typed
    /// [`ProviderError::Unsupported`] the router can reject on before signing.
    pub fn check(&self, req: &OrderRequest) -> Result<(), ProviderError> {
        if !self.supports_order_type(req.order_type) {
            return Err(ProviderError::Unsupported {
                venue: self.venue,
                what: format!("order type {:?}", req.order_type),
            });
        }
        if !self.supports_tif(req.tif) {
            return Err(ProviderError::Unsupported {
                venue: self.venue,
                what: format!("time-in-force {:?}", req.tif),
            });
        }
        if req.reduce_only && !self.reduce_only {
            return Err(ProviderError::Unsupported {
                venue: self.venue,
                what: "reduce_only orders".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axon_core::{Cloid, Decimal, Side, SymbolId};

    fn caps() -> Capabilities {
        Capabilities {
            venue: "test",
            order_types: &[OrderType::Limit],
            tifs: &[Tif::Gtc, Tif::Ioc, Tif::PostOnly],
            max_batch: 20,
            native_market_orders: false,
            reduce_only: true,
            rate_limit_model: RateLimitModel::VolumeGated,
        }
    }

    #[test]
    fn accepts_supported_request() {
        let req = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::from(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        assert!(caps().check(&req).is_ok());
    }

    #[test]
    fn rejects_unsupported_order_type() {
        let mut req = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::from(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        req.order_type = OrderType::Market;
        let err = caps().check(&req).unwrap_err();
        assert!(matches!(err, ProviderError::Unsupported { .. }));
    }

    #[test]
    fn rejects_unsupported_tif() {
        let mut req = OrderRequest::limit(
            SymbolId::new(1),
            Side::Buy,
            Decimal::ONE,
            Decimal::from(100),
            Tif::Gtc,
            Cloid::new(1),
        );
        req.tif = Tif::Fok;
        assert!(caps().check(&req).is_err());
    }
}
