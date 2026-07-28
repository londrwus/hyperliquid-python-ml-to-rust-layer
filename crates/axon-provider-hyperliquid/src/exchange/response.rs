//! Pure parsing of Hyperliquid `/exchange` responses → normalized outcomes.
//!
//! Offline-testable: the venue's JSON is decoded here with no network, so the
//! mapping to our types is unit-tested against captured response shapes. The
//! async client ([`super::ExchangeClient`]) turns these outcomes into
//! [`OrderAck`](axon_providers::OrderAck) / [`CancelAck`](axon_providers::CancelAck).
//!
//! Envelope: `{ status:"ok"|"err", response: … }`. On `"ok"`, `response.data.statuses`
//! holds one item per submitted order/cancel, **in input order**.

use axon_core::OrderId;
use axon_providers::ProviderError;
use serde::Deserialize;
use serde_json::Value;

/// Outcome of a single submitted order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderOutcome {
    Resting { oid: OrderId },
    Filled { oid: OrderId },
    Rejected { reason: String },
}

/// Outcome of a single cancel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelOutcome {
    Success,
    Rejected { reason: String },
}

#[derive(Deserialize)]
struct Envelope {
    status: String,
    response: Value,
}

#[derive(Deserialize)]
struct OkResponse {
    #[serde(rename = "type")]
    _kind: String,
    data: StatusData,
}

#[derive(Deserialize)]
struct StatusData {
    statuses: Vec<Value>,
}

// Externally-tagged: {"resting":{"oid":N}} / {"filled":{…}} / {"error":"…"}.
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum OrderStatusItem {
    Resting {
        oid: u64,
    },
    Filled {
        oid: u64,
        #[serde(rename = "totalSz")]
        _total_sz: String,
        #[serde(rename = "avgPx")]
        _avg_px: String,
    },
    Error(String),
}

// Untagged: the bare string "success" or {"error":"…"}.
#[derive(Deserialize)]
#[serde(untagged)]
enum CancelStatusItem {
    Ok(String),
    Err { error: String },
}

/// Pull `response.data.statuses` from an `"ok"` envelope; a top-level `"err"`
/// becomes a [`ProviderError::Rejected`] carrying the venue message.
fn extract_statuses(value: &Value) -> Result<Vec<Value>, ProviderError> {
    let env: Envelope = serde_json::from_value(value.clone())
        .map_err(|e| ProviderError::Rejected(format!("malformed /exchange envelope: {e}")))?;
    if env.status != "ok" {
        let msg = env
            .response
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| env.response.to_string());
        return Err(ProviderError::Rejected(msg));
    }
    let ok: OkResponse = serde_json::from_value(env.response)
        .map_err(|e| ProviderError::Rejected(format!("malformed /exchange response: {e}")))?;
    Ok(ok.data.statuses)
}

/// Parse an order-action response into one [`OrderOutcome`] per submitted order.
pub fn parse_order_outcomes(value: &Value) -> Result<Vec<OrderOutcome>, ProviderError> {
    extract_statuses(value)?
        .into_iter()
        .map(|s| {
            let item: OrderStatusItem = serde_json::from_value(s)
                .map_err(|e| ProviderError::Rejected(format!("bad order status: {e}")))?;
            Ok(match item {
                OrderStatusItem::Resting { oid } => OrderOutcome::Resting {
                    oid: OrderId::new(oid),
                },
                OrderStatusItem::Filled { oid, .. } => OrderOutcome::Filled {
                    oid: OrderId::new(oid),
                },
                OrderStatusItem::Error(reason) => OrderOutcome::Rejected { reason },
            })
        })
        .collect()
}

/// Parse a cancel-action response into one [`CancelOutcome`] per submitted cancel.
pub fn parse_cancel_outcomes(value: &Value) -> Result<Vec<CancelOutcome>, ProviderError> {
    extract_statuses(value)?
        .into_iter()
        .map(|s| {
            let item: CancelStatusItem = serde_json::from_value(s)
                .map_err(|e| ProviderError::Rejected(format!("bad cancel status: {e}")))?;
            Ok(match item {
                CancelStatusItem::Ok(s) if s == "success" => CancelOutcome::Success,
                // Any other bare string is an unexpected status — treat as a rejection.
                CancelStatusItem::Ok(other) => CancelOutcome::Rejected { reason: other },
                CancelStatusItem::Err { error } => CancelOutcome::Rejected { reason: error },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_resting_and_filled_and_rejected_orders() {
        let resp = json!({
            "status": "ok",
            "response": { "type": "order", "data": { "statuses": [
                { "resting": { "oid": 77738308u64 } },
                { "filled": { "oid": 77747314u64, "totalSz": "0.02", "avgPx": "1891.4" } },
                { "error": "Order must have minimum value of $10." }
            ]}}
        });
        let outcomes = parse_order_outcomes(&resp).unwrap();
        assert_eq!(
            outcomes,
            vec![
                OrderOutcome::Resting {
                    oid: OrderId::new(77738308)
                },
                OrderOutcome::Filled {
                    oid: OrderId::new(77747314)
                },
                OrderOutcome::Rejected {
                    reason: "Order must have minimum value of $10.".into()
                },
            ]
        );
    }

    #[test]
    fn top_level_error_becomes_provider_rejected() {
        let resp = json!({ "status": "err", "response": "Insufficient margin to place order." });
        let err = parse_order_outcomes(&resp).unwrap_err();
        assert!(matches!(err, ProviderError::Rejected(m) if m.contains("Insufficient margin")));
    }

    #[test]
    fn parses_cancel_success_and_error() {
        let resp = json!({
            "status": "ok",
            "response": { "type": "cancel", "data": { "statuses": [
                "success",
                { "error": "Order was never placed, already canceled, or filled." }
            ]}}
        });
        let outcomes = parse_cancel_outcomes(&resp).unwrap();
        assert_eq!(outcomes[0], CancelOutcome::Success);
        assert!(
            matches!(&outcomes[1], CancelOutcome::Rejected { reason } if reason.contains("never placed"))
        );
    }
}
