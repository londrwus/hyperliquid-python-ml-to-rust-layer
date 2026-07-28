//! Hyperliquid WS subscription request builders.
//!
//! A subscribe frame is `{"method":"subscribe","subscription":{"type":..,"coin":..}}`
//! (`candle` also carries an `interval`). We map our normalized [`Feed`] to the
//! venue's `type` string here so the rest of the adapter never hard-codes it.
//!
//! Account-scoped (**user**) channels use the same envelope but key on a `user`
//! address instead of a `coin` — see [`UserChannel`].

use axon_providers::{CandleInterval, Feed};
use serde_json::{json, Value};

/// Hyperliquid's `type` string for a feed.
fn feed_type(feed: Feed) -> &'static str {
    match feed {
        Feed::L2Book => "l2Book",
        Feed::Trades => "trades",
        Feed::Bbo => "bbo",
        Feed::Candles(_) => "candle",
        Feed::Ticker => "activeAssetCtx",
    }
}

/// Hyperliquid's interval string for a candle feed.
fn interval_str(i: CandleInterval) -> &'static str {
    match i {
        CandleInterval::M1 => "1m",
        CandleInterval::M5 => "5m",
        CandleInterval::M15 => "15m",
        CandleInterval::H1 => "1h",
        CandleInterval::H4 => "4h",
        CandleInterval::D1 => "1d",
    }
}

/// The `subscription` object for a feed on a coin.
fn subscription(feed: Feed, coin: &str) -> Value {
    let mut obj = json!({ "type": feed_type(feed), "coin": coin });
    if let Feed::Candles(i) = feed {
        obj["interval"] = json!(interval_str(i));
    }
    obj
}

/// A `subscribe` request frame (serialized JSON) for `feed` on `coin`.
pub fn subscribe_msg(feed: Feed, coin: &str) -> String {
    json!({ "method": "subscribe", "subscription": subscription(feed, coin) }).to_string()
}

/// An `unsubscribe` request frame (serialized JSON) for `feed` on `coin`.
pub fn unsubscribe_msg(feed: Feed, coin: &str) -> String {
    json!({ "method": "unsubscribe", "subscription": subscription(feed, coin) }).to_string()
}

/// The heartbeat frame. Hyperliquid closes idle connections after ~60s.
pub fn ping_msg() -> String {
    json!({ "method": "ping" }).to_string()
}

// ── account-scoped (user) channels ───────────────────────────────────────────

/// The account-scoped WS channels that report on *our* orders.
///
/// Each carries two distinct venue strings, which is the whole reason this enum
/// exists: [`wire`](Self::wire) is what we *send* as `subscription.type`, and
/// [`reply_channel`](Self::reply_channel) is what the venue *stamps on the frames*
/// it sends back. They differ for `userEvents`, whose frames arrive on channel
/// `"user"`. Matching incoming frames against the subscription type therefore
/// receives nothing at all, with no error to explain why — so the pairing is
/// defined once, here, and both the decoder and the client read it from this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UserChannel {
    /// Incremental fills, funding, liquidations and venue-side cancels. Never
    /// snapshots — an idle account produces zero frames.
    UserEvents,
    /// Fills only, with an initial `isSnapshot: true` replay.
    UserFills,
    /// Order lifecycle transitions. Never snapshots.
    OrderUpdates,
}

impl UserChannel {
    /// Every user channel, in the order a client should subscribe to them.
    pub const ALL: [UserChannel; 3] = [
        UserChannel::UserEvents,
        UserChannel::UserFills,
        UserChannel::OrderUpdates,
    ];

    /// The `subscription.type` string to send.
    pub const fn wire(self) -> &'static str {
        match self {
            UserChannel::UserEvents => "userEvents",
            UserChannel::UserFills => "userFills",
            UserChannel::OrderUpdates => "orderUpdates",
        }
    }

    /// The `channel` value the venue puts on frames for this subscription.
    ///
    /// Note `userEvents` → `"user"`: the venue does not echo the type back.
    pub const fn reply_channel(self) -> &'static str {
        match self {
            UserChannel::UserEvents => "user",
            UserChannel::UserFills => "userFills",
            UserChannel::OrderUpdates => "orderUpdates",
        }
    }

    /// Inverse of [`reply_channel`](Self::reply_channel) — used by the decoder to
    /// route an incoming frame. `None` for any channel that is not a user channel.
    pub fn from_reply_channel(channel: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.reply_channel() == channel)
    }
}

/// The `subscription` object for a user channel.
///
/// `user` must be the **master** (or sub-account) address — the account the orders
/// belong to — and *not* the agent/API-wallet address that signs them: the venue
/// answers an agent address with empty results rather than an error. There is no
/// signature and no token on these channels; the address is the only input.
///
/// `userFills` also accepts an optional `aggregateByTime`. We omit it, letting the
/// venue default it to `false`, because aggregated fills would collapse several
/// executions into one entry and the tracker dedups on the per-execution `tid`.
fn user_subscription(channel: UserChannel, user: &str) -> Value {
    json!({ "type": channel.wire(), "user": user })
}

/// A `subscribe` request frame (serialized JSON) for `channel` on account `user`.
pub fn subscribe_user_msg(channel: UserChannel, user: &str) -> String {
    json!({ "method": "subscribe", "subscription": user_subscription(channel, user) }).to_string()
}

/// An `unsubscribe` request frame (serialized JSON) for `channel` on account `user`.
pub fn unsubscribe_user_msg(channel: UserChannel, user: &str) -> String {
    json!({ "method": "unsubscribe", "subscription": user_subscription(channel, user) }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_l2book_subscribe() {
        let v: Value = serde_json::from_str(&subscribe_msg(Feed::L2Book, "BTC")).unwrap();
        assert_eq!(v["method"], "subscribe");
        assert_eq!(v["subscription"]["type"], "l2Book");
        assert_eq!(v["subscription"]["coin"], "BTC");
        assert!(v["subscription"].get("interval").is_none());
    }

    #[test]
    fn builds_candle_subscribe_with_interval() {
        let v: Value =
            serde_json::from_str(&subscribe_msg(Feed::Candles(CandleInterval::M5), "ETH")).unwrap();
        assert_eq!(v["subscription"]["type"], "candle");
        assert_eq!(v["subscription"]["interval"], "5m");
        assert_eq!(v["subscription"]["coin"], "ETH");
    }

    #[test]
    fn ticker_subscribes_to_the_perp_context_channel_only() {
        // `activeAssetCtx` and `activeSpotAssetCtx` are one word apart and carry
        // different `ctx` shapes. We only ever ask for the perp one; the decoder
        // refuses the spot reply outright rather than reading it as a perp.
        let v: Value = serde_json::from_str(&subscribe_msg(Feed::Ticker, "BTC")).unwrap();
        assert_eq!(v["subscription"]["type"], "activeAssetCtx");
        assert_eq!(v["subscription"]["coin"], "BTC");
        assert!(v["subscription"].get("interval").is_none());
    }

    #[test]
    fn unsubscribe_mirrors_subscribe() {
        let v: Value = serde_json::from_str(&unsubscribe_msg(Feed::Bbo, "SOL")).unwrap();
        assert_eq!(v["method"], "unsubscribe");
        assert_eq!(v["subscription"]["type"], "bbo");
        assert_eq!(v["subscription"]["coin"], "SOL");
    }

    #[test]
    fn ping_is_method_only() {
        let v: Value = serde_json::from_str(&ping_msg()).unwrap();
        assert_eq!(v["method"], "ping");
    }

    const ACCOUNT: &str = "0x9be9c0f9c1e4e4a1b0d0f2e3d4c5b6a7f8091a2b";

    #[test]
    fn builds_user_subscribe_with_address_and_no_coin() {
        let v: Value =
            serde_json::from_str(&subscribe_user_msg(UserChannel::UserEvents, ACCOUNT)).unwrap();
        assert_eq!(v["method"], "subscribe");
        assert_eq!(v["subscription"]["type"], "userEvents");
        assert_eq!(v["subscription"]["user"], ACCOUNT);
        assert!(v["subscription"].get("coin").is_none());
        // We never send aggregateByTime; the venue defaults it to false.
        assert!(v["subscription"].get("aggregateByTime").is_none());
    }

    #[test]
    fn unsubscribe_user_mirrors_subscribe_user() {
        let v: Value =
            serde_json::from_str(&unsubscribe_user_msg(UserChannel::OrderUpdates, ACCOUNT))
                .unwrap();
        assert_eq!(v["method"], "unsubscribe");
        assert_eq!(v["subscription"]["type"], "orderUpdates");
        assert_eq!(v["subscription"]["user"], ACCOUNT);
    }

    #[test]
    fn user_events_replies_on_a_different_channel_than_it_subscribes_to() {
        // The one asymmetry in the venue's naming, and a silent failure if missed.
        assert_eq!(UserChannel::UserEvents.wire(), "userEvents");
        assert_eq!(UserChannel::UserEvents.reply_channel(), "user");
        // The other two do echo their type back.
        assert_eq!(UserChannel::UserFills.reply_channel(), "userFills");
        assert_eq!(UserChannel::OrderUpdates.reply_channel(), "orderUpdates");
    }

    #[test]
    fn reply_channel_round_trips_for_every_user_channel() {
        for c in UserChannel::ALL {
            assert_eq!(UserChannel::from_reply_channel(c.reply_channel()), Some(c));
        }
        assert_eq!(UserChannel::from_reply_channel("l2Book"), None);
        assert_eq!(
            UserChannel::from_reply_channel("userEvents"),
            None,
            "the subscription type is not a reply channel"
        );
    }
}
