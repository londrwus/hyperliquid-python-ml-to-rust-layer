//! The canned session the offline run replays — market data *and* signals.
//!
//! `cargo run --bin axon` has to prove something more useful than "the config
//! parsed". These events go onto the real bus, through the real [`CoreHandler`], into
//! the real order tracker and mark cache; these signals go through the real
//! [`SignalReader`] and the real [`Planner`] into real
//! [`OrderRequest`](axon_providers::OrderRequest)s — the identical code path a live
//! session uses, with the venue adapters removed. So the offline run exercises the
//! fan-out, the mark precedence rule, order adoption, fill accounting **and the
//! Py→Rust join**, with no socket, no key and no tokio, and it does it
//! deterministically: fixed event times, fixed quantities, same result on every
//! machine.
//!
//! [`CoreHandler`]: crate::handler::CoreHandler
//! [`SignalReader`]: axon_strategy::SignalReader
//! [`Planner`]: axon_strategy::Planner

use axon_contracts::Signal;
use axon_core::{
    AccountSnapshot, Bbo, BookSnapshot, Cloid, Decimal, Event, ExecEvent, Fill, Level, Liquidity,
    MarketEvent, Nanos, OrderId, OrderStatus, OrderUpdate, Side, SymbolId, Ticker,
};
use axon_providers::{InstrumentSpec, InstrumentTable, PriceGrid, SizeGrid};

const SEC: Nanos = 1_000_000_000;

/// The wire scale for a quantity: `10^-8` units (`contracts/schema.toml`).
const UNIT: i64 = 100_000_000;

fn dec(units: i64, scale: u32) -> Decimal {
    Decimal::new(units, scale)
}

/// A seven-event session covering one of each thing the core has to get right.
///
/// The order matters and is part of what is being demonstrated: the BBO arrives
/// *after* the ticker, so a correct mark cache keeps the venue mark; the order update
/// arrives for an order this process never submitted, so a correct tracker adopts it
/// instead of ignoring it; the fill then closes that adopted order out; and a second
/// adopted order is left *resting*, because a restarted session finding somebody's
/// stale quote against a superseded target is the case the intent source's cancels
/// exist for.
pub fn events(btc: SymbolId, eth: SymbolId) -> Vec<Event> {
    vec![
        // 1. The venue's own mark price — the risk gate's preferred source.
        Event::Market(MarketEvent::Ticker(Ticker {
            symbol_id: btc,
            mark_px: dec(50_000, 0),
            index_px: Some(dec(49_995, 0)),
            mid_px: Some(dec(500_005, 1)),
            funding: None,
            open_interest: None,
            // Stamped as venue-timed so the canned run stays reproducible: a ticker
            // ordered on receipt time would make the offline session's event order
            // depend on the machine it ran on.
            ts_venue: Some(SEC),
            ts_ingest: SEC,
        })),
        // 2. A two-sided quote that must NOT displace the live mark above.
        Event::Market(MarketEvent::Bbo(Bbo {
            symbol_id: btc,
            bid_px: dec(49_999, 0),
            bid_sz: dec(1, 0),
            ask_px: dec(50_001, 0),
            ask_sz: dec(1, 0),
            ts_event: 2 * SEC,
        })),
        // 3. A second instrument with no ticker feed: its mark comes from the book.
        Event::Market(MarketEvent::Book(BookSnapshot {
            symbol_id: eth,
            bids: vec![Level::new(dec(2_999, 0), dec(10, 0))],
            asks: vec![Level::new(dec(3_001, 0), dec(10, 0))],
            ts_event: 3 * SEC,
        })),
        // 4. An order we never submitted — what a restart finds resting at the venue.
        Event::Exec(ExecEvent::Order(OrderUpdate {
            symbol_id: btc,
            order_id: OrderId::new(1),
            cloid: Some(Cloid::new(0xA11CE)),
            side: Side::Buy,
            status: OrderStatus::Resting,
            price: Some(dec(49_000, 0)),
            orig_qty: dec(1, 1),
            remaining_qty: dec(1, 1),
            cancel_reason: None,
            ts_event: 4 * SEC,
        })),
        // 5. …which then fills, moving the position.
        Event::Exec(ExecEvent::Fill(Fill {
            symbol_id: btc,
            order_id: OrderId::new(1),
            cloid: Some(Cloid::new(0xA11CE)),
            side: Side::Buy,
            qty: dec(1, 1),
            price: dec(49_000, 0),
            fee: dec(1, 2),
            closed_pnl: Decimal::ZERO,
            liquidity: Liquidity::Maker,
            trade_id: 1,
            ts_event: 5 * SEC,
        })),
        // 6. The venue's own account snapshot, the input to drift detection.
        Event::Exec(ExecEvent::Account(AccountSnapshot {
            equity: dec(10_000, 0),
            withdrawable: dec(9_500, 0),
            margin_used: dec(500, 0),
            ts_event: 6 * SEC,
        })),
        // 7. A second order we never submitted, and this one is still resting. It is
        //    what the first signal below has to cancel: an order working against a
        //    target nobody holds any more is a stale quote, and a stale quote is
        //    exactly what somebody else's taker is looking for.
        Event::Exec(ExecEvent::Order(OrderUpdate {
            symbol_id: btc,
            order_id: OrderId::new(2),
            cloid: None,
            side: Side::Sell,
            status: OrderStatus::Resting,
            price: Some(dec(51_000, 0)),
            orig_qty: dec(5, 2),
            remaining_qty: dec(5, 2),
            cancel_reason: None,
            ts_event: 7 * SEC,
        })),
    ]
}

/// The canned target positions the offline session plans from.
///
/// Three records, each chosen for one thing it proves about the join:
///
/// 1. **A signal that sat too long.** Stamped at `2 s` with a 500 ms window against an
///    event clock that has reached `7 s`, so it is refused and counted. A late
///    target-position signal is not a weaker opinion about the current market; it is a
///    firm opinion about a market that has already gone.
/// 2. **The order is the delta, and it cancels what it supersedes.** The event stream
///    above left us long `0.1` BTC with an order still resting, so "be long `0.3`" must
///    come out as one cancel and a buy for `0.2` — never a buy for `0.3`, which would
///    end at `0.4` and compound on the next signal.
/// 3. **An instrument priced off the book rather than a BBO feed.** ETH has no `bbo`
///    in the stream, so this only produces an order if the fallback to the L2 book
///    works; without it a config that subscribes to `l2Book` alone would look exactly
///    like an idle strategy.
pub fn signals(btc: SymbolId, eth: SymbolId) -> Vec<Signal> {
    vec![
        Signal::target_position(1, 2 * SEC, btc.get(), UNIT / 2, 0, 0, 500, 1, 0),
        Signal::target_position(2, 7 * SEC, btc.get(), 3 * UNIT / 10, 0, 0, 500, 1, 0),
        Signal::target_position(3, 7 * SEC, eth.get(), -UNIT, 2, 0, 500, 1, 0),
    ]
}

/// The declared instrument grids the offline run rounds against (ADR-0025).
///
/// **A fiction, and it says so:** no venue published these numbers. They live here
/// beside the canned events rather than on the port, so no live path can reach for
/// them — exactly like `StaticRiskContext`. Their value is that the offline gate
/// exercises the quantizer instead of skipping it, and they are chosen so that every
/// price and size in [`events`]/[`signals`] is *already* on the grid: 49 999 and 2 999
/// are integers on ticks of 1 and 0.1, 0.2 and 1 are whole multiples of both lots, and
/// both notionals are thousands of dollars against a $10 minimum. So this adds a code
/// path without moving a single expected value.
///
/// A spec is declared for **every** id handed in, not for the two the canned stream
/// happens to touch. `resolve_symbols` refuses a configured coin with no grid, so a
/// table built from the first two would make a three-symbol config — a shape
/// [`RuntimeConfig::coins`](crate::config::RuntimeConfig::coins) supports and is tested
/// for — refuse to start the offline gate, and refuse it by naming a venue this mode
/// never contacts. The extra ids simply receive no canned events, which is what they
/// did before.
///
/// The two shapes alternate, so a two-symbol run gets exactly the table it always got
/// and a longer one keeps exercising both grids. Which shape a given id draws is
/// deliberately not depended on anywhere: the canned prices and sizes are legal on
/// both.
pub fn instruments(ids: impl IntoIterator<Item = SymbolId>) -> InstrumentTable {
    let mut t = InstrumentTable::new();
    for (i, id) in ids.into_iter().enumerate() {
        // BTC-like: szDecimals 5, so one price decimal and a 1e-5 lot. ETH-like:
        // szDecimals 4, so two price decimals and a 1e-4 lot.
        t.insert(perp(id, if i % 2 == 0 { 5 } else { 4 }));
    }
    t
}

/// One Hyperliquid-shaped perp: `6 - szDecimals` decimal places capped at five
/// significant figures, a `10^-szDecimals` lot, and the venue's $10 minimum.
fn perp(id: SymbolId, sz_decimals: u32) -> InstrumentSpec {
    InstrumentSpec {
        symbol_id: id,
        price: PriceGrid::decimals_with_sig_figs(6 - sz_decimals, 5)
            .expect("a fixed, in-range grid"),
        size: SizeGrid::decimals(sz_decimals).expect("a fixed, in-range lot"),
        min_notional: Some(Decimal::TEN),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canned_session_is_already_on_the_grid_it_declares() {
        // The offline run is the default gate, so a fiction that moved its own expected
        // values would make every downstream assertion a test of this file. Every price
        // and size the canned stream carries has to be legal under the grid it declares.
        let (btc, eth) = (SymbolId::new(0), SymbolId::new(1));
        let t = instruments([btc, eth]);
        let b = t.get(btc).expect("BTC has a grid");
        let e = t.get(eth).expect("ETH has a grid");

        for px in [
            dec(49_999, 0),
            dec(50_001, 0),
            dec(49_000, 0),
            dec(51_000, 0),
        ] {
            assert!(b.price.is_valid(px), "{px} is off the BTC grid");
        }
        for px in [dec(2_999, 0), dec(3_001, 0)] {
            assert!(e.price.is_valid(px), "{px} is off the ETH grid");
        }
        for qty in [dec(1, 1), dec(5, 2), dec(2, 1), dec(3, 1)] {
            assert!(b.size.is_valid(qty), "{qty} is off the BTC lot");
        }
        assert!(e.size.is_valid(Decimal::ONE));
        // And both orders the canned signals produce clear the declared minimum, so the
        // offline run never exercises the refusal by accident.
        assert!(dec(2, 1) * dec(49_999, 0) > Decimal::TEN);
        assert!(Decimal::ONE * dec(2_999, 0) > Decimal::TEN);
    }
}
