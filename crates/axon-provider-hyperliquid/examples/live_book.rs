//! Live end-to-end demo: stream a Hyperliquid book and print BBO / mid / candle.
//!
//! This wires the whole Phase-2 market-data path together:
//!   fetch `meta` → real `SymbolMap` → WS adapter (tokio edge) → core bus
//!   (crossbeam) → deterministic drain → `MarketDataProcessor` → print.
//!
//! It hits the **live** network, so it is an example, not a test — the default
//! `cargo test` never runs it. Try it with:
//!   `cargo run -p axon-provider-hyperliquid --example live_book -- BTC`

use std::time::Duration;

use axon_core::drain_available;
use axon_marketdata::MarketDataProcessor;
use axon_provider_hyperliquid::ws::{fetch_meta, MAINNET_INFO};
use axon_provider_hyperliquid::{HyperliquidMarketData, SymbolMap};
use axon_providers::{CandleInterval, Feed};

#[tokio::main]
async fn main() {
    let coin = std::env::args().nth(1).unwrap_or_else(|| "BTC".to_string());
    println!("streaming {coin} from Hyperliquid mainnet — Ctrl-C to stop\n");

    // Resolve real asset indices from the venue; fall back to a single-coin map.
    let symbols = match fetch_meta(MAINNET_INFO).await {
        Ok(m) => {
            println!("loaded {} perp symbols from meta", m.len());
            m
        }
        Err(e) => {
            eprintln!("meta fetch failed ({e}); using single-coin fallback");
            SymbolMap::from_perps([coin.as_str()])
        }
    };
    let Some(sym) = symbols.id(&coin) else {
        eprintln!("coin {coin:?} not found in the perp universe");
        return;
    };

    let (tx, rx) = axon_core::bus(4096);
    let md = HyperliquidMarketData::mainnet(symbols, vec![coin.clone()], tx);
    md.subscribe_coin(Feed::Bbo, &coin);
    md.subscribe_coin(Feed::L2Book, &coin);
    md.subscribe_coin(Feed::Trades, &coin);
    md.subscribe_coin(Feed::Candles(CandleInterval::M1), &coin);

    // WS ingest lives on the tokio runtime (the async edge).
    tokio::spawn(async move { md.run_forever().await });

    // The core-side consumer. In production this is its own synchronous thread,
    // off the runtime; for a demo we poll it on a timer.
    let mut processor = MarketDataProcessor::new();
    loop {
        drain_available(&rx, &mut processor);
        let mid = processor.mid(sym);
        let last = processor.last_trade(sym).map(|t| t.px);
        let candle_close = processor.last_candle(sym).map(|c| c.close);
        match processor.bbo(sym) {
            Some(b) => println!(
                "{coin}  bid {} × {}   ask {} × {}   mid {}   last {}   1m-close {}",
                b.bid_px,
                b.bid_sz,
                b.ask_px,
                b.ask_sz,
                fmt(mid),
                fmt(last),
                fmt(candle_close),
            ),
            None => match mid {
                Some(m) => println!("{coin}  mid {m} (book only)"),
                None => println!("{coin}  … waiting for first update"),
            },
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn fmt(v: Option<axon_core::Decimal>) -> String {
    v.map(|d| d.to_string()).unwrap_or_else(|| "-".to_string())
}
