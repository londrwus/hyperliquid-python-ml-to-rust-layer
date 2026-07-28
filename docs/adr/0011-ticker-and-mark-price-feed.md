# ADR-0011 — A normalized ticker feed, and an event time the venue never sends

**Status:** Accepted · **Date:** 2026-07-25

## Context

The core had no notion of a **mark price**. That absence was not academic: ADR-0010's risk gate
is measured in one, and with nothing feeding it the gate's `MarkCache` is empty, so
`GuardedClient` fails closed on every order that could add exposure. The safe default was
working exactly as designed and the system could not trade. Closing the last data gap in Phase 2
— "Ticker / mark-price (`activeAssetCtx`) feed" (`docs/08-roadmap.md`) — is what makes the gate
usable rather than merely correct.

Hyperliquid's `activeAssetCtx` offers ten fields: `markPx`, `oraclePx`, `midPx`, `funding`,
`openInterest`, `prevDayPx`, `dayNtlVlm`, `premium`, `impactPxs`, and an undocumented
`dayBaseVlm`. The normalized vocabulary rule stated at the top of `axon-core::exec` says a field
earns its place only if the core, the risk gate or a strategy genuinely needs it **and** any
venue could plausibly supply it. Deciding which ten-minus-N survive that test is the first
question.

The second only became visible against real traffic. A 35-second capture off the testnet socket
(BTC, 34 `activeAssetCtx` frames) shows the frame is:

```json
{"channel":"activeAssetCtx","data":{"coin":"BTC","ctx":{"funding":"0.0000125",
 "openInterest":"63.91844","prevDayPx":"64031.0","dayNtlVlm":"1274069.4620399997",
 "premium":"0.0","oraclePx":"64232.0","markPx":"64254.0","midPx":"64257.5",
 "impactPxs":["63990.0","64674.9"],"dayBaseVlm":"19.8655"}}}
```

**There is no timestamp.** `bbo`, `trades`, `l2Book` and `candle` all carry the venue's own
`time`; this one carries nothing. That collides head-on with the rule the whole core is built
on — order on the event's own `ts_event`, never wall-clock at receipt — which exists so a
replay reproduces a live session exactly.

Third, spot contexts arrive on `activeSpotAssetCtx`: one word away from the perp channel, with a
`ctx` that drops `funding`/`oraclePx`/`openInterest` and adds `circulatingSupply`/`totalSupply`.

## Decision

**1. `Ticker` carries six fields, and the exclusions are the decision.**
`axon-core::market::Ticker` is `symbol_id`, `mark_px`, `index_px`, `mid_px`, `funding`,
`open_interest`, plus the two times below. `MarketEvent` gains a `Ticker` variant and
`axon-providers` re-exports the type, exactly as ADR-0008 established.

- `mark_px` is **not** an `Option`. A venue that cannot supply a mark must not emit a `Ticker`.
  An optional mark would push the "is there a price here?" test out to every call site, and the
  one call site that matters is the risk gate.
- `index_px` is the venue-neutral name for what Hyperliquid calls the *oracle* price and what
  CEXes call the index. `mark_px - index_px` is the basis, and it is what separates a mark being
  dragged by a local squeeze from one tracking the underlying.
- `mid_px` is `Option` and is **never** back-filled from the mark. Hyperliquid nulls it on a
  one-sided book. A mark is an oracle-anchored average; a strategy that read it as a mid would
  size a cross against a price no resting order supports, and nothing in the data would say so.
- `open_interest` is in **base** units, the same units as `Trade::sz`. It earns its place by
  being the only quantity here that no other feed can produce — books, trades and candles all
  describe flow; open interest describes the position still outstanding behind it.
- **Excluded:** `prevDayPx` and `dayNtlVlm`/`dayBaseVlm`, because each venue defines the window
  differently (UTC midnight on one, trailing-24h on the next) and a normalized field would invite
  comparisons between numbers that are not the same statistic — while anything a strategy wants
  from them is computable from `Candle`s over an interval we name explicitly. Also `premium` and
  `impactPxs`, which Hyperliquid computes from its own impact-notional depth constant, so two
  venues' `premium` are likewise not one quantity.

**2. Funding is a rate *and* its interval, as one value.** `Funding { rate, interval }` rather
than a bare `funding_rate: Decimal`. Hyperliquid charges hourly and publishes the hourly rate;
most CEX perps publish an eight-hourly one. A carry calculation that reads a bare rate and
assumes the wrong cadence is off by 8x with nothing on the wire to reveal it, and that error only
appears the day a second venue is added — long after the code was reviewed. The rate is stored
exactly as published, never rescaled to a common period, so accrued funding still reconciles
against the venue's own charges.

**3. A ticker carries `ts_venue: Option<Nanos>` and `ts_ingest: Nanos`, not a `ts_event`.**
This is the load-bearing decision. The event needs an ordering key and the venue supplies none,
so the receipt clock has to fill in — but it must not *pass itself off* as a venue time.

- `ts_venue` is `None` on Hyperliquid, stated rather than inferred.
- `ts_ingest` is stamped in `ws::client` the moment the frame is read off the socket, **before**
  parsing, so decode cost does not leak into the timestamp.
- `Ticker::ts_event()` resolves the fallback once (`ts_venue.unwrap_or(ts_ingest)`), so two
  handlers cannot order the same event differently. `Ticker::is_venue_timed()` names the
  distinction for consumers that care — principally the Phase-5 parity harness.
- Where both exist, `ts_ingest - ts_venue` is the feed latency (`docs/05-latency-model.md`).

The rejected alternative was a single `ts_event` field with a doc comment explaining that this
adapter fills it with receipt time. It compiles to the same bytes and hides the same hazard: the
event becomes indistinguishable from a reproducible one, and a replay silently reorders against a
live capture. There is no failure, only a PnL gap nobody can source. Making the absence a type-level
`Option` means a consumer that cares has to look, and one that does not cannot be misled.

**4. Receipt time is a decoder *parameter*, not a clock read.** `decode_ws_message(raw, symbols,
ts_ingest)` gained a third argument rather than calling `SystemClock` internally. The decoders'
purity is the reason they are separated from the client at all; a hidden wall-clock read would
make every ticker assertion non-deterministic and would put tokio-adjacent state into the offline
half of the adapter. The cost is a parameter that every other feed ignores.

**5. Spot contexts are refused, loudly, on two independent lines of defence.**
`activeSpotAssetCtx` returns `DecodeError::UnsupportedChannel` rather than falling through to the
"unknown channel → no events" path, because a silently ignored frame is a subscription that looks
healthy and never yields a price. Independently, `funding`/`oraclePx`/`openInterest` are
**required** on the perp wire struct, so a spot-shaped body that reached the perp decoder fails to
deserialize instead of producing a plausible `Ticker` with those three quietly `None` — which is
indistinguishable from a venue that simply does not publish them.

**6. `MarketDataProcessor::mark_px` returns `(Decimal, Nanos)`, never a bare price.** A *stale*
mark is more dangerous than a missing one. A missing mark makes the gate fail closed and costs one
trade; a mark from five minutes ago sails through the same gate and sizes a position against a
price that no longer exists. Handing out the price alone would let every call site drop the age on
the floor, so the time travels with it and the caller decides what "too old" means — only it knows
its own horizon. The full `Ticker` is available via `ticker()`; `mid()` deliberately still does
**not** consult it, because `mid()` is a book-derived quantity and answering it from a different
feed with a different cadence would hide which one a caller actually got.

## Consequences

- **+** The risk gate has a real price source. `MarkCache` can now be fed from the market-data
  handler instead of being hand-populated, which is the last thing standing between ADR-0010's gate
  and it being usable in the runtime.
- **+** Basis and funding are first-class, so a carry or funding strategy needs no venue-specific
  code — and cannot get the funding cadence wrong.
- **+** The feed rides the existing connect/subscribe/heartbeat/reconnect/resubscribe loop with no
  special case. `Feed::Ticker` lands in `desired` like any other feed, so it is replayed on
  reconnect by the same code path; a bespoke arm would have been one more place for a resubscribe
  to be forgotten, and a silently unsubscribed mark feed reads as a gate failing closed for no
  visible reason.
- **+** Decoder tests run against a byte-for-byte testnet capture, including the undocumented
  `dayBaseVlm` and the absent `time`, so the two things this feed actually gets wrong are pinned
  by fixtures rather than by hand-written JSON that agrees with the docs.
- **−** **The ticker stream does not replay deterministically.** Its ordering key is our wall
  clock, so a captured session only reproduces if the capture also records `ts_ingest` alongside
  each frame. Phase 5's parity harness must either persist receipt times or exclude tickers from
  event-order comparison; `is_venue_timed()` is how it tells. This is inherent to the venue, not to
  the design — the design only makes it visible.
- **−** `Ticker` is the one market type with two time fields, which is an inconsistency in the
  vocabulary. It is the honest one: it is also the one feed with a missing venue time.
- **−** `decode_ws_message` grew a third parameter that four of five feeds ignore.
- **−** A ticker's ingest time is the age of *our knowledge*, not of the venue's number. A gap can
  mean the venue went quiet or that our socket did, and the two are not distinguishable from this
  feed alone.
- **−** The venue publishes `activeAssetCtx` roughly once a second (34 frames in a 35 s capture),
  so a mark can be up to ~1 s old even on a perfectly healthy socket. Any staleness threshold a
  caller picks has to sit above that floor.
- **−** Spot instruments are unsupported, by refusal rather than omission. Adding them means a
  second normalized shape or a second wire struct, not a relaxed perp decoder.
- **−** There is no REST seed. `POST /info {"type":"metaAndAssetCtxs"}` would prefill every mark at
  startup the way `fetch_l2_snapshot` seeds a book; without it there is a window after each connect
  during which the gate still fails closed. The window is short — the venue pushes the current
  context immediately on subscribe — but a restart pays it, and it is exactly the moment a process
  most wants to be able to flatten.

See [ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the vocabulary/bus pattern this extends),
[ADR-0010](0010-execution-events-and-reconciliation.md) (the risk gate this feeds, and the
fail-closed-on-a-missing-mark rule), `docs/05-latency-model.md`,
`docs/research/hyperliquid-execution.md`.
