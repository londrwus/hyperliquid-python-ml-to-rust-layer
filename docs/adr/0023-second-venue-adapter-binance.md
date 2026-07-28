# ADR-0023 — The second venue: what adding Binance actually cost

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0004](0004-provider-abstraction-layer.md) states the project's central claim in one
sentence: *"Adding a venue = writing one adapter; the core, strategies, and other adapters
don't change. This is the concrete test of 'it's a Layer.'"* It was written when one venue
existed. Every ADR since has been designed against that claim and none of them has been
able to test it, because a port designed against a single venue and validated against that
same venue is a tautology.

Phase 7 is the test. This ADR records what a second venue cost, and — more usefully — every
place it did not fit.

The venue is **Binance USD-M futures**, chosen because it is the maximally different
plausible second venue: a centralized exchange with an HMAC-signed REST API against a DEX
with two EIP-712 schemes, a fixed tick against a significant-figure rule, native market
orders against synthesized ones, 730 instruments in one response against ~210 curated ones,
and no on-chain anything.

Two constraints shaped the work and both are honest limits on what it proves:

- **Binance production is geo-blocked from this host.** `fapi.binance.com` answers
  `HTTP 451`. Everything captured, and everything run, is **Binance USD-M futures testnet**
  (`testnet.binancefuture.com` / `fstream.binancefuture.com`). The wire format is the same;
  the liquidity is not, and neither is the undocumented `st` field that appears on every
  testnet frame.
- **There are no Binance credentials in this repository, none were sought, and this
  workstream is not authorized to trade there.** So the signed half of the adapter stops
  deliberately short of the wire — see decision 8.

## Decision

### 1. The venue is `binance-usdm`, not `binance`

Binance runs three matching engines behind three wire formats: spot, USD-M futures, COIN-M
futures. They have different endpoints, different symbol universes, and — the one that
matters — a *different frame shape for the same concept*: spot's partial-depth frame is
`{"lastUpdateId":…,"bids":[…],"asks":[…]}` with **no event time at all**, where the futures
one carries two. A `SymbolId` resolved against one and used against another addresses a
different instrument, successfully and silently. Naming the venue after the product is what
stops a future spot adapter from inheriting this one's name and its symbol table.

### 2. The combined stream, because the stream name is the only discriminator

`<sym>@depth20@100ms` (a 20-level snapshot, replace the book) and `<sym>@depth@100ms` (an
incremental diff against a REST seed) arrive as **the same event type with the same field
set**: `{"e":"depthUpdate","E":…,"T":…,"s":…,"U":…,"u":…,"pu":…,"b":[…],"a":[…]}`. Both are
in this crate's fixtures, captured off one socket in the same second, and a test asserts
their key sets are equal. Nothing inside a frame tells them apart.

Read as a snapshot, a diff replaces a forty-level book with the one or two levels that
changed — a book that is wrong, looks complete, and refreshes ten times a second with no
error anywhere. So the adapter connects to `/stream?streams=…`, whose envelope carries the
stream name, and never to `/ws/<name>`, which sends the bare payload. Routing is on the
stream name, not on `"e"`. An unenveloped frame is a **decode error**, not an ignored
control frame, because the alternative is that connecting to the wrong endpoint yields no
events, no errors, and a session that reads as healthy and is deaf.

The diff stream itself is refused by name (`DecodeError::UnsupportedStream`), the way
ADR-0011 refuses `activeSpotAssetCtx`, and for the same reason: a silently ignored frame is
a subscription that looks healthy and never yields data.

### 3. Event time is `T`, never `E`

Every futures frame carries two timestamps — `E`, when the venue pushed the bytes, and `T`,
when the thing happened. They are not close: on the captured `aggTrade` they are **154 ms**
apart, and on the captured depth frame **43 ms**. `E` is the venue's half of our own
`ts_ingest`; `T` is the event's own time, which is what the core orders on.

This is a hazard Hyperliquid does not have — its feeds carry one `time` each — and it is
invisible: a decoder reading `E` produces perfectly plausible events, ordered wrong by a
fraction of a second, which on a 100 ms book is four books' worth of interleaving against
every trade beside it.

### 4. `SymbolId` is ours to invent, dense, and ordered by listing date

Hyperliquid's `SymbolId` *is* the venue's asset index — the number that goes on the wire.
Binance addresses everything by symbol string and publishes no index, so the ids are ours.
Two upstream facts constrain the choice, and they pull in opposite directions:

- `axon_execution::InFlight` gives each symbol below `CAPACITY` (1024) its own bit and
  collapses everything at or past it into one shared overflow slot — which silently
  reinstates the global per-session gate that type exists to remove. Its own comment says
  the bound is safe because *"`SymbolId` is a dense index into the venue's own `meta`
  universe rather than a hash"*. So **the ids must be dense**: a hash of the symbol name
  would be stable forever and would put all 730 instruments past the bound, with no error,
  no counter and no log line.
- A capture records `SymbolId`s and no symbol table, so a replay resolves them against
  whatever universe the replaying process holds. That wants ids **stable across time**.

Dense and stable cannot both be had from a venue that supplies neither. This picks dense and
buys back what ordering can: ids are assigned over the kept rows sorted by
`(onboardDate, symbol)`. A new listing therefore **appends** and leaves every existing id
alone, which is the event that happens weekly. A **delisting still compacts every id above
it**, and that hole is stated rather than hidden — it is the same hole as ADR-0025's
out-of-scope #2 (a plan is a function of a network-fetched input the log does not carry) and
it has the same fix, which is to persist the mapping rather than derive it.

### 5. `exchangeInfo` skips rather than fails whole, and every skip carries its reason

Hyperliquid's `decode_universe` fails the **entire** `meta` decode over one asset that will
not build a spec, and that is right for ~210 curated perps. Binance answers with 730 rows in
one body: perpetuals beside quarterlies, `TRADING` beside `PENDING_TRADING`, `SETTLING`,
`DELIVERING` and `PRE_SETTLE`. Several carry, in perfectly good faith, values that are not a
grid — `ELSAUSDT` has `"tickSize":"0"` because it is a pending listing. Failing whole means
one instrument nobody asked to trade stops the adapter from starting.

So a row that cannot be griddled is skipped — and ADR-0025 is explicit that the danger of
skipping is *silence*: "an asset silently absent from the table fails closed later with
nothing attached saying why." The answer is `Universe::skipped: Vec<Skipped>` with one of
five typed reasons each, and `why_skipped(symbol)`, so a startup can say "you configured
ELSAUSDT and it is `PENDING_TRADING`" instead of "unknown symbol".

Two refusals inside that are decisions rather than parsing:

- **`"tickSize":"0"` is refused, not read as `PriceGrid::unconstrained()`.** Both are
  increment-zero, so the mapping type-checks and reads as tidy. `unconstrained` means "this
  venue has no price rules"; the venue means "not yet". Reading one as the other is
  fail-open on the instrument we know least about.
- **Dated contracts are refused by `contractType`**, the way ADR-0011 refuses spot. A
  quarterly has an expiry and a basis that decays into it, and the normalized vocabulary has
  no word for either — decoded as a perp it yields a plausible `Ticker` whose funding field
  describes a charge that instrument does not levy. `TRADIFI_PERPETUAL` (tokenised-equity
  perps) is excluded too, because it halts outside equity hours and a market-data gap that
  is *scheduled* reads exactly like a dead socket. Admitting it is a decision about market
  hours, not about parsing.

### 6. Only closed candles are emitted

This is the one place the normalized vocabulary came up a field short.
`axon_core::Candle` documents `ts_event` as "the candle's close time — the point at which it
is final for ordering", and has no way to say **not final**. Binance republishes the
in-progress bar on every trade: the capture behind these fixtures holds 112 `kline_1m`
frames for one symbol and **one** of them has `"x":true`. All 112 carry the same `T`, so
emitting them all would put 111 events on the bus that each claim to be the same final bar
and are each a different bar — with *identical ordering keys*, so an event-time sort cannot
even separate them.

Dropping the unclosed ones is the one-sided-BBO precedent: skip rather than fabricate. The
cost is stated plainly and is real — **no in-progress bar reaches a strategy through this
adapter** — and closing it is an `is_final` flag on `Candle` in `axon-core`, which is a port
change and not an adapter one. Hyperliquid's `candle` feed also publishes updates inside a
bar, and its decoder stamps every one of them as final; nothing in the tree says so today.

### 7. Funding is per symbol, from a second request, with a named default

`markPriceUpdate` carries the funding *rate* and the *next* funding time, and never the
period between them — the same gap `activeAssetCtx` has, arrived at differently.
`GET /fapi/v1/fundingInfo` publishes `fundingIntervalHours` per symbol — 616 entries
against a 730-row `exchangeInfo`, covering all 570 perpetuals this adapter kept on the day
it was run — and a symbol it does not mention funds on the documented eight-hour default.

So `SymbolTable` carries a per-symbol interval, `DEFAULT_FUNDING_INTERVAL_NS` is the
fallback, and `funding_is_published(id)` names the difference — the same distinction
`Ticker::is_venue_timed()` draws about time, for the same reason: a consumer that cares
whether a number came from the venue has to be able to ask.

ADR-0011's `Funding { rate, interval }` is what makes this safe, and it is worth naming
what it prevented: **Hyperliquid funds hourly, Binance every eight hours.** A bare
`funding_rate` field would have made every carry number on the second venue wrong by 8×,
with nothing on either wire to reveal it. That ADR said this error "only appears the day a
second venue is added". This is that day, and the pairing caught it.

### 8. Execution stops at the query string, on purpose

Binance signs an HMAC-SHA256 of the query string, hex-encoded, appended as the final
`signature=` parameter, with the API key in an `X-MBX-APIKEY` header. Everything up to the
bytes is here and tested — parameter order, the canonical string, hex encoding, the
`signature`-last assembly, and a `Signer`-based `sign_request` — and the MAC itself is not.

Two reasons, and both are stated as a test (`no_signer_in_this_workspace_can_actually_sign_for_this_venue`):

1. It needs two workspace dependencies this workstream may not add. The exact lines are in
   *Out of scope* below.
2. There are no Binance credentials here and this workstream is not authorized to trade
   there. An adapter that could sign but had nothing to sign with is a loaded path with the
   safety on.

**There is no HTTP client for `/fapi/v1/order` in this crate**, so there is no path from
this code to an order at Binance at all. The canonical payload is nevertheless pinned
against Binance's own published worked example, byte for byte, so the canonicalization —
the part that is easy to get wrong and impossible to debug, because a wrong string and a
wrong secret both return `-1022` — is verified today with no crypto.

## What fit, what did not, and what that says about the Layer claim

### The port held, and four things earned their keep

Nothing in `axon-core`, `axon-providers`, `axon-execution`, `axon-strategy` or
`axon-runtime` was changed. `MarketEvent`, `Ticker`, `Funding`, `Level`, `BookSnapshot`,
`Trade`, `Candle`, `Bbo`, `Feed`, `Capabilities`, `InstrumentTable`, `PriceGrid`,
`SizeGrid`, `Precision`, `OrderRequest`, `CancelId`, `Credentials`, `Signer`,
`SignatureScheme` and the bus all took the second venue unmodified. Four of those were
speculative when written and are now load-bearing:

1. **`PriceGrid { increment, sig_figs }` — one struct, no enum arm.** ADR-0025 rejected a
   `{ Digits, Grid }` enum on the argument that venue three would grow a `match` in every
   consumer. Hyperliquid's rule is *significant figures*; Binance's is a *fixed tick*. With
   `sig_figs: None`, `tick_at(px) = max(increment, sig_quantum(px))` degenerates exactly to
   `increment`. Binance sets one field, adds no arm, and nothing above the adapter noticed.
   **This is the single strongest piece of evidence in this ADR.**
2. **`SizeGrid::step` beside `SizeGrid::decimals`.** `ARKUSDT` has `"stepSize":"3"` — a lot
   of three whole units. A `szDecimals`-shaped model (one integer meaning `10^-n`) cannot
   express that at all. The general constructor made it a one-liner.
3. **`InstrumentSpec::min_notional: Option<Decimal>`.** ADR-0025's minus column called
   Hyperliquid's hard-coded `MIN_ORDER_NOTIONAL_USD` "a population weakness" and predicted
   "a CEX publishes its own in `exchangeInfo`". It does, **per symbol**: 50 USDT on
   BTCUSDT, 20 on ETHUSDT, 5 on most, 0.001 on the BTC-quoted `ETHBTC`. A constant would be
   wrong four ways in one response. The type was right; only Hyperliquid's population was
   weak.
4. **`Ticker::ts_venue: Option<Nanos>`.** ADR-0011 modelled Hyperliquid's missing timestamp
   as an `Option` and rejected "one `ts_event` field with a doc comment". Binance's
   `markPriceUpdate` carries `E`, so this is the **first time that `Option` has been
   `Some`** — the same type, the same consumer, and `is_venue_timed()` answering differently
   for two adapters that know nothing about each other. Under the rejected design the two
   feeds would be indistinguishable and the parity harness would compare orderings that were
   never comparable.

### Seven places it did not fit

Each is evidence, and none of them was edited — every one is in a crate this workstream does
not own.

1. **`SizeGrid` has no `min_qty`, and the gap is reachable and silent.** Six symbols on the
   captured universe have `LOT_SIZE.minQty != stepSize` — `SPELLUSDT` steps by 1 with a
   minimum of 100, `TUTUSDT` steps by 0.1 with a minimum of 1. A size of 1 SPELL is on the
   lot, clears the $5 minimum notional, passes `InstrumentSpec::check`, passes the encoder,
   and comes back `-1013 Filter failure: LOT_SIZE`. That is exactly the "well-formed action,
   valid signature, refusal" shape ADR-0025 exists to eliminate. Pinned by
   `an_order_the_port_calls_valid_can_still_break_a_filter_it_cannot_hold`.
   **Change needed** (`crates/axon-providers/src/instrument.rs`): `SizeGrid` gains
   `min_qty: Option<Decimal>` and `max_qty: Option<Decimal>`, checked in
   `InstrumentSpec::check` — with `min_qty` carrying the reduce-only exemption `min_notional`
   already has, because refusing to close a sub-minimum position strands it. ADR-0025 named
   these as out-of-scope #6 with "no venue behind them today"; there is one now.
2. **`InstrumentSpec` holds one `SizeGrid`, and Binance has two.** `MARKET_LOT_SIZE` is a
   separate lot that applies only to market orders and genuinely differs: `ARKUSDT` steps by
   3 as a limit and by 1 as a market, with maxima of 3 333 and 1 000 000; BTCUSDT's market
   maximum is 120 against a limit maximum of 1 000. This venue has *native* market orders, so
   the case is live rather than theoretical. **Change needed:** `InstrumentSpec` gains
   `market_size: Option<SizeGrid>` and `check` selects on `req.order_type`. Not attempted
   here — it changes a `Copy` struct every planner path holds.
3. **`axon_execution::InFlight::CAPACITY` assumes the venue issues the id.** Its comment
   says "the bound is on the number of instruments a venue lists and not on anything we
   choose"; on this venue we *do* choose, and 730 rows against a bound of 1024 leaves 29%
   headroom. Past the bound the per-symbol gate silently becomes the global one — no error,
   no counter, no log line. `SymbolTable::exceeds_inflight_bound()` computes it and the live
   test asserts it is zero, which is a smoke alarm rather than a fix. **Change needed**
   (`crates/axon-execution/src/inflight.rs`): either raise `CAPACITY` to 4096 (512 bytes) or
   surface the overflow count on the status line. This lands beside M9's per-symbol work.
4. **`ExecutionClient::cancel_all()` takes no symbol, and no venue can honour it in one
   request.** Binance's `DELETE /fapi/v1/allOpenOrders` is **per symbol**, so a venue-wide
   flatten is N requests; Hyperliquid has no native cancel-all either and reads open orders
   first. The port models a request neither venue has. **Change needed** (`traits.rs`):
   `cancel_all(&self, symbol: Option<SymbolId>)`, or an explicit contract that the adapter
   fans out — today each adapter invents its own answer.
5. **`axon_core::Candle` cannot say "not final".** Decision 6 above. **Change needed**
   (`crates/axon-core/src/market.rs`): `Candle` gains `is_final: bool`, and the two adapters'
   decoders stop deciding on the consumer's behalf. It is a serialized type on the parity
   path, so it is not a small change.
6. **`axon_providers::Signer` fits this venue exactly and does not fit the first one.**
   `fn sign(&self, payload: &[u8]) -> Vec<u8>` is precisely Binance's shape. `HlSigner` does
   **not** implement it — an EIP-712 signature is over a *typed structured* payload and
   `&[u8]` cannot carry the domain separator. So the port has a signing trait the first venue
   could not use and the second fits exactly. That is not an argument for deleting it; it is
   evidence the trait was written for CEXes and the DEX is the exception, which is the
   reverse of how the codebase currently reads. No change needed — the finding is that
   ADR-0004's `Credentials`/`Signer` seam was better than one venue could demonstrate.
7. **`Tif` has no `GTD`, and `OrderRequest` still has no order lifetime.** Binance offers
   `GTD` natively with an `goodTillDate` parameter. That is not a missing TIF: it is the
   *order lifetime distinct from `ttl_ms`* that the handoff and
   [ADR-0020](0020-runtime-intent-source.md) §4 already name as owed, and this venue would
   implement it in one parameter where Hyperliquid needs a client-side timer. Recorded as
   evidence for M9's field, not proposed as a `Tif` arm.

Two smaller ones, for completeness: `OrderRequest::price: Option<Decimal>` is `None` on a
live path for the first time here (Hyperliquid's encoder refuses that outright, correctly,
for a venue with no native market order); and `CancelId` carries a `SymbolId` where this
venue needs the symbol *string*, so `cancel_params` takes the symbol table that
Hyperliquid's `cancel_action` does not need. Neither required a change.

## Consequences

- **+** ADR-0004's claim survives its first real test. A maximally different second venue is
  one new crate and one line in the workspace `members` list. No crate outside
  `crates/axon-provider-binance/` was edited.
- **+** Four speculative design decisions are now demonstrated rather than argued:
  `PriceGrid`'s composed shape, `SizeGrid::step`, per-instrument `min_notional`, and
  `Ticker::ts_venue`'s `Option`. The last one in particular could not have been validated by
  any amount of work on Hyperliquid alone.
- **+** The seven misfits are all *additive* — fields on existing types, one trait signature
  — and none of them requires an enum arm keyed on a venue name. The abstraction is
  incomplete, not wrong-shaped, which is the distinction that decides whether it was worth
  building.
- **−** **The execution half is unfinished by choice and unusable as it stands.** There is no
  HMAC, no order client, no account (user-data) stream, no `listenKey` lifecycle, no
  reconciliation and no `AccountState` implementation. ADR-0010's whole execution-report
  path has no Binance adapter. A reader must not mistake "the encoding is tested" for "orders
  can be sent".
- **−** **The market-data path is observed on testnet and nothing else is observed at all.**
  See *Verification*.
- **−** Symbol ids are derived, not persisted, so a delisting re-maps every id above it and
  invalidates every capture taken before it. This is strictly worse than Hyperliquid, where
  the venue issues the id — and it is invisible: a replay after a delisting runs clean and
  describes a different session.
- **−** The book is a periodic snapshot (20 levels at 100 ms), not the venue's full depth. The
  incremental stream is refused, so nothing here can reconstruct a deeper book, and the
  `U`/`u`/`pu` sequence numbers that would let it are decoded and discarded. A gap in the
  snapshot stream is undetectable.
- **−** `open_interest` is `None` on every `Ticker` this adapter produces. Binance publishes
  it on REST only, at one weight per symbol per call, and stamping a polled value onto a
  streamed frame would make a number that is up to a poll interval old look live.
- **−** Two REST requests build one universe (`exchangeInfo` + `fundingInfo`), which reopens
  the seam ADR-0025 closed for Hyperliquid by insisting on one `meta` read. The venue offers
  no single endpoint carrying both. Mitigated only by the fact that the second one narrows an
  assumption we can already live with.
- **−** `Capabilities::rate_limit_model` is one value and this venue has two independent
  budgets: weight-per-IP (2 400/min in production, 6 000 on testnet) and orders-per-account
  (1 200/min and 300/10 s). There is no `RateGovernor` for Binance at all — Hyperliquid's is
  volume-gated and shares nothing.
- **−** Fixtures come from testnet, which carries an undocumented `st` field on every frame
  and a `nq` on `aggTrade`. They are decoded as extras and ignored, so a production frame
  without them decodes identically — but the fixtures are not literally production bytes and
  nobody should claim they are.
- **−→+** **This adapter shipped with the reconnect-backoff defect a Hyperliquid soak later
  exposed, and it is fixed here rather than left as a known copy.** The connect loop reset its
  wait only when `run_once` returned `Ok(())` and doubled it only on `Err`, with *no sleep at
  all* on the `Ok` path. It was re-measured for this venue rather than patched by analogy, and
  the measurements moved two of the three arguments:
  - A severed link never reaches `Ok` — that is a `tungstenite` property, not a venue one, and
    it is now proven against *this* client over loopback (`ResetWithoutClosingHandshake`). So
    the reset was unreachable from any network event and the wait only ever climbed.
  - Unlike Hyperliquid's, this venue's `Ok` path **is** reachable: it closes every connection
    at 24 h, and a Close frame returns `Ok`. The old no-sleep reset was therefore an
    unthrottled reconnect on a venue whose *normal* lifecycle includes a clean close. It still
    does not warrant an `Ok` branch — a 24 h connection clears the healthy threshold on
    duration alone — which is the evidence that judging the connection rather than its exit
    code is the venue-neutral shape.
  - The `heard_from_venue` condition is load-bearing here for the **opposite** reason it is on
    Hyperliquid. Measured: `/stream?streams=btcusdt@nosuchstream` is accepted and stays open
    with **zero frames, no error frame and no close**, so a stream-name typo produces a
    long-lived silent connection that a duration-only test would read as healthy. Hyperliquid
    needed the condition because hearing a frame there is too *cheap* to mean anything; this
    venue needs it because a connection can be up and mute indefinitely.
  A second defect was found beside it and closed: on bus shutdown the loop reconnected into a
  closed channel forever, and with no sleep on the `Ok` path that was a connect storm against
  the venue for as long as the process took to exit.

## Verification — what was run, and what it proves

Two `#[ignore]`d **read-only** tests were executed against Binance USD-M futures **testnet**
on 2026-07-26. No credential was used, none exists in this repository, and no code path in
this crate can place an order.

| Ran | Result |
|---|---|
| `live_exchange_info_still_decodes` | ✅ 570 perpetuals kept, grids built, `min_notional` present on all, `exceeds_inflight_bound() == 0`, 570/570 funding intervals published |
| `live_market_data_smoke` | ✅ combined stream connected; book, trade, BBO and mark-price events reached the bus in 8 s; every `markPriceUpdate` was venue-timed |

Five further single-connection **read-only probes** were run against the same endpoint to
settle the backoff question (see Consequences), and three of them are venue facts this crate
now depends on:

| Probe | Result |
|---|---|
| `/stream?streams=btcusdt@nosuchstream` | accepted, held open 6 s, **zero frames, no error frame, no Close** |
| `/stream?streams=` (empty) | accepted, held open, zero frames |
| a non-JSON message on a live socket | `{"error":{"code":3,"msg":"Invalid JSON: …"}}` — an error frame on a healthy socket, the first live confirmation of `venue_error`'s shape |
| a `SUBSCRIBE` for an unknown stream on a live socket | no error frame at all in 6 s; the good stream kept flowing |
| `/stream/nonsense` | HTTP 404 at the handshake, not a WebSocket close |
| time from socket open to first frame | 128 ms (`depth20@100ms`), 290 ms (`aggTrade`) — far inside the 30 s healthy threshold |

That is the whole of what has been observed. **Not observed, and not claimable:** any signed
request; any order, cancel or modify; any account/user-data stream; **any reconnection against
the venue, any induced outage, any soak — the backoff fix is verified over loopback and by
unit test, never against Binance**; the documented 24 h close, which cannot be reached in a
session; whether the venue's own 3-minute server ping satisfies `heard_from_venue` on a
subscription that is otherwise silent (its failure direction is safe — it withholds a reset
rather than granting one); mainnet, which is geo-blocked from this host; and any interaction
between this adapter and `axon-runtime`, which has no Binance wiring.

Everything else is offline: 93 tests, network-free, of which the decoder tests run against
frames captured byte for byte off the venue's own socket (an 85-second, 2 353-frame capture
of BTCUSDT and ETHUSDT across six streams, plus a real `exchangeInfo` response subsetted to
seven instructive symbols and a real `GET /fapi/v1/depth` body). The negative cases are
**hand-derived**, each by editing one field of a captured frame, and each says so where it
is written: an empty book side, an empty index price, an empty funding rate, a level with a
third element, an unmapped symbol, a non-numeric price and an unmapped candle interval. The
`exchange_info` failure cases (a dated contract, a missing `MIN_NOTIONAL`, a response with
nothing tradable in it) are hand-written in the venue's shape rather than captured, because
the captured universe does not contain them.

**This adapter is offline-verified with a read-only market-data smoke test on testnet. It has
never traded, and no part of it may be called proven.**

## Out of scope, named

1. **The HMAC.** Needs, in the root `[workspace.dependencies]`:
   `hmac = "0.12"` and `sha2 = "0.10"` (both already in `Cargo.lock` as transitive
   dependencies of `alloy-signer-local`), then `hmac = { workspace = true }` and
   `sha2 = { workspace = true }` in this crate. The implementation is a `Signer` whose
   `sign` is `Hmac::<Sha256>::new_from_slice(secret)?.chain_update(payload).finalize()`,
   landed against the known answer already pinned in `sign.rs`.
2. **An `ExecutionClient`.** Signing is a prerequisite; so is a decision about *where a
   Binance key lives*, which is a security question and not an adapter one.
3. **The user-data stream.** `POST /fapi/v1/listenKey`, kept alive every 30 minutes, over a
   dedicated socket. It is the entire `ExecEvent` half of ADR-0010 and it needs a key.
4. **`AccountState`.** `GET /fapi/v2/positionRisk` and `/fapi/v2/balance` are signed.
5. **A rate governor.** Two budgets, weight-per-endpoint, and the `X-MBX-USED-WEIGHT-1M`
   response header that reports consumption — none of which Hyperliquid's volume-gated
   `RateGovernor` models.
6. **Spot and COIN-M.** Different products, different frame shapes, a different venue name.
   Spot's partial-depth frame carries no event time at all, which is a `ts_venue`-shaped
   problem in the book rather than the ticker and would need its own decision.
7. **The incremental depth stream**, and with it any book deeper than 20 levels. It needs a
   REST-seed-plus-`U`/`u`/`pu`-resync state machine, which is a real piece of engineering
   and a real source of silent divergence.
8. **Persisted symbol ids.** Decision 4's hole. It belongs with ADR-0025 out-of-scope #2 and
   the log-schema work, not in an adapter.
9. **`PERCENT_PRICE`, `MAX_NUM_ORDERS`, `minPrice`/`maxPrice`, `maxQty`.** Filters the port
   cannot hold; `PERCENT_PRICE` in particular collides with the planner's own `price_band`
   and neither knows about the other.

See [ADR-0004](0004-provider-abstraction-layer.md) (the claim this tests),
[ADR-0008](0008-market-data-bus-and-ws-ingest.md) (the decoder/client split this mirrors),
[ADR-0011](0011-ticker-and-mark-price-feed.md) (`ts_venue` and `Funding`, both vindicated
here), [ADR-0025](0025-instrument-precision-and-rounding.md) (the grids this populates, and
the minus columns it collects on), [04](../04-provider-abstraction.md),
[08 — Phase 7](../08-roadmap.md).
