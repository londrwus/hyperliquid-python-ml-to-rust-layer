# 08 — Roadmap

Phased path from "docs only" to "live trading," then to "HFT-capable." Each phase has a clear
exit criterion. No dates — gated on completion, not calendar.

## Phase 0 — Research & design ✅ (this repo, now)

- [x] Research: ML inference in Rust, Python↔Rust IPC, Hyperliquid, hybrid quant architecture.
- [x] Architecture, boundary decision (B → A path), provider abstraction, fidelity plan.
- [x] Decision records (ADRs), glossary, this roadmap.
- **Exit:** the design is coherent and the big decisions are recorded. *(Done.)*

## Phase 1 — Contracts & skeleton ✅ (foundation landed)

Define the seams before writing behaviour.

- [x] Finalize the **`Signal` schema** (target-position default) and **fixed-layout record**
      in [`contracts/schema.toml`](../contracts/README.md) — the single source of truth for
      Python + Rust ([ADR-0006](adr/0006-signal-schema-and-spsc-ring.md)).
- [x] Finalize the **config** schema (`axon-runtime` config); **feature-spec** stubbed in
      `axon.features` (fleshed out in Phase 5).
- [x] Choose the **shared-memory implementation** (hand-rolled cache-padded SPSC ring over an
      mmap file) and settle the **Linux/Windows dev story** (WSL2 primary, portable mmap for
      Windows — [ADR-0007](adr/0007-linux-wsl2-dev-target.md), now superseded by
      [ADR-0024](adr/0024-native-ubuntu-dev-target.md): development is native Ubuntu 24.04 and
      Windows is a **CI claim** rather than a place anyone compiles, which is exactly why that
      CI cell has to stay).
- [x] Stand up the **Rust workspace** (`crates/*`, 10 crates) and **Python package** (`axon`)
      with trait/interface stubs — plus tested core pieces (SPSC ring, position math, risk,
      order book).
- **Exit:** ✅ both languages compile against a shared, versioned contract; a byte round-trips
  Python → ring → Rust **and back**, byte-identical (53 tests green; CI on Linux + Windows).

## Phase 2 — Market data & core loop (Rust) ✅ (increments 1–4 landed; the soak is run)

Landed in increment 1 — the **offline-testable backbone** ([ADR-0008](adr/0008-market-data-bus-and-ws-ingest.md)):

- [x] `axon-core`: in-process **bus** (bounded crossbeam channel), normalized `market`
      vocabulary + `Event`, and the deterministic **loop** driver (`event_loop`). Clock +
      `TimedQueue` were already in place.
- [x] `axon-marketdata`: `MarketDataProcessor` — the core-side `EventHandler` maintaining
      per-symbol order books + a BBO/last-trade cache.
- [x] `axon-provider-hyperliquid`: `SymbolMap`, pure WS **decoders** (l2Book/trades/bbo,
      fixed-point, unit-tested vs captured frames), subscription builders, and the
      `tokio-tungstenite` WS **client** with heartbeat + reconnect/resubscribe + REST
      snapshot seeding. Live connection behind an `#[ignore]`d smoke test + the
      `live_book` example.

Increment 2 added: **candle** feed (OHLCV vocabulary + decoder + cache) and **real symbol
resolution** — `meta` universe decode/fetch so `SymbolMap` uses true asset indices.

Increment 3 closed the remaining items:

- [x] **Ticker / mark-price (`activeAssetCtx`) feed** ([ADR-0011](adr/0011-ticker-and-mark-price-feed.md)):
      normalized `Ticker` + `Funding {rate, interval}` in `axon-core`, the perp decoder (spot
      rejected on two independent layers), and a per-symbol cache exposing `mark_px()` as
      `(px, ts)`. The decoder is tested against a byte-for-byte testnet capture. Note what the
      capture revealed: **`activeAssetCtx` carries no timestamp**, so `Ticker` splits
      `ts_venue: Option<Nanos>` from `ts_ingest` rather than passing receipt time off as event
      time — the one non-reproducible feed is now *detectable* instead of indistinguishable.
- [x] **Market-data Rust→Python ring** (`docs/01` step 2;
      [ADR-0012](adr/0012-market-data-ring-and-multi-record-contract.md)): a second record
      (`MdSlice`, 128 B) in `contracts/schema.toml`, `axon-ipc` generalized to
      `RingProducer<R>`/`RingConsumer<R>` over a `Record` trait, an explicit `record_kind` in
      the ring header so a reader can never map the wrong record onto a matching stride, and
      `axon.marketdata` reading batches with drop accounting.

Increment 4 gave that ring a **production writer**, collapsed two answers to "where is the top of
book" into one, and made two silent assumptions catchable:

- [x] **`MdPublisher`** (`axon-runtime`) sits at the end of `CoreHandler`'s fan-out, on the core
      thread, and turns the core's own event stream into `MdSlice`s — so Python computes features
      on the book the executing core saw rather than on a second connection to the same venue.
      The write trigger is an explicit `MdWritePolicy` (`on_change` default, `every_update`) and
      both variants are pure functions of the event stream. There is deliberately **no
      time-based cadence**: a wall-clock sampler makes the published stream depend on how fast
      the machine drained the bus, and the Phase-5 harness compares a live record against a
      backtest record for record. A full ring drops the newest slice and *spends* a `seq` doing
      it — exactly the gap `MdRingConsumer.dropped` infers a loss from — while a coalesced update
      spends none, so suppression is never mistaken for loss. Proven across both languages by
      driving the real `axon` binary from pytest and comparing every byte against fixtures Python
      builds from `contracts/schema.toml`. `[md_ring]` is **off by default**, because this end
      *creates and truncates* its file; validation refuses a path equal to the signal ring's,
      which would truncate the file Python is producing into. **The publisher has never run
      against a live venue**: every test drives it from canned or offline events, so the default
      capacity of 4096 and the value of `on_change` coalescing on a real `l2Book` feed at a few
      thousand updates a second are reasoned rather than measured.
- [x] **A second record, so a Rust feed can reach a Python `Bar`**
      ([ADR-0028](adr/0028-market-data-bars-and-the-ticker-tail.md)). `MdBar` is its own record on
      its own ring, at the same 128-byte stride — *not* a `kind` inside `MdSlice`, because two
      consecutive identical bars are two facts and `OnChange` must never coalesce one (a
      coalesced bar silently shortens every rolling feature window), because `MdSlice.kind`
      describes the update's *cause* rather than the record's type, and because the two share no
      fields past the header. The identical stride is deliberate: it makes ADR-0012 §3's
      `record_kind` load-bearing instead of belt-and-braces, since `record_size` can no longer
      discriminate at all, and both languages refuse the wrong ring by name. Only **closed** bars
      cross, and closure is *derived* rather than believed — the publisher holds each
      instrument+interval's latest frame and emits it when the venue starts the next one. That is
      not fussiness: 111 of 112 measured Binance `kline_1m` frames were in progress and every one
      carried the same close stamp, so a partial and a close cannot be told apart by timestamp.
      The costs are stated rather than hidden: a bar arrives one frame after its close, and a
      session's last bar is never published.
- [x] **The record's reserved bytes are spent, and on what ADR-0012 held them for.** `MdSlice` v2
      carries the venue's mark, the index it tracks, funding as a rate *paired with its interval*,
      and the mark's **two** clocks. The stride did not move, which is what reserving them was
      for. The tail rides a slice rather than having a record of its own because a ticker has no
      event time to stamp one with — `activeAssetCtx` carries none (ADR-0011), and a record
      ordered on our receipt clock does not replay; that is the same refusal that stops the
      publisher emitting on a `Ticker` at all, and it stands. `mark_ts_venue == 0` says exactly
      that, and the Binance adapter is the first evidence that both cases are real rather than
      argued. The mark's clocks are excluded from the `OnChange` change test, or a receipt stamp
      moving on every ticker frame would degrade `on_change` into `every_update` on any session
      with a ticker feed. `MdSlice` now has no reserved bytes left: the next change to it is a
      stride change or an `MdTicker` record.
- [x] **The startup banner names the market-data ring** — `md ring    : <path> (cap N, policy P);
      bars <path>` or `OFF - Python computes no features from this session`. The one place an
      operator learns whether a session publishes *before* it starts rather than an hour into one.
- [x] **One implementation of "what is the top of book"** (`axon_runtime::quote::top_of_book`),
      shared by the publisher and the planner — features computed against one book while orders
      are priced against another is the divergence that justified a single venue connection in
      the first place. The `bbo` feed answers while it is still speaking, the L2 book takes over
      when it is not, and a quote older than the mark window collapses into *no quote at all*.
      Ageing a book needed a per-symbol stamp: `MarketDataProcessor::last_ts` is global, so a
      frozen book on one instrument looked current for as long as any other instrument traded.
      That stamp is now `MarketDataProcessor::book_ts`, kept beside the levels it describes.
- [x] **The funding interval is now an assumption that can be caught being wrong.** The
      hard-coded constant is `ASSUMED_FUNDING_INTERVAL_NS`, and `ws/funding.rs` measures
      Hyperliquid's real period from `POST /info {"type":"fundingHistory"}` — the *smallest* gap
      between periods, so a missed period does not read as a doubled interval and an eight-hourly
      venue is still caught. A live session cross-checks once and logs either way, never failing
      the session. It is **reported, not applied**: a venue that changed its schedule would log
      `CHANGED` every session while every `Ticker` kept carrying the old constant until a human
      edited it — so every carry number would still be wrong, just no longer silently.
- [x] **A candle is stamped `T + 1 ms`, on both sides.** Hyperliquid's `T` is the bar's *last*
      millisecond, so a bar stamped `T` sorts equal to every trade printed inside it and an
      event-time sort can hand a strategy the closed bar before the tick that closed it.
      `decode_candle` and `axon.strategies.data.CLOSE_STAMP_OFFSET_MS` now agree; while they did
      not, an `align_by_event_time` between an online bar feature and its offline recompute
      intersected to **empty** and the feature-parity gate failed a long way from the cause.
- [x] **A candle never becomes the core's clock, because its timestamp is arithmetic.**
      `Candle::ts_event` is `open_time + interval` — the *same* number on a bar's first frame
      and on its last — so it says where a bar sorts, never what time it is. Hyperliquid
      republishes the bar it is still filling and marks none of them final: **1 321 candle
      frames over 12.9 minutes described 69 bars**, 63 of them republished, every frame of a
      bar carrying the same `T`, and **1 317 of the 1 321 received before that `T`** — one BTC
      5-minute bar republished 192 times. `CoreHandler` advanced `last_ts` on all of them, so
      any session subscribing candles ran a clock up to a full interval ahead: every signal on
      the ring aged against the future and refused as `expired`, and the pass schedule — whose
      subtraction is signed by design — left anchored past any event that would arrive, so **no
      further intent pass ran at all**. The session went on printing `OK` with `lag 0ms`,
      because `data_lag_ms` saturates. Closed in `handler::advances_the_clock`: only an event
      that *reports* a moment may move the clock, which needs no knowledge of whether a bar is
      finished and therefore cannot be wrong about one. A `Candle::closed` flag was rejected on
      the evidence — Hyperliquid publishes no finality bit and sends nothing after `T`, so the
      field would read "forming" on 1 317 of 1 321 frames while costing a serialized
      parity-path type; and dropping non-final frames the way the Binance adapter does would,
      on this venue, drop the **bars themselves**, since the last frame before the close *is*
      the bar (byte-identical to `candleSnapshot` for 6 of 7 consecutive BTC minutes; the
      seventh short by 0.001 in `v`, traded in the final 35 ms). Finality stays **derived** —
      `mdring` closes a bar when a later `open_time` arrives, and
      `axon.strategies.data.closed_rows` applies the same rule offline. The one cost is named
      and on the status line: a session fed nothing *but* bars has no clock at all and runs no
      passes, which now reads `BARS BUT NO CLOCK n` instead of looking like a quiet strategy.
      Candles remain out of the default `session.feeds`, which is the only reason this was
      never hit live. Three real frames for one bar are committed byte-for-byte in
      `ws/decode.rs`, because the finding is that no timestamp can tell them apart.
- **Exit:** ✅ a live, correct Hyperliquid order book maintained in Rust with resilient
  reconnection; events flowing on the bus. *(Live run: `./run.sh book BTC`.)*
  **The soak is no longer owed.** 2026-07-26: **1 h 44 m 16 s** of one continuous read-only
  testnet session (BTC/ETH/SOL × `bbo`/`l2Book`/`activeAssetCtx`) through **36 induced network
  outages** — 0.3 s to 10 minutes, the TCP connection severed under the socket by a loopback
  relay — plus **3 whole-process `SIGSTOP` freezes**, with `--capture` on throughout. Reconnect
  and resubscribe held: **37 connections, all nine (instrument × feed) streams back in 35 of the
  37 windows**, and both exceptions are windows shorter than a quiet testnet `bbo`'s own gap, so
  **no half-restored subscription was seen** — the failure [ADR-0020](adr/0020-runtime-intent-source.md)
  §7 warns hides behind a live `activeAssetCtx`. `userFills` replayed its whole snapshot on all
  37 reconnects (666 fill records, 18 × 37) and the dedup on `trade_id` held every time:
  `ORPHAN FILLS 18` on the first status line and on all 1 222. RSS peaked at **26.9 MiB in the
  first minute** and then oscillated between **15.7 and 20.5 MiB** for the remaining 94 minutes;
  threads and descriptors flat. The tape replayed **byte-identically twice**, summary and
  13 939-row trace.
  The reconnect defect it found is **closed**: the backoff reset only when `run_once` returned a
  clean WebSocket close, which no severed link produces — a FIN with no Close frame arrives as
  `Connection reset without closing handshake` and an RST as `Connection reset by peer`, both
  the error path — so the reset branch was unreachable from any real network event and eight
  disconnects pinned the wait at its 30 s cap for the rest of the session, with **45.8 % of the
  soak running under `STALE MARKS`** and the risk gate refusing every risk-increasing order. It
  now resets on a connection that both heard from the venue and outlived that cap, which is what
  bounds a flapping endpoint to one attempt per 30 s while still letting a recovered link back
  in 250 ms. Verified as an A/B against testnet through the soak's own relay, same script both
  sides: an 0.8 s outage cost **1.97 s** of blackout as a session's first and **17.31 s** as its
  last under the old code, against **2.03 s** and **2.00 s** under the new. The soak's fourth
  finding is closed with it — the venue's pong carries no `data` field, so every heartbeat was
  logged as a decode error, while the test asserting pongs were ignored passed against an
  invented `{"channel":"pong","data":null}` the venue does not send. The real 18-byte frame is
  now a committed fixture. Both defects lived in the code that runs when a connection is
  **down**, which no smoke test and no CI gate can reach — see
  [ADR-0008](adr/0008-market-data-bus-and-ws-ingest.md)'s 2026-07-26 amendment.

## Phase 3 — Execution path (Rust) ✅ (the fill is run; the soak is run)

Landed — the **signing core** (inc 1, offline; [ADR-0009](adr/0009-hyperliquid-signing.md)) and
the **REST executor** (inc 2, testnet):

- [x] Hyperliquid **L1 signing** (`alloy` crypto behind `HlSigner`): msgpack action hash +
      phantom-agent EIP-712 + secp256k1, unit-tested (known key↔address vector, byte-exact
      msgpack layout, sign→recover round-trip). Env-var key + **agent-wallet** model.
- [x] **Nonce manager** (strictly-monotonic ms timestamps).
- [x] Order/cancel/**modify** **wire encoding** (`a/b/p/s/r/t/c`; `order`/`cancel`/`cancelByCloid`/
      `modify` actions).
- [x] **Venue tick/lot precision** ([ADR-0025](adr/0025-instrument-precision-and-rounding.md)) —
      `InstrumentSpec`/`PriceGrid`/`SizeGrid` as venue-neutral types on the provider port,
      populated from `meta.universe` (`szDecimals` is now a *required* decode field). The planner
      quantizes, because it is the only component that can see that a size rounding to flat means
      "no order"; the encoder **refuses** an off-grid order rather than silently rounding, because
      rounding at the wire makes the bytes sent differ from the recorded `Plan` the parity harness
      diffs. Worth stating why this went unnoticed for a whole phase: the only order ever placed
      live was post-only *at the touch*, priced from a number the venue itself emitted, so it was
      valid by accident — while every *computed* price (notably an urgency-3 IOC at
      `far_touch × (1 + slippage)`, nine significant figures) would have been rejected.
- [x] Async **`ExchangeClient`** implementing the `ExecutionClient` trait: REST `POST /exchange`
      place/place_batch/cancel/modify, with typed response parsing. Response decoding unit-tested
      offline; a live testnet place-then-cancel sits behind an `#[ignore]`d test. `CancelId` now
      carries the symbol (venues key cancels on `(asset, id)`).
- [x] **Verified live on testnet (2026-07-25):** `place_then_cancel_on_testnet` placed a real
      post-only order, saw it rest (`oid#56968034936`), and cancelled it. This exercises the whole
      chain against the real venue — msgpack action hash → phantom-agent EIP-712 → secp256k1 →
      nonce → REST envelope → response parsing, plus `meta` symbol resolution and WS book seeding.
      Note it ran on a **zero-balance** account: Hyperliquid does not margin-check a post-only order
      resting well off the book, so *resting* is proven but **fills are not** — fill semantics need a
      funded account.
- [x] `axon-risk`: pre-trade checks on the hot path (already in place since Phase 1).
- [x] **Risk wired onto submit** — `axon-execution`'s `GuardedClient<C, X>` *is* an
      `ExecutionClient`, so the gate is structural rather than a convention a call site can forget.
      Batches are checked cumulatively (each order projected onto a running position as if it fills
      in full); cancels are deliberately never gated; a missing mark price fails closed for anything
      that could add exposure but still lets reduce-only through.

Landed in increment 3 (all offline-tested; ADR-0010):

- [x] **Execution vocabulary** in `axon-core` (`exec.rs`): `Fill`, `OrderUpdate`, `AccountSnapshot`,
      `ExecEvent`, and `Event::Exec` — one bus, so a fill orders against the trade that caused it.
      `OrderStatus` moved down from `axon-providers` (which re-exports it).
- [x] **`OrderTracker`** (`axon-execution`): reconciles acks against venue truth. Fill dedup on
      `trade_id` (a reconnect replays snapshots), adoption of orders we never submitted (a restart
      leaves them resting), monotonic fill quantities (the two channels' clocks disagree), and
      `risk_position` = filled **plus** worst-case resting.
- [x] **Risk gate wired onto submit**: `GuardedClient<C, X>` *is* an `ExecutionClient`, so the venue
      client is unreachable behind it. Cumulative batch checks, ungated cancels, fail-closed on a
      missing mark. `TrackerRiskContext` closes the sequential-orders aggregation gap.
- [x] **HL user channels**: `userEvents`/`userFills`/`orderUpdates` decoders → `ExecEvent`, the
      29-value status table, no-auth address subscriptions, replay on reconnect. A decode failure
      logs and drops the frame rather than tearing down the socket (a `userFills` snapshot would
      otherwise redeliver the bad entry forever).
- [x] **`POST /info` reads** (`info.rs`): `openOrders`/`frontendOpenOrders`/`historicalOrders`/
      `orderStatus`/`clearinghouseState`/`extraAgents`/`userRateLimit` — needed because
      `orderUpdates` **never snapshots**, so post-restart open state must come from REST.
- [x] **`scheduleCancel` dead-man's switch** (`arm_dead_mans_switch`), the only protection that
      survives this process dying; 5 s minimum lead enforced locally, `time` omitted (not `null`) to
      disarm because field presence changes the signed hash.
- [x] **Rate governor** (`governor.rs`): separate IP-weight and address-credit budgets, with places
      structurally unable to reach the cancel allowance (`place_ceiling < limit < cancel_allowance`)
      — that inequality chain *is* the guarantee that an unwind is always possible.
- [x] **Native `cancel_all`** via a `frontendOpenOrders` sweep (HL has no native action), and
      **`approveAgent`** — the user-signed EIP-712 scheme (`HyperliquidSignTransaction` domain),
      which is what makes a leaked hot key survivable.

Landed in increment 4 — the **runtime supervisor**
([ADR-0013](adr/0013-runtime-supervision-and-safety-loop.md)):

- [x] **`axon-runtime` is a real session**, not a smoke check. `cargo run --bin axon` runs an
      offline session (bus + deterministic loop + tracker + mark cache over a canned stream) with
      no network and no key; `sandbox`/`live` additionally spawn the WS market-data and user
      channels, the re-arming dead-man's-switch, and the `/info` reconciliation poll. The core
      runs on its own `std::thread` so tokio is never in scope for it.
- [x] **The submit pipeline is a stack of types**, each unbypassable:
      `HaltableClient → GuardedClient → GovernedClient → ExchangeClient`. Risk sits *above* rate,
      so a risk-refused order never spends finite venue budget; cancels are charged but never
      refused.
- [x] **`MarkCache` is fed and expires.** Precedence is venue mark → book mid, never last trade.
      Staleness needs a two-source clock: a dead feed emits no events, so event time freezes and
      would call every stale price fresh — backtests never advance the wall-clock side, so replay
      stays deterministic.
- [x] **Graded dead-man's-switch escalation**: retry while >1 interval of protection remains,
      halt placements at ≤1 (before the deadline, since orders placed in the last interval are the
      ones stranded), shut down at 0. The session **starts halted** and the first successful arm
      releases it.
- [x] **Agent-wallet tooling**: `examples/approve_agent.rs` — dry-run by default, fresh key from
      the OS CSPRNG written at mode 0600, user-signed EIP-712 (never the L1 scheme), mainnet
      guarded by retyping the account address. `wallet_info` now classifies the configured key
      against the venue's own `extraAgents`.
- [x] **Live agent wallet in use (2026-07-25).** Axon now signs as approved agent `rustml`
      for a master account holding **999 testnet USDC**. The containment
      property is real: the configured key can trade and cannot withdraw.

Hardening landed alongside Phase 4's join, and it belongs here because it is about the venue path
rather than about intents:

- [x] **Every `reqwest` client on the Hyperliquid path carries an explicit deadline.** `reqwest`
      has no default request timeout, so `ExchangeClient`'s bare `Client::new()` hung a
      `place_order` for as long as the kernel kept the socket, and `info.rs` had the identical
      defect on the *reconciliation* path — where a wedge means the session keeps trading against
      a position it has stopped re-reading. Both now stop at 10 s, and the two halves of `/info`
      share one constant so they cannot drift apart. The test drives a real loopback blackhole (a
      bound-but-never-accepted listener, so the handshake completes from the backlog and no
      response ever arrives) and asserts `is_timeout` specifically — a connect failure would have
      proven nothing about the hang.
- [x] **Shutdown can abort a submit task it could not join**, and says so. Dropping a `JoinHandle`
      only detaches the task, so an in-flight `place_order` could complete *after* the cancel-all
      sweep had read an empty book — the venue would rest that order behind a dead-man's switch
      the same shutdown had just disarmed. An abort is still not proof the placement never left,
      so the switch is left **armed** in that case, exactly as a failed sweep does, at the cost of
      one of the venue's ten daily triggers.

Remaining for Phase 3:

- [x] **Fill verification — run against testnet on 2026-07-26, and it passed on the first real
      run.** `./run.sh live` drives all six live tests green.
      `crates/axon-provider-hyperliquid/tests/live_fill_testnet.rs` resolved BTC from the venue's
      own `meta` universe (testnet index **3**, `szDecimals` 5, lot `0.00001`, tick 1 at the
      touch), crossed with `0.00019 @ 64679` ($12.29) against an ask of `64357.0`, and reported
      `PASS: fill observed, reconciled against the venue - the account was flattened, confirmed
      by clearinghouseState.` The account was left flat and order-free; two filled round trips
      cost **0.0259 USDC** of the 999.
      What the run settled, none of which had been observed before: **`orderUpdates` does emit
      for an IOC that fills outright** — two frames, `Resting` then `Filled`, so the venue
      publishes a resting transition even for an order that never rests. **The venue does echo
      our `cloid` on `userFills`**, all 16 bytes including our own leg tag, so attribution took
      the strong path and the oid fallback never fired. The REST `/exchange` ack for a marketable
      IOC comes back `status=Filled` synchronously. `OrderTracker` attribution held against the
      venue — no new orphans, position delta == executed, nothing left resting — and
      `clearinghouseState.szi == 0.00019`, which is
      [ADR-0010](adr/0010-execution-events-and-reconciliation.md)'s reconciliation claim checked
      against the venue rather than a fixture.
      **[ADR-0025](adr/0025-instrument-precision-and-rounding.md)'s quantizer produced prices the
      venue accepted first try in both directions** (marketable buy ceiled to `64679`,
      reduce-only sell floored to `64021`). A later run replayed the earlier run's two fills in
      the `userFills` snapshot and the delta-based orphan baselines absorbed them exactly as
      designed — an absolute `orphan_fills() == 0` would have failed — and one reduce-only IOC
      swept two book levels into **two fills sharing one oid and one cloid**, the multi-fill
      path, live. Also established: **`/info` reads are free**, since `nRequestsUsed` counts only
      signed `/exchange` actions.
      Getting there required fixing a bug that had kept the test unreachable: `live_info_reads`
      and `live_user_channel_subscribe_smoke` demanded `AXON_HL_USER` / `AXON_HL_ACCOUNT`, which
      are defined nowhere in this repo, and **cargo abandons a run at the first failing test
      *binary*** — so those two panics stopped `live_fill_testnet` from ever executing. An
      operator typing the documented command saw a failure from an unrelated test and never
      reached the one that mattered. Both now fall back to `AXON_HL_ACCOUNT_ADDRESS`.
      Separately, `place_then_cancel_on_testnet` was still doing its own venue arithmetic
      (`round_dp(0)` on the price, a hardcoded 4-decimal size) and was on-grid only by luck —
      integers are exempt from the five-significant-figure rule; it now goes through
      `InstrumentSpec::quantize`, which changed the wire from `0.0003` to `0.00029`. That is the
      second instance of that bug class, and the argument for a review rule rather than
      case-by-case fixes.
- [x] **The soak is run, and the two defects it found in *this* phase's code are closed.**
      2026-07-26: 1 h 44 m 16 s of one continuous read-only testnet session through 36 induced
      network outages and **3 whole-process `SIGSTOP` freezes** (harness in `scripts/soak/`,
      method and evidence in [07](07-parity-and-testing.md)). Reconciliation held for the whole
      run — roughly 415 `/info` polls, no failure and no divergence, including across a 10-minute
      blackout — and `userFills` replayed its entire snapshot on all 37 reconnects with the
      `trade_id` dedup absorbing every one. The reconnect half is closed in Phase 2 above. The
      freezes found these two:
      **The dead-man's switch escalated on re-arm *failures* and never on protection running
      out.** `Escalation` was computed only from `on_failure`, so
      [ADR-0013](adr/0013-runtime-supervision-and-safety-loop.md) §3's ladder — retry above one
      interval, halt placements at ≤1, shut down at 0 — was consulted only after a *failed*
      re-arm. The case the switch exists for is the opposite: a process that stalls while its
      arms keep succeeding. An 80 s `SIGSTOP` against a 60 s lead drove the status line to
      `dms 0s`, the venue-side switch having actually fired, with **no `HALTED`, no `UNPROTECTED`
      and nothing on stderr** — then `dms 55s` and healthy on the very next line. The loop now
      grades the same table on the elapsed **wall** clock before it arms, wall time being the one
      clock that can measure a deadline the venue holds (event time would call a frozen session
      infinitely protected; a monotonic `Instant` would miss a suspended machine). The claim is
      deliberately narrow and is written into the code: a **resumed** process escalates instead
      of continuing as if nothing happened — a stalled one still runs no code, and the venue-side
      deadline is still the only thing covering the gap itself. Ordinary jitter cannot trip it by
      arithmetic rather than by tuning: the lead is ≥3× the re-arm interval, so reaching the halt
      band takes a full missed beat. And a stall that recovers cannot hide, because the
      successful re-arm repairs every other field on the line — hence `late n` beside `dms` and
      `DMS PROTECTION LAPSED n` in the warnings. Verified against testnet on both rungs: a 42 s
      freeze resumed to `the re-arm loop did not run for 43 077 ms, only 15 767 ms of protection
      left - HALTING new orders`, then re-armed and traded on at `dms 58s (late 1)` for **zero**
      triggers; a 75 s freeze took the deadline with it, spent **one** of the ten daily triggers
      and ended the session. See [ADR-0013](adr/0013-runtime-supervision-and-safety-loop.md) §3's
      2026-07-26 amendment.
      **A replayed `userFills` snapshot dragged the core's event clock hours backwards.**
      `CoreHandler::on_event` *assigned* `last_ts = ts_event` rather than taking a maximum, and a
      replayed fill is a moment somebody genuinely observed — just a very old one, up to 2.94
      hours on this tape. Five status lines carried lags between 25.8 minutes and 2.49 hours with
      **`marks 3/3`** beside them, and three occurrences 820 s apart reported lags 820 064 ms
      apart: a clock pinned to a fixed historical instant, not a lagging one. `last_ts` is
      documented as the event-time **high-water mark**, and a high-water mark that goes down is
      not one, so it is now a maximum — in `MarketDataProcessor` as well, where the same
      assignment was latent because nothing outside that crate reads its global clock. The exact
      mirror of the forming-candle fix on the same line — candles pushed the clock forward, fills
      pull it back — and the `advances_the_clock` predicate that fix added was never the wrong
      half. What a maximum gives up: a genuinely out-of-order live stream no longer rewinds the
      clock, so `data_lag_ms` reports the age of the newest observation rather than of the latest
      arrival — the quantity anyone reading it wants, and the same clamp `RecordingSource` and
      `CapturedSignals` already had to apply downstream *for want of it*. Measured on the soak's
      own 5 031-record tape: **751 records arrive below the high-water mark in capture order, 317
      of them replayed fills, and the deepest backward step the old assignment took was 3.62
      hours.** Replay determinism is unaffected — two replays of that tape produce byte-identical
      summaries and byte-identical 5 031-row traces, under both `--order event-time` and
      `--order as-captured`.
- **Exit:** place/cancel/modify orders on **testnet**, reconcile fills, survive
  disconnects — all via the provider trait. *(Place/cancel/modify and **a filled order** now
  proven live, with reconciliation checked against `clearinghouseState` rather than a fixture;
  supervision and the safety rails built and offline-verified. **Surviving disconnects is now
  observed rather than reasoned about** — 1 h 44 m and 36 induced outages plus 3 process
  freezes, [07](07-parity-and-testing.md) — and the two defects that run turned up are closed
  above rather than merely recorded.)*

## Phase 4 — The Python↔Rust bridge ✅ (proven live on testnet; see what that does *not* mean)

- [x] `axon-ipc` ring + `axon.signals` publisher; `axon.strategy` base class.
- [x] **Python side**: `StrategyContext` owns `seq`/`model_version`/`ttl` and binds `ts_event`
      from the event being handled — it imports no clock at all, so a strategy *cannot* stamp
      wall-clock time, which is the training–serving skew [03](03-ml-fidelity-and-features.md)
      names as the #1 silent quality leak. `emit_target` takes real units and does the only
      fixed-point conversion. `axon.live` adds a runner with a FIFO backpressure outbox (never
      coalescing — that would gap `seq`, destroying the only proof nothing was lost) and an
      mmap liveness beacon.
- [x] **Rust side**: `Signal` → order intents ([ADR-0014](adr/0014-signal-to-order-planning.md)).
      `SignalReader` validates layout → meaning → ordering → freshness; `Planner` emits the
      *delta* to target with an urgency→TIF table, price-band clamping, and a deterministic
      `cloid` so a replayed signal is idempotent at the venue rather than a doubled position.
- [x] **The runtime has an intent source** ([ADR-0020](adr/0020-runtime-intent-source.md)):
      signal ring → `SignalReader` → `Planner` → a bounded `crossbeam` queue → the
      `Haltable→Guarded→Governed→Exchange` pipeline ADR-0013 shipped without a caller. The seam
      stays where ADR-0008 put it — the reader and planner run inside the core loop's existing
      iteration with no tokio handle in scope, so a replayed session re-plans to the same orders
      with the same `cloid`s. A pass reads position, top of book and working orders **once**,
      under one tracker lock, at one `CoreHandler::last_ts()` — the same clock the reader ages
      the signal against — and the pass *schedule* is on that clock too, so which records share
      a pass (and therefore which are superseded rather than planned) is a function of the event
      stream and not of how fast the machine drained the bus. Two structural rules stop one
      target becoming two orders: within a pass only the newest signal per symbol is planned,
      and between passes the core will not plan again **for a symbol** while the edge still has
      an intent for that symbol in flight.
      `ttl_ms == 0` now means "the operator's ceiling" on **both** sides
      (`axon.strategy.context.TTL_OPERATOR_CEILING`), and `urgency` has named constants matching
      ADR-0014's table in both languages. A missing ring is a named degraded state
      (`SIGNAL RING DETACHED`), not a crash and not silence.
- [x] **The join is proven offline, byte for byte.** `cargo run --bin axon` replays canned
      signals through the real reader and planner and prints the side, size, price and TIF of
      every order it plans — because "the join produced two intents" is equally true of a build
      that sends the target instead of the delta. Two runs plan identical intents, cancels
      included.
- [x] **The debts ADR-0014 and ADR-0020 named in their own minus columns are paid, and paying
      three of them required the fourth.** `TrackedOrder` now carries `tif`, `reduce_only` and
      `placed_ts` as `Option`s that say *unknown* rather than as a plausible `Gtc` — the ack is
      the only moment either field is knowable, because no venue frame carries them. That
      retires `WorkingOrder::placed_by_us`, which could only ever answer *no*: a restarted
      session used to cancel and replace every order its predecessor left, paying queue position
      on all of them for a field the process no longer had rather than a field that was wrong.
      The planner gains a **no-op band** measured in basis points of the *resting* order — the
      thing at stake is what we would cancel, so it is the only honest denominator, and a
      fraction rather than an absolute quantity because one number cannot be right for BTC and a
      $138 coin at once. It forgives a size and never a price (a price that moved is a stale
      quote, which is what somebody else's taker is looking for) and never a reduce-only order
      (a flatten that lands 20 bps short has not flattened). It is **off by default**: a band is
      a bounded, deliberate position error, and one that appeared because a default changed
      would be an error nobody chose. The in-flight gate is **per symbol**
      (`axon_execution::InFlight`, an atomic bitset beside `HaltSwitch` — a `Mutex<HashSet>`
      would add a second poisonable lock between the core and the edge), so a slow submit on BTC
      no longer delays ETH; a target for a symbol still at the venue is *held* rather than
      dropped, and re-aged through `SignalReader::still_fresh` into the reader's own `expired`
      counter, which is what makes it a held record instead of the second queue ADR-0020 §3
      refused to build. And the stall watcher moved from occupancy to **progress**: with one
      batch in flight at a time "something is outstanding" was a usable wedge signal, and per
      symbol it describes a healthy multi-instrument session just as well — what only a wedge
      produces is a completion count that has stopped moving.
- [x] **A resting order can now be pulled on age, which nothing did before.**
      `PlannerConfig::max_order_age_ms` (60 s) bounds the leave-it-resting exception, and it had
      to land *with* the three above rather than after them: carrying a real TIF and adding a
      band both **widen** that exception, and an exception with no bound preserves orders nobody
      currently intends. **This is not `ttl_ms`** — that is a signal-*admission* window
      `SignalReader` consumes before the planner ever sees the record, clamped to
      `min(ttl_ms, intent.max_signal_age_ms)` with the ceiling at 2 s, so `perp_bar`'s
      `ttl_ms = 60_000` buys an order exactly nothing. It is also only *half* an order lifetime,
      and the half the planner can honestly own: the planner runs on a signal, so it cannot pull
      a quote from a strategy that has gone silent. The other half is a sweeper on the pass, and
      is unwritten. Both belong to the ADR `docs/adr/README.md` books as **0031**.
- [x] **End-to-end: a Python strategy drives real testnet execution.** **Proven live on
      2026-07-26** ([ADR-0026](adr/0026-python-driven-execution-at-a-venue.md)). A `TargetProbe`
      running on `axon.live.probe` read the core's own market-data ring, emitted a target position
      stamped with the **venue's** event time, and the runtime turned it into an IOC that filled at
      the ask Python had recorded — then flattened it on a `FLAG_CLOSE` and left the account flat.
      The chain is traceable end to end because the venue echoed a `cloid` Python and Rust each
      derived independently from [ADR-0014](adr/0014-signal-to-order-planning.md) §5: probe stdout
      → the session's own signal log → `sig 2/0 sent 2+0c` on the status line → `userFills`.
      Twelve fills across six sessions, `swept: true, disarmed: true` every time, `accountValue`
      998.974072 → 998.89646 and verified flat three times over.
      The missing piece was never code — it was a **cause**. Nothing could put a real target on a
      real ring at a moment a live core would admit it: `axon.live.mdfeed` (the market-data ring as
      strategy events) did not exist, and `python -m axon.live`'s synthetic feed stamps from 2023,
      so every record it has ever written is ~2.5 years stale against a 2 000 ms ceiling. It now
      says so, loudly, on every non-`--drain` run.
      `session::live_sandbox_session` is no longer the last word: it now declares
      `intent.enabled = false` — the only way to *say* read-only
      ([ADR-0020](adr/0020-runtime-intent-source.md) §9), which it previously only hoped for; it
      had been running with the intent source ON against the *default* ring path, so it would
      have obeyed any producer left there. `a_signal_on_the_ring_becomes_an_order_at_the_venue`
      beside it asserts on a counter that cannot move without a venue. Both have been run.
- **Exit:** a Python strategy drives real testnet execution through shared memory. **Met, and the
  line means only itself.** What was observed is that the *path* works. What was not observed, and
  is listed at equal length in [ADR-0026](adr/0026-python-driven-execution-at-a-venue.md): every
  order was an IOC that filled outright, so **zero cancels have ever been sent through this path
  at a venue** and the whole cancel/replace half of ADR-0014 §6 is untouched; twelve fills on one
  instrument over thirty minutes says nothing about a strategy that trades continuously; no
  reconnect, no bar, no model, and no mainnet. The probe has no view — its −$0.078 is the cost of
  the experiment, not a result.

## Phase 5 — Fidelity & parity harness ✅ (both halves of Boundary A are gated; the ladder is climbed)

- [x] `axon.models`: export + versioned registry
      ([ADR-0015](adr/0015-model-artifacts-and-registry.md)). Trees → native format, sklearn/NN →
      ONNX FP32; every export is round-tripped through a reload before it is allowed to exist,
      artifacts are immutable (re-writing a version is an error, not an overwrite), and the
      written bytes are audited for the silent FP16 downcast [ADR-0005](adr/0005-fp32-no-quantization.md)
      warns about.
- [x] Model-parity, feature-parity and drift gates
      ([ADR-0016](adr/0016-feature-spec-and-parity-gates.md)), with a versioned `FeatureSpec`
      whose fingerprint covers the library version as well as the recipe. Decision invariance is
      an `and`, not advice — there is a required test where the numeric criterion alone would
      have shipped a prediction that flips a trade.
- [x] `axon.backtest` + deterministic replay
      ([ADR-0018](adr/0018-event-capture-and-golden-replay.md)): capture to versioned JSONL,
      republish through the *real* bus and the *real* `MarketDataProcessor`, and a golden test
      asserting two replays of one log are byte-identical. It states plainly what it refuses to
      claim: a log is a recording, not a counterparty, so an order a replay would place gets no
      fill.
- [x] **A session can record itself, and the recording replays.** `axon --capture PATH` writes
      two files, because a replay needs both halves: the event log, and the *signals* the pass
      read, teed at the source rather than at the drain callback (which only sees accepted
      records — a log missing the expired ones would replay with every refusal counter at zero).
      The core thread holds only a non-blocking `try_send` onto a bounded queue; the writer is
      its own thread, because a `write(2)` behind a full page cache blocks the one thread that
      must keep draining market data, and a stalled core is a stale book. A recording that
      cannot keep up **stops rather than drops** — the inverse of the market-data ring's rule,
      because every `MdSlice` is full state the newest supersedes, while a log with a hole
      replays a session that never happened and replays it *successfully*. `events + missed` is
      always the whole session. The log earns its real name only by closing cleanly: the writer
      works on `<path>.partial` and renames on an explicit finish and on no other path, so a
      truncated log never sits where a harness reaches for it.
- [x] **The replayed chain is the production chain.** `axon-replay` drives the real
      `axon_runtime::CoreHandler` (book → marks → tracker) *and* the real
      `axon_runtime::IntentSource` (reader → planner), reimplementing nothing; the direction of
      the dependency is the load-bearing part (runtime→replay normal, replay→runtime **dev**),
      because a live session must be able to record itself and cargo permits the cycle only
      through a dev edge. Planned orders are deliberately **not** written into the tracker —
      that would invent an `OrderAck` the venue never gave — so positions still move only on the
      log's own fills. A poisoned tracker reads as `Cell::Absent` plus a `dropped_exec_events`
      counter rather than as zeros, because zeros are byte-identical to a genuinely flat session.
      Verified: `axon --capture` then `replay_log` reproduces the same two orders, the same
      cancel target, the same `cloid`s and the same `accepted 2 / rejected 1 / expired 1`.
      **And now on real traffic**: a 171-event Hyperliquid testnet capture — 12 real fills,
      4 order updates, BTC at testnet index 3 — replays through the production chain to a
      byte-identical trace twice over, and **25 of its 171 records arrived behind the
      event-time high-water mark**, so the two traversals produce visibly different runs and
      ADR-0018 §4's "compare a live reference only as-captured" stopped being an argument and
      became an observation. Read that 25 carefully, because the first reading of it was
      wrong: **all 25 sat behind a *receipt*-stamped ticker, and the venue's own market data
      was 0 of 37 out of order.** It measured our receipt clock, not Hyperliquid. §4's
      conclusion survives unchanged — the traversals differ by 25 whatever the cause, which is
      all §4 needs — but "15% of a real tape arrives late" was never a fact about the network.
      **What is still not shown is a grid-correct replay of a live session**: that tape was
      recorded before a log carried an instrument table, so it replays unconstrained and
      says so.
- [x] **A recording outlives the session, and carries the grid it planned on**
      ([ADR-0027](adr/0027-streaming-logs-and-the-replay-grid.md)). ADR-0018 accepted "a log
      is loaded whole" and set the capture's cap where an artifact stopped being *loadable*
      rather than where a disk stopped being writable, so a multi-hour soak stopped its own
      recording at 512 MiB. A log is streamed now: `as-captured` is **O(1)** — 4.8 MB of RSS
      over a 163 MiB, 410 400-record log — and `event-time` holds an index of about 24 bytes
      a record instead of the records, because ordering still cannot begin until the last
      record has been seen. The cap is a disk guard, `max_bytes = 0` turns it off, and
      rotation stays refused for ADR-0018's reason: rotated segments are files each of which
      replays a *different* session. (The bound that would have defeated all of this was in
      the harness rather than the log: `ChainProbe` retained every row unconditionally, at
      141 MB per 27 MiB of tape. It is now opt-in behind `--trace`.) Separately,
      `SCHEMA_VERSION` 1 → 2 puts the instrument table in the log as its first body line,
      closing the hole ADR-0025 named as one its own increment made **bigger** — a replay used
      to plan `Unconstrained` while every live session plans `Known` and rounds, so a golden
      diff of a real capture reported a strategy flip on every order the grid moved. A log
      written before the bump is **refused by name** rather than replayed loosely, because a
      loud accept is a quiet accept with extra words and `docs/07` gate #5 compares
      `PlannedOrder.price` exactly without reading stderr; `--example upcast_log` carries one
      forward as a command an operator types, declaring no grid, because that is the only true
      statement about it. The *signal* log is deliberately **not** upcastable: zero in
      `max_order_age_ms` means "the operator's ceiling" rather than "absent", so an upcast
      would replay orders aged against a lifetime the live session never set.
- [x] **Ahead of schedule — native Rust inference**
      ([ADR-0019](adr/0019-native-rust-inference-backends.md), the Boundary-A cornerstone):
      `tract` for ONNX and a hand-written XGBoost JSON reader that is **bit-identical** to
      XGBoost over the holdout set, with FP16 refused on the `ModelProto` before the graph is
      built.
- [x] **Compute offload to Modal** ([ADR-0017](adr/0017-compute-offload-to-modal.md)) via the
      sibling `hwsched` project: dry-run-before-spend enforced as a *type* (a single-use,
      digest-bound `Approval`), artifacts through Modal Volumes. Verified with three real CPU
      jobs whose remote digest matched the local run byte for byte — and, since 2026-07-26,
      with the **GPU submission path, which had never executed at all**: four runs on Tesla
      T4s, 8/8 tasks succeeded at width and 8/8 checksums identical across two independent
      runs with TF32 disabled, the last two from `axon.compute.entry:gpu_probe` in this tree.
      Three things that cost an hour each are now recorded rather than folklore. **The
      `workload` label is the price, not just the placement**: it selects the duration model,
      so identical work priced $0.0099 as `inference` and $1.24 as `ml_train_dnn` (2 s vs
      1800 s assumed per task) — 350× and a budget refusal at the eight-task width — while the
      timeout does not enter the estimate at all. **A return value must be plain-typed, not
      merely small**: `torch.__version__` is a `str` subclass, so returning it fails the run on
      the *client* after the GPU work is done and paid for; the Volume artifact survives, the
      return value does not. (`ModalProvider.status()` deserializes every `FunctionCall` with
      `fc.get(timeout=0)` and counts the exception as a *failed task*, which is why this
      presents as a crash rather than as a serialization error.) **The interpreter is half of
      `sys.path`**: `axon.compute` falls back to bare `python3`, and `./run.sh` activates a
      `.venv` whose `python3` has no `pydantic`, which is the whole reason the gate reports
      five more skips than a bare pytest run. The first strategy's own sweep was planned and
      priced through the same protocol ($0.0095/$0.0107/$0.0150, 36 CPU tasks, budget APPROVE)
      and **deliberately not spent** — the whole grid runs in 100 s on this box. The job exists
      so that the version which genuinely does not fit here is a parameter change rather than a
      new integration; only the FAKE provider has been driven end to end for it.
- [x] **The model-parity gate now crosses the language boundary**
      ([ADR-0021](adr/0021-rust-model-parity-gate.md)). Until this, both sides of every
      model-parity comparison were Python's, which is structurally incapable of failing on the
      thing Boundary A turns on. The unit is a **parity bundle**: a directory written by
      `axon.parity.rust_gate` from a real registry artifact, holding the model bytes, a holdout
      matrix as raw little-endian f32, Python's scores over exactly those bytes, and Python's
      discretized decisions. The serialization guarantee is mechanical rather than careful — the
      writer scores the matrix it read *back off disk*, so the recorded reference is by
      construction the answer to the question Rust asks, and no decimal round-trip exists on the
      path (a feature one ULP off does not move a tree prediction by one ULP; it moves the row to
      the other leaf). `axon_model::parity::ParityBundle` reads it with no Python and no ML
      libraries: bit equality for trees, 1e-5 for ONNX, **and** decision invariance, and a bundle
      cannot buy itself a looser criterion than its family allows. It was verified to fail before
      it was believed — swapping the tree margin accumulator to f64 reddens 77 of 128 rows at a
      worst delta of 4.8e-7 with no decision flipped, invisible to a 1e-5 tolerance and caught
      only because trees are held to bits.
- [x] **One real strategy has climbed rung 1, and the honest result is that it is not
      tradeable** ([ADR-0022](adr/0022-first-ml-strategy.md)). `axon.strategies.perp_bar` is a
      nine-column, scale-free `FeatureSpec` (`perp_bar/v1#a21328ed1532ecd4`) over closed OHLCV
      bars plus an XGBoost classifier of the sign of the four-hour forward return, fitted on real
      mainnet hourly BTC+ETH (4,999 bars each, 208 days) pulled from the public read-only
      `candleSnapshot`. Every feature has a **finite lookback** — no EMA, no expanding statistic
      — which is what makes the serving path's bounded buffer reproduce the offline recompute bit
      for bit rather than "within tolerance"; the EMA counter-example is a test. The label is
      purged on a walk-forward defined on *event time*, not row position, because two coins are
      pooled. All three gates were run against the exported artifact: **model parity PASS** at
      `TREE_EPS=0` with zero decision flips, **feature parity PASS** at `max_abs_diff` exactly 0
      over 4,975 rows × 9 columns per coin, **drift OK** (worst PSI 0.1166). And the numbers:
      pooled out-of-sample AUC **0.5224** (folds 0.5125/0.5390/0.5498/0.4993), coverage 0.809,
      gross edge **+1.16 bps** per decision against a 3.0 bps maker round trip — while a constant
      short over the model's own 6,435 decision rows earns **+2.99 bps**, so the selection is
      worth **−1.83 bps**. What edge exists is a directional bias in a falling window. Three
      green gates and a model that should not trade is exactly the distinction this ladder was
      built to preserve.
- [x] **The feature half of Boundary A now has something to gate against, and it is held to
      bits** ([ADR-0035](adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)). The first half
      closed on 2026-07-26: the Rust model-parity gate stopped being seeded gaussians gating
      arithmetic — `zoo_xgboost` and `zoo_logistic` are the zoo's own artifacts over a real m1
      feature matrix, committed, taken **off the unassisted export path** rather than
      hand-stamped, so the gate that certifies the boundary finally ran on the path it certifies.
      The harder half closed the same day. `crates/axon-features` is a Rust implementation of all
      **seventeen** registered transforms, the versioned `FeatureSpec` re-identified in Rust, a
      bounded streaming runtime, and the cross-language gate that makes the whole thing
      admissible. A *feature bundle* freezes a spec, the input arrays it reads and the matrix
      **Python** computed from them; `axon_features::parity::FeatureBundle` opens it with no
      Python, no numpy, no network and no clock, recomputes, and compares. Measured over four
      committed bundles: **33 423 cells, 17 of 17 registered transforms, `max_abs_diff = 0e0`,
      0 bit mismatches, 0 NaN disagreements**. Two of the five are worth naming for their
      provenance rather than their size: **58 bars that crossed a live Hyperliquid testnet
      socket** during the ADR-0029 shadow run, carrying the two live pathologies no offline
      history has (a NaN `clv` on a minute that traded at one price, and gaps where the venue
      sent no frame at all); and **675 market-data slices off a recorded order book and tape**,
      venue frames → Rust core → publisher → the `MdSlice` ring, which is what gates the nine
      columns of `PERP_CORE_V1` — mid, spread, book imbalance, trade-flow imbalance and the EMA
      crossover — against real book state rather than against columns derived from bars.
      **What it cost is the part worth reading, because the obvious implementation is wrong.**
      NumPy does not sum a window left to right — it accumulates pairwise, eight accumulators in
      a fixed tree, splitting recursively above 128 — so a Rust crate writing `iter().sum()`
      disagrees with Python by **2 ULP at window 20, 4 at 32 and 8 at 128** on every rolling
      column. Both ways of not knowing that are quiet: widen the criterion to absorb it and the
      gate goes blind to the windowing off-by-one it was built for, or read it as a defect and
      spend a day inside transforms that are correct. So the order is **reproduced rather than
      tolerated**, transcribed from NumPy's own `DOUBLE_pairwise_sum`, and pinned against NumPy's
      output rather than against the transcription's description of itself — with a test that a
      naive summation *would have failed*, since the positive result means nothing without it.
      Verified by mutation: swapping in the naive loop reddens exactly the two tests that exist
      to catch it and nothing else.
      One operation is a **measurement rather than a guarantee** and is named on the wire.
      IEEE-754 requires `+ - * /` and `sqrt` to be correctly rounded and says nothing about
      `log`; NumPy and this host's libm agree at **0 ULP** over the 32 ratios the fixture pins
      and over a re-runnable 200 000-sample sweep across 26 decades,
      because both compute it well, not because either must. Every bundle's manifest therefore
      carries `libm_columns`, and **both languages derive that list from their own
      transform-level table and the two are compared** — a field only one side could produce
      would be a field the other side believes, and the signpost's whole job is to be trusted on
      a platform nobody here has measured. It can never excuse a mismatch.
      **Three libcs have now been measured**, using ADR-0017's Modal offload for a job that runs
      in seconds and buys a fact this box cannot produce at any price
      (`scripts/modal_libm_probe.py`): the fixture's own numbers, recomputed under **glibc 2.36
      on gVisor** and under **musl on Alpine**, against this box's **glibc 2.39 on an AMD Ryzen 9
      5950X**. On both containers all seven windows' `sum`/`mean`/`std` came back bit-identical at
      both `ddof`, all 32 recorded logarithms came back bit-identical, and NumPy's `log` measured
      against *that platform's own* `math.log` — the libm `f64::ln` calls — at **0 ULP over
      200 000 fresh samples across 26 decades**. So NumPy's summation order is not a property of
      one build, and glibc and musl produce the same logarithm as the frozen fixture on every
      value tested. The gate also reproduces every cell under `--release` with LTO, ruling out a
      bit-exactness that holds only at `opt-level = 0`.
      **Read what that chain does not say.** No `axon-features` binary has ever been built or run
      against a non-glibc libm — `rustup target list --installed` here has one entry. The
      *arithmetic* was ported; the *artifact* was not, and macOS and non-x86 are untouched.
      Two things the crate says about this repository rather than about a hypothetical. The
      streaming runtime **refuses** a spec whose lookback is unbounded, which turns the house
      rule *"finite lookback, no EMA, no expanding statistic"* into a construction error — and
      the first thing it refused was **`PERP_CORE_V1`, the repo's own reference perp spec**,
      because exactly one of its nine columns is an `ema_crossover`. What an EMA costs a bounded
      buffer is measured on a bare `ema` column rather than argued: a span-8 level over a 21-deep
      window lands **1.12 price units — −0.187 bps, 154 450 955 866 ULP** from the same level
      taken over the full history, systematically, and the gap is widest right after a restart. And the buffer depth is *derived*: `BAR_M1_V1` comes out
      at **21**, which is `BAR_M1_WARMUP_BARS`, with the two asserted against each other rather
      than one restating the other. Over 400 bars the streamed and batch matrices agree on
      **2 400 of 2 400 cells**, all 53 NaN cells NaN on both sides.
      **Read what this does not claim.** Nothing in the live core calls the crate: features are
      still computed in Python at Boundary B, exactly as before, and promoting a strategy to
      Boundary A is a separate decision and a separate change. The gate is only as broad as its
      corpus — a transform whose Rust body is wrong in a way no committed bundle exercises is not
      caught, which is why `all_transforms` exists and why its coverage is asserted rather than
      described. `push` allocates nothing itself, but the transforms beneath it take **exactly 5
      heap allocations and 824 bytes per push**, measured, all inside `rolling_zscore` and
      `realized_volatility` — so what is stated is the number, not zero-allocation.
- [x] **Shadow trading (rung 3) is built, tested and driven over 208 days of real venue prints.**
      *(This item's headline once ended "— and no bar in it has crossed a live socket". True when
      written, false now: see this phase's Exit, where 63 minutes of it ran off a live testnet
      socket, and the `bar_m1_testnet_live` feature bundle, which is 58 of those bars frozen as a
      cross-language fixture.)*
      ([ADR-0029](adr/0029-shadow-trading-and-the-continuous-diff.md)).
      `axon.strategies.shadow` drives `perp_bar` one closed bar at a time through the real
      `axon.live.StrategyRunner`, puts its would-be targets on a real signal ring, reads them
      back *off* it, and diffs each window against a recompute over **exactly the bars that run
      was shown** — 4,975 rows per coin, 79 windows, `max_abs_diff` **0.000e+00** at **coverage
      4,975/4,975**, with the registered artifact in the loop. The whole consumer half of
      [ADR-0028](adr/0028-market-data-bars-and-the-ticker-tail.md) is exercised end to end (real
      `MdBar` records, the ring's `record_kind` header, the continuity flags,
      `MdBarRingConsumer`); the **Rust publisher is stood in for**, and a stand-in is not
      evidence about the thing it stands in for. Nothing here is attested against the venue and
      the report prints that on every run, because an `MdBar` carries no source marker and a
      process that inferred *live* from the shape of its input would print the word on a run over
      a file. Two findings the walk-forward could not produce. The strategy emits **1,446 target
      changes over 4,999 BTC bars** — a mind changed every 3.5 hours against a four-hour label
      horizon, 2,413 position sides, **36% of notional in maker fees over 208 days and 109% at
      taker** — which is why `urgency = 0` is load-bearing rather than a preference; it is a
      count of what was emitted, not a P&L, and it is deliberately not divided by the gross edge,
      because that quotient needs the holding-period model ADR-0022 refuses to supply. And an
      hourly strategy has **no opinion for its first 25 hours**, because the spec's longest
      window is 24 bars and the publisher emits a bar only once the next one starts. That is
      arithmetic, and it is why this rung cannot be observed inside a short session.
- [x] **The parity gates now run as a loop, and an intersection can no longer hide its
      denominator** ([ADR-0030](adr/0030-live-parity-monitor-and-the-coverage-denominator.md)).
      The load-bearing half is the fix underneath the monitor: `align_by_event_time` matched on
      event time and *dropped* what did not match, so a serving path emitting half its rows was
      compared on that half, agreed with it to the last bit, and reported `PASS …
      max_abs_diff=0.000e+00` — the headline number at its most reassuring exactly when the feed
      is at its most broken. It now returns an `Alignment` (still the same index pair, so no call
      site changed) carrying a `Coverage`, and `aligned_feature_parity` folds that into the
      verdict. The classification is what makes it usable rather than noisy: offline rows
      *before* the online side's first event are legitimately out of scope, while a gap *inside*
      its span or a row *after* it stopped are both failures — an in-span rule alone would excuse
      a feed that produced perfect rows and then died. A caller that *knows* the owed set says so
      with `scope="declared"`, because a serving path blind through its opening rows has exactly
      the same first stamp as one that started on time, and inference alone cannot tell them
      apart. `perp_bar`'s green is unchanged and now **earned**: `max_abs_diff` exactly 0 over
      4,975 rows × 9 columns per coin, with `coverage=4975/4975` asserted beside it. An empty
      intersection also names itself now: a constant nearest-stamp offset is reported as two
      stamping conventions, with the 1 ms candle case called out by name, instead of "an empty
      feature matrix proves nothing".
- [x] **A live parity monitor exists, is testable offline, and has never seen a live session.**
      `axon.parity.monitor` runs the existing gates over windows — it reimplements no comparison
      — and adds the two things CI does not need: state across windows, and a verdict for a
      window that compared *nothing*. `OK < WARN < SILENT < ALARM`, where an empty window is
      `SILENT` and never `OK`, and a run in which no window compared a row fails outright: "OK,
      0 rows compared" is the same invisible denominator one level up. Silence is the one thing
      measured on a wall clock, because the absence of an event has no event time. An alarm
      **logs and does nothing else**, deliberately: the dead-man's switch and the intent source
      already own stopping trading (ADR-0013), and a second uncoordinated authority is the shape
      that ADR was written to avoid — and this detector's false-positive rate is unmeasured.
      Which is not hypothetical: over `perp_bar`'s own history in 256-row windows, with feature
      parity green on **every** window, PSI passes the conventional 0.25 band on 18 of 20 (BTC,
      peak 5.97) and 14 of 20 (ETH, peak 7.93). So drift is capped at `WARN` by default —
      ADR-0016 §5 flagged those bands as convention rather than derived, and this is the bill.
      It had been run against a frozen fixture, a cached mainnet history and synthetic windows,
      and against no live session at all — its 60 s silence deadline a stated starting value
      rather than a measurement. **That is closed; see the next item.**
- [x] **The parity monitor has watched a venue.** 2026-07-27, 55 minutes of a read-only
      Hyperliquid testnet session driven by one command (`scripts/sessions/parity-live.sh --live`,
      see [ADR-0030](adr/0030-live-parity-monitor-and-the-coverage-denominator.md)): the session's own
      `MdBar` ring, the real `StrategyRunner`, `perp_bar/v1#a21328ed1532ecd4`, a real signal ring,
      the beacon wired. **Three windows, all `OK`, `max_abs_diff = 0.000e+00` over 23/23 owed
      rows**, every one of nine columns at exactly zero, on 104 bars across two instruments with
      **0 ring drops** and bar coverage **53/54 (98.1 %)**.
      Read the run for what it *measured* rather than for the zero, because the zero is the least
      interesting thing in it.
      **The silence deadline stopped being a stated value.** Every run now records the wall-clock
      gaps between the windows that compared something, between consecutive observations, and
      between the beacon readings in which the ring advanced. This one: n = 2, min **362.7 s**,
      max **1 020.8 s** — and the widest gap **reached the 150 s deadline**, so every window
      inside it was graded `SILENT` on a feed that was demonstrably healthy. The report refuses to
      propose a replacement, in as many words: a deadline fitted to the one tape anybody has run
      is a detector calibrated to what that tape happened to do, which is the ratchet ADR-0021
      declined when it would not fit a tolerance to a measurement.
      **The beacon turned silence into a cause, live.** `publisher=publishing` on two of the three
      windows and `unknown` on the first — and the run's own arrivals line reported **51 bars for
      `symbol_id` 4 that no trader was built for**, so the thing this monitor most easily gets
      wrong ("it compared nothing, therefore the feed is quiet") is answered on the page. The
      session's final beat carries the `STOPPED` flag off a live socket for the first time:
      `beats 5 124 056` over 3 301 s (**1 552 a second**), `published 3 004`, `coalesced 797`,
      `dropped 0`, `bars_published 104`.
      **And it produced a finding the offline harness could not.** 53 bars arrived
      **1 067 / 9 406 / 62 074 ms** (min/median/max) after their own close; against a core event
      clock 1 564 ms behind wall time, **41 of the 53 would be refused as `Expired`** under the
      shipped 2 000 ms admission ceiling. A shadow run never reaches the signal reader, so nothing
      was refused here — but that is the arithmetic a live *trading* session would run into, and
      it is the same defect the handoff's Q3/R3 names: the fix is to stamp at the decision, not to
      widen the ceiling.
      What the run cannot say is written down first, in the runbook's §0: it compares the serving
      path against a recompute **over the bars that arrived**, so both sides would agree perfectly
      about a wrong bar; drift was not measured (no reference sample); and 3 windows over 55
      minutes is a transcript, not a soak.
- [x] **The monitor can now tell a quiet publisher from a dead one — in `axon-ipc`, not yet in a
      session.** `MdWritePolicy::OnChange` makes an empty ring ambiguous by design; the fix is the
      64-byte mmap'd sidecar specified in
      [ADR-0030](adr/0030-live-parity-monitor-and-the-coverage-denominator.md) and built as
      [ADR-0034](adr/0034-market-data-beacon-and-the-third-clock.md) — in `axon-ipc` because
      mapping a file is the `unsafe` that crate already holds. `axon.parity.beacon` reads it and
      `ParityMonitor(..., beacon=probe)` resolves silence into a cause. Two states nobody had named
      fell out of building it: **`STARVED`** (the process is beating and its event clock has not
      moved — the silent-healthy-socket case, which no backoff and no duration check can see) and
      **`PUBLISHING`** (the ring is moving and the monitor compared nothing anyway, so the fault is
      downstream and must not be excused). `last_beat_ns` is a **third named wall-clock exception**,
      on the same argument as the other two: the condition being detected is the absence of events,
      and the absence of an event has no event time. Two hazards recorded in the ADR: the five
      counters are `u32` and **wrap** (read with `wrapping_sub`, never print one as a total), and
      `create` must never `O_TRUNC`, because a monitor with the page mapped takes a `SIGBUS` for
      the instant the file is zero-length.
- [x] **The beacon is wired, and it has beaten against a live venue.** The gap this line used to
      carry — *"the wiring into the pass loop is not in the tree, so the beacon exists and nothing
      beats it yet"* — is closed. `core::run` beats once per **pass**, before the intent poll, so
      the counters describe the state the drain left behind; and once more, flagged `STOPPED`, on
      the way out, because "the session ended" and "the session died" want different words from
      whoever is woken at 03:00. `MdPublisher` creates it beside the two rings from the same
      `[md_ring]` switch and the same derived name, so there is no config field and no way to
      half-turn it; the pre-start banner now names **three** paths where it named two, and
      `RuntimeConfig::validate` refuses a `capture.path` aimed at the derived beacon — the worst
      collision in that list, since a capture truncates and a monitor with the page mapped takes
      the `SIGBUS`.
      **Observed on a live read-only Hyperliquid testnet session on 2026-07-26**, which is the
      first time anything has beaten off a socket rather than off a canned stream: `PUBLISHING`
      throughout, **~36 900 beats per 25 s** (the pass rate, not the record rate — the design),
      `published` +36 to +68 slices per 25 s, **0 dropped**, `event_advanced` true on every
      reading, and a `last_beat_ns` age of **0–3 ms**. Coalescing read 0 in that session and that
      is a property of its config rather than of the venue — it ran `policy = every_update`, and
      the `on_change` session beside it coalesced steadily. The number worth keeping is the beat
      age: at ~1 470 beats a second the pass period is **0.68 ms**, so the monitor's 60 s silence
      deadline is ~**88 000×** the beat period and ~**20 000×** the worst age observed. That is
      the first evidence anyone has had about whether that stated starting value is near right. Offline, by contrast, `last_beat_ns` is `0` and the reader
      says `had_wall_clock=False` — the sentinel ADR-0034 §1 predicted an offline session would
      be the thing to exercise, now confirmed from both ends.
      Eight tests hold the wiring, each named after the reading it prevents and each watched to
      redden with the wiring removed — including one that puts the beacon **on the slice ring's
      own file** when the path derivation is broken. (The `u32` wrap itself is covered on the
      Python side, where the reader lives: `test_beacon.py` drives `published` from `U32 − 3` to
      `2` and asserts a delta of **5**, which a saturating read would report as **0** — the wrong
      answer to the only question the sidecar is asked.) One honest exception is written into the
      test itself: the restart test reddens for the *unlink* hazard and **cannot** redden for the
      *truncation* one, because a sequential test is never inside the zero-length window.
- **Exit:** a real ML strategy passes the full validation ladder up to shadow trading.
  **Rung 3 is climbed.** `perp_bar` was shadow-traded for **63 minutes off a live Hyperliquid
  testnet socket** on 2026-07-26 — two instruments through one reader, the Rust publisher, the
  `MdBar` ring, the real `StrategyRunner` and a real signal ring — at `max_abs_diff = 0.000e+00`
  over **30/30 and 36/36 owed rows**. Read the denominators rather than the ratio, because that is
  the whole lesson: the bars it was shown were **58 of the 63 minutes its own cadence promised**,
  so the run printed `PASS`, a perfect ratio and a clean zero **on a feed missing 8 % of its bars**.
  The venue reconciliation came back the opposite way round from
  [ADR-0028](adr/0028-market-data-bars-and-the-ticker-tail.md)'s prediction: all five OHLCV fields
  matched `candleSnapshot` on **145 of 145 delivered bars, 0 short on volume**, while **8 minutes
  the venue itself lists never reached the ring at all** — every one a minute with `n = 0` trades,
  because Hyperliquid's `candle` subscription sends *no frame whatsoever* for a minute nothing
  traded, so the publisher never gets a later `open_time` to close the bar with. **Four checks
  fired on that perfectly healthy feed and every one printed FAIL**, all four unreachable from a
  fixture — which is the argument for running a harness rather than testing one. What remains
  untouched is the planner half: a `ShadowSignal` is still a target nobody priced.
  **And the phase's own last box is closed: both halves of Boundary A are now gated.** ADR-0021
  proved the model crosses; [ADR-0035](adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)
  proves the *features* do — `33 423 cells` over five committed bundles at `max_abs_diff = 0e0`,
  bit-exact, 17 of 17 registered transforms, on real venue rows including 58 bars that crossed a
  live socket. The phase asked for a harness that can fail, and every gate in it now can:
  the model gate reddens on one ULP of a tree margin, the feature gate reddens on one ULP of one
  cell, the streaming runtime refuses a spec it cannot serve, and the coverage denominator is
  asserted rather than reported. What the phase deliberately does **not** claim: nothing in the
  live core computes a feature in Rust. This is the ladder proven, not a boundary moved — and
  moving it is a per-strategy decision with its own evidence to produce.

## Phase 6 — First live strategy (mid-frequency) ✅ (closed on testnet; mainnet is a decision, not a gap)

**What this phase was for, and what it was not.** The goal was to prove the machinery works
*across model families*, not to make money. A strategy that loses money while every gate stays
green is a **successful** outcome — it means the plumbing is sound and the model is bad, which is
exactly the distinction the fidelity ladder exists to preserve. Every family below is reported
against that standard, and no AUC appears anywhere in it.

- [x] **Shadow-trade a real strategy on live data.** Done — see Phase 5's exit above. 63 minutes
      of m1 bars off a live testnet socket at `max_abs_diff = 0.000e+00` over 30/30 and 36/36
      owed rows, on a feed that was missing 8 % of its bars.
- [x] **A cancel has been sent through the intent path at a venue.** Every live order this project
      had ever placed was an IOC that filled outright, so the whole cancel/replace half of
      [ADR-0014](adr/0014-signal-to-order-planning.md) §6 was untouched by evidence. Five claims
      are now observed on testnet: a resting post-only placed *by the planner*; a moved target
      producing a cancel and a replacement that both landed; an adopted order's cancel re-addressed
      to the venue's own oid; and — the one a test that cancels unconditionally would pass by
      accident — an unchanged target leaving the order resting, **with the same oid at the same
      price afterwards**, the reason pinned to `AlreadyWorking` rather than inferred from the
      outcome. `orderUpdates` **does** emit `"status":"canceled"`, ~1.5 s behind a REST ack that
      returns `statuses:["success"]` synchronously; a cancel of an oid that is already gone is
      HTTP 200 carrying `"Order was never placed, already canceled, or filled. asset=3"` — **the
      same sentence the venue uses for a client id it has never seen**, so the two are
      indistinguishable from the reply alone. Cost: 26 signed actions, no fills, zero
      dead-man's-switch triggers.
- [x] **The model zoo — five families, and the matrix everyone assumed was wrong.**
      One shared spec, `BAR_M1_V1`: six finite-lookback columns over a closed m1 candle, longest
      window **21 bars = 21 minutes** on m1 (`perp_bar`'s 24 bars on *hourly* is 25 hours, which is
      the difference between a session you can observe today and one you cannot). Fitted over
      15 142 real m1 bars with zero gaps ([ADR-0032](adr/0032-the-model-zoo-and-what-actually-crosses-into-rust.md)).

      | Family | Model parity | Feature parity | Bundle | **Crosses into Rust** | Held to |
      |---|---|---|---|---|---|
      | XGBoost | PASS 0.0, flips 0 | PASS 0.0, 15 029/15 029 | ✅ | **✅ `max_abs_diff=0e0`, flips 0** | bit-exact |
      | sklearn GradientBoosting | PASS 1.323e-7 | PASS 0.0, 15 029/15 029 | ✅ | **❌ refused twice** | — |
      | LogisticRegression | PASS 5.960e-8 | PASS 0.0, 15 029/15 029 | ✅ | **✅ 8.940697e-8, flips 0** | 2 ULP |
      | LightGBM (binary) | — | — | ✅ | **✅ 1.1920929e-7 (one ULP), flips 0** | 2 ULP |
      | LightGBM (regressor) | — | — | ✅ | **❌ refused at load** | — |
      | No-model `baseline_z` | *category error* | PASS 0.0, 780/780 | n/a | n/a — nothing crosses but the `Signal` | — |

      **A graph bundle is held to two ULP, not ADR-0003 §3's 1e-5 family ceiling.** The three
      graphs this gate runs on disagree with Python by at most *one* ULP — `lgbm_binary`
      1.1920929e-7 (one ULP at 1.0, exactly), `zoo_logistic` 8.940697e-8, `mlp_regressor` **0e0** —
      so declaring the ceiling left ~42× of slack, and slack is where a regression passes green.
      The ceiling is unchanged as the bar a *reader* enforces; two ULP is what a *writer* here
      declares, asserted against the committed bundles in both languages
      ([ADR-0021](adr/0021-rust-model-parity-gate.md), amended). It is deliberately anchored to
      float32's resolution rather than fitted to the measurement: a tolerance derived from what
      today's runtime produced is one that ratchets, recording a regression as the new bar the
      next time it is regenerated.

      **Feature parity is exactly 0.0 at complete coverage for every family on every coin.** Drift
      alarms identically for all three fitted families, which is the tell that it is a property of
      the *sample* and not of any model.

      **And one of the six rows now has a seventh column filled in: XGBoost has driven orders at a
      venue.** Refitted on testnet's own m1 tape (`zoo_xgboost@1`, 15 595 bars over BTC/ETH/SOL,
      every gate green and the crossing still `0e0`), served live on 2026-07-27. That is the
      "for at least one of them, did it drive an order and reconcile" column the Phase-6 brief
      asked for, and until this run it was empty for every family.

      **The artifact kind does not decide what crosses; the ONNX operator does.** `tract` 0.23.4
      registers exactly five `ai.onnx.ml` operators and `TreeEnsembleRegressor` is not among them,
      so a boosted tree crosses **only** if its converter emits a classifier — and only LightGBM's
      does. Separately, every sklearn classifier emits two graph outputs, refused by both the
      bundle writer and the loader. The route around that refusal is a trap only the gate catches:
      deleting `base_values` builds and agrees with onnxruntime to seven decimals while silently
      being *the model without its intercept*; padding it makes tract and onnxruntime disagree by
      **0.18**, five orders past tolerance.
- [x] **LightGBM crosses by conversion, and no backend was written**
      ([ADR-0033](adr/0033-lightgbm-crosses-by-conversion-not-by-backend.md)). `onnxmltools`
      converts the booster, the artifact becomes an ordinary `onnx` kind, `tract` serves it. The
      gap [ADR-0019](adr/0019-native-rust-inference-backends.md) left was a **routing question**.
      `SERVABLE_KINDS` did not move, deliberately: widening it would only relocate the failure from
      bundle-write time to Rust *load* time, after the artifact has a version a signal can name.
- [x] **A no-model baseline drives the whole path.** `baseline_z` — a 20-bar rolling z-score with a
      realized-volatility floor, **no artifact, no registry entry, no export** — reaches a real
      signal ring through the real `StrategyRunner`. So a session where the ML strategies are
      silent and this one is not tells you the fault is the model, not the bridge. Exactly one
      thing in the pipeline demands a model and it demands a *string*: `StrategyConfig::model_ref`
      has no `Option` and no `#[serde(default)]`, so a config without it is refused at parse time —
      and **nothing in the runtime ever reads it**. A no-model session starts by naming a registry
      entry that does not exist.
- [x] **Go live small.** `baseline_z` on testnet m1, minimum notional, DMS armed, `--capture` on:
      **1 h 35 m, 89 bars, five target changes, six orders at the venue, four maker fills, two
      round trips, `sig 6/0` — every signal accepted, none expired — and the account flat at the
      start and at the end**, read back three times. The long→short flip went out as **one order of
      0.0006 BTC**, not two, which is ADR-0014's "the order is the delta" observed at a venue for
      the first time. Cost: `accountValue` 998.89646 → 998.913244, `nRequestsUsed` +281, fees
      0.011658 all maker, **zero dead-man's-switch triggers**. The +0.0168 USDC is two round trips
      and **is not a result**; nothing here is evidence about the strategy.
- [x] **[ADR-0031](adr/0031-order-lifetime-and-the-sweeper-on-the-pass.md) is written, and the
      sweeper cancelled at a venue** — twice, at **62 027 ms** and **60 035 ms** against a 60 s
      `max_order_age_ms`, both venue-confirmed. The planner bounds a resting order's age but runs
      *on a signal*, so it can never pull a quote from a strategy that has gone silent; the sweeper
      runs on the pass loop, so it advances when nothing arrives. Its cancel is never risk-gated,
      because a cancel reduces exposure and a gate that can refuse one pins an account into the
      position it is trying to leave. **The first live sweep immediately exposed a composition
      nobody had reasoned about**, recorded as an amendment: both swept orders were *exits*, a
      target position is idempotent, so nothing re-quoted them and a short sat open with no working
      order for ~12 minutes. The sweeper says "cancel what no signal speaks for"; the strategy says
      "say nothing when nothing changed". Each is right; the composition is not. The comfortable
      repair, sparing orders that look like exits, is the wrong one: `reduce_only` is a property of
      the order, not of the operator's intent.
      **Now fixed, and by the third option** ([ADR-0036](adr/0036-watching-a-live-session-money-latency-and-the-unquoted-target.md)):
      the pass retains the newest target per symbol and **re-quotes** one the strategy still holds
      when nothing is working toward it — at most `intent.max_requotes` times (default 3), never
      while halted, never on top of a working order, and never at a larger target than the strategy
      asked for. The budget is what keeps the repair from undoing the sweeper, whose whole subject
      is a producer that has *stopped*: past it the session stops placing and starts saying
      `UNQUOTED TARGET`, which is the one thing the twelve-minute hole could not do. The record is
      re-stamped with the pass's event time and keeps its `seq` — re-planning it byte for byte
      would mint the cancelled order's own `cloid` and the venue would de-duplicate the repair into
      nothing while every counter claimed success.
- [x] **Monitor P&L, parity, latency budgets** in a live session
      ([ADR-0036](adr/0036-watching-a-live-session-money-latency-and-the-unquoted-target.md)).
      All three now exist and a session that **trades** has been watched by them.
      *P&L*: `crates/axon-runtime/src/pnl.rs` reports our own accounting and the venue's
      `accountValue` delta side by side and never reconciles them — the difference is printed as
      `drift`, because funding is not a fill and no fill-derived accounting can see it. It refuses
      to price an unrealized P&L off a stale mark, naming the symbol as `POSITION UNPRICED`
      instead: contributing zero would make a position going wrong read as a position going
      nowhere. *Latency*: three declared budgets rather than by-product observations —
      `sig` (decision → the pass that planned it, on the event clock), `ack` (the venue round
      trip) and `e2e` (decision → order at the venue), each with a breach count against a ceiling
      somebody argued for, and a warning that fires on a **rate** rather than on the first late
      order. *Parity*: on the trading session, measured after it from its own capture
      (`scripts/sessions/session_parity.py`), because **an SPSC ring has one consumer and the
      strategy is it** — a live monitor attached beside the strategy would take bars away from the
      thing placing orders. The live monitor's own venue evidence stands from Phase 5.
      **Observed, 2026-07-27, 59 minutes on Hyperliquid testnet with an ML model trading:**
      *P&L* — `pnl -0.0281 (r -0.0013 fee 0.0268 u +0.0000) 8m/1t (+22 pre)` beside
      `eq 998.8849 -0.0282 drift +0.0001`. Our realized figure and the venue's summed `closedPnl`
      agree **to the last digit printed**, from different accounting on different machines, and the
      largest drift at any point in the run was **0.0137 USDC**. The `(+22 pre)` marker is the
      venue's own fill replay, named and excluded.
      *Latency* — `sig ·861 ms 0/9 | ack ·1024 ms 1/12 | e2e ·2717 ms 0/12` against ceilings of
      20 000 / 1 000 / 25 000 ms. **One breach**, 24 ms over a bound argued from the venue's
      published p99; the warning lit at `1/1` and **cleared itself** at `1/5` when the rate fell
      under the declared 25 %, which is the whole reason it is a rate.
      *Parity* — `PASS, max_abs_diff = 0.000e+00, coverage 30/30` over the 58 closed bars the
      session captured.
- [x] **An ML family has driven an order at a venue.** The gap the rest of this phase kept naming
      about itself. `zoo_xgboost@1` — refitted on **15 595 testnet m1 bars** across BTC/ETH/SOL,
      model parity `0e0`, feature parity `0.0` at complete coverage, crossing into Rust bit-exact —
      served through `PerpBar` on `BAR_M1_V1`, onto the real signal ring, through the real planner,
      to Hyperliquid testnet.
      **Fifty-nine minutes, 57 bars, eight target changes, `sig 9/0` — every signal admitted, none
      expired — 12 orders, 8 cancels, 9 fills (8 maker, 1 the deliberate taker that flattened), and
      the account flat and order-free at the start and at the end, read twice.** Two claims are
      firsts beyond the headline: **the strategy drove its own cancels** (the no-model run's
      superseded quotes were resolved by the venue or the sweeper, never by the planner replacing a
      working order), and **the sweeper's re-quote fired** — six sweeps, four re-quotes, the
      twelve-minute unquoted hole closed and observed rather than argued. Cost: −0.028263 USDC, of
      which **96 % is fees**; the strategy's own trading was −0.0013 over nine fills. That is what
      +1.44 bps of selection against 1.61 bps of maker drag looks like when it trades, and it is
      the offline verdict predicting the live one — **it is not a result**.
      ([ADR-0036](adr/0036-watching-a-live-session-money-latency-and-the-unquoted-target.md))
- [x] **The same model on a second coin, and an unplanned venue outage that tested the safety
      path.** `zoo_xgboost@1` on **ETH-PERP** warmed on ETH's own bars, changed its mind twice and
      filled maker at 1969.5 — the machinery is not BTC-shaped. Then Hyperliquid testnet went down
      for a network upgrade (502 on `/exchange`, `/info` and the socket) and three mechanisms
      proved themselves without anyone staging them: **the money view refused to price** the open
      position (`pnl -` … `POSITION UNPRICED ETH`) rather than reporting a comforting zero; **the
      dead-man's switch ran out of protection for the first time in this project's history**,
      halting new orders at the second failed re-arm and shutting the session down at the third,
      with a closing sweep that *failed* and said so; and **the re-quote's bound earned its keep** —
      the venue was accepting post-only orders exclusively, so the flatten's IOC was refused, three
      re-quotes were refused with it, and the budget then stopped and raised `UNQUOTED TARGET 1`
      for eight consecutive status lines. That terminal branch had been proven offline only an hour
      before. One finding against the runtime came out of it: the unprotected path logs "the
      venue-side switch **has fired**" when it has only observed its own deadline passing — the
      order placed before the outage was still resting when the venue returned. The account was
      restored **flat and order-free** by an operator, verified twice.
- [x] **Tier 0: the five things that stood between this and real money, and the two Phase 6 could
      not do** ([ADR-0037](adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md), 2026-07-27).
      The outage above did not only prove three mechanisms; it proved the **exit path did not work**,
      and this is that closed:
    - **A loss-based kill switch that never refuses a reducing order.** Every risk limit was
      size-only, so a strategy that was quietly wrong stayed inside all of them indefinitely.
      `axon_execution::LossLimiter` puts a session past its declared bound into **de-risk-only**
      rather than halting it, because halting strands the exposure that caused the loss in the
      market causing it. Two independent bounds — ours, and the venue's equity against a baseline
      that **survives a restart**, because a crash-restart loop is how a losing session restarts.
    - **`axon --flatten`**: adopt the venue's own position and drive it to zero. Every order is a
      `FLAG_CLOSE` reduce-only sized from a **fresh venue read**, so the operator error that turned
      a −0.01 short into a +0.01 long is unrepresentable; and the urgency is a **ladder**
      (IOC → crossing GTC → post-only), because the venue that refuses an IOC is exactly the venue
      an operator cannot wait for.
    - **The dust floor stops stranding closes.** `min_order_qty` is a churn bound and a close is
      not churn; both it and the venue minimum now exempt an order that reaches exactly flat. The
      residue that could not be closed twice in 59 minutes now can.
    - **The DMS says what it observed** — "the deadline we armed has passed", not "the venue-side
      switch has fired" — and the wording is asserted by a test rather than reviewed.
    - **Parity on a session that is trading.** The diff is extracted from the shadow harness so one
      bar-ring reader can dispatch to the strategy *and* to it (`--parity-diff`). Both reasons it
      was impossible turned out to be about *sessions*: one SPSC consumer, and a second session's
      account-wide sweep.
    - **A ceiling on bar-close → decision.** Schema **v3** puts `ts_cause` on the wire, spending the
      last of `reserved`. The largest number in the system — 951 / 12 051 / **111 475** ms over 57
      live bars — existed only in the producer's transcript, so it could be quoted and never
      budgeted. There is now a fourth latency stage, `bar`, additive with `e2e`.
    - **And one defect ADR-0036 recorded as *fixed* was not.** `PnlSnapshot::unreadable()` had no
      production caller: `core.rs` built the money view from an empty tracker on a poisoned lock,
      so a session holding a position it could not see printed `pnl +0.0000`. The comment above it
      said what the code must not do.
      **None of the above has fired at a venue.** Every branch is tested offline; the terminal
      states have been observed nowhere.
- [ ] **A trading soak.** The longest session that has ever *traded* is **59 minutes**, and the
      1 h 44 m soak was read-only market data. The harness exists — `scripts/soak/run-trading-soak.sh`,
      `soak-testnet-trading.toml`, the same relay and induced-outage plan, plus `--parity-diff` and
      the loss bounds — and the run does not. It needs hours of venue time rather than code, and it
      is the largest untested thing in the system: everything in Tier 0 above is motivated by an
      outage that happened inside an hour, and none of it has been asked to survive eight.
- [ ] **The loss bounds argued from more than one session.** `max_session_loss = 1.00` is a
      declaration backed by one 59-minute run that lost 0.028 — the same weakness ADR-0036 recorded
      for the latency ceilings, and it matters more for a gate than for a warning.
      `axon.strategies.loss_evidence` is the fan-out that replaces the guess: 240 non-overlapping
      hour-long windows, planned through hwsched at **240 tasks / 60 containers / est high $0.15,
      approved**, and **not run** — spending is the operator's decision.

- **Exit:** one mid-frequency ML strategy running live, with parity monitoring and the dead-man's
  switch armed. **Met on testnet. The mainnet clause is unmet by decision, not by capability.**
  Everything the criterion asks for — an ML strategy, live, with parity monitoring and an armed
  dead-man's switch — has now been run at a venue; the venue is Hyperliquid **testnet** because the
  key in `.env` is public (it was pasted into a chat) and because the operator's standing
  instruction for this work is testnet only. Mainnet needs a fresh key via `./run.sh agent`, the
  `approveAgent` ceremony against mainnet, and an explicit decision as its own conversation. That
  is a **decision, not a gap**, and the distinction matters: nothing technical is known to be
  missing, and the mainnet readiness review (R11 in the handoff) is the first thing anyone opening
  that conversation should read. What is genuinely still open is written into Phase 7 rather than hidden here.

## Phase 7 — Generalize the layer 🚧 (the second adapter exists; it has never traded)

- [x] **Second venue adapter — Binance USD-M futures** ([ADR-0023](adr/0023-second-venue-adapter-binance.md)):
      `crates/axon-provider-binance`, one new crate and one line in the workspace `members`
      list. **No other crate was edited**, which is the whole of what Phase 7 set out to test.
      It covers symbol/instrument resolution from `exchangeInfo` filters, the market-data
      decoders (partial-depth book, `aggTrade`, `bookTicker`, `markPriceUpdate`, `kline`),
      the `MarketData` port with connect/reconnect, and order/cancel query-string encoding.
      **Read this state carefully: the adapter is offline-verified and has never traded.**
      93 network-free tests pin the decoders against frames captured byte for byte off the
      venue's own socket, and two `#[ignore]`d **read-only** tests have been run against
      Binance USD-M *testnet* — `exchangeInfo` decodes to 570 perpetuals with real grids, and
      the combined stream delivers book/trade/BBO/mark events to the bus. Nothing else has
      been observed. There is **no HMAC, no order client, no user-data stream and no
      `AccountState`** in the crate, deliberately: there are no Binance credentials in this
      repository, none were sought, and no code path here can place an order. Production is
      geo-blocked from the dev host (`HTTP 451`), so mainnet is untouched.
      Four speculative port decisions were vindicated by a venue they were not written for —
      `PriceGrid`'s composed `{increment, sig_figs}` (a fixed tick and a significant-figure
      rule are one shape, no new enum arm), `SizeGrid::step` (Binance has a lot of **3**,
      which `szDecimals` cannot express), per-instrument `min_notional` (50/20/5/0.001 in one
      response), and `Ticker::ts_venue: Option` — which is `Some` here and `None` on
      Hyperliquid, the first time that `Option` has earned its keep. Seven places the port did
      **not** fit are enumerated in the ADR, each with the exact upstream change it needs;
      all seven are additive fields or one trait signature, and none needs an arm keyed on a
      venue name. The sharpest is live and silent: `SizeGrid` has no `min_qty`, so a 1-SPELL
      order (step 1, minimum 100) is on the lot, clears min-notional, passes
      `InstrumentSpec::check` and the encoder, and comes back `-1013 LOT_SIZE` — exactly the
      failure shape ADR-0025 exists to eliminate.
- [x] **The reconnect-backoff defect the Hyperliquid soak exposed was present here too, and is
      fixed** — re-measured for this venue rather than patched by analogy, which changed two of
      the three arguments. A severed link never reaches the `Ok` arm (a `tungstenite` property,
      now proven against this client over loopback as `ResetWithoutClosingHandshake`), so the
      old reset was unreachable and the wait only ever climbed. Unlike Hyperliquid's, this
      venue's `Ok` path *is* reachable — it closes every connection at 24 h — which made the
      old no-sleep reset an unthrottled reconnect on a venue whose normal lifecycle includes a
      clean close, and which is also why no `Ok` branch is needed: a 24 h connection clears the
      healthy threshold on duration alone. And `heard_from_venue` is load-bearing here for the
      **opposite** reason it is on Hyperliquid: a stream-name typo is accepted silently with no
      error frame and no close (measured), so a duration-only test would call a connection
      subscribed to nothing healthy. A shutdown connect-storm was closed beside it. **Verified
      over loopback and by unit test only — no Binance reconnect, induced outage or soak has
      ever been run.** One consequence nothing in this crate can catch: because a bad stream
      name is accepted silently, a session can hold a permanently healthy, permanently *silent*
      socket, and the backoff never fires because nothing ever disconnects. That needs a
      data-staleness watchdog above the adapter.
- [ ] Binance execution: HMAC-SHA256 signing, an `ExecutionClient`, the `listenKey` user-data
      stream, `AccountState`, and a weight-per-IP rate governor. Blocked on a credential
      decision, not on design — the canonical signed payload is already pinned against the
      venue's own published example.
- [x] **Multi-strategy / multi-symbol orchestration, and portfolio-level risk**
      ([ADR-0038](adr/0038-many-strategies-one-account.md)). The phase's headline, and the
      item the roadmap has carried since Phase 0. Until this, a session had **one signal
      ring, one producer and therefore one strategy on one instrument** — and the sentence
      that explains why is one line: an SPSC ring has one producer. Everything downstream
      that *looked* multi-instrument (the per-symbol in-flight gate of
      [ADR-0020](adr/0020-runtime-intent-source.md) §3, the per-symbol held target of
      [ADR-0036](adr/0036-watching-a-live-session-money-latency-and-the-unquoted-target.md),
      the per-symbol sweep of [ADR-0031](adr/0031-order-lifetime-and-the-sweeper-on-the-pass.md))
      had only ever been exercised at width one, because nothing upstream could put two
      symbols on the wire at once.
      **One ring per producer**, declared as `[[strategy.producer]]`, and the strategy is a
      property of the ring rather than a field on the record. The alternative — a
      `strategy_id` on the `Signal` — was rejected on the fact that decides it: **`seq` is
      per writer.** It is the only proof nothing was lost, so two producers interleaving
      into one stream the reader validates as one sequence means every record of the loser
      is refused as `stale_seq` — a strategy emitting normally, its own counters climbing,
      and nothing reaching the venue. (It would also have to re-cut the 64-byte layout,
      which schema v3 left fully named.) `validate` refuses two producers on one path by
      name, quoting the hazard ADR-0029 §5 **measured** on the bar ring: two readers of an
      SPSC queue do not share it, they steal from it.
      **Claims add.** The venue holds one position per instrument and a target-position
      signal is a claim on part of it, so two strategies on BTC are two claims on one
      position — `axon_strategy::TargetBook` sums them and the planner plans the delta to
      the sum. The naive rule the pass already had, *newest signal per symbol wins*, is
      correct at width one and silently wrong at width two: both producers' counters climb,
      both believe they are positioned, and the account holds whichever spoke last. Netting
      is nonetheless **opt-in** (`overlap = "exclusive"` by default, refused at *startup*),
      because two producers on one instrument is far more often a copy-pasted config than a
      decision, and the accident composes two strategies' risk into a position neither
      author sized.
      **A single contributor passes through byte for byte** — same `seq`, same `ts_event`,
      therefore the same `cloid` — and that is not an optimization: it is how the change is
      *verifiable*. The committed golden replay reproduces its 59 rows, four orders and
      every `cloid` unchanged. It also caught the one defect this work produced: a lone
      `FLAG_CLOSE` was being synthesized rather than passed through, and the order was
      identical in side, size, price and TIF **under a different `cloid`** — the one field
      the venue keys idempotency on, and the third time this project has hit that trap by a
      new route.
      **Three bounds that only exist across symbols** (`axon_risk::portfolio`): gross
      notional, net notional, and how many instruments may carry exposure at once. Ten
      instruments each at 90 % of their own `max_notional` are inside every limit
      `[strategy.risk]` can express, and no number in it would have moved. Enforced in
      `GuardedClient` as a type, cumulatively across a batch, and — for the third time,
      after cancels (ADR-0031) and the loss switch (ADR-0037) — **never refusing an order
      that reduces its own leg**. The pass *also* scales targets to fit, because a bound
      that only refuses presents as orders that keep failing rather than as a limit that is
      working; the gate is the guarantee and the scaler is the convergence.
      **The bounds are argued from a measurement, not declared.**
      `axon.strategies.portfolio_evidence` replays every subset of the (strategy, coin) legs
      over every non-overlapping window of the cached m1 corpus and reports the gross, the
      net, the breadth, and — for a grid of candidate caps — how often each would have bound
      and by how much. Planned and priced through `hwsched` at **100 tasks, CPU, 25
      containers, est high $0.063, budget APPROVE**, and run locally across eight cores
      because on today's corpus the grid *fits* — the enumeration is combinatorial in
      *fitted families*, and at five families over three coins it is 3 850 tasks and the run
      that needs a fleet.
      **Its finding is a negative one and it is in the shipped config**: over 16 000
      measured windows the net/gross ratio's median was **1.000**. These legs never offset —
      BTC, ETH and SOL move together and two of the three run the same rule — so
      `max_net_notional` below `max_gross_notional` is not justifiable on any book this repo
      can populate, and `scripts/sessions/portfolio-testnet-m1.toml` sets them equal and says
      why.
      **None of it has run at a venue.** No two producers have shared an account, no
      portfolio bound has refused a real order, no allocation has scaled one, and no silence
      policy has fired. The minus column in the ADR is as long as the plus column and is
      worth reading first — in particular, a silent producer's exposure is **held** by
      default, `intent.min_order_qty` is still one number for every instrument, and the loss
      switch is still **per session**, so a book where one producer loses and another wins
      trips on the sum with nothing attributing it.
- [ ] Strategy state persistence / warm restart hardening.
- **Exit:** two venues, multiple strategies, one core — the "Layer" property demonstrated.
  **Half reached, and the halves are independent.** *Multiple strategies, one core* is now
  built and gated: one session drives several producers over several instruments, their
  claims net into one position per symbol, and three across-symbol bounds enforce what no
  per-instrument limit could see — with the single-producer path proven unchanged by the
  golden replay reproducing every `cloid`. What that clause still lacks is a venue: **no two
  producers have ever shared an account**. *Two venues* is unchanged — one core speaks to two
  venues' market data, only one of them can be traded, and the second never has.

## Phase 8 — HFT headroom ⏸️ **deferred, by decision (2026-07-26)**

**Not being pursued.** The phase always said "only if a strategy needs it", and no strategy needs
it: `perp_bar` is a mid-frequency bar strategy whose honest verdict is that it should not trade,
and Phase 6's model zoo is deliberately about proving the machinery rather than chasing latency.
Nothing in Phase 6 may be justified by a latency argument. The items below are kept because the
*reasoning* stays true if a future strategy ever does need them — reopening this is a decision, not
a resumption.

Optimizations, not rewrites (see [05](05-latency-model.md)):
- [ ] **Colocate in Tokyo** (AWS `ap-northeast-1`) — the biggest single lever.
- [ ] Promote a latency-critical strategy to **Boundary A** (inference + features in Rust via
      `axon-model`), gated by the parity harness. The *inference* half of that gate now exists
      and crosses the language boundary ([ADR-0021](adr/0021-rust-model-parity-gate.md)); the
      *feature* half does not, because there is no Rust feature runtime to gate.
- [ ] Kernel-bypass I/O + thread-per-core in the core; private API gateway.
- **Exit:** a strategy running sub-second, colocated, with Python out of its live loop —
  reached without changing the architecture.

## Cross-cutting (all phases)

- CI green (unit + golden + parity), fixed-point money math, no hot-path allocations.
- Testnet before mainnet, always. Secrets/signing isolated behind the `Signer`.
- Every hard-to-reverse decision gets an [ADR](adr/README.md).
