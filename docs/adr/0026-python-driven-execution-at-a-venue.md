# ADR-0026 — A Python decision at a venue: closing Phase 4, and what closing it does not mean

**Status:** Accepted · **Date:** 2026-07-26

## Context

[ADR-0020](0020-runtime-intent-source.md) joined the two halves of Phase 4 and its first
consequence says so: *"Phase 4 is joined end to end. A `Signal` written by Python now becomes an
`OrderRequest` at the venue."* That sentence was true about a code path and false about the world.
Every component between a Python target and a Hyperliquid order existed, was unit-tested, and had
been driven end to end — against **canned bytes and a spy `ExecutionClient`**. The offline session
records intents instead of submitting them (§10), deliberately, so nothing in `./run.sh` could
observe the submit path at all. The roadmap's Phase-4 exit criterion was marked *Not met* for
exactly one reason: *"the claim in this line is about a venue, and no venue has seen it."*

Closing it needed no new component. It needed a **cause** — something that would put a real target
position on a real ring, derived from real market data, at a moment a live core would still admit
it. There was nothing in the repository that could do that, and the reason is worth stating
because it is not obvious and it had been shipped as a demo:

1. **The only program that drove a `StrategyRunner` fed it a synthetic clock.**
   `python -m axon.live` generates an Ornstein–Uhlenbeck price path stamped from
   `--start-ns`, whose default is `1_700_000_000_000_000_000` — November 2023.
   `StrategyContext` stamps an emitted signal with the event's own time, and `SignalReader` ages
   that stamp against the core's event clock with a ceiling of `intent.max_signal_age_ms`
   (2 000 ms). So every record that demo has ever written is roughly two and a half years stale
   the instant a live session reads it. The demo works, is tested, and **could never have driven
   a venue** — and nothing said so, because offline both ends run on the same fiction.
2. **Nothing turned the market-data ring into strategy events.** `axon.marketdata` reads the ring
   the Rust core publishes and `axon.live.runner` drives a strategy from an iterable of events;
   the join between them did not exist. That join is the only way Python gets an event time both
   sides agree about, so without it the first problem has no fix that is not a wall clock.
3. **`session::live_sandbox_session` asserted only what a read-only session can prove** — and did
   it while leaving `intent.enabled` at its default of **on**, pointed at the default ring path.
   It would have obeyed whatever an unrelated producer had left at `/dev/shm/axon-signal.ring`.
   Its doc comment claimed "it never places an order"; that was a hope about the machine's state,
   not a property of the test.
4. **It could not run at all.** It read `AXON_HL_ACCOUNT`, a variable defined nowhere in this
   repository — `.env`, `.env.example` and `scripts/with-env.sh` all define
   `AXON_HL_ACCOUNT_ADDRESS`. The panic came before the socket. Worse, **cargo abandons a run at
   the first failing test binary**, so this panic could stop an unrelated live test in another
   crate from ever executing, and the failure an operator sees names neither.

## Decision

### 1. Python's clock is the venue's, and it arrives through the market-data ring

`axon.live.mdfeed.MdRingFeed` reads `MdSlice`s off the ring the core publishes and yields
`axon.strategy.events.Bbo` stamped with the slice's own `ts_event`. That closes the loop the
architecture always described and had never run in one process pair:

```text
  venue ─▶ Rust core ─▶ md ring ─▶ MdRingFeed ─▶ strategy ─▶ StrategyContext ─▶ signal ring ─▶ Rust core ─▶ venue
```

Two alternatives were available and both are worse.

*A second venue connection from Python* is the obvious one: `pip install websockets`, subscribe,
done. It gives Python a real venue timestamp without any Rust involvement — and it is the exact
divergence [ADR-0012](0012-market-data-ring-and-multi-record-contract.md) exists to prevent.
Features would then be computed on one book while orders are priced against another, and every
parity gate in Phase 5 would be comparing two feeds rather than two computations of one feed.

*A wall clock in the producer* is the cheap one: stamp `ts_event = time.time_ns()` and the record
is always fresh. It is also the [03](../03-ml-fidelity-and-features.md) training/serving skew this
codebase has gone to unusual lengths to make unavailable — `StrategyContext` imports no clock at
all, precisely so a strategy *cannot* do this. Making the driver do it on the strategy's behalf
would defeat the design from outside it. It is also not even safe: on a session whose feed lags,
a wall-clock stamp is *ahead* of the core's event clock, so the record is admitted while being
arbitrarily old, and the TTL check stops meaning anything at all rather than failing loudly.

The feed makes two further calls that are decisions rather than plumbing. **A slice becomes a
`Bbo` and never a `Trade`**: `MdSlice` carries the *last* print rather than each print, and under
the publisher's default `OnChange` policy any quote move republishes the same print on the next
slice — so one `Trade` per slice counts one execution as many times as the book moved afterwards,
and a strategy keying on trade arrival reads a quote-flicker storm as volume. And **a zero bid or
ask is skipped, not passed on**: the publisher zeroes all four quote fields rather than republish a
top of book it would not itself price against, and a strategy differencing against a mid of zero
emits a target the size of the whole book.

### 2. The thing that decides is a probe, and it says so in its own name

`axon.live.probe.TargetProbe` opens a position of a stated size, holds it for a stated span of
**event time**, and flattens. It has no view, no features and no model, and the module says that
in its first paragraph. The temptation was to use the reference mean-reversion strategy, which
would have made the run look more like trading; it would also have made the number of orders a
function of what BTC did in the next minute, on an account whose exclusive lock was held for one
workstream, with a soak test queued behind it.

Three of its properties are load-bearing rather than convenient:

- **It holds by event time.** Paced on `time.monotonic()`, the flatten would land at a moment
  determined by how fast this machine drained the ring, so replaying the same market would produce
  a different pair of orders — and that pair is exactly what the Phase-5 harness diffs.
- **It waits for the book before it decides.** A core that has received no market data skips the
  intent pass entirely ([ADR-0020](0020-runtime-intent-source.md) §2). A signal emitted into that
  window is not queued for later; it is consumed, aged and refused, and the strategy's first
  targets are gone with nothing to show for it.
- **It flattens with `FLAG_CLOSE`, not with a zero target.** A zero target is an opinion about the
  position, and one computed against a fill we have not yet been told about overshoots into the
  opposite side. Close ignores `target_qty` and implies reduce-only for that reason
  ([ADR-0014](0014-signal-to-order-planning.md) §2).

### 3. The run exercises the reconciled readings, not the easy ones

`ttl_ms = 0` and `urgency = 3` were chosen because they are the two fields whose meaning ADR-0020
had to *reconcile* between the languages, and a run that used the obvious values would have proven
nothing about either. Zero is the operator's ceiling on both sides (§4), so this run's records
carry the value a producer with no opinion about staleness emits — and they were admitted. Three
is IOC through the far touch, which leaves no remainder resting: what the venue holds afterwards
is a position and never an order, which is also what makes "leave the account flat" verifiable
rather than hopeful.

### 4. The evidence is a `cloid` derived independently on both sides

[ADR-0014](0014-signal-to-order-planning.md) §5 specifies the client order id as a pure function of
the signal's identity, with nothing hashed, *so that an operator reading a cloid out of a venue log
can recover which signal produced it*. That property had never been used for anything. The probe
now derives the id from the record it just emitted, from the specification rather than from the
Rust code's answer, and prints it; the venue echoes it on `userFills`. Two independent derivations
agreeing is what makes "an order appeared" into "**this** Python decision became that order".
Deriving it once and copying it would have proved that copying works.

This costs a second implementation of a bit layout, which is normally the thing to avoid. It is
worth it here for the same reason a test does not call the function under test to compute its own
expectation.

### 5. The `#[ignore]`d live test exercises the pump, and the read-only one says "read-only" out loud

`session::live_sandbox_session` keeps its assertions and gains `intent.enabled = false`, which per
[ADR-0020](0020-runtime-intent-source.md) §9 is the only way to *say* read-only rather than merely
hope for it. Beside it, `a_signal_on_the_ring_becomes_an_order_at_the_venue` writes two records
onto a private ring and asserts on `IntentLine::orders` — a counter incremented only after the
venue has answered, and therefore the first assertion in this workspace that cannot be true
without a venue.

Two things about that harness would be bugs in production code and are deliberate in it. It writes
the records with `axon_ipc::Producer` rather than shelling out to `axon.live.probe`: what this
crate owes an assertion about is the bytes, and starting two processes in the right order is a
launch-script property. And it stamps `ts_event` from the **wall clock**, standing in for a
producer whose event source is the live venue — the same substitution inside the runtime would
make a replayed session call every signal infinitely stale, which is why `SignalReader` takes the
core's event clock and `StrategyContext` has no clock at all.

Both are still `#[ignore]`d. Nothing in the default gate may touch the network, and the trading one
now says `PLACES REAL ORDERS` in its ignore reason rather than only in a doc comment.

### 6. `stale_seq` reaches the status line

Counted since [ADR-0014](0014-signal-to-order-planning.md) §1 and reported nowhere, which is the
same as uncounted. A restarted Python producer's `seq` rewinds, every record it writes is refused
until the sequence passes the old baseline, and the only visible effect is that `accepted` stops
rising — indistinguishable from a strategy with nothing to say, while Python's own counters climb.
It is a warning (`SIGNAL SEQ REWOUND n`) rather than a number, because it is never a state a
healthy session passes through: one occurrence means the two sides were restarted independently.

## Verification

Run 2026-07-26 against Hyperliquid **testnet**, with the approved agent wallet `rustml` signing for
the master account. `AXON_HL_NETWORK` was `testnet` throughout and the driver explicitly unsets
`AXON_ALLOW_MAINNET`.

**Which tree, and why the question needs asking.** Eight agents were editing this checkout
concurrently. It did not compile for most of this workstream, so the first binaries were built from
a `git archive` of `4727385` with only the relevant files copied over — this workstream alone, then
this workstream **plus M9's per-symbol in-flight gate**, which lands in `intent.rs` and therefore
changes the path under test. A merge that changes the submit path invalidates evidence gathered
before it, so each tree was run at the venue rather than argued about.

The tree eventually compiled, and **the run below is the full one**: `SIGNAL_SIZE=64
SCHEMA_VERSION=2`, M4's `MdSlice` v2 with the ticker tail and its second bar ring, M9's per-symbol
gate and order lifetime, M3's streaming log with its instrument record — every agent's work in the
same process, driving a Python decision to a testnet fill. The earlier runs produced the same shape
(`sig 2/0 sent 2+0c`, `swept: true, disarmed: true`); their `cloid`s were `0x98c5c53a5aa7ef80…` /
`0x98c5c540fc22fd40…` and `0x98c5c6318406b940…` / `0x98c5c6353c452340…`, and they are on the venue's
fill history for anybody who wants to check.

### Run 1 — `axon.live.probe` against a `sandbox` session

Python's two decisions, verbatim from the probe's own stdout:

```
probe: decision {"ask_px": "64345.00000000", "bid_px": "64343.00000000",
  "cloid": "0x98c5c6d25671cc800000000000000003", "decision": "open", "flags": 0,
  "model_version": 1, "quotes_seen": 3, "seq": 0, "symbol_id": 3,
  "target_qty": "0.00020000", "target_qty_fixed": 20000,
  "ts_event": 1785051434018000000, "ttl_ms": 0, "urgency": 3}
probe: decision {"ask_px": "64352.00000000", "bid_px": "64343.00000000",
  "cloid": "0x98c5c6d316ab64c00000000100000003", "decision": "close", "flags": 2,
  "model_version": 1, "quotes_seen": 6, "seq": 1, "symbol_id": 3,
  "target_qty": "0E-8", "target_qty_fixed": 0,
  "ts_event": 1785051437243000000, "ttl_ms": 0, "urgency": 3}
```

The same two records from the session's own signal log — the middle link, written by the recording
tap on the ring itself, by a process that never saw the probe's stdout:

```
{"release_ts":1785051434748469871,"signal":{"seq":0,"ts_event":1785051434018000000,
  "symbol_id":3,"target_qty":20000,"price_band":0,"ttl_ms":0,"max_order_age_ms":0,
  "model_version":1,"flags":0,"urgency":3,"kind":0,"schema_version":2,
  "pad0":[0,0,0],"reserved":[0,0,0,0,0,0,0,0]}}
{"release_ts":1785051437889884310,"signal":{"seq":1,"ts_event":1785051437243000000,
  "symbol_id":3,"target_qty":0,"price_band":0,"ttl_ms":0,"max_order_age_ms":0,
  "model_version":1,"flags":2,"urgency":3,"kind":0,"schema_version":2,
  "pad0":[0,0,0],"reserved":[0,0,0,0,0,0,0,0]}}
```

The status line either side of each order — `sig <accepted>/<rejected> sent <orders>+<cancels>c`:

```
contract   : SIGNAL_SIZE=64  SCHEMA_VERSION=2  RING_HEADER_SIZE=192
md ring    : /dev/shm/axon-m2-md.ring (cap 4096, policy OnChange); bars /dev/shm/axon-m2-md.bars.ring
axon 0:00:02 … | open 0 flat       | … | sig 0/0 sent 0+0c | md 1 q 0/4096 | OK
axon 0:00:10 … | open 0 flat       | … | sig 2/0 sent 2+0c | md 8 q 0/4096 | ORPHAN FILLS 14
shutdown: ShutdownOutcome { swept: true, disarmed: true, errors: [] }
capture: 185 events + 2 signals -> data/captures/m2-venue-proof.jsonl
```

(The position appears and clears between two-second status lines here, because the probe held for
3 000 ms of event time rather than the 8 000 of the earlier run. The two intermediate lines from
that run read `open 0 BTC+0.0002 | … sig 1/0 sent 1+0c` and then `open 0 flat | … sig 2/0
sent 2+0c`.)

And the venue's own answer, `POST /info {"type":"userFills"}`, echoing both `cloid`s in full:

```
{"cloid":"0x98c5c6d25671cc800000000000000003","coin":"BTC","dir":"Open Long","side":"B",
 "sz":"0.0002","px":"64345.0","oid":57002801872,"fee":"0.005791",
 "closedPnl":"0.0","startPosition":"0.0","time":1785051435347}
{"cloid":"0x98c5c6d316ab64c00000000100000003","coin":"BTC","dir":"Close Long","side":"A",
 "sz":"0.0002","px":"64343.0","oid":57002803350,"fee":"0.00579",
 "closedPnl":"-0.0004","startPosition":"0.0002","time":1785051438520}
```

The open filled at `64345.0`, which is **the ask the Python decision recorded**, and the close at
`64343.0`, **the bid**. Signal `ts_event` to venue fill time: 1 329 ms and 1 277 ms.

### Run 2 — the `#[ignore]`d test, on its first run

`a_signal_on_the_ring_becomes_an_order_at_the_venue` reached `sig 2/0 sent 2+0c` with the same
position appearing and flattening, `swept: true, disarmed: true`, and `cloid`s
`0x98c5c653abf2d91e0000000100000003` / `0x98c5c6560002f6df0000000200000003` on the venue's fills —
`Open Long` at 64343.0 then `Close Long` at 64338.0, `startPosition` 0.0 then 0.0002.
`live_sandbox_session` was run immediately before it and passed, printing
`intents OFF (read-only session)`.

### The run that decided nothing, and said so

Between the two, one run produced **no order at all** and it is the more instructive result.
Testnet BTC's top of book moved 8 times in 75 seconds — `md 8 … coal 14` — against a probe that
wanted 10 quotes before deciding. It exited non-zero with
`probe: NO DECISION - the probe saw 8 usable quotes and needed 10. Nothing was emitted, so nothing
could trade.` That sentence is the whole reason the check exists: a probe that emitted nothing and
a session that placed nothing produce byte-identical evidence to a working system on a quiet
market, and every counter on the Rust side reads `sig 0/0 sent 0+0c | OK`.

A **rehearsal** was also run first, identical but with a target of 0.00001 BTC (~$0.64). It
exercised every link except the submit and stopped exactly where it should have —
`sig 2/0 sent 0+0c prec 1` — the planner refusing a notional under the venue's $10 floor and
counting it where "already at target" could not hide it. It cost no `/exchange` action. Rehearsing
a venue run against the *venue's own refusals* turns out to be free, and it is the cheapest way to
find out whether the plumbing works before finding out whether the money does.

### Run 3 — the tape, which is a second artifact

`run_live` and `run_offline` now start their recorder with `SessionRecorder::start_with`, handing
it the **same** `Arc<InstrumentTable>` the intent source plans against and the exchange client
encodes against. Without it a live capture declared no grid, so a replay of it planned under
`Precision::Unconstrained` while the session that wrote it planned under `Precision::Known` — the
golden diff then shows *unrounded* prices and reports the rounding as a strategy change. That is
[ADR-0025](0025-instrument-precision-and-rounding.md)'s hole and ADR-0027's fix meeting at the one
place that knows both, which is the composition root. The venue run's log opens with
`{"Instruments":{"Declared":{"unconstrained":false,"instruments":[…]}}}` and the startup line reads
`grid 210 instrument(s)` — the venue's whole perp universe, from the one `meta` read — instead of
`grid UNDECLARED`.

That tape is the first in this repository that both **declares a grid and traded a venue**, and it
closes the round trip on real traffic rather than on a generated fixture. Replayed through the
production chain:

```
replay_log: planning on the grid this log declared (210 instrument(s))
{"events":170,"late_arrivals":40,"intent_passes":2,
 "signals":{"records":2,"accepted":2,"rejected":0,"expired":0,"planned":2,"no_quote":0},
 "orders":[{"signal_seq":0,"cloid":"0x98c5c735fff96c000000000000000003","side":"buy",
            "qty":"0.00020000","price":"64672","tif":"ioc","reduce_only":false},
           {"signal_seq":1,"cloid":"0x98c5c7374d0b89000000000100000003","side":"sell",
            "qty":"0.0002","price":"64019","tif":"ioc","reduce_only":true}],"cancels":[]}
```

Same two `cloid`s, same sides, same sizes, same TIFs, same quantized limits as the session that
placed them. **40 of its 170 records arrived late** — 23%, against a synthetic fixture's single
deliberate inversion — which turns ADR-0018 §4's "compare a live reference only under
`--order as-captured`" from an argument into an observation.

### The account, after everything

Verified **three times, at three different venue timestamps** rather than once — a single hopeful
read is exactly what the live fill test refuses to call proof: `assetPositions: []`,
`totalNtlPos "0.0"`, `totalMarginUsed "0.0"`, `openOrders: []`, `frontendOpenOrders: []`, all three
identical. `accountValue` 998.974072 → 998.89646 across six round trips; the 0.077612 difference is
the twelve fills' fees and closed PnL and nothing else. `cumVlm` 48.88 → 203.26 — each round trip
~$12.87, inside the venue's $10 minimum and the $50 ceiling the operator set. `nRequestsUsed`
22 → 101 against a cap of 10 203. **No `scheduleCancel` trigger was spent**: every shutdown
disarmed cleanly, so the venue's ten-a-day budget is intact for the soak queued behind this.

## What this proves

- **A Python decision becomes an order at a venue, and the order is traceable back to the
  decision.** Not by inference from a timestamp: the venue echoed a 128-bit id that Python and
  Rust each derived independently from ADR-0014 §5's layout. Every link in
  `Signal → ring → SignalReader → Planner → queue → pump → Haltable→Guarded→Governed→Exchange →
  venue` carried real bytes at least twice.
- **The order was the delta, and the flatten actually flattened.** A build that sent the target
  instead of the delta would have shown `startPosition 0.0002` and `dir "Open Long"` on the second
  fill. It shows `"Close Long"`.
- **ADR-0020 §4's reading of `ttl_ms == 0` survives contact with a venue**, on records whose
  producer never stated a TTL, against a core whose feed lag was measured at 35–1 300 ms. Event
  time is why: both the stamp and the ceiling are the venue's clock, so the ~1 s the wall clocks
  disagree by never entered the decision. Had the producer stamped `time.time_ns()`, every record
  would have arrived up to 1.3 s *ahead* of the core's clock — admitted, counted in
  `ahead_of_clock`, and with the TTL check no longer measuring anything.
- **ADR-0025's quantizer priced twelve more fills the venue accepted first try**, in both
  directions, from a *computed* price rather than one the venue emitted — independently of M1's
  fill test, which established the same thing an hour earlier. The replay re-derived the same two
  quantized limits from the log's own declared grid.
- **The submit path's shutdown ordering works against a real venue.** `swept: true,
  disarmed: true` on eight consecutive sessions, and the account was left order-free every time,
  verified three times over.
- **The offline session and the live one plan the same way.** The rehearsal's refusal
  (`prec 1`, notional under the floor) is the same counter the offline tests assert on, reached
  live, from the venue's own `meta`-derived grid rather than a fixture's.

## What this does not prove — and this list is the more important one

- **Nothing about whether the strategy should trade.** `TargetProbe` has no view. It buys and
  sells because it was told to, and its P&L (−$0.078 over six round trips, almost all of it fees)
  is the cost of the experiment and not a result. The roadmap's *proven live* mark on this line
  says a path works; it says nothing about edge, and [ADR-0022](0022-first-ml-strategy.md) already
  records what happens when those two are conflated.
- **Nothing about a strategy that trades continuously.** Twelve fills, across six sessions, on one
  instrument, over about thirty minutes. The in-flight gate was observed doing its job (`busy 3`,
  on the pre-merge global version) and never observed under pressure — and M9's replacement, which
  is per symbol and *holds* a gated target rather than dropping it, was never contended at all on a
  single-instrument run. `INTENT STALLED`, `SIGNAL RING DETACHED`, `POISONED TRACKER`, `stale_seq`,
  the no-op band and the order-lifetime sweep have still only ever been seen in tests. The rate
  governor spent 52 address credits out of 10 151.
- **Nothing about the reconnect.** One socket, no disconnects, no resubscription, no replayed
  `userFills` snapshot after a gap. That is Phase 2's owed soak, and it is still owed.
- **Nothing about a resting order.** Every order in every run was an IOC that filled outright, so
  the whole cancel/replace half of [ADR-0014](0014-signal-to-order-planning.md) §6 — the
  leave-it-resting exception, cancel re-addressing by venue order id, the adopted-order case — was
  never reached. `sent 2+0c` every time: **zero cancels have ever been sent through this path at a
  venue.** A post-only or GTC strategy walks a path none of this touched, and M9's no-op band and
  order lifetime, which both act only on resting orders, were live and inert throughout.
- **Nothing about the market-data ring at rate.** BTC's testnet top of book moved between 8 and 22
  times per *minute* across these runs, so `dropped 0` everywhere and the deepest the ring ever got
  was 7 of 4096. Backpressure, coalescing at volume and the Python reader's `dropped` accounting
  are all untested at a rate that would matter. The same sparseness makes `hold_ms` a floor rather
  than a schedule: on the pre-merge run it was 8 000 ms and the two decisions came 28 479 ms apart
  in event time, because the next quote after the window simply took that long. That is correct
  behaviour and it is not obvious behaviour.
- **Nothing about a Rust-side model.** The decision was `if quotes >= 10`. No feature was computed,
  no artifact was loaded, and the whole of Phase 5's fidelity apparatus sat idle beside this run.
- **Nothing about mainnet.** `AXON_HL_NETWORK` was `testnet` throughout and `AXON_ALLOW_MAINNET`
  was explicitly unset by the driver. The key that signed these orders is public; Phase 6 needs
  a fresh one and a second pair of eyes, and neither is in scope here.
- **Nothing about the tree as it will finally stand.** The last run *was* the fully merged
  checkout, which is much better than the alternative — but it was a snapshot of a tree eight
  agents were still editing, taken at 10:20, and it is not the commit. In particular it does not
  cover any change made after that moment, and it exercised M4's `MdBar` ring only in the sense
  that the session created one: **no bar was ever published, read or consumed**, because no candle
  feed was subscribed.

## Consequences

- **+** Phase 4's exit criterion is met, and the roadmap can say *proven live* about the one line
  it has been unable to. The distinction this document exists to protect is that the line means
  what it says and nothing adjacent to it.
- **+** `axon.live.mdfeed` makes the Rust→Python direction usable rather than merely built. Every
  Python-side workstream that needs live features — the shadow-trade ladder rung, the live parity
  monitor — now has an event source that is not a fiction, and one that a `symbol_id` filter keeps
  from interleaving two books through one set of callbacks.
- **+** Two live tests now exist that place orders and clean up after themselves, so the next
  change to `submit_intent`, `pump` or the shutdown ordering is falsifiable in under a minute
  instead of by argument.
- **+** `SIGNAL SEQ REWOUND` closes the last counted-but-invisible refusal on the boundary. The
  status line can now distinguish all four ways a session with a healthy feed places no orders: no
  producer (`SIGNAL RING DETACHED`), a producer whose sequence rewound (this), a producer whose
  records expire (`sig n/m` with `expired` climbing), and a strategy that is simply flat
  (`sig n/0 sent 0+0c`).
- **−** `python -m axon.live`'s synthetic feed is still stamped from 2023, and it is still the
  first thing a reader finds. It now says so — its module docs, its `--start-ns` help and a
  stderr warning on any non-`--drain` run all name the consequence — but a warning is not a fix,
  and the default remains a value that cannot drive a live session. Changing it to "now" would
  cost the reproducibility that makes the demo worth having, so somebody has to decide which of
  those two the default should serve.
- **−** The probe is one more strategy-shaped object outside `axon.strategies`, and the only reason
  it lives in `axon.live` is file ownership during a parallel fan-out. It belongs beside the other
  strategies or in a `tools` package, and moving it is a rename nobody has done.
- **−** `MdRingFeed` yields `Bbo` and drops everything else, so a strategy that needs the trade
  print has to read `last_slice` and do its own de-duplication. The right fix is for the feed to
  emit a `Trade` when `last_trade_ts` advances, which is a comparison this version does not make.
- **−** The evidence chain has a second implementation of the `cloid` layout, in Python, with
  nothing enforcing that the two stay in step beyond a test that asserts the specification. A
  layout change (ADR-0014 §5 already warns that adding a leg index must re-cut it) breaks the
  Python copy silently, at exactly the moment the id it prints stops matching.
- **−** The `#[ignore]`d trading test stamps a wall clock, so it is the one place in the workspace
  where a `Signal`'s `ts_event` does not come from an event. It is confined to a test and
  commented, and it is still a shape somebody will copy.
- **−** Both live tests kill their own process with `SIGINT` on a timer. It exercises the real
  shutdown path, which is the point, and it means a harness that runs them in parallel signals
  the wrong test — they must be run one at a time, and nothing enforces that.
- **−** Adding two fields to `IntentConfig` broke **every config file on disk**, because `[intent]`
  already exists in all of them and a new required key makes the whole file fail to parse. It was
  found by a venue run refusing to start with `missing field noop_band_bps`, which is a message
  about TOML for a change about queue position. `#[serde(default)]` on each field fixes it and a
  test now pins it — but the shape recurs every time a table grows a key, and nothing but that test
  catches it.
- **−** These runs consumed no `scheduleCancel` triggers, because every shutdown was clean. A run
  that is interrupted between the sweep and the disarm leaves the deadline standing and spends one
  of the venue's ten daily triggers; that is [ADR-0013](0013-runtime-supervision-and-safety-loop.md)
  §6 working as designed, and it is a cost a soak in the same UTC day inherits.
- **−** **A forming candle moves the core's event clock into the future, and that clock is what
  ages every signal.** Found while characterising what the intent pass depends on, and pinned by
  `a_forming_candle_pushes_the_core_clock_into_the_future_and_stops_the_pass`. A candle's
  `ts_event` is `open_time + interval` — the moment the bar *closes* — and a venue republishes the
  forming bar many times before then, so `CoreHandler::last_ts` jumps up to a whole interval ahead
  of anything that has happened. Nothing on `axon_core::Candle` distinguishes a forming bar from a
  closed one. Two consequences, neither of which says anything: every record on the ring is aged
  against that clock and refused as expired, and `last_pass_ns` is left an interval ahead, so the
  schedule's signed subtraction — correct, and there to stop a late arrival triggering a pass —
  runs **no further pass at all** until event time catches up. The session prints `expired`
  climbing and `OK` beside it. Candles are not in the default `session.feeds`, which is the only
  reason nobody has hit it; the fix belongs in the venue decode or on `Candle` itself, not here.
- **−** Every session in this workstream started with `ORPHAN FILLS` between 5 and 13, because the
  `userFills` snapshot replays executions belonging to orders no fresh process has ever heard of.
  Correct — the tracker refuses to invent an order to hang them on — and it means a warning that
  should mean "we lost a fill" is on the status line of every cold start, where it will be learned
  as noise. Suppressing the replay window is `axon-execution`'s call, not this ADR's.

See [ADR-0020](0020-runtime-intent-source.md) (the intent source this exercises),
[ADR-0014](0014-signal-to-order-planning.md) (the reader, the planner and the `cloid` layout),
[ADR-0012](0012-market-data-ring-and-multi-record-contract.md) (the market-data ring this reads),
[ADR-0025](0025-instrument-precision-and-rounding.md) (the quantizer these prices came from),
[ADR-0013](0013-runtime-supervision-and-safety-loop.md) (the session, the switch and the sweep),
[02](../02-python-rust-boundary.md), [06](../06-strategy-contract.md), [08](../08-roadmap.md).
