# 07 — Parity & Testing

"No quality loss" is a *claim* until it's a *test*. This document defines the validation
ladder that turns the claim into a gate every deploy must pass.

Parity has to hold at four layers, each with its own failure mode and its own test (see
[03](03-ml-fidelity-and-features.md) for the model/feature detail):

| Layer | Failure mode | Test |
|---|---|---|
| **Model** | Python model ≠ Rust inference | Golden model test (exact for trees; ε-tolerance + decision-invariance for NN) |
| **Features** | Offline features ≠ online features (training–serving skew) | Feature parity test + live parity monitor |
| **Execution semantics** | Vectorized backtest ≠ event-driven live | One event-driven core in *both* → replay golden test |
| **Numerics** | Float/precision divergence (incl. platform width bugs) | Deterministic replay with explicit epsilon tolerances |

> Cautionary tale to remember: NautilusTrader ships **different precision on Windows** (64-bit,
> MSVC lacks `__int128`) vs elsewhere (128-bit) — *same logic, different numbers*. Pin
> precision mode explicitly and put it under test. Don't assume "same code" = "same number"
> across platforms.

## The validation ladder (every deploy climbs it)

```
  1. Golden / replay tests   ──▶  2. Paper / sandbox   ──▶  3. Shadow trading   ──▶  4. Live
     (deterministic, CI)          (testnet, real feed)      (live feed, no orders)     (real orders)
```

### 1. Golden / replay tests (deterministic, in CI)

- Capture a real event log (market data + timestamps). Replay it through the **exact
  production code path** (the same Rust core, the same Python strategy/features/model).
- Assert outputs match a stored reference **within tolerance**, and that **no discretized
  decision flips**.
- Because the core is a single-threaded deterministic loop keyed on **event time** (not
  processing time), replay is bit-reproducible. This is *why* the architecture insists on
  determinism.
- Includes the **model-parity gate** and **feature-parity gate** from [03](03-ml-fidelity-and-features.md).

**Both of those gates now cross the language boundary, and they are different gates.** Read the
distinction before trusting either, because for a long time only one existed and the other was
assumed:

| | what it compares | criterion | run by |
|---|---|---|---|
| **Model parity** ([ADR-0021](adr/0021-rust-model-parity-gate.md)) | Python's scores over a holdout matrix against **Rust's** scores over the same bytes | bit-exact for trees, two ULP for graphs, **and** no discretized decision flips | `./run.sh parity` |
| **Feature parity** ([ADR-0035](adr/0035-rust-feature-runtime-and-the-bit-exact-gate.md)) | Python's feature matrix against **Rust's**, each computed from the same market data | **bit-exact**, with NaN-on-both a match and NaN-on-one a mismatch | `./run.sh feature-parity` |

A model bundle hands Rust a matrix Python already computed, so it can prove that identical
vectors produce identical decisions and can never prove the two languages would *produce* the
same vectors. That is `docs/03`'s harder half, and until ADR-0035 nothing in Rust computed a
feature to compare against. The number the older gate reported as `max_abs_diff = 0.0` was a
Python path against a Python path — the Boundary-B guarantee, and precisely *why* it was zero.

There is also a **third** feature comparison, and it answers a different question again:
`aligned_feature_parity` diffs an *online* Python path against an *offline* Python recompute
(ADR-0016 §4, ADR-0030). That one catches staleness, late events and windowing bugs; it cannot
catch a language difference. All three are needed and none substitutes for another.

### 2. Paper / sandbox (Hyperliquid testnet)

- Run the full stack against Hyperliquid **testnet**: real WS feed, real signing/nonce paths,
  real order lifecycle — no real money.
- Validates the adapter (signing schemes, nonce windows, reconnection, `schedule_cancel`),
  not just the strategy.

### 3. Shadow trading (live feed, no orders sent)

- Run the production strategy against the **live** feed, computing signals and *would-be*
  orders, but **not** sending them.
- Continuously **diff live signals vs. the reference/backtest** on the same data. This is the
  test that catches real-world drift (latency, staleness, out-of-order events) that testnet
  and CI can't reproduce.
- Shadow trading is a *practice we assemble*, not a packaged product — the durable
  event-capture + replay half comes from the core; the diffing is ours.

### 4. Live

- Promote only after shadow trading shows parity within tolerance over a meaningful window.
- Keep the parity monitor running **in production** (see below) — parity is not a one-time
  gate, it's a continuous property.

## The live parity monitor (runs forever)

The mandatory backstop from [03](03-ml-fidelity-and-features.md):
- Sample live feature vectors + signals in production.
- Recompute them via the offline Python path.
- Alarm on divergence beyond tolerance; track feature drift (PSI/KL).
- Catches the classic silent killers: **timezone bugs, late/out-of-order events, rounding,
  windowing off-by-one, NaN/null handling, staleness.**

**It could not watch a session that was *trading*, and the reason was structural rather than a
missing feature.** The monitor attaches to the session's bar ring, and on a live run the strategy
*is* that ring's only consumer — an SPSC ring has one consumer, and two readers steal from each
other rather than sharing. Running a second, read-only session beside the trading one is worse: a
graceful shutdown calls `cancel_all`, which on Hyperliquid is account-wide, so the watcher would
cancel the trader's resting orders on its way out.

So for a trading session the diff was computed **after it**, from the session's own capture
(`scripts/sessions/session_parity.py`), over exactly the bars it saw. What that gives up is the
alarm — it cannot say mid-run that parity has broken.

**Closed 2026-07-27** ([ADR-0037](adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md)
§5), and the fix is the one this paragraph named: a **fan-out inside the bar-ring reader** — one
consumer, dispatched to both the strategy and the diff. `axon.strategies.live_runner --parity-diff`
is that reader, and what it dispatches to is
`axon.strategies.shadow.BarParityDiff`, **extracted from the shadow harness rather than written
beside it** — so there is one alignment rule, one window construction and one `ParityMonitor`
configuration between the two callers. A cut-down second comparison living next to a live order flow
is how two answers to one question get into a tree, which is the reason this was a fan-out and not a
second reader.

Two things it deliberately does not do. **A parity break alarms and never stops the run**: the
process is holding a position and has a signal ring the Rust core is reading, so raising would
abandon both with no flatten — acting on it is the operator's call, and `axon --flatten` is how they
act. And it is **off by default**, because it recomputes the whole spec at every window boundary on
the thread that is about to stamp a decision.

The after-the-fact capture route is unchanged and is still the right thing for a session that ran
without the flag.

## What makes all this possible: reproducibility by construction

Every live decision records **which model version** and **which input sequence** produced it
(carried on the `Signal`, see [06](06-strategy-contract.md)). Combined with the durable event
log, *any* live signal can be reproduced offline and compared. Without this, none of shadow
trading, audits, or post-mortems are possible.

## Test taxonomy (planned)

| Kind | Where | Example |
|---|---|---|
| Unit | Python + Rust | feature functions, order mapping, nonce manager |
| Golden / replay | CI | strategy output vs. reference on a captured log |
| Model parity | CI | Rust inference vs. Python on N-thousand inputs |
| Feature parity | CI + live monitor | offline vs. online feature vectors |
| Property / fuzz | Rust | order-book invariants, ring-buffer under load |
| Integration | testnet | full stack on Hyperliquid testnet |
| Soak / chaos | sandbox | reconnection, peer-crash, dead-man's-switch — **run**, see below |

## The soak, and the four things only duration found

Owed since Phase 2 and first run on **2026-07-26**: **1 h 44 m 16 s** of one continuous
read-only sandbox session against Hyperliquid testnet (BTC, ETH, SOL × `bbo`, `l2Book`,
`activeAssetCtx`), through **36 scripted network outages and 3 whole-process freezes**,
with `--capture` on throughout. Harness in [`scripts/soak/`](../scripts/soak/); the tape
is `data/captures/m7-soak.jsonl`.

Two things about the method are load-bearing, because a soak that induces its outages
badly measures its own harness:

- **The disconnects are real, and they are not simulated in the code under test.** The
  session is pointed at a loopback relay through `venue.ws_url` and the relay severs the
  TCP connection — RST, and the port then answers `ECONNREFUSED`. Asking the client to
  pretend to disconnect would test the pretence. `info_url` and `exchange_url` stay
  pointed at the real venue, so reconciliation and the safety loop keep talking to
  Hyperliquid *through* an outage that only cuts market data — the interleaving that
  cannot be arranged any other way.
- **Read-only is stated, not assumed.** `intent.enabled = false` is the only way to say
  it ([ADR-0020](adr/0020-runtime-intent-source.md) §9). None of the properties under
  test needs order flow.

### What survived

Reconnect and resubscribe, over **37 connections**: all nine (instrument × feed) streams
delivered events in 35 of the 37 connection windows, and the two exceptions are both
windows shorter than the quiet period of a `bbo` on a slow testnet coin (8.0 s and
6.1 s). **No half-restored subscription was observed** — the failure
[ADR-0020](adr/0020-runtime-intent-source.md) §7 warns can hide behind a live
`activeAssetCtx` keeping `MarkCache::get` fresh over a book that stopped updating.

`ORPHAN FILLS 18` on cold start, and **18 on every one of the 1 222 status lines** after
it. The `userFills` snapshot replays in full on each of the 37 reconnects — 666 fill
records in the tape, 18 × 37 — and `OrderTracker`'s dedup on `trade_id` absorbed every
one. The warning is correct behaviour and it is also permanent furniture, which is the
thing to fix about it.

`/info` reconciliation agreed with the tracker for **1 h 44 m and roughly 415 polls**
with no failure and no divergence, including across a 10-minute blackout.

Memory, threads and descriptors are flat: RSS peaked at **26.9 MiB in the first minute**
and spent the remaining 94 minutes oscillating between **15.7 and 20.5 MiB**; never more
than 12 threads or 16 descriptors. The capture wrote **5.54 MiB over 1 h 44 m**
(~3.2 MiB/h) with `max_bytes = 0`, and the core→writer queue's deepest excursion was
**23 of 16 384**, during a process freeze. Replaying that tape twice produced
**byte-identical** summaries *and* byte-identical 13 939-row traces, in 1.2 s at 105 MiB
peak RSS.

### What did not survive

Four findings, each of which needed the duration. None is fixed here — this workstream
owns no crate source, and a precise reproducer in the hands of the owner is worth more
than a patch from someone who cannot test it:

1. **The WS reconnect backoff never resets, so a session degrades monotonically.**
   `reconnect_forever` doubles the backoff on every `Err` and resets it only when
   `run_once` returns `Ok` — a *clean* close. A severed connection is never a clean
   close, so after eight disconnects the backoff is pinned at its 30 s cap for the rest
   of the session however healthy the link is in between. Measured: the session's **first**
   0.8 s outage cost 0.8 s of blackout; a **0.4 s** outage two hours later cost **30.0 s**
   — 75×. Seven outages of ≤ 3 s, 8.0 s of induced downtime in total, produced **127.8 s**
   of blackout. With `mark_max_age_ms = 10 000` every one of those windows expires every
   mark, and **45.8 % of the session ran under `STALE MARKS`** with the risk gate refusing
   every risk-increasing order.
2. **The dead-man's switch escalates on re-arm *failures*, never on protection running
   out.** `dms::run` computes an `Escalation` only from `on_failure`, which is only
   called when `arm()` returns `Err`. Freezing the process with `SIGSTOP` for 80 s
   against a 60 s lead drove the status line to `dms 0s` — protection gone, the
   venue-side switch fired — and the session printed **no `HALTED`, no `UNPROTECTED` and
   nothing on stderr**, then re-armed and reported healthy. A 48 s freeze reached `dms 9s`,
   deep inside the "one interval or less" band, with the same silence. ADR-0013 §3's
   table is graded by protection remaining; the loop only consults it after an error, so
   a stalled process — the case the switch exists for — walks the whole ladder without
   touching a rung.
3. **A replayed `userFills` snapshot drags the core's event clock hours backwards.**
   `CoreHandler::on_event` *assigns* `last_ts = ts_event` rather than taking a maximum,
   and `advances_the_clock` admits every `Event::Exec`. `userFills` replays its whole
   snapshot on every reconnect carrying the original execution times, so the clock is set
   to the oldest replayed fill. Observed on five status lines across the two sessions with
   **`marks 3/3`** beside them — a healthy feed and a clock between 25.8 minutes and
   **2.49 hours** in the past. The arithmetic settles it: three occurrences 820 s apart
   reported lags 820 064 ms apart, so the clock was pinned to a *fixed* historical instant,
   not lagging. It self-heals on the next market frame, which is why it has never been
   noticed; the reconnect-backoff defect above is what makes that window up to 30 s long.
   This is the same line the forming-candle fix touched, and the mirror image of it —
   candles pushed the clock forward, fills pull it back.
4. **Every heartbeat is logged as a decode error.** The venue's pong is
   `{"channel":"pong"}` with no `data` field; the `Envelope` struct requires one. 46
   spurious `WS decode error (frame dropped)` lines in 1 h 44 m — noise sitting exactly
   where a real decode error would appear. The unit test that says pongs are ignored
   hand-writes `{"channel":"pong","data":null}`, a frame the venue does not send.

### What a "late arrival" on a real tape is actually made of

`replay_log` reported **2 222 late arrivals of 13 939 records (15.94 %)** on a tape with
no candles in it — squarely in the 15–23 % band earlier synthetic and short tapes showed.
Decomposed, almost none of it is the network:

| subset | late | of | |
|---|---:|---:|---|
| everything (what the summary reports) | 2 222 | 13 939 | 15.94 % |
| excluding the `userFills` snapshot replays | 1 556 | 13 273 | 11.72 % |
| venue-stamped market data only (`bbo` + `l2Book`) | **3** | **3 876** | **0.08 %** |
| …within any single instrument's own stream | 1 | 3 876 | BTC only |

Two structural artefacts account for the rest, and both are the same shape as
[ADR-0027](adr/0027-streaming-logs-and-the-replay-grid.md)'s forming-bar case.
`activeAssetCtx` carries no venue timestamp and is stamped at **receipt**
([ADR-0011](adr/0011-ticker-and-mark-price-feed.md)), so it always advances the
high-water mark and every venue-stamped frame behind it is "late" by one network hop —
it is 65 % of this tape. And `userFills` replays its whole snapshot on every reconnect
carrying the original execution times, up to **2.94 hours** stale. So
`reordered_by_the_feed()` still needs the same subtraction for receipt-stamped feeds and
for snapshot replays that it now gets for candles, or it will keep reporting a network
problem on any tape that has a ticker or a reconnect in it.

The number worth carrying forward: **across 36 deliberate disconnects, Hyperliquid
testnet reordered venue-stamped market data 3 times in 3 876 records.**

### The candle feed, under the same treatment

A second **32 m 24 s** session followed the first with `{ Candles = "m1" }` added and
nothing else changed (`data/captures/m7-soak-candles.jsonl`), through 15 more induced
outages. It is the A/B for the forming-candle fix, and the fix holds:

| | candles off (1 h 44 m) | candles on (32 m) |
|---|---:|---:|
| event rate | 2.23 ev/s | 2.59 ev/s |
| core clock lag, mean | 976 ms | **977 ms** |
| core clock lag, p50 | 536 ms | 580 ms |
| `lag 0ms` samples | 0 | **0** |
| `BARS BUT NO CLOCK` | 0 | **0** |

Event rate rises by the candle frames; the clock is statistically unmoved. (Samples taken
during an induced blackout, and the clock-rewind samples from finding 3, are excluded from
both columns — including them would compare the outages rather than the feeds.) `lag 0ms`
is the symptom that would say the fix is absent, and `BARS BUT NO CLOCK` is the state a
half-restored resubscribe would leave behind; neither appeared in 16 reconnects.

The three late-arrival counters on that tape, reported together so the subtraction is
visible rather than asserted: **`late_arrivals` 3 942 of 5 031 (78.4 %),
`behind_forming_bars` 3 533, `reordered_by_the_feed()` 409.** 142 candle frames over
32 minutes on three coins is enough to put 70 % of the tape behind a bar that has not
closed. And the 409 the subtraction leaves are *still* not the network: decomposed the same
way as above, **0 of 1 244** venue-stamped `bbo`/`l2Book` records were out of order. The
remainder is the ticker's receipt stamp and the `userFills` replays — which is the argument
for giving `reordered_by_the_feed()` those two subtractions as well.

## Definition of "safe to go live"

A strategy is cleared for live when **all** hold:
1. Model-parity gate green (exact/ε, decision-invariant).
2. Feature-parity gate green.
3. Replay golden test green.
4. Testnet integration green (signing, lifecycle, resilience).
5. Shadow trading parity within tolerance over an agreed window.
6. Live parity monitor + dead-man's-switch armed.
