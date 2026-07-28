# ADR-0038 — Many strategies, one account: netting, allocation, and the bounds that only exist across symbols

- **Status:** Accepted
- **Date:** 2026-07-27
- **Supersedes / amends:** amends [0006](0006-signal-schema-and-spsc-ring.md) (one ring
  per producer, stated), [0014](0014-signal-to-order-planning.md) §6 (the planner now runs
  on a *netted* record), [0020](0020-runtime-intent-source.md) §3 (the pass has two stages
  where it had one), [0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md) /
  [0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md) (the held
  target and its re-quote budget move into a book keyed on *(strategy, symbol)*).

## Context

Every session this project has ever run has traded **one instrument through one
strategy**, and the reason is one sentence: an SPSC ring has one producer, there was one
ring, so there was one strategy. Everything downstream that *looks* multi-instrument — the
per-symbol in-flight gate ADR-0020 §3 built, the per-symbol held target ADR-0036 added, the
per-symbol sweep of ADR-0031 — had only ever been exercised at width one, because nothing
upstream could put two symbols on the wire at once.

The roadmap has carried "multi-strategy / multi-symbol orchestration; portfolio-level
risk" as Phase 7's second item since Phase 0. The handoff's R5 states the shape of it and
the trap in it:

> One signal ring has one producer, so today one session trades one instrument through one
> strategy. Note the loss switch is **per session**, not per strategy, and a portfolio
> bound is a different object.

That last clause is the whole problem. `[strategy.risk]`'s three limits — `max_position`,
`max_notional`, `max_order_qty` — are each about one instrument, and `LossLimiter` is about
one session's money. **Neither can see a book.** Ten instruments each sitting at 90 % of
their own `max_notional` are inside every limit the session declares and hold nine times
the exposure the operator had in mind, and no number in either table would have moved.

## Decision

Five pieces, each in the crate that already owns the concept.

### 1. One ring per producer, and the strategy is a property of the ring

A session declares `[[strategy.producer]]` tables: a name, a ring path, the instruments it
may speak for, a silence policy, and an allocation. `RuntimeConfig::producers()` resolves
that list — **and resolves a session with no tables into a one-element list built from
`[ipc]`**, so there is no branch anywhere downstream between one strategy and several. The
single-producer path is the one that has run at a venue; a branch would mean the tested
path is not the shipped one.

The alternative — one ring carrying a `strategy_id` on the record — was rejected on three
grounds, and the first is decisive. **`seq` is per writer.** It is the only proof that
nothing was lost, and two producers interleaving into one stream the reader validates as
one sequence means every record of the loser is refused as `stale_seq`: a strategy emitting
normally, its own counters climbing, and nothing reaching the venue — the exact failure
ADR-0014 §1 already documents for a *restarted* producer, arriving by a second route.
Second, the ring is SPSC by construction (ADR-0006) and putting two writers on it is not a
schema question. Third, the `Signal` record has been fully named since schema v3, so a
`strategy_id` field would have to **re-cut the 64-byte layout** — an expensive change to
buy a property config already has.

`validate` refuses two producers on one path by name, and the message says why: *"An SPSC
ring has one consumer: two readers do not share it, they steal from it."* That is the same
hazard ADR-0029 §5 measured on the **bar** ring on 2026-07-26, where two drainers each saw
about half the records and each reported the other's reads as drops.

### 2. Claims add — `axon_strategy::TargetBook`

**The venue holds one position per instrument, and a target-position signal is a claim on
part of it.** Two strategies that both trade BTC are not two positions; they are two claims
on one. So the claims *add*, and the account works toward the sum.

Addition is right rather than merely convenient. A target position is self-contained by
construction (ADR-0006) — it says what its author wants held, not how to change what is
held — so summing two of them sums two independent claims rather than composing two
instructions. The naive alternative, "newest signal per symbol wins", is what the pass did
at width one and it is silently wrong at width two: both producers' counters climb, both
believe they are positioned, and the account holds whichever spoke last.

**Netting is opt-in even though it is correct.** Two producers pointed at one instrument is
far more often a copy-pasted config than a decision, and netting them silently composes two
strategies' risk into a position neither author sized. `overlap = "exclusive"` is the
default, `validate` refuses a config whose producer universes overlap without
`overlap = "net"`, and the book refuses the second claim at runtime as a backstop. The
accident is caught at startup; the deliberate case costs one declared word.

**A single contributor passes through byte for byte** — same `seq`, same `ts_event`,
therefore the same `cloid`. That is not an optimization. It means every session that
exists, every capture, and every replay of one plans precisely what it planned before this
ADR, and it is how the change is verifiable rather than merely tested: the committed golden
replay reproduces its 59 rows, four orders and every `cloid` unchanged.

Only a genuinely multi-contributor symbol synthesizes, and then each field is combined by a
rule with a named failure mode: `target_qty` sums (saturating, counted); `ts_event` is the
newest *live* contributor's, and the pass's own clock when every contributor has been
silenced — because a target driven to zero by silence is a decision the runtime made *now*,
and stamping it with a dead producer's last event time would have it refused as expired on
the pass that created it; `urgency` is the highest, because under-executing an urgent exit
is the failure with no bound on it; `reduce_only` and `close` propagate only when *every*
contributor sets them; `ttl_ms` and `max_order_age_ms` take the shortest anyone actually
expressed, zeros filtered first — the same inversion `Planner::order_lifetime_ns`
documents.

**A `price_band` is dropped when the contributors disagree, and the drop is counted.** A
band is a ceiling for a buy and a floor for a sell, and the side of a *netted* order is not
known until the planner subtracts the position — so a band combined from disagreeing claims
would be enforced in whichever direction happened to apply. Dropping it is loud
(`PRICE BAND DROPPED n` on the status line) and leaves the order bounded by the urgency
table and the per-symbol gate; combining it silently would not be.

### 3. A synthesized `seq` lives in its own space

`cloid_for` derives the client id from `(ts_event, seq, symbol)` **and nothing else** — the
fact the handoff records as having cost this project two separate incidents. A netted
target therefore needs a sequence that can never collide with a producer's, so the
synthesized `seq` carries `NET_SEQ_TAG` in its top bit and a per-symbol counter below it.
The counter advances on **every** synthesis rather than only on a change, because a
re-planned target that reused its predecessor's `seq` would reuse its `cloid`, and the
venue de-duplicates a repeated `cloid` into nothing while every counter reports success.
Churn is not the cost it looks like: the planner compares a resting order field by field
and never by id, so an unchanged target still resolves to `AlreadyWorking`.

One consequence found by the golden replay rather than by argument, and worth recording
because the first version of this ADR's code got it wrong: **a lone `FLAG_CLOSE` must pass
through.** A close contributes zero *size* to a sum, which is right when there are other
claims to sum it with, and is not a statement about the record — `Planner::plan` does not
consult `target_qty` on a close at all. Without the exemption a single-producer session
synthesized its own flatten, and the order was identical in side, size, price and TIF under
a **different `cloid`**: the one field the venue keys idempotency on.

### 4. Bounds that only exist across symbols — `axon_risk::portfolio`

Three quantities, none expressible one instrument at a time:

| | what it bounds | why it is not the others |
|---|---|---|
| `max_gross_notional` | `Σ \|qty_i\| · mark_i` | what the account has at risk if every leg is wrong at once |
| `max_net_notional` | `\|Σ qty_i · mark_i\|` | directional exposure — the whole point of running strategies that disagree |
| `max_symbols` | how many instruments carry exposure | twenty $5 positions and one $100 position are the same gross and are not the same operational problem |

**A portfolio bound never refuses an order that reduces its own leg.** This is the third
time this project has learned the same lesson — ADR-0031 for cancels, ADR-0037 for the loss
switch — and it is sharpest here, because a portfolio bound is the limit most likely to be
*already breached* at the moment somebody needs to get out. A consequence is stated rather
than discovered: `|net|` can grow past its bound through de-risking alone (long 100, short
40 nets 60; reduce the short to 10 and the net is 90, on an order that took exposure off).
The bound binds on **new** exposure and never on an exit, and that is the honest
description of what it guarantees.

**An unpriced book is refused, not assumed.** `PortfolioExposure::gross` returns `None`
when any non-flat leg has no mark, and the gate turns that into a refusal for anything that
could add exposure — the same fail-closed shape `GuardedClient` already applies to a
missing mark on one order. `RiskContext::portfolio()` returns `Option`, and the `None` is
load-bearing: an empty book says "nothing is held" and satisfies every bound, while `None`
says "I cannot see what is held" and satisfies none of them.

Enforced in `GuardedClient`, as a type, cumulatively across a batch — because a batch that
opens five instruments at 30 % of the cap each is five orders that individually fit and
together do not.

### 5. The pass allocates before it plans

The gate refuses; the **allocator** makes the target reachable. Without one, a binding
bound presents as orders that keep failing on every pass rather than as a limit that is
working, and the position sits permanently short of a target it can never reach — the same
pathology ADR-0036 named as `UNQUOTED TARGET`, arriving from the risk side.

Three stages, and the order between them is the design:

1. **Each producer against its own allocation.** A strategy that has decided to be enormous
   crowds out *its own* future claims rather than the others' share. Scaling only at the
   portfolio level is proportional, and proportional is wrong here: one runaway producer
   would shrink three well-behaved ones by the same factor it shrank itself.
2. **The netted book against the portfolio bound**, applied to what stage 1 left. Both
   ceilings scale linearly (`|Σ k·q·m| = k·|Σ q·m|`), so one factor can satisfy either and
   the binding one is the smaller.
3. **Breadth**, which no factor can express: a cap on how many instruments may carry
   exposure is satisfied by not opening one, never by opening it smaller. Symbols already
   holding a position are never denied (that would refuse the order that closes them);
   candidates are admitted largest-notional-first, which is deterministic — a replay must
   admit the same instruments — and defensible, because a breadth cap should keep the
   positions the strategies care most about rather than the ones whose symbol id is low.

**An unpriced book is not scaled at all**, which is the opposite of the gate's rule and
deliberate. A gross computed over the legs that happen to be priced is a smaller, entirely
plausible number, so scaling on it would quietly shrink every position in the account
because one feed went quiet. The gate still refuses new exposure on the same book, so the
session stops adding rather than mis-sizing; doing nothing here is what leaves the existing
positions alone. `ALLOCATION UNPRICED n` says so.

Scaling is the one place in this project where a quantity is **rounded rather than
refused** (`axon_strategy::scale_fixed`), and toward zero specifically: away from zero would
let a scaled target exceed the very allocation being applied, so the bound would be the one
thing the arithmetic could breach.

## Consequences

**Plus.**

- A session can run several strategies over several instruments, and their claims compose
  by an operation with a stated meaning instead of by whoever spoke last.
- Three bounds exist that no per-symbol limit could express, enforced structurally.
- A binding bound *converges* rather than failing, and says which of the three it was.
- The single-producer path is unchanged, proven by the golden replay reproducing every
  `cloid` byte for byte rather than by inspection.
- The status line grew a fleet view — `strat 2 net 1 silent 1h/0f alloc 5000bp` — and five
  warnings: `SIGNAL RING DETACHED n/m (names)`, `STRATEGY SILENT name 312s`,
  `OVERLAP REFUSED n`, `PORTFOLIO SCALED n%`, `PRICE BAND DROPPED n`.

**Minus, and each is a real cost.**

- **A silent producer's exposure is held by default.** `on_silence = "hold"` is the
  behaviour every session before this had, because a target position is idempotent — but
  the account then holds exposure nobody is currently deciding about. `"flat"` is the
  alternative and it is a *trading* decision: it flattens that strategy's share into
  whatever market it went silent in. The default is `hold`, the silence is named on the
  status line, and neither option is safe in general.
- **`intent.min_order_qty` is one number for every instrument.** On a multi-instrument
  session that is $10 of BTC and $0.0003 of ETH. It errs harmless on the book this ADR
  ships a config for and it is the same shape as `SizeGrid`'s missing `min_qty`
  (ADR-0023): a per-instrument dust floor belongs with that, and does not exist.
- **The loss switch is still per session.** It was never per strategy and this does not
  make it one. A book where one producer is losing and another is winning trips on the sum,
  and nothing here attributes a loss to a producer.
- **One order carries one `urgency`.** The most impatient contributor decides how the whole
  netted delta is worked, so one strategy can put the book across the spread.
- **`IntentLine` is no longer `Copy`.** A producer's *name* is a `String`, and the
  alternative was a fixed-size id an operator has to map back to a config entry at 03:00.
- **A netted `cloid` cannot be traced to one record.** `Intent::seq` on a netted target is
  the book's counter, not a producer's sequence. The contributors are in the capture; the
  venue action names only the sum.

## What this does **not** claim

**None of it has run at a venue.** Every branch is tested offline — 90 new tests across
five crates and Python, the golden replay reproducing byte for byte, and a session config
that parses, validates and round-trips — and no order has been placed by two producers at
once. The list of what is therefore unobserved is the useful part:

- No two producers have shared an account at a venue. The netting arithmetic has never
  moved a real position, and `overlap = "net"` has never been used by a running session.
- No portfolio bound has refused a real order, and no allocation has scaled one. The
  `GuardedClient` path is exercised by unit tests against a spy client only.
- **The measured evidence says the bound the shipped config sets would not bind.** See §6 —
  on a book of `baseline` and `zoo_xgboost` over BTC/ETH/SOL at the live sizes, the gross
  peaked at **$63.02** over 16 000 windows and a cap of 70 binds on none of them. That is
  the intended shape for a first multi-producer session (a backstop, not a size), and it
  means the *scaling* path has been exercised by tests and by no measured window.
- A silence policy has never fired, in either mode.
- The Python `PortfolioRunner` has never read a live bar ring. Its tests drive real signal
  rings in `tmp_path` and canned bars.

## §6 — the evidence the bounds are argued from

`python -m axon.strategies.portfolio_evidence` measures what a book of several
(strategy, coin) legs would actually have held: for every subset of the legs, over every
non-overlapping window of every session length in the cached m1 corpus, the gross, the net,
the breadth, and — for a grid of candidate `max_gross_notional` values — how often that
bound would have bound and by how much.

The design of the fan-out is stated because getting it backwards is the standing mistake
here: one task is **one leg subset at one fee tier**, and every window is an inner loop,
because the expensive step (replaying a strategy over the corpus, one `predict` per bar) is
a property of the *leg*. A task that enumerated windows would repeat it once per window.

It is planned and priced through `hwsched` exactly as ADR-0017 requires — 100 tasks, CPU,
25 containers, one wave, **`est_cost_usd` 0.045 / high 0.063, budget APPROVE** — and the
same entrypoint runs locally across every core, which is what produced the numbers below.
That is not a rejection of the offload: the grid *fits* today because the zoo has one
fitted family, and the enumeration is combinatorial in families. At five families over
three coins it is 15 legs and 1 925 subsets, which is 3 850 tasks and the run that needs a
fleet.

**The run, 2026-07-27.** 100 books (every subset of size 2–4 of the six legs `baseline` and
`zoo_xgboost@1` form over BTC/ETH/SOL), **16 000 windows**, 0 unmeasurable, at the sizes
`scripts/sessions/portfolio-testnet-m1.toml` actually runs — 0.0003 BTC / 0.006 ETH /
0.06 SOL, about $19–20 a leg:

| | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| gross notional | 35.20 | 54.88 | 61.39 | **63.02** |
| net notional | 35.15 | 54.82 | 61.38 | 63.02 |
| instruments open | 3.0 | 4.0 | — | 4 |

| candidate `max_gross_notional` | windows it binds in | bars it binds on | worst scale |
|---:|---:|---:|---:|
| 10 | 98.0 % | 87.9 % | 0.159 |
| 20 | 90.0 % | 63.8 % | 0.317 |
| 30 | 68.0 % | 42.3 % | 0.476 |
| 40 | 34.3 % | 14.1 % | 0.635 |
| **60** | **2.0 %** | **0.6 %** | 0.952 |
| 80 | 0 % | 0 % | 1.000 |

That table is what a bound is chosen from, and the shape is the point: **the difference
between a limit and a position size nobody chose is a column, not an opinion.** A cap of 30
would be binding on 42 % of bars — at which point it is not a risk control, it is the
strategies' sizing, applied by the wrong object and reported nowhere the strategy author
would look. The shipped config sets **70**, between the last cap that binds on anything
(60, at 0.6 % of bars) and the first that binds on nothing: a backstop against a producer
that has gone wrong rather than a size chosen indirectly.

**And the netting finding, which is mostly negative.** The net/gross ratio's **median is
1.000** — on half the windows these legs did not offset *at all* — with a **minimum of
0.524**. BTC, ETH and SOL move together and two of the three legs run the same rule, so the
disagreement that would make a net bound worth having is present in a minority of windows
and absent in most. Since a bound has to hold in the worst case and the worst case is
1.000, `max_net_notional` below `max_gross_notional` is not justifiable on any book this
repo can populate today, and `PortfolioLimits::validate` refuses one *above* it as a
decoration. The config sets them equal and says why. The number that should come down is
waiting on model families that genuinely disagree — which is a fitting job, not a config
change.

**Read what the measurement refuses to claim**, which is written into the module: it models
no fill, no queue and no book, so it reports what would have been *held* — a risk quantity
— and not what would have been earned. It is not an edge claim and `zoo_xgboost`'s own
verdict is unchanged.
