# ADR-0036 — Watching a live session: the money, the clock, and the target nobody was working toward

**Status:** Accepted · **Date:** 2026-07-27

## Context

Phase 6's last open box asks for three things to be watched in a live session: **P&L, parity, and
latency budgets**. The roadmap's own accounting of that box was blunt about which of the three had
actually been done:

> *Parity*: the monitor has watched a venue — 55 minutes of testnet, three windows,
> `max_abs_diff = 0.000e+00`. *Latency*: measured as a by-product rather than budgeted — 53 bars
> arrived 1 067/9 406/62 074 ms after their own close; there is still no **budget**, only an
> observation. *P&L*: **not monitored at all.** The session that produced those numbers was
> read-only by construction, so it had no position and no P&L to monitor.

Three distinct problems sit behind that, and only the first is the one the box names.

1. **Nothing in the runtime computed a P&L.** Every number a session printed was about machinery —
   records accepted, orders placed, quotes swept. The pieces were all present (`Position` carries
   `realized_pnl`, `Fill` carries `fee` and `closed_pnl`, `MarkCache` has prices, the reconcile
   poll reads `accountValue`) and nothing composed them.
2. **A measurement is not a budget.** An observation says what happened. A budget says what was
   supposed to happen, so a session can report how often it did not — and a number nobody declared
   a ceiling for cannot regress, because there is nothing for it to regress against.
3. **A live session could not be left alone, and the reason was known and unfixed.** The first
   sweeper firing at a venue ([ADR-0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md), 2026-07-26)
   pulled two *exit* orders; a target position is idempotent so the strategy correctly re-emitted
   nothing; and **a short sat open with no working order for about twelve minutes**. The status
   line reported the position accurately the whole time and had no way to say that nothing was
   working toward it.

## Decision

### 1. The money view reports two independent answers and never reconciles them

`crates/axon-runtime/src/pnl.rs` composes one `PnlSnapshot` per status line:

- **ours** — realized P&L from our own average-cost book, fees from the fills we applied, and
  mark-to-market from `MarkCache`;
- **the venue's** — `accountValue` now, minus `accountValue` when the session started.

They are printed side by side and the difference is reported as `drift`. Merging them would destroy
the only cross-check a live session has, and they are *expected* to differ for reasons that are
features of the venue rather than bugs here: **funding** moves `accountValue` and is not a fill;
**anything else on the account** moves it too (this project's testnet key is shared and carries 22
inherited fills); and the venue's own unrealized uses its mark, not ours. So `drift` is a
**reported quantity, never a correction** — the same shape, and the same argument, as
`reconcile::PositionDrift`.

### 2. It refuses to price an unrealized P&L off a stale mark

`MarkCache::get` returns `None` past `mark_max_age_ms`. A symbol with an open position and no fresh
price makes `unrealized` — and therefore the whole bottom line — `None`, and puts the symbol's
**name** on the status line as `POSITION UNPRICED BTC`.

Contributing zero is the tempting shape and it is the dangerous one: an instrument whose feed died
stops moving the P&L, so a position going badly wrong reads as a position going nowhere. The
refusal composes with the loss alarm below it, and the ordering is deliberate — while anything is
unpriced the loss alarm *cannot* fire, so the unpriced warning outranks it.

### 3. The alarms warn; they never halt

`[pnl] max_session_loss` and `[pnl] equity_drift_alarm` are magnitudes in quote units, and both
default to `0` = **no bound declared**. A breach raises a status-line warning and changes nothing
about what the session will place.

Monitoring and risk are separate decisions with separate evidence. A loss limit that pulls the
trading switch is a risk control, belongs with the other gates in `axon-execution`, and needs its
own argument — not least because the first thing it would do is refuse the orders that *reduce* a
position, which is the same trap ADR-0031 identified for risk-gating a cancel.

They are magnitudes rather than percentages because the account this project runs against is
shared and its balance is not a scale anyone chose. `validate()` refuses a negative one: it is the
typo that fails in the dangerous direction, because the net figure it is compared against is itself
usually negative, so `-5` written where `5` was meant is an alarm that is on from the first tick
and therefore ignored by the time it means something.

### 4. Three latency stages, each a span between two stamps that already existed

| Stage | From | To | Clock |
|---|---|---|---|
| `sig` | the strategy's decision (`Signal::ts_event`) | the pass that planned it | the core's **event** clock |
| `ack` | the submit call | the venue's ack | wall |
| `e2e` | the strategy's decision | the venue's ack | wall |

`sig` is the one that already decides something: it is the same quantity `SignalReader` ages a
record against, so breaches here are the population `expired` is drawn from. `ack` and `e2e` are
wall clock, and that is a **named exception** in the same class as the dead-man's-switch deadline —
neither orders anything. `e2e` is not the sum of the other two; the gap between them is the queue,
the pass schedule and the in-flight rule, which is why all three are kept.

**An undeclared stage is still measured.** A book that went dark without a config would make
declaring a ceiling require guessing one first, and the first thing anyone setting a budget needs
is what the session actually does.

**A quantile here is an upper bound and says so.** Samples land in fixed logarithmic buckets, so
`p50`/`p99` are reported as the upper edge of the containing bucket (`p50_le_ms`). The maximum is
exact. This repo's rule is to assert the number rather than the bound; a bucketed quantile cannot
honour that, so it does not claim to. What it buys is a fixed-size, allocation-free, lock-free
histogram the core thread and the async edge can both write to.

The warning fires on a **rate** (`breach_warn_pct`), not on a count: a session long enough to be
worth watching breaches something eventually, and a warning that fires on the first late order is
permanently on by the second hour.

### 5. The pass re-quotes a target the sweeper pulled — a bounded number of times, and then says so

This amends [ADR-0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md), whose section on the
sweeper is the claim it complicates.

`IntentSource` retains the newest target per symbol. On the sweep's cadence, after the sweep, a
symbol is re-quoted when **all five** hold: no signal spoke for it in this pass; nothing is in
flight for it; it has **no working order**; the planner still produces an order against the
retained target; and the re-quote budget is not spent. The record is re-stamped with the pass's
event time and keeps its original `seq`.

Five things about that are decisions rather than plumbing:

- **The re-stamp is not cosmetic.** `cloid_for` derives the client id from
  `(ts_event, seq, symbol)`, so re-planning the original record byte for byte would mint the id of
  the order that was just cancelled — the venue would de-duplicate it into nothing while every
  counter here claimed success. Keeping `seq` is what lets a venue action be traced back to the
  record that set the target, *and* is what makes the new id provably distinct from any producer's:
  a record carrying that seq again is refused as `stale_seq` before it could ever be planned.
- **The no-working-order condition is the doubling guard.** Immediately after a sweep the swept
  order is still working — a cancel is not a removal until the venue says so — which is what makes
  the re-quote land a tick later by construction rather than by a delay anyone tuned.
- **The budget is what keeps the repair from undoing the sweeper.** The sweeper's whole subject is
  a producer that has *stopped*; an unbounded re-quote hands a dead strategy an immortal quote.
  After `intent.max_requotes` (default **3**) the session stops placing and starts *saying*
  `UNQUOTED TARGET` — which is the one thing the live run could not do. A budget resets when the
  strategy speaks again, because a producer that is talking is the evidence the budget waits for.
- **Not while halted.** The pipeline refuses a placement anyway and structurally, so this is not
  the protection; it is the difference between spending a target's whole budget on a halt that is
  about to clear and still having one when it does. A halt is also a *cause* of the unquoted state,
  because the shutdown sweep cancels everything at once.
- **The default is not zero.** A defect fixed only for operators who read the release notes is not
  fixed. `0` remains available and reproduces the old behaviour exactly — except that the session
  still *reports* `UNQUOTED TARGET`, because off means placing nothing, not seeing nothing.

**The two alternatives are rejected on their merits.** Teaching the sweeper to spare an order whose
symbol is not yet at target guts it — a quote for an unreached target is precisely the quote that
goes stale. Requiring producers to re-assert on a timer makes every strategy responsible for a
runtime invariant, and the first strategy written by somebody who had not read this record would
sit unquoted again.

### 6. Four counters that were counted and reported nowhere now reach the status line

`no_order`, `ahead_of_clock`, `swept` and `resweeps`, plus the new
`requotes`/`unquoted`/`stranded`.
`intent.rs` already stated the rule — *counted and not reported is the same as not counted* — and
applied it to `busy` and `stalled` but not to these. `swept` is a number that moves in production:
the earlier no-model live run had to read its two sweeps off the venue rather than off its own
status line.

## Consequences

**+ Phase 6's last box is closed on testnet.** A session that trades now reports what it did to
the account, how it fared against ceilings somebody declared in advance, and whether anything it
holds is unquoted or unpriced. Measured on 2026-07-27 over 59 minutes with `zoo_xgboost@1` trading
BTC on Hyperliquid testnet — 12 orders, 8 cancels, 9 fills, 6 sweeps, 4 re-quotes, account flat at
both ends.

**+ The two accountings agreed, and that is the result worth quoting.** Our realized P&L — average
cost over the fills we applied — came to **−0.0013**, and the venue's own `closedPnl` summed over
the same nine fills came to **−0.0013**. Different accounting, different machines, nothing
reconciling them. `drift` against `accountValue` peaked at **0.0137 USDC** across 237 status lines
and closed at **+0.0001**. A cross-check that never fires is only evidence if it *could* have; this
one had a 22-fill offset in it four hours earlier and reported it loudly.

**+ The rate-based latency warning earned its shape in one run.** A single 1 024 ms round trip
breached a 1 000 ms ceiling. The warning lit at `1/1`, stayed lit at `1/2`, and **cleared itself**
at `1/5` when the breach rate fell under the declared 25 %. On a count it would have stayed lit for
the remaining fifty minutes over one late order.

**+ The re-quote fired four times at a venue on its first run**, each time after a sweep left a
target with nothing working toward it.

**+ …and an hour later a real venue outage exercised every remaining branch at once.** A second run
on ETH was cut off mid-position when Hyperliquid testnet went down for a network upgrade — 502s on
`/exchange`, `/info` and the WebSocket. Three things this record argued for happened without anyone
staging them:

- **The money view refused to price.** The mark expired, the bottom line went to `-` rather than to
  a number, `drift` disappeared with it, and the line read `POSITION UNPRICED ETH` beside an open
  short. §2's argument — that contributing zero makes a position going wrong read as a position
  going nowhere — stopped being an argument.
- **The re-quote's *bound* is what mattered, not the re-quote.** Post-upgrade the venue accepted
  **post-only orders exclusively**, so the flatten's IOC was refused (`order rejected: Only
  post-only orders allowed immediately after network upgrade`). The pass re-quoted three times,
  each refused, then stopped and raised **`UNQUOTED TARGET 1` for eight consecutive status lines**
  until an operator acted. The terminal branch this ADR could only claim offline is now observed at
  a venue, and an unbounded re-quote would have spent the session hammering a venue that was
  refusing every taker order it sent.
- **The dead-man's switch ran out of protection for the first time in this project's history**,
  halting new orders at the second failed re-arm and shutting the session down at the third — with
  a closing sweep that *failed* and said so (`swept: false, disarmed: false`).

**− One string in the runtime overclaims, and the outage proved it.** The unprotected path logs
"the venue-side switch **has fired**". It has not observed that; it has observed its own deadline
passing. When the venue returned, the order placed before the outage was still resting. The honest
wording is "the deadline has passed", and it is left as a finding rather than patched in the same
session that found it.

**+ The first live run of the money view found a defect in it within fifteen seconds, and the
defect was the interesting kind.** The session opened — with no order placed, on a flat account —
reporting `r +0.0161 fee 0.1031` over 22 fills and a drift of `-0.0869` against an equity that had
not moved. Nothing was wrong with either number: Hyperliquid's `userFills` replays a snapshot of
recent fills on every subscribe, and the tracker applies them, which is exactly what lets a
restarted process know its own position. The **subtraction** was wrong. `accountValue` already
contains those trades, so a lifetime-ish P&L compared against a session-scoped equity delta carries
a permanent offset — and a cross-check with a constant offset in it cannot detect anything new,
which is the only thing it is for.

**The fix that failed first is worth recording, because it is the obvious one.** The baseline was
taken at the first `clearinghouseState` reply, on the theory that the two halves should be marked
at one instant. The `userFills` snapshot arrived *after* it, so every replayed fill still counted
as the session's and the status line was unchanged. Arrival order is not a fact anyone controls.
The split is now on each fill's **own execution time** (`OrderTracker::set_session_start`): a trade
that happened before this process started cannot be this process's work, whatever order the frames
arrive in. `0` — the default, and what a backtest and a replay get — counts every fill, because a
canned log has no earlier session to inherit from.

**+ The re-quote's first live run found the case it does *not* cover, within half an hour.** The
model's second target change produced a closing buy for `0.0003` BTC; it partially filled `0.00017`
at a maker price; the sweeper pulled the remainder sixty seconds later, exactly as designed; and
`0.00013` BTC — about **$8.50** — was left with no working order and no order that could ever
express it, because it is under the venue's own **$10** minimum notional and under this session's
`min_order_qty` floor. The re-quote was **right** to place nothing, and the status line had no word
for what was left.

`unquoted` is the wrong word for it: nothing is *unquoted*, it is unquotable, and no number of
re-quotes fixes it — only an operator can. So it has its own counter (`stranded`) and its own
warning (`STRANDED POSITION`), keyed off the `NoOrder` reason the planner already returns, which is
the only thing that can tell "already at target" from "the delta is too small to express". Both
answers arrive as an empty plan.

**− One case the split does not get right, and it is inherent.** A session that inherits an open
position and closes it books the whole realized P&L, including the part that accrued before it
started, while `accountValue` had already moved with the mark. The account is left flat between
sessions precisely so this does not arise; when it does, `drift` is what reports it.

**− The money view takes its own read of the tracker lock**, separate from the one that counts open
orders on the same line. A deliberate and bounded inconsistency: the status line is not a decision,
and holding the lock across both would put the P&L assembly inside the critical section the submit
path contends for.

**− `unrealized` is per configured universe while `realized` and `fees` are process-wide.** A
position on a coin this session never configured is invisible to the mark-to-market and cannot even
appear in `unpriced` — a session can only price what it subscribed to. `ReconcileReport::incomplete`
is the existing flag for that class of blindness.

**− A latency budget is a number somebody chose, and only one session's worth of evidence stands
behind the numbers in `live-ml-testnet-m1.toml`.** `signal_age_ms = 20000` is argued from a
strategy whose opinion is worth one m1 bar; `submit_ack_ms = 1000` from the venue's own published
p99. Both are declarations, not measurements, and the point of declaring them is that the session
now reports how wrong they were.

**− The re-quote is new behaviour on a live trading path and it is on by default.** It can only ever
re-express a target the operator's strategy already asked for, at a fresh price, at most three
times, never while halted, and never on top of a working order — but it is the first mechanism in
this runtime that places an order no signal asked for in that pass, and that is worth stating
plainly rather than burying in a config default.

## Alternatives considered

**Put the fee accumulator in a P&L component of its own.** It would need its own dedup set: a
reconnect replays fills, and a fee total that double-counts one is wrong in the direction that
makes a losing session look profitable. The tracker is where the `trade_id` dedup already lives, so
the money is accumulated behind it.

**Exclude orphan fills from the money.** Rejected, and it is the more tempting of the two errors: an
orphan fill already moves the *position*, so excluding its cost would account the two halves of one
fill on opposite sides of a filter — a session on a shared account would report a position it did
not ask for and a P&L that excludes what that position cost, which reads as free exposure.

**Propagate the venue's `isSnapshot` flag through `Fill`.** It is already decoded
(`ws::user::fills_is_snapshot`) and used informationally. Threading it into the normalized
execution vocabulary would put a venue's subscription semantics into a contract that is serialized
into every capture, to answer a question the fill's own timestamp answers without a contract change.

**Exact quantiles.** A reservoir or a sorted buffer would honour "assert the number, not the bound"
— at the cost of an allocation or a lock on a path two threads write. The maximum is exact, which
is the figure a latency run is actually quoted on; the quantiles are bounds and are named as bounds.
