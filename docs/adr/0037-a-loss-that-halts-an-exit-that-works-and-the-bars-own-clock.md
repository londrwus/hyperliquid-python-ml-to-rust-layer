# ADR-0037 — A loss that halts, an exit that works, and the bar's own clock

**Status:** Accepted · **Date:** 2026-07-27

## Context

[ADR-0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md) closed
Phase 6 on testnet: an ML model drove orders at a venue and the session could say what
they cost. It closed that box and left five things open, and a venue outage the same
afternoon turned four of them from "worth doing" into "the exit path failed during the one
real incident of the day":

1. **No loss-based kill switch.** Every risk limit was size-only — `max_position`,
   `max_notional`, `max_order_qty`. All three answer *how big may this get* and none of
   them answers *how much may this cost*, so a strategy that is quietly wrong stays inside
   every one of them for as long as anyone lets it run. ADR-0036 §3 made the P&L alarm a
   warning on purpose and said why: a loss limit that pulls the trading switch is a risk
   control, needs its own argument, and *the first thing it would do is refuse the orders
   that reduce a position*.
2. **An inherited position could not be flattened by the documented path.** `reconcile`
   reports position drift and never writes it; `--flatten-only` emits a target of zero,
   which the planner subtracts from the **tracked** position; a tracker that has not yet
   heard the `userFills` replay is flat, so the delta is zero and the flatten is a no-op
   against the one position it was run for. The operator who worked around that
   hand-wrote a target and turned a −0.01 short into a +0.01 long with one order.
3. **The flatten died when the venue said no.** Post-upgrade Hyperliquid accepted
   post-only orders *exclusively*; every `--flatten-urgency take` IOC was refused with
   `Only post-only orders allowed immediately after network upgrade`.
4. **Stranded residues.** A partial fill left `0.00013` BTC — about $8.50 — that nothing
   could close. Named (`STRANDED POSITION`) in the same session that found it, and not
   handled.
5. **One string in the runtime overclaimed.** The dead-man's switch logged "the venue-side
   switch **has fired**" when it had observed only its own deadline passing. When the
   venue returned, the order placed before the outage was still resting.

And two things Phase 6 could not do at all: watch parity on a session that was **trading**
(an SPSC bar ring has one consumer and the strategy is it), and put a ceiling on the
largest latency in the system (the bar's close was not on the signal wire).

## Decision

### 1. The loss switch puts the session into de-risk-only, and that is not a halt

`axon_execution::LossLimiter` is a new gate inside `GuardedClient` — not a fourth wrapper
in the submit pipeline, and not `HaltSwitch`. The distinction is the whole design.

`HaltSwitch::halt` refuses **every** placement, which is right for the two situations it
was built for: a lapsing dead-man's switch and a shutdown sweep. In both, the correct next
action is a *cancel*, and cancels pass through. It is wrong here. A session that has lost
more than its operator declared does not want to stop acting; it wants to **get out**, and
getting out is an order. Halting it strands the exposure that caused the loss in the market
that is causing it — the same trap [ADR-0031](0031-order-lifetime-and-the-sweeper-on-the-pass.md)
identified for risk-gating a cancel, and strictly worse: a refused cancel merely fails to
reduce risk, and this would actively hold it.

So a tripped limit admits an order **iff it strictly takes exposure off the position we
hold**. That predicate is `axon_risk::reduces_exposure`, which is the arithmetic
`RiskEngine` already applies to `reduce_only` — one definition, two callers, because two
copies of it would be two answers to "may this order go out while we are trying to get
smaller" and the copy that drifted would be the one enforcing the kill switch.

Four properties, each a way this could have been silently wrong:

- **It reads the position we hold, not the one risk projects.**
  `OrderTracker::risk_position` inflates the position by every resting order, which is
  exactly right for a size cap and exactly wrong here: flat with a resting buy projects
  *long*, so a sell would read as a reduction and would in fact open a short — the switch
  admitting new exposure on the session it was tripped to stop. `RiskContext` grew
  `held_position` for this, defaulting to `position` so no other context changed.
- **A `reduce_only` flag does not get past it.** The flag is a request to the venue;
  `reduces_exposure` is arithmetic about the position. They disagree exactly where it
  matters — a reduce-only order against a flat book reduces nothing and the venue drops it
  — and a session that trusted the flag would be one whose kill switch can be talked past
  by setting a bit.
- **A batch is measured as a unit.** Two sells of 1 against a long of 1 each "reduce" the
  same long if the held position is not carried forward, and together they leave a short.
  The same reasoning `check_batch` already applied to the size caps.
- **It is one-way.** Nothing un-trips it but a restart. A bound measured partly on
  unrealized P&L is measured on a number that moves both ways, so a re-arming limit would
  let a session resume on a bounce and stop again on the next tick — flickering in and out
  of the market at the moment nobody is watching, fastest when volatility is highest. The
  same reasoning makes `HaltState::Stopped` terminal.

**It judges two independent numbers, and the second one survives a restart.** The session
bound is our average-cost accounting; the day bound is the venue's `accountValue` against a
baseline `crates/axon-runtime/src/daybook.rs` persists per UTC day. They are the pair
ADR-0036 reports side by side and refuses to reconcile, used here for the reason the pair
exists: **a crash-restart loop resets our accounting and cannot reset the venue's**, so a
session bound alone is spent once per restart, and a crash-restart loop is precisely how a
losing session restarts. `validate()` refuses `max_daily_loss` with no
`daily_state_path` — a daily bound whose baseline dies with the process is a session bound
wearing a day's name, and the degradation is invisible from the status line.

**The session bound falls back to realized-less-fees when a mark is missing, and that
closes a hole.** ADR-0036 §2 makes `net()` `None` the moment any held symbol goes unpriced,
and the loss *warning* is silent for as long as that lasts — defensible for a warning
beside a louder `POSITION UNPRICED`, and not defensible for a kill switch, because it would
make a dead feed a way to switch the bound off on exactly the session that has a position
and cannot see it. What has actually been closed and paid needs no mark and cannot be wrong
in either direction.

**Where it is judged: the status-line pass, once per `status_interval_ms`.** The lag is
real, bounded, and the same lag every other number on that line already has. Recomputing a
mark-to-market on every event would put a walk of the position book on the hot path to
bound a quantity that moves on the timescale of a fill.

### 2. `axon --flatten` adopts the venue's position, and every order it sends is a close

A new operator entry point (`crates/axon-runtime/src/flatten.rs`). Not a session: no
market-data socket, no dead-man's switch, no signal ring, no capture. It reads `/info`,
places closes through a bare signing client, and reads `/info` again.

Three properties, each answering one step of the 2026-07-27 failure:

- **Every order is sized from a fresh venue read, never from an operator's number.** The
  position is re-read before each attempt, so a partial fill shrinks the next order rather
  than being added to it. There is no arithmetic here an operator can get wrong, because
  there is no number here an operator supplies.
- **Every order is a `FLAG_CLOSE` plan, so it cannot flip.** The planner ignores the
  record's quantity entirely for a close, takes the size from the position, and sets
  `reduce_only`. The overshoot that turned a −0.01 short into a +0.01 long is not made
  unlikely by this; it is made unrepresentable.
- **The urgency is a ladder, not a setting**: `[3, 2, 0]` — an IOC through the far touch,
  then a crossing GTC, then a post-only quote. A refusal escalates *down* the aggression
  table rather than giving up, because the venue that refuses an IOC is exactly the venue
  an operator cannot wait for. Rung 1 is deliberately absent: it is less likely to fill
  than 2 and less likely to be accepted than 0, so it would spend a rung's metered
  requests to be dominated on both axes.

Three deliberate absences. **No `GuardedClient`**: the size caps are about *adding*
exposure, and every order here is reduce-only against a size the venue reported — putting
the gate in would reproduce ADR-0031's trap, because an unpriced instrument or a breached
limit is a reason to want a position *gone*. **No dead-man's switch**: arming one costs a
signed action and 60 s of lead to protect a pass that is over in seconds, and the whole
point is that it runs when a session's protection has already lapsed; what it does instead
is never leave a marketable order working, and try the resting rung last. **No
`cancel_all`**: on Hyperliquid that is account-wide, so sweeping first would pull the
resting orders of any session still running.

`FlattenReport::flat` is a **venue read taken after the last attempt**, never a claim about
what was sent, and the CLI exits non-zero when any symbol is not flat *or* when the venue
could not be read. An unknown position is never reported as flat.

`OrderTracker::adopt_position` is the write, and it refuses three things: it does not touch
`realized_pnl`, the fees or the fill counts (a position snapshot is not a trade, and
synthesizing the difference as a `Fill` would book a P&L nobody earned on the side that
makes a losing session look better); it does not invent an `avg_px`; and it does not clear
resting orders or the equity baselines. It stays off the reconcile path — the argument for
never writing drift is unchanged, and this is an operator action.

### 3. The dust floor stops refusing a close, which is what stranded the residue

`min_order_qty` is a **churn** bound: it exists so a target differing from the position by
a rounding residue does not re-send an order on every signal. An order that takes the
position to exactly flat cannot be that — it converges by construction, there is at most
one of it, the leave-it-alone rule keeps it if it rests, and the re-quote budget bounds how
often it comes back. Applying the floor to it instead produced a position **nothing could
ever close**.

So `Planner` exempts an order that flattens exactly from `min_order_qty` *and* from the
venue's `min_notional`. The second exemption already existed for `reduce_only` and the
argument was already written: *the venue's published rule does not say whether it exempts
closes, and guessing in the closed direction strands a position — if we are wrong the venue
tells us, loudly and observably; if we refuse, nobody ever finds out.* What was missing is
that the condition is **arithmetic and not a flag**: the documented flatten emits a plain
target of zero, which carries neither `FLAG_CLOSE` nor `FLAG_REDUCE_ONLY`, so a flag-keyed
exemption missed exactly the residue it was written for.

`IntentStats::stranded` is therefore narrower than the run that produced it. What is left
is a residue below **one lot**, which no limit order can express at any price and which on
a venue whose positions are sums of lot-sized fills should never arise — so a non-zero
count now means a mid-session re-precisioning, which an operator needs to know about and a
re-quote cannot fix.

**This widens a pass-through [ADR-0025](0025-instrument-precision-and-rounding.md) recorded as
unbounded, and the thing that bounds it arrived in between.** That record's minus column says the
reduce-only min-notional exemption *"sends one rejected order per pass, unbounded, for a dust
position the venue may not let us close … Nothing backs it off."* True when written, and the
re-quote budget ([ADR-0036](0036-watching-a-live-session-money-latency-and-the-unquoted-target.md)
§5) is now what does: a close the venue keeps refusing is re-quoted at most `intent.max_requotes`
times and then reported as `UNQUOTED TARGET` rather than re-sent. So the exemption this section
widens is bounded where the one it copies was not — which is worth stating, because widening an
unbounded loop would have been the wrong trade and it is not the trade being made.

It is still loud in the wrong units, exactly as ADR-0025 said: an operator sees refusal lines and a
counter, not a "position you cannot close" alarm. What changed is that the refusals now stop.

### 4. `ts_cause` — schema version 3, and the last of `reserved`

`Signal` grows one field: the event time of the observation a decision answers, an m1 bar's
own **close**. `0` means the producer stated none, and the stage is then not measured.

**It had to be on the wire.** The largest latency in this system is the gap between a bar's
close and the strategy acting on it — measured 951 / 12 051 / **111 475** ms over 57 live
bars — and it lived only in the producer's private transcript. A record carrying only
`ts_event` makes a decision one second after a bar and one two minutes after it *the same
record* from the runtime's side, so the gap could be quoted afterwards and never budgeted.
A number nobody can measure cannot regress. The alternative — inferring the close by
rounding `ts_event` down to the bar interval — is a guess about the producer's cadence,
wrong for every strategy that is not on bars, and silently wrong rather than absent.

The record is now **fully named**: `ts_cause` spends the last eight reserved bytes, so the
next field has to re-cut the 64-byte layout rather than extend it. The version bump is
non-optional and the reason is the same one ADR-0031's bump had: a v2 log's zeroed tail
would decode as "no cause stated" and read as healthy forever.

`latency.rs` gains a fourth stage, `bar`, and it is **additive with `e2e`** — their sum is
bar-close to order-at-the-venue, which is the figure an operator means when they ask how far
behind the market a strategy is. It is a **cross-clock span** and says so: the venue stamps
the cause and the producer stamps the decision, both in epoch nanoseconds, so the difference
includes whatever skew exists between them. It is a separate stage from `sig` because the
two answer to different people: `sig` is ring→pass and is this runtime's to fix, `bar` is
bar→decision and is the producer's.

The producer writes it in `LiveRunner.on_bar`, over every record the bar produced, rather
than in `emit_target` — the runner is the only thing holding both stamps, and asking every
strategy author to remember a runtime invariant is the mistake ADR-0036 §5 names for the
re-quote.

**[ADR-0028](0028-market-data-bars-and-the-ticker-tail.md) predicted this field and warned
against a version of it.** Its closing consequence is about `release_ts` — the recorded stamp is
the core's event-time high-water mark rather than the producer's write time — and it says: *"A
wall-clock stamp in the reserved bytes would put two clocks on one record and make the gap look
closed while the number stayed incomparable. It needs a decision about how the two sides share a
clock, which is an ADR of its own."*

This is that ADR, and the warning is met rather than ignored, on the one distinction that matters:
**`ts_cause` is not a wall clock.** It is the *venue's* stamp for the bar — the same clock the core
orders every event on — copied from the bar the producer is holding. So the pair on the record is
(venue event time, producer wall clock), and the span between them is a real quantity with a named
cross-clock component, not two descriptions of one instant that can disagree.

What it does **not** close is `release_ts`. That gap is about when a record *crossed the ring*, and
Python still has no access to the clock the core orders by, so the sentence ADR-0028 wrote about it
stands unchanged. `ts_cause` answers a different question — *what was this a decision about* — and
it is answerable precisely because the answer is a stamp the venue already made.

### 5. The parity diff becomes a component, so one reader can serve two consumers

`axon.strategies.shadow.BarParityDiff` is extracted from `ShadowTrader`, which still holds
one. `LiveRunner` can now hold one too, under `--parity-diff`.

Two independent things stopped a live parity monitor watching a *trading* session, and
neither is about parity: **an SPSC ring has one consumer** and on a trading session the
strategy is it (two drainers on one ring each saw about half the records and each reported
the other's reads as drops), and **a second session's shutdown sweeps the account**
(`cancel_all` is account-wide). Both are properties of *sessions*. So the diff becomes a
thing a session can own: one reader, dispatching each bar to the strategy and to the diff.

**It is the same object the shadow path runs, and that is load-bearing.** A cut-down
comparison beside a live order flow is how two answers to one question get into a tree, and
they would disagree on exactly the window where it mattered. There is one alignment rule,
one window construction and one `ParityMonitor` configuration between the two callers.

**A parity break raises an alarm and never stops the run.** The process is holding a
position and has a signal ring the Rust core is reading; raising would abandon both with no
flatten. A diff that *itself* fails is caught for the same reason — a watcher that can take
down the thing it watches is a liability rather than a safeguard. What to do about a break
is the operator's call, and `axon --flatten` is how they act on it.

It is **off by default**, because it recomputes the whole spec at every window boundary on
the thread that is about to stamp a decision.

### 6. The DMS says what it observed

`Escalation::Unprotected` logs *"the deadline we armed has passed"* and points the operator
at `openOrders`. It used to say *"the venue-side switch has fired"*, which is an inference
stated as an observation, and the outage proved it wrong: the order placed before the
deadline lapsed was still resting when the venue came back. The line is now produced by
`dms::console_line`, a pure function, so the **wording is asserted** rather than reviewed —
the branch that matters only runs on the worst day, and this is the one an operator acts on
at 3am.

### 7. The loss bounds get a way to be argued from evidence

`axon.strategies.loss_evidence` is the first thing in this project that has genuinely
needed the compute fan-out ADR-0017 built rather than merely fitting it. The numbers now in
the shipped configs — `max_session_loss = 1.00`, `max_daily_loss = 2.00` — are declarations
backed by one 59-minute run that lost 0.028, and a loss bound chosen that way is worse than
a latency budget chosen that way, because it is not a warning: too tight and it halts a
healthy session on an ordinary hour, too loose and it is decoration.

So: 240 non-overlapping session-length windows, one task each, the serving strategy replayed
over every one, accounted the way the runtime's money view accounts. Non-overlapping because
overlapping windows share bars and a quantile over them understates the dispersion by
exactly the amount they overlap — the error that produces a confident-looking bound from one
afternoon of data. `summarize` reports quantiles of the **loss as a magnitude**, because
that is what the config takes; quantiles of the signed figure would put p99 at the *best*
session and a bound copied from it would never fire.

It is not a backtest and says so: it assumes every target change trades at the bar's close,
where the live path posts at the near touch and may be swept. That error is
**conservative** for a bound — a window that looks bad here would look less bad live —
which is the safe direction for a kill switch and the wrong direction for a profit claim.

## Consequences

**+ Nothing in Tier 0 is left open, and each fix is tested against the run that found it.**
1 136 Rust + 663 Python tests green (from 1 085 + 646), `clippy -D warnings` and `rustfmt` clean.

**+ The budget guard refused the first GPU plan and it was right to.** `axon-zoo-gpu-fit`
at 64 points priced at `$39.21` against a `$3.00` cap, and at 8 points at `$4.91`. The
reason is not that the grid was too big: **hwsched has no per-task runtime for a tree fit on
a GPU** — `GPU_TASK_TIME` covers `ml_train_dnn`, `ml_train_rl` and `inference`, and
everything else takes a flat 1800 s. So 64 points were priced as 32 GPU-hours on a number
nobody measured. The response is a **stated scope cut, not a raised cap**: four points,
`$2.48` high, approved — and named a *calibration* run, because its real product is the
measured wall time that would let a wide grid price honestly. Raising `max_usd` past a
refusal is how a budget guard becomes a formality.

**+ Both device classes have an approved plan and neither has spent anything.**
`axon-session-loss`: CPU, `walk_forward`, 240 tasks in 60 containers, ~63 s, est high
**$0.15**, `approve`. `axon-zoo-gpu-fit`: T4×1 pinned, 4 tasks, ~30 m, est high **$2.48**,
`approve`. Dry-run only — the decision to spend is the operator's, and ADR-0017's rule is
that Axon decides *whether the model is good enough* and hwsched decides *where the compute
runs*.

**− A defect ADR-0036 recorded as fixed was not fixed, and it took a fourth reader to see
it.** That record says *a poisoned tracker prints `pnl UNREADABLE`, not zeros*, and calls it
a bug caught in the same session — and `core.rs` still built the snapshot from a fresh,
empty `OrderTracker`, which returns `readable: true` and prints `pnl +0.0000`. The comment
directly above it said what the code must not do, and the code did it.
`PnlSnapshot::unreadable()` existed the whole time with **no production caller**. So a
session holding a position it could not see reported that it had done nothing. The handoff's
instruction — *assume this session's work is wrong in one more way nobody has looked for*
— was correct, and the way it was wrong was the same way it had already been wrong once.

**− The loss switch has never fired at a venue.** Every branch is tested offline, including
the three refusals, and the terminal state has been observed nowhere. That is the same
caveat ADR-0036 had to make about `UNQUOTED TARGET`, which an outage then exercised within
the hour; this record does not get to claim that.

**− `axon --flatten` has never run against a venue either.** Its whole point is the incident
path, and it is exercised against a fake venue that refuses TIFs with Hyperliquid's own
words. The one thing offline testing cannot cover is whether Hyperliquid accepts a
reduce-only close under its own minimum notional — which is precisely the question §3 argues
should be put to the venue rather than answered by refusing.

**− The trading soak is still not run.** The longest session that has ever *traded* is 59
minutes; the 1 h 44 m soak was read-only market data. Everything this record adds is
motivated by an outage that happened inside an hour, and a multi-hour session holding a
position through induced outages remains the largest untested thing in the system. It needs
hours of venue time rather than code.

**− The `bar` ceiling is one number chosen against one run.** `cause_to_decision_ms = 60000`
is argued from an m1 bar's opinion being worth one bar and set against a measured
951/12 051/111 475 ms — so the median is comfortably inside it and the maximum is nearly
double, which is the shape a useful budget has. It is still a declaration, and §7's job is
what would let it stop being one.

**− A window in the loss-evidence fan-out is not a session.** It has no queue, no partial
fill, no book, and no dead-man's switch running out. The distribution it produces bounds
what the *strategy's decisions* cost and says nothing about what an outage costs — and the
one loss this project has actually observed came from an outage.

## Alternatives considered

**Make the loss switch a fourth wrapper in the submit pipeline.** It would need a position
to tell an order that adds exposure from one that removes it, which means it would need a
`RiskContext`, which means it would be `GuardedClient` with a different name.

**Let the loss switch call `halt()`.** One line, and it strands the position that caused
the loss. This is the whole reason ADR-0036 refused to build it in the same session.

**Have `reconcile` adopt position drift when it finds it.** Tempting, and it would have
fixed the flatten as a side effect. Rejected for the reason reconcile already gives about
orders: a view this process corrected itself into agreement is a view that can no longer
disagree, and the disagreement is the finding. The fill replay is the right mechanism
because a fill is evidence and a snapshot is a summary.

**Add a `close_position` action to the provider contract.** Some venues have one and
Hyperliquid does not; a contract method with one implementation and a "not supported" arm
everywhere else is a contract that describes one venue.

**Infer the bar's close from `ts_event`.** Rounding down to the bar interval needs no schema
change and is a guess about the producer's cadence. Wrong for anything not on bars, and
wrong *silently*, which is worse than absent.

**Give `LiveRunner` its own cut-down diff.** Faster to write and it is the thing the Phase-6
runbook explicitly refused: two answers to one question, drifting on exactly the window
where it mattered.

**Raise the GPU job's `max_usd` past the refusal.** It was within the project's own
`per_job_max_usd = 5.00`, so it would have worked. It would also have made the first
interaction with the budget guard an override of it, on an estimate nobody had measured.
