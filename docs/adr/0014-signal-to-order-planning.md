# ADR-0014 — From `Signal` to order intents: validation, planning, and cancel/replace

**Status:** Accepted · **Date:** 2026-07-25

## Context

[06](../06-strategy-contract.md) states the division of labour in one sentence: *the strategy
declares what position it WANTS; the Rust execution engine decides HOW to get there.* Phase 1
built the wire ([ADR-0006](0006-signal-schema-and-spsc-ring.md)), Phase 3 built the execution
side ([ADR-0010](0010-execution-events-and-reconciliation.md)), and the sentence itself had no
implementation. `axon-strategy` was a context object and a trait; nothing turned a `Signal` into
an `OrderRequest`. This ADR closes the Phase-4 roadmap item "Rust strategy adapter: `Signal` →
order intents".

The gap is small in code and large in consequence, because every field on that 64-byte record is
a promise that only means something if somebody enforces it:

1. **Who decides a record is trustworthy?** `schema_version`, `kind` and the 15 reserved bytes
   exist so the two languages cannot drift silently. A reader that skips them makes them
   decorative — the schema's compile-time offset asserts guard *our* struct, not the bytes the
   producer actually wrote.
2. **What does a late signal mean?** `ttl_ms` is on the record. Nothing consumed it.
3. **What is the order?** The target, or the difference between the target and what we hold?
4. **What does `urgency` mean in venue terms**, and what does `price_band` bind?
5. **What happens to the orders the previous signal left working?** Hyperliquid's in-block
   priority (`cancel > post-only > GTC > IOC`, [04](../04-provider-abstraction.md)) makes this a
   real design input rather than a matter of taste.

## Decision

### 1. `SignalReader` validates before a record is allowed to mean anything

Records are checked in a fixed order — layout, then meaning, then ordering, then freshness —
because each check is only trustworthy once the ones before it have passed. `schema_version`
comes first: if the layout drifted, `target_qty` is not necessarily where we think it is, and a
misread target is a position taken for no reason.

Two of the five checks are not obvious:

- **Non-zero reserved bytes are a rejection.** They are the extension slot, so data in them means
  the producer is writing a field this build does not know about *and did not bump
  `schema_version`* — precisely the drift the version field cannot catch on its own.
- **A `seq` gap is counted, not rejected; a `seq` rewind is rejected.** These look symmetric and
  are not. Missing an older target does not make the newest one wrong — surviving loss is the
  entire reason ADR-0006 chose the target-position shape — so refusing the newest record because
  an older one never arrived would convert one dropped signal into an indefinite stall. A rewind
  is the opposite: it walks the target backwards through decisions we already superseded, which
  is what a restarted producer or a replayed ring looks like.

**A stale signal is dropped, counted, and surfaced.** A late target-position signal is not a
weaker opinion about the current market; it is a firm opinion about a market that has already
gone. Acting on it late is how a strategy buys the top of a move it correctly predicted and then
correctly abandoned. Staleness is measured on **event time** from the core clock, never
`SystemTime::now()`, so a replayed session makes the same decisions it made live.

The TTL that is enforced is `min(ttl_ms, ReaderConfig::max_age_ms)`, with `ttl_ms == 0` meaning
"the ceiling". Two failure modes need that ceiling. Zero is the Python default, so a strategy
that never thought about staleness would otherwise be the one with no protection at all; and a
strategy must not be able to raise its own staleness limit above what the operator configured,
because the operator is who answers for the fills. A signal stamped ahead of our clock by more
than a millisecond is **accepted but counted**: refusing it would refuse everything the moment
the producer's clock drifted, and the count is the alert that TTL enforcement has stopped
meaning anything.

### 2. The order is the delta, and the planner is pure

`Planner::plan(&Signal, &PlanContext) -> Plan` takes the signed position we hold, a top of book,
and the orders currently working; it returns cancels, orders, and — when it produces no order —
a typed reason. No clock, no I/O, no async, no `f64`. Wire integers become `Decimal` through
`Decimal::new(v, 8)`, which is a reinterpretation rather than a division, so `0.1` on the wire is
exactly `0.1` in the order. That exactness is load-bearing: with an `f64` round trip, `target ==
current` is never quite true and the planner re-sends dust forever.

Sending the target instead of the delta is the bug this component exists to prevent — "be long 3"
while already long 2 ends up long 5, silently, compounding on every signal until a risk limit
finally catches it.

**Flags.** `FLAG_CLOSE` flattens and ignores `target_qty` entirely, so a strategy can say "get me
out" without also having to be right about what it holds. Closing implies `reduce_only` even when
the strategy did not ask for it: if a fill lands between our position read and the venue's block,
an unqualified flatten overshoots straight into an opposite-side position, which is the one
outcome a flatten must never produce.

**Reduce-only is a projection, not a veto.** The permitted delta is the closed interval between
flat and the position held; a request to *grow* clamps to zero (no order), a request to *flip*
clamps to exactly flat. The alternative — passing the raw delta through and letting `axon-risk`
reject it — is worse than it looks: a risk rejection is indistinguishable in the logs from a
venue outage, and the planner already knows the answer.

### 3. Urgency is an explicit table

| `urgency` | TIF        | price anchor              | what it gives up |
|-----------|------------|---------------------------|------------------|
| 0         | `PostOnly` | near touch (buy@bid)      | fill certainty |
| 1         | `Gtc`      | near touch                | the maker-only guarantee |
| 2         | `Gtc`      | far touch (buy@ask)       | the spread |
| 3+        | `Ioc`      | far touch + slippage bps  | the spread, slippage, and any unfilled remainder |

What increases monotonically is not price aggression — levels 0 and 1 price identically — but
**what we are willing to give up to get the position on**. Each row is a Hyperliquid fact:

- **0 is post-only because in-block priority is `cancel > post-only > GTC > IOC`.** A passive
  quote submitted in the same block as somebody else's taker is processed first, so level 0 is not
  merely the cheap option, it is structurally ahead in the queue — and post-only *cannot* pay a
  taker fee, which is the guarantee a market-making strategy is actually asking for.
- **1 exists because post-only is rejected, not demoted,** if the book moved between our snapshot
  and the venue's block. A strategy that must have an order working cannot use level 0. Level 1
  buys certainty of *placement*, not of fill.
- **2 crosses the spread but stays a limit,** so a partial fill leaves the remainder resting at a
  price we chose and slippage is bounded by one spread rather than by the depth of the book.
- **3 is IOC because Hyperliquid has no native market order** — a market order there *is* an IOC
  limit priced through the book ([04](../04-provider-abstraction.md)). Leaving no residue is the
  point: an urgent exit that half-fills and rests is an unmanaged position.

The table **saturates**. `urgency` is a `u8`, and a strategy writing `255` means "as fast as
possible": rejecting the record would drop precisely the signal you least want dropped, and
clamping to passive would answer an urgent exit with a resting quote.

### 4. `price_band` is a wall, and a wall that makes the order pointless suppresses it

`price_band` is the worst acceptable price — a ceiling for a buy, a floor for a sell, `0` meaning
no band. It can therefore only ever move the limit *away* from the market, never through it, so a
passive order that the band pushes deeper into the book is sent as asked; that is a valid intent
and overriding it would be us second-guessing the strategy.

For the **marketable** levels (2 and 3) a band that leaves the limit unable to cross produces
**no order at all**. The two alternatives are both wrong. Sending it is a guaranteed no-op that
still consumes a nonce and a rate-limit credit and still writes a log line saying we tried.
Quietly resting it instead invents an intent nobody expressed: the strategy asked to trade *now*
at a price that cannot trade, and the honest response is to say so, not to substitute a different
order. A negative band is refused outright — it is not a price.

The planner also fails closed on a locked, crossed, or one-sided top of book. Pricing off one
produces an order that is aggressive or passive by accident.

### 5. `cloid` is derived from the signal's identity

```
bit 127     : CLOID_PLANNER_TAG
bits 126..64: ts_event nanoseconds (low 63 bits)
bits  63..32: seq (low 32 bits)
bits  31..0 : symbol_id
```

Determinism is the whole point. A submit that times out has to be retried, and a retry that mints
a fresh id is a *second order* — the venue's cloid de-duplication only saves us if we hand it the
same id twice. The same property makes a replayed session produce byte-identical order ids, which
is what will let the Phase-5 parity harness diff a backtest against a live run.

Nothing here is a hash, so an operator reading a cloid out of a venue log can recover which signal
produced it. The tag bit does two jobs: it guarantees the id is non-zero, and it keeps this id
space disjoint from `OrderTracker`'s adopted-order ids, which are a bare venue `oid` widened to
128 bits and therefore always have bit 127 clear. Two orders sharing a cloid is the one collision
that makes reconciliation attribute a fill to the wrong order.

### 6. A superseded working order is cancelled, not left resting

**Decision: cancel/replace, with one exception.** When a new signal arrives, every order still
working on that symbol is cancelled and the delta is recomputed against the *filled* position.
The exception: if there is exactly one working order and it is already precisely the order we
would place — same side, price, size, TIF and reduce-only flag — it is left alone.

The reasoning runs both ways and Hyperliquid decides it. Against replacing: cancel/replace is
charged in queue position, since a replaced post-only order goes to the back of its price level,
so re-issuing an identical order is a strict loss. In favour: an order resting against a
superseded target is a stale quote, and a stale quote is exactly what somebody else's taker is
looking for. The venue's in-block priority resolves the risk that normally makes cancel/replace
dangerous — with `cancel > post-only > GTC > IOC` inside one block, a cancel and its replacement
submitted together cannot both be live, so there is no window in which we hold double the intended
exposure. `Plan` therefore orders `cancels` before `orders` and the caller must submit them in that
order; on a venue without that guarantee the caller must await the cancel acks first.

Every no-order path still emits the cancels — including "already at target" and "no usable quote".
An order left working against a target we have reached, or against a market we can no longer
price, is the case the cancel priority exists for. Cancels are never risk-gated (ADR-0010), so
this cannot fail closed.

## Consequences

- **+** The boundary's version, kind, reserved, `seq` and `ttl_ms` fields now all do something, and
  every refusal is counted — `SignalStats` is the denominator for "are we trading on the signals
  Python thinks it sent?".
- **+** The planner is a pure function, so a recorded session re-plans to the same orders with the
  same cloids. That is the precondition for the Phase-5 parity harness, not a nice-to-have.
- **+** A retry is idempotent at the venue rather than at our discretion, which removes the worst
  failure mode of a timing-out REST submit.
- **+** `axon-strategy` still does not depend on `axon-execution`: `WorkingOrder` is a small local
  view the runtime projects its tracker state into. The strategy adapter feeds the execution
  engine; a dependency the other way would invert that edge.
- **−** The reader's staleness ceiling is one global number. A process running a fast and a slow
  strategy on one ring gets one policy for both; per-strategy ceilings need a second reader or a
  per-symbol config, and neither exists yet.
- **−** `seq` is a single stream, so the reader assumes one producer — which the SPSC ring already
  requires, but it means a second Python process publishing onto its own ring needs its own reader
  and its own baseline.
- **−** Cancel-then-replace on every changed target burns cancel allowance. The rate governor's
  budgets make that survivable and the "identical order" exception blunts the common case, but a
  strategy emitting a slightly different target every tick will churn, and the fix for that is a
  no-op band the planner does not have.
- **−** The delta is computed against the *filled* position on the assumption that the cancels
  land. If a cancel fails and the old order then fills, we overshoot. The tracker's
  `risk_position` and the guarded client are what bound the damage; the planner does not model it.
- **−** One order per signal. Slicing a target across child orders needs a leg index, and the cloid
  layout has no room left for one — that change must re-cut the layout rather than extend it.
- **−** `Plan` allocates two `Vec`s per signal. Signals are per-decision, not per-tick, so this is
  nowhere near the hot path, but it is not zero and a Phase-8 core would want it inline.

See [ADR-0006](0006-signal-schema-and-spsc-ring.md) (the record and the ring this reads),
[ADR-0010](0010-execution-events-and-reconciliation.md) (the risk gate and tracker this feeds),
[06](../06-strategy-contract.md), [04](../04-provider-abstraction.md).
