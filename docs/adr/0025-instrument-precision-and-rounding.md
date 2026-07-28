# ADR-0025 — Instrument precision at the port: tick, lot, and who rounds

**Status:** Accepted · **Date:** 2026-07-26

## Context

Axon had **no** tick-size or lot-size handling anywhere on the production path. Every price and
every size we computed went to the wire exactly as computed, and the venue's answer to a number
its grid does not admit is `tickRejected` or `minTradeNtlRejected` — a well-formed action, a
valid signature, and a refusal. From inside the process that is indistinguishable from a signing
bug, which is the single most expensive place to be wrong.

It had not bitten yet for one reason: the only order ever placed live was post-only *at the
touch*, priced from a number the venue itself emitted. It was valid by accident. Every path that
**computes** a price was broken:

- `Planner` urgency 3 prices at `far_touch × (1 + slippage_bps/10000)`. Against a real testnet ask
  of 64 452 that is 64 774.26 — nine significant figures against a five-figure cap. Rejected.
- Sizes arrive over the IPC ring as `Decimal::new(v, 8)`, eight decimal places. BTC is
  `szDecimals: 5`. A target of `0.00123456` is rejected.

Both are on the critical route for the two things about to run: the live fill test and the
Phase-4 Python-strategy-drives-testnet-execution session.

The rule itself (venue docs, *tick and lot size*, re-verified against the live testnet universe
and its own published examples):

- **Price:** at most **5 significant figures**, AND at most `MAX_DECIMALS - szDecimals` decimal
  places, where `MAX_DECIMALS` is **6 for perps** and **8 for spot**. **Integer prices are always
  legal, regardless of significant figures.**
- **Size:** exactly `szDecimals` decimals.
- `szDecimals` comes from `meta.universe[i]`. `decode_meta` parsed `name` and threw it away.

Six questions had to be answered, and each has an obvious answer that is wrong.

1. **Where does per-instrument precision live?** The obvious home is `Capabilities`, which is
   already the per-adapter descriptor. Wrong: `Capabilities` is a *venue* fact — order types,
   TIFs, batch size, the rate model — and it is `&'static` and const-constructible precisely so a
   capability check is free on the submit path. Tick and lot are per **instrument**, differ
   between two symbols on the same venue, and arrive over the network at startup. The other
   obvious home is `SymbolMap`, which already holds the venue's per-asset facts — but it is a
   venue crate type, and the planner reaching for it would be exactly the dependency ADR-0004 and
   ADR-0014 exist to prevent.
2. **Who rounds?** The encoder is unbypassable, which makes it the obvious choice — and it is the
   wrong one on its own, because a price silently changed at the wire is a price the recorded
   `Plan` does not contain. That plan is what the chain summary logs, what `WorkingOrder::is()`
   compares on the next pass, and what the parity harness diffs; a mutated price makes all three
   describe an order that was never sent.
3. **Which direction does a price round?** "Down for a buy" is the safe-sounding answer and it is
   wrong for half the orders: a **marketable** order rounded away from the market may no longer
   cross, which silently converts "take liquidity now" into a resting quote and leaves an
   unmanaged position — the exact outcome urgency 3 exists to prevent.
4. **Which direction does a size round?** Up overshoots the target. Down can reach zero, and a
   size that rounds below the venue's $10 minimum notional is a guaranteed rejection that still
   costs a nonce and a rate credit.
5. **Spot vs perps.** `MAX_DECIMALS` differs (8 vs 6) and `SymbolMap` already has a `SPOT_OFFSET`.
6. **What happens when precision is unknown** — an instrument whose `meta` was never fetched, a
   backtest with no venue? Failing closed costs a trade; failing open sends a rejected order.

There was also a seventh thing, and it is the one that makes this urgent rather than tidy.
`tests/live_fill_testnet.rs` carried its **own private `round_price`**, written by whoever hit
this wall first. That is the knowledge existing in the wrong place, and it is the dangerous shape
of the bug: the live fill test rounded its own price and **passed**, over a production encoder
that was still broken. A green test over a broken path manufactures confidence in the path, and
is strictly worse than a red one. (It was also *wrong*: it counted "integer digits" as zero for
every sub-$1 price and capped at five decimals where the venue permits six — throwing 3 bp away
on every cheap asset, which on a passive quote is the whole edge.)

## Decision

### 1. Precision is a venue-neutral type on the port, in its own module

`axon-providers::instrument` gains `PriceGrid`, `SizeGrid`, `InstrumentSpec`, `InstrumentTable`
and `Precision`. It sits beside `capabilities.rs` and uses only `HashMap`, `thiserror` and
`axon_core::{Decimal, Side, SymbolId}` — **no dependency edge is added anywhere in the
workspace**, and `axon-providers` stays a leaf over `axon-core`.

The price rule is **one shape with two fields**, not an enum:

```
PriceGrid { increment, sig_figs: Option<u32> }        tick_at(px) = max(increment, sig_quantum(px))
```

A two-variant enum (`{ Digits, Grid }`) is the tempting model and it is a venue leak wearing a
port's clothes: venue three adds arm three and every `match` in the tree grows. This composes
instead — a CEX with a fixed `PRICE_FILTER.tickSize` sets `increment` alone, Hyperliquid sets
both, a simulator sets neither, and no venue gets its own arm. It is also *exact*: `sig_quantum`
clamps at one, and that clamp **is** the integer exemption, expressed as arithmetic rather than as
a special case a reader has to remember. Every published example validates, plus the two
decade-crossing cases (`99999.9 → 100000`, `9.99999 → 10`) where rounding up lands on a value
that is a multiple of every coarser tick.

`Precision` has **three** variants, not two:

```
Known(&InstrumentSpec) | Unconstrained | Unknown
```

"This venue has no rules" and "we do not know this venue's rules" are different sentences.
Collapsing them is how a backtest becomes silently more permissive than the live session it
claims to reproduce.

None of the arithmetic touches `f64`. Counting significant figures with a float is the trap:
`(100000f64).log10()` is `4.999…`, which drops a digit at exactly the magnitude where a
five-figure rule bites, and the money on that path is a five-figure BTC price. The decimal
exponent is computed from the mantissa and the scale as integers.

### 2. The planner rounds; the encoder refuses

**One place decides, a different place cannot be bypassed.** The planner knows the intent, so it
is the only component that can round in the direction that preserves it and the only one that can
decide a size rounding to zero means *no order* rather than a zero-size order. `order_wire` then
refuses anything not already on the grid — and `submit_orders → order_action → order_wire` and
`modify → modify_action → order_wire` are the only two routes to the bytes, so the refusal cannot
be forgotten by a call site.

`order_wire`'s `precision` argument is **required and a three-way enum, not an `Option`**: an
adapter author cannot encode an order without naming what is known about the grid, and
`Unconstrained` is a word that has to be typed. That is ADR-0010's structural move — `GuardedClient`
*is* an `ExecutionClient`, so the gate cannot be skipped — one level down and cheaper, because
`order_wire` already has the unbypassable property a wrapper would have to manufacture. A
`PrecisionClient` pipeline wrapper was considered and **rejected**: there is no adapter without an
encoder in this tree, so it is speculative generality that duplicates a refusal every adapter must
make at its own wire anyway, and it would add a fifth site where the reduce-only asymmetry has to
be restated identically.

`InstrumentSpec::check` obeys exactly the rules `quantize` obeys, and a property test pins that
they agree. Any drift between them is a session that plans orders its own encoder rejects, which
in a log is indistinguishable from a venue outage.

`ProviderError` gains a distinct `Precision` variant rather than reusing `Rejected`: an off-grid
refusal means *our* table or *our* planner is wrong, and the fix is on this side of the wire.
Folded into `Rejected` it would read like a margin refusal and send an operator to the account.

### 3. A price rounds in the direction of its own intent

`PriceIntent::{Passive, Marketable}`, chosen from `UrgencyRule::is_marketable()`:

- **Passive (urgency 0, 1 — near touch) rounds AWAY from the market.** Not politeness: post-only
  is *rejected*, not demoted, if it would cross (ADR-0014 §3), so a level-0 buy rounded up into
  the spread is not a slightly worse quote — it is an `alo` rejection and a strategy sitting flat
  with nothing working. The worst case of rounding away is one tick deeper in the queue, which is
  a valid intent nobody needs protecting from.
- **Marketable (urgency 2, 3+ — far touch) rounds TOWARD the market.** Rounding a taker away can
  leave a limit that no longer crosses, which converts an urgent exit into a resting quote.

And when the grid *does* push a marketable order out of the market, the answer is
**`NoOrder::RoundedUnfillable` — no order at all**, which is ADR-0014 §4's precedent one layer
down: the strategy asked to trade now at a price that cannot trade, and the honest response is to
say so, not to substitute a different order. It is a **separate variant** from
`PriceBandUnfillable` because the operator's fix differs — "your band is too tight" versus "this
instrument's grid is coarser than your band" — and a shared variant would send somebody to change
the wrong number.

The band is re-applied **after** quantization, itself quantized away from the market. Without that
second clamp a marketable buy rounded toward the market steps through its own band by up to one
tick, and a risk bound violated by arithmetic is worse than one violated on purpose. The resulting
invariant is unconditional: **the price sent is never worse than the band the strategy set.**

Cost, stated: the common case is a no-op, because bid, ask and the venue's own touch are prices
the venue printed. Only urgency-3 slippage and bands ever produce an off-grid price — which is
precisely why this had not bitten yet. Where it does bite, a marketable order pays up to one tick
more than asked (on BTC above 10 000, one dollar), bounded, on an order that already accepted the
whole spread.

### 4. The **target** is quantized, not the delta; sizes truncate toward zero

Quantizing the delta is the intuitive choice and it is a bug. Sizes arrive at eight decimal
places; BTC's lot is five. A target of `0.00123456` against a held `0.00123` leaves a residue of
`0.00000456` that truncates to zero **on every signal, forever** — `AlreadyAtTarget` becomes
unreachable, and a strategy that is correctly at its target reads as one that is permanently
refused. Quantize the target and the delta between two on-grid numbers is on-grid by
construction: `FLAG_CLOSE`'s zero trivially, and the reduce-only projection clamps to
`[-position, 0]` where the position is a sum of venue-reported fills.

`qty` is re-quantized after the delta as insurance against the one case that cannot cover — a
venue re-precisioning an asset mid-session leaves the *held* position off the new grid — and a
zero there is `NoOrder::BelowLotSize`, never a zero-size order.

Quantizing the target has its own failure at the other end, and it is judged **from flat**. Held
`0.00123` against a target of `0.00123456`, the residue is genuinely smaller than one lot and
`AlreadyAtTarget` is the honest answer — that is the paragraph above. Held *nothing* against a
target of `0.36` on an instrument whose lot is one whole coin (125 of testnet's 210 perps have
`szDecimals: 0`), the lot erases the entire request and the delta is zero for the opposite
reason: the strategy asked for exposure and got none of it, on every signal, forever. Reported
as `AlreadyAtTarget` that lands in `IntentStats`' healthy arm, so `precision_refusals` stays
zero, neither `prec` nor `NOSPEC` reaches the status line, and a session being refused all day
is indistinguishable from one that chose to be flat. From flat it is `NoOrder::BelowLotSize` —
the existing variant, the existing counter — and it names the size asked for beside the lot that
erased it.

Sizes **truncate toward zero on every path, including reduce-only**. Rounding up overshoots the
target, which is the same class of error as sending the target instead of the delta; on a reduce
it exceeds the position and a flatten overshoots straight into an opposite-side one. It is also
the trap the live test already built a `MAX_NOTIONAL` tripwire around: on `szDecimals: 0` a $12
target rounds up to one whole unit, and one whole unit of a $1,000 asset is a $1,000 order.

Minimum notional is refused **only for orders that add exposure** (`NoOrder::BelowMinNotional`). A
sub-minimum *close* is sent. The published rule does not say whether the venue exempts closes, and
guessing in the closed direction strands a position on our own opinion; if we are wrong the venue
tells us, loudly and observably, and if we refuse nobody ever finds out. `InstrumentSpec::check`
carries the identical exemption, because the alternative is a planner whose own encoder rejects
its decisions.

### 5. Spot is refused, by name, rather than approximated

`decode_universe` reads `meta.universe` only, so every id at or above `SPOT_OFFSET` has no spec,
is refused by `resolve_symbols` at startup, and would reach the planner as `Unknown` if it somehow
arrived. The type already handles it — `SPOT_MAX_DECIMALS` is named and
`PriceGrid::decimals_with_sig_figs(SPOT_MAX_DECIMALS - szDecimals, 5)` is the whole change — so
landing it is a `decode_spot_meta` and an `insert`, not a type change.

It is refused rather than approximated because spot `szDecimals` lives on
`spotMeta.tokens[].szDecimals`, keyed by the pair's **base token in a different index space**.
Reusing a perp's value produces a size wrong by a factor of ten or a hundred — an order that is
*accepted, fills, and is the wrong size*, which is strictly worse than one that is rejected. A
refusal is visible; a silently-wrong size is not.

### 6. Unknown precision fails closed for exposure, and admits exactly one thing

The governing sentence: **with no grid, the only order we will send is a reduce-only order for the
whole position at a price the venue itself printed.** Both of those numbers came from the venue —
the size is `-position`, a sum of venue-reported fills and therefore a whole number of lots, and
the price is a touch the venue published and is therefore on the venue's own grid. Urgency-3
slippage is dropped on this path, because a slipped price is a number *we* computed; the downgrade
is documented and counted, not silent.

A **partial** reduce is refused. Its size is a number we computed and can be off-lot, and that
distinction is the difference between an exemption and a hole. Same for a band that would move the
price.

This is ADR-0010's asymmetry, applied again: whatever went wrong, removing exposure must keep
working while adding it must not. It holds at both layers — the planner refuses with
`NoOrder::UnknownPrecision`, and the encoder refuses independently with
`EncodeError::PrecisionUnknown` — and it is why an `ExchangeClient` nobody handed a table to
refuses every opening order and still flattens.

`resolve_symbols` refuses a configured coin with no spec **at startup**. The check is free (the
map and the table come out of one `meta` response) and it moves the failure from the first signal
to the one moment an operator is watching; without it the session runs, reconciles, prints OK and
never trades that instrument.

### 7. One `meta` read, both halves

`decode_universe` returns `Universe { symbols, instruments }` and `decode_meta` survives as a thin
wrapper over it, so the existing decode test and four call sites are untouched. One decode and not
two, because the asset index and the lot come out of the *same* `universe` array: two reads taken
a moment apart can disagree after a listing, and an order signed across that seam trades one coin
at a size computed for another. A `szDecimals` that will not build a spec is a `DecodeError`, not
a skipped asset — an asset silently absent from the table fails closed later with nothing attached
saying why. A universe entry with **no** `szDecimals` fails the decode outright rather than
defaulting, because a default would pick whatever number happens to suit BTC.

`run_live` builds **one** `Arc<InstrumentTable>` at a single site and hands it to both
`IntentSource` and `ExchangeClient::with_instruments`. Two tables that can drift apart would have
the planner rounding to one grid while the encoder refuses against another — a session-wide outage
that reads in the log exactly like a venue rejection.

### 8. The live fill test loses its private helper

`round_price`, `integer_digits` and the two constants are deleted. The test prices through the
same `PriceGrid::quantize` the planner uses, off the table `decode_universe` returns, and builds
its `ExchangeClient` with that table. It still asserts exactly what it asserted about fills, and
it is still `#[ignore]`d.

Its numbers move in the safe direction and the diff says so: the entry was
`round_price(ask × 1.005)` (nearest) and becomes ceil for a buy — `≥` the old value, still
marketable, still inside the `CROSS` budget by less than one tick — and the flatten's sell becomes
floor, likewise. One offline expected value changes, `0.00343 → 0.003429`, and the reason is
written into the test: the old helper was capping sub-$1 prices at five decimals where the venue
permits six. Both are legal; the new one is strictly finer.

## Consequences

- **+** The urgency-3 path and every ring-scaled size are now legal at the venue. That is the
  whole point: two paths on the critical route for the next two runs went from certain rejection
  to correct.
- **+** The rule is enforced twice, by two components that cannot both be forgotten — the planner
  decides, the encoder refuses — and a property test pins that they agree.
- **+** `szDecimals` is threaded from `meta` to the wire. Nothing above the adapter could
  previously answer "how many decimals may a BTC size have?"; now the planner can.
- **+** The knowledge left the live test. The test now exercises the production path it exists to
  validate, so a break in that path fails it instead of being hidden by it.
- **+** Determinism survives: rounding is a pure function of `(price, qty, spec)` and `cloid_for`
  never saw it, so a replayed session still plans byte-identical intents with identical cloids.
- **+** A latent churn bug is closed on the way past. `WorkingOrder::price` comes from the venue's
  own echo and is on-grid; an unquantized planner price could never equal it, so the
  leave-it-resting exception never fired on a coarse instrument and every pass cancel/replaced an
  order that was already correct.
- **+** No dependency edge is added anywhere. `axon-providers` stays a leaf over `axon-core`,
  `axon-strategy` gains nothing on `axon-execution` or any venue crate (ADR-0014 preserved), and
  `axon-replay`'s library is untouched — only its harness, over a dev-dependency it already had.
- **−** `Precision::Unconstrained` is a fail-open hole with a friendly name. It exists so the
  planner's existing tests compile unchanged, and a future call site reaching for
  `PlanContext::new` out of habit gets no rounding. The encoder's independent refusal is what
  bounds the damage — which makes that second refusal load-bearing, not belt-and-braces.
  Production cannot reach it by accident: `intent.rs` uses a struct literal and must name a field.
- **−** Nothing in the type system guarantees the planner's table is the client's table. One
  `Arc`, one construction site, and `resolve_symbols` refuses anything missing from it. The
  airtight fix (`fn instruments(&self)` on `ExecutionClient`) would fail every existing spy client
  closed and force edits to tests about something else.
- **−** `MIN_ORDER_NOTIONAL_USD` is a constant in our source that the venue owns. It is not in
  `meta`. The *type* is right — a CEX publishes its own in `exchangeInfo` — so this is a
  population weakness, the least bad place for it; but if the venue changes the number we are
  silently wrong in the direction that costs a nonce per signal.
- **−** The reduce-only min-notional pass-through sends one rejected order per pass, unbounded,
  for a dust position the venue may not let us close. The loud failure was chosen over the
  invisible one, and it is loud in the wrong units — an operator sees a rejection line, not a
  "position you cannot close" alarm — unless `precision_refusals` is read beside it. Nothing backs
  it off.
- **−** `max(increment, sig_quantum)` is wrong for a venue with both a non-power-of-ten increment
  *and* a significant-figure cap (a multiple of 0.25 can carry six significant figures). No venue
  is in that quadrant and `decimals_with_sig_figs` is the only constructor that can set
  `sig_figs`, so it cannot be reached today — but that is a refusal, not a solution.
- **−** A tick that varies by price **band** (Deribit options, several futures venues) does not fit
  at all: `increment` is one number. Supporting it turns `PriceGrid` from a `Copy` two-field struct
  into something that allocates, which propagates into `PlanContext` being `Copy`.
- **−** The offline grid in `selftest.rs` is a fiction, and says so. The default gate therefore
  proves the *wiring* of rounding, not the numbers. The only thing that proves the numbers against
  a real venue is the `#[ignore]`d live test. It declares a spec for **every** configured coin and
  not for the two the canned stream touches, because `resolve_symbols` refuses a coin with no
  grid: a mode with no socket, no key and no universe must not refuse a three-symbol config on a
  venue's behalf, in a sentence that sends an operator to look at that venue.
- **−** `min_notional` is checked against the planner's limit price, not the fill. A marketable
  buy's limit sits above the expected fill, so the check is pessimistic on the buy side and
  optimistic on the sell side, and an IOC can partial-fill and leave a sub-$10 remainder. The right
  number is unknowable before the fill; this is a permanent approximation, not a TODO.
- **−** The strategy cannot see or override the rounding direction. Its only lever is `price_band`,
  which the schema defaults to 0, so an urgency-3 order is rounded one tick toward the market on
  the engine's authority. Safe and bounded, and still the engine spending the strategy's money on a
  policy it never agreed to.

## Out of scope, named

1. **Spot.** §5 above.
2. **Carrying the table into the capture log** (`LogRecord::Instruments`, `SCHEMA_VERSION` 1→2)
   **and into `ChainSummary`** (`RESULT_SCHEMA_VERSION` 3→4). The plan is now a function of a
   network-fetched input that the log does not carry, and **this increment makes that hole
   bigger, not smaller.** It was "a replay of a capture taken after a re-precision plans a
   different size"; it is now "a replay of *any* live capture plans a different **price** on
   every order the grid moved" — urgency-3 slippage and every band — because the session rounded
   and the replay does not. `PlannedOrder.price` is compared exactly by
   `python/axon/backtest/golden.py`, so those differences surface as strategy flips inside the
   harness built to tell a strategy change from a harness change, and `docs/07` makes that diff
   promotion gate #5. What this increment does instead of fixing it: `ChainProbe::new` takes the
   table as a **required argument** (the `order_wire` move again — a caller must name what it
   claims to know), so a driver with a session's grid can hand it over; `replay_log` says on
   stderr that it has none; and `a_replay_handed_no_grid_plans_a_price_the_session_it_reproduces_could_not_have_sent`
   pins the divergence with the fixture's own numbers, so it cannot go quiet again. The real fix
   invalidates every stored log and every golden reference, which is a separate, reviewable
   decision and does not belong in a diff that lands on the signing path.
3. **Refreshing the table mid-session.** No TTL, no re-fetch on the reconciliation poll.
   `resolve_symbols` pins us to configured coins so the window is narrow, but a delist-and-relist
   is untradeable until restart. The `qty` re-quantization guards the size half; the price half is
   unguarded.
4. **Mapping the venue's own `tickRejected`/`minTradeNtlRejected` strings to
   `ProviderError::Precision`** for stale-table detection. The variant exists and is the natural
   target; wiring `response.rs` and a counter is the next increment.
5. **`isDelisted`.** The data is in the same `universe` objects and 53 of testnet's 210 perps carry
   it. It stays in the live test's private `asset_spec` deliberately: "this asset cannot be traded"
   and "this asset's numbers are unknown" are different failures with different messages, and
   folding a tradability check into a precision diff makes both harder to review.
6. **`minQty`/`maxQty`/`minPrice`/`maxPrice`.** No venue behind them today, and `maxQty` forces a
   real decision — clamp or slice — that should be made when a venue forces it. `SizeGrid` is the
   struct they land on.
7. **A `PrecisionClient` pipeline wrapper.** §2 above.
8. **Order slicing.** The `cloid` layout has no room for a leg index and `cloid_for` says so.

See [ADR-0004](0004-provider-abstraction-layer.md) (venues are adapters behind one interface),
[ADR-0010](0010-execution-events-and-reconciliation.md) (the unbypassable-gate pattern and the
reduce-only asymmetry this reuses), [ADR-0013](0013-runtime-supervision-and-safety-loop.md),
[ADR-0014](0014-signal-to-order-planning.md) (the planner this extends, and the no-order precedent
`RoundedUnfillable` follows), [04](../04-provider-abstraction.md).
