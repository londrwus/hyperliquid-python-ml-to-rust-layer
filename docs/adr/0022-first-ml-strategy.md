# ADR-0022 — The first ML strategy up the ladder, and what climbing it is allowed to claim

**Status:** Accepted · **Date:** 2026-07-25

## Context

Phase 5 built every rung of the validation ladder in [07](../07-parity-and-testing.md) and
nothing had climbed one. `axon.features` could pin a recipe no model had been fitted on,
`axon.models` could register an artifact nobody had trained, and `axon.parity` could fail three
gates nothing had ever been put through. A harness that has never carried a real strategy is not
a harness; it is a set of assertions about a strategy that does not exist.

So this ADR is about one small, real, honest strategy taken all the way up — and, more
importantly, about **what a green ladder is allowed to claim**, because the two obvious readings
are both wrong:

- *"The gates passed, so the strategy works."* The gates compare the live path to the research
  path. Every one of them would still be green for a model that loses money at maximum speed.
  They protect against training–serving skew, not against being wrong about the market.
- *"The gates are only about plumbing, so the numbers don't matter."* Also wrong, and worse:
  the whole value of the harness is that a signal measured in research is the signal that trades.
  If the research number is fiction — a leaky label, an overlapping split, a backtest with no
  costs — then parity guarantees the faithful delivery of fiction.

The claim a green ladder supports is exactly this: **the live path computes what research
computed, and the research number was honestly obtained.** Whether the number is worth trading is
a separate question, answered by the number itself. For this strategy the answer is no, and
saying so is the point of the exercise.

Five questions had to be answered, each with a reasonable-sounding wrong answer:

1. **What data?** "Generate a realistic series" is fast and it proves nothing about anything.
2. **What may a feature read?** "Whatever the research frame has" is how a strategy is built that
   can never be served.
3. **What is the label?** Any forward return is a lookahead; the question is which lookahead is
   *disclosed* and how the split is purged of it.
4. **Which model ships?** Refitting on everything after a walk-forward is standard practice and
   it ships a model that no reported number describes.
5. **What gets spent on Modal?** The first expensive job is the one that must not be an accident.

## Decision

### 1. Real Hyperliquid candles, cached, with a committed slice so the gate stays offline

`axon.strategies.data` pulls hourly BTC and ETH from the public, read-only
`POST /info {"type":"candleSnapshot"}` — no key, no order, no spend — and caches it under the
gitignored `data/`. A **verbatim 800-bar slice of each is committed** under
`python/axon/strategies/_fixtures/`, so `test_strategies.py` trains and gates on numbers the
venue actually printed without touching the network. A live fetch refuses unless
`AXON_ALLOW_NETWORK=1`; the default gate is offline by construction, not by convention.

Four properties of the loader are load-bearing:

- **The unclosed bar is dropped.** The venue returns the bar still forming; its close is a
  mid-bar price stamped with a close time in the future. It is also always the *last* row, which
  every walk-forward puts in its most recent test window — the block reported as out-of-sample.
- **Event time is the bar's close** — `T + 1 ms`, in nanoseconds (`CLOSE_STAMP_OFFSET_MS`). `T` is
  the bar's *last* millisecond, so a bar stamped `T` sorts equal to every trade printed inside it;
  `T + 1 ms` is the instant it is final for ordering, which is what `axon_core::Candle::ts_event`
  is documented to be ("the point at which it is final for ordering"). A bar stamped with its open
  is the textbook leak `axon.strategy.events.Bar` already warns about. **The Rust decoder does not
  carry this yet** — `decode_candle` stamps `close_time * MS_TO_NS`, one millisecond earlier — and
  that disagreement is recorded as a required edit below rather than papered over here.
- **Prices are parsed with `Decimal` into the contract's fixed-point integers** and become floats
  exactly once, in `axon.features.bar_inputs`. A `float(s)` in the loader would round differently
  from the `i64` a live `Bar` carries, and research and serving would disagree in the last bits
  before a single feature had been computed.
- **Gaps are reported, never filled.** An interpolated bar is a close nothing traded at, and every
  return feature downstream would then be measuring our own arithmetic.

**Why hourly bars.** Not a preference — a measurement. `candleSnapshot` keeps roughly 5,000
candles *per interval* and serves the most recent ones whatever start time is requested (a window
a year back returns an empty list, which is how this was discovered). At a fixed row cap, a longer
bar buys strictly more calendar coverage for the same number of training rows: 15m gives 52 days,
1h gives 208, 4h gives 2.3 years. 1h is the point where coverage is reasonable and the four-hour
holding period does not make unmodelled funding the dominant term.

**Mainnet data, testnet execution.** Deliberate and slightly uncomfortable. Testnet candles are
thin and largely self-dealt; a model fitted on them learns the testnet market maker. The
mismatch is recorded below as a real weakness of anything the testnet path later observes.

### 2. Nine features, every one computable from what the live feed actually delivers

`PERP_BAR_V1` (`perp_bar/v1#a21328ed1532ecd4`) is the strategy's own `FeatureSpec` — small,
scale-free, and defensible column by column. **Each one names its live source**, because a feature
that cannot be served is a research artifact:

| column | transform | live source |
|---|---|---|
| `ret_1` | `log_return(close, period=1)` | `Candle.close`, HL `candle` subscription (`axon-provider-hyperliquid/ws/sub.rs`) |
| `mom_4` | `momentum(close, window=4)` | same |
| `mom_24` | `momentum(close, window=24)` | same |
| `sma_x_6_24` | `sma_crossover(close, 6, 24)` | same |
| `z_24` | `rolling_zscore(close, 24)` | same |
| `vol_24` | `realized_volatility(close, 24)` | same |
| `range_bps` | `relative_range(high, low, close)` | `Candle.high/low/close` |
| `clv` | `close_location(high, low, close)` | same |
| `vol_z_24` | `rolling_zscore(volume, 24)` | `Candle.volume` |

Three transforms were added to `axon.features` for this: `relative_range` and `close_location`
(the two bar-shape features — a close-to-close series cannot tell a quiet 10 bps drift from a bar
that traded 80 bps in both directions and came back), and `sma_crossover`.

**Every feature has a finite lookback, and that is a decision, not an accident.** The obvious
trend feature is an EMA crossover, and `axon.features` already has one. An EMA never forgets its
seed, so a serving path holding a bounded buffer computes a *different* number from a research
pass over the whole history — and the gap is widest immediately after a restart, which is the
moment nobody is comparing feature values. With only windowed transforms in the spec, a buffer of
`SERVING_BUFFER_BARS` reproduces the offline matrix **bit for bit**, and the feature-parity gate
compares transforms instead of histories. That is the difference between "parity within
tolerance" and parity; `test_a_finite_window_crossover_survives_a_bounded_serving_buffer_where_an_ema_does_not`
is the demonstration, and it fails on the EMA on purpose.

`FEATURES_VERSION` is deliberately **not** bumped: adding transforms does not change the numerical
meaning of any existing one, and bumping would invalidate every artifact that ever recorded a
spec — the fingerprint exists to move when the arithmetic moves, not when the library grows.

### 3. The label is the sign of the four-hour forward return, and the split is purged of it

`y_t = 1[log(c_{t+4} / c_t) > 0]` on hourly bars — a four-hour horizon, stated because a strategy
whose holding period does not resemble its label horizon is trading a different question from the
one the model was asked. The last four rows of every series have no label and are dropped rather
than zero-filled; a fabricated flat tail would land squarely in the most recent test block.

The split is an **expanding-window walk-forward, purged and embargo-capable, defined on event
time**:

- **Purged**, because a training row within one horizon of the boundary is labelled with prices
  from *inside* the test window. Without the subtraction the model has read part of its own exam
  paper and the only symptom is that the backtest looks good.
- **On event time, never row position.** The two coins are pooled into one model (legitimate only
  because every column is scale-free), so an index range holds one coin's future next to the
  other's past. `test_pooled_coins_are_split_on_time_so_one_coins_future_is_never_in_training`.
- **The first block is training-only.** Testing on the earliest bars would score a model that had
  never been fitted.

**Costs are reported as a hurdle, never netted off.** Turning fold metrics into a P&L needs a
turnover model, and a turnover model living in the evaluator would be a second implementation of
the strategy's own hysteresis — the exact duplication [03](../03-ml-fidelity-and-features.md)
warns about. So the evaluation reports the gross edge per decision and the fee schedule beside it
(4.5 bps taker / 1.5 bps maker per side, an assumption, since nothing in this repo has yet been
filled), and leaves the subtraction to a reader who can see both numbers.

### 4. The artifact is the last fold's model, and it records the recipe it was fed

`export_walk_forward` registers the model fitted for the **final** fold, whose test block was
genuinely out of sample for it — not a refit on the whole sample. Refitting is the tidier habit
and it ships a model that no reported number describes. The `sample_input` handed to
`export_artifact` is drawn from that fold's test block, so the round trip is evidence on the rows
the reported numbers came from.

`PerpBar.from_registry` refuses an artifact whose `feature_spec_ref` is not the spec this build
computes. That check is the whole audit trail made operational: a reordered column or a widened
window produces perfectly plausible probabilities from a model that has never seen those numbers,
and nothing else in the system would notice.

### 5. The strategy recomputes, refuses, and chooses its urgency deliberately

`PerpBar.on_bar` appends the bar to a bounded buffer, **recomputes the whole spec over the
buffer**, and takes the last row. An incremental update would be a second implementation of every
transform in `axon.features`; at an hourly cadence the recompute costs under a millisecond, and a
strategy that ever needs the incremental form goes to Rust behind the bit-equivalence gate
(Boundary A), not into a second Python path.

It refuses rather than guesses. A non-finite feature (a bar with no range, a zero price in the
feed) or a non-finite probability emits **nothing** — not a flat target, because flat is an
instruction to close a position that may be perfectly fine. There is no warmup counter either:
"how many bars until this is usable" is a property of the spec's windows, and the NaN warmup every
transform guarantees answers it exactly. The one thing that *is* checked by hand is at
construction: a buffer shorter than the longest window leaves the last row NaN forever, so the
strategy emits nothing, forever, without an error — indistinguishable from a strategy with no
opinion. A synthetic ramp through the spec catches that in the constructor.

**`urgency = 0`** (most passive). At the measured edge — a couple of basis points per decision —
crossing the spread and paying the taker fee costs more than the opinion is worth. A passive fill
or no fill is the correct trade here, and a strategy whose edge cannot survive its own execution
should say so in its urgency rather than in its P&L.

**`ttl_ms = 60_000`**, and it means something narrower than it looks. `ttl_ms` is a signal
**admission** window, not an order lifetime: `SignalReader::effective_ttl_ns` admits a record only
while the core's event clock is within `min(ttl_ms, intent.max_signal_age_ms)` of its `ts_event`,
and `Planner::plan` never reads the field. The number is chosen as the *opinion's* useful life —
the price it was formed at goes stale long before the four-hour half-life does, and the next bar
restates the target anyway.

Two gaps follow, and both belong in the record rather than in a comment:

* **The shipped ceiling wins.** `IntentConfig::max_signal_age_ms` is 2 000, so the effective window
  is two seconds, not sixty. `perp_bar` deployed unmodified drops any bar whose signal reaches the
  ring more than 2 s after the close — counted as `SignalReject::Expired`, and not retried for an
  hour. An operator running this strategy must raise `intent.max_signal_age_ms` to at least
  `ttl_ms`.
* **Nothing pulls a resting order on age.** `PerpBar.on_bar` returns early when the target is
  unchanged, so the planner is never asked to cancel and replace, and no field it reads expresses a
  lifetime. If an order lifetime is genuinely wanted it needs a *separate* field the planner reads;
  `ttl_ms` cannot serve, because the reader consumes it before the planner sees the record. That is
  a design change, not a repair.

The module imports no clock. `ts_event` comes from the bar, the context binds it around the
callback, and `test_the_signal_takes_its_event_time_from_the_bar_and_never_a_wall_clock` asserts
the stamp is *not* inside the wall-clock window the test ran in.

### 6. The heavy path is planned and not spent

`axon.strategies.jobs` expresses the search as `axon.compute` jobs — a 36-point hyper-parameter
sweep (`param_sweep`) and a per-window walk-forward fan-out — with real entrypoints in
`axon.strategies.remote`, not stubs. `plan()` was run against the real hwsched CLI; `run()` was
run only against the `fake` provider with a throwaway ledger. Nothing was submitted to Modal.

Three constraints found by reasoning them through rather than by paying to discover them, all of
which fail at the far end of an image build:

- **Only the `axon` package is mounted** (`add_local_python_source("axon")`), so
  `contracts/schema.toml` — which `axon.contracts` resolves relative to the repo root — is not in
  the image and *every* import of Axon fails. Each job therefore sets `AXON_SCHEMA_PATH` to a copy
  on the volume, and `test_axon_cannot_be_imported_from_a_bare_package_mount_without_the_schema_path`
  reproduces the container's view by copying only the package.
- **A fan-out drops `kwargs`.** hwsched materializes tasks from `params`/`tasks` and never merges
  the spec's `kwargs`, so constants every point needs ride as single-element grid axes.
- **hwsched must be driven by an interpreter that has pydantic** — the system `python3`, not
  Axon's venv, which has the ML stack and not hwsched's dependencies.

## Result — the actual numbers

Reproduce with `python -m axon.strategies --cache --coins BTC ETH --folds 4` (the committed
fixture, without `--cache`, is a 33-day slice and says nothing about edge).

**Sample.** BTC + ETH hourly, 4,999 bars each, 2025-12-29 12:00 UTC → 2026-07-25 18:00 UTC
(208 days), no gaps. 9,942 usable rows after warmup and label; 7,954 scored out of sample.

**Walk-forward, 4 purged folds:**

| | fold 0 | fold 1 | fold 2 | fold 3 | pooled |
|---|---|---|---|---|---|
| train / test rows | 1980 / 1988 | 3968 / 1988 | 5956 / 1988 | 7944 / 1990 | — / 7954 |
| base rate (up) | 0.497 | 0.517 | 0.460 | 0.514 | 0.497 |
| **AUC** | 0.5125 | 0.5390 | 0.5498 | **0.4993** | **0.5224** |
| coverage | 0.841 | 0.811 | 0.803 | 0.781 | 0.809 |
| hit rate | 0.505 | 0.529 | 0.545 | 0.503 | 0.521 |
| gross edge | +0.05 bps | +0.74 bps | +3.24 bps | +0.66 bps | **+1.16 bps** |

Per coin: BTC AUC 0.5299, ETH 0.5154.

**The gates:**

- **Model parity — PASS.** n = 1,990, `max_abs_diff = 0.000e+00` at `TREE_EPS = 0`, 0 decision
  flips under the strategy's own entry band, 0 non-finite. The artifact's own round trip is also
  exactly 0. `perp_bar_xgb@1`, 215,146 bytes, `sha256:75b9354fe763…`, and the same content hash on
  every re-run (`n_jobs=1` is not a performance setting — histogram building has a
  non-deterministic reduction order, and a multi-threaded fit makes every artifact hash different).
- **Feature parity — PASS, exactly.** 4,975 rows × 9 columns per coin, `max_abs_diff = 0.000e+00`,
  0 mismatched cells, 0 NaN mismatches. The online side is the real serving path — a 256-bar
  buffer, one `on_bar` at a time — and the offline side is one vectorized pass over the whole
  history. Zero, not "within 1e-9". **The gate counts as well as compares**: 4,975 rows expected,
  4,975 produced by the serving path, 4,975 aligned and compared. The alignment is an intersection
  on event time, so a serving path that produced *no* row on a bar is not a mismatch to it — it is
  simply absent, and the rows that remain still agree to the last bit. A gate without the count
  reports `PASS … max_abs_diff = 0.000e+00` for a path that emitted half the rows, which is the
  same reading a healthy run gives in every field but one.
- **Drift — OK.** Worst PSI 0.1166 (`vol_24`, moderate) for the final test block against its own
  training fold; everything else stable. As a check that the gate can still see, SOL — a coin the
  model never trained on — reads 0.2318 on `vol_24` and 0.1760 on `range_bps` and *stable*
  everywhere else, which is a fact about the features rather than about SOL: they are scale-free,
  so a different symbol is largely not out of distribution. Drift detects a moved regime, not a
  misaimed strategy.

**And the honest reading: this model is not tradeable.**

An AUC of 0.5224 over 7,954 overlapping rows is a very weak signal, and the fold-by-fold spread
(0.4993 to 0.5498) is wide enough that the pooled number is not stable. Worse, the decomposition
says the little that is there is a directional bias and *nothing* else — the selection is worth
less than no selection at all. The command above prints this block; it is computed by
`axon.strategies.training.decompose` rather than worked out by hand beside the pipeline, which is
how the retracted version described two paragraphs down came to be published in the first place:

| | n | hit | gross edge |
|---|---|---|---|
| long decisions | 2,894 | 0.513 | **−2.04 bps** |
| short decisions | 3,541 | 0.527 | +3.78 bps |
| **every decision — the model** | **6,435** | **0.521** | **+1.16 bps** |
| *always short, those same 6,435 rows* | *6,435* | — | *+2.99 bps* |

The market drifted down over the window: mean forward return **−2.99 bps over the 6,435 rows the
model took a position on**, and −1.05 bps over all 7,954 scored rows. So the like-for-like
benchmark is a constant short held on exactly the model's own decision set — it earns **+2.99 bps
against the model's +1.16**, which makes the selection worth **−1.83 bps**. The model is not adding
to the drift; it is handing part of it back, because its long side loses 2.04 bps over 2,894 rows.

The benchmark has to be the model's own rows, and this is the trap: on the rows where the model is
short, a constant short *is* the model, so both are +3.78 bps by construction and the comparison is
empty. Subtracting a benchmark measured on one row set from a model measured on another is how a
directional bias reads as skill — an earlier version of this table did exactly that and credited
the model with +0.79 bps it does not have.

Against a maker round trip of 3.0 bps (9.0 bps taker), before spread, slippage and funding, the
+1.16 bps pooled gross edge does not clear the hurdle in any reading — and it does not clear the
one benchmark that costs nothing to run either. Score deciles are not monotone: the three highest
deciles have negative mean forward returns.

Raising the entry band to 0.10 lifts the gross edge to +3.86 bps on 25% coverage — and that number
is **selection on the test set**, quoted here only to be disowned. Held to the same benchmark it
fails the same way: a constant short over those 1,983 rows earns +4.89 bps. The band was not
tuned; it is the constructor default, and it stayed there.

## Consequences

- **+** A real strategy has climbed the ladder. The three gates ran against a real artifact fitted
  on real venue data, and two of them are exact rather than tolerant: model parity at `TREE_EPS = 0`
  and feature parity at `max_abs_diff = 0`.
- **+** Feature parity is exact *by construction*, not by luck. The finite-lookback rule means a
  bounded serving buffer and a full-history recompute are the same arithmetic on the same numbers,
  so the gate reports a real zero instead of a small number nobody can interpret.
- **+** The whole run is reproducible from one command, and the artifact hash is stable across
  runs. Every number above can be disagreed with by re-running rather than by trusting.
- **+** The Modal path is expressible, planned and priced without a dollar being spent, and the
  three ways a job silently fails in the container are now closed and tested.
- **−** **The strategy is not tradeable, and the harness cannot tell you that.** All three gates
  are green for a model whose long side loses money. This is the correct division of labour and it
  is worth stating plainly: parity protects the fidelity of a signal, never its value.
- **−** **The sample is small and covers one regime.** 208 days, one broadly declining market, two
  highly correlated coins. With a four-hour horizon on hourly bars the 7,954 scored rows contain
  roughly 2,000 non-overlapping observations across two coins that move together — call it a few
  hundred independent events. An AUC of 0.52 on that is inside the noise. The venue's ~5,000-candle
  retention is the binding constraint; a longer history needs a data source this repo does not have.
- **−** **Transaction costs are not modelled, only quoted.** No turnover, no spread, no slippage,
  no market impact, and — for a perp — **no funding**, which at Hyperliquid's baseline hourly rate
  is on the order of 1 bp per hour held and would by itself exceed the measured edge over a
  four-hour position. The fee numbers are the published schedule, not a measurement: nothing in
  this repo has ever been filled.
- **−** **The label is not the objective.** The model is fitted on the sign of the forward return
  and traded through a probability band that has to clear a cost hurdle the training never saw. A
  cost-aware label (a triple barrier, or a return threshold) would change what the model optimizes;
  it would also throw away rows and introduce its own selection, which is why it was not done here
  rather than because it is wrong.
- **−** **Feature parity compares the Python path against the Python path.** Both sides are
  `axon.features`, which is exactly the Boundary-B guarantee (`docs/03`: "there is only one
  function") and exactly why the number is zero. It says nothing about a future Rust
  implementation — that is `axon-model`'s bit-equivalence gate (ADR-0019), and it has not been run
  against this spec.
- **−** **Replay here is a recording, not a counterparty** (ADR-0018). `replay_bars` drives the
  strategy over a candle history: it proves what the strategy *would emit*, and nothing about what
  it would be filled at, what it would queue behind, or what the book would do when it arrived.
  No number in this ADR is a P&L, and the gross edge is an upper bound on one.
- **−** **No bar reaches Python yet.** `MdSlice` carries `quote`/`trade`/`snapshot` and no bar
  kind, so the Rust candle feed — which decodes, normalizes and caches `Candle` with a close-time
  `ts_event` — has no path to the Python `Bar` this strategy consumes. The far side is
  wired: the strategy runs under the real `axon.live.StrategyRunner` and puts a valid record on a
  signal ring (`test_the_strategy_reaches_the_signal_ring_through_the_live_runner`). The missing
  link is a bar record on the market-data ring, which is workstream N3's, and until it lands the
  live rung of the ladder cannot be attempted at all.
- **−** **The two halves stamp a bar one millisecond apart, and N3 must close that first.**
  `axon.strategies.data` stamps `T + 1 ms` — the instant the bar is final for ordering, which is
  what `axon_core::Candle::ts_event` documents — while `decode_candle`
  (`axon-provider-hyperliquid/src/ws/decode.rs`) stamps `close_time * MS_TO_NS`, i.e. `T`, the
  bar's last millisecond. Nothing joins the two today, so nothing is wrong today. The moment a bar
  record lands on the market-data ring, an `align_by_event_time` between Rust-stamped bars and this
  loader's history intersects to *nothing*: the two stamps for one bar are 1e6 ns apart and the
  hourly grid is 3.6e12 ns wide, so no pair ever collides. The gate then fails as "a parity gate
  over an empty feature matrix proves nothing", a long way from the cause. **The required edit is
  on the Rust side**: `ts_event: (data.close_time + 1) * MS_TO_NS`, with
  `decodes_candle_frame` updated to `1_700_000_060_000 * 1_000_000` and a test named for the
  failure — a bar stamped `T` ties with the trades printed inside its own final millisecond, so an
  event-time sort can hand a strategy the closed bar before the tick that closed it. Until it
  lands, `CLOSE_STAMP_OFFSET_MS` names the constant on the Python side so neither half is a magic
  `+ 1`.
- **−** **Hyper-parameters were never searched.** One configuration was chosen for its shape
  (depth 3, `min_child_weight=40` — small enough not to memorize a fold) and used. The sweep that
  would search them exists and was planned at $0.0095/$0.0107/$0.0150 (low/expected/high) for 36
  CPU tasks, inside a $1.00 per-job cap and $27.00 remaining. **The recommendation is not to spend
  it:** one point takes 2.8 s locally, so the whole grid is 100 seconds on this box. The job exists
  so that the version that genuinely does not fit here — multi-interval, multi-horizon, years of
  data — is a parameter change rather than a new integration. The operator decides.
- **−** **The committed fixture is 33 days.** It is enough to exercise the pipeline and the gates
  offline, and far too short to say anything about edge; the tests deliberately assert that the
  AUC is finite and never that it is good.
- **−** **A booster reloaded from its artifact does not pin its thread count.**
  `axon.models.inference.XgboostPredictor` leaves `nthread` at "every core", and a single-row
  prediction then costs **5.4 ms** against **0.22 ms** with `nthread=1` (8-core box) — the OMP
  fan-out dwarfs nine features of threshold traversal. Harmless at an hourly cadence, 25× at a
  tick cadence, and it is also the one thread-count in the serving path that `inference.py` pins
  for the ONNX session and not for this one. Verified bit-identical either way, so it is a latency
  bug and not a fidelity one. The fix belongs to `axon.models` and is recorded as a required edit
  rather than made here.
- **−** **Research data is mainnet; execution is testnet.** The two markets are not the same
  market, so a testnet shadow session can validate the *plumbing* of this strategy and never its
  edge.

See [03](../03-ml-fidelity-and-features.md) (training–serving skew, the thesis this implements),
[07](../07-parity-and-testing.md) (the ladder), [ADR-0015](0015-model-artifacts-and-registry.md)
(the artifact and registry used unmodified), [ADR-0016](0016-feature-spec-and-parity-gates.md)
(the spec and the three gates), [ADR-0017](0017-compute-offload-to-modal.md) (dry run before
spend), [ADR-0018](0018-event-capture-and-golden-replay.md) (why a recording is not a
counterparty), [ADR-0019](0019-native-rust-inference-backends.md) (the Boundary-A gate this spec
has not yet been through).
