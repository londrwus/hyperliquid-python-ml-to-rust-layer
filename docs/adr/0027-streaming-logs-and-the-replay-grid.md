# ADR-0027 — A log that outlives a soak, and a log that carries its own grid

**Status:** Accepted · **Date:** 2026-07-26

## Context

Two problems, one file format, one version bump. They are written together because they
are not separable: both change what a log *is*, and landing them apart would invalidate
every stored log twice.

### 1. A recording could not survive a long session

ADR-0018 §1 accepted "a log is loaded whole" as a consequence and drew the honest
conclusion from it: event-time ordering cannot begin until the last record has been seen,
so log size is bounded by memory. §9 then set the capture's size cap **where the artifact
stops being loadable rather than where the disk stops being writable** — 512 MiB — and
said out loud that a soak that needs more wants segments, and that stitching them needs
`EventLog` to stream.

That bill came due. A multi-hour capture against testnet is the artifact this whole crate
exists for, and it hits the cap and stops. Nothing lies about it — `CAPTURE STOPPED (size
cap)` reaches the status line, `events + missed` still accounts for the whole session, and
the log keeps its `.partial` name so no harness picks it up — but the tape ends, and the
one recording anybody wanted is the prefix.

### 2. A replay planned prices the session it reproduces could not have sent

ADR-0025 gave the planner a venue grid and made the encoder refuse anything off it. It
also named the hole its own increment made **bigger**, in "Out of scope" item 2: the plan
is now a function of a network-fetched input the log does not carry. Before it, the hole
was "a replay of a capture taken after a re-precision plans a different *size*". After it,
the hole is "a replay of **any** live capture plans a different **price** on every order
the grid moved" — urgency-3 slippage, every band — because the session rounded and the
replay does not.

`PlannedOrder.price` is compared **exactly** by `python/axon/backtest/golden.py`, and
`docs/07` makes that diff promotion gate #5. So the divergence surfaces as a strategy flip
inside the harness whose one job is to tell a strategy change from a harness change. That
is worse than an absent gate: it is a gate that fails for an artefact, and a gate that
fails for an artefact gets waived.

Five questions, and each has an answer that reads as obviously right until it costs the
thing the format exists for.

1. **Does streaming remove the memory bound, or move it?** "Read a line at a time" sounds
   like O(1), and for one of the two traversals it is a lie.
2. **Where does the instrument table live in the log?** In the header is the obvious
   place — it is already the "what this file is" line.
3. **What does a v1 log do under a v2 reader?** Refusing costs every stored log. Accepting
   costs the thing the bump was for.
4. **Who tells a replay which grid to plan on — the log, or its caller?** ADR-0025 §2's
   move was to make the caller name it. That argument does not obviously survive the log
   being able to answer.
5. **Does the committed fixture declare a grid?** It is the artifact the Python golden gate
   diffs, so whatever it declares becomes an assertion about a venue.

## Decision

### 1. A log is read one record at a time, and the bound *moves* rather than disappearing

`LogReader` is the one thing in the crate that turns a line into a record. `EventLog` — the
whole-file form ADR-0018 shipped — is now that reader drained into a `Vec`, so there is one
parser and not two. Two readers would be two opinions about what a log says, and the drift
would land inside the harness built to detect drift.

What streaming costs depends on the traversal, and this is the part of ADR-0018's reasoning
that survives intact rather than being overturned:

- **`ReplayOrder::AsCaptured` is O(1).** The file already *is* the order, so the producer
  reads a line, sends it, and forgets it. Nothing is held. This is also the only order a
  live-captured reference may legitimately be compared under (ADR-0018 §4), so the
  traversal a real soak tape needs is exactly the one that costs nothing — which is why
  this is worth doing at all rather than being a nice property of the wrong case.
- **`ReplayOrder::EventTime` holds an index**: `(ts_event, byte offset)` — 16 bytes of
  key and offset, ~24 measured once `TimedQueue`'s own bookkeeping is counted — built by
  one scan and consumed by a second, seeking pass. It cannot do better,
  and the reason is ADR-0018's own: a feed that can be late can be late by any amount, so
  ordering cannot emit its first event until it has seen the last record, and a bounded
  lookahead would be a guess. **The bound did not go away. It went from the size of the
  events to the size of a key** — from a few hundred bytes a record to two dozen — which
  is the difference between a session slice and a night of tape.

Two properties are preserved deliberately, and both cost a full extra pass:

- **`ReplaySource::open` scans the whole file before anything is dispatched.** Streaming's
  real risk is that a malformed record is discovered *mid-run*, after handlers have already
  seen a prefix — and a harness reporting success over a prefix is precisely the artifact
  ADR-0018 §9 built the `.partial` rename to prevent, one layer down. The scan buys back
  `EventLog::open`'s old behaviour: a corrupt log is an error at open, with a line number,
  before a single event reaches a handler.
- **The event-time index keys on `ts_event` alone and gets its tie-break from the core's
  own `TimedQueue` push order**, which *is* capture order. Reusing the core primitive
  rather than a local sort is ADR-0018 §4's rule, and it is what makes `seq` unnecessary in
  the index: a book snapshot and the trade that produced it cannot swap on one run in a
  thousand.

`ReplayReport` gains `dispatched`. It equals `events` on every healthy run; smaller means
the file changed underneath the replay between the scan and the pass, and what the handler
saw is a *prefix* of what the report describes. A shortfall that could only be inferred by
comparing two other numbers is a shortfall nobody notices — the same reasoning that made
`events + missed` an identity at the writing end.

### 2. The size cap becomes a disk guard, and there is still no rotation

`max_bytes` no longer answers "how large can an artifact be and still be loaded"; it
answers "how much of this volume may a recording take from the trading session that shares
it". `max_bytes == 0` turns it off, for an operator who has decided the answer is "all of
it". The message it prints when it fires says which knob to turn, because the old one told
an operator a fact about `EventLog` that is no longer true.

**Rotation stays out of scope, and for ADR-0018's reason rather than the number it was
attached to.** A set of rotated segments is a set of files each of which replays a
*different* session from the one recorded, and stitching them in event time is a merge
nothing in the harness performs. Streaming removed the *loadability* argument for a small
cap; it did not turn segments into a session.

### 3. The instrument table is the log's first body line, and every line is tagged

Every body line is now a `LogLine` — `Instruments(InstrumentSet)` or `Event(LogRecord)` —
and the first one is the instruments. Three choices, stated against their alternatives:

- **Not in the header.** The header is what `head -1` shows and what a post-mortem reads
  first, and testnet publishes 210 perps. A 17 KB table there makes the one human-readable
  line of the file unreadable. It is also shared with the signal log (`LogHeader::for_schema`),
  and an event-log-only field on a shared type is how the two formats start disagreeing.
- **A tagged enum, not a positional second line**, even though the tag costs about ten
  bytes on every event line (~5% on a real log). It buys exactly what ADR-0018 §1 bought
  for `LoggedEvent`: the match is *exhaustive*, so a body-line kind added later cannot be
  quietly dropped by a reader written before it. A positional rule would put the format's
  shape in prose, which is the same class of promise `SCHEMA_VERSION` already asks a human
  to keep once. ADR-0018 already made this trade explicitly — JSONL over a packed frame,
  paying bytes for readability — and 5% is small next to what that choice already spends.
- **`InstrumentSet` has three states, not two.** `Undeclared` says *this recording carries
  no grid*; `Declared { unconstrained: true, .. }` says *this venue has no grids*;
  `Declared { unconstrained: false, instruments }` says *these are the grids*. That is
  ADR-0025 §1's three-variant `Precision` written down, and collapsing any pair of them is
  how a backtest becomes silently more permissive than the session it claims to reproduce.
  In particular, `Undeclared` must not be recorded as an empty table: an empty
  `InstrumentTable` means every symbol is `Precision::Unknown`, which refuses every order
  that would add exposure — so a replay of it would report the session's own decisions as
  precision refusals, and a session refused all day is indistinguishable from one that
  chose to be flat (ADR-0025 §4's argument, one layer out).

The price grid is written as **the two numbers it is** — `price_increment` and
`price_sig_figs` — and not as an enum. ADR-0025 §1 argued a two-variant `{ Digits, Grid }`
shape is a venue leak wearing a port's clothes, because venue three adds arm three; a log
that reintroduced the enum would reintroduce the leak one layer down, and would need a new
arm *and* a `SCHEMA_VERSION` bump for a venue the port itself handles by setting a field.

`Capture::with_header` takes the set as a **required argument**, which is `order_wire`'s
move again (ADR-0025 §2): a writer cannot record a session without naming what that session
knew about the grid, and `Undeclared` is a word somebody has to type.

### 4. A v1 log is **refused**, by name, and the message says what to do

This is the decision with a real cost either way, and it is made in the direction that
refuses to keep the bug alive under a new version number.

Accepting a v1 log means planning it unconstrained. That is *exactly* the divergence the
bump exists to end — a golden diff of a real capture reporting a strategy flip on every
rounded order — still happening, now over a file that looks current, with a warning on
stderr. The status quo already prints such a warning on every `replay_log` run, and the
whole point of this ADR is that a warning was not enough: `docs/07` gate #5 compares
`price` exactly and does not read stderr. A "loud accept" is a quiet accept with extra
words.

The cost of refusing is stated plainly: **every log captured before this change is
unreplayable**, and the bill arrived within the hour. A live testnet capture taken earlier
in the same session (`data/captures/m2-venue-proof.jsonl`, 171 events of real Hyperliquid
traffic including twelve fills) is a v1 log, and the reader now refuses it by name. That
is the decision working, and it is also the only real tape anybody had.

So the refusal gets an **escape hatch that is a command, not a fallback**:
`--example upcast_log` rewrites a v1 log as a v2 one declaring
`InstrumentSet::Undeclared`. It invents nothing — a v1 log genuinely carried no grid, so
"undeclared" is the only true statement about it — and the result is loud on every replay
in exactly the way this ADR insists on. The distinction that matters is *who decides*: a
reader that upcast silently would put every stored log back on the loose path, which is
the state the bump exists to end, while an operator naming one file has decided for that
file. It goes through the real `Capture` writer, so every check the v2 reader applies is
applied on the way through and a corrupt v1 log cannot be laundered into a well-formed v2
one.

**The signal log has no honest upcast, and `upcast_log` does not offer one.**
`axon.signallog` went 1 → 2 for an unrelated reason (§4b), and a v1 record has no value
for `max_order_age_ms`. The value a naive upcast would write — zero — is not "absent": it
means *defer to the operator's ceiling*, so an upcast signal log replays with every order
carrying a lifetime the live session never set, and orders are pulled on age where the
session left them resting. That is a changed decision wearing a migration's clothes. An
upcast event log therefore replays with `--no-signals`, which re-observes a session
faithfully and does not re-decide it; getting both halves means re-capturing.

The refusal is a variant of its own, `ReplayError::LogPredatesInstruments`, and not the
generic `IncompatibleVersion`. The generic message says "replay it with the build that
wrote it", which sends an operator to look for a build; the real fix for a v1 log is to
re-capture or to upcast, and the message says so.

Two smaller refusals fall out of the same instinct. A v2 log whose first body line is *not*
the instruments is refused (`MissingInstruments`) rather than read on with the table
defaulted — a log that claims to carry a grid and does not is the single artifact this
version exists to make impossible. And a *second* instruments line is refused
(`RepeatedInstruments`), because `cat a.jsonl b.jsonl > both.jsonl` is a thing an operator
does at 03:00, and the result parses line by line and replays green over a session whose
`seq` restarts in the middle and whose halves were recorded against different grids.

### 4b. Two version bumps landed in the same session, and they are different bumps

A reader that hits both at once deserves to be told which is which, because the causes
have nothing to do with each other and the remedies differ.

- **`SCHEMA_VERSION` 1 → 2 (the event log, `axon.eventlog`) is a *log-format* change.**
  Nothing about an `Event` moved. What changed is the shape of a *file*: body lines are
  tagged and the first one is the instrument table. This ADR is its cause, and §4 above
  is what a v1 log now does.
- **`SIGNAL_SCHEMA_VERSION` 1 → 2 (the signal log, `axon.signallog`) is a *wire-record*
  change, and not ours.** `contracts/schema.toml` carved `max_order_age_ms` (and its
  explicit `pad0`) out of `Signal`'s reserved bytes, which drop from 15 to 8 — an order
  lifetime distinct from `ttl_ms`, which ADR-0020 §4 named as missing. `LoggedSignal`
  mirrors `Signal`, so the log had to follow it, and the constant on
  [`crate::signals`](../../crates/axon-replay/src/signals.rs) says which of the two it is
  versioning. The bump is *required* rather than tidy: a v1 record decoded into the v2
  shape would default the new field to zero, and zero on that field means "defer to the
  operator's ceiling" rather than "absent" — a replayed order silently given a lifetime
  nobody set.

They coincide, they do not compound: one file format and one record format, each with its
own constant, its own pinned wire test, and its own refusal. Both committed fixtures were
regenerated for their own reason, in one command, because a signal is a decision *about* a
market moment and the two halves have to be generated together (`make_fixture_log`).

While regenerating them, two version guards were found to have quietly stopped guarding.
`a_log_written_by_an_incompatible_build_is_refused_not_reinterpreted` — in both the event
log and the signal log — substituted a **hard-coded** "impossible future version" into the
header. The day the constant reaches that literal, the substitution matches nothing, the
test reads a log this build accepts, and it passes while asserting nothing. Both are now
written against `SCHEMA_VERSION + 1` with an `assert_ne!` proving the substitution
happened, because a guard that silently stops guarding after a version bump is precisely
the failure this whole ADR is about.

### 5. The log's own table is the default; a caller's is an override

`ChainOptions::instruments` becomes `Option<Arc<InstrumentTable>>`. `None` — the default —
means *use the table the log declared*, which is the only setting that reproduces the
session's own prices. `Some` is for a caller who knows better than the log: a test
comparing two grids, or a recovered grid for a recording that predates this format.

The direction matters and is not symmetric. Silently preferring the argument would make
every default caller round on `ChainOptions`' idea of a venue instead of the recording's,
which is the bug. Silently preferring the log would make the override useless the day a
grid has to be supplied from outside. So the log wins by default and the caller wins by
saying so.

ADR-0025 §2's move survives where it belongs: **`ChainProbe::new` still takes the table as
a required argument.** The *driver* resolves where a grid comes from; the probe still
cannot be constructed without naming one, because `replay_chain` is one caller of many and
a probe that can be built without a grid is a probe that will be.

Whichever grid is used, `replay_log` says so on **stderr** — the log's own, a caller's
override, a log that declares a venue with no grids at all, or a log that declares nothing.
Stderr and not the summary object, because `axon.backtest` parses stdout and the summary is
a compared artifact that a `RESULT_SCHEMA_VERSION` bump of its own would have to land with.
That half of ADR-0025 §out-of-scope 2 is **still open** and is named again below.

### 6. The committed fixture declares a venue with no grids, and that is the true sentence

`make_fixture_log` writes `Declared { unconstrained: true, instruments: [] }`. The fixture
is a synthetic session with no venue: nothing published a tick, so "this venue imposes no
price grid" is what actually happened. `Undeclared` would mean "a grid existed and we
failed to record it", which describes a different bug.

A *fabricated* Hyperliquid grid would be worse than either. The fixture is the artifact
`python/tests/test_backtest.py` diffs, so inventing tick sizes for it would turn a dozen
committed assertions into statements about this author's guess at a venue — and it would
change numbers in a file owned by a different workstream for no gain in evidence. The
grid-driven path is exercised where it is real instead: by
`a_log_that_declares_a_grid_replays_at_the_prices_that_grid_produces`, which re-emits the
same session with a real venue's shape attached and asserts the urgency-3 flatten floors
to `49757` — the price the session sent — with **no** caller help at all.

### 7. The `as-captured` traversal gets a golden of its own

It had none. The two traversals were only ever compared against each other, so a change
that moved both moved silently — and `as-captured` is the order ADR-0018 §4 says a
live-captured reference must be compared under, which makes it the order every real soak
tape will be replayed in. `the_as_captured_traversal_has_a_golden_of_its_own` pins its
orders by value.

### 8. The ordering metric's subject is the market data the venue timestamped

`late_arrivals` counts every place capture order disagrees with event-time order. That is
the right answer to *"do the two traversals differ"* — which is what ADR-0018 §4 needs and
what the golden summary carries — and it is a **terrible** answer to *"is the venue's feed
delivering out of order"*, which is the question anybody actually asks it. A 1 h 44 m
testnet soak across 36 deliberate disconnects settled the size of the gap: **15.94% of its
records were "late", and 0.08% of the venue-stamped market data was.**

Three records inflate it, and none of them is a fault:

1. **A bar's stamp describes a window, not a moment.** `ts_event` is `open_time +
   interval` — the moment the bar *will* close — and the venue republishes the bar it is
   still filling: 1 321 frames for 69 bars over 12.9 minutes, 1 317 arriving before their
   own `T`, one bar republished 192 times. One frame walks the high-water mark up to a
   whole interval into the future.
2. **`activeAssetCtx` carries no venue timestamp at all**, so a `Ticker` is ordered on our
   own receipt clock (ADR-0011) and *always* sits one network hop ahead of the
   venue-stamped data around it. It was **65% of the soak tape.**
3. **`userFills` replays its whole snapshot on every reconnect**, carrying execution times
   up to **2.94 hours** stale. Those records really are old and really did arrive now — but
   a snapshot being re-sent is not a feed delivering late, and counting it as one makes a
   reconnection test fail itself.

An earlier version of this section subtracted (1) from `late_arrivals` and called the
residue "reordered by the feed". **That was wrong, and the tape proved it**: on the candle
tape the residue was 409 and the truth was 0. A residue inherits every misattribution its
terms can make, and there were two more terms nobody had written down.

So the fix is not a third and fourth subtraction. **The metric gets a subject.**
`StampSource` classifies each record by what its `ts_event` is *evidence of* —
`Venue`, `Derived { window_start }`, or `Execution` — and the venue-stamped records are
ordered **against each other**, with their own high-water mark, as a second pass over a
subset. It cannot misattribute, because a record carrying nothing but a receipt stamp is
not in the subset to be blamed *or* excused. It is exact for its subject rather than
loose, and it is the only ordering claim about a venue this format can honestly make.

Three subtractions would each have named a real venue behaviour, which is the argument for
them — so those behaviours are still named, in `StampSource`'s own variants and in
`behind_derived_stamps`, which stays as the *explanation* of `late_arrivals`' size. What
changed is that the number an operator alarms on stopped being their arithmetic.

The classification match over `MarketEvent` is **exhaustive**, unlike the other diagnostics
in this crate: it decides what a published measurement about a venue is measured over, and
a new market event silently defaulting into or out of that subset would change what the
headline means with nothing anywhere to report it. `Ticker` is classified at *runtime*
through the core's own `Ticker::is_venue_timed`, not by a copy of the rule, so a venue that
starts stamping the frame moves into the measured subset with nothing here changing.

Three things this deliberately gives up, and they are the price of the number meaning one
thing:

- **A reordering inside the excluded records is invisible.** Bars, receipt-stamped tickers
  and the whole execution stream are out of the measurement. If Hyperliquid started
  delivering `userFills` genuinely out of order, this would not say so.
- **`behind_derived_stamps` is loose in one direction, and now much looser than when it
  only had to bound a bar.** A bar is blamed only for records inside its own window; a
  receipt stamp has *no* window and so excuses whatever follows it — and that is the 65%
  case. It explains a number; it is not evidence.
- **`late_arrivals` is now a number nobody should read alone.** It is kept whole because it
  is in the golden summary and in `python/axon/backtest`, and because it is still the right
  answer to its own question.

This is also, as the candle work already noted, a further argument against a
`Candle::closed` flag — and the soak widened it. A flag would have fixed neither the
receipt-stamped ticker nor the snapshot replay, because the problem was never a missing
field on one record type. It was that the metric had no subject.

## Verification

Measured, on this machine, against a **real** Hyperliquid testnet tape
(`data/captures/m2-venue-proof.jsonl` — 171 events, 21 book snapshots, 106 tickers, 16
BBOs, 12 fills, 4 order updates, 12 account snapshots, BTC at testnet asset index 3),
upcast to v2 and replayed through the production chain.

- **The bottom rung holds on real traffic, not only on a fixture.** Two replays of that
  tape produce a byte-identical 82 KB trace and a byte-identical summary. Before this,
  every claim of that kind rested on `session.jsonl`, which a generator wrote.
- **A real feed inverts far more often than the fixture does, and almost none of it is
  the feed.** 25 of 171 records arrived behind the event-time high-water mark — 15%,
  against the fixture's single deliberate one — and the two traversals produce visibly
  different traces from it. ADR-0018 §4's insistence that a live-captured reference may
  only be compared under `as-captured` was an argument; on this tape it is an observation.
  **An earlier draft of this line stopped there, and was wrong to.** All 25 sat behind a
  receipt-stamped ticker; the venue's own market data on that tape was **0 of 37** out of
  order (§8). The conclusion survives — the traversals differ, which is all §4 needs — but
  the 15% was measuring our own receipt clock, not Hyperliquid.
- **Streaming is O(1) in the order that matters.** A 163 MiB log of 410 400 records
  replays `as-captured` in **4.8 MB** of RSS, and at 27 MiB / 68 400 records in 4.5 MB —
  flat across a 6× size change. `event-time` over the same file peaks at **14.7 MB**,
  i.e. about 24 bytes a record for the index and its heap. Under ADR-0018's reader the
  same file would have been held whole.
- **The trace was the remaining bound, and it is now opt-in.** With `--trace` the same
  run peaks at 141 MB, because a `ChainRow` carries two `BTreeMap`s and is far larger
  than the record it describes. `replay_log` retains rows only when asked
  (`ChainOptions::keep_trace`); before that it retained them always, so a plain replay of
  a soak tape would have died holding a trace nobody requested.
- **A session's own grid survives the round trip.** With the two-line `session.rs` wiring
  applied, `axon --capture` writes
  `{"Instruments":{"Declared":{"unconstrained":false,"instruments":[…]}}}`, `replay_log`
  reports *"planning on the grid this log declared (2 instruments)"*, and the replayed
  orders — sides, sizes, limits, `cloid`s and the cancel's venue-id target — are identical
  to the intents the session printed.

What is **not** verified: none of this was replayed against a *live* session running at
the same time, and no capture taken by a session that both declared a grid and traded a
venue has been replayed — the grid-declaring wiring and the live tape belong to two
different runs. The numbers a real venue would settle are still settled only by
`#[ignore]`d tests.

## Consequences

- **+** A recording can outlive a soak. The cap that stopped one is a disk guard now, off
  by configuration, and the replay that made it necessary reads a line at a time.
- **+** The traversal a real capture must be replayed under (`as-captured`) costs O(1)
  memory, and now has a golden.
- **+** A replay of a log that declares its grid plans the prices the session sent, with no
  caller help — which is what makes a golden diff of a real capture mean "the strategy
  changed" again.
- **+** One parser. `EventLog` is `LogReader` drained, so the streamed path and the
  loaded path cannot come to disagree about what a log says; a test asserts they do not.
- **+** No cycle is added. `axon-replay`'s library gains a **normal** dependency on
  `axon-providers` — a leaf over `axon-core`, which it already names — while the edge that
  has to stay a dev edge (`axon-runtime`) is untouched. ADR-0025 could claim "no dependency
  edge is added anywhere"; this cannot, and the edge is the price of the table being a port
  type rather than a copy.
- **−** **Every log captured before this is unreplayable**, including any this session's
  earlier work produced. That is §4's decision and its whole cost.
- **−** Event-time replay is still bounded by memory — measured at ~24 bytes a record
  rather than the record. A billion-event tape is ~24 GB of index and is still not
  replayable in event-time order. `as-captured` has no such bound, so the honest advice
  for a very long tape is to replay it in the order it was recorded, which is the order a
  live comparison requires anyway.
- **+** **A measurement about the venue that was previously buried.** Across a 1 h 44 m
  soak and 36 deliberate disconnects, Hyperliquid testnet delivered venue-stamped market
  data out of order **3 times in 3 876 records** — and 0 in 1 244 on the candle tape.
  Under `late_arrivals` alone those tapes read as 15.94% and 78.4%. `replay_log` prints
  both numbers and which is which.
- **−** A reordering inside the excluded records — bars, receipt-stamped tickers, the whole
  execution stream — is invisible to the metric. Each exclusion is argued in §8, and each
  is a real blind spot rather than a rounding of one.
- **−** `late_arrivals` is now a number nobody should read alone, which is a documentation
  burden a single counter did not have. It is kept whole because it is in the golden
  summary and in `python/axon/backtest`; splitting it into disjoint counters would have
  been cleaner and would have invalidated every stored reference for a diagnostic.
- **−** `behind_derived_stamps` is best-effort and loose in one direction: a bar is blamed
  only for records inside its own window, but a receipt stamp has no window and excuses
  whatever follows it — the 65% case. It explains `late_arrivals`; it is not evidence, and
  nothing in a log can settle the ambiguity, because the format carries no receipt time to
  compare a stamp against.
- **−** A retained trace is still the largest thing in a replay by two orders of
  magnitude, and `--trace` over a soak tape will exhaust memory. Streaming rows to the
  file as they are produced is the fix and is not done here: it means threading a writer
  into `ChainProbe`, which every golden test currently reads rows back out of.
- **−** Event-time replay now reads the file **three** times: the scan at open, the index
  pass, and the seeking pass. Two of those are pure cost on a small log, paid so that a
  large one is possible and so that a corrupt record is still an error before dispatch
  rather than halfway through it.
- **−** The seeking pass assumes the file has not moved. `ReplayReport::dispatched` is what
  says it did; nothing prevents it, and a log being rewritten under a running replay is not
  a case the format defends against.
- **−** Every event line is ~10 bytes wider for its `{"Event":…}` tag, about 5% of a real
  log. Bought deliberately, for an exhaustive match over body-line kinds.
- **−** **The runtime does not yet hand its table to the recorder.** `SessionRecorder::start_with`
  exists and is tested; `session.rs`'s two `SessionRecorder::start` call sites still use the
  undeclared form, because that file is owned elsewhere this session. Until those two lines
  change, a live capture records `Undeclared` — and says so at startup, on the status line,
  rather than silently. The exact diff is in the handoff.
- **−** `ChainSummary` still does not carry which grid a run planned on, so a stored golden
  reference cannot say whether it was produced with one. That is the other half of ADR-0025
  §out-of-scope 2 and it needs `RESULT_SCHEMA_VERSION` 3 → 4 plus the Python reader, which
  is a different owner's file and a separate reviewable change.
- **−** A grid is written as `(increment, sig_figs)` and rebuilt through
  `PriceGrid::from_parts`. That pins the log to the port type's *current* shape: a third
  number on `PriceGrid` — a band-varying tick, which ADR-0025 already names as not fitting
  — is a format change and a version bump, not an additive field.
- **−** `InstrumentTable::specs()` allocates and sorts on every capture start. Once per
  session, and the sort is load-bearing: `HashMap` iteration order is randomized per
  process, and two captures of one session must be byte-identical.
- **−** The fixture still declares no real grid, so the default gate proves the *wiring* of
  grid-carrying replay and not a venue's numbers — the same limitation `selftest.rs`'s
  offline grid already carries (ADR-0025 §consequences). Only a capture taken against a
  real venue settles the numbers, and none has been.
- **−** `examples/chain/mod.rs` still keeps two pass schedules (`poll_every_event` and the
  `due()`-gated fast path). Collapsing them would remove the last place the harness can
  differ from the runtime, and it was left alone for the reason it was left alone last time:
  it would make `a_pass_with_nothing_due_is_a_no_op_so_skipping_one_is_free` vacuous, and
  that test is the only thing currently pinning the cheap schedule to the faithful one.

See [ADR-0018](0018-event-capture-and-golden-replay.md) (the format this extends, and the
"loaded whole" consequence §1 revisits), [ADR-0025](0025-instrument-precision-and-rounding.md)
(the grid this carries, and the hole its own §out-of-scope 2 named),
[ADR-0012](0012-market-data-ring-and-multi-record-contract.md) (the drop-rather-than-stop rule
capture is the inverse of), [ADR-0004](0004-provider-abstraction-layer.md) (why the table is a
port type and not a venue one).
