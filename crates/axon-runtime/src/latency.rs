//! Declared latency **budgets**, and what a session did against them.
//!
//! Phase 6 asked for three things to be watched in a live session: P&L, parity and
//! latency budgets. Latency was the one that came back half-answered — every number
//! this project has about it was measured *as a by-product* of something else. The
//! Phase-5 parity run reported 53 bars arriving 1 067 / 9 406 / 62 074 ms after their
//! own close, and the honest reading of that was written into the roadmap: "there is
//! still no budget, only an observation."
//!
//! The difference is not pedantic. An observation says what happened; a budget says
//! what was *supposed* to happen, so a session can report how often it did not — and
//! a number nobody declared a ceiling for cannot regress, because there is nothing for
//! it to regress against.
//!
//! ## The three stages, and why exactly these
//!
//! Each stage is a span between two stamps that already exist. Nothing here adds a
//! clock read to the hot path that was not already being taken.
//!
//! | Stage | From | To | Clock |
//! |---|---|---|---|
//! | [`Stage::SignalAge`] | the strategy's decision (`Signal::ts_event`) | the pass that planned it | the core's **event** clock |
//! | [`Stage::SubmitAck`] | the submit call | the venue's ack | wall |
//! | [`Stage::DecisionToAck`] | the strategy's decision | the venue's ack | wall |
//!
//! **`SignalAge` is the one that already decides something.** It is the same quantity
//! the reader ages a record against, so a budget on it is a budget on the admission
//! window itself: breaches here are the population from which `expired` is drawn, and
//! a breach count that climbs while `expired` stays flat means the ceiling is set
//! wider than the strategy's own `ttl_ms`. It is measured in event time because that
//! is the clock the decision is made on; the producer's stamp being a wall clock is
//! the named exception documented in `axon.strategies.live_runner`, and it is exactly
//! why this can read negative-age records (the reader counts those as
//! `ahead_of_clock`) — a negative span is clamped to zero rather than dropped, because
//! a record from the future is a clock-skew observation, not a latency sample.
//!
//! **`SubmitAck` and `DecisionToAck` are wall clock, and that is a named exception**
//! in the same class as the dead-man's-switch deadline: neither orders anything. The
//! first is a network round trip and the second spans two processes — and both of the
//! stamps it spans are wall clock (the producer's, and ours at the ack), so it is a
//! same-clock difference despite crossing a process boundary.
//!
//! `DecisionToAck` is the end-to-end number, and it is the only one that answers the
//! operator's actual question: from the strategy deciding, how long until the venue
//! had the order. It is not the sum of the other two — the gap between them is the
//! queue, the pass schedule and the in-flight rule — which is why all three are kept.
//!
//! ## What a quantile here is, and is not
//!
//! Samples land in fixed logarithmic buckets, so `p50` and `p99` are reported as the
//! **upper edge of the bucket the quantile falls in** — an upper bound, named as one
//! ([`StageReport::p50_le_ms`]). The maximum is exact, because a maximum is a single
//! value and there is no reason to lose it. This repo's rule is to assert the number
//! rather than the bound; a bucketed quantile cannot honour that, so it does not
//! claim to. What it buys is a fixed-size, allocation-free, lock-free histogram that
//! the async edge and the core thread can both write to on the hot path.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Bucket upper bounds, in milliseconds. The last bucket is everything above the last
/// bound and reports as `>` that number.
///
/// Chosen against the measurements this project already has rather than as a round
/// decade sweep: the interesting region for a bar strategy on this venue is 1 s to
/// 60 s (a median 9.4 s bar age, a 62 s worst case), and the interesting region for a
/// venue round trip is 100 ms to 2 s (a ~0.2 s block commit, p99 0.5–0.9 s). Both are
/// resolved to within a factor of about two everywhere they matter.
const BOUNDS_MS: [u64; 15] = [
    1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 60_000, 300_000,
];

/// The spans a session measures. Fixed and exhaustive: a stage that can be added at
/// runtime is a stage nothing declares a budget for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// From the observation a decision answers — an m1 bar's own **close** — to the
    /// decision itself. `Signal::ts_cause` → `Signal::ts_event`.
    ///
    /// **The largest number in this system, and until schema version 3 it had no
    /// ceiling because it was not measurable from here.** A closed m1 bar reached the
    /// strategy 951 / 12 051 / **111 475** ms after its own close on 2026-07-27, and the
    /// only place that appeared was the producer's private transcript: a record carried
    /// the moment the producer *decided* and nothing about what it was deciding about, so
    /// a decision one second after a bar and one two minutes after it were the same
    /// record. `ts_cause` put the bar's close on the wire and this stage is what it is
    /// for.
    ///
    /// Two properties worth knowing rather than rediscovering. It is a **cross-clock**
    /// span — the venue stamps the cause and the producer stamps the decision — so it
    /// includes whatever skew exists between them; both are epoch nanoseconds, which is
    /// what makes the subtraction meaningful. And it is **additive with
    /// [`DecisionToAck`](Self::DecisionToAck)**: `cause` + `e2e` is bar-close to
    /// order-at-the-venue, the whole loop, which is the figure an operator actually means
    /// when they ask how far behind the market a strategy is.
    CauseToDecision,
    /// How old a decision was when the core planned it.
    SignalAge,
    /// The venue round trip on one placement.
    SubmitAck,
    /// Decision to order-at-the-venue, end to end.
    DecisionToAck,
}

impl Stage {
    pub const ALL: [Stage; 4] = [
        Stage::CauseToDecision,
        Stage::SignalAge,
        Stage::SubmitAck,
        Stage::DecisionToAck,
    ];

    /// The short name the status line uses. Kept to four characters so the stages fit in
    /// a block an operator reads at a glance.
    ///
    /// `CauseToDecision` is first because it is the largest and the earliest: reading the
    /// block left to right walks a decision from the bar that caused it to the order at
    /// the venue.
    pub const fn short(self) -> &'static str {
        match self {
            Stage::CauseToDecision => "bar",
            Stage::SignalAge => "sig",
            Stage::SubmitAck => "ack",
            Stage::DecisionToAck => "e2e",
        }
    }
}

/// One stage's counters. Every field is atomic because the three stages are written
/// from two threads — `SignalAge` on the deterministic core, the other two on the
/// tokio edge — and neither may take a lock to record a measurement.
#[derive(Debug, Default)]
struct StageCounters {
    buckets: [AtomicU64; BOUNDS_MS.len() + 1],
    count: AtomicU64,
    breaches: AtomicU64,
    max_ms: AtomicU64,
    /// Sum of samples, for a mean. Kept because the mean and a bucketed median
    /// disagreeing is itself informative — a long tail moves one and not the other.
    total_ms: AtomicU64,
}

impl StageCounters {
    fn record(&self, ms: u64, budget_ms: u64) {
        let idx = BOUNDS_MS
            .iter()
            .position(|b| ms <= *b)
            .unwrap_or(BOUNDS_MS.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(ms, Ordering::Relaxed);
        self.max_ms.fetch_max(ms, Ordering::Relaxed);
        // A budget of zero is "no ceiling declared", never "everything breaches" —
        // the same reading of zero `intent.max_order_age_ms` already has. Filtering it
        // out here rather than comparing against it makes the inverted reading
        // unrepresentable rather than merely avoided.
        if budget_ms != 0 && ms > budget_ms {
            self.breaches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The upper edge of the bucket holding the `q`-quantile, or `None` with no
    /// samples. `q` is in the same units the caller states it in: 50 for the median.
    fn quantile_le_ms(&self, q: u64) -> Option<u64> {
        let n = self.count.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        // Ceiling of q% of n, so the median of a single sample is that sample's
        // bucket rather than the empty one below it.
        let target = (n * q).div_ceil(100).max(1);
        let mut seen = 0u64;
        for (i, b) in self.buckets.iter().enumerate() {
            seen += b.load(Ordering::Relaxed);
            if seen >= target {
                return Some(BOUNDS_MS.get(i).copied().unwrap_or(u64::MAX));
            }
        }
        Some(u64::MAX)
    }
}

/// The session's declared ceilings and what it measured against them.
///
/// Shared by `Arc` between the core thread and the submit task, exactly like
/// [`SessionHealth`](crate::health::SessionHealth) and for the same reason: the
/// numbers are written where the event happens and read where the operator looks.
#[derive(Debug)]
pub struct LatencyBook {
    stages: [StageCounters; Stage::ALL.len()],
    budgets: [u64; Stage::ALL.len()],
    /// The fraction of samples that may breach before the status line says so, in
    /// percent. `0` disables the warning without disabling the measurement.
    breach_warn_pct: u64,
}

impl LatencyBook {
    /// `budgets` in milliseconds, indexed by [`Stage`]; `0` declares no ceiling for
    /// that stage, which still measures it.
    pub fn new(budgets: [u64; Stage::ALL.len()], breach_warn_pct: u64) -> Self {
        Self {
            stages: Default::default(),
            budgets,
            breach_warn_pct,
        }
    }

    /// A book that measures everything and declares nothing — the default for a
    /// session whose config has no `[latency]` table.
    ///
    /// Measuring anyway is deliberate: the numbers cost nothing, and the first thing
    /// anyone setting a budget needs is what the session actually does. A book that
    /// went dark without a config would make "declare a ceiling" require guessing one.
    pub fn undeclared() -> Self {
        Self::new([0; Stage::ALL.len()], 0)
    }

    #[inline]
    pub fn record(&self, stage: Stage, ms: u64) {
        let i = stage as usize;
        self.stages[i].record(ms, self.budgets[i]);
    }

    /// Record a span given in nanoseconds, clamping a negative one to zero.
    ///
    /// The clamp is the whole reason this exists beside [`Self::record`]. A signal
    /// stamped by the producer's wall clock can be *ahead* of the core's event clock
    /// — that is the ordinary case on this venue, counted as `ahead_of_clock` by the
    /// reader — and a saturating subtraction turns it into a 0 ms sample. Dropping it
    /// instead would silently exclude the fastest records from the distribution and
    /// make the median look worse than the session was.
    #[inline]
    pub fn record_span_ns(&self, stage: Stage, from_ns: i64, to_ns: i64) {
        let ns = to_ns.saturating_sub(from_ns).max(0) as u64;
        self.record(stage, ns / 1_000_000);
    }

    pub fn budget_ms(&self, stage: Stage) -> u64 {
        self.budgets[stage as usize]
    }

    pub fn report(&self, stage: Stage) -> StageReport {
        let c = &self.stages[stage as usize];
        let count = c.count.load(Ordering::Relaxed);
        StageReport {
            stage,
            budget_ms: self.budgets[stage as usize],
            count,
            breaches: c.breaches.load(Ordering::Relaxed),
            p50_le_ms: c.quantile_le_ms(50),
            p99_le_ms: c.quantile_le_ms(99),
            max_ms: (count > 0).then(|| c.max_ms.load(Ordering::Relaxed)),
            mean_ms: (count > 0).then(|| c.total_ms.load(Ordering::Relaxed) / count),
        }
    }

    /// Every stage, in a fixed order.
    pub fn snapshot(&self) -> LatencySnapshot {
        LatencySnapshot {
            stages: Stage::ALL.map(|s| self.report(s)),
            breach_warn_pct: self.breach_warn_pct,
        }
    }
}

/// One stage, flattened for printing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageReport {
    pub stage: Stage,
    /// `0` when no ceiling was declared for this stage.
    pub budget_ms: u64,
    pub count: u64,
    /// Samples above the budget. Always `0` when none was declared — an undeclared
    /// stage cannot be breached, which is precisely why declaring one matters.
    pub breaches: u64,
    /// Upper bound of the bucket the quantile falls in. `None` with no samples.
    pub p50_le_ms: Option<u64>,
    pub p99_le_ms: Option<u64>,
    /// Exact, unlike the quantiles.
    pub max_ms: Option<u64>,
    pub mean_ms: Option<u64>,
}

impl StageReport {
    /// Breaches as a percentage of samples, rounded down. `0` with no samples, which
    /// is the right answer: nothing has been late.
    pub fn breach_pct(&self) -> u64 {
        // `checked_div` rather than a guard on `count`, because clippy is right that
        // the two are the same test written twice — and the fallback is not an
        // arbitrary default: a stage with no samples has had nothing be late.
        (self.breaches * 100).checked_div(self.count).unwrap_or(0)
    }
}

/// Every stage, as the status line and the warning list read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencySnapshot {
    pub stages: [StageReport; Stage::ALL.len()],
    pub breach_warn_pct: u64,
}

impl LatencySnapshot {
    /// Whether anything at all has been measured. A session with no samples prints no
    /// latency block, for the same reason an absent intent source prints no intent
    /// block: zeros read as "fast", and "nothing has happened yet" is a different
    /// statement.
    pub fn any_samples(&self) -> bool {
        self.stages.iter().any(|s| s.count > 0)
    }

    /// Stages that breached their declared budget too often to be called healthy.
    ///
    /// Reported as a rate rather than a count because a session long enough to matter
    /// will breach *something* eventually, and a warning that fires on the first late
    /// order is one an operator learns to ignore by the second hour.
    pub fn over_budget(&self) -> Vec<String> {
        if self.breach_warn_pct == 0 {
            return Vec::new();
        }
        self.stages
            .iter()
            .filter(|s| s.budget_ms != 0 && s.count > 0 && s.breach_pct() >= self.breach_warn_pct)
            .map(|s| {
                format!(
                    "{} {}/{} OVER {}ms",
                    s.stage.short(),
                    s.breaches,
                    s.count,
                    s.budget_ms
                )
            })
            .collect()
    }
}

/// `lat sig 5000/25000·62074 12/53 | ack 250/500·812 0/6`
///
/// Per stage: the two quantile bounds, then the exact maximum after a `·`, then
/// breaches over samples. A stage with no samples is skipped rather than printed as
/// zeros, and a stage with no declared budget still prints its distribution — the
/// numbers are the input to declaring one.
impl fmt::Display for LatencySnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lat")?;
        let mut first = true;
        for s in self.stages.iter().filter(|s| s.count > 0) {
            if !first {
                write!(f, " |")?;
            }
            first = false;
            write!(f, " {} ", s.stage.short())?;
            match (s.p50_le_ms, s.p99_le_ms, s.max_ms) {
                (Some(p50), Some(p99), Some(max)) => write!(f, "{p50}/{p99}·{max}")?,
                _ => write!(f, "-")?,
            }
            if s.budget_ms != 0 {
                write!(f, " {}/{}", s.breaches, s.count)?;
            } else {
                write!(f, " n{}", s.count)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_with_no_declared_budget_measures_and_never_breaches() {
        // Zero is "no ceiling", never "already over". The inverted reading would put
        // every sample of every undeclared stage into `breaches`, so a session that
        // declared nothing would report itself as entirely late — and the warning an
        // operator would then learn to ignore is the one that matters.
        let book = LatencyBook::undeclared();
        book.record(Stage::SignalAge, 60_000);
        let r = book.report(Stage::SignalAge);
        assert_eq!(r.count, 1);
        assert_eq!(r.breaches, 0);
        assert_eq!(r.max_ms, Some(60_000));
        assert!(book.snapshot().over_budget().is_empty());
    }

    #[test]
    fn a_sample_exactly_on_the_budget_is_inside_it() {
        // The boundary is a ceiling, not an exclusive bound: an operator who declares
        // 2 000 ms means 2 000 ms is acceptable. Off by one here turns a session that
        // exactly meets its budget into one that breaches on every sample.
        let book = LatencyBook::new([0, 2_000, 0, 0], 1);
        book.record(Stage::SignalAge, 2_000);
        book.record(Stage::SignalAge, 2_001);
        let r = book.report(Stage::SignalAge);
        assert_eq!(r.breaches, 1, "only the 2001 breaches");
        assert_eq!(r.count, 2);
        assert_eq!(r.breach_pct(), 50);
    }

    #[test]
    fn a_record_stamped_ahead_of_our_clock_is_a_zero_sample_and_not_a_dropped_one() {
        // The producer stamps with its own wall clock and the core's event clock runs
        // 1.4-1.9 s behind the market, so a healthy live session emits records from
        // "the future" routinely. Dropping them would exclude the fastest half of the
        // population and report a median worse than the session's.
        let book = LatencyBook::new([0, 2_000, 0, 0], 0);
        book.record_span_ns(Stage::SignalAge, 5_000_000_000, 3_000_000_000);
        let r = book.report(Stage::SignalAge);
        assert_eq!(r.count, 1, "counted");
        assert_eq!(
            r.max_ms,
            Some(0),
            "…as zero, not as a negative or a huge u64"
        );
        assert_eq!(r.breaches, 0);
    }

    #[test]
    fn the_bar_stage_is_declarable_separately_and_is_additive_with_the_end_to_end_one() {
        // The two things the fourth stage had to buy. It gets its **own** ceiling, so an
        // operator can hold the producer to a number without loosening the runtime's; and
        // `bar` + `e2e` is the whole loop, bar-close to order-at-the-venue, which is the
        // figure an operator means when they ask how far behind the market a strategy is.
        // Folding it into `sig` would have produced one number nobody could act on: `sig`
        // is ring→pass and is this runtime's to fix, `bar` is bar→decision and is the
        // producer's.
        let book = LatencyBook::new([20_000, 1_000, 0, 25_000], 25);

        // The three figures the live run measured, and the one that breaches.
        book.record(Stage::CauseToDecision, 951);
        book.record(Stage::CauseToDecision, 12_051);
        book.record(Stage::CauseToDecision, 111_475);
        let bar = book.report(Stage::CauseToDecision);
        assert_eq!(bar.max_ms, Some(111_475), "exact, and it is the headline");
        assert_eq!(bar.breaches, 1, "only the 111 s bar is past 20 s");
        assert_eq!(bar.budget_ms, 20_000);

        // …and the stage beside it is untouched by any of that.
        book.record(Stage::DecisionToAck, 2_717);
        let e2e = book.report(Stage::DecisionToAck);
        assert_eq!(e2e.count, 1);
        assert_eq!(e2e.breaches, 0);
        assert_eq!(
            book.report(Stage::SignalAge).count,
            0,
            "and `sig` is not it"
        );

        // The short names walk a decision left to right, from the bar that caused it to
        // the order at the venue. Pinned, because the block is read at a glance and a
        // reordering would silently re-label every column.
        assert_eq!(Stage::ALL.map(Stage::short), ["bar", "sig", "ack", "e2e"]);
    }

    #[test]
    fn an_undeclared_bar_stage_is_still_measured() {
        // The same asymmetry every other stage has, and it matters most here: the first
        // thing anyone setting *this* ceiling needs is what the session actually does,
        // and nobody had that number from the runtime's side until the field existed.
        let book = LatencyBook::undeclared();
        book.record(Stage::CauseToDecision, 12_051);
        let r = book.report(Stage::CauseToDecision);
        assert_eq!(r.count, 1);
        assert_eq!(r.breaches, 0);
        assert_eq!(r.budget_ms, 0);
        assert!(book.snapshot().over_budget().is_empty());
    }

    #[test]
    fn the_quantile_is_the_buckets_upper_edge_and_the_maximum_is_exact() {
        // The distinction this whole module has to be honest about. 99 samples at
        // ~1 ms and one at 62 074 ms: the median is a bound (5 ms, the bucket edge),
        // and the maximum is the number itself. Reporting the max as a bucket edge
        // would lose the single most quotable figure a latency run produces.
        let book = LatencyBook::new([0, 0, 0, 0], 0);
        for _ in 0..99 {
            book.record(Stage::SubmitAck, 3);
        }
        book.record(Stage::SubmitAck, 62_074);
        let r = book.report(Stage::SubmitAck);
        assert_eq!(r.p50_le_ms, Some(5), "the bucket 3ms lands in");
        assert_eq!(r.max_ms, Some(62_074), "exact");
        assert_eq!(
            r.p99_le_ms,
            Some(5),
            "99 of 100 samples are still in that bucket"
        );
    }

    #[test]
    fn a_sample_past_the_widest_bucket_still_counts_and_reports_its_own_maximum() {
        // The overflow bucket. A session with one absurd sample must not silently
        // drop it — that sample is the incident.
        let book = LatencyBook::new([0, 1_000, 0, 0], 50);
        book.record(Stage::SignalAge, 3_600_000);
        let r = book.report(Stage::SignalAge);
        assert_eq!(r.count, 1);
        assert_eq!(r.breaches, 1);
        assert_eq!(r.max_ms, Some(3_600_000));
        assert_eq!(r.p50_le_ms, Some(u64::MAX), "past every declared bound");
    }

    #[test]
    fn the_warning_fires_on_a_rate_and_not_on_a_single_late_sample() {
        // A session long enough to be worth watching breaches something eventually.
        // A warning on the first breach is a warning that is permanently on by the
        // second hour, which is the same as no warning at all.
        let book = LatencyBook::new([0, 100, 0, 0], 25);
        for _ in 0..99 {
            book.record(Stage::SignalAge, 10);
        }
        book.record(Stage::SignalAge, 5_000);
        assert!(book.snapshot().over_budget().is_empty(), "1 % is not 25 %");
        for _ in 0..40 {
            book.record(Stage::SignalAge, 5_000);
        }
        let warnings = book.snapshot().over_budget();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with("sig 41/140 OVER 100ms"),
            "{warnings:?}"
        );
    }

    #[test]
    fn a_zero_warn_threshold_silences_the_warning_without_silencing_the_measurement() {
        // The two switches are separate on purpose: an operator who has not yet
        // decided what rate is acceptable still wants the numbers.
        let book = LatencyBook::new([0, 100, 0, 0], 0);
        book.record(Stage::SignalAge, 9_999);
        assert!(book.snapshot().over_budget().is_empty());
        assert_eq!(book.report(Stage::SignalAge).breaches, 1, "still counted");
    }

    #[test]
    fn a_session_that_measured_nothing_prints_no_latency_block() {
        // Zeros read as "fast". "Nothing has happened yet" is a different statement
        // and the status line has to be able to make it.
        let book = LatencyBook::undeclared();
        let snap = book.snapshot();
        assert!(!snap.any_samples());
        assert_eq!(snap.to_string(), "lat");
    }

    #[test]
    fn the_rendered_line_separates_a_declared_stage_from_an_undeclared_one() {
        // `breaches/samples` only means something against a ceiling. An undeclared
        // stage prints its sample count instead, so `0/53` can never be read as
        // "53 samples, none late" when nothing was ever declared late.
        let book = LatencyBook::new([0, 1_000, 0, 0], 0);
        book.record(Stage::SignalAge, 9_000);
        book.record(Stage::SubmitAck, 200);
        let line = book.snapshot().to_string();
        assert!(line.contains("sig 10000/10000·9000 1/1"), "{line}");
        assert!(line.contains("ack 250/250·200 n1"), "{line}");
    }
}
