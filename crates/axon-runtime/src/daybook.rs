//! The UTC day's equity baseline, on disk, because a daily limit a restart clears is
//! not a daily limit.
//!
//! [`axon_execution::LossLimiter`] judges two independent numbers: this session's own
//! bottom line, and the **day's**. The first is in memory and that is correct — a
//! session is a process. The second cannot be, and the reason is the failure it exists
//! to stop: a strategy that loses money and takes the process down with it comes back
//! with a fresh accounting and a fresh budget, so a crash-restart loop spends the daily
//! allowance once per restart. The bound has to remember something the process does not.
//!
//! What it remembers is deliberately the **smallest** thing that works: one number and
//! one date. Not the fills, not our P&L, not a ledger — the venue's own `accountValue`
//! at the first reading of the day. Everything else can be recomputed or is somebody
//! else's job; this is the only quantity that is both authoritative and unrecoverable
//! after the fact.
//!
//! ## Three things this deliberately does not do
//!
//! **It does not reconcile with our accounting.** The day figure is the venue's answer
//! and the session figure is ours, kept apart for the reason [ADR-0036] keeps
//! `PnlSnapshot::drift` a reported quantity: two views of one fact, and the moment one
//! corrects the other there is only one view left.
//!
//! **It does not fail a session when the file cannot be written.** A read-only disk is
//! a reason to say so loudly and keep trading under an in-memory baseline — which is
//! strictly the pre-existing behaviour — not a reason to refuse to start. The
//! degradation is counted and named, because a daily bound silently reduced to a
//! session bound is the quiet kind of wrong.
//!
//! **It does not roll over on anything but the calendar.** "Day" is the UTC day, which
//! is a question about the world and therefore a **named wall-clock exception** in the
//! same class as [`OrderTracker::set_session_start`] — no event time answers "is it
//! tomorrow yet". Rolling over on elapsed hours instead would make the bound depend on
//! when the process happened to start, and two sessions on one account would disagree
//! about which day it was.
//!
//! [ADR-0036]: https://docs.rs/axon
//! [`OrderTracker::set_session_start`]: axon_execution::OrderTracker::set_session_start

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use axon_core::Decimal;
use serde::{Deserialize, Serialize};

/// Milliseconds in a UTC day. Leap seconds are not represented in Unix time, so this
/// division is exact for every instant Unix time can express.
const MS_PER_DAY: u64 = 86_400_000;

/// What is persisted. One number and one date, and the date is the one the code reads —
/// there is no second machine-readable copy of it to drift from the human-readable one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DayState {
    /// `YYYY-MM-DD`, UTC. Parsed on load, so an operator can read and correct it.
    pub utc_date: String,
    /// The venue's `accountValue` at the first reading of that day.
    pub start_equity: Decimal,
}

/// Why a day baseline is not being persisted. Reported, never fatal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DayBookFault {
    /// No path was configured, so the baseline lives only in this process.
    NotPersisted,
    /// A path was configured and the file could not be read or written.
    Io(String),
}

impl fmt::Display for DayBookFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DayBookFault::NotPersisted => f.write_str(
                "no daily state path: the day bound resets on restart and is therefore \
                 a session bound wearing a day's name",
            ),
            DayBookFault::Io(e) => write!(f, "daily state file unusable: {e}"),
        }
    }
}

/// The day's baseline, loaded once and rolled over on the calendar.
#[derive(Debug)]
pub struct DayBook {
    path: Option<PathBuf>,
    state: Mutex<Option<DayState>>,
    /// Set once, read by the status line. An `AtomicU64` of a discriminant rather than
    /// a second lock: the fault has to be reportable even when the state lock is the
    /// thing that went wrong.
    fault: Mutex<Option<DayBookFault>>,
    /// Rollovers observed. A session that runs through midnight should show exactly one.
    rollovers: AtomicU64,
}

impl DayBook {
    /// A book with nowhere to write. The day bound then covers this process only, and
    /// [`Self::fault`] says so.
    pub fn in_memory() -> Self {
        Self {
            path: None,
            state: Mutex::new(None),
            fault: Mutex::new(Some(DayBookFault::NotPersisted)),
            rollovers: AtomicU64::new(0),
        }
    }

    /// Load whatever is at `path`, or start empty.
    ///
    /// A file that does not exist is the ordinary first run and not a fault. A file that
    /// exists and cannot be parsed **is** one, and the baseline is discarded rather than
    /// guessed at: a corrupt baseline is a bound of an unknown size, which is worse than
    /// a bound that says it restarted.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let (state, fault) = match std::fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<DayState>(&text) {
                Ok(s) => (Some(s), None),
                Err(e) => (
                    None,
                    Some(DayBookFault::Io(format!(
                        "{} is not parseable: {e}",
                        path.display()
                    ))),
                ),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (None, None),
            Err(e) => (
                None,
                Some(DayBookFault::Io(format!("{}: {e}", path.display()))),
            ),
        };
        Self {
            path: Some(path),
            state: Mutex::new(state),
            fault: Mutex::new(fault),
            rollovers: AtomicU64::new(0),
        }
    }

    pub fn fault(&self) -> Option<DayBookFault> {
        self.fault.lock().ok().and_then(|f| f.clone())
    }

    pub fn rollovers(&self) -> u64 {
        self.rollovers.load(Ordering::Relaxed)
    }

    /// The baseline in force, if there is one.
    pub fn baseline(&self) -> Option<DayState> {
        self.state.lock().ok().and_then(|s| s.clone())
    }

    /// Feed the venue's latest `accountValue` and get the day's loss as a **magnitude**.
    ///
    /// `now_ms` is wall clock — the named exception this module's docs argue for.
    ///
    /// Returns `None` only on the reading that *establishes* a baseline, because at that
    /// instant the day's change is zero by construction and reporting a zero would be
    /// indistinguishable from having measured one. Every later reading in the same day
    /// returns `start_equity - equity`, which is positive when money has been lost.
    pub fn observe(&self, equity: Decimal, now_ms: u64) -> Option<Decimal> {
        let today = utc_date(now_ms / MS_PER_DAY);
        let mut guard = self.state.lock().ok()?;
        match guard.as_ref() {
            Some(s) if s.utc_date == today => Some(s.start_equity - equity),
            other => {
                if other.is_some() {
                    self.rollovers.fetch_add(1, Ordering::Relaxed);
                }
                let fresh = DayState {
                    utc_date: today,
                    start_equity: equity,
                };
                self.persist(&fresh);
                *guard = Some(fresh);
                None
            }
        }
    }

    /// Write the baseline out. A failure is recorded and swallowed — see the module
    /// docs for why a read-only disk must not take a trading session down.
    fn persist(&self, state: &DayState) {
        let Some(path) = &self.path else { return };
        let write = serde_json::to_string_pretty(state)
            .map_err(|e| e.to_string())
            .and_then(|text| {
                if let Some(dir) = path.parent() {
                    if !dir.as_os_str().is_empty() {
                        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                    }
                }
                std::fs::write(path, text).map_err(|e| e.to_string())
            });
        if let Err(e) = write {
            if let Ok(mut f) = self.fault.lock() {
                *f = Some(DayBookFault::Io(format!("{}: {e}", path.display())));
            }
        }
    }
}

/// `YYYY-MM-DD` for a count of days since the Unix epoch.
///
/// Hinnant's civil-from-days, which is exact for the whole representable range and
/// needs no dependency. A date crate would be a reasonable alternative and is not worth
/// it for one function on a path that runs once per status line — but the reason it is
/// hand-written is written down so the next person does not have to rediscover that it
/// is a known algorithm rather than an invention.
fn utc_date(days_since_epoch: u64) -> String {
    // Shift the era so the leap-day arithmetic works on a March-based year.
    let z = days_since_epoch as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    /// 2026-07-27T00:00:00Z, the day the run this module exists for happened.
    const DAY_START_MS: u64 = 1_785_110_400_000;
    const HOUR: u64 = 3_600_000;

    #[test]
    fn the_calendar_arithmetic_is_right_on_the_days_that_break_naive_versions() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(DAY_START_MS / MS_PER_DAY), "2026-07-27");
        // A leap day in a leap year, and the year-2000 rule that catches the
        // divide-by-four shortcut.
        assert_eq!(utc_date(11_016), "2000-02-29");
        assert_eq!(utc_date(11_017), "2000-03-01");
        // 1900 was not a leap year; 2100 will not be either.
        assert_eq!(utc_date(47_540), "2100-02-28");
        assert_eq!(utc_date(47_541), "2100-03-01");
    }

    #[test]
    fn the_first_reading_of_a_day_sets_the_baseline_and_measures_nothing() {
        // Zero would be indistinguishable from a measured zero, and the difference
        // matters to a bound: "we have not started measuring" and "we have measured no
        // loss" send an operator to different places.
        let b = DayBook::in_memory();
        assert_eq!(b.observe(dec!(1000), DAY_START_MS), None);
        assert_eq!(b.baseline().unwrap().start_equity, dec!(1000));
        assert_eq!(b.observe(dec!(999.5), DAY_START_MS + HOUR), Some(dec!(0.5)));
        assert_eq!(
            b.observe(dec!(1002), DAY_START_MS + 2 * HOUR),
            Some(dec!(-2)),
            "a gain is a negative loss, not an absence of one"
        );
    }

    #[test]
    fn a_restarted_process_inherits_the_days_baseline_from_disk() {
        // The failure this whole module exists for: a strategy that loses money and
        // takes the process down comes back with a fresh accounting, so a crash-restart
        // loop would spend the daily allowance once per restart.
        let dir = std::env::temp_dir().join(format!("axon-daybook-{}", std::process::id()));
        let path = dir.join("day.json");
        let _ = std::fs::remove_file(&path);

        let first = DayBook::load(&path);
        assert_eq!(first.observe(dec!(1000), DAY_START_MS), None);
        assert_eq!(first.observe(dec!(995), DAY_START_MS + HOUR), Some(dec!(5)));
        drop(first);

        // A new process, the same day. It must not re-baseline at 995 and report the
        // day as flat.
        let second = DayBook::load(&path);
        assert_eq!(
            second.observe(dec!(995), DAY_START_MS + 2 * HOUR),
            Some(dec!(5)),
            "the five lost before the restart is still lost"
        );
        assert_eq!(second.rollovers(), 0);
        assert!(second.fault().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn midnight_rolls_the_baseline_over_and_says_it_did() {
        let b = DayBook::in_memory();
        b.observe(dec!(1000), DAY_START_MS);
        assert_eq!(b.observe(dec!(990), DAY_START_MS + HOUR), Some(dec!(10)));
        // The next UTC day: yesterday's loss is not today's.
        assert_eq!(b.observe(dec!(990), DAY_START_MS + 24 * HOUR), None);
        assert_eq!(b.baseline().unwrap().utc_date, "2026-07-28");
        assert_eq!(
            b.observe(dec!(989), DAY_START_MS + 25 * HOUR),
            Some(dec!(1))
        );
        assert_eq!(b.rollovers(), 1);
    }

    #[test]
    fn a_book_with_nowhere_to_write_says_so_rather_than_pretending_to_be_a_day_bound() {
        // A daily bound silently reduced to a session bound is the quiet kind of wrong:
        // every number on the status line is right and the guarantee is not the one the
        // operator configured.
        let b = DayBook::in_memory();
        assert_eq!(b.fault(), Some(DayBookFault::NotPersisted));
        assert!(b.fault().unwrap().to_string().contains("resets on restart"));
    }

    #[test]
    fn a_corrupt_baseline_is_discarded_rather_than_guessed_at() {
        // A bound of an unknown size is worse than one that says it restarted.
        let dir = std::env::temp_dir().join(format!("axon-daybook-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("day.json");
        std::fs::write(&path, "{ this is not json").unwrap();

        let b = DayBook::load(&path);
        assert!(b.baseline().is_none());
        assert!(
            matches!(b.fault(), Some(DayBookFault::Io(_))),
            "{:?}",
            b.fault()
        );
        // …and it recovers by writing a fresh one rather than refusing to run.
        assert_eq!(b.observe(dec!(1000), DAY_START_MS), None);
        assert_eq!(b.observe(dec!(999), DAY_START_MS + HOUR), Some(dec!(1)));
        assert_eq!(
            DayBook::load(&path).baseline().unwrap().start_equity,
            dec!(1000),
            "the repaired file is readable by the next process"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
