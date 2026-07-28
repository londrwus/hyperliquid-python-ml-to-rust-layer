//! The funding cadence Hyperliquid does not put in the ticker frame, and how we
//! find out when it changes.
//!
//! `activeAssetCtx.funding` is the rate charged over **one funding period**, and the
//! frame never says how long that period is. [`Funding`](axon_core::Funding) refuses
//! to store a rate without its interval for a reason — a bare rate read on the wrong
//! cadence is off by whatever the ratio happens to be — so the adapter has to supply
//! the missing half from somewhere. Today it supplies a constant.
//!
//! That constant is the hazard this module exists for. Nothing on the ticker wire
//! contradicts it, so the day Hyperliquid moves to a four- or eight-hourly charge,
//! every expected-carry number silently scales by four or eight: no decode error, no
//! missing field, no gap in the data, just wrong answers that reconcile against
//! nothing. It is the same class of problem as the missing ticker timestamp in
//! [ADR-0011], and it gets the same answer — make the assumption *checkable* instead
//! of invisible.
//!
//! The venue does publish its own schedule, just not on the channel that needs it:
//! `POST /info {"type":"fundingHistory"}` returns one entry per funding period with
//! the venue's own `time`. Those timestamps *are* the cadence. [`decode_funding_cadence`]
//! measures it and [`FundingCadence`] states, in the type, whether the venue agrees
//! with the constant we stamp — and the live client cross-checks once per session
//! (see [`HyperliquidMarketData::verify_funding_cadence`](super::client::HyperliquidMarketData::verify_funding_cadence)).
//!
//! Deliberately measured from `fundingHistory` rather than from the `funding` entries
//! on `userEvents`: the latter only arrive while we *hold* a position, so an hour flat
//! is indistinguishable from an hour that no longer exists. `fundingHistory` is
//! market-wide and gap-free, which is what makes the measurement mean something.

use axon_core::Nanos;
use serde::Deserialize;

use super::decode::{DecodeError, MS_TO_NS};

/// The funding interval every [`Ticker`](axon_core::Ticker) this adapter emits is
/// stamped with: Hyperliquid charges hourly and publishes the hourly rate.
///
/// It is an **assumption**, not a decoded value, and it is named that way so a reader
/// cannot mistake it for something the venue sent. If it ever stops being true, no
/// frame changes shape and no field goes missing — carry just scales by the ratio and
/// keeps reconciling against nothing. [`decode_funding_cadence`] is the check that
/// makes that visible; do not change this number without re-running it.
pub const ASSUMED_FUNDING_INTERVAL_NS: Nanos = 3_600 * 1_000 * MS_TO_NS;

/// How far a measured period may sit from [`ASSUMED_FUNDING_INTERVAL_NS`] before the
/// cadence counts as *different*.
///
/// Real testnet spacing wanders by ~150 ms around the hour (the venue stamps when the
/// charge is applied, not when the period nominally ends), so the tolerance only has
/// to absorb jitter — it must stay far below the smallest cadence change worth
/// catching. A minute is three orders of magnitude above the observed noise and three
/// orders below an hour, so neither mistake is close.
pub const FUNDING_INTERVAL_TOLERANCE_NS: Nanos = 60 * 1_000 * MS_TO_NS;

/// How much history to measure over, in milliseconds.
///
/// Long enough to contain several periods even if the venue lengthens its interval,
/// short enough that a *recent* change is not diluted by the old cadence sitting in
/// the same window.
pub const CADENCE_WINDOW_MS: i64 = 12 * 60 * 60 * 1_000;

/// What the venue's own funding timestamps say about the interval we stamp.
///
/// The verdict is a type rather than a `bool` because "we could not tell" and "the
/// venue agrees" are different facts, and a caller that collapses them alarms on a
/// quiet market instead of on a changed venue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FundingCadence {
    /// Fewer than two funding periods in the window — nothing to measure a period
    /// *between*. Not a failure: a brand-new market has no history.
    Undetermined,
    /// The venue's measured period matches [`ASSUMED_FUNDING_INTERVAL_NS`].
    Agrees { measured: Nanos },
    /// The venue is funding on a different cadence than every `Ticker` is stamped
    /// with. Both numbers travel so a log line is self-contained.
    Differs { measured: Nanos, stamped: Nanos },
}

impl FundingCadence {
    /// The period actually observed, when there was enough history to observe one.
    pub fn measured(&self) -> Option<Nanos> {
        match *self {
            FundingCadence::Undetermined => None,
            FundingCadence::Agrees { measured } | FundingCadence::Differs { measured, .. } => {
                Some(measured)
            }
        }
    }

    /// The one thing worth alarming on. Deliberately false for
    /// [`Undetermined`](Self::Undetermined): an unmeasurable cadence is not evidence
    /// of a changed one, and treating it as such trains people to ignore the alarm.
    pub fn differs(&self) -> bool {
        matches!(self, FundingCadence::Differs { .. })
    }
}

/// One `fundingHistory` entry. Only `time` is read: the rate here is the *realized*
/// charge for a past period, while `activeAssetCtx.funding` is the current predicted
/// one, and quietly treating them as the same number would put a stale rate on a live
/// ticker.
#[derive(Deserialize)]
struct RawFundingPeriod {
    time: i64,
}

/// Measure Hyperliquid's funding period from a `POST /info {"type":"fundingHistory"}`
/// body and compare it with [`ASSUMED_FUNDING_INTERVAL_NS`].
///
/// The measurement is the **smallest** gap between consecutive periods, not the mean.
/// A venue outage or a market that was not yet listed leaves a long gap; a mean would
/// be dragged by it and report a cadence that never existed, while the smallest gap is
/// the tightest bound the data supports and is unmoved by a missing period. There is
/// no "is it a whole multiple of an hour?" rule on purpose: an eight-hourly venue
/// passes that test and is exactly the case worth catching.
///
/// Timestamps must be strictly increasing. A body that arrived in another order is
/// refused rather than measured, because `|Δt|` over a reordered list is not a period
/// and a plausible wrong number here is worse than an error.
pub fn decode_funding_cadence(raw: &str) -> Result<FundingCadence, DecodeError> {
    let periods: Vec<RawFundingPeriod> = serde_json::from_str(raw)?;
    let mut smallest: Option<Nanos> = None;
    for pair in periods.windows(2) {
        let gap = pair[1].time - pair[0].time;
        if gap <= 0 {
            return Err(DecodeError::Malformed("fundingHistory"));
        }
        let gap = gap * MS_TO_NS;
        smallest = Some(smallest.map_or(gap, |s: Nanos| s.min(gap)));
    }
    Ok(match smallest {
        None => FundingCadence::Undetermined,
        Some(measured)
            if (measured - ASSUMED_FUNDING_INTERVAL_NS).abs() <= FUNDING_INTERVAL_TOLERANCE_NS =>
        {
            FundingCadence::Agrees { measured }
        }
        Some(measured) => FundingCadence::Differs {
            measured,
            stamped: ASSUMED_FUNDING_INTERVAL_NS,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured verbatim off Hyperliquid **testnet** (`fundingHistory`, BTC,
    /// 2026-07-25). Kept unrounded: the ~150 ms wander around the hour is the whole
    /// reason the check has a tolerance, and a tidied fixture would hide it.
    const HISTORY_FRAME: &str = r#"[
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1784984400139},
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1784988000025},
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1784991600028},
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1784995200024},
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1784998800046},
        {"coin":"BTC","fundingRate":"0.0000125","premium":"0.0","time":1785002400011}]"#;

    /// Build a history whose periods are `gaps_ms` apart, starting from an arbitrary
    /// epoch — the absolute time is irrelevant, only the spacing is measured.
    fn history(gaps_ms: &[i64]) -> String {
        let mut t = 1_784_984_400_000i64;
        let mut entries = vec![format!(
            r#"{{"coin":"BTC","fundingRate":"0.0","premium":"0.0","time":{t}}}"#
        )];
        for gap in gaps_ms {
            t += gap;
            entries.push(format!(
                r#"{{"coin":"BTC","fundingRate":"0.0","premium":"0.0","time":{t}}}"#
            ));
        }
        format!("[{}]", entries.join(","))
    }

    const HOUR_MS: i64 = 3_600_000;

    #[test]
    fn sub_second_venue_jitter_is_not_read_as_a_cadence_change() {
        // The real capture: hourly to within ~150 ms. An exact-equality check would
        // cry wolf on every session and get switched off, which is how the genuine
        // change later goes unnoticed.
        let cadence = decode_funding_cadence(HISTORY_FRAME).unwrap();
        assert!(!cadence.differs(), "got {cadence:?}");
        let measured = cadence.measured().expect("six periods measure five gaps");
        assert!(
            (measured - ASSUMED_FUNDING_INTERVAL_NS).abs() < 1_000 * MS_TO_NS,
            "measured {measured} ns is more than a second off the hour"
        );
    }

    #[test]
    fn a_venue_cadence_change_is_measured_rather_than_silently_rescaling_carry() {
        // Halve the venue's period and every carry number computed from a Ticker is
        // double what it should be, with nothing on the ticker wire to say so.
        let cadence = decode_funding_cadence(&history(&[HOUR_MS / 2; 4])).unwrap();
        assert_eq!(
            cadence,
            FundingCadence::Differs {
                measured: (HOUR_MS / 2) * MS_TO_NS,
                stamped: ASSUMED_FUNDING_INTERVAL_NS,
            }
        );
        assert!(cadence.differs());
    }

    #[test]
    fn an_eight_hourly_venue_is_flagged_even_though_it_is_a_whole_number_of_hours() {
        // The case a "gaps must be whole multiples of the assumed interval" rule
        // would wave through — and the single most likely change, since most CEX
        // perps fund eight-hourly. Carry would come out 8x too large.
        let cadence = decode_funding_cadence(&history(&[8 * HOUR_MS; 3])).unwrap();
        assert_eq!(
            cadence,
            FundingCadence::Differs {
                measured: 8 * HOUR_MS * MS_TO_NS,
                stamped: ASSUMED_FUNDING_INTERVAL_NS,
            }
        );
    }

    #[test]
    fn a_single_missed_period_is_not_read_as_a_longer_interval() {
        // A venue outage (or a market listed mid-window) leaves one double gap. The
        // smallest gap is unmoved by it; a mean would report 1h20m and alarm.
        let cadence = decode_funding_cadence(&history(&[HOUR_MS, 2 * HOUR_MS, HOUR_MS])).unwrap();
        assert!(!cadence.differs(), "got {cadence:?}");
        assert_eq!(cadence.measured(), Some(HOUR_MS * MS_TO_NS));
    }

    #[test]
    fn one_funding_period_measures_nothing_rather_than_guessing() {
        // A single timestamp has no period in it. Reporting "agrees" here would be a
        // check that passes because it never ran.
        let one = r#"[{"coin":"BTC","fundingRate":"0.0","premium":"0.0","time":1784984400000}]"#;
        assert_eq!(
            decode_funding_cadence(one).unwrap(),
            FundingCadence::Undetermined
        );
        assert_eq!(
            decode_funding_cadence("[]").unwrap(),
            FundingCadence::Undetermined
        );
        assert!(!FundingCadence::Undetermined.differs());
        assert_eq!(FundingCadence::Undetermined.measured(), None);
    }

    #[test]
    fn out_of_order_history_is_refused_rather_than_measured_backwards() {
        // A descending body would yield negative gaps; taking |Δt| would turn a
        // reordering into a perfectly plausible hour and hide it forever.
        let descending = decode_funding_cadence(&history(&[-HOUR_MS, -HOUR_MS]));
        assert!(matches!(
            descending,
            Err(DecodeError::Malformed("fundingHistory"))
        ));
        // Two entries with the same timestamp are equally meaningless as a period.
        assert!(matches!(
            decode_funding_cadence(&history(&[0])),
            Err(DecodeError::Malformed("fundingHistory"))
        ));
    }

    #[test]
    fn a_non_history_body_is_a_decode_error_not_an_agreement() {
        // If the endpoint ever answers with an error object, the check must fail
        // loudly instead of returning `Undetermined`, which reads as "nothing to
        // report" and would retire the alarm permanently.
        assert!(matches!(
            decode_funding_cadence(r#"{"error":"unknown coin"}"#),
            Err(DecodeError::Json(_))
        ));
    }

    /// Live cross-check — the one that actually catches the venue changing its mind.
    /// Ignored by default so the gate stays offline; run with
    /// `cargo test -p axon-provider-hyperliquid -- --ignored`.
    ///
    /// Read-only: `POST /info` with no key, no order, nothing to sign.
    #[tokio::test]
    #[ignore = "hits the live Hyperliquid /info endpoint"]
    async fn the_venue_still_funds_at_the_interval_every_ticker_is_stamped_with() {
        use super::super::rest::{fetch_funding_cadence, TESTNET_INFO};
        use axon_core::{Clock, SystemClock};

        let now_ms = SystemClock.now_ns() / MS_TO_NS;
        let cadence = fetch_funding_cadence(TESTNET_INFO, "BTC", now_ms - CADENCE_WINDOW_MS)
            .await
            .expect("fundingHistory is a public read");
        assert!(
            !cadence.differs(),
            "Hyperliquid's funding cadence has moved: {cadence:?}. Every Ticker this \
             adapter emits stamps ASSUMED_FUNDING_INTERVAL_NS, so carry is now wrong by \
             the ratio — fix the constant before trusting any funding number."
        );
        assert!(
            cadence.measured().is_some(),
            "a live 12h window must contain at least two funding periods; got {cadence:?}"
        );
    }
}
