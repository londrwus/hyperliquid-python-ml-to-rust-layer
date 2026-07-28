//! Carrying a **v1 event log** forward to `SCHEMA_VERSION` 2 — as a command somebody
//! runs, never as something a reader does on its own.
//!
//! ADR-0027 §4 refuses a v1 log outright, and the argument is that accepting one means
//! planning it unconstrained: prices the session it reproduces could not have sent,
//! reported by the golden gate as strategy flips. Nothing here weakens that. What it
//! does is give the refusal a way out that keeps the same sentence true.
//!
//! The distinction is *who decides*. A reader that quietly upcast would put every stored
//! log back on the loose path with a warning on stderr, which is the state the version
//! bump exists to end. An operator who types this has decided, for one named file, that a
//! recording with **no declared grid** is what they want — and that is exactly what comes
//! out: [`InstrumentSet::Undeclared`], which every reader then says out loud. The upcast
//! invents nothing. A v1 log really did carry no grid, so `Undeclared` is not a
//! compromise; it is the only true statement available about it.
//!
//! It goes through the real [`Capture`] writer rather than rewriting lines, so an upcast
//! log is byte-for-byte a log this build would have written — including the `ts_event`
//! cross-check on every record, which is what stops a corrupt v1 file being laundered
//! into a well-formed v2 one.
//!
//! # The signal log is deliberately **not** upcastable
//!
//! `axon.signallog` also went 1 → 2, and for an unrelated reason: `Signal` gained
//! `max_order_age_ms`. A v1 record has no value for it, and the value a naive upcast
//! would write — zero — is not "absent". Zero means *defer to the operator's ceiling*,
//! so an upcast signal log would replay with every order carrying a lifetime the live
//! session never gave it, and orders would be pulled on age where the session left them
//! resting. That is a changed decision wearing a migration's clothes, and there is no
//! honest number to write instead. A v1 signal log has to be re-captured; an upcast event
//! log replays with `--no-signals`, which re-observes the session without re-deciding it.

use std::io::{BufRead, Write};

use axon_core::Event;

use crate::capture::Capture;
use crate::error::ReplayError;
use crate::instruments::InstrumentSet;
use crate::log::{LogHeader, LogRecord, SCHEMA};

/// The one version this knows how to carry forward.
const V1: u32 = 1;

/// What an upcast produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UpcastReport {
    pub events: u64,
    /// The v1 log's own `created_ns`, carried through. Provenance is the whole reason to
    /// upcast rather than re-record, so losing it would defeat the exercise.
    pub created_ns: i64,
}

/// Read a v1 event log and write it out as a v2 one that declares no grid.
///
/// Refuses anything that is not a v1 `axon.eventlog`: a v2 log needs nothing done to it,
/// and a signal log handed here is the wrong file (see the module docs).
pub fn upcast_v1(src: impl BufRead, out: impl Write) -> Result<UpcastReport, ReplayError> {
    let mut lines = src.lines().enumerate();

    let first = lines.next().ok_or(ReplayError::MissingHeader)?.1?;
    let header: LogHeader =
        serde_json::from_str(&first).map_err(|source| ReplayError::Json { line: 1, source })?;
    if header.schema != SCHEMA {
        return Err(ReplayError::ForeignLog {
            found: header.schema,
            expected: SCHEMA,
        });
    }
    if header.schema_version != V1 {
        return Err(ReplayError::IncompatibleVersion {
            schema: SCHEMA,
            found: header.schema_version,
        });
    }

    // `source` and `created_ns` survive; `writer` and `schema_version` become this
    // build's, because they describe the file in front of you rather than the session.
    let mut cap = Capture::with_header(
        out,
        LogHeader::new(header.source.clone(), header.created_ns),
        InstrumentSet::Undeclared,
    )?;

    let mut events = 0u64;
    for (idx, line) in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let lineno = idx + 1;
        // The v1 body line *is* a `LogRecord`: only the enclosing tag is new, so this
        // parses the old shape without a second definition of it living here.
        let rec: LogRecord = serde_json::from_str(&line).map_err(|source| ReplayError::Json {
            line: lineno,
            source,
        })?;
        // Written through `record`, which re-derives `seq` and re-checks `ts_event`
        // against the payload. A v1 log whose ordering key had already drifted must not
        // be laundered into a well-formed v2 one.
        let derived = rec.event.ts_event();
        if derived != rec.ts_event {
            return Err(ReplayError::OrderingKeyChanged {
                line: lineno,
                recorded: rec.ts_event,
                derived,
            });
        }
        cap.record(&Event::from(rec.event))?;
        events += 1;
    }
    cap.finish()?;
    Ok(UpcastReport {
        events,
        created_ns: header.created_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{a_fill, a_trade};
    use crate::{EventLog, LoggedEvent};

    /// A v1 log, spelled out: a header at version 1 and one flat record per line, with no
    /// instrument line and no per-line tag. Written by hand because no build can produce
    /// one any more, and the shape is the thing under test.
    fn v1_log(events: &[Event]) -> String {
        let mut out = String::from(
            r#"{"schema":"axon.eventlog","schema_version":1,"writer":"axon-replay 0.0.1","source":"a real session","created_ns":42}"#,
        );
        out.push('\n');
        for (seq, ev) in events.iter().enumerate() {
            let rec = LogRecord {
                seq: seq as u64,
                ts_event: ev.ts_event(),
                event: LoggedEvent::from(ev),
            };
            out.push_str(&serde_json::to_string(&rec).unwrap());
            out.push('\n');
        }
        out
    }

    #[test]
    fn an_upcast_log_keeps_every_record_and_the_provenance_of_the_session_that_wrote_it() {
        // The point of upcasting at all: a live tape costs a live session to obtain, and
        // `source`/`created_ns` are how anybody later knows which one it was. An upcast
        // that re-stamped them would produce a file indistinguishable from one recorded
        // this afternoon.
        let events = [a_trade(30), a_fill(10), a_trade(20)];
        let mut out = Vec::new();
        let report = upcast_v1(v1_log(&events).as_bytes(), &mut out).unwrap();

        assert_eq!(report.events, 3);
        assert_eq!(report.created_ns, 42);
        let log = EventLog::read(&out[..]).unwrap();
        assert_eq!(log.header().source, "a real session");
        assert_eq!(log.header().created_ns, 42);
        assert_eq!(log.header().schema_version, crate::SCHEMA_VERSION);
        let back: Vec<Event> = log
            .records()
            .iter()
            .map(|r| Event::from(r.event.clone()))
            .collect();
        assert_eq!(back, events, "capture order, verbatim");
        assert_eq!(
            log.records().iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn an_upcast_declares_no_grid_rather_than_inventing_one() {
        // The whole reason this is safe. A v1 log genuinely carried no instrument table,
        // so `Undeclared` is the only true statement about it — and every reader then
        // says so, which is the loudness ADR-0027 §4 refused to trade away. An upcast
        // that wrote `Declared { unconstrained: true }` instead would claim the *venue*
        // had no grid, which is a claim about Hyperliquid nobody is entitled to make.
        let mut out = Vec::new();
        upcast_v1(v1_log(&[a_trade(1)]).as_bytes(), &mut out).unwrap();
        let log = EventLog::read(&out[..]).unwrap();
        assert_eq!(log.instruments(), &InstrumentSet::Undeclared);
        assert!(log.instruments().to_table().unwrap().is_none());
    }

    #[test]
    fn a_v1_log_whose_ordering_key_had_already_drifted_is_not_laundered_into_a_v2_one() {
        // An upcast is a rewrite, and a rewrite is exactly where a corrupt file becomes a
        // well-formed one. Every check the v2 reader would apply is applied here, so a
        // log that could not be replayed before cannot be replayed after.
        let bad = v1_log(&[a_trade(1_000)])
            .replace("\"ts_event\":1000,\"event\"", "\"ts_event\":999,\"event\"");
        let err = upcast_v1(bad.as_bytes(), Vec::new()).unwrap_err();
        assert!(
            matches!(err, ReplayError::OrderingKeyChanged { recorded: 999, .. }),
            "{err}"
        );
    }

    #[test]
    fn a_log_that_is_already_current_or_is_the_wrong_file_is_refused() {
        // "Run it again and see" must not silently double-wrap a log, and a signal log
        // handed here is the wrong file rather than a corrupt one — the module docs say
        // why a signal log has no honest upcast at all.
        let current = crate::test_support::log_bytes(&[a_trade(1)]);
        assert!(matches!(
            upcast_v1(current.as_bytes(), Vec::new()).unwrap_err(),
            ReplayError::IncompatibleVersion { found, .. } if found == crate::SCHEMA_VERSION
        ));

        let foreign = v1_log(&[]).replace("axon.eventlog", "axon.signallog");
        assert!(matches!(
            upcast_v1(foreign.as_bytes(), Vec::new()).unwrap_err(),
            ReplayError::ForeignLog { .. }
        ));
    }
}
