//! The market-data beacon, driven by the thing that actually drives it: the core's
//! pass loop.
//!
//! Everything below runs the **real** [`axon_runtime::core::run`] against a real
//! [`MdPublisher`] and reads the result back through the same
//! [`MdBeaconReader`](axon_ipc::MdBeaconReader) the parity monitor's Python twin
//! mirrors. That is deliberate and it is the whole point of the file: the beacon type,
//! its layout and its reader were all unit-tested in both languages before anything
//! called `beat`, so the one property none of those tests could hold was that a session
//! beats at all. ADR-0034's consequences recorded exactly that hole — "the beacon is not
//! wired into the pass loop in the tree" — and these tests are what closes it.
//!
//! An integration test rather than a unit test inside `core.rs`, because the property
//! under test crosses three modules (the loop, the publisher, the mapped file) and one
//! process boundary's worth of decoding. A unit test that asserted `beat` was called
//! would prove the call and nothing about what a reader finds.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axon_core::{bus, Clock, EventReceiver, SymbolId, SystemClock};
use axon_execution::{HaltSwitch, MarkCache, OrderTracker};
use axon_ipc::{beacon_path, MdBeaconReader, MD_BEACON_FLAG_RUNNING, MD_BEACON_FLAG_STOPPED};
use axon_runtime::config::MdRingConfig;
use axon_runtime::core::{run, CoreControl};
use axon_runtime::dms::{now_ms, DmsState};
use axon_runtime::handler::CoreHandler;
use axon_runtime::health::SessionHealth;
use axon_runtime::mdring::{MdPublisher, MdWritePolicy};
use axon_runtime::selftest;

const BTC: SymbolId = SymbolId::new(0);
const ETH: SymbolId = SymbolId::new(1);
/// The shipped default mark window, so a quote ages here the way it ages in a session.
const WINDOW: i64 = 10_000_000_000;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(tag: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!(
            "axon-beacon-wiring-{tag}-{}-{n}.ring",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn md_cfg(path: &str) -> MdRingConfig {
    MdRingConfig {
        enabled: true,
        path: path.to_string(),
        capacity: 64,
        policy: MdWritePolicy::OnChange,
    }
}

fn publisher(path: &str) -> MdPublisher {
    MdPublisher::open(&md_cfg(path), WINDOW)
        .expect("the ring is creatable")
        .expect("publishing is on")
}

fn handler_with(md: Option<MdPublisher>) -> CoreHandler {
    let h = CoreHandler::new(
        Arc::new(RwLock::new(OrderTracker::new())),
        Arc::new(MarkCache::never_expires()),
    );
    match md {
        Some(md) => h.with_md_publisher(md),
        None => h,
    }
}

/// The offline session's own control block: `wall_time` off, because offline the only
/// clock that exists is event time and a wall-clock reading would make a deterministic
/// run depend on how fast the machine drained the bus.
fn control(stop: Arc<AtomicBool>, wall_time: bool) -> CoreControl {
    CoreControl {
        stop,
        poll: Duration::from_micros(100),
        status_every: Duration::from_secs(3_600), // never fires inside a test
        wall_time,
        mode: "test/offline".into(),
        symbols: vec![(BTC, "BTC".into()), (ETH, "ETH".into())],
        health: Arc::new(SessionHealth::new(now_ms())),
        halt: Arc::new(HaltSwitch::new()),
        dms: Arc::new(DmsState::default()),
        governor: None,
        dms_expected: false,
        capture: None,
        pnl_expected: false,
        pnl_limits: Default::default(),
        latency: Arc::new(axon_runtime::latency::LatencyBook::undeclared()),
        loss: Arc::new(axon_execution::LossLimiter::undeclared()),
        daybook: None,
    }
}

/// Drive `run` to completion over a bus that is already closed and a stop flag that is
/// already set — the shape [`axon_runtime::run_offline`] uses, and the one that makes
/// the pass count an exact number instead of a race against the scheduler.
fn drive(rx: &EventReceiver, handler: &mut CoreHandler, wall_time: bool) {
    let ctl = control(Arc::new(AtomicBool::new(true)), wall_time);
    run(rx, handler, &ctl, None);
}

/// Poll `f` until it answers, with a deadline.
///
/// Bounded rather than a bare `loop`, and the reason is not politeness: every use below
/// waits on something the *wiring under test* produces, so an unwired loop would hang
/// the suite instead of failing it — which is exactly what happened the first time this
/// file's assertions were checked by removing the beat from `core::run`. A test that
/// hangs when the thing it guards is missing is not a guard.
fn until<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting: {what}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn a_pass_that_drained_nothing_still_beats_because_the_absence_of_an_event_has_no_event_time() {
    // The failure this prevents is the whole reason the beacon is on the pass loop and
    // not on `on_event`: a publisher whose feed died drains nothing, publishes nothing
    // and — under `OnChange` — leaves a ring byte-identical to the one a quiet market
    // leaves. If the only thing that advanced the beacon were a published record, the
    // file would freeze at exactly the moment it is the only remaining evidence.
    let path = temp_path("silent");
    let (tx, rx) = bus(8);
    drop(tx); // nothing will ever arrive
    let mut handler = handler_with(Some(publisher(&path)));
    drive(&rx, &mut handler, false);

    let snap = MdBeaconReader::open(beacon_path(&path))
        .expect("the session created a beacon")
        .read();
    // Exactly two: one pass, which beat before it discovered the bus was empty and the
    // stop flag set, plus the closing beat. Not "at least one" — the count is the field
    // an operator watches move, and a test that accepted any positive number would pass
    // just as happily against a loop that beat once and stopped.
    assert_eq!(snap.beats, 2, "one pass, then the shutdown beat");
    assert_eq!(snap.last_event_ns, 0, "no event ever arrived to stamp one");
    assert_eq!(snap.published, 0);
    assert_eq!(snap.coalesced, 0);
    assert_eq!(snap.dropped, 0);
    // The beat advanced while every event-derived field stayed at zero. That pair is
    // the reading `PublisherState::STARVED` is made of, and before this wiring existed
    // there was nothing in the tree that could produce it.
    assert!(snap.running() || snap.stopped());
}

#[test]
fn the_event_time_high_water_mark_reaches_the_beacon_and_an_offline_session_stamps_no_wall_clock() {
    // Both clocks, checked against each other. `last_event_ns` must be the core's own
    // `ts_event` high-water mark — the clock every ordering decision is made on — and
    // `last_beat_ns` must be the sentinel zero, because `CoreControl::wall_time` is off
    // offline on purpose. A reader that treated that zero as a timestamp would report
    // every offline session as fifty-six years dead, which is why the reader has
    // `had_wall_clock` rather than a subtraction.
    let path = temp_path("offline");
    let (tx, rx) = bus(32);
    for ev in selftest::events(BTC, ETH) {
        tx.send(ev).unwrap();
    }
    drop(tx);
    let mut handler = handler_with(Some(publisher(&path)));
    drive(&rx, &mut handler, false);

    let stats = handler.md().expect("the publisher is attached").stats();
    let snap = MdBeaconReader::open(beacon_path(&path)).unwrap().read();

    // The canned stream's last venue-timed event, in nanoseconds. Asserted as the
    // number rather than as `> 0`: the bug this catches is a beacon stamped with the
    // *first* event, or with a candle's close time, both of which are positive.
    assert_eq!(snap.last_event_ns, 7_000_000_000);
    assert_eq!(snap.last_beat_ns, 0, "offline has no wall clock at all");
    assert_eq!(snap.pid, std::process::id());
    assert_eq!(snap.beats, 2);
    // The five counters are the publisher's own, not a second tally kept beside them.
    assert_eq!(u64::from(snap.published), stats.published);
    assert_eq!(u64::from(snap.coalesced), stats.coalesced);
    assert_eq!(u64::from(snap.dropped), stats.dropped);
    assert_eq!(u64::from(snap.bars_published), stats.bars_published);
    assert_eq!(u64::from(snap.stale_quote), stats.stale_quote);
    // And what those numbers actually are for this stream, so a change to the canned
    // events that silently changed what a session publishes cannot pass here.
    assert_eq!(snap.published, 2);
    assert_eq!(snap.coalesced, 0);
    assert_eq!(snap.dropped, 0);
}

#[test]
fn a_live_pass_loop_beats_repeatedly_and_stamps_the_loops_own_wall_clock_not_a_second_reading() {
    // The property an offline run cannot show: that the beat *advances* over time with
    // no event to advance it, and that `last_beat_ns` carries a real `CLOCK_REALTIME`
    // reading — the third named wall-clock exception in this codebase, beside the
    // dead-man's-switch deadline and the reconnect backoff.
    //
    // The reading is asserted to lie inside a window this test measured itself, rather
    // than against a bound: a beacon stamped from `Instant`, from a monotonic clock, or
    // from the epoch would all be "greater than zero".
    let path = temp_path("live");
    let stop = Arc::new(AtomicBool::new(false));
    let before = SystemClock.now_ns() as u64;

    let looper = std::thread::spawn({
        let path = path.clone();
        let stop = stop.clone();
        move || {
            // The producer is kept alive for the run, so this is a loop parked on a
            // silent-but-open bus — a live session whose feed has gone quiet.
            let (_tx, rx) = bus(8);
            let mut handler = handler_with(Some(publisher(&path)));
            let ctl = control(stop, true);
            run(&rx, &mut handler, &ctl, None);
        }
    });

    // The file appears when the publisher opens, which races this thread's first read.
    let reader = until("the session to create its beacon", || {
        MdBeaconReader::open(beacon_path(&path)).ok()
    });
    let first = reader.read();
    std::thread::sleep(Duration::from_millis(50));
    let second = reader.read();
    stop.store(true, Ordering::Release);
    looper.join().expect("the loop thread ran to completion");
    let last = reader.read();
    let after = SystemClock.now_ns() as u64;

    assert!(
        second.beats > first.beats,
        "the beat has to advance with no event to advance it: {} then {}",
        first.beats,
        second.beats
    );
    assert_eq!(second.last_event_ns, 0, "nothing arrived on this bus");
    assert_eq!(second.published, 0);
    assert!(
        (before..=after).contains(&second.last_beat_ns),
        "last_beat_ns {} is not a CLOCK_REALTIME reading taken inside this test's own \
         window [{before}, {after}]",
        second.last_beat_ns
    );
    assert!(last.beats >= second.beats);
}

#[test]
fn a_deliberate_shutdown_is_flagged_so_a_reader_can_tell_the_session_ending_from_the_session_dying()
{
    // A clean exit does not put the market data back, so a reader still has to treat
    // the feed as gone — but "the publisher stopped" and "the publisher died" want
    // different words from whoever is woken up by them at 03:00, and this side is the
    // only one that can tell them apart. Without the closing beat the last thing on the
    // file says RUNNING forever and every clean shutdown reads as a crash.
    let path = temp_path("stopflag");
    let stop = Arc::new(AtomicBool::new(false));

    let looper = std::thread::spawn({
        let path = path.clone();
        let stop = stop.clone();
        move || {
            let (_tx, rx) = bus(8);
            let mut handler = handler_with(Some(publisher(&path)));
            let ctl = control(stop, false);
            run(&rx, &mut handler, &ctl, None);
        }
    });

    let reader = until("the session to create its beacon", || {
        MdBeaconReader::open(beacon_path(&path)).ok()
    });
    // Wait for a beat before reading the flag: `create` also stamps RUNNING, so reading
    // at beat zero would assert the header rather than the loop.
    let running = until("the pass loop to beat at least once", || {
        let s = reader.read();
        (s.beats > 0).then_some(s)
    });
    assert_eq!(running.flags, MD_BEACON_FLAG_RUNNING);
    assert!(running.running() && !running.stopped());

    stop.store(true, Ordering::Release);
    looper.join().unwrap();
    let stopped = reader.read();
    assert_eq!(stopped.flags, MD_BEACON_FLAG_STOPPED);
    assert!(stopped.stopped() && !stopped.running());
}

#[test]
fn a_session_that_publishes_no_ring_creates_no_beacon_file_either() {
    // The beacon rides on `[md_ring] enabled` and creates a file, so it inherits that
    // switch's rule exactly: `cargo run --bin axon` with the shipped defaults must stay
    // a process that touches nothing. There is no `[md_beacon]` block to get wrong —
    // one switch cannot be half-turned, and an operator who enabled slices and forgot
    // the beacon would get a monitor that silently went back to guessing.
    let path = temp_path("disabled");
    let mut cfg = md_cfg(&path);
    cfg.enabled = false;
    assert!(MdPublisher::open(&cfg, WINDOW).unwrap().is_none());

    let (tx, rx) = bus(32);
    for ev in selftest::events(BTC, ETH) {
        tx.send(ev).unwrap();
    }
    drop(tx);
    // A whole pass loop with no publisher attached. The beat is behind an `if let`, so
    // the failure this guards against is a loop that unconditionally opened one.
    let mut handler = handler_with(None);
    drive(&rx, &mut handler, false);

    assert!(!Path::new(&path).exists(), "no slice ring");
    assert!(
        !beacon_path(&path).exists(),
        "no beacon: {}",
        beacon_path(&path).display()
    );
    assert!(handler.md().is_none());
}

#[test]
fn one_switch_creates_three_files_and_no_derived_path_can_shadow_another() {
    // The adversarial spelling, which is the one an extension-substituting derivation
    // collides on: an operator who points `md_ring.path` at something already ending in
    // `.beacon` must still get three distinct files. `beacon_path` *appends* rather than
    // substituting, so a string cannot equal itself plus a suffix and the collision is
    // unrepresentable rather than merely refused — there is no check anybody can forget
    // to call. Asserted here, at the layer that actually creates the three files,
    // because that is where the consequence lands: a beacon opened onto the slice ring
    // would truncate the transport it exists to describe.
    let dir = std::env::temp_dir().join(format!(
        "axon-beacon-shadow-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("axon-md.beacon").to_string_lossy().into_owned();

    let md = publisher(&path);
    let three: Vec<PathBuf> = vec![
        md.path().to_path_buf(),
        md.bar_path().to_path_buf(),
        md.beacon_path().to_path_buf(),
    ];
    for p in &three {
        assert!(p.exists(), "{} was not created", p.display());
    }
    let mut sorted = three.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 3, "three distinct files: {three:?}");
    // And the beacon is the one that is 64 bytes; the rings are ring-sized. A beacon
    // opened onto a ring's path would pass the distinctness check above and fail here.
    assert_eq!(std::fs::metadata(&three[2]).unwrap().len(), 64);
    let _ = std::fs::remove_dir_all(&dir);
}
