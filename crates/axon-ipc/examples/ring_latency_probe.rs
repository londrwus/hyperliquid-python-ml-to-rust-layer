//! Measure the **one-way Python→Rust latency** across the signal ring.
//!
//! `docs/05-latency-model.md` puts the shared-memory SPSC hop at ~66–135 ns, and that
//! row is a literature ladder — nothing in this repository had ever timed it. The
//! runtime's `sig` stage does not, either: it spans the producer's wall clock to the
//! core's *event* clock, so the transport is buried under a feed lag three to five
//! orders of magnitude larger and clamped at zero (`axon-runtime`'s `latency` module).
//!
//! This example is the reader end of a two-process measurement. `scripts/ipc_latency.py`
//! creates the ring, spawns this, and pushes records carrying three stamps:
//!
//! | Span | Meaning |
//! |---|---|
//! | `t1 - t0` | Python's own write path: stamp → `head` published |
//! | `t2 - t1` | **the wire**: publish → this process observed it |
//! | `t2 - t0` | what a production `ts_event`-based stage would report |
//!
//! `t0` rides on the wire in `ts_event`; `t1` stays in the driver and is joined on
//! `seq` afterwards. `t2` is taken here, the first statement after a successful pop.
//!
//! Both sides read `CLOCK_REALTIME` (Python's `time.time_ns`, Rust's `SystemTime`) on
//! one host, so the subtraction is same-clock despite crossing a process boundary —
//! the property `sig` does not have. `--poll-us 0` spins, which isolates transport;
//! a non-zero value reproduces the production core's `core_poll_us` sleep, which is
//! what a live session actually pays.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axon_ipc::Consumer;

/// Epoch nanoseconds off the same clock Python's `time.time_ns()` reads.
#[inline(always)]
fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos() as i64
}

/// The floor under every number this probe reports: two back-to-back clock reads.
/// Quoted rather than subtracted — a measurement that corrects itself against its own
/// noise floor is one nobody can check.
fn clock_floor_ns(n: usize) -> (i64, i64) {
    let mut deltas = Vec::with_capacity(n);
    for _ in 0..n {
        let a = now_ns();
        let b = now_ns();
        deltas.push(b - a);
    }
    deltas.sort_unstable();
    (deltas[0], deltas[n / 2])
}

fn arg(name: &str, default: u64) -> u64 {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_str(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = arg_str("--ring", "/dev/shm/axon-latency.ring");
    let count = arg("--count", 20_000) as usize;
    let poll_us = arg("--poll-us", 0);
    let out_path = arg_str("--out", "/tmp/ring_latency_rust.csv");

    let (floor_min, floor_p50) = clock_floor_ns(100_000);

    // The driver creates the ring, so racing it on startup is expected rather than an
    // error. Retry until it exists *and* validates — a half-written header is a
    // legitimate transient here and nowhere else.
    let deadline = Instant::now() + Duration::from_secs(15);
    let consumer = loop {
        match Consumer::open(&path) {
            Ok(c) => break c,
            Err(e) if Instant::now() < deadline => {
                if Instant::now() + Duration::from_millis(1) > deadline {
                    return Err(format!("ring never became readable: {e}").into());
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(e) => return Err(e.into()),
        }
    };

    // Preallocated: a reallocation inside the loop would land between two samples and
    // show up as a tail nobody could attribute.
    let mut samples: Vec<(u64, i64, i64)> = Vec::with_capacity(count);

    println!("READY floor_min={floor_min} floor_p50={floor_p50}");
    std::io::stdout().flush()?;

    // The control, and it is **off by default because it perturbs what it measures**.
    // A spinning reader that reports a 12 ms transport is either measuring a 12 ms ring
    // or measuring its own vCPU being taken away, and the two are not distinguishable
    // from the sample alone. So time the *empty* spin iterations: consecutive clock
    // reads with no ring access between them. Any gap there is the scheduler, by
    // construction — nothing else happens in that branch.
    //
    // The cost of knowing that is a vDSO clock read per spin iteration, which lengthens
    // the poll period and so inflates the median detection latency (measured: p50 61 ns
    // clean against 231 ns instrumented). Run it once for attribution, and take the
    // transport number from a clean run.
    let control = arg("--control", 0) != 0;
    let mut spins: u64 = 0;
    let mut spin_over_100us: u64 = 0;
    let mut spin_over_1ms: u64 = 0;
    let mut spin_max_ns: i64 = 0;
    let mut prev_spin: i64 = 0;

    let mut last_seen = Instant::now();
    while samples.len() < count {
        match consumer.try_pop() {
            Some(sig) => {
                // First statement after the pop, before anything else touches memory.
                let t2 = now_ns();
                samples.push((sig.seq, sig.ts_event, t2));
                last_seen = Instant::now();
                prev_spin = 0;
            }
            None => {
                if poll_us > 0 {
                    if last_seen.elapsed() > Duration::from_secs(10) {
                        break;
                    }
                    std::thread::sleep(Duration::from_micros(poll_us));
                    continue;
                }
                if control {
                    let t = now_ns();
                    if prev_spin != 0 {
                        let gap = t - prev_spin;
                        spins += 1;
                        spin_max_ns = spin_max_ns.max(gap);
                        if gap > 100_000 {
                            spin_over_100us += 1;
                        }
                        if gap > 1_000_000 {
                            spin_over_1ms += 1;
                        }
                    }
                    prev_spin = t;
                    if last_seen.elapsed() > Duration::from_secs(10) {
                        break;
                    }
                } else {
                    // The stall check is a clock read too, and on a clean run it is the
                    // *only* thing between two ring polls — leaving it every iteration
                    // would double the poll period this probe exists to keep short. Once
                    // every 4096 spins bounds the giving-up delay at microseconds while
                    // taking the read out of the measurement.
                    spins = spins.wrapping_add(1);
                    if spins % 4096 == 0 && last_seen.elapsed() > Duration::from_secs(10) {
                        break;
                    }
                }
                std::hint::spin_loop();
            }
        }
    }

    let mut w = BufWriter::new(File::create(&out_path)?);
    writeln!(w, "seq,t0_ns,t2_ns")?;
    for (seq, t0, t2) in &samples {
        writeln!(w, "{seq},{t0},{t2}")?;
    }
    w.flush()?;

    println!(
        "DONE n={} out={out_path} spins={spins} spin_over_100us={spin_over_100us} \
         spin_over_1ms={spin_over_1ms} spin_max_ns={spin_max_ns}",
        samples.len()
    );
    Ok(())
}
