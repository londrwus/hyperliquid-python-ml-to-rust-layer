//! Run the cross-language feature-parity gate and **print what it measured**.
//!
//! `cargo test -p axon-features --test feature_parity` already asserts this, and an
//! assertion that passes prints nothing. That is correct for CI and useless for the
//! two questions a human actually has: *how much was compared*, and *how close was
//! it*. A gate whose numbers nobody can see is a gate nobody can tell from a gate
//! that stopped running — the same argument `intent.rs` makes about a counter that
//! is incremented and never reported.
//!
//! So this walks every committed bundle, opens it, computes the matrix with **this**
//! build of the Rust runtime, and prints the report per bundle plus a total. It
//! exits non-zero if any bundle fails, so it is safe to put in front of an operator
//! and safe to put in a script.
//!
//! ```text
//! cargo run -p axon-features --example feature_gate
//! ./run.sh feature-parity
//! ```
//!
//! No Python, no numpy, no network, no clock — which is the whole claim a bundle
//! exists to make portable.

use std::path::{Path, PathBuf};

use axon_features::parity::FeatureBundle;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bundles");
    let bundles = match committed(&root) {
        Ok(found) if !found.is_empty() => found,
        Ok(_) => {
            // An empty sweep reporting PASS is the invisible denominator ADR-0030
            // spent a whole increment on one level up. Nothing to compare is a
            // failure of this program, not a clean bill of health for the runtime.
            eprintln!(
                "no committed feature-parity bundles under {}; regenerate with \
                 ./run.sh feature-bundles",
                root.display()
            );
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("reading {}: {e}", root.display());
            std::process::exit(2);
        }
    };

    println!(
        "cross-language feature parity (ADR-0035) — {} committed bundle(s) under {}",
        bundles.len(),
        root.display()
    );
    println!(
        "the question: do the two languages compute the same feature vectors from the \
         same market data?\n"
    );

    let (mut cells, mut failed, mut transforms) =
        (0usize, 0usize, std::collections::BTreeSet::new());
    for dir in &bundles {
        let name = dir.file_name().unwrap_or_default().to_string_lossy();
        let bundle = match FeatureBundle::open(dir) {
            Ok(b) => b,
            Err(e) => {
                // A malformed bundle is reported as a broken fixture, never folded
                // into the parity tally: "the bundle is wrong" and "the runtime is
                // wrong" send somebody to opposite ends of the tree.
                println!("{name}: BUNDLE ERROR — {e}");
                failed += 1;
                continue;
            }
        };
        for def in bundle.spec().features() {
            transforms.insert(def.feature().to_string());
        }
        let source = bundle
            .source()
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("(no description)");

        match bundle.check() {
            Ok(report) => {
                cells += report.cells_compared();
                if !report.passed() {
                    failed += 1;
                }
                println!("{}", report.summary());
                println!("    spec   {}", bundle.spec_ref());
                println!("    source {source}");
                if !report.libm_columns().is_empty() {
                    println!(
                        "    libm   {:?} — these columns pass through log, the one operation \
                         IEEE-754 does not require to be correctly rounded",
                        report.libm_columns()
                    );
                }
                println!();
            }
            Err(e) => {
                println!("{name}: GATE ERROR — {e}\n");
                failed += 1;
            }
        }
    }

    println!(
        "{} bundle(s), {cells} cells compared, {} distinct transform(s) exercised, {failed} failure(s)",
        bundles.len(),
        transforms.len(),
    );
    if failed > 0 {
        std::process::exit(1);
    }
}

fn committed(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    // Name order, so two runs print the same thing and a diff of two runs is
    // readable.
    out.sort();
    Ok(out)
}
