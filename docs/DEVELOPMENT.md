# Development

How to build, test, and run Axon.

**Start with the Ubuntu section below.** Native Ubuntu 24.04 is the primary development
and deploy target ([ADR-0024](adr/0024-native-ubuntu-dev-target.md), superseding
[ADR-0007](adr/0007-linux-wsl2-dev-target.md)). The codebase is still OS-agnostic and
the same commands work Windows-native and under WSL2 — CI proves it on every push by
running the whole gate on `windows-latest` too — but Ubuntu is where the work happens,
and it is the only path with a one-command installer.

## Prerequisites

- **Rust** (stable), pinned by [`rust-toolchain.toml`](../rust-toolchain.toml); the
  workspace needs **1.85+**. Install via <https://rustup.rs> — *not* apt, whose `rustc`
  is older than the minimum and shadows rustup on PATH. On Windows it additionally
  needs the MSVC build tools (VS 2022 Build Tools with the C++ workload + a Windows
  SDK); `rustup` uses `x86_64-pc-windows-msvc` by default.
- **Python 3.12 recommended** (`>=3.11,<3.14`; `tomllib` needs 3.11+). We pin 3.12
  because the ML stack (torch/onnx/xgboost/lightgbm) lags the newest Pythons — avoid
  3.14. Needs **numpy**, plus **pytest** for tests.

`scripts/install-ubuntu.sh` handles both of these for you; the list is here for the
platforms that have no installer.

## Ubuntu (native — the primary target)

On a fresh Ubuntu box (24.04 LTS is the tested target; 22.04+ works), two commands:

```bash
bash scripts/install-ubuntu.sh     # toolchain + venv + .env scaffold, then the gate
./run.sh                           # the full gate, any time after
```

The installer is idempotent — re-running it skips whatever is already done — and
`--dry-run` prints what it would do without changing anything. `--no-gate` installs
without running the tests. Run it **as yourself, never with sudo**: a root-owned
`~/.cargo`, `.venv` and `target/` would leave the next plain `./run.sh` failing with
permission errors, so it refuses outright.

It handles the three things that otherwise bite on 24.04: apt's `rustc` is older than
this workspace's 1.85 minimum *and* shadows rustup on PATH (the installer verifies the
version actually resolved, not merely that cargo exists); PEP 668 makes `pip install`
into the system Python fail, so everything Python lives in a repo-local `.venv`; and
`python3 -m venv` needs the separate `python3-venv` package. It also scaffolds `.env`
from `.env.example` at mode 0600 with placeholders only — no script ever writes a real
key. No `libssl-dev` is required: the workspace is rustls end to end.

`./run.sh` puts cargo and the venv on PATH itself, so it works from a bare shell with
no `source` step. `./run.sh help` lists every subcommand; the useful ones are
`check` (default), `test`, `runtime`, `book BTC`, `wallet`, `live`, and `doctor` —
which reports exactly what is installed and what is missing.

## The one command

```bash
bash scripts/check.sh    # or: ./run.sh check
```

Runs the full gate: `rustfmt --check`, `clippy -D warnings`, `cargo test --workspace`,
builds the IPC examples, then the Python tests (including the **Python ↔ Rust
round-trip**). This is exactly what CI runs, on both Linux and Windows.

## Piecemeal

```bash
# Rust
cargo test --workspace
cargo run --bin axon                 # contract/config smoke check
cargo build -p axon-ipc --examples   # ipc_reader / ipc_writer / md_writer (cross-language tests)

# Python (from repo root)
export PYTHONPATH=python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # see gotcha below
python -m pytest python/tests -v
```

## Gotchas

- **pytest plugin autoload.** If another globally-installed package ships a broken
  pytest plugin (e.g. `anchorpy` needing `pytest_asyncio`), pytest fails to start.
  Set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — our tests need no third-party plugins.
  `scripts/check.sh` and CI already do this.
- **Cross-language test discovery.** The cross-language tests run the `axon-ipc`
  example binaries out of `target/debug/examples` — `ipc_reader`/`ipc_writer` for the
  signal ring (`test_roundtrip.py`) and `md_writer` for the market-data ring
  (`test_md_ring.py`). `python/tests/conftest.py` owns the whole policy for all three:
  if any is missing it runs one `cargo build -p axon-ipc --examples`, and if `cargo`
  isn't found it *skips* the tests that needed the absent binary — so a Python-only
  environment still passes cleanly, and a target dir that predates one example does not
  skip the tests that do not use it. Add a fourth example to `_IPC_EXAMPLES` there,
  not to a test file.
- **Platform assumption:** x86_64, little-endian (see
  [ADR-0006](adr/0006-signal-schema-and-spsc-ring.md)). The ring's default path is
  `/dev/shm/axon-*.ring` on Linux (tmpfs); any temp path works on Windows.

## WSL2 (only if your machine is a Windows one)

Not the primary path any more ([ADR-0024](adr/0024-native-ubuntu-dev-target.md)) — use
the native Ubuntu section above if you have the choice. It is kept because it works and
because the two lessons below were paid for once already.

Set up WSL2 from Windows: in an **elevated** PowerShell run `wsl --install`, reboot,
create your Ubuntu user, then from the repo:

```bash
cd /mnt/c/Users/<you>/Documents/hyperliquid-ml-rust-layer
bash scripts/wsl-bootstrap.sh     # rustup + uv-managed Python 3.12 + full gate
```

The bootstrap:
- installs `build-essential` + rustup (stable),
- installs **`uv`** and a pinned **Python 3.12** venv at `~/axon-venv` (NOT the
  distro's system Python — Ubuntu 26.04 ships 3.14, too new for the ML stack),
- builds into a **Linux-native** `CARGO_TARGET_DIR=~/axon-target` so the
  `/mnt/c`-mounted repo builds fast and the Windows `target/` isn't clobbered.

In a new WSL shell, activate the env with:

```bash
source ~/axon-venv/bin/activate
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR="$HOME/axon-target"
bash scripts/check.sh
```

Building off `/mnt/c` works; for maximum speed clone into the WSL2 home filesystem
(the source reads are the only slow part once the target dir is native).

## Windows (native)

Supported, and there is no installer for it: rustup + the MSVC build tools, a Python
3.12 venv, `pip install numpy pytest`, then `bash scripts/check.sh` (Git Bash) or the
piecemeal commands above. Use `.\scripts\with-env.ps1 <command>` in place of
`scripts/with-env.sh`.

Nobody develops here day to day, so the `windows-latest` CI matrix cell is the only
thing that catches a Windows-only regression — which is why it runs the *whole* gate
and why removing it would quietly downgrade `axon-ipc`'s portability from a tested
property to a claim ([ADR-0024](adr/0024-native-ubuntu-dev-target.md)).

## Secrets and live venue access

Signing keys come from a gitignored `.env` (start from `.env.example`) and are loaded
into a child process by the wrappers, so a key never lands on a command line, in shell
history, or in a CI log:

```bash
bash scripts/with-env.sh cargo run -p axon-provider-hyperliquid --example wallet_info
bash scripts/with-env.sh cargo test -p axon-provider-hyperliquid -- --ignored
```

On Windows: `.\scripts\with-env.ps1 <command>`. Live-venue tests are `#[ignore]`d so
the default gate stays offline and deterministic. See
[`.env.example`](../.env.example) for key generation and
[ADR-0009](adr/0009-hyperliquid-signing.md) for the agent-wallet model.

## Editing the contract

`contracts/schema.toml` is the single source of truth. Change it and **both** sides
update: Rust regenerates + re-asserts offsets at build time (a mismatch is a compile
error), Python re-reads it at import. Always bump `schema_version` for a layout change
and keep the paired fixtures in sync — `make_signal` (`crates/axon-ipc/examples/ipc_writer.rs`
+ `axon.signals._fixtures`) and `make_md_slice` (`crates/axon-ipc/examples/md_writer.rs`
+ `axon.marketdata._fixtures`). Contract changes are ADR-worthy.

The file defines **three** records — `Signal` (Py→Rust, 64 B), `MdSlice` and `MdBar`
(both Rust→Py, 128 B) — so a ring's control block carries an explicit `record_kind`
alongside `record_size`. That tag is load-bearing rather than belt-and-braces: `MdSlice`
and `MdBar` share a stride, so `record_size` cannot tell them apart at all
([ADR-0028](adr/0028-market-data-bars-and-the-ticker-tail.md)). Adding a fourth record
means adding a kind; never reuse kind `0`, which is deliberately unassigned so an
unstamped writer is rejected rather than defaulting to "signal"
([ADR-0012](adr/0012-market-data-ring-and-multi-record-contract.md)).

**All three records are now fully named — none has a `reserved` block left.** `Signal`
reached schema version **3** when `ts_cause` spent its last eight bytes
([ADR-0037](adr/0037-a-loss-that-halts-an-exit-that-works-and-the-bars-own-clock.md) §4),
and `MdSlice` reached version 2 when the mark/funding tail spent its 48
([ADR-0028](adr/0028-market-data-bars-and-the-ticker-tail.md) §2). So the next field on
either has to **re-cut the layout** rather than extend it, which is a stride change and a
much larger decision than the additive bumps that came before it.

## Workspace map

See [`crates/README.md`](../crates/README.md), [`python/README.md`](../python/README.md),
and [`contracts/README.md`](../contracts/README.md).
