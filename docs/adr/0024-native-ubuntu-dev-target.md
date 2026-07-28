# ADR-0024 — Native Ubuntu 24.04 as the dev target; Windows kept as a CI claim

**Status:** Accepted · **Date:** 2026-07-25 · **Supersedes:** [ADR-0007](0007-linux-wsl2-dev-target.md)

## Context

[ADR-0007](0007-linux-wsl2-dev-target.md) decided, in Phase 1, that Linux was the
deploy target and **WSL2 was the development environment**, with the repo mounted from
Windows at `/mnt/c` and Windows-native builds kept working as the fast local loop. Two
facts made that the right call at the time and both were true: the machine in front of
the author was a Windows 11 box that already had the VS 2022 Build Tools, so
`x86_64-pc-windows-msvc` worked with nothing further to install; and the shared-memory
transport ([ADR-0006](0006-signal-schema-and-spsc-ring.md)) is a memory-mapped *file*,
which is coherent on both OSes. The decision let the entire Phase-1 foundation be built
and tested while WSL2 — which needs admin rights and a reboot — was still being set up.

None of that is wrong. What changed is where the work happens. Development, and every
live testnet session since, now runs on a **native Ubuntu 24.04 LTS** host: the same OS
family as production (`docs/05-latency-model.md` puts prod in AWS `ap-northeast-1`), the
box that holds the `.env` agent key, and the box `./run.sh` was written for. The repo
grew `scripts/install-ubuntu.sh` and `run.sh` to match, and `scripts/wsl-bootstrap.sh`
now describes a machine nobody uses.

That leaves ADR-0007 stating, as accepted fact, three things that are no longer true —
"development currently happens on a Windows 11 machine", "WSL2 (Ubuntu) is the primary
development environment", and that Windows-native dev "is used as a fast local
compile/test loop today". A reader onboarding from the docs is sent to install WSL2
first. Quietly editing those sentences would erase *why* they were right, which is the
one thing an ADR is for, so this is a superseding record rather than a patch.

The question worth deciding — the only one with a real cost either way — is what happens
to the **Windows claim**. Nobody develops there any more, so the cheap move is to drop
it: one CI matrix cell, one installer and one `#[cfg]`-free promise less to maintain.

## Decision

**1. Native Ubuntu 24.04 LTS is the primary development *and* deploy target.**
Dev/prod parity stops being a hop through a VM and becomes the default: the same kernel
family, the same `/dev/shm` tmpfs the ring defaults to, the same rustls-only dependency
graph, the same shell that runs the live session. The x86_64 little-endian assumption
([ADR-0006](0006-signal-schema-and-spsc-ring.md)) is unchanged and now describes the dev
box literally rather than by construction.

**2. `scripts/install-ubuntu.sh` is the one setup path and `./run.sh` the one entry
point.** The installer is idempotent, has a `--dry-run` that changes nothing, and exists
because three specific things bite on 24.04 and each fails in a way that reads as
something else:

- apt's `rustc` is older than the workspace's `rust-version = "1.85"` **and sits on PATH
  ahead of rustup's**, so the installer verifies the version that actually resolves
  rather than that `cargo` exists — "cargo is installed" and "cargo is new enough" are
  different facts and only the second one compiles.
- 24.04 enforces PEP 668, so `pip install` into the system Python fails; everything
  Python lives in a repo-local, gitignored `.venv`.
- `python3 -m venv` needs the separate `python3-venv` package, which is not implied by
  `python3`.

It also refuses to run as root — a sudo'd install leaves root-owned `~/.cargo`, `.venv`
and `target/` that the next plain `./run.sh` cannot write, and that failure surfaces
much later as unexplained permission errors. `.env` is scaffolded from `.env.example` at
mode 0600 with placeholders only; no key is ever written by a script. No `libssl-dev`:
the workspace is rustls end to end.

**3. Windows keeps its CI matrix cell, and that is now the *only* thing holding the
portability claim up.** `.github/workflows/ci.yml` still runs the whole gate — fmt,
clippy, `cargo test --workspace`, the IPC examples, and the Python↔Rust round-trip — on
`windows-latest` as well as `ubuntu-latest`.

Keeping it is the deliberate half of this ADR. Under ADR-0007 a Windows regression was
caught within minutes because someone was compiling there; now nothing local would
notice, so the cell is not redundant coverage — it *is* the coverage. Dropping it would
not remove the claim from the code, it would remove our knowledge of whether the claim
still holds, and the mmap-file transport was chosen over `shmem-ipc`/`iceoryx2`
*precisely* because it is portable ([ADR-0006](0006-signal-schema-and-spsc-ring.md)).
A portability property that is asserted and untested is worse than one that was never
claimed: the next person reads the module doc, believes it, and finds out at deployment.

**4. WSL2 stays documented, and stays out of the way.** `scripts/wsl-bootstrap.sh` and
the WSL2 section of `docs/DEVELOPMENT.md` remain for anyone whose machine is a Windows
one; they move below the Ubuntu path rather than in front of it. Deleting them would
cost a contributor the `/mnt/c` and Python-3.12-pinning lessons that were paid for once
already.

**5. No code changes, and that is the point.** The ring is still an mmap file, its
default path is still `/dev/shm/axon-*.ring` on Linux and any temp path on Windows, the
Linux-only zero-copy transports stay deferred behind the same `Producer`/`Consumer`
API, and `python/tests/conftest.py` still honours `CARGO_TARGET_DIR` — the escape hatch
WSL2 needed, harmless and occasionally useful elsewhere. This ADR records where people
work, not what the software does; if it had required a code change, ADR-0007's central
claim (OS-agnostic by construction) would have been wrong.

## Consequences

- **+** The environment that runs the tests is the environment that runs the venue
  session. "Works on my machine" and "works in prod" are now the same sentence, which
  matters most for the things that are hardest to test: file modes on `.env`, `/dev/shm`
  behaviour, and the live testnet paths.
- **+** The `/mnt/c` build penalty and the split `CARGO_TARGET_DIR` workflow are gone,
  along with the class of confusion where Windows and WSL2 held two different `target/`
  directories and one of them was stale.
- **+** A new contributor runs two commands and gets a green gate, instead of installing
  a VM first. `./run.sh doctor` reports what is present and what is missing.
- **−** **Windows is now exercised only in CI.** A Windows-only breakage is invisible
  until a pull request runs, and nobody on the project has a Windows box to bisect it on.
  Accepted knowingly: the alternative is not "cheaper maintenance", it is dropping a
  documented property of `axon-ipc`.
- **−** Two installers now exist (`install-ubuntu.sh`, `wsl-bootstrap.sh`) and Windows
  has neither — a Windows-native contributor still assembles rustup plus a venv by hand
  and runs `scripts/check.sh` directly. Left unsolved rather than papered over with a
  third script nobody would run.
- **−** The dev box is one ordinary Linux host, not a colocated one. Every latency
  number measured here is a *relative* number; `docs/05-latency-model.md`'s budgets are
  only meaningful from Tokyo, and running natively on Linux makes the numbers look more
  authoritative than they are.
- **−** ADR-0007's filename and title still say "Linux/WSL2 dev target", so the index
  shows the superseded decision under its original name. Deliberate: an ADR is a record
  of what was decided at the time, and renaming it would make the history unreadable.
  Its status line points here.

See [ADR-0007](0007-linux-wsl2-dev-target.md) (the decision this supersedes),
[ADR-0006](0006-signal-schema-and-spsc-ring.md) (the mmap-file transport and the
x86_64 assumption, both unchanged), [`docs/DEVELOPMENT.md`](../DEVELOPMENT.md),
[`scripts/install-ubuntu.sh`](../../scripts/install-ubuntu.sh),
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).
