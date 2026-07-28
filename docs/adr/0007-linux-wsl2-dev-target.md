# ADR-0007 — Linux/WSL2 as the primary dev & deploy target; portable mmap for Windows

**Status:** Superseded by [ADR-0024](0024-native-ubuntu-dev-target.md) · **Date:** 2026-07-12

## Context

[ADR-0002](0002-python-rust-boundary.md) left the Linux/Windows dev story `OPEN`,
leaning toward WSL2/Docker for prod-parity with a portable mmap fallback for local
dev. Production will run on Linux (colocation in AWS Tokyo `ap-northeast-1` is the
biggest HFT lever — [05](../05-latency-model.md)). Development currently happens on a
Windows 11 machine.

Two facts made this decidable in Phase 1: (a) the Windows box already has VS 2022
Build Tools (MSVC `cl.exe` + Windows SDK), so Rust's default `x86_64-pc-windows-msvc`
toolchain works with no extra heavyweight install; (b) the shared-memory transport
([ADR-0006](0006-signal-schema-and-spsc-ring.md)) is a memory-mapped **file**, which
is portable and coherent on both OSes.

## Decision

**Primary target is Linux; WSL2 (Ubuntu) is the primary development environment**
for prod-parity, with the repo mounted from Windows via `/mnt/c`. Because the code
is OS-agnostic by construction, **Windows-native dev is fully supported too** and is
used as a fast local compile/test loop today (the same tests pass on both).

- The ring's default path is `/dev/shm/axon-*.ring` on Linux (tmpfs, RAM-backed);
  on Windows any temp path works.
- CI runs the full gate (fmt + clippy + Rust tests + Python tests + the
  cross-language round-trip) on **both** `ubuntu-latest` and `windows-latest`, so the
  cross-platform claim is continuously proven, not asserted.
- The Linux-only zero-copy IPC options (`shmem-ipc`, `iceoryx2`) stay deferred behind
  the same `Producer`/`Consumer` API — they become available once we are Linux-native,
  but are not required.

## Consequences

- **+** One codebase runs on the dev OS and the prod OS with no `#[cfg]` forks in the
  hot path; parity is a CI matrix cell, not a hope.
- **+** No blocking on the WSL2 install (which needs admin + a reboot): the entire
  foundation was built and tested Windows-native while WSL2 is set up in parallel.
- **+** Clear migration: when a strategy needs the lowest tail, moving to a
  Linux-only transport is an adapter swap, not a rewrite.
- **−** Two OSes to keep green in CI (accepted — it is the guarantee we want).
- **−** WSL2 file access via `/mnt/c` is slower than a native Linux checkout; if build
  times bite, clone into the WSL2 home filesystem (a workflow choice, not an
  architecture change).

See [ADR-0006](0006-signal-schema-and-spsc-ring.md), [02](../02-python-rust-boundary.md),
[07](../07-parity-and-testing.md) (the "same code, different numbers across platforms"
caution — pinned toolchain + x86_64 assumption address it).
