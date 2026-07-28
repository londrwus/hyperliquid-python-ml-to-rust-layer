# ADR-0001 — Record architecture decisions

**Status:** Accepted · **Date:** 2026-07-12

## Context

This project spans three domains (Python ML, Rust systems, crypto execution) and will be
built over many phases and, likely, by multiple people. Decisions made now (the language
boundary, the fidelity strategy, the provider abstraction) are expensive to reverse and easy
to forget the *reasons* for. Six months from now, "why is Python off the hot loop?" or "why
FP32?" must have a findable answer.

## Decision

We will keep **Architecture Decision Records** in `docs/adr/`, one file per significant
decision, using the Nygard format (Context / Decision / Consequences / Status). ADRs are
immutable once Accepted; a reversal is a new ADR that supersedes the old one.

## Consequences

- **+** A durable, greppable record of *why*, not just *what*.
- **+** New contributors can read the ADRs and understand the shape of the system quickly.
- **−** Small ongoing discipline cost: significant decisions must be written down.
- Numbered sequentially (`NNNN-kebab-title.md`); index maintained in `adr/README.md`.
