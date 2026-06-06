<!--
  Title: use a conventional-commit prefix (feat/fix/docs/refactor/test/perf/build/ci/chore).
  See CONTRIBUTING.md for the full conventions.
-->

## What & why

<!-- What does this change do, and why is it needed? Enough context for a
     reviewer to understand the change without reading every line. Describe
     the final behavior, not the path you took to get there. -->

## Refactoring checkpoint

<!-- Required on every PR — see "The refactoring checkpoint" in CONTRIBUTING.md.
     Assess the code you touched (and the design it now sits in) for dead code,
     untested compatibility branches, duplication, and path-dependent residue.
     For each finding, either fix it here or defer it with a tracking issue.
     "Nothing to clean up" is a valid answer, but say so — don't leave this blank. -->

- [ ] I ran the refactoring checkpoint on the touched code and its design context.
- **Fixed in this PR:** <!-- e.g. removed dead fallback in _core.read_pages; deduped stub rendering — or "nothing to clean up" -->
- **Deferred (issue links):** <!-- e.g. #41 (row_groups dedup, out of scope) — or "none" -->

## Dogfooding

<!-- Required for any change that affects behavior (subcommand/flag/output/perf/
     library API). See "The dogfooding checkpoint" in CONTRIBUTING.md. Drive the
     tool like a user on realistic data (large/GB-scale where perf matters, plus
     edge-case shapes), evaluate performance AND UX, and fix or defer each finding.
     Name the scenarios — a bare "dogfooded it" is not enough. Exempt for
     pure-internal refactors / docs / test-only changes (say so). -->

- [ ] I drove the change on realistic data and assessed performance + UX (or: this PR is exempt — internal/docs/test-only).
- **Scenarios exercised:** <!-- e.g. 1.2 GB file (120 row groups × 40 cols), no-OffsetIndex variant, high-page-count column -->
- **Perf / UX findings:** <!-- what you observed, honestly scoped — or "perf and UX good, nothing to fix" -->
- **Fixed / deferred:** <!-- e.g. capped page listing with --limit; deferred #N — or "none" -->

## Tests

- [ ] `hatch run dev:check` is green (format, lint, type-check, tests, per-module 95% coverage).
- [ ] New / changed code paths assert observable behavior, not just execute lines.
