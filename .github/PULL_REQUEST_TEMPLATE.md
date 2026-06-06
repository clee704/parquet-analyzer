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

## Tests

- [ ] `hatch run dev:check` is green (format, lint, type-check, tests, per-module 95% coverage).
- [ ] New / changed code paths assert observable behavior, not just execute lines.
