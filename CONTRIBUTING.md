# Contributing to parquet-analyzer

Thanks for your interest. This doc captures the conventions that
keep the codebase coherent over time. They apply to PRs from
humans and AI agents alike.

## Writing PRs

### Titles

PR titles should describe what the PR does, in language that
**makes sense in `git log` years from now**.

- **Use conventional-commit prefixes.** This repo follows the
  [Conventional Commits](https://www.conventionalcommits.org/)
  convention. Prefix every PR title (and squash-merge subject) with
  one of:
  - `feat:` — user-visible new functionality
  - `fix:` — bug fix
  - `docs:` — documentation only
  - `refactor:` — code restructure with no behavior change
  - `test:` — test-only changes
  - `perf:` — performance work
  - `build:` — build system, packaging, deps
  - `ci:` — CI / GitHub Actions / pre-commit hooks
  - `chore:` — maintenance that doesn't fit above (sweeps, cleanups,
    version bumps)
  Scope is optional and rarely worth it on a small repo —
  `feat(cli):` is fine but `feat:` alone is usually enough.
- **Keep them clean.** After the prefix, a title should describe
  the change in plain English, not its position in a workflow or
  who asked for it.
- **No planning jargon.** Don't include labels that exist only
  inside a session, a planning doc, or an issue's sub-numbering
  scheme. Anything matching `Slice <N>`, `Phase <N>`, `Step <N>`,
  `Round <N>`, `Block <N>`, `Iteration <N>`, `T<N>a`, `SF-<NNN>`,
  etc. is planning jargon — it's only meaningful inside one
  ephemeral context and looks like noise everywhere else.
- **No work-item linkage.** Don't put `closes #N`, `fixes #N`,
  `(#N)`, etc. in the title. Those belong in the description
  (where they're parsed by GitHub for auto-close linkage and
  rendered as clickable links). Putting them in the title pollutes
  the merge-commit subject for no benefit.

| Avoid | Prefer |
|---|---|
| `docs: tree-schema v0 — footer-layer kind catalog (Slice 4a foundation)` | `docs: tree-schema v0 — footer-layer kind catalog` |
| `feat: verb-noun subcommands (Slice 3, closes #7)` | `feat: verb-noun subcommands` (with `Closes #7` in description) |
| `[Phase 2] add lazy chunk walk` | `feat: add lazy chunk walk to ColumnChunk` |
| `fix: handle iteration 1 review feedback` | `fix: <whatever the actual fix is>` |
| `verb-noun subcommands` | `feat: verb-noun subcommands` (missing prefix) |

### Descriptions

- **Put work-item linkage here.** `Closes #N` / `Fixes #N` /
  `Refs #N` all belong in the body, where GitHub renders them
  as links and acts on them for auto-close.
- **Describe the final state, not the development chronology.** The
  PR description is for someone discovering the change later. They
  don't care that you tried approach A first and switched to B —
  they care what the code does now and why.
- **Self-contained.** A reader should be able to understand the PR
  from title + description + diff alone, without scrolling chat
  history or planning docs.

### Plan jargon in shared artifacts

The "no planning jargon" rule isn't limited to PR titles — it
applies to anything that outlives the immediate working context:

- PR bodies and inline review comments
- Commit messages (body included)
- Code comments and docstrings
- Documentation files (`docs/`, `README.md`)
- Test names and test docstrings

References to GitHub issues / PRs by number (`#21`, `PR #19`) are
durable — those are real artifacts with permanent URLs. References
to plan slices ("Slice 4a", "Phase 2 of the lazy core work") are
not — they only make sense to whoever was in the room when the
planning was done.

If a piece of historical context is genuinely worth preserving
("the deprecated `file_offset` field is unreliable per
PARQUET-2139"), link the durable artifact (the spec PR, the issue,
the relevant code) — don't paste an internal label.

## Code and docs

For the project's substantive design rules (what subcommand output
may contain, what the tree-node schema is), see:

- [`docs/output-principles.md`](docs/output-principles.md) — the
  contract for verb-noun subcommand outputs (footer-bounded,
  walk-free, escape-hatch model).
- [`docs/tree-schema.md`](docs/tree-schema.md) — the kind catalog
  for the v2 tree output surface.

Read both before adding fields to existing subcommands or
introducing new node kinds.

## Testing

- `hatch run dev:check` runs format + lint + type-check + tests +
  coverage. PRs should leave it green.
- **Coverage is gated per module, not in total** (enforced by
  `scripts/check_coverage.py`, wired into `dev:check`). Every module
  must independently meet a **95%** line-coverage floor. A per-module
  gate is deliberately stricter than a total gate: a well-tested module
  can't mask an under-tested one by averaging.
  - New modules are gated automatically at 95% — there is no way to add
    code that silently escapes the floor.
  - A module that is legitimately below the floor (legacy code not yet
    covered) must be listed in `KNOWN_GAPS` in
    `scripts/check_coverage.py`, pinned to a **baseline that may only
    ratchet up** and a **tracking issue**. Coverage dropping below a
    recorded baseline fails the gate. When a gap module reaches 95%, it
    must be removed from `KNOWN_GAPS`. These entries are tracked,
    temporary debt — not permanent exemptions.
- **Tests must assert behavior, not merely execute lines.** Coverage is
  a floor, not a goal; a test that runs a code path without asserting
  its observable result (returned value, emitted JSON/tree shape, raised
  error) does not count as covering it. For the serializer/layout code
  especially, assert on the emitted structure, not just that it didn't
  crash.
- Benchmarks live in `tests/bench/` and are excluded from the
  default test run. Run explicitly with `pytest tests/bench/
  --benchmark-only`. Bench numbers are operator-driven on a single
  machine at a time — there's no CI gating on them.

## Refactoring and tech debt

This codebase does not accept tech debt as a cost of shipping. The
final state of a change must look like it was designed cleanly from the
start, not like the path the author took to get there. Before opening a
PR, refactor within the boundary of what you changed so that **none of
the following remain**:

- **Dead code** — branches, helpers, or fallbacks that can't execute
  given how callers now use them. Remove them. If a defensive guard is
  genuinely needed for malformed input, make it raise (and test it), not
  fabricate plausible-looking placeholder values.
- **Untested compatibility branches** — an `if`/`else` added to handle
  an edge case must have a test that exercises it, or it doesn't belong.
- **Duplication introduced by the change** — repeated logic across
  near-identical functions should be factored once the third copy
  appears, unless a comment justifies why not.
- **Path-dependent residue** — comments or docstrings describing the
  development chronology ("originally we…", "for now…", "extremely
  unusual"), abandoned approaches, or session/review-round framing.
  Describe the final behavior instead.

This applies to the diff you are submitting, not the whole repository:
clean up what you touched, and file an issue for pre-existing debt you
notice but can't fix in scope.

## Issues

- Issue titles should describe the work; same rule as PRs.
- Sub-issue links (`#20 → #21`) are fine in body text.
- Use the existing labels (`enhancement`, `documentation`,
  `tracking`, `rfc`) to categorize; don't invent new ones without
  a reason.
