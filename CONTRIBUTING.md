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
- Coverage gate is **90%** (enforced in `pyproject.toml`). New code
  should not regress it.
- Benchmarks live in `tests/bench/` and are excluded from the
  default test run. Run explicitly with `pytest tests/bench/
  --benchmark-only`. Bench numbers are operator-driven on a single
  machine at a time — there's no CI gating on them.

## Issues

- Issue titles should describe the work; same rule as PRs.
- Sub-issue links (`#20 → #21`) are fine in body text.
- Use the existing labels (`enhancement`, `documentation`,
  `tracking`, `rfc`) to categorize; don't invent new ones without
  a reason.
