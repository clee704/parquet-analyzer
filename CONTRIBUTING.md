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
start, not like the path the author took to get there. Keeping it that
way is a **recurring checkpoint**, not a one-time cleanup — you run it
at the end of every task (see "The refactoring checkpoint" below).

The checkpoint scans the code you touched, and the design it now sits
in, so that **none of the following remain**:

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

### The refactoring checkpoint

Run this **at every checkpoint** — before completing each task / opening
each PR, and again at the close of a multi-PR slice. It is a deliberate,
standing step in the workflow, not something to remember only when the
code feels messy. Treat it like the test gate: every PR passes through
it.

The checkpoint is two moves:

1. **Assess.** Re-read the code you touched and the design it sits in
   against the four criteria above. Dogfooding or implementing the next
   slice often exposes that an abstraction introduced earlier is now
   leaky, duplicated, or dead — this is the moment to catch it.
2. **Decide, per finding — fix now, or defer.** There is no third
   "leave it" option.
   - **Fix now (default)** when the cleanup is inside the boundary of
     what you changed and is low-risk. Do it in the same PR (or a small
     `refactor:` PR stacked on it).
   - **Defer by issue** when the finding is out of scope, large, or
     risky to fold in. File a tracking issue immediately and link it.
     Never leave a known problem unrecorded.

**Record the outcome in the PR.** The pull-request template has a
Refactoring-checkpoint section: list what you cleaned up and what you
deferred (with issue links). "Nothing to clean up" is a valid outcome,
but it must be stated — silence is not the same as having run the
checkpoint. This keeps the ritual auditable and unskippable.

The checkpoint operates on the diff you are submitting, not the whole
repository: clean up what you touched, and file an issue for
pre-existing debt you notice but can't fix in scope.

## Dogfooding

Tests prove the code does what the tests assert. They do **not** tell you
whether the tool is actually good to use — fast enough at scale, and
pleasant to drive. Those only show up when you use it the way a user
would. This codebase has twice been redirected by dogfooding that unit
tests sailed past: a feature that passed every test was scrapped because
driving it on a multi-GB file exposed a hidden cost cliff and a leaky
abstraction, and a navigation command that passed every test dumped a
474 KB wall of JSON for a single step on a real file. Neither was
visible from the small fixtures the tests use.

So dogfooding is a **recurring checkpoint**, the behavioral counterpart
to the refactoring checkpoint above.

### The dogfooding checkpoint

Run this **before completing any change that affects how the tool
behaves** — a new or changed subcommand, flag, output shape, performance
characteristic, or user-facing library API. (Pure-internal refactors,
docs, and test-only changes are exempt, though a quick sanity run is
cheap insurance.) It is a standing step, not something to do only when
something feels off.

The checkpoint is three moves:

1. **Drive it like a user.** Run the actual commands — chain them the
   way a real exploration would, read the output, follow the affordances.
   Don't settle for the unit tests having passed.
2. **On realistic data.** Small fixtures hide the problems that matter.
   Exercise the change on inputs that stress it: large / GB-scale files
   where performance is in play, and the edge-case shapes relevant to
   the change (e.g. files with and without an OffsetIndex, high page
   counts, wide schemas, empty or single-row files). Generate them if
   you don't have them, and **clean up large artifacts afterward.**
3. **Evaluate performance *and* UX, then decide per finding.** Time the
   operations that should be fast and confirm they are — and confirm the
   cost model is honest (no hidden cliff, no work that scales with the
   wrong dimension). Equally, judge the experience: is the output
   readable and appropriately sized, are the affordances clear, is
   anything surprising or awkward? For each finding — **fix now or defer
   by issue**, the same rule as the refactoring checkpoint. There is no
   "leave it".

**Record the outcome in the PR.** The pull-request template has a
Dogfooding section: name the scenarios you exercised (files, sizes,
edge cases), the perf you observed, the UX assessment, and what you
fixed or deferred. "Exercised X and Y; perf and UX are good; nothing to
fix" is a valid outcome — but it must be stated, with the scenarios
named, so the claim is grounded and auditable. A bare "dogfooded it" is
not enough.

Be honest about scope and cost. If a number looks good, say where it
came from; if a win is smaller or narrower than expected, say so rather
than overselling it. Grounding every claim in something you actually
observed is the whole point of the checkpoint.

**A material fix is unreviewed code.** When the checkpoint makes you
*change* code rather than just confirm it's fine, that change went in
after your normal review pass. If it is material — new logic, a new
flag, a behavior or output-shape change, not a one-line tweak — put it
back through your review process (e.g. the self-review loop) before
merging. The sequence is dogfooding finding → fix → **re-review** → ship,
not finding → fix → ship.

## Issues

- Issue titles should describe the work; same rule as PRs.
- Sub-issue links (`#20 → #21`) are fine in body text.
- Use the existing labels (`enhancement`, `documentation`,
  `tracking`, `rfc`) to categorize; don't invent new ones without
  a reason.
