# Subcommand output design principles

The contract behind `parquet-analyzer`'s v1 verb-noun CLI output
(`file kv/schema/...`, `column show/list`, `rowgroup list/show`).

## The contract

> v1 subcommand output is **footer-bounded and walk-free**. Every
> field is derivable from the parsed footer plus, at most, one
> additional parse of an index thrift the writer already emitted
> (OffsetIndex, ColumnIndex, BloomFilter header). Nothing in v1 may
> walk per-page thrift headers or read page bodies — that's the
> `page` subcommand surface (tracked in #21).

This is the one hard rule. It exists because the AI-agent and
human-investigation use cases that motivate this tool need a
predictable "instant, cheap, doesn't hit the disk too hard" surface
for footer-level questions. Crossing the page-walk line silently
inside `column show` would break that promise.

## Examples of what fits the contract

Roughly in order of increasing cost. All four kinds appear in the
current v1 output:

| Kind | Cost | Examples |
|---|---|---|
| Direct footer fields | O(0) | `num_rows`, `codec`, `encodings`, `data_page_offset`, `statistics` |
| Trivial derivations | O(1) per field | `footer_offset`, `has_dictionary`, `chunk_offset`, `column` (dot-joined `path`) |
| Small footer aggregates | O(row_groups × columns) over the parsed footer | file `compressed_page_size`, rg `total_compressed_size`, `total_num_values` on `column show` |
| Bounded extra index parses | one extra parse per chunk asked about | `num_pages` (OffsetIndex); future per-page min/max could use ColumnIndex the same way |

## The honesty pattern for fields near the boundary

When a field is *sometimes* computable cheaply and *sometimes* would
require crossing the page-walk line, the output says so honestly
instead of silently doing the walk. The canonical example is
`num_pages`: when the writer didn't emit an OffsetIndex, computing
the count would need a page walk, so v1 emits
`num_pages: null, num_pages_known: false`.

Use the same `<field>_known: bool` pattern whenever a future field
has the same shape.

## Guideline for adding a new field

The contract above is the only thing that's load-bearing. Past that,
use judgment:

- **Derived fields should be reasonably inexpensive.** Footer math
  and one extra index parse are fine; anything that would walk page
  headers crosses the contract line (see escape hatches below).
- **Prefer surfacing existing data over inventing new derivations.**
  When in doubt, lean toward "no, the consumer can compose it".
- **Symmetry helps.** If you're adding a field at one level (file /
  rowgroup / column), and an analogous field would make sense at
  another level, name and shape them consistently.

That's it. The contributor adding the field gets to decide whether
it earns its place — there's no committee.

## Escape hatches: explicit opt-in for fields beyond the contract

The contract describes the **default** behaviour of every subcommand.
It does not forbid subcommands from offering explicit opt-in flags
that cross the boundary, as long as the flag name signals the cost.

Two complementary mechanisms:

**1. The `page` subcommand surface IS an escape hatch.** When a
consumer wants page-level information (counts, headers, bodies,
decoded values), the right answer is to run a `page` verb. The
verb name does the signalling — running `page list` makes it
obvious you're paying for a page walk. This is the primary
escape hatch for everything that requires per-page work.

Example: getting `num_pages` for a file without OffsetIndex is
`parquet-analyzer page list <path> --column foo --row-group N`
and reading the resulting `total` field.

**2. Per-subcommand opt-in flags for "shape X plus walk-required
field Y in one go".** When a consumer wants the *shape* of an
existing footer-only subcommand (`column show`, `rowgroup show`)
but with one or more walk-required fields populated, the
subcommand may offer an opt-in flag named for the cost — e.g.
`column show <path> --column foo --walk-pages` flips
`num_pages_known` to `true` after a per-chunk page walk.

Rules for these flags:
- The default (flag absent) **must** stay footer-only, per the
  contract above. The flag opts the *single invocation* in.
- The flag name should describe the cost being incurred
  (`--walk-pages`, `--deep`, `--decode-values`), not the field
  being populated (`--include-num-pages` is the wrong shape — it
  hides the cost behind the value).
- When the flag enables a field that previously emitted as
  `<field>_known: false`, the post-walk output flips it to
  `true` (same honesty pattern, just both states now reachable).

The page-subcommand work tracked in #21 lands the page-walk
infrastructure that makes these flags possible. Until then, the
only escape is the Python API (`cc.pages()` on the wrapper).

## What this implies for the page subcommands (#21)

When the page-level subcommands land:

- `page header` / `page list` follow the same shape one level deeper:
  page header fields are direct, computed offsets are trivial
  derivations.
- `page extract` and `page decode` cross into body access and are
  explicitly opt-in by the verb name. They're allowed because the
  consumer is asking for exactly that work.
- The first per-subcommand escape-hatch flag (`column show
  --walk-pages`, and the equivalent on `column list` /
  `rowgroup show`) lands there, since that work is when the page-walk
  CLI infrastructure becomes available.
- `num_pages_known: true` becomes a reachable state on
  `--walk-pages` invocations (in addition to the existing
  OffsetIndex-present case).

The contract — footer-bounded and walk-free by default, with the
`page` surface and per-subcommand opt-in flags as the explicit
escape hatches for everything beyond — stays the same.

---

*Origin: synthesized during the discussion in PR #19 around
`num_pages` reporting and cross-row-group aggregates.*
