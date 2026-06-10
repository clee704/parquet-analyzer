# Page-command benchmark — demonstrated ratios

Records that the page subcommands (#21) avoid the full-file page-header
walk the eager pipeline pays. Source: `tests/bench/test_page_commands.py`.

Run:

```
pytest tests/bench/test_page_commands.py --benchmark-only \
    -o addopts="" \
    -W ignore::pytest_benchmark.logger.PytestBenchmarkWarning \
    --benchmark-columns=min,median,rounds
```

(`-o addopts=""` overrides the repo-wide `--ignore=tests/bench`, which
keeps the benchmarks out of the normal `hatch run dev:check` suite.)

All opens use `use_cache=False`, so every timed run pays the **real**
footer parse + the op — no warm footer-cache hits. (An earlier draft of
these notes quoted ~8–12x as a cached best-case; the figures below are
the honest uncached measurements.)

## Representative dev run (uncached)

| Fixture | Operation | min (ms) | median (ms) | vs full walk (min) |
|---|---|---:|---:|---:|
| `paged` (12 row groups, **no** OffsetIndex, ~38 pages/chunk) | `full_summary` (eager full-file walk) | ~173 | ~212 | 1x |
| `paged` | `page header` one page | ~7.7–11.5 | ~12–15 | **~15–22x** |
| `paged` | scoped `page list` one chunk (`pages`) | ~7.8–11.4 | ~12–17 | **~15–22x** |
| `indexed` (4 row groups, OffsetIndex, ~25 pages/chunk) | `full_summary` (eager full-file walk) | ~96 | ~118 | 1x |
| `indexed` | `page header` one page | ~8.7–9.6 | ~12–16 | **~10–11x** |
| `indexed` | `page list` via OffsetIndex (`page_stubs`) | ~8.8–9.1 | ~11–15 | **~11x** |
| `indexed` | scoped `page list` one chunk (`pages`) | ~9.0–10.0 | ~15–17 | **~10x** |

(Each operation is measured on a numeric chunk and a string chunk; the
range spans both. The string chunk is consistently a touch faster. These
are one representative run: `paged`'s ratio in particular is **noisy
run-to-run** — ~10–40x has been observed across runs — because its
denominator is a single few-millisecond op sensitive to GC and the OS
file cache. The stable, reproducible claim is the **order of magnitude**,
not the exact multiple; re-run to refresh.)

## Reading the result — honestly

The page subcommands answer a single-page / single-chunk question while
touching one chunk's bytes; the eager pipeline (`full_summary` /
`--output-mode default`) walks every page header in the file. The avoided
work is the per-page header walk across all chunks; the **footer parse**
is paid by both paths, so the ratio is largest when the page walk — not
the footer — is the bulk of the eager cost:

- **No-OffsetIndex, page-walk-dominated file (`paged`):** an
  **order of magnitude** faster (~15–22x in the representative run above,
  ~10–40x across runs). This is the honest "no fast path" case — there is
  no OffsetIndex to seek with, so the eager walk parses every page header
  (12 rg × 3 cols × ~38 pages ≈ 1300+ headers) while a single scoped page
  op parses one chunk's ~38. The footer is only a few % of the eager walk,
  so almost all of it is genuinely avoided.

- **OffsetIndex-present file (`indexed`):** ~**10–11x** faster (this one
  is tight run-to-run — its full-walk reference is steadier).
  `page_stubs` reads page extents straight from the OffsetIndex with no
  per-page header reads; even the no-index one-chunk walk is ~10x because
  the eager walk reads ~1000 page headers (4 rg × 10 cols × ~25) vs one
  chunk's ~25.

So the page subcommands reliably avoid the full-file page walk, for an
**order-of-magnitude speedup** — including on files with **no
OffsetIndex** (`paged`), where the win comes purely from scoping the page
walk to one chunk rather than from any index fast path.

The ratio shrinks toward ~1x only when the shared **footer parse** is
itself the bottleneck — files with thousands of column chunks, where the
~0.25 ms/chunk footer parse (paid by both paths) dwarfs the avoided page
walk. That footer-parse cost is the lazy core's *separate* concern,
benchmarked in `test_eager_baseline.py` (which is what the `deep` fixture
— 100 row groups → 1000 chunks — exists to measure); it is not what the
page surface targets, so this benchmark uses footer-light fixtures.

Important caveats:

- `page(0)` is a direct one-page seek **only** with an OffsetIndex;
  without one it falls back to `len(pages())` and walks that one chunk's
  headers (still one chunk, never the whole file).
- These benchmarks scope `page list` to a single column chunk. An
  *unscoped* `page list` (all row groups, no OffsetIndex) walks every
  selected chunk and approaches the full-file cost — use `--row-group` /
  a column-chunk path, or an OffsetIndex-present file, to keep it cheap.

Numbers are machine-dependent; re-run to refresh. The point is the
**regime** (order-of-magnitude whenever the footer parse is not itself
the bottleneck), not the exact figure.
