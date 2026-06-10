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
these notes quoted ~8–12x on `deep`; that was a footer-cache artifact and
is corrected below.)

## Representative dev run (uncached)

| Fixture | Operation | min (ms) | median (ms) | vs full walk (min) |
|---|---|---:|---:|---:|
| `indexed` (4 row groups, OffsetIndex, ~25 pages/chunk) | `full_summary` (eager full-file walk) | ~87 | ~92 | 1x |
| `indexed` | `page header` one page | ~8.8 | ~9.5 | **~9.9x** |
| `indexed` | `page list` via OffsetIndex (`page_stubs`) | ~8.9 | ~9.6 | **~9.8x** |
| `indexed` | scoped `page list` one chunk (`pages`) | ~9.6 | ~11.0 | **~9.1x** |
| `deep` (100 row groups, no OffsetIndex) | `full_summary` (eager full-file walk) | ~441 | ~471 | 1x |
| `deep` | `page header` one chunk | ~285 | ~306 | **~1.5x** |
| `deep` | scoped `page list` one chunk (`pages`) | ~267 | ~303 | **~1.6x** |

(Numeric and high-cardinality string chunks measured; figures are the
representative range across both.)

## Reading the result — honestly

The page subcommands answer a single-page / single-chunk question while
touching one chunk's bytes; the eager pipeline (`full_summary` /
`--output-mode default`) walks every page header in the file. The size of
the win depends on how much of the total cost is the page walk vs the
**footer parse** that both paths pay:

- **OffsetIndex-present file with a modest footer (`indexed`):**
  ~**10x** faster. The footer is cheap to parse, so skipping the
  full-file page walk is most of the cost — the clean demonstration of
  the DoD claim. `page_stubs` reads page extents from the OffsetIndex
  with no per-page header reads; even the no-index one-chunk walk is ~9x
  because the eager walk reads ~1000 page headers (4 rg x 10 cols x ~25
  pages) vs one chunk's ~25.

- **File with a very large footer (`deep`, 100 row groups -> 1000
  chunks):** only ~**1.5x**. Here the footer parse (~270 ms, paid by
  *both* the page command and the eager walk) dominates, so the avoided
  page walk is a smaller fraction of the total. The footer-parse cost is
  the lazy core's *separate* concern, benchmarked in
  `test_eager_baseline.py`; it is not what the page surface targets.

So: the page subcommands reliably avoid the full-file page walk, and that
is a **substantial (~order-of-magnitude) speedup whenever the footer
parse is not itself the bottleneck**. On pathologically chunk-heavy files
the footer parse bounds the ratio — an honest limit worth stating rather
than hiding behind a cached best-case number.

Important caveats:

- `page(0)` is a direct one-page seek **only** with an OffsetIndex;
  without one it falls back to `len(pages())` and walks that one chunk's
  headers (still one chunk, never the whole file).
- These benchmarks scope `page list` to a single column chunk. An
  *unscoped* `page list` (all row groups, no OffsetIndex) walks every
  selected chunk and approaches the full-file cost — use `--row-group` /
  a column-chunk path, or an OffsetIndex-present file, to keep it cheap.

Numbers are machine-dependent; re-run to refresh. The point is the
**regime** (order-of-magnitude when footer-light; footer-bound
otherwise), not the exact figure.
