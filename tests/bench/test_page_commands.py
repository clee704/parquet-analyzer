"""Benchmarks for the page subcommands (#21): they answer page-level
questions while touching a single chunk or page, instead of paying the
full-file page-header walk the eager pipeline does.

The page subcommands (``page list / header / extract / decode`` and
``column show --walk-pages``) are backed by the lazy ``ParquetFile`` /
``ColumnChunk`` API exercised here:

- ``ColumnChunk.page_stubs()`` lists a chunk's pages from the
  **OffsetIndex** -- one bounded thrift parse, no per-page header reads
  (the ``page list`` fast path; OffsetIndex-present files only).
- ``ColumnChunk.pages()`` walks **one chunk's** page headers (the
  scoped ``page list`` / ``--walk-pages`` path).
- ``ColumnChunk.page(i)`` seeks to one page **directly** when an
  OffsetIndex is present; **without** one it falls back to
  ``len(pages())`` and so is bounded to that one chunk's headers (still
  one chunk, never the whole file).

Each is compared against ``full_file_walk`` -- ``pf.full_summary``, the
eager "summarise every page in the file" workload that the page
subcommands let you avoid when you only care about one page or chunk.

**All opens use ``use_cache=False``** so every timed run pays the real
footer parse + the op (no warm footer-cache hits) -- the comparison is
of the actual work each path does, with the footer parse as the shared
cost on both sides.

Two reference fixtures (see ``generate.py``):

- ``paged`` -- 12 row groups, **no** OffsetIndex, a small
  ``data_page_size`` so each chunk holds ~tens of pages. The eager walk
  parses every page header across all chunks; a single scoped page op
  parses one chunk's -- the page-walk-avoidance win *without* the
  OffsetIndex fast path. The footer is only a few % of the full walk.
- ``indexed`` -- 4 row groups, OffsetIndex present, ~25 pages per chunk.
  ``page list`` reads the OffsetIndex (no header reads) where the eager
  walk reads every header.

These are excluded from the normal suite (``--ignore=tests/bench``); run
explicitly:

    pytest tests/bench/test_page_commands.py --benchmark-only -o addopts="" \\
        -W ignore::pytest_benchmark.logger.PytestBenchmarkWarning

The ``page_command`` group avoids the eager ``full_file_walk`` for an
~**order-of-magnitude** speedup on both a page-walk-dominated,
no-OffsetIndex file (``paged``) and an OffsetIndex-present file
(``indexed``). The win shrinks toward ~1x only when the shared footer
parse is itself the bottleneck (files with thousands of chunks) -- that
regime is the lazy core's separate concern, measured in
``test_eager_baseline.py``,
not here. Representative ratios (min + median, uncached) and the honest
caveats are recorded in ``baselines/page-commands-notes.md``.
"""

from __future__ import annotations

import pytest

from parquet_analyzer import ParquetFile

# A numeric chunk and a string chunk, so the page op cost isn't read off a
# single column type. Indices into the generated schema (type pool cycles
# int64, float64, string_low_card, string_high_card). Index 2 (a string
# chunk) exists on both the 3-column ``paged`` and 10-column ``indexed``
# fixtures.
_CHUNK_COLS = [0, 2]


# ---------------------------------------------------------------------------
# Reference: the eager full-file page walk the page commands avoid
# ---------------------------------------------------------------------------


def _full_file_walk(path: str):
    """``full_summary`` walks every page header in the file -- the cost the
    page subcommands let a consumer skip when they want one page/chunk."""
    pf = ParquetFile(path, use_cache=False)
    try:
        return pf.full_summary
    finally:
        pf.close()


@pytest.mark.benchmark(group="full_file_walk")
@pytest.mark.parametrize("fixture_name", ["paged_parquet", "indexed_parquet"])
def test_full_file_walk_reference(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_full_file_walk, path)


# ---------------------------------------------------------------------------
# Page commands: touch one chunk / one page, not the whole file
# ---------------------------------------------------------------------------


def _page_header_single(path: str, col_index: int):
    """``page header``: open + footer + access one page. Direct seek with an
    OffsetIndex; otherwise bounded to this one chunk's headers."""
    pf = ParquetFile(path, use_cache=False)
    try:
        return pf.row_groups[0].columns[col_index].page(0).type
    finally:
        pf.close()


def _page_list_offset_index(path: str, col_index: int):
    """``page list`` on an OffsetIndex-present chunk: read page extents from
    the OffsetIndex, no per-page header reads."""
    pf = ParquetFile(path, use_cache=False)
    try:
        return pf.row_groups[0].columns[col_index].page_stubs()
    finally:
        pf.close()


def _page_list_one_chunk_walk(path: str, col_index: int):
    """Scoped ``page list``: walk ONE chunk's page headers (not the whole
    file)."""
    pf = ParquetFile(path, use_cache=False)
    try:
        return pf.row_groups[0].columns[col_index].pages()
    finally:
        pf.close()


@pytest.mark.benchmark(group="page_command")
@pytest.mark.parametrize("fixture_name", ["paged_parquet", "indexed_parquet"])
@pytest.mark.parametrize("col_index", _CHUNK_COLS)
def test_page_header_single(benchmark, request, fixture_name, col_index):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_page_header_single, path, col_index)


@pytest.mark.benchmark(group="page_command")
@pytest.mark.parametrize("col_index", _CHUNK_COLS)
def test_page_list_offset_index(benchmark, indexed_parquet, col_index):
    """The OffsetIndex fast path -- only meaningful on the indexed fixture."""
    benchmark(_page_list_offset_index, str(indexed_parquet), col_index)


@pytest.mark.benchmark(group="page_command")
@pytest.mark.parametrize("fixture_name", ["paged_parquet", "indexed_parquet"])
@pytest.mark.parametrize("col_index", _CHUNK_COLS)
def test_page_list_one_chunk_walk(benchmark, request, fixture_name, col_index):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_page_list_one_chunk_walk, path, col_index)
