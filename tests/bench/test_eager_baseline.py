"""Baseline benchmarks against the ``ParquetFile`` lazy core.

These benchmarks compare against the ``eager-v0.4.0`` baseline JSON captured
before the lazy-core refactor (PR #16's harness ran against the pre-refactor
free-function eager pipeline). The test names are unchanged from that
baseline so ``--benchmark-compare=0001_eager-v0.4.0`` produces a direct
side-by-side ratio.

Two parameter groups, by intent:

- ``footer_only_equivalent`` -- operations the lazy core can serve from the
  footer alone (kv-metadata lookup, schema dump, column-encoding inspection).
  Pre-refactor these paid full eager-walk cost; post-refactor they're
  footer-only. **Target: >= 100x speedup vs baseline on `tall`.**
- ``full_parse`` -- operations the CLI ``default`` / ``html`` / ``segments``
  modes legitimately need (full segment list, full summary, full pages tree).
  Lazy core must not regress these. **Target: within 5% of baseline.**

Run:

    pytest tests/bench/ --benchmark-only

To compare a later run against the v0.4.0 baseline (the leading ``0001_`` is
pytest-benchmark's auto-prepended save-sequence prefix; the
``--benchmark-compare`` flag matches by prefix-anchored glob, so the
full filename stem is required). The ``-W`` flag suppresses
pytest-benchmark's "machine_info changed" warning, which fires noisily
on virtualized hosts where ``cpu.hz_actual`` jitters between runs even
on the same machine:

    pytest tests/bench/ --benchmark-only \\
        --benchmark-storage=file://tests/bench/baselines \\
        --benchmark-compare=0001_eager-v0.4.0 \\
        -W ignore::pytest_benchmark.logger.PytestBenchmarkWarning
"""

from __future__ import annotations

import pytest

from parquet_analyzer import ParquetFile


# ---------------------------------------------------------------------------
# Footer-only equivalents (lazy core target: >= 100x speedup on `tall`)
# ---------------------------------------------------------------------------


def _kv_metadata_lazy(path: str):
    pf = ParquetFile(path)
    try:
        return pf.kv_metadata
    finally:
        pf.close()


def _schema_lazy(path: str):
    pf = ParquetFile(path)
    try:
        return pf.schema
    finally:
        pf.close()


def _column_metadata_lazy(path: str):
    pf = ParquetFile(path)
    try:
        # Mirror the pre-refactor `footer["row_groups"][0]["columns"]` shape
        # by returning the first row group's column-chunk wrappers.
        return [
            {
                "path": cc.path,
                "type": cc.type,
                "encodings": cc.encodings,
                "codec": cc.codec,
                "num_values": cc.num_values,
            }
            for cc in pf.row_groups[0].columns
        ]
    finally:
        pf.close()


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_kv_metadata(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_kv_metadata_lazy, path)


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_schema(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_schema_lazy, path)


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_column_metadata(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_column_metadata_lazy, path)


# ---------------------------------------------------------------------------
# Full-parse paths (lazy core must not regress these by > 5%)
# ---------------------------------------------------------------------------


def _full_parse_lazy(path: str) -> tuple:
    """The complete pipeline used by `--output-mode default`."""
    pf = ParquetFile(path)
    try:
        # Order matters: full_summary triggers the eager walk; all_pages
        # reuses the same cached walk; footer is footer-only.
        summary = pf.full_summary
        footer = pf.footer
        pages = pf.all_pages()
        return summary, footer, pages
    finally:
        pf.close()


@pytest.mark.benchmark(group="full_parse")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_full_parse(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_full_parse_lazy, path)


@pytest.mark.benchmark(group="full_parse")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_segments_dump(benchmark, request, fixture_name):
    """The `--output-mode segments` workload: parse + JSON-transform every
    segment. Lazy core must not regress this -- `segments` mode genuinely
    needs to walk the whole file.

    The per-segment ``segment_to_json`` transform is kept in the timed
    workload so the comparison against ``0001_eager-v0.4.0`` (which had
    the same transform inside the timed block) stays apples-to-apples.
    """
    from parquet_analyzer import segment_to_json

    path = str(request.getfixturevalue(fixture_name))

    def run() -> list:
        pf = ParquetFile(path)
        try:
            return [segment_to_json(s) for s in pf.all_segments()]
        finally:
            pf.close()

    benchmark(run)
