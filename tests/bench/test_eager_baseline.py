"""Baseline benchmarks: eager ``parse_parquet_file`` against current code.

These are captured **before** any lazy-core work lands. They establish
the bar that Slice 2's lazy core has to beat for "footer-only"
equivalents (>= 100x faster) and stay within (<= 5% regression) for the
full-parse paths.

Two parameter groups, by intent:

- ``footer_only_equivalent`` -- operations a lazy core could serve from
  the footer alone (kv-metadata lookup, schema dump, column-encoding
  inspection). Today they all pay full-parse cost.
- ``full_parse`` -- operations that genuinely need to walk the whole
  file (the ``segments`` / ``html`` / ``default`` modes). The lazy
  core must not slow these down.

Run:

    pytest tests/bench/ --benchmark-only

To save a baseline JSON:

    pytest tests/bench/ --benchmark-only \\
        --benchmark-storage=file://tests/bench/baselines \\
        --benchmark-save=eager-v0.4.0

To compare a later run against this baseline (the leading ``0001_`` is
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

from parquet_analyzer._core import (
    find_footer_segment,
    get_pages,
    get_summary,
    parse_parquet_file,
    segment_to_json,
)


# ---------------------------------------------------------------------------
# Footer-only equivalents (eager today; lazy core targets >= 100x speedup)
# ---------------------------------------------------------------------------


def _kv_metadata_eager(path: str) -> dict[str, str] | None:
    """Today's path to read kv metadata: full eager parse, then walk."""
    segments, _ = parse_parquet_file(path)
    footer = segment_to_json(find_footer_segment(segments))
    return footer.get("key_value_metadata")


def _schema_eager(path: str) -> list:
    segments, _ = parse_parquet_file(path)
    footer = segment_to_json(find_footer_segment(segments))
    return footer.get("schema", [])


def _column_metadata_eager(path: str) -> list:
    """Today's path to inspect per-column metadata for the first row group."""
    segments, _ = parse_parquet_file(path)
    footer = segment_to_json(find_footer_segment(segments))
    return footer["row_groups"][0]["columns"]


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_kv_metadata(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_kv_metadata_eager, path)


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_schema(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_schema_eager, path)


@pytest.mark.benchmark(group="footer_only_equivalent")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_column_metadata(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_column_metadata_eager, path)


# ---------------------------------------------------------------------------
# Full-parse paths (lazy core must not regress these by > 5%)
# ---------------------------------------------------------------------------


def _full_parse(path: str) -> tuple:
    """The complete eager pipeline used by `--output-mode default`."""
    segments, column_offset_map = parse_parquet_file(path)
    footer = segment_to_json(find_footer_segment(segments))
    summary = get_summary(footer, segments)
    pages = get_pages(segments, column_offset_map)
    return summary, footer, pages


@pytest.mark.benchmark(group="full_parse")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_full_parse(benchmark, request, fixture_name):
    path = str(request.getfixturevalue(fixture_name))
    benchmark(_full_parse, path)


@pytest.mark.benchmark(group="full_parse")
@pytest.mark.parametrize(
    "fixture_name", ["wide_parquet", "tall_parquet", "deep_parquet"]
)
def test_baseline_segments_dump(benchmark, request, fixture_name):
    """The `--output-mode segments` workload: parse + emit every segment
    as JSON. Lazy core must not regress this -- `segments` mode genuinely
    needs to walk the whole file."""
    path = str(request.getfixturevalue(fixture_name))

    def run() -> list:
        segments, _ = parse_parquet_file(path)
        return [segment_to_json(s) for s in segments]

    benchmark(run)
