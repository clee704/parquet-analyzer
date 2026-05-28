"""Unit tests for the verb-noun subcommand surface (`parquet_analyzer._subcommands`).

Subcommand tests use ``_FakeParquetFile`` rather than real parquet files so
they exercise the argparse wiring, dispatch, JSON envelope, and error
contract in isolation. The fake **raises** if any eager API is touched
(``full_summary``, ``all_pages``, ``all_segments``, ``column_offset_map``)
to guard against accidental page walks — Slice 3's invariant is that no
subcommand triggers an eager walk.

End-to-end tests against real pyarrow-generated parquet files live in
``test_integration.py``.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from typing import Sequence

import pytest

from parquet_analyzer import _subcommands, cli


# ---------------------------------------------------------------------------
# Fake ParquetFile that raises on eager APIs
# ---------------------------------------------------------------------------


class _EagerWalkBlocked(Exception):
    """Raised by the fake when an eager API is touched.

    Surfaces as a test failure (caught by pytest) with a clear message
    identifying which API the handler accidentally hit.
    """


class _FakeColumnChunk:
    def __init__(self, path, num_pages_value=None, has_offset_index=False):
        self.path = path
        self._num_pages_value = num_pages_value
        self.has_offset_index = has_offset_index

    @property
    def num_pages(self):
        if not self.has_offset_index:
            raise _EagerWalkBlocked(
                "cc.num_pages called without has_offset_index — would walk pages"
            )
        return self._num_pages_value


class _FakeRowGroup:
    def __init__(self, columns):
        self.columns = columns


class _FakeParquetFile:
    """Minimal ParquetFile stand-in for subcommand tests.

    Constructed with a `footer` dict matching the real ``pf.footer`` shape.
    Eager APIs raise on access so any handler that accidentally walks
    pages fails the test loudly.
    """

    def __init__(
        self,
        *,
        footer=None,
        num_rows=0,
        num_columns=0,
        kv_metadata=None,
        footer_summary=None,
        row_group_wrappers=None,
    ):
        self.footer = footer if footer is not None else {"row_groups": []}
        self.num_rows = num_rows
        self.num_row_groups = len(self.footer.get("row_groups", []))
        self.num_columns = num_columns
        self.schema = self.footer.get("schema", [])
        self.kv_metadata = kv_metadata if kv_metadata is not None else []
        self.footer_summary = footer_summary or {
            "num_rows": num_rows,
            "num_row_groups": self.num_row_groups,
            "num_columns": num_columns,
            "uncompressed_page_size": 0,
            "compressed_page_size": 0,
            "column_index_size": 0,
            "offset_index_size": 0,
            "bloom_filter_size": 0,
            "footer_size": 0,
            "file_size": 0,
        }
        self.row_groups = row_group_wrappers if row_group_wrappers is not None else []
        self.closed = False

    # --- eager APIs intentionally raise ---------------------------------

    @property
    def full_summary(self):
        raise _EagerWalkBlocked("pf.full_summary accessed by a footer-only subcommand")

    def all_pages(self):
        raise _EagerWalkBlocked("pf.all_pages() called by a footer-only subcommand")

    def all_segments(self):
        raise _EagerWalkBlocked("pf.all_segments() called by a footer-only subcommand")

    @property
    def column_offset_map(self):
        raise _EagerWalkBlocked(
            "pf.column_offset_map accessed by a footer-only subcommand"
        )

    # --- lifecycle ------------------------------------------------------

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def _make_footer(row_groups):
    """Compose a footer dict from a list of (num_rows, total_byte_size, columns)."""
    return {
        "version": 1,
        "schema": [],
        "num_rows": sum(rg[0] for rg in row_groups),
        "row_groups": [
            {
                "num_rows": rg[0],
                "total_byte_size": rg[1],
                "columns": rg[2],
            }
            for rg in row_groups
        ],
    }


def _make_column(
    path,
    *,
    type_="INT32",
    num_values=100,
    compressed=120,
    uncompressed=400,
    codec="SNAPPY",
    encodings=("PLAIN",),
    dictionary_offset=None,
    offset_index_offset=None,
    column_index_offset=None,
    bloom_filter_offset=None,
    statistics=None,
):
    """Build a single column-chunk dict shaped like pf.footer's columns."""
    md = {
        "type": type_,
        "encodings": list(encodings),
        "path_in_schema": list(path),
        "codec": codec,
        "num_values": num_values,
        "total_uncompressed_size": uncompressed,
        "total_compressed_size": compressed,
        "data_page_offset": 100,
    }
    if dictionary_offset is not None:
        md["dictionary_page_offset"] = dictionary_offset
    if bloom_filter_offset is not None:
        md["bloom_filter_offset"] = bloom_filter_offset
    if statistics is not None:
        md["statistics"] = statistics
    cc = {"file_offset": 0, "meta_data": md}
    if offset_index_offset is not None:
        cc["offset_index_offset"] = offset_index_offset
        cc["offset_index_length"] = 20
    if column_index_offset is not None:
        cc["column_index_offset"] = column_index_offset
        cc["column_index_length"] = 30
    return cc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(argv: Sequence[str], monkeypatch, fake: _FakeParquetFile):
    """Run cli.main with a fake ParquetFile factory; return (stdout, stderr, exit_code)."""
    monkeypatch.setattr(_subcommands, "ParquetFile", lambda path: fake)

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            cli.main(list(argv))
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    return stdout.getvalue(), stderr.getvalue(), exit_code


def _parse_json(text):
    return json.loads(text)


# ---------------------------------------------------------------------------
# Dispatch / argv sniffing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["sample.parquet"], False),
        (["file", "summary", "sample.parquet"], True),
        (["rowgroup", "list", "sample.parquet"], True),
        (["column", "show", "sample.parquet", "--column", "a"], True),
        (["--log-level", "DEBUG", "file", "summary", "x.parquet"], True),
        # legacy --log-level positional immediately after must not be confused
        (["--log-level", "DEBUG", "x.parquet"], False),
        (["-o", "out.json", "file", "summary", "x.parquet"], True),
        # --output-mode forces legacy
        (["--output-mode", "segments", "x.parquet"], False),
        (["--output-mode=html", "x.parquet"], False),
        (["--html-sections", "summary", "schema", "x.parquet"], False),
        # explicit end-of-options
        (["--", "file", "summary", "x.parquet"], True),
        (["--", "x.parquet"], False),
        # --key=value style passes through
        (["--log-level=DEBUG", "file", "summary", "x.parquet"], True),
    ],
)
def test_is_subcommand_invocation(argv, expected):
    assert _subcommands.is_subcommand_invocation(argv) is expected


# ---------------------------------------------------------------------------
# --schema-version short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,expected_uri",
    [
        (["file", "summary", "--schema-version"], "parquet-analyzer/v1/file-summary"),
        (["file", "kv", "--schema-version"], "parquet-analyzer/v1/file-kv"),
        (["file", "schema", "--schema-version"], "parquet-analyzer/v1/file-schema"),
        (["file", "validate", "--schema-version"], "parquet-analyzer/v1/file-validate"),
        (["rowgroup", "list", "--schema-version"], "parquet-analyzer/v1/rowgroup-list"),
        (["rowgroup", "show", "--schema-version"], "parquet-analyzer/v1/rowgroup-show"),
        (["column", "list", "--schema-version"], "parquet-analyzer/v1/column-list"),
        (["column", "show", "--schema-version"], "parquet-analyzer/v1/column-show"),
    ],
)
def test_schema_version_short_circuit(argv, expected_uri, monkeypatch, capsys):
    # ParquetFile must not be touched — assert via a sentinel that would raise.
    def _trap(_path):
        raise AssertionError(
            "--schema-version must not open the file (path may be absent)"
        )

    monkeypatch.setattr(_subcommands, "ParquetFile", _trap)
    try:
        cli.main(argv)
    except SystemExit as exc:
        assert (exc.code or 0) == 0
    out = capsys.readouterr().out
    payload = _parse_json(out)
    assert payload == {"$schema": expected_uri}


# ---------------------------------------------------------------------------
# file summary
# ---------------------------------------------------------------------------


def test_file_summary_emits_footer_summary(monkeypatch):
    fake = _FakeParquetFile(
        num_rows=42,
        num_columns=3,
        footer_summary={
            "num_rows": 42,
            "num_row_groups": 1,
            "num_columns": 3,
            "uncompressed_page_size": 1000,
            "compressed_page_size": 600,
            "column_index_size": 0,
            "offset_index_size": 0,
            "bloom_filter_size": 0,
            "footer_size": 200,
            "file_size": 1206,
        },
    )
    out, err, code = _run(["file", "summary", "f.parquet"], monkeypatch, fake)
    assert code == 0
    assert err == ""
    payload = _parse_json(out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-summary"
    assert payload["num_rows"] == 42
    assert payload["compressed_page_size"] == 600
    assert fake.closed


# ---------------------------------------------------------------------------
# file kv
# ---------------------------------------------------------------------------


def test_file_kv_returns_all(monkeypatch):
    fake = _FakeParquetFile(kv_metadata=[("a", "1"), ("b", "2"), ("a", "3")])
    out, _err, code = _run(["file", "kv", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-kv"
    assert payload["items"] == [
        {"key": "a", "value": "1"},
        {"key": "b", "value": "2"},
        {"key": "a", "value": "3"},
    ]
    assert payload["total"] == 3
    assert payload["truncated"] is False


def test_file_kv_filters_by_key_and_preserves_duplicates(monkeypatch):
    fake = _FakeParquetFile(kv_metadata=[("a", "1"), ("b", "2"), ("a", "3")])
    out, _err, code = _run(["file", "kv", "f.parquet", "--key", "a"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["items"] == [
        {"key": "a", "value": "1"},
        {"key": "a", "value": "3"},
    ]
    assert payload["filter_key"] == "a"


def test_file_kv_limit_marks_truncated(monkeypatch):
    fake = _FakeParquetFile(kv_metadata=[("a", "1"), ("b", "2"), ("c", "3")])
    out, _err, code = _run(
        ["file", "kv", "f.parquet", "--limit", "2"], monkeypatch, fake
    )
    assert code == 0
    payload = _parse_json(out)
    assert len(payload["items"]) == 2
    assert payload["truncated"] is True
    assert payload["total"] == 3
    assert payload["returned"] == 2


# ---------------------------------------------------------------------------
# file schema
# ---------------------------------------------------------------------------


def test_file_schema_returns_elements(monkeypatch):
    fake = _FakeParquetFile(
        footer={
            "row_groups": [],
            "schema": [
                {"name": "root", "num_children": 1, "repetition_type": "REQUIRED"},
                {"name": "a", "type": "INT32", "repetition_type": "OPTIONAL"},
            ],
        }
    )
    out, _err, code = _run(["file", "schema", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-schema"
    assert len(payload["elements"]) == 2
    assert payload["elements"][1]["name"] == "a"


# ---------------------------------------------------------------------------
# file validate
# ---------------------------------------------------------------------------


def test_file_validate_passes_on_consistent_footer(monkeypatch):
    cols = [_make_column(("a",), num_values=2, compressed=10)]
    footer = _make_footer([(2, 10, cols)])
    fake = _FakeParquetFile(footer=footer, num_rows=2, num_columns=1)
    out, _err, code = _run(["file", "validate", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-validate"
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_file_validate_flags_row_count_mismatch(monkeypatch):
    cols = [_make_column(("a",), num_values=2, compressed=10)]
    # claim num_rows=5 in pf.num_rows but row group only has 2
    fake = _FakeParquetFile(
        footer=_make_footer([(2, 10, cols)]),
        num_rows=5,
        num_columns=1,
    )
    out, _err, code = _run(["file", "validate", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["valid"] is False
    codes = [e["code"] for e in payload["errors"]]
    assert "row_count_mismatch" in codes


def test_file_validate_flags_inconsistent_column_count(monkeypatch):
    cols1 = [_make_column(("a",)), _make_column(("b",))]
    cols2 = [_make_column(("a",))]  # only 1 column instead of 2
    fake = _FakeParquetFile(
        footer=_make_footer([(1, 10, cols1), (1, 10, cols2)]),
        num_rows=2,
        num_columns=2,
    )
    out, _err, code = _run(["file", "validate", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["valid"] is False
    codes = [e["code"] for e in payload["errors"]]
    assert "inconsistent_column_count" in codes


def test_file_validate_flags_empty_column_chunk(monkeypatch):
    cols = [_make_column(("a",), compressed=0)]
    fake = _FakeParquetFile(
        footer=_make_footer([(2, 10, cols)]),
        num_rows=2,
        num_columns=1,
    )
    out, _err, code = _run(["file", "validate", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["valid"] is False
    codes = [e["code"] for e in payload["errors"]]
    assert "empty_column_chunk" in codes


def test_file_validate_catches_parquet_file_construction_error(monkeypatch):
    def _bad(_path):
        raise ValueError("Not a valid Parquet file - missing PAR1 header")

    monkeypatch.setattr(_subcommands, "ParquetFile", _bad)
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            cli.main(["file", "validate", "bad.parquet"])
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    assert exit_code == 0  # validation finding, not CLI error
    payload = _parse_json(stdout.getvalue())
    assert payload["valid"] is False
    assert payload["errors"][0]["code"] == "footer_parse_failed"


# ---------------------------------------------------------------------------
# rowgroup list / show
# ---------------------------------------------------------------------------


def test_rowgroup_list(monkeypatch):
    cols = [_make_column(("a",)), _make_column(("b",))]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols), (20, 200, cols)]),
        num_rows=30,
        num_columns=2,
    )
    out, _err, code = _run(["rowgroup", "list", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["$schema"] == "parquet-analyzer/v1/rowgroup-list"
    assert payload["total"] == 2
    assert payload["items"][0]["num_rows"] == 10
    assert payload["items"][0]["num_columns"] == 2
    assert payload["items"][1]["total_byte_size"] == 200


def test_rowgroup_show_uses_wrapper_for_num_pages(monkeypatch):
    cc_a = _make_column(("a",), offset_index_offset=500)
    cc_b = _make_column(("b",))
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, [cc_a, cc_b])]),
        num_rows=10,
        num_columns=2,
        row_group_wrappers=[
            _FakeRowGroup(
                [
                    _FakeColumnChunk(("a",), num_pages_value=4, has_offset_index=True),
                    _FakeColumnChunk(("b",), has_offset_index=False),
                ]
            )
        ],
    )
    out, _err, code = _run(
        ["rowgroup", "show", "f.parquet", "--row-group", "0"], monkeypatch, fake
    )
    assert code == 0
    payload = _parse_json(out)
    assert payload["row_group"] == 0
    assert payload["num_rows"] == 10
    assert payload["columns"][0]["num_pages"] == 4
    assert payload["columns"][0]["num_pages_known"] is True
    assert payload["columns"][1]["num_pages"] is None
    assert payload["columns"][1]["num_pages_known"] is False


def test_rowgroup_show_requires_row_group_flag(monkeypatch):
    fake = _FakeParquetFile(footer=_make_footer([(1, 10, [_make_column(("a",))])]))
    out, err, code = _run(["rowgroup", "show", "f.parquet"], monkeypatch, fake)
    assert code == 1
    payload = _parse_json(err)
    assert payload["error"] == "missing_argument"


def test_rowgroup_show_out_of_range(monkeypatch):
    cols = [_make_column(("a",))]
    fake = _FakeParquetFile(footer=_make_footer([(1, 10, cols)]))
    out, err, code = _run(
        ["rowgroup", "show", "f.parquet", "--row-group", "5"], monkeypatch, fake
    )
    assert code == 1
    payload = _parse_json(err)
    assert payload["error"] == "row_group_out_of_range"


# ---------------------------------------------------------------------------
# column list / show
# ---------------------------------------------------------------------------


def test_column_list_across_row_groups(monkeypatch):
    cols = [_make_column(("a",)), _make_column(("b",))]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols), (10, 100, cols)]),
        num_rows=20,
        num_columns=2,
        row_group_wrappers=[
            _FakeRowGroup([_FakeColumnChunk(("a",)), _FakeColumnChunk(("b",))]),
            _FakeRowGroup([_FakeColumnChunk(("a",)), _FakeColumnChunk(("b",))]),
        ],
    )
    out, _err, code = _run(["column", "list", "f.parquet"], monkeypatch, fake)
    assert code == 0
    payload = _parse_json(out)
    assert payload["total"] == 4  # 2 rgs × 2 columns
    assert {(i["row_group"], i["column"]) for i in payload["items"]} == {
        (0, "a"),
        (0, "b"),
        (1, "a"),
        (1, "b"),
    }


def test_column_list_filtered_by_row_group(monkeypatch):
    cols = [_make_column(("a",)), _make_column(("b",))]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols), (10, 100, cols)]),
        num_rows=20,
        num_columns=2,
        row_group_wrappers=[
            _FakeRowGroup([_FakeColumnChunk(("a",)), _FakeColumnChunk(("b",))]),
            _FakeRowGroup([_FakeColumnChunk(("a",)), _FakeColumnChunk(("b",))]),
        ],
    )
    out, _err, code = _run(
        ["column", "list", "f.parquet", "--row-group", "1"], monkeypatch, fake
    )
    assert code == 0
    payload = _parse_json(out)
    assert payload["row_group"] == 1
    assert all(i["row_group"] == 1 for i in payload["items"])


def test_column_show_aggregates_across_row_groups(monkeypatch):
    cols = [_make_column(("a",), num_values=10)]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols), (10, 100, cols), (10, 100, cols)]),
        num_rows=30,
        num_columns=1,
        row_group_wrappers=[
            _FakeRowGroup([_FakeColumnChunk(("a",))]),
            _FakeRowGroup([_FakeColumnChunk(("a",))]),
            _FakeRowGroup([_FakeColumnChunk(("a",))]),
        ],
    )
    out, _err, code = _run(
        ["column", "show", "f.parquet", "--column", "a"], monkeypatch, fake
    )
    assert code == 0
    payload = _parse_json(out)
    assert payload["column"] == "a"
    assert payload["path"] == ["a"]
    assert len(payload["row_groups"]) == 3
    assert [rg["row_group"] for rg in payload["row_groups"]] == [0, 1, 2]


def test_column_show_requires_column(monkeypatch):
    fake = _FakeParquetFile(footer=_make_footer([(1, 10, [_make_column(("a",))])]))
    out, err, code = _run(["column", "show", "f.parquet"], monkeypatch, fake)
    assert code == 1
    payload = _parse_json(err)
    assert payload["error"] == "missing_argument"


def test_column_show_not_found_lists_available(monkeypatch):
    cols = [_make_column(("a",)), _make_column(("b",))]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols)]),
        num_rows=10,
        num_columns=2,
        row_group_wrappers=[_FakeRowGroup([])],
    )
    out, err, code = _run(
        ["column", "show", "f.parquet", "--column", "z"], monkeypatch, fake
    )
    assert code == 1
    payload = _parse_json(err)
    assert payload["error"] == "column_not_found"
    assert "Available: a, b" in payload["message"]


def test_column_show_filter_row_group_includes_marker(monkeypatch):
    cols = [_make_column(("a",))]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols), (10, 100, cols)]),
        num_rows=20,
        num_columns=1,
        row_group_wrappers=[
            _FakeRowGroup([_FakeColumnChunk(("a",))]),
            _FakeRowGroup([_FakeColumnChunk(("a",))]),
        ],
    )
    out, _err, code = _run(
        ["column", "show", "f.parquet", "--column", "a", "--row-group", "1"],
        monkeypatch,
        fake,
    )
    assert code == 0
    payload = _parse_json(out)
    assert payload["filter_row_group"] == 1
    assert len(payload["row_groups"]) == 1
    assert payload["row_groups"][0]["row_group"] == 1


def test_column_show_includes_num_pages_only_with_offset_index(monkeypatch):
    cols = [
        _make_column(("a",), offset_index_offset=500),
    ]
    fake = _FakeParquetFile(
        footer=_make_footer([(10, 100, cols)]),
        num_rows=10,
        num_columns=1,
        row_group_wrappers=[
            _FakeRowGroup(
                [_FakeColumnChunk(("a",), num_pages_value=7, has_offset_index=True)]
            )
        ],
    )
    out, _err, code = _run(
        ["column", "show", "f.parquet", "--column", "a"], monkeypatch, fake
    )
    assert code == 0
    payload = _parse_json(out)
    assert payload["row_groups"][0]["has_offset_index"] is True
    assert payload["row_groups"][0]["num_pages"] == 7
    assert payload["row_groups"][0]["num_pages_known"] is True


# ---------------------------------------------------------------------------
# Output redirect (-o)
# ---------------------------------------------------------------------------


def test_output_flag_writes_to_file(monkeypatch, tmp_path):
    fake = _FakeParquetFile(num_rows=1, num_columns=0)
    out_path = tmp_path / "out.json"
    out, _err, code = _run(
        ["file", "summary", "f.parquet", "-o", str(out_path)],
        monkeypatch,
        fake,
    )
    assert code == 0
    assert out == ""  # stdout silent when -o used
    payload = _parse_json(out_path.read_text())
    assert payload["$schema"] == "parquet-analyzer/v1/file-summary"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_unknown_subcommand_emits_json_error(monkeypatch, capsys):
    # argparse subparser handles unknown nouns via _JsonErrorParser
    try:
        cli.main(["file", "unknown-noun"])
    except SystemExit as exc:
        assert (exc.code or 0) == 2
    err = capsys.readouterr().err
    payload = _parse_json(err.splitlines()[0])
    assert payload["error"] == "invalid_arguments"


def test_file_not_found_error_contract(monkeypatch):
    def _missing(_path):
        raise FileNotFoundError("no such file: missing.parquet")

    monkeypatch.setattr(_subcommands, "ParquetFile", _missing)
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            cli.main(["file", "summary", "missing.parquet"])
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    assert exit_code == 1
    payload = _parse_json(stderr.getvalue())
    assert payload["error"] == "file_not_found"


def test_invalid_parquet_error_contract(monkeypatch):
    def _bad(_path):
        raise ValueError("Not a valid Parquet file - missing PAR1 header")

    monkeypatch.setattr(_subcommands, "ParquetFile", _bad)
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            cli.main(["file", "summary", "bad.parquet"])
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    assert exit_code == 1
    payload = _parse_json(stderr.getvalue())
    assert payload["error"] == "invalid_parquet_file"
    # fix command must reference the validate subcommand
    assert "file validate" in payload["fix"]
