"""End-to-end tests for the ``page`` verb subcommands (list/header/extract/decode).

These exercise the real lazy-core body decode through :func:`cli.main`, against
pyarrow-generated fixtures spanning V1/V2 pages, dictionary vs PLAIN encoding,
nullable vs repeated columns, single vs multiple row groups, and the
OffsetIndex fast-path. The ``page`` surface is the v1 escape hatch (#21); the
contract under test is the curated ``page-*`` JSON envelopes, the byte escape
hatch (``--as raw``), and the structured error mapping for decode failures.
"""

from __future__ import annotations

import base64
import json
import random

import pytest

from parquet_analyzer import cli

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Invocation helpers
# ---------------------------------------------------------------------------


def _run(argv, capsys):
    """Invoke the CLI on a success path and return the parsed stdout JSON."""
    cli.main([str(a) for a in argv])
    return json.loads(capsys.readouterr().out)


def _run_err(argv, capsys):
    """Invoke the CLI expecting a structured error; return the stderr JSON."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(a) for a in argv])
    assert exc_info.value.code == 1
    return json.loads(capsys.readouterr().err)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dict_v1(tmp_path):
    """Single row group, dict-encoded string column, V1 pages, no OffsetIndex.

    Page 0 is the dictionary page; page 1 is the RLE_DICTIONARY data page.
    """
    path = tmp_path / "dict_v1.parquet"
    pq.write_table(
        pa.table({"s": ["male", "female", "female", "male", "male", "female"]}),
        path,
        use_dictionary=True,
        write_page_index=False,
    )
    return path


@pytest.fixture()
def with_index(tmp_path):
    """Dict-encoded column written with an OffsetIndex (page-list fast path)."""
    path = tmp_path / "with_index.parquet"
    pq.write_table(
        pa.table({"s": ["a", "b", "a", "c", "b", "a"]}),
        path,
        use_dictionary=True,
        write_page_index=True,
    )
    return path


@pytest.fixture()
def v2_nulls(tmp_path):
    """A V2 (DATA_PAGE_V2) PLAIN int column containing a null."""
    path = tmp_path / "v2.parquet"
    pq.write_table(
        pa.table({"x": pa.array([1, 2, None, 4, 5], type=pa.int32())}),
        path,
        data_page_version="2.0",
        use_dictionary=False,
        compression="snappy",
    )
    return path


@pytest.fixture()
def multi_rg(tmp_path):
    """Four row groups (row_group_size 25 over 100 rows)."""
    path = tmp_path / "multi_rg.parquet"
    pq.write_table(
        pa.table({"x": pa.array(list(range(100)), type=pa.int32())}),
        path,
        row_group_size=25,
        use_dictionary=False,
    )
    return path


@pytest.fixture()
def plain_stats(tmp_path):
    """PLAIN int column with page-header statistics (no page index)."""
    path = tmp_path / "stats.parquet"
    pq.write_table(
        pa.table({"n": pa.array([5, 1, 9, 3], type=pa.int32())}),
        path,
        use_dictionary=False,
        write_statistics=True,
        write_page_index=False,
    )
    return path


@pytest.fixture()
def nested(tmp_path):
    """A repeated column (list<int>) — has non-trivial repetition levels."""
    path = tmp_path / "nested.parquet"
    pq.write_table(
        pa.table({"tags": pa.array([[1, 2], [3], []], type=pa.list_(pa.int32()))}),
        path,
        use_dictionary=False,
    )
    return path


@pytest.fixture()
def v2_compressed(tmp_path):
    """A V2 page whose values section is actually compressed (is_compressed)."""
    path = tmp_path / "v2_compressed.parquet"
    pq.write_table(
        pa.table({"x": pa.array([7] * 5000, type=pa.int32())}),
        path,
        data_page_version="2.0",
        use_dictionary=False,
        compression="snappy",
    )
    return path


@pytest.fixture()
def multi_data_page(tmp_path):
    """A dict-encoded column with one dictionary page and several data pages
    (forced via a tiny ``data_page_size``)."""
    path = tmp_path / "multi_data_page.parquet"
    pq.write_table(
        pa.table({"s": ["a", "b", "c", "d"] * 2000}),
        path,
        use_dictionary=True,
        data_page_size=256,
        write_page_index=False,
    )
    return path


# ---------------------------------------------------------------------------
# page list
# ---------------------------------------------------------------------------


def test_page_list_walks_headers_without_offset_index(dict_v1, capsys):
    payload = _run(["page", "list", dict_v1, "--column", "s"], capsys)
    assert payload["$schema"].endswith("/page-list")
    assert payload["column"] == "s"
    assert payload["total"] == 2
    dict_page, data_page = payload["items"]
    assert dict_page["kind"] == "dictionary_page"
    assert dict_page["data_page_index"] is None
    assert dict_page["first_row_index"] is None  # header-walk path
    assert data_page["kind"] == "data_page"
    assert data_page["data_page_index"] == 0
    assert data_page["_path"].endswith("/pages/1")
    assert data_page["offset"] > dict_page["offset"]
    # offset + length address the page's full on-disk span (header + body), so
    # consecutive pages are contiguous.
    assert dict_page["offset"] + dict_page["length"] == data_page["offset"]


def test_page_list_uses_offset_index_when_present(with_index, capsys):
    payload = _run(["page", "list", with_index, "--column", "s"], capsys)
    data_pages = [i for i in payload["items"] if i["kind"] == "data_page"]
    assert data_pages, "expected at least one data page"
    # The OffsetIndex fast path populates first_row_index.
    assert data_pages[0]["first_row_index"] == 0


def test_page_list_all_columns_without_column_filter(multi_rg, capsys):
    payload = _run(["page", "list", multi_rg], capsys)
    assert "column" not in payload
    assert payload["total"] == len(payload["items"])
    assert all(item["column"] == "x" for item in payload["items"])
    # Four row groups, each contributing at least one page.
    assert {item["row_group"] for item in payload["items"]} == {0, 1, 2, 3}


def test_page_list_restricts_to_row_group(multi_rg, capsys):
    payload = _run(["page", "list", multi_rg, "--row-group", "2"], capsys)
    assert payload["row_group"] == 2
    assert all(item["row_group"] == 2 for item in payload["items"])


def test_page_list_row_group_out_of_range(multi_rg, capsys):
    err = _run_err(["page", "list", multi_rg, "--row-group", "9"], capsys)
    assert err["error"] == "row_group_out_of_range"


def test_page_list_unknown_column_filters_to_empty(dict_v1, capsys):
    payload = _run(["page", "list", dict_v1, "--column", "nope"], capsys)
    assert payload["items"] == []
    assert payload["total"] == 0


def test_page_list_limit_truncates(dict_v1, capsys):
    payload = _run(["page", "list", dict_v1, "--column", "s", "--limit", "1"], capsys)
    assert payload["returned"] == 1
    assert payload["total"] == 2
    assert payload["truncated"] is True


def test_page_list_data_page_index_counts_data_pages(multi_data_page, capsys):
    """``data_page_index`` increments only over data pages — the leading
    dictionary page is `null` and does not consume an index."""
    payload = _run(["page", "list", multi_data_page, "--column", "s"], capsys)
    assert payload["total"] > 3  # one dict page + several data pages
    assert payload["items"][0]["kind"] == "dictionary_page"
    assert payload["items"][0]["data_page_index"] is None
    data_pages = [i for i in payload["items"] if i["kind"] == "data_page"]
    assert [i["data_page_index"] for i in data_pages] == list(range(len(data_pages)))
    # The third page overall is the second data page.
    assert payload["items"][2]["page_index"] == 2
    assert payload["items"][2]["data_page_index"] == 1


# ---------------------------------------------------------------------------
# page header
# ---------------------------------------------------------------------------


def test_page_header_v1_data_page(dict_v1, capsys):
    payload = _run(
        ["page", "header", dict_v1, "--column", "s", "--page-index", "1"], capsys
    )
    assert payload["$schema"].endswith("/page-header")
    assert payload["kind"] == "data_page"
    assert payload["page_type"] == "DATA_PAGE"
    assert payload["encoding"] == "RLE_DICTIONARY"
    assert payload["num_values"] == 6
    assert payload["data_page_index"] == 0
    # V1-specific level-encoding fields.
    assert payload["definition_level_encoding"] == "RLE"
    assert "statistics" in payload


def test_page_header_dictionary_page_has_no_level_fields(dict_v1, capsys):
    payload = _run(
        ["page", "header", dict_v1, "--column", "s", "--page-index", "0"], capsys
    )
    assert payload["kind"] == "dictionary_page"
    assert payload["data_page_index"] is None
    assert "definition_level_encoding" not in payload
    assert "statistics" not in payload


def test_page_header_v2_fields(v2_nulls, capsys):
    payload = _run(
        ["page", "header", v2_nulls, "--column", "x", "--page-index", "0"], capsys
    )
    assert payload["page_type"] == "DATA_PAGE_V2"
    assert payload["num_nulls"] == 1
    assert payload["num_rows"] == 5
    assert "is_compressed" in payload
    assert payload["definition_levels_byte_length"] >= 0
    assert payload["repetition_levels_byte_length"] == 0


def test_page_header_negative_index_addresses_from_end(dict_v1, capsys):
    payload = _run(
        ["page", "header", dict_v1, "--column", "s", "--page-index", "-1"], capsys
    )
    assert payload["page_index"] == 1
    assert payload["kind"] == "data_page"


def test_page_header_data_page_index_on_resolved_page(multi_data_page, capsys):
    """Resolving a single data page reports its position among data pages —
    page 2 overall is data page 1 (the dictionary page leads at index 0)."""
    payload = _run(
        ["page", "header", multi_data_page, "--column", "s", "--page-index", "2"],
        capsys,
    )
    assert payload["page_index"] == 2
    assert payload["kind"] == "data_page"
    assert payload["data_page_index"] == 1


def test_page_header_data_page_index_with_offset_index(with_index, capsys):
    """``data_page_index`` resolves correctly on the OffsetIndex fast path
    (the dictionary page leads, so the first data page is index 0)."""
    payload = _run(
        ["page", "header", with_index, "--column", "s", "--page-index", "1"],
        capsys,
    )
    assert payload["kind"] == "data_page"
    assert payload["data_page_index"] == 0


def test_page_header_page_out_of_range(dict_v1, capsys):
    err = _run_err(
        ["page", "header", dict_v1, "--column", "s", "--page-index", "9"], capsys
    )
    assert err["error"] == "page_out_of_range"
    assert "2 pages" in err["message"]


def test_page_header_unknown_column(dict_v1, capsys):
    err = _run_err(
        ["page", "header", dict_v1, "--column", "missing", "--page-index", "0"],
        capsys,
    )
    assert err["error"] == "column_not_found"
    assert "Available:" in err["message"]


def test_page_header_row_group_required_when_multiple(multi_rg, capsys):
    err = _run_err(
        ["page", "header", multi_rg, "--column", "x", "--page-index", "0"], capsys
    )
    assert err["error"] == "missing_argument"
    assert "--row-group is required" in err["message"]
    assert "4 row groups" in err["message"]


def test_page_header_row_group_out_of_range(multi_rg, capsys):
    err = _run_err(
        [
            "page",
            "header",
            multi_rg,
            "--column",
            "x",
            "--page-index",
            "0",
            "--row-group",
            "9",
        ],
        capsys,
    )
    assert err["error"] == "row_group_out_of_range"


# ---------------------------------------------------------------------------
# page extract
# ---------------------------------------------------------------------------


def test_page_extract_hex(dict_v1, capsys):
    payload = _run(
        [
            "page",
            "extract",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "0",
            "--as",
            "hex",
        ],
        capsys,
    )
    assert payload["$schema"].endswith("/page-extract")
    assert payload["encoding"] == "hex"
    assert payload["decompressed"] is False
    assert payload["byte_length"] == len(payload["data"]) // 2
    bytes.fromhex(payload["data"])  # round-trips as valid hex


def test_page_extract_base64_decompress(dict_v1, capsys):
    payload = _run(
        [
            "page",
            "extract",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "0",
            "--decompress",
            "--as",
            "base64",
        ],
        capsys,
    )
    assert payload["decompressed"] is True
    assert payload["encoding"] == "base64"
    decoded = base64.b64decode(payload["data"])
    assert len(decoded) == payload["byte_length"]


def test_page_extract_raw_to_stdout(dict_v1, capsysbinary):
    cli.main(
        [
            "page",
            "extract",
            str(dict_v1),
            "--column",
            "s",
            "--page-index",
            "0",
            "--as",
            "raw",
        ]
    )
    raw = capsysbinary.readouterr().out
    assert isinstance(raw, bytes)
    assert len(raw) > 0


def test_page_extract_raw_to_file(dict_v1, tmp_path, capsys):
    out_path = tmp_path / "body.bin"
    cli.main(
        [
            "page",
            "extract",
            str(dict_v1),
            "--column",
            "s",
            "--page-index",
            "0",
            "--as",
            "raw",
            "-o",
            str(out_path),
        ]
    )
    assert capsys.readouterr().out == ""
    assert out_path.read_bytes()  # non-empty body written verbatim


def test_page_extract_decompress_v2_preserves_levels(v2_nulls, capsys):
    """A V2 page keeps its uncompressed level streams and decompresses only the
    values section, concatenating to the page's uncompressed size. This small
    page is stored uncompressed (``is_compressed`` false), so the body is
    returned verbatim."""
    header = _run(
        ["page", "header", v2_nulls, "--column", "x", "--page-index", "0"], capsys
    )
    payload = _run(
        [
            "page",
            "extract",
            v2_nulls,
            "--column",
            "x",
            "--page-index",
            "0",
            "--decompress",
            "--as",
            "hex",
        ],
        capsys,
    )
    assert payload["decompressed"] is True
    assert payload["byte_length"] == header["uncompressed_size"]


def test_page_extract_decompress_v2_compressed_values(v2_compressed, capsys):
    """A V2 page with a genuinely compressed values section is expanded back to
    its uncompressed size while the (already uncompressed) levels pass through."""
    header = _run(
        ["page", "header", v2_compressed, "--column", "x", "--page-index", "0"],
        capsys,
    )
    assert header["is_compressed"] is True
    assert header["compressed_size"] < header["uncompressed_size"]
    payload = _run(
        [
            "page",
            "extract",
            v2_compressed,
            "--column",
            "x",
            "--page-index",
            "0",
            "--decompress",
            "--as",
            "hex",
        ],
        capsys,
    )
    assert payload["byte_length"] == header["uncompressed_size"]


def test_page_extract_decompress_unsupported_codec(tmp_path, capsys):
    path = tmp_path / "brotli.parquet"
    pq.write_table(
        pa.table({"s": ["a", "b", "a", "c"] * 50}),
        path,
        compression="brotli",
        use_dictionary=True,
    )
    err = _run_err(
        ["page", "extract", path, "--column", "s", "--page-index", "0", "--decompress"],
        capsys,
    )
    assert err["error"] == "codec_not_supported"
    assert err["codec"] == "BROTLI"


# ---------------------------------------------------------------------------
# page decode
# ---------------------------------------------------------------------------


def test_page_decode_values_dictionary(dict_v1, capsys):
    payload = _run(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "values",
            "--limit",
            "3",
        ],
        capsys,
    )
    assert payload["kind"] == "values"
    assert payload["encoding"] == "RLE_DICTIONARY"
    assert payload["total"] == 6  # non-null count
    assert payload["num_nulls"] == 0
    assert payload["returned"] == 3
    assert payload["truncated"] is True
    # Physical values resolve through the dictionary to the binary payloads.
    assert payload["values"][0]["type"] == "binary"


def test_page_decode_values_plain_ints(plain_stats, capsys):
    payload = _run(
        [
            "page",
            "decode",
            plain_stats,
            "--column",
            "n",
            "--page-index",
            "0",
            "--kind",
            "values",
        ],
        capsys,
    )
    assert payload["values"] == [5, 1, 9, 3]
    assert payload["total"] == 4
    assert payload["truncated"] is False


def test_page_decode_rle_runs_dictionary(tmp_path, capsys):
    """Long consecutive runs of one dictionary value encode as RLE runs."""
    path = tmp_path / "rle_runs.parquet"
    pq.write_table(pa.table({"s": ["x"] * 50 + ["y"] * 50}), path, use_dictionary=True)
    payload = _run(
        [
            "page",
            "decode",
            path,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "rle-runs",
        ],
        capsys,
    )
    assert payload["kind"] == "rle-runs"
    assert payload["bit_width"] >= 1
    assert payload["total"] == len(payload["runs"])
    rle_run = next(r for r in payload["runs"] if r["kind"] == "rle")
    assert set(rle_run) == {"kind", "value", "length"}
    assert rle_run["length"] >= 1


def test_page_decode_rle_runs_includes_bit_packed(tmp_path, capsys):
    """Random dictionary indices produce a bit-packed run, covering that view."""
    path = tmp_path / "bitpacked.parquet"
    rng = random.Random(1)
    vals = [f"v{rng.randint(0, 7)}" for _ in range(200)]
    pq.write_table(pa.table({"s": vals}), path, use_dictionary=True)
    payload = _run(
        [
            "page",
            "decode",
            path,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "rle-runs",
        ],
        capsys,
    )
    bit_packed = [r for r in payload["runs"] if r["kind"] == "bit_packed"]
    assert bit_packed, "expected at least one bit-packed run"
    assert set(bit_packed[0]) == {"kind", "length", "values"}
    assert isinstance(bit_packed[0]["values"], list)


def test_page_decode_rle_runs_unavailable_for_plain(plain_stats, capsys):
    err = _run_err(
        [
            "page",
            "decode",
            plain_stats,
            "--column",
            "n",
            "--page-index",
            "0",
            "--kind",
            "rle-runs",
        ],
        capsys,
    )
    assert err["error"] == "kind_not_available"
    assert "dictionary-encoded" in err["message"]


def test_page_decode_levels_definition_only(dict_v1, capsys):
    payload = _run(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "levels",
        ],
        capsys,
    )
    assert payload["definition_levels"] is not None
    assert payload["definition_levels"]["total"] == 6
    # A flat (non-repeated) column has no repetition levels.
    assert payload["repetition_levels"] is None


def test_page_decode_levels_repetition_present(nested, capsys):
    payload = _run(
        [
            "page",
            "decode",
            nested,
            "--column",
            "tags.list.element",
            "--page-index",
            "0",
            "--kind",
            "levels",
        ],
        capsys,
    )
    assert payload["repetition_levels"] is not None
    assert payload["repetition_levels"]["levels"] == [0, 1, 0, 0]
    assert payload["definition_levels"]["levels"] == [3, 3, 3, 1]


def test_page_decode_levels_limit(dict_v1, capsys):
    payload = _run(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "levels",
            "--limit",
            "2",
        ],
        capsys,
    )
    defs = payload["definition_levels"]
    assert defs["returned"] == 2
    assert defs["truncated"] is True


def test_page_decode_limit_zero_returns_empty(dict_v1, capsys):
    """``--limit 0`` caps to an empty list and reports truncation, matching the
    list verbs' truncation rule."""
    payload = _run(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "values",
            "--limit",
            "0",
        ],
        capsys,
    )
    assert payload["returned"] == 0
    assert payload["values"] == []
    assert payload["truncated"] is True
    assert payload["total"] == 6


def test_page_decode_statistics_from_header(plain_stats, capsys):
    payload = _run(
        [
            "page",
            "decode",
            plain_stats,
            "--column",
            "n",
            "--page-index",
            "0",
            "--kind",
            "statistics",
        ],
        capsys,
    )
    assert payload["statistics"] == {"null_count": 0, "min_value": 1, "max_value": 9}


def test_page_decode_statistics_unsupported_on_dictionary_page(dict_v1, capsys):
    err = _run_err(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "0",
            "--kind",
            "statistics",
        ],
        capsys,
    )
    assert err["error"] == "page_type_not_supported"


def test_page_decode_values_resolution_failure_maps_to_contract(
    dict_v1, monkeypatch, capsys
):
    """A dictionary-resolution failure inside physical_values() (the kind of
    thing a corrupt file can trigger) is mapped onto the JSON error contract
    rather than escaping as an uncaught traceback."""
    from parquet_analyzer.parquet_file import MissingDictionaryError, Page

    def boom(self):
        raise MissingDictionaryError(("s",))

    monkeypatch.setattr(Page, "physical_values", boom)
    err = _run_err(
        [
            "page",
            "decode",
            dict_v1,
            "--column",
            "s",
            "--page-index",
            "1",
            "--kind",
            "values",
        ],
        capsys,
    )
    assert err["error"] == "missing_dictionary"


def test_page_decode_unsupported_encoding_reports_detail(tmp_path, capsys):
    path = tmp_path / "delta.parquet"
    pq.write_table(
        pa.table({"x": pa.array(list(range(50)), type=pa.int32())}),
        path,
        column_encoding={"x": "DELTA_BINARY_PACKED"},
        use_dictionary=False,
    )
    err = _run_err(
        [
            "page",
            "decode",
            path,
            "--column",
            "x",
            "--page-index",
            "0",
            "--kind",
            "values",
        ],
        capsys,
    )
    assert err["error"] == "encoding_not_supported"
    assert err["encoding"] == "DELTA_BINARY_PACKED"


def test_page_decode_v2_values_skip_nulls(v2_nulls, capsys):
    payload = _run(
        [
            "page",
            "decode",
            v2_nulls,
            "--column",
            "x",
            "--page-index",
            "0",
            "--kind",
            "values",
        ],
        capsys,
    )
    assert payload["values"] == [1, 2, 4, 5]
    assert payload["num_nulls"] == 1
    assert payload["total"] == 4


# ---------------------------------------------------------------------------
# Required-argument and schema-version contract
# ---------------------------------------------------------------------------


def test_page_decode_requires_kind(dict_v1, capsys):
    err = _run_err(
        ["page", "decode", dict_v1, "--column", "s", "--page-index", "1"], capsys
    )
    assert err["error"] == "missing_argument"
    assert "--kind is required" in err["message"]


def test_page_header_requires_column(dict_v1, capsys):
    err = _run_err(["page", "header", dict_v1, "--page-index", "0"], capsys)
    assert err["error"] == "missing_argument"
    assert "--column is required" in err["message"]


def test_page_header_requires_page_index(dict_v1, capsys):
    err = _run_err(["page", "header", dict_v1, "--column", "s"], capsys)
    assert err["error"] == "missing_argument"
    assert "--page-index is required" in err["message"]


@pytest.mark.parametrize("noun", ["list", "header", "extract", "decode"])
def test_page_schema_version_short_circuits(noun, capsys):
    payload = _run(["page", noun, "--schema-version"], capsys)
    assert list(payload) == ["$schema"]
    assert payload["$schema"].endswith(f"/page-{noun}")
