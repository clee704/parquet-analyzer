"""Tests for the page body-decode engine (issue #21).

Strategy mirrors :mod:`tests.test_decoders`: round-trip through
pyarrow-generated parquet files so the tests exercise the same lazy
``ParquetFile`` -> ``ColumnChunk`` -> ``Page`` path real callers use, and
assert decoded values against the original data. The matrix covers V1/V2
pages x {PLAIN, dictionary} encodings x {required, optional, repeated}
columns, plus the unsupported-input error contract.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from parquet_analyzer import (
    ColumnChunk,
    MissingDictionaryError,
    ParquetFile,
    UnsupportedCodecError,
    UnsupportedEncodingError,
    UnsupportedPageTypeError,
)
from parquet_analyzer._core import column_decode_info
from parquet_analyzer.parquet_file import _dictionary_lookup, Page


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A flat table with one null in the optional int column; the string and float
# columns are fully populated. Used for the encoding/version matrix.
_FLAT = {
    "id": pa.array([1, 2, 3, None, 5, 6, 7, 8, 9, 10], type=pa.int32()),
    "name": pa.array(["a", "b", "a", "c", "b", "a", "a", "b", "c", "a"]),
    "val": pa.array([1.0 * i + 0.5 for i in range(10)], type=pa.float64()),
}

# Physical-type expectation of each flat column's NON-NULL values, in order
# (BYTE_ARRAY decodes to bytes, not str).
_FLAT_PHYSICAL = {
    "id": [1, 2, 3, 5, 6, 7, 8, 9, 10],
    "name": [b"a", b"b", b"a", b"c", b"b", b"a", b"a", b"b", b"c", b"a"],
    "val": [1.0 * i + 0.5 for i in range(10)],
}


def _write(path: Path, table: pa.Table, **kwargs) -> Path:
    pq.write_table(table, path, **kwargs)
    return path


def _flat(path: Path, **kwargs) -> Path:
    return _write(path, pa.table(_FLAT), **kwargs)


def _col(pf: ParquetFile, name: str) -> ColumnChunk:
    for cc in pf.row_groups[0].columns:
        if ".".join(cc.path) == name:
            return cc
    raise AssertionError(f"no column {name!r} in {pf.path}")


def _data_pages(cc: ColumnChunk) -> list[Page]:
    return [p for p in cc.pages() if p._kind != "dictionary_page"]


def _decoded_values(cc: ColumnChunk) -> list:
    out: list = []
    for p in _data_pages(cc):
        out.extend(p.physical_values())
    return out


# ---------------------------------------------------------------------------
# column_decode_info (the schema-walk level computation)
# ---------------------------------------------------------------------------


def test_column_decode_info_levels_and_type_length(tmp_path):
    schema = pa.schema(
        [
            pa.field("req_id", pa.int64(), nullable=False),
            pa.field("opt_id", pa.int32(), nullable=True),
            pa.field("tags", pa.list_(pa.int32())),
            pa.field("dec", pa.decimal128(5, 2)),
            pa.field("flba", pa.binary(4)),
        ]
    )
    table = pa.table(
        {
            "req_id": [1, 2],
            "opt_id": [None, 4],
            "tags": [[1, 2], [3]],
            "dec": [Decimal("1.23"), Decimal("4.56")],
            "flba": [b"abcd", b"wxyz"],
        },
        schema=schema,
    )
    path = _write(tmp_path / "schema.parquet", table)
    pf = ParquetFile(str(path))
    info = column_decode_info(pf.footer["schema"])

    assert info[("req_id",)] == {"max_def": 0, "max_rep": 0, "type_length": None}
    assert info[("opt_id",)] == {"max_def": 1, "max_rep": 0, "type_length": None}
    # tags.list.element: OPTIONAL (tags) + REPEATED (list) + OPTIONAL (element)
    assert info[("tags", "list", "element")] == {
        "max_def": 3,
        "max_rep": 1,
        "type_length": None,
    }
    assert info[("dec",)]["type_length"] == 3
    assert info[("flba",)]["type_length"] == 4


def test_column_decode_info_empty_schema():
    assert column_decode_info([]) == {}


def test_column_chunk_level_properties(tmp_path):
    path = _flat(tmp_path / "f.parquet")
    pf = ParquetFile(str(path))
    id_col = _col(pf, "id")
    assert id_col.max_definition_level == 1
    assert id_col.max_repetition_level == 0
    assert id_col.type_length is None


def test_decode_info_missing_schema_leaf_raises(tmp_path):
    path = _flat(tmp_path / "f.parquet")
    pf = ParquetFile(str(path))
    cc = _col(pf, "id")
    pf._decode_info_map_cache = {}  # simulate a schema/metadata inconsistency
    cc._decode_info_cache = None
    with pytest.raises(ValueError, match="no matching schema leaf"):
        _ = cc.max_definition_level


# ---------------------------------------------------------------------------
# Round-trip matrix: V1/V2 x {PLAIN, dict}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1.0", "2.0"])
@pytest.mark.parametrize("use_dictionary", [True, False])
@pytest.mark.parametrize("name", ["id", "name", "val"])
def test_roundtrip_matrix(tmp_path, version, use_dictionary, name):
    path = _flat(
        tmp_path / f"flat-{version}-{use_dictionary}.parquet",
        data_page_version=version,
        use_dictionary=use_dictionary,
        compression="snappy",
    )
    pf = ParquetFile(str(path))
    cc = _col(pf, name)

    page = _data_pages(cc)[0]
    expected_encoding = "RLE_DICTIONARY" if use_dictionary else "PLAIN"
    assert page.encoding == expected_encoding
    assert _decoded_values(cc) == _FLAT_PHYSICAL[name]


def test_optional_column_nulls_and_levels(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    decoded = page.decode()

    assert decoded.num_values == 10
    assert decoded.num_nulls == 1
    assert len(decoded.values) == 9
    # Index 3 is the lone null (definition level 0 < max_definition_level 1).
    assert decoded.definition_levels == [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert decoded.repetition_levels == [0] * 10
    assert page.definition_levels() == decoded.definition_levels


def test_required_column_has_no_levels_or_nulls(tmp_path):
    schema = pa.schema([pa.field("x", pa.int32(), nullable=False)])
    table = pa.table({"x": [10, 20, 30]}, schema=schema)
    path = _write(tmp_path / "req.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    decoded = _data_pages(_col(pf, "x"))[0].decode()

    assert _col(pf, "x").max_definition_level == 0
    assert decoded.num_nulls == 0
    assert decoded.definition_levels == [0, 0, 0]
    assert decoded.values == [10, 20, 30]


def test_repeated_column_repetition_levels(tmp_path):
    table = pa.table({"tags": [[1, 2], [3], []]})
    path = _write(tmp_path / "list.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    cc = _col(pf, "tags.list.element")
    assert cc.max_repetition_level == 1
    decoded = _data_pages(cc)[0].decode()
    # The non-null leaf values are the flattened list elements; the trailing
    # empty list contributes a null leaf (no value). A new list starts at
    # repetition level 0, a continuation at level 1.
    assert decoded.values == [1, 2, 3]
    assert decoded.repetition_levels == [0, 1, 0, 0]
    assert decoded.num_nulls == 1


def test_fixed_len_byte_array_decode(tmp_path):
    schema = pa.schema([pa.field("flba", pa.binary(4))])
    table = pa.table({"flba": [b"abcd", b"wxyz"]}, schema=schema)
    path = _write(tmp_path / "flba.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    cc = _col(pf, "flba")
    assert cc.type == "FIXED_LEN_BYTE_ARRAY"
    assert cc.type_length == 4
    assert _decoded_values(cc) == [b"abcd", b"wxyz"]


# ---------------------------------------------------------------------------
# Dictionary access
# ---------------------------------------------------------------------------


def test_dictionary_returns_entries_and_caches(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    dictionary = cc.dictionary()
    assert dictionary == [b"a", b"b", b"c"]
    assert cc.dictionary() is dictionary  # cached object identity


def test_dictionary_none_for_plain_column(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    assert _col(pf, "id").dictionary() is None


def test_dictionary_found_by_page_scan_when_offset_unset(tmp_path):
    """Older writers leave ``dictionary_page_offset`` unset and point
    ``data_page_offset`` at the dictionary page. Simulate that layout and
    confirm the page-scan fallback still resolves the dictionary."""
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    original_dict = cc._md.dictionary_page_offset
    cc._md.data_page_offset = original_dict
    cc._md.dictionary_page_offset = None
    assert cc.dictionary() == [b"a", b"b", b"c"]


def test_dictionary_offset_pointing_at_non_dictionary_raises(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    # Point the dictionary offset at the data page (a malformed footer).
    cc._md.dictionary_page_offset = cc._md.data_page_offset
    with pytest.raises(ValueError, match="does not point at a dictionary page"):
        cc.dictionary()


def test_level_convenience_methods(tmp_path):
    table = pa.table({"tags": [[1, 2], [3]]})
    path = _write(tmp_path / "list.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "tags.list.element"))[0]
    assert page.repetition_levels() == [0, 1, 0]
    assert page.definition_levels() == page.decode().definition_levels
    assert page.physical_values() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Body bytes + caching
# ---------------------------------------------------------------------------


def test_raw_body_length_and_offset(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    assert page.body_offset == page.segment["offset"] + page.segment["length"]
    assert len(page.raw_body()) == page.compressed_size


def test_decode_is_cached(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    assert page.decode() is page.decode()


@pytest.mark.parametrize("use_dictionary", [False, True])
def test_all_null_page_has_no_values(tmp_path, use_dictionary):
    table = pa.table({"x": pa.array([None, None, None], type=pa.int32())})
    path = _write(
        tmp_path / f"allnull-{use_dictionary}.parquet",
        table,
        use_dictionary=use_dictionary,
    )
    pf = ParquetFile(str(path))
    cc = _col(pf, "x")
    decoded = _data_pages(cc)[0].decode()
    assert decoded.num_nulls == 3
    assert decoded.values == []
    if use_dictionary:
        # A dictionary-encoded all-null page still resolves its (empty)
        # dictionary; the index stream is simply empty.
        assert decoded.encoding == "RLE_DICTIONARY"
        assert decoded.dictionary_indices == []
        assert cc.dictionary() == []
    else:
        assert decoded.dictionary_indices is None


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_unsupported_value_encoding(tmp_path):
    table = pa.table({"x": pa.array(list(range(100)), type=pa.int32())})
    path = _write(
        tmp_path / "delta.parquet",
        table,
        column_encoding={"x": "DELTA_BINARY_PACKED"},
        use_dictionary=False,
    )
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "x"))[0]
    with pytest.raises(UnsupportedEncodingError) as exc:
        page.decode()
    assert exc.value.code == "encoding_not_supported"
    assert exc.value.encoding == "DELTA_BINARY_PACKED"
    assert exc.value.context == "values"


def test_unsupported_codec(tmp_path):
    path = _flat(
        tmp_path / "brotli.parquet", compression="brotli", use_dictionary=False
    )
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    with pytest.raises(UnsupportedCodecError) as exc:
        page.decode()
    assert exc.value.code == "codec_not_supported"
    assert exc.value.codec == "BROTLI"


def test_decode_on_dictionary_page_raises(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    dict_page = next(p for p in cc.pages() if p._kind == "dictionary_page")
    with pytest.raises(UnsupportedPageTypeError) as exc:
        dict_page.decode()
    assert exc.value.code == "page_type_not_supported"


def test_missing_dictionary(tmp_path, monkeypatch):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    monkeypatch.setattr(ColumnChunk, "dictionary", lambda self: None)
    page = _data_pages(cc)[0]
    with pytest.raises(MissingDictionaryError) as exc:
        page.decode()
    assert exc.value.code == "missing_dictionary"
    assert exc.value.path == ("name",)


def test_unsupported_dictionary_page_encoding(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    dict_page = next(p for p in cc.pages() if p._kind == "dictionary_page")
    # Force a non-PLAIN dictionary encoding (DELTA_BYTE_ARRAY enum value).
    from parquet.ttypes import Encoding

    dict_page._t.dictionary_page_header.encoding = Encoding.DELTA_BYTE_ARRAY
    with pytest.raises(UnsupportedEncodingError) as exc:
        dict_page._decode_dictionary_entries()
    assert exc.value.context == "dictionary"


def test_v1_non_rle_level_encoding_raises():
    from parquet.ttypes import Encoding

    with pytest.raises(UnsupportedEncodingError) as exc:
        Page._require_rle_level_encoding(Encoding.BIT_PACKED, 1, "def")
    assert exc.value.context == "def level"
    # max_level 0 means there is no level block, so the encoding is ignored.
    Page._require_rle_level_encoding(Encoding.BIT_PACKED, 0, "def")


def test_dictionary_index_out_of_range():
    assert _dictionary_lookup([b"a", b"b"], 1, ("c",)) == b"b"
    with pytest.raises(ValueError, match="out of range"):
        _dictionary_lookup([b"a", b"b"], 5, ("c",))


def test_v2_levels_exceeding_body_raises(tmp_path):
    path = _flat(tmp_path / "f.parquet", data_page_version="2.0", use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    page._t.data_page_header_v2.definition_levels_byte_length = 10_000
    with pytest.raises(ValueError, match="exceed the page"):
        page.decode()


def test_v2_compressed_values_section(tmp_path):
    """A V2 page whose values section is large enough that pyarrow actually
    compresses it (is_compressed True) exercises the values-only
    decompression path."""
    table = pa.table({"s": pa.array(["x" * 64] * 500)})
    path = _write(
        tmp_path / "v2comp.parquet",
        table,
        data_page_version="2.0",
        compression="snappy",
        use_dictionary=False,
    )
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "s"))[0]
    assert page._t.data_page_header_v2.is_compressed is True
    assert page.physical_values() == [b"x" * 64] * 500
