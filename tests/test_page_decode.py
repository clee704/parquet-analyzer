"""Tests for the page body-decode engine (issue #21).

Strategy mirrors :mod:`tests.test_decoders`: round-trip through
pyarrow-generated parquet files so the tests exercise the same lazy
``ParquetFile`` -> ``ColumnChunk`` -> ``Page`` path real callers use, and
assert against the original data. The engine is **encoding-faithful**: a
page's level streams and its values section are exposed in their own
encoding structure (``RleBitPackedStream`` of runs for levels and dictionary
indices, ``PlainValues`` for PLAIN), with the decoded logical values
available separately via ``Page.physical_values()``. The matrix covers
V1/V2 × {PLAIN, dictionary} × {required, optional, repeated} columns, the
encoding-structure assertions, and the unsupported-input error contract.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from parquet_analyzer import (
    BodyExtent,
    ColumnChunk,
    MissingDictionaryError,
    ParquetFile,
    PlainValues,
    RleBitPackedStream,
    RleRun,
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


def _run_values(stream: RleBitPackedStream) -> list[int]:
    """Flatten a stream's runs by hand — must match the stream's own
    ``values`` (the faithful run structure round-trips to the value list)."""
    out: list[int] = []
    for run in stream.runs:
        if isinstance(run, RleRun):
            out.extend([run.value] * min(run.length, len(stream.values) - len(out)))
        else:
            out.extend(run.values)
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


# ---------------------------------------------------------------------------
# Encoding-faithful structure (the encoding's own data, not just the values)
# ---------------------------------------------------------------------------


def test_plain_values_section_is_plainvalues(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    decoded = _data_pages(_col(pf, "id"))[0].decode()
    assert isinstance(decoded.values, PlainValues)
    # PLAIN stores values verbatim — the section IS the non-null values.
    assert list(decoded.values.values) == _FLAT_PHYSICAL["id"]


def test_dictionary_values_section_is_index_stream(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    decoded = _data_pages(cc)[0].decode()
    # The values section of a dictionary page is the RLE/bit-packed index
    # stream (the page's actual on-disk data), NOT the resolved values.
    section = decoded.values
    assert isinstance(section, RleBitPackedStream)
    assert section.bit_width >= 1
    assert len(section.runs) >= 1
    # Resolving the indices through the dictionary yields the values.
    dictionary = cc.dictionary()
    assert [dictionary[i] for i in section.values] == _FLAT_PHYSICAL["name"]


def test_level_stream_exposes_runs(tmp_path):
    # A fully-defined optional column packs its definition levels as a single
    # RLE run (value 1, repeated num_values times).
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    decoded = _data_pages(_col(pf, "name"))[0].decode()
    stream = decoded.definition_levels
    assert isinstance(stream, RleBitPackedStream)
    assert stream.bit_width == 1
    assert stream.runs == (RleRun(value=1, length=10),)
    # The run structure round-trips to the expanded levels.
    assert list(stream.values) == [1] * 10
    assert _run_values(stream) == list(stream.values)


def test_repetition_levels_none_for_flat_column(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    decoded = _data_pages(_col(pf, "id"))[0].decode()
    # A non-repeated column has no repetition-level block on disk.
    assert decoded.repetition_levels is None


# ---------------------------------------------------------------------------
# Levels, nulls, and value counts
# ---------------------------------------------------------------------------


def test_optional_column_nulls_and_levels(tmp_path):
    path = _flat(tmp_path / "f.parquet", use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    decoded = page.decode()

    assert decoded.num_values == 10
    assert decoded.num_nulls == 1
    assert len(page.physical_values()) == 9
    # Index 3 is the lone null (definition level 0 < max_definition_level 1).
    expected_def = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert list(decoded.definition_levels.values) == expected_def
    assert page.definition_levels() == expected_def
    assert page.repetition_levels() == [0] * 10  # convenience derives [0]*n


def test_required_column_has_no_level_block(tmp_path):
    schema = pa.schema([pa.field("x", pa.int32(), nullable=False)])
    table = pa.table({"x": [10, 20, 30]}, schema=schema)
    path = _write(tmp_path / "req.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "x"))[0]
    decoded = page.decode()

    assert _col(pf, "x").max_definition_level == 0
    assert decoded.num_nulls == 0
    # A required column has no definition-level block on disk.
    assert decoded.definition_levels is None
    assert page.definition_levels() == [0, 0, 0]  # convenience derives [0]*n
    assert page.physical_values() == [10, 20, 30]


def test_repeated_column_repetition_levels(tmp_path):
    table = pa.table({"tags": [[1, 2], [3], []]})
    path = _write(tmp_path / "list.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    cc = _col(pf, "tags.list.element")
    assert cc.max_repetition_level == 1
    page = _data_pages(cc)[0]
    decoded = page.decode()
    # The non-null leaf values are the flattened list elements; the trailing
    # empty list contributes a null leaf (no value). A new list starts at
    # repetition level 0, a continuation at level 1.
    assert page.physical_values() == [1, 2, 3]
    assert decoded.num_nulls == 1
    assert isinstance(decoded.repetition_levels, RleBitPackedStream)
    assert page.repetition_levels() == [0, 1, 0, 0]


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
    assert _data_pages(cc)[0].physical_values() == []
    if use_dictionary:
        # A dictionary-encoded all-null page still yields an (empty) index
        # stream; the dictionary resolves to an empty list.
        assert decoded.encoding == "RLE_DICTIONARY"
        assert isinstance(decoded.values, RleBitPackedStream)
        assert decoded.values.values == ()
        assert decoded.values.runs == ()
        assert cc.dictionary() == []
    else:
        assert isinstance(decoded.values, PlainValues)
        assert decoded.values.values == ()


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


def test_missing_dictionary_on_resolution(tmp_path, monkeypatch):
    """``decode()`` of a dictionary page yields the index stream without the
    dictionary; only resolving to values (``physical_values``) needs it."""
    path = _flat(tmp_path / "f.parquet", use_dictionary=True)
    pf = ParquetFile(str(path))
    cc = _col(pf, "name")
    page = _data_pages(cc)[0]
    # decode() succeeds without the dictionary.
    assert isinstance(page.decode().values, RleBitPackedStream)
    monkeypatch.setattr(ColumnChunk, "dictionary", lambda self: None)
    with pytest.raises(MissingDictionaryError) as exc:
        page.physical_values()
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


# ---------------------------------------------------------------------------
# Body section extents (byte provenance of each decoded section)
# ---------------------------------------------------------------------------


def _read(path: Path, extent: BodyExtent) -> bytes:
    return path.read_bytes()[extent.offset : extent.offset + extent.length]


def test_v1_compressed_extents_share_body_region(tmp_path):
    """V1's whole body is one compressed region, so every section extent
    points at the same on-disk [body_offset, compressed_size) range and
    differs only in its decompressed sub-range."""
    path = _flat(
        tmp_path / "v1s.parquet", data_page_version="1.0", compression="snappy"
    )
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    decoded = page.decode()
    de = decoded.definition_levels_extent
    ve = decoded.values_extent

    for ext in (de, ve):
        assert ext.offset == page.body_offset
        assert ext.length == page.compressed_size
        assert ext.compression_codec == "SNAPPY"
    # The decompressed sub-ranges are contiguous and tile the decompressed body.
    assert de.offset_uncompressed == 0
    assert ve.offset_uncompressed == de.offset_uncompressed + de.length_uncompressed
    assert ve.offset_uncompressed + ve.length_uncompressed == page.uncompressed_size
    # The on-disk region is real, readable bytes.
    assert len(_read(path, ve)) == ve.length


def test_v1_uncompressed_extents_are_plain_file_ranges(tmp_path):
    """A V1 page under the UNCOMPRESSED codec is directly file-addressable, so
    each section gets a plain {offset, length} with no compression fields."""
    path = _flat(tmp_path / "v1u.parquet", data_page_version="1.0", compression="none")
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    decoded = page.decode()
    de = decoded.definition_levels_extent
    ve = decoded.values_extent

    for ext in (de, ve):
        assert ext.compression_codec is None
        assert ext.offset_uncompressed is None
        assert ext.length_uncompressed is None
    # def block then values are contiguous, directly on disk.
    assert ve.offset == de.offset + de.length
    assert ve.offset + ve.length == page.body_offset + page.compressed_size


def test_v2_level_extents_are_plain_values_extent_matches_header(tmp_path):
    """V2 stores levels uncompressed on disk (plain extents); the values
    section is its own on-disk region."""
    path = _flat(
        tmp_path / "v2s.parquet", data_page_version="2.0", compression="snappy"
    )
    pf = ParquetFile(str(path))
    page = _data_pages(_col(pf, "id"))[0]
    h = page._t.data_page_header_v2
    decoded = page.decode()
    de = decoded.definition_levels_extent
    ve = decoded.values_extent

    # def level block: plain, length == header's definition_levels_byte_length.
    assert de.compression_codec is None
    assert de.offset == page.body_offset + (h.repetition_levels_byte_length or 0)
    assert de.length == h.definition_levels_byte_length
    # values section starts right after the levels.
    assert ve.offset == de.offset + de.length


def test_v2_compressed_values_extent_carries_codec(tmp_path):
    """A genuinely-compressed V2 values section carries the compressed-region
    form (codec + decompressed length)."""
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
    ve = page.decode().values_extent
    assert ve.compression_codec == "SNAPPY"
    assert ve.offset_uncompressed == 0
    assert ve.length_uncompressed > ve.length  # 500x"x"*64 compresses well
    assert len(_read(path, ve)) == ve.length


def test_required_column_has_no_level_extents(tmp_path):
    """A required, non-repeated column writes no level blocks, so both level
    extents are None."""
    schema = pa.schema([pa.field("x", pa.int32(), nullable=False)])
    table = pa.table({"x": [1, 2, 3]}, schema=schema)
    path = _write(tmp_path / "req.parquet", table, use_dictionary=False)
    pf = ParquetFile(str(path))
    decoded = _data_pages(_col(pf, "x"))[0].decode()
    assert decoded.repetition_levels_extent is None
    assert decoded.definition_levels_extent is None
    assert decoded.values_extent is not None
