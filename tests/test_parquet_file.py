"""Tests for :class:`parquet_analyzer.ParquetFile` and its wrappers.

Two distinct kinds of coverage live in this file:

1. **API surface tests** -- exercise each public property and method on
   ParquetFile / RowGroup / ColumnChunk / Page against pyarrow-generated
   fixtures and the bundled titanic.parquet. Establishes that the new
   lazy API actually returns the right data.

2. **Behaviour preservation tests** -- the lazy core replaced the eager
   ``parse_parquet_file()`` free function. To prove no behaviour was lost,
   compare against a JSON snapshot captured against pre-refactor master
   (committed at ``tests/snapshots/titanic_parse_v0.4.0.json``). The
   snapshot is a change-detector, not a contract -- if the diff
   legitimately changes output, update the snapshot in the same commit
   and document the change in the PR description.

3. **Laziness proof** -- a sentinel read-count probe verifies that
   constructing a ParquetFile and accessing footer-only properties
   triggers only a small, bounded number of file reads (no per-page
   walks). This is what the perf claim actually is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parquet_analyzer import (
    ColumnChunk,
    Page,
    ParquetFile,
    RowGroup,
    json_encode,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TITANIC = Path(__file__).parent / "data" / "titanic.parquet"
SNAPSHOT = Path(__file__).parent / "snapshots" / "titanic_parse_v0.4.0.json"


@pytest.fixture()
def small_parquet(tmp_path):
    """3 rows x 2 columns, V1 data pages, with key-value metadata."""
    table = pa.table(
        {
            "ints": pa.array([1, 2, 3], type=pa.int32()),
            "floats": pa.array([1.0, 2.5, 3.25]),
        },
        metadata={"author": "test", "schema_version": "1"},
    )
    path = tmp_path / "small.parquet"
    pq.write_table(table, path, compression="snappy", data_page_version="1.0")
    return path


@pytest.fixture()
def nested_row_groups_parquet(tmp_path):
    """A file with multiple row groups + dictionary-encoded pages."""
    dict_array = pa.array(
        ["alpha", "beta", "gamma", "beta", "alpha"],
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    table = pa.table({"dict_col": dict_array, "ints": pa.array([1, 2, 3, 4, 5])})
    path = tmp_path / "multi.parquet"
    pq.write_table(
        table, path, row_group_size=2, use_dictionary=True, write_page_index=True
    )
    return path


# ---------------------------------------------------------------------------
# Construction / lifecycle
# ---------------------------------------------------------------------------


def test_construct_reads_footer_only(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        assert pf.path == str(small_parquet)
        assert pf.num_rows == 3
        assert pf.num_columns == 2
        assert pf.num_row_groups == 1
        # No exception, no eager walk required for these properties.
    finally:
        pf.close()


def test_context_manager_closes(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        assert pf.num_rows == 3
    # _f.closed should be true now
    assert pf._f.closed


def test_repr_shape(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        r = repr(pf)
        assert "ParquetFile(" in r
        assert "num_rows=3" in r
        assert "num_columns=2" in r
    finally:
        pf.close()


def test_invalid_header_raises(tmp_path):
    bad = tmp_path / "no-magic.parquet"
    bad.write_bytes(b"BAD!" + b"\x00" * 12)
    with pytest.raises(ValueError, match="missing PAR1 header"):
        ParquetFile(str(bad))


def test_invalid_footer_raises(tmp_path):
    import struct

    bad = tmp_path / "no-footer-magic.parquet"
    bad.write_bytes(b"PAR1" + b"\x00" * 12 + struct.pack("<I", 0) + b"BAD!")
    with pytest.raises(ValueError, match="missing PAR1 footer"):
        ParquetFile(str(bad))


def test_construct_closes_handle_on_invalid(tmp_path, monkeypatch):
    """If footer parse raises, the file handle should not leak."""
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"BAD!" + b"\x00" * 12)

    # Capture the handle the ctor opens so we can assert it was closed.
    captured = {}
    original_open = open

    def tracking_open(*args, **kwargs):
        f = original_open(*args, **kwargs)
        captured["f"] = f
        return f

    monkeypatch.setitem(__builtins__, "open", tracking_open) if isinstance(
        __builtins__, dict
    ) else monkeypatch.setattr("builtins.open", tracking_open)

    with pytest.raises(ValueError):
        ParquetFile(str(bad))
    assert captured["f"].closed, "file handle leaked on invalid construction"


# ---------------------------------------------------------------------------
# Footer-derived properties
# ---------------------------------------------------------------------------


def test_kv_metadata_returns_list_of_tuples(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        kv = pf.kv_metadata
        assert isinstance(kv, list)
        # pyarrow writes its own ARROW:schema entry plus our 2 keys
        keys = {k for k, _ in kv}
        assert "author" in keys
        assert "schema_version" in keys
    finally:
        pf.close()


def test_kv_metadata_lookup_first_wins(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        assert pf.kv_metadata_lookup("author") == "test"
        assert pf.kv_metadata_lookup("does-not-exist") is None
    finally:
        pf.close()


def test_footer_segment_shape(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        seg = pf.footer_segment
        assert seg["name"] == "footer"
        assert seg["offset"] == pf.footer_offset
        assert seg["length"] == pf.footer_size
    finally:
        pf.close()


def test_schema_is_list(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        schema = pf.schema
        assert isinstance(schema, list)
        assert len(schema) > 0
    finally:
        pf.close()


def test_file_size_matches_disk(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        assert pf.file_size == small_parquet.stat().st_size
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# RowGroup / ColumnChunk wrappers
# ---------------------------------------------------------------------------


def test_row_groups_returns_wrappers(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        rgs = pf.row_groups
        assert isinstance(rgs, tuple)
        assert len(rgs) == 1
        assert all(isinstance(rg, RowGroup) for rg in rgs)
        # Cached: same tuple identity on repeat access
        assert pf.row_groups is rgs
    finally:
        pf.close()


def test_row_group_properties(nested_row_groups_parquet):
    pf = ParquetFile(str(nested_row_groups_parquet))
    try:
        rgs = pf.row_groups
        # 5 rows, row_group_size=2 -> 3 row groups
        assert len(rgs) == 3
        assert rgs[0].num_rows == 2
        assert rgs[-1].num_rows == 1
        assert all(rg.total_byte_size > 0 for rg in rgs)
    finally:
        pf.close()


def test_column_chunk_properties(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        cols = pf.row_groups[0].columns
        assert isinstance(cols, tuple)
        assert len(cols) == 2
        assert all(isinstance(c, ColumnChunk) for c in cols)
        ints, floats = cols
        assert ints.path == ("ints",)
        assert ints.type == "INT32"
        assert ints.num_values == 3
        assert ints.codec == "SNAPPY"
        assert ints.total_compressed_size > 0
        assert ints.total_uncompressed_size > 0
        assert ints.data_page_offset >= 4  # at least past file header
        assert ints.has_bloom_filter is False
        assert floats.type == "DOUBLE"
    finally:
        pf.close()


def test_column_chunk_encodings_returns_strings(small_parquet):
    pf = ParquetFile(str(small_parquet))
    try:
        cc = pf.row_groups[0].columns[0]
        encs = cc.encodings
        assert isinstance(encs, tuple)
        assert all(isinstance(e, str) for e in encs)
        # PLAIN data pages on a small int column
        assert "PLAIN" in encs or "RLE" in encs
    finally:
        pf.close()


def test_column_chunk_page_indexes_offsets(nested_row_groups_parquet):
    """When pyarrow writes a page index, ColumnChunk should expose the
    column_index_offset and offset_index_offset."""
    pf = ParquetFile(str(nested_row_groups_parquet))
    try:
        cc = pf.row_groups[0].columns[0]
        assert cc.column_index_offset is not None
        assert cc.offset_index_offset is not None
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# Page wrapper (Phase 2 lazy walking within a chunk)
# ---------------------------------------------------------------------------


def test_pages_walk_per_chunk(nested_row_groups_parquet):
    pf = ParquetFile(str(nested_row_groups_parquet))
    try:
        rg = pf.row_groups[0]
        cc = rg.columns[0]
        pages = cc.pages()
        assert isinstance(pages, tuple)
        assert len(pages) > 0
        assert all(isinstance(p, Page) for p in pages)
        # Cached
        assert cc.pages() is pages
    finally:
        pf.close()


def test_page_properties(nested_row_groups_parquet):
    pf = ParquetFile(str(nested_row_groups_parquet))
    try:
        # dict_col is dictionary-encoded, so we expect at least one
        # DICTIONARY_PAGE + one DATA_PAGE in the first row group.
        cc = pf.row_groups[0].columns[0]
        pages = cc.pages()
        types = [p.type for p in pages]
        assert "DICTIONARY_PAGE" in types
        for p in pages:
            assert p.offset > 0
            assert p.header_size > 0
            assert p.uncompressed_size > 0
            assert p.compressed_size > 0
            assert isinstance(p.encoding, str)
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# Laziness proof: footer-only construction does not trigger page walks
# ---------------------------------------------------------------------------


def test_construction_does_not_read_pages(small_parquet, monkeypatch):
    """Constructing a ParquetFile and reading footer-only properties must
    NOT call read_thrift_segment with name='page'. This is the whole point
    of the lazy core: page-header parsing is deferred."""
    from parquet_analyzer import _core, parquet_file as _pf_module

    page_read_count = 0
    original = _core.read_thrift_segment

    def counting(f, offset, name, thrift_class):
        nonlocal page_read_count
        if name == "page":
            page_read_count += 1
        return original(f, offset, name, thrift_class)

    monkeypatch.setattr(_core, "read_thrift_segment", counting)
    # parquet_file binds `read_thrift_segment` at import (hoisted out of
    # the per-call function-local path for perf), so the monkeypatch above
    # alone doesn't redirect ColumnChunk.pages(); patch the rebound symbol
    # too. _walk_chunks_eager lives in _core itself so its callers still
    # see the patched version.
    monkeypatch.setattr(_pf_module, "read_thrift_segment", counting)

    pf = ParquetFile(str(small_parquet))
    try:
        # Touch footer-only surface.
        _ = pf.num_rows
        _ = pf.num_columns
        _ = pf.schema
        _ = pf.kv_metadata
        _ = pf.footer_summary
        # Walk row groups and column chunks (footer-derived).
        for rg in pf.row_groups:
            for cc in rg.columns:
                _ = cc.path
                _ = cc.type
                _ = cc.codec
                _ = cc.encodings
                _ = cc.num_values
    finally:
        pf.close()

    assert page_read_count == 0, (
        f"footer-only access triggered {page_read_count} page-header reads "
        "(should be 0)"
    )


def test_pages_call_triggers_only_that_chunks_headers(
    nested_row_groups_parquet, monkeypatch
):
    """Calling pages() on one column chunk reads only that chunk's page
    headers, not the whole file."""
    from parquet_analyzer import _core, parquet_file as _pf_module

    page_read_count = 0
    original = _core.read_thrift_segment

    def counting(f, offset, name, thrift_class):
        nonlocal page_read_count
        if name == "page":
            page_read_count += 1
        return original(f, offset, name, thrift_class)

    monkeypatch.setattr(_core, "read_thrift_segment", counting)
    monkeypatch.setattr(_pf_module, "read_thrift_segment", counting)

    pf = ParquetFile(str(nested_row_groups_parquet))
    try:
        cc = pf.row_groups[0].columns[0]
        assert page_read_count == 0  # constructor + row_groups was footer-only
        pages = cc.pages()
        # We touched ONE chunk; counted pages should match.
        assert page_read_count == len(pages)
    finally:
        pf.close()


def test_full_summary_triggers_eager_walk(small_parquet, monkeypatch):
    from parquet_analyzer import _core

    page_read_count = 0
    original = _core.read_thrift_segment

    def counting(f, offset, name, thrift_class):
        nonlocal page_read_count
        if name == "page":
            page_read_count += 1
        return original(f, offset, name, thrift_class)

    monkeypatch.setattr(_core, "read_thrift_segment", counting)

    pf = ParquetFile(str(small_parquet))
    try:
        _ = pf.full_summary
    finally:
        pf.close()

    assert page_read_count > 0, "full_summary should trigger page walks"


# ---------------------------------------------------------------------------
# Behaviour preservation: byte-identical output vs pre-refactor snapshot
# ---------------------------------------------------------------------------


def test_titanic_all_segments_matches_pre_refactor_snapshot():
    """The legacy parse_parquet_file()[0] output for the bundled titanic
    fixture, captured against pre-refactor master, must match
    pf.all_segments() exactly. This is the byte-identical guarantee for
    the CLI segments mode."""
    with open(SNAPSHOT) as f:
        snap = json.load(f)
    pf = ParquetFile(str(TITANIC))
    try:
        segs = pf.all_segments()
        # Roundtrip through the JSON encoder so tuples/bytes/etc. compare.
        observed = json.loads(json.dumps(segs, default=json_encode))
    finally:
        pf.close()
    assert observed == snap["segments"], "all_segments diverged from snapshot"


def test_titanic_all_pages_matches_pre_refactor_snapshot():
    with open(SNAPSHOT) as f:
        snap = json.load(f)
    pf = ParquetFile(str(TITANIC))
    try:
        pages = pf.all_pages()
        observed = json.loads(json.dumps(pages, default=json_encode))
    finally:
        pf.close()
    assert observed == snap["pages"], "all_pages diverged from snapshot"


def test_titanic_full_summary_matches_pre_refactor_snapshot():
    with open(SNAPSHOT) as f:
        snap = json.load(f)
    pf = ParquetFile(str(TITANIC))
    try:
        observed = pf.full_summary
    finally:
        pf.close()
    assert observed == snap["summary"], "full_summary diverged from snapshot"


def test_titanic_footer_matches_pre_refactor_snapshot():
    with open(SNAPSHOT) as f:
        snap = json.load(f)
    pf = ParquetFile(str(TITANIC))
    try:
        observed = json.loads(json.dumps(pf.footer, default=json_encode))
    finally:
        pf.close()
    assert observed == snap["footer"], "footer diverged from snapshot"


# ---------------------------------------------------------------------------
# Cache identity (the docstring promises must be kept)
# ---------------------------------------------------------------------------


def test_full_summary_caches_result(small_parquet):
    """full_summary docstring promises caching; assert true `is`-identity
    on the returned dict to lock in that promise. Pre-fix this rebuilt
    the summary dict per-call (~5ms repeat cost on a 100-rg file)."""
    pf = ParquetFile(str(small_parquet))
    try:
        first = pf.full_summary
        second = pf.full_summary
        assert first is second, "full_summary should return cached reference"
    finally:
        pf.close()


def test_all_pages_caches_result(small_parquet):
    """all_pages docstring promises caching; assert true `is`-identity."""
    pf = ParquetFile(str(small_parquet))
    try:
        first = pf.all_pages()
        second = pf.all_pages()
        assert first is second, "all_pages should return cached reference"
    finally:
        pf.close()


def test_column_offset_map_caches_result(small_parquet):
    """column_offset_map docstring promises caching; assert
    `is`-identity. (This one was already correctly cached as part of
    _ensure_eager_walked; the test pins the documented contract.)"""
    pf = ParquetFile(str(small_parquet))
    try:
        first = pf.column_offset_map
        second = pf.column_offset_map
        assert first is second
    finally:
        pf.close()


def test_all_segments_caches_result(small_parquet):
    """all_segments docstring promises caching; assert `is`-identity."""
    pf = ParquetFile(str(small_parquet))
    try:
        first = pf.all_segments()
        second = pf.all_segments()
        assert first is second
    finally:
        pf.close()
