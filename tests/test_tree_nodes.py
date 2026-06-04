"""Tests for the v0 tree-node interface (``_kind`` / ``_offset`` /
``_length`` / ``to_json``) on :class:`parquet_analyzer.ParquetFile` and
its wrappers.

The test suite enforces the contract from ``docs/tree-schema.md``:

1. **Universal node interface** — every materialized or stub node has
   ``_kind`` / ``_offset`` / ``_length``.
2. **Per-kind catalog** — each kind emits only its allowed set of
   content / children keys.
3. **Depth rules** — uniform rule (depth=N → root + N levels
   materialized, level N+1 stubbed; depth="all" → everything
   materialized; depth=0 → root stub only; ``$schema`` always emitted).
4. **Lazy markers** — the six lazy kinds (page kinds + opaque
   branches) carry ``_lazy: true`` when stubbed, drop it when
   materialized.
5. **Layout view structural invariants** — children sorted by
   ``_offset`` with no overlaps, ``unknown`` gap fills.
6. **Tree view structural invariants** — named-child reachability,
   the file/footer ``row_groups`` duplication.
7. **Read-count laziness** — page-header and opaque-branch thrift
   reads happen only when nodes are materialized.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parquet_analyzer import ParquetFile


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TITANIC = Path(__file__).parent / "data" / "titanic.parquet"


@pytest.fixture()
def small_parquet(tmp_path):
    """V1-page file with 1 row group, 2 columns, no dictionary."""
    table = pa.table(
        {
            "ints": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "floats": pa.array([1.0, 2.5, 3.25, 4.5, 5.0]),
        },
        metadata={"author": "test"},
    )
    path = tmp_path / "small.parquet"
    pq.write_table(
        table,
        path,
        compression="snappy",
        data_page_version="1.0",
        use_dictionary=False,
    )
    return path


@pytest.fixture()
def indexed_parquet(tmp_path):
    """Multi-row-group file with dictionary + OffsetIndex + ColumnIndex
    so the lazy kinds are observable."""
    dict_array = pa.array(
        ["alpha", "beta", "gamma", "beta", "alpha", "delta", "alpha", "gamma"],
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    table = pa.table(
        {"dict_col": dict_array, "ints": pa.array([1, 2, 3, 4, 5, 6, 7, 8])}
    )
    path = tmp_path / "indexed.parquet"
    pq.write_table(
        table,
        path,
        row_group_size=4,
        use_dictionary=True,
        write_page_index=True,
        write_statistics=True,
        compression="snappy",
    )
    return path


@pytest.fixture()
def bloomy_parquet(tmp_path):
    """File written with an explicit bloom filter for one column."""
    table = pa.table(
        {
            "strings": pa.array(["aa", "bb", "cc", "dd"]),
            "ints": pa.array([10, 20, 30, 40]),
        }
    )
    path = tmp_path / "bloomy.parquet"
    # Use ParquetWriter so we can pass bloom_filter_columns.
    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression="snappy",
        write_page_index=True,
    )
    try:
        writer.write_table(table)
    finally:
        writer.close()
    return path


@pytest.fixture()
def empty_parquet(tmp_path):
    """A 0-row dictionary-encoded file.

    pyarrow records ``dictionary_page_offset`` (truthy) but writes no
    data page, so ``data_page_offset == 0`` and ``ColumnChunk.pages()``
    returns ``()``. This is the geometry that exposed the
    ``column_chunk_data_region`` offset-0 overlap and the synthetic
    dictionary-page render crash.
    """
    table = pa.table(
        {
            "ints": pa.array([], type=pa.int32()),
            "strs": pa.array([], type=pa.string()),
        }
    )
    path = tmp_path / "empty.parquet"
    pq.write_table(table, path, use_dictionary=True)
    return path


# ---------------------------------------------------------------------------
# Per-kind allowed-key catalog
# ---------------------------------------------------------------------------

# Materialized form: which keys may appear (system + content + children).
# Drives universal-allowed-keys check. Excludes ``_lazy`` (only on stubs)
# and the response-shape ``$schema`` (only on root).
ALLOWED_MATERIALIZED_KEYS: dict[str, set[str]] = {
    "file": {
        "_kind",
        "_offset",
        "_length",
        "path",
        # tree view children
        "header_magic",
        "row_groups",
        "footer",
        "footer_length",
        "trailer_magic",
        # layout view children
        "children",
    },
    "header_magic": {"_kind", "_offset", "_length", "_value"},
    "trailer_magic": {"_kind", "_offset", "_length", "_value"},
    "footer_length": {"_kind", "_offset", "_length", "_value"},
    "footer": {
        "_kind",
        "_offset",
        "_length",
        "version",
        "num_rows",
        "created_by",
        # tree view children
        "schema",
        "kv_metadata",
        "row_groups",
        # layout view children
        "children",
    },
    "schema": {"_kind", "_offset", "_length", "_value"},
    "kv_metadata": {"_kind", "_offset", "_length", "_value"},
    "row_group": {
        "_kind",
        "_offset",
        "_length",
        "num_rows",
        "total_byte_size",
        "total_compressed_size",
        "ordinal",
        # tree view children
        "columns",
        # layout view children
        "children",
    },
    "column_chunk": {
        "_kind",
        "_offset",
        "_length",
        "path",
        "path_display",
        "type",
        "codec",
        "encodings",
        "num_values",
        "compressed_size",
        "uncompressed_size",
        "data_page_offset",
        "dictionary_page_offset",
        "file_offset",
        "statistics",
        # tree view children
        "dictionary_page",
        "pages",
        "offset_index",
        "column_index",
        "bloom_filter",
        # layout view refs (no children array; layout column_chunk is a leaf
        # wrt children)
        "data_region_ref",
        "offset_index_ref",
        "column_index_ref",
        "bloom_filter_ref",
    },
    "column_chunk_data_region": {
        "_kind",
        "_offset",
        "_length",
        "chunk_ref",
        "row_group_index",
        "column_position_in_row_group",
        "dictionary_page",
        "pages",
    },
    "dictionary_page": {
        "_kind",
        "_offset",
        "_length",
        "page_type",
        "encoding",
        "num_values",
        "uncompressed_size",
        "compressed_size",
        "is_compressed",
        "crc",
    },
    "data_page_v1": {
        "_kind",
        "_offset",
        "_length",
        "page_type",
        "encoding",
        "num_values",
        "uncompressed_size",
        "compressed_size",
        "definition_level_encoding",
        "repetition_level_encoding",
        "statistics",
        "crc",
    },
    "data_page_v2": {
        "_kind",
        "_offset",
        "_length",
        "page_type",
        "encoding",
        "num_values",
        "num_nulls",
        "num_rows",
        "is_compressed",
        "uncompressed_size",
        "compressed_size",
        "definition_levels_byte_length",
        "repetition_levels_byte_length",
        "statistics",
        "crc",
    },
    "offset_index": {"_kind", "_offset", "_length"},
    "column_index": {"_kind", "_offset", "_length"},
    "bloom_filter_header": {"_kind", "_offset", "_length"},
    "unknown": {"_kind", "_offset", "_length"},
}


LAZY_KINDS = {
    "dictionary_page",
    "data_page_v1",
    "data_page_v2",
    "offset_index",
    "column_index",
    "bloom_filter_header",
}


# ---------------------------------------------------------------------------
# Universal contract — walk all nodes and assert invariants
# ---------------------------------------------------------------------------


def _iter_nodes(node, *, include_root: bool = True):
    """Yield every dict-node in the JSON tree (including root)."""
    if not isinstance(node, dict):
        return
    if include_root:
        yield node
    for value in node.values():
        if isinstance(value, dict):
            yield from _iter_nodes(value, include_root=True)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from _iter_nodes(item, include_root=True)


def _assert_universal_contract(root: dict, *, view: str) -> None:
    """Assert every node in the tree obeys the universal contract."""
    assert root["$schema"] == f"parquet-analyzer/v2/{view}", (
        f"root $schema mismatch: {root['$schema']!r}"
    )
    # The root itself has $schema in addition to the kind's keys; allow it.
    seen_root = False
    for node in _iter_nodes(root):
        kind = node.get("_kind")
        # Synthetic helper dicts inside ``_value`` lists (schema elements,
        # kv pairs) don't have _kind — skip them.
        if kind is None:
            continue
        assert "_offset" in node, f"{kind} node missing _offset: {node}"
        assert "_length" in node, f"{kind} node missing _length: {node}"
        assert isinstance(node["_offset"], int), (
            f"{kind} _offset not int: {node['_offset']!r}"
        )
        assert isinstance(node["_length"], int), (
            f"{kind} _length not int: {node['_length']!r}"
        )
        allowed = ALLOWED_MATERIALIZED_KEYS[kind] | {"$schema", "_lazy"}
        extra = set(node.keys()) - allowed
        assert not extra, f"{kind} node has unexpected keys {extra}; allowed {allowed}"
        # Lazy markers only on lazy kinds.
        if "_lazy" in node:
            assert kind in LAZY_KINDS, f"{kind} carries _lazy but is not a lazy kind"
            assert node["_lazy"] is True
        # Branches never carry _value (only leaves do).
        if kind in {
            "file",
            "footer",
            "row_group",
            "column_chunk",
            "column_chunk_data_region",
        }:
            assert "_value" not in node, f"branch kind {kind} should not carry _value"
        if not seen_root:
            seen_root = True


def _assert_layout_invariants(out: dict) -> None:
    """Assert the layout-view ``children`` are offset-sorted, free of
    overlaps, and exactly contiguous (gap-filled) across ``[0, file)``."""
    children = out["children"]
    offsets = [c["_offset"] for c in children]
    assert offsets == sorted(offsets), f"layout children not offset-sorted: {offsets}"
    assert children[0]["_offset"] == 0, "first layout child must start at offset 0"
    assert children[-1]["_offset"] + children[-1]["_length"] == out["_length"], (
        "last layout child must end at file end"
    )
    for a, b in zip(children, children[1:]):
        assert b["_offset"] == a["_offset"] + a["_length"], (
            f"non-contiguous: {a['_kind']}@{a['_offset']}+{a['_length']} "
            f"then {b['_kind']}@{b['_offset']} (overlap or unfilled gap)"
        )


# ---------------------------------------------------------------------------
# Wrapper-level _kind / _offset / _length
# ---------------------------------------------------------------------------


def test_parquet_file_node_properties(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        assert pf._kind == "file"
        assert pf._offset == 0
        assert pf._length == pf.file_size


def test_row_group_node_properties(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        rg = pf.row_groups[0]
        assert rg._kind == "row_group"
        # rg._offset is inside the footer
        assert pf.footer_offset <= rg._offset < pf.footer_offset + pf.footer_size
        assert rg._length > 0


def test_column_chunk_node_properties(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        cc = pf.row_groups[0].columns[0]
        assert cc._kind == "column_chunk"
        # cc thrift extent is inside the row-group thrift extent.
        rg = pf.row_groups[0]
        assert rg._offset <= cc._offset
        assert cc._offset + cc._length <= rg._offset + rg._length


def test_page_node_properties(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        cc = pf.row_groups[0].columns[0]
        pages = cc.pages()
        assert pages, "fixture must produce at least one page"
        p = pages[0]
        assert p._kind in {"dictionary_page", "data_page_v1", "data_page_v2"}
        assert p._offset == p._segment["offset"]
        # length covers header thrift + compressed body
        assert p._length == p._segment["length"] + p._t.compressed_page_size


# ---------------------------------------------------------------------------
# Depth rule — uniform N + "all" + 0
# ---------------------------------------------------------------------------


def test_depth_zero_root_stub_only_tree(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=0)
    assert set(out.keys()) == {"$schema", "_kind", "_offset", "_length"}
    assert out["_kind"] == "file"
    assert out["$schema"] == "parquet-analyzer/v2/tree"


def test_depth_zero_root_stub_only_layout(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=0)
    assert set(out.keys()) == {"$schema", "_kind", "_offset", "_length"}
    assert out["$schema"] == "parquet-analyzer/v2/layout"


def test_depth_one_root_materialized_children_stubbed_tree(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=1)
    assert out["_kind"] == "file"
    # File-level fields present
    assert "path" in out
    assert "row_groups" in out
    assert "footer" in out
    # Direct children are stubs (just _kind/_offset/_length)
    rg_child = out["row_groups"][0]
    assert set(rg_child.keys()) == {"_kind", "_offset", "_length"}
    # Footer too (not a lazy kind, no _lazy)
    assert set(out["footer"].keys()) == {"_kind", "_offset", "_length"}
    # Header magic is a leaf at depth 1 -- still stub, no _value
    assert set(out["header_magic"].keys()) == {"_kind", "_offset", "_length"}


def test_depth_two_tree_materializes_through_row_group(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=2)
    rg = out["row_groups"][0]
    # row_group materialized at depth=2 (file→rg is level 1, content shown)
    assert rg["num_rows"] == 5
    assert "columns" in rg
    # but columns at level 2 are stubs
    cc = rg["columns"][0]
    assert set(cc.keys()) == {"_kind", "_offset", "_length"}


def test_depth_all_materializes_everything_no_lazy_markers(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth="all")
    _assert_universal_contract(out, view="tree")
    # No _lazy anywhere -- depth=all pays all I/O
    for node in _iter_nodes(out):
        assert "_lazy" not in node, (
            f"_lazy should be absent at depth=all on {node.get('_kind')}"
        )


def test_lazy_markers_present_on_lazy_kind_stubs(small_parquet):
    """At depth that produces stubs of page kinds, _lazy: true must appear."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=3)
    # Walk to a column_chunk and inspect its pages (which should be stubs
    # at depth 3 = file→rg→cc, pages at level 3 are stubs).
    cc = out["row_groups"][0]["columns"][0]
    assert cc["pages"], "column chunk must have at least one page"
    for page in cc["pages"]:
        assert page.get("_lazy") is True, f"page stub missing _lazy: {page}"


# ---------------------------------------------------------------------------
# $schema URI placement
# ---------------------------------------------------------------------------


def test_schema_uri_only_on_root(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth="all")
    assert out["$schema"] == "parquet-analyzer/v2/tree"
    # Walk all descendants; none should carry $schema
    for node in _iter_nodes(out, include_root=False):
        assert "$schema" not in node, (
            f"non-root node carries $schema: {node.get('_kind')}"
        )


def test_schema_uri_layout(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
    assert out["$schema"] == "parquet-analyzer/v2/layout"


# ---------------------------------------------------------------------------
# Tree view structural invariants
# ---------------------------------------------------------------------------


def test_tree_view_file_children_order(small_parquet):
    """File's named children in tree view: header_magic, row_groups[],
    footer, footer_length, trailer_magic."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=1)
    assert list(out.keys())[0] == "$schema"
    # System fields first, then path, then ordered children
    keys = list(out.keys())
    expected_tail = [
        "header_magic",
        "row_groups",
        "footer",
        "footer_length",
        "trailer_magic",
    ]
    assert keys[-len(expected_tail) :] == expected_tail


def test_tree_view_row_groups_duplicated_under_file_and_footer(small_parquet):
    """v0 schema doc accepts that row_groups appears under both file (as
    a tree-view shortcut) and footer; identical _offset/_length."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=1)
    file_rg = out["row_groups"][0]
    # Need depth=2 to see footer.row_groups (children one level deeper)
    with ParquetFile(str(small_parquet)) as pf:
        out2 = pf.to_json(view="tree", depth=2)
    footer_rg = out2["footer"]["row_groups"][0]
    assert file_rg["_offset"] == footer_rg["_offset"]
    assert file_rg["_length"] == footer_rg["_length"]


def test_tree_view_column_chunk_has_pages_no_data_region(small_parquet):
    """In tree view, column_chunk exposes pages/dictionary_page directly
    (NOT data_region — that's layout-only)."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth="all")
    cc = out["row_groups"][0]["columns"][0]
    assert "pages" in cc
    assert "dictionary_page" in cc
    # No data_region named child
    assert "data_region" not in cc
    assert "children" not in cc  # layout-only key


# ---------------------------------------------------------------------------
# Layout view structural invariants
# ---------------------------------------------------------------------------


def test_layout_view_children_sorted_by_offset(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
    children = out["children"]
    offsets = [c["_offset"] for c in children]
    assert offsets == sorted(offsets), f"layout children not offset-sorted: {offsets}"


def test_layout_view_children_no_overlaps(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
    children = out["children"]
    for a, b in zip(children, children[1:]):
        assert a["_offset"] + a["_length"] <= b["_offset"], (
            f"overlap between {a['_kind']}@{a['_offset']} (len {a['_length']}) "
            f"and {b['_kind']}@{b['_offset']}"
        )


def test_layout_view_continuity_via_unknown_gap_fill(small_parquet):
    """Adjacent layout children should be exactly contiguous, with
    ``unknown`` nodes filling any gaps."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
    children = out["children"]
    # First child starts at file offset 0
    assert children[0]["_offset"] == 0
    # Last child ends at file end
    assert children[-1]["_offset"] + children[-1]["_length"] == out["_length"]
    # No gaps between siblings (gap-fill via unknown nodes makes this true)
    for a, b in zip(children, children[1:]):
        assert b["_offset"] == a["_offset"] + a["_length"], (
            f"gap between {a['_kind']}@{a['_offset']}+{a['_length']} "
            f"and {b['_kind']}@{b['_offset']}; gap-fill missed"
        )


def test_layout_view_excludes_row_group_and_column_chunk_at_file_level(
    small_parquet,
):
    """row_group and column_chunk thrifts live inside footer; file.children
    must not contain them as direct children."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
    kinds = {c["_kind"] for c in out["children"]}
    assert "row_group" not in kinds
    assert "column_chunk" not in kinds


def test_layout_view_column_chunk_data_region_present_per_chunk(small_parquet):
    """One column_chunk_data_region per column chunk in the file."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=1)
        expected_data_regions = pf.num_row_groups * len(pf.row_groups[0].columns)
    actual = sum(1 for c in out["children"] if c["_kind"] == "column_chunk_data_region")
    assert actual == expected_data_regions


def test_layout_view_data_region_children(small_parquet):
    """column_chunk_data_region.children = dictionary_page (when present)
    + pages, sorted by _offset."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="layout", depth=2)
    data_region = next(
        c for c in out["children"] if c["_kind"] == "column_chunk_data_region"
    )
    # depth=2 → data_region materialized; its children are stubs (level 2)
    assert "dictionary_page" in data_region
    assert "pages" in data_region
    assert "chunk_ref" in data_region
    assert "row_group_index" in data_region
    assert data_region["row_group_index"] == 0


# ---------------------------------------------------------------------------
# Universal contract — fixture sweeps
# ---------------------------------------------------------------------------


def test_universal_contract_titanic_tree_all():
    """Apply the universal contract sweep over the real titanic fixture
    at depth=all in tree view."""
    with ParquetFile(str(TITANIC)) as pf:
        out = pf.to_json(view="tree", depth="all")
    _assert_universal_contract(out, view="tree")


def test_universal_contract_titanic_layout_all():
    with ParquetFile(str(TITANIC)) as pf:
        out = pf.to_json(view="layout", depth="all")
    _assert_universal_contract(out, view="layout")


def test_universal_contract_indexed_tree_all(indexed_parquet):
    """Apply the sweep over a fixture with dictionary pages + page index.
    Exercises additional kinds vs. titanic."""
    with ParquetFile(str(indexed_parquet)) as pf:
        out = pf.to_json(view="tree", depth="all")
    _assert_universal_contract(out, view="tree")


def test_universal_contract_indexed_layout_all(indexed_parquet):
    with ParquetFile(str(indexed_parquet)) as pf:
        out = pf.to_json(view="layout", depth="all")
    _assert_universal_contract(out, view="layout")


# ---------------------------------------------------------------------------
# Read-count laziness
# ---------------------------------------------------------------------------


def _install_read_probe(monkeypatch):
    """Install a read_thrift_segment probe counting calls per ``name``.

    Returns a dict whose keys are the ``name`` argument values and whose
    values are counts. Caller installs after construction so footer-parse
    reads are not counted.
    """
    from parquet_analyzer import _core, parquet_file as _pf_module

    counts: dict[str, int] = {}
    original = _core.read_thrift_segment

    def counting(f, offset, name, thrift_class):
        counts[name] = counts.get(name, 0) + 1
        return original(f, offset, name, thrift_class)

    monkeypatch.setattr(_core, "read_thrift_segment", counting)
    monkeypatch.setattr(_pf_module, "read_thrift_segment", counting)
    return counts


def test_tree_depth_2_skips_page_header_reads(small_parquet, monkeypatch):
    """tree depth=2 stops at column_chunk (level 2 = stubs); no page
    enumeration."""
    pf = ParquetFile(str(small_parquet))
    try:
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="tree", depth=2)
        assert counts.get("page", 0) == 0, (
            f"depth=2 tree triggered {counts.get('page', 0)} page-header "
            "reads; should be 0 (columns are stubbed)"
        )
    finally:
        pf.close()


def test_tree_depth_3_triggers_page_header_reads(small_parquet, monkeypatch):
    """tree depth=3 materializes column_chunks → must enumerate pages."""
    pf = ParquetFile(str(small_parquet))
    try:
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="tree", depth=3)
        assert counts.get("page", 0) > 0, "depth=3 tree must trigger page-header reads"
    finally:
        pf.close()


def test_layout_depth_1_skips_page_header_reads(small_parquet, monkeypatch):
    """layout depth=1 stubs column_chunk_data_region; no pages walked."""
    pf = ParquetFile(str(small_parquet))
    try:
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="layout", depth=1)
        assert counts.get("page", 0) == 0, (
            f"layout depth=1 triggered {counts.get('page', 0)} page reads; should be 0"
        )
    finally:
        pf.close()


def test_layout_depth_2_triggers_page_header_reads(small_parquet, monkeypatch):
    """layout depth=2 materializes column_chunk_data_region → enumerates
    pages for each chunk."""
    pf = ParquetFile(str(small_parquet))
    try:
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="layout", depth=2)
        assert counts.get("page", 0) > 0, (
            "layout depth=2 must trigger page-header reads"
        )
    finally:
        pf.close()


def test_depth_all_triggers_opaque_branch_reads(indexed_parquet, monkeypatch):
    """depth=all materializes offset_index / column_index → their
    underlying thrift reads must happen."""
    pf = ParquetFile(str(indexed_parquet))
    try:
        # Ensure the fixture actually has indexes
        has_oi = any(cc.has_offset_index for rg in pf.row_groups for cc in rg.columns)
        if not has_oi:
            pytest.skip("fixture lacks offset_index — pyarrow version mismatch")
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="tree", depth="all")
        assert counts.get("offset_index", 0) > 0, (
            "depth=all must trigger offset_index reads when present"
        )
    finally:
        pf.close()


def test_offset_index_stub_does_not_trigger_read(indexed_parquet, monkeypatch):
    """offset_index emitted as a stub (column_chunk materialized but
    deeper levels stubbed) must NOT trigger the read."""
    pf = ParquetFile(str(indexed_parquet))
    try:
        has_oi = any(cc.has_offset_index for rg in pf.row_groups for cc in rg.columns)
        if not has_oi:
            pytest.skip("fixture lacks offset_index")
        counts = _install_read_probe(monkeypatch)
        # depth=3 = file→rg(L1)→cc(L2 materialized)→offset_index(L3 stub)
        pf.to_json(view="tree", depth=3)
        assert counts.get("offset_index", 0) == 0, (
            f"offset_index stub triggered {counts.get('offset_index', 0)} "
            "reads; should be 0"
        )
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# Wrapper-level to_json delegation
# ---------------------------------------------------------------------------


def test_row_group_to_json_root_emits_schema_uri(small_parquet):
    """A row group can also be the root of a to_json call; $schema is
    still attached."""
    with ParquetFile(str(small_parquet)) as pf:
        rg = pf.row_groups[0]
        out = rg.to_json(view="tree", depth=1)
    assert out["_kind"] == "row_group"
    assert out["$schema"] == "parquet-analyzer/v2/tree"


def test_column_chunk_to_json_root(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        cc = pf.row_groups[0].columns[0]
        out = cc.to_json(view="tree", depth=1)
    assert out["_kind"] == "column_chunk"
    assert out["$schema"] == "parquet-analyzer/v2/tree"
    # depth=1 materializes the cc; its pages are stubs
    assert "pages" in out


def test_page_to_json_root(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        page = pf.row_groups[0].columns[0].pages()[0]
        out = page.to_json(view="tree", depth=1)
    assert out["_kind"] in {"dictionary_page", "data_page_v1", "data_page_v2"}
    assert out["$schema"] == "parquet-analyzer/v2/tree"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_view_rejected(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        with pytest.raises(ValueError, match="view must be"):
            pf.to_json(view="bogus", depth=1)


def test_invalid_depth_rejected(small_parquet):
    with ParquetFile(str(small_parquet)) as pf:
        with pytest.raises(ValueError, match="depth"):
            pf.to_json(view="tree", depth=-1)
        with pytest.raises(ValueError, match="depth"):
            pf.to_json(view="tree", depth="deep")


# ---------------------------------------------------------------------------
# Footer extent extraction
# ---------------------------------------------------------------------------


def test_row_group_thrift_extents_inside_footer(small_parquet):
    """RowGroup._offset/_length is the thrift-struct extent inside the
    footer, NOT the data extent."""
    with ParquetFile(str(small_parquet)) as pf:
        rg = pf.row_groups[0]
        footer_start = pf.footer_offset
        footer_end = footer_start + pf.footer_size
        assert footer_start <= rg._offset < footer_end
        assert rg._offset + rg._length <= footer_end


def test_column_chunk_thrift_extents_inside_row_group(small_parquet):
    """ColumnChunk thrift extents are nested inside their row group's
    extent."""
    with ParquetFile(str(small_parquet)) as pf:
        rg = pf.row_groups[0]
        for cc in rg.columns:
            assert rg._offset <= cc._offset
            assert cc._offset + cc._length <= rg._offset + rg._length


# ---------------------------------------------------------------------------
# Empty / 0-row files (regression: data-region offset-0 overlap + synthetic
# dictionary-page render crash)
# ---------------------------------------------------------------------------


def test_empty_file_tree_all_does_not_crash(empty_parquet):
    """A 0-row dictionary column sets ``dictionary_page_offset`` but has
    no page header; materializing the synthetic dictionary-page fallback
    must not raise."""
    with ParquetFile(str(empty_parquet)) as pf:
        out = pf.to_json()  # default view="tree", depth="all"
    _assert_universal_contract(out, view="tree")
    # The dictionary-page child materialized to system fields only.
    cc = out["row_groups"][0]["columns"][0]
    dp = cc["dictionary_page"]
    assert dp is not None
    assert dp["_kind"] == "dictionary_page"
    # System fields only — no content keys leaked from a half-rendered node.
    assert set(dp) == {"_kind", "_offset", "_length"}


def test_empty_file_layout_no_overlap_at_offset_zero(empty_parquet):
    """The ``data_page_offset == 0`` sentinel must not place a
    ``column_chunk_data_region`` at byte 0 overlapping ``header_magic``."""
    with ParquetFile(str(empty_parquet)) as pf:
        out = pf.to_json(view="layout", depth="all")
    _assert_layout_invariants(out)
    region = next(
        c for c in out["children"] if c["_kind"] == "column_chunk_data_region"
    )
    # Region starts at the dictionary page (offset 4), never at byte 0.
    assert region["_offset"] >= 4


@pytest.mark.parametrize("depth", [1, "all"])
@pytest.mark.parametrize(
    "fixture_name", ["small_parquet", "indexed_parquet", "bloomy_parquet"]
)
def test_layout_invariants_across_fixtures(request, fixture_name, depth):
    """Offset-sorted / no-overlap / contiguous gap-fill must hold for
    every fixture at both a shallow and a full depth — not just
    ``small_parquet`` at depth 1."""
    path = request.getfixturevalue(fixture_name)
    with ParquetFile(str(path)) as pf:
        out = pf.to_json(view="layout", depth=depth)
    _assert_layout_invariants(out)


def test_offset_index_length_defaults_to_zero_when_absent(monkeypatch, indexed_parquet):
    """A writer that records ``offset_index_offset`` without
    ``offset_index_length`` must still yield an int ``_length`` (0), not
    ``None`` — otherwise the universal contract and ``_gap_fill``
    arithmetic break."""
    from parquet_analyzer import _tree_json

    with ParquetFile(str(indexed_parquet)) as pf:
        cc = next(
            c
            for rg in pf.row_groups
            for c in rg.columns
            if c.offset_index_offset is not None
        )
        monkeypatch.setattr(cc._t, "offset_index_length", None, raising=False)
        node = _tree_json._offset_index_node(cc)
        assert node["_length"] == 0
        assert isinstance(node["_length"], int)
