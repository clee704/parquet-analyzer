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
def indexed_nodict_parquet(tmp_path):
    """File with an OffsetIndex but NO dictionary page (use_dictionary=False
    + write_page_index=True), so page_stubs() exercises the OffsetIndex
    fast path with no dictionary-page stub."""
    table = pa.table({"ints": pa.array(list(range(20)), type=pa.int32())})
    path = tmp_path / "indexed_nodict.parquet"
    pq.write_table(
        table,
        path,
        row_group_size=10,
        use_dictionary=False,
        write_page_index=True,
        compression="snappy",
    )
    return path


@pytest.fixture()
def bloomy_parquet(tmp_path):
    """File written with an explicit bloom filter for the ``strings``
    column, so the ``bloom_filter_header`` opaque-branch paths are live."""
    table = pa.table(
        {
            "strings": pa.array(["aa", "bb", "cc", "dd"]),
            "ints": pa.array([10, 20, 30, 40]),
        }
    )
    path = tmp_path / "bloomy.parquet"
    pq.write_table(
        table,
        path,
        compression="snappy",
        write_page_index=True,
        bloom_filter_options={"strings": True},
    )
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


@pytest.fixture()
def empty_nodict_parquet(tmp_path):
    """A 0-row file written WITHOUT dictionary encoding.

    Here ``dictionary_page_offset`` is ``None`` and ``data_page_offset``
    is the sentinel ``0`` with ``total_compressed_size == 0`` — the column
    has no on-disk page bytes at all. The ``column_chunk_data_region``
    must be omitted entirely rather than placed at offset 0.
    """
    table = pa.table(
        {
            "ints": pa.array([], type=pa.int32()),
            "strs": pa.array([], type=pa.string()),
        }
    )
    path = tmp_path / "empty_nodict.parquet"
    pq.write_table(table, path, use_dictionary=False)
    return path


@pytest.fixture()
def v2_parquet(tmp_path):
    """File written with V2 data pages so ``data_page_v2`` nodes and the
    ``_render_data_page_v2`` path are exercised."""
    table = pa.table(
        {
            "ints": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "strings": pa.array(["a", "b", "c", "d", "e"]),
        }
    )
    path = tmp_path / "v2.parquet"
    pq.write_table(
        table,
        path,
        data_page_version="2.0",
        use_dictionary=False,
        compression="snappy",
    )
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
    # Generic data-page kind used at the stub level (the v1/v2 version is a
    # materialized-only detail). Stub-only: never carries content fields.
    "data_page": {"_kind", "_offset", "_length"},
}


# Lower bound: content fields that MUST be present when a node of this kind
# is materialized (i.e. not a stub). Complements ALLOWED_MATERIALIZED_KEYS
# (the upper bound) so a renderer that silently drops a content field is
# caught. View-specific children are intentionally excluded (their presence
# is asserted by the dedicated tree/layout structural tests); these are the
# scalar content fields that must appear in both views.
#
# Kinds that legitimately materialize to system fields only are omitted:
# the opaque branches (offset_index / column_index / bloom_filter_header)
# and unknown are opaque in v0.
REQUIRED_MATERIALIZED_KEYS: dict[str, set[str]] = {
    "file": {"path"},
    "header_magic": {"_value"},
    "trailer_magic": {"_value"},
    "footer_length": {"_value"},
    "footer": {"version", "num_rows", "created_by"},
    "schema": {"_value"},
    "kv_metadata": {"_value"},
    "row_group": {"num_rows", "total_byte_size", "total_compressed_size", "ordinal"},
    "column_chunk": {
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
    },
    "column_chunk_data_region": {
        "chunk_ref",
        "row_group_index",
        "column_position_in_row_group",
    },
    "dictionary_page": {
        "page_type",
        "encoding",
        "num_values",
        "uncompressed_size",
        "compressed_size",
        "is_compressed",
        "crc",
    },
    "data_page_v1": {
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
}


LAZY_KINDS = {
    "dictionary_page",
    "data_page",
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
        # Lower bound: a materialized node must carry its required content
        # fields. A stub carries only system fields (+ _lazy), so skip those.
        content_keys = set(node.keys()) - {
            "_kind",
            "_offset",
            "_length",
            "_lazy",
            "$schema",
        }
        is_stub = not content_keys
        if not is_stub and kind in REQUIRED_MATERIALIZED_KEYS:
            missing = REQUIRED_MATERIALIZED_KEYS[kind] - set(node.keys())
            assert not missing, (
                f"materialized {kind} node missing required content fields {missing}"
            )
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
    """depth=all materializes offset_index / column_index → each one's
    OWN underlying thrift read must happen (a mis-wired _OPAQUE_READ_METHODS
    entry would leave the specific count at 0)."""
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
        assert counts.get("column_index", 0) > 0, (
            "depth=all must trigger column_index reads via _read_column_index"
        )
    finally:
        pf.close()


def test_depth_all_triggers_bloom_filter_read(bloomy_parquet, monkeypatch):
    """depth=all materializes bloom_filter_header → its own read must run
    (guards the _OPAQUE_READ_METHODS wiring for bloom_filter_header)."""
    pf = ParquetFile(str(bloomy_parquet))
    try:
        has_bf = any(
            cc.bloom_filter_offset is not None
            for rg in pf.row_groups
            for cc in rg.columns
        )
        if not has_bf:
            pytest.skip("fixture lacks a bloom filter — pyarrow version mismatch")
        counts = _install_read_probe(monkeypatch)
        pf.to_json(view="tree", depth="all")
        assert counts.get("bloom_filter_header", 0) > 0, (
            "depth=all must trigger bloom_filter_header reads via "
            "_read_bloom_filter_header"
        )
    finally:
        pf.close()


def test_offset_index_file_lists_pages_via_index_not_walk(indexed_parquet, monkeypatch):
    """#30: on a file with an OffsetIndex, listing a column's pages
    (column_chunk materialized, pages stubbed) must enumerate them via the
    OffsetIndex (cheap) and NOT by walking the per-page headers."""
    pf = ParquetFile(str(indexed_parquet))
    try:
        has_oi = any(cc.has_offset_index for rg in pf.row_groups for cc in rg.columns)
        if not has_oi:
            pytest.skip("fixture lacks offset_index")
        counts = _install_read_probe(monkeypatch)
        # depth=3 = file→rg(L1)→cc(L2 materialized)→pages(L3 stubs)
        pf.to_json(view="tree", depth=3)
        assert counts.get("page", 0) == 0, (
            f"listing pages walked {counts.get('page', 0)} page headers; the "
            "OffsetIndex fast path must read none"
        )
        assert counts.get("offset_index", 0) > 0, (
            "page listing must read the OffsetIndex to enumerate page stubs"
        )
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# Page stubs (#30) — list a column's pages without walking page headers
# ---------------------------------------------------------------------------


def test_page_stubs_no_dictionary_when_absent(indexed_nodict_parquet):
    """OffsetIndex present but no dictionary page: page_stubs() yields only
    data-page stubs (no dictionary_page stub), matching materialized pages."""
    with ParquetFile(str(indexed_nodict_parquet)) as pf:
        exercised = False
        for rg in pf.row_groups:
            for cc in rg.columns:
                stubs = cc.page_stubs()
                assert stubs is not None
                assert cc._md.dictionary_page_offset is None
                assert all(s.kind == "data_page" for s in stubs)
                pages = cc.pages()
                assert len(stubs) == len(pages)
                for stub, page in zip(stubs, pages):
                    assert stub.offset == page._offset
                    assert stub.length == page._length
                exercised = True
        assert exercised


def test_page_stubs_none_without_offset_index(small_parquet):
    """A column with no OffsetIndex cannot enumerate pages cheaply, so
    page_stubs() returns None (the caller must walk or show an affordance)."""
    with ParquetFile(str(small_parquet)) as pf:
        for rg in pf.row_groups:
            for cc in rg.columns:
                assert not cc.has_offset_index
                assert cc.page_stubs() is None


def test_dictionary_page_extent_normal_and_zero_row(tmp_path):
    """The dictionary-page extent is the gap to the first data page normally,
    and the whole compressed region for a 0-row column (where
    data_page_offset is 0, which would make the gap negative)."""
    # Normal column with data pages.
    normal = tmp_path / "normal.parquet"
    pq.write_table(
        pa.table({"v": pa.array(list(range(100)), pa.int64())}),
        normal,
        use_dictionary=True,
    )
    with ParquetFile(str(normal)) as pf:
        cc = pf.row_groups[0].columns[0]
        off, length = cc._dictionary_page_extent()
        assert off == cc._md.dictionary_page_offset
        assert length == cc._md.data_page_offset - off > 0

    # 0-row column: data_page_offset is 0 -> fall back to total_compressed_size.
    empty = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table({"v": pa.array([], type=pa.string())}), empty, use_dictionary=True
    )
    with ParquetFile(str(empty)) as pf:
        cc = pf.row_groups[0].columns[0]
        assert cc._md.data_page_offset == 0
        off, length = cc._dictionary_page_extent()
        assert length == cc._md.total_compressed_size > 0


def test_page_stubs_match_materialized_pages(indexed_parquet):
    """page_stubs() extents/kinds match the materialized pages() exactly —
    validating the dictionary-page arithmetic and the OffsetIndex-derived
    data-page extents against the real page headers."""
    with ParquetFile(str(indexed_parquet)) as pf:
        checked_dict = False
        checked_data = False
        for rg in pf.row_groups:
            for cc in rg.columns:
                stubs = cc.page_stubs()
                if stubs is None:
                    continue
                pages = cc.pages()
                assert len(stubs) == len(pages)
                for stub, page in zip(stubs, pages):
                    assert stub.offset == page._offset
                    assert stub.length == page._length
                    if page._kind == "dictionary_page":
                        assert stub.kind == "dictionary_page"
                        assert stub.first_row_index is None
                        checked_dict = True
                    else:
                        # Stub kind is the generic data_page; the version
                        # (v1/v2) is only on the materialized page.
                        assert stub.kind == "data_page"
                        assert stub.first_row_index is not None
                        checked_data = True
        assert checked_data, "fixture exercised no data-page stubs"
        assert checked_dict, "fixture exercised no dictionary-page stub"


def test_page_stubs_dict_extent_matches_offset_index(indexed_parquet):
    """The dictionary-page stub's data-page boundary equals the first data
    page's offset from the OffsetIndex (the arithmetic extent is exact)."""
    with ParquetFile(str(indexed_parquet)) as pf:
        for rg in pf.row_groups:
            for cc in rg.columns:
                stubs = cc.page_stubs()
                if stubs is None or stubs[0].kind != "dictionary_page":
                    continue
                first_data = next(s for s in stubs if s.kind == "data_page")
                dict_stub = stubs[0]
                assert dict_stub.offset + dict_stub.length == first_data.offset


def test_page_stubs_no_per_page_header_reads(indexed_parquet, monkeypatch):
    """Building page_stubs() reads the OffsetIndex but NOT any per-page
    header — the cost is independent of the page count."""
    pf = ParquetFile(str(indexed_parquet))
    try:
        counts = _install_read_probe(monkeypatch)
        for rg in pf.row_groups:
            for cc in rg.columns:
                cc.page_stubs()
        assert counts.get("page", 0) == 0, (
            f"page_stubs() read {counts.get('page', 0)} page headers; must read none"
        )
    finally:
        pf.close()


def test_stub_data_pages_use_generic_kind_offset_index(indexed_parquet):
    """At the stub level, data pages carry the generic ``data_page`` kind
    (OffsetIndex fast path)."""
    with ParquetFile(str(indexed_parquet)) as pf:
        out = pf.to_json(view="tree", depth=3)
    saw_data_page = False
    for rg in out["row_groups"]:
        for cc in rg["columns"]:
            for page in cc["pages"]:
                assert page["_kind"] == "data_page", page
                saw_data_page = True
            if cc["dictionary_page"] is not None:
                assert cc["dictionary_page"]["_kind"] == "dictionary_page"
    assert saw_data_page


def test_stub_data_pages_use_generic_kind_walk_fallback(small_parquet):
    """Stub-level data pages are generic ``data_page`` even on the
    no-OffsetIndex walk fallback — the contract is uniform across paths."""
    with ParquetFile(str(small_parquet)) as pf:
        out = pf.to_json(view="tree", depth=3)
    saw_data_page = False
    for rg in out["row_groups"]:
        for cc in rg["columns"]:
            for page in cc["pages"]:
                assert page["_kind"] == "data_page", page
                saw_data_page = True
    assert saw_data_page


def test_materialized_pages_keep_specific_version_kind(v2_parquet):
    """The v1/v2 distinction survives on MATERIALIZED page nodes — only the
    stub level is generic."""
    with ParquetFile(str(v2_parquet)) as pf:
        out = pf.to_json(view="tree", depth="all")
    kinds = {
        page["_kind"]
        for rg in out["row_groups"]
        for cc in rg["columns"]
        for page in cc["pages"]
    }
    assert "data_page_v2" in kinds, kinds
    assert "data_page" not in kinds, "materialized pages must not use the generic kind"


def test_layout_data_region_pages_contiguous_with_offset_index(indexed_parquet):
    """The OffsetIndex-derived page stubs tile the data region exactly
    (dictionary page + data pages, offset-contiguous) — validating the
    cheap-path extents against the physical layout."""
    with ParquetFile(str(indexed_parquet)) as pf:
        has_oi = any(cc.has_offset_index for rg in pf.row_groups for cc in rg.columns)
        if not has_oi:
            pytest.skip("fixture lacks offset_index")
        out = pf.to_json(view="layout", depth=2)
    checked = False
    for region in out["children"]:
        if region["_kind"] != "column_chunk_data_region":
            continue
        page_nodes = []
        if region["dictionary_page"] is not None:
            page_nodes.append(region["dictionary_page"])
        page_nodes.extend(region["pages"])
        page_nodes.sort(key=lambda p: p["_offset"])
        for a, b in zip(page_nodes, page_nodes[1:]):
            assert a["_offset"] + a["_length"] == b["_offset"], (
                f"non-contiguous pages in data region: {a} then {b}"
            )
            checked = True
    assert checked, "no multi-page data region exercised"


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
    """A 0-row dictionary column has a real (readable) dictionary page on
    disk; materializing it must recover its header content, not emit a
    stub-shaped node."""
    with ParquetFile(str(empty_parquet)) as pf:
        out = pf.to_json()  # default view="tree", depth="all"
    _assert_universal_contract(out, view="tree")
    cc = out["row_groups"][0]["columns"][0]
    dp = cc["dictionary_page"]
    assert dp is not None
    assert dp["_kind"] == "dictionary_page"
    # The dictionary page header is readable even for a 0-row column, so
    # the materialized node carries real content (not system-fields-only).
    assert dp["page_type"] == "DICTIONARY_PAGE"
    assert dp["num_values"] == 0
    assert "encoding" in dp
    # The dictionary page has real on-disk bytes; its length reflects them.
    assert dp["_length"] > 0


def test_empty_file_layout_data_region_tiled_by_dictionary_page(empty_parquet):
    """A 0-row dictionary column's data region is covered by its synthetic
    dictionary-page child: pages() is empty, but the dictionary page's
    bytes are still accounted for, so the region does not span bytes with
    no child."""
    with ParquetFile(str(empty_parquet)) as pf:
        out = pf.to_json(view="layout", depth="all")
    region = next(
        c for c in out["children"] if c["_kind"] == "column_chunk_data_region"
    )
    dp = region["dictionary_page"]
    assert dp is not None and dp["_kind"] == "dictionary_page"
    # The dictionary page exactly tiles the data region (no uncovered bytes).
    assert dp["_offset"] == region["_offset"]
    assert dp["_offset"] + dp["_length"] == region["_offset"] + region["_length"]


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
    "fixture_name",
    ["small_parquet", "indexed_parquet", "bloomy_parquet", "v2_parquet"],
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


def test_empty_nodict_file_omits_data_region(empty_nodict_parquet):
    """A 0-row column with no dictionary and no data page (total
    compressed size 0) must not produce a zero-length
    ``column_chunk_data_region`` at offset 0 overlapping ``header_magic``;
    the region is omitted entirely."""
    with ParquetFile(str(empty_nodict_parquet)) as pf:
        layout = pf.to_json(view="layout", depth="all")
        tree = pf.to_json(view="tree", depth="all")
    _assert_layout_invariants(layout)
    assert not any(
        c["_kind"] == "column_chunk_data_region" for c in layout["children"]
    ), "empty column should contribute no data region"
    # Each column_chunk (rendered inside footer in layout view) has a null
    # data_region_ref.
    ccs = [n for n in _iter_nodes(layout) if n.get("_kind") == "column_chunk"]
    assert ccs, "expected column_chunk nodes inside footer"
    assert all(cc["data_region_ref"] is None for cc in ccs)
    # Tree view still serializes without crashing.
    _assert_universal_contract(tree, view="tree")


def test_bloom_filter_header_present_in_both_views(bloomy_parquet):
    """The bloom-filter fixture actually writes a bloom filter, so the
    ``bloom_filter_header`` node appears in both views."""
    with ParquetFile(str(bloomy_parquet)) as pf:
        strings_cc = next(
            cc for cc in pf.row_groups[0].columns if cc.path[-1] == "strings"
        )
        assert strings_cc.bloom_filter_offset is not None, (
            "fixture must write a real bloom filter"
        )
        tree = pf.to_json(view="tree", depth="all")
        layout = pf.to_json(view="layout", depth="all")
    strings_node = next(
        cc for cc in tree["row_groups"][0]["columns"] if cc["path"][-1] == "strings"
    )
    bf = strings_node["bloom_filter"]
    assert bf is not None and bf["_kind"] == "bloom_filter_header"
    assert "bloom_filter_header" in {c["_kind"] for c in layout["children"]}


def test_bloom_filter_header_lazy_when_stubbed(bloomy_parquet):
    """The ``bloom_filter_header`` opaque branch carries ``_lazy: true``
    when stubbed and drops it when materialized."""
    with ParquetFile(str(bloomy_parquet)) as pf:
        shallow = pf.to_json(view="tree", depth=3)
        deep = pf.to_json(view="tree", depth="all")
    stub = next(
        n for n in _iter_nodes(shallow) if n.get("_kind") == "bloom_filter_header"
    )
    assert stub.get("_lazy") is True
    mat = next(n for n in _iter_nodes(deep) if n.get("_kind") == "bloom_filter_header")
    assert "_lazy" not in mat


def test_data_page_v2_contract_and_fields(v2_parquet):
    """A V2-page file exercises ``_render_data_page_v2`` (V2-only fields)
    and the ``DATA_PAGE_V2`` ``Page._kind`` branch."""
    with ParquetFile(str(v2_parquet)) as pf:
        tree = pf.to_json(view="tree", depth="all")
        layout = pf.to_json(view="layout", depth="all")
    _assert_universal_contract(tree, view="tree")
    _assert_universal_contract(layout, view="layout")
    v2_nodes = [n for n in _iter_nodes(tree) if n.get("_kind") == "data_page_v2"]
    assert v2_nodes, "expected data_page_v2 nodes in a V2 file"
    node = v2_nodes[0]
    for key in (
        "page_type",
        "num_nulls",
        "num_rows",
        "is_compressed",
        "definition_levels_byte_length",
        "repetition_levels_byte_length",
    ):
        assert key in node, f"data_page_v2 missing V2 field {key!r}"
    assert node["page_type"] == "DATA_PAGE_V2"


def test_gap_fill_emits_unknown_nodes_for_gaps():
    """``_gap_fill`` fills leading, interior, and trailing gaps with
    ``unknown`` nodes, and ``_render_unknown`` materializes them to system
    fields only. Real pyarrow files have no gaps, so this is unit-tested
    directly on synthetic extents."""
    from parquet_analyzer import _layout, _tree_json

    items = [
        {"_kind": "footer", "_offset": 10, "_length": 20},
        {"_kind": "footer_length", "_offset": 50, "_length": 10},
    ]
    filled = _layout._gap_fill(items, file_size=100)
    assert [(n["_kind"], n["_offset"], n["_length"]) for n in filled] == [
        ("unknown", 0, 10),
        ("footer", 10, 20),
        ("unknown", 30, 20),
        ("footer_length", 50, 10),
        ("unknown", 60, 40),
    ]
    rendered = _tree_json._render(
        {"_kind": "unknown", "_offset": 0, "_length": 10}, "layout", "all"
    )
    assert rendered == {"_kind": "unknown", "_offset": 0, "_length": 10}


# ---------------------------------------------------------------------------
# Malformed-footer guards (footer-segment / thrift inconsistency)
# ---------------------------------------------------------------------------


def test_row_group_extent_count_mismatch_raises():
    """A footer segment whose row-group element count disagrees with the
    parsed thrift is an internal inconsistency and must fail loudly, not
    fabricate placeholder extents."""
    from parquet_analyzer.parquet_file import _extract_row_group_extents

    # Segment has no row_groups field (-> 0 elements) but thrift has 2.
    with pytest.raises(ValueError, match="row-group count mismatch"):
        _extract_row_group_extents({"value": []}, [object(), object()])


def test_column_chunk_extent_missing_row_group_raises():
    """Requesting column extents for a row-group index the footer segment
    doesn't contain must raise rather than fabricate placeholders."""
    from parquet_analyzer.parquet_file import _extract_column_chunk_extents

    with pytest.raises(ValueError, match="missing extents for row group 0"):
        _extract_column_chunk_extents({"value": []}, 0, [object()])


def test_column_chunk_extent_count_mismatch_raises():
    """A row-group segment whose column count disagrees with the thrift
    must raise."""
    from parquet_analyzer.parquet_file import _extract_column_chunk_extents

    footer_segment = {
        "value": [
            {
                "name": "row_groups",
                "value": [{"name": "rg", "value": [{"name": "columns", "value": []}]}],
            }
        ]
    }
    # Segment row group 0 has 0 columns, thrift has 1.
    with pytest.raises(ValueError, match="column-chunk count mismatch"):
        _extract_column_chunk_extents(footer_segment, 0, [object()])


def test_schema_node_missing_schema_raises():
    """Schema is mandatory per the parquet spec; a footer segment lacking
    it must fail loudly rather than emit a fabricated zero-length node."""
    import types

    from parquet_analyzer import _tree_json

    fake_pf = types.SimpleNamespace(_footer_segment={"value": []})
    with pytest.raises(ValueError, match="missing mandatory schema"):
        _tree_json._schema_node(fake_pf)


def test_gap_fill_raises_on_overlap():
    """_gap_fill enforces the no-overlap invariant rather than silently
    emitting overlapping siblings."""
    from parquet_analyzer import _layout

    items = [
        {"_kind": "footer", "_offset": 10, "_length": 20},  # [10, 30)
        {"_kind": "footer_length", "_offset": 25, "_length": 10},  # overlaps
    ]
    with pytest.raises(ValueError, match="overlapping layout nodes"):
        _layout._gap_fill(items, file_size=100)


def test_gap_fill_raises_on_out_of_bounds():
    """_gap_fill rejects a node extending past file_size."""
    from parquet_analyzer import _layout

    items = [{"_kind": "footer", "_offset": 90, "_length": 20}]  # ends at 110
    with pytest.raises(ValueError, match="past file_size"):
        _layout._gap_fill(items, file_size=100)


def test_layout_data_region_real_dictionary_page_tiles(indexed_parquet):
    """A non-empty dictionary column's data region is covered by a real
    (non-null) dictionary page followed by data pages that tile the region
    exactly."""
    with ParquetFile(str(indexed_parquet)) as pf:
        # Find a column chunk that actually has a dictionary page.
        target = None
        for rg_idx, rg in enumerate(pf.row_groups):
            for cc in rg.columns:
                if cc.dictionary_page_offset:
                    target = (rg_idx, cc.dictionary_page_offset)
                    break
            if target:
                break
        assert target is not None, "fixture should have a dictionary-encoded column"
        out = pf.to_json(view="layout", depth="all")
    region = next(
        c
        for c in out["children"]
        if c["_kind"] == "column_chunk_data_region" and c["_offset"] == target[1]
    )
    dp = region["dictionary_page"]
    assert dp is not None and dp["_kind"] == "dictionary_page"
    assert dp["_offset"] == target[1]
    # dict page + data pages, offset-sorted, tile the region exactly.
    children = [dp] + region["pages"]
    children.sort(key=lambda c: c["_offset"])
    assert children[0]["_offset"] == region["_offset"]
    assert (
        children[-1]["_offset"] + children[-1]["_length"]
        == region["_offset"] + region["_length"]
    )
    for a, b in zip(children, children[1:]):
        assert b["_offset"] == a["_offset"] + a["_length"], "data region not tiled"


def test_pages_reads_dictionary_via_data_page_offset(indexed_parquet):
    """Older writers point data_page_offset at the dictionary page and
    leave dictionary_page_offset unset. The page walk must read the
    dictionary page and continue to the data pages, not stop at it."""
    with ParquetFile(str(indexed_parquet)) as pf:
        cc = next(
            c for rg in pf.row_groups for c in rg.columns if c.dictionary_page_offset
        )
        # Baseline: a normal pyarrow chunk has a dict page + data page(s).
        baseline = [p._kind for p in cc.pages()]
        assert "dictionary_page" in baseline
        assert any(k.startswith("data_page") for k in baseline)
        # Simulate the older-writer layout and re-walk.
        cc._md.data_page_offset = cc.dictionary_page_offset
        cc._md.dictionary_page_offset = None
        cc._pages_cache = None
        kinds = [p._kind for p in cc.pages()]
    assert kinds[0] == "dictionary_page", "dictionary page should be read first"
    assert any(k.startswith("data_page") for k in kinds), (
        "data pages must not be dropped when the walk starts at the dict page"
    )
    assert len(kinds) == len(baseline)


def test_column_chunk_statistics_shape():
    """column_chunk statistics carry null_count and DECODED scalar
    min_value/max_value — not raw thrift byte objects — and no longer
    include the deprecated min/max byte fields (#31)."""
    with ParquetFile(str(TITANIC)) as pf:
        out = pf.to_json(view="tree", depth="all")
    cc = out["row_groups"][0]["columns"][0]
    stats = cc["statistics"]
    assert isinstance(stats, dict)
    assert "null_count" in stats and isinstance(stats["null_count"], int)
    assert "min_value" in stats and "max_value" in stats
    # Decoded scalars, not the raw {"type":"binary","value":[...]} thrift form.
    for key in ("min_value", "max_value"):
        assert not isinstance(stats[key], dict), f"{key} should be a decoded scalar"
    # Deprecated min/max byte fields are dropped.
    assert "min" not in stats and "max" not in stats


def test_statistics_decoded_by_type():
    """min_value/max_value decode to typed scalars per the column's
    physical/logical type: ints, floats, UTF-8 strings, DECIMAL as a
    lossless string, and non-text binary as a hex string."""
    import decimal as _decimal

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "s": pa.array(["female", "male", "female"]),
            "n": pa.array([1, 2, 3], type=pa.int32()),
            "f": pa.array([1.5, 2.5, 3.5]),
            "dec": pa.array(
                [
                    _decimal.Decimal("1.23"),
                    _decimal.Decimal("4.56"),
                    _decimal.Decimal("0.01"),
                ],
                type=pa.decimal128(5, 2),
            ),
            "b": pa.array([b"\x00\xff", b"\x01\x02", b"\xab\xcd"], type=pa.binary()),
        }
    )
    import tempfile
    import os

    path = os.path.join(tempfile.mkdtemp(), "typed.parquet")
    pq.write_table(table, path, write_statistics=True)
    with ParquetFile(path) as pf:
        out = pf.to_json(view="tree", depth="all")
    by_name = {
        ".".join(c["path"]): c["statistics"] for c in out["row_groups"][0]["columns"]
    }
    assert by_name["s"]["min_value"] == "female" and by_name["s"]["max_value"] == "male"
    assert by_name["n"]["min_value"] == 1 and by_name["n"]["max_value"] == 3
    assert by_name["f"]["min_value"] == 1.5 and by_name["f"]["max_value"] == 3.5
    assert (
        by_name["dec"]["min_value"] == "0.01" and by_name["dec"]["max_value"] == "4.56"
    )
    # non-UTF-8 binary -> hex string
    assert all(isinstance(by_name["b"][k], str) for k in ("min_value", "max_value"))
    int(by_name["b"]["min_value"], 16)  # parses as hex


def test_statistics_fallback_to_deprecated_min_max():
    """Older writers set only the deprecated min/max byte fields (not
    min_value/max_value). _build_statistics decodes from them as a fallback
    while still emitting the modern min_value/max_value keys."""
    from types import SimpleNamespace

    from parquet_analyzer import _tree_json

    stats = SimpleNamespace(
        null_count=0,
        distinct_count=None,
        min_value=None,
        max_value=None,
        min=(5).to_bytes(4, "little", signed=True),
        max=(9).to_bytes(4, "little", signed=True),
    )
    out = _tree_json._build_statistics(stats, "INT32", {})
    assert out == {"null_count": 0, "min_value": 5, "max_value": 9}
    assert "min" not in out and "max" not in out
