"""Tests for the ``show`` navigation verb (:mod:`parquet_analyzer._navigate`)
and its CLI wiring."""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_analyzer import _core
from parquet_analyzer import _navigate
from parquet_analyzer._navigate import NavigationError, render, resolve
from parquet_analyzer._subcommands import run_subcommand
from parquet_analyzer.parquet_file import ParquetFile


@pytest.fixture()
def indexed(tmp_path):
    """Two row groups, dictionary + OffsetIndex (pages listable cheaply)."""
    p = tmp_path / "indexed.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(list(range(20)), pa.int64()),
                "name": pa.array([f"n{i}" for i in range(20)]),
            }
        ),
        p,
        row_group_size=10,
        use_dictionary=True,
        write_page_index=True,
    )
    return p


@pytest.fixture()
def no_index(tmp_path):
    """No OffsetIndex (pages require a walk)."""
    p = tmp_path / "nooi.parquet"
    pq.write_table(
        pa.table({"id": pa.array(list(range(20)), pa.int64())}),
        p,
        row_group_size=10,
        use_dictionary=False,
    )
    return p


@pytest.fixture()
def nested(tmp_path):
    p = tmp_path / "nested.parquet"
    addr = pa.array(
        [{"city": "x", "zip": 1}],
        type=pa.struct([("city", pa.string()), ("zip", pa.int64())]),
    )
    pq.write_table(
        pa.table({"addr": addr, "id": pa.array([1])}), p, write_page_index=True
    )
    return p


def _probe(monkeypatch):
    counts = {"page": 0}
    original = _core.read_thrift_segment

    def counting(f, offset, name, thrift_class):
        if name == "page":
            counts["page"] += 1
        return original(f, offset, name, thrift_class)

    monkeypatch.setattr(_core, "read_thrift_segment", counting)
    import parquet_analyzer.parquet_file as pfm

    monkeypatch.setattr(pfm, "read_thrift_segment", counting)
    return counts


# ---------------------------------------------------------------------------
# Spine traversal + path annotation
# ---------------------------------------------------------------------------


@pytest.fixture()
def empty_dict(tmp_path):
    """0-row column with a dictionary page (data_page_offset is 0, which
    breaks naive dict-extent arithmetic)."""
    p = tmp_path / "emptydict.parquet"
    pq.write_table(
        pa.table({"s": pa.array([], type=pa.string())}), p, use_dictionary=True
    )
    return p


def test_show_zero_row_dict_extent_not_negative(empty_dict):
    """Regression: a 0-row column's dict stub must report its true (positive)
    extent, not a negative length from `data_page_offset - dict_offset`."""
    with ParquetFile(str(empty_dict)) as pf:
        cc = pf.row_groups[0].columns[0]
        assert cc.dictionary_page_offset and cc._md.data_page_offset == 0
        out = render(pf, "row_groups/0/columns/0", walk_pages=False)
    dp = out["dictionary_page"]
    assert dp is not None
    assert dp["_location"]["length"] == cc._md.total_compressed_size
    assert dp["_location"]["length"] > 0


def test_show_file_not_found_uses_verb_only_fix(capsys):
    rc = run_subcommand(["show", "/no/such/file.parquet"])
    assert rc == 1
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "file_not_found"
    assert "show" in err["fix"] and "None" not in err["fix"]


def test_show_root_lists_row_groups_with_paths(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "", walk_pages=False)
    assert out["_kind"] == "file"
    nav = out["_navigation"]
    assert nav["path"] == "" and nav["parent"] is None and nav["kind"] == "file"
    paths = [rg["_path"] for rg in out["row_groups"]]
    assert paths == ["row_groups/0", "row_groups/1"]


def test_show_row_group_lists_columns_with_paths_and_names(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "row_groups/0", walk_pages=False)
    assert out["_kind"] == "row_group"
    assert out["_navigation"]["parent"] == ""
    cols = out["columns"]
    assert [c["_path"] for c in cols] == [
        "row_groups/0/columns/0",
        "row_groups/0/columns/1",
    ]
    assert [c["name"] for c in cols] == ["id", "name"]


def test_show_column_lists_pages_with_paths(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "row_groups/0/columns/0", walk_pages=False)
    assert out["_kind"] == "column_chunk"
    assert out["_navigation"]["parent"] == "row_groups/0"
    # dict page is index 0; data pages follow.
    assert out["dictionary_page"]["_path"] == "row_groups/0/columns/0/pages/0"
    assert out["pages"][0]["_path"] == "row_groups/0/columns/0/pages/1"


def test_show_page_is_leaf(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "row_groups/0/columns/0/pages/0", walk_pages=False)
    assert out["_kind"] == "dictionary_page"  # page 0 = dict
    assert out["_navigation"]["kind"] == "page"
    assert out["_navigation"]["parent"] == "row_groups/0/columns/0"


def test_show_nested_column_names(nested):
    with ParquetFile(str(nested)) as pf:
        out = render(pf, "row_groups/0", walk_pages=False)
    names = [c["name"] for c in out["columns"]]
    assert "addr.city" in names and "addr.zip" in names


def test_canonical_path_is_index_based(indexed):
    with ParquetFile(str(indexed)) as pf:
        _node, kind, canonical = resolve(pf, "row_groups/0/columns/1", walk_pages=False)
    assert kind == "column_chunk"
    assert canonical == "row_groups/0/columns/1"


# ---------------------------------------------------------------------------
# Bounded page listing (#30 / #36)
# ---------------------------------------------------------------------------


def test_show_column_no_offset_index_withholds_pages(no_index, monkeypatch):
    counts = _probe(monkeypatch)
    with ParquetFile(str(no_index)) as pf:
        out = render(pf, "row_groups/0/columns/0", walk_pages=False)
    # pages is cleanly null (never an object); the withheld affordance lives
    # in the _navigation listing block.
    assert out["pages"] is None
    nav = out["_navigation"]
    assert nav["walk_required"] is True
    assert nav["reason"] == "no OffsetIndex"
    assert "--walk-pages" in nav["hint"]
    assert nav["children_total"] is None
    assert nav["children_shown"] == 0
    assert nav["children_truncated"] is False
    assert out["dictionary_page"] is None
    assert counts["page"] == 0, "withheld listing must not read any page header"


def test_show_column_no_offset_index_walk_pages_lists(no_index):
    with ParquetFile(str(no_index)) as pf:
        out = render(pf, "row_groups/0/columns/0", walk_pages=True)
    assert isinstance(out["pages"], list)
    assert out["pages"][0]["_path"] == "row_groups/0/columns/0/pages/0"
    # Listed side of the withheld/listed asymmetry: no walk_required
    # affordance, and children_total is a concrete count (not null).
    nav = out["_navigation"]
    assert "walk_required" not in nav
    assert isinstance(nav["children_total"], int)


def test_show_column_offset_index_lists_without_walk(indexed, monkeypatch):
    counts = _probe(monkeypatch)
    with ParquetFile(str(indexed)) as pf:
        render(pf, "row_groups/0/columns/0", walk_pages=False)
    assert counts["page"] == 0, "OffsetIndex listing must not read page headers"


def test_page_address_requires_walk_without_offset_index(no_index):
    with ParquetFile(str(no_index)) as pf:
        with pytest.raises(NavigationError) as ei:
            resolve(pf, "row_groups/0/columns/0/pages/0", walk_pages=False)
    assert ei.value.code == "walk_required"
    assert "--walk-pages" in ei.value.fix


def test_page_address_works_with_walk_pages(no_index):
    with ParquetFile(str(no_index)) as pf:
        node, kind, canonical = resolve(
            pf, "row_groups/0/columns/0/pages/0", walk_pages=True
        )
    assert kind == "page"
    assert canonical == "row_groups/0/columns/0/pages/0"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "navpath,code",
    [
        ("row_groups/99", "row_group_out_of_range"),
        ("row_groups/0/columns/99", "column_out_of_range"),
        ("row_groups/0/foo/0", "invalid_path"),
        ("row_groups/x", "invalid_path"),
        ("row_groups/0/columns", "invalid_path"),
        ("columns/0", "invalid_path"),
    ],
)
def test_navigation_errors(indexed, navpath, code):
    with ParquetFile(str(indexed)) as pf:
        with pytest.raises(NavigationError) as ei:
            resolve(pf, navpath, walk_pages=False)
    assert ei.value.code == code
    assert ei.value.fix  # every error suggests a recovery command


def test_page_out_of_range(indexed):
    with ParquetFile(str(indexed)) as pf:
        with pytest.raises(NavigationError) as ei:
            resolve(pf, "row_groups/0/columns/0/pages/999", walk_pages=False)
    assert ei.value.code == "page_out_of_range"


@pytest.fixture()
def many_pages(tmp_path):
    """OffsetIndex column with many data pages (small page size + enough
    rows that pyarrow splits into well over a hundred pages)."""
    p = tmp_path / "manypages.parquet"
    pq.write_table(
        pa.table({"v": pa.array(list(range(200_000)), pa.int64())}),
        p,
        data_page_size=1024,
        use_dictionary=False,
        write_page_index=True,
    )
    return p


@pytest.fixture()
def indexed_dict_multipage(tmp_path):
    """OffsetIndex column with a dictionary page AND many data pages."""
    p = tmp_path / "idm.parquet"
    pq.write_table(
        pa.table({"s": pa.array([f"s{i % 50}" for i in range(100_000)])}),
        p,
        data_page_size=512,
        use_dictionary=True,
        write_page_index=True,
    )
    return p


@pytest.fixture()
def no_index_dict(tmp_path):
    """A dictionary page but NO OffsetIndex (withheld-listing path)."""
    p = tmp_path / "nid.parquet"
    pq.write_table(
        pa.table({"s": pa.array([f"s{i % 50}" for i in range(2000)])}),
        p,
        use_dictionary=True,
    )
    return p


def test_show_limit_excludes_dict_page_from_count(indexed_dict_multipage):
    """The dictionary page is always shown separately and is NOT subject to
    --limit, so children_shown counts only the (capped) data pages and never
    exceeds the limit (regression for the dict-page off-by-one)."""
    with ParquetFile(str(indexed_dict_multipage)) as pf:
        cc = pf.row_groups[0].columns[0]
        assert cc.dictionary_page_offset, "fixture should have a dictionary page"
        data_pages = cc.num_pages - 1  # minus the dict page
        out = render(pf, "row_groups/0/columns/0", walk_pages=False, limit=3)
    assert data_pages > 3, "fixture should have more than 3 data pages"
    assert out["dictionary_page"] is not None
    assert out["dictionary_page"]["_path"] == "row_groups/0/columns/0/pages/0"
    nav = out["_navigation"]
    assert nav["children_shown"] == 3  # <= limit, dict not counted
    assert nav["children_total"] == data_pages
    assert nav["children_truncated"] is True
    assert len(out["pages"]) == 3


def test_show_withheld_path_shows_dict_stub(no_index_dict):
    """A no-OffsetIndex column that has a dictionary page shows the dict stub
    (footer-derived extent) rather than a misleading null, while the data-page
    listing stays withheld."""
    with ParquetFile(str(no_index_dict)) as pf:
        cc = pf.row_groups[0].columns[0]
        assert not cc.has_offset_index and cc.dictionary_page_offset
        expected_len = cc.data_page_offset - cc.dictionary_page_offset
        out = render(pf, "row_groups/0/columns/0", walk_pages=False)
    dp = out["dictionary_page"]
    assert dp is not None
    assert dp["_kind"] == "dictionary_page"
    assert dp["_location"]["offset"] == cc.dictionary_page_offset
    assert dp["_location"]["length"] == expected_len
    assert out["pages"] is None
    assert out["_navigation"]["walk_required"] is True


def test_show_limit_caps_page_listing(many_pages):
    with ParquetFile(str(many_pages)) as pf:
        total = pf.row_groups[0].columns[0].num_pages
        out = render(pf, "row_groups/0/columns/0", walk_pages=False, limit=5)
    assert total > 5, "fixture should have more than 5 pages"
    assert len(out["pages"]) == 5
    nav = out["_navigation"]
    assert nav["children_total"] == total
    assert nav["children_shown"] == 5
    assert nav["children_truncated"] is True


def test_show_limit_caps_row_group_listing(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "", walk_pages=False, limit=1)
    assert len(out["row_groups"]) == 1
    nav = out["_navigation"]
    assert nav["children_total"] == 2
    assert nav["children_shown"] == 1
    assert nav["children_truncated"] is True


def test_show_limit_zero_lists_all(many_pages):
    with ParquetFile(str(many_pages)) as pf:
        total = pf.row_groups[0].columns[0].num_pages
        out = render(pf, "row_groups/0/columns/0", walk_pages=False, limit=0)
    assert len(out["pages"]) == total
    assert out["_navigation"]["children_truncated"] is False


def test_show_limit_does_not_block_addressing(many_pages):
    """Truncating the listing must not affect addressability — a page past
    the limit still resolves."""
    with ParquetFile(str(many_pages)) as pf:
        total = pf.row_groups[0].columns[0].num_pages
        node, kind, _ = resolve(
            pf, f"row_groups/0/columns/0/pages/{total - 1}", walk_pages=False
        )
    assert kind == "page"


def test_list_children_guards():
    # Missing / non-list child key → None, no error.
    assert _navigate._list_children({}, "row_groups", "", limit=0) is None
    out = {"row_groups": [{"_kind": "x"}, "not-a-dict"]}
    meta = _navigate._list_children(out, "row_groups", "", limit=0)
    assert out["row_groups"][0]["_path"] == "row_groups/0"
    assert meta["children_total"] == 2


def test_parent_path():
    assert _navigate._parent_path("") is None
    assert _navigate._parent_path("row_groups/0") == ""
    assert _navigate._parent_path("row_groups/0/columns/1") == "row_groups/0"


def test_annotate_children_guards():
    # Non-list and non-dict members are ignored without error.
    items = [{"_kind": "x"}, "not-a-dict"]
    _navigate._list_children({"row_groups": items}, "row_groups", "", limit=0)
    assert items[0]["_path"] == "row_groups/0"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_show_emits_schema(indexed, capsys):
    rc = run_subcommand(["show", str(indexed), "row_groups/0"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["$schema"] == "parquet-analyzer/v1/show"
    assert out["_kind"] == "row_group"


def test_cli_show_root(indexed, capsys):
    rc = run_subcommand(["show", str(indexed)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["_kind"] == "file"


def test_cli_show_error_maps_to_json(indexed, capsys):
    rc = run_subcommand(["show", str(indexed), "row_groups/99"])
    assert rc == 1
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "row_group_out_of_range"
    assert err["fix"]


def test_cli_show_schema_version(capsys):
    rc = run_subcommand(["show", "--schema-version"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["$schema"] == "parquet-analyzer/v1/show"


def test_cli_show_missing_path(capsys):
    rc = run_subcommand(["show"])
    assert rc == 1
    err = json.loads(capsys.readouterr().err)
    assert err["error"] == "missing_argument"
    assert err["message"].startswith("show:")


def test_cli_show_walk_pages_flag(no_index, capsys):
    rc = run_subcommand(
        ["show", str(no_index), "row_groups/0/columns/0", "--walk-pages"]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out["pages"], list)


def test_show_is_a_subcommand_verb():
    from parquet_analyzer._subcommands import is_subcommand_invocation

    assert is_subcommand_invocation(["show", "f.parquet"])
    assert is_subcommand_invocation(["show", "f.parquet", "row_groups/0"])
