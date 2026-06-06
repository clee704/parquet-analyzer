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


def test_show_root_lists_row_groups_with_paths(indexed):
    with ParquetFile(str(indexed)) as pf:
        out = render(pf, "", walk_pages=False)
    assert out["_kind"] == "file"
    assert out["_navigation"] == {"path": "", "parent": None, "kind": "file"}
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
    assert out["pages"]["_walk_required"] is True
    assert out["dictionary_page"] is None
    assert counts["page"] == 0, "withheld listing must not read any page header"


def test_show_column_no_offset_index_walk_pages_lists(no_index):
    with ParquetFile(str(no_index)) as pf:
        out = render(pf, "row_groups/0/columns/0", walk_pages=True)
    assert isinstance(out["pages"], list)
    assert out["pages"][0]["_path"] == "row_groups/0/columns/0/pages/0"


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


def test_parent_path():
    assert _navigate._parent_path("") is None
    assert _navigate._parent_path("row_groups/0") == ""
    assert _navigate._parent_path("row_groups/0/columns/1") == "row_groups/0"


def test_annotate_children_guards():
    # Non-list and non-dict members are ignored without error.
    _navigate._annotate_children(None, "row_groups", "")
    items = [{"_kind": "x"}, "not-a-dict"]
    _navigate._annotate_children(items, "row_groups", "")
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
