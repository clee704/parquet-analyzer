import importlib
import json
from pathlib import Path

import pytest

from parquet_analyzer import ParquetFile, cli


@pytest.fixture()
def sample_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pa.table(
        {
            "ints": pa.array([1, 2, 3], type=pa.int32()),
            "floats": pa.array([1.0, 2.5, 3.25]),
        }
    )

    path = tmp_path / "sample.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture()
def sample_parquet_with_page_index(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    dict_array = pa.array(
        ["alpha", "beta", "gamma", "beta", "alpha"],
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    table = pa.table(
        {
            "dict_col": dict_array,
            "floats": pa.array([1.0, 2.5, 3.25, 4.5, 5.75]),
        }
    )

    path = tmp_path / "with-index.parquet"
    pq.write_table(
        table,
        path,
        row_group_size=2,
        use_dictionary=True,
        write_page_index=True,
        data_page_version="2.0",
    )
    return path


def test_parquet_file_smoke(sample_parquet):
    pf = ParquetFile(str(sample_parquet))
    try:
        assert pf.num_rows == 3
        assert pf.num_row_groups == 1
        assert pf.num_columns == 2
        assert pf.footer_size > 0

        # full_summary triggers eager walk
        summary = pf.full_summary
        assert summary["num_data_pages"] >= 1

        pages = pf.all_pages()
        assert pages
        first_column = pages[0]
        assert first_column["row_groups"], "Row group data should be present"
        first_row_group = first_column["row_groups"][0]
        assert first_row_group["pages"], "Data pages should be listed"
    finally:
        pf.close()


def test_parquet_file_with_page_indexes(sample_parquet_with_page_index):
    pf = ParquetFile(str(sample_parquet_with_page_index))
    try:
        # column_offset_map triggers eager walk; verify per-column structure
        column = next(iter(pf.column_offset_map.values()))[0]
        assert "column_index" in column
        assert "offset_index" in column

        pages = pf.all_pages()
        row_group = pages[0]["row_groups"][0]
        assert "column_index" in row_group
        assert "offset_index" in row_group

        summary = pf.full_summary
        assert summary["num_dict_pages"] >= 1
        assert summary["column_index_size"] > 0
        assert summary["offset_index_size"] > 0
    finally:
        pf.close()


def test_parquet_file_duckdb_fixture():
    file_path = Path(__file__).parent / "data" / "titanic.parquet"
    pf = ParquetFile(str(file_path))
    try:
        # Footer-only assertions — no eager walk needed
        assert pf.num_rows == 891
        assert pf.num_columns == 12

        summary = pf.full_summary
        assert summary["num_dict_pages"] == 2
        assert summary["num_data_pages"] == 12

        pages = pf.all_pages()
        sex_column = next(col for col in pages if col["column"] == ("Sex",))
        sex_pages = sex_column["row_groups"][0]["pages"]
        assert sex_pages[0]["type"] == "DICTIONARY_PAGE"
        assert sex_pages[1]["type"] == "DATA_PAGE"
        assert sex_pages[0]["$offset"] == pf.column_offset_map[("Sex",)][0]["pages"][0]

        embarked_column = next(col for col in pages if col["column"] == ("Embarked",))
        embarked_pages = embarked_column["row_groups"][0]["pages"]
        assert embarked_pages[0]["type"] == "DICTIONARY_PAGE"
        assert (
            embarked_pages[0]["$offset"]
            == pf.column_offset_map[("Embarked",)][0]["pages"][0]
        )
    finally:
        pf.close()


def test_cli_with_real_file(sample_parquet, capsys):
    cli.main([str(sample_parquet)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["summary"]["num_rows"] == 3
    assert payload["pages"]


def test_main_module_invokes_cli(monkeypatch):
    invoked = {}

    def fake_main(argv=None):
        invoked["called"] = argv

    monkeypatch.setattr("parquet_analyzer.cli.main", fake_main)

    module = importlib.import_module("parquet_analyzer.__main__")
    module = importlib.reload(module)

    module.main()

    assert invoked == {"called": None}


# ---------------------------------------------------------------------------
# End-to-end subcommand tests against real pyarrow fixtures
# ---------------------------------------------------------------------------


def test_subcommand_file_summary_end_to_end(sample_parquet, capsys):
    cli.main(["file", "summary", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-summary"
    assert payload["num_rows"] == 3
    assert payload["num_columns"] == 2


def test_subcommand_file_schema_end_to_end(sample_parquet, capsys):
    cli.main(["file", "schema", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-schema"
    names = [e.get("name") for e in payload["elements"]]
    assert "ints" in names
    assert "floats" in names


def test_subcommand_file_kv_end_to_end(sample_parquet, capsys):
    cli.main(["file", "kv", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["$schema"] == "parquet-analyzer/v1/file-kv"
    # pyarrow writes ARROW:schema in kv metadata; just verify the shape.
    assert payload["truncated"] is False
    assert isinstance(payload["items"], list)
    for item in payload["items"]:
        assert set(item.keys()) == {"key", "value"}


def test_subcommand_file_validate_end_to_end(sample_parquet, capsys):
    cli.main(["file", "validate", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_subcommand_rowgroup_list_end_to_end(sample_parquet, capsys):
    cli.main(["rowgroup", "list", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 1
    assert payload["items"][0]["num_rows"] == 3
    assert payload["items"][0]["num_columns"] == 2


def test_subcommand_rowgroup_show_end_to_end(sample_parquet, capsys):
    cli.main(["rowgroup", "show", str(sample_parquet), "--row-group", "0"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["row_group"] == 0
    assert payload["num_columns"] == 2
    assert [c["column"] for c in payload["columns"]] == ["ints", "floats"]


def test_subcommand_column_list_end_to_end(sample_parquet, capsys):
    cli.main(["column", "list", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 2  # 1 row group × 2 columns
    cols = {(i["row_group"], i["column"]) for i in payload["items"]}
    assert cols == {(0, "ints"), (0, "floats")}


def test_subcommand_column_show_end_to_end(sample_parquet, capsys):
    cli.main(["column", "show", str(sample_parquet), "--column", "ints"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["column"] == "ints"
    assert payload["type"] == "INT32"
    assert len(payload["row_groups"]) == 1


@pytest.fixture()
def multi_rg_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "id": pa.array(list(range(40))),
            "name": pa.array([f"n{i}" for i in range(40)]),
        }
    )
    path = tmp_path / "multi.parquet"
    pq.write_table(table, path, row_group_size=10)  # 4 row groups × 2 columns
    return path


def _resolve_kind_name(path, navpath):
    """Resolve a show navigation path and return (kind, column display name)."""
    from parquet_analyzer import _navigate

    with ParquetFile(str(path)) as pf:
        node, kind, _ = _navigate.resolve(pf, navpath, walk_pages=False)
        name = ".".join(node.path) if kind == "column_chunk" else None
    return kind, name


def test_rowgroup_list_items_carry_show_path(multi_rg_parquet, capsys):
    cli.main(["rowgroup", "list", str(multi_rg_parquet)])
    items = json.loads(capsys.readouterr().out)["items"]
    assert [i["_path"] for i in items] == [f"row_groups/{i}" for i in range(4)]


def test_column_list_items_carry_resolvable_show_path(multi_rg_parquet, capsys):
    cli.main(["column", "list", str(multi_rg_parquet)])
    items = json.loads(capsys.readouterr().out)["items"]
    # Every item's _path resolves via show to the very column it describes.
    for item in items:
        kind, name = _resolve_kind_name(multi_rg_parquet, item["_path"])
        assert kind == "column_chunk"
        assert name == item["column"]
    # Index-based path for row group 2, column 1 (the "name" column).
    assert {"row_groups/2/columns/1"} <= {i["_path"] for i in items}


def test_rowgroup_show_carries_show_paths(multi_rg_parquet, capsys):
    cli.main(["rowgroup", "show", str(multi_rg_parquet), "--row-group", "2"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["_path"] == "row_groups/2"
    assert [c["_path"] for c in payload["columns"]] == [
        "row_groups/2/columns/0",
        "row_groups/2/columns/1",
    ]


def test_column_show_entries_carry_show_paths(multi_rg_parquet, capsys):
    cli.main(["column", "show", str(multi_rg_parquet), "--column", "name"])
    payload = json.loads(capsys.readouterr().out)
    # One entry per row group; "name" is column index 1 in each.
    assert [e["_path"] for e in payload["row_groups"]] == [
        f"row_groups/{i}/columns/1" for i in range(4)
    ]


def test_subcommand_column_show_no_offset_index_marks_pages_unknown(
    sample_parquet, capsys
):
    """pyarrow default writes no page index — `num_pages` must NOT be reported
    by default.

    This guards against accidental page-header walks (the footer-bounded
    default forbids them; the explicit `--walk-pages` opt-in is what allows
    them) by asserting the behavior on a real file that lacks an OffsetIndex.
    """
    cli.main(["column", "show", str(sample_parquet), "--column", "ints"])
    payload = json.loads(capsys.readouterr().out)
    rg = payload["row_groups"][0]
    assert rg["has_offset_index"] is False
    assert rg["num_pages"] is None
    assert rg["num_pages_known"] is False
    # The output is self-describing: it points at the --walk-pages opt-in.
    assert "--walk-pages" in rg["num_pages_hint"]


def test_subcommand_column_show_with_offset_index_reports_pages(
    sample_parquet_with_page_index, capsys
):
    cli.main(
        ["column", "show", str(sample_parquet_with_page_index), "--column", "dict_col"]
    )
    payload = json.loads(capsys.readouterr().out)
    for rg in payload["row_groups"]:
        assert rg["has_offset_index"] is True
        assert rg["num_pages_known"] is True
        assert rg["num_pages"] >= 1
        assert rg["num_pages_hint"] is None  # count known → no hint


def test_subcommand_column_show_walk_pages_counts_without_offset_index(
    sample_parquet, capsys
):
    """`--walk-pages` opts into a per-chunk page-header walk, so `num_pages` is
    reported even on a file with no OffsetIndex. The counted value is
    cross-checked against `page list` (an independent code path) rather than a
    bare ``>= 1``."""
    cli.main(
        ["column", "show", str(sample_parquet), "--column", "ints", "--walk-pages"]
    )
    payload = json.loads(capsys.readouterr().out)
    rg = payload["row_groups"][0]
    assert rg["has_offset_index"] is False
    assert rg["num_pages_known"] is True
    assert rg["num_pages_hint"] is None  # count now known → no hint

    # Ground truth: `page list` for the same chunk walks the pages independently.
    cli.main(["page", "list", str(sample_parquet), "--column", "ints"])
    pages_total = json.loads(capsys.readouterr().out)["total"]
    assert rg["num_pages"] == pages_total
    assert rg["num_pages"] >= 1


def test_subcommand_column_show_walk_pages_only_mutates_num_pages(
    sample_parquet, capsys
):
    """Regression guard: `--walk-pages` changes ONLY `num_pages` /
    `num_pages_known` — every other per-chunk field is identical to the
    footer-only default output."""
    cli.main(["column", "show", str(sample_parquet), "--column", "ints"])
    default = json.loads(capsys.readouterr().out)["row_groups"][0]
    cli.main(
        ["column", "show", str(sample_parquet), "--column", "ints", "--walk-pages"]
    )
    walked = json.loads(capsys.readouterr().out)["row_groups"][0]

    assert (default["num_pages"], default["num_pages_known"]) == (None, False)
    assert walked["num_pages_known"] is True and walked["num_pages"] >= 1
    # Everything except the page-count fields must be byte-identical.
    mutable = {"num_pages", "num_pages_known", "num_pages_hint"}
    assert {k: v for k, v in default.items() if k not in mutable} == {
        k: v for k, v in walked.items() if k not in mutable
    }


def test_subcommand_column_list_walk_pages_counts_without_offset_index(
    sample_parquet, capsys
):
    cli.main(["column", "list", str(sample_parquet), "--walk-pages"])
    items = json.loads(capsys.readouterr().out)["items"]
    assert items
    for item in items:
        assert item["has_offset_index"] is False
        assert item["num_pages_known"] is True
        assert item["num_pages"] >= 1


def test_subcommand_column_list_default_does_not_walk(sample_parquet, capsys):
    """Without the flag, the default stays footer-only: num_pages unknown, and
    every item carries the --walk-pages hint."""
    cli.main(["column", "list", str(sample_parquet)])
    items = json.loads(capsys.readouterr().out)["items"]
    assert items
    assert all(item["num_pages_known"] is False for item in items)
    assert all(item["num_pages"] is None for item in items)
    assert all("--walk-pages" in item["num_pages_hint"] for item in items)


def test_subcommand_rowgroup_show_walk_pages_counts_without_offset_index(
    sample_parquet, capsys
):
    cli.main(
        ["rowgroup", "show", str(sample_parquet), "--row-group", "0", "--walk-pages"]
    )
    columns = json.loads(capsys.readouterr().out)["columns"]
    assert columns
    for col in columns:
        assert col["num_pages_known"] is True
        assert col["num_pages"] >= 1
        assert col["num_pages_hint"] is None


def test_subcommand_rowgroup_show_default_does_not_walk(sample_parquet, capsys):
    cli.main(["rowgroup", "show", str(sample_parquet), "--row-group", "0"])
    columns = json.loads(capsys.readouterr().out)["columns"]
    assert columns
    assert all(col["num_pages_known"] is False for col in columns)
    assert all(col["num_pages"] is None for col in columns)
    assert all("--walk-pages" in col["num_pages_hint"] for col in columns)


def test_subcommand_column_show_walk_pages_harmless_with_offset_index(
    sample_parquet_with_page_index, capsys
):
    """On a file that already has an OffsetIndex, `--walk-pages` yields
    byte-identical output (the O(1) OffsetIndex fast path serves num_pages
    either way — the flag is a no-op here)."""
    cli.main(
        ["column", "show", str(sample_parquet_with_page_index), "--column", "dict_col"]
    )
    default = json.loads(capsys.readouterr().out)
    cli.main(
        [
            "column",
            "show",
            str(sample_parquet_with_page_index),
            "--column",
            "dict_col",
            "--walk-pages",
        ]
    )
    walked = json.loads(capsys.readouterr().out)
    assert default == walked
    for rg in walked["row_groups"]:
        assert rg["num_pages_known"] is True
        assert rg["num_pages"] >= 1


def test_subcommand_column_show_unknown_column_lists_available(sample_parquet, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["column", "show", str(sample_parquet), "--column", "missing"])
    assert exc_info.value.code == 1
    err_json = json.loads(capsys.readouterr().err)
    assert err_json["error"] == "column_not_found"
    assert "Available:" in err_json["message"]


def test_subcommand_legacy_invocation_still_works(sample_parquet, capsys):
    """Sanity: presence of subcommand dispatcher must not break legacy CLI."""
    cli.main(["--output-mode", "segments", str(sample_parquet)])
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert payload[0]["name"] == "magic_number"


def test_cli_html_output_mode(sample_parquet, capsys):
    """The --output-mode html CLI path renders an HTML report to stdout."""
    cli.main([str(sample_parquet), "--output-mode", "html"])
    out = capsys.readouterr().out
    assert "<html" in out.lower() or "<table" in out.lower()


def test_cli_html_output_to_file(sample_parquet, tmp_path, capsys):
    """The --output path writes the report to a file instead of stdout."""
    dest = tmp_path / "report.html"
    cli.main([str(sample_parquet), "--output-mode", "html", "--output", str(dest)])
    assert dest.exists()
    assert "<html" in dest.read_text().lower() or "<table" in dest.read_text().lower()
