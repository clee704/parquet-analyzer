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


def test_subcommand_column_show_no_offset_index_marks_pages_unknown(
    sample_parquet, capsys
):
    """pyarrow default writes no page index — `num_pages` must NOT be reported.

    This guards against accidental page-header walks (the v1 contract
    forbids them) by asserting the behavior on a real file that lacks
    an OffsetIndex.
    """
    cli.main(["column", "show", str(sample_parquet), "--column", "ints"])
    payload = json.loads(capsys.readouterr().out)
    rg = payload["row_groups"][0]
    assert rg["has_offset_index"] is False
    assert rg["num_pages"] is None
    assert rg["num_pages_known"] is False


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
