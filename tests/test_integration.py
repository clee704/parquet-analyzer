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
