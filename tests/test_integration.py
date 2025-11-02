import importlib
import json

import pytest

from parquet_analyzer import cli
from parquet_analyzer._core import (
    find_footer_segment,
    get_pages,
    get_summary,
    parse_parquet_file,
    segment_to_json,
)


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


def test_parse_parquet_file_smoke(sample_parquet):
    segments, column_offset_map = parse_parquet_file(str(sample_parquet))

    footer_segment = find_footer_segment(segments)
    assert footer_segment is not None

    footer_json = segment_to_json(footer_segment)
    summary = get_summary(footer_json, segments)

    assert summary["num_rows"] == 3
    assert summary["num_row_groups"] == 1
    assert summary["num_columns"] == 2
    assert summary["num_data_pages"] >= 1
    assert summary["footer_size"] > 0

    pages = get_pages(segments, column_offset_map)
    assert pages
    first_column = pages[0]
    assert first_column["row_groups"], "Row group data should be present"
    first_row_group = first_column["row_groups"][0]
    assert first_row_group["data_pages"], "Data pages should be listed"


def test_parse_parquet_file_with_page_indexes(sample_parquet_with_page_index):
    file_path = str(sample_parquet_with_page_index)
    segments, column_offset_map = parse_parquet_file(file_path)

    # Expect dictionary pages and indexes to be captured for at least one column
    column = next(iter(column_offset_map.values()))[0]
    assert "dictionary_page" in column
    assert "column_index" in column
    assert "offset_index" in column

    pages = get_pages(segments, column_offset_map)
    row_group = pages[0]["row_groups"][0]
    assert "dictionary_page" in row_group
    assert "column_index" in row_group
    assert "offset_index" in row_group

    footer = segment_to_json(find_footer_segment(segments))
    summary = get_summary(footer, segments)
    assert summary["num_dict_pages"] >= 1
    assert summary["column_index_size"] > 0
    assert summary["offset_index_size"] > 0


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
