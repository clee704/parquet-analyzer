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