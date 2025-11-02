import json

import pytest

from parquet_analyzer import cli


def test_build_argument_parser_parses_defaults():
    parser = cli.build_argument_parser()

    args = parser.parse_args(["sample.parquet"])

    assert args.parquet_file == "sample.parquet"
    assert args.show_offsets_and_thrift_details is False
    assert args.log_level == "INFO"


def test_cli_main_outputs_summary(monkeypatch, capsys):
    segments = [
        {
            "name": "footer",
            "offset": 0,
            "length": 0,
            "value": [],
            "metadata": {"type": "struct"},
        }
    ]
    offsets = {("col1",): []}

    monkeypatch.setattr(cli, "parse_parquet_file", lambda path: (segments, offsets))
    monkeypatch.setattr(cli, "segment_to_json", lambda segment: {"num_rows": 1})
    monkeypatch.setattr(cli, "get_summary", lambda footer, segs: {"num_rows": 1})
    monkeypatch.setattr(
        cli,
        "get_pages",
        lambda segs, offset_map: [{"column": ("col1",), "row_groups": []}],
    )

    cli.main(["example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert set(payload) == {"summary", "footer", "pages"}
    assert payload["summary"]["num_rows"] == 1


def test_cli_main_shows_segments(monkeypatch, capsys):
    segments = [{"name": "magic", "offset": 0, "length": 4}]

    monkeypatch.setattr(cli, "parse_parquet_file", lambda path: (segments, {}))

    cli.main(["--show-offsets-and-thrift-details", "example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload == segments


@pytest.mark.parametrize("level", ["DEBUG", "WARNING"])
def test_cli_main_accepts_log_level(monkeypatch, capsys, level):
    segments = [{"name": "magic", "offset": 0, "length": 4}]

    def fake_parse(_: str):
        return segments, {}

    monkeypatch.setattr(cli, "parse_parquet_file", fake_parse)
    monkeypatch.setattr(cli, "segment_to_json", lambda segment: {"num_rows": 1})
    monkeypatch.setattr(cli, "get_summary", lambda footer, segs: {"num_rows": 1})
    monkeypatch.setattr(cli, "get_pages", lambda segs, offsets: [])

    cli.main(["--log-level", level, "example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["summary"]["num_rows"] == 1
