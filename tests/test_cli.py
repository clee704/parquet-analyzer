import json

import pytest

from parquet_analyzer import cli


class _FakeParquetFile:
    """Minimal ParquetFile stand-in for CLI tests.

    Tracks calls and returns canned values for each property/method the
    CLI touches. The CLI calls `pf.close()` in a try/finally, so the
    stub implements that too.
    """

    def __init__(self, summary=None, footer=None, segments=None, pages=None):
        self.full_summary = summary if summary is not None else {"num_rows": 1}
        self.footer = footer if footer is not None else {"num_rows": 1}
        self._segments = segments if segments is not None else []
        self._pages = pages if pages is not None else []
        self.closed = False

    def all_segments(self):
        return self._segments

    def all_pages(self):
        return self._pages

    def close(self):
        self.closed = True


def test_build_argument_parser_parses_defaults():
    parser = cli.build_argument_parser()

    args = parser.parse_args(["sample.parquet"])

    assert args.parquet_file == "sample.parquet"
    assert args.output_mode == "default"
    assert args.log_level == "INFO"


def test_cli_main_outputs_summary(monkeypatch, capsys):
    fake = _FakeParquetFile(
        summary={"num_rows": 1},
        footer={"num_rows": 1},
        pages=[{"column": ("col1",), "row_groups": []}],
    )
    monkeypatch.setattr(cli, "ParquetFile", lambda path: fake)

    cli.main(["example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert set(payload) == {"summary", "footer", "pages"}
    assert payload["summary"]["num_rows"] == 1
    assert fake.closed, "CLI should close the ParquetFile when done"


def test_cli_main_shows_segments(monkeypatch, capsys):
    segments = [{"name": "magic", "offset": 0, "length": 4}]
    fake = _FakeParquetFile(segments=segments)
    monkeypatch.setattr(cli, "ParquetFile", lambda path: fake)

    cli.main(["--output-mode", "segments", "example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload == segments
    assert fake.closed


@pytest.mark.parametrize("level", ["DEBUG", "WARNING"])
def test_cli_main_accepts_log_level(monkeypatch, capsys, level):
    fake = _FakeParquetFile()
    monkeypatch.setattr(cli, "ParquetFile", lambda path: fake)

    cli.main(["--log-level", level, "example.parquet"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["summary"]["num_rows"] == 1
    assert fake.closed
