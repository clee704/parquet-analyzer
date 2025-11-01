"""Command line interface for :mod:`parquet_analyzer`."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence

from ._core import (
    find_footer_segment,
    get_pages,
    get_summary,
    json_encode,
    parse_parquet_file,
    segment_to_json,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parquet-analyzer")
    parser.add_argument("parquet_file", help="Path to the Parquet file to inspect")
    parser.add_argument(
        "-s",
        "--show-offsets-and-thrift-details",
        action="store_true",
        help="Print the raw segment structure including Thrift offsets",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Standard logging level (e.g. DEBUG, INFO, WARNING)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.getLevelNamesMapping()[args.log_level.upper()],
        format="%(asctime)s %(name)s [%(threadName)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    segments, column_chunk_data_offsets = parse_parquet_file(args.parquet_file)
    if args.show_offsets_and_thrift_details:
        output = segments
    else:
        footer = segment_to_json(find_footer_segment(segments))
        output = {
            "summary": get_summary(footer, segments),
            "footer": footer,
            "pages": get_pages(segments, column_chunk_data_offsets),
        }

    print(json.dumps(output, indent=2, default=json_encode))


if __name__ == "__main__":  # pragma: no cover
    main()
