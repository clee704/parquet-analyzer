from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Sequence

from ._core import (
    find_footer_segment,
    get_pages,
    get_summary,
    json_encode,
    parse_parquet_file,
    segment_to_json,
)
from ._html import generate_html_report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parquet-analyzer")
    parser.add_argument("parquet_file", help="path to the Parquet file to analyze")
    parser.add_argument(
        "--output-mode",
        choices=["default", "segments", "html"],
        default="default",
        help="set the output mode: 'default' for summary information, 'segments' for raw segment structure, 'html' for HTML report",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write output to the given file path instead of stdout",
    )
    parser.add_argument(
        "--html-sections",
        nargs="*",
        default=["summary", "schema", "key-value-metadata", "row-groups", "columns"],
        choices=[
            "summary",
            "schema",
            "key-value-metadata",
            "row-groups",
            "columns",
            "segments",
            "raw-footer",
        ],
        help="sections to include in the HTML report (only relevant if --output-mode=html)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
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
    if args.output_mode == "default":
        footer = segment_to_json(find_footer_segment(segments))
        output = json.dumps(
            {
                "summary": get_summary(footer, segments),
                "footer": footer,
                "pages": get_pages(segments, column_chunk_data_offsets),
            },
            indent=2,
            default=json_encode,
        )
    elif args.output_mode == "segments":
        output = json.dumps(segments, indent=2, default=json_encode)
    elif args.output_mode == "html":
        footer = segment_to_json(find_footer_segment(segments))
        summary = get_summary(footer, segments)
        output = generate_html_report(
            args.parquet_file,
            summary=summary,
            footer=footer,
            segments=segments,
            sections=args.html_sections,
        )
    else:
        raise ValueError(f"Unknown output mode: {args.output_mode}")

    if args.output:
        pathlib.Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":  # pragma: no cover
    main()
