from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Sequence

from ._core import json_encode
from ._html import generate_html_report
from ._subcommands import is_subcommand_invocation, run_subcommand
from .parquet_file import ParquetFile


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


def _run_legacy(argv: Sequence[str]) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.getLevelNamesMapping()[args.log_level.upper()],
        format="%(asctime)s %(name)s [%(threadName)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pf = ParquetFile(args.parquet_file)
    try:
        if args.output_mode == "default":
            output = json.dumps(
                {
                    "summary": pf.full_summary,
                    "footer": pf.footer,
                    "pages": pf.all_pages(),
                },
                indent=2,
                default=json_encode,
            )
        elif args.output_mode == "segments":
            output = json.dumps(pf.all_segments(), indent=2, default=json_encode)
        elif args.output_mode == "html":
            output = generate_html_report(
                args.parquet_file,
                summary=pf.full_summary,
                footer=pf.footer,
                segments=pf.all_segments(),
                sections=args.html_sections,
            )
        else:
            raise ValueError(f"Unknown output mode: {args.output_mode}")
    finally:
        pf.close()

    if args.output:
        pathlib.Path(args.output).write_text(output)
    else:
        print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    if is_subcommand_invocation(argv_list):
        exit_code = run_subcommand(argv_list)
    else:
        exit_code = _run_legacy(argv_list)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
