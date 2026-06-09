"""Verb-noun subcommand handlers for the AI-friendly CLI surface.

This module wires the lazy :class:`parquet_analyzer.ParquetFile` core to a
set of small, composable subcommands that each answer one question. The
v1 contract this module implements:

* one subcommand → one JSON object on stdout; list-shaped outputs wrap in
  ``{items, total, returned, truncated}``
* errors → stderr as ``{"error": "<code>", "message": "<human>", "fix": "<retry command>"}``
  with a nonzero exit code
* every output object carries a ``$schema`` field of the form
  ``parquet-analyzer/v1/<command>``; passing ``--schema-version`` short-circuits
  the command to print just the schema URI
* list-shaped outputs accept ``--limit N``

The handlers are intentionally footer-only: nothing here triggers an eager
walk through :meth:`ParquetFile.all_pages` /
:meth:`ParquetFile.all_segments`, and ``cc.num_pages`` is only consulted when
``cc.has_offset_index`` is true (otherwise the lookup would walk page
headers). Page-level subcommands (``page list / header / extract / decode``)
are tracked in #21.

**Adding a new field to any subcommand's output?** See
``docs/output-principles.md`` for the v1 contract — footer-bounded and
walk-free, with the ``page`` subcommand surface as the explicit escape
hatch for everything beyond.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import pathlib
import sys
from typing import Any, Callable, Iterable, Sequence

from ._core import json_encode
from . import _navigate
from .parquet_file import ParquetFile, PageDecodeError, PlainValues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_BASE = "parquet-analyzer/v1"

SUBCOMMAND_VERBS: frozenset[str] = frozenset(
    {"file", "rowgroup", "column", "page", "show"}
)


def _schema_uri(command: str) -> str:
    return f"{SCHEMA_BASE}/{command}"


# Mapping from (verb, noun) → schema name. Single source of truth so the
# `--schema-version` short-circuit and the per-handler ``$schema`` field
# can never drift apart.
SCHEMAS: dict[tuple[str, str | None], str] = {
    ("file", "summary"): "file-summary",
    ("file", "kv"): "file-kv",
    ("file", "schema"): "file-schema",
    ("file", "validate"): "file-validate",
    ("rowgroup", "list"): "rowgroup-list",
    ("rowgroup", "show"): "rowgroup-show",
    ("column", "list"): "column-list",
    ("column", "show"): "column-show",
    ("page", "list"): "page-list",
    ("page", "header"): "page-header",
    ("page", "extract"): "page-extract",
    ("page", "decode"): "page-decode",
    ("show", None): "show",
}


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


class CliError(Exception):
    """Raised by handlers when the command cannot complete.

    Carries the structured fields required by the v1 error contract.
    ``details`` holds optional extra fields (e.g. the unsupported
    ``encoding`` / ``codec``) that are merged into the emitted error object.
    """

    def __init__(
        self, code: str, message: str, fix: str, details: dict | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix
        self.details = details or {}


def emit_error(err: CliError, stream=None) -> None:
    """Write a single JSON error object to ``stream`` (defaults to stderr)."""
    if stream is None:
        stream = sys.stderr
    payload = {
        "$schema": _schema_uri("error"),
        "error": err.code,
        "message": err.message,
        "fix": err.fix,
        **err.details,
    }
    stream.write(json.dumps(payload, default=json_encode) + "\n")


class _JsonErrorParser(argparse.ArgumentParser):
    """ArgumentParser that emits parser errors in the JSON error contract.

    Used only by the subcommand grammar; the legacy ``parquet-analyzer
    <path>`` parser keeps argparse's default text behaviour for byte-identical
    backward compatibility.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        # Mirror argparse semantics (exit code 2) but emit JSON to stderr.
        err = CliError(
            code="invalid_arguments",
            message=message,
            fix=f"parquet-analyzer {self.prog.split(' ', 1)[-1] if ' ' in self.prog else ''} --help",
        )
        emit_error(err)
        self.exit(2)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _emit_json(payload: Any, output_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=json_encode)
    if output_path:
        pathlib.Path(output_path).write_text(text)
    else:
        sys.stdout.write(text + "\n")


def _wrap_list(items: list, limit: int | None, command: str) -> dict:
    total = len(items)
    if limit is not None and limit < total:
        returned_items = items[:limit]
        truncated = True
    else:
        returned_items = items
        truncated = False
    return {
        "$schema": _schema_uri(command),
        "items": returned_items,
        "returned": len(returned_items),
        "total": total,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Column-path helpers
# ---------------------------------------------------------------------------


def _path_tuple(footer_column: dict) -> tuple[str, ...]:
    return tuple(footer_column["meta_data"]["path_in_schema"])


def _path_display(path: Sequence[str]) -> str:
    return ".".join(path)


def _select_columns_by_name(
    footer: dict, column_name: str, row_group_index: int | None
) -> list[tuple[int, dict]]:
    """Return ``[(row_group_index, footer_column_dict), ...]`` matching ``column_name``.

    Matches against the dot-joined ``path_in_schema``. If multiple distinct
    paths join to the same display string (a pathological case — parquet
    permits dots inside field names), raises :class:`CliError` listing the
    candidate path tuples so the caller can disambiguate.
    """
    row_groups = footer["row_groups"]
    if row_group_index is not None:
        rg_iter: Iterable[tuple[int, dict]] = [
            (row_group_index, row_groups[row_group_index])
        ]
    else:
        rg_iter = enumerate(row_groups)

    matches: list[tuple[int, dict]] = []
    distinct_paths: set[tuple[str, ...]] = set()
    for rg_idx, rg in rg_iter:
        for cc in rg["columns"]:
            path = _path_tuple(cc)
            if _path_display(path) == column_name:
                matches.append((rg_idx, cc))
                distinct_paths.add(path)

    if len(distinct_paths) > 1:
        raise CliError(
            code="ambiguous_column",
            message=(
                f"column name {column_name!r} matches multiple distinct paths: "
                + ", ".join(repr(list(p)) for p in sorted(distinct_paths))
            ),
            fix="re-run with an unambiguous --column value or use a dotted path",
        )
    return matches


def _all_column_paths(footer: dict) -> list[tuple[str, ...]]:
    """Distinct column paths in the file (footer-only, preserves first-seen order)."""
    seen: dict[tuple[str, ...], None] = {}
    for rg in footer["row_groups"]:
        for cc in rg["columns"]:
            path = _path_tuple(cc)
            if path not in seen:
                seen[path] = None
    return list(seen)


# ---------------------------------------------------------------------------
# Per-column-chunk summarisation (footer-only)
# ---------------------------------------------------------------------------


def _column_chunk_summary(
    rg_index: int,
    col_index: int,
    footer_column: dict,
    cc_wrapper: Any | None = None,
) -> dict:
    """Footer-only per-chunk summary.

    Byte-offset / length fields surfaced (all footer-derived):

    * ``chunk_offset`` / ``chunk_length`` — the seek-and-read pair for the
      column chunk's on-disk bytes. ``chunk_offset`` is the dictionary page
      offset when a dictionary is present, otherwise the data page offset
      (per parquet spec: dictionary always precedes data within a chunk).
      ``chunk_length == total_compressed_size``; parquet's
      ``total_compressed_size`` already includes the dictionary page.
    * ``offset_index_offset`` / ``offset_index_length`` (null if absent).
    * ``column_index_offset`` / ``column_index_length`` (null if absent).
    * ``bloom_filter_offset`` / ``bloom_filter_length`` (null if absent).
    * Existing ``data_page_offset``, ``dictionary_page_offset``,
      ``compressed_size``, ``uncompressed_size`` are unchanged.

    ``num_pages`` is reported only when the chunk has an OffsetIndex
    (``offset_index_offset`` present) AND ``cc_wrapper`` is supplied — the
    wrapper does an O(1) OffsetIndex lookup via the offset_index_length
    that the footer records. Without an OffsetIndex, computing the count
    requires walking page headers; the v1 contract forbids that, so we
    report ``num_pages: null`` / ``num_pages_known: false`` instead.

    ``_path`` is the canonical ``show`` navigation path for this chunk
    (``row_groups/<rg_index>/columns/<col_index>``) — feed it to ``show``
    to drill into the column's pages.
    """
    md = footer_column["meta_data"]
    path = tuple(md["path_in_schema"])

    has_offset_index = "offset_index_offset" in footer_column
    has_column_index = "column_index_offset" in footer_column
    has_dictionary = "dictionary_page_offset" in md
    has_bloom_filter = "bloom_filter_offset" in md

    # Chunk byte range: dictionary precedes data when present.
    data_page_offset = md.get("data_page_offset")
    dictionary_page_offset = md.get("dictionary_page_offset")
    chunk_offset = (
        dictionary_page_offset
        if dictionary_page_offset is not None
        else data_page_offset
    )
    chunk_length = md.get("total_compressed_size")

    num_pages: int | None = None
    num_pages_known = False
    if has_offset_index and cc_wrapper is not None:
        # cc_wrapper.num_pages is O(1) when has_offset_index is True (the
        # ColumnChunk wrapper caches the OffsetIndex thrift parse). Verified
        # in parquet_file.ColumnChunk.num_pages.
        num_pages = cc_wrapper.num_pages
        num_pages_known = True

    out: dict[str, Any] = {
        "row_group": rg_index,
        "_path": f"row_groups/{rg_index}/columns/{col_index}",
        "column": _path_display(path),
        "path": list(path),
        "type": md.get("type"),
        "encodings": list(md.get("encodings", [])),
        "codec": md.get("codec"),
        "num_values": md.get("num_values"),
        "compressed_size": chunk_length,
        "uncompressed_size": md.get("total_uncompressed_size"),
        "chunk_offset": chunk_offset,
        "chunk_length": chunk_length,
        "data_page_offset": data_page_offset,
        "dictionary_page_offset": dictionary_page_offset,
        "has_dictionary": has_dictionary,
        "has_offset_index": has_offset_index,
        "offset_index_offset": footer_column.get("offset_index_offset"),
        "offset_index_length": footer_column.get("offset_index_length"),
        "has_column_index": has_column_index,
        "column_index_offset": footer_column.get("column_index_offset"),
        "column_index_length": footer_column.get("column_index_length"),
        "has_bloom_filter": has_bloom_filter,
        "bloom_filter_offset": md.get("bloom_filter_offset"),
        "bloom_filter_length": md.get("bloom_filter_length"),
        "num_pages": num_pages,
        "num_pages_known": num_pages_known,
    }

    statistics = md.get("statistics")
    if statistics is not None:
        out["statistics"] = statistics

    return out


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_file_summary(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        payload = {
            "$schema": _schema_uri("file-summary"),
            **pf.footer_summary,
            "footer_offset": pf.footer_offset,
        }
    _emit_json(payload, args.output)


def handle_file_kv(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        all_items = [{"key": k, "value": v} for k, v in pf.kv_metadata]
    if args.key is not None:
        items = [it for it in all_items if it["key"] == args.key]
    else:
        items = all_items
    payload = _wrap_list(items, args.limit, "file-kv")
    if args.key is not None:
        payload["filter_key"] = args.key
    _emit_json(payload, args.output)


def handle_file_schema(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        schema = pf.schema
    payload = {
        "$schema": _schema_uri("file-schema"),
        "elements": list(schema),
    }
    _emit_json(payload, args.output)


def handle_file_validate(args: argparse.Namespace) -> None:
    """Footer-only structural validation.

    A parse failure inside :class:`ParquetFile` (bad magic, unparseable
    footer) is reported as ``valid: false`` with the exception captured as
    a finding — not as a CLI error. Reserve nonzero exits for operational
    failures (e.g. missing file).
    """
    errors: list[dict] = []
    payload: dict[str, Any] = {
        "$schema": _schema_uri("file-validate"),
        "path": args.path,
    }
    try:
        pf = ParquetFile(args.path)
    except ValueError as exc:
        errors.append(
            {
                "code": "footer_parse_failed",
                "message": str(exc),
            }
        )
        payload["valid"] = False
        payload["errors"] = errors
        _emit_json(payload, args.output)
        return

    try:
        footer = pf.footer
        num_rows = pf.num_rows
        num_row_groups = pf.num_row_groups
        num_columns = pf.num_columns

        if num_rows < 0:
            errors.append(
                {"code": "negative_num_rows", "message": f"num_rows={num_rows} < 0"}
            )
        if num_rows > 0 and num_row_groups == 0:
            errors.append(
                {
                    "code": "missing_row_groups",
                    "message": (
                        f"file claims num_rows={num_rows} but has no row groups"
                    ),
                }
            )

        sum_rg_rows = 0
        for rg_idx, rg in enumerate(footer["row_groups"]):
            rg_rows = rg.get("num_rows", 0)
            sum_rg_rows += rg_rows
            if rg_rows < 0:
                errors.append(
                    {
                        "code": "negative_row_group_rows",
                        "message": (f"row_group[{rg_idx}] num_rows={rg_rows} < 0"),
                    }
                )
            cols = rg.get("columns", [])
            if len(cols) != num_columns:
                errors.append(
                    {
                        "code": "inconsistent_column_count",
                        "message": (
                            f"row_group[{rg_idx}] has {len(cols)} columns; "
                            f"file declares {num_columns}"
                        ),
                    }
                )
            for c_idx, cc in enumerate(cols):
                md = cc.get("meta_data", {})
                comp = md.get("total_compressed_size", 0)
                if rg_rows > 0 and comp <= 0:
                    errors.append(
                        {
                            "code": "empty_column_chunk",
                            "message": (
                                f"row_group[{rg_idx}].columns[{c_idx}] "
                                f"({_path_display(md.get('path_in_schema', []))}) "
                                f"has total_compressed_size={comp} with rg num_rows={rg_rows}"
                            ),
                        }
                    )

        if num_rows > 0 and sum_rg_rows != num_rows:
            errors.append(
                {
                    "code": "row_count_mismatch",
                    "message": (
                        f"sum of row-group num_rows ({sum_rg_rows}) does not "
                        f"equal file num_rows ({num_rows})"
                    ),
                }
            )
    finally:
        pf.close()

    payload["valid"] = not errors
    payload["errors"] = errors
    _emit_json(payload, args.output)


def _row_group_summary(rg_index: int, footer_rg: dict) -> dict:
    """Footer-only per-row-group summary (no column-chunk detail).

    Adds byte-offset / length context:

    * ``file_offset`` — start of the row group's data on disk (parquet's
      ``RowGroup.file_offset``, the offset of the first page in the rg).
    * ``total_byte_size`` — sum of *uncompressed* column data (parquet's
      ``RowGroup.total_byte_size``).
    * ``total_compressed_size`` — computed sum of
      ``ColumnChunk.meta_data.total_compressed_size`` across the rg.

    ``_path`` is the canonical ``show`` navigation path for this row group
    (``row_groups/<i>``) — feed it to ``show`` to drill in.
    """
    cols = footer_rg.get("columns", [])
    total_compressed = sum(
        (cc.get("meta_data", {}).get("total_compressed_size") or 0) for cc in cols
    )
    return {
        "row_group": rg_index,
        "_path": f"row_groups/{rg_index}",
        "num_rows": footer_rg.get("num_rows"),
        "file_offset": footer_rg.get("file_offset"),
        "total_byte_size": footer_rg.get("total_byte_size"),
        "total_compressed_size": total_compressed,
        "num_columns": len(cols),
    }


def handle_rowgroup_list(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        items = [
            _row_group_summary(rg_idx, rg)
            for rg_idx, rg in enumerate(footer["row_groups"])
        ]
    payload = _wrap_list(items, args.limit, "rowgroup-list")
    _emit_json(payload, args.output)


def handle_rowgroup_show(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        try:
            rg = footer["row_groups"][args.row_group]
        except IndexError as exc:
            raise CliError(
                code="row_group_out_of_range",
                message=(
                    f"row group {args.row_group} requested but file has "
                    f"{len(footer['row_groups'])} row group(s)"
                ),
                fix=f"parquet-analyzer rowgroup list {args.path}",
            ) from exc

        rg_wrapper = pf.row_groups[args.row_group]
        columns = [
            _column_chunk_summary(args.row_group, col_idx, footer_cc, cc_wrapper)
            for col_idx, (footer_cc, cc_wrapper) in enumerate(
                zip(rg.get("columns", []), rg_wrapper.columns)
            )
        ]

    payload = {
        "$schema": _schema_uri("rowgroup-show"),
        **_row_group_summary(args.row_group, rg),
        "columns": columns,
    }
    _emit_json(payload, args.output)


def handle_column_list(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        if args.row_group is not None:
            if not (0 <= args.row_group < len(footer["row_groups"])):
                raise CliError(
                    code="row_group_out_of_range",
                    message=(
                        f"row group {args.row_group} requested but file has "
                        f"{len(footer['row_groups'])} row group(s)"
                    ),
                    fix=f"parquet-analyzer rowgroup list {args.path}",
                )
            rg_indices = [args.row_group]
        else:
            rg_indices = list(range(len(footer["row_groups"])))

        items: list[dict] = []
        for rg_idx in rg_indices:
            footer_rg = footer["row_groups"][rg_idx]
            rg_wrapper = pf.row_groups[rg_idx]
            for col_idx, (footer_cc, cc_wrapper) in enumerate(
                zip(footer_rg.get("columns", []), rg_wrapper.columns)
            ):
                items.append(
                    _column_chunk_summary(rg_idx, col_idx, footer_cc, cc_wrapper)
                )
    payload = _wrap_list(items, args.limit, "column-list")
    if args.row_group is not None:
        payload["row_group"] = args.row_group
    _emit_json(payload, args.output)


def handle_column_show(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        if args.row_group is not None and not (
            0 <= args.row_group < len(footer["row_groups"])
        ):
            raise CliError(
                code="row_group_out_of_range",
                message=(
                    f"row group {args.row_group} requested but file has "
                    f"{len(footer['row_groups'])} row group(s)"
                ),
                fix=f"parquet-analyzer rowgroup list {args.path}",
            )

        matches = _select_columns_by_name(footer, args.column, args.row_group)
        if not matches:
            available = sorted(_path_display(p) for p in _all_column_paths(footer))
            raise CliError(
                code="column_not_found",
                message=(
                    f"column {args.column!r} not found"
                    + (
                        f" in row group {args.row_group}"
                        if args.row_group is not None
                        else ""
                    )
                    + f". Available: {', '.join(available)}"
                ),
                fix=f"parquet-analyzer column list {args.path}",
            )

        path = _path_tuple(matches[0][1])
        type_name = matches[0][1]["meta_data"].get("type")

        # Build wrapper lookup so num_pages is reported when OffsetIndex
        # exists (still O(1), still footer-bounded).
        row_group_details: list[dict] = []
        for rg_idx, footer_cc in matches:
            rg_wrapper = pf.row_groups[rg_idx]
            # Locate the same column chunk in the wrapper list. Matching by
            # path tuple is safe because parquet guarantees one chunk per
            # path per row group; its position is the column index used in
            # the navigation _path.
            col_idx, cc_wrapper = next(
                ((i, c) for i, c in enumerate(rg_wrapper.columns) if c.path == path),
                (0, None),
            )
            row_group_details.append(
                _column_chunk_summary(rg_idx, col_idx, footer_cc, cc_wrapper)
            )

    # Aggregates across matched row groups. All footer-derived (sum of
    # ColumnMetaData fields), so zero additional cost — and they answer
    # the "how big is column X overall?" / "how many values across the
    # whole file?" questions in one step. Symmetric with the existing
    # file-level (compressed_page_size) and rowgroup-level
    # (total_compressed_size) sums.
    total_compressed = sum(d["compressed_size"] or 0 for d in row_group_details)
    total_uncompressed = sum(d["uncompressed_size"] or 0 for d in row_group_details)
    total_num_values = sum(d["num_values"] or 0 for d in row_group_details)

    payload: dict[str, Any] = {
        "$schema": _schema_uri("column-show"),
        "column": _path_display(path),
        "path": list(path),
        "type": type_name,
        "num_row_groups": len(row_group_details),
        "total_num_values": total_num_values,
        "total_compressed_size": total_compressed,
        "total_uncompressed_size": total_uncompressed,
        "row_groups": row_group_details,
    }
    if args.row_group is not None:
        payload["filter_row_group"] = args.row_group
    _emit_json(payload, args.output)


# ---------------------------------------------------------------------------
# Page subcommands (#21) — the page-walk / body-decode escape-hatch surface
# ---------------------------------------------------------------------------


def _resolve_page_column(
    pf: ParquetFile, footer: dict, column: str, row_group: int | None, noun: str
) -> tuple[int, int, Any]:
    """Resolve ``(row_group_index, col_index, ColumnChunk)`` for a singular
    page verb. Enforces the ``--row-group`` required-when-multiple rule: a file
    with one row group defaults to 0; a file with several requires an explicit
    ``--row-group`` rather than silently defaulting."""
    n_rg = len(footer["row_groups"])
    if row_group is None:
        if n_rg <= 1:
            row_group = 0
        else:
            raise CliError(
                code="missing_argument",
                message=(
                    f"page {noun}: --row-group is required because the file has "
                    f"{n_rg} row groups"
                ),
                fix=(
                    f"parquet-analyzer page {noun} {pf.path} --column {column} "
                    f"--page-index 0 --row-group 0"
                ),
            )
    if not (0 <= row_group < n_rg):
        raise CliError(
            code="row_group_out_of_range",
            message=(
                f"row group {row_group} requested but file has {n_rg} row group(s)"
            ),
            fix=f"parquet-analyzer rowgroup list {pf.path}",
        )
    matches = _select_columns_by_name(footer, column, row_group)
    if not matches:
        available = sorted(_path_display(p) for p in _all_column_paths(footer))
        raise CliError(
            code="column_not_found",
            message=(
                f"column {column!r} not found in row group {row_group}. "
                f"Available: {', '.join(available)}"
            ),
            fix=f"parquet-analyzer column list {pf.path}",
        )
    path = _path_tuple(matches[0][1])
    rg_wrapper = pf.row_groups[row_group]
    col_idx, cc = next(
        (i, c) for i, c in enumerate(rg_wrapper.columns) if c.path == path
    )
    return row_group, col_idx, cc


def _chunk_has_leading_dict(cc: Any) -> bool:
    """Whether the chunk's first page is a dictionary page. Prefers the
    footer's ``dictionary_page_offset`` on the OffsetIndex fast path (the
    indexing model the core uses there); on the walk path it inspects the
    actual first page kind, so older writers that carry a dictionary page but
    leave ``dictionary_page_offset`` unset are still classified correctly."""
    if cc.has_offset_index:
        return bool(cc.dictionary_page_offset)
    pages = cc.pages()
    return bool(pages) and pages[0]._kind == "dictionary_page"


def _resolve_page(
    cc: Any, page_index: int, noun: str, path_str: str, column: str
) -> tuple[Any, int, int | None]:
    """Resolve one page by index (supporting negatives), returning
    ``(Page, page_index, data_page_index)``."""
    n = cc.num_pages
    idx = page_index + n if page_index < 0 else page_index
    if not 0 <= idx < n:
        raise CliError(
            code="page_out_of_range",
            message=(
                f"page index {page_index} out of range; column chunk has {n} pages"
            ),
            fix=f"parquet-analyzer page list {path_str} --column {column}",
        )
    page = cc.page(idx)
    if page._kind == "dictionary_page":
        data_page_index: int | None = None
    else:
        data_page_index = idx - (1 if _chunk_has_leading_dict(cc) else 0)
    return page, idx, data_page_index


def _cap(values: list, limit: int | None) -> tuple[list, bool]:
    """Apply ``--limit`` to a value list, returning ``(capped, truncated)``.
    Mirrors :func:`_wrap_list`'s truncation rule so the page surface caps
    list- and decode-shaped output consistently."""
    if limit is not None and limit < len(values):
        return values[:limit], True
    return values, False


def handle_page_list(args: argparse.Namespace) -> None:
    """List a column's (or every column's) pages as lightweight stubs. This is
    the page-walk surface — cheap when the writer emitted an OffsetIndex (the
    extents come from it), otherwise it walks the per-page headers."""
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        n_rg = len(footer["row_groups"])
        if args.row_group is not None and not (0 <= args.row_group < n_rg):
            raise CliError(
                code="row_group_out_of_range",
                message=(
                    f"row group {args.row_group} requested but file has "
                    f"{n_rg} row group(s)"
                ),
                fix=f"parquet-analyzer rowgroup list {args.path}",
            )
        rg_indices = [args.row_group] if args.row_group is not None else range(n_rg)
        items: list[dict] = []
        for rg_idx in rg_indices:
            rg_wrapper = pf.row_groups[rg_idx]
            for col_idx, cc in enumerate(rg_wrapper.columns):
                if args.column is not None and _path_display(cc.path) != args.column:
                    continue
                items.extend(_page_list_items(cc, rg_idx, col_idx))
    payload = _wrap_list(items, args.limit, "page-list")
    if args.row_group is not None:
        payload["row_group"] = args.row_group
    if args.column is not None:
        payload["column"] = args.column
    _emit_json(payload, args.output)


def _page_list_items(cc: Any, rg_idx: int, col_idx: int) -> list[dict]:
    """Per-page stub items for one column chunk. Uses the OffsetIndex stubs
    (no header reads) when available, else walks the page headers.

    ``data_page_index`` is assigned by a running counter over the enumerated
    pages, so it stays correct regardless of how the dictionary page is
    signalled (footer offset vs. an inline header on older writers)."""
    base = f"row_groups/{rg_idx}/columns/{col_idx}"
    stubs = cc.page_stubs()
    if stubs is not None:
        descriptors = [(s.kind, s.offset, s.length, s.first_row_index) for s in stubs]
    else:
        descriptors = [
            (
                "dictionary_page" if p._kind == "dictionary_page" else "data_page",
                p.offset,
                p._length,
                None,
            )
            for p in cc.pages()
        ]
    out: list[dict] = []
    data_index = 0
    for page_index, (kind, offset, length, first_row_index) in enumerate(descriptors):
        if kind == "dictionary_page":
            data_page_index: int | None = None
        else:
            data_page_index = data_index
            data_index += 1
        out.append(
            _page_list_item(
                cc,
                rg_idx,
                base,
                page_index,
                kind,
                offset,
                length,
                first_row_index,
                data_page_index,
            )
        )
    return out


def _page_list_item(
    cc: Any,
    rg_idx: int,
    base: str,
    page_index: int,
    kind: str,
    offset: int,
    length: int,
    first_row_index: int | None,
    data_page_index: int | None,
) -> dict:
    """One ``page list`` stub. ``offset`` + ``length`` address the page's full
    on-disk byte range — ``length`` is the whole page span (thrift header plus
    compressed body), matching the tree view's ``_location``. (Contrast
    ``page header``'s ``compressed_size``, which is the body-only thrift
    field.)"""
    return {
        "row_group": rg_idx,
        "column": _path_display(cc.path),
        "_path": f"{base}/pages/{page_index}",
        "page_index": page_index,
        "data_page_index": data_page_index,
        "kind": kind,
        "offset": offset,
        "length": length,
        "first_row_index": first_row_index,
    }


def _page_header_fields(page: Any) -> dict:
    """The curated per-page header fields, version-aware."""
    out: dict[str, Any] = {
        "kind": "dictionary_page" if page._kind == "dictionary_page" else "data_page",
        "page_type": page.type,
        "offset": page.offset,
        "header_size": page.header_size,
        "compressed_size": page.compressed_size,
        "uncompressed_size": page.uncompressed_size,
        "num_values": page.num_values,
        "encoding": page.encoding,
    }
    h1 = page._t.data_page_header
    h2 = page._t.data_page_header_v2
    if h1 is not None:
        out["definition_level_encoding"] = _enc_name(h1.definition_level_encoding)
        out["repetition_level_encoding"] = _enc_name(h1.repetition_level_encoding)
        out["statistics"] = _page_stats(page, h1)
    elif h2 is not None:
        out["num_nulls"] = h2.num_nulls
        out["num_rows"] = h2.num_rows
        out["is_compressed"] = h2.is_compressed
        out["definition_levels_byte_length"] = h2.definition_levels_byte_length
        out["repetition_levels_byte_length"] = h2.repetition_levels_byte_length
        out["statistics"] = _page_stats(page, h2)
    return out


def _enc_name(value: int) -> str:
    from .parquet_file import _ENCODING_NAMES

    return _ENCODING_NAMES.get(value, str(value))


def _page_stats(page: Any, header: Any) -> Any:
    """Decode a page header's statistics against the owning column's type."""
    from ._tree_json import _page_statistics

    return _page_statistics(header, page)


def handle_page_header(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        rg_idx, _col_idx, cc = _resolve_page_column(
            pf, footer, args.column, args.row_group, "header"
        )
        page, page_index, data_page_index = _resolve_page(
            cc, args.page_index, "header", args.path, args.column
        )
        payload = {
            "$schema": _schema_uri("page-header"),
            "row_group": rg_idx,
            "column": _path_display(cc.path),
            "page_index": page_index,
            "data_page_index": data_page_index,
            **_page_header_fields(page),
        }
    _emit_json(payload, args.output)


def handle_page_extract(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        rg_idx, _col_idx, cc = _resolve_page_column(
            pf, footer, args.column, args.row_group, "extract"
        )
        page, page_index, data_page_index = _resolve_page(
            cc, args.page_index, "extract", args.path, args.column
        )
        try:
            data = page.decompressed_body() if args.decompress else page.raw_body()
        except PageDecodeError as exc:
            raise CliError(
                exc.code,
                str(exc),
                fix=f"parquet-analyzer page extract {args.path} --column {args.column} "
                f"--page-index {args.page_index}",
                details=_decode_error_details(exc),
            ) from exc

    if args.as_format == "raw":
        # The deliberate JSON-contract escape hatch: write the bytes verbatim.
        if args.output:
            pathlib.Path(args.output).write_bytes(data)
        else:
            sys.stdout.buffer.write(data)
        return

    encoded = (
        data.hex()
        if args.as_format == "hex"
        else base64.b64encode(data).decode("ascii")
    )
    payload = {
        "$schema": _schema_uri("page-extract"),
        "row_group": rg_idx,
        "column": _path_display(cc.path),
        "page_index": page_index,
        "data_page_index": data_page_index,
        "decompressed": args.decompress,
        "encoding": args.as_format,
        "byte_length": len(data),
        "data": encoded,
    }
    _emit_json(payload, args.output)


def _decode_error_details(exc: PageDecodeError) -> dict:
    """The extra error fields a :class:`PageDecodeError` carries, for the
    JSON error contract (e.g. ``encoding`` / ``codec``)."""
    details: dict[str, Any] = {}
    for attr in ("encoding", "codec", "page_type"):
        if hasattr(exc, attr):
            details[attr] = getattr(exc, attr)
    return details


def _decode_cli_error(exc: PageDecodeError, args: Any) -> CliError:
    """Map a :class:`PageDecodeError` onto the structured CLI error contract,
    pointing the ``fix`` at ``page extract`` (the raw-bytes fallback)."""
    return CliError(
        exc.code,
        str(exc),
        fix=f"parquet-analyzer page extract {args.path} --column {args.column} "
        f"--page-index {args.page_index} --as hex",
        details=_decode_error_details(exc),
    )


def _run_dict(run: Any) -> dict:
    from .decoders import RleRun

    if isinstance(run, RleRun):
        return {"kind": "rle", "value": run.value, "length": run.length}
    return {"kind": "bit_packed", "length": run.length, "values": list(run.values)}


def _level_view(stream: Any, limit: int | None) -> dict | None:
    """Curated view of a level stream (`rep`/`def`), or ``None`` when the
    column has no such level block."""
    if stream is None:
        return None
    levels, truncated = _cap(list(stream.values), limit)
    return {
        "bit_width": stream.bit_width,
        "total": len(stream.values),
        "returned": len(levels),
        "truncated": truncated,
        "levels": levels,
    }


def handle_page_decode(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        footer = pf.footer
        rg_idx, _col_idx, cc = _resolve_page_column(
            pf, footer, args.column, args.row_group, "decode"
        )
        page, page_index, data_page_index = _resolve_page(
            cc, args.page_index, "decode", args.path, args.column
        )
        result = _decode_view(page, args.kind, args.limit, args)

        payload = {
            "$schema": _schema_uri("page-decode"),
            "row_group": rg_idx,
            "column": _path_display(cc.path),
            "page_index": page_index,
            "data_page_index": data_page_index,
            "kind": args.kind,
            "encoding": page.encoding,
            **result,
        }
    _emit_json(payload, args.output)


def _decode_view(page: Any, kind: str, limit: int | None, args: Any) -> dict:
    """Project one decoded view of a page. ``statistics`` is header-only (no
    body decode); the others decode the body and map any decode failure to the
    JSON error contract."""
    if kind == "statistics":
        header = page._t.data_page_header or page._t.data_page_header_v2
        if header is None:
            raise CliError(
                code="page_type_not_supported",
                message="statistics are only available on V1/V2 data pages",
                fix=f"parquet-analyzer page header {args.path} --column {args.column} "
                f"--page-index {args.page_index}",
            )
        return {"statistics": _page_stats(page, header)}

    try:
        decoded = page.decode()
    except PageDecodeError as exc:
        raise _decode_cli_error(exc, args) from exc

    if kind == "values":
        try:
            physical = page.physical_values()
        except PageDecodeError as exc:
            # physical_values() resolves dictionary indices through the sibling
            # dictionary page, which can fail independently of decode() — keep
            # that failure on the JSON error contract too.
            raise _decode_cli_error(exc, args) from exc
        values, truncated = _cap(physical, limit)
        return {
            "total": decoded.num_values - decoded.num_nulls,
            "returned": len(values),
            "truncated": truncated,
            "num_nulls": decoded.num_nulls,
            "values": values,
        }
    if kind == "levels":
        return {
            "definition_levels": _level_view(decoded.definition_levels, limit),
            "repetition_levels": _level_view(decoded.repetition_levels, limit),
        }
    # kind == "rle-runs"
    section = decoded.values
    if isinstance(section, PlainValues):
        raise CliError(
            code="kind_not_available",
            message=(
                "rle-runs is only available for dictionary-encoded pages; this "
                f"page uses {decoded.encoding} (no RLE/bit-packed run structure)"
            ),
            fix=f"parquet-analyzer page decode {args.path} --column {args.column} "
            f"--page-index {args.page_index} --kind values",
        )
    runs, truncated = _cap([_run_dict(r) for r in section.runs], limit)
    return {
        "bit_width": section.bit_width,
        "total": len(section.runs),
        "returned": len(runs),
        "truncated": truncated,
        "runs": runs,
    }


def handle_show(args: argparse.Namespace) -> None:
    with ParquetFile(args.path) as pf:
        try:
            rendered = _navigate.render(
                pf, args.navpath, walk_pages=args.walk_pages, limit=args.limit
            )
        except _navigate.NavigationError as exc:
            raise CliError(exc.code, exc.message, exc.fix) from exc
    payload = {"$schema": _schema_uri("show"), **rendered}
    _emit_json(payload, args.output)


HANDLERS: dict[tuple[str, str | None], Callable[[argparse.Namespace], None]] = {
    ("file", "summary"): handle_file_summary,
    ("file", "kv"): handle_file_kv,
    ("file", "schema"): handle_file_schema,
    ("file", "validate"): handle_file_validate,
    ("rowgroup", "list"): handle_rowgroup_list,
    ("rowgroup", "show"): handle_rowgroup_show,
    ("column", "list"): handle_column_list,
    ("column", "show"): handle_column_show,
    ("page", "list"): handle_page_list,
    ("page", "header"): handle_page_header,
    ("page", "extract"): handle_page_extract,
    ("page", "decode"): handle_page_decode,
    ("show", None): handle_show,
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "path",
        nargs="?",
        help="path to the Parquet file to analyze",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write JSON output to the given file path instead of stdout",
    )
    parser.add_argument(
        "--schema-version",
        action="store_true",
        help="print the JSON $schema URI for this subcommand's output and exit",
    )


def _add_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of items returned (default: all). truncation is reported in the output object.",
    )


def _add_page_selectors(parser: argparse.ArgumentParser) -> None:
    """``--column`` / ``--page-index`` / ``--row-group`` for the singular page
    verbs. ``--column`` and ``--page-index`` are optional at parse time so
    ``--schema-version`` can short-circuit; the run-time check enforces them."""
    parser.add_argument(
        "--column",
        required=False,
        help="column name (dot-joined for nested fields, e.g. 'list.element')",
    )
    parser.add_argument(
        "--page-index",
        type=int,
        default=None,
        help="page index over the full page order (dictionary page first, then "
        "data pages); negative indexes from the end",
    )
    parser.add_argument(
        "--row-group",
        type=int,
        default=None,
        help="row group (0-based); required when the file has multiple row groups",
    )


def build_subcommand_parser() -> argparse.ArgumentParser:
    parser = _JsonErrorParser(prog="parquet-analyzer")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )

    verb_subparsers = parser.add_subparsers(dest="verb", metavar="VERB", required=True)

    # --- file ------------------------------------------------------------
    file_parser = verb_subparsers.add_parser(
        "file", help="file-level inspection (summary, kv metadata, schema, validation)"
    )
    file_subparsers = file_parser.add_subparsers(
        dest="noun", metavar="NOUN", required=True
    )
    file_subparsers.parser_class = _JsonErrorParser  # type: ignore[attr-defined]

    fs = file_subparsers.add_parser(
        "summary", help="compact footer-only summary (row/column counts, sizes)"
    )
    _add_common_options(fs)

    fkv = file_subparsers.add_parser(
        "kv", help="key-value metadata (preserves duplicates and ordering)"
    )
    _add_common_options(fkv)
    _add_limit(fkv)
    fkv.add_argument(
        "--key", help="return only entries matching this key (all matches)"
    )

    fsc = file_subparsers.add_parser("schema", help="schema element list")
    _add_common_options(fsc)

    fv = file_subparsers.add_parser(
        "validate", help="footer-only structural validation"
    )
    _add_common_options(fv)

    # --- rowgroup --------------------------------------------------------
    rg_parser = verb_subparsers.add_parser("rowgroup", help="row-group inspection")
    rg_subparsers = rg_parser.add_subparsers(dest="noun", metavar="NOUN", required=True)
    rg_subparsers.parser_class = _JsonErrorParser  # type: ignore[attr-defined]

    rgl = rg_subparsers.add_parser("list", help="one entry per row group")
    _add_common_options(rgl)
    _add_limit(rgl)

    rgs = rg_subparsers.add_parser(
        "show", help="metadata for a single row group, including its column chunks"
    )
    _add_common_options(rgs)
    rgs.add_argument(
        "--row-group", type=int, required=False, help="row group index (0-based)"
    )

    # --- column ----------------------------------------------------------
    col_parser = verb_subparsers.add_parser("column", help="column-chunk inspection")
    col_subparsers = col_parser.add_subparsers(
        dest="noun", metavar="NOUN", required=True
    )
    col_subparsers.parser_class = _JsonErrorParser  # type: ignore[attr-defined]

    cl = col_subparsers.add_parser(
        "list", help="one entry per (row_group, column_chunk) pair"
    )
    _add_common_options(cl)
    _add_limit(cl)
    cl.add_argument(
        "--row-group",
        type=int,
        default=None,
        help="restrict to this row group (0-based); default: all row groups",
    )

    cs = col_subparsers.add_parser(
        "show", help="per-column-chunk metadata across row groups"
    )
    _add_common_options(cs)
    cs.add_argument(
        "--column",
        required=False,
        help="column name (dot-joined for nested fields, e.g. 'list.element')",
    )
    cs.add_argument(
        "--row-group",
        type=int,
        default=None,
        help="restrict to this row group (0-based); default: all row groups",
    )

    # --- page (page-walk / body-decode escape hatch) ---------------------
    page_parser = verb_subparsers.add_parser(
        "page", help="page-level inspection (list / header / extract / decode)"
    )
    page_subparsers = page_parser.add_subparsers(
        dest="noun", metavar="NOUN", required=True
    )
    page_subparsers.parser_class = _JsonErrorParser  # type: ignore[attr-defined]

    pl = page_subparsers.add_parser(
        "list", help="list a column's (or every column's) pages as stubs"
    )
    _add_common_options(pl)
    _add_limit(pl)
    pl.add_argument(
        "--column",
        default=None,
        help="restrict to this column (dot-joined); default: all columns",
    )
    pl.add_argument(
        "--row-group",
        type=int,
        default=None,
        help="restrict to this row group (0-based); default: all row groups",
    )

    ph = page_subparsers.add_parser("header", help="one page's header fields")
    _add_common_options(ph)
    _add_page_selectors(ph)

    pe = page_subparsers.add_parser(
        "extract", help="one page's raw body bytes (hex / base64 / raw)"
    )
    _add_common_options(pe)
    _add_page_selectors(pe)
    pe.add_argument(
        "--decompress",
        action="store_true",
        help="decompress the body before emitting (page-type aware)",
    )
    pe.add_argument(
        "--as",
        dest="as_format",
        choices=["hex", "base64", "raw"],
        default="hex",
        help="byte encoding; 'raw' writes bytes to stdout/-o with no JSON wrapper",
    )

    pd = page_subparsers.add_parser(
        "decode", help="decode one page's body and project a view"
    )
    _add_common_options(pd)
    _add_limit(pd)
    _add_page_selectors(pd)
    pd.add_argument(
        "--kind",
        required=False,
        choices=["values", "rle-runs", "levels", "statistics"],
        help="which decoded view to project",
    )

    # --- show (path-addressed navigation) --------------------------------
    show_parser = verb_subparsers.add_parser(
        "show",
        help="navigate the file as a tree: show a node and its children as "
        "stubs with paths to descend into",
    )
    show_parser.set_defaults(noun=None)
    _add_common_options(show_parser)
    show_parser.add_argument(
        "navpath",
        nargs="?",
        default="",
        help="navigation path along the row_groups/columns/pages spine, e.g. "
        "'row_groups/0/columns/3'; default: the file root",
    )
    show_parser.add_argument(
        "--walk-pages",
        action="store_true",
        help="allow listing/addressing a column's pages when the file has no "
        "OffsetIndex (reads every page header)",
    )
    show_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="cap how many child stubs are listed (a column can have many "
        "thousands of pages); 0 lists all. Truncation is reported in "
        "_navigation; every child stays addressable by index regardless.",
    )

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _command_label(args: argparse.Namespace) -> str:
    """The ``verb noun`` (or just ``verb`` for a nounless verb like ``show``)
    label used in error messages and ``fix`` commands."""
    return (
        args.verb if getattr(args, "noun", None) is None else f"{args.verb} {args.noun}"
    )


def _validate_required_for_run(args: argparse.Namespace) -> None:
    """Enforce per-subcommand required args that we made optional for the
    sake of ``--schema-version`` (which short-circuits before running)."""
    cmd = _command_label(args)
    if args.path is None:
        raise CliError(
            code="missing_argument",
            message=f"{cmd}: <path> is required",
            fix=f"parquet-analyzer {cmd} <path>",
        )

    if (args.verb, args.noun) == ("rowgroup", "show") and args.row_group is None:
        raise CliError(
            code="missing_argument",
            message="rowgroup show: --row-group is required",
            fix=f"parquet-analyzer rowgroup show {args.path} --row-group 0",
        )

    if (args.verb, args.noun) == ("column", "show") and args.column is None:
        raise CliError(
            code="missing_argument",
            message="column show: --column is required",
            fix=f"parquet-analyzer column show {args.path} --column <name>",
        )

    # Singular page verbs require a column + page index (syntactic; the
    # conditional --row-group rule is enforced in the handler once the file is
    # open and the row-group count is known).
    if args.verb == "page" and args.noun in ("header", "extract", "decode"):
        if args.column is None:
            raise CliError(
                code="missing_argument",
                message=f"page {args.noun}: --column is required",
                fix=f"parquet-analyzer page {args.noun} {args.path} --column <name> "
                "--page-index 0",
            )
        if args.page_index is None:
            raise CliError(
                code="missing_argument",
                message=f"page {args.noun}: --page-index is required",
                fix=f"parquet-analyzer page {args.noun} {args.path} "
                f"--column {args.column} --page-index 0",
            )
    if (args.verb, args.noun) == ("page", "decode") and args.kind is None:
        raise CliError(
            code="missing_argument",
            message="page decode: --kind is required",
            fix=f"parquet-analyzer page decode {args.path} --column {args.column} "
            f"--page-index {args.page_index} --kind values",
        )


def run_subcommand(argv: Sequence[str]) -> int:
    parser = build_subcommand_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.getLevelNamesMapping()[args.log_level.upper()],
        format="%(asctime)s %(name)s [%(threadName)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    key = (args.verb, args.noun)
    if key not in HANDLERS:
        emit_error(
            CliError(
                code="unknown_subcommand",
                message=f"unknown subcommand: {args.verb} {args.noun}",
                fix="parquet-analyzer --help",
            )
        )
        return 2

    if args.schema_version:
        _emit_json({"$schema": _schema_uri(SCHEMAS[key])}, args.output)
        return 0

    try:
        _validate_required_for_run(args)
        HANDLERS[key](args)
        return 0
    except CliError as err:
        emit_error(err)
        return 1
    except FileNotFoundError as exc:
        emit_error(
            CliError(
                code="file_not_found",
                message=str(exc),
                fix=f"check the path and re-run: parquet-analyzer {_command_label(args)} <path>",
            )
        )
        return 1
    except ValueError as exc:
        # ParquetFile raises ValueError on bad magic / unparseable footer.
        # For most subcommands this is fatal; `file validate` catches it
        # upstream and reports it as a finding.
        emit_error(
            CliError(
                code="invalid_parquet_file",
                message=str(exc),
                fix=f"parquet-analyzer file validate {args.path}",
            )
        )
        return 1


# ---------------------------------------------------------------------------
# argv sniffing — pick between legacy and subcommand grammars
# ---------------------------------------------------------------------------


# Global option flags that take a value. Used by the argv pre-scanner so
# tokens like ``parquet-analyzer --log-level DEBUG file.parquet`` correctly
# identify ``file.parquet`` (not ``DEBUG``) as the first positional.
_GLOBAL_OPTIONS_WITH_VALUE: frozenset[str] = frozenset(
    {"--log-level", "-o", "--output"}
)


def is_subcommand_invocation(argv: Sequence[str]) -> bool:
    """Return True if ``argv`` should be dispatched as a verb-noun subcommand.

    The legacy ``parquet-analyzer <path> [--output-mode ...]`` grammar coexists
    with the new subcommand grammar. We disambiguate by looking at the first
    non-flag positional: if it equals one of the verb names, dispatch
    subcommand; otherwise dispatch legacy.

    Legacy-only options (``--output-mode``, ``--html-sections``) force legacy
    dispatch when present. Likewise, the global ``--schema-version`` flag is
    a subcommand-only feature; bare ``--help`` is not enough to decide either
    way, so we fall back to subcommand dispatch (subcommand parser exposes
    the broader grammar in its help text).
    """
    if any(a == "--output-mode" or a.startswith("--output-mode=") for a in argv):
        return False
    if "--html-sections" in argv:
        return False

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            # explicit end-of-options marker; next token is positional
            i += 1
            if i < len(argv):
                return argv[i] in SUBCOMMAND_VERBS
            return False
        if tok in _GLOBAL_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("--") and "=" in tok:
            # `--key=value` consumes one token
            i += 1
            continue
        if tok.startswith("-"):
            # unknown short or long flag (assume zero args)
            i += 1
            continue
        # first non-flag positional
        return tok in SUBCOMMAND_VERBS
    return False
