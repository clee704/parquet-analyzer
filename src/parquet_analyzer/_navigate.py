"""Path-addressed navigation for the ``show`` verb.

``show FILE [PATH]`` renders the node at ``PATH`` — its own content fields
plus its immediate children as stubs — and annotates each navigable child
stub with the canonical path to descend into it. You explore the file like a
map, one bounded step at a time, by extending the path. See issue #36.

Paths are **canonical and index-based** along the row-group → column → page
spine:

``row_groups/<i>/columns/<k>/pages/<j>``

Indices are integers; column display names appear as a ``name`` field on the
stub (names can contain dots/slashes, so they are not used in the path).
Every step is bounded: listing a column's pages never forces a page-header
walk — with an OffsetIndex the pages are listed from it, and without one the
listing is withheld behind ``--walk-pages`` (issue #30).
"""

from __future__ import annotations

from typing import Any

from ._tree_json import (
    _column_chunk_content,
    _render_pages,
    render_tree_index_children,
)


class NavigationError(Exception):
    """Raised when a navigation path cannot be resolved. Carries the
    structured fields the CLI maps onto its JSON error contract."""

    def __init__(self, code: str, message: str, fix: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fix = fix


_SPINE = ("row_groups", "columns", "pages")


def _parse_navpath(navpath: str) -> list[tuple[str, str]]:
    """Split a navpath into ``(keyword, value)`` pairs.

    ``""`` → ``[]``; ``"row_groups/0/columns/3"`` →
    ``[("row_groups", "0"), ("columns", "3")]``.
    """
    trimmed = navpath.strip("/")
    if not trimmed:
        return []
    parts = trimmed.split("/")
    if len(parts) % 2 != 0:
        raise NavigationError(
            code="invalid_path",
            message=(
                f"path {navpath!r} has an unpaired segment; expected "
                "keyword/value pairs like 'row_groups/0/columns/3'"
            ),
            fix="parquet-analyzer show <file>",
        )
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]


def _parse_index(keyword: str, value: str, base_path: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise NavigationError(
            code="invalid_path",
            message=f"{keyword} index must be an integer, got {value!r}",
            fix=f"parquet-analyzer show <file> {base_path}".rstrip(),
        ) from exc


def resolve(pf: Any, navpath: str, *, walk_pages: bool) -> tuple[Any, str, str]:
    """Resolve ``navpath`` to ``(wrapper, kind, canonical_path)``.

    ``kind`` is one of ``file`` / ``row_group`` / ``column_chunk`` / ``page``.
    Raises :class:`NavigationError` for an out-of-range index, an unknown or
    misplaced keyword, or a page step on a column without an OffsetIndex when
    ``walk_pages`` is False.
    """
    node: Any = pf
    kind = "file"
    canonical: list[str] = []

    for keyword, value in _parse_navpath(navpath):
        base = "/".join(canonical)
        if kind == "file" and keyword == "row_groups":
            idx = _parse_index(keyword, value, base)
            n = pf.num_row_groups
            if not 0 <= idx < n:
                raise _out_of_range("row group", idx, n, base or "<file>")
            node, kind = pf.row_groups[idx], "row_group"
        elif kind == "row_group" and keyword == "columns":
            idx = _parse_index(keyword, value, base)
            cols = node.columns
            if not 0 <= idx < len(cols):
                raise _out_of_range("column", idx, len(cols), base)
            node, kind = cols[idx], "column_chunk"
        elif kind == "column_chunk" and keyword == "pages":
            idx = _parse_index(keyword, value, base)
            if not node.has_offset_index and not walk_pages:
                raise NavigationError(
                    code="walk_required",
                    message=(
                        "this column has no OffsetIndex, so addressing a page "
                        "requires walking the page headers"
                    ),
                    fix=f"parquet-analyzer show <file> {base}/pages/{idx} --walk-pages",
                )
            n = node.num_pages
            if not 0 <= idx < n:
                raise _out_of_range("page", idx, n, base)
            node, kind = node.page(idx), "page"
        else:
            raise NavigationError(
                code="invalid_path",
                message=f"cannot descend into {keyword!r} from a {kind}",
                fix=f"parquet-analyzer show <file> {base}".rstrip(),
            )
        canonical += [keyword, str(idx)]

    return node, kind, "/".join(canonical)


def _out_of_range(what: str, idx: int, total: int, parent: str) -> NavigationError:
    return NavigationError(
        code=f"{what.replace(' ', '_')}_out_of_range",
        message=f"{what} {idx} requested but only {total} exist at {parent or '<file>'}",
        fix=f"parquet-analyzer show <file> {parent}".rstrip(),
    )


def render(pf: Any, navpath: str, *, walk_pages: bool) -> dict:
    """Render the node at ``navpath`` with its children as path-annotated
    stubs and a ``_navigation`` block. The returned dict has no ``$schema``
    (the CLI handler attaches the ``show`` schema)."""
    node, kind, canonical = resolve(pf, navpath, walk_pages=walk_pages)

    if kind == "column_chunk":
        out = _render_column_show(node, canonical, walk_pages)
    else:
        out = node.to_json(view="tree", depth=1)
        out.pop("$schema", None)
        if kind == "file":
            _annotate_children(out.get("row_groups"), "row_groups", "")
        elif kind == "row_group":
            _annotate_children(out.get("columns"), "columns", canonical, node.columns)

    out["_navigation"] = {
        "path": canonical,
        "parent": _parent_path(canonical),
        "kind": kind,
    }
    return out


def _render_column_show(cc: Any, base: str, walk_pages: bool) -> dict:
    out = _column_chunk_content(cc)
    if cc.has_offset_index or walk_pages:
        dict_json, pages_json = _render_pages(cc, "tree", 0)
        out["dictionary_page"] = dict_json
        out["pages"] = pages_json
        idx = 0
        if dict_json is not None:
            dict_json["_path"] = f"{base}/pages/{idx}"
            idx += 1
        for stub in pages_json:
            stub["_path"] = f"{base}/pages/{idx}"
            idx += 1
    else:
        out["dictionary_page"] = None
        out["pages"] = {
            "_walk_required": True,
            "reason": "no OffsetIndex",
            "hint": (
                f"re-run with '{base}/pages/<n> --walk-pages' to address a "
                "page (reads every page header)"
            ),
        }
    out.update(render_tree_index_children(cc, 0))
    return out


def _annotate_children(
    stubs: Any, keyword: str, base: str, wrappers: Any = None
) -> None:
    """Attach the canonical descend ``_path`` (and, for columns, the display
    ``name``) to each child stub in place."""
    if not isinstance(stubs, list):
        return
    prefix = f"{base}/" if base else ""
    for i, stub in enumerate(stubs):
        if not isinstance(stub, dict):
            continue
        stub["_path"] = f"{prefix}{keyword}/{i}"
        if wrappers is not None:
            stub["name"] = ".".join(wrappers[i].path)


def _parent_path(canonical: str) -> str | None:
    if not canonical:
        return None
    parts = canonical.split("/")
    return "/".join(parts[:-2])
