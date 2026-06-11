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


def render(pf: Any, navpath: str, *, walk_pages: bool, limit: int = 100) -> dict:
    """Render the node at ``navpath`` with its children as path-annotated
    stubs and a ``_navigation`` block. The returned dict has no ``$schema``
    (the CLI handler attaches the ``show`` schema).

    ``limit`` caps how many child stubs are listed (a column can have many
    thousands of pages); ``limit <= 0`` lists all. The ``_navigation`` block
    reports ``children_total`` / ``children_shown`` / ``children_truncated``.
    Truncation only bounds the *listing* — every child remains addressable by
    its index regardless of ``limit``. When a column's pages can't be listed
    without a page-header walk (no OffsetIndex, no ``walk_pages``), ``pages``
    is ``null`` and ``_navigation`` carries a ``walk_required`` / ``reason`` /
    ``hint`` affordance instead.
    """
    node, kind, canonical = resolve(pf, navpath, walk_pages=walk_pages)

    if kind == "column_chunk":
        out = _render_column_show(node, canonical, walk_pages, limit)
        listing = out.pop("_listing", None)
    else:
        out = node.to_json(view="tree", depth=1)
        out.pop("$schema", None)
        if kind == "file":
            listing = _list_children(out, "row_groups", "", limit=limit)
        elif kind == "row_group":
            listing = _list_children(
                out, "columns", canonical, limit=limit, wrappers=node.columns
            )
        else:
            listing = None

    nav: dict[str, Any] = {
        "path": canonical,
        "parent": _parent_path(canonical),
        "kind": kind,
    }
    if listing is not None:
        nav.update(listing)
    out["_navigation"] = nav
    return out


def _render_column_show(cc: Any, base: str, walk_pages: bool, limit: int) -> dict:
    out = _column_chunk_content(cc)
    if cc.has_offset_index or walk_pages:
        dict_json, pages_json = _render_pages(cc, "tree", 0)
        # Page 0 is the dictionary page when present; data pages follow.
        idx = 0
        if dict_json is not None:
            dict_json["_path"] = f"{base}/pages/{idx}"
            idx += 1
        for stub in pages_json:
            stub["_path"] = f"{base}/pages/{idx}"
            idx += 1
        out["dictionary_page"] = dict_json
        # --limit caps the (potentially huge) data-page listing. The
        # dictionary page is a single, always-shown child — not part of the
        # capped listing — so children_* describe the data pages only,
        # keeping children_shown <= limit (as for row groups / columns).
        total = len(pages_json)
        if limit > 0 and total > limit:
            out["pages"] = pages_json[:limit]
            truncated = True
        else:
            out["pages"] = pages_json
            truncated = False
        out["_listing"] = {
            "children_total": total,
            "children_shown": len(out["pages"]),
            "children_truncated": truncated,
        }
    else:
        out["dictionary_page"] = _withheld_dict_stub(cc, base)
        # No OffsetIndex and no --walk-pages: the data-page listing is
        # withheld. `pages` stays cleanly null-or-list (never an object),
        # mirroring `dictionary_page` (null-or-node); the listing block is
        # the single source of listing state and carries the affordance
        # explaining why the listing is empty.
        out["pages"] = None
        out["_listing"] = {
            "children_total": None,
            "children_shown": 0,
            "children_truncated": False,
            "walk_required": True,
            "reason": "no OffsetIndex",
            "hint": (
                f"re-run with '{base}/pages/<n> --walk-pages' to address a "
                "page (reads every page header)"
            ),
        }
    out.update(render_tree_index_children(cc, 0))
    return out


def _withheld_dict_stub(cc: Any, base: str) -> dict | None:
    """The dictionary-page stub on the no-OffsetIndex path. Its extent is
    footer-derivable (see :meth:`ColumnChunk._dictionary_page_extent`), so it
    is shown even when the data-page listing is withheld (reporting ``null``
    here would be indistinguishable from a column that has no dictionary
    page)."""
    extent = cc._dictionary_page_extent()
    if extent is None:
        return None
    return {
        "_kind": "dictionary_page",
        "_location": {"offset": extent[0], "length": extent[1]},
        "_lazy": True,
        "_path": f"{base}/pages/0",
    }


def _list_children(
    out: dict, keyword: str, base: str, *, limit: int, wrappers: Any = None
) -> dict | None:
    """Cap ``out[keyword]`` to ``limit`` stubs, annotate each with its
    canonical descend ``_path`` (and, for columns, the display ``name``), and
    return the listing metadata. ``limit <= 0`` lists all."""
    stubs = out.get(keyword)
    if not isinstance(stubs, list):
        return None
    total = len(stubs)
    if limit > 0 and total > limit:
        stubs = stubs[:limit]
        out[keyword] = stubs
        truncated = True
    else:
        truncated = False
    prefix = f"{base}/" if base else ""
    for i, stub in enumerate(stubs):
        if not isinstance(stub, dict):
            continue
        stub["_path"] = f"{prefix}{keyword}/{i}"
        if wrappers is not None:
            stub["name"] = ".".join(wrappers[i].path)
    return {
        "children_total": total,
        "children_shown": len(stubs),
        "children_truncated": truncated,
    }


def _parent_path(canonical: str) -> str | None:
    if not canonical:
        return None
    parts = canonical.split("/")
    return "/".join(parts[:-2])
