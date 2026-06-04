"""Physical-layout children builder for the v0 ``layout`` view.

This module owns the per-file ``children[]`` arrangement: it walks every
column chunk to gather data regions and side structures, identifies the
file-level system structures (header magic, footer, footer length,
trailer magic), sorts everything by ``_offset``, and gap-fills any
byte ranges not covered by an existing node with synthetic ``unknown``
leaves.

The actual rendering of each child is done by :mod:`._tree_json` — this
module only assembles the *list* of physical children.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from . import _tree_json as _tj

if TYPE_CHECKING:
    from .parquet_file import ParquetFile

__all__ = ["build_file_layout_children"]


def build_file_layout_children(pf: "ParquetFile") -> List[Any]:
    """Return the file's layout-view children, sorted by ``_offset`` with
    ``unknown`` synthetic leaves filling any gaps.

    The list is independent of depth — each item is either a wrapper (for
    the footer / row-group / column-chunk pieces that live inside the
    footer) or a synthetic dict node (for column_chunk_data_region,
    offset_index, column_index, bloom_filter_header, header_magic,
    footer_length, trailer_magic, unknown).

    The renderer in :mod:`._tree_json` decides per-call whether each item
    is stubbed or materialized based on the current depth.
    """
    items: list[Any] = []

    # File-level system structures (always present).
    items.append(_tj._header_magic_node(pf))
    items.append(_tj._footer_node(pf))
    items.append(_tj._footer_length_node(pf))
    items.append(_tj._trailer_magic_node(pf))

    # Per-column-chunk physical regions. Each column may contribute up
    # to four physical regions: the data region (present when the column
    # has on-disk page bytes), the offset index, the column index, and the
    # bloom filter header.
    for rg_idx, rg in enumerate(pf.row_groups):
        for cc in rg.columns:
            if _tj._has_physical_data_region(cc):
                items.append(_tj._data_region_node(cc, rg_idx))
            if cc.offset_index_offset is not None:
                items.append(_tj._offset_index_node(cc))
            if cc.column_index_offset is not None:
                items.append(_tj._column_index_node(cc))
            if cc.bloom_filter_offset is not None:
                items.append(_tj._bloom_filter_header_node(cc))

    # Sort by offset; the loop above adds items in row-group order, but
    # indexes/bloom filters may live anywhere in the file (typically
    # after all column data, before the footer) so we sort by absolute
    # offset to produce the file's true byte layout.
    items.sort(key=_tj._offset_of)

    # Gap-fill: any byte range in [0, file_size) not covered by a sibling
    # becomes an ``unknown`` node. The schema doc requires children to be
    # contiguous in the layout view (per docs/tree-schema.md §"unknown").
    return _gap_fill(items, pf._file_size)


def _gap_fill(items: list[Any], file_size: int) -> list[Any]:
    """Walk sorted children and emit ``unknown`` nodes for any gaps.

    ``items`` must be sorted by ``_offset``. Enforces the layout-view
    contract that siblings are non-overlapping and contained in
    ``[0, file_size)``: a positive-length node starting before the running
    cursor (an overlap) or extending past ``file_size`` raises
    ``ValueError`` rather than emitting contract-violating JSON. Returns a
    new list with ``unknown`` nodes interleaved over the gaps.
    """
    out: list[Any] = []
    cursor = 0
    for item in items:
        offset = _tj._offset_of(item)
        length = _tj._length_of(item)
        if length > 0 and offset < cursor:
            raise ValueError(
                f"overlapping layout nodes: {_tj._kind_of(item)} at offset "
                f"{offset} starts before the end of the previous node ({cursor})"
            )
        if offset + length > file_size:
            raise ValueError(
                f"layout node {_tj._kind_of(item)} at offset {offset} "
                f"(length {length}) extends past file_size {file_size}"
            )
        if offset > cursor:
            out.append(_unknown_node(cursor, offset - cursor))
        out.append(item)
        cursor = max(cursor, offset + length)
    if cursor < file_size:
        out.append(_unknown_node(cursor, file_size - cursor))
    return out


def _unknown_node(offset: int, length: int) -> dict:
    return {
        "_kind": "unknown",
        "_offset": offset,
        "_length": length,
    }
