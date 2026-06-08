"""Tree-node serializer for the v0 ``tree`` and ``layout`` views.

This module implements ``to_json(view, depth)`` for every wrapper in
:mod:`parquet_analyzer.parquet_file`. It is intentionally private; the
public surface is the ``to_json`` method on each wrapper.

Layout: this module owns the per-kind materialization catalog (what
content fields and children every node emits) and the depth/stub
walker. The companion :mod:`._layout` owns physical-layout construction
(file-level children sorted by ``_offset``, ``column_chunk_data_region``
synthesis, ``unknown`` gap-fill). Splitting at that boundary keeps the
serializer free of layout-arithmetic and lets future layout-specific
work extend ``_layout`` without touching the catalog.

The kind catalog mirrors ``docs/tree-schema.md`` (v0). Field names,
field types, child shapes, and ``_lazy`` placement are all in sync with
the doc — when one moves, the other must move too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

from ._core import (
    json_encode,
    segment_to_json,
    _find_field,
    json_safe_stat_value as _json_safe_stat_value,
)

if TYPE_CHECKING:
    from .parquet_file import ColumnChunk, ParquetFile, RowGroup

__all__ = ["to_json_root"]


SCHEMA_URI_TREE = "parquet-analyzer/v3/tree"
SCHEMA_URI_LAYOUT = "parquet-analyzer/v3/layout"

# Kinds where materialization requires page-header parsing or extra
# thrift parsing beyond the footer parse — these carry ``_lazy: true``
# when emitted as stubs. See docs/tree-schema.md §"_lazy: true".
#
# ``data_page`` is the generic, version-agnostic kind used for a data
# page at the stub level (its ``data_page_v1`` / ``data_page_v2`` version
# is only known once the header is read, i.e. on materialization).
LAZY_KINDS = frozenset(
    {
        "dictionary_page",
        "data_page",
        "data_page_v1",
        "data_page_v2",
        "offset_index",
        "column_index",
        "bloom_filter_header",
        # Body-layer kinds (#21): materializing a page's body requires reading
        # and decoding the body bytes, beyond the page-header parse.
        "rep_block",
        "def_block",
        "values_block",
        "plain_values",
        "dict_indices",
    }
)

Depth = Union[int, str]


def to_json_root(node: Any, view: str, depth: Depth) -> dict:
    """Top-level entry: attach ``$schema`` and serialize the node tree.

    Always emits ``$schema`` (response-shape URI), even when the root is
    a stub (``depth == 0``).
    """
    _validate_view(view)
    _validate_depth(depth)
    out: dict[str, Any] = {
        "$schema": SCHEMA_URI_TREE if view == "tree" else SCHEMA_URI_LAYOUT,
    }
    out.update(_render(node, view, depth))
    return out


def _validate_view(view: str) -> None:
    if view not in ("tree", "layout"):
        raise ValueError(f"view must be 'tree' or 'layout', got {view!r}")


def _validate_depth(depth: Depth) -> None:
    if isinstance(depth, str):
        if depth != "all":
            raise ValueError(f"depth string must be 'all', got {depth!r}")
        return
    if not isinstance(depth, int) or depth < 0:
        raise ValueError(f"depth must be a non-negative int or 'all', got {depth!r}")


def _decr(depth: Depth) -> Depth:
    return "all" if depth == "all" else depth - 1  # type: ignore[operator]


def _is_stub_level(depth: Depth) -> bool:
    return depth == 0


# ---------------------------------------------------------------------------
# Render dispatch
# ---------------------------------------------------------------------------


def _render(node: Any, view: str, depth: Depth) -> dict:
    kind = _kind_of(node)
    if _is_stub_level(depth):
        return _stub(node, kind)
    renderer = _RENDERERS[kind]
    return renderer(node, view, _decr(depth))


def _location(offset: int, length: int) -> dict:
    """The ``_location`` system field carried by every emitted tree node:
    ``{"offset": <file byte offset>, "length": <byte length>}``.

    ``_location`` always describes **real file bytes** (the range you could
    ``dd`` / ``xxd`` out of the file). Inner keys are plain (no ``_`` prefix)
    — the ``_`` namespace distinguishes framework fields from kind-specific
    content *on a node*, and inside ``_location`` every key is framework
    content, so the prefix carries no information (matching the plain-key
    convention of ``_value`` sub-dicts). A future body-layer revision extends
    this object for sub-nodes that live inside a compressed region.
    """
    return {"offset": offset, "length": length}


def _stub(node: Any, kind: str) -> dict:
    out: dict[str, Any] = {
        "_kind": kind,
        "_location": _location(_offset_of(node), _length_of(node)),
    }
    if kind in LAZY_KINDS:
        out["_lazy"] = True
    return out


def _system_fields(node: Any, kind: str) -> dict:
    """Materialized form's universal fields. Differs from ``_stub`` only in
    NOT carrying ``_lazy`` (materialization paid the I/O)."""
    return {
        "_kind": kind,
        "_location": _location(_offset_of(node), _length_of(node)),
    }


def _kind_of(node: Any) -> str:
    # Synthetic dict nodes (from _layout) carry their kind as a key.
    if isinstance(node, dict):
        return node["_kind"]
    return node._kind  # wrappers


def _offset_of(node: Any) -> int:
    if isinstance(node, dict):
        return node["_offset"]
    return node._offset


def _length_of(node: Any) -> int:
    if isinstance(node, dict):
        return node["_length"]
    return node._length


# ---------------------------------------------------------------------------
# Per-kind renderers
# ---------------------------------------------------------------------------


def _render_file(pf: "ParquetFile", view: str, child_depth: Depth) -> dict:
    out = _system_fields(pf, "file")
    out["path"] = pf.path
    if view == "tree":
        out["header_magic"] = _render(_header_magic_node(pf), view, child_depth)
        out["row_groups"] = [_render(rg, view, child_depth) for rg in pf.row_groups]
        out["footer"] = _render(_footer_node(pf), view, child_depth)
        out["footer_length"] = _render(_footer_length_node(pf), view, child_depth)
        out["trailer_magic"] = _render(_trailer_magic_node(pf), view, child_depth)
    else:
        # Layout view: physical children sorted by _offset, with synthetic
        # column_chunk_data_region / opaque-branch nodes interleaved and
        # unknown leaves filling gaps. The whole arrangement lives in
        # _layout; here we just render each item.
        from ._layout import build_file_layout_children

        children = build_file_layout_children(pf)
        out["children"] = [_render(c, view, child_depth) for c in children]
    return out


def _render_header_magic(node: dict, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(node, "header_magic")
    out["_value"] = "PAR1"
    return out


def _render_trailer_magic(node: dict, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(node, "trailer_magic")
    out["_value"] = "PAR1"
    return out


def _render_footer_length(node: dict, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(node, "footer_length")
    out["_value"] = int(node["_pf"].footer_size)
    return out


def _render_footer(node: dict, view: str, child_depth: Depth) -> dict:
    pf = node["_pf"]
    out = _system_fields(node, "footer")
    out["version"] = pf._footer_thrift.version
    out["num_rows"] = pf._footer_thrift.num_rows
    out["created_by"] = pf._footer_thrift.created_by
    # Named children, identical in tree and layout (footer's children are
    # all physically contained inside the footer).
    if view == "tree":
        out["schema"] = _render(_schema_node(pf), view, child_depth)
        out["kv_metadata"] = _render(_kv_metadata_node(pf), view, child_depth)
        out["row_groups"] = [_render(rg, view, child_depth) for rg in pf.row_groups]
    else:
        # Layout view: footer.children is a single array sorted by offset.
        items: list[Any] = [_schema_node(pf), _kv_metadata_node(pf)]
        items.extend(pf.row_groups)
        items.sort(key=_offset_of)
        out["children"] = [_render(c, view, child_depth) for c in items]
    return out


def _render_schema(node: dict, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(node, "schema")
    out["_value"] = _make_json_safe(node["_value"])
    return out


def _render_kv_metadata(node: dict, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(node, "kv_metadata")
    out["_value"] = _make_json_safe(node["_value"])
    return out


def _render_row_group(rg: "RowGroup", view: str, child_depth: Depth) -> dict:
    out = _system_fields(rg, "row_group")
    out["num_rows"] = rg.num_rows
    out["total_byte_size"] = rg.total_byte_size
    out["total_compressed_size"] = sum(cc.total_compressed_size for cc in rg.columns)
    out["ordinal"] = rg._t.ordinal
    # Tree and layout both expose columns as children (all physically
    # contained inside the row-group thrift, which itself lives inside
    # the footer).
    if view == "tree":
        out["columns"] = [_render(cc, view, child_depth) for cc in rg.columns]
    else:
        # Already in offset order (the footer-segment walk preserves it).
        out["children"] = [_render(cc, view, child_depth) for cc in rg.columns]
    return out


def _column_chunk_content(cc: "ColumnChunk") -> dict:
    """The column chunk's own scalar fields (footer-derived, cheap) — shared
    by the tree/layout renderers and the ``show`` navigation verb, which
    each attach their own children."""
    out = _system_fields(cc, "column_chunk")
    out["path"] = list(cc.path)
    out["path_display"] = ".".join(cc.path)
    out["type"] = cc.type
    out["codec"] = cc.codec
    out["encodings"] = list(cc.encodings)
    out["num_values"] = cc.num_values
    out["compressed_size"] = cc.total_compressed_size
    out["uncompressed_size"] = cc.total_uncompressed_size
    out["data_page_offset"] = cc.data_page_offset
    out["dictionary_page_offset"] = cc.dictionary_page_offset
    out["file_offset"] = cc._t.file_offset
    out["statistics"] = _column_chunk_statistics(cc)
    return out


def render_tree_index_children(cc: "ColumnChunk", child_depth: Depth) -> dict:
    """The tree-view ``offset_index`` / ``column_index`` / ``bloom_filter``
    children of a column chunk (each ``None`` when absent). Shared by the
    tree renderer and ``show``."""
    return {
        "offset_index": (
            _render(_offset_index_node(cc), "tree", child_depth)
            if cc.offset_index_offset is not None
            else None
        ),
        "column_index": (
            _render(_column_index_node(cc), "tree", child_depth)
            if cc.column_index_offset is not None
            else None
        ),
        "bloom_filter": (
            _render(_bloom_filter_header_node(cc), "tree", child_depth)
            if cc.bloom_filter_offset is not None
            else None
        ),
    }


def _render_column_chunk(cc: "ColumnChunk", view: str, child_depth: Depth) -> dict:
    out = _column_chunk_content(cc)
    if view == "tree":
        # Tree view: page kinds + opaque branches as named children; no
        # data_region (per docs/tree-schema.md). null when absent.
        out["dictionary_page"], out["pages"] = _render_pages(cc, view, child_depth)
        out.update(render_tree_index_children(cc, child_depth))
    else:
        # Layout view: refs (stubs) to the nodes' physical positions; no
        # children array (the data lives elsewhere in the file tree).
        out["data_region_ref"] = (
            _ref_stub(_data_region_node(cc, _row_group_index_of(cc)))
            if _has_physical_data_region(cc)
            else None
        )
        out["offset_index_ref"] = (
            _ref_stub(_offset_index_node(cc))
            if cc.offset_index_offset is not None
            else None
        )
        out["column_index_ref"] = (
            _ref_stub(_column_index_node(cc))
            if cc.column_index_offset is not None
            else None
        )
        out["bloom_filter_ref"] = (
            _ref_stub(_bloom_filter_header_node(cc))
            if cc.bloom_filter_offset is not None
            else None
        )
    return out


def _render_data_region(node: dict, view: str, child_depth: Depth) -> dict:
    out = _system_fields(node, "column_chunk_data_region")
    cc = node["_cc"]
    out["chunk_ref"] = _ref_stub_from_wrapper(cc, "column_chunk")
    out["row_group_index"] = node["_row_group_index"]
    out["column_position_in_row_group"] = node["_column_position_in_row_group"]
    out["dictionary_page"], out["pages"] = _render_pages(cc, view, child_depth)
    return out


def _render_dictionary_page(p: Any, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(p, "dictionary_page")
    out["page_type"] = "DICTIONARY_PAGE"
    h = p._t.dictionary_page_header
    out["encoding"] = _enum_name(h.encoding, _ENC_NAMES) if h is not None else None
    out["num_values"] = h.num_values if h is not None else 0
    out["uncompressed_size"] = p._t.uncompressed_page_size
    out["compressed_size"] = p._t.compressed_page_size
    # is_compressed is V2-only per docs/tree-schema.md; dictionary pages
    # have no V2 form so always null.
    out["is_compressed"] = None
    out["crc"] = p._t.crc
    return out


def _render_data_page_v1(p: Any, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(p, "data_page_v1")
    out["page_type"] = "DATA_PAGE"
    h = p._t.data_page_header
    out["encoding"] = _enum_name(h.encoding, _ENC_NAMES) if h is not None else None
    out["num_values"] = h.num_values if h is not None else 0
    out["uncompressed_size"] = p._t.uncompressed_page_size
    out["compressed_size"] = p._t.compressed_page_size
    out["definition_level_encoding"] = (
        _enum_name(h.definition_level_encoding, _ENC_NAMES) if h is not None else None
    )
    out["repetition_level_encoding"] = (
        _enum_name(h.repetition_level_encoding, _ENC_NAMES) if h is not None else None
    )
    out["statistics"] = _page_statistics(h, p)
    out["crc"] = p._t.crc
    if _view == "tree":
        out.update(_render_page_body(p, _child_depth))
    return out


def _render_data_page_v2(p: Any, _view: str, _child_depth: Depth) -> dict:
    out = _system_fields(p, "data_page_v2")
    out["page_type"] = "DATA_PAGE_V2"
    h = p._t.data_page_header_v2
    out["encoding"] = _enum_name(h.encoding, _ENC_NAMES) if h is not None else None
    out["num_values"] = h.num_values if h is not None else 0
    out["num_nulls"] = h.num_nulls if h is not None else 0
    out["num_rows"] = h.num_rows if h is not None else 0
    out["is_compressed"] = h.is_compressed if h is not None else None
    out["uncompressed_size"] = p._t.uncompressed_page_size
    out["compressed_size"] = p._t.compressed_page_size
    out["definition_levels_byte_length"] = (
        h.definition_levels_byte_length if h is not None else 0
    )
    out["repetition_levels_byte_length"] = (
        h.repetition_levels_byte_length if h is not None else 0
    )
    out["statistics"] = _page_statistics(h, p)
    out["crc"] = p._t.crc
    if _view == "tree":
        out.update(_render_page_body(p, _child_depth))
    return out


# ---------------------------------------------------------------------------
# Body-layer kinds (#21) — tree view only
#
# A materialized data page's body decomposes into its level streams and its
# values section, each surfaced in the encoding's own structure. These nodes
# appear only in the tree view: in the layout view the page's bytes tile its
# data region at the page granularity, and the compressed V1 sub-blocks would
# share one on-disk range (no physical tiling), so the body stays opaque there.
# ---------------------------------------------------------------------------


def _body_location(extent: Any) -> dict:
    """``_location`` for a body section, from its :class:`BodyExtent`. Plain
    ``{offset, length}`` when the bytes are directly on disk; the compressed
    form (``compression_codec`` + decompressed coordinates) when the section
    lives inside a compressed region."""
    loc: dict[str, Any] = {"offset": extent.offset, "length": extent.length}
    if extent.compression_codec is not None:
        loc["compression_codec"] = extent.compression_codec
        loc["offset_uncompressed"] = extent.offset_uncompressed
        loc["length_uncompressed"] = extent.length_uncompressed
    return loc


def _render_run(run: Any) -> dict:
    """A single RLE/bit-packed run as a plain content dict (not a node).
    ``RleRun`` → ``{kind: "rle", value, length}``; ``BitPackedRun`` →
    ``{kind: "bit_packed", length, values}``."""
    if hasattr(run, "value"):  # RleRun
        return {"kind": "rle", "value": run.value, "length": run.length}
    return {"kind": "bit_packed", "length": run.length, "values": list(run.values)}


def _render_level_block(stream: Any, extent: Any, kind: str, stub: bool) -> dict:
    """A ``rep_block`` / ``def_block`` leaf from an
    :class:`~parquet_analyzer.decoders.RleBitPackedStream`. The runs are a
    content-field array (not child nodes), so the block is a leaf with a
    ``_value`` of the expanded per-value levels."""
    out: dict[str, Any] = {"_kind": kind, "_location": _body_location(extent)}
    if stub:
        out["_lazy"] = True
        return out
    out["bit_width"] = stream.bit_width
    out["runs"] = [_render_run(r) for r in stream.runs]
    out["_value"] = list(stream.values)
    return out


def _render_values_block(decoded: Any, extent: Any, stub: bool) -> dict:
    """The ``values_block`` branch: a ``plain_values`` leaf for a PLAIN page,
    or a ``dict_indices`` leaf for a dictionary page (the resolved values are
    not in the data page, so they are not a tree node — use
    ``Page.physical_values()``)."""
    out: dict[str, Any] = {
        "_kind": "values_block",
        "_location": _body_location(extent),
    }
    if stub:
        out["_lazy"] = True
        return out
    section = decoded.values
    from .parquet_file import PlainValues as _PlainValues

    if isinstance(section, _PlainValues):
        out["plain_values"] = {
            "_kind": "plain_values",
            "_location": _body_location(extent),
            "_value": _make_json_safe(list(section.values)),
        }
    else:
        # RleBitPackedStream of dictionary indices.
        out["dict_indices"] = {
            "_kind": "dict_indices",
            "_location": _body_location(extent),
            "bit_width": section.bit_width,
            "runs": [_render_run(r) for r in section.runs],
            "_value": list(section.values),
        }
    return out


def _values_block_error(p: Any, exc: Any) -> dict:
    """An opaque ``values_block`` for a page whose body could not be decoded
    (an out-of-scope encoding or codec). Carries the page body's on-disk
    region as ``_location`` and an ``_error`` describing why, so a
    ``depth='all'`` render of a file with unsupported encodings does not
    fail — the undecodable body is honestly marked instead."""
    loc: dict[str, Any] = {
        "offset": p.body_offset,
        "length": p._t.compressed_page_size,
    }
    if p._cc.codec != "UNCOMPRESSED":
        loc["compression_codec"] = p._cc.codec
    return {
        "_kind": "values_block",
        "_location": loc,
        "_error": {"code": exc.code, "message": str(exc)},
    }


def _render_page_body(p: Any, child_depth: Depth) -> dict:
    """The tree-view body children of a materialized data page:
    ``repetition_levels`` (``rep_block`` | null), ``definition_levels``
    (``def_block`` | null), and ``values`` (``values_block``). Decodes the
    page body (the page's ``_lazy`` cost). On an undecodable body, the
    ``values`` child is an opaque error ``values_block`` and the level
    children are null."""
    from .parquet_file import PageDecodeError

    try:
        decoded = p.decode()
    except PageDecodeError as exc:
        return {
            "repetition_levels": None,
            "definition_levels": None,
            "values": _values_block_error(p, exc),
        }
    stub = _is_stub_level(child_depth)
    rep = decoded.repetition_levels
    df = decoded.definition_levels
    return {
        "repetition_levels": (
            _render_level_block(
                rep, decoded.repetition_levels_extent, "rep_block", stub
            )
            if rep is not None
            else None
        ),
        "definition_levels": (
            _render_level_block(df, decoded.definition_levels_extent, "def_block", stub)
            if df is not None
            else None
        ),
        "values": _render_values_block(decoded, decoded.values_extent, stub),
    }


# in v0), but materialization must pay the underlying thrift read so the
# cost is observable per docs/tree-schema.md.
_OPAQUE_READ_METHODS = {
    "offset_index": "_read_offset_index",
    "column_index": "_read_column_index",
    "bloom_filter_header": "_read_bloom_filter_header",
}


def _render_opaque_branch(node: dict, _view: str, _child_depth: Depth) -> dict:
    kind = node["_kind"]
    getattr(node["_cc"], _OPAQUE_READ_METHODS[kind])()  # observable cost-payment
    return _system_fields(node, kind)


def _render_unknown(node: dict, _view: str, _child_depth: Depth) -> dict:
    # unknown is a leaf without _value per docs/tree-schema.md.
    return _system_fields(node, "unknown")


_RENDERERS: dict[str, Callable[[Any, str, Depth], dict]] = {
    "file": _render_file,
    "header_magic": _render_header_magic,
    "trailer_magic": _render_trailer_magic,
    "footer_length": _render_footer_length,
    "footer": _render_footer,
    "schema": _render_schema,
    "kv_metadata": _render_kv_metadata,
    "row_group": _render_row_group,
    "column_chunk": _render_column_chunk,
    "column_chunk_data_region": _render_data_region,
    "dictionary_page": _render_dictionary_page,
    "data_page_v1": _render_data_page_v1,
    "data_page_v2": _render_data_page_v2,
    "offset_index": _render_opaque_branch,
    "column_index": _render_opaque_branch,
    "bloom_filter_header": _render_opaque_branch,
    "unknown": _render_unknown,
}


# ---------------------------------------------------------------------------
# Node constructors (build dict-form synthetic nodes whose _kind / _offset /
# _length we already know)
# ---------------------------------------------------------------------------


def _header_magic_node(pf: "ParquetFile") -> dict:
    return {"_kind": "header_magic", "_offset": 0, "_length": 4}


def _trailer_magic_node(pf: "ParquetFile") -> dict:
    return {
        "_kind": "trailer_magic",
        "_offset": pf._file_size - 4,
        "_length": 4,
    }


def _footer_length_node(pf: "ParquetFile") -> dict:
    return {
        "_kind": "footer_length",
        "_offset": pf._file_size - 8,
        "_length": 4,
        "_pf": pf,
    }


def _footer_node(pf: "ParquetFile") -> dict:
    # Synthetic dict node for "the footer at this byte range" — the
    # actual content is fetched in _render_footer via the _pf back-ref.
    return {
        "_kind": "footer",
        "_offset": pf._footer_offset,
        "_length": pf.footer_size,
        "_pf": pf,
    }


def _schema_node(pf: "ParquetFile") -> dict:
    seg = _find_field(pf._footer_segment, "schema")
    if seg is None:
        # Schema is mandatory per the parquet spec; a footer missing it is
        # an internal inconsistency, so fail loudly rather than fabricate a
        # zero-length node.
        raise ValueError("footer segment missing mandatory schema field")
    return {
        "_kind": "schema",
        "_offset": seg["offset"],
        "_length": seg["length"],
        "_value": segment_to_json(seg),
    }


def _kv_metadata_node(pf: "ParquetFile") -> dict:
    seg = _find_field(pf._footer_segment, "key_value_metadata")
    if seg is None:
        return {
            "_kind": "kv_metadata",
            # Place at footer end with zero length — preserves layout-view
            # sort stability while signaling "no bytes here."
            "_offset": pf._footer_offset + pf.footer_size,
            "_length": 0,
            "_value": [],
        }
    # Use the API-side kv_metadata for a cleaner dict shape (segment_to_json
    # produces the right structure, but pf.kv_metadata is already (key, value)
    # pairs from thrift — same data, simpler).
    return {
        "_kind": "kv_metadata",
        "_offset": seg["offset"],
        "_length": seg["length"],
        "_value": [{"key": k, "value": v} for k, v in pf.kv_metadata],
    }


def _has_physical_data_region(cc: "ColumnChunk") -> bool:
    """Whether the column chunk has on-disk page bytes.

    True iff it has a truthy dictionary-page or data-page offset. A 0-row
    column can have neither — ``data_page_offset`` is the required-field
    sentinel ``0`` and ``dictionary_page_offset`` is ``None`` — in which
    case there is no ``column_chunk_data_region`` to place (placing a
    zero-length region at offset 0 would overlap ``header_magic``).
    """
    return bool(cc.dictionary_page_offset) or bool(cc.data_page_offset)


def _data_region_node(cc: "ColumnChunk", row_group_index: int) -> dict:
    dict_offset = cc.dictionary_page_offset
    data_offset = cc.data_page_offset
    # ``data_page_offset`` is a required thrift field, but pyarrow writes
    # 0 for a 0-row column (no data page); ``dictionary_page_offset`` is
    # optional. A real page can never sit at byte 0 (PAR1 occupies
    # [0, 4)), so the region starts at the first page offset that is > 0.
    # Callers must gate on ``_has_physical_data_region`` first, which
    # guarantees at least one such offset.
    page_offsets = [o for o in (dict_offset, data_offset) if o]
    start = min(page_offsets)
    return {
        "_kind": "column_chunk_data_region",
        "_offset": start,
        "_length": cc.total_compressed_size,
        "_cc": cc,
        "_row_group_index": row_group_index,
        "_column_position_in_row_group": _column_position(cc),
    }


def _offset_index_node(cc: "ColumnChunk") -> dict:
    return {
        "_kind": "offset_index",
        "_offset": cc.offset_index_offset,
        "_length": cc._t.offset_index_length or 0,
        "_cc": cc,
    }


def _column_index_node(cc: "ColumnChunk") -> dict:
    return {
        "_kind": "column_index",
        "_offset": cc.column_index_offset,
        "_length": cc._t.column_index_length or 0,
        "_cc": cc,
    }


def _bloom_filter_header_node(cc: "ColumnChunk") -> dict:
    md = cc._t.meta_data
    return {
        "_kind": "bloom_filter_header",
        "_offset": md.bloom_filter_offset,
        "_length": md.bloom_filter_length or 0,
        "_cc": cc,
    }


def _split_pages(cc: "ColumnChunk") -> tuple[Any, list]:
    """Split a column chunk's pages into ``(dictionary_page, data_pages)``.

    ``cc.pages()`` yields the dictionary page (when present) followed by the
    data pages — including for a 0-row column, whose dictionary page is read
    unconditionally. Returns ``(None, [...])`` when the column has no
    dictionary page.
    """
    dict_page = None
    data_pages: list = []
    for p in cc.pages():
        if _kind_of(p) == "dictionary_page":
            dict_page = p
        else:
            data_pages.append(p)
    return dict_page, data_pages


def _render_pages(
    cc: "ColumnChunk", view: str, child_depth: Depth
) -> tuple[dict | None, list[dict]]:
    """Return ``(dictionary_page_json, [page_json, ...])`` for a column
    chunk's pages, shared by the tree ``column_chunk`` and the layout
    ``column_chunk_data_region`` renderers.

    When the pages are only being stubbed (``child_depth == 0``), the
    OffsetIndex-derived stubs (:meth:`ColumnChunk.page_stubs`) are used
    when available — enumerating the pages without reading any per-page
    header, which is what keeps *listing* a column's pages cheap (#30).
    When the column has no OffsetIndex, the only source of page extents is
    a full header walk, so the stubs fall back to that walk. (The
    no-OffsetIndex "walk required" affordance — listing without any walk —
    is introduced with the ``show`` navigation verb; the tree/layout
    serializer here keeps the walk fallback.)

    When descending into the pages (``child_depth > 0``), the headers are
    required to render the page bodies regardless, so the walk is inherent.

    At the stub level a data page carries the generic ``data_page`` kind;
    the ``data_page_v1`` / ``data_page_v2`` distinction is a
    materialized-only detail (it is read from the header).
    """
    if _is_stub_level(child_depth):
        stubs = cc.page_stubs()
        if stubs is not None:
            dict_json = next(
                (_page_stub_json(s) for s in stubs if s.kind == "dictionary_page"),
                None,
            )
            pages_json = [_page_stub_json(s) for s in stubs if s.kind == "data_page"]
            return dict_json, pages_json
        dict_page, data_pages = _split_pages(cc)
        dict_json = _walked_page_stub_json(dict_page) if dict_page is not None else None
        return dict_json, [_walked_page_stub_json(p) for p in data_pages]
    dict_page, data_pages = _split_pages(cc)
    dict_json = _render(dict_page, view, child_depth) if dict_page is not None else None
    return dict_json, [_render(p, view, child_depth) for p in data_pages]


def _page_stub_json(stub: Any) -> dict:
    """JSON stub for an OffsetIndex-derived :class:`PageStub`."""
    return {
        "_kind": stub.kind,
        "_location": _location(stub.offset, stub.length),
        "_lazy": True,
    }


def _walked_page_stub_json(page: Any) -> dict:
    """JSON stub for a walked :class:`Page`. Data pages collapse to the
    generic ``data_page`` kind so a stub-level page kind is uniform
    whether it came from the OffsetIndex or a header walk; the specific
    version stays a materialized-only detail."""
    kind = "dictionary_page" if _kind_of(page) == "dictionary_page" else "data_page"
    return {
        "_kind": kind,
        "_location": _location(_offset_of(page), _length_of(page)),
        "_lazy": True,
    }


# ---------------------------------------------------------------------------
# Footer-segment helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stat / enum helpers
# ---------------------------------------------------------------------------


# Lazy imports to avoid import cycles with parquet_file.
def _enum_name(value: int, mapping: dict[int, str]) -> str:
    return mapping.get(value, str(value))


from parquet.ttypes import Encoding as _ThriftEncoding  # noqa: E402


def _build_encoding_names() -> dict[int, str]:
    return {
        getattr(_ThriftEncoding, name): name
        for name in dir(_ThriftEncoding)
        if not name.startswith("_") and isinstance(getattr(_ThriftEncoding, name), int)
    }


_ENC_NAMES = _build_encoding_names()


def _ref_stub(node: dict) -> dict:
    return {
        "_kind": node["_kind"],
        "_location": _location(node["_offset"], node["_length"]),
    }


def _ref_stub_from_wrapper(node: Any, kind: str) -> dict:
    return {
        "_kind": kind,
        "_location": _location(_offset_of(node), _length_of(node)),
    }


def _column_stat_info(cc: "ColumnChunk") -> dict:
    """The statistics type descriptor (``{logical, converted, scale}``) for
    ``cc``'s leaf column, from the file's cached schema-type map."""
    return cc._pf._stat_type_map.get(tuple(cc.path)) or {}


def _build_statistics(stats: Any, physical_type: str, info: dict) -> Any:
    """Build the v0 statistics object from a thrift ``Statistics``:
    ``null_count`` / ``distinct_count`` plus decoded ``min_value`` /
    ``max_value`` scalars. The deprecated ``min`` / ``max`` byte fields are
    dropped, but used as a fallback source for the value when a writer set
    only the deprecated fields."""
    if stats is None:
        return None
    logical = info.get("logical")
    converted = info.get("converted")
    scale = info.get("scale") or 0
    out: dict[str, Any] = {}
    if stats.null_count is not None:
        out["null_count"] = stats.null_count
    if stats.distinct_count is not None:
        out["distinct_count"] = stats.distinct_count
    min_raw = stats.min_value if stats.min_value is not None else stats.min
    max_raw = stats.max_value if stats.max_value is not None else stats.max
    if min_raw is not None:
        out["min_value"] = _json_safe_stat_value(
            min_raw, physical_type, logical, converted, scale
        )
    if max_raw is not None:
        out["max_value"] = _json_safe_stat_value(
            max_raw, physical_type, logical, converted, scale
        )
    return out or None


def _column_chunk_statistics(cc: "ColumnChunk") -> Any:
    """Statistics for ``cc`` with decoded ``min_value``/``max_value``.
    ``None`` when the writer omitted statistics."""
    return _build_statistics(cc._md.statistics, cc.type, _column_stat_info(cc))


def _page_statistics(header: Any, page: Any) -> Any:
    """Statistics for one page (from its header), decoded against the
    owning column's type. ``None`` when the page carries no statistics."""
    stats = getattr(header, "statistics", None) if header is not None else None
    if stats is None:
        return None
    cc = page._cc
    return _build_statistics(stats, cc.type, _column_stat_info(cc))


def _make_json_safe(value: Any) -> Any:
    """Recursively convert bytes (produced by binary thrift fields like
    statistics min/max for BYTE_ARRAY columns) into the structured form
    from :func:`parquet_analyzer.json_encode`. Everything else passes
    through unchanged."""
    if isinstance(value, bytes):
        return json_encode(value)
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_make_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Index-into-parent helpers (for column_chunk_data_region content fields)
# ---------------------------------------------------------------------------


def _row_group_index_of(cc: "ColumnChunk") -> int:
    rgs = cc._pf.row_groups
    for i, rg in enumerate(rgs):
        if rg is cc._rg:
            return i
    raise ValueError(
        f"column_chunk {cc.path!r} not attached to any row group of {cc._pf}"
    )


def _column_position(cc: "ColumnChunk") -> int:
    for i, col in enumerate(cc._rg.columns):
        if col is cc:
            return i
    raise ValueError(
        f"column_chunk {cc.path!r} not in its declared row group's columns"
    )
