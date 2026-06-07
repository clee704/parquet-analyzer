"""Lazy ParquetFile API — the primary entry point for inspecting a parquet file.

A :class:`ParquetFile` reads only the footer on construction (a few KB at the
end of the file). Per-row-group, per-column, and per-page metadata is exposed
through wrapper classes (:class:`RowGroup`, :class:`ColumnChunk`, :class:`Page`)
that defer expensive parsing until the caller actually asks for it.

The legacy :func:`parquet_analyzer._core.parse_parquet_file` function (eager,
walked every page header on every invocation) is gone. The exact
``(segments, column_chunk_data_offsets)`` shape it used to return is still
available — via :meth:`ParquetFile.all_segments` and
:attr:`ParquetFile.column_offset_map` — but only when the caller explicitly
asks for it.

Typical usage::

    from parquet_analyzer import ParquetFile

    pf = ParquetFile("example.parquet")

    # Footer-only — instant on any file size.
    print(pf.num_rows, pf.num_row_groups, pf.schema)
    print(pf.kv_metadata_lookup("com.acme.author"))

    # Full eager walk — only triggered explicitly.
    every_segment = pf.all_segments()
    every_page = pf.all_pages()
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Any

from parquet.ttypes import (
    BloomFilterHeader as _ThriftBloomFilterHeader,
    ColumnChunk as _ThriftColumnChunk,
    ColumnIndex as _ThriftColumnIndex,
    CompressionCodec as _ThriftCodec,
    Encoding as _ThriftEncoding,
    OffsetIndex as _ThriftOffsetIndex,
    PageHeader as _ThriftPageHeader,
    PageType as _ThriftPageType,
    RowGroup as _ThriftRowGroup,
    Type as _ThriftType,
)

from . import _footer_cache
from ._core import (
    _compute_pages,
    _compute_summary,
    _find_field,
    _iter_page_headers,
    _parse_footer,
    _walk_chunks_eager,
    column_decode_info,
    fill_gaps,
    read_thrift_segment,
    segment_to_json,
)
from .decoders import (
    RleBitPackedStream,
    decode_plain,
    decode_rle_bitpacked_hybrid_stream,
    decompress,
)
from ._tree_json import to_json_root as _to_json_root

__all__ = [
    "ColumnChunk",
    "DecodedPage",
    "MissingDictionaryError",
    "Page",
    "PageDecodeError",
    "ParquetFile",
    "PlainValues",
    "RowGroup",
    "UnsupportedCodecError",
    "UnsupportedEncodingError",
    "UnsupportedPageTypeError",
]

logger = logging.getLogger(__name__)

# Value encodings the body decoder supports (issue #21 scope). Other
# encodings (DELTA_*, BYTE_STREAM_SPLIT, ...) are deferred to #14 and raise
# a clean :class:`UnsupportedEncodingError`.
_DICTIONARY_ENCODINGS = frozenset({"PLAIN_DICTIONARY", "RLE_DICTIONARY"})


# ---------------------------------------------------------------------------
# Body-decode errors
# ---------------------------------------------------------------------------


class PageDecodeError(Exception):
    """Base class for the page-body decode errors.

    Each subclass carries a stable ``code`` string so a CLI layer can map it
    onto the JSON error contract (e.g. ``{"error": "encoding_not_supported",
    ...}``) without matching on the message text.
    """

    code = "page_decode_failed"


class UnsupportedEncodingError(PageDecodeError):
    """A value or level encoding outside the #21 decode scope (PLAIN,
    PLAIN_DICTIONARY, RLE_DICTIONARY for values; RLE for levels)."""

    code = "encoding_not_supported"

    def __init__(self, encoding: str, *, context: str = "values") -> None:
        self.encoding = encoding
        self.context = context
        super().__init__(
            f"{context} encoding {encoding!r} is not supported; this decoder "
            "handles PLAIN, PLAIN_DICTIONARY, and RLE_DICTIONARY values with "
            "RLE-encoded levels"
        )


class UnsupportedCodecError(PageDecodeError):
    """A compression codec this build cannot decompress (e.g. LZO, BROTLI),
    surfaced as a body-decode error rather than the decoder layer's
    ``NotImplementedError``."""

    code = "codec_not_supported"

    def __init__(self, codec: str) -> None:
        self.codec = codec
        super().__init__(f"compression codec {codec!r} is not supported")


class UnsupportedPageTypeError(PageDecodeError):
    """Body decode was requested on a page that is not a V1/V2 data page
    (e.g. an INDEX_PAGE, or a dictionary page asked to decode as data)."""

    code = "page_type_not_supported"

    def __init__(self, page_type: str) -> None:
        self.page_type = page_type
        super().__init__(
            f"cannot decode a {page_type} as a data page (only DATA_PAGE / "
            "DATA_PAGE_V2 carry decodable values + levels)"
        )


class MissingDictionaryError(PageDecodeError):
    """A dictionary-encoded data page whose column chunk has no decodable
    dictionary page — the indices cannot be resolved to values."""

    code = "missing_dictionary"

    def __init__(self, path: tuple[str, ...]) -> None:
        self.path = path
        super().__init__(
            f"column chunk {list(path)!r} has a dictionary-encoded data page "
            "but no dictionary page to resolve its indices"
        )


def _decompress(data: bytes, codec: str, uncompressed_size: int) -> bytes:
    """:func:`parquet_analyzer.decoders.decompress` with the decoder layer's
    ``NotImplementedError`` (unsupported/unknown codec) re-raised as
    :class:`UnsupportedCodecError`, keeping the body-decode error surface
    self-contained."""
    try:
        return decompress(data, codec, uncompressed_size)
    except NotImplementedError as exc:
        raise UnsupportedCodecError(codec) from exc


def _dictionary_lookup(dictionary: list, index: int, path: tuple[str, ...]) -> Any:
    """Resolve a dictionary index to its value, raising a clear
    ``ValueError`` (not a bare ``IndexError``) when a page's index falls
    outside the decoded dictionary — a corrupt or truncated dictionary."""
    if not 0 <= index < len(dictionary):
        raise ValueError(
            f"column chunk {list(path)!r}: dictionary index {index} out of "
            f"range for a dictionary of {len(dictionary)} entries"
        )
    return dictionary[index]


def _level_stream_v1(
    body: bytes, offset: int, max_level: int, num_values: int
) -> tuple[RleBitPackedStream | None, int]:
    """Decode a V1 level block (``[4-byte LE length][RLE/bit-packed stream]``)
    from a decompressed page body, returning ``(stream, next_offset)``.

    Returns ``(None, offset)`` for a column with ``max_level == 0`` (no level
    block exists on disk). The stream's ``bit_width`` is derived from
    ``max_level`` (``max_level.bit_length()``)."""
    if max_level == 0:
        return None, offset
    if offset + 4 > len(body):
        raise ValueError(
            f"truncated V1 level-block length prefix at offset {offset}: "
            f"need 4 bytes, have {len(body) - offset}"
        )
    (block_len,) = struct.unpack_from("<I", body, offset)
    start = offset + 4
    block_end = start + block_len
    if block_end > len(body):
        raise ValueError(
            f"V1 level block at offset {offset} claims {block_len} bytes but "
            f"only {len(body) - start} remain"
        )
    stream = decode_rle_bitpacked_hybrid_stream(
        body[start:block_end], max_level.bit_length(), num_values
    )
    return stream, block_end


def _level_stream_v2(
    stream_bytes: bytes, max_level: int, num_values: int
) -> RleBitPackedStream | None:
    """Decode a V2 level stream (stored uncompressed in a header-declared byte
    range, no length prefix), returning ``None`` for a ``max_level == 0``
    column. The stream's ``bit_width`` is ``max_level.bit_length()``."""
    if max_level == 0:
        return None
    return decode_rle_bitpacked_hybrid_stream(
        stream_bytes, max_level.bit_length(), num_values
    )


def _count_nulls(def_levels: RleBitPackedStream | None, max_def: int) -> int:
    """Count nulls from a V1 definition-level stream (a value is null when its
    definition level is below ``max_def``). ``0`` when there is no def block."""
    if def_levels is None:
        return 0
    return sum(1 for level in def_levels.values if level < max_def)


@dataclass(frozen=True)
class PlainValues:
    """The values section of a PLAIN-encoded page. PLAIN stores the values
    verbatim with no further run structure, so this is simply the decoded
    physical-type values (``bytes`` for ``BYTE_ARRAY`` /
    ``FIXED_LEN_BYTE_ARRAY`` / ``INT96``). Length is the page's non-null count."""

    values: tuple[Any, ...]


@dataclass
class DecodedPage:
    """The decoded body of a single V1/V2 data page, faithful to the page's
    on-disk encoding.

    Each encoded stream is exposed in its own encoding-logical form rather
    than as a flattened reconstruction:

    - ``repetition_levels`` / ``definition_levels`` are
      :class:`~parquet_analyzer.decoders.RleBitPackedStream` objects (the
      level RLE/bit-packed runs + the expanded per-value levels), or ``None``
      when the column has no such level block on disk.
    - ``values`` is the **values section**: a :class:`PlainValues` for a
      PLAIN page, or — because dictionary indices use the *same* RLE/bit-packed
      encoding as levels — an
      :class:`~parquet_analyzer.decoders.RleBitPackedStream` of the raw
      indices for a dictionary page. Resolving those indices to values (via
      the chunk's dictionary) is :meth:`Page.physical_values`, kept separate
      so the page's own data is not conflated with the sibling dictionary.

    The values section carries only the **non-null** values/indices (length
    ``num_values - num_nulls``); the nulls are represented by the definition
    levels. Reassembling a logical column (reinserting ``None``, assembling
    repeated columns) is a reader-level concern left to a higher layer.

    Returned by :meth:`Page.decode` and cached on the page; treat it as
    read-only.
    """

    encoding: str
    """Value encoding of the page (``PLAIN`` / ``PLAIN_DICTIONARY`` /
    ``RLE_DICTIONARY``)."""

    num_values: int
    """Total values in the page including nulls (from the page header)."""

    num_nulls: int
    """Number of nulls (V2: from the header; V1: counted from the
    definition levels)."""

    repetition_levels: RleBitPackedStream | None
    """The repetition-level stream (runs + expanded levels), or ``None`` for a
    non-repeated column (``max_repetition_level == 0``, no block on disk)."""

    definition_levels: RleBitPackedStream | None
    """The definition-level stream (runs + expanded levels), or ``None`` for a
    required column (``max_definition_level == 0``). A value is null when its
    definition level is below the column's ``max_definition_level``."""

    values: PlainValues | RleBitPackedStream
    """The values section: :class:`PlainValues` for a PLAIN page, or an
    :class:`~parquet_analyzer.decoders.RleBitPackedStream` of dictionary
    indices for a dictionary-encoded page."""

    values_body_offset: int
    """Byte offset where the values section starts — within the
    *decompressed* page body for a V1 page (whose levels live inside the
    compressed body), or within the on-disk page body for a V2 page (whose
    levels are stored uncompressed ahead of the values)."""


# ---------------------------------------------------------------------------
# ParquetFile
# ---------------------------------------------------------------------------


class ParquetFile:
    """Lazy parquet file reader.

    Construction reads only the footer (typically < 50 KB for wide schemas,
    < 5 KB for narrow ones). Page-header parsing, column-index walking,
    bloom-filter parsing, and the offset-recorded ``segments`` list are all
    deferred until the caller explicitly triggers them.

    The :class:`ParquetFile` instance represents a **snapshot** of the file
    at construction time. If the file is modified on disk after construction,
    later method calls may produce inconsistent results — construct a new
    instance instead.

    Args:
        path: filesystem path to a parquet file.
        use_cache: when True (default), the parsed footer is served from /
            saved to the on-disk footer cache for large footers (see
            :mod:`parquet_analyzer._footer_cache`). Set False to always
            parse and never touch the cache.

    Raises:
        ValueError: if the file is missing the ``PAR1`` header or trailer
            magic, or the footer thrift cannot be parsed.
    """

    def __init__(self, path: str, *, use_cache: bool = True) -> None:
        self._path = path
        self._f = open(path, "rb")
        try:
            self._f.seek(0, 2)
            self._file_size = self._f.tell()
            parsed = self._load_footer(use_cache)
            (
                self._footer_thrift,
                self._footer_segment,
                self._footer_offset,
                self._header_magic_segment,
                self._trailer_segments,
            ) = parsed
        except Exception:
            self._f.close()
            raise

        # Lazy caches for properties / methods that trigger work on first call.
        self._row_groups_cache: tuple[RowGroup, ...] | None = None
        self._footer_json_cache: dict | None = None
        self._eager_walked: bool = False
        self._eager_segments: list[dict] | None = None
        self._eager_column_offset_map: dict | None = None
        self._full_summary_cache: dict | None = None
        self._all_pages_cache: list[dict] | None = None
        self._stat_type_map_cache: dict | None = None
        self._decode_info_map_cache: dict | None = None

    def _load_footer(self, use_cache: bool) -> tuple:
        """Return the parsed-footer 5-tuple, served from the on-disk footer
        cache when available and re-parsed (and saved) otherwise.

        The cache is content-addressed on the footer bytes, so a hit is only
        ever served for a byte-identical footer; any read/unpickle failure
        falls through to a normal parse.
        """
        if not (use_cache and _footer_cache.enabled()):
            return _parse_footer(self._f, self._file_size)
        footer_bytes = _footer_cache.read_footer_bytes(self._f, self._file_size)
        key = (
            _footer_cache.compute_key(self._file_size, footer_bytes)
            if footer_bytes is not None
            else None
        )
        if key is not None:
            cached = _footer_cache.load(key)
            if cached is not None:
                return cached
        parsed = _parse_footer(self._f, self._file_size)
        if key is not None:
            _footer_cache.store(
                key, parsed, _footer_cache.column_chunk_count(parsed[0])
            )
        return parsed

    @property
    def _stat_type_map(self) -> dict:
        """Per-leaf statistics type descriptors (``path -> {logical,
        converted, scale}``), built once from the footer schema and cached.
        Used by the tree serializer to decode column/page statistics."""
        if self._stat_type_map_cache is None:
            from ._core import column_stat_types

            self._stat_type_map_cache = column_stat_types(self.footer["schema"])
        return self._stat_type_map_cache

    @property
    def _decode_info_map(self) -> dict:
        """Per-leaf body-decode descriptors (``path -> {max_def, max_rep,
        type_length}``), built once from the footer schema and cached. Used
        by :class:`ColumnChunk` / :class:`Page` to decode page bodies (level
        skipping needs the max levels; ``FIXED_LEN_BYTE_ARRAY`` decode needs
        ``type_length``)."""
        if self._decode_info_map_cache is None:
            self._decode_info_map_cache = column_decode_info(self.footer["schema"])
        return self._decode_info_map_cache

    def close(self) -> None:
        """Close the underlying file handle.

        :class:`ParquetFile` instances also work as context managers; using
        ``with ParquetFile(path) as pf: ...`` is the recommended pattern when
        the lifetime is naturally scoped. CLI-style one-shot use can rely on
        process exit to release the handle.
        """
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "ParquetFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ParquetFile({self._path!r}, num_rows={self.num_rows}, "
            f"num_row_groups={self.num_row_groups}, "
            f"num_columns={self.num_columns})"
        )

    # ----- Identity / size -------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def footer_offset(self) -> int:
        return self._footer_offset

    @property
    def footer_size(self) -> int:
        return self._footer_segment["length"]

    # ----- Tree-node interface (docs/tree-schema.md v0) --------------------

    @property
    def _kind(self) -> str:
        return "file"

    @property
    def _offset(self) -> int:
        return 0

    @property
    def _length(self) -> int:
        return self._file_size

    def to_json(self, *, view: str = "tree", depth: Any = "all") -> dict:
        """Serialize this node as v0 tree-schema JSON.

        Parameters
        ----------
        view : "tree" | "layout"
            ``tree`` exposes the logical (footer-derived) structure;
            ``layout`` exposes the physical byte arrangement with
            ``column_chunk_data_region`` synthesis and ``unknown``
            gap-fill. See ``docs/tree-schema.md``.
        depth : int | "all"
            ``0`` returns the root as a stub; ``N`` materializes
            ``N`` levels and stubs the rest; ``"all"`` materializes
            the entire tree (pays all lazy I/O). Defaults to
            ``"all"``.
        """
        return _to_json_root(self, view, depth)

    # ----- Footer access ---------------------------------------------------

    @property
    def footer_segment(self) -> dict:
        """The offset-recorded footer segment (raw dict, with byte ranges).

        This is the same segment that the legacy ``find_footer_segment()``
        used to return when filtering the segments list.
        """
        return self._footer_segment

    @property
    def footer(self) -> dict:
        """The footer as a JSON-friendly dict, decoded via
        :func:`parquet_analyzer.segment_to_json`. Cached.
        """
        if self._footer_json_cache is None:
            self._footer_json_cache = segment_to_json(self._footer_segment)
        return self._footer_json_cache

    @property
    def schema(self) -> list[dict]:
        """The schema as a list of element dicts (matches the
        ``footer["schema"]`` shape; pyarrow schema is not used here)."""
        return self.footer.get("schema", [])

    @property
    def kv_metadata(self) -> list[tuple[str, str | None]]:
        """Key-value metadata as an ordered list of ``(key, value)`` tuples.

        Parquet's spec allows duplicate keys; the list shape preserves both
        order and duplicates. ``value`` may be ``None`` when the writer
        recorded a key with no value.

        For a single-key lookup, prefer :meth:`kv_metadata_lookup`.
        """
        raw = self._footer_thrift.key_value_metadata
        if not raw:
            return []
        return [(kv.key, kv.value) for kv in raw]

    def kv_metadata_lookup(self, key: str) -> str | None:
        """First-wins lookup for a single kv-metadata key. Returns ``None``
        if the key is absent.
        """
        for k, v in self.kv_metadata:
            if k == key:
                return v
        return None

    @property
    def created_by(self) -> str | None:
        return self._footer_thrift.created_by

    @property
    def num_rows(self) -> int:
        return self._footer_thrift.num_rows

    @property
    def num_row_groups(self) -> int:
        return len(self._footer_thrift.row_groups or [])

    @property
    def num_columns(self) -> int:
        """Number of distinct column paths in the schema.

        Counts leaf columns from the first row group (every row group has the
        same column set, per parquet spec).
        """
        if not self._footer_thrift.row_groups:
            return 0
        return len(self._footer_thrift.row_groups[0].columns or [])

    # ----- Row groups ------------------------------------------------------

    @property
    def row_groups(self) -> tuple["RowGroup", ...]:
        if self._row_groups_cache is None:
            rg_thrifts = self._footer_thrift.row_groups or []
            extents = _extract_row_group_extents(self._footer_segment, rg_thrifts)
            self._row_groups_cache = tuple(
                RowGroup(self, rg, ext, idx)
                for idx, (rg, ext) in enumerate(zip(rg_thrifts, extents))
            )
        return self._row_groups_cache

    # ----- Summaries -------------------------------------------------------

    @property
    def footer_summary(self) -> dict[str, Any]:
        """Compact summary derived entirely from the footer.

        Always cheap — no page-header walking, no body reads. Includes
        row/group/column counts, footer size, file size, and aggregate
        compressed/uncompressed column-chunk sizes (which the parquet spec
        records in ``ColumnMetaData``, no walking needed).

        For the full summary including per-page counts and page-header
        sizes, use :attr:`full_summary` — but that triggers a full walk.
        """
        footer = self.footer
        uncompressed_page_size = 0
        compressed_page_size = 0
        column_index_size = 0
        offset_index_size = 0
        bloom_filter_size = 0
        for row_group in footer["row_groups"]:
            for column in row_group["columns"]:
                uncompressed_page_size += column["meta_data"]["total_uncompressed_size"]
                compressed_page_size += column["meta_data"]["total_compressed_size"]
                column_index_size += column.get("column_index_length", 0)
                offset_index_size += column.get("offset_index_length", 0)
                bloom_filter_size += column["meta_data"].get("bloom_filter_length", 0)

        return {
            "num_rows": self.num_rows,
            "num_row_groups": self.num_row_groups,
            "num_columns": self.num_columns,
            "uncompressed_page_size": uncompressed_page_size,
            "compressed_page_size": compressed_page_size,
            "column_index_size": column_index_size,
            "offset_index_size": offset_index_size,
            "bloom_filter_size": bloom_filter_size,
            "footer_size": self.footer_size,
            "file_size": self.file_size,
        }

    @property
    def full_summary(self) -> dict[str, Any]:
        """Full summary including per-page counts. **Triggers an eager walk.**

        Same shape as the legacy ``get_summary()`` output. Use this when
        you need the page-level fields (``num_pages``, ``num_data_pages``,
        ``num_v1_data_pages``, ``num_v2_data_pages``, ``num_dict_pages``,
        ``page_header_size``, ``uncompressed_page_data_size``,
        ``compressed_page_data_size``). Otherwise prefer :attr:`footer_summary`.

        Caches the computed dict after first call (subsequent accesses are
        ~O(1) dict lookup). Returned dict is the cached reference — do not
        mutate.
        """
        if self._full_summary_cache is None:
            self._ensure_eager_walked()
            assert self._eager_segments is not None
            self._full_summary_cache = _compute_summary(
                self.footer, self._eager_segments
            )
        return self._full_summary_cache

    # ----- Full eager walk (the "I really want everything" path) ----------

    def all_segments(self) -> list[dict]:
        """Walk every page header / column index / offset index / bloom
        filter and return the complete segments list — same shape as the
        legacy ``parse_parquet_file()[0]`` output (sorted by offset, with
        "unknown" segments filling any gaps).

        Cached after first call. Expensive; the only way to get the
        complete byte-range view of the file. Returned list is the cached
        reference — do not mutate (mutating it would corrupt subsequent
        :attr:`full_summary` calls that walk the same segments).
        """
        self._ensure_eager_walked()
        assert self._eager_segments is not None
        return self._eager_segments

    def all_pages(self) -> list[dict]:
        """Walk every page header and return the per-column pages tree —
        same shape as the legacy ``get_pages()`` output.

        Caches the computed tree after first call. Returned list is the
        cached reference — do not mutate.
        """
        if self._all_pages_cache is None:
            self._ensure_eager_walked()
            assert self._eager_segments is not None
            assert self._eager_column_offset_map is not None
            self._all_pages_cache = _compute_pages(
                self._eager_segments, self._eager_column_offset_map
            )
        return self._all_pages_cache

    @property
    def column_offset_map(self) -> dict[tuple[str, ...], list[dict]]:
        """The legacy ``parse_parquet_file()[1]`` shape: per-column-path
        list of dicts containing the byte offsets of each page / dictionary
        page / index per row group.

        **Triggers an eager walk** if not already done. Cached. Returned
        dict is the cached reference — do not mutate.
        """
        self._ensure_eager_walked()
        assert self._eager_column_offset_map is not None
        return self._eager_column_offset_map

    def _ensure_eager_walked(self) -> None:
        if self._eager_walked:
            return
        segments: list[dict] = [
            self._header_magic_segment,
            self._trailer_segments["trailer_magic"],
            self._trailer_segments["footer_length"],
            self._footer_segment,
        ]
        column_offset_map = _walk_chunks_eager(self._f, self._footer_thrift, segments)
        segments.sort(key=lambda s: s["offset"])
        self._eager_segments = fill_gaps(segments, self._file_size)
        self._eager_column_offset_map = column_offset_map
        self._eager_walked = True


# ---------------------------------------------------------------------------
# RowGroup
# ---------------------------------------------------------------------------


class RowGroup:
    """Lazy wrapper around a single row group's metadata.

    Constructed by :attr:`ParquetFile.row_groups`. All properties are
    footer-derived (cheap, no body reads). The :meth:`columns` accessor
    returns :class:`ColumnChunk` wrappers.
    """

    def __init__(
        self,
        parquet_file: "ParquetFile",
        thrift_obj: _ThriftRowGroup,
        extent: tuple[int, int],
        index: int,
    ) -> None:
        self._pf = parquet_file
        self._t = thrift_obj
        # Byte extent of this row-group's thrift struct *inside* the
        # footer — used for tree-node ``_offset``/``_length``. Captured
        # at construction time from the footer-segment walk so that
        # tree-node access is free.
        self._extent = extent
        # Position of this row group in the footer's row_groups list;
        # used to locate its column-chunk extents in the footer segment.
        self._rg_index = index
        self._columns_cache: tuple[ColumnChunk, ...] | None = None

    def __repr__(self) -> str:
        return (
            f"RowGroup(num_rows={self.num_rows}, "
            f"total_byte_size={self.total_byte_size}, "
            f"num_columns={len(self._t.columns or [])})"
        )

    @property
    def num_rows(self) -> int:
        return self._t.num_rows

    @property
    def total_byte_size(self) -> int:
        return self._t.total_byte_size

    @property
    def columns(self) -> tuple["ColumnChunk", ...]:
        if self._columns_cache is None:
            cc_thrifts = self._t.columns or []
            cc_extents = _extract_column_chunk_extents(
                self._pf._footer_segment, self._rg_index, cc_thrifts
            )
            self._columns_cache = tuple(
                ColumnChunk(self._pf, self, cc, ext)
                for cc, ext in zip(cc_thrifts, cc_extents)
            )
        return self._columns_cache

    # ----- Tree-node interface (docs/tree-schema.md v0) --------------------

    @property
    def _kind(self) -> str:
        return "row_group"

    @property
    def _offset(self) -> int:
        return self._extent[0]

    @property
    def _length(self) -> int:
        return self._extent[1]

    def to_json(self, *, view: str = "tree", depth: Any = "all") -> dict:
        """Serialize this row group as v0 tree-schema JSON. See
        :meth:`ParquetFile.to_json`."""
        return _to_json_root(self, view, depth)


# ---------------------------------------------------------------------------
# ColumnChunk
# ---------------------------------------------------------------------------


# Type enum-name lookups for ColumnChunk / Page properties. Parquet thrift
# stores enums as ints; callers expect strings ("INT32", "SNAPPY", "PLAIN").
# These dicts are built once at import (cheap, ~30 entries each total) and
# looked up by integer key on the hot path — critically, ColumnChunk.pages()
# iteration would otherwise rebuild these on every Page.type / Page.encoding
# property access.


def _enum_name_map(enum_class) -> dict[int, str]:
    return {
        getattr(enum_class, name): name
        for name in dir(enum_class)
        if not name.startswith("_") and isinstance(getattr(enum_class, name), int)
    }


_TYPE_NAMES = _enum_name_map(_ThriftType)
_ENCODING_NAMES = _enum_name_map(_ThriftEncoding)
_CODEC_NAMES = _enum_name_map(_ThriftCodec)
_PAGE_TYPE_NAMES = _enum_name_map(_ThriftPageType)


@dataclass(frozen=True)
class PageStub:
    """Lightweight descriptor of a single page, obtained WITHOUT reading
    the page's thrift header.

    Produced by :meth:`ColumnChunk.page_stubs` from the OffsetIndex (data
    pages) and the column-metadata offsets (dictionary page). It carries
    only what is knowable cheaply — the page's byte extent and, for data
    pages, the index of its first row. The page *version* and per-page
    header details are deliberately absent: recovering them requires
    reading the header, which is exactly what stub enumeration avoids. So
    ``kind`` is the generic ``"data_page"`` (or ``"dictionary_page"``);
    the specific ``data_page_v1`` / ``data_page_v2`` distinction appears
    only on a materialized :class:`Page`.
    """

    kind: str
    """``"dictionary_page"`` or the generic ``"data_page"``."""

    offset: int
    """Absolute file offset of the page (start of its header)."""

    length: int
    """Full page extent in bytes (header + body), matching
    :attr:`Page._length`."""

    first_row_index: int | None
    """Row index (within the row group) of the page's first row, from the
    OffsetIndex. ``None`` for the dictionary page (which has no rows)."""


class ColumnChunk:
    """Lazy wrapper around a single column chunk's metadata.

    Constructed by :attr:`RowGroup.columns`. All properties are
    footer-derived (cheap, no body reads). The :meth:`pages` accessor
    walks per-chunk page headers on first call (per-chunk lazy
    boundary — cheaper than full-file walk; only this chunk's headers
    are parsed).
    """

    def __init__(
        self,
        parquet_file: "ParquetFile",
        row_group: "RowGroup",
        thrift_obj: _ThriftColumnChunk,
        extent: tuple[int, int],
    ) -> None:
        self._pf = parquet_file
        self._rg = row_group
        self._t = thrift_obj
        self._md = thrift_obj.meta_data
        # Byte extent of the column-chunk thrift struct inside the
        # footer — used for tree-node ``_offset``/``_length``.
        self._extent = extent
        self._pages_cache: tuple[Page, ...] | None = None
        self._offset_index_cache: _ThriftOffsetIndex | None = None
        self._column_index_cache: _ThriftColumnIndex | None = None
        self._bloom_filter_header_cache: _ThriftBloomFilterHeader | None = None
        self._decode_info_cache: dict | None = None
        self._dictionary_cache: list[Any] | None = None
        self._dictionary_computed: bool = False

    def __repr__(self) -> str:
        return (
            f"ColumnChunk(path={self.path!r}, type={self.type!r}, "
            f"num_values={self.num_values})"
        )

    # ----- Footer-derived properties ---------------------------------------

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(self._md.path_in_schema)

    @property
    def type(self) -> str:
        return _TYPE_NAMES.get(self._md.type, str(self._md.type))

    @property
    def encodings(self) -> tuple[str, ...]:
        return tuple(_ENCODING_NAMES.get(e, str(e)) for e in (self._md.encodings or []))

    @property
    def codec(self) -> str:
        return _CODEC_NAMES.get(self._md.codec, str(self._md.codec))

    @property
    def num_values(self) -> int:
        return self._md.num_values

    @property
    def total_compressed_size(self) -> int:
        return self._md.total_compressed_size

    @property
    def total_uncompressed_size(self) -> int:
        return self._md.total_uncompressed_size

    @property
    def data_page_offset(self) -> int:
        return self._md.data_page_offset

    @property
    def dictionary_page_offset(self) -> int | None:
        return self._md.dictionary_page_offset

    @property
    def column_index_offset(self) -> int | None:
        return self._t.column_index_offset

    @property
    def offset_index_offset(self) -> int | None:
        return self._t.offset_index_offset

    @property
    def has_offset_index(self) -> bool:
        """Whether this chunk has an OffsetIndex thrift struct in the file.

        When ``True``, :attr:`num_pages` and the :meth:`page` random-access
        accessor can serve queries via a single small thrift parse without
        walking every page header — typically 50-200 bytes per page entry
        in the OffsetIndex.

        SNPW (Spark Native Parquet Writer) writes OffsetIndex on every
        column chunk; pyarrow writes it when ``write_page_index=True`` is
        passed to ``write_table``; older parquet-mr versions and many
        DuckDB-produced files do not have it. Without OffsetIndex, the
        only way to discover page count or seek to a specific page is to
        walk every prior page header — a fundamental parquet limitation,
        not an API choice.
        """
        return self._t.offset_index_offset is not None

    @property
    def num_pages(self) -> int:
        """Total number of pages in this column chunk (including the
        dictionary page, if present).

        Fast path (O(1) page walk): when :attr:`has_offset_index` is
        ``True``, reads the OffsetIndex thrift struct (one small parse,
        cached). The OffsetIndex itself only tracks data pages, so this
        property adds 1 if :attr:`dictionary_page_offset` is set —
        producing the same count as ``len(self.pages())`` either way.

        Slow path: when OffsetIndex is absent, falls back to walking
        every page header — same cost as ``len(self.pages())``. Use
        :attr:`has_offset_index` to decide whether the call will be cheap
        before invoking it on large chunks.
        """
        if self.has_offset_index:
            oi = self._read_offset_index()
            data_pages = len(oi.page_locations or [])
            dict_page = 1 if self._md.dictionary_page_offset else 0
            return data_pages + dict_page
        return len(self.pages())

    @property
    def offset_index(self) -> "_ThriftOffsetIndex | None":
        """The parsed OffsetIndex thrift for this chunk, or ``None`` when
        the chunk has no OffsetIndex.

        Public companion to :attr:`has_offset_index` and the private
        reader, returning ``None`` rather than raising when absent. The
        OffsetIndex records per-data-page byte offsets, compressed sizes,
        and first-row indices (it does not include the dictionary page);
        :meth:`page` uses it to seek without walking.
        """
        if not self.has_offset_index:
            return None
        return self._read_offset_index()

    def _read_offset_index(self) -> _ThriftOffsetIndex:
        """Read + cache the OffsetIndex thrift struct for this chunk.

        Caller must check :attr:`has_offset_index` first; this raises
        ``ValueError`` if the chunk has none. The cached object is
        returned on subsequent calls. The public :attr:`offset_index`
        property wraps this and returns ``None`` instead of raising.
        """
        if self._t.offset_index_offset is None:
            raise ValueError(
                f"column chunk {self.path!r} has no OffsetIndex; "
                "check has_offset_index before calling _read_offset_index"
            )
        if self._offset_index_cache is None:
            oi, _segment = read_thrift_segment(
                self._pf._f,
                self._t.offset_index_offset,
                "offset_index",
                _ThriftOffsetIndex,
            )
            self._offset_index_cache = oi
        return self._offset_index_cache

    def _read_column_index(self) -> _ThriftColumnIndex:
        """Read + cache the ColumnIndex thrift struct for this chunk.

        Private companion to :meth:`_read_offset_index`. The tree-node
        ``offset_index`` / ``column_index`` / ``bloom_filter_header``
        kinds are opaque branches in v0 — materialization pays the
        thrift parse but emits no content fields. A public accessor is
        tracked in #20 / #21.
        """
        if self._t.column_index_offset is None:
            raise ValueError(
                f"column chunk {self.path!r} has no ColumnIndex; "
                "check column_index_offset before calling _read_column_index"
            )
        if self._column_index_cache is None:
            ci, _segment = read_thrift_segment(
                self._pf._f,
                self._t.column_index_offset,
                "column_index",
                _ThriftColumnIndex,
            )
            self._column_index_cache = ci
        return self._column_index_cache

    def _read_bloom_filter_header(self) -> _ThriftBloomFilterHeader:
        """Read + cache the BloomFilterHeader thrift struct for this chunk.

        Private companion to :meth:`_read_offset_index` /
        :meth:`_read_column_index`. The bloom filter body itself is
        not parsed in v0 — only its header.
        """
        if self._md.bloom_filter_offset is None:
            raise ValueError(
                f"column chunk {self.path!r} has no BloomFilter; "
                "check bloom_filter_offset before calling _read_bloom_filter_header"
            )
        if self._bloom_filter_header_cache is None:
            bf, _segment = read_thrift_segment(
                self._pf._f,
                self._md.bloom_filter_offset,
                "bloom_filter_header",
                _ThriftBloomFilterHeader,
            )
            self._bloom_filter_header_cache = bf
        return self._bloom_filter_header_cache

    @property
    def bloom_filter_offset(self) -> int | None:
        return self._md.bloom_filter_offset

    @property
    def has_bloom_filter(self) -> bool:
        return self._md.bloom_filter_offset is not None

    @property
    def statistics(self) -> Any:
        """Raw thrift Statistics object (or ``None``).

        Returned as-is from parquet thrift; callers wanting a JSON shape
        can walk :attr:`ParquetFile.footer` instead. Kept as the raw object
        here to avoid committing to a JSON layout for the new API surface.
        """
        return self._md.statistics

    # ----- Body-decode descriptors (schema-derived, cheap) -----------------

    @property
    def _decode_info(self) -> dict:
        """This chunk's leaf-column decode descriptor (``{max_def, max_rep,
        type_length}``) from the file's cached schema walk. Raises
        ``ValueError`` when the path is absent — an internal inconsistency
        between the column metadata and the schema, never expected for a
        well-formed file."""
        if self._decode_info_cache is None:
            info = self._pf._decode_info_map.get(self.path)
            if info is None:
                raise ValueError(
                    f"column chunk {list(self.path)!r} has no matching schema "
                    "leaf; cannot derive definition/repetition levels"
                )
            self._decode_info_cache = info
        return self._decode_info_cache

    @property
    def max_definition_level(self) -> int:
        """Maximum definition level for this column (0 for a ``REQUIRED``
        column). Equals the number of ``OPTIONAL``/``REPEATED`` ancestors
        including the leaf — the bit budget the definition-level stream
        packs against, and the threshold a value's definition level must
        reach to be non-null."""
        return self._decode_info["max_def"]

    @property
    def max_repetition_level(self) -> int:
        """Maximum repetition level for this column (0 for a flat, i.e.
        non-repeated, column). Equals the number of ``REPEATED`` ancestors
        including the leaf."""
        return self._decode_info["max_rep"]

    @property
    def type_length(self) -> int | None:
        """Fixed width in bytes for a ``FIXED_LEN_BYTE_ARRAY`` column,
        ``None`` for every other physical type. Required to PLAIN-decode
        ``FIXED_LEN_BYTE_ARRAY`` values."""
        return self._decode_info["type_length"]

    def dictionary(self) -> list[Any] | None:
        """Decode + cache this chunk's dictionary page, returning the
        dictionary entries as a list of physical-type values, or ``None``
        when the chunk has no dictionary page.

        Resolution prefers the footer's ``dictionary_page_offset`` (one
        page-header read), falling back to scanning the walked page headers
        for a ``DICTIONARY_PAGE`` — older writers point ``data_page_offset``
        at the dictionary and leave ``dictionary_page_offset`` unset (see
        :func:`parquet_analyzer._core._iter_page_headers`). The decoded list
        is cached on the chunk so repeated data-page decodes in the same
        chunk pay the dictionary read once.
        """
        if not self._dictionary_computed:
            dict_page = self._find_dictionary_page()
            self._dictionary_cache = (
                dict_page._decode_dictionary_entries()
                if dict_page is not None
                else None
            )
            self._dictionary_computed = True
        return self._dictionary_cache

    def _find_dictionary_page(self) -> "Page | None":
        """Locate this chunk's dictionary page as a :class:`Page`, or
        ``None``. Uses the footer offset when present, else scans the walked
        page headers."""
        if self._md.dictionary_page_offset:
            thrift, segment = read_thrift_segment(
                self._pf._f, self._md.dictionary_page_offset, "page", _ThriftPageHeader
            )
            if thrift.dictionary_page_header is None:
                raise ValueError(
                    f"column chunk {list(self.path)!r} dictionary_page_offset "
                    f"{self._md.dictionary_page_offset} does not point at a "
                    "dictionary page"
                )
            return Page(self._pf, self, thrift, segment)
        for p in self.pages():
            if p._t.dictionary_page_header is not None:
                return p
        return None

    # ----- Page-header walking (per-chunk lazy boundary) -------------------

    def pages(self) -> tuple["Page", ...]:
        """Walk this column chunk's page headers on first call; cache and
        return :class:`Page` wrappers.

        Per-chunk page walking is cheap because it only touches one
        column's pages, not the whole file. Page bodies are NOT read —
        only the per-page Thrift header is parsed. Page body access and
        decode is available on each :class:`Page` (:meth:`Page.decode` and
        friends), read lazily on demand.
        """
        if self._pages_cache is None:
            self._pages_cache = tuple(
                Page(self._pf, self, thrift, segment)
                for thrift, segment in _iter_page_headers(
                    self._pf._f,
                    self._md.dictionary_page_offset,
                    self._md.data_page_offset,
                    self._md.num_values,
                )
            )
        return self._pages_cache

    def page_stubs(self) -> tuple["PageStub", ...] | None:
        """Enumerate this chunk's pages as lightweight :class:`PageStub`
        descriptors WITHOUT walking the per-page thrift headers.

        Returns ``None`` when the chunk has no OffsetIndex
        (:attr:`has_offset_index` is ``False``). The OffsetIndex is the
        only source of per-data-page extents short of a full header walk,
        so without it the pages cannot be enumerated cheaply; the caller
        then chooses to either walk (:meth:`pages`) or surface a "walk
        required" affordance.

        When an OffsetIndex is present the cost is one OffsetIndex parse
        (cached) and **no per-page header reads** — independent of the
        page count. This is the fast path that lets a column's pages be
        *listed* cheaply (#30):

        - The dictionary-page extent is computed from the column-metadata
          offsets (see :meth:`_dictionary_page_extent`) — footer-derived, no
          read.
        - Each data-page extent comes from an OffsetIndex ``PageLocation``;
          its ``compressed_page_size`` includes the page header per the
          parquet spec, so it matches the materialized :attr:`Page._length`.

        The enumeration covers the dictionary page (when present) and the
        data pages. INDEX_PAGE — deprecated and not written by any encoder
        that also writes an OffsetIndex — is not separately represented.
        """
        if not self.has_offset_index:
            return None
        oi = self._read_offset_index()
        stubs: list[PageStub] = []
        dict_extent = self._dictionary_page_extent()
        if dict_extent is not None:
            stubs.append(
                PageStub(
                    kind="dictionary_page",
                    offset=dict_extent[0],
                    length=dict_extent[1],
                    first_row_index=None,
                )
            )
        for loc in oi.page_locations or []:
            stubs.append(
                PageStub(
                    kind="data_page",
                    offset=loc.offset,
                    length=loc.compressed_page_size,
                    first_row_index=loc.first_row_index,
                )
            )
        return tuple(stubs)

    def _dictionary_page_extent(self) -> tuple[int, int] | None:
        """``(offset, length)`` of this chunk's dictionary page, or ``None``
        when there is none. Footer-derived (no read): the dictionary page
        spans ``[dictionary_page_offset, data_page_offset)``. For a column
        with no data pages — e.g. a 0-row column, where ``data_page_offset``
        is ``0`` rather than past the dictionary page — the dictionary page
        is the whole compressed region, whose size is ``total_compressed_size``.
        """
        off = self._md.dictionary_page_offset
        if not off:
            return None
        dpo = self._md.data_page_offset
        length = dpo - off if dpo and dpo > off else self._md.total_compressed_size
        return off, length

    def page(self, index: int) -> "Page":
        """Return the ``index``-th page (a :class:`Page`), supporting
        negative indices. Random-access counterpart to :meth:`pages`.

        When :attr:`has_offset_index` is ``True``, seeks directly to the
        requested page via the OffsetIndex (one small thrift parse + one
        page-header read) — no full page-header walk. Otherwise it indexes
        the walked list (materializing it once). Page 0 is the dictionary
        page when one is present, matching :meth:`pages` ordering.
        """
        n = self.num_pages
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"page index out of range: {index} (chunk has {n} pages)")
        if self._pages_cache is not None:
            return self._pages_cache[index]
        if self.has_offset_index:
            return self._page_via_offset_index(index)
        return self.pages()[index]

    def _page_via_offset_index(self, index: int) -> "Page":
        """Direct-seek the ``index``-th page using the OffsetIndex. The
        OffsetIndex tracks only data pages, so page 0 is the dictionary
        page (when present) and data page ``i`` is OffsetIndex entry ``i``
        (offset by 1 when a dictionary page leads)."""
        has_dict = bool(self._md.dictionary_page_offset)
        if has_dict and index == 0:
            offset = self._md.dictionary_page_offset
        else:
            loc_index = index - 1 if has_dict else index
            offset = self._read_offset_index().page_locations[loc_index].offset
        thrift, segment = read_thrift_segment(
            self._pf._f, offset, "page", _ThriftPageHeader
        )
        return Page(self._pf, self, thrift, segment)

    # ----- Tree-node interface (docs/tree-schema.md v0) --------------------

    @property
    def _kind(self) -> str:
        return "column_chunk"

    @property
    def _offset(self) -> int:
        return self._extent[0]

    @property
    def _length(self) -> int:
        return self._extent[1]

    def to_json(self, *, view: str = "tree", depth: Any = "all") -> dict:
        """Serialize this column chunk as v0 tree-schema JSON. See
        :meth:`ParquetFile.to_json`."""
        return _to_json_root(self, view, depth)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class Page:
    """Lazy wrapper around a single page's metadata and body.

    Constructed by :meth:`ColumnChunk.pages`. The header-derived properties
    (``type``, ``encoding``, ``num_values``, ``offset``, sizes) are cheap.
    Body access — :meth:`raw_body`, :meth:`decode`, :meth:`definition_levels`,
    :meth:`repetition_levels`, :meth:`physical_values` — reads (and decodes)
    the page body from disk on demand and caches the decoded result on the
    page.
    """

    def __init__(
        self,
        parquet_file: "ParquetFile",
        column_chunk: "ColumnChunk",
        thrift_obj,
        segment: dict,
    ) -> None:
        self._pf = parquet_file
        self._cc = column_chunk
        self._t = thrift_obj
        self._segment = segment
        self._decoded_cache: DecodedPage | None = None

    def __repr__(self) -> str:
        return (
            f"Page(type={self.type!r}, encoding={self.encoding!r}, "
            f"num_values={self.num_values}, offset={self.offset})"
        )

    @property
    def offset(self) -> int:
        """Absolute file offset of the page header (start of page)."""
        return self._segment["offset"]

    @property
    def header_size(self) -> int:
        return self._segment["length"]

    @property
    def uncompressed_size(self) -> int:
        return self._t.uncompressed_page_size

    @property
    def compressed_size(self) -> int:
        return self._t.compressed_page_size

    @property
    def type(self) -> str:
        return _PAGE_TYPE_NAMES.get(self._t.type, str(self._t.type))

    @property
    def num_values(self) -> int:
        h = (
            self._t.data_page_header
            or self._t.data_page_header_v2
            or self._t.dictionary_page_header
        )
        return h.num_values if h is not None else 0

    @property
    def encoding(self) -> str:
        h = (
            self._t.data_page_header
            or self._t.data_page_header_v2
            or self._t.dictionary_page_header
        )
        if h is None:
            return ""
        return _ENCODING_NAMES.get(h.encoding, str(h.encoding))

    @property
    def segment(self) -> dict:
        """The offset-recorded page-header segment (raw dict)."""
        return self._segment

    # ----- Body access + decode (issue #21) --------------------------------

    @property
    def body_offset(self) -> int:
        """Absolute file offset of the page body (immediately after the page
        header thrift)."""
        return self._segment["offset"] + self._segment["length"]

    def raw_body(self) -> bytes:
        """Read and return the page's on-disk body bytes (the
        ``compressed_page_size`` bytes immediately after the page header).

        These are the bytes as written: for a compressed page that is the
        compressed form (for a V2 page only the values section is
        compressed; the levels are stored uncompressed at the front). No
        decoding is performed.
        """
        self._pf._f.seek(self.body_offset)
        return self._pf._f.read(self._t.compressed_page_size)

    def decode(self) -> DecodedPage:
        """Decode this data page's body — its level streams and its values
        section — into the encoding-faithful :class:`DecodedPage` and cache it.

        Dispatches on page version (V1 levels are length-prefixed inside the
        compressed body; V2 levels are uncompressed in header-declared byte
        ranges ahead of an optionally-compressed values section). The values
        section is decoded into its encoding's own structure (PLAIN values, or
        a dictionary-index :class:`~parquet_analyzer.decoders.RleBitPackedStream`);
        it does **not** resolve dictionary indices to values — that is
        :meth:`physical_values`, which pulls in the sibling dictionary page.

        Raises:
            UnsupportedPageTypeError: the page is not a V1/V2 data page.
            UnsupportedEncodingError: the value encoding (or a V1 level
                encoding) is outside the supported set.
            UnsupportedCodecError: the page's codec cannot be decompressed.
        """
        if self._decoded_cache is None:
            if self._t.data_page_header is not None:
                self._decoded_cache = self._decode_v1()
            elif self._t.data_page_header_v2 is not None:
                self._decoded_cache = self._decode_v2()
            else:
                raise UnsupportedPageTypeError(self.type)
        return self._decoded_cache

    def definition_levels(self) -> list[int]:
        """The expanded per-value definition levels (length :attr:`num_values`);
        ``[0] * n`` for a required column. Convenience over
        ``decode().definition_levels`` (the full level stream)."""
        stream = self.decode().definition_levels
        return list(stream.values) if stream is not None else [0] * self.num_values

    def repetition_levels(self) -> list[int]:
        """The expanded per-value repetition levels (length :attr:`num_values`);
        ``[0] * n`` for a non-repeated column. Convenience over
        ``decode().repetition_levels`` (the full level stream)."""
        stream = self.decode().repetition_levels
        return list(stream.values) if stream is not None else [0] * self.num_values

    def physical_values(self) -> list[Any]:
        """The decoded **non-null** values of this data page in physical-type
        form (``bytes`` for ``BYTE_ARRAY`` / ``FIXED_LEN_BYTE_ARRAY`` /
        ``INT96``). Length is ``num_values - num_nulls``; the nulls are
        carried by the definition levels.

        For a PLAIN page these are the values verbatim; for a dictionary page
        the indices in ``decode().values`` are resolved through the chunk's
        dictionary (:meth:`ColumnChunk.dictionary`).

        Raises:
            MissingDictionaryError: a dictionary-encoded page whose chunk has
                no dictionary page.
        """
        section = self.decode().values
        if isinstance(section, PlainValues):
            return list(section.values)
        dictionary = self._cc.dictionary()
        if dictionary is None:
            raise MissingDictionaryError(self._cc.path)
        return [
            _dictionary_lookup(dictionary, i, self._cc.path) for i in section.values
        ]

    def _decode_v1(self) -> DecodedPage:
        """Decode a V1 data page: decompress the whole body, then read the
        repetition and definition level blocks (each ``[4-byte LE len][RLE]``,
        present only when the column's max level is > 0) followed by the
        encoded values."""
        h = self._t.data_page_header
        cc = self._cc
        num_values = h.num_values
        max_def = cc.max_definition_level
        self._require_rle_level_encoding(
            h.repetition_level_encoding, cc.max_repetition_level, "rep"
        )
        self._require_rle_level_encoding(h.definition_level_encoding, max_def, "def")

        body = _decompress(self.raw_body(), cc.codec, self._t.uncompressed_page_size)
        rep_levels, offset = _level_stream_v1(
            body, 0, cc.max_repetition_level, num_values
        )
        def_levels, offset = _level_stream_v1(body, offset, max_def, num_values)
        return self._assemble(
            encoding_value=h.encoding,
            values_buf=body[offset:],
            num_values=num_values,
            num_nulls=_count_nulls(def_levels, max_def),
            rep_levels=rep_levels,
            def_levels=def_levels,
            values_body_offset=offset,
        )

    def _decode_v2(self) -> DecodedPage:
        """Decode a V2 data page: the repetition and definition level streams
        are stored uncompressed at the front of the body in header-declared
        byte lengths; only the trailing values section is (optionally)
        compressed."""
        h = self._t.data_page_header_v2
        cc = self._cc
        num_values = h.num_values
        rep_len = h.repetition_levels_byte_length or 0
        def_len = h.definition_levels_byte_length or 0
        body = self.raw_body()
        if rep_len + def_len > len(body):
            raise ValueError(
                f"V2 page levels ({rep_len}+{def_len} bytes) exceed the page "
                f"body ({len(body)} bytes)"
            )
        rep_levels = _level_stream_v2(
            body[:rep_len], cc.max_repetition_level, num_values
        )
        def_levels = _level_stream_v2(
            body[rep_len : rep_len + def_len], cc.max_definition_level, num_values
        )
        values_section = body[rep_len + def_len :]
        # is_compressed defaults to True in the thrift; treat an unset value
        # as compressed too. Levels are never compressed, so the values'
        # uncompressed size is the page total minus the level bytes.
        if h.is_compressed is None or h.is_compressed:
            values_buf = _decompress(
                values_section,
                cc.codec,
                self._t.uncompressed_page_size - rep_len - def_len,
            )
        else:
            values_buf = values_section
        return self._assemble(
            encoding_value=h.encoding,
            values_buf=values_buf,
            num_values=num_values,
            num_nulls=h.num_nulls,
            rep_levels=rep_levels,
            def_levels=def_levels,
            values_body_offset=rep_len + def_len,
        )

    def _assemble(
        self,
        *,
        encoding_value: int,
        values_buf: bytes,
        num_values: int,
        num_nulls: int,
        rep_levels: RleBitPackedStream | None,
        def_levels: RleBitPackedStream | None,
        values_body_offset: int,
    ) -> DecodedPage:
        """Decode the values section (shared by V1/V2) and assemble the
        :class:`DecodedPage`. ``num_nulls`` values are absent from the
        section, so it carries exactly ``num_values - num_nulls`` values."""
        encoding = _ENCODING_NAMES.get(encoding_value, str(encoding_value))
        values = self._decode_values_section(
            encoding, values_buf, num_values - num_nulls
        )
        return DecodedPage(
            encoding=encoding,
            num_values=num_values,
            num_nulls=num_nulls,
            repetition_levels=rep_levels,
            definition_levels=def_levels,
            values=values,
            values_body_offset=values_body_offset,
        )

    def _decode_values_section(
        self, encoding: str, values_buf: bytes, num_non_null: int
    ) -> PlainValues | RleBitPackedStream:
        """Decode the values section into its encoding's own structure.

        PLAIN yields a :class:`PlainValues` (the values verbatim); a
        dictionary encoding yields an
        :class:`~parquet_analyzer.decoders.RleBitPackedStream` of the raw
        indices (the leading 1-byte ``bit_width`` + the RLE/bit-packed run
        structure) — the same encoding levels use. Resolving the indices to
        values is deferred to :meth:`physical_values`."""
        cc = self._cc
        if encoding == "PLAIN":
            return PlainValues(
                values=tuple(
                    decode_plain(values_buf, cc.type, num_non_null, cc.type_length)
                )
            )
        if encoding in _DICTIONARY_ENCODINGS:
            bit_width = values_buf[0] if values_buf else 0
            index_bytes = values_buf[1:] if values_buf else b""
            return decode_rle_bitpacked_hybrid_stream(
                index_bytes, bit_width, num_non_null
            )
        raise UnsupportedEncodingError(encoding)

    @staticmethod
    def _require_rle_level_encoding(
        encoding_value: int, max_level: int, which: str
    ) -> None:
        """V1 level blocks are RLE/bit-packed-hybrid; raise on a deprecated
        ``BIT_PACKED`` (or other) level encoding the decoder can't read.
        A column with ``max_level == 0`` has no level block, so its declared
        level encoding is irrelevant and not checked."""
        if max_level == 0:
            return
        name = _ENCODING_NAMES.get(encoding_value, str(encoding_value))
        if name != "RLE":
            raise UnsupportedEncodingError(name, context=f"{which} level")

    def _decode_dictionary_entries(self) -> list[Any]:
        """Decode this (dictionary) page's entries to physical-type values.
        Dictionary pages are PLAIN-encoded over the whole (single,
        possibly-compressed) body."""
        h = self._t.dictionary_page_header
        encoding = _ENCODING_NAMES.get(h.encoding, str(h.encoding))
        if encoding not in ("PLAIN", "PLAIN_DICTIONARY"):
            raise UnsupportedEncodingError(encoding, context="dictionary")
        body = _decompress(
            self.raw_body(), self._cc.codec, self._t.uncompressed_page_size
        )
        return decode_plain(body, self._cc.type, h.num_values, self._cc.type_length)

    # ----- Tree-node interface (docs/tree-schema.md v0) --------------------

    @property
    def _kind(self) -> str:
        # Map the parquet thrift PageType enum to the v0 schema's
        # page-kind string. INDEX_PAGE (1) is rare/deprecated; v0 has no
        # dedicated kind, so it falls through to ``data_page_v1`` to
        # avoid synthesising an unknown — every page in a v0-supported
        # file is one of dictionary / v1 / v2.
        t = self._t.type
        if t == _ThriftPageType.DICTIONARY_PAGE:
            return "dictionary_page"
        if t == _ThriftPageType.DATA_PAGE_V2:
            return "data_page_v2"
        return "data_page_v1"

    @property
    def _offset(self) -> int:
        return self._segment["offset"]

    @property
    def _length(self) -> int:
        # ``segment["length"]`` is the header thrift length;
        # ``compressed_page_size`` is the body length immediately after.
        # The node covers both per docs/tree-schema.md (page = header +
        # body).
        return self._segment["length"] + self._t.compressed_page_size

    def to_json(self, *, view: str = "tree", depth: Any = "all") -> dict:
        """Serialize this page as v0 tree-schema JSON. See
        :meth:`ParquetFile.to_json`."""
        return _to_json_root(self, view, depth)


# ---------------------------------------------------------------------------
# Footer-segment helpers (row-group / column-chunk extent extraction)
# ---------------------------------------------------------------------------


def _extract_row_group_extents(
    footer_segment: dict, rg_thrifts: list
) -> list[tuple[int, int]]:
    """Return ``(offset, length)`` for each row group's thrift struct
    inside the footer.

    Walks the offset-recorded footer segment to locate the ``row_groups``
    list and pull each element's ``offset`` / ``length``. Raises
    ``ValueError`` if the segment's row-group count doesn't match the
    parsed thrift — an internal inconsistency between the thrift decode
    and the offset recorder, never expected for a well-formed footer.
    """
    field = _find_field(footer_segment, "row_groups")
    elements = (field.get("value") if field else None) or []
    if len(elements) != len(rg_thrifts):
        raise ValueError(
            f"row-group count mismatch: footer segment has {len(elements)} "
            f"elements, thrift has {len(rg_thrifts)}"
        )
    return [(el["offset"], el["length"]) for el in elements]


def _extract_column_chunk_extents(
    footer_segment: dict,
    rg_index: int,
    cc_thrifts: list,
) -> list[tuple[int, int]]:
    """Return ``(offset, length)`` for each column chunk's thrift struct
    inside the footer, for the row group at ``rg_index``.

    Walks the footer segment to the row-group element at ``rg_index``,
    then its ``columns`` list. Raises ``ValueError`` if the segment lacks
    that row group or its column count doesn't match the parsed thrift —
    an internal inconsistency never expected for a well-formed footer.
    """
    rg_field = _find_field(footer_segment, "row_groups")
    rg_elements = (rg_field.get("value") if rg_field else None) or []
    if rg_index >= len(rg_elements):
        raise ValueError(
            f"footer segment missing extents for row group {rg_index} "
            f"(segment has {len(rg_elements)} row groups)"
        )
    columns_field = _find_field(rg_elements[rg_index], "columns")
    cc_elements = (columns_field.get("value") if columns_field else None) or []
    if len(cc_elements) != len(cc_thrifts):
        raise ValueError(
            f"column-chunk count mismatch in row group {rg_index}: "
            f"footer segment has {len(cc_elements)} elements, thrift has "
            f"{len(cc_thrifts)}"
        )
    return [(el["offset"], el["length"]) for el in cc_elements]
