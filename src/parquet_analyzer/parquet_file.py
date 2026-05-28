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
from typing import Any

from parquet.ttypes import (
    ColumnChunk as _ThriftColumnChunk,
    CompressionCodec as _ThriftCodec,
    Encoding as _ThriftEncoding,
    OffsetIndex as _ThriftOffsetIndex,
    PageHeader as _ThriftPageHeader,
    PageType as _ThriftPageType,
    RowGroup as _ThriftRowGroup,
    Type as _ThriftType,
)

from ._core import (
    _compute_pages,
    _compute_summary,
    _parse_footer,
    _walk_chunks_eager,
    fill_gaps,
    read_thrift_segment,
    segment_to_json,
)

__all__ = ["ColumnChunk", "Page", "ParquetFile", "RowGroup"]

logger = logging.getLogger(__name__)


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

    Raises:
        ValueError: if the file is missing the ``PAR1`` header or trailer
            magic, or the footer thrift cannot be parsed.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._f = open(path, "rb")
        try:
            self._f.seek(0, 2)
            self._file_size = self._f.tell()

            (
                self._footer_thrift,
                self._footer_segment,
                self._footer_offset,
                self._header_magic_segment,
                self._trailer_segments,
            ) = _parse_footer(self._f, self._file_size)
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
            self._row_groups_cache = tuple(
                RowGroup(self, rg) for rg in (self._footer_thrift.row_groups or [])
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
        self, parquet_file: "ParquetFile", thrift_obj: _ThriftRowGroup
    ) -> None:
        self._pf = parquet_file
        self._t = thrift_obj
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
            self._columns_cache = tuple(
                ColumnChunk(self._pf, self, cc) for cc in (self._t.columns or [])
            )
        return self._columns_cache


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


class ColumnChunk:
    """Lazy wrapper around a single column chunk's metadata.

    Constructed by :attr:`RowGroup.columns`. All properties are
    footer-derived (cheap, no body reads). The :meth:`pages` accessor
    walks per-chunk page headers on first call (Phase 2 lazy boundary —
    cheaper than full-file walk; only this chunk's headers are parsed).
    """

    def __init__(
        self,
        parquet_file: "ParquetFile",
        row_group: "RowGroup",
        thrift_obj: _ThriftColumnChunk,
    ) -> None:
        self._pf = parquet_file
        self._rg = row_group
        self._t = thrift_obj
        self._md = thrift_obj.meta_data
        self._pages_cache: tuple[Page, ...] | None = None
        self._offset_index_cache: _ThriftOffsetIndex | None = None

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

        When ``True``, :attr:`num_pages` and (future Slice 4) ``page(index)``
        can serve queries via a single small thrift parse without walking
        every page header — typically 50-200 bytes per page entry in the
        OffsetIndex.

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

    def _read_offset_index(self) -> _ThriftOffsetIndex:
        """Read + cache the OffsetIndex thrift struct for this chunk.

        Caller must check :attr:`has_offset_index` first; this raises
        ``ValueError`` if the chunk has none. The cached object is
        returned on subsequent calls.

        Slice 4 (#8) will expose a public ``offset_index`` property that
        returns this object; for now it stays private since the only
        consumer in this PR is :attr:`num_pages`.
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

    # ----- Page-header walking (Phase 2 lazy boundary) ---------------------

    def pages(self) -> tuple["Page", ...]:
        """Walk this column chunk's page headers on first call; cache and
        return :class:`Page` wrappers.

        Phase 2 of the lazy parsing work: per-chunk page walking is cheap
        because it only touches one column's pages, not the whole file.
        Page bodies are NOT read — only the per-page Thrift header is
        parsed. Body access (raw_bytes / decompress / decode) will be added
        with the Slice 4 CLI surface.
        """
        if self._pages_cache is None:
            pages: list[Page] = []
            remaining_values = self._md.num_values
            if self._md.dictionary_page_offset:
                offset = self._md.dictionary_page_offset
            else:
                offset = self._md.data_page_offset
            f = self._pf._f
            while remaining_values > 0:
                page_thrift, page_segment = read_thrift_segment(
                    f, offset, "page", _ThriftPageHeader
                )
                pages.append(Page(self._pf, self, page_thrift, page_segment))
                page_header_end = page_segment["offset"] + page_segment["length"]
                if page_thrift.data_page_header is not None:
                    num_values_read = page_thrift.data_page_header.num_values
                elif page_thrift.data_page_header_v2 is not None:
                    num_values_read = page_thrift.data_page_header_v2.num_values
                elif page_thrift.dictionary_page_header is not None:
                    num_values_read = 0
                else:
                    break
                remaining_values -= num_values_read
                offset = page_header_end + page_thrift.compressed_page_size
            self._pages_cache = tuple(pages)
        return self._pages_cache


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class Page:
    """Lazy wrapper around a single page's metadata.

    Constructed by :meth:`ColumnChunk.pages`. All properties are
    header-derived (cheap, no body reads).

    Page body access (``raw_bytes()``, ``decompressed_bytes()``,
    ``decode_values()``) is intentionally not exposed in this PR — that's
    the Slice 4 (#8) surface that wires the existing decoder primitives
    in :mod:`parquet_analyzer.decoders` to a CLI verb. Adding it later
    will not change any of the properties below.
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
