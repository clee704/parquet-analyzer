"""Decoders for Parquet encoded streams.

This module exposes the byte-level primitives every parquet inspection workflow
needs: decompression, the RLE/bit-packed-hybrid stream decoder used for
dictionary indices and definition/repetition levels, and the PLAIN value
decoder.

The functions here are stateless and operate on raw ``bytes`` slices, so they
compose cleanly with the byte ranges already exposed by
:func:`parquet_analyzer.parse_parquet_file` (``--output-mode segments``).

Encoding-level gotchas worth knowing
------------------------------------

* **V1 vs V2 data pages.** V1 pages prefix each level block with a 4-byte
  little-endian length; V2 pages instead record
  ``definition_levels_byte_length`` / ``repetition_levels_byte_length`` in the
  page header and store the streams unprefixed. :func:`decode_v1_level_block`
  handles the V1 prefix; callers reading V2 pages should slice using the
  header-supplied lengths and call :func:`decode_levels` directly.

* **Required columns have no level block.** When ``max_def_level == 0`` (a
  required field), the file contains no def-level bytes at all. Same for
  repetition: rep-levels exist only for repeated columns.
  :func:`decode_levels` returns ``[0] * num_values`` in that case.

* **V2 level streams are uncompressed.** Only the values section of a V2 page
  may be compressed (and only when ``is_compressed`` is true). Do not pass an
  entire V2 page body to :func:`decompress` as one block.

* **Per-page dictionary-index bit-width byte.** Dictionary-encoded data pages
  start their indices block with a 1-byte ``bit_width`` value chosen by the
  writer (it may exceed ``ceil(log2(dict_size))``). Read that byte first, then
  call :func:`decode_rle_bitpacked_hybrid`.

* **Bit-packed run alignment.** Bit-packed runs encode groups of 8 values
  packed at ``bit_width`` bits each. The next varint header begins at the next
  byte boundary; any leftover bits inside the last byte of a run are
  discarded.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import cramjam  # type: ignore[import-untyped]

__all__ = [
    "DecodeStats",
    "decode_levels",
    "decode_plain",
    "decode_rle_bitpacked_hybrid",
    "decode_v1_level_block",
    "decompress",
]


@dataclass(frozen=True)
class DecodeStats:
    """Run-classification stats produced by :func:`decode_rle_bitpacked_hybrid`.

    ``rle_run_values`` records the value of each RLE-run header in stream
    order. Bit-packed runs do not contribute to ``rle_run_values`` — value
    transitions inside a bit-packed run are visible only by inspecting the
    decoded values themselves.

    ``rle_run_lengths`` and ``bit_packed_run_lengths`` record the
    **encoder-declared** run lengths (the value count the run header claims),
    not the emitted-to-``values`` counts. For pages whose ``num_values`` is
    not a multiple of 8, the trailing bit-packed run reports its declared
    ``num_groups * 8`` length even though :func:`decode_rle_bitpacked_hybrid`
    truncates the emitted values to ``num_values``. As a result,
    ``sum(rle_run_lengths) + sum(bit_packed_run_lengths)`` is generally
    ``>= num_values``, not equal to it. This preserves "what the encoder
    actually wrote on disk" inspection semantics — knowing the on-disk run
    length is more useful for verifying writer behaviour than knowing the
    post-truncation emitted count.
    """

    rle_run_count: int
    bit_packed_run_count: int
    rle_run_lengths: tuple[int, ...]
    rle_run_values: tuple[int, ...]
    bit_packed_run_lengths: tuple[int, ...]


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------


_SUPPORTED_CODECS = frozenset(
    {"UNCOMPRESSED", "SNAPPY", "GZIP", "ZSTD", "LZ4", "LZ4_RAW"}
)
_KNOWN_UNSUPPORTED_CODECS = frozenset({"LZO", "BROTLI"})


def decompress(data: bytes, codec: str, uncompressed_size: int) -> bytes:
    """Decompress a compressed byte slice using a Parquet codec name.

    ``codec`` is the Parquet ``CompressionCodec`` enum name. The Parquet spec
    defines ``UNCOMPRESSED``, ``SNAPPY``, ``GZIP``, ``LZO``, ``BROTLI``,
    ``LZ4`` (legacy Hadoop-framed), ``ZSTD``, and ``LZ4_RAW``. This function
    supports everything except ``LZO`` and ``BROTLI`` (which would each pull in
    an additional native dependency).

    The function operates on a single compressed slice — for V2 data pages
    that means the encoded-values section only, not the whole page body. The
    decompressed output is verified against ``uncompressed_size`` and a
    ``ValueError`` is raised on mismatch.

    Raises:
        ValueError: ``uncompressed_size`` is negative, or the decompressed
            output does not match it.
        NotImplementedError: ``codec`` is recognised but unsupported here
            (``LZO``, ``BROTLI``), or unknown entirely.
    """
    if uncompressed_size < 0:
        raise ValueError(f"uncompressed_size must be >= 0, got {uncompressed_size}")

    codec_upper = codec.upper()
    if codec_upper in _KNOWN_UNSUPPORTED_CODECS:
        raise NotImplementedError(
            f"Parquet codec {codec_upper!r} is recognised but not supported by "
            "this decoder; rewrite the file with one of "
            f"{sorted(_SUPPORTED_CODECS)}."
        )
    if codec_upper not in _SUPPORTED_CODECS:
        raise NotImplementedError(
            f"Unknown Parquet codec {codec!r}; expected one of "
            f"{sorted(_SUPPORTED_CODECS | _KNOWN_UNSUPPORTED_CODECS)}."
        )

    if codec_upper == "UNCOMPRESSED":
        out = bytes(data)
    elif codec_upper == "SNAPPY":
        out = bytes(cramjam.snappy.decompress_raw(data))
    elif codec_upper == "GZIP":
        out = bytes(cramjam.gzip.decompress(data))
    elif codec_upper == "ZSTD":
        out = bytes(cramjam.zstd.decompress(data))
    elif codec_upper == "LZ4_RAW":
        out = bytes(cramjam.lz4.decompress_block(data, output_len=uncompressed_size))
    else:  # codec_upper == "LZ4"
        out = _decompress_lz4_hadoop(data, uncompressed_size)

    if len(out) != uncompressed_size:
        raise ValueError(
            f"{codec_upper} decompression produced {len(out)} bytes, "
            f"expected {uncompressed_size}"
        )
    return out


def _decompress_lz4_hadoop(data: bytes, uncompressed_size: int) -> bytes:
    """Decode the legacy ``LZ4`` codec used by Hadoop/older Parquet writers.

    Each block is prefixed with two 4-byte big-endian integers: total
    uncompressed length, then compressed length. A page may contain multiple
    such blocks concatenated until ``uncompressed_size`` bytes are produced.
    """
    out = bytearray()
    pos = 0
    end = len(data)
    while len(out) < uncompressed_size:
        if pos + 8 > end:
            raise ValueError(
                f"truncated LZ4 (Hadoop) frame at offset {pos}: "
                f"need 8 header bytes, have {end - pos}"
            )
        block_uncompressed, block_compressed = struct.unpack_from(">II", data, pos)
        pos += 8
        if pos + block_compressed > end:
            raise ValueError(
                f"truncated LZ4 (Hadoop) block at offset {pos}: "
                f"need {block_compressed} bytes, have {end - pos}"
            )
        block = bytes(
            cramjam.lz4.decompress_block(
                data[pos : pos + block_compressed],
                output_len=block_uncompressed,
            )
        )
        if len(block) != block_uncompressed:
            raise ValueError(
                f"LZ4 (Hadoop) block decoded to {len(block)} bytes, "
                f"header said {block_uncompressed}"
            )
        out.extend(block)
        pos += block_compressed
    return bytes(out)


# ---------------------------------------------------------------------------
# RLE / bit-packed-hybrid
# ---------------------------------------------------------------------------


def _read_varint(buf: bytes, pos: int, end: int) -> tuple[int, int]:
    """Read an unsigned base-128 varint. Returns ``(value, new_pos)``."""
    result = 0
    shift = 0
    while True:
        if pos >= end:
            raise ValueError(
                f"truncated varint at offset {pos}: ran past end-of-buffer ({end})"
            )
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError(f"varint at offset {pos} exceeds 64 bits")


def decode_rle_bitpacked_hybrid(
    data: bytes, bit_width: int, num_values: int
) -> tuple[list[int], DecodeStats]:
    """Decode a raw RLE/bit-packed-hybrid stream.

    The hybrid encoding interleaves two run kinds: literal RLE runs (one
    repeated value) and bit-packed runs (groups of 8 values packed at
    ``bit_width`` bits each). This is the encoding used for dictionary
    indices on data pages and for definition/repetition levels.

    The function operates on a *raw* stream: no length prefix, no leading
    bit-width byte. Callers reading a dictionary-encoded data page must read
    the 1-byte ``bit_width`` byte themselves before calling here; callers
    reading a V1 level block should use :func:`decode_v1_level_block` instead.

    Reads until ``num_values`` values have been produced; bit-packed runs are
    truncated to exactly ``num_values``. Raises ``ValueError`` if the buffer
    is exhausted before ``num_values`` values are produced.

    Args:
        data: raw RLE/bit-packed-hybrid stream bytes (no length prefix).
        bit_width: number of bits per packed value. Must satisfy
            ``0 <= bit_width <= 64``. ``bit_width == 0`` is valid and means
            every value is 0 (no bytes are consumed per RLE run).
        num_values: number of values the stream should produce.

    Returns:
        Tuple of ``(values, stats)`` where ``values`` has length
        ``num_values`` (or 0 if ``num_values == 0``) and ``stats`` records
        the run classification.

    Raises:
        ValueError: invalid arguments or truncated stream.
    """
    if bit_width < 0 or bit_width > 64:
        raise ValueError(f"bit_width must be in [0, 64], got {bit_width}")
    if num_values < 0:
        raise ValueError(f"num_values must be >= 0, got {num_values}")

    values: list[int] = []
    rle_run_lengths: list[int] = []
    rle_run_values: list[int] = []
    bit_packed_run_lengths: list[int] = []

    pos = 0
    end = len(data)
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1 if bit_width > 0 else 0

    while len(values) < num_values:
        if pos >= end:
            raise ValueError(
                f"RLE/bit-packed-hybrid stream truncated: produced {len(values)} "
                f"of {num_values} values before end-of-buffer"
            )
        header, pos = _read_varint(data, pos, end)
        if (header & 1) == 0:
            run_len = header >> 1
            if pos + byte_width > end:
                raise ValueError(
                    f"truncated RLE run at offset {pos}: need {byte_width} "
                    f"value bytes, have {end - pos}"
                )
            val = 0
            for i in range(byte_width):
                val |= data[pos + i] << (8 * i)
            pos += byte_width
            take = min(run_len, num_values - len(values))
            values.extend([val] * take)
            rle_run_lengths.append(run_len)
            rle_run_values.append(val)
        else:
            num_groups = header >> 1
            run_len = num_groups * 8
            bit_packed_run_lengths.append(run_len)
            bits_total = num_groups * 8 * bit_width
            bytes_total = (bits_total + 7) // 8
            if pos + bytes_total > end:
                raise ValueError(
                    f"truncated bit-packed run at offset {pos}: need "
                    f"{bytes_total} bytes, have {end - pos}"
                )
            bit_buf = 0
            bit_count = 0
            take = min(run_len, num_values - len(values))
            for _ in range(take):
                while bit_count < bit_width:
                    bit_buf |= data[pos] << bit_count
                    pos += 1
                    bit_count += 8
                values.append(bit_buf & mask)
                bit_buf >>= bit_width
                bit_count -= bit_width
            # When `take < run_len` the bit-packed run is being truncated to
            # land on `num_values`. The outer loop exits immediately after
            # this branch in that case, so we don't need to advance `pos`
            # past the remaining (unused) bytes of the run — the next varint
            # header will never be read.

    stats = DecodeStats(
        rle_run_count=len(rle_run_lengths),
        bit_packed_run_count=len(bit_packed_run_lengths),
        rle_run_lengths=tuple(rle_run_lengths),
        rle_run_values=tuple(rle_run_values),
        bit_packed_run_lengths=tuple(bit_packed_run_lengths),
    )
    return values, stats


def decode_levels(data: bytes, max_level: int, num_values: int) -> list[int]:
    """Decode a definition-level or repetition-level RLE stream.

    The bit width is derived from ``max_level`` (``max_level.bit_length()``),
    matching the Parquet spec: a column with ``max_def_level == 3`` packs
    levels at 2 bits each, ``max_def_level == 7`` packs at 3 bits, etc.

    ``data`` is the raw level stream — no length prefix. V1 data pages prefix
    the stream with a 4-byte little-endian length; use
    :func:`decode_v1_level_block` for those.

    When ``max_level == 0`` (a required column with no level block in the
    file), returns ``[0] * num_values`` without reading any bytes.

    Args:
        data: raw level stream bytes (no length prefix).
        max_level: maximum level value for this column. Must be >= 0.
        num_values: number of levels to decode.

    Returns:
        List of length ``num_values`` containing the decoded levels.

    Raises:
        ValueError: ``max_level`` is negative, or the stream is truncated.
    """
    if max_level < 0:
        raise ValueError(f"max_level must be >= 0, got {max_level}")
    if num_values < 0:
        raise ValueError(f"num_values must be >= 0, got {num_values}")
    if max_level == 0:
        return [0] * num_values
    bit_width = max_level.bit_length()
    values, _ = decode_rle_bitpacked_hybrid(data, bit_width, num_values)
    return values


def decode_v1_level_block(
    data: bytes, offset: int, max_level: int, num_values: int
) -> tuple[list[int], int]:
    """Decode a V1 data page level block (4-byte LE length prefix + RLE stream).

    V1 data pages serialize each level block as ``[4-byte LE length][RLE
    stream]``. V2 data pages do *not* — they store level byte lengths in the
    page header instead. Use this helper for V1; for V2, slice ``data`` using
    the header lengths and call :func:`decode_levels` directly.

    When ``max_level == 0`` no block exists in the file; this helper returns
    ``([0] * num_values, offset)`` without consuming any bytes.

    Args:
        data: a buffer containing (at least) the level block at ``offset``.
        offset: byte offset of the level block within ``data``.
        max_level: maximum level value for this column.
        num_values: number of levels to decode.

    Returns:
        Tuple of ``(levels, new_offset)`` where ``new_offset`` is positioned
        just past the level block (or equal to ``offset`` for max_level == 0).

    Raises:
        ValueError: ``max_level`` or ``num_values`` negative, ``offset`` out
            of range, truncated length prefix, or the level block's RLE
            stream is malformed.
    """
    if max_level < 0:
        raise ValueError(f"max_level must be >= 0, got {max_level}")
    if num_values < 0:
        raise ValueError(f"num_values must be >= 0, got {num_values}")
    if offset < 0 or offset > len(data):
        raise ValueError(
            f"offset {offset} out of range for buffer of length {len(data)}"
        )
    if max_level == 0:
        return [0] * num_values, offset
    if offset + 4 > len(data):
        raise ValueError(
            f"truncated V1 level-block length prefix at offset {offset}: "
            f"need 4 bytes, have {len(data) - offset}"
        )
    (block_len,) = struct.unpack_from("<I", data, offset)
    block_start = offset + 4
    block_end = block_start + block_len
    if block_end > len(data):
        raise ValueError(
            f"V1 level block at offset {offset} claims {block_len} bytes "
            f"but only {len(data) - block_start} remain"
        )
    levels = decode_levels(data[block_start:block_end], max_level, num_values)
    return levels, block_end


# ---------------------------------------------------------------------------
# PLAIN values
# ---------------------------------------------------------------------------


_PLAIN_FIXED_WIDTH: dict[str, tuple[int, str]] = {
    "INT32": (4, "<i"),
    "INT64": (8, "<q"),
    "FLOAT": (4, "<f"),
    "DOUBLE": (8, "<d"),
}
_KNOWN_PARQUET_TYPES = frozenset(
    {
        "BOOLEAN",
        "INT32",
        "INT64",
        "INT96",
        "FLOAT",
        "DOUBLE",
        "BYTE_ARRAY",
        "FIXED_LEN_BYTE_ARRAY",
    }
)


def decode_plain(
    data: bytes,
    parquet_type: str,
    num_values: int,
    type_length: int | None = None,
) -> list[Any]:
    """Decode PLAIN-encoded values.

    Supported ``parquet_type`` values (Parquet ``Type`` enum names):

    - ``BOOLEAN`` — bit-packed LSB-first, 1 bit per value. Returns
      ``list[bool]``.
    - ``INT32`` / ``INT64`` — little-endian signed integers. Returns
      ``list[int]``.
    - ``INT96`` — 12 raw bytes per value (deprecated timestamp encoding).
      Returns ``list[bytes]`` of 12-byte chunks; interpretation is the
      caller's responsibility.
    - ``FLOAT`` / ``DOUBLE`` — little-endian IEEE 754. Returns ``list[float]``.
    - ``BYTE_ARRAY`` — each value prefixed with a 4-byte LE length. Returns
      ``list[bytes]``.
    - ``FIXED_LEN_BYTE_ARRAY`` — ``type_length`` bytes per value (must be
      passed in). Returns ``list[bytes]``.

    Args:
        data: raw PLAIN-encoded bytes.
        parquet_type: Parquet ``Type`` enum name.
        num_values: number of values to decode.
        type_length: required for ``FIXED_LEN_BYTE_ARRAY``; ignored otherwise.

    Raises:
        ValueError: arguments are invalid, ``data`` is truncated, or a
            ``BYTE_ARRAY`` length prefix is malformed.
        NotImplementedError: ``parquet_type`` is not one of the supported
            names.
    """
    if num_values < 0:
        raise ValueError(f"num_values must be >= 0, got {num_values}")
    ptype = parquet_type.upper()
    if ptype not in _KNOWN_PARQUET_TYPES:
        raise NotImplementedError(
            f"Unknown Parquet type {parquet_type!r}; expected one of "
            f"{sorted(_KNOWN_PARQUET_TYPES)}."
        )

    if num_values == 0:
        return []

    if ptype == "BOOLEAN":
        return _decode_plain_boolean(data, num_values)
    if ptype in _PLAIN_FIXED_WIDTH:
        return _decode_plain_fixed_width(data, ptype, num_values)
    if ptype == "INT96":
        return _decode_plain_int96(data, num_values)
    if ptype == "BYTE_ARRAY":
        return _decode_plain_byte_array(data, num_values)
    # ptype == "FIXED_LEN_BYTE_ARRAY"
    if type_length is None:
        raise ValueError("type_length is required for FIXED_LEN_BYTE_ARRAY")
    if type_length < 0:
        raise ValueError(f"type_length must be >= 0, got {type_length}")
    return _decode_plain_fixed_len_byte_array(data, num_values, type_length)


def _decode_plain_fixed_width(
    data: bytes, parquet_type: str, num_values: int
) -> list[Any]:
    width, fmt = _PLAIN_FIXED_WIDTH[parquet_type]
    total = width * num_values
    if total > len(data):
        raise ValueError(
            f"PLAIN {parquet_type} truncated: need {total} bytes for "
            f"{num_values} values, have {len(data)}"
        )
    return [struct.unpack_from(fmt, data, i * width)[0] for i in range(num_values)]


def _decode_plain_int96(data: bytes, num_values: int) -> list[bytes]:
    total = 12 * num_values
    if total > len(data):
        raise ValueError(
            f"PLAIN INT96 truncated: need {total} bytes for {num_values} "
            f"values, have {len(data)}"
        )
    return [bytes(data[i * 12 : (i + 1) * 12]) for i in range(num_values)]


def _decode_plain_boolean(data: bytes, num_values: int) -> list[bool]:
    needed_bytes = (num_values + 7) // 8
    if needed_bytes > len(data):
        raise ValueError(
            f"PLAIN BOOLEAN truncated: need {needed_bytes} bytes for "
            f"{num_values} values, have {len(data)}"
        )
    values: list[bool] = []
    for i in range(num_values):
        byte = data[i // 8]
        values.append(bool((byte >> (i % 8)) & 1))
    return values


def _decode_plain_byte_array(data: bytes, num_values: int) -> list[bytes]:
    values: list[bytes] = []
    pos = 0
    end = len(data)
    for i in range(num_values):
        if pos + 4 > end:
            raise ValueError(
                f"PLAIN BYTE_ARRAY truncated: value {i} length prefix needs "
                f"4 bytes at offset {pos}, have {end - pos}"
            )
        (length,) = struct.unpack_from("<i", data, pos)
        pos += 4
        if length < 0:
            raise ValueError(f"PLAIN BYTE_ARRAY value {i} has negative length {length}")
        if pos + length > end:
            raise ValueError(
                f"PLAIN BYTE_ARRAY truncated: value {i} declares {length} "
                f"bytes at offset {pos}, have {end - pos}"
            )
        values.append(bytes(data[pos : pos + length]))
        pos += length
    return values


def _decode_plain_fixed_len_byte_array(
    data: bytes, num_values: int, type_length: int
) -> list[bytes]:
    total = type_length * num_values
    if total > len(data):
        raise ValueError(
            f"PLAIN FIXED_LEN_BYTE_ARRAY truncated: need {total} bytes for "
            f"{num_values} values of length {type_length}, have {len(data)}"
        )
    return [
        bytes(data[i * type_length : (i + 1) * type_length]) for i in range(num_values)
    ]
