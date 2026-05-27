"""Tests for :mod:`parquet_analyzer.decoders`.

Strategy: round-trip through pyarrow-generated parquet files wherever possible.
Page byte ranges are discovered with :func:`parquet_analyzer.parse_parquet_file`
(the existing thrift parser), so tests exercise the same code path real callers
will use. A small number of argument-validation tests do not need any bytes at
all; the only place hand-encoded bytes appear is one LZ4-Hadoop multi-block
test where the encoder is :mod:`cramjam` itself (not hand-rolled by us) and the
goal is to exercise the multi-block branch that single-page parquet output
doesn't reliably trigger.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import pytest

cramjam = pytest.importorskip("cramjam")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from parquet_analyzer._core import parse_parquet_file
from parquet_analyzer.decoders import (
    DecodeStats,
    decode_levels,
    decode_plain,
    decode_rle_bitpacked_hybrid,
    decode_v1_level_block,
    decompress,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_pages(path: Path) -> list[dict]:
    """Walk segments to find every page; return ``(header_info, body_bytes)``.

    Each dict has: ``page_type``, ``encoding``, ``num_values``,
    ``uncompressed_size``, ``compressed_size``, ``data_offset``,
    ``data_length``, ``data`` (the raw compressed bytes for this page).
    """
    segments, _ = parse_parquet_file(str(path))
    file_bytes = path.read_bytes()
    pages: list[dict] = []
    for i, s in enumerate(segments):
        if s["name"] != "page":
            continue
        info: dict = {
            "page_type": None,
            "encoding": None,
            "num_values": None,
            "uncompressed_size": None,
            "compressed_size": None,
        }
        for sub in s["value"]:
            if sub["name"] == "type":
                info["page_type"] = sub["metadata"]["enum_name"]
            elif sub["name"] == "uncompressed_page_size":
                info["uncompressed_size"] = sub["value"]
            elif sub["name"] == "compressed_page_size":
                info["compressed_size"] = sub["value"]
            elif sub["name"] in (
                "data_page_header",
                "dictionary_page_header",
                "data_page_header_v2",
            ):
                for k in sub["value"]:
                    if k["name"] == "encoding":
                        info["encoding"] = k["metadata"]["enum_name"]
                    elif k["name"] == "num_values":
                        info["num_values"] = k["value"]
        if i + 1 < len(segments) and segments[i + 1]["name"] == "page_data":
            d = segments[i + 1]
            info["data_offset"] = d["offset"]
            info["data_length"] = d["length"]
            info["data"] = file_bytes[d["offset"] : d["offset"] + d["length"]]
            pages.append(info)
    return pages


def _first_data_page(path: Path) -> dict:
    pages = _read_pages(path)
    for p in pages:
        if p["page_type"] == "DATA_PAGE":
            return p
    raise AssertionError(f"no V1 DATA_PAGE in {path}")


def _first_data_page_v2(path: Path) -> dict:
    """Find the first V2 data page and capture its V2-specific header fields.

    Augments the base ``_first_data_page`` shape with
    ``definition_levels_byte_length``, ``repetition_levels_byte_length``, and
    ``is_compressed`` — the page-header fields a V2 caller needs to slice
    the body into ``[rep_levels][def_levels][values]``.
    """
    segments, _ = parse_parquet_file(str(path))
    file_bytes = path.read_bytes()
    for i, s in enumerate(segments):
        if s["name"] != "page":
            continue
        page_type = None
        v2_info: dict = {
            "num_values": None,
            "encoding": None,
            "definition_levels_byte_length": None,
            "repetition_levels_byte_length": None,
            "is_compressed": None,
        }
        uncompressed_size = compressed_size = None
        for sub in s["value"]:
            if sub["name"] == "type":
                page_type = sub["metadata"]["enum_name"]
            elif sub["name"] == "uncompressed_page_size":
                uncompressed_size = sub["value"]
            elif sub["name"] == "compressed_page_size":
                compressed_size = sub["value"]
            elif sub["name"] == "data_page_header_v2":
                for k in sub["value"]:
                    name = k["name"]
                    if name in v2_info:
                        v2_info[name] = (
                            k["metadata"]["enum_name"]
                            if name == "encoding"
                            else k["value"]
                        )
        if page_type != "DATA_PAGE_V2":
            continue
        if i + 1 >= len(segments) or segments[i + 1]["name"] != "page_data":
            continue
        d = segments[i + 1]
        return {
            "page_type": page_type,
            "uncompressed_size": uncompressed_size,
            "compressed_size": compressed_size,
            "data_offset": d["offset"],
            "data_length": d["length"],
            "data": file_bytes[d["offset"] : d["offset"] + d["length"]],
            **v2_info,
        }
    raise AssertionError(f"no V2 DATA_PAGE_V2 in {path}")


def _write_simple(path: Path, *, compression: str, num_rows: int = 1000) -> Path:
    table = pa.table({"x": pa.array(list(range(num_rows)), type=pa.int32())})
    pq.write_table(table, path, compression=compression, use_dictionary=False)
    return path


_PARQUET_CODEC_FOR_PYARROW = {
    "none": "UNCOMPRESSED",
    "snappy": "SNAPPY",
    "gzip": "GZIP",
    "zstd": "ZSTD",
    "lz4_raw": "LZ4_RAW",
}


# ---------------------------------------------------------------------------
# decompress() — round-trip each codec via pyarrow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pyarrow_codec",
    ["none", "snappy", "gzip", "zstd", "lz4_raw"],
)
def test_decompress_roundtrip_via_pyarrow(tmp_path, pyarrow_codec):
    path = _write_simple(
        tmp_path / f"{pyarrow_codec}.parquet", compression=pyarrow_codec
    )
    page = _first_data_page(path)
    decompressed = decompress(
        page["data"],
        _PARQUET_CODEC_FOR_PYARROW[pyarrow_codec],
        page["uncompressed_size"],
    )
    assert len(decompressed) == page["uncompressed_size"]


def test_decompress_uncompressed_is_passthrough():
    payload = b"hello world"
    assert decompress(payload, "UNCOMPRESSED", len(payload)) == payload


def test_decompress_codec_name_case_insensitive():
    payload = b"hello world"
    assert decompress(payload, "uncompressed", len(payload)) == payload


def test_decompress_negative_uncompressed_size_raises():
    with pytest.raises(ValueError, match="uncompressed_size"):
        decompress(b"", "SNAPPY", -1)


def test_decompress_unknown_codec_raises():
    with pytest.raises(NotImplementedError, match="Unknown"):
        decompress(b"", "MAGIC", 0)


@pytest.mark.parametrize("codec", ["LZO", "BROTLI"])
def test_decompress_known_unsupported_codec_raises(codec):
    with pytest.raises(NotImplementedError, match="not supported"):
        decompress(b"", codec, 0)


def test_decompress_size_mismatch_raises(tmp_path):
    path = _write_simple(tmp_path / "snappy.parquet", compression="snappy")
    page = _first_data_page(path)
    with pytest.raises(ValueError, match="expected"):
        decompress(page["data"], "SNAPPY", page["uncompressed_size"] + 1)


def test_decompress_lz4_hadoop_multi_block():
    """Exercise the multi-block branch of the LZ4 (Hadoop) frame decoder.

    Parquet pages produced by pyarrow are typically encoded as a single LZ4
    block, so we construct a two-block frame using cramjam directly to cover
    the loop branch. The cramjam encoder is doing all the actual compression
    here — we're just stitching two block-headers together to match the
    legacy Hadoop framing the LZ4 codec uses on disk.
    """
    block_a = b"a" * 100 + b"variable content " * 5
    block_b = b"b" * 200 + b"more content " * 10
    parts: list[bytes] = []
    for block in (block_a, block_b):
        compressed = bytes(cramjam.lz4.compress_block(block, store_size=False))
        parts.append(struct.pack(">II", len(block), len(compressed)))
        parts.append(compressed)
    frame = b"".join(parts)
    out = decompress(frame, "LZ4", len(block_a) + len(block_b))
    assert out == block_a + block_b


def test_decompress_lz4_hadoop_truncated_header_raises():
    with pytest.raises(ValueError, match="truncated"):
        # Only 4 bytes — not enough for the 8-byte header.
        decompress(b"\x00\x00\x00\x10", "LZ4", 16)


def test_decompress_lz4_hadoop_truncated_block_raises():
    # Header declares 100 compressed bytes but only 4 follow.
    bad = struct.pack(">II", 100, 100) + b"abcd"
    with pytest.raises(ValueError, match="truncated"):
        decompress(bad, "LZ4", 100)


# ---------------------------------------------------------------------------
# decode_rle_bitpacked_hybrid() — round-trip via real parquet pages
# ---------------------------------------------------------------------------


def test_rle_decode_dictionary_indices_from_real_page(tmp_path):
    """Real PLAIN_DICTIONARY page → decompress → strip def-level + bit-width →
    decode indices → recover original values via dict lookup.

    Uses a nullable column WITH actual nulls so the test exercises the
    nullability path the README documents (indices count = non-null row
    count, not page num_values). A previous version of this test used a
    null-free fixture which silently passed even when the indices count
    was wrong; the explicit null pattern below would have caught that.
    """
    # 600 rows over 3 distinct strings + nulls — guarantees dict encoding
    # with bit_width >= 2, two-thirds non-null entries in the indices
    # stream (400 non-null + 200 null = 600 def-levels).
    pattern = ["red", "green", None, "blue", None, "red"]
    table = pa.table({"k": pa.array(pattern * 100, type=pa.string())})
    path = tmp_path / "dict.parquet"
    pq.write_table(table, path, use_dictionary=True, compression="snappy")

    pages = _read_pages(path)
    dict_page = next(p for p in pages if p["page_type"] == "DICTIONARY_PAGE")
    data_page = next(p for p in pages if p["page_type"] == "DATA_PAGE")
    assert data_page["encoding"] in ("PLAIN_DICTIONARY", "RLE_DICTIONARY")

    # Decode the dictionary entries (PLAIN BYTE_ARRAY) ourselves.
    dict_raw = decompress(dict_page["data"], "SNAPPY", dict_page["uncompressed_size"])
    dict_values = [
        v.decode("utf-8")
        for v in decode_plain(dict_raw, "BYTE_ARRAY", dict_page["num_values"])
    ]

    # Decode the data page: snappy → strip 4-byte def-level block prefix +
    # block + 1-byte bit_width → RLE/bit-packed-hybrid stream.
    raw = decompress(data_page["data"], "SNAPPY", data_page["uncompressed_size"])
    def_levels, after_def = decode_v1_level_block(
        raw, 0, max_level=1, num_values=data_page["num_values"]
    )
    bit_width = raw[after_def]
    # Indices stream contains entries only for non-null rows (this is what
    # the README's "Indices skip nulls" gotcha documents).
    num_non_null = sum(def_levels)
    assert num_non_null < data_page["num_values"], (
        "fixture must contain nulls so the non-null-vs-total distinction "
        "is exercised — otherwise the bug the README example used to have "
        "(passing num_values instead of num_non_null) would not be caught"
    )
    indices, stats = decode_rle_bitpacked_hybrid(
        raw[after_def + 1 :], bit_width, num_non_null
    )
    # Reassemble nulls back into the column using the def-level stream.
    it = iter(indices)
    decoded = [dict_values[next(it)] if d == 1 else None for d in def_levels]
    assert decoded == pattern * 100
    assert isinstance(stats, DecodeStats)
    assert stats.rle_run_count + stats.bit_packed_run_count > 0


def test_rle_decode_pure_rle_runs(tmp_path):
    """A column with one long run of one value → expect a single RLE run."""
    table = pa.table({"k": pa.array(["only-value"] * 10000, type=pa.string())})
    path = tmp_path / "rle-run.parquet"
    pq.write_table(table, path, use_dictionary=True, compression="snappy")

    pages = _read_pages(path)
    data_page = next(p for p in pages if p["page_type"] == "DATA_PAGE")
    raw = decompress(data_page["data"], "SNAPPY", data_page["uncompressed_size"])
    _, after_def = decode_v1_level_block(
        raw, 0, max_level=1, num_values=data_page["num_values"]
    )
    bit_width = raw[after_def]
    indices, stats = decode_rle_bitpacked_hybrid(
        raw[after_def + 1 :], bit_width, data_page["num_values"]
    )
    assert set(indices) == {0}
    assert stats.rle_run_count >= 1
    assert sum(stats.rle_run_lengths) == data_page["num_values"]


def test_rle_decode_bit_width_zero():
    """A stream with bit_width=0 means every value is 0; the value byte
    contributes 0 bytes to RLE runs. Use a hand-built but minimal stream
    (one RLE-run header) — cannot be exercised via pyarrow because pyarrow
    always picks bit_width >= 1 for dictionary indices."""
    # Header is varint for (run_len << 1) | 0 = (5 << 1) = 10.
    stream = bytes([10])
    values, stats = decode_rle_bitpacked_hybrid(stream, bit_width=0, num_values=5)
    assert values == [0] * 5
    assert stats.rle_run_count == 1
    assert stats.rle_run_lengths == (5,)
    assert stats.rle_run_values == (0,)
    assert stats.bit_packed_run_count == 0


def test_rle_decode_num_values_zero_returns_empty():
    values, stats = decode_rle_bitpacked_hybrid(b"", bit_width=4, num_values=0)
    assert values == []
    assert stats.rle_run_count == 0
    assert stats.bit_packed_run_count == 0


def test_rle_decode_truncated_stream_raises():
    # Empty buffer but requesting values — must raise.
    with pytest.raises(ValueError, match="truncated"):
        decode_rle_bitpacked_hybrid(b"", bit_width=1, num_values=10)


def test_rle_decode_truncated_rle_value_raises():
    # Header says "RLE run of 5 values" but no value byte follows.
    stream = bytes([10])  # (5 << 1) | 0
    with pytest.raises(ValueError, match="truncated RLE run"):
        decode_rle_bitpacked_hybrid(stream, bit_width=8, num_values=5)


def test_rle_decode_truncated_bitpacked_run_raises():
    # Header: (1 << 1) | 1 = 3 → one group of 8 values at 4 bits = 4 bytes.
    # Provide only 2 bytes.
    stream = bytes([3, 0xAA, 0xBB])
    with pytest.raises(ValueError, match="truncated bit-packed"):
        decode_rle_bitpacked_hybrid(stream, bit_width=4, num_values=8)


def test_rle_decode_truncated_varint_raises():
    # A single byte with the continuation bit set, then EOF.
    with pytest.raises(ValueError, match="varint"):
        decode_rle_bitpacked_hybrid(bytes([0x80]), bit_width=1, num_values=1)


def test_rle_decode_oversized_varint_raises():
    # Eleven bytes all with continuation bit set → overflows 64 bits.
    with pytest.raises(ValueError, match="varint"):
        decode_rle_bitpacked_hybrid(bytes([0x80] * 11), bit_width=1, num_values=1)


@pytest.mark.parametrize("bad_bit_width", [-1, 65])
def test_rle_decode_invalid_bit_width_raises(bad_bit_width):
    with pytest.raises(ValueError, match="bit_width"):
        decode_rle_bitpacked_hybrid(b"", bit_width=bad_bit_width, num_values=0)


def test_rle_decode_negative_num_values_raises():
    with pytest.raises(ValueError, match="num_values"):
        decode_rle_bitpacked_hybrid(b"", bit_width=1, num_values=-1)


def test_rle_decode_bitpacked_overshoot_is_trimmed():
    """Bit-packed runs encode in groups of 8; if num_values isn't a multiple
    of 8, the trailing values from the last group must be discarded so the
    output length equals num_values.

    DecodeStats pin: the recorded ``bit_packed_run_lengths`` reports the
    encoder-declared run length (``num_groups * 8 = 8``), NOT the
    truncated-emitted count (5). This is the documented contract on
    DecodeStats — encoder-declared lengths preserve "what the encoder
    actually wrote" inspection semantics, which is more useful than
    post-truncation counts for writer-behaviour verification.
    """
    # bit_width=1, one group of 8 values: 0,1,0,1,0,1,0,1 → byte 0b10101010 = 0xAA.
    stream = bytes([3, 0xAA])  # (1 << 1) | 1 = 3
    values, stats = decode_rle_bitpacked_hybrid(stream, bit_width=1, num_values=5)
    assert values == [0, 1, 0, 1, 0]
    assert len(values) == 5
    # Encoder declared one group of 8 values; stats reflect that declared
    # length, not the 5 actually emitted to `values`.
    assert stats.bit_packed_run_lengths == (8,)
    assert stats.bit_packed_run_count == 1
    assert stats.rle_run_count == 0
    # Concretely, the sum-of-run-lengths invariant does NOT hold for this
    # overshoot case — that's the documented behaviour we're pinning.
    assert sum(stats.bit_packed_run_lengths) + sum(stats.rle_run_lengths) > 5


# ---------------------------------------------------------------------------
# decode_levels()
# ---------------------------------------------------------------------------


def test_decode_levels_max_level_zero_returns_zeros():
    assert decode_levels(b"", max_level=0, num_values=7) == [0] * 7


def test_decode_levels_max_level_zero_ignores_data():
    # data is not read at all when max_level == 0
    assert decode_levels(b"garbage", max_level=0, num_values=3) == [0, 0, 0]


def test_decode_levels_one_bit():
    # max_level=1 → bit_width=1. RLE-run header for 4 values of 1:
    # (4 << 1) | 0 = 8, value byte = 0x01.
    stream = bytes([8, 0x01])
    assert decode_levels(stream, max_level=1, num_values=4) == [1, 1, 1, 1]


def test_decode_levels_two_bits():
    # max_level=3 → bit_width=2. RLE-run header for 3 values of 2:
    # (3 << 1) | 0 = 6, value byte = 0x02.
    stream = bytes([6, 0x02])
    assert decode_levels(stream, max_level=3, num_values=3) == [2, 2, 2]


def test_decode_levels_negative_max_level_raises():
    with pytest.raises(ValueError, match="max_level"):
        decode_levels(b"", max_level=-1, num_values=0)


def test_decode_levels_negative_num_values_raises():
    with pytest.raises(ValueError, match="num_values"):
        decode_levels(b"", max_level=1, num_values=-1)


# ---------------------------------------------------------------------------
# decode_v1_level_block()
# ---------------------------------------------------------------------------


def test_decode_v1_level_block_max_level_zero():
    levels, new_offset = decode_v1_level_block(
        b"unused", offset=3, max_level=0, num_values=5
    )
    assert levels == [0, 0, 0, 0, 0]
    assert new_offset == 3  # no bytes consumed


def test_decode_v1_level_block_real_v1_page(tmp_path):
    """Round-trip a real V1 page's def-level block."""
    table = pa.table({"k": pa.array([1, 2, None, 4, None] * 200, type=pa.int32())})
    path = tmp_path / "v1-nullable.parquet"
    pq.write_table(table, path, compression="snappy", data_page_version="1.0")

    data_page = _first_data_page(path)
    raw = decompress(data_page["data"], "SNAPPY", data_page["uncompressed_size"])
    levels, new_offset = decode_v1_level_block(
        raw, 0, max_level=1, num_values=data_page["num_values"]
    )
    assert len(levels) == data_page["num_values"]
    # Expected pattern: present, present, null, present, null → 1,1,0,1,0
    expected = [1, 1, 0, 1, 0] * 200
    assert levels == expected
    # new_offset is positioned past the 4-byte prefix + the block bytes.
    (declared,) = struct.unpack_from("<I", raw, 0)
    assert new_offset == 4 + declared


def test_decode_levels_real_v2_page(tmp_path):
    """End-to-end V2 caller flow — pin the V2 contract documented in the
    module docstring and README ("V2 records definition_levels_byte_length
    in the page header; slice the raw page body and call decode_levels
    directly, NOT decode_v1_level_block").

    Companion to ``test_decode_v1_level_block_real_v1_page`` — same
    nullable-column round-trip, V2 page format instead. Detects regressions
    where the V2 path silently breaks (e.g., a future change that adds a
    length prefix to ``decode_levels``).
    """
    table = pa.table({"k": pa.array([1, 2, None, 4, None] * 200, type=pa.int32())})
    path = tmp_path / "v2-nullable.parquet"
    pq.write_table(table, path, compression="snappy", data_page_version="2.0")

    data_page = _first_data_page_v2(path)
    assert data_page["page_type"] == "DATA_PAGE_V2"
    def_len = data_page["definition_levels_byte_length"]
    rep_len = data_page["repetition_levels_byte_length"]
    assert def_len > 0, "expected a def-level block for a nullable column"
    assert rep_len == 0, "non-nested column has no rep-level block"
    # V2 page body layout: [rep_levels (uncompressed)][def_levels
    # (uncompressed)][values (may be compressed if is_compressed)]. Levels are
    # never compressed in V2, so we can read them straight from the raw page
    # bytes without calling decompress() on the whole body.
    raw = data_page["data"]
    def_levels_bytes = raw[rep_len : rep_len + def_len]
    levels = decode_levels(
        def_levels_bytes, max_level=1, num_values=data_page["num_values"]
    )
    assert len(levels) == data_page["num_values"]
    expected = [1, 1, 0, 1, 0] * 200
    assert levels == expected


def test_decode_v1_level_block_negative_offset_raises():
    with pytest.raises(ValueError, match="offset"):
        decode_v1_level_block(
            b"\x04\x00\x00\x00\x00", offset=-1, max_level=1, num_values=0
        )


def test_decode_v1_level_block_offset_past_end_raises():
    with pytest.raises(ValueError, match="offset"):
        decode_v1_level_block(b"abc", offset=10, max_level=1, num_values=0)


def test_decode_v1_level_block_truncated_prefix_raises():
    with pytest.raises(ValueError, match="length prefix"):
        decode_v1_level_block(b"\x01\x02", offset=0, max_level=1, num_values=1)


def test_decode_v1_level_block_block_longer_than_buffer_raises():
    # Prefix says 100 bytes follow but buffer has only 5.
    bad = struct.pack("<I", 100) + b"abcde"
    with pytest.raises(ValueError, match="claims"):
        decode_v1_level_block(bad, offset=0, max_level=1, num_values=1)


def test_decode_v1_level_block_negative_max_level_raises():
    """Argument-validation symmetry with sibling decoders — without this
    check, a negative ``max_level`` slips past the ``max_level == 0`` short-
    circuit and fails later inside ``decode_levels``, producing a less
    direct error message than the other public decoders give for the same
    input."""
    with pytest.raises(ValueError, match="max_level"):
        decode_v1_level_block(b"\x00\x00\x00\x00", offset=0, max_level=-1, num_values=0)


def test_decode_v1_level_block_negative_num_values_raises():
    """Argument-validation symmetry — pre-fix, the ``max_level == 0``
    short-circuit returned ``([0] * -1, offset) == ([], offset)`` silently
    (Python permits negative repetition counts), while every sibling
    decoder rejects ``num_values < 0`` up front."""
    with pytest.raises(ValueError, match="num_values"):
        decode_v1_level_block(b"unused", offset=0, max_level=0, num_values=-1)
    # Also verify the same rejection on the non-short-circuit path so the
    # contract holds for both branches.
    with pytest.raises(ValueError, match="num_values"):
        decode_v1_level_block(
            b"\x04\x00\x00\x00\x00", offset=0, max_level=1, num_values=-1
        )


# ---------------------------------------------------------------------------
# decode_plain()
# ---------------------------------------------------------------------------


def _decode_plain_data_page(path: Path) -> tuple[dict, bytes]:
    """Return ``(data_page_info, decompressed_payload_with_levels_stripped)``."""
    data_page = _first_data_page(path)
    raw = decompress(data_page["data"], "SNAPPY", data_page["uncompressed_size"])
    return data_page, raw


def _strip_v1_levels(raw: bytes, num_values: int, max_def_level: int) -> bytes:
    _, after_def = decode_v1_level_block(
        raw, 0, max_level=max_def_level, num_values=num_values
    )
    return raw[after_def:]


@pytest.mark.parametrize(
    "pa_type,parquet_type,values",
    [
        (pa.int32(), "INT32", [-1, 0, 1, 2**30, -(2**30)]),
        (pa.int64(), "INT64", [-1, 0, 1, 2**60, -(2**60)]),
        (pa.float32(), "FLOAT", [1.5, -2.25, 0.0, 1e10]),
        (pa.float64(), "DOUBLE", [1.5, -2.25, 0.0, 1e100]),
    ],
)
def test_decode_plain_fixed_width_via_pyarrow(tmp_path, pa_type, parquet_type, values):
    table = pa.table({"x": pa.array(values, type=pa_type)})
    path = tmp_path / f"plain-{parquet_type.lower()}.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        compression="snappy",
        data_page_version="1.0",
    )
    page, raw = _decode_plain_data_page(path)
    payload = _strip_v1_levels(raw, page["num_values"], max_def_level=1)
    decoded = decode_plain(payload, parquet_type, page["num_values"])
    if parquet_type in ("FLOAT", "DOUBLE"):
        assert decoded == pytest.approx(values, rel=1e-6)
    else:
        assert decoded == values


def test_decode_plain_byte_array_via_pyarrow(tmp_path):
    values = [b"alpha", b"beta", b"", b"gamma-and-delta", b"x"]
    table = pa.table({"x": pa.array(values, type=pa.binary())})
    path = tmp_path / "plain-bytes.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        compression="snappy",
        data_page_version="1.0",
    )
    page, raw = _decode_plain_data_page(path)
    payload = _strip_v1_levels(raw, page["num_values"], max_def_level=1)
    decoded = decode_plain(payload, "BYTE_ARRAY", page["num_values"])
    assert decoded == values


def test_decode_plain_fixed_len_byte_array_via_pyarrow(tmp_path):
    values = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]
    table = pa.table({"x": pa.array(values, type=pa.binary(4))})
    path = tmp_path / "plain-flba.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        compression="snappy",
        data_page_version="1.0",
    )
    page, raw = _decode_plain_data_page(path)
    payload = _strip_v1_levels(raw, page["num_values"], max_def_level=1)
    decoded = decode_plain(
        payload, "FIXED_LEN_BYTE_ARRAY", page["num_values"], type_length=4
    )
    assert decoded == values


def test_decode_plain_boolean_via_pyarrow(tmp_path):
    values = [True, False, True, True, False, False, True, False, True]
    table = pa.table({"x": pa.array(values, type=pa.bool_())})
    path = tmp_path / "plain-bool.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        compression="snappy",
        data_page_version="1.0",
    )
    page, raw = _decode_plain_data_page(path)
    payload = _strip_v1_levels(raw, page["num_values"], max_def_level=1)
    decoded = decode_plain(payload, "BOOLEAN", page["num_values"])
    assert decoded == values


def test_decode_plain_int96_via_pyarrow(tmp_path):
    """INT96 was the legacy Parquet timestamp encoding. pyarrow can write
    int96 timestamps by setting ``use_deprecated_int96_timestamps``."""
    times = pa.array(
        [
            1_700_000_000 * 10**9,
            1_700_000_001 * 10**9,
            1_700_000_002 * 10**9,
        ],
        type=pa.timestamp("ns"),
    )
    table = pa.table({"t": times})
    path = tmp_path / "plain-int96.parquet"
    pq.write_table(
        table,
        path,
        use_dictionary=False,
        compression="snappy",
        use_deprecated_int96_timestamps=True,
        data_page_version="1.0",
    )
    page, raw = _decode_plain_data_page(path)
    payload = _strip_v1_levels(raw, page["num_values"], max_def_level=1)
    decoded = decode_plain(payload, "INT96", page["num_values"])
    assert len(decoded) == page["num_values"]
    assert all(isinstance(v, bytes) and len(v) == 12 for v in decoded)


def test_decode_plain_num_values_zero_returns_empty():
    assert decode_plain(b"", "INT32", 0) == []
    assert decode_plain(b"", "BOOLEAN", 0) == []
    assert decode_plain(b"", "BYTE_ARRAY", 0) == []


def test_decode_plain_negative_num_values_raises():
    with pytest.raises(ValueError, match="num_values"):
        decode_plain(b"", "INT32", -1)


def test_decode_plain_unknown_type_raises():
    with pytest.raises(NotImplementedError, match="Unknown"):
        decode_plain(b"", "MADE_UP_TYPE", 1)


def test_decode_plain_fixed_len_byte_array_requires_type_length():
    with pytest.raises(ValueError, match="type_length"):
        decode_plain(b"", "FIXED_LEN_BYTE_ARRAY", 1)


def test_decode_plain_fixed_len_byte_array_negative_type_length_raises():
    with pytest.raises(ValueError, match="type_length"):
        decode_plain(b"abc", "FIXED_LEN_BYTE_ARRAY", 1, type_length=-1)


def test_decode_plain_fixed_width_truncated_raises():
    with pytest.raises(ValueError, match="truncated"):
        decode_plain(b"\x01\x00\x00", "INT32", 2)


def test_decode_plain_boolean_truncated_raises():
    # 9 booleans need 2 bytes, only 1 provided.
    with pytest.raises(ValueError, match="truncated"):
        decode_plain(b"\xff", "BOOLEAN", 9)


def test_decode_plain_int96_truncated_raises():
    with pytest.raises(ValueError, match="truncated"):
        decode_plain(b"\x00" * 11, "INT96", 1)


def test_decode_plain_byte_array_truncated_prefix_raises():
    with pytest.raises(ValueError, match="length prefix"):
        decode_plain(b"\x01\x02", "BYTE_ARRAY", 1)


def test_decode_plain_byte_array_truncated_value_raises():
    # length prefix says 10 bytes, only 3 available.
    payload = struct.pack("<i", 10) + b"abc"
    with pytest.raises(ValueError, match="truncated"):
        decode_plain(payload, "BYTE_ARRAY", 1)


def test_decode_plain_byte_array_negative_length_raises():
    payload = struct.pack("<i", -1)
    with pytest.raises(ValueError, match="negative length"):
        decode_plain(payload, "BYTE_ARRAY", 1)


def test_decode_plain_fixed_len_byte_array_truncated_raises():
    with pytest.raises(ValueError, match="truncated"):
        decode_plain(b"AAAA", "FIXED_LEN_BYTE_ARRAY", 2, type_length=4)


# ---------------------------------------------------------------------------
# DecodeStats
# ---------------------------------------------------------------------------


def test_decode_stats_is_immutable():
    """Frozen dataclass — assignment should raise."""
    stats = DecodeStats(
        rle_run_count=1,
        bit_packed_run_count=0,
        rle_run_lengths=(5,),
        rle_run_values=(0,),
        bit_packed_run_lengths=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.rle_run_count = 2  # type: ignore[misc]
