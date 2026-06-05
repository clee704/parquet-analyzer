import struct
from types import SimpleNamespace

import pytest

from thrift.protocol.TProtocol import TType
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport

import parquet_analyzer._core as core
from parquet_analyzer._core import (
    OffsetRecordingCompactProtocol,
    TFileTransport,
    _compute_pages,
    _compute_summary,
    _find_footer_segment,
    create_segment,
    create_segment_from_offset_info,
    fill_gaps,
    json_encode,
    segment_to_json,
    read_bloom_filter,
    read_column_index,
    read_offset_index,
    read_pages,
)
from parquet_analyzer import ParquetFile
from parquet.ttypes import ColumnMetaData, CompressionCodec, Encoding


STRUCT_CLASS = type("Struct", (), {"thrift_spec": ()})


def test_create_segment_from_offset_info_struct():
    offset_info = {
        "name": "root",
        "type": "struct",
        "type_class": None,
        "spec": None,
        "range_from": 0,
        "range_to": 4,
        "value": [
            {
                "name": "field",
                "type": "i32",
                "type_class": None,
                "spec": None,
                "range_from": 0,
                "range_to": 4,
                "value": 7,
            }
        ],
    }

    segment = create_segment_from_offset_info(offset_info, base_offset=10)

    assert segment["name"] == "root"
    assert segment["offset"] == 10
    assert segment["length"] == 4
    child = segment["value"][0]
    assert child["name"] == "field"
    assert child["metadata"]["type"] == "i32"
    assert child["value"] == 7


def test_fill_gaps_inserts_unknown_segments():
    segments = [
        create_segment(0, 4, "magic", "PAR1"),
        create_segment(10, 12, "footer", None),
    ]

    result = fill_gaps(segments, file_size=15)

    assert result[1]["name"] == "unknown"
    assert result[1]["offset"] == 4
    assert result[1]["length"] == 6
    assert result[-1]["name"] == "unknown"
    assert result[-1]["offset"] == 12
    assert result[-1]["length"] == 3


def test_json_encode_truncates_long_binary():
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"

    encoded = json_encode(payload)

    assert encoded["type"] == "binary"
    assert encoded["length"] == len(payload)
    assert "value_truncated" in encoded
    assert len(encoded["value_truncated"]) == 32


def test_json_encode_rejects_non_bytes():
    with pytest.raises(ValueError):
        json_encode("not-bytes")


def test_get_summary_counts_pages_and_sizes():
    page_segment = create_segment(
        4,
        14,
        "page",
        value=[
            create_segment(
                4,
                5,
                "type",
                value=0,
                metadata={"enum_type": "PageType", "enum_name": "DATA_PAGE"},
            ),
            create_segment(5, 9, "uncompressed_page_size", value=256),
            create_segment(9, 13, "compressed_page_size", value=128),
            create_segment(
                13,
                14,
                "data_page_header",
                value=[],
                metadata={"type": "struct"},
            ),
        ],
        metadata={"type": "struct"},
    )

    segments = [
        create_segment(0, 4, "magic_number", "PAR1"),
        page_segment,
        create_segment(20, 35, "footer", value=None),
        create_segment(35, 39, "magic_number", "PAR1"),
    ]

    footer_json = {
        "num_rows": 10,
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "total_uncompressed_size": 256,
                            "total_compressed_size": 128,
                            "bloom_filter_length": 4,
                        },
                        "column_index_length": 12,
                        "offset_index_length": 6,
                    }
                ]
            }
        ],
    }

    summary = _compute_summary(footer_json, segments)

    assert summary["num_rows"] == 10
    assert summary["num_row_groups"] == 1
    assert summary["num_columns"] == 1
    assert summary["num_pages"] == 1
    assert summary["num_data_pages"] == 1
    assert summary["num_v1_data_pages"] == 1
    assert summary["num_v2_data_pages"] == 0
    assert summary["num_dict_pages"] == 0
    assert summary["page_header_size"] == page_segment["length"]
    assert summary["uncompressed_page_data_size"] == 256
    assert summary["compressed_page_data_size"] == 128
    assert summary["uncompressed_page_size"] == 256
    assert summary["compressed_page_size"] == 128
    assert summary["column_index_size"] == 12
    assert summary["offset_index_size"] == 6
    assert summary["bloom_filter_size"] == 4
    assert summary["footer_size"] == 15
    assert summary["file_size"] == 39


def test_get_pages_includes_offsets_with_page_details():
    page_segment = create_segment(
        4,
        14,
        "page",
        value=[
            create_segment(
                4,
                5,
                "type",
                value=0,
                metadata={"enum_type": "PageType", "enum_name": "DATA_PAGE"},
            ),
            create_segment(5, 9, "uncompressed_page_size", value=256),
            create_segment(9, 13, "compressed_page_size", value=128),
        ],
        metadata={"type": "struct"},
    )

    segments = [page_segment]

    column_chunk_data_offsets = {
        ("col1",): [
            {
                "pages": [4],
            }
        ]
    }

    pages = _compute_pages(segments, column_chunk_data_offsets)

    assert pages[0]["column"] == ("col1",)
    assert pages[0]["row_groups"][0]["pages"][0]["$offset"] == 4
    assert pages[0]["row_groups"][0]["pages"][0]["type"] == "DATA_PAGE"


def test_offset_recording_numeric_reads(monkeypatch):
    proto = OffsetRecordingCompactProtocol(
        TTransport.TMemoryBuffer(), "dummy", STRUCT_CLASS
    )
    proto._current = {"name": "numbers", "type": "list", "spec": None, "value": []}
    proto._parents = []

    for method, value in [
        ("readByte", 1),
        ("readI16", 2),
        ("readI32", 3),
        ("readI64", 4),
        ("readDouble", 5.5),
        ("readBool", True),
        ("readString", "abc"),
    ]:
        monkeypatch.setattr(
            TCompactProtocol.TCompactProtocol,
            method,
            lambda self, _value=value: _value,
        )
        result = getattr(proto, method)()
        assert result == value
        assert proto._current["value"][-1] == value


def test_offset_recording_read_binary_branches(monkeypatch):
    proto = OffsetRecordingCompactProtocol(
        TTransport.TMemoryBuffer(), "dummy", STRUCT_CLASS
    )
    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol,
        "readBinary",
        lambda self: b"payload",
    )

    proto._current = {"name": "bin", "type": "string", "spec": "BINARY", "value": None}
    assert proto.readBinary() == b"payload"
    assert proto._current["value"] == b"payload"

    proto._current = {
        "name": "list",
        "type": "list",
        "spec": (TType.STRING, "BINARY"),
        "value": [],
    }
    proto.readBinary()
    assert proto._current["value"] == [b"payload"]

    proto._current = {"name": "skip", "type": "string", "spec": None, "value": None}
    proto.readBinary()
    assert proto._current["value"] is None


def test_offset_recording_enum_annotation():
    proto = OffsetRecordingCompactProtocol(
        TTransport.TMemoryBuffer(), "dummy", STRUCT_CLASS
    )

    parent = {
        "name": "parent",
        "type": "struct",
        "type_class": ColumnMetaData,
        "value": [],
    }
    proto._parents = [parent]
    proto._current = {
        "name": "codec",
        "type": "i32",
        "type_class": None,
        "spec": None,
        "value": None,
    }
    proto._append_value(CompressionCodec.GZIP)
    assert proto._current["enum_type"] == "CompressionCodec"
    assert proto._current["enum_name"] == "GZIP"

    proto._parents = [parent]
    proto._current = {
        "name": "encodings",
        "type": "list",
        "type_class": None,
        "spec": None,
        "value": [],
    }
    proto._append_value(Encoding.PLAIN)
    proto._append_value(Encoding.RLE)
    assert proto._current["enum_type"] == "Encoding"
    assert proto._current["enum_name"] == ["PLAIN", "RLE"]


def test_offset_recording_collection_methods(monkeypatch):
    proto = OffsetRecordingCompactProtocol(
        TTransport.TMemoryBuffer(), "dummy", STRUCT_CLASS
    )

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readListBegin", lambda self: (TType.I32, 0)
    )
    assert proto.readListBegin() == (TType.I32, 0)

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readListEnd", lambda self: None
    )
    assert proto.readListEnd() is None

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol,
        "readMapBegin",
        lambda self: (TType.I32, TType.I32, 0),
    )
    assert proto.readMapBegin() == (TType.I32, TType.I32, 0)

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readMapEnd", lambda self: None
    )
    assert proto.readMapEnd() is None

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readSetBegin", lambda self: (TType.I32, 0)
    )
    assert proto.readSetBegin() == (TType.I32, 0)

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readSetEnd", lambda self: None
    )
    assert proto.readSetEnd() is None

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol,
        "readMessageBegin",
        lambda self: ("name", 1, 0),
    )
    assert proto.readMessageBegin() == ("name", 1, 0)

    monkeypatch.setattr(
        TCompactProtocol.TCompactProtocol, "readMessageEnd", lambda self: None
    )
    assert proto.readMessageEnd() is None


def test_offset_recording_get_pos():
    buf = TTransport.TMemoryBuffer()
    buf._buffer.write(b"abcdef")
    buf._buffer.seek(3)
    proto = OffsetRecordingCompactProtocol(buf, "dummy", STRUCT_CLASS)
    assert proto._get_pos() == 3

    class DummyTransport:
        pass

    bad_proto = OffsetRecordingCompactProtocol(DummyTransport(), "dummy", STRUCT_CLASS)
    with pytest.raises(RuntimeError):
        bad_proto._get_pos()


def test_offset_recording_child_management():
    proto = OffsetRecordingCompactProtocol(
        TTransport.TMemoryBuffer(), "dummy", STRUCT_CLASS
    )
    parent = {"name": "parent", "value": []}
    proto._parents = []
    proto._current = parent

    child = {"name": "child", "value": []}
    proto._new_child(child)
    assert proto._current is child
    assert proto._parents[-1] is parent

    proto._finish_child()
    assert proto._current is parent
    assert parent["value"][0] is child


def test_tfiletransport_operations(tmp_path):
    file_path = tmp_path / "data.bin"
    file_path.write_bytes(b"abcdef")

    with file_path.open("rb") as fh:
        transport = TFileTransport(fh)
        assert transport.isOpen()
        assert transport.read(2) == b"ab"
        assert transport.tell() == 2

        transport.seek(1, whence=1)
        assert transport.tell() == 3

        transport.seek(0)
        assert transport.tell() == 0

        transport.flush()
        transport.close()

        with pytest.raises(NotImplementedError):
            transport.write(b"x")
        with pytest.raises(NotImplementedError):
            transport.seek(0, whence=2)


def test_read_helpers_and_summary(monkeypatch):
    page_entries = [
        {
            "header": SimpleNamespace(
                data_page_header=SimpleNamespace(num_values=2),
                data_page_header_v2=None,
                compressed_page_size=2,
            ),
            "length": 5,
        },
        {
            "header": SimpleNamespace(
                data_page_header=None,
                data_page_header_v2=SimpleNamespace(num_values=1),
                compressed_page_size=3,
            ),
            "length": 4,
        },
    ]

    dictionary_header = SimpleNamespace(
        data_page_header=None,
        data_page_header_v2=None,
        dictionary_page_header=SimpleNamespace(num_values=1),
        compressed_page_size=13,
    )

    dictionary_segment = {
        "offset": 4,
        "length": 3,
        "name": "page",
        "compressed_page_size": 13,
    }
    column_index_segment = {"offset": 30, "length": 2, "name": "column_index"}
    offset_index_segment = {"offset": 40, "length": 3, "name": "offset_index"}
    bloom_segment = {"offset": 50, "length": 2, "name": "bloom_filter"}

    def fake_read_thrift(file_obj, offset, name, thrift_class):
        if name == "page" and offset == 4:
            return dictionary_header, dictionary_segment
        if name == "page":
            entry = page_entries.pop(0)
            segment = {"offset": offset, "length": entry["length"], "name": "page"}
            return entry["header"], segment
        if name == "column_index":
            return SimpleNamespace(), column_index_segment
        if name == "offset_index":
            return SimpleNamespace(), offset_index_segment
        if name == "bloom_filter":
            return SimpleNamespace(), bloom_segment
        raise AssertionError(f"Unexpected thrift read: {name} @ {offset}")

    monkeypatch.setattr(core, "read_thrift_segment", fake_read_thrift)

    column_chunk = SimpleNamespace(
        meta_data=SimpleNamespace(
            num_values=3,
            dictionary_page_offset=4,
            data_page_offset=20,
            bloom_filter_offset=50,
        ),
        column_index_offset=30,
        offset_index_offset=40,
    )

    segments: list[dict] = []
    offsets = read_pages(object(), column_chunk, segments)
    assert offsets == [4, 20, 27]
    dict_offset = 4

    col_index_offset = read_column_index(object(), column_chunk, segments)
    assert col_index_offset == 30

    off_index_offset = read_offset_index(object(), column_chunk, segments)
    assert off_index_offset == 40

    bloom_offset = read_bloom_filter(object(), column_chunk, segments)
    assert bloom_offset == 50

    json_lookup = {
        4: {
            "type": "DICTIONARY_PAGE",
            "uncompressed_page_size": 1,
            "compressed_page_size": 1,
        },
        20: {
            "type": "DATA_PAGE",
            "uncompressed_page_size": 4,
            "compressed_page_size": 2,
            "data_page_header": {},
        },
        27: {
            "type": "DATA_PAGE_V2",
            "uncompressed_page_size": 5,
            "compressed_page_size": 3,
            "data_page_header_v2": {},
        },
        30: {"column_index": True},
        40: {"offset_index": True},
        50: {"bloom_filter": True},
    }

    monkeypatch.setattr(
        core, "segment_to_json", lambda segment: json_lookup[segment["offset"]]
    )

    column_offsets = {
        ("col",): [
            {
                "data_pages": offsets,
                "dictionary_page": dict_offset,
                "column_index": col_index_offset,
                "offset_index": off_index_offset,
                "bloom_filter": bloom_offset,
            }
        ]
    }

    pages = _compute_pages(segments, column_offsets)
    row_group = pages[0]["row_groups"][0]
    assert row_group["dictionary_page"]["$offset"] == dict_offset
    assert row_group["column_index"]["$offset"] == col_index_offset
    assert row_group["offset_index"]["$offset"] == off_index_offset
    assert row_group["bloom_filter"]["$offset"] == bloom_offset

    footer_json = {
        "num_rows": 3,
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "total_uncompressed_size": 9,
                            "total_compressed_size": 6,
                            "bloom_filter_length": 1,
                        },
                        "column_index_length": 2,
                        "offset_index_length": 3,
                    }
                ]
            }
        ],
    }

    summary = _compute_summary(footer_json, segments)
    assert summary["num_pages"] == 3
    assert summary["num_data_pages"] == 2
    assert summary["num_v1_data_pages"] == 1
    assert summary["num_v2_data_pages"] == 1
    assert summary["num_dict_pages"] == 1
    assert summary["page_header_size"] == 12
    assert summary["uncompressed_page_data_size"] == 10
    assert summary["compressed_page_data_size"] == 6
    assert summary["bloom_filter_size"] == 1


def test_create_segment_preserves_metadata():
    segment = create_segment(0, 10, "field", value=123, metadata={"type": "i32"})

    assert segment == {
        "offset": 0,
        "length": 10,
        "name": "field",
        "value": 123,
        "metadata": {"type": "i32"},
    }


def test_create_segment_from_offset_info_returns_non_dict():
    sentinel = [1, 2, 3]

    assert create_segment_from_offset_info(sentinel, base_offset=5) is sentinel


def test_create_segment_from_offset_info_handles_list_children():
    offset_info = {
        "name": "values",
        "type": "list",
        "type_class": None,
        "spec": None,
        "range_from": 2,
        "range_to": 6,
        "value": [
            {
                "name": "element",
                "type": "i32",
                "type_class": None,
                "spec": None,
                "range_from": 2,
                "range_to": 4,
                "value": 11,
            },
            {
                "name": "element",
                "type": "i32",
                "type_class": None,
                "spec": None,
                "range_from": 4,
                "range_to": 6,
                "value": 22,
            },
        ],
    }

    segment = create_segment_from_offset_info(offset_info, base_offset=8)

    assert segment["offset"] == 10
    assert segment["length"] == 4
    assert segment["metadata"]["type"] == "list"
    assert [child["value"] for child in segment["value"]] == [11, 22]


def test_fill_gaps_no_missing_regions():
    segments = [
        create_segment(0, 3, "a"),
        create_segment(3, 6, "b"),
    ]

    assert fill_gaps(segments, file_size=6) == segments


def test_segment_to_json_struct_and_enum():
    segment = {
        "name": "wrapper",
        "offset": 0,
        "length": 4,
        "value": [
            {
                "name": "field",
                "offset": 0,
                "length": 4,
                "value": 1,
                "metadata": {"enum_type": "Example", "enum_name": "ONE"},
            }
        ],
        "metadata": {"type": "struct"},
    }

    assert segment_to_json(segment) == {"field": "ONE"}


def test_segment_to_json_list_without_enum():
    segment = {
        "name": "values",
        "offset": 0,
        "length": 4,
        "value": [
            {
                "name": "element",
                "offset": 0,
                "length": 2,
                "value": 7,
            },
            {
                "name": "element",
                "offset": 2,
                "length": 2,
                "value": 8,
            },
        ],
        "metadata": {"type": "list"},
    }

    assert segment_to_json(segment) == [7, 8]


def test_segment_to_json_enum_scalar():
    segment = {
        "name": "type",
        "offset": 0,
        "length": 1,
        "value": 0,
        "metadata": {"enum_type": "PageType", "enum_name": "DATA_PAGE"},
    }

    assert segment_to_json(segment) == "DATA_PAGE"


def test_find_footer_segment_returns_none():
    assert _find_footer_segment([create_segment(0, 1, "page")]) is None


def test_find_footer_segment_returns_match():
    footer = create_segment(0, 1, "footer")

    assert _find_footer_segment([footer]) is footer


def test_json_encode_short_binary():
    payload = b"abc"

    encoded = json_encode(payload)

    assert encoded == {"type": "binary", "length": 3, "value": [97, 98, 99]}


def test_parquet_file_invalid_header(tmp_path):
    target = tmp_path / "invalid-header.parquet"
    target.write_bytes(b"BAD!" + b"\x00" * 12)

    with pytest.raises(ValueError, match="missing PAR1 header"):
        ParquetFile(str(target))


def test_parquet_file_invalid_footer(tmp_path):
    target = tmp_path / "invalid-footer.parquet"
    content = b"PAR1" + b"\x00" * 12 + struct.pack("<I", 0) + b"BAD!"
    target.write_bytes(content)

    with pytest.raises(ValueError, match="missing PAR1 footer"):
        ParquetFile(str(target))


# ---------------------------------------------------------------------------
# Statistics decoding (shared kernel used by _html + the tree serializer)
# ---------------------------------------------------------------------------


from decimal import Decimal  # noqa: E402


def test_decode_stat_value_numeric_types():
    assert (
        core.decode_stat_value((1).to_bytes(4, "little", signed=True), "INT32", None)
        == 1
    )
    assert (
        core.decode_stat_value((-5).to_bytes(8, "little", signed=True), "INT64", None)
        == -5
    )
    assert core.decode_stat_value(struct.pack("<f", 1.5), "FLOAT", None) == 1.5
    assert core.decode_stat_value(struct.pack("<d", 2.5), "DOUBLE", None) == 2.5
    assert core.decode_stat_value(b"\x01", "BOOLEAN", None) is True
    assert core.decode_stat_value(b"\x00", "BOOLEAN", None) is False


def test_decode_stat_value_decimal():
    raw_int = (123).to_bytes(4, "little", signed=True)
    assert core.decode_stat_value(
        raw_int, "INT32", {"DECIMAL": {"scale": 2}}
    ) == Decimal("1.23")
    raw_flba = (456).to_bytes(2, "big", signed=True)
    assert core.decode_stat_value(
        raw_flba, "FIXED_LEN_BYTE_ARRAY", {"DECIMAL": {"scale": 2}}
    ) == Decimal("4.56")


def test_decode_stat_value_legacy_converted_decimal():
    """Legacy writers mark DECIMAL via converted_type + a separate scale,
    not logicalType. The kernel honors that form too."""
    raw = (123).to_bytes(4, "little", signed=True)
    assert core.decode_stat_value(
        raw, "INT32", None, converted_type="DECIMAL", scale=2
    ) == Decimal("1.23")


def test_decode_stat_value_byte_array_passthrough():
    assert core.decode_stat_value(b"hello", "BYTE_ARRAY", None) == b"hello"


def test_json_safe_stat_value():
    assert (
        core.json_safe_stat_value((7).to_bytes(4, "little", signed=True), "INT32", None)
        == 7
    )
    # BYTE_ARRAY of unknown type: UTF-8 when decodable, else hex.
    assert core.json_safe_stat_value(b"female", "BYTE_ARRAY", None) == "female"
    assert core.json_safe_stat_value(b"\x00\xff", "BYTE_ARRAY", None) == "00ff"
    # DECIMAL -> lossless string (logical and legacy converted forms).
    raw = (123).to_bytes(4, "little", signed=True)
    assert core.json_safe_stat_value(raw, "INT32", {"DECIMAL": {"scale": 2}}) == "1.23"
    assert (
        core.json_safe_stat_value(raw, "INT32", None, converted_type="DECIMAL", scale=2)
        == "1.23"
    )


def test_json_safe_stat_value_string_type_truncated():
    """A column known to be STRING decodes as (lossy) UTF-8 text even when
    the stat bytes are truncated mid-codepoint — not shown as hex."""
    truncated = "\u263a".encode("utf-8")[:2]  # ☺ is 3 bytes; keep 2
    out = core.json_safe_stat_value(truncated, "BYTE_ARRAY", {"STRING": {}})
    assert isinstance(out, str)
    # Same via legacy converted_type=UTF8.
    out2 = core.json_safe_stat_value(
        truncated, "BYTE_ARRAY", None, converted_type="UTF8"
    )
    assert isinstance(out2, str)


def test_column_stat_types_flat():
    schema = [
        {"name": "schema", "num_children": 2, "type": None},
        {
            "name": "s",
            "type": "BYTE_ARRAY",
            "converted_type": "UTF8",
            "logicalType": {"STRING": {}},
        },
        {"name": "n", "type": "INT32"},
    ]
    assert core.column_stat_types(schema) == {
        ("s",): {"logical": {"STRING": {}}, "converted": "UTF8", "scale": 0},
        ("n",): {"logical": None, "converted": None, "scale": 0},
    }


def test_column_stat_types_nested_and_decimal_scale():
    schema = [
        {"name": "schema", "num_children": 1, "type": None},
        {"name": "g", "num_children": 1, "type": None},
        {
            "name": "x",
            "type": "FIXED_LEN_BYTE_ARRAY",
            "converted_type": "DECIMAL",
            "scale": 2,
            "precision": 5,
        },
    ]
    assert core.column_stat_types(schema) == {
        ("g", "x"): {"logical": None, "converted": "DECIMAL", "scale": 2},
    }


def test_column_stat_types_empty():
    assert core.column_stat_types([]) == {}


def test_html_decode_stats_value_delegates_to_core():
    from parquet_analyzer import _html

    raw = (42).to_bytes(4, "little", signed=True)
    assert _html.decode_stats_value(raw, "INT32", None) == 42
