import struct

import pytest

from parquet_analyzer._core import (
    create_segment,
    create_segment_from_offset_info,
    fill_gaps,
    find_footer_segment,
    get_pages,
    get_summary,
    json_encode,
    parse_parquet_file,
    segment_to_json,
)


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
                        },
                        "column_index_length": 12,
                        "offset_index_length": 6,
                        "bloom_filter_length": 4,
                    }
                ]
            }
        ],
    }

    summary = get_summary(footer_json, segments)

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
    assert summary["bloom_fitler_size"] == 4
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
                "data_pages": [4],
            }
        ]
    }

    pages = get_pages(segments, column_chunk_data_offsets)

    assert pages[0]["column"] == ("col1",)
    assert pages[0]["row_groups"][0]["data_pages"][0]["$offset"] == 4
    assert pages[0]["row_groups"][0]["data_pages"][0]["type"] == "DATA_PAGE"


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
    assert find_footer_segment([create_segment(0, 1, "page")]) is None


def test_find_footer_segment_returns_match():
    footer = create_segment(0, 1, "footer")

    assert find_footer_segment([footer]) is footer


def test_json_encode_short_binary():
    payload = b"abc"

    encoded = json_encode(payload)

    assert encoded == {"type": "binary", "length": 3, "value": [97, 98, 99]}


def test_parse_parquet_file_invalid_header(tmp_path):
    target = tmp_path / "invalid-header.parquet"
    target.write_bytes(b"BAD!" + b"\x00" * 12)

    with pytest.raises(ValueError, match="missing PAR1 header"):
        parse_parquet_file(str(target))


def test_parse_parquet_file_invalid_footer(tmp_path):
    target = tmp_path / "invalid-footer.parquet"
    content = b"PAR1" + b"\x00" * 12 + struct.pack("<I", 0) + b"BAD!"
    target.write_bytes(content)

    with pytest.raises(ValueError, match="missing PAR1 footer"):
        parse_parquet_file(str(target))