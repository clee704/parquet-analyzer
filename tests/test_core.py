import pytest

from parquet_analyzer._core import (
    create_segment,
    create_segment_from_offset_info,
    fill_gaps,
    get_pages,
    get_summary,
    json_encode,
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