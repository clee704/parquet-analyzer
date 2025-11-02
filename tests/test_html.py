import json
import struct
from decimal import Decimal

import pytest

from parquet_analyzer import _html


def build_schema_elements(definitions):
    return [_html.SchemaElement.from_json(item) for item in definitions]


def test_build_schema_tree_and_logical_type_mapping():
    schema_defs = [
        {"name": "root", "num_children": 2, "logicalType": {"STRING": {}}},
        {
            "name": "group",
            "num_children": 1,
            "logicalType": {"INTEGER": {"bitWidth": 16, "isSigned": False}},
        },
        {"name": "nested", "num_children": 0, "logicalType": {"STRING": {}}},
        {
            "name": "decimal_field",
            "num_children": 0,
            "logicalType": {"DECIMAL": {"precision": 9, "scale": 2}},
        },
    ]

    schema_tree = _html.build_schema_tree(build_schema_elements(schema_defs))
    mapping = _html.build_logical_type_mapping(schema_tree)

    assert len(schema_tree) == 1
    assert schema_tree[0].children[0].name == "group"
    assert schema_tree[0].children[0].children[0].name == "nested"
    assert mapping[("group", "nested")]["STRING"] == {}
    assert mapping[("decimal_field",)]["DECIMAL"]["scale"] == 2


def test_get_codecs_and_encodings():
    footer = {
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "codec": "SNAPPY",
                            "encodings": ["PLAIN", "RLE"],
                        }
                    }
                ]
            },
            {
                "columns": [
                    {
                        "meta_data": {
                            "codec": "GZIP",
                            "encodings": ["RLE", "DELTA"],
                        }
                    }
                ]
            },
        ]
    }

    codecs = _html.get_codecs(footer)
    encodings = _html.get_encodings(footer)

    assert codecs == ["SNAPPY", "GZIP"]
    assert encodings == ["DELTA", "PLAIN", "RLE"]


def test_aggregate_column_chunks_aggregates_stats():
    footer = {
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "path_in_schema": ["col1"],
                            "type": "INT32",
                            "num_values": 3,
                            "total_uncompressed_size": 100,
                            "total_compressed_size": 50,
                            "encodings": ["PLAIN", "RLE"],
                            "statistics": {
                                "null_count": 1,
                                "min_value": struct.pack("<i", 5),
                                "max_value": struct.pack("<i", 20),
                                "is_min_value_exact": True,
                                "is_max_value_exact": True,
                            },
                            "encoding_stats": [
                                {
                                    "page_type": "DATA_PAGE",
                                    "encoding": "PLAIN",
                                    "count": 1,
                                }
                            ],
                            "codec": "SNAPPY",
                        }
                    }
                ]
            },
            {
                "columns": [
                    {
                        "meta_data": {
                            "path_in_schema": ["col1"],
                            "type": "INT32",
                            "num_values": 2,
                            "total_uncompressed_size": 70,
                            "total_compressed_size": 30,
                            "encodings": ["RLE"],
                            "statistics": {
                                "null_count": 2,
                                "min_value": struct.pack("<i", 3),
                                "max_value": struct.pack("<i", 25),
                                "is_min_value_exact": False,
                                "is_max_value_exact": True,
                            },
                            "encoding_stats": [
                                {
                                    "page_type": "DATA_PAGE",
                                    "encoding": "PLAIN",
                                    "count": 2,
                                },
                                {
                                    "page_type": "DICTIONARY_PAGE",
                                    "encoding": "RLE_DICTIONARY",
                                    "count": 1,
                                },
                            ],
                            "codec": "GZIP",
                        }
                    }
                ]
            },
        ]
    }

    logical_type_mapping = {("col1",): None}

    columns = _html.aggregate_column_chunks(footer, logical_type_mapping)

    assert len(columns) == 1
    column = columns[0]
    assert column["num_values"] == 5
    assert column["total_uncompressed_size"] == 170
    assert column["total_compressed_size"] == 80
    assert column["encodings"] == {"PLAIN", "RLE"}
    assert column["codecs"] == {"SNAPPY", "GZIP"}
    stats = column["statistics"]
    assert stats["null_count"] == 3
    assert stats["is_min_value_exact"] is False
    assert stats["is_max_value_exact"] is True
    assert struct.unpack("<i", stats["min_value"])[0] == 3
    assert struct.unpack("<i", stats["max_value"])[0] == 25
    assert column["encoding_stats"][("DATA_PAGE", "PLAIN")]["count"] == 3
    assert column["encoding_stats"][("DICTIONARY_PAGE", "RLE_DICTIONARY")]["count"] == 1


def test_group_segments_by_page():
    segments = [
        {"name": "page", "offset": 0, "length": 2, "value": []},
        {"name": "page_data", "offset": 2, "length": 3, "value": []},
        {"name": "other", "offset": 5, "length": 1, "value": []},
    ]

    grouped = _html.group_segments_by_page(segments)

    assert grouped[0]["name"] == _html.page_header_and_data_name
    assert grouped[0]["length"] == 5
    assert grouped[1]["name"] == "other"


def test_get_num_values_supports_headers():
    page_v1 = {
        "value": [
            {
                "name": "data_page_header",
                "value": [{"name": "num_values", "value": 7}],
            }
        ]
    }
    page_v2 = {
        "value": [
            {
                "name": "data_page_header_v2",
                "value": [{"name": "num_values", "value": 9}],
            }
        ]
    }

    assert _html.get_num_values(page_v1) == 7
    assert _html.get_num_values(page_v2) == 9
    assert _html.get_num_values({"value": []}) is None


def test_build_page_offset_to_column_chunk_mapping():
    page_segments = [
        {
            "name": "page",
            "offset": 300,
            "length": 10,
            "value": [
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                }
            ],
        },
        {
            "name": "page",
            "offset": 310,
            "length": 8,
            "value": [
                {
                    "name": "data_page_header_v2",
                    "value": [{"name": "num_values", "value": 3}],
                }
            ],
        },
    ]
    page_mapping = _html.get_page_mapping(page_segments)
    footer = {
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "path_in_schema": ["col1"],
                            "dictionary_page_offset": 200,
                            "data_page_offset": 300,
                            "num_values": 5,
                        }
                    }
                ]
            }
        ]
    }

    mapping = _html.build_page_offset_to_column_chunk_mapping(footer, page_mapping)

    assert mapping[200] == (0, 0)
    assert mapping[300] == (0, 0)
    assert mapping[310] == (0, 0)


def test_group_segments_combines_related_segments():
    segments = [
        {
            "name": "page",
            "offset": 300,
            "length": 10,
            "value": [
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                }
            ],
        },
        {"name": "page_data", "offset": 305, "length": 5, "value": []},
        {
            "name": "page",
            "offset": 310,
            "length": 10,
            "value": [
                {
                    "name": "data_page_header_v2",
                    "value": [{"name": "num_values", "value": 3}],
                }
            ],
        },
        {"name": "page_data", "offset": 320, "length": 5, "value": []},
        {"name": "column_index", "offset": 400, "length": 2, "value": []},
        {"name": "column_index", "offset": 402, "length": 2, "value": []},
        {"name": "offset_index", "offset": 500, "length": 1, "value": []},
        {"name": "offset_index", "offset": 501, "length": 1, "value": []},
        {"name": "bloom_filter", "offset": 600, "length": 1, "value": []},
        {"name": "bloom_filter", "offset": 601, "length": 1, "value": []},
        {"name": "other", "offset": 700, "length": 1, "value": []},
    ]
    footer = {
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "path_in_schema": ["col1"],
                            "data_page_offset": 300,
                            "num_values": 5,
                        }
                    }
                ]
            }
        ]
    }

    grouped = _html.group_segments(segments, footer)

    grouped_names = [segment["name"] for segment in grouped]
    assert _html.page_group_name in grouped_names
    assert _html.column_index_group_name in grouped_names
    assert _html.offset_index_group_name in grouped_names
    assert _html.bloom_filter_group_name in grouped_names

    pages_segment = next(
        item for item in grouped if item["name"] == _html.page_group_name
    )
    assert pages_segment["value"][0]["num_pages"] == 2
    assert pages_segment["value"][0]["row_group_index"] == 0
    assert pages_segment["value"][0]["column_index"] == 0


def test_format_helpers():
    assert _html.format_bytes(512) == "512 bytes"
    assert _html.format_bytes(2048) == "2.00 KB"

    int_type = {"INTEGER": {"bitWidth": 8, "isSigned": True}}
    assert _html.format_logical_type(int_type) == "SIGNED 8-BIT INTEGER"

    time_type = {"TIME": {"isAdjustedToUTC": True, "unit": {"MICROS": {}}}}
    assert _html.format_logical_type(time_type) == "TIME(MICROS) (adjusted to UTC)"

    decimal_type = {"DECIMAL": {"precision": 10, "scale": 2}}
    assert _html.format_logical_type(decimal_type) == "DECIMAL(10,2)"


def test_decode_encode_and_format_stats_value():
    decimal_type = {"DECIMAL": {"scale": 2}}
    encoded_decimal = _html.encode_stats_value(
        Decimal("12.34"), "INT32", 0, decimal_type
    )
    decoded_decimal = _html.decode_stats_value(encoded_decimal, "INT32", decimal_type)
    assert decoded_decimal == Decimal("12.34")

    encoded_float = _html.encode_stats_value(3.5, "FLOAT", 0, None)
    assert _html.decode_stats_value(encoded_float, "FLOAT", None) == pytest.approx(3.5)

    encoded_bool = _html.encode_stats_value(True, "BOOLEAN", 0, None)
    assert _html.decode_stats_value(encoded_bool, "BOOLEAN", None) is True

    raw_bytes = b"abc"
    assert _html.format_stats_value(raw_bytes, "BINARY", None) == "0x616263"

    str_bytes = _html.encode_stats_value("payload", "BINARY", 0, None)
    assert _html.format_stats_value(str_bytes, "BINARY", {"STRING": {}}) == "payload"


def test_to_nice_json_and_is_nested_segment():
    payload = {"a": 1}
    pretty = _html.to_nice_json(payload)
    assert json.loads(pretty) == payload

    segment_plain = {"name": "value", "value": 1}
    segment_group = {"name": ":group", "value": []}
    segment_type_class = {"name": "field", "metadata": {"type_class": object}}
    segment_list = {
        "name": "list",
        "metadata": {"type": "list"},
        "value": [segment_group],
    }

    assert _html.is_nested_segment(segment_plain) is False
    assert _html.is_nested_segment(segment_group) is True
    assert _html.is_nested_segment(segment_type_class) is True
    assert _html.is_nested_segment(segment_list) is True
