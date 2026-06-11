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
                {"name": "compressed_page_size", "value": 5},
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                },
            ],
        },
        {
            "name": "page",
            "offset": 315,
            "length": 8,
            "value": [
                {"name": "compressed_page_size", "value": 0},
                {
                    "name": "data_page_header_v2",
                    "value": [{"name": "num_values", "value": 3}],
                },
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
    assert mapping[315] == (0, 0)


def test_group_segments_combines_related_segments():
    segments = [
        {
            "name": "page",
            "offset": 300,
            "length": 10,
            "value": [
                {"name": "compressed_page_size", "value": 5},
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                },
            ],
        },
        {"name": "page_data", "offset": 310, "length": 5, "value": []},
        {
            "name": "page",
            "offset": 315,
            "length": 10,
            "value": [
                {"name": "compressed_page_size", "value": 5},
                {
                    "name": "data_page_header_v2",
                    "value": [{"name": "num_values", "value": 3}],
                },
            ],
        },
        {"name": "page_data", "offset": 325, "length": 5, "value": []},
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


# ---------------------------------------------------------------------------
# End-to-end report generation (the public --output-mode html entry point)
# ---------------------------------------------------------------------------


from pathlib import Path  # noqa: E402

from parquet_analyzer import ParquetFile  # noqa: E402

_TITANIC = Path(__file__).parent / "data" / "titanic.parquet"


@pytest.mark.parametrize(
    "sections",
    [[], ["segments"], ["schema", "columns", "segments"]],
)
def test_generate_html_report_end_to_end(sections):
    """Render a full report from a real file and assert on the HTML, not
    just that it doesn't raise. Exercises both the segments and the
    no-segments branches of generate_html_report."""
    pf = ParquetFile(str(_TITANIC))
    try:
        html = _html.generate_html_report(
            str(_TITANIC),
            summary=pf.full_summary,
            footer=pf.footer,
            segments=pf.all_segments(),
            sections=sections,
        )
    finally:
        pf.close()
    assert isinstance(html, str) and html
    assert "<html" in html.lower()
    assert "parquet analyzer" in html.lower()
    if "segments" in sections:
        assert "segment" in html.lower()


# ---------------------------------------------------------------------------
# Helper edge-case coverage (the branches the happy-path tests above don't hit)
# ---------------------------------------------------------------------------


def test_format_logical_type_all_variants():
    assert _html.format_logical_type({"STRING": {}}) == "STRING"
    assert _html.format_logical_type({"DATE": {}}) == "DATE"
    # TIME / TIMESTAMP across every unit, with and without UTC adjustment.
    assert (
        _html.format_logical_type({"TIME": {"unit": {"MILLIS": {}}}}) == "TIME(MILLIS)"
    )
    assert _html.format_logical_type({"TIME": {"unit": {"NANOS": {}}}}) == "TIME(NANOS)"
    assert _html.format_logical_type({"TIME": {"unit": {}}}) == "TIME(unknown unit)"
    assert (
        _html.format_logical_type(
            {"TIMESTAMP": {"isAdjustedToUTC": True, "unit": {"MILLIS": {}}}}
        )
        == "TIMESTAMP(MILLIS) (adjusted to UTC)"
    )
    assert (
        _html.format_logical_type({"TIMESTAMP": {"unit": {"MICROS": {}}}})
        == "TIMESTAMP(MICROS)"
    )
    assert (
        _html.format_logical_type({"TIMESTAMP": {"unit": {"NANOS": {}}}})
        == "TIMESTAMP(NANOS)"
    )
    assert (
        _html.format_logical_type({"TIMESTAMP": {"unit": {}}})
        == "TIMESTAMP(unknown unit)"
    )
    # Unknown logical type falls back to its repr.
    assert _html.format_logical_type({"MAP": {}}) == "{'MAP': {}}"


def test_encode_stats_value_decimal_int64_and_fixed_len():
    decimal_type = {"DECIMAL": {"scale": 2}}
    # INT64 decimal round-trips through the shared decode kernel.
    encoded_i64 = _html.encode_stats_value(Decimal("12.34"), "INT64", 0, decimal_type)
    assert struct.unpack("<q", encoded_i64)[0] == 1234
    assert _html.decode_stats_value(encoded_i64, "INT64", decimal_type) == Decimal(
        "12.34"
    )
    # FIXED_LEN_BYTE_ARRAY decimal is big-endian two's-complement. The branch
    # currently encodes to the minimal width and IGNORES type_length (passing 4
    # below still yields 2 bytes for 1234) — see issue #61. The HTML report only
    # round-trips the value for display, so the width is not user-visible; this
    # pins the current behavior until #61 tightens it to the schema width.
    encoded_flba = _html.encode_stats_value(
        Decimal("12.34"), "FIXED_LEN_BYTE_ARRAY", 4, decimal_type
    )
    assert encoded_flba == b"\x04\xd2"  # minimal width; type_length(4) not honored
    assert int.from_bytes(encoded_flba, byteorder="big", signed=True) == 1234
    assert _html.decode_stats_value(
        encoded_flba, "FIXED_LEN_BYTE_ARRAY", decimal_type
    ) == Decimal("12.34")


def test_format_stats_value_truncates_long_string_and_bytes():
    # Short string bytes render verbatim.
    assert _html.format_stats_value(b"hello", "BYTE_ARRAY", {"STRING": {}}) == "hello"
    # Long string bytes are truncated with a remaining-characters suffix.
    long_str = _html.format_stats_value(b"a" * 300, "BYTE_ARRAY", {"STRING": {}})
    assert long_str.startswith("a" * 256)
    assert long_str.endswith("… (44 more characters)")
    # Long non-string bytes are truncated with a remaining-bytes suffix.
    long_bytes = _html.format_stats_value(b"\x01" * 300, "BYTE_ARRAY", None)
    assert long_bytes.startswith("0x" + "01" * 256)
    assert long_bytes.endswith("… (44 more bytes)")


def test_group_segments_by_page_orphan_page_without_data():
    # A "page" with no following "page_data" is emitted as-is (warned, not grouped).
    segments = [{"name": "page", "offset": 0, "length": 2, "value": []}]
    grouped = _html.group_segments_by_page(segments)
    assert len(grouped) == 1
    assert grouped[0]["name"] == "page"


def test_get_num_values_dictionary_header_and_missing():
    dict_page = {
        "value": [
            {
                "name": "dictionary_page_header",
                "value": [{"name": "num_values", "value": 11}],
            }
        ]
    }
    assert _html.get_num_values(dict_page) == 11
    # No num_values present and a known offset → returns None (warns).
    assert _html.get_num_values({"offset": 42, "value": []}) is None


def test_get_next_page_offset_edge_cases():
    # Missing length/value keys.
    assert _html.get_next_page_offset(0, {}) is None
    # Non-int length.
    assert _html.get_next_page_offset(0, {"length": "x", "value": []}) is None
    # Non-int compressed_page_size.
    bad = {"length": 5, "value": [{"name": "compressed_page_size", "value": "x"}]}
    assert _html.get_next_page_offset(0, bad) is None
    # No compressed_page_size field at all.
    assert _html.get_next_page_offset(0, {"length": 5, "value": []}) is None
    # Happy path: current + header length + compressed size.
    good = {"length": 5, "value": [{"name": "compressed_page_size", "value": 7}]}
    assert _html.get_next_page_offset(100, good) == 112


def test_fix_duckdb_data_page_offset_branches():
    # No dictionary page offset → returned unchanged.
    assert _html.fix_duckdb_data_page_offset(50, None, {}) == 50
    # Dictionary offset not in the page mapping → unchanged.
    assert _html.fix_duckdb_data_page_offset(50, 10, {}) == 50
    # Dict page header length not an int → unchanged.
    assert _html.fix_duckdb_data_page_offset(50, 10, {10: {"length": "x"}}) == 50
    # compressed_page_size not an int → unchanged.
    mapping = {
        10: {"length": 5, "value": [{"name": "compressed_page_size", "value": "x"}]}
    }
    assert _html.fix_duckdb_data_page_offset(50, 10, mapping) == 50
    # No compressed_page_size (dict_page_size stays 0) → unchanged.
    assert (
        _html.fix_duckdb_data_page_offset(50, 10, {10: {"length": 5, "value": []}})
        == 50
    )
    # data_page_offset already at/after the expected position → unchanged.
    ok_mapping = {
        10: {"length": 5, "value": [{"name": "compressed_page_size", "value": 3}]}
    }
    assert _html.fix_duckdb_data_page_offset(100, 10, ok_mapping) == 100
    # data_page_offset before the expected position → fixed to dict+header+size.
    assert _html.fix_duckdb_data_page_offset(12, 10, ok_mapping) == 18


def test_sanitize_segment_truncates_and_recurses():
    # A long scalar value is replaced by a value_truncated record.
    seg = {"name": "x", "value": "a" * 300}
    _html.sanitize_segment(seg, 256)
    assert "value" not in seg
    assert seg["value_truncated"]["original_length"] == 300
    assert seg["value_truncated"]["remaining_length"] == 44
    assert seg["value_truncated"]["value"] == "a" * 256
    # Nested list values are sanitized recursively.
    nested = {"name": "g", "value": [{"name": "y", "value": b"b" * 300}]}
    _html.sanitize_segment(nested, 256)
    assert nested["value"][0]["value_truncated"]["original_length"] == 300


def test_aggregate_column_chunks_skips_chunk_without_path():
    footer = {
        "row_groups": [
            {"columns": [{"meta_data": {"type": "INT32"}}]}  # no path_in_schema
        ]
    }
    assert _html.aggregate_column_chunks(footer, {}) == []


def test_group_segments_unmapped_page_is_kept_standalone():
    # A page whose offset maps to no column chunk (footer has none) is kept
    # standalone rather than folded into a column-chunk-pages group.
    segments = [
        {
            "name": "page",
            "offset": 300,
            "length": 10,
            "value": [
                {"name": "compressed_page_size", "value": 5},
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                },
            ],
        },
        {"name": "page_data", "offset": 310, "length": 5, "value": []},
    ]
    footer = {"row_groups": []}  # nothing to map the page to
    grouped = _html.group_segments(segments, footer)
    names = [g["name"] for g in grouped]
    assert _html.page_header_and_data_name in names
    assert _html.page_group_name not in names


def test_build_page_mapping_stops_on_indeterminate_page_with_values_left():
    # First data page accounts for some values; the next reachable page has no
    # determinable num_values, so the walk breaks with values still unmapped.
    page_segments = [
        {
            "name": "page",
            "offset": 300,
            "length": 10,
            "value": [
                {"name": "compressed_page_size", "value": 5},
                {
                    "name": "data_page_header",
                    "value": [{"name": "num_values", "value": 2}],
                },
            ],
        },
        {
            "name": "page",
            "offset": 315,  # 300 + length(10) + compressed_page_size(5)
            "length": 8,
            "value": [{"name": "compressed_page_size", "value": 0}],  # no num_values
        },
    ]
    page_mapping = _html.get_page_mapping(page_segments)
    footer = {
        "row_groups": [
            {
                "columns": [
                    {
                        "meta_data": {
                            "path_in_schema": ["c"],
                            "data_page_offset": 300,
                            "num_values": 10,
                        }
                    }
                ]
            }
        ]
    }
    mapping = _html.build_page_offset_to_column_chunk_mapping(footer, page_mapping)
    # The first data page is mapped; the indeterminate second page is not.
    assert mapping[300] == (0, 0)
    assert 315 not in mapping
