import json
import logging
import pathlib
import struct
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby, islice, takewhile
from typing import Any, Tuple

from jinja2 import Environment, PackageLoader, select_autoescape

logger = logging.getLogger(__name__)

env = Environment(
    loader=PackageLoader("parquet_analyzer"),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

page_segment_name = ":page"
column_chunk_pages_group_name = ":column_chunk_pages"
row_group_pages_group_name = ":row_group_pages"
row_group_column_indexes_group_name = ":row_group_column_indexes"
row_group_offset_indexes_group_name = ":row_group_offset_indexes"
row_group_bloom_filters_group_name = ":row_group_bloom_filters"
pages_segment_name = ":pages"
column_indexes_segment_name = ":column_indexes"
offset_indexes_segment_name = ":offset_indexes"
bloom_filters_segment_name = ":bloom_filters"

group_segment_names = [
    page_segment_name,
    column_chunk_pages_group_name,
    row_group_pages_group_name,
    row_group_column_indexes_group_name,
    row_group_offset_indexes_group_name,
    row_group_bloom_filters_group_name,
    pages_segment_name,
    column_indexes_segment_name,
    offset_indexes_segment_name,
    bloom_filters_segment_name,
]


@dataclass
class SchemaElement:
    type: str | None
    type_length: int | None
    repetition_type: str | None
    name: str
    num_children: int | None
    converted_type: str | None
    scale: int | None
    precision: int | None
    field_id: int | None
    logical_type: dict | None
    children: list["SchemaElement"]

    @staticmethod
    def from_json(obj: dict) -> "SchemaElement":
        return SchemaElement(
            type=obj.get("type"),
            type_length=obj.get("type_length"),
            repetition_type=obj.get("repetition_type"),
            name=obj["name"],
            num_children=obj.get("num_children"),
            converted_type=obj.get("converted_type"),
            scale=obj.get("scale"),
            precision=obj.get("precision"),
            field_id=obj.get("field_id"),
            logical_type=obj.get("logicalType"),
            children=[],
        )


def build_schema_tree(schema_elements: list[SchemaElement]) -> list[SchemaElement]:
    def build_tree(index: int) -> tuple[SchemaElement, int]:
        element = schema_elements[index]
        node = SchemaElement(
            type=element.type,
            type_length=element.type_length,
            repetition_type=element.repetition_type,
            name=element.name,
            num_children=element.num_children,
            converted_type=element.converted_type,
            scale=element.scale,
            precision=element.precision,
            field_id=element.field_id,
            logical_type=element.logical_type,
            children=[],
        )
        index += 1
        if element.num_children:
            for _ in range(element.num_children):
                child, index = build_tree(index)
                node.children.append(child)
        return node, index

    tree = []
    index = 0
    while index < len(schema_elements):
        node, index = build_tree(index)
        tree.append(node)
    return tree


def build_logical_type_mapping(
    schema_tree: list[SchemaElement],
) -> dict[tuple[str, ...], dict]:
    mapping = {}

    def traverse(node: SchemaElement, path: tuple[str, ...]):
        current_path = path + (node.name,)
        if node.logical_type:
            mapping[current_path] = node.logical_type
        for child in node.children:
            traverse(child, current_path)

    for root in schema_tree:
        traverse(root, ())

    # Drop the first element which is the root schema
    return {k[1:]: v for k, v in mapping.items()}


def get_codecs(footer: dict) -> list[str]:
    codecs = []
    for row_group in footer["row_groups"]:
        for column_chunk in row_group["columns"]:
            codec = column_chunk.get("meta_data", {}).get("codec")
            if codec and codec not in codecs:
                codecs.append(codec)
    return codecs


def get_encodings(footer: dict) -> list[str]:
    encodings = []
    for row_group in footer["row_groups"]:
        for column_chunk in row_group["columns"]:
            for page in column_chunk.get("meta_data", {}).get("encodings", []):
                if page and page not in encodings:
                    encodings.append(page)
    return sorted(encodings)


def aggregate_column_chunks(
    footer: dict, logical_type_mapping: dict[tuple[str, ...], dict]
) -> list[dict]:
    columns = {}
    for row_group in footer["row_groups"]:
        for column_chunk in row_group["columns"]:
            if "path_in_schema" not in column_chunk.get("meta_data", {}):
                continue
            path_in_schema = tuple(column_chunk["meta_data"]["path_in_schema"])
            data_type = column_chunk.get("meta_data", {}).get("type")
            logical_type = logical_type_mapping.get(path_in_schema)
            if path_in_schema not in columns:
                columns[path_in_schema] = {
                    "path_in_schema": path_in_schema,
                    "type": data_type,
                    "type_length": column_chunk.get("meta_data", {}).get("type_length"),
                    "num_values": 0,
                    "total_uncompressed_size": 0,
                    "total_compressed_size": 0,
                    "encodings": set(),
                    "encoding_stats": {},
                    "codecs": set(),
                }
            columns[path_in_schema]["num_values"] += column_chunk.get(
                "meta_data", {}
            ).get("num_values", 0)
            columns[path_in_schema]["total_uncompressed_size"] += column_chunk.get(
                "meta_data", {}
            ).get("total_uncompressed_size", 0)
            columns[path_in_schema]["total_compressed_size"] += column_chunk.get(
                "meta_data", {}
            ).get("total_compressed_size", 0)
            columns[path_in_schema]["encodings"].update(
                column_chunk.get("meta_data", {}).get("encodings", [])
            )
            if column_chunk.get("meta_data", {}).get("statistics"):
                stats = column_chunk["meta_data"]["statistics"]
                stats_aggr = columns[path_in_schema].setdefault("statistics", {})
                if "null_count" in stats:
                    stats_aggr["null_count"] = (
                        stats_aggr.get("null_count", 0) + stats["null_count"]
                    )
                if "min_value" in stats and data_type is not None:
                    decoded_value = decode_stats_value(
                        stats["min_value"], data_type, logical_type
                    )
                    if "min_value" not in stats_aggr:
                        stats_aggr["min_value"] = decoded_value
                    else:
                        if decoded_value < stats_aggr["min_value"]:
                            stats_aggr["min_value"] = decoded_value
                if "max_value" in stats and data_type is not None:
                    decoded_value = decode_stats_value(
                        stats["max_value"], data_type, logical_type
                    )
                    if "max_value" not in stats_aggr:
                        stats_aggr["max_value"] = decoded_value
                    else:
                        if decoded_value > stats_aggr["max_value"]:
                            stats_aggr["max_value"] = decoded_value
                if "is_min_value_exact" in stats:
                    stats_aggr["is_min_value_exact"] = (
                        stats_aggr.get("is_min_value_exact", True)
                        and stats["is_min_value_exact"]
                    )
                if "is_max_value_exact" in stats:
                    stats_aggr["is_max_value_exact"] = (
                        stats_aggr.get("is_max_value_exact", True)
                        and stats["is_max_value_exact"]
                    )
            if column_chunk.get("meta_data", {}).get("encoding_stats"):
                for item in column_chunk.get("meta_data", {})["encoding_stats"]:
                    key = (item["page_type"], item["encoding"])
                    if key not in columns[path_in_schema]["encoding_stats"]:
                        columns[path_in_schema]["encoding_stats"][key] = {
                            "page_type": item["page_type"],
                            "encoding": item["encoding"],
                            "count": 0,
                        }
                    columns[path_in_schema]["encoding_stats"][key]["count"] += item[
                        "count"
                    ]
            columns[path_in_schema]["codecs"].add(
                column_chunk.get("meta_data", {}).get("codec")
            )
    for path_in_schema, col in columns.items():
        if "statistics" in col:
            logical_type = logical_type_mapping.get(path_in_schema)
            if "min_value" in col["statistics"]:
                col["statistics"]["min_value"] = encode_stats_value(
                    col["statistics"]["min_value"],
                    col["type"],
                    col["type_length"],
                    logical_type,
                )
            if "max_value" in col["statistics"]:
                col["statistics"]["max_value"] = encode_stats_value(
                    col["statistics"]["max_value"],
                    col["type"],
                    col["type_length"],
                    logical_type,
                )
    return list(columns.values())


def truncate_segments(segments: list[dict]) -> list[dict]:
    def truncate_segment(segment: dict, max_length: int):
        if "value" in segment:
            if isinstance(segment["value"], (bytes, str)):
                original_value = segment["value"]
                n = len(original_value)
                if n > max_length:
                    del segment["value"]
                    segment["value_truncated"] = {
                        "value": original_value[:max_length],
                        "original_length": n,
                        "remaining_length": n - max_length,
                    }
            elif isinstance(segment["value"], list):
                for item in segment["value"]:
                    if isinstance(item, dict):
                        truncate_segment(item, max_length)

    max_length = 256
    for segment in segments:
        truncate_segment(segment, max_length)
    return segments


def group_page_header_and_data(segments: list[dict]) -> list[dict]:
    grouped = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment["name"] == ":page_header":
            if (
                index + 1 < len(segments)
                and segments[index + 1]["name"] == ":page_data"
            ):
                grouped.append(
                    {
                        "name": page_segment_name,
                        "value": [segment, segments[index + 1]],
                        "offset": segment["offset"],
                        "length": segment["length"] + segments[index + 1]["length"],
                    }
                )
                index += 2
            else:
                logger.warning(
                    "Page at offset %d has no corresponding page_data segment",
                    segment["offset"],
                )
                grouped.append(segment)
                index += 1
        else:
            grouped.append(segment)
            index += 1
    return grouped


def get_page_mapping(segments: list[dict]) -> dict[int, dict]:
    mapping = {}
    for segment in segments:
        if segment["name"] == ":page_header":
            offset = segment["offset"]
            mapping[offset] = segment
    return mapping


def build_page_offset_to_column_chunk_mapping(
    footer: dict,
    page_mapping: dict[int, dict],
    offset_mapping: dict[int, Tuple[int, int]],
):
    # Fix DuckDB data_page_offset if dictionary_page_offset exists
    # https://github.com/duckdb/duckdb/issues/10829
    def fix_duckdb_data_page_offset(
        data_page_offset: int,
        dict_page_offset: int,
    ) -> int:
        dict_page = page_mapping.get(dict_page_offset)
        if dict_page is None:
            return data_page_offset
        dict_page_header_length = dict_page["length"]
        dict_page_size = 0
        for field in dict_page["value"]:
            if field["name"] == "compressed_page_size":
                compressed_page_size = field["value"]
                dict_page_size = compressed_page_size
                break
        if dict_page_size == 0:
            return data_page_offset
        expected_data_page_offset = (
            dict_page_offset + dict_page_header_length + dict_page_size
        )
        if data_page_offset < expected_data_page_offset:
            logger.warning(
                "Fixing DuckDB data_page_offset from %d to %d",
                data_page_offset,
                expected_data_page_offset,
            )
            return expected_data_page_offset
        return data_page_offset

    def get_num_values(page: dict) -> int:
        for item in page["value"]:
            if item["name"] in ("data_page_header", "data_page_header_v2"):
                for item2 in item["value"]:
                    if item2["name"] == "num_values":
                        return item2["value"]
        return 0

    def get_next_page_offset(current_offset: int, page: dict) -> int | None:
        length = page["length"]
        for item in page["value"]:
            if item["name"] == "compressed_page_size":
                compressed_page_size = item["value"]
                return current_offset + length + compressed_page_size
        return None

    for row_group_index, row_group in enumerate(footer["row_groups"]):
        for column_index, column_chunk in enumerate(row_group["columns"]):
            if "meta_data" not in column_chunk:
                logger.warning(
                    "Column chunk at row group %d, column %d has no meta_data",
                    row_group_index,
                    column_index,
                )
                continue
            metadata = column_chunk["meta_data"]

            data_page_offset = metadata["data_page_offset"]

            # Record dictionary page offset and fix data page offset if needed
            dict_page_offset = metadata.get("dictionary_page_offset")
            if dict_page_offset is not None:
                offset_mapping[dict_page_offset] = (row_group_index, column_index)
                data_page_offset = fix_duckdb_data_page_offset(
                    column_chunk["meta_data"]["data_page_offset"],
                    dict_page_offset,
                )

            remaining_values = metadata["num_values"]
            while remaining_values > 0 and data_page_offset is not None:
                page = page_mapping.get(data_page_offset)
                if page is None:
                    logger.warning(
                        "Data page at offset %d not found in page mapping",
                        data_page_offset,
                    )
                    break
                offset_mapping[data_page_offset] = (row_group_index, column_index)
                num_values = get_num_values(page)
                remaining_values -= num_values
                data_page_offset = get_next_page_offset(data_page_offset, page)

            if remaining_values > 0:
                logger.warning(
                    "Could not map all pages for column chunk at row group %d, column %d",
                    row_group_index,
                    column_index,
                )


def build_offset_to_column_chunk_mapping(
    footer: dict, offset_mapping: dict[int, Tuple[int, int]]
):
    for row_group_index, row_group in enumerate(footer["row_groups"]):
        for column_index, column_chunk in enumerate(row_group["columns"]):
            column_index_offset = column_chunk.get("column_index_offset")
            if column_index_offset is not None:
                offset_mapping[column_index_offset] = (row_group_index, column_index)
            offset_index_offset = column_chunk.get("offset_index_offset")
            if offset_index_offset is not None:
                offset_mapping[offset_index_offset] = (row_group_index, column_index)
            if "meta_data" not in column_chunk:
                logger.warning(
                    "Column chunk at row group %d, column %d has no meta_data",
                    row_group_index,
                    column_index,
                )
                continue
            metadata = column_chunk["meta_data"]
            bloom_filter_offset = metadata.get("bloom_filter_offset")
            if bloom_filter_offset is not None:
                offset_mapping[bloom_filter_offset] = (row_group_index, column_index)


def group_by_column_chunk(
    segments: list[dict], offset_mapping: dict[int, Tuple[int, int]]
) -> list[dict]:
    result = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment["name"] == page_segment_name:
            group_name = column_chunk_pages_group_name
            run = takewhile(
                lambda s: s["name"] == segment["name"], islice(segments, index, None)
            )
            grouped = groupby(
                run, key=lambda s: offset_mapping.get(s["offset"], (-1, -1))
            )
            for (row_group_index, column_index), group in grouped:
                items = list(group)
                result.append(
                    {
                        "name": group_name,
                        "value": items,
                        "offset": items[0]["offset"],
                        "length": sum(p["length"] for p in items),
                        "row_group_index": row_group_index,
                        "column_index": column_index,
                        "num_pages": len(items),
                    }
                )
                index += len(items)
        else:
            result.append(segment)
            index += 1
    return result


def group_by_row_group(
    segments: list[dict], offset_mapping: dict[int, Tuple[int, int]]
) -> list[dict]:
    result = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment["name"] in (
            column_chunk_pages_group_name,
            ":column_index",
            ":offset_index",
            ":bloom_filter",
        ):
            group_name_mapping = {
                column_chunk_pages_group_name: row_group_pages_group_name,
                ":column_index": row_group_column_indexes_group_name,
                ":offset_index": row_group_offset_indexes_group_name,
                ":bloom_filter": row_group_bloom_filters_group_name,
            }
            group_name = group_name_mapping[segment["name"]]
            run = takewhile(
                lambda s: s["name"] == segment["name"], islice(segments, index, None)
            )
            grouped = groupby(
                run, key=lambda s: offset_mapping.get(s["offset"], (-1, -1))[0]
            )
            for row_group_index, group in grouped:
                items = list(group)
                grouped_segment = {
                    "name": group_name,
                    "value": items,
                    "offset": items[0]["offset"],
                    "length": sum(p["length"] for p in items),
                    "row_group_index": row_group_index,
                }
                if segment["name"] == column_chunk_pages_group_name:
                    grouped_segment["num_pages"] = sum(
                        item["num_pages"] for item in items
                    )
                result.append(grouped_segment)
                index += len(items)
        else:
            result.append(segment)
            index += 1
    return result


def group_by_type(segments: list[dict]) -> list[dict]:
    result = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment["name"] in (
            row_group_pages_group_name,
            row_group_column_indexes_group_name,
            row_group_offset_indexes_group_name,
            row_group_bloom_filters_group_name,
        ):
            group_name_mapping = {
                row_group_pages_group_name: pages_segment_name,
                row_group_column_indexes_group_name: column_indexes_segment_name,
                row_group_offset_indexes_group_name: offset_indexes_segment_name,
                row_group_bloom_filters_group_name: bloom_filters_segment_name,
            }
            group_name = group_name_mapping[segment["name"]]
            run = takewhile(
                lambda s: s["name"] == segment["name"], islice(segments, index, None)
            )
            items = list(run)
            grouped_segment = {
                "name": group_name,
                "value": items,
                "offset": items[0]["offset"],
                "length": sum(p["length"] for p in items),
            }
            if segment["name"] == row_group_pages_group_name:
                grouped_segment["num_pages"] = sum(item["num_pages"] for item in items)
            result.append(grouped_segment)
            index += len(items)
        else:
            result.append(segment)
            index += 1
    return result


def group_segments(segments: list[dict], footer: dict) -> list[dict]:
    page_mapping = get_page_mapping(segments)
    segments = group_page_header_and_data(segments)
    offset_mapping: dict[int, Tuple[int, int]] = {}
    build_page_offset_to_column_chunk_mapping(footer, page_mapping, offset_mapping)
    build_offset_to_column_chunk_mapping(footer, offset_mapping)

    # Group :page into :column_chunk_pages
    segments = group_by_column_chunk(segments, offset_mapping)

    # Group :column_chunk_pages into :row_group_pages
    # and group :column_index, :offset_index, :bloom_filter segments into
    # :row_group_column_indexes, :row_group_offset_indexes, :row_group_bloom_filters
    segments = group_by_row_group(segments, offset_mapping)

    # Group :row_group_xxx into :xxx
    segments = group_by_type(segments)

    return segments


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: "bytes", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while size >= power and n < 4:
        size /= power
        n += 1
    if n == 0:
        return f"{int(size)} {power_labels[n]}"
    else:
        return f"{size:.2f} {power_labels[n]}"


def format_logical_type(logical_type: dict[str, Any]) -> str:
    if "INTEGER" in logical_type:
        int_info = logical_type["INTEGER"]
        bit_width = int_info.get("bitWidth", "unknown")
        is_signed = int_info.get("isSigned", True)
        sign_str = "SIGNED" if is_signed else "UNSIGNED"
        return f"{sign_str} {bit_width}-BIT INTEGER"
    if "STRING" in logical_type:
        return "STRING"
    if "DATE" in logical_type:
        return "DATE"
    if "TIME" in logical_type:
        time_info = logical_type["TIME"]
        is_adjusted_to_utc = time_info.get("isAdjustedToUTC", False)
        unit = time_info.get("unit", {})
        if "MILLIS" in unit:
            unit_str = "MILLIS"
        elif "MICROS" in unit:
            unit_str = "MICROS"
        elif "NANOS" in unit:
            unit_str = "NANOS"
        else:
            unit_str = "unknown unit"
        utc_str = " (adjusted to UTC)" if is_adjusted_to_utc else ""
        return f"TIME({unit_str}){utc_str}"
    if "TIMESTAMP" in logical_type:
        timestamp_info = logical_type["TIMESTAMP"]
        is_adjusted_to_utc = timestamp_info.get("isAdjustedToUTC", False)
        unit = timestamp_info.get("unit", {})
        if "MILLIS" in unit:
            unit_str = "MILLIS"
        elif "MICROS" in unit:
            unit_str = "MICROS"
        elif "NANOS" in unit:
            unit_str = "NANOS"
        else:
            unit_str = "unknown unit"
        utc_str = " (adjusted to UTC)" if is_adjusted_to_utc else ""
        return f"TIMESTAMP({unit_str}){utc_str}"
    if "DECIMAL" in logical_type:
        decimal_info = logical_type["DECIMAL"]
        precision = decimal_info.get("precision", "unknown")
        scale = decimal_info.get("scale", "unknown")
        return f"DECIMAL({precision},{scale})"
    return str(logical_type)


def decode_stats_value(binary_value, type_str: str, logical_type: dict | None) -> Any:
    if logical_type is not None and "DECIMAL" in logical_type:
        scale = logical_type["DECIMAL"].get("scale", 0)
        if type_str == "FIXED_LEN_BYTE_ARRAY":
            int_value = int.from_bytes(binary_value, byteorder="big", signed=True)
            return Decimal(int_value).scaleb(-scale)
        if type_str == "INT32" or type_str == "INT64":
            int_value = int.from_bytes(binary_value, byteorder="little", signed=True)
            return Decimal(int_value).scaleb(-scale)
    if type_str == "INT32" or type_str == "INT64":
        int_value = int.from_bytes(binary_value, byteorder="little", signed=True)
        return int_value
    if type_str == "FLOAT":
        float_value = struct.unpack("<f", binary_value)[0]
        return float_value
    if type_str == "DOUBLE":
        double_value = struct.unpack("<d", binary_value)[0]
        return double_value
    if type_str == "BOOLEAN":
        bool_value = bool(int.from_bytes(binary_value, byteorder="little"))
        return bool_value
    return binary_value


def encode_stats_value(
    value: Any, type_str: str, type_length: int, logical_type: dict | None
) -> bytes:
    if logical_type is not None and "DECIMAL" in logical_type:
        scale = logical_type["DECIMAL"].get("scale", 0)
        if type_str == "FIXED_LEN_BYTE_ARRAY":
            scaled = int(value.scaleb(scale))
            bitlen = scaled.bit_length() or 1
            length = (bitlen + 8) // 8
            return scaled.to_bytes(length, byteorder="big", signed=True)
        if type_str == "INT32":
            scaled = int(value.scaleb(scale))
            return struct.pack("<i", scaled)
        if type_str == "INT64":
            scaled = int(value.scaleb(scale))
            return struct.pack("<q", scaled)
    if type_str == "INT32":
        return struct.pack("<i", value)
    if type_str == "INT64":
        return struct.pack("<q", value)
    if type_str == "FLOAT":
        return struct.pack("<f", value)
    if type_str == "DOUBLE":
        return struct.pack("<d", value)
    if type_str == "BOOLEAN":
        return struct.pack("<?", value)
    return value


def format_stats_value(binary_value, type_str: str, logical_type: dict | None) -> str:
    decoded_value = decode_stats_value(binary_value, type_str, logical_type)
    if isinstance(decoded_value, bytes):
        max_length = 256
        if "STRING" in (logical_type or {}):
            s = decoded_value.decode("utf-8", errors="replace")
            if len(s) <= max_length:
                return s
            else:
                r = len(s) - max_length
                return s[:max_length] + f"… ({r} more characters)"
        else:
            if len(decoded_value) <= max_length:
                return f"0x{decoded_value.hex()}"
            else:
                r = len(decoded_value) - max_length
                return f"0x{decoded_value[:max_length].hex()}… ({r} more bytes)"
    return str(decoded_value)


def to_nice_json(value):
    return json.dumps(value, indent=2, default=lambda x: str(x))


def is_nested_segment(segment: Any) -> bool:
    if not isinstance(segment, dict):
        return False
    if segment["name"] in group_segment_names:
        return True
    if "metadata" in segment:
        metadata = segment["metadata"]
        if "type_class" in metadata:
            return True
        if metadata.get("type") == "list":
            return all(is_nested_segment(item) for item in segment.get("value", []))
    return False


env.globals["format_bytes"] = format_bytes
env.globals["format_logical_type"] = format_logical_type
env.globals["format_stats_value"] = format_stats_value
env.globals["to_nice_json"] = to_nice_json
env.globals["is_nested_segment"] = is_nested_segment
env.filters["tuple"] = lambda value: tuple(value)


@dataclass
class ReportElements:
    footer: dict
    summary: dict
    codecs: list[str]
    encodings: list[str]
    logical_type_mapping: dict[tuple[str, ...], dict]
    schema_tree: list[SchemaElement]
    columns: list[dict]
    grouped_segments: list[dict]


def make_report_elements(summary, footer, segments) -> ReportElements:
    codecs = get_codecs(footer)
    encodings = get_encodings(footer)
    schema_tree = build_schema_tree(
        [SchemaElement.from_json(elem) for elem in footer["schema"]]
    )
    logical_type_mapping = build_logical_type_mapping(schema_tree)
    columns = aggregate_column_chunks(footer, logical_type_mapping)
    grouped_segments = group_segments(segments, footer)
    return ReportElements(
        footer=footer,
        summary=summary,
        codecs=codecs,
        encodings=encodings,
        logical_type_mapping=logical_type_mapping,
        schema_tree=schema_tree,
        columns=columns,
        grouped_segments=grouped_segments,
    )


segment_class_mapping = {
    ":magic_number": "segment--magic",
    ":footer_length": "segment--value",
    ":page_header": "segment--page-header",
    ":page_data": "segment--page-data",
    ":column_index": "segment--column-index",
    ":offset_index": "segment--offset-index",
    ":bloom_filter": "segment--bloom-filter",
    page_segment_name: "segment--page",
    column_chunk_pages_group_name: "segment--column-chunk-pages",
    row_group_pages_group_name: "segment--row-group-pages",
    row_group_column_indexes_group_name: "segment--row-group-column-index",
    row_group_offset_indexes_group_name: "segment--row-group-offset-index",
    row_group_bloom_filters_group_name: "segment--row-group-bloom-filter",
    pages_segment_name: "segment--pages",
    column_indexes_segment_name: "segment--column-indexes",
    offset_indexes_segment_name: "segment--offset-indexes",
    bloom_filters_segment_name: "segment--bloom-filters",
}


def generate_html_report(
    file_path,
    elements: ReportElements,
    sections=[],
    template_name="report.html",
    **kwargs,
) -> str:
    template = env.get_template(template_name)
    html = template.render(
        filename=pathlib.Path(file_path).name,
        file_path=file_path,
        summary=elements.summary,
        footer=elements.footer,
        schema_tree=elements.schema_tree,
        codecs=elements.codecs,
        encodings=elements.encodings,
        columns=elements.columns,
        logical_type_mapping=elements.logical_type_mapping,
        grouped_segments=elements.grouped_segments,
        segment_class_mapping=segment_class_mapping,
        sections=sections,
        **kwargs,
    )
    return html
