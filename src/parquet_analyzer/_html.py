import json
import logging
import pathlib
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Tuple

from jinja2 import Environment, PackageLoader, select_autoescape

logger = logging.getLogger(__name__)

env = Environment(
    loader=PackageLoader("parquet_analyzer"),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

column_chunk_pages_name = ":column_chunk_pages"
page_header_and_data_name = ":page_header_and_data"

page_group_name = ":pages"
column_index_group_name = ":column_indexes"
offset_index_group_name = ":offset_indexes"
bloom_filter_group_name = ":bloom_filters"


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
    for row_group in footer.get("row_groups", []):
        for column_chunk in row_group.get("columns", []):
            codec = column_chunk.get("meta_data", {}).get("codec")
            if codec and codec not in codecs:
                codecs.append(codec)
    return codecs


def get_encodings(footer: dict) -> list[str]:
    encodings = []
    for row_group in footer.get("row_groups", []):
        for column_chunk in row_group.get("columns", []):
            for page in column_chunk.get("meta_data", {}).get("encodings", []):
                if page and page not in encodings:
                    encodings.append(page)
    return sorted(encodings)


def aggregate_column_chunks(
    footer: dict, logical_type_mapping: dict[tuple[str, ...], dict]
) -> list[dict]:
    columns = {}
    for row_group in footer.get("row_groups", []):
        for column_chunk in row_group.get("columns", []):
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


def group_segments_by_page(segments: list[dict]) -> list[dict]:
    grouped = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment["name"] == "page":
            if index + 1 < len(segments) and segments[index + 1]["name"] == "page_data":
                grouped.append(
                    {
                        "name": page_header_and_data_name,
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
        if segment["name"] == "page":
            offset = segment["offset"]
            mapping[offset] = segment
    return mapping


def get_num_values(page: dict) -> int | None:
    for item in page.get("value", []):
        if item.get("name") == "data_page_header":
            for item2 in item.get("value", []):
                if item2.get("name") == "num_values":
                    return item2.get("value")
        elif item.get("name") == "data_page_header_v2":
            for item2 in item.get("value", []):
                if item2.get("name") == "num_values":
                    return item2.get("value")
        elif item.get("name") == "dictionary_page_header":
            for item2 in item.get("value", []):
                if item2.get("name") == "num_values":
                    return item2.get("value")
    offset = page.get("offset")
    if offset is None:
        logger.warning("Could not find num_values in page with unknown offset")
    else:
        logger.warning("Could not find num_values in page at offset %d", offset)
    return None


def get_next_page_offset(current_offset: int, page: dict) -> int | None:
    if "length" not in page or "value" not in page:
        return None
    length = page["length"]
    if not isinstance(length, int):
        return None
    for item in page["value"]:
        if item["name"] == "compressed_page_size":
            compressed_page_size = item["value"]
            if not isinstance(compressed_page_size, int):
                return None
            return current_offset + length + compressed_page_size
    return None


def build_page_offset_to_column_chunk_mapping(
    footer: dict, page_mapping: dict[int, dict]
) -> dict[int, Tuple[int, int]]:
    page_offsets = {}
    for row_group_index, row_group in enumerate(footer.get("row_groups", [])):
        for column_index, column_chunk in enumerate(row_group.get("columns", [])):
            if (
                column_chunk.get("meta_data", {}).get("dictionary_page_offset")
                is not None
            ):
                dict_page_offset = column_chunk["meta_data"]["dictionary_page_offset"]
                page_offsets[dict_page_offset] = (row_group_index, column_index)
            if column_chunk.get("meta_data", {}).get("data_page_offset") is not None:
                data_page_offset = column_chunk["meta_data"]["data_page_offset"]
                page_offsets[data_page_offset] = (row_group_index, column_index)
                remaining_values = column_chunk.get("meta_data", {}).get(
                    "num_values", 0
                )
                page = page_mapping.get(data_page_offset)
                if page is not None:
                    num_values = get_num_values(page)
                    if num_values is not None and num_values < remaining_values:
                        remaining_values -= num_values
                        next_page_offset = get_next_page_offset(data_page_offset, page)
                        while (
                            next_page_offset is not None
                            and remaining_values > 0
                            and next_page_offset in page_mapping
                        ):
                            next_page = page_mapping[next_page_offset]
                            next_num_values = get_num_values(next_page)
                            if next_num_values is None:
                                break
                            page_offsets[next_page_offset] = (
                                row_group_index,
                                column_index,
                            )
                            remaining_values -= next_num_values
                            next_page_offset = get_next_page_offset(
                                next_page_offset, next_page
                            )
                        if remaining_values > 0:
                            logger.warning(
                                "Could not map all pages for column chunk at row group %d, column %d",
                                row_group_index,
                                column_index,
                            )
    return page_offsets


def sanitize_segments(segments: list[dict]) -> list[dict]:
    max_length = 256
    for segment in segments:
        sanitize_segment(segment, max_length)
    return segments


def sanitize_segment(segment: dict, max_length: int):
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
                    sanitize_segment(item, max_length)


def group_segments(segments: list[dict], footer: dict) -> list[dict]:
    grouped_segments = group_segments_by_page(segments)
    page_mapping = get_page_mapping(segments)
    page_offset_to_column_chunk = build_page_offset_to_column_chunk_mapping(
        footer, page_mapping
    )
    final_grouped = []
    current_row_group_index = None
    current_column_index = None
    current_group: list[dict] = []
    num_total_pages = 0

    def close_current_group():
        nonlocal \
            current_row_group_index, \
            current_column_index, \
            current_group, \
            num_total_pages
        if current_group:
            final_grouped.append(
                {
                    "name": column_chunk_pages_name,
                    "value": [item for seg in current_group for item in seg["value"]],
                    "offset": current_group[0]["offset"],
                    "length": sum(seg["length"] for seg in current_group),
                    "row_group_index": current_row_group_index,
                    "column_index": current_column_index,
                    "num_pages": len(current_group),
                }
            )
            current_row_group_index = None
            current_column_index = None
            num_total_pages += len(current_group)
            current_group = []

    for group in grouped_segments:
        if group["name"] == page_header_and_data_name:
            page_offset = group["value"][0]["offset"]
            column_chunk_info = page_offset_to_column_chunk.get(page_offset)
            if column_chunk_info:
                row_group_index, column_index = column_chunk_info
                if (
                    current_row_group_index is None
                    or current_row_group_index == row_group_index
                    and current_column_index == column_index
                ):
                    current_row_group_index = row_group_index
                    current_column_index = column_index
                    current_group.append(group)
                    continue
                else:
                    close_current_group()
                    current_row_group_index = row_group_index
                    current_column_index = column_index
                    current_group = [group]
                    continue
            else:
                logger.warning(
                    "Page at offset %d not mapped to any column chunk", page_offset
                )
                close_current_group()
                final_grouped.append(group)
                num_total_pages += 1
        else:
            close_current_group()
            final_grouped.append(group)

    def make_groups(items: list[dict], group_names: dict[str, str]) -> list[dict]:
        result = []
        index = 0
        while index < len(items):
            item = items[index]
            if item["name"] in group_names:
                group_name = group_names[item["name"]]
                group_items = []
                while index < len(items) and items[index]["name"] == item["name"]:
                    group_items.append(items[index])
                    index += 1
                result.append(
                    {
                        "name": group_name,
                        "value": group_items,
                        "offset": group_items[0]["offset"],
                        "length": sum(seg["length"] for seg in group_items),
                    }
                )
            else:
                result.append(item)
                index += 1
        return result

    return make_groups(
        final_grouped,
        {
            column_chunk_pages_name: ":pages",
            "column_index": ":column_indexes",
            "offset_index": ":offset_indexes",
            "bloom_filter": ":bloom_filters",
        },
    )


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
    if segment["name"].startswith(":"):
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


def generate_html_report(
    file_path,
    summary,
    footer,
    segments,
    sections=[],
) -> str:
    template = env.get_template("report.html")
    codecs = get_codecs(footer)
    encodings = get_encodings(footer)
    schema_tree = build_schema_tree(
        [SchemaElement.from_json(elem) for elem in footer["schema"]]
    )
    logical_type_mapping = build_logical_type_mapping(schema_tree)
    columns = aggregate_column_chunks(footer, logical_type_mapping)
    if "segments" in sections:
        segments = sanitize_segments(segments)
        grouped_segments = group_segments(segments, footer)
        segment_class_mapping = {
            "magic_number": "segment--magic",
            "footer_length": "segment--value",
            column_chunk_pages_name: "segment--column-chunk-pages",
            "page": "segment--page",
            "page_data": "segment--page",
            page_group_name: "segment--group",
            column_index_group_name: "segment--group",
            offset_index_group_name: "segment--group",
            bloom_filter_group_name: "segment--group",
        }
    else:
        grouped_segments = None
        segment_class_mapping = None
    html = template.render(
        filename=pathlib.Path(file_path).name,
        file_path=file_path,
        summary=summary,
        footer=footer,
        schema_tree=schema_tree,
        codecs=codecs,
        encodings=encodings,
        columns=columns,
        logical_type_mapping=logical_type_mapping,
        grouped_segments=grouped_segments,
        segment_class_mapping=segment_class_mapping,
        sections=sections,
    )
    return html
