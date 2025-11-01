"""Core functionality for inspecting Parquet files."""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict, Iterable, List, Optional, Tuple

from thrift.protocol import TCompactProtocol, TProtocol
from thrift.protocol.TProtocol import TType
from thrift.transport.TTransport import TMemoryBuffer, TTransportBase

from parquet.ttypes import (
    BloomFilterHeader,
    BoundaryOrder,
    ColumnIndex,
    ColumnMetaData,
    CompressionCodec,
    ConvertedType,
    DataPageHeader,
    DataPageHeaderV2,
    DictionaryPageHeader,
    EdgeInterpolationAlgorithm,
    Encoding,
    FieldRepetitionType,
    FileMetaData,
    GeographyType,
    OffsetIndex,
    PageEncodingStats,
    PageHeader,
    PageType,
    SchemaElement,
    Type,
)

__all__ = [
    "OffsetRecordingProtocol",
    "OffsetRecordingCompactProtocol",
    "TFileTransport",
    "create_segment",
    "create_segment_from_offset_info",
    "fill_gaps",
    "find_footer_segment",
    "get_pages",
    "get_summary",
    "json_encode",
    "parse_parquet_file",
    "read_bloom_filter",
    "read_column_index",
    "read_dictionary_page",
    "read_offset_index",
    "read_pages",
    "read_thrift_segment",
    "segment_to_json",
]


class OffsetRecordingProtocol(TProtocol.TProtocolBase):
    logger = logging.getLogger(__qualname__)

    type_map = {
        TType.BOOL: "bool",
        TType.BYTE: "i8",
        TType.I16: "i16",
        TType.I32: "i32",
        TType.I64: "i64",
        TType.DOUBLE: "double",
        TType.STRING: "string",
        TType.STRUCT: "struct",
        TType.MAP: "map",
        TType.SET: "set",
        TType.LIST: "list",
    }

    enum_map = {
        ColumnMetaData: {
            "codec": CompressionCodec,
            "type": Type,
            "encodings": Encoding,
        },
        SchemaElement: {
            "converted_type": ConvertedType,
            "type": Type,
            "repetition_type": FieldRepetitionType,
        },
        GeographyType: {
            "algorithm": EdgeInterpolationAlgorithm,
        },
        PageEncodingStats: {
            "encoding": Encoding,
            "page_type": PageType,
        },
        PageHeader: {
            "type": PageType,
        },
        DataPageHeader: {
            "encoding": Encoding,
            "definition_level_encoding": Encoding,
            "repetition_level_encoding": Encoding,
        },
        DataPageHeaderV2: {
            "encoding": Encoding,
        },
        DictionaryPageHeader: {
            "encoding": Encoding,
        },
        ColumnIndex: {
            "boundary_order": BoundaryOrder,
        },
    }

    def __init__(self, trans: TTransportBase, name: str, struct_class: Any):
        super().__init__(trans)
        self._parents: List[Dict[str, Any]] = []
        self._current: Dict[str, Any] = {
            "name": name,
            "type": "struct",
            "type_class": struct_class,
            "spec": (struct_class, struct_class.thrift_spec),
            "range_from": None,
            "range_to": None,
            "value": [],
        }

    def get_offset_info(self) -> Dict[str, Any]:
        return self._current

    def readStructBegin(self):  # type: ignore[override]
        ret = super().readStructBegin()
        self.logger.debug("readStructBegin: %s", ret)
        if self._current["type"] == "list":
            type_id, (struct_class, spec), required = self._current["spec"]
            assert type_id == TType.STRUCT
            self._new_child(
                {
                    "name": "element",
                    "type": "struct",
                    "type_class": struct_class,
                    "spec": (struct_class, spec),
                    "range_from": None,
                    "range_to": None,
                    "value": [],
                }
            )
        self._current["range_from"] = self._get_pos()
        return ret

    def readStructEnd(self):  # type: ignore[override]
        ret = super().readStructEnd()
        self.logger.debug("readStructEnd: %s", ret)
        self._current["range_to"] = self._get_pos()
        if self._has_parent(lambda p: p["type"] == "list"):
            self._finish_child()
        return ret

    def readFieldBegin(self):  # type: ignore[override]
        assert self._current["type"] == "struct"
        ret = super().readFieldBegin()
        self.logger.debug("readFieldBegin: %s (struct: %s)", ret, self._current["name"])
        _, type_id, field_id = ret
        if field_id > 0:
            spec = self._current["spec"]
            field_info = spec[1][field_id]
            field_id, field_type_id, field_name, field_spec, _ = field_info
            if field_type_id == TType.STRUCT:
                type_class = field_spec[0]
            else:
                type_class = None
            self._new_child(
                {
                    "name": field_name,
                    "type": self.type_map[field_type_id],
                    "type_class": type_class,
                    "spec": field_spec,
                    "range_from": self._get_pos(),
                    "range_to": None,
                    "value": [] if self._is_complex_type(field_type_id) else None,
                }
            )
        return ret

    def readFieldEnd(self):  # type: ignore[override]
        ret = super().readFieldEnd()
        self.logger.debug("readFieldEnd: %s", ret)
        self._current["range_to"] = self._get_pos()
        self._finish_child()
        return ret

    def readListBegin(self):  # type: ignore[override]
        ret = super().readListBegin()
        self.logger.debug("readListBegin: %s", ret)
        return ret

    def readListEnd(self):  # type: ignore[override]
        ret = super().readListEnd()
        self.logger.debug("readListEnd: %s", ret)
        return ret

    def readMapBegin(self):  # type: ignore[override]
        ret = super().readMapBegin()
        self.logger.debug("readMapBegin: %s", ret)
        return ret

    def readMapEnd(self):  # type: ignore[override]
        ret = super().readMapEnd()
        self.logger.debug("readMapEnd: %s", ret)
        return ret

    def readSetBegin(self):  # type: ignore[override]
        ret = super().readSetBegin()
        self.logger.debug("readSetBegin: %s", ret)
        return ret

    def readSetEnd(self):  # type: ignore[override]
        ret = super().readSetEnd()
        self.logger.debug("readSetEnd: %s", ret)
        return ret

    def readMessageBegin(self):  # type: ignore[override]
        ret = super().readMessageBegin()
        self.logger.debug("readMessageBegin: %s", ret)
        return ret

    def readMessageEnd(self):  # type: ignore[override]
        ret = super().readMessageEnd()
        self.logger.debug("readMessageEnd: %s", ret)
        return ret

    def readByte(self):  # type: ignore[override]
        ret = super().readByte()
        self.logger.debug("readByte: %s", ret)
        self._append_value(ret)
        return ret

    def readI16(self):  # type: ignore[override]
        ret = super().readI16()
        self.logger.debug("readI16: %s", ret)
        self._append_value(ret)
        return ret

    def readI32(self):  # type: ignore[override]
        ret = super().readI32()
        self.logger.debug("readI32: %s", ret)
        self._append_value(ret)
        return ret

    def readI64(self):  # type: ignore[override]
        ret = super().readI64()
        self.logger.debug("readI64: %s", ret)
        self._append_value(ret)
        return ret

    def readDouble(self):  # type: ignore[override]
        ret = super().readDouble()
        self.logger.debug("readDouble: %s", ret)
        self._append_value(ret)
        return ret

    def readBool(self):  # type: ignore[override]
        ret = super().readBool()
        self.logger.debug("readBool: %s", ret)
        self._append_value(ret)
        return ret

    def readString(self):  # type: ignore[override]
        ret = super().readString()
        self.logger.debug("readString: %s", ret)
        self._append_value(ret)
        return ret

    def readBinary(self):  # type: ignore[override]
        ret = super().readBinary()
        self.logger.debug("readBinary: %s", ret)
        if (
            self._current["type"] == "string"
            and self._current["spec"] == "BINARY"
            or (
                self._current["type"] == "list"
                and self._current["spec"][0] == TType.STRING
                and self._current["spec"][1] == "BINARY"
            )
        ):
            self._append_value(ret)
        return ret

    def _get_pos(self) -> int:
        if isinstance(self.trans, TMemoryBuffer):
            return self.trans._buffer.tell()
        if isinstance(self.trans, TFileTransport):
            return self.trans.tell()
        raise RuntimeError(f"unsupported transport: {self.trans}")

    def _is_complex_type(self, type_id: int) -> bool:
        return type_id in {TType.STRUCT, TType.MAP, TType.SET, TType.LIST}

    def _has_parent(self, predicate: Any) -> bool:
        return self._parents and predicate(self._parents[-1])

    def _get_parent(self) -> Dict[str, Any]:
        return self._parents[-1]

    def _append_value(self, value: Any) -> None:
        if isinstance(self._current["value"], list):
            self._current["value"].append(value)
        else:
            self._current["value"] = value
        self._annotate_enum()

    def _annotate_enum(self) -> None:
        if self._has_parent(
            lambda p: p["type"] in ("struct", "list")
            and self._is_enum(p["type_class"], self._current["name"])
        ):
            enum_class, name = self._get_enum(
                self._get_parent()["type_class"], self._current["name"]
            )
            self._current["enum_type"] = enum_class.__name__
            self._current["enum_name"] = name

    def _is_enum(self, parent_class: Any, field_name: str) -> bool:
        return self.enum_map.get(parent_class, {}).get(field_name) is not None

    def _get_enum(self, parent_class: Any, field_name: str) -> Tuple[Any, Any]:
        enum_class = self.enum_map.get(parent_class, {}).get(field_name)
        value = self._current["value"]
        if isinstance(value, list):
            return (enum_class, [enum_class._VALUES_TO_NAMES.get(v) for v in value])
        else:
            return (enum_class, enum_class._VALUES_TO_NAMES.get(value))

    def _new_child(self, child: Dict[str, Any]) -> None:
        self.logger.debug("Starting child for %s", self._current["name"])
        self.logger.debug("Push: %s", child)
        self._parents.append(self._current)
        self._current = child

    def _finish_child(self) -> None:
        self.logger.debug("Pop: %s", self._current)
        parent = self._parents.pop()
        parent["value"].append(self._current)
        self.logger.debug("Finished child for %s", parent["name"])
        self._current = parent


class OffsetRecordingCompactProtocol(
    OffsetRecordingProtocol, TCompactProtocol.TCompactProtocol
):
    """Compact protocol with offset recording support."""


class TFileTransport(TTransportBase):
    """A Thrift transport that reads from a file handle at specific offsets."""

    def __init__(self, file_handle, start_offset: Optional[int] = None):
        self._file = file_handle
        self._start_offset = start_offset or file_handle.tell()
        self._current_offset = self._start_offset

    def read(self, sz: int):
        self._file.seek(self._current_offset)
        data = self._file.read(sz)
        self._current_offset += len(data)
        return data

    def write(self, buf):
        raise NotImplementedError("TFileTransport is read-only")

    def flush(self):
        pass

    def close(self):
        pass

    def isOpen(self) -> bool:
        return not self._file.closed

    def tell(self) -> int:
        """Return current position relative to start offset."""

        return self._current_offset - self._start_offset

    def seek(self, offset: int, whence: int = 0) -> None:
        """Seek relative to start offset."""

        if whence == 0:  # absolute
            self._current_offset = self._start_offset + offset
        elif whence == 1:  # relative
            self._current_offset += offset
        elif whence == 2:  # from end - not supported
            raise NotImplementedError("Seek from end not supported")


def create_segment(
    range_start: int,
    range_end: int,
    name: str,
    value: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    segment: Dict[str, Any] = {}
    segment["offset"] = range_start
    segment["length"] = range_end - range_start
    segment["name"] = name
    segment["value"] = value
    if metadata:
        segment["metadata"] = metadata
    return segment


def create_segment_from_offset_info(info: Any, base_offset: int):
    if not isinstance(info, dict):
        return info
    if info["type"] in ("struct", "list"):
        value = []
        for value_info in info["value"]:
            value.append(create_segment_from_offset_info(value_info, base_offset))
    else:
        value = info["value"]
    metadata: Dict[str, Any] = {}
    metadata["type"] = info["type"]
    if info["type_class"]:
        metadata["type_class"] = info["type_class"].__name__
    if "enum_type" in info:
        metadata["enum_type"] = info["enum_type"]
        metadata["enum_name"] = info["enum_name"]
    return create_segment(
        base_offset + info["range_from"],
        base_offset + info["range_to"],
        info["name"],
        value,
        metadata,
    )


def read_thrift_segment(f, offset: int, name: str, thrift_class):
    f.seek(offset)
    protocol = OffsetRecordingCompactProtocol(
        TFileTransport(f),
        name,
        struct_class=thrift_class,
    )
    obj = thrift_class()
    obj.read(protocol)
    segment = create_segment_from_offset_info(
        protocol.get_offset_info(), base_offset=offset
    )
    return obj, segment


def read_pages(f, column_chunk, segments: List[Dict[str, Any]]):
    remaining_values = column_chunk.meta_data.num_values
    offset = column_chunk.meta_data.data_page_offset
    offsets: List[int] = []
    while remaining_values > 0:
        page, page_segment = read_thrift_segment(f, offset, "page", PageHeader)
        page_header_end = page_segment["offset"] + page_segment["length"]
        offsets.append(page_segment["offset"])
        segments.append(page_segment)
        segments.append(
            create_segment(
                page_header_end,
                page_header_end + page.compressed_page_size,
                "page_data",
            )
        )
        if page.data_page_header is not None:
            num_values = page.data_page_header.num_values
        elif page.data_page_header_v2 is not None:
            num_values = page.data_page_header_v2.num_values
        else:
            break
        remaining_values -= num_values
        offset = page_header_end + page.compressed_page_size
    return offsets


def read_dictionary_page(f, column_chunk, segments: List[Dict[str, Any]]):
    dict_page, dict_page_segment = read_thrift_segment(
        f,
        column_chunk.meta_data.dictionary_page_offset,
        "page",
        PageHeader,
    )
    segments.append(dict_page_segment)
    segments.append(
        create_segment(
            dict_page_segment["offset"] + dict_page_segment["length"],
            dict_page_segment["offset"]
            + dict_page_segment["length"]
            + dict_page.compressed_page_size,
            "page_data",
        )
    )
    return dict_page_segment["offset"]


def read_column_index(f, column_chunk, segments: List[Dict[str, Any]]):
    _, column_index_segment = read_thrift_segment(
        f, column_chunk.column_index_offset, "column_index", ColumnIndex
    )
    segments.append(column_index_segment)
    return column_index_segment["offset"]


def read_offset_index(f, column_chunk, segments: List[Dict[str, Any]]):
    _, offset_index_segment = read_thrift_segment(
        f, column_chunk.offset_index_offset, "offset_index", OffsetIndex
    )
    segments.append(offset_index_segment)
    return offset_index_segment["offset"]


def read_bloom_filter(f, column_chunk, segments: List[Dict[str, Any]]):
    _, bloom_filter_segment = read_thrift_segment(
        f, column_chunk.bloom_filter_offset, "bloom_filter", BloomFilterHeader
    )
    segments.append(bloom_filter_segment)
    return bloom_filter_segment["offset"]


def fill_gaps(segments: List[Dict[str, Any]], file_size: int):
    offset = 0
    new_segments = []
    for s in segments:
        if s["offset"] != offset:
            new_segments.append(create_segment(offset, s["offset"], "unknown"))
        new_segments.append(s)
        offset = s["offset"] + s["length"]
    if offset != file_size:
        new_segments.append(create_segment(offset, file_size, "unknown"))
    return new_segments


def parse_parquet_file(file_path: str):
    segments: List[Dict[str, Any]] = []

    with open(file_path, "rb") as f:
        # Read file header
        f.seek(0)
        header = f.read(4)
        if header != b"PAR1":
            raise ValueError("Not a valid Parquet file - missing PAR1 header")
        segments.append(create_segment(0, 4, "magic_number", "PAR1"))

        # Read footer length (last 8 bytes)
        f.seek(-8, 2)
        footer_size = f.read(4)
        footer_magic = f.read(4)
        file_size = f.tell()
        if footer_magic != b"PAR1":
            raise ValueError("Not a valid Parquet file - missing PAR1 footer")
        footer_size = struct.unpack("<I", footer_size)[0]
        segments.append(
            create_segment(file_size - 4, file_size, "magic_number", "PAR1")
        )
        segments.append(
            create_segment(file_size - 8, file_size - 4, "footer_length", footer_size)
        )

        # Parse footer with offset recording
        footer_offset = file_size - 8 - footer_size
        footer, footer_segment = read_thrift_segment(
            f, footer_offset, "footer", FileMetaData
        )
        segments.append(footer_segment)

        column_chunk_data_offsets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}

        for row_group in footer.row_groups:
            for column_chunk in row_group.columns:
                column_key = tuple(column_chunk.meta_data.path_in_schema)
                offset_list = column_chunk_data_offsets.setdefault(column_key, [])

                offsets: Dict[str, Any] = {}
                offsets["data_pages"] = read_pages(f, column_chunk, segments)

                if column_chunk.meta_data.dictionary_page_offset is not None:
                    offsets["dictionary_page"] = read_dictionary_page(
                        f, column_chunk, segments
                    )

                if column_chunk.column_index_offset is not None:
                    offsets["column_index"] = read_column_index(
                        f, column_chunk, segments
                    )

                if column_chunk.offset_index_offset is not None:
                    offsets["offset_index"] = read_offset_index(
                        f, column_chunk, segments
                    )

                if column_chunk.meta_data.bloom_filter_offset is not None:
                    offsets["bloom_filter"] = read_bloom_filter(
                        f, column_chunk, segments
                    )

                offset_list.append(offsets)

    segments.sort(key=lambda s: s["offset"])
    segments = fill_gaps(segments, file_size)
    return segments, column_chunk_data_offsets


def segment_to_json(segment):
    if isinstance(segment, dict):
        metadata = segment.get("metadata", {})
        if metadata.get("type") == "struct":
            return {v["name"]: segment_to_json(v) for v in segment["value"]}
        if metadata.get("type") == "list":
            if metadata.get("enum_type") is not None:
                return metadata["enum_name"]
            else:
                return [segment_to_json(v) for v in segment["value"]]
        if metadata.get("enum_type") is not None:
            return segment["metadata"]["enum_name"]
        return segment_to_json(segment["value"])
    return segment


def find_footer_segment(segments: Iterable[Dict[str, Any]]):
    for s in segments:
        if s["name"] == "footer":
            return s
    return None


def get_summary(footer, segments):
    summary: Dict[str, Any] = {}
    summary["num_rows"] = footer["num_rows"]
    summary["num_row_groups"] = len(footer["row_groups"])
    if footer["row_groups"]:
        summary["num_columns"] = len(footer["row_groups"][0]["columns"])

    num_pages = 0
    num_data_pages = 0
    num_v1_data_pages = 0
    num_v2_data_pages = 0
    num_dict_pages = 0
    page_header_size = 0
    uncompressed_page_data_size = 0
    compressed_page_data_size = 0
    for s in segments:
        if s["name"] == "page":
            num_pages += 1
            page_header_size += s["length"]
            page_json = segment_to_json(s)
            if page_json["type"] in ("DATA_PAGE", "DATA_PAGE_V2"):
                num_data_pages += 1
            elif page_json["type"] == "DICTIONARY_PAGE":
                num_dict_pages += 1
            if "data_page_header" in page_json:
                num_v1_data_pages += 1
            if "data_page_header_v2" in page_json:
                num_v2_data_pages += 1
            uncompressed_page_data_size += page_json["uncompressed_page_size"]
            compressed_page_data_size += page_json["compressed_page_size"]

    summary["num_pages"] = num_pages
    summary["num_data_pages"] = num_data_pages
    summary["num_v1_data_pages"] = num_v1_data_pages
    summary["num_v2_data_pages"] = num_v2_data_pages
    summary["num_dict_pages"] = num_dict_pages

    # Sum of page header sizes for all pages in the file
    summary["page_header_size"] = page_header_size
    summary["uncompressed_page_data_size"] = uncompressed_page_data_size
    summary["compressed_page_data_size"] = compressed_page_data_size

    uncompressed_page_size = 0
    compressed_page_size = 0
    column_index_size = 0
    offset_index_size = 0
    bloom_fitler_size = 0
    for row_group in footer["row_groups"]:
        for column in row_group["columns"]:
            uncompressed_page_size += column["meta_data"]["total_uncompressed_size"]
            compressed_page_size += column["meta_data"]["total_compressed_size"]
            column_index_size += column.get("column_index_length", 0)
            offset_index_size += column.get("offset_index_length", 0)
            bloom_fitler_size += column.get("bloom_filter_length", 0)

    # These page sizes include header size
    summary["uncompressed_page_size"] = uncompressed_page_size
    summary["compressed_page_size"] = compressed_page_size

    summary["column_index_size"] = column_index_size
    summary["offset_index_size"] = offset_index_size
    summary["bloom_fitler_size"] = bloom_fitler_size

    footer_segment = find_footer_segment(segments)
    if footer_segment is not None:
        summary["footer_size"] = footer_segment["length"]
    summary["file_size"] = segments[-1]["offset"] + segments[-1]["length"]

    return summary


def get_pages(segments, column_chunk_data_offsets):
    page_offset_map: Dict[int, Any] = {}
    for s in segments:
        if s["name"] in ("page", "column_index", "offset_index", "bloom_filter"):
            page_offset_map[s["offset"]] = segment_to_json(s)
    column_pages = []

    def with_offset(offset: int):
        obj = {"$offset": offset}
        obj.update(page_offset_map[offset])
        return obj

    for col_idx, (column_path, offsets) in enumerate(column_chunk_data_offsets.items()):
        pages = {"$index": col_idx, "column": column_path}
        row_groups = []
        for row_group_idx, offset_info in enumerate(offsets):
            row_group = {"$index": row_group_idx}
            if offset_info.get("dictionary_page"):
                row_group["dictionary_page"] = with_offset(
                    offset_info["dictionary_page"]
                )
            if offset_info.get("data_pages"):
                data_pages = []
                for offset in offset_info["data_pages"]:
                    data_pages.append(with_offset(offset))
                row_group["data_pages"] = data_pages
            if offset_info.get("column_index"):
                row_group["column_index"] = with_offset(offset_info["column_index"])
            if offset_info.get("offset_index"):
                row_group["offset_index"] = with_offset(offset_info["offset_index"])
            if offset_info.get("bloom_filter"):
                row_group["bloom_filter"] = with_offset(offset_info["bloom_filter"])
            row_groups.append(row_group)
        pages["row_groups"] = row_groups
        column_pages.append(pages)
    return column_pages


def json_encode(x, truncate_length: int = 32):
    if isinstance(x, bytes):
        j = {
            "type": "binary",
            "length": len(x),
        }
        if len(x) < truncate_length:
            j["value"] = list(x)
        else:
            j["value_truncated"] = list(x[:truncate_length])
        return j
    raise ValueError(f"cannot encode for json: {type(x)}")
