from __future__ import annotations

from ._core import (
    OffsetRecordingCompactProtocol,
    OffsetRecordingProtocol,
    TFileTransport,
    fill_gaps,
    json_encode,
    segment_to_json,
)
from .decoders import (
    DecodeStats,
    decode_levels,
    decode_plain,
    decode_rle_bitpacked_hybrid,
    decode_v1_level_block,
    decompress,
)
from .parquet_file import (
    ColumnChunk,
    DecodedPage,
    MissingDictionaryError,
    Page,
    PageDecodeError,
    ParquetFile,
    RowGroup,
    UnsupportedCodecError,
    UnsupportedEncodingError,
    UnsupportedPageTypeError,
)

__all__ = [
    "ColumnChunk",
    "DecodeStats",
    "DecodedPage",
    "MissingDictionaryError",
    "OffsetRecordingCompactProtocol",
    "OffsetRecordingProtocol",
    "Page",
    "PageDecodeError",
    "ParquetFile",
    "RowGroup",
    "TFileTransport",
    "UnsupportedCodecError",
    "UnsupportedEncodingError",
    "UnsupportedPageTypeError",
    "decode_levels",
    "decode_plain",
    "decode_rle_bitpacked_hybrid",
    "decode_v1_level_block",
    "decompress",
    "fill_gaps",
    "json_encode",
    "segment_to_json",
]

__version__ = "0.4.0.dev0"
