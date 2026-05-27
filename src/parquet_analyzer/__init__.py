from __future__ import annotations

from ._core import (
    OffsetRecordingCompactProtocol,
    OffsetRecordingProtocol,
    TFileTransport,
    fill_gaps,
    find_footer_segment,
    get_pages,
    get_summary,
    json_encode,
    parse_parquet_file,
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

__all__ = [
    "DecodeStats",
    "OffsetRecordingCompactProtocol",
    "OffsetRecordingProtocol",
    "TFileTransport",
    "decode_levels",
    "decode_plain",
    "decode_rle_bitpacked_hybrid",
    "decode_v1_level_block",
    "decompress",
    "fill_gaps",
    "find_footer_segment",
    "get_pages",
    "get_summary",
    "json_encode",
    "parse_parquet_file",
    "segment_to_json",
]

__version__ = "0.4.0.dev0"
