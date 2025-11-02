# Parquet Analyzer

A Python tool for deep inspection and analysis of Apache Parquet files, providing detailed insights into file structure, metadata, and binary layout.

## Installation

```bash
pip install parquet-analyzer
```

To work from a local clone instead, install in editable mode:

```bash
pip install -e .
```

### Requirements

- Python 3.11+
- thrift>=0.16 (installed automatically)

## Usage

### Basic usage

```bash
# Analyze a Parquet file and get summary information
parquet-analyzer example.parquet

# Divide the file into segments and show detailed offset and Thrift structure information for each segment
parquet-analyzer -s example.parquet

# Enable debug logging
parquet-analyzer --log-level DEBUG example.parquet

# Run via python -m if the console script is unavailable
python -m parquet_analyzer example.parquet
```

## Output Formats

### Standard output (default)

The default output provides a structured JSON view with three main sections:

#### Summary statistics
```json
{
  "summary": {
    "num_rows": 10,
    "num_row_groups": 1,
    "num_columns": 2,
    "num_pages": 2,
    "num_data_pages": 2,
    "num_v1_data_pages": 2,
    "num_v2_data_pages": 0,
    "num_dict_pages": 0,
    "page_header_size": 47,
    "uncompressed_page_data_size": 130,
    "compressed_page_data_size": 96,
    "uncompressed_page_size": 177,
    "compressed_page_size": 143,
    "column_index_size": 48,
    "offset_index_size": 23,
    "bloom_fitler_size": 0,
    "footer_size": 527,
    "file_size": 753
  }
}
```

#### Footer metadata
Complete Parquet file metadata including:
- Schema definition with column types and repetition levels
- Row group information
- Column chunk metadata
- Encoding and compression details

#### Page information
Detailed breakdown of all pages organized by column:
- Data pages with encoding and statistics
- Dictionary pages
- Column indexes
- Offset indexes
- Bloom filters

### Detailed output (`-s` flag)

When using the `-s` flag, the tool outputs a detailed segment-by-segment breakdown showing:

```json
[
  {
    "offset": 0,
    "length": 4,
    "name": "magic_number",
    "value": "PAR1"
  },
  {
    "offset": 4,
    "length": 24,
    "name": "page",
    "value": [
      {
        "offset": 5,
        "length": 1,
        "name": "type",
        "value": 0,
        "metadata": {
          "type": "i32",
          "enum_type": "PageType",
          "enum_name": "DATA_PAGE"
        }
      }
    ]
  }
]
```

This mode is useful for:
- Debugging Parquet file corruption
- Understanding exact binary layout
- Analyzing file format compliance
- Optimizing file structure

## Technical details

The tool uses a custom Thrift protocol implementation (`OffsetRecordingProtocol`) that wraps the standard Thrift compact protocol to track byte offsets and lengths of all decoded structures. This enables precise mapping of logical Parquet structures to their binary representation.

## Development

### Environment setup

```bash
pip install -e .[dev]
hatch run dev:lint
hatch run dev:test
hatch run dev:test-cov
# Or run everything at once
hatch run dev:check
```

The development extra pulls in tooling (`hatch`, `ruff`, `pytest`) and `pyarrow` so tests can generate Parquet fixtures on the fly.

### Regenerating Thrift bindings

The Python modules in `src/parquet` are generated from `parquet.thrift`.

1. Install the Apache Thrift compiler (`brew install thrift` on macOS, or download a release from the [Apache Thrift](https://thrift.apache.org/) project).
2. From the repository root, regenerate everything in one step:

   ```bash
   hatch run dev:update-thrift
   ```

   This refreshes `parquet.thrift`, runs the compiler, and removes any stray `src/__init__.py` the compiler may create.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

Released under the [MIT License](LICENSE).