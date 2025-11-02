# Parquet Analyzer

A Python tool for deep inspection and analysis of Apache Parquet files, providing detailed insights into file structure, metadata, and binary layout.

## Features

- **File Structure Analysis**: Parse and visualize the complete binary structure of Parquet files
- **Metadata Inspection**: Extract and display schema, row group, and column metadata
- **Page-Level Details**: Analyze data pages, dictionary pages, and their headers
- **Offset Tracking**: Show exact byte offsets and lengths of all file components
- **Statistics Summary**: Generate comprehensive file statistics and size breakdowns
- **Thrift Protocol Support**: Deep dive into Thrift-encoded metadata structures

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

### Basic Usage

```bash
# Analyze a Parquet file and get summary information
parquet-analyzer example.parquet

# Show detailed offset and Thrift structure information
parquet-analyzer -s example.parquet

# Enable debug logging
parquet-analyzer --log-level DEBUG example.parquet

# Run via python -m if the console script is unavailable
python -m parquet_analyzer example.parquet
```

### Command Line Options

- `parquet_file`: Path to the Parquet file to analyze (required)
- `-s, --show-offsets-and-thrift-details`: Show detailed byte offsets and Thrift structure information
- `--log-level LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Output Formats

### Standard Output (Default)

The default output provides a structured JSON view with three main sections:

#### 1. Summary Statistics
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

#### 2. Footer Metadata
Complete Parquet file metadata including:
- Schema definition with column types and repetition levels
- Row group information
- Column chunk metadata
- Encoding and compression details

#### 3. Page Information
Detailed breakdown of all pages organized by column and row group:
- Data pages with encoding and statistics
- Dictionary pages
- Column indexes
- Offset indexes
- Bloom filters

### Detailed Output (`-s` flag)

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

## Understanding the Output

### File Structure Components

- **Magic Numbers**: PAR1 headers at file start and end
- **Page Headers**: Thrift-encoded metadata for each data/dictionary page
- **Page Data**: Compressed/uncompressed column data
- **Column Indexes**: Statistics for data pages (optional)
- **Offset Indexes**: Byte offsets for data pages (optional)
- **Bloom Filters**: Bloom filter data for columns (optional)
- **Footer**: File metadata including schema and row group information
- **Footer Length**: 4-byte little-endian footer size

### Statistics Explained

- `num_rows`: Total number of rows across all row groups
- `num_row_groups`: Number of row groups in the file
- `num_columns`: Number of columns in the schema
- `num_pages`: Total pages (data + dictionary)
- `num_v1_data_pages`: Data pages using format v1
- `num_v2_data_pages`: Data pages using format v2
- `page_header_size`: Total bytes used by page headers
- `compressed_page_size`: Total compressed data size
- `uncompressed_page_size`: Total uncompressed data size

## Technical Details

### Architecture

The tool uses a custom Thrift protocol implementation (`OffsetRecordingProtocol`) that wraps the standard Thrift compact protocol to track byte offsets and lengths of all decoded structures. This enables precise mapping of logical Parquet structures to their binary representation.

### Key Components

- **OffsetRecordingProtocol**: Tracks byte positions during Thrift deserialization
- **TFileTransport**: File-based transport supporting seeking and offset tracking
- **Segment Creation**: Converts offset information into structured output
- **Gap Filling**: Identifies unknown or unaccounted byte ranges

### Supported Parquet Features

- All Parquet data types (primitive and logical)
- Compression codecs
- Encoding types
- Page formats (v1 and v2)
- Column indexes and offset indexes
- Bloom filters
- Nested schemas

## Use Cases

### Performance Analysis
- Identify compression efficiency across columns
- Analyze page sizes and distribution
- Understand storage overhead from metadata

### File Debugging
- Locate corrupted segments
- Verify file format compliance
- Analyze encoding choices

### Schema Evolution
- Compare file structures across versions
- Understand metadata changes
- Analyze backward compatibility

### Storage Optimization
- Identify opportunities for better compression
- Analyze row group sizing
- Optimize column ordering

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

Released under the [MIT License](LICENSE).

## Related Projects

- [Apache Parquet](https://parquet.apache.org/) - The Apache Parquet file format
- [parquet-python](https://github.com/dask/fastparquet) - Python Parquet libraries
- [parquet-tools](https://github.com/apache/parquet-mr/tree/master/parquet-tools) - Official Parquet command-line tools
