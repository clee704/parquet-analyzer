# Parquet Analyzer

The parquet inspection toolkit — a CLI and Python library for AI agents and
humans who need to verify encoding-level behavior: what codec a column
actually got, where on disk its pages live, whether dictionary encoding was
applied, what the RLE runs inside a page look like.

```bash
$ parquet-analyzer file summary data.parquet
$ parquet-analyzer column show data.parquet --column Sex
$ parquet-analyzer page decode data.parquet row_groups/0/columns/4/pages/1 --limit 3
```

Every subcommand emits a single JSON object on stdout, carries a
`$schema` field for machine validation, and exits non-zero only on
operational errors — designed from the start to be driven by both
humans and AI agents without extra parsing.

## Installation

```bash
pip install parquet-analyzer
```

Requires Python 3.11+.

## Quickstart

```bash
pip install parquet-analyzer
parquet-analyzer file summary data.parquet
```

```json
{
  "$schema": "parquet-analyzer/v1/file-summary",
  "num_rows": 891,
  "num_row_groups": 1,
  "num_columns": 12,
  "uncompressed_page_size": 90429,
  "compressed_page_size": 38839,
  "column_index_size": 0,
  "offset_index_size": 0,
  "bloom_filter_size": 0,
  "footer_size": 1162,
  "file_size": 40013,
  "footer_offset": 38843
}
```

From there: [`file schema`](#file--file-level-inspection) for the
column list, [`column show`](#column--column-chunk-inspection) to
drill into one column, [`page decode`](#page--page-level-inspection)
to see the on-disk encoding structure of a page.

## What you can do

### 1. Audit encoding and compression choices

Did your writer actually apply dictionary encoding to the low-cardinality
columns? What codec ended up on each chunk?

```bash
parquet-analyzer column list data.parquet
```

Returns one entry per `(row_group, column_chunk)`, each carrying
`encodings`, `codec`, `compressed_size`, `uncompressed_size`, and
`chunk_offset`/`chunk_length`. Pipe through `jq` to filter or sort:

```bash
# Find the largest column chunk by compressed size
parquet-analyzer column list data.parquet \
  | jq '.items | max_by(.compressed_size) | {column, encodings, compressed_size}'
```

Use [`column show`](#column--column-chunk-inspection) for an aggregated
view across row groups for one named column.

→ See [column](#column--column-chunk-inspection)

### 2. Locate exact byte ranges

Every chunk output includes `chunk_offset`/`chunk_length` and, when
present, the byte ranges of its OffsetIndex, ColumnIndex, and
BloomFilter. These are seek-and-read pairs: `read(chunk_offset,
chunk_length)` gives you the entire compressed chunk without
re-parsing the footer.

```bash
parquet-analyzer column show data.parquet --column Sex
# → chunk_offset: 24256, chunk_length: 501
# → dictionary_page_offset: 24256, data_page_offset: 24276
```

Use [`page list`](#page--page-level-inspection) to get per-page byte
ranges (from an OffsetIndex when the writer emitted one, or via a
page-header walk with `--walk-pages`).

→ See [column](#column--column-chunk-inspection), [page](#page--page-level-inspection)

### 3. Navigate the file structure top-down

Explore the file like a directory tree — one bounded step at a time —
without committing to a full scan. Each step returns the current node
plus its immediate children as stubs, each with a `_path` to feed into
the next call:

```bash
parquet-analyzer show data.parquet                          # file root: row groups
parquet-analyzer show data.parquet row_groups/0             # row group: column stubs
parquet-analyzer show data.parquet row_groups/0/columns/4   # column metadata + page stubs
```

Especially useful for AI agents: the output is bounded by construction
and every child stub carries a self-describing `_path`.

→ See [show](#show--path-addressed-navigation)

### 4. Decode a page body to its on-disk encoding structure

The `page decode` verb decodes a page body faithfully — not as a
flattened logical reconstruction but as the encoding scheme actually
stores it: RLE/bit-packed runs for dictionary-encoded columns, PLAIN
values for PLAIN columns, level streams for nullable or nested columns.

```bash
# List the pages of a column chunk
parquet-analyzer page list data.parquet --column Sex

# Decode a specific page (use the _path from page list)
parquet-analyzer page decode data.parquet row_groups/0/columns/4/pages/1 --limit 3
```

`--limit N` bounds each stream to N rows — useful for large pages with
millions of values. Use `--kind rle-runs` to see just the dictionary
index runs, `--kind values` to see resolved physical values, `--kind
levels` for the definition/repetition level streams.

→ See [page](#page--page-level-inspection)

### 5. Validate file structure

`file validate` runs footer-only structural checks without touching
page bodies: magic-number parse, row-count consistency, column-count
match, and non-zero compressed sizes for non-empty chunks. Parse
failures are reported as findings (`valid: false`, exit 0) rather
than CLI errors, so the command stays useful for partially-written files.

```bash
parquet-analyzer file validate data.parquet
```

→ See [file](#file--file-level-inspection)

### 6. Use it as a Python library

`ParquetFile` is a first-class Python API. Open a file lazily (footer-only
on construction), walk row groups and column chunks, and decode a page body
in your own code — no subprocess, no JSON parsing:

```python
from parquet_analyzer import ParquetFile

with ParquetFile("titanic.parquet") as pf:
    print(pf.num_rows, pf.num_row_groups, pf.num_columns)
    # 891 1 12

    cc = pf.row_groups[0].columns[4]   # Sex column
    print(cc.path, cc.type, cc.encodings)
    # ('Sex',) BYTE_ARRAY ('PLAIN', 'RLE_DICTIONARY')

    page = cc.page(1)                  # data page (index 1; 0 is the dict page)
    decoded = page.decode()
    print(decoded.definition_levels.bit_width, decoded.values.bit_width)
    # 1 2

    print(decoded.values.runs[:3])     # dictionary index RLE runs, on-disk structure
    # (RleRun(value=0, length=1), RleRun(value=1, length=3), RleRun(value=0, length=4))

    print(page.physical_values()[:5])  # dict indices resolved to physical-type values
    # [b'male', b'female', b'female', b'female', b'male']
```

→ See [Library API](#library-api)

## Compared to

| Tool | Strength | Parquet Analyzer adds |
|------|----------|-----------------------|
| **parquet-tools** (Apache/Java) | `meta`, `schema`, `head`, `dump` — quick human-readable inspection | JSON output; page-body decode; RLE run inspection; byte-range pairs; no Java required |
| **pqrs** (Rust) | Fast row-data extraction to CSV/JSON | Format-structure focus (encodings, page layout, byte ranges) rather than row data |
| **pyarrow.parquet** (Python library) | `read_metadata()`, `read_schema()` — scriptable; part of the Arrow ecosystem | CLI surface; path-addressed navigation; page-body decode without writing glue code |
| **DuckDB `parquet_metadata()`** | SQL-based metadata exploration; integrates with the query engine | Page-body inspection; RLE/encoding-faithful decode; `$schema`-versioned JSON; no SQL engine required |

All four are good at what they do. Reach for parquet-analyzer when you
need to inspect the format structure (encodings, page layout, byte
ranges, on-disk encoding streams) rather than the row data, and when
you need JSON output that's directly consumable by a script or AI agent.

## CLI Reference

`parquet-analyzer` ships two complementary CLI surfaces:

- **Verb-noun subcommands** (`parquet-analyzer file summary ...`,
  `column show ...`, etc.) — small, composable, AI-friendly. One
  subcommand → one JSON object. Footer-only and fast by default.
- **Legacy whole-file modes** (`parquet-analyzer <path> [--output-mode ...]`)
  — the original "show me everything" surface for interactive exploration
  and the interactive HTML report.

### Verb-noun subcommands

The verb-noun surface answers one question per invocation and is built for
chaining with `jq` and consumption by AI agents. The eight verb-noun
subcommands under `file`/`rowgroup`/`column` are **footer-only** — they do not
walk page headers or read page bodies. A ninth verb,
[`show`](#show--path-addressed-navigation), offers path-addressed tree
navigation (also bounded — it never walks page headers unless you pass
`--walk-pages`). The [`page`](#page--page-level-inspection) verb is the escape
hatch into the page layer: it lists a chunk's pages, dumps a single page
header, extracts raw body bytes, and decodes a page body into its
encoding-faithful structure. Reading page bodies is opt-in by construction —
you pay the per-page cost only for the page you name.

#### Conventions

- **One subcommand → one JSON object on stdout.** List-shaped outputs wrap in `{items, total, returned, truncated}`.
- **Every output object carries a `$schema` field** of the form `parquet-analyzer/v1/<command>` so downstream tools can validate the shape.
- **`--schema-version`** on any subcommand short-circuits to print just the schema URI — no file is opened.
- **`--limit N`** caps list-shaped outputs. Truncation is reported explicitly via the `truncated` field; nothing is silently dropped.
- **`-o / --output PATH`** writes JSON to a file instead of stdout.
- **`--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`** is the top-level global option.
- **Navigation cross-link.** Every `rowgroup`/`column` `list` and `show` item carries a `_path` — its canonical [`show`](#show--path-addressed-navigation) navigation path (`row_groups/2/columns/1`). Use the flat commands as a search/sort surface, then feed a `_path` straight back into `show` to drill in (no need to hand-build the path):
  ```bash
  parquet-analyzer column list data.parquet | jq -r '.items | max_by(.compressed_size)._path' \
    | xargs parquet-analyzer show data.parquet
  ```
- **Errors → stderr** as a single JSON line: `{"$schema": "parquet-analyzer/v1/error", "error": "<code>", "message": "<human>", "fix": "<retry command>"}`. The process exits non-zero on operational errors; `file validate` reports parse failures as findings (exit 0).
- **Column path matching.** Every emitted column carries both `column` (dot-joined display string) and `path` (list of segments). `--column NAME` matches against the dot-joined form. Ambiguous matches (when multiple distinct paths join to the same string) produce a JSON error listing the candidate paths.

> **Design contract for what goes in the output:** see
> [`docs/output-principles.md`](docs/output-principles.md). The TL;DR is
> "footer-bounded and walk-free": every emitted field is derivable from
> the parsed footer plus, at most, one extra parse of an index thrift
> the writer already wrote (OffsetIndex, ColumnIndex, BloomFilter
> header). Anything that would require walking per-page thrift headers
> or reading page bodies is handled by the
> [`page`](#page--page-level-inspection) subcommand surface — the deliberate,
> per-page opt-in into header walks and body reads.

#### `file` — file-level inspection

```bash
$ parquet-analyzer file summary titanic.parquet
{
  "$schema": "parquet-analyzer/v1/file-summary",
  "num_rows": 891,
  "num_row_groups": 1,
  "num_columns": 12,
  "uncompressed_page_size": 90429,
  "compressed_page_size": 38839,
  "column_index_size": 0,
  "offset_index_size": 0,
  "bloom_filter_size": 0,
  "footer_size": 1162,
  "file_size": 40013,
  "footer_offset": 38843
}

$ parquet-analyzer file kv titanic.parquet
{
  "$schema": "parquet-analyzer/v1/file-kv",
  "items": [],
  "returned": 0,
  "total": 0,
  "truncated": false
}

$ parquet-analyzer file kv pyarrow-file.parquet --key ARROW:schema  # filter by key
{
  "$schema": "parquet-analyzer/v1/file-kv",
  "items": [
    {
      "key": "ARROW:schema",
      "value": "/////6gAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAIAAABAAAAABAAAANj///8AAAEFEAAAABgAAAAEAAAAAAAAAAEAAAB5AAAABAAEAAQAAAAQABQACAAGAAcADAAAABAAEAAAAAAAAQIQAAAAHAAAAAQAAAAAAAAAAQAAAHgAAAAIAAwACAAHAAgAAAAAAAABQAAAAAAAAAA="
    }
  ],
  "returned": 1,
  "total": 1,
  "truncated": false,
  "filter_key": "ARROW:schema"
}

$ parquet-analyzer file schema titanic.parquet | jq '.elements[:4]'
[
  {"repetition_type": "REQUIRED", "name": "duckdb_schema", "num_children": 12},
  {"type": "INT64", "repetition_type": "OPTIONAL", "name": "PassengerId", "converted_type": "INT_64"},
  {"type": "INT64", "repetition_type": "OPTIONAL", "name": "Survived", "converted_type": "INT_64"},
  {"type": "INT64", "repetition_type": "OPTIONAL", "name": "Pclass", "converted_type": "INT_64"}
]

$ parquet-analyzer file validate titanic.parquet
{
  "$schema": "parquet-analyzer/v1/file-validate",
  "path": "titanic.parquet",
  "valid": true,
  "errors": []
}
```

`file kv` preserves duplicate keys and original ordering (parquet's spec
allows duplicate KV entries). `file validate` runs footer-only structural
checks: magic-number parse, row-count consistency between file and row
groups, per-row-group column count matches schema, and column chunks with
rows have non-zero compressed size. Parse failures (bad magic, unparseable
footer) are reported as findings with `valid: false` rather than as CLI
errors. Deep validation (byte-range overlap detection, page-count vs.
column metadata) requires an eager walk and will land in a future release.

#### `rowgroup` — row-group inspection

```bash
$ parquet-analyzer rowgroup list titanic.parquet
{
  "$schema": "parquet-analyzer/v1/rowgroup-list",
  "items": [
    {
      "row_group": 0,
      "_path": "row_groups/0",
      "num_rows": 891,
      "file_offset": 4,
      "total_byte_size": 306419,
      "total_compressed_size": 38839,
      "num_columns": 12
    }
  ],
  "returned": 1, "total": 1, "truncated": false
}

$ parquet-analyzer rowgroup show titanic.parquet --row-group 0 | jq '{row_group, num_rows, total_compressed_size, "first_column": .columns[0]}'
{
  "row_group": 0,
  "num_rows": 891,
  "total_compressed_size": 38839,
  "first_column": {
    "row_group": 0,
    "_path": "row_groups/0/columns/0",
    "column": "PassengerId",
    "path": ["PassengerId"],
    "type": "INT64",
    "encodings": ["PLAIN"],
    "codec": "SNAPPY",
    "num_values": 891,
    "compressed_size": 4357,
    "uncompressed_size": 7155,
    "chunk_offset": 4,
    "chunk_length": 4357,
    "data_page_offset": 4,
    "dictionary_page_offset": null,
    "has_dictionary": false,
    "has_offset_index": false,
    "offset_index_offset": null,
    "offset_index_length": null,
    "has_column_index": false,
    "column_index_offset": null,
    "column_index_length": null,
    "has_bloom_filter": false,
    "bloom_filter_offset": null,
    "bloom_filter_length": null,
    "num_pages": null,
    "num_pages_known": false,
    "num_pages_hint": "no OffsetIndex; re-run with --walk-pages to count pages (reads page headers)",
    "statistics": {
      "max": {"type": "binary", "length": 8, "value": [123, 3, 0, 0, 0, 0, 0, 0]},
      "min": {"type": "binary", "length": 8, "value": [1, 0, 0, 0, 0, 0, 0, 0]},
      "null_count": 0,
      "max_value": {"type": "binary", "length": 8, "value": [123, 3, 0, 0, 0, 0, 0, 0]},
      "min_value": {"type": "binary", "length": 8, "value": [1, 0, 0, 0, 0, 0, 0, 0]}
    }
  }
}
```

#### `column` — column-chunk inspection

```bash
# One entry per (row_group, column_chunk); --row-group N to filter
$ parquet-analyzer column list titanic.parquet --limit 3 | jq '.items[] | {column, encodings, codec, compressed_size}'
{"column": "PassengerId", "encodings": ["PLAIN"], "codec": "SNAPPY", "compressed_size": 4357}
{"column": "Survived", "encodings": ["PLAIN"], "codec": "SNAPPY", "compressed_size": 1051}
{"column": "Pclass", "encodings": ["PLAIN"], "codec": "SNAPPY", "compressed_size": 1332}

# Aggregated view across row groups for one named column
$ parquet-analyzer column show titanic.parquet --column Sex
{
  "$schema": "parquet-analyzer/v1/column-show",
  "column": "Sex",
  "path": ["Sex"],
  "type": "BYTE_ARRAY",
  "num_row_groups": 1,
  "total_num_values": 891,
  "total_compressed_size": 501,
  "total_uncompressed_size": 897,
  "row_groups": [
    {
      "row_group": 0,
      "_path": "row_groups/0/columns/4",
      "column": "Sex",
      "path": ["Sex"],
      "type": "BYTE_ARRAY",
      "encodings": ["PLAIN", "RLE_DICTIONARY"],
      "codec": "SNAPPY",
      "num_values": 891,
      "compressed_size": 501,
      "uncompressed_size": 897,
      "chunk_offset": 24256,
      "chunk_length": 501,
      "data_page_offset": 24276,
      "dictionary_page_offset": 24256,
      "has_dictionary": true,
      "has_offset_index": false,
      "offset_index_offset": null,
      "offset_index_length": null,
      "has_column_index": false,
      "column_index_offset": null,
      "column_index_length": null,
      "has_bloom_filter": false,
      "bloom_filter_offset": null,
      "bloom_filter_length": null,
      "num_pages": null,
      "num_pages_known": false,
      "num_pages_hint": "no OffsetIndex; re-run with --walk-pages to count pages (reads page headers)",
      "statistics": {
        "max": {"type": "binary", "length": 4, "value": [109, 97, 108, 101]},
        "min": {"type": "binary", "length": 6, "value": [102, 101, 109, 97, 108, 101]},
        "null_count": 0,
        "distinct_count": 2,
        "max_value": {"type": "binary", "length": 4, "value": [109, 97, 108, 101]},
        "min_value": {"type": "binary", "length": 6, "value": [102, 101, 109, 97, 108, 101]}
      }
    }
  ]
}
```

`num_pages` is reported (`num_pages_known: true`) when the chunk has an
`OffsetIndex` — an O(1) lookup. Without one, counting pages requires walking
page headers, which the footer-bounded default avoids (`num_pages: null`,
`num_pages_known: false`, plus a `num_pages_hint` string pointing at
`--walk-pages` so the output is self-describing). Pass **`--walk-pages`** (on
`column show`, `column list`, or `rowgroup show`) to opt into that walk and
populate `num_pages` anyway — it reads the page headers of every selected
chunk, so `--limit` caps the output but not the walk. SNPW (Spark Native
Parquet Writer) writes OffsetIndex on every chunk; pyarrow does when
`write_page_index=True`; older parquet-mr and many DuckDB files do not.

**Byte-range pairs.** Every per-chunk output carries the seek-and-read
pair `chunk_offset` / `chunk_length`, plus `offset_index_offset` /
`offset_index_length`, `column_index_offset` / `column_index_length`,
and `bloom_filter_offset` / `bloom_filter_length` (each `null` when the
writer didn't emit the corresponding structure). `chunk_offset` is the
dictionary page offset when a dictionary is present, otherwise the data
page offset (parquet's spec: dictionary always precedes data within a
chunk; `total_compressed_size` already includes the dictionary). Together
these let an AI agent issue `read(chunk_offset, chunk_length)` against
the file and get the entire compressed chunk bytes without re-parsing the
footer.

**Raw statistics.** The flat `column`/`rowgroup` verbs surface a chunk's
footer statistics as-is: `min`/`max` and `min_value`/`max_value` are raw
byte objects (`{type, length, value}`), not decoded — the value shown for
`Sex` above is the UTF-8 bytes of `female`/`male`. The `show` verb (below)
decodes min/max to typed scalars (`"female"`, `891`, …) for readability;
the flat verbs stay byte-exact so an agent can verify the on-disk encoding.

#### `show` — path-addressed navigation

`show FILE [PATH]` renders the node at `PATH` plus its immediate children
as stubs, each annotated with the canonical path to descend into it. You
explore the file like a map, one bounded step at a time, by extending the
path along the `row_groups → columns → pages` spine:

```bash
parquet-analyzer show data.parquet                              # file: row groups (stubs)
parquet-analyzer show data.parquet row_groups/0                  # that group: columns (stubs)
parquet-analyzer show data.parquet row_groups/0/columns/3        # column metadata + stats; pages (stubs)
parquet-analyzer show data.parquet row_groups/0/columns/3/pages/5  # that page (direct-seek)
```

Paths are **canonical and index-based**. Each child stub carries a `_path`
to feed back into the next `show`, columns also carry their display `name`,
and a `_navigation` block reports the current `path`, its `parent`, and the
node `kind` — so a caller (human or agent) never has to construct a path by
hand:

```bash
$ parquet-analyzer show titanic.parquet row_groups/0/columns/4
{
  "$schema": "parquet-analyzer/v1/show",
  "_kind": "column_chunk",
  "_location": {"offset": 39420, "length": 70},
  "path": ["Sex"],
  "path_display": "Sex",
  "type": "BYTE_ARRAY",
  "codec": "SNAPPY",
  "encodings": ["PLAIN", "RLE_DICTIONARY"],
  "num_values": 891,
  "compressed_size": 501,
  "uncompressed_size": 897,
  "data_page_offset": 24276,
  "dictionary_page_offset": 24256,
  "file_offset": 0,
  "statistics": {"null_count": 0, "distinct_count": 2, "min_value": "female", "max_value": "male"},
  "dictionary_page": {
    "_kind": "dictionary_page",
    "_location": {"offset": 24256, "length": 20},
    "_lazy": true,
    "_path": "row_groups/0/columns/4/pages/0"
  },
  "pages": null,
  "offset_index": null,
  "column_index": null,
  "bloom_filter": null,
  "_navigation": {
    "path": "row_groups/0/columns/4",
    "parent": "row_groups/0",
    "kind": "column_chunk",
    "children_total": null,
    "children_shown": 0,
    "children_truncated": false,
    "walk_required": true,
    "reason": "no OffsetIndex",
    "hint": "re-run with 'row_groups/0/columns/4/pages/<n> --walk-pages' to address a page (reads every page header)"
  }
}
```

(The `show` envelope is a v1 CLI response; the node stubs inside it carry the
v3 tree-node `_location` address — `{offset, length}`, the file byte range.)

**Listing a column's pages never forces a page-header walk.** With an
OffsetIndex the pages are listed from it (one small read, independent of
page count). Without one, the listing is withheld behind an explicit
`--walk-pages` opt-in: `pages` is `null` and the `_navigation` block carries
a `walk_required` affordance instead of paying an O(pages) walk.

`--walk-pages` enables listing/addressing pages of such a column (it reads
every page header). A column can have many thousands of pages, so the child
listing is capped by **`--limit N`** (default 100; `0` lists all); the
`_navigation` block reports `children_total` / `children_shown` /
`children_truncated`, and truncation only bounds the *listing* — every child
stays addressable by its index. `show` is tree-structured; the verb-noun
subcommands above remain the way to get flat, aggregated views.

#### `page` — page-level inspection

The `page` verb is the escape hatch into the page layer — the one surface that
deliberately reads past the footer. Where the footer-only verbs stop at the
column chunk, `page` walks into the pages of a single chunk to list them, dump
a header, extract raw body bytes, or decode a body into the structure of its
actual on-disk encoding. You pay the page cost only for the page(s) you name.

All four nouns share a page selector: `--column NAME`, `--page-index N` (over
the chunk's full page order — the dictionary page is index `0` when present,
then the data pages; negative indexes count from the end), and `--row-group M`
(required only when the file has more than one row group). The singular verbs
report both `page_index` (full order) and `data_page_index` (position among
data pages only — the `OffsetIndex` correspondence, `null` for the dictionary
page).

Equivalently, any page noun accepts a **navpath** positional — the same
`_path` that `page list` (and `show`) emit — instead of the selectors, so a
`page list` → `page decode` round-trip is copy-paste. For the singular verbs
the navpath addresses a page (`row_groups/0/columns/4/pages/1`); for `page
list` it scopes to a row group (`row_groups/0`) or a column chunk
(`row_groups/0/columns/4`). The navpath and the `--column`/`--page-index`/
`--row-group` selectors are mutually exclusive. Every singular-verb output
echoes the resolved `_path`.

**`--limit N`** bounds the potentially-large output (`page list` and `page
decode` accept it; `page header` and `page extract` don't — they return a
single object or raw bytes). For `page list` it caps the `items`. For `page
decode` it bounds each collection in the view to the first **N rows** of the
page: each level stream's `levels`, the resolved/PLAIN `values`, and the
dictionary index `runs` — with the `runs` *clipped* so they cover exactly N
rows (a single `{value: 0, length: 1000000}` run shows as `{value: 0, length:
5}` under `--limit 5`, not in full). Every bounded collection reports its own
`total` / `returned` / `truncated`, so a truncation is always explicit.
`--kind statistics` ignores it (the header's statistics aren't per-row).
Default is no limit (the whole page).

One subtlety on **nullable** columns: the level streams have one entry per row
(nulls included), while the value/index collections — resolved `values`, PLAIN
`values`, and the dictionary `runs` — only carry the **present (non-null)**
values, since parquet stores no index or value for a null. So `--limit N`
bounds the level streams to the first N rows and the value/index collections to
the first N present values; the two coincide exactly when the page has no
nulls, and otherwise their `total`s differ (rows vs. non-null count) and their
windows diverge by the nulls in the prefix.

This makes `--limit` the tool for a page that's tiny on disk but expands to
millions of values (one long RLE run): `page decode … --kind values --limit 10`
samples the first ten decoded values without materializing the rest.

```bash
$ parquet-analyzer page list data.parquet --column Sex | jq -r '.items[1]._path' \
  | xargs parquet-analyzer page decode data.parquet
```

**`page list`** is the page-walk surface: a column's (or every column's) pages
as lightweight stubs. It is cheap when the writer emitted an `OffsetIndex` (the
extents come from it, and stubs carry `first_row_index`); otherwise it walks
the per-page headers. Like the other `list` verbs it wraps in
`{items, total, returned, truncated}` and every item carries a `_path`.

```bash
$ parquet-analyzer page list titanic.parquet --column Sex
{
  "$schema": "parquet-analyzer/v1/page-list",
  "items": [
    {
      "row_group": 0, "column": "Sex", "_path": "row_groups/0/columns/4/pages/0",
      "page_index": 0, "data_page_index": null, "kind": "dictionary_page",
      "offset": 24256, "length": 33, "first_row_index": null
    },
    {
      "row_group": 0, "column": "Sex", "_path": "row_groups/0/columns/4/pages/1",
      "page_index": 1, "data_page_index": 0, "kind": "data_page",
      "offset": 24289, "length": 468, "first_row_index": null
    }
  ],
  "returned": 2, "total": 2, "truncated": false, "column": "Sex"
}
```

**`page header`** dumps one page's full header fields. It is version-aware: V1
data pages report the level encodings, V2 data pages report
`num_nulls`/`num_rows`/`is_compressed` and the level byte lengths.

```bash
$ parquet-analyzer page header titanic.parquet --column Sex --page-index 1
{
  "$schema": "parquet-analyzer/v1/page-header",
  "_path": "row_groups/0/columns/4/pages/1",
  "row_group": 0, "column": "Sex", "page_index": 1, "data_page_index": 0,
  "kind": "data_page", "page_type": "DATA_PAGE", "offset": 24289,
  "header_size": 20, "compressed_size": 448, "uncompressed_size": 846,
  "num_values": 891, "encoding": "RLE_DICTIONARY",
  "definition_level_encoding": "RLE", "repetition_level_encoding": "RLE",
  "statistics": null
}
```

**`page extract`** emits one page's raw body bytes. `--decompress` removes the
page codec (page-type aware: a V2 page keeps its uncompressed level streams and
expands only the values section). `--as {hex,base64,raw}` chooses the encoding;
`--as raw` is the JSON escape hatch — it writes the bytes verbatim to stdout
(or `-o FILE`) with no JSON envelope.

```bash
$ parquet-analyzer page extract titanic.parquet --column Sex --page-index 0 --as hex
$ parquet-analyzer page extract titanic.parquet --column Sex --page-index 1 --decompress --as raw -o body.bin
```

**`page decode`** decodes a page body. With no `--kind` it emits the full
**encoding-faithful** decode in one object — the definition/repetition level
streams plus the values section in its native encoding form, which the page's
encoding determines (no choice to make):

```bash
$ parquet-analyzer page decode titanic.parquet row_groups/0/columns/4/pages/1 --limit 3
{
  "$schema": "parquet-analyzer/v1/page-decode",
  "_path": "row_groups/0/columns/4/pages/1",
  "row_group": 0, "column": "Sex", "page_index": 1, "data_page_index": 0,
  "encoding": "RLE_DICTIONARY", "num_values": 891, "num_nulls": 0,
  "definition_levels": {
    "bit_width": 1, "total": 891, "returned": 3, "truncated": true,
    "levels": [1, 1, 1]
  },
  "repetition_levels": null,
  "encoded_values": {
    "kind": "dictionary_indices", "bit_width": 2,
    "total": 891, "returned": 3, "truncated": true,
    "runs": [
      {"kind": "rle", "value": 0, "length": 1},
      {"kind": "rle", "value": 1, "length": 2}
    ]
  }
}
```

`encoded_values` is the values section as it is literally stored: for a
dictionary-encoded page the index RLE/bit-packed runs (`kind:
"dictionary_indices"`); for a PLAIN page the verbatim values (`kind: "plain"`).
This is the encoding-faithful view — the data the encoding scheme actually
stores, not a flattened logical reconstruction.

`--kind` narrows to a single view (and is the only way to get the *resolved*
physical values):

- `values` — the page's **resolved** physical values, with dictionary indices
  resolved through the sibling dictionary page to the physical-type values.
  Nulls are skipped, so `total` is the non-null count.
- `levels` — just the definition and repetition level streams (each a
  `bit_width` plus the expanded `levels`; either `null` when the column has no
  such block — a flat column has no repetition levels, a required column no
  definition levels).
- `rle-runs` — just the dictionary index runs (`kind_not_available` on a PLAIN
  page).
- `statistics` — the page header's statistics (header-only; no body decode).

```bash
$ parquet-analyzer page decode titanic.parquet --column Sex --page-index 1 --kind values --limit 2
```

A page whose encoding or codec this tool does not yet decode returns a
structured error (`encoding_not_supported` / `codec_not_supported`) that names
the unsupported scheme — fall back to `page extract` for the raw bytes.

#### Schema-version discovery

Every subcommand accepts `--schema-version`, which short-circuits to print
the JSON `$schema` URI for that subcommand's output without opening the
file. Useful for tools that want to fetch the matching schema definition.

```bash
$ parquet-analyzer column show --schema-version
{
  "$schema": "parquet-analyzer/v1/column-show"
}
```

#### Errors

Errors are written to **stderr** as a single JSON line with the contract:

```json
{
  "$schema": "parquet-analyzer/v1/error",
  "error": "column_not_found",
  "message": "column 'foo' not found. Available: a, b, c",
  "fix": "parquet-analyzer column list example.parquet"
}
```

The `fix` field always contains an exact command to retry. The process
exits non-zero on operational errors. `file validate` is the exception:
parse failures are reported as findings (`valid: false`, exit 0) so the
subcommand stays useful for "is this file partially-written?" queries.

## Library API

`parquet-analyzer` is also a Python library. Anything the CLI does is available as an importable function, so you can build encoding-level verification scripts without shelling out.

### `ParquetFile` (the primary entry point)

```python
from parquet_analyzer import ParquetFile

with ParquetFile("example.parquet") as pf:
    # Footer-only — instant on any file size.
    print(pf.num_rows, pf.num_row_groups, pf.num_columns)
    print(pf.schema)
    print(pf.kv_metadata_lookup("com.acme.author"))

    # Walk row groups + column chunks — still footer-only, no body reads.
    for rg in pf.row_groups:
        for cc in rg.columns:
            print(cc.path, cc.type, cc.encodings, cc.codec, cc.num_values)
```

`ParquetFile` reads only the footer on construction (a few KB at end of file). Per-row-group / per-column / per-page metadata is exposed through wrapper classes that defer expensive parsing until you actually ask for it.

Two summary surfaces:

- `pf.footer_summary` — cheap, footer-only (row/group/column counts, footer + file size, aggregate compressed/uncompressed column-chunk sizes).
- `pf.full_summary` — same shape as legacy `get_summary()`, includes per-page counts (`num_pages`, `num_data_pages`, etc.). **Triggers a full eager walk.**

Per-chunk lazy page walking:

```python
# How many pages does this chunk have? Fast (O(1)) if the writer included
# an OffsetIndex; falls back to walking otherwise.
cc = pf.row_groups[0].columns[0]
if cc.has_offset_index:
    print(f"chunk has {cc.num_pages} pages (cheap lookup via OffsetIndex)")
else:
    print(f"chunk has {cc.num_pages} pages (paid full walk to count them)")

# Iterate the page headers themselves (walks once, caches).
for page in cc.pages():
    print(page.type, page.encoding, page.num_values, page.offset)
```

`columnchunk.pages()` walks only that one chunk's page headers (cached after first call) — much cheaper than the full-file walk. `num_pages` is even cheaper when an OffsetIndex is present (SNPW always writes one; pyarrow does when `write_page_index=True`; older parquet-mr files often don't).

### Decoding page bodies

Each `Page` decodes its own body on demand — **faithfully to the encoding**,
not as a flattened reconstruction — without walking the rest of the file.
Every encoded stream is exposed in its own structure:

```python
cc = pf.row_groups[0].columns[0]
page = cc.page(1)                 # seeks via the OffsetIndex when present

decoded = page.decode()           # a DecodedPage (cached on the page)
decoded.num_nulls                 # V2: from the header; V1: from def levels

# Level streams: RleBitPackedStream (the encoding parquet uses for levels),
# or None when the column has no such block on disk (required / non-repeated).
decoded.definition_levels         # RleBitPackedStream | None
decoded.repetition_levels         # RleBitPackedStream | None
lvl = decoded.definition_levels
lvl.bit_width                     # packing width (from max_definition_level)
lvl.runs                          # ordered (RleRun | BitPackedRun) — on-disk structure
lvl.values                        # the expanded per-value levels

# Values section: PlainValues for PLAIN, or — because dictionary indices use
# the SAME RLE/bit-packed encoding as levels — an RleBitPackedStream of the
# raw indices for a dictionary page.
decoded.values                    # PlainValues | RleBitPackedStream
# e.g. for a dict page: decoded.values.bit_width / .runs / .values (indices)

# Physical-TYPE values (one level below logical): the encoding decoded and
# dict indices resolved, but NOT logical-type-interpreted — a UTF8 column
# yields b'S', not 'S' (a logical_values() companion is planned):
page.physical_values()
page.definition_levels()          # convenience: expanded levels ([0]*n if no block)
page.repetition_levels()
page.raw_body()                   # the on-disk body bytes (no decode)

# The chunk's dictionary (decoded once, cached) when it has a dictionary page:
cc.dictionary()                   # list of physical-type values, or None
cc.max_definition_level           # schema-derived, cheap
cc.max_repetition_level
```

`decode()` gives you the encoding's own data: an `RleBitPackedStream`
(`bit_width` + ordered `RleRun` / `BitPackedRun`) for levels and dictionary
indices, and `PlainValues` for PLAIN. The values section carries only the
**non-null** entries (`num_values - num_nulls`); the nulls live in the
definition levels — this is the on-disk shape, not a reassembled column.

There are three levels between the on-disk bytes and the logical values:
the **encoding representation** (`decode().values` — indices/runs/PLAIN
bytes), the **physical-type values** (`physical_values()` — decoded to the
parquet physical type with the dictionary resolved, e.g. `b'S'`), and the
**logical-type values** (the physical values reinterpreted via the column's
logical type, e.g. `'S'` / `Decimal` / a timestamp — not yet implemented).
The dictionary is an *encoding*, not a type, so `physical_values()` yields
the same physical type whether or not the page was dictionary-encoded.

V1 (length-prefixed levels inside the compressed body) and V2 (uncompressed
levels ahead of an optionally-compressed values section) are handled
transparently. An out-of-scope encoding or an undecompressable codec on
`decode()`, a non-data page on `decode()`, or a missing dictionary on
`physical_values()` raises a typed `PageDecodeError` subclass
(`UnsupportedEncodingError`,
`UnsupportedCodecError`, `UnsupportedPageTypeError`, `MissingDictionaryError`)
carrying a stable `.code`.


### Full eager walk (legacy shape)

When you genuinely need the complete byte-range view of every structural element:

```python
with ParquetFile("example.parquet") as pf:
    every_segment = pf.all_segments()       # sorted + gap-filled
    every_page = pf.all_pages()             # per-column pages tree
    chunk_offset_map = pf.column_offset_map # legacy parse_parquet_file()[1] shape
```

These are the only methods that trigger the full per-page Thrift walk that the legacy `parse_parquet_file()` always did. The CLI's `--output-mode segments` mode uses `pf.all_segments()`; the `default` and `html` modes use `pf.full_summary` + `pf.footer` + `pf.all_pages()`.

**Breaking change in v0.4:** the free functions `parse_parquet_file()`, `get_summary()`, `get_pages()`, and `find_footer_segment()` were removed. Use the `ParquetFile` methods above as drop-in replacements.

`segment_to_json`, `json_encode`, and `fill_gaps` remain as module-level utilities.

### Decoders (`parquet_analyzer.decoders`)

Page-level decoders for the encoded streams a Parquet inspector actually needs to read:

```python
from parquet_analyzer.decoders import (
    decompress,
    decode_rle_bitpacked_hybrid,
    decode_rle_bitpacked_hybrid_stream,
    decode_levels,
    decode_v1_level_block,
    decode_plain,
    DecodeStats,
    RleBitPackedStream,
    RleRun,
    BitPackedRun,
)
```

| Function | What it does |
|---|---|
| `decompress(data, codec, uncompressed_size)` | Decompress a Parquet compressed byte slice. Supports `UNCOMPRESSED`, `SNAPPY`, `GZIP`, `ZSTD`, `LZ4` (legacy Hadoop framed), `LZ4_RAW`. |
| `decode_rle_bitpacked_hybrid_stream(data, bit_width, num_values)` | Decode a raw RLE / bit-packed-hybrid stream into an `RleBitPackedStream` — the encoding's own structure: `bit_width` plus the ordered `runs` (each an `RleRun(value, length)` or `BitPackedRun(length, values)`) plus the flattened `values`. This is the faithful "encoder-logical" view used for both level streams and dictionary indices. |
| `decode_rle_bitpacked_hybrid(data, bit_width, num_values)` | The same stream as a `(values, DecodeStats)` pair — the flattened values plus a flat run summary. Use `decode_rle_bitpacked_hybrid_stream` when the ordered, per-run structure matters. |
| `decode_levels(data, max_level, num_values)` | Decode definition or repetition levels to a flat list. Bit width is derived from `max_level`. Returns `[0] * num_values` if `max_level == 0` (no level block exists on disk for required columns). |
| `decode_v1_level_block(data, offset, max_level, num_values)` | V1-data-page helper that handles the 4-byte little-endian length prefix in front of each level block. Returns `(levels, new_offset)`. V2 data pages store level byte lengths in the page header instead — for those, slice the bytes yourself and call `decode_levels` directly. |
| `decode_plain(data, parquet_type, num_values, type_length=None)` | Decode `PLAIN`-encoded values. Supported types: `BOOLEAN`, `INT32`, `INT64`, `INT96`, `FLOAT`, `DOUBLE`, `BYTE_ARRAY`, `FIXED_LEN_BYTE_ARRAY` (`type_length` required). `INT96` is returned as 12-byte `bytes` — interpretation is the caller's responsibility. |

End-to-end example — decode the indices of a dictionary-encoded V1 data page
and recover the original values. For most callers, `Page.decode()` (above)
does exactly this; reach for the raw decoders only when you need a custom
decode path. The byte ranges come from the public `Page` API:

```python
from parquet_analyzer import ParquetFile
from parquet_analyzer.decoders import (
    decompress, decode_v1_level_block, decode_rle_bitpacked_hybrid, decode_plain,
)

pf = ParquetFile("example.parquet")
cc = pf.row_groups[0].columns[0]            # a dict-encoded BYTE_ARRAY column
dict_page = next(p for p in cc.pages() if p._kind == "dictionary_page")
data_page = cc.page(cc.num_pages - 1)

# Dictionary entries are PLAIN values of the column's physical type.
dict_raw = decompress(dict_page.raw_body(), cc.codec, dict_page.uncompressed_size)
dict_values = decode_plain(dict_raw, cc.type, dict_page.num_values)

# V1 data page layout: [4-byte LE def-level block length][def-level RLE stream]
#                      [1-byte indices bit-width][indices RLE/bit-packed stream]
raw = decompress(data_page.raw_body(), cc.codec, data_page.uncompressed_size)
def_levels, after_def = decode_v1_level_block(
    raw, 0, cc.max_definition_level, data_page.num_values
)
indices_bit_width = raw[after_def]

# In dict-encoded pages, the indices stream contains entries only for
# non-null rows (nulls are represented in the def-level stream above and
# carry no value). So the index count is `num_non_null`, NOT `num_values`.
num_non_null = sum(1 for d in def_levels if d == cc.max_definition_level)
indices, stats = decode_rle_bitpacked_hybrid(
    raw[after_def + 1:], indices_bit_width, num_non_null,
)
print(f"RLE runs: {stats.rle_run_count}, bit-packed runs: {stats.bit_packed_run_count}")

# Reassemble nulls into the final value list using the def-level stream.
it = iter(indices)
values = [
    dict_values[next(it)] if d == cc.max_definition_level else None
    for d in def_levels
]
```

Gotchas worth knowing:

* **V1 vs V2 level layout.** V1 prefixes each level block with a 4-byte LE length. V2 stores the byte length in the page header instead (see the next bullet for V2's body layout). Use `decode_v1_level_block` for V1; for V2 slice the bytes yourself and call `decode_levels`.
* **V2 page body layout is `[rep_levels][def_levels][values]`.** Levels are stored uncompressed regardless of `is_compressed`; only the values section is (optionally) compressed. Concretely, V2 callers should slice `raw[:rep_len]`, `raw[rep_len:rep_len + def_len]`, and `raw[rep_len + def_len:]` using `repetition_levels_byte_length` and `definition_levels_byte_length` from the page header, then call `decode_levels` directly on the level slices (and `decompress` on the values slice only when `is_compressed` is true). Do not pass an entire V2 page body to `decompress` as one block.
* **Required columns have no level block.** When `max_def_level == 0`, the file contains no def-level bytes at all. `decode_levels` returns `[0] * num_values` in that case.
* **Indices skip nulls.** For nullable columns, the indices stream length is the non-null row count (`sum(def_levels)` when `max_def_level == 1`), not the page's `num_values`. Asking `decode_rle_bitpacked_hybrid` for too many values raises `truncated` (or silently produces garbage indices if the indices block is followed by other bytes).
* **Per-page dictionary-index bit width.** Dictionary-encoded data pages start their indices block with a 1-byte `bit_width` chosen by the writer (it may exceed `ceil(log2(dict_size))`). Read that byte first, then pass the rest to `decode_rle_bitpacked_hybrid`.

### Legacy whole-file modes

The legacy CLI surface accepts a positional file path plus an optional
`--output-mode` flag. It performs a full eager walk of all page headers and
emits a complete JSON bundle or an interactive HTML report. Use the
verb-noun subcommands above for targeted inspection; reach for these modes
when you want a complete dump or the interactive HTML view.

```bash
# Full JSON bundle (summary + footer + pages)
parquet-analyzer example.parquet

# Raw segment structures (offsets, lengths, thrift payloads)
parquet-analyzer --output-mode segments example.parquet

# Interactive HTML report
parquet-analyzer --output-mode html -o report.html example.parquet

# HTML report with selected sections only
parquet-analyzer --output-mode html \
  --html-sections summary schema key-value-metadata row-groups columns segments \
  -o report.html example.parquet

# Enable debug logging
parquet-analyzer --log-level DEBUG example.parquet
```

#### Standard output (`--output-mode default`)

The default output provides a structured JSON payload with three main sections:

**Summary statistics**
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
    "bloom_filter_size": 0,
    "footer_size": 527,
    "file_size": 753
  }
}
```

**Footer metadata** — complete Parquet file metadata including schema
definition, row group information, column chunk metadata, and encoding and
compression details.

**Page information** — detailed breakdown of all pages organized by column:
data pages with encoding and statistics, dictionary pages, column indexes,
offset indexes, and bloom filters.

#### Segments (`--output-mode segments`)

When using `--output-mode segments`, the tool outputs a detailed segment-by-segment breakdown showing:

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
- Understanding exact binary layout
- Analyzing file format compliance
- Optimizing file structure

#### HTML report (`--output-mode html`)

Emits a standalone HTML document with collapsible sections for summary statistics, schema, key-value metadata, row groups, aggregated column statistics, segments, and the raw footer. Use the `--html-sections` flag to control which sections are rendered:

```bash
parquet-analyzer --output-mode html \
  --html-sections summary schema key-value-metadata row-groups columns segments \
  -o report.html \
  example.parquet
```

Example: https://clee704.github.io/parquet-analyzer/examples/example.html

## Design

The design contracts behind the CLI surfaces live in two documents:

- **[`docs/output-principles.md`](docs/output-principles.md)** — the contract
  for verb-noun subcommand outputs. The TL;DR: v1 subcommand output is
  "footer-bounded and walk-free" — every field is derivable from the parsed
  footer plus, at most, one additional parse of an index thrift the writer
  already emitted (OffsetIndex, ColumnIndex, BloomFilter header). Anything that
  requires per-page thrift header walks or page body reads belongs to the `page`
  verb, the deliberate per-page opt-in.
- **[`docs/tree-schema.md`](docs/tree-schema.md)** — the kind catalog for the
  v3 tree output surface, the node shapes that `show` emits (each carrying a
  `_location: {offset, length}` byte range).

Read both before adding fields to an existing subcommand or introducing new
node kinds.

## Technical details

The tool uses a custom Thrift protocol implementation (`OffsetRecordingProtocol`) that wraps the standard Thrift compact protocol to track byte offsets and lengths of all decoded structures. This enables precise mapping of logical Parquet structures to their binary representation.

### Footer cache

Because each command is a one-shot process, a complex footer (hundreds of row groups × tens of columns) would re-pay the offset-recording decode — seconds — on every invocation. To keep repeated inspection of the same file fast, the parsed footer of a large file is cached on disk and served on the next open of an unchanged file (e.g. an interactive sequence of commands drops from seconds to tens of milliseconds per call).

The cache is transparent and safe by construction:

- **Content-addressed** on the footer bytes, so it is self-invalidating — any change to the file produces a different key and a stale entry can never be served. The cache also invalidates automatically across tool/Thrift upgrades.
- Stored in a private (`0700`) per-user cache directory; if that directory cannot be confirmed private, the cache silently disables itself and parsing proceeds normally.
- Written only for footers large enough to be worth it, and bounded in total size (oldest entries evicted).

It is fully optional:

| Environment variable | Effect |
|---|---|
| `PARQUET_ANALYZER_NO_CACHE=1` | Disable the cache entirely (always parse). |
| `PARQUET_ANALYZER_CACHE_DIR=<path>` | Relocate the cache directory (default: `$XDG_CACHE_HOME/parquet-analyzer/` or `~/.cache/parquet-analyzer/`). |
| `PARQUET_ANALYZER_CACHE_MAX_BYTES=<n>` | Eviction ceiling for the cache directory (default 1 GiB). |

The `ParquetFile(path, use_cache=False)` keyword bypasses it programmatically.

## Development

### Environment setup

```bash
pip install -e .[dev]
hatch run dev:check  # will format, lint, type-check, test with coverage
```

The development extra pulls in tooling (`hatch`, `ruff`, `pytest`, `pytest-benchmark`) and `pyarrow` / `numpy` so tests can generate Parquet fixtures on the fly.

### Benchmarks

Benchmark tests live under `tests/bench/` and are **excluded from the default `pytest` run** (and from `hatch run dev:check`) — they're heavier than the unit tests and not meant for every commit. Run them explicitly:

```bash
# Run all benchmarks (synthetic parquet fixtures generated on the fly)
pytest tests/bench/ --benchmark-only

# Compare against the committed baseline (captured at the start of the
# lazy-core work; see tests/bench/baselines/ for the JSON).
# Note: the leading "0001_" is pytest-benchmark's auto-prepended save-
# sequence prefix; --benchmark-compare matches by prefix, so the full
# filename stem is required.
# The -W flag suppresses pytest-benchmark's "machine_info changed"
# warning, which fires noisily on virtualized hosts where cpu.hz_actual
# jitters between runs even on the same machine.
pytest tests/bench/ --benchmark-only \
  --benchmark-storage=file://tests/bench/baselines \
  --benchmark-compare=0001_eager-v0.4.0 \
  -W ignore::pytest_benchmark.logger.PytestBenchmarkWarning
```

The synthetic fixture generator (`tests/bench/generate.py`) produces three shapes that stress different lazy-parsing boundaries:

- `wide` — many columns, few rows (footer is large relative to body)
- `tall` — few columns, many rows (body is huge relative to footer; the canonical case for the lazy-core's footer-only fast path)
- `deep` — few columns, many rows split across many row groups (stresses per-row-group / per-chunk walking)

To capture a fresh baseline (e.g., after a behavior change in the eager path):

```bash
pytest tests/bench/ --benchmark-only \
  --benchmark-storage=file://tests/bench/baselines \
  --benchmark-save=eager-vX.Y.Z
```

Baseline files are platform-specific (CPU, Python version) — pytest-benchmark stores them under `tests/bench/baselines/<platform>/`.

> **⚠️ Cross-machine comparisons are not meaningful.** The committed baseline (`eager-v0.4.0`) was captured on the maintainer's machine (Linux/CPython 3.14/x86_64, AMD EPYC 9V74). The per-platform subdirectory prevents the most obvious mismatches (Linux vs macOS, x86 vs ARM), but does **not** distinguish between CPU SKUs in the same broad category — an old i7 and a current EPYC both land in `Linux-CPython-3.14-64bit/`, with wildly different absolute numbers. When validating perf changes against the lazy-core work, **capture your own baseline before the change and compare against that**, not against the committed one. The committed baseline is useful for the maintainer's own session-to-session comparisons and as a frozen reference for the perf claims in PR #18. There's no CI gating on benchmark numbers; perf validation is operator-driven on a single machine at a time.

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

This project is licensed under the [Apache License 2.0](LICENSE).

© 2025 Chungmin Lee
