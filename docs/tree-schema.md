# Tree schema: footer-layer kinds (v0)

This doc defines the **tree-node schema** for `parquet-analyzer`'s v2
output surface — the catalog of node kinds that represent parquet's
on-disk structure as a navigable lazy tree, plus the universal rules
every node follows.

Companion to [`output-principles.md`](output-principles.md), which
defines the *contract* (footer-bounded and walk-free, escape hatches,
honesty pattern). This doc defines the *shapes* the contract operates
on.

This is **v0** — the footer-layer kinds. Body-layer kinds (sub-page
structure: def_block, values_block, indices, dict_lookup, encoding-
specific leaves) land in v1 alongside the body-decode work in
[#21]. Both versions are additive within v2.

## Universal node contract

Every tree node — leaf or branch — carries three system fields:

| Field | Type | Always present? | Meaning |
|---|---|---|---|
| `_kind` | string | yes | identifies the node's schema; consumer reads this and looks up the rest in this doc |
| `_offset` | int | yes | start byte of this node's bytes within the file |
| `_length` | int | yes | byte length of this node's bytes |

Leaf nodes (kinds where the schema says "leaf with scalar value")
additionally carry:

| Field | Type | Meaning |
|---|---|---|
| `_value` | scalar or array | the decoded value this leaf represents |

Branch nodes (kinds where the schema lists named children or content
fields) do **not** have `_value`. Their identity is their content
fields and children.

Everything else on a node is **kind-specific**: content fields and
named child nodes (or arrays of child nodes), sitting flat alongside
the system fields. No `value` wrapper around children — the named
keys ARE the structure.

### Reserved namespaces

- `_*` prefix — system / annotation fields. Reserved for the
  framework. Today: `_kind`, `_offset`, `_length`, `_value`. Future:
  `_lazy` (see below), `_error`, etc.
- `$schema` at output root — the response-shape URI (existing JSON
  Schema convention, carried over from Slice 3). Format:
  `parquet-analyzer/v2/...`.

All other names are kind-specific content and may shadow nothing.

### Kind-defines-schema

`_kind` is the single source of truth for what fields a node has.
Consumers should not introspect by checking "does this node have a
`_value`?" — they should look up the kind in this catalog and know
the answer up front. This eliminates ambiguity (e.g., a leaf with
`_value: {"key": "val"}` vs a branch with a child named `value`).

## Lazy markers

Tree nodes can appear in JSON output in two forms:

**Materialized** — the full content per the kind's schema:
```json
{"_kind": "row_group", "_offset": 4, "_length": 306419,
 "num_rows": 891, "total_byte_size": 306419, "columns": [...]}
```

**Lazy stub** — only the system fields, plus `_lazy: true`:
```json
{"_kind": "row_group", "_offset": 4, "_length": 306419, "_lazy": true}
```

A lazy stub tells the consumer "this node exists at this offset with
this kind; to see its content, fetch it explicitly." Stubs appear
when serialization stops at a depth limit, or when a child node
hasn't been materialized yet because no caller asked for it.

The Python API distinguishes these implicitly — attribute access on
a lazy node triggers materialization. JSON serialization makes the
distinction explicit via `_lazy: true`.

## Tree containment

**Tree containment is logical, not strictly physical.** A child
node's `[_offset, _offset+_length)` range is NOT required to be a
subset of its parent's range. This relaxation matters for indexes:
a column chunk's `offset_index` typically lives elsewhere in the
file (often after all data, before the footer), but is logically a
child of the chunk it describes. Forcing strict containment would
either move indexes to a `file`-level flat list (awkward navigation)
or require duplicate node representation. The tree expresses logical
association; `_offset`/`_length` carry the physical truth.

Consumers needing a strict-physical-order view of the file should
sort all materialized nodes by `_offset` (the existing eager
`--output-mode segments` behavior, expressible as `tree --depth all`
followed by an offset-sort).

## Versioning policy

- Kinds are added freely (additive).
- A kind's schema (its field set, child set, `_value` semantics) is
  the v2 contract for that kind. Changes to an existing kind's schema
  bump the major version (v2 → v3).
- New optional fields on an existing kind are NOT breaking (they're
  additive within the same major version).
- The `$schema` URI on outputs carries the major version:
  `parquet-analyzer/v2/...`.

## The v0 kind catalog

Read top-down — `file` is the root.

### `file` (branch, root)

Spans the entire file. Convenience fields re-exposed from `footer`
to save consumers a navigation step.

| Field | Type | Notes |
|---|---|---|
| `path` | string | filesystem path used to open the file |
| `created_by` | string \| null | re-exposed from `footer.created_by` |
| `num_rows` | int | re-exposed from `footer.num_rows` |
| `num_row_groups` | int | derived |
| `num_columns` | int | derived |

Children:

| Name | Kind | Multiplicity |
|---|---|---|
| `header_magic` | `header_magic` | exactly 1 |
| `row_groups` | `row_group` | 0+ (array) |
| `footer` | `footer` | exactly 1 |
| `footer_length` | `footer_length` | exactly 1 |
| `trailer_magic` | `trailer_magic` | exactly 1 |

`_offset` = 0, `_length` = file size.

### `header_magic` (leaf)

The 4-byte `PAR1` at file start.

`_value`: `"PAR1"` (string)

`_offset` = 0, `_length` = 4.

### `trailer_magic` (leaf)

The 4-byte `PAR1` at file end.

`_value`: `"PAR1"` (string)

`_offset` = file size − 4, `_length` = 4.

### `footer_length` (leaf)

The 4-byte little-endian unsigned int immediately before
`trailer_magic`, encoding the size in bytes of the `footer` thrift.

`_value`: integer (the footer size in bytes)

`_offset` = file size − 8, `_length` = 4.

### `footer` (branch)

The parsed `FileMetaData` thrift. Located at
`_offset` = file size − 8 − footer_size, `_length` = footer_size.

| Field | Type | Notes |
|---|---|---|
| `version` | int | parquet format version |
| `num_rows` | int | total rows across all row groups |
| `created_by` | string \| null | writer identification string |

Children:

| Name | Kind | Multiplicity |
|---|---|---|
| `schema` | `schema_element` | 1+ (array; first element is the root, rest are flat tree per parquet spec) |
| `kv_metadata` | `kv_metadata_entry` | 0+ (array) |

Note: the footer's thrift contains a `row_groups` list, but the
actual row-group data lives elsewhere in the file. Per the tree
model, `row_group` nodes are children of `file` (with their own
`_offset`/`_length` describing the on-disk data extent), not of
`footer`. The footer carries the *description* of row groups; the
`file` node carries the *instances*. Both views are populated from
the same footer thrift parse.

### `schema_element` (branch with no children, or "leaf-like struct")

A single entry from the footer's flat schema list. Parquet's schema
is encoded as a depth-first flat list where each element has a
`num_children` count indicating its sub-tree size. v0 mirrors this
flat shape (consumers needing the tree structure can rebuild it
from `num_children`).

`_offset` / `_length` reference the byte range of this element's
thrift encoding within the footer.

| Field | Type | Notes |
|---|---|---|
| `name` | string | element name |
| `repetition_type` | string \| null | `REQUIRED` / `OPTIONAL` / `REPEATED` (null for the root) |
| `type` | string \| null | physical type (`INT32`, `BYTE_ARRAY`, etc.); null for non-leaf elements |
| `converted_type` | string \| null | logical type hint (`UTF8`, `DECIMAL`, etc.); legacy |
| `logical_type` | object \| null | structured logical-type info; modern equivalent of `converted_type` |
| `num_children` | int | 0 for leaves; >0 for STRUCT-like elements |
| `field_id` | int \| null | optional field ID |
| `precision` | int \| null | for DECIMAL |
| `scale` | int \| null | for DECIMAL |
| `type_length` | int \| null | for FIXED_LEN_BYTE_ARRAY |

No children, no `_value`.

### `kv_metadata_entry` (branch with no children)

A single key-value pair from the footer's key-value metadata list.
Parquet permits duplicate keys; the list shape preserves order and
duplicates.

`_offset` / `_length` reference the byte range of this entry within
the footer.

| Field | Type | Notes |
|---|---|---|
| `key` | string | |
| `value` | string \| null | may be null if the writer recorded a key with no value |

No children, no `_value`.

### `row_group` (branch)

A single row group's data on disk. `_offset` is the row group's
`file_offset` from the footer (start of first page in the rg).
`_length` is the sum of column-chunk compressed sizes (`compressed_size`
on each `column_chunk`).

| Field | Type | Notes |
|---|---|---|
| `num_rows` | int | rows in this row group |
| `total_byte_size` | int | sum of uncompressed column data (parquet's `RowGroup.total_byte_size`) |
| `total_compressed_size` | int | sum of `column_chunk.compressed_size`; computed |
| `ordinal` | int \| null | row group index in the file (parquet's `RowGroup.ordinal`, often null) |

Children:

| Name | Kind | Multiplicity |
|---|---|---|
| `columns` | `column_chunk` | 1+ (array) |

No `_value`.

### `column_chunk` (branch)

A single column's data within a row group. `_offset` is the chunk's
on-disk start (`dictionary_page_offset` if a dictionary is present,
else `data_page_offset`). `_length` is the chunk's
`total_compressed_size`.

| Field | Type | Notes |
|---|---|---|
| `path` | list of string | column path; flat columns have length 1 |
| `path_display` | string | dot-joined `path` for human display |
| `type` | string | physical type (`INT32`, `BYTE_ARRAY`, etc.) |
| `codec` | string | compression codec (`SNAPPY`, `UNCOMPRESSED`, ...) |
| `encodings` | list of string | encodings present in this chunk |
| `num_values` | int | including nulls |
| `compressed_size` | int | `total_compressed_size`; same as `_length` |
| `uncompressed_size` | int | `total_uncompressed_size` |
| `data_page_offset` | int | start of first data page (footer field) |
| `dictionary_page_offset` | int \| null | start of dictionary page if present |
| `statistics` | object \| null | parsed `ColumnMetaData.statistics` if present |

Children:

| Name | Kind | Multiplicity |
|---|---|---|
| `dictionary_page` | `dictionary_page` | 0 or 1 |
| `pages` | `data_page_v1` \| `data_page_v2` | 0+ (array; mixed kinds allowed within a chunk in theory, single-kind in practice) |
| `offset_index` | `offset_index` | 0 or 1 |
| `column_index` | `column_index` | 0 or 1 |
| `bloom_filter` | `bloom_filter_header` | 0 or 1 |

Note on physical containment: `dictionary_page` and `pages` are
physically within the chunk's `[_offset, _offset+_length)` range.
`offset_index`, `column_index`, and `bloom_filter` typically live
elsewhere in the file (per the relaxation in "Tree containment"
above).

No `_value`.

### `dictionary_page` (branch)

A dictionary page. `_offset` is the page-header start (which is
also the chunk's `dictionary_page_offset`). `_length` covers both
the page header thrift and the page body (compressed if applicable).

| Field | Type | Notes |
|---|---|---|
| `page_type` | string | always `"DICTIONARY_PAGE"` |
| `encoding` | string | dictionary-page encoding (`PLAIN` or `PLAIN_DICTIONARY`) |
| `num_values` | int | number of entries in the dictionary |
| `uncompressed_size` | int | from the page header |
| `compressed_size` | int | from the page header |
| `is_compressed` | bool | from the page header (V2 only; null for V1) |
| `crc` | int \| null | optional CRC from the page header |

Children: the page body decomposes into body-layer kinds (decoded
dictionary values, etc.). v0 leaves this opaque; v1 (#21) populates.

No `_value`.

### `data_page_v1` (branch)

A V1 data page. `_offset` and `_length` cover header + body.

| Field | Type | Notes |
|---|---|---|
| `page_type` | string | `"DATA_PAGE"` |
| `encoding` | string | values encoding (`PLAIN`, `PLAIN_DICTIONARY`, `RLE_DICTIONARY`, etc.) |
| `num_values` | int | values in this page (including nulls) |
| `uncompressed_size` | int | from page header |
| `compressed_size` | int | from page header |
| `definition_level_encoding` | string | usually `RLE` |
| `repetition_level_encoding` | string | usually `RLE` |
| `statistics` | object \| null | per-page statistics if writer included them |
| `crc` | int \| null | optional |

Children: body decomposes into level blocks + values (V1 layout:
length-prefixed level blocks first, then values). v0 opaque; v1
(#21) populates.

No `_value`.

### `data_page_v2` (branch)

A V2 data page. `_offset` and `_length` cover header + body.

| Field | Type | Notes |
|---|---|---|
| `page_type` | string | `"DATA_PAGE_V2"` |
| `encoding` | string | values encoding |
| `num_values` | int | including nulls |
| `num_nulls` | int | |
| `num_rows` | int | |
| `is_compressed` | bool | whether values section is compressed |
| `uncompressed_size` | int | |
| `compressed_size` | int | |
| `definition_levels_byte_length` | int | uncompressed level-block byte length |
| `repetition_levels_byte_length` | int | uncompressed level-block byte length |
| `statistics` | object \| null | per-page statistics |
| `crc` | int \| null | |

Children: body decomposes into rep_block + def_block + values
(V2 layout: levels uncompressed in known byte ranges from the
header, values optionally compressed). v0 opaque; v1 (#21) populates.

No `_value`.

### `offset_index` (opaque branch, v0)

Located at `column_chunk.offset_index_offset`, `_length` =
`column_chunk.offset_index_length`. Contains per-data-page byte
offsets and row indices.

v0: opaque — only `_kind`, `_offset`, `_length` populated. Per-page
internal structure lands in v1 with the body-decode work.

### `column_index` (opaque branch, v0)

Located at `column_chunk.column_index_offset`, `_length` =
`column_chunk.column_index_length`. Contains per-data-page min/max
statistics and null counts.

v0: opaque — only `_kind`, `_offset`, `_length`. Per-page internal
structure deferred.

### `bloom_filter_header` (opaque branch, v0)

Located at `column_chunk.bloom_filter_offset`, `_length` =
`column_chunk.bloom_filter_length`. Contains the bloom filter
header thrift plus the bloom-filter bitset.

v0: opaque — only `_kind`, `_offset`, `_length`. Header thrift +
bitset decomposition deferred.

## Out of scope for v0 (landing in v1 via [#21])

The following kinds will be added when the body-decode work lands:

- `def_block` (V1: 4-byte length prefix + RLE block; V2: bytes in known range)
- `rep_block` (V1 / V2 analog of def_block)
- `values_block` (the encoded values section of a page body)
- `plain_values` (decoded PLAIN values; leaf with `_value` array)
- `dict_indices` (RLE/bit-packed indices; leaf with `_value` array, content fields for bit_width / RLE-vs-bit-packed runs)
- `dict_lookup` (dict-indexed values resolved through the chunk's dictionary; leaf with `_value` array)
- Possibly: `rle_run`, `bit_packed_block` if sub-structure of `dict_indices` is worth surfacing

Each will get a one-line docstring at the introduction site, and
this doc will be expanded with the catalog entry as part of that
PR.

## Worked examples

A footer-only `tree --depth 2` against a typical small file
(materialized at depth 2, lazy stubs below):

```json
{
  "$schema": "parquet-analyzer/v2/tree",
  "_kind": "file",
  "_offset": 0,
  "_length": 40013,
  "path": "example.parquet",
  "created_by": "parquet-cpp version 1.5.1-SNAPSHOT",
  "num_rows": 891,
  "num_row_groups": 1,
  "num_columns": 12,
  "header_magic": {"_kind": "header_magic", "_offset": 0, "_length": 4, "_value": "PAR1"},
  "row_groups": [
    {
      "_kind": "row_group",
      "_offset": 4,
      "_length": 306419,
      "num_rows": 891,
      "total_byte_size": 306419,
      "total_compressed_size": 38839,
      "ordinal": null,
      "columns": [
        {"_kind": "column_chunk", "_offset": 4, "_length": 4357, "_lazy": true},
        {"_kind": "column_chunk", "_offset": 4361, "_length": 1051, "_lazy": true}
      ]
    }
  ],
  "footer": {"_kind": "footer", "_offset": 38843, "_length": 1162, "_lazy": true},
  "footer_length": {"_kind": "footer_length", "_offset": 39997, "_length": 4, "_value": 1162},
  "trailer_magic": {"_kind": "trailer_magic", "_offset": 40009, "_length": 4, "_value": "PAR1"}
}
```

A materialized single column chunk (e.g., `tree --path
row_groups/0/columns/foo --depth 1`):

```json
{
  "$schema": "parquet-analyzer/v2/tree",
  "_kind": "column_chunk",
  "_offset": 24256,
  "_length": 501,
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
  "statistics": {"min_value": "female", "max_value": "male", "null_count": 0},
  "dictionary_page": {"_kind": "dictionary_page", "_offset": 24256, "_length": 20, "_lazy": true},
  "pages": [
    {"_kind": "data_page_v1", "_offset": 24276, "_length": 481, "_lazy": true}
  ],
  "offset_index": null,
  "column_index": null,
  "bloom_filter": null
}
```

A leaf:

```json
{"_kind": "footer_length", "_offset": 39997, "_length": 4, "_value": 1162}
```

## How this doc grows

- New kinds added by appending to the catalog with a one-line
  rationale.
- Field additions to existing kinds (within the same major version)
  documented inline; old consumers ignore the new field.
- Breaking schema changes bump the major (v2 → v3) and require this
  doc to be reissued at the new version.
- The body-layer kinds (v1) get added by [#21]; each kind's
  introduction in code carries a docstring, and that docstring
  populates this doc's catalog entry.
