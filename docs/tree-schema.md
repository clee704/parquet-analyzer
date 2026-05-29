# Tree schema: footer-layer kinds (v0)

This doc defines the **tree-node schema** for `parquet-analyzer`'s v2
output surface — the catalog of node kinds that represent parquet's
on-disk structure as navigable lazy trees, plus the universal rules
every node follows.

Companion to [`output-principles.md`](output-principles.md), which
defines the *contract* (footer-bounded and walk-free by default,
escape hatches, honesty pattern). This doc defines the *shapes* the
contract operates on.

This is **v0** — the footer-layer kinds. Body-layer kinds (sub-page
structure: def_block, values_block, indices, dict_lookup, encoding-
specific leaves) land in v1 alongside the body-decode work in [#21].
Both versions are additive within v2.

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
| `_value` | scalar, array of scalars, or array of dicts | the decoded content this leaf represents — see "Leaf `_value` shapes" below |

Branch nodes do **not** have `_value`. Their identity is their
content fields and children.

Everything else on a node is **kind-specific**: content fields and
named child nodes (or arrays of child nodes), sitting flat alongside
the system fields. No `value` wrapper around children — the named
keys ARE the structure.

### Leaf `_value` shapes

`_value` represents the leaf's decoded content. Three permitted shapes:

| Shape | Example kind | Example value |
|---|---|---|
| Scalar (string / int / bool / float) | `header_magic`, `footer_length` | `"PAR1"`, `1162` |
| Array of scalars | `plain_values` (v1) | `[1, 2, 3, ...]` |
| Array of dicts | `kv_metadata`, `schema` | `[{"key": "k", "value": "v"}, ...]` |

**Not permitted**: a dict-with-named-keys at the top level of
`_value` (e.g., `_value: {"key": "k", "value": "v"}` for a single
entry). That shape collides with branch-with-named-content-fields
and would force the reader to disambiguate based on `_kind`. If a
node is genuinely "one structured thing with named parts," model it
as a branch with content fields, not a dict-`_value` leaf.

**Why array-of-dicts is fine**: the array IS the leaf's content (a
sequence). Each dict inside is uniform-shape and consumer-indexable
— same access pattern as array-of-scalars. The leaf/branch question
is decided at the `_kind` level: a kind whose schema says "leaf with
array `_value`" is a leaf, full stop.

This is the shape used by `kv_metadata` (list of `{key, value}`
entries — preserves order and duplicates per parquet spec) and
`schema` (list of `{name, type, ...}` elements — flat per parquet's
on-disk encoding).

**Future consideration**: per-entry addressability (each
`kv_metadata` entry as its own node with `_offset`/`_length` for
its byte range in the footer) is a viable alternative shape if a
forensics use case ever needs it. v0 doesn't, so the simpler
single-leaf shape wins.

### Reserved namespaces

- `_*` prefix — system / annotation fields. Reserved for the
  framework. Today: `_kind`, `_offset`, `_length`, `_value`. Future:
  `_lazy`, `_error`, etc.
- `$schema` at output root — the response-shape URI (existing JSON
  Schema convention, carried over from Slice 3). Format:
  `parquet-analyzer/v2/...`.

All other names are kind-specific content.

### Kind-defines-schema

`_kind` is the single source of truth for what fields a node has.
Consumers should not introspect by checking "does this node have a
`_value`?" — they should look up the kind in this catalog and know
the answer up front. Eliminates ambiguity (e.g., a leaf with
`_value: [...]` vs a branch with a child named `value`).

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

Stubs appear when serialization stops at a depth limit, or when a
child node hasn't been materialized because no caller asked for it.

The Python API distinguishes these implicitly — attribute access on
a lazy node triggers materialization. JSON output makes the
distinction explicit via `_lazy: true`.

## Two views: `tree` and `layout`

Parquet's on-disk reality has two equally-real shapes:

- **Logical structure** — file → row groups → column chunks → pages
  / indexes / bloom filters. This is how readers conceptualize the
  format and what users almost always want.
- **Physical layout** — actual byte positions on disk, including
  bytes that aren't referenced by any logical structure (writer
  quirks, deprecated fields, intentional gaps). This is what
  matters for caching/prefetch reasoning, compression-ratio
  analysis, layout-conformance auditing, and forensics (e.g.,
  spotting the bytes pyarrow used to write at end-of-chunk as a
  duplicate ColumnMetaData copy — see
  [PARQUET-2139](https://github.com/apache/parquet-format/pull/440) /
  [arrow#43427](https://github.com/apache/arrow/issues/43427)).

v0 supports both as first-class views. Same kind catalog (mostly —
see "Layout-view-only kinds" below). The views differ in **how
children are arranged**:

| | Tree view | Layout view |
|---|---|---|
| Surfaces what | Logical hierarchy | Physical byte ordering |
| Containment rule | A node's children are everything it's logically associated with | A node's children must be physically contained (`child._offset` within `parent._offset..parent._offset+parent._length`) |
| Non-contained logical children | Appear inline as full child nodes | Replaced by `<name>_ref` content fields carrying `{_kind, _offset, _length}`; the actual node lives at its physical position in the tree |
| Unreferenced bytes | Don't appear (nothing logically points to them) | Appear as `unknown` leaf nodes |
| `$schema` URI | `parquet-analyzer/v2/tree` | `parquet-analyzer/v2/layout` |
| Verbs | `tree`, plus all Slice 3 verb-noun verbs (curated tree views) | `layout` (replaces legacy `--output-mode segments`) |

The Python API hides the distinction — `cc.offset_index` returns
the offset_index node regardless of which view emits the JSON.
The view is purely a serialization choice.

### Per-kind: "logical children" + "physically contained?"

Each kind's catalog entry declares its logical children and, for
each child, whether it's physically contained in the parent's byte
range:

- **Physically contained** → appears inline in both views (just
  nested deeper in layout view).
- **Not physically contained** → appears inline in tree view;
  replaced by a `<name>_ref` content field in layout view (the
  actual node appears at its physical position elsewhere in the
  layout tree).

A small number of kinds exist **only in layout view** —
specifically `column_chunk_data_region` and `unknown`. They group
or label physical byte ranges that have no logical counterpart.

## Versioning policy

- Kinds are added freely (additive).
- A kind's schema (its field set, child set, `_value` semantics) is
  the v2 contract for that kind. Changes bump the major (v2 → v3).
- New optional fields on an existing kind are NOT breaking
  (additive within the same major version).
- The `$schema` URI on outputs carries the major version:
  `parquet-analyzer/v2/...`.

## Derived-field policy

Each fact in the file should live in **one node** in the tree.
Convenience pull-ups (copying a field from one node to another for
shallow-access ergonomics) are not allowed — they create two paths
to the same fact, raise "which one is canonical?" ambiguity, and
have no natural stopping point (if `num_rows` is pulled up, why
not `created_by`, `schema`, …?).

What IS allowed:

| Category | Example | OK? |
|---|---|---|
| **Same-node derivation** — alternative presentation of a field this node owns | `column_chunk.path_display` is the dot-joined form of `column_chunk.path` | yes |
| **Up-aggregation from children** — a derived value that doesn't exist as a single field anywhere; computed by summing/iterating children | `row_group.total_compressed_size` is the sum of its `column_chunk.compressed_size`s; no single field carries this | yes |
| **Cross-node pull-up** — copying a field that already exists on another node, just for ergonomic shallow access | `file.num_rows` copying `footer.num_rows` | **no** |

Verbs (the Slice 3 verb-noun surface — `file summary`, `column show`,
etc.) compose the curated output they need from across the tree. The
tree shouldn't pre-compose; the verbs do that work.

## The v0 kind catalog

Read top-down — `file` is the root.

### `file` (branch, root)

Spans the entire file. The only content field is `path` (a property
of *this parse*, not of the file content). Aggregates from other
nodes (e.g., `footer.num_rows`) are NOT re-exposed here — see
"Derived-field policy" below.

| Field | Type | Notes |
|---|---|---|
| `path` | string | filesystem path used to open the file |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `header_magic` | `header_magic` | exactly 1 | yes |
| `row_groups` | `row_group` | 0+ (array) | yes (each `row_group`'s `_offset`/`_length` cover its metadata-thrift bytes in footer; physically inside `footer`) |
| `footer` | `footer` | exactly 1 | yes |
| `footer_length` | `footer_length` | exactly 1 | yes |
| `trailer_magic` | `trailer_magic` | exactly 1 | yes |

Layout view additionally has these direct children (at their
physical positions in file-offset order):

- `column_chunk_data_region` (one per `(rg, col)` pair)
- `offset_index`, `column_index`, `bloom_filter_header` (when present)
- `unknown` (any unreferenced byte ranges)

`_offset` = 0, `_length` = file size.

### `header_magic` (leaf)

The 4-byte `PAR1` at file start.

`_value`: `"PAR1"` (string). `_offset` = 0, `_length` = 4.

### `trailer_magic` (leaf)

The 4-byte `PAR1` at file end.

`_value`: `"PAR1"` (string). `_offset` = file size − 4, `_length` = 4.

### `footer_length` (leaf)

The 4-byte little-endian unsigned int immediately before
`trailer_magic`, encoding the size in bytes of the `footer` thrift.

`_value`: integer (the footer size in bytes).
`_offset` = file size − 8, `_length` = 4.

### `footer` (branch)

The parsed `FileMetaData` thrift. Located at `_offset` = file size −
8 − footer_size, `_length` = footer_size.

| Field | Type | Notes |
|---|---|---|
| `version` | int | parquet format version |
| `num_rows` | int | total rows across all row groups |
| `created_by` | string \| null | writer identification string |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `schema` | `schema` | exactly 1 (leaf) | yes |
| `kv_metadata` | `kv_metadata` | exactly 1 (leaf; may have empty `_value`) | yes |
| `row_groups` | `row_group` (metadata) | 0+ (array) | yes (RowGroup thrifts are inside the footer) |

### `schema` (leaf)

The footer's schema list, encoded on disk as a depth-first flat list
where each element has a `num_children` count indicating its sub-tree
size. v0 surfaces this as a single leaf node with `_value` carrying
the list — consumers needing the tree structure rebuild it from
`num_children`.

`_offset` / `_length` reference the entire schema-list byte range
within the footer.

`_value`: list of dicts, each shaped:

| Key | Type | Notes |
|---|---|---|
| `name` | string | |
| `repetition_type` | string \| null | `REQUIRED` / `OPTIONAL` / `REPEATED` (null for root) |
| `type` | string \| null | physical type; null for non-leaf elements |
| `converted_type` | string \| null | legacy logical-type hint |
| `logical_type` | object \| null | modern logical-type info |
| `num_children` | int | 0 for leaves; >0 for STRUCT-like elements |
| `field_id` | int \| null | optional field ID |
| `precision` | int \| null | for DECIMAL |
| `scale` | int \| null | for DECIMAL |
| `type_length` | int \| null | for FIXED_LEN_BYTE_ARRAY |

No children, no other content fields.

### `kv_metadata` (leaf)

The footer's key-value metadata list. Parquet permits duplicate
keys; the list shape preserves order and duplicates.

`_offset` / `_length` reference the entire kv_metadata-list byte
range within the footer. When the writer emitted no kv_metadata,
the kind is still present as a leaf with `_value: []` (the
multiplicity-1 contract on `footer.kv_metadata` is consistent;
the absence-vs-empty distinction is carried by `_length` and
`_value` being empty).

`_value`: list of dicts, each shaped:

| Key | Type | Notes |
|---|---|---|
| `key` | string | |
| `value` | string \| null | may be null if the writer recorded a key with no value |

No children, no other content fields.

### `row_group` (branch)

The RowGroup thrift in the footer. Per spec-honest framing: this
node represents the **metadata** describing a row group; the actual
on-disk data extent lives as `column_chunk_data_region` nodes
(layout view only — see below).

`_offset` / `_length` reference this RowGroup thrift's byte range
within the footer.

| Field | Type | Notes |
|---|---|---|
| `num_rows` | int | rows in this row group |
| `total_byte_size` | int | sum of uncompressed column data (parquet's `RowGroup.total_byte_size`) |
| `total_compressed_size` | int | sum of `column_chunk.compressed_size`; derived |
| `ordinal` | int \| null | row group index in file (parquet's `RowGroup.ordinal`) |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `columns` | `column_chunk` | 1+ (array; each is a ColumnChunk thrift in footer) | yes |

In **tree view**, consumers usually want to drill from `row_group`
to `columns[i].pages` etc. The Python API resolves transparently.

In **layout view**, the column chunks' data lives elsewhere — see
the `column_chunk` and `column_chunk_data_region` entries below.

### `column_chunk` (branch)

The ColumnChunk thrift in the footer (or, rarely, inline at
`file_offset` — see deprecation note below). Per spec-honest framing:
this node represents the **metadata** describing a column chunk;
the actual on-disk data (dict page + data pages) lives as a
`column_chunk_data_region` node (layout view) and is reached via
`data_region` in tree view.

`_offset` / `_length` reference this ColumnChunk thrift's byte
range within the footer.

| Field | Type | Notes |
|---|---|---|
| `path` | list of string | column path; flat columns have length 1 |
| `path_display` | string | dot-joined `path` for human display |
| `type` | string | physical type (`INT32`, `BYTE_ARRAY`, etc.) |
| `codec` | string | compression codec (`SNAPPY`, `UNCOMPRESSED`, ...) |
| `encodings` | list of string | encodings present in this chunk |
| `num_values` | int | including nulls |
| `compressed_size` | int | `total_compressed_size`; size of the on-disk data region |
| `uncompressed_size` | int | `total_uncompressed_size` |
| `data_page_offset` | int | start of first data page on disk |
| `dictionary_page_offset` | int \| null | start of dictionary page on disk if present |
| `file_offset` | int | **deprecated** per [PARQUET-2139](https://github.com/apache/parquet-format/pull/440); historically unreliable ("in many cases, the ColumnMetaData at this location is wrong"). Modern writers set this to 0. Surfaced verbatim from the thrift but do not dereference. |
| `statistics` | object \| null | parsed `ColumnMetaData.statistics` if present |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `dictionary_page` | `dictionary_page` | 0 or 1 | no |
| `pages` | `data_page_v1` \| `data_page_v2` | 0+ (array) | no |
| `data_region` | `column_chunk_data_region` | exactly 1 (in tree view, optional; see below) | no |
| `offset_index` | `offset_index` | 0 or 1 | no |
| `column_index` | `column_index` | 0 or 1 | no |
| `bloom_filter` | `bloom_filter_header` | 0 or 1 | no |

**Asymmetric handling of `data_region`:**

- In **tree view**, `data_region` is omitted as a child; instead
  `dictionary_page` and `pages` appear directly under `column_chunk`.
  The `data_region` indirection only earns its place where physical
  contiguity matters, which is layout view.
- In **layout view**, `column_chunk` carries `data_region_ref`
  (with `{_kind, _offset, _length}` of the `column_chunk_data_region`
  node), `offset_index_ref`, `column_index_ref`, `bloom_filter_ref`
  fields. The actual nodes appear at their physical positions as
  direct children of `file`.

### `column_chunk_data_region` (branch, layout view only)

A pseudo-parent grouping the physically-contiguous bytes that make
up one column chunk's on-disk data: dictionary page (if present)
followed by data pages. This concept doesn't exist in the parquet
spec — it's a convenience node for navigating physical layout.
(The existing HTML view's `:column_chunk_pages` grouping is exactly
this.)

`_offset` = `min(dictionary_page_offset, data_page_offset)`,
`_length` = `column_chunk.compressed_size` (`total_compressed_size`).

Tied back to its logical chunk via `chunk_ref` content field
(`{_kind: "column_chunk", _offset: <metadata-thrift-offset>, _length: <thrift-size>}`).

| Field | Type | Notes |
|---|---|---|
| `chunk_ref` | object | back-pointer to the corresponding `column_chunk` node in the footer |
| `row_group_index` | int | derived |
| `column_index_in_row_group` | int | derived |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `dictionary_page` | `dictionary_page` | 0 or 1 | yes |
| `pages` | `data_page_v1` \| `data_page_v2` | 0+ (array) | yes |

No `_value`.

**Note on inter-chunk contiguity:** within a row group, parquet
does not guarantee that one chunk's data region is contiguous with
the next. Writers may insert arbitrary bytes (and historically have:
see the pyarrow `file_offset` history). Those bytes appear as
`unknown` nodes between `column_chunk_data_region` siblings in
layout view.

### `dictionary_page` (branch)

A dictionary page header + body. `_offset` is the page-header start
(= `column_chunk.dictionary_page_offset`). `_length` covers header
thrift + body.

| Field | Type | Notes |
|---|---|---|
| `page_type` | string | always `"DICTIONARY_PAGE"` |
| `encoding` | string | `PLAIN` or `PLAIN_DICTIONARY` |
| `num_values` | int | number of entries in the dictionary |
| `uncompressed_size` | int | from page header |
| `compressed_size` | int | from page header |
| `is_compressed` | bool \| null | V2 only; null for V1 |
| `crc` | int \| null | optional |

Children: body decomposes into body-layer kinds (decoded dictionary
values). v0 leaves opaque; v1 (#21) populates.

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

Children: body decomposes into level blocks + values. v0 opaque;
v1 (#21) populates.

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

Children: body decomposes into rep_block + def_block + values. v0
opaque; v1 (#21) populates.

No `_value`.

### `offset_index` (opaque branch, v0)

Located at `column_chunk.offset_index_offset`, `_length` =
`column_chunk.offset_index_length`. Contains per-data-page byte
offsets and row indices.

v0: opaque — only `_kind`, `_offset`, `_length`. Per-page internal
structure deferred to v1.

### `column_index` (opaque branch, v0)

Located at `column_chunk.column_index_offset`, `_length` =
`column_chunk.column_index_length`. Contains per-data-page min/max
statistics and null counts.

v0: opaque. Per-page internal structure deferred to v1.

### `bloom_filter_header` (opaque branch, v0)

Located at `column_chunk.bloom_filter_offset`, `_length` =
`column_chunk.bloom_filter_length`. Contains the bloom filter
header thrift plus the bloom-filter bitset.

v0: opaque. Header + bitset decomposition deferred to v1.

### `unknown` (leaf, layout view only)

A range of file bytes not referenced by any logical structure.
Surfaces writer quirks (e.g., the deprecated pyarrow
end-of-chunk ColumnMetaData copy), padding, or intentional gaps.

Carries only the system fields by default — `_value` is omitted to
keep output light (unknown ranges can be arbitrarily large). A
future opt-in flag (`--include-unknown-bytes`) could populate
`_value` with base64-encoded bytes for forensics workflows; small
ranges (e.g. `_length < 256`) may inline as hex by default in a
later revision. For now, consumers needing the raw bytes can use
`dd if=file bs=1 skip=$OFFSET count=$LENGTH` (or `xxd -s $OFFSET
-l $LENGTH file`) directly.

`_kind` = `"unknown"`. No content fields, no `_value`.

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
this doc will be expanded with the catalog entry as part of that PR.

Also deferred:
- Internal structure of `offset_index`, `column_index`,
  `bloom_filter_header` (all opaque in v0).
- `--include-unknown-bytes` flag for `unknown` leaves.

## Worked examples

### Tree view, shallow (`tree --depth 2`)

```json
{
  "$schema": "parquet-analyzer/v2/tree",
  "_kind": "file",
  "_offset": 0,
  "_length": 40013,
  "path": "example.parquet",
  "header_magic": {"_kind": "header_magic", "_offset": 0, "_length": 4, "_value": "PAR1"},
  "row_groups": [
    {
      "_kind": "row_group",
      "_offset": 38990,
      "_length": 850,
      "num_rows": 891,
      "total_byte_size": 306419,
      "total_compressed_size": 38839,
      "ordinal": null,
      "columns": [
        {"_kind": "column_chunk", "_offset": 38990, "_length": 70, "_lazy": true},
        {"_kind": "column_chunk", "_offset": 39060, "_length": 70, "_lazy": true}
      ]
    }
  ],
  "footer": {"_kind": "footer", "_offset": 38843, "_length": 1162, "_lazy": true},
  "footer_length": {"_kind": "footer_length", "_offset": 39997, "_length": 4, "_value": 1162},
  "trailer_magic": {"_kind": "trailer_magic", "_offset": 40009, "_length": 4, "_value": "PAR1"}
}
```

Note `file` carries only `path` as a content field — logical aggregates
like `num_rows`, `created_by` live on `footer` (per the derived-field
policy above). Consumers navigate `.footer.num_rows` or `pf.tree.footer.num_rows`.

Note `row_group._offset`/`_length` describe the RowGroup thrift in
the footer, not the on-disk data extent.

### Tree view, materialized column chunk (`tree --path row_groups/0/columns/foo --depth 1`)

```json
{
  "$schema": "parquet-analyzer/v2/tree",
  "_kind": "column_chunk",
  "_offset": 39050,
  "_length": 120,
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

`column_chunk._offset`/`_length` describe the metadata thrift
(in footer); `compressed_size` / `data_page_offset` /
`dictionary_page_offset` describe where the actual data lives.

### Layout view, shallow (`layout --depth 1`)

```json
{
  "$schema": "parquet-analyzer/v2/layout",
  "_kind": "file",
  "_offset": 0,
  "_length": 40013,
  "path": "example.parquet",
  "children_in_offset_order": [
    {"_kind": "header_magic", "_offset": 0, "_length": 4, "_value": "PAR1"},
    {"_kind": "column_chunk_data_region", "_offset": 4, "_length": 4357, "_lazy": true,
     "chunk_ref": {"_kind": "column_chunk", "_offset": 38990, "_length": 70}},
    {"_kind": "column_chunk_data_region", "_offset": 4361, "_length": 1051, "_lazy": true,
     "chunk_ref": {"_kind": "column_chunk", "_offset": 39060, "_length": 70}},
    // ... more column_chunk_data_regions ...
    {"_kind": "unknown", "_offset": 38500, "_length": 343},
    {"_kind": "footer", "_offset": 38843, "_length": 1162, "_lazy": true},
    {"_kind": "footer_length", "_offset": 39997, "_length": 4, "_value": 1162},
    {"_kind": "trailer_magic", "_offset": 40009, "_length": 4, "_value": "PAR1"}
  ]
}
```

(The `unknown` node at offset 38500 above is hypothetical — would
appear if a writer left unreferenced bytes between data and
footer.)

### Layout view, materialized data region

```json
{
  "$schema": "parquet-analyzer/v2/layout",
  "_kind": "column_chunk_data_region",
  "_offset": 24256,
  "_length": 501,
  "chunk_ref": {"_kind": "column_chunk", "_offset": 39050, "_length": 120},
  "row_group_index": 0,
  "column_index_in_row_group": 5,
  "dictionary_page": {"_kind": "dictionary_page", "_offset": 24256, "_length": 20, "_lazy": true},
  "pages": [
    {"_kind": "data_page_v1", "_offset": 24276, "_length": 481, "_lazy": true}
  ]
}
```

### Leaf

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
