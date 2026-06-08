# Tree schema: footer-layer kinds (v0)

This doc defines the **tree-node schema** for `parquet-analyzer`'s v3
output surface — the catalog of node kinds that represent parquet's
on-disk structure as navigable lazy trees, plus the universal rules
every node follows.

Companion to [`output-principles.md`](output-principles.md), which
defines the *contract* (footer-bounded and walk-free by default,
escape hatches, honesty pattern). This doc defines the *shapes* the
contract operates on.

This is **v0** — the footer-layer kinds. Body-layer kinds (sub-page
structure: def_block, values_block, indices, dict_lookup, encoding-
specific leaves) land in v1 alongside the body-decode work in #21.
Both versions are additive within v3. (The catalog version — v0 / v1 —
tracks which kinds exist; the `$schema` major — `parquet-analyzer/v3/...`
— tracks the universal node shape. The major moved v2 → v3 when every
node's `_offset` / `_length` were folded into a single `_location`
address object.)

## Universal node contract

Every tree node — leaf or branch — carries two system fields:

| Field | Type | Always present? | Meaning |
|---|---|---|---|
| `_kind` | string | yes | identifies the node's schema; consumer reads this and looks up the rest in this doc |
| `_location` | object | yes | where this node's bytes live in the file — see below |

`_location` is an object with plain (non-`_`-prefixed) inner keys:

| Key | Type | Meaning |
|---|---|---|
| `offset` | int | start byte of this node's bytes within the file |
| `length` | int | byte length of this node's bytes |

`_location` always describes **real file bytes** — the range you could
`dd`/`xxd` out of the file. (A later body-layer revision extends this
object with decompression fields for sub-nodes that live inside a
compressed region; v0's footer-layer nodes are all uncompressed on disk,
so they carry only `offset`/`length`.) The inner keys are plain because
the `_` prefix exists to separate framework fields from kind-specific
content *on a node*, and inside `_location` every key is framework-owned —
the same reason `_value` sub-dicts (below) use plain keys.

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
`kv_metadata` entry as its own node with its own `_location` for
its byte range in the footer) is a viable alternative shape if a
forensics use case ever needs it. v0 doesn't, so the simpler
single-leaf shape wins.

### Reserved namespaces

- `_*` prefix — system / annotation fields. Reserved for the
  framework. Today: `_kind`, `_location`, `_value`. Future:
  `_lazy`, `_error`, etc.
- `$schema` at output root — the response-shape URI (existing JSON
  Schema convention, carried over from PR #19). Format:
  `parquet-analyzer/v3/...`.

All other names are kind-specific content.

### Kind-defines-schema

`_kind` is the single source of truth for what fields a node has.
Consumers should not introspect by checking "does this node have a
`_value`?" — they should look up the kind in this catalog and know
the answer up front. Eliminates ambiguity (e.g., a leaf with
`_value: [...]` vs a branch with a child named `value`).

## Stubs and lazy markers

Tree nodes can appear in JSON output in two forms:

**Materialized** — the full content per the kind's schema:
```json
{"_kind": "row_group", "_location": {"offset": 4, "length": 306419},
 "num_rows": 891, "total_byte_size": 306419, "columns": [...]}
```

**Stub** — only the system fields, no content fields, no children:
```json
{"_kind": "row_group", "_location": {"offset": 4, "length": 306419}}
```

A stub is recognized by absence — the kind's schema declares content
and/or children that the stub doesn't carry. Stubs appear when
serialization stops at a depth limit (the consumer didn't ask for
this node's content), not when materializing would require I/O.

Most footer-derived nodes are NEVER `_lazy`: `ParquetFile(path)`
parses the entire footer thrift at construction, so `row_group`,
`column_chunk`, `schema`, `kv_metadata`, `footer` itself, and all
the `*_data_region` / `*_magic` / `footer_length` nodes are all
in-memory and free to access. They appear as stubs in JSON output
only because the consumer asked for a depth-limited view.

**Caveat — enumerating a `column_chunk`'s pages.** A `column_chunk`'s
own scalar fields are footer-derived, but emitting its `dictionary_page`
/ `pages` children — even as stubs, which still need each page's
`_location` (`offset`/`length`) — requires the per-page extents, which the footer
does not record. There are two cases:

- **With an `offset_index`** (written by most modern encoders), the page
  extents come from that one small thrift: listing a column's pages costs
  a single `offset_index` read and **no per-page header reads**,
  independent of the page count. The dictionary-page extent is derived
  from the column-metadata offsets (`[dictionary_page_offset,
  data_page_offset)` is exactly the dictionary page). This is the fast
  path (#30).
- **Without an `offset_index`**, the only source of page extents is the
  per-page header stream, so materializing a `column_chunk` (tree view)
  or a `column_chunk_data_region` (layout view) pays an O(pages)
  page-header walk to discover its page children.

Either way the page nodes carry `_lazy: true`. At the stub level a data
page uses the generic `data_page` kind (see below) — its `_location` is
known, but its version is not.

### `_lazy: true` — genuine I/O needed

A separate marker, `_lazy: true`, is reserved for nodes where
materializing actually triggers I/O or extra thrift parsing beyond
the footer parse:

- `dictionary_page`, `data_page` — a page reached without reading its
  header (its `_location` (`offset`/`length`) came from an `offset_index` or a
  header walk). `data_page` is the **generic, version-agnostic** stub
  kind: at the stub level a data page's `data_page_v1` / `data_page_v2`
  version is not yet known, because it lives in the page header the stub
  did not read.
- `data_page_v1`, `data_page_v2` — a **materialized** data page, whose
  version was read from its header.
- `offset_index`, `column_index`, `bloom_filter_header` — each requires
  reading + parsing an extra thrift from disk.

When these nodes appear as stubs in JSON output, they carry
`_lazy: true` to signal the materialization cost:

```json
{"_kind": "data_page", "_location": {"offset": 24276, "length": 481}, "_lazy": true}
```

A data page is therefore `data_page` when stubbed and `data_page_v1` /
`data_page_v2` when materialized — the version is a materialized-only
detail. The other lazy kinds keep the same kind string in both forms.

Without `_lazy: true`, a stub is just a depth-truncation indicator
(no I/O cost to materialize).

The Python API distinguishes these implicitly — attribute access on
any node triggers materialization if needed; the cost difference
between footer-derived and body-accessing nodes is invisible to the
caller (other than wall time).

### Depth semantics (uniform)

A depth limit truncates the tree at a **single, uniform level**, with
no per-kind exceptions:

- The root node (what the call is rooted at — `file` for whole-file
  output) is level 0. `row_group` / `footer` / `header_magic` /
  `footer_length` / `trailer_magic` are level 1; `column_chunk` and the
  footer's `schema` / `kv_metadata` are level 2; `dictionary_page` /
  `pages` / `offset_index` / `column_index` / `bloom_filter` are level 3.
- `depth=N` (N ≥ 1) materializes every node at levels 0 through N−1 and
  emits every node at level N as a stub.
- `depth=0` emits the root itself as a stub (only system fields, plus the
  root's `$schema`).
- `depth="all"` materializes the entire tree (no stubs); `_lazy` markers
  are dropped because the corresponding I/O is paid.

The rule is **uniform across siblings**: at a given depth, either all of
a node's children are materialized or all are stubbed — never a mix. In
particular `footer` and `row_groups` (both level-1 children of `file`)
are materialized together at `depth ≥ 2` and stubbed together at
`depth = 1`; the worked examples below follow this rule.

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
two kinds, `column_chunk_data_region` and `unknown`, exist only in
layout view; see "Per-kind: 'logical children' + 'physically
contained?'" below). The views differ in **how children are
arranged**:

| | Tree view | Layout view |
|---|---|---|
| Surfaces what | Logical hierarchy | Physical byte ordering |
| Containment rule | A node's children are everything it's logically associated with | A node's children must be physically contained (`child._location.offset` within `parent._location.offset..parent._location.offset+parent._location.length`) |
| Non-contained logical children | Appear inline as full child nodes | Replaced by `<name>_ref` content fields carrying `{_kind, _location}`; the actual node lives at its physical position in the tree |
| Unreferenced bytes | Don't appear (nothing logically points to them) | Appear as `unknown` leaf nodes |
| `$schema` URI | `parquet-analyzer/v3/tree` | `parquet-analyzer/v3/layout` |
| Verbs | `tree`, plus the existing curated verb-noun verbs (`file summary`, `column show`, etc.) — see PR #19 | `layout` (replaces legacy `--output-mode segments`) |

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
  the v3 contract for that kind. Changes bump the major (v3 → v4).
- New optional fields on an existing kind are NOT breaking
  (additive within the same major version).
- The `$schema` URI on outputs carries the major version:
  `parquet-analyzer/v3/...`.

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

Verbs (the existing curated verb-noun surface — `file summary`,
`column show`, etc.) compose the curated output they need from
across the tree. The tree shouldn't pre-compose; the verbs do that
work.

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
| `row_groups` | `row_group` | 0+ (array) | yes (each `row_group`'s `_location` (`offset`/`length`) cover its metadata-thrift bytes in footer; physically inside `footer`) |
| `footer` | `footer` | exactly 1 | yes |
| `footer_length` | `footer_length` | exactly 1 | yes |
| `trailer_magic` | `trailer_magic` | exactly 1 | yes |

**View-specific shape of `file.children`:**

- In **tree view**, `file` exposes the named-child fields above
  (`header_magic`, `row_groups`, `footer`, `footer_length`,
  `trailer_magic`) — order is irrelevant; consumers navigate by
  name.
- In **layout view**, the named-child fields collapse into a single
  `children` array ordered by `_location.offset` ascending. This array also
  includes the nodes that aren't tree-view children of `file` but
  ARE physical children of it: `column_chunk_data_region` (one per
  `(rg, col)`), `offset_index` / `column_index` /
  `bloom_filter_header` (when present), and `unknown` (any
  unreferenced byte ranges).

`_location` is `{offset: 0, length: <file size>}`.

**`row_groups` appears under both `file` and `footer` (tree view).**
In tree view, the `row_groups[]` array is a named child of `file`
(above) as a navigation shortcut, AND a named child of `footer` (where
the RowGroup thrifts physically live). The same `row_group` nodes are
reachable at both `file.row_groups` and `file.footer.row_groups`, with
identical `_location` (`offset`/`length`). This duplication is intentional in v0 —
the `file.row_groups` shortcut is the common access path and saves
consumers a hop through `footer` — but it means a consumer walking the
whole tree sees row groups twice and must dedupe. A future version may
replace the `file` shortcut with a `$ref` to `footer.row_groups`; that
tradeoff is tracked in #26. (Layout view has no such duplication: row
groups appear only inside `footer`.)

### `header_magic` (leaf)

The 4-byte `PAR1` at file start.

`_value`: `"PAR1"` (string). `_location` is `{offset: 0, length: 4}`.

### `trailer_magic` (leaf)

The 4-byte `PAR1` at file end.

`_value`: `"PAR1"` (string). `_location` is `{offset: <file size − 4>, length: 4}`.

### `footer_length` (leaf)

The 4-byte little-endian unsigned int immediately before
`trailer_magic`, encoding the size in bytes of the `footer` thrift.

`_value`: integer (the footer size in bytes).
`_location` is `{offset: <file size − 8>, length: 4}`.

### `footer` (branch)

The parsed `FileMetaData` thrift. Located at `_location` `{offset: <file size − 8 − footer_size>, length:
footer_size}`.

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

`_location` (`offset`/`length`) reference the entire schema-list byte range
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

`_location` (`offset`/`length`) reference the entire kv_metadata-list byte
range within the footer. When the writer emitted no kv_metadata,
the kind is still present as a leaf with `_value: []` (the
multiplicity-1 contract on `footer.kv_metadata` is consistent;
the absence-vs-empty distinction is carried by `_location.length` and
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

`_location` (`offset`/`length`) reference this RowGroup thrift's byte range
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

`_location` (`offset`/`length`) reference this ColumnChunk thrift's byte
range within the footer.

| Field | Type | Notes |
|---|---|---|
| `path` | list of strings | column path; flat columns have length 1 |
| `path_display` | string | dot-joined `path` for human display |
| `type` | string | physical type (`INT32`, `BYTE_ARRAY`, etc.) |
| `codec` | string | compression codec (`SNAPPY`, `UNCOMPRESSED`, ...) |
| `encodings` | list of string | encodings present in this chunk |
| `num_values` | int | including nulls |
| `compressed_size` | int | `total_compressed_size`; size of the on-disk data region |
| `uncompressed_size` | int | `total_uncompressed_size` |
| `data_page_offset` | int | start of first data page on disk |
| `dictionary_page_offset` | int \| null | the raw `ColumnMetaData.dictionary_page_offset` thrift value — start of the dictionary page on disk, or `null` when the writer omitted the field. **Note:** some older writers leave this `null` while still writing a dictionary page (reached via `data_page_offset`); in that case the `dictionary_page` child is populated even though this scalar is `null`. The field is surfaced verbatim (a cheap footer value); the `dictionary_page` child is authoritative for the on-disk location. See #32. |
| `file_offset` | int | **deprecated** per [PARQUET-2139](https://github.com/apache/parquet-format/pull/440); historically unreliable ("in many cases, the ColumnMetaData at this location is wrong"). Modern writers set this to 0. Surfaced verbatim from the thrift but do not dereference. |
| `statistics` | object \| null | parsed `ColumnMetaData.statistics` if present |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `dictionary_page` | `dictionary_page` | 0 or 1 | no |
| `pages` | `data_page` (stub) / `data_page_v1` \| `data_page_v2` (materialized) | 0+ (array) | no |
| `data_region` | `column_chunk_data_region` | view-specific (see below) | no |
| `offset_index` | `offset_index` | 0 or 1 | no |
| `column_index` | `column_index` | 0 or 1 | no |
| `bloom_filter` | `bloom_filter_header` | 0 or 1 | no |

**Asymmetric handling of `data_region`:**

- In **tree view**, `data_region` is omitted as a child; instead
  `dictionary_page` and `pages` appear directly under `column_chunk`.
  The `data_region` indirection only earns its place where physical
  contiguity matters, which is layout view.
- In **layout view**, `column_chunk` carries `data_region_ref`
  (with `{_kind, _location}` of the `column_chunk_data_region`
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

`_location.offset` = `min(dictionary_page_offset, data_page_offset)`,
`_location.length` = `column_chunk.compressed_size` (`total_compressed_size`).

Tied back to its logical chunk via `chunk_ref` content field
(`{_kind: "column_chunk", _location: {offset: <metadata-thrift-offset>, length: <thrift-size>}}`).

| Field | Type | Notes |
|---|---|---|
| `chunk_ref` | object | back-pointer to the corresponding `column_chunk` node in the footer |
| `row_group_index` | int | derived |
| `column_position_in_row_group` | int | derived; 0-based positional index of this chunk within its row group (named to avoid collision with the `column_index` kind, which is parquet's per-page-stats thrift) |

Logical children:

| Name | Kind | Multiplicity | Physically contained? |
|---|---|---|---|
| `dictionary_page` | `dictionary_page` | 0 or 1 | yes |
| `pages` | `data_page` (stub) / `data_page_v1` \| `data_page_v2` (materialized) | 0+ (array) | yes |

No `_value`.

**Note on inter-chunk contiguity:** within a row group, parquet
does not guarantee that one chunk's data region is contiguous with
the next. Writers may insert arbitrary bytes (and historically have:
see the pyarrow `file_offset` history). Those bytes appear as
`unknown` nodes between `column_chunk_data_region` siblings in
layout view.

### `dictionary_page` (branch)

A dictionary page header + body. `_location.offset` is the page-header start
(= `column_chunk.dictionary_page_offset`). `_location.length` covers header
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

A V1 data page. `_location` covers header + body.

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

A V2 data page. `_location` covers header + body.

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

Located at `column_chunk.offset_index_offset`; `_location.length` =
`column_chunk.offset_index_length`. Contains per-data-page byte
offsets and row indices.

v0: opaque — only `_kind`, `_location`. Per-page internal
structure deferred to v1.

### `column_index` (opaque branch, v0)

Located at `column_chunk.column_index_offset`; `_location.length` =
`column_chunk.column_index_length`. Contains per-data-page min/max
statistics and null counts.

v0: opaque. Per-page internal structure deferred to v1.

### `bloom_filter_header` (opaque branch, v0)

Located at `column_chunk.bloom_filter_offset`; `_location.length` =
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
ranges (e.g. `_location.length < 256`) may inline as hex by default in a
later revision. For now, consumers needing the raw bytes can use
`dd if=file bs=1 skip=$OFFSET count=$LENGTH` (or `xxd -s $OFFSET
-l $LENGTH file`) directly.

`_kind` = `"unknown"`. No content fields, no `_value`.

## Out of scope for v0 (landing in v1 via #21)

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

At `depth 2`, level-0 (`file`) and all of level-1 are materialized;
level-2 nodes (`column_chunk`, and the footer's `schema` /
`kv_metadata` / `row_groups`) are stubs. Note `footer` is materialized
here — uniformly with its level-1 sibling `row_groups`, per the depth
rule above — and that `row_groups` appears twice: materialized under
`file` and stubbed under `footer` (the documented v0 duplication).

```json
{
  "$schema": "parquet-analyzer/v3/tree",
  "_kind": "file",
  "_location": {"offset": 0, "length": 40013},
  "path": "example.parquet",
  "header_magic": {"_kind": "header_magic", "_location": {"offset": 0, "length": 4}, "_value": "PAR1"},
  "row_groups": [
    {
      "_kind": "row_group",
      "_location": {"offset": 38990, "length": 850},
      "num_rows": 891,
      "total_byte_size": 306419,
      "total_compressed_size": 38839,
      "ordinal": null,
      "columns": [
        {"_kind": "column_chunk", "_location": {"offset": 38990, "length": 70}},
        {"_kind": "column_chunk", "_location": {"offset": 39060, "length": 70}}
      ]
    }
  ],
  "footer": {
    "_kind": "footer",
    "_location": {"offset": 38843, "length": 1162},
    "version": 2,
    "num_rows": 891,
    "created_by": "parquet-cpp-arrow version 14.0.0",
    "schema": {"_kind": "schema", "_location": {"offset": 38846, "length": 193}},
    "kv_metadata": {"_kind": "kv_metadata", "_location": {"offset": 39960, "length": 45}},
    "row_groups": [
      {"_kind": "row_group", "_location": {"offset": 38990, "length": 850}}
    ]
  },
  "footer_length": {"_kind": "footer_length", "_location": {"offset": 40005, "length": 4}, "_value": 1162},
  "trailer_magic": {"_kind": "trailer_magic", "_location": {"offset": 40009, "length": 4}, "_value": "PAR1"}
}
```

Note `file` carries only `path` as a content field — logical aggregates
like `num_rows`, `created_by` live on `footer` (per the derived-field
policy above). Consumers navigate `.footer.num_rows` or `pf.tree.footer.num_rows`.

Note `row_group._location` describes the RowGroup thrift in
the footer, not the on-disk data extent.

### Tree view, materialized column chunk (`tree --path row_groups/0/columns/foo --depth 1`)

```json
{
  "$schema": "parquet-analyzer/v3/tree",
  "_kind": "column_chunk",
  "_location": {"offset": 39050, "length": 120},
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
  "dictionary_page": {"_kind": "dictionary_page", "_location": {"offset": 24256, "length": 20}, "_lazy": true},
  "pages": [
    {"_kind": "data_page", "_location": {"offset": 24276, "length": 481}, "_lazy": true}
  ],
  "offset_index": null,
  "column_index": null,
  "bloom_filter": null
}
```

`column_chunk._location` describes the metadata thrift
(in footer); `compressed_size` / `data_page_offset` /
`dictionary_page_offset` describe where the actual data lives.

### Layout view, shallow (`layout --depth 1`)

At `depth 1`, `file` is materialized and its physical children are
stubs — a byte map (each child's `_kind`/`_location`) without
materializing content. Go to `depth 2` to materialize each child (a
`column_chunk_data_region`'s `chunk_ref` / `row_group_index` / page
stubs, the footer's fields, etc.).

```json
{
  "$schema": "parquet-analyzer/v3/layout",
  "_kind": "file",
  "_location": {"offset": 0, "length": 40013},
  "path": "example.parquet",
  "children": [
    {"_kind": "header_magic", "_location": {"offset": 0, "length": 4}},
    {"_kind": "column_chunk_data_region", "_location": {"offset": 4, "length": 4357}},
    {"_kind": "column_chunk_data_region", "_location": {"offset": 4361, "length": 1051}},
    {"_kind": "unknown", "_location": {"offset": 38500, "length": 343}},
    {"_kind": "footer", "_location": {"offset": 38843, "length": 1162}},
    {"_kind": "footer_length", "_location": {"offset": 40005, "length": 4}},
    {"_kind": "trailer_magic", "_location": {"offset": 40009, "length": 4}}
  ]
}
```

(The example elides the `column_chunk_data_region`s between offset 5412
and the `unknown` node for brevity. The `unknown` node at offset 38500
is hypothetical — it would appear only if a writer left unreferenced
bytes between the data and the footer.)

### Layout view, materialized data region

```json
{
  "$schema": "parquet-analyzer/v3/layout",
  "_kind": "column_chunk_data_region",
  "_location": {"offset": 24256, "length": 501},
  "chunk_ref": {"_kind": "column_chunk", "_location": {"offset": 39050, "length": 120}},
  "row_group_index": 0,
  "column_position_in_row_group": 5,
  "dictionary_page": {"_kind": "dictionary_page", "_location": {"offset": 24256, "length": 20}, "_lazy": true},
  "pages": [
    {"_kind": "data_page", "_location": {"offset": 24276, "length": 481}, "_lazy": true}
  ]
}
```

### Leaf

```json
{"_kind": "footer_length", "_location": {"offset": 40005, "length": 4}, "_value": 1162}
```

## How this doc grows

- New kinds added by appending to the catalog with a one-line
  rationale.
- Field additions to existing kinds (within the same major version)
  documented inline; old consumers ignore the new field.
- Breaking schema changes bump the major (e.g. v3 → v4) and require this
  doc to be reissued at the new version.
- The body-layer kinds (v1) get added by #21; each kind's
  introduction in code carries a docstring, and that docstring
  populates this doc's catalog entry.
