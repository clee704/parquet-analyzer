"""Synthetic parquet fixture generator for benchmarks.

Three "shapes" cover the lazy-parsing boundaries the lazy core (PR #18)
is about exposing — see RFC #3:

- ``wide``  — many columns, few rows (stresses per-column metadata
  walking; footer is large because it lists every column).
  **No-regression check** for footer-heavy schemas: lazy can only win
  modestly here (~2-3×) because the footer IS most of the eager cost.
- ``tall``  — few columns, many rows (large file body relative to
  footer — the canonical "footer is < 1% of file" case the lazy core
  targets). **Sized large enough to clear 100×** at default bench
  shape: ~300 MB file, ~2 KB footer, file/footer ratio ~150,000×.
- ``deep``  — few columns, many rows split across many row groups
  (stresses per-row-group / per-chunk metadata walking). Lazy
  eliminates ~1000 page-header parses; eager pays for each.
  ~50-100× ratio, bounded by footer-parse cost which grows with
  row-group count.
- ``indexed`` — multi-row-group AND OffsetIndex-present, with several
  data pages per chunk (small ``data_page_size``). Targets the page
  subcommands (#21): ``page list`` reads page extents from the
  OffsetIndex (one bounded parse per chunk, no per-page header reads),
  and a single ``page header`` / ``column show --walk-pages`` touches
  one chunk rather than the whole file — both avoid the full-file page
  walk the eager pipeline pays.

The shapes are sized for CI / dev-laptop runtime (generates in tens
of seconds, ~MB-GB on disk). Session-scoped fixtures (see
``conftest.py``) amortize the generation cost over all benchmarks in
a single run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

Shape = Literal["wide", "tall", "deep", "indexed"]


# Shape parameters tuned so generation completes in a few seconds while
# still producing files large enough that the eager-vs-lazy parse-time
# delta is unambiguous (file body >> footer for "tall" and "deep").
#
# Per-shape purpose (the ratios the lazy core targets are bounded by
# different things on each shape):
#
# - "wide" tests footer-heavy schemas (200 cols => ~50KB footer).
#   Footer parsing IS most of the eager cost here, so lazy can't win
#   much (~2-3x). This shape exists as a no-regression check, NOT as a
#   100x demonstration.
# - "tall" is the canonical "footer is < 1% of file" case (~2KB footer
#   vs ~300MB body). Lazy reads only the footer; eager scans the whole
#   file. Sized large enough that the ratio comfortably clears 100x.
# - "deep" tests the per-row-group / per-chunk walking that lazy
#   eliminates. 100 row groups means eager parses ~1000 page headers;
#   lazy parses a single (larger) footer. Ratio ~50-100x bounded by
#   footer-parse cost growing with row-group count.
SHAPES: dict[str, dict] = {
    "wide": {
        "num_columns": 200,
        "num_rows": 5_000,
        "row_group_size": 5_000,
    },
    "tall": {
        # ~300 MB file; eager ~300-500 ms; lazy ~1-2 ms => ~150-500x ratio.
        # Generation cost ~10-20 s (dominated by high-cardinality string
        # column construction); amortized via session-scoped fixture.
        "num_columns": 10,
        "num_rows": 5_000_000,
        "row_group_size": 5_000_000,
    },
    "deep": {
        "num_columns": 10,
        "num_rows": 500_000,
        "row_group_size": 5_000,  # -> 100 row groups
    },
    # OffsetIndex-present, multi-row-group, several pages per chunk. Smaller
    # than `deep` (so generation + the eager full-walk reference stay quick)
    # but with `write_page_index=True` and a small `data_page_size` so each
    # chunk holds many pages -- the case where `page list` reads the
    # OffsetIndex instead of walking page headers.
    "indexed": {
        "num_columns": 10,
        "num_rows": 200_000,
        "row_group_size": 50_000,  # -> 4 row groups
        "write_page_index": True,
        "data_page_size": 4 * 1024,  # small pages -> many pages per chunk
    },
}


def generate(path: Path, shape: Shape, *, seed: int = 42) -> Path:
    """Write a synthetic parquet file of the requested shape at ``path``.

    File contents are deterministic given a fixed ``seed`` -- the same
    shape + seed always produces a byte-identical file. This matters for
    bench reproducibility.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(SHAPES)}")
    cfg = SHAPES[shape]
    rng = np.random.default_rng(seed)

    # Mixed-type columns so PLAIN + dictionary + numeric paths are all
    # exercised. Cycle through this 4-entry pool as we add columns.
    type_pool: list = [
        ("int64", pa.int64()),
        ("float64", pa.float64()),
        ("string_low_card", pa.string()),
        ("string_high_card", pa.string()),
    ]

    columns: dict[str, pa.Array] = {}
    num_rows = cfg["num_rows"]
    for i in range(cfg["num_columns"]):
        kind, pa_type = type_pool[i % len(type_pool)]
        name = f"col_{i:03d}_{kind}"
        if kind == "int64":
            data = rng.integers(low=-(2**31), high=2**31, size=num_rows, dtype=np.int64)
            columns[name] = pa.array(data, type=pa_type)
        elif kind == "float64":
            data = rng.standard_normal(num_rows).astype(np.float64)
            columns[name] = pa.array(data, type=pa_type)
        elif kind == "string_low_card":
            choices = np.array([f"v{j}" for j in range(8)])
            idx = rng.integers(0, len(choices), size=num_rows)
            columns[name] = pa.array(choices[idx].tolist(), type=pa_type)
        else:  # string_high_card
            data = [f"id_{n:08d}_{rng.integers(0, 2**20):x}" for n in range(num_rows)]
            columns[name] = pa.array(data, type=pa_type)

    table = pa.table(columns)
    write_kwargs: dict = {
        "compression": "snappy",
        "row_group_size": cfg["row_group_size"],
        "use_dictionary": True,
    }
    if cfg.get("write_page_index"):
        write_kwargs["write_page_index"] = True
    if "data_page_size" in cfg:
        write_kwargs["data_page_size"] = cfg["data_page_size"]
    pq.write_table(table, path, **write_kwargs)
    return path
