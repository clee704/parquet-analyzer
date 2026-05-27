"""Synthetic parquet fixture generator for benchmarks.

Three "shapes" cover the lazy-parsing boundaries the Slice 2 work is
about exposing — see RFC #3:

- ``wide``  — many columns, few rows (stresses per-column metadata
  walking; footer is large because it lists every column)
- ``tall``  — few columns, many rows (large file body relative to footer
  — the canonical "footer is < 1% of file" case the lazy core targets)
- ``deep``  — few columns, many rows split across many row groups
  (stresses per-row-group / per-chunk metadata walking)

The shapes are intentionally sized for CI / dev-laptop runtime (~tens
of MB, generates in a few seconds). Even at this scale the eager vs
lazy delta is dramatic — eager reads the full file, lazy reads the
footer only, so the ratio is bounded by file_size / footer_size which
is already >= 100x for these shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

Shape = Literal["wide", "tall", "deep"]


# Shape parameters tuned so generation completes in a few seconds while
# still producing files large enough that the eager-vs-lazy parse-time
# delta is unambiguous (file body >> footer for "tall" and "deep").
SHAPES: dict[str, dict] = {
    "wide": {
        "num_columns": 200,
        "num_rows": 5_000,
        "row_group_size": 5_000,
    },
    "tall": {
        "num_columns": 10,
        "num_rows": 1_000_000,
        "row_group_size": 1_000_000,
    },
    "deep": {
        "num_columns": 10,
        "num_rows": 500_000,
        "row_group_size": 5_000,  # -> 100 row groups
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
    pq.write_table(
        table,
        path,
        compression="snappy",
        row_group_size=cfg["row_group_size"],
        use_dictionary=True,
    )
    return path
