"""Shared fixtures for benchmark tests.

Synthetic parquet files are generated once per session per shape and
cached in ``tmp_path_factory``'s session-scoped directory so the
generator cost (a few seconds) isn't paid per benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.bench import generate as _generate


def _make_fixture(shape: str):
    @pytest.fixture(scope="session")
    def _fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
        base = tmp_path_factory.mktemp(f"bench-{shape}")
        path = base / f"{shape}.parquet"
        return _generate.generate(path, shape=shape)

    _fixture.__name__ = f"{shape}_parquet"
    return _fixture


wide_parquet = _make_fixture("wide")
tall_parquet = _make_fixture("tall")
deep_parquet = _make_fixture("deep")
